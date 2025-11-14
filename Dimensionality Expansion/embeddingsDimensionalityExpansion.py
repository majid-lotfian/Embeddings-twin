import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ================
# Setup
# ================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
embedding_output_dir = os.path.join(BASE_DIR, 'embeddings_classical_expansion', 'CBC')
os.makedirs(embedding_output_dir, exist_ok=True)

# ================
# Load Dataset
# ================

data = pd.read_csv(dataset_path)
data.columns = data.columns.str.strip()

if 'ID' not in data.columns:
    data['ID'] = data.index + 2

num_cols = ['white_blood_cells', 'neutrophil_count', 'lymphocyte_count', 'monocyte_count', 'eosinophil_count', 'basophil_count',
            'red_blood_cells', 'hemoglobin', 'hematocrit', 'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin',
            'mean_corpuscular_hemoglobin_concentration', 'red_cell_distribution_width', 'platelets', 'mean_platelet_volume',
            'plateletcrit', 'platelet_distribution_width', 'SD', 'SDTSD', 'TSD', 'ferritin', 'folate', 'vitamin_b12']
bin_cols = ['gender']

feature_cols = num_cols + bin_cols
X = data[feature_cols].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

row_indices = data['ID'].values
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ================
# Define Variational Autoencoder (VAE)
# ================

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

# ================
# Define VAE Loss Function
# ================

def vae_loss(reconstructed_x, x, mu, logvar, beta=4.0):
    recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# ================
# Expansion Embedding Sizes
# ================

expansion_dims = [32, 64, 96, 128]

# ================
# Apply Each Method
# ================

for dim in expansion_dims:
    print(f"\n🔵 Generating {dim}D embeddings...")

    # --- Kernel PCA ---
    # print(f"Starting Kernel PCA ({dim}D)...")
    # kpca = KernelPCA(n_components=dim, kernel='rbf', random_state=42)
    # kpca_embeddings = kpca.fit_transform(X)
    # np.save(os.path.join(embedding_output_dir, f'kernelpca_embeddings_{dim}d.npy'), kpca_embeddings)
    # np.save(os.path.join(embedding_output_dir, f'kernelpca_row_indices_{dim}d.npy'), row_indices)

    # --- Random Projection ---
    print(f"Starting Random Projection ({dim}D)...")
    rand_proj = GaussianRandomProjection(n_components=dim, random_state=42)
    randproj_embeddings = rand_proj.fit_transform(X)
    np.save(os.path.join(embedding_output_dir, f'randomproj_embeddings_{dim}d.npy'), randproj_embeddings)
    np.save(os.path.join(embedding_output_dir, f'randomproj_row_indices_{dim}d.npy'), row_indices)

    # --- Polynomial Expansion ---
    print(f"Starting Polynomial Expansion ({dim}D)...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X)
    if poly_features.shape[1] < dim:
        raise ValueError(f"Polynomial expansion degree too small for {dim}D. Got only {poly_features.shape[1]} features.")
    poly_embeddings = poly_features[:, :dim]
    np.save(os.path.join(embedding_output_dir, f'polyexpand_embeddings_{dim}d.npy'), poly_embeddings)
    np.save(os.path.join(embedding_output_dir, f'polyexpand_row_indices_{dim}d.npy'), row_indices)

    # --- VAE (Overcomplete, β-VAE) ---
    print(f"Starting VAE ({dim}D)...")
    vae = VariationalAutoencoder(input_dim=X.shape[1], latent_dim=dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    beta = 10.0

    vae_log_path = os.path.join(embedding_output_dir, f'vae_loss_log_{dim}d.txt')
    with open(vae_log_path, 'w') as log_file:
        vae.train()
        for epoch in range(50):
            total_loss = 0
            for batch in loader:
                x_batch = batch[0]
                reconstructed, mu, logvar = vae(x_batch)
                loss, recon_loss, kl_loss = vae_loss(reconstructed, x_batch, mu, logvar, beta=beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                log_file.write(f"Epoch {epoch+1} - Recon: {recon_loss.item():.6f} | KL: {kl_loss.item():.6f} | Total: {loss.item():.6f}\n")

            if (epoch + 1) % 10 == 0:
                print(f"VAE {dim}D - Epoch {epoch+1} - Avg Total Loss: {total_loss / len(loader):.6f}")

    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X_tensor)
        embeddings = vae.reparameterize(mu, logvar)
        embeddings = embeddings.cpu().numpy()

    np.save(os.path.join(embedding_output_dir, f'vae_embeddings_{dim}d.npy'), embeddings)
    np.save(os.path.join(embedding_output_dir, f'vae_row_indices_{dim}d.npy'), row_indices)

print("\n🎯 Dimensionality Expansion Embeddings generated successfully!")
