import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ================
# Setup
# ================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
embedding_output_dir = os.path.join(BASE_DIR, 'embeddings_classical_reduction', 'CBC', 'v1')
os.makedirs(embedding_output_dir, exist_ok=True)

# ================
# Load Dataset
# ================

data = pd.read_csv(dataset_path)
data.columns = data.columns.str.strip()  # Clean column names

if 'ID' not in data.columns:
    data['ID'] = data.index + 2

num_cols = ['white_blood_cells', 'neutrophil_count', 'lymphocyte_count', 'monocyte_count', 'eosinophil_count', 'basophil_count', 'red_blood_cells', 'hemoglobin', 'hematocrit',
            'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin', 'mean_corpuscular_hemoglobin_concentration', 'red_cell_distribution_width', 'platelets',
            'mean_platelet_volume', 'plateletcrit', 'platelet_distribution_width', 'SD', 'SDTSD', 'TSD', 'ferritin', 'folate', 'vitamin_b12']
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
# Define VAE Class
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
            nn.Linear(128, input_dim),
            nn.ReLU()
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
# Define Loss Function
# ================

def vae_loss(reconstructed_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss

# ================
# Reduction Embedding Sizes
# ================

reduction_dims = [16, 20]

# ================
# Apply Each Method
# ================

for dim in reduction_dims:
    print(f"\n🔵 Generating {dim}D embeddings...")

    if dim >= X.shape[1]:
        raise ValueError(f"⚠️ Reduction dimension {dim} must be less than feature count {X.shape[1]}.")


    # --- PCA ---
    print(f"Starting PCA for {dim}D")
    pca = PCA(n_components=dim, random_state=42)
    pca_embeddings = pca.fit_transform(X)
    np.save(os.path.join(embedding_output_dir, f'pca_embeddings_{dim}d.npy'), pca_embeddings)
    np.save(os.path.join(embedding_output_dir, f'pca_row_indices_{dim}d.npy'), row_indices)


    # --- t-SNE ---
    # print(f"Starting t-SNE ({dim}D)...")
    # tsne_method = 'barnes_hut' if dim <= 3 else 'exact'
    # tsne = TSNE(n_components=dim, method=tsne_method, random_state=42, perplexity=30, n_iter=1000)
    # tsne_embeddings = tsne.fit_transform(X)
    # np.save(os.path.join(embedding_output_dir, f'tsne_embeddings_{dim}d.npy'), tsne_embeddings)
    # np.save(os.path.join(embedding_output_dir, f'tsne_row_indices_{dim}d.npy'), row_indices)

    # --- UMAP ---
    print(f"Starting UMAP ({dim}D)...")
    umap_model = umap.UMAP(n_components=dim, random_state=42)
    umap_embeddings = umap_model.fit_transform(X)
    np.save(os.path.join(embedding_output_dir, f'umap_embeddings_{dim}d.npy'), umap_embeddings)
    np.save(os.path.join(embedding_output_dir, f'umap_row_indices_{dim}d.npy'), row_indices)

    # --- VAE ---
    print(f"Starting VAE ({dim}D)...")
    vae = VariationalAutoencoder(input_dim=X.shape[1], latent_dim=dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    vae.train()
    for epoch in range(50):
        total_loss = 0
        for batch in loader:
            x_batch = batch[0]
            reconstructed, mu, logvar = vae(x_batch)
            loss = vae_loss(reconstructed, x_batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"VAE {dim}D - Epoch {epoch+1} - Loss: {total_loss/len(loader):.6f}")

    # Save VAE embeddings after training
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X_tensor)
        embeddings = vae.reparameterize(mu, logvar)
        embeddings = embeddings.cpu().numpy()

    np.save(os.path.join(embedding_output_dir, f'vae_embeddings_{dim}d.npy'), embeddings)
    np.save(os.path.join(embedding_output_dir, f'vae_row_indices_{dim}d.npy'), row_indices)

print("\n🎯 Dimensionality Reduction Embeddings generated successfully!")
