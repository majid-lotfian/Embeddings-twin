import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import umap

# ================ Setup ================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'Iraq', 'data_processed.xlsx')  # make sure it's CSV
embedding_output_dir = os.path.join(BASE_DIR, 'Iraq')
os.makedirs(embedding_output_dir, exist_ok=True)

# ================ Load Dataset ================

data = pd.read_excel(dataset_path, engine='openpyxl')
data.columns = data.columns.str.strip().str.lower()

# Store and drop ID column
row_indices = data['id'].values
data = data.drop(columns=['id'])

# Encode 'sex' if present
if 'sex' in data.columns:
    data['sex'] = LabelEncoder().fit_transform(data['sex'].astype(str))

# Drop any non-numeric columns if they exist
data = data.select_dtypes(include=[np.number])

# Standardize
X = data.values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ================ Define VAE ================

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(reconstructed_x, x, mu, logvar, beta=10.0):
    recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# ================ Run Reductions for 8, 12, 16, 20 ================

reduction_dims = [8, 12, 16, 20]

for dim in reduction_dims:
    print(f"\n🔵 Generating {dim}D embeddings...")

   #if dim >= X.shape[1]:
    #    raise ValueError(f"⚠️ Dimension {dim} must be less than input feature count ({X.shape[1]})")

    # --- PCA ---
    print(f"▶ PCA ({dim}D)...")
    pca = PCA(n_components=dim, random_state=42)
    pca_embeddings = pca.fit_transform(X)
    print(f"    ✅ Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    np.save(os.path.join(embedding_output_dir, f'pca_embeddings_{dim}d.npy'), pca_embeddings)
    np.save(os.path.join(embedding_output_dir, f'pca_row_indices_{dim}d.npy'), row_indices)

    # --- UMAP ---
    print(f"▶ UMAP ({dim}D)...")
    umap_model = umap.UMAP(n_components=dim, random_state=42)
    umap_embeddings = umap_model.fit_transform(X)
    np.save(os.path.join(embedding_output_dir, f'umap_embeddings_{dim}d.npy'), umap_embeddings)
    np.save(os.path.join(embedding_output_dir, f'umap_row_indices_{dim}d.npy'), row_indices)

    # --- VAE ---
    print(f"▶ VAE ({dim}D)...")
    vae = VariationalAutoencoder(input_dim=X.shape[1], latent_dim=dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    log_path = os.path.join(embedding_output_dir, f'vae_loss_log_{dim}d.txt')

    with open(log_path, 'w') as log_file:
        vae.train()
        for epoch in range(50):
            total_loss = 0
            for batch in loader:
                x_batch = batch[0]
                recon, mu, logvar = vae(x_batch)
                loss, recon_loss, kl_loss = vae_loss(recon, x_batch, mu, logvar, beta=10.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                log_file.write(f"Epoch {epoch+1} - Recon: {recon_loss.item():.6f} | KL: {kl_loss.item():.6f} | Total: {loss.item():.6f}\n")

            if (epoch + 1) % 10 == 0:
                print(f"    🔁 VAE Epoch {epoch+1} - Avg Loss: {total_loss/len(loader):.6f}")

    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X_tensor)
        z = vae.reparameterize(mu, logvar)
        vae_embeddings = z.cpu().numpy()

    np.save(os.path.join(embedding_output_dir, f'vae_embeddings_{dim}d.npy'), vae_embeddings)
    np.save(os.path.join(embedding_output_dir, f'vae_row_indices_{dim}d.npy'), row_indices)

print("\n✅ All embeddings for Iraq dataset generated successfully.")
