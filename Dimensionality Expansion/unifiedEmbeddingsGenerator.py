import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# ============================
# Setup
# ============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_configs = {
    'CBC': {
        'path': os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv'),
        'format': 'csv',
        'dims': [32, 64, 96, 128],
        'has_id': False,
        'encode_cols': ['gender'],
        'drop_cols': []
    },
    'COVID19': {
        'path': os.path.join(BASE_DIR, '..', '..', 'datasets', 'Covid19', 'data_processed.csv'),
        'format': 'csv',
        'dims': [32, 64, 96, 128],
        'has_id': True,
        'encode_cols': ['sex'],
        'drop_cols': ['pcr_date', 'exam_blood_test_date']
    },
    'Iraq': {
        'path': os.path.join(BASE_DIR, '..', '..', 'datasets', 'Iraq', 'data_processed.xlsx'),
        'format': 'xlsx',
        'dims': [32, 64, 96, 128],
        'has_id': True,
        'encode_cols': ['sex'],
        'drop_cols': []
    },
    'Liverpool': {
        'path': os.path.join(BASE_DIR, '..', '..', 'datasets', 'Liverpool', 'data_processed.csv'),
        'format': 'csv',
        #'dims': [12, 16, 20, 32, 64, 96, 128],
        'dims': [96, 128],
        'has_id': True,
        'encode_cols': [],
        'drop_cols': []
    }
}




# ============================
# Define VAE
# ============================

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

# ============================
# Run All Datasets
# ============================

for name, cfg in dataset_configs.items():
    print(f"\n📁 Processing {name} dataset")

    if cfg['format'] == 'csv':
        data = pd.read_csv(cfg['path'])
    elif cfg['format'] == 'xlsx':
        data = pd.read_excel(cfg['path'], engine='openpyxl')
    else:
        raise ValueError("Unsupported file format")

    data.columns = data.columns.str.strip().str.lower()

    if cfg['has_id']:
        row_indices = data['id'].values
        data = data.drop(columns=['id'])
    else:
        row_indices = np.arange(2, len(data) + 2)

    for col in cfg['drop_cols']:
        if col in data.columns:
            data = data.drop(columns=[col])

    for col in cfg['encode_cols']:
        if col in data.columns:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    data = data.select_dtypes(include=[np.number])
    X = data.values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    output_dir = os.path.join(BASE_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    for dim in cfg['dims']:
        print(f"\n🔷 {name} - Generating {dim}D embeddings")

        print(f"    ➤ Random Projection ({dim}D)...")
        rand_proj = GaussianRandomProjection(n_components=dim, random_state=42)
        randproj_embeddings = rand_proj.fit_transform(X)
        np.save(os.path.join(output_dir, f'randomproj_embeddings_{dim}d.npy'), randproj_embeddings)
        np.save(os.path.join(output_dir, f'randomproj_row_indices_{dim}d.npy'), row_indices)

        print(f"    ➤ Polynomial Expansion ({dim}D)...")
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X)
        if poly_features.shape[1] < dim:
            print(f"    ⚠️ Skipping Polynomial Expansion: only {poly_features.shape[1]} features generated, need {dim}.")
        else:
            poly_embeddings = poly_features[:, :dim]
            np.save(os.path.join(output_dir, f'polyexpand_embeddings_{dim}d.npy'), poly_embeddings)
            np.save(os.path.join(output_dir, f'polyexpand_row_indices_{dim}d.npy'), row_indices)

        print(f"    ➤ VAE ({dim}D)...")
        vae = VariationalAutoencoder(input_dim=X.shape[1], latent_dim=dim).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        log_path = os.path.join(output_dir, f'vae_loss_log_{dim}d.txt')

        with open(log_path, 'w') as log_file:
            vae.train()
            for epoch in range(50):
                total_loss = 0
                for batch in loader:
                    x_batch = batch[0]
                    recon, mu, logvar = vae(x_batch)
                    loss, recon_loss, kl_loss = vae_loss(recon, x_batch, mu, logvar)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    log_file.write(f"Epoch {epoch+1} - Recon: {recon_loss.item():.6f} | KL: {kl_loss.item():.6f} | Total: {loss.item():.6f}\n")

                if (epoch + 1) % 10 == 0:
                    print(f"        🔁 VAE {dim}D Epoch {epoch+1} - Avg Loss: {total_loss/len(loader):.6f}")

        vae.eval()
        with torch.no_grad():
            mu, logvar = vae.encode(X_tensor)
            z = vae.reparameterize(mu, logvar)
            vae_embeddings = z.cpu().numpy()

        np.save(os.path.join(output_dir, f'vae_embeddings_{dim}d.npy'), vae_embeddings)
        np.save(os.path.join(output_dir, f'vae_row_indices_{dim}d.npy'), row_indices)

print("\n✅ All dimensionality expansion completed successfully.")
