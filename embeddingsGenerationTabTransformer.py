import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
checkpoint_root_dir = os.path.join(BASE_DIR, 'checkpoints_tabtransformer')
embedding_output_dir = os.path.join(BASE_DIR, 'embeddings_tabtransformer')

os.makedirs(checkpoint_root_dir, exist_ok=True)
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================

data = pd.read_csv(dataset_path)

if 'ID' not in data.columns:
    data['ID'] = data.index + 2

cat_cols = []  # If you had categorical columns, define them
num_cols = ['Age', 'Hemoglobin', 'Platelet_Count', 'White_Blood_Cells', 'Red_Blood_Cells',
            'Mean_Corpuscular_Volume', 'Mean_Corpuscular_Hemoglobin', 'Mean_Corpuscular_Hemoglobin_Concentration']
bin_cols = ['Gender']

for col in bin_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

feature_cols = num_cols + bin_cols + cat_cols
X_num = data[feature_cols].values

scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

X_num_tensor = torch.tensor(X_num, dtype=torch.float32)
X_cat_tensor = torch.zeros((X_num.shape[0], 1), dtype=torch.long)  # Dummy for no categorical

dataset = TensorDataset(X_num_tensor, X_cat_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ======================
# Embedding Sizes to Train
# ======================

embedding_sizes = [16, 20, 32, 64]

# ======================
# Training Loop
# ======================

for embed_dim in embedding_sizes:
    print(f"🔵 Processing embedding size {embed_dim}...")

    model = TabTransformer(
        categories=[1],  # Dummy for no categorical columns
        num_continuous=len(feature_cols),
        dim=embed_dim,
        dim_out=embed_dim,
        depth=3,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0.0
        for batch in train_loader:
            x_num, x_cat = batch[0].to(device), batch[1].to(device)

            mask_prob = 0.15
            mask = (torch.rand(x_num.shape) < mask_prob).float().to(device)
            masked_x_num = x_num.clone()
            masked_x_num[mask == 1] = 0

            outputs = model(x_cat, masked_x_num)
            loss = criterion(outputs * mask, x_num * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 - Loss: {total_loss/len(train_loader):.6f}")

    # ======================
    # Generate Embeddings
    # ======================

    model.eval()
    all_embeddings = []

    full_loader = DataLoader(dataset, batch_size=512, shuffle=False)
    with torch.no_grad():
        for batch in full_loader:
            x_num, x_cat = batch[0].to(device), batch[1].to(device)
            embeddings = model(x_cat, x_num)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # Save embeddings and IDs separately
    np.save(os.path.join(embedding_output_dir, f'tabtransformer_embeddings_{embed_dim}d.npy'), all_embeddings)
    np.save(os.path.join(embedding_output_dir, f'tabtransformer_row_indices_{embed_dim}d.npy'), data['ID'].values)

    print(f"✅ Saved embeddings and IDs for {embed_dim}D.")

print("\n🎯 All TabTransformer embeddings generated and saved successfully!")
