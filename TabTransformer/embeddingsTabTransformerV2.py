import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
embedding_output_dir = os.path.join(BASE_DIR, 'CBC')
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================

data = pd.read_csv(dataset_path)
data.columns = data.columns.str.strip()

if 'ID' not in data.columns:
    data['ID'] = data.index + 2

cat_cols = []  # No categorical columns
num_cols = [
    'white_blood_cells', 'neutrophil_count', 'lymphocyte_count', 'monocyte_count', 'eosinophil_count', 'basophil_count',
    'red_blood_cells', 'hemoglobin', 'hematocrit', 'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin',
    'mean_corpuscular_hemoglobin_concentration', 'red_cell_distribution_width', 'platelets', 'mean_platelet_volume',
    'plateletcrit', 'platelet_distribution_width', 'SD', 'SDTSD', 'TSD', 'ferritin', 'folate', 'vitamin_b12'
]
bin_cols = ['gender']

for col in bin_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

feature_cols = num_cols + bin_cols + cat_cols
X_num = data[feature_cols].values

scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# Dummy categorical tensor
X_cat = np.zeros((X_num.shape[0], 1), dtype=np.int64)

X_num_tensor = torch.tensor(X_num, dtype=torch.float32)
X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)

dataset = TensorDataset(X_num_tensor, X_cat_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ======================
# Embedding Sizes to Train
# ======================

#embedding_sizes = [8, 12, 16, 20, 32, 64]
embedding_sizes = [96, 128]


# ======================
# Training Loop (masked feature reconstruction)
# ======================

def mask_inputs(x, mask_ratio=0.3):
    mask = torch.rand_like(x) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0  # You can also use mean imputation if needed
    return x_masked, mask

for embed_dim in embedding_sizes:
    print(f"\n🔵 Processing embedding size {embed_dim}...")

    model = TabTransformer(
        categories=[1],  # Dummy
        num_continuous=len(feature_cols),
        dim=embed_dim,
        dim_out=len(feature_cols),  # Predict original features
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

            x_masked, mask = mask_inputs(x_num)
            outputs = model(x_cat, x_masked)

            loss = criterion(outputs[mask], x_num[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 - Avg Loss: {total_loss/len(train_loader):.6f}")

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

    np.save(os.path.join(embedding_output_dir, f'tabtransformer_embeddings_{embed_dim}d.npy'), all_embeddings)
    np.save(os.path.join(embedding_output_dir, f'tabtransformer_row_indices_{embed_dim}d.npy'), data['ID'].values)

    print(f"✅ Saved embeddings and IDs for {embed_dim}D.")

print("\n🎯 All TabTransformer embeddings generated successfully!")
