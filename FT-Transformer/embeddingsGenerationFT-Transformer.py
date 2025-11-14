import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from rtdl_revisiting_models import FTTransformer

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
checkpoint_root_dir = os.path.join(BASE_DIR, 'checkpoints_fttransformer')
embedding_output_dir = os.path.join(BASE_DIR, 'embeddings_fttransformer')

os.makedirs(checkpoint_root_dir, exist_ok=True)
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================

data = pd.read_csv(dataset_path)

if 'ID' not in data.columns:
    data['ID'] = data.index + 2

cat_cols = []  # No categorical features here
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

dataset = TensorDataset(X_num_tensor)
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

    model = FTTransformer(
        n_cont_features=len(feature_cols),
        cat_cardinalities=[],
        d_out=embed_dim,
        n_blocks=3,
        d_block=embed_dim,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0.0
        for batch in train_loader:
            x_num = batch[0].to(device)

            mask_prob = 0.15
            mask = (torch.rand(x_num.shape) < mask_prob).float().to(device)
            masked_x_num = x_num.clone()
            masked_x_num[mask == 1] = 0

            outputs = model(masked_x_num, None)
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
            x_num = batch[0].to(device)
            embeddings = model(x_num, None)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # Save embeddings and IDs separately
    np.save(os.path.join(embedding_output_dir, f'fttransformer_embeddings_{embed_dim}d.npy'), all_embeddings)
    np.save(os.path.join(embedding_output_dir, f'fttransformer_row_indices_{embed_dim}d.npy'), data['ID'].values)

    print(f"✅ Saved embeddings and IDs for {embed_dim}D.")

print("\n🎯 All FT-Transformer embeddings generated and saved successfully!")
