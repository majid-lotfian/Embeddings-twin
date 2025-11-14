import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from rtdl_revisiting_models import FTTransformer

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'Iraq', 'data_processed.xlsx')  # replace with your actual filename
embedding_output_dir = os.path.join(BASE_DIR, 'Iraq')
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load and Preprocess Data
# ======================

data = pd.read_excel(dataset_path)
data.columns = data.columns.str.strip().str.lower()

# Save and drop ID column
ids = data['id'].values
data = data.drop(columns=['id'])

# All columns are numerical
feature_cols = list(data.columns)

# Normalize
X = data[feature_cols].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ======================
# Embedding Sizes to Train
# ======================

embedding_sizes = [8, 16, 32, 64, 96, 128]

# ======================
# Training Loop
# ======================

for embed_dim in embedding_sizes:
    print(f"\n🔵 Processing embedding size {embed_dim}...")

    model = FTTransformer(
        n_cont_features=len(feature_cols),
        cat_cardinalities=[],  # No categorical features
        d_out=embed_dim,
        n_blocks=3,
        d_block=embed_dim,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4/3,
        ffn_dropout=0.1,
        residual_dropout=0.0
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0.0
        for batch in train_loader:
            x_num = batch[0].to(device)
            outputs = model(x_num, None)
            loss = criterion(outputs, torch.zeros_like(outputs))  # stability loss

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
            x_num = batch[0].to(device)
            embeddings = model(x_num, None)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    np.save(os.path.join(embedding_output_dir, f'fttransformer_embeddings_{embed_dim}d.npy'), all_embeddings)
    np.save(os.path.join(embedding_output_dir, f'fttransformer_row_indices_{embed_dim}d.npy'), ids)

    print(f"✅ Saved embeddings and IDs for {embed_dim}D.")

print("\n🎯 All FT-Transformer embeddings generated successfully!")
