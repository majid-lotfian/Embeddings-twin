import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetEncoder

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
checkpoint_root_dir = os.path.join(BASE_DIR, 'checkpoints_tabnet')
embedding_output_dir = os.path.join(BASE_DIR, 'embeddings_tabnet')

os.makedirs(checkpoint_root_dir, exist_ok=True)
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================

data = pd.read_csv(dataset_path)

# If ID column not present, create it
if 'ID' not in data.columns:
    data['ID'] = data.index + 2

cat_cols = []  # No categorical features
num_cols = ['Age', 'Hemoglobin', 'Platelet_Count', 'White_Blood_Cells', 'Red_Blood_Cells',
            'Mean_Corpuscular_Volume', 'Mean_Corpuscular_Hemoglobin', 'Mean_Corpuscular_Hemoglobin_Concentration']
bin_cols = ['Gender']

for col in bin_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

feature_cols = num_cols + bin_cols + cat_cols
X = data[feature_cols].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X_tensor)
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

    model = TabNetEncoder(
        input_dim=X_tensor.shape[1],
        output_dim=embed_dim,
        n_d=embed_dim,
        n_a=embed_dim,
        n_steps=3,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=128,
        momentum=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            outputs, M_loss = model(x)
            loss = criterion(outputs, x)

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
            x = batch[0].to(device)
            embeddings, _ = model(x)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # ======================
    # Save embeddings and IDs
    # ======================

    np.save(os.path.join(embedding_output_dir, f'tabnet_embeddings_{embed_dim}d.npy'), all_embeddings)
    np.save(os.path.join(embedding_output_dir, f'tabnet_row_indices_{embed_dim}d.npy'), data['ID'].values)

    print(f"✅ Saved embeddings and IDs for {embed_dim}D.")

print("\n🎯 All TabNet embeddings generated and saved successfully!")
