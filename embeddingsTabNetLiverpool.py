import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'Liverpool', 'data_processed.csv')
embedding_output_dir = os.path.join(BASE_DIR, 'Liverpool')
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load and Preprocess Data
# ======================

data = pd.read_csv(dataset_path)
data.columns = data.columns.str.strip().str.lower()

# Extract and remove ID
row_indices = data['id'].values
data = data.drop(columns=['id'])

# All features are numeric (including 'sex' as 0/1)
feature_cols = list(data.columns)
X = data[feature_cols].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ======================
# Embedding Sizes to Train
# ======================

embedding_sizes = [8, 12, 16, 20, 32, 64, 96, 128]

# ======================
# Pretraining and Saving
# ======================

for emb_dim in embedding_sizes:
    print(f"\n🔵 Training TabNetPretrainer for {emb_dim}D embeddings...")

    model = TabNetPretrainer(
        input_dim=X.shape[1],
        n_d=emb_dim,
        n_a=emb_dim,
        n_steps=3,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        momentum=0.3,
        mask_type="entmax"
    )

    model.fit(
        X_train=X,
        eval_set=[X],
        max_epochs=50,
        batch_size=64,
        num_workers=0,
        drop_last=False
    )

    # Use the model's reconstruction output as embeddings
    #embeddings = model.predict(X)
    embeddings = model.network.embedder.forward(X)

    np.save(os.path.join(embedding_output_dir, f'tabnet_embeddings_{emb_dim}d.npy'), embeddings)
    np.save(os.path.join(embedding_output_dir, f'tabnet_row_indices_{emb_dim}d.npy'), row_indices)

    print(f"✅ Saved embeddings and row indices for {emb_dim}D.")

print("\n🎯 All TabNet embeddings generated successfully.")
