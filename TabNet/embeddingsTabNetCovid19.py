import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'Covid19', 'data_processed.csv')
embedding_output_dir = os.path.join(BASE_DIR, 'Covid19')
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load and Prepare Dataset
# ======================

data = pd.read_csv(dataset_path)
data.columns = data.columns.str.strip().str.lower()

# Store and drop ID column
row_indices = data['id'].values
data = data.drop(columns=['id'])

# ======================
# Define Feature Groups
# ======================

num_cols = [
    'age', 'leukocyte_count', 'neutrophil_percentage', 'lymphocyte_percentage',
    'monocyte_percentage', 'eosinophil_percentage', 'basophil_percentage',
    'neutrophil_count', 'lymphocyte_count', 'monocyte_count',
    'eosinophil_count', 'basophil_count', 'red_blood_cell_count',
    'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin',
    'mean_corpuscular_hemoglobin_concentration', 'red_cell_distribution_width',
    'hemoglobin', 'hematocrit', 'platelet_count', 'mean_platelet_volume'
]

bin_cols = ['sex']
cat_cols = []

# Encode binary categorical columns
for col in bin_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# ======================
# Feature Preprocessing
# ======================

feature_cols = num_cols + bin_cols + cat_cols
X = data[feature_cols].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ======================
# Embedding Sizes to Train
# ======================

embedding_sizes = [8, 12, 16, 20, 32, 64, 96, 128]

# ======================
# Train and Save Embeddings
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

    # Extract row-wise embeddings (proxy: reconstruction output)
    #embeddings = model.predict(X)
    embeddings = model.network.embedder.forward(X)

    np.save(os.path.join(embedding_output_dir, f'tabnet_embeddings_{emb_dim}d.npy'), embeddings)
    np.save(os.path.join(embedding_output_dir, f'tabnet_row_indices_{emb_dim}d.npy'), row_indices)

    print(f"✅ Saved embeddings and row indices for {emb_dim}D.")

print("\n🎯 All TabNet embeddings (multi-dim) generated successfully.")
