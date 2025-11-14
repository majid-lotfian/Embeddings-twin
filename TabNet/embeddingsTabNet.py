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
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
#embedding_output_dir = os.path.join(BASE_DIR, 'CBC', 'embeddings_tabnet_multiple')
embedding_output_dir = os.path.join(BASE_DIR, 'CBC', 'embeddings_corrected')

os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================

data = pd.read_csv(dataset_path)
data.columns = data.columns.str.strip()

if 'ID' not in data.columns:
    data['ID'] = data.index + 2

# Define features
cat_cols = []
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
X = data[feature_cols].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
row_indices = data['ID'].values

# ======================
# Embedding Sizes to Train
# ======================

#embedding_sizes = [8, 12, 16, 20, 32, 64, 96, 128]
embedding_sizes = [128]


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

    # Extract row-wise embeddings (reconstruction output as proxy)
    #embeddings = model.predict(X)
    embeddings = model.network.embedder.forward(X)


    np.save(os.path.join(embedding_output_dir, f'tabnet_embeddings_{emb_dim}d.npy'), embeddings)
    np.save(os.path.join(embedding_output_dir, f'tabnet_row_indices_{emb_dim}d.npy'), row_indices)

    print(f"✅ Saved embeddings and row indices for {emb_dim}D.")

print("\n🎯 All TabNet embeddings (multi-dim) generated successfully.")
