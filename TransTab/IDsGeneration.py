import os
import pandas as pd
import numpy as np

# ======================
# Setup
# ======================

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the dataset used during TransTab embedding generation
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')

# Path where you saved your TransTab embeddings
embedding_output_dir = os.path.join(BASE_DIR, 'embeddings_transtab')

# Make sure output folder exists
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================

data = pd.read_csv(dataset_path)

# If ID column wasn't there before, recreate it exactly like original training
if 'ID' not in data.columns:
    data['ID'] = data.index + 2

# Recreate the train split (assuming original split was 80% random sampling)
train_data = data.sample(frac=0.8, random_state=42)  # Same random_state for exact matching

# Extract row indices (IDs)
row_indices = train_data['ID'].values  # or use train_data.index.values if you prefer

# ======================
# Save Row Indices
# ======================

# Save one .npy file for row indices
np.save(os.path.join(embedding_output_dir, 'transtab_row_indices.npy'), row_indices)

print(f"✅ Saved TransTab row indices at {os.path.join(embedding_output_dir, 'transtab_row_indices.npy')}")
