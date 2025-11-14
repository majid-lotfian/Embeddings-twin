import os
import torch
import transtab
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ======================
# Setup
# ======================

# Detect device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path (relative)
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')

# Checkpoints and embeddings output directories (relative)
checkpoint_root_dir = os.path.join(BASE_DIR, 'checkpoints')
embedding_output_dir = os.path.join(BASE_DIR, 'embeddings')

# Create folders if they don't exist
os.makedirs(checkpoint_root_dir, exist_ok=True)
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================
original_data = pd.read_csv(dataset_path)

# Process binary columns
if 'Gender' in original_data.columns:
    original_data['Gender'] = original_data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Define column types
cat_cols = []  # Add categorical column names if any
num_cols = ['Age', 'Hemoglobin', 'Platelet_Count', 'White_Blood_Cells', 'Red_Blood_Cells',
            'Mean_Corpuscular_Volume', 'Mean_Corpuscular_Hemoglobin', 'Mean_Corpuscular_Hemoglobin_Concentration']
bin_cols = ['Gender']  # Extend if needed

# ======================
# Split Dataset
# ======================
train_data = original_data.sample(frac=0.8, random_state=42)
remaining_data = original_data.drop(train_data.index)
val_data = remaining_data.sample(frac=0.5, random_state=42)
test_data = remaining_data.drop(val_data.index)

# Dummy targets for unsupervised contrastive learning
dummy_targets_train = pd.Series([0] * len(train_data))
trainset = (train_data, dummy_targets_train)

# ======================
# Embedding Sizes to Train
# ======================
embedding_sizes = [16, 20, 32, 64]

# ======================
# Train and Save Embeddings
# ======================
for embed_dim in embedding_sizes:
    print(f"🔵 Processing embedding size {embed_dim}...")

    # Checkpoint path for this size
    checkpoint_dir = os.path.join(checkpoint_root_dir, f'{embed_dim}D')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build contrastive learner
    model, collate_fn = transtab.build_contrastive_learner(
        cat_cols=cat_cols,
        num_cols=num_cols,
        bin_cols=bin_cols,
        supervised=False,
        num_partition=4,
        overlap_ratio=0.5,
        embed_dim=embed_dim
    )

    # Move model to GPU/CPU
    model = model.to(device)

    # Training arguments
    training_arguments = {
        'num_epoch': 50,
        'batch_size': 64,
        'lr': 1e-4,
        'eval_metric': 'val_loss',
        'eval_less_is_better': True,
        'output_dir': checkpoint_dir
    }

    # Train model
    transtab.train(model, trainset, valset=None, collate_fn=collate_fn, **training_arguments)

    # Build encoder
    enc = transtab.build_encoder(
        binary_columns=bin_cols,
        checkpoint=checkpoint_dir,
        num_layer=0
    )

    # Move encoder to GPU/CPU
    enc = enc.to(device)

    # Generate embeddings
    output = enc(train_data)
    embeddings = output['embedding']

    # Detach, move to CPU, convert to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)

    # Normalize embeddings (recommended for metric evaluation)
    scaler = StandardScaler()
    embeddings_np = scaler.fit_transform(embeddings_np)

    # Save as .npy
    npy_output_path = os.path.join(embedding_output_dir, f'transtab_embeddings_{embed_dim}d.npy')
    np.save(npy_output_path, embeddings_np)

    print(f"✅ Saved {embed_dim}D embeddings at {npy_output_path}")

print("\n🎯 All embeddings generated and saved as .npy files successfully!")
