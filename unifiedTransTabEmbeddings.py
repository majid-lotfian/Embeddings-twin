import os
import torch
import transtab
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================
# Setup
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
datasets_root = os.path.join(BASE_DIR, '..', '..', 'datasets')
embedding_root = os.path.join(BASE_DIR, 'embeddings_transtab')
os.makedirs(embedding_root, exist_ok=True)

# ==========================
# Dataset Configs
# ==========================

dataset_configs = {
    'CBC': {
        'file': 'data_processed.csv',
        'format': 'csv',
        'subdir': 'CBC',
        'id_col': 'ID',
        'encode_cols': ['gender'],
        'bin_cols': ['gender'],
        'cat_cols': [],
        'num_cols': [
            'white_blood_cells', 'neutrophil_count', 'lymphocyte_count', 'monocyte_count',
            'eosinophil_count', 'basophil_count', 'red_blood_cells', 'hemoglobin',
            'hematocrit', 'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin',
            'mean_corpuscular_hemoglobin_concentration', 'red_cell_distribution_width',
            'platelets', 'mean_platelet_volume', 'plateletcrit', 'platelet_distribution_width',
            'sd', 'sdtsd', 'tsd', 'ferritin', 'folate', 'vitamin_b12'
        ]
    },
    'Covid19': {
        'file': 'data_processed.csv',
        'format': 'csv',
        'subdir': 'Covid19',
        'id_col': 'id',
        'encode_cols': ['sex'],
        'bin_cols': ['sex'],
        'cat_cols': [],
        'num_cols': [
            'age', 'leukocyte_count', 'neutrophil_percentage', 'lymphocyte_percentage',
            'monocyte_percentage', 'eosinophil_percentage', 'basophil_percentage',
            'neutrophil_count', 'lymphocyte_count', 'monocyte_count',
            'eosinophil_count', 'basophil_count', 'red_blood_cell_count',
            'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin',
            'mean_corpuscular_hemoglobin_concentration', 'red_cell_distribution_width',
            'hemoglobin', 'hematocrit', 'platelet_count', 'mean_platelet_volume'
        ]
    },
    'Iraq': {
        'file': 'data_processed.xlsx',
        'format': 'xlsx',
        'subdir': 'Iraq',
        'id_col': 'id',
        'encode_cols': [],
        'bin_cols': [],
        'cat_cols': [],
        'num_cols': [
            'white_blood_cell_count', 'lymphocyte_percentage','other_wbc_percentage','neutrophil_percentage','lymphocyte_count',
            'other_wbc_count','neutrophil_count','red_blood_cell_count','hemoglobin','hematocrit','mean_corpuscular_volume',
            'mean_corpuscular_hemoglobin','mean_corpuscular_hemoglobin_concentration',
            'red_blood_cell_distribution_width_standard_deviation','red_blood_cell_distribution_width_coefficient_variation',
            'platelet_count','mean_platelet_volume','platelet_distribution_width','platelet_crit','platelet_large_cell_ratio'
        ]
    },
    'Liverpool': {
        'file': 'data_processed.csv',
        'format': 'csv',
        'subdir': 'Liverpool',
        'id_col': 'id',
        'encode_cols': [],
        'bin_cols': ['sex'],
        'cat_cols': [],
        'num_cols': [
            'age','red_blood_cell_count','Packed_Cell_Volume','Mean_Cell_Volume','Mean_Cell_Hemoglobin',
            'mean_corpuscular_hemoglobin_concentration','Red_Cell_Distribution_width','White_Blood_Cell	Platelet','Hemoglobin'
        ]
    }
}

embedding_sizes = [8, 12, 16, 20, 32, 64, 96, 128]

# ==========================
# Run For All Datasets
# ==========================

for name, cfg in dataset_configs.items():
    print(f"\n📁 Processing {name} dataset")

    dataset_path = os.path.join(datasets_root, cfg['subdir'], cfg['file'])
    if cfg['format'] == 'csv':
        data = pd.read_csv(dataset_path)
    else:
        data = pd.read_excel(dataset_path, engine='openpyxl')

    for col in cfg['encode_cols']:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    if cfg['id_col'] not in data.columns:
        data[cfg['id_col']] = data.index + 2

    train_data = data.sample(frac=0.8, random_state=42)
    dummy_targets = pd.Series([0] * len(train_data))
    trainset = (train_data, dummy_targets)
    row_indices = train_data[cfg['id_col']].values

    output_dir = os.path.join(embedding_root, name)
    checkpoints_dir = os.path.join(BASE_DIR, 'checkpoints_transtab', name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    dims = embedding_sizes if name != 'Liverpool' else [12, 16, 20, 32, 64, 96, 128]

    for embed_dim in dims:
        print(f"\n🔵 {name} - Embedding size {embed_dim}D")
        checkpoint_dim_dir = os.path.join(checkpoints_dir, f'{embed_dim}D')
        os.makedirs(checkpoint_dim_dir, exist_ok=True)

        model, collate_fn = transtab.build_contrastive_learner(
            cat_cols=cfg['cat_cols'],
            num_cols=cfg['num_cols'],
            bin_cols=cfg['bin_cols'],
            supervised=False,
            num_partition=4,
            overlap_ratio=0.5,
            embed_dim=embed_dim
        )

        model = model.to(device)
        training_arguments = {
            'num_epoch': 50,
            'batch_size': 64,
            'lr': 1e-4,
            'eval_metric': 'val_loss',
            'eval_less_is_better': True,
            'output_dir': checkpoint_dim_dir
        }

        transtab.train(model, trainset, valset=None, collate_fn=collate_fn, **training_arguments)

        enc = transtab.build_encoder(
            binary_columns=cfg['bin_cols'],
            checkpoint=checkpoint_dim_dir,
            num_layer=0
        ).to(device)

        output = enc(train_data)
        embeddings = output['embedding'].detach().cpu().numpy().reshape(len(train_data), -1)

        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        np.save(os.path.join(output_dir, f'transtab_embeddings_{embed_dim}d.npy'), embeddings)
        np.save(os.path.join(output_dir, f'transtab_row_indices_{embed_dim}d.npy'), row_indices)

        print(f"✅ Saved {embed_dim}D embeddings and IDs")

print("\n🎯 All TransTab embeddings generated successfully!")
