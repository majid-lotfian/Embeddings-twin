import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Adjust the path if needed
saint_path = os.path.join(os.path.dirname(__file__), 'saint-main' ,'models')
sys.path.append(saint_path)


from model import TabAttention as SAINT  # Ensure SAINT is installed and available

# ======================
# Setup
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', 'CBC', 'data_processed.csv')
checkpoint_root_dir = os.path.join(BASE_DIR, 'CBC' ,'checkpoints_saint')
embedding_output_dir = os.path.join(BASE_DIR, 'CBC' ,'embeddings_saint')

os.makedirs(checkpoint_root_dir, exist_ok=True)
os.makedirs(embedding_output_dir, exist_ok=True)

# ======================
# Load Dataset
# ======================

data = pd.read_csv(dataset_path)
data.columns = data.columns.str.strip()

if 'ID' not in data.columns:
    data['ID'] = data.index + 2

cat_cols = []  # Add categorical features if available
num_cols = ['white_blood_cells', 'neutrophil_count', 'lymphocyte_count', 'monocyte_count', 'eosinophil_count', 'basophil_count',
            'red_blood_cells', 'hemoglobin', 'hematocrit', 'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin',
            'mean_corpuscular_hemoglobin_concentration', 'red_cell_distribution_width', 'platelets', 'mean_platelet_volume',
            'plateletcrit', 'platelet_distribution_width', 'SD', 'SDTSD', 'TSD', 'ferritin', 'folate', 'vitamin_b12']
bin_cols = ['gender']

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

embedding_sizes = [8, 12, 16, 20, 32, 64, 96, 128]

# ======================
# Training and Embedding Generation
# ======================

for embed_dim in embedding_sizes:
    print(f"\n🔵 Processing embedding size {embed_dim}...")

    model = SAINT(
        categories=[],
        num_continuous=len(feature_cols),
        dim=embed_dim,
        dim_out=len(feature_cols),
        depth=3,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.1,
        mlp_hidden_mults=(4,),
        attentiontype='col'
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0.0
        for batch in train_loader:
            x_num = batch[0].to(device)

            # Apply input masking
            mask_prob = 0.15
            mask = (torch.rand(x_num.shape) < mask_prob).float().to(device)
            masked_x_num = x_num.clone()
            masked_x_num[mask == 1] = 0

            # Prepare inputs for SAINT
            x_cont = masked_x_num
            x_categ = torch.empty((x_cont.size(0), 0), dtype=torch.long).to(device)  # 2D dummy
            x_categ_enc = torch.empty((x_cont.size(0), 0, model.dim)).to(device)     # 3D dummy

            x_cont_enc = torch.stack([
                model.simple_MLP[i](x_cont[:, i].unsqueeze(1))
                for i in range(x_cont.size(1))
            ], dim=1)

            # Forward pass
            outputs = model(x_categ, x_cont, x_categ_enc, x_cont_enc)

            # Loss and optimization
            loss = criterion(outputs * mask, x_num * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 - Loss: {total_loss/len(train_loader):.6f}")


        # ======================
        # Generate Sample-Level Embeddings
        # ======================

        model.eval()
        all_embeddings = []

        full_loader = DataLoader(dataset, batch_size=512, shuffle=False)
        with torch.no_grad():
            for batch in full_loader:
                x_num = batch[0].to(device)

                # Prepare dummy categorical inputs
                x_categ = torch.empty((x_num.size(0), 0), dtype=torch.long).to(device)
                x_categ_enc = torch.empty((x_num.size(0), 0, model.dim)).to(device)

                x_cont_enc = torch.stack([
                    model.simple_MLP[i](x_num[:, i].unsqueeze(1))
                    for i in range(x_num.size(1))
                ], dim=1)

                # Use model output directly as embeddings
                sample_embeddings = model.get_embeddings(x_categ, x_num, x_categ_enc, x_cont_enc)
                all_embeddings.append(sample_embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()



    # ======================
    # Save Embeddings and IDs
    # ======================

    np.save(os.path.join(embedding_output_dir, f'saint_embeddings_{embed_dim}d.npy'), all_embeddings)
    np.save(os.path.join(embedding_output_dir, f'saint_row_indices_{embed_dim}d.npy'), data['ID'].values)

    print(f"✅ Saved sample-level embeddings and IDs for {embed_dim}D.")

print("\n🎯 All SAINT sample-level embeddings generated successfully!")
