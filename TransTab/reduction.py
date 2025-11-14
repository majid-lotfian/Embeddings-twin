import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
import itertools

# ================ SETUP ================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
method = 'TransTab'  # Change this to the method you're evaluating
datasets = ['Covid19', 'Iraq', 'Liverpool','CBC']
#datasets = ['CBC']

dimensions = [8, 12, 16, 20, 32, 64, 96, 128]

# ================ METRICS ================

def knn_preservation(X, X_embedded, k=5):
    nn_orig = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    nn_embed = NearestNeighbors(n_neighbors=k).fit(X_embedded).kneighbors(return_distance=False)
    matches = [len(set(nn_orig[i]) & set(nn_embed[i])) / k for i in range(len(X))]
    return np.mean(matches)

def pairwise_distance_mse(X, X_embedded):
    D1 = pairwise_distances(X)
    D2 = pairwise_distances(X_embedded)
    return np.mean((D1 - D2) ** 2)

def shepard_correlation(X, X_embedded):
    D1 = pairwise_distances(X).ravel()
    D2 = pairwise_distances(X_embedded).ravel()
    return spearmanr(D1, D2).correlation

# ================ RUN PER DATASET ================

for dataset in datasets:
    print(f"\n📁 Processing {dataset}")

    file_ext = 'xlsx' if dataset.lower() == 'iraq' else 'csv'
    dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', dataset, f'data_processed.{file_ext}')
    embedding_dir = os.path.join(BASE_DIR, f'embeddings_{method.lower()}', dataset)

    # Load data
    if file_ext == 'xlsx':
        data = pd.read_excel(dataset_path)
    else:
        data = pd.read_csv(dataset_path)

    data.columns = data.columns.str.strip()
    row_ids = data['ID'].values if 'ID' in data.columns else np.arange(2, len(data) + 2)
    data = data.drop(columns=['ID'], errors='ignore')

    # Handle categorical
    if 'gender' in data.columns:
        data['gender'] = data['gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
    if 'sex' in data.columns:
        data['sex'] = data['sex'].astype(str).str.lower().map({'male': 1, 'female': 0})

    # Clean and preprocess
    print(f"🔍 Before cleaning: {data.shape[0]} rows")
    data = data.select_dtypes(include=[np.number])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna(axis=1, how='all')
    print(f"✅ After cleaning: {data.shape[0]} rows")

    scaler = StandardScaler()
    original_data = scaler.fit_transform(data.values)

    method_results = {'dimension': [], 'trustworthiness': [], 'knn_preservation': [],
                      'pairwise_mse': [], 'shepard_corr': []}

    for dim in dimensions:
        print(f"➡️ {method} {dim}D")

        embed_path = os.path.join(embedding_dir, f'{method.lower()}_embeddings_{dim}d.npy')
        id_path = os.path.join(embedding_dir, f'{method.lower()}_row_indices_{dim}d.npy')

        if not os.path.exists(embed_path) or not os.path.exists(id_path):
            print(f"⚠️ Skipping {method} {dim}D: missing files")
            continue

        embeddings = np.load(embed_path)
        ids = np.load(id_path)

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"IDs shape: {ids.shape}")
        print(f"Row IDs shape: {row_ids.shape}")

        # If embeddings are 3D, flatten to 2D by selecting the first slice
        if embeddings.ndim == 3:
            print("⚠️ Detected 3D embeddings. Using the first slice [0].")
            embeddings = embeddings[0]



        id_to_index = {rid: i for i, rid in enumerate(row_ids)}
        valid_pairs = [(id_to_index[i], j) for j, i in enumerate(ids) if i in id_to_index]

        if not valid_pairs:
            print(f"⚠️ No valid indices for {dim}D. Skipping.")
            continue

        original_idxs, embedding_idxs = zip(*valid_pairs)
        X_ref = original_data[list(original_idxs)]
        X_emb = embeddings[list(embedding_idxs)]

        # Subsample ONLY for CBC if large
        if dataset.lower() == 'cbc' and X_ref.shape[0] > 15000:
            print("🔄 Subsampling CBC to 15,000 for memory efficiency.")
            indices = np.random.choice(X_ref.shape[0], 15000, replace=False)
            X_ref = X_ref[indices]
            X_emb = X_emb[indices]

        if np.isnan(X_ref).any() or np.isnan(X_emb).any():
            print(f"⚠️ NaNs detected in data for {dim}D. Skipping.")
            continue

        trust = trustworthiness(X_ref, X_emb, n_neighbors=5)
        knn = knn_preservation(X_ref, X_emb, k=5)
        mse = pairwise_distance_mse(X_ref, X_emb)
        shep = shepard_correlation(X_ref, X_emb)

        method_results['dimension'].append(dim)
        method_results['trustworthiness'].append(trust)
        method_results['knn_preservation'].append(knn)
        method_results['pairwise_mse'].append(mse)
        method_results['shepard_corr'].append(shep)

    # Save results
    output_dir = os.path.join(BASE_DIR, 'results', dataset)
    os.makedirs(output_dir, exist_ok=True)
    df_results = pd.DataFrame(method_results)
    df_results.to_csv(os.path.join(output_dir, f'metrics_{method.lower()}_{dataset.lower()}.csv'), index=False)

    # Radar plot
    df = df_results.set_index('dimension')
    normalized = df.copy()
    for col in df.columns:
        if col == 'pairwise_mse':
            normalized[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    labels = normalized.columns
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'x', 'D', '*', 'v', '+']
    style_cycle = itertools.cycle([(ls, m) for ls in linestyles for m in markers])

    for idx, row in normalized.iterrows():
        values = row.tolist() + [row.tolist()[0]]
        linestyle, marker = next(style_cycle)
        ax.plot(angles, values, linestyle=linestyle, marker=marker, label=f'{idx}D')
        ax.fill(angles, values, alpha=0.05)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"{method} - {dataset} Embedding Quality", size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'radar_{method.lower()}_{dataset.lower()}.png'))
    plt.close()

print("\n✅ All evaluations completed.")
