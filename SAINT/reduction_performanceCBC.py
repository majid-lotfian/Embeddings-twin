import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
#from sklearn.metrics import trustworthiness
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# ================
# Setup
# ================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
method = 'SAINT'  # Set this to the method you want to evaluate
#datasets = ['CBC', 'Covid19', 'Iraq', 'Liverpool']
datasets = ['Iraq']
dimensions = [8, 12, 16, 20, 32, 64, 96, 128]

# ================
# Metric Definitions
# ================

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

# ================
# Run Evaluation Per Dataset
# ================

for dataset in datasets:
    print(f"\n📁 Processing {dataset} dataset")

    #dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', dataset, 'data_processed.csv')
    dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', dataset, 'data_processed.xlsx')

    embedding_dir = os.path.join(BASE_DIR, dataset, 'embeddings_saint')
    #embedding_dir = os.path.join(BASE_DIR, dataset)

    #data = pd.read_csv(dataset_path) # for other datasets
    data = pd.read_excel(dataset_path) # for Iraq dataset
    data.columns = data.columns.str.strip()
    if 'ID' in data.columns:
        row_ids = data['ID'].values
        data = data.drop(columns=['ID'])
    else:
        row_ids = np.arange(2, len(data) + 2)

    if 'gender' in data.columns:
        data['gender'] = data['gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
    if 'sex' in data.columns:
        data['sex'] = data['sex'].astype(str).str.lower().map({'male': 1, 'female': 0})

    data = data.select_dtypes(include=[np.number])
    #data = data.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"🔍 Before cleaning: {data.shape[0]} rows")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    print(f"✅ After cleaning: {data.shape[0]} rows")



    # Drop constant columns
    data = data.loc[:, data.nunique() > 1]

    # Drop all-NaN columns
    data = data.dropna(axis=1, how='all')

    # Apply scaling
    scaler = StandardScaler()
    original_data = scaler.fit_transform(data.values)


    method_results = {'dimension': [], 'trustworthiness': [], 'knn_preservation': [],
                      'pairwise_mse': [], 'shepard_corr': []}

    for dim in dimensions:
        print(f"Processing {method} {dim}D in {dataset}.")
        embed_path = os.path.join(embedding_dir, f'{method.lower()}_embeddings_{dim}d.npy')
        id_path = os.path.join(embedding_dir, f'{method.lower()}_row_indices_{dim}d.npy')

        if not os.path.exists(embed_path) or not os.path.exists(id_path):
            print(f"Skipping {method} {dim}D for {dataset}: missing files")
            continue

        embeddings = np.load(embed_path)
        ids = np.load(id_path)

        id_to_index = {rid: i for i, rid in enumerate(row_ids)}
        #valid_indices = [id_to_index[i] for i in ids if i in id_to_index]
        #X_ref = original_data[valid_indices]
        #X_emb = embeddings[:len(valid_indices)]


        # Create aligned pairs of (embedding, original row)
        valid_pairs = [(id_to_index[i], j) for j, i in enumerate(ids) if i in id_to_index]

        if not valid_pairs:
            print(f" No valid indices found for {method} {dim}D in {dataset}. Skipping.")
            continue

        original_idxs, embedding_idxs = zip(*valid_pairs)
        X_ref = original_data[list(original_idxs)]
        X_emb = embeddings[list(embedding_idxs)]

        # Ensure no NaNs
        if np.isnan(X_ref).any() or np.isnan(X_emb).any():
                print(f"⚠️ NaNs detected in data for {method} {dim}D in {dataset}. Skipping.")
                continue

        '''# Subsample for speed & memory
        sample_size = 5000
        if X_ref.shape[0] > sample_size:
                indices = np.random.choice(X_ref.shape[0], sample_size, replace=False)
                X_ref_sample = X_ref[indices]
                X_emb_sample = X_emb[indices]
        else:
                X_ref_sample = X_ref
                X_emb_sample = X_emb'''

        X_ref_sample = X_ref
        X_emb_sample = X_emb


        # 🔹 Compute metrics on subsampled data
        trust = trustworthiness(X_ref_sample, X_emb_sample, n_neighbors=5)
        knn = knn_preservation(X_ref_sample, X_emb_sample, k=5)
        mse = pairwise_distance_mse(X_ref_sample, X_emb_sample)
        shep = shepard_correlation(X_ref_sample, X_emb_sample)

        # 🔹 Store results
        method_results['dimension'].append(dim)
        method_results['trustworthiness'].append(trust)
        method_results['knn_preservation'].append(knn)
        method_results['pairwise_mse'].append(mse)
        method_results['shepard_corr'].append(shep)



    df_results = pd.DataFrame(method_results)
    df_results.to_csv(os.path.join(BASE_DIR, 'results', dataset, f'metrics_{method.lower()}_{dataset.lower()}.csv'), index=False)

    # Radar Plot
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

    for idx, row in normalized.iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, label=f'{idx}D')
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"{method} - {dataset} Embedding Quality", size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'results', dataset, f'radar_{method.lower()}_{dataset.lower()}.png'))
    plt.close()


print("\n✅ Metric evaluation and radar plots completed.")
