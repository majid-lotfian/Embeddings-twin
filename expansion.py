import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import itertools

# ================ CONFIG ================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_list = ['CBC', 'Covid19', 'Iraq', 'Liverpool']
dim_map = {
    'CBC': [8, 16, 32, 64, 96, 128],
    'Covid19': [8, 16, 32, 64, 96, 128],
    'Iraq': [8, 16, 32, 64, 96, 128],
    'Liverpool': [8, 16, 32, 64, 96, 128]
}
method = 'FTTransformer'  # Change this to the FM name you're evaluating

# ================ METRICS ================
def continuity(X, X_emb, k=5):
    n = X.shape[0]
    D_orig = pairwise_distances(X)
    D_emb = pairwise_distances(X_emb)
    orig_neighbors = np.argsort(D_orig, axis=1)[:, 1:k+1]
    emb_neighbors = np.argsort(D_emb, axis=1)[:, 1:k+1]
    ranks = np.zeros(n)
    for i in range(n):
        missing = set(orig_neighbors[i]) - set(emb_neighbors[i])
        for m in missing:
            rank = np.where(orig_neighbors[i] == m)[0]
            if rank.size > 0:
                ranks[i] += rank[0] - k + 1
    Q = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * np.sum(ranks)
    return Q

def lid_mle_np(X, k=20):
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    D_sorted = np.sort(D, axis=1)[:, :k]
    r_k = D_sorted[:, -1]
    lids = -k / np.sum(np.log(D_sorted / r_k[:, None] + 1e-10), axis=1)
    return np.mean(lids)

def neighborhood_hit_rate(X, X_emb, k=5):
    D_orig = pairwise_distances(X)
    D_emb = pairwise_distances(X_emb)
    orig_neighbors = np.argsort(D_orig, axis=1)[:, 1:k+1]
    emb_neighbors = np.argsort(D_emb, axis=1)[:, 1:k+1]
    matches = [len(set(orig_neighbors[i]) & set(emb_neighbors[i])) / k for i in range(X.shape[0])]
    return np.mean(matches)

def redundancy_ratio(X, threshold=0.95):
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i, j]) > threshold:
                count += 1
    total = n * (n - 1) / 2
    return count / total if total > 0 else 0

# ================ RUN ====================
for dataset in dataset_list:
    print(f"\n📁 Dataset: {dataset}")
    file_ext = 'xlsx' if dataset.lower() == 'iraq' else 'csv'
    dataset_path = os.path.join(BASE_DIR, '..', '..', 'datasets', dataset, f'data_processed.{file_ext}')
    embedding_dir = os.path.join(BASE_DIR, dataset, f'embeddings_{method.lower()}')
    output_dir = os.path.join(BASE_DIR, 'resultsExpansion', dataset)
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_excel(dataset_path) if file_ext == 'xlsx' else pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()
    row_ids = data['ID'].values if 'ID' in data.columns else np.arange(2, len(data) + 2)
    data.drop(columns=['ID'], inplace=True, errors='ignore')

    if 'gender' in data.columns:
        data['gender'] = data['gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
    if 'sex' in data.columns:
        data['sex'] = data['sex'].astype(str).str.lower().map({'male': 1, 'female': 0})

    print(f"🔍 Before cleaning: {data.shape[0]} rows")
    data = data.select_dtypes(include=[np.number])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna(axis=1, how='all')
    print(f"✅ After cleaning: {data.shape[0]} rows")

    original_data = StandardScaler().fit_transform(data.values)
    method_results = {'dimension': [], 'continuity': [], 'lid': [],
                      'neighborhood_hit_rate': [], 'redundancy_ratio': []}

    for dim in dim_map[dataset]:
        print(f"➡️ {method} {dim}D")

        embed_path = os.path.join(embedding_dir, f'{method.lower()}_embeddings_{dim}d.npy')
        id_path = os.path.join(embedding_dir, f'{method.lower()}_row_indices_{dim}d.npy')

        if not os.path.exists(embed_path) or not os.path.exists(id_path):
            print(f"⚠️ Skipping {dim}D: Missing files.")
            continue

        embeddings = np.load(embed_path)
        ids = np.load(id_path)

        id_to_index = {rid: i for i, rid in enumerate(row_ids)}
        valid_pairs = [(id_to_index[i], j) for j, i in enumerate(ids) if i in id_to_index]

        if not valid_pairs:
            print(f"⚠️ No valid indices for {dim}D. Skipping.")
            continue

        original_idxs, embedding_idxs = zip(*valid_pairs)
        X_ref = original_data[list(original_idxs)]
        X_emb = embeddings[list(embedding_idxs)]

        if np.isnan(X_ref).any() or np.isnan(X_emb).any():
            print(f"⚠️ NaNs detected in data for {dim}D. Skipping.")
            continue

        if dataset.lower() == 'cbc' and X_ref.shape[0] > 15000:
            print("🔄 Subsampling CBC to 15,000 for memory efficiency.")
            indices = np.random.choice(X_ref.shape[0], 15000, replace=False)
            X_ref = X_ref[indices]
            X_emb = X_emb[indices]

        cont = continuity(X_ref, X_emb)
        lid_val = lid_mle_np(X_emb)
        nhr = neighborhood_hit_rate(X_ref, X_emb)
        red = redundancy_ratio(X_emb)

        method_results['dimension'].append(dim)
        method_results['continuity'].append(cont)
        method_results['lid'].append(lid_val)
        method_results['neighborhood_hit_rate'].append(nhr)
        method_results['redundancy_ratio'].append(red)

    df_results = pd.DataFrame(method_results)
    df_results.to_csv(os.path.join(output_dir, f'metrics_{method.lower()}_{dataset.lower()}.csv'), index=False)

    # Radar plot
    df = df_results.set_index('dimension')
    normalized = df.copy()
    for col in df.columns:
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
    ax.set_title(f"{method} - {dataset} Expansion Quality", size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'radar_{method.lower()}_{dataset.lower()}.png'))
    plt.close()

print("\n✅ All foundation model expansion evaluations completed.")
