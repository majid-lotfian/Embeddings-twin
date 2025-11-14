import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import seaborn as sns

# ===============================
# CONFIGURATION
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folders = ['reduction', 'expansion']
datasets = ['cbc', 'covid19', 'iraq', 'liverpool']
methods_fm = ['tabnet', 'saint', 'transtab', 'tabtransformer', 'fttransformer']
methods_reduction = ['pca', 'vae', 'umap']
methods_expansion = ['vae', 'polyexpand', 'randomproj']

# Optimized weights for reduction and expansion
optimized_weights = {
    'reduction': {
        'cbc': {'trustworthiness': 0.971, 'knn_preservation': 0.010, 'pairwise_mse': 0.010, 'shepard_corr': 0.010},
        'covid19': {'trustworthiness': 0.806, 'knn_preservation': 0.017, 'pairwise_mse': 0.018, 'shepard_corr': 0.158},
        'iraq': {'trustworthiness': 0.036, 'knn_preservation': 0.258, 'pairwise_mse': 0.662, 'shepard_corr': 0.043},
        'liverpool': {'trustworthiness': 0.184, 'knn_preservation': 0.220, 'pairwise_mse': 0.217, 'shepard_corr': 0.379},
    },
    'expansion': {
        'cbc': {'continuity': 0.971, 'lid': 0.010, 'neighborhood_hit_rate': 0.010, 'redundancy_ratio': 0.010},
        'covid19': {'continuity': 0.010, 'lid': 0.010, 'neighborhood_hit_rate': 0.971, 'redundancy_ratio': 0.010},
        'iraq': {'continuity': 0.010, 'lid': 0.010, 'neighborhood_hit_rate': 0.971, 'redundancy_ratio': 0.010},
        'liverpool': {'continuity': 0.026, 'lid': 0.047, 'neighborhood_hit_rate': 0.903, 'redundancy_ratio': 0.024},
    }
}

# ===============================
# SCORE COMPUTATION AND PLOTTING
# ===============================
all_results = []
plot_data = {'reduction': {}, 'expansion': {}}

for direction in folders:
    metric_set = list(optimized_weights[direction]['cbc'].keys())
    for dataset in datasets:
        valid_dims = [8] if dataset == 'liverpool' and direction == 'reduction' else [32, 64, 96, 128] if dataset == 'liverpool' else ([8, 12, 16, 20] if direction == 'reduction' else [32, 64, 96, 128])

        for method in methods_fm + (methods_reduction if direction == 'reduction' else methods_expansion):
            file_path = os.path.join(BASE_DIR, direction, f"metrics_{method}_{dataset}.csv")
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            df['dimension'] = df['dimension'].astype(int)
            df = df[df['dimension'].isin(valid_dims)]

            if method == 'polyexpand' and direction == 'expansion':
                df = df[~df['dimension'].isin([96, 128])]
            if method == 'fttransformer':
                df = df[df['dimension'] % 8 == 0]

            scores = []
            for dim in sorted(df['dimension'].unique()):
                row = df[df['dimension'] == dim]
                if row.empty:
                    continue
                values = row[metric_set].values[0]
                norm_vals = minmax_scale(values) if np.ptp(values) > 1e-6 else np.ones_like(values) * 0.5
                weights = np.array([optimized_weights[direction][dataset][m] for m in metric_set])
                score = np.dot(norm_vals, weights)
                scores.append({'dataset': dataset, 'method': method, 'dimension': dim, 'score': score, 'direction': direction})

            plot_data[direction].setdefault(dataset, []).extend(scores)
            all_results.extend(scores)

# ===============================
# EXPORT TO EXCEL
# ===============================
results_df = pd.DataFrame(all_results)
excel_path = os.path.join(BASE_DIR, 'results', 'iqr_iqe_scores.xlsx')
results_df.to_excel(excel_path, index=False)

# ===============================
# PLOTTING PER DATASET
# ===============================
output_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(output_dir, exist_ok=True)

for direction in ['reduction', 'expansion']:
    for dataset in datasets:
        df = pd.DataFrame(plot_data[direction].get(dataset, []))
        if df.empty:
            continue
        plt.figure(figsize=(8, 5))
        for method in df['method'].unique():
            sub = df[df['method'] == method]
            dims = sub['dimension'].values
            scores = sub['score'].values
            if len(dims) < 2:
                continue
            sorted_idx = np.argsort(dims)
            plt.plot(dims[sorted_idx], scores[sorted_idx], label=method, marker='o')
        plt.xticks(sorted(df['dimension'].unique()))
        plt.title(f"{'IQR' if direction == 'reduction' else 'IQE'} Trend - {dataset.upper()}")
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Score')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{direction}_trend_{dataset}.png"))
        plt.close()

print("\n✅ IQR and IQE trends saved as plots and Excel file.")
