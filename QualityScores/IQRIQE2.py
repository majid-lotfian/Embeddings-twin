import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from glob import glob
import seaborn as sns

# ========== CONFIG ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
reduction_dims = [8, 12, 16, 20]
expansion_dims = [32, 64, 96, 128]
datasets = ['cbc', 'covid19', 'iraq', 'liverpool']
methods = ['tabnet', 'saint', 'transtab', 'tabtransformer', 'fttransformer',
           'pca', 'vae', 'umap', 'polyexpand', 'randomproj']
metric_weights_path = os.path.join(BASE_DIR,'results', 'optimized_weights_full.xlsx')
data_folders = {
    'reduction': os.path.join(BASE_DIR, 'reduction'),
    'expansion': os.path.join(BASE_DIR, 'expansion'),
}
fig_dir = os.path.join(BASE_DIR, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ========== LOAD WEIGHTS ==========
weights_df = pd.read_excel(metric_weights_path)
weights_df['direction'] = weights_df['direction'].str.lower()
weights_df['dataset'] = weights_df['dataset'].str.lower()

# ========== PREP ========== 
all_scores = []

for direction in ['reduction', 'expansion']:
    dims = reduction_dims if direction == 'reduction' else expansion_dims
    # Determine direction-specific metrics
    if direction == 'reduction':
        metrics = ['trustworthiness', 'knn_preservation', 'pairwise_mse', 'shepard_corr']
    else:
        metrics = ['continuity', 'lid', 'neighborhood_hit_rate', 'redundancy_ratio']    
    for dataset in datasets:
        for method in methods:
            fname = f'metrics_{method}_{dataset}.csv'
            fpath = os.path.join(data_folders[direction], fname)
            if not os.path.exists(fpath):
                continue

            df = pd.read_csv(fpath)
            df.columns = df.columns.str.strip().str.lower()
            df['dimension'] = df['dimension'].astype(int)
            # Ensure only available metric columns are used
            metrics = [m for m in metrics if m in df.columns]
            if len(metrics) < 2:
                continue  # skip if insufficient metrics

            # Special case: skip dims unavailable for polyexpand or fttransformer
            if method == 'polyexpand' and dataset == 'liverpool':
                df = df[df['dimension'].isin([32, 64])]  # only valid ones
            if method == 'fttransformer':
                df = df[df['dimension'] % 8 == 0]

            if df.empty:
                continue

            # Normalize metrics
            metric_vals = df[metrics].copy()
            metric_vals = metric_vals.replace([np.inf, -np.inf], np.nan).dropna()
            if metric_vals.empty:
                continue

            norm_metrics = pd.DataFrame(minmax_scale(metric_vals, axis=0), columns=metrics)
            norm_metrics['dimension'] = df['dimension'].values

            # Get correct weights
            row = weights_df[(weights_df['dataset'] == dataset) & (weights_df['direction'] == direction)]
            if row.empty:
                continue
            weights = row[metrics].values.flatten()

            for dim in norm_metrics['dimension'].unique():
                sub = norm_metrics[norm_metrics['dimension'] == dim][metrics]
                if sub.empty:
                    continue
                score = np.average(sub.values.flatten(), weights=weights)
                all_scores.append({
                    'dataset': dataset,
                    'method': method,
                    'direction': direction,
                    'dimension': dim,
                    'score': score
                })

# ========== DF AND EXCEL ==========
df_scores = pd.DataFrame(all_scores)
excel_out_path = os.path.join(BASE_DIR, 'results','iqr_iqe_scores_by_dim.xlsx')
with pd.ExcelWriter(excel_out_path) as writer:
    for direction in ['reduction', 'expansion']:
        df_dir = df_scores[df_scores['direction'] == direction]
        pivot = df_dir.pivot_table(index=['dataset', 'method'], columns='dimension', values='score')
        pivot.to_excel(writer, sheet_name=direction)

print(f"✅ Saved IQR/IQE scores to Excel at {excel_out_path}")

# ========== PLOTTING ==========
for direction in ['reduction', 'expansion']:
    dims = reduction_dims if direction == 'reduction' else expansion_dims
    df_dir = df_scores[df_scores['direction'] == direction]

    for dataset in datasets:
        df_plot = df_dir[df_dir['dataset'] == dataset]
        if df_plot.empty:
            continue

        plt.figure(figsize=(8, 6))
        for method in df_plot['method'].unique():
            d = df_plot[df_plot['method'] == method]
            d = d[d['dimension'].isin(dims)].sort_values('dimension')
            if d.empty:
                continue
            plt.plot(d['dimension'], d['score'], marker='o', label=method)

        plt.xticks(dims)
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Normalized Score")
        plt.title(f"{'IQR' if direction == 'reduction' else 'IQE'} Trend - {dataset.upper()}")
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{direction}_trend_{dataset}.png"))
        plt.close()

print("📊 Trend plots generated for IQR and IQE.")
