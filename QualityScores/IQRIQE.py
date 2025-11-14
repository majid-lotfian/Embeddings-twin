import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.preprocessing import minmax_scale

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folders = ['reduction', 'expansion']
datasets = ['cbc', 'covid19', 'iraq', 'liverpool']
methods_fm = ['tabnet', 'saint', 'transtab', 'tabtransformer', 'fttransformer']
methods_reduction = ['pca', 'vae', 'umap']
methods_expansion = ['vae', 'polyexpand', 'randomproj']

# Task-specific valid dimensions
reduction_dims = {
    'cbc': [8, 12, 16, 20],
    'covid19': [8, 12, 16, 20],
    'iraq': [8, 12, 16, 20],
    'liverpool': [8],
}
expansion_dims = {
    'cbc': [32, 64, 96, 128],
    'covid19': [32, 64, 96, 128],
    'iraq': [32, 64, 96, 128],
    'liverpool': [12, 16, 20, 32, 64, 96, 128],
}

# Load optimized weights
weights_path = os.path.join(BASE_DIR, 'results', 'optimized_weights_full.xlsx')
weights_df = pd.read_excel(weights_path)

# ==========================
# LOAD AND COMPUTE IQR/IQE SCORES
# ==========================
df_all = []

for folder in folders:
    filepaths = glob(os.path.join(BASE_DIR, folder, '*.csv'))
    direction = folder
    metrics = list(weights_df.columns[2:])

    for path in filepaths:
        filename = os.path.basename(path).replace('.csv', '')
        _, method, dataset = filename.split('_')

        # Determine valid dims for this direction/dataset
        valid_dims = reduction_dims[dataset] if direction == 'reduction' else expansion_dims[dataset]

        # Load and filter
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df['dimension'] = df['dimension'].astype(int)
        df = df[df['dimension'].isin(valid_dims)]

        # FTTransformer only: use only dims divisible by 8
        if method == 'fttransformer':
            df = df[df['dimension'] % 8 == 0]

        # Polynomial Expansion lacks dims 96, 128
        if method == 'polyexpand':
            if direction == 'expansion':
                df = df[~df['dimension'].isin([96, 128])]

        # Skip empty
        if df.empty:
            continue

        # Normalize values
        norm_df = df.copy()
        for metric in metrics:
            if metric not in df.columns:
                continue
            vals = norm_df[metric].values.astype(float)
            if metric in ['pairwise_mse', 'lid']:
                vals = 1.0 / (vals + 1e-8)
            norm_df[metric] = minmax_scale(vals)

        # Get weights for this (dataset, direction)
        row = weights_df[(weights_df['dataset'] == dataset) & (weights_df['direction'] == direction)]
        if row.empty:
            continue
        weights = row[metrics].values.flatten()

        for _, row in norm_df.iterrows():
            score = sum([row[m] * w for m, w in zip(metrics, weights) if m in row and pd.notna(row[m])])
            df_all.append({
                'dataset': dataset,
                'method': method,
                'dimension': row['dimension'],
                'direction': direction,
                'score': score
            })

# ==========================
# SAVE TO EXCEL
# ==========================
df_scores = pd.DataFrame(df_all)
excel_out = df_scores.pivot_table(
    index=['direction', 'dataset', 'method'],
    columns='dimension',
    values='score',
    aggfunc='mean'
).reset_index()

output_excel_path = os.path.join(BASE_DIR, 'results', 'iqr_iqe_scores_by_dim.xlsx')
excel_out.to_excel(output_excel_path, index=False)
print(f"📄 Dimension-wise IQR/IQE scores saved to: {output_excel_path}")

# ==========================
# PLOT TREND LINES
# ==========================
import seaborn as sns
sns.set(style="whitegrid")

for direction in ['reduction', 'expansion']:
    for dataset in datasets:
        df_subset = df_scores[(df_scores['dataset'] == dataset) & (df_scores['direction'] == direction)]
        if df_subset.empty:
            continue

        plt.figure(figsize=(8, 6))
        for method in df_subset['method'].unique():
            df_plot = df_subset[df_subset['method'] == method].sort_values('dimension')
            plt.plot(df_plot['dimension'], df_plot['score'], marker='o', label=method)

        plt.title(f'IQR/IQE Trends - {dataset.title()} ({direction.title()})')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('IQR / IQE Score')
        plt.xticks(sorted(df_subset['dimension'].unique()))
        plt.ylim(0, 1.05)
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        fig_path = os.path.join(BASE_DIR, f'fig_iq_{direction}_{dataset}.png')
        plt.savefig(fig_path)
        plt.close()

print("✅ IQR/IQE trend plots generated and saved.")
