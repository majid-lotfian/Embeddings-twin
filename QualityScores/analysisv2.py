import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import minmax_scale

# ===============================
# CONFIGURATION
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folders = ['reduction', 'expansion']
datasets = ['cbc', 'covid19', 'iraq', 'liverpool']
methods_fm = ['tabnet', 'saint', 'transtab', 'tabtransformer', 'fttransformer']
methods_reduction = ['pca', 'vae', 'umap']
methods_expansion = ['vae', 'polyexpand', 'randomproj']

# Combine all known methods for validation
all_methods = list(set(methods_fm + methods_reduction + methods_expansion))

# ===============================
# METRIC NORMALIZATION RULES
# ===============================
invert_metrics = ['pairwise_mse', 'lid']
all_metrics = {
    'reduction': ['trustworthiness', 'knn_preservation', 'pairwise_mse', 'shepard_corr'],
    'expansion': ['continuity', 'lid', 'neighborhood_hit_rate', 'redundancy_ratio']
}

# ===============================
# LOAD AND NORMALIZE METRICS
# ===============================
all_data = []
epsilon = 1e-8

for folder in folders:
    filepaths = glob(os.path.join(BASE_DIR, folder, '*.csv'))
    for path in filepaths:
        filename = os.path.basename(path).replace('.csv', '')
        parts = filename.split('_')
        if len(parts) != 3:
            continue
        _, method, dataset = parts
        if method not in all_methods or dataset not in datasets:
            continue

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df['method'] = method
        df['dataset'] = dataset
        df['task'] = folder

        # Determine task based on dataset and dimension
        df['dimension'] = df['dimension'].astype(int)
        if dataset == 'liverpool':
            reduction_dims = [8]
        else:
            reduction_dims = [8, 12, 16, 20]

        df['inferred_task'] = df['dimension'].apply(
            lambda d: 'reduction' if d in reduction_dims else 'expansion'
        )

        if method == 'fttransformer':
            df = df[df['dimension'] % 8 == 0]

        value_vars = all_metrics[folder]
        df_melt = df.melt(id_vars=['method', 'dataset', 'dimension', 'inferred_task'],
                          value_vars=value_vars,
                          var_name='metric', value_name='value')
        df_melt = df_melt.dropna(subset=['value'])
        all_data.append(df_melt)

if not all_data:
    raise ValueError("No valid metric files found to process.")

metrics_df = pd.concat(all_data, ignore_index=True)

# ===============================
# NORMALIZATION
# ===============================
norm_df = metrics_df.copy()

for task in folders:
    for metric in all_metrics[task]:
        mask = (norm_df['metric'] == metric) & (norm_df['inferred_task'] == task)
        vals = norm_df.loc[mask, 'value'].copy()
        if metric in invert_metrics:
            vals = 1.0 / (vals + epsilon)
        norm_df.loc[mask, 'value'] = minmax_scale(vals)

# ===============================
# AGGREGATE METRICS
# ===============================
agg_df = norm_df.groupby(['method', 'dataset', 'inferred_task', 'metric'])['value'].mean().reset_index()
global_avg = agg_df.groupby(['method', 'inferred_task', 'metric'])['value'].mean().reset_index()

# ===============================
# HEATMAPS PER TASK AND METRIC
# ===============================
os.makedirs(os.path.join(BASE_DIR, 'figures'), exist_ok=True)

for task in folders:
    task_data = global_avg[global_avg['inferred_task'] == task]
    metrics_to_keep = all_metrics[task]
    task_data = task_data[task_data['metric'].isin(metrics_to_keep)]
    pivot = task_data.pivot(index='method', columns='metric', values='value')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, linewidths=0.5)
    plt.title(f'Normalized Metric Scores - {task.title()}')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'figures', f'heatmap_{task}.png'))
    plt.close()

# ===============================
# LINE PLOTS: Metric vs Dimension
# ===============================
for task in folders:
    for metric in all_metrics[task]:
        plt.figure(figsize=(8, 6))
        for method in norm_df['method'].unique():
            df_plot = norm_df[(norm_df['metric'] == metric) &
                              (norm_df['inferred_task'] == task) &
                              (norm_df['method'] == method)]
            mean_by_dim = df_plot.groupby('dimension')['value'].mean()
            plt.plot(mean_by_dim.index, mean_by_dim.values, label=method, marker='o')
        dims = sorted(norm_df['dimension'].unique())
        plt.xticks(dims, labels=[str(x) for x in dims])
        plt.title(f'{metric.replace("_", " ").title()} vs Dimension ({task.title()})')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Normalized Score')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'figures', f'{task}_{metric}_lineplot.png'))
        plt.close()

# ===============================
# SURFACE PLOTS PER DATASET
# ===============================
from matplotlib.ticker import LinearLocator

for dataset in datasets:
    for method in norm_df['method'].unique():
        df_plot = norm_df[(norm_df['dataset'] == dataset) &
                          (norm_df['method'] == method)]
        if df_plot.empty:
            continue

        dims_list = sorted(df_plot['dimension'].unique())
        metrics_list = sorted(df_plot['metric'].unique())
        if len(dims_list) < 2 or len(metrics_list) < 2:
            print(f"⚠️ Skipping surface plot for {method} on {dataset}: insufficient data.")
            continue

        X, Y = np.meshgrid(dims_list, range(len(metrics_list)))
        Z = np.zeros_like(X, dtype=float)

        for i, dim in enumerate(dims_list):
            for j, metric in enumerate(metrics_list):
                val = df_plot[(df_plot['dimension'] == dim) &
                              (df_plot['metric'] == metric)]['value'].mean()
                Z[j, i] = val if not np.isnan(val) else 0

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none')
        ax.set_title(f'{method.title()} - {dataset.upper()}')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Metric')
        ax.set_zlabel('Score')
        ax.set_xticks(dims_list)
        ax.set_xticklabels([str(d) for d in dims_list])
        ax.set_yticks(range(len(metrics_list)))
        ax.set_yticklabels([m.replace('_', '\n') for m in metrics_list], fontsize=8)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'figures', f'surface_{method}_{dataset}.png'))
        plt.close()

print("\n✅ All analysis completed and visualizations saved under ./figures")
