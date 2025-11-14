import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import minmax_scale
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# ========== CONFIGURATION ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folders = ['reduction', 'expansion']
metrics_map = {
    'reduction': ['trustworthiness', 'knn_preservation', 'pairwise_mse', 'shepard_corr'],
    'expansion': ['continuity', 'lid', 'neighborhood_hit_rate', 'redundancy_ratio']
}
invert_metrics = ['pairwise_mse', 'lid']
epsilon = 1e-8

# ========== LOAD & NORMALIZE METRICS ==========
all_data = []

for direction in folders:
    filepaths = glob(os.path.join(BASE_DIR, direction, 'metrics_*.csv'))
    for path in filepaths:
        filename = os.path.basename(path).replace('.csv', '')
        parts = filename.split('_')
        if len(parts) != 3:
            continue
        _, method, dataset = parts
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df['method'] = method
        df['dataset'] = dataset
        df['direction'] = direction
        df['dimension'] = df['dimension'].astype(int)
        df = df.melt(id_vars=['method', 'dataset', 'direction', 'dimension'],
                     value_vars=metrics_map[direction],
                     var_name='metric', value_name='value')
        df.dropna(subset=['value'], inplace=True)
        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Normalize each metric within each direction
df_norm = df_all.copy()
for direction in folders:
    for metric in metrics_map[direction]:
        mask = (df_norm['direction'] == direction) & (df_norm['metric'] == metric)
        values = df_norm.loc[mask, 'value']
        if metric in invert_metrics:
            values = 1.0 / (values + epsilon)
        df_norm.loc[mask, 'value'] = minmax_scale(values)

# ========== BAYESIAN OPTIMIZATION PER DIRECTION ==========
opt_results = {}

for direction in folders:
    metric_list = metrics_map[direction]
    metric_spaces = [Real(0.0, 1.0, name=m) for m in metric_list]

    def score_func(w, df=df_norm[df_norm['direction'] == direction]):
        weights = np.array(w)
        weights /= weights.sum()  # Normalize weights to sum to 1
        grouped = df.groupby(['method', 'dataset', 'dimension', 'metric'])['value'].mean().unstack()
        scores = np.dot(grouped[metric_list].values, weights)
        return -np.mean(scores)  # Minimize negative mean score

    @use_named_args(metric_spaces)
    def objective(**kwargs):
        w = np.array([kwargs[m] for m in metric_list])
        return score_func(w)

    result = gp_minimize(objective, dimensions=metric_spaces,
                         acq_func='EI', n_calls=30, random_state=42)

    best_weights = result.x
    best_weights = best_weights / np.sum(best_weights)
    opt_results[direction] = dict(zip(metric_list, best_weights))

# ========== SAVE RESULTS ==========
# Save to Excel
df_weights = pd.DataFrame(opt_results).T
excel_path = os.path.join(BASE_DIR, 'optimized_weights.xlsx')
df_weights.to_excel(excel_path)

# Save LaTeX Table
latex_table = df_weights.round(3).to_latex(index=True,
                                           caption='Optimized Metric Weights for IQR and IQE',
                                           label='tab:metric_weights')
with open(os.path.join(BASE_DIR, 'optimized_weights.tex'), 'w') as f:
    f.write(latex_table)

print("✅ Optimization complete. Weights saved as Excel and LaTeX.")
