import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import minmax_scale
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# ======================
# CONFIGURATION
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folders = ['reduction', 'expansion']
metrics_map = {
    'reduction': ['trustworthiness', 'knn_preservation', 'pairwise_mse', 'shepard_corr'],
    'expansion': ['continuity', 'lid', 'neighborhood_hit_rate', 'redundancy_ratio']
}
invert_metrics = ['pairwise_mse', 'lid']
epsilon = 1e-8

# ======================
# LOAD AND NORMALIZE METRICS
# ======================
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

# Normalize each metric per (dataset, direction)
df_norm = df_all.copy()
for direction in folders:
    for dataset in df_all['dataset'].unique():
        for metric in metrics_map[direction]:
            mask = (df_norm['direction'] == direction) & \
                   (df_norm['dataset'] == dataset) & \
                   (df_norm['metric'] == metric)
            values = df_norm.loc[mask, 'value']
            if metric in invert_metrics:
                values = 1.0 / (values + epsilon)
            df_norm.loc[mask, 'value'] = minmax_scale(values)

# ======================
# BAYESIAN OPTIMIZATION PER DATASET+TASK
# ======================
opt_weights = []

for direction in folders:
    for dataset in df_norm['dataset'].unique():
        print(f"Optimizing weights for {dataset.upper()} ({direction})")
        metric_list = metrics_map[direction]
        metric_spaces = [Real(0.0, 1.0, name=m) for m in metric_list]

        df_sub = df_norm[(df_norm['direction'] == direction) &
                         (df_norm['dataset'] == dataset)]

        # Aggregated metric scores per (method, dimension)
        pivot_df = df_sub.pivot_table(index=['method', 'dimension'],
                                      columns='metric',
                                      values='value').dropna()

        def score_func(weights):
            weights = np.array(weights)
            weights = weights / (weights.sum() + epsilon)
            scores = pivot_df[metric_list].values @ weights
            return -np.mean(scores)

        @use_named_args(metric_spaces)
        def objective(**kwargs):
            w = np.array([kwargs[m] for m in metric_list])
            return score_func(w)

        result = gp_minimize(objective, dimensions=metric_spaces,
                             acq_func='EI', n_calls=40, random_state=42)
        best_w = np.array(result.x)
        best_w /= best_w.sum() + epsilon

        opt_weights.append({
            'dataset': dataset,
            'direction': direction,
            **{metric_list[i]: round(best_w[i], 5) for i in range(len(metric_list))}
        })

# ======================
# SAVE RESULTS
# ======================
df_opt = pd.DataFrame(opt_weights)
# Ensure both metric groups are included in output
for metric in metrics_map['reduction'] + metrics_map['expansion']:
    if metric not in df_opt.columns:
        df_opt[metric] = np.nan

df_opt = df_opt[['dataset', 'direction'] + metrics_map['reduction'] + metrics_map['expansion']]
df_opt.to_excel(os.path.join(BASE_DIR, 'optimized_weights_full.xlsx'), index=False)

# Also LaTeX version
latex_table = df_opt.round(3).to_latex(index=False,
    caption='Optimized Metric Weights Per Dataset and Task',
    label='tab:opt_weights')
with open(os.path.join(BASE_DIR, 'optimized_weights_full.tex'), 'w') as f:
    f.write(latex_table)

print("✅ Optimization complete: Weights saved.")
