"""
bayesian_opt_weights.py
---------------------------------
Optimise metric-combination weights (IQR for reduction, IQE for expansion)
per (dataset, direction) pair.  Guarantees:

• Uses only valid dimensions for each dataset / task.
• Every metric weight ≥ 0.01 (no zeros).
• Entropy penalty discourages one-hot weight vectors.
• Outputs 8 rows (4 datasets × 2 directions) to Excel + LaTeX.

Folder structure
├── reduction/metrics_<method>_<dataset>.csv
└── expansion/metrics_<method>_<dataset>.csv
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import minmax_scale
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
folders    = ['reduction', 'expansion']

metrics_map = {
    'reduction': ['trustworthiness', 'knn_preservation',
                  'pairwise_mse',  'shepard_corr'],
    'expansion': ['continuity', 'lid',
                  'neighborhood_hit_rate', 'redundancy_ratio']
}

invert_metrics = ['pairwise_mse', 'lid']          # lower-is-better metrics
epsilon = 1e-8

# Dataset-specific valid dimensions for reduction / expansion
dim_masks = {
    'cbc':       {'reduction': [8, 12, 16, 20],
                  'expansion': [32, 64, 96, 128]},
    'covid19':   {'reduction': [8, 12, 16, 20],
                  'expansion': [32, 64, 96, 128]},
    'iraq':      {'reduction': [8, 12, 16, 20],
                  'expansion': [32, 64, 96, 128]},
    'liverpool': {'reduction': [8],
                  'expansion': [12, 16, 20, 32, 64, 96, 128]}
}

# ------------------------------------------------------------------
# LOAD  &  NORMALISE  ALL  METRICS
# ------------------------------------------------------------------
records = []
for direction in folders:
    for path in glob(os.path.join(BASE_DIR, direction, 'metrics_*.csv')):
        name = os.path.basename(path).replace('.csv', '')
        parts = name.split('_')
        if len(parts) != 3:
            continue                         # unexpected filename
        _, method, dataset = parts

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df['method']    = method
        df['dataset']   = dataset
        df['direction'] = direction
        df['dimension'] = df['dimension'].astype(int)
        df = df.melt(id_vars=['method', 'dataset', 'direction', 'dimension'],
                     value_vars=metrics_map[direction],
                     var_name='metric', value_name='value')
        df.dropna(subset=['value'], inplace=True)
        records.append(df)

df_all = pd.concat(records, ignore_index=True)

# Min–max normalisation per (dataset, direction, metric)
df_norm = df_all.copy()
for direction in folders:
    for dataset in df_norm['dataset'].unique():
        for metric in metrics_map[direction]:
            msk = (df_norm['direction'] == direction) & \
                  (df_norm['dataset']   == dataset)   & \
                  (df_norm['metric']    == metric)
            vals = df_norm.loc[msk, 'value']
            if metric in invert_metrics:
                vals = 1.0 / (vals + epsilon)
            df_norm.loc[msk, 'value'] = minmax_scale(vals)

# ------------------------------------------------------------------
#  BAYESIAN  OPTIMISATION
# ------------------------------------------------------------------
opt_rows = []

for direction in folders:
    for dataset in df_norm['dataset'].unique():
        metric_list = metrics_map[direction]

        # restrict to valid dimensions
        allowed_dims = dim_masks[dataset][direction]
        sub = df_norm[(df_norm['direction'] == direction) &
                      (df_norm['dataset']   == dataset)   &
                      (df_norm['dimension'].isin(allowed_dims))]

        # pivot to (method, dimension) × metrics and drop rows w/ NaN
        pivot = (sub.pivot_table(index=['method', 'dimension'],
                                 columns='metric',
                                 values='value')
                    .dropna())                # only keep complete rows

        missing_cols = [m for m in metric_list if m not in pivot.columns]
        if missing_cols:
            raise ValueError(f"{dataset}-{direction}: "
                             f"missing metrics {missing_cols}")

        # strictly-positive search space (≥0.01)
        space = [Real(0.01, 1.0, name=m) for m in metric_list]

        def _objective(w):
            w = np.asarray(w)
            w = w / w.sum()                  # project onto simplex
            mean_score = np.mean(pivot[metric_list].values @ w)

            # entropy penalty to discourage one-hot
            H = -np.sum(w * np.log(w + 1e-12))
            maxH = np.log(len(metric_list))
            penalty = 0.05 * (maxH - H)
            return -(mean_score - penalty)    # minimise negative

        @use_named_args(space)
        def objective(**kwargs):
            return _objective([kwargs[m] for m in metric_list])

        res = gp_minimize(objective,
                          dimensions=space,
                          acq_func='EI',
                          n_calls=40,
                          random_state=42)

        best_w = np.array(res.x)
        best_w /= best_w.sum()

        opt_rows.append({
            'dataset': dataset,
            'direction': direction,
            **{metric_list[i]: round(best_w[i], 5)
               for i in range(len(metric_list))}
        })

# ------------------------------------------------------------------
# SAVE  RESULTS
# ------------------------------------------------------------------
df_opt = pd.DataFrame(opt_rows)

# ensure all metric columns exist for uniform table
for m in metrics_map['reduction'] + metrics_map['expansion']:
    if m not in df_opt.columns:
        df_opt[m] = np.nan

ordered_cols = (['dataset', 'direction'] +
                metrics_map['reduction'] +
                metrics_map['expansion'])
df_opt = df_opt[ordered_cols]

df_opt.to_excel(os.path.join(BASE_DIR,
            'optimized_weights_full.xlsx'), index=False)

with open(os.path.join(BASE_DIR, 'optimized_weights_full.tex'), 'w') as f:
    f.write(df_opt.round(3).to_latex(index=False,
             caption='Optimised Metric Weights per Dataset and Task',
             label='tab:opt_weights'))

print("✅ Weight optimisation completed — results saved.")
