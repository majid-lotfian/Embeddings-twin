# ===============================================================
# 0. Imports & global paths
# ===============================================================
import os, numpy as np, pandas as pd
from glob import glob
from sklearn.preprocessing import minmax_scale             # optional

BASE = os.path.dirname(os.path.abspath(__file__))

METRIC_DIR = {
    'reduction': os.path.join(BASE, 'reduction'),
    'expansion': os.path.join(BASE, 'expansion')
}

WEIGHT_FILE = os.path.join(BASE, 'results', 'optimized_weights_full.xlsx')
OUT_DIR     = os.path.join(BASE, 'results'); os.makedirs(OUT_DIR, exist_ok=True)

#DATASETS    = ['cbc', 'covid19', 'iraq', 'liverpool']
DATASETS    = ['cbc']

FMs         = ['tabnet', 'saint', 'transtab', 'tabtransformer', 'fttransformer']
RED_BASE    = ['pca', 'vae', 'umap']
EXP_BASE    = ['vae', 'polyexpand', 'randomproj']

DIM_MAP = {                      # valid dimensions per dataset & task
    'reduction': {
        'cbc':       [8, 12, 16, 20],
        'covid19':   [8, 12, 16, 20],
        'iraq':      [8, 12, 16, 20],
        'liverpool': [8]                       # only 8D for Liverpool-reduction
    },
    'expansion': {
        'cbc':       [32, 64, 96, 128],
        'covid19':   [32, 64, 96, 128],
        'iraq':      [32, 64, 96, 128],
        'liverpool': [32, 64, 96, 128]
    }
}

METRICS = {
    'reduction': ['trustworthiness', 'knn_preservation',
                  'pairwise_mse', 'shepard_corr'],
    'expansion': ['continuity', 'lid',
                  'neighborhood_hit_rate', 'redundancy_ratio']
}

# ===============================================================
# 1. Load optimised weights (8 rows: 4 datasets × 2 directions)
# ===============================================================
w_df = (pd.read_excel(WEIGHT_FILE)
          .assign(dataset=lambda d: d.dataset.str.lower(),
                  direction=lambda d: d.direction.str.lower()))

def weights_for(ds, direction):
    row = w_df.query("dataset==@ds and direction==@direction").iloc[0]
    return np.array([row[m] for m in METRICS[direction]])

# ===============================================================
# 2. Helper: normalise metric block once per (dataset,direction)
# ===============================================================
def normalise_block(df, metric_cols):
    """
    Min–max normalises each metric column *across the block*.
    If a metric is (almost) constant, return 0.5 everywhere to avoid /0.
    """
    vals  = df[metric_cols].values.astype(float)
    mins  = vals.min(axis=0);  maxs = vals.max(axis=0)
    rng   = np.where((maxs - mins) > 1e-9, maxs - mins, 1.0)
    norm  = (vals - mins) / rng
    # Handle perfectly flat metrics: give them 0.5 so they still contribute
    flat  = (maxs - mins) <= 1e-9
    norm[:, flat] = 0.5
    df[metric_cols] = norm
    return df

# ===============================================================
# 3. Iterate datasets → compute scores & write Excel
# ===============================================================
for direction in ['reduction', 'expansion']:
    metric_cols = METRICS[direction]

    # choose candidate list for this direction
    cand_methods = FMs + (RED_BASE if direction == 'reduction' else EXP_BASE)

    for ds in DATASETS:
        rows_out = []                                         # accumulates dicts
        dims_ok  = DIM_MAP[direction][ds]
        weights  = weights_for(ds, direction)

        # concatenate all metric CSVs into a single DF for normalisation
        slice_frames = []
        for method in cand_methods:
            f = os.path.join(METRIC_DIR[direction],
                             f"metrics_{method}_{ds}.csv")
            if not os.path.exists(f):
                continue
            df = (pd.read_csv(f).rename(columns=str.lower)
                                    .assign(method=method))
            df['dimension'] = df['dimension'].astype(int)

            # special gaps ----------------------------------------------------
            if direction == 'expansion' and method == 'polyexpand':
                df = df[~df['dimension'].isin([96, 128])]
            if method == 'fttransformer':
                df = df[df['dimension'] % 8 == 0]

            df = df[df['dimension'].isin(dims_ok)]
            if not df.empty:
                slice_frames.append(df)

        if not slice_frames:
            continue

        block = pd.concat(slice_frames, ignore_index=True, sort=False)

        # remove metrics not present (shouldn't happen now, but safe)
        keep_cols = ['method', 'dimension'] + \
                    [m for m in metric_cols if m in block.columns]
        block = block[keep_cols]

        # -------- normalise  & score ----------------------------------------
        block = normalise_block(block, metric_cols)

        for _, r in block.iterrows():
            score = (r[metric_cols].values * weights).sum()
            rows_out.append({'method':    r['method'],
                             'dimension': int(r['dimension']),
                             'score':     score})

        # -------- write Excel for this dataset & direction ------------------
        out_df = pd.DataFrame(rows_out)
        out_pivot = (out_df.pivot_table(index='method',
                                        columns='dimension',
                                        values='score')
                             .sort_index())
        out_name = f"{ds}_{direction}_scores.xlsx"
        out_df.to_excel(os.path.join(OUT_DIR, out_name), index=False)

        # also write the pivot view (one sheet) for convenience
        with pd.ExcelWriter(os.path.join(OUT_DIR,
                                         f"{ds}_{direction}_pivot.xlsx")) as wr:
            out_pivot.to_excel(wr, sheet_name='scores')

        print(f"Finished {ds} – {direction}")

print("\n✅ Per-dataset IQR/IQE Excel files saved in ./results/")
