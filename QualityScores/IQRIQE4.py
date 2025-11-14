# ===================================================================
# 0. Imports & paths
# ===================================================================
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
from sklearn.preprocessing import minmax_scale        # single normalisation step

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DIRS       = {'reduction': os.path.join(BASE_DIR, 'reduction'),
              'expansion': os.path.join(BASE_DIR, 'expansion')}
FIG_DIR    = os.path.join(BASE_DIR, 'results'); os.makedirs(FIG_DIR, exist_ok=True)
OUT_XLSX   = os.path.join(BASE_DIR, 'results', 'iqr_iqe_scores.xlsx')
WEIGHTS_XL = os.path.join(BASE_DIR, 'results', 'optimized_weights_full.xlsx')

# -------------------------------------------------------------------
DATASETS   = ['cbc', 'covid19', 'iraq', 'liverpool']
RED_DIMS   = [8, 12, 16, 20]
EXP_DIMS   = [32, 64, 96, 128]

# models
FMs        = ['tabnet', 'saint', 'transtab', 'tabtransformer', 'fttransformer']
RED_BASE   = ['pca', 'vae', 'umap']
EXP_BASE   = ['vae', 'polyexpand', 'randomproj']

# metric families – **order is fixed** and matches weight columns
METRICS = {'reduction': ['trustworthiness', 'knn_preservation',
                         'pairwise_mse', 'shepard_corr'],
           'expansion': ['continuity', 'lid',
                         'neighborhood_hit_rate', 'redundancy_ratio']}

# -------------------------------------------------------------------
# 1. Load optimised weights  (8 rows: 4×2)
# -------------------------------------------------------------------
w_df = (pd.read_excel(WEIGHTS_XL)
          .assign(dataset=lambda d: d.dataset.str.lower(),
                  direction=lambda d: d.direction.str.lower()))

def get_weights(ds, direction):
    row   = w_df.query("dataset==@ds and direction==@direction").iloc[0]
    return np.array([row[m] for m in METRICS[direction]])

# -------------------------------------------------------------------
# 2. Gather per-dimension metric rows
# -------------------------------------------------------------------
records = []
for direction, folder in DIRS.items():
    dims_master = RED_DIMS if direction == "reduction" else EXP_DIMS
    metrics     = METRICS[direction]

    for csv_path in glob(os.path.join(folder, "metrics_*_*.csv")):
        _, method, dataset = os.path.basename(csv_path).replace(".csv", "").split("_")
        if dataset not in DATASETS:              # guard against stray files
            continue

        df = (pd.read_csv(csv_path)
                .rename(columns=str.lower)
                .assign(dimension=lambda d: d.dimension.astype(int)))

        # --- allowed dimensions for this dataset/direction
        if direction == "reduction":
            dims_allowed = [8] if dataset == "liverpool" else RED_DIMS
        else:
            dims_allowed = EXP_DIMS
            # skip 96/128 for polynomial expansion
            if method == "polyexpand":
                dims_allowed = [d for d in dims_allowed if d not in (96, 128)]
        # ft-transformer has only multiples of 8
        if method == "fttransformer":
            dims_allowed = [d for d in dims_allowed if d % 8 == 0]

        df = df[df.dimension.isin(dims_allowed)]
        if df.empty:
            continue

        # keep only the metrics belonging to this direction
        df = df[['dimension'] + [m for m in metrics if m in df.columns]]

        for _, row in df.iterrows():
            rec = {'dataset': dataset, 'method': method,
                   'direction': direction, 'dimension': int(row.dimension)}
            rec.update({m: row[m] for m in metrics})
            records.append(rec)

master = pd.DataFrame(records)

# -------------------------------------------------------------------
# 3. Normalise each metric once per (dataset,direction) & compute score
# -------------------------------------------------------------------
all_scores = []
for (ds, direction), slice_df in master.groupby(['dataset', 'direction']):
    metrics = METRICS[direction]
    vals_matrix = slice_df[metrics].values.astype(float)
    m_min, m_max = vals_matrix.min(axis=0), vals_matrix.max(axis=0)
    norm = (vals_matrix - m_min) / np.where(m_max > m_min, m_max - m_min, 1)  # avoid /0
    slice_df.loc[:, metrics] = norm                                            # write back

    weights = get_weights(ds, direction)
    slice_df['score'] = (slice_df[metrics].values * weights).sum(axis=1)
    all_scores.append(slice_df)

scores_df = pd.concat(all_scores, ignore_index=True)
scores_df.to_excel(OUT_XLSX, index=False)

# -------------------------------------------------------------------
# 4. Plot trend lines  (separate figs for IQR & IQE)
# -------------------------------------------------------------------
for direction in ['reduction', 'expansion']:
    for ds in DATASETS:
        sub = scores_df.query("dataset==@ds and direction==@direction")
        if sub.empty: continue
        plt.figure(figsize=(8, 5))
        for meth, grp in sub.groupby('method'):
            grp = grp.sort_values('dimension')
            if len(grp) < 2:                 # only one available point → skip trendline
                continue
            plt.plot(grp.dimension, grp.score, marker='o', label=meth)

        plt.xlabel("Embedding Dimension")
        plt.ylabel("Score")
        plt.title(f"{'IQR' if direction=='reduction' else 'IQE'} Trend – {ds.upper()}")
        plt.xticks(sorted(sub.dimension.unique()))
        plt.grid(True); plt.legend(fontsize='small', ncol=3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{direction}_trend_{ds}.png"), dpi=300)
        plt.close()

print("✅ Scores written to Excel; IQR/IQE trend plots regenerated.")
