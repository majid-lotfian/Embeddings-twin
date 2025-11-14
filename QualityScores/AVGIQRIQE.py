# ============================================================
# 0. Imports & paths
# ============================================================
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns                          # nicer colours
sns.set_style("whitegrid")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

PIVOT_DIR  = os.path.join(BASE_DIR, "results")   # <-- where *_pivot.xlsx live
OUT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
DATASETS         = ["cbc", "covid19", "iraq", "liverpool"]
METHODS_REDUCTION = ['tabnet','saint','transtab','tabtransformer','fttransformer',
                     'pca','vae','umap']
METHODS_EXPANSION = ['tabnet','saint','transtab','tabtransformer','fttransformer',
                     'vae','polyexpand','randomproj']

CONFIG = {"reduction": {"outfile": "mean_iqr.xlsx",
                        "methods": METHODS_REDUCTION,
                        "title": "Average IQR across dimensions (Reduction)",
                        "figfile": "mean_iqr_barchart.png"},
          "expansion": {"outfile": "mean_iqe.xlsx",
                        "methods": METHODS_EXPANSION,
                        "title": "Average IQE across dimensions (Expansion)",
                        "figfile": "mean_iqe_barchart.png"}}

# ============================================================
# 1. Helper: gather mean scores for a given direction
# ============================================================
def gather_means(direction: str):
    pieces = []
    for ds in DATASETS:
        piv_path = os.path.join(PIVOT_DIR, f"{ds}_{direction}_pivot.xlsx")
        if not os.path.exists(piv_path):
            print(f"⚠️  Missing pivot file for {ds}-{direction}: {piv_path}")
            continue
        pivot = (pd.read_excel(piv_path, index_col=0)
                   .rename_axis("method"))
        # simple arithmetic mean across dimension columns
        pivot["mean_score"] = pivot.mean(axis=1, skipna=True)
        long = (pivot[["mean_score"]]
                .reset_index()
                .assign(dataset=ds)[["dataset","method","mean_score"]])
        pieces.append(long)
    if not pieces:        # nothing collected
        return pd.DataFrame(columns=["dataset","method","mean_score"])
    return pd.concat(pieces, ignore_index=True)

# ============================================================
# 2. Process both directions, save Excel & plot bars
# ============================================================
for direction, cfg in CONFIG.items():
    long_df = gather_means(direction)
    if long_df.empty:
        print(f"❌  No data collected for {direction}. Skipping.")
        continue

    # wide table: rows = dataset, cols = method
    wide_df = (long_df.pivot(index="dataset",
                             columns="method",
                             values="mean_score")
                       .reindex(DATASETS))             # keep row order

    # ensure column order is stable
    wide_df = wide_df[[m for m in cfg["methods"] if m in wide_df.columns]]

    # ---------- Excel ----------
    out_xlsx = os.path.join(OUT_DIR, cfg["outfile"])
    with pd.ExcelWriter(out_xlsx) as xlw:
        wide_df.to_excel(xlw, sheet_name="mean_scores")
        long_df.to_excel(xlw, sheet_name="tidy_long", index=False)
    print(f"✅  Saved mean scores to {out_xlsx}")

    # ---------- Bar-plot ----------
    plt.figure(figsize=(10,5))
    wide_df.plot(kind="bar",
                 ax=plt.gca(),
                 width=0.9,
                 edgecolor='black',
                 linewidth=0.5)
    plt.ylabel("Mean Normalised Score")
    plt.title(cfg["title"])
    plt.xticks(rotation=0)
    plt.legend(title="Method", fontsize='small', ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, cfg["figfile"]), dpi=300)
    plt.close()

print("🎉  Finished: Excel tables + bar-charts generated.")
