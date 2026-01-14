# This file generates embeddings for CH tabular data
# The table contains these features:
# kind	sex	Year	CH	CH3	pregnancy_dur	birthweight	hp_hour	T4-sd	T4	TSH	TBG	T4TBG	c101	c14	c142	c161	c181oh	c6	tyr	C14:1/C16	c16	c2	c5c2	c5	c5dc	c5oh	c8	c8c10	c10	c141	c141c2	c16oh	phe	sa	val	phetyr	leu	c0

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

# Optional: HDBSCAN
try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False

# assumes you already have these from your previous script:
# - set_seed
# - fit_preprocess, transform_preprocess
# - TabDataset
# - FTTransformer
# - compute_class_weights
# - train_one_fold
# - extract_embeddings
# - plot_2d (pca/umap/tsne) and HAVE_UMAP variable

def cluster_hdbscan(E: np.ndarray) -> np.ndarray:
    if not HAVE_HDBSCAN:
        raise RuntimeError("hdbscan not installed. pip install hdbscan")
    n = len(E)
    min_cs = max(5, min(25, n // 20))
    min_cs = min(min_cs, n - 1)  # ensure < n
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs)
    return clusterer.fit_predict(E)  # -1 is noise

def cluster_gmm(E: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    n = len(E)
    k = max(1, min(int(k), int(n)))  # safe
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    return gmm.fit_predict(E)

def clustering_report(E: np.ndarray, y: np.ndarray, c: np.ndarray) -> dict:
    out = {}
    out["n_clusters"] = int(len(set(c)) - (1 if -1 in c else 0))
    out["noise_frac"] = float(np.mean(c == -1)) if -1 in c else 0.0

    # Unsupervised metrics computed only on non-noise points and only if >=2 clusters
    valid = (c != -1) if -1 in c else np.ones_like(c, dtype=bool)
    if np.sum(valid) >= 10 and len(set(c[valid])) >= 2:
        out["silhouette"] = float(silhouette_score(E[valid], c[valid]))
        out["davies_bouldin"] = float(davies_bouldin_score(E[valid], c[valid]))
    else:
        out["silhouette"] = None
        out["davies_bouldin"] = None

    # Label agreement (since you have CH3)
    out["ARI_vs_CH3"] = float(adjusted_rand_score(y, c))
    out["NMI_vs_CH3"] = float(normalized_mutual_info_score(y, c))
    return out

def run_ft_final_only_with_clustering(
    df: pd.DataFrame,
    out_dir: str = "./ft_final_only_outputs",
    embedding_dims = [8, 16, 32, 64, 128],
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    earlystop_val_frac: float = 0.2,
    do_hdbscan: bool = True,
    gmm_ks = (3, 4, 5),
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    if "CH3" not in df.columns or "CH" not in df.columns:
        raise ValueError("Expected both 'CH' (binary) and 'CH3' (multiclass) columns in df.")

    y = df["CH3"].astype(int).values
    Xdf = df.drop(columns=["CH", "CH3"]).copy()

    # categorical detection + force known categoricals
    cat_cols = [c for c in Xdf.columns if Xdf[c].dtype == "object"]
    for c in ["kind", "sex"]:
        if c in Xdf.columns and c not in cat_cols:
            cat_cols.append(c)
    num_cols = [c for c in Xdf.columns if c not in cat_cols]

    print("Numeric cols:", num_cols)
    print("Categorical cols:", cat_cols)

    # internal split only for early stopping (not for evaluation claims)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=earlystop_val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(Xdf, y))

    df_tr = Xdf.iloc[tr_idx].copy()
    df_va = Xdf.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    art = fit_preprocess(df_tr, num_cols, cat_cols)
    Xtr_num, Xtr_cat = transform_preprocess(df_tr, art)
    Xva_num, Xva_cat = transform_preprocess(df_va, art)

    cat_cardinalities = [len(art.cat_maps[c]) for c in cat_cols]

    train_loader = torch.utils.data.DataLoader(
        TabDataset(Xtr_num, Xtr_cat, y_tr), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        TabDataset(Xva_num, Xva_cat, y_va), batch_size=batch_size, shuffle=False
    )

    # transform all data once using the SAME preprocessing
    Xall_num, Xall_cat = transform_preprocess(Xdf, art)

    overall = {
        "seed": seed,
        "earlystop_val_frac": earlystop_val_frac,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dims": [],
        "clustering": {
            "hdbscan": do_hdbscan and HAVE_HDBSCAN,
            "gmm_ks": list(gmm_ks),
        }
    }

    for d in embedding_dims:
        print(f"\n=== repr_dim={d} ===")
        dim_dir = os.path.join(out_dir, f"repr_dim_{d}")
        os.makedirs(dim_dir, exist_ok=True)

        d_token = 64 if d >= 32 else 32
        n_heads = 8 if d_token >= 64 else 4

        model = FTTransformer(
            n_num=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=3,
            dropout=0.15,
            repr_dim=d,
            n_classes=3,
        )

        class_w = compute_class_weights(y_tr, n_classes=3)

        train_info = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_w,
            device=device,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=300,
            patience=25,
        )
        print("Early-stop best val macro-F1 (for stopping only):", train_info["best_val_macro_f1"])

        # embeddings in ORIGINAL dimension d
        E_all = extract_embeddings(model, Xall_num, Xall_cat, device=device, batch_size=256)
        np.save(os.path.join(dim_dir, "final_full_embeddings.npy"), E_all)
        np.save(os.path.join(dim_dir, "labels_CH3.npy"), y)

        # --- clustering in embedding space R^d ---
        clustering_summaries = []

        if do_hdbscan:
            if HAVE_HDBSCAN:
                c_h = cluster_hdbscan(E_all)
                np.save(os.path.join(dim_dir, "clusters_hdbscan.npy"), c_h)
                rep_h = clustering_report(E_all, y, c_h)
                rep_h["method"] = "hdbscan"
                clustering_summaries.append(rep_h)
            else:
                print("HDBSCAN requested but not installed; skipping.")

        for k in gmm_ks:
            c_g = cluster_gmm(E_all, k=k, seed=seed)
            np.save(os.path.join(dim_dir, f"clusters_gmm_k{k}.npy"), c_g)
            rep_g = clustering_report(E_all, y, c_g)
            rep_g["method"] = f"gmm_k{k}"
            clustering_summaries.append(rep_g)

        # --- plots (visual inspection only) ---
        plot_2d(E_all, y,
                title=f"FT-Transformer (repr_dim={d}) PCA colored by CH3",
                outpath=os.path.join(dim_dir, "final_pca_ch3.png"),
                method="pca", seed=seed)

        if HAVE_UMAP:
            plot_2d(E_all, y,
                    title=f"FT-Transformer (repr_dim={d}) UMAP colored by CH3",
                    outpath=os.path.join(dim_dir, "final_umap_ch3.png"),
                    method="umap", seed=seed)

        plot_2d(E_all, y,
                title=f"FT-Transformer (repr_dim={d}) t-SNE colored by CH3",
                outpath=os.path.join(dim_dir, "final_tsne_ch3.png"),
                method="tsne", seed=seed)

        # save summary
        dim_summary = {
            "repr_dim": d,
            "train_info": train_info,
            "d_token": d_token,
            "n_heads": n_heads,
            "clustering": clustering_summaries,
        }
        with open(os.path.join(dim_dir, "summary.json"), "w") as f:
            json.dump(dim_summary, f, indent=2)

        overall["dims"].append(dim_summary)

    with open(os.path.join(out_dir, "overall.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\nDone. Outputs in:", out_dir)
