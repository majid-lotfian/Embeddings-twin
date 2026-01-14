# This file generates embeddings for CH tabular data
# The table contains these features:
# kind	sex	Year	CH	CH3	pregnancy_dur	birthweight	hp_hour	T4-sd	T4	TSH	TBG	T4TBG	c101	c14	c142	c161	c181oh	c6	tyr	C14:1/C16	c16	c2	c5c2	c5	c5dc	c5oh	c8	c8c10	c10	c141	c141c2	c16oh	phe	sa	val	phetyr	leu	c0

# FT-Transformer multiclass (CH3) embedding pipeline with CV + clustering + plots
# - CH  = binary label (ignored here)
# - CH3 = multiclass label (0=no-CH, 1=CH-T, 2=CH-?)  <-- used
#
# Requirements:
#   pip install torch numpy pandas scikit-learn matplotlib umap-learn hdbscan
#
# Key correctness points:
# - Do NOT stitch embeddings across CV folds into one matrix for clustering/UMAP
#   (fold models are different => embedding spaces are not aligned).
# - Cluster + score PER-FOLD on that fold's validation embeddings, then aggregate.
# - If you want ONE coherent embedding space for plots: train ONE final model and embed all rows.

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

# Optional: UMAP and HDBSCAN
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Data handling
# -----------------------------
@dataclass
class PreprocessArtifacts:
    num_imputer: SimpleImputer
    num_scaler: StandardScaler
    cat_maps: Dict[str, Dict[str, int]]  # mapping per cat column (train-only)
    num_cols: List[str]
    cat_cols: List[str]


def build_cat_maps(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, int]]:
    maps = {}
    for c in cat_cols:
        vals = df[c].astype("object").fillna("__MISSING__").astype(str).unique().tolist()
        vals = sorted(vals)
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def fit_preprocess(
    df_train: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> PreprocessArtifacts:
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()

    Xn = df_train[num_cols].astype(float).values
    Xn_imp = num_imputer.fit_transform(Xn)
    num_scaler.fit(Xn_imp)

    cat_maps = build_cat_maps(df_train, cat_cols)

    return PreprocessArtifacts(
        num_imputer=num_imputer,
        num_scaler=num_scaler,
        cat_maps=cat_maps,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )


def transform_preprocess(
    df: pd.DataFrame,
    art: PreprocessArtifacts,
) -> Tuple[np.ndarray, np.ndarray]:
    # numeric
    Xn = df[art.num_cols].astype(float).values
    Xn = art.num_imputer.transform(Xn)
    Xn = art.num_scaler.transform(Xn)

    # categorical -> int codes (unknown -> unk_id bucket)
    if len(art.cat_cols) == 0:
        Xc = np.zeros((len(df), 0), dtype=np.int64)
    else:
        Xc_list = []
        for c in art.cat_cols:
            col = df[c].astype("object").fillna("__MISSING__").astype(str).values
            mapping = art.cat_maps[c]
            unk_id = len(mapping)  # unknown bucket
            codes = np.array([mapping.get(v, unk_id) for v in col], dtype=np.int64)
            Xc_list.append(codes)
        Xc = np.stack(Xc_list, axis=1)

    return Xn.astype(np.float32), Xc


class TabDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_num[idx]),
            torch.from_numpy(self.X_cat[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# -----------------------------
# FT-Transformer (minimal + practical)
# -----------------------------
class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        repr_dim: int = 32,
        n_classes: int = 3,
    ):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token

        # Numeric tokenizers: one linear per numeric feature (scalar -> d_token)
        self.num_tokenizers = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_num)]) if n_num > 0 else None

        # Categorical tokenizers: embedding per cat col (+1 unknown bucket)
        self.cat_embeds = nn.ModuleList([nn.Embedding(card + 1, d_token) for card in cat_cardinalities])

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)

        hidden = max(d_token, repr_dim)
        self.repr_head = nn.Sequential(
            nn.Linear(d_token, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, repr_dim),
        )
        self.classifier = nn.Linear(repr_dim, n_classes)

        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        B = x_num.size(0)
        tokens = []

        # numeric tokens
        if self.n_num > 0:
            # (B,) -> (B,1) -> (B,d_token) -> (B,1,d_token)
            for i in range(self.n_num):
                ti = self.num_tokenizers[i](x_num[:, i].unsqueeze(-1))
                tokens.append(ti.unsqueeze(1))

        # categorical tokens
        for j in range(self.n_cat):
            tj = self.cat_embeds[j](x_cat[:, j])  # (B,d_token)
            tokens.append(tj.unsqueeze(1))

        x = torch.cat(tokens, dim=1) if len(tokens) > 0 else torch.zeros((B, 0, self.d_token), device=x_num.device)

        cls = self.cls.expand(B, -1, -1)  # (B,1,d_token)
        x = torch.cat([cls, x], dim=1)    # (B,1+T,d_token)

        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0, :]
        repr_vec = self.repr_head(cls_out)
        logits = self.classifier(repr_vec)
        return logits, repr_vec

    @torch.no_grad()
    def embed(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, r = self.forward(x_num, x_cat)
        return r


# -----------------------------
# Training utilities
# -----------------------------
def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    # inverse frequency; safe if some classes absent in a fold
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(
    model: FTTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 25,
) -> Dict:
    model.to(device)
    class_weights = class_weights.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = -1.0
    bad = 0
    epochs_ran = 0

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        model.train()

        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)

            logits, _ = model(xb_num, xb_cat)
            loss = F.cross_entropy(logits, yb, weight=class_weights)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb_num, xb_cat, yb in val_loader:
                xb_num = xb_num.to(device)
                xb_cat = xb_cat.to(device)
                logits, _ = model(xb_num, xb_cat)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(pred)

        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        val_macro_f1 = f1_score(y_true, y_pred, average="macro")

        if val_macro_f1 > best_val + 1e-4:
            best_val = val_macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_macro_f1": float(best_val), "epochs_ran": int(epochs_ran)}


@torch.no_grad()
def extract_embeddings(
    model: FTTransformer,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    model.to(device)
    embs = []

    for i in range(0, len(X_num), batch_size):
        xb_num = torch.from_numpy(X_num[i:i + batch_size]).to(device)
        xb_cat = torch.from_numpy(X_cat[i:i + batch_size]).to(device)
        r = model.embed(xb_num, xb_cat).cpu().numpy()
        embs.append(r)

    return np.vstack(embs)


# -----------------------------
# Clustering + evaluation
# -----------------------------
def _safe_gmm_k(k: int, n: int) -> int:
    # GMM requires 1 <= n_components <= n_samples
    return max(1, min(int(k), int(n)))


def cluster_and_score(
    E: np.ndarray,
    y: np.ndarray,
    method: str = "hdbscan",
    gmm_k: int = 3,
    random_state: int = 42,
) -> Tuple[Dict, np.ndarray]:
    n = len(E)
    if n < 10:
        return {
            "cluster_method": method,
            "n_clusters": 0,
            "noise_frac": None,
            "silhouette": None,
            "davies_bouldin": None,
            "ARI": None,
            "NMI": None,
        }, np.full((n,), -1, dtype=int)

    if method == "hdbscan":
        if not HAVE_HDBSCAN:
            raise RuntimeError("hdbscan not installed. pip install hdbscan")
        # ensure min_cluster_size < n
        min_cs = max(5, min(25, n // 20))
        min_cs = min(min_cs, n - 1)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs)
        c = clusterer.fit_predict(E)  # -1 is noise

    elif method == "gmm":
        k = _safe_gmm_k(gmm_k, n)
        # If k==1, clustering is trivial; metrics still computed but will be degenerate.
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        c = gmm.fit_predict(E)

    else:
        raise ValueError("method must be 'hdbscan' or 'gmm'")

    out = {"cluster_method": method}
    out["n_clusters"] = int(len(set(c)) - (1 if -1 in c else 0))
    out["noise_frac"] = float(np.mean(c == -1)) if -1 in c else 0.0

    valid = (c != -1) if -1 in c else np.ones_like(c, dtype=bool)
    # silhouette/DB need >=2 clusters and enough points
    if np.sum(valid) >= 10 and len(set(c[valid])) >= 2:
        out["silhouette"] = float(silhouette_score(E[valid], c[valid]))
        out["davies_bouldin"] = float(davies_bouldin_score(E[valid], c[valid]))
    else:
        out["silhouette"] = None
        out["davies_bouldin"] = None

    out["ARI"] = float(adjusted_rand_score(y, c))
    out["NMI"] = float(normalized_mutual_info_score(y, c))
    return out, c


# -----------------------------
# Visualization (single embedding space only!)
# -----------------------------
def plot_2d(
    E: np.ndarray,
    y: np.ndarray,
    title: str,
    outpath: Optional[str] = None,
    method: str = "umap",
    seed: int = 42,
):
    n = len(E)
    if n < 5:
        return

    if method == "pca":
        Z = PCA(n_components=2, random_state=seed).fit_transform(E)

    elif method == "umap":
        if not HAVE_UMAP:
            raise RuntimeError("umap-learn not installed. pip install umap-learn")
        n_neighbors = min(15, max(2, n - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        )
        Z = reducer.fit_transform(E)

    elif method == "tsne":
        # perplexity must be < (n-1)/3
        max_perp = max(2, (n - 1) // 3)
        perp = min(25, max_perp)
        Z = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca").fit_transform(E)

    else:
        raise ValueError("method must be 'pca', 'umap', or 'tsne'")

    plt.figure(figsize=(6.5, 5.5))
    for lab in np.unique(y):
        idx = (y == lab)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.8, label=str(lab))
    plt.title(title)
    plt.legend(title="CH3", fontsize=9)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()


# -----------------------------
# Main experiment runner
# -----------------------------
def run_ft_transformer_experiment(
    df: pd.DataFrame,
    out_dir: str = "./ft_experiment_outputs",
    embedding_dims: List[int] = [8, 16, 32, 64, 128],
    n_splits: int = 5,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    make_final_full_model_plots: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    if "CH3" not in df.columns or "CH" not in df.columns:
        raise ValueError("Expected both 'CH' (binary) and 'CH3' (multiclass) columns in df.")

    y = df["CH3"].astype(int).values
    feature_df = df.drop(columns=["CH", "CH3"]).copy()

    # Categorical detection: object columns + force known categorical columns if present
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    for c in ["kind", "sex"]:
        if c in feature_df.columns and c not in cat_cols:
            cat_cols.append(c)

    num_cols = [c for c in feature_df.columns if c not in cat_cols]

    print("Numeric cols:", num_cols)
    print("Categorical cols:", cat_cols)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_results = []

    for d in embedding_dims:
        print(f"\n=== Embedding dim: {d} ===")
        dim_dir = os.path.join(out_dir, f"repr_dim_{d}")
        os.makedirs(dim_dir, exist_ok=True)

        fold_summaries = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(feature_df, y), start=1):
            print(f"\n-- Fold {fold}/{n_splits}")

            df_tr = feature_df.iloc[tr_idx].copy()
            df_va = feature_df.iloc[va_idx].copy()
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            # Fit preprocess on train only
            art = fit_preprocess(df_tr, num_cols, cat_cols)
            Xtr_num, Xtr_cat = transform_preprocess(df_tr, art)
            Xva_num, Xva_cat = transform_preprocess(df_va, art)

            cat_cardinalities = [len(art.cat_maps[c]) for c in cat_cols]

            train_loader = DataLoader(TabDataset(Xtr_num, Xtr_cat, y_tr), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TabDataset(Xva_num, Xva_cat, y_va), batch_size=batch_size, shuffle=False)

            # Modest model for <1000
            d_token = 64 if d >= 32 else 32
            n_heads = 8 if d_token >= 64 else 4
            n_layers = 3
            dropout = 0.15

            model = FTTransformer(
                n_num=len(num_cols),
                cat_cardinalities=cat_cardinalities,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
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
            print("Best val macro-F1:", train_info["best_val_macro_f1"], "epochs:", train_info["epochs_ran"])

            # Supervised val performance (optional)
            model.eval()
            with torch.no_grad():
                logits, _ = model(
                    torch.from_numpy(Xva_num).to(device),
                    torch.from_numpy(Xva_cat).to(device),
                )
                pred = torch.argmax(logits, dim=1).cpu().numpy()

            val_acc = accuracy_score(y_va, pred)
            val_f1 = f1_score(y_va, pred, average="macro")

            # Embeddings for this fold's validation set (coherent space)
            E_va = extract_embeddings(model, Xva_num, Xva_cat, device=device, batch_size=256)

            # Clustering + metrics on THIS fold only
            cluster_reports = []

            if HAVE_HDBSCAN:
                rep, c = cluster_and_score(E_va, y_va, method="hdbscan", random_state=seed)
                cluster_reports.append(rep)
                np.save(os.path.join(dim_dir, f"fold{fold}_val_clusters_hdbscan.npy"), c)

            # GMM: safe k in case a fold has very small n
            rep, c = cluster_and_score(E_va, y_va, method="gmm", gmm_k=3, random_state=seed)
            cluster_reports.append(rep)
            np.save(os.path.join(dim_dir, f"fold{fold}_val_clusters_gmm_k3.npy"), c)

            # Save fold embeddings + labels
            np.save(os.path.join(dim_dir, f"fold{fold}_val_embeddings.npy"), E_va)
            np.save(os.path.join(dim_dir, f"fold{fold}_val_labels_CH3.npy"), y_va)

            fold_summaries.append({
                "repr_dim": d,
                "fold": fold,
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_f1),
                **train_info,
                "clustering": cluster_reports,
                "n_val": int(len(y_va)),
            })

        # Save per-dimension summary
        dim_summary = {
            "repr_dim": d,
            "folds": fold_summaries,
            "seed": seed,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }
        with open(os.path.join(dim_dir, "summary.json"), "w") as f:
            json.dump(dim_summary, f, indent=2)

        all_results.append(dim_summary)

        # Optional: train ONE final model for consistent global visualization
        if make_final_full_model_plots:
            print("Training final model for plots (single coherent embedding space)...")

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr2, va2 = next(sss.split(feature_df, y))

            df_tr2 = feature_df.iloc[tr2].copy()
            df_va2 = feature_df.iloc[va2].copy()
            y_tr2 = y[tr2]
            y_va2 = y[va2]

            art2 = fit_preprocess(df_tr2, num_cols, cat_cols)
            Xtr2_num, Xtr2_cat = transform_preprocess(df_tr2, art2)
            Xva2_num, Xva2_cat = transform_preprocess(df_va2, art2)

            cat_cardinalities2 = [len(art2.cat_maps[c]) for c in cat_cols]

            train_loader2 = DataLoader(TabDataset(Xtr2_num, Xtr2_cat, y_tr2), batch_size=batch_size, shuffle=True)
            val_loader2 = DataLoader(TabDataset(Xva2_num, Xva2_cat, y_va2), batch_size=batch_size, shuffle=False)

            d_token = 64 if d >= 32 else 32
            n_heads = 8 if d_token >= 64 else 4

            model2 = FTTransformer(
                n_num=len(num_cols),
                cat_cardinalities=cat_cardinalities2,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=3,
                dropout=0.15,
                repr_dim=d,
                n_classes=3,
            )
            class_w2 = compute_class_weights(y_tr2, n_classes=3)

            _ = train_one_fold(
                model=model2,
                train_loader=train_loader2,
                val_loader=val_loader2,
                class_weights=class_w2,
                device=device,
                lr=1e-3,
                weight_decay=1e-4,
                max_epochs=300,
                patience=25,
            )

            # Embed all rows using SAME preprocessing + SAME model
            Xall_num, Xall_cat = transform_preprocess(feature_df, art2)
            E_all = extract_embeddings(model2, Xall_num, Xall_cat, device=device, batch_size=256)
            np.save(os.path.join(dim_dir, "final_full_embeddings.npy"), E_all)
            np.save(os.path.join(dim_dir, "labels_CH3.npy"), y)

            # Plots
            plot_2d(E_all, y, title=f"FT-Transformer (repr_dim={d}) PCA colored by CH3",
                    outpath=os.path.join(dim_dir, "final_pca_ch3.png"), method="pca", seed=seed)

            if HAVE_UMAP:
                plot_2d(E_all, y, title=f"FT-Transformer (repr_dim={d}) UMAP colored by CH3",
                        outpath=os.path.join(dim_dir, "final_umap_ch3.png"), method="umap", seed=seed)

            plot_2d(E_all, y, title=f"FT-Transformer (repr_dim={d}) t-SNE colored by CH3",
                    outpath=os.path.join(dim_dir, "final_tsne_ch3.png"), method="tsne", seed=seed)

    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDone. Outputs in:", out_dir)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_csv("your_dataset.csv")  # or read_excel(...)
    # run_ft_transformer_experiment(df, out_dir="./ft_outputs")
    pass
