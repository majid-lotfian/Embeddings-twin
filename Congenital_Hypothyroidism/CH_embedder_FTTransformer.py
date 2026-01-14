# This file generates embeddings for CH tabular data
# The table contains these features:
# kind	sex	Year	CH	CH3	pregnancy_dur	birthweight	hp_hour	T4-sd	T4	TSH	TBG	T4TBG	c101	c14	c142	c161	c181oh	c6	tyr	C14:1/C16	c16	c2	c5c2	c5	c5dc	c5oh	c8	c8c10	c10	c141	c141c2	c16oh	phe	sa	val	phetyr	leu	c0

# ================================================================
# FT-Transformer (final-only) embeddings + clustering + 2D plots
# - Uses CH3 as multiclass label (0/1/2)
# - Ignores CH (binary label)
# - For each embedding dimension d:
#     1) Train ONE model (early-stopping split only)
#     2) Generate ONE embedding matrix E_all (n x d) for all rows
#     3) Cluster in ORIGINAL embedding space (R^d): HDBSCAN (optional) + GMM(k)
#     4) Save 3 plots (PCA / UMAP / t-SNE), colored by CH3
#
# Outputs per dimension d in: out_dir/repr_dim_{d}/
#   - final_full_embeddings.npy
#   - labels_CH3.npy
#   - clusters_hdbscan.npy (if available)
#   - clusters_gmm_k{K}.npy
#   - final_pca_ch3.png
#   - final_umap_ch3.png (if available)
#   - final_tsne_ch3.png
#   - summary.json
#
# Requirements:
#   pip install torch numpy pandas scikit-learn matplotlib
# Optional (recommended):
#   pip install umap-learn hdbscan
# ================================================================

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

import matplotlib.pyplot as plt

# Optional dependencies
try:
    import umap  # umap-learn
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Preprocessing
# -----------------------------
@dataclass
class PreprocessArtifacts:
    num_imputer: SimpleImputer
    num_scaler: StandardScaler
    cat_maps: Dict[str, Dict[str, int]]  # per cat column mapping, fit on train split only
    num_cols: List[str]
    cat_cols: List[str]


def build_cat_maps(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        vals = df[c].astype("object").fillna("__MISSING__").astype(str).unique().tolist()
        vals = sorted(vals)
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def fit_preprocess(df_train: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> PreprocessArtifacts:
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


def transform_preprocess(df: pd.DataFrame, art: PreprocessArtifacts) -> Tuple[np.ndarray, np.ndarray]:
    # numeric
    Xn = df[art.num_cols].astype(float).values
    Xn = art.num_imputer.transform(Xn)
    Xn = art.num_scaler.transform(Xn)

    # categorical to int codes (unknown -> unk_id bucket)
    if len(art.cat_cols) == 0:
        Xc = np.zeros((len(df), 0), dtype=np.int64)
    else:
        cols = []
        for c in art.cat_cols:
            col = df[c].astype("object").fillna("__MISSING__").astype(str).values
            mapping = art.cat_maps[c]
            unk_id = len(mapping)  # unknown bucket ID
            codes = np.array([mapping.get(v, unk_id) for v in col], dtype=np.int64)
            cols.append(codes)
        Xc = np.stack(cols, axis=1)

    return Xn.astype(np.float32), Xc


# -----------------------------
# Dataset
# -----------------------------
class TabDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X_num[idx]),
            torch.from_numpy(self.X_cat[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# -----------------------------
# FT-Transformer (minimal and practical)
# -----------------------------
class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.15,
        repr_dim: int = 32,
        n_classes: int = 3,
    ):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token

        # One scalar -> token projection per numeric feature
        self.num_tokenizers = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_num)]) if n_num > 0 else None

        # Categorical embeddings (+1 for unknown bucket)
        self.cat_embeds = nn.ModuleList([nn.Embedding(card + 1, d_token) for card in cat_cardinalities])

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_num.size(0)
        tokens = []

        # numeric tokens (one per feature)
        if self.n_num > 0:
            for i in range(self.n_num):
                ti = self.num_tokenizers[i](x_num[:, i].unsqueeze(-1))  # (B, d_token)
                tokens.append(ti.unsqueeze(1))  # (B, 1, d_token)

        # categorical tokens (one per feature)
        for j in range(self.n_cat):
            tj = self.cat_embeds[j](x_cat[:, j])  # (B, d_token)
            tokens.append(tj.unsqueeze(1))

        if len(tokens) > 0:
            x = torch.cat(tokens, dim=1)  # (B, T, d_token)
        else:
            x = torch.zeros((B, 0, self.d_token), device=x_num.device)

        cls = self.cls.expand(B, -1, -1)          # (B, 1, d_token)
        x = torch.cat([cls, x], dim=1)            # (B, 1+T, d_token)

        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0, :]                      # (B, d_token)
        repr_vec = self.repr_head(cls_out)        # (B, repr_dim)
        logits = self.classifier(repr_vec)        # (B, n_classes)
        return logits, repr_vec

    @torch.no_grad()
    def embed(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, r = self.forward(x_num, x_cat)
        return r


# -----------------------------
# Training + embedding extraction
# -----------------------------
def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_with_early_stopping(
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

        # Validate (macro-F1 used only for early stopping)
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

        # avoid importing sklearn.metrics.f1_score at top? We already imported metrics except f1_score.
        # We'll compute macro-F1 manually with sklearn if needed, but simplest: import here.
        from sklearn.metrics import f1_score
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

    out = []
    n = len(X_num)
    for i in range(0, n, batch_size):
        xb_num = torch.from_numpy(X_num[i:i + batch_size]).to(device)
        xb_cat = torch.from_numpy(X_cat[i:i + batch_size]).to(device)
        r = model.embed(xb_num, xb_cat).cpu().numpy()
        out.append(r)
    return np.vstack(out)


# -----------------------------
# Clustering in embedding space R^d
# -----------------------------
def cluster_hdbscan(E: np.ndarray) -> np.ndarray:
    if not HAVE_HDBSCAN:
        raise RuntimeError("hdbscan is not installed. Install with: pip install hdbscan")
    n = len(E)
    if n < 2:
        return np.full((n,), -1, dtype=int)
    min_cs = max(5, min(25, n // 20))
    min_cs = min(min_cs, n - 1)  # ensure < n
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs)
    return clusterer.fit_predict(E)  # -1 is noise


def cluster_gmm(E: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    n = len(E)
    k = max(1, min(int(k), int(n)))  # safe
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    return gmm.fit_predict(E)


def clustering_report(E: np.ndarray, y: np.ndarray, c: np.ndarray) -> Dict:
    out: Dict = {}
    out["n_clusters"] = int(len(set(c)) - (1 if -1 in c else 0))
    out["noise_frac"] = float(np.mean(c == -1)) if -1 in c else 0.0

    valid = (c != -1) if -1 in c else np.ones_like(c, dtype=bool)
    if np.sum(valid) >= 10 and len(set(c[valid])) >= 2:
        out["silhouette"] = float(silhouette_score(E[valid], c[valid]))
        out["davies_bouldin"] = float(davies_bouldin_score(E[valid], c[valid]))
    else:
        out["silhouette"] = None
        out["davies_bouldin"] = None

    out["ARI_vs_CH3"] = float(adjusted_rand_score(y, c))
    out["NMI_vs_CH3"] = float(normalized_mutual_info_score(y, c))
    return out


# -----------------------------
# 2D plots (for inspection only)
# -----------------------------
def plot_2d(
    E: np.ndarray,
    y: np.ndarray,
    title: str,
    method: str,
    seed: int,
    outpath: Optional[str] = None,
) -> None:
    n = len(E)
    if n < 5:
        return

    if method == "pca":
        Z = PCA(n_components=2, random_state=seed).fit_transform(E)

    elif method == "umap":
        if not HAVE_UMAP:
            raise RuntimeError("umap-learn is not installed. Install with: pip install umap-learn")
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
        # Perplexity constraint: perplexity < (n-1)/3
        max_perp = max(2, (n - 1) // 3)
        perp = min(25, max_perp)
        Z = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca").fit_transform(E)

    else:
        raise ValueError("method must be one of: {'pca','umap','tsne'}")

    plt.figure(figsize=(6.5, 5.5))
    for lab in np.unique(y):
        idx = (y == lab)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.8, label=str(lab))
    plt.title(title)
    plt.legend(title="CH3", fontsize=9)
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=200)
    plt.show()


# -----------------------------
# Main runner: final-only per dimension
# -----------------------------
def run_ft_final_only_with_clustering(
    df: pd.DataFrame,
    out_dir: str = "./ft_outputs",
    embedding_dims: Sequence[int] = (8, 16, 32, 64, 128),
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    earlystop_val_frac: float = 0.2,
    do_hdbscan: bool = True,
    gmm_ks: Sequence[int] = (3, 4, 5),
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    if "CH" not in df.columns or "CH3" not in df.columns:
        raise ValueError("DataFrame must contain both 'CH' (binary) and 'CH3' (multiclass) columns.")

    # labels
    y = df["CH3"].astype(int).values
    Xdf = df.drop(columns=["CH", "CH3"]).copy()

    # Identify categorical cols: object dtype + force known categoricals if present
    cat_cols = [c for c in Xdf.columns if Xdf[c].dtype == "object"]
    for c in ["kind", "sex"]:
        if c in Xdf.columns and c not in cat_cols:
            cat_cols.append(c)
    num_cols = [c for c in Xdf.columns if c not in cat_cols]

    print("Numeric cols:", num_cols)
    print("Categorical cols:", cat_cols)
    if do_hdbscan and not HAVE_HDBSCAN:
        print("Note: HDBSCAN requested but not installed; it will be skipped.")
    if not HAVE_UMAP:
        print("Note: UMAP not installed; UMAP plots will be skipped.")

    # Early-stopping split (used only to stop training; not for evaluation claims)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=earlystop_val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(Xdf, y))

    df_tr = Xdf.iloc[tr_idx].copy()
    df_va = Xdf.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    # Fit preprocessing on train split only
    art = fit_preprocess(df_tr, num_cols, cat_cols)
    Xtr_num, Xtr_cat = transform_preprocess(df_tr, art)
    Xva_num, Xva_cat = transform_preprocess(df_va, art)

    cat_cardinalities = [len(art.cat_maps[c]) for c in cat_cols]

    train_loader = DataLoader(TabDataset(Xtr_num, Xtr_cat, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabDataset(Xva_num, Xva_cat, y_va), batch_size=batch_size, shuffle=False)

    # Transform ALL rows once with the same preprocessing artifacts
    Xall_num, Xall_cat = transform_preprocess(Xdf, art)

    overall = {
        "seed": seed,
        "device": device,
        "earlystop_val_frac": earlystop_val_frac,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "embedding_dims": list(map(int, embedding_dims)),
        "hdbscan_enabled": bool(do_hdbscan and HAVE_HDBSCAN),
        "gmm_ks": list(map(int, gmm_ks)),
        "dims": [],
    }

    for d in embedding_dims:
        d = int(d)
        print(f"\n=== repr_dim={d} ===")
        dim_dir = os.path.join(out_dir, f"repr_dim_{d}")
        os.makedirs(dim_dir, exist_ok=True)

        # modest capacity for <1000
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

        train_info = train_with_early_stopping(
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
        print("Early-stop best val macro-F1 (stopping only):", train_info["best_val_macro_f1"])

        # Embeddings for ALL rows in original space R^d
        E_all = extract_embeddings(model, Xall_num, Xall_cat, device=device, batch_size=256)

        # Save embeddings + labels
        np.save(os.path.join(dim_dir, "final_full_embeddings.npy"), E_all)
        np.save(os.path.join(dim_dir, "labels_CH3.npy"), y)

        # Clustering in embedding space
        clustering_summaries = []

        if do_hdbscan and HAVE_HDBSCAN:
            c_h = cluster_hdbscan(E_all)
            np.save(os.path.join(dim_dir, "clusters_hdbscan.npy"), c_h)
            rep_h = clustering_report(E_all, y, c_h)
            rep_h["method"] = "hdbscan"
            clustering_summaries.append(rep_h)

        for k in gmm_ks:
            c_g = cluster_gmm(E_all, k=int(k), seed=seed)
            np.save(os.path.join(dim_dir, f"clusters_gmm_k{int(k)}.npy"), c_g)
            rep_g = clustering_report(E_all, y, c_g)
            rep_g["method"] = f"gmm_k{int(k)}"
            clustering_summaries.append(rep_g)

        # 2D plots (visual inspection only)
        plot_2d(
            E_all, y,
            title=f"FT-Transformer (repr_dim={d}) PCA colored by CH3",
            method="pca",
            seed=seed,
            outpath=os.path.join(dim_dir, "final_pca_ch3.png"),
        )

        if HAVE_UMAP:
            plot_2d(
                E_all, y,
                title=f"FT-Transformer (repr_dim={d}) UMAP colored by CH3",
                method="umap",
                seed=seed,
                outpath=os.path.join(dim_dir, "final_umap_ch3.png"),
            )

        plot_2d(
            E_all, y,
            title=f"FT-Transformer (repr_dim={d}) t-SNE colored by CH3",
            method="tsne",
            seed=seed,
            outpath=os.path.join(dim_dir, "final_tsne_ch3.png"),
        )

        # Save summary
        dim_summary = {
            "repr_dim": d,
            "model_params": {"d_token": d_token, "n_heads": n_heads, "n_layers": 3, "dropout": 0.15},
            "train_info": train_info,
            "clustering": clustering_summaries,
        }
        with open(os.path.join(dim_dir, "summary.json"), "w") as f:
            json.dump(dim_summary, f, indent=2)

        overall["dims"].append(dim_summary)

    with open(os.path.join(out_dir, "overall.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\nDone. Outputs in:", out_dir)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example:
    # df = pd.read_csv("your_dataset.csv")      # or: pd.read_excel("your_dataset.xlsx")
    # run_ft_final_only_with_clustering(df, out_dir="./ft_outputs")

    pass
