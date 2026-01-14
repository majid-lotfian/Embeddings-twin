import numpy as np
import pandas as pd

def cluster_label_composition(
    labels_path: str,
    clusters_path: str,
    label_name: str = "CH3",
    cluster_name: str = "cluster",
    drop_noise: bool = True,   # for HDBSCAN (-1)
) -> pd.DataFrame:
    y = np.load(labels_path)      # shape (N,)
    c = np.load(clusters_path)    # shape (N,)

    if y.shape[0] != c.shape[0]:
        raise ValueError(f"Mismatch: labels N={y.shape[0]} vs clusters N={c.shape[0]}")

    df = pd.DataFrame({cluster_name: c, label_name: y})

    if drop_noise:
        df = df[df[cluster_name] != -1].copy()

    # counts: rows=cluster, cols=label
    counts = pd.crosstab(df[cluster_name], df[label_name])

    # proportions within each cluster: P(label | cluster)
    props = counts.div(counts.sum(axis=1), axis=0)

    # add cluster sizes
    props.insert(0, "n_in_cluster", counts.sum(axis=1))

    # optional: majority label and its fraction (purity-like)
    majority_label = counts.idxmax(axis=1)
    majority_frac = counts.max(axis=1) / counts.sum(axis=1)

    props["majority_label"] = majority_label
    props["majority_frac"] = majority_frac

    # nice ordering: largest clusters first
    props = props.sort_values("n_in_cluster", ascending=False)

    return props

# Example usage:
# labels = "repr_dim_32/labels_CH3.npy"
# clusters = "repr_dim_32/clusters_hdbscan.npy"   # or clusters_gmm_k3.npy
# table = cluster_label_composition(labels, clusters, drop_noise=True)
# print(table.to_string())
