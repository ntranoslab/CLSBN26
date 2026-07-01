from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import igraph as ig
import leidenalg as la
import networkx as nx
import community as community_louvain
from tqdm import tqdm

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}

def _validate_feature_table(feature_table: pd.DataFrame, aa_col: str | None = None) -> pd.DataFrame:
    ft = feature_table.copy()
    if aa_col is not None:
        if aa_col not in ft.columns:
            raise ValueError(f"aa_col='{aa_col}' not found in feature_table columns: {list(ft.columns)}")
        ft[aa_col] = ft[aa_col].astype(str)
        ft = ft.set_index(aa_col)
    else:
        ft.index = ft.index.astype(str)

    missing = [a for a in AA20 if a not in ft.index]
    if missing:
        raise ValueError(f"feature_table missing amino acids: {missing} (need {AA20})")

    num = ft.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("feature_table has no numeric columns.")
    return num.loc[list(AA20)]


def _peptides_to_embedding_torch(
    peptides: pd.Series,
    L: int,
    feature_table: pd.DataFrame | None,
    feature_aa_col: str | None,
    device: torch.device,
    unknown_strategy: str = "zero",
) -> torch.Tensor:
    """
    Returns X [N, D] float32 on device.
    - If feature_table is None: one-hot per position => D=L*20
    - If feature_table provided: concat AA feature vectors per position => D=L*F
    Unknown residues => zeros (or error).
    """
    peps = pd.Series(peptides).astype(str).values
    N = len(peps)

    if feature_table is None:
        D = L * 20
        X = torch.zeros((N, D), dtype=torch.float32, device=device)
        for i, s in enumerate(peps):
            if len(s) != L:
                continue
            base = i
            for p, ch in enumerate(s):
                j = AA_TO_IDX.get(ch, None)
                if j is None:
                    if unknown_strategy == "error":
                        raise ValueError(f"Unknown AA '{ch}' in peptide '{s}'")
                    continue
                X[base, p * 20 + j] = 1.0
        return X

    ft = _validate_feature_table(feature_table, aa_col=feature_aa_col)
    F = ft.shape[1]
    D = L * F
    X = torch.zeros((N, D), dtype=torch.float32, device=device)

    feat_map = {aa: torch.tensor(ft.loc[aa].to_numpy(np.float32), device=device) for aa in AA20}

    for i, s in enumerate(peps):
        if len(s) != L:
            continue
        for p, ch in enumerate(s):
            v = feat_map.get(ch, None)
            if v is None:
                if unknown_strategy == "error":
                    raise ValueError(f"Unknown AA '{ch}' in peptide '{s}'")
                continue
            X[i, p * F : (p + 1) * F] = v
    return X


@torch.no_grad()
def _knn_topk_torch(
    X: torch.Tensor,
    k: int,
    metric: str = "cosine",
    chunk_size: int = 2048,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k neighbors for each row of X using torch (GPU-friendly).
    Returns:
      idx [N,k] int64 (neighbors)
      val [N,k] float32 (similarity for cosine, negative distance for euclidean)
    Notes:
      - O(N^2) compute but chunked; good up to ~50k depending on D/GPU.
      - For cosine: uses dot product on normalized vectors.
      - For euclidean: uses squared distances; returns negative dist as "affinity".
    """
    device = X.device
    N, D = X.shape
    k = int(k)
    if k >= N:
        raise ValueError(f"k must be < N. Got k={k}, N={N}")

    if metric not in ("cosine", "euclidean"):
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    if metric == "cosine":
        Xn = torch.nn.functional.normalize(X, dim=1)
        Y = Xn  # [N,D]
    else:
        # For euclidean, keep X as is and precompute norms
        Y = X
        y_norm = (Y * Y).sum(dim=1)  # [N]

    all_idx = []
    all_val = []

    it = range(0, N, chunk_size)
    if verbose:
        it = tqdm(it, desc=f"kNN ({metric}) chunks", leave=True)

    for s in it:
        e = min(N, s + chunk_size)
        Q = Y[s:e] # [B,D]

        if metric == "cosine":
            # sim = Q @ Y.T
            sim = Q @ Y.T # [B,N]
            # exclude self
            ar = torch.arange(s, e, device=device)
            sim[torch.arange(e - s, device=device), ar] = -1e9
            val, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        else:
            # dist^2 = ||q||^2 + ||y||^2 - 2 q.y
            q_norm = (Q * Q).sum(dim=1)  # [B]
            dot = Q @ Y.T  # [B,N]
            dist2 = q_norm[:, None] + y_norm[None, :] - 2.0 * dot
            ar = torch.arange(s, e, device=device)
            dist2[torch.arange(e - s, device=device), ar] = 1e9
            # pick smallest distances -> largest affinity = -dist2
            neg = -dist2
            val, idx = torch.topk(neg, k=k, dim=1, largest=True, sorted=False)

        all_idx.append(idx.detach().cpu().numpy())
        all_val.append(val.detach().cpu().numpy())

    return np.vstack(all_idx), np.vstack(all_val)


def _build_sparse_graph(
    knn_idx: np.ndarray,
    knn_val: np.ndarray,
    symmetrize: str = "mutual", # "mutual" or "union"
    self_loops: bool = False,
) -> sp.csr_matrix:
    """
    Build sparse adjacency from kNN results.
    - knn_val assumed to be an affinity (larger = closer)
    - symmetrize:
        "union"  : A = A OR A^T
        "mutual" : keep edges i->j only if j->i also exists (intersection)
    """
    N, k = knn_idx.shape
    rows = np.repeat(np.arange(N), k)
    cols = knn_idx.reshape(-1)
    data = knn_val.reshape(-1).astype(np.float32)

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    if not self_loops:
        A.setdiag(0)
        A.eliminate_zeros()

    if symmetrize == "union":
        A = A.maximum(A.T)
    elif symmetrize == "mutual":
        # keep only edges present in both directions
        A = A.minimum(A.T)
    else:
        raise ValueError("symmetrize must be 'union' or 'mutual'")

    A.eliminate_zeros()
    return A

def _louvain_partition(
    A: sp.csr_matrix,
    resolution: float = 1.0,
    random_state: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    N = A.shape[0]

    try:

        if verbose:
            print("[louvain] Using igraph+leidenalg backend")

        coo = A.tocoo()
        g = ig.Graph(n=N, edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=False)
        g.es["weight"] = coo.data.tolist()

        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )
        labels = np.array(part.membership, dtype=np.int64)
        return labels
    except Exception:
        pass

    try:

        if verbose:
            print("[louvain] Using python-louvain (community) backend")

        G = nx.from_scipy_sparse_array(A, edge_attribute="weight")
        part = community_louvain.best_partition(G, weight="weight", resolution=resolution, random_state=random_state)
        labels = np.array([part[i] for i in range(N)], dtype=np.int64)
        return labels
    except Exception as e:
        raise ImportError(
            "No Louvain backend available. Install one of:\n"
            "  - igraph + leidenalg  (recommended)\n"
            "  - networkx + python-louvain (pip install python-louvain)\n"
        ) from e


def _encode_peptides_int(peptides: pd.Series, L: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    peps = pd.Series(peptides).astype(str).values
    N = len(peps)
    X = torch.zeros((N, L), dtype=torch.long, device=device)
    M = torch.ones((N, L), dtype=torch.bool, device=device)
    for i, s in enumerate(peps):
        if len(s) != L:
            M[i, :] = False
            continue
        for p, ch in enumerate(s):
            j = AA_TO_IDX.get(ch, None)
            if j is None:
                M[i, p] = False
                X[i, p] = 0
            else:
                X[i, p] = j
    return X, M

def _pwm_from_XM(X: torch.Tensor, M: torch.Tensor, L: int, alpha: float = 0.5) -> torch.Tensor:
    counts = torch.full((L, 20), float(alpha), device=X.device, dtype=torch.float32)
    for p in range(L):
        valid = M[:, p]
        if not torch.any(valid):
            continue
        aa = X[valid, p]
        counts[p].scatter_add_(0, aa, torch.ones_like(aa, dtype=torch.float32))
    theta = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return theta

def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)

def _js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps=eps) + 0.5 * _kl(q, m, eps=eps)

def _agg(v: torch.Tensor, agg: str) -> float:
    if agg == "sum":
        return float(v.sum().item())
    if agg == "mean":
        return float(v.mean().item())
    raise ValueError("agg must be 'sum' or 'mean'")

@torch.no_grad()
def louvain_cluster_peptides(
    df: pd.DataFrame,
    peptide_col: str = "peptide",
    length: int = 9,
    feature_table: pd.DataFrame | None = None,
    feature_aa_col: str | None = None,
    out_cluster_col: str = "louvain_cluster",
    # kNN graph params
    knn_k: int = 30,
    knn_metric: str = "cosine", # "cosine" or "euclidean"
    knn_chunk_size: int = 2048,
    symmetrize: str = "mutual", # "mutual" or "union"
    # louvain params
    resolution: float = 1.0,
    random_state: int = 0,
    # misc
    device: str | None = None,
    max_points: int | None = None, # optional safety cap
    # small cluster grouping
    small_cluster_min_count: int | None = None, # if set, groups all clusters below into one id
    small_cluster_id: int = -1, # optional JS matrix
    compute_js_matrix: bool = False,
    js_min_cluster_count: int = 50,
    js_alpha: float = 0.5,
    js_agg: str = "sum",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Louvain clustering on peptide embeddings (one-hot or AA feature table).
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = df.copy()
    out[out_cluster_col] = np.nan

    sub = out[out[peptide_col].notna()].copy()
    sub[peptide_col] = sub[peptide_col].astype(str)
    sub = sub[sub[peptide_col].str.len() == length].copy()

    if max_points is not None and len(sub) > max_points:
        sub = sub.sample(max_points, random_state=random_state)
        if verbose:
            print(f"[louvain] Downsampled to max_points={max_points}")

    if len(sub) < (knn_k + 1):
        raise ValueError(f"Not enough peptides (n={len(sub)}) for knn_k={knn_k}")

    if verbose:
        print(f"[louvain] n={len(sub)} peptides | L={length} | metric={knn_metric} | k={knn_k} | device={dev}")

    if verbose:
        print("[louvain] Encoding peptides -> embedding matrix ...")
    X = _peptides_to_embedding_torch(
        sub[peptide_col], L=length,
        feature_table=feature_table, feature_aa_col=feature_aa_col,
        device=dev, unknown_strategy="zero"
    )

    if verbose:
        print("[louvain] Computing kNN ...")
    knn_idx, knn_val = _knn_topk_torch(
        X, k=knn_k, metric=knn_metric, chunk_size=knn_chunk_size, verbose=True
    )

    if verbose:
        print(f"[louvain] Building sparse graph (symmetrize={symmetrize}) ...")
    A = _build_sparse_graph(knn_idx, knn_val, symmetrize=symmetrize, self_loops=False)
    if verbose:
        print(f"[louvain] Graph edges: {A.nnz:,}")

    if verbose:
        print(f"[louvain] Running Louvain (resolution={resolution}) ...")
    labels = _louvain_partition(A, resolution=resolution, random_state=random_state, verbose=verbose)

    if small_cluster_min_count is not None:
        vc = pd.Series(labels).value_counts()
        small = set(vc[vc < small_cluster_min_count].index.tolist())
        if verbose:
            print(f"[louvain] Grouping {len(small)} clusters with n<{small_cluster_min_count} into id={small_cluster_id}")
        labels = np.array([small_cluster_id if int(x) in small else int(x) for x in labels], dtype=np.int64)

    out.loc[sub.index, out_cluster_col] = labels.astype(int)

    js_df = None
    if compute_js_matrix:
        if verbose:
            print(f"[js] Computing PWM-JS matrix (min_cluster_count={js_min_cluster_count}, alpha={js_alpha}, agg={js_agg})")

        tmp = out.loc[sub.index, [peptide_col, out_cluster_col]].copy()
        tmp[out_cluster_col] = tmp[out_cluster_col].astype(int)

        counts = tmp[out_cluster_col].value_counts()
        keep = counts[counts >= js_min_cluster_count].index.tolist()
        keep = sorted([int(k) for k in keep])

        if len(keep) < 2:
            if verbose:
                print("[js] Fewer than 2 clusters meet min_cluster_count; returning js_df=None")
            return out, None

        pwms = {}
        for k in tqdm(keep, desc="[js] cluster PWMs", leave=True):
            peps_k = tmp.loc[tmp[out_cluster_col] == k, peptide_col]
            Xk, Mk = _encode_peptides_int(peps_k, length, dev)
            pwms[k] = _pwm_from_XM(Xk, Mk, length, alpha=js_alpha)

        K = len(keep)
        M = np.zeros((K, K), dtype=np.float32)

        for i in tqdm(range(K), desc="[js] pairwise", leave=True):
            pi = pwms[keep[i]]
            for j in range(i + 1, K):
                pj = pwms[keep[j]]
                d = _agg(_js(pi, pj), js_agg)
                M[i, j] = d
                M[j, i] = d

        js_df = pd.DataFrame(M, index=keep, columns=keep)

    return out, js_df

@torch.no_grad()
def compute_louvain_js_matrix(
    df: pd.DataFrame,
    *,
    peptide_col: str = "peptide",
    cluster_col: str = "louvain_cluster",
    length: int = 9,
    js_min_cluster_count: int = 50,
    js_alpha: float = 0.5,
    js_agg: str = "sum",
    device: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """
    Compute PWM-JS distance matrix between clusters in `cluster_col`.
    """
    if js_agg not in ("sum", "mean"):
        raise ValueError("js_agg must be 'sum' or 'mean'")

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp = df[[peptide_col, cluster_col]].dropna().copy()
    tmp[peptide_col] = tmp[peptide_col].astype(str)
    tmp = tmp[tmp[peptide_col].str.len() == int(length)].copy()
    tmp[cluster_col] = tmp[cluster_col].astype(int)

    if len(tmp) == 0:
        return None

    counts = tmp[cluster_col].value_counts()
    keep = counts[counts >= int(js_min_cluster_count)].index.tolist()
    keep = sorted([int(k) for k in keep])

    if len(keep) < 2:
        if verbose:
            print("[js] Fewer than 2 clusters meet min_cluster_count; returning None")
        return None

    if verbose:
        print(f"[js] Computing PWM-JS matrix for {len(keep)} clusters "
              f"(min_cluster_count={js_min_cluster_count}, alpha={js_alpha}, agg={js_agg})")

    pwms = {}
    for k in tqdm(keep, desc="[js] cluster PWMs", leave=True, disable=not verbose):
        peps_k = tmp.loc[tmp[cluster_col] == k, peptide_col]
        Xk, Mk = _encode_peptides_int(peps_k, int(length), dev)
        pwms[k] = _pwm_from_XM(Xk, Mk, int(length), alpha=float(js_alpha))

    K = len(keep)
    M = np.zeros((K, K), dtype=np.float32)

    for i in tqdm(range(K), desc="[js] pairwise", leave=True, disable=not verbose):
        pi = pwms[keep[i]]
        for j in range(i + 1, K):
            pj = pwms[keep[j]]
            d = _agg(_js(pi, pj), js_agg)
            M[i, j] = d
            M[j, i] = d

    return pd.DataFrame(M, index=keep, columns=keep)


def js_mean_of_nearest_neighbors_multi_w_to_rest(
    js_mat: pd.DataFrame,
    neighbors: list[int] | tuple[int, ...] = (1, 3, 5),
    cluster_col: str = "cluster_id",
    out_prefix: str = "js_mean_nn",
    ignore_ids: set[int] = None,
    include_self: bool = False,
    to_rest: bool = False,
) -> pd.DataFrame:
    """
    For each cluster i, compute mean JS distance to its k nearest neighbors for each k in `neighbors`.
    """
    if not isinstance(js_mat, pd.DataFrame):
        raise ValueError("js_mat must be a pandas DataFrame")

    if neighbors is None:
        raise ValueError("neighbors must be a non-empty list/tuple of positive ints")
    ks = [int(k) for k in neighbors]
    if len(ks) == 0 or any(k <= 0 for k in ks):
        raise ValueError(f"neighbors must contain only positive ints; got {neighbors}")
    ks = sorted(set(ks))

    ids = list(js_mat.index)
    if list(js_mat.columns) != ids:
        js_mat = js_mat.loc[ids, ids]

    ignore_ids = set() if ignore_ids is None else set(int(x) for x in ignore_ids)

    rows = []
    for i in ids:
        i = int(i)
        if i in ignore_ids:
            continue

        row = js_mat.loc[i].copy()
        if ignore_ids:
            row = row.drop(labels=[x for x in row.index if int(x) in ignore_ids], errors="ignore")

        if not include_self:
            row = row.drop(labels=[i], errors="ignore")

        row = row.replace([np.inf, -np.inf], np.nan).dropna()
        out = {cluster_col: i}

        if row.empty:
            for k in ks:
                out[f"{out_prefix}_{k}"] = np.nan
                out[f"nn_used_{k}"] = 0

            if to_rest:
                out["dist_to_rest"] = np.nan
                out["nn_used_rest"] = 0

            rows.append(out)
            continue

        sorted_vals = row.sort_values(ascending=True).to_numpy(dtype=float)
        n_avail = sorted_vals.size

        cumsum = np.cumsum(sorted_vals, dtype=float)
        for k in ks:
            kk = min(k, n_avail)
            out[f"{out_prefix}_{k}"] = float(cumsum[kk - 1] / kk)
            out[f"nn_used_{k}"] = int(kk)

        if to_rest:
            out["dist_to_rest"] = float(cumsum[n_avail - 1] / n_avail)
            out["nn_used_rest"] = int(n_avail)

        rows.append(out)

    return pd.DataFrame(rows).reset_index(drop=True)

def minmax_normalize_js(
    df: pd.DataFrame,
    js_col: str,
    out_col: str = None,
    clip: bool = True,
) -> pd.DataFrame:
    """
    Min-max normalize a JS column to [0, 1] based on min/max in the table.
    """
    if js_col not in df.columns:
        raise KeyError(f"Column '{js_col}' not found in dataframe.")

    d = df.copy()
    x = d[js_col].astype(float)

    x_min = np.nanmin(x.values)
    x_max = np.nanmax(x.values)

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("JS column contains no finite values.")

    if x_max == x_min:
        norm = np.zeros_like(x, dtype=float)
    else:
        norm = (x - x_min) / (x_max - x_min)

    if clip:
        norm = np.clip(norm, 0.0, 1.0)

    if out_col is None:
        d[js_col] = norm
    else:
        d[out_col] = norm

    return d