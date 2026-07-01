import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

AA_ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"

def _build_aa_lut(aa_alphabet: str):
    lut = np.full(256, -1, dtype=np.int16)
    for i, aa in enumerate(aa_alphabet):
        lut[ord(aa)] = i
    return lut

def _encode_seqs_to_idx(seqs, L, lut):
    N = len(seqs)
    if N == 0:
        return np.empty((0, L), dtype=np.int64)
    joined = "".join(seqs).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(N, L)
    idx = lut[arr]
    return idx.astype(np.int64)

def compute_multi_allele_length_pwm_logodds_gpu(
    background_peptides_by_length: dict,
    ref_df: pd.DataFrame,
    df: pd.DataFrame,
    *,
    allele_col: str = "allele",
    seq_col: str = "seq",
    out_col: str = "pwm_logodds",
    peptide_lengths=None, # list[int], default 8..14
    aa_alphabet: str = AA_ALPHABET_20,
    pseudocount_bg: float = 1.0,
    pseudocount_ref: float = 50.0,
    device: str = "cuda",
):
    if peptide_lengths is None:
        peptide_lengths = list(range(8, 15))
    else:
        peptide_lengths = list(peptide_lengths)

    device = device.lower()
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if allele_col not in df.columns or seq_col not in df.columns:
        raise ValueError(f"df must contain columns: {allele_col!r}, {seq_col!r}")
    if allele_col not in ref_df.columns or seq_col not in ref_df.columns:
        raise ValueError(f"ref_df must contain columns: {allele_col!r}, {seq_col!r}")

    torch_device = torch.device(device)

    lut = _build_aa_lut(aa_alphabet)
    K = len(aa_alphabet)
    eps = 1e-12

    P_bg_gpu_by_L = {}
    for L in peptide_lengths:
        bg_list = background_peptides_by_length.get(int(L), None)
        if not bg_list:
            continue

        bg_filtered = [p for p in bg_list if len(p) == int(L)]
        if len(bg_filtered) == 0:
            continue

        bg_idx = _encode_seqs_to_idx(bg_filtered, int(L), lut)
        bg_flat = bg_idx.reshape(-1)
        valid = (bg_flat >= 0) & (bg_flat < K)
        bg_valid = bg_flat[valid]
        if bg_valid.size == 0:
            continue

        bg_counts = np.bincount(bg_valid, minlength=K).astype(np.float64)
        bg_counts += pseudocount_bg * (1.0 / K)
        P_bg = bg_counts / bg_counts.sum() # (K,)
        P_bg_gpu_by_L[int(L)] = torch.from_numpy(P_bg.astype(np.float32)).to(torch_device) # (K,)

    ref_tmp = ref_df[[allele_col, seq_col]].copy()
    ref_tmp[seq_col] = ref_tmp[seq_col].astype(str)
    ref_tmp["_L"] = ref_tmp[seq_col].str.len()

    ref_tmp = ref_tmp[ref_tmp["_L"].isin(peptide_lengths)]
    ref_map = {}
    if len(ref_tmp):
        for (a, L), sub in ref_tmp.groupby([allele_col, "_L"], sort=False):
            ref_map[(a, int(L))] = sub[seq_col].tolist()

    out = df.copy()
    scores = np.full(len(out), np.nan, dtype=np.float32)
    out[out_col] = scores

    seqs_all = out[seq_col].astype(str)
    lengths_all = seqs_all.str.len().to_numpy()
    alleles_all = out[allele_col].to_numpy()

    unique_alleles = pd.unique(alleles_all)

    total = len(unique_alleles) * len(peptide_lengths)

    with tqdm(total=total, desc="Computing PWM log-odds (allelexlen)", leave=True) as pbar:
        for allele in unique_alleles:
            allele_mask = (alleles_all == allele)
            if not allele_mask.any():
                pbar.update(len(peptide_lengths))
                continue

            allele_rows = np.nonzero(allele_mask)[0]
            allele_lengths = lengths_all[allele_rows]

            for L in peptide_lengths:
                L = int(L)

                idx_rows = allele_rows[allele_lengths == L]
                if idx_rows.size == 0:
                    pbar.update(1)
                    continue

                P_bg_gpu = P_bg_gpu_by_L.get(L, None)
                if P_bg_gpu is None:
                    pbar.update(1)
                    continue

                ref_seqs = ref_map.get((allele, L), None)
                if not ref_seqs:
                    pbar.update(1)
                    continue

                test_seqs = seqs_all.iloc[idx_rows].tolist()
                test_idx_np = _encode_seqs_to_idx(test_seqs, L, lut) # (N_test, L)
                test_idx_gpu = torch.from_numpy(test_idx_np).to(torch_device) # int64
                invalid_pos_mask = (test_idx_gpu < 0) | (test_idx_gpu >= K)
                invalid_pep_mask = invalid_pos_mask.any(dim=1) # (N_test,)

                ref_seqs_L = [p for p in ref_seqs if len(p) == L]
                if len(ref_seqs_L) == 0:
                    pbar.update(1)
                    continue

                ref_idx_np = _encode_seqs_to_idx(ref_seqs_L, L, lut)
                ref_idx_gpu = torch.from_numpy(ref_idx_np).to(torch_device)
                N_ref = ref_idx_gpu.shape[0]

                counts_ref = torch.zeros(L, K, device=torch_device, dtype=torch.float32)
                for pos in range(L):
                    aa_pos = ref_idx_gpu[:, pos]
                    mask_valid = (aa_pos >= 0) & (aa_pos < K)
                    aa_valid = aa_pos[mask_valid]
                    if aa_valid.numel() == 0:
                        continue
                    counts_ref[pos] = torch.bincount(aa_valid, minlength=K).float()

                N_pos_counts = counts_ref.sum(dim=1, keepdim=True) # (L, 1)
                counts_ref_reg = counts_ref + pseudocount_ref * P_bg_gpu.unsqueeze(0) # (L, K)
                N_pos_reg = N_pos_counts + pseudocount_ref # (L, 1)
                N_pos_reg = torch.clamp(N_pos_reg, min=1e-12)
                P_ref = counts_ref_reg / N_pos_reg # (L, K)

                log_odds = torch.log((P_ref + eps) / (P_bg_gpu.unsqueeze(0) + eps)) # (L, K)
                log_odds_flat = log_odds.reshape(-1) # (L*K,)

                base = torch.arange(L, device=torch_device, dtype=torch.long) * K # (L,)
                aa_idx_clamped = test_idx_gpu.clamp(min=0, max=K - 1)
                lin_idx = base.unsqueeze(0) + aa_idx_clamped # (N_test, L)
                contrib = log_odds_flat[lin_idx] # (N_test, L)
                contrib[invalid_pos_mask] = 0.0
                scores_gpu = contrib.sum(dim=1) # (N_test,)

                scores_np = scores_gpu.detach().cpu().numpy().astype(np.float32)
                invalid_pep_np = invalid_pep_mask.detach().cpu().numpy()
                scores_np[invalid_pep_np] = np.nan

                out.loc[out.index[idx_rows], out_col] = scores_np
                pbar.update(1)

    return out

AA20_ASSIGN = "ACDEFGHIKLMNPQRSTVWY"

def _build_aa_lut_assign(aa_alphabet: str = AA20_ASSIGN):
    lut = np.full(256, -1, dtype=np.int16)
    for i, aa in enumerate(aa_alphabet):
        lut[ord(aa)] = i
    return lut

def _encode_seqs_to_idx_assign(seqs, L, lut):
    """
    Encode equal-length amino acid strings to integer indices.
    Invalid AAs become -1.
    """
    seqs = list(seqs)
    N = len(seqs)
    if N == 0:
        return np.empty((0, L), dtype=np.int64)

    joined = "".join(seqs).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(N, L)
    idx = lut[arr]
    return idx.astype(np.int64)


def assign_peptides_to_clusters_logodds_gpu(
    ref_df: pd.DataFrame,
    peptides_to_assign,
    *,
    peptide_col: str = "peptide",
    cluster_col: str = "louvain_cluster",
    background_peptides_by_length: dict = None,
    aa_alphabet: str = AA20_ASSIGN,
    pseudocount_bg: float = 1.0,
    pseudocount_ref: float = 50.0,
    device: str = "cuda",
    prob_prefix: str = "cluster_prob_",
    return_logodds: bool = False,
):

    if background_peptides_by_length is None:
        raise ValueError(
            "background_peptides_by_length is required for true log-odds scoring. "
            "Without it, this would only be PWM likelihood-style scoring."
        )

    device = device.lower()
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if peptide_col not in ref_df.columns:
        raise ValueError(f"ref_df must contain peptide_col={peptide_col!r}")
    if cluster_col not in ref_df.columns:
        raise ValueError(f"ref_df must contain cluster_col={cluster_col!r}")

    torch_device = torch.device(device)
    lut = _build_aa_lut_assign(aa_alphabet)
    K = len(aa_alphabet)
    eps = 1e-12

    ref = ref_df[[peptide_col, cluster_col]].copy()
    ref = ref[ref[peptide_col].notna() & ref[cluster_col].notna()].copy()
    ref[peptide_col] = ref[peptide_col].astype(str)

    if len(ref) == 0:
        raise ValueError("No valid reference rows after filtering peptide/cluster non-null.")

    lengths_ref = ref[peptide_col].str.len()
    uniq_lengths = lengths_ref.unique()
    if len(uniq_lengths) != 1:
        raise ValueError(
            f"ref_df must contain peptides of exactly one length for this function. "
            f"Found lengths: {sorted(map(int, uniq_lengths))}"
        )
    L = int(uniq_lengths[0])

    ref = ref[lengths_ref == L].copy()

    bg_list = background_peptides_by_length.get(L, None)
    if not bg_list:
        raise ValueError(f"No background peptides found for peptide length {L}.")

    bg_filtered = [p for p in bg_list if isinstance(p, str) and len(p) == L]
    if len(bg_filtered) == 0:
        raise ValueError(f"Background exists for length {L} but no valid peptides of exact length {L} found.")

    bg_idx = _encode_seqs_to_idx_assign(bg_filtered, L, lut)
    bg_flat = bg_idx.reshape(-1)
    valid_bg = (bg_flat >= 0) & (bg_flat < K)
    bg_valid = bg_flat[valid_bg]
    if bg_valid.size == 0:
        raise ValueError(f"Background peptides for length {L} contain no valid amino acids in the chosen alphabet.")

    bg_counts = np.bincount(bg_valid, minlength=K).astype(np.float64)
    bg_counts += pseudocount_bg * (1.0 / K)
    P_bg = bg_counts / bg_counts.sum()
    P_bg_gpu = torch.from_numpy(P_bg.astype(np.float32)).to(torch_device) # (K,)

    clusters = sorted(pd.unique(ref[cluster_col]))
    if len(clusters) == 0:
        raise ValueError("No clusters found in ref_df after filtering.")

    cluster_to_logodds = {}
    cluster_sizes = {}

    for cl in tqdm(clusters, desc="Building cluster PWMs", leave=True):
        seqs_cl = ref.loc[ref[cluster_col] == cl, peptide_col].tolist()
        seqs_cl = [s for s in seqs_cl if len(s) == L]
        cluster_sizes[cl] = len(seqs_cl)

        if len(seqs_cl) == 0:
            continue

        ref_idx_np = _encode_seqs_to_idx_assign(seqs_cl, L, lut)
        ref_idx_gpu = torch.from_numpy(ref_idx_np).to(torch_device) # (N_ref, L)

        counts_ref = torch.zeros((L, K), device=torch_device, dtype=torch.float32)

        for pos in range(L):
            aa_pos = ref_idx_gpu[:, pos]
            valid = (aa_pos >= 0) & (aa_pos < K)
            aa_valid = aa_pos[valid]
            if aa_valid.numel() == 0:
                continue
            counts_ref[pos] = torch.bincount(aa_valid, minlength=K).float()

        N_pos_counts = counts_ref.sum(dim=1, keepdim=True) # (L, 1)
        counts_ref_reg = counts_ref + pseudocount_ref * P_bg_gpu.unsqueeze(0) # (L, K)
        N_pos_reg = torch.clamp(N_pos_counts + pseudocount_ref, min=1e-12)
        P_ref = counts_ref_reg / N_pos_reg

        log_odds = torch.log((P_ref + eps) / (P_bg_gpu.unsqueeze(0) + eps)) # (L, K)
        cluster_to_logodds[cl] = log_odds

    usable_clusters = [cl for cl in clusters if cl in cluster_to_logodds]
    if len(usable_clusters) == 0:
        raise ValueError("No usable clusters could be modeled.")

    assign_df = pd.DataFrame({peptide_col: pd.Series(peptides_to_assign, dtype="object")})
    assign_df[peptide_col] = assign_df[peptide_col].astype(str)

    out = assign_df.copy()
    out[cluster_col] = np.nan

    for cl in usable_clusters:
        out[f"{prob_prefix}{cl}"] = np.nan
        if return_logodds:
            out[f"logodds_{cl}"] = np.nan

    if len(out) == 0:
        return out

    lengths_assign = out[peptide_col].str.len().to_numpy()
    valid_len_mask = (lengths_assign == L)

    idx_rows = np.nonzero(valid_len_mask)[0]
    if idx_rows.size == 0:
        return out

    test_seqs = out.iloc[idx_rows][peptide_col].tolist()
    test_idx_np = _encode_seqs_to_idx_assign(test_seqs, L, lut)
    test_idx_gpu = torch.from_numpy(test_idx_np).to(torch_device) # (N_test, L)

    invalid_pos_mask = (test_idx_gpu < 0) | (test_idx_gpu >= K)
    invalid_pep_mask = invalid_pos_mask.any(dim=1)

    base = torch.arange(L, device=torch_device, dtype=torch.long) * K # (L,)

    score_tensors = []

    for cl in tqdm(usable_clusters, desc="Scoring peptides vs clusters", leave=True):
        log_odds = cluster_to_logodds[cl] # (L, K)
        log_odds_flat = log_odds.reshape(-1) # (L*K,)

        aa_idx_clamped = test_idx_gpu.clamp(min=0, max=K - 1)
        lin_idx = base.unsqueeze(0) + aa_idx_clamped # (N_test, L)
        contrib = log_odds_flat[lin_idx] # (N_test, L)
        contrib[invalid_pos_mask] = 0.0
        scores_gpu = contrib.sum(dim=1) # (N_test,)
        score_tensors.append(scores_gpu)

    score_mat = torch.stack(score_tensors, dim=1) # (N_test, C)
    score_mat[invalid_pep_mask, :] = float("-inf")

    prob_mat = torch.softmax(score_mat, dim=1) # (N_test, C)

    best_idx = torch.argmax(score_mat, dim=1) # (N_test,)
    best_clusters = np.array([usable_clusters[i] for i in best_idx.detach().cpu().numpy()], dtype=object)

    invalid_pep_np = invalid_pep_mask.detach().cpu().numpy()
    best_clusters[invalid_pep_np] = np.nan

    prob_np = prob_mat.detach().cpu().numpy().astype(np.float32)
    prob_np[invalid_pep_np, :] = np.nan

    out.loc[out.index[idx_rows], cluster_col] = best_clusters

    for j, cl in enumerate(usable_clusters):
        out.loc[out.index[idx_rows], f"{prob_prefix}{cl}"] = prob_np[:, j]

    if return_logodds:
        score_np = score_mat.detach().cpu().numpy().astype(np.float32)
        score_np[invalid_pep_np, :] = np.nan
        for j, cl in enumerate(usable_clusters):
            out.loc[out.index[idx_rows], f"logodds_{cl}"] = score_np[:, j]

    return out