from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

AA_ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"


def _build_aa_lut_resamp(aa_alphabet: str):
    lut = np.full(256, -1, dtype=np.int16)
    for i, aa in enumerate(aa_alphabet):
        lut[ord(aa)] = i
    return lut


def _encode_seqs_to_idx_resamp(seqs, L, lut):
    N = len(seqs)
    if N == 0:
        return np.empty((0, L), dtype=np.int64)
    joined = "".join(seqs).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(N, L)
    idx = lut[arr]
    return idx.astype(np.int64)


@torch.no_grad()
def _pwm_probs_from_seqs_gpu_resamp(
    seqs: list[str],
    L: int,
    lut: np.ndarray,
    K: int,
    *,
    pseudocount: float = 1.0,
    device: torch.device,
    drop_invalid_peptides: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, int, int]:
    """
    Returns:
      P: (L,K) float32 numpy probs
      N_total: number of peptides provided (after length filter)
      N_used: number of peptides used (after invalid filter if drop_invalid_peptides)
    """
    seqs_L = [s for s in seqs if len(s) == L]
    N_total = len(seqs_L)
    if N_total == 0:
        return np.full((L, K), np.nan, dtype=np.float32), 0, 0

    idx_np = _encode_seqs_to_idx_resamp(seqs_L, L, lut)
    idx = torch.from_numpy(idx_np).to(device)

    valid_pos = (idx >= 0) & (idx < K)
    invalid_pep = (~valid_pos).any(dim=1)

    if drop_invalid_peptides:
        keep = ~invalid_pep
        N_used = int(keep.sum().item())
        if N_used == 0:
            return np.full((L, K), np.nan, dtype=np.float32), N_total, 0
        idx = idx[keep]
        valid_pos = valid_pos[keep]
    else:
        N_used = N_total

    N = idx.shape[0]

    pos_ids = torch.arange(L, device=device).view(1, L).expand(N, L)
    aa_flat = idx.reshape(-1)
    pos_flat = pos_ids.reshape(-1)
    valid_flat = valid_pos.reshape(-1)

    aa_v = aa_flat[valid_flat]
    pos_v = pos_flat[valid_flat]

    lin = pos_v * K + aa_v
    counts_flat = torch.bincount(lin, minlength=L * K).to(torch.float32)
    counts = counts_flat.view(L, K)

    denom = counts.sum(dim=1, keepdim=True)

    if pseudocount is not None and pseudocount > 0:
        counts = counts + (pseudocount / K)
        denom = denom + pseudocount

    P = counts / torch.clamp(denom, min=eps)
    return P.detach().cpu().numpy().astype(np.float32), N_total, N_used


def _js_divergence_pwm_resamp(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    eps: float = 1e-12,
    normalize_by_length: bool = True,
) -> float:
    """
    Jensen-Shannon divergence between two PWMs P and Q, both shape (L,K), row-stochastic.
    Returns JS in nats. If normalize_by_length=True, returns mean JS per position.
    """
    if P.shape != Q.shape or P.ndim != 2:
        return np.nan
    if not np.isfinite(P).all() or not np.isfinite(Q).all():
        return np.nan

    P2 = np.clip(P.astype(np.float64, copy=False), eps, 1.0)
    Q2 = np.clip(Q.astype(np.float64, copy=False), eps, 1.0)
    M = 0.5 * (P2 + Q2)

    kl_pm = np.sum(P2 * (np.log(P2) - np.log(M)))
    kl_qm = np.sum(Q2 * (np.log(Q2) - np.log(M)))
    js = 0.5 * (kl_pm + kl_qm)

    if normalize_by_length:
        L = P.shape[0]
        js = js / max(L, 1)

    return float(js)


def _resolve_subsample_size_resamp(
    n: int,
    subsample_size_or_frac: float | int,
) -> int:
    """
    If >= 1, interpret as count.
    If 0 < x < 1, interpret as fraction of n.
    """
    if isinstance(subsample_size_or_frac, (int, np.integer)):
        k = int(subsample_size_or_frac)
    else:
        x = float(subsample_size_or_frac)
        if x <= 0:
            raise ValueError("subsample_size_or_frac must be > 0")
        if x < 1:
            k = int(np.floor(n * x))
        else:
            k = int(round(x))

    return max(k, 1)


def compute_js_divergence_pwms_gpu_resampled(
    dfs: dict[str, pd.DataFrame],
    ref_df_key: str,
    *,
    allele_col: str = "allele",
    seq_col: str = "seq",
    peptide_lengths=None,
    aa_alphabet: str = AA_ALPHABET_20,
    pseudocount: float = 1.0,
    device: str = "cuda",
    drop_invalid_peptides: bool = True,
    min_peptides_per_group: int = 10,
    normalize_js_by_length: bool = True,
    iterations: int = 25,
    subsample_size_or_frac: float | int = 0.05,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if ref_df_key not in dfs:
        raise ValueError(f"ref_df_key {ref_df_key!r} not found in dfs")

    device = device.lower()
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    torch_device = torch.device(device)

    for name, d in dfs.items():
        if allele_col not in d.columns or seq_col not in d.columns:
            raise ValueError(f"dfs[{name!r}] must contain columns {allele_col!r} and {seq_col!r}")

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    lut = _build_aa_lut_resamp(aa_alphabet)
    K = len(aa_alphabet)

    ref_df = dfs[ref_df_key]
    test_dfs = {k: v for k, v in dfs.items() if k != ref_df_key}

    if peptide_lengths is None:
        lens = set(ref_df[seq_col].astype(str).str.len().unique().tolist())
        for d in test_dfs.values():
            lens.update(d[seq_col].astype(str).str.len().unique().tolist())
        peptide_lengths = sorted(int(x) for x in lens)
    else:
        peptide_lengths = list(map(int, peptide_lengths))

    ref_tmp = ref_df[[allele_col, seq_col]].copy()
    ref_tmp[seq_col] = ref_tmp[seq_col].astype(str)
    ref_tmp["_L"] = ref_tmp[seq_col].str.len().astype(np.int16)
    ref_tmp = ref_tmp[ref_tmp["_L"].isin(peptide_lengths)]

    ref_groups = {}
    if len(ref_tmp):
        for (a, L), sub in ref_tmp.groupby([allele_col, "_L"], sort=False):
            ref_groups[(a, int(L))] = sub[seq_col].tolist()

    ref_pwm_full: dict[tuple[str, int], np.ndarray] = {}
    ref_counts_full: dict[tuple[str, int], tuple[int, int]] = {}

    with tqdm(total=len(ref_groups), desc=f"Full ref PWM ({ref_df_key})", leave=True) as pbar:
        for (a, L), seqs in ref_groups.items():
            P, N_total, N_used = _pwm_probs_from_seqs_gpu_resamp(
                seqs,
                L,
                lut,
                K,
                pseudocount=pseudocount,
                device=torch_device,
                drop_invalid_peptides=drop_invalid_peptides,
            )
            ref_counts_full[(a, L)] = (N_total, N_used)
            if N_used >= min_peptides_per_group and np.isfinite(P).all():
                ref_pwm_full[(a, L)] = P
            pbar.update(1)

    keys = sorted(ref_pwm_full.keys(), key=lambda x: (str(x[0]), int(x[1])))
    idx_labels = [f"{a}_{L}" for a, L in keys]

    rng = np.random.default_rng(random_state)
    ref_self_js = np.full(len(keys), np.nan, dtype=np.float32)
    ref_self_sample_counts: dict[tuple[str, int], int] = {}

    total_iters = len(keys) * iterations
    with tqdm(total=total_iters, desc=f"Self-JS baseline ({ref_df_key})", leave=True) as pbar:
        for i, (a, L) in enumerate(keys):
            seqs_full = ref_groups.get((a, L), [])
            n_full = len(seqs_full)

            if n_full < min_peptides_per_group:
                pbar.update(iterations)
                continue

            k = _resolve_subsample_size_resamp(n_full, subsample_size_or_frac)
            if k < min_peptides_per_group or (n_full - k) < min_peptides_per_group:
                pbar.update(iterations)
                continue

            js_vals_group = []
            sample_used_counts = []

            seqs_full_arr = np.asarray(seqs_full, dtype=object)

            for _ in range(iterations):
                sample_idx = rng.choice(n_full, size=k, replace=False)
                mask = np.zeros(n_full, dtype=bool)
                mask[sample_idx] = True

                seqs_test = seqs_full_arr[mask].tolist()
                seqs_ref = seqs_full_arr[~mask].tolist()

                P_ref, _, N_ref_used = _pwm_probs_from_seqs_gpu_resamp(
                    seqs_ref,
                    L,
                    lut,
                    K,
                    pseudocount=pseudocount,
                    device=torch_device,
                    drop_invalid_peptides=drop_invalid_peptides,
                )
                P_test, _, N_test_used = _pwm_probs_from_seqs_gpu_resamp(
                    seqs_test,
                    L,
                    lut,
                    K,
                    pseudocount=pseudocount,
                    device=torch_device,
                    drop_invalid_peptides=drop_invalid_peptides,
                )

                if (
                    N_ref_used < min_peptides_per_group
                    or N_test_used < min_peptides_per_group
                    or not np.isfinite(P_ref).all()
                    or not np.isfinite(P_test).all()
                ):
                    pbar.update(1)
                    continue

                js = _js_divergence_pwm_resamp(
                    P_ref,
                    P_test,
                    normalize_by_length=normalize_js_by_length,
                )
                js_vals_group.append(js)
                sample_used_counts.append(N_test_used)
                pbar.update(1)

            if js_vals_group:
                ref_self_js[i] = np.float32(np.mean(js_vals_group))
                ref_self_sample_counts[(a, L)] = int(round(np.mean(sample_used_counts)))

    js_cols = {ref_df_key: ref_self_js}
    allele_len_counts_by_test: dict[str, dict[tuple[str, int], int]] = {
        ref_df_key: ref_self_sample_counts
    }

    for test_name, test_df in test_dfs.items():
        tmp = test_df[[allele_col, seq_col]].copy()
        tmp[seq_col] = tmp[seq_col].astype(str)
        tmp["_L"] = tmp[seq_col].str.len().astype(np.int16)
        tmp = tmp[tmp["_L"].isin(peptide_lengths)]

        test_groups = {}
        if len(tmp):
            for (a, L), sub in tmp.groupby([allele_col, "_L"], sort=False):
                test_groups[(a, int(L))] = sub[seq_col].tolist()

        length_counts: dict[tuple[str, int], int] = {}
        js_vals = np.full(len(keys), np.nan, dtype=np.float32)

        with tqdm(total=len(keys), desc=f"JS vs full ref: {test_name}", leave=True) as pbar:
            for i, (a, L) in enumerate(keys):
                refP = ref_pwm_full.get((a, L), None)
                if refP is None:
                    pbar.update(1)
                    continue

                seqs = test_groups.get((a, L), None)
                if not seqs:
                    pbar.update(1)
                    continue

                P_test, _, N_used = _pwm_probs_from_seqs_gpu_resamp(
                    seqs,
                    L,
                    lut,
                    K,
                    pseudocount=pseudocount,
                    device=torch_device,
                    drop_invalid_peptides=drop_invalid_peptides,
                )
                length_counts[(a, L)] = int(N_used)

                if N_used < min_peptides_per_group or not np.isfinite(P_test).all():
                    pbar.update(1)
                    continue

                js = _js_divergence_pwm_resamp(
                    refP,
                    P_test,
                    normalize_by_length=normalize_js_by_length,
                )
                js_vals[i] = np.float32(js)
                pbar.update(1)

        js_cols[test_name] = js_vals
        allele_len_counts_by_test[test_name] = length_counts

    # Output 1
    ordered_cols = [ref_df_key] + [k for k in dfs.keys() if k != ref_df_key]
    js_by_allele_len = pd.DataFrame(
        {k: js_cols.get(k, np.full(len(keys), np.nan, dtype=np.float32)) for k in ordered_cols},
        index=pd.Index(idx_labels, name="allele_length"),
    )

    # Output 2 and 3
    alleles = sorted({a for a, _L in keys})
    js_by_allele_unweighted = pd.DataFrame(
        index=pd.Index(alleles, name="allele"),
        columns=ordered_cols,
        dtype=np.float32,
    )
    js_by_allele_weighted = pd.DataFrame(
        index=pd.Index(alleles, name="allele"),
        columns=ordered_cols,
        dtype=np.float32,
    )

    lengths_by_allele: dict[str, list[int]] = {}
    for a, L in keys:
        lengths_by_allele.setdefault(a, []).append(L)

    for test_name in ordered_cols:
        for a in alleles:
            Ls = lengths_by_allele.get(a, [])
            if not Ls:
                continue

            vals = []
            for L in Ls:
                lab = f"{a}_{L}"
                v = js_by_allele_len.loc[lab, test_name]
                if pd.notna(v):
                    vals.append(float(v))

            if vals:
                js_by_allele_unweighted.loc[a, test_name] = np.float32(np.mean(vals))
            else:
                js_by_allele_unweighted.loc[a, test_name] = np.nan

        wcounts = allele_len_counts_by_test.get(test_name, {})
        for a in alleles:
            Ls = lengths_by_allele.get(a, [])
            if not Ls:
                continue

            num = 0.0
            den = 0.0
            for L in Ls:
                lab = f"{a}_{L}"
                v = js_by_allele_len.loc[lab, test_name]
                if pd.isna(v):
                    continue
                w = float(wcounts.get((a, L), 0))
                if w <= 0:
                    continue
                num += w * float(v)
                den += w

            if den > 0:
                js_by_allele_weighted.loc[a, test_name] = np.float32(num / den)
            else:
                js_by_allele_weighted.loc[a, test_name] = np.nan

    return js_by_allele_len, js_by_allele_unweighted, js_by_allele_weighted