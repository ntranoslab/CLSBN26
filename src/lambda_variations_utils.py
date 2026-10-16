import numpy as np


def build_ablation_model_groups(
    method_prefixes,
    step_budgets,
    learning_rates,
    seeds,
    ckpt_to_save_per_budget,
    learning_rate_strings=None,
    lr_string_fn=None,
):
    """
    Construct:
        final_model_name -> [checkpoint score columns]

    Expected naming:
        {prefix}steps_{steps}_lr_{lr_string}_seed_{seed}
        {base_name}_step_{checkpoint}

    Parameters
    ----------
    method_prefixes : str, list[str], or dict
        Examples:
            "holec_ft_"

        or:
            ["holec_ft_", "holec_pepcl_r_dr_"]

        or:
            {
                "ft": "holec_ft_",
                "pepcl_r_dr": "holec_pepcl_r_dr_",
            }

    step_budgets : list[int]
        Example:
            [25, 50, 100, 200, 400, 800]

    learning_rates : list[float]
        Example:
            [5e-5, 2e-4, 8e-4]

    seeds : list[int]
        Example:
            [0, 1, 2]

    ckpt_to_save_per_budget : dict[int, list[int]]
        Exact checkpoint schedule used during training.

    learning_rate_strings : list[str] or None
        Best option if already generated with your existing lr_to_strings().
        This guarantees exact agreement with the score-column names.

    lr_string_fn : callable or None
        Alternative function converting learning_rates -> strings.

    Returns
    -------
    dict
        {
            final_model_name: [
                checkpoint_column_1,
                checkpoint_column_2,
                ...
            ],
            ...
        }
    """

    if isinstance(method_prefixes, str):
        prefixes = [method_prefixes]
    elif isinstance(method_prefixes, dict):
        prefixes = list(method_prefixes.values())
    else:
        prefixes = list(method_prefixes)

    if learning_rate_strings is None:
        if lr_string_fn is not None:
            learning_rate_strings = lr_string_fn(learning_rates)
        else:
            learning_rate_strings = [str(x) for x in learning_rates]

    if len(learning_rates) != len(learning_rate_strings):
        raise ValueError(
            "learning_rates and learning_rate_strings must have the same length."
        )

    model_groups = {}

    for prefix in prefixes:
        for seed in seeds:
            for steps in step_budgets:
                if steps not in ckpt_to_save_per_budget:
                    raise KeyError(
                        f"No checkpoint schedule found for step budget {steps}."
                    )

                for lr_string in learning_rate_strings:
                    base_name = (
                        f"{prefix}"
                        f"steps_{steps}_"
                        f"lr_{lr_string}_"
                        f"seed_{seed}"
                    )

                    checkpoint_cols = [
                        f"{base_name}_step_{checkpoint_step}"
                        for checkpoint_step
                        in ckpt_to_save_per_budget[steps]
                    ]

                    model_groups[base_name] = checkpoint_cols

    return model_groups
import os
import glob
import pandas as pd
from tqdm import tqdm

def load_and_merge_dir_dfs(directories=None, key_cols=("seq", "allele", "label"), how="outer", df_names=None, path_lists=None):
    key_cols = list(key_cols)
    wanted = set(df_names) if df_names is not None else None
    store = {}

    if directories is not None:
        for di, directory in enumerate(directories):
            csvs = sorted(glob.glob(os.path.join(directory, "*.csv")))
            if wanted is not None:
                csvs = [p for p in csvs if os.path.splitext(os.path.basename(p))[0] in wanted]
            print(f"[{directory}] found {len(csvs)} df(s)" + (f" (filtered to {len(wanted)} requested)" if wanted is not None else ""))
            for path in tqdm(csvs, desc=f"dir {di}: {os.path.basename(os.path.normpath(directory))}", leave=True):
                name = os.path.splitext(os.path.basename(path))[0]
                df = pd.read_csv(path)
                if di == 0 or name not in store:
                    store[name] = df
                else:
                    existing = store[name]
                    new_cols = [c for c in df.columns if c not in existing.columns or c in key_cols]
                    incoming = df[new_cols]
                    missing_keys = [k for k in key_cols if k not in existing.columns or k not in incoming.columns]
                    if missing_keys:
                        raise KeyError(f"{name}: missing key cols {missing_keys} for merge")
                    store[name] = existing.merge(incoming, on=key_cols, how=how)

    if path_lists is not None:
        for name, paths in tqdm(path_lists.items(), desc="path-list merges", leave=True):
            merged = None
            for path in paths:
                df = pd.read_csv(path)
                if merged is None:
                    merged = df
                else:
                    new_cols = [c for c in df.columns if c not in merged.columns or c in key_cols]
                    incoming = df[new_cols]
                    missing_keys = [k for k in key_cols if k not in merged.columns or k not in incoming.columns]
                    if missing_keys:
                        raise KeyError(f"{name}: missing key cols {missing_keys} for merge")
                    merged = merged.merge(incoming, on=key_cols, how="inner")
            store[name] = merged
            print(f"[path_list {name}] merged {len(paths)} df(s) -> {len(merged)} overlapping rows")

    print(f"total unique dfs in store: {len(store)}")
    return store
def filter_by_allele_len_pairs(df1, df2, allele_col="allele", seq_col="seq", filter_both=False):
    df1 = df1.copy()
    df2 = df2.copy()
    for d in (df1, df2):
        d["peptide"] = d[seq_col].str.replace("$", "")
        d["seq_len"] = d["peptide"].apply(len)

    key1 = set(df1[allele_col].astype(str) + "_" + df1["seq_len"].astype(str))
    key2 = set(df2[allele_col].astype(str) + "_" + df2["seq_len"].astype(str))

    k1 = df1[allele_col].astype(str) + "_" + df1["seq_len"].astype(str)
    if filter_both:
        common = key1 & key2
        k2 = df2[allele_col].astype(str) + "_" + df2["seq_len"].astype(str)
        return df1[k1.isin(common)].copy(), df2[k2.isin(common)].copy()
    else:
        return df1[k1.isin(key2)].copy(), df2

def lambda_to_string(lam):
    lam=float(lam)
    if lam==0: return "0"
    mantissa,exponent=f"{lam:.12e}".split("e")
    mantissa=mantissa.rstrip("0").rstrip(".")
    exponent=int(exponent)
    return f"{mantissa}eN{abs(exponent)}" if exponent<0 else f"{mantissa}e{exponent}"

def lambdas_to_strings(lambdas):
    if np.isscalar(lambdas): return lambda_to_string(lambdas)
    return [lambda_to_string(x) for x in lambdas]
def keep_present_cols(df, score_cols):
    return [c for c in score_cols if c in df.columns]
from sklearn.metrics import average_precision_score

def mixed_df_per_allele_ap_pos_focused(df1, df2, score_cols, allele_pos_counts, label_col1='label', label_col2='label', allele_col='allele', neg_ratio=99, n_iterations=10, seed=42):
    d1 = df1.rename(columns={label_col1: '_label'}).assign(_source=1).reset_index(drop=True)
    d2 = df2.rename(columns={label_col2: '_label'}).assign(_source=2).reset_index(drop=True)
    d2.index = d2.index + len(d1)

    all_data = pd.concat([d1, d2])
    score_arrays = {col: all_data[col].values for col in score_cols}
    label_array = all_data['_label'].values
    allele_array = all_data[allele_col].values
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_data.index)}

    def build_index(d, has_neg=True):
        groups = {}
        for allele, grp in d.groupby(allele_col):
            groups[allele] = {'pos': grp.index[grp['_label'] == 1].values}
            if has_neg:
                groups[allele]['neg'] = grp.index[grp['_label'] == 0].values
        return groups

    g1 = build_index(d1, has_neg=True)
    g2 = build_index(d2, has_neg=False)
    all_alleles = set(allele_pos_counts.keys())

    for allele in sorted(all_alleles):
        n1 = len(g1.get(allele, {}).get('pos', []))
        n2 = len(g2.get(allele, {}).get('pos', []))
        requested = allele_pos_counts[allele]
        if n1 + n2 < requested:
            print(f"[WARNING] Allele {allele} has only {n1+n2} positives (df1={n1}, df2={n2}), below requested {requested}")

    per_allele_ap = {col: {a: [] for a in all_alleles} for col in score_cols}
    rng = np.random.default_rng(seed)

    for _ in tqdm(range(n_iterations), desc="Iterations"):
        iter_indices = []
        for allele in all_alleles:
            pos1_idx = g1.get(allele, {}).get('pos', np.array([]))
            neg1_idx = g1.get(allele, {}).get('neg', np.array([]))
            pos2_idx = g2.get(allele, {}).get('pos', np.array([]))
            requested = allele_pos_counts[allele]
            total_pos_avail = len(pos1_idx) + len(pos2_idx)
            if total_pos_avail == 0:
                continue
            n_pos = min(requested, total_pos_avail)
            frac1_pos = rng.uniform(0, 1)
            n_pos1 = int(round(frac1_pos * n_pos)); n_pos2 = n_pos - n_pos1
            if n_pos1 > len(pos1_idx):
                n_pos1 = len(pos1_idx); n_pos2 = min(n_pos - n_pos1, len(pos2_idx))
            elif n_pos2 > len(pos2_idx):
                n_pos2 = len(pos2_idx); n_pos1 = min(n_pos - n_pos2, len(pos1_idx))
            s_pos1 = rng.choice(pos1_idx, n_pos1, replace=False) if n_pos1 > 0 else np.array([])
            s_pos2 = rng.choice(pos2_idx, n_pos2, replace=False) if n_pos2 > 0 else np.array([])
            actual_pos = n_pos1 + n_pos2
            n_neg = actual_pos * neg_ratio
            n_neg1 = min(n_neg, len(neg1_idx))
            s_neg1 = rng.choice(neg1_idx, int(n_neg1), replace=False) if n_neg1 > 0 else np.array([])
            iter_indices.append(np.concatenate([s_pos1, s_pos2, s_neg1]))

        all_iter_idx = np.concatenate(iter_indices)
        pos_in_array = np.array([idx_to_pos[i] for i in all_iter_idx])
        iter_labels = label_array[pos_in_array]
        iter_alleles = allele_array[pos_in_array]
        unique_alleles = np.unique(iter_alleles)

        for col in score_cols:
            iter_scores = score_arrays[col][pos_in_array]
            for allele in unique_alleles:
                mask = iter_alleles == allele
                a_labels = iter_labels[mask]; a_scores = iter_scores[mask]
                valid = ~np.isnan(a_scores)
                a_labels = a_labels[valid]; a_scores = a_scores[valid]
                if len(np.unique(a_labels)) < 2:
                    continue
                per_allele_ap[col][allele].append(average_precision_score(a_labels, a_scores))

    alleles_sorted = sorted(all_alleles)
    matrix = pd.DataFrame(index=alleles_sorted, columns=score_cols, dtype=float)
    for col in score_cols:
        for allele in alleles_sorted:
            vals = per_allele_ap[col][allele]
            matrix.loc[allele, col] = np.mean(vals) if vals else np.nan
    matrix.index.name = allele_col
    return matrix


def mixed_df_per_allele_ap_both(df1, df2, score_cols, allele_pos_counts, label_col1='label', label_col2='label', allele_col='allele', neg_ratio=99, n_iterations=10, seed=42):
    d1 = df1.rename(columns={label_col1: '_label'}).assign(_source=1).reset_index(drop=True)
    d2 = df2.rename(columns={label_col2: '_label'}).assign(_source=2).reset_index(drop=True)
    d2.index = d2.index + len(d1)

    all_data = pd.concat([d1, d2])
    score_arrays = {col: all_data[col].values for col in score_cols}
    label_array = all_data['_label'].values
    allele_array = all_data[allele_col].values
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_data.index)}

    def build_index(d):
        groups = {}
        for allele, grp in d.groupby(allele_col):
            groups[allele] = {'pos': grp.index[grp['_label'] == 1].values, 'neg': grp.index[grp['_label'] == 0].values}
        return groups

    g1 = build_index(d1)
    g2 = build_index(d2)
    all_alleles = set(allele_pos_counts.keys())

    for allele in sorted(all_alleles):
        n1 = len(g1.get(allele, {}).get('pos', []))
        n2 = len(g2.get(allele, {}).get('pos', []))
        requested = allele_pos_counts[allele]
        if n1 + n2 < requested:
            print(f"[WARNING] Allele {allele} has only {n1+n2} positives (df1={n1}, df2={n2}), below requested {requested}")

    per_allele_ap = {col: {a: [] for a in all_alleles} for col in score_cols}
    rng = np.random.default_rng(seed)

    for _ in tqdm(range(n_iterations), desc="Iterations"):
        iter_indices = []
        for allele in all_alleles:
            pos1_idx = g1.get(allele, {}).get('pos', np.array([]))
            neg1_idx = g1.get(allele, {}).get('neg', np.array([]))
            pos2_idx = g2.get(allele, {}).get('pos', np.array([]))
            neg2_idx = g2.get(allele, {}).get('neg', np.array([]))
            requested = allele_pos_counts[allele]
            total_pos_avail = len(pos1_idx) + len(pos2_idx)
            if total_pos_avail == 0:
                continue
            n_pos = min(requested, total_pos_avail)
            frac1_pos = rng.uniform(0, 1)
            n_pos1 = int(round(frac1_pos * n_pos)); n_pos2 = n_pos - n_pos1
            if n_pos1 > len(pos1_idx):
                n_pos1 = len(pos1_idx); n_pos2 = min(n_pos - n_pos1, len(pos2_idx))
            elif n_pos2 > len(pos2_idx):
                n_pos2 = len(pos2_idx); n_pos1 = min(n_pos - n_pos2, len(pos1_idx))
            s_pos1 = rng.choice(pos1_idx, n_pos1, replace=False) if n_pos1 > 0 else np.array([])
            s_pos2 = rng.choice(pos2_idx, n_pos2, replace=False) if n_pos2 > 0 else np.array([])
            actual_pos = n_pos1 + n_pos2
            n_neg_total = actual_pos * neg_ratio
            total_neg_avail = len(neg1_idx) + len(neg2_idx)
            n_neg = min(n_neg_total, total_neg_avail)
            frac1_neg = rng.uniform(0, 1)
            n_neg1 = int(round(frac1_neg * n_neg)); n_neg2 = n_neg - n_neg1
            if n_neg1 > len(neg1_idx):
                n_neg1 = len(neg1_idx); n_neg2 = min(n_neg - n_neg1, len(neg2_idx))
            elif n_neg2 > len(neg2_idx):
                n_neg2 = len(neg2_idx); n_neg1 = min(n_neg - n_neg2, len(neg1_idx))
            s_neg1 = rng.choice(neg1_idx, int(n_neg1), replace=False) if n_neg1 > 0 else np.array([])
            s_neg2 = rng.choice(neg2_idx, int(n_neg2), replace=False) if n_neg2 > 0 else np.array([])
            iter_indices.append(np.concatenate([s_pos1, s_pos2, s_neg1, s_neg2]))

        all_iter_idx = np.concatenate(iter_indices)
        pos_in_array = np.array([idx_to_pos[i] for i in all_iter_idx])
        iter_labels = label_array[pos_in_array]
        iter_alleles = allele_array[pos_in_array]
        unique_alleles = np.unique(iter_alleles)

        for col in score_cols:
            iter_scores = score_arrays[col][pos_in_array]
            for allele in unique_alleles:
                mask = iter_alleles == allele
                a_labels = iter_labels[mask]; a_scores = iter_scores[mask]
                valid = ~np.isnan(a_scores)
                a_labels = a_labels[valid]; a_scores = a_scores[valid]
                if len(np.unique(a_labels)) < 2:
                    continue
                per_allele_ap[col][allele].append(average_precision_score(a_labels, a_scores))

    alleles_sorted = sorted(all_alleles)
    matrix = pd.DataFrame(index=alleles_sorted, columns=score_cols, dtype=float)
    for col in score_cols:
        for allele in alleles_sorted:
            vals = per_allele_ap[col][allele]
            matrix.loc[allele, col] = np.mean(vals) if vals else np.nan
    matrix.index.name = allele_col
    return matrix


def balance_two_dfs_by_allele(df1, df2, alleles, ratio=99, allele_col="allele", label_col="label", seed=0, return_combined=True, source_col="source", df1_name="df1", df2_name="df2", subsample_to_ratio=False):
    rng = np.random.default_rng(seed)
    d1 = df1.reset_index(drop=True); d2 = df2.reset_index(drop=True)

    def grp(d):
        pos = d.loc[d[label_col] == 1].groupby(allele_col, sort=False).indices
        neg = d.loc[d[label_col] == 0].groupby(allele_col, sort=False).indices
        return pos, neg, np.where((d[label_col]==1).to_numpy())[0], np.where((d[label_col]==0).to_numpy())[0]

    p1, n1, p1g, n1g = grp(d1)
    p2, n2, p2g, n2g = grp(d2)

    take1, take2 = [], []
    for allele in alleles:
        pos1 = p1g[p1[allele]] if allele in p1 else np.array([], dtype=int)
        pos2 = p2g[p2[allele]] if allele in p2 else np.array([], dtype=int)
        neg1 = n1g[n1[allele]] if allele in n1 else np.array([], dtype=int)
        neg2 = n2g[n2[allele]] if allele in n2 else np.array([], dtype=int)

        k = min(len(pos1), len(pos2))
        if k == 0:
            print(f"[WARNING] allele {allele}: one df has 0 positives (df1={len(pos1)}, df2={len(pos2)}); skipping")
            continue

        has_neg1 = len(neg1) > 0; has_neg2 = len(neg2) > 0

        if subsample_to_ratio:
            if has_neg1 and has_neg2:
                k_cap = min(len(neg1) // ratio, len(neg2) // ratio)
            else:
                total_neg = len(neg1) + len(neg2)
                k_cap = total_neg // (2 * ratio)
            if k_cap < k:
                print(f"[INFO] allele {allele}: subsampling pos {k}->{k_cap} to hold 1:{ratio} ratio")
                k = k_cap
            if k <= 0:
                print(f"[WARNING] allele {allele}: cannot hold ratio even at 1 pos/df; skipping")
                continue

        s_pos1 = pos1[rng.permutation(len(pos1))[:k]]
        s_pos2 = pos2[rng.permutation(len(pos2))[:k]]
        total_pos = 2 * k

        if has_neg1 and has_neg2:
            want_each = k * ratio
            t1 = min(want_each, len(neg1)); t2 = min(want_each, len(neg2))
            if (t1 < want_each or t2 < want_each) and not subsample_to_ratio:
                print(f"[WARNING] allele {allele}: neg shortfall (wanted {want_each}/df, df1_neg={len(neg1)}, df2_neg={len(neg2)})")
            s_neg1 = neg1[rng.permutation(len(neg1))[:t1]]
            s_neg2 = neg2[rng.permutation(len(neg2))[:t2]]
        else:
            want_total = total_pos * ratio
            pools = [("d1", neg1)] * has_neg1 + [("d2", neg2)] * has_neg2
            total_avail = sum(len(n) for _, n in pools)
            s_neg1 = np.array([], dtype=int); s_neg2 = np.array([], dtype=int)
            if total_avail == 0:
                print(f"[WARNING] allele {allele}: no negatives in either df; taking none")
            else:
                take_total = min(want_total, total_avail)
                if take_total < want_total and not subsample_to_ratio:
                    print(f"[WARNING] allele {allele}: neg shortfall (wanted {want_total}, avail {total_avail}); taking all")
                remaining = take_total
                for j, (src, negpool) in enumerate(pools):
                    t = min(remaining, len(negpool)) if j == len(pools) - 1 else min(int(round(take_total * len(negpool) / total_avail)), len(negpool))
                    remaining -= t
                    chosen = negpool[rng.permutation(len(negpool))[:t]]
                    if src == "d1": s_neg1 = chosen
                    else: s_neg2 = chosen

        take1.append(np.concatenate([s_pos1, s_neg1]))
        take2.append(np.concatenate([s_pos2, s_neg2]))

    sel1 = np.concatenate(take1) if take1 else np.array([], dtype=int)
    sel2 = np.concatenate(take2) if take2 else np.array([], dtype=int)
    out1 = d1.iloc[sel1].copy(); out2 = d2.iloc[sel2].copy()
    if not return_combined:
        return out1, out2
    out1[source_col] = df1_name; out2[source_col] = df2_name
    return pd.concat([out1, out2], ignore_index=True)
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange, get_num_threads, set_num_threads

@njit(parallel=True)
def _average_precision_columns_numba(labels, scores):
    """
    Binary AP independently for every column of scores.
    Returns NaN when, after NaN removal, only one label class remains,
    matching the skip behavior in the original functions.
    """
    n_rows,n_cols=scores.shape
    out=np.empty(n_cols,dtype=np.float64)

    for j in prange(n_cols):
        n_valid=0
        for i in range(n_rows):
            if not np.isnan(scores[i,j]): n_valid+=1

        if n_valid==0:
            out[j]=np.nan
            continue

        s=np.empty(n_valid,dtype=np.float64)
        y=np.empty(n_valid,dtype=np.int8)
        k=0
        for i in range(n_rows):
            if not np.isnan(scores[i,j]):
                s[k]=scores[i,j]
                y[k]=1 if labels[i]==1 else 0
                k+=1

        total_pos=0
        for i in range(n_valid): total_pos+=y[i]

        if total_pos==0 or total_pos==n_valid:
            out[j]=np.nan
            continue

        # Ascending sort, then process from highest score downward.
        order=np.argsort(s)
        tp=0
        seen=0
        ap=0.0
        i=n_valid-1

        while i>=0:
            score=s[order[i]]
            group_pos=0
            group_n=0
            k=i

            while k>=0 and s[order[k]]==score:
                group_pos+=y[order[k]]
                group_n+=1
                k-=1

            tp+=group_pos
            seen+=group_n

            # Recall increase for this score threshold * precision.
            if group_pos>0:
                ap+=(group_pos/total_pos)*(tp/seen)

            i=k

        out[j]=ap

    return out


def _accumulate_ap_fast(score_matrix, labels, sampled_idx, ap_sum_row, ap_count_row, chunk_size):
    """Accumulate AP for one sampled allele over all score columns."""
    sampled_idx=np.asarray(sampled_idx,dtype=np.int64)
    a_labels=labels[sampled_idx]

    if len(np.unique(a_labels))<2:
        return

    n_cols=score_matrix.shape[1]
    for start in range(0,n_cols,chunk_size):
        end=min(start+chunk_size,n_cols)
        block=score_matrix[sampled_idx,start:end]
        vals=_average_precision_columns_numba(a_labels,block)
        valid=~np.isnan(vals)
        ap_sum_row[start:end][valid]+=vals[valid]
        ap_count_row[start:end][valid]+=1


def mixed_df_per_allele_ap_pos_focused_fast(
    df1, df2, score_cols, allele_pos_counts,
    label_col1="label", label_col2="label", allele_col="allele",
    neg_ratio=99, n_iterations=10, seed=42,
    chunk_size=256, num_threads=None,
):
    """
    Faster equivalent of mixed_df_per_allele_ap_pos_focused.

    Positives are mixed between df1/df2 exactly as before.
    Negatives come only from df1 exactly as before.
    """
    score_cols=list(score_cols)
    if not score_cols: raise ValueError("score_cols cannot be empty.")

    # Score matrix: rows are df1 followed by df2.
    s1=df1[score_cols].to_numpy(copy=False)
    s2=df2[score_cols].to_numpy(copy=False)
    score_matrix=np.concatenate([s1,s2],axis=0)

    y1=df1[label_col1].to_numpy()
    y2=df2[label_col2].to_numpy()
    labels=np.concatenate([y1,y2])

    a1=df1[allele_col].to_numpy()
    a2=df2[allele_col].to_numpy()
    n1=len(df1)

    all_alleles=set(allele_pos_counts.keys())
    alleles_sorted=sorted(all_alleles)
    allele_to_row={a:i for i,a in enumerate(alleles_sorted)}

    g1={}; g2={}
    for allele in all_alleles:
        mask1=a1==allele
        mask2=a2==allele
        g1[allele]={
            "pos":np.flatnonzero(mask1&(y1==1)).astype(np.int64),
            "neg":np.flatnonzero(mask1&(y1==0)).astype(np.int64),
        }
        g2[allele]={
            "pos":(np.flatnonzero(mask2&(y2==1))+n1).astype(np.int64)
        }

    for allele in alleles_sorted:
        np1=len(g1[allele]["pos"]); np2=len(g2[allele]["pos"])
        requested=allele_pos_counts[allele]
        if np1+np2<requested:
            print(f"[WARNING] Allele {allele} has only {np1+np2} positives (df1={np1}, df2={np2}), below requested {requested}")

    ap_sum=np.zeros((len(alleles_sorted),len(score_cols)),dtype=np.float64)
    ap_count=np.zeros((len(alleles_sorted),len(score_cols)),dtype=np.int32)
    rng=np.random.default_rng(seed)

    old_threads=get_num_threads()
    if num_threads is not None: set_num_threads(int(num_threads))

    try:
        # Trigger JIT compilation before tqdm timing becomes meaningful.
        _average_precision_columns_numba(
            np.array([0,1],dtype=np.int8),
            np.array([[0.0],[1.0]],dtype=np.float64)
        )

        for _ in tqdm(range(n_iterations),desc="Iterations"):
            # Keep the same set iteration logic as the original function.
            # for allele in all_alleles:
            for allele in alleles_sorted:

                pos1_idx=g1[allele]["pos"]
                neg1_idx=g1[allele]["neg"]
                pos2_idx=g2[allele]["pos"]
                requested=allele_pos_counts[allele]

                total_pos_avail=len(pos1_idx)+len(pos2_idx)
                if total_pos_avail==0: continue

                n_pos=min(requested,total_pos_avail)
                frac1_pos=rng.uniform(0,1)
                n_pos1=int(round(frac1_pos*n_pos))
                n_pos2=n_pos-n_pos1

                if n_pos1>len(pos1_idx):
                    n_pos1=len(pos1_idx)
                    n_pos2=min(n_pos-n_pos1,len(pos2_idx))
                elif n_pos2>len(pos2_idx):
                    n_pos2=len(pos2_idx)
                    n_pos1=min(n_pos-n_pos2,len(pos1_idx))

                s_pos1=rng.choice(pos1_idx,n_pos1,replace=False) if n_pos1>0 else np.empty(0,dtype=np.int64)
                s_pos2=rng.choice(pos2_idx,n_pos2,replace=False) if n_pos2>0 else np.empty(0,dtype=np.int64)

                actual_pos=n_pos1+n_pos2
                n_neg=actual_pos*neg_ratio
                n_neg1=min(n_neg,len(neg1_idx))
                s_neg1=rng.choice(neg1_idx,int(n_neg1),replace=False) if n_neg1>0 else np.empty(0,dtype=np.int64)

                sampled_idx=np.concatenate([s_pos1,s_pos2,s_neg1]).astype(np.int64,copy=False)
                r=allele_to_row[allele]
                _accumulate_ap_fast(
                    score_matrix,labels,sampled_idx,
                    ap_sum[r],ap_count[r],chunk_size
                )
    finally:
        if num_threads is not None: set_num_threads(old_threads)

    result=np.full_like(ap_sum,np.nan,dtype=np.float64)
    valid=ap_count>0
    result[valid]=ap_sum[valid]/ap_count[valid]

    matrix=pd.DataFrame(result,index=alleles_sorted,columns=score_cols)
    matrix.index.name=allele_col
    return matrix


def mixed_df_per_allele_ap_both_fast(
    df1, df2, score_cols, allele_pos_counts,
    label_col1="label", label_col2="label", allele_col="allele",
    neg_ratio=99, n_iterations=10, seed=42,
    chunk_size=256, num_threads=None,
):
    """
    Faster equivalent of mixed_df_per_allele_ap_both.

    Positives and negatives can both come from df1/df2 exactly as in the
    original sampling procedure.
    """
    score_cols=list(score_cols)
    if not score_cols: raise ValueError("score_cols cannot be empty.")

    s1=df1[score_cols].to_numpy(copy=False)
    s2=df2[score_cols].to_numpy(copy=False)
    score_matrix=np.concatenate([s1,s2],axis=0)

    y1=df1[label_col1].to_numpy()
    y2=df2[label_col2].to_numpy()
    labels=np.concatenate([y1,y2])

    a1=df1[allele_col].to_numpy()
    a2=df2[allele_col].to_numpy()
    n1=len(df1)

    all_alleles=set(allele_pos_counts.keys())
    alleles_sorted=sorted(all_alleles)
    allele_to_row={a:i for i,a in enumerate(alleles_sorted)}

    g1={}; g2={}
    for allele in all_alleles:
        mask1=a1==allele
        mask2=a2==allele

        g1[allele]={
            "pos":np.flatnonzero(mask1&(y1==1)).astype(np.int64),
            "neg":np.flatnonzero(mask1&(y1==0)).astype(np.int64),
        }
        g2[allele]={
            "pos":(np.flatnonzero(mask2&(y2==1))+n1).astype(np.int64),
            "neg":(np.flatnonzero(mask2&(y2==0))+n1).astype(np.int64),
        }

    for allele in alleles_sorted:
        np1=len(g1[allele]["pos"]); np2=len(g2[allele]["pos"])
        requested=allele_pos_counts[allele]
        if np1+np2<requested:
            print(f"[WARNING] Allele {allele} has only {np1+np2} positives (df1={np1}, df2={np2}), below requested {requested}")

    ap_sum=np.zeros((len(alleles_sorted),len(score_cols)),dtype=np.float64)
    ap_count=np.zeros((len(alleles_sorted),len(score_cols)),dtype=np.int32)
    rng=np.random.default_rng(seed)

    old_threads=get_num_threads()
    if num_threads is not None: set_num_threads(int(num_threads))

    try:
        _average_precision_columns_numba(
            np.array([0,1],dtype=np.int8),
            np.array([[0.0],[1.0]],dtype=np.float64)
        )

        for _ in tqdm(range(n_iterations),desc="Iterations"):
            # for allele in all_alleles:
            for allele in alleles_sorted:
                pos1_idx=g1[allele]["pos"]; neg1_idx=g1[allele]["neg"]
                pos2_idx=g2[allele]["pos"]; neg2_idx=g2[allele]["neg"]
                requested=allele_pos_counts[allele]

                total_pos_avail=len(pos1_idx)+len(pos2_idx)
                if total_pos_avail==0: continue

                n_pos=min(requested,total_pos_avail)
                frac1_pos=rng.uniform(0,1)
                n_pos1=int(round(frac1_pos*n_pos))
                n_pos2=n_pos-n_pos1

                if n_pos1>len(pos1_idx):
                    n_pos1=len(pos1_idx)
                    n_pos2=min(n_pos-n_pos1,len(pos2_idx))
                elif n_pos2>len(pos2_idx):
                    n_pos2=len(pos2_idx)
                    n_pos1=min(n_pos-n_pos2,len(pos1_idx))

                s_pos1=rng.choice(pos1_idx,n_pos1,replace=False) if n_pos1>0 else np.empty(0,dtype=np.int64)
                s_pos2=rng.choice(pos2_idx,n_pos2,replace=False) if n_pos2>0 else np.empty(0,dtype=np.int64)

                actual_pos=n_pos1+n_pos2
                n_neg_total=actual_pos*neg_ratio
                total_neg_avail=len(neg1_idx)+len(neg2_idx)
                n_neg=min(n_neg_total,total_neg_avail)

                frac1_neg=rng.uniform(0,1)
                n_neg1=int(round(frac1_neg*n_neg))
                n_neg2=n_neg-n_neg1

                if n_neg1>len(neg1_idx):
                    n_neg1=len(neg1_idx)
                    n_neg2=min(n_neg-n_neg1,len(neg2_idx))
                elif n_neg2>len(neg2_idx):
                    n_neg2=len(neg2_idx)
                    n_neg1=min(n_neg-n_neg2,len(neg1_idx))

                s_neg1=rng.choice(neg1_idx,int(n_neg1),replace=False) if n_neg1>0 else np.empty(0,dtype=np.int64)
                s_neg2=rng.choice(neg2_idx,int(n_neg2),replace=False) if n_neg2>0 else np.empty(0,dtype=np.int64)

                sampled_idx=np.concatenate([s_pos1,s_pos2,s_neg1,s_neg2]).astype(np.int64,copy=False)
                r=allele_to_row[allele]
                _accumulate_ap_fast(
                    score_matrix,labels,sampled_idx,
                    ap_sum[r],ap_count[r],chunk_size
                )
    finally:
        if num_threads is not None: set_num_threads(old_threads)

    result=np.full_like(ap_sum,np.nan,dtype=np.float64)
    valid=ap_count>0
    result[valid]=ap_sum[valid]/ap_count[valid]

    matrix=pd.DataFrame(result,index=alleles_sorted,columns=score_cols)
    matrix.index.name=allele_col
    return matrix
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_lambda_ap_trajectory(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes, ref_col="mhcprime",
    ms_name="MS", assay_name="Assay", ms_map_col="mAP", assay_map_col="mAP",
    highlight_lambda=None, marker_size=80, highlight_size=220,
    line_width=1.5, point_alpha=0.9, equal_tradeoff_line=True,
    fig_size=(6,6), dpi=300, show_plot=True,
):
    """
    Plot lambda trajectories as MS ΔmAP vs assay ΔmAP.

    Expected score format:
      {prefix}steps_{steps}_lr_{lr}_dr_lambda_{lambda}_seed_{seed}

    Lambda strings may be e.g.:
      0, 1eN8, 1eN7, 1eN6, ...

    method_prefixes example:
      {
          "pepcl_r_dr":"holec_pepcl_r_dr_",
          "pepcl_dr":"holec_pepcl_dr_",
      }

    highlight_lambda:
      None, scalar lambda, or dict by method.
      Examples:
        highlight_lambda="1eN5"
        highlight_lambda={"pepcl_r_dr":"1eN5","pepcl_dr":"1eN6"}

    Returns:
      trajectory_df, fig, ax
    """
    def _get_map(df,map_col):
        if map_col in df.columns:
            s=df[map_col].copy(); s.index=s.index.astype(str)
            return s.astype(float)
        return df.mean(axis=0).astype(float)

    def _lambda_float(x):
        s=str(x)
        if s in {"0","0.0"}: return 0.0
        if "eN" in s:
            a,b=s.split("eN"); return float(a)*10**(-int(b))
        return float(s)

    ms_map=_get_map(ms_ap_df,ms_map_col); assay_map=_get_map(assay_ap_df,assay_map_col)
    if ref_col not in ms_map.index or ref_col not in assay_map.index:
        raise ValueError(f"Reference '{ref_col}' must exist in both inputs.")

    score_cols=list(dict.fromkeys(score_cols))
    usable=[c for c in score_cols if c in ms_map.index and c in assay_map.index]
    if not usable: raise ValueError("No score columns are present in both inputs.")

    ordered_methods=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)
    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix): return name
        return None

    pattern=re.compile(
        r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lam>.+?)_seed_(?P<seed>\d+)$"
    )

    rows=[]; bad=[]
    ref_ms=float(ms_map[ref_col]); ref_assay=float(assay_map[ref_col])

    for col in usable:
        method=_method(col); m=pattern.search(str(col))
        if method is None or m is None: bad.append(col); continue
        lam=m["lam"]
        rows.append({
            "score_col":col,"method":method,"steps":int(m["steps"]),"learning_rate":m["lr"],
            "dr_lambda":lam,"dr_lambda_float":_lambda_float(lam),"seed":int(m["seed"]),
            f"{ms_name}_mAP":float(ms_map[col]),f"{assay_name}_mAP":float(assay_map[col]),
            f"{ms_name}_mAP_delta":float(ms_map[col])-ref_ms,
            f"{assay_name}_mAP_delta":float(assay_map[col])-ref_assay,
        })

    raw_df=pd.DataFrame(rows)
    if raw_df.empty: raise ValueError("No valid lambda-sweep columns could be parsed.")
    if bad: print(f"WARNING: skipped {len(bad)} unparsable columns.")

    group_cols=["method","steps","learning_rate","dr_lambda","dr_lambda_float"]
    value_cols=[f"{ms_name}_mAP",f"{assay_name}_mAP",f"{ms_name}_mAP_delta",f"{assay_name}_mAP_delta"]

    trajectory_df=(raw_df.groupby(group_cols,as_index=False,sort=False)
        .agg(**{c:(c,"mean") for c in value_cols},
             n_seeds=("seed","nunique"),
             seed_ids=("seed",lambda x:tuple(sorted(set(x))))))

    trajectory_df=trajectory_df.sort_values(["method","dr_lambda_float"]).reset_index(drop=True)

    lambdas=(trajectory_df[["dr_lambda","dr_lambda_float"]]
        .drop_duplicates().sort_values("dr_lambda_float"))
    lambda_order=lambdas["dr_lambda"].tolist()

    cmap=plt.get_cmap("viridis")
    lambda_colors={lam:cmap(i/max(len(lambda_order)-1,1)) for i,lam in enumerate(lambda_order)}

    x=f"{ms_name}_mAP_delta"; y=f"{assay_name}_mAP_delta"
    fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)

    if equal_tradeoff_line:
        ax.axline((0,0),slope=-1,linestyle="--",linewidth=1.2,color="0.7",zorder=0)

    markers=["o","s","^","D","P","X"]
    method_order=[m for m in method_prefixes if m in trajectory_df["method"].unique()]
    marker_map={m:markers[i%len(markers)] for i,m in enumerate(method_order)}

    for method in method_order:
        g=trajectory_df[trajectory_df["method"]==method].sort_values("dr_lambda_float")
        ax.plot(g[x],g[y],linewidth=line_width,alpha=0.7,color="0.5",zorder=1)

        for _,r in g.iterrows():
            ax.scatter(r[x],r[y],s=marker_size,marker=marker_map[method],
                       color=lambda_colors[r["dr_lambda"]],edgecolor="black",
                       linewidth=0.6,alpha=point_alpha,zorder=3)

    if highlight_lambda is not None:
        targets=highlight_lambda if isinstance(highlight_lambda,dict) else {m:highlight_lambda for m in method_order}
        for method,lam in targets.items():
            lam=str(lam)
            trg=trajectory_df[(trajectory_df["method"]==method)&(trajectory_df["dr_lambda"].astype(str)==lam)]
            if trg.empty:
                print(f"WARNING: no highlight target for method={method}, lambda={lam}")
                continue
            for _,r in trg.iterrows():
                ax.scatter(r[x],r[y],s=highlight_size,marker="*",color=lambda_colors[r["dr_lambda"]],
                           edgecolor="black",linewidth=1.0,zorder=6)

    ax.axvline(0,linestyle="--",linewidth=1,color="0.75")
    ax.axhline(0,linestyle="--",linewidth=1,color="0.75")
    ax.set_xlabel(f"{ms_name} ΔmAP vs {ref_col}")
    ax.set_ylabel(f"{assay_name} ΔmAP vs {ref_col}")
    ax.set_box_aspect(1)

    lambda_handles=[
        plt.Line2D([0],[0],marker="o",linestyle="none",color=lambda_colors[lam],
                   markeredgecolor="black",markersize=7,label=f"λ={lam}")
        for lam in lambda_order
    ]
    method_handles=[
        plt.Line2D([0],[0],marker=marker_map[m],linestyle="none",color="0.5",
                   markeredgecolor="black",markersize=7,label=m)
        for m in method_order
    ]
    ax.legend(handles=method_handles+lambda_handles,frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")

    fig.tight_layout()
    if show_plot: plt.show()
    return trajectory_df,fig,ax
import re
"""
ms_map
assay_map
ms_sd / ms_se
assay_sd / assay_se

ms_change_vs_base
ms_decrease_vs_base
ms_loss_vs_base
ms_change_vs_base_pct
ms_decrease_vs_base_pct

assay_gain_vs_base
assay_gain_vs_base_pct

ms_change_vs_prev
ms_decrease_vs_prev
ms_change_vs_prev_pct
ms_decrease_vs_prev_pct

assay_change_vs_prev
assay_gain_vs_prev
assay_change_vs_prev_pct
assay_gain_vs_prev_pct

adaptation_norm
retention_norm

net_benefit
signed_net_benefit
normalized_net_benefit

balanced_harmonic
balanced_geometric
balanced_arithmetic
balanced_min

assay_available_gain_fraction
ms_fraction_lost
brnb
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# def summarize_lambda_performance(
#     ms_ap_df, assay_ap_df, score_cols, method_prefixes,
#     ref_col="mhcprime",
#     per_allele=False,
#     per_allele_aggregate=False,
#     match_alleles=False,
# ):
#     """
#     Summarize performance across a DR-lambda sweep.

#     Modes
#     -----
#     Default:
#         per_allele=False
#         per_allele_aggregate=False

#         Macro-average each dataset within each seed/configuration, compute
#         metrics, then summarize across seeds.

#         If match_alleles=False:
#             MS and assay each use their own full allele population.

#         If match_alleles=True:
#             MS and assay are first restricted to their common alleles.

#     per_allele_aggregate=True:
#         OLD per_allele behavior.

#         1. Restrict MS and assay to common alleles.
#         2. Compute all metrics separately for every allele x seed.
#         3. Average metrics across alleles within each seed.
#         4. Summarize those seed-level values across seeds.

#         Final output remains one row per lambda/configuration.

#     per_allele=True:
#         Actual per-allele output.

#         1. Restrict MS and assay to common alleles.
#         2. Compute all metrics separately for every allele x seed.
#         3. For every allele x lambda/configuration, summarize across seeds.

#         Final output contains one row per allele x lambda/configuration,
#         including mean, SD, SE, and 95% Student-t CI for every metric.

#     Notes
#     -----
#     per_allele=True and per_allele_aggregate=True are mutually exclusive.

#     Per-allele modes always use common/matched alleles because paired metrics
#     such as BRNB require MS and assay values for the same allele.
#     """
#     if per_allele and per_allele_aggregate:
#         raise ValueError(
#             "per_allele and per_allele_aggregate are mutually exclusive."
#         )

#     paired_allele_mode=per_allele or per_allele_aggregate

#     def _lam(x):
#         s=str(x)
#         if s in {"0","0.0"}:
#             return 0.0
#         if "eN" in s:
#             a,b=s.split("eN")
#             return float(a)*10**(-int(b))
#         return float(s)

#     ordered=sorted(
#         method_prefixes.items(),
#         key=lambda x:len(x[1]),
#         reverse=True
#     )

#     def _method(c):
#         for name,prefix in ordered:
#             if str(c).startswith(prefix):
#                 return name
#         return None

#     pattern=re.compile(
#         r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_"
#         r"(?P<lam>.+?)_seed_(?P<seed>\d+)$"
#     )

#     # ---------------------------------------------------------
#     # Identify common score/configuration columns
#     # ---------------------------------------------------------
#     usable=[
#         c for c in dict.fromkeys(score_cols)
#         if c in ms_ap_df.columns
#         and c in assay_ap_df.columns
#         and c!=ref_col
#     ]

#     if ref_col not in ms_ap_df.columns or ref_col not in assay_ap_df.columns:
#         raise ValueError(
#             f"'{ref_col}' must exist in both inputs."
#         )

#     meta=[]

#     for c in usable:
#         method=_method(c)
#         m=pattern.search(str(c))

#         if method is not None and m is not None:
#             meta.append({
#                 "score_col":c,
#                 "method":method,
#                 "steps":int(m["steps"]),
#                 "learning_rate":m["lr"],
#                 "lambda_string":m["lam"],
#                 "lambda":_lam(m["lam"]),
#                 "seed":int(m["seed"]),
#             })

#     meta=pd.DataFrame(meta)

#     if meta.empty:
#         raise ValueError(
#             "No lambda-sweep columns could be parsed."
#         )

#     model_cols=meta["score_col"].tolist()

#     # ---------------------------------------------------------
#     # Select allele populations
#     # ---------------------------------------------------------
#     if paired_allele_mode or match_alleles:
#         common_idx=ms_ap_df.index.intersection(
#             assay_ap_df.index
#         )

#         if len(common_idx)==0:
#             raise ValueError(
#                 "No overlapping allele indices found between MS and assay."
#             )

#         ms=ms_ap_df.loc[
#             common_idx,
#             [ref_col]+model_cols
#         ].copy()

#         assay=assay_ap_df.loc[
#             common_idx,
#             [ref_col]+model_cols
#         ].copy()

#     else:
#         # Default behavior: each dataset retains its own allele population.
#         ms=ms_ap_df.loc[
#             :,
#             [ref_col]+model_cols
#         ].copy()

#         assay=assay_ap_df.loc[
#             :,
#             [ref_col]+model_cols
#         ].copy()

#     # ---------------------------------------------------------
#     # Raw seed/configuration table
#     # ---------------------------------------------------------
#     rows=[]

#     for _,r in meta.iterrows():
#         c=r["score_col"]

#         if paired_allele_mode:
#             for allele in ms.index:
#                 rows.append({
#                     **r.to_dict(),
#                     "allele":allele,
#                     "ms_map":float(ms.loc[allele,c]),
#                     "assay_map":float(assay.loc[allele,c]),
#                     "ms_baseline":float(ms.loc[allele,ref_col]),
#                     "assay_baseline":float(assay.loc[allele,ref_col]),
#                 })

#         else:
#             rows.append({
#                 **r.to_dict(),
#                 "ms_map":float(ms[c].mean()),
#                 "assay_map":float(assay[c].mean()),
#                 "ms_baseline":float(ms[ref_col].mean()),
#                 "assay_baseline":float(assay[ref_col].mean()),
#             })

#     d=pd.DataFrame(rows)

#     # =========================================================
#     # BASE-RELATIVE METRICS
#     # =========================================================
#     d["ms_change_vs_base"]=(
#         d["ms_map"]-d["ms_baseline"]
#     )

#     d["ms_decrease_vs_base"]=(
#         d["ms_baseline"]-d["ms_map"]
#     )

#     d["ms_loss_vs_base"]=np.maximum(
#         0,
#         d["ms_decrease_vs_base"]
#     )

#     d["assay_gain_vs_base"]=(
#         d["assay_map"]-d["assay_baseline"]
#     )

#     with np.errstate(divide="ignore",invalid="ignore"):
#         d["ms_change_vs_base_pct"]=(
#             100*d["ms_change_vs_base"] /
#             d["ms_baseline"]
#         )

#         d["ms_decrease_vs_base_pct"]=(
#             100*d["ms_decrease_vs_base"] /
#             d["ms_baseline"]
#         )

#         d["assay_gain_vs_base_pct"]=(
#             100*d["assay_gain_vs_base"] /
#             d["assay_baseline"]
#         )

#         # -----------------------------------------------------
#         # BRNB
#         # -----------------------------------------------------
#         d["assay_available_gain_fraction"]=(
#             (d["assay_map"]-d["assay_baseline"]) /
#             (1-d["assay_baseline"])
#         )

#         d["ms_fraction_lost"]=(
#             (d["ms_baseline"]-d["ms_map"]) /
#             d["ms_baseline"]
#         )

#     d["brnb"]=(
#         d["assay_available_gain_fraction"] -
#         d["ms_fraction_lost"]
#     )

#     # =========================================================
#     # PREVIOUS-LAMBDA METRICS
#     # =========================================================
#     prev_group=["method","seed"]

#     if paired_allele_mode:
#         prev_group.append("allele")

#     d=d.sort_values(
#         prev_group+["lambda"]
#     )

#     d["ms_change_vs_prev"]=(
#         d.groupby(prev_group)["ms_map"]
#         .diff()
#     )

#     d["ms_decrease_vs_prev"]=(
#         -d["ms_change_vs_prev"]
#     )

#     d["assay_change_vs_prev"]=(
#         d.groupby(prev_group)["assay_map"]
#         .diff()
#     )

#     d["assay_gain_vs_prev"]=(
#         d["assay_change_vs_prev"]
#     )

#     d["ms_change_vs_prev_pct"]=(
#         100*d.groupby(prev_group)["ms_map"]
#         .pct_change(fill_method=None)
#     )

#     d["ms_decrease_vs_prev_pct"]=(
#         -d["ms_change_vs_prev_pct"]
#     )

#     d["assay_change_vs_prev_pct"]=(
#         100*d.groupby(prev_group)["assay_map"]
#         .pct_change(fill_method=None)
#     )

#     d["assay_gain_vs_prev_pct"]=(
#         d["assay_change_vs_prev_pct"]
#     )

#     # =========================================================
#     # NORMALIZED ADAPTATION / RETENTION
#     # =========================================================
#     norm_group=["method","seed"]

#     if paired_allele_mode:
#         norm_group.append("allele")

#     max_gain=(
#         d.groupby(norm_group)["assay_gain_vs_base"]
#         .transform("max")
#         .clip(lower=0)
#     )

#     d["adaptation_norm"]=np.where(
#         max_gain>0,
#         np.clip(
#             d["assay_gain_vs_base"]/max_gain,
#             0,
#             1
#         ),
#         0
#     )

#     d["retention_norm"]=np.clip(
#         d["ms_map"]/d["ms_baseline"],
#         0,
#         1
#     )

#     a=d["adaptation_norm"]
#     r=d["retention_norm"]

#     # =========================================================
#     # COMPOSITE METRICS
#     # =========================================================
#     d["net_benefit"]=(
#         d["assay_gain_vs_base"] -
#         d["ms_loss_vs_base"]
#     )

#     d["signed_net_benefit"]=(
#         d["assay_gain_vs_base"] +
#         d["ms_change_vs_base"]
#     )

#     d["normalized_net_benefit"]=(
#         a+r-1
#     )

#     d["balanced_harmonic"]=np.where(
#         a+r>0,
#         2*a*r/(a+r),
#         0
#     )

#     d["balanced_geometric"]=np.sqrt(
#         a*r
#     )

#     d["balanced_arithmetic"]=(
#         a+r
#     )/2

#     d["balanced_min"]=np.minimum(
#         a,
#         r
#     )

#     # Replace invalid ratio-derived values with NaN.
#     d=d.replace(
#         [np.inf,-np.inf],
#         np.nan
#     )

#     # ---------------------------------------------------------
#     # Identify metrics available for aggregation
#     # ---------------------------------------------------------
#     id_seed=[
#         "method",
#         "steps",
#         "learning_rate",
#         "lambda_string",
#         "lambda",
#         "seed",
#     ]

#     ignore=set(
#         id_seed+
#         ["score_col","allele"]
#     )

#     metrics=[
#         c for c in d.columns
#         if c not in ignore
#         and pd.api.types.is_numeric_dtype(d[c])
#     ]

#     # =========================================================
#     # MODE 1: ACTUAL PER-ALLELE OUTPUT
#     # =========================================================
#     if per_allele:
#         id_final=[
#             "method",
#             "steps",
#             "learning_rate",
#             "lambda_string",
#             "lambda",
#             "allele",
#         ]

#         means=(
#             d.groupby(
#                 id_final,
#                 as_index=False
#             )
#             .agg(
#                 **{
#                     c:(c,"mean")
#                     for c in metrics
#                 },
#                 n_seeds=("seed","nunique"),
#                 seed_ids=(
#                     "seed",
#                     lambda x:tuple(
#                         sorted(set(x))
#                     )
#                 ),
#             )
#         )

#         spreads=(
#             d.groupby(id_final)[metrics]
#             .std()
#             .add_suffix("_sd")
#             .reset_index()
#         )

#         out=means.merge(
#             spreads,
#             on=id_final,
#             how="left"
#         )

#         crit=out["n_seeds"].apply(
#             lambda n:
#                 student_t.ppf(0.975,n-1)
#                 if n>1
#                 else np.nan
#         )

#         for metric in metrics:
#             out[f"{metric}_se"]=(
#                 out[f"{metric}_sd"] /
#                 np.sqrt(out["n_seeds"])
#             )

#             out[f"{metric}_ci95"]=(
#                 crit*out[f"{metric}_se"]
#             )

#         out["alleles_matched"]=True

#         return (
#             out
#             .sort_values(
#                 ["method","allele","lambda"]
#             )
#             .reset_index(drop=True)
#         )

#     # =========================================================
#     # MODE 2: OLD PER-ALLELE-AGGREGATE BEHAVIOR
#     # =========================================================
#     if per_allele_aggregate:
#         seed_df=(
#             d.groupby(
#                 id_seed,
#                 as_index=False
#             )
#             .agg(
#                 **{
#                     c:(c,"mean")
#                     for c in metrics
#                 },
#                 n_alleles=("allele","nunique"),
#             )
#         )

#     # =========================================================
#     # MODE 3: STANDARD MACRO BEHAVIOR
#     # =========================================================
#     else:
#         seed_df=d[
#             id_seed+metrics
#         ].copy()

#     # ---------------------------------------------------------
#     # Seed-level values -> final mean + uncertainty
#     # ---------------------------------------------------------
#     id_final=[
#         "method",
#         "steps",
#         "learning_rate",
#         "lambda_string",
#         "lambda",
#     ]

#     metrics=[
#         c for c in metrics
#         if c in seed_df.columns
#     ]

#     means=(
#         seed_df
#         .groupby(
#             id_final,
#             as_index=False
#         )
#         .agg(
#             **{
#                 c:(c,"mean")
#                 for c in metrics
#             },
#             n_seeds=("seed","nunique"),
#             seed_ids=(
#                 "seed",
#                 lambda x:tuple(
#                     sorted(set(x))
#                 )
#             ),
#         )
#     )

#     spreads=(
#         seed_df
#         .groupby(id_final)[metrics]
#         .std()
#         .add_suffix("_sd")
#         .reset_index()
#     )

#     out=means.merge(
#         spreads,
#         on=id_final,
#         how="left"
#     )

#     crit=out["n_seeds"].apply(
#         lambda n:
#             student_t.ppf(0.975,n-1)
#             if n>1
#             else np.nan
#     )

#     for metric in metrics:
#         out[f"{metric}_se"]=(
#             out[f"{metric}_sd"] /
#             np.sqrt(out["n_seeds"])
#         )

#         out[f"{metric}_ci95"]=(
#             crit*out[f"{metric}_se"]
#         )

#     if per_allele_aggregate:
#         n_alleles_df=(
#             seed_df
#             .groupby(
#                 id_final,
#                 as_index=False
#             )["n_alleles"]
#             .mean()
#         )

#         out=out.merge(
#             n_alleles_df,
#             on=id_final,
#             how="left"
#         )

#     out["alleles_matched"]=bool(
#         paired_allele_mode or match_alleles
#     )

#     out["n_ms_alleles"]=len(ms.index)
#     out["n_assay_alleles"]=len(assay.index)

#     # Backward-compatible aliases.
#     out["ms_sd"]=out["ms_map_sd"]
#     out["ms_se"]=out["ms_map_se"]
#     out["assay_sd"]=out["assay_map_sd"]
#     out["assay_se"]=out["assay_map_se"]

#     return (
#         out
#         .sort_values(["method","lambda"])
#         .reset_index(drop=True)
#     )


def summarize_lambda_performance(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes,
    ref_col="mhcprime",
    per_allele=False,
    per_allele_aggregate=False,
    match_alleles=False,
):
    """
    Summarize performance across a DR-lambda sweep.

    Modes
    -----
    Default:
        per_allele=False
        per_allele_aggregate=False

        Macro-average each dataset within each seed/configuration, compute
        metrics, then summarize across seeds.

        If match_alleles=False:
            MS and assay each use their own full allele population.

        If match_alleles=True:
            MS and assay are first restricted to their common alleles.

    per_allele_aggregate=True:
        OLD per_allele behavior.

        1. Restrict MS and assay to common alleles.
        2. Compute all metrics separately for every allele x seed.
        3. Average metrics across alleles within each seed.
        4. Summarize those seed-level values across seeds.

        Final output remains one row per lambda/configuration.

    per_allele=True:
        Actual per-allele output.

        1. Restrict MS and assay to common alleles.
        2. Compute all metrics separately for every allele x seed.
        3. For every allele x lambda/configuration, summarize across seeds.

        Final output contains one row per allele x lambda/configuration,
        including mean, SD, SE, and 95% Student-t CI for every metric.

    Notes
    -----
    per_allele=True and per_allele_aggregate=True are mutually exclusive.

    Per-allele modes always use common/matched alleles because paired metrics
    such as BRNB require MS and assay values for the same allele.

    MHCPrime baseline handling
    --------------------------
    MS validation is fixed across replicates, so every seed uses ref_col.

    Assay validation uses:
        seeds 0-4 -> ref_col
        seed 5    -> ref_col_seed_5
        seed 6    -> ref_col_seed_6
        ...

    Every MHCPrime-relative quantity is therefore computed using the reference
    matched to the same replicate before any across-seed aggregation.
    """
    if per_allele and per_allele_aggregate:
        raise ValueError(
            "per_allele and per_allele_aggregate are mutually exclusive."
        )

    paired_allele_mode=per_allele or per_allele_aggregate

    def _lam(x):
        s=str(x)
        if s in {"0","0.0"}:
            return 0.0
        if "eN" in s:
            a,b=s.split("eN")
            return float(a)*10**(-int(b))
        return float(s)

    ordered=sorted(
        method_prefixes.items(),
        key=lambda x:len(x[1]),
        reverse=True
    )

    def _method(c):
        for name,prefix in ordered:
            if str(c).startswith(prefix):
                return name
        return None

    pattern=re.compile(
        r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_"
        r"(?P<lam>.+?)_seed_(?P<seed>\d+)$"
    )

    # ---------------------------------------------------------
    # Identify common score/configuration columns
    # ---------------------------------------------------------
    def _is_ref_col(c):
        c=str(c)
        return c==ref_col or c.startswith(f"{ref_col}_seed_")

    def _assay_ref_col_for_seed(seed):
        seed=int(seed)
        return ref_col if seed<=4 else f"{ref_col}_seed_{seed}"

    usable=[
        c for c in dict.fromkeys(score_cols)
        if c in ms_ap_df.columns
        and c in assay_ap_df.columns
        and not _is_ref_col(c)
    ]

    if ref_col not in ms_ap_df.columns or ref_col not in assay_ap_df.columns:
        raise ValueError(
            f"'{ref_col}' must exist in both inputs."
        )

    meta=[]

    for c in usable:
        method=_method(c)
        m=pattern.search(str(c))

        if method is not None and m is not None:
            meta.append({
                "score_col":c,
                "method":method,
                "steps":int(m["steps"]),
                "learning_rate":m["lr"],
                "lambda_string":m["lam"],
                "lambda":_lam(m["lam"]),
                "seed":int(m["seed"]),
            })

    meta=pd.DataFrame(meta)

    if meta.empty:
        raise ValueError(
            "No lambda-sweep columns could be parsed."
        )

    model_cols=meta["score_col"].tolist()

    # ---------------------------------------------------------
    # Select allele populations
    # ---------------------------------------------------------
    assay_ref_cols_needed=sorted({
        _assay_ref_col_for_seed(seed)
        for seed in meta["seed"].unique()
    })

    missing_assay_refs=[
        c for c in assay_ref_cols_needed
        if c not in assay_ap_df.columns
    ]
    if missing_assay_refs:
        raise ValueError(
            f"Missing assay reference columns required by parsed seeds: {missing_assay_refs}"
        )

    if paired_allele_mode or match_alleles:
        common_idx=ms_ap_df.index.intersection(
            assay_ap_df.index
        )

        if len(common_idx)==0:
            raise ValueError(
                "No overlapping allele indices found between MS and assay."
            )

        ms=ms_ap_df.loc[
            common_idx,
            [ref_col]+model_cols
        ].copy()

        assay=assay_ap_df.loc[
            common_idx,
            list(dict.fromkeys(assay_ref_cols_needed+model_cols))
        ].copy()

    else:
        # Default behavior: each dataset retains its own allele population.
        ms=ms_ap_df.loc[
            :,
            [ref_col]+model_cols
        ].copy()

        assay=assay_ap_df.loc[
            :,
            list(dict.fromkeys(assay_ref_cols_needed+model_cols))
        ].copy()

    # ---------------------------------------------------------
    # Raw seed/configuration table
    # ---------------------------------------------------------
    rows=[]

    for _,r in meta.iterrows():
        c=r["score_col"]
        seed=int(r["seed"])
        assay_ref_col=_assay_ref_col_for_seed(seed)

        if paired_allele_mode:
            for allele in ms.index:
                rows.append({
                    **r.to_dict(),
                    "allele":allele,
                    "ms_map":float(ms.loc[allele,c]),
                    "assay_map":float(assay.loc[allele,c]),
                    "ms_baseline":float(ms.loc[allele,ref_col]),
                    "assay_baseline":float(assay.loc[allele,assay_ref_col]),
                    "ms_ref_col":ref_col,
                    "assay_ref_col":assay_ref_col,
                })

        else:
            rows.append({
                **r.to_dict(),
                "ms_map":float(ms[c].mean()),
                "assay_map":float(assay[c].mean()),
                "ms_baseline":float(ms[ref_col].mean()),
                "assay_baseline":float(assay[assay_ref_col].mean()),
                "ms_ref_col":ref_col,
                "assay_ref_col":assay_ref_col,
            })

    d=pd.DataFrame(rows)

    # =========================================================
    # BASE-RELATIVE METRICS
    # =========================================================
    d["ms_change_vs_base"]=(
        d["ms_map"]-d["ms_baseline"]
    )

    d["ms_decrease_vs_base"]=(
        d["ms_baseline"]-d["ms_map"]
    )

    d["ms_loss_vs_base"]=np.maximum(
        0,
        d["ms_decrease_vs_base"]
    )

    d["assay_gain_vs_base"]=(
        d["assay_map"]-d["assay_baseline"]
    )

    with np.errstate(divide="ignore",invalid="ignore"):
        d["ms_change_vs_base_pct"]=(
            100*d["ms_change_vs_base"] /
            d["ms_baseline"]
        )

        d["ms_decrease_vs_base_pct"]=(
            100*d["ms_decrease_vs_base"] /
            d["ms_baseline"]
        )

        d["assay_gain_vs_base_pct"]=(
            100*d["assay_gain_vs_base"] /
            d["assay_baseline"]
        )

        # -----------------------------------------------------
        # BRNB
        # -----------------------------------------------------
        d["assay_available_gain_fraction"]=(
            (d["assay_map"]-d["assay_baseline"]) /
            (1-d["assay_baseline"])
        )

        d["ms_fraction_lost"]=(
            (d["ms_baseline"]-d["ms_map"]) /
            d["ms_baseline"]
        )

    d["brnb"]=(
        d["assay_available_gain_fraction"] -
        d["ms_fraction_lost"]
    )

    # =========================================================
    # PREVIOUS-LAMBDA METRICS
    # =========================================================
    prev_group=["method","seed"]

    if paired_allele_mode:
        prev_group.append("allele")

    d=d.sort_values(
        prev_group+["lambda"]
    )

    d["ms_change_vs_prev"]=(
        d.groupby(prev_group)["ms_map"]
        .diff()
    )

    d["ms_decrease_vs_prev"]=(
        -d["ms_change_vs_prev"]
    )

    d["assay_change_vs_prev"]=(
        d.groupby(prev_group)["assay_map"]
        .diff()
    )

    d["assay_gain_vs_prev"]=(
        d["assay_change_vs_prev"]
    )

    d["ms_change_vs_prev_pct"]=(
        100*d.groupby(prev_group)["ms_map"]
        .pct_change(fill_method=None)
    )

    d["ms_decrease_vs_prev_pct"]=(
        -d["ms_change_vs_prev_pct"]
    )

    d["assay_change_vs_prev_pct"]=(
        100*d.groupby(prev_group)["assay_map"]
        .pct_change(fill_method=None)
    )

    d["assay_gain_vs_prev_pct"]=(
        d["assay_change_vs_prev_pct"]
    )

    # =========================================================
    # NORMALIZED ADAPTATION / RETENTION
    # =========================================================
    norm_group=["method","seed"]

    if paired_allele_mode:
        norm_group.append("allele")

    max_gain=(
        d.groupby(norm_group)["assay_gain_vs_base"]
        .transform("max")
        .clip(lower=0)
    )

    d["adaptation_norm"]=np.where(
        max_gain>0,
        np.clip(
            d["assay_gain_vs_base"]/max_gain,
            0,
            1
        ),
        0
    )

    d["retention_norm"]=np.clip(
        d["ms_map"]/d["ms_baseline"],
        0,
        1
    )

    a=d["adaptation_norm"]
    r=d["retention_norm"]

    # =========================================================
    # COMPOSITE METRICS
    # =========================================================
    d["net_benefit"]=(
        d["assay_gain_vs_base"] -
        d["ms_loss_vs_base"]
    )

    d["signed_net_benefit"]=(
        d["assay_gain_vs_base"] +
        d["ms_change_vs_base"]
    )

    d["normalized_net_benefit"]=(
        a+r-1
    )

    d["balanced_harmonic"]=np.where(
        a+r>0,
        2*a*r/(a+r),
        0
    )

    d["balanced_geometric"]=np.sqrt(
        a*r
    )

    d["balanced_arithmetic"]=(
        a+r
    )/2

    d["balanced_min"]=np.minimum(
        a,
        r
    )

    # Replace invalid ratio-derived values with NaN.
    d=d.replace(
        [np.inf,-np.inf],
        np.nan
    )

    # ---------------------------------------------------------
    # Identify metrics available for aggregation
    # ---------------------------------------------------------
    id_seed=[
        "method",
        "steps",
        "learning_rate",
        "lambda_string",
        "lambda",
        "seed",
    ]

    ignore=set(
        id_seed+
        ["score_col","allele","ms_ref_col","assay_ref_col"]
    )

    metrics=[
        c for c in d.columns
        if c not in ignore
        and pd.api.types.is_numeric_dtype(d[c])
    ]

    # =========================================================
    # MODE 1: ACTUAL PER-ALLELE OUTPUT
    # =========================================================
    if per_allele:
        id_final=[
            "method",
            "steps",
            "learning_rate",
            "lambda_string",
            "lambda",
            "allele",
        ]

        means=(
            d.groupby(
                id_final,
                as_index=False
            )
            .agg(
                **{
                    c:(c,"mean")
                    for c in metrics
                },
                n_seeds=("seed","nunique"),
                seed_ids=(
                    "seed",
                    lambda x:tuple(
                        sorted(set(x))
                    )
                ),
            )
        )

        spreads=(
            d.groupby(id_final)[metrics]
            .std()
            .add_suffix("_sd")
            .reset_index()
        )

        out=means.merge(
            spreads,
            on=id_final,
            how="left"
        )

        crit=out["n_seeds"].apply(
            lambda n:
                student_t.ppf(0.975,n-1)
                if n>1
                else np.nan
        )

        for metric in metrics:
            out[f"{metric}_se"]=(
                out[f"{metric}_sd"] /
                np.sqrt(out["n_seeds"])
            )

            out[f"{metric}_ci95"]=(
                crit*out[f"{metric}_se"]
            )

        out["alleles_matched"]=True

        return (
            out
            .sort_values(
                ["method","allele","lambda"]
            )
            .reset_index(drop=True)
        )

    # =========================================================
    # MODE 2: OLD PER-ALLELE-AGGREGATE BEHAVIOR
    # =========================================================
    if per_allele_aggregate:
        seed_df=(
            d.groupby(
                id_seed,
                as_index=False
            )
            .agg(
                **{
                    c:(c,"mean")
                    for c in metrics
                },
                n_alleles=("allele","nunique"),
            )
        )

    # =========================================================
    # MODE 3: STANDARD MACRO BEHAVIOR
    # =========================================================
    else:
        seed_df=d[
            id_seed+metrics
        ].copy()

    # ---------------------------------------------------------
    # Seed-level values -> final mean + uncertainty
    # ---------------------------------------------------------
    id_final=[
        "method",
        "steps",
        "learning_rate",
        "lambda_string",
        "lambda",
    ]

    metrics=[
        c for c in metrics
        if c in seed_df.columns
    ]

    means=(
        seed_df
        .groupby(
            id_final,
            as_index=False
        )
        .agg(
            **{
                c:(c,"mean")
                for c in metrics
            },
            n_seeds=("seed","nunique"),
            seed_ids=(
                "seed",
                lambda x:tuple(
                    sorted(set(x))
                )
            ),
        )
    )

    spreads=(
        seed_df
        .groupby(id_final)[metrics]
        .std()
        .add_suffix("_sd")
        .reset_index()
    )

    out=means.merge(
        spreads,
        on=id_final,
        how="left"
    )

    crit=out["n_seeds"].apply(
        lambda n:
            student_t.ppf(0.975,n-1)
            if n>1
            else np.nan
    )

    for metric in metrics:
        out[f"{metric}_se"]=(
            out[f"{metric}_sd"] /
            np.sqrt(out["n_seeds"])
        )

        out[f"{metric}_ci95"]=(
            crit*out[f"{metric}_se"]
        )

    if per_allele_aggregate:
        n_alleles_df=(
            seed_df
            .groupby(
                id_final,
                as_index=False
            )["n_alleles"]
            .mean()
        )

        out=out.merge(
            n_alleles_df,
            on=id_final,
            how="left"
        )

    out["alleles_matched"]=bool(
        paired_allele_mode or match_alleles
    )

    out["n_ms_alleles"]=len(ms.index)
    out["n_assay_alleles"]=len(assay.index)

    # Audit trail for the baseline used by each replicate.
    ref_map=(
        d[["seed","ms_ref_col","assay_ref_col"]]
        .drop_duplicates()
        .sort_values("seed")
    )
    out["ms_ref_col_by_seed"]=[
        dict(zip(ref_map["seed"],ref_map["ms_ref_col"]))
        for _ in range(len(out))
    ]
    out["assay_ref_col_by_seed"]=[
        dict(zip(ref_map["seed"],ref_map["assay_ref_col"]))
        for _ in range(len(out))
    ]

    # Backward-compatible aliases.
    out["ms_sd"]=out["ms_map_sd"]
    out["ms_se"]=out["ms_map_se"]
    out["assay_sd"]=out["assay_map_sd"]
    out["assay_se"]=out["assay_map_se"]

    return (
        out
        .sort_values(["method","lambda"])
        .reset_index(drop=True)
    )



def plot_df_line(
    df, x, y, group=None, reverse_x=False, discrete_x=False, log_x=False, scientific_x=False,
    marker="o", xlabel=None, ylabel=None, title=None, fig_size=(6,5), dpi=300,
    linewidth=1.5, marker_size=6, show_plot=True,
):
    """
    Generic dataframe line plot.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
        Columns to plot.
    group : str or None
        Optional column defining separate lines.
    reverse_x : bool
        Plot x from largest -> smallest.
    discrete_x : bool
        Treat observed x values as equally spaced categories.
    log_x : bool
        Use log x-axis. If x contains 0, symlog is used automatically.
        Only observed x values are shown as ticks.
    scientific_x : bool
        Format numeric x ticks as 1e-5, 1e-6, etc.
    """
    if discrete_x and log_x:
        raise ValueError("Choose either discrete_x=True or log_x=True, not both.")

    d=df.copy()
    vals=sorted(d[x].dropna().unique(),reverse=reverse_x)
    if not vals: raise ValueError(f"No valid values found in column '{x}'.")

    def _format_num(v):
        if float(v)==0: return "0"
        m,e=f"{float(v):.12e}".split("e")
        m=m.rstrip("0").rstrip("."); e=int(e)
        return f"{m}e{e}"

    fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)
    groups=[(None,d)] if group is None else d.groupby(group,sort=False)

    if discrete_x:
        xpos={v:i for i,v in enumerate(vals)}
        for name,g in groups:
            g=g.sort_values(x,ascending=not reverse_x)
            ax.plot([xpos[v] for v in g[x]],g[y],marker=marker,linewidth=linewidth,
                    markersize=marker_size,label=None if name is None else str(name))
        labels=[_format_num(v) if scientific_x else str(v) for v in vals]
        ax.set_xticks(range(len(vals)),labels=labels)

    else:
        for name,g in groups:
            g=g.sort_values(x,ascending=not reverse_x)
            ax.plot(g[x],g[y],marker=marker,linewidth=linewidth,
                    markersize=marker_size,label=None if name is None else str(name))

        if log_x:
            if (d[x]<0).any(): raise ValueError("log_x requires non-negative x values.")
            positive=sorted(v for v in vals if v>0)
            if not positive: raise ValueError("log_x requires at least one positive x value.")

            if any(float(v)==0 for v in vals):
                ax.set_xscale("symlog",linthresh=min(positive)/10)
            else:
                ax.set_xscale("log")

            ax.set_xticks(vals)
            if scientific_x:
                ax.set_xticklabels([_format_num(v) for v in vals])

        elif scientific_x:
            ax.set_xticks(vals)
            ax.set_xticklabels([_format_num(v) for v in vals])

        if reverse_x:
            ax.invert_xaxis()

    ax.set_xlabel(x if xlabel is None else xlabel)
    ax.set_ylabel(y if ylabel is None else ylabel)
    if title is not None: ax.set_title(title)
    if group is not None: ax.legend(frameon=False)

    fig.tight_layout()
    if show_plot: plt.show()
    return fig,ax
def merge_dfs_by_prefix(df_dict, prefixes):
    out = {}
    for prefix in prefixes:
        matching = [v for k, v in df_dict.items() if k.startswith(prefix)]
        if not matching:
            continue
        merged = pd.concat(matching, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out[prefix] = merged
    return out

def merge_two_dicts_by_keys(dict1, dict2, keys):
    out = {}
    for key in keys:
        merged = pd.concat([dict1[key], dict2[key]], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out[key] = merged
    return out

def select_lambda_validation_candidates_v2(
    ms_val_df, assay_val_df, score_cols, method_prefixes, ref_col="mhcprime",
    min_seeds=3, focus_seeds=None, dual_selection_modes=("balanced",),
    net_benefit_metric="net_benefit", balanced_metric="harmonic",
    include_common_balanced=True, selection_se_multiplier=1.0,
    lambda_tie_break="all", verbose=True,
):
    """
    Lambda-specific validation selection.

    Expected:
      {prefix}steps_N_lr_X_dr_lambda_L_seed_S

    All metrics are computed per replicate first, then averaged across replicates.

    dual_selection_modes:
      "balanced"
      "bounded_assay_then_ms"
      "bounded_ms_then_assay"
      "net_benefit"

    net_benefit_metric:
      "net_benefit"                   raw assay gain - clipped MS loss
      "brnb"                          baseline-relative normalized benefit
      "net_benefit_normalized"        method-specific adaptation + retention - 1
      "net_benefit_normalized_common" common-scale adaptation + retention - 1

    balanced_metric:
      "harmonic", "geometric", "arithmetic", "min"

    lambda_tie_break:
      "all"     -> return all lambdas in the final SE-eligible pool
      "lowest"  -> return lowest lambda in that pool
      "highest" -> return highest lambda in that pool

    Returns
    -------
    candidate_df, selected_df, selection_log
    """
    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty {method: prefix} dictionary.")

    allowed_modes={"balanced","bounded_assay_then_ms","bounded_ms_then_assay","net_benefit"}
    if isinstance(dual_selection_modes,str): dual_selection_modes=(dual_selection_modes,)
    else: dual_selection_modes=tuple(dual_selection_modes)
    dual_selection_modes=tuple(dict.fromkeys(dual_selection_modes))
    bad=set(dual_selection_modes)-allowed_modes
    if bad: raise ValueError(f"Unknown dual_selection_modes: {sorted(bad)}.")

    allowed_nb={"net_benefit","brnb","net_benefit_normalized","net_benefit_normalized_common"}
    if net_benefit_metric not in allowed_nb: raise ValueError(f"net_benefit_metric must be one of {sorted(allowed_nb)}.")
    allowed_bal={"harmonic","geometric","arithmetic","min"}
    if balanced_metric not in allowed_bal: raise ValueError(f"balanced_metric must be one of {sorted(allowed_bal)}.")
    if lambda_tie_break not in {"all","lowest","highest"}: raise ValueError("lambda_tie_break must be 'all', 'lowest', or 'highest'.")

    selection_se_multiplier=float(selection_se_multiplier)
    if not np.isfinite(selection_se_multiplier) or selection_se_multiplier<0:
        raise ValueError("selection_se_multiplier must be finite and >=0.")

    if focus_seeds is not None:
        if np.isscalar(focus_seeds): focus_seeds=[focus_seeds]
        focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in focus_seeds)))
        if not focus_seeds or any(s<0 for s in focus_seeds): raise ValueError("focus_seeds must contain non-negative replicate IDs.")

    for df,name in [(ms_val_df,"ms_val_df"),(assay_val_df,"assay_val_df")]:
        if not df.columns.is_unique:
            dup=df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"{name} contains duplicate columns: {dup[:10]}")
    if ref_col not in ms_val_df or ref_col not in assay_val_df:
        raise ValueError(f"'{ref_col}' must exist in both validation dataframes.")

    score_cols=list(dict.fromkeys(score_cols))
    cols=[c for c in score_cols if c!=ref_col]
    assay_cols,ms_cols=set(assay_val_df.columns),set(ms_val_df.columns)
    assay_usable=[c for c in cols if c in assay_cols]
    dual_usable=[c for c in cols if c in assay_cols and c in ms_cols]
    if not assay_usable: raise ValueError("No supplied candidate columns exist in assay_val_df.")

    def _eval(df,cols,name):
        base=df[ref_col].notna()
        bad=[c for c in cols if not df[c].notna().equals(base)]
        if bad: raise ValueError(f"{name}: candidates do not share '{ref_col}' evaluable rows. Examples: {bad[:10]}")
        out=df.loc[base,[ref_col]+cols].copy()
        if verbose: print(f"{name}: common evaluable rows = {len(out)}")
        return out.mean(axis=0)

    assay_metric=_eval(assay_val_df,assay_usable,"Assay"); assay_base=float(assay_metric[ref_col])
    ms_metric=_eval(ms_val_df,dual_usable,"MS") if dual_usable else None
    ms_base=float(ms_metric[ref_col]) if ms_metric is not None else float(ms_val_df.loc[ms_val_df[ref_col].notna(),ref_col].mean())
    if not np.isfinite(ms_base) or ms_base<=0: raise ValueError("MS reference performance must be >0.")

    def _lam(x):
        s=str(x)
        if s in {"0","0.0"}: return 0.0
        if "eN" in s:
            a,b=s.split("eN"); return float(a)*10**(-int(b))
        return float(s)

    ordered_methods=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)
    pattern=re.compile(r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lam>.+?)_seed_(?P<seed>\d+)$")

    def _method(c):
        for name,prefix in ordered_methods:
            if str(c).startswith(prefix): return name
        return None

    def _parse(cols,with_ms):
        rows=[]; bad=[]
        for c in cols:
            method=_method(c); m=pattern.search(str(c))
            if method is None or m is None: bad.append(c); continue
            assay=float(assay_metric[c])
            row={"score_col":c,"method":method,"learning_rate":m["lr"],"step_budget":int(m["steps"]),
                 "lambda_string":m["lam"],"dr_lambda":_lam(m["lam"]),"seed":int(m["seed"]),
                 "assay_performance":assay}
            if with_ms:
                ms=float(ms_metric[c]); ag=assay-assay_base; mc=ms-ms_base; ml=max(0.0,-mc)
                brnb=(ag/(1-assay_base) + mc/ms_base) if assay_base<1 else np.nan
                row.update({"ms_performance":ms,"assay_gain":ag,"ms_change":mc,"ms_loss":ml,
                            "net_benefit":ag-ml,"assay_available_gain_fraction":ag/(1-assay_base) if assay_base<1 else np.nan,
                            "ms_relative_change":mc/ms_base,"brnb":brnb})
            rows.append(row)
        return pd.DataFrame(rows),bad

    assay_seed_df,assay_bad=_parse(assay_usable,False)
    dual_seed_df,dual_bad=_parse(dual_usable,True)
    if assay_seed_df.empty: raise ValueError("No lambda candidates could be parsed.")
    if verbose and assay_bad: print(f"Skipped {len(assay_bad)} unparsable assay columns.")

    # This selector is specifically for fixed step/LR lambda sweeps.
    for method in method_prefixes:
        for df,name in [(assay_seed_df,"ASSAY"),(dual_seed_df,"DUAL")]:
            s=df[df["method"]==method]
            if s.empty: continue
            hp=s[["learning_rate","step_budget"]].drop_duplicates()
            if len(hp)!=1:
                raise ValueError(f"{name}: method '{method}' has {len(hp)} LR/step combinations; lambda selector requires fixed LR/steps.")

    if focus_seeds is not None:
        assay_seed_df=assay_seed_df[assay_seed_df.seed.isin(focus_seeds)].copy()
        dual_seed_df=dual_seed_df[dual_seed_df.seed.isin(focus_seeds)].copy()

    candidate_keys=["method","learning_rate","step_budget","dr_lambda"]
    config_keys=["dr_lambda"]

    def _eligible(df,name):
        if df.empty: return df,pd.DataFrame()
        info=(df.groupby(candidate_keys)["seed"]
              .agg(seed_ids=lambda x:tuple(sorted(set(x))),n_seeds="nunique").reset_index())
        if focus_seeds is not None:
            req=tuple(focus_seeds)
            info=info[info.seed_ids.map(lambda x:x==req)].copy()
            if info.empty:
                msg=f"{name}: no candidates contain every requested replicate {req}."
                if name=="ASSAY": raise ValueError(msg)
                return df.iloc[0:0].copy(),info
        else:
            info=info[info.n_seeds>=min_seeds].copy()
            valid=[]
            for method in method_prefixes:
                s=info[info.method==method]
                if s.empty: continue
                sets=s.seed_ids.drop_duplicates().tolist()
                if len(sets)>1: raise ValueError(f"{name}: method '{method}' has inconsistent replicate sets: {sets}")
                valid.append(s)
            info=pd.concat(valid,ignore_index=True) if valid else pd.DataFrame()
        valid=set(map(tuple,info[candidate_keys].to_numpy())) if not info.empty else set()
        return df[df[candidate_keys].apply(tuple,axis=1).isin(valid)].copy(),info

    assay_seed_df,assay_info=_eligible(assay_seed_df,"ASSAY")
    dual_seed_df,dual_info=_eligible(dual_seed_df,"DUAL")
    assay_methods=[m for m in method_prefixes if (assay_seed_df.method==m).any()]
    dual_methods=[m for m in method_prefixes if not dual_seed_df.empty and (dual_seed_df.method==m).any()]

    if (dual_selection_modes or include_common_balanced):
        missing=[m for m in assay_methods if m not in dual_methods]
        if missing: raise ValueError(f"Methods {missing} have no eligible dual candidates.")

    # Shared lambda values across methods for common-scale metrics.
    if dual_methods:
        present=(dual_seed_df.groupby("dr_lambda")["method"].agg(lambda x:set(x)).reset_index(name="methods"))
        common_lambdas=set(present.loc[present.methods.map(lambda x:x==set(dual_methods)),"dr_lambda"])
        dual_seed_df["common_candidate"]=dual_seed_df.dr_lambda.isin(common_lambdas)
    else:
        dual_seed_df["common_candidate"]=False
    common_seed_df=dual_seed_df[dual_seed_df.common_candidate].copy()

    if (include_common_balanced or net_benefit_metric=="net_benefit_normalized_common") and common_seed_df.empty:
        raise ValueError("No lambda values are shared across all methods.")

    # Method-specific normalization, PER REPLICATE.
    normalization_log={}
    if not dual_seed_df.empty:
        dual_seed_df["retention_raw"]=dual_seed_df["ms_performance"]/ms_base
        dual_seed_df["retention"]=dual_seed_df["retention_raw"].clip(0,1)
        dual_seed_df["adaptation"]=np.nan

        for method in dual_methods:
            normalization_log[method]={}
            md=dual_seed_df[dual_seed_df.method==method]
            for seed in sorted(md.seed.unique()):
                idx=md.index[md.seed==seed]; mx=float(dual_seed_df.loc[idx,"assay_performance"].max()); den=mx-assay_base
                dual_seed_df.loc[idx,"adaptation"]=0 if den<=0 else ((dual_seed_df.loc[idx,"assay_performance"]-assay_base)/den).clip(0,1)
                normalization_log[method][seed]={"assay_base":assay_base,"assay_seed_max":mx,"adaptation_denominator":den}

        a,r=dual_seed_df["adaptation"],dual_seed_df["retention"]
        dual_seed_df["balanced_harmonic"]=np.where(a+r>0,2*a*r/(a+r),0)
        dual_seed_df["balanced_geometric"]=np.sqrt(a*r)
        dual_seed_df["balanced_arithmetic"]=(a+r)/2
        dual_seed_df["balanced_min"]=np.minimum(a,r)
        dual_seed_df["net_benefit_normalized"]=a+r-1

    # Common cross-method normalization, also PER REPLICATE.
    dual_seed_df["adaptation_common"]=np.nan
    common_normalization_log={}
    if not common_seed_df.empty:
        for seed in sorted(common_seed_df.seed.unique()):
            idx=dual_seed_df.index[(dual_seed_df.seed==seed)&dual_seed_df.common_candidate]
            mx=float(dual_seed_df.loc[idx,"assay_performance"].max()); den=mx-assay_base
            dual_seed_df.loc[idx,"adaptation_common"]=0 if den<=0 else ((dual_seed_df.loc[idx,"assay_performance"]-assay_base)/den).clip(0,1)
            common_normalization_log[seed]={"assay_base":assay_base,"assay_seed_max_common":mx,"adaptation_denominator_common":den}

        a,r=dual_seed_df["adaptation_common"],dual_seed_df["retention"]
        dual_seed_df["balanced_harmonic_common"]=np.where(a.notna()&(a+r>0),2*a*r/(a+r),np.nan)
        dual_seed_df["balanced_geometric_common"]=np.sqrt(a*r)
        dual_seed_df["balanced_arithmetic_common"]=(a+r)/2
        dual_seed_df["balanced_min_common"]=np.minimum(a,r)
        dual_seed_df["net_benefit_normalized_common"]=a+r-1
    else:
        for c in ["balanced_harmonic_common","balanced_geometric_common","balanced_arithmetic_common",
                  "balanced_min_common","net_benefit_normalized_common"]:
            dual_seed_df[c]=np.nan

    bal_col=f"balanced_{balanced_metric}"
    bal_common_col=f"{bal_col}_common"
    dual_seed_df["balanced_score"]=dual_seed_df[bal_col]
    dual_seed_df["balanced_score_common"]=dual_seed_df[bal_common_col]

    # Aggregate only AFTER all replicate-level metrics exist.
    assay_candidate_df=(assay_seed_df.groupby(candidate_keys,sort=False)
        .agg(lambda_string=("lambda_string","first"),
             assay_only_performance_mean=("assay_performance","mean"),
             assay_only_performance_std=("assay_performance","std"),
             assay_only_performance_var=("assay_performance","var"),
             assay_only_n_seeds=("seed","nunique"),
             assay_only_seed_ids=("seed",lambda x:tuple(sorted(set(x)))),
             assay_only_score_cols_by_seed=("score_col",lambda x:tuple(x))).reset_index())
    assay_candidate_df["assay_only_performance_se"]=assay_candidate_df.assay_only_performance_std/np.sqrt(assay_candidate_df.assay_only_n_seeds)

    metrics=["assay_performance","ms_performance","assay_gain","ms_change","ms_loss","net_benefit",
             "assay_available_gain_fraction","ms_relative_change","brnb","adaptation","retention_raw","retention",
             "balanced_harmonic","balanced_geometric","balanced_arithmetic","balanced_min","balanced_score",
             "net_benefit_normalized","adaptation_common","balanced_harmonic_common","balanced_geometric_common",
             "balanced_arithmetic_common","balanced_min_common","balanced_score_common","net_benefit_normalized_common"]

    if not dual_seed_df.empty:
        dual_candidate_df=dual_seed_df.groupby(candidate_keys,sort=False)[metrics].agg(["mean","std","var","count"])
        dual_candidate_df.columns=["_".join(x) for x in dual_candidate_df.columns]
        dual_candidate_df=dual_candidate_df.reset_index().rename(columns={"assay_performance_count":"n_seeds"})
        for metric in metrics:
            c=f"{metric}_count"
            if c in dual_candidate_df and c!="assay_performance_count": dual_candidate_df.drop(columns=c,inplace=True)
            dual_candidate_df[f"{metric}_se"]=dual_candidate_df[f"{metric}_std"]/np.sqrt(dual_candidate_df.n_seeds)

        raw=[]
        for keys,g in dual_seed_df.groupby(candidate_keys,sort=False):
            g=g.sort_values("seed"); row=dict(zip(candidate_keys,keys))
            row.update({"lambda_string":g.lambda_string.iloc[0],"seed_ids":tuple(g.seed),
                        "common_candidate":bool(g.common_candidate.all()),
                        "score_cols_by_seed":dict(zip(g.seed,g.score_col)),
                        **{f"{m}_by_seed":dict(zip(g.seed,g[m])) for m in metrics}})
            raw.append(row)
        dual_candidate_df=dual_candidate_df.merge(pd.DataFrame(raw),on=candidate_keys,how="left")
    else:
        dual_candidate_df=pd.DataFrame(columns=candidate_keys)

    candidate_df=assay_candidate_df.merge(dual_candidate_df,on=candidate_keys,how="left",suffixes=("","_dual"))
    if "lambda_string_dual" in candidate_df:
        candidate_df["lambda_string"]=candidate_df.lambda_string.fillna(candidate_df.lambda_string_dual)
        candidate_df.drop(columns="lambda_string_dual",inplace=True)
    candidate_df["assay_candidate"]=True
    candidate_df["dual_candidate"]=candidate_df["ms_performance_mean"].notna()
    candidate_df["common_candidate"]=candidate_df["common_candidate"].fillna(False)

    for s in ["mean","std","var","se"]:
        candidate_df[f"assay_performance_{s}"]=candidate_df.get(f"assay_performance_{s}",candidate_df[f"assay_only_performance_{s}"])
        candidate_df[f"assay_performance_{s}"]=candidate_df[f"assay_performance_{s}"].fillna(candidate_df[f"assay_only_performance_{s}"])
    candidate_df["n_seeds"]=candidate_df["n_seeds"].fillna(candidate_df.assay_only_n_seeds)

    # Pareto is method-specific.
    candidate_df["pareto_dominated"]=False
    candidate_df["pareto_dominated_common"]=False

    def _pareto(mask,out):
        for method in dual_methods:
            idx=candidate_df.index[(candidate_df.method==method)&mask]
            vals=candidate_df.loc[idx,["assay_performance_mean","ms_performance_mean"]].to_numpy()
            dominated=[]
            for i,(a,m) in enumerate(vals):
                dominated.append(any(j!=i and aa>=a and mm>=m and (aa>a or mm>m) for j,(aa,mm) in enumerate(vals)))
            candidate_df.loc[idx,out]=dominated

    _pareto(candidate_df.dual_candidate,"pareto_dominated")
    _pareto(candidate_df.common_candidate,"pareto_dominated_common")
    candidate_df["pareto_optimal"]=candidate_df.dual_candidate&~candidate_df.pareto_dominated
    candidate_df["pareto_optimal_common"]=candidate_df.common_candidate&~candidate_df.pareto_dominated_common

    selected=[]; selection_log={}
    def _se_stage(pool,metric):
        if pool.empty: raise ValueError(f"Empty candidate pool for '{metric}'.")
        mc,sc=f"{metric}_mean",f"{metric}_se"; best=pool.loc[pool[mc].idxmax()]
        if not np.isfinite(best[sc]): raise ValueError(f"SE for best '{metric}' is not finite; >=2 replicates are required.")
        thr=float(best[mc]-selection_se_multiplier*best[sc])
        return best,thr,pool[pool[mc]>=thr].copy()

    def _resolve(pool):
        if pool.empty: raise ValueError("Empty final lambda pool.")
        pool=pool.sort_values("dr_lambda").copy()
        if lambda_tie_break=="all": return pool
        lam=pool.dr_lambda.min() if lambda_tie_break=="lowest" else pool.dr_lambda.max()
        return pool[pool.dr_lambda==lam].iloc[[0]].copy()

    nb_types={"net_benefit":"assay_ms_net_benefit","brnb":"assay_ms_brnb",
              "net_benefit_normalized":"assay_ms_net_benefit_normalized",
              "net_benefit_normalized_common":"assay_ms_net_benefit_normalized_common"}

    for method in assay_methods:
        all_assay=candidate_df[candidate_df.method==method].copy()
        for s in ["mean","std","var","se"]: all_assay[f"assay_performance_{s}"]=all_assay[f"assay_only_performance_{s}"]
        all_assay["n_seeds"]=all_assay.assay_only_n_seeds
        all_dual=candidate_df[(candidate_df.method==method)&candidate_df.dual_candidate].copy()
        pareto=all_dual[~all_dual.pareto_dominated].copy()
        common=all_dual[all_dual.common_candidate].copy()
        common_pareto=common[~common.pareto_dominated_common].copy()

        selection_log[method]={"focus_seeds":focus_seeds,"lambda_tie_break":lambda_tie_break,
            "balanced_metric":balanced_metric,"normalization":normalization_log.get(method,{}),
            "common_normalization":common_normalization_log,"selection_se_multiplier":selection_se_multiplier}

        best,thr,pool=_se_stage(all_assay,"assay_performance"); final=_resolve(pool); final["selection_type"]="assay_only"
        selection_log[method]["assay_only"]={"best_candidate":best.to_frame().T,"threshold":thr,
                                             "eligible_candidates":pool.copy(),"final_selection":final.copy()}
        selected.append(final)
        if all_dual.empty: continue

        if "balanced" in dual_selection_modes:
            best,thr,pool=_se_stage(pareto,"balanced_score"); final=_resolve(pool); final["selection_type"]="assay_ms_balanced"
            selection_log[method]["assay_ms_balanced"]={"metric":bal_col,"pareto_candidates":pareto.copy(),
                "best_candidate":best.to_frame().T,"threshold":thr,"eligible_candidates":pool.copy(),"final_selection":final.copy()}
            selected.append(final)

        if "bounded_assay_then_ms" in dual_selection_modes:
            b1,t1,p1=_se_stage(all_dual,"assay_performance"); b2,t2,p2=_se_stage(p1,"ms_performance")
            final=_resolve(p2); final["selection_type"]="assay_ms_bounded_assay_first"
            selection_log[method]["assay_ms_bounded_assay_first"]={
                "stage1_best":b1.to_frame().T,"stage1_threshold":t1,"stage1_candidates":p1.copy(),
                "stage2_best":b2.to_frame().T,"stage2_threshold":t2,"stage2_candidates":p2.copy(),
                "final_selection":final.copy()}
            selected.append(final)

        if "bounded_ms_then_assay" in dual_selection_modes:
            b1,t1,p1=_se_stage(all_dual,"ms_performance"); b2,t2,p2=_se_stage(p1,"assay_performance")
            final=_resolve(p2); final["selection_type"]="assay_ms_bounded_ms_first"
            selection_log[method]["assay_ms_bounded_ms_first"]={
                "stage1_best":b1.to_frame().T,"stage1_threshold":t1,"stage1_candidates":p1.copy(),
                "stage2_best":b2.to_frame().T,"stage2_threshold":t2,"stage2_candidates":p2.copy(),
                "final_selection":final.copy()}
            selected.append(final)

        if "net_benefit" in dual_selection_modes:
            pool0=common_pareto if net_benefit_metric=="net_benefit_normalized_common" else pareto
            best,thr,pool=_se_stage(pool0,net_benefit_metric); final=_resolve(pool)
            st=nb_types[net_benefit_metric]; final["selection_type"]=st
            selection_log[method][st]={"metric":net_benefit_metric,"pareto_candidates":pool0.copy(),
                "best_candidate":best.to_frame().T,"threshold":thr,"eligible_candidates":pool.copy(),
                "final_selection":final.copy()}
            selected.append(final)

        if include_common_balanced:
            if common_pareto.empty: raise ValueError(f"Method '{method}' has no common Pareto candidates.")
            best,thr,pool=_se_stage(common_pareto,"balanced_score_common"); final=_resolve(pool)
            final["selection_type"]="assay_ms_balanced_common"
            selection_log[method]["assay_ms_balanced_common"]={"metric":bal_common_col,
                "pareto_candidates":common_pareto.copy(),"best_candidate":best.to_frame().T,
                "threshold":thr,"eligible_candidates":pool.copy(),"final_selection":final.copy()}
            selected.append(final)

    selected_df=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame()

    if verbose and not selected_df.empty:
        print(f"\n{'='*105}\nFINAL LAMBDA SELECTIONS\n{'='*105}")
        cols=["method","selection_type","lambda_string","dr_lambda","learning_rate","step_budget",
              "assay_performance_mean","ms_performance_mean","net_benefit_mean","brnb_mean",
              "adaptation_mean","retention_mean","balanced_score_mean"]
        print(selected_df[[c for c in cols if c in selected_df]].to_string(index=False))

    return candidate_df,selected_df,selection_log
def plot_lambda_dfs(
    lambda_dfs, x, y, method=None, error=None, reverse_x=False, discrete_x=False,
    log_x=False, scientific_x=False, marker="o", xlabel=None, ylabel=None, title=None,
    fig_size=(7,5), dpi=300, linewidth=1.5, marker_size=6, capsize=3, show_plot=True,
):
    if discrete_x and log_x: raise ValueError("Choose either discrete_x=True or log_x=True.")
    if error not in {None,"sd","se","ci95"}: raise ValueError("error must be None, 'sd', 'se', or 'ci95'.")

    frames=[]
    for dataset,df in lambda_dfs.items():
        d=df.copy()
        if method is not None: d=d[d["method"]==method]
        if not d.empty: d["dataset"]=dataset; frames.append(d)
    if not frames: raise ValueError("No rows remain after filtering.")

    data=pd.concat(frames,ignore_index=True)
    datasets=list(dict.fromkeys(data["dataset"]))
    methods=list(dict.fromkeys(data["method"]))
    vals=sorted(data[x].dropna().unique(),reverse=reverse_x)

    def _fmt(v):
        if float(v)==0: return "0"
        m,e=f"{float(v):.12e}".split("e")
        return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

    err_col=f"{y}_{error}" if error else None
    if err_col and err_col not in data.columns: raise ValueError(f"Error column '{err_col}' not found.")

    fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)
    cmap=plt.get_cmap("tab10")
    colors={ds:cmap(i%10) for i,ds in enumerate(datasets)}
    linestyles=["-","--","-.",":"]

    for mi,m in enumerate(methods):
        for ds in datasets:
            g=data[(data["method"]==m)&(data["dataset"]==ds)].sort_values(x,ascending=not reverse_x)
            if g.empty: continue
            label=ds if len(methods)==1 else f"{ds} | {m}"
            xx=[{v:i for i,v in enumerate(vals)}[v] for v in g[x]] if discrete_x else g[x]
            kw=dict(marker=marker,linewidth=linewidth,markersize=marker_size,color=colors[ds],
                    linestyle=linestyles[mi%len(linestyles)],label=label)
            if error: ax.errorbar(xx,g[y],yerr=g[err_col],capsize=capsize,**kw)
            else: ax.plot(xx,g[y],**kw)

    if discrete_x:
        ax.set_xticks(range(len(vals)),[_fmt(v) if scientific_x else str(v) for v in vals])
    else:
        if log_x:
            positive=[v for v in vals if v>0]
            if not positive: raise ValueError("log_x requires positive x values.")
            ax.set_xscale("symlog",linthresh=min(positive)/10) if 0 in vals else ax.set_xscale("log")
        if scientific_x: ax.set_xticks(vals); ax.set_xticklabels([_fmt(v) for v in vals])
        if reverse_x: ax.invert_xaxis()

    ax.set_xlabel(x if xlabel is None else xlabel); ax.set_ylabel(y if ylabel is None else ylabel)
    if title is not None: ax.set_title(title)
    ax.legend(frameon=False); fig.tight_layout()
    if show_plot: plt.show()
    return fig,ax


def analyze_pepcl_components(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes, dr_lambda=1e-5,
    ref_col="mhcprime", steps=None, learning_rate=None, focus_seeds=None,
    n_boot=10000, ci=0.95, random_state=42, verbose=True,
):
    import re
    import numpy as np
    import pandas as pd
    from scipy import stats

    """
    2x2 PepCL component ablation:

                        DR OFF                  DR ON
      Replay OFF       pepcl_dr, λ=0           pepcl_dr, λ=dr_lambda
      Replay ON        pepcl_r_dr, λ=0         pepcl_r_dr, λ=dr_lambda

      = FT              = DR only
      = Replay only     = Full PepCL

    Lambda identity is determined from the parsed model-name lambda string,
    NOT approximate floating-point comparison.

    Allele AP -> macro performance per replicate -> paired contrasts/effects.

    If ref_col is provided, MHCPrime-relative quantities are computed
    replicate by replicate. MS always uses ref_col because the MS validation
    set is fixed. Assay seeds 0-4 use ref_col, while CV seeds 5+ use
    ref_col_seed_<seed>. The deterministic MHCPrime row in summary_df is
    reported over the same replicate evaluation sets.

    Pairwise component contrasts and factorial effects remain based on direct
    model-vs-model differences. These are mathematically unchanged by baseline
    correction because the matched replicate baseline cancels exactly.

    Returns
    -------
    replicate_df, summary_df, contrasts_df, factorial_df
    """

    def _lam(x):
        s=str(x)
        if s in {"0","0.0"}: return 0.
        if "eN" in s:
            a,b=s.split("eN"); return float(a)*10**(-int(b))
        return float(s)

    def _lam_string(x):
        s=str(x)
        if s in {"0","0.0"}: return "0"
        if "eN" in s: return s
        v=float(x)
        if v==0: return "0"
        m,e=f"{v:.12e}".split("e")
        m=m.rstrip("0").rstrip("."); e=int(e)
        return f"{m}eN{abs(e)}" if e<0 else f"{m}e{e}"

    def _boot_ci(x):
        x=np.asarray(x,float); x=x[np.isfinite(x)]
        if len(x)<2: return np.nan,np.nan
        rng=np.random.default_rng(random_state)
        b=x[rng.integers(0,len(x),(n_boot,len(x)))].mean(1)
        a=(1-ci)/2
        return tuple(np.quantile(b,[a,1-a]))

    def _t_ci(x):
        x=np.asarray(x,float); x=x[np.isfinite(x)]
        if len(x)<2: return np.nan,np.nan
        mean=x.mean()
        se=stats.sem(x)
        tcrit=stats.t.ppf(1-(1-ci)/2,df=len(x)-1)
        return mean-tcrit*se,mean+tcrit*se

    if not {"pepcl_dr","pepcl_r_dr"}.issubset(method_prefixes):
        raise ValueError("method_prefixes must contain 'pepcl_dr' and 'pepcl_r_dr'.")

    target_lambda_string=_lam_string(dr_lambda)
    if target_lambda_string=="0":
        raise ValueError("dr_lambda must be nonzero; lambda 0 is reserved for DR-OFF cells.")

    ordered=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

    def _method(c):
        for name,prefix in ordered:
            if str(c).startswith(prefix): return name
        return None

    pattern=re.compile(
        r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lam>.+?)_seed_(?P<seed>\d+)$"
    )

    def _is_ref_col(c):
        c=str(c)
        return c==ref_col or c.startswith(f"{ref_col}_seed_")

    def _assay_ref_col_for_seed(seed):
        seed=int(seed)
        return ref_col if seed<=4 else f"{ref_col}_seed_{seed}"

    usable=[
        c for c in dict.fromkeys(score_cols)
        if not _is_ref_col(c) and c in ms_ap_df.columns and c in assay_ap_df.columns
    ]

    meta=[]
    for c in usable:
        method=_method(c); m=pattern.search(str(c))
        if method in {"pepcl_dr","pepcl_r_dr"} and m is not None:
            meta.append({
                "score_col":c,"method":method,"steps":int(m["steps"]),
                "learning_rate":m["lr"],"lambda_string":m["lam"],
                "lambda":_lam(m["lam"]),"seed":int(m["seed"])
            })

    meta=pd.DataFrame(meta)
    if meta.empty:
        raise ValueError("No PepCL lambda-sweep columns could be parsed.")

    if steps is not None:
        meta=meta[meta["steps"]==int(steps)]

    if learning_rate is not None:
        meta=meta[meta["learning_rate"]==str(learning_rate)]

    if focus_seeds is not None:
        focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in np.atleast_1d(focus_seeds))))
        meta=meta[meta["seed"].isin(focus_seeds)]

    # Exact experimental cells only: lambda 0 + chosen nonzero DR lambda.
    wanted=meta[
        meta["lambda_string"].isin({"0",target_lambda_string})
    ].copy()

    if wanted.empty:
        raise ValueError(
            f"No component-ablation columns found for lambda 0 and "
            f"{target_lambda_string}."
        )

    # One fixed LR/step operating point.
    hp=wanted[["steps","learning_rate"]].drop_duplicates()

    if len(hp)!=1:
        raise ValueError(
            f"Found {len(hp)} LR/step combinations. Supply steps= and/or learning_rate= "
            "to select one fixed operating point."
        )

    # Macro performance independently for assay and MS.
    assay_mean=assay_ap_df[wanted["score_col"]].mean(axis=0)
    ms_mean=ms_ap_df[wanted["score_col"]].mean(axis=0)

    rows=[]

    for _,r in wanted.iterrows():
        replay=int(r["method"]=="pepcl_r_dr")
        dr=int(r["lambda_string"]==target_lambda_string)

        if r["lambda_string"]=="0":
            dr=0

        config={
            (0,0):"FT",
            (1,0):"Replay only",
            (0,1):"DR only",
            (1,1):"Full PepCL",
        }[(replay,dr)]

        rows.append({
            **r.to_dict(),
            "replay":replay,
            "dr":dr,
            "configuration":config,
            "assay_performance":float(assay_mean[r["score_col"]]),
            "ms_performance":float(ms_mean[r["score_col"]]),
        })

    replicate_df=pd.DataFrame(rows)

    # Replicate-specific MHCPrime references.
    if ref_col is not None:
        if ref_col not in assay_ap_df.columns:
            raise ValueError(f"ref_col '{ref_col}' not found in assay_ap_df.")
        if ref_col not in ms_ap_df.columns:
            raise ValueError(f"ref_col '{ref_col}' not found in ms_ap_df.")

        ref_ms=float(pd.to_numeric(ms_ap_df[ref_col],errors="coerce").mean())
        assay_ref_by_seed={}
        assay_ref_col_by_seed={}

        for seed in sorted(replicate_df["seed"].unique()):
            seed=int(seed)
            assay_ref_col=_assay_ref_col_for_seed(seed)
            if assay_ref_col not in assay_ap_df.columns:
                raise ValueError(
                    f"Assay reference '{assay_ref_col}' required for seed {seed} is missing."
                )
            assay_ref_col_by_seed[seed]=assay_ref_col
            assay_ref_by_seed[seed]=float(
                pd.to_numeric(assay_ap_df[assay_ref_col],errors="coerce").mean()
            )

        replicate_df["assay_ref_col"]=replicate_df["seed"].map(assay_ref_col_by_seed)
        replicate_df["ms_ref_col"]=ref_col
        replicate_df["assay_ref_performance"]=replicate_df["seed"].map(assay_ref_by_seed).astype(float)
        replicate_df["ms_ref_performance"]=ref_ms
        replicate_df["assay_delta_vs_mhcprime"]=(
            replicate_df["assay_performance"]-replicate_df["assay_ref_performance"]
        )
        replicate_df["ms_delta_vs_mhcprime"]=(
            replicate_df["ms_performance"]-replicate_df["ms_ref_performance"]
        )
    else:
        assay_ref_by_seed={}
        assay_ref_col_by_seed={}
        ref_ms=np.nan

    # Exactly one row per replicate × factorial cell.
    key=["seed","replay","dr"]
    dup=replicate_df.duplicated(key,keep=False)

    if dup.any():
        raise ValueError(
            "Duplicate component cells found after exact lambda filtering:\n"
            f"{replicate_df.loc[dup,key+['lambda_string','score_col']]}"
        )

    expected={(0,0),(1,0),(0,1),(1,1)}
    valid=[]
    incomplete={}

    for seed,g in replicate_df.groupby("seed"):
        observed=set(map(tuple,g[["replay","dr"]].to_numpy()))

        if observed==expected:
            valid.append(seed)
        else:
            incomplete[seed]=observed

    replicate_df=replicate_df[
        replicate_df["seed"].isin(valid)
    ].copy()

    if replicate_df.empty:
        raise ValueError(
            f"No replicate contains all four required cells. Observed: {incomplete}"
        )

    # ------------------------------------------------------------
    # Four-cell summaries + optional deterministic MHCPrime reference
    #
    # Ablation-model CI = Student-t CI across replicate-level
    # performances.
    #
    # MHCPrime is deterministic, so no SD/SE/CI is assigned.
    # ------------------------------------------------------------
    summary=[]

    if ref_col is not None:
        valid_seeds=tuple(sorted(replicate_df["seed"].unique()))
        ref_assay_values=np.asarray(
            [assay_ref_by_seed[int(seed)] for seed in valid_seeds],dtype=float
        )
        ref_ms_values=np.asarray([ref_ms for _ in valid_seeds],dtype=float)

        summary.append({
            "configuration":"MHCPrime",
            "replay":np.nan,
            "dr":np.nan,
            "n_replicates":len(valid_seeds),
            "replicate_ids":valid_seeds,

            "assay_performance_mean":float(ref_assay_values.mean()),
            "assay_performance_sd":np.nan,
            "assay_performance_se":np.nan,
            "assay_performance_ci_low":np.nan,
            "assay_performance_ci_high":np.nan,

            "ms_performance_mean":float(ref_ms_values.mean()),
            "ms_performance_sd":np.nan,
            "ms_performance_se":np.nan,
            "ms_performance_ci_low":np.nan,
            "ms_performance_ci_high":np.nan,

            "assay_performance_by_seed":{
                int(seed):float(assay_ref_by_seed[int(seed)]) for seed in valid_seeds
            },
            "ms_performance_by_seed":{
                int(seed):float(ref_ms) for seed in valid_seeds
            },
            "assay_ref_col_by_seed":{
                int(seed):assay_ref_col_by_seed[int(seed)] for seed in valid_seeds
            },
            "ms_ref_col_by_seed":{
                int(seed):ref_col for seed in valid_seeds
            },

            "assay_delta_vs_mhcprime_mean":0.0,
            "assay_delta_vs_mhcprime_sd":0.0,
            "assay_delta_vs_mhcprime_se":0.0,
            "assay_delta_vs_mhcprime_ci_low":0.0,
            "assay_delta_vs_mhcprime_ci_high":0.0,

            "ms_delta_vs_mhcprime_mean":0.0,
            "ms_delta_vs_mhcprime_sd":0.0,
            "ms_delta_vs_mhcprime_se":0.0,
            "ms_delta_vs_mhcprime_ci_low":0.0,
            "ms_delta_vs_mhcprime_ci_high":0.0,
        })

    order=["FT","Replay only","DR only","Full PepCL"]

    for config in order:
        g=replicate_df[
            replicate_df["configuration"]==config
        ].sort_values("seed")

        row={
            "configuration":config,
            "replay":int(g["replay"].iloc[0]),
            "dr":int(g["dr"].iloc[0]),
            "n_replicates":g["seed"].nunique(),
            "replicate_ids":tuple(g["seed"]),
        }

        for metric in ["assay_performance","ms_performance"]:
            x=g[metric].to_numpy(float)
            sd=x.std(ddof=1)
            se=sd/np.sqrt(len(x))
            lo,hi=_t_ci(x)

            row.update({
                f"{metric}_mean":x.mean(),
                f"{metric}_sd":sd,
                f"{metric}_se":se,
                f"{metric}_ci_low":lo,
                f"{metric}_ci_high":hi,
            })

        row["assay_performance_by_seed"]=dict(zip(g["seed"],g["assay_performance"]))
        row["ms_performance_by_seed"]=dict(zip(g["seed"],g["ms_performance"]))

        if ref_col is not None:
            row["assay_ref_col_by_seed"]=dict(zip(g["seed"],g["assay_ref_col"]))
            row["ms_ref_col_by_seed"]=dict(zip(g["seed"],g["ms_ref_col"]))

            for metric in ["assay","ms"]:
                col=f"{metric}_delta_vs_mhcprime"
                x=g[col].to_numpy(float)
                sd=x.std(ddof=1)
                se=sd/np.sqrt(len(x))
                lo,hi=_t_ci(x)

                row.update({
                    f"{col}_mean":x.mean(),
                    f"{col}_sd":sd,
                    f"{col}_se":se,
                    f"{col}_ci_low":lo,
                    f"{col}_ci_high":hi,
                    f"{col}_by_seed":dict(zip(g["seed"],g[col])),
                })

        summary.append(row)

    summary_df=pd.DataFrame(summary)

    assay_w=replicate_df.pivot(
        index="seed",
        columns="configuration",
        values="assay_performance",
    )

    ms_w=replicate_df.pivot(
        index="seed",
        columns="configuration",
        values="ms_performance",
    )

    # ------------------------------------------------------------
    # Paired configuration contrasts
    # delta = model_a - model_b
    # ------------------------------------------------------------
    comparisons=[
        ("Full PepCL","FT"),
        ("Full PepCL","Replay only"),
        ("Full PepCL","DR only"),
        ("Replay only","DR only"),
    ]

    contrasts=[]

    for model_a,model_b in comparisons:
        da=assay_w[model_a]-assay_w[model_b]
        dm=ms_w[model_a]-ms_w[model_b]

        alo,ahi=_boot_ci(da)
        mlo,mhi=_boot_ci(dm)

        asd=da.std(ddof=1)
        msd=dm.std(ddof=1)

        contrasts.append({
            "comparison":f"{model_a} vs {model_b}",
            "model_a":model_a,
            "model_b":model_b,
            "n_replicates":len(da),

            "assay_delta_mean":da.mean(),
            "assay_delta_sd":asd,
            "assay_delta_se":asd/np.sqrt(len(da)),
            "assay_delta_ci_low":alo,
            "assay_delta_ci_high":ahi,

            "ms_delta_mean":dm.mean(),
            "ms_delta_sd":msd,
            "ms_delta_se":msd/np.sqrt(len(dm)),
            "ms_delta_ci_low":mlo,
            "ms_delta_ci_high":mhi,

            "mean_pareto_dominance":bool(
                da.mean()>=0
                and dm.mean()>=0
                and (da.mean()>0 or dm.mean()>0)
            ),

            "ci_supported_pareto_dominance":bool(
                alo>=0
                and mlo>=0
                and (alo>0 or mlo>0)
            ),
        })

    contrasts_df=pd.DataFrame(contrasts)

    # ------------------------------------------------------------
    # Paired 2×2 factorial contrasts
    # ------------------------------------------------------------
    def _factorial(wide,outcome):
        ft=wide["FT"]
        r=wide["Replay only"]
        d=wide["DR only"]
        full=wide["Full PepCL"]

        effects={
            "Replay":0.5*((r-ft)+(full-d)),
            "DR":0.5*((d-ft)+(full-r)),
            "Replay x DR":(full-r)-(d-ft),
        }

        rows=[]

        for name,x in effects.items():
            sd=x.std(ddof=1)
            lo,hi=_boot_ci(x)

            rows.append({
                "outcome":outcome,
                "effect":name,
                "n_replicates":len(x),
                "effect_mean":x.mean(),
                "effect_sd":sd,
                "effect_se":sd/np.sqrt(len(x)),
                "effect_ci_low":lo,
                "effect_ci_high":hi,
                "effect_by_replicate":dict(zip(x.index,x.values)),
            })

        return rows

    factorial_df=pd.DataFrame(
        _factorial(assay_w,"assay_performance")
        +_factorial(ms_w,"ms_performance")
    )

    if verbose:
        steps0=int(replicate_df["steps"].iloc[0])
        lr0=replicate_df["learning_rate"].iloc[0]
        seeds=tuple(sorted(replicate_df["seed"].unique()))

        print(
            f"\n{'='*90}\n"
            f"PEPCL COMPONENT ABLATION — 2x2 REPLAY × DR\n"
            f"{'='*90}"
        )

        print(
            f"Steps: {steps0} | LR: {lr0} | "
            f"DR lambda: {target_lambda_string}"
        )

        print(
            f"Matched replicates: n={len(seeds)} {seeds}"
        )

        print("\nCells:")
        if ref_col is not None:
            print(f"  MHCPrime    = {ref_col}")
        print("  FT          = pepcl_dr,   lambda 0")
        print("  Replay only = pepcl_r_dr, lambda 0")
        print(
            f"  DR only     = pepcl_dr,   "
            f"lambda {target_lambda_string}"
        )
        print(
            f"  Full PepCL  = pepcl_r_dr, "
            f"lambda {target_lambda_string}"
        )

        print(
            f"\n{'CONFIGURATION PERFORMANCE':-^90}"
        )

        cols=[
            "configuration",
            "replay",
            "dr",
            "assay_performance_mean",
            "assay_performance_se",
            "ms_performance_mean",
            "ms_performance_se",
        ]

        print(
            summary_df[cols].to_string(
                index=False
            )
        )

        print(
            f"\n{'PAIRED CONFIGURATION DIFFERENCES':-^90}"
        )

        cols=[
            "comparison",
            "assay_delta_mean",
            "assay_delta_ci_low",
            "assay_delta_ci_high",
            "ms_delta_mean",
            "ms_delta_ci_low",
            "ms_delta_ci_high",
            "mean_pareto_dominance",
            "ci_supported_pareto_dominance",
        ]

        print(
            contrasts_df[cols].to_string(
                index=False
            )
        )

        print(
            f"\n{'FACTORIAL COMPONENT EFFECTS':-^90}"
        )

        cols=[
            "outcome",
            "effect",
            "effect_mean",
            "effect_se",
            "effect_ci_low",
            "effect_ci_high",
        ]

        print(
            factorial_df[cols].to_string(
                index=False
            )
        )

        if incomplete:
            print(
                f"\nExcluded incomplete replicates: "
                f"{sorted(incomplete)}"
            )

    return replicate_df,summary_df,contrasts_df,factorial_df



import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import t as student_t


def _lambda_float(x):
    s=str(x)
    if s in {"0","0.0"}: return 0.0
    if "eN" in s:
        a,b=s.split("eN"); return float(a)*10**(-int(b))
    return float(s)


def _lambda_numeric_label(x):
    """Numeric lambda label: 0, 1e-8, 1e-7, ..., 1e-2."""
    v=_lambda_float(x)
    if v==0: return "0"
    m,e=f"{v:.12e}".split("e")
    m=m.rstrip("0").rstrip(".")
    e=int(e)
    return f"{m}e{e}"

def _build_lambda_trajectory_df(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes, ref_col="mhcprime",
    ms_name="MS", assay_name="Assay", ms_map_col="mAP", assay_map_col="mAP",
    method=None,
):
    def _get_map(df,map_col):
        if map_col in df.columns:
            s=df[map_col].copy(); s.index=s.index.astype(str)
            return s.astype(float)
        return df.mean(axis=0).astype(float)

    ms_map=_get_map(ms_ap_df,ms_map_col); assay_map=_get_map(assay_ap_df,assay_map_col)
    if ref_col not in ms_map.index or ref_col not in assay_map.index:
        raise ValueError(f"Reference '{ref_col}' must exist in both inputs.")

    methods=[method] if isinstance(method,str) else list(method) if method is not None else None
    ordered=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

    def _method(c):
        for name,prefix in ordered:
            if str(c).startswith(prefix): return name
        return None

    pattern=re.compile(r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lam>.+?)_seed_(?P<seed>\d+)$")
    usable=[c for c in dict.fromkeys(score_cols) if c in ms_map.index and c in assay_map.index]
    ref_ms=float(ms_map[ref_col]); ref_assay=float(assay_map[ref_col])

    rows=[]; bad=[]
    for c in usable:
        m=pattern.search(str(c)); meth=_method(c)
        if m is None or meth is None:
            if c!=ref_col: bad.append(c)
            continue
        if methods is not None and meth not in methods: continue

        lam=m["lam"]
        rows.append({
            "score_col":c,"method":meth,"steps":int(m["steps"]),"learning_rate":m["lr"],
            "dr_lambda":lam,"dr_lambda_float":_lambda_float(lam),"seed":int(m["seed"]),
            f"{ms_name}_mAP":float(ms_map[c]),f"{assay_name}_mAP":float(assay_map[c]),
            f"{ms_name}_mAP_delta":float(ms_map[c])-ref_ms,
            f"{assay_name}_mAP_delta":float(assay_map[c])-ref_assay,
        })

    raw=pd.DataFrame(rows)
    if raw.empty: raise ValueError("No valid lambda-sweep columns remain after parsing/filtering.")
    if bad: print(f"WARNING: skipped {len(bad)} unparsable columns.")

    groups=["method","steps","learning_rate","dr_lambda","dr_lambda_float"]
    values=[f"{ms_name}_mAP",f"{assay_name}_mAP",f"{ms_name}_mAP_delta",f"{assay_name}_mAP_delta"]

    mean=(raw.groupby(groups,as_index=False,sort=False)
          .agg(**{c:(c,"mean") for c in values},
               n_seeds=("seed","nunique"),
               seed_ids=("seed",lambda x:tuple(sorted(set(x))))))

    sd=raw.groupby(groups)[values].std().add_suffix("_sd").reset_index()
    out=mean.merge(sd,on=groups,how="left")

    for c in values:
        out[f"{c}_se"]=out[f"{c}_sd"]/np.sqrt(out["n_seeds"])
        crit=out["n_seeds"].apply(lambda n:student_t.ppf(.975,n-1) if n>1 else np.nan)
        out[f"{c}_ci95"]=crit*out[f"{c}_se"]

    return out.sort_values(["method","dr_lambda_float"]).reset_index(drop=True)


def _lambda_color_map(lambda_values,cmap="viridis",lambda_colors=None,colorbar_order="ascending"):
    if colorbar_order not in {"ascending","descending"}:
        raise ValueError("colorbar_order must be 'ascending' or 'descending'.")

    vals=sorted({_lambda_float(x):str(x) for x in lambda_values}.items(),
                reverse=colorbar_order=="descending")
    order=[s for _,s in vals]

    cm=plt.get_cmap(cmap) if isinstance(cmap,str) else cmap
    colors={s:cm(i/max(len(order)-1,1)) for i,s in enumerate(order)}

    if lambda_colors:
        for k,v in lambda_colors.items():
            kf=_lambda_float(k)
            for s in order:
                if _lambda_float(s)==kf: colors[s]=v

    return order,colors


def _make_lambda_colorbar(
    fig, ax_or_axes, lambda_order, lambda_colors,
    label="DR λ", orientation="vertical",
    pad=.03, fraction=.045, tick_fontsize=9,
    label_fontsize=10, outline_width=1.0,
):
    cmap=ListedColormap([lambda_colors[x] for x in lambda_order])
    norm=BoundaryNorm(np.arange(len(lambda_order)+1)-.5,len(lambda_order))
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])

    cb=fig.colorbar(
        sm,ax=ax_or_axes,ticks=np.arange(len(lambda_order)),
        orientation=orientation,pad=pad,fraction=fraction
    )
    labels=[_lambda_numeric_label(x) for x in lambda_order]

    if orientation=="vertical":
        cb.ax.set_yticklabels(labels)
    else:
        cb.ax.set_xticklabels(labels)

    cb.set_label(label,fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.outline.set_linewidth(outline_width)
    return cb


def _format_axis(
    ax, tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
    tick_fontsize=9, tick_length=4, tick_width=1.2, tick_pad=5,
    spine_width=1.4, grid=False, grid_alpha=.2,
):
    def _fmt(v,_):
        if abs(v)<10**(-(tick_decimals+2)): v=0
        s=f"{v:.{tick_decimals}f}"
        return s.rstrip("0").rstrip(".") if trim_tick_zeros else s

    if n_ticks is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.tick_params(axis="both",labelsize=tick_fontsize,length=tick_length,
                   width=tick_width,pad=tick_pad)

    for spine in ax.spines.values(): spine.set_linewidth(spine_width)

    if grid: ax.grid(True,alpha=grid_alpha)
    else: ax.grid(False)


def _draw_lambda_trajectory(
    ax, trajectory_df, ms_name="MS", assay_name="Assay", ref_col="mhcprime",
    lambda_order=None, lambda_colors=None, highlight_lambda=None,
    crosshair_lambda=None, xlabel=None, ylabel=None, title=None,
    marker_size=80, highlight_size=220, point_alpha=1.0,
    trajectory_line_kwargs=None, tradeoff_line=True, tradeoff_line_kwargs=None,
    zero_lines=True, zero_line_kwargs=None, crosshair_line_kwargs=None,
    show_ci95=True, ci_alpha=1.0, ci_kwargs=None,
    xlim=None, ylim=None, square=True,
    axis_label_fontsize=11, title_fontsize=12,
    tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
    tick_fontsize=9, tick_length=4, tick_width=1.2, tick_pad=5,
    spine_width=1.4, grid=False, grid_alpha=.2,
):
    x=f"{ms_name}_mAP_delta"; y=f"{assay_name}_mAP_delta"
    xci=f"{x}_ci95"; yci=f"{y}_ci95"

    trajectory_line_kwargs={"color":"0.55","linewidth":1.5,"alpha":1.0,"zorder":1,
                            **(trajectory_line_kwargs or {})}
    tradeoff_line_kwargs={"color":"0.7","linestyle":"--","linewidth":1.2,"alpha":1.0,"zorder":0,
                          **(tradeoff_line_kwargs or {})}
    zero_line_kwargs={"color":"0.75","linestyle":"--","linewidth":1.0,"alpha":1.0,"zorder":0,
                      **(zero_line_kwargs or {})}
    crosshair_line_kwargs={"color":"0.55","linestyle":"--","linewidth":1.2,"alpha":1.0,"zorder":0,
                           **(crosshair_line_kwargs or {})}
    ci_kwargs={"elinewidth":1.0,"capsize":2.5,"capthick":1.0,"zorder":2,
               **(ci_kwargs or {})}

    markers=["o","s","^","D","P","X"]
    methods=list(dict.fromkeys(trajectory_df["method"]))
    marker_map={m:markers[i%len(markers)] for i,m in enumerate(methods)}

    if tradeoff_line: ax.axline((0,0),slope=-1,**tradeoff_line_kwargs)
    if zero_lines:
        ax.axvline(0,**zero_line_kwargs)
        ax.axhline(0,**zero_line_kwargs)

    # Trajectory order must remain numerical λ order, regardless of colorbar display order.
    for method in methods:
        g=trajectory_df[trajectory_df["method"]==method].sort_values("dr_lambda_float")
        ax.plot(g[x],g[y],**trajectory_line_kwargs)

        for _,r in g.iterrows():
            c=lambda_colors[str(r["dr_lambda"])]

            if show_ci95 and np.isfinite(r.get(xci,np.nan)) and np.isfinite(r.get(yci,np.nan)):
                ax.errorbar(r[x],r[y],xerr=r[xci],yerr=r[yci],fmt="none",
                            ecolor=c,alpha=ci_alpha,**ci_kwargs)

            ax.scatter(
                r[x],r[y],s=marker_size,marker=marker_map[method],
                color=c,edgecolor="black",linewidth=.7,
                alpha=point_alpha,zorder=3
            )

    if highlight_lambda is not None:
        targets=highlight_lambda if isinstance(highlight_lambda,dict) else {m:highlight_lambda for m in methods}
        for method,lam in targets.items():
            trg=trajectory_df[
                (trajectory_df["method"]==method)&
                np.isclose(trajectory_df["dr_lambda_float"],_lambda_float(lam),rtol=1e-12,atol=0)
            ]
            for _,r in trg.iterrows():
                ax.scatter(
                    r[x],r[y],s=highlight_size,marker="*",
                    color=lambda_colors[str(r["dr_lambda"])],
                    edgecolor="black",linewidth=1.1,
                    alpha=point_alpha,zorder=6
                )

    if crosshair_lambda is not None:
        targets=crosshair_lambda if isinstance(crosshair_lambda,dict) else {m:crosshair_lambda for m in methods}
        for method,lam in targets.items():
            trg=trajectory_df[
                (trajectory_df["method"]==method)&
                np.isclose(trajectory_df["dr_lambda_float"],_lambda_float(lam),rtol=1e-12,atol=0)
            ]
            if trg.empty:
                print(f"WARNING: no crosshair target for method={method}, lambda={lam}")
                continue
            r=trg.iloc[0]
            ax.axvline(r[x],**crosshair_line_kwargs)
            ax.axhline(r[y],**crosshair_line_kwargs)

    ax.set_xlabel(xlabel or f"{ms_name} ΔmAP vs {ref_col}",fontsize=axis_label_fontsize,labelpad=8)
    ax.set_ylabel(ylabel or f"{assay_name} ΔmAP vs {ref_col}",fontsize=axis_label_fontsize,labelpad=8)
    if title is not None: ax.set_title(title,fontsize=title_fontsize)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if square: ax.set_box_aspect(1)

    _format_axis(
        ax,tick_decimals,trim_tick_zeros,n_ticks,tick_fontsize,
        tick_length,tick_width,tick_pad,spine_width,grid,grid_alpha
    )
    return ax


def plot_lambda_ap_trajectory(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes, ref_col="mhcprime",
    ms_name="MS", assay_name="Assay", ms_map_col="mAP", assay_map_col="mAP",
    method=None, highlight_lambda=None, crosshair_lambda=None,
    cmap="viridis", lambda_colors=None,
    colorbar=True, colorbar_label="DR λ", colorbar_order="ascending",
    colorbar_orientation="vertical", colorbar_pad=.03, colorbar_fraction=.045,
    colorbar_tick_fontsize=9, colorbar_label_fontsize=10, colorbar_outline_width=1.0,
    marker_size=80, highlight_size=220, point_alpha=1.0,
    trajectory_line_kwargs=None, tradeoff_line=True, tradeoff_line_kwargs=None,
    zero_lines=True, zero_line_kwargs=None, crosshair_line_kwargs=None,
    show_ci95=True, ci_alpha=1.0, ci_kwargs=None,
    xlabel=None, ylabel=None, title=None, xlim=None, ylim=None,
    fig_size=(6,6), dpi=300, square=True,
    axis_label_fontsize=11, title_fontsize=12,
    tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
    tick_fontsize=9, tick_length=4, tick_width=1.2, tick_pad=5,
    spine_width=1.4, grid=False, grid_alpha=.2,
    save_path=None, show_plot=True, ax=None,
):
    """
    Single lambda trajectory.

    colorbar_order:
      "ascending"  -> smallest lambda at bottom, largest at top
      "descending" -> largest lambda at bottom, smallest at top
    """
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=["Nimbus Sans","Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    traj=_build_lambda_trajectory_df(
        ms_ap_df,assay_ap_df,score_cols,method_prefixes,ref_col,
        ms_name,assay_name,ms_map_col,assay_map_col,method
    )
    order,colors=_lambda_color_map(
        traj["dr_lambda"],cmap,lambda_colors,colorbar_order=colorbar_order
    )

    if ax is None: fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)
    else: fig=ax.figure

    _draw_lambda_trajectory(
        ax,traj,ms_name,assay_name,ref_col,order,colors,
        highlight_lambda,crosshair_lambda,xlabel,ylabel,title,
        marker_size,highlight_size,point_alpha,
        trajectory_line_kwargs,tradeoff_line,tradeoff_line_kwargs,
        zero_lines,zero_line_kwargs,crosshair_line_kwargs,
        show_ci95,ci_alpha,ci_kwargs,xlim,ylim,square,
        axis_label_fontsize,title_fontsize,
        tick_decimals,trim_tick_zeros,n_ticks,
        tick_fontsize,tick_length,tick_width,tick_pad,
        spine_width,grid,grid_alpha
    )

    cb=None
    if colorbar:
        cb=_make_lambda_colorbar(
            fig,ax,order,colors,label=colorbar_label,
            orientation=colorbar_orientation,pad=colorbar_pad,
            fraction=colorbar_fraction,
            tick_fontsize=colorbar_tick_fontsize,
            label_fontsize=colorbar_label_fontsize,
            outline_width=colorbar_outline_width
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf",format="pdf",bbox_inches="tight",dpi=dpi)
        fig.savefig(f"{save_path}.png",format="png",bbox_inches="tight",dpi=dpi)

    if show_plot: plt.show()
    return traj,fig,ax,cb


def plot_lambda_ap_trajectory_grid(
    dataset_dfs, score_cols_by_dataset, method_prefixes_by_dataset,
    method="pepcl_r_dr", ref_col="mhcprime",
    ms_key="ms", assay_key="assay", ms_name="MS",
    assay_names=None, ms_map_col="mAP", assay_map_col="mAP",
    highlight_lambda=None, crosshair_lambda=None,
    cmap="viridis", lambda_colors=None,
    colorbar_label="DR λ", colorbar_order="ascending",
    colorbar_orientation="vertical", colorbar_pad=.02, colorbar_fraction=.025,
    colorbar_tick_fontsize=9, colorbar_label_fontsize=10, colorbar_outline_width=1.0,
    titles=None, xlabels=None, ylabels=None,
    xlims=None, ylims=None,
    marker_size=70, highlight_size=200, point_alpha=1.0,
    trajectory_line_kwargs=None, tradeoff_line=True, tradeoff_line_kwargs=None,
    zero_lines=True, zero_line_kwargs=None, crosshair_line_kwargs=None,
    show_ci95=True, ci_alpha=1.0, ci_kwargs=None,
    fig_size=(22,4.5), dpi=300, square=True,
    axis_label_fontsize=10, title_fontsize=11,
    tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
    tick_fontsize=8, tick_length=4, tick_width=1.2, tick_pad=4,
    spine_width=1.4, grid=False, grid_alpha=.2,
    wspace=.32, right=.92,
    suptitle=None, suptitle_fontsize=14,
    save_path=None, show_plot=True,
):
    """
    1 x N lambda-trajectory grid with one shared discrete numeric lambda colorbar.

    colorbar_order:
      "ascending"  -> smallest lambda at bottom, largest at top
      "descending" -> largest lambda at bottom, smallest at top
    """
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=["Nimbus Sans","Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    datasets=list(dataset_dfs)

    def _arg(x,key,default=None):
        return x.get(key,default) if isinstance(x,dict) else default if x is None else x

    trajectories={}
    for ds in datasets:
        if ms_key not in dataset_dfs[ds] or assay_key not in dataset_dfs[ds]:
            raise ValueError(f"dataset_dfs['{ds}'] must contain '{ms_key}' and '{assay_key}'.")

        assay_name=_arg(assay_names,ds,ds)
        trajectories[ds]=_build_lambda_trajectory_df(
            dataset_dfs[ds][ms_key],dataset_dfs[ds][assay_key],
            score_cols_by_dataset[ds],method_prefixes_by_dataset[ds],
            ref_col,ms_name,assay_name,ms_map_col,assay_map_col,method
        )

    all_lams=[]
    for d in trajectories.values(): all_lams.extend(d["dr_lambda"].tolist())

    order,colors=_lambda_color_map(
        all_lams,cmap,lambda_colors,colorbar_order=colorbar_order
    )

    fig,axes=plt.subplots(1,len(datasets),figsize=fig_size,dpi=dpi,squeeze=False)
    axes=axes.ravel()

    for ax,ds in zip(axes,datasets):
        assay_name=_arg(assay_names,ds,ds)

        _draw_lambda_trajectory(
            ax,trajectories[ds],ms_name,assay_name,ref_col,order,colors,
            _arg(highlight_lambda,ds),_arg(crosshair_lambda,ds),
            _arg(xlabels,ds,f"{ms_name} ΔmAP vs {ref_col}"),
            _arg(ylabels,ds,f"{assay_name} ΔmAP vs {ref_col}"),
            _arg(titles,ds,ds),
            marker_size,highlight_size,point_alpha,
            trajectory_line_kwargs,tradeoff_line,tradeoff_line_kwargs,
            zero_lines,zero_line_kwargs,crosshair_line_kwargs,
            show_ci95,ci_alpha,ci_kwargs,
            _arg(xlims,ds),_arg(ylims,ds),square,
            axis_label_fontsize,title_fontsize,
            tick_decimals,trim_tick_zeros,n_ticks,
            tick_fontsize,tick_length,tick_width,tick_pad,
            spine_width,grid,grid_alpha
        )

    cb=_make_lambda_colorbar(
        fig,axes.tolist(),order,colors,
        label=colorbar_label,
        orientation=colorbar_orientation,
        pad=colorbar_pad,fraction=colorbar_fraction,
        tick_fontsize=colorbar_tick_fontsize,
        label_fontsize=colorbar_label_fontsize,
        outline_width=colorbar_outline_width
    )

    if suptitle is not None:
        fig.suptitle(suptitle,fontsize=suptitle_fontsize)

    fig.subplots_adjust(wspace=wspace,right=right)

    if save_path:
        fig.savefig(f"{save_path}.pdf",format="pdf",bbox_inches="tight",dpi=dpi)
        fig.savefig(f"{save_path}.png",format="png",bbox_inches="tight",dpi=dpi)

    if show_plot: plt.show()
    return trajectories,fig,axes,cb

# def plot_lambda_assay_diagnostic_grid(
#     ms_ap_df, assay_ap_df, score_cols, method_prefixes, lambda_df=None,
#     ref_col="mhcprime", method="pepcl_r_dr",
#     ms_name="MS", assay_name="Assay", ms_map_col="mAP", assay_map_col="mAP",

#     # Metric panels.
#     metric1="ms_decrease_vs_base",
#     metric2="assay_gain_vs_base",
#     errors=("ci95","ci95","ci95"),

#     # Selected lambda / crosshairs.
#     highlight_lambda=None,
#     crosshair_lambda=None,
#     crosshair_modes=("both","both","both"),

#     # Lambda coloring / colorbar.
#     cmap="viridis", lambda_colors=None,
#     colorbar_label="DR λ", colorbar_order="ascending",
#     colorbar_position="pareto",   # "pareto" or "right"
#     colorbar_size=0.035,
#     colorbar_pad=0.035,
#     colorbar_tick_fontsize=9, colorbar_label_fontsize=10,
#     colorbar_outline_width=1.0,

#     # Pareto styling.
#     pareto_marker_size=80, highlight_size=220, point_alpha=1.0,
#     trajectory_line_kwargs=None,
#     tradeoff_line=True, tradeoff_line_kwargs=None,
#     zero_lines=True, zero_line_kwargs=None,
#     pareto_error_alpha=1.0, pareto_error_kwargs=None,

#     # Lambda-metric styling.
#     metric_line_colors=("0.15","0.15"),
#     metric_color_by_lambda=False,
#     metric_markers=("o","o"),
#     metric_marker_size=7,
#     metric_linewidth=1.8,
#     metric_linestyles=("-","-"),
#     metric_point_alpha=1.0,
#     metric_error_alpha=1.0,
#     metric_error_kwargs=None,

#     # Crosshair styling.
#     crosshair_line_kwargs=None,

#     # Per-panel labels/titles.
#     titles=None,
#     xlabels=None,
#     ylabels=None,
#     xlims=None,
#     ylims=None,

#     # General styling.
#     fig_size=(16,6), dpi=300,
#     square_panels=True,
#     axis_label_fontsize=11, title_fontsize=12,
#     tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
#     tick_fontsize=9, tick_length=4, tick_width=1.2, tick_pad=5,
#     spine_width=1.4, grid=False, grid_alpha=.2,
#     categorical_lambda_order="ascending",
#     wspace=.42,
#     middle_panel_shift=0.0,
#     right_margin=.96,

#     # Output.
#     save_path=None, show_plot=True,
# ):
#     """
#     Single-assay 1x3 lambda diagnostic:

#         [ adaptation-retention | lambda vs metric1 | lambda vs metric2 ]

#     Lambda on panels 2/3 is STRICTLY categorical:
#         0, 1e-8, 1e-7, ..., 1e-2
#     are equally spaced visually.

#     errors:
#         None / "sd" / "se" / "ci95"
#         scalar or length-3 corresponding to Pareto / metric1 / metric2.

#     crosshair_modes:
#         "none", "vertical", "horizontal", "both"
#         scalar or length-3.

#     colorbar_position:
#         "pareto" -> beside panel 1.
#         "right"  -> after panel 3.

#     colorbar_size:
#         Width as fraction of anchor-panel width.
#         Height is forced to exactly equal the anchor plotting-box height.

#     middle_panel_shift:
#         Horizontal shift of panel 2 as a fraction of its own width.
#         Positive values move panel 2 right.

#     metric_color_by_lambda:
#         False -> constant-colored markers/error bars on panels 2/3.
#         True  -> markers/error bars use the same lambda colors as Pareto.
#     """
#     import numpy as np
#     import matplotlib
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import ListedColormap, BoundaryNorm
#     from matplotlib.ticker import MaxNLocator, FuncFormatter

#     matplotlib.rcParams["font.family"]="sans-serif"
#     matplotlib.rcParams["font.sans-serif"]=["Nimbus Sans","Nimbus Sans L","Arial","Helvetica"]
#     matplotlib.rcParams["pdf.fonttype"]=42

#     valid_errors={None,"sd","se","ci95"}
#     valid_modes={"none","vertical","horizontal","both"}

#     if colorbar_position not in {"pareto","right"}:
#         raise ValueError("colorbar_position must be 'pareto' or 'right'.")

#     def _three(x,name):
#         if isinstance(x,str) or x is None: x=(x,x,x)
#         if len(x)!=3: raise ValueError(f"{name} must be scalar or length 3.")
#         return tuple(x)

#     def _arg(x,i,default=None):
#         if x is None: return default
#         if isinstance(x,dict): return x.get(i,x.get(["pareto","metric1","metric2"][i],default))
#         if isinstance(x,(list,tuple)): return x[i]
#         return x

#     errors=_three(errors,"errors")
#     crosshair_modes=_three(crosshair_modes,"crosshair_modes")

#     if any(e not in valid_errors for e in errors):
#         raise ValueError("errors entries must be None, 'sd', 'se', or 'ci95'.")
#     if any(m not in valid_modes for m in crosshair_modes):
#         raise ValueError("crosshair_modes entries must be 'none', 'vertical', 'horizontal', or 'both'.")
#     if categorical_lambda_order not in {"ascending","descending"}:
#         raise ValueError("categorical_lambda_order must be 'ascending' or 'descending'.")

#     # ------------------------------------------------------------------
#     # Build trajectory + summary.
#     # ------------------------------------------------------------------
#     traj=_build_lambda_trajectory_df(
#         ms_ap_df,assay_ap_df,score_cols,method_prefixes,
#         ref_col=ref_col,ms_name=ms_name,assay_name=assay_name,
#         ms_map_col=ms_map_col,assay_map_col=assay_map_col,
#         method=method,
#     )

#     if lambda_df is None:
#         lambda_df=summarize_lambda_performance(
#             ms_ap_df,assay_ap_df,score_cols,method_prefixes,
#             ref_col=ref_col,per_allele=False,
#         )

#     ld=lambda_df.copy()
#     if "method" in ld.columns: ld=ld[ld["method"]==method].copy()
#     if ld.empty: raise ValueError(f"No lambda summary rows remain for method='{method}'.")

#     # ------------------------------------------------------------------
#     # Shared lambda colors + categorical lambda coordinates.
#     # ------------------------------------------------------------------
#     all_lams=list(traj["dr_lambda"])+list(ld["lambda"])
#     color_order,colors=_lambda_color_map(
#         all_lams,cmap=cmap,lambda_colors=lambda_colors,
#         colorbar_order=colorbar_order,
#     )

#     lambda_vals=sorted(
#         ld["lambda"].dropna().astype(float).unique(),
#         reverse=categorical_lambda_order=="descending"
#     )
#     xpos={float(v):i for i,v in enumerate(lambda_vals)}
#     lambda_tick_labels=[_lambda_numeric_label(v) for v in lambda_vals]

#     def _lambda_color(v):
#         key=min(color_order,key=lambda z:abs(_lambda_float(z)-float(v)))
#         return colors[key]

#     # ------------------------------------------------------------------
#     # Three plotting panels.
#     # ------------------------------------------------------------------
#     fig,axes=plt.subplots(1,3,figsize=fig_size,dpi=dpi)
#     ax0,ax1,ax2=axes

#     trajectory_line_kwargs={
#         "color":"0.55","linewidth":1.5,"alpha":1.0,"zorder":1,
#         **(trajectory_line_kwargs or {})
#     }
#     tradeoff_line_kwargs={
#         "color":"0.7","linestyle":"--","linewidth":1.2,"alpha":1.0,"zorder":0,
#         **(tradeoff_line_kwargs or {})
#     }
#     zero_line_kwargs={
#         "color":"0.75","linestyle":"--","linewidth":1.0,"alpha":1.0,"zorder":0,
#         **(zero_line_kwargs or {})
#     }
#     crosshair_line_kwargs={
#         "color":"0.45","linestyle":"--","linewidth":1.2,"alpha":1.0,"zorder":0,
#         **(crosshair_line_kwargs or {})
#     }
#     pareto_error_kwargs={
#         "elinewidth":1.0,"capsize":2.5,"capthick":1.0,"zorder":2,
#         **(pareto_error_kwargs or {})
#     }
#     metric_error_kwargs={
#         "elinewidth":1.0,"capsize":3,"capthick":1.0,"zorder":2,
#         **(metric_error_kwargs or {})
#     }

#     # ==================================================================
#     # PANEL 1 — adaptation-retention
#     # ==================================================================
#     px=f"{ms_name}_mAP_delta"
#     py=f"{assay_name}_mAP_delta"

#     if tradeoff_line: ax0.axline((0,0),slope=-1,**tradeoff_line_kwargs)
#     if zero_lines:
#         ax0.axvline(0,**zero_line_kwargs)
#         ax0.axhline(0,**zero_line_kwargs)

#     g=traj.sort_values("dr_lambda_float")
#     ax0.plot(g[px],g[py],**trajectory_line_kwargs)

#     pareto_error=errors[0]
#     for _,r in g.iterrows():
#         c=_lambda_color(r["dr_lambda_float"])

#         if pareto_error is not None:
#             xe=f"{px}_{pareto_error}"
#             ye=f"{py}_{pareto_error}"
#             if xe not in g.columns or ye not in g.columns:
#                 raise ValueError(f"Pareto error columns '{xe}'/'{ye}' not found.")

#             if np.isfinite(r[xe]) and np.isfinite(r[ye]):
#                 ax0.errorbar(
#                     r[px],r[py],xerr=r[xe],yerr=r[ye],
#                     fmt="none",ecolor=c,alpha=pareto_error_alpha,
#                     **pareto_error_kwargs
#                 )

#         ax0.scatter(
#             r[px],r[py],s=pareto_marker_size,
#             color=c,edgecolor="black",linewidth=.7,
#             alpha=point_alpha,zorder=3
#         )

#     if highlight_lambda is not None:
#         target=_lambda_float(highlight_lambda)
#         hit=g[np.isclose(g["dr_lambda_float"],target,rtol=1e-12,atol=0)]

#         if hit.empty:
#             print(f"WARNING: lambda {highlight_lambda} not found in Pareto trajectory.")
#         else:
#             r=hit.iloc[0]
#             ax0.scatter(
#                 r[px],r[py],s=highlight_size,marker="*",
#                 color=_lambda_color(target),edgecolor="black",
#                 linewidth=1.1,alpha=point_alpha,zorder=6
#             )

#     pareto_cross_lam=crosshair_lambda if crosshair_lambda is not None else highlight_lambda
#     if pareto_cross_lam is not None:
#         target=_lambda_float(pareto_cross_lam)
#         hit=g[np.isclose(g["dr_lambda_float"],target,rtol=1e-12,atol=0)]

#         if not hit.empty:
#             r=hit.iloc[0]; mode=crosshair_modes[0]
#             if mode in {"vertical","both"}: ax0.axvline(r[px],**crosshair_line_kwargs)
#             if mode in {"horizontal","both"}: ax0.axhline(r[py],**crosshair_line_kwargs)

#     ax0.set_xlabel(
#         _arg(xlabels,0,f"{ms_name} ΔmAP vs {ref_col}"),
#         fontsize=axis_label_fontsize,labelpad=8
#     )
#     ax0.set_ylabel(
#         _arg(ylabels,0,f"{assay_name} ΔmAP vs {ref_col}"),
#         fontsize=axis_label_fontsize,labelpad=8
#     )
#     ax0.set_title(_arg(titles,0,"Adaptation–retention"),fontsize=title_fontsize)

#     if _arg(xlims,0) is not None: ax0.set_xlim(_arg(xlims,0))
#     if _arg(ylims,0) is not None: ax0.set_ylim(_arg(ylims,0))

#     _format_axis(
#         ax0,tick_decimals,trim_tick_zeros,n_ticks,
#         tick_fontsize,tick_length,tick_width,tick_pad,
#         spine_width,grid,grid_alpha
#     )

#     # ==================================================================
#     # PANELS 2/3 — categorical lambda trajectories
#     # ==================================================================
#     metrics=[metric1,metric2]

#     if isinstance(metric_line_colors,str):
#         metric_line_colors=(metric_line_colors,metric_line_colors)
#     if isinstance(metric_markers,str):
#         metric_markers=(metric_markers,metric_markers)
#     if isinstance(metric_linestyles,str):
#         metric_linestyles=(metric_linestyles,metric_linestyles)

#     for j,(ax,metric) in enumerate(zip([ax1,ax2],metrics),start=1):
#         if metric not in ld.columns:
#             raise ValueError(f"Metric '{metric}' not found in lambda_df.")

#         d=ld.sort_values(
#             "lambda",
#             ascending=categorical_lambda_order=="ascending"
#         ).copy()

#         xx=np.array([xpos[float(v)] for v in d["lambda"]],dtype=float)
#         yy=d[metric].to_numpy(float)

#         # Constant-color connecting line.
#         ax.plot(
#             xx,yy,
#             color=metric_line_colors[j-1],
#             linewidth=metric_linewidth,
#             linestyle=metric_linestyles[j-1],
#             alpha=metric_point_alpha,
#             zorder=2
#         )

#         # Points.
#         for x_i,(_,r) in zip(xx,d.iterrows()):
#             c=_lambda_color(r["lambda"]) if metric_color_by_lambda else metric_line_colors[j-1]

#             ax.scatter(
#                 x_i,r[metric],
#                 s=metric_marker_size**2,
#                 marker=metric_markers[j-1],
#                 color=c,edgecolor="black",linewidth=.6,
#                 alpha=metric_point_alpha,zorder=4
#             )

#         # Error bars.
#         err=errors[j]
#         err_col=f"{metric}_{err}" if err is not None else None

#         if err_col is not None:
#             if err_col not in d.columns:
#                 raise ValueError(f"Error column '{err_col}' not found in lambda_df.")

#             for x_i,(_,r) in zip(xx,d.iterrows()):
#                 if not np.isfinite(r[err_col]): continue

#                 c=_lambda_color(r["lambda"]) if metric_color_by_lambda else metric_line_colors[j-1]

#                 ax.errorbar(
#                     x_i,r[metric],yerr=r[err_col],
#                     fmt="none",ecolor=c,
#                     alpha=metric_error_alpha,
#                     **metric_error_kwargs
#                 )

#         # Highlight.
#         if highlight_lambda is not None:
#             target=_lambda_float(highlight_lambda)
#             hit=d[np.isclose(d["lambda"].astype(float),target,rtol=1e-12,atol=0)]

#             if not hit.empty:
#                 r=hit.iloc[0]
#                 hx=xpos[float(r["lambda"])]
#                 hy=float(r[metric])
#                 hc=_lambda_color(target) if metric_color_by_lambda else metric_line_colors[j-1]

#                 ax.scatter(
#                     hx,hy,s=highlight_size,marker="*",
#                     color=hc,edgecolor="black",linewidth=1.1,
#                     alpha=metric_point_alpha,zorder=6
#                 )

#         # Crosshair.
#         cross_lam=crosshair_lambda if crosshair_lambda is not None else highlight_lambda
#         if cross_lam is not None:
#             target=_lambda_float(cross_lam)
#             hit=d[np.isclose(d["lambda"].astype(float),target,rtol=1e-12,atol=0)]

#             if not hit.empty:
#                 r=hit.iloc[0]
#                 hx=xpos[float(r["lambda"])]
#                 hy=float(r[metric])
#                 mode=crosshair_modes[j]

#                 if mode in {"vertical","both"}:
#                     ax.axvline(hx,**crosshair_line_kwargs)
#                 if mode in {"horizontal","both"}:
#                     ax.axhline(hy,**crosshair_line_kwargs)

#         # Pure categorical lambda x-axis.
#         ax.set_xticks(np.arange(len(lambda_vals)))
#         ax.set_xticklabels(lambda_tick_labels,fontsize=tick_fontsize,rotation=0)

#         if _arg(xlims,j) is not None:
#             ax.set_xlim(_arg(xlims,j))
#         else:
#             ax.set_xlim(-.35,len(lambda_vals)-.65)

#         ax.set_xlabel(
#             _arg(xlabels,j,"DR λ"),
#             fontsize=axis_label_fontsize,labelpad=8
#         )
#         ax.set_ylabel(
#             _arg(ylabels,j,metric),
#             fontsize=axis_label_fontsize,labelpad=8
#         )
#         ax.set_title(
#             _arg(titles,j,metric),
#             fontsize=title_fontsize
#         )

#         if _arg(ylims,j) is not None:
#             ax.set_ylim(_arg(ylims,j))

#         def _yfmt(v,_):
#             if abs(v)<10**(-(tick_decimals+2)): v=0
#             s=f"{v:.{tick_decimals}f}"
#             return s.rstrip("0").rstrip(".") if trim_tick_zeros else s

#         if n_ticks is not None:
#             ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

#         ax.yaxis.set_major_formatter(FuncFormatter(_yfmt))
#         ax.tick_params(
#             axis="both",labelsize=tick_fontsize,
#             length=tick_length,width=tick_width,pad=tick_pad
#         )

#         for spine in ax.spines.values():
#             spine.set_linewidth(spine_width)

#         if grid: ax.grid(True,alpha=grid_alpha)
#         else: ax.grid(False)

#     # ==================================================================
#     # FINAL PANEL GEOMETRY
#     # ==================================================================
#     if square_panels:
#         for ax in axes:
#             ax.set_box_aspect(1)

#     actual_right=right_margin if colorbar_position=="pareto" else min(right_margin,.91)
#     fig.subplots_adjust(wspace=wspace,right=actual_right)

#     # Force square-panel geometry to be finalized.
#     fig.canvas.draw()

#     # Visually compensate for colorbar occupying panel1 -> panel2 whitespace.
#     if middle_panel_shift:
#         p=ax1.get_position()
#         shift=p.width*float(middle_panel_shift)

#         ax1.set_position([
#             p.x0+shift,
#             p.y0,
#             p.width,
#             p.height,
#         ])

#     # Force updated positions to finalize before colorbar placement.
#     fig.canvas.draw()

#     # ==================================================================
#     # COLORBAR — constructed from FINAL physical axis dimensions.
#     # ==================================================================
#     anchor_ax=ax0 if colorbar_position=="pareto" else ax2
#     pos=anchor_ax.get_position()

#     cb_width=pos.width*float(colorbar_size)
#     cb_pad=pos.width*float(colorbar_pad)

#     cax=fig.add_axes([
#         pos.x1+cb_pad,
#         pos.y0,
#         cb_width,
#         pos.height,
#     ])

#     cm=ListedColormap([colors[x] for x in color_order])
#     norm=BoundaryNorm(np.arange(len(color_order)+1)-.5,len(color_order))
#     sm=plt.cm.ScalarMappable(cmap=cm,norm=norm)
#     sm.set_array([])

#     cb=fig.colorbar(
#         sm,cax=cax,
#         ticks=np.arange(len(color_order)),
#         spacing="uniform"
#     )

#     cb.set_ticks(
#         np.arange(len(color_order)),
#         labels=[_lambda_numeric_label(x) for x in color_order]
#     )
#     cb.set_label(colorbar_label,fontsize=colorbar_label_fontsize)
#     cb.ax.tick_params(labelsize=colorbar_tick_fontsize)
#     cb.outline.set_linewidth(colorbar_outline_width)

#     # ------------------------------------------------------------------
#     # Save/show.
#     # ------------------------------------------------------------------
#     if save_path:
#         fig.savefig(f"{save_path}.pdf",format="pdf",bbox_inches="tight",dpi=dpi)
#         fig.savefig(f"{save_path}.png",format="png",bbox_inches="tight",dpi=dpi)

#     if show_plot: plt.show()

#     return {
#         "trajectory_df":traj,
#         "lambda_df":ld,
#         "fig":fig,
#         "axes":axes,
#         "colorbar":cb,
#     }

def plot_lambda_assay_diagnostic_grid(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes, lambda_df=None,
    ref_col="mhcprime", method="pepcl_r_dr",
    ms_name="MS", assay_name="Assay", ms_map_col="mAP", assay_map_col="mAP",

    # Metric panels.
    metric1="ms_decrease_vs_base",
    metric2="assay_gain_vs_base",
    errors=("ci95","ci95","ci95"),

    # Selected lambda / crosshairs.
    highlight_lambda=None,
    crosshair_lambda=None,
    crosshair_modes=("both","both","both"),

    # Lambda coloring / colorbar.
    cmap="viridis", lambda_colors=None,
    colorbar_label="DR λ", colorbar_order="ascending",
    colorbar_position="pareto",   # "pareto" or "right"
    colorbar_size=0.035,
    colorbar_pad=0.035,
    colorbar_tick_fontsize=9, colorbar_label_fontsize=10,
    colorbar_outline_width=1.0,

    # Pareto styling.
    pareto_marker_size=80, highlight_size=220, point_alpha=1.0,
    trajectory_line_kwargs=None,
    tradeoff_line=True, tradeoff_line_kwargs=None,
    zero_lines=True, zero_line_kwargs=None,
    pareto_error_alpha=1.0, pareto_error_kwargs=None,

    # Lambda-metric styling.
    metric_line_colors=("0.15","0.15"),
    metric_color_by_lambda=False,
    metric_markers=("o","o"),
    metric_marker_size=7,
    metric_linewidth=1.8,
    metric_linestyles=("-","-"),
    metric_point_alpha=1.0,
    metric_error_alpha=1.0,
    metric_error_kwargs=None,

    # Crosshair styling.
    crosshair_line_kwargs=None,

    # Per-panel labels/titles.
    titles=None,
    xlabels=None,
    ylabels=None,
    xlims=None,
    ylims=None,

    # General styling.
    fig_size=(16,6), dpi=300,
    square_panels=True,
    axis_label_fontsize=11, title_fontsize=12,
    tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
    tick_fontsize=9, tick_length=4, tick_width=1.2, tick_pad=5,
    spine_width=1.4, grid=False, grid_alpha=.2,
    categorical_lambda_order="ascending",
    wspace=.42,
    middle_panel_shift=0.0,
    right_margin=.96,

    # Output.
    save_path=None, show_plot=True,
):
    """
    Single-assay 1x3 lambda diagnostic:

        [ adaptation-retention | lambda vs metric1 | lambda vs metric2 ]

    Lambda on panels 2/3 is STRICTLY categorical:
        0, 1e-8, 1e-7, ..., 1e-2
    are equally spaced visually.

    errors:
        None / "sd" / "se" / "ci95"
        scalar or length-3 corresponding to Pareto / metric1 / metric2.

    crosshair_modes:
        "none", "vertical", "horizontal", "both"
        scalar or length-3.

    colorbar_position:
        "pareto" -> beside panel 1.
        "right"  -> after panel 3.

    colorbar_size:
        Width as fraction of anchor-panel width.
        Height is forced to exactly equal the anchor plotting-box height.

    middle_panel_shift:
        Horizontal shift of panel 2 as a fraction of its own width.
        Positive values move panel 2 right.

    metric_color_by_lambda:
        False -> constant-colored markers/error bars on panels 2/3.
        True  -> markers/error bars use the same lambda colors as Pareto.

    Baseline handling
    -----------------
    All three panels are derived from the same lambda_df. When lambda_df is
    generated by summarize_lambda_performance, assay deltas use the
    MHCPrime reference matched to each replicate before across-seed
    aggregation, while MS uses the fixed MHCPrime reference. This prevents
    the Pareto panel from silently using different baseline semantics than
    panels 2/3.
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=["Nimbus Sans","Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    valid_errors={None,"sd","se","ci95"}
    valid_modes={"none","vertical","horizontal","both"}

    if colorbar_position not in {"pareto","right"}:
        raise ValueError("colorbar_position must be 'pareto' or 'right'.")

    def _three(x,name):
        if isinstance(x,str) or x is None: x=(x,x,x)
        if len(x)!=3: raise ValueError(f"{name} must be scalar or length 3.")
        return tuple(x)

    def _arg(x,i,default=None):
        if x is None: return default
        if isinstance(x,dict): return x.get(i,x.get(["pareto","metric1","metric2"][i],default))
        if isinstance(x,(list,tuple)): return x[i]
        return x

    errors=_three(errors,"errors")
    crosshair_modes=_three(crosshair_modes,"crosshair_modes")

    if any(e not in valid_errors for e in errors):
        raise ValueError("errors entries must be None, 'sd', 'se', or 'ci95'.")
    if any(m not in valid_modes for m in crosshair_modes):
        raise ValueError("crosshair_modes entries must be 'none', 'vertical', 'horizontal', or 'both'.")
    if categorical_lambda_order not in {"ascending","descending"}:
        raise ValueError("categorical_lambda_order must be 'ascending' or 'descending'.")

    # ------------------------------------------------------------------
    # Build one corrected summary and use it for ALL three panels.
    #
    # This intentionally avoids rebuilding panel 1 from raw AP data through
    # a separate trajectory helper. The supplied/recomputed lambda summary
    # already contains the replicate-aware MHCPrime corrections, so using
    # that same table for the Pareto panel and both lambda-metric panels
    # guarantees identical baseline semantics everywhere.
    # ------------------------------------------------------------------
    if lambda_df is None:
        lambda_df=summarize_lambda_performance(
            ms_ap_df,assay_ap_df,score_cols,method_prefixes,
            ref_col=ref_col,per_allele=False,
        )

    ld=lambda_df.copy()
    if "method" in ld.columns:
        ld=ld[ld["method"]==method].copy()
    if ld.empty:
        raise ValueError(f"No lambda summary rows remain for method='{method}'.")

    required_traj_cols={"lambda","ms_change_vs_base","assay_gain_vs_base"}
    missing_traj=required_traj_cols-set(ld.columns)
    if missing_traj:
        raise ValueError(
            "lambda_df is missing corrected trajectory columns: "
            f"{sorted(missing_traj)}"
        )

    # Backward-compatible trajectory dataframe expected by the plotting code.
    traj=ld.copy()
    traj["dr_lambda"]=traj["lambda"]
    traj["dr_lambda_float"]=traj["lambda"].astype(float)

    px=f"{ms_name}_mAP_delta"
    py=f"{assay_name}_mAP_delta"

    traj[px]=traj["ms_change_vs_base"]
    traj[py]=traj["assay_gain_vs_base"]

    # Map uncertainty columns from the corrected summary into the historical
    # trajectory column names used below.
    for err in ("sd","se","ci95"):
        ms_src=f"ms_change_vs_base_{err}"
        assay_src=f"assay_gain_vs_base_{err}"

        if ms_src in traj.columns:
            traj[f"{px}_{err}"]=traj[ms_src]
        if assay_src in traj.columns:
            traj[f"{py}_{err}"]=traj[assay_src]

    # ------------------------------------------------------------------
    # Shared lambda colors + categorical lambda coordinates.
    # ------------------------------------------------------------------
    all_lams=list(traj["dr_lambda"])+list(ld["lambda"])
    color_order,colors=_lambda_color_map(
        all_lams,cmap=cmap,lambda_colors=lambda_colors,
        colorbar_order=colorbar_order,
    )

    lambda_vals=sorted(
        ld["lambda"].dropna().astype(float).unique(),
        reverse=categorical_lambda_order=="descending"
    )
    xpos={float(v):i for i,v in enumerate(lambda_vals)}
    lambda_tick_labels=[_lambda_numeric_label(v) for v in lambda_vals]

    def _lambda_color(v):
        key=min(color_order,key=lambda z:abs(_lambda_float(z)-float(v)))
        return colors[key]

    # ------------------------------------------------------------------
    # Three plotting panels.
    # ------------------------------------------------------------------
    fig,axes=plt.subplots(1,3,figsize=fig_size,dpi=dpi)
    ax0,ax1,ax2=axes

    trajectory_line_kwargs={
        "color":"0.55","linewidth":1.5,"alpha":1.0,"zorder":1,
        **(trajectory_line_kwargs or {})
    }
    tradeoff_line_kwargs={
        "color":"0.7","linestyle":"--","linewidth":1.2,"alpha":1.0,"zorder":0,
        **(tradeoff_line_kwargs or {})
    }
    zero_line_kwargs={
        "color":"0.75","linestyle":"--","linewidth":1.0,"alpha":1.0,"zorder":0,
        **(zero_line_kwargs or {})
    }
    crosshair_line_kwargs={
        "color":"0.45","linestyle":"--","linewidth":1.2,"alpha":1.0,"zorder":0,
        **(crosshair_line_kwargs or {})
    }
    pareto_error_kwargs={
        "elinewidth":1.0,"capsize":2.5,"capthick":1.0,"zorder":2,
        **(pareto_error_kwargs or {})
    }
    metric_error_kwargs={
        "elinewidth":1.0,"capsize":3,"capthick":1.0,"zorder":2,
        **(metric_error_kwargs or {})
    }

    # ==================================================================
    # PANEL 1 — adaptation-retention
    # ==================================================================
    px=f"{ms_name}_mAP_delta"
    py=f"{assay_name}_mAP_delta"

    if tradeoff_line: ax0.axline((0,0),slope=-1,**tradeoff_line_kwargs)
    if zero_lines:
        ax0.axvline(0,**zero_line_kwargs)
        ax0.axhline(0,**zero_line_kwargs)

    g=traj.sort_values("dr_lambda_float")
    ax0.plot(g[px],g[py],**trajectory_line_kwargs)

    pareto_error=errors[0]
    for _,r in g.iterrows():
        c=_lambda_color(r["dr_lambda_float"])

        if pareto_error is not None:
            xe=f"{px}_{pareto_error}"
            ye=f"{py}_{pareto_error}"
            if xe not in g.columns or ye not in g.columns:
                raise ValueError(f"Pareto error columns '{xe}'/'{ye}' not found.")

            if np.isfinite(r[xe]) and np.isfinite(r[ye]):
                ax0.errorbar(
                    r[px],r[py],xerr=r[xe],yerr=r[ye],
                    fmt="none",ecolor=c,alpha=pareto_error_alpha,
                    **pareto_error_kwargs
                )

        ax0.scatter(
            r[px],r[py],s=pareto_marker_size,
            color=c,edgecolor="black",linewidth=.7,
            alpha=point_alpha,zorder=3
        )

    if highlight_lambda is not None:
        target=_lambda_float(highlight_lambda)
        hit=g[np.isclose(g["dr_lambda_float"],target,rtol=1e-12,atol=0)]

        if hit.empty:
            print(f"WARNING: lambda {highlight_lambda} not found in Pareto trajectory.")
        else:
            r=hit.iloc[0]
            ax0.scatter(
                r[px],r[py],s=highlight_size,marker="*",
                color=_lambda_color(target),edgecolor="black",
                linewidth=1.1,alpha=point_alpha,zorder=6
            )

    pareto_cross_lam=crosshair_lambda if crosshair_lambda is not None else highlight_lambda
    if pareto_cross_lam is not None:
        target=_lambda_float(pareto_cross_lam)
        hit=g[np.isclose(g["dr_lambda_float"],target,rtol=1e-12,atol=0)]

        if not hit.empty:
            r=hit.iloc[0]; mode=crosshair_modes[0]
            if mode in {"vertical","both"}: ax0.axvline(r[px],**crosshair_line_kwargs)
            if mode in {"horizontal","both"}: ax0.axhline(r[py],**crosshair_line_kwargs)

    ax0.set_xlabel(
        _arg(xlabels,0,f"{ms_name} ΔmAP vs {ref_col}"),
        fontsize=axis_label_fontsize,labelpad=8
    )
    ax0.set_ylabel(
        _arg(ylabels,0,f"{assay_name} ΔmAP vs {ref_col}"),
        fontsize=axis_label_fontsize,labelpad=8
    )
    ax0.set_title(_arg(titles,0,"Adaptation–retention"),fontsize=title_fontsize)

    if _arg(xlims,0) is not None: ax0.set_xlim(_arg(xlims,0))
    if _arg(ylims,0) is not None: ax0.set_ylim(_arg(ylims,0))

    _format_axis(
        ax0,tick_decimals,trim_tick_zeros,n_ticks,
        tick_fontsize,tick_length,tick_width,tick_pad,
        spine_width,grid,grid_alpha
    )

    # ==================================================================
    # PANELS 2/3 — categorical lambda trajectories
    # ==================================================================
    metrics=[metric1,metric2]

    if isinstance(metric_line_colors,str):
        metric_line_colors=(metric_line_colors,metric_line_colors)
    if isinstance(metric_markers,str):
        metric_markers=(metric_markers,metric_markers)
    if isinstance(metric_linestyles,str):
        metric_linestyles=(metric_linestyles,metric_linestyles)

    for j,(ax,metric) in enumerate(zip([ax1,ax2],metrics),start=1):
        if metric not in ld.columns:
            raise ValueError(f"Metric '{metric}' not found in lambda_df.")

        d=ld.sort_values(
            "lambda",
            ascending=categorical_lambda_order=="ascending"
        ).copy()

        xx=np.array([xpos[float(v)] for v in d["lambda"]],dtype=float)
        yy=d[metric].to_numpy(float)

        # Constant-color connecting line.
        ax.plot(
            xx,yy,
            color=metric_line_colors[j-1],
            linewidth=metric_linewidth,
            linestyle=metric_linestyles[j-1],
            alpha=metric_point_alpha,
            zorder=2
        )

        # Points.
        for x_i,(_,r) in zip(xx,d.iterrows()):
            c=_lambda_color(r["lambda"]) if metric_color_by_lambda else metric_line_colors[j-1]

            ax.scatter(
                x_i,r[metric],
                s=metric_marker_size**2,
                marker=metric_markers[j-1],
                color=c,edgecolor="black",linewidth=.6,
                alpha=metric_point_alpha,zorder=4
            )

        # Error bars.
        err=errors[j]
        err_col=f"{metric}_{err}" if err is not None else None

        if err_col is not None:
            if err_col not in d.columns:
                raise ValueError(f"Error column '{err_col}' not found in lambda_df.")

            for x_i,(_,r) in zip(xx,d.iterrows()):
                if not np.isfinite(r[err_col]): continue

                c=_lambda_color(r["lambda"]) if metric_color_by_lambda else metric_line_colors[j-1]

                ax.errorbar(
                    x_i,r[metric],yerr=r[err_col],
                    fmt="none",ecolor=c,
                    alpha=metric_error_alpha,
                    **metric_error_kwargs
                )

        # Highlight.
        if highlight_lambda is not None:
            target=_lambda_float(highlight_lambda)
            hit=d[np.isclose(d["lambda"].astype(float),target,rtol=1e-12,atol=0)]

            if not hit.empty:
                r=hit.iloc[0]
                hx=xpos[float(r["lambda"])]
                hy=float(r[metric])
                hc=_lambda_color(target) if metric_color_by_lambda else metric_line_colors[j-1]

                ax.scatter(
                    hx,hy,s=highlight_size,marker="*",
                    color=hc,edgecolor="black",linewidth=1.1,
                    alpha=metric_point_alpha,zorder=6
                )

        # Crosshair.
        cross_lam=crosshair_lambda if crosshair_lambda is not None else highlight_lambda
        if cross_lam is not None:
            target=_lambda_float(cross_lam)
            hit=d[np.isclose(d["lambda"].astype(float),target,rtol=1e-12,atol=0)]

            if not hit.empty:
                r=hit.iloc[0]
                hx=xpos[float(r["lambda"])]
                hy=float(r[metric])
                mode=crosshair_modes[j]

                if mode in {"vertical","both"}:
                    ax.axvline(hx,**crosshair_line_kwargs)
                if mode in {"horizontal","both"}:
                    ax.axhline(hy,**crosshair_line_kwargs)

        # Pure categorical lambda x-axis.
        ax.set_xticks(np.arange(len(lambda_vals)))
        ax.set_xticklabels(lambda_tick_labels,fontsize=tick_fontsize,rotation=0)

        if _arg(xlims,j) is not None:
            ax.set_xlim(_arg(xlims,j))
        else:
            ax.set_xlim(-.35,len(lambda_vals)-.65)

        ax.set_xlabel(
            _arg(xlabels,j,"DR λ"),
            fontsize=axis_label_fontsize,labelpad=8
        )
        ax.set_ylabel(
            _arg(ylabels,j,metric),
            fontsize=axis_label_fontsize,labelpad=8
        )
        ax.set_title(
            _arg(titles,j,metric),
            fontsize=title_fontsize
        )

        if _arg(ylims,j) is not None:
            ax.set_ylim(_arg(ylims,j))

        def _yfmt(v,_):
            if abs(v)<10**(-(tick_decimals+2)): v=0
            s=f"{v:.{tick_decimals}f}"
            return s.rstrip("0").rstrip(".") if trim_tick_zeros else s

        if n_ticks is not None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

        ax.yaxis.set_major_formatter(FuncFormatter(_yfmt))
        ax.tick_params(
            axis="both",labelsize=tick_fontsize,
            length=tick_length,width=tick_width,pad=tick_pad
        )

        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        if grid: ax.grid(True,alpha=grid_alpha)
        else: ax.grid(False)

    # ==================================================================
    # FINAL PANEL GEOMETRY
    # ==================================================================
    if square_panels:
        for ax in axes:
            ax.set_box_aspect(1)

    actual_right=right_margin if colorbar_position=="pareto" else min(right_margin,.91)
    fig.subplots_adjust(wspace=wspace,right=actual_right)

    # Force square-panel geometry to be finalized.
    fig.canvas.draw()

    # Visually compensate for colorbar occupying panel1 -> panel2 whitespace.
    if middle_panel_shift:
        p=ax1.get_position()
        shift=p.width*float(middle_panel_shift)

        ax1.set_position([
            p.x0+shift,
            p.y0,
            p.width,
            p.height,
        ])

    # Force updated positions to finalize before colorbar placement.
    fig.canvas.draw()

    # ==================================================================
    # COLORBAR — constructed from FINAL physical axis dimensions.
    # ==================================================================
    anchor_ax=ax0 if colorbar_position=="pareto" else ax2
    pos=anchor_ax.get_position()

    cb_width=pos.width*float(colorbar_size)
    cb_pad=pos.width*float(colorbar_pad)

    cax=fig.add_axes([
        pos.x1+cb_pad,
        pos.y0,
        cb_width,
        pos.height,
    ])

    cm=ListedColormap([colors[x] for x in color_order])
    norm=BoundaryNorm(np.arange(len(color_order)+1)-.5,len(color_order))
    sm=plt.cm.ScalarMappable(cmap=cm,norm=norm)
    sm.set_array([])

    cb=fig.colorbar(
        sm,cax=cax,
        ticks=np.arange(len(color_order)),
        spacing="uniform"
    )

    cb.set_ticks(
        np.arange(len(color_order)),
        labels=[_lambda_numeric_label(x) for x in color_order]
    )
    cb.set_label(colorbar_label,fontsize=colorbar_label_fontsize)
    cb.ax.tick_params(labelsize=colorbar_tick_fontsize)
    cb.outline.set_linewidth(colorbar_outline_width)

    # ------------------------------------------------------------------
    # Save/show.
    # ------------------------------------------------------------------
    if save_path:
        fig.savefig(f"{save_path}.pdf",format="pdf",bbox_inches="tight",dpi=dpi)
        fig.savefig(f"{save_path}.png",format="png",bbox_inches="tight",dpi=dpi)

    if show_plot: plt.show()

    return {
        "trajectory_df":traj,
        "lambda_df":ld,
        "fig":fig,
        "axes":axes,
        "colorbar":cb,
    }



import re
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

def compute_lambda_transition_table(
    dataset_dfs, score_cols_by_dataset, method_prefixes_by_dataset,
    method="pepcl_r_dr",
    transitions=((1e-6,1e-5),(1e-5,1e-4)),
    ms_key="ms", assay_key="assay",
    steps=None, learning_rate=None, focus_seeds=None,
    ci_method="bootstrap", ci=0.95, n_boot=10000, random_state=42,
    verbose=True,
):
    """
    Compute paired adjacent-lambda transition effects across datasets.

    Parameters
    ----------
    dataset_dfs : dict
        Example:
        {
            "YD":{"ms":holec_ms_df,"assay":holec_assay_df},
            "EpiScan":{"ms":epi_ms_df,"assay":epi_assay_df},
            ...
        }

    score_cols_by_dataset : dict[str, iterable]
        Dataset -> model score columns.

    method_prefixes_by_dataset : dict[str, dict]
        Dataset -> method_prefixes dictionary.

    method : str
        Method to analyze, default "pepcl_r_dr".

    transitions : iterable[(low_lambda, high_lambda)]
        Default:
            1e-6 -> 1e-5
            1e-5 -> 1e-4

    steps, learning_rate :
        None, scalar, or dict keyed by dataset.
        If multiple step/LR settings remain after parsing, function raises.

    focus_seeds :
        None, iterable, or dict keyed by dataset.
        If provided, exactly those matched replicate IDs are required.

    ci_method : {"bootstrap","t"}
        CI calculated on PAIRED replicate-level differences.

    Returns
    -------
    transition_df : pd.DataFrame
        One row per dataset, with mean/SD/SE/CI for every transition.

    replicate_df : pd.DataFrame
        One row per dataset x transition x replicate.
    """
    if ci_method not in {"bootstrap","t"}:
        raise ValueError("ci_method must be 'bootstrap' or 't'.")

    def _arg(x,ds,default=None):
        return x.get(ds,default) if isinstance(x,dict) else default if x is None else x

    def _lam(x):
        s=str(x)
        if s in {"0","0.0"}: return 0.0
        if "eN" in s:
            a,b=s.split("eN")
            return float(a)*10**(-int(b))
        return float(s)

    def _lam_string(x):
        v=float(x)
        if v==0: return "0"
        m,e=f"{v:.12e}".split("e")
        m=m.rstrip("0").rstrip("."); e=int(e)
        return f"{m}eN{abs(e)}" if e<0 else f"{m}e{e}"

    def _ci(x):
        x=np.asarray(x,float)
        x=x[np.isfinite(x)]
        n=len(x)
        if n<2: return np.nan,np.nan

        if ci_method=="t":
            se=x.std(ddof=1)/np.sqrt(n)
            crit=student_t.ppf((1+ci)/2,n-1)
            return x.mean()-crit*se,x.mean()+crit*se

        rng=np.random.default_rng(random_state)
        boot=x[rng.integers(0,n,size=(n_boot,n))].mean(axis=1)
        a=(1-ci)/2
        return tuple(np.quantile(boot,[a,1-a]))

    pattern=re.compile(
        r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lam>.+?)_seed_(?P<seed>\d+)$"
    )

    replicate_rows=[]
    summary_rows=[]

    for dataset,dfs in dataset_dfs.items():
        if ms_key not in dfs or assay_key not in dfs:
            raise ValueError(f"dataset_dfs['{dataset}'] must contain '{ms_key}' and '{assay_key}'.")
        if dataset not in score_cols_by_dataset:
            raise ValueError(f"Missing score_cols for dataset '{dataset}'.")
        if dataset not in method_prefixes_by_dataset:
            raise ValueError(f"Missing method_prefixes for dataset '{dataset}'.")

        ms_df=dfs[ms_key]
        assay_df=dfs[assay_key]
        prefixes=method_prefixes_by_dataset[dataset]

        if method not in prefixes:
            raise ValueError(f"Method '{method}' missing from method_prefixes for '{dataset}'.")

        prefix=prefixes[method]
        score_cols=list(dict.fromkeys(score_cols_by_dataset[dataset]))

        rows=[]
        for c in score_cols:
            if c not in ms_df.columns or c not in assay_df.columns:
                continue
            if not str(c).startswith(prefix):
                continue

            m=pattern.search(str(c))
            if m is None:
                continue

            rows.append({
                "score_col":c,
                "steps":int(m["steps"]),
                "learning_rate":m["lr"],
                "lambda":_lam(m["lam"]),
                "lambda_string":m["lam"],
                "seed":int(m["seed"]),
            })

        meta=pd.DataFrame(rows)
        if meta.empty:
            raise ValueError(f"{dataset}: no lambda columns parsed for method '{method}'.")

        ds_steps=_arg(steps,dataset)
        ds_lr=_arg(learning_rate,dataset)

        if ds_steps is not None:
            meta=meta[meta["steps"]==int(ds_steps)].copy()
        if ds_lr is not None:
            meta=meta[meta["learning_rate"].astype(str)==str(ds_lr)].copy()

        required_lambdas=sorted(set(float(x) for pair in transitions for x in pair))
        meta=meta[meta["lambda"].isin(required_lambdas)].copy()

        if meta.empty:
            raise ValueError(f"{dataset}: no requested transition lambdas remain.")

        hp=meta[["steps","learning_rate"]].drop_duplicates()
        if len(hp)!=1:
            raise ValueError(
                f"{dataset}: found {len(hp)} step/LR settings. "
                "Supply steps= and/or learning_rate= to select one."
            )

        # One model per lambda x replicate.
        dup=meta.duplicated(["lambda","seed"],keep=False)
        if dup.any():
            raise ValueError(
                f"{dataset}: duplicate lambda/seed cells:\n"
                f"{meta.loc[dup,['lambda','seed','score_col']]}"
            )

        ds_focus=_arg(focus_seeds,dataset)
        if ds_focus is not None:
            ds_focus=tuple(sorted(set(int(s) for s in ds_focus)))
            meta=meta[meta["seed"].isin(ds_focus)].copy()

        # Require the same replicate set across every required lambda.
        seed_sets={
            lam:tuple(sorted(meta.loc[meta["lambda"]==lam,"seed"].unique()))
            for lam in required_lambdas
        }

        if any(len(v)==0 for v in seed_sets.values()):
            missing=[lam for lam,v in seed_sets.items() if not v]
            raise ValueError(f"{dataset}: missing required lambdas {missing}.")

        if ds_focus is not None:
            bad={lam:s for lam,s in seed_sets.items() if s!=ds_focus}
            if bad:
                raise ValueError(
                    f"{dataset}: not all lambdas contain focus_seeds={ds_focus}. "
                    f"Observed: {bad}"
                )
            common_seeds=ds_focus
        else:
            unique_sets=set(seed_sets.values())
            if len(unique_sets)!=1:
                raise ValueError(
                    f"{dataset}: lambda values have inconsistent replicate sets: {seed_sets}. "
                    "Supply focus_seeds or filter to matched replicates."
                )
            common_seeds=next(iter(unique_sets))

        # Per-replicate macro metric. MS and assay are averaged over their
        # respective evaluable rows independently.
        perf_rows=[]
        for _,r in meta.iterrows():
            c=r["score_col"]

            ms_vals=ms_df[c].dropna()
            assay_vals=assay_df[c].dropna()

            if ms_vals.empty or assay_vals.empty:
                raise ValueError(f"{dataset}: empty metric values for '{c}'.")

            perf_rows.append({
                **r.to_dict(),
                "ms_performance":float(ms_vals.mean()),
                "assay_performance":float(assay_vals.mean()),
            })

        perf=pd.DataFrame(perf_rows)

        wide_ms=perf.pivot(index="seed",columns="lambda",values="ms_performance")
        wide_assay=perf.pivot(index="seed",columns="lambda",values="assay_performance")

        row={
            "dataset":dataset,
            "method":method,
            "steps":int(hp.iloc[0]["steps"]),
            "learning_rate":hp.iloc[0]["learning_rate"],
            "n_replicates":len(common_seeds),
            "replicate_ids":tuple(common_seeds),
        }

        for low,high in transitions:
            low=float(low); high=float(high)

            dms=wide_ms[high]-wide_ms[low]
            dassay=wide_assay[high]-wide_assay[low]

            label=f"{_lam_string(low)}_to_{_lam_string(high)}"

            for seed in common_seeds:
                replicate_rows.append({
                    "dataset":dataset,
                    "method":method,
                    "seed":seed,
                    "lambda_low":low,
                    "lambda_high":high,
                    "transition":label,
                    "ms_low":wide_ms.loc[seed,low],
                    "ms_high":wide_ms.loc[seed,high],
                    "ms_delta":dms.loc[seed],
                    "assay_low":wide_assay.loc[seed,low],
                    "assay_high":wide_assay.loc[seed,high],
                    "assay_delta":dassay.loc[seed],
                })

            for name,x in [("ms",dms),("assay",dassay)]:
                x=np.asarray(x,float)
                sd=x.std(ddof=1)
                se=sd/np.sqrt(len(x))
                lo,hi=_ci(x)

                row.update({
                    f"{name}_{label}_mean":x.mean(),
                    f"{name}_{label}_sd":sd,
                    f"{name}_{label}_se":se,
                    f"{name}_{label}_ci_low":lo,
                    f"{name}_{label}_ci_high":hi,
                })

        summary_rows.append(row)

        if verbose:
            print(f"\n{dataset}: n={len(common_seeds)} replicates {tuple(common_seeds)}")
            for low,high in transitions:
                label=f"{_lam_string(low)}_to_{_lam_string(high)}"
                print(
                    f"  {_lam_string(low)} -> {_lam_string(high)} | "
                    f"ΔMS={row[f'ms_{label}_mean']:+.5f} "
                    f"[{row[f'ms_{label}_ci_low']:+.5f}, {row[f'ms_{label}_ci_high']:+.5f}] | "
                    f"ΔAssay={row[f'assay_{label}_mean']:+.5f} "
                    f"[{row[f'assay_{label}_ci_low']:+.5f}, {row[f'assay_{label}_ci_high']:+.5f}]"
                )

    transition_df=pd.DataFrame(summary_rows)
    replicate_df=pd.DataFrame(replicate_rows)

    return transition_df,replicate_df

def plot_lambda_transition_table(
    transition_df,
    transitions=((1e-6,1e-5),(1e-5,1e-4)),
    dataset_order=None,
    dataset_labels=None,
    column_order=("ms1","assay1","ms2","assay2"),
    column_labels=None,
    col_widths=None,

    # Number formatting.
    decimals=3,
    show_plus=True,
    ci_brackets=(" [",", ","]"),
    lambda_header_format="math",

    # Figure / fonts.
    fig_size=None,
    dpi=300,
    title="Lambda transition analysis",
    title_fontsize=16,
    header_fontsize=11,
    body_fontsize=11,
    font_family="Nimbus Sans",

    # Table geometry.
    table_bbox=(0.0,0.02,1.0,0.92),
    row_height_scale=1.45,
    dataset_align="left",
    value_align="center",

    # Lines.
    top_linewidth=1.5,
    header_linewidth=0.8,
    bottom_linewidth=1.5,
    line_color="black",

    # Optional row styling.
    bold_datasets=None,
    bold_all_dataset_names=False,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot a publication-style lambda transition table.

    Expected input:
        transition_df returned by compute_lambda_transition_table().

    lambda_header_format:
        "math" (default) -> exponent notation in headers, e.g. 10^{-6}
        "scientific"     -> plain notation in headers, e.g. 1e-6, 1e-4
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    if transition_df.empty:
        raise ValueError("transition_df is empty.")

    if len(transitions)!=2:
        raise ValueError("This plotting helper currently expects exactly two transitions.")

    def _lam_string(x):
        v=float(x)
        if v==0: return "0"
        m,e=f"{v:.12e}".split("e")
        m=m.rstrip("0").rstrip("."); e=int(e)
        return f"{m}eN{abs(e)}" if e<0 else f"{m}e{e}"

    def _math_lam(x):
        v=float(x)
        if v==0: return "0"
        e=int(np.floor(np.log10(abs(v))))
        m=v/(10**e)
        if np.isclose(m,1):
            return rf"$10^{{{e}}}$"
        return rf"${m:g}\times10^{{{e}}}$"

    def _sci_lam(x):
        v=float(x)
        if v==0: return "0"
        e=int(np.floor(np.log10(abs(v))))
        m=v/(10**e)
        return f"1e{e}" if np.isclose(m,1) else f"{m:g}e{e}"

    def _lam_header(x):
        if lambda_header_format=="math":
            return _math_lam(x)
        if lambda_header_format=="scientific":
            return _sci_lam(x)
        raise ValueError("lambda_header_format must be 'math' or 'scientific'.")

    def _key(low,high):
        return f"{_lam_string(low)}_to_{_lam_string(high)}"

    def _fmt(mean,lo,hi):
        sign="+" if show_plus else ""
        fmt=f"{{:{sign}.{decimals}f}}"
        return (
            fmt.format(mean)
            +ci_brackets[0]
            +fmt.format(lo)
            +ci_brackets[1]
            +fmt.format(hi)
            +ci_brackets[2]
        )

    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,"Nimbus Sans","Nimbus Sans L","DejaVu Sans"
    ]

    d=transition_df.copy()

    if dataset_order is None:
        dataset_order=d["dataset"].tolist()
    else:
        missing=[x for x in dataset_order if x not in set(d["dataset"])]
        if missing:
            raise ValueError(f"dataset_order contains missing datasets: {missing}")
        d=d.set_index("dataset").loc[dataset_order].reset_index()

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)

    (l1,h1),(l2,h2)=transitions
    k1=_key(l1,h1); k2=_key(l2,h2)

    specs={
        "ms1":(
            f"ms_{k1}_mean",f"ms_{k1}_ci_low",f"ms_{k1}_ci_high",
            rf"$\Delta$MS AP, {_lam_header(l1)} $\rightarrow$ {_lam_header(h1)} [95% CI]"
        ),
        "assay1":(
            f"assay_{k1}_mean",f"assay_{k1}_ci_low",f"assay_{k1}_ci_high",
            rf"$\Delta$Assay, {_lam_header(l1)} $\rightarrow$ {_lam_header(h1)} [95% CI]"
        ),
        "ms2":(
            f"ms_{k2}_mean",f"ms_{k2}_ci_low",f"ms_{k2}_ci_high",
            rf"$\Delta$MS AP, {_lam_header(l2)} $\rightarrow$ {_lam_header(h2)} [95% CI]"
        ),
        "assay2":(
            f"assay_{k2}_mean",f"assay_{k2}_ci_low",f"assay_{k2}_ci_high",
            rf"$\Delta$Assay, {_lam_header(l2)} $\rightarrow$ {_lam_header(h2)} [95% CI]"
        ),
    }

    bad=[x for x in column_order if x not in specs]
    if bad:
        raise ValueError(f"Unknown column_order entries: {bad}")

    display=pd.DataFrame()
    display["Assay"]=[
        dataset_labels.get(x,x)
        for x in d["dataset"]
    ]

    final_labels=["Assay"]

    for name in column_order:
        mean_col,lo_col,hi_col,default_label=specs[name]

        missing=[c for c in [mean_col,lo_col,hi_col] if c not in d.columns]
        if missing:
            raise ValueError(f"transition_df missing columns: {missing}")

        label=column_labels.get(name,default_label) if column_labels else default_label
        final_labels.append(label)

        display[label]=[
            _fmt(m,lo,hi)
            for m,lo,hi in zip(d[mean_col],d[lo_col],d[hi_col])
        ]

    ncols=len(display.columns)

    if col_widths is None:
        # More width to effect/CI columns.
        col_widths=[0.13]+[(0.87/(ncols-1))]*(ncols-1)

    if len(col_widths)!=ncols:
        raise ValueError(f"col_widths must have length {ncols}, got {len(col_widths)}.")

    if fig_size is None:
        fig_size=(13,max(2.8,0.52*(len(display)+2)))

    fp_body=FontProperties(family=font_family,size=body_fontsize)
    fp_header=FontProperties(family=font_family,size=header_fontsize,weight="bold")
    fp_title=FontProperties(family=font_family,size=title_fontsize)

    fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)
    ax.axis("off")

    if title is not None:
        ax.text(
            0.0,1.02,title,
            transform=ax.transAxes,
            ha="left",va="bottom",
            fontproperties=fp_title
        )

    tbl=ax.table(
        cellText=display.values.tolist(),
        colLabels=final_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=table_bbox,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(body_fontsize)
    tbl.scale(1.0,row_height_scale)

    nrows=len(display)+1

    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)

    # Header.
    for c in range(ncols):
        tbl[(0,c)].get_text().set_fontproperties(fp_header)

    # Alignment.
    for r in range(nrows):
        tbl[(r,0)].get_text().set_ha(dataset_align)
        tbl[(r,0)].PAD=.03

        for c in range(1,ncols):
            tbl[(r,c)].get_text().set_ha(value_align)

    bold_datasets=set() if bold_datasets is None else set(bold_datasets)

    for i,ds in enumerate(d["dataset"],start=1):
        if bold_all_dataset_names or ds in bold_datasets:
            tbl[(i,0)].get_text().set_weight("bold")

    fig.canvas.draw()

    left=tbl[(0,0)].get_x()
    right=tbl[(0,ncols-1)].get_x()+tbl[(0,ncols-1)].get_width()
    header_top=tbl[(0,0)].get_y()+tbl[(0,0)].get_height()
    header_bottom=tbl[(0,0)].get_y()
    bottom=tbl[(nrows-1,0)].get_y()

    # Booktabs-style horizontal rules only.
    ax.hlines(
        header_top,left,right,
        transform=ax.transAxes,
        linewidth=top_linewidth,color=line_color,zorder=5
    )
    ax.hlines(
        header_bottom,left,right,
        transform=ax.transAxes,
        linewidth=header_linewidth,color=line_color,zorder=5
    )
    ax.hlines(
        bottom,left,right,
        transform=ax.transAxes,
        linewidth=bottom_linewidth,color=line_color,zorder=5
    )

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf",dpi=dpi,bbox_inches="tight")
        fig.savefig(f"{save_path}.png",dpi=dpi,bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig,ax,tbl,display
# import os
# import re
# import pickle
# import numpy as np
# import pandas as pd

# def summarize_pepcl_gradient_logs(
#     logs_dir,
#     dataset,
#     objective_family,
#     method_contains="pepcl_r_dr",
#     stage_width=0.10,
#     verbose=True,
# ):
#     """
#     Load PepCL gradient/loss diagnostic logs and construct:

#       1) long_df:
#          one row per dataset x lambda x seed x diagnostic optimization step

#       2) summary_df:
#          median / Q25 / Q75 within dataset x lambda x training stage

#     Training stages:
#       early  = first `stage_width` fraction of full training budget
#       middle = central `stage_width` fraction
#       late   = final `stage_width` fraction

#     Derived quantities:
#       DR loss share =
#           |weighted DR loss| /
#           (|weighted task loss| + |weighted replay loss| + |weighted DR loss|)

#       plasticity gradient =
#           g_task + g_replay

#       DR gradient ratio =
#           ||g_DR_weighted|| / ||g_task_weighted + g_replay_weighted||

#       gradient cosine =
#           dot(g_DR, g_task + g_replay) /
#           (||g_DR|| * ||g_task + g_replay||)

#     The combined plasticity-gradient norm is reconstructed from the logged
#     task/replay norms and their dot product; individual norms are NOT added.

#     For full PepCL, cosine is calculated only if BOTH DR-task and DR-replay
#     dot products are available.

#     At lambda=0:
#       weighted DR loss = 0
#       weighted DR gradient norm = 0
#       DR loss share = 0
#       DR gradient ratio = 0
#       raw DR quantities may remain NaN if DR was not evaluated
#       cosine is NaN because the DR gradient has zero norm
#     """
#     logs_dir=os.fspath(logs_dir)

#     if objective_family not in {"classification","continuous"}:
#         raise ValueError("objective_family must be 'classification' or 'continuous'.")
#     if not (0 < stage_width < 0.5):
#         raise ValueError("stage_width must be between 0 and 0.5.")

#     def _load_pickle(path):
#         with open(path,"rb") as f:
#             return pickle.load(f)

#     def _lambda_float(x):
#         if x is None: return np.nan
#         if isinstance(x,(int,float,np.integer,np.floating)): return float(x)
#         s=str(x)
#         if "eN" in s:
#             a,b=s.split("eN",1)
#             return float(a)*10**(-int(b))
#         return float(s)

#     def _safe_float(x):
#         try: return float(x)
#         except (TypeError,ValueError): return np.nan

#     def _first_notna(*xs):
#         for x in xs:
#             if x is None: continue
#             try:
#                 if not pd.isna(x): return x
#             except (TypeError,ValueError):
#                 return x
#         return np.nan

#     def _stage(step,budget):
#         frac=float(step)/float(budget)
#         early_hi=stage_width
#         middle_lo=0.5-stage_width/2
#         middle_hi=0.5+stage_width/2
#         late_lo=1.0-stage_width

#         if frac<=early_hi: return "early"
#         if middle_lo<=frac<=middle_hi: return "middle"
#         if frac>=late_lo: return "late"
#         return None

#     pattern=re.compile(
#         r"_steps_(?P<budget>\d+)"
#         r"_lr_(?P<lr>.+?)"
#         r"_dr_lambda_(?P<lambda>.+?)"
#         r"_seed_(?P<seed>\d+)"
#         r"(?:_step_(?P<checkpoint>\d+))?$"
#     )

#     files=sorted(
#         f for f in os.listdir(logs_dir)
#         if f.endswith(".pkl") and method_contains in f
#     )

#     if not files:
#         raise ValueError(f"No .pkl files containing '{method_contains}' found in:\n{logs_dir}")

#     if verbose:
#         print(f"Dataset: {dataset}")
#         print(f"Objective family: {objective_family}")
#         print(f"Logs directory: {logs_dir}")
#         print(f"Candidate log files: {len(files)}\n")

#     rows=[]
#     used_files=0

#     for filename in files:
#         stem=os.path.splitext(filename)[0]
#         m=pattern.search(stem)

#         if m is None:
#             if verbose: print(f"Skipping unparsed filename: {filename}")
#             continue

#         log=_load_pickle(os.path.join(logs_dir,filename))

#         if not isinstance(log,dict):
#             if verbose: print(f"Skipping non-dict log: {filename}")
#             continue

#         config=log.get("config",{})
#         grad=log.get("gradient_telemetry",None)

#         if grad is None:
#             if verbose: print(f"Skipping log with no gradient_telemetry: {filename}")
#             continue

#         if isinstance(grad,list):
#             grad=pd.DataFrame(grad)
#         elif isinstance(grad,pd.DataFrame):
#             grad=grad.copy()
#         else:
#             try: grad=pd.DataFrame(grad)
#             except Exception:
#                 if verbose: print(f"Could not convert gradient telemetry: {filename}")
#                 continue

#         if grad.empty:
#             if verbose: print(f"Skipping empty gradient telemetry: {filename}")
#             continue

#         budget=int(m["budget"])
#         checkpoint=int(m["checkpoint"]) if m["checkpoint"] is not None else budget
#         seed=int(m["seed"])
#         learning_rate=m["lr"]
#         lambda_value=_lambda_float(m["lambda"])
#         is_lambda_zero=(float(lambda_value)==0.0)

#         for _,r in grad.iterrows():
#             step=int(r["step"])

#             # ----------------------------------------------------------
#             # Raw scalar losses
#             # ----------------------------------------------------------
#             task_loss_raw=_safe_float(r.get("ft_loss_raw",np.nan))
#             replay_loss_raw=_safe_float(r.get("ms_classification_loss_raw",np.nan))
#             dr_loss_raw=_safe_float(r.get("drift_loss_raw",np.nan))

#             task_weight=_safe_float(
#                 _first_notna(r.get("ft_weight",None),config.get("ft_lambda",None),1.0)
#             )
#             replay_weight=_safe_float(
#                 _first_notna(r.get("replay_weight",None),config.get("ms_lambda",None),1.0)
#             )
#             dr_weight=_safe_float(
#                 _first_notna(r.get("dr_weight",None),config.get("dr_lambda",None),lambda_value)
#             )

#             task_loss_weighted=_safe_float(r.get("ft_loss_weighted",np.nan))
#             replay_loss_weighted=_safe_float(r.get("ms_classification_loss_weighted",np.nan))
#             dr_loss_weighted=_safe_float(r.get("drift_loss_weighted",np.nan))

#             if not np.isfinite(task_loss_weighted) and np.isfinite(task_loss_raw):
#                 task_loss_weighted=task_weight*task_loss_raw
#             if not np.isfinite(replay_loss_weighted) and np.isfinite(replay_loss_raw):
#                 replay_loss_weighted=replay_weight*replay_loss_raw
#             if not np.isfinite(dr_loss_weighted) and np.isfinite(dr_loss_raw):
#                 dr_loss_weighted=dr_weight*dr_loss_raw

#             # True lambda=0 means the weighted DR contribution is exactly zero,
#             # even if raw DR was not evaluated/logged.
#             if is_lambda_zero:
#                 dr_loss_weighted=0.0

#             # ----------------------------------------------------------
#             # Raw + weighted component gradient norms
#             # ----------------------------------------------------------
#             task_grad_raw=_safe_float(r.get("ft_grad_norm_raw",np.nan))
#             replay_grad_raw=_safe_float(r.get("replay_grad_norm_raw",np.nan))
#             dr_grad_raw=_safe_float(r.get("dr_grad_norm_raw",np.nan))

#             task_grad_weighted=_safe_float(r.get("ft_grad_norm_weighted",np.nan))
#             replay_grad_weighted=_safe_float(r.get("replay_grad_norm_weighted",np.nan))
#             dr_grad_weighted=_safe_float(r.get("dr_grad_norm_weighted",np.nan))

#             if is_lambda_zero:
#                 dr_grad_weighted=0.0

#             # ----------------------------------------------------------
#             # Pairwise weighted gradient dot products
#             # ----------------------------------------------------------
#             replay_task_dot=_safe_float(r.get("replay_ft_dot_weighted",np.nan))
#             dr_task_dot=_safe_float(r.get("dr_ft_dot_weighted",np.nan))
#             dr_replay_dot=_safe_float(r.get("replay_dr_dot_weighted",np.nan))

#             if is_lambda_zero:
#                 # Mathematically these weighted DR dot products are zero.
#                 dr_task_dot=0.0
#                 dr_replay_dot=0.0

#             # ----------------------------------------------------------
#             # Plasticity gradient norm:
#             #
#             # ||g_task + g_replay||^2
#             # = ||g_task||^2 + ||g_replay||^2
#             #   + 2 dot(g_task,g_replay)
#             # ----------------------------------------------------------
#             if np.isfinite(task_grad_weighted) and np.isfinite(replay_grad_weighted):
#                 if np.isfinite(replay_task_dot):
#                     plasticity_sq=(
#                         task_grad_weighted**2+
#                         replay_grad_weighted**2+
#                         2.0*replay_task_dot
#                     )
#                     plasticity_grad_norm=np.sqrt(max(plasticity_sq,0.0))
#                 else:
#                     plasticity_grad_norm=np.nan
#             elif np.isfinite(task_grad_weighted):
#                 plasticity_grad_norm=task_grad_weighted
#             elif np.isfinite(replay_grad_weighted):
#                 plasticity_grad_norm=replay_grad_weighted
#             else:
#                 plasticity_grad_norm=np.nan

#             # ----------------------------------------------------------
#             # DR scalar-loss share
#             # ----------------------------------------------------------
#             if is_lambda_zero:
#                 dr_loss_share=0.0
#             else:
#                 vals=[task_loss_weighted,replay_loss_weighted,dr_loss_weighted]
#                 loss_denom=sum(abs(x) for x in vals if np.isfinite(x))
#                 dr_loss_share=(
#                     abs(dr_loss_weighted)/loss_denom
#                     if loss_denom>0 and np.isfinite(dr_loss_weighted)
#                     else np.nan
#                 )

#             # ----------------------------------------------------------
#             # Weighted DR / plasticity gradient ratio
#             # ----------------------------------------------------------
#             if is_lambda_zero:
#                 gradient_ratio=0.0
#             else:
#                 gradient_ratio=(
#                     dr_grad_weighted/plasticity_grad_norm
#                     if (
#                         np.isfinite(dr_grad_weighted) and
#                         np.isfinite(plasticity_grad_norm) and
#                         plasticity_grad_norm>0
#                     )
#                     else np.nan
#                 )

#             # ----------------------------------------------------------
#             # cos(g_DR, g_task + g_replay)
#             #
#             # For full PepCL require BOTH dot products. Do not calculate
#             # from a partial plasticity-gradient decomposition.
#             # ----------------------------------------------------------
#             if (
#                 np.isfinite(dr_task_dot) and
#                 np.isfinite(dr_replay_dot) and
#                 np.isfinite(dr_grad_weighted) and
#                 np.isfinite(plasticity_grad_norm) and
#                 dr_grad_weighted>0 and
#                 plasticity_grad_norm>0
#             ):
#                 dr_plasticity_dot=dr_task_dot+dr_replay_dot
#                 gradient_cosine=float(np.clip(
#                     dr_plasticity_dot/(dr_grad_weighted*plasticity_grad_norm),
#                     -1.0,1.0
#                 ))
#             else:
#                 dr_plasticity_dot=np.nan
#                 gradient_cosine=np.nan

#             rows.append({
#                 "dataset":dataset,
#                 "objective_family":objective_family,
#                 "lambda":lambda_value,
#                 "lambda_string":m["lambda"],
#                 "learning_rate":learning_rate,
#                 "seed":seed,
#                 "step_budget":budget,
#                 "checkpoint":checkpoint,
#                 "optimization_step":step,
#                 "optimization_fraction":step/budget,
#                 "stage":_stage(step,budget),
#                 "log_file":filename,

#                 "task_coefficient":task_weight,
#                 "replay_coefficient":replay_weight,
#                 "dr_coefficient":dr_weight,

#                 "task_loss_raw":task_loss_raw,
#                 "replay_loss_raw":replay_loss_raw,
#                 "dr_loss_raw":dr_loss_raw,

#                 "task_loss_weighted":task_loss_weighted,
#                 "replay_loss_weighted":replay_loss_weighted,
#                 "dr_loss_weighted":dr_loss_weighted,

#                 "task_gradient_norm_raw":task_grad_raw,
#                 "replay_gradient_norm_raw":replay_grad_raw,
#                 "dr_gradient_norm_raw":dr_grad_raw,

#                 "task_gradient_norm_weighted":task_grad_weighted,
#                 "replay_gradient_norm_weighted":replay_grad_weighted,
#                 "weighted_dr_gradient_norm":dr_grad_weighted,

#                 "task_replay_gradient_dot":replay_task_dot,
#                 "dr_task_gradient_dot":dr_task_dot,
#                 "dr_replay_gradient_dot":dr_replay_dot,

#                 "plasticity_gradient_norm":plasticity_grad_norm,
#                 "dr_plasticity_gradient_dot":dr_plasticity_dot,

#                 "dr_loss_share":dr_loss_share,
#                 "gradient_ratio":gradient_ratio,
#                 "gradient_cosine":gradient_cosine,

#                 "total_gradient_norm_actual":_safe_float(
#                     r.get("total_grad_norm_actual",np.nan)
#                 ),
#             })

#         used_files+=1

#     long_df=pd.DataFrame(rows)

#     if long_df.empty:
#         raise ValueError(
#             "No usable gradient observations were extracted. "
#             "Check filename format and gradient_telemetry contents."
#         )

#     long_df=long_df.sort_values(
#         ["lambda","seed","step_budget","optimization_step"]
#     ).reset_index(drop=True)

#     # --------------------------------------------------------------
#     # Stage-wise descriptive summaries
#     # --------------------------------------------------------------
#     stage_df=long_df[long_df["stage"].notna()].copy()

#     value_cols=[
#         "task_loss_raw",
#         "replay_loss_raw",
#         "dr_loss_raw",
#         "task_loss_weighted",
#         "replay_loss_weighted",
#         "dr_loss_weighted",
#         "task_gradient_norm_weighted",
#         "replay_gradient_norm_weighted",
#         "weighted_dr_gradient_norm",
#         "plasticity_gradient_norm",
#         "dr_loss_share",
#         "gradient_ratio",
#         "gradient_cosine",
#     ]

#     def _q25(x): return x.quantile(0.25)
#     def _q75(x): return x.quantile(0.75)
#     _q25.__name__="q25"
#     _q75.__name__="q75"

#     summary_df=(
#         stage_df
#         .groupby(
#             ["dataset","objective_family","lambda","lambda_string","stage"],
#             sort=False,dropna=False
#         )[value_cols]
#         .agg(["median",_q25,_q75,"count"])
#         .reset_index()
#     )

#     summary_df.columns=[
#         "_".join(str(x) for x in col if str(x)!="")
#         if isinstance(col,tuple) else col
#         for col in summary_df.columns
#     ]

#     stage_order={"early":0,"middle":1,"late":2}
#     summary_df["_stage_order"]=summary_df["stage"].map(stage_order)
#     summary_df=(
#         summary_df
#         .sort_values(["lambda","_stage_order"])
#         .drop(columns="_stage_order")
#         .reset_index(drop=True)
#     )

#     if verbose:
#         print(f"Processed log files: {used_files}")
#         print(f"Long-form diagnostic observations: {len(long_df):,}")
#         print(f"Lambdas: {sorted(long_df['lambda'].dropna().unique())}")
#         print(f"Seeds: {sorted(long_df['seed'].unique())}")
#         print(f"Step budgets: {sorted(long_df['step_budget'].unique())}")

#         print("\nStage counts:")
#         print(long_df["stage"].value_counts(dropna=False).rename_axis("stage").to_string())

#         n_cos=long_df["gradient_cosine"].notna().sum()
#         print(f"\nValid gradient cosine observations: {n_cos:,}/{len(long_df):,}")

#     return long_df,summary_df
import os
import re
import pickle
import numpy as np
import pandas as pd

def summarize_pepcl_gradient_logs(
    logs_dir, dataset, objective_family, method_contains="pepcl_r_dr",
    stage_width=0.10, verbose=True,
):
    """
    Load PepCL gradient/loss diagnostic logs.

    Returns
    -------
    long_df:
        one row per dataset x lambda x seed x diagnostic step

    summary_df:
        median/Q25/Q75/count per dataset x lambda x stage

    Existing combined diagnostics:
      S_D,total = |lambda L_D| / (|L_A| + |L_R| + |lambda L_D|)
      R_D,total = ||g_D|| / ||g_A + g_R||
      C_D,total = cos(g_D, g_A + g_R)

    Added assay-specific diagnostics:
      S_D|A = |lambda L_D| / (|L_A| + |lambda L_D|)
      R_D|A = ||g_D|| / ||g_A||
      C_D,A = cos(g_D, g_A)
    """
    logs_dir=os.fspath(logs_dir)
    if objective_family not in {"classification","continuous"}:
        raise ValueError("objective_family must be 'classification' or 'continuous'.")
    if not (0<stage_width<0.5): raise ValueError("stage_width must be between 0 and 0.5.")

    def _load(path):
        with open(path,"rb") as f: return pickle.load(f)

    def _lambda_float(x):
        if x is None: return np.nan
        if isinstance(x,(int,float,np.integer,np.floating)): return float(x)
        s=str(x)
        if "eN" in s:
            a,b=s.split("eN",1); return float(a)*10**(-int(b))
        return float(s)

    def _float(x):
        try: return float(x)
        except (TypeError,ValueError): return np.nan

    def _first(*xs):
        for x in xs:
            if x is None: continue
            try:
                if not pd.isna(x): return x
            except (TypeError,ValueError): return x
        return np.nan

    def _stage(step,budget):
        f=float(step)/float(budget)
        if f<=stage_width: return "early"
        if 0.5-stage_width/2<=f<=0.5+stage_width/2: return "middle"
        if f>=1-stage_width: return "late"
        return None

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lambda>.+?)"
        r"_seed_(?P<seed>\d+)(?:_step_(?P<checkpoint>\d+))?$"
    )

    files=sorted(f for f in os.listdir(logs_dir) if f.endswith(".pkl") and method_contains in f)
    if not files: raise ValueError(f"No .pkl files containing '{method_contains}' found in:\n{logs_dir}")

    if verbose:
        print(f"Dataset: {dataset}")
        print(f"Objective family: {objective_family}")
        print(f"Logs directory: {logs_dir}")
        print(f"Candidate log files: {len(files)}\n")

    rows=[]; used_files=0

    for filename in files:
        m=pattern.search(os.path.splitext(filename)[0])
        if m is None:
            if verbose: print(f"Skipping unparsed filename: {filename}")
            continue

        log=_load(os.path.join(logs_dir,filename))
        if not isinstance(log,dict): continue

        config=log.get("config",{})
        grad=log.get("gradient_telemetry",None)
        if grad is None:
            if verbose: print(f"Skipping log with no gradient_telemetry: {filename}")
            continue

        grad=grad.copy() if isinstance(grad,pd.DataFrame) else pd.DataFrame(grad)
        if grad.empty: continue

        budget=int(m["budget"])
        checkpoint=int(m["checkpoint"]) if m["checkpoint"] is not None else budget
        seed=int(m["seed"])
        lambda_value=_lambda_float(m["lambda"])
        is_zero=float(lambda_value)==0.0

        for _,r in grad.iterrows():
            step=int(r["step"])

            # ------------------------------------------------------
            # Losses
            # ------------------------------------------------------
            task_loss_raw=_float(r.get("ft_loss_raw",np.nan))
            replay_loss_raw=_float(r.get("ms_classification_loss_raw",np.nan))
            dr_loss_raw=_float(r.get("drift_loss_raw",np.nan))

            task_weight=_float(_first(r.get("ft_weight",None),config.get("ft_lambda",None),1.0))
            replay_weight=_float(_first(r.get("replay_weight",None),config.get("ms_lambda",None),1.0))
            dr_weight=_float(_first(r.get("dr_weight",None),config.get("dr_lambda",None),lambda_value))

            task_loss_weighted=_float(r.get("ft_loss_weighted",np.nan))
            replay_loss_weighted=_float(r.get("ms_classification_loss_weighted",np.nan))
            dr_loss_weighted=_float(r.get("drift_loss_weighted",np.nan))

            if not np.isfinite(task_loss_weighted) and np.isfinite(task_loss_raw):
                task_loss_weighted=task_weight*task_loss_raw
            if not np.isfinite(replay_loss_weighted) and np.isfinite(replay_loss_raw):
                replay_loss_weighted=replay_weight*replay_loss_raw
            if not np.isfinite(dr_loss_weighted) and np.isfinite(dr_loss_raw):
                dr_loss_weighted=dr_weight*dr_loss_raw
            if is_zero: dr_loss_weighted=0.0

            # ------------------------------------------------------
            # Gradient norms
            # ------------------------------------------------------
            task_grad_raw=_float(r.get("ft_grad_norm_raw",np.nan))
            replay_grad_raw=_float(r.get("replay_grad_norm_raw",np.nan))
            dr_grad_raw=_float(r.get("dr_grad_norm_raw",np.nan))

            task_grad_weighted=_float(r.get("ft_grad_norm_weighted",np.nan))
            replay_grad_weighted=_float(r.get("replay_grad_norm_weighted",np.nan))
            dr_grad_weighted=_float(r.get("dr_grad_norm_weighted",np.nan))
            if is_zero: dr_grad_weighted=0.0

            # Logged weighted pairwise dots
            replay_task_dot=_float(r.get("replay_ft_dot_weighted",np.nan))
            dr_task_dot_weighted=_float(r.get("dr_ft_dot_weighted",np.nan))
            dr_replay_dot=_float(r.get("replay_dr_dot_weighted",np.nan))

            if is_zero:
                dr_task_dot_weighted=0.0
                dr_replay_dot=0.0

            # Recover dot(weighted DR, raw assay gradient).
            # In these experiments task_weight == 1, so this is identical
            # to dr_task_dot_weighted.
            dr_assay_dot=(
                dr_task_dot_weighted/task_weight
                if np.isfinite(dr_task_dot_weighted) and np.isfinite(task_weight) and task_weight!=0
                else np.nan
            )

            # ------------------------------------------------------
            # Existing combined non-DR gradient
            # ------------------------------------------------------
            if np.isfinite(task_grad_weighted) and np.isfinite(replay_grad_weighted):
                if np.isfinite(replay_task_dot):
                    sq=task_grad_weighted**2+replay_grad_weighted**2+2*replay_task_dot
                    non_dr_grad_norm=np.sqrt(max(sq,0.0))
                else: non_dr_grad_norm=np.nan
            elif np.isfinite(task_grad_weighted): non_dr_grad_norm=task_grad_weighted
            elif np.isfinite(replay_grad_weighted): non_dr_grad_norm=replay_grad_weighted
            else: non_dr_grad_norm=np.nan

            # ------------------------------------------------------
            # Existing total DR scalar share
            # ------------------------------------------------------
            if is_zero:
                dr_loss_share=0.0
            else:
                denom=sum(abs(x) for x in [task_loss_raw,replay_loss_raw,dr_loss_weighted] if np.isfinite(x))
                dr_loss_share=abs(dr_loss_weighted)/denom if denom>0 and np.isfinite(dr_loss_weighted) else np.nan

            # Existing DR / combined non-DR gradient ratio
            if is_zero:
                gradient_ratio=0.0
            else:
                gradient_ratio=(dr_grad_weighted/non_dr_grad_norm
                                if np.isfinite(dr_grad_weighted) and np.isfinite(non_dr_grad_norm) and non_dr_grad_norm>0
                                else np.nan)

            # Existing cosine: DR vs assay+replay
            if (
                np.isfinite(dr_task_dot_weighted) and np.isfinite(dr_replay_dot) and
                np.isfinite(dr_grad_weighted) and np.isfinite(non_dr_grad_norm) and
                dr_grad_weighted>0 and non_dr_grad_norm>0
            ):
                dr_non_dr_dot=dr_task_dot_weighted+dr_replay_dot
                gradient_cosine=float(np.clip(
                    dr_non_dr_dot/(dr_grad_weighted*non_dr_grad_norm),-1,1
                ))
            else:
                dr_non_dr_dot=np.nan
                gradient_cosine=np.nan

            # ======================================================
            # NEW: DR relative specifically to assay objective
            # ======================================================

            # S_D|A = |lambda L_D| / (|L_A| + |lambda L_D|)
            if is_zero:
                dr_loss_share_vs_assay=0.0
            else:
                denom=(
                    abs(task_loss_raw)+abs(dr_loss_weighted)
                    if np.isfinite(task_loss_raw) and np.isfinite(dr_loss_weighted)
                    else np.nan
                )
                dr_loss_share_vs_assay=(
                    abs(dr_loss_weighted)/denom
                    if np.isfinite(denom) and denom>0 else np.nan
                )

            # R_D|A = ||weighted DR gradient|| / ||raw assay gradient||
            if is_zero:
                dr_gradient_ratio_vs_assay=0.0
            else:
                dr_gradient_ratio_vs_assay=(
                    dr_grad_weighted/task_grad_raw
                    if np.isfinite(dr_grad_weighted) and np.isfinite(task_grad_raw) and task_grad_raw>0
                    else np.nan
                )

            # C_D,A = cosine(weighted DR gradient, raw assay gradient)
            if (
                np.isfinite(dr_assay_dot) and
                np.isfinite(dr_grad_weighted) and np.isfinite(task_grad_raw) and
                dr_grad_weighted>0 and task_grad_raw>0
            ):
                dr_assay_gradient_cosine=float(np.clip(
                    dr_assay_dot/(dr_grad_weighted*task_grad_raw),-1,1
                ))
            else:
                dr_assay_gradient_cosine=np.nan

            rows.append({
                "dataset":dataset,"objective_family":objective_family,
                "lambda":lambda_value,"lambda_string":m["lambda"],
                "learning_rate":m["lr"],"seed":seed,
                "step_budget":budget,"checkpoint":checkpoint,
                "optimization_step":step,"optimization_fraction":step/budget,
                "stage":_stage(step,budget),"log_file":filename,

                "task_coefficient":task_weight,"replay_coefficient":replay_weight,"dr_coefficient":dr_weight,

                "task_loss_raw":task_loss_raw,"replay_loss_raw":replay_loss_raw,"dr_loss_raw":dr_loss_raw,
                "task_loss_weighted":task_loss_weighted,"replay_loss_weighted":replay_loss_weighted,
                "dr_loss_weighted":dr_loss_weighted,

                "task_gradient_norm_raw":task_grad_raw,"replay_gradient_norm_raw":replay_grad_raw,
                "dr_gradient_norm_raw":dr_grad_raw,
                "task_gradient_norm_weighted":task_grad_weighted,
                "replay_gradient_norm_weighted":replay_grad_weighted,
                "weighted_dr_gradient_norm":dr_grad_weighted,

                "task_replay_gradient_dot":replay_task_dot,
                "dr_task_gradient_dot":dr_task_dot_weighted,
                "dr_replay_gradient_dot":dr_replay_dot,

                # Existing combined diagnostics
                "plasticity_gradient_norm":non_dr_grad_norm,
                "dr_plasticity_gradient_dot":dr_non_dr_dot,
                "dr_loss_share":dr_loss_share,
                "gradient_ratio":gradient_ratio,
                "gradient_cosine":gradient_cosine,

                # NEW assay-specific diagnostics
                "dr_assay_gradient_dot":dr_assay_dot,
                "dr_loss_share_vs_assay":dr_loss_share_vs_assay,
                "dr_gradient_ratio_vs_assay":dr_gradient_ratio_vs_assay,
                "dr_assay_gradient_cosine":dr_assay_gradient_cosine,

                "total_gradient_norm_actual":_float(r.get("total_grad_norm_actual",np.nan)),
            })

        used_files+=1

    long_df=pd.DataFrame(rows)
    if long_df.empty: raise ValueError("No usable gradient observations were extracted.")

    long_df=long_df.sort_values(["lambda","seed","step_budget","optimization_step"]).reset_index(drop=True)
    stage_df=long_df[long_df["stage"].notna()].copy()

    value_cols=[
        "task_loss_raw","replay_loss_raw","dr_loss_raw",
        "task_loss_weighted","replay_loss_weighted","dr_loss_weighted",
        "task_gradient_norm_raw","replay_gradient_norm_raw","dr_gradient_norm_raw",
        "task_gradient_norm_weighted","replay_gradient_norm_weighted","weighted_dr_gradient_norm",
        "plasticity_gradient_norm",
        "dr_loss_share","gradient_ratio","gradient_cosine",
        "dr_loss_share_vs_assay","dr_gradient_ratio_vs_assay","dr_assay_gradient_cosine",
    ]

    def _q25(x): return x.quantile(.25)
    def _q75(x): return x.quantile(.75)
    _q25.__name__="q25"; _q75.__name__="q75"

    summary_df=(stage_df
        .groupby(["dataset","objective_family","lambda","lambda_string","stage"],sort=False,dropna=False)[value_cols]
        .agg(["median",_q25,_q75,"count"]).reset_index())

    summary_df.columns=[
        "_".join(str(x) for x in c if str(x)!="") if isinstance(c,tuple) else c
        for c in summary_df.columns
    ]

    order={"early":0,"middle":1,"late":2}
    summary_df["_stage_order"]=summary_df["stage"].map(order)
    summary_df=(summary_df.sort_values(["lambda","_stage_order"])
                .drop(columns="_stage_order").reset_index(drop=True))

    if verbose:
        print(f"Processed log files: {used_files}")
        print(f"Long-form diagnostic observations: {len(long_df):,}")
        print(f"Lambdas: {sorted(long_df['lambda'].dropna().unique())}")
        print(f"Seeds: {sorted(long_df['seed'].unique())}")
        print(f"Step budgets: {sorted(long_df['step_budget'].unique())}")

    return long_df,summary_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepcl_gradient_diagnostics(
    long_df,
    dataset_col="dataset",
    lambda_col="lambda",
    loss_share_col="dr_loss_share",
    gradient_ratio_col="gradient_ratio",
    objective_col="objective_family",
    trajectory_stage="late",
    highlight_lambda=1e-5,
    figsize=(12,5),
    separate_objective_families=False,
    gradient_symlog=True,
    symlog_linthresh=0.1,
    show_reference_lines=True,
):
    """
    Plot PepCL drift-regularization diagnostics over the complete lambda sweep.

    Panels:
      1) DR scalar-loss share vs lambda
      2) Weighted DR / plasticity gradient-norm ratio vs lambda

    Within each assay x lambda:
      center = median over selected diagnostic observations
      ribbon = Q25-Q75 (IQR), NOT a confidence interval

    trajectory_stage:
      "early"  -> first 10% of training
      "middle" -> central 10%
      "late"   -> final 10% (recommended primary diagnostic)
      None     -> entire trajectory

    Lambda is treated as a categorical axis so lambda=0 can be displayed
    separately from the logarithmically spaced positive coefficients.

    Reference lines:
      DR loss share = 0.5
      DR gradient ratio = 1.0
    """
    required=[
        dataset_col,
        lambda_col,
        loss_share_col,
        gradient_ratio_col,
    ]
    missing=[c for c in required if c not in long_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if trajectory_stage not in {"early","middle","late",None}:
        raise ValueError("trajectory_stage must be 'early', 'middle', 'late', or None.")

    df=long_df.copy()

    # --------------------------------------------------------------
    # Select training stage BEFORE aggregation.
    # --------------------------------------------------------------
    if trajectory_stage is not None:
        if "stage" not in df.columns:
            raise ValueError(
                "long_df has no 'stage' column. Use the output from "
                "summarize_pepcl_gradient_logs()."
            )
        df=df[df["stage"]==trajectory_stage].copy()

        if df.empty:
            raise ValueError(f"No observations found for trajectory_stage='{trajectory_stage}'.")

    # --------------------------------------------------------------
    # True lambda=0 only. Do NOT use np.isclose here because its
    # default atol would also classify 1e-8 as zero.
    # --------------------------------------------------------------
    lambda_numeric=pd.to_numeric(df[lambda_col],errors="coerce")
    zero_mask=lambda_numeric.eq(0.0)

    df.loc[zero_mask,loss_share_col]=0.0
    df.loc[zero_mask,gradient_ratio_col]=0.0

    # --------------------------------------------------------------
    # Aggregate median + IQR.
    # --------------------------------------------------------------
    group_cols=[dataset_col,lambda_col]
    if objective_col in df.columns:
        group_cols.insert(1,objective_col)

    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)

    plot_df=(
        df.groupby(group_cols,dropna=False,sort=False)
        .agg(
            dr_loss_share_median=(loss_share_col,"median"),
            dr_loss_share_q25=(loss_share_col,q25),
            dr_loss_share_q75=(loss_share_col,q75),

            gradient_ratio_median=(gradient_ratio_col,"median"),
            gradient_ratio_q25=(gradient_ratio_col,q25),
            gradient_ratio_q75=(gradient_ratio_col,q75),

            n_observations=(lambda_col,"size"),
        )
        .reset_index()
    )

    # --------------------------------------------------------------
    # Categorical lambda axis.
    # --------------------------------------------------------------
    lambda_values=sorted(
        plot_df[lambda_col].dropna().astype(float).unique()
    )
    lambda_to_x={lam:i for i,lam in enumerate(lambda_values)}

    def lambda_label(lam):
        lam=float(lam)

        if lam==0.0:
            return "0"

        exp=int(np.floor(np.log10(abs(lam))))
        mant=lam/(10**exp)

        if np.isclose(mant,1.0,rtol=1e-12,atol=0.0):
            return rf"$10^{{{exp}}}$"

        return rf"${mant:g}\times10^{{{exp}}}$"

    x_ticks=[lambda_to_x[x] for x in lambda_values]
    x_labels=[lambda_label(x) for x in lambda_values]

    # --------------------------------------------------------------
    # Figure helper.
    # --------------------------------------------------------------
    def _make_figure(sub_df,title_suffix=None):
        fig,axes=plt.subplots(1,2,figsize=figsize)

        assays=list(sub_df[dataset_col].dropna().unique())

        for assay in assays:
            d=sub_df[sub_df[dataset_col]==assay].copy()
            d=d.sort_values(lambda_col)

            x=np.array(
                [lambda_to_x[float(v)] for v in d[lambda_col]],
                dtype=float
            )

            # ------------------------------------------------------
            # Panel 1: DR scalar-loss share
            # ------------------------------------------------------
            y=d["dr_loss_share_median"].to_numpy(dtype=float)
            lo=d["dr_loss_share_q25"].to_numpy(dtype=float)
            hi=d["dr_loss_share_q75"].to_numpy(dtype=float)

            axes[0].plot(x,y,marker="o",label=str(assay))
            axes[0].fill_between(x,lo,hi,alpha=0.2)

            # ------------------------------------------------------
            # Panel 2: weighted DR / plasticity gradient norm
            # ------------------------------------------------------
            y=d["gradient_ratio_median"].to_numpy(dtype=float)
            lo=d["gradient_ratio_q25"].to_numpy(dtype=float)
            hi=d["gradient_ratio_q75"].to_numpy(dtype=float)

            axes[1].plot(x,y,marker="o",label=str(assay))
            axes[1].fill_between(x,lo,hi,alpha=0.2)

        # ----------------------------------------------------------
        # Highlight lambda=1e-5.
        # ----------------------------------------------------------
        if float(highlight_lambda) in lambda_to_x:
            hx=lambda_to_x[float(highlight_lambda)]
            for ax in axes:
                ax.axvline(hx,linestyle="--",linewidth=1)

        # ----------------------------------------------------------
        # Mechanistic reference lines.
        # ----------------------------------------------------------
        if show_reference_lines:
            axes[0].axhline(0.5,linestyle=":",linewidth=1)
            axes[1].axhline(1.0,linestyle=":",linewidth=1)

        axes[0].set_title("DR scalar-loss share")
        axes[0].set_ylabel("Median DR loss share")

        axes[1].set_title("Weighted DR / plasticity gradient norm")
        axes[1].set_ylabel("Median gradient ratio")

        # Keeps the important low-ratio region visible while still
        # accommodating very large ratios at high lambda.
        if gradient_symlog:
            axes[1].set_yscale("symlog",linthresh=symlog_linthresh)

        for ax in axes:
            ax.set_xlabel(r"$\lambda$")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels,rotation=45,ha="right")
            ax.legend()

        if title_suffix is not None:
            fig.suptitle(title_suffix)

        fig.tight_layout()
        return fig,axes

    stage_text=(
        "Full trajectory"
        if trajectory_stage is None
        else f"{trajectory_stage.capitalize()} training"
    )

    # --------------------------------------------------------------
    # All assays on one figure.
    # --------------------------------------------------------------
    if not separate_objective_families:
        fig,axes=_make_figure(plot_df,title_suffix=stage_text)
        return fig,axes,plot_df

    # --------------------------------------------------------------
    # Optional separate figures by objective family.
    # --------------------------------------------------------------
    if objective_col not in plot_df.columns:
        raise ValueError(
            f"'{objective_col}' not found, so separate_objective_families=True "
            "cannot be used."
        )

    figures={}

    for family in plot_df[objective_col].dropna().unique():
        sub=plot_df[plot_df[objective_col]==family].copy()

        fig,axes=_make_figure(
            sub,
            title_suffix=f"{str(family).capitalize()} — {stage_text}",
        )

        figures[str(family)]=(fig,axes)

    return figures,plot_df

# # ======================================================================
# # Shared helpers
# # ======================================================================

# def _pepcl_lambda_mask(series, value):
#     """Numerically strict lambda comparison; does not confuse 1e-8 with zero."""
#     x=pd.to_numeric(series,errors="coerce").astype(float)
#     value=float(value)
#     if value==0.0:
#         return x.eq(0.0)
#     return np.isclose(x,value,rtol=1e-12,atol=0.0)


# def _pepcl_lambda_label(lam):
#     lam=float(lam)
#     if lam==0.0: return "0"
#     exp=int(np.floor(np.log10(abs(lam))))
#     mant=lam/(10**exp)
#     if np.isclose(mant,1.0,rtol=1e-12,atol=0.0):
#         return rf"$10^{{{exp}}}$"
#     return rf"${mant:g}\times10^{{{exp}}}$"


# def _pepcl_q25(x): return x.quantile(0.25)
# def _pepcl_q75(x): return x.quantile(0.75)


# def _pepcl_fmt_med_iqr(median,q25,q75,digits=3):
#     """Format as median [Q25, Q75]."""
#     vals=[median,q25,q75]
#     if not all(np.isfinite(v) for v in vals): return "NA"

#     max_abs=max(abs(v) for v in vals)
#     nonzero=[abs(v) for v in vals if v!=0]
#     min_abs=min(nonzero) if nonzero else 0

#     if (0<max_abs<10**(-digits)) or max_abs>=10**4 or (min_abs and max_abs/min_abs>=1e4):
#         return f"{median:.2e} [{q25:.2e}, {q75:.2e}]"

#     return f"{median:.{digits}f} [{q25:.{digits}f}, {q75:.{digits}f}]"


# def _pepcl_dataset_label(x,dataset_labels=None):
#     return dataset_labels.get(x,x) if dataset_labels else x


# # ======================================================================
# # MAIN FIGURE
# # Effective DR influence across lambda: 3 rows x 2 columns
# # ======================================================================

# def plot_pepcl_main_dr_influence(
#     long_df,
#     stages=("early","middle","late"),
#     dataset_col="dataset",
#     objective_col="objective_family",
#     lambda_col="lambda",
#     loss_share_col="dr_loss_share",
#     gradient_ratio_col="gradient_ratio",
#     highlight_lambda=1e-5,
#     gradient_symlog=True,
#     symlog_linthresh=0.1,
#     dataset_labels=None,
#     figsize=(12,12),
# ):
#     """
#     Main PepCL diagnostic figure.

#     Rows:
#       early, middle, late training

#     Columns:
#       left  = DR scalar-loss share S_D
#       right = relative DR gradient norm R_D

#     Each assay x lambda x stage:
#       line   = median over diagnostic observations
#       ribbon = Q25-Q75 IQR

#     References:
#       vertical: lambda = 1e-5
#       horizontal:
#         S_D = 0.5
#         R_D = 1

#     Returns
#     -------
#     fig, axes, plot_df
#     """
#     required=[
#         dataset_col,lambda_col,"stage",
#         loss_share_col,gradient_ratio_col
#     ]
#     missing=[c for c in required if c not in long_df.columns]
#     if missing: raise ValueError(f"Missing required columns: {missing}")

#     df=long_df.copy()
#     df=df[df["stage"].isin(stages)].copy()
#     if df.empty: raise ValueError("No observations remain for requested stages.")

#     # True lambda=0 only.
#     zero_mask=pd.to_numeric(df[lambda_col],errors="coerce").eq(0.0)
#     df.loc[zero_mask,loss_share_col]=0.0
#     df.loc[zero_mask,gradient_ratio_col]=0.0

#     group_cols=[dataset_col]
#     if objective_col in df.columns: group_cols.append(objective_col)
#     group_cols += [lambda_col,"stage"]

#     plot_df=(
#         df.groupby(group_cols,dropna=False,sort=False)
#         .agg(
#             dr_loss_share_median=(loss_share_col,"median"),
#             dr_loss_share_q25=(loss_share_col,_pepcl_q25),
#             dr_loss_share_q75=(loss_share_col,_pepcl_q75),
#             gradient_ratio_median=(gradient_ratio_col,"median"),
#             gradient_ratio_q25=(gradient_ratio_col,_pepcl_q25),
#             gradient_ratio_q75=(gradient_ratio_col,_pepcl_q75),
#             n_observations=(lambda_col,"size"),
#         )
#         .reset_index()
#     )

#     lambda_values=sorted(
#         pd.to_numeric(plot_df[lambda_col],errors="coerce").dropna().unique()
#     )
#     lambda_to_x={float(lam):i for i,lam in enumerate(lambda_values)}
#     xticks=list(range(len(lambda_values)))
#     xlabels=[_pepcl_lambda_label(x) for x in lambda_values]

#     fig,axes=plt.subplots(
#         len(stages),2,figsize=figsize,
#         squeeze=False,
#         sharex=True,
#     )

#     assays=list(plot_df[dataset_col].dropna().unique())

#     for row,stage in enumerate(stages):
#         stage_df=plot_df[plot_df["stage"]==stage]

#         for assay in assays:
#             d=stage_df[stage_df[dataset_col]==assay].sort_values(lambda_col)
#             if d.empty: continue

#             x=np.array([lambda_to_x[float(v)] for v in d[lambda_col]])

#             # S_D
#             y=d["dr_loss_share_median"].to_numpy(float)
#             lo=d["dr_loss_share_q25"].to_numpy(float)
#             hi=d["dr_loss_share_q75"].to_numpy(float)

#             line,=axes[row,0].plot(
#                 x,y,marker="o",
#                 label=_pepcl_dataset_label(assay,dataset_labels),
#             )
#             axes[row,0].fill_between(
#                 x,lo,hi,
#                 alpha=0.18,
#                 color=line.get_color(),
#             )

#             # R_D
#             y=d["gradient_ratio_median"].to_numpy(float)
#             lo=d["gradient_ratio_q25"].to_numpy(float)
#             hi=d["gradient_ratio_q75"].to_numpy(float)

#             line,=axes[row,1].plot(
#                 x,y,marker="o",
#                 label=_pepcl_dataset_label(assay,dataset_labels),
#             )
#             axes[row,1].fill_between(
#                 x,lo,hi,
#                 alpha=0.18,
#                 color=line.get_color(),
#             )

#         # Vertical lambda reference.
#         if float(highlight_lambda) in lambda_to_x:
#             hx=lambda_to_x[float(highlight_lambda)]
#             axes[row,0].axvline(hx,linestyle="--",linewidth=1)
#             axes[row,1].axvline(hx,linestyle="--",linewidth=1)

#         # Horizontal mechanistic references.
#         axes[row,0].axhline(0.5,linestyle=":",linewidth=1)
#         axes[row,1].axhline(1.0,linestyle=":",linewidth=1)

#         axes[row,0].set_ylabel(
#             f"{stage.capitalize()}\nMedian $S_D$"
#         )
#         axes[row,1].set_ylabel(
#             f"{stage.capitalize()}\nMedian $R_D$"
#         )

#         if gradient_symlog:
#             axes[row,1].set_yscale(
#                 "symlog",
#                 linthresh=symlog_linthresh,
#             )

#     axes[0,0].set_title("DR scalar-loss share, $S_D$")
#     axes[0,1].set_title("Relative DR gradient norm, $R_D$")

#     for ax in axes[-1,:]:
#         ax.set_xlabel(r"$\lambda$")
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xlabels,rotation=45,ha="right")

#     # One figure-level legend.
#     handles,labels=axes[0,0].get_legend_handles_labels()
#     if handles:
#         fig.legend(
#             handles,labels,
#             loc="upper center",
#             bbox_to_anchor=(0.5,0.995),
#             ncol=max(1,len(labels)),
#         )

#     fig.suptitle(
#         "Effective drift-regularization influence across training",
#         y=1.025,
#     )
#     fig.tight_layout()

#     return fig,axes,plot_df
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

def plot_pepcl_main_dr_influence(
    long_df, stages=("early","middle","late"),
    dataset_col="dataset", objective_col="objective_family", lambda_col="lambda",
    loss_share_col="dr_loss_share", gradient_ratio_col="gradient_ratio",
    dataset_labels=None, dataset_order=None,

    # labels / titles
    title="Effective drift-regularization influence across training", show_title=True,
    left_title="DR scalar-loss share, S_reg|N",
    right_title="Relative DR gradient norm, R_reg|N",
    left_ylabel="Median S_reg|N", right_ylabel="Median R_reg|N", x_label="λ",
    stage_labels=None,

    # figure geometry
    figsize=(12,12), left=0.09, right=0.98, bottom=0.09, top=0.88,
    wspace=0.22, hspace=0.08,

    # fonts / export
    font_family="sans-serif",
    sans_serif_fonts=("Nimbus Sans","Nimbus Sans L","Arial","Helvetica","DejaVu Sans"),
    title_fontsize=15, panel_title_fontsize=12, axis_label_fontsize=11,
    tick_fontsize=9, legend_fontsize=9, reference_text_fontsize=8,
    dpi=300, save_path=None, save_pdf=True, save_png=True, save_svg=False,
    transparent=False, bbox_inches="tight", pad_inches=None,

    # main curves
    line_colors=None, line_width=1.8, line_style="-",
    show_markers=True, marker="o", marker_size=5, marker_edge_width=0.7,
    ribbon_alpha=0.18, show_ribbon=True, curve_clip_on=False,

    # lambda reference
    highlight_lambda=1e-5,
    vertical_ref_color="tab:blue", vertical_ref_linewidth=1.0,
    vertical_ref_linestyle="--", vertical_ref_alpha=1.0,

    # horizontal references
    left_ref_y=0.5, right_ref_y=1.0,
    left_ref_color="tab:blue", right_ref_color="tab:blue",
    horizontal_ref_linewidth=1.0, horizontal_ref_linestyle=":",
    horizontal_ref_alpha=1.0,

    # optional text above horizontal references
    show_reference_text=True,
    left_ref_text="S_reg|N = 0.5", right_ref_text="R_reg|N = 1",
    reference_text_x=0.02, reference_text_pad=3,
    reference_text_color=None, reference_text_fontweight="normal",

    # x tick labels
    lambda_tick_format="scientific",   # "scientific" -> 1e-5 ; "math" -> 10^{-5}
    x_tick_rotation=0, x_tick_ha="center",

    # y scales / limits
    gradient_symlog=True, symlog_linthresh=0.1,
    left_ylim=None, right_ylim=None,
    left_lower_pad=0.015,
    right_lower_bound=0.0,
    right_upper_factor=1.25,

    # axes styling
    spine_width=0.8, tick_width=0.8, tick_length=3.5, tick_pad=3.0,

    # legend
    show_legend=True, legend_position="top", legend_frame=False,
    legend_ncol=None, legend_bbox=None, legend_linewidth=None,

    show_plot=True,
):
    plt.rcParams.update({
        "font.family":font_family,
        "font.sans-serif":list(sans_serif_fonts),
        "pdf.fonttype":42,
        "ps.fonttype":42,
        "svg.fonttype":"none",
        "figure.dpi":dpi,
        "savefig.dpi":dpi,
    })

    def _q25(x): return x.quantile(.25)
    def _q75(x): return x.quantile(.75)

    def _lambda_label(x):
        x=float(x)
        if x==0: return "0"
        exp=int(np.floor(np.log10(abs(x))))
        mant=x/(10**exp)

        if lambda_tick_format=="scientific":
            return f"1e{exp}" if np.isclose(mant,1,rtol=1e-12,atol=0) else f"{mant:g}e{exp}"

        if lambda_tick_format=="math":
            return rf"$10^{{{exp}}}$" if np.isclose(mant,1,rtol=1e-12,atol=0) else rf"${mant:g}\times10^{{{exp}}}$"

        raise ValueError("lambda_tick_format must be 'scientific' or 'math'.")

    def _plain_symlog_tick(v,_):
        """Plain-text symlog tick labels; no MathText/superscripts."""
        if not np.isfinite(v): return ""
        if np.isclose(v,0): return "0"

        sign="-" if v<0 else ""
        a=abs(float(v))
        exp=int(np.round(np.log10(a)))

        if np.isclose(a,10**exp,rtol=1e-8,atol=1e-12):
            if exp==0: return f"{sign}1"
            if exp>0:
                # 10, 100, 1000 remain ordinary text.
                return f"{sign}{10**exp:g}"
            # e.g. 10-1, 10-2, 10-3 -- ordinary text, not superscript.
            return f"{sign}10{exp}"

        return f"{v:g}"

    required=[dataset_col,lambda_col,"stage",loss_share_col,gradient_ratio_col]
    missing=[c for c in required if c not in long_df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

    df=long_df[long_df["stage"].isin(stages)].copy()
    if df.empty: raise ValueError("No observations remain for requested stages.")

    zero=pd.to_numeric(df[lambda_col],errors="coerce").eq(0.0)
    df.loc[zero,loss_share_col]=0.0
    df.loc[zero,gradient_ratio_col]=0.0

    group=[dataset_col]
    if objective_col in df.columns: group.append(objective_col)
    group += [lambda_col,"stage"]

    plot_df=(df.groupby(group,dropna=False,sort=False)
        .agg(
            dr_loss_share_median=(loss_share_col,"median"),
            dr_loss_share_q25=(loss_share_col,_q25),
            dr_loss_share_q75=(loss_share_col,_q75),
            gradient_ratio_median=(gradient_ratio_col,"median"),
            gradient_ratio_q25=(gradient_ratio_col,_q25),
            gradient_ratio_q75=(gradient_ratio_col,_q75),
            n_observations=(lambda_col,"size"),
        ).reset_index())

    lambdas=sorted(pd.to_numeric(plot_df[lambda_col],errors="coerce").dropna().unique())
    lambda_to_x={float(v):i for i,v in enumerate(lambdas)}
    xticks=np.arange(len(lambdas))
    xlabels=[_lambda_label(v) for v in lambdas]

    if dataset_order is None: dataset_order=list(plot_df[dataset_col].drop_duplicates())
    else: dataset_order=list(dataset_order)

    plot_df=plot_df[plot_df[dataset_col].isin(dataset_order)].copy()

    labels={d:(dataset_labels.get(d,d) if dataset_labels else d) for d in dataset_order}
    stage_labels=stage_labels or {s:s.capitalize() for s in stages}

    default_colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if line_colors is None:
        colors={d:default_colors[i%len(default_colors)] for i,d in enumerate(dataset_order)}
    elif isinstance(line_colors,dict):
        colors={d:line_colors.get(d,default_colors[i%len(default_colors)]) for i,d in enumerate(dataset_order)}
    else:
        colors={d:line_colors[i%len(line_colors)] for i,d in enumerate(dataset_order)}

    fig,axes=plt.subplots(len(stages),2,figsize=figsize,squeeze=False,sharex=True)
    fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)

    for row,stage in enumerate(stages):
        d_stage=plot_df[plot_df["stage"]==stage]

        for assay in dataset_order:
            d=d_stage[d_stage[dataset_col]==assay].sort_values(lambda_col)
            if d.empty: continue
            x=np.array([lambda_to_x[float(v)] for v in d[lambda_col]])

            # left: scalar-loss share
            y=d["dr_loss_share_median"].to_numpy(float)
            lo=d["dr_loss_share_q25"].to_numpy(float)
            hi=d["dr_loss_share_q75"].to_numpy(float)

            axes[row,0].plot(
                x,y,color=colors[assay],lw=line_width,ls=line_style,
                marker=marker if show_markers else None,ms=marker_size,
                markeredgewidth=marker_edge_width,label=labels[assay],
                clip_on=curve_clip_on,zorder=3,
            )

            if show_ribbon:
                axes[row,0].fill_between(
                    x,lo,hi,color=colors[assay],alpha=ribbon_alpha,lw=0,zorder=2
                )

            # right: gradient ratio
            y=d["gradient_ratio_median"].to_numpy(float)
            lo=d["gradient_ratio_q25"].to_numpy(float)
            hi=d["gradient_ratio_q75"].to_numpy(float)

            axes[row,1].plot(
                x,y,color=colors[assay],lw=line_width,ls=line_style,
                marker=marker if show_markers else None,ms=marker_size,
                markeredgewidth=marker_edge_width,label=labels[assay],
                clip_on=curve_clip_on,zorder=3,
            )

            if show_ribbon:
                axes[row,1].fill_between(
                    x,lo,hi,color=colors[assay],alpha=ribbon_alpha,lw=0,zorder=2
                )

        # vertical reference
        if float(highlight_lambda) in lambda_to_x:
            hx=lambda_to_x[float(highlight_lambda)]
            for ax in axes[row]:
                ax.axvline(
                    hx,color=vertical_ref_color,lw=vertical_ref_linewidth,
                    ls=vertical_ref_linestyle,alpha=vertical_ref_alpha,zorder=1
                )

        # horizontal references
        axes[row,0].axhline(
            left_ref_y,color=left_ref_color,lw=horizontal_ref_linewidth,
            ls=horizontal_ref_linestyle,alpha=horizontal_ref_alpha,zorder=1
        )
        axes[row,1].axhline(
            right_ref_y,color=right_ref_color,lw=horizontal_ref_linewidth,
            ls=horizontal_ref_linestyle,alpha=horizontal_ref_alpha,zorder=1
        )

        if show_reference_text:
            if left_ref_text:
                axes[row,0].annotate(
                    left_ref_text,xy=(reference_text_x,left_ref_y),
                    xycoords=("axes fraction","data"),
                    xytext=(0,reference_text_pad),textcoords="offset points",
                    ha="left",va="bottom",fontsize=reference_text_fontsize,
                    fontweight=reference_text_fontweight,
                    color=reference_text_color or left_ref_color
                )

            if right_ref_text:
                axes[row,1].annotate(
                    right_ref_text,xy=(reference_text_x,right_ref_y),
                    xycoords=("axes fraction","data"),
                    xytext=(0,reference_text_pad),textcoords="offset points",
                    ha="left",va="bottom",fontsize=reference_text_fontsize,
                    fontweight=reference_text_fontweight,
                    color=reference_text_color or right_ref_color
                )

        axes[row,0].set_ylabel(
            f"{stage_labels.get(stage,stage)}\n{left_ylabel}",
            fontsize=axis_label_fontsize
        )
        axes[row,1].set_ylabel(
            f"{stage_labels.get(stage,stage)}\n{right_ylabel}",
            fontsize=axis_label_fontsize
        )

        # Left-panel limits
        if left_ylim is not None:
            axes[row,0].set_ylim(*left_ylim)
        else:
            upper=max(
                1.04,
                np.nanmax(d_stage["dr_loss_share_q75"].to_numpy(float))*1.02
                if len(d_stage) else 1.04
            )
            axes[row,0].set_ylim(-abs(left_lower_pad),upper)

        # Right-panel scale / limits
        if gradient_symlog:
            axes[row,1].set_yscale("symlog",linthresh=symlog_linthresh)

            # IMPORTANT: replace Matplotlib's MathText log formatter
            # with ordinary editable text.
            axes[row,1].yaxis.set_major_formatter(
                FuncFormatter(_plain_symlog_tick)
            )

        if right_ylim is not None:
            axes[row,1].set_ylim(*right_ylim)
        else:
            vals=np.concatenate([
                d_stage["gradient_ratio_q75"].to_numpy(float),
                d_stage["gradient_ratio_median"].to_numpy(float)
            ])
            vals=vals[np.isfinite(vals)]
            ymax=float(vals.max()) if len(vals) else 1.0
            axes[row,1].set_ylim(
                bottom=right_lower_bound,
                top=max(ymax*right_upper_factor,right_ref_y*1.25)
            )

        for ax in axes[row]:
            ax.tick_params(
                axis="both",labelsize=tick_fontsize,
                width=tick_width,length=tick_length,pad=tick_pad
            )
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)

    axes[0,0].set_title(left_title,fontsize=panel_title_fontsize)
    axes[0,1].set_title(right_title,fontsize=panel_title_fontsize)

    for ax in axes[-1]:
        ax.set_xlabel(x_label,fontsize=axis_label_fontsize)
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xlabels,rotation=x_tick_rotation,
            ha=x_tick_ha,fontsize=tick_fontsize
        )

    if show_legend:
        lw=legend_linewidth if legend_linewidth is not None else line_width
        handles=[
            Line2D(
                [0],[0],color=colors[d],lw=lw,ls=line_style,
                marker=None,label=labels[d]
            )
            for d in dataset_order
        ]
        ncol=legend_ncol or len(handles)

        if legend_bbox is None:
            if legend_position=="top":
                loc,bbox="upper center",(0.5,top+0.055)
            elif legend_position=="bottom":
                loc,bbox="lower center",(0.5,0.015)
            else:
                raise ValueError("legend_position must be 'top' or 'bottom'.")
        else:
            loc="upper center" if legend_position=="top" else "lower center"
            bbox=legend_bbox

        fig.legend(
            handles=handles,loc=loc,bbox_to_anchor=bbox,
            ncol=ncol,frameon=legend_frame,fontsize=legend_fontsize
        )

    if show_title and title is not None:
        title_y=(top+0.095) if show_legend and legend_position=="top" else (top+0.045)
        fig.suptitle(title,y=title_y,fontsize=title_fontsize)

    if save_path is not None:
        root,ext=os.path.splitext(os.fspath(save_path))
        save_root=root if ext.lower() in {".pdf",".png",".svg"} else os.fspath(save_path)

        kw=dict(
            dpi=dpi,bbox_inches=bbox_inches,
            transparent=transparent,facecolor="white"
        )
        if pad_inches is not None:
            kw["pad_inches"]=pad_inches

        if save_pdf:
            fig.savefig(f"{save_root}.pdf",format="pdf",**kw)
        if save_png:
            fig.savefig(f"{save_root}.png",format="png",**kw)
        if save_svg:
            fig.savefig(f"{save_root}.svg",format="svg",**kw)

    if show_plot:
        plt.show()

    return fig,axes,plot_df
# def build_pepcl_supplementary_tables(long_df):
#     """
#     Construct numerical Supplementary Tables S1 and S2.

#     S1:
#       every dataset x lambda x stage
#       all raw and weighted scalar losses + S_D

#     S2:
#       every dataset x lambda x stage
#       component gradient norms + G_B + G_D^raw + G_D^w + R_D + C_D

#     No p-values or confidence intervals are calculated.

#     Returns
#     -------
#     s1_df, s2_df
#     """
#     required=[
#         "dataset","objective_family","lambda","lambda_string","stage",
#         "task_coefficient","replay_coefficient","dr_coefficient",

#         "task_loss_raw","replay_loss_raw","dr_loss_raw",
#         "task_loss_weighted","replay_loss_weighted","dr_loss_weighted",
#         "dr_loss_share",

#         "task_gradient_norm_raw","replay_gradient_norm_raw",
#         "dr_gradient_norm_raw",
#         "task_gradient_norm_weighted","replay_gradient_norm_weighted",
#         "weighted_dr_gradient_norm",
#         "plasticity_gradient_norm",
#         "gradient_ratio","gradient_cosine",

#         "optimization_step","seed",
#     ]

#     missing=[c for c in required if c not in long_df.columns]
#     if missing: raise ValueError(f"Missing required columns: {missing}")

#     df=long_df[long_df["stage"].notna()].copy()

#     keys=[
#         "dataset",
#         "objective_family",
#         "lambda",
#         "lambda_string",
#         "stage",
#     ]

#     def summarize(metrics):
#         rows=[]

#         for key_vals,d in df.groupby(keys,sort=False,dropna=False):
#             row=dict(zip(keys,key_vals))

#             row["n_observations"]=len(d)
#             row["n_unique_steps"]=d["optimization_step"].nunique()
#             row["n_seeds"]=d["seed"].nunique()

#             # Coefficients are expected to be fixed within a run/group.
#             row["task_coefficient"]=d["task_coefficient"].dropna().median()
#             row["replay_coefficient"]=d["replay_coefficient"].dropna().median()
#             row["dr_coefficient"]=d["dr_coefficient"].dropna().median()

#             for metric in metrics:
#                 x=d[metric].dropna()

#                 row[f"{metric}_median"]=x.median() if len(x) else np.nan
#                 row[f"{metric}_q25"]=x.quantile(.25) if len(x) else np.nan
#                 row[f"{metric}_q75"]=x.quantile(.75) if len(x) else np.nan

#             rows.append(row)

#         out=pd.DataFrame(rows)

#         stage_order={"early":0,"middle":1,"late":2}
#         out["_stage_order"]=out["stage"].map(stage_order)

#         return (
#             out.sort_values(
#                 ["dataset","lambda","_stage_order"]
#             )
#             .drop(columns="_stage_order")
#             .reset_index(drop=True)
#         )

#     # --------------------------------------------------------------
#     # Supplementary Table S1: scalar losses
#     # --------------------------------------------------------------
#     s1_metrics=[
#         "task_loss_raw",
#         "replay_loss_raw",
#         "dr_loss_raw",
#         "task_loss_weighted",
#         "replay_loss_weighted",
#         "dr_loss_weighted",
#         "dr_loss_share",
#     ]

#     s1_df=summarize(s1_metrics)

#     # --------------------------------------------------------------
#     # Supplementary Table S2: gradient diagnostics
#     #
#     # Dot products intentionally omitted. Those stay in long_df / QC.
#     # --------------------------------------------------------------
#     s2_metrics=[
#         "task_gradient_norm_raw",
#         "replay_gradient_norm_raw",
#         "dr_gradient_norm_raw",

#         "task_gradient_norm_weighted",
#         "replay_gradient_norm_weighted",
#         "weighted_dr_gradient_norm",

#         "plasticity_gradient_norm",
#         "gradient_ratio",
#         "gradient_cosine",
#     ]

#     s2_df=summarize(s2_metrics)

#     # Rename manuscript-level derived quantities while retaining explicit
#     # component names.
#     s2_df=s2_df.rename(columns={
#         "plasticity_gradient_norm_median":"non_dr_gradient_norm_median",
#         "plasticity_gradient_norm_q25":"non_dr_gradient_norm_q25",
#         "plasticity_gradient_norm_q75":"non_dr_gradient_norm_q75",

#         "gradient_ratio_median":"relative_dr_gradient_norm_median",
#         "gradient_ratio_q25":"relative_dr_gradient_norm_q25",
#         "gradient_ratio_q75":"relative_dr_gradient_norm_q75",

#         "gradient_cosine_median":"dr_non_dr_gradient_cosine_median",
#         "gradient_cosine_q25":"dr_non_dr_gradient_cosine_q25",
#         "gradient_cosine_q75":"dr_non_dr_gradient_cosine_q75",
#     })

#     return s1_df,s2_df
def build_pepcl_supplementary_tables(long_df):
    """
    S1: dataset x lambda x stage scalar diagnostics.
    S2: dataset x lambda x stage gradient diagnostics.

    Includes both:
      - combined assay+replay DR diagnostics
      - assay-specific DR diagnostics
    """
    required=[
        "dataset","objective_family","lambda","lambda_string","stage",
        "task_coefficient","replay_coefficient","dr_coefficient",
        "task_loss_raw","replay_loss_raw","dr_loss_raw",
        "task_loss_weighted","replay_loss_weighted","dr_loss_weighted",
        "dr_loss_share","dr_loss_share_vs_assay",
        "task_gradient_norm_raw","replay_gradient_norm_raw","dr_gradient_norm_raw",
        "task_gradient_norm_weighted","replay_gradient_norm_weighted","weighted_dr_gradient_norm",
        "plasticity_gradient_norm","gradient_ratio","gradient_cosine",
        "dr_gradient_ratio_vs_assay","dr_assay_gradient_cosine",
        "optimization_step","seed",
    ]
    missing=[c for c in required if c not in long_df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

    df=long_df[long_df["stage"].notna()].copy()
    keys=["dataset","objective_family","lambda","lambda_string","stage"]

    def summarize(metrics):
        rows=[]
        for vals,d in df.groupby(keys,sort=False,dropna=False):
            row=dict(zip(keys,vals))
            row["n_observations"]=len(d)
            row["n_unique_steps"]=d["optimization_step"].nunique()
            row["n_seeds"]=d["seed"].nunique()
            row["task_coefficient"]=d["task_coefficient"].dropna().median()
            row["replay_coefficient"]=d["replay_coefficient"].dropna().median()
            row["dr_coefficient"]=d["dr_coefficient"].dropna().median()

            for metric in metrics:
                x=d[metric].dropna()
                row[f"{metric}_median"]=x.median() if len(x) else np.nan
                row[f"{metric}_q25"]=x.quantile(.25) if len(x) else np.nan
                row[f"{metric}_q75"]=x.quantile(.75) if len(x) else np.nan
            rows.append(row)

        out=pd.DataFrame(rows)
        order={"early":0,"middle":1,"late":2}
        out["_stage_order"]=out["stage"].map(order)
        return out.sort_values(["dataset","lambda","_stage_order"]).drop(columns="_stage_order").reset_index(drop=True)

    # S1: scalar-loss diagnostics
    s1_df=summarize([
        "task_loss_raw","replay_loss_raw","dr_loss_raw",
        "task_loss_weighted","replay_loss_weighted","dr_loss_weighted",

        # Existing combined share
        "dr_loss_share",

        # NEW: DR relative specifically to assay
        "dr_loss_share_vs_assay",
    ])

    # S2: gradient diagnostics
    s2_df=summarize([
        "task_gradient_norm_raw","replay_gradient_norm_raw","dr_gradient_norm_raw",
        "task_gradient_norm_weighted","replay_gradient_norm_weighted","weighted_dr_gradient_norm",

        # Existing combined non-DR diagnostics
        "plasticity_gradient_norm","gradient_ratio","gradient_cosine",

        # NEW assay-specific diagnostics
        "dr_gradient_ratio_vs_assay","dr_assay_gradient_cosine",
    ])

    s2_df=s2_df.rename(columns={
        # Combined assay+replay
        "plasticity_gradient_norm_median":"non_dr_gradient_norm_median",
        "plasticity_gradient_norm_q25":"non_dr_gradient_norm_q25",
        "plasticity_gradient_norm_q75":"non_dr_gradient_norm_q75",

        "gradient_ratio_median":"relative_dr_vs_non_dr_gradient_norm_median",
        "gradient_ratio_q25":"relative_dr_vs_non_dr_gradient_norm_q25",
        "gradient_ratio_q75":"relative_dr_vs_non_dr_gradient_norm_q75",

        "gradient_cosine_median":"dr_non_dr_gradient_cosine_median",
        "gradient_cosine_q25":"dr_non_dr_gradient_cosine_q25",
        "gradient_cosine_q75":"dr_non_dr_gradient_cosine_q75",

        # Assay-specific
        "dr_gradient_ratio_vs_assay_median":"relative_dr_vs_assay_gradient_norm_median",
        "dr_gradient_ratio_vs_assay_q25":"relative_dr_vs_assay_gradient_norm_q25",
        "dr_gradient_ratio_vs_assay_q75":"relative_dr_vs_assay_gradient_norm_q75",

        "dr_assay_gradient_cosine_median":"dr_assay_gradient_cosine_median",
        "dr_assay_gradient_cosine_q25":"dr_assay_gradient_cosine_q25",
        "dr_assay_gradient_cosine_q75":"dr_assay_gradient_cosine_q75",
    })

    return s1_df,s2_df
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter, FuncFormatter, MaxNLocator

def plot_pepcl_loss_heatmaps(
    all_grad_long,
    dataset_col="dataset", lambda_col="lambda", step_col="optimization_step",
    task_col="task_loss_raw", replay_col="replay_loss_raw", dr_weighted_col="dr_loss_weighted",
    step_budget_col="step_budget", highlight_lambda=1e-5,
    dataset_order=None, dataset_labels=None,

    # metric / colorbar labels
    metric_titles=None, colorbar_labels=None,

    # figure geometry
    figsize=(15,14), left=0.08, right=0.98, bottom=0.07, top=0.83,
    wspace=0.04, hspace=0.10,

    # fonts
    font_family="sans-serif",
    sans_serif_fonts=("Nimbus Sans","Nimbus Sans L","Arial","Helvetica","DejaVu Sans"),
    title_fontsize=15, column_title_fontsize=12, axis_label_fontsize=11,
    x_tick_fontsize=9, y_tick_fontsize=9, colorbar_ticksize=9, colorbar_labelsize=10,

    # main title
    show_suptitle=True,
    suptitle="PepCL loss trajectories across drift-regularization strengths",
    suptitle_y=None, title_pad=0.025,

    # axes / spines / ticks
    spine_width=0.8, tick_width=0.8, tick_length=3.5, tick_pad=3.0,
    max_x_ticks=8,

    # lambda labels
    lambda_tick_format="scientific",  # "scientific" -> 1e-5, "math" -> 10^-5

    # colormap
    cmap="viridis",                   # str or matplotlib cmap
    gradient_colors=None,             # e.g. ["#081d58","#225ea8","#41b6c4","#ffffcc"]
    gradient_name="pepcl_custom",

    # color normalization
    color_scale="log",                # "log", "linear", or dict by metric
    colorbar_number_format="auto",    # "auto", "decimal", "scientific", "log"
    colorbar_tick_mode="auto",        # "auto" or "decades" for log norm
    colorbar_max_ticks=5,
    decimal_digits=4, scientific_digits=1,

    # colorbar placement
    colorbar_position="top",          # "top", "bottom", None
    colorbar_height=0.016,            # thickness
    colorbar_pad=0.030,               # separation from heatmaps
    colorbar_width_frac=0.98,

    # highlighted lambda row
    highlight_color="black", highlight_linewidth=1.8, highlight_linestyle="-",

    # stage boundaries
    mark_stage_boundaries=True,
    stage_color="tab:blue", stage_linewidth=0.6, stage_linestyle="--", stage_alpha=0.7,

    # lambda=0 DR row
    zero_dr_color="white", zero_dr_text="0", zero_dr_text_color="black",
    zero_dr_text_fontsize=9, annotate_zero_dr=True,

    # validation
    require_identical_steps=True,

    # output
    save_path=None, save_pdf=True, save_png=True, dpi=300,
    transparent=False, bbox_inches="tight", pad_inches=None,
    show_plot=True,
):
    """
    Publication-ready PepCL heatmap grid.

    Rows
    ----
    datasets

    Columns
    -------
    raw assay loss L_A
    raw replay loss L_R
    weighted DR loss lambda L_D

    Notes
    -----
    - One shared color scale per metric column.
    - lambda is categorical on y-axis.
    - lambda=0 weighted DR row is rendered using `zero_dr_color`.
    - PDF uses Type-42 fonts for editable text.
    """

    # ------------------------------------------------------------------
    # Publication rcParams
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family":font_family,
        "font.sans-serif":list(sans_serif_fonts),
        "pdf.fonttype":42,
        "ps.fonttype":42,
        "svg.fonttype":"none",
        "figure.dpi":dpi,
        "savefig.dpi":dpi,
    })

    metrics=[task_col,replay_col,dr_weighted_col]
    if metric_titles is None:
        metric_titles={task_col:r"$L_A$",replay_col:r"$L_R$",dr_weighted_col:r"$\lambda L_D$"}
    elif not isinstance(metric_titles,dict):
        metric_titles=dict(zip(metrics,metric_titles))

    if colorbar_labels is None:
        colorbar_labels={m:"Loss magnitude" for m in metrics}
    elif not isinstance(colorbar_labels,dict):
        colorbar_labels=dict(zip(metrics,colorbar_labels))

    def _lambda_label(x):
        x=float(x)
        if x==0.0: return "0"
        if lambda_tick_format=="scientific":
            mant,exp=f"{x:.12e}".split("e"); mant=mant.rstrip("0").rstrip("."); exp=int(exp)
            return f"{mant}e{exp}"
        if lambda_tick_format=="math":
            exp=int(np.floor(np.log10(abs(x)))); mant=x/(10**exp)
            return rf"$10^{{{exp}}}$" if np.isclose(mant,1.0,rtol=1e-12,atol=0) else rf"${mant:g}\times10^{{{exp}}}$"
        raise ValueError("lambda_tick_format must be 'scientific' or 'math'.")

    def _scale(metric):
        s=color_scale.get(metric,"log") if isinstance(color_scale,dict) else color_scale
        if s not in {"log","linear"}: raise ValueError("color_scale must be 'log', 'linear', or dict.")
        return s

    def _dec_fmt(x,pos=None): return f"{x:.{decimal_digits}f}"
    def _sci_fmt(x,pos=None):
        if x==0: return "0"
        mant,exp=f"{x:.{scientific_digits}e}".split("e"); mant=mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp)}"

    def _format_cbar(cb,scale_mode):
        fmt=colorbar_number_format
        if fmt=="auto":
            if scale_mode=="linear":
                sf=ScalarFormatter(useMathText=False); sf.set_scientific(False); sf.set_useOffset(False)
                cb.formatter=sf; cb.update_ticks()
        elif fmt=="decimal":
            cb.formatter=FuncFormatter(_dec_fmt); cb.update_ticks()
        elif fmt=="scientific":
            cb.formatter=FuncFormatter(_sci_fmt); cb.update_ticks()
        elif fmt=="log":
            cb.formatter=LogFormatterMathtext(); cb.update_ticks()
        else:
            raise ValueError("colorbar_number_format must be 'auto', 'decimal', 'scientific', or 'log'.")

    def _log_ticks(vmin,vmax,n):
        lo=int(np.ceil(np.log10(vmin))); hi=int(np.floor(np.log10(vmax)))
        if hi<lo: return []
        ticks=np.array([10.0**k for k in range(lo,hi+1)])
        if len(ticks)>n:
            idx=np.unique(np.linspace(0,len(ticks)-1,n,dtype=int)); ticks=ticks[idx]
        return ticks.tolist()

    # ------------------------------------------------------------------
    # Validate / prepare data
    # ------------------------------------------------------------------
    required=[dataset_col,lambda_col,step_col,task_col,replay_col,dr_weighted_col,step_budget_col]
    missing=[c for c in required if c not in all_grad_long.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

    df=all_grad_long.copy()
    df[lambda_col]=pd.to_numeric(df[lambda_col],errors="coerce").astype(float)
    df[step_col]=pd.to_numeric(df[step_col],errors="coerce")
    df=df.dropna(subset=[dataset_col,lambda_col,step_col]).copy()

    dataset_order=list(df[dataset_col].drop_duplicates()) if dataset_order is None else list(dataset_order)
    df=df[df[dataset_col].isin(dataset_order)].copy()
    if df.empty: raise ValueError("No rows remain after dataset filtering.")

    lambda_values=sorted(df[lambda_col].dropna().unique())
    if len(lambda_values)==0: raise ValueError("No valid lambda values found.")

    if require_identical_steps:
        step_sets=df.groupby([dataset_col,lambda_col])[step_col].apply(lambda x:tuple(sorted(pd.unique(x.astype(int)))))
        ref=step_sets.iloc[0]; bad=step_sets[step_sets.map(lambda x:x!=ref)]
        if len(bad):
            examples=[{"dataset":k[0],"lambda":k[1],"steps":v} for k,v in bad.items()]
            raise ValueError(f"Diagnostic steps are not identical.\nReference: {ref}\nExamples: {examples[:10]}")
        diagnostic_steps=np.array(ref,dtype=int)
    else:
        diagnostic_steps=np.array(sorted(pd.unique(df[step_col].astype(int))),dtype=int)

    budgets=sorted(pd.unique(df[step_budget_col].dropna().astype(int)))
    if len(budgets)!=1: raise ValueError(f"Expected one unique step budget, found: {budgets}")
    T=int(budgets[0])

    # ------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------
    if gradient_colors is not None:
        plot_cmap=LinearSegmentedColormap.from_list(gradient_name,list(gradient_colors))
    else:
        plot_cmap=plt.get_cmap(cmap) if isinstance(cmap,str) else cmap

    # Custom masked color controls lambda=0 DR region.
    try: plot_cmap=plot_cmap.copy()
    except AttributeError: pass
    plot_cmap.set_bad(zero_dr_color)

    # ------------------------------------------------------------------
    # Shared normalization per metric column
    # ------------------------------------------------------------------
    norms={}; scale_modes={}
    for metric in metrics:
        vals=pd.to_numeric(df[metric],errors="coerce").to_numpy(float)
        vals=vals[np.isfinite(vals)]
        if len(vals)==0: raise ValueError(f"Metric '{metric}' has no finite values.")
        mode=_scale(metric); scale_modes[metric]=mode
        if mode=="log":
            vals=vals[vals>0]
            if len(vals)==0: raise ValueError(f"Metric '{metric}' has no positive values for LogNorm.")
            vmin,vmax=float(vals.min()),float(vals.max())
            if np.isclose(vmin,vmax): vmax=vmin*1.01
            norms[metric]=LogNorm(vmin=vmin,vmax=vmax)
        else:
            vmin,vmax=float(vals.min()),float(vals.max())
            if np.isclose(vmin,vmax): vmax=vmin+1e-12
            norms[metric]=Normalize(vmin=vmin,vmax=vmax)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    nrows=len(dataset_order)
    fig,axes=plt.subplots(nrows,3,figsize=figsize,squeeze=False,sharex="col",sharey="row")
    fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)

    lambda_to_y={float(v):i for i,v in enumerate(lambda_values)}
    nearest=lambda target:int(np.argmin(np.abs(diagnostic_steps-target)))
    stage_x=[nearest(0.10*T),nearest(0.45*T),nearest(0.55*T),nearest(0.90*T)]
    images=[None]*3; heatmap_data={}

    for i,dataset in enumerate(dataset_order):
        dset=df[df[dataset_col]==dataset]
        heatmap_data[dataset]={}

        for j,metric in enumerate(metrics):
            ax=axes[i,j]
            pivot=(dset.pivot_table(index=lambda_col,columns=step_col,values=metric,aggfunc="median")
                        .reindex(index=lambda_values,columns=diagnostic_steps))
            arr=pivot.to_numpy(float)

            if metric==dr_weighted_col and 0.0 in lambda_to_y:
                arr[lambda_to_y[0.0],:]=np.nan

            im=ax.imshow(np.ma.masked_invalid(arr),aspect="auto",origin="lower",
                         interpolation="nearest",cmap=plot_cmap,norm=norms[metric])
            if images[j] is None: images[j]=im
            heatmap_data[dataset][metric]=pivot

            if float(highlight_lambda) in lambda_to_y:
                y=lambda_to_y[float(highlight_lambda)]
                ax.add_patch(plt.Rectangle(
                    (-0.5,y-0.5),len(diagnostic_steps),1,fill=False,
                    ec=highlight_color,lw=highlight_linewidth,ls=highlight_linestyle
                ))

            if mark_stage_boundaries:
                for x in stage_x:
                    ax.axvline(x-0.5,color=stage_color,lw=stage_linewidth,
                               ls=stage_linestyle,alpha=stage_alpha)

            if metric==dr_weighted_col and annotate_zero_dr and 0.0 in lambda_to_y:
                y=lambda_to_y[0.0]
                ax.text((len(diagnostic_steps)-1)/2,y,zero_dr_text,
                        ha="center",va="center",color=zero_dr_text_color,
                        fontsize=zero_dr_text_fontsize)

            if i==0: ax.set_title(metric_titles[metric],fontsize=column_title_fontsize)

            if j==0:
                label=dataset_labels.get(dataset,dataset) if dataset_labels else dataset
                ax.set_ylabel(label,fontsize=axis_label_fontsize)
                ax.set_yticks(range(len(lambda_values)))
                ax.set_yticklabels([_lambda_label(v) for v in lambda_values],fontsize=y_tick_fontsize)
            else:
                ax.tick_params(axis="y",labelleft=False)

            tick_idx=np.arange(len(diagnostic_steps)) if len(diagnostic_steps)<=max_x_ticks else \
                     np.unique(np.linspace(0,len(diagnostic_steps)-1,max_x_ticks,dtype=int))
            ax.set_xticks(tick_idx)

            if i==nrows-1:
                ax.set_xticklabels([str(diagnostic_steps[k]) for k in tick_idx],fontsize=x_tick_fontsize)
                ax.set_xlabel("Optimization step",fontsize=axis_label_fontsize)
            else:
                ax.tick_params(axis="x",labelbottom=False)

            ax.tick_params(axis="both",width=tick_width,length=tick_length,pad=tick_pad)
            for spine in ax.spines.values(): spine.set_linewidth(spine_width)

    # ------------------------------------------------------------------
    # Colorbars
    # ------------------------------------------------------------------
    if colorbar_position not in {"top","bottom",None}:
        raise ValueError("colorbar_position must be 'top', 'bottom', or None.")

    cbar_top=None
    if colorbar_position is not None:
        fig.canvas.draw()

        for j,metric in enumerate(metrics):
            col_axes=axes[:,j]
            x0=min(a.get_position().x0 for a in col_axes); x1=max(a.get_position().x1 for a in col_axes)
            y0=min(a.get_position().y0 for a in col_axes); y1=max(a.get_position().y1 for a in col_axes)
            width=x1-x0; cbw=width*colorbar_width_frac; cbx=x0+(width-cbw)/2
            cby=y1+colorbar_pad if colorbar_position=="top" else y0-colorbar_pad-colorbar_height

            cax=fig.add_axes([cbx,cby,cbw,colorbar_height])
            cb=fig.colorbar(images[j],cax=cax,orientation="horizontal")
            mode=scale_modes[metric]

            if mode=="log":
                if colorbar_tick_mode=="decades":
                    ticks=_log_ticks(norms[metric].vmin,norms[metric].vmax,colorbar_max_ticks)
                    if ticks: cb.set_ticks(ticks)
                elif colorbar_tick_mode!="auto":
                    raise ValueError("colorbar_tick_mode must be 'auto' or 'decades'.")
            else:
                cb.locator=MaxNLocator(nbins=colorbar_max_ticks); cb.update_ticks()

            _format_cbar(cb,mode)
            cb.ax.tick_params(labelsize=colorbar_ticksize,width=tick_width,length=tick_length,pad=tick_pad)
            cb.set_label(colorbar_labels[metric],fontsize=colorbar_labelsize)

            for spine in cax.spines.values(): spine.set_linewidth(spine_width)

            if colorbar_position=="top":
                cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
                cbar_top=max(cbar_top or 0,cby+colorbar_height)

    # ------------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------------
    if show_suptitle and suptitle is not None:
        if suptitle_y is None:
            title_y=(cbar_top if colorbar_position=="top" and cbar_top is not None else top)+title_pad
        else:
            title_y=suptitle_y
        fig.suptitle(suptitle,y=title_y,fontsize=title_fontsize)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_path is not None:
        save_path=os.fspath(save_path)
        root,ext=os.path.splitext(save_path)
        if ext.lower() in {".pdf",".png"}: save_path=root

        save_kwargs=dict(dpi=dpi,bbox_inches=bbox_inches,transparent=transparent,facecolor="white")
        if pad_inches is not None: save_kwargs["pad_inches"]=pad_inches

        if save_pdf: fig.savefig(f"{save_path}.pdf",format="pdf",**save_kwargs)
        if save_png: fig.savefig(f"{save_path}.png",format="png",**save_kwargs)

    if show_plot: plt.show()
    return fig,axes,heatmap_data
import os, re, math, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter

def plot_lambda_training_losses(
    logs_dir, dataset, objective_family, lambdas,
    method_contains="pepcl_r_dr", seed=None,
    every_n_steps=1,

    # grid / figure
    nrows=1, ncols=None,
    figsize=(12,7), dpi=300,
    wspace=0.28, hspace=0.55,
    left=0.07, right=0.98, top=0.95, bottom=0.14,

    # fonts
    default_font="Nimbus Sans",
    title_fontsize=11,
    label_fontsize=10,
    tick_fontsize=9,
    legend_fontsize=9,
    title_bold=False,

    # axes
    xlabel="Optimization step",
    ylabel="Loss",
    outer_axis_labels_only=True,
    remove_top_spine=True,
    remove_right_spine=True,
    axis_linewidth=0.6,
    tick_length=3,
    tick_width=0.6,
    xlabel_pad=4,
    ylabel_pad=4,
    y_decimals=3,

    # titles
    title_prefix=None,
    title_suffix=None,
    show_dataset=True,
    lambda_symbol="λ",
    lambda_label=None,
    lambda_in_parens=True,

    # styles
    loss_styles=None,

    # legend
    legend_ncol=4,
    legend_loc="lower center",
    legend_bbox=(0.5,0.015),
    legend_frame=False,
    legend_handlelength=2.4,
    custom_legend_handles=None,

    # saving
    save_path=None,
):
    """
    Plot component-wise PepCL training-loss trajectories across DR lambdas.

    Loss convention
    ---------------
    FT loss         = raw FT loss
    MS replay loss  = raw MS classification loss
    DR loss         = weighted drift loss
    Total loss      = FT raw + replay raw + weighted DR

    For lambda=0:
        weighted DR = 0
        total = FT + replay

    Grid behavior
    -------------
    If ncols=None:
        ncols = ceil(len(lambdas) / nrows)

    outer_axis_labels_only=True:
        - ylabel only on leftmost column
        - xlabel only on bottom row

    y_decimals:
        Maximum decimal precision allowed for y-axis ticks.
        Tick formatting is uniform within each individual axis.

    lambda_in_parens:
        If True (default), the lambda term is wrapped in parentheses in the
        panel title, e.g. "DatasetName (λ = 1e-5)".
        If False, it is shown without parentheses, e.g. "DatasetName λ = 1e-5".

    Examples
    --------
    Natural ticks:
        0, 0.025, 0.050, 0.075, 0.100

    y_decimals=3:
        0.000, 0.025, 0.050, 0.075, 0.100

    y_decimals=2:
        0.00, 0.05, 0.10
        # incompatible intermediate ticks are removed

    Natural ticks:
        0, 0.1, 0.2, 0.3, 0.4

    y_decimals=3:
        0.0, 0.1, 0.2, 0.3, 0.4
        # reduced uniformly because every tick supports one decimal

    Returns
    -------
    fig, axes, loss_df
    """
    logs_dir=os.fspath(logs_dir)

    if objective_family not in {"classification","continuous"}:
        raise ValueError("objective_family must be 'classification' or 'continuous'.")
    if every_n_steps<1:
        raise ValueError("every_n_steps must be >= 1.")
    if nrows<1:
        raise ValueError("nrows must be >= 1.")
    if y_decimals is not None and y_decimals<0:
        raise ValueError("y_decimals must be >= 0 or None.")

    lambdas=[float(x) for x in lambdas]

    # ==============================================================
    # Grid shape
    # ==============================================================
    if ncols is None:
        ncols=math.ceil(len(lambdas)/nrows)

    if ncols<1:
        raise ValueError("ncols must be >= 1.")

    if nrows*ncols<len(lambdas):
        raise ValueError(
            f"Grid {nrows}x{ncols} has only {nrows*ncols} panels "
            f"for {len(lambdas)} lambdas."
        )

    # ==============================================================
    # Matplotlib font setup
    # ==============================================================
    plt.rcParams["font.family"]="sans-serif"
    plt.rcParams["font.sans-serif"]=[
        default_font,"Nimbus Sans L","Arial","Helvetica"
    ]
    plt.rcParams["pdf.fonttype"]=42

    # ==============================================================
    # Default styles
    # ==============================================================
    default_loss_styles={
        "FT loss":{
            "color":"#8f8f8f",
            "linestyle":"-",
            "linewidth":1.0,
            "alpha":1.0,
            "marker":None,
            "markersize":3,
            "label":"FT loss (LogSmoothAP)"
                if objective_family=="classification"
                else "FT loss (ranking)",
            "legend_color":"#8f8f8f",
            "legend_linestyle":"-",
            "legend_linewidth":1.2,
            "legend_marker":None,
        },

        "MS replay loss":{
            "color":"#DA7C81",
            "linestyle":"-",
            "linewidth":1.0,
            "alpha":1.0,
            "marker":None,
            "markersize":3,
            "label":"MS replay loss (LogSmoothAP)",
            "legend_color":"#DA7C81",
            "legend_linestyle":"-",
            "legend_linewidth":1.2,
            "legend_marker":None,
        },

        "Weighted DR loss":{
            "color":"#6D83FF",
            "linestyle":"-",
            "linewidth":1.0,
            "alpha":1.0,
            "marker":None,
            "markersize":3,
            "label":"λDR loss (MSE)",
            "legend_color":"#6D83FF",
            "legend_linestyle":"-",
            "legend_linewidth":1.2,
            "legend_marker":None,
        },

        "Total loss":{
            "color":"#ffb55c",
            "linestyle":"-",
            "linewidth":1.0,
            "alpha":1.0,
            "marker":None,
            "markersize":3,
            "label":"Total loss",
            "legend_color":"#ffb55c",
            "legend_linestyle":"-",
            "legend_linewidth":1.2,
            "legend_marker":None,
        },
    }

    def _merge_styles(defaults,user):
        out={k:dict(v) for k,v in defaults.items()}

        if user is not None:
            for k,v in user.items():
                if k not in out:
                    out[k]={}
                out[k].update(v)

        return out

    loss_styles=_merge_styles(default_loss_styles,loss_styles)

    # ==============================================================
    # Helpers
    # ==============================================================
    def _lambda_float(x):
        if isinstance(x,(int,float,np.integer,np.floating)):
            return float(x)

        s=str(x)

        if "eN" in s:
            a,b=s.split("eN",1)
            return float(a)*10**(-int(b))

        return float(s)

    def _format_lambda(x):
        x=float(x)

        if x==0.0:
            return "0"

        s=f"{x:.0e}"
        s=re.sub(r"e([+-])0+(\d+)",r"e\1\2",s)
        s=s.replace("e+","e")

        return s

    def _build_title(lam):
        lam_text=_format_lambda(lam)

        core=(
            f"{lambda_symbol} = {lam_text}"
            if lambda_label is None
            else f"{lambda_label} = {lam_text}"
        )

        parts=[]

        if title_prefix:
            parts.append(str(title_prefix))

        if show_dataset:
            parts.append(str(dataset))

        base=" ".join(parts)

        if base:
            title=f"{base} ({core})" if lambda_in_parens else f"{base} {core}"
        else:
            title=f"({core})" if lambda_in_parens else core

        if title_suffix:
            title=f"{title} {title_suffix}"

        return title

    def _value_fits_decimals(x,decimals,tol=1e-10):
        return np.isclose(
            x,
            np.round(x,decimals),
            rtol=0.0,
            atol=tol,
        )

    def _format_y_axis(ax,max_decimals):
        """
        Keep one uniform precision across the entire y axis.

        Ticks requiring more than max_decimals are removed.
        Among the remaining ticks, use the lowest decimal precision
        capable of representing every tick exactly.
        """
        if max_decimals is None:
            return

        ticks=np.asarray(ax.get_yticks(),dtype=float)

        if len(ticks)==0:
            return

        ymin,ymax=ax.get_ylim()
        eps=max(abs(ymax-ymin)*1e-10,1e-12)

        ticks=ticks[
            (ticks>=ymin-eps)&
            (ticks<=ymax+eps)
        ]

        keep=np.asarray([
            t for t in ticks
            if _value_fits_decimals(t,max_decimals)
        ],dtype=float)

        if len(keep)==0:
            return

        decimals=max_decimals

        for d in range(max_decimals+1):
            if all(_value_fits_decimals(t,d) for t in keep):
                decimals=d
                break

        keep[np.isclose(
            keep,0.0,
            rtol=0.0,
            atol=1e-12,
        )]=0.0

        ax.yaxis.set_major_locator(
            FixedLocator(keep)
        )

        ax.yaxis.set_major_formatter(
            FormatStrFormatter(
                "%.0f" if decimals==0
                else f"%.{decimals}f"
            )
        )

    def _find_training_stats(log):
        candidates=[
            log.get("training_stats"),
            log.get("training_metrics"),
            log.get("metrics"),
            log.get("losses"),
        ]

        for x in candidates:
            if isinstance(x,pd.DataFrame) and not x.empty:
                return x.copy()

            if isinstance(x,dict) and x:
                try:
                    df=pd.DataFrame(x)

                    if not df.empty:
                        return df

                except (ValueError,TypeError):
                    pass

        try:
            df=pd.DataFrame(log)

            if {
                "ft_loss_raw",
                "ms_classification_loss_raw",
            }.intersection(df.columns):
                return df

        except (ValueError,TypeError):
            pass

        raise ValueError(
            "Could not find usable training-loss statistics in log."
        )

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lambda>.+?)"
        r"_seed_(?P<seed>\d+)(?:_step_(?P<checkpoint>\d+))?$"
    )

    # ==============================================================
    # Locate logs
    # ==============================================================
    files=sorted(
        f for f in os.listdir(logs_dir)
        if f.endswith(".pkl") and method_contains in f
    )

    if not files:
        raise ValueError(
            f"No .pkl files containing '{method_contains}' found in:\n"
            f"{logs_dir}"
        )

    matches={lam:[] for lam in lambdas}

    for filename in files:
        stem=os.path.splitext(filename)[0]
        m=pattern.search(stem)

        if m is None:
            continue

        lam=_lambda_float(m["lambda"])
        file_seed=int(m["seed"])

        if seed is not None and file_seed!=seed:
            continue

        # Exact equality is intentional.
        for target_lambda in lambdas:
            if lam==target_lambda:
                matches[target_lambda].append({
                    "filename":filename,
                    "seed":file_seed,
                    "budget":int(m["budget"]),
                    "lr":m["lr"],
                    "checkpoint":(
                        None if m["checkpoint"] is None
                        else int(m["checkpoint"])
                    ),
                })

    # ==============================================================
    # Extract training-loss trajectories
    # ==============================================================
    rows=[]

    for lam in lambdas:
        candidates=matches[lam]

        if not candidates:
            raise ValueError(
                f"No matching log found for lambda={_format_lambda(lam)}"
            )

        full_logs=[
            x for x in candidates
            if x["checkpoint"] is None
        ]

        candidates=full_logs if full_logs else candidates

        if len(candidates)>1:
            seeds=sorted(set(x["seed"] for x in candidates))

            raise ValueError(
                f"Multiple logs found for lambda={_format_lambda(lam)}: "
                f"{len(candidates)} files, seeds={seeds}. "
                f"Set seed=... or make the selection more specific."
            )

        info=candidates[0]

        with open(os.path.join(logs_dir,info["filename"]),"rb") as f:
            log=pickle.load(f)

        if not isinstance(log,dict):
            raise ValueError(
                f"Expected dictionary log: {info['filename']}"
            )

        stats=_find_training_stats(log)

        if "step" in stats.columns:
            steps=pd.to_numeric(
                stats["step"],
                errors="coerce",
            ).to_numpy()

        elif "optimization_step" in stats.columns:
            steps=pd.to_numeric(
                stats["optimization_step"],
                errors="coerce",
            ).to_numpy()

        else:
            steps=np.arange(1,len(stats)+1)

        def _get(col):
            if col not in stats.columns:
                return np.full(
                    len(stats),
                    np.nan,
                    dtype=float,
                )

            return pd.to_numeric(
                stats[col],
                errors="coerce",
            ).to_numpy(dtype=float)

        assay_raw=_get("ft_loss_raw")
        replay_raw=_get("ms_classification_loss_raw")
        dr_raw=_get("drift_loss_raw")
        dr_weighted=_get("drift_loss_weighted")

        missing=~np.isfinite(dr_weighted) & np.isfinite(dr_raw)
        dr_weighted[missing]=lam*dr_raw[missing]

        # Only literal lambda=0 is replay-only.
        if lam==0.0:
            dr_weighted=np.zeros(
                len(stats),
                dtype=float,
            )

        total=assay_raw+replay_raw+dr_weighted

        for i,step in enumerate(steps):
            if not np.isfinite(step):
                continue

            step=int(step)

            if every_n_steps>1 and (step-1)%every_n_steps!=0:
                continue

            rows.extend([
                {
                    "dataset":dataset,
                    "objective_family":objective_family,
                    "lambda":lam,
                    "seed":info["seed"],
                    "step_budget":info["budget"],
                    "learning_rate":info["lr"],
                    "step":step,
                    "loss_name":"assay_loss_raw",
                    "loss":assay_raw[i],
                },
                {
                    "dataset":dataset,
                    "objective_family":objective_family,
                    "lambda":lam,
                    "seed":info["seed"],
                    "step_budget":info["budget"],
                    "learning_rate":info["lr"],
                    "step":step,
                    "loss_name":"ms_replay_loss_raw",
                    "loss":replay_raw[i],
                },
                {
                    "dataset":dataset,
                    "objective_family":objective_family,
                    "lambda":lam,
                    "seed":info["seed"],
                    "step_budget":info["budget"],
                    "learning_rate":info["lr"],
                    "step":step,
                    "loss_name":"drift_loss_weighted",
                    "loss":dr_weighted[i],
                },
                {
                    "dataset":dataset,
                    "objective_family":objective_family,
                    "lambda":lam,
                    "seed":info["seed"],
                    "step_budget":info["budget"],
                    "learning_rate":info["lr"],
                    "step":step,
                    "loss_name":"total_loss",
                    "loss":total[i],
                },
            ])

    loss_df=pd.DataFrame(rows)

    # ==============================================================
    # Figure
    # ==============================================================
    fig,axes=plt.subplots(
        nrows,ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    flat_axes=axes.ravel()

    plot_components=[
        ("assay_loss_raw","FT loss"),
        ("ms_replay_loss_raw","MS replay loss"),
        ("drift_loss_weighted","Weighted DR loss"),
        ("total_loss","Total loss"),
    ]

    for panel_idx,(ax,lam) in enumerate(zip(flat_axes,lambdas)):
        row=panel_idx//ncols
        col=panel_idx%ncols

        d=loss_df[
            loss_df["lambda"]==lam
        ]

        for loss_name,style_key in plot_components:
            style=loss_styles[style_key]

            x=d[
                d["loss_name"]==loss_name
            ].sort_values("step")

            if x.empty:
                continue

            ax.plot(
                x["step"],
                x["loss"],
                color=style.get("color"),
                linestyle=style.get("linestyle","-"),
                linewidth=style.get("linewidth",1.0),
                alpha=style.get("alpha",1.0),
                marker=style.get("marker",None),
                markersize=style.get("markersize",3),
            )

        ax.set_title(
            _build_title(lam),
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight="bold" if title_bold else "normal",
        )

        # ----------------------------------------------------------
        # Outer-only axis titles
        # ----------------------------------------------------------
        if outer_axis_labels_only:
            ax.set_ylabel(
                ylabel if col==0 else "",
                fontsize=label_fontsize,
                fontfamily=default_font,
                labelpad=ylabel_pad,
            )

            ax.set_xlabel(
                xlabel if row==nrows-1 else "",
                fontsize=label_fontsize,
                fontfamily=default_font,
                labelpad=xlabel_pad,
            )

        else:
            ax.set_ylabel(
                ylabel,
                fontsize=label_fontsize,
                fontfamily=default_font,
                labelpad=ylabel_pad,
            )

            ax.set_xlabel(
                xlabel,
                fontsize=label_fontsize,
                fontfamily=default_font,
                labelpad=xlabel_pad,
            )

        ax.set_yscale("linear")

        ax.spines["left"].set_linewidth(axis_linewidth)
        ax.spines["bottom"].set_linewidth(axis_linewidth)

        ax.spines["top"].set_visible(
            not remove_top_spine
        )
        ax.spines["right"].set_visible(
            not remove_right_spine
        )

        if not remove_top_spine:
            ax.spines["top"].set_linewidth(
                axis_linewidth
            )

        if not remove_right_spine:
            ax.spines["right"].set_linewidth(
                axis_linewidth
            )

        ax.tick_params(
            axis="both",
            which="major",
            length=tick_length,
            width=tick_width,
            direction="out",
            labelsize=tick_fontsize,
        )

        _format_y_axis(
            ax,
            y_decimals,
        )

        for lbl in (
            ax.get_xticklabels()+
            ax.get_yticklabels()
        ):
            lbl.set_fontfamily(default_font)

    # Hide unused cells.
    for ax in flat_axes[len(lambdas):]:
        ax.set_visible(False)

    # ==============================================================
    # Shared legend
    # ==============================================================
    if custom_legend_handles is None:
        handles=[]

        for key in [
            "FT loss",
            "MS replay loss",
            "Weighted DR loss",
            "Total loss",
        ]:
            s=loss_styles[key]

            handles.append(
                Line2D(
                    [0],[0],
                    color=s.get(
                        "legend_color",
                        s.get("color"),
                    ),
                    linestyle=s.get(
                        "legend_linestyle",
                        s.get("linestyle","-"),
                    ),
                    linewidth=s.get(
                        "legend_linewidth",
                        s.get("linewidth",1.0),
                    ),
                    marker=s.get(
                        "legend_marker",
                        None,
                    ),
                    label=s.get(
                        "label",
                        key,
                    ),
                )
            )

    else:
        handles=custom_legend_handles

    fig.legend(
        handles=handles,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
        frameon=legend_frame,
        fontsize=legend_fontsize,
        handlelength=legend_handlelength,
    )

    # ==============================================================
    # Spacing
    # ==============================================================
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
    )

    # ==============================================================
    # Save
    # ==============================================================
    if save_path is not None:
        fig.savefig(
            f"{save_path}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=dpi,
        )

        fig.savefig(
            f"{save_path}.png",
            format="png",
            bbox_inches="tight",
            dpi=dpi,
        )

    return fig,axes,loss_df
import pandas as pd

def filter_dicts_to_common_alleles(df_dict, keys, verbose=True):
    common = None
    for k in keys:
        idx = set(df_dict[k].index)
        common = idx if common is None else (common & idx)

    out = {}
    for k in keys:
        before = len(df_dict[k])
        filtered = df_dict[k].loc[df_dict[k].index.isin(common)].copy()
        removed = before - len(filtered)
        out[k] = filtered
        if verbose:
            print(f"{k}: {before} -> {len(filtered)} alleles ({removed} removed)")

    if verbose:
        all_alleles = set()
        for k in keys:
            all_alleles |= set(df_dict[k].index)
        mismatched = sorted(all_alleles - common)
        print(f"common alleles: {len(common)} | total across keys: {len(all_alleles)}")
        if mismatched:
            print(f"mismatched (not in all): {len(mismatched)} -> {mismatched}")
        else:
            print("no mismatches; all keys shared the same alleles")
    return out
def plot_lambda_metric(
    lambda_stats, metric, method="pepcl_r_dr",
    error="ci95",

    # Lambda ordering / selected lambda.
    categorical_lambda_order="ascending",
    highlight_lambda=None,
    crosshair_lambda=None,
    crosshair_mode="vertical",

    # Line / point styling.
    line_color="black",
    marker="o",
    marker_size=7,
    linewidth=1.8,
    linestyle="-",
    point_alpha=1.0,
    error_alpha=1.0,
    error_kwargs=None,
    highlight_size=180,

    # Crosshair styling.
    crosshair_line_kwargs=None,

    # Labels / limits.
    title=None,
    xlabel="DR λ",
    ylabel=None,
    xlim=None,
    ylim=None,

    # Grid geometry.
    grid_shape=(2,3),
    fig_size_single=(5.2,5.2),
    fig_size_grid=(15,9),
    square_panels=True,
    wspace=.35,
    hspace=.35,

    # General styling.
    dpi=300,
    axis_label_fontsize=11,
    title_fontsize=12,
    tick_decimals=2,
    trim_tick_zeros=True,
    n_ticks=5,
    tick_fontsize=9,
    tick_length=4,
    tick_width=1.2,
    tick_pad=5,
    spine_width=1.4,
    grid=False,
    grid_alpha=.2,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot one lambda-dependent metric from one lambda summary dataframe or
    a dictionary of lambda summary dataframes.

    Parameters
    ----------
    lambda_stats : pd.DataFrame or dict[str, pd.DataFrame]
        Output(s) from summarize_lambda_performance().

        DataFrame:
            -> one plot.

        Dict:
            -> one subplot per dictionary entry.
               Dictionary keys become subplot titles.

    metric : str or dict[str, str]
        Metric column to plot.

        String:
            same metric is used for every dataset.

        Dictionary:
            dataset key -> metric column.

            Example:
                {
                    "YD": "brnb",
                    "EpiScan R9": "brnb",
                    "BA": "net_benefit",
                }

    ylabel : str, dict[str, str], or None
        String:
            common y-axis label.

        Dictionary:
            dataset key -> y-axis label.

        None:
            uses the corresponding metric name.

    error : {None, "sd", "se", "ci95"}
        Error-bar type. For each dataset the expected column is:
            f"{dataset_metric}_{error}"

    Lambda is plotted categorically, so:
        0, 1e-8, 1e-7, ..., 1e-2
    are equally spaced visually.

    Returns
    -------
    dict with:
        "lambda_dfs" : filtered dataframes used for plotting
        "metrics"    : dataset -> metric mapping actually used
        "ylabels"    : dataset -> ylabel mapping actually used
        "fig"        : matplotlib figure
        "axes"       : plotted axes
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        "Nimbus Sans","Nimbus Sans L","Arial","Helvetica"
    ]
    matplotlib.rcParams["pdf.fonttype"]=42

    valid_errors={None,"sd","se","ci95"}
    valid_crosshair_modes={"none","vertical","horizontal","both"}

    if error not in valid_errors:
        raise ValueError(
            "error must be None, 'sd', 'se', or 'ci95'."
        )

    if crosshair_mode not in valid_crosshair_modes:
        raise ValueError(
            "crosshair_mode must be 'none', 'vertical', "
            "'horizontal', or 'both'."
        )

    if categorical_lambda_order not in {
        "ascending","descending"
    }:
        raise ValueError(
            "categorical_lambda_order must be "
            "'ascending' or 'descending'."
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _lambda_float(x):
        if isinstance(x,str):
            x=x.strip().lower().replace("−","-")
        return float(x)

    def _lambda_numeric_label(x):
        x=float(x)

        if x==0:
            return "0"

        return (
            f"{x:.0e}"
            .replace("e-0","e-")
            .replace("e+0","e+")
        )

    def _yfmt(v,_):
        if abs(v)<10**(-(tick_decimals+2)):
            v=0

        s=f"{v:.{tick_decimals}f}"

        return (
            s.rstrip("0").rstrip(".")
            if trim_tick_zeros
            else s
        )

    error_kwargs={
        "elinewidth":1.0,
        "capsize":3,
        "capthick":1.0,
        "zorder":2,
        **(error_kwargs or {})
    }

    crosshair_line_kwargs={
        "color":"0.45",
        "linestyle":"--",
        "linewidth":1.2,
        "alpha":1.0,
        "zorder":0,
        **(crosshair_line_kwargs or {})
    }

    # ---------------------------------------------------------
    # Normalize single dataframe vs dictionary input.
    # ---------------------------------------------------------
    single=isinstance(
        lambda_stats,
        pd.DataFrame
    )

    if single:
        dataset_name=(
            title
            if title is not None
            else ""
        )

        stats_dict={
            dataset_name:lambda_stats
        }

    elif isinstance(lambda_stats,dict):
        if not lambda_stats:
            raise ValueError(
                "lambda_stats dictionary is empty."
            )

        stats_dict=dict(
            lambda_stats
        )

    else:
        raise TypeError(
            "lambda_stats must be a pandas DataFrame "
            "or dict of DataFrames."
        )

    dataset_names=list(
        stats_dict
    )

    # ---------------------------------------------------------
    # Resolve dataset-specific metrics.
    # ---------------------------------------------------------
    if isinstance(metric,str):
        metric_dict={
            name:metric
            for name in dataset_names
        }

    elif isinstance(metric,dict):
        if single:
            raise ValueError(
                "A metric dictionary is only supported when "
                "lambda_stats is also a dictionary."
            )

        missing_metrics=[
            name
            for name in dataset_names
            if name not in metric
        ]

        if missing_metrics:
            raise ValueError(
                "metric dictionary is missing dataset keys: "
                f"{missing_metrics}"
            )

        metric_dict={
            name:metric[name]
            for name in dataset_names
        }

    else:
        raise TypeError(
            "metric must be a string or dictionary of "
            "dataset -> metric."
        )

    # ---------------------------------------------------------
    # Resolve dataset-specific y-axis labels.
    # ---------------------------------------------------------
    if ylabel is None:
        ylabel_dict={
            name:metric_dict[name]
            for name in dataset_names
        }

    elif isinstance(ylabel,str):
        ylabel_dict={
            name:ylabel
            for name in dataset_names
        }

    elif isinstance(ylabel,dict):
        if single:
            raise ValueError(
                "A ylabel dictionary is only supported when "
                "lambda_stats is also a dictionary."
            )

        missing_ylabels=[
            name
            for name in dataset_names
            if name not in ylabel
        ]

        if missing_ylabels:
            raise ValueError(
                "ylabel dictionary is missing dataset keys: "
                f"{missing_ylabels}"
            )

        ylabel_dict={
            name:ylabel[name]
            for name in dataset_names
        }

    else:
        raise TypeError(
            "ylabel must be None, a string, or a dictionary "
            "of dataset -> ylabel."
        )

    # ---------------------------------------------------------
    # Filter / validate each dataframe independently using its
    # corresponding metric.
    # ---------------------------------------------------------
    plot_dfs={}

    for name,df in stats_dict.items():
        if not isinstance(df,pd.DataFrame):
            raise TypeError(
                f"{name!r}: expected pandas DataFrame, "
                f"got {type(df).__name__}."
            )

        d=df.copy()
        current_metric=metric_dict[name]

        if "method" in d.columns:
            d=d[
                d["method"]==method
            ].copy()

        if d.empty:
            raise ValueError(
                f"{name!r}: no rows remain for "
                f"method={method!r}."
            )

        if "lambda" not in d.columns:
            raise ValueError(
                f"{name!r}: missing required column 'lambda'."
            )

        if current_metric not in d.columns:
            raise ValueError(
                f"{name!r}: metric "
                f"{current_metric!r} not found."
            )

        if error is not None:
            err_col=(
                f"{current_metric}_{error}"
            )

            if err_col not in d.columns:
                raise ValueError(
                    f"{name!r}: error column "
                    f"{err_col!r} not found."
                )

        d["lambda"]=d[
            "lambda"
        ].astype(float)

        d=d.sort_values(
            "lambda",
            ascending=(
                categorical_lambda_order
                =="ascending"
            )
        ).reset_index(drop=True)

        plot_dfs[name]=d

    # ---------------------------------------------------------
    # Common categorical lambda axis across all panels.
    # ---------------------------------------------------------
    all_lambdas=sorted(
        {
            float(v)
            for d in plot_dfs.values()
            for v in d["lambda"].dropna()
        },
        reverse=(
            categorical_lambda_order
            =="descending"
        )
    )

    xpos={
        float(v):i
        for i,v in enumerate(
            all_lambdas
        )
    }

    lambda_tick_labels=[
        _lambda_numeric_label(v)
        for v in all_lambdas
    ]

    # ---------------------------------------------------------
    # Figure / axes
    # ---------------------------------------------------------
    if single:
        fig,ax=plt.subplots(
            1,1,
            figsize=fig_size_single,
            dpi=dpi
        )

        axes=np.array(
            [ax],
            dtype=object
        )

    else:
        nrows,ncols=grid_shape
        capacity=nrows*ncols

        if len(plot_dfs)>capacity:
            raise ValueError(
                f"grid_shape={grid_shape} holds "
                f"{capacity} panels, but "
                f"{len(plot_dfs)} dataframes were supplied."
            )

        fig,axs=plt.subplots(
            nrows,
            ncols,
            figsize=fig_size_grid,
            dpi=dpi,
            squeeze=False
        )

        axes=axs.ravel()

    # ---------------------------------------------------------
    # Draw one trajectory per dataset.
    # ---------------------------------------------------------
    used_axes=[]

    for ax,(name,d) in zip(
        axes,
        plot_dfs.items()
    ):
        used_axes.append(ax)

        current_metric=metric_dict[name]
        current_ylabel=ylabel_dict[name]

        xx=np.array(
            [
                xpos[float(v)]
                for v in d["lambda"]
            ],
            dtype=float
        )

        yy=d[
            current_metric
        ].to_numpy(float)

        # -----------------------------------------------------
        # Connecting line
        # -----------------------------------------------------
        ax.plot(
            xx,
            yy,
            color=line_color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=point_alpha,
            zorder=2
        )

        # -----------------------------------------------------
        # Points
        # -----------------------------------------------------
        ax.scatter(
            xx,
            yy,
            s=marker_size**2,
            marker=marker,
            color=line_color,
            edgecolor="black",
            linewidth=.6,
            alpha=point_alpha,
            zorder=4
        )

        # -----------------------------------------------------
        # Error bars
        # -----------------------------------------------------
        if error is not None:
            err_col=(
                f"{current_metric}_{error}"
            )

            for x_i,(_,r) in zip(
                xx,
                d.iterrows()
            ):
                err=float(
                    r[err_col]
                )

                if not np.isfinite(err):
                    continue

                ax.errorbar(
                    x_i,
                    float(
                        r[current_metric]
                    ),
                    yerr=err,
                    fmt="none",
                    ecolor=line_color,
                    alpha=error_alpha,
                    **error_kwargs
                )

        # -----------------------------------------------------
        # Highlight lambda
        # -----------------------------------------------------
        if highlight_lambda is not None:
            target=_lambda_float(
                highlight_lambda
            )

            hit=d[
                np.isclose(
                    d["lambda"].astype(float),
                    target,
                    rtol=1e-12,
                    atol=0
                )
            ]

            if not hit.empty:
                r=hit.iloc[0]

                ax.scatter(
                    xpos[
                        float(r["lambda"])
                    ],
                    float(
                        r[current_metric]
                    ),
                    s=highlight_size,
                    marker="*",
                    color=line_color,
                    edgecolor="black",
                    linewidth=1.1,
                    alpha=point_alpha,
                    zorder=6
                )

        # -----------------------------------------------------
        # Crosshair
        # -----------------------------------------------------
        cross_lam=(
            crosshair_lambda
            if crosshair_lambda is not None
            else highlight_lambda
        )

        if (
            cross_lam is not None
            and crosshair_mode!="none"
        ):
            target=_lambda_float(
                cross_lam
            )

            hit=d[
                np.isclose(
                    d["lambda"].astype(float),
                    target,
                    rtol=1e-12,
                    atol=0
                )
            ]

            if not hit.empty:
                r=hit.iloc[0]

                hx=xpos[
                    float(r["lambda"])
                ]

                hy=float(
                    r[current_metric]
                )

                if crosshair_mode in {
                    "vertical","both"
                }:
                    ax.axvline(
                        hx,
                        **crosshair_line_kwargs
                    )

                if crosshair_mode in {
                    "horizontal","both"
                }:
                    ax.axhline(
                        hy,
                        **crosshair_line_kwargs
                    )

        # -----------------------------------------------------
        # Categorical lambda axis
        # -----------------------------------------------------
        ax.set_xticks(
            np.arange(
                len(all_lambdas)
            )
        )

        ax.set_xticklabels(
            lambda_tick_labels,
            fontsize=tick_fontsize,
            rotation=0
        )

        if xlim is not None:
            ax.set_xlim(
                xlim
            )
        else:
            ax.set_xlim(
                -.35,
                len(all_lambdas)-.65
            )

        ax.set_xlabel(
            xlabel,
            fontsize=axis_label_fontsize,
            labelpad=8
        )

        ax.set_ylabel(
            current_ylabel,
            fontsize=axis_label_fontsize,
            labelpad=8
        )

        # -----------------------------------------------------
        # Titles
        # -----------------------------------------------------
        if single:
            if title is not None:
                ax.set_title(
                    title,
                    fontsize=title_fontsize
                )

        else:
            ax.set_title(
                str(name),
                fontsize=title_fontsize
            )

        # -----------------------------------------------------
        # Limits / ticks
        # -----------------------------------------------------
        if ylim is not None:
            ax.set_ylim(
                ylim
            )

        if n_ticks is not None:
            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=n_ticks
                )
            )

        ax.yaxis.set_major_formatter(
            FuncFormatter(
                _yfmt
            )
        )

        ax.tick_params(
            axis="both",
            labelsize=tick_fontsize,
            length=tick_length,
            width=tick_width,
            pad=tick_pad
        )

        for spine in ax.spines.values():
            spine.set_linewidth(
                spine_width
            )

        if grid:
            ax.grid(
                True,
                alpha=grid_alpha
            )
        else:
            ax.grid(False)

        if square_panels:
            ax.set_box_aspect(1)

    # ---------------------------------------------------------
    # Hide unused panels
    # ---------------------------------------------------------
    if not single:
        for ax in axes[
            len(plot_dfs):
        ]:
            ax.axis("off")

        fig.subplots_adjust(
            wspace=wspace,
            hspace=hspace
        )

    # ---------------------------------------------------------
    # Save / show
    # ---------------------------------------------------------
    if save_path:
        fig.savefig(
            f"{save_path}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=dpi
        )

        fig.savefig(
            f"{save_path}.png",
            format="png",
            bbox_inches="tight",
            dpi=dpi
        )

    if show_plot:
        plt.show()

    return {
        "lambda_dfs":plot_dfs,
        "metrics":metric_dict,
        "ylabels":ylabel_dict,
        "fig":fig,
        "axes":used_axes,
    }
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_per_allele_lambda_metric(
    df, metric, lambda_colors,

    # ---------------------------------------------------------
    # Data selection
    # ---------------------------------------------------------
    method="pepcl_r_dr",
    method_col="method",
    allele_col="allele",
    lambda_col="lambda_string",
    lambda_order_col="lambda",

    # ---------------------------------------------------------
    # Labels / titles
    # ---------------------------------------------------------
    title=None,
    xlabel=None,
    ylabel="Allele",

    # ---------------------------------------------------------
    # Splitting / figure geometry
    # ---------------------------------------------------------
    split_n=1,
    panel_figsize=(6,8),
    allele_slots_per_panel=20,
    left_margin=0.12,
    right_margin=0.97,
    top_margin=0.91,
    plot_bottom=0.18,
    panel_wspace=0.07,

    # ---------------------------------------------------------
    # Main point styling
    # ---------------------------------------------------------
    marker="o",
    marker_size=32,
    point_alpha=1.0,

    # Marker outlines
    marker_outline=False,
    marker_edgecolor="black",
    marker_edgewidth=0.8,

    # ---------------------------------------------------------
    # Grid styling
    # ---------------------------------------------------------
    grid=True,
    grid_axis="both",
    grid_alpha=0.22,
    grid_linewidth=0.8,
    grid_linestyle="-",

    # ---------------------------------------------------------
    # Axis / tick styling
    # ---------------------------------------------------------
    xlim=None,
    spine_width=1.0,
    tick_fontsize=9,
    tick_length=4,
    tick_width=1.0,
    tick_pad=5,

    # ---------------------------------------------------------
    # Text styling
    # ---------------------------------------------------------
    title_fontsize=12,
    label_fontsize=10,
    allele_fontsize=9,

    # ---------------------------------------------------------
    # Legend styling
    # ---------------------------------------------------------
    legend_title="DR λ",
    legend_nrows=1,
    legend_fontsize=9,
    legend_title_fontsize=10,
    legend_marker_size=7,
    legend_marker_alpha=1.0,

    # Legend marker outlines
    legend_marker_outline=None,
    legend_marker_edgecolor=None,
    legend_marker_edgewidth=None,

    legend_frame=False,
    legend_frame_alpha=1.0,
    legend_columnspacing=1.5,
    legend_handletextpad=0.5,
    legend_handlelength=1.0,
    legend_labelspacing=0.5,
    legend_y_pad=0.055,

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    save_path=None,
    dpi=300,
    show=True,
):
    """
    Plot one lambda-dependent metric for each allele.

    Intended input:
        summarize_lambda_performance(..., per_allele=True)

    Each point represents:
        one allele x one lambda setting

    If multiple methods are present in df, only `method` is plotted.

    Parameters
    ----------
    method : str
        Method to plot. Default "pepcl_r_dr".

    split_n : int
        Number of side-by-side panels used to split alleles.

    allele_slots_per_panel : int or None
        Controls PHYSICAL allele-row spacing, but does NOT create empty
        y-axis rows.

        Example:
            split contains 8 alleles
            allele_slots_per_panel=20

        -> axis contains exactly 8 allele rows
        -> those rows have the physical spacing of a 20-row reference panel
        -> remaining figure area is plain white space

    marker_outline : bool
        If False, plot points have no marker edge.
        If True, marker_edgecolor / marker_edgewidth are used.

    legend_marker_outline : bool or None
        If None, inherits marker_outline.

    Returns
    -------
    dict with:
        fig
        axes
        plot_df
        alleles
        allele_chunks
        lambda_order
        lambda_labels
    """

    # =========================================================
    # Helpers
    # =========================================================
    def _lambda_float(x):
        if pd.isna(x):
            return np.nan

        s=str(x).strip().lower().replace("−","-")

        if s in {"0","0.0","0e0","0e+0","0e-0"}:
            return 0.0

        # Project convention: 1eN8 -> 1e-8
        s=s.replace("en","e-")

        try:
            return float(s)
        except ValueError:
            raise ValueError(
                f"Could not parse lambda value {x!r}."
            )

    def _lambda_label(x):
        """
        Numerical scientific notation with no accidental rounding to 0.

        Examples:
            0       -> 0
            1e-8    -> 1e-8
            1e-5    -> 1e-5
            0.001   -> 1e-3
        """
        x=float(x)

        if x==0:
            return "0"

        s=f"{x:.10e}"
        mantissa,exponent=s.split("e")

        mantissa=mantissa.rstrip("0").rstrip(".")
        exponent=int(exponent)

        return f"{mantissa}e{exponent}"

    def _format_allele(a):
        """
        Convert allele forms such as:
            A0201       -> A*02:01
            HLA-A0201   -> A*02:01
            A02:01      -> A*02:01
            HLA-A*02:01 -> A*02:01
        """
        s=str(a).strip().upper()
        s=s.replace("HLA-","")

        # Already formatted, e.g. A*02:01
        m=re.match(r"^([ABC])\*(\d+):(\d+)$",s)
        if m:
            return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"

        # A02:01
        m=re.match(r"^([ABC])(\d+):(\d+)$",s)
        if m:
            return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"

        # A0201 / B3502 / C1202
        m=re.match(r"^([ABC])(\d{2})(\d{2,})$",s)
        if m:
            return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"

        return str(a)

    def _allele_sort_key(a):
        s=_format_allele(a)

        m=re.match(
            r"^([ABC])\*(\d+):(\d+)$",
            s
        )

        if m:
            return (
                {"A":0,"B":1,"C":2}[m.group(1)],
                int(m.group(2)),
                int(m.group(3)),
                s
            )

        return (
            3,
            float("inf"),
            float("inf"),
            s
        )

    # Normalize lambda color dictionary numerically.
    numeric_lambda_colors={}

    for key,color in lambda_colors.items():
        numeric_lambda_colors[
            _lambda_float(key)
        ]=color

    def _lambda_color(value):
        value=float(value)

        candidates=np.array(
            list(numeric_lambda_colors),
            dtype=float
        )

        hit=np.where(
            np.isclose(
                candidates,
                value,
                rtol=1e-12,
                atol=0
            )
        )[0]

        if len(hit)==0:
            raise ValueError(
                f"No color supplied for lambda "
                f"{_lambda_label(value)}."
            )

        return numeric_lambda_colors[
            candidates[hit[0]]
        ]

    # =========================================================
    # Validate / filter dataframe
    # =========================================================
    required={
        allele_col,
        lambda_col,
        metric
    }

    missing=required-set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    plot_df=df.copy()

    # ---------------------------------------------------------
    # Method filtering
    # ---------------------------------------------------------
    if method_col in plot_df.columns:
        plot_df=plot_df[
            plot_df[method_col]==method
        ].copy()

        if plot_df.empty:
            raise ValueError(
                f"No rows remain for method={method!r}."
            )

    elif method is not None:
        # Only complain if user explicitly relies on method filtering but
        # the column does not exist.
        print(
            f"WARNING: method column {method_col!r} not found; "
            "plotting all rows."
        )

    plot_df[allele_col]=(
        plot_df[allele_col]
        .astype(str)
    )

    plot_df[metric]=pd.to_numeric(
        plot_df[metric],
        errors="coerce"
    )

    plot_df["_lambda_float"]=(
        plot_df[lambda_col]
        .map(_lambda_float)
    )

    plot_df=plot_df.dropna(
        subset=[
            allele_col,
            "_lambda_float",
            metric
        ]
    ).copy()

    if plot_df.empty:
        raise ValueError(
            "No plottable rows remain after filtering."
        )

    # =========================================================
    # Allele ordering / splitting
    # =========================================================
    alleles=sorted(
        plot_df[allele_col].unique(),
        key=_allele_sort_key
    )

    split_n=max(
        1,
        int(split_n)
    )

    split_n=min(
        split_n,
        len(alleles)
    )

    allele_chunks=[
        list(chunk)
        for chunk in np.array_split(
            alleles,
            split_n
        )
    ]

    max_actual_rows=max(
        len(chunk)
        for chunk in allele_chunks
    )

    if allele_slots_per_panel is None:
        reference_slots=max_actual_rows
    else:
        reference_slots=int(
            allele_slots_per_panel
        )

        if reference_slots<max_actual_rows:
            raise ValueError(
                f"allele_slots_per_panel={reference_slots} is smaller "
                f"than largest split ({max_actual_rows} alleles). "
                "Increase allele_slots_per_panel or split_n."
            )

    # =========================================================
    # Lambda ordering
    # =========================================================
    lambda_order=sorted(
        plot_df["_lambda_float"]
        .unique()
    )

    lambda_labels=[
        _lambda_label(v)
        for v in lambda_order
    ]

    # =========================================================
    # Marker edges
    # =========================================================
    if marker_outline:
        plot_edgecolors=marker_edgecolor
        plot_linewidths=marker_edgewidth
    else:
        plot_edgecolors="none"
        plot_linewidths=0

    if legend_marker_outline is None:
        legend_marker_outline=marker_outline

    if legend_marker_edgecolor is None:
        legend_marker_edgecolor=marker_edgecolor

    if legend_marker_edgewidth is None:
        legend_marker_edgewidth=marker_edgewidth

    legend_edgecolor=(
        legend_marker_edgecolor
        if legend_marker_outline
        else "none"
    )

    legend_edgewidth=(
        legend_marker_edgewidth
        if legend_marker_outline
        else 0
    )

    # =========================================================
    # Figure geometry
    # =========================================================
    panel_width,panel_height=panel_figsize

    fig_width=(
        panel_width*split_n
    )

    fig_height=panel_height

    fig=plt.figure(
        figsize=(
            fig_width,
            fig_height
        ),
        dpi=dpi
    )

    usable_width=(
        right_margin-left_margin
    )

    if split_n==1:
        axis_width=usable_width
        gap=0
    else:
        gap=panel_wspace

        axis_width=(
            usable_width -
            gap*(split_n-1)
        )/split_n

    reference_plot_height=(
        top_margin-plot_bottom
    )

    axes=[]

    # =========================================================
    # Draw panels
    # =========================================================
    for panel_i,chunk in enumerate(
        allele_chunks
    ):
        n_rows=len(chunk)

        # Fixed physical height per allele row.
        axis_height=(
            reference_plot_height *
            n_rows/reference_slots
        )

        # Anchor at common top so blank canvas occurs BELOW axes.
        axis_bottom=(
            top_margin-axis_height
        )

        axis_left=(
            left_margin +
            panel_i*(
                axis_width+gap
            )
        )

        ax=fig.add_axes([
            axis_left,
            axis_bottom,
            axis_width,
            axis_height
        ])

        axes.append(ax)

        # -----------------------------------------------------
        # Panel subset
        # -----------------------------------------------------
        sub=plot_df[
            plot_df[allele_col]
            .isin(chunk)
        ].copy()

        y_map={
            allele:i
            for i,allele in enumerate(chunk)
        }

        sub["_y"]=(
            sub[allele_col]
            .map(y_map)
        )

        # -----------------------------------------------------
        # Plot lambdas
        # -----------------------------------------------------
        for lam in lambda_order:
            ss=sub[
                np.isclose(
                    sub["_lambda_float"],
                    lam,
                    rtol=1e-12,
                    atol=0
                )
            ]

            if ss.empty:
                continue

            ax.scatter(
                ss[metric].to_numpy(float),
                ss["_y"].to_numpy(float),
                marker=marker,
                s=marker_size,
                color=_lambda_color(lam),
                alpha=point_alpha,
                edgecolors=plot_edgecolors,
                linewidths=plot_linewidths,
                zorder=3
            )

        # -----------------------------------------------------
        # Allele axis
        # -----------------------------------------------------
        ax.set_yticks(
            np.arange(n_rows)
        )

        ax.set_yticklabels(
            [
                _format_allele(a)
                for a in chunk
            ],
            fontsize=allele_fontsize
        )

        ax.set_ylim(
            n_rows-0.5,
            -0.5
        )

        # -----------------------------------------------------
        # X axis
        # -----------------------------------------------------
        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_xlabel(
            metric if xlabel is None else xlabel,
            fontsize=label_fontsize,
            labelpad=7
        )

        if panel_i==0:
            ax.set_ylabel(
                ylabel,
                fontsize=label_fontsize,
                labelpad=8
            )

        # -----------------------------------------------------
        # Grid
        # -----------------------------------------------------
        if grid:
            ax.grid(
                True,
                axis=grid_axis,
                alpha=grid_alpha,
                linewidth=grid_linewidth,
                linestyle=grid_linestyle,
                zorder=0
            )

            ax.set_axisbelow(True)

        else:
            ax.grid(False)

        # -----------------------------------------------------
        # Spines / ticks
        # -----------------------------------------------------
        ax.tick_params(
            axis="both",
            labelsize=tick_fontsize,
            length=tick_length,
            width=tick_width,
            pad=tick_pad
        )

        for spine in ax.spines.values():
            spine.set_linewidth(
                spine_width
            )

    # =========================================================
    # Title
    # =========================================================
    if title is None:
        title=metric

    fig.suptitle(
        title,
        fontsize=title_fontsize,
        y=0.975
    )

    # =========================================================
    # Legend
    # =========================================================
    legend_handles=[
        Line2D(
            [0],[0],
            marker=marker,
            linestyle="",
            markersize=legend_marker_size,
            markerfacecolor=_lambda_color(lam),
            markeredgecolor=legend_edgecolor,
            markeredgewidth=legend_edgewidth,
            alpha=legend_marker_alpha,
            label=_lambda_label(lam)
        )
        for lam in lambda_order
    ]

    legend_nrows=max(
        1,
        int(legend_nrows)
    )

    legend_ncols=int(
        math.ceil(
            len(legend_handles) /
            legend_nrows
        )
    )

    # Attach legend to bottom of the ACTUAL axes area.
    lowest_axis_bottom=min(
        ax.get_position().y0
        for ax in axes
    )

    legend_y=max(
        0.01,
        lowest_axis_bottom -
        legend_y_pad
    )

    legend=fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            legend_y
        ),
        ncol=legend_ncols,
        frameon=legend_frame,
        framealpha=legend_frame_alpha,
        title=legend_title,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        columnspacing=legend_columnspacing,
        handletextpad=legend_handletextpad,
        handlelength=legend_handlelength,
        labelspacing=legend_labelspacing,
        borderaxespad=0
    )

    # =========================================================
    # Save / show
    # =========================================================
    if save_path is not None:
        fig.savefig(
            f"{save_path}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=dpi
        )

        fig.savefig(
            f"{save_path}.png",
            format="png",
            bbox_inches="tight",
            dpi=dpi
        )

    if show:
        plt.show()

    return {
        "fig":fig,
        "axes":axes,
        "legend":legend,
        "plot_df":plot_df,
        "alleles":alleles,
        "allele_chunks":allele_chunks,
        "lambda_order":lambda_order,
        "lambda_labels":lambda_labels,
        "method":method,
    }
def mixed_df_per_allele_ap_balanced_fast(df1, df2, score_cols, label_col1="label", label_col2="label", allele_col="allele", neg_ratio=99, n_iterations=10, seed=42, chunk_size=256, num_threads=None):
    """
    Per-allele AP where, each iteration, df1 and df2 are balanced against each other
    per class: positives are sampled to min(#pos_df1, #pos_df2) from each source, and
    negatives to min(#neg_df1, #neg_df2) from each source. Classes already balanced for
    an allele are used whole (no sampling). APs are averaged across iterations.

    neg_ratio is accepted for input compatibility but is NOT used by this balancing
    method (negatives are balanced to the cross-source minimum, not set by a ratio).
    """
    score_cols=list(score_cols)
    if not score_cols: raise ValueError("score_cols cannot be empty.")

    s1=df1[score_cols].to_numpy(copy=False)
    s2=df2[score_cols].to_numpy(copy=False)
    score_matrix=np.concatenate([s1,s2],axis=0)

    y1=df1[label_col1].to_numpy()
    y2=df2[label_col2].to_numpy()
    labels=np.concatenate([y1,y2])

    a1=df1[allele_col].to_numpy()
    a2=df2[allele_col].to_numpy()
    n1=len(df1)

    all_alleles=set(a1).union(set(a2))
    alleles_sorted=sorted(all_alleles)
    allele_to_row={a:i for i,a in enumerate(alleles_sorted)}

    g1={}; g2={}
    for allele in all_alleles:
        mask1=a1==allele
        mask2=a2==allele
        g1[allele]={
            "pos":np.flatnonzero(mask1&(y1==1)).astype(np.int64),
            "neg":np.flatnonzero(mask1&(y1==0)).astype(np.int64),
        }
        g2[allele]={
            "pos":(np.flatnonzero(mask2&(y2==1))+n1).astype(np.int64),
            "neg":(np.flatnonzero(mask2&(y2==0))+n1).astype(np.int64),
        }

    for allele in alleles_sorted:
        n_pos_each=min(len(g1[allele]["pos"]),len(g2[allele]["pos"]))
        n_neg_each=min(len(g1[allele]["neg"]),len(g2[allele]["neg"]))
        if n_pos_each==0 or n_neg_each==0:
            print(f"[WARNING] Allele {allele} cannot be balanced (pos_each={n_pos_each}, neg_each={n_neg_each}, df1_pos={len(g1[allele]['pos'])}, df2_pos={len(g2[allele]['pos'])}, df1_neg={len(g1[allele]['neg'])}, df2_neg={len(g2[allele]['neg'])}); skipping")

    ap_sum=np.zeros((len(alleles_sorted),len(score_cols)),dtype=np.float64)
    ap_count=np.zeros((len(alleles_sorted),len(score_cols)),dtype=np.int32)
    rng=np.random.default_rng(seed)

    old_threads=get_num_threads()
    if num_threads is not None: set_num_threads(int(num_threads))

    try:
        _average_precision_columns_numba(
            np.array([0,1],dtype=np.int8),
            np.array([[0.0],[1.0]],dtype=np.float64)
        )

        for _ in tqdm(range(n_iterations),desc="Iterations"):
            for allele in alleles_sorted:
                pos1_idx=g1[allele]["pos"]; neg1_idx=g1[allele]["neg"]
                pos2_idx=g2[allele]["pos"]; neg2_idx=g2[allele]["neg"]

                n_pos_each=min(len(pos1_idx),len(pos2_idx))
                n_neg_each=min(len(neg1_idx),len(neg2_idx))
                if n_pos_each==0 or n_neg_each==0: continue

                s_pos1=pos1_idx if len(pos1_idx)==n_pos_each else rng.choice(pos1_idx,n_pos_each,replace=False)
                s_pos2=pos2_idx if len(pos2_idx)==n_pos_each else rng.choice(pos2_idx,n_pos_each,replace=False)
                s_neg1=neg1_idx if len(neg1_idx)==n_neg_each else rng.choice(neg1_idx,n_neg_each,replace=False)
                s_neg2=neg2_idx if len(neg2_idx)==n_neg_each else rng.choice(neg2_idx,n_neg_each,replace=False)

                sampled_idx=np.concatenate([s_pos1,s_pos2,s_neg1,s_neg2]).astype(np.int64,copy=False)
                r=allele_to_row[allele]
                _accumulate_ap_fast(
                    score_matrix,labels,sampled_idx,
                    ap_sum[r],ap_count[r],chunk_size
                )
    finally:
        if num_threads is not None: set_num_threads(old_threads)

    result=np.full_like(ap_sum,np.nan,dtype=np.float64)
    valid=ap_count>0
    result[valid]=ap_sum[valid]/ap_count[valid]

    matrix=pd.DataFrame(result,index=alleles_sorted,columns=score_cols)
    matrix.index.name=allele_col
    return matrix

import numpy as np
import pandas as pd

def compute_ft_crossovers(
    mean_df: pd.DataFrame,
    ci_df: pd.DataFrame = None,
    *,
    ft_col: str = "holec_ft",
    pepcl_cols=None,
    x_col: str = "ms_percentage",
    require_single_crossing: bool = True,
) -> pd.DataFrame:
    """
    Compute the interpolated MS% where each PepCL curve crosses the FT curve.

    Parameters
    ----------
    mean_df : pd.DataFrame
        Wide mean mixture dataframe from the mixture analysis.
        Can have ms_percentage as index or column.

    ci_df : pd.DataFrame or None
        Wide CI dataframe with:
            <curve>_ci_low
            <curve>_ci_high

        If provided, an approximate crossover interval is obtained by
        comparing the relevant CI bounds between PepCL and FT.

    ft_col : str
        FT curve column.

    pepcl_cols : list[str] or None
        PepCL curves to compare against FT.
        If None, all mean_df columns except ft_col and x_col are used.

    x_col : str
        Mixture x-axis variable.

    require_single_crossing : bool
        If True, error if a PepCL-vs-FT difference curve crosses zero
        more than once.

    Returns
    -------
    pd.DataFrame with:
        curve
        crossover_ms_percentage
        ft_delta_map_at_crossover
        pepcl_delta_map_at_crossover
        crossover_delta_map
        crossover_ci_low
        crossover_ci_high
        lower_bracket_ms_percentage
        upper_bracket_ms_percentage
    """

    def _prepare(df):
        d=df.copy()

        if x_col not in d.columns:
            if d.index.name == x_col:
                d=d.reset_index()
            else:
                raise ValueError(
                    f"{x_col!r} not found as a column or named index."
                )

        return d.sort_values(x_col).reset_index(drop=True)

    def _find_zero_crossing(x, y):
        x=np.asarray(x, dtype=float)
        y=np.asarray(y, dtype=float)

        valid=np.isfinite(x) & np.isfinite(y)
        x=x[valid]
        y=y[valid]

        if len(x)<2:
            return np.nan, np.nan, np.nan

        exact=np.where(
            np.isclose(y, 0.0, rtol=0, atol=1e-15)
        )[0]

        if len(exact):
            if require_single_crossing and len(exact)>1:
                raise ValueError("Multiple exact crossover points found.")

            i=exact[0]
            return float(x[i]), float(x[i]), float(x[i])

        crossing_idx=np.where(
            y[:-1]*y[1:]<0
        )[0]

        if len(crossing_idx)==0:
            return np.nan, np.nan, np.nan

        if require_single_crossing and len(crossing_idx)>1:
            raise ValueError(
                "Curve crosses FT more than once."
            )

        i=crossing_idx[0]

        x1,x2=x[i],x[i+1]
        y1,y2=y[i],y[i+1]

        x_cross=x1 + (-y1)*(x2-x1)/(y2-y1)

        return float(x_cross), float(x1), float(x2)

    def _interp_y(x, y, x_target):
        if not np.isfinite(x_target):
            return np.nan

        x=np.asarray(x, dtype=float)
        y=np.asarray(y, dtype=float)

        valid=np.isfinite(x) & np.isfinite(y)
        x=x[valid]
        y=y[valid]

        order=np.argsort(x)

        return float(
            np.interp(
                x_target,
                x[order],
                y[order]
            )
        )

    mean_d=_prepare(mean_df)

    if ft_col not in mean_d.columns:
        raise ValueError(
            f"FT column {ft_col!r} not found in mean_df."
        )

    if pepcl_cols is None:
        pepcl_cols=[
            c for c in mean_d.columns
            if c not in {x_col, ft_col}
        ]

    ci_d=_prepare(ci_df) if ci_df is not None else None

    x=mean_d[x_col].to_numpy(float)
    ft_y=mean_d[ft_col].to_numpy(float)

    rows=[]

    for curve in pepcl_cols:
        if curve not in mean_d.columns:
            raise ValueError(
                f"Curve {curve!r} not found in mean_df."
            )

        pepcl_y=mean_d[curve].to_numpy(float)

        # PepCL advantage relative to FT.
        diff=pepcl_y-ft_y

        cross,x1,x2=_find_zero_crossing(
            x,
            diff
        )

        ft_at_cross=_interp_y(
            x,
            ft_y,
            cross
        )

        pepcl_at_cross=_interp_y(
            x,
            pepcl_y,
            cross
        )

        ci_low=np.nan
        ci_high=np.nan

        if ci_d is not None:
            required=[
                f"{curve}_ci_low",
                f"{curve}_ci_high",
                f"{ft_col}_ci_low",
                f"{ft_col}_ci_high",
            ]

            missing=[
                c for c in required
                if c not in ci_d.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing CI columns: {missing}"
                )

            # Conservative difference-band bounds:
            #
            # lower difference:
            #   PepCL lower - FT upper
            #
            # upper difference:
            #   PepCL upper - FT lower
            pepcl_low=ci_d[f"{curve}_ci_low"].to_numpy(float)
            pepcl_high=ci_d[f"{curve}_ci_high"].to_numpy(float)

            ft_low=ci_d[f"{ft_col}_ci_low"].to_numpy(float)
            ft_high=ci_d[f"{ft_col}_ci_high"].to_numpy(float)

            diff_low=pepcl_low-ft_high
            diff_high=pepcl_high-ft_low

            low_cross,_,_=_find_zero_crossing(
                ci_d[x_col],
                diff_low
            )

            high_cross,_,_=_find_zero_crossing(
                ci_d[x_col],
                diff_high
            )

            candidates=[
                v for v in [low_cross,high_cross]
                if np.isfinite(v)
            ]

            if candidates:
                ci_low=min(candidates)
                ci_high=max(candidates)

        rows.append({
            "curve":curve,
            "crossover_ms_percentage":cross,
            "ft_delta_map_at_crossover":ft_at_cross,
            "pepcl_delta_map_at_crossover":pepcl_at_cross,
            "crossover_delta_map":pepcl_at_cross,
            "crossover_ci_low":ci_low,
            "crossover_ci_high":ci_high,
            "lower_bracket_ms_percentage":x1,
            "upper_bracket_ms_percentage":x2,
        })

    return pd.DataFrame(rows)

def build_joint_retraining_loss_trajectory(
    training_stats,
    assay_val_df,
    ms_val_df,
    base_col,
    checkpoint_steps,
    final_step,
    assay_loss_type="logsmoothap",
    continuous_target_col=None,
    label_col="label",
    allele_col="allele",
    assay_batch_size=3072,
    ms_batch_size=3072,
    assay_neg_ratio=1.0,
    ms_neg_ratio=1.0,
    assay_alpha=10.0,
    ms_alpha=10.0,
    validation_sampling_seed=42000,
    pairwise_chunk_size=2048,
    device="cpu",
    strict=True,
    verbose=True,
):
    """
    Build training and reconstructed validation-loss trajectories for one
    joint-retraining run.

    This intentionally preserves the previous validation-loss methodology:
        - assay and MS validation batches sampled independently
        - allele-aware sampling for both
        - default sampled batch size = 3072 for each stream
        - deterministic sampling
        - one sampled batch per stream reused for every checkpoint
        - LogSmoothAP / soft-Spearman implementations unchanged
        - MS validation always uses LogSmoothAP
        - no DR term
        - no AP/Spearman performance computation

    Parameters
    ----------
    training_stats : str/path or dict
        Saved training_stats.pkl returned directly by trainer.train(), or the
        already-loaded training stats dictionary.

    assay_val_df, ms_val_df : pd.DataFrame
        Full validation dataframes containing checkpoint prediction columns.

    base_col : str
        Unsuffixed final-model score column.

    checkpoint_steps : iterable[int]
        Saved/scored intermediate steps.

    final_step : int
        Final optimizer step represented by base_col.

    Returns
    -------
    dict
        {
            "training": training-loss dataframe,
            "validation": reconstructed validation-loss dataframe,
            "validation_samples": validation sampling summary,
            "score_map": {step: score_column},
        }
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    if assay_loss_type not in {"logsmoothap","spearman","mse"}:
        raise ValueError("assay_loss_type must be 'logsmoothap', 'spearman', or 'mse'.")
    if assay_loss_type in {"spearman","mse"} and continuous_target_col is None:
        raise ValueError("continuous_target_col is required for continuous validation loss.")

    device=torch.device(device)

    # ============================================================
    # SCORE COLUMNS
    #
    # Intermediate:
    #   base_col_step_N
    #
    # Final:
    #   base_col
    # ============================================================
    score_map=joint_retraining_score_cols(
        base_col=base_col,
        steps=checkpoint_steps,
        final_step=final_step,
        include_final=True,
        return_map=True,
    )

    # ============================================================
    # LOAD TRAINING STATS
    # ============================================================
    if isinstance(training_stats,(str,os.PathLike)):
        if verbose:
            print(f"Loading training stats: {training_stats}")
        with open(training_stats,"rb") as f:
            metrics=pickle.load(f)
    else:
        metrics=training_stats

    if not isinstance(metrics,dict):
        raise TypeError("training_stats must be a path to training_stats.pkl or a dict.")

    required_training_keys=["ft_loss_raw","ms_classification_loss_raw"]
    missing=[k for k in required_training_keys if k not in metrics]
    if missing:
        raise KeyError(
            f"training_stats is missing required loss keys: {missing}\n"
            f"Available keys: {sorted(metrics.keys())}"
        )

    assay_train=np.asarray(metrics["ft_loss_raw"],dtype=float)
    ms_train=np.asarray(metrics["ms_classification_loss_raw"],dtype=float)

    if len(assay_train)!=len(ms_train):
        raise ValueError(
            f"Training loss lengths differ: assay={len(assay_train)}, MS={len(ms_train)}."
        )

    n_steps=len(assay_train)
    step_values=np.arange(1,n_steps+1)

    # Use the trainer's actual logged total loss when available.
    # Otherwise fall back to assay + MS for standard no-DR retraining.
    if "total_loss" in metrics and len(metrics["total_loss"])==n_steps:
        total_train=np.asarray(metrics["total_loss"],dtype=float)
    else:
        total_train=assay_train+ms_train

    training=pd.concat([
        pd.DataFrame({
            "step":step_values,
            "loss_name":"assay_loss_raw",
            "loss":assay_train,
        }),
        pd.DataFrame({
            "step":step_values,
            "loss_name":"ms_loss_raw",
            "loss":ms_train,
        }),
        pd.DataFrame({
            "step":step_values,
            "loss_name":"total_loss",
            "loss":total_train,
        }),
    ],ignore_index=True)

    if verbose:
        print(f"Loaded {n_steps:,} training steps.")
        print(f"Validation checkpoints requested: {len(score_map)}")

    # ============================================================
    # LOSS FUNCTIONS
    #
    # SAME IMPLEMENTATIONS AS PREVIOUS LOSS-TRAJECTORY FUNCTION.
    # ============================================================
    def _logsmoothap(scores,labels,alpha):
        s=torch.as_tensor(
            np.asarray(scores,dtype=np.float32),
            device=device,
        )
        y=torch.as_tensor(
            np.asarray(labels,dtype=np.float32),
            device=device,
        )

        if len(s)==0 or float(y.sum())<=0:
            return np.nan

        P=y.sum().clamp(min=1.0)
        eps=1e-8
        total=torch.tensor(0.,device=device)

        for start in range(0,len(s),pairwise_chunk_size):
            q=s[start:start+pairwise_chunk_size]
            S=torch.sigmoid(
                alpha*(
                    s.unsqueeze(0)
                    - q.unsqueeze(1)
                )
            )

            soft_prec=(
                S.matmul(y)
                / S.sum(1).clamp(min=eps)
            )

            total+=(
                soft_prec
                * y[start:start+len(q)]
            ).sum()

        return float(
            torch.log(
                1.0/(total/P+eps)
            ).item()
        )

    def _soft_rank(x,tau):
        x=torch.as_tensor(
            np.asarray(x,dtype=np.float32),
            device=device,
        )

        out=[]

        for start in range(0,len(x),pairwise_chunk_size):
            q=x[start:start+pairwise_chunk_size]

            out.append(
                1.0+
                torch.sigmoid(
                    (
                        x.unsqueeze(0)
                        - q.unsqueeze(1)
                    )/tau
                ).sum(-1)
            )

        return torch.cat(out,dim=0)

    def _soft_spearman(pred,target,tau):
        if len(pred)==0:
            return np.nan

        rp=_soft_rank(pred,tau)
        rt=_soft_rank(target,1e-4)

        rp=rp-rp.mean()
        rt=rt-rt.mean()

        return float(
            (
                1.0-
                (rp*rt).sum()
                /
                (
                    torch.norm(rp)
                    * torch.norm(rt)
                    + 1e-8
                )
            ).item()
        )

    def _mse(a,b):
        a=np.asarray(a,dtype=np.float64)
        b=np.asarray(b,dtype=np.float64)

        return (
            float(np.mean((a-b)**2))
            if len(a)
            else np.nan
        )

    def _assay_loss(df,score_col):
        if assay_loss_type=="logsmoothap":
            return _logsmoothap(
                df[score_col].values,
                df[label_col].values,
                assay_alpha,
            )

        if assay_loss_type=="spearman":
            return _soft_spearman(
                df[score_col].values,
                df[continuous_target_col].values,
                assay_alpha,
            )

        return _mse(
            df[score_col].values,
            df[continuous_target_col].values,
        )

    # ============================================================
    # EXACT PREVIOUS ALLELE-AWARE VALIDATION SAMPLING
    # ============================================================
    def _integer_quota(weights,caps,total):
        weights=np.asarray(weights,float)
        caps=np.asarray(caps,int)
        total=int(total)

        q=np.zeros_like(caps,dtype=int)
        valid=caps>0

        if total<=0 or not np.any(valid):
            return q

        max_total=min(
            total,
            int(caps.sum()),
        )

        w=np.zeros_like(weights,float)

        w[valid]=(
            weights[valid]/weights[valid].sum()
            if weights[valid].sum()>0
            else 1/valid.sum()
        )

        remaining=max_total

        while remaining>0:
            capacity=caps-q
            active=capacity>0

            if not np.any(active):
                break

            wa=np.zeros_like(w)

            wa[active]=(
                w[active]/w[active].sum()
                if w[active].sum()>0
                else 1/active.sum()
            )

            raw=remaining*wa

            add=np.minimum(
                np.floor(raw).astype(int),
                capacity,
            )

            if add.sum()>0:
                q+=add
                remaining-=int(add.sum())

            if remaining<=0:
                break

            capacity=caps-q
            active=capacity>0

            if not np.any(active):
                break

            frac=raw-np.floor(raw)
            frac[~active]=-1
            added=False

            for i in np.argsort(-frac):
                if remaining<=0:
                    break

                if capacity[i]<=0:
                    continue

                q[i]+=1
                remaining-=1
                added=True

            if not added:
                break

        return q

    def _allele_quotas(df,batch_size,neg_ratio):
        counts=(
            df.groupby(
                [allele_col,label_col],
                observed=True,
            )
            .size()
            .unstack(fill_value=0)
        )

        pos=(
            counts[1]
            if 1 in counts.columns
            else pd.Series(
                0,
                index=counts.index,
            )
        )

        neg=(
            counts[0]
            if 0 in counts.columns
            else pd.Series(
                0,
                index=counts.index,
            )
        )

        d=pd.DataFrame({
            "allele":counts.index,
            "n_pos":pos.astype(int).values,
            "n_neg":neg.astype(int).values,
        })

        d=d[
            d.n_pos>0
        ].reset_index(drop=True)

        if d.empty:
            return {}

        A=len(d)
        P=d.n_pos.to_numpy(float)

        target=max(
            1,
            int(
                batch_size/
                (1+float(neg_ratio))
            ),
        )

        uniform=np.ones(A)/A
        prevalence=P/P.sum()
        denom=np.square(P).sum()

        A_eff=(
            float(A)
            if denom<=0
            else float(
                (P.sum()**2)/denom
            )
        )

        balance=float(
            np.clip(
                A_eff/A,
                0,
                1,
            )
        )

        weights=(
            balance*uniform
            +
            (1-balance)*prevalence
        )

        weights/=weights.sum()

        q=_integer_quota(
            weights,
            d.n_pos.to_numpy(int),
            target,
        )

        return {
            a:int(n)
            for a,n
            in zip(
                d.allele.tolist(),
                q,
            )
            if int(n)>0
        }

    def _sample_allele_aware(
        df,
        batch_size,
        neg_ratio,
        rng,
    ):
        quotas=_allele_quotas(
            df,
            batch_size,
            neg_ratio,
        )

        idx=[]

        for allele,q in quotas.items():
            g=df[
                df[allele_col]==allele
            ]

            labels=g[
                label_col
            ].astype(int)

            pos=g.index[
                labels==1
            ].to_numpy()

            neg=g.index[
                labels==0
            ].to_numpy()

            if len(pos)==0:
                continue

            p_take=min(
                len(pos),
                q,
            )

            p=rng.choice(
                pos,
                p_take,
                replace=False,
            )

            idx.extend(p)

            n_req=int(
                p_take*neg_ratio
            )

            if (
                n_req>0
                and len(neg)>0
            ):
                idx.extend(
                    rng.choice(
                        neg,
                        min(
                            len(neg),
                            n_req,
                        ),
                        replace=False,
                    )
                )

        idx=np.asarray(
            idx,
            dtype=int,
        )

        rng.shuffle(idx)

        return idx

    def _sample_meta(
        df,
        idx,
        dataset,
    ):
        g=df.loc[idx]

        labels=g[
            label_col
        ].astype(int)

        return {
            "dataset":dataset,
            "n_sampled":len(g),
            "n_pos":int(
                (labels==1).sum()
            ),
            "n_neg":int(
                (labels==0).sum()
            ),
            "n_alleles":int(
                g[allele_col].nunique()
            ),
        }

    # ============================================================
    # SAMPLE VALIDATION BATCHES ONCE
    #
    # Assay: seed 42000
    # MS:    seed 52000
    #
    # Same batches reused for every checkpoint.
    # ============================================================
    if verbose:
        print()
        print("Sampling assay validation batch...")

    assay_idx=_sample_allele_aware(
        assay_val_df,
        assay_batch_size,
        assay_neg_ratio,
        np.random.default_rng(
            validation_sampling_seed
        ),
    )

    if verbose:
        print("Sampling MS validation batch...")

    ms_idx=_sample_allele_aware(
        ms_val_df,
        ms_batch_size,
        ms_neg_ratio,
        np.random.default_rng(
            validation_sampling_seed
            +10000
        ),
    )

    assay_sample=assay_val_df.loc[
        assay_idx
    ]

    ms_sample=ms_val_df.loc[
        ms_idx
    ]

    validation_samples=pd.DataFrame([
        _sample_meta(
            assay_val_df,
            assay_idx,
            "assay",
        ),
        _sample_meta(
            ms_val_df,
            ms_idx,
            "ms",
        ),
    ])

    if verbose:
        print()
        print("Validation sampling summary:")
        print(validation_samples.to_string(index=False))
        print()

    # ============================================================
    # VALIDATION LOSS TRAJECTORIES
    #
    # Only these sampled ~3072-row batches are used.
    # ============================================================
    val_rows=[]

    checkpoint_iter=tqdm(
        score_map.items(),
        total=len(score_map),
        desc="Validation losses",
        leave=True,
    )

    for step,score_col in checkpoint_iter:

        checkpoint_iter.set_postfix(
            step=step
        )

        missing=[]

        if score_col not in assay_val_df.columns:
            missing.append("assay")

        if score_col not in ms_val_df.columns:
            missing.append("ms")

        if missing:
            msg=(
                f"Score column '{score_col}' "
                f"missing from: {', '.join(missing)}"
            )

            if strict:
                raise KeyError(msg)

            if verbose:
                tqdm.write(
                    f"Warning: {msg}"
                )

            continue

        # Assay validation loss.
        assay_loss=_assay_loss(
            assay_sample,
            score_col,
        )

        # MS validation loss.
        ms_loss=_logsmoothap(
            ms_sample[
                score_col
            ].values,
            ms_sample[
                label_col
            ].values,
            ms_alpha,
        )

        # Standard joint retraining:
        # assay + MS, no DR.
        total_loss=(
            assay_loss
            + ms_loss
        )

        val_rows.extend([
            {
                "step":int(step),
                "loss_name":"assay_loss_raw",
                "loss":assay_loss,
                "score_col":score_col,
            },
            {
                "step":int(step),
                "loss_name":"ms_loss_raw",
                "loss":ms_loss,
                "score_col":score_col,
            },
            {
                "step":int(step),
                "loss_name":"total_loss",
                "loss":total_loss,
                "score_col":score_col,
            },
        ])

    validation=pd.DataFrame(
        val_rows
    )

    # No sorting is required here:
    # score_map is already step ordered and the plotting function
    # selects each loss component independently.
    if verbose:
        print()
        print(
            "Validation checkpoints reconstructed:",
            (
                validation["step"].nunique()
                if not validation.empty
                else 0
            ),
        )
        print("Done.")

    return {
        "training":training,
        "validation":validation,
        "validation_samples":validation_samples,
        "score_map":score_map,
    }

def plot_joint_retraining_loss_and_performance_grid(
    plot_dict,
    loss_key="loss_dfs",
    assay_perf_key="assay_perf",
    ms_perf_key="ms_perf",
    base_col_key="base_col",
    final_step_key="final_step",
    checkpoint_steps_key="checkpoint_steps",
    assay_order=None,
    save_path=None,

    # Figure
    figsize=(9,3.5),
    dpi=300,
    panel_wspace=0.22,
    panel_hspace=0.5,

    # Fonts
    default_font="Nimbus Sans",
    title_fontsize=7,
    label_fontsize=7,
    tick_fontsize=7,
    legend_fontsize=7,
    title_bold=False,

    # Labels
    loss_ylabel="Loss",
    performance_ylabel="Performance",
    xlabel="Optimization step",

    # Axes
    remove_top_spine=True,
    remove_right_spine=True,
    axis_linewidth=0.5,
    tick_length=2.0,
    tick_width=0.5,
    xlabel_pad=1.0,
    ylabel_pad=1.0,

    # Styles
    loss_styles=None,
    performance_styles=None,

    # Legend
    legend=True,
    legend_ncol=4,
    legend_loc="lower center",
    legend_bbox=(0.5,-0.06),
    legend_frame=False,
    legend_handlelength=1.5,

    # Saving
    save_pdf=True,
    save_png=True,
):
    """
    Plot joint-retraining training/validation losses and externally computed
    validation performance.

    Layout
    ------
    1-3 assays:
        2 x 3 fixed grid
        row 0 = loss
        row 1 = validation performance

        Unused columns remain blank. This preserves the same geometry as the
        previous landscape plotting function.

    4-6 assays:
        4 x 3 grid
        rows 0/1 = first three assays
        rows 2/3 = next three assays

    Performance dataframes
    ----------------------
    Rows = alleles
    Columns = checkpoint/model score columns
    Values = externally computed AP or Spearman

    The function averages each score column over rows to obtain mAP or mean
    Spearman. It does not recompute AP/Spearman.
    """
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        default_font,"Nimbus Sans L","Arial","Helvetica"
    ]
    matplotlib.rcParams["pdf.fonttype"]=42

    if assay_order is None:
        assay_order=list(plot_dict.keys())

    if len(assay_order)==0:
        raise ValueError("assay_order cannot be empty.")

    if len(assay_order)>6:
        raise ValueError("This layout supports at most 6 assays.")

    # ============================================================
    # DEFAULT STYLES
    # ============================================================
    default_loss_styles={
        "assay_loss_raw":{
            "color":"#456be8",
            "train_linestyle":"-",
            "val_linestyle":"--",
            "linewidth":1.2,
            "label":"Assay loss",
        },
        "ms_loss_raw":{
            "color":"#ffb55c",
            "train_linestyle":"-",
            "val_linestyle":"--",
            "linewidth":1.2,
            "label":"MS loss",
        },
        "total_loss":{
            "color":"#9b72cf",
            "train_linestyle":"-",
            "val_linestyle":"--",
            "linewidth":1.2,
            "label":"Total loss",
        },
    }

    default_performance_styles={
        "assay":{
            "color":"#456be8",
            "linestyle":"-",
            "linewidth":1.2,
            "label":"Assay val.",
        },
        "ms":{
            "color":"#77C4A0",
            "linestyle":"-",
            "linewidth":1.2,
            "label":"MS val.",
        },
    }

    def _merge_styles(defaults,user):
        out={k:dict(v) for k,v in defaults.items()}

        if user is not None:
            for k,v in user.items():
                if k not in out:
                    out[k]={}
                out[k].update(v)

        return out

    loss_styles=_merge_styles(
        default_loss_styles,
        loss_styles,
    )

    performance_styles=_merge_styles(
        default_performance_styles,
        performance_styles,
    )

    # ============================================================
    # LAYOUT
    #
    # <=3 assays: EXACT old-style 2 x 3 landscape geometry.
    # >3 assays: 4 x 3.
    # ============================================================
    if len(assay_order)<=3:
        nrows=2
        ncols=3

        fig,axes=plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
            gridspec_kw={
                "wspace":panel_wspace,
                "hspace":panel_hspace,
            },
        )

        positions=[
            (0,i,1,i)
            for i in range(len(assay_order))
        ]

    else:
        nrows=4
        ncols=3

        fig,axes=plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
            gridspec_kw={
                "wspace":panel_wspace,
                "hspace":panel_hspace,
            },
        )

        positions=[]

        for i in range(len(assay_order)):
            if i<3:
                positions.append(
                    (0,i,1,i)
                )
            else:
                j=i-3
                positions.append(
                    (2,j,3,j)
                )

    # Turn unused panels off, but DO NOT crop or resize anything.
    used=set()

    for lr,lc,pr,pc in positions:
        used.add((lr,lc))
        used.add((pr,pc))

    for r in range(nrows):
        for c in range(ncols):
            if (r,c) not in used:
                axes[r,c].axis("off")

    performance_rows=[]

    # ============================================================
    # PLOT EACH ASSAY
    # ============================================================
    for assay_name,(lr,lc,pr,pc) in zip(
        assay_order,
        positions,
    ):
        cfg=plot_dict[assay_name]

        loss_dfs=cfg[loss_key]
        assay_perf_df=cfg[assay_perf_key]
        ms_perf_df=cfg[ms_perf_key]

        base_col=cfg[base_col_key]
        final_step=int(
            cfg[final_step_key]
        )
        checkpoint_steps=cfg[
            checkpoint_steps_key
        ]

        score_map=joint_retraining_score_cols(
            base_col=base_col,
            steps=checkpoint_steps,
            final_step=final_step,
            include_final=True,
            return_map=True,
        )

        train=loss_dfs["training"]
        val=loss_dfs["validation"]

        ax_loss=axes[lr,lc]
        ax_perf=axes[pr,pc]

        # ========================================================
        # LOSS CURVES
        # ========================================================
        for loss_name in [
            "total_loss",
            "assay_loss_raw",
            "ms_loss_raw",
        ]:
            style=loss_styles[
                loss_name
            ]

            tr=train[
                train["loss_name"]
                ==loss_name
            ]

            va=val[
                val["loss_name"]
                ==loss_name
            ]

            if not tr.empty:
                ax_loss.plot(
                    tr["step"],
                    tr["loss"],
                    color=style.get(
                        "color"
                    ),
                    linestyle=style.get(
                        "train_linestyle",
                        "-",
                    ),
                    linewidth=style.get(
                        "linewidth",
                        1.2,
                    ),
                    alpha=style.get(
                        "alpha",
                        1.0,
                    ),
                )

            if not va.empty:
                ax_loss.plot(
                    va["step"],
                    va["loss"],
                    color=style.get(
                        "val_color",
                        style.get(
                            "color"
                        ),
                    ),
                    linestyle=style.get(
                        "val_linestyle",
                        "--",
                    ),
                    linewidth=style.get(
                        "val_linewidth",
                        style.get(
                            "linewidth",
                            1.2,
                        ),
                    ),
                    alpha=style.get(
                        "val_alpha",
                        style.get(
                            "alpha",
                            1.0,
                        ),
                    ),
                )

        # ========================================================
        # EXTERNALLY COMPUTED PERFORMANCE
        #
        # One row per allele -> average score column = mAP /
        # mean Spearman.
        # ========================================================
        assay_rows=[]
        ms_rows=[]

        for step,col in score_map.items():

            if col in assay_perf_df.columns:
                x=assay_perf_df.loc[:,col]

                if isinstance(x,pd.DataFrame):
                    raise ValueError(
                        f"Duplicate assay performance "
                        f"columns found for '{col}': "
                        f"{x.shape[1]} columns."
                    )

                value=pd.to_numeric(
                    x,
                    errors="coerce",
                ).mean()

                assay_rows.append({
                    "assay":assay_name,
                    "dataset":"assay",
                    "step":step,
                    "score_col":col,
                    "performance":value,
                })

            if col in ms_perf_df.columns:
                x=ms_perf_df.loc[:,col]

                if isinstance(x,pd.DataFrame):
                    raise ValueError(
                        f"Duplicate MS performance "
                        f"columns found for '{col}': "
                        f"{x.shape[1]} columns."
                    )

                value=pd.to_numeric(
                    x,
                    errors="coerce",
                ).mean()

                ms_rows.append({
                    "assay":assay_name,
                    "dataset":"ms",
                    "step":step,
                    "score_col":col,
                    "performance":value,
                })

        assay_summary=pd.DataFrame(
            assay_rows
        )

        ms_summary=pd.DataFrame(
            ms_rows
        )

        if not assay_summary.empty:
            performance_rows.extend(
                assay_summary.to_dict(
                    "records"
                )
            )

            s=performance_styles[
                "assay"
            ]

            ax_perf.plot(
                assay_summary["step"],
                assay_summary["performance"],
                color=s.get(
                    "color"
                ),
                linestyle=s.get(
                    "linestyle",
                    "-",
                ),
                linewidth=s.get(
                    "linewidth",
                    1.2,
                ),
                alpha=s.get(
                    "alpha",
                    1.0,
                ),
            )

        if not ms_summary.empty:
            performance_rows.extend(
                ms_summary.to_dict(
                    "records"
                )
            )

            s=performance_styles[
                "ms"
            ]

            ax_perf.plot(
                ms_summary["step"],
                ms_summary["performance"],
                color=s.get(
                    "color"
                ),
                linestyle=s.get(
                    "linestyle",
                    "-",
                ),
                linewidth=s.get(
                    "linewidth",
                    1.2,
                ),
                alpha=s.get(
                    "alpha",
                    1.0,
                ),
            )

        # ========================================================
        # TITLES
        # ========================================================
        ax_loss.set_title(
            f"{assay_name} loss",
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight=(
                "bold"
                if title_bold
                else "normal"
            ),
        )

        ax_perf.set_title(
            f"{assay_name} validation",
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight=(
                "bold"
                if title_bold
                else "normal"
            ),
        )

        # ========================================================
        # AXES
        # ========================================================
        for ax,ylabel in [
            (
                ax_loss,
                loss_ylabel,
            ),
            (
                ax_perf,
                performance_ylabel,
            ),
        ]:
            ax.set_xlabel(
                xlabel,
                fontsize=label_fontsize,
                fontfamily=default_font,
                labelpad=xlabel_pad,
            )

            ax.set_ylabel(
                ylabel,
                fontsize=label_fontsize,
                fontfamily=default_font,
                labelpad=ylabel_pad,
            )

            ax.spines[
                "left"
            ].set_linewidth(
                axis_linewidth
            )

            ax.spines[
                "bottom"
            ].set_linewidth(
                axis_linewidth
            )

            ax.spines[
                "top"
            ].set_visible(
                not remove_top_spine
            )

            ax.spines[
                "right"
            ].set_visible(
                not remove_right_spine
            )

            if not remove_top_spine:
                ax.spines[
                    "top"
                ].set_linewidth(
                    axis_linewidth
                )

            if not remove_right_spine:
                ax.spines[
                    "right"
                ].set_linewidth(
                    axis_linewidth
                )

            ax.tick_params(
                axis="both",
                which="major",
                length=tick_length,
                width=tick_width,
                direction="out",
                labelsize=tick_fontsize,
            )

            for lbl in (
                ax.get_xticklabels()
                +ax.get_yticklabels()
            ):
                lbl.set_fontfamily(
                    default_font
                )

    # ============================================================
    # SHARED LEGEND
    # ============================================================
    if legend:
        handles=[
            Line2D(
                [0],[0],
                color=loss_styles[
                    "assay_loss_raw"
                ]["color"],
                linewidth=loss_styles[
                    "assay_loss_raw"
                ]["linewidth"],
                label=loss_styles[
                    "assay_loss_raw"
                ]["label"],
            ),

            Line2D(
                [0],[0],
                color=loss_styles[
                    "ms_loss_raw"
                ]["color"],
                linewidth=loss_styles[
                    "ms_loss_raw"
                ]["linewidth"],
                label=loss_styles[
                    "ms_loss_raw"
                ]["label"],
            ),

            Line2D(
                [0],[0],
                color=loss_styles[
                    "total_loss"
                ]["color"],
                linewidth=loss_styles[
                    "total_loss"
                ]["linewidth"],
                label=loss_styles[
                    "total_loss"
                ]["label"],
            ),

            Line2D(
                [0],[0],
                color="black",
                linewidth=1.2,
                linestyle="-",
                label="Train",
            ),

            Line2D(
                [0],[0],
                color="black",
                linewidth=1.2,
                linestyle="--",
                label="Val.",
            ),

            Line2D(
                [0],[0],
                color=performance_styles[
                    "assay"
                ]["color"],
                linewidth=performance_styles[
                    "assay"
                ]["linewidth"],
                label=performance_styles[
                    "assay"
                ]["label"],
            ),

            Line2D(
                [0],[0],
                color=performance_styles[
                    "ms"
                ]["color"],
                linewidth=performance_styles[
                    "ms"
                ]["linewidth"],
                label=performance_styles[
                    "ms"
                ]["label"],
            ),
        ]

        fig.legend(
            handles=handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=legend_ncol,
            frameon=legend_frame,
            fontsize=legend_fontsize,
            handlelength=legend_handlelength,
        )

        # Same basic behavior as old function:
        # reserve only a little room for shared bottom legend.
        fig.subplots_adjust(
            bottom=0.12
        )

    # ============================================================
    # SAVE
    # ============================================================
    if save_path is not None:

        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=dpi,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                bbox_inches="tight",
                dpi=dpi,
            )

    performance_df=(
        pd.DataFrame(
            performance_rows
        )
        if performance_rows
        else pd.DataFrame(
            columns=[
                "assay",
                "dataset",
                "step",
                "score_col",
                "performance",
            ]
        )
    )

    return fig,axes,performance_df

def joint_retraining_score_cols(base_col,steps,final_step,include_final=True,return_map=False):
    """
    Build joint-retraining score columns.

    Intermediate checkpoint:
        {base_col}_step_{step}

    Final checkpoint:
        {base_col}

    The unsuffixed base_col is always assigned to final_step. If final_step is
    also supplied in steps, the suffixed duplicate is omitted.
    """
    final_step=int(final_step)
    steps=sorted({int(s) for s in steps if 0<=int(s)<=final_step})
    step_map={s:f"{base_col}_step_{s}" for s in steps if s!=final_step}
    if include_final: step_map[final_step]=base_col
    step_map=dict(sorted(step_map.items()))
    return step_map if return_map else list(step_map.values())

def find_joint_retraining_target_steps(
    assay_perf_df,
    ms_perf_df,
    base_col,
    checkpoint_steps,
    final_step,
    assay_target,
    ms_target,
):
    """
    Find when joint retraining reaches target assay/MS validation performance.

    Performance dataframes are expected to have:
        rows    = alleles
        columns = score/model columns
        values  = per-allele AP or Spearman

    Each score column is averaged across rows to obtain mAP / mean Spearman.

    If a target is reached:
        report the FIRST checkpoint whose mean performance >= target.

    If a target is not reached:
        report the checkpoint with the MAXIMUM observed performance.

    Also reports the checkpoint closest to the target in absolute difference.

    Returns
    -------
    {
        "assay": {...},
        "ms": {...},
    }
    """
    import numpy as np
    import pandas as pd

    score_map=joint_retraining_score_cols(
        base_col=base_col,
        steps=checkpoint_steps,
        final_step=final_step,
        include_final=True,
        return_map=True,
    )

    def _summarize(df,target,name):
        rows=[]

        for step,col in score_map.items():
            if col not in df.columns:
                continue

            x=df.loc[:,col]

            if isinstance(x,pd.DataFrame):
                raise ValueError(
                    f"Duplicate {name} performance columns found for "
                    f"'{col}': {x.shape[1]} columns."
                )

            value=pd.to_numeric(x,errors="coerce").mean()

            if np.isfinite(value):
                rows.append({
                    "step":int(step),
                    "value":float(value),
                    "score_col":col,
                })

        if not rows:
            return {
                "reached":False,
                "status":"no_data",
                "target":float(target),
                "step":None,
                "value":None,
                "closest_step":None,
                "closest_value":None,
                "closest_diff":None,
                "max_step":None,
                "max_value":None,
            }

        trajectory=pd.DataFrame(rows)

        # First checkpoint that reaches/crosses target.
        crossed=trajectory[
            trajectory["value"]>=target
        ]

        # Closest checkpoint to target.
        closest_idx=(
            trajectory["value"]-target
        ).abs().idxmin()

        closest=trajectory.loc[
            closest_idx
        ]

        # Maximum observed checkpoint.
        max_idx=trajectory["value"].idxmax()
        max_row=trajectory.loc[max_idx]

        if not crossed.empty:
            first_cross=crossed.iloc[0]

            step=int(first_cross["step"])
            value=float(first_cross["value"])
            reached=True
            status="crossed"

        else:
            step=int(max_row["step"])
            value=float(max_row["value"])
            reached=False
            status="not_reached"

        return {
            "reached":reached,
            "status":status,
            "target":float(target),

            # First crossing if reached; otherwise maximum observed point.
            "step":step,
            "value":value,

            # Closest observed point to target.
            "closest_step":int(closest["step"]),
            "closest_value":float(closest["value"]),
            "closest_diff":float(
                closest["value"]-target
            ),

            # Maximum observed performance.
            "max_step":int(max_row["step"]),
            "max_value":float(max_row["value"]),
        }

    return {
        "assay":_summarize(
            assay_perf_df,
            assay_target,
            "assay",
        ),
        "ms":_summarize(
            ms_perf_df,
            ms_target,
            "MS",
        ),
    }

def find_joint_retraining_target_points(
    assay_perf_df,
    ms_perf_df,
    base_col,
    checkpoint_steps,
    final_step,
    assay_target,
    ms_target,
):
    import numpy as np
    import pandas as pd

    score_map=joint_retraining_score_cols(
        base_col=base_col,
        steps=checkpoint_steps,
        final_step=final_step,
        include_final=True,
        return_map=True,
    )

    rows=[]
    for step,col in score_map.items():
        if col not in assay_perf_df.columns or col not in ms_perf_df.columns:
            continue

        assay_x=assay_perf_df.loc[:,col]
        ms_x=ms_perf_df.loc[:,col]

        if isinstance(assay_x,pd.DataFrame):
            raise ValueError(f"Duplicate assay columns found for '{col}': {assay_x.shape[1]}.")
        if isinstance(ms_x,pd.DataFrame):
            raise ValueError(f"Duplicate MS columns found for '{col}': {ms_x.shape[1]}.")

        assay_value=pd.to_numeric(assay_x,errors="coerce").mean()
        ms_value=pd.to_numeric(ms_x,errors="coerce").mean()

        if np.isfinite(assay_value) and np.isfinite(ms_value):
            rows.append({
                "step":int(step),
                "assay_value":float(assay_value),
                "ms_value":float(ms_value),
            })

    trajectory=pd.DataFrame(rows)

    if trajectory.empty:
        return {
            "assay_target_point":None,
            "ms_target_point":None,
            "pareto_target_point":None,
            "assay_max_point":None,
            "ms_max_point":None,
        }

    def _target_point(metric,target,other_metric,other_target):
        crossed=trajectory[trajectory[metric]>=target]

        if not crossed.empty:
            row=crossed.iloc[0]
            reached=True
            status="crossed"
        else:
            row=trajectory.loc[trajectory[metric].idxmax()]
            reached=False
            status="not_reached"

        return {
            "reached":reached,
            "status":status,
            "step":int(row["step"]),
            "target":float(target),
            "value":float(row[metric]),
            "delta_from_target":float(row[metric]-target),
            "other_target":float(other_target),
            "other_value":float(row[other_metric]),
            "other_delta_from_target":float(row[other_metric]-other_target),
        }

    def _max_point(metric,target,other_metric,other_target):
        row=trajectory.loc[trajectory[metric].idxmax()]

        return {
            "step":int(row["step"]),
            "target":float(target),
            "value":float(row[metric]),
            "delta_from_target":float(row[metric]-target),
            "other_target":float(other_target),
            "other_value":float(row[other_metric]),
            "other_delta_from_target":float(row[other_metric]-other_target),
        }

    # Pareto-efficient checkpoints
    pareto_mask=[]
    for _,row in trajectory.iterrows():
        dominated=(
            (trajectory["assay_value"]>=row["assay_value"]) &
            (trajectory["ms_value"]>=row["ms_value"]) &
            (
                (trajectory["assay_value"]>row["assay_value"]) |
                (trajectory["ms_value"]>row["ms_value"])
            )
        ).any()
        pareto_mask.append(not dominated)

    pareto=trajectory.loc[pareto_mask].copy()

    pareto["target_distance"]=np.sqrt(
        (pareto["assay_value"]-assay_target)**2 +
        (pareto["ms_value"]-ms_target)**2
    )

    pareto_row=pareto.loc[pareto["target_distance"].idxmin()]

    pareto_target_point={
        "step":int(pareto_row["step"]),
        "assay_target":float(assay_target),
        "assay_value":float(pareto_row["assay_value"]),
        "assay_delta_from_target":float(pareto_row["assay_value"]-assay_target),
        "ms_target":float(ms_target),
        "ms_value":float(pareto_row["ms_value"]),
        "ms_delta_from_target":float(pareto_row["ms_value"]-ms_target),
        "target_distance":float(pareto_row["target_distance"]),
        "assay_target_reached":bool(pareto_row["assay_value"]>=assay_target),
        "ms_target_reached":bool(pareto_row["ms_value"]>=ms_target),
    }

    return {
        "assay_target_point":_target_point(
            metric="assay_value",
            target=assay_target,
            other_metric="ms_value",
            other_target=ms_target,
        ),
        "ms_target_point":_target_point(
            metric="ms_value",
            target=ms_target,
            other_metric="assay_value",
            other_target=assay_target,
        ),
        "pareto_target_point":pareto_target_point,
        "assay_max_point":_max_point(
            metric="assay_value",
            target=assay_target,
            other_metric="ms_value",
            other_target=ms_target,
        ),
        "ms_max_point":_max_point(
            metric="ms_value",
            target=ms_target,
            other_metric="assay_value",
            other_target=assay_target,
        ),
    }

def find_joint_retraining_descriptive_points(
    assay_perf,
    ms_perf,
    base_col,
    checkpoint_steps,
    final_step,
    joint_method="minmax_sum",
):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    valid_joint_methods={"minmax_sum","pareto_rank"}
    if joint_method not in valid_joint_methods:
        raise ValueError(
            f"joint_method must be one of {sorted(valid_joint_methods)}, "
            f"got '{joint_method}'."
        )

    final_step=int(final_step)
    checkpoint_steps=sorted({
        int(s) for s in checkpoint_steps
        if 0<int(s)<=final_step
    })

    # ------------------------------------------------------------
    # Build step -> score-column map.
    #
    # Final step always uses the unsuffixed full-model column.
    # ------------------------------------------------------------

    score_map={
        step:f"{base_col}_step_{step}"
        for step in checkpoint_steps
        if step!=final_step
    }

    score_map[final_step]=base_col
    score_map=dict(sorted(score_map.items()))

    missing_assay=[
        col for col in score_map.values()
        if col not in assay_perf.columns
    ]

    missing_ms=[
        col for col in score_map.values()
        if col not in ms_perf.columns
    ]

    if missing_assay:
        raise ValueError(
            f"Missing assay performance columns: {missing_assay}"
        )

    if missing_ms:
        raise ValueError(
            f"Missing MS performance columns: {missing_ms}"
        )

    # ------------------------------------------------------------
    # Build trajectory.
    #
    # Performance at each checkpoint = mean across allele rows.
    # ------------------------------------------------------------

    rows=[]

    for step,col in score_map.items():
        rows.append({
            "step":int(step),
            "assay":float(
                pd.to_numeric(
                    assay_perf[col],
                    errors="coerce",
                ).mean()
            ),
            "ms":float(
                pd.to_numeric(
                    ms_perf[col],
                    errors="coerce",
                ).mean()
            ),
        })

    trajectory=(
        pd.DataFrame(rows)
        .sort_values("step")
        .reset_index(drop=True)
    )

    if trajectory[["assay","ms"]].isna().any().any():
        raise ValueError(
            "NaN assay/MS performance encountered in checkpoint trajectory."
        )

    # ============================================================
    # 1. MAXIMUM ASSAY CHECKPOINT
    # ============================================================

    assay_max_value=trajectory["assay"].max()

    assay_max_row=(
        trajectory.loc[
            trajectory["assay"]==assay_max_value
        ]
        .sort_values("step")
        .iloc[0]
    )

    assay_max_point={
        "step":int(assay_max_row["step"]),
        "assay":float(assay_max_row["assay"]),
        "ms":float(assay_max_row["ms"]),
    }

    # ============================================================
    # 2. MAXIMUM MS CHECKPOINT
    # ============================================================

    ms_max_value=trajectory["ms"].max()

    ms_max_row=(
        trajectory.loc[
            trajectory["ms"]==ms_max_value
        ]
        .sort_values("step")
        .iloc[0]
    )

    ms_max_point={
        "step":int(ms_max_row["step"]),
        "assay":float(ms_max_row["assay"]),
        "ms":float(ms_max_row["ms"]),
    }

    # ============================================================
    # 3. JOINT CHECKPOINT
    # ============================================================

    if joint_method=="minmax_sum":

        # --------------------------------------------------------
        # Independently normalize each complete trajectory to [0,1].
        #
        # This is deliberately fit separately:
        #   assay scaler sees assay trajectory only
        #   MS scaler sees MS trajectory only
        # --------------------------------------------------------

        assay_scaler=MinMaxScaler()
        ms_scaler=MinMaxScaler()

        trajectory["assay_normalized"]=(
            assay_scaler
            .fit_transform(
                trajectory[["assay"]]
            )
            .ravel()
        )

        trajectory["ms_normalized"]=(
            ms_scaler
            .fit_transform(
                trajectory[["ms"]]
            )
            .ravel()
        )

        trajectory["joint_score"]=(
            trajectory["assay_normalized"]
            +trajectory["ms_normalized"]
        )

        # Maximum joint score.
        # Exact tie -> earlier checkpoint.
        joint_max=trajectory["joint_score"].max()

        joint_row=(
            trajectory.loc[
                trajectory["joint_score"]==joint_max
            ]
            .sort_values("step")
            .iloc[0]
        )

        joint_point={
            "step":int(joint_row["step"]),
            "assay":float(joint_row["assay"]),
            "ms":float(joint_row["ms"]),
            "assay_normalized":float(
                joint_row["assay_normalized"]
            ),
            "ms_normalized":float(
                joint_row["ms_normalized"]
            ),
            "joint_score":float(
                joint_row["joint_score"]
            ),
        }

        pareto_trajectory=None

    # ============================================================
    # OPTIONAL: PREVIOUS PARETO-RANK METHOD
    # ============================================================

    else:

        values=trajectory[
            ["assay","ms"]
        ].to_numpy(dtype=float)

        pareto_mask=np.ones(
            len(trajectory),
            dtype=bool,
        )

        for i in range(len(trajectory)):

            dominated=(
                (values[:,0]>=values[i,0])
                &(values[:,1]>=values[i,1])
                &(
                    (values[:,0]>values[i,0])
                    |(values[:,1]>values[i,1])
                )
            )

            if dominated.any():
                pareto_mask[i]=False

        pareto=trajectory.loc[
            pareto_mask
        ].copy()

        pareto["assay_rank"]=pareto[
            "assay"
        ].rank(
            ascending=False,
            method="min",
        )

        pareto["ms_rank"]=pareto[
            "ms"
        ].rank(
            ascending=False,
            method="min",
        )

        pareto["worst_rank"]=pareto[
            ["assay_rank","ms_rank"]
        ].max(axis=1)

        pareto["rank_sum"]=(
            pareto["assay_rank"]
            +pareto["ms_rank"]
        )

        joint_row=(
            pareto.sort_values(
                [
                    "worst_rank",
                    "rank_sum",
                    "step",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        joint_point={
            "step":int(joint_row["step"]),
            "assay":float(joint_row["assay"]),
            "ms":float(joint_row["ms"]),
            "assay_rank":int(
                joint_row["assay_rank"]
            ),
            "ms_rank":int(
                joint_row["ms_rank"]
            ),
            "worst_rank":int(
                joint_row["worst_rank"]
            ),
            "rank_sum":int(
                joint_row["rank_sum"]
            ),
        }

        pareto_trajectory=(
            pareto
            .sort_values("step")
            .reset_index(drop=True)
        )

    # ============================================================
    # UNIQUE SELECTED STEPS
    #
    # Directly usable as early_stopping_points.
    # ============================================================

    early_stopping_steps=sorted({
        assay_max_point["step"],
        ms_max_point["step"],
        joint_point["step"],
    })

    # ============================================================
    # NAMED SELECTED STEPS + CORRESPONDING SCORE COLUMNS
    # ============================================================

    selected_steps={
        "assay_max":assay_max_point["step"],
        "ms_max":ms_max_point["step"],
        "joint_max":joint_point["step"],
    }

    selected_columns={
        "assay_max":score_map[selected_steps["assay_max"]],
        "ms_max":score_map[selected_steps["ms_max"]],
        "joint_max":score_map[selected_steps["joint_max"]],
    }

    return {
        "assay_max_point":assay_max_point,
        "ms_max_point":ms_max_point,
        "joint_point":joint_point,
        "joint_method":joint_method,
        "early_stopping_steps":early_stopping_steps,
        "trajectory":trajectory,
        "pareto_trajectory":pareto_trajectory,
        "selected_steps":selected_steps,
        "selected_columns":selected_columns,
    }

from pathlib import Path
import pandas as pd

def load_joint_retraining_dataframes(
    dataset_names,
    joint_retraining_path,
    index_col=0,
    merge_on=("seq","allele","label"),
    verbose=True,
):
    """
    Load CSV dataframes produced for selected joint-retraining datasets.

    Parameters
    ----------
    dataset_names : list
        Dataset directory names to load.

    joint_retraining_path : str, Path, list, or tuple
        Either:
            - a single joint-retraining root directory
            - multiple root directories

        For multiple roots:
            - datasets are matched only by dataset name
            - CSVs are matched only by filename stem/key
              (e.g. ms_val merges only with ms_val)
            - matched dataframes are merged on merge_on
            - duplicate merge-key rows are dropped first, keeping the first
            - duplicate non-key columns from later dataframes are ignored
            - datasets/keys found in only one root are retained unchanged

    index_col : int or None
        Passed to pd.read_csv.

    merge_on : tuple/list
        Columns used to merge matching CSVs across roots.

    verbose : bool
        Print loaded shapes, duplicate removal, and merge information.

    Returns
    -------
    dict
        {
            dataset_name: {
                csv_filename_without_extension: dataframe,
                ...
            },
            ...
        }
    """

    # ------------------------------------------------------------
    # Normalize path input while preserving single-path behavior
    # ------------------------------------------------------------
    if isinstance(joint_retraining_path,(str,Path)):
        roots=[Path(joint_retraining_path)]

    elif isinstance(joint_retraining_path,(list,tuple)):
        roots=[Path(p) for p in joint_retraining_path]

        if not roots:
            raise ValueError(
                "joint_retraining_path list cannot be empty."
            )

    else:
        raise TypeError(
            "joint_retraining_path must be a path "
            "or a list/tuple of paths."
        )

    merge_on=list(merge_on)
    out={}

    # ============================================================
    # DATASETS
    # ============================================================
    for dataset in dataset_names:

        if verbose:
            print(f"\n{dataset}:")

        # --------------------------------------------------------
        # Load this dataset independently from each root
        # --------------------------------------------------------
        per_root=[]

        for root in roots:

            dataset_dir=root/dataset

            if not dataset_dir.exists():
                if verbose:
                    print(f"  not found: {dataset_dir}")
                continue

            csv_paths=sorted(
                dataset_dir.glob("*.csv")
            )

            if not csv_paths:
                if verbose:
                    print(f"  no CSVs: {dataset_dir}")
                continue

            loaded={}

            for path in csv_paths:

                key=path.stem

                loaded[key]=pd.read_csv(
                    path,
                    index_col=index_col,
                )

                if verbose:
                    print(
                        f"  {root.name}/{dataset}/{key}: "
                        f"{loaded[key].shape}"
                    )

            per_root.append(
                (root,loaded)
            )

        # Dataset not found anywhere.
        if not per_root:
            raise FileNotFoundError(
                f"Dataset '{dataset}' was not found with CSV files "
                f"in any supplied path: {roots}"
            )

        # --------------------------------------------------------
        # Single source for this dataset:
        # preserve original behavior exactly
        # --------------------------------------------------------
        if len(per_root)==1:
            out[dataset]=per_root[0][1]
            continue

        # ========================================================
        # MULTIPLE ROOTS
        # ========================================================
        merged_dataset={}

        # Union of CSV keys across roots.
        all_keys=sorted({
            key
            for _,loaded in per_root
            for key in loaded
        })

        for key in all_keys:

            # Only dataframes with the exact same key participate.
            sources=[
                (root,loaded[key])
                for root,loaded in per_root
                if key in loaded
            ]

            # ----------------------------------------------------
            # Key occurs in only one root -> retain unchanged
            # ----------------------------------------------------
            if len(sources)==1:

                merged_dataset[key]=sources[0][1]

                if verbose:
                    print(
                        f"  {key}: only in {sources[0][0]}, "
                        f"keeping unchanged {merged_dataset[key].shape}"
                    )

                continue

            # ----------------------------------------------------
            # First dataframe
            # ----------------------------------------------------
            first_root,first_df=sources[0]
            merged_df=first_df.copy()

            missing_keys=[
                col
                for col in merge_on
                if col not in merged_df.columns
            ]

            if missing_keys:
                raise ValueError(
                    f"Cannot merge dataset='{dataset}', key='{key}' "
                    f"from {first_root}. "
                    f"Missing merge columns: {missing_keys}"
                )

            # Drop duplicate merge keys, keeping first.
            n_before=len(merged_df)

            merged_df=merged_df.drop_duplicates(
                subset=merge_on,
                keep="first",
            ).copy()

            n_dropped=n_before-len(merged_df)

            if verbose and n_dropped>0:
                print(
                    f"  {key}: dropped {n_dropped} duplicate row(s) "
                    f"from {first_root} based on {merge_on}"
                )

            # ----------------------------------------------------
            # Merge matching dataframes from later roots
            # ----------------------------------------------------
            for root,next_df in sources[1:]:

                next_df=next_df.copy()

                missing_keys=[
                    col
                    for col in merge_on
                    if col not in next_df.columns
                ]

                if missing_keys:
                    raise ValueError(
                        f"Cannot merge dataset='{dataset}', key='{key}' "
                        f"from {root}. "
                        f"Missing merge columns: {missing_keys}"
                    )

                # Drop duplicate merge keys, keeping first.
                n_before=len(next_df)

                next_df=next_df.drop_duplicates(
                    subset=merge_on,
                    keep="first",
                ).copy()

                n_dropped=n_before-len(next_df)

                if verbose and n_dropped>0:
                    print(
                        f"  {key}: dropped {n_dropped} duplicate row(s) "
                        f"from {root} based on {merge_on}"
                    )

                # Only add genuinely new non-key columns.
                # Any column already present in merged_df is discarded
                # from the later dataframe.
                new_cols=[
                    col
                    for col in next_df.columns
                    if col not in merged_df.columns
                    and col not in merge_on
                ]

                if not new_cols:

                    if verbose:
                        print(
                            f"  {key}: {root} has no new columns; "
                            "skipping merge."
                        )

                    continue

                right=next_df[
                    merge_on+new_cols
                ].copy()

                merged_df=merged_df.merge(
                    right,
                    on=merge_on,
                    how="outer",
                    validate="one_to_one",
                )

                if verbose:
                    print(
                        f"  {key}: merged {len(new_cols)} new column(s) "
                        f"from {root} -> {merged_df.shape}"
                    )

            merged_dataset[key]=merged_df

        out[dataset]=merged_dataset

    return out

def steps_to_max(max_val, interval=50, start=0):
    out = list(range(start, max_val, interval))
    if not out or out[-1] != max_val:
        out.append(max_val)
    return out


def summarize_retraining_val_test_selected_points(
    assay_ms_val_full_ms_ap_dfs,
    assay_ms_val_matched_ms_ap_dfs,
    assay_ms_test_full_ms_ap_dfs,
    assay_ms_test_matched_ms_ap_dfs,
    full_ms_es_selected_columns_per_assay,
    matched_ms_es_selected_columns_per_assay,
    dataset_order=None,
    dataset_labels=None,
    verbose=True,

    # Optional comparison configurations.
    summary_dfs_val=None,
    summary_dfs_test=None,
    config_names=None,
    config_labels=None,
):
    """
    Summarize validation-selected joint-retraining checkpoints on both
    validation and test sets, with optional existing model configurations
    appended after the retraining points.

    For each dataset and MS-training condition:
        - assay_max
        - ms_max
        - joint_max

    The checkpoint/model column selected using validation data is applied
    unchanged to the corresponding test AP/performance matrices.

    Optional comparison configurations
    ----------------------------------
    If summary_dfs_val, summary_dfs_test, and config_names are supplied,
    selected existing model configurations are appended AFTER the retraining
    points for each dataset.

    config_names are matched exactly against:
        summary_df["configuration"]

    and are displayed in the supplied order. MHCPrime may be included.

    Expected nested retraining structures
    -------------------------------------
    Validation:
        ap_dfs[dataset]["assay_val"]
        ap_dfs[dataset]["ms_val_matched"]

    Test:
        ap_dfs[dataset]["assay_test"]
        ap_dfs[dataset]["ms_test_matched"]

    Selected-column dictionaries:
        selected_cols[dataset] = {
            "assay_max": ...,
            "ms_max": ...,
            "joint_max": ...,
        }

    Expected summary dataframe columns
    ----------------------------------
    configuration
    assay_performance_mean
    ms_performance_mean

    Returns
    -------
    pd.DataFrame
        Retraining rows followed by optional comparison-configuration rows
        within each dataset.
    """

    if dataset_order is None:
        dataset_order=list(full_ms_es_selected_columns_per_assay.keys())

    dataset_labels=dataset_labels or {}
    config_labels=config_labels or {}
    point_order=["assay_max","ms_max","joint_max"]

    conditions=[
        (
            "Full MS",
            assay_ms_val_full_ms_ap_dfs,
            assay_ms_test_full_ms_ap_dfs,
            full_ms_es_selected_columns_per_assay,
        ),
        (
            "Matched MS",
            assay_ms_val_matched_ms_ap_dfs,
            assay_ms_test_matched_ms_ap_dfs,
            matched_ms_es_selected_columns_per_assay,
        ),
    ]

    def _mean_metric(df,col,dataset,set_name):
        if col not in df.columns:
            raise KeyError(
                f"{dataset}/{set_name}: selected column '{col}' not found."
            )

        x=pd.to_numeric(df[col],errors="coerce")

        if x.notna().sum()==0:
            raise ValueError(
                f"{dataset}/{set_name}/{col}: no finite values found."
            )

        return float(x.mean())

    def _get_config_row(df,config,dataset,set_name):
        if "configuration" not in df.columns:
            raise KeyError(
                f"{dataset}/{set_name}: summary dataframe is missing "
                "'configuration' column."
            )

        sub=df.loc[
            df["configuration"]==config
        ]

        if len(sub)==0:
            raise ValueError(
                f"{dataset}/{set_name}: configuration '{config}' not found."
            )

        if len(sub)>1:
            raise ValueError(
                f"{dataset}/{set_name}: configuration '{config}' occurs "
                f"{len(sub)} times; expected exactly one row."
            )

        return sub.iloc[0]

    # ------------------------------------------------------------
    # Validate optional comparison-config inputs.
    # ------------------------------------------------------------
    use_configs=config_names is not None

    if use_configs:
        config_names=list(config_names)

        if summary_dfs_val is None or summary_dfs_test is None:
            raise ValueError(
                "summary_dfs_val and summary_dfs_test must both be supplied "
                "when config_names is provided."
            )

    rows=[]

    for dataset in dataset_order:
        dataset_label=dataset_labels.get(dataset,dataset)

        # ========================================================
        # 1. RETRAINING POINTS
        # ========================================================
        for training_label,val_dict,test_dict,selected_dict in conditions:

            if dataset not in val_dict:
                raise KeyError(
                    f"'{dataset}' missing from {training_label} "
                    "validation dictionary."
                )

            if dataset not in test_dict:
                raise KeyError(
                    f"'{dataset}' missing from {training_label} "
                    "test dictionary."
                )

            if dataset not in selected_dict:
                raise KeyError(
                    f"'{dataset}' missing from {training_label} "
                    "selected-columns dictionary."
                )

            assay_val_df=val_dict[dataset]["assay_val"]
            ms_val_df=val_dict[dataset]["ms_val_matched"]

            assay_test_df=test_dict[dataset]["assay_test"]
            ms_test_df=test_dict[dataset]["ms_test_matched"]

            for point in point_order:
                if point not in selected_dict[dataset]:
                    raise KeyError(
                        f"{dataset}/{training_label}: selected-columns "
                        f"dictionary is missing '{point}'."
                    )

                col=selected_dict[dataset][point]

                rows.append({
                    "dataset":dataset,
                    "dataset_label":dataset_label,

                    # Existing retraining identifiers.
                    "ms_training":training_label,
                    "selected_point":point,
                    "score_col":col,

                    # Added only to make retraining/config rows explicit.
                    "row_type":"retraining",
                    "configuration":None,
                    "display_name":f"{training_label} {point}",

                    "assay_val":_mean_metric(
                        assay_val_df,
                        col,
                        dataset,
                        "assay_val",
                    ),

                    "assay_test":_mean_metric(
                        assay_test_df,
                        col,
                        dataset,
                        "assay_test",
                    ),

                    "ms_val":_mean_metric(
                        ms_val_df,
                        col,
                        dataset,
                        "ms_val_matched",
                    ),

                    "ms_test":_mean_metric(
                        ms_test_df,
                        col,
                        dataset,
                        "ms_test_matched",
                    ),
                })

        # ========================================================
        # 2. EXISTING CONFIGURATIONS
        #
        # Appended after ALL retraining points for this dataset.
        # ========================================================
        if use_configs:

            if dataset not in summary_dfs_val:
                raise KeyError(
                    f"'{dataset}' missing from summary_dfs_val."
                )

            if dataset not in summary_dfs_test:
                raise KeyError(
                    f"'{dataset}' missing from summary_dfs_test."
                )

            val_summary=summary_dfs_val[dataset]
            test_summary=summary_dfs_test[dataset]

            required_cols={
                "configuration",
                "assay_performance_mean",
                "ms_performance_mean",
            }

            missing_val=required_cols-set(val_summary.columns)
            missing_test=required_cols-set(test_summary.columns)

            if missing_val:
                raise ValueError(
                    f"{dataset}/summary_dfs_val missing columns: "
                    f"{sorted(missing_val)}"
                )

            if missing_test:
                raise ValueError(
                    f"{dataset}/summary_dfs_test missing columns: "
                    f"{sorted(missing_test)}"
                )

            for config in config_names:
                val_row=_get_config_row(
                    val_summary,
                    config,
                    dataset,
                    "validation summary",
                )

                test_row=_get_config_row(
                    test_summary,
                    config,
                    dataset,
                    "test summary",
                )

                rows.append({
                    "dataset":dataset,
                    "dataset_label":dataset_label,

                    # Not applicable to fixed comparison configurations.
                    "ms_training":None,
                    "selected_point":None,
                    "score_col":None,

                    "row_type":"configuration",
                    "configuration":config,
                    "display_name":config_labels.get(config,config),

                    "assay_val":float(
                        pd.to_numeric(
                            val_row["assay_performance_mean"],
                            errors="coerce",
                        )
                    ),

                    "assay_test":float(
                        pd.to_numeric(
                            test_row["assay_performance_mean"],
                            errors="coerce",
                        )
                    ),

                    "ms_val":float(
                        pd.to_numeric(
                            val_row["ms_performance_mean"],
                            errors="coerce",
                        )
                    ),

                    "ms_test":float(
                        pd.to_numeric(
                            test_row["ms_performance_mean"],
                            errors="coerce",
                        )
                    ),
                })

    result=pd.DataFrame(rows)

    # Preserve explicit construction order:
    #
    # dataset
    #   Full MS assay_max
    #   Full MS ms_max
    #   Full MS joint_max
    #   Matched MS assay_max
    #   Matched MS ms_max
    #   Matched MS joint_max
    #   config 1
    #   config 2
    #   ...
    #
    # Do NOT re-sort configurations ahead of retraining points.
    result["dataset"]=pd.Categorical(
        result["dataset"],
        categories=dataset_order,
        ordered=True,
    )

    # ============================================================
    # Verbose output
    # ============================================================
    if verbose:
        for dataset in dataset_order:
            label=dataset_labels.get(dataset,dataset)

            print("\n"+"="*100)
            print(label)
            print("="*100)

            sub=result[
                result["dataset"]==dataset
            ].copy()

            retraining_sub=sub[
                sub["row_type"]=="retraining"
            ]

            print("\nValidation-selected retraining points:")

            print(
                retraining_sub[
                    [
                        "ms_training",
                        "selected_point",
                        "assay_val",
                        "assay_test",
                        "ms_val",
                        "ms_test",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x:f"{x:.4f}",
                )
            )

            if use_configs:
                config_sub=sub[
                    sub["row_type"]=="configuration"
                ]

                print("\nComparison configurations:")

                print(
                    config_sub[
                        [
                            "display_name",
                            "assay_val",
                            "assay_test",
                            "ms_val",
                            "ms_test",
                        ]
                    ].to_string(
                        index=False,
                        float_format=lambda x:f"{x:.4f}",
                    )
                )

    return result

def plot_df_table(
    df,
    columns=None,
    column_labels=None,
    col_widths=None,

    # Figure / fonts.
    fig_size=None,
    dpi=300,
    title="",
    title_fontsize=16,
    header_fontsize=11,
    body_fontsize=11,
    font_family="Nimbus Sans",

    # Table geometry.
    table_bbox=(0.0,0.02,1.0,0.92),
    row_height_scale=1.45,
    cell_pad=None,

    # First-column grouping.
    merge_repeated_first_col=False,
    first_col_text_pad=0.03,

    # Bold styling.
    bold_columns=None,
    bold_rows=None,
    bold_cells=None,

    # Horizontal rules.
    top_linewidth=1.5,
    header_linewidth=0.8,
    bottom_linewidth=1.5,
    row_linewidth=0.5,
    row_lines=None,
    row_line_alpha=.35,
    line_color="black",
    independent_lines=True,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot a dataframe using manuscript-style table formatting.

    If merge_repeated_first_col=True, consecutive identical values in the
    first displayed column are shown once, vertically centered across their
    contiguous row block and left-aligned. All other columns are center-aligned.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to display.

    columns : list, optional
        Columns to include and their order. Defaults to all dataframe columns.

    column_labels : dict or list, optional
        Display labels. A dict maps original dataframe columns to labels.

    col_widths : list, optional
        Relative widths of displayed columns.

    merge_repeated_first_col : bool
        Merge consecutive identical values in the first displayed column.

    first_col_text_pad : float
        Fraction of the first-column width used as left padding for merged
        first-column labels.

    bold_columns : list, optional
        Entire body columns to render bold.

    bold_rows : list, optional
        Zero-based displayed dataframe row positions to render bold.

    bold_cells : list, optional
        Individual cells as (row_position, column) tuples.

    row_lines : list, optional
        Zero-based displayed row positions AFTER which a horizontal separator
        is drawn.

    Returns
    -------
    fig, ax, tbl, display_df
    """
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    if not isinstance(df,pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    display=df.copy()

    if columns is None:
        columns=list(display.columns)
    else:
        missing=[c for c in columns if c not in display.columns]
        if missing:
            raise ValueError(f"columns contains missing dataframe columns: {missing}")
        display=display.loc[:,columns].copy()

    columns=list(display.columns)
    ncols=len(columns)

    if ncols==0:
        raise ValueError("No columns available to plot.")

    if column_labels is None:
        final_labels=[str(c) for c in columns]
    elif isinstance(column_labels,dict):
        final_labels=[column_labels.get(c,str(c)) for c in columns]
    else:
        if len(column_labels)!=ncols:
            raise ValueError(f"column_labels must have length {ncols}, got {len(column_labels)}.")
        final_labels=list(column_labels)

    if col_widths is not None and len(col_widths)!=ncols:
        raise ValueError(f"col_widths must have length {ncols}, got {len(col_widths)}.")

    if fig_size is None:
        fig_size=(max(8,2.0*ncols),max(2.5,.42*(len(display)+2)))

    bold_columns=[] if bold_columns is None else list(bold_columns)
    bold_rows=[] if bold_rows is None else list(bold_rows)
    bold_cells=[] if bold_cells is None else list(bold_cells)

    missing=[c for c in bold_columns if c not in columns]
    if missing:
        raise ValueError(f"bold_columns contains missing displayed columns: {missing}")

    missing=[c for _,c in bold_cells if c not in columns]
    if missing:
        raise ValueError(f"bold_cells contains missing displayed columns: {missing}")

    for row in bold_rows:
        if row<0 or row>=len(display):
            raise ValueError(f"bold_rows contains invalid row position {row}.")

    for row,column in bold_cells:
        if row<0 or row>=len(display):
            raise ValueError(f"bold_cells contains invalid row position {row}.")
        if column not in columns:
            raise ValueError(f"bold_cells contains missing displayed column: {column}")

    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,
        "Nimbus Sans",
        "Nimbus Sans L",
        "DejaVu Sans",
    ]

    fp_body=FontProperties(family=font_family,size=body_fontsize)
    fp_body_bold=FontProperties(family=font_family,size=body_fontsize,weight="bold")
    fp_header=FontProperties(family=font_family,size=header_fontsize,weight="bold")
    fp_title=FontProperties(family=font_family,size=title_fontsize)

    fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)
    ax.axis("off")

    if title is not None:
        ax.text(
            0.0,1.02,title,
            transform=ax.transAxes,
            ha="left",va="bottom",
            fontproperties=fp_title
        )

    # If grouping the first column, remove its normal cell text and redraw
    # each contiguous block separately after the table geometry is known.
    table_display=display.copy()
    first_col=columns[0]

    if merge_repeated_first_col:
        table_display.iloc[:,0]=""

    tbl=ax.table(
        cellText=table_display.values.tolist(),
        colLabels=final_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=table_bbox,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(body_fontsize)
    tbl.scale(1.0,row_height_scale)

    nrows=len(display)+1

    # Remove matplotlib cell borders/backgrounds.
    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)
        if cell_pad is not None:
            cell.PAD=cell_pad

    # Headers: bold and centered.
    for c in range(ncols):
        txt=tbl[(0,c)].get_text()
        txt.set_fontproperties(fp_header)
        txt.set_ha("center")

    # Body: first column left-aligned, all other columns centered.
    for r in range(1,nrows):
        tbl[(r,0)].get_text().set_ha("left")
        for c in range(1,ncols):
            tbl[(r,c)].get_text().set_ha("center")

    # Bold complete columns.
    for column in bold_columns:
        c=columns.index(column)
        if merge_repeated_first_col and c==0:
            continue
        for r in range(1,nrows):
            tbl[(r,c)].get_text().set_fontproperties(fp_body_bold)

    # Bold complete rows.
    for row in bold_rows:
        table_row=row+1
        for c in range(ncols):
            if merge_repeated_first_col and c==0:
                continue
            tbl[(table_row,c)].get_text().set_fontproperties(fp_body_bold)

    # Bold individual cells.
    for row,column in bold_cells:
        c=columns.index(column)
        if merge_repeated_first_col and c==0:
            continue
        tbl[(row+1,c)].get_text().set_fontproperties(fp_body_bold)

    fig.canvas.draw()

    # ==============================================================
    # Merged consecutive first-column labels
    # ==============================================================
    if merge_repeated_first_col and len(display)>0:
        values=display.iloc[:,0].tolist()
        groups=[]
        start=0

        for i in range(1,len(values)+1):
            if i==len(values) or values[i]!=values[start]:
                groups.append((start,i-1,values[start]))
                start=i

        for start,end,value in groups:
            r1=start+1
            r2=end+1

            cell_top=tbl[(r1,0)]
            cell_bottom=tbl[(r2,0)]

            x=cell_top.get_x()
            width=cell_top.get_width()
            top=cell_top.get_y()+cell_top.get_height()
            bottom=cell_bottom.get_y()
            y=(top+bottom)/2

            # Determine whether the merged label should be bold.
            is_bold=(
                first_col in bold_columns
                or any(r in bold_rows for r in range(start,end+1))
                or any(
                    row>=start and row<=end and column==first_col
                    for row,column in bold_cells
                )
            )

            ax.text(
                x+first_col_text_pad*width,
                y,
                "" if pd.isna(value) else str(value),
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontproperties=fp_body_bold if is_bold else fp_body,
                zorder=10,
            )

    # ==============================================================
    # Independent table rules
    # ==============================================================
    left=tbl[(0,0)].get_x()
    right=tbl[(0,ncols-1)].get_x()+tbl[(0,ncols-1)].get_width()
    header_top=tbl[(0,0)].get_y()+tbl[(0,0)].get_height()
    header_bottom=tbl[(0,0)].get_y()
    bottom=tbl[(nrows-1,0)].get_y()

    def _add_rule(y,linewidth,alpha=1.0):
        if independent_lines:
            line=Line2D(
                [left,right],[y,y],
                transform=ax.transAxes,
                color=line_color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle="-",
                solid_capstyle="butt",
                clip_on=False,
                zorder=5,
            )
            ax.add_line(line)
            return line

        return ax.hlines(
            y,left,right,
            transform=ax.transAxes,
            linewidth=linewidth,
            color=line_color,
            alpha=alpha,
            zorder=5
        )

    _add_rule(header_top,top_linewidth)
    _add_rule(header_bottom,header_linewidth)

    if row_lines is not None:
        for row in row_lines:
            if row<0 or row>=len(display)-1:
                raise ValueError(
                    f"row_lines contains invalid row position {row}; "
                    "a separator must occur before the final row."
                )
            y=tbl[(row+1,0)].get_y()
            _add_rule(y,row_linewidth,alpha=row_line_alpha)

    _add_rule(bottom,bottom_linewidth)

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf",dpi=dpi,bbox_inches="tight")
        fig.savefig(f"{save_path}.png",dpi=dpi,bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig,ax,tbl,display
    
import re
import numpy as np
import pandas as pd


def plot_joint_retraining_table(
    assay_ms_val_full_ms_ap_dfs,
    assay_ms_val_matched_ms_ap_dfs,
    assay_ms_test_full_ms_ap_dfs,
    assay_ms_test_matched_ms_ap_dfs,
    full_ms_es_selected_columns_per_assay,
    matched_ms_es_selected_columns_per_assay,
    dataset_order=None,
    dataset_labels=None,

    # Step comparison.
    pepcl_steps=300,

    # Value formatting.
    performance_decimals=3,

    # Table formatting.
    col_widths=None,
    title="",
    fig_size=(20,8),
    dpi=300,
    header_fontsize=11,
    body_fontsize=11,
    title_fontsize=16,
    table_bbox=(0.0,0.02,1.0,0.92),
    row_height_scale=1.45,
    first_col_text_pad=0.03,
    top_linewidth=1.5,
    header_linewidth=0.8,
    bottom_linewidth=1.5,
    row_linewidth=0.5,
    row_line_alpha=0.35,
    independent_lines=True,
    save_path=None,
    show_plot=True,
):
    """
    Build and plot a manuscript-style table summarizing validation-selected
    joint retraining checkpoints.

    Rows
    ----
    For each assay:
        Full MS:
            Assay max
            MS max
            Joint max

        Matched MS:
            Assay max
            MS max
            Joint max

    Columns
    -------
    Assay
    MS set
    Selected point
    Steps
    Steps w.r.t. PepCL
    Assay val.
    MS val.
    Assay test
    MS test

    Notes
    -----
    - The validation-selected checkpoint is evaluated unchanged on test data.
    - PepCL is assumed to use 300 steps by default.
    - Actual checkpoint steps are parsed preferentially from `_step_<N>`.
      If absent, `_steps_<N>` is used.
    - Dataset names are merged/centered across their 6-row blocks.
    """

    if dataset_order is None:
        dataset_order=list(
            full_ms_es_selected_columns_per_assay.keys()
        )

    dataset_labels={} if dataset_labels is None else dataset_labels

    point_order=[
        "assay_max",
        "ms_max",
        "joint_max",
    ]

    point_labels={
        "assay_max":"Assay max",
        "ms_max":"MS max",
        "joint_max":"Joint max",
    }

    conditions=[
        (
            "Full",
            assay_ms_val_full_ms_ap_dfs,
            assay_ms_test_full_ms_ap_dfs,
            full_ms_es_selected_columns_per_assay,
        ),
        (
            "Matched",
            assay_ms_val_matched_ms_ap_dfs,
            assay_ms_test_matched_ms_ap_dfs,
            matched_ms_es_selected_columns_per_assay,
        ),
    ]

    def _mean_metric(df,col,dataset,set_name):
        if col not in df.columns:
            raise KeyError(
                f"{dataset}/{set_name}: selected column "
                f"'{col}' not found."
            )

        x=pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if x.notna().sum()==0:
            raise ValueError(
                f"{dataset}/{set_name}/{col}: "
                "no finite values found."
            )

        return float(x.mean())

    def _extract_steps(col):
        """
        Prefer an explicit checkpoint suffix `_step_<N>`.
        Otherwise use configured/full-budget `_steps_<N>`.
        """

        s=str(col)

        checkpoint_matches=re.findall(
            r"_step_(\d+)(?:_|$)",
            s,
        )

        if checkpoint_matches:
            return int(
                checkpoint_matches[-1]
            )

        budget_matches=re.findall(
            r"_steps_(\d+)(?:_|$)",
            s,
        )

        if budget_matches:
            return int(
                budget_matches[-1]
            )

        raise ValueError(
            f"Could not parse step count from selected "
            f"column: {col!r}"
        )

    # ============================================================
    # BUILD RAW TABLE
    # ============================================================

    rows=[]

    for dataset in dataset_order:
        dataset_label=dataset_labels.get(
            dataset,
            dataset,
        )

        for ms_set,val_dict,test_dict,selected_dict in conditions:

            if dataset not in val_dict:
                raise KeyError(
                    f"{dataset!r} missing from {ms_set} "
                    "validation dictionary."
                )

            if dataset not in test_dict:
                raise KeyError(
                    f"{dataset!r} missing from {ms_set} "
                    "test dictionary."
                )

            if dataset not in selected_dict:
                raise KeyError(
                    f"{dataset!r} missing from {ms_set} "
                    "selected-column dictionary."
                )

            assay_val_df=val_dict[dataset]["assay_val"]
            ms_val_df=val_dict[dataset]["ms_val_matched"]

            assay_test_df=test_dict[dataset]["assay_test"]
            ms_test_df=test_dict[dataset]["ms_test_matched"]

            for point in point_order:

                if point not in selected_dict[dataset]:
                    raise KeyError(
                        f"{dataset}/{ms_set}: selected-column "
                        f"dictionary missing {point!r}."
                    )

                col=selected_dict[dataset][point]
                steps=_extract_steps(col)

                rows.append({
                    "Assay":dataset_label,
                    "MS set":ms_set,
                    "Selected point":point_labels[point],
                    "Steps":steps,
                    "Steps w.r.t. PepCL":steps/pepcl_steps,
                    "Assay val.":_mean_metric(
                        assay_val_df,
                        col,
                        dataset,
                        "assay_val",
                    ),
                    "MS val.":_mean_metric(
                        ms_val_df,
                        col,
                        dataset,
                        "ms_val_matched",
                    ),
                    "Assay test":_mean_metric(
                        assay_test_df,
                        col,
                        dataset,
                        "assay_test",
                    ),
                    "MS test":_mean_metric(
                        ms_test_df,
                        col,
                        dataset,
                        "ms_test_matched",
                    ),
                    "_score_col":col,
                })

    table_df=pd.DataFrame(rows)

    # ============================================================
    # FORMAT DISPLAY VALUES
    # ============================================================

    display_df=table_df.copy()

    display_df["Steps"]=display_df["Steps"].map(
        lambda x:f"{int(x):,}"
    )

    display_df["Steps w.r.t. PepCL"]=display_df[
        "Steps w.r.t. PepCL"
    ].map(
        lambda x:f"{x:.1f}x"
    )

    metric_cols=[
        "Assay val.",
        "MS val.",
        "Assay test",
        "MS test",
    ]

    for col in metric_cols:
        display_df[col]=display_df[col].map(
            lambda x:f"{x:.{performance_decimals}f}"
            if pd.notna(x)
            else "-"
        )

    columns=[
        "Assay",
        "MS set",
        "Selected point",
        "Steps",
        "Steps w.r.t. PepCL",
        "Assay val.",
        "MS val.",
        "Assay test",
        "MS test",
    ]

    display_df=display_df[columns].copy()

    # ============================================================
    # DATASET SEPARATORS
    #
    # Each dataset contributes 6 rows:
    #   Full    x 3
    #   Matched x 3
    #
    # Draw separators only between datasets so the merged dataset
    # label is not crossed by a line between Full and Matched.
    # ============================================================

    rows_per_dataset=6

    row_lines=[
        rows_per_dataset*(i+1)-1
        for i in range(len(dataset_order)-1)
    ]

    if col_widths is None:
        col_widths=[
            0.13,  # Assay
            0.07,  # MS set
            0.10,  # Selected point
            0.06,  # Steps
            0.11,  # Steps w.r.t. PepCL
            0.08,  # Assay val.
            0.08,  # MS val.
            0.08,  # Assay test
            0.08,  # MS test
        ]

    # ============================================================
    # PLOT USING EXISTING GENERIC TABLE FUNCTION
    # ============================================================

    fig,ax,tbl,plotted_df=plot_df_table(
        display_df,
        columns=columns,
        col_widths=col_widths,
        merge_repeated_first_col=True,
        first_col_text_pad=first_col_text_pad,
        bold_columns=["Assay"],
        title=title,
        fig_size=fig_size,
        dpi=dpi,
        header_fontsize=header_fontsize,
        body_fontsize=body_fontsize,
        title_fontsize=title_fontsize,
        table_bbox=table_bbox,
        row_height_scale=row_height_scale,
        top_linewidth=top_linewidth,
        header_linewidth=header_linewidth,
        bottom_linewidth=bottom_linewidth,
        row_lines=row_lines,
        row_linewidth=row_linewidth,
        row_line_alpha=row_line_alpha,
        independent_lines=independent_lines,
        save_path=save_path,
        show_plot=show_plot,
    )

    return fig,ax,tbl,plotted_df,table_df

def plot_retraining_selected_points_val_test_grid(
    assay_ms_val_full_ms_ap_dfs,
    assay_ms_val_matched_ms_ap_dfs,
    assay_ms_test_full_ms_ap_dfs,
    assay_ms_test_matched_ms_ap_dfs,
    full_ms_es_selected_columns_per_assay,
    matched_ms_es_selected_columns_per_assay,
    summary_dfs_val,
    summary_dfs_test,
    config_names,
    datasets=None,
    dataset_labels=None,
    point_colors=None,
    axis_limits=None,

    # Points.
    marker_size=60,

    # Retraining step labels.
    show_retraining_step_labels=False,
    step_label_fontsize=9,
    step_label_linewidth=1.0,  # retained for backward compatibility; no lines drawn

    # Axes.
    square_panel=True,
    show_top_spine=False,
    show_right_spine=False,
    xlabel="MS mAP",
    assay_metric_names=None,

    # Legend.
    legend=False,
    separate_legend=False,
    legend_ncol=None,
    legend_yoffset=0.0,
    legend_height_ratio=0.16,
    legend_fig_size=None,

    # Figure / grid.
    fig_size=(12,16),
    dpi=300,
    wspace=0.35,
    hspace=0.30,
    left_margin=0.08,
    right_margin=0.97,
    bottom_margin=0.06,
    top_margin=0.97,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot validation-selected retraining points and selected comparison
    configurations on validation/test assay-vs-MS performance planes.

    Each dataset occupies one row:
        validation | test

    x = MS mAP
    y = assay metric specified by assay_metric_names.

    If legend=False, the main figure is a direct n_datasets x 2 GridSpec.
    A separate legend figure may still be generated with
    separate_legend=True.

    Retraining step labels
    ----------------------
    If show_retraining_step_labels=True, only validation-selected joint
    retraining points are annotated with their selected optimization step.
    Existing configurations such as MHCPrime, FT, or PepCL are not labeled.

    Step labels are collision-aware and are positioned to avoid plotted
    points and other labels. No connector/leader lines are drawn.

    If a selected point is an explicit checkpoint:
        ..._step_8101 -> "8,101"

    If it is the final unsuffixed model:
        ..._steps_15720 -> "15,720"

    If joint_max overlaps assay_max and/or ms_max, only the joint point is
    plotted and therefore receives only one step label.

    assay_metric_names example
    --------------------------
    {
        "yd":"mAP",
        "episcan_r9":"mAP",
        "episcan_r9rt":"mAP",
        "ba_con":"Spearman",
        "escape_con":"Spearman",
    }

    axis_limits example
    -------------------
    {
        "yd":{
            "val":{"xlim":(0.4,0.9),"ylim":(0.1,0.7)},
            "test":{"xlim":(0.4,0.9),"ylim":(0.1,0.7)},
        }
    }

    Returns
    -------
    results, fig, axes
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        "Nimbus Sans","Nimbus Sans L","Arial","Helvetica"
    ]
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)
    assay_metric_names={} if assay_metric_names is None else dict(assay_metric_names)
    axis_limits={} if axis_limits is None else dict(axis_limits)
    config_names=list(config_names)

    if datasets is None:
        datasets=list(full_ms_es_selected_columns_per_assay.keys())
    else:
        datasets=list(datasets)

    if not datasets:
        raise ValueError("datasets cannot be empty.")

    if legend_ncol is not None and int(legend_ncol)<1:
        raise ValueError("legend_ncol must be >= 1 or None.")

    if legend_height_ratio<=0:
        raise ValueError("legend_height_ratio must be > 0.")

    default_colors={
        "assay_max_full_ms":"#B5B5B5",
        "ms_max_full_ms":"#7F7F7F",
        "joint_max_full_ms":"#3F3F3F",
        "assay_max_matched_ms":"#9CB1E8",
        "ms_max_matched_ms":"#6785DB",
        "joint_max_matched_ms":"#3156B8",
        "MHCPrime":"#111111",
        "FT":"#D9544D",
        "Replay only":"#E49A35",
        "DR only":"#6FA866",
        "Full PepCL":"#7146A8",
    }

    if point_colors is not None:
        default_colors.update(point_colors)

    point_colors=default_colors

    # ================================================================
    # Helpers
    # ================================================================
    def _mean_metric(df,col,dataset,set_name):
        if col not in df.columns:
            raise KeyError(
                f"{dataset}/{set_name}: column '{col}' not found."
            )

        x=pd.to_numeric(df[col],errors="coerce")

        if x.notna().sum()==0:
            raise ValueError(
                f"{dataset}/{set_name}/{col}: no finite values found."
            )

        return float(x.mean())

    def _get_config_row(df,config,dataset,set_name):
        if "configuration" not in df.columns:
            raise KeyError(
                f"{dataset}/{set_name}: missing 'configuration' column."
            )

        sub=df.loc[df["configuration"]==config]

        if len(sub)!=1:
            raise ValueError(
                f"{dataset}/{set_name}: expected exactly one row for "
                f"'{config}', found {len(sub)}."
            )

        return sub.iloc[0]

    def _pretty_point(point):
        return {
            "assay_max":"Assay max",
            "ms_max":"MS max",
            "joint_max":"Joint max",
        }[point]

    def _pretty_training(training):
        return {
            "full_ms":"full MS",
            "matched_ms":"matched MS",
        }[training]

    def _selected_step(col):
        """
        Explicit checkpoint takes precedence. Otherwise the configured
        full optimization budget is the selected endpoint.
        """
        s=str(col)

        checkpoint=re.search(
            r"_step_(\d+)$",
            s,
        )

        if checkpoint is not None:
            return int(checkpoint.group(1))

        budget=re.search(
            r"_steps_(\d+)",
            s,
        )

        if budget is not None:
            return int(budget.group(1))

        raise ValueError(
            f"Could not determine optimization step from score column '{col}'."
        )

    def _resolve_retraining_points(selected_cols,training):
        groups={}

        for point in ["assay_max","ms_max","joint_max"]:
            if point not in selected_cols:
                raise KeyError(
                    f"Selected-columns dictionary missing '{point}'."
                )

            col=selected_cols[point]
            groups.setdefault(col,[]).append(point)

        resolved=[]

        for col,roles in groups.items():

            # Joint takes precedence if selections overlap.
            if "joint_max" in roles:
                primary="joint_max"
            elif "assay_max" in roles:
                primary="assay_max"
            else:
                primary="ms_max"

            others=[
                role for role in roles
                if role!=primary
            ]

            label=(
                f"{_pretty_point(primary)} "
                f"({_pretty_training(training)})"
            )

            if len(others)==1:
                label+=(
                    f" (same as "
                    f"{_pretty_point(others[0]).lower()})"
                )

            elif len(others)>1:
                label+=(
                    " (same as "
                    +" and ".join(
                        _pretty_point(x).lower()
                        for x in others
                    )
                    +")"
                )

            resolved.append({
                "selected_point":primary,
                "score_col":col,
                "step":_selected_step(col),
                "point_key":f"{primary}_{training}",
                "legend_label":label,
                "overlapping_roles":tuple(others),
            })

        order={
            "assay_max":0,
            "ms_max":1,
            "joint_max":2,
        }

        return sorted(
            resolved,
            key=lambda x:order[x["selected_point"]],
        )

    def _add_step_labels(ax,plot_df):
        """
        Collision-aware retraining-step labels.

        Placement behavior is intentionally unchanged from the previous
        version. Text positions are selected using the same candidate
        offsets, point/label collision checks, crowding order, connector-path
        collision checks and fallback search.

        The final rendered annotations contain text only; no connector or
        leader lines are drawn.
        """
        import numpy as np
        from matplotlib.transforms import Bbox
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties

        retraining_df=plot_df[
            plot_df["row_type"]=="retraining"
        ].reset_index(drop=True)

        if retraining_df.empty:
            return

        fig.canvas.draw()

        # ------------------------------------------------------------
        # Geometry in DISPLAY coordinates (pixels), which makes spacing
        # independent of x/y data ranges.
        # ------------------------------------------------------------
        all_data=plot_df[
            ["ms_performance","assay_performance"]
        ].to_numpy(float)

        all_xy=ax.transData.transform(all_data)

        retrain_data=retraining_df[
            ["ms_performance","assay_performance"]
        ].to_numpy(float)

        retrain_xy=ax.transData.transform(retrain_data)

        axes_box=ax.get_window_extent()

        # Marker radius in pixels + extra safety margin.
        marker_radius=np.sqrt(marker_size)/2.0
        point_clearance=marker_radius+7.0

        fp=FontProperties(size=step_label_fontsize)

        # ------------------------------------------------------------
        # Candidate offsets in pixels.
        # ------------------------------------------------------------
        candidate_offsets=[
            # Near diagonals.
            (-28,-22),(28,-22),(-28,22),(28,22),
            (-40,-28),(40,-28),(-40,28),(40,28),

            # Horizontal.
            (-55,0),(55,0),
            (-75,0),(75,0),

            # Vertical.
            (0,-42),(0,42),
            (0,-62),(0,62),

            # Moderately distant diagonals.
            (-65,-45),(65,-45),(-65,45),(65,45),
            (-90,-55),(90,-55),(-90,55),(90,55),

            # More strongly exploit lower / left empty space.
            (-105,-35),(-105,-70),
            (-70,-85),(70,-85),
            (105,-35),(105,-70),
        ]

        def _label_size(text):
            path=TextPath(
                (0,0),
                text,
                prop=fp,
            )

            bb=path.get_extents()

            # TextPath dimensions are in points. Convert to pixels.
            scale=fig.dpi/72.0

            width=bb.width*scale+8
            height=bb.height*scale+6

            return width,height

        def _bbox_from_center(cx,cy,w,h):
            return Bbox.from_extents(
                cx-w/2,
                cy-h/2,
                cx+w/2,
                cy+h/2,
            )

        def _bbox_contains_point(box,p,margin=0):
            return (
                p[0]>=box.x0-margin
                and p[0]<=box.x1+margin
                and p[1]>=box.y0-margin
                and p[1]<=box.y1+margin
            )

        def _segment_point_distance(a,b,p):
            """Minimum pixel distance from p to line segment a->b."""
            a=np.asarray(a,float)
            b=np.asarray(b,float)
            p=np.asarray(p,float)

            ab=b-a
            denom=np.dot(ab,ab)

            if denom==0:
                return np.linalg.norm(p-a)

            t=np.dot(p-a,ab)/denom
            t=np.clip(t,0,1)

            closest=a+t*ab

            return np.linalg.norm(p-closest)

        def _connector_hits_other_point(start,end,target_idx):
            """
            Retained intentionally because this constraint participated in
            selecting the established text positions. No connector is actually
            drawn in the final figure.
            """
            for j,p in enumerate(all_xy):
                target_match=np.linalg.norm(
                    p-retrain_xy[target_idx]
                )<1e-6

                if target_match:
                    continue

                if _segment_point_distance(
                    start,
                    end,
                    p,
                )<point_clearance:
                    return True

            return False

        # ------------------------------------------------------------
        # Place the most crowded retraining points FIRST.
        # ------------------------------------------------------------
        crowding=[]

        for i,p in enumerate(retrain_xy):
            d=np.sqrt(
                np.sum(
                    (all_xy-p)**2,
                    axis=1,
                )
            )

            d=d[d>1e-6]

            crowd_score=np.mean(
                np.sort(d)[:min(3,len(d))]
            )

            crowding.append(
                (crowd_score,i)
            )

        placement_order=[
            i for _,i in sorted(crowding)
        ]

        placements={}
        placed_boxes=[]

        # ------------------------------------------------------------
        # Candidate search
        # ------------------------------------------------------------
        for idx in placement_order:
            r=retraining_df.iloc[idx]

            target=retrain_xy[idx]
            label=f"{int(r['step']):,}"

            w,h=_label_size(label)

            best=None
            best_score=np.inf

            for dx,dy in candidate_offsets:
                center=np.array([
                    target[0]+dx,
                    target[1]+dy,
                ])

                box=_bbox_from_center(
                    center[0],
                    center[1],
                    w,
                    h,
                )

                # ----------------------------------------
                # Rule 1: keep text inside the axes.
                # ----------------------------------------
                edge_pad=3

                if (
                    box.x0<axes_box.x0+edge_pad
                    or box.x1>axes_box.x1-edge_pad
                    or box.y0<axes_box.y0+edge_pad
                    or box.y1>axes_box.y1-edge_pad
                ):
                    continue

                # ----------------------------------------
                # Rule 2: text cannot overlap ANY point.
                # ----------------------------------------
                point_collision=False

                for p in all_xy:
                    if _bbox_contains_point(
                        box,
                        p,
                        margin=point_clearance,
                    ):
                        point_collision=True
                        break

                if point_collision:
                    continue

                # ----------------------------------------
                # Rule 3: labels cannot overlap.
                # ----------------------------------------
                if any(
                    box.overlaps(other)
                    for other in placed_boxes
                ):
                    continue

                # ----------------------------------------
                # Rule 4: retain previous connector-path
                # constraint so text placement is identical.
                # ----------------------------------------
                if _connector_hits_other_point(
                    center,
                    target,
                    idx,
                ):
                    continue

                # ----------------------------------------
                # Scoring.
                # ----------------------------------------
                distance=np.hypot(dx,dy)
                score=distance

                if dx<0:
                    score-=2

                if dy<0:
                    score-=2

                if score<best_score:
                    best_score=score
                    best={
                        "center":center,
                        "box":box,
                        "dx":dx,
                        "dy":dy,
                    }

            # ------------------------------------------------------------
            # If no candidate satisfies every hard constraint, expand the
            # search instead of allowing an overlap.
            # ------------------------------------------------------------
            if best is None:
                fallback_offsets=[]

                for radius in [
                    110,
                    130,
                    150,
                ]:
                    for angle in np.linspace(
                        0,
                        2*np.pi,
                        24,
                        endpoint=False,
                    ):
                        fallback_offsets.append(
                            (
                                radius*np.cos(angle),
                                radius*np.sin(angle),
                            )
                        )

                for dx,dy in fallback_offsets:
                    center=np.array([
                        target[0]+dx,
                        target[1]+dy,
                    ])

                    box=_bbox_from_center(
                        center[0],
                        center[1],
                        w,
                        h,
                    )

                    if (
                        box.x0<axes_box.x0+3
                        or box.x1>axes_box.x1-3
                        or box.y0<axes_box.y0+3
                        or box.y1>axes_box.y1-3
                    ):
                        continue

                    if any(
                        _bbox_contains_point(
                            box,
                            p,
                            margin=point_clearance,
                        )
                        for p in all_xy
                    ):
                        continue

                    if any(
                        box.overlaps(other)
                        for other in placed_boxes
                    ):
                        continue

                    # Retained so selected text positions remain unchanged.
                    if _connector_hits_other_point(
                        center,
                        target,
                        idx,
                    ):
                        continue

                    best={
                        "center":center,
                        "box":box,
                        "dx":dx,
                        "dy":dy,
                    }
                    break

            if best is None:
                raise RuntimeError(
                    f"Could not find a collision-free annotation position "
                    f"for retraining point '{r['point_key']}' in this panel."
                )

            placements[idx]=best
            placed_boxes.append(
                best["box"]
            )

        # ------------------------------------------------------------
        # Draw final annotations.
        #
        # IMPORTANT: placement is unchanged. Only the connector/leader
        # line has been removed.
        # ------------------------------------------------------------
        pixel_to_point=72.0/fig.dpi

        for idx in range(len(retraining_df)):
            r=retraining_df.iloc[idx]
            best=placements[idx]

            color=point_colors[
                r["point_key"]
            ]

            dx_points=best["dx"]*pixel_to_point
            dy_points=best["dy"]*pixel_to_point

            ax.annotate(
                f"{int(r['step']):,}",
                xy=(
                    float(r["ms_performance"]),
                    float(r["assay_performance"]),
                ),
                xytext=(
                    dx_points,
                    dy_points,
                ),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=step_label_fontsize,
                color=color,
                annotation_clip=True,
                zorder=7,
            )

    def _legend_specs_from_df(plot_df):
        specs=[]

        for _,r in plot_df.iterrows():
            key=r["point_key"]
            label=r["legend_label"]

            if key not in [x[0] for x in specs]:
                specs.append(
                    (key,label)
                )

        return specs

    def _legend_handles(specs):
        return [
            Line2D(
                [0],[0],
                marker="o",
                linestyle="none",
                markerfacecolor=point_colors[key],
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=np.sqrt(marker_size),
                label=label,
            )
            for key,label in specs
        ]

    # ================================================================
    # Figure layout
    # ================================================================
    n_datasets=len(datasets)

    fig=plt.figure(
        figsize=fig_size,
        dpi=dpi,
    )

    if legend:
        # Shared legend underneath each validation/test pair.
        outer=fig.add_gridspec(
            n_datasets,
            1,
            hspace=hspace,
            left=left_margin,
            right=right_margin,
            bottom=bottom_margin,
            top=top_margin,
        )

    else:
        # Direct n_datasets x 2 grid.
        outer=fig.add_gridspec(
            n_datasets,
            2,
            wspace=wspace,
            hspace=hspace,
            left=left_margin,
            right=right_margin,
            bottom=bottom_margin,
            top=top_margin,
        )

    axes=np.empty(
        (n_datasets,2),
        dtype=object,
    )

    results={}
    separate_legend_data={}

    # ================================================================
    # Datasets
    # ================================================================
    for row_i,dataset in enumerate(datasets):

        required_dicts=[
            assay_ms_val_full_ms_ap_dfs,
            assay_ms_val_matched_ms_ap_dfs,
            assay_ms_test_full_ms_ap_dfs,
            assay_ms_test_matched_ms_ap_dfs,
            full_ms_es_selected_columns_per_assay,
            matched_ms_es_selected_columns_per_assay,
            summary_dfs_val,
            summary_dfs_test,
        ]

        if any(dataset not in d for d in required_dicts):
            raise KeyError(
                f"Dataset '{dataset}' is missing from one or more inputs."
            )

        # ------------------------------------------------------------
        # Axes
        # ------------------------------------------------------------
        if legend:
            row_grid=outer[row_i].subgridspec(
                2,
                1,
                height_ratios=[1,legend_height_ratio],
                hspace=0.02,
            )

            panel_grid=row_grid[0].subgridspec(
                1,
                2,
                wspace=wspace,
            )

            ax_val=fig.add_subplot(
                panel_grid[0,0]
            )

            ax_test=fig.add_subplot(
                panel_grid[0,1]
            )

            legend_ax=fig.add_subplot(
                row_grid[1]
            )

            legend_ax.axis("off")

        else:
            ax_val=fig.add_subplot(
                outer[row_i,0]
            )

            ax_test=fig.add_subplot(
                outer[row_i,1]
            )

            legend_ax=None

        axes[row_i,0]=ax_val
        axes[row_i,1]=ax_test

        if square_panel:
            ax_val.set_box_aspect(1)
            ax_test.set_box_aspect(1)

        # ------------------------------------------------------------
        # Dataset sources
        # ------------------------------------------------------------
        val_full=assay_ms_val_full_ms_ap_dfs[dataset]
        val_matched=assay_ms_val_matched_ms_ap_dfs[dataset]

        test_full=assay_ms_test_full_ms_ap_dfs[dataset]
        test_matched=assay_ms_test_matched_ms_ap_dfs[dataset]

        selected_full=full_ms_es_selected_columns_per_assay[dataset]
        selected_matched=matched_ms_es_selected_columns_per_assay[dataset]

        retraining_specs=[]

        retraining_specs.extend([
            {
                **spec,
                "training":"full_ms",
            }
            for spec in _resolve_retraining_points(
                selected_full,
                "full_ms",
            )
        ])

        retraining_specs.extend([
            {
                **spec,
                "training":"matched_ms",
            }
            for spec in _resolve_retraining_points(
                selected_matched,
                "matched_ms",
            )
        ])

        split_rows={
            "val":[],
            "test":[],
        }

        # ============================================================
        # Retraining points
        # ============================================================
        for spec in retraining_specs:
            training=spec["training"]
            col=spec["score_col"]

            if training=="full_ms":
                val_source=val_full
                test_source=test_full
            else:
                val_source=val_matched
                test_source=test_matched

            common={
                "row_type":"retraining",
                "point_key":spec["point_key"],
                "legend_label":spec["legend_label"],
                "selected_point":spec["selected_point"],
                "training":training,
                "configuration":None,
                "score_col":col,
                "step":spec["step"],
            }

            split_rows["val"].append({
                **common,
                "ms_performance":_mean_metric(
                    val_source["ms_val_matched"],
                    col,
                    dataset,
                    "ms_val_matched",
                ),
                "assay_performance":_mean_metric(
                    val_source["assay_val"],
                    col,
                    dataset,
                    "assay_val",
                ),
            })

            split_rows["test"].append({
                **common,
                "ms_performance":_mean_metric(
                    test_source["ms_test_matched"],
                    col,
                    dataset,
                    "ms_test_matched",
                ),
                "assay_performance":_mean_metric(
                    test_source["assay_test"],
                    col,
                    dataset,
                    "assay_test",
                ),
            })

        # ============================================================
        # Existing configurations
        # ============================================================
        for config in config_names:
            val_row=_get_config_row(
                summary_dfs_val[dataset],
                config,
                dataset,
                "validation summary",
            )

            test_row=_get_config_row(
                summary_dfs_test[dataset],
                config,
                dataset,
                "test summary",
            )

            common={
                "row_type":"configuration",
                "point_key":config,
                "legend_label":config,
                "selected_point":None,
                "training":None,
                "configuration":config,
                "score_col":None,
                "step":np.nan,
            }

            split_rows["val"].append({
                **common,
                "ms_performance":float(
                    val_row["ms_performance_mean"]
                ),
                "assay_performance":float(
                    val_row["assay_performance_mean"]
                ),
            })

            split_rows["test"].append({
                **common,
                "ms_performance":float(
                    test_row["ms_performance_mean"]
                ),
                "assay_performance":float(
                    test_row["assay_performance_mean"]
                ),
            })

        val_plot_df=pd.DataFrame(
            split_rows["val"]
        )

        test_plot_df=pd.DataFrame(
            split_rows["test"]
        )

        # ============================================================
        # Plot validation / test
        # ============================================================
        for split,ax,plot_df in [
            ("val",ax_val,val_plot_df),
            ("test",ax_test,test_plot_df),
        ]:

            for _,r in plot_df.iterrows():
                key=r["point_key"]

                if key not in point_colors:
                    raise KeyError(
                        f"No color supplied for point '{key}'."
                    )

                ax.scatter(
                    r["ms_performance"],
                    r["assay_performance"],
                    s=marker_size,
                    color=point_colors[key],
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )

            dataset_label=dataset_labels.get(
                dataset,
                dataset,
            )

            ax.set_title(
                f"{dataset_label} – "
                f"{'Validation' if split=='val' else 'Test'}"
            )

            ax.set_xlabel(
                xlabel
            )

            metric=assay_metric_names.get(
                dataset,
                "performance",
            )

            ax.set_ylabel(
                f"Assay {metric}"
            )

            ax.spines["top"].set_visible(
                show_top_spine
            )

            ax.spines["right"].set_visible(
                show_right_spine
            )

            panel_limits=(
                axis_limits
                .get(dataset,{})
                .get(split,{})
            )

            if panel_limits.get("xlim") is not None:
                ax.set_xlim(
                    panel_limits["xlim"]
                )

            if panel_limits.get("ylim") is not None:
                ax.set_ylim(
                    panel_limits["ylim"]
                )

            if show_retraining_step_labels:
                _add_step_labels(
                    ax,
                    plot_df,
                )

        # ============================================================
        # Legend specification
        # ============================================================
        legend_specs=_legend_specs_from_df(
            val_plot_df
        )

        separate_legend_data[dataset]=legend_specs

        # Integrated shared legend if requested.
        if legend:
            handles=_legend_handles(
                legend_specs
            )

            ncol=(
                len(handles)
                if legend_ncol is None
                else int(legend_ncol)
            )

            legend_ax.legend(
                handles=handles,
                loc="center",
                bbox_to_anchor=(
                    0.5,
                    0.5+legend_yoffset,
                ),
                frameon=False,
                ncol=ncol,
            )

        results[dataset]={
            "val":val_plot_df,
            "test":test_plot_df,
            "retraining_specs":retraining_specs,
            "axes":{
                "val":ax_val,
                "test":ax_test,
            },
            "legend_ax":legend_ax,
        }

    # ================================================================
    # Separate legend figure
    # ================================================================
    legend_fig=None

    if separate_legend:
        if legend_fig_size is None:
            legend_fig_size=(
                fig_size[0],
                max(1.5,1.35*n_datasets),
            )

        legend_fig,legend_axes=plt.subplots(
            n_datasets,
            1,
            figsize=legend_fig_size,
            dpi=dpi,
            squeeze=False,
        )

        for i,dataset in enumerate(datasets):
            lax=legend_axes[i,0]
            lax.axis("off")

            specs=separate_legend_data[dataset]

            handles=_legend_handles(
                specs
            )

            ncol=(
                len(handles)
                if legend_ncol is None
                else int(legend_ncol)
            )

            lax.set_title(
                dataset_labels.get(
                    dataset,
                    dataset,
                ),
                fontsize=11,
                pad=3,
            )

            lax.legend(
                handles=handles,
                loc="center",
                frameon=False,
                ncol=ncol,
            )

        legend_fig.tight_layout()

    # ================================================================
    # Save
    # ================================================================
    if save_path is not None:
        fig.savefig(
            f"{save_path}.pdf",
            format="pdf",
            dpi=dpi,
            bbox_inches="tight",
        )

        fig.savefig(
            f"{save_path}.png",
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )

        if separate_legend and legend_fig is not None:
            legend_fig.savefig(
                f"{save_path}_legends.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches="tight",
            )

            legend_fig.savefig(
                f"{save_path}_legends.png",
                format="png",
                dpi=dpi,
                bbox_inches="tight",
            )

    if show_plot:
        plt.show()

    return results,fig,axes