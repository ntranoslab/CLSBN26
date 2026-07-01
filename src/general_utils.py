import gc
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from collections import Counter, defaultdict
import random
from random import sample
import numpy as np
import ahocorasick # pip install pyahocorasick
from scipy.stats import wilcoxon
import os
from pathlib import Path
import re
from typing import List, Tuple, Union, Optional, Iterable, Literal, Sequence, Dict
from scipy.stats import spearmanr, pearsonr, wilcoxon
import copy
import math
from scipy import stats
from itertools import combinations
import tempfile
from pathlib import Path
from urllib.parse import quote
import requests
import zipfile


def save_dict_pickle(dictionary, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def clear_all_gpu_memory():

    torch.cuda.empty_cache()
    for obj in list(globals().values()):
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            del obj

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
        except Exception as e:
            print(f"Error while deleting tensor: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("All GPU memory has been cleared.")
    print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def get_trainable_parameters(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]

def replace_label_column(df, base_label_col, new_label_col):
    if base_label_col == new_label_col:
        return df
    df = df.drop(columns=[base_label_col])
    df = df.rename(columns={new_label_col: base_label_col})
    return df

def create_seq_allele_column(df):
    df["seq_allele"]=[f"{s}_{a}" for s,a in zip(df.seq, df.allele)]
    return df

def filter_canonical_sequences(df, column_name, amino_acids='ACDEFGHIKLMNPQRSTVWY'):
    allowed_set = set(amino_acids)
    return df[df[column_name].apply(lambda seq: set(seq).issubset(allowed_set))]

def df_col_set_intersection(df1, df2, col, print_output=True, return_output=False):
    overlap=set(df1[col]).intersection(set(df2[col]))
    if print_output:
        print(len(overlap))
    if return_output:
        return overlap

def create_seq_allele_label_column(df):
    df["seq_allele_label"]=[f"{s}_{a}_{l}" for s,a,l in zip(df.seq, df.allele, df.label)]
    return df

def remove_flanks(df):
    df["n_flank"]="$"*10
    df["c_flank"]="$"*10
    return df

def filter_seq_len(df, min_length=8, max_length=14):
    df["seq_len"]=df.seq.str.replace("$","").apply(len)
    df_f=df.query(f"seq_len >= {min_length} and seq_len <= {max_length}")
    return df_f
    
def standardize_allele_format(df: pd.DataFrame, no_hla_prefix=False) -> pd.DataFrame:
    df = df.copy()
    
    def clean_allele(allele: str) -> Optional[str]:
        if pd.isna(allele) or allele == 'Neg':
            return allele
            
        if '_' in allele:
            parts = allele.split('_')
            cleaned = [clean_allele(part) for part in parts]
            return '_'.join(sorted(filter(None, cleaned)))
            
        if re.match(r'^[A-C]\d{4}$', allele):
            return allele
            
        match = re.match(r'HLA-([A-C])\*(\d{2}):(\d{2})', allele)
        if no_hla_prefix:
            match = re.match(r'([A-C])\*(\d{2}):(\d{2})', allele)
        if match:
            return f"{match.group(1)}{match.group(2)}{match.group(3)}"
            
        return None
    
    df['allele'] = df['allele'].apply(clean_allele)
    return df.dropna(subset=['allele'])


def map_allele_sequences_efficient_alt(df, allele_dict):
    df = df.copy()
    
    df["mhc_a_1"] = None
    df["mhc_b_1"] = None
    df["mhc_c_1"] = None
    df["temp_seq"] = df["allele"].map(allele_dict)
    df["allele_type"] = df["allele"].str[0:1]

    for prefix in ['A', 'B', 'C']:
        column_name = f"mhc_{prefix.lower()}_1"
        mask = df["allele_type"] == prefix
        df.loc[mask, column_name] = df.loc[mask, "temp_seq"]
    
    df = df.drop(columns=["temp_seq", "allele_type"])
    
    return df

def pad_peptides(df: pd.DataFrame, column: str = 'seq', max_length: int = 14, pad_char: str = '$', pad_side: str = 'right') -> pd.DataFrame:
    if pad_side == 'right':
        df[column] = df[column].apply(lambda x: x.ljust(max_length, pad_char))
    elif pad_side == 'left':
        df[column] = df[column].apply(lambda x: x.rjust(max_length, pad_char))
    return df

# NOTE: check here if flanks contain nans
def full_process_data(df, mhc_seq_dict, remove_flank=False, add_pseudo_label=False):
    df=pad_peptides(df)

    if "n_flank" not in df.columns.tolist():
        df["n_flank"]="$"*10

    if "c_flank" not in df.columns.tolist():
        df["c_flank"]="$"*10

    if remove_flank:
        df=remove_flanks(df)
    
    df=pad_peptides(df, column='n_flank', max_length=10)
    df=pad_peptides(df, column='c_flank', max_length=10)
    df=map_allele_sequences_efficient_alt(df, mhc_seq_dict)
    df["mhc_a_2"]=None
    df["mhc_b_2"]=None
    df["mhc_c_2"]=None
    df["sa_ma"]="SA"

    if add_pseudo_label:
        df["label"] = 0
    return df

def download_concat_ligand_mhc_atlas(
    hla_df: pd.DataFrame,
    dataset_type: str = "cancer",
    name_col: str = "name",
    base_url: str = "http://modinfor.com/Ligand.MHC-Atlas/Download",
    inner_file: str = None,
    sep: str = None,
    encoding: str = None,
    request_timeout: int = 60,
    session: requests.Session = None,
    add_source_cols: bool = True,
    ignore_errors: bool = False,
    return_failures: bool = False,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Download many Ligand.MHC Atlas allele zip files, read the tabular contents, and concatenate them into one DataFrame.
    """

    if name_col not in hla_df.columns:
        raise ValueError(f"Column '{name_col}' not found in hla_df.")

    dataset_type = dataset_type.lower().strip()
    if dataset_type not in {"cancer", "benchmark"}:
        raise ValueError("dataset_type must be either 'cancer' or 'benchmark'.")

    allele_names = (
        hla_df[name_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(allele_names) == 0:
        empty = pd.DataFrame()
        if return_failures:
            return empty, pd.DataFrame(columns=["allele_name", "zip_name", "url", "error"])
        return empty

    norm_alleles = []
    seen = set()
    for a in allele_names:
        if not a.startswith("HLA-"):
            a = f"HLA-{a}"
        if a not in seen:
            seen.add(a)
            norm_alleles.append(a)

    own_session = session is None
    sess = session or requests.Session()

    frames = []
    failures = []

    def _pick_member(members: list[str], extract_dir: Path) -> Path:
        if inner_file is not None:
            p = extract_dir / inner_file
            if not p.exists():
                raise FileNotFoundError(
                    f"Requested inner_file '{inner_file}' not found. "
                    f"Available: {members}"
                )
            return p

        candidate_suffixes = {".txt", ".tsv", ".csv"}
        candidates = [
            extract_dir / m
            for m in members
            if (not m.endswith("/")) and Path(m).suffix.lower() in candidate_suffixes
        ]

        if not candidates:
            raise ValueError(
                f"No tabular file (.txt/.tsv/.csv) found inside archive. "
                f"Available: {members}"
            )

        candidates = sorted(candidates, key=lambda p: (len(p.parts), len(p.name)))
        return candidates[0]

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for allele_name in tqdm(norm_alleles, desc="Downloading alleles", leave=True):
                zip_stem = allele_name if dataset_type == "cancer" else f"Benchmark-{allele_name}"
                zip_name = f"{zip_stem}.zip"
                url = f"{base_url}/{quote(zip_name)}"

                zip_path = tmpdir / zip_name
                extract_dir = tmpdir / zip_stem

                try:
                    with sess.get(url, stream=True, timeout=request_timeout) as resp:
                        resp.raise_for_status()
                        with open(zip_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)

                    extract_dir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        members = [m for m in zf.namelist() if not m.endswith("/")]
                        if not members:
                            raise ValueError(f"No files inside archive: {zip_name}")
                        zf.extractall(extract_dir)

                    target_path = _pick_member(members, extract_dir)

                    this_sep = sep
                    if this_sep is None:
                        this_sep = "," if target_path.suffix.lower() == ".csv" else "\t"

                    try:
                        df_part = pd.read_csv(
                            target_path,
                            sep=this_sep,
                            encoding=encoding,
                            low_memory=low_memory,
                        )
                    except Exception:
                        df_part = pd.read_csv(
                            target_path,
                            sep=None,
                            engine="python",
                            encoding=encoding,
                        )

                    if add_source_cols:
                        df_part["atlas_allele_name"] = allele_name
                        df_part["atlas_dataset_type"] = dataset_type
                        df_part["atlas_zip_name"] = zip_name

                    frames.append(df_part)

                except Exception as e:
                    failures.append(
                        {
                            "allele_name": allele_name,
                            "zip_name": zip_name,
                            "url": url,
                            "error": str(e),
                        }
                    )
                    if not ignore_errors:
                        raise

                finally:
                    try:
                        if zip_path.exists():
                            zip_path.unlink()
                    except Exception:
                        pass

                    try:
                        if extract_dir.exists():
                            for p in sorted(extract_dir.rglob("*"), reverse=True):
                                if p.is_file():
                                    p.unlink(missing_ok=True)
                                elif p.is_dir():
                                    try:
                                        p.rmdir()
                                    except OSError:
                                        pass
                            extract_dir.rmdir()
                    except Exception:
                        pass

    finally:
        if own_session:
            sess.close()

    if len(frames) == 0:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(frames, axis=0, ignore_index=True, sort=False, copy=False)

    failures_df = pd.DataFrame(failures)

    if return_failures:
        return combined, failures_df
    return combined

def expand_alleles_by_donor(df, peptide_col, donor_col, donor_dict):
    result = []
    for donor, alleles in tqdm(donor_dict.items(), desc="Expanding donors"):
        donor_df = df[df[donor_col] == donor].copy()
        expanded = []
        for allele in alleles:
            allele_df = donor_df.copy()
            allele_df['allele'] = allele
            expanded.append(allele_df)
        if expanded:
            result.append(pd.concat(expanded, ignore_index=True))
    return pd.concat(result, ignore_index=True)

def print_data_stats(df, allele_dist=False, label_col="label"):

    df=replace_label_column(df, "label", label_col)

    total=df.shape[0]
    pos=df.query("label == 1").shape[0]
    neg=df.query("label == 0").shape[0]
    print(f"Total: {total:,}")
    print(f"Pos: {pos:,} | {(pos/total)*100:.2f}%")
    print(f"Neg: {neg:,} | {(neg/total)*100:.2f}%")
    print("N alleles:", df.allele.nunique())
    if allele_dist:
        for a in df.allele.unique():
            df_a=df.query(f"allele == '{a}'")
            total=df_a.shape[0]
            pos=df_a.query("label == 1").shape[0]
            neg=df_a.query("label == 0").shape[0]
            print(a)
            print(f"  - Total: {total:,}")
            print(f"  - Pos: {pos:,} | {(pos/total)*100:.2f}%")
            print(f"  - Neg: {neg:,} | {(neg/total)*100:.2f}%")

def strip_pt_suffix(df):
    df = df.copy()
    df.columns = [c[:-3] if isinstance(c, str) and c.endswith('.pt') else c for c in df.columns]
    return df

def summarize_datasets_wide(df_dict, label_col='label', seq_col='seq', allele_col='allele'):
    records = []

    for name, val in df_dict.items():
        if isinstance(val, tuple):
            df, group_col = val
            groups = [(f"{name}_{group}", grp) for group, grp in df.groupby(group_col)]
        else:
            df = val
            groups = [(name, val)]

        for group_name, group_df in groups:
            row = {'dataset': group_name}

            for label_id, suffix in [(1, 'pos'), (0, 'neg')]:
                subset = group_df[group_df[label_col] == label_id]
                row[f'{suffix}_unique_seq_allele'] = subset[[seq_col, allele_col]].drop_duplicates().shape[0]
                row[f'{suffix}_unique_seq'] = subset[seq_col].nunique()
                row[f'{suffix}_unique_alleles'] = subset[allele_col].nunique()

            row['total_unique_alleles'] = group_df[allele_col].nunique()
            records.append(row)

    return pd.DataFrame(records)

def compute_recovery(df, score_cols, score_cols_external, threshold):
    records = []
    for score_col in score_cols + score_cols_external:
        col = score_col if score_col in score_cols_external else score_col + "_rank_r_chp"
        recovered = df[df[col] >= threshold]
        records.append({"model_name": score_col, "recovered": len(recovered) / len(df)})
    return pd.DataFrame(records)

def compute_mean_median_rank(df, score_cols, score_cols_external):
    records = []
    for score_col in score_cols + score_cols_external:
        col = score_col if score_col in score_cols_external else score_col + "_rank_r_chp"
        records.append({
            "model_name": score_col,
            "mean_rank": df[col].mean(),
            "median_rank": df[col].median()
        })
    return pd.DataFrame(records)

def mean_scores_per_allele(df, allele_col, score_cols):
    return df.groupby(allele_col)[score_cols].mean()

def compute_hamming(df, seq_col1, seq_col2, out_col):
    result = df.copy()
    s1 = df[seq_col1].values
    s2 = df[seq_col2].values
    result[out_col] = np.array([
        sum(a != b for a, b in zip(x, y)) for x, y in zip(s1, s2)
    ])
    return result

def spearman_vs_base(df, cols, base_col, index_name='spearman'):
    records = {}
    for col in cols:
        corr, pval = spearmanr(df[col], df[base_col])
        records[col] = corr
    return pd.DataFrame([records], index=[index_name])

def flatten_gene_model_df(df, name_col='model_name', value_col='value'):
    records = []
    for gene in df.index:
        for model in df.columns:
            records.append({
                name_col: f"{model}_{gene}",
                value_col: df.loc[gene, model]
            })
    return pd.DataFrame(records)

def merge_with_unique_peptide_ids(dfs, peptide_id_col):
    dfs = [df.copy() for df in dfs]
    
    if any((df[peptide_id_col] == 0).any() for df in dfs):
        for df in dfs:
            df[peptide_id_col] += 1
    
    offset = 0
    shifted_dfs = []
    for df in dfs:
        df[peptide_id_col] += offset
        shifted_dfs.append(df)
        offset += df[peptide_id_col].max() - offset
    
    return pd.concat(shifted_dfs, ignore_index=True)

def pairwise_group_overlap(df_dict, group_col, label_col="label", split_by_pos_neg=False):
    if not split_by_pos_neg:
        sets = {name: set(df[group_col]) for name, df in df_dict.items()}
        names = list(sets.keys())
        overlap = pd.DataFrame(index=names, columns=names, dtype=int)
        for n1 in names:
            for n2 in names:
                overlap.loc[n1, n2] = len(sets[n1] & sets[n2])
        return overlap.astype(int)

    sets = {}
    for name, df in df_dict.items():
        pos = set(df[df[label_col] == 1][group_col])
        neg = set(df[df[label_col] == 0][group_col])
        if len(pos) > 0:
            sets[f"{name}+"] = pos
        if len(neg) > 0:
            sets[f"{name}−"] = neg

    keys = list(sets.keys())
    overlap = pd.DataFrame(index=keys, columns=keys, dtype=int)
    for k1 in keys:
        for k2 in keys:
            overlap.loc[k1, k2] = len(sets[k1] & sets[k2])
    return overlap.astype(int)

def get_model_gene_dict(df):
    genes = df.index.tolist()
    model_gene_dict = {
        model: [f"{model}_{gene}" for gene in genes]
        for model in df.columns
    }
    return model_gene_dict, genes

def filter_to_shared_pairs(df_small, df_large, allele_col='allele', seq_len_col='seq_len'):
    valid_pairs = set(zip(df_large[allele_col], df_large[seq_len_col]))
    mask = [pair in valid_pairs for pair in zip(df_small[allele_col], df_small[seq_len_col])]
    return df_small[mask].copy()

def filter_domain0_to_domain1_pairs(df, allele_col='allele', seq_len_col='seq_len', domain_col='domain_id'):
    d1 = df[df[domain_col] == 1]
    d0 = df[df[domain_col] == 0]

    valid_pairs = set(zip(d1[allele_col], d1[seq_len_col]))
    mask = [pair in valid_pairs for pair in zip(d0[allele_col], d0[seq_len_col])]
    d0_kept = d0[mask]

    return pd.concat([d1, d0_kept]).copy()

def split_percentage_with_bounds(df, pct, min_count, max_count, label_col='label', allele_col='allele', seq_col='seq', neg_ratio=1, seed=42):
    rng = np.random.default_rng(seed)

    main_rows = []
    split_rows = []

    for allele, grp in tqdm(df.groupby(allele_col), desc="Splitting alleles"):
        pos = grp[grp[label_col] == 1]
        neg = grp[grp[label_col] == 0]
        n_pos = len(pos)

        if n_pos < min_count:
            main_rows.append(grp)
            continue

        n_take = int(round(n_pos * pct / 100))

        if n_take < min_count:
            if n_pos - min_count >= min_count:
                n_take = min_count
            else:
                main_rows.append(grp)
                continue
        else:
            n_take = min(n_take, max_count)

        if n_pos - n_take < n_take:
            main_rows.append(grp)
            continue

        split_pos_idx = rng.choice(pos.index, n_take, replace=False)
        split_pos = pos.loc[split_pos_idx]
        main_pos = pos.drop(split_pos_idx)

        n_neg_split = min(int(n_take * neg_ratio), len(neg))
        if n_neg_split > 0:
            split_neg_idx = rng.choice(neg.index, n_neg_split, replace=False)
            split_neg = neg.loc[split_neg_idx]
            main_neg = neg.drop(split_neg_idx)
        else:
            split_neg = pd.DataFrame()
            main_neg = neg

        main_rows.append(pd.concat([main_pos, main_neg]))
        split_rows.append(pd.concat([split_pos, split_neg]))

    main_df = pd.concat(main_rows, ignore_index=True) if main_rows else pd.DataFrame()
    split_df = pd.concat(split_rows, ignore_index=True) if split_rows else pd.DataFrame()

    return main_df, split_df

def split_ms_with_synthetic_negatives(
    ms_pos_df,
    other_dfs,
    synthetic_negatives_by_length,
    test_frac=0.20,
    min_test_per_group=5,
    max_test_per_group=500,
    train_neg_per_pos=50,
    test_neg_per_pos=99,
    seq_col="seq",
    allele_col="allele",
    label_col="label",
    seed=0,
    leave=True,
):
    """
    Split MS-positive peptide-MHC data into train/test positives by allele x peptide-length group, then sample
    synthetic negatives per allele x length at separate train/test negative-per-positive ratios while preventing
    seq-allele overlap between train/test negatives, MS positives, and all provided external datasets.
    """
    rng = np.random.default_rng(seed)

    required_cols = {seq_col, allele_col, label_col}
    missing = required_cols - set(ms_pos_df.columns)
    if missing:
        raise ValueError(f"ms_pos_df is missing columns: {missing}")

    target_alleles = set(ms_pos_df[allele_col].unique())

    ms_work = ms_pos_df[[seq_col, allele_col, label_col]].copy()
    lengths = ms_work[seq_col].str.len().to_numpy()
    ms_work["_len"] = lengths

    group_indices = ms_work.groupby([allele_col, "_len"], sort=False).indices

    test_mask = np.zeros(len(ms_work), dtype=bool)
    group_pos_counts = {}

    for key, idx in tqdm(group_indices.items(), desc="Splitting MS positives", leave=leave):
        idx = np.asarray(idx)
        n = idx.size
        n_test = int(np.floor(n * test_frac))
        if n_test < min_test_per_group:
            n_test = 0
        else:
            n_test = min(n_test, max_test_per_group)
            n_test = min(n_test, n - 1)
        if n_test > 0:
            chosen = rng.choice(idx, size=n_test, replace=False)
            test_mask[chosen] = True
        group_pos_counts[key] = (n - n_test, n_test)

    train_pos = ms_work.loc[~test_mask, [seq_col, allele_col, label_col]].copy()
    test_pos = ms_work.loc[test_mask, [seq_col, allele_col, label_col]].copy()

    forbidden_by_allele = defaultdict(set)

    for allele, vals in ms_work.groupby(allele_col, sort=False)[seq_col]:
        forbidden_by_allele[allele].update(vals.to_numpy())

    for df in tqdm(other_dfs.values(), desc="Indexing external seq-allele exclusions", leave=leave):
        if seq_col not in df.columns or allele_col not in df.columns:
            raise ValueError("Each external dataframe must contain seq_col and allele_col.")
        sub = df.loc[df[allele_col].isin(target_alleles), [seq_col, allele_col]]
        for allele, vals in sub.groupby(allele_col, sort=False)[seq_col]:
            forbidden_by_allele[allele].update(vals.to_numpy())

    neg_pools = {
        int(k): np.asarray(v, dtype=object)
        for k, v in synthetic_negatives_by_length.items()
    }

    def sample_unique_from_pool(pool, n_needed, forbidden, allele, length):
        if n_needed == 0:
            return []

        pool_size = len(pool)
        if pool_size == 0:
            raise ValueError(f"No synthetic negatives available for length={length}.")

        selected = []
        selected_set = set()

        if n_needed <= max(10000, int(0.20 * pool_size)):
            batch_size = min(pool_size, max(10000, int(n_needed * 1.5)))
            stale_rounds = 0

            while len(selected) < n_needed:
                before = len(selected)
                draw_idx = rng.integers(0, pool_size, size=batch_size)
                for j in draw_idx:
                    pep = pool[j]
                    if pep not in forbidden and pep not in selected_set:
                        selected.append(pep)
                        selected_set.add(pep)
                        if len(selected) == n_needed:
                            break
                if len(selected) == before:
                    stale_rounds += 1
                else:
                    stale_rounds = 0
                if stale_rounds >= 3:
                    break
                remaining = n_needed - len(selected)
                batch_size = min(pool_size, max(10000, int(remaining * 2)))

        if len(selected) < n_needed:
            perm = rng.permutation(pool_size)
            for j in perm:
                pep = pool[j]
                if pep not in forbidden and pep not in selected_set:
                    selected.append(pep)
                    selected_set.add(pep)
                    if len(selected) == n_needed:
                        break

        if len(selected) < n_needed:
            raise ValueError(
                f"Insufficient valid synthetic negatives for allele={allele}, length={length}. "
                f"Needed {n_needed:,}, found {len(selected):,}."
            )

        return selected

    train_neg_seqs = []
    train_neg_alleles = []
    test_neg_seqs = []
    test_neg_alleles = []

    for (allele, length), (n_train_pos, n_test_pos) in tqdm(
        group_pos_counts.items(),
        desc="Sampling synthetic negatives",
        leave=leave,
    ):
        n_train_neg = int(n_train_pos * train_neg_per_pos)
        n_test_neg = int(n_test_pos * test_neg_per_pos)
        n_total_neg = n_train_neg + n_test_neg

        if n_total_neg == 0:
            continue

        if int(length) not in neg_pools:
            raise ValueError(f"synthetic_negatives_by_length has no entry for length={length}.")

        sampled = sample_unique_from_pool(
            pool=neg_pools[int(length)],
            n_needed=n_total_neg,
            forbidden=forbidden_by_allele[allele],
            allele=allele,
            length=int(length),
        )

        if n_train_neg:
            train_sampled = sampled[:n_train_neg]
            train_neg_seqs.extend(train_sampled)
            train_neg_alleles.extend([allele] * n_train_neg)

        if n_test_neg:
            test_sampled = sampled[n_train_neg:]
            test_neg_seqs.extend(test_sampled)
            test_neg_alleles.extend([allele] * n_test_neg)

    train_neg = pd.DataFrame(
        {
            seq_col: train_neg_seqs,
            allele_col: train_neg_alleles,
            label_col: np.zeros(len(train_neg_seqs), dtype=np.int8),
        }
    )

    test_neg = pd.DataFrame(
        {
            seq_col: test_neg_seqs,
            allele_col: test_neg_alleles,
            label_col: np.zeros(len(test_neg_seqs), dtype=np.int8),
        }
    )

    train_df = pd.concat([train_pos, train_neg], ignore_index=True, copy=False)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True, copy=False)

    return train_df, test_df

def summarize_pos_neg_per_allele(df, allele_col='allele', label_col='label'):
    summary = (
        df.groupby(allele_col)[label_col]
          .agg(n_pos=lambda s: (s == 1).sum(),
               n_neg=lambda s: (s == 0).sum())
          .reset_index()
    )
    summary['total'] = summary['n_pos'] + summary['n_neg']
    return summary

def make_random_allele_sets(alleles, n_sets, n_per_set, seed=None):
    rng = np.random.default_rng(seed)
    alleles = list(alleles)
    if n_per_set > len(alleles):
        raise ValueError(f"n_per_set={n_per_set} exceeds available alleles ({len(alleles)})")
    return {i: rng.choice(alleles, size=n_per_set, replace=False).tolist()
            for i in range(1, n_sets + 1)}

def make_balanced_allele_sets(alleles, n_sets, n_per_set, seed=None):
    rng = np.random.default_rng(seed)
    alleles = list(alleles)
    if n_per_set > len(alleles):
        raise ValueError(f"n_per_set={n_per_set} exceeds available alleles ({len(alleles)})")

    counts = {a: 0 for a in alleles}
    result = {}
    for i in range(1, n_sets + 1):
        ranked = sorted(alleles, key=lambda a: (counts[a], rng.random()))
        chosen = ranked[:n_per_set]
        for a in chosen:
            counts[a] += 1
        rng.shuffle(chosen)
        result[i] = chosen
    return result

def sample_max_ratio(df, neg_ratio=99, label_col='label', allele_col='allele',
                     seq_col='seq', seed=42):
    rng = np.random.default_rng(seed)

    sampled_rows = []
    remaining_rows = []

    for allele, grp in tqdm(df.groupby(allele_col), desc="Sampling alleles"):
        pos = grp[grp[label_col] == 1]
        neg = grp[grp[label_col] == 0]

        n_pos = len(pos)
        n_neg = len(neg)

        if n_pos == 0 or n_neg == 0:
            remaining_rows.append(grp)
            continue

        # max positives supportable by available negatives
        max_pos = n_neg // neg_ratio

        if max_pos == 0:
            remaining_rows.append(grp)
            continue

        take_pos = min(max_pos, n_pos)
        take_neg = take_pos * neg_ratio

        pos_idx = rng.choice(len(pos), take_pos, replace=False)
        neg_idx = rng.choice(len(neg), take_neg, replace=False)

        sampled_pos = pos.iloc[pos_idx]
        sampled_neg = neg.iloc[neg_idx]
        remaining_pos = pos.iloc[~np.isin(np.arange(len(pos)), pos_idx)]
        remaining_neg = neg.iloc[~np.isin(np.arange(len(neg)), neg_idx)]

        sampled_rows.append(pd.concat([sampled_pos, sampled_neg]))
        remaining_rows.append(pd.concat([remaining_pos, remaining_neg]))

    sampled_df = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else pd.DataFrame()
    remaining_df = pd.concat(remaining_rows, ignore_index=True) if remaining_rows else pd.DataFrame()

    return sampled_df, remaining_df

def add_domain_id_based_on_pt_ft_split(df, domain_dict):
    """
    Add domain_id column mapping seq_allele to domain based on pt/ft dictionary.
    """
    pt_set = set(domain_dict['pt'])
    ft_set = set(domain_dict['ft'])
    df['domain_id'] = df['seq_allele'].isin(ft_set).astype(int)
    return df

def create_ft_splits(df, domain_dict, seed=42):
    """
    Create ft_split column dividing data into ~10% intervals.
    """
    rng = np.random.default_rng(seed)
    df = add_domain_id_based_on_pt_ft_split(df.copy(), domain_dict)
    n_total = len(df)
    n_ft = (df['domain_id'] == 1).sum()
    ft_percentage = n_ft / n_total
    n_splits = max(1, round(ft_percentage * 10))

    print(f"Total samples: {n_total:,}")
    print(f"FT samples: {n_ft:,} ({ft_percentage:.1%})")
    print(f"Creating {n_splits} FT splits (~{100/n_splits:.0f}% each)")
    df['ft_split'] = 0

    if n_splits == 1:
        df.loc[df['domain_id'] == 1, 'ft_split'] = 1
        return df

    ft_mask = df['domain_id'] == 1
    ft_data = df[ft_mask].copy()
    ft_alleles = ft_data['allele'].unique()
    split_assignments = np.zeros(len(ft_data), dtype=int)

    print(f"Processing {len(ft_alleles)} alleles...")
    for allele in tqdm(ft_alleles, desc="Splitting alleles"):
        allele_mask = ft_data['allele'] == allele
        allele_indices = np.where(allele_mask)[0]

        pos_mask = ft_data.loc[allele_mask, 'label'] == 1
        neg_mask = ft_data.loc[allele_mask, 'label'] == 0

        pos_indices = allele_indices[pos_mask.values]
        neg_indices = allele_indices[neg_mask.values]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        actual_splits = min(n_splits, n_pos) if n_pos > 0 else 0

        if actual_splits == 0:
            continue

        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        if n_pos <= actual_splits:
            for i, idx in enumerate(pos_indices):
                split_assignments[idx] = i + 1
        else:
            samples_per_split = n_pos / actual_splits
            for i, idx in enumerate(pos_indices):
                split_num = min(int(i / samples_per_split) + 1, actual_splits)
                split_assignments[idx] = split_num

        if n_neg > 0 and actual_splits > 0:
            split_pos_counts = np.zeros(actual_splits + 1, dtype=int)
            for idx in pos_indices:
                split_pos_counts[split_assignments[idx]] += 1

            neg_idx = 0
            for split_num in range(1, actual_splits + 1):
                n_pos_in_split = split_pos_counts[split_num]
                if n_pos_in_split == 0:
                    continue

                ratio = n_neg / n_pos if n_pos > 0 else 0
                n_neg_for_split = int(np.round(n_pos_in_split * ratio))
                n_neg_for_split = min(n_neg_for_split, n_neg - neg_idx)

                for i in range(n_neg_for_split):
                    if neg_idx >= n_neg:
                        break
                    split_assignments[neg_indices[neg_idx]] = split_num
                    neg_idx += 1

            while neg_idx < n_neg:
                for split_num in range(1, actual_splits + 1):
                    if split_pos_counts[split_num] > 0 and neg_idx < n_neg:
                        split_assignments[neg_indices[neg_idx]] = split_num
                        neg_idx += 1

    df.loc[ft_mask, 'ft_split'] = split_assignments

    print("\nSplit distribution:")
    for split_num in range(n_splits + 1):
        count = (df['ft_split'] == split_num).sum()
        pct = count / n_total * 100
        print(f"  Split {split_num}: {count:,} samples ({pct:.1f}%)")

    return df

def split_df_by_percentage(df, percentage, min_positive_count=30, allele_col='allele', label_col='label', seed=42):
    """
    Split dataframe into two based on percentage, maintaining allele-label distributions
    while ensuring minimum positive counts per allele in the main dataframe.
    
    For alleles below minimum positive count: no data is removed for df2
    For alleles above minimum: the actual percentage removed is capped by what maintains
    the minimum count, and this capped percentage is applied to both positives and negatives
    """
    rng = np.random.default_rng(seed)
    
    df = df.reset_index(drop=True)
    df['_temp_idx'] = df.index
    
    pos_counts = df[df[label_col] == 1].groupby(allele_col).size()
    
    max_allowed_pct = ((pos_counts - min_positive_count) / pos_counts).clip(lower=0)
    actual_pct_per_allele = pd.DataFrame({
        'max_allowed': max_allowed_pct,
        'requested': percentage
    }).min(axis=1)
    
    allele_sample_pct = actual_pct_per_allele.to_dict()
    
    pos_df = df[df[label_col] == 1].copy()
    pos_df['_sample_pct'] = pos_df[allele_col].map(allele_sample_pct)
    
    pos_grouped = pos_df.groupby(allele_col, sort=False)
    
    sampled_pos_indices = []
    for allele, group in tqdm(pos_grouped):
        sample_pct = allele_sample_pct.get(allele, 0)
        if sample_pct > 0:
            n_to_sample = int(len(group) * sample_pct)
            if n_to_sample > 0:
                sampled = group.sample(n=n_to_sample, random_state=rng)
                sampled_pos_indices.extend(sampled['_temp_idx'].tolist())
    
    neg_df = df[df[label_col] == 0].copy()
    neg_df['_sample_pct'] = neg_df[allele_col].map(allele_sample_pct)
    
    neg_grouped = neg_df.groupby(allele_col, sort=False)
    
    sampled_neg_indices = []
    for allele, group in neg_grouped:
        sample_pct = allele_sample_pct.get(allele, 0)
        if sample_pct > 0:
            n_to_sample = int(len(group) * sample_pct)
            if n_to_sample > 0:
                sampled = group.sample(n=n_to_sample, random_state=rng)
                sampled_neg_indices.extend(sampled['_temp_idx'].tolist())
    
    all_sampled_indices = sampled_pos_indices + sampled_neg_indices
    
    df2 = df[df['_temp_idx'].isin(all_sampled_indices)].drop('_temp_idx', axis=1)
    df1 = df[~df['_temp_idx'].isin(all_sampled_indices)].drop('_temp_idx', axis=1)
    
    return df1, df2


def add_louvain_flags(train_df, louvain_dict, louvain_ids, seq_allele_col='seq_allele'):
    """
    Add domain_id and ft_split columns flagging seq_alleles in louvain clusters.
    """
    train_df = train_df.copy()
    
    seq_alleles_to_flag = set()
    for louvain_id in louvain_ids:
        if louvain_id in louvain_dict:
            seq_alleles_to_flag.update(louvain_dict[louvain_id])
    
    mask = train_df[seq_allele_col].isin(seq_alleles_to_flag)
    train_df['domain_id'] = mask.astype(int)
    train_df['ft_split'] = mask.astype(int)
    
    return train_df

def duplicate_negatives_to_domain1(train_df, allele, label_col='label', allele_col='allele'):
    """
    Duplicate domain_id=0 negatives for an allele into domain_id=1.
    """
    mask = (train_df[allele_col] == allele) & (train_df[label_col] == 0) & (train_df['domain_id'] == 0)
    
    duplicates = train_df[mask].copy()
    duplicates['domain_id'] = 1
    duplicates['ft_split'] = 1
    
    return pd.concat([train_df, duplicates], ignore_index=True)


def assign_domain_split(df, seq_allele_dict, percentage, split_percentages_dict, seq_allele_col='seq_allele'):
    included = split_percentages_dict[percentage]
    next_split = max(included) + 1

    domain0 = set().union(*(seq_allele_dict[k] for k in included))
    domain1 = set(seq_allele_dict[next_split])

    df = df[df[seq_allele_col].isin(domain0 | domain1)].copy()
    df['domain_id'] = df[seq_allele_col].isin(domain1).astype(int)
    return df

def prefer_nonnegative_in_mixed_score_duplicates(
    df,
    score_col,
    seq_allele_col="seq_allele",
    threshold=0.0,
):
    """
    For duplicated seq_allele groups with mixed score signs, keep rows with
    score >= threshold and discard rows with score < threshold.

    Homogeneous duplicate groups are left unchanged:
        - all scores >= threshold: keep all rows
        - all scores < threshold: keep all rows

    Non-duplicated rows are also left unchanged.
    """
    counts = df[seq_allele_col].value_counts()
    duplicated = counts[counts > 1].index

    unique_df = df[~df[seq_allele_col].isin(duplicated)].copy()
    dup_df = df[df[seq_allele_col].isin(duplicated)].copy()

    keep_rows = []

    for seq_allele, group in dup_df.groupby(seq_allele_col):
        nonnegative_mask = group[score_col] >= threshold
        n_nonnegative = nonnegative_mask.sum()
        n_negative = (~nonnegative_mask).sum()

        if n_nonnegative > 0 and n_negative > 0:
            # Mixed group: prefer valid/nonnegative rows.
            keep_rows.append(group[nonnegative_mask])
        else:
            # Homogeneous group: preserve as-is.
            keep_rows.append(group)

    resolved = pd.concat([unique_df] + keep_rows, ignore_index=True)

    return resolved

def build_split_dict(df, split_col='ft_split', seq_allele_col='seq_allele'):
    return {k: g[seq_allele_col].tolist() for k, g in df.groupby(split_col)}

def merge_splits(split_dict, keys):
    return [sa for k in keys for sa in split_dict.get(k, [])]

def compute_number_of_epochs(train_data_pt, batch_size=3072, max_pos=200_000, n_epochs=120):
    expected_steps_per_epoch=math.ceil((max_pos*2)/batch_size) # x2 for negatives as well
    total_steps=n_epochs*expected_steps_per_epoch
    print("Expected steps per epoch:", expected_steps_per_epoch)
    print("Total steps:", total_steps)

    num_pos=len(train_data_pt.query("label == 1"))
    num_pos=min(num_pos, max_pos)
    current_num_steps_per_epoch=math.ceil((num_pos*2)/batch_size) # x2 for negatives as well
    current_total_steps=n_epochs*current_num_steps_per_epoch
    print("Current steps per epoch:", current_num_steps_per_epoch)
    print("Current total steps:", current_total_steps)

    needed_epochs=math.ceil(total_steps/current_num_steps_per_epoch)
    print("Needed number of epochs:", needed_epochs)
    return needed_epochs

def cluster_to_seq_alleles(df, cluster_col='cluster_id', seq_allele_col='seq_allele'):
    """
    Create dict mapping cluster to list of seq_alleles.
    """
    return df.groupby(cluster_col)[seq_allele_col].apply(list).to_dict()

def filter_allele_by_length(df, allele, length, allele_col='allele', peptide_col='peptide'):
    """
    Remove peptides for a specific allele that don't match target length.
    """
    mask = (df[allele_col] == allele) & (df[peptide_col].str.len() != length)
    return df[~mask]

def remove_test_overlap(main_df, test_df, seq_allele_col='seq_allele'):
    """
    Remove seq_alleles from main_df that appear in test_df.
    """
    overlap = set(test_df[seq_allele_col])
    return main_df[~main_df[seq_allele_col].isin(overlap)]

def remove_louvain_clusters(train_df, louvain_dict, louvain_ids, seq_allele_col='seq_allele'):
    """
    Remove seq_alleles from training df based on louvain cluster IDs.
    """
    if not louvain_ids:
        return train_df
    
    seq_alleles_to_remove = set()
    for louvain_id in louvain_ids:
        if louvain_id in louvain_dict:
            seq_alleles_to_remove.update(louvain_dict[louvain_id])
    
    return train_df[~train_df[seq_allele_col].isin(seq_alleles_to_remove)]

# resample negatives based on cluster
def resample_negatives_from_df2(df1, df2, cluster_col, seq_col='seq', label_col='label', seed=42):
    rng = np.random.default_rng(seed)

    result = df1.copy()

    df2_cluster_seqs = {cluster: grp[seq_col].values for cluster, grp in df2.groupby(cluster_col)}

    neg_mask = result[label_col] == 0
    neg_df = result[neg_mask]

    for cluster, grp in tqdm(neg_df.groupby(cluster_col), desc="Resampling clusters"):
        n = len(grp)
        pool = df2_cluster_seqs.get(cluster, np.array([]))

        if len(pool) == 0:
            continue

        sampled = pool[rng.choice(len(pool), n, replace=len(pool) < n)]
        result.loc[grp.index, seq_col] = sampled

    return result

def remove_sequences(main_list, remove_list):
    remove_set = set(remove_list)
    return [s for s in main_list if s not in remove_set]

def sample_pos_neg_per_cluster(pos_df, neg_df, n_per_cluster, min_count=1, ratio=99, peptide_col='peptide', cluster_col='cluster_id', pos_label=1, neg_label=0, seed=42):
    rng = np.random.default_rng(seed)
    
    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    
    sampled_rows = []
    
    cluster_counts = pos_df[cluster_col].value_counts()
    eligible_clusters = cluster_counts[cluster_counts >= min_count].index
    
    for cluster in eligible_clusters:
        cluster_pos = pos_df[pos_df[cluster_col] == cluster]
        n_sample_pos = min(n_per_cluster, len(cluster_pos))
        n_sample_neg = n_sample_pos * ratio
        
        sampled_pos_idx = rng.choice(cluster_pos.index, size=n_sample_pos, replace=False)
        sampled_pos = pos_df.loc[sampled_pos_idx].copy()
        sampled_pos['label'] = pos_label
        
        if len(neg_df) >= n_sample_neg:
            sampled_neg_idx = rng.choice(neg_df.index, size=n_sample_neg, replace=False)
            sampled_neg = neg_df.loc[sampled_neg_idx].copy()
            sampled_neg[cluster_col] = cluster
            sampled_neg['label'] = neg_label
            
            pos_df = pos_df.drop(index=sampled_pos_idx)
            neg_df = neg_df.drop(index=sampled_neg_idx)
            
            sampled_rows.append(sampled_pos)
            sampled_rows.append(sampled_neg)
    
    joint_df = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else pd.DataFrame()
    
    return pos_df, neg_df, joint_df

def sample_pos_neg_per_cluster_percent(pos_df, neg_df, percent=0.1, min_samples=1,  max_samples=None, min_count=1, ratio=99, peptide_col='peptide', cluster_col='cluster_id', pos_label=1, neg_label=0, seed=42):
    rng = np.random.default_rng(seed)
    
    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    
    sampled_rows = []
    
    cluster_counts = pos_df[cluster_col].value_counts()
    eligible_clusters = cluster_counts[cluster_counts >= min_count].index
    
    for cluster in eligible_clusters:
        cluster_pos = pos_df[pos_df[cluster_col] == cluster]
        cluster_size = len(cluster_pos)
        
        n_sample_pos = int(cluster_size * percent)
        
        if n_sample_pos < min_samples:
            n_sample_pos = min_samples
            if cluster_size - n_sample_pos <= 0:
                continue
        
        if max_samples is not None:
            n_sample_pos = min(n_sample_pos, max_samples)
        
        n_sample_pos = min(n_sample_pos, cluster_size - 1)
        
        if n_sample_pos <= 0:
            continue
        
        n_sample_neg = n_sample_pos * ratio
        
        sampled_pos_idx = rng.choice(cluster_pos.index, size=n_sample_pos, replace=False)
        sampled_pos = pos_df.loc[sampled_pos_idx].copy()
        sampled_pos['label'] = pos_label
        
        if len(neg_df) >= n_sample_neg:
            sampled_neg_idx = rng.choice(neg_df.index, size=n_sample_neg, replace=False)
            sampled_neg = neg_df.loc[sampled_neg_idx].copy()
            sampled_neg[cluster_col] = cluster
            sampled_neg['label'] = neg_label
            
            pos_df = pos_df.drop(index=sampled_pos_idx)
            neg_df = neg_df.drop(index=sampled_neg_idx)
            
            sampled_rows.append(sampled_pos)
            sampled_rows.append(sampled_neg)
    
    joint_df = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else pd.DataFrame()
    
    return pos_df, neg_df, joint_df


def train_test_split_by_allele(df, seq_col, allele_col, label_col, n_pos, pos_neg_ratio, seed=42):
    train_chunks, test_chunks = [], []
    
    for allele, group in df.groupby(allele_col):
        pos = group[group[label_col] == 1].sample(frac=1, random_state=seed)
        neg = group[group[label_col] == 0].sample(frac=1, random_state=seed)
        
        n_test_pos = min(n_pos, len(pos))
        n_test_neg = min(int(n_test_pos * pos_neg_ratio), len(neg))
        
        test_pos = pos.iloc[:n_test_pos]
        test_neg = neg.iloc[:n_test_neg]
        train_pos = pos.iloc[n_test_pos:]
        train_neg = neg.iloc[n_test_neg:]
        
        test_chunks.extend([test_pos, test_neg])
        train_chunks.extend([train_pos, train_neg])
    
    train = pd.concat(train_chunks, ignore_index=True).sample(frac=1, random_state=seed)
    test = pd.concat(test_chunks, ignore_index=True).sample(frac=1, random_state=seed)
    
    return train, test

def create_wt_mut_gene_train_test_sets(mut_df, wt_df, ms_train_df, ms_test_df, genes):
    mut_df_gene=mut_df[mut_df.gene.isin(genes)]

    wt_df_gene=wt_df[wt_df.name2.isin(mut_df_gene.name2.unique().tolist())]

    mut_df_no_gene=mut_df[~mut_df.seq_allele.isin(mut_df_gene.seq_allele.unique().tolist())]
    wt_df_no_gene=wt_df[~wt_df.seq_allele.isin(wt_df_gene.seq_allele.unique().tolist())]

    mut_df_no_gene_resolved=prefer_nonnegative_in_mixed_score_duplicates(mut_df_no_gene, "escore", seq_allele_col='seq_allele')

    wt_df_no_gene_resolved=prefer_nonnegative_in_mixed_score_duplicates(wt_df_no_gene, "escore", seq_allele_col='seq_allele')

    mut_wt_df_no_gene_resolved=pd.concat([mut_df_no_gene_resolved, wt_df_no_gene_resolved])

    mut_wt_df_no_gene_resolved.reset_index(inplace=True, drop=True)

    mut_wt_df_no_gene_resolved=create_seq_allele_column(mut_wt_df_no_gene_resolved)
    wt_df_gene=create_seq_allele_column(wt_df_gene)

    # remove test overlap from train
    seq_allele_overlap=set(ms_test_df.seq_allele).intersection(set(mut_wt_df_no_gene_resolved.seq_allele))
    print(1, len(seq_allele_overlap))

    mut_wt_df_no_gene_resolved=mut_wt_df_no_gene_resolved[~mut_wt_df_no_gene_resolved.seq_allele.isin(seq_allele_overlap)]

    # remove ms train overlap from escape seq data
    seq_allele_overlap=set(ms_train_df.seq_allele).intersection(set(mut_wt_df_no_gene_resolved.seq_allele))
    print(2, len(seq_allele_overlap))

    mut_wt_df_no_gene_resolved=mut_wt_df_no_gene_resolved[~mut_wt_df_no_gene_resolved.seq_allele.isin(seq_allele_overlap)]

    # seq overlap
    seq_overlap=set(ms_train_df.seq).intersection(set(mut_wt_df_no_gene_resolved.seq))
    print(3, len(seq_overlap))

    mut_wt_df_no_gene_resolved=mut_wt_df_no_gene_resolved[~mut_wt_df_no_gene_resolved.seq.isin(seq_overlap)]

    mut_df_gene["mut_type"]="mut"
    wt_df_gene["mut_type"]="wt"

    mut_wt_df_gene=pd.concat([mut_df_gene, wt_df_gene])

    mut_wt_df_gene.reset_index(inplace=True, drop=True)

    seq_allele_overlap=set(ms_train_df.seq_allele).intersection(set(mut_wt_df_gene.seq_allele))
    print(4, len(seq_allele_overlap))

    mut_wt_df_gene=mut_wt_df_gene[~mut_wt_df_gene.seq_allele.isin(seq_allele_overlap)]

    seq_overlap=set(ms_train_df.seq).intersection(set(mut_wt_df_gene.seq))
    print(5, len(seq_overlap))

    mut_wt_df_gene=mut_wt_df_gene[~mut_wt_df_gene.seq.isin(seq_overlap)]

    seq_allele_overlap=set(mut_wt_df_no_gene_resolved.seq_allele).intersection(set(mut_wt_df_gene.seq_allele))
    print(6, len(seq_allele_overlap))

    if len(seq_allele_overlap) != 0:
        print(mut_wt_df_no_gene_resolved[mut_wt_df_no_gene_resolved.seq_allele.isin(seq_allele_overlap)].gene.unique().tolist())

    seq_allele_overlap=set(ms_train_df.seq_allele).intersection(set(mut_wt_df_gene.seq_allele))
    print(7, len(seq_allele_overlap))

    seq_overlap=set(mut_wt_df_no_gene_resolved.seq).intersection(set(mut_wt_df_gene.seq))
    print(8, len(seq_overlap))

    seq_overlap=set(ms_train_df.seq).intersection(set(mut_wt_df_gene.seq))
    print(9, len(seq_overlap))

    return mut_wt_df_no_gene_resolved, mut_wt_df_gene

def pair_wt_mut_and_diff_with_quadrants_v2(
    df: pd.DataFrame,
    score_cols: list[str],
    id_col: str,
    allele_col: str = "allele",
    mut_type_col: str = "mut_type",
    wt_value: str = "wt",
    mut_value: str = "mut",
    label_col: str = "label",
    mutation_col: str = "mutation",
    seq_col: str = "seq",
    quadrant_col: str = "wt_mut_quadrant",
    seed = None,
    dup_mode: str = "random",
    dup_score_col = None,   # used for max/min
) -> pd.DataFrame:
    """
    Split df into WT and MUT by mut_type_col, resolve duplicates per (id_col, allele_col)
    using dup_mode, then inner-join WT/MUT on (id_col, allele_col). Compute:
      - <score>_wt, <score>_mut, and <score> = (mut - wt)
      - quadrant labels from existing binary label_col

    dup_mode options:
      - "random": pick one row per key uniformly at random (seeded)
      - "max":    pick row with max dup_score_col per key
      - "min":    pick row with min dup_score_col per key
      - "mean":   average numeric cols (score_cols + label_col (+ dup_score_col if given)) per key
      - "pair_enumeration": aggregate replicates per (key, seq) then enumerate all WTxMUT pairs
                            for that key with seq_wt != seq_mut (unique seq pairing).
    """
    rng = np.random.default_rng(seed)
    key_cols = [id_col, allele_col]
    dup_mode = dup_mode.lower().strip()

    wt_raw = df[df[mut_type_col] == wt_value].copy()
    mut_raw = df[df[mut_type_col] == mut_value].copy()

    # --- helper: resolve duplicates within a single split (WT or MUT) ---
    def resolve_dups_simple(x: pd.DataFrame) -> pd.DataFrame:
        if x.empty:
            return x

        if dup_mode == "random":
            x = x.assign(_rand=rng.random(len(x)))
            x = x.sort_values(key_cols + ["_rand"]).drop_duplicates(key_cols, keep="first")
            return x.drop(columns="_rand")

        if dup_mode in {"max", "min"}:
            if dup_score_col is None:
                raise ValueError("dup_score_col must be provided when dup_mode is 'max' or 'min'")
            if dup_score_col not in x.columns:
                raise ValueError(f"dup_score_col='{dup_score_col}' not found in df")

            tmp = x.copy()
            tmp["_dup_score"] = pd.to_numeric(tmp[dup_score_col], errors="coerce")
            asc = (dup_mode == "min")
            tmp = tmp.sort_values(key_cols + ["_dup_score"], ascending=[True, True, asc])
            return tmp.drop_duplicates(key_cols, keep="first").drop(columns="_dup_score")

        if dup_mode == "mean":
            avg_cols = list(score_cols)
            if label_col in x.columns:
                avg_cols.append(label_col)
            if dup_score_col is not None and dup_score_col in x.columns:
                avg_cols.append(dup_score_col)
            avg_cols = list(dict.fromkeys(avg_cols))

            tmp = x.copy()
            for c in avg_cols:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

            agg = {c: "mean" for c in avg_cols}
            for c in [mutation_col, seq_col, mut_type_col]:
                if c in tmp.columns:
                    agg[c] = "first"

            return tmp.groupby(key_cols, sort=False, as_index=False).agg(agg)

        raise ValueError("resolve_dups_simple called with unsupported dup_mode")

    # --- NEW: pair enumeration path (do NOT collapse to one row per key) ---
    if dup_mode == "pair_enumeration":
        # We enumerate across unique sequences per key. To avoid blowing up from technical replicates,
        # we aggregate numeric fields per (key, seq) within WT and within MUT.
        def agg_by_key_seq(x: pd.DataFrame) -> pd.DataFrame:
            if x.empty:
                return x

            # numeric cols to average per (key, seq)
            avg_cols = list(score_cols)
            if label_col in x.columns:
                avg_cols.append(label_col)
            if dup_score_col is not None and dup_score_col in x.columns:
                avg_cols.append(dup_score_col)
            avg_cols = list(dict.fromkeys(avg_cols))

            tmp = x.copy()
            for c in avg_cols:
                if c in tmp.columns:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

            # group by (id, allele, seq) -> average numeric; keep first for mutation/seq/mut_type
            gcols = key_cols + [seq_col]
            agg = {c: "mean" for c in avg_cols if c in tmp.columns}
            if mutation_col in tmp.columns:
                agg[mutation_col] = "first"
            if mut_type_col in tmp.columns:
                agg[mut_type_col] = "first"

            out = tmp.groupby(gcols, sort=False, as_index=False).agg(agg)
            return out

        wt_df = agg_by_key_seq(wt_raw)
        mut_df = agg_by_key_seq(mut_raw)

        # Rename seq columns before merge
        wt_df = wt_df.rename(columns={seq_col: "seq_wt"})
        mut_df = mut_df.rename(columns={seq_col: "seq_mut"})

        # Merge on key_cols: because multiple rows per key exist in each df, merge produces Cartesian pairs
        paired = wt_df.merge(mut_df, on=key_cols, suffixes=("_wt", "_mut"), how="inner")

        # Enforce seq mismatch (critical)
        paired = paired[paired["seq_wt"] != paired["seq_mut"]]

        # Drop duplicate pairings if any (same key + same seq_wt/seq_mut repeated)
        paired = paired.drop_duplicates(key_cols + ["seq_wt", "seq_mut"], keep="first")

    else:
        # Original behavior: collapse duplicates to 1 row per key in each split
        wt_df = resolve_dups_simple(wt_raw)
        mut_df = resolve_dups_simple(mut_raw)

        needed = key_cols + [mutation_col, seq_col, label_col] + score_cols
        wt_df = wt_df[[c for c in needed if c in wt_df.columns]].rename(columns={seq_col: "seq_wt"})
        mut_df = mut_df[[c for c in needed if c in mut_df.columns]].rename(columns={seq_col: "seq_mut"})

        paired = wt_df.merge(mut_df, on=key_cols, suffixes=("_wt", "_mut"), how="inner")

    # Choose mutation label (prefer WT if present)
    if f"{mutation_col}_wt" in paired.columns and f"{mutation_col}_mut" in paired.columns:
        paired[mutation_col] = paired[f"{mutation_col}_wt"].combine_first(paired[f"{mutation_col}_mut"])
    elif f"{mutation_col}_wt" in paired.columns:
        paired[mutation_col] = paired[f"{mutation_col}_wt"]
    elif f"{mutation_col}_mut" in paired.columns:
        paired[mutation_col] = paired[f"{mutation_col}_mut"]
    else:
        paired[mutation_col] = np.nan

    out = pd.DataFrame({
        id_col: paired[id_col],
        allele_col: paired[allele_col],
        mutation_col: paired[mutation_col],
        "seq_wt": paired.get("seq_wt"),
        "seq_mut": paired.get("seq_mut"),
        "wt_label": pd.to_numeric(paired.get(f"{label_col}_wt"), errors="coerce"),
        "mut_label": pd.to_numeric(paired.get(f"{label_col}_mut"), errors="coerce"),
    })

    # Add raw WT/MUT scores + deltas for each score col
    for c in score_cols:
        c_wt, c_mut = f"{c}_wt", f"{c}_mut"
        if c_wt in paired.columns and c_mut in paired.columns:
            wt_vals = pd.to_numeric(paired[c_wt], errors="coerce")
            mut_vals = pd.to_numeric(paired[c_mut], errors="coerce")
            out[c_wt] = wt_vals
            out[c_mut] = mut_vals
            out[c] = mut_vals - wt_vals
        else:
            out[c_wt] = np.nan
            out[c_mut] = np.nan
            out[c] = np.nan

    # Quadrants from existing binary labels
    wt = out["wt_label"]
    mut = out["mut_label"]
    out[quadrant_col] = np.select(
        condlist=[
            (wt == 0) & (mut == 0),
            (wt == 1) & (mut == 1),
            (wt == 1) & (mut == 0),
            (wt == 0) & (mut == 1),
        ],
        choicelist=[
            "WT-/MUT-",
            "WT+/MUT+",
            "WT+/MUT-",
            "WT-/MUT+",
        ],
        default=np.nan,
    )

    return out

def add_quadrant_binary_label(
    df: pd.DataFrame,
    quadrant_col: str,
    quadrant_name: str,
    out_label_col: str = "quadrant_label",
) -> pd.DataFrame:
    """
    Add a binary label column marking rows in `quadrant_name` as 1, else 0.
    """
    out = df.copy()
    out[out_label_col] = (out[quadrant_col] == quadrant_name).astype(np.int8)
    return out

def filter_sequences_with_x(df, seq_col='seq'):
    return df[df[seq_col].str.contains('X', na=False)].reset_index(drop=True)

def filter_sequences_without_x(df, seq_col='seq'):
    return df[~df[seq_col].str.contains('X', na=False)].reset_index(drop=True)

def allele_spearman_matrix(df, score_cols, ref_col, allele_col='allele'):
    records = []
    
    for allele, group in df.groupby(allele_col):
        if len(group) < 3:
            continue
        row = {'allele': allele}
        for col in score_cols:
            corr, _ = spearmanr(group[col], group[ref_col])
            row[col] = corr
        records.append(row)
    
    return pd.DataFrame(records).set_index('allele')

def filter_dict_by_peptides(peptide_dict, exclude_list):
    exclude_set = set(exclude_list)
    return {length: [p for p in peptides if p not in exclude_set]
            for length, peptides in peptide_dict.items()}

def generate_kmers_from_proteome(proteome_df, kmer_lengths=[9, 10, 11]):
    """
    Generate length-specific kmers from a proteome dataframe.
    """
    kmer_dict = {length: [] for length in kmer_lengths}
    protein_sequences = proteome_df['seq'].tolist()
    
    for protein in tqdm(protein_sequences, desc="Generating kmers from proteome"):
        if not isinstance(protein, str):
            continue
            
        for length in kmer_lengths:
            if len(protein) >= length:
                for i in range(len(protein) - length + 1):
                    kmer = protein[i:i+length]
                    if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in kmer):
                        kmer_dict[length].append(kmer)
    
    return kmer_dict


def generate_random_peptides(n_peptides=10000, lengths=[9, 10, 11]):
    """
    Generate random peptides of specified lengths.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    peptide_dict = {length: [] for length in lengths}
    
    for length in tqdm(lengths, desc="Generating random peptides"):
        for _ in range(n_peptides):
            # Generate a random peptide of specified length
            peptide = ''.join(random.choice(amino_acids) for _ in range(length))
            peptide_dict[length].append(peptide)
    
    return peptide_dict


def generate_synthetic_negatives(
    df, 
    proteome_kmers, 
    random_kmers,
    negative_sets=5, 
    proteome_pct=0.9,
    random_pct=0.1,
    anchor_dict=None
):
    """
    Generate synthetic negatives for each allele.
    """
    synthetic_negatives = {}
    alleles = df['allele'].unique()
    
    for allele in tqdm(alleles, desc="Generating synthetic negatives per allele"):
        allele_positives = df[(df['allele'] == allele) & (df['label'] == 1)]['seq'].tolist()
        
        if not allele_positives:
            continue
        
        length_counts = defaultdict(int)
        for peptide in allele_positives:
            length_counts[len(peptide)] += 1
        
        total_positives = len(allele_positives)
        length_proportions = {length: count / total_positives for length, count in length_counts.items()}
        
        total_negatives_needed = total_positives * negative_sets
        proteome_negatives_needed = int(total_negatives_needed * proteome_pct)
        random_negatives_needed = total_negatives_needed - proteome_negatives_needed
        
        proteome_per_length = {
            length: int(proteome_negatives_needed * prop) 
            for length, prop in length_proportions.items()
        }
        random_per_length = {
            length: int(random_negatives_needed * prop) 
            for length, prop in length_proportions.items()
        }
        
        proteome_total = sum(proteome_per_length.values())
        if proteome_total < proteome_negatives_needed:
            most_common_length = max(length_counts.items(), key=lambda x: x[1])[0]
            proteome_per_length[most_common_length] += proteome_negatives_needed - proteome_total
            
        random_total = sum(random_per_length.values())
        if random_total < random_negatives_needed:
            most_common_length = max(length_counts.items(), key=lambda x: x[1])[0]
            random_per_length[most_common_length] += random_negatives_needed - random_total
        
        allele_negatives = []
        
        positive_set = set(allele_positives)
        positive_substrings = set()
        for peptide in allele_positives:
            for i in range(len(peptide) - 8 + 1):  # Consider 8-mers and longer
                positive_substrings.add(peptide[i:i+8])
        
        for length, count in proteome_per_length.items():
            if length not in proteome_kmers or not proteome_kmers[length]:
                continue
                
            available_kmers = proteome_kmers[length].copy()
            random.shuffle(available_kmers)
            
            sampled = 0
            for kmer in available_kmers:
                is_valid = True
                
                if kmer in positive_set:
                    is_valid = False
                    
                if is_valid and len(kmer) >= 8:
                    for i in range(len(kmer) - 8 + 1):
                        if kmer[i:i+8] in positive_substrings:
                            is_valid = False
                            break
                
                if is_valid:
                    allele_negatives.append(kmer)
                    sampled += 1
                    
                if sampled >= count:
                    break
        
        for length, count in random_per_length.items():
            if length not in random_kmers or not random_kmers[length]:
                continue
                
            available_kmers = random_kmers[length].copy()
            random.shuffle(available_kmers)
            
            sampled = 0
            for kmer in available_kmers:
                is_valid = True
                if kmer in positive_set:
                    is_valid = False
                    
                if is_valid and len(kmer) >= 8:
                    for i in range(len(kmer) - 8 + 1):
                        if kmer[i:i+8] in positive_substrings:
                            is_valid = False
                            break
                
                if is_valid:
                    allele_negatives.append(kmer)
                    sampled += 1
                    
                if sampled >= count:
                    break
        
        synthetic_negatives[allele] = allele_negatives
        
        print(f"Allele {allele}: {len(allele_positives)} positives, {len(allele_negatives)} synthetic negatives generated")
    
    return synthetic_negatives


def build_aho(patterns):
    A = ahocorasick.Automaton()
    for p in patterns:
        A.add_word(p, p)
    A.make_automaton()
    return A

def filter_kmers(candidates, forbid_set, automaton):
    """Return list of kmers that are not in forbid_set and contain no forbidden 8-mer."""
    out = []
    for k in candidates:
        if k in forbid_set: # exact positive
            continue
        if any(automaton.iter(k)): # shares an 8-mer
            continue
        out.append(k)
    return out

def flatten_peptide_allele_table(df):
    df_reset = df.reset_index()
    peptide_col = df_reset.columns[0]
    
    df_melted = df_reset.melt(
        id_vars=peptide_col,
        var_name='allele',
        value_name='label'
    )
    df_melted = df_melted.rename(columns={peptide_col: 'peptide'})
    df_melted['label'] = df_melted['label'].map({True: 1, False: 0})
    
    return df_melted

def process_binary_excel_files(folder_path):
    folder_path = Path(folder_path)
    binary_files = sorted(folder_path.glob("*_binary.xlsx"))
    if not binary_files:
        binary_files = sorted(folder_path.glob("*_binary.xls"))
    all_data = []
    for file_path in tqdm(binary_files, desc="Processing files"):
        protein_set = file_path.stem.replace("_binary", "")
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in tqdm(excel_file.sheet_names, desc=f"Processing {protein_set} sheets", leave=False):
            df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
            df_flat = flatten_peptide_allele_table_fast(df_sheet)
            df_flat['allele'] = df_flat['allele'].str.replace('*', '', regex=False).str.replace(':', '', regex=False)
            df_flat['protein'] = sheet_name
            df_flat['protein_set'] = protein_set
            all_data.append(df_flat)

    print("Combining all data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[['seq', 'allele', 'label', 'protein', 'protein_set']]
    return combined_df

def flatten_peptide_allele_table_fast(df):
    df_reset = df.reset_index()
    seq_col = df_reset.columns[0]
    df_melted = df_reset.melt(
        id_vars=seq_col,
        var_name='allele',
        value_name='label'
    )
    df_melted = df_melted.rename(columns={seq_col: 'seq'})
    df_melted['label'] = df_melted['label'].astype(int)
    
    return df_melted


"""
Allele transfer learning
"""

def get_transfer_learning_outputs(output_dir_path, file_name_pattern):
    allele_test_set_full=pd.DataFrame()
    for file in tqdm(os.listdir(output_dir_path)):
        if file.startswith(file_name_pattern): # for final version
            allele=file.removesuffix(".csv").split("_")[-1]
            allele_test_set=pd.read_csv(f"{output_dir_path}/{file}")
            
            model_name=f"model_ft_{allele}" # no replay
            allele_test_set.rename(columns={model_name: "model_ft"}, inplace=True)

            model_name=f"model_pepcl_{allele}" # no replay
            allele_test_set.rename(columns={model_name: "model_pepcl"}, inplace=True)

            allele_test_set_full=pd.concat([allele_test_set_full, allele_test_set])

    allele_test_set_full=create_seq_allele_column(allele_test_set_full)
    allele_test_set_full.drop_duplicates(subset=["seq_allele"], inplace=True)
    allele_test_set_full.reset_index(inplace=True, drop=True)
    return allele_test_set_full

def compute_max_ratio_counts(df, label_col='label', allele_col='allele', neg_ratio=99):
    counts = df.groupby([allele_col, label_col]).size().unstack(fill_value=0)
    if 1 not in counts.columns:
        counts[1] = 0
    if 0 not in counts.columns:
        counts[0] = 0

    records = []
    for allele in tqdm(counts.index, desc="Computing counts"):
        n_pos = counts.loc[allele, 1]
        n_neg = counts.loc[allele, 0]

        # maximize positives given neg constraint
        max_pos_from_neg = n_neg // neg_ratio
        take_pos = min(n_pos, max_pos_from_neg)
        take_neg = take_pos * neg_ratio

        records.append({
            'allele': allele,
            'available_pos': n_pos,
            'available_neg': n_neg,
            'take_pos': take_pos,
            'take_neg': take_neg,
        })

    return pd.DataFrame(records).set_index('allele')

def add_domain_id_with_alleles(df, alleles, allele_col='allele'):
    df = df.copy()
    df['domain_id'] = df[allele_col].isin(alleles).astype(int)
    return df
    

"""
Background generation.
"""

# convert swiss prot to df
def fasta_to_dataframe(fasta_path):
    metadata_list = []
    sequence_list = []
    
    with open(fasta_path, 'r') as f:
        current_metadata = None
        current_sequence = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(">"):
                if current_metadata is not None:
                    metadata_list.append(current_metadata)
                    sequence_list.append("".join(current_sequence))
                
                current_metadata = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if current_metadata is not None:
            metadata_list.append(current_metadata)
            sequence_list.append("".join(current_sequence))
            
    df = pd.DataFrame({
        "Metadata": metadata_list,
        "seq": sequence_list
    })
    
    return df

# human proteome df
def sample_background_distribution(kmer_dict, alleles, lengths, n_sample, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    
    total = len(alleles) * len(lengths)
    with tqdm(total=total, desc="Sampling background") as pbar:
        for length in lengths:
            peptides = kmer_dict[length]
            n_avail = len(peptides)
            peptides_arr = np.array(peptides)
            
            for allele in alleles:
                idx = rng.integers(0, n_avail, size=n_sample)
                sampled = peptides_arr[idx]
                records.append(pd.DataFrame({
                    'seq': sampled,
                    'allele': allele,
                    'seq_len': length
                }))
                pbar.update(1)
    
    return pd.concat(records, ignore_index=True)


# sample swiss prot background peptides
def sample_background_peptides(df, N, lengths, seed=42):
    """
    Randomly samples N peptides for each length from a proteome DataFrame.
    """
    rng = np.random.default_rng(seed)
    print("Cleaning sequences (removing non-standard amino acids)...")
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    df = df[df['seq'].apply(lambda x: all(char in valid_aa for char in x))].copy()
    
    sequences = df['seq'].tolist()
    results = {}

    for k in tqdm(lengths, desc="Processing Lengths"):
        counts = np.array([len(s) - k + 1 for s in sequences])
        counts[counts < 0] = 0
        
        cumsum = np.cumsum(counts)
        total_possible_peptides = cumsum[-1]
        
        if total_possible_peptides < N:
            print(f"Warning: Only {total_possible_peptides} possible peptides for length {k}. Sampling all.")
            N_to_sample = total_possible_peptides
        else:
            N_to_sample = N

        random_indices = rng.choice(total_possible_peptides, size=N_to_sample, replace=False)
        prot_indices = np.searchsorted(cumsum, random_indices, side='right')
        
        starts = np.zeros(N_to_sample, dtype=int)
        mask = prot_indices > 0
        starts[mask] = random_indices[mask] - cumsum[prot_indices[mask] - 1]
        starts[~mask] = random_indices[~mask]

        sampled_peptides = []
        for i in tqdm(range(N_to_sample), desc=f"Extracting {k}-mers", leave=False):
            p_idx = prot_indices[i]
            s_pos = starts[i]
            sampled_peptides.append(sequences[p_idx][s_pos : s_pos + k])
            
        results[k] = sampled_peptides

    return results

# assign alleles
def assign_alleles(df, alleles, allele_col='allele'):
    return pd.concat(
        [df.assign(**{allele_col: allele}) for allele in alleles],
        ignore_index=True
    )

"""
Compute performance metrics
"""

def compute_ap_matrix_optimized(df, score_cols, label_col='label', allele_col='allele', *, return_float64=True, disable_tqdm=False):
    score_cols = [c for c in score_cols if c in df.columns]

    alleles = sorted(df[allele_col].unique())
    groups = df.groupby(allele_col, sort=False).indices

    y_all = df[label_col].to_numpy()
    score_arrays = [df[c].to_numpy() for c in score_cols]

    dtype = np.float64 if return_float64 else float
    ap_values = np.full((len(alleles), len(score_cols)), np.nan, dtype=dtype)

    for i, allele in enumerate(tqdm(alleles, disable=disable_tqdm)):
        idx = groups.get(allele)
        if idx is None or len(idx) == 0:
            continue

        y = y_all[idx]
        y_notna = ~pd.isna(y)

        for j, s_arr in enumerate(score_arrays):
            s = s_arr[idx]
            valid = y_notna & (~pd.isna(s))

            if valid.sum() > 1:
                y_true = y[valid]
                if len(np.unique(y_true)) >= 2:
                    y_score = s[valid]
                    try:
                        ap_values[i, j] = average_precision_score(y_true, y_score)
                    except Exception:
                        ap_values[i, j] = np.nan
                else:
                    ap_values[i, j] = np.nan
            else:
                ap_values[i, j] = np.nan

    return pd.DataFrame(ap_values, index=alleles, columns=score_cols)

def compute_aho_split_map_matrix(ap_matrix, allele_splits, full_col='mhcprime'):
    index_alleles = list(ap_matrix.index)

    col_specs = [
        ('mhcprime',    lambda split: full_col),
        ('model_base',  lambda split: f"mhcprime_base_aho_split_{split}"),
        ('model_ft',    lambda split: f"mhcprime_ft_aho_split_{split}"),
        ('model_pepcl', lambda split: f"mhcprime_pepcl_aho_split_{split}"),
    ]

    def safe_mean(alleles, col):
        if col not in ap_matrix.columns or not alleles:
            return np.nan
        return np.nanmean(ap_matrix.loc[alleles, col])

    records = []
    for split, allele_list in allele_splits.items():
        in_split = set(allele_list)
        aho_alleles = [a for a in index_alleles if a in in_split]       # held-out alleles for this split
        pt_alleles  = [a for a in index_alleles if a not in in_split]   # all remaining alleles

        row = {'split': split}
        for prefix, col_fn in col_specs:                                # first set: _pt
            row[f'{prefix}_pt_alleles_map'] = safe_mean(pt_alleles, col_fn(split))
        for prefix, col_fn in col_specs:                                # second set: _aho
            row[f'{prefix}_aho_alleles_map'] = safe_mean(aho_alleles, col_fn(split))
        records.append(row)

    return pd.DataFrame(records).set_index('split')

def _raw_auc01(y_true, y_score, max_fpr=0.1):
    """
    Raw, unnormalized partial AUROC over FPR [0, max_fpr].
    Maximum possible value is max_fpr.
    For max_fpr=0.1, max possible raw AUC0.1 is 0.1.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Add max_fpr by interpolation if it is not already present.
    if not np.any(np.isclose(fpr, max_fpr)):
        tpr_at_max = np.interp(max_fpr, fpr, tpr)
        fpr = np.append(fpr, max_fpr)
        tpr = np.append(tpr, tpr_at_max)

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    mask = fpr <= max_fpr
    return auc(fpr[mask], tpr[mask])


def _ppvn(y_true, y_score):
    """
    PPVn where n is the number of positives in y_true.
    PPVn = fraction of true positives among the top n scored samples.
    """
    n = int(np.sum(y_true == 1))

    if n <= 0:
        return np.nan

    n = min(n, len(y_true))

    # Stable descending sort.
    top_idx = np.argsort(-y_score, kind="mergesort")[:n]
    return np.mean(y_true[top_idx] == 1)


def compute_metric_matrix_optimized(
    df,
    score_cols,
    method,
    label_col="label",
    allele_col="allele",
    *,
    auc01_max_fpr=0.1,
    auc01_normalized=True,
    return_float64=True,
    disable_tqdm=False,
):
    """
    Compute per-allele metric matrix for multiple score columns.
    """

    valid_methods = {"auc01", "auc", "ppvn"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got: {method}")

    score_cols = [c for c in score_cols if c in df.columns]

    alleles = sorted(df[allele_col].unique())
    groups = df.groupby(allele_col, sort=False).indices

    y_all = df[label_col].to_numpy()
    score_arrays = [df[c].to_numpy() for c in score_cols]

    dtype = np.float64 if return_float64 else float
    metric_values = np.full((len(alleles), len(score_cols)), np.nan, dtype=dtype)

    for i, allele in enumerate(tqdm(alleles, disable=disable_tqdm)):
        idx = groups.get(allele)
        if idx is None or len(idx) == 0:
            continue

        y = y_all[idx]
        y_notna = ~pd.isna(y)

        for j, s_arr in enumerate(score_arrays):
            s = s_arr[idx]
            valid = y_notna & (~pd.isna(s))

            if valid.sum() <= 1:
                continue

            y_true = y[valid]
            y_score = s[valid]

            if len(np.unique(y_true)) < 2:
                continue

            try:
                if method == "auc":
                    metric_values[i, j] = roc_auc_score(y_true, y_score)

                elif method == "auc01":
                    if auc01_normalized:
                        metric_values[i, j] = roc_auc_score(
                            y_true,
                            y_score,
                            max_fpr=auc01_max_fpr,
                        )
                    else:
                        metric_values[i, j] = _raw_auc01(
                            y_true,
                            y_score,
                            max_fpr=auc01_max_fpr,
                        )

                elif method == "ppvn":
                    metric_values[i, j] = _ppvn(y_true, y_score)

            except Exception:
                metric_values[i, j] = np.nan

    return pd.DataFrame(metric_values, index=alleles, columns=score_cols)

def compute_spearman_matrix_optimized(df, score_cols, reference_col, allele_col='allele', *, return_float64=True, disable_tqdm=False):
    score_cols = [c for c in score_cols if c in df.columns and c != reference_col]

    alleles = sorted(df[allele_col].unique())
    groups = df.groupby(allele_col, sort=False).indices

    ref_all = df[reference_col].to_numpy()
    score_arrays = [df[c].to_numpy() for c in score_cols]

    dtype = np.float64 if return_float64 else float
    rho_values = np.full((len(alleles), len(score_cols)), np.nan, dtype=dtype)

    for i, allele in enumerate(tqdm(alleles, disable=disable_tqdm)):
        idx = groups.get(allele)
        if idx is None or len(idx) == 0:
            continue

        ref = ref_all[idx]
        ref_notna = ~pd.isna(ref)

        for j, s_arr in enumerate(score_arrays):
            s = s_arr[idx]
            valid = ref_notna & (~pd.isna(s))

            if valid.sum() > 1:
                ref_valid = ref[valid]
                s_valid = s[valid]
                # spearman is undefined if either side is constant
                if len(np.unique(ref_valid)) >= 2 and len(np.unique(s_valid)) >= 2:
                    try:
                        rho_values[i, j] = spearmanr(ref_valid, s_valid).correlation
                    except Exception:
                        rho_values[i, j] = np.nan
                else:
                    rho_values[i, j] = np.nan
            else:
                rho_values[i, j] = np.nan

    return pd.DataFrame(rho_values, index=alleles, columns=score_cols)

def pairwise_spearman_by_group(df, score_cols, split_col='allele', disable_tqdm=False):
    score_cols = [c for c in score_cols if c in df.columns]
    n = len(score_cols)

    sum_mat = np.zeros((n, n))
    cnt_mat = np.zeros((n, n))

    for _, grp in tqdm(df.groupby(split_col, sort=False), disable=disable_tqdm):
        cmat = grp[score_cols].corr(method='spearman').to_numpy()
        valid = ~np.isnan(cmat)
        sum_mat[valid] += cmat[valid]
        cnt_mat[valid] += 1

    with np.errstate(invalid='ignore'):
        avg = sum_mat / cnt_mat

    return pd.DataFrame(avg, index=score_cols, columns=score_cols)

def mixed_df_map_pos_focused_nan_agnostic_fast(df1, df2, score_cols, allele_pos_counts,label_col1='label', label_col2='label', allele_col='allele', neg_ratio=99, n_iterations=10, seed=42):

    d1 = df1.rename(columns={label_col1: '_label'}).assign(_source=1).reset_index(drop=True)
    d2 = df2.rename(columns={label_col2: '_label'}).assign(_source=2).reset_index(drop=True)
    d2.index = d2.index + len(d1)

    nan_summary = {}
    for col in score_cols:
        nan_rows = []
        nan_mask1 = d1[col].isna()
        nan_mask2 = d2[col].isna()
        if nan_mask1.any():
            nan_rows += list(zip(d1.loc[nan_mask1, 'seq'], d1.loc[nan_mask1, allele_col]))
        if nan_mask2.any():
            nan_rows += list(zip(d2.loc[nan_mask2, 'seq'], d2.loc[nan_mask2, allele_col]))
        if nan_rows:
            nan_summary[col] = nan_rows

    all_data = pd.concat([d1, d2])
    score_arrays = {col: all_data[col].values for col in score_cols}
    label_array = all_data['_label'].values
    allele_array = all_data[allele_col].values
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_data.index)}

    def build_index(d, has_neg=True):
        groups = {}
        for allele, grp in d.groupby(allele_col):
            pos_idx = grp.index[grp['_label'] == 1].values
            groups[allele] = {'pos': pos_idx}
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

    iteration_maps = {col: [] for col in score_cols}
    allele_split_iter = []
    rng = np.random.default_rng(seed)

    for _ in tqdm(range(n_iterations), desc="Iterations"):
        iter_indices = []
        iter_split = {}

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
            n_pos1 = int(round(frac1_pos * n_pos))
            n_pos2 = n_pos - n_pos1

            if n_pos1 > len(pos1_idx):
                n_pos1 = len(pos1_idx)
                n_pos2 = min(n_pos - n_pos1, len(pos2_idx))
            elif n_pos2 > len(pos2_idx):
                n_pos2 = len(pos2_idx)
                n_pos1 = min(n_pos - n_pos2, len(pos1_idx))

            s_pos1 = rng.choice(pos1_idx, n_pos1, replace=False) if n_pos1 > 0 else np.array([])
            s_pos2 = rng.choice(pos2_idx, n_pos2, replace=False) if n_pos2 > 0 else np.array([])
            actual_pos = n_pos1 + n_pos2

            n_neg = actual_pos * neg_ratio
            n_neg1 = min(n_neg, len(neg1_idx))
            s_neg1 = rng.choice(neg1_idx, int(n_neg1), replace=False) if n_neg1 > 0 else np.array([])

            iter_indices.append(np.concatenate([s_pos1, s_pos2, s_neg1]))
            iter_split[allele] = {
                'd1_pos_pct': n_pos1 / actual_pos * 100 if actual_pos > 0 else np.nan,
                'd2_pos_pct': n_pos2 / actual_pos * 100 if actual_pos > 0 else np.nan,
                'd1_neg_count': n_neg1,
            }

        allele_split_iter.append(iter_split)
        all_iter_idx = np.concatenate(iter_indices)
        pos_in_array = np.array([idx_to_pos[i] for i in all_iter_idx])

        iter_labels = label_array[pos_in_array]
        iter_alleles = allele_array[pos_in_array]
        unique_alleles = np.unique(iter_alleles)

        for col in score_cols:
            iter_scores = score_arrays[col][pos_in_array]
            allele_aps = []
            for allele in unique_alleles:
                mask = iter_alleles == allele
                a_labels = iter_labels[mask]
                a_scores = iter_scores[mask]
                valid = ~np.isnan(a_scores)
                a_labels = a_labels[valid]
                a_scores = a_scores[valid]
                if len(np.unique(a_labels)) < 2:
                    continue
                ap = average_precision_score(a_labels, a_scores)
                allele_aps.append(ap)
            iteration_maps[col].append(np.mean(allele_aps) if allele_aps else np.nan)

    records = []
    for col in score_cols:
        vals = np.array(iteration_maps[col])
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            records.append({'score_col': col, 'mAP': np.nan, 'variance': np.nan,
                            'std': np.nan, 'ci95_low': np.nan, 'ci95_high': np.nan})
            continue
        ci_low, ci_high = stats.t.interval(0.95, df=len(vals)-1, loc=vals.mean(), scale=stats.sem(vals))
        records.append({
            'score_col': col, 'mAP': vals.mean(), 'variance': vals.var(),
            'std': vals.std(), 'ci95_low': ci_low, 'ci95_high': ci_high
        })
    map_df = pd.DataFrame(records).set_index('score_col')

    pval_matrix = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
    for col1, col2 in combinations(score_cols, 2):
        v1 = np.array(iteration_maps[col1])
        v2 = np.array(iteration_maps[col2])
        mask = ~np.isnan(v1) & ~np.isnan(v2)
        if mask.sum() < 2:
            pval_matrix.loc[col1, col2] = np.nan
            pval_matrix.loc[col2, col1] = np.nan
        else:
            _, pval = stats.ttest_rel(v1[mask], v2[mask])
            pval_matrix.loc[col1, col2] = pval
            pval_matrix.loc[col2, col1] = pval
    np.fill_diagonal(pval_matrix.values, 1.0)

    split_rows = []
    for allele in sorted(all_alleles):
        vals = {k: [] for k in ['d1_pos_pct', 'd2_pos_pct', 'd1_neg_count']}
        for it in allele_split_iter:
            if allele in it:
                for k in vals:
                    vals[k].append(it[allele][k])
        row = {'allele': allele}
        for k in ['d1_pos_pct', 'd2_pos_pct']:
            arr = np.array(vals[k])
            row[f'{k}_avg'] = np.nanmean(arr)
            row[f'{k}_min'] = np.nanmin(arr)
            row[f'{k}_max'] = np.nanmax(arr)
        arr = np.array(vals['d1_neg_count'])
        row['d1_neg_count_avg'] = np.nanmean(arr)
        row['d1_neg_count_min'] = np.nanmin(arr)
        row['d1_neg_count_max'] = np.nanmax(arr)
        split_rows.append(row)

    split_df = pd.DataFrame(split_rows).set_index('allele')

    return map_df, split_df, pval_matrix, nan_summary

def mixed_df_map_both_nan_agnostic_fast(df1, df2, score_cols, allele_pos_counts,label_col1='label', label_col2='label', allele_col='allele', neg_ratio=99, n_iterations=10, seed=42):

    d1 = df1.rename(columns={label_col1: '_label'}).assign(_source=1).reset_index(drop=True)
    d2 = df2.rename(columns={label_col2: '_label'}).assign(_source=2).reset_index(drop=True)
    d2.index = d2.index + len(d1)

    nan_summary = {}
    for col in score_cols:
        nan_rows = []
        nan_mask1 = d1[col].isna()
        nan_mask2 = d2[col].isna()
        if nan_mask1.any():
            nan_rows += list(zip(d1.loc[nan_mask1, 'seq'], d1.loc[nan_mask1, allele_col]))
        if nan_mask2.any():
            nan_rows += list(zip(d2.loc[nan_mask2, 'seq'], d2.loc[nan_mask2, allele_col]))
        if nan_rows:
            nan_summary[col] = nan_rows

    all_data = pd.concat([d1, d2])
    score_arrays = {col: all_data[col].values for col in score_cols}
    label_array = all_data['_label'].values
    allele_array = all_data[allele_col].values
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_data.index)}

    def build_index(d):
        groups = {}
        for allele, grp in d.groupby(allele_col):
            pos_idx = grp.index[grp['_label'] == 1].values
            neg_idx = grp.index[grp['_label'] == 0].values
            groups[allele] = {'pos': pos_idx, 'neg': neg_idx}
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

    iteration_maps = {col: [] for col in score_cols}
    allele_split_iter = []
    rng = np.random.default_rng(seed)

    for _ in tqdm(range(n_iterations), desc="Iterations"):
        iter_indices = []
        iter_split = {}

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
            n_pos1 = int(round(frac1_pos * n_pos))
            n_pos2 = n_pos - n_pos1

            if n_pos1 > len(pos1_idx):
                n_pos1 = len(pos1_idx)
                n_pos2 = min(n_pos - n_pos1, len(pos2_idx))
            elif n_pos2 > len(pos2_idx):
                n_pos2 = len(pos2_idx)
                n_pos1 = min(n_pos - n_pos2, len(pos1_idx))

            s_pos1 = rng.choice(pos1_idx, n_pos1, replace=False) if n_pos1 > 0 else np.array([])
            s_pos2 = rng.choice(pos2_idx, n_pos2, replace=False) if n_pos2 > 0 else np.array([])
            actual_pos = n_pos1 + n_pos2

            n_neg_total = actual_pos * neg_ratio
            total_neg_avail = len(neg1_idx) + len(neg2_idx)
            n_neg = min(n_neg_total, total_neg_avail)

            frac1_neg = rng.uniform(0, 1)
            n_neg1 = int(round(frac1_neg * n_neg))
            n_neg2 = n_neg - n_neg1

            if n_neg1 > len(neg1_idx):
                n_neg1 = len(neg1_idx)
                n_neg2 = min(n_neg - n_neg1, len(neg2_idx))
            elif n_neg2 > len(neg2_idx):
                n_neg2 = len(neg2_idx)
                n_neg1 = min(n_neg - n_neg2, len(neg1_idx))

            s_neg1 = rng.choice(neg1_idx, int(n_neg1), replace=False) if n_neg1 > 0 else np.array([])
            s_neg2 = rng.choice(neg2_idx, int(n_neg2), replace=False) if n_neg2 > 0 else np.array([])

            iter_indices.append(np.concatenate([s_pos1, s_pos2, s_neg1, s_neg2]))
            iter_split[allele] = {
                'd1_pos_pct': n_pos1 / actual_pos * 100 if actual_pos > 0 else np.nan,
                'd2_pos_pct': n_pos2 / actual_pos * 100 if actual_pos > 0 else np.nan,
                'd1_neg_count': n_neg1,
                'd2_neg_count': n_neg2,
            }

        allele_split_iter.append(iter_split)
        all_iter_idx = np.concatenate(iter_indices)
        pos_in_array = np.array([idx_to_pos[i] for i in all_iter_idx])

        iter_labels = label_array[pos_in_array]
        iter_alleles = allele_array[pos_in_array]
        unique_alleles = np.unique(iter_alleles)

        for col in score_cols:
            iter_scores = score_arrays[col][pos_in_array]
            allele_aps = []
            for allele in unique_alleles:
                mask = iter_alleles == allele
                a_labels = iter_labels[mask]
                a_scores = iter_scores[mask]
                valid = ~np.isnan(a_scores)
                a_labels = a_labels[valid]
                a_scores = a_scores[valid]
                if len(np.unique(a_labels)) < 2:
                    continue
                ap = average_precision_score(a_labels, a_scores)
                allele_aps.append(ap)
            iteration_maps[col].append(np.mean(allele_aps) if allele_aps else np.nan)

    records = []
    for col in score_cols:
        vals = np.array(iteration_maps[col])
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            records.append({'score_col': col, 'mAP': np.nan, 'variance': np.nan,
                            'std': np.nan, 'ci95_low': np.nan, 'ci95_high': np.nan})
            continue
        ci_low, ci_high = stats.t.interval(0.95, df=len(vals)-1, loc=vals.mean(), scale=stats.sem(vals))
        records.append({
            'score_col': col, 'mAP': vals.mean(), 'variance': vals.var(),
            'std': vals.std(), 'ci95_low': ci_low, 'ci95_high': ci_high
        })
    map_df = pd.DataFrame(records).set_index('score_col')

    pval_matrix = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
    for col1, col2 in combinations(score_cols, 2):
        v1 = np.array(iteration_maps[col1])
        v2 = np.array(iteration_maps[col2])
        mask = ~np.isnan(v1) & ~np.isnan(v2)
        if mask.sum() < 2:
            pval_matrix.loc[col1, col2] = np.nan
            pval_matrix.loc[col2, col1] = np.nan
        else:
            _, pval = stats.ttest_rel(v1[mask], v2[mask])
            pval_matrix.loc[col1, col2] = pval
            pval_matrix.loc[col2, col1] = pval
    np.fill_diagonal(pval_matrix.values, 1.0)

    split_rows = []
    for allele in sorted(all_alleles):
        vals = {k: [] for k in ['d1_pos_pct', 'd2_pos_pct', 'd1_neg_count', 'd2_neg_count']}
        for it in allele_split_iter:
            if allele in it:
                for k in vals:
                    vals[k].append(it[allele][k])
        row = {'allele': allele}
        for k in ['d1_pos_pct', 'd2_pos_pct']:
            arr = np.array(vals[k])
            row[f'{k}_avg'] = np.nanmean(arr)
            row[f'{k}_min'] = np.nanmin(arr)
            row[f'{k}_max'] = np.nanmax(arr)
        for k in ['d1_neg_count', 'd2_neg_count']:
            arr = np.array(vals[k])
            row[f'{k}_avg'] = np.nanmean(arr)
            row[f'{k}_min'] = np.nanmin(arr)
            row[f'{k}_max'] = np.nanmax(arr)
        split_rows.append(row)

    split_df = pd.DataFrame(split_rows).set_index('allele')

    return map_df, split_df, pval_matrix, nan_summary

def iterative_neg_subsample_map_nan_agnostic(df, score_cols, label_col='label', allele_col='allele', neg_ratio=99, n_iterations=10, seed=42):

    rng = np.random.default_rng(seed)

    missing_cols = [col for col in score_cols if col not in df.columns]
    if missing_cols:
        print(f"Excluding score cols not in df: {missing_cols}")
    score_cols = [col for col in score_cols if col in df.columns]

    nan_cols = [col for col in score_cols if df[col].isna().any()]
    if nan_cols:
        print(f"Excluding score cols with NaNs: {nan_cols}")
    score_cols = [col for col in score_cols if col not in nan_cols]

    groups = {allele: grp for allele, grp in df.groupby(allele_col)}
    iteration_maps = {col: [] for col in score_cols}

    for _ in tqdm(range(n_iterations), desc="Iterations"):
        iter_rows = []

        for allele, group in groups.items():
            pos = group[group[label_col] == 1]
            neg = group[group[label_col] == 0]
            n_neg = min(len(pos) * neg_ratio, len(neg))
            sampled_neg = neg.iloc[rng.choice(len(neg), n_neg, replace=False)] if n_neg > 0 else pd.DataFrame()
            iter_rows.append(pd.concat([pos, sampled_neg]))

        sampled = pd.concat(iter_rows, ignore_index=True)

        for col in score_cols:
            allele_aps = []
            for allele, group in sampled.groupby(allele_col):
                if group[label_col].nunique() < 2:
                    continue
                ap = average_precision_score(group[label_col], group[col])
                allele_aps.append(ap)
            iteration_maps[col].append(np.mean(allele_aps))

    records = []
    for col in score_cols:
        vals = np.array(iteration_maps[col])
        ci_low, ci_high = stats.t.interval(0.95, df=len(vals)-1, loc=vals.mean(), scale=stats.sem(vals))
        records.append({
            'score_col': col, 'mAP': vals.mean(), 'variance': vals.var(),
            'std': vals.std(), 'ci95_low': ci_low, 'ci95_high': ci_high
        })
    map_df = pd.DataFrame(records).set_index('score_col')

    pval_matrix = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
    for col1, col2 in combinations(score_cols, 2):
        _, pval = stats.ttest_rel(iteration_maps[col1], iteration_maps[col2])
        pval_matrix.loc[col1, col2] = pval
        pval_matrix.loc[col2, col1] = pval
    np.fill_diagonal(pval_matrix.values, 1.0)

    return map_df, pval_matrix


def compute_score_differences_allele(df, base_scores, list2_scores, list3_scores):
    alleles = df.index.tolist()
    
    base_diff_per_allele = {allele: [] for allele in alleles}
    list2_diff_per_allele = {allele: [] for allele in alleles}
    list3_diff_per_allele = {allele: [] for allele in alleles}

    for i in range(len(base_scores) - 1):
        curr_base = base_scores[i]
        next_base = base_scores[i + 1]
        curr_list2 = list2_scores[i]
        curr_list3 = list3_scores[i]

        for allele in alleles:
            base_diff_per_allele[allele].append(df.loc[allele, next_base] - df.loc[allele, curr_base])
            list2_diff_per_allele[allele].append(df.loc[allele, curr_list2] - df.loc[allele, curr_base])
            list3_diff_per_allele[allele].append(df.loc[allele, curr_list3] - df.loc[allele, curr_base])

    records = []
    for allele in alleles:
        records.append({
            'allele': allele,
            'base_diff': np.nanmean(base_diff_per_allele[allele]),
            'list2_diff': np.nanmean(list2_diff_per_allele[allele]),
            'list3_diff': np.nanmean(list3_diff_per_allele[allele]),
        })

    return pd.DataFrame(records).set_index('allele')

def iterative_neg_subsample_map_per_allele(df, score_cols, label_col='label', allele_col='allele', neg_ratio=99, n_iterations=10, seed=42):

    rng = np.random.default_rng(seed)
    groups = {allele: grp for allele, grp in df.groupby(allele_col)}
    iteration_maps = {allele: {col: [] for col in score_cols} for allele in groups}

    for _ in tqdm(range(n_iterations), desc="Iterations"):
        for allele, group in groups.items():
            pos = group[group[label_col] == 1]
            neg = group[group[label_col] == 0]

            n_pos = len(pos)
            n_neg = len(neg)

            if n_pos == 0 or n_neg == 0:
                continue

            n_neg_target = n_pos * neg_ratio

            if n_neg >= n_neg_target:
                sampled_pos = pos
                sampled_neg = neg.iloc[rng.choice(n_neg, n_neg_target, replace=False)]
            else:
                n_pos_target = max(1, n_neg // neg_ratio)
                sampled_pos = pos.iloc[rng.choice(n_pos, min(n_pos_target, n_pos), replace=False)]
                sampled_neg = neg

            sampled = pd.concat([sampled_pos, sampled_neg])

            if sampled[label_col].nunique() < 2:
                continue

            for col in score_cols:
                ap = average_precision_score(sampled[label_col], sampled[col])
                iteration_maps[allele][col].append(ap)

    records = []
    for allele in sorted(groups.keys()):
        row = {'allele': allele}
        for col in score_cols:
            vals = iteration_maps[allele][col]
            row[col] = np.mean(vals) if vals else np.nan
        records.append(row)

    return pd.DataFrame(records).set_index('allele')

def mean_ap_by_allele_resampled_v2(
    df: pd.DataFrame,
    score_cols: list[str],
    *,
    allele_col: str = "allele",
    label_col: str = "label",
    n_pos: int = 200,
    pos_to_neg_ratio: int = 99,
    n_iter: int = 100,
    seed: int = 0,
    use_max_possible_pos: bool = False,
) -> pd.DataFrame:
    """
    Compute mean AP per allele per score column by repeated fixed-ratio resampling.
    """
    needed_cols = [allele_col, label_col] + list(score_cols)
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if n_pos <= 0:
        raise ValueError("n_pos must be > 0")
    if pos_to_neg_ratio <= 0:
        raise ValueError("pos_to_neg_ratio must be > 0")
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")

    rng = np.random.default_rng(seed)

    tmp = df[needed_cols].copy()
    tmp = tmp.dropna(subset=[allele_col, label_col])
    tmp[label_col] = tmp[label_col].astype(int)

    valid_score_cols = [c for c in score_cols if tmp[c].notna().any()]
    if len(valid_score_cols) == 0:
        raise ValueError("None of the score columns contain non-missing values.")

    allele_data = {}
    for allele, sub in tmp.groupby(allele_col, sort=True):
        pos = sub.loc[sub[label_col] == 1, valid_score_cols].to_numpy(dtype=float, copy=True)
        neg = sub.loc[sub[label_col] == 0, valid_score_cols].to_numpy(dtype=float, copy=True)

        allele_data[allele] = {
            "pos": pos,
            "neg": neg,
            "n_pos_total": pos.shape[0],
            "n_neg_total": neg.shape[0],
        }

    alleles = sorted(allele_data.keys())
    out = np.full((len(alleles), len(valid_score_cols)), np.nan, dtype=float)

    for a_idx, allele in enumerate(
        tqdm(alleles, desc="Alleles", leave=True)
    ):
        pos_scores = allele_data[allele]["pos"]  # (P, n_scores)
        neg_scores = allele_data[allele]["neg"]  # (N, n_scores)

        P = pos_scores.shape[0]
        N = neg_scores.shape[0]

        if use_max_possible_pos:
            n_pos_eff = min(P, N // pos_to_neg_ratio)
        else:
            n_pos_eff = n_pos

        n_neg_eff = n_pos_eff * pos_to_neg_ratio

        if n_pos_eff <= 0 or P < n_pos_eff or N < n_neg_eff:
            continue

        ap_sum = np.zeros(len(valid_score_cols), dtype=float)
        ap_count = np.zeros(len(valid_score_cols), dtype=int)

        y_true = np.concatenate([
            np.ones(n_pos_eff, dtype=np.int8),
            np.zeros(n_neg_eff, dtype=np.int8),
        ])

        for _ in range(n_iter):
            pos_idx = rng.choice(P, size=n_pos_eff, replace=False)
            neg_idx = rng.choice(N, size=n_neg_eff, replace=False)

            pos_sample = pos_scores[pos_idx]   # (n_pos_eff, n_scores)
            neg_sample = neg_scores[neg_idx]   # (n_neg_eff, n_scores)
            y_score = np.vstack([pos_sample, neg_sample])  # (n_pos_eff+n_neg_eff, n_scores)

            # AP per score column
            for s_idx in range(y_score.shape[1]):
                col_scores = y_score[:, s_idx]
                if np.isnan(col_scores).any():
                    continue
                ap_sum[s_idx] += average_precision_score(y_true, col_scores)
                ap_count[s_idx] += 1

        with np.errstate(invalid="ignore", divide="ignore"):
            out[a_idx, :] = ap_sum / ap_count

    result = pd.DataFrame(out, index=alleles, columns=valid_score_cols)
    result.index.name = allele_col
    return result

def compute_net_adaptation_benefit(
    ms_df: pd.DataFrame,
    holec_df: pd.DataFrame,
    base_col: str,
    score_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Compute per-allele net adaptation benefit:
    
    net_benefit = Holec_gain - MS_loss = (Holec_score - Holec_base) - max(0, MS_base - MS_score)
    """
    required_ms = {base_col, score_col}
    required_holec = {base_col, score_col}

    missing_ms = required_ms - set(ms_df.columns)
    missing_holec = required_holec - set(holec_df.columns)

    if missing_ms:
        raise ValueError(f"ms_df is missing required columns: {sorted(missing_ms)}")
    if missing_holec:
        raise ValueError(f"holec_df is missing required columns: {sorted(missing_holec)}")

    common_alleles = ms_df.index.intersection(holec_df.index)
    if len(common_alleles) == 0:
        raise ValueError("No overlapping allele indices found between ms_df and holec_df.")

    ms_sub = ms_df.loc[common_alleles, [base_col, score_col]].copy()
    holec_sub = holec_df.loc[common_alleles, [base_col, score_col]].copy()
    ms_sub = ms_sub.apply(pd.to_numeric, errors="coerce")
    holec_sub = holec_sub.apply(pd.to_numeric, errors="coerce")

    # yd gain relative to base
    holec_gain = holec_sub[score_col] - holec_sub[base_col]

    # ms loss relative to base (only penalize decreases; gains count as 0 loss)
    ms_loss = (ms_sub[base_col] - ms_sub[score_col]).clip(lower=0)

    net_benefit = holec_gain - ms_loss

    out_df = pd.DataFrame({output_col: net_benefit})
    out_df.index.name = ms_df.index.name

    return out_df

def compute_cluster_vs_prior_from_matrix(ap_matrix, cluster_id, model_dict):
    records = []
    other_clusters = [c for c in ap_matrix.index if c != cluster_id]

    for short_name, col_name in model_dict.items():
        if col_name not in ap_matrix.columns:
            records.append({'model': short_name, 'cluster': np.nan, 'prior': np.nan})
            continue
        records.append({
            'model': short_name,
            'cluster': ap_matrix.loc[cluster_id, col_name],
            'prior': ap_matrix.loc[other_clusters, col_name].mean() if other_clusters else np.nan,
        })

    return pd.DataFrame(records).set_index('model')

def _fast_average_precision_multi(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("scores must be 2D of shape (N, C)")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D of shape (N,)")

    n_pos = int(labels.sum())
    c = scores.shape[1]
    out = np.full(c, np.nan, dtype=np.float64)

    if n_pos == 0:
        return out

    labels_u8 = labels.astype(np.uint8, copy=False)

    for j in range(c):
        order = np.argsort(-scores[:, j], kind="mergesort")
        y = labels_u8[order]
        pos_idx = np.flatnonzero(y)
        if pos_idx.size == 0:
            continue
        tp = np.cumsum(y, dtype=np.int64)
        precision_at_pos = tp[pos_idx] / (pos_idx + 1.0)
        out[j] = precision_at_pos.mean()

    return out


def _validate_required_columns(df: pd.DataFrame, required: List[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _build_source_arrays_by_allele(
    df: pd.DataFrame,
    *,
    allele_col: str,
    label_col: str,
    score_cols: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    use_cols = [allele_col, label_col] + score_cols
    d = df[use_cols].copy()
    d[allele_col] = d[allele_col].astype(str)
    d[label_col] = pd.to_numeric(d[label_col], errors="coerce").astype(np.int8)

    for sc in score_cols:
        d[sc] = pd.to_numeric(d[sc], errors="coerce").astype(np.float32)

    for allele, sub in d.groupby(allele_col, sort=False):
        pos = sub.loc[sub[label_col] == 1, score_cols].to_numpy(dtype=np.float32, copy=True)
        neg = sub.loc[sub[label_col] == 0, score_cols].to_numpy(dtype=np.float32, copy=True)
        out[str(allele)] = {"pos": pos, "neg": neg}

    return out

def compute_mixed_ms_holec_map_bootstrap_delta_from_base(
    ms_df: pd.DataFrame,
    holec_df: pd.DataFrame,
    *,
    base_col: str,
    score_cols: List[str],
    allele_col: str = "allele",
    ms_label_col: str = "label",
    holec_label_col: str = "label",
    neg_per_pos_ratio: int = 99,
    mix_step_pct: float = 1.0,
    n_iterations: int = 100,
    ci_alpha: float = 0.95,
    random_state: int = 42,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Smoothly mix MS and Holec test sets and compute bootstrapped delta-mAP per mix,
    where each score column is compared as:

    delta = mAP(score_col) - mAP(base_col)
    """
    score_cols = [c for c in score_cols if c != base_col]
    if len(score_cols) == 0:
        raise ValueError("After excluding base_col, score_cols is empty.")

    all_score_cols = [base_col] + score_cols

    if neg_per_pos_ratio < 0:
        raise ValueError("neg_per_pos_ratio must be >= 0")
    if mix_step_pct <= 0 or mix_step_pct > 100:
        raise ValueError("mix_step_pct must be in (0, 100]")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be > 0")
    if not (0 < ci_alpha < 1):
        raise ValueError("ci_alpha must be in (0,1)")

    _validate_required_columns(ms_df, [allele_col, ms_label_col] + all_score_cols, "ms_df")
    _validate_required_columns(holec_df, [allele_col, holec_label_col] + all_score_cols, "holec_df")

    rng = np.random.default_rng(random_state)

    ms_map = _build_source_arrays_by_allele(
        ms_df,
        allele_col=allele_col,
        label_col=ms_label_col,
        score_cols=all_score_cols,
    )
    holec_map = _build_source_arrays_by_allele(
        holec_df,
        allele_col=allele_col,
        label_col=holec_label_col,
        score_cols=all_score_cols,
    )

    common_alleles = sorted(set(ms_map.keys()) & set(holec_map.keys()))
    if len(common_alleles) == 0:
        raise ValueError("No overlapping alleles between ms_df and holec_df")

    allele_plan = {}
    skipped = {}

    for allele in common_alleles:
        ms_pos = ms_map[allele]["pos"]
        ms_neg = ms_map[allele]["neg"]
        holec_pos = holec_map[allele]["pos"]
        holec_neg = holec_map[allele]["neg"]

        ms_pos_n = ms_pos.shape[0]
        holec_pos_n = holec_pos.shape[0]
        min_pos = min(ms_pos_n, holec_pos_n)

        if min_pos < 1:
            skipped[allele] = "min_pos_lt_1"
            continue

        total_neg = min_pos * neg_per_pos_ratio

        if ms_neg.shape[0] < total_neg:
            skipped[allele] = f"ms_neg_insufficient_for_ratio_{neg_per_pos_ratio}"
            continue
        if holec_neg.shape[0] < total_neg:
            skipped[allele] = f"holec_neg_insufficient_for_ratio_{neg_per_pos_ratio}"
            continue

        allele_plan[allele] = {
            "min_pos": int(min_pos),
            "total_neg": int(total_neg),
            "ms_pos": ms_pos,
            "ms_neg": ms_neg,
            "holec_pos": holec_pos,
            "holec_neg": holec_neg,
        }

    usable_alleles = sorted(allele_plan.keys())
    if len(usable_alleles) == 0:
        raise ValueError("No usable alleles after applying min positive / negative availability constraints")

    ms_percentages = np.arange(100.0, -1e-9, -float(mix_step_pct))
    ms_percentages = np.round(ms_percentages, 10)
    if ms_percentages[-1] != 0.0:
        ms_percentages = np.append(ms_percentages, 0.0)

    n_steps = len(ms_percentages)
    n_total_scores = len(all_score_cols)
    n_delta_scores = len(score_cols)

    bootstrap_deltas = np.full((n_steps, n_iterations, n_delta_scores), np.nan, dtype=np.float64)

    outer_bar = tqdm(
        enumerate(ms_percentages),
        total=n_steps,
        desc="Mixing steps",
        leave=True,
        disable=not show_progress,
    )

    for step_idx, ms_pct in outer_bar:
        ms_frac = ms_pct / 100.0

        iter_bar = tqdm(
            range(n_iterations),
            desc=f"Bootstrap @ {ms_pct:.2f}% MS",
            leave=True,
            disable=not show_progress,
        )

        for it in iter_bar:
            ap_by_allele = np.full((len(usable_alleles), n_total_scores), np.nan, dtype=np.float64)

            for a_idx, allele in enumerate(usable_alleles):
                plan = allele_plan[allele]

                min_pos = plan["min_pos"]
                total_neg = plan["total_neg"]

                ms_pos_take = int(np.rint(min_pos * ms_frac))
                ms_pos_take = min(max(ms_pos_take, 0), min_pos)
                holec_pos_take = min_pos - ms_pos_take

                ms_neg_take = int(np.rint(total_neg * ms_frac))
                ms_neg_take = min(max(ms_neg_take, 0), total_neg)
                holec_neg_take = total_neg - ms_neg_take

                parts = []
                labels_parts = []

                if ms_pos_take > 0:
                    arr = plan["ms_pos"]
                    idx = rng.choice(arr.shape[0], size=ms_pos_take, replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.ones(ms_pos_take, dtype=np.uint8))

                if holec_pos_take > 0:
                    arr = plan["holec_pos"]
                    idx = rng.choice(arr.shape[0], size=holec_pos_take, replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.ones(holec_pos_take, dtype=np.uint8))

                if ms_neg_take > 0:
                    arr = plan["ms_neg"]
                    idx = rng.choice(arr.shape[0], size=ms_neg_take, replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.zeros(ms_neg_take, dtype=np.uint8))

                if holec_neg_take > 0:
                    arr = plan["holec_neg"]
                    idx = rng.choice(arr.shape[0], size=holec_neg_take, replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.zeros(holec_neg_take, dtype=np.uint8))

                if len(parts) == 0:
                    continue

                scores_mix = np.vstack(parts) # (N_mix, C_total)
                labels_mix = np.concatenate(labels_parts)  # (N_mix,)

                ap_by_allele[a_idx] = _fast_average_precision_multi(scores_mix, labels_mix)

            map_iter = np.nanmean(ap_by_allele, axis=0) # shape (n_total_scores,)

            base_map = map_iter[0]
            deltas_iter = map_iter[1:] - base_map # paired deltas vs base
            bootstrap_deltas[step_idx, it] = deltas_iter

    mean_arr = np.nanmean(bootstrap_deltas, axis=1)
    var_arr = np.nanvar(bootstrap_deltas, axis=1)

    alpha_lo = (1.0 - ci_alpha) / 2.0 * 100.0
    alpha_hi = (1.0 + ci_alpha) / 2.0 * 100.0
    ci_low_arr = np.nanpercentile(bootstrap_deltas, alpha_lo, axis=1)
    ci_high_arr = np.nanpercentile(bootstrap_deltas, alpha_hi, axis=1)

    mean_df = pd.DataFrame(mean_arr, index=ms_percentages, columns=score_cols)
    mean_df.index.name = "ms_percentage"

    var_df = pd.DataFrame(var_arr, index=ms_percentages, columns=score_cols)
    var_df.index.name = "ms_percentage"

    ci_cols = {}
    for j, sc in enumerate(score_cols):
        ci_cols[f"{sc}_ci_low"] = ci_low_arr[:, j]
        ci_cols[f"{sc}_ci_high"] = ci_high_arr[:, j]

    ci_df = pd.DataFrame(ci_cols, index=ms_percentages)
    ci_df.index.name = "ms_percentage"

    return mean_df, var_df, ci_df

"""
Ranking and recovery
"""
RankMode = Literal["global", "allele", "allele_len"]

def add_background_percentile_ranks(
    test_df: pd.DataFrame,
    background_df: pd.DataFrame,
    allele_col: str = "allele",
    seq_len_col: str = "seq_len",
    score_cols: Optional[Sequence[str]] = None,
    rank_mode: RankMode = "global",
    suffix: str = "_rank",
    reverse: bool = True,
    copy: bool = True,
    dtype: str = "float32",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add percentile-rank columns to `test_df` by placing each test peptide score into
    the score distribution of `background_df`.
    """
    if copy:
        out_df = test_df.copy()
    else:
        out_df = test_df

    if score_cols is None:
        excluded = {allele_col, seq_len_col}
        score_cols = [
            c for c in test_df.columns
            if c in background_df.columns and c not in excluded
        ]
    else:
        score_cols = [c for c in score_cols if c in test_df.columns and c in background_df.columns]

    if len(score_cols) == 0:
        raise ValueError("No valid score columns found in both test_df and background_df.")

    bg_gid, test_gid = _build_group_ids(
        background_df=background_df,
        test_df=test_df,
        allele_col=allele_col,
        seq_len_col=seq_len_col,
        rank_mode=rank_mode,
    )

    n_test = len(test_df)
    n_bg = len(background_df)

    bg_order, bg_unique_gids, bg_starts, bg_counts = _group_structure(bg_gid)
    test_order, test_unique_gids, test_starts, test_counts = _group_structure(test_gid)

    bg_gid_to_pos = {gid: i for i, gid in enumerate(bg_unique_gids)}

    iterator = tqdm(score_cols, leave=True, disable=not show_progress, desc="Ranking score columns")

    for score_col in iterator:
        bg_vals = pd.to_numeric(background_df[score_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        test_vals = pd.to_numeric(test_df[score_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        bg_vals_ord = bg_vals[bg_order]
        test_vals_ord = test_vals[test_order]

        out_ord = np.full(n_test, np.nan, dtype=np.float64)

        if rank_mode == "global":
            valid_bg = np.isfinite(bg_vals_ord)
            bg_sorted = np.sort(bg_vals_ord[valid_bg])

            valid_test = np.isfinite(test_vals_ord)
            if bg_sorted.size > 0 and valid_test.any():
                if reverse:
                    pct = 100.0 * np.searchsorted(bg_sorted, test_vals_ord[valid_test], side="right") / bg_sorted.size
                else:
                    pct = 100.0 * (bg_sorted.size - np.searchsorted(bg_sorted, test_vals_ord[valid_test], side="left")) / bg_sorted.size
                out_ord[valid_test] = pct

        else:
            for gid, t_start, t_count in zip(test_unique_gids, test_starts, test_counts):
                if gid < 0:
                    continue

                bg_pos = bg_gid_to_pos.get(gid, None)
                if bg_pos is None:
                    continue

                b_start = bg_starts[bg_pos]
                b_count = bg_counts[bg_pos]

                bg_slice = bg_vals_ord[b_start:b_start + b_count]
                test_slice = test_vals_ord[t_start:t_start + t_count]

                valid_bg = np.isfinite(bg_slice)
                if not valid_bg.any():
                    continue

                bg_sorted = np.sort(bg_slice[valid_bg])

                valid_test = np.isfinite(test_slice)
                if not valid_test.any():
                    continue

                if reverse:
                    pct = 100.0 * np.searchsorted(bg_sorted, test_slice[valid_test], side="right") / bg_sorted.size
                else:
                    pct = 100.0 * (bg_sorted.size - np.searchsorted(bg_sorted, test_slice[valid_test], side="left")) / bg_sorted.size

                tmp = np.full(t_count, np.nan, dtype=np.float64)
                tmp[valid_test] = pct
                out_ord[t_start:t_start + t_count] = tmp

        out = np.empty(n_test, dtype=np.float64)
        out[test_order] = out_ord

        out_df[f"{score_col}{suffix}"] = out.astype(dtype, copy=False)

    return out_df


def _group_structure(group_ids: np.ndarray):
    """
    Return sorted order and contiguous group metadata.
    """
    order = np.argsort(group_ids, kind="mergesort")
    gids_ord = group_ids[order]

    if gids_ord.size == 0:
        return order, np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    change = np.empty(gids_ord.size, dtype=bool)
    change[0] = True
    change[1:] = gids_ord[1:] != gids_ord[:-1]

    starts = np.flatnonzero(change)
    unique_gids = gids_ord[starts]
    counts = np.diff(np.append(starts, gids_ord.size))

    return order, unique_gids, starts, counts


def _build_group_ids(
    background_df: pd.DataFrame,
    test_df: pd.DataFrame,
    allele_col: str,
    seq_len_col: str,
    rank_mode: RankMode,
):
    """
    Build integer group IDs for background and test in a way that test groups absent
    from background become -1.
    """
    n_bg = len(background_df)
    n_test = len(test_df)

    if rank_mode == "global":
        bg_gid = np.zeros(n_bg, dtype=np.int64)
        test_gid = np.zeros(n_test, dtype=np.int64)
        return bg_gid, test_gid

    bg_alleles = pd.Index(pd.unique(background_df[allele_col]))
    bg_allele_codes = pd.Categorical(background_df[allele_col], categories=bg_alleles).codes.astype(np.int64, copy=False)
    test_allele_codes = pd.Categorical(test_df[allele_col], categories=bg_alleles).codes.astype(np.int64, copy=False)

    if rank_mode == "allele":
        return bg_allele_codes, test_allele_codes

    if rank_mode != "allele_len":
        raise ValueError("rank_mode must be one of {'global', 'allele', 'allele_len'}")

    bg_len = pd.to_numeric(background_df[seq_len_col], errors="coerce").to_numpy(np.int64, copy=False)
    test_len = pd.to_numeric(test_df[seq_len_col], errors="coerce").to_numpy(np.int64, copy=False)

    min_len = min(bg_len.min(initial=0), test_len.min(initial=0))
    if min_len < 0:
        bg_len = bg_len - min_len
        test_len = test_len - min_len

    max_len = int(max(bg_len.max(initial=0), test_len.max(initial=0)))
    stride = max_len + 1

    bg_comp = bg_allele_codes * stride + bg_len

    valid_test = test_allele_codes >= 0
    test_comp = np.full(n_test, -1, dtype=np.int64)
    test_comp[valid_test] = test_allele_codes[valid_test] * stride + test_len[valid_test]

    unique_bg_comp, bg_gid = np.unique(bg_comp, return_inverse=True)
    test_gid = np.full(n_test, -1, dtype=np.int64)

    valid_test2 = test_comp >= 0
    if valid_test2.any():
        mapped = np.searchsorted(unique_bg_comp, test_comp[valid_test2])
        in_range = mapped < unique_bg_comp.size
        found = np.zeros(valid_test2.sum(), dtype=bool)
        found[in_range] = unique_bg_comp[mapped[in_range]] == test_comp[valid_test2][in_range]
        test_gid_valid = np.full(valid_test2.sum(), -1, dtype=np.int64)
        test_gid_valid[found] = mapped[found]
        test_gid[valid_test2] = test_gid_valid

    return bg_gid, test_gid

def keep_max_score_per_id(df, score_col, id_col):
    idx = df.groupby(id_col)[score_col].idxmax()
    return df.loc[idx].reset_index(drop=True)

def pct_above_threshold_per_subject(df, study_col, subject_col, score_col, threshold, direction='>='):
    ops = {
        '>=': lambda x: x >= threshold,
        '<=': lambda x: x <= threshold,
        '>':  lambda x: x > threshold,
        '<':  lambda x: x < threshold,
        '==': lambda x: x == threshold,
    }
    op = ops[direction]
    result = {}
    for study, study_df in df.groupby(study_col):
        records = []
        for subject, subj_df in study_df.groupby(subject_col):
            pct = op(subj_df[score_col]).mean() * 100
            records.append({subject_col: subject, score_col: pct})
        result[study] = pd.DataFrame(records).set_index(subject_col)
    return result

def pct_above_threshold_per_subject_single(df, subject_col, score_col, threshold, direction='>='):
    ops = {
        '>=': lambda x: x >= threshold,
        '<=': lambda x: x <= threshold,
        '>':  lambda x: x > threshold,
        '<':  lambda x: x < threshold,
        '==': lambda x: x == threshold,
    }
    op = ops[direction]
    records = []
    for subject, subj_df in df.groupby(subject_col):
        pct = op(subj_df[score_col]).mean() * 100
        records.append({subject_col: subject, score_col: pct})
    return pd.DataFrame(records).set_index(subject_col)

def reorganize_performance_df(df, allele):
    df = df.copy()
    df.index = df.index.astype(int)

    records = []
    cluster_ids = df.index.tolist()

    for lid in cluster_ids:
        base_col = f"mhcprime_holdout_{allele.lower()}_{str(lid)}"
        ft_col = f"model_ft_{allele.lower()}_{str(lid)}"
        pepcl_col = f"model_pepcl_{allele.lower()}_{str(lid)}_lambda_1eN4"
        full_col = f"mhcprime_holdout_{allele.lower()}_full"
        null_col = f"mhcprime_holdout_{allele.lower()}_null"

        other_clusters = [c for c in cluster_ids if c != lid]
        row = {'cluster': lid}

        for col, prefix in [(base_col, 'mhcprime'), (ft_col, 'model_ft'), (pepcl_col, 'model_pepcl')]:
            if col not in df.columns:
                row[f'{prefix}_cluster'] = np.nan
                row[f'{prefix}_prior'] = np.nan
                continue
            row[f'{prefix}_cluster'] = df.loc[lid, col]
            row[f'{prefix}_prior'] = df.loc[other_clusters, col].mean() if other_clusters else np.nan

        for col, prefix in [(full_col, 'full'), (null_col, 'null')]:
            if col not in df.columns:
                row[f'{prefix}_cluster'] = np.nan
                row[f'{prefix}_prior'] = np.nan
                continue
            row[f'{prefix}_cluster'] = df.loc[lid, col]
            row[f'{prefix}_prior'] = df.loc[other_clusters, col].mean() if other_clusters else np.nan

        records.append(row)

    return pd.DataFrame(records).set_index('cluster')

def compute_cluster_prior_ap(df, score_cols, allele, cluster_col='louvain_id', label_col='label'):
    score_cols = [c for c in score_cols if c in df.columns]

    df = df.copy()
    df[cluster_col] = df[cluster_col].astype(int)
    cluster_ids = sorted(df[cluster_col].unique())

    y_all = df[label_col].to_numpy()
    score_arrays = {c: df[c].to_numpy() for c in score_cols}
    cluster_idx = {lid: df.index.get_indexer(df[df[cluster_col] == lid].index) for lid in cluster_ids}

    records = []

    for lid in tqdm(cluster_ids, desc="Clusters"):
        base_col = f"mhcprime_holdout_{allele.lower()}_{lid}"
        ft_col = f"model_ft_{allele.lower()}_{lid}"
        pepcl_col = f"model_pepcl_{allele.lower()}_{lid}_lambda_1eN4"
        full_col = f"mhcprime_holdout_{allele.lower()}_full"
        null_col = f"mhcprime_holdout_{allele.lower()}_null"

        curr_idx = cluster_idx[lid]
        other_idx = np.concatenate([cluster_idx[c] for c in cluster_ids if c != lid])

        row = {'cluster': lid}

        def compute_ap(idx, col):
            if col not in score_arrays:
                return np.nan
            y = y_all[idx]
            s = score_arrays[col][idx]
            valid = ~pd.isna(y) & ~pd.isna(s)
            if valid.sum() <= 1:
                return np.nan
            y_true = y[valid]
            if len(np.unique(y_true)) < 2:
                return np.nan
            try:
                return average_precision_score(y_true, s[valid])
            except Exception:
                return np.nan

        for col, prefix in [(base_col, 'mhcprime'), (ft_col, 'model_ft'), (pepcl_col, 'model_pepcl')]:
            row[f'{prefix}_cluster'] = compute_ap(curr_idx, col)
            row[f'{prefix}_prior'] = compute_ap(other_idx, col)

        for col, prefix in [(full_col, 'full'), (null_col, 'null')]:
            row[f'{prefix}_cluster'] = compute_ap(curr_idx, col)
            row[f'{prefix}_prior'] = compute_ap(other_idx, col)

        records.append(row)

    return pd.DataFrame(records).set_index('cluster')

def extract_holdout_mean_ap(df, score_col_name='score_col', cluster_col_name='cluster_id', mean_ap_col_name='mean_ap_holdout'):
    """
    For each cluster-specific score column, compute mean AP across all other clusters.
    For score column targeting cluster i, compute mean AP over all clusters except i.
    """
    results = []
    
    for col in df.columns:
        if col.endswith('Full') or col.endswith('Null') or col.endswith('full') or col.endswith('null'):
            continue
        cluster_id = int(col.rsplit('_', 1)[-1])
        all_aps = df[col].drop(index=cluster_id)
        mean_ap = all_aps.mean()
        results.append({
            score_col_name: col,
            cluster_col_name: cluster_id,
            mean_ap_col_name: mean_ap
        })
    
    return pd.DataFrame(results)

def extract_holdout_mean_ap_dict(df, cluster_to_col, score_col_name='score_col',cluster_col_name='cluster_id', mean_ap_col_name='mean_ap_holdout'):
    """
    For each cluster-specific score column, compute mean AP across all other clusters.
    """
    results = []
    for cluster_id, col_name in cluster_to_col.items():
        if col_name in df.columns and cluster_id in df.index:
            # Get all AP values for this column except the target cluster
            all_aps = df[col_name].drop(index=cluster_id)
            mean_ap = all_aps.mean()
            
            results.append({
                score_col_name: col_name,
                cluster_col_name: cluster_id,
                mean_ap_col_name: mean_ap
            })
    return pd.DataFrame(results)

"""
Compute p-values
"""

def compute_pairwise_pvalues_general(df):
    cols = df.columns.tolist()
    n = len(cols)
    pval_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                pval_matrix[i, j] = 1.0
            elif i < j:
                stat, pval = wilcoxon(df[cols[i]], df[cols[j]])
                pval_matrix[i, j] = pval
                pval_matrix[j, i] = pval
    
    return pd.DataFrame(pval_matrix, index=cols, columns=cols)


def paired_wilcoxon_vs_ref(
    df,
    ref_col,
    test_cols,
    alternative="two-sided",
    zero_method="wilcox",
    correction=False,
    drop_nonfinite=True,
):
    """
    For each test column, isolate the matched rows where both ref_col and test_col
    are non-null, then compute a paired Wilcoxon signed-rank test between the
    reference and test values.
    """

    if ref_col not in df.columns:
        raise ValueError(f"ref_col '{ref_col}' not found in df.columns")

    missing_test_cols = [c for c in test_cols if c not in df.columns]
    if missing_test_cols:
        raise ValueError(f"Missing test columns: {missing_test_cols}")

    rows = []

    ref_present = df[ref_col].notna()

    for c in test_cols:
        test_present = df[c].notna()

        matched_mask = ref_present & test_present
        ref_only_mask = ref_present & (~test_present)
        test_only_mask = (~ref_present) & test_present

        x_ref = df.loc[matched_mask, ref_col].to_numpy()
        x_test = df.loc[matched_mask, c].to_numpy()

        if drop_nonfinite:
            finite_mask = np.isfinite(x_ref) & np.isfinite(x_test)
            x_ref = x_ref[finite_mask]
            x_test = x_test[finite_mask]

        n_matched = len(x_ref)
        n_ref_only = int(ref_only_mask.sum())
        n_test_only = int(test_only_mask.sum())

        if n_matched == 0:
            stat = np.nan
            pval = np.nan
            median_ref = np.nan
            median_test = np.nan
            median_diff = np.nan
            mean_diff = np.nan
        else:
            diffs = x_test - x_ref
            median_ref = float(np.median(x_ref))
            median_test = float(np.median(x_test))
            median_diff = float(np.median(diffs))
            mean_diff = float(np.mean(diffs))

            if np.allclose(diffs, 0):
                stat = 0.0
                pval = 1.0
            else:
                res = wilcoxon(
                    x_test,
                    x_ref,
                    alternative=alternative,
                    zero_method=zero_method,
                    correction=correction,
                )
                stat = float(res.statistic)
                pval = float(res.pvalue)

        rows.append({
            "dataset": c,
            "ref_col": ref_col,
            "n_matched_groups": n_matched,
            "n_ref_has_test_missing": n_ref_only,
            "n_test_has_ref_missing": n_test_only,
            "wilcoxon_stat": stat,
            "wilcoxon_pvalue": pval,
            "median_ref": median_ref,
            "median_test": median_test,
            "median_test_minus_ref": median_diff,
            "mean_test_minus_ref": mean_diff,
        })

    out_df = pd.DataFrame(rows)
    return out_df

def add_star_col(df, pval_col, star_col):
    def stars(p):
        if pd.isna(p):
            return ''
        elif p < 0.0001:
            return '****'
        elif p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'
    
    result = df.copy()
    result[star_col] = result[pval_col].map(stars)
    return result

def pval_to_stars(pval_df):
    def stars(p):
        if pd.isna(p):
            return ''
        elif p < 0.0001:
            return '****'
        elif p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'
    
    return pval_df.applymap(stars)

def log2_enrichment_vs_base(df, base_col, eps=0.5):
    cols = [c for c in df.columns if c != base_col]
    out = np.log2(df[cols].add(eps).div(df[base_col].add(eps), axis=0))
    # out.columns = [f"{c}_vs_{base_col}_log2" for c in out.columns]
    return out

# for gravy score
def delta_vs_base(df, base_col):
    cols = [c for c in df.columns if c != base_col]
    out = df[cols].sub(df[base_col], axis=0)
    # out.columns = [f"{c}_minus_{base_col}" for c in out.columns]
    return out

def residue_enrichment_top_n(
    df: pd.DataFrame,
    peptide_col: str,
    cluster_col: str,
    score_cols: list,
    top_n: int,
    residue: str = 'C',
    fraction: bool = False,
    position: int = None,
):
    clusters = sorted(df[cluster_col].unique())
    records = []

    for cluster in clusters:
        cluster_df = df[df[cluster_col] == cluster]
        row = {'cluster': cluster}

        for col in score_cols:
            top_df = cluster_df.nlargest(top_n, col)
            peptides = top_df[peptide_col].dropna()
            
            if position is not None:
                count = (peptides.str[position - 1] == residue).sum()
            else:
                count = peptides.str.contains(residue).sum()
            
            row[col] = count / min(top_n, len(top_df)) if fraction else count

        records.append(row)

    return pd.DataFrame(records).set_index('cluster')

def avg_col_top_n(
    df: pd.DataFrame,
    cluster_col: str,
    score_cols: list,
    top_n: int,
    target_col: str,
):
    clusters = sorted(df[cluster_col].unique())
    records = []

    for cluster in clusters:
        cluster_df = df[df[cluster_col] == cluster]
        row = {'cluster': cluster}

        for col in score_cols:
            top_df = cluster_df.nlargest(top_n, col)
            row[col] = top_df[target_col].mean()

        records.append(row)

    return pd.DataFrame(records).set_index('cluster')

def compute_log2_vl_shift_long(
    df,
    peptide_col,
    allele_col,
    alleles,
    base_to_models,
    top_n=2000,
    out_value_col="log2_vl_vs_base",
    eps=1e-6,
):
    """
    For each allele and each score column:
      1) take top_n peptides by score within that allele
      2) among those, count peptides ending in V and L
      3) compute V/L ratio
      4) for each base -> model mapping, compute log2[(V/L)_model / (V/L)_base]
    """
    all_score_cols = sorted(
        set(base_to_models.keys()).union(*[set(v) for v in base_to_models.values()])
    )
    ratio_dict = {}

    for allele in alleles:
        sub = df[df[allele_col] == allele]

        for score_col in all_score_cols:
            top = sub.nlargest(top_n, score_col)[[peptide_col, score_col]]

            last_aa = top[peptide_col].astype(str).str[-1]
            n_v = (last_aa == "V").sum()
            n_l = (last_aa == "L").sum()

            ratio = (n_v + eps) / (n_l + eps)
            ratio_dict[(allele, score_col)] = ratio

    rows = []
    for base_col, model_cols in base_to_models.items():
        for model_col in model_cols:
            for allele in alleles:
                base_ratio = ratio_dict[(allele, base_col)]
                model_ratio = ratio_dict[(allele, model_col)]
                log2_shift = np.log2(model_ratio / base_ratio)

                rows.append({
                    "score_col_allele": f"{model_col}_{str(allele).lower()}",
                    out_value_col: log2_shift,
                })

    out_df = pd.DataFrame(rows)
    return out_df

AA_ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"
def _build_aa_lut(aa_alphabet: str = AA_ALPHABET_20) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int16)
    for i, aa in enumerate(aa_alphabet):
        lut[ord(aa)] = i
    return lut

def _encode_seqs_to_idx(seqs: list[str], L: int, lut: np.ndarray) -> np.ndarray:
    N = len(seqs)
    if N == 0:
        return np.empty((0, L), dtype=np.int64)
    joined = "".join(seqs).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(N, L)
    idx = lut[arr]
    return idx.astype(np.int64)


@torch.no_grad()
def _pwm_from_idx_gpu(idx_gpu: torch.Tensor, K: int, pseudocount: float) -> torch.Tensor:
    device = idx_gpu.device
    N, L = idx_gpu.shape
    counts = torch.zeros((L, K), device=device, dtype=torch.float32)

    for pos in range(L):
        aa = idx_gpu[:, pos]
        v = (aa >= 0) & (aa < K)
        aa = aa[v]
        if aa.numel():
            counts[pos] = torch.bincount(aa, minlength=K).float()

    denom = counts.sum(dim=1, keepdim=True)  # (L,1)
    if pseudocount > 0:
        counts = counts + (pseudocount / K)
        denom = denom + pseudocount
    return counts / torch.clamp(denom, min=1e-12)


@torch.no_grad()
def _pairwise_jsd_chunked_gpu(
    P: torch.Tensor, # (A,L,K) float32 on device
    *,
    chunk_size: int = 64,
    eps: float = 1e-12,
    normalize_by_length: bool = True,
) -> torch.Tensor:
    device = P.device
    A, L, K = P.shape
    J = torch.empty((A, A), device=device, dtype=torch.float32)

    P = torch.clamp(P, min=eps)
    logP = torch.log(P)

    denom = float(L) if normalize_by_length else 1.0

    for i0 in tqdm(range(0, A, chunk_size), desc="[JSD] chunks", leave=True):
        i1 = min(A, i0 + chunk_size)
        Pi = P[i0:i1] # (B,L,K)
        logPi = logP[i0:i1] # (B,L,K)

        # broadcast vs all
        Pi_b = Pi.unsqueeze(1) # (B,1,L,K)
        Pj_b = P.unsqueeze(0) # (1,A,L,K)
        M = 0.5 * (Pi_b + Pj_b) # (B,A,L,K)
        logM = torch.log(M)

        # KL(Pi||M) + KL(Pj||M)
        # KL(Pi||M): sum Pi*(logPi-logM)
        kl_i = (Pi_b * (logPi.unsqueeze(1) - logM)).sum(dim=(2, 3)) # (B,A)

        # KL(Pj||M): sum Pj*(logPj-logM)
        kl_j = (Pj_b * (logP.unsqueeze(0) - logM)).sum(dim=(2, 3)) # (B,A)

        js = 0.5 * (kl_i + kl_j) / denom
        J[i0:i1] = js

    # force symmetry / zero diag (numerical)
    J = 0.5 * (J + J.t())
    J.fill_diagonal_(0.0)
    return J



AA_ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"
def compute_bootstrap_train_vs_topk_jsd_by_allele_gpu(
    ms_train_df: pd.DataFrame,
    holec_train_df: pd.DataFrame,
    scored_bg_df: pd.DataFrame,
    score_cols: List[str],
    *,
    allele_col: str = "allele",
    seq_col: str = "seq",
    top_k: int = 2000,
    n_iterations: int = 100,
    aa_alphabet: str = AA_ALPHABET_20,
    pseudocount: float = 1.0,
    device: str = "cuda",
    jsd_chunk_size: int = 64,
    random_state: int = 42,
    show_progress: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Per allele:
      - Build a source set consisting of:
          1) ms_train
          2) holec_train
          3) top-K peptides per score column from scored_bg_df
      - Determine min_count across all available sources for that allele
      - Run n_iterations of matched-size subsampling:
          * the source already at min_count is used as-is
          * all larger sources are randomly subsampled down to min_count
      - Compute PWM per source on GPU
      - Compute pairwise JSD matrix on GPU
      - Return mean and variance JSD matrices per allele
    """

    req_ms = {allele_col, seq_col}
    req_holec = {allele_col, seq_col}
    req_bg = {allele_col, seq_col, *score_cols}

    missing_ms = req_ms - set(ms_train_df.columns)
    missing_holec = req_holec - set(holec_train_df.columns)
    missing_bg = req_bg - set(scored_bg_df.columns)

    if missing_ms:
        raise ValueError(f"ms_train_df missing required columns: {missing_ms}")
    if missing_holec:
        raise ValueError(f"holec_train_df missing required columns: {missing_holec}")
    if missing_bg:
        raise ValueError(f"scored_bg_df missing required columns: {missing_bg}")

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be > 0")

    dev = device.lower()
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    torch_device = torch.device(dev)

    lut = _build_aa_lut(aa_alphabet)
    K_aa = len(aa_alphabet)

    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)

    ms_d = ms_train_df[[allele_col, seq_col]].copy()
    holec_d = holec_train_df[[allele_col, seq_col]].copy()
    bg_d = scored_bg_df[[allele_col, seq_col] + score_cols].copy()

    ms_d[allele_col] = ms_d[allele_col].astype(str)
    ms_d[seq_col] = ms_d[seq_col].astype(str)

    holec_d[allele_col] = holec_d[allele_col].astype(str)
    holec_d[seq_col] = holec_d[seq_col].astype(str)

    bg_d[allele_col] = bg_d[allele_col].astype(str)
    bg_d[seq_col] = bg_d[seq_col].astype(str)
    for c in score_cols:
        bg_d[c] = pd.to_numeric(bg_d[c], errors="coerce")

    ms_map = {
        str(a): s.tolist()
        for a, s in ms_d.groupby(allele_col, sort=False)[seq_col]
    }
    holec_map = {
        str(a): s.tolist()
        for a, s in holec_d.groupby(allele_col, sort=False)[seq_col]
    }

    bg_group_indices = bg_d.groupby(allele_col, sort=False).indices

    all_alleles = sorted(
        set(ms_map.keys()) | set(holec_map.keys()) | set(map(str, bg_group_indices.keys()))
    )

    if len(all_alleles) == 0:
        return {}, {}, pd.DataFrame()

    def _get_topk_seqs_per_score_for_allele(bg_sub: pd.DataFrame) -> Dict[str, List[str]]:
        """
        For one allele's scored background block:
        returns dict[score_col] -> top-K sequences by largest score
        """
        out = {}
        seqs_np = bg_sub[seq_col].to_numpy(dtype=object, copy=False)

        for sc in score_cols:
            vals = bg_sub[sc].to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(vals)
            if not np.any(valid):
                out[sc] = []
                continue

            valid_idx = np.flatnonzero(valid)
            valid_vals = vals[valid]

            k_eff = min(top_k, valid_vals.size)
            if k_eff <= 0:
                out[sc] = []
                continue

            take = np.argpartition(valid_vals, -k_eff)[-k_eff:]
            chosen_idx = valid_idx[take]

            out[sc] = seqs_np[chosen_idx].tolist()

        return out

    jsd_mean_by_allele: Dict[str, pd.DataFrame] = {}
    jsd_var_by_allele: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    allele_iter = tqdm(
        all_alleles,
        desc="[bootstrap-jsd] alleles",
        leave=True,
        disable=not show_progress,
    )

    for allele in allele_iter:
        ms_seqs = ms_map.get(allele, [])
        holec_seqs = holec_map.get(allele, [])

        if allele in bg_group_indices:
            bg_sub = bg_d.loc[bg_group_indices[allele], [seq_col] + score_cols]
            topk_map = _get_topk_seqs_per_score_for_allele(bg_sub)
        else:
            topk_map = {sc: [] for sc in score_cols}

        count_map = {
            "ms_train_n": len(ms_seqs),
            "holec_train_n": len(holec_seqs),
        }
        for sc in score_cols:
            count_map[f"{sc}_topk_n"] = len(topk_map.get(sc, []))

        source_names = ["ms_train", "holec_train"] + list(score_cols)
        source_seqs_cpu = {
            "ms_train": ms_seqs,
            "holec_train": holec_seqs,
            **topk_map,
        }

        counts_in_order = [len(source_seqs_cpu[name]) for name in source_names]
        min_count = min(counts_in_order) if len(counts_in_order) else 0

        status = "ok"
        if min_count < 1:
            status = "skipped_min_count_lt_1"

        summary_row = {
            allele_col: allele,
            **count_map,
            "min_count_used": int(min_count),
            "n_sources_in_matrix": len(source_names),
            "n_iterations": int(n_iterations),
            "status": status,
        }

        if status != "ok":
            summary_rows.append(summary_row)
            continue

        first_nonempty = None
        for nm in source_names:
            if len(source_seqs_cpu[nm]) > 0:
                first_nonempty = source_seqs_cpu[nm]
                break

        if first_nonempty is None:
            summary_row["status"] = "skipped_all_empty"
            summary_rows.append(summary_row)
            continue

        L = len(first_nonempty[0])

        source_idx_gpu = {}
        bad_length = False

        source_iter = tqdm(
            source_names,
            desc=f"[encode] {allele}",
            leave=True,
            disable=not show_progress,
        )

        for nm in source_iter:
            seqs = source_seqs_cpu[nm]

            if len(seqs) > 0:
                if any(len(s) != L for s in seqs):
                    bad_length = True
                    break

            idx_np = _encode_seqs_to_idx(seqs, L, lut)
            source_idx_gpu[nm] = torch.from_numpy(idx_np).to(torch_device)

        if bad_length:
            summary_row["status"] = "skipped_mixed_lengths_within_allele"
            summary_rows.append(summary_row)
            continue

        S = len(source_names)
        sum_J = np.zeros((S, S), dtype=np.float64)
        sumsq_J = np.zeros((S, S), dtype=np.float64)

        iter_bar = tqdm(
            range(n_iterations),
            desc=f"[bootstrap] {allele}",
            leave=True,
            disable=not show_progress,
        )

        for _ in iter_bar:
            pwm_list = []

            for nm in source_names:
                idx_full = source_idx_gpu[nm]
                n_src = idx_full.shape[0]

                if n_src == min_count:
                    idx_sample = idx_full
                else:
                    pick = torch.randperm(n_src, device=torch_device)[:min_count]
                    idx_sample = idx_full[pick]

                pwm = _pwm_from_idx_gpu(
                    idx_sample,
                    K=K_aa,
                    pseudocount=pseudocount,
                )
                pwm_list.append(pwm)

            P_all = torch.stack(pwm_list, dim=0)
            J_gpu = _pairwise_jsd_chunked_gpu(
                P_all,
                chunk_size=jsd_chunk_size,
                normalize_by_length=True,
            )
            J_np = J_gpu.detach().cpu().numpy().astype(np.float64)

            sum_J += J_np
            sumsq_J += J_np * J_np

        mean_J = sum_J / float(n_iterations)
        var_J = (sumsq_J / float(n_iterations)) - (mean_J * mean_J)

        mean_J = 0.5 * (mean_J + mean_J.T)
        var_J = 0.5 * (var_J + var_J.T)
        mean_J = np.clip(mean_J, 0.0, None)
        var_J = np.clip(var_J, 0.0, None)
        np.fill_diagonal(mean_J, 0.0)
        np.fill_diagonal(var_J, 0.0)

        jsd_mean_by_allele[allele] = pd.DataFrame(
            mean_J.astype(np.float32),
            index=source_names,
            columns=source_names,
        )
        jsd_var_by_allele[allele] = pd.DataFrame(
            var_J.astype(np.float32),
            index=source_names,
            columns=source_names,
        )

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)

    ordered_cols = (
        [allele_col, "status", "min_count_used", "n_sources_in_matrix", "n_iterations"]
        + ["ms_train_n", "holec_train_n"]
        + [f"{sc}_topk_n" for sc in score_cols]
    )
    ordered_cols = [c for c in ordered_cols if c in summary_df.columns] + [
        c for c in summary_df.columns if c not in ordered_cols
    ]
    summary_df = summary_df[ordered_cols]

    return jsd_mean_by_allele, jsd_var_by_allele, summary_df

@torch.no_grad()
def _pwm_probs_from_seqs_gpu(
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

    idx_np = _encode_seqs_to_idx(seqs_L, L, lut) 
    idx = torch.from_numpy(idx_np).to(device)

    valid_pos = (idx >= 0) & (idx < K) # (N,L)
    invalid_pep = (~valid_pos).any(dim=1) # (N,)

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

    pos_ids = torch.arange(L, device=device).view(1, L).expand(N, L) # (N,L)
    aa_flat = idx.reshape(-1)
    pos_flat = pos_ids.reshape(-1)
    valid_flat = valid_pos.reshape(-1)

    aa_v = aa_flat[valid_flat]
    pos_v = pos_flat[valid_flat]

    lin = pos_v * K + aa_v # (num_valid_positions,)
    counts_flat = torch.bincount(lin, minlength=L * K).to(torch.float32)
    counts = counts_flat.view(L, K) # (L,K)

    denom = counts.sum(dim=1, keepdim=True) # (L,1)

    if pseudocount is not None and pseudocount > 0:
        counts = counts + (pseudocount / K)
        denom = denom + pseudocount

    P = counts / torch.clamp(denom, min=eps) # (L,K)
    return P.detach().cpu().numpy().astype(np.float32), N_total, N_used



def _js_divergence_pwm(P: np.ndarray, Q: np.ndarray, *, eps: float = 1e-12, normalize_by_length: bool = True) -> float:
    """
    Jensen-Shannon divergence between two PWMs P and Q, both shape (L,K), row-stochastic.
    Returns JS in nats. If normalize_by_length=True, returns mean JS per position (divide by L).
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


def compute_js_divergence_pwms_gpu(
    ref_df: pd.DataFrame,
    test_dfs: dict[str, pd.DataFrame],
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build PWMs per (allele,length) for ref_df and each test df, then compute JS divergence
    between PWMs for matching (allele,length).
    """
    device = device.lower()
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    torch_device = torch.device(device)

    for name, d in [("ref_df", ref_df), *[(f"test_dfs[{k}]", v) for k, v in test_dfs.items()]]:
        if allele_col not in d.columns or seq_col not in d.columns:
            raise ValueError(f"{name} must contain columns {allele_col!r} and {seq_col!r}")

    lut = _build_aa_lut(aa_alphabet)
    K = len(aa_alphabet)

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

    ref_pwm: dict[tuple[str, int], np.ndarray] = {}
    ref_counts: dict[tuple[str, int], tuple[int, int]] = {}

    total_ref = len(ref_groups)
    with tqdm(total=total_ref, desc="Ref PWM (allele×len)", leave=True) as pbar:
        for (a, L), seqs in ref_groups.items():
            P, N_total, N_used = _pwm_probs_from_seqs_gpu(
                seqs, L, lut, K,
                pseudocount=pseudocount,
                device=torch_device,
                drop_invalid_peptides=drop_invalid_peptides,
            )
            ref_counts[(a, L)] = (N_total, N_used)
            if N_used >= min_peptides_per_group and np.isfinite(P).all():
                ref_pwm[(a, L)] = P
            pbar.update(1)

    keys = sorted(ref_pwm.keys(), key=lambda x: (str(x[0]), int(x[1])))
    idx_labels = [f"{a}_{L}" for a, L in keys]

    js_cols = {}
    allele_len_counts_by_test: dict[str, dict[tuple[str, int], int]] = {} 

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

        total = len(keys)
        with tqdm(total=total, desc=f"JS vs ref: {test_name}", leave=True) as pbar:
            for i, (a, L) in enumerate(keys):
                refP = ref_pwm.get((a, L), None)
                if refP is None:
                    pbar.update(1)
                    continue

                seqs = test_groups.get((a, L), None)
                if not seqs:
                    pbar.update(1)
                    continue

                P_test, N_total, N_used = _pwm_probs_from_seqs_gpu(
                    seqs, L, lut, K,
                    pseudocount=pseudocount,
                    device=torch_device,
                    drop_invalid_peptides=drop_invalid_peptides,
                )
                length_counts[(a, L)] = int(N_used)

                # require enough peptides on test side too
                if N_used < min_peptides_per_group or not np.isfinite(P_test).all():
                    pbar.update(1)
                    continue

                js = _js_divergence_pwm(refP, P_test, normalize_by_length=normalize_js_by_length)
                js_vals[i] = np.float32(js)
                pbar.update(1)

        js_cols[test_name] = js_vals
        allele_len_counts_by_test[test_name] = length_counts

    js_by_allele_len = pd.DataFrame(js_cols, index=pd.Index(idx_labels, name="allele_length"))

    alleles = sorted({a for a, _L in keys})
    js_by_allele_unweighted = pd.DataFrame(index=pd.Index(alleles, name="allele"), columns=list(js_cols.keys()), dtype=np.float32)

    js_by_allele_weighted = pd.DataFrame(index=pd.Index(alleles, name="allele"), columns=list(js_cols.keys()), dtype=np.float32)

    lengths_by_allele: dict[str, list[int]] = {}
    for a, L in keys:
        lengths_by_allele.setdefault(a, []).append(L)

    for test_name in js_cols.keys():
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


AA_ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

def _build_aa_lut_entropy(aa_alphabet: str = AA_ALPHABET_20) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int16)
    for i, aa in enumerate(aa_alphabet):
        lut[ord(aa)] = i
    return lut


def _encode_seqs_to_idx_entropy(seqs: List[str], L: int, lut: np.ndarray) -> np.ndarray:
    """
    Encode equal-length amino acid sequences to integer index array of shape (N, L).
    Invalid amino acids get -1.
    """
    N = len(seqs)
    if N == 0:
        return np.empty((0, L), dtype=np.int64)
    joined = "".join(seqs).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(N, L)
    idx = lut[arr]
    return idx.astype(np.int64, copy=False)


def _entropy_from_idx_cpu(
    idx: np.ndarray,
    K: int,
    log_base: float = 2.0,
    pseudocount: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    idx: (N, L) int array with valid AA indices in [0, K-1], invalids = -1
    returns: (L,) entropy per position
    """
    if idx.ndim != 2:
        raise ValueError("idx must have shape (N, L)")
    N, L = idx.shape
    ent = np.full(L, np.nan, dtype=np.float32)

    if N == 0:
        return ent

    log_denom = np.log(log_base)

    for pos in range(L):
        aa = idx[:, pos]
        valid = (aa >= 0) & (aa < K)
        aa = aa[valid]
        if aa.size == 0:
            continue

        counts = np.bincount(aa, minlength=K).astype(np.float64)
        if pseudocount > 0:
            counts = counts + (pseudocount / K)

        probs = counts / np.clip(counts.sum(), eps, None)
        nz = probs > 0
        h = -np.sum(probs[nz] * (np.log(probs[nz]) / log_denom))
        ent[pos] = np.float32(h)

    return ent


def _entropy_from_idx_torch(
    idx: np.ndarray,
    K: int,
    log_base: float = 2.0,
    pseudocount: float = 0.0,
    eps: float = 1e-12,
    device: str = "cuda",
) -> np.ndarray:
    """
    Torch/GPU version. Usually only worth it if you already want GPU.
    idx: (N, L) numpy int64
    returns: (L,) numpy float32
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    t = torch.from_numpy(idx).to(device)
    N, L = t.shape
    out = torch.full((L,), float("nan"), device=device, dtype=torch.float32)

    if N == 0:
        return out.cpu().numpy()

    log_denom = float(np.log(log_base))

    for pos in range(L):
        aa = t[:, pos]
        valid = (aa >= 0) & (aa < K)
        aa = aa[valid]
        if aa.numel() == 0:
            continue

        counts = torch.bincount(aa, minlength=K).float()
        if pseudocount > 0:
            counts = counts + (pseudocount / K)

        probs = counts / torch.clamp(counts.sum(), min=eps)
        probs = torch.clamp(probs, min=eps)
        h = -(probs * (torch.log(probs) / log_denom)).sum()
        out[pos] = h

    return out.cpu().numpy().astype(np.float32, copy=False)


def compute_topk_position_entropy_by_allele(
    df: pd.DataFrame,
    score_cols: List[str],
    *,
    allele_col: str = "allele",
    seq_col: str = "seq",
    top_k: int = 2000,
    aa_alphabet: str = AA_ALPHABET_20,
    log_base: float = 2.0,
    pseudocount: float = 0.0,
    device: str = "cpu",
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Per allele, per score column:
      - isolate top_k peptides by score
      - compute position-specific entropy from those peptides
    """
    required = {allele_col, seq_col, *score_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be > 0 and != 1")

    use_gpu = (device.lower() == "cuda")
    if use_gpu and (not _TORCH_AVAILABLE or not torch.cuda.is_available()):
        device = "cpu"
        use_gpu = False

    d = df[[allele_col, seq_col] + score_cols].copy()
    d[allele_col] = d[allele_col].astype(str)
    d[seq_col] = d[seq_col].astype(str)
    for sc in score_cols:
        d[sc] = pd.to_numeric(d[sc], errors="coerce")

    # infer peptide length from first non-empty sequence
    nonempty = d[seq_col].dropna()
    if len(nonempty) == 0:
        raise ValueError("No sequences found in seq_col")
    L = len(nonempty.iloc[0])

    # defensive: ensure same length
    seq_lens = d[seq_col].str.len().to_numpy()
    if np.any(seq_lens != L):
        raise ValueError("Not all sequences are the same length. This function assumes fixed-length peptides.")

    lut = _build_aa_lut_entropy(aa_alphabet)
    K = len(aa_alphabet)

    grouped = d.groupby(allele_col, sort=False).indices
    alleles = list(grouped.keys())

    # nested dict storage: score_col -> allele -> entropy vector
    out_map = {sc: {} for sc in score_cols}

    allele_iter = tqdm(
        alleles,
        desc="Alleles",
        leave=True,
        disable=not show_progress,
    )

    for allele in allele_iter:
        idx_rows = grouped[allele]
        sub = d.iloc[idx_rows]

        seqs = sub[seq_col].to_numpy(dtype=object, copy=False)

        score_iter = tqdm(
            score_cols,
            desc=f"Score cols ({allele})",
            leave=True,
            disable=not show_progress,
        )

        for sc in score_iter:
            vals = sub[sc].to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(vals)
            if not np.any(valid):
                out_map[sc][allele] = np.full(L, np.nan, dtype=np.float32)
                continue

            valid_idx = np.flatnonzero(valid)
            valid_vals = vals[valid]

            k_eff = min(top_k, valid_vals.size)
            if k_eff == 0:
                out_map[sc][allele] = np.full(L, np.nan, dtype=np.float32)
                continue

            top_take = np.argpartition(valid_vals, -k_eff)[-k_eff:]
            chosen_idx = valid_idx[top_take]
            top_seqs = seqs[chosen_idx].tolist()

            idx_encoded = _encode_seqs_to_idx_entropy(top_seqs, L, lut)

            if use_gpu:
                ent = _entropy_from_idx_torch(
                    idx_encoded,
                    K=K,
                    log_base=log_base,
                    pseudocount=pseudocount,
                    device=device,
                )
            else:
                ent = _entropy_from_idx_cpu(
                    idx_encoded,
                    K=K,
                    log_base=log_base,
                    pseudocount=pseudocount,
                )

            out_map[sc][allele] = ent

    # convert to dfs
    result = {}
    pos_cols = list(range(1, L + 1))
    for sc in score_cols:
        mat = np.vstack([out_map[sc].get(a, np.full(L, np.nan, dtype=np.float32)) for a in alleles])
        result[sc] = pd.DataFrame(mat, index=alleles, columns=pos_cols)

    return result

AA_ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"


def _build_aa_lut_jsd(aa_alphabet: str = AA_ALPHABET_20) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int16)
    for i, aa in enumerate(aa_alphabet):
        lut[ord(aa)] = i
    return lut


def _encode_seqs_to_idx_jsd(seqs: List[str], L: int, lut: np.ndarray) -> np.ndarray:
    """
    Encode equal-length amino acid sequences to integer index array of shape (N, L).
    Invalid amino acids get -1.
    """
    N = len(seqs)
    if N == 0:
        return np.empty((0, L), dtype=np.int64)
    joined = "".join(seqs).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(N, L)
    idx = lut[arr]
    return idx.astype(np.int64, copy=False)


def _pwm_from_idx_cpu_jsd(
    idx: np.ndarray,
    K: int,
    pseudocount: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    idx: (N, L) int array with valid AA indices in [0, K-1], invalids = -1
    returns: (L, K) PWM
    """
    if idx.ndim != 2:
        raise ValueError("idx must have shape (N, L)")
    N, L = idx.shape
    pwm = np.full((L, K), np.nan, dtype=np.float32)

    if N == 0:
        return pwm

    for pos in range(L):
        aa = idx[:, pos]
        valid = (aa >= 0) & (aa < K)
        aa = aa[valid]
        if aa.size == 0:
            continue

        counts = np.bincount(aa, minlength=K).astype(np.float64)
        if pseudocount > 0:
            counts = counts + (pseudocount / K)

        probs = counts / np.clip(counts.sum(), eps, None)
        pwm[pos] = probs.astype(np.float32, copy=False)

    return pwm


def _pwm_from_idx_torch_jsd(
    idx: np.ndarray,
    K: int,
    pseudocount: float = 0.0,
    eps: float = 1e-12,
    device: str = "cuda",
) -> np.ndarray:
    """
    Torch/GPU version.
    idx: (N, L) numpy int64
    returns: (L, K) numpy float32 PWM
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    t = torch.from_numpy(idx).to(device)
    N, L = t.shape
    out = torch.full((L, K), float("nan"), device=device, dtype=torch.float32)

    if N == 0:
        return out.cpu().numpy()

    for pos in range(L):
        aa = t[:, pos]
        valid = (aa >= 0) & (aa < K)
        aa = aa[valid]
        if aa.numel() == 0:
            continue

        counts = torch.bincount(aa, minlength=K).float()
        if pseudocount > 0:
            counts = counts + (pseudocount / K)

        probs = counts / torch.clamp(counts.sum(), min=eps)
        out[pos] = probs

    return out.cpu().numpy().astype(np.float32, copy=False)


def _jsd_per_position_from_pwms(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    log_base: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    P, Q: (L, K) PWMs
    returns: (L,) JSD per position
    """
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    valid_rows = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    out = np.full(P.shape[0], np.nan, dtype=np.float32)

    if not np.any(valid_rows):
        return out

    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)
    M = 0.5 * (P + Q)

    log_denom = np.log(log_base)

    kl_pm = np.sum(P * ((np.log(P) - np.log(M)) / log_denom), axis=1)
    kl_qm = np.sum(Q * ((np.log(Q) - np.log(M)) / log_denom), axis=1)
    jsd = 0.5 * (kl_pm + kl_qm)

    out[valid_rows] = jsd[valid_rows].astype(np.float32, copy=False)
    return out


def compute_topk_position_jsd_by_allele(
    df: pd.DataFrame,
    score_cols: List[str],
    *,
    ref_score_col: str,
    allele_col: str = "allele",
    seq_col: str = "seq",
    top_k: int = 2000,
    aa_alphabet: str = AA_ALPHABET_20,
    log_base: float = 2.0,
    pseudocount: float = 0.0,
    device: str = "cpu",
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Per allele:
      - isolate top_k peptides by ref_score_col
      - isolate top_k peptides by each score column
      - compute PWM per set
      - compute position-specific JSD between ref PWM and score-col PWM
    """
    all_score_cols = [ref_score_col] + list(score_cols)
    required = {allele_col, seq_col, *all_score_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be > 0 and != 1")

    score_cols = [sc for sc in score_cols if sc != ref_score_col]

    use_gpu = (device.lower() == "cuda")
    if use_gpu and (not _TORCH_AVAILABLE or not torch.cuda.is_available()):
        device = "cpu"
        use_gpu = False

    d = df[[allele_col, seq_col] + all_score_cols].copy()
    d[allele_col] = d[allele_col].astype(str)
    d[seq_col] = d[seq_col].astype(str)
    for sc in all_score_cols:
        d[sc] = pd.to_numeric(d[sc], errors="coerce")

    nonempty = d[seq_col].dropna()
    if len(nonempty) == 0:
        raise ValueError("No sequences found in seq_col")
    L = len(nonempty.iloc[0])

    seq_lens = d[seq_col].str.len().to_numpy()
    if np.any(seq_lens != L):
        raise ValueError("Not all sequences are the same length. This function assumes fixed-length peptides.")

    lut = _build_aa_lut_jsd(aa_alphabet)
    K = len(aa_alphabet)

    grouped = d.groupby(allele_col, sort=False).indices
    alleles = list(grouped.keys())

    # nested dict storage: score_col -> allele -> jsd vector
    out_map = {sc: {} for sc in score_cols}

    allele_iter = tqdm(
        alleles,
        desc="Alleles",
        leave=True,
        disable=not show_progress,
    )

    for allele in allele_iter:
        idx_rows = grouped[allele]
        sub = d.iloc[idx_rows]

        seqs = sub[seq_col].to_numpy(dtype=object, copy=False)

        # reference top-k
        ref_vals = sub[ref_score_col].to_numpy(dtype=np.float64, copy=False)
        ref_valid = np.isfinite(ref_vals)

        if not np.any(ref_valid):
            ref_pwm = None
        else:
            ref_valid_idx = np.flatnonzero(ref_valid)
            ref_valid_vals = ref_vals[ref_valid]
            ref_k_eff = min(top_k, ref_valid_vals.size)
            ref_take = np.argpartition(ref_valid_vals, -ref_k_eff)[-ref_k_eff:]
            ref_idx = ref_valid_idx[ref_take]
            ref_seqs = seqs[ref_idx].tolist()
            ref_encoded = _encode_seqs_to_idx_jsd(ref_seqs, L, lut)

            if use_gpu:
                ref_pwm = _pwm_from_idx_torch_jsd(
                    ref_encoded,
                    K=K,
                    pseudocount=pseudocount,
                    device=device,
                )
            else:
                ref_pwm = _pwm_from_idx_cpu_jsd(
                    ref_encoded,
                    K=K,
                    pseudocount=pseudocount,
                )

        score_iter = tqdm(
            score_cols,
            desc=f"Score cols ({allele})",
            leave=True,
            disable=not show_progress,
        )

        for sc in score_iter:
            if ref_pwm is None:
                out_map[sc][allele] = np.full(L, np.nan, dtype=np.float32)
                continue

            vals = sub[sc].to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(vals)

            if not np.any(valid):
                out_map[sc][allele] = np.full(L, np.nan, dtype=np.float32)
                continue

            valid_idx = np.flatnonzero(valid)
            valid_vals = vals[valid]

            k_eff = min(top_k, valid_vals.size)
            if k_eff == 0:
                out_map[sc][allele] = np.full(L, np.nan, dtype=np.float32)
                continue

            take = np.argpartition(valid_vals, -k_eff)[-k_eff:]
            chosen_idx = valid_idx[take]
            top_seqs = seqs[chosen_idx].tolist()

            idx_encoded = _encode_seqs_to_idx_jsd(top_seqs, L, lut)

            if use_gpu:
                pwm = _pwm_from_idx_torch_jsd(
                    idx_encoded,
                    K=K,
                    pseudocount=pseudocount,
                    device=device,
                )
            else:
                pwm = _pwm_from_idx_cpu_jsd(
                    idx_encoded,
                    K=K,
                    pseudocount=pseudocount,
                )

            jsd_pos = _jsd_per_position_from_pwms(
                ref_pwm,
                pwm,
                log_base=log_base,
            )
            out_map[sc][allele] = jsd_pos

    result = {}
    pos_cols = list(range(1, L + 1))
    for sc in score_cols:
        mat = np.vstack([out_map[sc].get(a, np.full(L, np.nan, dtype=np.float32)) for a in alleles])
        result[sc] = pd.DataFrame(mat, index=alleles, columns=pos_cols)

    return result


def map_peptides_to_proteins(peptides: List[str], protein_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Maps peptides to proteins using Aho-Corasick algorithm for efficient string matching.
    """
    if not peptides or protein_df.empty:
        return {}
    result = {peptide: {} for peptide in peptides}
    
    try:
        import ahocorasick
        
        print("Building Aho-Corasick automaton...")
        automaton = ahocorasick.Automaton()
        for i, peptide in enumerate(tqdm(peptides, desc="Adding peptides to automaton")):
            automaton.add_word(peptide, (i, peptide))
        
        print("Finalizing automaton...")
        automaton.make_automaton()
        
        print("Searching for peptides in proteins...")
        for _, protein in tqdm(protein_df.iterrows(), total=len(protein_df), desc="Processing proteins"):
            protein_id = protein['id']
            protein_seq = protein['seq']
            
            for end_pos, (_, peptide) in automaton.iter(protein_seq):
                start_pos = end_pos - len(peptide) + 1
                
                if protein_id not in result[peptide]:
                    result[peptide][protein_id] = {}
                
                mapping_idx = len(result[peptide][protein_id])
                
                n_flank_start = max(0, start_pos - 10)
                n_flank = protein_seq[n_flank_start:start_pos]
                
                c_flank_end = min(len(protein_seq), end_pos + 11)
                c_flank = protein_seq[end_pos + 1:c_flank_end]
                
                result[peptide][protein_id][mapping_idx] = {
                    "start": start_pos,
                    "end": end_pos,
                    "n_flank": n_flank,
                    "c_flank": c_flank
                }
        
    except ImportError:
        print("Warning: ahocorasick not found. Using fallback method.")
        return fallback_map_peptides(peptides, protein_df)
    
    return result


def fallback_map_peptides(peptides: List[str], protein_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Fallback function that maps peptides to proteins without using Aho-Corasick.
    This is used if ahocorasick is not available.
    
    Args:
        peptides: List of peptide sequences
        protein_df: DataFrame with columns 'id' for protein ID and 'seq' for protein sequence
        
    Returns:
        Dictionary with the same structure as map_peptides_to_proteins
    """
    result = {peptide: {} for peptide in peptides}
    
    print("Using fallback method for peptide mapping...")
    for _, protein in tqdm(protein_df.iterrows(), total=len(protein_df), desc="Processing proteins"):
        protein_id = protein['id']
        protein_seq = protein['seq']
        
        for peptide in tqdm(peptides, desc=f"Searching peptides in protein {protein_id}", leave=False):
            start_pos = 0
            
            while start_pos <= len(protein_seq) - len(peptide):
                pos = protein_seq.find(peptide, start_pos)
                if pos == -1:
                    break
                
                if protein_id not in result[peptide]:
                    result[peptide][protein_id] = {}
                
                mapping_idx = len(result[peptide][protein_id])
                
                end_pos = pos + len(peptide) - 1
                
                n_flank_start = max(0, pos - 10)
                n_flank = protein_seq[n_flank_start:pos]
                
                c_flank_end = min(len(protein_seq), end_pos + 11)
                c_flank = protein_seq[end_pos + 1:c_flank_end]
                
                result[peptide][protein_id][mapping_idx] = {
                    "start": pos,
                    "end": end_pos,
                    "n_flank": n_flank,
                    "c_flank": c_flank
                }
                
                start_pos = pos + 1
    
    return result

def compute_nonzero_percentage(count_dict):
    """
    Compute the percentage of values in a dictionary that are nonzero.
    """
    if not count_dict:
        return 0.0
    
    total_items = len(count_dict)
    nonzero_items = sum(1 for value in count_dict.values() if value != 0)
    
    return (nonzero_items / total_items) # * 100


def compute_corrected_cysteine_l2fc_vs_ms(
    df_dict,
    ms_key,
    peptide_col,
    allele_col,
    label_col,
    smoothing=False,
    pseudocount_num=0.5,
    pseudocount_den=1.0,
    deduplicate=True,
):
    """
    Compute baseline-corrected per-allele cysteine enrichment relative to MS.
    """

    if ms_key not in df_dict:
        raise ValueError(f"ms_key '{ms_key}' not found in df_dict.")

    required_cols = {peptide_col, allele_col, label_col}
    for name, df in df_dict.items():
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset '{name}' missing required columns: {sorted(missing)}")

    def _prep_df(df):
        out = df[[allele_col, peptide_col, label_col]].copy()
        out = out.dropna(subset=[allele_col, peptide_col, label_col])
        out[allele_col] = out[allele_col].astype(str)
        out[peptide_col] = out[peptide_col].astype(str).str.upper()
        out[label_col] = out[label_col].astype(int)

        out = out[out[label_col].isin([0, 1])].copy()

        if deduplicate:
            out = out.drop_duplicates(subset=[allele_col, peptide_col, label_col])

        return out

    def _compute_prop(n_cys, n_total):
        if smoothing:
            return (n_cys + pseudocount_num) / (n_total + pseudocount_den)
        return n_cys / n_total

    def _allele_label_summary(df):
        tmp = df.copy()
        tmp["has_cys"] = tmp[peptide_col].str.contains("C", regex=False)

        summ = (
            tmp.groupby([allele_col, label_col], dropna=False)
            .agg(
                n_total=(peptide_col, "size"),
                n_cys=("has_cys", "sum"),
            )
            .reset_index()
        )

        bind = (
            summ[summ[label_col] == 1]
            .drop(columns=[label_col])
            .rename(columns={"n_total": "bind_n_total", "n_cys": "bind_n_cys"})
        )

        bg = (
            summ[summ[label_col] == 0]
            .drop(columns=[label_col])
            .rename(columns={"n_total": "bg_n_total", "n_cys": "bg_n_cys"})
        )

        merged = bind.merge(bg, on=allele_col, how="inner")

        if merged.empty:
            return merged

        merged["bind_prop_cys"] = _compute_prop(merged["bind_n_cys"], merged["bind_n_total"])
        merged["bg_prop_cys"] = _compute_prop(merged["bg_n_cys"], merged["bg_n_total"])

        with np.errstate(divide="ignore", invalid="ignore"):
            merged["within_dataset_cys_enrichment_l2fc"] = np.log2(
                merged["bind_prop_cys"] / merged["bg_prop_cys"]
            )

        return merged

    ms_df = _prep_df(df_dict[ms_key])
    ms_summ = _allele_label_summary(ms_df).rename(
        columns={
            "bind_n_total": "ms_bind_n_total",
            "bind_n_cys": "ms_bind_n_cys",
            "bind_prop_cys": "ms_bind_prop_cys",
            "bg_n_total": "ms_bg_n_total",
            "bg_n_cys": "ms_bg_n_cys",
            "bg_prop_cys": "ms_bg_prop_cys",
            "within_dataset_cys_enrichment_l2fc": "ms_within_dataset_cys_enrichment_l2fc",
        }
    )

    results = []

    for dataset_name, df in df_dict.items():
        if dataset_name == ms_key:
            continue

        ext_df = _prep_df(df)
        ext_summ = _allele_label_summary(ext_df).rename(
            columns={
                "bind_n_total": "ext_bind_n_total",
                "bind_n_cys": "ext_bind_n_cys",
                "bind_prop_cys": "ext_bind_prop_cys",
                "bg_n_total": "ext_bg_n_total",
                "bg_n_cys": "ext_bg_n_cys",
                "bg_prop_cys": "ext_bg_prop_cys",
                "within_dataset_cys_enrichment_l2fc": "ext_within_dataset_cys_enrichment_l2fc",
            }
        )

        merged = ms_summ.merge(ext_summ, on=allele_col, how="inner")
        if merged.empty:
            continue

        merged["dataset"] = dataset_name

        with np.errstate(divide="ignore", invalid="ignore"):
            merged["cys_l2fc_ext_vs_ms"] = (
                merged["ext_within_dataset_cys_enrichment_l2fc"]
                - merged["ms_within_dataset_cys_enrichment_l2fc"]
            )

        merged["matched_allele"] = True

        merged = merged[
            [
                "dataset",
                allele_col,

                "ms_bind_n_total",
                "ms_bind_n_cys",
                "ms_bind_prop_cys",
                "ms_bg_n_total",
                "ms_bg_n_cys",
                "ms_bg_prop_cys",
                "ms_within_dataset_cys_enrichment_l2fc",

                "ext_bind_n_total",
                "ext_bind_n_cys",
                "ext_bind_prop_cys",
                "ext_bg_n_total",
                "ext_bg_n_cys",
                "ext_bg_prop_cys",
                "ext_within_dataset_cys_enrichment_l2fc",

                "cys_l2fc_ext_vs_ms",
                "matched_allele",
            ]
        ].copy()

        results.append(merged)

    if not results:
        return pd.DataFrame(
            columns=[
                "dataset",
                allele_col,
                "ms_bind_n_total",
                "ms_bind_n_cys",
                "ms_bind_prop_cys",
                "ms_bg_n_total",
                "ms_bg_n_cys",
                "ms_bg_prop_cys",
                "ms_within_dataset_cys_enrichment_l2fc",
                "ext_bind_n_total",
                "ext_bind_n_cys",
                "ext_bind_prop_cys",
                "ext_bg_n_total",
                "ext_bg_n_cys",
                "ext_bg_prop_cys",
                "ext_within_dataset_cys_enrichment_l2fc",
                "cys_l2fc_ext_vs_ms",
                "matched_allele",
            ]
        )

    out_df = pd.concat(results, ignore_index=True)
    return out_df


def summarize_cysteine_l2fc_output(
    cys_l2fc_df: pd.DataFrame,
    dataset_col: str = "dataset",
    allele_col: str = "allele",
    l2fc_col: str = "cys_l2fc_ext_vs_ms",
):
    required_cols = {dataset_col, allele_col, l2fc_col}
    missing = required_cols - set(cys_l2fc_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    per_dataset_dict = {}
    for ds, subdf in cys_l2fc_df.groupby(dataset_col, sort=False):
        per_dataset_dict[ds] = (
            subdf[[allele_col, l2fc_col]]
            .copy()
            .reset_index(drop=True)
        )

    summary_df = (
        cys_l2fc_df
        .groupby(dataset_col, sort=False)
        .agg(
            mean_l2fc_cys=(l2fc_col, "mean"),
            median_l2fc_cys=(l2fc_col, "median"),
            n_matched_alleles=(allele_col, "nunique"),
        )
        .reset_index()
    )

    return per_dataset_dict, summary_df

import pandas as pd
import numpy as np


def compute_delta_gravy_vs_ms(
    df_dict,
    ms_key,
    peptide_col,
    allele_col,
    deduplicate=True,
):
    """
    Compute per-allele delta GRAVY (external - MS) for matched alleles.
    """
    if ms_key not in df_dict:
        raise ValueError(f"ms_key '{ms_key}' not found in df_dict.")

    required_cols = {peptide_col, allele_col}
    for name, df in df_dict.items():
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset '{name}' missing required columns: {sorted(missing)}")

    kd_scale = {
        "I": 4.5,
        "V": 4.2,
        "L": 3.8,
        "F": 2.8,
        "C": 2.5,
        "M": 1.9,
        "A": 1.8,
        "G": -0.4,
        "T": -0.7,
        "S": -0.8,
        "W": -0.9,
        "Y": -1.3,
        "P": -1.6,
        "H": -3.2,
        "E": -3.5,
        "Q": -3.5,
        "D": -3.5,
        "N": -3.5,
        "K": -3.9,
        "R": -4.5,
    }

    def _prep_df(df):
        out = df[[allele_col, peptide_col]].copy()
        out = out.dropna(subset=[allele_col, peptide_col])
        out[allele_col] = out[allele_col].astype(str)
        out[peptide_col] = out[peptide_col].astype(str).str.upper()

        if deduplicate:
            out = out.drop_duplicates(subset=[allele_col, peptide_col])

        return out

    def _compute_peptide_gravy(seq):
        vals = [kd_scale[aa] for aa in seq if aa in kd_scale]
        if len(vals) != len(seq):
            return np.nan
        return float(np.mean(vals))

    def _allele_gravy_summary(df):
        tmp = df.copy()
        tmp["peptide_gravy"] = tmp[peptide_col].map(_compute_peptide_gravy)
        tmp = tmp.dropna(subset=["peptide_gravy"])

        summ = (
            tmp.groupby(allele_col, dropna=False)
            .agg(
                n_peptides=(peptide_col, "size"),
                mean_gravy=("peptide_gravy", "mean"),
            )
            .reset_index()
        )
        return summ

    ms_df = _prep_df(df_dict[ms_key])
    ms_summ = _allele_gravy_summary(ms_df).rename(
        columns={
            "n_peptides": "ms_n_peptides",
            "mean_gravy": "ms_mean_gravy",
        }
    )

    per_dataset_dict = {}
    summary_rows = []

    for dataset_name, df in df_dict.items():
        if dataset_name == ms_key:
            continue

        ext_df = _prep_df(df)
        ext_summ = _allele_gravy_summary(ext_df).rename(
            columns={
                "n_peptides": "ext_n_peptides",
                "mean_gravy": "ext_mean_gravy",
            }
        )

        merged = ms_summ.merge(ext_summ, on=allele_col, how="inner")
        if merged.empty:
            per_dataset_dict[dataset_name] = pd.DataFrame(
                columns=[allele_col, "ms_mean_gravy", "ext_mean_gravy", "delta_gravy_ext_vs_ms"]
            )
            summary_rows.append({
                "dataset": dataset_name,
                "mean_delta_gravy": np.nan,
                "median_delta_gravy": np.nan,
                "n_matched_alleles": 0,
            })
            continue

        merged["delta_gravy_ext_vs_ms"] = (
            merged["ext_mean_gravy"] - merged["ms_mean_gravy"]
        )

        out_df = merged[
            [allele_col, "ms_mean_gravy", "ext_mean_gravy", "delta_gravy_ext_vs_ms"]
        ].copy()

        per_dataset_dict[dataset_name] = out_df

        summary_rows.append({
            "dataset": dataset_name,
            "mean_delta_gravy": out_df["delta_gravy_ext_vs_ms"].mean(),
            "median_delta_gravy": out_df["delta_gravy_ext_vs_ms"].median(),
            "n_matched_alleles": out_df[allele_col].nunique(),
        })

    summary_df = pd.DataFrame(summary_rows)

    return per_dataset_dict, summary_df