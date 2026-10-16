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

def average_score_columns_by_prefix(df,prefix_map,drop_original=True,suffix="_rank_r_chp"):
    """
    Average score columns row-wise by prefix, optionally requiring a suffix.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    prefix_map : dict[str,str]
        {new_column_name: column_prefix}

    drop_original : bool
        If True, remove the original matched score columns after averaging.

    suffix : str or None
        If provided, only columns that start with the specified prefix AND
        end with this suffix are averaged. Default="_rank_r_chp".
        Set to None to match columns by prefix only.

    Returns
    -------
    pd.DataFrame
        Copy of df with one averaged score column per prefix-map entry.
    """
    out=df.copy()
    matched_cols=[]

    for new_name,prefix in prefix_map.items():
        cols=[
            c for c in out.columns
            if str(c).startswith(prefix) and (suffix is None or str(c).endswith(suffix))
        ]

        if not cols:
            suffix_msg=f" and suffix '{suffix}'" if suffix is not None else ""
            raise ValueError(
                f"No columns found for '{new_name}' with prefix '{prefix}'{suffix_msg}"
            )

        out[new_name]=out[cols].mean(axis=1)
        matched_cols.extend(cols)

    if drop_original:
        matched_cols=list(dict.fromkeys(matched_cols))
        matched_cols=[c for c in matched_cols if c not in prefix_map]
        out=out.drop(columns=matched_cols)

    return out

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
    print(8, len(seq_overlap)) # all overlaps are resolved beforehand or based on the grouping of genes and structure of testing.

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
    
def filter_single_df_to_cols(df, cols):
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()

def lr_to_strings(lrs):
    out = []
    for lr in lrs:
        mant, exp = f"{lr:e}".split("e") # force scientific: '5.000000', '-05'
        mant = mant.rstrip("0").rstrip(".") # trim trailing zeros: '5', '1', '2'
        exp_int = int(exp) # -5, -4, ...
        if exp_int < 0:
            exp_str = f"N{abs(exp_int)}"
        else:
            exp_str = str(exp_int)
        out.append(f"{mant}e{exp_str}")
    return out


def save_log_dict(data_dict, path):
    with open(path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_log_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# NOTE: make sure this function does not exist in utils
def sample_by_allele_len_group(df, pct, min_samples, max_samples, ratio=99, allele_col="allele", seq_len_col="seq_len", label_col="label", seed=0):
    rng = np.random.default_rng(seed)
    work = df.reset_index(drop=True)
    n = len(work)

    is_pos = (work[label_col] == 1).to_numpy()
    is_neg = (work[label_col] == 0).to_numpy()
    pos_global = np.where(is_pos)[0]
    neg_global = np.where(is_neg)[0]
    pos_groups = work.loc[is_pos].groupby([allele_col, seq_len_col], sort=False).indices
    neg_groups = work.loc[is_neg].groupby([allele_col, seq_len_col], sort=False).indices

    sampled_positions = []
    for key, local_pos in tqdm(pos_groups.items(), desc="Sampling groups", leave=True):
        gpos = pos_global[local_pos]
        n_pos_avail = len(gpos)
        want_pos = int(round(pct * n_pos_avail))
        if want_pos < min_samples:
            continue                       # N% falls below the minimum -> skip group entirely
        want_pos = min(want_pos, max_samples, n_pos_avail)
        if want_pos <= 0:
            continue
        chosen_pos = gpos[rng.permutation(n_pos_avail)[:want_pos]]

        want_neg = want_pos * ratio
        if key in neg_groups:
            gneg = neg_global[neg_groups[key]]
            n_neg_avail = len(gneg)
            take_neg = min(want_neg, n_neg_avail)
            chosen_neg = gneg[rng.permutation(n_neg_avail)[:take_neg]]
        else:
            chosen_neg = np.array([], dtype=np.int64)

        sampled_positions.append(chosen_pos)
        sampled_positions.append(chosen_neg)

    samp_pos = np.concatenate(sampled_positions) if sampled_positions else np.array([], dtype=np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[samp_pos] = True
    return work.iloc[np.where(~mask)[0]].copy(), work.iloc[np.where(mask)[0]].copy()

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

def compute_holec_gain_ms_loss(
    ms_df: pd.DataFrame,
    holec_df: pd.DataFrame,
    base_col: str,
    score_col: str,
) -> pd.DataFrame:
    """
    Compute per-allele Holec/YD gain and MS loss using the exact
    definitions from compute_net_adaptation_benefit().

    holec_gain = Holec_score - Holec_base
    ms_loss    = max(0, MS_base - MS_score)

    Returns
    -------
    pd.DataFrame
        Indexed by allele with columns:
        - holec_gain
        - ms_loss
    """
    required_ms = {base_col, score_col}
    required_holec = {base_col, score_col}

    missing_ms = required_ms - set(ms_df.columns)
    missing_holec = required_holec - set(holec_df.columns)

    if missing_ms:
        raise ValueError(
            f"ms_df is missing required columns: {sorted(missing_ms)}"
        )
    if missing_holec:
        raise ValueError(
            f"holec_df is missing required columns: {sorted(missing_holec)}"
        )

    common_alleles = ms_df.index.intersection(holec_df.index)

    if len(common_alleles) == 0:
        raise ValueError(
            "No overlapping allele indices found between ms_df and holec_df."
        )

    ms_sub = ms_df.loc[common_alleles, [base_col, score_col]].copy()
    holec_sub = holec_df.loc[common_alleles, [base_col, score_col]].copy()

    ms_sub = ms_sub.apply(pd.to_numeric, errors="coerce")
    holec_sub = holec_sub.apply(pd.to_numeric, errors="coerce")

    # Exact same definitions as net adaptation benefit
    holec_gain = holec_sub[score_col] - holec_sub[base_col]
    ms_loss = (ms_sub[base_col] - ms_sub[score_col]).clip(lower=0)

    out_df = pd.DataFrame({
        "holec_gain": holec_gain,
        "ms_loss": ms_loss,
    })

    out_df.index.name = ms_df.index.name

    return out_df

import re
import numpy as np
import pandas as pd


def compute_grouped_holec_gain_ms_loss(
    ms_df: pd.DataFrame,
    holec_df: pd.DataFrame,
    base_col: str,
    model_groups: dict = None,
    prefix_groups: dict = None,
    format_alleles: bool = False,
    add_allele_col: bool = True,
    verbose: bool = True,
):
    """
    Compute seed/model-averaged per-allele Holec/YD gain and MS loss.

    For every individual score column:

        holec_gain = Holec_score - Holec_base
        ms_loss    = max(0, MS_base - MS_score)

    The gain and loss components are computed FIRST for each individual
    score column, then averaged across columns belonging to the same
    final model group.

    Parameters
    ----------
    ms_df : pd.DataFrame
        Per-allele MS performance dataframe.

    holec_df : pd.DataFrame
        Per-allele Holec/YD performance dataframe.

    base_col : str
        Shared pretrained/base model column.

    model_groups : dict or None
        Explicit mapping:

            {
                final_name: [score_col_1, score_col_2, ...],
                ...
            }

        Example:
            {
                "holec_ft_steps_300_lr_2eN4": [
                    "holec_ft_steps_300_lr_2eN4_seed_0",
                    "holec_ft_steps_300_lr_2eN4_seed_1",
                    ...
                ]
            }

    prefix_groups : dict or None
        Alternative mapping:

            {
                final_name: prefix,
                ...
            }

        Example:
            {
                "holec_ft_steps_300_lr_2eN4":
                    "holec_ft_steps_300_lr_2eN4",
                "holec_pepcl_r_dr_steps_300_lr_2eN4":
                    "holec_pepcl_r_dr_steps_300_lr_2eN4",
            }

        Prefix matching selects exact final seeded models of the form:

            {prefix}_seed_<seed>

        and intentionally excludes checkpoint columns such as:
            {prefix}_seed_0_step_100

    format_alleles : bool
        If True, convert allele indices such as A0201 -> A*02:01.

    add_allele_col : bool
        If True, add an 'allele' column containing the final index.

    verbose : bool
        Print grouped columns used for each model.

    Returns
    -------
    dict
        {
            final_name: pd.DataFrame(
                index=allele,
                columns=["holec_gain", "ms_loss", "allele"]
            ),
            ...
        }
    """
    if model_groups is None and prefix_groups is None:
        raise ValueError("Provide model_groups and/or prefix_groups.")

    if base_col not in ms_df.columns:
        raise ValueError(f"ms_df is missing base column {base_col!r}.")
    if base_col not in holec_df.columns:
        raise ValueError(f"holec_df is missing base column {base_col!r}.")

    # ------------------------------------------------------------
    # Build final_name -> score columns mapping
    # ------------------------------------------------------------
    groups={}

    if model_groups is not None:
        for final_name,cols in model_groups.items():
            if isinstance(cols,str):
                cols=[cols]
            groups[final_name]=list(cols)

    if prefix_groups is not None:
        all_cols=set(ms_df.columns).intersection(holec_df.columns)

        for final_name,prefix in prefix_groups.items():
            # Match only final seeded models, not intermediate checkpoints.
            pattern=re.compile(
                rf"^{re.escape(prefix)}_seed_([^_]+)$"
            )

            matched=sorted(
                [c for c in all_cols if pattern.match(str(c))],
                key=str,
            )

            if not matched:
                raise ValueError(
                    f"{final_name!r}: no shared final seeded columns found "
                    f"for prefix {prefix!r}."
                )

            if final_name in groups:
                raise ValueError(
                    f"{final_name!r} was supplied in both model_groups "
                    f"and prefix_groups."
                )

            groups[final_name]=matched

    # ------------------------------------------------------------
    # Common allele set
    # ------------------------------------------------------------
    common_alleles=ms_df.index.intersection(holec_df.index)

    if len(common_alleles)==0:
        raise ValueError(
            "No overlapping allele indices found between ms_df and holec_df."
        )

    ms_base=pd.to_numeric(
        ms_df.loc[common_alleles,base_col],
        errors="coerce",
    )

    holec_base=pd.to_numeric(
        holec_df.loc[common_alleles,base_col],
        errors="coerce",
    )

    results={}

    # ------------------------------------------------------------
    # Compute components per column FIRST, then average
    # ------------------------------------------------------------
    for final_name,score_cols in groups.items():
        if not score_cols:
            raise ValueError(
                f"{final_name!r} has no score columns."
            )

        missing_ms=[c for c in score_cols if c not in ms_df.columns]
        missing_holec=[c for c in score_cols if c not in holec_df.columns]

        if missing_ms:
            raise ValueError(
                f"{final_name!r}: ms_df is missing columns: {missing_ms}"
            )
        if missing_holec:
            raise ValueError(
                f"{final_name!r}: holec_df is missing columns: {missing_holec}"
            )

        gain_cols=[]
        loss_cols=[]

        for score_col in score_cols:
            ms_score=pd.to_numeric(
                ms_df.loc[common_alleles,score_col],
                errors="coerce",
            )

            holec_score=pd.to_numeric(
                holec_df.loc[common_alleles,score_col],
                errors="coerce",
            )

            # Exact original definitions, independently per seed/model.
            holec_gain=holec_score-holec_base
            ms_loss=(ms_base-ms_score).clip(lower=0)

            gain_cols.append(holec_gain.rename(score_col))
            loss_cols.append(ms_loss.rename(score_col))

        gain_df=pd.concat(gain_cols,axis=1)
        loss_df=pd.concat(loss_cols,axis=1)

        out_df=pd.DataFrame(
            {
                "holec_gain":gain_df.mean(axis=1,skipna=True),
                "ms_loss":loss_df.mean(axis=1,skipna=True),
            },
            index=common_alleles,
        )

        out_df.index.name=ms_df.index.name

        if format_alleles:
            def _format_allele(a):
                a=str(a)
                return f"{a[0]}*{a[1:3]}:{a[3:]}"
            out_df.index=out_df.index.map(_format_allele)

        if add_allele_col:
            out_df["allele"]=out_df.index

        results[final_name]=out_df

        if verbose:
            print(
                f"{final_name}: averaged {len(score_cols)} columns"
            )
            for c in score_cols:
                print(f"  {c}")

    return results

def compute_brnb(
    ms_df: pd.DataFrame,
    assay_df: pd.DataFrame,
    base_col: str,
    score_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Compute per-allele BRNB:

    assay_available_gain_fraction =
        (assay_score - assay_base) / (1 - assay_base)

    ms_fraction_lost =
        (ms_base - ms_score) / ms_base

    BRNB =
        assay_available_gain_fraction - ms_fraction_lost
    """
    required_ms={base_col,score_col}
    required_assay={base_col,score_col}

    missing_ms=required_ms-set(ms_df.columns)
    missing_assay=required_assay-set(assay_df.columns)

    if missing_ms:
        raise ValueError(
            f"ms_df is missing required columns: {sorted(missing_ms)}"
        )
    if missing_assay:
        raise ValueError(
            f"assay_df is missing required columns: {sorted(missing_assay)}"
        )

    common_alleles=ms_df.index.intersection(assay_df.index)

    if len(common_alleles)==0:
        raise ValueError(
            "No overlapping allele indices found between ms_df and assay_df."
        )

    ms_sub=ms_df.loc[
        common_alleles,
        [base_col,score_col]
    ].copy()

    assay_sub=assay_df.loc[
        common_alleles,
        [base_col,score_col]
    ].copy()

    ms_sub=ms_sub.apply(
        pd.to_numeric,
        errors="coerce"
    )

    assay_sub=assay_sub.apply(
        pd.to_numeric,
        errors="coerce"
    )

    with np.errstate(divide="ignore",invalid="ignore"):
        assay_available_gain_fraction=(
            assay_sub[score_col]-assay_sub[base_col]
        )/(1.0-assay_sub[base_col])

        ms_fraction_lost=(
            ms_sub[base_col]-ms_sub[score_col]
        )/ms_sub[base_col]

        brnb=(
            assay_available_gain_fraction -
            ms_fraction_lost
        )

    brnb=brnb.replace(
        [np.inf,-np.inf],
        np.nan
    )

    out_df=pd.DataFrame({
        output_col:brnb
    })

    out_df.index.name=ms_df.index.name

    return out_df

def average_models_by_prefix(df, prefix_map, model_col="model_name", score_col="recovered"):
    matched = set()
    rows = []
    for final_name, prefix in prefix_map.items():
        mask = df[model_col].astype(str).str.startswith(prefix)
        if mask.any():
            matched.update(df.loc[mask, model_col].tolist())
            rows.append({model_col: final_name, score_col: df.loc[mask, score_col].mean()})
    averaged = pd.DataFrame(rows)
    leftover = df[~df[model_col].isin(matched)].copy()
    return pd.concat([averaged, leftover[[model_col, score_col]]], ignore_index=True)


def scores_by_prefix_wide(df, prefix_map, model_col="model_name", score_col="recovered"):
    cols = {}
    for final_name, prefix in prefix_map.items():
        mask = df[model_col].astype(str).str.startswith(prefix)
        cols[final_name] = df.loc[mask, score_col].to_numpy()
    return pd.DataFrame(cols)

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

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def compute_reference_pvalues_with_correction(
    df,
    reference_col,
    correction="fdr_bh",
    alpha=0.05,
):
    """
    Compare every non-reference column against a predefined reference
    using paired Wilcoxon signed-rank tests, then correct the resulting
    p-values as one hypothesis family.

    Parameters
    ----------
    df : pd.DataFrame
        Rows should represent matched observations, e.g. alleles.
        Columns represent models.
    reference_col : str
        Column used as the predefined reference model.
    correction : str or None
        Multiple-testing correction passed to statsmodels
        multipletests. Use "fdr_bh" for Benjamini-Hochberg FDR.
        If None, only raw p-values are returned.
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        Comparison, raw p-value, adjusted p-value, and significance.
    """

    rows = []

    for col in df.columns:
        if col == reference_col:
            continue

        pair = df[[reference_col, col]].dropna()

        stat, pval = wilcoxon(
            pair[reference_col],
            pair[col],
        )

        rows.append({
            "reference": reference_col,
            "comparison": col,
            "statistic": stat,
            "p_raw": pval,
        })

    results = pd.DataFrame(rows)

    if correction is not None:
        reject, p_adj, _, _ = multipletests(
            results["p_raw"].values,
            alpha=alpha,
            method=correction,
        )

        results["p_adj"] = p_adj
        results["significant"] = reject
    else:
        results["p_adj"] = results["p_raw"]
        results["significant"] = results["p_raw"] < alpha

    return results

def update_pval_matrix_fdr(pval_matrix, corrected_pvals, reference="mhcprime"):
    out = pval_matrix.copy()

    for _, row in corrected_pvals.iterrows():
        model = row["comparison"]
        out.loc[reference, model] = row["p_adj"]
        out.loc[model, reference] = row["p_adj"]

    out.loc[reference, reference] = 1.0
    return out

def fdr_correct_3x3_pval_matrix(pval_matrix, method="fdr_bh", alpha=0.05):
    """
    Apply multiple-testing correction to the 3 unique pairwise p-values
    in a symmetric 3x3 p-value matrix, then return the same matrix format.
    """
    out = pval_matrix.copy()
    cols = out.columns.tolist()

    # Extract the 3 unique pairwise comparisons
    pairs = [(0, 1), (0, 2), (1, 2)]
    pvals = [out.iloc[i, j] for i, j in pairs]

    # Adjust together as one family
    pvals_adj = multipletests(pvals, method=method, alpha=alpha)[1]

    # Write adjusted values back symmetrically
    for (i, j), p_adj in zip(pairs, pvals_adj):
        out.iloc[i, j] = p_adj
        out.iloc[j, i] = p_adj

    # Keep diagonal at 1
    for i in range(3):
        out.iloc[i, i] = 1.0

    return out

from statsmodels.stats.multitest import multipletests
import pandas as pd


def fdr_correct_selected_comparisons(
    pval_matrix,
    comparisons,
    alpha=0.05,
    method="fdr_bh",
):
    """
    Apply multiple-testing correction to selected comparisons in a
    symmetric p-value matrix.

    Parameters
    ----------
    pval_matrix : pd.DataFrame
        Symmetric pairwise p-value matrix.

    comparisons : dict
        Mapping:
            comparison_name -> [column_1, column_2]

        Example:
        {
            "FT CON vs FT CLS": ["ba_ft_con", "ba_ft_cls"],
            "PepCL CON vs PepCL CLS": ["ba_pepcl_con", "ba_pepcl_cls"],
        }

    alpha : float
        Significance level passed to multipletests.

    method : str
        Multiple-testing method. Default is Benjamini-Hochberg FDR.

    Returns
    -------
    revised_matrix : pd.DataFrame
        Original matrix with only the selected comparison entries
        replaced by adjusted p-values.

    results_df : pd.DataFrame
        Columns:
            comparison
            pval
            pval_adj
    """
    revised_matrix = pval_matrix.copy()

    names = []
    pairs = []
    pvals = []

    for comparison, pair in comparisons.items():
        col1, col2 = pair

        pval = pval_matrix.loc[col1, col2]

        names.append(comparison)
        pairs.append((col1, col2))
        pvals.append(pval)

    pvals_adj = multipletests(
        pvals,
        alpha=alpha,
        method=method,
    )[1]

    for (col1, col2), p_adj in zip(pairs, pvals_adj):
        revised_matrix.loc[col1, col2] = p_adj
        revised_matrix.loc[col2, col1] = p_adj

    results_df = pd.DataFrame({
        "comparison": names,
        "pval": pvals,
        "pval_adj": pvals_adj,
    })

    return revised_matrix, results_df

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


import re
import numpy as np
import pandas as pd
from scipy.stats import t


def summarize_paired_seed_metric_delta(
    metric_dict,
    ft_prefix,
    pepcl_prefix,
    confidence=0.95,
    require_matching_seeds=True,
):
    """
    Compute paired FT - PepCL deltas per seed, then summarize across seeds.

    Parameters
    ----------
    metric_dict : dict[str, pd.DataFrame]
        Output from compute_topk_position_entropy_by_allele() or
        compute_topk_position_jsd_by_allele().

        Each value must be an allele x position dataframe.

    ft_prefix : str
        Model prefix excluding `_seed_N`.

        Example:
            "holec_ft_steps_300_lr_2eN4"

    pepcl_prefix : str
        Model prefix excluding `_seed_N`.

        Example:
            "holec_pepcl_r_dr_steps_300_lr_2eN4"

    confidence : float
        Student-t confidence interval level. Default 0.95.

    require_matching_seeds : bool
        If True, FT and PepCL must contain exactly the same seeds.

    Returns
    -------
    result : dict
        {
            "delta_by_seed": {seed: DataFrame},
            "mean": DataFrame,
            "ci_low": DataFrame,
            "ci_high": DataFrame,
            "sd": DataFrame,
            "se": DataFrame,
            "n": DataFrame,
            "seeds": [...]
        }

        All summary DataFrames are allele x position.
    """
    def _get_seed_map(prefix):
        pattern=re.compile(rf"^{re.escape(prefix)}_seed_(\d+)$")
        out={}

        for name in metric_dict:
            m=pattern.match(str(name))
            if m is not None:
                seed=int(m.group(1))
                if seed in out:
                    raise ValueError(f"Duplicate seed {seed} found for prefix '{prefix}'.")
                out[seed]=name

        return out

    ft_map=_get_seed_map(ft_prefix)
    pepcl_map=_get_seed_map(pepcl_prefix)

    if not ft_map:
        raise ValueError(f"No models found for FT prefix: {ft_prefix}")
    if not pepcl_map:
        raise ValueError(f"No models found for PepCL prefix: {pepcl_prefix}")

    ft_seeds=set(ft_map)
    pepcl_seeds=set(pepcl_map)

    if require_matching_seeds and ft_seeds != pepcl_seeds:
        raise ValueError(
            f"FT/PepCL seeds do not match.\n"
            f"FT only: {sorted(ft_seeds-pepcl_seeds)}\n"
            f"PepCL only: {sorted(pepcl_seeds-ft_seeds)}"
        )

    seeds=sorted(ft_seeds & pepcl_seeds)
    if not seeds:
        raise ValueError("No matched FT/PepCL seeds found.")

    delta_by_seed={}

    for seed in seeds:
        ft=metric_dict[ft_map[seed]]
        pepcl=metric_dict[pepcl_map[seed]]

        if not ft.index.equals(pepcl.index):
            raise ValueError(f"Allele indices differ between FT and PepCL for seed {seed}.")
        if not ft.columns.equals(pepcl.columns):
            raise ValueError(f"Position columns differ between FT and PepCL for seed {seed}.")

        delta_by_seed[seed]=ft-pepcl

    # shape: seeds x alleles x positions
    arr=np.stack([delta_by_seed[s].to_numpy(dtype=float) for s in seeds],axis=0)

    n=np.sum(np.isfinite(arr),axis=0)
    mean=np.nanmean(arr,axis=0)

    # sample SD across paired seed-level deltas
    with np.errstate(invalid="ignore",divide="ignore"):
        sd=np.nanstd(arr,axis=0,ddof=1)
        se=sd/np.sqrt(n)

    crit=np.full(n.shape,np.nan,dtype=float)
    valid_n=n>=2
    crit[valid_n]=t.ppf((1+confidence)/2,df=n[valid_n]-1)

    ci_low=mean-crit*se
    ci_high=mean+crit*se

    template=delta_by_seed[seeds[0]]

    def _df(x):
        return pd.DataFrame(x,index=template.index,columns=template.columns)

    return {
        "delta_by_seed":delta_by_seed,
        "mean":_df(mean),
        "ci_low":_df(ci_low),
        "ci_high":_df(ci_high),
        "sd":_df(sd),
        "se":_df(se),
        "n":_df(n),
        "seeds":seeds,
    }

import numpy as np
import pandas as pd
from scipy.stats import t

def average_columns_by_prefix_with_ci(df, prefix_map, ci_level=0.95, min_replicates=2):
    """
    Average seeded score columns by prefix within each allele row and compute
    Student-t confidence intervals across the matched replicate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = alleles, columns = score/model columns.

    prefix_map : dict
        {final_name: prefix}

        Example:
        {
            "episcan_r9_ft": "episcan_r9_ft_steps_300_lr_2eN4",
            "episcan_r9_pepcl": "episcan_r9_pepcl_r_dr_steps_300_lr_2eN4",
        }

    ci_level : float
        Confidence level, default 0.95.

    min_replicates : int
        Minimum number of non-NaN replicate values required to compute a CI.

    Returns
    -------
    avg_df : pd.DataFrame
        Same allele index.
        Prefix-matched seed columns are replaced by one averaged column.
        Unmatched columns are retained unchanged.

    ci_df : pd.DataFrame
        Same allele index, with:
            <final_name>_ci_low
            <final_name>_ci_high

        Only averaged prefix groups receive CI columns.
    """
    if not 0<ci_level<1:
        raise ValueError("ci_level must be in (0,1).")
    if min_replicates<2:
        raise ValueError("min_replicates must be >= 2.")

    matched_cols=set()
    averaged={}
    ci_data={}

    for final_name,prefix in prefix_map.items():
        cols=[c for c in df.columns if str(c).startswith(prefix)]

        if not cols:
            continue

        matched_cols.update(cols)

        vals=df[cols].apply(pd.to_numeric,errors="coerce")

        n=vals.notna().sum(axis=1)
        mean=vals.mean(axis=1)
        sd=vals.std(axis=1,ddof=1)
        se=sd/np.sqrt(n)

        crit=pd.Series(
            np.where(
                n>=min_replicates,
                t.ppf((1+ci_level)/2,n-1),
                np.nan
            ),
            index=df.index
        )

        margin=crit*se

        averaged[final_name]=mean
        ci_data[f"{final_name}_ci_low"]=mean-margin
        ci_data[f"{final_name}_ci_high"]=mean+margin

    # Preserve every column not consumed by a prefix group
    leftover=df[[c for c in df.columns if c not in matched_cols]].copy()

    averaged_df=pd.DataFrame(averaged,index=df.index)
    avg_df=pd.concat([leftover,averaged_df],axis=1)

    ci_df=pd.DataFrame(ci_data,index=df.index)

    return avg_df,ci_df

import re
import numpy as np
import pandas as pd
from scipy.stats import t


def average_seeded_long_by_prefix_allele_with_ci(
    df,
    prefix_map,
    model_col="model_name",
    value_col="p8_enrich",
    confidence=0.95,
    retain_unmatched=True,
    allele_pattern=r"_([abc]\d{4})$",
    seed_pattern=r"_seed_(\d+)",
):
    """
    Average long-format seeded values by model prefix AND allele.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe containing model names and one value column.

        Example model names:
            episcan_r9rt_ft_steps_400_lr_2eN4_seed_0_a0201
            episcan_r9rt_ft_steps_400_lr_2eN4_seed_1_a0201
            episcan_r9rt_pepcl_r_dr_steps_300_lr_2eN4_seed_0_b0801
            episcan_r9rt_log_odds_a0201

    prefix_map : dict
        Mapping:
            output model prefix -> input seeded model prefix

        Example:
            {
                "episcan_r9rt_ft":
                    "episcan_r9rt_ft_steps_400_lr_2eN4",

                "episcan_r9rt_pepcl":
                    "episcan_r9rt_pepcl_r_dr_steps_300_lr_2eN4",
            }

        Thus:
            input:
                episcan_r9rt_ft_steps_400_lr_2eN4_seed_0_a0201
                ...
            output:
                episcan_r9rt_ft_a0201

    model_col : str
        Column containing model names.

    value_col : str
        Numeric value column to average.

    confidence : float
        Confidence level for Student-t CI. Default 0.95.

    retain_unmatched : bool
        If True, rows not matching any supplied seeded prefix are retained
        unchanged in the mean dataframe. Their CI/statistics are NaN except n=1.

    allele_pattern : str
        Regex identifying the terminal allele suffix.

    seed_pattern : str
        Regex identifying seed number.

    Returns
    -------
    mean_df : pd.DataFrame
        Same long-format structure as the input:
            model_name | <value_col>

        Seeded rows are collapsed across seeds separately for each allele.
        Unmatched rows are retained if retain_unmatched=True.

    ci_df : pd.DataFrame
        Interval/statistics dataframe directly compatible with
        plot_model_performance_bars_or_box_v3.

        Columns are model names and rows are:
            ci_low
            ci_high
            sd
            se
            n

        Example:
                    model_A   model_B
            ci_low    ...
            ci_high   ...
            sd        ...
            se        ...
            n         ...

        Retained unmatched rows have NaN uncertainty.
    """

    required={model_col,value_col}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work=df.copy()
    work[value_col]=pd.to_numeric(work[value_col],errors="coerce")

    matched=np.zeros(len(work),dtype=bool)
    grouped_rows=[]

    # Identify seeded rows belonging to each prefix and allele
    for output_prefix,input_prefix in prefix_map.items():
        for idx,row in work.iterrows():
            name=str(row[model_col])

            if not name.startswith(input_prefix):
                continue

            # Require a seed marker so an unseeded summary/reference row is
            # not accidentally absorbed into the replicate average.
            seed_match=re.search(seed_pattern,name,re.IGNORECASE)
            allele_match=re.search(allele_pattern,name,re.IGNORECASE)

            if seed_match is None or allele_match is None:
                continue

            allele=allele_match.group(1).lower()

            grouped_rows.append({
                "_original_index":idx,
                "_output_name":f"{output_prefix}_{allele}",
                "_allele":allele,
                "_seed":int(seed_match.group(1)),
                "_value":row[value_col],
            })

            matched[work.index.get_loc(idx)]=True

    grouped=pd.DataFrame(grouped_rows)

    mean_rows=[]
    ci_rows=[]

    # Aggregate seed replicates within model × allele
    if not grouped.empty:
        for output_name,g in grouped.groupby("_output_name",sort=False):

            # If duplicate rows somehow exist for one seed, fail rather than
            # silently overweight that seed.
            duplicate_seeds=g["_seed"].duplicated(keep=False)

            if duplicate_seeds.any():
                bad=sorted(g.loc[duplicate_seeds,"_seed"].unique())
                raise ValueError(
                    f"Duplicate seed rows found for '{output_name}': {bad}"
                )

            vals=pd.to_numeric(g["_value"],errors="coerce").dropna().to_numpy(float)
            n=len(vals)
            mean=np.mean(vals) if n else np.nan

            if n>=2:
                sd=np.std(vals,ddof=1)
                se=sd/np.sqrt(n)
                crit=t.ppf((1+confidence)/2,df=n-1)
                ci_low=mean-crit*se
                ci_high=mean+crit*se
            else:
                sd=se=ci_low=ci_high=np.nan

            mean_rows.append({
                model_col:output_name,
                value_col:mean,
            })

            ci_rows.append({
                model_col:output_name,
                "ci_low":ci_low,
                "ci_high":ci_high,
                "sd":sd,
                "se":se,
                "n":n,
            })

    # Retain rows not belonging to any supplied seeded prefix
    if retain_unmatched:
        unmatched=work.loc[~matched,[model_col,value_col]].copy()

        mean_rows.extend(unmatched.to_dict("records"))

        for _,row in unmatched.iterrows():
            ci_rows.append({
                model_col:row[model_col],
                "ci_low":np.nan,
                "ci_high":np.nan,
                "sd":np.nan,
                "se":np.nan,
                "n":1,
            })

    mean_df=pd.DataFrame(mean_rows,columns=[model_col,value_col])

    # Format directly for plot_model_performance_bars_or_box_v3:
    # rows = interval/statistic names, columns = model names.
    if ci_rows:
        ci_df=pd.DataFrame(ci_rows).set_index(model_col)[
            ["ci_low","ci_high","sd","se","n"]
        ].T
    else:
        ci_df=pd.DataFrame(index=["ci_low","ci_high","sd","se","n"])

    return mean_df,ci_df

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


"""
New functions
"""

def seq_allele_overlap_table(train_dfs, test_dfs, seq_col="seq", allele_col="allele", remove_test_from_train=False):
    datasets = {}
    for name, df in train_dfs.items():
        datasets[f"{name}_train"] = df
    for name, df in test_dfs.items():
        datasets[f"{name}_test"] = df

    pair_sets = {}
    for name, df in tqdm(datasets.items(), desc="Building pair sets", leave=True):
        pairs = set(map(tuple, df[[seq_col, allele_col]].to_numpy()))
        pair_sets[name] = pairs

    names = list(pair_sets.keys())
    table = pd.DataFrame(index=names, columns=names, dtype=int)
    for a in tqdm(names, desc="Computing overlaps", leave=True):
        sa = pair_sets[a]
        for b in names:
            table.loc[a, b] = len(sa & pair_sets[b])

    if not remove_test_from_train:
        return table

    # union of all test (seq, allele) pairs across every test df
    all_test_pairs = set()
    for name, df in test_dfs.items():
        all_test_pairs |= set(map(tuple, df[[seq_col, allele_col]].to_numpy()))

    filtered_train_dfs = {}
    for name, df in tqdm(train_dfs.items(), desc="Filtering train sets", leave=True):
        pairs = list(map(tuple, df[[seq_col, allele_col]].to_numpy()))
        mask = np.array([p not in all_test_pairs for p in pairs], dtype=bool)
        filtered_train_dfs[name] = df.loc[mask].copy()

    return table, filtered_train_dfs

def validate_test_alleles_in_train(train_dfs, test_dfs, allele_col="allele", raise_on_fail=False):
    rows = []
    shared_keys = [k for k in test_dfs if k in train_dfs]
    for key in tqdm(shared_keys, desc="Validating alleles", leave=True):
        train_alleles = set(train_dfs[key][allele_col].unique())
        test_alleles = set(test_dfs[key][allele_col].unique())
        missing = sorted(test_alleles - train_alleles)
        rows.append({
            "dataset": key,
            "n_test_alleles": len(test_alleles),
            "n_train_alleles": len(train_alleles),
            "n_missing_in_train": len(missing),
            "all_present": len(missing) == 0,
            "missing_alleles": missing,
        })
    report = pd.DataFrame(rows)
    if raise_on_fail and not report.empty and not report["all_present"].all():
        bad = report.loc[~report["all_present"], ["dataset", "missing_alleles"]]
        raise ValueError(f"Test alleles missing from train:\n{bad.to_string(index=False)}")
    return report

def allele_overlap_table(train_dfs, test_dfs, allele_col="allele"):
    datasets = {}
    for name, df in train_dfs.items():
        datasets[f"{name}_train"] = df
    for name, df in test_dfs.items():
        datasets[f"{name}_test"] = df
    allele_sets = {}
    for name, df in tqdm(datasets.items(), desc="Building allele sets", leave=True):
        allele_sets[name] = set(df[allele_col].unique())
    names = list(allele_sets.keys())
    table = pd.DataFrame(index=names, columns=names, dtype=int)
    for a in tqdm(names, desc="Computing allele overlaps", leave=True):
        sa = allele_sets[a]
        for b in names:
            table.loc[a, b] = len(sa & allele_sets[b])
    return table

def filter_dict_columns(df_dict, columns):
    out = {}
    for k, df in df_dict.items():
        keep = [c for c in columns if c in df.columns]
        out[k] = df[keep].copy()
    return out


import os
import pandas as pd

# def get_remaining_hyperparameter_combinations(
#     df_path, step_budgets, learning_rate_strings, seeds, method_prefixes, verbose=True
# ):
#     """
#     Identify completed and remaining hyperparameter combinations from model-score
#     columns already present in a saved CSV.

#     Expected loop order:
#         seed -> step budget -> learning rate

#     Expected full-model naming:
#         {method_prefix}_steps_{steps}_lr_{lr_string}_seed_{seed}

#     Parameters
#     ----------
#     df_path : str
#         Path to saved CSV containing model score columns.

#     step_budgets : list[int]
#         Example: [25, 50, 100, 200, 400, 800]

#     learning_rate_strings : list[str]
#         Example: ["2eN5", "5eN5", "2eN4"]

#     seeds : list[int]
#         Example: [0, 1, 2, 3, 4]

#     method_prefixes : dict or list[str]
#         Either:
#             {"ft":"holec_ft", "pepcl_r_dr":"holec_pepcl_r_dr"}
#         or:
#             ["holec_ft", "holec_pepcl_r_dr"]

#         A combination is considered complete only if the full-model column
#         exists for EVERY supplied method prefix.

#     Returns
#     -------
#     remaining : list[tuple]
#         [(seed, steps, learning_rate), ...] in the exact natural loop order.

#     completed : list[tuple]
#         Completed combinations in the same loop order.
#     """

#     if not os.path.exists(df_path):
#         raise FileNotFoundError(f"Dataframe not found: {df_path}")

#     # We only need the header, not the entire potentially huge dataframe.
#     cols=set(pd.read_csv(df_path,nrows=0).columns)

#     if isinstance(method_prefixes,dict):
#         prefixes=list(method_prefixes.values())
#     else:
#         prefixes=list(method_prefixes)

#     prefixes=[str(p).rstrip("_") for p in prefixes]

#     def _lr_string_to_float(s):
#         # 2eN5 -> 2e-5, 8eN4 -> 8e-4
#         return float(str(s).replace("eN","e-").replace("EN","e-"))

#     lr_pairs=[
#         (lr_string,_lr_string_to_float(lr_string))
#         for lr_string in learning_rate_strings
#     ]

#     # Exact ordering of your original nested loops:
#     # seed -> steps -> learning rate
#     all_combinations=[
#         (seed,steps,lr_value,lr_string)
#         for seed in seeds
#         for steps in step_budgets
#         for lr_string,lr_value in lr_pairs
#     ]

#     completed=[]
#     remaining=[]
#     completion_details={}

#     for seed,steps,lr_value,lr_string in all_combinations:
#         expected_cols=[
#             f"{prefix}_steps_{steps}_lr_{lr_string}_seed_{seed}"
#             for prefix in prefixes
#         ]

#         present=[c for c in expected_cols if c in cols]
#         missing=[c for c in expected_cols if c not in cols]

#         combo=(seed,steps,lr_value)
#         completion_details[combo]={
#             "learning_rate_string":lr_string,
#             "expected_columns":expected_cols,
#             "present_columns":present,
#             "missing_columns":missing,
#             "complete":len(missing)==0,
#         }

#         if not missing:
#             completed.append(combo)
#         else:
#             remaining.append(combo)

#     total=len(all_combinations)

#     if verbose:
#         print("="*80)
#         print("HYPERPARAMETER RESUME CHECK")
#         print("="*80)
#         print(f"Dataframe: {df_path}")
#         print(f"Methods required: {prefixes}")
#         print(f"Seeds: {seeds}")
#         print(f"Step budgets: {step_budgets}")
#         print(f"Learning rates: {[x[1] for x in lr_pairs]}")
#         print()
#         print(f"Completed: {len(completed)}/{total}")
#         print(f"Remaining: {len(remaining)}/{total}")

#         if completed:
#             seed,steps,lr=completed[-1]
#             print(
#                 f"\nLast completed combination in loop order:\n"
#                 f"  seed={seed}, steps={steps}, learning_rate={lr}"
#             )
#         else:
#             print("\nNo completed combinations found.")

#         if remaining:
#             seed,steps,lr=remaining[0]
#             print(
#                 f"\nNext combination to run:\n"
#                 f"  seed={seed}, steps={steps}, learning_rate={lr}"
#             )
#         else:
#             print("\nAll requested combinations are complete.")

#         # Detect holes, e.g. an incomplete combination followed by completed ones.
#         complete_flags=[
#             completion_details[(seed,steps,lr_value)]["complete"]
#             for seed,steps,lr_value,_ in all_combinations
#         ]

#         if False in complete_flags:
#             first_missing=complete_flags.index(False)
#             if any(complete_flags[first_missing+1:]):
#                 print(
#                     "\nWARNING: completed combinations exist after an incomplete "
#                     "combination in the expected loop order. This is fine for resume, "
#                     "but indicates the saved results are not one contiguous prefix."
#                 )

#     return remaining, completed

import os
import pandas as pd

def get_remaining_hyperparameter_combinations(
    df_path,
    step_budgets,
    learning_rate_strings,
    seeds,
    method_prefixes,
    verbose=True
):
    """
    Determine which hyperparameter combinations remain unfinished based on
    model-score columns already present in one or more CSV files.

    Parameters
    ----------
    df_path : str, os.PathLike, or list/tuple of str/os.PathLike
        Either:
          - one CSV path, or
          - multiple CSV paths.

        When multiple paths are provided, their column names are unioned before
        determining completion. Rows/indexes do not need to match between files.

    step_budgets : iterable of int
        Total training-step budgets.

    learning_rate_strings : iterable of str
        Learning-rate strings as encoded in model names, e.g.
        ["2eN5", "5eN5", "1eN4", "2eN4"].

    seeds : iterable of int
        Seeds to test.

    method_prefixes : dict or iterable
        If dict, values are treated as score-column prefixes:
            {
                "ft": "holec_ft_",
                "pepcl": "holec_pepcl_r_dr_"
            }

        An iterable of prefixes is also accepted.

    verbose : bool
        Print completion information.

    Returns
    -------
    remaining : list of tuple
        (seed, steps, learning_rate_float) combinations still requiring work.

    completed : list of tuple
        (seed, steps, learning_rate_float) combinations already complete.

    Notes
    -----
    A hyperparameter combination is complete only when the expected FULL-model
    score column exists for every method prefix somewhere across the supplied
    CSV file(s).

    Expected full-model naming:
        {prefix}steps_{steps}_lr_{lr_string}_seed_{seed}

    Final checkpoint columns such as:
        ..._step_{steps}
    do not substitute for the explicit full-model column here.
    """

    # ------------------------------------------------------------------
    # Normalize one path -> list of paths
    # ------------------------------------------------------------------
    if isinstance(df_path,(str,os.PathLike)):
        paths=[df_path]
    elif isinstance(df_path,(list,tuple)):
        if not df_path:
            raise ValueError("df_path list cannot be empty.")
        paths=list(df_path)
    else:
        raise TypeError(
            "df_path must be a file path or a list/tuple of file paths."
        )

    # ------------------------------------------------------------------
    # Method prefixes
    # ------------------------------------------------------------------
    if isinstance(method_prefixes,dict):
        prefixes=list(method_prefixes.values())
    else:
        prefixes=list(method_prefixes)

    if not prefixes:
        raise ValueError("method_prefixes cannot be empty.")

    # ------------------------------------------------------------------
    # LR conversion
    # ------------------------------------------------------------------
    def _lr_string_to_float(lr):
        return float(str(lr).replace("eN","e-"))

    # ------------------------------------------------------------------
    # Read headers only and union all available score columns
    # ------------------------------------------------------------------
    all_columns=set()
    columns_by_path={}

    for path in paths:
        path=os.fspath(path)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        cols=pd.read_csv(path,nrows=0).columns.tolist()
        columns_by_path[path]=set(cols)
        all_columns.update(cols)

    # ------------------------------------------------------------------
    # Determine completion in natural loop order:
    # seed -> step budget -> LR
    # ------------------------------------------------------------------
    remaining=[]
    completed=[]

    for seed in seeds:
        for steps in step_budgets:
            for lr_string in learning_rate_strings:
                lr_float=_lr_string_to_float(lr_string)

                expected_cols=[
                    f"{prefix}steps_{steps}_lr_{lr_string}_seed_{seed}"
                    for prefix in prefixes
                ]

                is_complete=all(
                    col in all_columns
                    for col in expected_cols
                )

                combo=(seed,steps,lr_float)

                if is_complete:
                    completed.append(combo)
                else:
                    remaining.append(combo)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    if verbose:
        total=len(completed)+len(remaining)

        print("="*80)
        print("HYPERPARAMETER COMPLETION SUMMARY")
        print("="*80)

        print(f"CSV files checked: {len(paths)}")
        for path in paths:
            print(
                f"  {path}: "
                f"{len(columns_by_path[os.fspath(path)])} columns"
            )

        print(f"\nUnique columns across all files: {len(all_columns)}")
        print(f"Completed combinations: {len(completed)}/{total}")
        print(f"Remaining combinations: {len(remaining)}/{total}")

        if completed:
            seed,steps,lr=completed[-1]
            print(
                f"\nLast completed combination in loop order: "
                f"seed={seed}, steps={steps}, lr={lr:g}"
            )

        if remaining:
            seed,steps,lr=remaining[0]
            print(
                f"Next remaining combination in loop order: "
                f"seed={seed}, steps={steps}, lr={lr:g}"
            )
        else:
            print("\nAll hyperparameter combinations are complete.")

        # Detect holes: completed combinations appearing after an unfinished one.
        all_combos=[
            (seed,steps,_lr_string_to_float(lr))
            for seed in seeds
            for steps in step_budgets
            for lr in learning_rate_strings
        ]

        status=[combo in completed for combo in all_combos]

        if False in status:
            first_missing=status.index(False)

            if any(status[first_missing+1:]):
                print(
                    "\nWARNING: completion contains holes: at least one later "
                    "combination is complete after an earlier incomplete combination."
                )

    return remaining,completed

def lr_to_string(lr):
    mantissa, exponent = f"{float(lr):.12e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)
    return f"{mantissa}eN{abs(exponent)}" if exponent < 0 else f"{mantissa}e{exponent}"

from scipy.stats import pearsonr, spearmanr

def compare_score_cols(df, base_col, rapid_col):
    x=df[base_col].to_numpy(float)
    y=df[rapid_col].to_numpy(float)
    d=y-x
    ad=abs(d)

    return {
        "base_mean":x.mean(), "rapid_mean":y.mean(),
        "base_median":np.median(x), "rapid_median":np.median(y),
        "base_min":x.min(), "rapid_min":y.min(),
        "base_max":x.max(), "rapid_max":y.max(),
        "abs_diff_mean":ad.mean(), "abs_diff_min":ad.min(), "abs_diff_max":ad.max(),
        "pearson":pearsonr(x,y).statistic,
        "spearman":spearmanr(x,y).statistic,
    }

from scipy.stats import t

def average_models_by_prefix_with_ci(
    df,
    prefix_map,
    model_col="model_name",
    score_col="recovered",
    ci=0.95,
    ci_low_row="ci_low",
    ci_high_row="ci_high",
):
    """
    Average replicate-level scores grouped by model-name prefix and compute
    Student-t confidence intervals across the matched replicate values.

    Returns
    -------
    averaged_df : pd.DataFrame
        Same basic format as average_models_by_prefix():
            model_col, score_col

        Prefix-matched replicate models are replaced by one averaged row.
        Unmatched rows are retained unchanged.

    interval_df : pd.DataFrame
        Columns = final averaged model names
        Rows    = ci_low / ci_high

        Only prefix-grouped models with >=2 finite replicate values are included.
        This format is directly compatible with the v3 plotting function.
    """
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0,1).")

    matched=set()
    rows=[]
    intervals={}

    for final_name,prefix in prefix_map.items():
        mask=df[model_col].astype(str).str.startswith(prefix)
        vals=pd.to_numeric(df.loc[mask,score_col],errors="coerce").dropna().to_numpy(float)

        if len(vals)==0:
            continue

        matched.update(df.loc[mask,model_col].tolist())

        mean=float(np.mean(vals))
        rows.append({
            model_col:final_name,
            score_col:mean,
        })

        # CI requires at least two replicate observations.
        if len(vals)>=2:
            sd=float(np.std(vals,ddof=1))
            se=sd/np.sqrt(len(vals))
            crit=float(t.ppf((1+ci)/2,len(vals)-1))

            intervals[final_name]={
                ci_low_row:mean-crit*se,
                ci_high_row:mean+crit*se,
            }

    averaged=pd.DataFrame(rows)

    leftover=df[
        ~df[model_col].isin(matched)
    ][[model_col,score_col]].copy()

    averaged_df=pd.concat(
        [averaged,leftover],
        ignore_index=True
    )

    interval_df=pd.DataFrame(intervals)

    # Preserve expected row order even if no intervals were available.
    interval_df=interval_df.reindex(
        [ci_low_row,ci_high_row]
    )

    return averaged_df,interval_df

import re
import numpy as np
import pandas as pd
from scipy.stats import t

def compute_seeded_net_adaptation_benefit(
    ms_df,
    assay_df,
    base_col,
    prefix_map,
    ci_level=0.95,
    min_replicates=2,
    verbose=True,
):
    """
    Compute per-allele net adaptation benefit separately for every replicate,
    then summarize across replicates with a Student-t confidence interval.

    Net benefit within replicate:
        assay_gain = assay_score - assay_base
        ms_loss    = max(0, ms_base - ms_score)
        net_benefit = assay_gain - ms_loss

    Parameters
    ----------
    ms_df, assay_df : pd.DataFrame
        Per-allele AP dataframes. Rows are alleles and model columns contain
        replicate-specific scores.

    base_col : str
        Reference column, e.g. "mhcprime".

    prefix_map : dict
        Mapping from desired final output column name to exact model prefix
        before `_seed_N`.

        Example:
        {
            "ft_net_benefit":
                "holec_ft_steps_300_lr_2eN4",
            "pepcl_net_benefit":
                "holec_pepcl_r_dr_steps_200_lr_2eN4",
        }

    Returns
    -------
    mean_df : pd.DataFrame
        Per-allele mean net benefit across replicates.

    ci_df : pd.DataFrame
        Per-allele confidence interval endpoints with columns:
            <output_col>_ci_low
            <output_col>_ci_high
    """
    if not 0<ci_level<1:
        raise ValueError("ci_level must be in (0,1).")
    if min_replicates<2:
        raise ValueError("min_replicates must be >= 2.")
    if base_col not in ms_df.columns:
        raise ValueError(f"ms_df is missing base_col '{base_col}'.")
    if base_col not in assay_df.columns:
        raise ValueError(f"assay_df is missing base_col '{base_col}'.")

    common_alleles=ms_df.index.intersection(assay_df.index)
    if len(common_alleles)==0:
        raise ValueError("No overlapping allele indices between ms_df and assay_df.")

    ms_base=pd.to_numeric(ms_df.loc[common_alleles,base_col],errors="coerce")
    assay_base=pd.to_numeric(assay_df.loc[common_alleles,base_col],errors="coerce")

    mean_df=pd.DataFrame(index=common_alleles)
    ci_df=pd.DataFrame(index=common_alleles)

    for output_col,prefix in prefix_map.items():
        pattern=re.compile(rf"^{re.escape(prefix)}_seed_(\d+)$")

        matches=[]
        for col in ms_df.columns:
            m=pattern.match(str(col))
            if m is not None and col in assay_df.columns:
                matches.append((int(m.group(1)),col))

        matches=sorted(matches,key=lambda x:x[0])

        if not matches:
            raise ValueError(
                f"No shared MS/assay columns found for prefix '{prefix}'. "
                "Expected columns ending in '_seed_N'."
            )

        seeds=[seed for seed,_ in matches]
        if len(seeds)!=len(set(seeds)):
            raise ValueError(
                f"Prefix '{prefix}' matched duplicate replicate IDs: {seeds}"
            )

        replicate_values={}

        for seed,col in matches:
            ms_score=pd.to_numeric(
                ms_df.loc[common_alleles,col],
                errors="coerce"
            )

            assay_score=pd.to_numeric(
                assay_df.loc[common_alleles,col],
                errors="coerce"
            )

            assay_gain=assay_score-assay_base
            ms_loss=(ms_base-ms_score).clip(lower=0)

            replicate_values[seed]=assay_gain-ms_loss

        rep_df=pd.DataFrame(
            replicate_values,
            index=common_alleles
        )

        n=rep_df.notna().sum(axis=1)
        mean=rep_df.mean(axis=1)
        sd=rep_df.std(axis=1,ddof=1)
        se=sd/np.sqrt(n)

        crit=pd.Series(
            np.where(
                n>=min_replicates,
                t.ppf((1+ci_level)/2,n-1),
                np.nan
            ),
            index=common_alleles
        )

        margin=crit*se

        mean_df[output_col]=mean
        ci_df[f"{output_col}_ci_low"]=mean-margin
        ci_df[f"{output_col}_ci_high"]=mean+margin

        if verbose:
            print(
                f"{output_col}: prefix='{prefix}', "
                f"replicates={seeds}, n={len(seeds)}"
            )

    mean_df.index.name=ms_df.index.name
    ci_df.index.name=ms_df.index.name

    return mean_df,ci_df

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_mixed_ms_holec_ft_pepcl_delta_from_base(
    ms_df: pd.DataFrame,
    holec_df: pd.DataFrame,
    *,
    base_col: str,
    ft_prefix: str,
    pepcl_prefix: str,
    ft_output_col: str="holec_ft",
    pepcl_output_col: str="holec_pepcl",
    allele_col: str="allele",
    ms_label_col: str="label",
    holec_label_col: str="label",
    neg_per_pos_ratio: int=99,
    mix_step_pct: float=10.0,
    n_iterations: int=100,
    interval_pct: float=95.0,
    require_same_replicates: bool=True,
    random_state: int=42,
    show_progress: bool=True,
):
    """
    Mix MS and Holec/YD test sets and compute delta-mAP relative to MHCPrime.

    For each MS/YD mixture percentage:
      1. Draw one peptide mixture.
      2. Evaluate that exact mixture with all FT and PepCL replicates.
      3. Compute delta-mAP vs base separately for every replicate.
      4. Average replicate values within that mixture realization.
      5. Repeat for n_iterations independently sampled mixtures.
      6. Plotting mean = mean across mixture realizations.
      7. Interval = empirical central interval across mixture realizations.

    Thus the returned interval characterizes variation due to mixture
    composition after averaging across model replicates.

    Returns
    -------
    mean_df
        Columns = ft_output_col, pepcl_output_col.

    var_df
        Variance across seed-averaged mixture realizations.

    ci_df
        Empirical interval endpoints, named for direct compatibility
        with plot_lines():
            <col>_ci_low
            <col>_ci_high
    """

    def _validate_required_columns(df, required, name):
        missing=[c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    def _find_seed_cols(df, prefix):
        pattern=re.compile(rf"^{re.escape(prefix)}_seed_(\d+)$")
        out={}
        for col in df.columns:
            m=pattern.match(str(col))
            if m:
                seed=int(m.group(1))
                if seed in out:
                    raise ValueError(f"Duplicate seed {seed} found for prefix '{prefix}'.")
                out[seed]=col
        return dict(sorted(out.items()))

    def _fast_average_precision_multi(scores, labels):
        if scores.ndim!=2:
            raise ValueError("scores must be 2D.")
        if labels.ndim!=1:
            raise ValueError("labels must be 1D.")

        n_pos=int(labels.sum())
        n_cols=scores.shape[1]
        out=np.full(n_cols, np.nan, dtype=np.float64)

        if n_pos==0:
            return out

        y0=labels.astype(np.uint8, copy=False)

        for j in range(n_cols):
            order=np.argsort(-scores[:,j], kind="mergesort")
            y=y0[order]
            pos_idx=np.flatnonzero(y)

            if pos_idx.size==0:
                continue

            tp=np.cumsum(y, dtype=np.int64)
            out[j]=(tp[pos_idx]/(pos_idx+1.0)).mean()

        return out

    def _build_source_arrays_by_allele(df, label_col, score_cols):
        use_cols=[allele_col,label_col]+score_cols
        d=df[use_cols].copy()

        d[allele_col]=d[allele_col].astype(str)
        d[label_col]=pd.to_numeric(d[label_col], errors="coerce").astype(np.int8)

        for col in score_cols:
            d[col]=pd.to_numeric(d[col], errors="coerce").astype(np.float32)

        out={}

        for allele,sub in d.groupby(allele_col, sort=False):
            out[str(allele)]={
                "pos":sub.loc[sub[label_col]==1,score_cols].to_numpy(dtype=np.float32,copy=True),
                "neg":sub.loc[sub[label_col]==0,score_cols].to_numpy(dtype=np.float32,copy=True),
            }

        return out

    if neg_per_pos_ratio<0:
        raise ValueError("neg_per_pos_ratio must be >= 0.")
    if not 0<mix_step_pct<=100:
        raise ValueError("mix_step_pct must be in (0,100].")
    if n_iterations<=0:
        raise ValueError("n_iterations must be > 0.")
    if not 0<interval_pct<100:
        raise ValueError("interval_pct must be in (0,100).")

    # --------------------------------------------------------------
    # Seeded columns
    # --------------------------------------------------------------
    ft_ms=_find_seed_cols(ms_df, ft_prefix)
    ft_holec=_find_seed_cols(holec_df, ft_prefix)
    pep_ms=_find_seed_cols(ms_df, pepcl_prefix)
    pep_holec=_find_seed_cols(holec_df, pepcl_prefix)

    ft_seeds=sorted(set(ft_ms)&set(ft_holec))
    pep_seeds=sorted(set(pep_ms)&set(pep_holec))

    if not ft_seeds:
        raise ValueError(f"No shared FT seed columns found for prefix '{ft_prefix}'.")
    if not pep_seeds:
        raise ValueError(f"No shared PepCL seed columns found for prefix '{pepcl_prefix}'.")

    if require_same_replicates and ft_seeds!=pep_seeds:
        raise ValueError(f"FT and PepCL replicate IDs differ.\nFT: {ft_seeds}\nPepCL: {pep_seeds}")

    replicate_ids=ft_seeds if require_same_replicates else sorted(set(ft_seeds)&set(pep_seeds))
    if not replicate_ids:
        raise ValueError("FT and PepCL have no shared replicate IDs.")

    for seed in replicate_ids:
        if ft_ms[seed]!=ft_holec[seed]:
            raise ValueError(f"FT seed {seed} column differs between MS and Holec.")
        if pep_ms[seed]!=pep_holec[seed]:
            raise ValueError(f"PepCL seed {seed} column differs between MS and Holec.")

    ft_cols=[ft_ms[s] for s in replicate_ids]
    pep_cols=[pep_ms[s] for s in replicate_ids]
    all_score_cols=[base_col]+ft_cols+pep_cols

    _validate_required_columns(ms_df,[allele_col,ms_label_col]+all_score_cols,"ms_df")
    _validate_required_columns(holec_df,[allele_col,holec_label_col]+all_score_cols,"holec_df")

    if show_progress:
        print(f"FT prefix: {ft_prefix}")
        print(f"PepCL prefix: {pepcl_prefix}")
        print(f"Replicates: {replicate_ids}")
        print(f"Mixture iterations: {n_iterations}")
        print(f"Empirical interval: {interval_pct:g}%")

    # --------------------------------------------------------------
    # Compact per-allele arrays
    # --------------------------------------------------------------
    ms_map=_build_source_arrays_by_allele(ms_df, ms_label_col, all_score_cols)
    holec_map=_build_source_arrays_by_allele(holec_df, holec_label_col, all_score_cols)

    common_alleles=sorted(set(ms_map)&set(holec_map))
    if not common_alleles:
        raise ValueError("No overlapping alleles between ms_df and holec_df.")

    allele_plan={}
    skipped={}

    for allele in common_alleles:
        ms_pos=ms_map[allele]["pos"]; ms_neg=ms_map[allele]["neg"]
        holec_pos=holec_map[allele]["pos"]; holec_neg=holec_map[allele]["neg"]

        min_pos=min(ms_pos.shape[0],holec_pos.shape[0])
        if min_pos<1:
            skipped[allele]="min_pos_lt_1"
            continue

        total_neg=min_pos*neg_per_pos_ratio

        if ms_neg.shape[0]<total_neg:
            skipped[allele]="insufficient_ms_negatives"
            continue
        if holec_neg.shape[0]<total_neg:
            skipped[allele]="insufficient_holec_negatives"
            continue

        allele_plan[allele]={
            "min_pos":int(min_pos),
            "total_neg":int(total_neg),
            "ms_pos":ms_pos,
            "ms_neg":ms_neg,
            "holec_pos":holec_pos,
            "holec_neg":holec_neg,
        }

    usable_alleles=sorted(allele_plan)

    if not usable_alleles:
        raise ValueError("No usable alleles remain after sampling constraints.")

    if show_progress:
        print(f"Usable alleles: {len(usable_alleles)} / {len(common_alleles)}")

    # --------------------------------------------------------------
    # Mixture percentages
    # --------------------------------------------------------------
    ms_percentages=np.arange(100.0,-1e-9,-float(mix_step_pct))
    ms_percentages=np.round(ms_percentages,10)

    if ms_percentages[-1]!=0:
        ms_percentages=np.append(ms_percentages,0.0)

    n_mix=len(ms_percentages)
    n_rep=len(replicate_ids)

    ft_mean=np.full(n_mix,np.nan)
    pep_mean=np.full(n_mix,np.nan)
    ft_var=np.full(n_mix,np.nan)
    pep_var=np.full(n_mix,np.nan)
    ft_low=np.full(n_mix,np.nan)
    ft_high=np.full(n_mix,np.nan)
    pep_low=np.full(n_mix,np.nan)
    pep_high=np.full(n_mix,np.nan)

    rng=np.random.default_rng(random_state)

    ft_slice=slice(1,1+n_rep)
    pep_slice=slice(1+n_rep,1+2*n_rep)

    alpha=(100.0-interval_pct)/2.0
    lower_pct=alpha
    upper_pct=100.0-alpha

    # --------------------------------------------------------------
    # Mixture experiment
    # --------------------------------------------------------------
    outer=tqdm(
        enumerate(ms_percentages),
        total=n_mix,
        desc="MS/YD mixtures",
        leave=True,
        disable=not show_progress,
    )

    for mix_idx,ms_pct in outer:
        ms_frac=ms_pct/100.0

        # Rows = mixture realizations, columns = model replicates.
        ft_iter=np.full((n_iterations,n_rep),np.nan,dtype=np.float64)
        pep_iter=np.full((n_iterations,n_rep),np.nan,dtype=np.float64)

        inner=tqdm(
            range(n_iterations),
            desc=f"{ms_pct:g}% MS",
            leave=True,
            disable=not show_progress,
        )

        for it in inner:
            ap_by_allele=np.full(
                (len(usable_alleles),len(all_score_cols)),
                np.nan,
                dtype=np.float64
            )

            for allele_idx,allele in enumerate(usable_alleles):
                plan=allele_plan[allele]
                min_pos=plan["min_pos"]
                total_neg=plan["total_neg"]

                ms_pos_take=int(np.rint(min_pos*ms_frac))
                ms_pos_take=min(max(ms_pos_take,0),min_pos)
                holec_pos_take=min_pos-ms_pos_take

                ms_neg_take=int(np.rint(total_neg*ms_frac))
                ms_neg_take=min(max(ms_neg_take,0),total_neg)
                holec_neg_take=total_neg-ms_neg_take

                parts=[]
                labels_parts=[]

                if ms_pos_take>0:
                    arr=plan["ms_pos"]
                    idx=rng.choice(arr.shape[0],size=ms_pos_take,replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.ones(ms_pos_take,dtype=np.uint8))

                if holec_pos_take>0:
                    arr=plan["holec_pos"]
                    idx=rng.choice(arr.shape[0],size=holec_pos_take,replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.ones(holec_pos_take,dtype=np.uint8))

                if ms_neg_take>0:
                    arr=plan["ms_neg"]
                    idx=rng.choice(arr.shape[0],size=ms_neg_take,replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.zeros(ms_neg_take,dtype=np.uint8))

                if holec_neg_take>0:
                    arr=plan["holec_neg"]
                    idx=rng.choice(arr.shape[0],size=holec_neg_take,replace=False)
                    parts.append(arr[idx])
                    labels_parts.append(np.zeros(holec_neg_take,dtype=np.uint8))

                if not parts:
                    continue

                scores=np.vstack(parts)
                labels=np.concatenate(labels_parts)

                ap_by_allele[allele_idx]=_fast_average_precision_multi(scores,labels)

            map_values=np.nanmean(ap_by_allele,axis=0)
            base_map=map_values[0]

            ft_iter[it]=map_values[ft_slice]-base_map
            pep_iter[it]=map_values[pep_slice]-base_map

        # ----------------------------------------------------------
        # Average model replicates WITHIN each mixture realization.
        # Each of these arrays now has length n_iterations.
        # ----------------------------------------------------------
        ft_mix_values=np.nanmean(ft_iter,axis=1)
        pep_mix_values=np.nanmean(pep_iter,axis=1)

        # Mean curve.
        ft_mean[mix_idx]=np.nanmean(ft_mix_values)
        pep_mean[mix_idx]=np.nanmean(pep_mix_values)

        # Variance among mixture realizations.
        ft_var[mix_idx]=np.nanvar(ft_mix_values,ddof=1)
        pep_var[mix_idx]=np.nanvar(pep_mix_values,ddof=1)

        # Empirical central mixture-resampling interval.
        ft_low[mix_idx],ft_high[mix_idx]=np.nanpercentile(
            ft_mix_values,[lower_pct,upper_pct]
        )

        pep_low[mix_idx],pep_high[mix_idx]=np.nanpercentile(
            pep_mix_values,[lower_pct,upper_pct]
        )

    # --------------------------------------------------------------
    # Plot-ready outputs
    # --------------------------------------------------------------
    mean_df=pd.DataFrame(
        {
            ft_output_col:ft_mean,
            pepcl_output_col:pep_mean,
        },
        index=ms_percentages
    )
    mean_df.index.name="ms_percentage"

    var_df=pd.DataFrame(
        {
            ft_output_col:ft_var,
            pepcl_output_col:pep_var,
        },
        index=ms_percentages
    )
    var_df.index.name="ms_percentage"

    # Keep *_ci_low/high names for compatibility with plot_lines(),
    # although these are empirical mixture-resampling intervals.
    ci_df=pd.DataFrame(
        {
            f"{ft_output_col}_ci_low":ft_low,
            f"{ft_output_col}_ci_high":ft_high,
            f"{pepcl_output_col}_ci_low":pep_low,
            f"{pepcl_output_col}_ci_high":pep_high,
        },
        index=ms_percentages
    )
    ci_df.index.name="ms_percentage"

    if show_progress:
        print("Finished mixture analysis.")
        print(
            f"Each plotted point averages {n_iterations} independently sampled mixtures; "
            f"each mixture value first averages {n_rep} model replicates."
        )

    return mean_df,var_df,ci_df


import glob

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

def flatten_dict_keys_and_values(d):
    return list(d.keys()) + [x for v in d.values() for x in v]

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

def average_all_seed_groups(df, keep_non_seed=True):
    pattern = re.compile(r"^(.*)_seed_\d+$")
    groups = defaultdict(list)
    seed_cols = []
    for c in df.columns:
        m = pattern.match(str(c))
        if m:
            groups[m.group(1)].append(c)
            seed_cols.append(c)
    out = df.copy()
    for prefix, cols in groups.items():
        out[prefix] = out[cols].mean(axis=1)
    out = out.drop(columns=seed_cols)
    if not keep_non_seed:
        out = out[list(groups.keys())]
    return out

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

def merge_keep_first_dups(df1, df2, key_cols=("seq","allele","label"), how="outer"):
    key_cols = list(key_cols)
    dup_cols = [c for c in df2.columns if c in df1.columns and c not in key_cols]
    df2_trimmed = df2.drop(columns=dup_cols)
    return df1.merge(df2_trimmed, on=key_cols, how=how)

def keep_present_cols(df, score_cols):
    return [c for c in score_cols if c in df.columns]

def summarize_seeded_ap_per_allele(
    ap_df,
    prefix_map,
    ci=0.95,
    ci_method="t",
    n_boot=10000,
    random_state=42,
    suffix=None,
    drop_original=True,
):
    """
    Average seeded model AP columns per allele and compute per-allele CIs.

    Any columns not matched by prefix_map are preserved unchanged in mean_df.

    Parameters
    ----------
    ap_df : pd.DataFrame
        Rows = alleles
        Columns = model score columns
        Values = AP per allele.

    prefix_map : dict
        output_model_name -> seeded column prefix

        Example:
            {
                "FT":"holec_ft_steps_300_lr_2eN4_seed_",
                "PepCL":"holec_pepcl_r_dr_steps_300_lr_2eN4_seed_",
            }

    ci_method : {"t","bootstrap"}

    suffix : str or None
        Optional required suffix for columns matched to a prefix.

    drop_original : bool
        If True, remove individual seeded columns after replacing them
        with the averaged model column. All unmatched columns are retained.

    Returns
    -------
    mean_df : pd.DataFrame
        Original unmatched columns + per-allele averaged model columns.

    ci_df : pd.DataFrame
        Per-allele CI columns:
            <model>_ci_low
            <model>_ci_high
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import t

    if ci_method not in {"t","bootstrap"}:
        raise ValueError("ci_method must be 't' or 'bootstrap'.")

    rng=np.random.default_rng(random_state)

    # Start from full AP dataframe so all unrelated columns are retained.
    mean_df=ap_df.copy()
    ci_df=pd.DataFrame(index=ap_df.index.copy())

    matched_cols=[]

    for model_name,prefix in prefix_map.items():

        cols=[
            c for c in ap_df.columns
            if str(c).startswith(prefix)
            and (suffix is None or str(c).endswith(suffix))
        ]

        if not cols:
            suffix_msg=f" and suffix '{suffix}'" if suffix is not None else ""
            raise ValueError(
                f"No columns found for '{model_name}' "
                f"with prefix '{prefix}'{suffix_msg}."
            )

        matched_cols.extend(cols)

        vals=ap_df[cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Per-allele mean across seeds/replicates.
        mean_df[model_name]=vals.mean(axis=1)

        ci_low=[]
        ci_high=[]

        for _,row in vals.iterrows():

            x=row.to_numpy(dtype=float)
            x=x[np.isfinite(x)]

            if len(x)<2:
                ci_low.append(np.nan)
                ci_high.append(np.nan)
                continue

            if ci_method=="t":
                mean=x.mean()
                se=x.std(ddof=1)/np.sqrt(len(x))
                crit=t.ppf((1+ci)/2,len(x)-1)

                low=mean-crit*se
                high=mean+crit*se

            else:
                boot=x[
                    rng.integers(
                        0,
                        len(x),
                        size=(n_boot,len(x)),
                    )
                ].mean(axis=1)

                alpha=(1-ci)/2
                low,high=np.quantile(
                    boot,
                    [alpha,1-alpha],
                )

            ci_low.append(low)
            ci_high.append(high)

        ci_df[f"{model_name}_ci_low"]=ci_low
        ci_df[f"{model_name}_ci_high"]=ci_high

    # Remove only the individual seeded columns that were summarized.
    # Everything else remains untouched.
    if drop_original:
        matched_cols=list(dict.fromkeys(matched_cols))
        matched_cols=[
            c for c in matched_cols
            if c not in prefix_map
        ]
        mean_df=mean_df.drop(columns=matched_cols)

    return mean_df,ci_df

def partition_datasets_by_allele(train_sets, partition_counts, allele_col="allele", label_col="label", seed=0):
    rng = np.random.default_rng(seed)

    per_dataset_partitions = {}
    for key, df in tqdm(train_sets.items(), desc="Partitioning datasets", leave=True):
        n_part = partition_counts[key]
        alleles = np.array(sorted(df[allele_col].unique()))
        n_alleles = len(alleles)
        if n_part > n_alleles:
            raise ValueError(f"dataset '{key}': requested {n_part} partitions but only {n_alleles} alleles available")
        shuffled = alleles[rng.permutation(n_alleles)]
        allele_splits = np.array_split(shuffled, n_part)
        parts = []
        for i, allele_group in enumerate(allele_splits, start=1):
            name = f"{key}_{i}"
            group_list = [str(a) for a in allele_group]
            sub = df[df[allele_col].isin(set(group_list))].copy()
            parts.append((name, sub, group_list))
        per_dataset_partitions[key] = parts

    keys = list(train_sets.keys())
    max_parts = max(len(per_dataset_partitions[k]) for k in keys)
    partitioned = {}
    for i in range(max_parts):
        for key in keys:
            parts = per_dataset_partitions[key]
            if i < len(parts):
                name, sub, _ = parts[i]
                partitioned[name] = sub

    summary = {}
    for key in keys:
        for name, sub, allele_group in per_dataset_partitions[key]:
            summary[name] = {
                "alleles": allele_group,
                "n_alleles": len(allele_group),
                "n_total": len(sub),
                "n_pos": int((sub[label_col] == 1).sum()),
                "n_neg": int((sub[label_col] == 0).sum()),
            }
    summary = {name: summary[name] for name in partitioned}

    return partitioned, summary

def sample_synthetic_negatives_fast(
    pos_df,
    seqs_by_length,
    forbidden_seq_alleles,
    neg_ratio=99,
    seed=42,
    seq_col="seq",
    allele_col="allele",
    label_col="label",
    batch_size=1_000_000,
    oversample=1.25,
    unique_negatives=False,
    return_combined=True,
    verbose=True,
):
    """
    Sample synthetic negatives per allele x peptide-length group.

    Constraints:
      - negatives are sampled from seqs_by_length[length]
      - exactly neg_ratio negatives are requested per positive row
      - negatives inherit the positive group's allele
      - negative label = 0
      - sampled seq cannot equal a positive seq from that allele x length group
      - sampled seq_allele cannot occur in forbidden_seq_alleles
      - no full scan/filter of the background peptide pools

    forbidden_seq_alleles should ideally already be a set/frozenset.
    """

    if not isinstance(neg_ratio, (int, np.integer)) or neg_ratio < 1:
        raise ValueError("neg_ratio must be a positive integer.")

    required={seq_col,allele_col,label_col}
    missing=required-set(pos_df.columns)
    if missing:
        raise ValueError(f"pos_df missing required columns: {sorted(missing)}")

    if not (pos_df[label_col].to_numpy()==1).all():
        raise ValueError("pos_df must contain only positive rows with label == 1.")

    # Exact forbidden membership requires an indexed lookup.
    if isinstance(forbidden_seq_alleles,(set,frozenset)):
        forbidden=forbidden_seq_alleles
    else:
        if verbose:
            print(
                f"Building exact forbidden lookup from "
                f"{len(forbidden_seq_alleles):,} seq_allele entries..."
            )
        forbidden=set(forbidden_seq_alleles)
        if verbose:
            print(f"Forbidden lookup ready: {len(forbidden):,} unique entries.")

    rng=np.random.default_rng(seed)

    # Work only on the much smaller positive dataframe.
    seq_arr=pos_df[seq_col].astype(str).to_numpy()
    allele_arr=pos_df[allele_col].astype(str).to_numpy()
    seq_len=np.fromiter((len(s) for s in seq_arr),dtype=np.int16,count=len(seq_arr))

    work=pd.DataFrame({
        "_row":np.arange(len(pos_df),dtype=np.int64),
        "_allele":allele_arr,
        "_len":seq_len,
    })

    group_idx=work.groupby(["_allele","_len"],sort=True).indices
    groups=list(group_idx.items())

    # Exact output size is known in advance, so avoid accumulating
    # millions of small DataFrames/lists.
    total_neg=len(pos_df)*neg_ratio
    neg_seqs=np.empty(total_neg,dtype=object)
    neg_alleles=np.empty(total_neg,dtype=object)

    # Lazily convert only peptide lengths that are actually needed.
    pool_cache={}
    out_pos=0

    for (allele,length),rows in tqdm(
        groups,
        total=len(groups),
        desc="Sampling allele×length negatives",
        leave=True,
    ):
        rows=np.asarray(rows,dtype=np.int64)
        n_pos=len(rows)
        n_need=n_pos*neg_ratio

        if length not in seqs_by_length:
            raise KeyError(
                f"No background peptide pool supplied for length {length} "
                f"({allele}, n_pos={n_pos:,})."
            )

        if length not in pool_cache:
            # dtype=object avoids copying all strings into a large fixed-width
            # Unicode matrix; this is primarily an array of object references.
            pool_cache[length]=np.asarray(seqs_by_length[length],dtype=object)

        pool=pool_cache[length]
        pool_n=len(pool)

        if pool_n==0:
            raise ValueError(f"Background pool for length {length} is empty.")

        positive_set=set(seq_arr[rows])
        allele_suffix="_"+allele

        selected=set() if unique_negatives else None
        written=0
        accept_rate=None
        empty_batches=0

        while written<n_need:
            remaining=n_need-written

            if accept_rate is None:
                draw_n=int(np.ceil(remaining*oversample))
            else:
                # Adapt subsequent batches to the observed rejection rate.
                draw_n=int(np.ceil(remaining/max(accept_rate,1e-4)*1.05))

            draw_n=max(min(draw_n,batch_size),min(4096,batch_size))

            # Draw random indices only; never permute/copy the full pool.
            idx=rng.integers(0,pool_n,size=draw_n)
            candidates=pool[idx]

            if unique_negatives:
                keep=np.fromiter(
                    (
                        (s not in positive_set) and
                        (s not in selected) and
                        ((s+allele_suffix) not in forbidden)
                        for s in candidates
                    ),
                    dtype=bool,
                    count=len(candidates),
                )
            else:
                keep=np.fromiter(
                    (
                        (s not in positive_set) and
                        ((s+allele_suffix) not in forbidden)
                        for s in candidates
                    ),
                    dtype=bool,
                    count=len(candidates),
                )

            accepted=candidates[keep]
            accept_rate=len(accepted)/len(candidates)

            if len(accepted)==0:
                empty_batches+=1
                if empty_batches>=25:
                    raise RuntimeError(
                        f"Unable to find valid negatives for allele={allele}, "
                        f"length={length} after 25 consecutive empty batches. "
                        "The usable background may be exhausted or almost "
                        "entirely forbidden."
                    )
                continue

            empty_batches=0

            if unique_negatives:
                # Candidates can repeat within a random batch. Preserve their
                # random encounter order while removing those duplicates.
                accepted=np.asarray(
                    list(dict.fromkeys(accepted.tolist())),
                    dtype=object,
                )

                if len(accepted)==0:
                    continue

            take=min(remaining,len(accepted))
            accepted=accepted[:take]

            start=out_pos+written
            stop=start+take

            neg_seqs[start:stop]=accepted
            neg_alleles[start:stop]=allele

            if unique_negatives:
                selected.update(accepted.tolist())

            written+=take

        out_pos+=n_need

    if out_pos!=total_neg:
        raise RuntimeError(
            f"Internal count mismatch: generated {out_pos:,} negatives, "
            f"expected {total_neg:,}."
        )

    neg_df=pd.DataFrame({
        seq_col:neg_seqs,
        allele_col:neg_alleles,
        label_col:np.zeros(total_neg,dtype=np.int8),
    })

    if verbose:
        print(f"\nPositive rows: {len(pos_df):,}")
        print(f"Negative rows: {len(neg_df):,}")
        print(f"Requested ratio: 1:{neg_ratio}")
        print(f"Allele×length groups: {len(groups):,}")
        print(f"Seed: {seed}")

    if not return_combined:
        return neg_df

    pos_out=pos_df[[seq_col,allele_col,label_col]].copy()
    pos_out[label_col]=1

    return pd.concat(
        [pos_out,neg_df],
        ignore_index=True,
        copy=False,
    )



#####################

def thresholded_recovery_across_datasets(
    dfs,
    score_cols,
    thresholds,
    prefix_map,
    peptide_id_col="peptide_id",
    donor_split_col="all",
    rank_suffix="_rank_r_chp",
    model_col="model",
    score_col="pct_recovery",
    ci=0.95,
    include_all=False,
    all_key="all",
):
    """
    Compute thresholded peptide recovery across multiple datasets and thresholds,
    then average replicate models by prefix and calculate Student-t CIs.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        {dataset_name: dataframe}

    score_cols : list[str]
        Base model score-column names, without rank_suffix.

    thresholds : list[float]
        Percentile thresholds, e.g. [90,95,98,99,99.5].

    prefix_map : dict[str, str]
        Passed directly to average_models_by_prefix_with_ci().
        Keys are final averaged model names and values are prefixes.

    peptide_id_col : str
        Peptide identifier used by keep_max_score_per_id().

    donor_split_col : str
        Grouping column passed to pct_above_threshold_per_subject_single().
        Default "all" reproduces the current whole-dataset calculation.

    rank_suffix : str
        Suffix appended to each score column.

    model_col : str
        Model-name column in output.

    score_col : str
        Recovery-score column in output.

    ci : float
        Confidence level for Student-t interval.

    include_all : bool
        If True, additionally computes a merged "all" result using the exact
        existing methodology:
            - keep max score per peptide ID separately within each dataset
            - concatenate the reduced datasets
            - compute thresholded recovery on the merged dataframe

    all_key : str
        Key used for the merged result.

    Returns
    -------
    dict
        {
            dataset_name: {
                "recovery": DataFrame[
                    model, threshold, pct_recovery
                ],
                "ci95": DataFrame[
                    model, threshold, ci_low, ci_high
                ],
            },
            ...
            "all": {
                "recovery": ...,
                "ci95": ...,
            }
        }
    """
    results={}

    # ============================================================
    # Individual datasets
    # ============================================================

    for dataset_name,df in dfs.items():
        print(f"\n{dataset_name}")

        df_to_use=df.copy()
        df_to_use[donor_split_col]=1

        recovery_frames=[]
        ci_frames=[]

        for threshold in thresholds:
            print(f"  Threshold: {threshold}")

            above_threshold_df=pd.DataFrame()

            for base_score_col in score_cols:
                ranked_score_col=f"{base_score_col}{rank_suffix}"

                df_max=keep_max_score_per_id(
                    df_to_use,
                    ranked_score_col,
                    peptide_id_col,
                )

                recovered=pct_above_threshold_per_subject_single(
                    df_max,
                    donor_split_col,
                    ranked_score_col,
                    threshold=threshold,
                )

                above_threshold_df=pd.concat(
                    [above_threshold_df,recovered],
                    axis=1,
                )

            above_threshold_df=above_threshold_df.T.reset_index()
            above_threshold_df.columns=[model_col,score_col]

            recovery_avg,ci_df=average_models_by_prefix_with_ci(
                above_threshold_df,
                prefix_map,
                model_col=model_col,
                score_col=score_col,
                ci=ci,
            )

            recovery_avg["threshold"]=threshold

            recovery_frames.append(
                recovery_avg[
                    [model_col,"threshold",score_col]
                ]
            )

            if not ci_df.empty:
                ci_long=(
                    ci_df.T
                    .reset_index()
                    .rename(columns={
                        "index":model_col,
                        "ci_low":"ci_low",
                        "ci_high":"ci_high",
                    })
                )

                ci_long["threshold"]=threshold

                ci_frames.append(
                    ci_long[
                        [model_col,"threshold","ci_low","ci_high"]
                    ]
                )

        recovery_df=pd.concat(
            recovery_frames,
            ignore_index=True,
        )

        if ci_frames:
            ci95_df=pd.concat(
                ci_frames,
                ignore_index=True,
            )
        else:
            ci95_df=pd.DataFrame(
                columns=[
                    model_col,
                    "threshold",
                    "ci_low",
                    "ci_high",
                ]
            )

        results[dataset_name]={
            "recovery":recovery_df,
            "ci95":ci95_df,
        }


    # ============================================================
    # Merged "all" dataset
    #
    # IMPORTANT:
    # keep_max_score_per_id() is applied separately within each
    # dataset BEFORE concatenation, matching the existing code.
    # ============================================================

    if include_all:
        print(f"\n{all_key}")

        recovery_frames=[]
        ci_frames=[]

        for threshold in thresholds:
            print(f"  Threshold: {threshold}")

            above_threshold_df=pd.DataFrame()

            for base_score_col in score_cols:
                ranked_score_col=f"{base_score_col}{rank_suffix}"

                max_scored_dfs=[]

                for dataset_name,df in dfs.items():
                    df_max=keep_max_score_per_id(
                        df,
                        ranked_score_col,
                        peptide_id_col,
                    )

                    max_scored_dfs.append(df_max)

                merged_max_scored=pd.concat(
                    max_scored_dfs,
                    ignore_index=True,
                )

                merged_max_scored[donor_split_col]=1

                recovered=pct_above_threshold_per_subject_single(
                    merged_max_scored,
                    donor_split_col,
                    ranked_score_col,
                    threshold=threshold,
                )

                above_threshold_df=pd.concat(
                    [above_threshold_df,recovered],
                    axis=1,
                )

            above_threshold_df=above_threshold_df.T.reset_index()
            above_threshold_df.columns=[model_col,score_col]

            recovery_avg,ci_df=average_models_by_prefix_with_ci(
                above_threshold_df,
                prefix_map,
                model_col=model_col,
                score_col=score_col,
                ci=ci,
            )

            recovery_avg["threshold"]=threshold

            recovery_frames.append(
                recovery_avg[
                    [model_col,"threshold",score_col]
                ]
            )

            if not ci_df.empty:
                ci_long=(
                    ci_df.T
                    .reset_index()
                    .rename(columns={
                        "index":model_col,
                        "ci_low":"ci_low",
                        "ci_high":"ci_high",
                    })
                )

                ci_long["threshold"]=threshold

                ci_frames.append(
                    ci_long[
                        [model_col,"threshold","ci_low","ci_high"]
                    ]
                )

        recovery_df=pd.concat(
            recovery_frames,
            ignore_index=True,
        )

        if ci_frames:
            ci95_df=pd.concat(
                ci_frames,
                ignore_index=True,
            )
        else:
            ci95_df=pd.DataFrame(
                columns=[
                    model_col,
                    "threshold",
                    "ci_low",
                    "ci_high",
                ]
            )

        results[all_key]={
            "recovery":recovery_df,
            "ci95":ci95_df,
        }

    return results

def plot_thresholded_recovery_points_v2(
    results,
    dataset,
    colors,
    model_order=None,
    model_labels=None,
    model_col="model",
    score_col="pct_recovery",
    threshold_col="threshold",
    ci_low_col="ci_low",
    ci_high_col="ci_high",
    title="",
    title_fontsize=18,
    title_bold=False,
    xlabel="Selection threshold",
    ylabel="Fraction recovered",
    xlabel_fontsize=15,
    ylabel_fontsize=15,
    xlabel_pad=8.0,
    ylabel_pad=10.0,
    figsize=(10,6),
    dpi=300,
    default_font="Nimbus Sans",
    ylim=None,
    ylim_bottom_pad_frac=0.03,
    ylim_top_pad_frac=0.01,
    x_margin=0.06,
    y_margin=0.02,
    xtick_labelsize=13,
    ytick_labelsize=13,
    xtick_rotation=0,
    axis_linewidth=1.0,
    tick_length=5.0,
    tick_width=1.0,
    tick_direction="out",
    marker_size=8.0,
    marker_style="o",
    marker_edgewidth=0.0,
    method_spacing=0.16,
    errorbar_linewidth=1.8,
    errorbar_capsize=4.0,
    errorbar_capthick=None,
    errorbar_alpha=1.0,
    grid_alpha=0.0,
    grid_linewidth=0.6,
    grid_linestyle="-",
    legend=True,
    legend_loc="best",
    legend_fontsize=13,
    legend_cols=1,
    legend_frameon=False,
    legend_handlelength=1.2,
    legend_handletextpad=0.4,
    remove_top_right_spines=True,
    show_left_spine=True,
    show_bottom_spine=True,
    divide_by_100=True,
    save_path=None,
    save_png=True,
    save_pdf=True,
    show=True,
):
    """
    Plot thresholded recovery for one dataset as dodged points with CI bars.

    Expected input format:
        results[dataset]["recovery"] with columns:
            model, threshold, pct_recovery
        results[dataset]["ci95"] with columns:
            model, threshold, ci_low, ci_high

    Notes
    -----
    - If divide_by_100=True, recovery/CI values are assumed to be in percent
      units and are converted to fractions by /100.
    - If divide_by_100=False, recovery/CI values are plotted as provided.
    - Threshold 90 is displayed as "Top 10%", 98 as "Top 2%", etc.
    """
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[default_font,"Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    if dataset not in results:
        raise KeyError(f"Dataset '{dataset}' not found in results. Available: {list(results)}")

    recovery_df=results[dataset]["recovery"].copy()
    ci_df=results[dataset]["ci95"].copy()

    required_recovery={model_col,threshold_col,score_col}
    missing_recovery=required_recovery-set(recovery_df.columns)
    if missing_recovery:
        raise ValueError(f"Recovery df missing required columns: {sorted(missing_recovery)}")

    required_ci={model_col,threshold_col,ci_low_col,ci_high_col}
    if len(ci_df)>0:
        missing_ci=required_ci-set(ci_df.columns)
        if missing_ci:
            raise ValueError(f"CI df missing required columns: {sorted(missing_ci)}")

    recovery_df[threshold_col]=pd.to_numeric(recovery_df[threshold_col],errors="coerce")
    recovery_df[score_col]=pd.to_numeric(recovery_df[score_col],errors="coerce")
    if len(ci_df)>0:
        ci_df[threshold_col]=pd.to_numeric(ci_df[threshold_col],errors="coerce")
        ci_df[ci_low_col]=pd.to_numeric(ci_df[ci_low_col],errors="coerce")
        ci_df[ci_high_col]=pd.to_numeric(ci_df[ci_high_col],errors="coerce")

    thresholds=sorted(recovery_df[threshold_col].dropna().unique().tolist())
    if not thresholds:
        raise ValueError("No valid thresholds found.")

    if model_order is None:
        model_order=list(recovery_df[model_col].drop_duplicates())

    if model_labels is None:
        model_labels={m:m for m in model_order}

    missing_colors=[m for m in model_order if m not in colors]
    if missing_colors:
        raise ValueError(f"Missing colors for models: {missing_colors}")

    fig,ax=plt.subplots(figsize=figsize,dpi=dpi)

    n_methods=len(model_order)
    offsets=(np.arange(n_methods)-(n_methods-1)/2.0)*method_spacing
    x_base=np.arange(len(thresholds),dtype=float)

    scale=100.0 if divide_by_100 else 1.0
    all_y=[]

    for i,model in enumerate(model_order):
        sub=(recovery_df[recovery_df[model_col]==model]
             .set_index(threshold_col)
             .reindex(thresholds))
        x=x_base+offsets[i]
        y=sub[score_col].to_numpy(dtype=float)/scale

        color=colors[model]
        label=model_labels.get(model,model)

        ci_sub=(ci_df[ci_df[model_col]==model]
                .set_index(threshold_col)
                .reindex(thresholds)) if len(ci_df)>0 else pd.DataFrame(index=thresholds)

        valid=np.isfinite(y)
        if valid.any():
            all_y.extend(y[valid].tolist())

        has_ci=(len(ci_sub)>0 and ci_low_col in ci_sub.columns and ci_high_col in ci_sub.columns)

        if has_ci:
            ci_low=ci_sub[ci_low_col].to_numpy(dtype=float)/scale
            ci_high=ci_sub[ci_high_col].to_numpy(dtype=float)/scale
            valid_ci=valid & np.isfinite(ci_low) & np.isfinite(ci_high)

            if valid_ci.any():
                yerr=np.vstack([y[valid_ci]-ci_low[valid_ci],ci_high[valid_ci]-y[valid_ci]])
                eb=ax.errorbar(
                    x[valid_ci],y[valid_ci],yerr=yerr,
                    fmt=marker_style,markersize=marker_size,
                    color=color,markerfacecolor=color,markeredgecolor=color,
                    elinewidth=errorbar_linewidth,
                    capsize=errorbar_capsize,
                    capthick=errorbar_linewidth if errorbar_capthick is None else errorbar_capthick,
                    alpha=errorbar_alpha,label=label,
                )

                # Marker edge width applies only to the point marker, not the CI caps
                if eb.lines[0] is not None:
                    eb.lines[0].set_markeredgewidth(marker_edgewidth)

                valid_no_ci=valid & ~valid_ci
                if valid_no_ci.any():
                    ax.plot(
                        x[valid_no_ci],y[valid_no_ci],linestyle="None",marker=marker_style,markersize=marker_size,
                        color=color,markerfacecolor=color,markeredgecolor=color,markeredgewidth=marker_edgewidth,
                    )
            else:
                ax.plot(
                    x[valid],y[valid],linestyle="None",marker=marker_style,markersize=marker_size,
                    color=color,markerfacecolor=color,markeredgecolor=color,markeredgewidth=marker_edgewidth,
                    label=label,
                )
        else:
            ax.plot(
                x[valid],y[valid],linestyle="None",marker=marker_style,markersize=marker_size,
                color=color,markerfacecolor=color,markeredgecolor=color,markeredgewidth=marker_edgewidth,
                label=label,
            )

    top_pct=[100-float(t) for t in thresholds]
    xticklabels=[f"Top {int(v)}%" if float(v).is_integer() else f"Top {v:g}%" for v in top_pct]

    ax.set_xticks(x_base)
    ax.set_xticklabels(xticklabels,rotation=xtick_rotation,ha="center",fontsize=xtick_labelsize)
    ax.set_xlabel(xlabel,fontsize=xlabel_fontsize,fontfamily=default_font,labelpad=xlabel_pad)
    ax.set_ylabel(ylabel,fontsize=ylabel_fontsize,fontfamily=default_font,labelpad=ylabel_pad)

    title_weight="bold" if title_bold else "normal"
    ax.set_title(title,fontsize=title_fontsize,fontfamily=default_font,fontweight=title_weight)

    if ylim is not None:
        y0,y1=ylim
        yr=y1-y0
        pad_bottom=yr*ylim_bottom_pad_frac
        pad_top=yr*ylim_top_pad_frac
        ax.set_ylim(y0-pad_bottom,y1+pad_top)
    elif all_y:
        y0=min(all_y)
        y1=max(all_y)
        if y0==y1:
            pad=0.05 if y0==0 else abs(y0)*0.05
            ax.set_ylim(y0-pad,y1+pad)
        else:
            yr=y1-y0
            ax.set_ylim(y0-yr*ylim_bottom_pad_frac,y1+yr*ylim_top_pad_frac)
        ax.margins(y=y_margin)

    ax.margins(x=x_margin)

    ax.tick_params(axis="both",which="major",length=tick_length,width=tick_width,direction=tick_direction)
    ax.tick_params(axis="x",labelsize=xtick_labelsize)
    ax.tick_params(axis="y",labelsize=ytick_labelsize)

    ax.spines["left"].set_visible(show_left_spine)
    ax.spines["bottom"].set_visible(show_bottom_spine)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    if remove_top_right_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_linewidth(axis_linewidth)
        ax.spines["right"].set_linewidth(axis_linewidth)

    if grid_alpha>0:
        ax.grid(True,alpha=grid_alpha,linewidth=grid_linewidth,linestyle=grid_linestyle,axis="y",zorder=0)

    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if legend:
        handles,labels=ax.get_legend_handles_labels()
        seen=set()
        handles_clean,labels_clean=[],[]
        for h,lbl in zip(handles,labels):
            if lbl not in seen:
                handles_clean.append(h)
                labels_clean.append(lbl)
                seen.add(lbl)
        ax.legend(
            handles_clean,labels_clean,loc=legend_loc,fontsize=legend_fontsize,ncol=legend_cols,
            frameon=legend_frameon,handlelength=legend_handlelength,handletextpad=legend_handletextpad,
        )

    plt.tight_layout()

    if save_path:
        if save_pdf:
            fig.savefig(f"{save_path}.pdf",format="pdf",bbox_inches="tight",dpi=dpi)
        if save_png:
            fig.savefig(f"{save_path}.png",format="png",bbox_inches="tight",dpi=dpi)

    if show:
        plt.show()

    return fig,ax



def get_ft_matches_to_pepcl(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes, pepcl_target_lr, pepcl_target_step,
    ref_col="mhcprime", ft_method="ft", pepcl_method="pepcl_r_dr",
    pepcl_target_step_budget=None, min_seeds=1, focus_seeds=None,

    # Matching options.
    se_aware_matching=False, se_band=1.0,
    frontier_matching=False,

    # Signed comparison direction used by existing trajectory plots.
    difference_direction="ft_minus_pepcl",

    # Confidence intervals.
    ci_level=0.95,

    verbose=True,
):
    """
    Match a target PepCL configuration to:
      1) FT closest in assay/adaptation performance.
      2) FT closest in MS/retention performance.

    Matching modes
    --------------
    Default:
        Exact closest FT point on the matched axis.

    se_aware_matching=True:
        Finds the closest FT point on the matched axis, treats candidates
        within +/- se_band * SE as effectively matched, then selects the
        candidate with greatest performance on the other axis.

    frontier_matching=True:
        Restricts FT candidates to the discrete Pareto frontier obtained by
        jointly maximizing assay performance and MS performance.

        Assay match:
            frontier point closest to PepCL assay performance;
            ties favor greater MS performance.

        MS match:
            frontier point closest to PepCL MS performance;
            ties favor greater assay performance.

    `se_aware_matching` and `frontier_matching` are mutually exclusive.

    difference_direction:
        "ft_minus_pepcl"
            Existing signed-difference behavior.

        "pepcl_minus_ft"
            Positive values indicate PepCL gain over the matched FT point.

    Confidence intervals
    --------------------
    Candidate assay/MS performance intervals are Student-t confidence
    intervals across replicate-level macro performance values.

    For matched FT comparisons, paired delta intervals are computed by first
    evaluating PepCL - FT within each shared replicate and then calculating
    the Student-t confidence interval across those paired differences.

    Returns
    -------
    comparison_df, candidate_df
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy.stats import t

    if ft_method not in method_prefixes or pepcl_method not in method_prefixes:
        raise ValueError(f"'{ft_method}' and '{pepcl_method}' must exist in method_prefixes.")
    if se_band<0:
        raise ValueError("se_band must be >= 0.")
    if se_aware_matching and frontier_matching:
        raise ValueError("se_aware_matching and frontier_matching cannot both be True.")
    if difference_direction not in {"ft_minus_pepcl","pepcl_minus_ft"}:
        raise ValueError("difference_direction must be 'ft_minus_pepcl' or 'pepcl_minus_ft'.")
    if not 0<ci_level<1:
        raise ValueError("ci_level must be in (0,1).")

    def _lr_string(lr):
        s=str(lr)
        if "eN" in s: return s
        try:
            mantissa,exp=f"{float(lr):.12e}".split("e")
            mantissa=mantissa.rstrip("0").rstrip(".")
            exp=int(exp)
            return f"{mantissa}eN{abs(exp)}" if exp<0 else f"{mantissa}e{exp}"
        except (TypeError,ValueError):
            return s

    def _summary_ci(values):
        x=np.asarray(values,dtype=float)
        x=x[np.isfinite(x)]
        n=len(x)

        if n==0:
            return dict(mean=np.nan,sd=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,n=0)

        mean=float(np.mean(x))

        if n<2:
            return dict(mean=mean,sd=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,n=n)

        sd=float(np.std(x,ddof=1))
        se=sd/np.sqrt(n)
        crit=float(t.ppf((1+ci_level)/2,n-1))

        return dict(
            mean=mean,
            sd=sd,
            se=se,
            ci_low=mean-crit*se,
            ci_high=mean+crit*se,
            n=n,
        )

    def _pareto_frontier_max(df,x_col,y_col):
        x=df[x_col].to_numpy(float)
        y=df[y_col].to_numpy(float)
        keep=np.ones(len(df),dtype=bool)

        for i in range(len(df)):
            dominated=(
                (x>=x[i])
                & (y>=y[i])
                & ((x>x[i]) | (y>y[i]))
            )
            dominated[i]=False

            if np.any(dominated):
                keep[i]=False

        return pd.Series(keep,index=df.index)

    def _paired_pepcl_minus_model(pep_row,model_row):
        """
        Paired PepCL - model differences across shared replicate IDs.
        """
        pep_assay=pep_row["assay_performance_by_seed"]
        model_assay=model_row["assay_performance_by_seed"]

        pep_ms=pep_row["ms_performance_by_seed"]
        model_ms=model_row["ms_performance_by_seed"]

        common=sorted(
            set(pep_assay)
            & set(model_assay)
            & set(pep_ms)
            & set(model_ms)
        )

        assay_by_seed={}
        ms_by_seed={}

        for seed in common:
            pa=float(pep_assay[seed])
            fa=float(model_assay[seed])
            pm=float(pep_ms[seed])
            fm=float(model_ms[seed])

            if np.isfinite(pa) and np.isfinite(fa):
                assay_by_seed[seed]=pa-fa

            if np.isfinite(pm) and np.isfinite(fm):
                ms_by_seed[seed]=pm-fm

        assay_stats=_summary_ci(list(assay_by_seed.values()))
        ms_stats=_summary_ci(list(ms_by_seed.values()))

        return {
            "assay":assay_stats,
            "ms":ms_stats,
            "assay_by_seed":assay_by_seed,
            "ms_by_seed":ms_by_seed,
            "paired_seed_ids":tuple(common),
        }

    if focus_seeds is not None:
        if np.isscalar(focus_seeds):
            focus_seeds=[focus_seeds]

        focus_seeds=tuple(
            sorted(dict.fromkeys(int(s) for s in focus_seeds))
        )

        if not focus_seeds:
            raise ValueError("focus_seeds cannot be empty.")

    target_lr=_lr_string(pepcl_target_lr)
    target_step=int(pepcl_target_step)

    if ref_col not in ms_ap_df.columns or ref_col not in assay_ap_df.columns:
        raise ValueError(f"Reference '{ref_col}' must exist in both AP dataframes.")

    if not ms_ap_df.columns.is_unique or not assay_ap_df.columns.is_unique:
        raise ValueError("Both AP dataframes must have unique column names.")

    score_cols=list(dict.fromkeys(score_cols))

    usable=[
        c for c in score_cols
        if c!=ref_col
        and c in ms_ap_df.columns
        and c in assay_ap_df.columns
    ]

    if not usable:
        raise ValueError("No supplied score columns are present in both AP dataframes.")

    # Macro across evaluable alleles for every replicate/model column.
    ms_map=ms_ap_df[[ref_col]+usable].mean(axis=0)
    assay_map=assay_ap_df[[ref_col]+usable].mean(axis=0)

    ref_ms=float(ms_map[ref_col])
    ref_assay=float(assay_map[ref_col])

    ordered_methods=sorted(
        method_prefixes.items(),
        key=lambda x:len(x[1]),
        reverse=True
    )

    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix):
                return name
        return None

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
        r"(?:_step_(?P<checkpoint>\d+))?$"
    )

    rows=[]

    for col in usable:
        method=_method(col)
        m=pattern.search(str(col))

        if method not in {ft_method,pepcl_method} or m is None:
            continue

        budget=int(m["budget"])
        cp=m["checkpoint"]
        is_full=cp is None
        checkpoint=budget if is_full else int(cp)

        if checkpoint>budget:
            continue

        rows.append({
            "score_col":col,
            "method":method,
            "learning_rate":m["lr"],
            "step_budget":budget,
            "checkpoint":checkpoint,
            "seed":int(m["seed"]),
            "is_full_model":is_full,
            "assay_performance":float(assay_map[col]),
            "ms_performance":float(ms_map[col]),
        })

    seed_df=pd.DataFrame(rows)

    if seed_df.empty:
        raise ValueError("No FT/PepCL candidates could be parsed.")

    # Prefer explicit full endpoint over duplicate `_step_budget`.
    seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)

    seed_df=(
        seed_df.sort_values(
            ["method","learning_rate","step_budget","seed","checkpoint","_prefer_full"],
            ascending=[True,True,True,True,True,False]
        )
        .drop_duplicates(
            ["method","learning_rate","step_budget","seed","checkpoint"],
            keep="first"
        )
        .drop(columns="_prefer_full")
        .reset_index(drop=True)
    )

    keys=["method","learning_rate","step_budget","checkpoint"]

    if focus_seeds is not None:
        seed_df=seed_df[
            seed_df["seed"].isin(focus_seeds)
        ].copy()

        if seed_df.empty:
            raise ValueError(f"No candidates remain for focus_seeds={focus_seeds}.")

        info=(
            seed_df.groupby(keys)["seed"]
            .agg(
                seed_ids=lambda x:tuple(sorted(set(x))),
                n_seeds="nunique",
            )
            .reset_index()
        )

        info=info[
            info["seed_ids"].map(lambda x:x==focus_seeds)
        ].copy()

        if info.empty:
            raise ValueError(
                f"No candidates contain all focused replicate IDs {focus_seeds}."
            )

        valid=set(map(tuple,info[keys].to_numpy()))

        seed_df=seed_df[
            seed_df[keys].apply(tuple,axis=1).isin(valid)
        ].copy()

    # ------------------------------------------------------------------
    # Candidate-level summaries.
    # ------------------------------------------------------------------
    candidate_df=(
        seed_df.groupby(keys,sort=False)
        .agg(
            assay_performance=("assay_performance","mean"),
            assay_sd=("assay_performance","std"),
            ms_performance=("ms_performance","mean"),
            ms_sd=("ms_performance","std"),
            n_seeds=("seed","nunique"),
            seed_ids=("seed",lambda x:tuple(sorted(set(x)))),
            score_cols_by_seed=("score_col",lambda x:tuple(x)),
        )
        .reset_index()
    )

    if focus_seeds is None:
        candidate_df=candidate_df[
            candidate_df["n_seeds"]>=min_seeds
        ].copy()

    candidate_df["assay_se"]=(
        candidate_df["assay_sd"] /
        np.sqrt(candidate_df["n_seeds"])
    )

    candidate_df["ms_se"]=(
        candidate_df["ms_sd"] /
        np.sqrt(candidate_df["n_seeds"])
    )

    # Retain exact replicate-level values.
    raw_rows=[]

    for key_vals,g in seed_df.groupby(keys,sort=False):
        g=g.sort_values("seed")

        row=dict(zip(keys,key_vals))

        row.update({
            "assay_performance_by_seed":
                dict(zip(g["seed"],g["assay_performance"])),

            "ms_performance_by_seed":
                dict(zip(g["seed"],g["ms_performance"])),

            "score_col_by_seed":
                dict(zip(g["seed"],g["score_col"])),
        })

        raw_rows.append(row)

    candidate_df=candidate_df.merge(
        pd.DataFrame(raw_rows),
        on=keys,
        how="left"
    )

    # Student-t CI for each candidate operating point.
    assay_ci=[]
    ms_ci=[]

    for _,r in candidate_df.iterrows():
        a=_summary_ci(
            list(r["assay_performance_by_seed"].values())
        )

        m=_summary_ci(
            list(r["ms_performance_by_seed"].values())
        )

        assay_ci.append(
            (a["ci_low"],a["ci_high"])
        )

        ms_ci.append(
            (m["ci_low"],m["ci_high"])
        )

    candidate_df["assay_performance_ci_low"]=[
        x[0] for x in assay_ci
    ]

    candidate_df["assay_performance_ci_high"]=[
        x[1] for x in assay_ci
    ]

    candidate_df["ms_performance_ci_low"]=[
        x[0] for x in ms_ci
    ]

    candidate_df["ms_performance_ci_high"]=[
        x[1] for x in ms_ci
    ]

    candidate_df["assay_delta_vs_ref"]=(
        candidate_df["assay_performance"] -
        ref_assay
    )

    candidate_df["ms_delta_vs_ref"]=(
        candidate_df["ms_performance"] -
        ref_ms
    )

    candidate_df["step"]=candidate_df["checkpoint"]

    candidate_df["model_name"]=candidate_df.apply(
        lambda r:
            f"{r['method']}_steps_{int(r['step_budget'])}_"
            f"lr_{r['learning_rate']}_step_{int(r['checkpoint'])}",
        axis=1
    )

    # ------------------------------------------------------------------
    # Identify PepCL target.
    # ------------------------------------------------------------------
    target=candidate_df[
        (candidate_df["method"]==pepcl_method)
        & (candidate_df["learning_rate"].astype(str)==target_lr)
        & (candidate_df["checkpoint"]==target_step)
    ].copy()

    if pepcl_target_step_budget is not None:
        target=target[
            target["step_budget"]==int(pepcl_target_step_budget)
        ]

    if target.empty:
        raise ValueError(
            f"No PepCL target found for LR={target_lr}, step={target_step}"
            +(f", budget={pepcl_target_step_budget}"
              if pepcl_target_step_budget is not None else "")
            +"."
        )

    if len(target)>1:
        opts=target[
            [
                "model_name",
                "learning_rate",
                "step_budget",
                "checkpoint",
                "n_seeds",
            ]
        ].to_dict("records")

        raise ValueError(
            "PepCL target is ambiguous. Provide pepcl_target_step_budget. "
            f"Matches: {opts}"
        )

    target=target.iloc[0]

    pep_assay=float(target["assay_performance"])
    pep_ms=float(target["ms_performance"])

    ft=candidate_df[
        candidate_df["method"]==ft_method
    ].copy()

    if ft.empty:
        raise ValueError(
            f"No eligible FT candidates found for '{ft_method}'."
        )

    ft["adaptation_match_gap"]=(
        ft["assay_performance"] -
        pep_assay
    ).abs()

    ft["ms_match_gap"]=(
        ft["ms_performance"] -
        pep_ms
    ).abs()

    # ------------------------------------------------------------------
    # Pareto frontier.
    # ------------------------------------------------------------------
    candidate_df["is_ft_pareto_frontier"]=False

    ft_frontier_mask=_pareto_frontier_max(
        ft,
        "assay_performance",
        "ms_performance"
    )

    ft_frontier=ft[
        ft_frontier_mask
    ].copy()

    candidate_df.loc[
        candidate_df.index.isin(ft_frontier.index),
        "is_ft_pareto_frontier"
    ]=True

    # ------------------------------------------------------------------
    # Original exact closest matches.
    # ------------------------------------------------------------------
    exact_ft_match_assay=ft.sort_values(
        [
            "adaptation_match_gap",
            "checkpoint",
            "step_budget",
            "learning_rate",
        ],
        ascending=[True,True,True,True]
    ).iloc[0]

    exact_ft_match_ms=ft.sort_values(
        [
            "ms_match_gap",
            "checkpoint",
            "step_budget",
            "learning_rate",
        ],
        ascending=[True,True,True,True]
    ).iloc[0]

    ft_match_assay=exact_ft_match_assay
    ft_match_ms=exact_ft_match_ms

    # ------------------------------------------------------------------
    # SE-aware matching.
    # ------------------------------------------------------------------
    if se_aware_matching:
        assay_se=float(
            exact_ft_match_assay["assay_se"]
        )

        if np.isfinite(assay_se):
            center=float(
                exact_ft_match_assay["assay_performance"]
            )

            radius=float(se_band)*assay_se

            assay_band=ft[
                (ft["assay_performance"]>=center-radius)
                & (ft["assay_performance"]<=center+radius)
            ].copy()

            if not assay_band.empty:
                ft_match_assay=assay_band.sort_values(
                    [
                        "ms_performance",
                        "adaptation_match_gap",
                        "checkpoint",
                        "step_budget",
                        "learning_rate",
                    ],
                    ascending=[
                        False,
                        True,
                        True,
                        True,
                        True,
                    ]
                ).iloc[0]

        elif verbose:
            print(
                "WARNING: assay SE for closest FT adaptation match is "
                "non-finite; using the original closest assay match."
            )

        ms_se=float(
            exact_ft_match_ms["ms_se"]
        )

        if np.isfinite(ms_se):
            center=float(
                exact_ft_match_ms["ms_performance"]
            )

            radius=float(se_band)*ms_se

            ms_band=ft[
                (ft["ms_performance"]>=center-radius)
                & (ft["ms_performance"]<=center+radius)
            ].copy()

            if not ms_band.empty:
                ft_match_ms=ms_band.sort_values(
                    [
                        "assay_performance",
                        "ms_match_gap",
                        "checkpoint",
                        "step_budget",
                        "learning_rate",
                    ],
                    ascending=[
                        False,
                        True,
                        True,
                        True,
                        True,
                    ]
                ).iloc[0]

        elif verbose:
            print(
                "WARNING: MS SE for closest FT retention match is "
                "non-finite; using the original closest MS match."
            )

    # ------------------------------------------------------------------
    # Pareto-frontier matching.
    # ------------------------------------------------------------------
    elif frontier_matching:
        if ft_frontier.empty:
            raise ValueError(
                "No FT candidates remain on the Pareto frontier."
            )

        ft_frontier["adaptation_match_gap"]=(
            ft_frontier["assay_performance"] -
            pep_assay
        ).abs()

        ft_frontier["ms_match_gap"]=(
            ft_frontier["ms_performance"] -
            pep_ms
        ).abs()

        ft_match_assay=ft_frontier.sort_values(
            [
                "adaptation_match_gap",
                "ms_performance",
                "checkpoint",
                "step_budget",
                "learning_rate",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
            ]
        ).iloc[0]

        ft_match_ms=ft_frontier.sort_values(
            [
                "ms_match_gap",
                "assay_performance",
                "checkpoint",
                "step_budget",
                "learning_rate",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
            ]
        ).iloc[0]

    # ------------------------------------------------------------------
    # Build comparison dataframe.
    # ------------------------------------------------------------------
    direction_sign=(
        1.0
        if difference_direction=="ft_minus_pepcl"
        else -1.0
    )

    def _row(role,r=None,match_axis=None):
        if role=="mhcprime":
            model_name=method=ref_col
            lr=budget=step=0
            n_seeds=0
            score_cols=(ref_col,)

            assay=ref_assay
            ms=ref_ms

            assay_sd=assay_se=ms_sd=ms_se=0.0
            assay_ci_low=assay_ci_high=np.nan
            ms_ci_low=ms_ci_high=np.nan

            assay_by_seed={}
            ms_by_seed={}

            paired={
                "assay":_summary_ci([]),
                "ms":_summary_ci([]),
                "assay_by_seed":{},
                "ms_by_seed":{},
                "paired_seed_ids":tuple(),
            }

        else:
            model_name=r["model_name"]
            method=r["method"]
            lr=r["learning_rate"]
            budget=int(r["step_budget"])
            step=int(r["checkpoint"])
            n_seeds=int(r["n_seeds"])
            score_cols=r["score_cols_by_seed"]

            assay=float(r["assay_performance"])
            ms=float(r["ms_performance"])

            assay_sd=float(r["assay_sd"])
            assay_se=float(r["assay_se"])
            ms_sd=float(r["ms_sd"])
            ms_se=float(r["ms_se"])

            assay_ci_low=float(r["assay_performance_ci_low"])
            assay_ci_high=float(r["assay_performance_ci_high"])
            ms_ci_low=float(r["ms_performance_ci_low"])
            ms_ci_high=float(r["ms_performance_ci_high"])

            assay_by_seed=r["assay_performance_by_seed"]
            ms_by_seed=r["ms_performance_by_seed"]

            paired=_paired_pepcl_minus_model(
                target,
                r
            )

        assay_ft_minus_pep=assay-pep_assay
        ms_ft_minus_pep=ms-pep_ms

        assay_pep_minus_model=pep_assay-assay
        ms_pep_minus_model=pep_ms-ms

        assay_display=(
            direction_sign *
            assay_ft_minus_pep
        )

        ms_display=(
            direction_sign *
            ms_ft_minus_pep
        )

        return {
            "role":role,
            "model_name":model_name,
            "method":method,
            "learning_rate":lr,
            "step_budget":budget,
            "step":step,
            "n_seeds":n_seeds,
            "score_cols_by_seed":score_cols,
            "match_axis":match_axis,

            # Operating-point performance.
            "assay_performance":assay,
            "assay_sd":assay_sd,
            "assay_se":assay_se,
            "assay_performance_ci_low":assay_ci_low,
            "assay_performance_ci_high":assay_ci_high,

            "ms_performance":ms,
            "ms_sd":ms_sd,
            "ms_se":ms_se,
            "ms_performance_ci_low":ms_ci_low,
            "ms_performance_ci_high":ms_ci_high,

            # Replicate-level performance.
            "assay_performance_by_seed":assay_by_seed,
            "ms_performance_by_seed":ms_by_seed,

            # Relative to MHCPrime.
            "assay_delta_vs_mhcprime":assay-ref_assay,
            "ms_delta_vs_mhcprime":ms-ref_ms,

            # Existing signed plot-facing quantities.
            "assay_difference_vs_pepcl":assay_display,
            "ms_difference_vs_pepcl":ms_display,

            # Explicit marginal signed quantities.
            "assay_ft_minus_pepcl":assay_ft_minus_pep,
            "ms_ft_minus_pepcl":ms_ft_minus_pep,

            "assay_pepcl_minus_model":assay_pep_minus_model,
            "ms_pepcl_minus_model":ms_pep_minus_model,

            # Paired PepCL - model differences.
            "assay_paired_delta_mean":paired["assay"]["mean"],
            "assay_paired_delta_sd":paired["assay"]["sd"],
            "assay_paired_delta_se":paired["assay"]["se"],
            "assay_paired_delta_ci_low":paired["assay"]["ci_low"],
            "assay_paired_delta_ci_high":paired["assay"]["ci_high"],

            "ms_paired_delta_mean":paired["ms"]["mean"],
            "ms_paired_delta_sd":paired["ms"]["sd"],
            "ms_paired_delta_se":paired["ms"]["se"],
            "ms_paired_delta_ci_low":paired["ms"]["ci_low"],
            "ms_paired_delta_ci_high":paired["ms"]["ci_high"],

            "assay_paired_delta_by_seed":paired["assay_by_seed"],
            "ms_paired_delta_by_seed":paired["ms_by_seed"],
            "paired_seed_ids":paired["paired_seed_ids"],

            # Existing clipped quantities.
            "adaptation_loss_vs_pepcl":
                max(0.0,-assay_ft_minus_pep),

            "ms_retention_loss_vs_pepcl":
                max(0.0,-ms_ft_minus_pep),

            "adaptation_gain_vs_pepcl":
                max(0.0,assay_ft_minus_pep),

            "ms_retention_gain_vs_pepcl":
                max(0.0,ms_ft_minus_pep),

            "matched_axis_absolute_gap":
                abs(assay_ft_minus_pep)
                if match_axis=="assay"
                else abs(ms_ft_minus_pep)
                if match_axis=="ms"
                else 0.0
                if role=="pepcl_target"
                else np.nan,
        }

    comparison_df=pd.DataFrame([
        _row("mhcprime"),
        _row("pepcl_target",target),
        _row(
            "ft_match_adaptation",
            ft_match_assay,
            "assay"
        ),
        _row(
            "ft_match_ms",
            ft_match_ms,
            "ms"
        ),
    ])

    if verbose:
        mode=(
            "pareto_frontier"
            if frontier_matching
            else "se_band"
            if se_aware_matching
            else "closest"
        )

        print(
            f"Replicates used: "
            f"{focus_seeds if focus_seeds is not None else 'min_seeds='+str(min_seeds)}"
        )

        print(f"Matching mode: {mode}")
        print(f"CI level: {ci_level:.3f}")
        print(f"Difference direction: {difference_direction}")

        if se_aware_matching:
            print(f"SE band: {se_band:g} SE")

        print(
            f"PepCL target: {target['model_name']} | "
            f"assay={pep_assay:.6f}, MS={pep_ms:.6f}"
        )

        if frontier_matching:
            print(
                f"\nFT candidates: {len(ft)} | "
                f"Pareto-frontier candidates: {len(ft_frontier)}"
            )

        for name,r in [
            ("FT matched assay",ft_match_assay),
            ("FT matched MS",ft_match_ms),
        ]:
            paired=_paired_pepcl_minus_model(
                target,
                r
            )

            print(
                f"\n{name}: {r['model_name']} | "
                f"assay={r['assay_performance']:.6f} "
                f"[{r['assay_performance_ci_low']:.6f}, "
                f"{r['assay_performance_ci_high']:.6f}] | "
                f"MS={r['ms_performance']:.6f} "
                f"[{r['ms_performance_ci_low']:.6f}, "
                f"{r['ms_performance_ci_high']:.6f}]"
            )

            print(
                f"  PepCL - FT paired ΔAssay="
                f"{paired['assay']['mean']:.6f} "
                f"[{paired['assay']['ci_low']:.6f}, "
                f"{paired['assay']['ci_high']:.6f}]"
            )

            print(
                f"  PepCL - FT paired ΔMS="
                f"{paired['ms']['mean']:.6f} "
                f"[{paired['ms']['ci_low']:.6f}, "
                f"{paired['ms']['ci_high']:.6f}]"
            )

        print("\nComparison dataframe:")
        print(
            comparison_df.to_string(index=False)
        )

    return comparison_df,candidate_df


def plot_matched_ft_comparison_table(
    comparison_dfs,
    dataset_order=None,
    dataset_labels=None,

    # Column labels.
    assay_col_label="Assay",
    match_col_label="Matched dimension",
    config_col_label="FT configuration",
    assay_performance_label="Assay performance [95% CI]",
    ms_performance_label="MS performance [95% CI]",
    assay_delta_label="ΔAssay [95% CI]",
    ms_delta_label="ΔMS [95% CI]",

    # Row labels.
    adaptation_match_label="Assay matched",
    ms_match_label="MS matched",

    # Number/config formatting.
    decimals=3,
    show_plus=True,
    ci_brackets=(" [",", ","]"),
    config_format="LR: {lr}, Steps: {steps}",

    # Figure / fonts.
    fig_size=None,
    dpi=300,
    title="Matched FT configurations and PepCL performance gains",
    title_fontsize=16,
    header_fontsize=11,
    body_fontsize=11,
    font_family="Nimbus Sans",

    # Table geometry.
    table_bbox=(0.0,0.02,1.0,0.92),
    row_height_scale=1.45,
    col_widths=None,
    assay_align="left",
    match_align="left",
    config_align="left",
    value_align="center",

    # Lines.
    top_linewidth=1.5,
    header_linewidth=0.8,
    bottom_linewidth=1.5,
    dataset_group_linewidth=0.5,
    line_color="black",
    independent_lines=True,

    # Styling.
    bold_dataset_names=True,
    bold_matched_gain=True,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot the two matched FT comparators for each dataset.

    Rows
    ----
    Assay matched:
        FT frontier configuration closest to PepCL assay performance.

    MS matched:
        FT frontier configuration closest to PepCL MS performance.

    Performance columns
    -------------------
    Report the selected FT model's replicate mean and 95% CI.

    Delta columns
    -------------
    Always represent paired:

        PepCL - matched FT

    calculated within replicate before summary.

    Therefore positive values indicate a PepCL advantage.

    Styling
    -------
    Assay-matched row:
        ΔMS is bold.

    MS-matched row:
        ΔAssay is bold.

    Dataset names are vertically centered across their two rows.

    Horizontal table rules are independent Line2D artists by default.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    if not isinstance(comparison_dfs,dict) or not comparison_dfs:
        raise ValueError(
            "comparison_dfs must be a non-empty dictionary."
        )

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)

    if dataset_order is None:
        dataset_order=list(comparison_dfs)

    else:
        missing=[
            d for d in dataset_order
            if d not in comparison_dfs
        ]

        if missing:
            raise ValueError(
                f"dataset_order contains missing datasets: {missing}"
            )

    def _lr_label(v):
        s=str(v)

        if "eN" in s:
            a,b=s.split("eN",1)
            return f"{a}e-{int(b)}"

        try:
            x=float(s)

            if x==0:
                return "0"

            m,e=f"{x:.10e}".split("e")

            return (
                f"{m.rstrip('0').rstrip('.')}e{int(e)}"
            )

        except (TypeError,ValueError):
            return s

    def _number(v,signed=False):
        if not np.isfinite(v):
            return ""

        v=float(v)

        if round(v,decimals)==0:
            return "0.0"

        s=f"{abs(v):.{decimals}f}"

        if signed and show_plus:
            return (
                ("+" if v>0 else "-")
                +s
            )

        if signed:
            return f"{v:.{decimals}f}"

        return s

    def _fmt_ci(mean,lo,hi,signed=False):
        if not np.isfinite(mean):
            return ""

        if not np.isfinite(lo) or not np.isfinite(hi):
            return _number(
                mean,
                signed=signed
            )

        return (
            _number(mean,signed=signed)
            +ci_brackets[0]
            +_number(lo,signed=signed)
            +ci_brackets[1]
            +_number(hi,signed=signed)
            +ci_brackets[2]
        )

    rows=[]

    for dataset in dataset_order:
        d=comparison_dfs[
            dataset
        ].copy()

        if "role" not in d.columns:
            raise ValueError(
                f"{dataset}: comparison_df is missing 'role'."
            )

        role_map={
            r["role"]:r
            for _,r in d.iterrows()
        }

        required_roles=[
            "ft_match_adaptation",
            "ft_match_ms",
        ]

        missing=[
            r for r in required_roles
            if r not in role_map
        ]

        if missing:
            raise ValueError(
                f"{dataset}: missing roles {missing}."
            )

        required_cols=[
            "learning_rate",
            "step",

            "assay_performance",
            "assay_performance_ci_low",
            "assay_performance_ci_high",

            "ms_performance",
            "ms_performance_ci_low",
            "ms_performance_ci_high",

            "assay_paired_delta_mean",
            "assay_paired_delta_ci_low",
            "assay_paired_delta_ci_high",

            "ms_paired_delta_mean",
            "ms_paired_delta_ci_low",
            "ms_paired_delta_ci_high",
        ]

        missing_cols=[
            c for c in required_cols
            if c not in d.columns
        ]

        if missing_cols:
            raise ValueError(
                f"{dataset}: comparison_df missing columns {missing_cols}. "
                "Rerun with the updated get_ft_matches_to_pepcl()."
            )

        display_name=dataset_labels.get(
            dataset,
            dataset
        )

        for role,match_label in [
            (
                "ft_match_adaptation",
                adaptation_match_label,
            ),
            (
                "ft_match_ms",
                ms_match_label,
            ),
        ]:
            r=role_map[
                role
            ]

            config=config_format.format(
                lr=_lr_label(
                    r["learning_rate"]
                ),
                steps=int(
                    r["step"]
                ),
            )

            rows.append({
                assay_col_label:display_name,
                match_col_label:match_label,
                config_col_label:config,

                assay_performance_label:_fmt_ci(
                    float(r["assay_performance"]),
                    float(r["assay_performance_ci_low"]),
                    float(r["assay_performance_ci_high"]),
                ),

                ms_performance_label:_fmt_ci(
                    float(r["ms_performance"]),
                    float(r["ms_performance_ci_low"]),
                    float(r["ms_performance_ci_high"]),
                ),

                assay_delta_label:_fmt_ci(
                    float(r["assay_paired_delta_mean"]),
                    float(r["assay_paired_delta_ci_low"]),
                    float(r["assay_paired_delta_ci_high"]),
                    signed=True,
                ),

                ms_delta_label:_fmt_ci(
                    float(r["ms_paired_delta_mean"]),
                    float(r["ms_paired_delta_ci_low"]),
                    float(r["ms_paired_delta_ci_high"]),
                    signed=True,
                ),

                "_dataset":dataset,
                "_role":role,
            })

    display=pd.DataFrame(
        rows
    )

    internal_role=display.pop(
        "_role"
    )

    display.pop(
        "_dataset"
    )

    final_labels=list(
        display.columns
    )

    ncols=len(
        final_labels
    )

    if col_widths is None:
        col_widths=[
            0.09,  # Assay
            0.12,  # Matched dimension
            0.20,  # FT configuration
            0.17,  # Assay performance
            0.16,  # MS performance
            0.13,  # Delta assay
            0.13,  # Delta MS
        ]

    if len(col_widths)!=ncols:
        raise ValueError(
            f"col_widths must have length {ncols}, "
            f"got {len(col_widths)}."
        )

    if fig_size is None:
        fig_size=(
            17,
            max(
                3.0,
                .42*(len(display)+2)
            )
        )

    matplotlib.rcParams[
        "pdf.fonttype"
    ]=42

    matplotlib.rcParams[
        "ps.fonttype"
    ]=42

    matplotlib.rcParams[
        "svg.fonttype"
    ]="none"

    matplotlib.rcParams[
        "font.family"
    ]="sans-serif"

    matplotlib.rcParams[
        "font.sans-serif"
    ]=[
        font_family,
        "Nimbus Sans",
        "Nimbus Sans L",
        "DejaVu Sans",
    ]

    fp_body=FontProperties(
        family=font_family,
        size=body_fontsize
    )

    fp_body_bold=FontProperties(
        family=font_family,
        size=body_fontsize,
        weight="bold"
    )

    fp_header=FontProperties(
        family=font_family,
        size=header_fontsize,
        weight="bold"
    )

    fp_dataset=FontProperties(
        family=font_family,
        size=body_fontsize,
        weight=(
            "bold"
            if bold_dataset_names
            else "normal"
        )
    )

    fp_title=FontProperties(
        family=font_family,
        size=title_fontsize
    )

    fig,ax=plt.subplots(
        figsize=fig_size,
        dpi=dpi
    )

    ax.axis(
        "off"
    )

    if title is not None:
        ax.text(
            0.0,
            1.02,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontproperties=fp_title
        )

    # Draw assay labels separately so each can span two rows.
    table_display=display.copy()
    table_display[
        assay_col_label
    ]=""

    tbl=ax.table(
        cellText=table_display.values.tolist(),
        colLabels=final_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=table_bbox,
    )

    tbl.auto_set_font_size(
        False
    )

    tbl.set_fontsize(
        body_fontsize
    )

    tbl.scale(
        1.0,
        row_height_scale
    )

    nrows=len(display)+1

    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")

        cell.get_text().set_fontproperties(
            fp_body
        )

    # Header.
    for c in range(ncols):
        tbl[
            (0,c)
        ].get_text().set_fontproperties(
            fp_header
        )

    # Alignment.
    for r in range(nrows):
        if r>0:
            tbl[
                (r,1)
            ].get_text().set_ha(
                match_align
            )

            tbl[
                (r,2)
            ].get_text().set_ha(
                config_align
            )

        for c in range(
            3,
            ncols
        ):
            tbl[
                (r,c)
            ].get_text().set_ha(
                value_align
            )

    # Assay matched -> bold ΔMS.
    # MS matched -> bold ΔAssay.
    if bold_matched_gain:
        for r,role in enumerate(
            internal_role,
            start=1
        ):
            if role=="ft_match_adaptation":
                tbl[
                    (r,6)
                ].get_text().set_fontproperties(
                    fp_body_bold
                )

            elif role=="ft_match_ms":
                tbl[
                    (r,5)
                ].get_text().set_fontproperties(
                    fp_body_bold
                )

    fig.canvas.draw()

    # ==============================================================
    # Vertically centered assay labels
    # ==============================================================
    for ds_idx,dataset in enumerate(
        dataset_order
    ):
        r1=1+2*ds_idx
        r2=r1+1

        cell1=tbl[
            (r1,0)
        ]

        cell2=tbl[
            (r2,0)
        ]

        x=cell1.get_x()
        width=cell1.get_width()

        top=(
            cell1.get_y()
            +cell1.get_height()
        )

        bottom=cell2.get_y()

        y=(
            top+bottom
        )/2

        if assay_align=="left":
            tx=x+.03*width
            ha="left"

        elif assay_align=="right":
            tx=x+.97*width
            ha="right"

        else:
            tx=x+.5*width
            ha="center"

        ax.text(
            tx,
            y,
            dataset_labels.get(
                dataset,
                dataset
            ),
            transform=ax.transAxes,
            ha=ha,
            va="center",
            fontproperties=fp_dataset,
            zorder=10,
        )

    # ==============================================================
    # Independent table rules
    # ==============================================================
    left=tbl[
        (0,0)
    ].get_x()

    right=(
        tbl[
            (0,ncols-1)
        ].get_x()
        +tbl[
            (0,ncols-1)
        ].get_width()
    )

    header_top=(
        tbl[
            (0,0)
        ].get_y()
        +tbl[
            (0,0)
        ].get_height()
    )

    header_bottom=tbl[
        (0,0)
    ].get_y()

    bottom=tbl[
        (nrows-1,0)
    ].get_y()

    def _add_rule(
        y,
        linewidth,
        alpha=1.0
    ):
        if independent_lines:
            line=Line2D(
                [left,right],
                [y,y],
                transform=ax.transAxes,
                color=line_color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle="-",
                solid_capstyle="butt",
                clip_on=False,
                zorder=5,
            )

            ax.add_line(
                line
            )

            return line

        return ax.hlines(
            y,
            left,
            right,
            transform=ax.transAxes,
            linewidth=linewidth,
            color=line_color,
            alpha=alpha,
            zorder=5
        )

    _add_rule(
        header_top,
        top_linewidth
    )

    _add_rule(
        header_bottom,
        header_linewidth
    )

    for ds_idx in range(
        len(dataset_order)-1
    ):
        r2=2+2*ds_idx

        y=tbl[
            (r2,0)
        ].get_y()

        _add_rule(
            y,
            dataset_group_linewidth,
            alpha=.35
        )

    _add_rule(
        bottom,
        bottom_linewidth
    )

    if save_path is not None:
        fig.savefig(
            f"{save_path}.pdf",
            dpi=dpi,
            bbox_inches="tight"
        )

        fig.savefig(
            f"{save_path}.png",
            dpi=dpi,
            bbox_inches="tight"
        )

    if show_plot:
        plt.show()

    return fig,ax,tbl,display

def find_validation_overlap_candidates(
    ms_val_df, assay_val_df, score_cols, method_prefixes, target_step_budget, target_learning_rate,
    ref_col="mhcprime", min_seeds=3, focus_seeds=None, n_uncertainty=1.0, uncertainty="se",
    overlap_mode_by_method=None, include_target=True, verbose=True,
):
    """
    Find configurations whose mean assay and/or MS validation performance lies
    within ±N SE/SD of a method-specific target endpoint.

    target_step_budget and target_learning_rate can each be:
      scalar -> same target for every method
      dict   -> method-specific targets, keyed by method name

    overlap_mode_by_method:
      None -> original behavior; every method uses "or"

      dict keyed by method name, with values:
        "or"    -> assay OR MS overlap
        "and"   -> assay AND MS overlap
        "ms"    -> MS overlap only
        "assay" -> assay overlap only

      Example:
        overlap_mode_by_method={
            "ft":"assay",
            "pepcl_r_dr":"or",
        }

    Returns
    -------
    candidate_df, overlap_df, target_df
    """
    import re
    import numpy as np
    import pandas as pd

    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty dict.")
    if uncertainty not in {"se","sd"}:
        raise ValueError("uncertainty must be 'se' or 'sd'.")

    try:
        n_uncertainty=float(n_uncertainty)
    except (TypeError,ValueError):
        raise TypeError("n_uncertainty must be numeric.")

    if not np.isfinite(n_uncertainty) or n_uncertainty<0:
        raise ValueError("n_uncertainty must be finite and >= 0.")
    if min_seeds<1:
        raise ValueError("min_seeds must be >= 1.")

    valid_overlap_modes={"or","and","ms","assay"}

    if overlap_mode_by_method is None:
        overlap_modes={m:"or" for m in method_prefixes}
    else:
        if not isinstance(overlap_mode_by_method,dict):
            raise TypeError("overlap_mode_by_method must be None or a dictionary.")

        missing=[m for m in method_prefixes if m not in overlap_mode_by_method]
        extra=[m for m in overlap_mode_by_method if m not in method_prefixes]

        if missing:
            raise ValueError(f"overlap_mode_by_method missing methods: {missing}")
        if extra:
            raise ValueError(f"overlap_mode_by_method contains unknown methods: {extra}")

        overlap_modes={}
        for method,mode in overlap_mode_by_method.items():
            mode=str(mode).lower()
            if mode not in valid_overlap_modes:
                raise ValueError(
                    f"Invalid overlap mode '{mode}' for method '{method}'. "
                    f"Use one of {sorted(valid_overlap_modes)}."
                )
            overlap_modes[method]=mode

    if focus_seeds is not None:
        if np.isscalar(focus_seeds):
            focus_seeds=[focus_seeds]

        try:
            focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in focus_seeds)))
        except (TypeError,ValueError):
            raise TypeError("focus_seeds must contain integer replicate IDs.")

        if not focus_seeds:
            raise ValueError("focus_seeds cannot be empty.")

    for df,name in [(ms_val_df,"ms_val_df"),(assay_val_df,"assay_val_df")]:
        if not df.columns.is_unique:
            dup=df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"{name} contains duplicate columns: {dup[:10]}")

    if ref_col not in ms_val_df.columns or ref_col not in assay_val_df.columns:
        raise ValueError(f"Reference '{ref_col}' must exist in both validation dataframes.")

    def _normalize_lr(lr):
        s=str(lr)
        if "eN" in s:
            return s
        try:
            m,e=f"{float(lr):.12e}".split("e")
            m=m.rstrip("0").rstrip(".")
            e=int(e)
            return f"{m}eN{abs(e)}" if e<0 else f"{m}e{e}"
        except (TypeError,ValueError):
            return s

    def _resolve(value,name,fn):
        if isinstance(value,dict):
            missing=[m for m in method_prefixes if m not in value]
            extra=[m for m in value if m not in method_prefixes]

            if missing:
                raise ValueError(f"{name} missing methods: {missing}")
            if extra:
                raise ValueError(f"{name} contains unknown methods: {extra}")

            return {m:fn(value[m]) for m in method_prefixes}

        return {m:fn(value) for m in method_prefixes}

    target_budgets=_resolve(target_step_budget,"target_step_budget",int)
    target_lrs=_resolve(target_learning_rate,"target_learning_rate",_normalize_lr)

    score_cols=list(dict.fromkeys(score_cols))
    candidates=[c for c in score_cols if c!=ref_col]
    usable=[c for c in candidates if c in ms_val_df.columns and c in assay_val_df.columns]

    if not usable:
        raise ValueError("No non-reference score columns are present in both validation dataframes.")

    if verbose:
        print(f"Score columns supplied: {len(score_cols)}")
        print(f"Candidate columns present in both dfs: {len(usable)}")
        print("Targets:")

        for m in method_prefixes:
            print(
                f"  {m}: steps={target_budgets[m]}, LR={target_lrs[m]}, "
                f"overlap_mode={overlap_modes[m]}"
            )

        print(
            f"Replicates: "
            f"{'focus '+str(focus_seeds) if focus_seeds is not None else f'min_seeds={min_seeds}'}"
        )
        print(f"Overlap criterion: ±{n_uncertainty:g} {uncertainty.upper()} of target mean")

    def _validate_mask(df,cols,name):
        base=df[ref_col].notna()
        bad=[c for c in cols if not df[c].notna().equals(base)]

        if bad:
            raise ValueError(
                f"{name}: {len(bad)} candidates do not share the evaluable allele set "
                f"of '{ref_col}'. Examples: {bad[:10]}"
            )

        out=df.loc[base,[ref_col]+cols].copy()

        if out.empty:
            raise ValueError(f"{name}: no evaluable rows remain.")

        if verbose:
            print(f"{name}: common evaluable allele set = {len(out)} rows.")

        return out

    ms_metric=_validate_mask(ms_val_df,usable,"MS").mean(axis=0)
    assay_metric=_validate_mask(assay_val_df,usable,"Assay").mean(axis=0)

    ordered_methods=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix):
                return name
        return None

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
        r"(?:_step_(?P<checkpoint>\d+))?$"
    )

    rows=[]
    bad=[]

    for col in usable:
        method=_method(col)
        m=pattern.search(str(col))

        if method is None or m is None:
            bad.append(col)
            continue

        budget=int(m["budget"])
        seed=int(m["seed"])
        cp=m["checkpoint"]
        is_full=cp is None
        checkpoint=budget if is_full else int(cp)

        if checkpoint>budget:
            bad.append(col)
            continue

        rows.append({
            "score_col":col,
            "method":method,
            "learning_rate":m["lr"],
            "step_budget":budget,
            "checkpoint":checkpoint,
            "seed":seed,
            "model_type":"full" if is_full else "checkpoint",
            "is_full_model":is_full,
            "is_terminal_endpoint":checkpoint==budget,
            "assay_performance":float(assay_metric[col]),
            "ms_performance":float(ms_metric[col]),
        })

    seed_df=pd.DataFrame(rows)

    if seed_df.empty:
        raise ValueError("No valid candidate score columns could be parsed.")

    if verbose and bad:
        print(f"Skipped {len(bad)} unparsable/invalid columns.")

    before=len(seed_df)
    seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)

    seed_df=(
        seed_df.sort_values(
            ["method","learning_rate","step_budget","seed","checkpoint","_prefer_full"],
            ascending=[True,True,True,True,True,False]
        )
        .drop_duplicates(
            ["method","learning_rate","step_budget","seed","checkpoint"],
            keep="first"
        )
        .drop(columns="_prefer_full")
        .reset_index(drop=True)
    )

    if verbose:
        print(f"Duplicate full/final-checkpoint endpoints removed: {before-len(seed_df)}")

    candidate_keys=["method","learning_rate","step_budget","checkpoint"]

    if focus_seeds is not None:
        before=len(seed_df)
        seed_df=seed_df[seed_df["seed"].isin(focus_seeds)].copy()

        if seed_df.empty:
            raise ValueError(f"No candidates remain for focus_seeds={focus_seeds}.")

        if verbose:
            print(f"Focused replicate rows: {before} -> {len(seed_df)}")

    seed_info=(
        seed_df.groupby(candidate_keys)["seed"]
        .agg(
            seed_ids=lambda x:tuple(sorted(set(x))),
            n_seeds="nunique"
        )
        .reset_index()
    )

    if focus_seeds is not None:
        required=tuple(focus_seeds)
        seed_info=seed_info[seed_info["seed_ids"].map(lambda x:x==required)].copy()

        if seed_info.empty:
            raise ValueError(f"No candidates contain all focused replicate IDs {required}.")

    else:
        seed_info=seed_info[seed_info["n_seeds"]>=min_seeds].copy()

        if seed_info.empty:
            raise ValueError(f"No candidates have at least {min_seeds} distinct seeds.")

        valid=[]

        for method in method_prefixes:
            s=seed_info[seed_info["method"]==method].copy()

            if s.empty:
                continue

            sets=s["seed_ids"].drop_duplicates().tolist()

            if len(sets)>1:
                raise ValueError(
                    f"Method '{method}' has inconsistent seed identifiers: {sets}. "
                    "Filter to a common seed set."
                )

            valid.append(s)

        if not valid:
            raise ValueError("No methods contain eligible candidates.")

        seed_info=pd.concat(valid,ignore_index=True)

    valid_keys=set(map(tuple,seed_info[candidate_keys].to_numpy()))
    seed_df=seed_df[seed_df[candidate_keys].apply(tuple,axis=1).isin(valid_keys)].copy()

    candidate_df=(
        seed_df.groupby(candidate_keys,sort=False)
        .agg(
            assay_performance_mean=("assay_performance","mean"),
            assay_performance_std=("assay_performance","std"),
            ms_performance_mean=("ms_performance","mean"),
            ms_performance_std=("ms_performance","std"),
            n_seeds=("seed","nunique"),
            seed_ids=("seed",lambda x:tuple(sorted(set(x))))
        )
        .reset_index()
    )

    candidate_df["assay_performance_se"]=(
        candidate_df["assay_performance_std"]/
        np.sqrt(candidate_df["n_seeds"])
    )

    candidate_df["ms_performance_se"]=(
        candidate_df["ms_performance_std"]/
        np.sqrt(candidate_df["n_seeds"])
    )

    raw=[]

    for keys,g in seed_df.groupby(candidate_keys,sort=False):
        g=g.sort_values("seed")
        row=dict(zip(candidate_keys,keys))

        rep=(
            "full" if g["is_full_model"].all()
            else "checkpoint" if (~g["is_full_model"]).all()
            else "mixed_endpoint"
        )

        row.update({
            "model_type":rep,
            "is_terminal_endpoint":bool((g["checkpoint"]==g["step_budget"]).all()),
            "score_cols_by_seed":dict(zip(g["seed"],g["score_col"]))
        })

        raw.append(row)

    candidate_df=candidate_df.merge(
        pd.DataFrame(raw),
        on=candidate_keys,
        how="left"
    )

    candidate_df["actual_steps"]=candidate_df["checkpoint"]

    # ------------------------------------------------------------------
    # Identify method-specific target endpoints.
    # ------------------------------------------------------------------
    target_rows=[]

    for method in method_prefixes:
        d=candidate_df[candidate_df["method"]==method]

        if d.empty:
            continue

        tb=target_budgets[method]
        tlr=target_lrs[method]

        target=d[
            (d["step_budget"]==tb)
            & (d["learning_rate"].astype(str)==tlr)
            & (d["checkpoint"]==tb)
        ].copy()

        if target.empty:
            raise ValueError(
                f"No target endpoint for '{method}' with steps={tb}, LR={tlr}."
            )

        if len(target)!=1:
            raise ValueError(
                f"Expected one aggregated target for '{method}', found {len(target)}."
            )

        target_rows.append(target.iloc[0])

    if not target_rows:
        raise ValueError("No target configurations found.")

    target_df=pd.DataFrame(target_rows).reset_index(drop=True)

    assay_u_col=(
        "assay_performance_se"
        if uncertainty=="se"
        else "assay_performance_std"
    )

    ms_u_col=(
        "ms_performance_se"
        if uncertainty=="se"
        else "ms_performance_std"
    )

    if target_df[[assay_u_col,ms_u_col]].isna().any().any():
        raise ValueError(
            f"Target {uncertainty.upper()} contains NaN; "
            "at least two replicates are required."
        )

    # ------------------------------------------------------------------
    # Calculate overlap bands and apply per-method selection rule.
    # ------------------------------------------------------------------
    pieces=[]

    for _,target in target_df.iterrows():
        method=target["method"]
        tb=target_budgets[method]
        tlr=target_lrs[method]
        overlap_mode=overlap_modes[method]

        d=candidate_df[candidate_df["method"]==method].copy()

        assay_center=float(target["assay_performance_mean"])
        ms_center=float(target["ms_performance_mean"])

        assay_u=float(target[assay_u_col])
        ms_u=float(target[ms_u_col])

        assay_low=assay_center-n_uncertainty*assay_u
        assay_high=assay_center+n_uncertainty*assay_u

        ms_low=ms_center-n_uncertainty*ms_u
        ms_high=ms_center+n_uncertainty*ms_u

        d["target_step_budget"]=tb
        d["target_learning_rate"]=tlr

        d["target_assay_mean"]=assay_center
        d[f"target_assay_{uncertainty}"]=assay_u
        d["target_assay_band_low"]=assay_low
        d["target_assay_band_high"]=assay_high

        d["target_ms_mean"]=ms_center
        d[f"target_ms_{uncertainty}"]=ms_u
        d["target_ms_band_low"]=ms_low
        d["target_ms_band_high"]=ms_high

        d["assay_delta_from_target"]=(
            d["assay_performance_mean"]-assay_center
        )

        d["ms_delta_from_target"]=(
            d["ms_performance_mean"]-ms_center
        )

        d["assay_overlap"]=d["assay_performance_mean"].between(
            assay_low,
            assay_high,
            inclusive="both"
        )

        d["ms_overlap"]=d["ms_performance_mean"].between(
            ms_low,
            ms_high,
            inclusive="both"
        )

        # Keep original derived columns for backward compatibility.
        d["overlap_either"]=d["assay_overlap"]|d["ms_overlap"]
        d["overlap_both"]=d["assay_overlap"]&d["ms_overlap"]

        # New selected criterion.
        if overlap_mode=="or":
            d["overlap_selected"]=d["overlap_either"]

        elif overlap_mode=="and":
            d["overlap_selected"]=d["overlap_both"]

        elif overlap_mode=="ms":
            d["overlap_selected"]=d["ms_overlap"]

        elif overlap_mode=="assay":
            d["overlap_selected"]=d["assay_overlap"]

        d["overlap_mode"]=overlap_mode

        d["is_target"]=(
            (d["step_budget"]==tb)
            & (d["learning_rate"].astype(str)==tlr)
            & (d["checkpoint"]==tb)
        )

        pieces.append(d)

    candidate_df=pd.concat(pieces,ignore_index=True)

    # This is the only behavioral change:
    # filter according to the requested method-specific overlap criterion.
    overlap_df=candidate_df[candidate_df["overlap_selected"]].copy()

    if not include_target:
        overlap_df=overlap_df[~overlap_df["is_target"]].copy()

    overlap_df=overlap_df.sort_values(
        [
            "method",
            "overlap_both",
            "assay_overlap",
            "ms_overlap",
            "checkpoint",
            "step_budget",
            "learning_rate",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            True,
            True,
        ]
    ).reset_index(drop=True)

    if verbose:
        print(
            f"\n{'='*90}\n"
            f"VALIDATION OVERLAP ANALYSIS\n"
            f"{'='*90}"
        )

        for method in target_df["method"]:
            target=target_df[target_df["method"]==method].iloc[0]

            tb=target_budgets[method]
            tlr=target_lrs[method]
            mode=overlap_modes[method]

            sub=overlap_df[
                overlap_df["method"]==method
            ]

            print(f"\nMETHOD: {method}")
            print(
                f"Target: LR={tlr}, steps={tb}, "
                f"replicates={target['seed_ids']}"
            )

            print(
                f"Assay: {target['assay_performance_mean']:.6f} "
                f"± {n_uncertainty:g}×{uncertainty.upper()} "
                f"({target[assay_u_col]:.6f})"
            )

            print(
                f"MS:    {target['ms_performance_mean']:.6f} "
                f"± {n_uncertainty:g}×{uncertainty.upper()} "
                f"({target[ms_u_col]:.6f})"
            )

            print(f"Overlap mode: {mode}")
            print(f"Candidates satisfying selected criterion: {len(sub)}")

            if not sub.empty:
                print(
                    sub[[
                        "learning_rate",
                        "step_budget",
                        "checkpoint",
                        "model_type",
                        "n_seeds",
                        "seed_ids",
                        "assay_performance_mean",
                        "assay_performance_std",
                        "assay_performance_se",
                        "ms_performance_mean",
                        "ms_performance_std",
                        "ms_performance_se",
                        "assay_overlap",
                        "ms_overlap",
                        "overlap_either",
                        "overlap_both",
                        "overlap_selected",
                        "overlap_mode",
                        "is_target",
                    ]].to_string(index=False)
                )

    return candidate_df,overlap_df,target_df

import numpy as np
import pandas as pd

def summarize_overlap_hyperparameter_frequency(
    overlap_dfs, method_col="method", lr_col="learning_rate", step_col="checkpoint",
    budget_col="step_budget", include_step_budget=False, include_target=True,
    sort=True, verbose=True,
):
    """
    Summarize how frequently each LR x step combination appears across a dict
    of validation-overlap dataframes.

    Parameters
    ----------
    overlap_dfs : dict[str, pd.DataFrame]
        Dataset/assay name -> overlap_df returned by
        find_validation_overlap_candidates (or equivalent helper).

    method_col : str
        Method column.

    lr_col : str
        Learning-rate column.

    step_col : str
        Actual optimization-step column. Usually "checkpoint".

    budget_col : str
        Total training-budget column.

    include_step_budget : bool
        False -> frequency is based on method x LR x actual step.
                 Different total-budget trajectories reaching the same LR/step
                 are collapsed within each dataset.

        True  -> method x LR x total budget x actual step are treated as
                 distinct configurations.

    include_target : bool
        If False and "is_target" exists, target rows are removed before
        calculating frequencies.

    Returns
    -------
    frequency_df : pd.DataFrame
        One row per method/configuration with:
          frequency
          frequency_fraction
          datasets
          n_datasets_total

    method_frequency_dfs : dict[str, pd.DataFrame]
        Same results separated by method.
    """
    if not isinstance(overlap_dfs,dict) or not overlap_dfs:
        raise ValueError("overlap_dfs must be a non-empty dict of dataset -> DataFrame.")

    group_cols=[method_col,lr_col]
    if include_step_budget: group_cols.append(budget_col)
    group_cols.append(step_col)

    rows=[]
    for dataset,df in overlap_dfs.items():
        if not isinstance(df,pd.DataFrame):
            raise TypeError(f"overlap_dfs['{dataset}'] must be a DataFrame.")

        required=[method_col,lr_col,step_col]+([budget_col] if include_step_budget else [])
        missing=[c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset '{dataset}' is missing columns: {missing}")

        d=df.copy()
        if not include_target and "is_target" in d.columns:
            d=d[~d["is_target"].fillna(False)].copy()

        # A configuration contributes at most once per dataset.
        d=d[group_cols].dropna().drop_duplicates()
        d["dataset"]=dataset
        rows.append(d)

    presence_df=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=group_cols+["dataset"])
    if presence_df.empty:
        frequency_df=pd.DataFrame(columns=group_cols+["frequency","frequency_fraction","datasets","n_datasets_total"])
        return frequency_df,{}

    n_datasets=len(overlap_dfs)

    frequency_df=(presence_df.groupby(group_cols,sort=False)
        .agg(
            frequency=("dataset","nunique"),
            datasets=("dataset",lambda x:tuple(sorted(set(x))))
        )
        .reset_index())

    frequency_df["frequency_fraction"]=frequency_df["frequency"]/n_datasets
    frequency_df["n_datasets_total"]=n_datasets

    if sort:
        sort_cols=[method_col,"frequency",step_col,lr_col]
        frequency_df=frequency_df.sort_values(
            sort_cols,
            ascending=[True,False,True,True]
        ).reset_index(drop=True)

    method_frequency_dfs={
        method:g.reset_index(drop=True)
        for method,g in frequency_df.groupby(method_col,sort=False)
    }

    if verbose:
        print(f"Datasets analyzed: {n_datasets}")
        print(f"Dataset keys: {list(overlap_dfs)}")
        print(f"Configuration definition: LR + {'budget + ' if include_step_budget else ''}step")

        for method,g in method_frequency_dfs.items():
            print(f"\n{'='*80}")
            print(f"METHOD: {method}")
            print(f"{'='*80}")
            show=[lr_col]
            if include_step_budget: show.append(budget_col)
            show += [step_col,"frequency","frequency_fraction","datasets"]
            print(g[show].to_string(index=False))

    return frequency_df,method_frequency_dfs

import re
import numpy as np
import pandas as pd

def summarize_pepcl_default_operating_point(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes,
    optimal_lr, optimal_step, default_lr, default_step,
    pepcl_method="pepcl_r_dr", ref_col="mhcprime",
    optimal_step_budget=None, default_step_budget=None,
    min_seeds=1, focus_seeds=None, verbose=True,
):
    """
    Compare a default PepCL operating point with a validation-selected/optimal
    PepCL point.

    Reports:
      assay_gain_recovery_pct =
          100 * (A_default - A_base) / (A_optimal - A_base)

      baseline_ms_retained_pct =
          100 * M_default / M_base

      ms_default_minus_optimal =
          M_default - M_optimal

    Returns
    -------
    summary_df : one-row dataframe with raw and derived statistics
    candidate_df : aggregated PepCL candidate dataframe
    """
    if pepcl_method not in method_prefixes:
        raise ValueError(f"pepcl_method='{pepcl_method}' not found in method_prefixes.")

    def _lr_string(lr):
        s=str(lr)
        if "eN" in s: return s
        try:
            mant,exp=f"{float(lr):.12e}".split("e"); mant=mant.rstrip("0").rstrip("."); exp=int(exp)
            return f"{mant}eN{abs(exp)}" if exp<0 else f"{mant}e{exp}"
        except (TypeError,ValueError): return s

    if focus_seeds is not None:
        if np.isscalar(focus_seeds): focus_seeds=[focus_seeds]
        focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in focus_seeds)))
        if not focus_seeds: raise ValueError("focus_seeds cannot be empty.")

    optimal_lr=_lr_string(optimal_lr); default_lr=_lr_string(default_lr)
    optimal_step=int(optimal_step); default_step=int(default_step)

    if ref_col not in ms_ap_df.columns or ref_col not in assay_ap_df.columns:
        raise ValueError(f"Reference '{ref_col}' must exist in both AP dataframes.")

    score_cols=list(dict.fromkeys(score_cols))
    prefix=method_prefixes[pepcl_method]
    usable=[c for c in score_cols if c!=ref_col and c in ms_ap_df.columns and c in assay_ap_df.columns and str(c).startswith(prefix)]
    if not usable: raise ValueError(f"No PepCL score columns found for prefix '{prefix}'.")

    # Require a common evaluable allele mask independently for MS and assay.
    for df,name in [(ms_ap_df,"MS"),(assay_ap_df,"Assay")]:
        base=df[ref_col].notna()
        bad=[c for c in usable if not df[c].notna().equals(base)]
        if bad: raise ValueError(f"{name}: {len(bad)} candidates do not share the reference evaluable allele mask. Examples: {bad[:10]}")

    ms_map=ms_ap_df.loc[ms_ap_df[ref_col].notna(),[ref_col]+usable].mean(axis=0)
    assay_map=assay_ap_df.loc[assay_ap_df[ref_col].notna(),[ref_col]+usable].mean(axis=0)
    ms_base=float(ms_map[ref_col]); assay_base=float(assay_map[ref_col])

    pattern=re.compile(r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)(?:_step_(?P<checkpoint>\d+))?$")
    rows=[]
    for col in usable:
        m=pattern.search(str(col))
        if m is None: continue
        budget=int(m["budget"]); cp=m["checkpoint"]; is_full=cp is None
        checkpoint=budget if is_full else int(cp)
        if checkpoint>budget: continue
        rows.append({
            "score_col":col,"method":pepcl_method,"learning_rate":m["lr"],
            "step_budget":budget,"checkpoint":checkpoint,"seed":int(m["seed"]),
            "is_full_model":is_full,"assay_performance":float(assay_map[col]),
            "ms_performance":float(ms_map[col]),
        })

    seed_df=pd.DataFrame(rows)
    if seed_df.empty: raise ValueError("No valid PepCL candidates could be parsed.")

    # Explicit full model wins over duplicate `_step_budget`.
    seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)
    seed_df=(seed_df.sort_values(["learning_rate","step_budget","seed","checkpoint","_prefer_full"],
                                ascending=[True,True,True,True,False])
        .drop_duplicates(["learning_rate","step_budget","seed","checkpoint"],keep="first")
        .drop(columns="_prefer_full").reset_index(drop=True))

    keys=["method","learning_rate","step_budget","checkpoint"]

    if focus_seeds is not None:
        seed_df=seed_df[seed_df["seed"].isin(focus_seeds)].copy()
        info=(seed_df.groupby(keys)["seed"].agg(seed_ids=lambda x:tuple(sorted(set(x))),n_seeds="nunique").reset_index())
        info=info[info["seed_ids"].map(lambda x:x==focus_seeds)].copy()
        if info.empty: raise ValueError(f"No PepCL candidates contain all focused replicate IDs {focus_seeds}.")
        valid=set(map(tuple,info[keys].to_numpy()))
        seed_df=seed_df[seed_df[keys].apply(tuple,axis=1).isin(valid)].copy()

    candidate_df=(seed_df.groupby(keys,sort=False)
        .agg(
            assay_performance=("assay_performance","mean"),assay_sd=("assay_performance","std"),
            ms_performance=("ms_performance","mean"),ms_sd=("ms_performance","std"),
            n_seeds=("seed","nunique"),seed_ids=("seed",lambda x:tuple(sorted(set(x)))),
            score_cols_by_seed=("score_col",lambda x:tuple(x)),
        ).reset_index())

    if focus_seeds is None: candidate_df=candidate_df[candidate_df["n_seeds"]>=min_seeds].copy()
    candidate_df["assay_se"]=candidate_df["assay_sd"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["ms_se"]=candidate_df["ms_sd"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["actual_steps"]=candidate_df["checkpoint"]
    candidate_df["assay_gain_vs_base"]=candidate_df["assay_performance"]-assay_base
    candidate_df["ms_change_vs_base"]=candidate_df["ms_performance"]-ms_base

    def _get_point(lr,step,budget,name):
        d=candidate_df[(candidate_df["learning_rate"].astype(str)==lr)&(candidate_df["checkpoint"]==step)].copy()
        if budget is not None: d=d[d["step_budget"]==int(budget)]
        if d.empty:
            raise ValueError(f"No {name} PepCL point found for LR={lr}, step={step}"
                             +(f", budget={budget}" if budget is not None else "")+".")
        if len(d)>1:
            opts=d[["learning_rate","step_budget","checkpoint","seed_ids"]].to_dict("records")
            raise ValueError(f"{name} point is ambiguous; provide {name}_step_budget. Matches: {opts}")
        return d.iloc[0]

    optimal=_get_point(optimal_lr,optimal_step,optimal_step_budget,"optimal")
    default=_get_point(default_lr,default_step,default_step_budget,"default")

    A_opt=float(optimal["assay_performance"]); A_def=float(default["assay_performance"])
    M_opt=float(optimal["ms_performance"]); M_def=float(default["ms_performance"])
    assay_opt_gain=A_opt-assay_base; assay_def_gain=A_def-assay_base

    assay_gain_recovery=np.nan if np.isclose(assay_opt_gain,0) else 100.0*assay_def_gain/assay_opt_gain
    baseline_ms_retained=np.nan if np.isclose(ms_base,0) else 100.0*M_def/ms_base

    summary_df=pd.DataFrame([{
        # Reference
        "assay_base":assay_base,"ms_base":ms_base,

        # Optimal / validation-selected PepCL
        "optimal_lr":optimal["learning_rate"],"optimal_step_budget":int(optimal["step_budget"]),
        "optimal_step":int(optimal["checkpoint"]),"optimal_n_seeds":int(optimal["n_seeds"]),
        "optimal_seed_ids":optimal["seed_ids"],
        "optimal_assay":A_opt,"optimal_assay_sd":float(optimal["assay_sd"]),"optimal_assay_se":float(optimal["assay_se"]),
        "optimal_ms":M_opt,"optimal_ms_sd":float(optimal["ms_sd"]),"optimal_ms_se":float(optimal["ms_se"]),
        "optimal_assay_gain_vs_base":assay_opt_gain,"optimal_ms_change_vs_base":M_opt-ms_base,

        # Default PepCL
        "default_lr":default["learning_rate"],"default_step_budget":int(default["step_budget"]),
        "default_step":int(default["checkpoint"]),"default_n_seeds":int(default["n_seeds"]),
        "default_seed_ids":default["seed_ids"],
        "default_assay":A_def,"default_assay_sd":float(default["assay_sd"]),"default_assay_se":float(default["assay_se"]),
        "default_ms":M_def,"default_ms_sd":float(default["ms_sd"]),"default_ms_se":float(default["ms_se"]),
        "default_assay_gain_vs_base":assay_def_gain,"default_ms_change_vs_base":M_def-ms_base,

        # Requested operating-point statistics
        "assay_gain_recovery_pct":assay_gain_recovery,
        "baseline_ms_retained_pct":baseline_ms_retained,
        "ms_default_minus_optimal":M_def-M_opt,

        # Useful complementary quantities
        "assay_default_minus_optimal":A_def-A_opt,
        "assay_gain_not_recovered_pct":100.0-assay_gain_recovery if np.isfinite(assay_gain_recovery) else np.nan,
        "baseline_ms_lost_pct":100.0-baseline_ms_retained if np.isfinite(baseline_ms_retained) else np.nan,
    }])

    if verbose:
        print(f"Optimal PepCL: LR={optimal['learning_rate']}, budget={int(optimal['step_budget'])}, step={int(optimal['checkpoint'])}")
        print(f"  Assay={A_opt:.6f} ± {optimal['assay_sd']:.6f} SD | MS={M_opt:.6f} ± {optimal['ms_sd']:.6f} SD")
        print(f"Default PepCL: LR={default['learning_rate']}, budget={int(default['step_budget'])}, step={int(default['checkpoint'])}")
        print(f"  Assay={A_def:.6f} ± {default['assay_sd']:.6f} SD | MS={M_def:.6f} ± {default['ms_sd']:.6f} SD")
        print(f"\nAssay-gain recovery: {assay_gain_recovery:.2f}%")
        print(f"Baseline MS retained: {baseline_ms_retained:.2f}%")
        print(f"Signed MS difference, default - optimal: {M_def-M_opt:+.6f}")

    return summary_df,candidate_df


# def summarize_pepcl_default_operating_point_relative_to_optimal(
#     ms_ap_df, assay_ap_df, score_cols, method_prefixes,
#     optimal_lr, optimal_step, default_lr, default_step,
#     pepcl_method="pepcl_r_dr", ref_col="mhcprime",
#     optimal_step_budget=None, default_step_budget=None,
#     min_seeds=1, focus_seeds=None, ms_comparison="optimal", verbose=True,
# ):
#     """
#     Compare a default PepCL operating point with a validation-selected/optimal
#     PepCL point.

#     Reports:
#       assay_gain_recovery_pct =
#           100 * (A_default - A_base) / (A_optimal - A_base)

#       baseline_ms_retained_pct =
#           100 * M_default / M_base

#       optimal_ms_retained_pct =
#           100 * M_optimal / M_base

#       ms_retention_vs_optimal_pct =
#           100 * (M_default / M_base) / (M_optimal / M_base)
#         = 100 * M_default / M_optimal

#       ms_default_minus_optimal =
#           M_default - M_optimal

#     ms_comparison:
#       "optimal"  -> emphasize default MS retention relative to optimal
#       "baseline" -> emphasize absolute fraction of baseline MHCPrime MS retained
#       "both"     -> print both

#     Returns
#     -------
#     summary_df : one-row dataframe with raw and derived statistics
#     candidate_df : aggregated PepCL candidate dataframe
#     """
#     if pepcl_method not in method_prefixes:
#         raise ValueError(f"pepcl_method='{pepcl_method}' not found in method_prefixes.")
#     if ms_comparison not in {"optimal","baseline","both"}:
#         raise ValueError("ms_comparison must be 'optimal', 'baseline', or 'both'.")

#     def _lr_string(lr):
#         s=str(lr)
#         if "eN" in s: return s
#         try:
#             mant,exp=f"{float(lr):.12e}".split("e"); mant=mant.rstrip("0").rstrip("."); exp=int(exp)
#             return f"{mant}eN{abs(exp)}" if exp<0 else f"{mant}e{exp}"
#         except (TypeError,ValueError): return s

#     if focus_seeds is not None:
#         if np.isscalar(focus_seeds): focus_seeds=[focus_seeds]
#         focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in focus_seeds)))
#         if not focus_seeds: raise ValueError("focus_seeds cannot be empty.")

#     optimal_lr=_lr_string(optimal_lr); default_lr=_lr_string(default_lr)
#     optimal_step=int(optimal_step); default_step=int(default_step)

#     if ref_col not in ms_ap_df.columns or ref_col not in assay_ap_df.columns:
#         raise ValueError(f"Reference '{ref_col}' must exist in both AP dataframes.")

#     score_cols=list(dict.fromkeys(score_cols))
#     prefix=method_prefixes[pepcl_method]
#     usable=[c for c in score_cols if c!=ref_col and c in ms_ap_df.columns and c in assay_ap_df.columns and str(c).startswith(prefix)]
#     if not usable: raise ValueError(f"No PepCL score columns found for prefix '{prefix}'.")

#     # Require a common evaluable allele mask independently for MS and assay.
#     for df,name in [(ms_ap_df,"MS"),(assay_ap_df,"Assay")]:
#         base=df[ref_col].notna()
#         bad=[c for c in usable if not df[c].notna().equals(base)]
#         if bad: raise ValueError(f"{name}: {len(bad)} candidates do not share the reference evaluable allele mask. Examples: {bad[:10]}")

#     ms_map=ms_ap_df.loc[ms_ap_df[ref_col].notna(),[ref_col]+usable].mean(axis=0)
#     assay_map=assay_ap_df.loc[assay_ap_df[ref_col].notna(),[ref_col]+usable].mean(axis=0)
#     ms_base=float(ms_map[ref_col]); assay_base=float(assay_map[ref_col])

#     pattern=re.compile(r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)(?:_step_(?P<checkpoint>\d+))?$")
#     rows=[]
#     for col in usable:
#         m=pattern.search(str(col))
#         if m is None: continue
#         budget=int(m["budget"]); cp=m["checkpoint"]; is_full=cp is None
#         checkpoint=budget if is_full else int(cp)
#         if checkpoint>budget: continue
#         rows.append({
#             "score_col":col,"method":pepcl_method,"learning_rate":m["lr"],
#             "step_budget":budget,"checkpoint":checkpoint,"seed":int(m["seed"]),
#             "is_full_model":is_full,"assay_performance":float(assay_map[col]),
#             "ms_performance":float(ms_map[col]),
#         })

#     seed_df=pd.DataFrame(rows)
#     if seed_df.empty: raise ValueError("No valid PepCL candidates could be parsed.")

#     # Explicit full model wins over duplicate `_step_budget`.
#     seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)
#     seed_df=(seed_df.sort_values(["learning_rate","step_budget","seed","checkpoint","_prefer_full"],
#                                 ascending=[True,True,True,True,False])
#         .drop_duplicates(["learning_rate","step_budget","seed","checkpoint"],keep="first")
#         .drop(columns="_prefer_full").reset_index(drop=True))

#     keys=["method","learning_rate","step_budget","checkpoint"]

#     if focus_seeds is not None:
#         seed_df=seed_df[seed_df["seed"].isin(focus_seeds)].copy()
#         info=(seed_df.groupby(keys)["seed"].agg(seed_ids=lambda x:tuple(sorted(set(x))),n_seeds="nunique").reset_index())
#         info=info[info["seed_ids"].map(lambda x:x==focus_seeds)].copy()
#         if info.empty: raise ValueError(f"No PepCL candidates contain all focused replicate IDs {focus_seeds}.")
#         valid=set(map(tuple,info[keys].to_numpy()))
#         seed_df=seed_df[seed_df[keys].apply(tuple,axis=1).isin(valid)].copy()

#     candidate_df=(seed_df.groupby(keys,sort=False)
#         .agg(
#             assay_performance=("assay_performance","mean"),assay_sd=("assay_performance","std"),
#             ms_performance=("ms_performance","mean"),ms_sd=("ms_performance","std"),
#             n_seeds=("seed","nunique"),seed_ids=("seed",lambda x:tuple(sorted(set(x)))),
#             score_cols_by_seed=("score_col",lambda x:tuple(x)),
#         ).reset_index())

#     if focus_seeds is None: candidate_df=candidate_df[candidate_df["n_seeds"]>=min_seeds].copy()
#     candidate_df["assay_se"]=candidate_df["assay_sd"]/np.sqrt(candidate_df["n_seeds"])
#     candidate_df["ms_se"]=candidate_df["ms_sd"]/np.sqrt(candidate_df["n_seeds"])
#     candidate_df["actual_steps"]=candidate_df["checkpoint"]
#     candidate_df["assay_gain_vs_base"]=candidate_df["assay_performance"]-assay_base
#     candidate_df["ms_change_vs_base"]=candidate_df["ms_performance"]-ms_base

#     def _get_point(lr,step,budget,name):
#         d=candidate_df[(candidate_df["learning_rate"].astype(str)==lr)&(candidate_df["checkpoint"]==step)].copy()
#         if budget is not None: d=d[d["step_budget"]==int(budget)]
#         if d.empty:
#             raise ValueError(f"No {name} PepCL point found for LR={lr}, step={step}"
#                              +(f", budget={budget}" if budget is not None else "")+".")
#         if len(d)>1:
#             opts=d[["learning_rate","step_budget","checkpoint","seed_ids"]].to_dict("records")
#             raise ValueError(f"{name} point is ambiguous; provide {name}_step_budget. Matches: {opts}")
#         return d.iloc[0]

#     optimal=_get_point(optimal_lr,optimal_step,optimal_step_budget,"optimal")
#     default=_get_point(default_lr,default_step,default_step_budget,"default")

#     A_opt=float(optimal["assay_performance"]); A_def=float(default["assay_performance"])
#     M_opt=float(optimal["ms_performance"]); M_def=float(default["ms_performance"])
#     assay_opt_gain=A_opt-assay_base; assay_def_gain=A_def-assay_base

#     assay_gain_recovery=np.nan if np.isclose(assay_opt_gain,0) else 100.0*assay_def_gain/assay_opt_gain

#     # Absolute retention relative to original MHCPrime baseline.
#     baseline_ms_retained=np.nan if np.isclose(ms_base,0) else 100.0*M_def/ms_base
#     optimal_ms_retained=np.nan if np.isclose(ms_base,0) else 100.0*M_opt/ms_base

#     # Relative retention: how much MS performance the default retains compared
#     # with the validation-selected optimal PepCL point.
#     ms_retention_vs_optimal=np.nan if np.isclose(M_opt,0) else 100.0*M_def/M_opt

#     summary_df=pd.DataFrame([{
#         # Reference
#         "assay_base":assay_base,"ms_base":ms_base,

#         # Optimal / validation-selected PepCL
#         "optimal_lr":optimal["learning_rate"],"optimal_step_budget":int(optimal["step_budget"]),
#         "optimal_step":int(optimal["checkpoint"]),"optimal_n_seeds":int(optimal["n_seeds"]),
#         "optimal_seed_ids":optimal["seed_ids"],
#         "optimal_assay":A_opt,"optimal_assay_sd":float(optimal["assay_sd"]),"optimal_assay_se":float(optimal["assay_se"]),
#         "optimal_ms":M_opt,"optimal_ms_sd":float(optimal["ms_sd"]),"optimal_ms_se":float(optimal["ms_se"]),
#         "optimal_assay_gain_vs_base":assay_opt_gain,"optimal_ms_change_vs_base":M_opt-ms_base,

#         # Default PepCL
#         "default_lr":default["learning_rate"],"default_step_budget":int(default["step_budget"]),
#         "default_step":int(default["checkpoint"]),"default_n_seeds":int(default["n_seeds"]),
#         "default_seed_ids":default["seed_ids"],
#         "default_assay":A_def,"default_assay_sd":float(default["assay_sd"]),"default_assay_se":float(default["assay_se"]),
#         "default_ms":M_def,"default_ms_sd":float(default["ms_sd"]),"default_ms_se":float(default["ms_se"]),
#         "default_assay_gain_vs_base":assay_def_gain,"default_ms_change_vs_base":M_def-ms_base,

#         # Operating-point comparison
#         "assay_gain_recovery_pct":assay_gain_recovery,
#         "ms_retention_vs_optimal_pct":ms_retention_vs_optimal,
#         "ms_default_minus_optimal":M_def-M_opt,
#         "assay_default_minus_optimal":A_def-A_opt,

#         # Absolute MS retention relative to original MHCPrime
#         "baseline_ms_retained_pct":baseline_ms_retained,
#         "optimal_ms_retained_pct":optimal_ms_retained,
#         "baseline_ms_lost_pct":100.0-baseline_ms_retained if np.isfinite(baseline_ms_retained) else np.nan,

#         # Complementary assay quantity
#         "assay_gain_not_recovered_pct":100.0-assay_gain_recovery if np.isfinite(assay_gain_recovery) else np.nan,
#     }])

#     if verbose:
#         print(f"Optimal PepCL: LR={optimal['learning_rate']}, budget={int(optimal['step_budget'])}, step={int(optimal['checkpoint'])}")
#         print(f"  Assay={A_opt:.6f} ± {optimal['assay_sd']:.6f} SD | MS={M_opt:.6f} ± {optimal['ms_sd']:.6f} SD")
#         print(f"Default PepCL: LR={default['learning_rate']}, budget={int(default['step_budget'])}, step={int(default['checkpoint'])}")
#         print(f"  Assay={A_def:.6f} ± {default['assay_sd']:.6f} SD | MS={M_def:.6f} ± {default['ms_sd']:.6f} SD")

#         print(f"\nAssay-gain recovery vs optimal: {assay_gain_recovery:.2f}%")
#         if ms_comparison in {"optimal","both"}:
#             print(f"MS retention vs optimal: {ms_retention_vs_optimal:.2f}%")
#         if ms_comparison in {"baseline","both"}:
#             print(f"Default MS retained vs MHCPrime baseline: {baseline_ms_retained:.2f}%")
#             print(f"Optimal MS retained vs MHCPrime baseline: {optimal_ms_retained:.2f}%")
#         print(f"Signed MS difference, default - optimal: {M_def-M_opt:+.6f}")

#     return summary_df,candidate_df

import re
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

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
    usable=[
        c for c in dict.fromkeys(score_cols)
        if c in ms_ap_df.columns
        and c in assay_ap_df.columns
        and c!=ref_col
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
            [ref_col]+model_cols
        ].copy()

    else:
        # Default behavior: each dataset retains its own allele population.
        ms=ms_ap_df.loc[
            :,
            [ref_col]+model_cols
        ].copy()

        assay=assay_ap_df.loc[
            :,
            [ref_col]+model_cols
        ].copy()

    # ---------------------------------------------------------
    # Raw seed/configuration table
    # ---------------------------------------------------------
    rows=[]

    for _,r in meta.iterrows():
        c=r["score_col"]

        if paired_allele_mode:
            for allele in ms.index:
                rows.append({
                    **r.to_dict(),
                    "allele":allele,
                    "ms_map":float(ms.loc[allele,c]),
                    "assay_map":float(assay.loc[allele,c]),
                    "ms_baseline":float(ms.loc[allele,ref_col]),
                    "assay_baseline":float(assay.loc[allele,ref_col]),
                })

        else:
            rows.append({
                **r.to_dict(),
                "ms_map":float(ms[c].mean()),
                "assay_map":float(assay[c].mean()),
                "ms_baseline":float(ms[ref_col].mean()),
                "assay_baseline":float(assay[ref_col].mean()),
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
        ["score_col","allele"]
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

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import t
from tqdm import tqdm


def compute_mixed_ms_holec_ft_lambda_delta_from_base(
    ms_ft_df: pd.DataFrame,
    holec_ft_df: pd.DataFrame,
    ms_lambda_df: pd.DataFrame,
    holec_lambda_df: pd.DataFrame,
    *,
    base_col: str,
    ft_score_cols: List[str],
    lambda_score_cols_by_group: Dict[str, List[str]],
    ft_group_name: str = "holec_ft",
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
    Compute mixed MS/Holec delta-mAP curves for:
      - FT across replicate seeds
      - PepCL-R-DR lambda values across replicate seeds

    FT predictions and lambda-sweep predictions may live in separate pairs
    of MS/Holec dataframes.

    Each curve is processed independently using only:
        [base_col] + replicate score columns

    This preserves the faster behavior of the original implementation rather
    than placing every lambda/seed column into one very wide score matrix.

    For each curve:
      1. Repeatedly construct MS/Holec mixtures.
      2. Compute allele-specific AP and macro-mAP.
      3. Compute paired delta-mAP versus base.
      4. Average repeated mixture samples within each replicate.
      5. Compute mean, variance and CI95 across replicate values.

    Returns
    -------
    mean_df
        Wide dataframe indexed by ms_percentage:
            holec_ft
            holec_pepcl_lambda_0
            holec_pepcl_lambda_1eN8
            ...

    var_df
        Same curve columns, containing variance across replicate-level values.

    ci_df
        Wide dataframe containing:
            <curve>_ci_low
            <curve>_ci_high

        Directly compatible with plot_lines().
    """

    # =========================================================
    # Original fast helpers
    # =========================================================
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
        d[label_col] = pd.to_numeric(
            d[label_col], errors="coerce"
        ).astype(np.int8)

        for sc in score_cols:
            d[sc] = pd.to_numeric(
                d[sc], errors="coerce"
            ).astype(np.float32)

        for allele, sub in d.groupby(allele_col, sort=False):
            pos = sub.loc[
                sub[label_col] == 1, score_cols
            ].to_numpy(dtype=np.float32, copy=True)

            neg = sub.loc[
                sub[label_col] == 0, score_cols
            ].to_numpy(dtype=np.float32, copy=True)

            out[str(allele)] = {
                "pos": pos,
                "neg": neg,
            }

        return out

    # =========================================================
    # Validation
    # =========================================================
    if not ft_score_cols:
        raise ValueError("ft_score_cols is empty.")

    if not lambda_score_cols_by_group:
        raise ValueError("lambda_score_cols_by_group is empty.")

    if neg_per_pos_ratio < 0:
        raise ValueError("neg_per_pos_ratio must be >= 0.")

    if not (0 < mix_step_pct <= 100):
        raise ValueError("mix_step_pct must be in (0, 100].")

    if n_iterations <= 0:
        raise ValueError("n_iterations must be > 0.")

    if not (0 < ci_alpha < 1):
        raise ValueError("ci_alpha must be in (0, 1).")

    ft_score_cols = list(ft_score_cols)
    lambda_score_cols_by_group = {
        str(group): list(cols)
        for group, cols in lambda_score_cols_by_group.items()
    }

    n_replicates = len(ft_score_cols)

    bad_groups = {
        group: len(cols)
        for group, cols in lambda_score_cols_by_group.items()
        if len(cols) != n_replicates
    }

    if bad_groups:
        raise ValueError(
            f"FT has {n_replicates} replicate columns, but these lambda "
            f"groups differ: {bad_groups}"
        )

    if ft_group_name in lambda_score_cols_by_group:
        raise ValueError(
            f"ft_group_name={ft_group_name!r} already exists in "
            "lambda_score_cols_by_group."
        )

    # =========================================================
    # Mixture percentages shared by every curve
    # =========================================================
    ms_percentages = np.arange(
        100.0,
        -1e-9,
        -float(mix_step_pct),
    )

    ms_percentages = np.round(
        ms_percentages,
        10,
    )

    if ms_percentages[-1] != 0.0:
        ms_percentages = np.append(
            ms_percentages,
            0.0,
        )

    n_steps = len(ms_percentages)

    # =========================================================
    # One fast original-style computation for ONE curve/group
    # =========================================================
    def _compute_one_group(
        ms_df,
        holec_df,
        score_cols,
        group_name,
    ):
        all_score_cols = [base_col] + list(score_cols)

        _validate_required_columns(
            ms_df,
            [allele_col, ms_label_col] + all_score_cols,
            f"MS dataframe ({group_name})",
        )

        _validate_required_columns(
            holec_df,
            [allele_col, holec_label_col] + all_score_cols,
            f"Holec dataframe ({group_name})",
        )

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

        common_alleles = sorted(
            set(ms_map.keys()) &
            set(holec_map.keys())
        )

        if not common_alleles:
            raise ValueError(
                f"{group_name}: no overlapping alleles between MS and Holec."
            )

        # -----------------------------------------------------
        # Fixed allele sampling plan
        # -----------------------------------------------------
        allele_plan = {}

        for allele in common_alleles:
            ms_pos = ms_map[allele]["pos"]
            ms_neg = ms_map[allele]["neg"]
            holec_pos = holec_map[allele]["pos"]
            holec_neg = holec_map[allele]["neg"]

            min_pos = min(
                ms_pos.shape[0],
                holec_pos.shape[0],
            )

            if min_pos < 1:
                continue

            total_neg = int(
                min_pos * neg_per_pos_ratio
            )

            if ms_neg.shape[0] < total_neg:
                continue

            if holec_neg.shape[0] < total_neg:
                continue

            allele_plan[allele] = {
                "min_pos": int(min_pos),
                "total_neg": total_neg,
                "ms_pos": ms_pos,
                "ms_neg": ms_neg,
                "holec_pos": holec_pos,
                "holec_neg": holec_neg,
            }

        usable_alleles = sorted(
            allele_plan.keys()
        )

        if not usable_alleles:
            raise ValueError(
                f"{group_name}: no usable alleles after sampling constraints."
            )

        # Same random seed for every group -> same deterministic sampling
        # sequence whenever the underlying validation sets are identical.
        rng = np.random.default_rng(
            random_state
        )

        # mixture x iteration x replicate
        sampled_deltas = np.full(
            (
                n_steps,
                n_iterations,
                len(score_cols),
            ),
            np.nan,
            dtype=np.float64,
        )

        mix_bar = tqdm(
            enumerate(ms_percentages),
            total=n_steps,
            desc=group_name,
            leave=True,
            disable=not show_progress,
        )

        for step_idx, ms_pct in mix_bar:
            ms_frac = ms_pct / 100.0

            for it in range(n_iterations):
                ap_by_allele = np.full(
                    (
                        len(usable_alleles),
                        len(all_score_cols),
                    ),
                    np.nan,
                    dtype=np.float64,
                )

                for a_idx, allele in enumerate(
                    usable_alleles
                ):
                    plan = allele_plan[allele]

                    min_pos = plan["min_pos"]
                    total_neg = plan["total_neg"]

                    # -----------------------------------------
                    # Number drawn from each source
                    # -----------------------------------------
                    ms_pos_take = int(
                        np.rint(
                            min_pos * ms_frac
                        )
                    )

                    ms_pos_take = min(
                        max(ms_pos_take, 0),
                        min_pos,
                    )

                    holec_pos_take = (
                        min_pos -
                        ms_pos_take
                    )

                    ms_neg_take = int(
                        np.rint(
                            total_neg * ms_frac
                        )
                    )

                    ms_neg_take = min(
                        max(ms_neg_take, 0),
                        total_neg,
                    )

                    holec_neg_take = (
                        total_neg -
                        ms_neg_take
                    )

                    parts = []
                    labels_parts = []

                    # -----------------------------------------
                    # MS positives
                    # -----------------------------------------
                    if ms_pos_take > 0:
                        arr = plan["ms_pos"]

                        idx = rng.choice(
                            arr.shape[0],
                            size=ms_pos_take,
                            replace=False,
                        )

                        parts.append(
                            arr[idx]
                        )

                        labels_parts.append(
                            np.ones(
                                ms_pos_take,
                                dtype=np.uint8,
                            )
                        )

                    # -----------------------------------------
                    # Holec positives
                    # -----------------------------------------
                    if holec_pos_take > 0:
                        arr = plan["holec_pos"]

                        idx = rng.choice(
                            arr.shape[0],
                            size=holec_pos_take,
                            replace=False,
                        )

                        parts.append(
                            arr[idx]
                        )

                        labels_parts.append(
                            np.ones(
                                holec_pos_take,
                                dtype=np.uint8,
                            )
                        )

                    # -----------------------------------------
                    # MS negatives
                    # -----------------------------------------
                    if ms_neg_take > 0:
                        arr = plan["ms_neg"]

                        idx = rng.choice(
                            arr.shape[0],
                            size=ms_neg_take,
                            replace=False,
                        )

                        parts.append(
                            arr[idx]
                        )

                        labels_parts.append(
                            np.zeros(
                                ms_neg_take,
                                dtype=np.uint8,
                            )
                        )

                    # -----------------------------------------
                    # Holec negatives
                    # -----------------------------------------
                    if holec_neg_take > 0:
                        arr = plan["holec_neg"]

                        idx = rng.choice(
                            arr.shape[0],
                            size=holec_neg_take,
                            replace=False,
                        )

                        parts.append(
                            arr[idx]
                        )

                        labels_parts.append(
                            np.zeros(
                                holec_neg_take,
                                dtype=np.uint8,
                            )
                        )

                    scores_mix = np.vstack(
                        parts
                    )

                    labels_mix = np.concatenate(
                        labels_parts
                    )

                    ap_by_allele[a_idx] = (
                        _fast_average_precision_multi(
                            scores_mix,
                            labels_mix,
                        )
                    )

                # Macro-average across alleles.
                map_iter = np.nanmean(
                    ap_by_allele,
                    axis=0,
                )

                # Column 0 = common base.
                sampled_deltas[
                    step_idx,
                    it,
                ] = (
                    map_iter[1:] -
                    map_iter[0]
                )

        # -----------------------------------------------------
        # Average repeated mixture draws within each replicate.
        #
        # shape = mixture x replicate
        # -----------------------------------------------------
        replicate_deltas = np.nanmean(
            sampled_deltas,
            axis=1,
        )

        # -----------------------------------------------------
        # Aggregate five replicate curves
        # -----------------------------------------------------
        mean = np.nanmean(
            replicate_deltas,
            axis=1,
        )

        var = np.full(
            n_steps,
            np.nan,
            dtype=np.float64,
        )

        ci_low = np.full(
            n_steps,
            np.nan,
            dtype=np.float64,
        )

        ci_high = np.full(
            n_steps,
            np.nan,
            dtype=np.float64,
        )

        confidence_q = (
            1.0 + ci_alpha
        ) / 2.0

        for i in range(n_steps):
            vals = replicate_deltas[i]
            vals = vals[
                np.isfinite(vals)
            ]

            if len(vals) < 2:
                continue

            var[i] = np.var(
                vals,
                ddof=1,
            )

            sd = np.sqrt(
                var[i]
            )

            se = sd / np.sqrt(
                len(vals)
            )

            ci_half = (
                t.ppf(
                    confidence_q,
                    df=len(vals) - 1,
                ) *
                se
            )

            ci_low[i] = (
                mean[i] -
                ci_half
            )

            ci_high[i] = (
                mean[i] +
                ci_half
            )

        return mean, var, ci_low, ci_high

    # =========================================================
    # Run FT + lambda groups independently
    # =========================================================
    mean_data = {}
    var_data = {}
    ci_data = {}

    total_groups = (
        1 +
        len(lambda_score_cols_by_group)
    )

    print(
        f"Running {total_groups} curves "
        f"({n_replicates} replicates each)..."
    )

    # ---------------------------------------------------------
    # FT
    # ---------------------------------------------------------
    print(f"\n[1/{total_groups}] {ft_group_name}")

    mean, var, ci_low, ci_high = _compute_one_group(
        ms_ft_df,
        holec_ft_df,
        ft_score_cols,
        ft_group_name,
    )

    mean_data[ft_group_name] = mean
    var_data[ft_group_name] = var
    ci_data[f"{ft_group_name}_ci_low"] = ci_low
    ci_data[f"{ft_group_name}_ci_high"] = ci_high

    # ---------------------------------------------------------
    # Lambda curves
    # ---------------------------------------------------------
    for group_idx, (group_name, score_cols) in enumerate(
        lambda_score_cols_by_group.items(),
        start=2,
    ):
        print(
            f"\n[{group_idx}/{total_groups}] "
            f"{group_name}"
        )

        mean, var, ci_low, ci_high = _compute_one_group(
            ms_lambda_df,
            holec_lambda_df,
            score_cols,
            group_name,
        )

        mean_data[group_name] = mean
        var_data[group_name] = var
        ci_data[f"{group_name}_ci_low"] = ci_low
        ci_data[f"{group_name}_ci_high"] = ci_high

    # =========================================================
    # Wide output for plot_lines()
    # =========================================================
    mean_df = pd.DataFrame(
        mean_data,
        index=ms_percentages,
    )
    mean_df.index.name = "ms_percentage"

    var_df = pd.DataFrame(
        var_data,
        index=ms_percentages,
    )
    var_df.index.name = "ms_percentage"

    ci_df = pd.DataFrame(
        ci_data,
        index=ms_percentages,
    )
    ci_df.index.name = "ms_percentage"

    return mean_df, var_df, ci_df

import numpy as np
import pandas as pd

def compute_curve_crossovers(
    mean_df: pd.DataFrame,
    ci_df: pd.DataFrame = None,
    *,
    x_col: str = "ms_percentage",
    target_y: float = 0.0,
    curve_cols = None,
    require_single_crossing: bool = True,
) -> pd.DataFrame:
    """
    Compute the interpolated x-position where each curve crosses target_y.

    Intended for outputs from:
        compute_mixed_ms_holec_ft_lambda_delta_from_base(...)

    Parameters
    ----------
    mean_df : pd.DataFrame
        Wide dataframe containing one mean curve per column.
        `ms_percentage` may either be the index or a normal column.

    ci_df : pd.DataFrame or None
        Wide CI dataframe containing:
            <curve>_ci_low
            <curve>_ci_high

        If provided, crossover positions are also estimated independently
        from the lower and upper CI curves.

    x_col : str
        Name of x variable. Default "ms_percentage".

    target_y : float
        Horizontal level whose crossing is desired. Default 0.0,
        corresponding to delta-mAP = 0.

    curve_cols : list[str] or None
        Curves to process. If None, all mean-dataframe columns except x_col
        are used.

    require_single_crossing : bool
        If True, raises an error when a curve crosses target_y multiple times.
        If False, returns the first crossing in ascending x order.

    Returns
    -------
    pd.DataFrame with one row per curve:
        curve
        crossover_ms_percentage
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
        return d

    def _find_crossing(x, y):
        x=np.asarray(x, dtype=float)
        y=np.asarray(y, dtype=float)

        valid=np.isfinite(x) & np.isfinite(y)
        x=x[valid]
        y=y[valid]

        if len(x) < 2:
            return np.nan, np.nan, np.nan

        # Sort by increasing MS percentage so behavior is deterministic.
        order=np.argsort(x)
        x=x[order]
        y=y[order]

        z=y-target_y

        # Exact sampled point at target.
        exact=np.where(np.isclose(z, 0.0, rtol=0, atol=1e-15))[0]
        if len(exact):
            if require_single_crossing and len(exact) > 1:
                raise ValueError(
                    "Curve contains multiple exact target crossings."
                )
            i=exact[0]
            return float(x[i]), float(x[i]), float(x[i])

        # Adjacent sign changes.
        crossing_idx=np.where(
            (z[:-1] * z[1:]) < 0
        )[0]

        if len(crossing_idx) == 0:
            return np.nan, np.nan, np.nan

        if require_single_crossing and len(crossing_idx) > 1:
            raise ValueError(
                f"Curve crosses target_y={target_y} more than once."
            )

        i=crossing_idx[0]

        x1,x2=x[i],x[i+1]
        y1,y2=y[i],y[i+1]

        # Linear interpolation:
        # x_cross = x1 + (target-y1)*(x2-x1)/(y2-y1)
        x_cross = x1 + (
            (target_y-y1) * (x2-x1) / (y2-y1)
        )

        return float(x_cross), float(x1), float(x2)

    mean_d=_prepare(mean_df)

    if curve_cols is None:
        curve_cols=[
            c for c in mean_d.columns
            if c != x_col
        ]

    ci_d=_prepare(ci_df) if ci_df is not None else None

    rows=[]

    for curve in curve_cols:
        if curve not in mean_d.columns:
            raise ValueError(
                f"Curve {curve!r} not found in mean_df."
            )

        x=mean_d[x_col].to_numpy(float)
        y=mean_d[curve].to_numpy(float)

        cross, x1, x2 = _find_crossing(
            x,
            y
        )

        ci_cross_1=np.nan
        ci_cross_2=np.nan

        if ci_d is not None:
            low_col=f"{curve}_ci_low"
            high_col=f"{curve}_ci_high"

            if low_col not in ci_d.columns or high_col not in ci_d.columns:
                raise ValueError(
                    f"Missing CI columns for {curve!r}: "
                    f"{low_col!r}, {high_col!r}"
                )

            low_cross,_,_=_find_crossing(
                ci_d[x_col],
                ci_d[low_col]
            )

            high_cross,_,_=_find_crossing(
                ci_d[x_col],
                ci_d[high_col]
            )

            finite_crosses=[
                v for v in [low_cross, high_cross]
                if np.isfinite(v)
            ]

            if finite_crosses:
                ci_cross_1=min(finite_crosses)
                ci_cross_2=max(finite_crosses)

        rows.append({
            "curve":curve,
            "crossover_ms_percentage":cross,
            "crossover_delta_map":target_y,
            "crossover_ci_low":ci_cross_1,
            "crossover_ci_high":ci_cross_2,
            "lower_bracket_ms_percentage":x1,
            "upper_bracket_ms_percentage":x2,
        })

    return pd.DataFrame(rows)

def plot_lambda_crossovers(
    zero_crossover_df,
    ft_crossover_df,
    *,
    zero_metric="crossover_ms_percentage",
    ft_metrics=("crossover_ms_percentage", "crossover_delta_map"),

    # Labels
    zero_ylabel=None,
    ft_ylabels=None,
    panel_titles=None,
    xlabel="DR λ",

    # Curve naming
    curve_col="curve",
    ft_curve_name="holec_ft",
    lambda_order="ascending",

    # Line / point styling
    line_color="black",
    marker="o",
    marker_size=7,
    linewidth=1.0,
    linestyle="-",
    point_alpha=1.0,
    error_alpha=1.0,
    error_kwargs=None,

    # Optional reference lines
    hlines=None,
    hline_kwargs=None,

    # Axis limits
    ylims=None,

    # Figure geometry
    figsize=(16,5),
    square_panels=True,
    wspace=0.42,

    # General styling
    dpi=300,
    axis_label_fontsize=11,
    title_fontsize=12,
    tick_decimals=2,
    trim_tick_zeros=False,
    n_ticks=5,
    tick_fontsize=9,
    tick_length=4,
    tick_width=1.0,
    tick_pad=5,
    spine_width=1.0,
    grid=False,
    grid_alpha=.2,

    # Output
    save_path=None,
    show_plot=True,
):
    """
    Plot three crossover summaries.

    Panel 1:
        ΔmAP=0 crossover for FT followed by PepCL λ values.
        x-axis:
            FT, 0, 1e-8, ..., 1e-2

    Panels 2-3:
        PepCL-vs-FT crossover metrics as functions of λ.
        x-axis:
            0, 1e-8, ..., 1e-2

    NaN crossover values are not plotted, but their categorical x-axis
    positions remain visible.
    """
    import re
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

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _parse_lambda_curve(x):
        s=str(x)
        m=re.search(r"lambda_([^_]+)$",s)

        if m is None:
            return None

        lam_s=m.group(1)

        if lam_s=="0":
            return 0.0,"0"

        val=float(
            lam_s.lower().replace("en","e-")
        )

        label=(
            f"{val:.0e}"
            .replace("e-0","e-")
            .replace("e+0","e+")
        )

        return val,label

    def _yfmt(v,_):
        if abs(v)<10**(-(tick_decimals+2)):
            v=0

        s=f"{v:.{tick_decimals}f}"

        return (
            s.rstrip("0").rstrip(".")
            if trim_tick_zeros
            else s
        )

    def _get_ci_cols(d,metric):
        if (
            metric=="crossover_ms_percentage"
            and "crossover_ci_low" in d.columns
            and "crossover_ci_high" in d.columns
        ):
            return "crossover_ci_low","crossover_ci_high"

        low=f"{metric}_ci_low"
        high=f"{metric}_ci_high"

        if low in d.columns and high in d.columns:
            return low,high

        return None,None

    def _draw_panel(
        ax,d,metric,x_positions,x_labels,
        ylabel,title,ylim,hline
    ):
        yy=pd.to_numeric(
            d[metric],
            errors="coerce"
        ).to_numpy(float)

        xx=np.asarray(
            x_positions,
            dtype=float
        )

        finite=np.isfinite(yy)

        # Line + points only through finite observations.
        ax.plot(
            xx[finite],
            yy[finite],
            color=line_color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=point_alpha,
            zorder=2
        )

        ax.scatter(
            xx[finite],
            yy[finite],
            s=marker_size**2,
            marker=marker,
            color=line_color,
            edgecolor="black",
            linewidth=.6,
            alpha=point_alpha,
            zorder=4
        )

        # Error bars when present.
        low_col,high_col=_get_ci_cols(
            d,
            metric
        )

        if low_col is not None:
            lows=pd.to_numeric(
                d[low_col],
                errors="coerce"
            ).to_numpy(float)

            highs=pd.to_numeric(
                d[high_col],
                errors="coerce"
            ).to_numpy(float)

            for x_i,y_i,lo,hi in zip(
                xx,yy,lows,highs
            ):
                if not (
                    np.isfinite(y_i)
                    and np.isfinite(lo)
                    and np.isfinite(hi)
                ):
                    continue

                yerr=np.array([
                    [max(0.,y_i-lo)],
                    [max(0.,hi-y_i)]
                ])

                ax.errorbar(
                    x_i,
                    y_i,
                    yerr=yerr,
                    fmt="none",
                    ecolor=line_color,
                    alpha=error_alpha,
                    **error_kwargs
                )

        if hline is not None:
            ax.axhline(
                hline,
                **hline_kwargs
            )

        ax.set_xticks(
            np.arange(len(x_labels))
        )

        ax.set_xticklabels(
            x_labels,
            fontsize=tick_fontsize,
            rotation=0
        )

        ax.set_xlim(
            -.35,
            len(x_labels)-.65
        )

        ax.set_xlabel(
            xlabel,
            fontsize=axis_label_fontsize,
            labelpad=8
        )

        ax.set_ylabel(
            ylabel,
            fontsize=axis_label_fontsize,
            labelpad=8
        )

        if title:
            ax.set_title(
                title,
                fontsize=title_fontsize
            )

        if ylim is not None:
            ax.set_ylim(ylim)

        if n_ticks is not None:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=n_ticks)
            )

        ax.yaxis.set_major_formatter(
            FuncFormatter(_yfmt)
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
    # Validate
    # ---------------------------------------------------------
    if len(ft_metrics)!=2:
        raise ValueError(
            "ft_metrics must contain exactly two metrics."
        )

    zero_d=zero_crossover_df.copy()
    ft_d=ft_crossover_df.copy()

    for name,d in [
        ("zero_crossover_df",zero_d),
        ("ft_crossover_df",ft_d),
    ]:
        if curve_col not in d.columns:
            raise ValueError(
                f"{name} is missing {curve_col!r}."
            )

    if zero_metric not in zero_d.columns:
        raise ValueError(
            f"{zero_metric!r} not found in zero_crossover_df."
        )

    for metric in ft_metrics:
        if metric not in ft_d.columns:
            raise ValueError(
                f"{metric!r} not found in ft_crossover_df."
            )

    if lambda_order not in {"ascending","descending"}:
        raise ValueError(
            "lambda_order must be 'ascending' or 'descending'."
        )

    # ---------------------------------------------------------
    # Prepare ZERO-CROSSOVER panel:
    # FT first, then λ=0 -> λ=1e-2
    # ---------------------------------------------------------
    zero_ft=zero_d[
        zero_d[curve_col]==ft_curve_name
    ].copy()

    zero_lambda=zero_d[
        zero_d[curve_col]!=ft_curve_name
    ].copy()

    parsed=zero_lambda[curve_col].map(
        _parse_lambda_curve
    )

    valid=parsed.notna()

    zero_lambda=zero_lambda.loc[
        valid
    ].copy()

    parsed=parsed.loc[
        valid
    ]

    zero_lambda["_lambda"]=[
        x[0] for x in parsed
    ]

    zero_lambda["_lambda_label"]=[
        x[1] for x in parsed
    ]

    zero_lambda=zero_lambda.sort_values(
        "_lambda",
        ascending=lambda_order=="ascending"
    )

    # FT always first regardless of lambda sort direction.
    zero_plot=pd.concat(
        [zero_ft,zero_lambda],
        ignore_index=True
    )

    zero_labels=(
        ["FT"] if len(zero_ft)>0 else []
    ) + zero_lambda[
        "_lambda_label"
    ].tolist()

    # ---------------------------------------------------------
    # Prepare FT-crossing panels:
    # λ only
    # ---------------------------------------------------------
    parsed=ft_d[curve_col].map(
        _parse_lambda_curve
    )

    valid=parsed.notna()

    ft_plot=ft_d.loc[
        valid
    ].copy()

    parsed=parsed.loc[
        valid
    ]

    ft_plot["_lambda"]=[
        x[0] for x in parsed
    ]

    ft_plot["_lambda_label"]=[
        x[1] for x in parsed
    ]

    ft_plot=ft_plot.sort_values(
        "_lambda",
        ascending=lambda_order=="ascending"
    ).reset_index(drop=True)

    ft_labels=ft_plot[
        "_lambda_label"
    ].tolist()

    # ---------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------
    if zero_ylabel is None:
        zero_ylabel=zero_metric

    if ft_ylabels is None:
        ft_ylabels=list(ft_metrics)

    if panel_titles is None:
        panel_titles=["","",""]

    if ylims is None:
        ylims=[None,None,None]

    if hlines is None:
        hlines=[None,None,None]

    error_kwargs={
        "elinewidth":1.0,
        "capsize":3,
        "capthick":1.0,
        "zorder":2,
        **(error_kwargs or {})
    }

    hline_kwargs={
        "color":"0.5",
        "linestyle":"--",
        "linewidth":1.0,
        "alpha":0.8,
        "zorder":0,
        **(hline_kwargs or {})
    }

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    fig,axes=plt.subplots(
        1,3,
        figsize=figsize,
        dpi=dpi
    )

    fig.subplots_adjust(
        wspace=wspace
    )

    # Panel 1: FT + lambdas.
    _draw_panel(
        axes[0],
        zero_plot,
        zero_metric,
        np.arange(len(zero_plot)),
        zero_labels,
        zero_ylabel,
        panel_titles[0],
        ylims[0],
        hlines[0]
    )

    # Panels 2-3: lambda only.
    _draw_panel(
        axes[1],
        ft_plot,
        ft_metrics[0],
        np.arange(len(ft_plot)),
        ft_labels,
        ft_ylabels[0],
        panel_titles[1],
        ylims[1],
        hlines[1]
    )

    _draw_panel(
        axes[2],
        ft_plot,
        ft_metrics[1],
        np.arange(len(ft_plot)),
        ft_labels,
        ft_ylabels[1],
        panel_titles[2],
        ylims[2],
        hlines[2]
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
        "fig":fig,
        "axes":axes,
        "zero_plot_df":zero_plot,
        "ft_plot_df":ft_plot,
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

# def analyze_pepcl_components(
#     ms_ap_df, assay_ap_df, score_cols, method_prefixes, dr_lambda=1e-5,
#     ref_col="mhcprime", steps=None, learning_rate=None, focus_seeds=None,
#     n_boot=10000, ci=0.95, random_state=42, verbose=True,
# ):
#     import re
#     import numpy as np
#     import pandas as pd
#     from scipy import stats

#     """
#     2x2 PepCL component ablation:

#                         DR OFF                  DR ON
#       Replay OFF       pepcl_dr, λ=0           pepcl_dr, λ=dr_lambda
#       Replay ON        pepcl_r_dr, λ=0         pepcl_r_dr, λ=dr_lambda

#       = FT              = DR only
#       = Replay only     = Full PepCL

#     Lambda identity is determined from the parsed model-name lambda string,
#     NOT approximate floating-point comparison.

#     Allele AP -> macro performance per replicate -> paired contrasts/effects.

#     If ref_col is provided, its deterministic baseline performance is included
#     in summary_df only. No uncertainty interval is assigned to the reference.

#     Returns
#     -------
#     replicate_df, summary_df, contrasts_df, factorial_df
#     """

#     def _lam(x):
#         s=str(x)
#         if s in {"0","0.0"}: return 0.
#         if "eN" in s:
#             a,b=s.split("eN"); return float(a)*10**(-int(b))
#         return float(s)

#     def _lam_string(x):
#         s=str(x)
#         if s in {"0","0.0"}: return "0"
#         if "eN" in s: return s
#         v=float(x)
#         if v==0: return "0"
#         m,e=f"{v:.12e}".split("e")
#         m=m.rstrip("0").rstrip("."); e=int(e)
#         return f"{m}eN{abs(e)}" if e<0 else f"{m}e{e}"

#     def _boot_ci(x):
#         x=np.asarray(x,float); x=x[np.isfinite(x)]
#         if len(x)<2: return np.nan,np.nan
#         rng=np.random.default_rng(random_state)
#         b=x[rng.integers(0,len(x),(n_boot,len(x)))].mean(1)
#         a=(1-ci)/2
#         return tuple(np.quantile(b,[a,1-a]))

#     def _t_ci(x):
#         x=np.asarray(x,float); x=x[np.isfinite(x)]
#         if len(x)<2: return np.nan,np.nan
#         mean=x.mean()
#         se=stats.sem(x)
#         tcrit=stats.t.ppf(1-(1-ci)/2,df=len(x)-1)
#         return mean-tcrit*se,mean+tcrit*se

#     if not {"pepcl_dr","pepcl_r_dr"}.issubset(method_prefixes):
#         raise ValueError("method_prefixes must contain 'pepcl_dr' and 'pepcl_r_dr'.")

#     target_lambda_string=_lam_string(dr_lambda)
#     if target_lambda_string=="0":
#         raise ValueError("dr_lambda must be nonzero; lambda 0 is reserved for DR-OFF cells.")

#     ordered=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

#     def _method(c):
#         for name,prefix in ordered:
#             if str(c).startswith(prefix): return name
#         return None

#     pattern=re.compile(
#         r"_steps_(?P<steps>\d+)_lr_(?P<lr>.+?)_dr_lambda_(?P<lam>.+?)_seed_(?P<seed>\d+)$"
#     )

#     usable=[
#         c for c in dict.fromkeys(score_cols)
#         if c!=ref_col and c in ms_ap_df.columns and c in assay_ap_df.columns
#     ]

#     meta=[]
#     for c in usable:
#         method=_method(c); m=pattern.search(str(c))
#         if method in {"pepcl_dr","pepcl_r_dr"} and m is not None:
#             meta.append({
#                 "score_col":c,"method":method,"steps":int(m["steps"]),
#                 "learning_rate":m["lr"],"lambda_string":m["lam"],
#                 "lambda":_lam(m["lam"]),"seed":int(m["seed"])
#             })

#     meta=pd.DataFrame(meta)
#     if meta.empty:
#         raise ValueError("No PepCL lambda-sweep columns could be parsed.")

#     if steps is not None:
#         meta=meta[meta["steps"]==int(steps)]

#     if learning_rate is not None:
#         meta=meta[meta["learning_rate"]==str(learning_rate)]

#     if focus_seeds is not None:
#         focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in np.atleast_1d(focus_seeds))))
#         meta=meta[meta["seed"].isin(focus_seeds)]

#     # Exact experimental cells only: lambda 0 + chosen nonzero DR lambda.
#     wanted=meta[
#         meta["lambda_string"].isin({"0",target_lambda_string})
#     ].copy()

#     if wanted.empty:
#         raise ValueError(
#             f"No component-ablation columns found for lambda 0 and "
#             f"{target_lambda_string}."
#         )

#     # One fixed LR/step operating point.
#     hp=wanted[["steps","learning_rate"]].drop_duplicates()

#     if len(hp)!=1:
#         raise ValueError(
#             f"Found {len(hp)} LR/step combinations. Supply steps= and/or learning_rate= "
#             "to select one fixed operating point."
#         )

#     # Macro performance independently for assay and MS.
#     assay_mean=assay_ap_df[wanted["score_col"]].mean(axis=0)
#     ms_mean=ms_ap_df[wanted["score_col"]].mean(axis=0)

#     rows=[]

#     for _,r in wanted.iterrows():
#         replay=int(r["method"]=="pepcl_r_dr")
#         dr=int(r["lambda_string"]==target_lambda_string)

#         if r["lambda_string"]=="0":
#             dr=0

#         config={
#             (0,0):"FT",
#             (1,0):"Replay only",
#             (0,1):"DR only",
#             (1,1):"Full PepCL",
#         }[(replay,dr)]

#         rows.append({
#             **r.to_dict(),
#             "replay":replay,
#             "dr":dr,
#             "configuration":config,
#             "assay_performance":float(assay_mean[r["score_col"]]),
#             "ms_performance":float(ms_mean[r["score_col"]]),
#         })

#     replicate_df=pd.DataFrame(rows)

#     # Exactly one row per replicate × factorial cell.
#     key=["seed","replay","dr"]
#     dup=replicate_df.duplicated(key,keep=False)

#     if dup.any():
#         raise ValueError(
#             "Duplicate component cells found after exact lambda filtering:\n"
#             f"{replicate_df.loc[dup,key+['lambda_string','score_col']]}"
#         )

#     expected={(0,0),(1,0),(0,1),(1,1)}
#     valid=[]
#     incomplete={}

#     for seed,g in replicate_df.groupby("seed"):
#         observed=set(map(tuple,g[["replay","dr"]].to_numpy()))

#         if observed==expected:
#             valid.append(seed)
#         else:
#             incomplete[seed]=observed

#     replicate_df=replicate_df[
#         replicate_df["seed"].isin(valid)
#     ].copy()

#     if replicate_df.empty:
#         raise ValueError(
#             f"No replicate contains all four required cells. Observed: {incomplete}"
#         )

#     # ------------------------------------------------------------
#     # Four-cell summaries + optional deterministic MHCPrime reference
#     #
#     # Ablation-model CI = Student-t CI across replicate-level
#     # performances.
#     #
#     # MHCPrime is deterministic, so no SD/SE/CI is assigned.
#     # ------------------------------------------------------------
#     summary=[]

#     if ref_col is not None:
#         if ref_col not in assay_ap_df.columns:
#             raise ValueError(
#                 f"ref_col '{ref_col}' not found in assay_ap_df."
#             )

#         if ref_col not in ms_ap_df.columns:
#             raise ValueError(
#                 f"ref_col '{ref_col}' not found in ms_ap_df."
#             )

#         ref_assay=pd.to_numeric(
#             assay_ap_df[ref_col],
#             errors="coerce",
#         ).mean()

#         ref_ms=pd.to_numeric(
#             ms_ap_df[ref_col],
#             errors="coerce",
#         ).mean()

#         summary.append({
#             "configuration":"MHCPrime",
#             "replay":np.nan,
#             "dr":np.nan,
#             "n_replicates":np.nan,
#             "replicate_ids":tuple(),

#             "assay_performance_mean":float(ref_assay),
#             "assay_performance_sd":np.nan,
#             "assay_performance_se":np.nan,
#             "assay_performance_ci_low":np.nan,
#             "assay_performance_ci_high":np.nan,

#             "ms_performance_mean":float(ref_ms),
#             "ms_performance_sd":np.nan,
#             "ms_performance_se":np.nan,
#             "ms_performance_ci_low":np.nan,
#             "ms_performance_ci_high":np.nan,
#         })

#     order=["FT","Replay only","DR only","Full PepCL"]

#     for config in order:
#         g=replicate_df[
#             replicate_df["configuration"]==config
#         ].sort_values("seed")

#         row={
#             "configuration":config,
#             "replay":int(g["replay"].iloc[0]),
#             "dr":int(g["dr"].iloc[0]),
#             "n_replicates":g["seed"].nunique(),
#             "replicate_ids":tuple(g["seed"]),
#         }

#         for metric in ["assay_performance","ms_performance"]:
#             x=g[metric].to_numpy(float)
#             sd=x.std(ddof=1)
#             se=sd/np.sqrt(len(x))
#             lo,hi=_t_ci(x)

#             row.update({
#                 f"{metric}_mean":x.mean(),
#                 f"{metric}_sd":sd,
#                 f"{metric}_se":se,
#                 f"{metric}_ci_low":lo,
#                 f"{metric}_ci_high":hi,
#             })

#         summary.append(row)

#     summary_df=pd.DataFrame(summary)

#     assay_w=replicate_df.pivot(
#         index="seed",
#         columns="configuration",
#         values="assay_performance",
#     )

#     ms_w=replicate_df.pivot(
#         index="seed",
#         columns="configuration",
#         values="ms_performance",
#     )

#     # ------------------------------------------------------------
#     # Paired configuration contrasts
#     # delta = model_a - model_b
#     # ------------------------------------------------------------
#     comparisons=[
#         ("Full PepCL","FT"),
#         ("Full PepCL","Replay only"),
#         ("Full PepCL","DR only"),
#         ("Replay only","DR only"),
#     ]

#     contrasts=[]

#     for model_a,model_b in comparisons:
#         da=assay_w[model_a]-assay_w[model_b]
#         dm=ms_w[model_a]-ms_w[model_b]

#         alo,ahi=_boot_ci(da)
#         mlo,mhi=_boot_ci(dm)

#         asd=da.std(ddof=1)
#         msd=dm.std(ddof=1)

#         contrasts.append({
#             "comparison":f"{model_a} vs {model_b}",
#             "model_a":model_a,
#             "model_b":model_b,
#             "n_replicates":len(da),

#             "assay_delta_mean":da.mean(),
#             "assay_delta_sd":asd,
#             "assay_delta_se":asd/np.sqrt(len(da)),
#             "assay_delta_ci_low":alo,
#             "assay_delta_ci_high":ahi,

#             "ms_delta_mean":dm.mean(),
#             "ms_delta_sd":msd,
#             "ms_delta_se":msd/np.sqrt(len(dm)),
#             "ms_delta_ci_low":mlo,
#             "ms_delta_ci_high":mhi,

#             "mean_pareto_dominance":bool(
#                 da.mean()>=0
#                 and dm.mean()>=0
#                 and (da.mean()>0 or dm.mean()>0)
#             ),

#             "ci_supported_pareto_dominance":bool(
#                 alo>=0
#                 and mlo>=0
#                 and (alo>0 or mlo>0)
#             ),
#         })

#     contrasts_df=pd.DataFrame(contrasts)

#     # ------------------------------------------------------------
#     # Paired 2×2 factorial contrasts
#     # ------------------------------------------------------------
#     def _factorial(wide,outcome):
#         ft=wide["FT"]
#         r=wide["Replay only"]
#         d=wide["DR only"]
#         full=wide["Full PepCL"]

#         effects={
#             "Replay":0.5*((r-ft)+(full-d)),
#             "DR":0.5*((d-ft)+(full-r)),
#             "Replay x DR":(full-r)-(d-ft),
#         }

#         rows=[]

#         for name,x in effects.items():
#             sd=x.std(ddof=1)
#             lo,hi=_boot_ci(x)

#             rows.append({
#                 "outcome":outcome,
#                 "effect":name,
#                 "n_replicates":len(x),
#                 "effect_mean":x.mean(),
#                 "effect_sd":sd,
#                 "effect_se":sd/np.sqrt(len(x)),
#                 "effect_ci_low":lo,
#                 "effect_ci_high":hi,
#                 "effect_by_replicate":dict(zip(x.index,x.values)),
#             })

#         return rows

#     factorial_df=pd.DataFrame(
#         _factorial(assay_w,"assay_performance")
#         +_factorial(ms_w,"ms_performance")
#     )

#     if verbose:
#         steps0=int(replicate_df["steps"].iloc[0])
#         lr0=replicate_df["learning_rate"].iloc[0]
#         seeds=tuple(sorted(replicate_df["seed"].unique()))

#         print(
#             f"\n{'='*90}\n"
#             f"PEPCL COMPONENT ABLATION — 2x2 REPLAY × DR\n"
#             f"{'='*90}"
#         )

#         print(
#             f"Steps: {steps0} | LR: {lr0} | "
#             f"DR lambda: {target_lambda_string}"
#         )

#         print(
#             f"Matched replicates: n={len(seeds)} {seeds}"
#         )

#         print("\nCells:")
#         if ref_col is not None:
#             print(f"  MHCPrime    = {ref_col}")
#         print("  FT          = pepcl_dr,   lambda 0")
#         print("  Replay only = pepcl_r_dr, lambda 0")
#         print(
#             f"  DR only     = pepcl_dr,   "
#             f"lambda {target_lambda_string}"
#         )
#         print(
#             f"  Full PepCL  = pepcl_r_dr, "
#             f"lambda {target_lambda_string}"
#         )

#         print(
#             f"\n{'CONFIGURATION PERFORMANCE':-^90}"
#         )

#         cols=[
#             "configuration",
#             "replay",
#             "dr",
#             "assay_performance_mean",
#             "assay_performance_se",
#             "ms_performance_mean",
#             "ms_performance_se",
#         ]

#         print(
#             summary_df[cols].to_string(
#                 index=False
#             )
#         )

#         print(
#             f"\n{'PAIRED CONFIGURATION DIFFERENCES':-^90}"
#         )

#         cols=[
#             "comparison",
#             "assay_delta_mean",
#             "assay_delta_ci_low",
#             "assay_delta_ci_high",
#             "ms_delta_mean",
#             "ms_delta_ci_low",
#             "ms_delta_ci_high",
#             "mean_pareto_dominance",
#             "ci_supported_pareto_dominance",
#         ]

#         print(
#             contrasts_df[cols].to_string(
#                 index=False
#             )
#         )

#         print(
#             f"\n{'FACTORIAL COMPONENT EFFECTS':-^90}"
#         )

#         cols=[
#             "outcome",
#             "effect",
#             "effect_mean",
#             "effect_se",
#             "effect_ci_low",
#             "effect_ci_high",
#         ]

#         print(
#             factorial_df[cols].to_string(
#                 index=False
#             )
#         )

#         if incomplete:
#             print(
#                 f"\nExcluded incomplete replicates: "
#                 f"{sorted(incomplete)}"
#             )

#     return replicate_df,summary_df,contrasts_df,factorial_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_pepcl_compute_logs(log_dict, plot=True):
    """
    Compare resource/compute logs from PepCL_Logging_Ext runs.

    Parameters
    ----------
    log_dict : dict
        Method name -> output from pepcl.get_logs(as_dataframes=True)

        Example:
            {
                "FT": ft_logs,
                "PepCL-R": pepcl_logs,
                "PepCL-R-DR": pepcl_dr_logs,
            }

    plot : bool
        If True, display basic compute/memory figures.

    Returns
    -------
    dict with:
        summary
        timing
        compute
        memory_snapshots
        step_telemetry
    """

    summary_rows = []
    timing_rows = []
    compute_rows = []
    memory_frames = []
    telemetry_frames = []

    for method, logs in log_dict.items():
        timing = logs["timing"]
        compute = logs["compute"]
        data = logs["data"]
        model = logs["model"]

        mem = logs["memory_snapshots"].copy()
        telem = logs["step_telemetry"].copy()

        if len(mem):
            mem.insert(0, "method", method)
            memory_frames.append(mem)

        if len(telem):
            telem.insert(0, "method", method)
            telemetry_frames.append(telem)

        # -----------------------------------------------------
        # Headline summary
        # -----------------------------------------------------
        peak_host_gb = np.nan
        peak_system_gb = np.nan
        peak_gpu_alloc_gb = np.nan
        peak_gpu_reserved_gb = np.nan

        if len(mem):
            if "process_tree_rss_gb" in mem:
                peak_host_gb = mem["process_tree_rss_gb"].max()
            elif "process_rss_gb" in mem:
                peak_host_gb = mem["process_rss_gb"].max()

            if "system_used_gb" in mem:
                peak_system_gb = mem["system_used_gb"].max()

            if "gpu_peak_allocated_gb" in mem:
                peak_gpu_alloc_gb = mem["gpu_peak_allocated_gb"].max()

            if "gpu_peak_reserved_gb" in mem:
                peak_gpu_reserved_gb = mem["gpu_peak_reserved_gb"].max()

        summary_rows.append({
            "method": method,

            # input/storage
            "input_rows": data.get("input_dataframe_rows", np.nan),
            "input_df_gb": data.get("input_dataframe_memory_bytes", np.nan) / 1024**3,
            "compact_df_gb": data.get("compact_dataframe_memory_bytes", np.nan) / 1024**3,

            # timing
            "constructor_s": timing.get("constructor_total_seconds", np.nan),
            "train_call_s": timing.get("train_call_total_seconds", np.nan),
            "training_loop_s": timing.get("training_loop_seconds", np.nan),
            "steps_per_s": timing.get("average_steps_per_second", np.nan),
            "s_per_step": timing.get("average_seconds_per_step", np.nan),

            # host memory
            "peak_process_tree_ram_gb": peak_host_gb,
            "peak_system_ram_used_gb": peak_system_gb,

            # gpu memory
            "peak_gpu_allocated_gb": peak_gpu_alloc_gb,
            "peak_gpu_reserved_gb": peak_gpu_reserved_gb,

            # model
            "student_params_m": model.get("student_parameter_count", np.nan) / 1e6,

            # compute
            "student_forward_passes": compute.get("student_forward_passes_total", 0),
            "teacher_forward_passes": compute.get("teacher_forward_passes", 0),
            "total_forward_passes": compute.get("total_forward_passes_including_teacher", 0),
            "backward_passes": compute.get("backward_passes", 0),
            "optimizer_steps": compute.get("optimizer_steps", 0),
            "student_forward_samples": compute.get("student_forward_samples_total", 0),
            "teacher_forward_samples": compute.get("teacher_forward_samples", 0),
        })

        # -----------------------------------------------------
        # Timing breakdown
        # -----------------------------------------------------
        timing_rows.append({
            "method": method,
            "data_compaction_s": timing.get("data_compaction_seconds", 0),
            "dataset_init_s": timing.get("minimal_dataset_init_seconds", 0),
            "model_to_device_s": timing.get("model_to_device_seconds", 0),
            "teacher_copy_s": timing.get("teacher_copy_seconds", 0),
            "sampling_setup_s": timing.get("sampling_setup_seconds", 0),
            "ms_loader_construct_s": timing.get("ms_loader_construct_seconds", 0),
            "ms_loader_start_s": timing.get("ms_loader_iterator_start_seconds", 0),
            "ft_loader_construct_s": timing.get("ft_loader_construct_seconds", 0),
            "ft_loader_start_s": timing.get("ft_loader_iterator_start_seconds", 0),
            "optimizer_setup_s": timing.get("optimizer_scheduler_scaler_setup_seconds", 0),
            "training_loop_s": timing.get("training_loop_seconds", 0),
        })

        # -----------------------------------------------------
        # Compute / transfer timing
        # -----------------------------------------------------
        compute_rows.append({
            "method": method,
            "ms_batch_fetch_s": compute.get("ms_batch_fetch_seconds", 0),
            "ft_batch_fetch_s": compute.get("ft_batch_fetch_seconds", 0),
            "ms_host_to_device_s": compute.get("ms_host_to_device_seconds", 0),
            "ft_host_to_device_s": compute.get("ft_host_to_device_seconds", 0),
            "checkpoint_s": compute.get("checkpoint_seconds", 0),
            "ms_batches": compute.get("ms_batches", 0),
            "ft_batches": compute.get("ft_batches", 0),
            "ms_samples": compute.get("ms_samples", 0),
            "ft_samples": compute.get("ft_samples", 0),
        })

    summary_df = pd.DataFrame(summary_rows).set_index("method")
    timing_df = pd.DataFrame(timing_rows).set_index("method")
    compute_df = pd.DataFrame(compute_rows).set_index("method")

    memory_df = (
        pd.concat(memory_frames, ignore_index=True)
        if memory_frames else pd.DataFrame()
    )

    telemetry_df = (
        pd.concat(telemetry_frames, ignore_index=True)
        if telemetry_frames else pd.DataFrame()
    )

    # =========================================================
    # Plot 1: headline resource comparison
    # =========================================================
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        summary_df["training_loop_s"].plot(
            kind="bar", ax=axes[0, 0], title="Training wall time"
        )
        axes[0, 0].set_ylabel("Seconds")

        summary_df["peak_process_tree_ram_gb"].plot(
            kind="bar", ax=axes[0, 1], title="Peak host RAM"
        )
        axes[0, 1].set_ylabel("GB")

        summary_df["peak_gpu_allocated_gb"].plot(
            kind="bar", ax=axes[1, 0], title="Peak GPU tensor memory"
        )
        axes[1, 0].set_ylabel("GB")

        summary_df["steps_per_s"].plot(
            kind="bar", ax=axes[1, 1], title="Training throughput"
        )
        axes[1, 1].set_ylabel("Steps / second")

        fig.tight_layout()
        plt.show()

        # =====================================================
        # Plot 2: timing decomposition
        # =====================================================
        plot_cols = [
            "data_compaction_s",
            "dataset_init_s",
            "teacher_copy_s",
            "sampling_setup_s",
            "ms_loader_start_s",
            "ft_loader_start_s",
            "optimizer_setup_s",
            "training_loop_s",
        ]

        timing_df[plot_cols].plot(
            kind="bar",
            stacked=True,
            figsize=(12, 6),
        )

        plt.ylabel("Seconds")
        plt.title("PepCL runtime decomposition")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

        # =====================================================
        # Plot 3: GPU memory through execution
        # =====================================================
        if len(memory_df) and "gpu_memory_allocated_gb" in memory_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            for method, g in memory_df.groupby("method"):
                ax.plot(
                    np.arange(len(g)),
                    g["gpu_memory_allocated_gb"],
                    marker="o",
                    label=method,
                )

            ax.set_xlabel("Logged execution phase")
            ax.set_ylabel("GPU allocated memory (GB)")
            ax.set_title("GPU-memory trajectory")
            ax.legend()

            plt.tight_layout()
            plt.show()

        # =====================================================
        # Plot 4: host RAM through execution
        # =====================================================
        ram_col = None
        if "process_tree_rss_gb" in memory_df.columns:
            ram_col = "process_tree_rss_gb"
        elif "process_rss_gb" in memory_df.columns:
            ram_col = "process_rss_gb"

        if len(memory_df) and ram_col is not None:
            fig, ax = plt.subplots(figsize=(12, 6))

            for method, g in memory_df.groupby("method"):
                ax.plot(
                    np.arange(len(g)),
                    g[ram_col],
                    marker="o",
                    label=method,
                )

            ax.set_xlabel("Logged execution phase")
            ax.set_ylabel("Host RAM (GB)")
            ax.set_title("Host-memory trajectory")
            ax.legend()

            plt.tight_layout()
            plt.show()

        # =====================================================
        # Plot 5: step-level throughput
        # =====================================================
        if len(telemetry_df):
            fig, ax = plt.subplots(figsize=(10, 6))

            for method, g in telemetry_df.groupby("method"):
                ax.plot(
                    g["step"],
                    g["cumulative_steps_per_second"],
                    marker="o",
                    label=method,
                )

            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Cumulative steps / second")
            ax.set_title("Training throughput across optimization")
            ax.legend()

            plt.tight_layout()
            plt.show()

    return {
        "summary": summary_df,
        "timing": timing_df,
        "compute": compute_df,
        "memory_snapshots": memory_df,
        "step_telemetry": telemetry_df,
    }

def plot_pepcl_compute_summary_table(
    resource_results,
    decimals=2,
    method_order=None,
    fig_size=(7.5,2.2),
    dpi=300,
    header_fontsize=10,
    body_fontsize=10,
    font_family="Nimbus Sans",
    col_widths=None,
    table_bbox=(0.0,0.08,1.0,0.84),
    row_height_scale=1.25,
    header_height_scale=1.45,
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
    Plot a compact compute/resource summary table from the output of
    summarize_pepcl_compute_logs().

    Columns
    -------
    Method
    Optimizer steps
    Wall-clock time (s)
    Throughput (steps/s)
    Peak GPU memory (GB)
    Peak host memory (GB)

    Returns
    -------
    fig, ax, tbl, display_df
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    if "summary" not in resource_results:
        raise KeyError("resource_results must contain a 'summary' dataframe.")

    summary=resource_results["summary"].copy()

    required=[
        "optimizer_steps",
        "training_loop_s",
        "steps_per_s",
        "peak_gpu_allocated_gb",
        "peak_process_tree_ram_gb",
    ]
    missing=[c for c in required if c not in summary.columns]
    if missing:
        raise ValueError(f"Summary dataframe is missing columns: {missing}")

    summary=summary.reset_index()

    if "method" not in summary.columns:
        raise ValueError("Could not recover method names from summary dataframe.")

    if method_order is not None:
        method_order=list(method_order)
        missing_methods=[m for m in method_order if m not in set(summary["method"])]
        if missing_methods:
            raise ValueError(f"Methods not found in summary: {missing_methods}")

        summary=summary.set_index("method").loc[method_order].reset_index()

    def _fmt(x,d=decimals):
        return "–" if pd.isna(x) else f"{float(x):.{d}f}"

    # Wrapped headers keep the table compact and prevent collisions.
    display_df=pd.DataFrame({
        "Method":summary["method"].astype(str),
        "Optimizer\nsteps":[
            "–" if pd.isna(x) else f"{int(x):,}"
            for x in summary["optimizer_steps"]
        ],
        "Wall-clock\ntime (s)":[
            _fmt(x) for x in summary["training_loop_s"]
        ],
        "Throughput\n(steps/s)":[
            _fmt(x) for x in summary["steps_per_s"]
        ],
        "Peak GPU\nmemory (GB)":[
            _fmt(x) for x in summary["peak_gpu_allocated_gb"]
        ],
        "Peak host\nmemory (GB)":[
            _fmt(x) for x in summary["peak_process_tree_ram_gb"]
        ],
    })

    columns=list(display_df.columns)

    if col_widths is None:
        col_widths=[0.13,0.13,0.17,0.17,0.20,0.20]

    if len(col_widths)!=len(columns):
        raise ValueError(
            f"col_widths must contain {len(columns)} values."
        )

    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,"Nimbus Sans","Nimbus Sans L","Arial","Helvetica"
    ]

    fp_body=FontProperties(
        family=font_family,
        size=body_fontsize,
    )

    fp_header=FontProperties(
        family=font_family,
        size=header_fontsize,
        weight="bold",
    )

    fig,ax=plt.subplots(
        figsize=fig_size,
        dpi=dpi,
    )
    ax.axis("off")

    tbl=ax.table(
        cellText=display_df.values.tolist(),
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=table_bbox,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(body_fontsize)
    tbl.scale(1.0,row_height_scale)

    nrows=len(display_df)+1
    ncols=len(columns)

    # Remove all native table borders/backgrounds.
    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)
        cell.get_text().set_verticalalignment("center")
        cell.get_text().set_horizontalalignment("center")

    # Header formatting.
    for c in range(ncols):
        tbl[(0,c)].get_text().set_fontproperties(fp_header)
        tbl[(0,c)].get_text().set_multialignment("center")

    # Increase header height to accommodate wrapped labels.
    base_header_height=tbl[(0,0)].get_height()

    for c in range(ncols):
        tbl[(0,c)].set_height(
            base_header_height*header_height_scale
        )

    fig.canvas.draw()

    # ================================================================
    # Independent horizontal rules
    # ================================================================
    left=tbl[(0,0)].get_x()

    right=(
        tbl[(0,ncols-1)].get_x()
        +tbl[(0,ncols-1)].get_width()
    )

    header_top=(
        tbl[(0,0)].get_y()
        +tbl[(0,0)].get_height()
    )

    header_bottom=tbl[(0,0)].get_y()

    bottom=tbl[(nrows-1,0)].get_y()

    def _rule(y,lw,alpha=1.0):
        if independent_lines:
            ax.add_line(
                Line2D(
                    [left,right],
                    [y,y],
                    transform=ax.transAxes,
                    color="black",
                    linewidth=lw,
                    alpha=alpha,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=5,
                )
            )
        else:
            ax.hlines(
                y,
                left,
                right,
                transform=ax.transAxes,
                color="black",
                linewidth=lw,
                alpha=alpha,
                zorder=5,
            )

    # Top rule.
    _rule(
        header_top,
        top_linewidth,
    )

    # Header separator.
    _rule(
        header_bottom,
        header_linewidth,
    )

    # Light separators between body rows.
    for r in range(1,nrows-1):
        _rule(
            tbl[(r,0)].get_y(),
            row_linewidth,
            row_line_alpha,
        )

    # Bottom rule.
    _rule(
        bottom,
        bottom_linewidth,
    )

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

    if show_plot:
        plt.show()

    return fig,ax,tbl,display_df

def compute_train_test_overlaps_fast(
    test_dfs,
    model_train_dfs,
    output="percent",
    test_label_subset="both",
    label_col="label",
    allele_col="allele",
    seq_col="seq",
    seq_allele_col="seq_allele",
):
    """
    Fast unique-overlap analysis between model training sets and benchmark
    test sets.

    Test sets may optionally be restricted by label:
        test_label_subset="both" -> use all test rows
        test_label_subset="pos"  -> use label == 1 only
        test_label_subset="neg"  -> use label == 0 only

    Training sets are always used in full and do not require a label column.

    Completely index-independent:
    dataframe indices may be duplicated, non-contiguous, shuffled, or arbitrary.

    For each overlap type:
        overlap = |unique(test subset) ∩ train|

    If output="percent":
        100 * overlap / |unique(test subset)|

    Returns
    -------
    pandas.DataFrame
        Model | overlap_type | test_1 | test_2 | ...
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    if output not in {"percent","count"}:
        raise ValueError("output must be 'percent' or 'count'.")

    if test_label_subset not in {"both","pos","neg"}:
        raise ValueError("test_label_subset must be 'both', 'pos', or 'neg'.")

    col_map={
        "allele":allele_col,
        "peptide":seq_col,
        "peptide-allele":seq_allele_col,
    }

    # --------------------------------------------------------------
    # Validate required columns.
    # --------------------------------------------------------------
    for name,df in test_dfs.items():
        required=list(col_map.values())

        if test_label_subset!="both":
            required.append(label_col)

        missing=[c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"Test dataframe '{name}' is missing columns: {missing}"
            )

    for name,df in model_train_dfs.items():
        missing=[c for c in col_map.values() if c not in df.columns]

        if missing:
            raise ValueError(
                f"Training dataframe '{name}' is missing columns: {missing}"
            )

    # --------------------------------------------------------------
    # 1. Precompute unique TEST entities.
    # Only the requested test-label subset is considered.
    # --------------------------------------------------------------
    print(
        f"Precomputing unique benchmark entities "
        f"(test_label_subset='{test_label_subset}')..."
    )

    test_unique={name:{} for name in test_dfs}

    for test_name,df in tqdm(
        test_dfs.items(),
        total=len(test_dfs),
        desc="Test datasets",
        leave=True
    ):
        # Purely positional mask; no index assumptions.
        if test_label_subset=="both":
            label_mask=None

        elif test_label_subset=="pos":
            label_mask=(df[label_col].to_numpy(copy=False)==1)

        else:
            label_mask=(df[label_col].to_numpy(copy=False)==0)

        for overlap_type,col in col_map.items():
            values=df[col].to_numpy(copy=False)

            if label_mask is not None:
                values=values[label_mask]

            # Remove missing values positionally.
            valid_mask=~pd.isna(values)
            values=values[valid_mask]

            test_unique[test_name][overlap_type]=pd.Index(
                pd.unique(values),
                copy=False
            )

    # --------------------------------------------------------------
    # 2. Build union of all benchmark entities.
    # --------------------------------------------------------------
    print("Building benchmark unions...")

    test_union={}

    for overlap_type in col_map:
        arrays=[
            test_unique[test_name][overlap_type].to_numpy(copy=False)
            for test_name in test_dfs
        ]

        if arrays:
            test_union[overlap_type]=pd.Index(
                pd.unique(np.concatenate(arrays)),
                copy=False
            )
        else:
            test_union[overlap_type]=pd.Index([])

    for overlap_type in col_map:
        print(
            f"  {overlap_type}: "
            f"{len(test_union[overlap_type]):,} unique benchmark values"
        )

    # --------------------------------------------------------------
    # 3. Scan each full training dataframe once per overlap type.
    # --------------------------------------------------------------
    rows=[]
    total=len(model_train_dfs)*len(col_map)

    print("Scanning training datasets...")

    with tqdm(
        total=total,
        desc="Training scans",
        leave=True
    ) as pbar:

        for model_name,train_df in model_train_dfs.items():
            matched_train={}

            for overlap_type,col in col_map.items():
                candidates=test_union[overlap_type]

                pbar.set_postfix_str(
                    f"{model_name} | {overlap_type}"
                )

                if len(candidates)==0:
                    matched_train[overlap_type]=pd.Index([])
                    pbar.update(1)
                    continue

                # Raw positional values only.
                values=train_df[col].to_numpy(copy=False)

                # Membership test against benchmark union.
                mask=train_df[col].isin(candidates).to_numpy()

                matched_values=values[mask]

                if len(matched_values)==0:
                    matched_train[overlap_type]=pd.Index([])
                else:
                    matched_train[overlap_type]=pd.Index(
                        pd.unique(matched_values),
                        copy=False
                    )

                pbar.update(1)

            # ----------------------------------------------------------
            # 4. Per-test overlap calculations.
            # ----------------------------------------------------------
            for overlap_type in col_map:
                row={
                    "Model":model_name,
                    "overlap_type":overlap_type,
                }

                matched=matched_train[overlap_type]

                for test_name in test_dfs:
                    test_values=test_unique[test_name][overlap_type]
                    n_test=len(test_values)

                    if n_test==0:
                        value=np.nan
                    else:
                        n_overlap=int(
                            test_values.isin(matched).sum()
                        )

                        if output=="count":
                            value=n_overlap
                        else:
                            value=100.0*n_overlap/n_test

                    row[test_name]=value

                rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            "overlap_type",
            *test_dfs.keys()
        ]
    )

def plot_model_train_test_overlap_table(
    df,
    columns,
    model_labels=None,
    model_col="Model",
    overlap_type_col="overlap_type",
    overlap_type_order=None,
    decimals=2,

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
    col_widths=None,
    cell_pad=None,
    first_col_text_pad=0.03,

    # Lines.
    top_linewidth=1.5,
    header_linewidth=0.8,
    bottom_linewidth=1.5,
    model_group_linewidth=0.5,
    model_group_line_alpha=.35,
    line_color="black",
    independent_lines=True,

    # Styling.
    bold_model_names=True,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot model train/test overlap results.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from compute_train_test_overlaps_fast().

    columns : dict
        Mapping of dataframe column -> displayed column label.
        Dictionary order determines displayed column order.

    model_labels : dict, optional
        Mapping of model tags in `model_col` to display names.

    overlap_type_order : list, optional
        Custom order for overlap rows within each model.
        Example:
            ["peptide-allele", "peptide", "allele"]

        If None, the order already present in df is preserved.

    decimals : int
        Number of decimals for non-integer numeric values.
        Integer-like values are always shown with one decimal:
        0 -> 0.0, 90 -> 90.0, 100 -> 100.0.

    Notes
    -----
    Consecutive rows with the same model are displayed with one model label,
    vertically centered across the model block.

    The Model header and model labels are left-aligned.
    All other columns are centered.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    if not isinstance(df,pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(columns,dict) or not columns:
        raise ValueError("columns must be a non-empty dict of column -> display label.")

    model_labels={} if model_labels is None else dict(model_labels)
    display_cols=list(columns.keys())
    final_labels=list(columns.values())

    missing=[c for c in display_cols if c not in df.columns]
    if missing:
        raise ValueError(f"columns contains missing dataframe columns: {missing}")
    if model_col not in display_cols:
        raise ValueError(f"model_col '{model_col}' must be included in columns.")
    if overlap_type_col not in display_cols:
        raise ValueError(f"overlap_type_col '{overlap_type_col}' must be included in columns.")

    display=df.loc[:,display_cols].copy()
    ncols=len(display_cols)

    if col_widths is not None and len(col_widths)!=ncols:
        raise ValueError(f"col_widths must have length {ncols}, got {len(col_widths)}.")

    # --------------------------------------------------------------
    # Optional overlap-type ordering within each model.
    # Preserve original model ordering.
    # --------------------------------------------------------------
    if overlap_type_order is not None:
        overlap_type_order=list(overlap_type_order)

        present=list(pd.unique(display[overlap_type_col]))
        missing_types=[x for x in overlap_type_order if x not in present]

        if missing_types:
            raise ValueError(
                f"overlap_type_order contains values not present in "
                f"'{overlap_type_col}': {missing_types}"
            )

        unspecified=[x for x in present if x not in overlap_type_order]
        full_order=overlap_type_order+unspecified
        order_map={v:i for i,v in enumerate(full_order)}

        model_order=list(pd.unique(display[model_col]))
        model_order_map={v:i for i,v in enumerate(model_order)}

        display["_model_order"]=display[model_col].map(model_order_map)
        display["_overlap_order"]=display[overlap_type_col].map(order_map)

        display=display.sort_values(
            ["_model_order","_overlap_order"],
            kind="stable"
        ).drop(columns=["_model_order","_overlap_order"]).reset_index(drop=True)

    def _format_number(v):
        if pd.isna(v):
            return ""

        x=float(v)

        if np.isclose(
            x,
            round(x),
            rtol=0,
            atol=10**(-(decimals+2))
        ):
            return f"{x:.1f}"

        return f"{x:.{decimals}f}"

    # Apply model display labels.
    display[model_col]=display[model_col].map(
        lambda x:model_labels.get(x,x)
    )

    # Format numeric result columns.
    for col in display_cols:
        if col in {model_col,overlap_type_col}:
            continue

        if pd.api.types.is_numeric_dtype(display[col]):
            display[col]=display[col].map(_format_number)

    # Identify consecutive model groups.
    model_values=display[model_col].tolist()
    model_groups=[]

    if len(model_values)>0:
        start=0

        for i in range(1,len(model_values)+1):
            if i==len(model_values) or model_values[i]!=model_values[start]:
                model_groups.append(
                    (start,i-1,model_values[start])
                )
                start=i

    # Blank model cells; redraw one centered label per model block.
    table_display=display.copy()
    table_display[model_col]=""

    if fig_size is None:
        fig_size=(
            max(10,1.8*ncols),
            max(3.0,.42*(len(display)+2))
        )

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

    fp_body=FontProperties(
        family=font_family,
        size=body_fontsize
    )

    fp_model=FontProperties(
        family=font_family,
        size=body_fontsize,
        weight="bold" if bold_model_names else "normal"
    )

    fp_header=FontProperties(
        family=font_family,
        size=header_fontsize,
        weight="bold"
    )

    fp_title=FontProperties(
        family=font_family,
        size=title_fontsize
    )

    fig,ax=plt.subplots(
        figsize=fig_size,
        dpi=dpi
    )

    ax.axis("off")

    if title is not None:
        ax.text(
            0.0,
            1.02,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontproperties=fp_title
        )

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
    model_idx=display_cols.index(model_col)

    # Remove native matplotlib borders/backgrounds.
    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)

        if c==model_idx:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("center")

        if cell_pad is not None:
            cell.PAD=cell_pad

    # Headers.
    for c in range(ncols):
        txt=tbl[(0,c)].get_text()
        txt.set_fontproperties(fp_header)

        if c==model_idx:
            txt.set_ha("left")
        else:
            txt.set_ha("center")

    fig.canvas.draw()

    # ==============================================================
    # Vertically centered model labels
    # ==============================================================
    for start,end,label in model_groups:
        r1=start+1
        r2=end+1

        cell_top=tbl[(r1,model_idx)]
        cell_bottom=tbl[(r2,model_idx)]

        x=cell_top.get_x()
        width=cell_top.get_width()

        top=cell_top.get_y()+cell_top.get_height()
        bottom=cell_bottom.get_y()

        y=(top+bottom)/2

        ax.text(
            x+first_col_text_pad*width,
            y,
            str(label),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontproperties=fp_model,
            zorder=10,
        )

    # ==============================================================
    # Independent table rules
    # ==============================================================
    left=tbl[(0,0)].get_x()
    right=tbl[(0,ncols-1)].get_x()+tbl[(0,ncols-1)].get_width()

    header_top=(
        tbl[(0,0)].get_y()
        +tbl[(0,0)].get_height()
    )

    header_bottom=tbl[(0,0)].get_y()
    bottom=tbl[(nrows-1,0)].get_y()

    def _add_rule(y,linewidth,alpha=1.0):
        if independent_lines:
            line=Line2D(
                [left,right],
                [y,y],
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
            y,
            left,
            right,
            transform=ax.transAxes,
            linewidth=linewidth,
            color=line_color,
            alpha=alpha,
            zorder=5
        )

    _add_rule(
        header_top,
        top_linewidth
    )

    _add_rule(
        header_bottom,
        header_linewidth
    )

    # Rule after each model group except final group.
    for _,end,_ in model_groups[:-1]:
        y=tbl[(end+1,0)].get_y()

        _add_rule(
            y,
            model_group_linewidth,
            alpha=model_group_line_alpha
        )

    _add_rule(
        bottom,
        bottom_linewidth
    )

    if save_path is not None:
        fig.savefig(
            f"{save_path}.pdf",
            dpi=dpi,
            bbox_inches="tight"
        )

        fig.savefig(
            f"{save_path}.png",
            dpi=dpi,
            bbox_inches="tight"
        )

    if show_plot:
        plt.show()

    return fig,ax,tbl,display


def plot_model_train_test_overlap_table_paired(
    df_main,
    df_parenthetical,
    columns,
    model_labels=None,
    model_col="Model",
    overlap_type_col="overlap_type",
    overlap_type_order=None,
    decimals=2,

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
    col_widths=None,
    cell_pad=None,
    first_col_text_pad=0.03,

    # Lines.
    top_linewidth=1.5,
    header_linewidth=0.8,
    bottom_linewidth=1.5,
    model_group_linewidth=0.5,
    model_group_line_alpha=.35,
    line_color="black",
    independent_lines=True,

    # Styling.
    bold_model_names=True,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot two matched model train/test overlap dataframes in paired form.

    Each numeric cell is shown as:

        main_value (parenthetical_value)

    Example:
        100.0 (24.3)

    Matching between dataframes is based on:
        model_col + overlap_type_col

    Both dataframes must contain the same model/overlap-type combinations
    for the rows being plotted.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    if not isinstance(df_main,pd.DataFrame) or not isinstance(df_parenthetical,pd.DataFrame):
        raise TypeError("df_main and df_parenthetical must both be pandas DataFrames.")
    if not isinstance(columns,dict) or not columns:
        raise ValueError("columns must be a non-empty dict of column -> display label.")

    model_labels={} if model_labels is None else dict(model_labels)
    display_cols=list(columns.keys())
    final_labels=list(columns.values())

    for name,df in [("df_main",df_main),("df_parenthetical",df_parenthetical)]:
        missing=[c for c in display_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing displayed columns: {missing}")
        if model_col not in df.columns:
            raise ValueError(f"{name} is missing model_col '{model_col}'.")
        if overlap_type_col not in df.columns:
            raise ValueError(f"{name} is missing overlap_type_col '{overlap_type_col}'.")

    if model_col not in display_cols:
        raise ValueError(f"model_col '{model_col}' must be included in columns.")
    if overlap_type_col not in display_cols:
        raise ValueError(f"overlap_type_col '{overlap_type_col}' must be included in columns.")

    ncols=len(display_cols)

    if col_widths is not None and len(col_widths)!=ncols:
        raise ValueError(f"col_widths must have length {ncols}, got {len(col_widths)}.")

    # --------------------------------------------------------------
    # Match the two dataframes by Model + overlap_type.
    # --------------------------------------------------------------
    key_cols=[model_col,overlap_type_col]

    if df_main.duplicated(key_cols).any():
        raise ValueError(f"df_main contains duplicate rows for keys {key_cols}.")
    if df_parenthetical.duplicated(key_cols).any():
        raise ValueError(f"df_parenthetical contains duplicate rows for keys {key_cols}.")

    main_keys=pd.MultiIndex.from_frame(df_main[key_cols])
    paren_keys=pd.MultiIndex.from_frame(df_parenthetical[key_cols])

    missing_in_paren=main_keys.difference(paren_keys)
    missing_in_main=paren_keys.difference(main_keys)

    if len(missing_in_paren)>0 or len(missing_in_main)>0:
        raise ValueError(
            "df_main and df_parenthetical do not contain identical "
            f"{model_col} + {overlap_type_col} combinations. "
            f"Missing in df_parenthetical: {list(missing_in_paren)}, "
            f"missing in df_main: {list(missing_in_main)}"
        )

    display=df_main.loc[:,display_cols].copy()

    paren_lookup=df_parenthetical.set_index(key_cols)

    # --------------------------------------------------------------
    # Optional overlap-type ordering within each model.
    # Preserve original model ordering.
    # --------------------------------------------------------------
    if overlap_type_order is not None:
        overlap_type_order=list(overlap_type_order)

        present=list(pd.unique(display[overlap_type_col]))
        missing_types=[x for x in overlap_type_order if x not in present]

        if missing_types:
            raise ValueError(
                f"overlap_type_order contains values not present in "
                f"'{overlap_type_col}': {missing_types}"
            )

        unspecified=[x for x in present if x not in overlap_type_order]
        full_order=overlap_type_order+unspecified
        order_map={v:i for i,v in enumerate(full_order)}

        model_order=list(pd.unique(display[model_col]))
        model_order_map={v:i for i,v in enumerate(model_order)}

        display["_model_order"]=display[model_col].map(model_order_map)
        display["_overlap_order"]=display[overlap_type_col].map(order_map)

        display=display.sort_values(
            ["_model_order","_overlap_order"],
            kind="stable"
        ).drop(columns=["_model_order","_overlap_order"]).reset_index(drop=True)

    def _format_number(v):
        if pd.isna(v):
            return ""

        x=float(v)

        if np.isclose(
            x,
            round(x),
            rtol=0,
            atol=10**(-(decimals+2))
        ):
            return f"{x:.1f}"

        return f"{x:.{decimals}f}"

    # --------------------------------------------------------------
    # Construct paired numeric strings.
    # --------------------------------------------------------------
    for i,row in display.iterrows():
        key=(row[model_col],row[overlap_type_col])

        for col in display_cols:
            if col in {model_col,overlap_type_col}:
                continue

            main_value=row[col]
            paren_value=paren_lookup.loc[key,col]

            main_text=_format_number(main_value)
            paren_text=_format_number(paren_value)

            if main_text=="" and paren_text=="":
                text=""
            elif paren_text=="":
                text=main_text
            elif main_text=="":
                text=f"({paren_text})"
            else:
                text=f"{main_text} ({paren_text})"

            display.at[i,col]=text

    # Apply model display labels after dataframe matching.
    display[model_col]=display[model_col].map(
        lambda x:model_labels.get(x,x)
    )

    # Identify consecutive model groups.
    model_values=display[model_col].tolist()
    model_groups=[]

    if len(model_values)>0:
        start=0

        for i in range(1,len(model_values)+1):
            if i==len(model_values) or model_values[i]!=model_values[start]:
                model_groups.append(
                    (start,i-1,model_values[start])
                )
                start=i

    table_display=display.copy()
    table_display[model_col]=""

    if fig_size is None:
        fig_size=(
            max(10,1.8*ncols),
            max(3.0,.42*(len(display)+2))
        )

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

    fp_body=FontProperties(
        family=font_family,
        size=body_fontsize
    )

    fp_model=FontProperties(
        family=font_family,
        size=body_fontsize,
        weight="bold" if bold_model_names else "normal"
    )

    fp_header=FontProperties(
        family=font_family,
        size=header_fontsize,
        weight="bold"
    )

    fp_title=FontProperties(
        family=font_family,
        size=title_fontsize
    )

    fig,ax=plt.subplots(
        figsize=fig_size,
        dpi=dpi
    )

    ax.axis("off")

    if title is not None:
        ax.text(
            0.0,
            1.02,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontproperties=fp_title
        )

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
    model_idx=display_cols.index(model_col)

    # Remove native borders/backgrounds.
    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)

        if c==model_idx:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("center")

        if cell_pad is not None:
            cell.PAD=cell_pad

    # Headers.
    for c in range(ncols):
        txt=tbl[(0,c)].get_text()
        txt.set_fontproperties(fp_header)

        if c==model_idx:
            txt.set_ha("left")
        else:
            txt.set_ha("center")

    fig.canvas.draw()

    # ==============================================================
    # Vertically centered model labels
    # ==============================================================
    for start,end,label in model_groups:
        r1=start+1
        r2=end+1

        cell_top=tbl[(r1,model_idx)]
        cell_bottom=tbl[(r2,model_idx)]

        x=cell_top.get_x()
        width=cell_top.get_width()
        top=cell_top.get_y()+cell_top.get_height()
        bottom=cell_bottom.get_y()
        y=(top+bottom)/2

        ax.text(
            x+first_col_text_pad*width,
            y,
            str(label),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontproperties=fp_model,
            zorder=10,
        )

    # ==============================================================
    # Independent table rules
    # ==============================================================
    left=tbl[(0,0)].get_x()
    right=tbl[(0,ncols-1)].get_x()+tbl[(0,ncols-1)].get_width()

    header_top=(
        tbl[(0,0)].get_y()
        +tbl[(0,0)].get_height()
    )

    header_bottom=tbl[(0,0)].get_y()
    bottom=tbl[(nrows-1,0)].get_y()

    def _add_rule(y,linewidth,alpha=1.0):
        if independent_lines:
            line=Line2D(
                [left,right],
                [y,y],
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
            y,
            left,
            right,
            transform=ax.transAxes,
            linewidth=linewidth,
            color=line_color,
            alpha=alpha,
            zorder=5
        )

    _add_rule(header_top,top_linewidth)
    _add_rule(header_bottom,header_linewidth)

    for _,end,_ in model_groups[:-1]:
        y=tbl[(end+1,0)].get_y()

        _add_rule(
            y,
            model_group_linewidth,
            alpha=model_group_line_alpha
        )

    _add_rule(bottom,bottom_linewidth)

    if save_path is not None:
        fig.savefig(
            f"{save_path}.pdf",
            dpi=dpi,
            bbox_inches="tight"
        )

        fig.savefig(
            f"{save_path}.png",
            dpi=dpi,
            bbox_inches="tight"
        )

    if show_plot:
        plt.show()

    return fig,ax,tbl,display