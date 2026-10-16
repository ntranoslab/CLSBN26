import numpy as np
import matplotlib

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

def remove_from_first_by_second(df1, df2, level="pair", allele_col="allele", seq_col="seq"):
    df1 = df1.copy()
    if level == "pair":
        df2 = df2.copy()
        for d in (df1, df2):
            d["peptide"] = d[seq_col].str.replace("$", "")
            d["seq_len"] = d["peptide"].apply(len)
        remove = set(df2[allele_col].astype(str) + "_" + df2["seq_len"].astype(str))
        k1 = df1[allele_col].astype(str) + "_" + df1["seq_len"].astype(str)
        return df1[~k1.isin(remove)].copy()
    elif level == "allele":
        remove_alleles = set(df2[allele_col])
        return df1[~df1[allele_col].isin(remove_alleles)].copy()
    else:
        raise ValueError("level must be 'pair' or 'allele'")
        
def flatten_dict_keys_and_values(d):
    return list(d.keys()) + [x for v in d.values() for x in v]

def keep_present_cols(df, score_cols):
    return [c for c in score_cols if c in df.columns]
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

import pandas as pd

def merge_two_dicts_by_keys_substring(dict1, dict2, keys, substring):
    out = {}
    for key in keys:
        d1 = dict1[key]
        d1_sub = d1[[c for c in d1.columns if substring in c]]
        merged = pd.concat([d1_sub, dict2[key]], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out[key] = merged
    return out

import pandas as pd

def merge_two_dicts_by_keys_cols(dict1, dict2, keys, keep_cols):
    out = {}
    for key in keys:
        d1 = dict1[key]
        d1_sub = d1[[c for c in keep_cols if c in d1.columns]]
        merged = pd.concat([d1_sub, dict2[key]], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out[key] = merged
    return out

def strip_step_suffix(df):
    pattern = re.compile(r"_step_\d+$")
    out = df.copy()
    out.columns = [pattern.sub("", str(c)) for c in out.columns]
    return out

def merge_dicts_prefer_second_stripped(dict1, dict2, keys):
    pattern = re.compile(r"_step_\d+$")
    out = {}
    for key in keys:
        d1 = dict1[key]
        d2 = dict2[key].copy()
        d2.columns = [pattern.sub("", str(c)) for c in d2.columns]

        overlap = [c for c in d1.columns if c in set(d2.columns)]
        print(f"{key}: {len(overlap)} overlapping column(s)")

        d1_kept = d1.drop(columns=overlap)
        merged = pd.concat([d1_kept, d2], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out[key] = merged
    return out

def merge_dicts_prefer_second_both_stripped(dict1, dict2, keys):
    pattern = re.compile(r"_step_\d+$")
    def strip(d):
        d = d.copy()
        d.columns = [pattern.sub("", str(c)) for c in d.columns]
        return d
    out = {}
    for key in keys:
        d1 = strip(dict1[key])
        d2 = strip(dict2[key])
        overlap = [c for c in d1.columns if c in set(d2.columns)]
        print(f"{key}: {len(overlap)} overlapping column(s)")
        d1_kept = d1.drop(columns=overlap)
        merged = pd.concat([d1_kept, d2], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out[key] = merged
    return out

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

def plot_map_trajectories_grid_from_ap(
    ms_ap_df, ft_ap_df, score_cols, method_prefixes, seed_groups, panel_titles=None, highlight_targets=None,
    ref_col="mhcprime", ms_name="MS", ft_name="Holec", ms_map_col="mAP", ft_map_col="mAP",
    method_style="fill", method_cmaps=None, fig_size=(16,6.2), dpi=300, step_cmap="viridis",
    marker_size=70, line_width=1.5, line_alpha=0.6, point_alpha=0.9, hollow_linewidth=1.8,
    highlight_marker_size=200, checkpoint_only_mode=False, expected_total_steps=None,
    pareto_front=False, pareto_smooth="pchip", pareto_by_method=False, pareto_linewidth=2.5,
    pareto_alpha=0.9, pareto_smooth_points=300, equal_tradeoff_line=False,
    equal_tradeoff_linewidth=1.5, equal_tradeoff_linecolor="lightgrey", zero_line_color="0.35", zero_line_width=1.0, zero_line_style="--",

    # Panel geometry / labels.
    sharex=True, sharey=True, square_panels=True,
    xlabels=None, ylabels=None, xlims=None, ylims=None,
    axis_label_fontsize=11, title_fontsize=12, axis_label_pad=8,

    # Axis styling.
    tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
    tick_fontsize=9, tick_length=4, tick_width=1.2, tick_pad=5,
    spine_width=1.4, grid=False, grid_alpha=.2,

    # LR markers / legend.
    lr_markers=None, lr_marker_scales=None,

    # Method legend.
    method_legend_labels=None, method_legend_color="0.6",

    # Colorbar.
    colorbar_width=0.035, colorbar_pad=0.035,
    colorbar_label="Optimization step",
    colorbar_tick_fontsize=9, colorbar_label_fontsize=10, colorbar_outline_width=1.0,

    # Optimal-point annotation.
    show_target_text=True, optimal_method_names=None,
    optimal_text_fontsize=9, optimal_text_xpad=.03, optimal_text_ypad=.03,
    optimal_text_linespacing=1.3,

    # Legend.
    legend_ncol=6, legend_fontsize=10, legend_loc="lower center",
    legend_bbox=(0.5,0.01), legend_frame=False,

    # Figure spacing.
    wspace=None, left_margin=.06, right_margin=.97, bottom_margin=.22, top_margin=.92,

    # Output / logging.
    save_path=None, show_plot=True, verbose=True,
):
    """
    1x3 trajectory grid.

    seed_groups explicitly controls replicate IDs in each panel:
        seed_groups={1:[...], 2:[...], 3:[...]}

    highlight_targets specifies optional optimal models:
        {
            1:{prefix:(lr,step), ...},
            2:{prefix:(lr,step), ...},
            3:{prefix:(lr,step), ...},
        }

    A method omitted from a panel's highlight_targets receives no star and
    no optimal-parameter text. An empty dict disables all optimal points
    for that panel.

    method_legend_labels:
        Optional sequence aligned with method_prefixes/method_order.
        Example: ["FT","PepCL"]

    optimal_method_names:
        Optional sequence or dict controlling names in optimal text.
        Example: ["FT","PepCL"]

    optimal_text_xpad / optimal_text_ypad:
        Fractions of the x/y plotting ranges. Text is positioned from the
        left x-axis boundary and upward from y=0.

    lr_marker_scales:
        Optional marker -> relative size mapping. Default makes circles
        slightly larger:
            {"o":1.20, "^":1.0, "s":1.0, ...}

    If save_path is provided, both PDF and PNG are saved.

    Returns
    -------
    panel_dfs, fig, axes
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    from scipy.interpolate import PchipInterpolator

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=["Nimbus Sans","Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    if method_style not in {"fill","gradient"}:
        raise ValueError("method_style must be 'fill' or 'gradient'.")
    if pareto_smooth not in {None,"linear","pchip"}:
        raise ValueError("pareto_smooth must be None, 'linear', or 'pchip'.")
    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty dict.")
    if not isinstance(seed_groups,dict):
        raise TypeError("seed_groups must be a dictionary.")

    def _parse_sci_string(s):
        s=str(s)
        if "eN" in s:
            a,b=s.split("eN",1)
            return float(a)*10**(-int(b))
        return float(s)

    def _sci_label(v):
        v=float(v)
        if v==0: return "0"
        mant,exp=f"{v:.10e}".split("e")
        mant=mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp)}"

    def _panel_arg(x,p,default=None):
        if x is None: return default
        if isinstance(x,dict): return x.get(p,default)
        if isinstance(x,(list,tuple)):
            if len(x)!=3:
                raise ValueError("Panel-specific arguments must contain exactly 3 values.")
            return x[p-1]
        return x

    def _tick_fmt(v,_):
        if abs(v)<10**(-(tick_decimals+2)): v=0
        s=f"{v:.{tick_decimals}f}"
        return s.rstrip("0").rstrip(".") if trim_tick_zeros else s

    seed_groups=dict(seed_groups)
    for p in (1,2,3):
        if p not in seed_groups:
            raise ValueError(f"seed_groups must contain key {p}.")
        vals=seed_groups[p]
        if np.isscalar(vals): vals=[vals]
        try:
            vals=list(dict.fromkeys(int(s) for s in vals))
        except (TypeError,ValueError):
            raise TypeError(f"seed_groups[{p}] must contain integer replicate IDs.")
        if not vals:
            raise ValueError(f"seed_groups[{p}] cannot be empty.")
        seed_groups[p]=vals

    extra_seed_keys=set(seed_groups)-{1,2,3}
    if extra_seed_keys:
        raise ValueError(f"Invalid seed_groups panel keys: {sorted(extra_seed_keys)}")

    if panel_titles is None:
        panel_titles={1:"Panel 1",2:"Panel 2",3:"Panel 3"}
    elif isinstance(panel_titles,(list,tuple)):
        if len(panel_titles)!=3:
            raise ValueError("panel_titles must contain exactly 3 titles.")
        panel_titles={1:panel_titles[0],2:panel_titles[1],3:panel_titles[2]}
    elif isinstance(panel_titles,dict):
        panel_titles={p:panel_titles.get(p,f"Panel {p}") for p in (1,2,3)}
    else:
        raise TypeError("panel_titles must be None, a 3-item list/tuple, or a dict.")

    highlight_targets={} if highlight_targets is None else dict(highlight_targets)
    bad_keys=set(highlight_targets)-{1,2,3}
    if bad_keys:
        raise ValueError(f"Invalid highlight panel keys: {sorted(bad_keys)}")

    def _get_map(df,map_col,name):
        if map_col in df.columns:
            s=df[map_col].copy()
            s.index=s.index.astype(str)
            if verbose: print(f"{name}: using precomputed '{map_col}' ({len(s)} models).")
            return s.astype(float)
        if verbose: print(f"{name}: averaging allele-specific AP -> mAP.")
        return df.mean(axis=0).astype(float)

    ms_map=_get_map(ms_ap_df,ms_map_col,ms_name)
    ft_map=_get_map(ft_ap_df,ft_map_col,ft_name)

    if ref_col not in ms_map.index or ref_col not in ft_map.index:
        raise ValueError(f"Reference '{ref_col}' must exist in both inputs.")

    score_cols=list(dict.fromkeys(score_cols))
    usable=[c for c in score_cols if c in ms_map.index and c in ft_map.index]
    if not usable:
        raise ValueError("No score columns are present in both inputs.")

    if verbose:
        print(f"Using {len(usable)} score columns; skipping {len(score_cols)-len(usable)} absent from one/both inputs.")

    ref_ms=float(ms_map[ref_col])
    ref_ft=float(ft_map[ref_col])
    ordered_methods=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix): return name
        return None

    if checkpoint_only_mode:
        pattern=re.compile(
            r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
            r"_step_(?P<checkpoint>\d+)$"
        )
    else:
        pattern=re.compile(
            r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
            r"(?:_step_(?P<checkpoint>\d+))?$"
        )

    rows=[]
    bad=[]
    invalid=[]

    for col in usable:
        method=_method(col)
        m=pattern.search(str(col))

        if method is None or m is None:
            bad.append(col)
            continue

        budget=int(m["budget"])
        seed=int(m["seed"])
        checkpoint=m["checkpoint"]
        step=budget if checkpoint is None else int(checkpoint)

        if step>budget:
            invalid.append((col,budget,step))
            continue

        ms_val=float(ms_map[col])
        ft_val=float(ft_map[col])
        rows.append({
            "score_col":col, "method":method, "step_budget":budget,
            "learning_rate":m["lr"], "learning_rate_float":_parse_sci_string(m["lr"]),
            "seed":seed, "step":step, "is_explicit_full":checkpoint is None,
            "is_final_model":step==budget,
            f"{ms_name}_mAP":ms_val, f"{ft_name}_mAP":ft_val,
            f"{ms_name}_mAP_delta":ms_val-ref_ms, f"{ft_name}_mAP_delta":ft_val-ref_ft,
        })

    raw_df=pd.DataFrame(rows)

    if raw_df.empty:
        raise ValueError("No valid score columns could be parsed.")
    if bad and verbose:
        print(f"WARNING: skipped {len(bad)} unparsable columns.")
    if invalid and verbose:
        print(f"WARNING: skipped {len(invalid)} checkpoints exceeding total budget.")

    if checkpoint_only_mode:
        budgets=sorted(raw_df["step_budget"].unique())
        if len(budgets)>1:
            raise ValueError(f"checkpoint_only_mode requires one total budget; found {budgets}.")
        if expected_total_steps is not None and budgets[0]!=expected_total_steps:
            raise ValueError(f"Expected steps_{expected_total_steps}, found steps_{budgets[0]}.")

    # Prefer explicit full model over duplicate `_step_budget`.
    raw_df["_prefer_full"]=raw_df["is_explicit_full"].astype(int)
    raw_df=(
        raw_df.sort_values(
            ["method","learning_rate","seed","step_budget","step","_prefer_full"],
            ascending=[True,True,True,True,True,False]
        )
        .drop_duplicates(["method","learning_rate","seed","step_budget","step"],keep="first")
        .drop(columns="_prefer_full").reset_index(drop=True)
    )

    all_seeds=sorted(raw_df["seed"].unique())
    panel_seed_groups={p:list(seed_groups[p]) for p in (1,2,3)}

    if verbose: print("\nPanel replicate groups:")
    for p in (1,2,3):
        missing=[s for s in panel_seed_groups[p] if s not in all_seeds]
        if missing:
            raise ValueError(f"Panel {p} requests unavailable seeds: {missing}")
        if verbose:
            print(f"  Panel {p} ({panel_titles[p]}): {panel_seed_groups[p]}")

    value_cols=[
        f"{ms_name}_mAP",f"{ft_name}_mAP",
        f"{ms_name}_mAP_delta",f"{ft_name}_mAP_delta",
    ]
    group_cols=["method","learning_rate","learning_rate_float","step_budget","step"]

    def _average(df,seeds,panel):
        d=df[df["seed"].isin(seeds)].copy()
        if d.empty:
            raise ValueError(f"Panel {panel}: no rows remain.")

        required=set(seeds)
        sets=d.groupby(group_cols)["seed"].agg(lambda x:tuple(sorted(set(x)))).reset_index(name="seed_ids")
        incomplete=sets[sets["seed_ids"].map(lambda x:set(x)!=required)]

        if not incomplete.empty:
            raise ValueError(
                f"Panel {panel}: {len(incomplete)} configurations do not contain all requested replicates "
                f"{sorted(required)}. Examples: {incomplete.head(10).to_dict('records')}"
            )

        out=(
            d.groupby(group_cols,as_index=False,sort=False)
            .agg(**{c:(c,"mean") for c in value_cols},
                 n_seeds=("seed","nunique"),
                 seed_ids=("seed",lambda x:tuple(sorted(set(x)))))
        )

        out["seed"]="mean"
        out["is_final_model"]=out["step"].eq(out["step_budget"])
        out["score_col"]=out.apply(
            lambda r:f"{r['method']}_lr_{r['learning_rate']}_budget_{r['step_budget']}_step_{r['step']}_seed_mean",
            axis=1
        )
        return out

    panel_dfs={p:_average(raw_df,panel_seed_groups[p],p) for p in (1,2,3)}
    combined=pd.concat(panel_dfs.values(),ignore_index=True)

    method_order=[m for m in method_prefixes if m in combined["method"].unique()]

    lr_info=(
        combined[["learning_rate","learning_rate_float"]]
        .drop_duplicates()
        .sort_values("learning_rate_float")
    )
    lr_order=lr_info["learning_rate"].tolist()
    lr_float_map=dict(zip(lr_info["learning_rate"],lr_info["learning_rate_float"]))

    default_markers=["o","^","s","p","D","v","P","X"]
    if lr_markers is None:
        lr_markers=default_markers
    if isinstance(lr_markers,dict):
        missing=[lr for lr in lr_order if lr not in lr_markers]
        if missing:
            raise ValueError(f"lr_markers is missing learning rates: {missing}")
        marker_map={lr:lr_markers[lr] for lr in lr_order}
    else:
        lr_markers=list(lr_markers)
        marker_map={lr:lr_markers[i%len(lr_markers)] for i,lr in enumerate(lr_order)}

    default_scales={"o":1.20,"^":1.0,"s":1.0,"p":1.0,"D":1.0,"v":1.0,"P":1.0,"X":1.0}
    if lr_marker_scales is not None:
        default_scales.update(lr_marker_scales)
    marker_scale_map={lr:default_scales.get(marker_map[lr],1.0) for lr in lr_order}

    step_order=sorted(combined["step"].unique())

    if isinstance(step_cmap,dict):
        missing=[s for s in step_order if s not in step_cmap]
        if missing:
            raise ValueError(f"step_cmap is missing plotted steps: {missing}")
        step_color_map={s:step_cmap[s] for s in step_order}
        common_colors=[step_color_map[s] for s in step_order]
    else:
        cm=plt.get_cmap(step_cmap)
        common_colors=[cm(i/max(len(step_order)-1,1)) for i in range(len(step_order))]
        step_color_map=dict(zip(step_order,common_colors))

    discrete_cmap=ListedColormap(common_colors)
    discrete_norm=BoundaryNorm(np.arange(len(step_order)+1)-0.5,discrete_cmap.N)

    defaults=["Blues","Reds","Greens","Purples","Oranges","Greys"]
    if method_cmaps is None:
        method_cmaps={m:defaults[i%len(defaults)] for i,m in enumerate(method_order)}
    else:
        method_cmaps=dict(method_cmaps)
        for i,m in enumerate(method_order):
            method_cmaps.setdefault(m,defaults[i%len(defaults)])

    method_step_colors={}
    for method in method_order:
        cm=plt.get_cmap(method_cmaps[method])
        method_step_colors[method]=dict(zip(
            step_order,[cm(p) for p in np.linspace(0.35,0.95,len(step_order))]
        ))

    method_fill={m:("filled" if i%2==0 else "hollow") for i,m in enumerate(method_order)}

    if method_legend_labels is not None:
        if len(method_legend_labels)!=len(method_order):
            raise ValueError(
                f"method_legend_labels must contain {len(method_order)} labels "
                f"for methods {method_order}."
            )
        method_legend_label_map=dict(zip(method_order,method_legend_labels))
    else:
        method_legend_label_map={m:f"{m} ({method_fill[m]})" for m in method_order}

    if optimal_method_names is None:
        defaults=["FT","PepCL"]
        optimal_name_map={
            m:(defaults[i] if i<len(defaults) else m)
            for i,m in enumerate(method_order)
        }
    elif isinstance(optimal_method_names,dict):
        optimal_name_map={m:optimal_method_names.get(m,m) for m in method_order}
        for method,prefix in method_prefixes.items():
            if prefix in optimal_method_names:
                optimal_name_map[method]=optimal_method_names[prefix]
    else:
        if len(optimal_method_names)!=len(method_order):
            raise ValueError(
                f"optimal_method_names must contain {len(method_order)} names "
                f"for methods {method_order}."
            )
        optimal_name_map=dict(zip(method_order,optimal_method_names))

    x_col=f"{ms_name}_mAP_delta"
    y_col=f"{ft_name}_mAP_delta"

    fig=plt.figure(figsize=fig_size,dpi=dpi)
    gs=fig.add_gridspec(
        1,4,width_ratios=[1,1,1,colorbar_width],
        wspace=colorbar_pad if wspace is None else wspace
    )

    axes=np.empty(3,dtype=object)
    axes[0]=fig.add_subplot(gs[0,0])
    axes[1]=fig.add_subplot(gs[0,1],sharex=axes[0] if sharex else None,sharey=axes[0] if sharey else None)
    axes[2]=fig.add_subplot(gs[0,2],sharex=axes[0] if sharex else None,sharey=axes[0] if sharey else None)
    cax=fig.add_subplot(gs[0,3])

    if square_panels:
        for ax in axes:
            ax.set_box_aspect(1)

    def _parse_targets(panel):
        cfg=highlight_targets.get(panel,{})
        if cfg is None:
            return []
        if not isinstance(cfg,dict):
            raise TypeError(f"highlight_targets[{panel}] must be a dict.")

        targets=[]
        for prefix,target in cfg.items():
            if not isinstance(target,(tuple,list)) or len(target)!=2:
                raise ValueError(f"highlight_targets[{panel}]['{prefix}'] must be (LR, step).")

            matches=[m for m,p in method_prefixes.items() if p==prefix]
            if not matches:
                raise ValueError(f"Panel {panel}: unknown method prefix '{prefix}'.")

            targets.append({
                "prefix":prefix,
                "method":matches[0],
                "learning_rate_float":_parse_sci_string(target[0]),
                "step":int(target[1]),
            })
        return targets

    def _pareto(d):
        d=(
            d.dropna(subset=[x_col,y_col])
            .sort_values([x_col,y_col],ascending=[False,False])
            .drop_duplicates(x_col,keep="first")
        )
        keep=[]
        best_y=-np.inf

        for idx,r in d.iterrows():
            if r[y_col]>best_y:
                keep.append(idx)
                best_y=r[y_col]

        return d.loc[keep].sort_values(x_col).reset_index(drop=True)

    for panel,ax in enumerate(axes,start=1):
        df=panel_dfs[panel]

        if equal_tradeoff_line:
            ax.axline(
                (0,0),slope=-1,color=equal_tradeoff_linecolor,
                linewidth=equal_tradeoff_linewidth,linestyle="--",zorder=0
            )

        for _,g in df.groupby(["method","learning_rate","step_budget"],sort=False):
            g=g.sort_values("step")
            method=g["method"].iloc[0]
            lr=g["learning_rate"].iloc[0]

            line_color=(
                method_step_colors[method][g["step"].iloc[-1]]
                if method_style=="gradient" else "0.55"
            )

            ax.plot(
                g[x_col],g[y_col],color=line_color,
                linewidth=line_width,alpha=line_alpha,zorder=1
            )

            for _,r in g.iterrows():
                step=r["step"]

                if method_style=="gradient":
                    color=method_step_colors[method][step]
                    face=[color]
                    edge="black"
                    lw=0.5
                    z=3
                else:
                    color=step_color_map[step]
                    hollow=method_fill[method]=="hollow"
                    face="none" if hollow else [color]
                    edge=[color] if hollow else "black"
                    lw=hollow_linewidth if hollow else 0.5
                    z=4 if hollow else 3

                ax.scatter(
                    r[x_col],r[y_col],facecolors=face,edgecolors=edge,
                    marker=marker_map[lr],
                    s=marker_size*marker_scale_map[lr],
                    alpha=point_alpha,linewidths=lw,zorder=z
                )

        targets=_parse_targets(panel)
        target_text_lines=[]

        for target in targets:
            trg=df[
                (df["method"]==target["method"])
                & np.isclose(
                    df["learning_rate_float"].astype(float),
                    target["learning_rate_float"],rtol=1e-12,atol=0
                )
                & (df["step"]==target["step"])
            ].copy()

            if trg.empty:
                if verbose:
                    print(
                        f"WARNING panel {panel}: no target for {target['prefix']} "
                        f"LR={_sci_label(target['learning_rate_float'])}, "
                        f"step={target['step']}"
                    )
                continue

            for _,r in trg.iterrows():
                method=r["method"]
                color=(
                    method_step_colors[method][r["step"]]
                    if method_style=="gradient"
                    else step_color_map[r["step"]]
                )
                hollow=method_style=="fill" and method_fill[method]=="hollow"

                ax.scatter(
                    r[x_col],r[y_col],
                    facecolors="none" if hollow else [color],
                    edgecolors=[color] if hollow else "black",
                    marker="*",s=highlight_marker_size,
                    linewidths=max(hollow_linewidth,2.0) if hollow else 1.2,
                    zorder=8
                )

            display_name=optimal_name_map.get(target["method"],target["method"])
            target_text_lines.append(
                f"{display_name}: LR={_sci_label(target['learning_rate_float'])}, "
                f"Steps={target['step']}"
            )

            if verbose:
                print(
                    f"\nPanel {panel} target — {display_name}: "
                    f"LR={_sci_label(target['learning_rate_float'])}, "
                    f"Steps={target['step']}"
                )
                print(trg[[
                    "score_col","method","learning_rate","seed_ids","step_budget","step",
                    f"{ms_name}_mAP",f"{ft_name}_mAP",
                    f"{ms_name}_mAP_delta",f"{ft_name}_mAP_delta",
                ]].to_string(index=False))

        pareto_df=pd.DataFrame()

        if pareto_front:
            groups=(
                [(m,df[df["method"]==m]) for m in method_order]
                if pareto_by_method else [("Global",df)]
            )
            fronts=[]

            for group_name,d in groups:
                front=_pareto(d)
                if front.empty:
                    continue

                front=front.copy()
                front["pareto_group"]=group_name
                fronts.append(front)

                pcolor=(
                    plt.get_cmap(method_cmaps[group_name])(0.85)
                    if pareto_by_method else "black"
                )

                ax.plot(
                    front[x_col],front[y_col],color=pcolor,
                    linewidth=pareto_linewidth,alpha=pareto_alpha,
                    marker="o",markersize=4,zorder=6
                )

                smooth=front.sort_values(x_col).drop_duplicates(x_col,keep="last")

                if pareto_smooth=="pchip" and len(smooth)>=3:
                    xs=smooth[x_col].to_numpy()
                    ys=smooth[y_col].to_numpy()
                    xx=np.linspace(xs.min(),xs.max(),pareto_smooth_points)

                    ax.plot(
                        xx,PchipInterpolator(xs,ys,extrapolate=False)(xx),
                        color=pcolor,linewidth=pareto_linewidth,
                        alpha=min(pareto_alpha,0.75),linestyle="--",zorder=5
                    )

                elif pareto_smooth=="linear" and len(smooth)>=2:
                    ax.plot(
                        smooth[x_col],smooth[y_col],color=pcolor,
                        linewidth=pareto_linewidth,alpha=min(pareto_alpha,0.75),
                        linestyle="--",zorder=5
                    )

            if fronts:
                pareto_df=pd.concat(fronts,ignore_index=True)

        ax.pareto_df=pareto_df

        # Preserve original zero reference lines.
        ax.axvline(0,color=zero_line_color,linestyle=zero_line_style,linewidth=zero_line_width)
        ax.axhline(0,color=zero_line_color,linestyle=zero_line_style,linewidth=zero_line_width)

        ax.set_title(panel_titles[panel],fontsize=title_fontsize)
        ax.set_xlabel(
            _panel_arg(xlabels,panel,f"{ms_name} ΔmAP vs {ref_col}"),
            fontsize=axis_label_fontsize,labelpad=axis_label_pad
        )

        if panel==1:
            ax.set_ylabel(
                _panel_arg(ylabels,panel,f"{ft_name} ΔmAP vs {ref_col}"),
                fontsize=axis_label_fontsize,labelpad=axis_label_pad
            )
        elif _panel_arg(ylabels,panel) is not None:
            ax.set_ylabel(
                _panel_arg(ylabels,panel),
                fontsize=axis_label_fontsize,labelpad=axis_label_pad
            )
        elif sharey:
            ax.tick_params(labelleft=False)

        if _panel_arg(xlims,panel) is not None:
            ax.set_xlim(_panel_arg(xlims,panel))
        if _panel_arg(ylims,panel) is not None:
            ax.set_ylim(_panel_arg(ylims,panel))

        if n_ticks is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

        ax.xaxis.set_major_formatter(FuncFormatter(_tick_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(_tick_fmt))
        ax.tick_params(
            axis="both",labelsize=tick_fontsize,
            length=tick_length,width=tick_width,pad=tick_pad
        )

        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        if grid:
            ax.grid(True,alpha=grid_alpha)
        else:
            ax.grid(False)

        # Frameless optimal-parameter annotation:
        # x = padded from left plotting boundary
        # y = padded upward from horizontal y=0 line.
        if show_target_text and target_text_lines:
            xmin,xmax=ax.get_xlim()
            ymin,ymax=ax.get_ylim()
            xtext=xmin+float(optimal_text_xpad)*(xmax-xmin)
            ytext=0+float(optimal_text_ypad)*(ymax-ymin)

            ax.text(
                xtext,ytext,"\n".join(target_text_lines),
                ha="left",va="bottom",fontsize=optimal_text_fontsize,
                linespacing=optimal_text_linespacing,zorder=10
            )

    if method_style=="fill":
        sm=plt.cm.ScalarMappable(cmap=discrete_cmap,norm=discrete_norm)
        sm.set_array([])

        cbar=fig.colorbar(
            sm,cax=cax,ticks=np.arange(len(step_order)),
            spacing="uniform"
        )
        cbar.set_ticks(
            np.arange(len(step_order)),
            labels=[str(s) for s in step_order]
        )
        cbar.set_label(colorbar_label,fontsize=colorbar_label_fontsize)
        cbar.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
            length=tick_length,width=tick_width,pad=tick_pad
        )
        cbar.outline.set_linewidth(colorbar_outline_width)
    else:
        cax.set_visible(False)

    # Learning-rate legend: explicitly numerical and smallest -> largest.
    lr_handles=[]
    for lr in lr_order:
        marker=marker_map[lr]
        scale=marker_scale_map[lr]

        lr_handles.append(
            Line2D(
                [0],[0],marker=marker,linestyle="none",
                markerfacecolor="0.6",markeredgecolor="black",
                markersize=np.sqrt(marker_size*scale),
                label=f"LR = {_sci_label(lr_float_map[lr])}"
            )
        )

    # Method legend: neutral gray instead of optimization-step color.
    method_handles=[]

    if method_style=="fill":
        for method in method_order:
            hollow=method_fill[method]=="hollow"

            method_handles.append(
                Line2D(
                    [0],[0],marker="o",linestyle="none",
                    markerfacecolor="none" if hollow else method_legend_color,
                    markeredgecolor=method_legend_color if hollow else "black",
                    markeredgewidth=hollow_linewidth if hollow else 1,
                    markersize=np.sqrt(marker_size),
                    label=method_legend_label_map[method]
                )
            )
    else:
        for method in method_order:
            color=plt.get_cmap(method_cmaps[method])(0.75)
            method_handles.append(
                Line2D(
                    [0],[0],marker="o",linestyle="-",color=color,
                    markerfacecolor=color,markeredgecolor="black",
                    markersize=np.sqrt(marker_size),
                    label=method_legend_label_map[method]
                )
            )

    pareto_handles=[]
    if pareto_front:
        if pareto_by_method:
            for method in method_order:
                color=plt.get_cmap(method_cmaps[method])(0.85)
                pareto_handles.append(
                    Line2D(
                        [0],[0],color=color,linewidth=pareto_linewidth,
                        label=f"{method_legend_label_map[method]} Pareto"
                    )
                )
        else:
            pareto_handles=[
                Line2D(
                    [0],[0],color="black",
                    linewidth=pareto_linewidth,label="Pareto frontier"
                )
            ]

    handles=method_handles+lr_handles+pareto_handles

    fig.legend(
        handles=handles,loc=legend_loc,bbox_to_anchor=legend_bbox,
        ncol=min(len(handles),legend_ncol),frameon=legend_frame,
        fontsize=legend_fontsize
    )

    fig.subplots_adjust(
        left=left_margin,right=right_margin,
        bottom=bottom_margin,top=top_margin
    )

    if save_path:
        fig.savefig(f"{save_path}.pdf",format="pdf",bbox_inches="tight",dpi=dpi)
        fig.savefig(f"{save_path}.png",format="png",bbox_inches="tight",dpi=dpi)

    if show_plot:
        plt.show()

    return panel_dfs,fig,axes

# def get_ft_matches_to_pepcl(
#     ms_ap_df, assay_ap_df, score_cols, method_prefixes, pepcl_target_lr, pepcl_target_step,
#     ref_col="mhcprime", ft_method="ft", pepcl_method="pepcl_r_dr",
#     pepcl_target_step_budget=None, min_seeds=1, focus_seeds=None,

#     # Matching options.
#     se_aware_matching=False, se_band=1.0,
#     frontier_matching=False,

#     # Signed comparison direction used by existing trajectory plots.
#     difference_direction="ft_minus_pepcl",

#     # Confidence intervals.
#     ci_level=0.95,

#     verbose=True,
# ):
#     """
#     Match a target PepCL configuration to:
#       1) FT closest in assay/adaptation performance.
#       2) FT closest in MS/retention performance.

#     Matching modes
#     --------------
#     Default:
#         Exact closest FT point on the matched axis.

#     se_aware_matching=True:
#         Finds the closest FT point on the matched axis, treats candidates
#         within +/- se_band * SE as effectively matched, then selects the
#         candidate with greatest performance on the other axis.

#     frontier_matching=True:
#         Restricts FT candidates to the discrete Pareto frontier obtained by
#         jointly maximizing assay performance and MS performance.

#         Assay match:
#             frontier point closest to PepCL assay performance;
#             ties favor greater MS performance.

#         MS match:
#             frontier point closest to PepCL MS performance;
#             ties favor greater assay performance.

#     `se_aware_matching` and `frontier_matching` are mutually exclusive.

#     difference_direction:
#         "ft_minus_pepcl"
#             Existing signed-difference behavior.

#         "pepcl_minus_ft"
#             Positive values indicate PepCL gain over the matched FT point.

#     Confidence intervals
#     --------------------
#     Candidate assay/MS performance intervals are Student-t confidence
#     intervals across replicate-level macro performance values.

#     For matched FT comparisons, paired delta intervals are computed by first
#     evaluating PepCL - FT within each shared replicate and then calculating
#     the Student-t confidence interval across those paired differences.

#     Returns
#     -------
#     comparison_df, candidate_df
#     """
#     import re
#     import numpy as np
#     import pandas as pd
#     from scipy.stats import t

#     if ft_method not in method_prefixes or pepcl_method not in method_prefixes:
#         raise ValueError(f"'{ft_method}' and '{pepcl_method}' must exist in method_prefixes.")
#     if se_band<0:
#         raise ValueError("se_band must be >= 0.")
#     if se_aware_matching and frontier_matching:
#         raise ValueError("se_aware_matching and frontier_matching cannot both be True.")
#     if difference_direction not in {"ft_minus_pepcl","pepcl_minus_ft"}:
#         raise ValueError("difference_direction must be 'ft_minus_pepcl' or 'pepcl_minus_ft'.")
#     if not 0<ci_level<1:
#         raise ValueError("ci_level must be in (0,1).")

#     def _lr_string(lr):
#         s=str(lr)
#         if "eN" in s: return s
#         try:
#             mantissa,exp=f"{float(lr):.12e}".split("e")
#             mantissa=mantissa.rstrip("0").rstrip(".")
#             exp=int(exp)
#             return f"{mantissa}eN{abs(exp)}" if exp<0 else f"{mantissa}e{exp}"
#         except (TypeError,ValueError):
#             return s

#     def _summary_ci(values):
#         x=np.asarray(values,dtype=float)
#         x=x[np.isfinite(x)]
#         n=len(x)

#         if n==0:
#             return dict(mean=np.nan,sd=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,n=0)

#         mean=float(np.mean(x))

#         if n<2:
#             return dict(mean=mean,sd=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,n=n)

#         sd=float(np.std(x,ddof=1))
#         se=sd/np.sqrt(n)
#         crit=float(t.ppf((1+ci_level)/2,n-1))

#         return dict(
#             mean=mean,
#             sd=sd,
#             se=se,
#             ci_low=mean-crit*se,
#             ci_high=mean+crit*se,
#             n=n,
#         )

#     def _pareto_frontier_max(df,x_col,y_col):
#         x=df[x_col].to_numpy(float)
#         y=df[y_col].to_numpy(float)
#         keep=np.ones(len(df),dtype=bool)

#         for i in range(len(df)):
#             dominated=(
#                 (x>=x[i])
#                 & (y>=y[i])
#                 & ((x>x[i]) | (y>y[i]))
#             )
#             dominated[i]=False

#             if np.any(dominated):
#                 keep[i]=False

#         return pd.Series(keep,index=df.index)

#     def _paired_pepcl_minus_model(pep_row,model_row):
#         """
#         Paired PepCL - model differences across shared replicate IDs.
#         """
#         pep_assay=pep_row["assay_performance_by_seed"]
#         model_assay=model_row["assay_performance_by_seed"]

#         pep_ms=pep_row["ms_performance_by_seed"]
#         model_ms=model_row["ms_performance_by_seed"]

#         common=sorted(
#             set(pep_assay)
#             & set(model_assay)
#             & set(pep_ms)
#             & set(model_ms)
#         )

#         assay_by_seed={}
#         ms_by_seed={}

#         for seed in common:
#             pa=float(pep_assay[seed])
#             fa=float(model_assay[seed])
#             pm=float(pep_ms[seed])
#             fm=float(model_ms[seed])

#             if np.isfinite(pa) and np.isfinite(fa):
#                 assay_by_seed[seed]=pa-fa

#             if np.isfinite(pm) and np.isfinite(fm):
#                 ms_by_seed[seed]=pm-fm

#         assay_stats=_summary_ci(list(assay_by_seed.values()))
#         ms_stats=_summary_ci(list(ms_by_seed.values()))

#         return {
#             "assay":assay_stats,
#             "ms":ms_stats,
#             "assay_by_seed":assay_by_seed,
#             "ms_by_seed":ms_by_seed,
#             "paired_seed_ids":tuple(common),
#         }

#     if focus_seeds is not None:
#         if np.isscalar(focus_seeds):
#             focus_seeds=[focus_seeds]

#         focus_seeds=tuple(
#             sorted(dict.fromkeys(int(s) for s in focus_seeds))
#         )

#         if not focus_seeds:
#             raise ValueError("focus_seeds cannot be empty.")

#     target_lr=_lr_string(pepcl_target_lr)
#     target_step=int(pepcl_target_step)

#     if ref_col not in ms_ap_df.columns or ref_col not in assay_ap_df.columns:
#         raise ValueError(f"Reference '{ref_col}' must exist in both AP dataframes.")

#     if not ms_ap_df.columns.is_unique or not assay_ap_df.columns.is_unique:
#         raise ValueError("Both AP dataframes must have unique column names.")

#     score_cols=list(dict.fromkeys(score_cols))

#     usable=[
#         c for c in score_cols
#         if c!=ref_col
#         and c in ms_ap_df.columns
#         and c in assay_ap_df.columns
#     ]

#     if not usable:
#         raise ValueError("No supplied score columns are present in both AP dataframes.")

#     # Macro across evaluable alleles for every replicate/model column.
#     ms_map=ms_ap_df[[ref_col]+usable].mean(axis=0)
#     assay_map=assay_ap_df[[ref_col]+usable].mean(axis=0)

#     ref_ms=float(ms_map[ref_col])
#     ref_assay=float(assay_map[ref_col])

#     ordered_methods=sorted(
#         method_prefixes.items(),
#         key=lambda x:len(x[1]),
#         reverse=True
#     )

#     def _method(col):
#         for name,prefix in ordered_methods:
#             if str(col).startswith(prefix):
#                 return name
#         return None

#     pattern=re.compile(
#         r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
#         r"(?:_step_(?P<checkpoint>\d+))?$"
#     )

#     rows=[]

#     for col in usable:
#         method=_method(col)
#         m=pattern.search(str(col))

#         if method not in {ft_method,pepcl_method} or m is None:
#             continue

#         budget=int(m["budget"])
#         cp=m["checkpoint"]
#         is_full=cp is None
#         checkpoint=budget if is_full else int(cp)

#         if checkpoint>budget:
#             continue

#         rows.append({
#             "score_col":col,
#             "method":method,
#             "learning_rate":m["lr"],
#             "step_budget":budget,
#             "checkpoint":checkpoint,
#             "seed":int(m["seed"]),
#             "is_full_model":is_full,
#             "assay_performance":float(assay_map[col]),
#             "ms_performance":float(ms_map[col]),
#         })

#     seed_df=pd.DataFrame(rows)

#     if seed_df.empty:
#         raise ValueError("No FT/PepCL candidates could be parsed.")

#     # Prefer explicit full endpoint over duplicate `_step_budget`.
#     seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)

#     seed_df=(
#         seed_df.sort_values(
#             ["method","learning_rate","step_budget","seed","checkpoint","_prefer_full"],
#             ascending=[True,True,True,True,True,False]
#         )
#         .drop_duplicates(
#             ["method","learning_rate","step_budget","seed","checkpoint"],
#             keep="first"
#         )
#         .drop(columns="_prefer_full")
#         .reset_index(drop=True)
#     )

#     keys=["method","learning_rate","step_budget","checkpoint"]

#     if focus_seeds is not None:
#         seed_df=seed_df[
#             seed_df["seed"].isin(focus_seeds)
#         ].copy()

#         if seed_df.empty:
#             raise ValueError(f"No candidates remain for focus_seeds={focus_seeds}.")

#         info=(
#             seed_df.groupby(keys)["seed"]
#             .agg(
#                 seed_ids=lambda x:tuple(sorted(set(x))),
#                 n_seeds="nunique",
#             )
#             .reset_index()
#         )

#         info=info[
#             info["seed_ids"].map(lambda x:x==focus_seeds)
#         ].copy()

#         if info.empty:
#             raise ValueError(
#                 f"No candidates contain all focused replicate IDs {focus_seeds}."
#             )

#         valid=set(map(tuple,info[keys].to_numpy()))

#         seed_df=seed_df[
#             seed_df[keys].apply(tuple,axis=1).isin(valid)
#         ].copy()

#     # ------------------------------------------------------------------
#     # Candidate-level summaries.
#     # ------------------------------------------------------------------
#     candidate_df=(
#         seed_df.groupby(keys,sort=False)
#         .agg(
#             assay_performance=("assay_performance","mean"),
#             assay_sd=("assay_performance","std"),
#             ms_performance=("ms_performance","mean"),
#             ms_sd=("ms_performance","std"),
#             n_seeds=("seed","nunique"),
#             seed_ids=("seed",lambda x:tuple(sorted(set(x)))),
#             score_cols_by_seed=("score_col",lambda x:tuple(x)),
#         )
#         .reset_index()
#     )

#     if focus_seeds is None:
#         candidate_df=candidate_df[
#             candidate_df["n_seeds"]>=min_seeds
#         ].copy()

#     candidate_df["assay_se"]=(
#         candidate_df["assay_sd"] /
#         np.sqrt(candidate_df["n_seeds"])
#     )

#     candidate_df["ms_se"]=(
#         candidate_df["ms_sd"] /
#         np.sqrt(candidate_df["n_seeds"])
#     )

#     # Retain exact replicate-level values.
#     raw_rows=[]

#     for key_vals,g in seed_df.groupby(keys,sort=False):
#         g=g.sort_values("seed")

#         row=dict(zip(keys,key_vals))

#         row.update({
#             "assay_performance_by_seed":
#                 dict(zip(g["seed"],g["assay_performance"])),

#             "ms_performance_by_seed":
#                 dict(zip(g["seed"],g["ms_performance"])),

#             "score_col_by_seed":
#                 dict(zip(g["seed"],g["score_col"])),
#         })

#         raw_rows.append(row)

#     candidate_df=candidate_df.merge(
#         pd.DataFrame(raw_rows),
#         on=keys,
#         how="left"
#     )

#     # Student-t CI for each candidate operating point.
#     assay_ci=[]
#     ms_ci=[]

#     for _,r in candidate_df.iterrows():
#         a=_summary_ci(
#             list(r["assay_performance_by_seed"].values())
#         )

#         m=_summary_ci(
#             list(r["ms_performance_by_seed"].values())
#         )

#         assay_ci.append(
#             (a["ci_low"],a["ci_high"])
#         )

#         ms_ci.append(
#             (m["ci_low"],m["ci_high"])
#         )

#     candidate_df["assay_performance_ci_low"]=[
#         x[0] for x in assay_ci
#     ]

#     candidate_df["assay_performance_ci_high"]=[
#         x[1] for x in assay_ci
#     ]

#     candidate_df["ms_performance_ci_low"]=[
#         x[0] for x in ms_ci
#     ]

#     candidate_df["ms_performance_ci_high"]=[
#         x[1] for x in ms_ci
#     ]

#     candidate_df["assay_delta_vs_ref"]=(
#         candidate_df["assay_performance"] -
#         ref_assay
#     )

#     candidate_df["ms_delta_vs_ref"]=(
#         candidate_df["ms_performance"] -
#         ref_ms
#     )

#     candidate_df["step"]=candidate_df["checkpoint"]

#     candidate_df["model_name"]=candidate_df.apply(
#         lambda r:
#             f"{r['method']}_steps_{int(r['step_budget'])}_"
#             f"lr_{r['learning_rate']}_step_{int(r['checkpoint'])}",
#         axis=1
#     )

#     # ------------------------------------------------------------------
#     # Identify PepCL target.
#     # ------------------------------------------------------------------
#     target=candidate_df[
#         (candidate_df["method"]==pepcl_method)
#         & (candidate_df["learning_rate"].astype(str)==target_lr)
#         & (candidate_df["checkpoint"]==target_step)
#     ].copy()

#     if pepcl_target_step_budget is not None:
#         target=target[
#             target["step_budget"]==int(pepcl_target_step_budget)
#         ]

#     if target.empty:
#         raise ValueError(
#             f"No PepCL target found for LR={target_lr}, step={target_step}"
#             +(f", budget={pepcl_target_step_budget}"
#               if pepcl_target_step_budget is not None else "")
#             +"."
#         )

#     if len(target)>1:
#         opts=target[
#             [
#                 "model_name",
#                 "learning_rate",
#                 "step_budget",
#                 "checkpoint",
#                 "n_seeds",
#             ]
#         ].to_dict("records")

#         raise ValueError(
#             "PepCL target is ambiguous. Provide pepcl_target_step_budget. "
#             f"Matches: {opts}"
#         )

#     target=target.iloc[0]

#     pep_assay=float(target["assay_performance"])
#     pep_ms=float(target["ms_performance"])

#     ft=candidate_df[
#         candidate_df["method"]==ft_method
#     ].copy()

#     if ft.empty:
#         raise ValueError(
#             f"No eligible FT candidates found for '{ft_method}'."
#         )

#     ft["adaptation_match_gap"]=(
#         ft["assay_performance"] -
#         pep_assay
#     ).abs()

#     ft["ms_match_gap"]=(
#         ft["ms_performance"] -
#         pep_ms
#     ).abs()

#     # ------------------------------------------------------------------
#     # Pareto frontier.
#     # ------------------------------------------------------------------
#     candidate_df["is_ft_pareto_frontier"]=False

#     ft_frontier_mask=_pareto_frontier_max(
#         ft,
#         "assay_performance",
#         "ms_performance"
#     )

#     ft_frontier=ft[
#         ft_frontier_mask
#     ].copy()

#     candidate_df.loc[
#         candidate_df.index.isin(ft_frontier.index),
#         "is_ft_pareto_frontier"
#     ]=True

#     # ------------------------------------------------------------------
#     # Original exact closest matches.
#     # ------------------------------------------------------------------
#     exact_ft_match_assay=ft.sort_values(
#         [
#             "adaptation_match_gap",
#             "checkpoint",
#             "step_budget",
#             "learning_rate",
#         ],
#         ascending=[True,True,True,True]
#     ).iloc[0]

#     exact_ft_match_ms=ft.sort_values(
#         [
#             "ms_match_gap",
#             "checkpoint",
#             "step_budget",
#             "learning_rate",
#         ],
#         ascending=[True,True,True,True]
#     ).iloc[0]

#     ft_match_assay=exact_ft_match_assay
#     ft_match_ms=exact_ft_match_ms

#     # ------------------------------------------------------------------
#     # SE-aware matching.
#     # ------------------------------------------------------------------
#     if se_aware_matching:
#         assay_se=float(
#             exact_ft_match_assay["assay_se"]
#         )

#         if np.isfinite(assay_se):
#             center=float(
#                 exact_ft_match_assay["assay_performance"]
#             )

#             radius=float(se_band)*assay_se

#             assay_band=ft[
#                 (ft["assay_performance"]>=center-radius)
#                 & (ft["assay_performance"]<=center+radius)
#             ].copy()

#             if not assay_band.empty:
#                 ft_match_assay=assay_band.sort_values(
#                     [
#                         "ms_performance",
#                         "adaptation_match_gap",
#                         "checkpoint",
#                         "step_budget",
#                         "learning_rate",
#                     ],
#                     ascending=[
#                         False,
#                         True,
#                         True,
#                         True,
#                         True,
#                     ]
#                 ).iloc[0]

#         elif verbose:
#             print(
#                 "WARNING: assay SE for closest FT adaptation match is "
#                 "non-finite; using the original closest assay match."
#             )

#         ms_se=float(
#             exact_ft_match_ms["ms_se"]
#         )

#         if np.isfinite(ms_se):
#             center=float(
#                 exact_ft_match_ms["ms_performance"]
#             )

#             radius=float(se_band)*ms_se

#             ms_band=ft[
#                 (ft["ms_performance"]>=center-radius)
#                 & (ft["ms_performance"]<=center+radius)
#             ].copy()

#             if not ms_band.empty:
#                 ft_match_ms=ms_band.sort_values(
#                     [
#                         "assay_performance",
#                         "ms_match_gap",
#                         "checkpoint",
#                         "step_budget",
#                         "learning_rate",
#                     ],
#                     ascending=[
#                         False,
#                         True,
#                         True,
#                         True,
#                         True,
#                     ]
#                 ).iloc[0]

#         elif verbose:
#             print(
#                 "WARNING: MS SE for closest FT retention match is "
#                 "non-finite; using the original closest MS match."
#             )

#     # ------------------------------------------------------------------
#     # Pareto-frontier matching.
#     # ------------------------------------------------------------------
#     elif frontier_matching:
#         if ft_frontier.empty:
#             raise ValueError(
#                 "No FT candidates remain on the Pareto frontier."
#             )

#         ft_frontier["adaptation_match_gap"]=(
#             ft_frontier["assay_performance"] -
#             pep_assay
#         ).abs()

#         ft_frontier["ms_match_gap"]=(
#             ft_frontier["ms_performance"] -
#             pep_ms
#         ).abs()

#         ft_match_assay=ft_frontier.sort_values(
#             [
#                 "adaptation_match_gap",
#                 "ms_performance",
#                 "checkpoint",
#                 "step_budget",
#                 "learning_rate",
#             ],
#             ascending=[
#                 True,
#                 False,
#                 True,
#                 True,
#                 True,
#             ]
#         ).iloc[0]

#         ft_match_ms=ft_frontier.sort_values(
#             [
#                 "ms_match_gap",
#                 "assay_performance",
#                 "checkpoint",
#                 "step_budget",
#                 "learning_rate",
#             ],
#             ascending=[
#                 True,
#                 False,
#                 True,
#                 True,
#                 True,
#             ]
#         ).iloc[0]

#     # ------------------------------------------------------------------
#     # Build comparison dataframe.
#     # ------------------------------------------------------------------
#     direction_sign=(
#         1.0
#         if difference_direction=="ft_minus_pepcl"
#         else -1.0
#     )

#     def _row(role,r=None,match_axis=None):
#         if role=="mhcprime":
#             model_name=method=ref_col
#             lr=budget=step=0
#             n_seeds=0
#             score_cols=(ref_col,)

#             assay=ref_assay
#             ms=ref_ms

#             assay_sd=assay_se=ms_sd=ms_se=0.0
#             assay_ci_low=assay_ci_high=np.nan
#             ms_ci_low=ms_ci_high=np.nan

#             assay_by_seed={}
#             ms_by_seed={}

#             paired={
#                 "assay":_summary_ci([]),
#                 "ms":_summary_ci([]),
#                 "assay_by_seed":{},
#                 "ms_by_seed":{},
#                 "paired_seed_ids":tuple(),
#             }

#         else:
#             model_name=r["model_name"]
#             method=r["method"]
#             lr=r["learning_rate"]
#             budget=int(r["step_budget"])
#             step=int(r["checkpoint"])
#             n_seeds=int(r["n_seeds"])
#             score_cols=r["score_cols_by_seed"]

#             assay=float(r["assay_performance"])
#             ms=float(r["ms_performance"])

#             assay_sd=float(r["assay_sd"])
#             assay_se=float(r["assay_se"])
#             ms_sd=float(r["ms_sd"])
#             ms_se=float(r["ms_se"])

#             assay_ci_low=float(r["assay_performance_ci_low"])
#             assay_ci_high=float(r["assay_performance_ci_high"])
#             ms_ci_low=float(r["ms_performance_ci_low"])
#             ms_ci_high=float(r["ms_performance_ci_high"])

#             assay_by_seed=r["assay_performance_by_seed"]
#             ms_by_seed=r["ms_performance_by_seed"]

#             paired=_paired_pepcl_minus_model(
#                 target,
#                 r
#             )

#         assay_ft_minus_pep=assay-pep_assay
#         ms_ft_minus_pep=ms-pep_ms

#         assay_pep_minus_model=pep_assay-assay
#         ms_pep_minus_model=pep_ms-ms

#         assay_display=(
#             direction_sign *
#             assay_ft_minus_pep
#         )

#         ms_display=(
#             direction_sign *
#             ms_ft_minus_pep
#         )

#         return {
#             "role":role,
#             "model_name":model_name,
#             "method":method,
#             "learning_rate":lr,
#             "step_budget":budget,
#             "step":step,
#             "n_seeds":n_seeds,
#             "score_cols_by_seed":score_cols,
#             "match_axis":match_axis,

#             # Operating-point performance.
#             "assay_performance":assay,
#             "assay_sd":assay_sd,
#             "assay_se":assay_se,
#             "assay_performance_ci_low":assay_ci_low,
#             "assay_performance_ci_high":assay_ci_high,

#             "ms_performance":ms,
#             "ms_sd":ms_sd,
#             "ms_se":ms_se,
#             "ms_performance_ci_low":ms_ci_low,
#             "ms_performance_ci_high":ms_ci_high,

#             # Replicate-level performance.
#             "assay_performance_by_seed":assay_by_seed,
#             "ms_performance_by_seed":ms_by_seed,

#             # Relative to MHCPrime.
#             "assay_delta_vs_mhcprime":assay-ref_assay,
#             "ms_delta_vs_mhcprime":ms-ref_ms,

#             # Existing signed plot-facing quantities.
#             "assay_difference_vs_pepcl":assay_display,
#             "ms_difference_vs_pepcl":ms_display,

#             # Explicit marginal signed quantities.
#             "assay_ft_minus_pepcl":assay_ft_minus_pep,
#             "ms_ft_minus_pepcl":ms_ft_minus_pep,

#             "assay_pepcl_minus_model":assay_pep_minus_model,
#             "ms_pepcl_minus_model":ms_pep_minus_model,

#             # Paired PepCL - model differences.
#             "assay_paired_delta_mean":paired["assay"]["mean"],
#             "assay_paired_delta_sd":paired["assay"]["sd"],
#             "assay_paired_delta_se":paired["assay"]["se"],
#             "assay_paired_delta_ci_low":paired["assay"]["ci_low"],
#             "assay_paired_delta_ci_high":paired["assay"]["ci_high"],

#             "ms_paired_delta_mean":paired["ms"]["mean"],
#             "ms_paired_delta_sd":paired["ms"]["sd"],
#             "ms_paired_delta_se":paired["ms"]["se"],
#             "ms_paired_delta_ci_low":paired["ms"]["ci_low"],
#             "ms_paired_delta_ci_high":paired["ms"]["ci_high"],

#             "assay_paired_delta_by_seed":paired["assay_by_seed"],
#             "ms_paired_delta_by_seed":paired["ms_by_seed"],
#             "paired_seed_ids":paired["paired_seed_ids"],

#             # Existing clipped quantities.
#             "adaptation_loss_vs_pepcl":
#                 max(0.0,-assay_ft_minus_pep),

#             "ms_retention_loss_vs_pepcl":
#                 max(0.0,-ms_ft_minus_pep),

#             "adaptation_gain_vs_pepcl":
#                 max(0.0,assay_ft_minus_pep),

#             "ms_retention_gain_vs_pepcl":
#                 max(0.0,ms_ft_minus_pep),

#             "matched_axis_absolute_gap":
#                 abs(assay_ft_minus_pep)
#                 if match_axis=="assay"
#                 else abs(ms_ft_minus_pep)
#                 if match_axis=="ms"
#                 else 0.0
#                 if role=="pepcl_target"
#                 else np.nan,
#         }

#     comparison_df=pd.DataFrame([
#         _row("mhcprime"),
#         _row("pepcl_target",target),
#         _row(
#             "ft_match_adaptation",
#             ft_match_assay,
#             "assay"
#         ),
#         _row(
#             "ft_match_ms",
#             ft_match_ms,
#             "ms"
#         ),
#     ])

#     if verbose:
#         mode=(
#             "pareto_frontier"
#             if frontier_matching
#             else "se_band"
#             if se_aware_matching
#             else "closest"
#         )

#         print(
#             f"Replicates used: "
#             f"{focus_seeds if focus_seeds is not None else 'min_seeds='+str(min_seeds)}"
#         )

#         print(f"Matching mode: {mode}")
#         print(f"CI level: {ci_level:.3f}")
#         print(f"Difference direction: {difference_direction}")

#         if se_aware_matching:
#             print(f"SE band: {se_band:g} SE")

#         print(
#             f"PepCL target: {target['model_name']} | "
#             f"assay={pep_assay:.6f}, MS={pep_ms:.6f}"
#         )

#         if frontier_matching:
#             print(
#                 f"\nFT candidates: {len(ft)} | "
#                 f"Pareto-frontier candidates: {len(ft_frontier)}"
#             )

#         for name,r in [
#             ("FT matched assay",ft_match_assay),
#             ("FT matched MS",ft_match_ms),
#         ]:
#             paired=_paired_pepcl_minus_model(
#                 target,
#                 r
#             )

#             print(
#                 f"\n{name}: {r['model_name']} | "
#                 f"assay={r['assay_performance']:.6f} "
#                 f"[{r['assay_performance_ci_low']:.6f}, "
#                 f"{r['assay_performance_ci_high']:.6f}] | "
#                 f"MS={r['ms_performance']:.6f} "
#                 f"[{r['ms_performance_ci_low']:.6f}, "
#                 f"{r['ms_performance_ci_high']:.6f}]"
#             )

#             print(
#                 f"  PepCL - FT paired ΔAssay="
#                 f"{paired['assay']['mean']:.6f} "
#                 f"[{paired['assay']['ci_low']:.6f}, "
#                 f"{paired['assay']['ci_high']:.6f}]"
#             )

#             print(
#                 f"  PepCL - FT paired ΔMS="
#                 f"{paired['ms']['mean']:.6f} "
#                 f"[{paired['ms']['ci_low']:.6f}, "
#                 f"{paired['ms']['ci_high']:.6f}]"
#             )

#         print("\nComparison dataframe:")
#         print(
#             comparison_df.to_string(index=False)
#         )

#     return comparison_df,candidate_df

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

    Baseline handling
    -----------------
    MS validation uses one fixed MHCPrime baseline for every replicate.

    Assay validation uses:
        seeds 0-4 -> ref_col
        seed 5    -> ref_col_seed_5
        seed 6    -> ref_col_seed_6
        ...

    All MHCPrime-relative quantities are calculated within replicate first
    and then summarized across replicates. Absolute FT/PepCL matching,
    Pareto-frontier selection, and paired PepCL-vs-FT comparisons are
    unchanged mathematically because the shared baseline cancels.

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

    def _is_ref_col(c):
        c=str(c)
        return c==ref_col or c.startswith(f"{ref_col}_seed_")

    def _assay_ref_col_for_seed(seed):
        seed=int(seed)
        return ref_col if seed<=4 else f"{ref_col}_seed_{seed}"

    score_cols=list(dict.fromkeys(score_cols))

    usable=[
        c for c in score_cols
        if not _is_ref_col(c)
        and c in ms_ap_df.columns
        and c in assay_ap_df.columns
    ]

    if not usable:
        raise ValueError("No supplied score columns are present in both AP dataframes.")

    # Absolute model performance. Reference-relative quantities are computed
    # replicate by replicate below so each assay CV split uses its own MHCPrime baseline.
    ms_map=ms_ap_df[usable].mean(axis=0)
    assay_map=assay_ap_df[usable].mean(axis=0)

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

    # ------------------------------------------------------------------
    # Replicate-specific MHCPrime references.
    #
    # MS validation is fixed across all replicates, so every seed uses the
    # same ref_col baseline. Assay validation changes for CV replicates:
    # seeds 0-4 use ref_col and seeds >=5 use ref_col_seed_<seed>.
    # ------------------------------------------------------------------
    seed_ids_all=tuple(sorted(seed_df["seed"].unique()))

    ref_ms_value=float(ms_ap_df[ref_col].mean())
    ref_ms_by_seed={int(seed):ref_ms_value for seed in seed_ids_all}
    ref_ms_col_by_seed={int(seed):ref_col for seed in seed_ids_all}

    ref_assay_by_seed={}
    ref_assay_col_by_seed={}

    for seed in seed_ids_all:
        seed=int(seed)
        assay_ref_col=_assay_ref_col_for_seed(seed)

        if assay_ref_col not in assay_ap_df.columns:
            raise ValueError(
                f"Assay reference '{assay_ref_col}' required for seed {seed} is missing."
            )

        ref_assay_col_by_seed[seed]=assay_ref_col
        ref_assay_by_seed[seed]=float(assay_ap_df[assay_ref_col].mean())

    seed_df["assay_ref_col"]=seed_df["seed"].map(ref_assay_col_by_seed)
    seed_df["ms_ref_col"]=seed_df["seed"].map(ref_ms_col_by_seed)
    seed_df["assay_ref_performance"]=seed_df["seed"].map(ref_assay_by_seed).astype(float)
    seed_df["ms_ref_performance"]=seed_df["seed"].map(ref_ms_by_seed).astype(float)

    seed_df["assay_delta_vs_ref"]=(
        seed_df["assay_performance"]-seed_df["assay_ref_performance"]
    )
    seed_df["ms_delta_vs_ref"]=(
        seed_df["ms_performance"]-seed_df["ms_ref_performance"]
    )

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

            assay_ref_performance=("assay_ref_performance","mean"),
            ms_ref_performance=("ms_ref_performance","mean"),
            assay_delta_vs_ref=("assay_delta_vs_ref","mean"),
            ms_delta_vs_ref=("ms_delta_vs_ref","mean"),

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

            "assay_ref_performance_by_seed":
                dict(zip(g["seed"],g["assay_ref_performance"])),

            "ms_ref_performance_by_seed":
                dict(zip(g["seed"],g["ms_ref_performance"])),

            "assay_delta_vs_ref_by_seed":
                dict(zip(g["seed"],g["assay_delta_vs_ref"])),

            "ms_delta_vs_ref_by_seed":
                dict(zip(g["seed"],g["ms_delta_vs_ref"])),

            "assay_ref_col_by_seed":
                dict(zip(g["seed"],g["assay_ref_col"])),

            "ms_ref_col_by_seed":
                dict(zip(g["seed"],g["ms_ref_col"])),

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

    # Reference-relative uncertainty is also computed from replicate-level
    # corrected deltas, not from a single pooled MHCPrime baseline.
    assay_delta_ci=[]
    ms_delta_ci=[]

    for _,r in candidate_df.iterrows():
        a=_summary_ci(list(r["assay_delta_vs_ref_by_seed"].values()))
        m=_summary_ci(list(r["ms_delta_vs_ref_by_seed"].values()))
        assay_delta_ci.append(a)
        ms_delta_ci.append(m)

    candidate_df["assay_delta_vs_ref_sd"]=[x["sd"] for x in assay_delta_ci]
    candidate_df["assay_delta_vs_ref_se"]=[x["se"] for x in assay_delta_ci]
    candidate_df["assay_delta_vs_ref_ci_low"]=[x["ci_low"] for x in assay_delta_ci]
    candidate_df["assay_delta_vs_ref_ci_high"]=[x["ci_high"] for x in assay_delta_ci]

    candidate_df["ms_delta_vs_ref_sd"]=[x["sd"] for x in ms_delta_ci]
    candidate_df["ms_delta_vs_ref_se"]=[x["se"] for x in ms_delta_ci]
    candidate_df["ms_delta_vs_ref_ci_low"]=[x["ci_low"] for x in ms_delta_ci]
    candidate_df["ms_delta_vs_ref_ci_high"]=[x["ci_high"] for x in ms_delta_ci]

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

            target_seed_ids=tuple(target["seed_ids"])
            n_seeds=len(target_seed_ids)

            assay_by_seed={
                int(seed):float(ref_assay_by_seed[int(seed)])
                for seed in target_seed_ids
            }
            ms_by_seed={
                int(seed):float(ref_ms_by_seed[int(seed)])
                for seed in target_seed_ids
            }

            assay_ref_cols_by_seed={
                int(seed):ref_assay_col_by_seed[int(seed)]
                for seed in target_seed_ids
            }
            ms_ref_cols_by_seed={
                int(seed):ref_ms_col_by_seed[int(seed)]
                for seed in target_seed_ids
            }

            score_cols=tuple(
                assay_ref_cols_by_seed[seed]
                for seed in target_seed_ids
            )

            assay_stats=_summary_ci(list(assay_by_seed.values()))
            ms_stats=_summary_ci(list(ms_by_seed.values()))

            assay=assay_stats["mean"]
            ms=ms_stats["mean"]

            assay_sd=assay_stats["sd"]
            assay_se=assay_stats["se"]
            ms_sd=ms_stats["sd"]
            ms_se=ms_stats["se"]

            assay_ci_low=assay_stats["ci_low"]
            assay_ci_high=assay_stats["ci_high"]
            ms_ci_low=ms_stats["ci_low"]
            ms_ci_high=ms_stats["ci_high"]

            baseline_model_row={
                "assay_performance_by_seed":assay_by_seed,
                "ms_performance_by_seed":ms_by_seed,
            }
            paired=_paired_pepcl_minus_model(target,baseline_model_row)

            assay_delta_vs_mhcprime=0.0
            ms_delta_vs_mhcprime=0.0
            assay_ref_by_seed=assay_by_seed
            ms_ref_by_seed=ms_by_seed
            assay_ref_col_output=assay_ref_cols_by_seed
            ms_ref_col_output=ms_ref_cols_by_seed

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

            assay_delta_vs_mhcprime=float(r["assay_delta_vs_ref"])
            ms_delta_vs_mhcprime=float(r["ms_delta_vs_ref"])

            assay_ref_by_seed=r["assay_ref_performance_by_seed"]
            ms_ref_by_seed=r["ms_ref_performance_by_seed"]
            assay_ref_col_output=r["assay_ref_col_by_seed"]
            ms_ref_col_output=r["ms_ref_col_by_seed"]

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

            # Replicate-aware MHCPrime references.
            "assay_mhcprime_performance_by_seed":assay_ref_by_seed,
            "ms_mhcprime_performance_by_seed":ms_ref_by_seed,
            "assay_mhcprime_col_by_seed":assay_ref_col_output,
            "ms_mhcprime_col_by_seed":ms_ref_col_output,

            # Relative to MHCPrime, corrected within replicate before averaging.
            "assay_delta_vs_mhcprime":assay_delta_vs_mhcprime,
            "ms_delta_vs_mhcprime":ms_delta_vs_mhcprime,

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

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

def plot_map_trajectories_grid_from_ap(
    ms_ap_df, ft_ap_df, score_cols, method_prefixes, seed_groups, panel_titles=None, highlight_targets=None,
    ref_col="mhcprime", ms_name="MS", ft_name="Holec", ms_map_col="mAP", ft_map_col="mAP",
    method_style="fill", method_cmaps=None, fig_size=(16,6.2), dpi=300, step_cmap="viridis",
    marker_size=70, line_width=1.5, line_alpha=0.6, point_alpha=0.9, hollow_linewidth=1.8,
    highlight_marker_size=200, checkpoint_only_mode=False, expected_total_steps=None,
    pareto_front=False, pareto_smooth="pchip", pareto_by_method=False, pareto_linewidth=2.5,
    pareto_alpha=0.9, pareto_smooth_points=300, equal_tradeoff_line=False,
    equal_tradeoff_linewidth=1.5, equal_tradeoff_linecolor="lightgrey", zero_line_color="0.35", zero_line_width=1.0, zero_line_style="--",

    # Panel geometry / labels.
    sharex=True, sharey=True, square_panels=True,
    xlabels=None, ylabels=None, xlims=None, ylims=None,
    axis_label_fontsize=11, title_fontsize=12, axis_label_pad=8,

    # Axis styling.
    tick_decimals=2, trim_tick_zeros=True, n_ticks=5,
    tick_fontsize=9, tick_length=4, tick_width=1.2, tick_pad=5,
    spine_width=1.4, grid=False, grid_alpha=.2,

    # LR markers / legend.
    lr_markers=None, lr_marker_scales=None,

    # Method legend.
    method_legend_labels=None, method_legend_color="0.6",

    # Colorbar.
    colorbar_width=0.035, colorbar_pad=0.035,
    colorbar_label="Optimization step",
    colorbar_tick_fontsize=9, colorbar_label_fontsize=10, colorbar_outline_width=1.0,

    # Optimal-point annotation.
    show_target_text=True, optimal_method_names=None,
    optimal_text_fontsize=9, optimal_text_xpad=.03, optimal_text_ypad=.03,
    optimal_text_linespacing=1.3,

    # Legend.
    legend_ncol=6, legend_fontsize=10, legend_loc="lower center",
    legend_bbox=(0.5,0.01), legend_frame=False,

    # Figure spacing.
    wspace=None, left_margin=.06, right_margin=.97, bottom_margin=.22, top_margin=.92,

    # Output / logging.
    save_path=None, show_plot=True, verbose=True,
):
    """
    1x3 trajectory grid.

    Seed-specific MHCPrime reference behavior:
        seeds 0-4 -> ref_col, e.g. "mhcprime"
        seed 5    -> "mhcprime_seed_5"
        seed 6    -> "mhcprime_seed_6"
        seed 7    -> "mhcprime_seed_7"
        seed 8    -> "mhcprime_seed_8"

    Each seed-level delta mAP is calculated against its corresponding
    MHCPrime reference before averaging across the requested panel seeds.

    seed_groups explicitly controls replicate IDs in each panel:
        seed_groups={1:[...], 2:[...], 3:[...]}

    highlight_targets specifies optional optimal models:
        {
            1:{prefix:(lr,step), ...},
            2:{prefix:(lr,step), ...},
            3:{prefix:(lr,step), ...},
        }

    A method omitted from a panel's highlight_targets receives no star and
    no optimal-parameter text. An empty dict disables all optimal points
    for that panel.

    method_legend_labels:
        Optional sequence aligned with method_prefixes/method_order.
        Example: ["FT","PepCL"]

    optimal_method_names:
        Optional sequence or dict controlling names in optimal text.
        Example: ["FT","PepCL"]

    optimal_text_xpad / optimal_text_ypad:
        Fractions of the x/y plotting ranges. Text is positioned from the
        left x-axis boundary and upward from y=0.

    lr_marker_scales:
        Optional marker -> relative size mapping. Default makes circles
        slightly larger:
            {"o":1.20, "^":1.0, "s":1.0, ...}

    If save_path is provided, both PDF and PNG are saved.

    Returns
    -------
    panel_dfs, fig, axes
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    from scipy.interpolate import PchipInterpolator

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=["Nimbus Sans","Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    if method_style not in {"fill","gradient"}:
        raise ValueError("method_style must be 'fill' or 'gradient'.")
    if pareto_smooth not in {None,"linear","pchip"}:
        raise ValueError("pareto_smooth must be None, 'linear', or 'pchip'.")
    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty dict.")
    if not isinstance(seed_groups,dict):
        raise TypeError("seed_groups must be a dictionary.")

    def _parse_sci_string(s):
        s=str(s)
        if "eN" in s:
            a,b=s.split("eN",1)
            return float(a)*10**(-int(b))
        return float(s)

    def _sci_label(v):
        v=float(v)
        if v==0: return "0"
        mant,exp=f"{v:.10e}".split("e")
        mant=mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp)}"

    def _panel_arg(x,p,default=None):
        if x is None: return default
        if isinstance(x,dict): return x.get(p,default)
        if isinstance(x,(list,tuple)):
            if len(x)!=3:
                raise ValueError("Panel-specific arguments must contain exactly 3 values.")
            return x[p-1]
        return x

    def _tick_fmt(v,_):
        if abs(v)<10**(-(tick_decimals+2)): v=0
        s=f"{v:.{tick_decimals}f}"
        return s.rstrip("0").rstrip(".") if trim_tick_zeros else s

    seed_groups=dict(seed_groups)
    for p in (1,2,3):
        if p not in seed_groups:
            raise ValueError(f"seed_groups must contain key {p}.")
        vals=seed_groups[p]
        if np.isscalar(vals): vals=[vals]
        try:
            vals=list(dict.fromkeys(int(s) for s in vals))
        except (TypeError,ValueError):
            raise TypeError(f"seed_groups[{p}] must contain integer replicate IDs.")
        if not vals:
            raise ValueError(f"seed_groups[{p}] cannot be empty.")
        seed_groups[p]=vals

    extra_seed_keys=set(seed_groups)-{1,2,3}
    if extra_seed_keys:
        raise ValueError(f"Invalid seed_groups panel keys: {sorted(extra_seed_keys)}")

    if panel_titles is None:
        panel_titles={1:"Panel 1",2:"Panel 2",3:"Panel 3"}
    elif isinstance(panel_titles,(list,tuple)):
        if len(panel_titles)!=3:
            raise ValueError("panel_titles must contain exactly 3 titles.")
        panel_titles={1:panel_titles[0],2:panel_titles[1],3:panel_titles[2]}
    elif isinstance(panel_titles,dict):
        panel_titles={p:panel_titles.get(p,f"Panel {p}") for p in (1,2,3)}
    else:
        raise TypeError("panel_titles must be None, a 3-item list/tuple, or a dict.")

    highlight_targets={} if highlight_targets is None else dict(highlight_targets)
    bad_keys=set(highlight_targets)-{1,2,3}
    if bad_keys:
        raise ValueError(f"Invalid highlight panel keys: {sorted(bad_keys)}")

    def _get_map(df,map_col,name):
        if map_col in df.columns:
            s=df[map_col].copy()
            s.index=s.index.astype(str)
            if verbose: print(f"{name}: using precomputed '{map_col}' ({len(s)} models).")
            return s.astype(float)
        if verbose: print(f"{name}: averaging allele-specific AP -> mAP.")
        return df.mean(axis=0).astype(float)

    ms_map=_get_map(ms_ap_df,ms_map_col,ms_name)
    ft_map=_get_map(ft_ap_df,ft_map_col,ft_name)

    if ref_col not in ms_map.index or ref_col not in ft_map.index:
        raise ValueError(f"Reference '{ref_col}' must exist in both inputs.")

    score_cols=list(dict.fromkeys(score_cols))
    usable=[c for c in score_cols if c in ms_map.index and c in ft_map.index]
    if not usable:
        raise ValueError("No score columns are present in both inputs.")

    if verbose:
        print(f"Using {len(usable)} score columns; skipping {len(score_cols)-len(usable)} absent from one/both inputs.")

    ordered_methods=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix): return name
        return None

    # Seed-specific reference.
    def _ref_col_for_seed(seed):
        seed=int(seed)
        return ref_col if seed<=4 else f"{ref_col}_seed_{seed}"

    if checkpoint_only_mode:
        pattern=re.compile(
            r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
            r"_step_(?P<checkpoint>\d+)$"
        )
    else:
        pattern=re.compile(
            r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
            r"(?:_step_(?P<checkpoint>\d+))?$"
        )

    rows=[]
    bad=[]
    invalid=[]

    for col in usable:
        method=_method(col)
        m=pattern.search(str(col))

        if method is None or m is None:
            bad.append(col)
            continue

        budget=int(m["budget"])
        seed=int(m["seed"])
        checkpoint=m["checkpoint"]
        step=budget if checkpoint is None else int(checkpoint)

        if step>budget:
            invalid.append((col,budget,step))
            continue

        # ------------------------------------------------------------
        # Seed-specific MHCPrime baseline.
        # ------------------------------------------------------------
        seed_ref_col=_ref_col_for_seed(seed)

        if seed_ref_col not in ms_map.index:
            raise ValueError(
                f"MS reference '{seed_ref_col}' required for seed {seed} "
                f"is missing."
            )

        if seed_ref_col not in ft_map.index:
            raise ValueError(
                f"{ft_name} reference '{seed_ref_col}' required for seed {seed} "
                f"is missing."
            )

        ref_ms=float(ms_map[seed_ref_col])
        ref_ft=float(ft_map[seed_ref_col])

        ms_val=float(ms_map[col])
        ft_val=float(ft_map[col])

        rows.append({
            "score_col":col,
            "method":method,
            "step_budget":budget,
            "learning_rate":m["lr"],
            "learning_rate_float":_parse_sci_string(m["lr"]),
            "seed":seed,
            "step":step,
            "is_explicit_full":checkpoint is None,
            "is_final_model":step==budget,
            "reference_col":seed_ref_col,
            f"{ms_name}_mAP":ms_val,
            f"{ft_name}_mAP":ft_val,
            f"{ms_name}_mAP_delta":ms_val-ref_ms,
            f"{ft_name}_mAP_delta":ft_val-ref_ft,
        })

    raw_df=pd.DataFrame(rows)

    if raw_df.empty:
        raise ValueError("No valid score columns could be parsed.")
    if bad and verbose:
        print(f"WARNING: skipped {len(bad)} unparsable columns.")
    if invalid and verbose:
        print(f"WARNING: skipped {len(invalid)} checkpoints exceeding total budget.")

    if checkpoint_only_mode:
        budgets=sorted(raw_df["step_budget"].unique())
        if len(budgets)>1:
            raise ValueError(f"checkpoint_only_mode requires one total budget; found {budgets}.")
        if expected_total_steps is not None and budgets[0]!=expected_total_steps:
            raise ValueError(f"Expected steps_{expected_total_steps}, found steps_{budgets[0]}.")

    # Prefer explicit full model over duplicate `_step_budget`.
    raw_df["_prefer_full"]=raw_df["is_explicit_full"].astype(int)
    raw_df=(
        raw_df.sort_values(
            ["method","learning_rate","seed","step_budget","step","_prefer_full"],
            ascending=[True,True,True,True,True,False]
        )
        .drop_duplicates(["method","learning_rate","seed","step_budget","step"],keep="first")
        .drop(columns="_prefer_full").reset_index(drop=True)
    )

    all_seeds=sorted(raw_df["seed"].unique())
    panel_seed_groups={p:list(seed_groups[p]) for p in (1,2,3)}

    if verbose: print("\nPanel replicate groups:")
    for p in (1,2,3):
        missing=[s for s in panel_seed_groups[p] if s not in all_seeds]
        if missing:
            raise ValueError(f"Panel {p} requests unavailable seeds: {missing}")
        if verbose:
            print(f"  Panel {p} ({panel_titles[p]}): {panel_seed_groups[p]}")

    value_cols=[
        f"{ms_name}_mAP",f"{ft_name}_mAP",
        f"{ms_name}_mAP_delta",f"{ft_name}_mAP_delta",
    ]
    group_cols=["method","learning_rate","learning_rate_float","step_budget","step"]

    def _average(df,seeds,panel):
        d=df[df["seed"].isin(seeds)].copy()
        if d.empty:
            raise ValueError(f"Panel {panel}: no rows remain.")

        required=set(seeds)
        sets=d.groupby(group_cols)["seed"].agg(lambda x:tuple(sorted(set(x)))).reset_index(name="seed_ids")
        incomplete=sets[sets["seed_ids"].map(lambda x:set(x)!=required)]

        if not incomplete.empty:
            raise ValueError(
                f"Panel {panel}: {len(incomplete)} configurations do not contain all requested replicates "
                f"{sorted(required)}. Examples: {incomplete.head(10).to_dict('records')}"
            )

        out=(
            d.groupby(group_cols,as_index=False,sort=False)
            .agg(
                **{c:(c,"mean") for c in value_cols},
                n_seeds=("seed","nunique"),
                seed_ids=("seed",lambda x:tuple(sorted(set(x))))
            )
        )

        out["seed"]="mean"
        out["is_final_model"]=out["step"].eq(out["step_budget"])
        out["score_col"]=out.apply(
            lambda r:f"{r['method']}_lr_{r['learning_rate']}_budget_{r['step_budget']}_step_{r['step']}_seed_mean",
            axis=1
        )

        return out

    panel_dfs={p:_average(raw_df,panel_seed_groups[p],p) for p in (1,2,3)}
    combined=pd.concat(panel_dfs.values(),ignore_index=True)

    method_order=[m for m in method_prefixes if m in combined["method"].unique()]

    lr_info=(
        combined[["learning_rate","learning_rate_float"]]
        .drop_duplicates()
        .sort_values("learning_rate_float")
    )
    lr_order=lr_info["learning_rate"].tolist()
    lr_float_map=dict(zip(lr_info["learning_rate"],lr_info["learning_rate_float"]))

    default_markers=["o","^","s","p","D","v","P","X"]
    if lr_markers is None:
        lr_markers=default_markers
    if isinstance(lr_markers,dict):
        missing=[lr for lr in lr_order if lr not in lr_markers]
        if missing:
            raise ValueError(f"lr_markers is missing learning rates: {missing}")
        marker_map={lr:lr_markers[lr] for lr in lr_order}
    else:
        lr_markers=list(lr_markers)
        marker_map={lr:lr_markers[i%len(lr_markers)] for i,lr in enumerate(lr_order)}

    default_scales={"o":1.20,"^":1.0,"s":1.0,"p":1.0,"D":1.0,"v":1.0,"P":1.0,"X":1.0}
    if lr_marker_scales is not None:
        default_scales.update(lr_marker_scales)
    marker_scale_map={lr:default_scales.get(marker_map[lr],1.0) for lr in lr_order}

    step_order=sorted(combined["step"].unique())

    if isinstance(step_cmap,dict):
        missing=[s for s in step_order if s not in step_cmap]
        if missing:
            raise ValueError(f"step_cmap is missing plotted steps: {missing}")
        step_color_map={s:step_cmap[s] for s in step_order}
        common_colors=[step_color_map[s] for s in step_order]
    else:
        cm=plt.get_cmap(step_cmap)
        common_colors=[cm(i/max(len(step_order)-1,1)) for i in range(len(step_order))]
        step_color_map=dict(zip(step_order,common_colors))

    discrete_cmap=ListedColormap(common_colors)
    discrete_norm=BoundaryNorm(np.arange(len(step_order)+1)-0.5,discrete_cmap.N)

    defaults=["Blues","Reds","Greens","Purples","Oranges","Greys"]
    if method_cmaps is None:
        method_cmaps={m:defaults[i%len(defaults)] for i,m in enumerate(method_order)}
    else:
        method_cmaps=dict(method_cmaps)
        for i,m in enumerate(method_order):
            method_cmaps.setdefault(m,defaults[i%len(defaults)])

    method_step_colors={}
    for method in method_order:
        cm=plt.get_cmap(method_cmaps[method])
        method_step_colors[method]=dict(zip(
            step_order,[cm(p) for p in np.linspace(0.35,0.95,len(step_order))]
        ))

    method_fill={m:("filled" if i%2==0 else "hollow") for i,m in enumerate(method_order)}

    if method_legend_labels is not None:
        if len(method_legend_labels)!=len(method_order):
            raise ValueError(
                f"method_legend_labels must contain {len(method_order)} labels "
                f"for methods {method_order}."
            )
        method_legend_label_map=dict(zip(method_order,method_legend_labels))
    else:
        method_legend_label_map={m:f"{m} ({method_fill[m]})" for m in method_order}

    if optimal_method_names is None:
        defaults=["FT","PepCL"]
        optimal_name_map={
            m:(defaults[i] if i<len(defaults) else m)
            for i,m in enumerate(method_order)
        }
    elif isinstance(optimal_method_names,dict):
        optimal_name_map={m:optimal_method_names.get(m,m) for m in method_order}
        for method,prefix in method_prefixes.items():
            if prefix in optimal_method_names:
                optimal_name_map[method]=optimal_method_names[prefix]
    else:
        if len(optimal_method_names)!=len(method_order):
            raise ValueError(
                f"optimal_method_names must contain {len(method_order)} names "
                f"for methods {method_order}."
            )
        optimal_name_map=dict(zip(method_order,optimal_method_names))

    x_col=f"{ms_name}_mAP_delta"
    y_col=f"{ft_name}_mAP_delta"

    fig=plt.figure(figsize=fig_size,dpi=dpi)
    gs=fig.add_gridspec(
        1,4,width_ratios=[1,1,1,colorbar_width],
        wspace=colorbar_pad if wspace is None else wspace
    )

    axes=np.empty(3,dtype=object)
    axes[0]=fig.add_subplot(gs[0,0])
    axes[1]=fig.add_subplot(gs[0,1],sharex=axes[0] if sharex else None,sharey=axes[0] if sharey else None)
    axes[2]=fig.add_subplot(gs[0,2],sharex=axes[0] if sharex else None,sharey=axes[0] if sharey else None)
    cax=fig.add_subplot(gs[0,3])

    if square_panels:
        for ax in axes:
            ax.set_box_aspect(1)

    def _parse_targets(panel):
        cfg=highlight_targets.get(panel,{})
        if cfg is None:
            return []
        if not isinstance(cfg,dict):
            raise TypeError(f"highlight_targets[{panel}] must be a dict.")

        targets=[]
        for prefix,target in cfg.items():
            if not isinstance(target,(tuple,list)) or len(target)!=2:
                raise ValueError(f"highlight_targets[{panel}]['{prefix}'] must be (LR, step).")

            matches=[m for m,p in method_prefixes.items() if p==prefix]
            if not matches:
                raise ValueError(f"Panel {panel}: unknown method prefix '{prefix}'.")

            targets.append({
                "prefix":prefix,
                "method":matches[0],
                "learning_rate_float":_parse_sci_string(target[0]),
                "step":int(target[1]),
            })
        return targets

    def _pareto(d):
        d=(
            d.dropna(subset=[x_col,y_col])
            .sort_values([x_col,y_col],ascending=[False,False])
            .drop_duplicates(x_col,keep="first")
        )
        keep=[]
        best_y=-np.inf

        for idx,r in d.iterrows():
            if r[y_col]>best_y:
                keep.append(idx)
                best_y=r[y_col]

        return d.loc[keep].sort_values(x_col).reset_index(drop=True)

    for panel,ax in enumerate(axes,start=1):
        df=panel_dfs[panel]

        if equal_tradeoff_line:
            ax.axline(
                (0,0),slope=-1,color=equal_tradeoff_linecolor,
                linewidth=equal_tradeoff_linewidth,linestyle="--",zorder=0
            )

        for _,g in df.groupby(["method","learning_rate","step_budget"],sort=False):
            g=g.sort_values("step")
            method=g["method"].iloc[0]
            lr=g["learning_rate"].iloc[0]

            line_color=(
                method_step_colors[method][g["step"].iloc[-1]]
                if method_style=="gradient" else "0.55"
            )

            ax.plot(
                g[x_col],g[y_col],color=line_color,
                linewidth=line_width,alpha=line_alpha,zorder=1
            )

            for _,r in g.iterrows():
                step=r["step"]

                if method_style=="gradient":
                    color=method_step_colors[method][step]
                    face=[color]
                    edge="black"
                    lw=0.5
                    z=3
                else:
                    color=step_color_map[step]
                    hollow=method_fill[method]=="hollow"
                    face="none" if hollow else [color]
                    edge=[color] if hollow else "black"
                    lw=hollow_linewidth if hollow else 0.5
                    z=4 if hollow else 3

                ax.scatter(
                    r[x_col],r[y_col],facecolors=face,edgecolors=edge,
                    marker=marker_map[lr],
                    s=marker_size*marker_scale_map[lr],
                    alpha=point_alpha,linewidths=lw,zorder=z
                )

        targets=_parse_targets(panel)
        target_text_lines=[]

        for target in targets:
            trg=df[
                (df["method"]==target["method"])
                & np.isclose(
                    df["learning_rate_float"].astype(float),
                    target["learning_rate_float"],rtol=1e-12,atol=0
                )
                & (df["step"]==target["step"])
            ].copy()

            if trg.empty:
                if verbose:
                    print(
                        f"WARNING panel {panel}: no target for {target['prefix']} "
                        f"LR={_sci_label(target['learning_rate_float'])}, "
                        f"step={target['step']}"
                    )
                continue

            for _,r in trg.iterrows():
                method=r["method"]
                color=(
                    method_step_colors[method][r["step"]]
                    if method_style=="gradient"
                    else step_color_map[r["step"]]
                )
                hollow=method_style=="fill" and method_fill[method]=="hollow"

                ax.scatter(
                    r[x_col],r[y_col],
                    facecolors="none" if hollow else [color],
                    edgecolors=[color] if hollow else "black",
                    marker="*",s=highlight_marker_size,
                    linewidths=max(hollow_linewidth,2.0) if hollow else 1.2,
                    zorder=8
                )

            display_name=optimal_name_map.get(target["method"],target["method"])
            target_text_lines.append(
                f"{display_name}: LR={_sci_label(target['learning_rate_float'])}, "
                f"Steps={target['step']}"
            )

            if verbose:
                print(
                    f"\nPanel {panel} target - {display_name}: "
                    f"LR={_sci_label(target['learning_rate_float'])}, "
                    f"Steps={target['step']}"
                )
                print(trg[[
                    "score_col","method","learning_rate","seed_ids","step_budget","step",
                    f"{ms_name}_mAP",f"{ft_name}_mAP",
                    f"{ms_name}_mAP_delta",f"{ft_name}_mAP_delta",
                ]].to_string(index=False))

        pareto_df=pd.DataFrame()

        if pareto_front:
            groups=(
                [(m,df[df["method"]==m]) for m in method_order]
                if pareto_by_method else [("Global",df)]
            )
            fronts=[]

            for group_name,d in groups:
                front=_pareto(d)
                if front.empty:
                    continue

                front=front.copy()
                front["pareto_group"]=group_name
                fronts.append(front)

                pcolor=(
                    plt.get_cmap(method_cmaps[group_name])(0.85)
                    if pareto_by_method else "black"
                )

                ax.plot(
                    front[x_col],front[y_col],color=pcolor,
                    linewidth=pareto_linewidth,alpha=pareto_alpha,
                    marker="o",markersize=4,zorder=6
                )

                smooth=front.sort_values(x_col).drop_duplicates(x_col,keep="last")

                if pareto_smooth=="pchip" and len(smooth)>=3:
                    xs=smooth[x_col].to_numpy()
                    ys=smooth[y_col].to_numpy()
                    xx=np.linspace(xs.min(),xs.max(),pareto_smooth_points)

                    ax.plot(
                        xx,PchipInterpolator(xs,ys,extrapolate=False)(xx),
                        color=pcolor,linewidth=pareto_linewidth,
                        alpha=min(pareto_alpha,0.75),linestyle="--",zorder=5
                    )

                elif pareto_smooth=="linear" and len(smooth)>=2:
                    ax.plot(
                        smooth[x_col],smooth[y_col],color=pcolor,
                        linewidth=pareto_linewidth,alpha=min(pareto_alpha,0.75),
                        linestyle="--",zorder=5
                    )

            if fronts:
                pareto_df=pd.concat(fronts,ignore_index=True)

        ax.pareto_df=pareto_df

        ax.axvline(0,color=zero_line_color,linestyle=zero_line_style,linewidth=zero_line_width)
        ax.axhline(0,color=zero_line_color,linestyle=zero_line_style,linewidth=zero_line_width)

        ax.set_title(panel_titles[panel],fontsize=title_fontsize)
        ax.set_xlabel(
            _panel_arg(xlabels,panel,f"{ms_name} ΔmAP vs {ref_col}"),
            fontsize=axis_label_fontsize,labelpad=axis_label_pad
        )

        if panel==1:
            ax.set_ylabel(
                _panel_arg(ylabels,panel,f"{ft_name} ΔmAP vs {ref_col}"),
                fontsize=axis_label_fontsize,labelpad=axis_label_pad
            )
        elif _panel_arg(ylabels,panel) is not None:
            ax.set_ylabel(
                _panel_arg(ylabels,panel),
                fontsize=axis_label_fontsize,labelpad=axis_label_pad
            )
        elif sharey:
            ax.tick_params(labelleft=False)

        if _panel_arg(xlims,panel) is not None:
            ax.set_xlim(_panel_arg(xlims,panel))
        if _panel_arg(ylims,panel) is not None:
            ax.set_ylim(_panel_arg(ylims,panel))

        if n_ticks is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

        ax.xaxis.set_major_formatter(FuncFormatter(_tick_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(_tick_fmt))
        ax.tick_params(
            axis="both",labelsize=tick_fontsize,
            length=tick_length,width=tick_width,pad=tick_pad
        )

        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        if grid:
            ax.grid(True,alpha=grid_alpha)
        else:
            ax.grid(False)

        if show_target_text and target_text_lines:
            xmin,xmax=ax.get_xlim()
            ymin,ymax=ax.get_ylim()
            xtext=xmin+float(optimal_text_xpad)*(xmax-xmin)
            ytext=0+float(optimal_text_ypad)*(ymax-ymin)

            ax.text(
                xtext,ytext,"\n".join(target_text_lines),
                ha="left",va="bottom",fontsize=optimal_text_fontsize,
                linespacing=optimal_text_linespacing,zorder=10
            )

    if method_style=="fill":
        sm=plt.cm.ScalarMappable(cmap=discrete_cmap,norm=discrete_norm)
        sm.set_array([])

        cbar=fig.colorbar(
            sm,cax=cax,ticks=np.arange(len(step_order)),
            spacing="uniform"
        )
        cbar.set_ticks(
            np.arange(len(step_order)),
            labels=[str(s) for s in step_order]
        )
        cbar.set_label(colorbar_label,fontsize=colorbar_label_fontsize)
        cbar.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
            length=tick_length,width=tick_width,pad=tick_pad
        )
        cbar.outline.set_linewidth(colorbar_outline_width)
    else:
        cax.set_visible(False)

    lr_handles=[]
    for lr in lr_order:
        marker=marker_map[lr]
        scale=marker_scale_map[lr]

        lr_handles.append(
            Line2D(
                [0],[0],marker=marker,linestyle="none",
                markerfacecolor="0.6",markeredgecolor="black",
                markersize=np.sqrt(marker_size*scale),
                label=f"LR = {_sci_label(lr_float_map[lr])}"
            )
        )

    method_handles=[]

    if method_style=="fill":
        for method in method_order:
            hollow=method_fill[method]=="hollow"

            method_handles.append(
                Line2D(
                    [0],[0],marker="o",linestyle="none",
                    markerfacecolor="none" if hollow else method_legend_color,
                    markeredgecolor=method_legend_color if hollow else "black",
                    markeredgewidth=hollow_linewidth if hollow else 1,
                    markersize=np.sqrt(marker_size),
                    label=method_legend_label_map[method]
                )
            )
    else:
        for method in method_order:
            color=plt.get_cmap(method_cmaps[method])(0.75)
            method_handles.append(
                Line2D(
                    [0],[0],marker="o",linestyle="-",color=color,
                    markerfacecolor=color,markeredgecolor="black",
                    markersize=np.sqrt(marker_size),
                    label=method_legend_label_map[method]
                )
            )

    pareto_handles=[]
    if pareto_front:
        if pareto_by_method:
            for method in method_order:
                color=plt.get_cmap(method_cmaps[method])(0.85)
                pareto_handles.append(
                    Line2D(
                        [0],[0],color=color,linewidth=pareto_linewidth,
                        label=f"{method_legend_label_map[method]} Pareto"
                    )
                )
        else:
            pareto_handles=[
                Line2D(
                    [0],[0],color="black",
                    linewidth=pareto_linewidth,label="Pareto frontier"
                )
            ]

    handles=method_handles+lr_handles+pareto_handles

    fig.legend(
        handles=handles,loc=legend_loc,bbox_to_anchor=legend_bbox,
        ncol=min(len(handles),legend_ncol),frameon=legend_frame,
        fontsize=legend_fontsize
    )

    fig.subplots_adjust(
        left=left_margin,right=right_margin,
        bottom=bottom_margin,top=top_margin
    )

    if save_path:
        fig.savefig(f"{save_path}.pdf",format="pdf",bbox_inches="tight",dpi=dpi)
        fig.savefig(f"{save_path}.png",format="png",bbox_inches="tight",dpi=dpi)

    if show_plot:
        plt.show()

    return panel_dfs,fig,axes

import re
import numpy as np
import pandas as pd

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

def summarize_pepcl_default_operating_point_relative_to_optimal(
    ms_ap_df, assay_ap_df, score_cols, method_prefixes,
    optimal_lr, optimal_step, default_lr, default_step,
    pepcl_method="pepcl_r_dr", ref_col="mhcprime",
    optimal_step_budget=None, default_step_budget=None,
    min_seeds=1, focus_seeds=None, ms_comparison="optimal", verbose=True,
):
    """
    Compare a default PepCL operating point with a validation-selected/optimal
    PepCL point.

    Assay gains relative to MHCPrime are computed within replicate using the
    matching validation-split reference before aggregation:
      seeds 0-4 -> ref_col
      seed >=5  -> ref_col_seed_<seed>
    MS continues to use the fixed ref_col baseline for every replicate.

    Reports:
      assay_gain_recovery_pct =
          100 * (A_default - A_base) / (A_optimal - A_base)

      baseline_ms_retained_pct =
          100 * M_default / M_base

      optimal_ms_retained_pct =
          100 * M_optimal / M_base

      ms_retention_vs_optimal_pct =
          100 * (M_default / M_base) / (M_optimal / M_base)
        = 100 * M_default / M_optimal

      ms_default_minus_optimal =
          M_default - M_optimal

    ms_comparison:
      "optimal"  -> emphasize default MS retention relative to optimal
      "baseline" -> emphasize absolute fraction of baseline MHCPrime MS retained
      "both"     -> print both

    Returns
    -------
    summary_df : one-row dataframe with raw and derived statistics
    candidate_df : aggregated PepCL candidate dataframe
    """
    if pepcl_method not in method_prefixes:
        raise ValueError(f"pepcl_method='{pepcl_method}' not found in method_prefixes.")
    if ms_comparison not in {"optimal","baseline","both"}:
        raise ValueError("ms_comparison must be 'optimal', 'baseline', or 'both'.")

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

    assay_ref_cols=[
        c for c in assay_ap_df.columns
        if c==ref_col or str(c).startswith(f"{ref_col}_seed_")
    ]
    assay_map=assay_ap_df.loc[
        assay_ap_df[ref_col].notna(),
        list(dict.fromkeys(assay_ref_cols+usable))
    ].mean(axis=0)

    ms_base=float(ms_map[ref_col])

    def _assay_ref_col_for_seed(seed):
        seed=int(seed)
        return ref_col if seed<=4 else f"{ref_col}_seed_{seed}"

    pattern=re.compile(r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)(?:_step_(?P<checkpoint>\d+))?$")
    rows=[]
    for col in usable:
        m=pattern.search(str(col))
        if m is None: continue
        budget=int(m["budget"]); cp=m["checkpoint"]; is_full=cp is None
        checkpoint=budget if is_full else int(cp)
        if checkpoint>budget: continue
        seed=int(m["seed"])
        assay_ref_col=_assay_ref_col_for_seed(seed)
        if assay_ref_col not in assay_map.index:
            raise ValueError(
                f"Assay reference '{assay_ref_col}' required for seed {seed} is missing."
            )

        assay_performance=float(assay_map[col])
        assay_base_seed=float(assay_map[assay_ref_col])
        ms_performance=float(ms_map[col])

        rows.append({
            "score_col":col,"method":pepcl_method,"learning_rate":m["lr"],
            "step_budget":budget,"checkpoint":checkpoint,"seed":seed,
            "is_full_model":is_full,
            "assay_ref_col":assay_ref_col,
            "assay_base":assay_base_seed,
            "ms_base":ms_base,
            "assay_performance":assay_performance,
            "ms_performance":ms_performance,
            "assay_gain_vs_base":assay_performance-assay_base_seed,
            "ms_change_vs_base":ms_performance-ms_base,
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
            assay_base=("assay_base","mean"),
            ms_base=("ms_base","mean"),
            assay_gain_vs_base=("assay_gain_vs_base","mean"),
            ms_change_vs_base=("ms_change_vs_base","mean"),
            n_seeds=("seed","nunique"),seed_ids=("seed",lambda x:tuple(sorted(set(x)))),
            score_cols_by_seed=("score_col",lambda x:tuple(x)),
            assay_ref_cols_by_seed=("assay_ref_col",lambda x:tuple(x)),
        ).reset_index())

    if focus_seeds is None: candidate_df=candidate_df[candidate_df["n_seeds"]>=min_seeds].copy()
    candidate_df["assay_se"]=candidate_df["assay_sd"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["ms_se"]=candidate_df["ms_sd"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["actual_steps"]=candidate_df["checkpoint"]

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
    assay_opt_gain=float(optimal["assay_gain_vs_base"])
    assay_def_gain=float(default["assay_gain_vs_base"])

    assay_gain_recovery=np.nan if np.isclose(assay_opt_gain,0) else 100.0*assay_def_gain/assay_opt_gain

    # Absolute retention relative to original MHCPrime baseline.
    baseline_ms_retained=np.nan if np.isclose(ms_base,0) else 100.0*M_def/ms_base
    optimal_ms_retained=np.nan if np.isclose(ms_base,0) else 100.0*M_opt/ms_base

    # Relative retention: how much MS performance the default retains compared
    # with the validation-selected optimal PepCL point.
    ms_retention_vs_optimal=np.nan if np.isclose(M_opt,0) else 100.0*M_def/M_opt

    summary_df=pd.DataFrame([{
        # Reference
        "assay_base":float(default["assay_base"]),
        "optimal_assay_base":float(optimal["assay_base"]),
        "default_assay_base":float(default["assay_base"]),
        "ms_base":ms_base,

        # Optimal / validation-selected PepCL
        "optimal_lr":optimal["learning_rate"],"optimal_step_budget":int(optimal["step_budget"]),
        "optimal_step":int(optimal["checkpoint"]),"optimal_n_seeds":int(optimal["n_seeds"]),
        "optimal_seed_ids":optimal["seed_ids"],
        "optimal_assay_ref_cols_by_seed":optimal["assay_ref_cols_by_seed"],
        "optimal_assay":A_opt,"optimal_assay_sd":float(optimal["assay_sd"]),"optimal_assay_se":float(optimal["assay_se"]),
        "optimal_ms":M_opt,"optimal_ms_sd":float(optimal["ms_sd"]),"optimal_ms_se":float(optimal["ms_se"]),
        "optimal_assay_gain_vs_base":assay_opt_gain,"optimal_ms_change_vs_base":M_opt-ms_base,

        # Default PepCL
        "default_lr":default["learning_rate"],"default_step_budget":int(default["step_budget"]),
        "default_step":int(default["checkpoint"]),"default_n_seeds":int(default["n_seeds"]),
        "default_seed_ids":default["seed_ids"],
        "default_assay_ref_cols_by_seed":default["assay_ref_cols_by_seed"],
        "default_assay":A_def,"default_assay_sd":float(default["assay_sd"]),"default_assay_se":float(default["assay_se"]),
        "default_ms":M_def,"default_ms_sd":float(default["ms_sd"]),"default_ms_se":float(default["ms_se"]),
        "default_assay_gain_vs_base":assay_def_gain,"default_ms_change_vs_base":M_def-ms_base,

        # Operating-point comparison
        "assay_gain_recovery_pct":assay_gain_recovery,
        "ms_retention_vs_optimal_pct":ms_retention_vs_optimal,
        "ms_default_minus_optimal":M_def-M_opt,
        "assay_default_minus_optimal":A_def-A_opt,

        # Absolute MS retention relative to original MHCPrime
        "baseline_ms_retained_pct":baseline_ms_retained,
        "optimal_ms_retained_pct":optimal_ms_retained,
        "baseline_ms_lost_pct":100.0-baseline_ms_retained if np.isfinite(baseline_ms_retained) else np.nan,

        # Complementary assay quantity
        "assay_gain_not_recovered_pct":100.0-assay_gain_recovery if np.isfinite(assay_gain_recovery) else np.nan,
    }])

    if verbose:
        print(f"Optimal PepCL: LR={optimal['learning_rate']}, budget={int(optimal['step_budget'])}, step={int(optimal['checkpoint'])}")
        print(f"  Assay={A_opt:.6f} ± {optimal['assay_sd']:.6f} SD | MS={M_opt:.6f} ± {optimal['ms_sd']:.6f} SD")
        print(f"Default PepCL: LR={default['learning_rate']}, budget={int(default['step_budget'])}, step={int(default['checkpoint'])}")
        print(f"  Assay={A_def:.6f} ± {default['assay_sd']:.6f} SD | MS={M_def:.6f} ± {default['ms_sd']:.6f} SD")

        print(f"\nAssay-gain recovery vs optimal: {assay_gain_recovery:.2f}%")
        if ms_comparison in {"optimal","both"}:
            print(f"MS retention vs optimal: {ms_retention_vs_optimal:.2f}%")
        if ms_comparison in {"baseline","both"}:
            print(f"Default MS retained vs MHCPrime baseline: {baseline_ms_retained:.2f}%")
            print(f"Optimal MS retained vs MHCPrime baseline: {optimal_ms_retained:.2f}%")
        print(f"Signed MS difference, default - optimal: {M_def-M_opt:+.6f}")

    return summary_df,candidate_df

import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

def find_validation_nonsignificant_candidates(
    assay_val_df, score_cols, method_prefixes, target_step_budget, target_learning_rate,
    ms_val_df=None, ref_col="mhcprime", test="wilcoxon", alpha=0.05,
    keep_by="both", focus_seeds=None, include_target=True,
    permutation_resamples="exact", permutation_seed=42, verbose=True,
):
    """
    Compare every candidate configuration against a method-specific target
    configuration using paired replicate-level mAP values.

    Each score column is first macro-averaged across validation rows/alleles,
    giving one mAP value per replicate. Candidate and target configurations are
    then compared across matched replicate IDs.

    Parameters
    ----------
    assay_val_df : pd.DataFrame
        Assay per-allele AP dataframe.

    ms_val_df : pd.DataFrame or None
        MS per-allele AP dataframe. May be None if all requested methods use
        keep_by='assay'.

    score_cols : iterable
        Model score columns.

    method_prefixes : dict
        Example:
            {"ft":"escape_con_ft_",
             "pepcl_r_dr":"escape_con_pepcl_r_dr_"}

    target_step_budget, target_learning_rate :
        Scalar applying to every method, OR dict keyed by method.

    test : {"wilcoxon","permutation","ttest"}
        Paired significance test across replicate-level mAP values.

    alpha : float
        Candidate is considered not significantly different when p > alpha.

    keep_by : {"assay","ms","both","either"} or dict
        Filtering criterion.

        assay  -> assay_p > alpha
        ms     -> ms_p > alpha
        both   -> assay_p > alpha AND ms_p > alpha
        either -> assay_p > alpha OR ms_p > alpha

        Can be specified per method, e.g.:
            {"ft":"assay", "pepcl_r_dr":"both"}

    focus_seeds : iterable or None
        None -> use all replicate IDs available for each method's target and
                require each candidate to contain that exact replicate set.
        iterable -> use exactly those replicate IDs and require every target
                    and candidate to contain all of them.

    permutation_resamples : {"exact"} or int
        For paired sign-flip permutation test. Exact is practical for small
        replicate counts such as n=9 (2^9 = 512 sign assignments).

    Returns
    -------
    candidate_df : pd.DataFrame
        All eligible configurations with means/SD/SE, paired differences,
        p-values, significance indicators, and retention flag.

    nonsig_df : pd.DataFrame
        Candidates satisfying the requested keep_by criterion.

    target_df : pd.DataFrame
        Target configuration for each method.
    """
    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty dict.")
    if test not in {"wilcoxon","permutation","ttest"}:
        raise ValueError("test must be 'wilcoxon', 'permutation', or 'ttest'.")
    if not 0<alpha<1: raise ValueError("alpha must be between 0 and 1.")

    def _method_arg(x,method,name):
        if isinstance(x,dict):
            if method not in x: raise ValueError(f"{name} missing method '{method}'.")
            return x[method]
        return x

    def _keep_mode(method):
        mode=_method_arg(keep_by,method,"keep_by")
        if mode not in {"assay","ms","both","either"}:
            raise ValueError(f"keep_by for '{method}' must be 'assay', 'ms', 'both', or 'either'.")
        return mode

    def _normalize_lr(lr):
        s=str(lr)
        if "eN" in s: return s
        try:
            mant,exp=f"{float(lr):.12e}".split("e")
            mant=mant.rstrip("0").rstrip("."); exp=int(exp)
            return f"{mant}eN{abs(exp)}" if exp<0 else f"{mant}e{exp}"
        except (TypeError,ValueError): return s

    if focus_seeds is not None:
        if np.isscalar(focus_seeds): focus_seeds=[focus_seeds]
        try: focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in focus_seeds)))
        except (TypeError,ValueError): raise TypeError("focus_seeds must contain integer replicate IDs.")
        if not focus_seeds: raise ValueError("focus_seeds cannot be empty.")

    need_ms=any(_keep_mode(m) in {"ms","both","either"} for m in method_prefixes)
    if need_ms and ms_val_df is None:
        raise ValueError("ms_val_df is required because at least one method uses keep_by='ms', 'both', or 'either'.")

    if ref_col not in assay_val_df.columns:
        raise ValueError(f"Reference '{ref_col}' not found in assay_val_df.")
    if need_ms and ref_col not in ms_val_df.columns:
        raise ValueError(f"Reference '{ref_col}' not found in ms_val_df.")

    if not assay_val_df.columns.is_unique:
        dup=assay_val_df.columns[assay_val_df.columns.duplicated()].tolist()
        raise ValueError(f"assay_val_df contains duplicate columns: {dup[:10]}")
    if ms_val_df is not None and not ms_val_df.columns.is_unique:
        dup=ms_val_df.columns[ms_val_df.columns.duplicated()].tolist()
        raise ValueError(f"ms_val_df contains duplicate columns: {dup[:10]}")

    score_cols=list(dict.fromkeys(score_cols))
    ordered_methods=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix): return name
        return None

    # Candidate availability is method-specific. FT may require assay only,
    # while PepCL can require assay + MS.
    usable=[]
    for c in score_cols:
        if c==ref_col: continue
        method=_method(c)
        if method is None or c not in assay_val_df.columns: continue
        if _keep_mode(method) in {"ms","both","either"} and (ms_val_df is None or c not in ms_val_df.columns):
            continue
        usable.append(c)

    if not usable: raise ValueError("No usable candidate columns found.")

    if verbose:
        print(f"Score columns supplied: {len(score_cols)}")
        print(f"Usable candidate columns: {len(usable)}")
        print(f"Test: {test} | alpha={alpha}")
        print(f"Replicates: {'focus '+str(focus_seeds) if focus_seeds is not None else 'all target replicates'}")

    def _metric(df,cols,name):
        if not cols: return pd.Series(dtype=float)
        base=df[ref_col].notna()
        bad=[c for c in cols if not df[c].notna().equals(base)]
        if bad:
            raise ValueError(
                f"{name}: {len(bad)} candidates do not share the evaluable row set "
                f"of '{ref_col}'. Examples: {bad[:10]}"
            )
        out=df.loc[base,cols]
        if out.empty: raise ValueError(f"{name}: no evaluable rows remain.")
        if verbose: print(f"{name}: common evaluable row set = {len(out)}")
        return out.mean(axis=0)

    assay_metric=_metric(assay_val_df,usable,"Assay")
    ms_cols=[c for c in usable if ms_val_df is not None and c in ms_val_df.columns]
    ms_metric=_metric(ms_val_df,ms_cols,"MS") if ms_cols else pd.Series(dtype=float)

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
        r"(?:_step_(?P<checkpoint>\d+))?$"
    )

    rows=[]; bad=[]
    for col in usable:
        method=_method(col); m=pattern.search(str(col))
        if method is None or m is None:
            bad.append(col); continue

        budget=int(m["budget"]); seed=int(m["seed"])
        cp=m["checkpoint"]; is_full=cp is None
        checkpoint=budget if is_full else int(cp)

        if checkpoint>budget:
            bad.append(col); continue

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
            "ms_performance":float(ms_metric[col]) if col in ms_metric.index else np.nan,
        })

    seed_df=pd.DataFrame(rows)
    if seed_df.empty: raise ValueError("No valid candidate columns could be parsed.")
    if verbose and bad: print(f"Skipped {len(bad)} unparsable/invalid columns.")

    # Explicit full model wins over duplicate `_step_budget`.
    before=len(seed_df)
    seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)
    seed_df=(seed_df.sort_values(
            ["method","learning_rate","step_budget","seed","checkpoint","_prefer_full"],
            ascending=[True,True,True,True,True,False])
        .drop_duplicates(
            ["method","learning_rate","step_budget","seed","checkpoint"],
            keep="first"
        )
        .drop(columns="_prefer_full")
        .reset_index(drop=True))

    if verbose:
        print(f"Duplicate full/final-checkpoint endpoints removed: {before-len(seed_df)}")

    if focus_seeds is not None:
        before=len(seed_df)
        seed_df=seed_df[seed_df["seed"].isin(focus_seeds)].copy()
        if seed_df.empty: raise ValueError(f"No candidate rows remain for focus_seeds={focus_seeds}.")
        if verbose: print(f"Focused replicate rows: {before} -> {len(seed_df)}")

    keys=["method","learning_rate","step_budget","checkpoint"]

    # Determine the exact replicate set from each method's target.
    required_seed_ids={}
    for method in method_prefixes:
        budget=int(_method_arg(target_step_budget,method,"target_step_budget"))
        lr=_normalize_lr(_method_arg(target_learning_rate,method,"target_learning_rate"))

        t=seed_df[
            (seed_df["method"]==method)&
            (seed_df["step_budget"]==budget)&
            (seed_df["learning_rate"].astype(str)==lr)&
            (seed_df["checkpoint"]==budget)
        ].copy()

        if t.empty:
            raise ValueError(f"No target endpoint for '{method}' with steps={budget}, LR={lr}.")

        ids=tuple(sorted(t["seed"].unique()))

        if focus_seeds is not None:
            required=tuple(focus_seeds)
            if ids!=required:
                raise ValueError(
                    f"Target '{method}' has replicate IDs {ids}, expected {required}."
                )
        else:
            required=ids

        if len(required)<2:
            raise ValueError(
                f"Target '{method}' has only {len(required)} paired replicate(s); "
                "at least 2 are required."
            )

        required_seed_ids[method]=required

    # Require every candidate to contain exactly the same paired replicates as
    # the target for that method.
    info=(seed_df.groupby(keys)["seed"]
          .agg(
              seed_ids=lambda x:tuple(sorted(set(x))),
              n_seeds="nunique"
          ).reset_index())

    valid_info=[]
    for method in method_prefixes:
        required=required_seed_ids[method]
        d=info[info["method"]==method].copy()
        d=d[d["seed_ids"].map(lambda x:x==required)]
        valid_info.append(d)

    info=pd.concat(valid_info,ignore_index=True)
    if info.empty:
        raise ValueError("No candidates contain the required paired replicate sets.")

    valid=set(map(tuple,info[keys].to_numpy()))
    seed_df=seed_df[seed_df[keys].apply(tuple,axis=1).isin(valid)].copy()

    candidate_df=(seed_df.groupby(keys,sort=False)
        .agg(
            assay_performance_mean=("assay_performance","mean"),
            assay_performance_std=("assay_performance","std"),
            ms_performance_mean=("ms_performance","mean"),
            ms_performance_std=("ms_performance","std"),
            n_seeds=("seed","nunique"),
            seed_ids=("seed",lambda x:tuple(sorted(set(x))))
        ).reset_index())

    candidate_df["assay_performance_se"]=candidate_df["assay_performance_std"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["ms_performance_se"]=candidate_df["ms_performance_std"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["actual_steps"]=candidate_df["checkpoint"]

    raw=[]
    for k,g in seed_df.groupby(keys,sort=False):
        g=g.sort_values("seed")
        row=dict(zip(keys,k))
        rep="full" if g["is_full_model"].all() else "checkpoint" if (~g["is_full_model"]).all() else "mixed_endpoint"

        row.update({
            "model_type":rep,
            "is_terminal_endpoint":bool((g["checkpoint"]==g["step_budget"]).all()),
            "score_cols_by_seed":dict(zip(g["seed"],g["score_col"])),
            "assay_values_by_seed":dict(zip(g["seed"],g["assay_performance"])),
            "ms_values_by_seed":dict(zip(g["seed"],g["ms_performance"])),
        })
        raw.append(row)

    candidate_df=candidate_df.merge(pd.DataFrame(raw),on=keys,how="left")

    # Build target dataframe.
    target_rows=[]
    for method in method_prefixes:
        budget=int(_method_arg(target_step_budget,method,"target_step_budget"))
        lr=_normalize_lr(_method_arg(target_learning_rate,method,"target_learning_rate"))

        t=candidate_df[
            (candidate_df["method"]==method)&
            (candidate_df["step_budget"]==budget)&
            (candidate_df["learning_rate"].astype(str)==lr)&
            (candidate_df["checkpoint"]==budget)
        ]

        if len(t)!=1:
            raise ValueError(
                f"Expected exactly one aggregated target for '{method}', found {len(t)}."
            )

        target_rows.append(t.iloc[0])

    target_df=pd.DataFrame(target_rows).reset_index(drop=True)
    rng=np.random.default_rng(permutation_seed)

    def _paired_test(candidate,target):
        x=np.asarray(candidate,dtype=float)
        y=np.asarray(target,dtype=float)
        mask=np.isfinite(x)&np.isfinite(y)
        x=x[mask]; y=y[mask]

        if len(x)<2: return np.nan,np.nan,len(x)

        d=x-y
        effect=float(np.mean(d))

        if np.allclose(d,0):
            return effect,1.0,len(d)

        if test=="wilcoxon":
            try:
                p=float(wilcoxon(x,y,alternative="two-sided").pvalue)
            except ValueError:
                p=1.0 if np.allclose(d,0) else np.nan

        elif test=="ttest":
            p=float(ttest_rel(x,y,nan_policy="omit").pvalue)

        else:
            observed=abs(effect)
            n=len(d)

            if permutation_resamples=="exact":
                if n>20:
                    raise ValueError(
                        "Exact permutation requested with >20 pairs; "
                        "use an integer permutation_resamples."
                    )

                signs=((np.arange(2**n)[:,None]>>np.arange(n))&1)*2-1
                stats=np.abs((signs*d).mean(axis=1))
                p=float(np.mean(stats>=observed-1e-15))

            else:
                n_resamples=int(permutation_resamples)
                if n_resamples<1:
                    raise ValueError(
                        "permutation_resamples must be 'exact' or a positive integer."
                    )

                count=0
                for _ in range(n_resamples):
                    signs=rng.choice((-1,1),size=n)
                    count+=abs(np.mean(signs*d))>=observed-1e-15

                p=float((count+1)/(n_resamples+1))

        return effect,p,len(d)

    tested=[]
    for _,row in candidate_df.iterrows():
        method=row["method"]
        target=target_df[target_df["method"]==method].iloc[0]
        ids=required_seed_ids[method]
        mode=_keep_mode(method)

        assay_candidate=np.array(
            [row["assay_values_by_seed"][s] for s in ids],
            dtype=float
        )
        assay_target=np.array(
            [target["assay_values_by_seed"][s] for s in ids],
            dtype=float
        )

        assay_delta,assay_p,assay_n=_paired_test(
            assay_candidate,
            assay_target
        )

        if mode in {"ms","both","either"}:
            ms_candidate=np.array(
                [row["ms_values_by_seed"][s] for s in ids],
                dtype=float
            )
            ms_target=np.array(
                [target["ms_values_by_seed"][s] for s in ids],
                dtype=float
            )

            ms_delta,ms_p,ms_n=_paired_test(
                ms_candidate,
                ms_target
            )
        else:
            ms_delta=ms_p=np.nan
            ms_n=0

        assay_nonsig=bool(assay_p>alpha) if np.isfinite(assay_p) else False
        ms_nonsig=bool(ms_p>alpha) if np.isfinite(ms_p) else False

        if mode=="assay":
            retain=assay_nonsig
        elif mode=="ms":
            retain=ms_nonsig
        elif mode=="both":
            retain=assay_nonsig and ms_nonsig
        else:  # either
            retain=assay_nonsig or ms_nonsig

        tested.append({
            **{k:row[k] for k in keys},
            "test":test,
            "alpha":alpha,
            "keep_by":mode,

            "assay_delta_from_target":assay_delta,
            "assay_pvalue":assay_p,
            "assay_n_pairs":assay_n,
            "assay_nonsignificant":assay_nonsig,

            "ms_delta_from_target":ms_delta,
            "ms_pvalue":ms_p,
            "ms_n_pairs":ms_n,
            "ms_nonsignificant":ms_nonsig,

            "nonsignificant_either":assay_nonsig or ms_nonsig,
            "nonsignificant_both":assay_nonsig and ms_nonsig,
            "retain_candidate":retain,
        })

    test_df=pd.DataFrame(tested)
    candidate_df=candidate_df.merge(test_df,on=keys,how="left")

    # Mark target configuration.
    candidate_df["is_target"]=False

    for method in method_prefixes:
        budget=int(_method_arg(target_step_budget,method,"target_step_budget"))
        lr=_normalize_lr(_method_arg(target_learning_rate,method,"target_learning_rate"))

        mask=(
            (candidate_df["method"]==method)&
            (candidate_df["step_budget"]==budget)&
            (candidate_df["learning_rate"].astype(str)==lr)&
            (candidate_df["checkpoint"]==budget)
        )

        candidate_df.loc[mask,"is_target"]=True

    nonsig_df=candidate_df[candidate_df["retain_candidate"]].copy()

    if not include_target:
        nonsig_df=nonsig_df[~nonsig_df["is_target"]].copy()

    nonsig_df=nonsig_df.sort_values(
        ["method","checkpoint","step_budget","learning_rate"],
        ascending=[True,True,True,True]
    ).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*90}")
        print("VALIDATION P-VALUE ANALYSIS")
        print(f"{'='*90}")
        print(f"Test: {test} | alpha={alpha}")
        print(
            f"Replicates: "
            f"{'focus '+str(focus_seeds) if focus_seeds is not None else 'all target replicates'}"
        )

        for method in method_prefixes:
            target=target_df[target_df["method"]==method].iloc[0]
            sub=nonsig_df[nonsig_df["method"]==method]
            mode=_keep_mode(method)

            print(f"\nMETHOD: {method}")
            print(
                f"Target: LR={target['learning_rate']}, "
                f"steps={int(target['step_budget'])}, "
                f"replicates={target['seed_ids']}"
            )

            if mode=="assay":
                desc="assay P > alpha"
            elif mode=="ms":
                desc="MS P > alpha"
            elif mode=="both":
                desc="assay P > alpha AND MS P > alpha"
            else:
                desc="assay P > alpha OR MS P > alpha"

            print(f"Filter: {mode} ({desc})")
            print(f"Retained candidates: {len(sub)}")

            cols=[
                "learning_rate","step_budget","checkpoint","model_type",
                "n_seeds","seed_ids",
                "assay_performance_mean","assay_performance_std",
                "assay_delta_from_target","assay_pvalue",
                "assay_nonsignificant"
            ]

            if mode in {"ms","both","either"}:
                cols += [
                    "ms_performance_mean","ms_performance_std",
                    "ms_delta_from_target","ms_pvalue",
                    "ms_nonsignificant",
                    "nonsignificant_either",
                    "nonsignificant_both"
                ]

            cols += ["retain_candidate","is_target"]

            if not sub.empty:
                print(sub[cols].to_string(index=False))

    return candidate_df,nonsig_df,target_df

import re
import numpy as np
import pandas as pd

def evaluate_ft_early_stopping(
    assay_val_df, model_groups, ft_prefix, ref_col="mhcprime", focus_seeds=None,
    rule="patience", patience=3, patience_mode="evaluations", min_delta=0.001,
    relative_delta=False, min_steps=0, min_evaluations=1, restore_best=True,
    rolling_window=3, rolling_min_delta=0.0,
    slope_window=4, slope_threshold=0.0,
    drop_threshold=0.005, drop_patience=2,
    require_complete_seeds=True, include_full_endpoint=True, verbose=True,
):
    """
    Reconstruct FT validation trajectories and simulate conventional early stopping
    independently for each LR x total-step-budget trajectory.

    Each checkpoint performance is the mean mAP across seeds/replicates.

    Parameters
    ----------
    assay_val_df : pd.DataFrame
        Per-allele AP dataframe. Rows are validation alleles/groups and model
        score columns contain AP. Each score column is macro-averaged across rows
        to obtain one mAP value for that seed/checkpoint.

    model_groups : dict
        Mapping:
            full_model_name -> iterable of checkpoint_model_names

        Full names follow:
            ..._steps_{budget}_lr_{lr}_seed_{seed}

        Checkpoints follow:
            ..._steps_{budget}_lr_{lr}_seed_{seed}_step_{checkpoint}

    ft_prefix : str
        Prefix identifying FT models.

    focus_seeds : iterable or None
        Replicate IDs to use. None uses all available replicates.
        If require_complete_seeds=True, every trajectory point must contain the
        same required replicate set.

    rule : {"patience","rolling","slope","drop"}
        patience:
            Standard early stopping. Track the historical best validation mAP.
            A new point counts as improvement only if it exceeds the best by
            min_delta. Stop after `patience` failures.

        rolling:
            Compare mean validation performance in the latest `rolling_window`
            evaluations with the preceding window. Stop when improvement is
            <= rolling_min_delta.

        slope:
            Fit validation mAP ~ training step over the latest `slope_window`
            points. Stop when slope <= slope_threshold.

        drop:
            Stop after `drop_patience` consecutive evaluations whose mean
            performance is at least `drop_threshold` below the historical best.

    patience : int
        Patience for rule="patience".

    patience_mode : {"evaluations","steps"}
        "evaluations": patience counts validation checkpoints.
        "steps": patience is the number of training steps since the last
                 qualifying improvement.

    min_delta : float
        Minimum improvement required by the patience rule.

    relative_delta : bool
        If True, min_delta is interpreted relative to abs(best):
            required improvement = abs(best) * min_delta

    min_steps : int
        Do not trigger early stopping before this actual training step.

    min_evaluations : int
        Minimum number of observed validation points before stopping.

    restore_best : bool
        If True, selected_step is the best checkpoint observed at/before the
        stop point. If False, selected_step equals stop_step.

    rolling_window : int
        Window size for rule="rolling".

    rolling_min_delta : float
        Required improvement of recent-window mean over previous-window mean.

    slope_window : int
        Number of recent validation points used for rule="slope".

    slope_threshold : float
        Stop when fitted slope <= this value.

    drop_threshold : float
        Absolute decrease from historical best required for rule="drop".

    drop_patience : int
        Consecutive dropped evaluations required for rule="drop".

    require_complete_seeds : bool
        Require the same replicate IDs at every trajectory point.

    include_full_endpoint : bool
        Include explicit full model as the terminal checkpoint. If both an
        explicit full model and `_step_{budget}` exist, explicit full wins.

    Returns
    -------
    trajectory_df : pd.DataFrame
        One row per LR x budget x checkpoint. performance_mean is the average
        mAP across seeds/replicates at that checkpoint.

    result_df : pd.DataFrame
        One row per LR x budget giving the simulated stopping decision.
    """
    valid_rules={"patience","rolling","slope","drop"}
    if rule not in valid_rules:
        raise ValueError(f"rule must be one of {sorted(valid_rules)}.")
    if patience_mode not in {"evaluations","steps"}:
        raise ValueError("patience_mode must be 'evaluations' or 'steps'.")
    if patience<1 or drop_patience<1:
        raise ValueError("patience and drop_patience must be >= 1.")
    if rolling_window<1 or slope_window<2:
        raise ValueError("rolling_window >= 1 and slope_window >= 2 required.")
    if min_evaluations<1:
        raise ValueError("min_evaluations must be >= 1.")

    if focus_seeds is not None:
        if np.isscalar(focus_seeds):
            focus_seeds=[focus_seeds]
        focus_seeds=tuple(sorted(dict.fromkeys(int(x) for x in focus_seeds)))
        if not focus_seeds:
            raise ValueError("focus_seeds cannot be empty.")

    if ref_col not in assay_val_df.columns:
        raise ValueError(f"'{ref_col}' not found in assay_val_df.")

    base_mask=assay_val_df[ref_col].notna()

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
        r"(?:_step_(?P<checkpoint>\d+))?$"
    )

    # Expand model_groups into full models + checkpoints.
    model_names=[]
    for full_name,checkpoints in model_groups.items():
        if str(full_name).startswith(ft_prefix):
            if include_full_endpoint:
                model_names.append(full_name)
            if checkpoints is not None:
                model_names.extend([
                    c for c in checkpoints
                    if str(c).startswith(ft_prefix)
                ])

    model_names=list(dict.fromkeys(model_names))

    rows=[]
    skipped=[]

    for col in model_names:
        if col not in assay_val_df.columns:
            skipped.append(col)
            continue

        m=pattern.search(str(col))
        if m is None:
            skipped.append(col)
            continue

        budget=int(m["budget"])
        seed=int(m["seed"])
        cp=m["checkpoint"]
        is_full=cp is None
        checkpoint=budget if is_full else int(cp)

        if checkpoint>budget:
            skipped.append(col)
            continue

        if focus_seeds is not None and seed not in focus_seeds:
            continue

        if not assay_val_df[col].notna().equals(base_mask):
            raise ValueError(
                f"'{col}' does not share the evaluable row set of '{ref_col}'."
            )

        rows.append({
            "score_col":col,
            "learning_rate":m["lr"],
            "step_budget":budget,
            "checkpoint":checkpoint,
            "seed":seed,
            "is_full_model":is_full,
            "map":float(assay_val_df.loc[base_mask,col].mean()),
        })

    seed_df=pd.DataFrame(rows)

    if seed_df.empty:
        raise ValueError("No valid FT trajectory columns were found.")

    # Explicit full model wins over duplicate `_step_budget`.
    seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)

    seed_df=(
        seed_df.sort_values(
            ["learning_rate","step_budget","seed","checkpoint","_prefer_full"],
            ascending=[True,True,True,True,False]
        )
        .drop_duplicates(
            ["learning_rate","step_budget","seed","checkpoint"],
            keep="first"
        )
        .drop(columns="_prefer_full")
        .reset_index(drop=True)
    )

    keys=["learning_rate","step_budget","checkpoint"]

    info=(
        seed_df.groupby(keys)["seed"]
        .agg(
            seed_ids=lambda x:tuple(sorted(set(x))),
            n_seeds="nunique"
        )
        .reset_index()
    )

    if focus_seeds is not None:
        required=focus_seeds
        info=info[
            info["seed_ids"].map(lambda x:x==required)
        ].copy()

    elif require_complete_seeds:
        sets=info["seed_ids"].drop_duplicates().tolist()

        if len(sets)>1:
            raise ValueError(
                f"Inconsistent replicate sets across FT trajectory points: {sets}. "
                "Provide focus_seeds or set require_complete_seeds=False."
            )

    if info.empty:
        raise ValueError(
            "No trajectory points remain after replicate filtering."
        )

    valid_keys=set(map(tuple,info[keys].to_numpy()))

    seed_df=seed_df[
        seed_df[keys].apply(tuple,axis=1).isin(valid_keys)
    ].copy()

    # Average checkpoint mAP across seeds/replicates.
    trajectory_df=(
        seed_df.groupby(keys,sort=False)
        .agg(
            performance_mean=("map","mean"),
            performance_sd=("map","std"),
            n_seeds=("seed","nunique"),
            seed_ids=("seed",lambda x:tuple(sorted(set(x))))
        )
        .reset_index()
    )

    trajectory_df["performance_se"]=(
        trajectory_df["performance_sd"] /
        np.sqrt(trajectory_df["n_seeds"])
    )

    trajectory_df=trajectory_df.sort_values(
        ["learning_rate","step_budget","checkpoint"]
    ).reset_index(drop=True)

    def _simulate(g):
        g=g.sort_values("checkpoint").reset_index(drop=True)

        steps=g["checkpoint"].to_numpy(dtype=int)
        vals=g["performance_mean"].to_numpy(dtype=float)

        best_idx=0
        best_val=vals[0]
        last_improvement_step=steps[0]

        bad_count=0
        drop_count=0

        stopped=False
        stop_idx=len(g)-1
        reason="budget_exhausted"

        for i in range(1,len(g)):
            step=steps[i]
            value=vals[i]

            delta=(
                min_delta*abs(best_val)
                if relative_delta
                else min_delta
            )

            improved=value>best_val+delta

            if improved:
                best_idx=i
                best_val=value
                last_improvement_step=step
                bad_count=0
            else:
                bad_count+=1

            can_stop=(
                step>=min_steps and
                (i+1)>=min_evaluations
            )

            if rule=="patience" and can_stop:

                if patience_mode=="evaluations":
                    trigger=bad_count>=patience

                else:
                    trigger=(
                        step-last_improvement_step
                    )>=patience

                if trigger:
                    stopped=True
                    stop_idx=i
                    reason=f"patience_{patience_mode}"
                    break

            elif rule=="rolling" and can_stop and i+1>=2*rolling_window:

                previous=vals[
                    i+1-2*rolling_window:
                    i+1-rolling_window
                ].mean()

                recent=vals[
                    i+1-rolling_window:
                    i+1
                ].mean()

                if recent-previous<=rolling_min_delta:
                    stopped=True
                    stop_idx=i
                    reason="rolling_plateau"
                    break

            elif rule=="slope" and can_stop and i+1>=slope_window:

                xs=steps[
                    i+1-slope_window:
                    i+1
                ].astype(float)

                ys=vals[
                    i+1-slope_window:
                    i+1
                ]

                slope=np.polyfit(xs,ys,1)[0]

                if slope<=slope_threshold:
                    stopped=True
                    stop_idx=i
                    reason="slope"
                    break

            elif rule=="drop":

                historical_best=np.max(vals[:i+1])

                if historical_best-value>=drop_threshold:
                    drop_count+=1
                else:
                    drop_count=0

                if can_stop and drop_count>=drop_patience:
                    stopped=True
                    stop_idx=i
                    reason="sustained_drop"
                    break

        # Best checkpoint using only points observed before stopping.
        observed_vals=vals[:stop_idx+1]
        observed_best_idx=int(np.argmax(observed_vals))

        selected_idx=(
            observed_best_idx
            if restore_best
            else stop_idx
        )

        return {
            "learning_rate":g.loc[0,"learning_rate"],
            "step_budget":int(g.loc[0,"step_budget"]),
            "rule":rule,

            "stopped_early":bool(
                stopped and
                steps[stop_idx]<int(g.loc[0,"step_budget"])
            ),

            "stop_step":int(steps[stop_idx]),
            "stop_performance":float(vals[stop_idx]),

            "selected_step":int(steps[selected_idx]),
            "selected_performance":float(vals[selected_idx]),

            "best_observed_step":int(steps[observed_best_idx]),
            "best_observed_performance":float(
                vals[observed_best_idx]
            ),

            "final_budget_step":int(steps[-1]),
            "final_budget_performance":float(vals[-1]),

            "steps_saved":int(
                g.loc[0,"step_budget"]-steps[stop_idx]
            ),

            "selected_steps_saved":int(
                g.loc[0,"step_budget"]-steps[selected_idx]
            ),

            "n_evaluations_total":len(g),
            "n_evaluations_seen":int(stop_idx+1),

            "stop_reason":reason,
            "seed_ids":g.loc[0,"seed_ids"],
            "n_seeds":int(g.loc[0,"n_seeds"]),
        }

    results=[]

    for _,g in trajectory_df.groupby(
        ["learning_rate","step_budget"],
        sort=False
    ):
        results.append(_simulate(g))

    result_df=pd.DataFrame(results)

    result_df=result_df.sort_values(
        ["learning_rate","step_budget"]
    ).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*95}")
        print("FT EARLY-STOPPING ANALYSIS")
        print(f"{'='*95}")
        print(f"Rule: {rule}")

        if rule=="patience":
            print(
                f"patience={patience} ({patience_mode}), "
                f"min_delta={min_delta}"
                +(" relative" if relative_delta else " absolute")
            )

        elif rule=="rolling":
            print(
                f"rolling_window={rolling_window}, "
                f"rolling_min_delta={rolling_min_delta}"
            )

        elif rule=="slope":
            print(
                f"slope_window={slope_window}, "
                f"slope_threshold={slope_threshold}"
            )

        else:
            print(
                f"drop_threshold={drop_threshold}, "
                f"drop_patience={drop_patience}"
            )

        print(
            f"min_steps={min_steps}, "
            f"min_evaluations={min_evaluations}, "
            f"restore_best={restore_best}"
        )

        print(
            f"Trajectory points: {len(trajectory_df)} | "
            f"LR×budget trajectories: {len(result_df)}"
        )

        if skipped:
            print(
                f"Missing/unparsable model columns skipped: "
                f"{len(skipped)}"
            )

        print()

        print(
            result_df[[
                "learning_rate",
                "step_budget",
                "stopped_early",
                "stop_step",
                "selected_step",
                "selected_performance",
                "final_budget_performance",
                "steps_saved",
                "n_evaluations_seen",
                "stop_reason"
            ]].to_string(index=False)
        )

    return trajectory_df,result_df

def plot_ft_matches_trajectory_grid_from_ap(
    assay_configs, plotting_params=None,
    nrows=3, ncols=2, fig_size=(12,16), dpi=300,
    wspace=0.32, hspace=0.30,
    left_margin=.08, right_margin=.97, bottom_margin=.06, top_margin=.97,
    save_path=None, show_plot=True, verbose=False,
):
    """
    Plot multiple FT-vs-PepCL matched-trajectory panels in one grid.

    Parameters
    ----------
    assay_configs : dict
        Dictionary keyed by dataset/assay name.

        Each value must contain:
            comparison_df
            ms_ap_df
            assay_ap_df
            score_cols
            method_prefixes

        Any plotting parameter may also be supplied inside a dataset's dict
        to override plotting_params for that panel.

        Example:
            assay_configs={
                "yd":{
                    "comparison_df":holec_comparison_df,
                    "ms_ap_df":holec_pepcl_ft_es_ap_dfs["ms_val_matched"],
                    "assay_ap_df":holec_pepcl_ft_es_ap_dfs["assay_val"],
                    "score_cols":["mhcprime"] + flatten_dict_keys_and_values(holec_model_groups),
                    "method_prefixes":holec_method_prefixes,
                    "title":"YD",
                    "assay_name":"YD",
                    "ylabel":"YD ΔmAP vs MHCPrime",
                    "vertical_delta_prefix":"ΔYD = ",
                },
                ...
            }

    plotting_params : dict or None
        Shared plotting parameters used by all panels. Dataset-specific
        entries in assay_configs override these values.

    nrows, ncols : int
        Grid dimensions. Default 3 x 2.

    wspace, hspace : float
        Spacing between outer grid cells.

    Notes
    -----
    - MHCPrime-relative coordinates are computed within replicate before
      aggregation. MS always uses `ref_col`; assay seeds 0-4 use `ref_col`
      and seeds >=5 use `{ref_col}_seed_{seed}`.
    - Unused grid cells are left blank.
    - Each panel gets its own colorbar region, preventing neighboring axes,
      colorbars, and labels from overlapping.
    - Panel-level fig_size, dpi, show_plot, save_path, and subplot margins
      are ignored because the grid controls those globally.

    Returns
    -------
    results : dict
        Per-dataset plotting outputs.
    fig : matplotlib Figure
    axes : ndarray
        nrows x ncols array of main plotting axes.
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=["Nimbus Sans","Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    if not isinstance(assay_configs,dict) or not assay_configs:
        raise ValueError("assay_configs must be a non-empty dictionary.")
    if nrows<1 or ncols<1:
        raise ValueError("nrows and ncols must be >= 1.")
    if len(assay_configs)>nrows*ncols:
        raise ValueError(
            f"Grid has {nrows*ncols} cells but assay_configs contains "
            f"{len(assay_configs)} assays."
        )

    plotting_params={} if plotting_params is None else dict(plotting_params)

    required={
        "comparison_df","ms_ap_df","assay_ap_df",
        "score_cols","method_prefixes"
    }

    # Parameters controlled by the outer grid, not individual panels.
    structural={
        "fig_size","dpi","save_path","show_plot",
        "left_margin","right_margin","bottom_margin","top_margin"
    }

    defaults=dict(
        seeds=None,
        ref_col="mhcprime", ms_name="MS", assay_name="Assay",
        ms_map_col="mAP", assay_map_col="mAP",
        ft_method="ft", pepcl_method="pepcl_r_dr",

        # Point / trajectory styling.
        method_style="fill", method_cmaps=None, method_fill=None,
        step_cmap="viridis",
        marker_size=70, line_width=1.5, hollow_linewidth=1.8,
        background_point_color="0.80",
        background_line_color="0.82",
        background_point_alpha=.55,
        background_line_alpha=.45,

        # Background edges.
        background_filled_edge_color="black",
        background_filled_edge_width=.5,
        background_hollow_edge_color=None,
        background_hollow_edge_width=None,

        matched_point_alpha=1.0,
        target_point_alpha=1.0,
        highlight_marker_size=200,

        # LR markers / legend.
        lr_markers=None,
        lr_marker_scales=None,
        method_legend_labels=None,
        method_legend_color="0.6",

        # Optional FT highlight.
        ft_highlight_target=None,
        ft_highlight_name="FT",
        ft_highlight_use_step_color=True,

        # Target labels.
        pepcl_target_name="PepCL",
        ft_adaptation_name="FT matched adaptation",
        ft_ms_name="FT matched retention",

        # Horizontal comparison.
        horizontal_bar=True,
        horizontal_bar_color="0.15",
        horizontal_bar_linewidth=2.5,
        horizontal_bar_linestyle="-",
        horizontal_cap_size=.018,
        horizontal_bar_end_padding=0.0,
        horizontal_target_clearance=.018,
        horizontal_match_clearance=.012,
        horizontal_delta_prefix="ΔMS = ",
        horizontal_delta_color="0.15",
        horizontal_delta_fontsize=10,
        horizontal_delta_rotation=0,
        horizontal_delta_xoffset=0.0,
        horizontal_delta_yoffset=.015,
        horizontal_delta_ha="center",
        horizontal_delta_va="bottom",

        # Vertical comparison.
        vertical_bar=True,
        vertical_bar_color="0.15",
        vertical_bar_linewidth=2.5,
        vertical_bar_linestyle="-",
        vertical_cap_size=.018,
        vertical_bar_end_padding=0.0,
        vertical_target_clearance=.018,
        vertical_match_clearance=.012,
        vertical_delta_prefix="ΔAssay = ",
        vertical_delta_color="0.15",
        vertical_delta_fontsize=10,
        vertical_delta_rotation=0,
        vertical_delta_xoffset=.015,
        vertical_delta_yoffset=0.0,
        vertical_delta_ha="left",
        vertical_delta_va="center",

        # Delta formatting.
        delta_decimals=2,
        trim_delta_zeros=True,

        # Reference lines.
        zero_lines=True,
        zero_line_color="0.35",
        zero_line_width=1.0,
        zero_line_style="--",
        equal_tradeoff_line=False,
        equal_tradeoff_linecolor="0.8",
        equal_tradeoff_linewidth=1.2,
        equal_tradeoff_linestyle="--",

        # Axes.
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        square_panel=True,
        axis_label_fontsize=11,
        title_fontsize=12,
        axis_label_pad=8,
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

        # Colorbar.
        show_step_colorbar=True,
        colorbar_width=.035,
        colorbar_pad=.10,
        colorbar_label="Optimization step",
        colorbar_tick_fontsize=9,
        colorbar_label_fontsize=10,
        colorbar_outline_width=1.0,

        # Parameter text.
        show_target_text=True,
        show_matched_ft_text=True,
        target_text_position="bottom_left_above_zero",
        target_text_fontsize=9,
        target_text_rotation=0,
        target_text_xpad=.03,
        target_text_ypad=.03,
        target_text_linespacing=1.3,
        target_text_ha=None,
        target_text_va=None,

        # Legend.
        legend=False,
        legend_ncol=6,
        legend_fontsize=9,
        legend_loc="lower center",
        legend_bbox=(0.5,-.16),
        legend_frame=False,
        legend_columnspacing=1.2,
        legend_handletextpad=.5,
    )

    def _plot_one(ax,cax,cfg,dataset_name):
        p=dict(defaults)
        p.update({k:v for k,v in plotting_params.items() if k not in structural})
        p.update({k:v for k,v in cfg.items() if k not in required and k not in structural})

        comparison_df=cfg["comparison_df"]
        ms_ap_df=cfg["ms_ap_df"]
        assay_ap_df=cfg["assay_ap_df"]
        score_cols=cfg["score_cols"]
        method_prefixes=cfg["method_prefixes"]

        if p["seeds"] is None:
            raise ValueError(
                f"Dataset '{dataset_name}' has no seeds specified. "
                "Supply seeds in plotting_params or its assay config."
            )

        required_roles={"pepcl_target","ft_match_adaptation","ft_match_ms"}
        if "role" not in comparison_df.columns:
            raise ValueError(f"{dataset_name}: comparison_df must contain 'role'.")

        missing=required_roles-set(comparison_df["role"])
        if missing:
            raise ValueError(
                f"{dataset_name}: comparison_df is missing roles: {sorted(missing)}"
            )

        if p["ft_method"] not in method_prefixes or p["pepcl_method"] not in method_prefixes:
            raise ValueError(
                f"{dataset_name}: '{p['ft_method']}' and '{p['pepcl_method']}' "
                "must exist in method_prefixes."
            )

        if p["method_style"] not in {"fill","gradient"}:
            raise ValueError("method_style must be 'fill' or 'gradient'.")

        def _parse_lr(x):
            s=str(x)
            if "eN" in s:
                a,b=s.split("eN",1)
                return float(a)*10**(-int(b))
            return float(s)

        def _lr_label(v):
            v=float(v)
            if v==0: return "0"
            mant,exp=f"{v:.10e}".split("e")
            return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"

        def _number_label(v):
            v=float(v)
            s=f"{abs(v):.{p['delta_decimals']}f}"
            s=s.rstrip("0").rstrip(".") if p["trim_delta_zeros"] else s
            if np.isclose(v,0.0): return s
            return ("+" if v>0 else "-")+s

        def _tick_fmt(v,_):
            if abs(v)<10**(-(p["tick_decimals"]+2)): v=0
            s=f"{v:.{p['tick_decimals']}f}"
            return s.rstrip("0").rstrip(".") if p["trim_tick_zeros"] else s

        def _get_map(df,map_col,name):
            if map_col in df.columns:
                s=df[map_col].copy()
                s.index=s.index.astype(str)
                return s.astype(float)
            return df.mean(axis=0).astype(float)

        def _edge_spec(color,width,default_color,default_width):
            if color is None: color=default_color
            if color is False or str(color).lower()=="none":
                return "none",0.0
            if width is None: width=default_width
            return color,float(width)

        seeds=p["seeds"]
        if np.isscalar(seeds): seeds=[seeds]
        seeds=tuple(sorted(dict.fromkeys(int(s) for s in seeds)))
        if not seeds:
            raise ValueError(f"{dataset_name}: seeds cannot be empty.")

        ms_map=_get_map(ms_ap_df,p["ms_map_col"],p["ms_name"])
        assay_map=_get_map(assay_ap_df,p["assay_map_col"],p["assay_name"])

        if p["ref_col"] not in ms_map.index or p["ref_col"] not in assay_map.index:
            raise ValueError(
                f"{dataset_name}: reference '{p['ref_col']}' must exist in both inputs."
            )

        score_cols=list(dict.fromkeys(score_cols))
        usable=[c for c in score_cols if c in ms_map.index and c in assay_map.index]
        if not usable:
            raise ValueError(f"{dataset_name}: no score columns are present in both inputs.")

        ref_ms=float(ms_map[p["ref_col"]])

        def _assay_ref_col_for_seed(seed):
            seed=int(seed)
            return p["ref_col"] if seed<=4 else f"{p['ref_col']}_seed_{seed}"

        ordered_methods=sorted(
            method_prefixes.items(),
            key=lambda x:len(x[1]),
            reverse=True
        )

        def _method(col):
            for name,prefix in ordered_methods:
                if str(col).startswith(prefix): return name
            return None

        pattern=re.compile(
            r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
            r"(?:_step_(?P<checkpoint>\d+))?$"
        )

        rows=[]

        for col in usable:
            method=_method(col)
            m=pattern.search(str(col))

            if method not in {p["ft_method"],p["pepcl_method"]} or m is None:
                continue

            budget=int(m["budget"])
            cp=m["checkpoint"]
            step=budget if cp is None else int(cp)

            if step>budget:
                continue

            seed=int(m["seed"])
            assay_ref_col=_assay_ref_col_for_seed(seed)

            if assay_ref_col not in assay_map.index:
                raise ValueError(
                    f"{dataset_name}: assay reference '{assay_ref_col}' required "
                    f"for seed {seed} is missing."
                )

            rows.append({
                "score_col":col,
                "method":method,
                "learning_rate":m["lr"],
                "learning_rate_float":_parse_lr(m["lr"]),
                "step_budget":budget,
                "step":step,
                "seed":seed,
                "is_explicit_full":cp is None,
                "ms_ref_col":p["ref_col"],
                "assay_ref_col":assay_ref_col,
                f"{p['ms_name']}_score":float(ms_map[col]),
                f"{p['assay_name']}_score":float(assay_map[col]),
                f"{p['ms_name']}_delta":float(ms_map[col])-ref_ms,
                f"{p['assay_name']}_delta":float(assay_map[col])-float(assay_map[assay_ref_col]),
            })

        raw_df=pd.DataFrame(rows)

        if raw_df.empty:
            raise ValueError(f"{dataset_name}: no FT/PepCL trajectory points could be parsed.")

        raw_df["_prefer_full"]=raw_df["is_explicit_full"].astype(int)

        raw_df=(
            raw_df.sort_values(
                ["method","learning_rate","seed","step_budget","step","_prefer_full"],
                ascending=[True,True,True,True,True,False]
            )
            .drop_duplicates(
                ["method","learning_rate","seed","step_budget","step"],
                keep="first"
            )
            .drop(columns="_prefer_full")
            .reset_index(drop=True)
        )

        raw_df=raw_df[raw_df["seed"].isin(seeds)].copy()

        group_cols=[
            "method","learning_rate","learning_rate_float",
            "step_budget","step"
        ]

        seed_sets=(
            raw_df.groupby(group_cols)["seed"]
            .agg(lambda x:tuple(sorted(set(x))))
            .reset_index(name="seed_ids")
        )

        incomplete=seed_sets[seed_sets["seed_ids"].map(lambda x:x!=seeds)]

        if not incomplete.empty:
            raise ValueError(
                f"{dataset_name}: {len(incomplete)} configurations do not contain "
                f"all requested replicates {seeds}."
            )

        value_cols=[
            f"{p['ms_name']}_score",
            f"{p['assay_name']}_score",
            f"{p['ms_name']}_delta",
            f"{p['assay_name']}_delta",
        ]

        plot_df=(
            raw_df.groupby(group_cols,as_index=False,sort=False)
            .agg(
                **{c:(c,"mean") for c in value_cols},
                n_seeds=("seed","nunique"),
                seed_ids=("seed",lambda x:tuple(sorted(set(x))))
            )
        )

        method_order=[
            m for m in method_prefixes
            if m in plot_df["method"].unique()
        ]

        if p["method_fill"] is None:
            method_fill={
                m:("filled" if i%2==0 else "hollow")
                for i,m in enumerate(method_order)
            }
        else:
            method_fill=dict(p["method_fill"])

        lr_info=(
            plot_df[["learning_rate","learning_rate_float"]]
            .drop_duplicates()
            .sort_values("learning_rate_float")
        )

        lr_order=lr_info["learning_rate"].tolist()

        default_markers=["o","^","s","p","D","v","P","X"]

        if p["lr_markers"] is None:
            marker_map={
                lr:default_markers[i%len(default_markers)]
                for i,lr in enumerate(lr_order)
            }
        elif isinstance(p["lr_markers"],dict):
            marker_map={lr:p["lr_markers"][lr] for lr in lr_order}
        else:
            markers=list(p["lr_markers"])
            marker_map={
                lr:markers[i%len(markers)]
                for i,lr in enumerate(lr_order)
            }

        scales={
            "o":1.20,"^":1.0,"s":1.0,"p":1.0,
            "D":1.0,"v":1.0,"P":1.0,"X":1.0
        }

        if p["lr_marker_scales"] is not None:
            scales.update(p["lr_marker_scales"])

        marker_scale_map={
            lr:scales.get(marker_map[lr],1.0)
            for lr in lr_order
        }

        step_order=sorted(plot_df["step"].unique())

        if isinstance(p["step_cmap"],dict):
            missing=[s for s in step_order if s not in p["step_cmap"]]
            if missing:
                raise ValueError(
                    f"{dataset_name}: step_cmap missing plotted steps {missing}"
                )

            step_color_map={s:p["step_cmap"][s] for s in step_order}
            common_colors=[step_color_map[s] for s in step_order]

        else:
            cm=plt.get_cmap(p["step_cmap"])
            common_colors=[
                cm(i/max(len(step_order)-1,1))
                for i in range(len(step_order))
            ]
            step_color_map=dict(zip(step_order,common_colors))

        discrete_cmap=ListedColormap(common_colors)

        discrete_norm=BoundaryNorm(
            np.arange(len(step_order)+1)-.5,
            discrete_cmap.N
        )

        cmap_defaults=["Blues","Reds","Greens","Purples","Oranges","Greys"]

        if p["method_cmaps"] is None:
            method_cmaps={
                m:cmap_defaults[i%len(cmap_defaults)]
                for i,m in enumerate(method_order)
            }
        else:
            method_cmaps=dict(p["method_cmaps"])

            for i,m in enumerate(method_order):
                method_cmaps.setdefault(
                    m,
                    cmap_defaults[i%len(cmap_defaults)]
                )

        method_step_colors={}

        for method in method_order:
            cm=plt.get_cmap(method_cmaps[method])

            method_step_colors[method]=dict(
                zip(
                    step_order,
                    [cm(x) for x in np.linspace(.35,.95,len(step_order))]
                )
            )

        def _step_color(method,step):
            if p["method_style"]=="gradient":
                return method_step_colors[method][step]
            return step_color_map[step]

        role_rows={
            r["role"]:r
            for _,r in comparison_df.iterrows()
        }

        def _match_role(role):
            r=role_rows[role]

            method=(
                p["pepcl_method"]
                if role=="pepcl_target"
                else p["ft_method"]
            )

            lr=_parse_lr(r["learning_rate"])
            step=int(r["step"])
            budget=int(r["step_budget"])

            hit=plot_df[
                (plot_df["method"]==method)
                & np.isclose(
                    plot_df["learning_rate_float"],
                    lr,
                    rtol=1e-12,
                    atol=0
                )
                & (plot_df["step"]==step)
                & (plot_df["step_budget"]==budget)
            ]

            if hit.empty:
                raise ValueError(
                    f"{dataset_name}: could not find role '{role}' "
                    f"(method={method}, LR={_lr_label(lr)}, "
                    f"budget={budget}, step={step})."
                )

            return hit.iloc[0]

        pep=_match_role("pepcl_target")
        ft_adapt=_match_role("ft_match_adaptation")
        ft_ms=_match_role("ft_match_ms")

        x_col=f"{p['ms_name']}_delta"
        y_col=f"{p['assay_name']}_delta"

        if p["square_panel"]:
            ax.set_box_aspect(1)

        # Reference lines.
        if p["equal_tradeoff_line"]:
            ax.axline(
                (0,0),slope=-1,
                color=p["equal_tradeoff_linecolor"],
                linewidth=p["equal_tradeoff_linewidth"],
                linestyle=p["equal_tradeoff_linestyle"],
                zorder=0
            )

        if p["zero_lines"]:
            ax.axvline(
                0,color=p["zero_line_color"],
                linestyle=p["zero_line_style"],
                linewidth=p["zero_line_width"],zorder=0
            )

            ax.axhline(
                0,color=p["zero_line_color"],
                linestyle=p["zero_line_style"],
                linewidth=p["zero_line_width"],zorder=0
            )

        filled_edge_color,filled_edge_width=_edge_spec(
            p["background_filled_edge_color"],
            p["background_filled_edge_width"],
            "black",.5
        )

        hollow_edge_color,hollow_edge_width=_edge_spec(
            p["background_hollow_edge_color"],
            p["background_hollow_edge_width"],
            p["background_point_color"],
            p["hollow_linewidth"]
        )

        # Background trajectories.
        for _,g in plot_df.groupby(
            ["method","learning_rate","step_budget"],
            sort=False
        ):
            g=g.sort_values("step")
            method=g["method"].iloc[0]
            lr=g["learning_rate"].iloc[0]
            hollow=method_fill[method]=="hollow"

            ax.plot(
                g[x_col],g[y_col],
                color=p["background_line_color"],
                linewidth=p["line_width"],
                alpha=p["background_line_alpha"],
                zorder=1
            )

            for _,r in g.iterrows():
                if hollow:
                    face="none"
                    edge=hollow_edge_color
                    lw=hollow_edge_width
                else:
                    face=p["background_point_color"]
                    edge=filled_edge_color
                    lw=filled_edge_width

                ax.scatter(
                    r[x_col],r[y_col],
                    facecolors=face,
                    edgecolors=edge,
                    marker=marker_map[lr],
                    s=p["marker_size"]*marker_scale_map[lr],
                    alpha=p["background_point_alpha"],
                    linewidths=lw,
                    zorder=2
                )

        # Restore matched FT points.
        seen=set()

        for r in [ft_adapt,ft_ms]:
            key=(
                r["method"],r["learning_rate"],
                r["step_budget"],r["step"]
            )

            if key in seen: continue
            seen.add(key)

            method=r["method"]
            lr=r["learning_rate"]
            color=_step_color(method,r["step"])
            hollow=method_fill[method]=="hollow"

            ax.scatter(
                r[x_col],r[y_col],
                facecolors="none" if hollow else [color],
                edgecolors=[color] if hollow else "black",
                marker=marker_map[lr],
                s=p["marker_size"]*marker_scale_map[lr],
                alpha=p["matched_point_alpha"],
                linewidths=p["hollow_linewidth"] if hollow else .7,
                zorder=6
            )

        # PepCL target.
        pep_color=_step_color(pep["method"],pep["step"])
        pep_hollow=method_fill[pep["method"]]=="hollow"

        ax.scatter(
            pep[x_col],pep[y_col],
            facecolors="none" if pep_hollow else [pep_color],
            edgecolors=[pep_color] if pep_hollow else "black",
            marker=marker_map[pep["learning_rate"]],
            s=p["marker_size"]*marker_scale_map[pep["learning_rate"]],
            alpha=p["target_point_alpha"],
            linewidths=p["hollow_linewidth"] if pep_hollow else .7,
            zorder=7
        )

        ax.scatter(
            pep[x_col],pep[y_col],
            marker="*",
            s=p["highlight_marker_size"],
            facecolor=pep_color,
            edgecolor="black",
            linewidth=1.2,
            alpha=p["target_point_alpha"],
            zorder=9
        )

        # Optional FT highlight.
        ft_highlight=None

        if p["ft_highlight_target"] is not None:
            target=p["ft_highlight_target"]

            if not isinstance(target,(tuple,list)) or len(target) not in {2,3}:
                raise ValueError(
                    "ft_highlight_target must be (lr, step) "
                    "or (lr, step, step_budget)."
                )

            lr=_parse_lr(target[0])
            step=int(target[1])

            hit=plot_df[
                (plot_df["method"]==p["ft_method"])
                & np.isclose(
                    plot_df["learning_rate_float"],lr,
                    rtol=1e-12,atol=0
                )
                & (plot_df["step"]==step)
            ]

            if len(target)==3:
                hit=hit[hit["step_budget"]==int(target[2])]

            if len(hit)!=1:
                raise ValueError(
                    f"{dataset_name}: FT highlight matched {len(hit)} rows."
                )

            ft_highlight=hit.iloc[0]

            ft_color=(
                _step_color(
                    ft_highlight["method"],
                    ft_highlight["step"]
                )
                if p["ft_highlight_use_step_color"]
                else p["background_point_color"]
            )

            ax.scatter(
                ft_highlight[x_col],
                ft_highlight[y_col],
                marker="*",
                s=p["highlight_marker_size"],
                facecolor=ft_color,
                edgecolor="black",
                linewidth=1.2,
                zorder=9
            )

        if p["xlim"] is not None:
            ax.set_xlim(p["xlim"])

        if p["ylim"] is not None:
            ax.set_ylim(p["ylim"])

        # Labels / axes formatting.
        title=dataset_name if p["title"] is None else p["title"]

        ax.set_title(
            title,
            fontsize=p["title_fontsize"]
        )

        ax.set_xlabel(
            (
                f"{p['ms_name']} ΔmAP vs {p['ref_col']}"
                if p["xlabel"] is None
                else p["xlabel"]
            ),
            fontsize=p["axis_label_fontsize"],
            labelpad=p["axis_label_pad"]
        )

        ax.set_ylabel(
            (
                f"{p['assay_name']} ΔmAP vs {p['ref_col']}"
                if p["ylabel"] is None
                else p["ylabel"]
            ),
            fontsize=p["axis_label_fontsize"],
            labelpad=p["axis_label_pad"]
        )

        if p["n_ticks"] is not None:
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=p["n_ticks"])
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=p["n_ticks"])
            )

        ax.xaxis.set_major_formatter(
            FuncFormatter(_tick_fmt)
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(_tick_fmt)
        )

        ax.tick_params(
            axis="both",
            labelsize=p["tick_fontsize"],
            length=p["tick_length"],
            width=p["tick_width"],
            pad=p["tick_pad"]
        )

        for spine in ax.spines.values():
            spine.set_linewidth(
                p["spine_width"]
            )

        if p["grid"]:
            ax.grid(True,alpha=p["grid_alpha"])
        else:
            ax.grid(False)

        fig.canvas.draw()

        xmin,xmax=ax.get_xlim()
        ymin,ymax=ax.get_ylim()
        xrange=xmax-xmin
        yrange=ymax-ymin

        # Horizontal comparison.
        if p["horizontal_bar"]:
            x1=float(pep[x_col])
            x2=float(ft_adapt[x_col])
            y=float(pep[y_col])

            dx=x2-x1
            direction=np.sign(dx) if dx!=0 else 1

            pad_dist=abs(dx)*float(
                p["horizontal_bar_end_padding"]
            )

            start=x1+direction*(
                p["horizontal_target_clearance"]*xrange
                +pad_dist
            )

            end=x2-direction*(
                p["horizontal_match_clearance"]*xrange
                +pad_dist
            )

            cap=p["horizontal_cap_size"]*yrange

            ax.plot(
                [start,end],[y,y],
                color=p["horizontal_bar_color"],
                linewidth=p["horizontal_bar_linewidth"],
                linestyle=p["horizontal_bar_linestyle"],
                zorder=8
            )

            ax.plot(
                [start,start],[y-cap/2,y+cap/2],
                color=p["horizontal_bar_color"],
                linewidth=p["horizontal_bar_linewidth"],
                zorder=8
            )

            ax.plot(
                [end,end],[y-cap/2,y+cap/2],
                color=p["horizontal_bar_color"],
                linewidth=p["horizontal_bar_linewidth"],
                zorder=8
            )

            delta=float(
                role_rows["ft_match_adaptation"][
                    "ms_difference_vs_pepcl"
                ]
            )

            ax.text(
                (start+end)/2+p["horizontal_delta_xoffset"]*xrange,
                y+p["horizontal_delta_yoffset"]*yrange,
                f"{p['horizontal_delta_prefix']}{_number_label(delta)}",
                color=p["horizontal_delta_color"],
                fontsize=p["horizontal_delta_fontsize"],
                rotation=p["horizontal_delta_rotation"],
                ha=p["horizontal_delta_ha"],
                va=p["horizontal_delta_va"],
                zorder=10
            )

        # Vertical comparison.
        if p["vertical_bar"]:
            x=float(pep[x_col])
            y1=float(pep[y_col])
            y2=float(ft_ms[y_col])

            dy=y2-y1
            direction=np.sign(dy) if dy!=0 else 1

            pad_dist=abs(dy)*float(
                p["vertical_bar_end_padding"]
            )

            start=y1+direction*(
                p["vertical_target_clearance"]*yrange
                +pad_dist
            )

            end=y2-direction*(
                p["vertical_match_clearance"]*yrange
                +pad_dist
            )

            cap=p["vertical_cap_size"]*xrange

            ax.plot(
                [x,x],[start,end],
                color=p["vertical_bar_color"],
                linewidth=p["vertical_bar_linewidth"],
                linestyle=p["vertical_bar_linestyle"],
                zorder=8
            )

            ax.plot(
                [x-cap/2,x+cap/2],[start,start],
                color=p["vertical_bar_color"],
                linewidth=p["vertical_bar_linewidth"],
                zorder=8
            )

            ax.plot(
                [x-cap/2,x+cap/2],[end,end],
                color=p["vertical_bar_color"],
                linewidth=p["vertical_bar_linewidth"],
                zorder=8
            )

            delta=float(
                role_rows["ft_match_ms"][
                    "assay_difference_vs_pepcl"
                ]
            )

            ax.text(
                x+p["vertical_delta_xoffset"]*xrange,
                (start+end)/2+p["vertical_delta_yoffset"]*yrange,
                f"{p['vertical_delta_prefix']}{_number_label(delta)}",
                color=p["vertical_delta_color"],
                fontsize=p["vertical_delta_fontsize"],
                rotation=p["vertical_delta_rotation"],
                ha=p["vertical_delta_ha"],
                va=p["vertical_delta_va"],
                zorder=10
            )

        # Parameter text.
        target_lines=[
            f"{p['pepcl_target_name']}: "
            f"LR={_lr_label(pep['learning_rate_float'])}, "
            f"Steps={int(pep['step'])}"
        ]

        if p["show_matched_ft_text"]:
            target_lines.append(
                f"{p['ft_adaptation_name']}: "
                f"LR={_lr_label(ft_adapt['learning_rate_float'])}, "
                f"Steps={int(ft_adapt['step'])}"
            )

            target_lines.append(
                f"{p['ft_ms_name']}: "
                f"LR={_lr_label(ft_ms['learning_rate_float'])}, "
                f"Steps={int(ft_ms['step'])}"
            )

        if ft_highlight is not None:
            target_lines.insert(
                0,
                f"{p['ft_highlight_name']}: "
                f"LR={_lr_label(ft_highlight['learning_rate_float'])}, "
                f"Steps={int(ft_highlight['step'])}"
            )

        if p["show_target_text"]:
            if p["target_text_position"]=="bottom_left_above_zero":
                tx=xmin+p["target_text_xpad"]*xrange
                base_y=0 if ymin<=0<=ymax else ymin
                ty=base_y+p["target_text_ypad"]*yrange
                ha="left" if p["target_text_ha"] is None else p["target_text_ha"]
                va="bottom" if p["target_text_va"] is None else p["target_text_va"]

            elif p["target_text_position"]=="bottom_right":
                tx=xmax-p["target_text_xpad"]*xrange
                ty=ymin+p["target_text_ypad"]*yrange
                ha="right" if p["target_text_ha"] is None else p["target_text_ha"]
                va="bottom" if p["target_text_va"] is None else p["target_text_va"]

            else:
                raise ValueError(
                    "target_text_position must be "
                    "'bottom_left_above_zero' or 'bottom_right'."
                )

            ax.text(
                tx,ty,"\n".join(target_lines),
                ha=ha,va=va,
                fontsize=p["target_text_fontsize"],
                rotation=p["target_text_rotation"],
                linespacing=p["target_text_linespacing"],
                zorder=10
            )

        # Per-panel colorbar.
        cbar=None

        if p["show_step_colorbar"]:
            sm=plt.cm.ScalarMappable(
                cmap=discrete_cmap,
                norm=discrete_norm
            )

            sm.set_array([])

            cbar=fig.colorbar(
                sm,
                cax=cax,
                ticks=np.arange(len(step_order)),
                spacing="uniform"
            )

            cbar.set_ticks(
                np.arange(len(step_order)),
                labels=[str(s) for s in step_order]
            )

            cbar.set_label(
                p["colorbar_label"],
                fontsize=p["colorbar_label_fontsize"]
            )

            cbar.ax.tick_params(
                labelsize=p["colorbar_tick_fontsize"]
            )

            cbar.outline.set_linewidth(
                p["colorbar_outline_width"]
            )

        else:
            cax.set_visible(False)

        # Per-panel legend if requested.
        if p["legend"]:
            if p["method_legend_labels"] is not None:
                if len(p["method_legend_labels"])!=len(method_order):
                    raise ValueError(
                        f"{dataset_name}: method_legend_labels must contain "
                        f"{len(method_order)} labels."
                    )

                method_label_map=dict(
                    zip(
                        method_order,
                        p["method_legend_labels"]
                    )
                )
            else:
                method_label_map={m:m for m in method_order}

            method_handles=[]

            for method in method_order:
                hollow=method_fill[method]=="hollow"

                method_handles.append(
                    Line2D(
                        [0],[0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=(
                            "none"
                            if hollow
                            else p["method_legend_color"]
                        ),
                        markeredgecolor=(
                            p["method_legend_color"]
                            if hollow
                            else "black"
                        ),
                        markeredgewidth=(
                            p["hollow_linewidth"]
                            if hollow else 1
                        ),
                        markersize=np.sqrt(
                            p["marker_size"]
                        ),
                        label=method_label_map[method]
                    )
                )

            lr_handles=[]

            for lr in lr_order:
                lr_handles.append(
                    Line2D(
                        [0],[0],
                        marker=marker_map[lr],
                        linestyle="none",
                        markerfacecolor="0.6",
                        markeredgecolor="black",
                        markersize=np.sqrt(
                            p["marker_size"]*
                            marker_scale_map[lr]
                        ),
                        label=f"LR = {_lr_label(_parse_lr(lr))}"
                    )
                )

            handles=method_handles+lr_handles

            ax.legend(
                handles=handles,
                loc=p["legend_loc"],
                bbox_to_anchor=p["legend_bbox"],
                ncol=min(
                    len(handles),
                    p["legend_ncol"]
                ),
                frameon=p["legend_frame"],
                fontsize=p["legend_fontsize"],
                columnspacing=p["legend_columnspacing"],
                handletextpad=p["legend_handletextpad"]
            )

        return {
            "plot_df":plot_df,
            "comparison_df":comparison_df,
            "pepcl_target":pep,
            "ft_match_adaptation":ft_adapt,
            "ft_match_ms":ft_ms,
            "ft_highlight":ft_highlight,
            "ax":ax,
            "colorbar":cbar,
        }

    # ------------------------------------------------------------------
    # GRID LAYOUT
    # ------------------------------------------------------------------
    fig=plt.figure(figsize=fig_size,dpi=dpi)

    outer=fig.add_gridspec(
        nrows,ncols,
        wspace=wspace,
        hspace=hspace,
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin
    )

    axes=np.empty((nrows,ncols),dtype=object)
    caxes=np.empty((nrows,ncols),dtype=object)

    assay_items=list(assay_configs.items())
    results={}

    for i in range(nrows*ncols):
        row=i//ncols
        col=i%ncols

        if i>=len(assay_items):
            ax=fig.add_subplot(outer[row,col])
            ax.set_visible(False)
            axes[row,col]=ax
            caxes[row,col]=None
            continue

        dataset_name,cfg=assay_items[i]

        missing=required-set(cfg)
        if missing:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required entries: "
                f"{sorted(missing)}"
            )

        # Resolve panel params first so colorbar geometry can be allocated.
        panel_params=dict(defaults)
        panel_params.update({
            k:v for k,v in plotting_params.items()
            if k not in structural
        })
        panel_params.update({
            k:v for k,v in cfg.items()
            if k not in required and k not in structural
        })

        if panel_params["show_step_colorbar"]:
            inner=outer[row,col].subgridspec(
                1,2,
                width_ratios=[
                    1,
                    panel_params["colorbar_width"]
                ],
                wspace=panel_params["colorbar_pad"]
            )

            ax=fig.add_subplot(inner[0,0])
            cax=fig.add_subplot(inner[0,1])

        else:
            ax=fig.add_subplot(outer[row,col])
            cax=None

        axes[row,col]=ax
        caxes[row,col]=cax

        if verbose:
            print(
                f"Plotting {dataset_name} "
                f"at row {row+1}, column {col+1}..."
            )

        results[dataset_name]=_plot_one(
            ax,cax,cfg,dataset_name
        )

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

    return results,fig,axes



import os, pickle, re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from scipy.stats import t as student_t
except ImportError:
    student_t=None


import os, pickle, re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from scipy.stats import t as student_t
except ImportError:
    student_t=None


def build_loss_trajectory_tables(
    log_dirs, configs, seed_val_dfs, cv_val_dfs, assay_loss_type="logsmoothap", continuous_target_col=None,
    ref_col="mhcprime", label_col="label", allele_col="allele", assay_batch_size=3072, ms_batch_size=3072,
    assay_neg_ratio=1.0, ms_neg_ratio=1.0, assay_alpha=10.0, ms_alpha=10.0, split_seed_offset=4,
    main_seeds=(0,1,2,3,4), cv_splits=(1,2,3,4), include_step0=False, validation_sampling_seed=42000,
    pairwise_chunk_size=2048, device="cpu", strict=True, verbose=True,
):
    """
    Organize training and reconstructed validation-loss trajectories across
    ordinary seeds + CV/data-split replicates.

    log_dirs may be either:

        Single directory set:
            [main_logs, split1_logs, split2_logs, split3_logs, split4_logs]

        Multiple directory sets:
            [
                [main_logs_set1, split1_logs_set1, ..., split4_logs_set1],
                [main_logs_set2, split1_logs_set2, ..., split4_logs_set2],
                ...
            ]

    For each target model/replicate, the corresponding directory position is
    searched across every supplied directory set. This allows different target
    models to live in different logging runs/directories while preserving the
    same replicate identities.

    If the exact same target log is found in more than one candidate directory,
    an error is raised to avoid double-counting the same replicate.
    """

    # ------------------------------------------------------------------
    # Normalize log_dirs into list-of-directory-sets
    # ------------------------------------------------------------------
    if len(log_dirs)==0:
        raise ValueError("log_dirs cannot be empty.")

    if isinstance(log_dirs[0],(str,os.PathLike)):
        log_dir_sets=[list(log_dirs)]
    else:
        log_dir_sets=[list(x) for x in log_dirs]

    expected_n_dirs=1+len(cv_splits)
    for i,dirs in enumerate(log_dir_sets):
        if len(dirs)!=expected_n_dirs:
            raise ValueError(
                f"log_dirs set {i} must contain 1 main seed directory + "
                f"{len(cv_splits)} CV directories ({expected_n_dirs} total); "
                f"got {len(dirs)}."
            )

    if assay_loss_type not in {"logsmoothap","spearman","mse"}:
        raise ValueError("assay_loss_type must be 'logsmoothap', 'spearman', or 'mse'.")
    if assay_loss_type in {"spearman","mse"} and continuous_target_col is None:
        raise ValueError("continuous_target_col is required for continuous validation loss.")

    for label,cfg in configs.items():
        if cfg.get("method") not in {"ft","pepcl"}:
            raise ValueError(f"configs['{label}']['method'] must be 'ft' or 'pepcl'.")
        if "prefix" not in cfg and "log_prefix" not in cfg and "score_prefix" not in cfg:
            raise ValueError(f"Config '{label}' needs 'prefix' (or explicit log_prefix/score_prefix).")

    device=torch.device(device)

    # ------------------------------------------------------------------
    # Find one target log across all equivalent directory sets
    # ------------------------------------------------------------------
    def _find_log(model_name,dir_position):
        filename=f"{model_name}.pkl"
        matches=[]

        for dirs in log_dir_sets:
            path=os.path.join(str(dirs[dir_position]),filename)
            if os.path.isfile(path):
                matches.append(path)

        if len(matches)==1:
            return matches[0]

        if len(matches)>1:
            raise ValueError(
                f"Multiple training logs found for '{model_name}' in the same "
                f"replicate position:\n" + "\n".join(f"  {p}" for p in matches)
            )

        return None

    def _logsmoothap(scores,labels,alpha):
        s=torch.as_tensor(np.asarray(scores,dtype=np.float32),device=device)
        y=torch.as_tensor(np.asarray(labels,dtype=np.float32),device=device)
        if len(s)==0 or float(y.sum())<=0: return np.nan
        P=y.sum().clamp(min=1.0); eps=1e-8; total=torch.tensor(0.,device=device)
        for start in range(0,len(s),pairwise_chunk_size):
            q=s[start:start+pairwise_chunk_size]
            S=torch.sigmoid(alpha*(s.unsqueeze(0)-q.unsqueeze(1)))
            soft_prec=S.matmul(y)/S.sum(1).clamp(min=eps)
            total+=(soft_prec*y[start:start+len(q)]).sum()
        return float(torch.log(1.0/(total/P+eps)).item())

    def _soft_rank(x,tau):
        x=torch.as_tensor(np.asarray(x,dtype=np.float32),device=device); out=[]
        for start in range(0,len(x),pairwise_chunk_size):
            q=x[start:start+pairwise_chunk_size]
            out.append(1.0+torch.sigmoid((x.unsqueeze(0)-q.unsqueeze(1))/tau).sum(-1))
        return torch.cat(out,dim=0)

    def _soft_spearman(pred,target,tau):
        if len(pred)==0: return np.nan
        rp=_soft_rank(pred,tau); rt=_soft_rank(target,1e-4)
        rp=rp-rp.mean(); rt=rt-rt.mean()
        return float((1.0-(rp*rt).sum()/(torch.norm(rp)*torch.norm(rt)+1e-8)).item())

    def _mse(a,b):
        a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
        return float(np.mean((a-b)**2)) if len(a) else np.nan

    def _assay_loss(df,score_col):
        if assay_loss_type=="logsmoothap":
            return _logsmoothap(df[score_col].values,df[label_col].values,assay_alpha)
        if assay_loss_type=="spearman":
            return _soft_spearman(df[score_col].values,df[continuous_target_col].values,assay_alpha)
        return _mse(df[score_col].values,df[continuous_target_col].values)

    def _integer_quota(weights,caps,total):
        weights=np.asarray(weights,float); caps=np.asarray(caps,int); total=int(total)
        q=np.zeros_like(caps,dtype=int)
        valid=caps>0
        if total<=0 or not np.any(valid): return q
        max_total=min(total,int(caps.sum())); w=np.zeros_like(weights,float)
        w[valid]=weights[valid]/weights[valid].sum() if weights[valid].sum()>0 else 1/valid.sum()
        remaining=max_total
        while remaining>0:
            capacity=caps-q; active=capacity>0
            if not np.any(active): break
            wa=np.zeros_like(w)
            wa[active]=w[active]/w[active].sum() if w[active].sum()>0 else 1/active.sum()
            raw=remaining*wa; add=np.minimum(np.floor(raw).astype(int),capacity)
            if add.sum()>0:
                q+=add; remaining-=int(add.sum())
            if remaining<=0: break
            capacity=caps-q; active=capacity>0
            if not np.any(active): break
            frac=raw-np.floor(raw); frac[~active]=-1; added=False
            for i in np.argsort(-frac):
                if remaining<=0: break
                if capacity[i]<=0: continue
                q[i]+=1; remaining-=1; added=True
            if not added: break
        return q

    def _allele_quotas(df,batch_size,neg_ratio):
        counts=df.groupby([allele_col,label_col],observed=True).size().unstack(fill_value=0)
        pos=counts[1] if 1 in counts.columns else pd.Series(0,index=counts.index)
        neg=counts[0] if 0 in counts.columns else pd.Series(0,index=counts.index)
        d=pd.DataFrame({
            "allele":counts.index,
            "n_pos":pos.astype(int).values,
            "n_neg":neg.astype(int).values,
        })
        d=d[d.n_pos>0].reset_index(drop=True)
        if d.empty: return {}
        A=len(d); P=d.n_pos.to_numpy(float); target=max(1,int(batch_size/(1+float(neg_ratio))))
        uniform=np.ones(A)/A; prevalence=P/P.sum(); denom=np.square(P).sum()
        A_eff=float(A) if denom<=0 else float((P.sum()**2)/denom)
        balance=float(np.clip(A_eff/A,0,1))
        weights=balance*uniform+(1-balance)*prevalence
        weights/=weights.sum()
        q=_integer_quota(weights,d.n_pos.to_numpy(int),target)
        return {a:int(n) for a,n in zip(d.allele.tolist(),q) if int(n)>0}

    def _sample_allele_aware(df,batch_size,neg_ratio,rng):
        quotas=_allele_quotas(df,batch_size,neg_ratio); idx=[]
        for allele,q in quotas.items():
            g=df[df[allele_col]==allele]; labels=g[label_col].astype(int)
            pos=g.index[labels==1].to_numpy(); neg=g.index[labels==0].to_numpy()
            if len(pos)==0: continue
            p_take=min(len(pos),q)
            p=rng.choice(pos,p_take,replace=False)
            idx.extend(p)
            n_req=int(p_take*neg_ratio)
            if n_req>0 and len(neg)>0:
                idx.extend(rng.choice(neg,min(len(neg),n_req),replace=False))
        idx=np.asarray(idx,dtype=int)
        rng.shuffle(idx)
        return idx

    def _sample_meta(df,idx,dataset):
        g=df.loc[idx]; labels=g[label_col].astype(int)
        return {
            "dataset":dataset,
            "n_sampled":len(g),
            "n_pos":int((labels==1).sum()),
            "n_neg":int((labels==0).sum()),
            "n_alleles":int(g[allele_col].nunique()),
        }

    def _extract_loss_dict(path):
        with open(path,"rb") as f:
            log=pickle.load(f)

        candidates=[]
        if isinstance(log,dict):
            for key in ["training_stats","metrics","losses","training_metrics"]:
                if isinstance(log.get(key),dict):
                    candidates.append(log[key])
            candidates.append(log)

        for d in candidates:
            if any(k in d for k in [
                "ft_loss_raw","ms_classification_loss_raw",
                "drift_loss_raw","continuous_loss_raw",
            ]):
                return d

        available=sorted(log.keys()) if isinstance(log,dict) else []
        raise KeyError(
            f"Explicit raw training-loss keys not found in {path}. "
            f"Top-level keys: {available}"
        )

    def _summary(df):
        if df.empty:
            return df.copy()

        group_cols=["config","method","step","loss_name"]
        rows=[]

        for keys,g in df.groupby(group_cols,sort=True,dropna=False):
            x=g["loss"].dropna().astype(float).to_numpy()
            n=len(x)
            mean=float(np.mean(x)) if n else np.nan
            sd=float(np.std(x,ddof=1)) if n>1 else np.nan
            se=sd/np.sqrt(n) if n>1 else np.nan
            crit=float(student_t.ppf(0.975,n-1)) if n>1 and student_t is not None else (1.96 if n>1 else np.nan)
            half=crit*se if n>1 else np.nan

            rows.append(dict(
                zip(group_cols,keys),
                n=n,
                mean=mean,
                sd=sd,
                se=se,
                ci95_low=mean-half if n>1 else np.nan,
                ci95_high=mean+half if n>1 else np.nan,
                ci95_halfwidth=half,
            ))

        return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Replicate specification
    #
    # dir_position points to the corresponding folder in EVERY log-dir set.
    # main seeds -> position 0
    # CV split 1 -> position 1, etc.
    # ------------------------------------------------------------------
    reps=[]

    for seed in main_seeds:
        reps.append({
            "replicate":int(seed),
            "source":"seed",
            "datasplit":np.nan,
            "dir_position":0,
            "log_seed":int(seed),
        })

    for i,ds in enumerate(cv_splits,start=1):
        reps.append({
            "replicate":int(ds+split_seed_offset),
            "source":"cv",
            "datasplit":int(ds),
            "dir_position":i,
            "log_seed":0,
        })

    # ------------------------------------------------------------------
    # Training trajectories
    # ------------------------------------------------------------------
    train_rows=[]

    for cfg_label,cfg in configs.items():
        method=cfg["method"]
        prefix=cfg.get("log_prefix",cfg.get("prefix"))
        dr_lambda=float(cfg.get("dr_lambda",1e-5 if method=="pepcl" else 0.0))

        if verbose:
            print(f"\nSearching training logs for: {cfg_label}")

        for rep in reps:
            model_name=f"{prefix}_seed_{rep['log_seed']}"
            path=_find_log(model_name,rep["dir_position"])

            if path is None:
                candidate_dirs=[
                    str(dirs[rep["dir_position"]])
                    for dirs in log_dir_sets
                ]

                msg=(
                    f"Missing training log '{model_name}.pkl'. "
                    f"Searched:\n" +
                    "\n".join(f"  {d}" for d in candidate_dirs)
                )

                if strict:
                    raise FileNotFoundError(msg)
                if verbose:
                    print("Warning:",msg)
                continue

            if verbose:
                print(f"  replicate {rep['replicate']}: {path}")

            loss_dict=_extract_loss_dict(path)

            required=["ft_loss_raw"]+(
                ["ms_classification_loss_raw","drift_loss_raw"]
                if method=="pepcl" else []
            )

            missing=[k for k in required if k not in loss_dict]
            if missing:
                msg=f"{path} missing required explicit raw losses: {missing}"
                if strict:
                    raise KeyError(msg)
                if verbose:
                    print("Warning:",msg)
                continue

            n_steps=len(loss_dict["ft_loss_raw"])

            for j in range(n_steps):
                base={
                    "config":cfg_label,
                    "method":method,
                    "replicate":rep["replicate"],
                    "source":rep["source"],
                    "datasplit":rep["datasplit"],
                    "step":j+1,
                    "log_path":path,
                }

                assay=float(loss_dict["ft_loss_raw"][j])
                train_rows.append({
                    **base,
                    "loss_name":"assay_loss_raw",
                    "loss":assay,
                })

                if method=="pepcl":
                    ms=float(loss_dict["ms_classification_loss_raw"][j])
                    dr=float(loss_dict["drift_loss_raw"][j])
                    drw=dr*dr_lambda

                    train_rows.extend([
                        {**base,"loss_name":"ms_replay_loss_raw","loss":ms},
                        {**base,"loss_name":"drift_loss_raw","loss":dr},
                        {**base,"loss_name":"drift_loss_weighted","loss":drw},
                        {**base,"loss_name":"total_loss","loss":assay+ms+drw},
                    ])
                else:
                    train_rows.append({
                        **base,
                        "loss_name":"total_loss",
                        "loss":assay,
                    })

    training_replicates=pd.DataFrame(train_rows)
    training_summary=_summary(training_replicates)

    # ------------------------------------------------------------------
    # Deterministic validation sampling
    # ------------------------------------------------------------------
    sampled={}
    sample_rows=[]

    for rep in reps:
        rid=rep["replicate"]

        if rep["source"]=="seed":
            val_pair=seed_val_dfs
        else:
            if rep["datasplit"] not in cv_val_dfs:
                raise KeyError(f"cv_val_dfs missing split {rep['datasplit']}.")
            val_pair=cv_val_dfs[rep["datasplit"]]

        if "assay" not in val_pair:
            raise KeyError(f"Validation inputs for replicate {rid} are missing 'assay'.")

        assay_df=val_pair["assay"]

        if assay_loss_type in {"spearman","mse"} and continuous_target_col not in assay_df.columns:
            raise KeyError(
                f"'{continuous_target_col}' missing from assay validation "
                f"dataframe for replicate {rid}."
            )

        ai=_sample_allele_aware(
            assay_df,
            assay_batch_size,
            assay_neg_ratio,
            np.random.default_rng(validation_sampling_seed+rid),
        )

        sampled[(rid,"assay")]=(assay_df,ai)
        sample_rows.append({
            "replicate":rid,
            "source":rep["source"],
            "datasplit":rep["datasplit"],
            **_sample_meta(assay_df,ai,"assay"),
        })

        if "ms" in val_pair and val_pair["ms"] is not None:
            ms_df=val_pair["ms"]

            mi=_sample_allele_aware(
                ms_df,
                ms_batch_size,
                ms_neg_ratio,
                np.random.default_rng(validation_sampling_seed+10000+rid),
            )

            sampled[(rid,"ms")]=(ms_df,mi)
            sample_rows.append({
                "replicate":rid,
                "source":rep["source"],
                "datasplit":rep["datasplit"],
                **_sample_meta(ms_df,mi,"ms"),
            })

    validation_samples=pd.DataFrame(sample_rows)

    # ------------------------------------------------------------------
    # Reconstructed validation trajectories
    # ------------------------------------------------------------------
    val_rows=[]

    for cfg_label,cfg in configs.items():
        method=cfg["method"]
        prefix=cfg.get("score_prefix",cfg.get("prefix"))
        dr_lambda=float(cfg.get("dr_lambda",1e-5 if method=="pepcl" else 0.0))

        for rep in tqdm(reps,desc=f"Validation losses: {cfg_label}",leave=True):
            rid=rep["replicate"]
            assay_df,ai=sampled[(rid,"assay")]
            assay_sample=assay_df.loc[ai]

            score_base=f"{prefix}_seed_{rid}"
            pattern=re.compile(rf"^{re.escape(score_base)}_step_(\d+)$")
            found={int(m.group(1)):c for c in assay_df.columns if (m:=pattern.match(str(c)))}

            if method=="pepcl":
                if (rid,"ms") not in sampled:
                    msg=f"PepCL config '{cfg_label}' needs MS validation data for replicate {rid}."
                    if strict:
                        raise KeyError(msg)
                    if verbose:
                        print("Warning:",msg)
                    continue

                ms_df,mi=sampled[(rid,"ms")]
                found={s:c for s,c in found.items() if c in ms_df.columns}

            if include_step0:
                found={0:ref_col,**found}

            if not found:
                msg=f"No checkpoint columns found for '{score_base}_step_N' in replicate {rid}."
                if strict:
                    raise ValueError(msg)
                if verbose:
                    print("Warning:",msg)
                continue

            for step,col in sorted(found.items()):
                base={
                    "config":cfg_label,
                    "method":method,
                    "replicate":rid,
                    "source":rep["source"],
                    "datasplit":rep["datasplit"],
                    "step":int(step),
                    "score_col":col,
                }

                assay=_assay_loss(assay_sample,col)
                val_rows.append({
                    **base,
                    "loss_name":"assay_loss_raw",
                    "loss":assay,
                })

                if method=="pepcl":
                    ms_sample=ms_df.loc[mi]
                    ms=_logsmoothap(
                        ms_sample[col].values,
                        ms_sample[label_col].values,
                        ms_alpha,
                    )
                    dr=_mse(
                        ms_sample[col].values,
                        ms_sample[ref_col].values,
                    )
                    drw=dr*dr_lambda

                    val_rows.extend([
                        {**base,"loss_name":"ms_replay_loss_raw","loss":ms},
                        {**base,"loss_name":"drift_loss_raw","loss":dr},
                        {**base,"loss_name":"drift_loss_weighted","loss":drw},
                        {**base,"loss_name":"total_loss","loss":assay+ms+drw},
                    ])
                else:
                    val_rows.append({
                        **base,
                        "loss_name":"total_loss",
                        "loss":assay,
                    })

    validation_replicates=pd.DataFrame(val_rows)
    validation_summary=_summary(validation_replicates)

    if verbose:
        print(
            f"\nTraining: "
            f"{training_replicates['replicate'].nunique() if not training_replicates.empty else 0} "
            f"replicates, {len(training_replicates):,} rows"
        )
        print(
            f"Validation: "
            f"{validation_replicates['replicate'].nunique() if not validation_replicates.empty else 0} "
            f"replicates, {len(validation_replicates):,} rows"
        )
        print("Validation sampling summary:")
        print(
            validation_samples.groupby("dataset")[
                ["n_sampled","n_pos","n_neg"]
            ].agg(["min","median","max"])
        )

    return {
        "training_replicates":training_replicates,
        "training_summary":training_summary,
        "validation_replicates":validation_replicates,
        "validation_summary":validation_summary,
        "validation_samples":validation_samples,
    }

import re
import numpy as np
import pandas as pd
from scipy.stats import t


def summarize_ft_pepcl_validation_sweep(
    ms_val_df,
    assay_val_df,
    score_cols,
    method_prefixes,
    ft_early_stopping_config=None,
    ref_col="mhcprime",
    ft_method="ft",
    pepcl_method="pepcl_r_dr",
    method_labels=None,
    focus_seeds=None,
    ci_level=0.95,
    require_complete_replicates=True,
    verbose=True,
):
    """
    Summarize the complete FT/PepCL validation hyperparameter sweep without
    performing any model selection.

    Intended for merged early-stopping validation data, where:
      - PepCL unsuffixed model columns contain the full-budget endpoint.
      - FT unsuffixed model columns may contain the performance of a
        retroactively selected early-stopping checkpoint.
      - ft_early_stopping_config records which FT LR x budget combinations
        stopped early and the checkpoint represented by the merged FT column.

    Only UNSUFFIXED model columns are summarized:
        ..._steps_{budget}_lr_{lr}_seed_{replicate}

    Explicit intermediate columns:
        ..._step_{checkpoint}

    are ignored. This is intentional because the merged unsuffixed FT column
    is treated as the final reported FT configuration for that LR x budget.

    Statistics
    ----------
    For each model configuration and replicate:
        1. Average across evaluable allele rows to obtain replicate-level
           assay performance.
        2. Average across evaluable allele rows to obtain replicate-level
           matched-MS performance.

    Across requested replicates:
        - mean
        - sample SD
        - SE
        - two-sided Student-t confidence interval

    Replicate IDs may represent either independent training seeds or additional
    validation splits. The function treats the supplied IDs as evaluation
    replicates and does not reinterpret their origin.

    Parameters
    ----------
    ms_val_df, assay_val_df : pd.DataFrame
        Per-allele/group validation metric dataframes. Model score columns
        contain allele-level AP, Spearman, or the corresponding metric.

    score_cols : iterable[str]
        Candidate model columns. May include checkpoint columns; only
        unsuffixed FT/PepCL columns are summarized.

    method_prefixes : dict
        Example:
            {
                "ft":"holec_ft_",
                "pepcl_r_dr":"holec_pepcl_r_dr_",
            }

    ft_early_stopping_config : dict or None
        Output from early_stopping_df_to_dict(), e.g.
            {
                0: {"lr":2e-4, "steps":300, "stop":150},
                ...
            }

        Only true early-stopping configurations need to be present.
        FT configurations absent from this dictionary are assigned their
        full step budget as checkpoint.

    focus_seeds : iterable[int] or None
        Exact replicate IDs to summarize. Examples:

            Primary 5 training seeds:
                [0,1,2,3,4]

            Primary split under seed 0 + 4 additional validation splits:
                [0,5,6,7,8]

        If supplied, every retained LR x budget x method configuration must
        contain exactly this replicate set.

    require_complete_replicates : bool
        When focus_seeds=None, require all configurations within each method
        to share the same replicate IDs.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per method x LR x configured step budget.

    replicate_df : pd.DataFrame
        One row per method x LR x budget x replicate before aggregation.
    """

    # ================================================================
    # 0. Validation / setup
    # ================================================================
    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty dictionary.")

    for method in (ft_method,pepcl_method):
        if method not in method_prefixes:
            raise ValueError(
                f"'{method}' is missing from method_prefixes."
            )

    if not (0<ci_level<1):
        raise ValueError("ci_level must be in (0,1).")

    if method_labels is None:
        method_labels={
            ft_method:"FT",
            pepcl_method:"PepCL",
        }
    else:
        method_labels=dict(method_labels)

    if focus_seeds is not None:
        if np.isscalar(focus_seeds):
            focus_seeds=[focus_seeds]

        focus_seeds=tuple(
            sorted(
                dict.fromkeys(
                    int(x) for x in focus_seeds
                )
            )
        )

        if not focus_seeds:
            raise ValueError(
                "focus_seeds cannot be empty."
            )

    for df,name in [
        (ms_val_df,"ms_val_df"),
        (assay_val_df,"assay_val_df"),
    ]:
        if not df.columns.is_unique:
            dup=df.columns[
                df.columns.duplicated()
            ].tolist()

            raise ValueError(
                f"{name} contains duplicate columns: {dup[:10]}"
            )

    if (
        ref_col not in assay_val_df.columns
        or ref_col not in ms_val_df.columns
    ):
        raise ValueError(
            f"Reference column '{ref_col}' must exist in both dataframes."
        )

    score_cols=list(
        dict.fromkeys(score_cols)
    )

    # ================================================================
    # 1. Common evaluable allele/group sets
    # ================================================================
    def _validate_mask(df,cols,name):
        base=df[ref_col].notna()

        inconsistent=[
            c for c in cols
            if c in df.columns
            and not df[c].notna().equals(base)
        ]

        if inconsistent:
            raise ValueError(
                f"{name}: model columns do not share the evaluable row set "
                f"of '{ref_col}'. Examples: {inconsistent[:10]}"
            )

        return base

    assay_present=[
        c for c in score_cols
        if c!=ref_col and c in assay_val_df.columns
    ]

    ms_present=[
        c for c in score_cols
        if c!=ref_col and c in ms_val_df.columns
    ]

    assay_mask=_validate_mask(
        assay_val_df,
        assay_present,
        "Assay",
    )

    ms_mask=_validate_mask(
        ms_val_df,
        ms_present,
        "MS",
    )

    assay_metric=assay_val_df.loc[
        assay_mask
    ].mean(axis=0)

    ms_metric=ms_val_df.loc[
        ms_mask
    ].mean(axis=0)

    # ================================================================
    # 2. Method / model parser
    # ================================================================
    ordered_methods=sorted(
        method_prefixes.items(),
        key=lambda x:len(x[1]),
        reverse=True,
    )

    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix):
                return name
        return None

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_"
        r"lr_(?P<lr>.+?)_"
        r"seed_(?P<seed>\d+)"
        r"(?:_step_(?P<checkpoint>\d+))?$"
    )

    def _lr_float(x):
        if isinstance(
            x,
            (int,float,np.integer,np.floating),
        ):
            return float(x)

        s=str(x)

        if "eN" in s:
            a,b=s.split("eN",1)
            return float(a)*10**(-int(b))

        return float(s)

    # ================================================================
    # 3. FT early-stopping lookup
    # ================================================================
    early_stop_entries=[]

    if ft_early_stopping_config is not None:
        for config_id,cfg in ft_early_stopping_config.items():

            missing=[
                k for k in ("lr","steps","stop")
                if k not in cfg
            ]

            if missing:
                raise ValueError(
                    f"FT early-stopping config '{config_id}' is missing "
                    f"required entries: {missing}"
                )

            early_stop_entries.append({
                "config_id":config_id,
                "learning_rate":_lr_float(
                    cfg["lr"]
                ),
                "step_budget":int(
                    cfg["steps"]
                ),
                "checkpoint":int(
                    cfg["stop"]
                ),
            })

    def _ft_checkpoint(lr,budget):
        matches=[
            x for x in early_stop_entries
            if x["step_budget"]==int(budget)
            and np.isclose(
                x["learning_rate"],
                float(lr),
                rtol=1e-12,
                atol=0,
            )
        ]

        if len(matches)>1:
            raise ValueError(
                f"Multiple FT early-stopping entries found for "
                f"LR={lr}, steps={budget}."
            )

        if not matches:
            return int(budget),False

        checkpoint=int(
            matches[0]["checkpoint"]
        )

        if checkpoint>int(budget):
            raise ValueError(
                f"FT early-stopping checkpoint {checkpoint} exceeds "
                f"step budget {budget} for LR={lr}."
            )

        return checkpoint,checkpoint<int(budget)

    # ================================================================
    # 4. Parse UNSUFFIXED FT/PepCL models
    # ================================================================
    rows=[]

    for col in score_cols:
        if col==ref_col:
            continue

        if (
            col not in assay_val_df.columns
            or col not in ms_val_df.columns
        ):
            continue

        method=_method(col)
        m=pattern.search(str(col))

        if method not in {
            ft_method,
            pepcl_method,
        } or m is None:
            continue

        # Explicit checkpoint models are intentionally excluded.
        if m["checkpoint"] is not None:
            continue

        step_budget=int(
            m["budget"]
        )

        seed=int(
            m["seed"]
        )

        if (
            focus_seeds is not None
            and seed not in focus_seeds
        ):
            continue

        lr_string=m["lr"]
        lr_float=_lr_float(
            lr_string
        )

        if method==ft_method:
            checkpoint,early_stopped=_ft_checkpoint(
                lr_float,
                step_budget,
            )

        else:
            checkpoint=step_budget
            early_stopped=False

        # Canonical model/configuration name with replicate suffix removed.
        model_name=re.sub(
            r"_seed_\d+$",
            "",
            str(col),
        )

        rows.append({
            "score_col":col,
            "model_name":model_name,
            "method_key":method,
            "method":method_labels.get(
                method,
                method,
            ),
            "steps":step_budget,
            "learning_rate":lr_float,
            "learning_rate_string":lr_string,
            "checkpoint":checkpoint,
            "early_stopped":early_stopped,
            "replicate":seed,
            "assay_performance":float(
                assay_metric[col]
            ),
            "ms_performance":float(
                ms_metric[col]
            ),
        })

    replicate_df=pd.DataFrame(
        rows
    )

    if replicate_df.empty:
        raise ValueError(
            "No unsuffixed FT/PepCL model columns could be parsed in both "
            "assay_val_df and ms_val_df."
        )

    # ================================================================
    # 5. Validate exactly one model per replicate/configuration
    # ================================================================
    config_keys=[
        "method_key",
        "steps",
        "learning_rate_string",
    ]

    dup=replicate_df.duplicated(
        config_keys+["replicate"],
        keep=False,
    )

    if dup.any():
        raise ValueError(
            "Duplicate endpoint models were found for the same "
            "method/LR/budget/replicate:\n"
            +replicate_df.loc[
                dup,
                config_keys+[
                    "replicate",
                    "score_col",
                ],
            ].to_string(index=False)
        )

    # ================================================================
    # 6. Replicate eligibility
    # ================================================================
    replicate_info=(
        replicate_df
        .groupby(
            config_keys,
            sort=False,
        )["replicate"]
        .agg(
            replicate_ids=lambda x:tuple(
                sorted(set(x))
            ),
            n_replicates="nunique",
        )
        .reset_index()
    )

    if focus_seeds is not None:
        required=tuple(
            focus_seeds
        )

        bad=replicate_info[
            replicate_info["replicate_ids"].map(
                lambda x:x!=required
            )
        ]

        if not bad.empty:
            raise ValueError(
                "Some configurations do not contain exactly the requested "
                f"replicates {required}:\n"
                +bad.to_string(index=False)
            )

    elif require_complete_replicates:
        for method in (
            ft_method,
            pepcl_method,
        ):
            sub=replicate_info[
                replicate_info["method_key"]==method
            ]

            replicate_sets=(
                sub["replicate_ids"]
                .drop_duplicates()
                .tolist()
            )

            if len(replicate_sets)>1:
                raise ValueError(
                    f"Method '{method}' has inconsistent replicate sets "
                    f"across configurations: {replicate_sets}. "
                    "Provide focus_seeds explicitly or set "
                    "require_complete_replicates=False."
                )

    # ================================================================
    # 7. Student-t summary helper
    # ================================================================
    def _summary(values):
        x=np.asarray(
            values,
            dtype=float,
        )

        x=x[
            np.isfinite(x)
        ]

        n=len(x)

        if n==0:
            return {
                "mean":np.nan,
                "sd":np.nan,
                "se":np.nan,
                "ci_low":np.nan,
                "ci_high":np.nan,
                "n":0,
            }

        mean=float(
            np.mean(x)
        )

        if n<2:
            return {
                "mean":mean,
                "sd":np.nan,
                "se":np.nan,
                "ci_low":np.nan,
                "ci_high":np.nan,
                "n":n,
            }

        sd=float(
            np.std(
                x,
                ddof=1,
            )
        )

        se=sd/np.sqrt(n)

        crit=float(
            t.ppf(
                (1+ci_level)/2,
                n-1,
            )
        )

        return {
            "mean":mean,
            "sd":sd,
            "se":se,
            "ci_low":mean-crit*se,
            "ci_high":mean+crit*se,
            "n":n,
        }

    # ================================================================
    # 8. Aggregate configurations
    # ================================================================
    summary_rows=[]

    for keys,g in replicate_df.groupby(
        config_keys,
        sort=False,
    ):
        g=g.sort_values(
            "replicate"
        )

        assay_stats=_summary(
            g["assay_performance"]
        )

        ms_stats=_summary(
            g["ms_performance"]
        )

        first=g.iloc[0]

        # These should be invariant within a grouped configuration.
        for invariant in [
            "model_name",
            "method",
            "checkpoint",
            "early_stopped",
            "learning_rate",
        ]:
            if g[invariant].nunique(dropna=False)!=1:
                raise ValueError(
                    f"Configuration {keys} has inconsistent '{invariant}' "
                    "values across replicates."
                )

        summary_rows.append({
            # --------------------------------------------------------
            # Informational columns first.
            # --------------------------------------------------------
            "model_name":first["model_name"],
            "method":first["method"],
            "steps":int(first["steps"]),
            "learning_rate":float(
                first["learning_rate"]
            ),
            "learning_rate_string":
                first["learning_rate_string"],
            "checkpoint":int(
                first["checkpoint"]
            ),
            "early_stopped":bool(
                first["early_stopped"]
            ),

            # --------------------------------------------------------
            # Assay.
            # --------------------------------------------------------
            "assay_performance_mean":
                assay_stats["mean"],
            "assay_performance_sd":
                assay_stats["sd"],
            "assay_performance_se":
                assay_stats["se"],
            "assay_performance_ci_low":
                assay_stats["ci_low"],
            "assay_performance_ci_high":
                assay_stats["ci_high"],

            # --------------------------------------------------------
            # MS.
            # --------------------------------------------------------
            "ms_performance_mean":
                ms_stats["mean"],
            "ms_performance_sd":
                ms_stats["sd"],
            "ms_performance_se":
                ms_stats["se"],
            "ms_performance_ci_low":
                ms_stats["ci_low"],
            "ms_performance_ci_high":
                ms_stats["ci_high"],

            # --------------------------------------------------------
            # Replicate bookkeeping.
            # --------------------------------------------------------
            "n_replicates":
                assay_stats["n"],

            "replicate_ids":
                tuple(g["replicate"]),

            "assay_performance_by_replicate":
                dict(
                    zip(
                        g["replicate"],
                        g["assay_performance"],
                    )
                ),

            "ms_performance_by_replicate":
                dict(
                    zip(
                        g["replicate"],
                        g["ms_performance"],
                    )
                ),

            "score_cols_by_replicate":
                dict(
                    zip(
                        g["replicate"],
                        g["score_col"],
                    )
                ),
        })

    summary_df=pd.DataFrame(
        summary_rows
    )

    # ================================================================
    # 9. Stable output ordering
    # ================================================================
    method_order={
        pepcl_method:0,
        ft_method:1,
    }

    summary_df["_method_order"]=summary_df["method"].map({
        method_labels.get(pepcl_method,pepcl_method):0,
        method_labels.get(ft_method,ft_method):1,
    }).fillna(99)

    summary_df=(
        summary_df
        .sort_values(
            [
                "_method_order",
                "steps",
                "learning_rate",
            ]
        )
        .drop(
            columns="_method_order"
        )
        .reset_index(
            drop=True
        )
    )

    replicate_df["_method_order"]=replicate_df["method_key"].map(
        method_order
    ).fillna(99)

    replicate_df=(
        replicate_df
        .sort_values(
            [
                "_method_order",
                "steps",
                "learning_rate",
                "replicate",
            ]
        )
        .drop(
            columns="_method_order"
        )
        .reset_index(
            drop=True
        )
    )

    # ================================================================
    # 10. Logging
    # ================================================================
    if verbose:
        print(
            f"\n{'='*110}\n"
            "FT / PEPCL VALIDATION HYPERPARAMETER SWEEP SUMMARY\n"
            f"{'='*110}"
        )

        if focus_seeds is not None:
            print(
                f"Replicates: {focus_seeds}"
            )
        else:
            print(
                "Replicates: all eligible replicates"
            )

        print(
            f"CI: {ci_level:.1%} Student-t"
        )

        print(
            f"Configurations: {len(summary_df)}"
        )

        print()

        print(
            summary_df[
                [
                    "method",
                    "steps",
                    "learning_rate",
                    "checkpoint",
                    "early_stopped",
                    "assay_performance_mean",
                    "assay_performance_se",
                    "ms_performance_mean",
                    "ms_performance_se",
                    "n_replicates",
                ]
            ].to_string(
                index=False
            )
        )

    return summary_df,replicate_df


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
def plot_selected_ft_pepcl_table(
    selected_dfs,
    dataset_order=None,
    dataset_labels=None,
    ft_method="ft",
    pepcl_method="pepcl_r_dr",

    # Labels.
    assay_col_label="Assay",
    method_col_label="Update",
    config_col_label="Configuration",
    assay_perf_label="Assay performance [95% CI]",
    ms_perf_label="MS performance [95% CI]",
    assay_delta_label="ΔAssay [95% CI]",
    ms_delta_label="ΔMS [95% CI]",

    ft_label="FT",
    pepcl_label="PepCL",

    # Formatting.
    decimals=3,
    show_plus=True,
    ci_brackets=(" [",", ","]"),
    config_format="LR: {lr}, Steps: {steps}",
    ci_level=0.95,

    # Figure / fonts.
    fig_size=None,
    dpi=300,
    title="Validation-selected FT and PepCL configurations",
    title_fontsize=16,
    header_fontsize=11,
    body_fontsize=11,
    font_family="Nimbus Sans",

    # Geometry.
    table_bbox=(0.0,0.02,1.0,0.92),
    row_height_scale=1.45,
    col_widths=None,
    assay_align="left",
    method_align="left",
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

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot validation-selected PepCL and FT configurations with 95% CIs.

    Performance columns:
        replicate mean [Student-t 95% CI]

    Delta columns:
        paired replicate differences.

        PepCL row:
            PepCL - FT

        FT row:
            FT - PepCL

    Thus the two delta rows are exact sign reversals.

    Table rules are independent Line2D objects by default.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import t
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    if not isinstance(selected_dfs,dict) or not selected_dfs:
        raise ValueError("selected_dfs must be a non-empty dictionary.")

    if not (0<ci_level<1):
        raise ValueError("ci_level must be in (0,1).")

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)

    if dataset_order is None:
        dataset_order=list(selected_dfs)
    else:
        missing=[d for d in dataset_order if d not in selected_dfs]
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
            return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

        except (TypeError,ValueError):
            return s

    def _clean_number(v,signed=False):
        if not np.isfinite(v):
            return ""

        rounded=round(float(v),decimals)

        if rounded==0:
            return "0.0"

        s=f"{abs(float(v)):.{decimals}f}"

        if signed and show_plus:
            return ("+" if v>0 else "-")+s

        if signed:
            return f"{float(v):.{decimals}f}"

        return s

    def _fmt_ci(mean,lo,hi,signed=False):
        if not all(np.isfinite(v) for v in [mean,lo,hi]):
            return _clean_number(mean,signed=signed)

        return (
            _clean_number(mean,signed=signed)
            +ci_brackets[0]
            +_clean_number(lo,signed=signed)
            +ci_brackets[1]
            +_clean_number(hi,signed=signed)
            +ci_brackets[2]
        )

    def _paired_summary(a_dict,b_dict):
        """
        Summarize paired differences a - b over matched replicate IDs.
        """
        if not isinstance(a_dict,dict) or not isinstance(b_dict,dict):
            raise ValueError(
                "Selected rows must contain replicate-level dictionaries."
            )

        common=sorted(set(a_dict) & set(b_dict))

        if not common:
            raise ValueError(
                "No matched replicate IDs are shared between selected methods."
            )

        diff=np.asarray(
            [
                float(a_dict[s])-float(b_dict[s])
                for s in common
            ],
            dtype=float
        )

        diff=diff[np.isfinite(diff)]
        n=len(diff)

        if n==0:
            return {
                "mean":np.nan,
                "ci_low":np.nan,
                "ci_high":np.nan,
                "n":0,
                "replicate_ids":tuple(),
            }

        mean=float(np.mean(diff))

        if n<2:
            return {
                "mean":mean,
                "ci_low":np.nan,
                "ci_high":np.nan,
                "n":n,
                "replicate_ids":tuple(common),
            }

        sd=float(np.std(diff,ddof=1))
        se=sd/np.sqrt(n)
        crit=float(t.ppf((1+ci_level)/2,n-1))

        return {
            "mean":mean,
            "ci_low":mean-crit*se,
            "ci_high":mean+crit*se,
            "n":n,
            "replicate_ids":tuple(common),
        }

    rows=[]

    for dataset in dataset_order:
        d=selected_dfs[dataset].copy()

        required=[
            "method",
            "learning_rate",
            "checkpoint",

            "assay_performance_mean",
            "assay_performance_ci_low",
            "assay_performance_ci_high",

            "ms_performance_mean",
            "ms_performance_ci_low",
            "ms_performance_ci_high",

            "assay_performance_by_seed",
            "ms_performance_by_seed",
        ]

        missing=[c for c in required if c not in d.columns]

        if missing:
            raise ValueError(
                f"{dataset}: selected_df missing columns {missing}."
            )

        pep=d[d["method"]==pepcl_method]
        ft=d[d["method"]==ft_method]

        if len(pep)!=1:
            raise ValueError(
                f"{dataset}: expected exactly one PepCL row, found {len(pep)}."
            )

        if len(ft)!=1:
            raise ValueError(
                f"{dataset}: expected exactly one FT row, found {len(ft)}."
            )

        pep=pep.iloc[0]
        ft=ft.iloc[0]

        # ----------------------------------------------------------
        # Paired differences: PepCL - FT
        # ----------------------------------------------------------
        assay_delta=_paired_summary(
            pep["assay_performance_by_seed"],
            ft["assay_performance_by_seed"],
        )

        ms_delta=_paired_summary(
            pep["ms_performance_by_seed"],
            ft["ms_performance_by_seed"],
        )

        # FT - PepCL is the exact sign reversal.
        ft_assay_delta={
            "mean":-assay_delta["mean"],
            "ci_low":-assay_delta["ci_high"],
            "ci_high":-assay_delta["ci_low"],
        }

        ft_ms_delta={
            "mean":-ms_delta["mean"],
            "ci_low":-ms_delta["ci_high"],
            "ci_high":-ms_delta["ci_low"],
        }

        display_name=dataset_labels.get(dataset,dataset)

        for method_label,r,da,dm in [
            (
                pepcl_label,
                pep,
                assay_delta,
                ms_delta,
            ),
            (
                ft_label,
                ft,
                ft_assay_delta,
                ft_ms_delta,
            ),
        ]:
            config=config_format.format(
                lr=_lr_label(r["learning_rate"]),
                steps=int(r["checkpoint"]),
            )

            rows.append({
                assay_col_label:display_name,
                method_col_label:method_label,
                config_col_label:config,

                assay_perf_label:_fmt_ci(
                    float(r["assay_performance_mean"]),
                    float(r["assay_performance_ci_low"]),
                    float(r["assay_performance_ci_high"]),
                ),

                ms_perf_label:_fmt_ci(
                    float(r["ms_performance_mean"]),
                    float(r["ms_performance_ci_low"]),
                    float(r["ms_performance_ci_high"]),
                ),

                assay_delta_label:_fmt_ci(
                    da["mean"],
                    da["ci_low"],
                    da["ci_high"],
                    signed=True,
                ),

                ms_delta_label:_fmt_ci(
                    dm["mean"],
                    dm["ci_low"],
                    dm["ci_high"],
                    signed=True,
                ),

                "_dataset":dataset,
            })

    display=pd.DataFrame(rows)
    display.pop("_dataset")

    final_labels=list(display.columns)
    ncols=len(final_labels)

    if col_widths is None:
        col_widths=[
            0.10,  # Assay
            0.07,  # Update
            0.18,  # Configuration
            0.19,  # Assay performance
            0.18,  # MS performance
            0.14,  # delta assay
            0.14,  # delta MS
        ]

    if len(col_widths)!=ncols:
        raise ValueError(
            f"col_widths must have length {ncols}, got {len(col_widths)}."
        )

    if fig_size is None:
        fig_size=(17,max(3.0,0.42*(len(display)+2)))

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

    fp_header=FontProperties(
        family=font_family,
        size=header_fontsize,
        weight="bold"
    )

    fp_dataset=FontProperties(
        family=font_family,
        size=body_fontsize,
        weight="bold" if bold_dataset_names else "normal"
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
            0.0,1.02,title,
            transform=ax.transAxes,
            ha="left",va="bottom",
            fontproperties=fp_title
        )

    # Dataset labels are drawn separately across the paired rows.
    table_display=display.copy()
    table_display[assay_col_label]=""

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

    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)

    for c in range(ncols):
        tbl[(0,c)].get_text().set_fontproperties(fp_header)

    # Alignment.
    for r in range(nrows):
        if r>0:
            tbl[(r,1)].get_text().set_ha(method_align)
            tbl[(r,2)].get_text().set_ha(config_align)

        for c in range(3,ncols):
            tbl[(r,c)].get_text().set_ha(value_align)

    fig.canvas.draw()

    # ==============================================================
    # Vertically centered dataset labels
    # ==============================================================
    for ds_idx,dataset in enumerate(dataset_order):
        r1=1+2*ds_idx
        r2=r1+1

        cell1=tbl[(r1,0)]
        cell2=tbl[(r2,0)]

        x=cell1.get_x()
        width=cell1.get_width()

        top=cell1.get_y()+cell1.get_height()
        bottom=cell2.get_y()
        y=(top+bottom)/2

        if assay_align=="left":
            tx=x+0.03*width
            ha="left"
        elif assay_align=="right":
            tx=x+0.97*width
            ha="right"
        else:
            tx=x+0.5*width
            ha="center"

        ax.text(
            tx,y,
            dataset_labels.get(dataset,dataset),
            transform=ax.transAxes,
            ha=ha,
            va="center",
            fontproperties=fp_dataset,
            zorder=10,
        )

    # ==============================================================
    # Independent table rules
    # ==============================================================
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
            y,left,right,
            transform=ax.transAxes,
            linewidth=linewidth,
            color=line_color,
            alpha=alpha,
            zorder=5
        )

    _add_rule(header_top,top_linewidth)
    _add_rule(header_bottom,header_linewidth)

    for ds_idx in range(len(dataset_order)-1):
        r2=2+2*ds_idx
        y=tbl[(r2,0)].get_y()

        _add_rule(
            y,
            dataset_group_linewidth,
            alpha=.35
        )

    _add_rule(bottom,bottom_linewidth)

    # ==============================================================
    # Save
    # ==============================================================
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

import re
import numpy as np
import pandas as pd
from scipy.stats import t

def select_ft_pepcl_validation_candidates(
    ms_val_df,
    assay_val_df,
    score_cols,
    method_prefixes,
    ref_col="mhcprime",
    ft_method="ft",
    pepcl_method="pepcl_r_dr",
    min_seeds=3,
    focus_seeds=None,
    selection_se_multiplier=1.0,
    include_checkpoint_models=True,
    ci_level=0.95,
    verbose=True,
):
    """
    Select exactly two validation configurations:

    FT:
        assay_only

    PepCL:
        assay_ms_bounded_assay_first

    The selected rows retain replicate-level assay/MS values so paired
    PepCL-vs-FT differences can subsequently be computed correctly.

    Returns
    -------
    candidate_df, selected_df, selection_log
    """

    # ==============================================================
    # 0. Validation / setup
    # ==============================================================
    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty dictionary.")

    for method in (ft_method,pepcl_method):
        if method not in method_prefixes:
            raise ValueError(f"'{method}' is missing from method_prefixes.")

    selection_se_multiplier=float(selection_se_multiplier)
    if not np.isfinite(selection_se_multiplier) or selection_se_multiplier<0:
        raise ValueError("selection_se_multiplier must be finite and >= 0.")

    if not (0<ci_level<1):
        raise ValueError("ci_level must be in (0,1).")

    if focus_seeds is not None:
        if np.isscalar(focus_seeds):
            focus_seeds=[focus_seeds]

        focus_seeds=tuple(
            sorted(dict.fromkeys(int(s) for s in focus_seeds))
        )

        if not focus_seeds:
            raise ValueError("focus_seeds cannot be empty.")

    for df,name in [
        (ms_val_df,"ms_val_df"),
        (assay_val_df,"assay_val_df"),
    ]:
        if not df.columns.is_unique:
            dup=df.columns[df.columns.duplicated()].tolist()
            raise ValueError(
                f"{name} contains duplicate columns: {dup[:10]}"
            )

    if ref_col not in ms_val_df.columns or ref_col not in assay_val_df.columns:
        raise ValueError(
            f"Reference column '{ref_col}' must exist in both dataframes."
        )

    score_cols=list(dict.fromkeys(score_cols))
    score_cols=[c for c in score_cols if c!=ref_col]

    assay_cols=set(assay_val_df.columns)
    ms_cols=set(ms_val_df.columns)

    assay_usable=[c for c in score_cols if c in assay_cols]
    dual_usable=[c for c in score_cols if c in assay_cols and c in ms_cols]

    if not assay_usable:
        raise ValueError("No candidate columns are available in assay_val_df.")
    if not dual_usable:
        raise ValueError(
            "No candidate columns are available in both assay and MS validation data."
        )

    # ==============================================================
    # 1. Common evaluable allele sets
    # ==============================================================
    def _validate_mask(df,cols,name):
        base=df[ref_col].notna()
        inconsistent=[c for c in cols if not df[c].notna().equals(base)]

        if inconsistent:
            raise ValueError(
                f"{name}: candidates do not share the evaluable allele set "
                f"of '{ref_col}'. Examples: {inconsistent[:10]}"
            )

        out=df.loc[
            base,
            list(dict.fromkeys([ref_col]+list(cols)))
        ].copy()

        if out.empty:
            raise ValueError(f"{name}: no evaluable allele rows remain.")

        return out

    assay_eval=_validate_mask(assay_val_df,assay_usable,"Assay")
    ms_eval=_validate_mask(ms_val_df,dual_usable,"MS")

    assay_metric=assay_eval.mean(axis=0)
    ms_metric=ms_eval.mean(axis=0)

    # ==============================================================
    # 2. Parse model columns
    # ==============================================================
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

    def _parse(cols,with_ms):
        rows=[]

        for col in cols:
            method=_method(col)
            m=pattern.search(str(col))

            if method not in {ft_method,pepcl_method} or m is None:
                continue

            budget=int(m["budget"])
            seed=int(m["seed"])
            is_full=m["checkpoint"] is None
            checkpoint=budget if is_full else int(m["checkpoint"])

            if checkpoint>budget:
                continue

            row={
                "score_col":col,
                "method":method,
                "learning_rate":m["lr"],
                "step_budget":budget,
                "checkpoint":checkpoint,
                "seed":seed,
                "is_full_model":is_full,
                "model_type":"full" if is_full else "checkpoint",
                "assay_performance":float(assay_metric[col]),
            }

            if with_ms:
                row["ms_performance"]=float(ms_metric[col])

            rows.append(row)

        return pd.DataFrame(rows)

    assay_seed_df=_parse(assay_usable,with_ms=False)
    dual_seed_df=_parse(dual_usable,with_ms=True)

    if assay_seed_df.empty:
        raise ValueError("No assay candidates could be parsed.")
    if dual_seed_df.empty:
        raise ValueError("No dual assay/MS candidates could be parsed.")

    # ==============================================================
    # 3. Endpoint deduplication
    # ==============================================================
    def _dedup(df):
        df=df.copy()
        df["_prefer_full"]=df["is_full_model"].astype(int)

        df=(
            df.sort_values(
                [
                    "method","learning_rate","step_budget",
                    "seed","checkpoint","_prefer_full"
                ],
                ascending=[True,True,True,True,True,False]
            )
            .drop_duplicates(
                [
                    "method","learning_rate","step_budget",
                    "seed","checkpoint"
                ],
                keep="first"
            )
            .drop(columns="_prefer_full")
            .reset_index(drop=True)
        )

        if not include_checkpoint_models:
            df=df[
                df["checkpoint"]==df["step_budget"]
            ].copy().reset_index(drop=True)

        return df

    assay_seed_df=_dedup(assay_seed_df)
    dual_seed_df=_dedup(dual_seed_df)

    if focus_seeds is not None:
        assay_seed_df=assay_seed_df[
            assay_seed_df["seed"].isin(focus_seeds)
        ].copy()

        dual_seed_df=dual_seed_df[
            dual_seed_df["seed"].isin(focus_seeds)
        ].copy()

    candidate_keys=[
        "method",
        "learning_rate",
        "step_budget",
        "checkpoint",
    ]

    # ==============================================================
    # 4. Replicate eligibility
    # ==============================================================
    def _eligible(df,name):
        info=(
            df.groupby(candidate_keys)["seed"]
            .agg(
                seed_ids=lambda x:tuple(sorted(set(x))),
                n_seeds="nunique",
            )
            .reset_index()
        )

        if focus_seeds is not None:
            required=tuple(focus_seeds)
            info=info[
                info["seed_ids"].map(lambda x:x==required)
            ].copy()

        else:
            info=info[
                info["n_seeds"]>=min_seeds
            ].copy()

            for method in (ft_method,pepcl_method):
                s=info[info["method"]==method]
                replicate_sets=s["seed_ids"].drop_duplicates().tolist()

                if len(replicate_sets)>1:
                    raise ValueError(
                        f"{name}: method '{method}' has inconsistent replicate IDs "
                        f"across candidates: {replicate_sets}"
                    )

        if info.empty:
            raise ValueError(f"{name}: no eligible candidates remain.")

        valid=set(map(tuple,info[candidate_keys].to_numpy()))

        return df[
            df[candidate_keys].apply(tuple,axis=1).isin(valid)
        ].copy()

    assay_seed_df=_eligible(assay_seed_df,"ASSAY")
    dual_seed_df=_eligible(dual_seed_df,"DUAL")

    # ==============================================================
    # 5. Summary helper
    # ==============================================================
    def _summary(values):
        x=np.asarray(values,dtype=float)
        x=x[np.isfinite(x)]
        n=len(x)

        if n==0:
            return dict(
                mean=np.nan,std=np.nan,se=np.nan,
                ci_low=np.nan,ci_high=np.nan,n=0
            )

        mean=float(np.mean(x))

        if n<2:
            return dict(
                mean=mean,std=np.nan,se=np.nan,
                ci_low=np.nan,ci_high=np.nan,n=n
            )

        sd=float(np.std(x,ddof=1))
        se=sd/np.sqrt(n)
        crit=float(t.ppf((1+ci_level)/2,n-1))

        return dict(
            mean=mean,
            std=sd,
            se=se,
            ci_low=mean-crit*se,
            ci_high=mean+crit*se,
            n=n,
        )

    # ==============================================================
    # 6. Aggregate ASSAY universe
    # ==============================================================
    assay_rows=[]

    for keys,g in assay_seed_df.groupby(candidate_keys,sort=False):
        g=g.sort_values("seed")
        stats=_summary(g["assay_performance"])

        row=dict(zip(candidate_keys,keys))
        row.update({
            "assay_performance_mean":stats["mean"],
            "assay_performance_std":stats["std"],
            "assay_performance_se":stats["se"],
            "assay_performance_ci_low":stats["ci_low"],
            "assay_performance_ci_high":stats["ci_high"],
            "n_seeds":stats["n"],
            "seed_ids":tuple(g["seed"]),
            "assay_performance_by_seed":
                dict(zip(g["seed"],g["assay_performance"])),
        })

        assay_rows.append(row)

    assay_candidate_df=pd.DataFrame(assay_rows)

    # ==============================================================
    # 7. Aggregate DUAL universe
    # ==============================================================
    dual_rows=[]

    for keys,g in dual_seed_df.groupby(candidate_keys,sort=False):
        g=g.sort_values("seed")

        assay_stats=_summary(g["assay_performance"])
        ms_stats=_summary(g["ms_performance"])

        row=dict(zip(candidate_keys,keys))
        row.update({
            "assay_performance_mean":assay_stats["mean"],
            "assay_performance_std":assay_stats["std"],
            "assay_performance_se":assay_stats["se"],
            "assay_performance_ci_low":assay_stats["ci_low"],
            "assay_performance_ci_high":assay_stats["ci_high"],

            "ms_performance_mean":ms_stats["mean"],
            "ms_performance_std":ms_stats["std"],
            "ms_performance_se":ms_stats["se"],
            "ms_performance_ci_low":ms_stats["ci_low"],
            "ms_performance_ci_high":ms_stats["ci_high"],

            "n_seeds":assay_stats["n"],
            "seed_ids":tuple(g["seed"]),

            "assay_performance_by_seed":
                dict(zip(g["seed"],g["assay_performance"])),

            "ms_performance_by_seed":
                dict(zip(g["seed"],g["ms_performance"])),
        })

        dual_rows.append(row)

    dual_candidate_df=pd.DataFrame(dual_rows)

    # ==============================================================
    # 8. Selection helpers
    # ==============================================================
    def _se_stage(pool,metric):
        if pool.empty:
            raise ValueError(f"No candidates available for '{metric}' selection.")

        mean_col=f"{metric}_mean"
        se_col=f"{metric}_se"

        best=pool.loc[pool[mean_col].idxmax()]

        if not np.isfinite(best[se_col]):
            raise ValueError(
                f"Best '{metric}' candidate has non-finite SE."
            )

        threshold=(
            float(best[mean_col])
            -selection_se_multiplier*float(best[se_col])
        )

        eligible=pool[
            pool[mean_col]>=threshold
        ].copy()

        return best,float(threshold),eligible

    def _tie(df,mean_col,std_col):
        return (
            df.sort_values(
                [
                    "checkpoint",
                    "step_budget",
                    mean_col,
                    std_col,
                    "learning_rate",
                ],
                ascending=[True,True,False,True,True]
            )
            .iloc[[0]]
            .copy()
        )

    selection_log={}

    # ==============================================================
    # 9. FT: assay_only
    # ==============================================================
    ft_pool=assay_candidate_df[
        assay_candidate_df["method"]==ft_method
    ].copy()

    if ft_pool.empty:
        raise ValueError(f"No eligible FT candidates for '{ft_method}'.")

    ft_best,ft_threshold,ft_eligible=_se_stage(
        ft_pool,
        "assay_performance"
    )

    ft_selected=_tie(
        ft_eligible,
        "assay_performance_mean",
        "assay_performance_std"
    )

    ft_selected["selection_type"]="assay_only"

    # Add the dual assay/MS summary for the exact selected FT config,
    # without using MS during FT selection.
    ft_selected=ft_selected[
        candidate_keys+["selection_type"]
    ].merge(
        dual_candidate_df[
            dual_candidate_df["method"]==ft_method
        ],
        on=candidate_keys,
        how="left"
    )

    if ft_selected["ms_performance_mean"].isna().any():
        raise ValueError(
            "The assay-selected FT configuration is not evaluable on MS. "
            "Paired FT/PepCL MS comparison cannot be constructed."
        )

    selection_log[ft_method]={
        "selection_type":"assay_only",
        "best_candidate":ft_best.to_frame().T,
        "one_se_threshold":ft_threshold,
        "se_multiplier":selection_se_multiplier,
        "eligible_candidates":ft_eligible.copy(),
        "final_selection":ft_selected.copy(),
    }

    # ==============================================================
    # 10. PepCL: bounded assay -> MS
    # ==============================================================
    pep_pool=dual_candidate_df[
        dual_candidate_df["method"]==pepcl_method
    ].copy()

    if pep_pool.empty:
        raise ValueError(f"No eligible PepCL candidates for '{pepcl_method}'.")

    pep_assay_best,pep_assay_threshold,pep_assay_eligible=_se_stage(
        pep_pool,
        "assay_performance"
    )

    pep_ms_best,pep_ms_threshold,pep_ms_eligible=_se_stage(
        pep_assay_eligible,
        "ms_performance"
    )

    pep_selected=_tie(
        pep_ms_eligible,
        "ms_performance_mean",
        "ms_performance_std"
    )

    pep_selected[
        "selection_type"
    ]="assay_ms_bounded_assay_first"

    selection_log[pepcl_method]={
        "selection_type":"assay_ms_bounded_assay_first",
        "se_multiplier":selection_se_multiplier,

        "stage1_metric":"assay_performance",
        "stage1_best_candidate":pep_assay_best.to_frame().T,
        "stage1_threshold":pep_assay_threshold,
        "stage1_eligible_candidates":pep_assay_eligible.copy(),

        "stage2_metric":"ms_performance",
        "stage2_best_candidate":pep_ms_best.to_frame().T,
        "stage2_threshold":pep_ms_threshold,
        "stage2_eligible_candidates":pep_ms_eligible.copy(),

        "final_selection":pep_selected.copy(),
    }

    # ==============================================================
    # 11. Final selected output: PepCL then FT
    # ==============================================================
    selected_df=pd.concat(
        [pep_selected,ft_selected],
        ignore_index=True,
        sort=False
    )

    selected_cols=[
        "method",
        "selection_type",
        "learning_rate",
        "step_budget",
        "checkpoint",

        "assay_performance_mean",
        "assay_performance_std",
        "assay_performance_se",
        "assay_performance_ci_low",
        "assay_performance_ci_high",

        "ms_performance_mean",
        "ms_performance_std",
        "ms_performance_se",
        "ms_performance_ci_low",
        "ms_performance_ci_high",

        "n_seeds",
        "seed_ids",
        "assay_performance_by_seed",
        "ms_performance_by_seed",
    ]

    selected_df=selected_df[selected_cols].copy()

    # Minimal candidate table useful for inspection.
    assay_candidate_out=assay_candidate_df.copy()
    assay_candidate_out["candidate_universe"]="assay"

    dual_candidate_out=dual_candidate_df.copy()
    dual_candidate_out["candidate_universe"]="dual"

    candidate_df=pd.concat(
        [assay_candidate_out,dual_candidate_out],
        ignore_index=True,
        sort=False
    )

    if verbose:
        print("\nFINAL VALIDATION SELECTIONS")
        print(
            selected_df[
                [
                    "method",
                    "selection_type",
                    "learning_rate",
                    "step_budget",
                    "checkpoint",
                    "assay_performance_mean",
                    "ms_performance_mean",
                    "n_seeds",
                ]
            ].to_string(index=False)
        )

    return candidate_df,selected_df,selection_log

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
def select_ft_early_stopping_candidate(
    assay_val_df, score_cols, method_prefixes, ref_col="mhcprime", ft_method="ft",
    min_seeds=3, focus_seeds=None, n_uncertainty=1.0, uncertainty="se", verbose=True,
):
    """
    FT assay-only early-stopping analysis.

    1. Macro-average allele AP for each FT score column.
    2. Aggregate identical (LR, budget, checkpoint) configurations over replicates.
    3. Find the candidate with highest mean assay performance.
    4. Threshold = best_mean - N * best_SE/SD.
    5. Among candidates >= threshold, select the fewest ACTUAL optimization steps.

    `checkpoint` is treated as actual steps run:
        steps_800 ... step_100 -> actual_steps=100
        steps_100 full model    -> actual_steps=100

    Tie-breaking among equal actual_steps:
        smaller total step budget -> higher assay mean -> lower uncertainty -> LR.

    Returns
    -------
    candidate_df : all eligible aggregated FT candidates
    eligible_df  : candidates within N SE/SD of the best
    selected_df  : one lowest-step selected candidate
    """
    if ft_method not in method_prefixes: raise ValueError(f"ft_method='{ft_method}' is not in method_prefixes.")
    if uncertainty not in {"se","sd"}: raise ValueError("uncertainty must be 'se' or 'sd'.")
    try: n_uncertainty=float(n_uncertainty)
    except (TypeError,ValueError): raise TypeError("n_uncertainty must be numeric.")
    if not np.isfinite(n_uncertainty) or n_uncertainty<0: raise ValueError("n_uncertainty must be finite and >= 0.")
    if ref_col not in assay_val_df.columns: raise ValueError(f"Reference '{ref_col}' must exist in assay_val_df.")

    if focus_seeds is not None:
        if np.isscalar(focus_seeds): focus_seeds=[focus_seeds]
        focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in focus_seeds)))
        if not focus_seeds: raise ValueError("focus_seeds cannot be empty.")

    prefix=method_prefixes[ft_method]
    score_cols=list(dict.fromkeys(score_cols))
    usable=[c for c in score_cols if c!=ref_col and c in assay_val_df.columns and str(c).startswith(prefix)]
    if not usable: raise ValueError(f"No FT score columns found for prefix '{prefix}'.")

    base_mask=assay_val_df[ref_col].notna()
    bad=[c for c in usable if not assay_val_df[c].notna().equals(base_mask)]
    if bad: raise ValueError(f"{len(bad)} FT columns do not share the reference evaluable allele mask. Examples: {bad[:10]}")
    assay_metric=assay_val_df.loc[base_mask,usable].mean(axis=0)

    pattern=re.compile(r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)(?:_step_(?P<checkpoint>\d+))?$")
    rows=[]; unparsable=[]
    for col in usable:
        m=pattern.search(str(col))
        if m is None: unparsable.append(col); continue
        budget=int(m["budget"]); seed=int(m["seed"]); cp=m["checkpoint"]; is_full=cp is None; checkpoint=budget if is_full else int(cp)
        if checkpoint>budget: unparsable.append(col); continue
        rows.append({
            "score_col":col,"method":ft_method,"learning_rate":m["lr"],"step_budget":budget,
            "checkpoint":checkpoint,"actual_steps":checkpoint,"seed":seed,
            "is_full_model":is_full,"model_type":"full" if is_full else "checkpoint",
            "assay_performance":float(assay_metric[col]),
        })

    seed_df=pd.DataFrame(rows)
    if seed_df.empty: raise ValueError("No valid FT columns could be parsed.")

    seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)
    seed_df=(seed_df.sort_values(["learning_rate","step_budget","seed","checkpoint","_prefer_full"],
                                ascending=[True,True,True,True,False])
        .drop_duplicates(["learning_rate","step_budget","seed","checkpoint"],keep="first")
        .drop(columns="_prefer_full").reset_index(drop=True))

    keys=["method","learning_rate","step_budget","checkpoint"]

    if focus_seeds is not None:
        seed_df=seed_df[seed_df["seed"].isin(focus_seeds)].copy()
        if seed_df.empty: raise ValueError(f"No FT rows remain for focus_seeds={focus_seeds}.")

    info=(seed_df.groupby(keys)["seed"]
        .agg(seed_ids=lambda x:tuple(sorted(set(x))),n_seeds="nunique").reset_index())

    if focus_seeds is not None:
        required=tuple(focus_seeds)
        info=info[info["seed_ids"].map(lambda x:x==required)].copy()
        if info.empty: raise ValueError(f"No FT candidates contain all focused replicate IDs {required}.")
    else:
        info=info[info["n_seeds"]>=min_seeds].copy()
        if info.empty: raise ValueError(f"No FT candidates have at least {min_seeds} replicates.")
        sets=info["seed_ids"].drop_duplicates().tolist()
        if len(sets)>1:
            raise ValueError(f"FT candidates have inconsistent replicate sets: {sets}. Use focus_seeds to define a common set.")

    valid_keys=set(map(tuple,info[keys].to_numpy()))
    seed_df=seed_df[seed_df[keys].apply(tuple,axis=1).isin(valid_keys)].copy()

    candidate_df=(seed_df.groupby(keys,sort=False)
        .agg(
            assay_performance_mean=("assay_performance","mean"),
            assay_performance_std=("assay_performance","std"),
            n_seeds=("seed","nunique"),
            seed_ids=("seed",lambda x:tuple(sorted(set(x)))),
            score_cols_by_seed=("score_col",lambda x:dict(zip(seed_df.loc[x.index,"seed"],x)))
        ).reset_index())

    candidate_df["assay_performance_se"]=candidate_df["assay_performance_std"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["actual_steps"]=candidate_df["checkpoint"]

    best=candidate_df.loc[candidate_df["assay_performance_mean"].idxmax()]
    u_col="assay_performance_se" if uncertainty=="se" else "assay_performance_std"

    if not np.isfinite(best[u_col]):
        raise ValueError(f"Best candidate {uncertainty.upper()} is NaN; at least two replicates are required.")

    threshold=float(best["assay_performance_mean"])-n_uncertainty*float(best[u_col])
    eligible_df=candidate_df[candidate_df["assay_performance_mean"]>=threshold].copy()
    eligible_df["performance_delta_from_best"]=eligible_df["assay_performance_mean"]-float(best["assay_performance_mean"])
    eligible_df["within_best_band"]=True

    selected_df=(eligible_df.sort_values(
        ["actual_steps","step_budget","assay_performance_mean",u_col,"learning_rate"],
        ascending=[True,True,False,True,True]
    ).iloc[[0]].copy())

    candidate_df["best_assay_mean"]=float(best["assay_performance_mean"])
    candidate_df[f"best_assay_{uncertainty}"]=float(best[u_col])
    candidate_df["selection_threshold"]=threshold
    candidate_df["within_best_band"]=candidate_df["assay_performance_mean"]>=threshold

    if verbose:
        print(f"FT candidates: {len(candidate_df)}")
        print(f"Replicates: {'focus '+str(focus_seeds) if focus_seeds is not None else candidate_df['seed_ids'].iloc[0]}")
        print(
            f"\nBest assay candidate: LR={best['learning_rate']}, budget={int(best['step_budget'])}, "
            f"step={int(best['checkpoint'])}, mean={best['assay_performance_mean']:.6f}, "
            f"{uncertainty.upper()}={best[u_col]:.6f}"
        )
        print(f"{n_uncertainty:g}-{uncertainty.upper()} lower threshold: {threshold:.6f}")
        print(f"Candidates within threshold: {len(eligible_df)}")

        s=selected_df.iloc[0]
        print(
            f"\nEARLY-STOPPING SELECTION: LR={s['learning_rate']}, budget={int(s['step_budget'])}, "
            f"step={int(s['checkpoint'])}, mean={s['assay_performance_mean']:.6f}"
        )
        print(
            f"Step reduction vs best candidate: "
            f"{int(best['checkpoint'])} -> {int(s['checkpoint'])} "
            f"({int(best['checkpoint'])-int(s['checkpoint'])} fewer actual steps)"
        )

    return candidate_df,eligible_df,selected_df

import re
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

def find_validation_interval_overlap_candidates(
    assay_val_df, score_cols, method_prefixes, target_step_budget, target_learning_rate,
    ms_val_df=None, ref_col="mhcprime", uncertainty="se",
    target_multiplier=1.0, candidate_multiplier=1.0,
    keep_by="both", focus_seeds=None, min_seeds=2,
    include_target=True, verbose=True,
):
    """
    Find candidate configurations whose uncertainty intervals overlap with a
    method-specific target configuration.

    uncertainty="se":
        mean ± multiplier * SE

    uncertainty="sd":
        mean ± multiplier * SD

    uncertainty="ci95":
        mean ± t(0.975, n-1) * SE
        target_multiplier and candidate_multiplier must both equal 1.

    Two intervals overlap when:
        max(target_low, candidate_low) <= min(target_high, candidate_high)

    Parameters
    ----------
    assay_val_df : pd.DataFrame
        Assay per-allele AP dataframe.

    ms_val_df : pd.DataFrame or None
        MS per-allele AP dataframe. May be None if all requested methods use
        keep_by='assay'.

    score_cols : iterable
        Model score columns.

    method_prefixes : dict
        Example:
            {"ft":"escape_con_ft_",
             "pepcl_r_dr":"escape_con_pepcl_r_dr_"}

    target_step_budget, target_learning_rate :
        Scalar applying to every method, OR dict keyed by method.

    uncertainty : {"se","sd","ci95"}
        Uncertainty interval type.

    target_multiplier : float
        Number of target SEs/SDs used around target mean.
        Must equal 1 for uncertainty="ci95".

    candidate_multiplier : float
        Number of candidate SEs/SDs used around candidate mean.
        Must equal 1 for uncertainty="ci95".

    keep_by : {"assay","ms","both","either"} or dict
        assay  -> assay intervals overlap
        ms     -> MS intervals overlap
        both   -> assay AND MS intervals overlap
        either -> assay OR MS intervals overlap

        May be supplied per method:
            {"ft":"assay", "pepcl_r_dr":"both"}

    focus_seeds : iterable or None
        iterable -> use exactly these replicate IDs and require every candidate
                    and target to contain all of them.
        None -> retain candidates satisfying min_seeds, while requiring a
                consistent replicate set within each method.

    min_seeds : int
        Minimum number of replicates.

    Returns
    -------
    candidate_df : pd.DataFrame
        All eligible candidates with means, SD/SE/CI95, target/candidate
        intervals, overlap indicators, and retain_candidate.

    overlap_df : pd.DataFrame
        Candidates satisfying keep_by.

    target_df : pd.DataFrame
        Target configuration for every method.
    """
    if not isinstance(method_prefixes,dict) or not method_prefixes:
        raise ValueError("method_prefixes must be a non-empty dict.")
    if uncertainty not in {"se","sd","ci95"}:
        raise ValueError("uncertainty must be 'se', 'sd', or 'ci95'.")
    if min_seeds<2:
        raise ValueError("min_seeds must be >= 2 because uncertainty requires replicate variability.")

    try:
        target_multiplier=float(target_multiplier)
        candidate_multiplier=float(candidate_multiplier)
    except (TypeError,ValueError):
        raise TypeError("target_multiplier and candidate_multiplier must be numeric.")

    if not np.isfinite(target_multiplier) or target_multiplier<0:
        raise ValueError("target_multiplier must be finite and >= 0.")
    if not np.isfinite(candidate_multiplier) or candidate_multiplier<0:
        raise ValueError("candidate_multiplier must be finite and >= 0.")
    if uncertainty=="ci95" and (target_multiplier!=1 or candidate_multiplier!=1):
        raise ValueError(
            "For uncertainty='ci95', target_multiplier and candidate_multiplier "
            "must both equal 1 because the CI95 column is already the full "
            "95% confidence-interval half-width."
        )

    def _method_arg(x,method,name):
        if isinstance(x,dict):
            if method not in x: raise ValueError(f"{name} missing method '{method}'.")
            return x[method]
        return x

    def _keep_mode(method):
        mode=_method_arg(keep_by,method,"keep_by")
        if mode not in {"assay","ms","both","either"}:
            raise ValueError(f"keep_by for '{method}' must be 'assay', 'ms', 'both', or 'either'.")
        return mode

    def _normalize_lr(lr):
        s=str(lr)
        if "eN" in s: return s
        try:
            mant,exp=f"{float(lr):.12e}".split("e")
            mant=mant.rstrip("0").rstrip("."); exp=int(exp)
            return f"{mant}eN{abs(exp)}" if exp<0 else f"{mant}e{exp}"
        except (TypeError,ValueError):
            return s

    if focus_seeds is not None:
        if np.isscalar(focus_seeds): focus_seeds=[focus_seeds]
        try:
            focus_seeds=tuple(sorted(dict.fromkeys(int(s) for s in focus_seeds)))
        except (TypeError,ValueError):
            raise TypeError("focus_seeds must contain integer replicate IDs.")
        if not focus_seeds: raise ValueError("focus_seeds cannot be empty.")
        if len(focus_seeds)<2: raise ValueError("At least two focus_seeds are required for uncertainty.")

    need_ms=any(_keep_mode(m) in {"ms","both","either"} for m in method_prefixes)
    if need_ms and ms_val_df is None:
        raise ValueError("ms_val_df is required because at least one method uses keep_by='ms', 'both', or 'either'.")

    if ref_col not in assay_val_df.columns:
        raise ValueError(f"Reference '{ref_col}' not found in assay_val_df.")
    if need_ms and ref_col not in ms_val_df.columns:
        raise ValueError(f"Reference '{ref_col}' not found in ms_val_df.")

    if not assay_val_df.columns.is_unique:
        dup=assay_val_df.columns[assay_val_df.columns.duplicated()].tolist()
        raise ValueError(f"assay_val_df contains duplicate columns: {dup[:10]}")
    if ms_val_df is not None and not ms_val_df.columns.is_unique:
        dup=ms_val_df.columns[ms_val_df.columns.duplicated()].tolist()
        raise ValueError(f"ms_val_df contains duplicate columns: {dup[:10]}")

    score_cols=list(dict.fromkeys(score_cols))
    ordered_methods=sorted(method_prefixes.items(),key=lambda x:len(x[1]),reverse=True)

    def _method(col):
        for name,prefix in ordered_methods:
            if str(col).startswith(prefix): return name
        return None

    # Method-specific availability: FT can exist only in assay_val_df, while
    # PepCL may require presence in both assay and MS.
    usable=[]
    for c in score_cols:
        if c==ref_col: continue
        method=_method(c)
        if method is None or c not in assay_val_df.columns: continue
        if _keep_mode(method) in {"ms","both","either"} and (ms_val_df is None or c not in ms_val_df.columns):
            continue
        usable.append(c)

    if not usable:
        raise ValueError("No usable candidate columns found.")

    if verbose:
        print(f"Score columns supplied: {len(score_cols)}")
        print(f"Usable candidate columns: {len(usable)}")
        print(f"Uncertainty: {uncertainty.upper()}")
        if uncertainty=="ci95":
            print("Intervals: two-sided 95% t confidence intervals")
        else:
            print(f"Target interval: ±{target_multiplier:g} {uncertainty.upper()}")
            print(f"Candidate interval: ±{candidate_multiplier:g} {uncertainty.upper()}")
        print(f"Replicates: {'focus '+str(focus_seeds) if focus_seeds is not None else f'min_seeds={min_seeds}'}")

    def _metric(df,cols,name):
        if not cols: return pd.Series(dtype=float)
        base=df[ref_col].notna()
        bad=[c for c in cols if not df[c].notna().equals(base)]
        if bad:
            raise ValueError(
                f"{name}: {len(bad)} candidates do not share the evaluable row set "
                f"of '{ref_col}'. Examples: {bad[:10]}"
            )
        out=df.loc[base,cols]
        if out.empty: raise ValueError(f"{name}: no evaluable rows remain.")
        if verbose: print(f"{name}: common evaluable row set = {len(out)}")
        return out.mean(axis=0)

    assay_metric=_metric(assay_val_df,usable,"Assay")
    ms_cols=[c for c in usable if ms_val_df is not None and c in ms_val_df.columns]
    ms_metric=_metric(ms_val_df,ms_cols,"MS") if ms_cols else pd.Series(dtype=float)

    pattern=re.compile(
        r"_steps_(?P<budget>\d+)_lr_(?P<lr>.+?)_seed_(?P<seed>\d+)"
        r"(?:_step_(?P<checkpoint>\d+))?$"
    )

    rows=[]; bad=[]
    for col in usable:
        method=_method(col); m=pattern.search(str(col))
        if method is None or m is None:
            bad.append(col); continue

        budget=int(m["budget"]); seed=int(m["seed"])
        cp=m["checkpoint"]; is_full=cp is None
        checkpoint=budget if is_full else int(cp)

        if checkpoint>budget:
            bad.append(col); continue

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
            "ms_performance":float(ms_metric[col]) if col in ms_metric.index else np.nan,
        })

    seed_df=pd.DataFrame(rows)
    if seed_df.empty:
        raise ValueError("No valid candidate score columns could be parsed.")
    if verbose and bad:
        print(f"Skipped {len(bad)} unparsable/invalid columns.")

    # Collapse explicit full model and duplicate `_step_budget`, preferring full.
    before=len(seed_df)
    seed_df["_prefer_full"]=seed_df["is_full_model"].astype(int)
    seed_df=(seed_df.sort_values(
            ["method","learning_rate","step_budget","seed","checkpoint","_prefer_full"],
            ascending=[True,True,True,True,True,False])
        .drop_duplicates(
            ["method","learning_rate","step_budget","seed","checkpoint"],
            keep="first"
        )
        .drop(columns="_prefer_full")
        .reset_index(drop=True))

    if verbose:
        print(f"Duplicate full/final-checkpoint endpoints removed: {before-len(seed_df)}")

    keys=["method","learning_rate","step_budget","checkpoint"]

    if focus_seeds is not None:
        before=len(seed_df)
        seed_df=seed_df[seed_df["seed"].isin(focus_seeds)].copy()
        if seed_df.empty:
            raise ValueError(f"No candidate rows remain for focus_seeds={focus_seeds}.")
        if verbose:
            print(f"Focused replicate rows: {before} -> {len(seed_df)}")

    seed_info=(seed_df.groupby(keys)["seed"]
        .agg(seed_ids=lambda x:tuple(sorted(set(x))),n_seeds="nunique")
        .reset_index())

    if focus_seeds is not None:
        required=tuple(focus_seeds)
        seed_info=seed_info[seed_info["seed_ids"].map(lambda x:x==required)].copy()
        if seed_info.empty:
            raise ValueError(f"No candidates contain all focused replicate IDs {required}.")
    else:
        seed_info=seed_info[seed_info["n_seeds"]>=min_seeds].copy()
        if seed_info.empty:
            raise ValueError(f"No candidates have at least {min_seeds} distinct replicates.")

        # Preserve rigorous same-replicate-set behavior within each method.
        valid=[]
        for method in method_prefixes:
            s=seed_info[seed_info["method"]==method].copy()
            if s.empty: continue

            sets=s["seed_ids"].drop_duplicates().tolist()
            if len(sets)>1:
                raise ValueError(
                    f"Method '{method}' has inconsistent replicate identifiers: {sets}. "
                    "Use focus_seeds or filter to a common replicate set."
                )
            valid.append(s)

        if not valid:
            raise ValueError("No methods contain eligible candidates.")
        seed_info=pd.concat(valid,ignore_index=True)

    valid_keys=set(map(tuple,seed_info[keys].to_numpy()))
    seed_df=seed_df[seed_df[keys].apply(tuple,axis=1).isin(valid_keys)].copy()

    candidate_df=(seed_df.groupby(keys,sort=False)
        .agg(
            assay_performance_mean=("assay_performance","mean"),
            assay_performance_std=("assay_performance","std"),
            ms_performance_mean=("ms_performance","mean"),
            ms_performance_std=("ms_performance","std"),
            n_seeds=("seed","nunique"),
            seed_ids=("seed",lambda x:tuple(sorted(set(x))))
        ).reset_index())

    candidate_df["assay_performance_se"]=candidate_df["assay_performance_std"]/np.sqrt(candidate_df["n_seeds"])
    candidate_df["ms_performance_se"]=candidate_df["ms_performance_std"]/np.sqrt(candidate_df["n_seeds"])

    # 95% t confidence-interval HALF-WIDTHS.
    candidate_df["ci95_t_critical"]=student_t.ppf(0.975,candidate_df["n_seeds"]-1)
    candidate_df["assay_performance_ci95"]=candidate_df["ci95_t_critical"]*candidate_df["assay_performance_se"]
    candidate_df["ms_performance_ci95"]=candidate_df["ci95_t_critical"]*candidate_df["ms_performance_se"]
    candidate_df["actual_steps"]=candidate_df["checkpoint"]

    raw=[]
    for k,g in seed_df.groupby(keys,sort=False):
        g=g.sort_values("seed")
        row=dict(zip(keys,k))
        rep="full" if g["is_full_model"].all() else "checkpoint" if (~g["is_full_model"]).all() else "mixed_endpoint"

        row.update({
            "model_type":rep,
            "is_terminal_endpoint":bool((g["checkpoint"]==g["step_budget"]).all()),
            "score_cols_by_seed":dict(zip(g["seed"],g["score_col"])),
        })
        raw.append(row)

    candidate_df=candidate_df.merge(pd.DataFrame(raw),on=keys,how="left")

    # Identify method-specific targets.
    target_rows=[]
    for method in method_prefixes:
        budget=int(_method_arg(target_step_budget,method,"target_step_budget"))
        lr=_normalize_lr(_method_arg(target_learning_rate,method,"target_learning_rate"))

        t=candidate_df[
            (candidate_df["method"]==method)&
            (candidate_df["step_budget"]==budget)&
            (candidate_df["learning_rate"].astype(str)==lr)&
            (candidate_df["checkpoint"]==budget)
        ].copy()

        if t.empty:
            raise ValueError(f"No target endpoint for '{method}' with steps={budget}, LR={lr}.")
        if len(t)!=1:
            raise ValueError(f"Expected one aggregated target for '{method}', found {len(t)}.")

        target_rows.append(t.iloc[0])

    target_df=pd.DataFrame(target_rows).reset_index(drop=True)

    if uncertainty=="se":
        assay_u_col="assay_performance_se"
        ms_u_col="ms_performance_se"
    elif uncertainty=="sd":
        assay_u_col="assay_performance_std"
        ms_u_col="ms_performance_std"
    else:
        assay_u_col="assay_performance_ci95"
        ms_u_col="ms_performance_ci95"

    # All candidates need finite assay uncertainty; methods using MS also need
    # finite MS uncertainty.
    if candidate_df[assay_u_col].isna().any():
        raise ValueError(f"Some candidate assay {uncertainty.upper()} values are NaN.")

    for method in method_prefixes:
        if _keep_mode(method) in {"ms","both","either"}:
            d=candidate_df[candidate_df["method"]==method]
            if d[ms_u_col].isna().any():
                raise ValueError(f"Some '{method}' MS {uncertainty.upper()} values are NaN.")

    pieces=[]
    for _,target in target_df.iterrows():
        method=target["method"]
        mode=_keep_mode(method)
        d=candidate_df[candidate_df["method"]==method].copy()

        assay_target_mean=float(target["assay_performance_mean"])
        assay_target_u=float(target[assay_u_col])
        assay_target_low=assay_target_mean-target_multiplier*assay_target_u
        assay_target_high=assay_target_mean+target_multiplier*assay_target_u

        d["target_step_budget"]=int(target["step_budget"])
        d["target_learning_rate"]=target["learning_rate"]
        d["target_assay_mean"]=assay_target_mean
        d[f"target_assay_{uncertainty}"]=assay_target_u
        d["target_assay_interval_low"]=assay_target_low
        d["target_assay_interval_high"]=assay_target_high

        d["candidate_assay_interval_low"]=(
            d["assay_performance_mean"]-
            candidate_multiplier*d[assay_u_col]
        )
        d["candidate_assay_interval_high"]=(
            d["assay_performance_mean"]+
            candidate_multiplier*d[assay_u_col]
        )

        d["assay_interval_overlap"]=(
            np.maximum(
                d["candidate_assay_interval_low"],
                assay_target_low
            )
            <=
            np.minimum(
                d["candidate_assay_interval_high"],
                assay_target_high
            )
        )

        d["assay_interval_overlap_width"]=np.maximum(
            0.0,
            np.minimum(d["candidate_assay_interval_high"],assay_target_high)-
            np.maximum(d["candidate_assay_interval_low"],assay_target_low)
        )

        d["assay_delta_from_target"]=d["assay_performance_mean"]-assay_target_mean

        if mode in {"ms","both","either"}:
            ms_target_mean=float(target["ms_performance_mean"])
            ms_target_u=float(target[ms_u_col])
            ms_target_low=ms_target_mean-target_multiplier*ms_target_u
            ms_target_high=ms_target_mean+target_multiplier*ms_target_u

            d["target_ms_mean"]=ms_target_mean
            d[f"target_ms_{uncertainty}"]=ms_target_u
            d["target_ms_interval_low"]=ms_target_low
            d["target_ms_interval_high"]=ms_target_high

            d["candidate_ms_interval_low"]=(
                d["ms_performance_mean"]-
                candidate_multiplier*d[ms_u_col]
            )
            d["candidate_ms_interval_high"]=(
                d["ms_performance_mean"]+
                candidate_multiplier*d[ms_u_col]
            )

            d["ms_interval_overlap"]=(
                np.maximum(
                    d["candidate_ms_interval_low"],
                    ms_target_low
                )
                <=
                np.minimum(
                    d["candidate_ms_interval_high"],
                    ms_target_high
                )
            )

            d["ms_interval_overlap_width"]=np.maximum(
                0.0,
                np.minimum(d["candidate_ms_interval_high"],ms_target_high)-
                np.maximum(d["candidate_ms_interval_low"],ms_target_low)
            )

            d["ms_delta_from_target"]=d["ms_performance_mean"]-ms_target_mean
        else:
            d["target_ms_mean"]=np.nan
            d[f"target_ms_{uncertainty}"]=np.nan
            d["target_ms_interval_low"]=np.nan
            d["target_ms_interval_high"]=np.nan
            d["candidate_ms_interval_low"]=np.nan
            d["candidate_ms_interval_high"]=np.nan
            d["ms_interval_overlap"]=False
            d["ms_interval_overlap_width"]=np.nan
            d["ms_delta_from_target"]=np.nan

        d["interval_overlap_either"]=d["assay_interval_overlap"]|d["ms_interval_overlap"]
        d["interval_overlap_both"]=d["assay_interval_overlap"]&d["ms_interval_overlap"]

        if mode=="assay":
            d["retain_candidate"]=d["assay_interval_overlap"]
        elif mode=="ms":
            d["retain_candidate"]=d["ms_interval_overlap"]
        elif mode=="both":
            d["retain_candidate"]=d["interval_overlap_both"]
        else:
            d["retain_candidate"]=d["interval_overlap_either"]

        d["keep_by"]=mode
        d["uncertainty"]=uncertainty
        d["target_multiplier"]=target_multiplier
        d["candidate_multiplier"]=candidate_multiplier

        d["is_target"]=(
            (d["step_budget"]==int(target["step_budget"]))&
            (d["learning_rate"].astype(str)==str(target["learning_rate"]))&
            (d["checkpoint"]==int(target["checkpoint"]))
        )

        pieces.append(d)

    candidate_df=pd.concat(pieces,ignore_index=True)

    overlap_df=candidate_df[candidate_df["retain_candidate"]].copy()
    if not include_target:
        overlap_df=overlap_df[~overlap_df["is_target"]].copy()

    overlap_df=overlap_df.sort_values(
        ["method","checkpoint","step_budget","learning_rate"],
        ascending=[True,True,True,True]
    ).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*90}")
        print("VALIDATION INTERVAL-OVERLAP ANALYSIS")
        print(f"{'='*90}")

        if uncertainty=="ci95":
            print("Uncertainty: two-sided 95% t confidence intervals")
        else:
            print(f"Uncertainty: {uncertainty.upper()}")
            print(f"Target interval multiplier: ±{target_multiplier:g}")
            print(f"Candidate interval multiplier: ±{candidate_multiplier:g}")

        for method in method_prefixes:
            target=target_df[target_df["method"]==method].iloc[0]
            sub=overlap_df[overlap_df["method"]==method]
            mode=_keep_mode(method)

            print(f"\nMETHOD: {method}")
            print(
                f"Target: LR={target['learning_rate']}, "
                f"steps={int(target['step_budget'])}, "
                f"replicates={target['seed_ids']}"
            )
            print(f"Filter: {mode} interval overlap")

            if uncertainty=="ci95":
                print(
                    f"Assay target: {target['assay_performance_mean']:.6f} "
                    f"± CI95 half-width ({target[assay_u_col]:.6f})"
                )
            else:
                print(
                    f"Assay target: {target['assay_performance_mean']:.6f} "
                    f"± {target_multiplier:g}×{uncertainty.upper()} "
                    f"({target[assay_u_col]:.6f})"
                )

            if mode in {"ms","both","either"}:
                if uncertainty=="ci95":
                    print(
                        f"MS target:    {target['ms_performance_mean']:.6f} "
                        f"± CI95 half-width ({target[ms_u_col]:.6f})"
                    )
                else:
                    print(
                        f"MS target:    {target['ms_performance_mean']:.6f} "
                        f"± {target_multiplier:g}×{uncertainty.upper()} "
                        f"({target[ms_u_col]:.6f})"
                    )

            print(f"Retained candidates: {len(sub)}")

            cols=[
                "learning_rate","step_budget","checkpoint","model_type",
                "n_seeds","seed_ids",
                "assay_performance_mean",assay_u_col,
                "candidate_assay_interval_low","candidate_assay_interval_high",
                "assay_interval_overlap","assay_interval_overlap_width",
            ]

            if mode in {"ms","both","either"}:
                cols += [
                    "ms_performance_mean",ms_u_col,
                    "candidate_ms_interval_low","candidate_ms_interval_high",
                    "ms_interval_overlap","ms_interval_overlap_width",
                    "interval_overlap_either","interval_overlap_both",
                ]

            cols += ["retain_candidate","is_target"]

            if not sub.empty:
                print(sub[cols].to_string(index=False))

    return candidate_df,overlap_df,target_df

import re
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_ft_matches_to_pepcl_bars(
    comparison_df, include_mhcprime=True, uncertainty="sd",
    ms_color="#9BB0ED", assay_color="#FFB55C",
    mhcprime_ms_color="#081470", mhcprime_assay_color="#BABABA",
    mhcprime_ms_linestyle="--", mhcprime_assay_linestyle="--",
    mhcprime_ms_linewidth=1.0, mhcprime_assay_linewidth=1.0,
    bar_width=0.35, within_pair_spacing=0.0, pair_spacing=0.65,
    bar_edgecolor="black", bar_linewidth=1.0,
    uncertainty_color="black", uncertainty_linewidth=1.5,
    x_labels=None, xtick_rotation=45, wrap_width=32, y_lim=None,
    title="PepCL and matched FT performance", xlabel="", ylabel="mAP",
    title_fontsize=14, xlabel_fontsize=12, ylabel_fontsize=12,
    xtick_fontsize=10, ytick_fontsize=10,
    fig_size=(10,6), dpi=300,
    legend=True, legend_loc="upper center", legend_bbox=(0.5,-0.22),
    legend_ncol=4, legend_frameon=False, legend_fontsize=10,
    legend_title=None, legend_title_fontsize=10,
    legend_columnspacing=1.5, legend_handletextpad=0.6,
    legend_square_markersize=10,
    legend_mhcprime_ms_linewidth=1.5, legend_mhcprime_assay_linewidth=1.5,
    legend_labels=None, legend_handles=None,
    bottom_margin=0.32, show_plot=True,
):
    """
    Plot paired MS/assay bars from get_ft_matches_to_pepcl comparison_df.

    Font-size controls:
      title_fontsize
      xlabel_fontsize
      ylabel_fontsize
      xtick_fontsize
      ytick_fontsize
      legend_fontsize
      legend_title_fontsize
    """
    if uncertainty not in {"sd","se",None}: raise ValueError("uncertainty must be 'sd', 'se', or None.")
    required={"mhcprime","pepcl_target","ft_match_adaptation","ft_match_ms"}
    missing=required-set(comparison_df["role"])
    if missing: raise ValueError(f"comparison_df is missing roles: {sorted(missing)}")

    def _lr_numeric_string(x):
        if x is None or (isinstance(x,float) and np.isnan(x)): return ""
        if isinstance(x,(int,float,np.integer,np.floating)) and float(x)==0: return "0"
        s=str(x); m=re.fullmatch(r"([0-9.]+)eN(\d+)",s)
        if m: return f"{m.group(1)}e-{int(m.group(2))}"
        try:
            v=float(s)
            if v==0: return "0"
            mant,exp=f"{v:.10e}".split("e"); mant=mant.rstrip("0").rstrip(".")
            return f"{mant}e{int(exp)}"
        except ValueError: return s

    row_map={r["role"]:r for _,r in comparison_df.iterrows()}
    roles=["mhcprime","pepcl_target","ft_match_adaptation","ft_match_ms"] if include_mhcprime else ["pepcl_target","ft_match_adaptation","ft_match_ms"]

    def _default_label(role):
        r=row_map[role]; lr=_lr_numeric_string(r["learning_rate"]); step=int(r["step"])
        if role=="mhcprime": return "MHCPrime"
        base={"pepcl_target":"PepCL","ft_match_adaptation":"FT matched assay adaptation","ft_match_ms":"FT matched MS retention"}[role]
        return f"{base} (LR: {lr}, Steps: {step})"

    labels=[_default_label(r) for r in roles]
    if isinstance(x_labels,dict): labels=[x_labels.get(r,l) for r,l in zip(roles,labels)]
    elif x_labels is not None:
        if len(x_labels)!=len(roles): raise ValueError(f"x_labels must contain {len(roles)} labels.")
        labels=list(x_labels)
    labels=[textwrap.fill(str(x),width=wrap_width) if wrap_width else str(x) for x in labels]

    pair_width=2*bar_width+within_pair_spacing
    centers=np.arange(len(roles))*(pair_width+pair_spacing)
    ms_x=centers-(bar_width+within_pair_spacing)/2
    assay_x=centers+(bar_width+within_pair_spacing)/2

    ms_vals=np.array([float(row_map[r]["ms_performance"]) for r in roles])
    assay_vals=np.array([float(row_map[r]["assay_performance"]) for r in roles])

    fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)
    ax.bar(ms_x,ms_vals,width=bar_width,color=ms_color,edgecolor=bar_edgecolor,linewidth=bar_linewidth)
    ax.bar(assay_x,assay_vals,width=bar_width,color=assay_color,edgecolor=bar_edgecolor,linewidth=bar_linewidth)

    if uncertainty is not None:
        ms_err=np.array([float(row_map[r][f"ms_{uncertainty}"]) for r in roles])
        assay_err=np.array([float(row_map[r][f"assay_{uncertainty}"]) for r in roles])

        valid=np.isfinite(ms_err)
        ax.vlines(ms_x[valid],ms_vals[valid]-ms_err[valid],ms_vals[valid]+ms_err[valid],
                  color=uncertainty_color,linewidth=uncertainty_linewidth,zorder=5)

        valid=np.isfinite(assay_err)
        ax.vlines(assay_x[valid],assay_vals[valid]-assay_err[valid],assay_vals[valid]+assay_err[valid],
                  color=uncertainty_color,linewidth=uncertainty_linewidth,zorder=5)

    ref=row_map["mhcprime"]
    if not include_mhcprime:
        ax.axhline(float(ref["ms_performance"]),color=mhcprime_ms_color,
                   linestyle=mhcprime_ms_linestyle,linewidth=mhcprime_ms_linewidth)
        ax.axhline(float(ref["assay_performance"]),color=mhcprime_assay_color,
                   linestyle=mhcprime_assay_linestyle,linewidth=mhcprime_assay_linewidth)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels,rotation=xtick_rotation,ha="right" if xtick_rotation else "center",fontsize=xtick_fontsize)
    ax.tick_params(axis="y",labelsize=ytick_fontsize)

    if y_lim is not None: ax.set_ylim(y_lim)
    ax.set_title(title,fontsize=title_fontsize)
    ax.set_xlabel(xlabel,fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel,fontsize=ylabel_fontsize)

    if legend:
        legend_labels_default={
            "ms":"MS","assay":"Assay",
            "mhcprime_ms":"MHCPrime MS","mhcprime_assay":"MHCPrime assay",
        }
        if legend_labels is not None: legend_labels_default.update(legend_labels)

        default_handles={
            "ms":Line2D([0],[0],marker="s",linestyle="none",markersize=legend_square_markersize,
                        markerfacecolor=ms_color,markeredgecolor=bar_edgecolor,
                        markeredgewidth=bar_linewidth,label=legend_labels_default["ms"]),
            "assay":Line2D([0],[0],marker="s",linestyle="none",markersize=legend_square_markersize,
                           markerfacecolor=assay_color,markeredgecolor=bar_edgecolor,
                           markeredgewidth=bar_linewidth,label=legend_labels_default["assay"]),
            "mhcprime_ms":Line2D([0],[0],color=mhcprime_ms_color,linestyle=mhcprime_ms_linestyle,
                                 linewidth=legend_mhcprime_ms_linewidth,label=legend_labels_default["mhcprime_ms"]),
            "mhcprime_assay":Line2D([0],[0],color=mhcprime_assay_color,linestyle=mhcprime_assay_linestyle,
                                    linewidth=legend_mhcprime_assay_linewidth,label=legend_labels_default["mhcprime_assay"]),
        }

        if legend_handles is not None:
            for key,handle in legend_handles.items():
                if key not in default_handles:
                    raise ValueError(f"Unknown legend handle key '{key}'. Use {list(default_handles)}.")
                default_handles[key]=handle

        handle_keys=["ms","assay"]
        if not include_mhcprime: handle_keys+=["mhcprime_ms","mhcprime_assay"]
        handles=[default_handles[k] for k in handle_keys]

        leg=ax.legend(
            handles=handles,loc=legend_loc,bbox_to_anchor=legend_bbox,
            ncol=min(legend_ncol,len(handles)),frameon=legend_frameon,
            fontsize=legend_fontsize,title=legend_title,
            columnspacing=legend_columnspacing,handletextpad=legend_handletextpad,borderaxespad=0
        )
        if legend_title is not None: leg.get_title().set_fontsize(legend_title_fontsize)

    fig.subplots_adjust(bottom=bottom_margin)
    if show_plot: plt.show()
    return fig,ax

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

def summarize_frequency_hyperparameters_relative_to_optimal(
    validation_dfs,
    optimal_points,
    frequency_df,
    score_cols_by_dataset,
    method_prefixes_by_dataset,
    method,
    optimal_method_key=None,
    dataset_ms_key="ms_val",
    dataset_assay_key="assay_val",
    method_col="method",
    lr_col="learning_rate",
    step_col="checkpoint",
    frequency_col="frequency",
    frequency_filter=None,
    ref_col="mhcprime",
    min_seeds=1,
    focus_seeds=None,
    verbose=True,
):
    """
    For every selected LR x step combination in frequency_df for one method,
    evaluate that combination on every dataset relative to that dataset's
    method-specific optimal operating point.

    Parameters
    ----------
    validation_dfs : dict
        {
            dataset:{
                "ms_val": ms_validation_df,
                "assay_val": assay_validation_df,
            },
            ...
        }

    optimal_points : dict
        {
            dataset:{
                "ft": (lr, step),
                "pepcl": (lr, step),
            },
            ...
        }

        Tuples may optionally be:
            (lr, step, step_budget)

    frequency_df : pd.DataFrame
        Output from summarize_overlap_hyperparameter_frequency. May contain
        multiple methods; it is filtered to `method`.

    score_cols_by_dataset : dict
        dataset -> list of model score columns.

    method_prefixes_by_dataset : dict
        dataset -> method-prefix dictionary.

    method : str
        Method name used in model columns/frequency_df, e.g.
        "ft" or "pepcl_r_dr".

    optimal_method_key : str or None
        Key used inside optimal_points.

        Example:
            method="pepcl_r_dr", optimal_method_key="pepcl"
            method="ft",         optimal_method_key="ft"

        If None, uses `method`.

    frequency_filter : iterable[int] or int or None
        Restrict evaluation to HP combinations having one of these frequencies.

        Examples:
            None      -> evaluate all frequency values
            [4]       -> evaluate only combinations with frequency 4
            [4,3,2]   -> evaluate combinations with frequency 4, 3, or 2
            4         -> same as [4]

    Returns
    -------
    results_by_hp : dict
        hp label -> dataset -> one-row summary as dict.

    long_df : pd.DataFrame
        One row per hyperparameter x dataset.

    assay_recovery_df : pd.DataFrame
        datasets x hyperparameters, values = assay_gain_recovery_pct.

    ms_retention_df : pd.DataFrame
        datasets x hyperparameters, values = ms_retention_vs_optimal_pct.

    col_label_dict : dict
        Column key -> displayed LR, Steps label.

    hp_frequency_df : pd.DataFrame
        Filtered/sorted hyperparameter frequency dataframe actually evaluated.
    """
    import numpy as np
    import pandas as pd

    optimal_method_key=method if optimal_method_key is None else optimal_method_key

    def _lr_float(x):
        return float(str(x).replace("eN","e-"))

    def _lr_normalized(x):
        s=str(x)
        if "eN" in s:
            return s
        try:
            mant,exp=f"{float(x):.12e}".split("e")
            mant=mant.rstrip("0").rstrip(".")
            exp=int(exp)
            return f"{mant}eN{abs(exp)}" if exp<0 else f"{mant}e{exp}"
        except (TypeError,ValueError):
            return s

    def _lr_display(x):
        v=_lr_float(x)
        if v==0:
            return "0"
        mant,exp=f"{v:.10e}".split("e")
        return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"

    def _parse_point(point,name):
        if not isinstance(point,(tuple,list)) or len(point) not in {2,3}:
            raise ValueError(
                f"{name} must be (lr, step) or (lr, step, step_budget)."
            )

        lr=point[0]
        step=int(point[1])
        budget=None if len(point)==2 else int(point[2])
        return lr,step,budget

    # --------------------------------------------------------------
    # Validate frequency filter.
    # --------------------------------------------------------------
    if frequency_filter is not None:
        if np.isscalar(frequency_filter):
            frequency_filter=[frequency_filter]

        try:
            frequency_filter=tuple(
                dict.fromkeys(int(x) for x in frequency_filter)
            )
        except (TypeError,ValueError):
            raise TypeError(
                "frequency_filter must be None, an integer, "
                "or an iterable of integers."
            )

        if not frequency_filter:
            raise ValueError("frequency_filter cannot be empty.")

        if any(x<0 for x in frequency_filter):
            raise ValueError("frequency_filter values must be >= 0.")

    # --------------------------------------------------------------
    # Validate dataset dictionaries.
    # --------------------------------------------------------------
    datasets=list(validation_dfs)

    if not datasets:
        raise ValueError("validation_dfs cannot be empty.")

    for dataset in datasets:
        if dataset not in optimal_points:
            raise ValueError(
                f"Dataset '{dataset}' missing from optimal_points."
            )

        if dataset not in score_cols_by_dataset:
            raise ValueError(
                f"Dataset '{dataset}' missing from score_cols_by_dataset."
            )

        if dataset not in method_prefixes_by_dataset:
            raise ValueError(
                f"Dataset '{dataset}' missing from method_prefixes_by_dataset."
            )

        if (
            dataset_ms_key not in validation_dfs[dataset]
            or dataset_assay_key not in validation_dfs[dataset]
        ):
            raise ValueError(
                f"validation_dfs['{dataset}'] must contain "
                f"'{dataset_ms_key}' and '{dataset_assay_key}'."
            )

        if optimal_method_key not in optimal_points[dataset]:
            raise ValueError(
                f"optimal_points['{dataset}'] missing key "
                f"'{optimal_method_key}'."
            )

        if method not in method_prefixes_by_dataset[dataset]:
            raise ValueError(
                f"Method '{method}' missing from "
                f"method_prefixes_by_dataset['{dataset}']."
            )

    # --------------------------------------------------------------
    # Filter frequency dataframe to requested method.
    # --------------------------------------------------------------
    hp=frequency_df.copy()

    if method_col in hp.columns:
        hp=hp[hp[method_col]==method].copy()

    required=[lr_col,step_col,frequency_col]
    missing=[c for c in required if c not in hp.columns]

    if missing:
        raise ValueError(
            f"frequency_df is missing required columns: {missing}"
        )

    if hp.empty:
        raise ValueError(
            f"No frequency rows found for method '{method}'."
        )

    # --------------------------------------------------------------
    # Optional frequency filtering.
    # --------------------------------------------------------------
    if frequency_filter is not None:
        before=len(hp)

        hp=hp[
            hp[frequency_col].astype(int).isin(frequency_filter)
        ].copy()

        if verbose:
            print(
                f"Frequency filter {list(frequency_filter)}: "
                f"{before} -> {len(hp)} HP combinations"
            )

        if hp.empty:
            raise ValueError(
                f"No hyperparameter combinations remain for method '{method}' "
                f"after applying frequency_filter={list(frequency_filter)}."
            )

    hp["_lr_float"]=hp[lr_col].map(_lr_float)

    # Frequency high -> low; ties prefer fewer steps, then lower LR.
    hp=(
        hp.sort_values(
            [frequency_col,step_col,"_lr_float"],
            ascending=[False,True,True]
        )
        .drop_duplicates([lr_col,step_col])
        .reset_index(drop=True)
    )

    if verbose:
        print(
            f"\nMethod: {method}"
            f"\nDatasets: {len(datasets)}"
            f"\nHP combinations to evaluate: {len(hp)}"
        )

        show_cols=[lr_col,step_col,frequency_col]
        print("\nHyperparameter combinations:")
        print(hp[show_cols].to_string(index=False))

    # --------------------------------------------------------------
    # Evaluate every selected HP combination on every dataset.
    # --------------------------------------------------------------
    results_by_hp={}
    long_rows=[]
    hp_keys=[]
    col_label_dict={}

    for _,hp_row in hp.iterrows():
        default_lr=hp_row[lr_col]
        default_step=int(hp_row[step_col])
        frequency=int(hp_row[frequency_col])

        hp_key=f"lr_{_lr_normalized(default_lr)}_step_{default_step}"
        hp_label=f"{_lr_display(default_lr)}, {default_step}"

        hp_keys.append(hp_key)
        col_label_dict[hp_key]=hp_label
        results_by_hp[hp_key]={}

        if verbose:
            print(f"\n{'='*90}")
            print(
                f"{method}: LR={_lr_display(default_lr)}, "
                f"Steps={default_step}, Frequency={frequency}"
            )
            print(f"{'='*90}")

        for dataset in datasets:
            optimal_lr,optimal_step,optimal_budget=_parse_point(
                optimal_points[dataset][optimal_method_key],
                f"optimal_points['{dataset}']['{optimal_method_key}']"
            )

            summary_df,_=summarize_pepcl_default_operating_point_relative_to_optimal(
                validation_dfs[dataset][dataset_ms_key],
                validation_dfs[dataset][dataset_assay_key],
                score_cols_by_dataset[dataset],
                method_prefixes_by_dataset[dataset],
                optimal_lr=optimal_lr,
                optimal_step=optimal_step,
                default_lr=default_lr,
                default_step=default_step,
                pepcl_method=method,
                ref_col=ref_col,
                optimal_step_budget=optimal_budget,
                default_step_budget=None,
                min_seeds=min_seeds,
                focus_seeds=focus_seeds,
                verbose=False,
            )

            s=summary_df.iloc[0].to_dict()
            results_by_hp[hp_key][dataset]=s

            long_rows.append({
                "method":method,
                "dataset":dataset,
                "hp_key":hp_key,
                "hp_label":hp_label,
                "frequency":frequency,
                "learning_rate":_lr_normalized(default_lr),
                "step":default_step,
                **s,
            })

            if verbose:
                print(
                    f"{dataset}: "
                    f"assay recovery={s['assay_gain_recovery_pct']:.2f}% | "
                    f"MS retention={s['ms_retention_vs_optimal_pct']:.2f}%"
                )

    long_df=pd.DataFrame(long_rows)

    # --------------------------------------------------------------
    # Wide matrices for jitter plotting.
    # --------------------------------------------------------------
    assay_recovery_df=(
        long_df
        .pivot(
            index="dataset",
            columns="hp_key",
            values="assay_gain_recovery_pct"
        )
        .reindex(
            index=datasets,
            columns=hp_keys
        )
    )

    ms_retention_df=(
        long_df
        .pivot(
            index="dataset",
            columns="hp_key",
            values="ms_retention_vs_optimal_pct"
        )
        .reindex(
            index=datasets,
            columns=hp_keys
        )
    )

    hp_frequency_df=hp.drop(columns="_lr_float").copy()

    return (
        results_by_hp,
        long_df,
        assay_recovery_df,
        ms_retention_df,
        col_label_dict,
        hp_frequency_df,

    )
def plot_loss_and_validation_performance_grid(
    loss_dfs, assay_perf_df, ms_perf_df, configs,
    save_path=None,

    # figure
    figsize=(4.8,6.0), dpi=300,
    panel_wspace=0.30, panel_hspace=0.45,

    # layout
    landscape=False,

    # font
    default_font="Nimbus Sans",
    title_fontsize=7, label_fontsize=6,
    tick_fontsize=5.5, legend_fontsize=5.5,
    title_bold=False,

    # axes
    loss_ylabel="Loss",
    performance_ylabel="Performance",
    xlabel="Optimization step",
    remove_top_spine=True, remove_right_spine=True,
    axis_linewidth=0.35, tick_length=2.2, tick_width=0.35,
    xlabel_pad=1.5, ylabel_pad=1.5,

    # title behavior
    title_prefix=None,
    title_decorate="both",  # "both", "loss", "performance", None
    loss_title_suffix=None,
    performance_title_suffix=None,
    show_lr_steps=True,
    lr_label="LR",
    steps_label="Steps",

    # styling dictionaries
    loss_styles=None,
    performance_styles=None,

    # legend
    legend_ncol=4,
    legend_loc="lower center",
    legend_bbox=(0.5,-0.01),
    legend_frame=False,
    legend_handlelength=2.0,
    custom_legend_handles=None,

    # plotting
    sd_alpha=0.18,
    include_mhcprime_step0=False,
    mhcprime_col="mhcprime",
):
    """
    Plot training/validation loss trajectories and validation performance.

    If include_mhcprime_step0=True, prepend the MHCPrime performance
    from `mhcprime_col` to assay/MS validation curves at optimization
    step 0.

    landscape:
        If False (default), lays out one config per ROW, with loss on the
        left column and performance on the right column (portrait pairs).
        If True, lays out loss curves along the TOP row and the matched
        validation performances along the BOTTOM row, one config per column
        (landscape pairs, up/down instead of left/right).

    Expected configs format
    -----------------------
    {
        "ft":{
            "base_title":"FT",
            "prefix":"yd_ft_steps_300_lr_2eN4",
            "method":"ft",
            "lr":"2e-4",
            "steps":300,
        },
        "pepcl":{
            "base_title":"PepCL",
            "prefix":"yd_pepcl_r_dr_steps_200_lr_2eN4",
            "method":"pepcl",
            "lr":"2e-4",
            "steps":200,
        },
        "pepcl_default":{
            "base_title":"PepCL default",
            "prefix":"yd_pepcl_r_dr_steps_300_lr_2eN4",
            "method":"pepcl",
            "lr":"2e-4",
            "steps":300,
        },
    }

    Returns
    -------
    fig, axes, performance_df
    """

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[default_font,"Nimbus Sans L","Arial","Helvetica"]
    matplotlib.rcParams["pdf.fonttype"]=42

    train=loss_dfs["training_summary"]
    val=loss_dfs["validation_summary"]

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    default_loss_styles={
        "FT loss":{
            "color":"#456be8",
            "train_linestyle":"-",
            "val_linestyle":"--",
            "linewidth":1.2,
            "label":"FT loss",
        },
        "MS replay loss":{
            "color":"#ffb55c",
            "train_linestyle":"-",
            "val_linestyle":"--",
            "linewidth":1.2,
            "label":"MS replay loss",
        },
        "Weighted DR loss":{
            "color":"#D34A4D",
            "train_linestyle":"-",
            "val_linestyle":"--",
            "linewidth":1.2,
            "label":"Weighted DR loss",
        },
        "Total loss":{
            "color":"#9b72cf",
            "train_linestyle":"-",
            "val_linestyle":"--",
            "linewidth":1.2,
            "label":"Total loss",
        },
    }

    default_performance_styles={
        "Assay val":{
            "color":"#456be8",
            "linestyle":"-",
            "linewidth":1.2,
            "label":"Assay val. AP",
        },
        "MS val":{
            "color":"#77C4A0",
            "linestyle":"-",
            "linewidth":1.2,
            "label":"MS val. AP",
        },
    }

    def _merge_styles(defaults,user):
        out={k:dict(v) for k,v in defaults.items()}
        if user is not None:
            for k,v in user.items():
                if k not in out: out[k]={}
                out[k].update(v)
        return out

    loss_styles=_merge_styles(default_loss_styles,loss_styles)
    performance_styles=_merge_styles(default_performance_styles,performance_styles)

    # ------------------------------------------------------------------
    # Performance summarization across replicates
    # ------------------------------------------------------------------
    def _performance_summary(df,prefix,dataset):
        rows=[]
        pattern=re.compile(rf"^{re.escape(prefix)}_seed_(\d+)_step_(\d+)$")

        for col in df.columns:
            m=pattern.match(str(col))
            if m:
                rows.append({
                    "replicate":int(m.group(1)),
                    "step":int(m.group(2)),
                    "value":float(df[col].mean()),
                    "dataset":dataset,
                })

        if not rows:
            return pd.DataFrame(columns=["step","dataset","n","mean","sd"])

        rep_df=pd.DataFrame(rows)

        out=(
            rep_df.groupby(["step","dataset"],as_index=False)["value"]
            .agg(n="count",mean="mean",sd="std")
            .sort_values("step")
        )

        # Optional MHCPrime baseline at step 0
        if include_mhcprime_step0:
            if mhcprime_col not in df.columns:
                raise KeyError(
                    f"include_mhcprime_step0=True, but '{mhcprime_col}' "
                    f"was not found in the {dataset} dataframe."
                )

            baseline=float(df[mhcprime_col].mean())
            n_reps=rep_df["replicate"].nunique()

            step0=pd.DataFrame({
                "step":[0],
                "dataset":[dataset],
                "n":[n_reps],
                "mean":[baseline],
                "sd":[0.0],
            })

            out=pd.concat([step0,out],ignore_index=True).sort_values("step")

        return out

    # ------------------------------------------------------------------
    # Titles
    # ------------------------------------------------------------------
    def _build_title(cfg,side):
        if side=="loss":
            base=cfg.get("loss_title",cfg.get("base_title",""))
        else:
            base=cfg.get("performance_title",cfg.get("base_title",""))

        decorated=title_decorate in {"both",side}

        if decorated:
            if title_prefix:
                base=f"{title_prefix} {base}"

            if show_lr_steps:
                lr=cfg.get("lr",None)
                steps=cfg.get("steps",None)

                param_parts=[]
                if lr is not None: param_parts.append(f"{lr_label}: {lr}")
                if steps is not None: param_parts.append(f"{steps_label}: {steps}")

                if param_parts:
                    base=f"{base} ({', '.join(param_parts)})"

        suffix=loss_title_suffix if side=="loss" else performance_title_suffix
        if suffix:
            base=f"{base} {suffix}"

        return base

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    if landscape:
        fig,axes=plt.subplots(
            2,len(configs),figsize=figsize,dpi=dpi,squeeze=False,
            gridspec_kw={"wspace":panel_wspace,"hspace":panel_hspace},
        )
    else:
        fig,axes=plt.subplots(
            len(configs),2,figsize=figsize,dpi=dpi,squeeze=False,
            gridspec_kw={"wspace":panel_wspace,"hspace":panel_hspace},
        )

    perf_dfs=[]

    for idx,(config_key,cfg) in enumerate(configs.items()):
        method=cfg["method"]
        prefix=cfg.get("score_prefix",cfg.get("prefix"))
        if landscape:
            ax_loss=axes[0,idx]
            ax_perf=axes[1,idx]
        else:
            ax_loss,ax_perf=axes[idx]

        # ==============================================================
        # Losses
        # ==============================================================
        if method=="ft":
            components=[
                ("assay_loss_raw","FT loss"),
            ]
        else:
            components=[
                ("total_loss","Total loss"),
                ("assay_loss_raw","FT loss"),
                ("ms_replay_loss_raw","MS replay loss"),
                ("drift_loss_weighted","Weighted DR loss"),
            ]

        for loss_name,label in components:
            style=loss_styles[label]

            tr=train[
                (train["config"]==config_key)&
                (train["loss_name"]==loss_name)
            ].sort_values("step")

            va=val[
                (val["config"]==config_key)&
                (val["loss_name"]==loss_name)
            ].sort_values("step")

            if not tr.empty:
                ax_loss.plot(
                    tr["step"],tr["mean"],
                    color=style.get("color"),
                    linestyle=style.get("train_linestyle","-"),
                    linewidth=style.get("linewidth",1.2),
                    alpha=style.get("alpha",1.0),
                )

            if not va.empty:
                ax_loss.plot(
                    va["step"],va["mean"],
                    color=style.get("val_color",style.get("color")),
                    linestyle=style.get("val_linestyle","--"),
                    linewidth=style.get("val_linewidth",style.get("linewidth",1.2)),
                    alpha=style.get("val_alpha",style.get("alpha",1.0)),
                    marker=style.get("val_marker",None),
                    markersize=style.get("val_markersize",3),
                )

        # ==============================================================
        # Validation performance
        # ==============================================================
        assay_perf=_performance_summary(assay_perf_df,prefix,"Assay val")

        if not assay_perf.empty:
            assay_perf=assay_perf.assign(config=config_key)
            perf_dfs.append(assay_perf)

            s=performance_styles["Assay val"]

            ax_perf.plot(
                assay_perf["step"],assay_perf["mean"],
                color=s.get("color"),
                linestyle=s.get("linestyle","-"),
                linewidth=s.get("linewidth",1.2),
                marker=s.get("marker",None),
                markersize=s.get("markersize",3),
                alpha=s.get("alpha",1.0),
            )

            ax_perf.fill_between(
                assay_perf["step"],
                assay_perf["mean"]-assay_perf["sd"],
                assay_perf["mean"]+assay_perf["sd"],
                color=s.get("color"),
                alpha=s.get("sd_alpha",sd_alpha),
                linewidth=0,
            )

        if method=="pepcl":
            ms_perf=_performance_summary(ms_perf_df,prefix,"MS val")

            if not ms_perf.empty:
                ms_perf=ms_perf.assign(config=config_key)
                perf_dfs.append(ms_perf)

                s=performance_styles["MS val"]

                ax_perf.plot(
                    ms_perf["step"],ms_perf["mean"],
                    color=s.get("color"),
                    linestyle=s.get("linestyle","-"),
                    linewidth=s.get("linewidth",1.2),
                    marker=s.get("marker",None),
                    markersize=s.get("markersize",3),
                    alpha=s.get("alpha",1.0),
                )

                ax_perf.fill_between(
                    ms_perf["step"],
                    ms_perf["mean"]-ms_perf["sd"],
                    ms_perf["mean"]+ms_perf["sd"],
                    color=s.get("color"),
                    alpha=s.get("sd_alpha",sd_alpha),
                    linewidth=0,
                )

        # ==============================================================
        # Titles / axes
        # ==============================================================
        ax_loss.set_title(
            _build_title(cfg,"loss"),
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight="bold" if title_bold else "normal",
        )

        ax_perf.set_title(
            _build_title(cfg,"performance"),
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight="bold" if title_bold else "normal",
        )

        for ax,ylabel in [(ax_loss,loss_ylabel),(ax_perf,performance_ylabel)]:
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

            ax.spines["left"].set_linewidth(axis_linewidth)
            ax.spines["bottom"].set_linewidth(axis_linewidth)
            ax.spines["top"].set_visible(not remove_top_spine)
            ax.spines["right"].set_visible(not remove_right_spine)

            if not remove_top_spine:
                ax.spines["top"].set_linewidth(axis_linewidth)

            if not remove_right_spine:
                ax.spines["right"].set_linewidth(axis_linewidth)

            ax.tick_params(
                axis="both",
                which="major",
                length=tick_length,
                width=tick_width,
                direction="out",
                labelsize=tick_fontsize,
            )

            for lbl in ax.get_xticklabels()+ax.get_yticklabels():
                lbl.set_fontfamily(default_font)

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    if custom_legend_handles is None:
        handles=[]

        for key in ["FT loss","MS replay loss","Weighted DR loss","Total loss"]:
            s=loss_styles[key]

            handles.append(Line2D(
                [0],[0],
                color=s.get("color"),
                linewidth=s.get("linewidth",1.2),
                linestyle="-",
                label=s.get("label",key),
            ))

        handles.extend([
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
        ])

        for key in ["Assay val","MS val"]:
            s=performance_styles[key]

            handles.append(Line2D(
                [0],[0],
                color=s.get("color"),
                linewidth=s.get("linewidth",1.2),
                linestyle=s.get("linestyle","-"),
                label=s.get("label",key),
            ))

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

    # leave room for bottom legend
    fig.subplots_adjust(bottom=0.12)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
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

    performance_df=pd.concat(perf_dfs,ignore_index=True) if perf_dfs else pd.DataFrame()

    return fig,axes,performance_df

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

def early_stopping_df_to_dict(early_stop_df):
    """Convert true early-stopping rows from evaluate_ft_early_stopping() to indexed config dict."""
    def _lr_float(x):
        if isinstance(x,(int,float,np.integer,np.floating)): return float(x)
        s=str(x)
        if "eN" in s:
            a,b=s.split("eN"); return float(a)*10**(-int(b))
        return float(s)

    required=["stopped_early","learning_rate","step_budget","stop_step"]
    missing=[c for c in required if c not in early_stop_df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

    d=early_stop_df[early_stop_df["stopped_early"].fillna(False)].copy().reset_index(drop=True)

    return {
        i:{
            "lr":_lr_float(r["learning_rate"]),
            "steps":int(r["step_budget"]),
            "stop":int(r["stop_step"]),
        }
        for i,(_,r) in enumerate(d.iterrows())
    }

import numpy as np
import pandas as pd

def balance_across_dataset_sources(df, allele_col="allele", label_col="label", source_col="dataset_source", seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    work = df.reset_index(drop=True)

    sources = list(pd.unique(work[source_col]))
    if len(sources) != 2:
        raise ValueError(f"expected exactly 2 dataset sources, found {len(sources)}: {sources}")

    keep_idx = []
    for (allele, label), grp in work.groupby([allele_col, label_col], sort=False):
        counts = {s: grp.index[grp[source_col] == s].to_numpy() for s in sources}
        n_each = min(len(counts[sources[0]]), len(counts[sources[1]]))
        if n_each == 0:
            if verbose:
                print(f"[SKIP] allele={allele} label={label}: one source has 0 (counts {sources[0]}={len(counts[sources[0]])}, {sources[1]}={len(counts[sources[1]])})")
            continue
        for s in sources:
            pool = counts[s]
            chosen = pool[rng.permutation(len(pool))[:n_each]]
            keep_idx.append(chosen)
            if verbose and len(pool) != n_each:
                print(f"allele={allele} label={label} source={s}: {len(pool)} -> {n_each} (sampled down)")

    keep_idx = np.concatenate(keep_idx) if keep_idx else np.array([], dtype=int)
    return work.iloc[np.sort(keep_idx)].copy()

def format_numeric_columns_two_decimals(df):
    df=df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col]=df[col].map(lambda x:f"{x:.2f}" if pd.notna(x) else x)

    return df

def format_pepcl_compute_summary_df(resource_results, method_order=None):
    """
    Convert the output of summarize_pepcl_compute_logs() into the compact
    resource-summary dataframe used for the manuscript table.

    Returns columns:
        Method
        Optimizer steps
        Wall-clock time (s)
        Throughput (steps/s)
        Peak GPU memory (GB)
        Peak host memory (GB)
    """
    import pandas as pd

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

    return pd.DataFrame({
        "Method":summary["method"].astype(str),
        "Optimizer steps":summary["optimizer_steps"],
        "Wall-clock time (s)":summary["training_loop_s"],
        "Throughput (steps/s)":summary["steps_per_s"],
        "Peak GPU memory (GB)":summary["peak_gpu_allocated_gb"],
        "Peak host memory (GB)":summary["peak_process_tree_ram_gb"],
    })

def plot_pepcl_compute_summary_table_df(
    resource_df,
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
    Plot a compact compute/resource summary table from a formatted dataframe.

    Expected columns
    ----------------
    Method
    Optimizer steps
    Wall-clock time (s)
    Throughput (steps/s)
    Peak GPU memory (GB)
    Peak host memory (GB)

    All numeric values are displayed with exactly `decimals` decimal places.

    Returns
    -------
    fig, ax, tbl, display_df
    """
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    required=[
        "Method",
        "Optimizer steps",
        "Wall-clock time (s)",
        "Throughput (steps/s)",
        "Peak GPU memory (GB)",
        "Peak host memory (GB)",
    ]
    missing=[c for c in required if c not in resource_df.columns]
    if missing:
        raise ValueError(f"resource_df is missing columns: {missing}")

    df=resource_df[required].copy()

    if method_order is not None:
        method_order=list(method_order)
        missing_methods=[m for m in method_order if m not in set(df["Method"])]
        if missing_methods:
            raise ValueError(f"Methods not found in resource_df: {missing_methods}")

        df=df.set_index("Method").loc[method_order].reset_index()

    def _fmt(x):
        return "–" if pd.isna(x) else f"{float(x):.{decimals}f}"

    # Format specifically for rendering.
    display_df=pd.DataFrame({
        "Method":df["Method"].astype(str),
        "Optimizer\nsteps":[_fmt(x) for x in df["Optimizer steps"]],
        "Wall-clock\ntime (s)":[_fmt(x) for x in df["Wall-clock time (s)"]],
        "Throughput\n(steps/s)":[_fmt(x) for x in df["Throughput (steps/s)"]],
        "Peak GPU\nmemory (GB)":[_fmt(x) for x in df["Peak GPU memory (GB)"]],
        "Peak host\nmemory (GB)":[_fmt(x) for x in df["Peak host memory (GB)"]],
    })

    columns=list(display_df.columns)

    if col_widths is None:
        col_widths=[0.13,0.13,0.17,0.17,0.20,0.20]

    if len(col_widths)!=len(columns):
        raise ValueError(f"col_widths must contain {len(columns)} values.")

    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,"Nimbus Sans","Nimbus Sans L","Arial","Helvetica"
    ]

    fp_body=FontProperties(family=font_family,size=body_fontsize)
    fp_header=FontProperties(family=font_family,size=header_fontsize,weight="bold")

    fig,ax=plt.subplots(figsize=fig_size,dpi=dpi)
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

    # Remove native borders/backgrounds.
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

    base_header_height=tbl[(0,0)].get_height()
    for c in range(ncols):
        tbl[(0,c)].set_height(base_header_height*header_height_scale)

    fig.canvas.draw()

    # Independent horizontal rules.
    left=tbl[(0,0)].get_x()
    right=tbl[(0,ncols-1)].get_x()+tbl[(0,ncols-1)].get_width()
    header_top=tbl[(0,0)].get_y()+tbl[(0,0)].get_height()
    header_bottom=tbl[(0,0)].get_y()
    bottom=tbl[(nrows-1,0)].get_y()

    def _rule(y,lw,alpha=1.0):
        if independent_lines:
            ax.add_line(
                Line2D(
                    [left,right],[y,y],
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
                y,left,right,
                transform=ax.transAxes,
                color="black",
                linewidth=lw,
                alpha=alpha,
                zorder=5,
            )

    _rule(header_top,top_linewidth)
    _rule(header_bottom,header_linewidth)

    for r in range(1,nrows-1):
        _rule(tbl[(r,0)].get_y(),row_linewidth,row_line_alpha)

    _rule(bottom,bottom_linewidth)

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf",format="pdf",dpi=dpi,bbox_inches="tight")
        fig.savefig(f"{save_path}.png",format="png",dpi=dpi,bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig,ax,tbl,display_df