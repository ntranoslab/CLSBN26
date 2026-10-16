from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from scipy import stats
from scipy.stats import gaussian_kde, wilcoxon, mannwhitneyu
from sklearn.metrics import average_precision_score
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logomaker
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.transforms
from matplotlib.cm import ScalarMappable
import re
import math
from matplotlib.patches import Rectangle

from .general_utils import *

def global_min_max(dfs, col, min_offset=0, max_offset=0):
    gmin = min(df[col].min() for df in dfs)
    gmax = max(df[col].max() for df in dfs)
    return (gmin - min_offset, gmax + max_offset)

def plot_publication_dataset_table(
    df: pd.DataFrame,
    dataset_meta: dict,
    col_widths: list[float],
    save_path: str = None,
    title: str = "Dataset summary statistics",
):
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if len(col_widths) != len(df.columns):
        raise ValueError(
            f"col_widths must have length {len(df.columns)}, got {len(col_widths)}."
        )

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Nimbus Sans",
        "Nimbus Sans L",
        "NimbusSanL",
        "DejaVu Sans",
    ]

    dataset_col = df.columns[0]
    value_cols = list(df.columns[1:])

    df = df.copy()

    meta_order = list(dataset_meta.keys())
    df = df[df[dataset_col].isin(meta_order)].copy()

    if df[dataset_col].duplicated().any():
        dupes = df.loc[df[dataset_col].duplicated(), dataset_col].tolist()
        raise ValueError(
            f"Duplicate dataset keys found in dataframe: {dupes}. "
            "Each dataset key should appear at most once."
        )

    df = df.set_index(dataset_col).loc[meta_order].reset_index()

    display_df = df.copy()

    row_types = []
    rendered_labels = []

    subgroup_prefix = "    "

    for raw_name in display_df[dataset_col]:
        row_type, display_name = dataset_meta[raw_name]
        row_types.append(row_type)

        if row_type == "subgroup":
            rendered_labels.append(f"{subgroup_prefix}{display_name}")
        else:
            rendered_labels.append(display_name)

    display_df[dataset_col] = rendered_labels

    for col in value_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{int(x):,}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
        )

    col_labels = list(display_df.columns)
    cell_text = display_df.values.tolist()

    fp_body = FontProperties(family="Nimbus Sans", size=13)
    fp_header = FontProperties(family="Nimbus Sans", size=13, weight="bold")
    fp_title = FontProperties(family="Nimbus Sans", size=18)

    n_data_rows = len(display_df)
    fig_height = max(2.8, 0.34 * (n_data_rows + 2))
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    ax.axis("off")

    ax.text(
        0.0,
        1.04,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontproperties=fp_title,
    )

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.02, 1.0, 0.94],
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.0, 1.45)

    nrows = len(cell_text) + 1
    ncols = len(col_labels)

    for (r, c), cell in tbl.get_celld().items():
        cell.visible_edges = ""
        cell.set_linewidth(0.0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)

    for c in range(ncols):
        cell = tbl[(0, c)]
        cell.get_text().set_fontproperties(fp_header)
        cell.set_height(cell.get_height() * 1.15)

    for r in range(nrows):
        cell = tbl[(r, 0)]
        cell.get_text().set_ha("left")
        cell.PAD = 0.03

    for r in range(nrows):
        for c in range(1, ncols):
            tbl[(r, c)].get_text().set_ha("center")

    for i, row_type in enumerate(row_types):
        table_r = i + 1
        if row_type == "group":
            for c in range(ncols):
                tbl[(table_r, c)].get_text().set_weight("bold")

    top_lw = 1.6
    mid_lw = 0.8
    bottom_lw = 1.6

    for c in range(ncols):
        cell = tbl[(0, c)]
        cell.visible_edges = "TB"
        cell.set_linewidth(top_lw)

    fig.canvas.draw()

    header_left = tbl[(0, 0)].get_x()
    header_right = tbl[(0, ncols - 1)].get_x() + tbl[(0, ncols - 1)].get_width()
    header_bottom = tbl[(0, 0)].get_y()
    table_bottom = tbl[(nrows - 1, 0)].get_y()

    ax.hlines(
        y=header_bottom,
        xmin=header_left,
        xmax=header_right,
        transform=ax.transAxes,
        linewidth=3.0,
        color="white",
        zorder=5,
    )
    ax.hlines(
        y=header_bottom,
        xmin=header_left,
        xmax=header_right,
        transform=ax.transAxes,
        linewidth=mid_lw,
        color="black",
        zorder=6,
    )

    ax.hlines(
        y=table_bottom,
        xmin=header_left,
        xmax=header_right,
        transform=ax.transAxes,
        linewidth=bottom_lw,
        color="black",
        zorder=6,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_jitter_ms_vs_external(
    df_dict,
    score_cols,
    col_label_dict=None,
    colors=None,
    ref_key='MS',
    label_col='label',
    allele_col='allele',
    seq_len_col='seq_len',
    *,
    figsize=(10, 5),
    dpi=150,
    default_font='Nimbus Sans',
    jitter=0.2,
    point_size=40,
    point_alpha=1.0,
    point_edge_color='none',
    point_edge_width=0,
    point_marker='o',
    ref_marker='_',
    ref_marker_size=40,
    ref_line_width=2.0,
    t_cap_width=0.02,
    t_line_width=1.0,
    t_cap_line_width=None,
    show_mean=True,
    mean_above_points=False,
    mean_color='black',
    mean_width=1.2,
    mean_style='--',
    mean_length_pad=0.05,
    title=None,
    title_fontsize=14,
    title_bold=False,
    xlabel=None,
    ylabel=None,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    xtick_labelsize=10,
    ytick_labelsize=10,
    xtick_rotation=0,
    xtick_ha='center',
    despine=True,
    spine_width=None,
    tick_length=None,
    tick_width=None,
    tick_direction='out',
    ylim=None,
    grid_alpha=0.0,
    hide_legend=False,
    legend_fontsize=8,
    legend_loc='upper center',
    legend_bbox=(0.5, -0.15),
    legend_ncol=4,
    legend_frameon=False,
    square_aspect=False,
    save_path=None,
    show=True,
):

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if col_label_dict is None:
        col_label_dict = {c: c for c in score_cols}

    default_color_cycle = [mcolors.to_hex(c) for c in plt.cm.tab10.colors]
    if colors is not None:
        score_col_colors = {c: colors[i % len(colors)] for i, c in enumerate(score_cols)}
    else:
        score_col_colors = {c: default_color_cycle[i % len(default_color_cycle)] for i, c in enumerate(score_cols)}

    _t_cap_line_width = t_cap_line_width if t_cap_line_width is not None else t_line_width

    ref_df = df_dict[ref_key]
    external_keys = [k for k in df_dict.keys() if k != ref_key]

    plot_data_records = []
    summary_records = []
    ext_mean_aps = {}
    ref_mean_aps = {}

    for ext_key in tqdm(external_keys, desc="Processing datasets"):
        ext_df = df_dict[ext_key]

        ext_combos = set(zip(ext_df[allele_col], ext_df[seq_len_col]))
        ref_mask = pd.Series(list(zip(ref_df[allele_col], ref_df[seq_len_col])),index=ref_df.index).isin(ext_combos)
        ext_mask = pd.Series(list(zip(ext_df[allele_col], ext_df[seq_len_col])),index=ext_df.index).isin(set(zip(ref_df[allele_col], ref_df[seq_len_col])))

        ref_filtered = ref_df[ref_mask]
        ext_filtered = ext_df[ext_mask]

        n_shared = len(set(zip(ref_filtered[allele_col], ref_filtered[seq_len_col])))
        summary_records.append({'dataset': ext_key, 'n_shared_allele_seqlen': n_shared})

        ref_ap = compute_ap_matrix_optimized(ref_filtered, score_cols,
                                              label_col=label_col, allele_col=allele_col,
                                              disable_tqdm=True)
        ext_ap = compute_ap_matrix_optimized(ext_filtered, score_cols,
                                              label_col=label_col, allele_col=allele_col,
                                              disable_tqdm=True)

        ref_means = ref_ap.mean()
        ext_means = ext_ap.mean()
        ref_mean_aps[ext_key] = ref_means
        ext_mean_aps[ext_key] = ext_means

        for sc in score_cols:
            plot_data_records.append({
                'dataset': ext_key,
                'score_col': sc,
                f'{ext_key}_mAP': ext_means.get(sc, np.nan),
                f'{ref_key}_mAP': ref_means.get(sc, np.nan),
            })

    plot_data_df = pd.DataFrame(plot_data_records)
    summary_df = pd.DataFrame(summary_records)

    n_score_cols = len(score_cols)
    if jitter > 0 and n_score_cols > 1:
        offsets = np.linspace(-jitter / 2, jitter / 2, n_score_cols)
    else:
        offsets = np.zeros(n_score_cols)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    mean_zorder = 4 if mean_above_points else 2
    point_zorder = 3
    legend_cols_seen = set()

    for x_pos, ext_key in enumerate(external_keys):
        for si, (sc, offset) in enumerate(zip(score_cols, offsets)):
            ext_val = float(ext_mean_aps[ext_key].get(sc, np.nan))
            ref_val = float(ref_mean_aps[ext_key].get(sc, np.nan))
            color = score_col_colors[sc]
            x = x_pos + offset

            if not np.isnan(ext_val):
                label = None
                if not hide_legend and sc not in legend_cols_seen:
                    label = col_label_dict.get(sc, sc)
                    legend_cols_seen.add(sc)

                ax.scatter(x, ext_val, color=color, s=point_size, alpha=point_alpha,
                           marker=point_marker,
                           edgecolors=point_edge_color if point_edge_color != 'none' else 'none',
                           linewidths=point_edge_width, zorder=point_zorder, label=label)

            if not np.isnan(ref_val):
                if ref_marker == 'T':
                    ax.plot([x - t_cap_width, x + t_cap_width], [ref_val, ref_val],
                            color=color, linewidth=_t_cap_line_width,
                            solid_capstyle='butt', zorder=point_zorder)
                    if not np.isnan(ext_val):
                        ax.plot([x, x], [ext_val, ref_val],
                                color=color, linewidth=t_line_width,
                                solid_capstyle='butt', zorder=point_zorder - 1,
                                alpha=0.5)

                elif ref_marker == 'Tv':
                    ax.scatter(x, ref_val, color=color, s=point_size,
                               marker='v', linewidths=0,
                               zorder=point_zorder, alpha=point_alpha)
                    if not np.isnan(ext_val):
                        ax.plot([x, x], [ext_val, ref_val],
                                color=color, linewidth=t_line_width,
                                solid_capstyle='butt', zorder=point_zorder - 1,
                                alpha=0.5)

                else:
                    ax.scatter(x, ref_val, color=color, s=ref_marker_size,
                               marker=ref_marker, linewidths=ref_line_width,
                               zorder=point_zorder, alpha=point_alpha)

        if show_mean:
            col_vals = [float(ext_mean_aps[ext_key].get(sc, np.nan))
                        for sc in score_cols
                        if not np.isnan(float(ext_mean_aps[ext_key].get(sc, np.nan)))]
            if col_vals:
                mean_val = np.mean(col_vals)
                half_extent = jitter / 2 + mean_length_pad
                ax.plot([x_pos - half_extent, x_pos + half_extent],
                        [mean_val, mean_val],
                        color=mean_color, linewidth=mean_width,
                        linestyle=mean_style, zorder=mean_zorder)

    ax.set_xticks(range(len(external_keys)))
    ax.set_xticklabels(external_keys, rotation=xtick_rotation, ha=xtick_ha,
                       fontsize=xtick_labelsize)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(default_font)

    ax.tick_params(axis='y', labelsize=ytick_labelsize)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font,
                      labelpad=xlabel_pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font,
                      labelpad=ylabel_pad)

    if title is not None:
        title_weight = 'bold' if title_bold else 'normal'
        ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font,
                     fontweight=title_weight)

    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if spine_width is not None:
        for side in ['left', 'bottom', 'top', 'right']:
            if ax.spines[side].get_visible():
                ax.spines[side].set_linewidth(spine_width)

    tick_kw = {'direction': tick_direction}
    if tick_length is not None:
        tick_kw['length'] = tick_length
    if tick_width is not None:
        tick_kw['width'] = tick_width
    ax.tick_params(axis='both', which='major', **tick_kw)

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, axis='y', zorder=0)

    if square_aspect:
        ax.set_box_aspect(1)

    if not hide_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if show_mean:
                handles.append(Line2D([0], [0], color=mean_color, linewidth=mean_width,
                                      linestyle=mean_style))
                labels.append('Mean')
            if ref_marker == 'T':
                handles.append(Line2D([0], [0], color='gray', linewidth=_t_cap_line_width))
            elif ref_marker == 'Tv':
                handles.append(Line2D([0], [0], color='gray', marker='v',
                                      linewidth=0, markersize=8))
            else:
                handles.append(Line2D([0], [0], color='gray', marker=ref_marker,
                                      linewidth=0, markersize=8,
                                      markeredgewidth=ref_line_width))
            labels.append(ref_key)
            leg = fig.legend(handles, labels, loc=legend_loc, bbox_to_anchor=legend_bbox,
                             ncol=legend_ncol, frameon=legend_frameon,
                             fontsize=legend_fontsize)
            for text in leg.get_texts():
                text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return ax, plot_data_df, summary_df


def plot_model_performance_bars_or_box_v3(
    df,
    model_col,
    score_col,
    model_groups,
    x_ticks,
    colors=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    bar_width=0.8,
    group_spacing=0.3,
    bar_spacing=0.0,
    bar_outline_color='black',
    bar_outline_width=1.0,
    mode='bar',
    interval_df=None,
    interval_low_row='ci_low',
    interval_high_row='ci_high',
    interval_color='black',
    interval_linewidth=0.8,
    interval_capsize=0.0,
    interval_capthick=None,
    interval_alpha=1.0,
    interval_zorder=4,
    box_flier_marker='o',
    box_flier_size=3,
    box_flier_color=None,
    box_flier_alpha=0.6,
    box_median_color='black',
    box_median_width=1.5,
    box_whisker_style='-',
    box_whisker_width=1.0,
    box_notch=False,
    box_patch_alpha=0.9,
    pval_df=None,
    pval_comparisons=None,
    pval_pairs=None,
    pval_base_only=False,
    pval_base_col=None,
    pval_merge_nonoverlapping=True,
    pval_format='stars',
    pval_fontsize=5,
    pval_linewidth=0.35,
    pval_color='black',
    pval_bar_offset=0.035,
    pval_bar_spacing=0.055,
    pval_text_offset=0.004,
    pval_star_text_offset=0.001,
    pval_ns_text_offset=0.004,
    pval_outside_axes=False,
    reference_hlines=None,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    x_margin=None,
    xtick_rotation=0,
    edge_margin=None,
    legend_mode='separate',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    ax=None,
    show=True,
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'title_y': 1.04,
        'title_color': 'black',
        'title_bbox': None,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (10, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    if reference_hlines is None:
        reference_hlines = {}

    if mode not in {'bar', 'box'}:
        raise ValueError("mode must be 'bar' or 'box'.")

    if interval_df is not None:
        if not isinstance(interval_df, pd.DataFrame):
            raise TypeError("interval_df must be a pandas DataFrame or None.")
        missing_rows = [r for r in (interval_low_row, interval_high_row) if r not in interval_df.index]
        if missing_rows:
            raise ValueError(
                f"interval_df is missing required interval rows: {missing_rows}. "
                f"Expected index labels {interval_low_row!r} and {interval_high_row!r}."
            )

    if x_ticks is None or len(x_ticks) == 0:
        hide_x_ticks = True
        n_orig = max(len(model_list) for model_list in model_groups.values())
        x_ticks_working = [''] * n_orig
    else:
        hide_x_ticks = all(str(tick) == '' for tick in x_ticks)
        x_ticks_working = x_ticks

    def _pval_to_stars(p):
        if isinstance(p, str):
            p_str = p.strip()
            if p_str in {'ns', '*', '**', '***', '****'}:
                return p_str
            try:
                p = float(p_str)
            except ValueError:
                return p_str

        if pd.isna(p):
            return None

        p = float(p)

        if p <= 1e-4:
            return '****'
        elif p <= 1e-3:
            return '***'
        elif p <= 1e-2:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return 'ns'

    hline_col_set = set(reference_hlines.keys())

    hline_entries = []
    for col_name, style in reference_hlines.items():
        if mode == 'box':
            if col_name in df.columns:
                agg = style.get('agg', 'mean')
                if agg == 'median':
                    y_val = df[col_name].dropna().median()
                else:
                    y_val = df[col_name].dropna().mean()
            else:
                continue
        else:
            row = df[df[model_col] == col_name]
            if len(row) > 0:
                y_val = row[score_col].values[0]
            else:
                continue
        hline_entries.append((y_val, style, col_name))

    filtered_groups = {}
    for label, model_list in model_groups.items():
        filtered = []
        for col in model_list:
            if col in hline_col_set:
                filtered.append(None)
            else:
                filtered.append(col)
        filtered_groups[label] = filtered

    n_orig = len(x_ticks_working)
    keep_ticks = []
    for tick_idx in range(n_orig):
        has_any = False
        for label, flist in filtered_groups.items():
            if tick_idx < len(flist) and flist[tick_idx] is not None:
                has_any = True
                break
        if has_any:
            keep_ticks.append(tick_idx)

    x_ticks_final = [x_ticks_working[i] for i in keep_ticks]
    final_groups = {}
    for label, flist in filtered_groups.items():
        final_groups[label] = [flist[i] for i in keep_ticks if i < len(flist)]

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(model_groups.keys())
        }

    created_own_ax = ax is None
    if created_own_ax:
        fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])
    else:
        fig = ax.figure

    n_groups = len(x_ticks_final)

    bars_per_position = [
        sum(
            1 for label, model_list in final_groups.items()
            if tick_idx < len(model_list) and model_list[tick_idx] is not None
        )
        for tick_idx in range(n_groups)
    ]

    cluster_widths = [
        max(n, 1) * bar_width + max(n - 1, 0) * bar_spacing
        for n in bars_per_position
    ]

    group_positions = np.zeros(n_groups)
    cursor = 0.0
    for i, cw in enumerate(cluster_widths):
        group_positions[i] = cursor + cw / 2.0
        cursor += cw + group_spacing

    edge_margin = x_margin if x_margin is not None else group_spacing / 2.0
    ax.set_xlim(
        group_positions[0] - cluster_widths[0] / 2.0 - edge_margin,
        group_positions[-1] + cluster_widths[-1] / 2.0 + edge_margin,
    )

    plotted_positions = {}
    plotted_upper_whiskers = {}

    def _get_interval(col_name, center):
        if interval_df is None or col_name not in interval_df.columns:
            return None

        low = pd.to_numeric(pd.Series([interval_df.at[interval_low_row, col_name]]), errors='coerce').iloc[0]
        high = pd.to_numeric(pd.Series([interval_df.at[interval_high_row, col_name]]), errors='coerce').iloc[0]
        if pd.isna(low) or pd.isna(high):
            return None

        low, high, center = float(low), float(high), float(center)
        if low > high:
            raise ValueError(
                f"interval_df has low > high for {col_name!r}: {low} > {high}."
            )
        if center < low or center > high:
            raise ValueError(
                f"Bar value for {col_name!r} ({center}) is outside its interval [{low}, {high}]."
            )
        return low, high

    def _draw_interval(x_pos, center, col_name):
        interval = _get_interval(col_name, center)
        if interval is None:
            return None

        low, high = interval
        yerr = np.array([[center - low], [high - center]], dtype=float)
        ax.errorbar(
            x_pos, center, yerr=yerr, fmt='none',
            ecolor=interval_color, elinewidth=interval_linewidth,
            capsize=interval_capsize, capthick=interval_capthick,
            alpha=interval_alpha, zorder=interval_zorder,
        )
        return low, high

    def _upper_boxplot_whisker(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        valid = vals[vals <= upper_limit]

        if len(valid) == 0:
            return np.max(vals)

        return np.max(valid)

    for group_idx, (label, model_list) in enumerate(final_groups.items()):
        color = colors[label]

        if mode == 'bar':
            scores = []
            for col_name in model_list:
                if col_name is None:
                    scores.append(None)
                    continue

                score = df[df[model_col] == col_name][score_col].values
                scores.append(score[0] if len(score) > 0 else 0)

            first_plotted = True
            for tick_idx in range(len(scores)):
                if tick_idx >= n_groups or scores[tick_idx] is None:
                    continue

                col_name = model_list[tick_idx]
                if col_name is None:
                    continue

                cw = cluster_widths[tick_idx]
                left_edge = group_positions[tick_idx] - cw / 2.0
                x_pos = left_edge + group_idx * (bar_width + bar_spacing) + bar_width / 2.0

                plotted_positions[(tick_idx, col_name)] = x_pos

                ax.bar(
                    x_pos,
                    scores[tick_idx],
                    width=bar_width,
                    color=color,
                    edgecolor=bar_outline_color,
                    linewidth=bar_outline_width,
                    label=label if first_plotted else "",
                    zorder=2,
                )

                interval = _draw_interval(x_pos, scores[tick_idx], col_name)
                plotted_upper_whiskers[(tick_idx, col_name)] = (
                    interval[1] if interval is not None else scores[tick_idx]
                )
                first_plotted = False

        elif mode == 'box':
            first_plotted = True
            for tick_idx, col_name in enumerate(model_list):
                if tick_idx >= n_groups or col_name is None:
                    continue

                if col_name not in df.columns:
                    continue

                vals = df[col_name].dropna().values.tolist()
                if not vals:
                    continue

                cw = cluster_widths[tick_idx]
                left_edge = group_positions[tick_idx] - cw / 2.0
                x_pos = left_edge + group_idx * (bar_width + bar_spacing) + bar_width / 2.0

                plotted_positions[(tick_idx, col_name)] = x_pos
                plotted_upper_whiskers[(tick_idx, col_name)] = _upper_boxplot_whisker(vals)

                flier_c = box_flier_color if box_flier_color is not None else color

                ax.boxplot(
                    [vals],
                    positions=[x_pos],
                    widths=bar_width * 0.9,
                    patch_artist=True,
                    notch=box_notch,
                    manage_ticks=False,
                    zorder=2,
                    flierprops=dict(
                        marker=box_flier_marker,
                        markersize=box_flier_size,
                        markerfacecolor=flier_c,
                        markeredgecolor=flier_c,
                        alpha=box_flier_alpha,
                    ),
                    medianprops=dict(
                        color=box_median_color,
                        linewidth=box_median_width,
                    ),
                    whiskerprops=dict(
                        color=bar_outline_color,
                        linewidth=box_whisker_width,
                        linestyle=box_whisker_style,
                    ),
                    capprops=dict(
                        color='none',
                        linewidth=0,
                    ),
                    boxprops=dict(
                        facecolor=color,
                        edgecolor=bar_outline_color,
                        linewidth=bar_outline_width,
                        alpha=box_patch_alpha,
                    ),
                )
                first_plotted = False

    if pval_df is not None and mode in ['box', 'bar']:
        for tick_idx in range(n_groups):
            cols_at_tick = []
            for label, model_list in final_groups.items():
                if tick_idx < len(model_list):
                    col_name = model_list[tick_idx]
                    if col_name is not None and (tick_idx, col_name) in plotted_positions:
                        cols_at_tick.append(col_name)

            if len(cols_at_tick) < 2:
                continue

            comparison_source = pval_pairs if pval_pairs is not None else pval_comparisons

            if comparison_source is None:
                if pval_base_only:
                    if pval_base_col is None:
                        raise ValueError(
                            "pval_base_col must be provided when pval_base_only=True."
                        )
                    if pval_base_col not in cols_at_tick:
                        comparisons = []
                    else:
                        comparisons = [
                            (pval_base_col, col)
                            for col in cols_at_tick
                            if col != pval_base_col
                        ]
                else:
                    comparisons = [
                        (cols_at_tick[i], cols_at_tick[j])
                        for i in range(len(cols_at_tick) - 1)
                        for j in range(i + 1, len(cols_at_tick))
                    ]
            else:
                comparisons = comparison_source

            upper_vals = [
                plotted_upper_whiskers[(tick_idx, col)]
                for col in cols_at_tick
                if (tick_idx, col) in plotted_upper_whiskers
            ]

            if len(upper_vals) == 0:
                continue

            if pval_outside_axes:
                base_y = 1.0 + pval_bar_offset
                pval_transform = ax.get_xaxis_transform()
            else:
                base_y = np.nanmax(upper_vals) + pval_bar_offset
                pval_transform = None

            pval_levels = []
            hierarchical_counter = 0

            def _intervals_overlap(interval_a, interval_b, eps=1e-12):
                a1, a2 = interval_a
                b1, b2 = interval_b
                return max(a1, b1) < min(a2, b2) - eps

            for col_a, col_b in comparisons:
                if (tick_idx, col_a) not in plotted_positions:
                    continue
                if (tick_idx, col_b) not in plotted_positions:
                    continue

                x_a = plotted_positions[(tick_idx, col_a)]
                x_b = plotted_positions[(tick_idx, col_b)]

                if x_a <= x_b:
                    x1, x2 = x_a, x_b
                else:
                    x1, x2 = x_b, x_a

                pval_value = None

                if col_a in pval_df.index and col_b in pval_df.columns:
                    pval_value = pval_df.loc[col_a, col_b]
                elif col_b in pval_df.index and col_a in pval_df.columns:
                    pval_value = pval_df.loc[col_b, col_a]

                if pval_value is None:
                    continue
                if pd.isna(pval_value):
                    continue

                if pval_format == 'stars':
                    pval_text = _pval_to_stars(pval_value)
                else:
                    pval_text = str(pval_value)

                if pval_text is None:
                    continue
                if str(pval_text) == '':
                    continue

                current_interval = (x1, x2)

                if pval_merge_nonoverlapping:
                    assigned_level = None

                    for level_idx, occupied_intervals in enumerate(pval_levels):
                        has_overlap = any(
                            _intervals_overlap(current_interval, existing_interval)
                            for existing_interval in occupied_intervals
                        )

                        if not has_overlap:
                            assigned_level = level_idx
                            break

                    if assigned_level is None:
                        assigned_level = len(pval_levels)
                        pval_levels.append([])

                    pval_levels[assigned_level].append(current_interval)

                else:
                    assigned_level = hierarchical_counter
                    hierarchical_counter += 1

                y = base_y + assigned_level * pval_bar_spacing

                if pval_outside_axes:
                    ax.plot(
                        [x1, x2],
                        [y, y],
                        transform=pval_transform,
                        color=pval_color,
                        linewidth=pval_linewidth,
                        solid_capstyle='butt',
                        clip_on=False,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        [x1, x2],
                        [y, y],
                        color=pval_color,
                        linewidth=pval_linewidth,
                        solid_capstyle='butt',
                        zorder=3,
                    )

                if str(pval_text).lower() == 'ns':
                    text_offset = pval_ns_text_offset
                elif set(str(pval_text)) <= {'*'}:
                    text_offset = pval_star_text_offset
                else:
                    text_offset = pval_text_offset

                if pval_outside_axes:
                    ax.text(
                        (x1 + x2) / 2.0,
                        y + text_offset,
                        str(pval_text),
                        transform=pval_transform,
                        ha='center',
                        va='bottom',
                        fontsize=pval_fontsize,
                        fontfamily=default_font,
                        color=pval_color,
                        clip_on=False,
                        zorder=4,
                    )
                else:
                    ax.text(
                        (x1 + x2) / 2.0,
                        y + text_offset,
                        str(pval_text),
                        ha='center',
                        va='bottom',
                        fontsize=pval_fontsize,
                        fontfamily=default_font,
                        color=pval_color,
                        zorder=4,
                    )

    for y_val, style, col_name in hline_entries:
        ax.axhline(
            y=y_val,
            color=style.get('color', 'gray'),
            linestyle=style.get('linestyle', '--'),
            linewidth=style.get('linewidth', 1.5),
            alpha=style.get('alpha', 1.0),
            label=style.get('label', col_name),
            zorder=1,
        )

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None

        ax.axhline(
            y=hline_y,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            label=line_label,
            zorder=1,
        )

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            x_range = group_positions[-1] - group_positions[0] if len(group_positions) > 1 else 1
            text_x = group_positions[0] + (x_range * hline_text_x_offset)

            if hline_text_position == 'above':
                text_y = hline_y + hline_text_offset
                va = 'bottom'
            else:
                text_y = hline_y - hline_text_offset
                va = 'top'

            ax.text(
                text_x,
                text_y,
                hline_text,
                color=text_color,
                fontsize=hline_text_fontsize,
                fontfamily=default_font,
                va=va,
                ha='left',
                zorder=3,
            )

    if hide_x_ticks:
        ax.set_xticks([])
    else:
        ax.set_xticks(group_positions)

    ax.set_xlabel(
        params['xlabel'],
        fontsize=params['xlabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['xlabel_pad'],
    )
    ax.set_ylabel(
        params['ylabel'],
        fontsize=params['ylabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['ylabel_pad'],
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(
        axis='both',
        which='major',
        length=params['tick_length'],
        width=params['axis_linewidth'],
        direction='out',
        pad=params['tick_length'] + 2,
    )
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if not hide_x_ticks:
        ax.set_xticklabels(x_ticks_final, rotation=xtick_rotation, ha='center')

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis='y',
            zorder=0,
        )

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []

    for lbl_key in model_groups.keys():
        c = colors[lbl_key]
        handles.append(
            mpatches.Patch(
                facecolor=c,
                edgecolor=bar_outline_color,
                linewidth=bar_outline_width,
            )
        )
        labels_list.append(lbl_key)

    for y_val, style, col_name in hline_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get('color', 'gray'),
                linestyle=style.get('linestyle', '--'),
                linewidth=style.get('linewidth', 1.5) * params['legend_handle_size'],
                alpha=style.get('alpha', 1.0),
            )
        )
        labels_list.append(style.get('label', col_name))

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(
            Line2D(
                [0],
                [0],
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width * params['legend_handle_size'],
            )
        )
        labels_list.append(hline_label)

    if legend_mode == 'integrated':
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        legend = fig.legend(
            handles,
            labels_list,
            loc=legend_loc,
            bbox_to_anchor=bbox,
            ncol=legend_ncol,
            frameon=False,
            fontsize=params['legend_fontsize'],
        )
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

    if created_own_ax:
        plt.tight_layout()

    if params.get('title', '') not in ['', None]:
        title_weight = 'bold' if params['title_bold'] else 'normal'

        ax.text(
            0.5,
            params.get('title_y', 1.04),
            params['title'],
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=params['title_fontsize'],
            fontfamily=default_font,
            fontweight=title_weight,
            color=params.get('title_color', 'black'),
            bbox=params.get('title_bbox', None),
            clip_on=False,
            zorder=100,
        )

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''

        fig.savefig(
            f"{save_path}{suffix}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )
        fig.savefig(
            f"{save_path}{suffix}.png",
            format='png',
            dpi=300,
            bbox_inches='tight',
        )

    if show:
        plt.show()

    if legend_mode == 'separate' and created_own_ax:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(
                params['legend_handle_size'] * 2,
                n_items * 0.4 * params['legend_handle_size'],
            ),
            dpi=params['dpi'],
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')

        legend = ax_legend.legend(
            handles,
            labels_list,
            loc='center',
            frameon=False,
            fontsize=params['legend_fontsize'],
            ncol=legend_ncol,
        )
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(
                f"{save_path}_legend.pdf",
                format='pdf',
                bbox_inches='tight',
                dpi=300,
            )
            fig_legend.savefig(
                f"{save_path}_legend.png",
                format='png',
                dpi=300,
            )

        if show:
            plt.show()

    return fig, ax

def plot_model_performance_box_grid_v3(
    plot_dict,
    save_path=None,
    panel_gap=0.0,
    grid_dpi=300,
    **base_plotting_parameters,
):
    """
    Plot a horizontal grid of bar/box panels using
    plot_model_performance_bars_or_box_v3().

    Every base plotting argument can be overridden independently for each
    panel by placing it directly inside that panel's plot_dict entry.

    Examples of per-panel overrides include:
        mode, model_col, score_col, model_groups, x_ticks, colors, ylim,
        grid_alpha, bar/box styling, interval_df and interval styling,
        p-value settings, reference lines, legend settings, etc.

    `plot_params` is merged separately so individual panels can override
    ylabel, xlabel, font sizes, title styling, tick styling, etc.

    Reserved panel keys:
        df          : dataframe plotted in that panel
        title       : convenience title override
        plot_params : dictionary merged over the global plot_params

    The grid itself controls ax/show/save_path, so those are not forwarded
    from individual panel dictionaries.
    """
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if not isinstance(plot_dict, dict) or len(plot_dict) == 0:
        raise ValueError("plot_dict must be a non-empty dictionary.")

    base_plot_params = base_plotting_parameters.get('plot_params', {}).copy()
    single_figsize = base_plot_params.get('figsize', (1.20, 1.65))
    single_w, single_h = single_figsize

    n_panels = len(plot_dict)
    fig_w = n_panels * single_w + max(n_panels - 1, 0) * panel_gap
    fig_h = single_h

    # Probe one panel to preserve the same physical axes geometry used by the
    # original grid function. Individual panel labels/styles may differ, but
    # all panels retain the same slot size for consistent figure assembly.
    probe_fig, probe_ax = plt.subplots(figsize=single_figsize, dpi=grid_dpi)
    probe_ax.set_xlabel(
        base_plot_params.get('xlabel', ''),
        fontsize=base_plot_params.get('xlabel_fontsize', 12),
        fontfamily=base_plotting_parameters.get('default_font', 'Nimbus Sans'),
        labelpad=base_plot_params.get('xlabel_pad', 4.0),
    )
    probe_ax.set_ylabel(
        base_plot_params.get('ylabel', ''),
        fontsize=base_plot_params.get('ylabel_fontsize', 12),
        fontfamily=base_plotting_parameters.get('default_font', 'Nimbus Sans'),
        labelpad=base_plot_params.get('ylabel_pad', 4.0),
    )
    probe_ylim = base_plotting_parameters.get('ylim', (0.0, 1.0))
    if probe_ylim is not None:
        probe_ax.set_ylim(probe_ylim)
    probe_ax.set_xticks([])
    probe_ax.tick_params(
        axis='both', which='major',
        length=base_plot_params.get('tick_length', 4.0),
        width=base_plot_params.get('axis_linewidth', 0.8),
        direction='out',
        pad=base_plot_params.get('tick_length', 4.0) + 2,
    )
    probe_ax.tick_params(axis='x', labelsize=base_plot_params.get('xtick_labelsize', 10))
    probe_ax.tick_params(axis='y', labelsize=base_plot_params.get('ytick_labelsize', 10))

    plt.figure(probe_fig.number)
    plt.tight_layout()
    probe_pos = probe_ax.get_position()
    plt.close(probe_fig)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=grid_dpi)
    axes = []

    reserved_keys = {'df', 'title', 'plot_params', 'ax', 'show', 'save_path'}

    for i, (panel_key, panel_info) in enumerate(plot_dict.items()):
        if not isinstance(panel_info, dict):
            raise TypeError(f"plot_dict[{panel_key!r}] must be a dictionary.")
        if 'df' not in panel_info:
            raise ValueError(f"plot_dict[{panel_key!r}] is missing required key 'df'.")

        slot_left = i * (single_w + panel_gap)
        ax_left = (slot_left + probe_pos.x0 * single_w) / fig_w
        ax_bottom = (probe_pos.y0 * single_h) / fig_h
        ax_width = (probe_pos.width * single_w) / fig_w
        ax_height = (probe_pos.height * single_h) / fig_h

        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
        axes.append(ax)

        # Start from global defaults, then let this panel override ANY argument
        # accepted by plot_model_performance_bars_or_box_v3(). This includes
        # model_col/score_col, mode, model_groups, x_ticks, colors, ylim,
        # interval settings, p-value settings, bar/box styles, etc.
        panel_params = base_plotting_parameters.copy()
        for key, value in panel_info.items():
            if key not in reserved_keys:
                panel_params[key] = value

        # plot_params is nested, so merge rather than replace it.
        panel_plot_params = base_plot_params.copy()
        panel_plot_params['title'] = panel_info.get('title', panel_key)
        if panel_info.get('plot_params') is not None:
            panel_plot_params.update(panel_info['plot_params'])
        panel_params['plot_params'] = panel_plot_params

        # Grid owns these.
        panel_params.pop('ax', None)
        panel_params.pop('show', None)
        panel_params.pop('save_path', None)

        plot_model_performance_bars_or_box_v3(
            df=panel_info['df'],
            ax=ax,
            show=False,
            save_path=None,
            **panel_params,
        )

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=grid_dpi)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=grid_dpi)

    plt.show()
    return fig, axes

def plot_model_performance_bars_or_box_v2(
    df,
    model_col,
    score_col,
    model_groups,
    x_ticks,
    colors=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    bar_width=0.8,
    group_spacing=0.3,
    bar_spacing=0.0,
    bar_outline_color='black',
    bar_outline_width=1.0,
    mode='bar',
    box_flier_marker='o',
    box_flier_size=3,
    box_flier_color=None,
    box_flier_alpha=0.6,
    box_median_color='black',
    box_median_width=1.5,
    box_whisker_style='-',
    box_whisker_width=1.0,
    box_notch=False,
    box_patch_alpha=0.9,
    pval_df=None,
    pval_comparisons=None,
    pval_pairs=None,
    pval_base_only=False,
    pval_base_col=None,
    pval_merge_nonoverlapping=True,
    pval_format='stars',
    pval_fontsize=5,
    pval_linewidth=0.35,
    pval_color='black',
    pval_bar_offset=0.035,
    pval_bar_spacing=0.055,
    pval_text_offset=0.004,
    pval_star_text_offset=0.001,
    pval_ns_text_offset=0.004,
    reference_hlines=None,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    x_margin=None,
    xtick_rotation=0,
    edge_margin=None,
    legend_mode='separate',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    ax=None,
    show=True,
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'title_y': 1.04,
        'title_color': 'black',
        'title_bbox': None,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (10, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    if reference_hlines is None:
        reference_hlines = {}

    if x_ticks is None or len(x_ticks) == 0:
        hide_x_ticks = True
        n_orig = max(len(model_list) for model_list in model_groups.values())
        x_ticks_working = [''] * n_orig
    else:
        hide_x_ticks = all(str(tick) == '' for tick in x_ticks)
        x_ticks_working = x_ticks

    def _pval_to_stars(p):
        if isinstance(p, str):
            p_str = p.strip()
            if p_str in {'ns', '*', '**', '***', '****'}:
                return p_str
            try:
                p = float(p_str)
            except ValueError:
                return p_str

        if pd.isna(p):
            return None

        p = float(p)

        if p <= 1e-4:
            return '****'
        elif p <= 1e-3:
            return '***'
        elif p <= 1e-2:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return 'ns'

    hline_col_set = set(reference_hlines.keys())

    hline_entries = []
    for col_name, style in reference_hlines.items():
        if mode == 'box':
            if col_name in df.columns:
                agg = style.get('agg', 'mean')
                if agg == 'median':
                    y_val = df[col_name].dropna().median()
                else:
                    y_val = df[col_name].dropna().mean()
            else:
                continue
        else:
            row = df[df[model_col] == col_name]
            if len(row) > 0:
                y_val = row[score_col].values[0]
            else:
                continue
        hline_entries.append((y_val, style, col_name))

    filtered_groups = {}
    for label, model_list in model_groups.items():
        filtered = []
        for col in model_list:
            if col in hline_col_set:
                filtered.append(None)
            else:
                filtered.append(col)
        filtered_groups[label] = filtered

    n_orig = len(x_ticks_working)
    keep_ticks = []
    for tick_idx in range(n_orig):
        has_any = False
        for label, flist in filtered_groups.items():
            if tick_idx < len(flist) and flist[tick_idx] is not None:
                has_any = True
                break
        if has_any:
            keep_ticks.append(tick_idx)

    x_ticks_final = [x_ticks_working[i] for i in keep_ticks]
    final_groups = {}
    for label, flist in filtered_groups.items():
        final_groups[label] = [flist[i] for i in keep_ticks if i < len(flist)]

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(model_groups.keys())
        }

    created_own_ax = ax is None
    if created_own_ax:
        fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])
    else:
        fig = ax.figure

    n_groups = len(x_ticks_final)

    bars_per_position = [
        sum(
            1 for label, model_list in final_groups.items()
            if tick_idx < len(model_list) and model_list[tick_idx] is not None
        )
        for tick_idx in range(n_groups)
    ]

    cluster_widths = [
        max(n, 1) * bar_width + max(n - 1, 0) * bar_spacing
        for n in bars_per_position
    ]

    group_positions = np.zeros(n_groups)
    cursor = 0.0
    for i, cw in enumerate(cluster_widths):
        group_positions[i] = cursor + cw / 2.0
        cursor += cw + group_spacing

    edge_margin = x_margin if x_margin is not None else group_spacing / 2.0
    ax.set_xlim(
        group_positions[0] - cluster_widths[0] / 2.0 - edge_margin,
        group_positions[-1] + cluster_widths[-1] / 2.0 + edge_margin,
    )

    plotted_positions = {}
    plotted_upper_whiskers = {}

    def _upper_boxplot_whisker(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        valid = vals[vals <= upper_limit]

        if len(valid) == 0:
            return np.max(vals)

        return np.max(valid)

    for group_idx, (label, model_list) in enumerate(final_groups.items()):
        color = colors[label]

        if mode == 'bar':
            scores = []
            for col_name in model_list:
                if col_name is None:
                    scores.append(None)
                    continue

                score = df[df[model_col] == col_name][score_col].values
                scores.append(score[0] if len(score) > 0 else 0)

            first_plotted = True
            for tick_idx in range(len(scores)):
                if tick_idx >= n_groups or scores[tick_idx] is None:
                    continue

                col_name = model_list[tick_idx]
                if col_name is None:
                    continue

                cw = cluster_widths[tick_idx]
                left_edge = group_positions[tick_idx] - cw / 2.0
                x_pos = left_edge + group_idx * (bar_width + bar_spacing) + bar_width / 2.0

                plotted_positions[(tick_idx, col_name)] = x_pos
                plotted_upper_whiskers[(tick_idx, col_name)] = scores[tick_idx]

                ax.bar(
                    x_pos,
                    scores[tick_idx],
                    width=bar_width,
                    color=color,
                    edgecolor=bar_outline_color,
                    linewidth=bar_outline_width,
                    label=label if first_plotted else "",
                    zorder=2,
                )
                first_plotted = False

        elif mode == 'box':
            first_plotted = True
            for tick_idx, col_name in enumerate(model_list):
                if tick_idx >= n_groups or col_name is None:
                    continue

                if col_name not in df.columns:
                    continue

                vals = df[col_name].dropna().values.tolist()
                if not vals:
                    continue

                cw = cluster_widths[tick_idx]
                left_edge = group_positions[tick_idx] - cw / 2.0
                x_pos = left_edge + group_idx * (bar_width + bar_spacing) + bar_width / 2.0

                plotted_positions[(tick_idx, col_name)] = x_pos
                plotted_upper_whiskers[(tick_idx, col_name)] = _upper_boxplot_whisker(vals)

                flier_c = box_flier_color if box_flier_color is not None else color

                ax.boxplot(
                    [vals],
                    positions=[x_pos],
                    widths=bar_width * 0.9,
                    patch_artist=True,
                    notch=box_notch,
                    manage_ticks=False,
                    zorder=2,
                    flierprops=dict(
                        marker=box_flier_marker,
                        markersize=box_flier_size,
                        markerfacecolor=flier_c,
                        markeredgecolor=flier_c,
                        alpha=box_flier_alpha,
                    ),
                    medianprops=dict(
                        color=box_median_color,
                        linewidth=box_median_width,
                    ),
                    whiskerprops=dict(
                        color=bar_outline_color,
                        linewidth=box_whisker_width,
                        linestyle=box_whisker_style,
                    ),
                    capprops=dict(
                        color='none',
                        linewidth=0,
                    ),
                    boxprops=dict(
                        facecolor=color,
                        edgecolor=bar_outline_color,
                        linewidth=bar_outline_width,
                        alpha=box_patch_alpha,
                    ),
                )
                first_plotted = False

    if pval_df is not None and mode in ['box', 'bar']:
        for tick_idx in range(n_groups):
            cols_at_tick = []
            for label, model_list in final_groups.items():
                if tick_idx < len(model_list):
                    col_name = model_list[tick_idx]
                    if col_name is not None and (tick_idx, col_name) in plotted_positions:
                        cols_at_tick.append(col_name)

            if len(cols_at_tick) < 2:
                continue

            comparison_source = pval_pairs if pval_pairs is not None else pval_comparisons

            if comparison_source is None:
                if pval_base_only:
                    if pval_base_col is None:
                        raise ValueError(
                            "pval_base_col must be provided when pval_base_only=True."
                        )
                    if pval_base_col not in cols_at_tick:
                        comparisons = []
                    else:
                        comparisons = [
                            (pval_base_col, col)
                            for col in cols_at_tick
                            if col != pval_base_col
                        ]
                else:
                    comparisons = [
                        (cols_at_tick[i], cols_at_tick[j])
                        for i in range(len(cols_at_tick) - 1)
                        for j in range(i + 1, len(cols_at_tick))
                    ]
            else:
                comparisons = comparison_source

            upper_vals = [
                plotted_upper_whiskers[(tick_idx, col)]
                for col in cols_at_tick
                if (tick_idx, col) in plotted_upper_whiskers
            ]

            if len(upper_vals) == 0:
                continue

            base_y = np.nanmax(upper_vals) + pval_bar_offset

            pval_levels = []
            hierarchical_counter = 0

            def _intervals_overlap(interval_a, interval_b, eps=1e-12):
                a1, a2 = interval_a
                b1, b2 = interval_b
                return max(a1, b1) < min(a2, b2) - eps

            for col_a, col_b in comparisons:
                if (tick_idx, col_a) not in plotted_positions:
                    continue
                if (tick_idx, col_b) not in plotted_positions:
                    continue

                x_a = plotted_positions[(tick_idx, col_a)]
                x_b = plotted_positions[(tick_idx, col_b)]

                if x_a <= x_b:
                    x1, x2 = x_a, x_b
                else:
                    x1, x2 = x_b, x_a

                pval_value = None

                if col_a in pval_df.index and col_b in pval_df.columns:
                    pval_value = pval_df.loc[col_a, col_b]
                elif col_b in pval_df.index and col_a in pval_df.columns:
                    pval_value = pval_df.loc[col_b, col_a]

                if pval_value is None:
                    continue
                if pd.isna(pval_value):
                    continue

                if pval_format == 'stars':
                    pval_text = _pval_to_stars(pval_value)
                else:
                    pval_text = str(pval_value)

                if pval_text is None:
                    continue
                if str(pval_text) == '':
                    continue

                current_interval = (x1, x2)

                if pval_merge_nonoverlapping:
                    assigned_level = None

                    for level_idx, occupied_intervals in enumerate(pval_levels):
                        has_overlap = any(
                            _intervals_overlap(current_interval, existing_interval)
                            for existing_interval in occupied_intervals
                        )

                        if not has_overlap:
                            assigned_level = level_idx
                            break

                    if assigned_level is None:
                        assigned_level = len(pval_levels)
                        pval_levels.append([])

                    pval_levels[assigned_level].append(current_interval)

                else:
                    assigned_level = hierarchical_counter
                    hierarchical_counter += 1

                y = base_y + assigned_level * pval_bar_spacing

                ax.plot(
                    [x1, x2],
                    [y, y],
                    color=pval_color,
                    linewidth=pval_linewidth,
                    solid_capstyle='butt',
                    zorder=3,
                )

                if str(pval_text).lower() == 'ns':
                    text_offset = pval_ns_text_offset
                elif set(str(pval_text)) <= {'*'}:
                    text_offset = pval_star_text_offset
                else:
                    text_offset = pval_text_offset

                ax.text(
                    (x1 + x2) / 2.0,
                    y + text_offset,
                    str(pval_text),
                    ha='center',
                    va='bottom',
                    fontsize=pval_fontsize,
                    fontfamily=default_font,
                    color=pval_color,
                    zorder=4,
                )

    for y_val, style, col_name in hline_entries:
        ax.axhline(
            y=y_val,
            color=style.get('color', 'gray'),
            linestyle=style.get('linestyle', '--'),
            linewidth=style.get('linewidth', 1.5),
            alpha=style.get('alpha', 1.0),
            label=style.get('label', col_name),
            zorder=1,
        )

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None

        ax.axhline(
            y=hline_y,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            label=line_label,
            zorder=1,
        )

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            x_range = group_positions[-1] - group_positions[0] if len(group_positions) > 1 else 1
            text_x = group_positions[0] + (x_range * hline_text_x_offset)

            if hline_text_position == 'above':
                text_y = hline_y + hline_text_offset
                va = 'bottom'
            else:
                text_y = hline_y - hline_text_offset
                va = 'top'

            ax.text(
                text_x,
                text_y,
                hline_text,
                color=text_color,
                fontsize=hline_text_fontsize,
                fontfamily=default_font,
                va=va,
                ha='left',
                zorder=3,
            )

    if hide_x_ticks:
        ax.set_xticks([])
    else:
        ax.set_xticks(group_positions)

    ax.set_xlabel(
        params['xlabel'],
        fontsize=params['xlabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['xlabel_pad'],
    )
    ax.set_ylabel(
        params['ylabel'],
        fontsize=params['ylabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['ylabel_pad'],
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(
        axis='both',
        which='major',
        length=params['tick_length'],
        width=params['axis_linewidth'],
        direction='out',
        pad=params['tick_length'] + 2,
    )
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if not hide_x_ticks:
        ax.set_xticklabels(x_ticks_final, rotation=xtick_rotation, ha='center')

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis='y',
            zorder=0,
        )

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []

    for lbl_key in model_groups.keys():
        c = colors[lbl_key]
        handles.append(
            mpatches.Patch(
                facecolor=c,
                edgecolor=bar_outline_color,
                linewidth=bar_outline_width,
            )
        )
        labels_list.append(lbl_key)

    for y_val, style, col_name in hline_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get('color', 'gray'),
                linestyle=style.get('linestyle', '--'),
                linewidth=style.get('linewidth', 1.5) * params['legend_handle_size'],
                alpha=style.get('alpha', 1.0),
            )
        )
        labels_list.append(style.get('label', col_name))

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(
            Line2D(
                [0],
                [0],
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width * params['legend_handle_size'],
            )
        )
        labels_list.append(hline_label)

    if legend_mode == 'integrated':
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        legend = fig.legend(
            handles,
            labels_list,
            loc=legend_loc,
            bbox_to_anchor=bbox,
            ncol=legend_ncol,
            frameon=False,
            fontsize=params['legend_fontsize'],
        )
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

    if created_own_ax:
        plt.tight_layout()

    if params.get('title', '') not in ['', None]:
        title_weight = 'bold' if params['title_bold'] else 'normal'

        ax.text(
            0.5,
            params.get('title_y', 1.04),
            params['title'],
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=params['title_fontsize'],
            fontfamily=default_font,
            fontweight=title_weight,
            color=params.get('title_color', 'black'),
            bbox=params.get('title_bbox', None),
            clip_on=False,
            zorder=100,
        )

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''

        fig.savefig(
            f"{save_path}{suffix}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )
        fig.savefig(
            f"{save_path}{suffix}.png",
            format='png',
            dpi=300,
            bbox_inches='tight',
        )

    if show:
        plt.show()

    if legend_mode == 'separate' and created_own_ax:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(
                params['legend_handle_size'] * 2,
                n_items * 0.4 * params['legend_handle_size'],
            ),
            dpi=params['dpi'],
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')

        legend = ax_legend.legend(
            handles,
            labels_list,
            loc='center',
            frameon=False,
            fontsize=params['legend_fontsize'],
            ncol=legend_ncol,
        )
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(
                f"{save_path}_legend.pdf",
                format='pdf',
                bbox_inches='tight',
                dpi=300,
            )
            fig_legend.savefig(
                f"{save_path}_legend.png",
                format='png',
                dpi=300,
            )

        if show:
            plt.show()

    return fig, ax

def plot_model_performance_box_grid_v2(
    plot_dict,
    save_path=None,
    panel_gap=0.0,
    grid_dpi=300,
    **base_plotting_parameters,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    base_plot_params = base_plotting_parameters.get('plot_params', {}).copy()
    single_figsize = base_plot_params.get('figsize', (1.20, 1.65))
    single_w, single_h = single_figsize

    n_panels = len(plot_dict)
    fig_w = n_panels * single_w + max(n_panels - 1, 0) * panel_gap
    fig_h = single_h

    # Probe the single-panel axes location after tight_layout, without title.
    probe_fig, probe_ax = plt.subplots(figsize=single_figsize, dpi=grid_dpi)

    probe_ax.set_xlabel(
        base_plot_params.get('xlabel', ''),
        fontsize=base_plot_params.get('xlabel_fontsize', 12),
        fontfamily=base_plotting_parameters.get('default_font', 'Nimbus Sans'),
        labelpad=base_plot_params.get('xlabel_pad', 4.0),
    )
    probe_ax.set_ylabel(
        base_plot_params.get('ylabel', ''),
        fontsize=base_plot_params.get('ylabel_fontsize', 12),
        fontfamily=base_plotting_parameters.get('default_font', 'Nimbus Sans'),
        labelpad=base_plot_params.get('ylabel_pad', 4.0),
    )

    probe_ax.set_ylim(base_plotting_parameters.get('ylim', (0.0, 1.0)))
    probe_ax.set_xticks([])

    probe_ax.tick_params(
        axis='both',
        which='major',
        length=base_plot_params.get('tick_length', 4.0),
        width=base_plot_params.get('axis_linewidth', 0.8),
        direction='out',
        pad=base_plot_params.get('tick_length', 4.0) + 2,
    )
    probe_ax.tick_params(axis='x', labelsize=base_plot_params.get('xtick_labelsize', 10))
    probe_ax.tick_params(axis='y', labelsize=base_plot_params.get('ytick_labelsize', 10))

    plt.figure(probe_fig.number)
    plt.tight_layout()
    probe_pos = probe_ax.get_position()
    plt.close(probe_fig)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=grid_dpi)
    axes = []

    for i, (panel_key, panel_info) in enumerate(plot_dict.items()):
        slot_left = i * (single_w + panel_gap)

        ax_left = (slot_left + probe_pos.x0 * single_w) / fig_w
        ax_bottom = (probe_pos.y0 * single_h) / fig_h
        ax_width = (probe_pos.width * single_w) / fig_w
        ax_height = (probe_pos.height * single_h) / fig_h

        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
        axes.append(ax)

        panel_params = base_plotting_parameters.copy()

        panel_plot_params = base_plot_params.copy()
        panel_plot_params['title'] = panel_info.get('title', panel_key)

        if 'plot_params' in panel_info:
            panel_plot_params.update(panel_info['plot_params'])

        panel_params['plot_params'] = panel_plot_params
        panel_params['pval_df'] = panel_info.get('pval_df', None)

        if 'pval_comparisons' in panel_info:
            panel_params['pval_comparisons'] = panel_info['pval_comparisons']

        plot_model_performance_bars_or_box_v2(
            df=panel_info['df'],
            ax=ax,
            show=False,
            save_path=None,
            **panel_params,
        )

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=grid_dpi)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=grid_dpi)

    plt.show()

    return fig, axes




def plot_single_metric_comparison_precomputed_y(
    df1,
    df2,
    score_cols,
    label_col1=None,
    reference_col=None,
    metric='ap',
    threshold=0.5,
    y_col=None,
    model_col=None,
    prior_col=None,
    show_reference_lines=True,
    title_suffix="",
    xlabel=None,
    ylabel=None,
    xlabel_suffix="",
    ylabel_suffix="",
    figsize=(6, 6),
    dpi=300,
    title_fontsize=12,
    axis_label_fontsize=10,
    tick_fontsize=9,
    legend_fontsize=8,
    legend_ncol=4,
    point_size=40,
    point_colors=None,
    point_labels=None,
    highlight_cols=None,
    highlight_colors=None,
    highlight_labels=None,
    default_color='lightgrey',
    grid_alpha=0.5,
    reference_line_color='red',
    reference_line_style='--',
    reference_line_width=1,
    reference_line_labels=None,
    xlim=(0, 1),
    ylim=(0, 1),
    suptitle=None,
    suptitle_fontsize=16,
    legend_to_plot_space=-0.08,
    point_markers=None,
    ncols=2,
    allele_col=None,
    alleles=None,
    curve_groups=None,
    curve_colors=None,
    curve_linestyles=None,
    curve_linewidths=None,
    curve_alphas=None,
    curve_labels_in_legend=True,
    curve_smooth_radius=0.02,
    tick_length=None,
    tick_width=None,
    tick_pad=None,
    axis_margin=None,
    spine_width=None,
    title=None,
    save_path=None,
    n_ticks=None,
    use_map=False,
    map_column=None,
    map_axis='both',
    y_ci_low_col=None,
    y_ci_high_col=None,
    ci_line_width=1.0,
    ci_color=None,
    ci_cap_style='line',
    ci_cap_width=4.0,
    hide_legend=False,
    square_aspect=False,
    crosshair_col=None,
    crosshair_linewidth=1.0,
    crosshair_style='--',
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if model_col is not None:
        y_lookup = dict(zip(df2[model_col], df2[y_col]))
    else:
        y_lookup = dict(zip(df2.index, df2[y_col]))

    ci_low_lookup = {}
    ci_high_lookup = {}
    if y_ci_low_col is not None and y_ci_high_col is not None:
        if model_col is not None:
            ci_low_lookup = dict(zip(df2[model_col], df2[y_ci_low_col]))
            ci_high_lookup = dict(zip(df2[model_col], df2[y_ci_high_col]))
        else:
            ci_low_lookup = dict(zip(df2.index, df2[y_ci_low_col]))
            ci_high_lookup = dict(zip(df2.index, df2[y_ci_high_col]))

    if prior_col is not None and not isinstance(prior_col, list):
        prior_col = [prior_col]

    if prior_col is not None:
        n_prior = len(prior_col)
        if not isinstance(reference_line_color, list):
            reference_line_color = [reference_line_color] * n_prior
        if not isinstance(reference_line_style, list):
            reference_line_style = [reference_line_style] * n_prior
        if not isinstance(reference_line_width, (list, tuple)):
            reference_line_width = [reference_line_width] * n_prior
        if reference_line_labels is None:
            reference_line_labels = [f'Prior {i+1}' for i in range(n_prior)]

    if highlight_cols is not None:
        if isinstance(highlight_cols, dict):
            _hl_items = list(highlight_cols.items())
            highlight_cols = [k for k, _ in _hl_items]
            if highlight_colors is None:
                highlight_colors = [v for _, v in _hl_items]
        if highlight_colors is None:
            default_hl = ['orange', 'green', 'purple', 'brown', 'pink', 'cyan', 'magenta']
            highlight_colors = default_hl[:len(highlight_cols)]
        if highlight_labels is None:
            highlight_labels = [f"Model {i+1}" for i in range(len(highlight_cols))]

    if point_colors is None:
        point_colors = {}
    if point_labels is None:
        point_labels = {}
    if point_markers is None:
        point_markers = {}

    if curve_groups is None:
        curve_groups = {}
    if curve_colors is None:
        curve_colors = {}
    if curve_linestyles is None:
        curve_linestyles = {}
    if curve_linewidths is None:
        curve_linewidths = {}
    if curve_alphas is None:
        curve_alphas = {}

    default_curve_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for ci, cname in enumerate(curve_groups):
        curve_colors.setdefault(cname, default_curve_palette[ci % len(default_curve_palette)])
        curve_linestyles.setdefault(cname, '-')
        curve_linewidths.setdefault(cname, 1.5)
        curve_alphas.setdefault(cname, 0.7)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=0.98)

    all_labels = {}

    metric_name = metric.upper() if metric != 'spearman' else 'Spearman Corr'
    if xlabel is None:
        xlabel = f"{metric_name} {xlabel_suffix}".strip()
    if ylabel is None:
        ylabel = f"{metric_name} {ylabel_suffix}".strip()

    def _compute_metric(df_sub, score_col, label_col, ref_col, metric, threshold):
        if score_col not in df_sub.columns:
            warnings.warn(f"score_col '{score_col}' not in dataframe columns")
            return None

        if metric == 'spearman':
            if ref_col is None or ref_col not in df_sub.columns:
                return None
            valid_mask = df_sub[[score_col, ref_col]].notna().all(axis=1)
            valid_data = df_sub[valid_mask]
            if len(valid_data) > 1:
                val, _ = spearmanr(valid_data[score_col], valid_data[ref_col])
                return val
            return None
        else:
            if label_col is None or label_col not in df_sub.columns:
                return None
            valid_mask = df_sub[[score_col, label_col]].notna().all(axis=1)
            valid_data = df_sub[valid_mask]
            if len(valid_data) < 2 or len(np.unique(valid_data[label_col])) < 2:
                return None
            y_true = valid_data[label_col].values
            y_score = valid_data[score_col].values
            try:
                if metric == 'ap':
                    return average_precision_score(y_true, y_score)
                elif metric == 'auroc':
                    return roc_auc_score(y_true, y_score)
                elif metric == 'precision':
                    return precision_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
                elif metric == 'recall':
                    return recall_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
                elif metric == 'f1':
                    return f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
                else:
                    warnings.warn(f"Unknown metric '{metric}', returning None")
                    return None
            except Exception as e:
                warnings.warn(f"Metric computation failed for '{score_col}': {e}")
                return None

    def _compute_metric_map(df_sub, score_col, label_col, ref_col, metric, threshold, map_col):
        if map_col is None or map_col not in df_sub.columns:
            return _compute_metric(df_sub, score_col, label_col, ref_col, metric, threshold)

        groups = df_sub[map_col].dropna().unique()
        metrics = []
        for g in groups:
            g_sub = df_sub[df_sub[map_col] == g]
            val = _compute_metric(g_sub, score_col, label_col, ref_col, metric, threshold)
            if val is not None:
                metrics.append(val)
        return np.mean(metrics) if metrics else None

    def _pareto_front_maximize(cx, cy):
        cx = np.asarray(cx, dtype=float)
        cy = np.asarray(cy, dtype=float)
        m = np.isfinite(cx) & np.isfinite(cy)
        cx = cx[m]
        cy = cy[m]
        if len(cx) == 0:
            return np.array([]), np.array([])
        order = np.argsort(cx)
        cx_s = cx[order]
        cy_s = cy[order]
        ux = []
        uy = []
        i = 0
        while i < len(cx_s):
            x0 = cx_s[i]
            j = i
            y0 = cy_s[i]
            while j < len(cx_s) and cx_s[j] == x0:
                if cy_s[j] > y0:
                    y0 = cy_s[j]
                j += 1
            ux.append(x0)
            uy.append(y0)
            i = j
        ux = np.array(ux, dtype=float)
        uy = np.array(uy, dtype=float)
        rev = np.argsort(-ux)
        ux_r = ux[rev]
        uy_r = uy[rev]
        front_x = []
        front_y = []
        max_y = -np.inf
        for x_i, y_i in zip(ux_r, uy_r):
            if y_i > max_y:
                front_x.append(x_i)
                front_y.append(y_i)
                max_y = y_i
        front_x = np.array(front_x[::-1], dtype=float)
        front_y = np.array(front_y[::-1], dtype=float)
        return front_x, front_y

    def _enforce_monotone_nonincreasing(y):
        y = np.asarray(y, dtype=float).copy()
        for i in range(1, len(y)):
            if y[i] > y[i - 1]:
                y[i] = y[i - 1]
        return y

    def _anchor_and_smooth_frontier(front_x, front_y, smooth_radius, xlim, ylim):
        fx = np.asarray(front_x, dtype=float)
        fy = np.asarray(front_y, dtype=float)
        if len(fx) < 2:
            return fx, fy
        x_left = float(xlim[0]) if xlim is not None else 0.0
        y_bottom = float(ylim[0]) if ylim is not None else 0.0
        order = np.argsort(fx)
        fx = fx[order]
        fy = fy[order]
        y0 = float(np.max(fy))
        if ylim is not None:
            y0 = max(min(y0, float(ylim[1])), float(ylim[0]))
        x_end = float(np.max(fx))
        if xlim is not None:
            x_end = min(max(x_end, float(xlim[0])), float(xlim[1]))
        ctrl_x = np.concatenate([[x_left], fx, [x_end]])
        ctrl_y = np.concatenate([[y0], fy, [y_bottom]])
        order = np.argsort(ctrl_x)
        ctrl_x = ctrl_x[order]
        ctrl_y = ctrl_y[order]
        ux = []
        uy = []
        i = 0
        while i < len(ctrl_x):
            x_val = ctrl_x[i]
            j = i
            y_val = ctrl_y[i]
            while j < len(ctrl_x) and ctrl_x[j] == x_val:
                if ctrl_y[j] > y_val:
                    y_val = ctrl_y[j]
                j += 1
            ux.append(x_val)
            uy.append(y_val)
            i = j
        ux = np.array(ux, dtype=float)
        uy = np.array(uy, dtype=float)
        if len(ux) < 2:
            return ux, uy
        uy_mono = _enforce_monotone_nonincreasing(uy)
        if smooth_radius is None or smooth_radius <= 0 or len(ux) < 3:
            return ux, uy_mono
        try:
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(ux, uy_mono, extrapolate=False)
        except Exception:
            return ux, uy_mono
        x_span = float(ux.max() - ux.min())
        step = float(max(smooth_radius, 1e-6))
        n_samples = int(np.ceil(x_span / step))
        n_samples = max(250, min(4000, n_samples))
        xs = np.linspace(ux.min(), ux.max(), n_samples)
        ys = interp(xs)
        m = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[m]
        ys = ys[m]
        if xlim is not None:
            xs = np.clip(xs, float(xlim[0]), float(xlim[1]))
        if ylim is not None:
            ys = np.clip(ys, float(ylim[0]), float(ylim[1]))
        return xs, ys

    def _get_x_metric(df_sub, score_col, label_col, ref_col, metric, threshold):
        if use_map and map_axis in ('x', 'both'):
            return _compute_metric_map(df_sub, score_col, label_col, ref_col, metric, threshold, map_column)
        return _compute_metric(df_sub, score_col, label_col, ref_col, metric, threshold)

    def _get_y_value(score_col):
        val = y_lookup.get(score_col, None)
        if val is not None and not np.isnan(val):
            return val
        return None

    x_values = []
    y_values = []
    colors = []
    point_label_list = []
    score_cols_used = []
    xy_by_col = {}

    for score_col in score_cols:
        x_val = _get_x_metric(df1, score_col, label_col1, reference_col, metric, threshold)
        if x_val is None:
            continue
        y_val = _get_y_value(score_col)
        if y_val is None:
            continue

        if crosshair_col is not None and score_col == crosshair_col:
            xy_by_col[score_col] = (x_val, y_val)
            continue

        x_values.append(x_val)
        y_values.append(y_val)
        score_cols_used.append(score_col)
        xy_by_col[score_col] = (x_val, y_val)

        color = default_color
        current_label = None

        if score_col in point_colors:
            color = point_colors[score_col]
            if score_col in point_labels:
                current_label = point_labels[score_col]
        elif highlight_cols is not None and score_col in highlight_cols:
            highlight_idx = highlight_cols.index(score_col)
            color = highlight_colors[highlight_idx]
            current_label = highlight_labels[highlight_idx]

        colors.append(color)
        point_label_list.append(current_label)

        if current_label is not None:
            all_labels[current_label] = color

    for cname, ccols in curve_groups.items():
        cx = []
        cy = []
        for sc in ccols:
            if sc in xy_by_col:
                cx.append(xy_by_col[sc][0])
                cy.append(xy_by_col[sc][1])
        if len(cx) < 2:
            continue

        cx = np.array(cx, dtype=float)
        cy = np.array(cy, dtype=float)

        front_x, front_y = _pareto_front_maximize(cx, cy)
        if len(front_x) < 2:
            continue

        path_x, path_y = _anchor_and_smooth_frontier(
            front_x, front_y, curve_smooth_radius, xlim, ylim
        )

        label_for_curve = cname if curve_labels_in_legend else None
        ax.plot(
            path_x, path_y,
            color=curve_colors[cname],
            linestyle=curve_linestyles[cname],
            linewidth=curve_linewidths[cname],
            alpha=curve_alphas[cname],
            label=label_for_curve,
            zorder=1
        )

        if curve_labels_in_legend:
            all_labels[cname] = curve_colors[cname]

    # --- Plot points ---
    legend_labels_used = set()

    for i, (x_val, y_val, color, current_label) in enumerate(
        zip(x_values, y_values, colors, point_label_list)
    ):
        score_col = score_cols_used[i]

        label = None
        if current_label is not None and current_label not in legend_labels_used:
            label = current_label
            legend_labels_used.add(current_label)

        marker_cfg = point_markers.get(score_col, None)

        try:
            if marker_cfg is None:
                ax.scatter(
                    x_val, y_val, color=color, s=point_size,
                    zorder=3 if label else 2,
                    edgecolors='white', linewidths=0.5,
                    label=label
                )
            else:
                m_mk = marker_cfg.get('marker', 'o')
                s = marker_cfg.get('size', point_size)
                lw = marker_cfg.get('edgewidth', 0.5)
                fc = marker_cfg.get('facecolor', None)

                scatter_kwargs = dict(
                    x=x_val, y=y_val, s=s, marker=m_mk, linewidths=lw,
                    zorder=3 if label else 2,
                )

                if isinstance(fc, str) and fc.lower() == 'none':
                    scatter_kwargs['facecolors'] = 'none'
                    scatter_kwargs['edgecolors'] = color
                elif fc is not None:
                    scatter_kwargs['facecolors'] = fc
                    scatter_kwargs['edgecolors'] = color
                else:
                    scatter_kwargs['color'] = color
                    scatter_kwargs['edgecolors'] = 'white'

                if label is not None:
                    scatter_kwargs['label'] = label

                ax.scatter(**scatter_kwargs)
        except Exception as e:
            warnings.warn(f"Scatter failed for '{score_col}' with marker config, falling back: {e}")
            ax.scatter(
                x_val, y_val, color=color, s=point_size,
                zorder=2, edgecolors='white', linewidths=0.5,
                label=label
            )

    if crosshair_col is not None and crosshair_col in xy_by_col:
        ch_x, ch_y = xy_by_col[crosshair_col]
        ch_color = default_color
        if isinstance(point_colors, dict):
            ch_color = point_colors.get(crosshair_col, default_color)
        if highlight_cols is not None and crosshair_col in highlight_cols:
            ch_color = highlight_colors[highlight_cols.index(crosshair_col)]
        ax.axvline(ch_x, color=ch_color, linestyle=crosshair_style,
                   linewidth=crosshair_linewidth, zorder=1)
        ax.axhline(ch_y, color=ch_color, linestyle=crosshair_style,
                   linewidth=crosshair_linewidth, zorder=1)

    # --- Plot Y confidence intervals from df2 ---
    if y_ci_low_col is not None and y_ci_high_col is not None:
        for i, (x_val, y_val) in enumerate(zip(x_values, y_values)):
            score_col = score_cols_used[i]
            ci_lo = ci_low_lookup.get(score_col, None)
            ci_hi = ci_high_lookup.get(score_col, None)
            if ci_lo is None or ci_hi is None:
                continue
            try:
                ci_lo = float(ci_lo)
                ci_hi = float(ci_hi)
            except (ValueError, TypeError):
                continue
            if np.isnan(ci_lo) or np.isnan(ci_hi):
                continue

            line_color = ci_color if ci_color is not None else colors[i]

            ax.plot(
                [x_val, x_val], [ci_lo, ci_hi],
                color=line_color, linewidth=ci_line_width,
                solid_capstyle='butt', zorder=4
            )

            if ci_cap_style == 'cap':
                ax_xlim = xlim if xlim is not None else (min(x_values) - 0.01, max(x_values) + 0.01)
                x_range = ax_xlim[1] - ax_xlim[0]
                half_cap = (ci_cap_width / 1000.0) * x_range
                ax.plot(
                    [x_val - half_cap, x_val + half_cap], [ci_lo, ci_lo],
                    color=line_color, linewidth=ci_line_width, zorder=4
                )
                ax.plot(
                    [x_val - half_cap, x_val + half_cap], [ci_hi, ci_hi],
                    color=line_color, linewidth=ci_line_width, zorder=4
                )

    if metric != 'spearman' and show_reference_lines and prior_col is not None:
        for prior_idx, prior_score in enumerate(prior_col):
            if label_col1 is not None and prior_score in df1.columns:
                valid_mask1_prior = df1[[prior_score, label_col1]].notna().all(axis=1)
                valid_data1_prior = df1[valid_mask1_prior]

                if len(valid_data1_prior) >= 2 and len(np.unique(valid_data1_prior[label_col1])) >= 2:
                    y_true1_prior = valid_data1_prior[label_col1].values
                    y_score1_prior = valid_data1_prior[prior_score].values

                    try:
                        if metric == 'ap':
                            x_prior = average_precision_score(y_true1_prior, y_score1_prior)
                        elif metric == 'auroc':
                            x_prior = roc_auc_score(y_true1_prior, y_score1_prior)
                        elif metric == 'precision':
                            x_prior = precision_score(y_true1_prior, (y_score1_prior >= threshold).astype(int), zero_division=0)
                        elif metric == 'recall':
                            x_prior = recall_score(y_true1_prior, (y_score1_prior >= threshold).astype(int), zero_division=0)
                        elif metric == 'f1':
                            x_prior = f1_score(y_true1_prior, (y_score1_prior >= threshold).astype(int), zero_division=0)

                        ax.axvline(x_prior, color=reference_line_color[prior_idx],
                                   linestyle=reference_line_style[prior_idx],
                                   linewidth=reference_line_width[prior_idx],
                                   label=reference_line_labels[prior_idx], zorder=1)
                        all_labels[reference_line_labels[prior_idx]] = reference_line_color[prior_idx]
                    except Exception:
                        pass

            y_prior = y_lookup.get(prior_score, None)
            if y_prior is not None and not np.isnan(y_prior):
                ax.axhline(y_prior, color=reference_line_color[prior_idx],
                           linestyle=reference_line_style[prior_idx],
                           linewidth=reference_line_width[prior_idx],
                           zorder=1)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    else:
        ax.set_title(f"{metric_name} {title_suffix}".strip(), fontsize=title_fontsize)

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if axis_margin is not None:
        if xlim is not None:
            margin_x = (xlim[1] - xlim[0]) * axis_margin
            ax.set_xlim(xlim[0] - margin_x, xlim[1] + margin_x)
        if ylim is not None:
            margin_y = (ylim[1] - ylim[0]) * axis_margin
            ax.set_ylim(ylim[0] - margin_y, ylim[1] + margin_y)

    def _smart_decimals(ticks):
        for d in range(1, 6):
            fmt = [f'{t:.{d}f}' for t in ticks]
            if len(set(fmt)) == len(ticks):
                return d
        return 2

    if n_ticks is not None:
        if xlim is not None:
            xticks = np.linspace(xlim[0], xlim[1], n_ticks)
            ax.set_xticks(xticks)
            xdec = _smart_decimals(xticks)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=xdec: f'{v:.{d}f}'))
        if ylim is not None:
            yticks = np.linspace(ylim[0], ylim[1], n_ticks)
            ax.set_yticks(yticks)
            ydec = _smart_decimals(yticks)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=ydec: f'{v:.{d}f}'))
    else:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10]))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.canvas.draw_idle()
        xdec = _smart_decimals(ax.get_xticks())
        ydec = _smart_decimals(ax.get_yticks())
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=xdec: f'{v:.{d}f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=ydec: f'{v:.{d}f}'))

    if axis_margin is not None:
        if xlim is not None:
            margin_x = (xlim[1] - xlim[0]) * axis_margin
            ax.set_xlim(xlim[0] - margin_x, xlim[1] + margin_x)
        if ylim is not None:
            margin_y = (ylim[1] - ylim[0]) * axis_margin
            ax.set_ylim(ylim[0] - margin_y, ylim[1] + margin_y)

    tick_kwargs = dict(axis='both', labelsize=tick_fontsize)
    if tick_length is not None:
        tick_kwargs['length'] = tick_length
    if tick_width is not None:
        tick_kwargs['width'] = tick_width
    tick_kwargs['pad'] = tick_pad if tick_pad is not None else 6
    ax.tick_params(**tick_kwargs)

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10

    if spine_width is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

    ax.grid(alpha=grid_alpha)

    if not hide_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='lower center',
                bbox_to_anchor=(0.5, legend_to_plot_space),
                ncol=legend_ncol,
                frameon=False,
                fontsize=legend_fontsize
            )

    if square_aspect:
        ax.set_box_aspect(1)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()
    
def plot_single_metric_comparison_precomputed_y_v2(
    df1, df2, score_cols,

    # X input
    x_precomputed=False,
    x_col=None,
    x_model_col=None,

    # Existing Y input
    label_col1=None,
    reference_col=None,
    metric='ap',
    threshold=0.5,
    y_col=None,
    model_col=None,
    prior_col=None,

    show_reference_lines=True,
    title_suffix="",
    xlabel=None,
    ylabel=None,
    xlabel_suffix="",
    ylabel_suffix="",
    figsize=(6,6),
    dpi=300,
    title_fontsize=12,
    axis_label_fontsize=10,
    tick_fontsize=9,
    legend_fontsize=8,
    legend_ncol=4,
    point_size=40,
    point_colors=None,
    point_labels=None,
    highlight_cols=None,
    highlight_colors=None,
    highlight_labels=None,
    default_color='lightgrey',
    grid_alpha=0.5,
    reference_line_color='red',
    reference_line_style='--',
    reference_line_width=1,
    reference_line_labels=None,
    xlim=(0,1),
    ylim=(0,1),
    suptitle=None,
    suptitle_fontsize=16,
    legend_to_plot_space=-0.08,
    point_markers=None,
    ncols=2,
    allele_col=None,
    alleles=None,
    curve_groups=None,
    curve_colors=None,
    curve_linestyles=None,
    curve_linewidths=None,
    curve_alphas=None,
    curve_labels_in_legend=True,
    curve_smooth_radius=0.02,
    tick_length=None,
    tick_width=None,
    tick_pad=None,
    axis_margin=None,
    spine_width=None,
    title=None,
    save_path=None,
    n_ticks=None,
    use_map=False,
    map_column=None,
    map_axis='both',
    y_ci_low_col=None,
    y_ci_high_col=None,
    ci_line_width=1.0,
    ci_color=None,
    ci_cap_style='line',
    ci_cap_width=4.0,
    hide_legend=False,
    square_aspect=False,
    crosshair_col=None,
    crosshair_linewidth=1.0,
    crosshair_style='--',
):
    import warnings
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

    matplotlib.rcParams['font.family']='sans-serif'
    matplotlib.rcParams['font.sans-serif']=['Nimbus Sans','Nimbus Sans L','Arial','Helvetica']
    matplotlib.rcParams['pdf.fonttype']=42

    # ==============================================================
    # Precomputed Y lookup
    # ==============================================================
    if y_col is None:
        raise ValueError("y_col must be provided.")

    if model_col is not None:
        if model_col not in df2.columns:
            raise ValueError(f"model_col '{model_col}' not found in df2.")
        y_lookup=dict(zip(df2[model_col],df2[y_col]))
    else:
        y_lookup=dict(zip(df2.index,df2[y_col]))

    # Optional Y confidence intervals
    ci_low_lookup={}
    ci_high_lookup={}

    if y_ci_low_col is not None and y_ci_high_col is not None:
        if model_col is not None:
            ci_low_lookup=dict(zip(df2[model_col],df2[y_ci_low_col]))
            ci_high_lookup=dict(zip(df2[model_col],df2[y_ci_high_col]))
        else:
            ci_low_lookup=dict(zip(df2.index,df2[y_ci_low_col]))
            ci_high_lookup=dict(zip(df2.index,df2[y_ci_high_col]))

    # ==============================================================
    # NEW: optional precomputed X lookup
    # ==============================================================
    x_lookup={}

    if x_precomputed:
        if x_col is None:
            raise ValueError("x_col must be provided when x_precomputed=True.")

        if x_col not in df1.columns:
            raise ValueError(f"x_col '{x_col}' not found in df1.")

        if x_model_col is not None:
            if x_model_col not in df1.columns:
                raise ValueError(f"x_model_col '{x_model_col}' not found in df1.")
            x_lookup=dict(zip(df1[x_model_col],df1[x_col]))
        else:
            x_lookup=dict(zip(df1.index,df1[x_col]))

    # ==============================================================
    # Existing setup
    # ==============================================================
    if prior_col is not None and not isinstance(prior_col,list):
        prior_col=[prior_col]

    if prior_col is not None:
        n_prior=len(prior_col)

        if not isinstance(reference_line_color,list):
            reference_line_color=[reference_line_color]*n_prior
        if not isinstance(reference_line_style,list):
            reference_line_style=[reference_line_style]*n_prior
        if not isinstance(reference_line_width,(list,tuple)):
            reference_line_width=[reference_line_width]*n_prior

        if reference_line_labels is None:
            reference_line_labels=[f'Prior {i+1}' for i in range(n_prior)]

    if highlight_cols is not None:
        if isinstance(highlight_cols,dict):
            _hl_items=list(highlight_cols.items())
            highlight_cols=[k for k,_ in _hl_items]
            if highlight_colors is None:
                highlight_colors=[v for _,v in _hl_items]

        if highlight_colors is None:
            default_hl=['orange','green','purple','brown','pink','cyan','magenta']
            highlight_colors=default_hl[:len(highlight_cols)]

        if highlight_labels is None:
            highlight_labels=[f"Model {i+1}" for i in range(len(highlight_cols))]

    if point_colors is None: point_colors={}
    if point_labels is None: point_labels={}
    if point_markers is None: point_markers={}
    if curve_groups is None: curve_groups={}
    if curve_colors is None: curve_colors={}
    if curve_linestyles is None: curve_linestyles={}
    if curve_linewidths is None: curve_linewidths={}
    if curve_alphas is None: curve_alphas={}

    default_curve_palette=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

    for ci,cname in enumerate(curve_groups):
        curve_colors.setdefault(cname,default_curve_palette[ci%len(default_curve_palette)])
        curve_linestyles.setdefault(cname,'-')
        curve_linewidths.setdefault(cname,1.5)
        curve_alphas.setdefault(cname,0.7)

    fig,ax=plt.subplots(1,1,figsize=figsize,dpi=dpi)

    if suptitle is not None:
        fig.suptitle(suptitle,fontsize=suptitle_fontsize,y=0.98)

    all_labels={}
    metric_name=metric.upper() if metric!='spearman' else 'Spearman Corr'

    if xlabel is None:
        xlabel=f"{metric_name} {xlabel_suffix}".strip()

    if ylabel is None:
        ylabel=f"{metric_name} {ylabel_suffix}".strip()

    # ==============================================================
    # Raw metric computation - retained for backward compatibility
    # ==============================================================
    def _compute_metric(df_sub,score_col,label_col,ref_col,metric,threshold):
        if score_col not in df_sub.columns:
            warnings.warn(f"score_col '{score_col}' not in dataframe columns")
            return None

        if metric=='spearman':
            if ref_col is None or ref_col not in df_sub.columns:
                return None

            valid_mask=df_sub[[score_col,ref_col]].notna().all(axis=1)
            valid_data=df_sub[valid_mask]

            if len(valid_data)>1:
                val,_=spearmanr(valid_data[score_col],valid_data[ref_col])
                return val

            return None

        if label_col is None or label_col not in df_sub.columns:
            return None

        valid_mask=df_sub[[score_col,label_col]].notna().all(axis=1)
        valid_data=df_sub[valid_mask]

        if len(valid_data)<2 or len(np.unique(valid_data[label_col]))<2:
            return None

        y_true=valid_data[label_col].values
        y_score=valid_data[score_col].values

        try:
            if metric=='ap':
                return average_precision_score(y_true,y_score)
            elif metric=='auroc':
                return roc_auc_score(y_true,y_score)
            elif metric=='precision':
                return precision_score(y_true,(y_score>=threshold).astype(int),zero_division=0)
            elif metric=='recall':
                return recall_score(y_true,(y_score>=threshold).astype(int),zero_division=0)
            elif metric=='f1':
                return f1_score(y_true,(y_score>=threshold).astype(int),zero_division=0)
            else:
                warnings.warn(f"Unknown metric '{metric}', returning None")
                return None
        except Exception as e:
            warnings.warn(f"Metric computation failed for '{score_col}': {e}")
            return None

    def _compute_metric_map(df_sub,score_col,label_col,ref_col,metric,threshold,map_col):
        if map_col is None or map_col not in df_sub.columns:
            return _compute_metric(df_sub,score_col,label_col,ref_col,metric,threshold)

        groups=df_sub[map_col].dropna().unique()
        metrics=[]

        for g in groups:
            g_sub=df_sub[df_sub[map_col]==g]
            val=_compute_metric(g_sub,score_col,label_col,ref_col,metric,threshold)

            if val is not None:
                metrics.append(val)

        return np.mean(metrics) if metrics else None

    # ==============================================================
    # Pareto / frontier helpers
    # ==============================================================
    def _pareto_front_maximize(cx,cy):
        cx=np.asarray(cx,dtype=float)
        cy=np.asarray(cy,dtype=float)

        m=np.isfinite(cx)&np.isfinite(cy)
        cx=cx[m]
        cy=cy[m]

        if len(cx)==0:
            return np.array([]),np.array([])

        order=np.argsort(cx)
        cx_s=cx[order]
        cy_s=cy[order]

        ux=[]
        uy=[]
        i=0

        while i<len(cx_s):
            x0=cx_s[i]
            j=i
            y0=cy_s[i]

            while j<len(cx_s) and cx_s[j]==x0:
                if cy_s[j]>y0:
                    y0=cy_s[j]
                j+=1

            ux.append(x0)
            uy.append(y0)
            i=j

        ux=np.array(ux,dtype=float)
        uy=np.array(uy,dtype=float)

        rev=np.argsort(-ux)
        ux_r=ux[rev]
        uy_r=uy[rev]

        front_x=[]
        front_y=[]
        max_y=-np.inf

        for x_i,y_i in zip(ux_r,uy_r):
            if y_i>max_y:
                front_x.append(x_i)
                front_y.append(y_i)
                max_y=y_i

        return np.array(front_x[::-1],dtype=float),np.array(front_y[::-1],dtype=float)

    def _enforce_monotone_nonincreasing(y):
        y=np.asarray(y,dtype=float).copy()

        for i in range(1,len(y)):
            if y[i]>y[i-1]:
                y[i]=y[i-1]

        return y

    def _anchor_and_smooth_frontier(front_x,front_y,smooth_radius,xlim,ylim):
        fx=np.asarray(front_x,dtype=float)
        fy=np.asarray(front_y,dtype=float)

        if len(fx)<2:
            return fx,fy

        x_left=float(xlim[0]) if xlim is not None else 0.0
        y_bottom=float(ylim[0]) if ylim is not None else 0.0

        order=np.argsort(fx)
        fx=fx[order]
        fy=fy[order]

        y0=float(np.max(fy))

        if ylim is not None:
            y0=max(min(y0,float(ylim[1])),float(ylim[0]))

        x_end=float(np.max(fx))

        if xlim is not None:
            x_end=min(max(x_end,float(xlim[0])),float(xlim[1]))

        ctrl_x=np.concatenate([[x_left],fx,[x_end]])
        ctrl_y=np.concatenate([[y0],fy,[y_bottom]])

        order=np.argsort(ctrl_x)
        ctrl_x=ctrl_x[order]
        ctrl_y=ctrl_y[order]

        ux=[]
        uy=[]
        i=0

        while i<len(ctrl_x):
            x_val=ctrl_x[i]
            j=i
            y_val=ctrl_y[i]

            while j<len(ctrl_x) and ctrl_x[j]==x_val:
                if ctrl_y[j]>y_val:
                    y_val=ctrl_y[j]
                j+=1

            ux.append(x_val)
            uy.append(y_val)
            i=j

        ux=np.array(ux,dtype=float)
        uy=np.array(uy,dtype=float)

        if len(ux)<2:
            return ux,uy

        uy_mono=_enforce_monotone_nonincreasing(uy)

        if smooth_radius is None or smooth_radius<=0 or len(ux)<3:
            return ux,uy_mono

        try:
            from scipy.interpolate import PchipInterpolator
            interp=PchipInterpolator(ux,uy_mono,extrapolate=False)
        except Exception:
            return ux,uy_mono

        x_span=float(ux.max()-ux.min())
        step=float(max(smooth_radius,1e-6))
        n_samples=int(np.ceil(x_span/step))
        n_samples=max(250,min(4000,n_samples))

        xs=np.linspace(ux.min(),ux.max(),n_samples)
        ys=interp(xs)

        m=np.isfinite(xs)&np.isfinite(ys)
        xs=xs[m]
        ys=ys[m]

        if xlim is not None:
            xs=np.clip(xs,float(xlim[0]),float(xlim[1]))

        if ylim is not None:
            ys=np.clip(ys,float(ylim[0]),float(ylim[1]))

        return xs,ys

    # ==============================================================
    # X and Y retrieval
    # ==============================================================
    def _get_x_metric(df_sub,score_col,label_col,ref_col,metric,threshold):
        # NEW behavior
        if x_precomputed:
            val=x_lookup.get(score_col,None)

            if val is None:
                return None

            try:
                val=float(val)
            except (TypeError,ValueError):
                return None

            return val if np.isfinite(val) else None

        # Original behavior
        if use_map and map_axis in ('x','both'):
            return _compute_metric_map(
                df_sub,score_col,label_col,ref_col,
                metric,threshold,map_column
            )

        return _compute_metric(
            df_sub,score_col,label_col,ref_col,
            metric,threshold
        )

    def _get_y_value(score_col):
        val=y_lookup.get(score_col,None)

        if val is not None:
            try:
                val=float(val)
            except (TypeError,ValueError):
                return None

            if np.isfinite(val):
                return val

        return None

    # ==============================================================
    # Gather XY coordinates
    # ==============================================================
    x_values=[]
    y_values=[]
    colors=[]
    point_label_list=[]
    score_cols_used=[]
    xy_by_col={}

    for score_col in score_cols:
        x_val=_get_x_metric(df1,score_col,label_col1,reference_col,metric,threshold)

        if x_val is None:
            continue

        y_val=_get_y_value(score_col)

        if y_val is None:
            continue

        if crosshair_col is not None and score_col==crosshair_col:
            xy_by_col[score_col]=(x_val,y_val)
            continue

        x_values.append(x_val)
        y_values.append(y_val)
        score_cols_used.append(score_col)
        xy_by_col[score_col]=(x_val,y_val)

        color=default_color
        current_label=None

        if score_col in point_colors:
            color=point_colors[score_col]

            if score_col in point_labels:
                current_label=point_labels[score_col]

        elif highlight_cols is not None and score_col in highlight_cols:
            highlight_idx=highlight_cols.index(score_col)
            color=highlight_colors[highlight_idx]
            current_label=highlight_labels[highlight_idx]

        colors.append(color)
        point_label_list.append(current_label)

        if current_label is not None:
            all_labels[current_label]=color

    # ==============================================================
    # Optional Pareto curves
    # ==============================================================
    for cname,ccols in curve_groups.items():
        cx=[]
        cy=[]

        for sc in ccols:
            if sc in xy_by_col:
                cx.append(xy_by_col[sc][0])
                cy.append(xy_by_col[sc][1])

        if len(cx)<2:
            continue

        cx=np.asarray(cx,dtype=float)
        cy=np.asarray(cy,dtype=float)

        front_x,front_y=_pareto_front_maximize(cx,cy)

        if len(front_x)<2:
            continue

        path_x,path_y=_anchor_and_smooth_frontier(
            front_x,front_y,curve_smooth_radius,xlim,ylim
        )

        label_for_curve=cname if curve_labels_in_legend else None

        ax.plot(
            path_x,path_y,
            color=curve_colors[cname],
            linestyle=curve_linestyles[cname],
            linewidth=curve_linewidths[cname],
            alpha=curve_alphas[cname],
            label=label_for_curve,
            zorder=1
        )

        if curve_labels_in_legend:
            all_labels[cname]=curve_colors[cname]

    # ==============================================================
    # Points
    # ==============================================================
    legend_labels_used=set()

    for i,(x_val,y_val,color,current_label) in enumerate(
        zip(x_values,y_values,colors,point_label_list)
    ):
        score_col=score_cols_used[i]
        label=None

        if current_label is not None and current_label not in legend_labels_used:
            label=current_label
            legend_labels_used.add(current_label)

        marker_cfg=point_markers.get(score_col,None)

        try:
            if marker_cfg is None:
                ax.scatter(
                    x_val,y_val,color=color,s=point_size,
                    zorder=3 if label else 2,
                    edgecolors='white',linewidths=0.5,label=label
                )
            else:
                m_mk=marker_cfg.get('marker','o')
                s=marker_cfg.get('size',point_size)
                lw=marker_cfg.get('edgewidth',0.5)
                fc=marker_cfg.get('facecolor',None)

                scatter_kwargs=dict(
                    x=x_val,y=y_val,s=s,marker=m_mk,linewidths=lw,
                    zorder=3 if label else 2
                )

                if isinstance(fc,str) and fc.lower()=='none':
                    scatter_kwargs['facecolors']='none'
                    scatter_kwargs['edgecolors']=color
                elif fc is not None:
                    scatter_kwargs['facecolors']=fc
                    scatter_kwargs['edgecolors']=color
                else:
                    scatter_kwargs['color']=color
                    scatter_kwargs['edgecolors']='white'

                if label is not None:
                    scatter_kwargs['label']=label

                ax.scatter(**scatter_kwargs)

        except Exception as e:
            warnings.warn(f"Scatter failed for '{score_col}', falling back: {e}")
            ax.scatter(
                x_val,y_val,color=color,s=point_size,
                zorder=2,edgecolors='white',linewidths=0.5,label=label
            )

    # ==============================================================
    # Crosshair
    # ==============================================================
    if crosshair_col is not None and crosshair_col in xy_by_col:
        ch_x,ch_y=xy_by_col[crosshair_col]
        ch_color=default_color

        if isinstance(point_colors,dict):
            ch_color=point_colors.get(crosshair_col,default_color)

        if highlight_cols is not None and crosshair_col in highlight_cols:
            ch_color=highlight_colors[highlight_cols.index(crosshair_col)]

        ax.axvline(
            ch_x,color=ch_color,linestyle=crosshair_style,
            linewidth=crosshair_linewidth,zorder=1
        )

        ax.axhline(
            ch_y,color=ch_color,linestyle=crosshair_style,
            linewidth=crosshair_linewidth,zorder=1
        )

    # ==============================================================
    # Y confidence intervals
    # ==============================================================
    if y_ci_low_col is not None and y_ci_high_col is not None:
        for i,(x_val,y_val) in enumerate(zip(x_values,y_values)):
            score_col=score_cols_used[i]
            ci_lo=ci_low_lookup.get(score_col,None)
            ci_hi=ci_high_lookup.get(score_col,None)

            if ci_lo is None or ci_hi is None:
                continue

            try:
                ci_lo=float(ci_lo)
                ci_hi=float(ci_hi)
            except (ValueError,TypeError):
                continue

            if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
                continue

            line_color=ci_color if ci_color is not None else colors[i]

            ax.plot(
                [x_val,x_val],[ci_lo,ci_hi],
                color=line_color,linewidth=ci_line_width,
                solid_capstyle='butt',zorder=4
            )

            if ci_cap_style=='cap':
                ax_xlim=xlim if xlim is not None else (
                    min(x_values)-0.01,
                    max(x_values)+0.01
                )

                x_range=ax_xlim[1]-ax_xlim[0]
                half_cap=(ci_cap_width/1000.0)*x_range

                ax.plot(
                    [x_val-half_cap,x_val+half_cap],[ci_lo,ci_lo],
                    color=line_color,linewidth=ci_line_width,zorder=4
                )

                ax.plot(
                    [x_val-half_cap,x_val+half_cap],[ci_hi,ci_hi],
                    color=line_color,linewidth=ci_line_width,zorder=4
                )

    # ==============================================================
    # Prior/reference lines
    # ==============================================================
    if metric!='spearman' and show_reference_lines and prior_col is not None:
        for prior_idx,prior_score in enumerate(prior_col):

            # X prior
            if x_precomputed:
                x_prior=x_lookup.get(prior_score,None)

                if x_prior is not None:
                    try:
                        x_prior=float(x_prior)
                    except (TypeError,ValueError):
                        x_prior=None

                if x_prior is not None and np.isfinite(x_prior):
                    ax.axvline(
                        x_prior,
                        color=reference_line_color[prior_idx],
                        linestyle=reference_line_style[prior_idx],
                        linewidth=reference_line_width[prior_idx],
                        label=reference_line_labels[prior_idx],
                        zorder=1
                    )

                    all_labels[
                        reference_line_labels[prior_idx]
                    ]=reference_line_color[prior_idx]

            elif label_col1 is not None and prior_score in df1.columns:
                valid_mask1_prior=df1[[prior_score,label_col1]].notna().all(axis=1)
                valid_data1_prior=df1[valid_mask1_prior]

                if len(valid_data1_prior)>=2 and len(np.unique(valid_data1_prior[label_col1]))>=2:
                    y_true1_prior=valid_data1_prior[label_col1].values
                    y_score1_prior=valid_data1_prior[prior_score].values

                    try:
                        if metric=='ap':
                            x_prior=average_precision_score(y_true1_prior,y_score1_prior)
                        elif metric=='auroc':
                            x_prior=roc_auc_score(y_true1_prior,y_score1_prior)
                        elif metric=='precision':
                            x_prior=precision_score(
                                y_true1_prior,(y_score1_prior>=threshold).astype(int),zero_division=0
                            )
                        elif metric=='recall':
                            x_prior=recall_score(
                                y_true1_prior,(y_score1_prior>=threshold).astype(int),zero_division=0
                            )
                        elif metric=='f1':
                            x_prior=f1_score(
                                y_true1_prior,(y_score1_prior>=threshold).astype(int),zero_division=0
                            )
                        else:
                            x_prior=None

                        if x_prior is not None:
                            ax.axvline(
                                x_prior,
                                color=reference_line_color[prior_idx],
                                linestyle=reference_line_style[prior_idx],
                                linewidth=reference_line_width[prior_idx],
                                label=reference_line_labels[prior_idx],
                                zorder=1
                            )

                            all_labels[
                                reference_line_labels[prior_idx]
                            ]=reference_line_color[prior_idx]

                    except Exception:
                        pass

            # Y prior
            y_prior=y_lookup.get(prior_score,None)

            if y_prior is not None:
                try:
                    y_prior=float(y_prior)
                except (TypeError,ValueError):
                    y_prior=None

            if y_prior is not None and np.isfinite(y_prior):
                ax.axhline(
                    y_prior,
                    color=reference_line_color[prior_idx],
                    linestyle=reference_line_style[prior_idx],
                    linewidth=reference_line_width[prior_idx],
                    zorder=1
                )

    # ==============================================================
    # Titles / axes
    # ==============================================================
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    else:
        ax.set_title(f"{metric_name} {title_suffix}".strip(),fontsize=title_fontsize)

    ax.set_xlabel(xlabel,fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel,fontsize=axis_label_fontsize)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if axis_margin is not None:
        if xlim is not None:
            margin_x=(xlim[1]-xlim[0])*axis_margin
            ax.set_xlim(xlim[0]-margin_x,xlim[1]+margin_x)

        if ylim is not None:
            margin_y=(ylim[1]-ylim[0])*axis_margin
            ax.set_ylim(ylim[0]-margin_y,ylim[1]+margin_y)

    def _smart_decimals(ticks):
        for d in range(1,6):
            fmt=[f'{t:.{d}f}' for t in ticks]
            if len(set(fmt))==len(ticks):
                return d
        return 2

    if n_ticks is not None:
        if xlim is not None:
            xticks=np.linspace(xlim[0],xlim[1],n_ticks)
            ax.set_xticks(xticks)
            xdec=_smart_decimals(xticks)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v,_,d=xdec:f'{v:.{d}f}')
            )

        if ylim is not None:
            yticks=np.linspace(ylim[0],ylim[1],n_ticks)
            ax.set_yticks(yticks)
            ydec=_smart_decimals(yticks)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v,_,d=ydec:f'{v:.{d}f}')
            )

    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto',steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins='auto',steps=[1,2,2.5,5,10]))

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        fig.canvas.draw_idle()

        xdec=_smart_decimals(ax.get_xticks())
        ydec=_smart_decimals(ax.get_yticks())

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_,d=xdec:f'{v:.{d}f}')
        )

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_,d=ydec:f'{v:.{d}f}')
        )

    if axis_margin is not None:
        if xlim is not None:
            margin_x=(xlim[1]-xlim[0])*axis_margin
            ax.set_xlim(xlim[0]-margin_x,xlim[1]+margin_x)

        if ylim is not None:
            margin_y=(ylim[1]-ylim[0])*axis_margin
            ax.set_ylim(ylim[0]-margin_y,ylim[1]+margin_y)

    tick_kwargs=dict(axis='both',labelsize=tick_fontsize)

    if tick_length is not None:
        tick_kwargs['length']=tick_length

    if tick_width is not None:
        tick_kwargs['width']=tick_width

    tick_kwargs['pad']=tick_pad if tick_pad is not None else 6

    ax.tick_params(**tick_kwargs)

    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10

    if spine_width is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

    ax.grid(alpha=grid_alpha)

    if not hide_legend:
        handles,labels=ax.get_legend_handles_labels()
        by_label=dict(zip(labels,handles))

        if by_label:
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='lower center',
                bbox_to_anchor=(0.5,legend_to_plot_space),
                ncol=legend_ncol,
                frameon=False,
                fontsize=legend_fontsize
            )

    if square_aspect:
        ax.set_box_aspect(1)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf",format='pdf',bbox_inches='tight',dpi=300)
        fig.savefig(f"{save_path}.png",format='png',bbox_inches='tight',dpi=300)

    plt.show()

    return fig,ax

def plot_simple_bars(
    data_dict,
    colors=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    bar_width=0.8,
    bar_spacing=0.0,
    bar_outline_color='black',
    bar_outline_width=1.0,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    show_legend=True,
    save_path=None,
):

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (8, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params
    
    default_colors = [
        '#1f77b4',
        '#d62728',
        '#2ca02c',
        '#ff7f0e',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]
    
    x_labels = list(data_dict.keys())
    y_values = list(data_dict.values())
    n_bars = len(x_labels)
    
    if colors is None:
        colors = {label: default_colors[i % len(default_colors)] 
                  for i, label in enumerate(x_labels)}
    
    fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])
    
    x_positions = np.arange(n_bars) * (bar_width + bar_spacing)
    
    for i, (label, value) in enumerate(zip(x_labels, y_values)):
        color = colors.get(label, default_colors[i % len(default_colors)])
        
        ax.bar(x_positions[i], value,
               width=bar_width,
               color=color,
               edgecolor=bar_outline_color,
               linewidth=bar_outline_width,
               label=label,
               zorder=2)
    
    if hline_y is not None:
        label_for_line = hline_label if hline_in_legend else None
        ax.axhline(y=hline_y, color=hline_color, linestyle=hline_style, 
                   linewidth=hline_width, label=label_for_line, zorder=1)
        
        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            
            x_range = x_positions[-1] - x_positions[0] if len(x_positions) > 1 else 1
            text_x = x_positions[0] + (x_range * hline_text_x_offset)
            
            if hline_text_position == 'above':
                text_y = hline_y + hline_text_offset
                va = 'bottom'
            else:
                text_y = hline_y - hline_text_offset
                va = 'top'
            
            ax.text(text_x, text_y, hline_text, 
                   color=text_color, 
                   fontsize=hline_text_fontsize,
                   fontfamily=default_font,
                   va=va, ha='left',
                   zorder=3)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    
    ax.set_xlabel(params['xlabel'], fontsize=params['xlabel_fontsize'], 
                  fontfamily=default_font, labelpad=params['xlabel_pad'])
    ax.set_ylabel(params['ylabel'], fontsize=params['ylabel_fontsize'], 
                  fontfamily=default_font, labelpad=params['ylabel_pad'])
    
    title_weight = 'bold' if params['title_bold'] else 'normal'
    ax.set_title(params['title'], fontsize=params['title_fontsize'], 
                 fontfamily=default_font, fontweight=title_weight)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])
    
    ax.tick_params(axis='both', which='major', 
                   length=params['tick_length'],
                   width=params['axis_linewidth'],
                   direction='out',
                   pad=params['tick_length'] + 2)
    
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, 
                axis='y', zorder=0)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(default_font)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()
    
    if show_legend:
        n_items = n_bars + (1 if hline_y is not None and hline_in_legend else 0)
        fig_legend = plt.figure(figsize=(params['legend_handle_size'] * 2, 
                                          n_items * 0.4 * params['legend_handle_size']),
                                dpi=params['dpi'])
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        
        handles = []
        labels_list = []
        
        for label in x_labels:
            color = colors.get(label, default_colors[0])
            handle = mpatches.Patch(facecolor=color, 
                                   edgecolor=bar_outline_color,
                                   linewidth=bar_outline_width)
            handles.append(handle)
            labels_list.append(label)
        
        if hline_y is not None and hline_label is not None and hline_in_legend:
            hline_handle = Line2D([0], [0], color=hline_color, linestyle=hline_style, 
                                 linewidth=hline_width * params['legend_handle_size'])
            handles.append(hline_handle)
            labels_list.append(hline_label)
        legend = ax_legend.legend(handles, labels_list, 
                                 loc='center', 
                                 frameon=False,
                                 fontsize=params['legend_fontsize'],
                                 ncol=params['legend_cols'])
        
        for text in legend.get_texts():
            text.set_fontfamily(default_font)
        
        plt.tight_layout()
        plt.show()
        
def plot_simple_bars_v2(
    data_dict,
    colors=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    bar_width=0.8,
    bar_spacing=0.0,
    bar_outline_color='black',
    bar_outline_width=1.0,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    show_legend=True,
    save_path=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    default_params={
        'title':'',
        'title_fontsize':14,
        'title_bold':True,
        'xlabel':'',
        'ylabel':'',
        'xlabel_fontsize':12,
        'ylabel_fontsize':12,
        'figsize':(8,6),
        'dpi':100,
        'legend_fontsize':10,
        'legend_cols':1,
        'legend_handle_size':1.0,
        'axis_linewidth':0.8,
        'tick_length':4.0,
        'xlabel_pad':4.0,
        'ylabel_pad':4.0,

        # Tick label styling.
        'xtick_labelsize':10,
        'ytick_labelsize':10,
        'xtick_rotation':0,
        'xtick_ha':'center',
        'xtick_va':'top',
        'ytick_rotation':0,
        'ytick_ha':'right',
        'ytick_va':'center',
    }

    if plot_params is not None:
        default_params.update(plot_params)
    params=default_params

    default_colors=[
        '#1f77b4',
        '#d62728',
        '#2ca02c',
        '#ff7f0e',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]

    x_labels=list(data_dict.keys())
    y_values=list(data_dict.values())
    n_bars=len(x_labels)

    if colors is None:
        colors={label:default_colors[i%len(default_colors)] for i,label in enumerate(x_labels)}

    fig,ax=plt.subplots(figsize=params['figsize'],dpi=params['dpi'])
    x_positions=np.arange(n_bars)*(bar_width+bar_spacing)

    for i,(label,value) in enumerate(zip(x_labels,y_values)):
        color=colors.get(label,default_colors[i%len(default_colors)])

        ax.bar(
            x_positions[i],
            value,
            width=bar_width,
            color=color,
            edgecolor=bar_outline_color,
            linewidth=bar_outline_width,
            label=label,
            zorder=2
        )

    if hline_y is not None:
        label_for_line=hline_label if hline_in_legend else None

        ax.axhline(
            y=hline_y,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            label=label_for_line,
            zorder=1
        )

        if hline_text is not None:
            text_color=hline_text_color if hline_text_color is not None else hline_color
            x_range=x_positions[-1]-x_positions[0] if len(x_positions)>1 else 1
            text_x=x_positions[0]+x_range*hline_text_x_offset

            if hline_text_position=='above':
                text_y=hline_y+hline_text_offset
                va='bottom'
            else:
                text_y=hline_y-hline_text_offset
                va='top'

            ax.text(
                text_x,
                text_y,
                hline_text,
                color=text_color,
                fontsize=hline_text_fontsize,
                fontfamily=default_font,
                va=va,
                ha='left',
                zorder=3
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        x_labels,
        rotation=params['xtick_rotation'],
        ha=params['xtick_ha'],
        va=params['xtick_va'],
        rotation_mode='anchor'
    )

    ax.set_xlabel(
        params['xlabel'],
        fontsize=params['xlabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['xlabel_pad']
    )

    ax.set_ylabel(
        params['ylabel'],
        fontsize=params['ylabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['ylabel_pad']
    )

    title_weight='bold' if params['title_bold'] else 'normal'

    ax.set_title(
        params['title'],
        fontsize=params['title_fontsize'],
        fontfamily=default_font,
        fontweight=title_weight
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(
        axis='both',
        which='major',
        length=params['tick_length'],
        width=params['axis_linewidth'],
        direction='out',
        pad=params['tick_length']+2
    )

    ax.tick_params(axis='x',labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y',labelsize=params['ytick_labelsize'])

    for label in ax.get_yticklabels():
        label.set_rotation(params['ytick_rotation'])
        label.set_ha(params['ytick_ha'])
        label.set_va(params['ytick_va'])
        label.set_rotation_mode('anchor')

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha>0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis='y',
            zorder=0
        )

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf",format='pdf',bbox_inches='tight',dpi=300)
        fig.savefig(f"{save_path}.png",format='png',bbox_inches='tight',dpi=300)

    plt.show()

    if show_legend:
        n_items=n_bars+(1 if hline_y is not None and hline_in_legend else 0)

        fig_legend=plt.figure(
            figsize=(
                params['legend_handle_size']*2,
                n_items*0.4*params['legend_handle_size']
            ),
            dpi=params['dpi']
        )

        ax_legend=fig_legend.add_subplot(111)
        ax_legend.axis('off')

        handles=[]
        labels_list=[]

        for label in x_labels:
            color=colors.get(label,default_colors[0])

            handle=mpatches.Patch(
                facecolor=color,
                edgecolor=bar_outline_color,
                linewidth=bar_outline_width
            )

            handles.append(handle)
            labels_list.append(label)

        if hline_y is not None and hline_label is not None and hline_in_legend:
            hline_handle=Line2D(
                [0],[0],
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width*params['legend_handle_size']
            )

            handles.append(hline_handle)
            labels_list.append(hline_label)

        legend=ax_legend.legend(
            handles,
            labels_list,
            loc='center',
            frameon=False,
            fontsize=params['legend_fontsize'],
            ncol=params['legend_cols']
        )

        for text in legend.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()
        plt.show()

def plot_violins_sns(
    df,
    columns,
    colors=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    violin_width=0.3,
    violin_outline_color='black',
    violin_outline_width=1.0,
    violin_patch_alpha=0.9,
    violin_inner='box',
    violin_cut=2,
    violin_bw_method='scott',
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    reference_hlines=None,
    legend_mode='integrated',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    save_path=None,
):

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (8, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    if reference_hlines is None:
        reference_hlines = {}

    hline_col_set = set(reference_hlines.keys())

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = {label: default_colors[i % len(default_colors)]
                  for i, label in enumerate(columns.keys())}

    violin_items = []
    hline_entries = []

    for label, col_name in columns.items():
        if col_name in hline_col_set:
            vals = df[col_name].dropna().values
            if len(vals) > 0:
                style = reference_hlines[col_name]
                hline_entries.append((vals.mean(), style, col_name))
        else:
            violin_items.append((label, col_name, colors.get(label, default_colors[0])))

    fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])

    long_rows = []
    for label, col_name, color in violin_items:
        vals = df[col_name].dropna().values
        for v in vals:
            long_rows.append({'value': v, 'group': label})
    long_df = pd.DataFrame(long_rows)

    order = [label for label, _, _ in violin_items]
    palette = {label: color for label, _, color in violin_items}

    sns.violinplot(
        data=long_df,
        x='group',
        y='value',
        order=order,
        palette=palette,
        ax=ax,
        linewidth=violin_outline_width,
        cut=violin_cut,
        inner=violin_inner,
        width=violin_width * 0.9,
        bw_method=violin_bw_method,
        saturation=1.0,
    )

    for collection in ax.collections:
        collection.set_alpha(violin_patch_alpha)
        collection.set_edgecolor(violin_outline_color)

    for y_val, style, col_name in hline_entries:
        ax.axhline(y=y_val, color=style.get('color', 'gray'),
                   linestyle=style.get('linestyle', '--'),
                   linewidth=style.get('linewidth', 1.5),
                   alpha=style.get('alpha', 1.0), zorder=1)

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None
        ax.axhline(y=hline_y, color=hline_color, linestyle=hline_style,
                   linewidth=hline_width, label=line_label, zorder=1)

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            n_violins = len(violin_items)
            x_range = n_violins - 1 if n_violins > 1 else 1
            text_x = x_range * hline_text_x_offset
            text_y = hline_y + hline_text_offset if hline_text_position == 'above' else hline_y - hline_text_offset
            va = 'bottom' if hline_text_position == 'above' else 'top'
            ax.text(text_x, text_y, hline_text, color=text_color,
                    fontsize=hline_text_fontsize, fontfamily=default_font,
                    va=va, ha='left', zorder=3)

    ax.set_xlabel(params['xlabel'], fontsize=params['xlabel_fontsize'],
                  fontfamily=default_font, labelpad=params['xlabel_pad'])
    ax.set_ylabel(params['ylabel'], fontsize=params['ylabel_fontsize'],
                  fontfamily=default_font, labelpad=params['ylabel_pad'])

    title_weight = 'bold' if params['title_bold'] else 'normal'
    ax.set_title(params['title'], fontsize=params['title_fontsize'],
                 fontfamily=default_font, fontweight=title_weight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(axis='both', which='major',
                   length=params['tick_length'],
                   width=params['axis_linewidth'],
                   direction='out', pad=params['tick_length'] + 2)
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, axis='y', zorder=0)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []
    for label, col_name, color in violin_items:
        handles.append(mpatches.Patch(facecolor=color, edgecolor=violin_outline_color,
                                      linewidth=violin_outline_width, alpha=violin_patch_alpha))
        labels_list.append(label)

    for y_val, style, col_name in hline_entries:
        handles.append(Line2D([0], [0],
                              color=style.get('color', 'gray'),
                              linestyle=style.get('linestyle', '--'),
                              linewidth=style.get('linewidth', 1.5) * params['legend_handle_size'],
                              alpha=style.get('alpha', 1.0)))
        labels_list.append(style.get('label', col_name))

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(Line2D([0], [0], color=hline_color, linestyle=hline_style,
                              linewidth=hline_width * params['legend_handle_size']))
        labels_list.append(hline_label)

    if legend_mode == 'integrated' and handles:
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        legend = fig.legend(handles, labels_list, loc=legend_loc,
                            bbox_to_anchor=bbox, ncol=legend_ncol,
                            frameon=False, fontsize=params['legend_fontsize'])
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(f"{save_path}{suffix}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}{suffix}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

    if legend_mode == 'separate' and handles:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(params['legend_handle_size'] * 2,
                     n_items * 0.4 * params['legend_handle_size']),
            dpi=params['dpi'])
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        legend = ax_legend.legend(handles, labels_list, loc='center', frameon=False,
                                  fontsize=params['legend_fontsize'], ncol=legend_ncol)
        for text in legend.get_texts():
            text.set_fontfamily(default_font)
        plt.tight_layout()
        if save_path:
            fig_legend.savefig(f"{save_path}_legend.pdf", format='pdf', bbox_inches='tight', dpi=300)
            fig_legend.savefig(f"{save_path}_legend.png", format='png', bbox_inches='tight', dpi=300)
        plt.show()

def plot_bar_grouped_advanced(
    df,
    cols,
    figsize=(10, 4),
    dpi=150,
    default_font='Nimbus Sans',
    colors=None,
    bar_width=0.25,
    group_spacing=0.3,
    bar_edgecolor='none',
    bar_edgewidth=0.3,
    alpha=1.0,
    x_margin=None,
    horizontal=False,
    xlabel='',
    ylabel='',
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    title='',
    title_fontsize=12,
    title_bold=True,
    xtick_labelsize=8,
    ytick_labelsize=8,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction='out',
    rotation=0,
    axis_linewidth=0.8,
    ylim=None,
    xlim=None,
    grid_alpha=0.0,
    legend_mode='separate',
    legend_loc='upper right',
    legend_bbox=None,
    legend_ncol=1,
    legend_fontsize=9,
    legend_handle_size=1.0,
    legend_labels=None,
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(len(cols))]

    if legend_labels is None:
        legend_labels = cols

    labels = df.index.tolist()
    n_groups = len(labels)
    n_bars = len(cols)

    cluster_width = n_bars * bar_width
    group_positions = np.array([
        i * (cluster_width + group_spacing) + cluster_width / 2.0
        for i in range(n_groups)
    ])

    edge_margin = x_margin if x_margin is not None else group_spacing / 2.0
    axis_min = group_positions[0]  - cluster_width / 2.0 - edge_margin
    axis_max = group_positions[-1] + cluster_width / 2.0 + edge_margin

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    handles = []
    for bar_idx, (col, color) in enumerate(zip(cols, colors)):
        values = df[col].values
        offset = -cluster_width / 2.0 + bar_idx * bar_width + bar_width / 2.0
        positions = group_positions + offset

        if horizontal:
            bars = ax.barh(
                positions, values,
                color=color,
                edgecolor=bar_edgecolor,
                linewidth=bar_edgewidth,
                alpha=alpha,
                height=bar_width,
            )
        else:
            bars = ax.bar(
                positions, values,
                color=color,
                edgecolor=bar_edgecolor,
                linewidth=bar_edgewidth,
                alpha=alpha,
                width=bar_width,
            )

        handles.append(bars[0])

    if horizontal:
        ax.set_ylim(axis_min, axis_max)
        ax.set_yticks(group_positions)
        ax.set_yticklabels(labels, fontsize=ytick_labelsize)
        ax.invert_yaxis()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    else:
        ax.set_xlim(axis_min, axis_max)
        ax.set_xticks(group_positions)
        ax.set_xticklabels(labels, rotation=rotation, ha='center', fontsize=xtick_labelsize)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad)

    title_weight = 'bold' if title_bold else 'normal'
    ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font, fontweight=title_weight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(axis_linewidth)
    ax.spines['bottom'].set_linewidth(axis_linewidth)

    ax.tick_params(
        axis='both', which='major',
        length=tick_length,
        width=tick_width,
        direction=tick_direction,
        pad=tick_length + 2,
    )
    ax.tick_params(axis='x', labelsize=xtick_labelsize)
    ax.tick_params(axis='y', labelsize=ytick_labelsize)

    if grid_alpha > 0:
        grid_axis = 'x' if horizontal else 'y'
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, axis=grid_axis, zorder=0)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if legend_mode == 'integrated':
        legend_kw = dict(frameon=False, fontsize=legend_fontsize, ncol=legend_ncol)
        if legend_bbox is not None:
            legend_kw.update(bbox_to_anchor=legend_bbox, loc=legend_loc)
        else:
            legend_kw.update(loc=legend_loc)
        legend = ax.legend(handles, legend_labels, **legend_kw)
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(f"{save_path}{suffix}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}{suffix}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

    if legend_mode == 'separate':
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(legend_handle_size * 2, n_items * 0.4 * legend_handle_size),
            dpi=dpi,
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        legend = ax_legend.legend(
            handles, legend_labels,
            loc='center', frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
        )
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(f"{save_path}_legend.pdf", format='pdf', bbox_inches='tight', dpi=300)
            fig_legend.savefig(f"{save_path}_legend.png", format='png', bbox_inches='tight', dpi=300)

        plt.show()

def plot_bar_grouped_advanced_v2(
    df,
    cols,
    figsize=(10,4),
    dpi=150,
    default_font='Nimbus Sans',
    colors=None,
    bar_width=0.25,
    group_spacing=0.3,
    bar_edgecolor='none',
    bar_edgewidth=0.3,
    alpha=1.0,
    x_margin=None,
    horizontal=False,
    xlabel='',
    ylabel='',
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    title='',
    title_fontsize=12,
    title_bold=True,
    xtick_labelsize=8,
    ytick_labelsize=8,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction='out',
    rotation=0,
    axis_linewidth=0.8,
    ylim=None,
    xlim=None,
    grid_alpha=0.0,

    # Confidence intervals
    interval_df=None,
    interval_low_suffix="_ci_low",
    interval_high_suffix="_ci_high",
    interval_color="black",
    interval_linewidth=0.8,
    interval_linestyle="-",
    interval_capsize=0.0,
    interval_capthick=None,
    interval_alpha=1.0,
    interval_zorder=4,

    legend_mode='separate',
    legend_loc='upper right',
    legend_bbox=None,
    legend_ncol=1,
    legend_fontsize=9,
    legend_handle_size=1.0,
    legend_labels=None,
    save_path=None,
):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams['font.family']='sans-serif'
    matplotlib.rcParams['font.sans-serif']=[
        'Nimbus Sans','Nimbus Sans L','Arial','Helvetica'
    ]
    matplotlib.rcParams['pdf.fonttype']=42

    default_colors=[
        '#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd',
        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
    ]

    if colors is None:
        colors=[
            default_colors[i%len(default_colors)]
            for i in range(len(cols))
        ]

    if legend_labels is None:
        legend_labels=cols

    labels=df.index.tolist()
    n_groups=len(labels)
    n_bars=len(cols)

    cluster_width=n_bars*bar_width

    group_positions=np.array([
        i*(cluster_width+group_spacing)+cluster_width/2.0
        for i in range(n_groups)
    ])

    edge_margin=x_margin if x_margin is not None else group_spacing/2.0

    axis_min=group_positions[0]-cluster_width/2.0-edge_margin
    axis_max=group_positions[-1]+cluster_width/2.0+edge_margin

    if interval_df is not None:
        interval_df=interval_df.reindex(df.index)

    def _interval_setting(value,col,idx):
        if isinstance(value,dict):
            return value.get(col,None)
        if isinstance(value,(list,tuple,np.ndarray)):
            return value[idx]
        return value

    fig,ax=plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    handles=[]

    for bar_idx,(col,color) in enumerate(
        zip(cols,colors)
    ):
        values=pd.to_numeric(
            df[col],
            errors="coerce"
        ).to_numpy(float)

        offset=(
            -cluster_width/2.0
            +bar_idx*bar_width
            +bar_width/2.0
        )

        positions=group_positions+offset

        if horizontal:
            bars=ax.barh(
                positions,
                values,
                color=color,
                edgecolor=bar_edgecolor,
                linewidth=bar_edgewidth,
                alpha=alpha,
                height=bar_width,
            )

        else:
            bars=ax.bar(
                positions,
                values,
                color=color,
                edgecolor=bar_edgecolor,
                linewidth=bar_edgewidth,
                alpha=alpha,
                width=bar_width,
            )

        handles.append(
            bars[0]
        )

        # ----------------------------------------------------------
        # Optional confidence intervals
        # ----------------------------------------------------------
        if interval_df is not None:
            low_col=f"{col}{interval_low_suffix}"
            high_col=f"{col}{interval_high_suffix}"

            if (
                low_col in interval_df.columns
                and high_col in interval_df.columns
            ):
                low=pd.to_numeric(
                    interval_df[low_col],
                    errors="coerce"
                ).to_numpy(float)

                high=pd.to_numeric(
                    interval_df[high_col],
                    errors="coerce"
                ).to_numpy(float)

                valid=(
                    np.isfinite(values)
                    & np.isfinite(low)
                    & np.isfinite(high)
                )

                if np.any(valid):
                    lower_err=values[valid]-low[valid]
                    upper_err=high[valid]-values[valid]

                    if np.any(lower_err<0) or np.any(upper_err<0):
                        raise ValueError(
                            f"Invalid confidence interval for '{col}': "
                            "expected ci_low <= value <= ci_high."
                        )

                    err=np.vstack([
                        lower_err,
                        upper_err
                    ])

                    ecolor=_interval_setting(
                        interval_color,
                        col,
                        bar_idx
                    )

                    elw=_interval_setting(
                        interval_linewidth,
                        col,
                        bar_idx
                    )

                    els=_interval_setting(
                        interval_linestyle,
                        col,
                        bar_idx
                    )

                    ealpha=_interval_setting(
                        interval_alpha,
                        col,
                        bar_idx
                    )

                    capthick=(
                        elw
                        if interval_capthick is None
                        else _interval_setting(
                            interval_capthick,
                            col,
                            bar_idx
                        )
                    )

                    if horizontal:
                        err_container=ax.errorbar(
                            values[valid],
                            positions[valid],
                            xerr=err,
                            fmt='none',
                            ecolor=ecolor,
                            elinewidth=elw,
                            capsize=interval_capsize,
                            capthick=capthick,
                            alpha=ealpha,
                            zorder=interval_zorder,
                        )

                    else:
                        err_container=ax.errorbar(
                            positions[valid],
                            values[valid],
                            yerr=err,
                            fmt='none',
                            ecolor=ecolor,
                            elinewidth=elw,
                            capsize=interval_capsize,
                            capthick=capthick,
                            alpha=ealpha,
                            zorder=interval_zorder,
                        )

                    # errorbar line collections can be styled independently
                    for collection in err_container.lines[2]:
                        collection.set_linestyle(els)

    if horizontal:
        ax.set_ylim(
            axis_min,
            axis_max
        )

        ax.set_yticks(
            group_positions
        )

        ax.set_yticklabels(
            labels,
            fontsize=ytick_labelsize
        )

        ax.invert_yaxis()

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    else:
        ax.set_xlim(
            axis_min,
            axis_max
        )

        ax.set_xticks(
            group_positions
        )

        ax.set_xticklabels(
            labels,
            rotation=rotation,
            ha='center',
            fontsize=xtick_labelsize
        )

        if ylim is not None:
            ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)

    ax.set_xlabel(
        xlabel,
        fontsize=xlabel_fontsize,
        fontfamily=default_font,
        labelpad=xlabel_pad
    )

    ax.set_ylabel(
        ylabel,
        fontsize=ylabel_fontsize,
        fontfamily=default_font,
        labelpad=ylabel_pad
    )

    title_weight='bold' if title_bold else 'normal'

    ax.set_title(
        title,
        fontsize=title_fontsize,
        fontfamily=default_font,
        fontweight=title_weight
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(
        axis_linewidth
    )

    ax.spines['bottom'].set_linewidth(
        axis_linewidth
    )

    ax.tick_params(
        axis='both',
        which='major',
        length=tick_length,
        width=tick_width,
        direction=tick_direction,
        pad=tick_length+2,
    )

    ax.tick_params(
        axis='x',
        labelsize=xtick_labelsize
    )

    ax.tick_params(
        axis='y',
        labelsize=ytick_labelsize
    )

    if grid_alpha>0:
        grid_axis='x' if horizontal else 'y'

        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis=grid_axis,
            zorder=0
        )

    for lbl in (
        ax.get_xticklabels()
        +ax.get_yticklabels()
    ):
        lbl.set_fontfamily(
            default_font
        )

    if legend_mode=='integrated':
        legend_kw=dict(
            frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol
        )

        if legend_bbox is not None:
            legend_kw.update(
                bbox_to_anchor=legend_bbox,
                loc=legend_loc
            )

        else:
            legend_kw.update(
                loc=legend_loc
            )

        legend=ax.legend(
            handles,
            legend_labels,
            **legend_kw
        )

        for text in legend.get_texts():
            text.set_fontfamily(
                default_font
            )

    plt.tight_layout()

    if save_path:
        suffix='_plot' if legend_mode=='separate' else ''

        fig.savefig(
            f"{save_path}{suffix}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300
        )

        fig.savefig(
            f"{save_path}{suffix}.png",
            format='png',
            bbox_inches='tight',
            dpi=300
        )

    plt.show()

    if legend_mode=='separate':
        n_items=len(handles)

        fig_legend=plt.figure(
            figsize=(
                legend_handle_size*2,
                n_items*0.4*legend_handle_size
            ),
            dpi=dpi,
        )

        ax_legend=fig_legend.add_subplot(111)
        ax_legend.axis('off')

        legend=ax_legend.legend(
            handles,
            legend_labels,
            loc='center',
            frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
        )

        for text in legend.get_texts():
            text.set_fontfamily(
                default_font
            )

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(
                f"{save_path}_legend.pdf",
                format='pdf',
                bbox_inches='tight',
                dpi=300
            )

            fig_legend.savefig(
                f"{save_path}_legend.png",
                format='png',
                bbox_inches='tight',
                dpi=300
            )

        plt.show()

def plot_lines(
    df,
    x_col,
    y_cols,
    line_styles=None,
    ci_df=None,
    hline_y=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_alpha=1.0,
    hline_label=None,
    hline_zorder=1,
    figsize=(8, 5),
    dpi=150,
    default_font='Nimbus Sans',
    xlabel='',
    ylabel='',
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    title='',
    title_fontsize=12,
    title_bold=True,
    xtick_labelsize=9,
    ytick_labelsize=9,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction='out',
    xtick_rotation=0,
    axis_linewidth=0.8,
    xlim=None,
    ylim=None,
    grid_alpha=0.0,
    legend_mode='separate',
    legend_loc='upper right',
    legend_bbox=None,
    legend_ncol=1,
    legend_fontsize=9,
    legend_handle_size=1.0,
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if line_styles is None:
        line_styles = {}

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    handles = []
    labels_list = []

    for i, col in enumerate(y_cols):
        style = line_styles.get(col, {})

        color       = style.get('color',            default_colors[i % len(default_colors)])
        linewidth   = style.get('linewidth',        1.5)
        linestyle   = style.get('linestyle',        '-')
        marker      = style.get('marker',           None)
        markersize  = style.get('markersize',       4)
        markerfacecolor = style.get('markerfacecolor', color)
        markeredgewidth = style.get('markeredgewidth', 0.8)
        alpha       = style.get('alpha',            1.0)
        label       = style.get('label',            col)
        zorder      = style.get('zorder',           2)

        line, = ax.plot(
            df[x_col],
            df[col],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            markerfacecolor=markerfacecolor,
            markeredgewidth=markeredgewidth,
            markeredgecolor=color,
            alpha=alpha,
            zorder=zorder,
        )

        if ci_df is not None:
            ci_low_col  = f'{col}_ci_low'
            ci_high_col = f'{col}_ci_high'
            if ci_low_col in ci_df.columns and ci_high_col in ci_df.columns:
                ci_color = style.get('ci_color', color)
                ci_alpha = style.get('ci_alpha', 0.15)
                ax.fill_between(
                    ci_df[x_col],
                    ci_df[ci_low_col],
                    ci_df[ci_high_col],
                    color=ci_color,
                    alpha=ci_alpha,
                    linewidth=0,
                    zorder=zorder - 1,
                )

        handles.append(line)
        labels_list.append(label)

    if hline_y is not None:
        ax.axhline(
            y=hline_y,
            color=hline_color,
            linewidth=hline_width,
            linestyle=hline_style,
            alpha=hline_alpha,
            zorder=hline_zorder,
        )
        if hline_label is not None:
            handles.append(Line2D([0], [0], color=hline_color, linewidth=hline_width,
                                  linestyle=hline_style, alpha=hline_alpha))
            labels_list.append(hline_label)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, axis='y', zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(axis_linewidth)
    ax.spines['bottom'].set_linewidth(axis_linewidth)

    ax.tick_params(
        axis='both', which='major',
        length=tick_length,
        width=tick_width,
        direction=tick_direction,
        pad=tick_length + 2,
    )
    ax.tick_params(axis='x', labelsize=xtick_labelsize)
    ax.tick_params(axis='y', labelsize=ytick_labelsize)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=xtick_rotation,
        ha='center',
    )

    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad)

    title_weight = 'bold' if title_bold else 'normal'
    ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font, fontweight=title_weight)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if legend_mode == 'integrated':
        bbox = legend_bbox if legend_bbox is not None else None
        legend_kw = dict(frameon=False, fontsize=legend_fontsize, ncol=legend_ncol)
        if bbox is not None:
            legend_kw.update(bbox_to_anchor=bbox, loc=legend_loc)
        else:
            legend_kw.update(loc=legend_loc)
        legend = ax.legend(handles, labels_list, **legend_kw)
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(f"{save_path}{suffix}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}{suffix}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

    if legend_mode == 'separate':
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(legend_handle_size * 2, n_items * 0.4 * legend_handle_size),
            dpi=dpi,
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        legend = ax_legend.legend(
            handles, labels_list,
            loc='center', frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
        )
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(f"{save_path}_legend.pdf", format='pdf', bbox_inches='tight', dpi=300)
            fig_legend.savefig(f"{save_path}_legend.png", format='png', bbox_inches='tight', dpi=300)

        plt.show()

def plot_split_violin_grid_with_ap_modified_betweenx_updated(
    dfs,
    allele_col=None,
    label_cols=None,
    score_col=None,
    aggregate_ap='total',
    hide_ap=False,
    neg_color='#74a9cf',
    pos_color='#e34a33',
    neg_edge_color='black',
    pos_edge_color='black',
    neg_edge_width=0.5,
    pos_edge_width=0.5,
    ap_line_color='black',
    ap_marker='o',
    ap_marker_size=4,
    ap_line_width=1.0,
    figsize=None,
    dpi=150,
    suptitle=None,
    ylabel_violin=None,
    ylabel_ap='AP',
    tick_fontsize=8,
    label_fontsize=9,
    title_fontsize=10,
    suptitle_fontsize=12,
    legend_fontsize=8,
    violin_width=0.8,
    violin_alpha=0.8,
    ylim_violin=None,
    ap_ylim=(-0.05, 1.05),
    height_ratio=0.3,
    hspace=0.05,
    kde_extend=3.0,
    kde_points=200,
    spine_width_violin=None,
    spine_width_ap=None,
    tick_width_violin=None,
    tick_width_ap=None,
    tick_length_violin=None,
    tick_length_ap=None,
    ap_tick_margin=None,
    hide_legend=False,
    square_aspect=False,
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    df_names = list(dfs.keys())
    n_dfs = len(df_names)

    if isinstance(label_cols, str):
        label_cols = {name: label_cols for name in df_names}

    if allele_col is not None:
        first_df = dfs[df_names[0]]
        panels = sorted(first_df[allele_col].unique())
    else:
        panels = [None]

    n_panels = len(panels)

    if figsize is None:
        if n_panels == 1:
            figsize = (5, 6 if not hide_ap else 5)
        else:
            figsize = (max(3 * n_panels, 6), 6 if not hide_ap else 5)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    if hide_ap:
        gs = GridSpec(1, n_panels, figure=fig)
    else:
        gs = GridSpec(
            2, n_panels, figure=fig,
            height_ratios=[height_ratio, 1],
            hspace=hspace,
        )

    ax_aps = []
    ax_violins = []
    for pi in range(n_panels):
        if hide_ap:
            ax_v = fig.add_subplot(gs[0, pi])
            ax_aps.append(None)
        else:
            ax_ap = fig.add_subplot(gs[0, pi])
            ax_v = fig.add_subplot(gs[1, pi])
            ax_aps.append(ax_ap)
        ax_violins.append(ax_v)

    positions = np.arange(n_dfs)

    def _compute_ap_for_df(df, lcol, score_col, allele_col, panel, aggregate_ap):
        if allele_col is not None and panel is not None:
            sub = df[df[allele_col] == panel]
        else:
            sub = df

        if aggregate_ap == 'per_allele' and allele_col is not None:
            alleles_in_df = sub[allele_col].unique()
            aps = []
            for a in alleles_in_df:
                a_sub = sub[sub[allele_col] == a]
                valid = a_sub[[score_col, lcol]].dropna()
                if len(valid) > 0 and valid[lcol].nunique() == 2:
                    aps.append(average_precision_score(valid[lcol], valid[score_col]))
            return np.mean(aps) if aps else None
        else:
            valid = sub[[score_col, lcol]].dropna()
            if len(valid) > 0 and valid[lcol].nunique() == 2:
                return average_precision_score(valid[lcol], valid[score_col])
            return None

    def _get_scores(df, lcol, score_col, allele_col, panel):
        if allele_col is not None and panel is not None:
            sub = df[df[allele_col] == panel]
        else:
            sub = df
        neg = sub.loc[sub[lcol] == 0, score_col].dropna().values
        pos = sub.loc[sub[lcol] == 1, score_col].dropna().values
        return neg, pos

    def _style_spines(ax, spine_width, sides=('left', 'bottom')):
        if spine_width is not None:
            for s in sides:
                if ax.spines[s].get_visible():
                    ax.spines[s].set_linewidth(spine_width)

    def _make_half_violin_polygon(y_eval, density, center, side='left'):
        if side == 'left':
            x_edge = center - density
            x_center = np.full_like(y_eval, center)
            verts_x = np.concatenate([x_edge, x_center[::-1]])
            verts_y = np.concatenate([y_eval, y_eval[::-1]])
        else:
            x_center = np.full_like(y_eval, center)
            x_edge = center + density
            verts_x = np.concatenate([x_center, x_edge[::-1]])
            verts_y = np.concatenate([y_eval, y_eval[::-1]])
        return np.column_stack([verts_x, verts_y])

    for pi, panel in enumerate(panels):
        ax_ap = ax_aps[pi]
        ax_v = ax_violins[pi]

        ap_values = []
        kde_ymin = np.inf
        kde_ymax = -np.inf

        for di, df_name in enumerate(df_names):
            df = dfs[df_name]
            lcol = label_cols[df_name]

            neg_scores, pos_scores = _get_scores(df, lcol, score_col, allele_col, panel)

            ap = _compute_ap_for_df(df, lcol, score_col, allele_col, panel, aggregate_ap)
            ap_values.append(ap)

            if len(neg_scores) > 1:
                kde = gaussian_kde(neg_scores)
                bw = kde.scotts_factor() * neg_scores.std()
                y_eval = np.linspace(
                    neg_scores.min() - kde_extend * bw,
                    neg_scores.max() + kde_extend * bw,
                    kde_points
                )
                density = kde(y_eval)
                density = density / density.max() * (violin_width / 2)
                kde_ymin = min(kde_ymin, y_eval[0])
                kde_ymax = max(kde_ymax, y_eval[-1])

                verts = _make_half_violin_polygon(y_eval, density, center=di, side='left')
                poly = MplPolygon(
                    verts, closed=True,
                    facecolor=neg_color, edgecolor=neg_edge_color,
                    linewidth=neg_edge_width, alpha=violin_alpha, zorder=2
                )
                ax_v.add_patch(poly)

            if len(pos_scores) > 1:
                kde = gaussian_kde(pos_scores)
                bw = kde.scotts_factor() * pos_scores.std()
                y_eval = np.linspace(
                    pos_scores.min() - kde_extend * bw,
                    pos_scores.max() + kde_extend * bw,
                    kde_points
                )
                density = kde(y_eval)
                density = density / density.max() * (violin_width / 2)
                kde_ymin = min(kde_ymin, y_eval[0])
                kde_ymax = max(kde_ymax, y_eval[-1])

                verts = _make_half_violin_polygon(y_eval, density, center=di, side='right')
                poly = MplPolygon(
                    verts, closed=True,
                    facecolor=pos_color, edgecolor=pos_edge_color,
                    linewidth=pos_edge_width, alpha=violin_alpha, zorder=2
                )
                ax_v.add_patch(poly)

        ax_v.set_xticks(positions)
        ax_v.set_xticklabels(df_names, fontsize=tick_fontsize)
        for lbl in ax_v.get_xticklabels():
            lbl.set_fontfamily('Nimbus Sans')
        ax_v.tick_params(axis='y', labelsize=tick_fontsize)
        ax_v.spines['top'].set_visible(False)
        ax_v.spines['right'].set_visible(False)
        _style_spines(ax_v, spine_width_violin)

        v_tick_kw = {}
        if tick_width_violin is not None:
            v_tick_kw['width'] = tick_width_violin
        if tick_length_violin is not None:
            v_tick_kw['length'] = tick_length_violin
        if v_tick_kw:
            ax_v.tick_params(axis='both', **v_tick_kw)

        if ylim_violin is not None:
            ax_v.set_ylim(ylim_violin)
        elif np.isfinite(kde_ymin) and np.isfinite(kde_ymax):
            kde_range = kde_ymax - kde_ymin
            ax_v.set_ylim(kde_ymin - kde_range * 0.02, kde_ymax + kde_range * 0.02)

        ax_v.set_xlim(-0.5, n_dfs - 0.5)
        ax_v.autoscale_view()

        if square_aspect:
            ax_v.set_box_aspect(1)

        if pi == 0:
            ax_v.set_ylabel(ylabel_violin or score_col, fontsize=label_fontsize)
        else:
            ax_v.tick_params(axis='y', labelleft=False)

        for lbl in ax_v.get_yticklabels():
            lbl.set_fontfamily('Nimbus Sans')

        # --- Title placement ---
        if hide_ap:
            if panel is not None:
                ax_v.set_title(panel, fontsize=title_fontsize)
        else:
            valid_mask = [v is not None for v in ap_values]
            valid_pos = [positions[i] for i in range(n_dfs) if valid_mask[i]]
            valid_ap = [ap_values[i] for i in range(n_dfs) if valid_mask[i]]

            if valid_ap:
                ax_ap.plot(valid_pos, valid_ap, color=ap_line_color,
                           linewidth=ap_line_width, marker=ap_marker,
                           markersize=ap_marker_size, zorder=2)

            ap_lo, ap_hi = ap_ylim
            nice_ticks = [v for v in [0, 0.5, 1.0] if ap_lo <= v <= ap_hi]
            if len(nice_ticks) < 2:
                nice_ticks = np.linspace(ap_lo, ap_hi, 3).tolist()

            if ap_tick_margin is not None:
                tick_lo = nice_ticks[0]
                tick_hi = nice_ticks[-1]
                ax_ap.set_ylim(tick_lo - ap_tick_margin, tick_hi + ap_tick_margin)
            else:
                ax_ap.set_ylim(ap_ylim)

            ax_ap.set_yticks(nice_ticks)
            ax_ap.set_yticklabels(
                [f'{v:g}' for v in nice_ticks], fontsize=tick_fontsize
            )
            for lbl in ax_ap.get_yticklabels():
                lbl.set_fontfamily('Nimbus Sans')

            ax_ap.set_xlim(-0.5, n_dfs - 0.5)

            ax_ap.set_xticks([])
            ax_ap.set_xticklabels([])
            ax_ap.tick_params(axis='x', length=0)

            ax_ap.spines['bottom'].set_visible(True)
            ax_ap.spines['top'].set_visible(False)
            ax_ap.spines['right'].set_visible(False)
            _style_spines(ax_ap, spine_width_ap)

            ap_tick_kw = {}
            if tick_width_ap is not None:
                ap_tick_kw['width'] = tick_width_ap
            if tick_length_ap is not None:
                ap_tick_kw['length'] = tick_length_ap
            if ap_tick_kw:
                ax_ap.tick_params(axis='both', **ap_tick_kw)

            if square_aspect:
                ax_ap.set_box_aspect(1)

            if panel is not None:
                ax_ap.set_title(panel, fontsize=title_fontsize)

            if pi == 0:
                ax_ap.set_ylabel(ylabel_ap, fontsize=label_fontsize)
            else:
                ax_ap.tick_params(axis='y', labelleft=False)

    for ax_v in ax_violins:
        ax_v.set_xlim(-0.5, n_dfs - 0.5)

    if not hide_legend:
        legend_handles = [
            Patch(facecolor=neg_color, edgecolor=neg_edge_color,
                  linewidth=neg_edge_width, alpha=violin_alpha),
            Patch(facecolor=pos_color, edgecolor=pos_edge_color,
                  linewidth=pos_edge_width, alpha=violin_alpha),
        ]
        legend = fig.legend(
            legend_handles, ['Negative', 'Positive'],
            loc='lower center', bbox_to_anchor=(0.5, -0.04),
            ncol=2, frameon=False, fontsize=legend_fontsize
        )
        for text in legend.get_texts():
            text.set_fontfamily('Nimbus Sans')

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=1.02,
                     fontfamily='Nimbus Sans')

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_grouped_boxplots(
    groups,
    colors=None,
    legend=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    box_width=0.6,
    group_spacing=1.0,
    box_spacing=0.05,
    box_outline_color='black',
    box_outline_width=1.0,
    box_flier_marker='o',
    box_flier_size=3,
    box_flier_color=None,
    box_flier_alpha=0.6,
    box_flier_edge_width=0,
    box_median_color='black',
    box_median_width=1.5,
    box_whisker_style='-',
    box_whisker_width=1.0,
    box_notch=False,
    box_patch_alpha=0.9,
    sort_by=None,
    sort_ascending=False,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    reference_hlines=None,
    legend_mode='integrated',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    save_path=None,
    compute_pvalues=False,
    pvalue_test='wilcoxon',
    cluster_col=None,

    # New p-value annotation options
    pval_df=None,
    pval_ref_col=None,
    pval_ref_stars_only=True,
    pval_format='stars',
    pval_fontsize=8,
    pval_color='black',
    pval_star_y=None,
    pval_star_y_frac=0.985,
    pval_star_text_offset=0.0,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (10, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
        'xtick_rotation': 0,
        'title_y': None,
        'title_pad': 6.0,

    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    if reference_hlines is None:
        reference_hlines = {}

    hline_col_set = set(reference_hlines.keys())

    def _normalize_group(group_value):
        """
        Normalize group value to a list of (df, col) pairs.
        Accepts:
          - (df, col1, col2, ...)           single tuple
          - [(df1, col1), (df2, col2, ...)] list of tuples
        Returns: [(df, col), (df, col), ...
        """
        if isinstance(group_value, list):
            tuples = group_value
        else:
            tuples = [group_value]
        pairs = []
        for t in tuples:
            df_g = t[0]
            cols = t[1:]
            for c in cols:
                pairs.append((df_g, c))
        return pairs

    def _pval_to_stars(p):
        if isinstance(p, str):
            p_str = p.strip()
            if p_str in {'ns', '*', '**', '***', '****'}:
                return p_str
            try:
                p = float(p_str)
            except ValueError:
                return p_str

        if pd.isna(p):
            return None

        p = float(p)

        if p <= 1e-4:
            return '****'
        elif p <= 1e-3:
            return '***'
        elif p <= 1e-2:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return 'ns'

    def _lookup_symmetric_pval(pval_table, col_a, col_b):
        if pval_table is None:
            return None

        if col_a in pval_table.index and col_b in pval_table.columns:
            return pval_table.loc[col_a, col_b]
        elif col_b in pval_table.index and col_a in pval_table.columns:
            return pval_table.loc[col_b, col_a]

        return None

    all_cols = set()
    for tick_label, group_value in groups.items():
        for df_g, col in _normalize_group(group_value):
            all_cols.add(col)

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = {}
        for i, col in enumerate(sorted(all_cols)):
            colors[col] = default_colors[i % len(default_colors)]

    hline_entries = []
    hline_vals_accum = {}
    for tick_label, group_value in groups.items():
        for df_g, col in _normalize_group(group_value):
            if col in hline_col_set and col in df_g.columns:
                if col not in hline_vals_accum:
                    hline_vals_accum[col] = []
                hline_vals_accum[col].extend(df_g[col].dropna().values.tolist())

    for col_name, vals in hline_vals_accum.items():
        if vals:
            style = reference_hlines[col_name]
            agg = style.get('agg', 'mean')
            if agg == 'median':
                y_val = np.median(vals)
            else:
                y_val = np.mean(vals)
            hline_entries.append((y_val, style, col_name))

    filtered_groups = {}
    for tick_label, group_value in groups.items():
        pairs = [
            (df_g, col)
            for df_g, col in _normalize_group(group_value)
            if col not in hline_col_set
        ]
        if pairs:
            filtered_groups[tick_label] = pairs

    if sort_by is not None:
        def _group_stat(pairs):
            vals = []
            for df_g, col in pairs:
                if col in df_g.columns:
                    vals.extend(df_g[col].dropna().values.tolist())
            if not vals:
                return np.nan
            return np.median(vals) if sort_by == 'median' else np.mean(vals)

        def _sort_key(item):
            stat = _group_stat(item[1])
            if np.isnan(stat):
                return np.inf if sort_ascending else -np.inf
            return stat

        filtered_groups = dict(
            sorted(
                filtered_groups.items(),
                key=_sort_key,
                reverse=not sort_ascending,
            )
        )

    max_cols = max(len(pairs) for pairs in filtered_groups.values()) if filtered_groups else 1
    n_groups = len(filtered_groups)

    fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])

    group_width = max_cols * box_width + (max_cols - 1) * box_spacing
    group_centers = np.arange(n_groups) * (group_width + group_spacing)

    tick_labels = []

    # Store box centers for later p-value star annotation.
    plotted_positions = {}
    plotted_upper_whiskers = {}

    def _upper_boxplot_whisker(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        valid = vals[vals <= upper_limit]

        if len(valid) == 0:
            return np.max(vals)

        return np.max(valid)

    for gi, (tick_label, pairs) in enumerate(filtered_groups.items()):
        tick_labels.append(tick_label)
        n_cols = len(pairs)

        total_box_width = n_cols * box_width + (n_cols - 1) * box_spacing
        start = group_centers[gi] - total_box_width / 2 + box_width / 2

        for ci, (df_g, col) in enumerate(pairs):
            if col not in df_g.columns:
                continue
            vals = df_g[col].dropna().values.tolist()
            if not vals:
                continue

            x_pos = start + ci * (box_width + box_spacing)
            color = colors.get(col, default_colors[0])
            flier_c = box_flier_color if box_flier_color is not None else color

            plotted_positions[(tick_label, col)] = x_pos
            plotted_upper_whiskers[(tick_label, col)] = _upper_boxplot_whisker(vals)

            bp = ax.boxplot(
                [vals],
                positions=[x_pos],
                widths=box_width * 0.9,
                patch_artist=True,
                notch=box_notch,
                manage_ticks=False,
                zorder=2,
                flierprops=dict(
                    marker=box_flier_marker,
                    markersize=box_flier_size,
                    markerfacecolor=flier_c,
                    markeredgecolor=flier_c,
                    markeredgewidth=box_flier_edge_width,
                    alpha=box_flier_alpha,
                ),
                medianprops=dict(
                    color=box_median_color,
                    linewidth=box_median_width,
                ),
                whiskerprops=dict(
                    color=box_outline_color,
                    linewidth=box_whisker_width,
                    linestyle=box_whisker_style,
                ),
                capprops=dict(
                    color='none',
                    linewidth=0,
                ),
                boxprops=dict(
                    facecolor=color,
                    edgecolor=box_outline_color,
                    linewidth=box_outline_width,
                    alpha=box_patch_alpha,
                ),
            )

    for y_val, style, col_name in hline_entries:
        ax.axhline(
            y=y_val,
            color=style.get('color', 'gray'),
            linestyle=style.get('linestyle', '--'),
            linewidth=style.get('linewidth', 1.5),
            alpha=style.get('alpha', 1.0),
            zorder=3,
        )

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None
        ax.axhline(
            y=hline_y,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            label=line_label,
            zorder=3,
        )

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            if n_groups > 1:
                x_range = group_centers[-1] - group_centers[0]
            else:
                x_range = 1
            text_x = group_centers[0] + (x_range * hline_text_x_offset)
            if hline_text_position == 'above':
                text_y = hline_y + hline_text_offset
                va = 'bottom'
            else:
                text_y = hline_y - hline_text_offset
                va = 'top'
            ax.text(
                text_x,
                text_y,
                hline_text,
                color=text_color,
                fontsize=hline_text_fontsize,
                fontfamily=default_font,
                va=va,
                ha='left',
                zorder=3,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        tick_labels,
        rotation=params['xtick_rotation'],
        ha='right' if params['xtick_rotation'] != 0 else 'center',
    )

    ax.set_xlabel(
        params['xlabel'],
        fontsize=params['xlabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['xlabel_pad'],
    )
    ax.set_ylabel(
        params['ylabel'],
        fontsize=params['ylabel_fontsize'],
        fontfamily=default_font,
        labelpad=params['ylabel_pad'],
    )

    title_weight = 'bold' if params['title_bold'] else 'normal'
    ax.set_title(
        params['title'],
        fontsize=params['title_fontsize'],
        fontfamily=default_font,
        fontweight=title_weight,
        y=params.get('title_y', None),
        pad=params.get('title_pad', 6.0),

    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(
        axis='both',
        which='major',
        length=params['tick_length'],
        width=params['axis_linewidth'],
        direction='out',
        pad=params['tick_length'] + 2,
    )
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis='y',
            zorder=0,
        )

    # Reference-column p-value star annotation.
    if (
        pval_df is not None
        and pval_ref_col is not None
        and pval_ref_stars_only
        and len(plotted_positions) > 0
    ):
        y_low, y_high = ax.get_ylim()

        if pval_star_y is None:
            star_y_base = y_low + (y_high - y_low) * pval_star_y_frac
        else:
            star_y_base = pval_star_y

        star_y = star_y_base + pval_star_text_offset

        for (tick_label, col), x_pos in plotted_positions.items():
            if col == pval_ref_col:
                continue

            pval_value = _lookup_symmetric_pval(pval_df, pval_ref_col, col)

            if pval_value is None:
                continue
            if pd.isna(pval_value):
                continue

            if pval_format == 'stars':
                pval_text = _pval_to_stars(pval_value)
            else:
                pval_text = str(pval_value)

            if pval_text is None:
                continue
            if str(pval_text) == '':
                continue

            ax.text(
                x_pos,
                star_y,
                str(pval_text),
                ha='center',
                va='bottom',
                fontsize=pval_fontsize,
                fontfamily=default_font,
                color=pval_color,
                clip_on=False,
                zorder=10,
            )

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []

    if legend is not None:
        for lbl_text, color in legend.items():
            handles.append(
                mpatches.Patch(
                    facecolor=color,
                    edgecolor=box_outline_color,
                    linewidth=box_outline_width,
                    alpha=box_patch_alpha,
                )
            )
            labels_list.append(lbl_text)

    for y_val, style, col_name in hline_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get('color', 'gray'),
                linestyle=style.get('linestyle', '--'),
                linewidth=style.get('linewidth', 1.5) * params['legend_handle_size'],
                alpha=style.get('alpha', 1.0),
            )
        )
        labels_list.append(style.get('label', col_name))

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(
            Line2D(
                [0],
                [0],
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width * params['legend_handle_size'],
            )
        )
        labels_list.append(hline_label)

    if legend_mode == 'integrated' and handles:
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        leg = fig.legend(
            handles,
            labels_list,
            loc=legend_loc,
            bbox_to_anchor=bbox,
            ncol=legend_ncol,
            frameon=False,
            fontsize=params['legend_fontsize'],
        )
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(
            f"{save_path}{suffix}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )
        fig.savefig(
            f"{save_path}{suffix}.png",
            format='png',
            bbox_inches='tight',
            dpi=300,
        )

    plt.show()

    if legend_mode == 'separate' and handles:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(
                params['legend_handle_size'] * 2,
                n_items * 0.4 * params['legend_handle_size'],
            ),
            dpi=params['dpi'],
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')

        leg = ax_legend.legend(
            handles,
            labels_list,
            loc='center',
            frameon=False,
            fontsize=params['legend_fontsize'],
            ncol=legend_ncol,
        )
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(
                f"{save_path}_legend.pdf",
                format='pdf',
                bbox_inches='tight',
                dpi=300,
            )
            fig_legend.savefig(
                f"{save_path}_legend.png",
                format='png',
                dpi=300,
            )

        plt.show()

    if compute_pvalues:

        all_pairs = []
        for tick_label, group_value in groups.items():
            all_pairs.extend(_normalize_group(group_value))

        seen = set()
        unique_pairs = []
        for df_g, col in all_pairs:
            if col not in seen:
                seen.add(col)
                unique_pairs.append((df_g, col))

        col_names = [col for _, col in unique_pairs]
        n = len(col_names)
        pval_matrix = np.full((n, n), np.nan)

        for i in range(n):
            df_i, col_i = unique_pairs[i]
            for j in range(n):
                if i == j:
                    pval_matrix[i, j] = 1.0
                    continue
                if not np.isnan(pval_matrix[j, i]):
                    pval_matrix[i, j] = pval_matrix[j, i]
                    continue

                df_j, col_j = unique_pairs[j]

                if (
                    cluster_col is not None
                    and cluster_col in df_i.columns
                    and cluster_col in df_j.columns
                ):
                    merged = pd.merge(
                        df_i[[cluster_col, col_i]].dropna(),
                        df_j[[cluster_col, col_j]].dropna(),
                        on=cluster_col,
                        how='inner',
                    )
                    vals_i = merged[col_i].values
                    vals_j = merged[col_j].values
                else:
                    vals_i = df_i[col_i].dropna().values
                    vals_j = df_j[col_j].dropna().values

                min_len = min(len(vals_i), len(vals_j))
                if min_len < 2:
                    pval_matrix[i, j] = np.nan
                    continue

                vals_i = vals_i[:min_len]
                vals_j = vals_j[:min_len]

                try:
                    if pvalue_test == 'wilcoxon':
                        diff = vals_i - vals_j
                        if np.all(diff == 0):
                            pval_matrix[i, j] = 1.0
                        else:
                            _, p = wilcoxon(vals_i, vals_j)
                            pval_matrix[i, j] = p
                    elif pvalue_test == 'mannwhitneyu':
                        _, p = mannwhitneyu(vals_i, vals_j, alternative='two-sided')
                        pval_matrix[i, j] = p
                    else:
                        pval_matrix[i, j] = np.nan
                except Exception:
                    pval_matrix[i, j] = np.nan

        pval_df_out = pd.DataFrame(pval_matrix, index=col_names, columns=col_names)
        return pval_df_out

    return None

def plot_grouped_barplots_with_jitter(
    groups,
    colors=None,
    legend=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    bar_width=0.6,
    group_spacing=1.0,
    bar_spacing=0.05,
    bar_outline_color='black',
    bar_outline_width=1.0,
    bar_patch_alpha=0.9,
    bar_agg='mean',
    show_jitter=True,
    jitter_amount=0.15,
    jitter_size=15,
    jitter_color=None,
    jitter_alpha=0.6,
    jitter_edge_color='none',
    jitter_edge_width=0,
    jitter_seed=None,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    reference_hlines=None,
    legend_mode='integrated',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    save_path=None,
    compute_pvalues=False,
    pvalue_test='wilcoxon',
    cluster_col=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    rng = np.random.default_rng(jitter_seed)

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (10, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    if reference_hlines is None:
        reference_hlines = {}

    hline_col_set = set(reference_hlines.keys())

    def _normalize_group(group_value):
        """
        Normalize group value to a list of (df, col) pairs.
        """
        if isinstance(group_value, list):
            tuples = group_value
        else:
            tuples = [group_value]
        pairs = []
        for t in tuples:
            df_g = t[0]
            cols = t[1:]
            for c in cols:
                pairs.append((df_g, c))
        return pairs

    all_cols = set()
    for tick_label, group_value in groups.items():
        for df_g, col in _normalize_group(group_value):
            all_cols.add(col)

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = {}
        for i, col in enumerate(sorted(all_cols)):
            colors[col] = default_colors[i % len(default_colors)]

    hline_entries = []
    hline_vals_accum = {}
    for tick_label, group_value in groups.items():
        for df_g, col in _normalize_group(group_value):
            if col in hline_col_set and col in df_g.columns:
                if col not in hline_vals_accum:
                    hline_vals_accum[col] = []
                hline_vals_accum[col].extend(df_g[col].dropna().values.tolist())

    for col_name, vals in hline_vals_accum.items():
        if vals:
            style = reference_hlines[col_name]
            agg = style.get('agg', 'mean')
            if agg == 'median':
                y_val = np.median(vals)
            else:
                y_val = np.mean(vals)
            hline_entries.append((y_val, style, col_name))

    filtered_groups = {}
    for tick_label, group_value in groups.items():
        pairs = [(df_g, col) for df_g, col in _normalize_group(group_value)
                 if col not in hline_col_set]
        if pairs:
            filtered_groups[tick_label] = pairs

    max_cols = max(len(pairs) for pairs in filtered_groups.values()) if filtered_groups else 1
    n_groups = len(filtered_groups)

    fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])

    group_width = max_cols * bar_width + (max_cols - 1) * bar_spacing
    group_centers = np.arange(n_groups) * (group_width + group_spacing)

    tick_labels = []

    for gi, (tick_label, pairs) in enumerate(filtered_groups.items()):
        tick_labels.append(tick_label)
        n_cols = len(pairs)

        total_bar_width = n_cols * bar_width + (n_cols - 1) * bar_spacing
        start = group_centers[gi] - total_bar_width / 2 + bar_width / 2

        for ci, (df_g, col) in enumerate(pairs):
            if col not in df_g.columns:
                continue
            vals = df_g[col].dropna().values
            if len(vals) == 0:
                continue

            x_pos = start + ci * (bar_width + bar_spacing)
            color = colors.get(col, default_colors[0])

            bar_val = np.median(vals) if bar_agg == 'median' else np.mean(vals)

            ax.bar(
                x_pos,
                bar_val,
                width=bar_width, # bar_width * 0.9,
                color=color,
                edgecolor=bar_outline_color,
                linewidth=bar_outline_width,
                alpha=bar_patch_alpha,
                zorder=2,
            )

            if show_jitter:
                jitter_c = jitter_color if jitter_color is not None else 'black'
                x_jit = x_pos + rng.uniform(
                    -jitter_amount * bar_width,
                    jitter_amount * bar_width,
                    size=len(vals),
                )
                ax.scatter(
                    x_jit,
                    vals,
                    s=jitter_size,
                    facecolor=jitter_c,
                    edgecolor=jitter_edge_color,
                    linewidth=jitter_edge_width,
                    alpha=jitter_alpha,
                    zorder=3,
                )

    for y_val, style, col_name in hline_entries:
        ax.axhline(
            y=y_val,
            color=style.get('color', 'gray'),
            linestyle=style.get('linestyle', '--'),
            linewidth=style.get('linewidth', 1.5),
            alpha=style.get('alpha', 1.0),
            zorder=4,
        )

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None
        ax.axhline(y=hline_y, color=hline_color, linestyle=hline_style,
                   linewidth=hline_width, label=line_label, zorder=4)

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            if n_groups > 1:
                x_range = group_centers[-1] - group_centers[0]
            else:
                x_range = 1
            text_x = group_centers[0] + (x_range * hline_text_x_offset)
            if hline_text_position == 'above':
                text_y = hline_y + hline_text_offset
                va = 'bottom'
            else:
                text_y = hline_y - hline_text_offset
                va = 'top'
            ax.text(text_x, text_y, hline_text,
                    color=text_color, fontsize=hline_text_fontsize,
                    fontfamily=default_font, va=va, ha='left', zorder=4)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel(params['xlabel'], fontsize=params['xlabel_fontsize'],
                  fontfamily=default_font, labelpad=params['xlabel_pad'])
    ax.set_ylabel(params['ylabel'], fontsize=params['ylabel_fontsize'],
                  fontfamily=default_font, labelpad=params['ylabel_pad'])

    title_weight = 'bold' if params['title_bold'] else 'normal'
    ax.set_title(params['title'], fontsize=params['title_fontsize'],
                 fontfamily=default_font, fontweight=title_weight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(axis='both', which='major',
                   length=params['tick_length'],
                   width=params['axis_linewidth'],
                   direction='out',
                   pad=params['tick_length'] + 2)
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5,
                axis='y', zorder=0)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []

    if legend is not None:
        for lbl_text, color in legend.items():
            handles.append(mpatches.Patch(facecolor=color, edgecolor=bar_outline_color,
                                          linewidth=bar_outline_width, alpha=bar_patch_alpha))
            labels_list.append(lbl_text)

    for y_val, style, col_name in hline_entries:
        handles.append(Line2D([0], [0],
                              color=style.get('color', 'gray'),
                              linestyle=style.get('linestyle', '--'),
                              linewidth=style.get('linewidth', 1.5) * params['legend_handle_size'],
                              alpha=style.get('alpha', 1.0)))
        labels_list.append(style.get('label', col_name))

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(Line2D([0], [0], color=hline_color, linestyle=hline_style,
                              linewidth=hline_width * params['legend_handle_size']))
        labels_list.append(hline_label)

    if legend_mode == 'integrated' and handles:
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        leg = fig.legend(handles, labels_list,
                         loc=legend_loc,
                         bbox_to_anchor=bbox,
                         ncol=legend_ncol,
                         frameon=False,
                         fontsize=params['legend_fontsize'])
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(f"{save_path}{suffix}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}{suffix}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

    if legend_mode == 'separate' and handles:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(params['legend_handle_size'] * 2,
                     n_items * 0.4 * params['legend_handle_size']),
            dpi=params['dpi']
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')

        leg = ax_legend.legend(handles, labels_list,
                               loc='center', frameon=False,
                               fontsize=params['legend_fontsize'],
                               ncol=legend_ncol)
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(f"{save_path}_legend.pdf", format='pdf',
                               bbox_inches='tight', dpi=300)
            fig_legend.savefig(f"{save_path}_legend.png", format='png',
                               bbox_inches='tight', dpi=300)

        plt.show()

    if compute_pvalues:
        all_pairs = []
        for tick_label, group_value in groups.items():
            all_pairs.extend(_normalize_group(group_value))

        seen = set()
        unique_pairs = []
        for df_g, col in all_pairs:
            if col not in seen:
                seen.add(col)
                unique_pairs.append((df_g, col))

        col_names = [col for _, col in unique_pairs]
        n = len(col_names)
        pval_matrix = np.full((n, n), np.nan)

        for i in range(n):
            df_i, col_i = unique_pairs[i]
            for j in range(n):
                if i == j:
                    pval_matrix[i, j] = 1.0
                    continue
                if not np.isnan(pval_matrix[j, i]):
                    pval_matrix[i, j] = pval_matrix[j, i]
                    continue

                df_j, col_j = unique_pairs[j]

                if cluster_col is not None and cluster_col in df_i.columns and cluster_col in df_j.columns:
                    merged = pd.merge(
                        df_i[[cluster_col, col_i]].dropna(),
                        df_j[[cluster_col, col_j]].dropna(),
                        on=cluster_col, how='inner'
                    )
                    vals_i = merged[col_i].values
                    vals_j = merged[col_j].values
                else:
                    vals_i = df_i[col_i].dropna().values
                    vals_j = df_j[col_j].dropna().values

                min_len = min(len(vals_i), len(vals_j))
                if min_len < 2:
                    pval_matrix[i, j] = np.nan
                    continue

                vals_i = vals_i[:min_len]
                vals_j = vals_j[:min_len]

                try:
                    if pvalue_test == 'wilcoxon':
                        diff = vals_i - vals_j
                        if np.all(diff == 0):
                            pval_matrix[i, j] = 1.0
                        else:
                            _, p = wilcoxon(vals_i, vals_j)
                            pval_matrix[i, j] = p
                    elif pvalue_test == 'mannwhitneyu':
                        _, p = mannwhitneyu(vals_i, vals_j, alternative='two-sided')
                        pval_matrix[i, j] = p
                    else:
                        pval_matrix[i, j] = np.nan
                except Exception:
                    pval_matrix[i, j] = np.nan

        pval_df = pd.DataFrame(pval_matrix, index=col_names, columns=col_names)
        return pval_df

    return None

def plot_scatter_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    ax: plt.Axes = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    scatter_alpha: float = 0.85,
    scatter_s: float = 28.0,
    scatter_color: str = '#1f77b4',
    scatter_marker: str = 'o',
    scatter_edge_color: str = 'none',
    scatter_edge_width: float = 0,
    hue_col: str = None,
    hue_colors: dict = None,
    hue_order: list = None,
    show_regression: bool = True,
    regression_color: str = 'black',
    regression_width: float = 1.5,
    regression_style: str = '-',
    regression_alpha: float = 1.0,
    show_ci: bool = True,
    ci_level: float = 0.95,
    ci_color: str = None,
    ci_alpha: float = 0.2,
    show_spearman: bool = True,
    spearman_fontsize: float = 10,
    spearman_loc: str = 'upper right',
    spearman_bbox_alpha: float = 0.8,
    spearman_bbox_color: str = 'white',
    figsize: tuple = (6.5, 4.5),
    dpi: int = 150,
    default_font: str = 'Nimbus Sans',
    remove_top_right_spines: bool = True,
    spine_width: float = None,
    tick_length: float = None,
    tick_width: float = None,
    tick_direction: str = 'out',
    title_fontsize: float = 14,
    title_bold: bool = False,
    xlabel_fontsize: float = 12,
    ylabel_fontsize: float = 12,
    xtick_labelsize: float = 10,
    ytick_labelsize: float = 10,
    xlabel_pad: float = 4.0,
    ylabel_pad: float = 4.0,
    legend_fontsize: float = 10,
    legend_loc: str = 'best',
    legend_frameon: bool = False,
    xlim: tuple = None,
    ylim: tuple = None,
    grid_alpha: float = 0.0,
    save_path: str = None,
    show: bool = True,
    output_stats_df: bool = False,
) -> plt.Axes | tuple[plt.Axes, pd.DataFrame]:
    """
    Scatter + OLS regression line + CI band with Spearman annotation.
    """
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"df must contain columns '{x_col}' and '{y_col}'")

    cols_needed = [x_col, y_col]
    if hue_col is not None:
        cols_needed.append(hue_col)
    d = df[cols_needed].dropna(subset=[x_col, y_col])

    if len(d) < 3:
        raise ValueError(f"Need at least 3 non-missing rows, got {len(d)}")

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    if Sxx <= 0:
        raise ValueError("x has zero variance; cannot fit regression.")

    b = np.sum((x - x_mean) * (y - y_mean)) / Sxx
    a = y_mean - b * x_mean

    y_hat = a + b * x
    resid = y - y_hat
    n = x.size
    dof = n - 2
    if dof <= 0:
        raise ValueError("Not enough degrees of freedom for CI band.")

    s2 = np.sum(resid ** 2) / dof
    s = np.sqrt(s2)

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = a + b * x_grid

    se_mean = s * np.sqrt((1.0 / n) + ((x_grid - x_mean) ** 2) / Sxx)
    alpha_ci = 1.0 - ci_level
    tcrit = stats.t.ppf(1.0 - alpha_ci / 2.0, dof)
    lo = y_grid - tcrit * se_mean
    hi = y_grid + tcrit * se_mean

    rho_s, p_s = stats.spearmanr(x, y)
    rho_p, p_p = stats.pearsonr(x, y)

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    ss_reg = ss_tot - ss_res
    if dof > 0 and ss_res > 0:
        f_stat = (ss_reg / 1) / (ss_res / dof)
        p_r2 = 1.0 - stats.f.cdf(f_stat, 1, dof)
    else:
        p_r2 = np.nan

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if hue_col is not None:
        hue_vals = d[hue_col]
        if hue_order is None:
            hue_order = sorted(hue_vals.unique())

        default_palette = [
            '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ]
        if hue_colors is None:
            hue_colors = {v: default_palette[i % len(default_palette)]
                          for i, v in enumerate(hue_order)}

        for hv in hue_order:
            mask = hue_vals == hv
            ax.scatter(
                x[mask.values], y[mask.values],
                s=scatter_s, alpha=scatter_alpha,
                color=hue_colors.get(hv, scatter_color),
                marker=scatter_marker,
                edgecolors=scatter_edge_color,
                linewidths=scatter_edge_width,
                label=str(hv),
                zorder=2,
            )
    else:
        ax.scatter(
            x, y,
            s=scatter_s, alpha=scatter_alpha,
            color=scatter_color,
            marker=scatter_marker,
            edgecolors=scatter_edge_color,
            linewidths=scatter_edge_width,
            zorder=2,
        )

    if show_regression:
        ax.plot(x_grid, y_grid,
                color=regression_color,
                linewidth=regression_width,
                linestyle=regression_style,
                alpha=regression_alpha,
                zorder=3)

    if show_ci:
        fill_color = ci_color if ci_color is not None else regression_color
        ax.fill_between(x_grid, lo, hi,
                        color=fill_color, alpha=ci_alpha, zorder=1)

    ax.set_xlabel(xlabel if xlabel is not None else x_col,
                  fontsize=xlabel_fontsize, fontfamily=default_font,
                  labelpad=xlabel_pad)
    ax.set_ylabel(ylabel if ylabel is not None else y_col,
                  fontsize=ylabel_fontsize, fontfamily=default_font,
                  labelpad=ylabel_pad)

    if title is not None:
        title_weight = 'bold' if title_bold else 'normal'
        ax.set_title(title, fontsize=title_fontsize,
                     fontfamily=default_font, fontweight=title_weight)

    if remove_top_right_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if spine_width is not None:
        for side in ['left', 'bottom', 'top', 'right']:
            if ax.spines[side].get_visible():
                ax.spines[side].set_linewidth(spine_width)

    tick_kw = {'direction': tick_direction}
    if tick_length is not None:
        tick_kw['length'] = tick_length
    if tick_width is not None:
        tick_kw['width'] = tick_width
    ax.tick_params(axis='both', which='major', **tick_kw)
    ax.tick_params(axis='x', labelsize=xtick_labelsize)
    ax.tick_params(axis='y', labelsize=ytick_labelsize)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.margins(x=0.02, y=0.05)

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, zorder=0)

    if show_spearman:
        def _fmt_pval(pv):
            if np.isnan(pv):
                return "p=NA"
            if pv >= 0.01:
                return f"p={pv:.3g}"
            # Find the order of magnitude floor
            exp = int(np.floor(np.log10(pv)))
            return f"p<1e{exp}"

        lines = []
        lines.append(f"R²={r2:.2f}, {_fmt_pval(p_r2)}")
        lines.append(f"Pearson r={rho_p:.2f}, {_fmt_pval(p_p)}")
        lines.append(f"Spearman ρ={rho_s:.2f}, {_fmt_pval(p_s)}")
        lines.append(f"n={n}")
        ann = "\n".join(lines)

        loc_map = {
            "upper right": (0.98, 0.98, "right", "top"),
            "upper left": (0.02, 0.98, "left", "top"),
            "lower right": (0.98, 0.02, "right", "bottom"),
            "lower left": (0.02, 0.02, "left", "bottom"),
        }
        if spearman_loc not in loc_map:
            raise ValueError(f"spearman_loc must be one of {list(loc_map.keys())}")
        x0, y0, ha, va = loc_map[spearman_loc]
        ax.text(
            x0, y0, ann,
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=spearman_fontsize,
            fontfamily=default_font,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=spearman_bbox_color,
                      alpha=spearman_bbox_alpha,
                      edgecolor="none"),
        )

    if hue_col is not None:
        leg = ax.legend(fontsize=legend_fontsize, loc=legend_loc,
                        frameon=legend_frameon)
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig = ax.get_figure()
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    if output_stats_df:
        def _compute_stats(x_arr, y_arr, group_label='global'):
            """Compute all regression stats for a given x/y pair."""
            nn = len(x_arr)
            if nn < 3:
                return {
                    'group': group_label, 'n': nn,
                    'slope': np.nan, 'intercept': np.nan,
                    'r2': np.nan, 'r2_pval': np.nan,
                    'pearson_r': np.nan, 'pearson_pval': np.nan,
                    'spearman_rho': np.nan, 'spearman_pval': np.nan,
                }
            xm = x_arr.mean()
            ym = y_arr.mean()
            sxx = np.sum((x_arr - xm) ** 2)
            if sxx <= 0:
                return {
                    'group': group_label, 'n': nn,
                    'slope': np.nan, 'intercept': np.nan,
                    'r2': np.nan, 'r2_pval': np.nan,
                    'pearson_r': np.nan, 'pearson_pval': np.nan,
                    'spearman_rho': np.nan, 'spearman_pval': np.nan,
                }
            bb = np.sum((x_arr - xm) * (y_arr - ym)) / sxx
            aa = ym - bb * xm
            yh = aa + bb * x_arr
            res = y_arr - yh
            dd = nn - 2

            ss_res = np.sum(res ** 2)
            ss_tot = np.sum((y_arr - ym) ** 2)
            r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            if dd > 0 and ss_res > 0:
                ss_reg = ss_tot - ss_res
                f_val = (ss_reg / 1) / (ss_res / dd)
                p_r2_val = 1.0 - stats.f.cdf(f_val, 1, dd)
            else:
                p_r2_val = np.nan

            rp, pp = stats.pearsonr(x_arr, y_arr)
            rs, ps = stats.spearmanr(x_arr, y_arr)

            return {
                'group': group_label,
                'n': nn,
                'slope': bb,
                'intercept': aa,
                'r2': r2_val,
                'r2_pval': p_r2_val,
                'pearson_r': rp,
                'pearson_pval': pp,
                'spearman_rho': rs,
                'spearman_pval': ps,
            }

        rows = []
        rows.append(_compute_stats(x, y, group_label='global'))

        if hue_col is not None:
            hue_vals = d[hue_col]
            hue_groups = hue_order if hue_order is not None else sorted(hue_vals.unique())
            for hv in hue_groups:
                mask = (hue_vals == hv).values
                if mask.sum() < 3:
                    rows.append({
                        'group': hv, 'n': int(mask.sum()),
                        'slope': np.nan, 'intercept': np.nan,
                        'r2': np.nan, 'r2_pval': np.nan,
                        'pearson_r': np.nan, 'pearson_pval': np.nan,
                        'spearman_rho': np.nan, 'spearman_pval': np.nan,
                    })
                else:
                    rows.append(_compute_stats(x[mask], y[mask], group_label=hv))

        stats_df = pd.DataFrame(rows)
        return ax, stats_df

    return ax

def plot_scatter_simple(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    ax: plt.Axes = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    point_size: float = 28.0,
    point_alpha: float = 0.85,
    point_marker: str = 'o',
    point_edge_color: str = 'none',
    point_edge_width: float = 0.0,
    hue_col: str = 'allele',
    hue_colors: dict = None,
    hue_order: list = None,
    default_point_color: str = '#1f77b4',
    show_xy_line: bool = True,
    xy_line_color: str = 'darkgray',
    xy_line_style: str = '--',
    xy_line_width: float = 0.5,
    xy_line_alpha: float = 1.0,
    xy_line_zorder: int = 1,
    xy_line_pad_frac: float = 0.02,
    xy_line_in_legend: bool = False,
    xy_line_label: str = 'x = y',
    figsize: tuple = (6.5, 4.5),
    dpi: int = 150,
    default_font: str = 'Nimbus Sans',
    remove_top_right_spines: bool = True,
    spine_width: float = 0.8,
    tick_length: float = 4.0,
    tick_width: float = None,
    tick_direction: str = 'out',
    title_fontsize: float = 14,
    title_bold: bool = False,
    xlabel_fontsize: float = 12,
    ylabel_fontsize: float = 12,
    xtick_labelsize: float = 10,
    ytick_labelsize: float = 10,
    xlabel_pad: float = 4.0,
    ylabel_pad: float = 4.0,
    show_legend: bool = True,
    legend_fontsize: float = 10,
    legend_loc: str = 'best',
    legend_frameon: bool = False,
    legend_markerscale: float = 1.5,
    xlim: tuple = None,
    ylim: tuple = None,
    equal_aspect: bool = False,
    grid_alpha: float = 0.0,
    save_path: str = None,
    show: bool = True,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"df must contain columns '{x_col}' and '{y_col}'")

    cols_needed = [x_col, y_col]
    if hue_col is not None and hue_col in df.columns:
        cols_needed.append(hue_col)
    d = df[cols_needed].dropna(subset=[x_col, y_col])

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    default_palette = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    use_hue = hue_col is not None and hue_col in d.columns

    if use_hue:
        hue_vals = d[hue_col]
        if hue_order is None:
            hue_order = sorted(hue_vals.unique())

        if hue_colors is None:
            hue_colors = {
                v: default_palette[i % len(default_palette)]
                for i, v in enumerate(hue_order)
            }

        for hv in hue_order:
            mask = (hue_vals == hv).values
            if not mask.any():
                continue
            ax.scatter(
                x[mask], y[mask],
                s=point_size,
                alpha=point_alpha,
                color=hue_colors.get(hv, default_point_color),
                marker=point_marker,
                edgecolors=point_edge_color,
                linewidths=point_edge_width,
                label=str(hv),
                zorder=2,
            )
    else:
        ax.scatter(
            x, y,
            s=point_size,
            alpha=point_alpha,
            color=default_point_color,
            marker=point_marker,
            edgecolors=point_edge_color,
            linewidths=point_edge_width,
            zorder=2,
        )

    if show_xy_line:
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        rng = hi - lo
        pad = rng * xy_line_pad_frac if rng > 0 else 0.05
        line_lo, line_hi = lo - pad, hi + pad

        ax.plot(
            [line_lo, line_hi], [line_lo, line_hi],
            color=xy_line_color,
            linestyle=xy_line_style,
            linewidth=xy_line_width,
            alpha=xy_line_alpha,
            label=xy_line_label if xy_line_in_legend else None,
            zorder=xy_line_zorder,
        )

    ax.set_xlabel(
        xlabel if xlabel is not None else x_col,
        fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad,
    )
    ax.set_ylabel(
        ylabel if ylabel is not None else y_col,
        fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad,
    )

    if title is not None:
        title_weight = 'bold' if title_bold else 'normal'
        ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font,
                     fontweight=title_weight)

    if remove_top_right_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if spine_width is not None:
        for side in ['left', 'bottom', 'top', 'right']:
            if ax.spines[side].get_visible():
                ax.spines[side].set_linewidth(spine_width)

    tick_kw = {'direction': tick_direction, 'length': tick_length}
    if tick_width is not None:
        tick_kw['width'] = tick_width
    elif spine_width is not None:
        tick_kw['width'] = spine_width
    ax.tick_params(axis='both', which='major', **tick_kw)
    ax.tick_params(axis='x', labelsize=xtick_labelsize)
    ax.tick_params(axis='y', labelsize=ytick_labelsize)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, zorder=0)

    if show_legend and (use_hue or (show_xy_line and xy_line_in_legend)):
        leg = ax.legend(
            fontsize=legend_fontsize,
            loc=legend_loc,
            frameon=legend_frameon,
            markerscale=legend_markerscale,
        )
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return ax

def plot_dual_scatter(df1, df2, x_col, y_col, *, title1=None, title2=None, suptitle=None, wspace=0.25,
                      point_size=28.0, point_alpha=0.85, point_marker='o', point_edge_color='none', point_edge_width=0.0,
                      hue_col='allele', hue_colors=None, hue_order=None, default_point_color='#1f77b4',
                      show_xy_line=True, xy_line_color='darkgray', xy_line_style='--', xy_line_width=0.5, xy_line_alpha=1.0, xy_line_zorder=1, xy_line_pad_frac=0.02,
                      figsize=(12, 5), dpi=150, default_font='Nimbus Sans', remove_top_right_spines=True, spine_width=0.8,
                      tick_length=4.0, tick_width=None, tick_direction='out',
                      title_fontsize=14, title_bold=False, suptitle_fontsize=15, xlabel=None, ylabel=None, xlabel_fontsize=12, ylabel_fontsize=12,
                      xtick_labelsize=10, ytick_labelsize=10, xlabel_pad=4.0, ylabel_pad=4.0,
                      show_legend=True, legend_fontsize=10, legend_ncol=4, legend_frameon=False, legend_markerscale=1.5, legend_y=-0.02,
                      xlim=None, ylim=None, equal_aspect=False, grid_alpha=0.0, save_path=None, show=True):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    for d in (df1, df2):
        if x_col not in d.columns or y_col not in d.columns:
            raise KeyError(f"both dfs must contain columns '{x_col}' and '{y_col}'")

    default_palette = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    use_hue = hue_col is not None and hue_col in df1.columns

    if use_hue and hue_order is None:
        vals = pd.unique(pd.concat([df1[hue_col], df2[hue_col]], ignore_index=True))
        hue_order = sorted(vals)
    if use_hue and hue_colors is None:
        hue_colors = {v: default_palette[i % len(default_palette)] for i, v in enumerate(hue_order)}

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    panels = [(axes[0], df1, title1), (axes[1], df2, title2)]

    for ax, df, title in panels:
        cols_needed = [x_col, y_col] + ([hue_col] if use_hue else [])
        d = df[cols_needed].dropna(subset=[x_col, y_col])
        x = d[x_col].to_numpy(float); y = d[y_col].to_numpy(float)

        if use_hue:
            hue_vals = d[hue_col]
            for hv in hue_order:
                mask = (hue_vals == hv).values
                if not mask.any():
                    continue
                ax.scatter(x[mask], y[mask], s=point_size, alpha=point_alpha, color=hue_colors.get(hv, default_point_color), marker=point_marker, edgecolors=point_edge_color, linewidths=point_edge_width, zorder=2)
        else:
            ax.scatter(x, y, s=point_size, alpha=point_alpha, color=default_point_color, marker=point_marker, edgecolors=point_edge_color, linewidths=point_edge_width, zorder=2)

        # set limits BEFORE drawing the x=y line so the line can span the full axes
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')

        if show_xy_line:
            cur_xlim = ax.get_xlim(); cur_ylim = ax.get_ylim()
            if xlim is None and ylim is None:
                lo = min(x.min(), y.min()); hi = max(x.max(), y.max()); rng = hi - lo
                pad = rng * xy_line_pad_frac if rng > 0 else 0.05
                line_lo, line_hi = lo - pad, hi + pad
            else:
                line_lo = min(cur_xlim[0], cur_ylim[0])
                line_hi = max(cur_xlim[1], cur_ylim[1])
            ax.plot([line_lo, line_hi], [line_lo, line_hi], color=xy_line_color, linestyle=xy_line_style, linewidth=xy_line_width, alpha=xy_line_alpha, zorder=xy_line_zorder, clip_on=True)
            # restore limits (plotting the line can nudge autoscale when no explicit lims)
            ax.set_xlim(cur_xlim); ax.set_ylim(cur_ylim)

        ax.set_xlabel(xlabel if xlabel is not None else x_col, fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad)
        ax.set_ylabel(ylabel if ylabel is not None else y_col, fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad)
        if title is not None:
            ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font, fontweight='bold' if title_bold else 'normal')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if spine_width is not None:
            for side in ['left','bottom','top','right']:
                if ax.spines[side].get_visible():
                    ax.spines[side].set_linewidth(spine_width)
        tick_kw = {'direction': tick_direction, 'length': tick_length}
        if tick_width is not None:
            tick_kw['width'] = tick_width
        elif spine_width is not None:
            tick_kw['width'] = spine_width
        ax.tick_params(axis='both', which='major', **tick_kw)
        ax.tick_params(axis='x', labelsize=xtick_labelsize); ax.tick_params(axis='y', labelsize=ytick_labelsize)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(default_font)
        if grid_alpha > 0:
            ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, zorder=0)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, fontfamily=default_font)
    fig.subplots_adjust(wspace=wspace)

    if show_legend and use_hue:
        handles = [Line2D([0],[0], marker=point_marker, linestyle='none', markersize=np.sqrt(point_size)*legend_markerscale, markerfacecolor=hue_colors.get(hv, default_point_color), markeredgecolor='none', label=str(hv)) for hv in hue_order]
        leg = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, legend_y), ncol=legend_ncol, fontsize=legend_fontsize, frameon=legend_frameon)
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    return fig, axes

def plot_dual_ladder(df1, df2, value_cols, *, df1_label="df1", df2_label="df2", titles=None, suptitle=None, wspace=0.3,
                     point_size=28.0, point_alpha=0.85, point_marker='o', point_edge_color='none', point_edge_width=0.0,
                     connector_width=0.8, connector_alpha=0.5, connector_zorder=1,
                     hue_col='allele', hue_colors=None, hue_order=None, default_point_color='#1f77b4',
                     figsize=(12, 5), dpi=150, default_font='Nimbus Sans', remove_top_right_spines=True, spine_width=0.8,
                     tick_length=4.0, tick_width=None, tick_direction='out',
                     title_fontsize=14, title_bold=False, suptitle_fontsize=15, ylabel=None, ylabel_fontsize=12,
                     xtick_labelsize=10, ytick_labelsize=10, xlabel_pad=4.0, ylabel_pad=4.0, x_jitter=0.0,
                     show_legend=True, legend_fontsize=10, legend_ncol=4, legend_frameon=False, legend_markerscale=1.5, legend_y=-0.02,
                     ylim=None, grid_alpha=0.0, save_path=None, show=True, seed=0):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if isinstance(value_cols, str):
        value_cols = [value_cols]
    for c in value_cols:
        if c not in df1.columns or c not in df2.columns:
            raise KeyError(f"both dfs must contain value column '{c}'")

    default_palette = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    use_hue = hue_col is not None and hue_col in df1.columns

    if use_hue and hue_order is None:
        vals = pd.unique(pd.concat([df1[hue_col], df2[hue_col]], ignore_index=True))
        hue_order = sorted(vals)
    if use_hue and hue_colors is None:
        hue_colors = {v: default_palette[i % len(default_palette)] for i, v in enumerate(hue_order)}

    rng = np.random.default_rng(seed)
    n_panels = len(value_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes[0]
    if titles is None:
        titles = value_cols

    for pi, (ax, col) in enumerate(zip(axes, value_cols)):
        v1 = df1[col].to_numpy(float); v2 = df2[col].to_numpy(float)
        hues = df1[hue_col].to_numpy() if use_hue else np.array([None] * len(v1))
        for i in range(len(v1)):
            if np.isnan(v1[i]) or np.isnan(v2[i]):
                continue
            c = hue_colors.get(hues[i], default_point_color) if use_hue else default_point_color
            jx = rng.uniform(-x_jitter, x_jitter) if x_jitter > 0 else 0.0
            xa, xb = 0 + jx, 1 + jx
            ax.plot([xa, xb], [v1[i], v2[i]], color=c, linewidth=connector_width, alpha=connector_alpha, zorder=connector_zorder)
            ax.scatter([xa, xb], [v1[i], v2[i]], s=point_size, alpha=point_alpha, color=c, marker=point_marker, edgecolors=point_edge_color, linewidths=point_edge_width, zorder=2)

        ax.set_xticks([0, 1]); ax.set_xticklabels([df1_label, df2_label], fontsize=xtick_labelsize, fontfamily=default_font)
        ax.set_xlim(-0.5, 1.5)

        if isinstance(ylabel, (list, tuple)):
            yl = ylabel[pi] if pi < len(ylabel) else None
        else:
            yl = ylabel
        ax.set_ylabel(yl if yl is not None else "value", fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad)

        ax.set_title(titles[pi], fontsize=title_fontsize, fontfamily=default_font, fontweight='bold' if title_bold else 'normal')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if spine_width is not None:
            for side in ['left','bottom','top','right']:
                if ax.spines[side].get_visible():
                    ax.spines[side].set_linewidth(spine_width)
        tick_kw = {'direction': tick_direction, 'length': tick_length}
        if tick_width is not None:
            tick_kw['width'] = tick_width
        elif spine_width is not None:
            tick_kw['width'] = spine_width
        ax.tick_params(axis='both', which='major', **tick_kw)
        ax.tick_params(axis='y', labelsize=ytick_labelsize)
        for lbl in ax.get_yticklabels():
            lbl.set_fontfamily(default_font)
        if ylim is not None:
            ax.set_ylim(ylim)
        if grid_alpha > 0:
            ax.grid(True, axis='y', alpha=grid_alpha, linestyle='-', linewidth=0.5, zorder=0)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, fontfamily=default_font)
    fig.subplots_adjust(wspace=wspace)

    if show_legend and use_hue:
        handles = [Line2D([0],[0], marker=point_marker, linestyle='none', markersize=np.sqrt(point_size)*legend_markerscale, markerfacecolor=hue_colors.get(hv, default_point_color), markeredgecolor='none', label=str(hv)) for hv in hue_order]
        leg = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, legend_y), ncol=legend_ncol, fontsize=legend_fontsize, frameon=legend_frameon)
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    return fig, axes

# seq2logo version
STANDARD_COLOR_SCHEME = {
    'A': '#A6A6A6',
    'C': '#66A61E',
    'D': '#D7191C',
    'E': '#D7191C',
    'F': '#404040',
    'G': '#C7C7C7',
    'H': '#2C7BB6',
    'I': '#A6A6A6',
    'K': '#2C7BB6',
    'L': '#A6A6A6',
    'M': '#A6A6A6',
    'N': '#4DAF4A',
    'P': '#984EA3',
    'Q': '#4DAF4A',
    'R': '#2C7BB6',
    'S': '#4DAF4A',
    'T': '#4DAF4A',
    'V': '#A6A6A6',
    'W': '#404040',
    'Y': '#404040',
}

def compute_pwm_bits(peptides):
    aa_list = list('ACDEFGHIKLMNPQRSTVWY')
    max_len = max(len(p) for p in peptides)
    counts = pd.DataFrame(0, index=range(max_len), columns=aa_list)
    for p in peptides:
        for i, aa in enumerate(p):
            if aa in aa_list:
                counts.loc[i, aa] += 1
    freq = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    n_aa = 20
    max_bits = np.log2(n_aa)
    entropy = -np.sum(freq * np.log2(freq.replace(0, 1)), axis=1)
    info = max_bits - entropy
    pwm = freq.mul(info, axis=0)
    return pwm


def plot_pwm_logo_single(
    df,
    peptide_col,
    *,
    figsize=(6, 3),
    dpi=150,
    default_font='Nimbus Sans',
    title=None,
    title_fontsize=9,
    title_bold=True,
    show_n_in_title=True,
    ylabel='Bits',
    xlabel='Position',
    xlabel_fontsize=8,
    ylabel_fontsize=8,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    xtick_labelsize=7,
    ytick_labelsize=7,
    color_scheme=None,
    vpad=0.05,
    logo_width=0.9,
    highlight_cysteine=False,
    cysteine_color='#ff7f00',
    non_cysteine_color='#999999',
    ylim=None,
    spine_width=0.8,
    remove_top_right_spines=True,
    tick_length=None,
    tick_width=None,
    tick_direction='out',
    square_aspect=False,
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if highlight_cysteine:
        aa_list = list('ACDEFGHIKLMNPQRSTVWY')
        cs = {aa: non_cysteine_color for aa in aa_list}
        cs['C'] = cysteine_color
    elif color_scheme is not None:
        cs = color_scheme
    else:
        cs = STANDARD_COLOR_SCHEME

    ylim_final = ylim if ylim is not None else (0, np.log2(20))

    peptides = df[peptide_col].dropna().tolist()
    n = len(peptides)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    pwm = compute_pwm_bits(peptides)

    logo = logomaker.Logo(pwm, ax=ax, color_scheme=cs, vpad=vpad, width=logo_width)

    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True, linewidth=spine_width)
    if not remove_top_right_spines:
        logo.style_spines(spines=['top', 'right'], visible=True, linewidth=spine_width)

    ax.set_ylim(ylim_final)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad)

    ax.set_xticks(range(len(pwm)))
    ax.set_xticklabels(range(1, len(pwm) + 1))

    tick_kw = {'direction': tick_direction}
    if tick_length is not None:
        tick_kw['length'] = tick_length
    if tick_width is not None:
        tick_kw['width'] = tick_width
    ax.tick_params(axis='both', which='major', **tick_kw)
    ax.tick_params(axis='x', labelsize=xtick_labelsize)
    ax.tick_params(axis='y', labelsize=ytick_labelsize)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if title is not None:
        title_weight = 'bold' if title_bold else 'normal'
        title_text = f'{title} (n={n})' if show_n_in_title else title
        ax.set_title(title_text, fontsize=title_fontsize,
                     fontfamily=default_font, fontweight=title_weight)

    if square_aspect:
        ax.set_box_aspect(1)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_pwm_logo_grid_v2_multi_df(
    df_tuples,
    alleles,
    allele_col,
    top_n=2000,
    *,
    figsize_per_cell=(6, 3),
    dpi=150,
    default_font='Nimbus Sans',
    title=None,
    title_fontsize=9,
    title_bold=True,
    show_n_in_title=True,
    ylabel='Bits',
    xlabel='Position',
    xlabel_fontsize=8,
    ylabel_fontsize=8,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    xtick_labelsize=7,
    ytick_labelsize=7,
    color_scheme=None,
    vpad=0.05,
    logo_width=0.9,
    highlight_cysteine=False,
    cysteine_color='#ff7f00',
    non_cysteine_color='#999999',
    ylim=None,
    spine_width=0.8,
    remove_top_right_spines=True,
    tick_length=None,
    tick_width=None,
    tick_direction='out',
    square_aspect=False,
    save_path=None,
):
    """
    Plot a grid of PWM sequence logos from multiple dataframes.
    """
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    n_rows = len(df_tuples)
    n_cols = len(alleles)

    if highlight_cysteine:
        aa_list = list('ACDEFGHIKLMNPQRSTVWY')
        cs = {aa: non_cysteine_color for aa in aa_list}
        cs['C'] = cysteine_color
    elif color_scheme is not None:
        cs = color_scheme
    else:
        cs = STANDARD_COLOR_SCHEME

    ylim_final = ylim if ylim is not None else (0, np.log2(20))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        dpi=dpi,
        squeeze=False,
    )

    for row_idx, (df, col_name) in enumerate(df_tuples):
        for col_idx, allele in enumerate(alleles):
            ax = axes[row_idx][col_idx]

            allele_df = df[df[allele_col] == allele]

            if col_name == 'peptide':
                peptides = allele_df['peptide'].dropna().tolist()
            else:
                peptides = (
                    allele_df
                    .nlargest(top_n, col_name)
                    ['peptide']
                    .dropna()
                    .tolist()
                )

            n = len(peptides)

            if n == 0:
                ax.axis('off')
                continue

            pwm = compute_pwm_bits(peptides)

            logo = logomaker.Logo(
                pwm,
                ax=ax,
                color_scheme=cs,
                vpad=vpad,
                width=logo_width,
            )

            logo.style_spines(visible=False)
            logo.style_spines(spines=['left', 'bottom'], visible=True, linewidth=spine_width)
            if not remove_top_right_spines:
                logo.style_spines(spines=['top', 'right'], visible=True, linewidth=spine_width)

            ax.set_ylim(ylim_final)

            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font,
                          labelpad=ylabel_pad)
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font,
                          labelpad=xlabel_pad)

            ax.set_xticks(range(len(pwm)))
            ax.set_xticklabels(range(1, len(pwm) + 1))

            tick_kw = {'direction': tick_direction}
            if tick_length is not None:
                tick_kw['length'] = tick_length
            if tick_width is not None:
                tick_kw['width'] = tick_width
            ax.tick_params(axis='both', which='major', **tick_kw)
            ax.tick_params(axis='x', labelsize=xtick_labelsize)
            ax.tick_params(axis='y', labelsize=ytick_labelsize)

            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontfamily(default_font)

            if title is not None:
                title_weight = 'bold' if title_bold else 'normal'
                title_text = f'{title} (n={n})' if show_n_in_title else title
                ax.set_title(title_text, fontsize=title_fontsize,
                             fontfamily=default_font, fontweight=title_weight)

            if square_aspect:
                ax.set_box_aspect(1)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import logomaker


def plot_pwm_logo_grid_seed_averaged(
    holec_train_df,
    background_df,
    ms_train_df,
    ft_score_cols,
    pepcl_score_cols,
    alleles,
    allele_col,
    peptide_col="peptide",
    top_n=2000,
    *,
    aa_alphabet="ACDEFGHIKLMNPQRSTVWY",
    require_matching_seeds=True,
    figsize_per_cell=(6, 3),
    dpi=150,
    default_font='Nimbus Sans',
    title=None,
    title_fontsize=9,
    title_bold=True,
    show_n_in_title=True,
    ylabel='Bits',
    xlabel='Position',
    xlabel_fontsize=8,
    ylabel_fontsize=8,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    xtick_labelsize=7,
    ytick_labelsize=7,
    color_scheme=None,
    vpad=0.05,
    logo_width=0.9,
    highlight_cysteine=False,
    cysteine_color='#ff7f00',
    non_cysteine_color='#999999',
    ylim=None,
    spine_width=0.8,
    remove_top_right_spines=True,
    tick_length=None,
    tick_width=None,
    tick_direction='out',
    square_aspect=False,
    save_path=None,
):
    """
    Plot a 4-row PWM logo grid:

        row 1: Holec training positives
        row 2: FT top-N motif averaged across seed replicates
        row 3: PepCL top-N motif averaged across seed replicates
        row 4: MS training positives

    FT/PepCL averaging:
        1. select top-N peptides independently for each seed/model
        2. compute position-specific amino-acid frequencies per seed
        3. average frequency matrices equally across seeds
        4. convert the mean frequency matrix to information-content bits

    Thus model scores themselves are never averaged/ensembled.
    """

    matplotlib.rcParams['font.family']='sans-serif'
    matplotlib.rcParams['font.sans-serif']=['Nimbus Sans','Nimbus Sans L','Arial','Helvetica']
    matplotlib.rcParams['pdf.fonttype']=42

    # ------------------------------------------------------------------
    # Validate score columns and, where possible, matched seed sets
    # ------------------------------------------------------------------
    ft_score_cols=list(ft_score_cols)
    pepcl_score_cols=list(pepcl_score_cols)

    missing_ft=[c for c in ft_score_cols if c not in background_df.columns]
    missing_pepcl=[c for c in pepcl_score_cols if c not in background_df.columns]
    if missing_ft:
        raise ValueError(f"Missing FT score columns: {missing_ft}")
    if missing_pepcl:
        raise ValueError(f"Missing PepCL score columns: {missing_pepcl}")

    def _seed_map(cols):
        out={}
        for col in cols:
            m=re.search(r"_seed_(\d+)(?:_|$)",str(col))
            if m is not None:
                seed=int(m.group(1))
                if seed in out:
                    raise ValueError(f"Duplicate seed {seed} found in score columns.")
                out[seed]=col
        return out

    ft_seed_map=_seed_map(ft_score_cols)
    pepcl_seed_map=_seed_map(pepcl_score_cols)

    if require_matching_seeds and ft_seed_map and pepcl_seed_map:
        if set(ft_seed_map)!=set(pepcl_seed_map):
            raise ValueError(
                f"FT/PepCL seed sets do not match.\n"
                f"FT only: {sorted(set(ft_seed_map)-set(pepcl_seed_map))}\n"
                f"PepCL only: {sorted(set(pepcl_seed_map)-set(ft_seed_map))}"
            )
        seeds=sorted(ft_seed_map)
        ft_score_cols=[ft_seed_map[s] for s in seeds]
        pepcl_score_cols=[pepcl_seed_map[s] for s in seeds]

    if not ft_score_cols or not pepcl_score_cols:
        raise ValueError("ft_score_cols and pepcl_score_cols must both contain at least one score column.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    aa_list=list(aa_alphabet)
    max_bits=np.log2(len(aa_list))

    def _frequency_matrix(peptides):
        peptides=[str(p) for p in peptides if pd.notna(p)]
        if not peptides:
            return None

        L=len(peptides[0])
        if any(len(p)!=L for p in peptides):
            raise ValueError("All peptides within a logo must have the same length.")

        counts=np.zeros((L,len(aa_list)),dtype=float)
        aa_to_idx={aa:i for i,aa in enumerate(aa_list)}

        for peptide in peptides:
            for pos,aa in enumerate(peptide):
                if aa in aa_to_idx:
                    counts[pos,aa_to_idx[aa]]+=1

        denom=counts.sum(axis=1,keepdims=True)
        with np.errstate(divide='ignore',invalid='ignore'):
            freq=np.divide(counts,denom,out=np.zeros_like(counts),where=denom>0)

        return pd.DataFrame(freq,index=range(L),columns=aa_list)

    def _frequency_to_bits(freq):
        p=freq.to_numpy(dtype=float)

        with np.errstate(divide='ignore',invalid='ignore'):
            logp=np.where(p>0,np.log2(p),0.0)
            entropy=-(p*logp).sum(axis=1)

        information=max_bits-entropy
        bits=p*information[:,None]

        return pd.DataFrame(bits,index=freq.index,columns=freq.columns)

    def _raw_logo(df,allele):
        allele_df=df[df[allele_col].astype(str)==str(allele)]
        peptides=allele_df[peptide_col].dropna().astype(str).tolist()
        if not peptides:
            return None,0

        freq=_frequency_matrix(peptides)
        return _frequency_to_bits(freq),len(peptides)

    def _seed_averaged_logo(df,allele,score_cols):
        allele_df=df[df[allele_col].astype(str)==str(allele)]
        freq_mats=[]

        for score_col in score_cols:
            sub=allele_df[[peptide_col,score_col]].copy()
            sub[score_col]=pd.to_numeric(sub[score_col],errors='coerce')
            sub=sub.dropna(subset=[peptide_col,score_col])

            if sub.empty:
                continue

            peptides=(
                sub.nlargest(min(top_n,len(sub)),score_col)[peptide_col]
                .astype(str)
                .tolist()
            )

            freq=_frequency_matrix(peptides)
            if freq is not None:
                freq_mats.append(freq)

        if not freq_mats:
            return None,0

        ref_index=freq_mats[0].index
        ref_cols=freq_mats[0].columns

        for freq in freq_mats[1:]:
            if not freq.index.equals(ref_index) or not freq.columns.equals(ref_cols):
                raise ValueError("Replicate frequency matrices do not have identical dimensions.")

        mean_freq=sum(freq_mats)/len(freq_mats)
        return _frequency_to_bits(mean_freq),len(freq_mats)

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------
    if highlight_cysteine:
        cs={aa:non_cysteine_color for aa in aa_list}
        cs['C']=cysteine_color
    elif color_scheme is not None:
        cs=color_scheme
    else:
        cs=STANDARD_COLOR_SCHEME

    ylim_final=ylim if ylim is not None else (0,np.log2(20))

    # Fixed row ordering matching the original figure:
    # Holec train -> FT -> PepCL -> MS train
    row_specs=[
        ("raw",holec_train_df,None),
        ("seed_average",background_df,ft_score_cols),
        ("seed_average",background_df,pepcl_score_cols),
        ("raw",ms_train_df,None),
    ]

    n_rows=len(row_specs)
    n_cols=len(alleles)

    fig,axes=plt.subplots(
        n_rows,n_cols,
        figsize=(figsize_per_cell[0]*n_cols,figsize_per_cell[1]*n_rows),
        dpi=dpi,
        squeeze=False,
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    for row_idx,(row_type,row_df,row_score_cols) in enumerate(row_specs):
        for col_idx,allele in enumerate(alleles):
            ax=axes[row_idx][col_idx]

            if row_type=="raw":
                pwm,n=_raw_logo(row_df,allele)
            else:
                pwm,n=_seed_averaged_logo(row_df,allele,row_score_cols)

            if pwm is None or pwm.empty:
                ax.axis('off')
                continue

            logo=logomaker.Logo(
                pwm,
                ax=ax,
                color_scheme=cs,
                vpad=vpad,
                width=logo_width,
            )

            logo.style_spines(visible=False)
            logo.style_spines(
                spines=['left','bottom'],
                visible=True,
                linewidth=spine_width,
            )

            if not remove_top_right_spines:
                logo.style_spines(
                    spines=['top','right'],
                    visible=True,
                    linewidth=spine_width,
                )

            ax.set_ylim(ylim_final)

            ax.set_ylabel(
                ylabel,
                fontsize=ylabel_fontsize,
                fontfamily=default_font,
                labelpad=ylabel_pad,
            )
            ax.set_xlabel(
                xlabel,
                fontsize=xlabel_fontsize,
                fontfamily=default_font,
                labelpad=xlabel_pad,
            )

            ax.set_xticks(range(len(pwm)))
            ax.set_xticklabels(range(1,len(pwm)+1))

            tick_kw={'direction':tick_direction}
            if tick_length is not None:
                tick_kw['length']=tick_length
            if tick_width is not None:
                tick_kw['width']=tick_width

            ax.tick_params(axis='both',which='major',**tick_kw)
            ax.tick_params(axis='x',labelsize=xtick_labelsize)
            ax.tick_params(axis='y',labelsize=ytick_labelsize)

            for lbl in ax.get_xticklabels()+ax.get_yticklabels():
                lbl.set_fontfamily(default_font)

            if title is not None:
                title_weight='bold' if title_bold else 'normal'

                if show_n_in_title:
                    if row_type=="raw":
                        title_text=f"{title} (n={n})"
                    else:
                        title_text=f"{title} (n={n} replicates)"
                else:
                    title_text=title

                ax.set_title(
                    title_text,
                    fontsize=title_fontsize,
                    fontfamily=default_font,
                    fontweight=title_weight,
                )

            if square_aspect:
                ax.set_box_aspect(1)

    plt.tight_layout()

    if save_path:
        fig.savefig(
            f"{save_path}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )
        fig.savefig(
            f"{save_path}.png",
            format='png',
            bbox_inches='tight',
            dpi=300,
        )

    plt.show()
    return fig,axes

def plot_cluster_vs_prior_advanced(
    df,
    line_styles=None,
    base_color='#2166ac',
    full_color='#d73027',
    cluster_color='#4dac26',
    prior_color='#888888',
    smooth=False,
    smooth_window=5,
    smooth_poly=2,
    show_markers=True,
    figsize=(10, 5),
    dpi=150,
    default_font='Nimbus Sans',
    xlabel='',
    ylabel='AP',
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    title='',
    title_fontsize=12,
    title_bold=True,
    xtick_labelsize=8,
    ytick_labelsize=9,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction='out',
    xtick_rotation=45,
    xtick_ha='right',
    axis_linewidth=0.8,
    xlim=None,
    ylim=None,
    grid_alpha=0.0,
    legend_mode='integrated',
    legend_loc='upper center',
    legend_bbox=(0.5, -0.12),
    legend_ncol=None,
    legend_fontsize=8,
    legend_handle_size=1.0,
    delta_base=False,
    save_path=None,
):

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    if line_styles is None:
        line_styles = {}

    default_marker = 'o' if show_markers else None
    default_markersize = 4 if show_markers else 0

    def _get(row_key, col_key, attr, fallback):
        return (line_styles
                .get(row_key, {})
                .get(col_key, {})
                .get(attr, fallback))

    special_rows = ['Base', 'Full']
    regular_rows = [r for r in df.index if r not in special_rows]
    x = np.arange(len(regular_rows))

    plot_df = df.copy().astype(float)
    if delta_base and 'Base' in plot_df.index:
        base_cluster = plot_df.loc['Base', 'cluster']
        base_prior = plot_df.loc['Base', 'prior']
        for row in plot_df.index:
            if row == 'Base':
                continue
            plot_df.loc[row, 'cluster'] = plot_df.loc[row, 'cluster'] - base_cluster
            plot_df.loc[row, 'prior'] = plot_df.loc[row, 'prior'] - base_prior
        effective_special_rows = ['Full']
        ylabel_final = ylabel if ylabel != 'AP' else 'ΔAP (vs Base)'
    else:
        effective_special_rows = special_rows
        ylabel_final = ylabel

    def maybe_smooth(vals):
        if smooth and len(vals) >= smooth_window:
            return savgol_filter(vals, window_length=smooth_window, polyorder=smooth_poly)
        return vals

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    handles = []
    labels_list = []

    special_defaults = {
        'Base': {'cluster': (base_color, '-',  'Base Cluster'),
                 'prior':   (base_color, ':',  'Base Prior')},
        'Full': {'cluster': (full_color, '-',  'Full Cluster'),
                 'prior':   (full_color, ':',  'Full Prior')},
    }

    for row in effective_special_rows:
        if row not in plot_df.index:
            continue
        for col_key in ['cluster', 'prior']:
            if col_key not in plot_df.columns:
                continue
            val = plot_df.loc[row, col_key]
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue

            def_color, def_ls, def_label = special_defaults[row][col_key]
            color  = _get(row, col_key, 'color',     def_color)
            lw     = _get(row, col_key, 'linewidth', 1.5)
            ls     = _get(row, col_key, 'linestyle', def_ls)
            alpha  = _get(row, col_key, 'alpha',     1.0)
            zorder = _get(row, col_key, 'zorder',    1)
            label  = _get(row, col_key, 'label',     def_label)

            ax.axhline(val, color=color, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=zorder)
            handles.append(Line2D([0], [0], color=color, linewidth=lw,
                                  linestyle=ls, alpha=alpha))
            labels_list.append(label)

    if delta_base:
        ax.axhline(0, color=base_color, linewidth=1.5, linestyle='-', zorder=1)
        handles.append(Line2D([0], [0], color=base_color, linewidth=1.5, linestyle='-'))
        labels_list.append('Base')

    for col_key, def_color, def_label in [
        ('cluster', cluster_color, 'Cluster'),
        ('prior',   prior_color,   'Prior'),
    ]:
        if col_key not in plot_df.columns:
            continue

        style_key = f'__{col_key}__'
        color  = _get(style_key, col_key, 'color',           def_color)
        lw     = _get(style_key, col_key, 'linewidth',       1.5)
        ls     = _get(style_key, col_key, 'linestyle',       '-')
        marker = _get(style_key, col_key, 'marker',          default_marker)
        ms     = _get(style_key, col_key, 'markersize',      default_markersize)
        mfc    = _get(style_key, col_key, 'markerfacecolor', color)
        mew    = _get(style_key, col_key, 'markeredgewidth', 0.8)
        alpha  = _get(style_key, col_key, 'alpha',           1.0)
        zorder = _get(style_key, col_key, 'zorder',          2)
        label  = _get(style_key, col_key, 'label',           def_label)

        vals = maybe_smooth(plot_df.loc[regular_rows, col_key].values.astype(float))

        line, = ax.plot(
            x, vals,
            color=color, linewidth=lw, linestyle=ls,
            marker=marker, markersize=ms,
            markerfacecolor=mfc, markeredgewidth=mew,
            markeredgecolor=color,
            alpha=alpha, zorder=zorder,
        )
        handles.append(line)
        labels_list.append(label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(regular_rows, rotation=xtick_rotation, ha=xtick_ha,
                       fontsize=xtick_labelsize)

    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font,
                  labelpad=xlabel_pad)
    ax.set_ylabel(ylabel_final, fontsize=ylabel_fontsize, fontfamily=default_font,
                  labelpad=ylabel_pad)

    title_weight = 'bold' if title_bold else 'normal'
    ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font,
                 fontweight=title_weight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(axis_linewidth)
    ax.spines['bottom'].set_linewidth(axis_linewidth)

    ax.tick_params(axis='both', which='major',
                   length=tick_length, width=tick_width,
                   direction=tick_direction, pad=tick_length + 2)
    ax.tick_params(axis='x', labelsize=xtick_labelsize)
    ax.tick_params(axis='y', labelsize=ytick_labelsize)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5,
                axis='y', zorder=0)

    if legend_ncol is None:
        legend_ncol = len(handles)

    if legend_mode == 'integrated' and handles:
        legend_kw = dict(frameon=False, fontsize=legend_fontsize, ncol=legend_ncol)
        if legend_bbox is not None:
            legend_kw.update(bbox_to_anchor=legend_bbox, loc=legend_loc)
        else:
            legend_kw.update(loc=legend_loc)
        legend = ax.legend(handles, labels_list, **legend_kw)
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(f"{save_path}{suffix}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}{suffix}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

    if legend_mode == 'separate' and handles:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(legend_handle_size * 2, n_items * 0.4 * legend_handle_size),
            dpi=dpi,
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        legend = ax_legend.legend(
            handles, labels_list,
            loc='center', frameon=False,
            fontsize=legend_fontsize, ncol=legend_ncol,
        )
        for text in legend.get_texts():
            text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(f"{save_path}_legend.pdf", format='pdf',
                               bbox_inches='tight', dpi=300)
            fig_legend.savefig(f"{save_path}_legend.png", format='png',
                               bbox_inches='tight', dpi=300)

        plt.show()


import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


def plot_jitter_advanced(
    df,
    col_label_dict,
    *,
    ci_low_df=None,
    ci_high_df=None,
    figsize=(10, 5),
    dpi=150,
    default_font='Nimbus Sans',
    jitter=0.2,
    point_size=40,
    point_alpha=1.0,
    point_edge_color='none',
    point_edge_width=0,
    point_marker='o',
    colors=None,
    show_mean=True,
    mean_above_points=False,
    mean_color='black',
    mean_width=1.2,
    mean_style='--',
    mean_length_pad=0.05,
    ci_line_width=1.0,
    ci_color='black',
    ci_cap_style='line',
    ci_cap_width=4.0,
    ci_alpha=1.0,
    title=None,
    title_fontsize=14,
    title_bold=False,
    xlabel=None,
    ylabel=None,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    xtick_labelsize=10,
    ytick_labelsize=10,
    xtick_rotation=0,
    xtick_ha='center',
    despine=True,
    spine_width=None,
    tick_length=None,
    tick_width=None,
    tick_direction='out',
    ylim=None,
    y_bottom_pad=0.0,
    y_clip_top=True,
    grid_alpha=0.0,
    hide_legend=False,
    legend_fontsize=8,
    legend_loc='upper center',
    legend_bbox=(0.5, -0.15),
    legend_ncol=4,
    legend_frameon=False,
    square_aspect=False,
    save_path=None,
    show=True,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    alleles = df.index.tolist()
    n_alleles = len(alleles)
    cols = list(col_label_dict.keys())

    if colors is not None:
        allele_colors = {allele: colors[i % len(colors)] for i, allele in enumerate(alleles)}
    else:
        palette = [mcolors.to_hex(c) for c in plt.cm.tab20.colors[:max(n_alleles, 1)]]
        allele_colors = {allele: palette[i % len(palette)] for i, allele in enumerate(alleles)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if jitter > 0 and n_alleles > 1:
        offsets = np.linspace(-jitter / 2, jitter / 2, n_alleles)
    else:
        offsets = np.zeros(n_alleles)

    mean_zorder = 4 if mean_above_points else 2
    point_zorder = 3

    legend_alleles_seen = set()

    for x_pos, col in enumerate(cols):
        if col not in df.columns:
            continue

        for ai, (allele, offset) in enumerate(zip(alleles, offsets)):
            val = df.loc[allele, col] if allele in df.index else np.nan
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
            if np.isnan(val):
                continue

            x = x_pos + offset

            if ci_low_df is not None and ci_high_df is not None:
                if col in ci_low_df.columns and col in ci_high_df.columns:
                    ci_lo = ci_low_df.loc[allele, col] if allele in ci_low_df.index else None
                    ci_hi = ci_high_df.loc[allele, col] if allele in ci_high_df.index else None
                    if ci_lo is not None and ci_hi is not None:
                        try:
                            ci_lo = float(ci_lo)
                            ci_hi = float(ci_hi)
                        except (ValueError, TypeError):
                            ci_lo = None
                            ci_hi = None

                        if ci_lo is not None and ci_hi is not None and not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                            ax.plot(
                                [x, x], [ci_lo, ci_hi],
                                color=ci_color, linewidth=ci_line_width,
                                alpha=ci_alpha, solid_capstyle='butt',
                                zorder=point_zorder - 1,
                            )

                            if ci_cap_style == 'cap':
                                half_cap = jitter * (ci_cap_width / 100.0) if jitter > 0 else 0.02
                                ax.plot(
                                    [x - half_cap, x + half_cap], [ci_lo, ci_lo],
                                    color=ci_color, linewidth=ci_line_width,
                                    alpha=ci_alpha, zorder=point_zorder - 1,
                                )
                                ax.plot(
                                    [x - half_cap, x + half_cap], [ci_hi, ci_hi],
                                    color=ci_color, linewidth=ci_line_width,
                                    alpha=ci_alpha, zorder=point_zorder - 1,
                                )

            label = None
            if not hide_legend and allele not in legend_alleles_seen:
                label = allele
                legend_alleles_seen.add(allele)

            ec = point_edge_color
            ew = point_edge_width

            ax.scatter(
                x, val,
                color=allele_colors[allele],
                s=point_size,
                alpha=point_alpha,
                marker=point_marker,
                edgecolors=ec if ec != 'none' else 'none',
                linewidths=ew,
                zorder=point_zorder,
                label=label,
            )

        if show_mean:
            col_vals = [float(df.loc[a, col]) for a in alleles
                        if a in df.index and col in df.columns
                        and not np.isnan(float(df.loc[a, col]))]
            if col_vals:
                mean_val = np.mean(col_vals)
                half_extent = jitter / 2 + mean_length_pad
                ax.plot(
                    [x_pos - half_extent, x_pos + half_extent],
                    [mean_val, mean_val],
                    color=mean_color, linewidth=mean_width,
                    linestyle=mean_style, zorder=mean_zorder,
                )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        list(col_label_dict.values()),
        rotation=xtick_rotation, ha=xtick_ha,
        fontsize=xtick_labelsize,
    )
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(default_font)

    ax.tick_params(axis='y', labelsize=ytick_labelsize)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font,
                      labelpad=xlabel_pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font,
                      labelpad=ylabel_pad)

    if title is not None:
        title_weight = 'bold' if title_bold else 'normal'
        ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font,
                     fontweight=title_weight)

    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if spine_width is not None:
        for side in ['left', 'bottom', 'top', 'right']:
            if ax.spines[side].get_visible():
                ax.spines[side].set_linewidth(spine_width)

    tick_kw = {'direction': tick_direction}
    if tick_length is not None:
        tick_kw['length'] = tick_length
    if tick_width is not None:
        tick_kw['width'] = tick_width
    ax.tick_params(axis='both', which='major', **tick_kw)

    if ylim is not None:
        ax.set_ylim(ylim)

    if y_bottom_pad > 0 or y_clip_top:
        fig.canvas.draw_idle()
        yticks = ax.get_yticks()
        cur_ymin, cur_ymax = ax.get_ylim()

        visible_ticks = yticks[(yticks >= cur_ymin) & (yticks <= cur_ymax)]
        if len(visible_ticks) > 0:
            if y_bottom_pad > 0:
                new_ymin = visible_ticks[0] - y_bottom_pad
                ax.set_ylim(bottom=new_ymin)
            if y_clip_top:
                ax.set_ylim(top=visible_ticks[-1])
                ax.spines['left'].set_bounds(ax.get_ylim()[0], visible_ticks[-1])

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5,
                axis='y', zorder=0)

    if square_aspect:
        ax.set_box_aspect(1)

    if not hide_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if show_mean:
                handles.append(Line2D([0], [0], color=mean_color, linewidth=mean_width,
                                      linestyle=mean_style))
                labels.append('Mean')

            leg = fig.legend(
                handles, labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                ncol=legend_ncol,
                frameon=legend_frameon,
                fontsize=legend_fontsize,
            )
            for text in leg.get_texts():
                text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return ax

def plot_jitter_advanced_v2(
    df,
    col_label_dict,
    *,
    ci_low_df=None,
    ci_high_df=None,
    figsize=(10,5),
    dpi=150,
    default_font='Nimbus Sans',

    # Points.
    jitter=0.2,
    point_size=40,
    point_alpha=1.0,
    point_edge_color='none',
    point_edge_width=0,
    point_marker='o',
    colors=None,

    # Mean.
    show_mean=True,
    mean_above_points=False,
    mean_color='black',
    mean_width=1.2,
    mean_style='--',
    mean_length_pad=0.05,

    # CI.
    ci_line_width=1.0,
    ci_color='black',
    ci_cap_style='line',
    ci_cap_width=4.0,
    ci_alpha=1.0,

    # Titles / labels.
    title=None,
    title_fontsize=14,
    title_bold=False,
    xlabel=None,
    ylabel=None,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    xlabel_pad=4.0,
    ylabel_pad=4.0,

    # Tick labels.
    xtick_labelsize=10,
    ytick_labelsize=10,
    xtick_rotation=0,
    xtick_ha='center',
    xtick_va='top',
    xtick_rotation_mode='anchor',

    # Axes.
    despine=True,
    spine_width=None,
    tick_length=None,
    tick_width=None,
    tick_direction='out',
    ylim=None,
    y_bottom_pad=0.0,
    y_clip_top=True,
    grid_alpha=0.0,

    # Legend.
    hide_legend=False,
    legend_fontsize=8,
    legend_loc='upper center',
    legend_bbox=(0.5,-0.15),
    legend_ncol=4,
    legend_frameon=False,

    # Layout / output.
    square_aspect=False,
    save_path=None,
    show=True,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import numpy as np

    matplotlib.rcParams['font.family']='sans-serif'
    matplotlib.rcParams['font.sans-serif']=['Nimbus Sans','Nimbus Sans L','Arial','Helvetica']
    matplotlib.rcParams['pdf.fonttype']=42

    alleles=df.index.tolist()
    n_alleles=len(alleles)
    cols=list(col_label_dict.keys())

    if colors is not None:
        allele_colors={
            allele:colors[i%len(colors)]
            for i,allele in enumerate(alleles)
        }
    else:
        palette=[
            mcolors.to_hex(c)
            for c in plt.cm.tab20.colors[:max(n_alleles,1)]
        ]
        allele_colors={
            allele:palette[i%len(palette)]
            for i,allele in enumerate(alleles)
        }

    fig,ax=plt.subplots(figsize=figsize,dpi=dpi)

    if jitter>0 and n_alleles>1:
        offsets=np.linspace(-jitter/2,jitter/2,n_alleles)
    else:
        offsets=np.zeros(n_alleles)

    mean_zorder=4 if mean_above_points else 2
    point_zorder=3

    legend_alleles_seen=set()

    for x_pos,col in enumerate(cols):
        if col not in df.columns:
            continue

        for allele,offset in zip(alleles,offsets):
            val=df.loc[allele,col] if allele in df.index else np.nan

            try:
                val=float(val)
            except (ValueError,TypeError):
                continue

            if np.isnan(val):
                continue

            x=x_pos+offset

            # Optional confidence intervals.
            if ci_low_df is not None and ci_high_df is not None:
                if col in ci_low_df.columns and col in ci_high_df.columns:
                    ci_lo=ci_low_df.loc[allele,col] if allele in ci_low_df.index else None
                    ci_hi=ci_high_df.loc[allele,col] if allele in ci_high_df.index else None

                    if ci_lo is not None and ci_hi is not None:
                        try:
                            ci_lo=float(ci_lo)
                            ci_hi=float(ci_hi)
                        except (ValueError,TypeError):
                            ci_lo=None
                            ci_hi=None

                        if (
                            ci_lo is not None
                            and ci_hi is not None
                            and not (np.isnan(ci_lo) or np.isnan(ci_hi))
                        ):
                            ax.plot(
                                [x,x],[ci_lo,ci_hi],
                                color=ci_color,
                                linewidth=ci_line_width,
                                alpha=ci_alpha,
                                solid_capstyle='butt',
                                zorder=point_zorder-1,
                            )

                            if ci_cap_style=='cap':
                                half_cap=(
                                    jitter*(ci_cap_width/100.0)
                                    if jitter>0 else 0.02
                                )

                                ax.plot(
                                    [x-half_cap,x+half_cap],
                                    [ci_lo,ci_lo],
                                    color=ci_color,
                                    linewidth=ci_line_width,
                                    alpha=ci_alpha,
                                    zorder=point_zorder-1,
                                )

                                ax.plot(
                                    [x-half_cap,x+half_cap],
                                    [ci_hi,ci_hi],
                                    color=ci_color,
                                    linewidth=ci_line_width,
                                    alpha=ci_alpha,
                                    zorder=point_zorder-1,
                                )

            label=None

            if not hide_legend and allele not in legend_alleles_seen:
                label=allele
                legend_alleles_seen.add(allele)

            ax.scatter(
                x,
                val,
                color=allele_colors[allele],
                s=point_size,
                alpha=point_alpha,
                marker=point_marker,
                edgecolors=point_edge_color if point_edge_color!='none' else 'none',
                linewidths=point_edge_width,
                zorder=point_zorder,
                label=label,
            )

        # Mean line.
        if show_mean:
            col_vals=[]

            for allele in alleles:
                if allele not in df.index or col not in df.columns:
                    continue

                try:
                    value=float(df.loc[allele,col])
                except (ValueError,TypeError):
                    continue

                if not np.isnan(value):
                    col_vals.append(value)

            if col_vals:
                mean_val=np.mean(col_vals)
                half_extent=jitter/2+mean_length_pad

                ax.plot(
                    [x_pos-half_extent,x_pos+half_extent],
                    [mean_val,mean_val],
                    color=mean_color,
                    linewidth=mean_width,
                    linestyle=mean_style,
                    zorder=mean_zorder,
                )

    # X ticks.
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        list(col_label_dict.values()),
        rotation=xtick_rotation,
        ha=xtick_ha,
        va=xtick_va,
        rotation_mode=xtick_rotation_mode,
        fontsize=xtick_labelsize,
    )

    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(default_font)

    # Y ticks.
    ax.tick_params(axis='y',labelsize=ytick_labelsize)

    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    # Axis labels.
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=xlabel_fontsize,
            fontfamily=default_font,
            labelpad=xlabel_pad,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=ylabel_fontsize,
            fontfamily=default_font,
            labelpad=ylabel_pad,
        )

    # Title.
    if title is not None:
        title_weight='bold' if title_bold else 'normal'

        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight=title_weight,
        )

    # Spines.
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if spine_width is not None:
        for side in ['left','bottom','top','right']:
            if ax.spines[side].get_visible():
                ax.spines[side].set_linewidth(spine_width)

    # Tick geometry.
    tick_kw={'direction':tick_direction}

    if tick_length is not None:
        tick_kw['length']=tick_length

    if tick_width is not None:
        tick_kw['width']=tick_width

    ax.tick_params(axis='both',which='major',**tick_kw)

    # Y limits.
    if ylim is not None:
        ax.set_ylim(ylim)

    if y_bottom_pad>0 or y_clip_top:
        fig.canvas.draw_idle()

        yticks=ax.get_yticks()
        cur_ymin,cur_ymax=ax.get_ylim()

        visible_ticks=yticks[
            (yticks>=cur_ymin)
            & (yticks<=cur_ymax)
        ]

        if len(visible_ticks)>0:
            if y_bottom_pad>0:
                new_ymin=visible_ticks[0]-y_bottom_pad
                ax.set_ylim(bottom=new_ymin)

            if y_clip_top:
                ax.set_ylim(top=visible_ticks[-1])
                ax.spines['left'].set_bounds(
                    ax.get_ylim()[0],
                    visible_ticks[-1]
                )

    # Grid.
    if grid_alpha>0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis='y',
            zorder=0,
        )

    if square_aspect:
        ax.set_box_aspect(1)

    # Legend.
    if not hide_legend:
        handles,labels=ax.get_legend_handles_labels()

        if handles:
            if show_mean:
                handles.append(
                    Line2D(
                        [0],[0],
                        color=mean_color,
                        linewidth=mean_width,
                        linestyle=mean_style,
                    )
                )
                labels.append('Mean')

            leg=fig.legend(
                handles,
                labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                ncol=legend_ncol,
                frameon=legend_frameon,
                fontsize=legend_fontsize,
            )

            for text in leg.get_texts():
                text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(
            f"{save_path}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )

        fig.savefig(
            f"{save_path}.png",
            format='png',
            bbox_inches='tight',
            dpi=300,
        )

    if show:
        plt.show()

    return ax

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


def plot_jitter_advanced_v3(
    df,
    col_label_dict,
    *,
    ci_df=None,
    ci_low_suffix="_ci_low",
    ci_high_suffix="_ci_high",

    figsize=(10,5),
    dpi=150,
    default_font="Nimbus Sans",

    # points
    jitter=0.2,
    point_size=40,
    point_alpha=1.0,
    point_edge_color="none",
    point_edge_width=0,
    point_marker="o",
    colors=None,

    # mean
    show_mean=True,
    mean_above_points=False,
    mean_color="black",
    mean_width=1.2,
    mean_style="--",
    mean_length_pad=0.05,

    # confidence intervals
    show_ci=True,
    ci_color=None,                 # None = same color as corresponding marker
    ci_line_width=1.0,
    ci_linestyle="-",
    ci_alpha=1.0,
    ci_show_caps=True,
    ci_cap_size=4.0,               # points
    ci_cap_thickness=None,         # None -> ci_line_width
    ci_zorder=None,                # None -> directly behind points

    # labels/title
    title=None,
    title_fontsize=14,
    title_bold=False,
    xlabel=None,
    ylabel=None,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    xlabel_pad=4.0,
    ylabel_pad=4.0,

    # ticks
    xtick_labelsize=10,
    ytick_labelsize=10,
    xtick_rotation=0,
    xtick_ha="center",

    # axes
    despine=True,
    spine_width=None,
    tick_length=None,
    tick_width=None,
    tick_direction="out",
    ylim=None,
    y_bottom_pad=0.0,
    y_clip_top=True,
    grid_alpha=0.0,

    # legend
    hide_legend=False,
    legend_fontsize=8,
    legend_loc="upper center",
    legend_bbox=(0.5,-0.15),
    legend_ncol=4,
    legend_frameon=False,

    square_aspect=False,

    # saving
    save_path=None,
    save_pdf=True,
    save_png=True,
    save_dpi=300,

    show=True,
):
    """
    Advanced allele jitter plot with optional per-point confidence intervals.

    CI dataframe format
    -------------------
    For each plotted dataframe column `<col>`, ci_df should contain:

        <col>_ci_low
        <col>_ci_high

    with the same allele index as df.

    Example
    -------
    Main dataframe:
        episcan_r9rt_ft
        episcan_r9rt_pepcl

    CI dataframe:
        episcan_r9rt_ft_ci_low
        episcan_r9rt_ft_ci_high
        episcan_r9rt_pepcl_ci_low
        episcan_r9rt_pepcl_ci_high

    By default:
        - CI color matches the corresponding allele marker.
        - CI bars have T caps.
        - CI bars are drawn behind the marker.
    """

    # ================================================================
    # Global plotting defaults
    # ================================================================
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
    ]
    matplotlib.rcParams["pdf.fonttype"]=42

    alleles=df.index.tolist()
    n_alleles=len(alleles)
    cols=list(col_label_dict.keys())

    # ================================================================
    # Resolve allele colors
    # ================================================================
    if colors is not None:
        allele_colors={
            allele:colors[i % len(colors)]
            for i,allele in enumerate(alleles)
        }
    else:
        palette=[
            mcolors.to_hex(c)
            for c in plt.cm.tab20.colors[:max(n_alleles,1)]
        ]

        allele_colors={
            allele:palette[i % len(palette)]
            for i,allele in enumerate(alleles)
        }

    # ================================================================
    # Figure
    # ================================================================
    fig,ax=plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    # ================================================================
    # Fixed jitter offsets
    # ================================================================
    if jitter>0 and n_alleles>1:
        offsets=np.linspace(
            -jitter/2,
            jitter/2,
            n_alleles,
        )
    else:
        offsets=np.zeros(
            n_alleles
        )

    mean_zorder=(
        4
        if mean_above_points
        else 2
    )

    point_zorder=3

    resolved_ci_zorder=(
        point_zorder-1
        if ci_zorder is None
        else ci_zorder
    )

    legend_alleles_seen=set()

    # ================================================================
    # Plot each model / dataframe column
    # ================================================================
    for x_pos,col in enumerate(cols):

        if col not in df.columns:
            continue

        # ------------------------------------------------------------
        # Plot each allele
        # ------------------------------------------------------------
        for allele,offset in zip(
            alleles,
            offsets,
        ):
            if allele not in df.index:
                continue

            val=df.loc[
                allele,
                col
            ]

            try:
                val=float(val)
            except (ValueError,TypeError):
                continue

            if not np.isfinite(val):
                continue

            x=x_pos+offset
            marker_color=allele_colors[allele]

            # ========================================================
            # Confidence interval
            # ========================================================
            if (
                show_ci
                and ci_df is not None
            ):
                low_col=f"{col}{ci_low_suffix}"
                high_col=f"{col}{ci_high_suffix}"

                if (
                    allele in ci_df.index
                    and low_col in ci_df.columns
                    and high_col in ci_df.columns
                ):
                    ci_lo=ci_df.loc[
                        allele,
                        low_col
                    ]

                    ci_hi=ci_df.loc[
                        allele,
                        high_col
                    ]

                    try:
                        ci_lo=float(ci_lo)
                        ci_hi=float(ci_hi)
                    except (ValueError,TypeError):
                        ci_lo=np.nan
                        ci_hi=np.nan

                    if (
                        np.isfinite(ci_lo)
                        and np.isfinite(ci_hi)
                    ):
                        # ------------------------------------------------
                        # Matplotlib expects non-negative error distances,
                        # not absolute CI bounds.
                        # ------------------------------------------------
                        lower_err=max(
                            0.0,
                            val-ci_lo
                        )

                        upper_err=max(
                            0.0,
                            ci_hi-val
                        )

                        resolved_ci_color=(
                            marker_color
                            if ci_color is None
                            else ci_color
                        )

                        # ------------------------------------------------
                        # IMPORTANT:
                        #
                        # Do NOT pass elinestyle= here because older
                        # Matplotlib versions do not support it directly
                        # in errorbar().
                        # ------------------------------------------------
                        err=ax.errorbar(
                            x,
                            val,

                            yerr=np.array([
                                [lower_err],
                                [upper_err],
                            ]),

                            fmt="none",

                            ecolor=resolved_ci_color,
                            elinewidth=ci_line_width,

                            capsize=(
                                ci_cap_size
                                if ci_show_caps
                                else 0
                            ),

                            capthick=(
                                ci_line_width
                                if ci_cap_thickness is None
                                else ci_cap_thickness
                            ),

                            alpha=ci_alpha,
                            zorder=resolved_ci_zorder,
                        )

                        # ------------------------------------------------
                        # Apply linestyle AFTER creating the errorbar.
                        #
                        # err[1] -> cap Line2D objects
                        # err[2] -> bar LineCollection objects
                        # ------------------------------------------------
                        for capline in err[1]:
                            capline.set_linestyle(
                                ci_linestyle
                            )

                        for barlinecol in err[2]:
                            barlinecol.set_linestyle(
                                ci_linestyle
                            )

            # ========================================================
            # Marker
            # ========================================================
            label=None

            if (
                not hide_legend
                and allele not in legend_alleles_seen
            ):
                label=allele
                legend_alleles_seen.add(
                    allele
                )

            ax.scatter(
                x,
                val,

                color=marker_color,
                s=point_size,
                alpha=point_alpha,
                marker=point_marker,

                edgecolors=(
                    point_edge_color
                    if point_edge_color!="none"
                    else "none"
                ),

                linewidths=point_edge_width,

                zorder=point_zorder,
                label=label,
            )

        # ============================================================
        # Mean line
        # ============================================================
        if show_mean:
            col_vals=[]

            for allele in alleles:
                if allele not in df.index:
                    continue

                try:
                    v=float(
                        df.loc[
                            allele,
                            col
                        ]
                    )
                except (ValueError,TypeError):
                    continue

                if np.isfinite(v):
                    col_vals.append(v)

            if col_vals:
                mean_val=np.mean(
                    col_vals
                )

                half_extent=(
                    jitter/2
                    + mean_length_pad
                )

                ax.plot(
                    [
                        x_pos-half_extent,
                        x_pos+half_extent,
                    ],
                    [
                        mean_val,
                        mean_val,
                    ],

                    color=mean_color,
                    linewidth=mean_width,
                    linestyle=mean_style,
                    zorder=mean_zorder,
                )

    # ================================================================
    # X ticks
    # ================================================================
    ax.set_xticks(
        range(len(cols))
    )

    ax.set_xticklabels(
        list(
            col_label_dict.values()
        ),
        rotation=xtick_rotation,
        ha=xtick_ha,
        fontsize=xtick_labelsize,
    )

    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(
            default_font
        )

    # ================================================================
    # Y ticks
    # ================================================================
    ax.tick_params(
        axis="y",
        labelsize=ytick_labelsize,
    )

    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(
            default_font
        )

    # ================================================================
    # Axis labels
    # ================================================================
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=xlabel_fontsize,
            fontfamily=default_font,
            labelpad=xlabel_pad,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=ylabel_fontsize,
            fontfamily=default_font,
            labelpad=ylabel_pad,
        )

    # ================================================================
    # Title
    # ================================================================
    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight=(
                "bold"
                if title_bold
                else "normal"
            ),
        )

    # ================================================================
    # Spines
    # ================================================================
    if despine:
        ax.spines["top"].set_visible(
            False
        )
        ax.spines["right"].set_visible(
            False
        )

    if spine_width is not None:
        for side in [
            "left",
            "bottom",
            "top",
            "right",
        ]:
            if ax.spines[side].get_visible():
                ax.spines[side].set_linewidth(
                    spine_width
                )

    # ================================================================
    # Tick styling
    # ================================================================
    tick_kw={
        "direction":tick_direction
    }

    if tick_length is not None:
        tick_kw["length"]=tick_length

    if tick_width is not None:
        tick_kw["width"]=tick_width

    ax.tick_params(
        axis="both",
        which="major",
        **tick_kw,
    )

    # ================================================================
    # Explicit y limits
    # ================================================================
    if ylim is not None:
        ax.set_ylim(
            ylim
        )

    # ================================================================
    # Optional bottom padding / top clipping
    # ================================================================
    if (
        y_bottom_pad>0
        or y_clip_top
    ):
        fig.canvas.draw_idle()

        yticks=ax.get_yticks()
        cur_ymin,cur_ymax=ax.get_ylim()

        visible_ticks=yticks[
            (yticks>=cur_ymin)
            & (yticks<=cur_ymax)
        ]

        if len(visible_ticks)>0:

            if y_bottom_pad>0:
                new_ymin=(
                    visible_ticks[0]
                    - y_bottom_pad
                )

                ax.set_ylim(
                    bottom=new_ymin
                )

            if y_clip_top:
                ax.set_ylim(
                    top=visible_ticks[-1]
                )

                ax.spines["left"].set_bounds(
                    ax.get_ylim()[0],
                    visible_ticks[-1],
                )

    # ================================================================
    # Grid
    # ================================================================
    if grid_alpha>0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle="-",
            linewidth=0.5,
            axis="y",
            zorder=0,
        )

    # ================================================================
    # Square panel
    # ================================================================
    if square_aspect:
        ax.set_box_aspect(
            1
        )

    # ================================================================
    # Legend
    # ================================================================
    if not hide_legend:
        handles,labels=(
            ax.get_legend_handles_labels()
        )

        if handles:

            if show_mean:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=mean_color,
                        linewidth=mean_width,
                        linestyle=mean_style,
                    )
                )

                labels.append(
                    "Mean"
                )

            leg=fig.legend(
                handles,
                labels,

                loc=legend_loc,
                bbox_to_anchor=legend_bbox,

                ncol=legend_ncol,
                frameon=legend_frameon,
                fontsize=legend_fontsize,
            )

            for text in leg.get_texts():
                text.set_fontfamily(
                    default_font
                )

    # ================================================================
    # Layout
    # ================================================================
    plt.tight_layout()

    # ================================================================
    # Save
    # ================================================================
    if save_path:
        base=str(
            save_path
        )

        if base.lower().endswith(
            (".pdf",".png")
        ):
            base=base.rsplit(
                ".",
                1
            )[0]

        if save_pdf:
            fig.savefig(
                f"{base}.pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=save_dpi,
            )

        if save_png:
            fig.savefig(
                f"{base}.png",
                format="png",
                bbox_inches="tight",
                dpi=save_dpi,
            )

    # ================================================================
    # Show
    # ================================================================
    if show:
        plt.show()

    return ax

def plot_bar_advanced(
    df,
    col,
    figsize=(10, 4),
    dpi=150,
    default_font='Nimbus Sans',
    color='steelblue',
    bar_width=0.8,
    bar_edgecolor='white',
    bar_edgewidth=0.3,
    alpha=1.0,
    highlight_ticks=None,
    highlight_colors=None,
    horizontal=False,
    xlabel='',
    ylabel=None,
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    title='',
    title_fontsize=12,
    title_bold=True,
    xtick_labelsize=8,
    ytick_labelsize=8,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction='out',
    rotation=0,
    axis_linewidth=0.8,
    ylim=None,
    xlim=None,
    grid_alpha=0.0,
    save_path=None,
):
    """
    Plot a single-column bar chart (vertical or horizontal).
    """
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    labels = df.index.tolist()
    values = df[col].values

    if highlight_ticks is not None and highlight_colors is not None:
        highlight_set = set(highlight_ticks)
        bar_colors = [highlight_colors[0] if lbl in highlight_set else highlight_colors[1]
                      for lbl in labels]
    else:
        bar_colors = [color] * len(labels)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if horizontal:
        ax.barh(
            range(len(df)), values,
            color=bar_colors,
            edgecolor=bar_edgecolor,
            linewidth=bar_edgewidth,
            alpha=alpha,
            height=bar_width,
        )
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels, fontsize=ytick_labelsize)
        ax.set_xlabel(
            ylabel if ylabel is not None else col,
            fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad,
        )
        if xlabel:
            ax.set_ylabel(xlabel, fontsize=ylabel_fontsize, fontfamily=default_font,
                          labelpad=ylabel_pad)
        ax.invert_yaxis()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    else:
        ax.bar(
            range(len(df)), values,
            color=bar_colors,
            edgecolor=bar_edgecolor,
            linewidth=bar_edgewidth,
            alpha=alpha,
            width=bar_width,
        )
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=rotation, ha='center', fontsize=xtick_labelsize)
        ax.set_ylabel(
            ylabel if ylabel is not None else col,
            fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad,
        )
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font,
                          labelpad=xlabel_pad)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

    title_weight = 'bold' if title_bold else 'normal'
    ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font, fontweight=title_weight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(axis_linewidth)
    ax.spines['bottom'].set_linewidth(axis_linewidth)

    ax.tick_params(
        axis='both', which='major',
        length=tick_length,
        width=tick_width,
        direction=tick_direction,
        pad=tick_length + 2,
    )
    ax.tick_params(axis='x', labelsize=xtick_labelsize)
    ax.tick_params(axis='y', labelsize=ytick_labelsize)

    if grid_alpha > 0:
        grid_axis = 'x' if horizontal else 'y'
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, axis=grid_axis, zorder=0)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_bar_advanced_w_ci(
    df,
    col,
    figsize=(10, 4),
    dpi=150,
    default_font='Nimbus Sans',
    color='steelblue',
    bar_width=0.8,
    bar_edgecolor='white',
    bar_edgewidth=0.3,
    alpha=1.0,
    highlight_ticks=None,
    highlight_colors=None,
    horizontal=False,
    xlabel='',
    ylabel=None,
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4.0,
    ylabel_pad=4.0,
    title='',
    title_fontsize=12,
    title_bold=True,
    xtick_labelsize=8,
    ytick_labelsize=8,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction='out',
    rotation=0,
    axis_linewidth=0.8,
    ylim=None,
    xlim=None,
    grid_alpha=0.0,
    interval_low_df=None,
    interval_high_df=None,
    interval_color='black',
    interval_linewidth=0.8,
    interval_capsize=0.0,
    interval_capthick=None,
    interval_alpha=1.0,
    interval_zorder=4,
    save_path=None,
):
    """
    Plot a single-column bar chart (vertical or horizontal).

    Optional interval_low_df and interval_high_df should have the same
    allele x position layout as df. Rows are aligned to df.index.
    """
    matplotlib.rcParams['font.family']='sans-serif'
    matplotlib.rcParams['font.sans-serif']=['Nimbus Sans','Nimbus Sans L','Arial','Helvetica']
    matplotlib.rcParams['pdf.fonttype']=42

    labels=df.index.tolist()
    values=pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)

    if (interval_low_df is None) != (interval_high_df is None):
        raise ValueError("Provide both interval_low_df and interval_high_df, or neither.")

    low=high=None
    if interval_low_df is not None:
        if col not in interval_low_df.columns or col not in interval_high_df.columns:
            raise ValueError(f"Column {col!r} missing from interval dataframe(s).")

        low=pd.to_numeric(interval_low_df.reindex(labels)[col],errors='coerce').to_numpy(dtype=float)
        high=pd.to_numeric(interval_high_df.reindex(labels)[col],errors='coerce').to_numpy(dtype=float)

        valid=np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
        if np.any(low[valid]>high[valid]):
            raise ValueError("Interval low exceeds interval high for at least one row.")
        if np.any(values[valid]<low[valid]) or np.any(values[valid]>high[valid]):
            raise ValueError("At least one bar value lies outside its interval.")

    if highlight_ticks is not None and highlight_colors is not None:
        highlight_set=set(highlight_ticks)
        bar_colors=[
            highlight_colors[0] if lbl in highlight_set else highlight_colors[1]
            for lbl in labels
        ]
    else:
        bar_colors=[color]*len(labels)

    fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
    positions=np.arange(len(df))

    if horizontal:
        ax.barh(
            positions,values,
            color=bar_colors,
            edgecolor=bar_edgecolor,
            linewidth=bar_edgewidth,
            alpha=alpha,
            height=bar_width,
        )

        if low is not None:
            valid=np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
            xerr=np.vstack([values[valid]-low[valid],high[valid]-values[valid]])

            ax.errorbar(
                values[valid],positions[valid],
                xerr=xerr,
                fmt='none',
                ecolor=interval_color,
                elinewidth=interval_linewidth,
                capsize=interval_capsize,
                capthick=interval_capthick,
                alpha=interval_alpha,
                zorder=interval_zorder,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(labels,fontsize=ytick_labelsize)
        ax.set_xlabel(
            ylabel if ylabel is not None else col,
            fontsize=xlabel_fontsize,fontfamily=default_font,labelpad=xlabel_pad,
        )

        if xlabel:
            ax.set_ylabel(
                xlabel,fontsize=ylabel_fontsize,
                fontfamily=default_font,labelpad=ylabel_pad,
            )

        ax.invert_yaxis()

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    else:
        ax.bar(
            positions,values,
            color=bar_colors,
            edgecolor=bar_edgecolor,
            linewidth=bar_edgewidth,
            alpha=alpha,
            width=bar_width,
        )

        if low is not None:
            valid=np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
            yerr=np.vstack([values[valid]-low[valid],high[valid]-values[valid]])

            ax.errorbar(
                positions[valid],values[valid],
                yerr=yerr,
                fmt='none',
                ecolor=interval_color,
                elinewidth=interval_linewidth,
                capsize=interval_capsize,
                capthick=interval_capthick,
                alpha=interval_alpha,
                zorder=interval_zorder,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            labels,rotation=rotation,ha='center',
            fontsize=xtick_labelsize,
        )

        ax.set_ylabel(
            ylabel if ylabel is not None else col,
            fontsize=ylabel_fontsize,
            fontfamily=default_font,
            labelpad=ylabel_pad,
        )

        if xlabel:
            ax.set_xlabel(
                xlabel,fontsize=xlabel_fontsize,
                fontfamily=default_font,labelpad=xlabel_pad,
            )

        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

    title_weight='bold' if title_bold else 'normal'
    ax.set_title(
        title,fontsize=title_fontsize,
        fontfamily=default_font,fontweight=title_weight,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(axis_linewidth)
    ax.spines['bottom'].set_linewidth(axis_linewidth)

    ax.tick_params(
        axis='both',which='major',
        length=tick_length,
        width=tick_width,
        direction=tick_direction,
        pad=tick_length+2,
    )
    ax.tick_params(axis='x',labelsize=xtick_labelsize)
    ax.tick_params(axis='y',labelsize=ytick_labelsize)

    if grid_alpha>0:
        grid_axis='x' if horizontal else 'y'
        ax.grid(
            True,alpha=grid_alpha,linestyle='-',
            linewidth=0.5,axis=grid_axis,zorder=0,
        )

    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf",format='pdf',bbox_inches='tight',dpi=300)
        fig.savefig(f"{save_path}.png",format='png',bbox_inches='tight',dpi=300)

    plt.show()
    return fig,ax

def plot_heatmap(
    df: pd.DataFrame,
    *,
    ax: plt.Axes = None,
    lower_triangle_only: bool = False,
    exclude_diagonal: bool = False,
    cmap: str = None,
    color_list: list = None,
    vmin: float = None,
    vmax: float = None,
    cmap_n_bins: int = 256,
    x_col_labels: dict = None,
    y_col_labels: dict = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    title_fontsize: float = 14,
    title_bold: bool = False,
    xlabel_fontsize: float = 12,
    ylabel_fontsize: float = 12,
    xlabel_pad: float = 4.0,
    ylabel_pad: float = 4.0,
    xtick_labelsize: float = 10,
    ytick_labelsize: float = 10,
    xtick_rotation: float = 0,
    xtick_ha: str = None,
    xtick_va: str = None,
    xtick_x_offset: float = 0.0,
    xtick_y_offset: float = 0.0,
    show_xticks: bool = False,
    show_yticks: bool = False,
    xtick_length: float = 3.5,
    ytick_length: float = 3.5,
    xtick_width: float = 0.8,
    ytick_width: float = 0.8,
    xtick_direction: str = "out",
    ytick_direction: str = "out",
    tick_color: str = "black",
    cell_outline_color: str = 'black',
    cell_outline_width: float = 0.5,
    rasterize_cells: bool = False,
    annot: bool = False,
    annot_fontsize: float = 9,
    annot_decimals: int = 2,
    annot_color: str = None,
    annot_color_light: str = 'white',
    annot_color_dark: str = 'black',
    annot_auto_contrast: bool = True,
    annot_luminance_threshold: float = 0.55,
    show_colorbar: bool = True,
    cbar_label: str = None,
    cbar_label_fontsize: float = 12,
    cbar_labelsize: float = 10,
    cbar_shrink: float = 1.0,
    cbar_aspect: float = 20,
    cbar_pad: float = 0.02,
    cbar_tick_length: float = 4.0,
    cbar_outline_width: float = 0.8,
    cbar_fraction: float = 0.046,
    figsize: tuple = (8, 7),
    dpi: int = 150,
    default_font: str = 'Nimbus Sans',
    save_path: str = None,
    show: bool = True,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    mat = df.to_numpy(dtype=float).copy()
    row_names = list(df.index)
    col_names = list(df.columns)
    n_rows, n_cols = mat.shape

    mask = np.zeros_like(mat, dtype=bool)
    if lower_triangle_only:
        mask |= np.triu(np.ones_like(mat, dtype=bool), k=1)
    if exclude_diagonal:
        if n_rows == n_cols:
            mask |= np.eye(n_rows, dtype=bool)

    visible = ~mask
    masked_mat = np.ma.array(mat, mask=mask)

    if color_list is not None:
        cmap_obj = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=cmap_n_bins)
    elif cmap is not None:
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = plt.get_cmap('viridis')

    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(color=(0, 0, 0, 0))

    if vmin is None:
        vmin = np.nanmin(mat[visible]) if visible.any() else np.nanmin(mat)
    if vmax is None:
        vmax = np.nanmax(mat[visible]) if visible.any() else np.nanmax(mat)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    norm = Normalize(vmin=vmin, vmax=vmax)

    if rasterize_cells:
        im = ax.imshow(
            masked_mat,
            cmap=cmap_obj,
            norm=norm,
            aspect='equal',
            interpolation='none',
            zorder=2,
        )
    else:
        rects = []
        rect_colors = []
        for i in range(n_rows):
            for j in range(n_cols):
                if not visible[i, j]:
                    continue
                val = mat[i, j]
                if np.isnan(val):
                    continue
                rects.append(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0))
                rect_colors.append(cmap_obj(norm(val)))

        cell_collection = PatchCollection(
            rects,
            facecolors=rect_colors,
            edgecolors='none',
            antialiased=False,
            zorder=2,
        )
        ax.add_collection(cell_collection)

        im = ScalarMappable(norm=norm, cmap=cmap_obj)
        im.set_array([])

    if cell_outline_width > 0:
        segments = []
        for i in range(n_rows):
            for j in range(n_cols):
                if not visible[i, j]:
                    continue
                x0, x1 = j - 0.5, j + 0.5
                y0, y1 = i - 0.5, i + 0.5
                segments.append([(x0, y0), (x1, y0)])
                segments.append([(x0, y1), (x1, y1)])
                segments.append([(x0, y0), (x0, y1)])
                segments.append([(x1, y0), (x1, y1)])

        edge_collection = LineCollection(
            segments,
            colors=cell_outline_color,
            linewidths=cell_outline_width,
            zorder=3,
            capstyle='projecting',
            joinstyle='miter',
        )
        ax.add_collection(edge_collection)

    if annot:
        norm_range = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        for i in range(n_rows):
            for j in range(n_cols):
                if mask[i, j]:
                    continue
                val = mat[i, j]
                if np.isnan(val):
                    continue

                if annot_color is not None:
                    txt_color = annot_color
                elif annot_auto_contrast:
                    norm_val = np.clip((val - vmin) / norm_range, 0.0, 1.0)
                    r, g, b, _ = cmap_obj(norm_val)
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    txt_color = annot_color_light if luminance < annot_luminance_threshold else annot_color_dark
                else:
                    txt_color = annot_color_dark

                ax.text(
                    j, i,
                    f"{val:.{annot_decimals}f}",
                    ha='center', va='center',
                    fontsize=annot_fontsize,
                    fontfamily=default_font,
                    color=txt_color,
                    zorder=4,
                )

    x_display = [x_col_labels.get(c, c) if x_col_labels else c for c in col_names]
    y_display = [y_col_labels.get(r, r) if y_col_labels else r for r in row_names]

    x_tick_positions = list(range(n_cols))
    y_tick_positions = list(range(n_rows))

    if lower_triangle_only:
        x_tick_positions = x_tick_positions[:-1]
        x_display = x_display[:-1]
        y_tick_positions = y_tick_positions[1:]
        y_display = y_display[1:]

    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)

    if xtick_rotation != 0:
        ha = xtick_ha if xtick_ha is not None else 'right'
        va = xtick_va if xtick_va is not None else 'center'
        ax.set_xticklabels(
            x_display,
            rotation=xtick_rotation,
            ha=ha,
            va=va,
            rotation_mode='anchor',
            fontsize=xtick_labelsize,
            fontfamily=default_font,
        )
    else:
        ha = xtick_ha if xtick_ha is not None else 'center'
        va = xtick_va if xtick_va is not None else 'top'
        ax.set_xticklabels(
            x_display,
            ha=ha,
            va=va,
            fontsize=xtick_labelsize,
            fontfamily=default_font,
        )

    if xtick_x_offset != 0 or xtick_y_offset != 0:
        offset = matplotlib.transforms.ScaledTranslation(
            xtick_x_offset / 72.0,
            xtick_y_offset / 72.0,
            fig.dpi_scale_trans,
        )
        for lbl in ax.get_xticklabels():
            lbl.set_transform(lbl.get_transform() + offset)

    ax.set_yticklabels(y_display, ha='right', va='center', fontsize=ytick_labelsize, fontfamily=default_font)

    ax.tick_params(
        axis='x',
        which='both',
        bottom=show_xticks,
        top=False,
        labelbottom=True,
        length=xtick_length if show_xticks else 0,
        width=xtick_width,
        direction=xtick_direction,
        color=tick_color,
    )
    ax.tick_params(
        axis='y',
        which='both',
        left=show_yticks,
        right=False,
        labelleft=True,
        length=ytick_length if show_yticks else 0,
        width=ytick_width,
        direction=ytick_direction,
        color=tick_color,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad)

    if title is not None:
        title_weight = 'bold' if title_bold else 'normal'
        ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font, fontweight=title_weight)

    if show_colorbar:
        divider_box = ax.get_position()
        cbar_height = divider_box.height * cbar_shrink
        cbar_bottom = divider_box.y0 + (divider_box.height - cbar_height) / 2.0
        cbar_width = divider_box.height / cbar_aspect
        cbar_left = divider_box.x1 + cbar_pad

        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.outline.set_linewidth(cbar_outline_width)
        cbar.ax.tick_params(length=cbar_tick_length, labelsize=cbar_labelsize)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontfamily(default_font)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=cbar_label_fontsize, fontfamily=default_font)

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return ax

def plot_cluster_holdout_ap_heatmap(
    df: pd.DataFrame,
    allele: str,
    *,
    ax: plt.Axes = None,
    allele_specific_clusters: dict = None,
    model_group_labels: list = None,
    full_label: str = "Full",
    null_label: str = "Null",
    pepcl_lambda: str = "1eN4",
    include_missing_columns: bool = False,
    sort_rows: bool = True,
    cmap: str = None,
    color_list: list = None,
    vmin: float = None,
    vmax: float = None,
    cmap_n_bins: int = 256,
    title: str = None,
    xlabel: str = None,
    ylabel: str = "Cluster",
    title_fontsize: float = 14,
    title_bold: bool = False,
    xlabel_fontsize: float = 12,
    ylabel_fontsize: float = 12,
    xlabel_pad: float = 4.0,
    ylabel_pad: float = 4.0,
    xtick_labelsize: float = 10,
    ytick_labelsize: float = 10,
    full_null_xtick_rotation: float = 90,
    full_null_xtick_ha: str = "right",
    full_null_xtick_va: str = "center",
    full_null_xtick_x_offset: float = -3.0,
    full_null_xtick_y_offset: float = 0.0,
    cluster_xtick_rotation: float = 0,
    cluster_xtick_ha: str = "center",
    cluster_xtick_va: str = "top",
    cluster_xtick_x_offset: float = 0.0,
    cluster_xtick_y_offset: float = 0.0,
    show_xticks: bool = False,
    show_yticks: bool = False,
    xtick_length: float = 3.5,
    ytick_length: float = 3.5,
    xtick_width: float = 0.8,
    ytick_width: float = 0.8,
    xtick_direction: str = "out",
    ytick_direction: str = "out",
    tick_color: str = "black",
    cell_outline_color: str = "black",
    cell_outline_width: float = 0.5,
    group_line_color: str = "black",
    group_line_width: float = 0.8,
    group_line_y: float = -0.095,
    group_label_y: float = -0.165,
    group_line_to_xtick_gap: float = None,
    group_label_to_line_gap: float = None,
    group_label_fontsize: float = 10,
    group_label_pad_cols: float = 0.15,
    group_separator_width: float = 0.0,
    group_separator_color: str = "black",
    annot: bool = False,
    annot_fontsize: float = 9,
    annot_decimals: int = 2,
    annot_color: str = None,
    annot_color_light: str = "white",
    annot_color_dark: str = "black",
    annot_auto_contrast: bool = True,
    annot_luminance_threshold: float = 0.55,
    show_colorbar: bool = True,
    cbar_label: str = "AP",
    cbar_label_fontsize: float = 12,
    cbar_labelsize: float = 10,
    cbar_shrink: float = 1.0,
    cbar_aspect: float = 20,
    cbar_pad: float = 0.02,
    cbar_tick_length: float = 4.0,
    cbar_tick_width: float = 0.8,
    cbar_outline_width: float = 0.8,
    cbar_fraction: float = 0.046,
    aspect: str = "equal",
    figsize: tuple = (8, 4),
    dpi: int = 150,
    default_font: str = "Nimbus Sans",
    save_path: str = None,
    show: bool = True,
):
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Nimbus Sans", "Nimbus Sans L", "Arial", "Helvetica"]
    matplotlib.rcParams["pdf.fonttype"] = 42

    allele_fmt = allele.lower()
    allele_key = allele.upper()

    if model_group_labels is None:
        model_group_labels = ["MHCPrime", "FT", "PepCL"]

    if allele_specific_clusters is not None and allele_key in allele_specific_clusters:
        clusters = [int(x) for x in allele_specific_clusters[allele_key]]
    else:
        clusters = []
        patterns = [
            rf"^mhcprime_holdout_{re.escape(allele_fmt)}_(\d+)$",
            rf"^model_ft_{re.escape(allele_fmt)}_(\d+)$",
            rf"^model_pepcl_{re.escape(allele_fmt)}_(\d+)_lambda_{re.escape(pepcl_lambda)}$",
        ]
        for c in df.columns:
            for p in patterns:
                m = re.match(p, str(c))
                if m is not None:
                    clusters.append(int(m.group(1)))
        clusters = sorted(set(clusters))

    full_col = f"mhcprime_holdout_{allele_fmt}_full"
    null_col = f"mhcprime_holdout_{allele_fmt}_null"
    mhcprime_cols = [f"mhcprime_holdout_{allele_fmt}_{lid}" for lid in clusters]
    ft_cols = [f"model_ft_{allele_fmt}_{lid}" for lid in clusters]
    pepcl_cols = [f"model_pepcl_{allele_fmt}_{lid}_lambda_{pepcl_lambda}" for lid in clusters]

    ordered_cols_raw = [full_col, null_col] + mhcprime_cols + ft_cols + pepcl_cols

    if include_missing_columns:
        plot_df = df.copy()
        for c in ordered_cols_raw:
            if c not in plot_df.columns:
                plot_df[c] = np.nan
        ordered_cols = ordered_cols_raw
    else:
        ordered_cols = [c for c in ordered_cols_raw if c in df.columns]
        plot_df = df.copy()

    plot_df = plot_df.loc[:, ordered_cols].copy()

    try:
        row_ints = [int(float(x)) for x in plot_df.index]
    except Exception:
        row_ints = [int(x) for x in plot_df.index.astype(int)]

    plot_df.index = row_ints

    if sort_rows:
        plot_df = plot_df.sort_index()

    col_labels = []
    col_is_full_null = []

    for c in ordered_cols:
        if c == full_col:
            col_labels.append(full_label)
            col_is_full_null.append(True)
        elif c == null_col:
            col_labels.append(null_label)
            col_is_full_null.append(True)
        else:
            m = re.search(rf"_{re.escape(allele_fmt)}_(\d+)", c)
            col_labels.append(str(int(m.group(1))) if m is not None else str(c))
            col_is_full_null.append(False)

    group_spans = []
    grouped_cols = [mhcprime_cols, ft_cols, pepcl_cols]
    leading_cols = [c for c in [full_col, null_col] if c in ordered_cols]

    cursor = len(leading_cols)
    for label, cols in zip(model_group_labels, grouped_cols):
        present = [c for c in cols if c in ordered_cols]
        if len(present) > 0:
            start = cursor
            end = cursor + len(present) - 1
            group_spans.append((start, end, label))
            cursor += len(present)

    mat = plot_df.to_numpy(dtype=float)
    n_rows, n_cols = mat.shape
    visible = ~np.isnan(mat)
    masked_mat = np.ma.array(mat, mask=np.isnan(mat))

    if color_list is not None:
        cmap_obj = LinearSegmentedColormap.from_list("custom_cmap", color_list, N=cmap_n_bins)
    elif cmap is not None:
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = plt.get_cmap("viridis")

    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(color=(0, 0, 0, 0))

    if vmin is None:
        vmin = np.nanmin(mat) if np.any(visible) else 0.0
    if vmax is None:
        vmax = np.nanmax(mat) if np.any(visible) else 1.0

    if group_line_to_xtick_gap is not None:
        group_line_y = -abs(group_line_to_xtick_gap)
    if group_label_to_line_gap is not None:
        group_label_y = group_line_y - abs(group_label_to_line_gap)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        masked_mat,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        zorder=2,
    )

    if cell_outline_width > 0:
        segments = []
        for i in range(n_rows):
            for j in range(n_cols):
                if not visible[i, j]:
                    continue
                x0, x1 = j - 0.5, j + 0.5
                y0, y1 = i - 0.5, i + 0.5
                segments.append([(x0, y0), (x1, y0)])
                segments.append([(x0, y1), (x1, y1)])
                segments.append([(x0, y0), (x0, y1)])
                segments.append([(x1, y0), (x1, y1)])

        edge_collection = LineCollection(
            segments,
            colors=cell_outline_color,
            linewidths=cell_outline_width,
            zorder=3,
            capstyle="projecting",
            joinstyle="miter",
        )
        ax.add_collection(edge_collection)

    if group_separator_width > 0:
        sep_segments = []
        boundaries = []
        if len(leading_cols) > 0 and len(group_spans) > 0:
            boundaries.append(group_spans[0][0] - 0.5)
        for start, end, label in group_spans[:-1]:
            boundaries.append(end + 0.5)
        for x in boundaries:
            sep_segments.append([(x, -0.5), (x, n_rows - 0.5)])
        sep_collection = LineCollection(
            sep_segments,
            colors=group_separator_color,
            linewidths=group_separator_width,
            zorder=5,
            capstyle="butt",
        )
        ax.add_collection(sep_collection)

    if annot:
        norm_range = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if np.isnan(val):
                    continue

                if annot_color is not None:
                    txt_color = annot_color
                elif annot_auto_contrast:
                    norm_val = np.clip((val - vmin) / norm_range, 0.0, 1.0)
                    r, g, b, _ = cmap_obj(norm_val)
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    txt_color = annot_color_light if luminance < annot_luminance_threshold else annot_color_dark
                else:
                    txt_color = annot_color_dark

                ax.text(
                    j,
                    i,
                    f"{val:.{annot_decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    fontfamily=default_font,
                    color=txt_color,
                    zorder=4,
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    ax.set_xticklabels(col_labels, fontsize=xtick_labelsize, fontfamily=default_font)

    for lbl, is_full_null in zip(ax.get_xticklabels(), col_is_full_null):
        if is_full_null:
            lbl.set_rotation(full_null_xtick_rotation)
            lbl.set_horizontalalignment(full_null_xtick_ha)
            lbl.set_verticalalignment(full_null_xtick_va)
            lbl.set_rotation_mode("anchor")
            if full_null_xtick_x_offset != 0 or full_null_xtick_y_offset != 0:
                offset = matplotlib.transforms.ScaledTranslation(
                    full_null_xtick_x_offset / 72.0,
                    full_null_xtick_y_offset / 72.0,
                    fig.dpi_scale_trans,
                )
                lbl.set_transform(lbl.get_transform() + offset)
        else:
            lbl.set_rotation(cluster_xtick_rotation)
            lbl.set_horizontalalignment(cluster_xtick_ha)
            lbl.set_verticalalignment(cluster_xtick_va)
            if cluster_xtick_rotation != 0:
                lbl.set_rotation_mode("anchor")
            if cluster_xtick_x_offset != 0 or cluster_xtick_y_offset != 0:
                offset = matplotlib.transforms.ScaledTranslation(
                    cluster_xtick_x_offset / 72.0,
                    cluster_xtick_y_offset / 72.0,
                    fig.dpi_scale_trans,
                )
                lbl.set_transform(lbl.get_transform() + offset)

    ax.set_yticklabels(
        [str(int(x)) for x in plot_df.index],
        ha="right",
        va="center",
        fontsize=ytick_labelsize,
        fontfamily=default_font,
    )

    ax.tick_params(
        axis="x",
        which="both",
        bottom=show_xticks,
        top=False,
        labelbottom=True,
        length=xtick_length if show_xticks else 0,
        width=xtick_width,
        direction=xtick_direction,
        color=tick_color,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=show_yticks,
        right=False,
        labelleft=True,
        length=ytick_length if show_yticks else 0,
        width=ytick_width,
        direction=ytick_direction,
        color=tick_color,
    )

    xaxis_transform = ax.get_xaxis_transform()

    for start, end, label in group_spans:
        x0 = start - 0.5 + group_label_pad_cols
        x1 = end + 0.5 - group_label_pad_cols
        xc = (start + end) / 2.0
        ax.plot(
            [x0, x1],
            [group_line_y, group_line_y],
            transform=xaxis_transform,
            color=group_line_color,
            linewidth=group_line_width,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            xc,
            group_label_y,
            label,
            transform=xaxis_transform,
            ha="center",
            va="top",
            fontsize=group_label_fontsize,
            fontfamily=default_font,
            clip_on=False,
            zorder=6,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontfamily=default_font, labelpad=xlabel_pad)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontfamily=default_font, labelpad=ylabel_pad)

    if title is not None:
        title_weight = "bold" if title_bold else "normal"
        ax.set_title(title, fontsize=title_fontsize, fontfamily=default_font, fontweight=title_weight)

    if show_colorbar:
        divider_box = ax.get_position()
        cbar_height = divider_box.height * cbar_shrink
        cbar_bottom = divider_box.y0 + (divider_box.height - cbar_height) / 2.0
        cbar_width = divider_box.height / cbar_aspect
        cbar_left = divider_box.x1 + cbar_pad

        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.outline.set_linewidth(cbar_outline_width)
        cbar.ax.tick_params(
            length=cbar_tick_length,
            width=cbar_tick_width,
            labelsize=cbar_labelsize,
        )
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontfamily(default_font)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=cbar_label_fontsize, fontfamily=default_font)

    if save_path:
        fig.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return ax

def format_hla_allele(allele: str) -> str:
    allele = str(allele).strip().upper().replace("HLA-", "").replace("*", "").replace(":", "")
    gene = allele[0]
    digits = allele[1:]

    if len(digits) < 4:
        raise ValueError(f"Expected allele like A0201, got {allele}")

    return f"HLA-{gene}*{digits[:2]}:{digits[2:4]}"


def plot_cluster_holdout_ap_heatmap_grid_manual(
    ap_matrix_dict: dict,
    *,
    allele_specific_clusters: dict,
    allele_order: list = None,
    ncols: int = 2,
    panel_width: float = 10.0,
    panel_height: float = 2.0,
    left_margin: float = 0.45,
    right_margin: float = 0.25,
    top_margin: float = 0.25,
    bottom_margin: float = 0.30,
    col_gap: float = 0.35,
    row_gap: float = 0.55,
    dpi: int = 300,
    title_func=None,
    model_group_labels: list = None,
    cmap: str = "coolwarm_r",
    color_list: list = None,
    cmap_n_bins: int = 256,
    vmin: float = 0.0,
    vmax: float = 1.0,
    heatmap_kwargs: dict = None,
    row_cbar: bool = True,
    cbar_label: str = "AP",
    cbar_label_fontsize: float = 6,
    cbar_labelsize: float = 6,
    cbar_tick_length: float = 2.0,
    cbar_tick_width: float = 0.35,
    cbar_outline_width: float = 0.35,
    cbar_width: float = 0.055,
    cbar_pad: float = 0.08,
    cbar_height_frac: float = 1.0,
    default_font: str = "Nimbus Sans",
    save_path: str = None,
    show: bool = True,
):
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Nimbus Sans", "Nimbus Sans L", "Arial", "Helvetica"]
    matplotlib.rcParams["pdf.fonttype"] = 42

    if allele_order is None:
        allele_order = list(ap_matrix_dict.keys())

    if title_func is None:
        title_func = format_hla_allele

    if model_group_labels is None:
        model_group_labels = ["Cluster Holdout", "FT", "PepCL"]

    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    n_panels = len(allele_order)
    nrows = math.ceil(n_panels / ncols)

    extra_cbar_width = cbar_pad + cbar_width if row_cbar else 0.0

    fig_width = (
        left_margin
        + ncols * panel_width
        + (ncols - 1) * col_gap
        + extra_cbar_width
        + right_margin
    )
    fig_height = (
        top_margin
        + nrows * panel_height
        + (nrows - 1) * row_gap
        + bottom_margin
    )

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    if color_list is not None:
        cmap_obj = LinearSegmentedColormap.from_list("custom_cmap_grid", color_list, N=cmap_n_bins)
    else:
        cmap_obj = plt.get_cmap(cmap)

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    axes = np.empty((nrows, ncols), dtype=object)

    for idx, allele in enumerate(allele_order):
        r = idx // ncols
        c = idx % ncols

        x_in = left_margin + c * (panel_width + col_gap)
        y_in = fig_height - top_margin - (r + 1) * panel_height - r * row_gap

        ax = fig.add_axes([
            x_in / fig_width,
            y_in / fig_height,
            panel_width / fig_width,
            panel_height / fig_height,
        ])

        axes[r, c] = ax

        plot_cluster_holdout_ap_heatmap(
            ap_matrix_dict[allele],
            allele=allele,
            ax=ax,
            allele_specific_clusters=allele_specific_clusters,
            model_group_labels=model_group_labels,
            cmap=cmap,
            color_list=color_list,
            cmap_n_bins=cmap_n_bins,
            vmin=vmin,
            vmax=vmax,
            title=title_func(allele),
            show_colorbar=False,
            show=False,
            default_font=default_font,
            **heatmap_kwargs,
        )

    for idx in range(n_panels, nrows * ncols):
        r = idx // ncols
        c = idx % ncols

        x_in = left_margin + c * (panel_width + col_gap)
        y_in = fig_height - top_margin - (r + 1) * panel_height - r * row_gap

        ax = fig.add_axes([
            x_in / fig_width,
            y_in / fig_height,
            panel_width / fig_width,
            panel_height / fig_height,
        ])
        ax.axis("off")
        axes[r, c] = ax

    if row_cbar:
        for r in range(nrows):
            row_indices = [r * ncols + c for c in range(ncols)]
            row_indices = [i for i in row_indices if i < n_panels]
            if len(row_indices) == 0:
                continue

            rightmost_c = row_indices[-1] % ncols
            x_in = left_margin + rightmost_c * (panel_width + col_gap) + panel_width + cbar_pad
            base_y_in = fig_height - top_margin - (r + 1) * panel_height - r * row_gap

            cbar_height = panel_height * cbar_height_frac
            y_in = base_y_in + (panel_height - cbar_height) / 2.0

            cax = fig.add_axes([
                x_in / fig_width,
                y_in / fig_height,
                cbar_width / fig_width,
                cbar_height / fig_height,
            ])

            cbar = fig.colorbar(sm, cax=cax)
            cbar.outline.set_linewidth(cbar_outline_width)
            cbar.ax.tick_params(
                length=cbar_tick_length,
                width=cbar_tick_width,
                labelsize=cbar_labelsize,
            )
            for lbl in cbar.ax.get_yticklabels():
                lbl.set_fontfamily(default_font)
            if cbar_label is not None:
                cbar.set_label(cbar_label, fontsize=cbar_label_fontsize, fontfamily=default_font)

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, axes

def paired_boxplot_vs_ref_styled(
    df,
    ref_col,
    test_cols,
    colors=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font='Nimbus Sans',
    box_width=0.6,
    group_spacing=1.0,
    box_spacing=0.05,
    box_outline_color='black',
    box_outline_width=1.0,
    box_flier_marker='o',
    box_flier_size=3,
    box_flier_color=None,
    box_flier_alpha=0.6,
    box_flier_edge_width=0,
    box_median_color='black',
    box_median_width=1.5,
    box_whisker_style='-',
    box_whisker_width=1.0,
    box_notch=False,
    box_patch_alpha=0.9,
    showfliers=False,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    legend_mode='integrated',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    xtick_rotation=0,
    save_path=None,
):
    """
    Paired boxplot comparing a reference column against one or more test columns.

    Each group on the x-axis corresponds to one test_col, showing two side-by-side
    boxes: the reference distribution (filtered to rows where both ref and test are
    non-null) and the test distribution.
    """
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (10, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    default_color_cycle = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    if colors is None:
        ref_color = default_color_cycle[0]
        test_color_map = {c: default_color_cycle[(i + 1) % len(default_color_cycle)]
                         for i, c in enumerate(test_cols)}
    elif isinstance(colors, list):
        ref_color = colors[0]
        test_color_map = {c: colors[1] for c in test_cols}
    else:
        ref_color = colors.get(ref_col, default_color_cycle[0])
        test_color_map = {c: colors.get(c, default_color_cycle[(i + 1) % len(default_color_cycle)])
                          for i, c in enumerate(test_cols)}

    groups_data = []
    for c in test_cols:
        mask = df[ref_col].notna() & df[c].notna()
        vals_ref  = df.loc[mask, ref_col].values
        vals_test = df.loc[mask, c].values
        groups_data.append((c, vals_ref, vals_test))

    n_groups = len(groups_data)

    n_boxes = 2
    group_width = n_boxes * box_width + (n_boxes - 1) * box_spacing
    group_centers = np.arange(n_groups) * (group_width + group_spacing)

    fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])

    for gi, (test_col, vals_ref, vals_test) in enumerate(groups_data):
        center = group_centers[gi]
        total_w = n_boxes * box_width + (n_boxes - 1) * box_spacing
        start   = center - total_w / 2 + box_width / 2

        pos_ref  = start
        pos_test = start + box_width + box_spacing

        for pos, vals, color in [
            (pos_ref,  vals_ref,  ref_color),
            (pos_test, vals_test, test_color_map[test_col]),
        ]:
            if len(vals) == 0:
                continue
            flier_c = box_flier_color if box_flier_color is not None else color
            ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_width * 0.9,
                patch_artist=True,
                notch=box_notch,
                manage_ticks=False,
                showfliers=showfliers,
                zorder=2,
                flierprops=dict(
                    marker=box_flier_marker,
                    markersize=box_flier_size,
                    markerfacecolor=flier_c,
                    markeredgecolor=flier_c,
                    markeredgewidth=box_flier_edge_width,
                    alpha=box_flier_alpha,
                ),
                medianprops=dict(color=box_median_color, linewidth=box_median_width),
                whiskerprops=dict(color=box_outline_color, linewidth=box_whisker_width,
                                  linestyle=box_whisker_style),
                capprops=dict(color='none', linewidth=0),
                boxprops=dict(facecolor=color, edgecolor=box_outline_color,
                              linewidth=box_outline_width, alpha=box_patch_alpha),
            )

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None
        ax.axhline(y=hline_y, color=hline_color, linestyle=hline_style,
                   linewidth=hline_width, label=line_label, zorder=3)

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            x_range = group_centers[-1] - group_centers[0] if n_groups > 1 else 1
            text_x = group_centers[0] + x_range * hline_text_x_offset
            text_y = (hline_y + hline_text_offset if hline_text_position == 'above'
                      else hline_y - hline_text_offset)
            va = 'bottom' if hline_text_position == 'above' else 'top'
            ax.text(text_x, text_y, hline_text, color=text_color,
                    fontsize=hline_text_fontsize, fontfamily=default_font,
                    va=va, ha='left', zorder=3)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([c for c, _, _ in groups_data],
                       rotation=xtick_rotation, ha='center')

    ax.set_xlabel(params['xlabel'], fontsize=params['xlabel_fontsize'],
                  fontfamily=default_font, labelpad=params['xlabel_pad'])
    ax.set_ylabel(params['ylabel'], fontsize=params['ylabel_fontsize'],
                  fontfamily=default_font, labelpad=params['ylabel_pad'])

    title_weight = 'bold' if params['title_bold'] else 'normal'
    ax.set_title(params['title'], fontsize=params['title_fontsize'],
                 fontfamily=default_font, fontweight=title_weight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(axis='both', which='major',
                   length=params['tick_length'], width=params['axis_linewidth'],
                   direction='out', pad=params['tick_length'] + 2)
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, axis='y', zorder=0)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []

    handles.append(mpatches.Patch(facecolor=ref_color, edgecolor=box_outline_color,
                                  linewidth=box_outline_width, alpha=box_patch_alpha))
    labels_list.append(ref_col)

    seen_colors = {ref_color}
    for c in test_cols:
        tc = test_color_map[c]
        if tc not in seen_colors:
            handles.append(mpatches.Patch(facecolor=tc, edgecolor=box_outline_color,
                                          linewidth=box_outline_width, alpha=box_patch_alpha))
            labels_list.append(c)
            seen_colors.add(tc)
        elif isinstance(colors, dict):
            handles.append(mpatches.Patch(facecolor=tc, edgecolor=box_outline_color,
                                          linewidth=box_outline_width, alpha=box_patch_alpha))
            labels_list.append(c)

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(Line2D([0], [0], color=hline_color, linestyle=hline_style,
                              linewidth=hline_width * params['legend_handle_size']))
        labels_list.append(hline_label)

    if legend_mode == 'integrated' and handles:
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        leg = fig.legend(handles, labels_list,
                         loc=legend_loc, bbox_to_anchor=bbox,
                         ncol=legend_ncol, frameon=False,
                         fontsize=params['legend_fontsize'])
        for text in leg.get_texts():
            text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(f"{save_path}{suffix}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}{suffix}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

    if legend_mode == 'separate' and handles:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(params['legend_handle_size'] * 2,
                     n_items * 0.4 * params['legend_handle_size']),
            dpi=params['dpi']
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        leg = ax_legend.legend(handles, labels_list,
                               loc='center', frameon=False,
                               fontsize=params['legend_fontsize'],
                               ncol=legend_ncol)
        for text in leg.get_texts():
            text.set_fontfamily(default_font)
        plt.tight_layout()

        if save_path:
            fig_legend.savefig(f"{save_path}_legend.pdf", format='pdf',
                               bbox_inches='tight', dpi=300)
            fig_legend.savefig(f"{save_path}_legend.png", format='png',
                               bbox_inches='tight', dpi=300)
        plt.show()

def plot_horizontal_legend(
    legend_dict,
    swatch_shape='square',
    swatch_size=12,
    text_size=12,
    item_spacing=30,
    swatch_text_gap=6,
    swatch_outline_color='black',
    swatch_outline_width=0.8,
    figsize=None,
    dpi=300,
    default_font='Nimbus Sans',
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica'
    ]
    matplotlib.rcParams['pdf.fonttype'] = 42

    n_items = len(legend_dict)

    handles = []
    labels = list(legend_dict.keys())
    for label, value in legend_dict.items():
        if isinstance(value, tuple):
            # (color, linewidth) -> render as a line swatch
            color, linewidth = value
            handle = Line2D(
                [0], [0],
                color=color,
                linewidth=linewidth,
                linestyle='-',
            )
        elif swatch_shape == 'circle':
            # filled circle marker, no outline
            handle = Line2D(
                [0], [0],
                marker='o',
                linestyle='none',
                markerfacecolor=value,
                markeredgecolor='none',
                markeredgewidth=0,
                markersize=swatch_size,
            )
        else:
            # square swatch with outline (default)
            handle = mpatches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle='square,pad=0',
                facecolor=value,
                edgecolor=swatch_outline_color,
                linewidth=swatch_outline_width,
            )
        handles.append(handle)

    if figsize is None:
        avg_label_width_pt = max(len(lbl) for lbl in labels) * text_size * 0.55
        item_width_pt = swatch_size + swatch_text_gap + avg_label_width_pt
        total_width_pt = n_items * item_width_pt + (n_items - 1) * item_spacing
        fig_width_in = (total_width_pt + 40) / 72.0  # padding
        fig_height_in = (swatch_size + 16) / 72.0
        figsize = (max(fig_width_in, 1.0), max(fig_height_in, 0.3))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')

    legend = ax.legend(
        handles,
        labels,
        loc='center',
        ncol=n_items,
        frameon=False,
        fontsize=text_size,
        handleheight=swatch_size / 10.0,
        handlelength=swatch_size / 10.0,
        handletextpad=swatch_text_gap / 10.0,
        columnspacing=item_spacing / 10.0,
        borderpad=0,
    )

    for text in legend.get_texts():
        text.set_fontfamily(default_font)

    plt.tight_layout(pad=0.2)

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=dpi)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=dpi)

    plt.show()

def plot_style_legend(
    line_styles,
    labels=None,
    default_font='Nimbus Sans',
    fontsize=12,
    ncol=1,
    handlelength=2.5,
    figsize=None,
    dpi=100,
    save_path=None,
):
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    handles = []
    auto_labels = []
    for key, sub in line_styles.items():
        for subkey in ('cluster', 'prior'):
            if subkey not in sub:
                continue
            s = sub[subkey]
            handles.append(Line2D(
                [0], [0],
                color=s.get('color', 'black'),
                linewidth=s.get('linewidth', 1.0),
                linestyle=s.get('linestyle', '-'),
                marker=s.get('marker', None),
                alpha=s.get('alpha', 1.0),
            ))
            auto_labels.append(f"{key} {subkey}")

    if labels is None:
        labels = auto_labels

    if figsize is None:
        figsize = (3, len(handles) * 0.4)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend = ax.legend(handles, labels, loc='center', frameon=False,
                       fontsize=fontsize, ncol=ncol, handlelength=handlelength)
    for text in legend.get_texts():
        text.set_fontfamily(default_font)

    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_circle_marker_legend(color_dict, ncol=4, figsize=(10, 2), dpi=150,
                fontsize=8, marker='o', marker_size=8,
                default_font='Nimbus Sans', save_path=None,
                row_spacing=0.8, extra_labels=None):

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    handles = []
    labels = []

    if extra_labels is not None:
        handles.append(Line2D([0], [0], color='black', marker=marker, linewidth=0,
                               markersize=marker_size))
        labels.append(extra_labels[0])

        n_pad_mid = ncol // 2 - 1
        for _ in range(n_pad_mid):
            handles.append(Line2D([0], [0], linewidth=0, markersize=0, alpha=0))
            labels.append('')

        handles.append(Line2D([0], [0], color='black', marker=marker, linewidth=0,
                               markersize=marker_size))
        labels.append(extra_labels[1])

        n_pad_end = ncol - ncol // 2 - 1
        for _ in range(n_pad_end):
            handles.append(Line2D([0], [0], linewidth=0, markersize=0, alpha=0))
            labels.append('')

    for name, color in color_dict.items():
        handles.append(Line2D([0], [0], color=color, marker=marker, linewidth=0,
                               markersize=marker_size, label=name))
        labels.append(name)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')

    leg = ax.legend(handles=handles, labels=labels,
                    loc='center', frameon=False, fontsize=fontsize,
                    ncol=ncol, labelspacing=row_spacing)
    for text in leg.get_texts():
        text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_legend(color_dict, ncol=4, figsize=(10, 2), dpi=150,
                fontsize=8, marker='o', marker_size=8,
                default_font='Nimbus Sans', save_path=None,
                row_spacing=0.8, extra_labels=None):

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Nimbus Sans L', 'Arial', 'Helvetica']
    matplotlib.rcParams['pdf.fonttype'] = 42

    handles = []
    labels = []

    if extra_labels is not None:
        # first entry at column 0
        handles.append(Line2D([0], [0], color='black', marker=marker, linewidth=0,
                               markersize=marker_size))
        labels.append(extra_labels[0])

        # pad to middle of row
        n_pad_mid = ncol // 2 - 1
        for _ in range(n_pad_mid):
            handles.append(Line2D([0], [0], linewidth=0, markersize=0, alpha=0))
            labels.append('')

        # second entry at column ncol//2
        handles.append(Line2D([0], [0], color='black', marker=marker, linewidth=0,
                               markersize=marker_size))
        labels.append(extra_labels[1])

        # pad rest of first row
        n_pad_end = ncol - ncol // 2 - 1
        for _ in range(n_pad_end):
            handles.append(Line2D([0], [0], linewidth=0, markersize=0, alpha=0))
            labels.append('')

    for name, color in color_dict.items():
        handles.append(Line2D([0], [0], color=color, marker=marker, linewidth=0,
                               markersize=marker_size, label=name))
        labels.append(name)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')

    leg = ax.legend(handles=handles, labels=labels,
                    loc='center', frameon=False, fontsize=fontsize,
                    ncol=ncol, labelspacing=row_spacing)
    for text in leg.get_texts():
        text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_split_ap_heatmaps(
    ap_matrix,
    allele_holdout_splits,
    splits=(1, 2, 3, 4, 5),
    base_tpl="mhcprime_base_aho_split_{s}",
    ft_tpl="mhcprime_ft_aho_split_{s}",
    pepcl_tpl="mhcprime_pepcl_aho_split_{s}",
    ref_col="mhcprime",
    method_labels=("Base", "FT", "PepCL"),
    include_ref=False,
    delta=False,
    delta_ref="base",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    per_plot_scale=False,
    unify_rows=False,
    order_panels_by_block=True,
    label_split_numbers=False,
    renumber_splits=True,
    split_into_two=False,
    split_gene_sets=False,
    bottom_vmin=None,
    bottom_vmax=None,
    gene_set_gap=1,
    edge_color=None,
    edge_width=0.5,
    figsize=(20, 12),
    dpi=150,
    default_font="Nimbus Sans",
    annotate=False,
    annotation_fontsize=None,
    save_path=None,
    save_pdf=True,
    save_png=True,
):
    import re
    from itertools import combinations
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    # Publication / Illustrator-friendly export settings.
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        default_font,
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42

    def _fmt(a):
        a=str(a)
        if "*" in a or ":" in a:
            return a
        m=re.match(r"^([A-Za-z]+)(\d+)$",a)
        if not m:
            return a
        loc,dig=m.groups()
        if len(dig)<4:
            return a
        return f"{loc}*{dig[:2]}:{dig[2:]}"

    def _get_hold(s):
        return allele_holdout_splits[s] if s in allele_holdout_splits else allele_holdout_splits.get(str(s),[])

    all_alleles=list(ap_matrix.index)
    ap_set=set(all_alleles)
    split_list=list(splits)
    hold_all={s:set(_get_hold(s)) & ap_set for s in split_list}
    unified=unify_rows or split_into_two
    _div=isinstance(cmap,str) and any(
        k in cmap
        for k in ["coolwarm","RdBu","bwr","seismic","RdYlBu","Spectral","PiYG","PRGn"]
    )

    if split_into_two and len(split_list)>=2:
        W={(i,j):len(hold_all[i] & hold_all[j]) for i,j in combinations(split_list,2)}

        def _w(i,j):
            return W[(i,j)] if (i,j) in W else W[(j,i)]

        kA=len(split_list)//2
        first,rest=split_list[0],split_list[1:]
        best,bw=None,None

        for combo in combinations(rest,kA-1):
            A=[first]+list(combo)
            B=[s for s in split_list if s not in set(A)]

            if abs(len(A)-len(B))>1:
                continue

            w=sum(_w(i,j) for i,j in combinations(A,2))+sum(_w(i,j) for i,j in combinations(B,2))

            if bw is None or w<bw:
                bw,best=w,(A,B)

        groups=[best[0],best[1]]

    else:
        groups=[split_list]

    def _unified_order(group):
        hs={s:hold_all[s] for s in group}
        union=sorted(set().union(*hs.values())) if hs else []
        never=sorted([a for a in all_alleles if a not in set(union)])

        if len(union)>2:
            M=np.array([[1.0 if a in hs[s] else 0.0 for s in group] for a in union])
            d=pdist(M,metric="jaccard")
            Z=optimal_leaf_ordering(linkage(d,method="average"),d)
            top=[union[i] for i in leaves_list(Z)]
            memb={a:tuple(1 if a in hs[s] else 0 for s in group) for a in top}

            tidy,i=[],0
            while i<len(top):
                j=i
                while j<len(top) and memb[top[j]]==memb[top[i]]:
                    j+=1
                tidy+=sorted(top[i:j])
                i=j

            top=tidy

        else:
            top=union

        return top,top+never

    def _process(group):
        hs={s:hold_all[s] for s in group}

        if not unified:
            return dict(panels=list(group),order=None,n_hold=None,hs=hs)

        top,order=_unified_order(group)
        panels=list(group)

        if order_panels_by_block:
            posmap={a:i for i,a in enumerate(order)}
            panels=sorted(
                group,
                key=lambda s:np.mean([posmap[a] for a in hs[s]]) if hs[s] else np.inf,
            )

        return dict(panels=panels,order=order,n_hold=len(top),hs=hs)

    group_info=[_process(g) for g in groups]

    display_num,c={},1
    for gi in group_info:
        for s in gi["panels"]:
            display_num[s]=c
            c+=1

    def _disp(s):
        return display_num[s] if renumber_splits else s

    def _rows_for(gi,s):
        hs=gi["hs"]

        if unified:
            order=gi["order"]
            nh=gi["n_hold"]
            return order[:nh],order[nh:]

        hold=sorted(hs[s])
        rest=sorted([a for a in all_alleles if a not in hs[s]])
        return hold,rest

    def _matrix(order,s):
        pairs=[
            (method_labels[0],base_tpl.format(s=s)),
            (method_labels[1],ft_tpl.format(s=s)),
            (method_labels[2],pepcl_tpl.format(s=s)),
        ]

        if include_ref:
            pairs.append(("MHCPrime",ref_col))

        if delta:
            if delta_ref=="base":
                ref_vals=ap_matrix.loc[order,pairs[0][1]].astype(float).values
                keep=pairs[1:]
            else:
                ref_vals=ap_matrix.loc[order,ref_col].astype(float).values
                keep=[(lab,cc) for lab,cc in pairs if cc!=ref_col]

            data=(
                np.column_stack([
                    ap_matrix.loc[order,cc].astype(float).values-ref_vals
                    for _,cc in keep
                ])
                if order else np.empty((0,len(keep)))
            )
            labels=[lab for lab,_ in keep]

        else:
            data=(
                ap_matrix.loc[order,[cc for _,cc in pairs]].to_numpy(dtype=float)
                if order else np.empty((0,len(pairs)))
            )
            labels=[lab for lab,_ in pairs]

        return data,labels

    def _heatmap(ax,data,cmap,vmin,vmax):
        nr,nc=data.shape

        x=np.arange(nc+1)-0.5
        y=np.arange(nr+1)-0.5

        mesh=ax.pcolormesh(
            x,
            y,
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            edgecolors="none",
            linewidth=0,
            antialiased=False,
            rasterized=False,
        )

        ax.set_xlim(-0.5,nc-0.5)
        ax.set_ylim(nr-0.5,-0.5)

        return mesh

    def _annotate(ax,data,mesh):
        if not annotate or data.size==0:
            return

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value=data[i,j]

                if not np.isfinite(value):
                    continue

                rgba=mesh.cmap(mesh.norm(value))
                r,g,b=rgba[:3]

                # Relative luminance of rendered cell color.
                luminance=0.2126*r+0.7152*g+0.0722*b
                text_color="white" if luminance<0.5 else "black"

                if value>0:
                    text=f"+{value:.2f}"
                elif value<0:
                    text=f"{value:.2f}"
                else:
                    text="0.00"

                kwargs=dict(
                    ha="center",
                    va="center",
                    color=text_color,
                    fontfamily=default_font,
                    fontweight="normal",
                    zorder=4,
                )

                if annotation_fontsize is not None:
                    kwargs["fontsize"]=annotation_fontsize

                ax.text(j,i,text,**kwargs)

    def _outline(ax,nr,nc):
        if edge_color is None or nr==0 or nc==0:
            return

        # Interior separators only, never the outer boundary.
        for x in np.arange(0.5,nc-0.5,1.0):
            ax.axvline(x,color=edge_color,lw=edge_width,zorder=3)

        for y in np.arange(0.5,nr-0.5,1.0):
            ax.axhline(y,color=edge_color,lw=edge_width,zorder=3)

        ax.set_xlim(-0.5,nc-0.5)
        ax.set_ylim(nr-0.5,-0.5)

        # Keep black frame on top of grid.
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("black")
            sp.set_linewidth(1.0)
            sp.set_zorder(5)

    def _ylab(a,panels,hs):
        if label_split_numbers:
            ids=sorted(_disp(s) for s in panels if a in hs[s])
            return (f"[{','.join(map(str,ids))}] " if ids else "")+_fmt(a)

        return _fmt(a)

    def _save(fig,group_idx):
        if save_path is None:
            return

        suffix=f"_group_{group_idx+1}" if len(group_info)>1 else ""
        path=f"{save_path}{suffix}"

        if save_pdf:
            fig.savefig(
                f"{path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

        if save_png:
            fig.savefig(
                f"{path}.png",
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

    def _render_split(gi,group_idx):
        panels,hs=gi["panels"],gi["hs"]
        per,bmins,bmaxs,babs=[],[],[],[]

        for s in panels:
            hrows,rrows=_rows_for(gi,s)
            tdat,labels=_matrix(hrows,s)
            bdat,_=_matrix(rrows,s)
            per.append((s,hrows,rrows,tdat,bdat,labels))

            if bdat.size:
                bmins.append(np.nanmin(bdat))
                bmaxs.append(np.nanmax(bdat))
                babs.append(np.nanmax(np.abs(bdat)))

        if bottom_vmin is not None and bottom_vmax is not None:
            bvlo,bvhi=bottom_vmin,bottom_vmax

        elif babs:
            if _div:
                m=max(babs)
                bvlo,bvhi=-m,m
            else:
                bvlo,bvhi=min(bmins),max(bmaxs)

        else:
            bvlo,bvhi=vmin,vmax

        ncols=len(panels)
        fig=plt.figure(figsize=figsize,dpi=dpi)
        gs=fig.add_gridspec(1,ncols,left=0.08,right=0.9,wspace=0.3)

        imt=imb=None
        top_axes,bot_axes=[],[]

        for ci,(s,hrows,rrows,tdat,bdat,labels) in enumerate(per):
            sub=gs[0,ci].subgridspec(
                3,
                1,
                height_ratios=[
                    max(len(hrows),1),
                    max(gene_set_gap,1e-9),
                    max(len(rrows),1),
                ],
                hspace=0.0,
            )

            axt=fig.add_subplot(sub[0])
            axgap=fig.add_subplot(sub[1])
            axb=fig.add_subplot(sub[2])

            axgap.axis("off")

            top_axes.append(axt)
            bot_axes.append(axb)

            imt=_heatmap(axt,tdat,cmap,vmin,vmax)
            imb=_heatmap(axb,bdat,cmap,bvlo,bvhi)

            _annotate(axt,tdat,imt)
            _annotate(axb,bdat,imb)

            _outline(axt,tdat.shape[0],tdat.shape[1])
            _outline(axb,bdat.shape[0],bdat.shape[1])

            axt.set_title(f"Split {_disp(s)}")
            axt.set_xticks([])

            axb.set_xticks(range(len(labels)))
            axb.set_xticklabels(labels,rotation=45,ha="right")

            if ci==0 or not unified:
                axt.set_yticks(range(len(hrows)))
                axt.set_yticklabels(
                    [_ylab(a,panels,hs) for a in hrows],
                    fontsize=7,
                )

                axb.set_yticks(range(len(rrows)))
                axb.set_yticklabels(
                    [_ylab(a,panels,hs) for a in rrows],
                    fontsize=7,
                )

            else:
                axt.set_yticks([])
                axb.set_yticks([])

        fig.canvas.draw()

        pt=top_axes[-1].get_position()
        pb=bot_axes[-1].get_position()

        fig.colorbar(
            imt,
            cax=fig.add_axes([0.915,pt.y0,0.012,pt.height]),
        )

        fig.colorbar(
            imb,
            cax=fig.add_axes([0.915,pb.y0,0.012,pb.height]),
        )

        _save(fig,group_idx)
        plt.show()

    def _render(gi,group_idx):
        panels,hs=gi["panels"],gi["hs"]

        fig,axes=plt.subplots(
            1,
            len(panels),
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
        )

        axes=axes[0]
        last_im=None

        for ci,(ax,s) in enumerate(zip(axes,panels)):
            if unified:
                order,n_bold=gi["order"],gi["n_hold"]

            else:
                hold=sorted(hs[s])
                rest=sorted([a for a in all_alleles if a not in hs[s]])
                order,n_bold=hold+rest,len(hold)

            data,labels=_matrix(order,s)

            vlo,vhi=(
                (float(np.nanmin(data)),float(np.nanmax(data)))
                if (per_plot_scale and not unified)
                else (vmin,vmax)
            )

            im=_heatmap(ax,data,cmap,vlo,vhi)
            last_im=im

            _annotate(ax,data,im)
            _outline(ax,data.shape[0],data.shape[1])

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels,rotation=45,ha="right")
            ax.set_title(f"Split {_disp(s)}")

            if unified and ci!=0:
                ax.set_yticks([])

            else:
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels(
                    [_ylab(a,panels,hs) for a in order],
                    fontsize=7,
                )

                bold_on=(n_bold>0) and not label_split_numbers

                for i,lab in enumerate(ax.get_yticklabels()):
                    lab.set_fontweight(
                        "bold" if (bold_on and i<n_bold) else "normal"
                    )

            if not unified:
                div=make_axes_locatable(ax)
                cax=div.append_axes("right",size="6%",pad=0.08)
                fig.colorbar(im,cax=cax)

        if unified:
            fig.tight_layout(rect=[0,0,0.94,1])
            fig.colorbar(
                last_im,
                cax=fig.add_axes([0.955,0.2,0.012,0.6]),
            )

        else:
            fig.tight_layout(w_pad=2.5)

        _save(fig,group_idx)
        plt.show()

    for group_idx,gi in enumerate(group_info):
        if split_gene_sets:
            _render_split(gi,group_idx)
        else:
            _render(gi,group_idx)

    return display_num,[gi["panels"] for gi in group_info]

def plot_recovery_heatmaps(
    df_dict,
    diff=False,
    full_delta=False,
    mhcprime_col="mhcprime",
    subval_pos="below",
    subval_fontsize=6,
    annot_fontsize=8,
    cmap="coolwarm_r",
    shared_yaxis=True,
    edge_color=None,
    edge_width=0.5,
    figsize=(18,4),
    dpi=150,
    default_font="Nimbus Sans",
    xtick_labels=None,
    xtick_rotation=45,
    dataset_title_labels=None,
    save_path=None,
    save_pdf=True,
    save_png=True,
):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Illustrator/publication-friendly settings.
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        default_font,
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42

    xtick_labels={} if xtick_labels is None else dict(xtick_labels)
    dataset_title_labels={} if dataset_title_labels is None else dict(dataset_title_labels)

    names=list(df_dict.keys())
    n=len(names)

    fig,axes=plt.subplots(
        1,n,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )
    axes=axes[0]

    def _ylabels(idx):
        return [f"Top {100-float(t):g}%" for t in idx]

    def _txt_color(im,v):
        r,g,b,_=im.cmap(im.norm(v))
        return "white" if (0.299*r+0.587*g+0.114*b)<0.5 else "black"

    def _heatmap(ax,vals,cmap,norm):
        nr,nc=vals.shape

        x=np.arange(nc+1)-0.5
        y=np.arange(nr+1)-0.5

        mesh=ax.pcolormesh(
            x,
            y,
            vals,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="none",
            linewidth=0,
            antialiased=False,
            rasterized=False,
        )

        ax.set_xlim(-0.5,nc-0.5)
        ax.set_ylim(nr-0.5,-0.5)

        return mesh

    def _outline(ax,nr,nc):
        if edge_color is None or nr==0 or nc==0:
            return

        for x in np.arange(0.5,nc-0.5,1.0):
            ax.axvline(
                x,
                color=edge_color,
                lw=edge_width,
                zorder=3,
            )

        for y in np.arange(0.5,nr-0.5,1.0):
            ax.axhline(
                y,
                color=edge_color,
                lw=edge_width,
                zorder=3,
            )

        ax.set_xlim(-0.5,nc-0.5)
        ax.set_ylim(nr-0.5,-0.5)

        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("black")
            sp.set_linewidth(1.0)
            sp.set_zorder(5)

    for i,(name,ax) in enumerate(zip(names,axes)):
        d=df_dict[name].copy()

        raw_start=raw_end=None

        if full_delta:
            if mhcprime_col not in d.columns:
                raise KeyError(
                    f"{name}: missing '{mhcprime_col}' for full_delta"
                )

            base=d[mhcprime_col].astype(float)
            models=[c for c in d.columns if c!=mhcprime_col]

            vals=(
                d[models]
                .astype(float)
                .sub(base,axis=0)
                .to_numpy()
            )

            raw_end=d[models].astype(float).to_numpy()
            raw_start=np.repeat(
                base.to_numpy()[:,None],
                len(models),
                axis=1,
            )

            cols=models

        elif diff:
            if mhcprime_col in d.columns:
                d=d.drop(columns=[mhcprime_col])

            vals=d.to_numpy(dtype=float)
            cols=list(d.columns)

        else:
            vals=d.to_numpy(dtype=float)
            cols=list(d.columns)

        if diff or full_delta:
            vmax=np.nanmax(np.abs(vals)) if vals.size else 1.0
            norm=Normalize(vmin=-vmax,vmax=vmax)

        else:
            norm=Normalize(
                vmin=np.nanmin(vals),
                vmax=np.nanmax(vals),
            )

        im=_heatmap(
            ax,
            vals,
            cmap=cmap,
            norm=norm,
        )

        _outline(
            ax,
            vals.shape[0],
            vals.shape[1],
        )

        display_cols=[
            xtick_labels.get(c,c)
            for c in cols
        ]

        ax.set_xticks(
            range(len(cols)),
            labels=display_cols,
            rotation=xtick_rotation,
            ha="center" if xtick_rotation==0 else "right",
        )

        ax.set_yticks(range(vals.shape[0]))

        ax.set_title(
            dataset_title_labels.get(name,name)
        )

        if i==0 or not shared_yaxis:
            ax.set_yticklabels(
                _ylabels(d.index)
            )
        else:
            ax.set_yticklabels([])

        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                v=vals[r,c]

                if np.isnan(v):
                    continue

                col=_txt_color(im,v)

                main=(
                    f"{v:+.2f}"
                    if (diff or full_delta)
                    else f"{v:.2f}"
                )

                if full_delta:
                    sub=(
                        f"{raw_start[r,c]:.1f}"
                        f"\u2192"
                        f"{raw_end[r,c]:.1f}"
                    )

                    if subval_pos=="topleft":
                        ax.text(
                            c,r,
                            main,
                            ha="center",
                            va="center",
                            fontsize=annot_fontsize,
                            fontfamily=default_font,
                            fontweight="normal",
                            color=col,
                        )

                        ax.text(
                            c-0.47,
                            r-0.44,
                            sub,
                            ha="left",
                            va="top",
                            fontsize=subval_fontsize,
                            fontfamily=default_font,
                            fontweight="normal",
                            color=col,
                        )

                    else:
                        ax.text(
                            c,
                            r-0.13,
                            main,
                            ha="center",
                            va="center",
                            fontsize=annot_fontsize,
                            fontfamily=default_font,
                            fontweight="normal",
                            color=col,
                        )

                        ax.text(
                            c,
                            r+0.24,
                            f"({sub})",
                            ha="center",
                            va="center",
                            fontsize=subval_fontsize,
                            fontfamily=default_font,
                            fontweight="normal",
                            color=col,
                        )

                else:
                    ax.text(
                        c,r,
                        main,
                        ha="center",
                        va="center",
                        fontsize=annot_fontsize,
                        fontfamily=default_font,
                        fontweight="normal",
                        color=col,
                    )

        div=make_axes_locatable(ax)

        cax=div.append_axes(
            "right",
            size="5%",
            pad=0.08,
        )

        fig.colorbar(
            im,
            cax=cax,
        )

    # fig.tight_layout(w_pad=2.5)
    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.18,
        top=0.90,
        wspace=0.35,
    )

    if save_path is not None:
        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

    plt.show()



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
        ref_assay=float(assay_map[p["ref_col"]])

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

            rows.append({
                "score_col":col,
                "method":method,
                "learning_rate":m["lr"],
                "learning_rate_float":_parse_lr(m["lr"]),
                "step_budget":budget,
                "step":step,
                "seed":int(m["seed"]),
                "is_explicit_full":cp is None,
                f"{p['ms_name']}_score":float(ms_map[col]),
                f"{p['assay_name']}_score":float(assay_map[col]),
                f"{p['ms_name']}_delta":float(ms_map[col])-ref_ms,
                f"{p['assay_name']}_delta":float(assay_map[col])-ref_assay,
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
    # Build trajectory + summary.
    # ------------------------------------------------------------------
    traj=_build_lambda_trajectory_df(
        ms_ap_df,assay_ap_df,score_cols,method_prefixes,
        ref_col=ref_col,ms_name=ms_name,assay_name=assay_name,
        ms_map_col=ms_map_col,assay_map_col=assay_map_col,
        method=method,
    )

    if lambda_df is None:
        lambda_df=summarize_lambda_performance(
            ms_ap_df,assay_ap_df,score_cols,method_prefixes,
            ref_col=ref_col,per_allele=False,
        )

    ld=lambda_df.copy()
    if "method" in ld.columns: ld=ld[ld["method"]==method].copy()
    if ld.empty: raise ValueError(f"No lambda summary rows remain for method='{method}'.")

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

def plot_pepcl_component_pairwise_forest_grid(
    replicate_dfs,

    # optional second condition
    replicate_dfs_no_allele_aware=None,

    # raw allele x score-column AP matrices, only needed if match_alleles=True
    raw_ap_dfs=None,
    raw_ap_dfs_no_allele_aware=None,
    match_alleles=False,

    dataset_order=None,
    dataset_labels=None,

    # comparisons
    comparisons=None,
    comparison_labels=None,

    # CI
    ci_method="t",
    ci=0.95,
    n_boot=10000,
    random_state=42,

    # figure
    figsize=(8.0,8.0),
    dpi=300,

    # ------------------------------------------------------------
    # GRANULAR GRID SPACING
    # ------------------------------------------------------------
    row_gap=0.35,
    comparison_col_gap=0.30,
    within_pair_gap=0.15,

    left_margin=0.08,
    right_margin=0.98,
    top_margin=0.95,
    bottom_margin=0.10,

    # optional outer-grid width/height weighting
    comparison_width_ratios=(1.0,1.0),
    row_height_ratios=(1.0,1.0,1.0),

    # allele-aware ON style
    assay_color="#2947a1",
    ms_color="#2947a1",
    marker="o",

    # allele-aware OFF style
    assay_color_no_allele_aware="#D34A4D",
    ms_color_no_allele_aware="#D34A4D",
    marker_no_allele_aware="o",

    # vertical separation around dataset tick
    condition_offset=0.10,

    # points / CI
    marker_size=5,
    marker_edgecolor="none",
    marker_edgewidth=0.0,
    ci_linewidth=1.0,
    ci_cap_size=3.0,
    ci_cap_thickness=1.0,

    # zero reference
    zero_color="#777777",
    zero_linewidth=0.8,
    zero_linestyle="--",

    # ------------------------------------------------------------
    # AXES
    # ------------------------------------------------------------
    assay_xlim=None,
    ms_xlim=None,

    # Optional per-comparison overrides.
    # Keys can be comparison names such as "FT vs DR only".
    assay_xlim_by_comparison=None,
    ms_xlim_by_comparison=None,

    assay_xlabel=None,
    ms_xlabel=None,
    ylabel="",

    # Whether each 2-panel comparison gets its own dataset labels.
    # True makes each comparison easier to move independently in Illustrator.
    show_dataset_labels_each_comparison=True,

    # Whether x-axis labels appear on every row.
    show_xlabels_each_row=True,

    # titles
    title=None,
    title_fontsize=8,
    title_y=0.98,

    comparison_title_fontsize=8,
    comparison_title_pad=5,
    comparison_title_y=None,

    # fonts / axes
    label_fontsize=7,
    tick_fontsize=7,
    axis_linewidth=0.5,
    tick_length=3,
    tick_width=0.5,

    # grid
    grid=False,
    grid_alpha=0.2,
    grid_linewidth=0.5,

    show_top_spine=False,
    show_right_spine=False,

    # ------------------------------------------------------------
    # LEGEND
    #
    # legend_mode:
    #   None       -> no legend
    #   "last"     -> one legend below final comparison
    #   "figure"   -> one legend below entire figure
    # ------------------------------------------------------------
    legend_mode="last",

    allele_aware_label="Allele-aware",
    no_allele_aware_label="Allele-aware off",

    legend_fontsize=7,
    legend_ncol=2,
    legend_marker_size=5,
    legend_frame=False,
    legend_columnspacing=1.5,
    legend_handletextpad=0.5,
    legend_handlelength=1.0,

    # legend beneath last comparison
    legend_last_x=0.5,
    legend_last_y=-0.28,

    # figure-level legend
    legend_figure_x=0.5,
    legend_figure_y=0.01,

    # output
    save_path=None,
    show=True,
):
    """
    Plot all six PepCL component-ablation pairwise comparisons in a 3 x 2 grid.

    Each comparison is itself a two-panel forest plot:
        left  = assay performance delta
        right = MS performance delta

    Delta:
        model_a - model_b

    Default layout
    --------------
    Row 1:
        FT vs DR only
        Replay only vs DR only

    Row 2:
        FT vs Replay only
        Full PepCL vs DR only

    Row 3:
        FT vs Full PepCL
        Full PepCL vs Replay only

    Returns
    -------
    plot_df : pd.DataFrame
        Long dataframe containing all comparison summaries.

    fig : matplotlib.figure.Figure

    axes : np.ndarray, shape (3,2,2)
        axes[row, comparison_column, metric_panel]
        metric_panel:
            0 = assay
            1 = MS
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import t
    from matplotlib.lines import Line2D

    valid_models={"FT","Replay only","DR only","Full PepCL"}

    # ============================================================
    # DEFAULT SIX COMPARISONS
    # ============================================================

    if comparisons is None:
        comparisons=[
            ("FT","DR only"),
            ("FT","Replay only"),
            ("FT","Full PepCL"),
            ("Replay only","DR only"),
            ("Full PepCL","DR only"),
            ("Full PepCL","Replay only"),
        ]

    if len(comparisons)!=6:
        raise ValueError(
            "comparisons must contain exactly 6 pairwise comparisons "
            "for the 3 x 2 layout."
        )

    for model_a,model_b in comparisons:
        if model_a not in valid_models or model_b not in valid_models:
            raise ValueError(
                f"Models must be in {sorted(valid_models)}. "
                f"Got {model_a} vs {model_b}."
            )
        if model_a==model_b:
            raise ValueError(
                f"Invalid comparison: {model_a} vs {model_b}."
            )

    # Reorder into the requested visual layout:
    #
    # LEFT COLUMN:
    #   FT vs DR
    #   FT vs Replay
    #   FT vs Full
    #
    # RIGHT COLUMN:
    #   Replay vs DR
    #   PepCL vs DR
    #   PepCL vs Replay
    #
    comparison_grid=[
        [comparisons[0],comparisons[3]],
        [comparisons[1],comparisons[4]],
        [comparisons[2],comparisons[5]],
    ]

    if comparison_labels is None:
        comparison_labels={
            ("FT","DR only"):"FT vs DR only",
            ("FT","Replay only"):"FT vs Replay only",
            ("FT","Full PepCL"):"FT vs Full PepCL",
            ("Replay only","DR only"):"Replay only vs DR only",
            ("Full PepCL","DR only"):"Full PepCL vs DR only",
            ("Full PepCL","Replay only"):"Full PepCL vs Replay only",
        }

    if ci_method not in {"bootstrap","t"}:
        raise ValueError("ci_method must be 'bootstrap' or 't'.")

    if legend_mode not in {None,"last","figure"}:
        raise ValueError(
            "legend_mode must be None, 'last', or 'figure'."
        )

    if dataset_order is None:
        dataset_order=list(replicate_dfs.keys())

    dataset_labels=dataset_labels or {}
    assay_xlim_by_comparison=assay_xlim_by_comparison or {}
    ms_xlim_by_comparison=ms_xlim_by_comparison or {}

    rng=np.random.default_rng(random_state)

    # ============================================================
    # VALIDATE ALLELE-MATCHING INPUT
    # ============================================================

    if match_alleles:
        if raw_ap_dfs is None:
            raise ValueError(
                "match_alleles=True requires raw_ap_dfs with "
                "{dataset: {'ms': ms_ap_df, 'assay': assay_ap_df}}."
            )

        if (
            replicate_dfs_no_allele_aware is not None
            and raw_ap_dfs_no_allele_aware is None
        ):
            raise ValueError(
                "When match_alleles=True and "
                "replicate_dfs_no_allele_aware is supplied, "
                "raw_ap_dfs_no_allele_aware must also be supplied."
            )

    # ============================================================
    # HELPERS
    # ============================================================

    def _ci(x):
        x=np.asarray(x,float)
        x=x[np.isfinite(x)]

        if len(x)<2:
            return np.nan,np.nan

        if ci_method=="bootstrap":
            b=x[
                rng.integers(
                    0,
                    len(x),
                    (n_boot,len(x)),
                )
            ].mean(axis=1)

            alpha=(1-ci)/2
            return tuple(
                np.quantile(
                    b,
                    [alpha,1-alpha],
                )
            )

        mean=x.mean()
        se=x.std(ddof=1)/np.sqrt(len(x))
        crit=t.ppf((1+ci)/2,len(x)-1)

        return (
            mean-crit*se,
            mean+crit*se,
        )

    def _get_performance_df(
        dataset,
        replicate_df,
        raw_dict,
    ):
        """
        Default:
            use assay_performance/ms_performance already present.

        match_alleles=True:
            recompute both using the common assay/MS allele set.
        """

        out=replicate_df.copy()

        if not match_alleles:
            return out

        if dataset not in raw_dict:
            raise KeyError(
                f"'{dataset}' missing from raw AP dataframe dictionary."
            )

        if not {"ms","assay"}.issubset(raw_dict[dataset]):
            raise KeyError(
                f"raw AP entry for '{dataset}' must contain "
                "'ms' and 'assay'."
            )

        ms_df=raw_dict[dataset]["ms"]
        assay_df=raw_dict[dataset]["assay"]

        common_alleles=(
            ms_df.index
            .intersection(assay_df.index)
        )

        if len(common_alleles)==0:
            raise ValueError(
                f"No overlapping assay/MS alleles for dataset '{dataset}'."
            )

        if "score_col" not in out.columns:
            raise ValueError(
                "match_alleles=True requires replicate_df "
                "to contain 'score_col'."
            )

        assay_vals=[]
        ms_vals=[]

        for score_col in out["score_col"]:

            if score_col not in assay_df.columns:
                raise KeyError(
                    f"'{score_col}' missing from assay AP dataframe "
                    f"for '{dataset}'."
                )

            if score_col not in ms_df.columns:
                raise KeyError(
                    f"'{score_col}' missing from MS AP dataframe "
                    f"for '{dataset}'."
                )

            assay_vals.append(
                pd.to_numeric(
                    assay_df.loc[
                        common_alleles,
                        score_col,
                    ],
                    errors="coerce",
                ).mean()
            )

            ms_vals.append(
                pd.to_numeric(
                    ms_df.loc[
                        common_alleles,
                        score_col,
                    ],
                    errors="coerce",
                ).mean()
            )

        out["assay_performance"]=assay_vals
        out["ms_performance"]=ms_vals
        out["n_matched_alleles"]=len(common_alleles)

        return out

    def _summarize_one_condition(
        dfs,
        condition,
        model_a,
        model_b,
        raw_dict=None,
    ):
        rows=[]

        for dataset in dataset_order:

            if dataset not in dfs:
                raise KeyError(
                    f"'{dataset}' not found in "
                    f"{condition} replicate dfs."
                )

            df=_get_performance_df(
                dataset=dataset,
                replicate_df=dfs[dataset],
                raw_dict=raw_dict,
            )

            required={
                "seed",
                "configuration",
                "assay_performance",
                "ms_performance",
            }

            missing=required-set(df.columns)

            if missing:
                raise ValueError(
                    f"{dataset} ({condition}) replicate_df "
                    f"missing columns: {sorted(missing)}"
                )

            a=df[
                df["configuration"]==model_a
            ][
                [
                    "seed",
                    "assay_performance",
                    "ms_performance",
                ]
            ].copy()

            b=df[
                df["configuration"]==model_b
            ][
                [
                    "seed",
                    "assay_performance",
                    "ms_performance",
                ]
            ].copy()

            paired=a.merge(
                b,
                on="seed",
                how="inner",
                suffixes=("_a","_b"),
                validate="one_to_one",
            )

            if paired.empty:
                raise ValueError(
                    f"No matched replicates for "
                    f"{model_a} vs {model_b} "
                    f"in {dataset} ({condition})."
                )

            assay_delta=(
                paired["assay_performance_a"]
                -paired["assay_performance_b"]
            ).to_numpy(float)

            ms_delta=(
                paired["ms_performance_a"]
                -paired["ms_performance_b"]
            ).to_numpy(float)

            assay_lo,assay_hi=_ci(assay_delta)
            ms_lo,ms_hi=_ci(ms_delta)

            row={
                "dataset":dataset,
                "dataset_label":dataset_labels.get(
                    dataset,
                    dataset,
                ),
                "condition":condition,
                "model_a":model_a,
                "model_b":model_b,
                "comparison":f"{model_a} vs {model_b}",
                "n_replicates":len(paired),

                "assay_delta_mean":assay_delta.mean(),
                "assay_delta_ci_low":assay_lo,
                "assay_delta_ci_high":assay_hi,

                "ms_delta_mean":ms_delta.mean(),
                "ms_delta_ci_low":ms_lo,
                "ms_delta_ci_high":ms_hi,

                "assay_delta_by_replicate":dict(
                    zip(
                        paired["seed"],
                        assay_delta,
                    )
                ),

                "ms_delta_by_replicate":dict(
                    zip(
                        paired["seed"],
                        ms_delta,
                    )
                ),
            }

            if match_alleles:
                row["n_matched_alleles"]=int(
                    df["n_matched_alleles"].iloc[0]
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def _summarize_comparison(
        model_a,
        model_b,
    ):
        aware_df=_summarize_one_condition(
            replicate_dfs,
            condition="allele_aware",
            model_a=model_a,
            model_b=model_b,
            raw_dict=raw_ap_dfs,
        )

        dfs=[aware_df]

        if replicate_dfs_no_allele_aware is not None:

            no_aware_df=_summarize_one_condition(
                replicate_dfs_no_allele_aware,
                condition="no_allele_aware",
                model_a=model_a,
                model_b=model_b,
                raw_dict=raw_ap_dfs_no_allele_aware,
            )

            dfs.append(no_aware_df)

        out=pd.concat(
            dfs,
            ignore_index=True,
        )

        out["dataset"]=pd.Categorical(
            out["dataset"],
            categories=dataset_order,
            ordered=True,
        )

        out=out.sort_values(
            ["dataset","condition"]
        ).reset_index(drop=True)

        return out

    # ============================================================
    # BUILD ALL SIX SUMMARIES
    # ============================================================

    comparison_dfs={}
    all_dfs=[]

    for row_pairs in comparison_grid:
        for model_a,model_b in row_pairs:

            key=(model_a,model_b)

            comparison_dfs[key]=_summarize_comparison(
                model_a,
                model_b,
            )

            all_dfs.append(
                comparison_dfs[key]
            )

    plot_df=pd.concat(
        all_dfs,
        ignore_index=True,
    )

    # ============================================================
    # FIGURE / NESTED GRID
    # ============================================================

    fig=plt.figure(
        figsize=figsize,
        dpi=dpi,
    )

    outer=fig.add_gridspec(
        3,
        2,
        width_ratios=comparison_width_ratios,
        height_ratios=row_height_ratios,
        wspace=comparison_col_gap,
        hspace=row_gap,
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
    )

    axes=np.empty(
        (3,2,2),
        dtype=object,
    )

    y_centers=np.arange(
        len(dataset_order)
    )[::-1]

    condition_styles={
        "allele_aware":{
            "offset":(
                condition_offset
                if replicate_dfs_no_allele_aware is not None
                else 0
            ),
            "assay_color":assay_color,
            "ms_color":ms_color,
            "marker":marker,
        },

        "no_allele_aware":{
            "offset":-condition_offset,
            "assay_color":assay_color_no_allele_aware,
            "ms_color":ms_color_no_allele_aware,
            "marker":marker_no_allele_aware,
        },
    }

    # ============================================================
    # DRAW EACH COMPARISON
    # ============================================================

    for row_idx,row_pairs in enumerate(comparison_grid):

        for col_idx,(model_a,model_b) in enumerate(row_pairs):

            inner=outer[
                row_idx,
                col_idx,
            ].subgridspec(
                1,
                2,
                wspace=within_pair_gap,
            )

            ax_assay=fig.add_subplot(
                inner[0,0]
            )

            ax_ms=fig.add_subplot(
                inner[0,1],
                sharey=ax_assay,
            )

            axes[row_idx,col_idx,0]=ax_assay
            axes[row_idx,col_idx,1]=ax_ms

            sub_df=comparison_dfs[
                (model_a,model_b)
            ]

            comp_name=comparison_labels.get(
                (model_a,model_b),
                f"{model_a} vs {model_b}",
            )

            panels=[
                (
                    ax_assay,
                    "assay_delta_mean",
                    "assay_delta_ci_low",
                    "assay_delta_ci_high",
                    assay_xlim_by_comparison.get(
                        comp_name,
                        assay_xlim,
                    ),
                    assay_xlabel
                    or (
                        "Δ assay performance\n"
                        f"({model_a} - {model_b})"
                    ),
                    "assay_color",
                ),

                (
                    ax_ms,
                    "ms_delta_mean",
                    "ms_delta_ci_low",
                    "ms_delta_ci_high",
                    ms_xlim_by_comparison.get(
                        comp_name,
                        ms_xlim,
                    ),
                    ms_xlabel
                    or (
                        "Δ MS mAP\n"
                        f"({model_a} - {model_b})"
                    ),
                    "ms_color",
                ),
            ]

            # --------------------------------------------------------
            # Draw assay + MS panels
            # --------------------------------------------------------

            for (
                panel_idx,
                (
                    ax,
                    mean_col,
                    lo_col,
                    hi_col,
                    xlim,
                    xlabel,
                    color_key,
                ),
            ) in enumerate(panels):

                for condition,style in condition_styles.items():

                    condition_df=sub_df[
                        sub_df["condition"]==condition
                    ].copy()

                    if condition_df.empty:
                        continue

                    condition_df=(
                        condition_df
                        .set_index("dataset")
                        .reindex(dataset_order)
                        .reset_index()
                    )

                    means=condition_df[
                        mean_col
                    ].to_numpy(float)

                    lows=condition_df[
                        lo_col
                    ].to_numpy(float)

                    highs=condition_df[
                        hi_col
                    ].to_numpy(float)

                    xerr=np.vstack(
                        [
                            means-lows,
                            highs-means,
                        ]
                    )

                    y=(
                        y_centers
                        +style["offset"]
                    )

                    color=style[color_key]

                    eb=ax.errorbar(
                        means,
                        y,
                        xerr=xerr,

                        fmt=style["marker"],
                        markersize=marker_size,

                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=marker_edgecolor,

                        ecolor=color,
                        elinewidth=ci_linewidth,
                        capsize=ci_cap_size,
                        capthick=ci_cap_thickness,

                        linestyle="none",
                        zorder=3,
                    )

                    if eb.lines[0] is not None:
                        eb.lines[0].set_markeredgewidth(marker_edgewidth)

                ax.axvline(
                    0,
                    color=zero_color,
                    linewidth=zero_linewidth,
                    linestyle=zero_linestyle,
                    zorder=1,
                )

                if (
                    show_xlabels_each_row
                    or row_idx==2
                ):
                    ax.set_xlabel(
                        xlabel,
                        fontsize=label_fontsize,
                    )
                else:
                    ax.set_xlabel("")

                if xlim is not None:
                    ax.set_xlim(xlim)

                if grid:
                    ax.grid(
                        axis="x",
                        alpha=grid_alpha,
                        linewidth=grid_linewidth,
                        zorder=0,
                    )

                ax.spines["top"].set_visible(
                    show_top_spine
                )

                ax.spines["right"].set_visible(
                    show_right_spine
                )

                ax.spines["left"].set_linewidth(
                    axis_linewidth
                )

                ax.spines["bottom"].set_linewidth(
                    axis_linewidth
                )

                ax.tick_params(
                    axis="both",
                    labelsize=tick_fontsize,
                    length=tick_length,
                    width=tick_width,
                )

            # --------------------------------------------------------
            # Dataset tick labels
            # --------------------------------------------------------

            ax_assay.set_yticks(
                y_centers
            )

            if show_dataset_labels_each_comparison:

                ax_assay.set_yticklabels(
                    [
                        dataset_labels.get(d,d)
                        for d in dataset_order
                    ],
                    fontsize=tick_fontsize,
                )

            elif col_idx==0:

                ax_assay.set_yticklabels(
                    [
                        dataset_labels.get(d,d)
                        for d in dataset_order
                    ],
                    fontsize=tick_fontsize,
                )

            else:

                ax_assay.tick_params(
                    axis="y",
                    left=False,
                    labelleft=False,
                )

            ax_ms.tick_params(
                axis="y",
                left=False,
                labelleft=False,
            )

            if ylabel:
                ax_assay.set_ylabel(
                    ylabel,
                    fontsize=label_fontsize,
                )

            # --------------------------------------------------------
            # Comparison title spanning the pair visually
            # --------------------------------------------------------

            if comparison_title_y is None:
                title_y=1.02
            else:
                title_y=comparison_title_y

            # Put title on the assay panel, but shift toward the center
            # of the two-panel comparison.
            ax_assay.set_title(
                comp_name,
                fontsize=comparison_title_fontsize,
                pad=comparison_title_pad,
                y=title_y,
                loc="left",
            )

    # ============================================================
    # LEGEND
    # ============================================================

    if (
        replicate_dfs_no_allele_aware is not None
        and legend_mode is not None
    ):

        legend_handles=[
            Line2D(
                [0],[0],
                marker=marker,
                linestyle="-",
                linewidth=ci_linewidth,
                markersize=legend_marker_size,
                color=assay_color,
                markerfacecolor=assay_color,
                markeredgecolor=assay_color,
                label=allele_aware_label,
            ),

            Line2D(
                [0],[0],
                marker=marker_no_allele_aware,
                linestyle="-",
                linewidth=ci_linewidth,
                markersize=legend_marker_size,
                color=assay_color_no_allele_aware,
                markerfacecolor=assay_color_no_allele_aware,
                markeredgecolor=assay_color_no_allele_aware,
                label=no_allele_aware_label,
            ),
        ]

        if legend_mode=="last":

            # Attach once below the final panel:
            # Full PepCL vs Replay only, MS axis.
            last_ax=axes[2,1,1]

            last_ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(
                    legend_last_x,
                    legend_last_y,
                ),
                ncol=legend_ncol,
                frameon=legend_frame,
                fontsize=legend_fontsize,
                columnspacing=legend_columnspacing,
                handletextpad=legend_handletextpad,
                handlelength=legend_handlelength,
                borderaxespad=0,
            )

        elif legend_mode=="figure":

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(
                    legend_figure_x,
                    legend_figure_y,
                ),
                ncol=legend_ncol,
                frameon=legend_frame,
                fontsize=legend_fontsize,
                columnspacing=legend_columnspacing,
                handletextpad=legend_handletextpad,
                handlelength=legend_handlelength,
            )

    # ============================================================
    # OPTIONAL OVERALL TITLE
    # ============================================================

    if title:
        fig.suptitle(
            title,
            fontsize=title_fontsize,
            y=title_y,
        )

    # ============================================================
    # SAVE
    # ============================================================

    if save_path:

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

    if show:
        plt.show()

    return plot_df,fig,axes

def plot_component_summary_table(
    summary_dfs,
    summary_dfs_aa_off,
    dataset_labels,
    configuration_labels=None,
    configuration_order=("FT","DR only","Replay only","Full PepCL"),
    include_mhcprime=True,
    mhcprime_configuration="MHCPrime",
    decimals=3,
    aa_sampling_order="paired",

    # Figure / fonts.
    fig_size=None,
    dpi=300,
    title="",
    title_fontsize=16,
    header_fontsize=11,
    body_fontsize=11,
    font_family="Nimbus Sans",

    # Table geometry.
    col_widths=None,
    table_bbox=(0.0,0.02,1.0,0.92),
    row_height_scale=1.45,
    header_height_scale=1.0,
    header_gap_scale=0.0,
    section_gap_scale=0.0,
    top_data_gap_scale=0.0,
    bottom_data_gap_scale=0.0,
    cell_pad=None,

    # Column alignment.
    assay_alignment="center",
    setting_alignment="center",
    left_align_pad=0.03,

    # Horizontal rules.
    top_linewidth=1.5,
    header_linewidth=0.8,
    bottom_linewidth=1.5,
    row_linewidth=0.5,
    row_line_alpha=0.35,
    line_color="black",
    independent_lines=True,

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Plot manuscript-style component-ablation summary table from AA-on and
    AA-off summary dataframes.

    aa_sampling_order
    -----------------
    "paired":
        FT On
        FT Off
        DR only On
        DR only Off
        ...

    "on_then_off":
        FT On
        DR only On
        Replay only On
        PepCL On
        FT Off
        DR only Off
        Replay only Off
        PepCL Off

    Alignment controls
    ------------------
    assay_alignment:
        "center" or "left". Controls the merged Assay labels.

    setting_alignment:
        "center" or "left". Controls the Setting body column.

    left_align_pad:
        Fraction of the corresponding column width used as left padding
        when assay_alignment or setting_alignment is "left".

    Spacing controls
    ----------------
    header_height_scale:
        Scales the header row itself.

    header_gap_scale:
        Extra blank space between the header-bottom rule and first data block.

    section_gap_scale:
        Extra blank space centered around assay-section separator rules.

    top_data_gap_scale:
        Extra blank space immediately above the first data row.

    bottom_data_gap_scale:
        Extra blank space immediately below the final data row and above
        the bottom table rule.

    All gap scales are expressed relative to one normal body-row height.
    Defaults preserve the original layout.

    MHCPrime is taken only from summary_dfs and displayed without a CI.

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

    valid_aa_orders={"paired","on_then_off"}
    if aa_sampling_order not in valid_aa_orders:
        raise ValueError(
            f"aa_sampling_order must be one of {sorted(valid_aa_orders)}, "
            f"got '{aa_sampling_order}'."
        )

    valid_alignments={"left","center"}
    if assay_alignment not in valid_alignments:
        raise ValueError(
            f"assay_alignment must be one of {sorted(valid_alignments)}, "
            f"got '{assay_alignment}'."
        )
    if setting_alignment not in valid_alignments:
        raise ValueError(
            f"setting_alignment must be one of {sorted(valid_alignments)}, "
            f"got '{setting_alignment}'."
        )

    for name,value in {
        "header_height_scale":header_height_scale,
        "header_gap_scale":header_gap_scale,
        "section_gap_scale":section_gap_scale,
        "top_data_gap_scale":top_data_gap_scale,
        "bottom_data_gap_scale":bottom_data_gap_scale,
    }.items():
        if name=="header_height_scale":
            if value<=0:
                raise ValueError("header_height_scale must be > 0.")
        elif value<0:
            raise ValueError(f"{name} must be >= 0.")

    if configuration_labels is None:
        configuration_labels={
            "MHCPrime":"MHCPrime",
            "FT":"FT",
            "DR only":"DR only",
            "Replay only":"Replay only",
            "Full PepCL":"PepCL",
        }

    if col_widths is None:
        col_widths=[0.12,0.13,0.09,0.25,0.25]

    if len(col_widths)!=5:
        raise ValueError("col_widths must contain exactly 5 values.")

    # ================================================================
    # Helpers
    # ================================================================
    def _get_row(df,config,dataset,source):
        if "configuration" not in df.columns:
            raise ValueError(
                f"{dataset}/{source}: missing 'configuration' column."
            )

        sub=df.loc[df["configuration"]==config]

        if len(sub)==0:
            raise ValueError(
                f"{dataset}/{source}: configuration '{config}' not found."
            )

        if len(sub)>1:
            raise ValueError(
                f"{dataset}/{source}: configuration '{config}' occurs "
                f"{len(sub)} times; expected exactly one row."
            )

        return sub.iloc[0]

    def _fmt_value(row,prefix,with_ci=True):
        mean=row[f"{prefix}_mean"]

        if pd.isna(mean):
            return "–"

        if not with_ci:
            return f"{float(mean):.{decimals}f}"

        lo=row[f"{prefix}_ci_low"]
        hi=row[f"{prefix}_ci_high"]

        if pd.isna(lo) or pd.isna(hi):
            return f"{float(mean):.{decimals}f}"

        return (
            f"{float(mean):.{decimals}f} "
            f"[{float(lo):.{decimals}f}, {float(hi):.{decimals}f}]"
        )

    def _make_row(dataset,config,aa_label,source_df):
        row=_get_row(
            source_df,
            config,
            dataset,
            "AA-on" if aa_label=="On" else "AA-off",
        )

        return {
            "Assay":dataset_labels.get(dataset,dataset),
            "Setting":configuration_labels.get(config,config),
            "AA sampling":aa_label,
            "Assay performance [95% CI]":_fmt_value(
                row,
                "assay_performance",
                with_ci=True,
            ),
            "MS performance [95% CI]":_fmt_value(
                row,
                "ms_performance",
                with_ci=True,
            ),
        }

    # ================================================================
    # Validate dataset dictionaries
    # ================================================================
    dataset_order=list(dataset_labels.keys())

    for dataset in dataset_order:
        if dataset not in summary_dfs:
            raise KeyError(
                f"Dataset '{dataset}' missing from summary_dfs."
            )

        if dataset not in summary_dfs_aa_off:
            raise KeyError(
                f"Dataset '{dataset}' missing from summary_dfs_aa_off."
            )

    # ================================================================
    # Build logical display dataframe
    # ================================================================
    rows=[]
    logical_groups=[]

    for dataset in dataset_order:
        df_on=summary_dfs[dataset]
        df_off=summary_dfs_aa_off[dataset]

        group_start=len(rows)

        if include_mhcprime:
            ref=_get_row(
                df_on,
                mhcprime_configuration,
                dataset,
                "AA-on",
            )

            rows.append({
                "Assay":dataset_labels.get(dataset,dataset),
                "Setting":configuration_labels.get(
                    mhcprime_configuration,
                    mhcprime_configuration,
                ),
                "AA sampling":"–",
                "Assay performance [95% CI]":_fmt_value(
                    ref,
                    "assay_performance",
                    with_ci=False,
                ),
                "MS performance [95% CI]":_fmt_value(
                    ref,
                    "ms_performance",
                    with_ci=False,
                ),
            })

        if aa_sampling_order=="paired":
            for config in configuration_order:
                rows.append(
                    _make_row(dataset,config,"On",df_on)
                )
                rows.append(
                    _make_row(dataset,config,"Off",df_off)
                )

        else:
            for config in configuration_order:
                rows.append(
                    _make_row(dataset,config,"On",df_on)
                )

            for config in configuration_order:
                rows.append(
                    _make_row(dataset,config,"Off",df_off)
                )

        group_end=len(rows)-1

        logical_groups.append(
            (
                group_start,
                group_end,
                dataset_labels.get(dataset,dataset),
            )
        )

    display_df=pd.DataFrame(rows)

    columns=[
        "Assay",
        "Setting",
        "AA sampling",
        "Assay performance [95% CI]",
        "MS performance [95% CI]",
    ]

    # ================================================================
    # Build rendering dataframe with optional spacer rows
    # ================================================================
    render_rows=[]
    render_groups=[]
    spacer_info=[]

    def _append_spacer(kind):
        idx=len(render_rows)

        render_rows.append({
            "Assay":"",
            "Setting":"",
            "AA sampling":"",
            "Assay performance [95% CI]":"",
            "MS performance [95% CI]":"",
            "_is_spacer":True,
            "_spacer_kind":kind,
        })

        spacer_info.append(
            (idx,kind)
        )

    if top_data_gap_scale>0:
        _append_spacer("top_data")

    if header_gap_scale>0:
        _append_spacer("header_gap")

    for group_i,(start,end,label) in enumerate(logical_groups):
        render_start=len(render_rows)

        for i in range(start,end+1):
            row=display_df.iloc[i].to_dict()
            row["_is_spacer"]=False
            row["_spacer_kind"]=None
            render_rows.append(row)

        render_end=len(render_rows)-1

        render_groups.append(
            (
                render_start,
                render_end,
                label,
            )
        )

        if (
            section_gap_scale>0
            and group_i<len(logical_groups)-1
        ):
            _append_spacer("section")

    if bottom_data_gap_scale>0:
        _append_spacer("bottom_data")

    render_df=pd.DataFrame(render_rows)

    # ================================================================
    # Matplotlib / Illustrator-compatible settings
    # ================================================================
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
        size=body_fontsize,
    )

    fp_body_bold=FontProperties(
        family=font_family,
        size=body_fontsize,
        weight="bold",
    )

    fp_header=FontProperties(
        family=font_family,
        size=header_fontsize,
        weight="bold",
    )

    fp_title=FontProperties(
        family=font_family,
        size=title_fontsize,
    )

    # ================================================================
    # Figure
    # ================================================================
    if fig_size is None:
        fig_size=(
            14,
            max(
                3.0,
                0.36*(len(render_df)+2),
            ),
        )

    fig,ax=plt.subplots(
        figsize=fig_size,
        dpi=dpi,
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
            fontproperties=fp_title,
        )

    table_display=render_df[columns].copy()
    table_display["Assay"]=""

    tbl=ax.table(
        cellText=table_display.values.tolist(),
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=table_bbox,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(body_fontsize)
    tbl.scale(
        1.0,
        row_height_scale,
    )

    nrows=len(render_df)+1
    ncols=len(columns)

    # ================================================================
    # Remove default borders/backgrounds
    # ================================================================
    for (r,c),cell in tbl.get_celld().items():
        cell.visible_edges=""
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(fp_body)

        if cell_pad is not None:
            cell.PAD=cell_pad

    # ================================================================
    # Header
    # ================================================================
    for c in range(ncols):
        txt=tbl[(0,c)].get_text()
        txt.set_fontproperties(fp_header)
        txt.set_ha("center")

    base_header_height=tbl[(0,0)].get_height()

    for c in range(ncols):
        tbl[(0,c)].set_height(
            base_header_height*header_height_scale
        )

    # ================================================================
    # Body alignment
    #
    # Assay is redrawn manually below.
    # Setting may be independently left/center aligned.
    # Remaining columns stay centered.
    # ================================================================
    for r in range(1,nrows):
        # Setting column.
        setting_txt=tbl[(r,1)].get_text()

        if setting_alignment=="left":
            setting_txt.set_ha("left")
            tbl[(r,1)].PAD=left_align_pad
        else:
            setting_txt.set_ha("center")

        # Remaining non-Assay columns.
        for c in range(2,ncols):
            tbl[(r,c)].get_text().set_ha("center")

    # ================================================================
    # Set spacer-row heights
    # ================================================================
    if nrows>1:
        normal_height=tbl[(1,0)].get_height()

        spacer_scales={
            "header_gap":header_gap_scale,
            "section":section_gap_scale,
            "top_data":top_data_gap_scale,
            "bottom_data":bottom_data_gap_scale,
        }

        for render_row,kind in spacer_info:
            table_row=render_row+1
            scale=spacer_scales[kind]

            for c in range(ncols):
                tbl[(table_row,c)].set_height(
                    normal_height*scale
                )

    fig.canvas.draw()

    # ================================================================
    # Draw merged assay labels
    # ================================================================
    for start,end,label in render_groups:
        r1=start+1
        r2=end+1

        cell_top=tbl[(r1,0)]
        cell_bottom=tbl[(r2,0)]

        x=cell_top.get_x()
        width=cell_top.get_width()

        top=(
            cell_top.get_y()
            +cell_top.get_height()
        )

        bottom=cell_bottom.get_y()
        center_y=(top+bottom)/2

        if assay_alignment=="left":
            text_x=x+left_align_pad*width
            ha="left"
        else:
            text_x=x+width/2
            ha="center"

        ax.text(
            text_x,
            center_y,
            label,
            transform=ax.transAxes,
            ha=ha,
            va="center",
            fontproperties=fp_body_bold,
            zorder=10,
        )

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
    table_bottom=tbl[(nrows-1,0)].get_y()

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
            zorder=5,
        )

    _add_rule(
        header_top,
        top_linewidth,
    )

    _add_rule(
        header_bottom,
        header_linewidth,
    )

    section_spacers=[
        idx
        for idx,kind in spacer_info
        if kind=="section"
    ]

    for spacer_render_row in section_spacers:
        table_row=spacer_render_row+1
        spacer_cell=tbl[(table_row,0)]

        y=(
            spacer_cell.get_y()
            +spacer_cell.get_height()/2
        )

        _add_rule(
            y,
            row_linewidth,
            alpha=row_line_alpha,
        )

    _add_rule(
        table_bottom,
        bottom_linewidth,
    )

    # ================================================================
    # Save
    # ================================================================
    if save_path is not None:
        fig.savefig(
            f"{save_path}.pdf",
            dpi=dpi,
            bbox_inches="tight",
        )

        fig.savefig(
            f"{save_path}.png",
            dpi=dpi,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    return fig,ax,tbl,display_df

import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_allele_aware_heatmap_panel(
    ap_dfs,
    ap_dfs_aa_off,
    replicate_dfs,
    replicate_dfs_aa_off,
    method="Full PepCL",

    dataset_order=("holec","episcan_r9","episcan_r9rt","ba_con","escape_con"),
    dataset_labels=None,
    allele_labels=None,
    match_alleles=False,

    # ------------------------------------------------------------------
    # Color
    # ------------------------------------------------------------------
    cmap="coolwarm_r",                 # negative=red, positive=blue
    heatmap_absmax=None,               # None -> LOCAL scale per dataset
                                        # scalar -> same manual limit everywhere
                                        # dict -> manual limit per dataset
    nan_color="white",

    # ------------------------------------------------------------------
    # Cell annotations
    # ------------------------------------------------------------------
    annotate=True,
    annotation_decimals=3,
    annotation_fontsize=6,
    annotation_switch_frac=0.55,

    # ------------------------------------------------------------------
    # Overall figure
    # ------------------------------------------------------------------
    figsize=(15,7),
    dpi=300,
    left=0.045,
    right=0.99,
    bottom=0.10,
    plot_top=0.80,
    panel_wspace=0.45,

    # ------------------------------------------------------------------
    # Figure-level title
    # ------------------------------------------------------------------
    title=None,
    title_fontsize=11,
    title_y=0.985,

    # ------------------------------------------------------------------
    # Dataset titles
    # These are positioned in FIGURE coordinates relative to each heatmap.
    # ------------------------------------------------------------------
    dataset_title_fontsize=8,
    dataset_title_gap=0.070,

    # ------------------------------------------------------------------
    # LOCAL horizontal colorbars
    # Also positioned in FIGURE coordinates.
    # ------------------------------------------------------------------
    colorbar_gap=0.018,
    colorbar_height=0.018,
    colorbar_ticksize=5.5,
    colorbar_tick_length=2,
    colorbar_tick_width=0.5,
    colorbar_tick_pad=1.5,
    colorbar_label=None,
    colorbar_labelsize=6,
    colorbar_labelpad=2,

    # ------------------------------------------------------------------
    # Heatmap axes
    # ------------------------------------------------------------------
    allele_fontsize=6,
    column_label_fontsize=7,
    spine_width=0.6,

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    save_path=None,
    save_pdf=True,
    save_png=True,
    save_dpi=None,
    save_bbox_inches="tight",
    save_pad_inches=0.05,

    show=True,
):
    """
    Plot allele-aware ON vs OFF effects across datasets as side-by-side heatmaps.

    Each dataset heatmap:
        rows:
            alleles

        columns:
            Assay adaptation advantage
            MS retention advantage

    For each matched seed:
        advantage = performance(AA-on) - performance(AA-off)

    Because the same MHCPrime baseline would cancel from both terms, this is
    numerically equivalent to:

        [(AA-on - MHCPrime) - (AA-off - MHCPrime)]

    Interpretation
    --------------
    positive -> AA-on better
    negative -> AA-off better

    Color scale
    -----------
    By default, each dataset receives its OWN LOCAL symmetric color scale:

        [-max_abs(dataset), +max_abs(dataset)]

    This prevents datasets with larger effects from washing out smaller
    but potentially meaningful effects in other datasets.

    heatmap_absmax may be:
        None:
            automatically determine local scale for each dataset.

        scalar:
            force the same +/- limit for all datasets.

        dict:
            specify limits separately, e.g.
            {
                "holec": 0.03,
                "episcan_r9": 0.015,
                "episcan_r9rt": 0.05,
                "ba_con": 0.05,
                "escape_con": 0.06,
            }

    Allele matching
    ---------------
    match_alleles=False:
        assay and MS use all available alleles. Missing assay/MS combinations
        appear white.

    match_alleles=True:
        within each dataset, assay and MS are restricted to their common allele
        set so the two columns have exact row parity.

    Allele display format
    ---------------------
    Internal allele identifiers are NOT changed.

    Display labels are formatted as:
        A0201 -> A*02:01
        B5701 -> B*57:01
        C1203 -> C*12:03

    Returns
    -------
    heatmap_df : pd.DataFrame
        Tidy allele-level mean paired effects.

    matrices : dict
        Dataset -> plotted heatmap dataframe.

    seed_df : pd.DataFrame
        Underlying seed-level paired AA-on minus AA-off values.

    fig, axes
    """

    # ================================================================
    # Matplotlib / Illustrator-compatible font + vector export settings
    # ================================================================
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
    ]
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"
    matplotlib.rcParams["text.usetex"]=False
    matplotlib.rcParams["image.composite_image"]=False

    default_font="Nimbus Sans"

    # ================================================================
    # Defaults
    # ================================================================
    dataset_labels=dataset_labels or {
        "holec":"YD",
        "episcan_r9":"EpiScan R9",
        "episcan_r9rt":"EpiScan R9/RT",
        "ba_con":"BA",
        "escape_con":"ESCAPE-seq",
    }

    allele_labels=allele_labels or {}

    # ================================================================
    # Allele formatting
    # ================================================================
    def _format_allele(a):
        """
        Format common compact HLA-I allele identifiers for display only.

        Examples
        --------
        A0201       -> A*02:01
        B5701       -> B*57:01
        C1203       -> C*12:03
        HLA-A0201   -> A*02:01
        A02:01      -> A*02:01
        HLA-A*02:01 -> A*02:01

        Unknown formats are returned unchanged rather than guessed.
        """
        original=str(a)
        s=original.strip().upper()
        s=s.replace("HLA-","").replace("*","").replace(":","")

        m=re.fullmatch(r"([ABC])(\d{2})(\d{2})",s)

        if m:
            return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"

        return original

    def _display_allele(a):
        formatted=_format_allele(a)

        if callable(allele_labels):
            return allele_labels(formatted)

        if allele_labels:
            return allele_labels.get(
                a,
                allele_labels.get(
                    formatted,
                    formatted,
                ),
            )

        return formatted

    # ================================================================
    # Helpers
    # ================================================================
    def _get_method_cols(rep_df,method):
        required={
            "seed",
            "configuration",
            "score_col",
        }

        missing=required-set(rep_df.columns)

        if missing:
            raise ValueError(
                f"replicate_df missing required columns: {sorted(missing)}"
            )

        out=(
            rep_df.loc[
                rep_df["configuration"]==method,
                [
                    "seed",
                    "score_col",
                ],
            ]
            .drop_duplicates()
        )

        if out.empty:
            raise ValueError(
                f"No rows found for method='{method}'."
            )

        if out["seed"].duplicated().any():
            dup=sorted(
                out.loc[
                    out["seed"].duplicated(False),
                    "seed",
                ].unique()
            )

            raise ValueError(
                f"Method '{method}' contains more than one score column "
                f"for seed(s): {dup}"
            )

        return dict(
            zip(
                out["seed"],
                out["score_col"],
            )
        )

    def _prepare_df(df):
        out=df.copy()

        if "allele" in out.columns:
            out=out.set_index("allele")

        out.index=out.index.astype(str)

        return out

    # ================================================================
    # Build seed-level paired AA-on - AA-off effects
    # ================================================================
    rows=[]

    for dataset in dataset_order:

        if dataset not in ap_dfs:
            raise KeyError(
                f"'{dataset}' missing from ap_dfs."
            )

        if dataset not in ap_dfs_aa_off:
            raise KeyError(
                f"'{dataset}' missing from ap_dfs_aa_off."
            )

        if dataset not in replicate_dfs:
            raise KeyError(
                f"'{dataset}' missing from replicate_dfs."
            )

        if dataset not in replicate_dfs_aa_off:
            raise KeyError(
                f"'{dataset}' missing from replicate_dfs_aa_off."
            )

        # ------------------------------------------------------------
        # Match AA-on and AA-off runs by seed
        # ------------------------------------------------------------
        cols_on=_get_method_cols(
            replicate_dfs[dataset],
            method,
        )

        cols_off=_get_method_cols(
            replicate_dfs_aa_off[dataset],
            method,
        )

        seeds=sorted(
            set(cols_on)&set(cols_off)
        )

        if not seeds:
            raise ValueError(
                f"No matched AA-on/AA-off seeds for "
                f"{dataset}, method='{method}'."
            )

        # ------------------------------------------------------------
        # Expected standardized dictionary structure:
        #
        # component_ap_dfs = {
        #     dataset:{
        #         "assay": df,
        #         "ms": df,
        #     }
        # }
        # ------------------------------------------------------------
        for key in (
            "assay",
            "ms",
        ):

            if key not in ap_dfs[dataset]:
                raise KeyError(
                    f"{dataset}: missing '{key}' in ap_dfs."
                )

            if key not in ap_dfs_aa_off[dataset]:
                raise KeyError(
                    f"{dataset}: missing '{key}' in ap_dfs_aa_off."
                )

        assay_on=_prepare_df(
            ap_dfs[dataset]["assay"]
        )

        assay_off=_prepare_df(
            ap_dfs_aa_off[dataset]["assay"]
        )

        ms_on=_prepare_df(
            ap_dfs[dataset]["ms"]
        )

        ms_off=_prepare_df(
            ap_dfs_aa_off[dataset]["ms"]
        )

        # ------------------------------------------------------------
        # First match AA-on vs AA-off within each domain
        # ------------------------------------------------------------
        assay_alleles=(
            assay_on.index
            .intersection(
                assay_off.index
            )
        )

        ms_alleles=(
            ms_on.index
            .intersection(
                ms_off.index
            )
        )

        # ------------------------------------------------------------
        # Optional assay/MS allele matching
        # ------------------------------------------------------------
        if match_alleles:
            common=(
                assay_alleles
                .intersection(
                    ms_alleles
                )
            )

            assay_alleles=common
            ms_alleles=common

        # ------------------------------------------------------------
        # Collect seed-level paired differences
        # ------------------------------------------------------------
        domains=[
            (
                "assay",
                assay_on,
                assay_off,
                assay_alleles,
            ),
            (
                "ms",
                ms_on,
                ms_off,
                ms_alleles,
            ),
        ]

        for domain,on_df,off_df,alleles in domains:

            for seed in seeds:

                col_on=cols_on[seed]
                col_off=cols_off[seed]

                if col_on not in on_df.columns:
                    raise KeyError(
                        f"{dataset}/{domain}: "
                        f"AA-on missing '{col_on}'."
                    )

                if col_off not in off_df.columns:
                    raise KeyError(
                        f"{dataset}/{domain}: "
                        f"AA-off missing '{col_off}'."
                    )

                on=pd.to_numeric(
                    on_df.loc[
                        alleles,
                        col_on,
                    ],
                    errors="coerce",
                )

                off=pd.to_numeric(
                    off_df.loc[
                        alleles,
                        col_off,
                    ],
                    errors="coerce",
                )

                valid=(
                    on.notna()
                    &off.notna()
                )

                for allele in alleles[valid]:

                    rows.append({
                        "dataset":dataset,

                        "dataset_label":dataset_labels.get(
                            dataset,
                            dataset,
                        ),

                        "domain":domain,

                        "allele":str(
                            allele
                        ),

                        "allele_label":_display_allele(
                            str(allele)
                        ),

                        "seed":seed,

                        "method":method,

                        "aa_on":float(
                            on.loc[
                                allele
                            ]
                        ),

                        "aa_off":float(
                            off.loc[
                                allele
                            ]
                        ),

                        "advantage":float(
                            on.loc[
                                allele
                            ]
                            -off.loc[
                                allele
                            ]
                        ),
                    })

    seed_df=pd.DataFrame(
        rows
    )

    if seed_df.empty:
        raise ValueError(
            "No paired allele/seed observations were generated."
        )

    # ================================================================
    # Aggregate paired differences across seeds WITHIN allele
    # ================================================================
    heatmap_df=(
        seed_df
        .groupby(
            [
                "dataset",
                "dataset_label",
                "domain",
                "allele",
                "allele_label",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            advantage_mean=(
                "advantage",
                "mean",
            ),
            advantage_sd=(
                "advantage",
                "std",
            ),
            n_seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    # ================================================================
    # Build matrix for each dataset
    # ================================================================
    matrices={}

    for dataset in dataset_order:

        sub=heatmap_df[
            heatmap_df["dataset"]==dataset
        ].copy()

        assay=sub[
            sub["domain"]=="assay"
        ].copy()

        ms=sub[
            sub["domain"]=="ms"
        ].copy()

        assay_labels=list(
            dict.fromkeys(
                assay[
                    "allele_label"
                ].tolist()
            )
        )

        ms_labels=list(
            dict.fromkeys(
                ms[
                    "allele_label"
                ].tolist()
            )
        )

        if match_alleles:

            ms_set=set(
                ms_labels
            )

            allele_order=[
                a
                for a in assay_labels
                if a in ms_set
            ]

        else:

            allele_order=list(
                dict.fromkeys(
                    assay_labels
                    +ms_labels
                )
            )

        mat=(
            sub
            .pivot(
                index="allele_label",
                columns="domain",
                values="advantage_mean",
            )
            .reindex(
                index=allele_order,
                columns=[
                    "assay",
                    "ms",
                ],
            )
        )

        mat.columns=[
            "Assay\nadaptation",
            "MS\nretention",
        ]

        matrices[dataset]=mat

    # ================================================================
    # LOCAL color normalization per dataset
    # ================================================================
    norms={}

    for dataset,mat in matrices.items():

        vals=np.abs(
            mat.to_numpy(
                float
            )
        )

        vals=vals[
            np.isfinite(
                vals
            )
        ]

        if heatmap_absmax is None:

            vmax=(
                float(
                    vals.max()
                )
                if len(vals)
                else 1.0
            )

        elif isinstance(
            heatmap_absmax,
            dict,
        ):

            if dataset not in heatmap_absmax:
                raise KeyError(
                    f"heatmap_absmax dict is missing "
                    f"dataset '{dataset}'."
                )

            vmax=float(
                heatmap_absmax[
                    dataset
                ]
            )

        else:

            vmax=float(
                heatmap_absmax
            )

        vmax=max(
            vmax,
            1e-12,
        )

        norms[dataset]=Normalize(
            vmin=-vmax,
            vmax=vmax,
        )

    # ================================================================
    # Figure
    # ================================================================
    fig,axes=plt.subplots(
        1,
        len(dataset_order),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    axes=axes.ravel()

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=plot_top,
        wspace=panel_wspace,
    )

    cmap_obj=plt.get_cmap(
        cmap
    ).copy()

    cmap_obj.set_bad(
        nan_color
    )

    # Finalize axes positions before manually placing titles/colorbars.
    fig.canvas.draw()

    # ================================================================
    # Draw dataset panels
    # ================================================================
    for ax,dataset in zip(
        axes,
        dataset_order,
    ):

        mat=matrices[
            dataset
        ]

        vals=mat.to_numpy(
            float
        )

        norm=norms[
            dataset
        ]

        vmax=norm.vmax

        # ------------------------------------------------------------
        # VECTOR HEATMAP
        #
        # IMPORTANT:
        # Do NOT use imshow here. imshow embeds the heatmap as a raster
        # image inside the PDF, which Illustrator may resample or import
        # incorrectly.
        #
        # pcolormesh creates true vector cells in the PDF.
        # ------------------------------------------------------------
        masked_vals=np.ma.masked_invalid(
            vals
        )

        im=ax.pcolormesh(
            np.arange(
                vals.shape[1]+1
            ),
            np.arange(
                vals.shape[0]+1
            ),
            masked_vals,
            cmap=cmap_obj,
            norm=norm,
            shading="flat",
            edgecolors="none",
            linewidth=0,
            rasterized=False,
        )

        # Match the original imshow orientation and rectangular geometry.
        ax.set_xlim(
            0,
            vals.shape[1],
        )

        ax.set_ylim(
            vals.shape[0],
            0,
        )

        ax.set_aspect(
            "auto"
        )

        # ------------------------------------------------------------
        # X labels
        #
        # pcolormesh coordinates are cell boundaries, so ticks are at
        # cell centers (+0.5).
        # ------------------------------------------------------------
        ax.set_xticks(
            np.arange(
                mat.shape[1]
            )
            +0.5
        )

        ax.set_xticklabels(
            mat.columns,
            fontsize=column_label_fontsize,
            fontfamily=default_font,
        )

        # ------------------------------------------------------------
        # Allele labels, e.g. A*02:01
        # ------------------------------------------------------------
        ax.set_yticks(
            np.arange(
                mat.shape[0]
            )
            +0.5
        )

        ax.set_yticklabels(
            mat.index,
            fontsize=allele_fontsize,
            fontfamily=default_font,
        )

        for tick in (
            ax.get_xticklabels()
            +ax.get_yticklabels()
        ):
            tick.set_fontfamily(
                default_font
            )

        for spine in ax.spines.values():
            spine.set_linewidth(
                spine_width
            )

        # ------------------------------------------------------------
        # Cell annotations
        #
        # pcolormesh cell centers are (j+0.5, i+0.5).
        # ------------------------------------------------------------
        if annotate:

            for i in range(
                vals.shape[0]
            ):

                for j in range(
                    vals.shape[1]
                ):

                    v=vals[
                        i,
                        j,
                    ]

                    if not np.isfinite(
                        v
                    ):
                        continue

                    ax.text(
                        j+0.5,
                        i+0.5,
                        f"{v:+.{annotation_decimals}f}",
                        ha="center",
                        va="center",
                        fontsize=annotation_fontsize,
                        fontfamily=default_font,
                        color=(
                            "white"
                            if abs(v)
                            >=annotation_switch_frac*vmax
                            else "black"
                        ),
                    )

        # ============================================================
        # Manual figure-coordinate layout
        # ============================================================
        pos=ax.get_position()

        # ------------------------------------------------------------
        # Dataset title
        # ------------------------------------------------------------
        title_pos_y=(
            pos.y1
            +dataset_title_gap
        )

        fig.text(
            (
                pos.x0
                +pos.x1
            )/2,
            title_pos_y,
            dataset_labels.get(
                dataset,
                dataset,
            ),
            ha="center",
            va="bottom",
            fontsize=dataset_title_fontsize,
            fontfamily=default_font,
        )

        # ------------------------------------------------------------
        # LOCAL horizontal colorbar
        #
        # Width exactly matches heatmap width.
        # Bar sits above heatmap.
        # Tick labels sit BELOW the bar.
        # ------------------------------------------------------------
        cbar_bottom=(
            pos.y1
            +colorbar_gap
        )

        cax=fig.add_axes([
            pos.x0,
            cbar_bottom,
            pos.width,
            colorbar_height,
        ])

        cb=fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            ticklocation="bottom",
        )

        # Explicitly force BOTH ticks and tick labels underneath.
        cb.ax.xaxis.set_ticks_position(
            "bottom"
        )

        cb.ax.xaxis.set_label_position(
            "bottom"
        )

        cb.ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
            labelsize=colorbar_ticksize,
            length=colorbar_tick_length,
            width=colorbar_tick_width,
            pad=colorbar_tick_pad,
        )

        for tick in cb.ax.get_xticklabels():
            tick.set_fontfamily(
                default_font
            )

            tick.set_fontsize(
                colorbar_ticksize
            )

        if colorbar_label:

            cb.set_label(
                colorbar_label,
                fontsize=colorbar_labelsize,
                labelpad=colorbar_labelpad,
                fontfamily=default_font,
            )

    # ================================================================
    # Figure-level title
    # ================================================================
    if title is None:

        title=(
            f"{method}: allele-aware sampling advantage"
        )

        if match_alleles:

            title+=(
                " (matched assay/MS alleles)"
            )

    fig.suptitle(
        title,
        fontsize=title_fontsize,
        y=title_y,
        fontfamily=default_font,
    )

    # ================================================================
    # Save
    # ================================================================
    if save_path is not None:

        base=str(
            save_path
        )

        # Allows:
        #   save_path="figure"
        #   save_path="figure.pdf"
        #   save_path="figure.png"
        #
        # while still producing the requested formats below.
        if base.lower().endswith(
            (
                ".pdf",
                ".png",
            )
        ):

            base=base.rsplit(
                ".",
                1,
            )[0]

        out_dpi=(
            dpi
            if save_dpi is None
            else save_dpi
        )

        if save_pdf:

            fig.savefig(
                f"{base}.pdf",
                format="pdf",
                dpi=out_dpi,
                bbox_inches=save_bbox_inches,
                pad_inches=save_pad_inches,
            )

        if save_png:

            fig.savefig(
                f"{base}.png",
                format="png",
                dpi=out_dpi,
                bbox_inches=save_bbox_inches,
                pad_inches=save_pad_inches,
            )

    if show:
        plt.show()

    return (
        heatmap_df,
        matrices,
        seed_df,
        fig,
        axes,
    )

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