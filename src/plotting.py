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
from matplotlib.lines import Line2D


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