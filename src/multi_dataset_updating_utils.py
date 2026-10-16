from sklearn.metrics import average_precision_score
from IPython.display import display

def analyze_multidataset_cl_results(
    test_sets,
    task_orders,
    method_suffixes,
    base_score_col="mhcprime",
    label_col="label",
    allele_col="allele",
    save_dir=None,
    dpi=300,
    ap_matrix_fn=None,
    show_main=True,
    show_supplementary=True,
    annotate_heatmaps=True,
):
    """
    Complete continual-learning analysis for the multi-dataset PepCL experiment.

    Expected model-score naming convention
    --------------------------------------
    Base:
        mhcprime

    Sequential checkpoints:
        {task_order_name}_s{stage}{method_suffix}

    Example:
        yd_episcan_ba_escape_s2_ft
        yd_episcan_ba_escape_s2_pepcl_replay_full_buffer

    Parameters
    ----------
    test_sets : dict[str, pd.DataFrame]
        Evaluation dataframes keyed by task:
            {"ms", "yd", "episcan", "ba", "escape"}

        Each dataframe must contain:
            allele_col
            label_col
            base_score_col
            all sequential model score columns.

    task_orders : dict[str, list[str]]
        Example:
            {
                "yd_episcan_ba_escape":
                    ["yd", "episcan", "ba", "escape"],
                ...
            }

    method_suffixes : dict[str, str]
        Short plotting name -> model-column suffix.

        Example:
            {
                "ft": "_ft",
                "pepcl_r": "_pepcl_replay_full_buffer",
                "pepcl_r_dr": "_pepcl_replay_dr_full_buffer",
                "pepcl_multi_r": "_pepcl_multi_replay_full_buffer",
                "pepcl_multi_r_dr": "_pepcl_multi_replay_dr_full_buffer",
            }

    base_score_col : str
        Base MHCPrime score column.

    save_dir : str or None
        If None, nothing is saved.
        Otherwise figures and tables are written here.

    ap_matrix_fn : callable or None
        Optionally provide your existing compute_ap_matrix_optimized().
        If None, the function will use compute_ap_matrix_optimized from
        globals() if available; otherwise it falls back to sklearn AP.

    Returns
    -------
    dict containing:
        allele_ap
        map_by_dataset
        performance_matrices
        stage_metrics
        final_metrics_by_order
        task_metrics
        position_metrics
        tables
    """

    # =========================================================
    # Helpers
    # =========================================================

    def _save_fig(fig, name):
        if save_dir is None:
            return
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches="tight")

    def _save_table(df, name, index=False):
        if save_dir is not None:
            df.to_csv(os.path.join(save_dir, f"{name}.csv"), index=index)

    def _fallback_ap_matrix(df, score_cols):
        rows = []
        grouped = list(df.groupby(allele_col, sort=False))

        for allele, g in tqdm(
            grouped,
            desc="    Allele AP",
            leave=True,
        ):
            y = g[label_col].to_numpy()
            row = {"allele": allele}

            for col in score_cols:
                # AP is undefined in a useful biological sense without positives.
                # Such alleles are excluded from the allele-macro average.
                if np.sum(y == 1) == 0:
                    row[col] = np.nan
                else:
                    row[col] = average_precision_score(y, g[col].to_numpy())

            rows.append(row)

        return pd.DataFrame(rows).set_index("allele")

    def _format_mean_sd(df, group_col, value_cols, digits=4):
        rows = []

        for name, g in df.groupby(group_col, sort=False):
            row = {group_col: name}

            for col in value_cols:
                vals = g[col].dropna()
                if len(vals) == 0:
                    row[col] = "NA"
                else:
                    row[col] = f"{vals.mean():.{digits}f} ± {vals.std(ddof=1):.{digits}f}"

            rows.append(row)

        return pd.DataFrame(rows).set_index(group_col)

    # =========================================================
    # Validation / setup
    # =========================================================

    print("=" * 80)
    print("MULTI-DATASET CONTINUAL-LEARNING ANALYSIS")
    print("=" * 80)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving outputs to: {save_dir}")

    dataset_names = list(test_sets.keys())
    method_names = list(method_suffixes.keys())
    n_stages = len(next(iter(task_orders.values())))

    if "ms" not in test_sets:
        raise ValueError("test_sets must contain the base MS evaluation set under key 'ms'.")

    for order_name, order in task_orders.items():
        if len(order) != n_stages:
            raise ValueError("All task orders must contain the same number of stages.")

        if len(set(order)) != len(order):
            raise ValueError(f"Task order '{order_name}' contains duplicate tasks.")

        missing = set(order) - set(test_sets)
        if missing:
            raise ValueError(f"Task order '{order_name}' contains tasks missing from test_sets: {missing}")

    # =========================================================
    # Determine all expected score columns
    # =========================================================

    print("\n[1/9] Building expected model-column list...")

    model_columns = {}

    for order_name, order in task_orders.items():
        model_columns[order_name] = {}

        for method, suffix in method_suffixes.items():
            cols = [base_score_col]
            cols += [f"{order_name}_s{stage}{suffix}" for stage in range(1, n_stages + 1)]
            model_columns[order_name][method] = cols

    all_score_cols = [base_score_col]

    for order_name in task_orders:
        for method in method_names:
            all_score_cols.extend(model_columns[order_name][method][1:])

    all_score_cols = list(dict.fromkeys(all_score_cols))

    print(f"  Task orders: {len(task_orders)}")
    print(f"  Methods: {len(method_names)}")
    print(f"  Sequential checkpoints expected: {len(all_score_cols) - 1}")
    print(f"  Total score columns including base: {len(all_score_cols)}")

    print("\nValidating score columns in all test sets...")

    for dataset_name, df in test_sets.items():
        missing = [c for c in all_score_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing {len(missing)} expected model columns.\n"
                f"First missing columns: {missing[:10]}"
            )

    # =========================================================
    # Compute allele-specific AP and mAP
    # =========================================================

    print("\n[2/9] Computing allele-specific AP and dataset mAP...")

    if ap_matrix_fn is None:
        ap_matrix_fn = globals().get("compute_ap_matrix_optimized", None)

    allele_ap = {}
    map_by_dataset = {}

    for dataset_name, df in tqdm(
        test_sets.items(),
        desc="Datasets",
        leave=True,
    ):
        print(f"\n  Computing AP: {dataset_name} | N={len(df):,} | alleles={df[allele_col].nunique():,}")

        if ap_matrix_fn is not None:
            try:
                ap_df = ap_matrix_fn(
                    df,
                    all_score_cols,
                    label_col=label_col,
                    allele_col=allele_col,
                    disable_tqdm=True,
                )
            except TypeError:
                ap_df = ap_matrix_fn(
                    df,
                    all_score_cols,
                    label_col=label_col,
                    allele_col=allele_col,
                )
        else:
            ap_df = _fallback_ap_matrix(df, all_score_cols)

        allele_ap[dataset_name] = ap_df
        map_by_dataset[dataset_name] = ap_df.mean(axis=0, skipna=True)

    map_df = pd.DataFrame(map_by_dataset)

    print("\n  Finished AP computation.")
    print(f"  mAP matrix shape: {map_df.shape}")

    # =========================================================
    # Build order × method performance matrices
    # =========================================================

    print("\n[3/9] Building task-performance matrices...")

    performance_matrices = {}

    for order_name, order in tqdm(
        task_orders.items(),
        desc="Performance matrices",
        leave=True,
    ):
        performance_matrices[order_name] = {}

        for method in method_names:
            cols = model_columns[order_name][method]

            perf = map_df.loc[cols, dataset_names].copy()
            perf.index = ["s0"] + [f"s{i}" for i in range(1, n_stages + 1)]

            performance_matrices[order_name][method] = perf

    # =========================================================
    # Stage-level CL metrics
    # =========================================================

    print("\n[4/9] Computing continual-learning metrics...")

    stage_rows = []
    task_rows = []

    for order_name, order in tqdm(
        task_orders.items(),
        desc="CL metrics",
        leave=True,
    ):
        full_order = ["ms"] + list(order)

        acquisition_stage = {"ms": 0}
        acquisition_stage.update({task: i for i, task in enumerate(order, start=1)})

        for method in method_names:
            perf = performance_matrices[order_name][method]

            for stage in range(0, n_stages + 1):
                current_task = full_order[stage]
                seen_tasks = full_order[:stage + 1]
                previous_tasks = full_order[:stage]

                seen_task_mean = perf.loc[f"s{stage}", seen_tasks].mean()
                all_domain_mean = perf.loc[f"s{stage}", dataset_names].mean()
                current_task_perf = perf.loc[f"s{stage}", current_task]

                if stage == 0:
                    current_task_gain = np.nan
                else:
                    current_task_gain = (
                        perf.loc[f"s{stage}", current_task]
                        - perf.loc[f"s{stage - 1}", current_task]
                    )

                ms_retention = perf.loc[f"s{stage}", "ms"] - perf.loc["s0", "ms"]

                bwt_vals = []
                forgetting_vals = []

                for previous_task in previous_tasks:
                    acquired = acquisition_stage[previous_task]

                    current_perf = perf.loc[f"s{stage}", previous_task]
                    acquisition_perf = perf.loc[f"s{acquired}", previous_task]

                    # BWT: current minus performance immediately after acquisition.
                    bwt = current_perf - acquisition_perf

                    # Forgetting: best PREVIOUS checkpoint since acquisition
                    # minus current checkpoint.
                    prior_stages = [f"s{i}" for i in range(acquired, stage)]
                    historical_best = perf.loc[prior_stages, previous_task].max()
                    forgetting = historical_best - current_perf

                    bwt_vals.append(bwt)
                    forgetting_vals.append(forgetting)

                    task_rows.append({
                        "order": order_name,
                        "method": method,
                        "stage": stage,
                        "current_task": current_task,
                        "task": previous_task,
                        "task_position": acquisition_stage[previous_task],
                        "acquisition_performance": acquisition_perf,
                        "current_performance": current_perf,
                        "historical_best_before_current": historical_best,
                        "backward_transfer": bwt,
                        "forgetting": forgetting,
                    })

                stage_rows.append({
                    "order": order_name,
                    "method": method,
                    "stage": stage,
                    "stage_name": f"s{stage}",
                    "current_task": current_task,
                    "seen_task_mean": seen_task_mean,
                    "all_domain_mean": all_domain_mean,
                    "current_task_performance": current_task_perf,
                    "current_task_gain": current_task_gain,
                    "backward_transfer": np.mean(bwt_vals) if bwt_vals else np.nan,
                    "average_forgetting": np.mean(forgetting_vals) if forgetting_vals else np.nan,
                    "maximum_forgetting": np.max(forgetting_vals) if forgetting_vals else np.nan,
                    "ms_retention": ms_retention,
                })

    stage_metrics = pd.DataFrame(stage_rows)
    task_stage_metrics = pd.DataFrame(task_rows)

    # =========================================================
    # Final/order-level metrics
    # =========================================================

    print("\n[5/9] Computing final retention and task-position statistics...")

    final_metrics_by_order = (
        stage_metrics[stage_metrics["stage"] == n_stages]
        .copy()
        .reset_index(drop=True)
    )

    task_final_rows = []

    for order_name, order in task_orders.items():
        acquisition_stage = {"ms": 0}
        acquisition_stage.update({task: i for i, task in enumerate(order, start=1)})

        for method in method_names:
            perf = performance_matrices[order_name][method]

            for task in ["ms"] + list(order):
                pos = acquisition_stage[task]
                acquisition_perf = perf.loc[f"s{pos}", task]
                final_perf = perf.loc[f"s{n_stages}", task]

                stages_from_acquisition = [f"s{i}" for i in range(pos, n_stages + 1)]
                historical_best = perf.loc[stages_from_acquisition, task].max()

                # Final BWT-like retention relative to acquisition.
                final_minus_acquisition = final_perf - acquisition_perf

                # Nonnegative final forgetting from best-ever checkpoint.
                final_forgetting = historical_best - final_perf

                task_final_rows.append({
                    "order": order_name,
                    "method": method,
                    "task": task,
                    "task_position": pos,
                    "n_subsequent_updates": n_stages - pos,
                    "acquisition_performance": acquisition_perf,
                    "final_performance": final_perf,
                    "final_minus_acquisition": final_minus_acquisition,
                    "historical_best": historical_best,
                    "final_forgetting": final_forgetting,
                })

    task_metrics = pd.DataFrame(task_final_rows)

    # Position analysis: non-MS tasks only.
    position_metrics = (
        task_metrics[task_metrics["task"] != "ms"]
        .groupby(["method", "task", "task_position"], as_index=False)
        .agg(
            n_orders=("order", "count"),
            acquisition_mean=("acquisition_performance", "mean"),
            acquisition_sd=("acquisition_performance", "std"),
            final_mean=("final_performance", "mean"),
            final_sd=("final_performance", "std"),
            retention_mean=("final_minus_acquisition", "mean"),
            retention_sd=("final_minus_acquisition", "std"),
            forgetting_mean=("final_forgetting", "mean"),
            forgetting_sd=("final_forgetting", "std"),
        )
    )

    # =========================================================
    # Tables
    # =========================================================

    print("\n[6/9] Constructing main and supplementary tables...")

    main_metric_cols = [
        "seen_task_mean",
        "backward_transfer",
        "maximum_forgetting",
        "ms_retention",
    ]

    # ---------- Main Table 1: aggregate final CL metrics ----------
    main_final_numeric = (
        final_metrics_by_order
        .groupby("method", as_index=False)
        .agg(
            final_performance_mean=("seen_task_mean", "mean"),
            final_performance_sd=("seen_task_mean", "std"),
            bwt_mean=("backward_transfer", "mean"),
            bwt_sd=("backward_transfer", "std"),
            max_forgetting_mean=("maximum_forgetting", "mean"),
            max_forgetting_sd=("maximum_forgetting", "std"),
            ms_retention_mean=("ms_retention", "mean"),
            ms_retention_sd=("ms_retention", "std"),
        )
    )

    main_final_formatted = pd.DataFrame(index=method_names)

    for method in method_names:
        row = main_final_numeric[main_final_numeric["method"] == method].iloc[0]

        main_final_formatted.loc[method, "Final mean performance ↑"] = (
            f"{row['final_performance_mean']:.4f} ± {row['final_performance_sd']:.4f}"
        )
        main_final_formatted.loc[method, "BWT ↑"] = (
            f"{row['bwt_mean']:.4f} ± {row['bwt_sd']:.4f}"
        )
        main_final_formatted.loc[method, "Maximum forgetting ↓"] = (
            f"{row['max_forgetting_mean']:.4f} ± {row['max_forgetting_sd']:.4f}"
        )
        main_final_formatted.loc[method, "MS retention ↑"] = (
            f"{row['ms_retention_mean']:.4f} ± {row['ms_retention_sd']:.4f}"
        )

    # ---------- Main Table 2: task-specific retention ----------
    nonms_retention = task_metrics[
        (task_metrics["task"] != "ms")
        & (task_metrics["task_position"] < n_stages)
    ].copy()

    task_retention_summary = (
        nonms_retention
        .groupby(["method", "task"], as_index=False)
        .agg(
            n_orders=("order", "count"),
            retention_mean=("final_minus_acquisition", "mean"),
            retention_sd=("final_minus_acquisition", "std"),
            forgetting_mean=("final_forgetting", "mean"),
            forgetting_sd=("final_forgetting", "std"),
        )
    )

    task_names = [x for x in dataset_names if x != "ms"]

    main_retention_formatted = pd.DataFrame(index=method_names)

    for method in method_names:
        for task in task_names:
            q = task_retention_summary[
                (task_retention_summary["method"] == method)
                & (task_retention_summary["task"] == task)
            ]

            if len(q):
                row = q.iloc[0]
                main_retention_formatted.loc[method, f"{task} retention ↑"] = (
                    f"{row['retention_mean']:.4f} ± {row['retention_sd']:.4f}"
                )

    # ---------- Supplement: order-level final metrics ----------
    supp_final_by_order = final_metrics_by_order[
        [
            "order", "method",
            "seen_task_mean", "all_domain_mean",
            "backward_transfer",
            "average_forgetting",
            "maximum_forgetting",
            "ms_retention",
        ]
    ].copy()

    # ---------- Supplement: complete stage-level metrics ----------
    supp_stage_metrics = stage_metrics.copy()

    # ---------- Supplement: task acquisition / final retention ----------
    supp_task_retention = task_metrics.copy()

    # ---------- Supplement: position summaries ----------
    supp_position_summary = position_metrics.copy()

    # ---------- Supplement: performance matrices in long form ----------
    matrix_rows = []

    for order_name in task_orders:
        for method in method_names:
            perf = performance_matrices[order_name][method]

            for stage_name, row in perf.iterrows():
                for task, value in row.items():
                    matrix_rows.append({
                        "order": order_name,
                        "method": method,
                        "stage": int(stage_name.replace("s", "")),
                        "evaluation_task": task,
                        "map": value,
                    })

    supp_performance_matrix_long = pd.DataFrame(matrix_rows)

    tables = {
        "main_final_numeric": main_final_numeric,
        "main_final_formatted": main_final_formatted,
        "main_task_retention_numeric": task_retention_summary,
        "main_task_retention_formatted": main_retention_formatted,
        "supp_final_by_order": supp_final_by_order,
        "supp_stage_metrics": supp_stage_metrics,
        "supp_task_retention": supp_task_retention,
        "supp_position_summary": supp_position_summary,
        "supp_performance_matrix_long": supp_performance_matrix_long,
    }

    # =========================================================
    # MAIN FIGURES
    # =========================================================

    print("\n[7/9] Creating main figures...")

    # ---------------------------------------------------------
    # MAIN FIGURE 1
    # Aggregate CL trajectories
    # ---------------------------------------------------------

    trajectory_specs = [
        ("seen_task_mean", "Seen-task average performance", "mAP"),
        ("all_domain_mean", "All-domain mean performance", "mAP"),
        ("backward_transfer", "Backward transfer", "BWT"),
        ("maximum_forgetting", "Maximum forgetting", "Forgetting"),
        ("ms_retention", "MS retention", "ΔmAP from base MS"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes, trajectory_specs):
        for method in method_names:
            sub = stage_metrics[stage_metrics["method"] == method]

            grouped = (
                sub.groupby("stage")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            x = grouped["stage"].to_numpy()
            y = grouped["mean"].to_numpy()
            sd = grouped["std"].to_numpy()

            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)

        if metric in {"backward_transfer", "maximum_forgetting", "ms_retention"}:
            ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_title(title)
        ax.set_xlabel("Sequential update stage")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(n_stages + 1))
        ax.set_xticklabels(["Base"] + [f"s{i}" for i in range(1, n_stages + 1)])

    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center")

    fig.suptitle("Continual-learning performance across task orders")
    fig.tight_layout()

    _save_fig(fig, "main_figure_1_cl_trajectories")

    if show_main:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # MAIN FIGURE 2
    # Final metric distributions across task orders
    # ---------------------------------------------------------

    final_specs = [
        ("seen_task_mean", "Final mean performance", "mAP"),
        ("backward_transfer", "Final backward transfer", "BWT"),
        ("maximum_forgetting", "Final maximum forgetting", "Forgetting"),
        ("ms_retention", "Final MS retention", "ΔmAP"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes, final_specs):
        values = [
            final_metrics_by_order.loc[
                final_metrics_by_order["method"] == method,
                metric
            ].dropna().to_numpy()
            for method in method_names
        ]

        ax.boxplot(values, tick_labels=method_names)

        # Raw order-level observations.
        for i, vals in enumerate(values, start=1):
            ax.scatter(np.full(len(vals), i), vals, alpha=0.7)

        if metric in {"backward_transfer", "maximum_forgetting", "ms_retention"}:
            ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Final continual-learning metrics across task orders")
    fig.tight_layout()

    _save_fig(fig, "main_figure_2_final_metrics")

    if show_main:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # MAIN FIGURE 3
    # Assay-specific retention
    #
    # Excludes instances where an assay was learned last because
    # those have zero subsequent opportunities for forgetting.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(task_names))
    width = 0.8 / max(len(method_names), 1)

    for i, method in enumerate(method_names):
        means, sds = [], []

        for task in task_names:
            q = task_retention_summary[
                (task_retention_summary["method"] == method)
                & (task_retention_summary["task"] == task)
            ]

            means.append(q["retention_mean"].iloc[0] if len(q) else np.nan)
            sds.append(q["retention_sd"].iloc[0] if len(q) else np.nan)

        xpos = x - 0.4 + width / 2 + i * width

        ax.errorbar(
            xpos,
            means,
            yerr=sds,
            marker="o",
            linestyle="none",
            capsize=3,
            label=method,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_xlabel("Previously learned assay")
    ax.set_ylabel("Final mAP − acquisition mAP")
    ax.set_title("Task-specific retention across task orders")
    ax.legend()

    fig.tight_layout()

    _save_fig(fig, "main_figure_3_task_retention")

    if show_main:
        plt.show()
    else:
        plt.close(fig)

    # =========================================================
    # SUPPLEMENTARY FIGURES
    # =========================================================

    print("\n[8/9] Creating supplementary figures...")

    # ---------------------------------------------------------
    # SUPPLEMENTARY FIGURE 1
    # Order-specific CL trajectories
    # 8 rows × 4 metrics
    # ---------------------------------------------------------

    supp_metrics = [
        ("seen_task_mean", "Seen-task mean"),
        ("backward_transfer", "BWT"),
        ("maximum_forgetting", "Max forgetting"),
        ("ms_retention", "MS retention"),
    ]

    n_orders = len(task_orders)

    fig, axes = plt.subplots(
        n_orders,
        len(supp_metrics),
        figsize=(18, max(4, 3 * n_orders)),
        squeeze=False,
    )

    for r, (order_name, order) in enumerate(task_orders.items()):
        for c, (metric, title) in enumerate(supp_metrics):
            ax = axes[r, c]

            for method in method_names:
                q = stage_metrics[
                    (stage_metrics["order"] == order_name)
                    & (stage_metrics["method"] == method)
                ]

                ax.plot(q["stage"], q[metric], marker="o", label=method)

            if metric in {"backward_transfer", "maximum_forgetting", "ms_retention"}:
                ax.axhline(0, linestyle="--", linewidth=1)

            ax.set_xticks(range(n_stages + 1))
            ax.set_xticklabels(["Base"] + order, rotation=30, ha="right")

            if r == 0:
                ax.set_title(title)

            if c == 0:
                ax.set_ylabel(order_name)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(method_names))
    fig.suptitle("Order-specific continual-learning trajectories", y=1.002)
    fig.tight_layout()

    _save_fig(fig, "supp_figure_1_order_specific_trajectories")

    if show_supplementary:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # SUPPLEMENTARY FIGURE 2
    # Full performance-matrix grid
    # rows = methods
    # columns = task orders
    # ---------------------------------------------------------

    all_matrix_values = []

    for order_name in task_orders:
        for method in method_names:
            all_matrix_values.extend(
                performance_matrices[order_name][method].to_numpy().ravel()
            )

    vmin = np.nanmin(all_matrix_values)
    vmax = np.nanmax(all_matrix_values)

    fig, axes = plt.subplots(
        len(method_names),
        len(task_orders),
        figsize=(4 * len(task_orders), 3.5 * len(method_names)),
        squeeze=False,
    )

    last_im = None

    for r, method in enumerate(method_names):
        for c, (order_name, order) in enumerate(task_orders.items()):
            ax = axes[r, c]
            perf = performance_matrices[order_name][method]

            last_im = ax.imshow(
                perf.to_numpy(),
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks(np.arange(len(dataset_names)))
            ax.set_xticklabels(dataset_names, rotation=45, ha="right")
            ax.set_yticks(np.arange(n_stages + 1))
            ax.set_yticklabels(["Base"] + order)

            if r == 0:
                ax.set_title(order_name)

            if c == 0:
                ax.set_ylabel(method)

            if annotate_heatmaps:
                for i in range(perf.shape[0]):
                    for j in range(perf.shape[1]):
                        ax.text(
                            j,
                            i,
                            f"{perf.iloc[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                        )

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.5, label="mAP")

    fig.suptitle("Complete continual-learning performance matrices")
    fig.subplots_adjust(top=0.94, wspace=0.35, hspace=0.55)

    _save_fig(fig, "supp_figure_2_performance_matrix_grid")

    if show_supplementary:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # SUPPLEMENTARY FIGURE 3
    # Task-position effect on retention
    #
    # Positions 1-3 only. Position 4 has no subsequent update,
    # therefore retention is mechanically zero at final acquisition.
    # ---------------------------------------------------------

    nonms_position = task_metrics[
        (task_metrics["task"] != "ms")
        & (task_metrics["task_position"] < n_stages)
    ].copy()

    ncols = 2
    nrows = math.ceil(len(task_names) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.5 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, task in zip(axes, task_names):
        q_task = nonms_position[nonms_position["task"] == task]

        for method in method_names:
            q = q_task[q_task["method"] == method]

            grouped = (
                q.groupby("task_position")["final_minus_acquisition"]
                .agg(["mean", "std"])
                .reset_index()
            )

            ax.errorbar(
                grouped["task_position"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                label=method,
            )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(task)
        ax.set_xlabel("Position learned in sequence")
        ax.set_ylabel("Final − acquisition mAP")
        ax.set_xticks(range(1, n_stages))

    for ax in axes[len(task_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(method_names))
    fig.suptitle("Effect of task position on retention")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _save_fig(fig, "supp_figure_3_task_position_retention")

    if show_supplementary:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # SUPPLEMENTARY FIGURE 4
    # Task-position effect on acquisition/plasticity
    # Includes positions 1-4.
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.5 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, task in zip(axes, task_names):
        q_task = task_metrics[
            (task_metrics["task"] == task)
            & (task_metrics["task"] != "ms")
        ]

        for method in method_names:
            q = q_task[q_task["method"] == method]

            grouped = (
                q.groupby("task_position")["acquisition_performance"]
                .agg(["mean", "std"])
                .reset_index()
            )

            ax.errorbar(
                grouped["task_position"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                label=method,
            )

        ax.set_title(task)
        ax.set_xlabel("Position learned in sequence")
        ax.set_ylabel("mAP immediately after acquisition")
        ax.set_xticks(range(1, n_stages + 1))

    for ax in axes[len(task_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(method_names))
    fig.suptitle("Effect of task position on acquisition performance")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _save_fig(fig, "supp_figure_4_task_position_acquisition")

    if show_supplementary:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # SUPPLEMENTARY FIGURE 5
    # Average forgetting trajectory
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in method_names:
        q = stage_metrics[stage_metrics["method"] == method]

        grouped = (
            q.groupby("stage")["average_forgetting"]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.plot(grouped["stage"], grouped["mean"], marker="o", label=method)
        ax.fill_between(
            grouped["stage"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(range(n_stages + 1))
    ax.set_xticklabels(["Base"] + [f"s{i}" for i in range(1, n_stages + 1)])
    ax.set_xlabel("Sequential update stage")
    ax.set_ylabel("Average forgetting")
    ax.set_title("Average forgetting across task orders")
    ax.legend()

    fig.tight_layout()

    _save_fig(fig, "supp_figure_5_average_forgetting")

    if show_supplementary:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------
    # SUPPLEMENTARY FIGURE 6
    # Task-specific final forgetting
    # Excludes last-position tasks.
    # ---------------------------------------------------------

    forgetting_summary = (
        nonms_retention
        .groupby(["method", "task"], as_index=False)
        .agg(
            forgetting_mean=("final_forgetting", "mean"),
            forgetting_sd=("final_forgetting", "std"),
        )
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(task_names))
    width = 0.8 / max(len(method_names), 1)

    for i, method in enumerate(method_names):
        means, sds = [], []

        for task in task_names:
            q = forgetting_summary[
                (forgetting_summary["method"] == method)
                & (forgetting_summary["task"] == task)
            ]

            means.append(q["forgetting_mean"].iloc[0] if len(q) else np.nan)
            sds.append(q["forgetting_sd"].iloc[0] if len(q) else np.nan)

        xpos = x - 0.4 + width / 2 + i * width

        ax.errorbar(
            xpos,
            means,
            yerr=sds,
            marker="o",
            linestyle="none",
            capsize=3,
            label=method,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_xlabel("Previously learned assay")
    ax.set_ylabel("Final forgetting")
    ax.set_title("Task-specific forgetting across task orders")
    ax.legend()

    fig.tight_layout()

    _save_fig(fig, "supp_figure_6_task_specific_forgetting")

    if show_supplementary:
        plt.show()
    else:
        plt.close(fig)

    # =========================================================
    # Display / save tables
    # =========================================================

    print("\n[9/9] Displaying tables...")

    print("\nMAIN TABLE 1 — Final continual-learning performance across task orders")
    display(main_final_formatted)

    print("\nMAIN TABLE 2 — Task-specific retention")
    print("Note: task-order instances where the assay was learned last are excluded.")
    display(main_retention_formatted)

    print("\nSUPPLEMENTARY TABLE 1 — Final metrics for every method × task order")
    display(supp_final_by_order.round(4))

    print("\nSUPPLEMENTARY TABLE 2 — Complete stage-level continual-learning metrics")
    display(supp_stage_metrics.round(4))

    print("\nSUPPLEMENTARY TABLE 3 — Task acquisition and final retention")
    display(supp_task_retention.round(4))

    print("\nSUPPLEMENTARY TABLE 4 — Task-position analysis")
    display(supp_position_summary.round(4))

    if save_dir is not None:
        print("\nSaving tables...")

        _save_table(main_final_numeric, "main_table_1_final_metrics_numeric")
        _save_table(main_final_formatted.reset_index(), "main_table_1_final_metrics_formatted")

        _save_table(task_retention_summary, "main_table_2_task_retention_numeric")
        _save_table(main_retention_formatted.reset_index(), "main_table_2_task_retention_formatted")

        _save_table(supp_final_by_order, "supp_table_1_final_metrics_by_order")
        _save_table(supp_stage_metrics, "supp_table_2_stage_metrics")
        _save_table(supp_task_retention, "supp_table_3_task_retention")
        _save_table(supp_position_summary, "supp_table_4_task_position")
        _save_table(supp_performance_matrix_long, "supp_performance_matrices_long")

        # Raw allele-specific AP tables.
        allele_dir = os.path.join(save_dir, "allele_ap")
        os.makedirs(allele_dir, exist_ok=True)

        for dataset_name, ap_df in allele_ap.items():
            ap_df.to_csv(os.path.join(allele_dir, f"{dataset_name}_allele_ap.csv"))

        map_df.to_csv(os.path.join(save_dir, "all_dataset_map.csv"))

        print("Saving complete.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        "allele_ap": allele_ap,
        "map_by_dataset": map_df,
        "model_columns": model_columns,
        "performance_matrices": performance_matrices,
        "stage_metrics": stage_metrics,
        "task_stage_metrics": task_stage_metrics,
        "final_metrics_by_order": final_metrics_by_order,
        "task_metrics": task_metrics,
        "position_metrics": position_metrics,
        "tables": tables,
    }


import warnings

def merge_test_set_dicts(test_set_dicts, key_col="seq_allele"):
    """
    Merge N dictionaries of matched test DataFrames.

    Duplicate key_col rows are reduced to the first occurrence before merging.
    """
    if len(test_set_dicts) < 1:
        raise ValueError("test_set_dicts must contain at least one dictionary.")

    reference_keys = set(test_set_dicts[0].keys())

    for i, d in enumerate(test_set_dicts[1:], start=2):
        if set(d.keys()) != reference_keys:
            raise ValueError(
                f"Dictionary {i} has different dataset keys.\n"
                f"Expected: {sorted(reference_keys)}\n"
                f"Found: {sorted(d.keys())}"
            )

    merged_test_sets = {}

    for test_set_name in tqdm(
        test_set_dicts[0].keys(),
        desc="Merging test-set dictionaries",
        leave=True,
    ):
        print(f"\nMerging: {test_set_name}")

        base_df = test_set_dicts[0][test_set_name].copy()

        if key_col not in base_df.columns:
            raise ValueError(f"{test_set_name}: '{key_col}' missing from dictionary 1.")

        # keep one copy of duplicated peptide-allele samples
        n_dup = base_df.duplicated(subset=key_col).sum()
        if n_dup > 0:
            print(f"  Dictionary 1: dropping {n_dup:,} duplicate {key_col} rows.")
            base_df = base_df.drop_duplicates(subset=key_col, keep="first").reset_index(drop=True)

        base_keys = set(base_df[key_col])

        for dict_idx, test_dict in enumerate(test_set_dicts[1:], start=2):
            other_df = test_dict[test_set_name].copy()

            if key_col not in other_df.columns:
                raise ValueError(
                    f"{test_set_name}: '{key_col}' missing from dictionary {dict_idx}."
                )

            n_dup = other_df.duplicated(subset=key_col).sum()
            if n_dup > 0:
                print(
                    f"  Dictionary {dict_idx}: dropping {n_dup:,} "
                    f"duplicate {key_col} rows."
                )
                other_df = other_df.drop_duplicates(
                    subset=key_col,
                    keep="first"
                ).reset_index(drop=True)

            other_keys = set(other_df[key_col])

            if base_keys != other_keys:
                warnings.warn(
                    f"{test_set_name}: sample mismatch for dictionary {dict_idx}. "
                    f"Missing from incoming DF: {len(base_keys - other_keys)}; "
                    f"extra in incoming DF: {len(other_keys - base_keys)}."
                )

            # only add columns not already present
            new_cols = [
                col for col in other_df.columns
                if col == key_col or col not in base_df.columns
            ]

            base_df = base_df.merge(
                other_df[new_cols],
                on=key_col,
                how="inner",
                validate="one_to_one",
            )

            print(
                f"  Dict {dict_idx}: merged | "
                f"rows={len(base_df):,} | cols={base_df.shape[1]:,}"
            )

        merged_test_sets[test_set_name] = base_df

    print("\nFinished merging test-set dictionaries.")
    return merged_test_sets




import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score


def analyze_multidataset_matched_group_results(
    test_sets, task_orders, method_suffixes,
    base_score_col="mhcprime", label_col="label", allele_col="allele", seq_col="seq", pad_token="$",
    match_modes=("allele","allele_len"), save_dir=None, dpi=300, show_plots=True,
    plot_group_transfer_methods=None, max_group_rows=20, annotate_heatmaps=True,
    verbose=True,
):
    """
    Exploratory one-seed matched-group analysis for sequential multi-dataset CL.

    Matching membership is defined ONLY from group presence in each test dataframe,
    independent of positive/negative labels. Performance is AP within each matched
    group, macro-averaged across groups with a defined AP.

    match_modes:
      - "allele": exact allele match.
      - "allele_len": exact allele x unpadded peptide-length match.

    Main analyses
    -------------
    1) Pairwise matching coverage across datasets.
    2) Immediate acquisition effect for each assay, evaluated on groups shared with MS:
         assay change after update, matched-MS change after update,
         assay gain vs MHCPrime, matched-MS change vs MHCPrime.
    3) Cross-assay transfer after every update:
         source-task update -> change on every other assay over groups shared by
         source and target, separated into previously-seen vs future-unseen targets.
    4) Group-level transfer summaries identifying alleles / allele-length groups
       associated with positive or negative cross-assay transfer.

    Expected score naming follows analyze_multidataset_cl_results():
        {order_name}_s{stage}{method_suffix}

    Returns
    -------
    dict with:
      group_membership, group_ap, matching_summary,
      matched_ms_update_effects, matched_ms_update_summary,
      cross_transfer_events, cross_transfer_summary,
      cross_transfer_group_level, cross_transfer_group_summary,
      figures
    """
    valid_modes={"allele","allele_len"}
    if not set(match_modes).issubset(valid_modes):
        raise ValueError(f"match_modes must contain only {valid_modes}.")
    if "ms" not in test_sets:
        raise ValueError("test_sets must contain the MS dataframe under key 'ms'.")
    if max_group_rows<1:
        raise ValueError("max_group_rows must be >= 1.")

    dataset_names=list(test_sets.keys())
    assay_names=[x for x in dataset_names if x!="ms"]
    method_names=list(method_suffixes.keys())
    n_stages=len(next(iter(task_orders.values())))

    for order_name,order in task_orders.items():
        if len(order)!=n_stages: raise ValueError("All task orders must have the same length.")
        if len(set(order))!=len(order): raise ValueError(f"Task order '{order_name}' contains duplicate tasks.")
        missing=set(order)-set(test_sets)
        if missing: raise ValueError(f"Task order '{order_name}' contains tasks missing from test_sets: {missing}")

    def _print(msg=""):
        if verbose: print(msg)

    def _save(fig,name):
        if save_dir is None: return
        os.makedirs(save_dir,exist_ok=True)
        fig.savefig(os.path.join(save_dir,f"{name}.png"),dpi=dpi,bbox_inches="tight")
        fig.savefig(os.path.join(save_dir,f"{name}.pdf"),bbox_inches="tight")

    def _score_col(order_name,stage,method):
        return base_score_col if stage==0 else f"{order_name}_s{stage}{method_suffixes[method]}"

    def _group_display(mode,key):
        if mode=="allele": return str(key)
        if isinstance(key,tuple): return f"{key[0]}|{int(key[1])}"
        return str(key)

    def _shared_keys(membership,mode,a,b):
        return membership[(mode,a)]["groups"] & membership[(mode,b)]["groups"]

    def _macro_pair(ap_tables,mode,a,b,col_a_before,col_a_after,col_b_before,col_b_after,keys):
        """Use one common finite group set across both datasets and both checkpoints."""
        if not keys:
            return dict(n_groups=0,a_before=np.nan,a_after=np.nan,b_before=np.nan,b_after=np.nan)
        aa=ap_tables[(mode,a)].reindex(list(keys))
        bb=ap_tables[(mode,b)].reindex(list(keys))
        needed_a=aa[[col_a_before,col_a_after]].replace([np.inf,-np.inf],np.nan)
        needed_b=bb[[col_b_before,col_b_after]].replace([np.inf,-np.inf],np.nan)
        valid=needed_a.notna().all(axis=1) & needed_b.notna().all(axis=1)
        aa=aa.loc[valid]; bb=bb.loc[valid]
        return dict(
            n_groups=int(valid.sum()),
            a_before=float(aa[col_a_before].mean()) if len(aa) else np.nan,
            a_after=float(aa[col_a_after].mean()) if len(aa) else np.nan,
            b_before=float(bb[col_b_before].mean()) if len(bb) else np.nan,
            b_after=float(bb[col_b_after].mean()) if len(bb) else np.nan,
        )

    def _aggregate_mean_sd(df,group_cols,value_cols):
        out_cols=list(group_cols)
        for c in value_cols: out_cols += [f"{c}_mean",f"{c}_sd"]
        out_cols += ["n_events"]
        if df.empty: return pd.DataFrame(columns=out_cols)
        missing=[c for c in list(group_cols)+list(value_cols) if c not in df.columns]
        if missing: raise KeyError(f"Aggregation input missing columns: {missing}")
        named={}
        for c in value_cols:
            named[f"{c}_mean"]=(c,"mean")
            named[f"{c}_sd"]=(c,"std")
        named["n_events"]=(value_cols[0],"count")
        return df.groupby(group_cols,as_index=False,dropna=False).agg(**named)

    _print("="*88)
    _print("MATCHED ALLELE / ALLELE×LENGTH CONTINUAL-LEARNING ANALYSIS")
    _print("="*88)
    _print(f"Datasets: {dataset_names}")
    _print(f"Task orders: {len(task_orders)} | Methods: {len(method_names)} | Stages: {n_stages}")
    _print(f"Matching modes: {list(match_modes)}")

    # ------------------------------------------------------------------
    # Expected score columns + validation.
    # ------------------------------------------------------------------
    _print("\n[1/7] Validating score columns and building lightweight group metadata...")
    all_score_cols=[base_score_col]
    for order_name in task_orders:
        for method in method_names:
            all_score_cols += [_score_col(order_name,s,method) for s in range(1,n_stages+1)]
    all_score_cols=list(dict.fromkeys(all_score_cols))

    group_membership={}
    seq_len_by_dataset={}
    for ds,df in test_sets.items():
        missing=[c for c in [allele_col,label_col,seq_col]+all_score_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset '{ds}' is missing required columns. First missing: {missing[:10]}")
        if df[seq_col].isna().any():
            raise ValueError(f"Dataset '{ds}' contains missing values in '{seq_col}'.")

        # Remove all PAD characters before measuring peptide length.
        seq_len=df[seq_col].astype(str).str.replace(pad_token,"",regex=False).str.len().astype(np.int16)
        seq_len_by_dataset[ds]=seq_len
        _print(f"  {ds}: N={len(df):,} | alleles={df[allele_col].nunique():,} | lengths={sorted(seq_len.unique())}")

        alleles=df[allele_col].astype(str)
        if "allele" in match_modes:
            groups=set(alleles.unique().tolist())
            counts=alleles.value_counts()
            group_membership[("allele",ds)]={"groups":groups,"counts":counts}
        if "allele_len" in match_modes:
            keys=pd.MultiIndex.from_arrays([alleles.to_numpy(),seq_len.to_numpy()],names=[allele_col,"seq_len"])
            ser=pd.Series(list(keys),index=df.index)
            counts=ser.value_counts()
            group_membership[("allele_len",ds)]={"groups":set(counts.index.tolist()),"counts":counts}

    # ------------------------------------------------------------------
    # Pairwise matching coverage.
    # ------------------------------------------------------------------
    _print("\n[2/7] Summarizing pairwise matching coverage...")
    match_rows=[]
    for mode in match_modes:
        for i,a in enumerate(dataset_names):
            for b in dataset_names[i+1:]:
                ma=group_membership[(mode,a)]; mb=group_membership[(mode,b)]
                shared=ma["groups"] & mb["groups"]
                ca,cb=ma["counts"],mb["counts"]
                rows_a=int(sum(int(ca.get(g,0)) for g in shared)); rows_b=int(sum(int(cb.get(g,0)) for g in shared))
                match_rows.append({
                    "match_mode":mode,"dataset_a":a,"dataset_b":b,
                    "n_groups_a":len(ma["groups"]),"n_groups_b":len(mb["groups"]),"n_shared_groups":len(shared),
                    "shared_fraction_a":len(shared)/len(ma["groups"]) if ma["groups"] else np.nan,
                    "shared_fraction_b":len(shared)/len(mb["groups"]) if mb["groups"] else np.nan,
                    "rows_a_in_shared_groups":rows_a,"rows_b_in_shared_groups":rows_b,
                    "row_fraction_a":rows_a/len(test_sets[a]) if len(test_sets[a]) else np.nan,
                    "row_fraction_b":rows_b/len(test_sets[b]) if len(test_sets[b]) else np.nan,
                })
    matching_summary=pd.DataFrame(match_rows)

    # ------------------------------------------------------------------
    # AP by allele and allele×length for every score column.
    # ------------------------------------------------------------------
    _print("\n[3/7] Computing group-level AP matrices...")
    _print("      Group membership ignores labels; AP is then evaluated within each matched group.")
    group_ap={}
    for mode in match_modes:
        for ds,df in test_sets.items():
            _print(f"\n  {mode}: {ds}")
            if mode=="allele":
                group_values=df[allele_col].astype(str)
            else:
                group_values=pd.Series(
                    list(zip(df[allele_col].astype(str).to_numpy(),seq_len_by_dataset[ds].to_numpy())),
                    index=df.index,
                )

            grouped=group_values.groupby(group_values,sort=False).groups
            rows=[]
            for key,idx in tqdm(grouped.items(),desc=f"AP {ds} [{mode}]",leave=True):
                g=df.loc[idx]
                y=g[label_col].to_numpy()
                row={"group":key,"n_rows":len(g),"n_pos":int(np.sum(y==1)),"n_neg":int(np.sum(y!=1))}
                if row["n_pos"]==0:
                    for c in all_score_cols: row[c]=np.nan
                else:
                    for c in all_score_cols:
                        row[c]=average_precision_score(y,g[c].to_numpy())
                rows.append(row)
            ap=pd.DataFrame(rows).set_index("group")
            group_ap[(mode,ds)]=ap
            _print(f"    groups={len(ap):,} | groups with >=1 positive={(ap['n_pos']>0).sum():,}")

    # ------------------------------------------------------------------
    # Immediate acquisition effects on current assay + matched MS.
    # ------------------------------------------------------------------
    _print("\n[4/7] Computing immediate assay adaptation and matched-MS effects...")
    update_rows=[]
    for order_name,order in tqdm(task_orders.items(),desc="Task orders: matched MS",leave=True):
        for stage,current_task in enumerate(order,start=1):
            for method in method_names:
                before_col=_score_col(order_name,stage-1,method)
                after_col=_score_col(order_name,stage,method)
                for mode in match_modes:
                    keys=_shared_keys(group_membership,mode,current_task,"ms")
                    z=_macro_pair(group_ap,mode,current_task,"ms",before_col,after_col,before_col,after_col,keys)

                    # Also evaluate after vs original MHCPrime on the exact same finite group set.
                    zbase=_macro_pair(group_ap,mode,current_task,"ms",base_score_col,after_col,base_score_col,after_col,keys)
                    update_rows.append({
                        "order":order_name,"method":method,"stage":stage,"task":current_task,"match_mode":mode,
                        "n_shared_groups_raw":len(keys),"n_eval_groups":z["n_groups"],
                        "assay_before":z["a_before"],"assay_after":z["a_after"],
                        "assay_immediate_gain":z["a_after"]-z["a_before"],
                        "ms_before":z["b_before"],"ms_after":z["b_after"],
                        "ms_immediate_change":z["b_after"]-z["b_before"],
                        "assay_gain_vs_base":zbase["a_after"]-zbase["a_before"],
                        "ms_change_vs_base":zbase["b_after"]-zbase["b_before"],
                    })
    matched_ms_update_effects=pd.DataFrame(update_rows)
    matched_ms_update_summary=_aggregate_mean_sd(
        matched_ms_update_effects,["match_mode","method","task"],
        ["assay_immediate_gain","ms_immediate_change","assay_gain_vs_base","ms_change_vs_base","n_eval_groups"],
    )

    # ------------------------------------------------------------------
    # Cross-assay transfer at each update.
    # ------------------------------------------------------------------
    _print("\n[5/7] Computing cross-assay transfer on shared groups...")
    event_rows=[]; group_rows=[]
    iterator=list(task_orders.items())
    for order_name,order in tqdm(iterator,desc="Task orders: cross-assay",leave=True):
        for stage,source_task in enumerate(order,start=1):
            seen_before=set(order[:stage-1])
            before_col_by_method={m:_score_col(order_name,stage-1,m) for m in method_names}
            after_col_by_method={m:_score_col(order_name,stage,m) for m in method_names}

            for target_task in assay_names:
                if target_task==source_task: continue
                target_status="previously_seen" if target_task in seen_before else "future_unseen"

                for mode in match_modes:
                    keys=_shared_keys(group_membership,mode,source_task,target_task)
                    if not keys: continue
                    src_ap=group_ap[(mode,source_task)].reindex(list(keys))
                    tgt_ap=group_ap[(mode,target_task)].reindex(list(keys))

                    for method in method_names:
                        before_col=before_col_by_method[method]; after_col=after_col_by_method[method]
                        vals=pd.DataFrame({
                            "source_before":src_ap[before_col],"source_after":src_ap[after_col],
                            "target_before":tgt_ap[before_col],"target_after":tgt_ap[after_col],
                        })
                        valid=vals.notna().all(axis=1)
                        vals=vals.loc[valid]
                        if vals.empty: continue
                        vals["source_delta"]=vals["source_after"]-vals["source_before"]
                        vals["target_delta"]=vals["target_after"]-vals["target_before"]

                        event_rows.append({
                            "order":order_name,"method":method,"stage":stage,"source_task":source_task,
                            "target_task":target_task,"target_status":target_status,"match_mode":mode,
                            "n_shared_groups_raw":len(keys),"n_eval_groups":len(vals),
                            "source_gain":float(vals["source_delta"].mean()),
                            "target_change":float(vals["target_delta"].mean()),
                        })

                        for key,r in vals.iterrows():
                            allele=key if mode=="allele" else key[0]
                            length=np.nan if mode=="allele" else int(key[1])
                            group_rows.append({
                                "order":order_name,"method":method,"stage":stage,"source_task":source_task,
                                "target_task":target_task,"target_status":target_status,"match_mode":mode,
                                "group":_group_display(mode,key),"allele":str(allele),"seq_len":length,
                                "source_delta":float(r["source_delta"]),"target_delta":float(r["target_delta"]),
                            })

    cross_transfer_events=pd.DataFrame(event_rows)
    cross_transfer_group_level=pd.DataFrame(group_rows)
    cross_transfer_summary=_aggregate_mean_sd(
        cross_transfer_events,
        ["match_mode","method","source_task","target_task","target_status"],
        ["source_gain","target_change","n_eval_groups"],
    )

    expected_cross_summary={
        "match_mode","method","source_task","target_task","target_status",
        "source_gain_mean","source_gain_sd","target_change_mean","target_change_sd",
        "n_eval_groups_mean","n_eval_groups_sd","n_events",
    }
    missing_cross_summary=expected_cross_summary-set(cross_transfer_summary.columns)
    if missing_cross_summary:
        raise RuntimeError(f"Internal cross-transfer summary schema mismatch: {sorted(missing_cross_summary)}")

    if cross_transfer_group_level.empty:
        cross_transfer_group_summary=pd.DataFrame()
    else:
        cross_transfer_group_summary=(
            cross_transfer_group_level.groupby(
                ["match_mode","method","source_task","target_task","target_status","group","allele","seq_len"],
                as_index=False,dropna=False,
            ).agg(
                n_events=("target_delta","count"),
                source_delta_mean=("source_delta","mean"),source_delta_sd=("source_delta","std"),
                target_delta_mean=("target_delta","mean"),target_delta_sd=("target_delta","std"),
            )
        )

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------
    _print("\n[6/7] Plotting exploratory matched-group analyses...")
    figures={}

    # Fig 1: MS matching coverage.
    fig,axes=plt.subplots(1,len(match_modes),figsize=(6*len(match_modes),4.5),dpi=dpi,squeeze=False)
    for ax,mode in zip(axes.ravel(),match_modes):
        d=matching_summary[(matching_summary["match_mode"]==mode) & ((matching_summary["dataset_a"]=="ms") | (matching_summary["dataset_b"]=="ms"))].copy()
        d["assay"]=np.where(d["dataset_a"]=="ms",d["dataset_b"],d["dataset_a"])
        d=d.set_index("assay").reindex(assay_names).reset_index()
        ax.bar(d["assay"],d["n_shared_groups"])
        ax.set_title(f"MS matching coverage: {mode}")
        ax.set_ylabel("Shared groups")
        ax.tick_params(axis="x",rotation=30)
    fig.suptitle("Matched evaluation-group coverage")
    fig.tight_layout()
    figures["matching_coverage"]=fig; _save(fig,"matched_group_1_coverage")
    if show_plots: plt.show()
    else: plt.close(fig)

    # Fig 2: immediate current-assay adaptation vs matched-MS change.
    fig,axes=plt.subplots(1,len(match_modes),figsize=(6.5*len(match_modes),5.5),dpi=dpi,squeeze=False)
    for ax,mode in zip(axes.ravel(),match_modes):
        d=matched_ms_update_summary[matched_ms_update_summary["match_mode"]==mode]
        for method in method_names:
            q=d[d["method"]==method]
            if q.empty: continue
            ax.errorbar(
                q["ms_immediate_change_mean"],q["assay_immediate_gain_mean"],
                xerr=q["ms_immediate_change_sd"],yerr=q["assay_immediate_gain_sd"],
                marker="o",linestyle="none",capsize=3,label=method,
            )
            for _,r in q.iterrows():
                ax.annotate(r["task"],(r["ms_immediate_change_mean"],r["assay_immediate_gain_mean"]),fontsize=7,xytext=(3,3),textcoords="offset points")
        ax.axvline(0,ls="--",lw=1); ax.axhline(0,ls="--",lw=1)
        ax.set_xlabel("Immediate matched-MS ΔmAP")
        ax.set_ylabel("Immediate assay ΔmAP")
        ax.set_title(f"Immediate update effect: {mode}")
    handles,labels=axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=min(len(method_names),4),frameon=False)
    fig.suptitle("Matched adaptation-retention immediately after acquisition")
    fig.tight_layout(rect=[0,0.10,1,0.95])
    figures["matched_ms_immediate_effects"]=fig; _save(fig,"matched_group_2_immediate_update_effects")
    if show_plots: plt.show()
    else: plt.close(fig)

    # Fig 3+: cross-assay transfer matrices, rows=methods, cols=seen/unseen.
    task_index={x:i for i,x in enumerate(assay_names)}
    for mode in match_modes:
        fig,axes=plt.subplots(len(method_names),2,figsize=(9,max(3.2,3.0*len(method_names))),dpi=dpi,squeeze=False)
        finite_vals=cross_transfer_summary.loc[cross_transfer_summary["match_mode"]==mode,"target_change_mean"].to_numpy(float)
        finite_vals=finite_vals[np.isfinite(finite_vals)]
        vmax=max(abs(finite_vals).max(),1e-6) if len(finite_vals) else 1.0
        last_im=None
        for r,method in enumerate(method_names):
            for c,status in enumerate(["previously_seen","future_unseen"]):
                ax=axes[r,c]
                q=cross_transfer_summary[
                    (cross_transfer_summary["match_mode"]==mode)&
                    (cross_transfer_summary["method"]==method)&
                    (cross_transfer_summary["target_status"]==status)
                ]
                mat=np.full((len(assay_names),len(assay_names)),np.nan)
                for _,x in q.iterrows():
                    mat[task_index[x["source_task"]],task_index[x["target_task"]]]=x["target_change_mean"]
                last_im=ax.imshow(mat,vmin=-vmax,vmax=vmax,cmap="coolwarm",aspect="auto")
                ax.set_xticks(range(len(assay_names))); ax.set_xticklabels(assay_names,rotation=35,ha="right")
                ax.set_yticks(range(len(assay_names))); ax.set_yticklabels(assay_names)
                if r==0: ax.set_title("Previously seen target" if status=="previously_seen" else "Future/unseen target")
                if c==0: ax.set_ylabel(f"{method}\nUpdate source")
                if annotate_heatmaps:
                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            if np.isfinite(mat[i,j]): ax.text(j,i,f"{mat[i,j]:+.3f}",ha="center",va="center",fontsize=7)
        if last_im is not None: fig.colorbar(last_im,ax=axes.ravel().tolist(),shrink=.65,label="Target ΔmAP after source update")
        fig.suptitle(f"Cross-assay transfer on shared {mode.replace('_',' × ')} groups")
        fig.subplots_adjust(top=.95,wspace=.35,hspace=.45)
        figures[f"cross_transfer_{mode}"]=fig; _save(fig,f"matched_group_3_cross_transfer_{mode}")
        if show_plots: plt.show()
        else: plt.close(fig)

    # Group-specific directed transfer patterns. One figure per selected method.
    detail_methods=method_names if plot_group_transfer_methods is None else list(plot_group_transfer_methods)
    if not cross_transfer_group_summary.empty:
        pair_order=[f"{s}→{t}" for s in assay_names for t in assay_names if s!=t]
        for method in detail_methods:
            if method not in method_names: continue
            fig,axes=plt.subplots(1,len(match_modes),figsize=(8*len(match_modes),max(5,0.32*max_group_rows+2)),dpi=dpi,squeeze=False)
            for ax,mode in zip(axes.ravel(),match_modes):
                q=cross_transfer_group_summary[
                    (cross_transfer_group_summary["method"]==method)&
                    (cross_transfer_group_summary["match_mode"]==mode)
                ].copy()
                if q.empty:
                    ax.axis("off"); continue
                q["pair"]=q["source_task"]+"→"+q["target_task"]
                rank=(q.groupby("group").agg(n_events=("n_events","sum"),n_pairs=("pair","nunique"))
                      .sort_values(["n_pairs","n_events"],ascending=False).head(max_group_rows))
                groups=rank.index.tolist()
                mat=q[q["group"].isin(groups)].pivot_table(index="group",columns="pair",values="target_delta_mean",aggfunc="mean")
                mat=mat.reindex(index=groups,columns=[p for p in pair_order if p in mat.columns])
                arr=mat.to_numpy(float); vals=arr[np.isfinite(arr)]
                vmax=max(abs(vals).max(),1e-6) if len(vals) else 1.0
                im=ax.imshow(arr,vmin=-vmax,vmax=vmax,cmap="coolwarm",aspect="auto")
                ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index,fontsize=7)
                ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns,rotation=45,ha="right",fontsize=7)
                ax.set_title(f"{mode}: top shared groups")
                ax.set_xlabel("Update source → evaluated target")
                ax.set_ylabel("Matched group")
                fig.colorbar(im,ax=ax,shrink=.7,label="Mean target ΔmAP")
            fig.suptitle(f"Group-specific cross-assay transfer: {method}")
            fig.tight_layout(rect=[0,0,1,.96])
            figures[f"group_transfer_{method}"]=fig; _save(fig,f"matched_group_4_group_transfer_{method}")
            if show_plots: plt.show()
            else: plt.close(fig)

    _print("\n[7/7] Analysis complete.")
    _print(f"Matching summary rows: {len(matching_summary):,}")
    _print(f"Matched MS update rows: {len(matched_ms_update_effects):,}")
    _print(f"Cross-assay event rows: {len(cross_transfer_events):,}")
    _print(f"Cross-assay group rows: {len(cross_transfer_group_level):,}")

    return {
        "group_membership":group_membership,
        "group_ap":group_ap,
        "matching_summary":matching_summary,
        "matched_ms_update_effects":matched_ms_update_effects,
        "matched_ms_update_summary":matched_ms_update_summary,
        "cross_transfer_events":cross_transfer_events,
        "cross_transfer_summary":cross_transfer_summary,
        "cross_transfer_group_level":cross_transfer_group_level,
        "cross_transfer_group_summary":cross_transfer_group_summary,
        "figures":figures,
    }


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def analyze_restricted_ms_cl_metrics(
    matched_results, task_orders, method_suffixes, base_score_col="mhcprime",
    ms_dataset="ms", ms_modes=("cumulative","global_union"), cumulative_base="first_task",
    method_labels=None, method_colors=None, plot_metrics=None, show_order_lines=False,
    order_line_alpha=0.12, order_linewidth=0.7, mean_linewidth=1.8, marker="o", markersize=4,
    figsize=(14,8), dpi=200, wspace=0.28, hspace=0.38, save_dir=None, show_plots=True, verbose=True,
):
    """
    Recompute continual-learning metrics while restricting ONLY the MS evaluation task
    to assay-matched alleles. Uses allele-level AP matrices already stored in the output
    of analyze_multidataset_matched_group_results(), so no AP recomputation is needed.

    MS modes
    --------
    cumulative:
        At stage t, MS is evaluated on the union of MS-matched alleles from assays
        encountered through stage t. Overlapping alleles are counted once.

        Because the MS subset changes with stage, MS retention/BWT/forgetting at stage t
        are computed by reevaluating ALL compared checkpoints on the SAME current stage-t
        allele set. This prevents changes in allele composition from being mistaken for
        forgetting.

        At s0 there are no encountered assays. cumulative_base="first_task" therefore
        uses the first assay's MS-matched allele set for the displayed base point.
        cumulative_base=None leaves the s0 MS value as NaN.

    global_union:
        MS is evaluated at every stage on the fixed union of alleles shared between MS
        and ANY non-MS assay. This gives a constant MS evaluation population across the
        whole trajectory and across task orders.

    Assay tasks are NOT restricted; their performance remains the macro mean over all
    evaluable alleles in that assay, matching the original multidataset analysis.

    Returns
    -------
    dict with:
        performance_matrices : mode -> order -> method -> stage x dataset dataframe
        stage_metrics        : one row per mode x order x method x stage
        task_stage_metrics   : task-level BWT/forgetting rows
        ms_group_summary     : MS allele sets/counts used at every stage
        order_summary        : mean/SD across task orders for each method/stage
        figures              : one figure per MS restriction mode
    """
    if "group_ap" not in matched_results or "group_membership" not in matched_results:
        raise KeyError("matched_results must contain 'group_ap' and 'group_membership'.")
    if cumulative_base not in {"first_task",None}:
        raise ValueError("cumulative_base must be 'first_task' or None.")
    valid_modes={"cumulative","global_union"}
    if not set(ms_modes).issubset(valid_modes):
        raise ValueError(f"ms_modes must contain only {sorted(valid_modes)}.")

    group_ap=matched_results["group_ap"]
    membership=matched_results["group_membership"]
    method_names=list(method_suffixes)
    assay_names=list(next(iter(task_orders.values())))
    dataset_names=[ms_dataset]+assay_names
    n_stages=len(assay_names)
    method_labels={} if method_labels is None else dict(method_labels)
    method_colors={} if method_colors is None else dict(method_colors)

    for order_name,order in task_orders.items():
        if len(order)!=n_stages or set(order)!=set(assay_names):
            raise ValueError(f"Task order '{order_name}' does not contain the same task set as the first order.")
    for ds in dataset_names:
        if ("allele",ds) not in group_ap or ("allele",ds) not in membership:
            raise KeyError(f"matched_results is missing allele-level data for '{ds}'.")

    def _print(x=""):
        if verbose: print(x)

    def _score_col(order_name,stage,method):
        return base_score_col if stage==0 else f"{order_name}_s{stage}{method_suffixes[method]}"

    def _macro(ap_df,col,alleles=None):
        if col not in ap_df.columns: raise KeyError(f"Score column '{col}' is missing.")
        s=ap_df[col]
        if alleles is not None: s=s.reindex(list(alleles))
        s=s.replace([np.inf,-np.inf],np.nan).dropna()
        return float(s.mean()) if len(s) else np.nan

    def _ms_matched_to_task(task):
        return set(membership[("allele",ms_dataset)]["groups"]) & set(membership[("allele",task)]["groups"])

    ms_all=set(membership[("allele",ms_dataset)]["groups"])
    matched_by_task={task:_ms_matched_to_task(task) for task in assay_names}
    global_union=set().union(*matched_by_task.values()) if matched_by_task else set()
    if not global_union: raise ValueError("No MS alleles overlap any assay dataset.")

    _print("="*88)
    _print("RESTRICTED-MS CONTINUAL-LEARNING ANALYSIS")
    _print("="*88)
    _print(f"Methods: {len(method_names)} | Orders: {len(task_orders)} | Assays: {assay_names}")
    _print(f"MS alleles total: {len(ms_all)} | MS alleles matched to >=1 assay: {len(global_union)}")
    for task in assay_names: _print(f"  {task}: {len(matched_by_task[task])} shared MS alleles")

    # Full assay mAP for every score column is reused throughout.
    _print("\n[1/4] Preparing assay macro-AP lookup...")
    assay_ap={task:group_ap[("allele",task)] for task in assay_names}
    ms_ap=group_ap[("allele",ms_dataset)]

    performance_matrices={mode:{} for mode in ms_modes}
    ms_group_rows=[]; stage_rows=[]; task_rows=[]

    _print("\n[2/4] Computing restricted-MS trajectories and CL metrics...")
    for mode in ms_modes:
        for order_name,order in tqdm(task_orders.items(),desc=f"CL metrics [{mode}]",leave=True):
            # Define the MS allele set used at each displayed stage.
            stage_ms_sets={}
            if mode=="global_union":
                stage_ms_sets={stage:set(global_union) for stage in range(n_stages+1)}
            else:
                running=set()
                for stage in range(1,n_stages+1):
                    running |= matched_by_task[order[stage-1]]
                    stage_ms_sets[stage]=set(running)
                stage_ms_sets[0]=set(matched_by_task[order[0]]) if cumulative_base=="first_task" else set()

            for stage in range(n_stages+1):
                ms_set=stage_ms_sets[stage]
                ms_group_rows.append({
                    "ms_mode":mode,"order":order_name,"stage":stage,"stage_name":f"s{stage}",
                    "current_task":"ms" if stage==0 else order[stage-1],"n_ms_alleles":len(ms_set),
                    "ms_alleles":tuple(sorted(ms_set)),
                })

            performance_matrices[mode][order_name]={}
            acquisition_stage={"ms":0,**{task:i for i,task in enumerate(order,start=1)}}
            full_order=["ms"]+list(order)

            for method in method_names:
                # Stage x domain matrix used for seen/all-domain means. MS is stage-specific.
                perf=pd.DataFrame(index=[f"s{i}" for i in range(n_stages+1)],columns=dataset_names,dtype=float)
                for stage in range(n_stages+1):
                    col=_score_col(order_name,stage,method)
                    ms_set=stage_ms_sets[stage]
                    perf.loc[f"s{stage}",ms_dataset]=_macro(ms_ap,col,ms_set) if ms_set else np.nan
                    for task in assay_names: perf.loc[f"s{stage}",task]=_macro(assay_ap[task],col)
                performance_matrices[mode][order_name][method]=perf

                for stage in range(n_stages+1):
                    current_task=full_order[stage]; seen_tasks=full_order[:stage+1]; previous_tasks=full_order[:stage]
                    current_col=_score_col(order_name,stage,method); current_ms_set=stage_ms_sets[stage]
                    seen_mean=float(perf.loc[f"s{stage}",seen_tasks].mean())
                    all_mean=float(perf.loc[f"s{stage}",dataset_names].mean())
                    current_perf=float(perf.loc[f"s{stage}",current_task])
                    current_gain=np.nan if stage==0 else current_perf-float(perf.loc[f"s{stage-1}",current_task])

                    # Baseline-relative domain gains. MS baseline is ALWAYS recomputed on the current MS subset.
                    seen_gains=[]; all_gains=[]
                    for task in dataset_names:
                        if task==ms_dataset:
                            base=_macro(ms_ap,base_score_col,current_ms_set) if current_ms_set else np.nan
                            cur=_macro(ms_ap,current_col,current_ms_set) if current_ms_set else np.nan
                        else:
                            base=_macro(assay_ap[task],base_score_col); cur=_macro(assay_ap[task],current_col)
                        gain=cur-base
                        all_gains.append(gain)
                        if task in seen_tasks: seen_gains.append(gain)
                    seen_gain=float(np.nanmean(seen_gains)) if seen_gains else np.nan
                    all_gain=float(np.nanmean(all_gains)) if all_gains else np.nan

                    # MS retention compares current and MHCPrime on the exact same current allele set.
                    ms_base=_macro(ms_ap,base_score_col,current_ms_set) if current_ms_set else np.nan
                    ms_current=_macro(ms_ap,current_col,current_ms_set) if current_ms_set else np.nan
                    ms_retention=ms_current-ms_base

                    bwt_vals=[]; forgetting_vals=[]
                    for previous_task in previous_tasks:
                        acquired=acquisition_stage[previous_task]
                        if previous_task==ms_dataset:
                            # Crucial for cumulative mode: all checkpoint comparisons use G_t, not G_k.
                            current_val=_macro(ms_ap,current_col,current_ms_set)
                            acquisition_col=_score_col(order_name,acquired,method)
                            acquisition_val=_macro(ms_ap,acquisition_col,current_ms_set)
                            prior_cols=[_score_col(order_name,i,method) for i in range(acquired,stage)]
                            prior_vals=[_macro(ms_ap,c,current_ms_set) for c in prior_cols]
                        else:
                            current_val=_macro(assay_ap[previous_task],current_col)
                            acquisition_col=_score_col(order_name,acquired,method)
                            acquisition_val=_macro(assay_ap[previous_task],acquisition_col)
                            prior_cols=[_score_col(order_name,i,method) for i in range(acquired,stage)]
                            prior_vals=[_macro(assay_ap[previous_task],c) for c in prior_cols]

                        bwt=current_val-acquisition_val
                        historical_best=float(np.nanmax(prior_vals)) if np.any(np.isfinite(prior_vals)) else np.nan
                        forgetting=historical_best-current_val if np.isfinite(historical_best) else np.nan
                        bwt_vals.append(bwt); forgetting_vals.append(forgetting)
                        task_rows.append({
                            "ms_mode":mode,"order":order_name,"method":method,"stage":stage,
                            "current_task":current_task,"task":previous_task,"task_position":acquired,
                            "n_ms_alleles":len(current_ms_set) if previous_task==ms_dataset else np.nan,
                            "acquisition_performance":acquisition_val,"current_performance":current_val,
                            "historical_best_before_current":historical_best,"backward_transfer":bwt,"forgetting":forgetting,
                        })

                    stage_rows.append({
                        "ms_mode":mode,"order":order_name,"method":method,"stage":stage,"stage_name":f"s{stage}",
                        "current_task":current_task,"n_ms_alleles":len(current_ms_set),
                        "seen_task_mean":seen_mean,"seen_task_gain_vs_base":seen_gain,
                        "all_domain_mean":all_mean,"all_domain_gain_vs_base":all_gain,
                        "current_task_performance":current_perf,"current_task_gain":current_gain,
                        "backward_transfer":float(np.nanmean(bwt_vals)) if bwt_vals else np.nan,
                        "average_forgetting":float(np.nanmean(forgetting_vals)) if forgetting_vals else np.nan,
                        "maximum_forgetting":float(np.nanmax(forgetting_vals)) if forgetting_vals else np.nan,
                        "ms_retention":ms_retention,"ms_performance":ms_current,"ms_baseline_same_subset":ms_base,
                    })

    stage_metrics=pd.DataFrame(stage_rows); task_stage_metrics=pd.DataFrame(task_rows); ms_group_summary=pd.DataFrame(ms_group_rows)
    metric_cols=["seen_task_mean","seen_task_gain_vs_base","all_domain_mean","all_domain_gain_vs_base","backward_transfer","average_forgetting","maximum_forgetting","ms_retention"]
    order_summary=(stage_metrics.groupby(["ms_mode","method","stage","stage_name"],as_index=False)
                   .agg(**{f"{m}_mean":(m,"mean") for m in metric_cols},**{f"{m}_sd":(m,"std") for m in metric_cols},n_orders=("order","nunique")))

    _print("\n[3/4] Plotting per-method trajectories...")
    if plot_metrics is None:
        plot_metrics=[
            ("seen_task_mean","Seen-task mean","Performance"),
            ("seen_task_gain_vs_base","Seen-task mean gain vs MHCPrime","Δ performance"),
            ("all_domain_mean","All-domain mean","Performance"),
            ("backward_transfer","Backward transfer","BWT"),
            ("maximum_forgetting","Maximum forgetting","Forgetting"),
            ("ms_retention","MS retention","ΔmAP from MHCPrime"),
        ]
    figures={}; stage_labels=["Base"]+[f"s{i}" for i in range(1,n_stages+1)]; x=np.arange(n_stages+1)

    for mode in ms_modes:
        nplots=len(plot_metrics); ncols=3; nrows=int(np.ceil(nplots/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=figsize,dpi=dpi,squeeze=False); axes=axes.ravel()
        for ax,(metric,title,ylabel) in zip(axes,plot_metrics):
            if metric not in metric_cols and metric not in stage_metrics.columns: raise KeyError(f"Unknown plot metric: {metric}")
            for method in method_names:
                color=method_colors.get(method,None); label=method_labels.get(method,method)
                q=stage_metrics[(stage_metrics["ms_mode"]==mode)&(stage_metrics["method"]==method)]
                if show_order_lines:
                    for _,oq in q.groupby("order",sort=False):
                        oq=oq.sort_values("stage"); ax.plot(oq["stage"],oq[metric],color=color,lw=order_linewidth,alpha=order_line_alpha)
                mean=q.groupby("stage")[metric].mean().reindex(range(n_stages+1))
                ax.plot(x,mean.to_numpy(),color=color,lw=mean_linewidth,marker=marker,ms=markersize,label=label)
            ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xticks(x); ax.set_xticklabels(stage_labels); ax.set_xlabel("Sequential update stage")
            if metric in {"seen_task_gain_vs_base","all_domain_gain_vs_base","backward_transfer","ms_retention"}: ax.axhline(0,color="black",lw=.8,ls="--",alpha=.7)
        for ax in axes[nplots:]: ax.set_visible(False)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=min(len(method_names),4),frameon=False)
        title="Cumulative assay-matched MS alleles" if mode=="cumulative" else "Fixed MS allele union matched across all assays"
        fig.suptitle(title); fig.subplots_adjust(wspace=wspace,hspace=hspace,bottom=.13,top=.92)
        if save_dir is not None:
            os.makedirs(save_dir,exist_ok=True)
            fig.savefig(os.path.join(save_dir,f"restricted_ms_cl_{mode}.png"),dpi=dpi,bbox_inches="tight")
            fig.savefig(os.path.join(save_dir,f"restricted_ms_cl_{mode}.pdf"),bbox_inches="tight")
        figures[mode]=fig
        if show_plots: plt.show()
        else: plt.close(fig)

    _print("\n[4/4] Done.")
    return {
        "performance_matrices":performance_matrices,
        "stage_metrics":stage_metrics,
        "task_stage_metrics":task_stage_metrics,
        "ms_group_summary":ms_group_summary,
        "matched_ms_alleles_by_task":matched_by_task,
        "global_union_ms_alleles":tuple(sorted(global_union)),
        "order_summary":order_summary,
        "figures":figures,
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def plot_assay_only_cl_metrics(
    test_sets, task_orders, method_suffixes, base_score_col="mhcprime",
    label_col="label", allele_col="allele", ap_matrix_fn=None,
    method_labels=None, method_colors=None, figsize=(16,9), dpi=150
):
    """
    CL metrics treating ONLY non-MS assays as tasks.
    MS is evaluated separately only through MS retention.

    Seen-task mean:
      Base = baseline performance on the first task in that order
      s1   = task 1 after update
      s2   = mean(task 1, task 2)
      ...
    All-domain mean:
      mean over all assay tasks at every stage, including Base.
    BWT / forgetting:
      previously learned assay tasks only; undefined at s1.
    MS retention:
      MS(stage) - MS(base).
    """
    assay_tasks=list(next(iter(task_orders.values())))
    n_stages=len(assay_tasks)
    methods=list(method_suffixes)
    method_labels=method_labels or {m:m for m in methods}

    # Expected model columns.
    model_cols={o:{m:[base_score_col]+[f"{o}_s{s}{method_suffixes[m]}" for s in range(1,n_stages+1)]
                   for m in methods} for o in task_orders}
    score_cols=list(dict.fromkeys([base_score_col]+[c for o in model_cols.values() for cols in o.values() for c in cols[1:]]))

    # Allele-level AP -> macro mAP per dataset.
    def _fallback(df):
        rows=[]
        for allele,g in df.groupby(allele_col,sort=False):
            y=g[label_col].to_numpy()
            row={"allele":allele}
            for c in score_cols:
                row[c]=np.nan if np.sum(y==1)==0 else average_precision_score(y,g[c].to_numpy())
            rows.append(row)
        return pd.DataFrame(rows).set_index("allele")

    allele_ap={}; maps={}
    for name,df in tqdm(test_sets.items(),desc="Computing dataset AP",leave=True):
        missing=[c for c in score_cols if c not in df.columns]
        if missing: raise ValueError(f"{name} missing score columns: {missing[:5]}")
        if ap_matrix_fn is not None:
            try: ap=ap_matrix_fn(df,score_cols,label_col=label_col,allele_col=allele_col,disable_tqdm=True)
            except TypeError: ap=ap_matrix_fn(df,score_cols,label_col=label_col,allele_col=allele_col)
        else: ap=_fallback(df)
        allele_ap[name]=ap
        maps[name]=ap.mean(axis=0,skipna=True)

    map_df=pd.DataFrame(maps)
    rows=[]; performance_matrices={}

    for order_name,order in task_orders.items():
        performance_matrices[order_name]={}
        acquisition={task:i for i,task in enumerate(order,start=1)}

        for method in methods:
            cols=model_cols[order_name][method]
            perf=pd.DataFrame(index=[f"s{i}" for i in range(n_stages+1)],columns=["ms"]+list(order),dtype=float)

            for stage,col in enumerate(cols):
                for task in ["ms"]+list(order): perf.loc[f"s{stage}",task]=map_df.loc[col,task]

            performance_matrices[order_name][method]=perf

            for stage in range(n_stages+1):
                s=f"s{stage}"

                if stage==0:
                    seen_mean=perf.loc["s0",order[0]]              # first task baseline
                    current_task=None
                    current_perf=np.nan
                    current_gain=np.nan
                    bwt=avg_forgetting=max_forgetting=np.nan
                else:
                    seen=list(order[:stage])
                    current_task=order[stage-1]
                    seen_mean=perf.loc[s,seen].mean()
                    current_perf=perf.loc[s,current_task]
                    current_gain=current_perf-perf.loc[f"s{stage-1}",current_task]

                    previous=list(order[:stage-1])
                    bwt_vals=[]; forgetting_vals=[]

                    for task in previous:
                        a=acquisition[task]
                        acquisition_perf=perf.loc[f"s{a}",task]
                        current=perf.loc[s,task]
                        bwt_vals.append(current-acquisition_perf)

                        historical=perf.loc[[f"s{i}" for i in range(a,stage)],task].max()
                        forgetting_vals.append(historical-current)

                    bwt=np.mean(bwt_vals) if bwt_vals else np.nan
                    avg_forgetting=np.mean(forgetting_vals) if forgetting_vals else np.nan
                    max_forgetting=np.max(forgetting_vals) if forgetting_vals else np.nan

                all_domain_mean=perf.loc[s,list(order)].mean()
                ms_retention=perf.loc[s,"ms"]-perf.loc["s0","ms"]

                rows.append({
                    "order":order_name,"method":method,"stage":stage,
                    "current_task":current_task,"seen_task_mean":seen_mean,
                    "all_domain_mean":all_domain_mean,
                    "current_task_performance":current_perf,"current_task_gain":current_gain,
                    "backward_transfer":bwt,"average_forgetting":avg_forgetting,
                    "maximum_forgetting":max_forgetting,"ms_retention":ms_retention,
                })

    stage_metrics=pd.DataFrame(rows)

    # Mean +/- SD across task orders, descriptive just like your current plot.
    specs=[
        ("seen_task_mean","Seen-task average performance","mAP"),
        ("all_domain_mean","All-domain mean performance","mAP"),
        ("backward_transfer","Backward transfer","BWT"),
        ("maximum_forgetting","Maximum forgetting","Forgetting"),
        ("ms_retention","MS retention","ΔmAP from base MS"),
    ]

    fig,axes=plt.subplots(2,3,figsize=figsize,dpi=dpi)
    axes=axes.ravel()
    x=np.arange(n_stages+1)

    for ax,(metric,title,ylabel) in zip(axes,specs):
        for method in methods:
            q=stage_metrics[stage_metrics["method"]==method]
            summary=q.groupby("stage")[metric].agg(["mean","std"]).reindex(x)
            kwargs={"marker":"o","label":method_labels.get(method,method)}
            if method_colors and method in method_colors: kwargs["color"]=method_colors[method]
            line=ax.plot(x,summary["mean"],**kwargs)[0]
            ax.fill_between(x,summary["mean"]-summary["std"],summary["mean"]+summary["std"],
                            color=line.get_color(),alpha=.15)

        if metric in {"backward_transfer","maximum_forgetting","ms_retention"}:
            ax.axhline(0,ls="--",lw=1)

        ax.set_xticks(x); ax.set_xticklabels(["Base"]+[f"s{i}" for i in range(1,n_stages+1)])
        ax.set_xlabel("Sequential update stage"); ax.set_ylabel(ylabel); ax.set_title(title)

    axes[-1].axis("off")
    handles,labels=axes[0].get_legend_handles_labels()
    axes[-1].legend(handles,labels,loc="center")
    fig.suptitle("Assay-task continual-learning performance across task orders")
    fig.tight_layout()

    return {
        "allele_ap":allele_ap,
        "map_by_dataset":map_df,
        "performance_matrices":performance_matrices,
        "stage_metrics":stage_metrics,
        "figure":fig,
    }

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def compute_seeded_allele_ap(
    seeded_test_sets, task_orders, method_suffixes, base_score_col="mhcprime",
    label_col="label", allele_col="allele", seed_tag="seed", verbose=True
):
    """
    Compute per-allele AP for test sets from multiple training seeds and merge
    seed-specific model AP columns within each test dataset.

    Parameters
    ----------
    seeded_test_sets : dict
        {seed: {test_set_name: dataframe}}
    task_orders : dict
        {order_name: [task1, task2, ...]}
    method_suffixes : dict
        {method_name: model_column_suffix}
    base_score_col : str
        Reference model column, e.g. "mhcprime".

    Returns
    -------
    dict
        {test_set_name: allele_AP_dataframe}

        Reference column is stored once:
            mhcprime

        Sequential-model columns are seed-tagged:
            order_s1_suffix_seed_0
            order_s1_suffix_seed_1
            ...
    """
    if not seeded_test_sets: raise ValueError("seeded_test_sets is empty.")

    seeds=list(seeded_test_sets)
    first_seed=seeds[0]
    test_names=list(seeded_test_sets[first_seed])

    # Validate test-set keys.
    for seed in seeds:
        if set(seeded_test_sets[seed])!=set(test_names):
            raise ValueError(f"Seed {seed} does not contain the same test-set keys as seed {first_seed}.")

    # Expected model score columns.
    n_stages=len(next(iter(task_orders.values())))
    score_cols=[base_score_col]
    for order_name in task_orders:
        for suffix in method_suffixes.values():
            score_cols.extend(f"{order_name}_s{s}{suffix}" for s in range(1,n_stages+1))
    score_cols=list(dict.fromkeys(score_cols))

    if verbose:
        print(f"Seeds: {seeds}")
        print(f"Test sets: {test_names}")
        print(f"Score columns per seed: {len(score_cols)}")

    per_seed={}
    for seed in seeds:
        per_seed[seed]={}

        for test_name,df in seeded_test_sets[seed].items():
            missing=[c for c in [label_col,allele_col]+score_cols if c not in df.columns]
            if missing: raise ValueError(f"Seed {seed} | {test_name} missing columns: {missing[:10]}")

            grouped=list(df.groupby(allele_col,sort=False))
            rows=[]

            for allele,g in tqdm(grouped,desc=f"Seed {seed} | {test_name} AP",leave=True):
                y=g[label_col].to_numpy()
                row={allele_col:allele}

                if np.sum(y==1)==0:
                    for col in score_cols: row[col]=np.nan
                else:
                    for col in score_cols:
                        row[col]=average_precision_score(y,g[col].to_numpy())

                rows.append(row)

            per_seed[seed][test_name]=pd.DataFrame(rows).set_index(allele_col)

    # Merge seeds separately for each test dataset.
    merged={}

    for test_name in test_names:
        ref=per_seed[first_seed][test_name]
        allele_order=ref.index
        parts=[ref[[base_score_col]].copy()]

        for seed in seeds:
            ap=per_seed[seed][test_name]

            if set(ap.index)!=set(allele_order):
                raise ValueError(f"{test_name}: allele set differs for seed {seed}.")

            ap=ap.reindex(allele_order)

            # The base model should be identical across seed-specific test sets.
            if not np.allclose(
                ref[base_score_col].to_numpy(float),
                ap[base_score_col].to_numpy(float),
                equal_nan=True,
            ):
                raise ValueError(
                    f"{test_name}: '{base_score_col}' AP differs for seed {seed}; "
                    "reference-model values should be identical across seeds."
                )

            x=ap.drop(columns=base_score_col).copy()
            x.columns=[f"{c}_{seed_tag}_{seed}" for c in x.columns]
            parts.append(x)

        merged[test_name]=pd.concat(parts,axis=1)

        if verbose:
            print(
                f"{test_name}: {len(merged[test_name])} alleles × "
                f"{merged[test_name].shape[1]} AP columns"
            )

    return merged
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

def plot_seeded_cl_average_metrics(
    seeded_allele_ap,
    task_orders,
    method_suffixes,
    base_score_col="mhcprime",
    order_name=None,
    interval="ci95",
    interval_unit="orders",  # "orders" or "seeds"
    method_labels=None,
    method_colors=None,
    figsize=(16,9),
    dpi=150,

    # Lines / markers
    linewidth=1.5,
    markersize=5,

    # Error bars
    error_capsize=3.0,
    error_linewidth=1.0,
    error_capthick=1.0,
    error_alpha=1.0,

    show=True,
):
    """
    Compute and plot seeded continual-learning aggregate trajectories twice:
      1) MS treated as a continual-learning task.
      2) MS excluded from the task sequence and evaluated only by MS retention.

    Aggregation
    -----------
    interval_unit="orders":
        Compute each metric independently for every seed x task-order trajectory,
        average seeds within each task order, then summarize across task orders.
        This characterizes task-order sensitivity at each sequential stage.

    interval_unit="seeds":
        Reproduces the previous hierarchy: average task orders within each seed,
        then summarize across seeds.

    interval:
        "ci95" -> 95% Student-t interval across the selected aggregation units
        "sd"   -> mean +/- 1 sample SD
        "se"   -> mean +/- 1 SE
        "range"-> observed min/max
        None   -> no interval

    Returns
    -------
    dict with:
        seed_order_metrics
        plot_unit_metrics
        seed_plot_metrics
        order_plot_metrics
        summaries
        figures
        seeds
    """
    if "ms" not in seeded_allele_ap:
        raise ValueError("seeded_allele_ap must contain an 'ms' test set.")
    if order_name is not None and order_name not in task_orders:
        raise ValueError(f"Unknown order_name: {order_name}")
    if interval not in {"ci95","sd","se","range",None}:
        raise ValueError("interval must be 'ci95', 'sd', 'se', 'range', or None.")
    if interval_unit not in {"orders","seeds"}:
        raise ValueError("interval_unit must be 'orders' or 'seeds'.")
    if order_name is not None and interval_unit=="orders":
        raise ValueError(
            "interval_unit='orders' requires multiple task orders. "
            "For a single order_name, use interval_unit='seeds'."
        )

    methods=list(method_suffixes)
    method_labels=method_labels or {m:m for m in methods}
    orders={order_name:task_orders[order_name]} if order_name is not None else task_orders
    n_stages=len(next(iter(task_orders.values())))
    assay_tasks=list(next(iter(task_orders.values())))

    if any(len(v)!=n_stages for v in task_orders.values()):
        raise ValueError("All task orders must have the same number of stages.")

    if set(assay_tasks)!=(set(seeded_allele_ap)-{"ms"}):
        missing=set(assay_tasks)-set(seeded_allele_ap)
        if missing:
            raise ValueError(f"Missing assay AP dataframes: {missing}")

    # ---------------------------------------------------------
    # Infer seed IDs and construct dataset-level macro-mAP table
    # ---------------------------------------------------------
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            m=seed_pattern.search(str(col))
            if m:
                seeds.append(m.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError("No `_seed_<seed>` score columns were found.")

    def _score_col(order,stage,method,seed):
        return f"{order}_s{stage}{method_suffixes[method]}_seed_{seed}"

    required=[base_score_col]

    for order in task_orders:
        for method in methods:
            for stage in range(1,n_stages+1):
                for seed in seeds:
                    required.append(_score_col(order,stage,method,seed))

    for dataset,df in seeded_allele_ap.items():
        missing=[c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{dataset}: missing {len(missing)} required AP columns. "
                f"First: {missing[:5]}"
            )

    maps={
        dataset:df.mean(axis=0,skipna=True)
        for dataset,df in seeded_allele_ap.items()
    }

    # ---------------------------------------------------------
    # Compute one complete metric trajectory per seed x order
    # ---------------------------------------------------------
    rows=[]

    for include_ms in (True,False):
        mode="ms_as_task" if include_ms else "assay_tasks_only"

        for order,task_order in orders.items():
            acquisition={
                "ms":0,
                **{task:i for i,task in enumerate(task_order,start=1)}
            }

            for method in methods:
                for seed in seeds:
                    perf=pd.DataFrame(
                        index=range(n_stages+1),
                        columns=["ms"]+list(task_order),
                        dtype=float
                    )

                    # Baseline
                    for task in perf.columns:
                        perf.loc[0,task]=maps[task][base_score_col]

                    # Sequential checkpoints
                    for stage in range(1,n_stages+1):
                        col=_score_col(order,stage,method,seed)

                        for task in perf.columns:
                            perf.loc[stage,task]=maps[task][col]

                    # CL metrics
                    for stage in range(n_stages+1):
                        if include_ms:
                            full_order=["ms"]+list(task_order)
                            seen=full_order[:stage+1]
                            previous=full_order[:stage]

                            seen_mean=perf.loc[stage,seen].mean()
                            all_domain_mean=perf.loc[
                                stage,
                                ["ms"]+list(task_order)
                            ].mean()

                        else:
                            if stage==0:
                                # No assay encountered yet: baseline performance
                                # of the assay that will be learned first.
                                seen_mean=perf.loc[0,task_order[0]]
                                previous=[]
                            else:
                                seen=list(task_order[:stage])
                                previous=list(task_order[:stage-1])
                                seen_mean=perf.loc[stage,seen].mean()

                            all_domain_mean=perf.loc[
                                stage,
                                list(task_order)
                            ].mean()

                        bwt_vals=[]
                        forgetting_vals=[]

                        for task in previous:
                            acquired=acquisition[task]
                            current=perf.loc[stage,task]
                            acquired_perf=perf.loc[acquired,task]
                            prior_best=perf.loc[acquired:stage-1,task].max()

                            bwt_vals.append(current-acquired_perf)
                            forgetting_vals.append(prior_best-current)

                        rows.append({
                            "mode":mode,
                            "order":order,
                            "method":method,
                            "seed":seed,
                            "stage":stage,
                            "seen_task_mean":seen_mean,
                            "all_domain_mean":all_domain_mean,
                            "backward_transfer":np.mean(bwt_vals) if bwt_vals else np.nan,
                            "maximum_forgetting":np.max(forgetting_vals) if forgetting_vals else np.nan,
                            "ms_retention":perf.loc[stage,"ms"]-perf.loc[0,"ms"],
                        })

    seed_order_metrics=pd.DataFrame(rows)

    metric_cols=[
        "seen_task_mean",
        "all_domain_mean",
        "backward_transfer",
        "maximum_forgetting",
        "ms_retention",
    ]

    # ---------------------------------------------------------
    # Both aggregation hierarchies are retained in output
    # ---------------------------------------------------------

    # Old hierarchy:
    # average orders within each seed -> seed-level values
    seed_plot_metrics=(
        seed_order_metrics
        .groupby(["mode","method","seed","stage"],as_index=False)[metric_cols]
        .mean()
    )

    # New hierarchy:
    # average seeds within each order -> order-level values
    order_plot_metrics=(
        seed_order_metrics
        .groupby(["mode","method","order","stage"],as_index=False)[metric_cols]
        .mean()
    )

    plot_unit_metrics=(
        order_plot_metrics
        if interval_unit=="orders"
        else seed_plot_metrics
    )

    # ---------------------------------------------------------
    # Summaries across selected unit
    # ---------------------------------------------------------
    summary_rows=[]

    for keys,g in plot_unit_metrics.groupby(
        ["mode","method","stage"],
        sort=False
    ):
        mode,method,stage=keys

        for metric in metric_cols:
            vals=g[metric].dropna().to_numpy(float)
            n=len(vals)

            if n==0:
                mean=sd=se=lower=upper=np.nan

            else:
                mean=vals.mean()
                sd=vals.std(ddof=1) if n>1 else np.nan
                se=sd/np.sqrt(n) if n>1 else np.nan

                if interval is None:
                    lower=upper=np.nan

                elif interval=="sd":
                    lower,upper=mean-sd,mean+sd

                elif interval=="se":
                    lower,upper=mean-se,mean+se

                elif interval=="ci95":
                    half=(
                        student_t.ppf(.975,n-1)*se
                        if n>1 else np.nan
                    )
                    lower,upper=mean-half,mean+half

                elif interval=="range":
                    lower,upper=vals.min(),vals.max()

            summary_rows.append({
                "mode":mode,
                "method":method,
                "stage":stage,
                "metric":metric,
                "interval_unit":interval_unit,
                "n_units":n,
                "mean":mean,
                "sd":sd,
                "se":se,
                "lower":lower,
                "upper":upper,
            })

    summaries=pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    specs=[
        ("seen_task_mean","Seen-task average performance","mAP"),
        ("all_domain_mean","All-domain mean performance","mAP"),
        ("backward_transfer","Backward transfer","BWT"),
        ("maximum_forgetting","Maximum forgetting","Forgetting"),
        ("ms_retention","MS retention","ΔmAP from base MS"),
    ]

    interval_names={
        "ci95":"95% CI",
        "sd":"±1 SD",
        "se":"±1 SE",
        "range":"range",
        None:"no interval",
    }

    unit_names={
        "orders":"task orders",
        "seeds":"seeds",
    }

    figures={}

    for mode,title_prefix in [
        ("ms_as_task","MS included as a task"),
        ("assay_tasks_only","MS excluded from task sequence"),
    ]:
        fig,axes=plt.subplots(2,3,figsize=figsize,dpi=dpi)
        axes=axes.ravel()

        for ax,(metric,title,ylabel) in zip(axes,specs):
            for method in methods:
                q=(
                    summaries[
                        (summaries["mode"]==mode)&
                        (summaries["method"]==method)&
                        (summaries["metric"]==metric)
                    ]
                    .sort_values("stage")
                )

                color=method_colors.get(method) if method_colors else None

                x=q["stage"].to_numpy(float)
                y=q["mean"].to_numpy(float)

                # Plot line/markers first
                line=ax.plot(
                    x,y,
                    marker="o",
                    linewidth=linewidth,
                    markersize=markersize,
                    label=method_labels.get(method,method),
                    color=color,
                )[0]

                # T-capped interval bars
                if interval is not None:
                    low=q["lower"].to_numpy(float)
                    high=q["upper"].to_numpy(float)

                    valid=(
                        np.isfinite(x)&
                        np.isfinite(y)&
                        np.isfinite(low)&
                        np.isfinite(high)
                    )

                    if valid.any():
                        lower_err=y[valid]-low[valid]
                        upper_err=high[valid]-y[valid]

                        ax.errorbar(
                            x[valid],
                            y[valid],
                            yerr=np.vstack([lower_err,upper_err]),
                            fmt="none",
                            ecolor=line.get_color(),
                            elinewidth=error_linewidth,
                            capsize=error_capsize,
                            capthick=error_capthick,
                            alpha=error_alpha,
                            zorder=3,
                        )

            if metric in {
                "backward_transfer",
                "maximum_forgetting",
                "ms_retention"
            }:
                ax.axhline(0,ls="--",lw=1)

            # With MS excluded, BWT/forgetting undefined until s2.
            if (
                mode=="assay_tasks_only"
                and metric in {
                    "backward_transfer",
                    "maximum_forgetting"
                }
            ):
                ticks=np.arange(2,n_stages+1)
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"s{i}" for i in ticks])

            else:
                ticks=np.arange(n_stages+1)
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    ["Base"]+[f"s{i}" for i in range(1,n_stages+1)]
                )

            ax.set_xlabel("Sequential update stage")
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        axes[-1].axis("off")

        handles,labels=axes[0].get_legend_handles_labels()
        axes[-1].legend(handles,labels,loc="center")

        scope=order_name if order_name is not None else "task orders"

        fig.suptitle(
            f"{title_prefix}: continual-learning performance across {scope}\n"
            f"Lines = mean across {unit_names[interval_unit]}; "
            f"error bars = {interval_names[interval]} across {unit_names[interval_unit]}"
        )

        fig.tight_layout()
        figures[mode]=fig

        if show:
            plt.show()

    return {
        "seed_order_metrics":seed_order_metrics,
        "plot_unit_metrics":plot_unit_metrics,
        "seed_plot_metrics":seed_plot_metrics,
        "order_plot_metrics":order_plot_metrics,
        "summaries":summaries,
        "figures":figures,
        "seeds":seeds,
    }
import re, textwrap
import numpy as np
import matplotlib.pyplot as plt

def plot_seeded_cl_heatmap_grids(
    seeded_allele_ap, task_orders, method_suffixes, base_score_col="mhcprime",
    method_order=None, order_names=None, dataset_labels=None,
    change_mode="delta", change_vmin=None, change_vmax=None,
    decimals=2, annotate=True, cmap_raw="viridis", cmap_change="coolwarm_r",
    figsize_per_panel=(1.8,1.55), figsize_average=(12,10), dpi=200,
    x_tick_rotation=45, method_wrap_width=28, method_label_fontsize=7,
    summary_title_wrap_width=24, show=True
):
    """
    Plot 10 seeded continual-learning heatmap figures.

    Order-specific grids:
      1. Raw mAP, with MS
      2. Raw mAP, without MS
      3. Change from Base, with MS
      4. Change from Base, without MS
      5. Change from previous checkpoint, with MS
      6. Change from previous checkpoint, without MS
      7. Change from prior historical maximum, with MS
      8. Change from prior historical maximum, without MS

    Order-averaged positional grids:
      9.  All four metrics averaged across task orders, with MS
      10. All four metrics averaged across task orders, without MS

    The order-averaged matrices align by sequence position:
        rows = Base, s1, s2, s3, s4
        cols = MS, s1, s2, s3, s4       (with MS)
        cols = s1, s2, s3, s4           (without MS)

    change_mode:
        "delta"   : current - reference
        "percent" : 100 * (current - reference) / reference

    Historical maximum:
        At stage t, every evaluation domain is compared against its best
        performance at any checkpoint BEFORE t:

            current_t - max(Base, ..., stage t-1)

        positive = new historical best
        ~0       = near prior best
        negative = below prior historical best

    change_vmin / change_vmax:
        Optional fixed visualization limits for all change heatmaps.
        Values outside the limits retain their true values but saturate
        at the colormap endpoints.

    All nonlinear transformations are computed independently within each
    seed first, then averaged across seeds.
    """
    if change_mode not in {"delta","percent"}:
        raise ValueError("change_mode must be 'delta' or 'percent'.")
    if change_vmin is not None and change_vmax is not None and change_vmin>=change_vmax:
        raise ValueError("change_vmin must be smaller than change_vmax.")

    method_order=list(method_suffixes) if method_order is None else list(method_order)
    order_names=list(task_orders) if order_names is None else list(order_names)
    dataset_labels={} if dataset_labels is None else dict(dataset_labels)

    for method in method_order:
        if method not in method_suffixes:
            raise ValueError(f"Method {method!r} is not in method_suffixes.")
    for order_name in order_names:
        if order_name not in task_orders:
            raise ValueError(f"Task order {order_name!r} is not in task_orders.")

    n_stages=len(next(iter(task_orders.values())))
    if any(len(task_orders[o])!=n_stages for o in order_names):
        raise ValueError("All task orders must contain the same number of stages.")

    # ---------------------------------------------------------
    # Detect seeds
    # ---------------------------------------------------------
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            m=seed_pattern.search(str(col))
            if m: seeds.append(m.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError("No columns ending in '_seed_<seed>' were detected.")

    def _score_col(order_name,stage,method,seed):
        return f"{order_name}_s{stage}{method_suffixes[method]}_seed_{seed}"

    def _label(x):
        return dataset_labels.get(x,x)

    def _change(current,reference):
        current=np.asarray(current,dtype=float)
        reference=np.asarray(reference,dtype=float)

        if change_mode=="delta":
            return current-reference

        with np.errstate(divide="ignore",invalid="ignore"):
            out=100.0*(current-reference)/reference

        out[~np.isfinite(out)]=np.nan
        return out

    # ---------------------------------------------------------
    # Dataset macro-mAP from per-allele AP.
    # Ignore nonnumeric identifier columns such as allele.
    # ---------------------------------------------------------
    maps={}

    for name,df in seeded_allele_ap.items():
        numeric=df.select_dtypes(include=[np.number])

        if base_score_col not in numeric.columns:
            raise ValueError(
                f"{name}: base score column {base_score_col!r} "
                "is missing or nonnumeric."
            )

        maps[name]=numeric.mean(axis=0,skipna=True)

    raw_full={}; raw_no_ms={}
    base_full={}; base_no_ms={}
    previous_full={}; previous_no_ms={}
    historical_max_full={}; historical_max_no_ms={}

    # =========================================================
    # BUILD ORDER-SPECIFIC MATRICES
    # =========================================================
    for method in method_order:
        for order_name in order_names:
            order=list(task_orders[order_name])

            row_keys=["Base"]+order
            col_keys=["ms"]+order

            seed_raw=[]
            seed_base=[]
            seed_previous=[]
            seed_historical=[]

            for seed in seeds:
                raw=np.full(
                    (len(row_keys),len(col_keys)),
                    np.nan,
                    dtype=float
                )

                # Base checkpoint
                for j,dataset in enumerate(col_keys):
                    if dataset not in maps:
                        raise ValueError(f"Missing dataset {dataset!r}.")
                    if base_score_col not in maps[dataset].index:
                        raise ValueError(
                            f"{dataset}: missing base column {base_score_col!r}."
                        )

                    raw[0,j]=maps[dataset][base_score_col]

                # Sequential checkpoints
                for stage in range(1,len(row_keys)):
                    score_col=_score_col(
                        order_name,
                        stage,
                        method,
                        seed
                    )

                    for j,dataset in enumerate(col_keys):
                        if score_col not in maps[dataset].index:
                            raise ValueError(
                                f"{dataset}: missing score column {score_col!r}."
                            )

                        raw[stage,j]=maps[dataset][score_col]

                # Change from Base
                base_change=np.zeros_like(raw)

                for stage in range(1,len(row_keys)):
                    base_change[stage]=_change(
                        raw[stage],
                        raw[0]
                    )

                # Change from immediately previous checkpoint
                previous_change=np.zeros_like(raw)

                for stage in range(1,len(row_keys)):
                    previous_change[stage]=_change(
                        raw[stage],
                        raw[stage-1]
                    )

                # Change from PRIOR historical maximum.
                # Current checkpoint is excluded from the reference.
                historical_change=np.zeros_like(raw)

                for stage in range(1,len(row_keys)):
                    prior_max=np.nanmax(
                        raw[:stage],
                        axis=0
                    )

                    historical_change[stage]=_change(
                        raw[stage],
                        prior_max
                    )

                seed_raw.append(raw)
                seed_base.append(base_change)
                seed_previous.append(previous_change)
                seed_historical.append(historical_change)

            # Seed aggregation occurs AFTER the transformations.
            raw=np.nanmean(np.stack(seed_raw),axis=0)
            base_change=np.nanmean(np.stack(seed_base),axis=0)
            previous_change=np.nanmean(np.stack(seed_previous),axis=0)
            historical_change=np.nanmean(np.stack(seed_historical),axis=0)

            key=(method,order_name)

            raw_full[key]={
                "values":raw,
                "row_keys":row_keys,
                "col_keys":col_keys
            }

            raw_no_ms[key]={
                "values":raw[:,1:],
                "row_keys":row_keys,
                "col_keys":order
            }

            base_full[key]={
                "values":base_change,
                "row_keys":row_keys,
                "col_keys":col_keys
            }

            base_no_ms[key]={
                "values":base_change[:,1:],
                "row_keys":row_keys,
                "col_keys":order
            }

            previous_full[key]={
                "values":previous_change,
                "row_keys":row_keys,
                "col_keys":col_keys
            }

            previous_no_ms[key]={
                "values":previous_change[:,1:],
                "row_keys":row_keys,
                "col_keys":order
            }

            historical_max_full[key]={
                "values":historical_change,
                "row_keys":row_keys,
                "col_keys":col_keys
            }

            historical_max_no_ms[key]={
                "values":historical_change[:,1:],
                "row_keys":row_keys,
                "col_keys":order
            }

    # =========================================================
    # COLOR LIMITS
    # =========================================================
    def _data_limits(matrix_dict,symmetric=False):
        vals=[]

        for item in matrix_dict.values():
            x=np.asarray(item["values"],dtype=float)
            x=x[np.isfinite(x)]

            if x.size:
                vals.append(x)

        if not vals:
            return (-1,1) if symmetric else (0,1)

        vals=np.concatenate(vals)

        if symmetric:
            vmax=float(np.nanmax(np.abs(vals)))

            if vmax==0:
                vmax=1e-12

            return -vmax,vmax

        return (
            float(np.nanmin(vals)),
            float(np.nanmax(vals))
        )

    def _change_limits(matrix_dict):
        auto_min,auto_max=_data_limits(
            matrix_dict,
            symmetric=True
        )

        vmin=(
            auto_min
            if change_vmin is None
            else float(change_vmin)
        )

        vmax=(
            auto_max
            if change_vmax is None
            else float(change_vmax)
        )

        if vmin>=vmax:
            raise ValueError(
                f"Invalid change color limits: "
                f"vmin={vmin}, vmax={vmax}"
            )

        return vmin,vmax

    raw_full_lim=_data_limits(raw_full)
    raw_no_ms_lim=_data_limits(raw_no_ms)

    base_full_lim=_change_limits(base_full)
    base_no_ms_lim=_change_limits(base_no_ms)

    previous_full_lim=_change_limits(previous_full)
    previous_no_ms_lim=_change_limits(previous_no_ms)

    historical_full_lim=_change_limits(historical_max_full)
    historical_no_ms_lim=_change_limits(historical_max_no_ms)

    # =========================================================
    # ORDER-SPECIFIC GRID HELPER
    # =========================================================
    def _draw_grid(
        matrix_dict,title,cmap,vmin,vmax,colorbar_label
    ):
        n_methods=len(method_order)
        n_orders=len(order_names)
        panel_w,panel_h=figsize_per_panel

        fig,axes=plt.subplots(
            n_methods,
            n_orders,
            figsize=(
                panel_w*n_orders+1.9,
                panel_h*n_methods+1.1
            ),
            dpi=dpi,
            squeeze=False
        )

        fig.subplots_adjust(
            left=0.12,
            right=0.905,
            bottom=0.07,
            top=0.94,
            wspace=0.44,
            hspace=0.62
        )

        im=None

        for i,method in enumerate(method_order):
            wrapped_method="\n".join(
                textwrap.wrap(
                    method,
                    width=method_wrap_width
                )
            )

            for j,order_name in enumerate(order_names):
                ax=axes[i,j]
                item=matrix_dict[(method,order_name)]
                values=np.asarray(
                    item["values"],
                    dtype=float
                )

                im=ax.imshow(
                    values,
                    aspect="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                )

                ax.set_xticks(
                    np.arange(len(item["col_keys"]))
                )

                ax.set_xticklabels(
                    [_label(x) for x in item["col_keys"]],
                    rotation=x_tick_rotation,
                    ha="right",
                    fontsize=6
                )

                ax.set_yticks(
                    np.arange(len(item["row_keys"]))
                )

                ax.set_yticklabels(
                    [_label(x) for x in item["row_keys"]],
                    fontsize=6
                )

                if i==0:
                    ax.set_title(
                        order_name,
                        fontsize=7,
                        pad=5
                    )

                if j==0:
                    ax.set_ylabel(
                        wrapped_method,
                        fontsize=method_label_fontsize,
                        rotation=90,
                        labelpad=14,
                        va="center"
                    )

                if annotate:
                    for r in range(values.shape[0]):
                        for c in range(values.shape[1]):
                            value=values[r,c]

                            if np.isfinite(value):
                                ax.text(
                                    c,
                                    r,
                                    f"{value:.{decimals}f}",
                                    ha="center",
                                    va="center",
                                    fontsize=5
                                )

                ax.tick_params(length=0)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)

        # Dedicated external colorbar.
        cax=fig.add_axes([
            0.93,
            0.15,
            0.014,
            0.70
        ])

        cb=fig.colorbar(
            im,
            cax=cax
        )

        cb.ax.tick_params(
            labelsize=7
        )

        cb.set_label(
            colorbar_label,
            fontsize=8
        )

        fig.suptitle(
            title,
            fontsize=12
        )

        return fig

    # =========================================================
    # ORIGINAL 8 GRIDS
    # =========================================================
    change_text=(
        "ΔAP"
        if change_mode=="delta"
        else "% change"
    )

    base_label=f"{change_text} from base"
    previous_label=f"{change_text} from previous checkpoint"
    historical_label=f"{change_text} from prior historical maximum"

    figures={}

    figures["raw_full"]=_draw_grid(
        raw_full,
        "Complete continual-learning performance matrices",
        cmap_raw,
        *raw_full_lim,
        "mAP"
    )

    figures["raw_no_ms"]=_draw_grid(
        raw_no_ms,
        "Continual-learning performance matrices without MS column",
        cmap_raw,
        *raw_no_ms_lim,
        "mAP"
    )

    figures["change_from_base_full"]=_draw_grid(
        base_full,
        f"Continual-learning performance relative to base ({base_label})",
        cmap_change,
        *base_full_lim,
        base_label
    )

    figures["change_from_base_no_ms"]=_draw_grid(
        base_no_ms,
        f"Continual-learning performance relative to base without MS column ({base_label})",
        cmap_change,
        *base_no_ms_lim,
        base_label
    )

    figures["change_from_previous_full"]=_draw_grid(
        previous_full,
        f"Continual-learning performance relative to previous checkpoint ({previous_label})",
        cmap_change,
        *previous_full_lim,
        previous_label
    )

    figures["change_from_previous_no_ms"]=_draw_grid(
        previous_no_ms,
        f"Continual-learning performance relative to previous checkpoint without MS column ({previous_label})",
        cmap_change,
        *previous_no_ms_lim,
        previous_label
    )

    figures["change_from_historical_max_full"]=_draw_grid(
        historical_max_full,
        f"Continual-learning performance relative to prior historical maximum ({historical_label})",
        cmap_change,
        *historical_full_lim,
        historical_label
    )

    figures["change_from_historical_max_no_ms"]=_draw_grid(
        historical_max_no_ms,
        f"Continual-learning performance relative to prior historical maximum without MS column ({historical_label})",
        cmap_change,
        *historical_no_ms_lim,
        historical_label
    )

    # =========================================================
    # ORDER-AVERAGED POSITIONAL MATRICES
    # =========================================================
    #
    # Every order-specific matrix already follows sequence position:
    #
    # rows:
    #   Base, task learned at s1, task learned at s2, ...
    #
    # columns:
    #   MS, task learned at s1, task learned at s2, ...
    #
    # Therefore elementwise averaging across orders gives a positional
    # matrix that can be relabeled Base/s1/s2/... and MS/s1/s2/...
    # =========================================================
    def _average_orders(matrix_dict,with_ms):
        out={}

        row_keys=["Base"]+[
            f"s{i}"
            for i in range(1,n_stages+1)
        ]

        col_keys=(
            ["ms"]+[
                f"s{i}"
                for i in range(1,n_stages+1)
            ]
            if with_ms
            else [
                f"s{i}"
                for i in range(1,n_stages+1)
            ]
        )

        for method in method_order:
            arrays=[
                np.asarray(
                    matrix_dict[(method,order_name)]["values"],
                    dtype=float
                )
                for order_name in order_names
            ]

            values=np.nanmean(
                np.stack(arrays,axis=0),
                axis=0
            )

            out[method]={
                "values":values,
                "row_keys":row_keys,
                "col_keys":col_keys
            }

        return out

    avg_raw_full=_average_orders(
        raw_full,
        with_ms=True
    )

    avg_raw_no_ms=_average_orders(
        raw_no_ms,
        with_ms=False
    )

    avg_base_full=_average_orders(
        base_full,
        with_ms=True
    )

    avg_base_no_ms=_average_orders(
        base_no_ms,
        with_ms=False
    )

    avg_previous_full=_average_orders(
        previous_full,
        with_ms=True
    )

    avg_previous_no_ms=_average_orders(
        previous_no_ms,
        with_ms=False
    )

    avg_historical_full=_average_orders(
        historical_max_full,
        with_ms=True
    )

    avg_historical_no_ms=_average_orders(
        historical_max_no_ms,
        with_ms=False
    )

    # =========================================================
    # SUMMARY GRID HELPER
    #
    # Rows = methods
    # Columns = four heatmap types
    #
    # Each column receives its own colorbar because raw mAP and the
    # three change metrics use different scales/units.
    # =========================================================
    def _draw_average_summary_grid(
        matrix_specs,
        title
    ):
        n_methods=len(method_order)
        n_metrics=len(matrix_specs)

        fig,axes=plt.subplots(
            n_methods,
            n_metrics,
            figsize=figsize_average,
            dpi=dpi,
            squeeze=False
        )

        # Leave enough room for wrapped method labels and bottom colorbars.
        fig.subplots_adjust(
            left=0.13,
            right=0.98,
            bottom=0.12,
            top=0.90,
            wspace=0.50,
            hspace=0.55
        )

        column_images=[]

        for j,spec in enumerate(matrix_specs):
            matrix_dict=spec["matrix"]

            for i,method in enumerate(method_order):
                ax=axes[i,j]
                item=matrix_dict[method]
                values=np.asarray(
                    item["values"],
                    dtype=float
                )

                im=ax.imshow(
                    values,
                    aspect="auto",
                    cmap=spec["cmap"],
                    vmin=spec["vmin"],
                    vmax=spec["vmax"]
                )

                if i==0:
                    wrapped_title="\n".join(
                        textwrap.wrap(
                            spec["title"],
                            width=summary_title_wrap_width
                        )
                    )

                    ax.set_title(
                        wrapped_title,
                        fontsize=9,
                        pad=8
                    )

                ax.set_xticks(
                    np.arange(
                        len(item["col_keys"])
                    )
                )

                ax.set_xticklabels(
                    [
                        _label(x)
                        for x in item["col_keys"]
                    ],
                    rotation=x_tick_rotation,
                    ha="right",
                    fontsize=7
                )

                ax.set_yticks(
                    np.arange(
                        len(item["row_keys"])
                    )
                )

                ax.set_yticklabels(
                    [
                        _label(x)
                        for x in item["row_keys"]
                    ],
                    fontsize=7
                )

                if j==0:
                    wrapped_method="\n".join(
                        textwrap.wrap(
                            method,
                            width=method_wrap_width
                        )
                    )

                    ax.set_ylabel(
                        wrapped_method,
                        fontsize=method_label_fontsize,
                        rotation=90,
                        labelpad=14,
                        va="center"
                    )

                if annotate:
                    for r in range(values.shape[0]):
                        for c in range(values.shape[1]):
                            value=values[r,c]

                            if np.isfinite(value):
                                ax.text(
                                    c,
                                    r,
                                    f"{value:.{decimals}f}",
                                    ha="center",
                                    va="center",
                                    fontsize=5
                                )

                ax.tick_params(length=0)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)

            column_images.append(im)

        # One horizontal colorbar per metric column.
        # fig.colorbar(..., ax=entire column) automatically places each
        # colorbar relative to its own column rather than over another axis.
        for j,(spec,im) in enumerate(
            zip(matrix_specs,column_images)
        ):
            cb=fig.colorbar(
                im,
                ax=axes[:,j],
                orientation="horizontal",
                fraction=0.025,
                pad=0.03,
                aspect=30
            )

            cb.ax.tick_params(
                labelsize=7
            )

            cb.set_label(
                spec["colorbar_label"],
                fontsize=8
            )

        fig.suptitle(
            title,
            fontsize=12,
            y=0.97
        )

        return fig

    # ---------------------------------------------------------
    # Plot 9: averaged across orders, WITH MS
    # ---------------------------------------------------------
    full_specs=[
        {
            "matrix":avg_raw_full,
            "title":"Raw mean performance",
            "cmap":cmap_raw,
            "vmin":raw_full_lim[0],
            "vmax":raw_full_lim[1],
            "colorbar_label":"mAP"
        },
        {
            "matrix":avg_base_full,
            "title":base_label,
            "cmap":cmap_change,
            "vmin":base_full_lim[0],
            "vmax":base_full_lim[1],
            "colorbar_label":base_label
        },
        {
            "matrix":avg_previous_full,
            "title":previous_label,
            "cmap":cmap_change,
            "vmin":previous_full_lim[0],
            "vmax":previous_full_lim[1],
            "colorbar_label":previous_label
        },
        {
            "matrix":avg_historical_full,
            "title":historical_label,
            "cmap":cmap_change,
            "vmin":historical_full_lim[0],
            "vmax":historical_full_lim[1],
            "colorbar_label":historical_label
        }
    ]

    figures["order_average_full"]=_draw_average_summary_grid(
        full_specs,
        "Order-averaged continual-learning heatmaps - with MS"
    )

    # ---------------------------------------------------------
    # Plot 10: averaged across orders, WITHOUT MS
    # ---------------------------------------------------------
    no_ms_specs=[
        {
            "matrix":avg_raw_no_ms,
            "title":"Raw mean performance",
            "cmap":cmap_raw,
            "vmin":raw_no_ms_lim[0],
            "vmax":raw_no_ms_lim[1],
            "colorbar_label":"mAP"
        },
        {
            "matrix":avg_base_no_ms,
            "title":base_label,
            "cmap":cmap_change,
            "vmin":base_no_ms_lim[0],
            "vmax":base_no_ms_lim[1],
            "colorbar_label":base_label
        },
        {
            "matrix":avg_previous_no_ms,
            "title":previous_label,
            "cmap":cmap_change,
            "vmin":previous_no_ms_lim[0],
            "vmax":previous_no_ms_lim[1],
            "colorbar_label":previous_label
        },
        {
            "matrix":avg_historical_no_ms,
            "title":historical_label,
            "cmap":cmap_change,
            "vmin":historical_no_ms_lim[0],
            "vmax":historical_no_ms_lim[1],
            "colorbar_label":historical_label
        }
    ]

    figures["order_average_no_ms"]=_draw_average_summary_grid(
        no_ms_specs,
        "Order-averaged continual-learning heatmaps - without MS"
    )

    if show:
        plt.show()

    return {
        # Original order-specific matrices
        "raw_full":raw_full,
        "raw_no_ms":raw_no_ms,
        "change_from_base_full":base_full,
        "change_from_base_no_ms":base_no_ms,
        "change_from_previous_full":previous_full,
        "change_from_previous_no_ms":previous_no_ms,
        "change_from_historical_max_full":historical_max_full,
        "change_from_historical_max_no_ms":historical_max_no_ms,

        # New order-averaged positional matrices
        "order_average":{
            "raw_full":avg_raw_full,
            "raw_no_ms":avg_raw_no_ms,
            "change_from_base_full":avg_base_full,
            "change_from_base_no_ms":avg_base_no_ms,
            "change_from_previous_full":avg_previous_full,
            "change_from_previous_no_ms":avg_previous_no_ms,
            "change_from_historical_max_full":avg_historical_full,
            "change_from_historical_max_no_ms":avg_historical_no_ms,
        },

        "figures":figures,
        "seeds":seeds,
    }
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import t as student_t

def plot_seeded_task_retention_boxplots(
    seeded_allele_ap, task_orders, method_suffixes,
    assay_order=None, allele_col="allele",
    all_post_acquisition=False, method_colors=None,
    figsize=(16,7), dpi=150, box_width=0.10,
    point_size=22, point_alpha=0.75,
    legend_nrows=2, show_points=True, show=True
):
    """
    Plot task-specific retention distributions across task orders.

    Retention is calculated within each seed first:

        retention = later mAP - acquisition mAP

    Seeds are then averaged for each unique order/checkpoint context.

    IMPORTANT
    ---------
    The plotted points are NOT seed replicates.

    Each plotted point corresponds to one unique retention context:

        final-only mode:
            one point per qualifying order

        all-post-acquisition mode:
            one point per qualifying (order, later checkpoint) pair

    Seeds are replicates of that context and are averaged before plotting.

    The x-axis labels report n for each assay, where n is the number of
    unique order/checkpoint contexts contributing to each assay/method box.

    Modes
    -----
    all_post_acquisition=False
        Use only final performance relative to acquisition:

            final mAP - acquisition mAP

        Orders where the assay is acquired at the final stage are excluded,
        because there is no subsequent update and retention would be 0 by
        construction.

    all_post_acquisition=True
        Use every checkpoint strictly AFTER acquisition:

            mAP(stage t) - acquisition mAP, for all t > acquisition stage

        The acquisition checkpoint itself is never included.
        Orders where the assay is acquired last contribute no observations.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per method x assay x order x evaluation checkpoint after
        averaging across seeds.

    fig : matplotlib.figure.Figure
    """
    methods=list(method_suffixes)

    if assay_order is None:
        assay_order=list(next(iter(task_orders.values())))
    assay_order=list(assay_order)

    # ---------------------------------------------------------
    # Detect seeds
    # ---------------------------------------------------------
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            match=seed_pattern.search(str(col))
            if match:
                seeds.append(match.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError(
            "No seed-specific columns ending in '_seed_<seed>' were found."
        )

    # ---------------------------------------------------------
    # Validate assays
    # ---------------------------------------------------------
    missing=[a for a in assay_order if a not in seeded_allele_ap]

    if missing:
        raise ValueError(
            f"Missing assay AP dataframes: {missing}"
        )

    def _score_col(order_name,stage,method,seed):
        return (
            f"{order_name}_s{stage}"
            f"{method_suffixes[method]}_seed_{seed}"
        )

    # ---------------------------------------------------------
    # Macro mAP over alleles
    # ---------------------------------------------------------
    maps={}

    for assay in assay_order:
        df=seeded_allele_ap[assay]
        numeric=df.select_dtypes(include=[np.number])

        if numeric.shape[1]==0:
            raise ValueError(
                f"{assay}: no numeric AP columns found."
            )

        maps[assay]=numeric.mean(
            axis=0,
            skipna=True
        )

    # ---------------------------------------------------------
    # Compute retention separately within every seed/order
    # ---------------------------------------------------------
    seed_rows=[]

    for order_name,order in task_orders.items():
        order=list(order)
        n_stages=len(order)

        for assay in assay_order:
            if assay not in order:
                raise ValueError(
                    f"{order_name}: assay {assay!r} is absent from task order."
                )

            acquisition_stage=order.index(assay)+1

            # No later checkpoint exists if learned last.
            if acquisition_stage==n_stages:
                continue

            if all_post_acquisition:
                eval_stages=list(
                    range(
                        acquisition_stage+1,
                        n_stages+1
                    )
                )
            else:
                eval_stages=[n_stages]

            for method in methods:
                for seed in seeds:
                    acquisition_col=_score_col(
                        order_name,
                        acquisition_stage,
                        method,
                        seed
                    )

                    if acquisition_col not in maps[assay].index:
                        raise ValueError(
                            f"{assay}: missing column "
                            f"{acquisition_col!r}."
                        )

                    acquisition_map=maps[assay][
                        acquisition_col
                    ]

                    for eval_stage in eval_stages:
                        eval_col=_score_col(
                            order_name,
                            eval_stage,
                            method,
                            seed
                        )

                        if eval_col not in maps[assay].index:
                            raise ValueError(
                                f"{assay}: missing column "
                                f"{eval_col!r}."
                            )

                        evaluation_map=maps[assay][
                            eval_col
                        ]

                        seed_rows.append({
                            "method":method,
                            "assay":assay,
                            "order":order_name,
                            "seed":seed,
                            "acquisition_stage":acquisition_stage,
                            "evaluation_stage":eval_stage,
                            "updates_after_acquisition":
                                eval_stage-acquisition_stage,
                            "acquisition_map":acquisition_map,
                            "evaluation_map":evaluation_map,
                            "retention":
                                evaluation_map-acquisition_map,
                        })

    seed_df=pd.DataFrame(seed_rows)

    if seed_df.empty:
        raise ValueError(
            "No valid post-acquisition retention observations were generated."
        )

    # ---------------------------------------------------------
    # Average replicate seeds for each unique context
    #
    # Unique point =
    #   method x assay x order x acquisition stage x evaluation stage
    # ---------------------------------------------------------
    group_cols=[
        "method",
        "assay",
        "order",
        "acquisition_stage",
        "evaluation_stage",
        "updates_after_acquisition",
    ]

    summary_rows=[]

    for keys,g in seed_df.groupby(
        group_cols,
        sort=False
    ):
        vals=g["retention"].dropna().to_numpy(float)
        n_seed=len(vals)

        mean=np.mean(vals) if n_seed else np.nan
        sd=np.std(vals,ddof=1) if n_seed>1 else np.nan
        se=sd/np.sqrt(n_seed) if n_seed>1 else np.nan

        ci95=(
            student_t.ppf(0.975,n_seed-1)*se
            if n_seed>1
            else np.nan
        )

        row=dict(zip(group_cols,keys))

        row.update({
            "n_seeds":n_seed,
            "mean_retention":mean,
            "sd_across_seeds":sd,
            "se_across_seeds":se,
            "ci95_halfwidth":ci95,
            "ci95_lower":
                mean-ci95 if n_seed>1 else np.nan,
            "ci95_upper":
                mean+ci95 if n_seed>1 else np.nan,
        })

        summary_rows.append(row)

    summary_df=pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # Determine n for each assay box.
    #
    # Count UNIQUE order/evaluation-stage contexts after seeds
    # have already been collapsed.
    #
    # Verify methods agree on n for each assay.
    # ---------------------------------------------------------
    context_counts=(
        summary_df
        .groupby(["method","assay"])
        .size()
        .unstack("method")
    )

    assay_n={}

    for assay in assay_order:
        if assay not in context_counts.index:
            assay_n[assay]=0
            continue

        counts=(
            context_counts
            .loc[assay]
            .dropna()
            .astype(int)
        )

        if len(counts)==0:
            assay_n[assay]=0

        elif counts.nunique()!=1:
            raise ValueError(
                f"{assay}: methods have different numbers of unique "
                f"retention contexts: {counts.to_dict()}"
            )

        else:
            assay_n[assay]=int(
                counts.iloc[0]
            )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    n_methods=len(methods)
    n_assays=len(assay_order)

    fig,ax=plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    default_colors=(
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    colors={
        method:(
            method_colors[method]
            if method_colors and method in method_colors
            else default_colors[i % len(default_colors)]
        )
        for i,method in enumerate(methods)
    }

    # Width allocated to all methods within one assay group.
    group_width=min(
        0.78,
        box_width*n_methods*1.45
    )

    if n_methods==1:
        offsets=np.array([0.0])
    else:
        offsets=np.linspace(
            -group_width/2,
            group_width/2,
            n_methods
        )

    centers=np.arange(n_assays)

    for method_i,method in enumerate(methods):
        positions=centers+offsets[method_i]

        data=[]

        for assay in assay_order:
            vals=summary_df.loc[
                (summary_df["method"]==method) &
                (summary_df["assay"]==assay),
                "mean_retention"
            ].dropna().to_numpy(float)

            data.append(vals)

        bp=ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            medianprops={
                "color":"black",
                "linewidth":1.2
            },
            whiskerprops={
                "linewidth":1
            },
            capprops={
                "linewidth":1
            },
            boxprops={
                "linewidth":1
            },
        )

        for box in bp["boxes"]:
            box.set_facecolor(
                colors[method]
            )
            box.set_alpha(0.30)
            box.set_edgecolor(
                colors[method]
            )

        for whisker in bp["whiskers"]:
            whisker.set_color(
                colors[method]
            )

        for cap in bp["caps"]:
            cap.set_color(
                colors[method]
            )

        # Overlay the unique seed-averaged context points.
        if show_points:
            for assay_i,vals in enumerate(data):
                if len(vals)==0:
                    continue

                if len(vals)==1:
                    jitter=np.array([0.0])
                else:
                    jitter=np.linspace(
                        -box_width*0.22,
                        box_width*0.22,
                        len(vals)
                    )

                ax.scatter(
                    positions[assay_i]+jitter,
                    vals,
                    s=point_size,
                    alpha=point_alpha,
                    color=colors[method],
                    zorder=3
                )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    ax.axhline(
        0,
        ls="--",
        lw=1
    )

    ax.set_xticks(centers)

    ax.set_xticklabels([
        f"{assay} (n={assay_n[assay]})"
        for assay in assay_order
    ])

    ax.set_xlabel(
        "Previously learned assay"
    )

    ax.set_ylabel(
        "mAP - acquisition mAP"
    )

    if all_post_acquisition:
        ax.set_title(
            "Task-specific retention across all post-acquisition checkpoints"
        )
    else:
        ax.set_title(
            "Task-specific final retention across task orders"
        )

    # ---------------------------------------------------------
    # Legend below plot in approximately two rows
    # ---------------------------------------------------------
    legend_handles=[
        Patch(
            facecolor=colors[method],
            edgecolor=colors[method],
            alpha=0.35,
            label=method
        )
        for method in methods
    ]

    ncols=int(
        np.ceil(
            len(methods)/legend_nrows
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5,-0.13),
        ncol=ncols,
        frameon=True
    )

    fig.subplots_adjust(
        bottom=0.23
    )

    if show:
        plt.show()

    return summary_df, fig
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import t as student_t

def plot_seeded_task_forgetting_boxplots(
    seeded_allele_ap, task_orders, method_suffixes,
    assay_order=None, allele_col="allele",
    all_post_acquisition=False, method_colors=None,
    figsize=(16,7), dpi=150, box_width=0.10,
    point_size=22, point_alpha=0.75,
    legend_nrows=2, show_points=True, show=True
):
    """
    Plot task-specific forgetting distributions across task orders.

    Forgetting is calculated separately within each seed:

        forgetting(stage t)
            = best mAP achieved from acquisition through stage t
              - mAP at stage t

    Seeds are then averaged for each unique order/checkpoint context.

    IMPORTANT
    ---------
    Each plotted point is a UNIQUE order/checkpoint context, not a seed.

    Seeds are replicate estimates of that context and are averaged before
    plotting.

    Modes
    -----
    all_post_acquisition=False
        Use final forgetting only:

            max(mAP from acquisition through final)
            - final mAP

        Orders where the assay is acquired last are excluded because there
        are no later updates and forgetting is structurally 0.

    all_post_acquisition=True
        Use every checkpoint strictly AFTER acquisition.

        At each later stage t:

            max(mAP from acquisition through t)
            - mAP at t

        The acquisition checkpoint itself is excluded because forgetting
        there is necessarily 0.

        Orders where the assay is acquired last contribute no observations.

    The x-axis labels report n for each assay, where n is the number of
    unique order/checkpoint contexts contributing to each assay/method box,
    after collapsing seed replicates.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per method x assay x order x evaluation checkpoint after
        averaging across seeds.

    fig : matplotlib.figure.Figure
    """
    methods=list(method_suffixes)

    if assay_order is None:
        assay_order=list(next(iter(task_orders.values())))
    assay_order=list(assay_order)

    # ---------------------------------------------------------
    # Detect seeds
    # ---------------------------------------------------------
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            match=seed_pattern.search(str(col))
            if match:
                seeds.append(match.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError(
            "No seed-specific columns ending in '_seed_<seed>' were found."
        )

    # ---------------------------------------------------------
    # Validate assays
    # ---------------------------------------------------------
    missing=[a for a in assay_order if a not in seeded_allele_ap]

    if missing:
        raise ValueError(
            f"Missing assay AP dataframes: {missing}"
        )

    def _score_col(order_name,stage,method,seed):
        return (
            f"{order_name}_s{stage}"
            f"{method_suffixes[method]}_seed_{seed}"
        )

    # ---------------------------------------------------------
    # Macro mAP over alleles
    # ---------------------------------------------------------
    maps={}

    for assay in assay_order:
        df=seeded_allele_ap[assay]
        numeric=df.select_dtypes(include=[np.number])

        if numeric.shape[1]==0:
            raise ValueError(
                f"{assay}: no numeric AP columns found."
            )

        maps[assay]=numeric.mean(
            axis=0,
            skipna=True
        )

    # ---------------------------------------------------------
    # Compute forgetting separately within every seed/order
    # ---------------------------------------------------------
    seed_rows=[]

    for order_name,order in task_orders.items():
        order=list(order)
        n_stages=len(order)

        for assay in assay_order:
            if assay not in order:
                raise ValueError(
                    f"{order_name}: assay {assay!r} is absent from task order."
                )

            acquisition_stage=order.index(assay)+1

            # No forgetting opportunity if acquired at final stage.
            if acquisition_stage==n_stages:
                continue

            if all_post_acquisition:
                eval_stages=list(
                    range(
                        acquisition_stage+1,
                        n_stages+1
                    )
                )
            else:
                eval_stages=[n_stages]

            for method in methods:
                for seed in seeds:

                    # Collect assay trajectory from acquisition onward.
                    stage_values={}

                    for stage in range(
                        acquisition_stage,
                        n_stages+1
                    ):
                        col=_score_col(
                            order_name,
                            stage,
                            method,
                            seed
                        )

                        if col not in maps[assay].index:
                            raise ValueError(
                                f"{assay}: missing column {col!r}."
                            )

                        stage_values[stage]=maps[assay][col]

                    for eval_stage in eval_stages:
                        history=np.array([
                            stage_values[s]
                            for s in range(
                                acquisition_stage,
                                eval_stage+1
                            )
                        ], dtype=float)

                        historical_best=np.nanmax(history)
                        current_map=stage_values[eval_stage]
                        forgetting=historical_best-current_map

                        seed_rows.append({
                            "method":method,
                            "assay":assay,
                            "order":order_name,
                            "seed":seed,
                            "acquisition_stage":acquisition_stage,
                            "evaluation_stage":eval_stage,
                            "updates_after_acquisition":
                                eval_stage-acquisition_stage,
                            "historical_best_map":historical_best,
                            "evaluation_map":current_map,
                            "forgetting":forgetting,
                        })

    seed_df=pd.DataFrame(seed_rows)

    if seed_df.empty:
        raise ValueError(
            "No valid post-acquisition forgetting observations were generated."
        )

    # ---------------------------------------------------------
    # Average seed replicates for each unique context
    # ---------------------------------------------------------
    group_cols=[
        "method",
        "assay",
        "order",
        "acquisition_stage",
        "evaluation_stage",
        "updates_after_acquisition",
    ]

    summary_rows=[]

    for keys,g in seed_df.groupby(
        group_cols,
        sort=False
    ):
        vals=g["forgetting"].dropna().to_numpy(float)
        n_seed=len(vals)

        mean=np.mean(vals) if n_seed else np.nan
        sd=np.std(vals,ddof=1) if n_seed>1 else np.nan
        se=sd/np.sqrt(n_seed) if n_seed>1 else np.nan

        ci95=(
            student_t.ppf(0.975,n_seed-1)*se
            if n_seed>1
            else np.nan
        )

        row=dict(zip(group_cols,keys))

        row.update({
            "n_seeds":n_seed,
            "mean_forgetting":mean,
            "sd_across_seeds":sd,
            "se_across_seeds":se,
            "ci95_halfwidth":ci95,
            "ci95_lower":
                mean-ci95 if n_seed>1 else np.nan,
            "ci95_upper":
                mean+ci95 if n_seed>1 else np.nan,
        })

        summary_rows.append(row)

    summary_df=pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # Determine n per assay box
    #
    # n = number of unique order/checkpoint contexts,
    # NOT number of seeds.
    # ---------------------------------------------------------
    context_counts=(
        summary_df
        .groupby(["method","assay"])
        .size()
        .unstack("method")
    )

    assay_n={}

    for assay in assay_order:
        if assay not in context_counts.index:
            assay_n[assay]=0
            continue

        counts=(
            context_counts
            .loc[assay]
            .dropna()
            .astype(int)
        )

        if len(counts)==0:
            assay_n[assay]=0

        elif counts.nunique()!=1:
            raise ValueError(
                f"{assay}: methods have different numbers of unique "
                f"forgetting contexts: {counts.to_dict()}"
            )

        else:
            assay_n[assay]=int(
                counts.iloc[0]
            )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    n_methods=len(methods)
    n_assays=len(assay_order)

    fig,ax=plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    default_colors=(
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    colors={
        method:(
            method_colors[method]
            if method_colors and method in method_colors
            else default_colors[i % len(default_colors)]
        )
        for i,method in enumerate(methods)
    }

    group_width=min(
        0.78,
        box_width*n_methods*1.45
    )

    if n_methods==1:
        offsets=np.array([0.0])
    else:
        offsets=np.linspace(
            -group_width/2,
            group_width/2,
            n_methods
        )

    centers=np.arange(n_assays)

    for method_i,method in enumerate(methods):
        positions=centers+offsets[method_i]

        data=[]

        for assay in assay_order:
            vals=summary_df.loc[
                (summary_df["method"]==method) &
                (summary_df["assay"]==assay),
                "mean_forgetting"
            ].dropna().to_numpy(float)

            data.append(vals)

        bp=ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            medianprops={
                "color":"black",
                "linewidth":1.2
            },
            whiskerprops={
                "linewidth":1
            },
            capprops={
                "linewidth":1
            },
            boxprops={
                "linewidth":1
            },
        )

        for box in bp["boxes"]:
            box.set_facecolor(
                colors[method]
            )
            box.set_alpha(0.30)
            box.set_edgecolor(
                colors[method]
            )

        for whisker in bp["whiskers"]:
            whisker.set_color(
                colors[method]
            )

        for cap in bp["caps"]:
            cap.set_color(
                colors[method]
            )

        if show_points:
            for assay_i,vals in enumerate(data):
                if len(vals)==0:
                    continue

                if len(vals)==1:
                    jitter=np.array([0.0])
                else:
                    jitter=np.linspace(
                        -box_width*0.22,
                        box_width*0.22,
                        len(vals)
                    )

                ax.scatter(
                    positions[assay_i]+jitter,
                    vals,
                    s=point_size,
                    alpha=point_alpha,
                    color=colors[method],
                    zorder=3
                )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    ax.axhline(
        0,
        ls="--",
        lw=1
    )

    ax.set_xticks(centers)

    ax.set_xticklabels([
        f"{assay} (n={assay_n[assay]})"
        for assay in assay_order
    ])

    ax.set_xlabel(
        "Previously learned assay"
    )

    ax.set_ylabel(
        "Forgetting (mAP)"
    )

    if all_post_acquisition:
        ax.set_title(
            "Task-specific forgetting across all post-acquisition checkpoints"
        )
    else:
        ax.set_title(
            "Task-specific final forgetting across task orders"
        )

    # ---------------------------------------------------------
    # Legend below plot, ~2 rows
    # ---------------------------------------------------------
    legend_handles=[
        Patch(
            facecolor=colors[method],
            edgecolor=colors[method],
            alpha=0.35,
            label=method
        )
        for method in methods
    ]

    ncols=int(
        np.ceil(
            len(methods)/legend_nrows
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5,-0.13),
        ncol=ncols,
        frameon=True
    )

    fig.subplots_adjust(
        bottom=0.23
    )

    if show:
        plt.show()

    return summary_df, fig
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import t as student_t

def plot_seeded_task_acquisition_boxplots(
    seeded_allele_ap, task_orders, method_suffixes,
    assay_order=None, allele_col="allele", method_colors=None,
    figsize=(16,7), dpi=150, box_width=0.10,
    point_size=22, point_alpha=0.75,
    legend_nrows=2, show_points=True, show=True
):
    """
    Plot task-specific acquisition mAP distributions across task orders.

    Acquisition performance is calculated within each seed:

        acquisition mAP = assay mAP immediately after that assay is updated

    Seeds are then averaged for each unique task-order/acquisition context.

    IMPORTANT
    ---------
    The plotted points are NOT seed replicates.

    Each plotted point corresponds to one task-order context after averaging
    the replicate seeds for that exact assay acquisition.

    Unlike retention plots, all acquisition positions are valid, including
    assays acquired at the final stage.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per method x assay x order after averaging across seeds.

    fig : matplotlib.figure.Figure
    """
    methods=list(method_suffixes)

    if assay_order is None:
        assay_order=list(next(iter(task_orders.values())))
    assay_order=list(assay_order)

    # ---------------------------------------------------------
    # Detect seeds
    # ---------------------------------------------------------
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            match=seed_pattern.search(str(col))
            if match:
                seeds.append(match.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError("No seed-specific columns ending in '_seed_<seed>' were found.")

    # ---------------------------------------------------------
    # Validate assays
    # ---------------------------------------------------------
    missing=[a for a in assay_order if a not in seeded_allele_ap]
    if missing:
        raise ValueError(f"Missing assay AP dataframes: {missing}")

    def _score_col(order_name,stage,method,seed):
        return f"{order_name}_s{stage}{method_suffixes[method]}_seed_{seed}"

    # ---------------------------------------------------------
    # Macro mAP across alleles for every score column
    # ---------------------------------------------------------
    maps={}

    for assay in assay_order:
        df=seeded_allele_ap[assay]
        numeric=df.select_dtypes(include=[np.number])

        if numeric.shape[1]==0:
            raise ValueError(f"{assay}: no numeric AP columns found.")

        maps[assay]=numeric.mean(axis=0,skipna=True)

    # ---------------------------------------------------------
    # Acquisition mAP within every seed x order
    # ---------------------------------------------------------
    seed_rows=[]

    for order_name,order in task_orders.items():
        order=list(order)

        for assay in assay_order:
            if assay not in order:
                raise ValueError(f"{order_name}: assay {assay!r} is absent from task order.")

            acquisition_stage=order.index(assay)+1

            for method in methods:
                for seed in seeds:
                    acquisition_col=_score_col(
                        order_name, acquisition_stage, method, seed
                    )

                    if acquisition_col not in maps[assay].index:
                        raise ValueError(
                            f"{assay}: missing column {acquisition_col!r}."
                        )

                    seed_rows.append({
                        "method":method,
                        "assay":assay,
                        "order":order_name,
                        "seed":seed,
                        "acquisition_stage":acquisition_stage,
                        "acquisition_map":maps[assay][acquisition_col],
                    })

    seed_df=pd.DataFrame(seed_rows)

    if seed_df.empty:
        raise ValueError("No valid acquisition observations were generated.")

    # ---------------------------------------------------------
    # Average seeds within each acquisition context
    # Unique plotted point = method x assay x order x acquisition stage
    # ---------------------------------------------------------
    group_cols=["method","assay","order","acquisition_stage"]
    summary_rows=[]

    for keys,g in seed_df.groupby(group_cols,sort=False):
        vals=g["acquisition_map"].dropna().to_numpy(float)
        n_seed=len(vals)

        mean=np.mean(vals) if n_seed else np.nan
        sd=np.std(vals,ddof=1) if n_seed>1 else np.nan
        se=sd/np.sqrt(n_seed) if n_seed>1 else np.nan
        ci95=student_t.ppf(0.975,n_seed-1)*se if n_seed>1 else np.nan

        row=dict(zip(group_cols,keys))
        row.update({
            "n_seeds":n_seed,
            "mean_acquisition_map":mean,
            "sd_across_seeds":sd,
            "se_across_seeds":se,
            "ci95_halfwidth":ci95,
            "ci95_lower":mean-ci95 if n_seed>1 else np.nan,
            "ci95_upper":mean+ci95 if n_seed>1 else np.nan,
        })
        summary_rows.append(row)

    summary_df=pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # Count unique acquisition contexts per assay after seeds collapse
    # ---------------------------------------------------------
    context_counts=(
        summary_df.groupby(["method","assay"]).size().unstack("method")
    )

    assay_n={}

    for assay in assay_order:
        if assay not in context_counts.index:
            assay_n[assay]=0
            continue

        counts=context_counts.loc[assay].dropna().astype(int)

        if len(counts)==0:
            assay_n[assay]=0
        elif counts.nunique()!=1:
            raise ValueError(
                f"{assay}: methods have different numbers of acquisition "
                f"contexts: {counts.to_dict()}"
            )
        else:
            assay_n[assay]=int(counts.iloc[0])

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    n_methods=len(methods)
    n_assays=len(assay_order)

    fig,ax=plt.subplots(figsize=figsize,dpi=dpi)

    default_colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors={
        method:(
            method_colors[method]
            if method_colors and method in method_colors
            else default_colors[i%len(default_colors)]
        )
        for i,method in enumerate(methods)
    }

    group_width=min(0.78,box_width*n_methods*1.45)

    if n_methods==1:
        offsets=np.array([0.0])
    else:
        offsets=np.linspace(-group_width/2,group_width/2,n_methods)

    centers=np.arange(n_assays)

    for method_i,method in enumerate(methods):
        positions=centers+offsets[method_i]
        data=[]

        for assay in assay_order:
            vals=summary_df.loc[
                (summary_df["method"]==method) &
                (summary_df["assay"]==assay),
                "mean_acquisition_map"
            ].dropna().to_numpy(float)

            data.append(vals)

        bp=ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            medianprops={"color":"black","linewidth":1.2},
            whiskerprops={"linewidth":1},
            capprops={"linewidth":1},
            boxprops={"linewidth":1},
        )

        for box in bp["boxes"]:
            box.set_facecolor(colors[method])
            box.set_alpha(0.30)
            box.set_edgecolor(colors[method])

        for whisker in bp["whiskers"]:
            whisker.set_color(colors[method])

        for cap in bp["caps"]:
            cap.set_color(colors[method])

        # Seed-averaged acquisition contexts
        if show_points:
            for assay_i,vals in enumerate(data):
                if len(vals)==0:
                    continue

                jitter=(
                    np.array([0.0])
                    if len(vals)==1
                    else np.linspace(-box_width*0.22,box_width*0.22,len(vals))
                )

                ax.scatter(
                    positions[assay_i]+jitter,
                    vals,
                    s=point_size,
                    alpha=point_alpha,
                    color=colors[method],
                    zorder=3
                )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    ax.set_xticks(centers)
    ax.set_xticklabels([
        f"{assay} (n={assay_n[assay]})"
        for assay in assay_order
    ])

    ax.set_xlabel("Acquired assay")
    ax.set_ylabel("Acquisition mAP")
    ax.set_title("Task-specific acquisition performance across task orders")

    legend_handles=[
        Patch(
            facecolor=colors[method],
            edgecolor=colors[method],
            alpha=0.35,
            label=method
        )
        for method in methods
    ]

    ncols=int(np.ceil(len(methods)/legend_nrows))

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5,-0.13),
        ncol=ncols,
        frameon=True
    )

    fig.subplots_adjust(bottom=0.23)

    if show:
        plt.show()

    return summary_df,fig
# import re
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.colors import to_rgba
# from scipy.stats import t as student_t

# def plot_seeded_task_metric_boxplots(
#     seeded_allele_ap, task_orders, method_suffixes,
#     metric="acquisition", retention_mode="final", base_score_col="mhcprime",
#     method_labels=None, assay_labels=None, method_colors=None,

#     # Figure / fonts
#     figsize=(3.6,1.65), dpi=300, font_family="Nimbus Sans",
#     title=None, xlabel=None, ylabel=None,
#     title_fontsize=6, xlabel_fontsize=6, ylabel_fontsize=6,
#     xtick_fontsize=5.5, ytick_fontsize=6,
#     title_bold=False, title_pad=4,
#     xlabel_pad=1, ylabel_pad=1.5,

#     # Box geometry / styling
#     box_width=0.24, group_width=0.32, box_alpha=1.0,
#     box_facecolor_mode="method", box_edgecolor_mode="black",
#     box_linewidth=0.40,
#     median_color="black", median_linewidth=0.60, median_linestyle="-",
#     whisker_color_mode="black", whisker_linewidth=0.40, whisker_linestyle="-",
#     cap_color_mode="black", cap_linewidth=0.40,
#     show_caps=False, show_fliers=False, flier_size=2,
#     flier_marker="o", flier_alpha=0.7,
#     show_mean=False, mean_marker="D", mean_marker_size=3, mean_color="black",

#     # Context points - scalar OR {method:value}
#     show_points=True,
#     point_colors=None,
#     point_edgecolors="none",
#     point_edgewidths=0.0,
#     point_alphas=0.75,
#     point_sizes=10,
#     point_markers="o",
#     point_jitter_fraction=0.20,
#     point_zorder=3,

#     # Axes
#     ylim=None, xlim=None,
#     axis_linewidth=0.35,
#     show_top_spine=False, show_right_spine=False,
#     show_left_spine=True, show_bottom_spine=True,
#     tick_length=2.2, tick_width=0.35,
#     tick_direction="out", tick_pad=2,
#     xtick_rotation=0,

#     # Grid
#     grid=False, grid_axis="y", grid_alpha=0.0,
#     grid_linewidth=0.35, grid_linestyle="-",

#     # Optional horizontal reference
#     show_reference_line=None,
#     reference_y=0.0, reference_color="black",
#     reference_linewidth=0.35, reference_linestyle="--",
#     reference_zorder=0,

#     # Legend
#     show_legend=True, legend_nrows=1, legend_ncol=None,
#     legend_loc="upper center", legend_bbox=(0.5,-0.18),
#     legend_fontsize=6, legend_frame=False,
#     legend_title=None, legend_title_fontsize=6,
#     legend_box_alpha=1.0,
#     legend_handlelength=0.8, legend_columnspacing=1.0,

#     # Labels / layout
#     show_n=True, n_format="{assay} (n={n})",
#     left_margin=None, right_margin=None, top_margin=None, bottom_margin=None,

#     # Output
#     save_path=None, save_pdf=True, save_png=True,
#     bbox_inches="tight", show=True,
# ):
#     """
#     Plot seed-averaged task-context distributions.

#     metric:
#         "acquisition"
#             acquisition mAP immediately after the assay is learned.

#         "acquisition_delta"
#             acquisition mAP - MHCPrime mAP for that assay.

#         "retention"
#             retention_mode="final":
#                 final mAP - acquisition mAP.

#             retention_mode="all":
#                 every later checkpoint mAP - acquisition mAP.

#     Each plotted point is one unique task-order/context after averaging
#     seed/split replicates within that context.

#     point_colors, point_edgecolors, point_edgewidths, point_alphas,
#     point_sizes, and point_markers may each be either:
#         - one value applied to all methods, or
#         - {method_name: value}.

#     Returns
#     -------
#     summary_df, fig, ax, seed_df
#     """
#     valid_metrics={"acquisition","acquisition_delta","retention"}
#     if metric not in valid_metrics:
#         raise ValueError(f"metric must be one of {sorted(valid_metrics)}.")
#     if retention_mode not in {"final","all"}:
#         raise ValueError("retention_mode must be 'final' or 'all'.")
#     if not task_orders:
#         raise ValueError("task_orders cannot be empty.")
#     if not method_suffixes:
#         raise ValueError("method_suffixes cannot be empty.")

#     methods=list(method_suffixes)
#     method_labels={} if method_labels is None else dict(method_labels)
#     assay_labels={} if assay_labels is None else dict(assay_labels)

#     # ---------------------------------------------------------
#     # Assays are inferred from task_orders. All orders are used.
#     # ---------------------------------------------------------
#     assays=list(next(iter(task_orders.values())))
#     expected_assays=set(assays)

#     for order_name,order in task_orders.items():
#         order=list(order)
#         if len(order)!=len(assays) or set(order)!=expected_assays:
#             raise ValueError(
#                 f"{order_name}: expected assay set {assays}, got {order}."
#             )

#     missing=[a for a in assays if a not in seeded_allele_ap]
#     if missing:
#         raise ValueError(f"Missing assay AP dataframes: {missing}")

#     # ---------------------------------------------------------
#     # Detect seeds
#     # ---------------------------------------------------------
#     seed_pattern=re.compile(r"_seed_(.+)$")
#     seeds=[]

#     for assay in assays:
#         for col in seeded_allele_ap[assay].columns:
#             m=seed_pattern.search(str(col))
#             if m:
#                 seeds.append(m.group(1))

#     seeds=list(dict.fromkeys(seeds))

#     if not seeds:
#         raise ValueError("No score columns ending in '_seed_<seed>' were found.")

#     def _score_col(order_name,stage,method,seed):
#         return f"{order_name}_s{stage}{method_suffixes[method]}_seed_{seed}"

#     # ---------------------------------------------------------
#     # Macro mAP over alleles
#     # ---------------------------------------------------------
#     maps={}

#     for assay in assays:
#         df=seeded_allele_ap[assay]

#         if base_score_col not in df.columns:
#             raise ValueError(f"{assay}: missing base column {base_score_col!r}.")

#         numeric=df.select_dtypes(include=[np.number])
#         if numeric.shape[1]==0:
#             raise ValueError(f"{assay}: no numeric AP columns found.")

#         maps[assay]=numeric.mean(axis=0,skipna=True)

#     # ---------------------------------------------------------
#     # Seed-level values
#     # ---------------------------------------------------------
#     seed_rows=[]

#     for order_name,order in task_orders.items():
#         order=list(order)
#         n_stages=len(order)

#         for assay in assays:
#             acquisition_stage=order.index(assay)+1
#             base_map=float(maps[assay][base_score_col])

#             for method in methods:
#                 for seed in seeds:
#                     acquisition_col=_score_col(
#                         order_name, acquisition_stage, method, seed
#                     )

#                     if acquisition_col not in maps[assay].index:
#                         raise ValueError(f"{assay}: missing column {acquisition_col!r}.")

#                     acquisition_map=float(maps[assay][acquisition_col])

#                     if metric in {"acquisition","acquisition_delta"}:
#                         value=(
#                             acquisition_map
#                             if metric=="acquisition"
#                             else acquisition_map-base_map
#                         )

#                         seed_rows.append({
#                             "method":method,
#                             "assay":assay,
#                             "order":order_name,
#                             "seed":seed,
#                             "acquisition_stage":acquisition_stage,
#                             "evaluation_stage":acquisition_stage,
#                             "updates_after_acquisition":0,
#                             "base_map":base_map,
#                             "acquisition_map":acquisition_map,
#                             "evaluation_map":acquisition_map,
#                             "value":value,
#                         })
#                         continue

#                     # Retention
#                     if acquisition_stage==n_stages:
#                         continue

#                     eval_stages=(
#                         list(range(acquisition_stage+1,n_stages+1))
#                         if retention_mode=="all"
#                         else [n_stages]
#                     )

#                     for eval_stage in eval_stages:
#                         eval_col=_score_col(order_name,eval_stage,method,seed)

#                         if eval_col not in maps[assay].index:
#                             raise ValueError(f"{assay}: missing column {eval_col!r}.")

#                         evaluation_map=float(maps[assay][eval_col])

#                         seed_rows.append({
#                             "method":method,
#                             "assay":assay,
#                             "order":order_name,
#                             "seed":seed,
#                             "acquisition_stage":acquisition_stage,
#                             "evaluation_stage":eval_stage,
#                             "updates_after_acquisition":eval_stage-acquisition_stage,
#                             "base_map":base_map,
#                             "acquisition_map":acquisition_map,
#                             "evaluation_map":evaluation_map,
#                             "value":evaluation_map-acquisition_map,
#                         })

#     seed_df=pd.DataFrame(seed_rows)

#     if seed_df.empty:
#         raise ValueError(f"No valid observations generated for metric={metric!r}.")

#     # ---------------------------------------------------------
#     # Average seeds within unique plotted context
#     # ---------------------------------------------------------
#     group_cols=[
#         "method","assay","order","acquisition_stage",
#         "evaluation_stage","updates_after_acquisition"
#     ]

#     summary_rows=[]

#     for keys,g in seed_df.groupby(group_cols,sort=False):
#         vals=g["value"].dropna().to_numpy(float)
#         n=len(vals)

#         mean=vals.mean() if n else np.nan
#         sd=vals.std(ddof=1) if n>1 else np.nan
#         se=sd/np.sqrt(n) if n>1 else np.nan
#         ci_half=student_t.ppf(.975,n-1)*se if n>1 else np.nan

#         row=dict(zip(group_cols,keys))
#         row.update({
#             "n_seeds":n,
#             "mean_value":mean,
#             "sd_across_seeds":sd,
#             "se_across_seeds":se,
#             "ci95_halfwidth":ci_half,
#             "ci95_lower":mean-ci_half if n>1 else np.nan,
#             "ci95_upper":mean+ci_half if n>1 else np.nan,
#             "mean_base_map":g["base_map"].mean(),
#             "mean_acquisition_map":g["acquisition_map"].mean(),
#             "mean_evaluation_map":g["evaluation_map"].mean(),
#         })
#         summary_rows.append(row)

#     summary_df=pd.DataFrame(summary_rows)

#     # ---------------------------------------------------------
#     # Context counts
#     # ---------------------------------------------------------
#     context_counts=(
#         summary_df.groupby(["method","assay"])
#         .size()
#         .unstack("method")
#     )

#     assay_n={}

#     for assay in assays:
#         if assay not in context_counts.index:
#             assay_n[assay]=0
#             continue

#         counts=context_counts.loc[assay].dropna().astype(int)

#         if len(counts)==0:
#             assay_n[assay]=0
#         elif counts.nunique()!=1:
#             raise ValueError(
#                 f"{assay}: methods have different context counts: {counts.to_dict()}"
#             )
#         else:
#             assay_n[assay]=int(counts.iloc[0])

#     # ---------------------------------------------------------
#     # Matplotlib styling
#     # ---------------------------------------------------------
#     matplotlib.rcParams["pdf.fonttype"]=42
#     matplotlib.rcParams["ps.fonttype"]=42
#     matplotlib.rcParams["font.family"]="sans-serif"
#     matplotlib.rcParams["font.sans-serif"]=[
#         font_family,"Nimbus Sans","Nimbus Sans L",
#         "Arial","Helvetica","DejaVu Sans"
#     ]

#     default_colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]

#     colors={
#         method:(
#             method_colors[method]
#             if method_colors is not None and method in method_colors
#             else default_colors[i%len(default_colors)]
#         )
#         for i,method in enumerate(methods)
#     }

#     def _method_setting(value,method,default=None):
#         if isinstance(value,dict):
#             return value.get(method,default)
#         return value

#     def _component_color(mode,method,default):
#         if isinstance(mode,dict):
#             return mode.get(method,default)
#         if mode=="method":
#             return colors[method]
#         return mode

#     point_face={
#         method:_method_setting(point_colors,method,colors[method])
#         if point_colors is not None else colors[method]
#         for method in methods
#     }

#     point_edge={
#         method:_method_setting(point_edgecolors,method,"none")
#         for method in methods
#     }

#     point_lw={
#         method:_method_setting(point_edgewidths,method,0.0)
#         for method in methods
#     }

#     point_alpha_final={
#         method:_method_setting(point_alphas,method,1.0)
#         for method in methods
#     }

#     point_size_final={
#         method:_method_setting(point_sizes,method,10)
#         for method in methods
#     }

#     point_marker_final={
#         method:_method_setting(point_markers,method,"o")
#         for method in methods
#     }

#     # ---------------------------------------------------------
#     # Geometry
#     # ---------------------------------------------------------
#     n_methods=len(methods)
#     n_assays=len(assays)

#     if group_width is None:
#         group_width=min(0.78,box_width*n_methods*1.45)

#     offsets=(
#         np.array([0.0])
#         if n_methods==1
#         else np.linspace(-group_width/2,group_width/2,n_methods)
#     )

#     centers=np.arange(n_assays,dtype=float)

#     fig,ax=plt.subplots(figsize=figsize,dpi=dpi)

#     # ---------------------------------------------------------
#     # Boxes
#     # ---------------------------------------------------------
#     for method_i,method in enumerate(methods):
#         positions=centers+offsets[method_i]
#         data=[]

#         for assay in assays:
#             vals=summary_df.loc[
#                 (summary_df["method"]==method)&
#                 (summary_df["assay"]==assay),
#                 "mean_value"
#             ].dropna().to_numpy(float)

#             data.append(vals)

#         edge_color=_component_color(
#             box_edgecolor_mode,method,colors[method]
#         )
#         whisker_color=_component_color(
#             whisker_color_mode,method,colors[method]
#         )
#         cap_color=_component_color(
#             cap_color_mode,method,colors[method]
#         )

#         bp=ax.boxplot(
#             data,
#             positions=positions,
#             widths=box_width,
#             patch_artist=True,
#             manage_ticks=False,
#             showcaps=show_caps,
#             showfliers=show_fliers,
#             showmeans=show_mean,

#             boxprops={
#                 "linewidth":box_linewidth,
#                 "edgecolor":edge_color,
#             },
#             medianprops={
#                 "color":median_color,
#                 "linewidth":median_linewidth,
#                 "linestyle":median_linestyle,
#             },
#             whiskerprops={
#                 "color":whisker_color,
#                 "linewidth":whisker_linewidth,
#                 "linestyle":whisker_linestyle,
#             },
#             capprops={
#                 "color":cap_color,
#                 "linewidth":cap_linewidth,
#             },
#             flierprops={
#                 "marker":flier_marker,
#                 "markersize":flier_size,
#                 "markerfacecolor":colors[method],
#                 "markeredgecolor":colors[method],
#                 "alpha":flier_alpha,
#             },
#             meanprops={
#                 "marker":mean_marker,
#                 "markersize":mean_marker_size,
#                 "markerfacecolor":mean_color,
#                 "markeredgecolor":mean_color,
#             },
#         )

#         # Fill alpha only; edge remains fully opaque
#         for box in bp["boxes"]:
#             if box_facecolor_mode=="method":
#                 face=to_rgba(colors[method],box_alpha)
#             elif box_facecolor_mode is None or box_facecolor_mode=="none":
#                 face="none"
#             else:
#                 face=to_rgba(box_facecolor_mode,box_alpha)

#             box.set_facecolor(face)
#             box.set_edgecolor(edge_color)
#             box.set_linewidth(box_linewidth)
#             box.set_alpha(None)

#         # -----------------------------------------------------
#         # Seed-averaged context points
#         # -----------------------------------------------------
#         if show_points:
#             for assay_i,vals in enumerate(data):
#                 if len(vals)==0:
#                     continue

#                 jitter=(
#                     np.array([0.0])
#                     if len(vals)==1
#                     else np.linspace(
#                         -box_width*point_jitter_fraction,
#                         box_width*point_jitter_fraction,
#                         len(vals)
#                     )
#                 )

#                 ax.scatter(
#                     positions[assay_i]+jitter,
#                     vals,
#                     s=point_size_final[method],
#                     marker=point_marker_final[method],
#                     facecolors=point_face[method],
#                     edgecolors=point_edge[method],
#                     linewidths=point_lw[method],
#                     alpha=point_alpha_final[method],
#                     zorder=point_zorder,
#                 )

#     # ---------------------------------------------------------
#     # Metric defaults
#     # ---------------------------------------------------------
#     if show_reference_line is None:
#         show_reference_line=metric in {"acquisition_delta","retention"}

#     if title is None:
#         if metric=="acquisition":
#             title="Task-specific acquisition mAP"
#         elif metric=="acquisition_delta":
#             title="Task-specific acquisition ΔmAP vs MHCPrime"
#         elif retention_mode=="all":
#             title="Task-specific retention across post-acquisition checkpoints"
#         else:
#             title="Task-specific final retention"

#     if xlabel is None:
#         xlabel="Assay"

#     if ylabel is None:
#         if metric=="acquisition":
#             ylabel="Acquisition mAP"
#         elif metric=="acquisition_delta":
#             ylabel="Acquisition ΔmAP vs MHCPrime"
#         else:
#             ylabel="mAP - acquisition mAP"

#     # ---------------------------------------------------------
#     # Reference / grid
#     # ---------------------------------------------------------
#     if show_reference_line:
#         ax.axhline(
#             reference_y,
#             color=reference_color,
#             linewidth=reference_linewidth,
#             linestyle=reference_linestyle,
#             zorder=reference_zorder
#         )

#     if grid:
#         ax.grid(
#             True,
#             axis=grid_axis,
#             alpha=grid_alpha,
#             linewidth=grid_linewidth,
#             linestyle=grid_linestyle,
#             zorder=0
#         )

#     # ---------------------------------------------------------
#     # X labels
#     # ---------------------------------------------------------
#     ax.set_xticks(centers)

#     tick_labels=[]
#     for assay in assays:
#         label=assay_labels.get(assay,assay)

#         if show_n:
#             label=n_format.format(
#                 assay=label,
#                 n=assay_n[assay]
#             )

#         tick_labels.append(label)

#     ax.set_xticklabels(
#         tick_labels,
#         rotation=xtick_rotation,
#         fontsize=xtick_fontsize
#     )

#     # ---------------------------------------------------------
#     # Axes / labels
#     # ---------------------------------------------------------
#     ax.tick_params(
#         axis="both",
#         length=tick_length,
#         width=tick_width,
#         direction=tick_direction,
#         pad=tick_pad
#     )
#     ax.tick_params(axis="y",labelsize=ytick_fontsize)

#     ax.set_xlabel(
#         xlabel,
#         fontsize=xlabel_fontsize,
#         fontfamily=font_family,
#         labelpad=xlabel_pad
#     )
#     ax.set_ylabel(
#         ylabel,
#         fontsize=ylabel_fontsize,
#         fontfamily=font_family,
#         labelpad=ylabel_pad
#     )
#     ax.set_title(
#         title,
#         fontsize=title_fontsize,
#         fontfamily=font_family,
#         fontweight="bold" if title_bold else "normal",
#         pad=title_pad
#     )

#     if ylim is not None:
#         ax.set_ylim(ylim)
#     if xlim is not None:
#         ax.set_xlim(xlim)

#     # ---------------------------------------------------------
#     # Spines
#     # ---------------------------------------------------------
#     ax.spines["top"].set_visible(show_top_spine)
#     ax.spines["right"].set_visible(show_right_spine)
#     ax.spines["left"].set_visible(show_left_spine)
#     ax.spines["bottom"].set_visible(show_bottom_spine)

#     for name in ["top","right","left","bottom"]:
#         if ax.spines[name].get_visible():
#             ax.spines[name].set_linewidth(axis_linewidth)

#     for lbl in ax.get_xticklabels()+ax.get_yticklabels():
#         lbl.set_fontfamily(font_family)

#     # ---------------------------------------------------------
#     # Legend
#     # ---------------------------------------------------------
#     if show_legend:
#         legend_handles=[]

#         for method in methods:
#             edge=_component_color(
#                 box_edgecolor_mode,method,colors[method]
#             )

#             legend_handles.append(
#                 Patch(
#                     facecolor=to_rgba(colors[method],legend_box_alpha),
#                     edgecolor=edge,
#                     linewidth=box_linewidth,
#                     label=method_labels.get(method,method)
#                 )
#             )

#         if legend_ncol is None:
#             legend_ncol=int(
#                 np.ceil(len(methods)/max(1,legend_nrows))
#             )

#         legend=ax.legend(
#             handles=legend_handles,
#             loc=legend_loc,
#             bbox_to_anchor=legend_bbox,
#             ncol=legend_ncol,
#             frameon=legend_frame,
#             fontsize=legend_fontsize,
#             title=legend_title,
#             handlelength=legend_handlelength,
#             columnspacing=legend_columnspacing
#         )

#         if legend_title is not None:
#             legend.get_title().set_fontsize(legend_title_fontsize)
#             legend.get_title().set_fontfamily(font_family)

#         for text in legend.get_texts():
#             text.set_fontfamily(font_family)

#     # ---------------------------------------------------------
#     # Explicit margins if desired
#     # ---------------------------------------------------------
#     adjust={}
#     if left_margin is not None: adjust["left"]=left_margin
#     if right_margin is not None: adjust["right"]=right_margin
#     if top_margin is not None: adjust["top"]=top_margin
#     if bottom_margin is not None: adjust["bottom"]=bottom_margin

#     if adjust:
#         fig.subplots_adjust(**adjust)

#     # ---------------------------------------------------------
#     # Save / show
#     # ---------------------------------------------------------
#     if save_path is not None:
#         if save_pdf:
#             fig.savefig(
#                 f"{save_path}.pdf",
#                 format="pdf",
#                 dpi=dpi,
#                 bbox_inches=bbox_inches
#             )
#         if save_png:
#             fig.savefig(
#                 f"{save_path}.png",
#                 format="png",
#                 dpi=dpi,
#                 bbox_inches=bbox_inches
#             )

#     if show:
#         plt.show()

#     return summary_df,fig,ax,seed_df

import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from scipy.stats import t as student_t


def plot_seeded_task_metric_boxplots(
    seeded_allele_ap, task_orders, method_suffixes,
    metric="acquisition", retention_mode="final", base_score_col="mhcprime",
    method_labels=None, assay_labels=None, method_colors=None,

    # Figure / fonts
    figsize=(3.6,1.65), dpi=300, font_family="Nimbus Sans",
    title=None, xlabel=None, ylabel=None,
    title_fontsize=6, xlabel_fontsize=6, ylabel_fontsize=6,
    xtick_fontsize=5.5, ytick_fontsize=6,
    title_bold=False, title_pad=4,
    xlabel_pad=1, ylabel_pad=1.5,

    # Box geometry / styling
    #
    # box_width:
    #     Desired maximum width of each individual box.
    #
    # group_width:
    #     Preferred total width occupied by all methods within an assay.
    #
    # auto_expand_group:
    #     If the requested box widths require more horizontal room than
    #     group_width provides, automatically expand the cluster using the
    #     available categorical space instead of shrinking boxes prematurely.
    #
    # max_group_width:
    #     Absolute maximum fraction of the 1.0 assay-to-assay spacing that
    #     one complete method cluster may occupy.
    #
    # method_gap:
    #     Preferred absolute gap between adjacent boxes.
    box_width=0.12,
    group_width=0.78,
    auto_expand_group=True,
    max_group_width=0.88,
    method_gap=0.015,

    box_alpha=1.0,
    box_facecolor_mode="method",
    box_edgecolor_mode="black",
    box_linewidth=0.40,

    median_color="black",
    median_linewidth=0.60,
    median_linestyle="-",

    whisker_color_mode="black",
    whisker_linewidth=0.40,
    whisker_linestyle="-",

    cap_color_mode="black",
    cap_linewidth=0.40,

    show_caps=False,
    show_fliers=False,
    flier_size=2,
    flier_marker="o",
    flier_alpha=0.7,

    show_mean=False,
    mean_marker="D",
    mean_marker_size=3,
    mean_color="black",

    # Context points
    # Each may be scalar OR {method:value}.
    show_points=True,
    point_colors=None,
    point_edgecolors="none",
    point_edgewidths=0.0,
    point_alphas=0.75,
    point_sizes=10,
    point_size_scale=1.0,
    point_markers="o",
    point_jitter_fraction=0.20,
    point_zorder=3,

    # Axes
    ylim=None, xlim=None,
    axis_linewidth=0.35,
    show_top_spine=False, show_right_spine=False,
    show_left_spine=True, show_bottom_spine=True,
    tick_length=2.2, tick_width=0.35,
    tick_direction="out", tick_pad=2,
    xtick_rotation=0,

    # Grid
    grid=False, grid_axis="y", grid_alpha=0.0,
    grid_linewidth=0.35, grid_linestyle="-",

    # Optional horizontal reference
    show_reference_line=None,
    reference_y=0.0, reference_color="black",
    reference_linewidth=0.35, reference_linestyle="--",
    reference_zorder=0,

    # Legend
    show_legend=True, legend_nrows=1, legend_ncol=None,
    legend_loc="upper center", legend_bbox=(0.5,-0.18),
    legend_fontsize=6, legend_frame=False,
    legend_title=None, legend_title_fontsize=6,
    legend_box_alpha=1.0,
    legend_handlelength=0.8, legend_columnspacing=1.0,

    # Labels / layout
    show_n=True, n_format="{assay} (n={n})",
    left_margin=None, right_margin=None,
    top_margin=None, bottom_margin=None,

    # Output
    save_path=None, save_pdf=True, save_png=True,
    bbox_inches="tight", show=True,
):
    """
    Plot seed-averaged task-context distributions.

    metric
    ------
    "acquisition"
        Acquisition mAP immediately after the assay is learned.

    "acquisition_delta"
        Acquisition mAP - MHCPrime mAP for that assay.

    "retention"
        retention_mode="final":
            final mAP - acquisition mAP.

        retention_mode="all":
            every later checkpoint mAP - acquisition mAP.

    Each plotted point is one unique task-order/context after averaging
    seed/split replicates within that context.

    Box geometry
    ------------
    Assay centers are fixed one categorical unit apart.

    The requested `box_width` is treated as the desired maximum individual
    box width. The complete method cluster expands automatically, when space
    permits, so a small user-specified group_width cannot force the boxes into
    thin slivers.

    The cluster is still bounded by max_group_width, so boxes from adjacent
    assays can never overlap.

    Returns
    -------
    summary_df, fig, ax, seed_df
    """

    # ================================================================
    # Validation
    # ================================================================
    valid_metrics={"acquisition","acquisition_delta","retention"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {sorted(valid_metrics)}.")
    if retention_mode not in {"final","all"}:
        raise ValueError("retention_mode must be 'final' or 'all'.")
    if not task_orders:
        raise ValueError("task_orders cannot be empty.")
    if not method_suffixes:
        raise ValueError("method_suffixes cannot be empty.")

    if box_width<=0:
        raise ValueError("box_width must be > 0.")
    if group_width is not None and group_width<=0:
        raise ValueError("group_width must be > 0 or None.")
    if not 0<max_group_width<1:
        raise ValueError("max_group_width must be > 0 and < 1.")
    if method_gap<0:
        raise ValueError("method_gap must be >= 0.")
    if point_jitter_fraction<0:
        raise ValueError("point_jitter_fraction must be >= 0.")
    if point_size_scale<=0:
        raise ValueError("point_size_scale must be > 0.")

    methods=list(method_suffixes)
    method_labels={} if method_labels is None else dict(method_labels)
    assay_labels={} if assay_labels is None else dict(assay_labels)

    # ================================================================
    # Assays inferred from task orders
    # ================================================================
    assays=list(next(iter(task_orders.values())))
    expected_assays=set(assays)

    for order_name,order in task_orders.items():
        order=list(order)
        if len(order)!=len(assays) or set(order)!=expected_assays:
            raise ValueError(
                f"{order_name}: expected assay set {assays}, got {order}."
            )

    missing=[a for a in assays if a not in seeded_allele_ap]
    if missing:
        raise ValueError(f"Missing assay AP dataframes: {missing}")

    # ================================================================
    # Detect seeds
    # ================================================================
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for assay in assays:
        for col in seeded_allele_ap[assay].columns:
            m=seed_pattern.search(str(col))
            if m:
                seeds.append(m.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError(
            "No score columns ending in '_seed_<seed>' were found."
        )

    def _score_col(order_name,stage,method,seed):
        return f"{order_name}_s{stage}{method_suffixes[method]}_seed_{seed}"

    # ================================================================
    # Macro mAP over alleles
    #
    # UNCHANGED
    # ================================================================
    maps={}

    for assay in assays:
        df=seeded_allele_ap[assay]

        if base_score_col not in df.columns:
            raise ValueError(
                f"{assay}: missing base column {base_score_col!r}."
            )

        numeric=df.select_dtypes(include=[np.number])

        if numeric.shape[1]==0:
            raise ValueError(
                f"{assay}: no numeric AP columns found."
            )

        maps[assay]=numeric.mean(axis=0,skipna=True)

    # ================================================================
    # Seed-level values
    #
    # UNCHANGED
    # ================================================================
    seed_rows=[]

    for order_name,order in task_orders.items():
        order=list(order)
        n_stages=len(order)

        for assay in assays:
            acquisition_stage=order.index(assay)+1
            base_map=float(maps[assay][base_score_col])

            for method in methods:
                for seed in seeds:
                    acquisition_col=_score_col(
                        order_name,
                        acquisition_stage,
                        method,
                        seed,
                    )

                    if acquisition_col not in maps[assay].index:
                        raise ValueError(
                            f"{assay}: missing column {acquisition_col!r}."
                        )

                    acquisition_map=float(
                        maps[assay][acquisition_col]
                    )

                    if metric in {"acquisition","acquisition_delta"}:
                        value=(
                            acquisition_map
                            if metric=="acquisition"
                            else acquisition_map-base_map
                        )

                        seed_rows.append({
                            "method":method,
                            "assay":assay,
                            "order":order_name,
                            "seed":seed,
                            "acquisition_stage":acquisition_stage,
                            "evaluation_stage":acquisition_stage,
                            "updates_after_acquisition":0,
                            "base_map":base_map,
                            "acquisition_map":acquisition_map,
                            "evaluation_map":acquisition_map,
                            "value":value,
                        })
                        continue

                    # Retention
                    if acquisition_stage==n_stages:
                        continue

                    eval_stages=(
                        list(range(acquisition_stage+1,n_stages+1))
                        if retention_mode=="all"
                        else [n_stages]
                    )

                    for eval_stage in eval_stages:
                        eval_col=_score_col(
                            order_name,
                            eval_stage,
                            method,
                            seed,
                        )

                        if eval_col not in maps[assay].index:
                            raise ValueError(
                                f"{assay}: missing column {eval_col!r}."
                            )

                        evaluation_map=float(
                            maps[assay][eval_col]
                        )

                        seed_rows.append({
                            "method":method,
                            "assay":assay,
                            "order":order_name,
                            "seed":seed,
                            "acquisition_stage":acquisition_stage,
                            "evaluation_stage":eval_stage,
                            "updates_after_acquisition":
                                eval_stage-acquisition_stage,
                            "base_map":base_map,
                            "acquisition_map":acquisition_map,
                            "evaluation_map":evaluation_map,
                            "value":evaluation_map-acquisition_map,
                        })

    seed_df=pd.DataFrame(seed_rows)

    if seed_df.empty:
        raise ValueError(
            f"No valid observations generated for metric={metric!r}."
        )

    # ================================================================
    # Average seeds within each plotted context
    #
    # UNCHANGED
    # ================================================================
    group_cols=[
        "method",
        "assay",
        "order",
        "acquisition_stage",
        "evaluation_stage",
        "updates_after_acquisition",
    ]

    summary_rows=[]

    for keys,g in seed_df.groupby(group_cols,sort=False):
        vals=g["value"].dropna().to_numpy(float)
        n=len(vals)

        mean=vals.mean() if n else np.nan
        sd=vals.std(ddof=1) if n>1 else np.nan
        se=sd/np.sqrt(n) if n>1 else np.nan
        ci_half=(
            student_t.ppf(.975,n-1)*se
            if n>1
            else np.nan
        )

        row=dict(zip(group_cols,keys))

        row.update({
            "n_seeds":n,
            "mean_value":mean,
            "sd_across_seeds":sd,
            "se_across_seeds":se,
            "ci95_halfwidth":ci_half,
            "ci95_lower":
                mean-ci_half if n>1 else np.nan,
            "ci95_upper":
                mean+ci_half if n>1 else np.nan,
            "mean_base_map":
                g["base_map"].mean(),
            "mean_acquisition_map":
                g["acquisition_map"].mean(),
            "mean_evaluation_map":
                g["evaluation_map"].mean(),
        })

        summary_rows.append(row)

    summary_df=pd.DataFrame(summary_rows)

    # ================================================================
    # Context counts
    #
    # UNCHANGED
    # ================================================================
    context_counts=(
        summary_df
        .groupby(["method","assay"])
        .size()
        .unstack("method")
    )

    assay_n={}

    for assay in assays:
        if assay not in context_counts.index:
            assay_n[assay]=0
            continue

        counts=context_counts.loc[assay].dropna().astype(int)

        if len(counts)==0:
            assay_n[assay]=0

        elif counts.nunique()!=1:
            raise ValueError(
                f"{assay}: methods have different "
                f"context counts: {counts.to_dict()}"
            )

        else:
            assay_n[assay]=int(counts.iloc[0])

    # ================================================================
    # Matplotlib styling
    # ================================================================
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]

    default_colors=(
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    colors={
        method:(
            method_colors[method]
            if method_colors is not None
            and method in method_colors
            else default_colors[i%len(default_colors)]
        )
        for i,method in enumerate(methods)
    }

    def _method_setting(value,method,default=None):
        if isinstance(value,dict):
            return value.get(method,default)
        return value

    def _component_color(mode,method,default):
        if isinstance(mode,dict):
            return mode.get(method,default)
        if mode=="method":
            return colors[method]
        return mode

    point_face={
        method:(
            _method_setting(
                point_colors,
                method,
                colors[method],
            )
            if point_colors is not None
            else colors[method]
        )
        for method in methods
    }

    point_edge={
        method:_method_setting(
            point_edgecolors,
            method,
            "none",
        )
        for method in methods
    }

    point_lw={
        method:_method_setting(
            point_edgewidths,
            method,
            0.0,
        )
        for method in methods
    }

    point_alpha_final={
        method:_method_setting(
            point_alphas,
            method,
            1.0,
        )
        for method in methods
    }

    point_size_final={
        method:(
            float(
                _method_setting(
                    point_sizes,
                    method,
                    10,
                )
            )*point_size_scale
        )
        for method in methods
    }

    point_marker_final={
        method:_method_setting(
            point_markers,
            method,
            "o",
        )
        for method in methods
    }

    # ================================================================
    # SAFE, SPACE-USING CATEGORICAL GEOMETRY
    # ================================================================
    n_methods=len(methods)
    n_assays=len(assays)

    # Assay categories are always one unit apart.
    assay_spacing=1.0
    centers=np.arange(n_assays,dtype=float)*assay_spacing

    # The cluster may use almost the entire categorical cell, but never
    # more than max_group_width.
    max_safe_group_width=float(max_group_width)*assay_spacing

    desired_box_width=float(box_width)
    desired_gap=float(method_gap)

    # How much total width would be needed to preserve the user's desired
    # box width AND desired gap?
    desired_required_width=(
        desired_box_width
        if n_methods==1
        else (
            n_methods*desired_box_width
            +(n_methods-1)*desired_gap
        )
    )

    if group_width is None:
        requested_group_width=desired_required_width

    elif auto_expand_group:
        # A small group_width can never force the boxes to collapse.
        # Expand toward the room required by the desired box geometry.
        requested_group_width=max(
            float(group_width),
            desired_required_width,
        )

    else:
        requested_group_width=float(group_width)

    # Never invade the neighboring assay category.
    actual_group_width=min(
        requested_group_width,
        max_safe_group_width,
    )

    if n_methods==1:
        actual_box_width=min(
            desired_box_width,
            actual_group_width,
        )
        actual_gap=0.0
        offsets=np.array([0.0])

    else:
        # First preserve the requested inter-method gap whenever possible.
        available_for_boxes=(
            actual_group_width
            -(n_methods-1)*desired_gap
        )

        # If there is not even enough room for the requested gaps,
        # shrink gaps first instead of destroying the boxes.
        if available_for_boxes<=0:
            actual_gap=(
                actual_group_width
                /(n_methods*4.0)
            )
        else:
            actual_gap=desired_gap

        max_box_width=(
            actual_group_width
            -(n_methods-1)*actual_gap
        )/n_methods

        actual_box_width=min(
            desired_box_width,
            max_box_width,
        )

        # If box_width is smaller than the available slot, use the remaining
        # room as equal spacing so the method cluster occupies the requested
        # width cleanly instead of collapsing into the center.
        if actual_box_width<max_box_width:
            actual_gap=(
                actual_group_width
                -n_methods*actual_box_width
            )/(n_methods-1)

        occupied_width=(
            n_methods*actual_box_width
            +(n_methods-1)*actual_gap
        )

        first_center=(
            -occupied_width/2
            +actual_box_width/2
        )

        offsets=(
            first_center
            +np.arange(n_methods,dtype=float)
            *(actual_box_width+actual_gap)
        )

    # Keep points inside their own box footprint.
    safe_jitter_fraction=min(
        float(point_jitter_fraction),
        0.46,
    )

    # ================================================================
    # Figure
    # ================================================================
    fig,ax=plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    # ================================================================
    # Boxes
    # ================================================================
    for method_i,method in enumerate(methods):

        positions=centers+offsets[method_i]
        data=[]

        for assay in assays:
            vals=summary_df.loc[
                (summary_df["method"]==method)
                &(summary_df["assay"]==assay),
                "mean_value",
            ].dropna().to_numpy(float)

            data.append(vals)

        edge_color=_component_color(
            box_edgecolor_mode,
            method,
            colors[method],
        )

        whisker_color=_component_color(
            whisker_color_mode,
            method,
            colors[method],
        )

        cap_color=_component_color(
            cap_color_mode,
            method,
            colors[method],
        )

        bp=ax.boxplot(
            data,
            positions=positions,
            widths=actual_box_width,
            patch_artist=True,
            manage_ticks=False,
            showcaps=show_caps,
            showfliers=show_fliers,
            showmeans=show_mean,

            boxprops={
                "linewidth":box_linewidth,
                "edgecolor":edge_color,
            },

            medianprops={
                "color":median_color,
                "linewidth":median_linewidth,
                "linestyle":median_linestyle,
            },

            whiskerprops={
                "color":whisker_color,
                "linewidth":whisker_linewidth,
                "linestyle":whisker_linestyle,
            },

            capprops={
                "color":cap_color,
                "linewidth":cap_linewidth,
            },

            flierprops={
                "marker":flier_marker,
                "markersize":flier_size,
                "markerfacecolor":colors[method],
                "markeredgecolor":colors[method],
                "alpha":flier_alpha,
            },

            meanprops={
                "marker":mean_marker,
                "markersize":mean_marker_size,
                "markerfacecolor":mean_color,
                "markeredgecolor":mean_color,
            },
        )

        # ------------------------------------------------------------
        # Fill alpha only; edge remains opaque
        # ------------------------------------------------------------
        for box in bp["boxes"]:

            if box_facecolor_mode=="method":
                face=to_rgba(
                    colors[method],
                    box_alpha,
                )

            elif (
                box_facecolor_mode is None
                or box_facecolor_mode=="none"
            ):
                face="none"

            else:
                face=to_rgba(
                    box_facecolor_mode,
                    box_alpha,
                )

            box.set_facecolor(face)
            box.set_edgecolor(edge_color)
            box.set_linewidth(box_linewidth)
            box.set_alpha(None)

        # ------------------------------------------------------------
        # Seed-averaged context points
        # ------------------------------------------------------------
        if show_points:
            for assay_i,vals in enumerate(data):

                if len(vals)==0:
                    continue

                jitter_extent=(
                    actual_box_width
                    *safe_jitter_fraction
                )

                jitter=(
                    np.array([0.0])
                    if len(vals)==1
                    else np.linspace(
                        -jitter_extent,
                        jitter_extent,
                        len(vals),
                    )
                )

                ax.scatter(
                    positions[assay_i]+jitter,
                    vals,
                    s=point_size_final[method],
                    marker=point_marker_final[method],
                    facecolors=point_face[method],
                    edgecolors=point_edge[method],
                    linewidths=point_lw[method],
                    alpha=point_alpha_final[method],
                    zorder=point_zorder,
                )

    # ================================================================
    # Metric defaults
    # ================================================================
    if show_reference_line is None:
        show_reference_line=(
            metric in {
                "acquisition_delta",
                "retention",
            }
        )

    if title is None:
        if metric=="acquisition":
            title="Task-specific acquisition mAP"

        elif metric=="acquisition_delta":
            title="Task-specific acquisition ΔmAP vs MHCPrime"

        elif retention_mode=="all":
            title=(
                "Task-specific retention across "
                "post-acquisition checkpoints"
            )

        else:
            title="Task-specific final retention"

    if xlabel is None:
        xlabel="Assay"

    if ylabel is None:
        if metric=="acquisition":
            ylabel="Acquisition mAP"

        elif metric=="acquisition_delta":
            ylabel="Acquisition ΔmAP vs MHCPrime"

        else:
            ylabel="mAP - acquisition mAP"

    # ================================================================
    # Reference / grid
    # ================================================================
    if show_reference_line:
        ax.axhline(
            reference_y,
            color=reference_color,
            linewidth=reference_linewidth,
            linestyle=reference_linestyle,
            zorder=reference_zorder,
        )

    if grid:
        ax.grid(
            True,
            axis=grid_axis,
            alpha=grid_alpha,
            linewidth=grid_linewidth,
            linestyle=grid_linestyle,
            zorder=0,
        )

    # ================================================================
    # X labels
    # ================================================================
    ax.set_xticks(centers)

    tick_labels=[]

    for assay in assays:
        label=assay_labels.get(
            assay,
            assay,
        )

        if show_n:
            label=n_format.format(
                assay=label,
                n=assay_n[assay],
            )

        tick_labels.append(label)

    ax.set_xticklabels(
        tick_labels,
        rotation=xtick_rotation,
        fontsize=xtick_fontsize,
    )

    # Keep each assay centered in a fixed categorical cell.
    if xlim is None:
        ax.set_xlim(
            centers[0]-assay_spacing/2,
            centers[-1]+assay_spacing/2,
        )
    else:
        ax.set_xlim(xlim)

    # ================================================================
    # Axes / labels
    # ================================================================
    ax.tick_params(
        axis="both",
        length=tick_length,
        width=tick_width,
        direction=tick_direction,
        pad=tick_pad,
    )

    ax.tick_params(
        axis="y",
        labelsize=ytick_fontsize,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=xlabel_fontsize,
        fontfamily=font_family,
        labelpad=xlabel_pad,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=ylabel_fontsize,
        fontfamily=font_family,
        labelpad=ylabel_pad,
    )

    ax.set_title(
        title,
        fontsize=title_fontsize,
        fontfamily=font_family,
        fontweight=(
            "bold"
            if title_bold
            else "normal"
        ),
        pad=title_pad,
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    # ================================================================
    # Spines
    # ================================================================
    ax.spines["top"].set_visible(show_top_spine)
    ax.spines["right"].set_visible(show_right_spine)
    ax.spines["left"].set_visible(show_left_spine)
    ax.spines["bottom"].set_visible(show_bottom_spine)

    for name in ["top","right","left","bottom"]:
        if ax.spines[name].get_visible():
            ax.spines[name].set_linewidth(axis_linewidth)

    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontfamily(font_family)

    # ================================================================
    # Legend
    # ================================================================
    if show_legend:
        legend_handles=[]

        for method in methods:
            edge=_component_color(
                box_edgecolor_mode,
                method,
                colors[method],
            )

            legend_handles.append(
                Patch(
                    facecolor=to_rgba(
                        colors[method],
                        legend_box_alpha,
                    ),
                    edgecolor=edge,
                    linewidth=box_linewidth,
                    label=method_labels.get(
                        method,
                        method,
                    ),
                )
            )

        if legend_ncol is None:
            legend_ncol=int(
                np.ceil(
                    len(methods)
                    /max(1,legend_nrows)
                )
            )

        legend=ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=legend_ncol,
            frameon=legend_frame,
            fontsize=legend_fontsize,
            title=legend_title,
            handlelength=legend_handlelength,
            columnspacing=legend_columnspacing,
        )

        if legend_title is not None:
            legend.get_title().set_fontsize(
                legend_title_fontsize
            )
            legend.get_title().set_fontfamily(
                font_family
            )

        for text in legend.get_texts():
            text.set_fontfamily(font_family)

    # ================================================================
    # Explicit margins
    # ================================================================
    adjust={}

    if left_margin is not None:
        adjust["left"]=left_margin

    if right_margin is not None:
        adjust["right"]=right_margin

    if top_margin is not None:
        adjust["top"]=top_margin

    if bottom_margin is not None:
        adjust["bottom"]=bottom_margin

    if adjust:
        fig.subplots_adjust(**adjust)

    # ================================================================
    # Save / show
    # ================================================================
    if save_path is not None:

        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches=bbox_inches,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                dpi=dpi,
                bbox_inches=bbox_inches,
            )

    if show:
        plt.show()

    return summary_df,fig,ax,seed_df
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


def plot_seeded_cl_seen_performance_and_ms_retention(
    seeded_allele_ap,
    task_orders,
    method_suffixes,
    base_score_col="mhcprime",
    order_name=None,
    interval="ci95",
    interval_unit="orders",
    method_labels=None,
    method_colors=None,

    # Figure
    figsize=(8, 4),
    dpi=300,
    default_font="Nimbus Sans",

    # Panel titles
    seen_title="Seen-task average performance",
    ms_title="MS retention",
    title_fontsize=11,
    title_bold=False,
    title_pad=8,

    # Axis labels
    xlabel="Sequential update stage",
    seen_ylabel="mAP",
    ms_ylabel="ΔmAP from base MS",
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4,
    ylabel_pad=4,

    # Axis limits
    seen_ylim=None,
    ms_ylim=None,
    seen_xlim=None,
    ms_xlim=None,

    # Lines / markers
    linewidth=1.5,
    marker="o",
    markersize=5,
    marker_edgewidth=0,
    marker_edgecolor=None,

    # CI / interval bars
    error_colors=None,
    error_linewidth=1.0,
    error_capsize=3.0,
    error_capthick=1.0,
    error_alpha=1.0,
    error_zorder=3,

    # Zero reference line on MS-retention panel
    ms_zero_line=True,
    zero_line_color="black",
    zero_line_width=0.8,
    zero_line_style="--",
    zero_line_alpha=1.0,

    # Axes / ticks
    remove_top_spine=True,
    remove_right_spine=True,
    axis_linewidth=0.8,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction="out",
    xtick_labelsize=9,
    ytick_labelsize=9,
    xtick_rotation=0,

    # Grid
    grid_alpha=0.0,
    grid_linewidth=0.5,
    grid_linestyle="-",

    # Legend
    show_legend=True,
    legend_panel=0,
    legend_loc="best",
    legend_bbox=None,
    legend_ncol=1,
    legend_fontsize=9,
    legend_frame=False,
    legend_handlelength=2.0,

    # Panel spacing
    wspace=0.3,

    # Saving / display
    save_path=None,
    show=True,
):
    """
    Plot two continual-learning trajectories:

        Left:
            Seen-task average performance, with MS excluded from the
            continual-learning task sequence.

        Right:
            MS retention = current MS mAP - baseline MS mAP.

    Aggregation
    -----------
    interval_unit="orders":
        Compute each metric per seed x task order, average seeds within
        each task order, then summarize across task orders.

    interval_unit="seeds":
        Compute each metric per seed x task order, average task orders
        within each seed, then summarize across seeds.

    interval:
        "ci95"  -> 95% Student-t CI
        "sd"    -> mean +/- 1 sample SD
        "se"    -> mean +/- 1 SE
        "range" -> observed min/max
        None    -> no interval

    Parameters
    ----------
    seeded_allele_ap : dict[str, pd.DataFrame]
        Dictionary containing one allele-level AP dataframe per assay plus
        an "ms" dataframe.

    task_orders : dict[str, list[str]]
        Mapping from task-order name to sequential assay ordering.

    method_suffixes : dict[str, str]
        Mapping from method key to the suffix used in score-column names.

    method_labels : dict, optional
        Display labels for methods.

    method_colors : dict, optional
        Line/marker colors by method.

    error_colors : dict or str, optional
        Error-bar colors.

        None:
            Each method's error bars use its line color.

        dict:
            {"FT": "...", "PepCL": "..."}

        str:
            Same error-bar color for every method.

    save_path : str, optional
        Path + filename WITHOUT extension.
        Saves:
            <save_path>.pdf
            <save_path>.png

    Returns
    -------
    dict
        {
            "seed_order_metrics": ...,
            "seed_plot_metrics": ...,
            "order_plot_metrics": ...,
            "plot_unit_metrics": ...,
            "summaries": ...,
            "figure": fig,
            "axes": axes,
            "seeds": seeds,
        }
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if "ms" not in seeded_allele_ap:
        raise ValueError("seeded_allele_ap must contain an 'ms' test set.")

    if order_name is not None and order_name not in task_orders:
        raise ValueError(f"Unknown order_name: {order_name}")

    if interval not in {"ci95", "sd", "se", "range", None}:
        raise ValueError("interval must be 'ci95', 'sd', 'se', 'range', or None.")

    if interval_unit not in {"orders", "seeds"}:
        raise ValueError("interval_unit must be 'orders' or 'seeds'.")

    if order_name is not None and interval_unit=="orders":
        raise ValueError(
            "interval_unit='orders' requires multiple task orders. "
            "For a single order_name, use interval_unit='seeds'."
        )

    methods=list(method_suffixes)
    method_labels=method_labels or {m:m for m in methods}
    orders={order_name:task_orders[order_name]} if order_name is not None else task_orders

    n_stages=len(next(iter(task_orders.values())))
    assay_tasks=list(next(iter(task_orders.values())))

    if any(len(v)!=n_stages for v in task_orders.values()):
        raise ValueError("All task orders must have the same number of stages.")

    if set(assay_tasks)!=(set(seeded_allele_ap)-{"ms"}):
        missing=set(assay_tasks)-set(seeded_allele_ap)
        if missing:
            raise ValueError(f"Missing assay AP dataframes: {missing}")

    # ------------------------------------------------------------------
    # Infer seed IDs
    # ------------------------------------------------------------------
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            m=seed_pattern.search(str(col))
            if m:
                seeds.append(m.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError("No `_seed_<seed>` score columns were found.")

    def _score_col(order,stage,method,seed):
        return f"{order}_s{stage}{method_suffixes[method]}_seed_{seed}"

    # ------------------------------------------------------------------
    # Validate score columns
    # ------------------------------------------------------------------
    required=[base_score_col]

    for order in task_orders:
        for method in methods:
            for stage in range(1,n_stages+1):
                for seed in seeds:
                    required.append(_score_col(order,stage,method,seed))

    for dataset,df in seeded_allele_ap.items():
        missing=[c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{dataset}: missing {len(missing)} required AP columns. "
                f"First: {missing[:5]}"
            )

    # Macro-mAP over alleles
    maps={
        dataset:df.mean(axis=0,skipna=True)
        for dataset,df in seeded_allele_ap.items()
    }

    # ------------------------------------------------------------------
    # Compute metrics for each seed x order
    # MS is NOT considered a CL task here.
    # ------------------------------------------------------------------
    rows=[]

    for order,task_order in orders.items():
        for method in methods:
            for seed in seeds:

                # stage x [MS + assay tasks]
                perf=pd.DataFrame(
                    index=range(n_stages+1),
                    columns=["ms"]+list(task_order),
                    dtype=float,
                )

                # Baseline performance
                for task in perf.columns:
                    perf.loc[0,task]=maps[task][base_score_col]

                # Sequential checkpoints
                for stage in range(1,n_stages+1):
                    score_col=_score_col(order,stage,method,seed)

                    for task in perf.columns:
                        perf.loc[stage,task]=maps[task][score_col]

                # Metrics
                for stage in range(n_stages+1):

                    if stage==0:
                        # Same convention as original function:
                        # baseline performance of first assay to be encountered.
                        seen_mean=perf.loc[0,task_order[0]]
                    else:
                        seen=list(task_order[:stage])
                        seen_mean=perf.loc[stage,seen].mean()

                    ms_retention=perf.loc[stage,"ms"]-perf.loc[0,"ms"]

                    rows.append({
                        "order":order,
                        "method":method,
                        "seed":seed,
                        "stage":stage,
                        "seen_task_mean":seen_mean,
                        "ms_retention":ms_retention,
                    })

    seed_order_metrics=pd.DataFrame(rows)

    metric_cols=["seen_task_mean","ms_retention"]

    # ------------------------------------------------------------------
    # Both aggregation hierarchies
    # ------------------------------------------------------------------

    # Average task orders within each seed
    seed_plot_metrics=(
        seed_order_metrics
        .groupby(["method","seed","stage"],as_index=False)[metric_cols]
        .mean()
    )

    # Average seeds within each task order
    order_plot_metrics=(
        seed_order_metrics
        .groupby(["method","order","stage"],as_index=False)[metric_cols]
        .mean()
    )

    plot_unit_metrics=(
        order_plot_metrics
        if interval_unit=="orders"
        else seed_plot_metrics
    )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    summary_rows=[]

    for (method,stage),g in plot_unit_metrics.groupby(
        ["method","stage"],sort=False
    ):
        for metric in metric_cols:
            vals=g[metric].dropna().to_numpy(float)
            n=len(vals)

            if n==0:
                mean=sd=se=lower=upper=np.nan

            else:
                mean=vals.mean()
                sd=vals.std(ddof=1) if n>1 else np.nan
                se=sd/np.sqrt(n) if n>1 else np.nan

                if interval is None:
                    lower=upper=np.nan

                elif interval=="sd":
                    lower,upper=mean-sd,mean+sd

                elif interval=="se":
                    lower,upper=mean-se,mean+se

                elif interval=="ci95":
                    half=student_t.ppf(0.975,n-1)*se if n>1 else np.nan
                    lower,upper=mean-half,mean+half

                elif interval=="range":
                    lower,upper=vals.min(),vals.max()

            summary_rows.append({
                "method":method,
                "stage":stage,
                "metric":metric,
                "interval_unit":interval_unit,
                "n_units":n,
                "mean":mean,
                "sd":sd,
                "se":se,
                "lower":lower,
                "upper":upper,
            })

    summaries=pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        default_font,"Nimbus Sans","Nimbus Sans L","Arial","Helvetica"
    ]
    matplotlib.rcParams["pdf.fonttype"]=42

    fig,axes=plt.subplots(
        1,2,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )
    axes=axes.ravel()

    panel_specs=[
        ("seen_task_mean",seen_title,seen_ylabel,seen_ylim,seen_xlim),
        ("ms_retention",ms_title,ms_ylabel,ms_ylim,ms_xlim),
    ]

    for panel_idx,(ax,(metric,panel_title,ylabel,ylim,xlim)) in enumerate(
        zip(axes,panel_specs)
    ):
        for method in methods:
            q=(
                summaries[
                    (summaries["method"]==method)&
                    (summaries["metric"]==metric)
                ]
                .sort_values("stage")
            )

            color=method_colors.get(method) if method_colors else None

            x=q["stage"].to_numpy(float)
            y=q["mean"].to_numpy(float)

            line=ax.plot(
                x,y,
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                markeredgewidth=marker_edgewidth,
                markeredgecolor=marker_edgecolor,
                label=method_labels.get(method,method),
                color=color,
                zorder=2,
            )[0]

            # Error bars
            if interval is not None:
                low=q["lower"].to_numpy(float)
                high=q["upper"].to_numpy(float)

                valid=(
                    np.isfinite(x)&
                    np.isfinite(y)&
                    np.isfinite(low)&
                    np.isfinite(high)
                )

                if valid.any():
                    lower_err=y[valid]-low[valid]
                    upper_err=high[valid]-y[valid]

                    if error_colors is None:
                        error_color=line.get_color()
                    elif isinstance(error_colors,dict):
                        error_color=error_colors.get(method,line.get_color())
                    else:
                        error_color=error_colors

                    ax.errorbar(
                        x[valid],
                        y[valid],
                        yerr=np.vstack([lower_err,upper_err]),
                        fmt="none",
                        ecolor=error_color,
                        elinewidth=error_linewidth,
                        capsize=error_capsize,
                        capthick=error_capthick,
                        alpha=error_alpha,
                        zorder=error_zorder,
                    )

        # MS zero-retention reference
        if metric=="ms_retention" and ms_zero_line:
            ax.axhline(
                0,
                color=zero_line_color,
                linewidth=zero_line_width,
                linestyle=zero_line_style,
                alpha=zero_line_alpha,
                zorder=1,
            )

        # Stages
        ticks=np.arange(n_stages+1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            ["Base"]+[f"S{i}" for i in range(1,n_stages+1)],
            rotation=xtick_rotation,
        )

        ax.set_xlabel(
            xlabel,
            fontsize=xlabel_fontsize,
            fontfamily=default_font,
            labelpad=xlabel_pad,
        )
        ax.set_ylabel(
            ylabel,
            fontsize=ylabel_fontsize,
            fontfamily=default_font,
            labelpad=ylabel_pad,
        )

        ax.set_title(
            panel_title,
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight="bold" if title_bold else "normal",
            pad=title_pad,
        )

        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

        if remove_top_spine:
            ax.spines["top"].set_visible(False)
        else:
            ax.spines["top"].set_linewidth(axis_linewidth)

        if remove_right_spine:
            ax.spines["right"].set_visible(False)
        else:
            ax.spines["right"].set_linewidth(axis_linewidth)

        ax.spines["left"].set_linewidth(axis_linewidth)
        ax.spines["bottom"].set_linewidth(axis_linewidth)

        ax.tick_params(
            axis="both",
            which="major",
            length=tick_length,
            width=tick_width,
            direction=tick_direction,
            pad=tick_length+2,
        )
        ax.tick_params(axis="x",labelsize=xtick_labelsize)
        ax.tick_params(axis="y",labelsize=ytick_labelsize)

        for lbl in ax.get_xticklabels()+ax.get_yticklabels():
            lbl.set_fontfamily(default_font)

        if grid_alpha>0:
            ax.grid(
                True,
                axis="y",
                alpha=grid_alpha,
                linewidth=grid_linewidth,
                linestyle=grid_linestyle,
                zorder=0,
            )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    if show_legend:
        legend_ax=axes[legend_panel]
        handles,labels=legend_ax.get_legend_handles_labels()

        legend_kwargs=dict(
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=legend_frame,
            fontsize=legend_fontsize,
            handlelength=legend_handlelength,
        )

        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"]=legend_bbox

        legend=legend_ax.legend(handles,labels,**legend_kwargs)

        for txt in legend.get_texts():
            txt.set_fontfamily(default_font)

    # No figure-level/suptitle.
    fig.subplots_adjust(wspace=wspace)
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_path:
        fig.savefig(
            f"{save_path}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300,
        )
        fig.savefig(
            f"{save_path}.png",
            format="png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()

    return {
        "seed_order_metrics":seed_order_metrics,
        "seed_plot_metrics":seed_plot_metrics,
        "order_plot_metrics":order_plot_metrics,
        "plot_unit_metrics":plot_unit_metrics,
        "summaries":summaries,
        "figure":fig,
        "axes":axes,
        "seeds":seeds,
    }
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


def plot_seeded_cl_average_metrics_clean(
    seeded_allele_ap,
    task_orders,
    method_suffixes,
    base_score_col="mhcprime",
    order_name=None,
    interval="ci95",
    interval_unit="orders",
    method_labels=None,
    method_colors=None,

    # CL task definition.
    include_ms_as_task=False,

    # Panel ordering: exactly 5 metrics.
    panel_order=None,

    # Figure.
    figsize=(12,7),
    dpi=300,
    default_font="Nimbus Sans",

    # Panel titles / labels.
    panel_titles=None,
    panel_ylabels=None,
    title_fontsize=11,
    title_bold=False,
    title_pad=8,

    # Axis labels.
    xlabel="Sequential update stage",
    xlabel_fontsize=10,
    ylabel_fontsize=10,
    xlabel_pad=4,
    ylabel_pad=4,

    # Per-panel axis limits.
    ylims=None,
    xlims=None,

    # Lines / markers.
    linewidth=1.5,
    marker="o",
    markersize=5,
    marker_edgewidth=0,
    marker_edgecolor=None,

    # CI / interval bars.
    error_colors=None,
    error_linewidth=1.0,
    error_capsize=3.0,
    error_capthick=1.0,
    error_alpha=1.0,
    error_zorder=3,

    # Zero reference lines.
    zero_lines=True,
    zero_line_color="black",
    zero_line_width=0.8,
    zero_line_style="--",
    zero_line_alpha=1.0,

    # Axes / ticks.
    remove_top_spine=True,
    remove_right_spine=True,
    axis_linewidth=0.8,
    tick_length=4.0,
    tick_width=0.8,
    tick_direction="out",
    xtick_labelsize=9,
    ytick_labelsize=9,
    xtick_rotation=0,

    # Grid.
    grid_alpha=0.0,
    grid_linewidth=0.5,
    grid_linestyle="-",

    # Legend.
    show_legend=True,
    legend_loc="center",
    legend_bbox=None,
    legend_ncol=1,
    legend_fontsize=9,
    legend_frame=False,
    legend_handlelength=2.0,

    # Panel spacing.
    wspace=0.30,
    hspace=0.35,
    left_margin=None,
    right_margin=None,
    bottom_margin=None,
    top_margin=None,

    # Saving / display.
    save_path=None,
    show=True,
):
    """
    Plot 5 continual learning metrics in a fixed 2 x 3 grid.

    Layout
    ------
        panel 1 | panel 2 | panel 3
        panel 4 | panel 5 | legend

    `panel_order` determines which metric occupies each of the 5 plot cells.

    Metrics
    -------
    seen_task_mean
        Mean performance across tasks encountered through the current stage.

    all_domain_mean
        Mean performance across all assay domains at the current stage.

    backward_transfer
        Mean current performance minus acquisition-time performance across
        previously encountered tasks.

    maximum_forgetting
        Maximum historical-best minus current performance among previously
        encountered tasks.

    ms_retention
        Current MS mAP minus baseline MS mAP.

    CL task definition
    ------------------
    include_ms_as_task=False:
        MS is excluded from seen-task average, all-domain average, BWT, and
        maximum forgetting. MS is evaluated only through ms_retention.

    include_ms_as_task=True:
        MS is treated as the initial continual learning task for those 4 CL
        metrics. ms_retention remains separately defined in exactly the same way.

    Aggregation
    -----------
    interval_unit="orders":
        Compute each metric independently for every seed x task-order trajectory,
        average seeds within each task order, then summarize across task orders.

    interval_unit="seeds":
        Average task orders within each seed, then summarize across seeds.

    interval:
        "ci95"  -> 95% Student-t CI
        "sd"    -> mean +/- 1 sample SD
        "se"    -> mean +/- 1 SE
        "range" -> observed min/max
        None    -> no interval

    Returns
    -------
    dict containing:
        seed_order_metrics
        seed_plot_metrics
        order_plot_metrics
        plot_unit_metrics
        summaries
        figure
        axes
        plot_axes
        legend_ax
        seeds
        mode
    """

    # ================================================================
    # 0. Validation / setup
    # ================================================================
    if "ms" not in seeded_allele_ap:
        raise ValueError("seeded_allele_ap must contain an 'ms' test set.")

    if order_name is not None and order_name not in task_orders:
        raise ValueError(f"Unknown order_name: {order_name}")

    if interval not in {"ci95","sd","se","range",None}:
        raise ValueError(
            "interval must be 'ci95', 'sd', 'se', 'range', or None."
        )

    if interval_unit not in {"orders","seeds"}:
        raise ValueError(
            "interval_unit must be 'orders' or 'seeds'."
        )

    if order_name is not None and interval_unit=="orders":
        raise ValueError(
            "interval_unit='orders' requires multiple task orders. "
            "For a single order_name, use interval_unit='seeds'."
        )

    methods=list(method_suffixes)

    method_labels=(
        {m:m for m in methods}
        if method_labels is None
        else dict(method_labels)
    )

    orders=(
        {order_name:task_orders[order_name]}
        if order_name is not None
        else task_orders
    )

    n_stages=len(next(iter(task_orders.values())))
    assay_tasks=list(next(iter(task_orders.values())))

    if any(len(v)!=n_stages for v in task_orders.values()):
        raise ValueError(
            "All task orders must have the same number of stages."
        )

    if set(assay_tasks)!=(set(seeded_allele_ap)-{"ms"}):
        missing=set(assay_tasks)-set(seeded_allele_ap)

        if missing:
            raise ValueError(
                f"Missing assay AP dataframes: {missing}"
            )

    valid_metrics=[
        "seen_task_mean",
        "all_domain_mean",
        "backward_transfer",
        "maximum_forgetting",
        "ms_retention",
    ]

    if panel_order is None:
        panel_order=[
            "seen_task_mean",
            "all_domain_mean",
            "backward_transfer",
            "maximum_forgetting",
            "ms_retention",
        ]
    else:
        panel_order=list(panel_order)

    if len(panel_order)!=5:
        raise ValueError(
            "panel_order must contain exactly 5 metrics."
        )

    if set(panel_order)!=set(valid_metrics):
        raise ValueError(
            "panel_order must contain each of these exactly once: "
            f"{valid_metrics}"
        )

    default_titles={
        "seen_task_mean":"Seen-task average performance",
        "all_domain_mean":"All-domain mean performance",
        "backward_transfer":"Backward transfer",
        "maximum_forgetting":"Maximum forgetting",
        "ms_retention":"MS retention",
    }

    default_ylabels={
        "seen_task_mean":"mAP",
        "all_domain_mean":"mAP",
        "backward_transfer":"BWT",
        "maximum_forgetting":"Forgetting",
        "ms_retention":"ΔmAP from base MS",
    }

    if panel_titles is not None:
        default_titles.update(panel_titles)

    if panel_ylabels is not None:
        default_ylabels.update(panel_ylabels)

    panel_titles=default_titles
    panel_ylabels=default_ylabels
    ylims={} if ylims is None else dict(ylims)
    xlims={} if xlims is None else dict(xlims)

    # ================================================================
    # 1. Infer seed IDs
    # ================================================================
    seed_pattern=re.compile(r"_seed_(.+)$")
    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            m=seed_pattern.search(str(col))

            if m:
                seeds.append(m.group(1))

    seeds=list(dict.fromkeys(seeds))

    if not seeds:
        raise ValueError(
            "No `_seed_<seed>` score columns were found."
        )

    def _score_col(order,stage,method,seed):
        return (
            f"{order}_s{stage}"
            f"{method_suffixes[method]}"
            f"_seed_{seed}"
        )

    # ================================================================
    # 2. Validate required score columns
    # ================================================================
    required=[base_score_col]

    for order in task_orders:
        for method in methods:
            for stage in range(1,n_stages+1):
                for seed in seeds:
                    required.append(
                        _score_col(
                            order,
                            stage,
                            method,
                            seed,
                        )
                    )

    for dataset,df in seeded_allele_ap.items():
        missing=[
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset}: missing {len(missing)} required AP columns. "
                f"First: {missing[:5]}"
            )

    # Macro-mAP over alleles.
    maps={
        dataset:df.mean(axis=0,skipna=True)
        for dataset,df in seeded_allele_ap.items()
    }

    # ================================================================
    # 3. Compute complete seed x order trajectories
    # ================================================================
    rows=[]

    mode=(
        "ms_as_task"
        if include_ms_as_task
        else "assay_tasks_only"
    )

    for order,task_order in orders.items():

        acquisition={
            "ms":0,
            **{
                task:i
                for i,task in enumerate(
                    task_order,
                    start=1,
                )
            },
        }

        for method in methods:
            for seed in seeds:

                perf=pd.DataFrame(
                    index=range(n_stages+1),
                    columns=["ms"]+list(task_order),
                    dtype=float,
                )

                # Baseline.
                for task in perf.columns:
                    perf.loc[0,task]=maps[task][base_score_col]

                # Sequential checkpoints.
                for stage in range(1,n_stages+1):
                    col=_score_col(
                        order,
                        stage,
                        method,
                        seed,
                    )

                    for task in perf.columns:
                        perf.loc[stage,task]=maps[task][col]

                # ----------------------------------------------------
                # Metrics at each sequential stage.
                # ----------------------------------------------------
                for stage in range(n_stages+1):

                    if include_ms_as_task:
                        full_order=["ms"]+list(task_order)

                        seen=full_order[:stage+1]
                        previous=full_order[:stage]

                        seen_mean=perf.loc[
                            stage,
                            seen,
                        ].mean()

                        all_domain_mean=perf.loc[
                            stage,
                            full_order,
                        ].mean()

                    else:
                        if stage==0:
                            # Preserve convention from original function:
                            # baseline performance on the first assay that
                            # will subsequently be encountered.
                            seen_mean=perf.loc[
                                0,
                                task_order[0],
                            ]

                            previous=[]

                        else:
                            seen=list(
                                task_order[:stage]
                            )

                            previous=list(
                                task_order[:stage-1]
                            )

                            seen_mean=perf.loc[
                                stage,
                                seen,
                            ].mean()

                        all_domain_mean=perf.loc[
                            stage,
                            list(task_order),
                        ].mean()

                    bwt_vals=[]
                    forgetting_vals=[]

                    for task in previous:
                        acquired=acquisition[task]

                        current=perf.loc[
                            stage,
                            task,
                        ]

                        acquired_perf=perf.loc[
                            acquired,
                            task,
                        ]

                        prior_best=perf.loc[
                            acquired:stage-1,
                            task,
                        ].max()

                        bwt_vals.append(
                            current-acquired_perf
                        )

                        forgetting_vals.append(
                            prior_best-current
                        )

                    rows.append({
                        "mode":mode,
                        "order":order,
                        "method":method,
                        "seed":seed,
                        "stage":stage,
                        "seen_task_mean":seen_mean,
                        "all_domain_mean":all_domain_mean,
                        "backward_transfer":(
                            np.mean(bwt_vals)
                            if bwt_vals else np.nan
                        ),
                        "maximum_forgetting":(
                            np.max(forgetting_vals)
                            if forgetting_vals else np.nan
                        ),
                        "ms_retention":(
                            perf.loc[stage,"ms"]
                            -perf.loc[0,"ms"]
                        ),
                    })

    seed_order_metrics=pd.DataFrame(rows)

    metric_cols=[
        "seen_task_mean",
        "all_domain_mean",
        "backward_transfer",
        "maximum_forgetting",
        "ms_retention",
    ]

    # ================================================================
    # 4. Both aggregation hierarchies
    # ================================================================

    # Average task orders within each seed.
    seed_plot_metrics=(
        seed_order_metrics
        .groupby(
            ["mode","method","seed","stage"],
            as_index=False,
        )[metric_cols]
        .mean()
    )

    # Average seeds within each task order.
    order_plot_metrics=(
        seed_order_metrics
        .groupby(
            ["mode","method","order","stage"],
            as_index=False,
        )[metric_cols]
        .mean()
    )

    plot_unit_metrics=(
        order_plot_metrics
        if interval_unit=="orders"
        else seed_plot_metrics
    )

    # ================================================================
    # 5. Summary statistics
    # ================================================================
    summary_rows=[]

    for (method,stage),g in plot_unit_metrics.groupby(
        ["method","stage"],
        sort=False,
    ):
        for metric in metric_cols:
            vals=g[metric].dropna().to_numpy(float)
            n=len(vals)

            if n==0:
                mean=sd=se=lower=upper=np.nan

            else:
                mean=vals.mean()
                sd=vals.std(ddof=1) if n>1 else np.nan
                se=sd/np.sqrt(n) if n>1 else np.nan

                if interval is None:
                    lower=upper=np.nan

                elif interval=="sd":
                    lower,upper=mean-sd,mean+sd

                elif interval=="se":
                    lower,upper=mean-se,mean+se

                elif interval=="ci95":
                    half=(
                        student_t.ppf(
                            0.975,
                            n-1,
                        )*se
                        if n>1 else np.nan
                    )

                    lower,upper=mean-half,mean+half

                elif interval=="range":
                    lower,upper=vals.min(),vals.max()

            summary_rows.append({
                "mode":mode,
                "method":method,
                "stage":stage,
                "metric":metric,
                "interval_unit":interval_unit,
                "n_units":n,
                "mean":mean,
                "sd":sd,
                "se":se,
                "lower":lower,
                "upper":upper,
            })

    summaries=pd.DataFrame(summary_rows)

    # ================================================================
    # 6. Plot styling
    # ================================================================
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        default_font,
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
    ]
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"

    fig,axes=plt.subplots(
        2,
        3,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    flat_axes=axes.ravel()
    plot_axes=flat_axes[:5]
    legend_ax=flat_axes[5]

    # ================================================================
    # 7. Plot the 5 panels
    # ================================================================
    for ax,metric in zip(
        plot_axes,
        panel_order,
    ):

        for method in methods:
            q=(
                summaries[
                    (summaries["method"]==method)
                    &(summaries["metric"]==metric)
                ]
                .sort_values("stage")
            )

            color=(
                method_colors.get(method)
                if method_colors is not None
                else None
            )

            x=q["stage"].to_numpy(float)
            y=q["mean"].to_numpy(float)

            line=ax.plot(
                x,
                y,
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                markeredgewidth=marker_edgewidth,
                markeredgecolor=marker_edgecolor,
                label=method_labels.get(
                    method,
                    method,
                ),
                color=color,
                zorder=2,
            )[0]

            # --------------------------------------------------------
            # Error bars.
            # --------------------------------------------------------
            if interval is not None:
                low=q["lower"].to_numpy(float)
                high=q["upper"].to_numpy(float)

                valid=(
                    np.isfinite(x)
                    &np.isfinite(y)
                    &np.isfinite(low)
                    &np.isfinite(high)
                )

                if valid.any():
                    lower_err=y[valid]-low[valid]
                    upper_err=high[valid]-y[valid]

                    if error_colors is None:
                        error_color=line.get_color()

                    elif isinstance(error_colors,dict):
                        error_color=error_colors.get(
                            method,
                            line.get_color(),
                        )

                    else:
                        error_color=error_colors

                    ax.errorbar(
                        x[valid],
                        y[valid],
                        yerr=np.vstack([
                            lower_err,
                            upper_err,
                        ]),
                        fmt="none",
                        ecolor=error_color,
                        elinewidth=error_linewidth,
                        capsize=error_capsize,
                        capthick=error_capthick,
                        alpha=error_alpha,
                        zorder=error_zorder,
                    )

        # ------------------------------------------------------------
        # Zero reference.
        # ------------------------------------------------------------
        if (
            zero_lines
            and metric in {
                "backward_transfer",
                "maximum_forgetting",
                "ms_retention",
            }
        ):
            ax.axhline(
                0,
                color=zero_line_color,
                linewidth=zero_line_width,
                linestyle=zero_line_style,
                alpha=zero_line_alpha,
                zorder=1,
            )

        # ------------------------------------------------------------
        # Stage ticks.
        #
        # With MS excluded, BWT / forgetting are undefined until stage 2.
        # ------------------------------------------------------------
        if (
            not include_ms_as_task
            and metric in {
                "backward_transfer",
                "maximum_forgetting",
            }
        ):
            ticks=np.arange(
                2,
                n_stages+1,
            )

            ax.set_xticks(ticks)

            ax.set_xticklabels(
                [f"S{i}" for i in ticks],
                rotation=xtick_rotation,
            )

        else:
            ticks=np.arange(
                n_stages+1
            )

            ax.set_xticks(ticks)

            ax.set_xticklabels(
                ["Base"]+[
                    f"S{i}"
                    for i in range(
                        1,
                        n_stages+1,
                    )
                ],
                rotation=xtick_rotation,
            )

        # ------------------------------------------------------------
        # Labels / title.
        # ------------------------------------------------------------
        ax.set_xlabel(
            xlabel,
            fontsize=xlabel_fontsize,
            fontfamily=default_font,
            labelpad=xlabel_pad,
        )

        ax.set_ylabel(
            panel_ylabels[metric],
            fontsize=ylabel_fontsize,
            fontfamily=default_font,
            labelpad=ylabel_pad,
        )

        ax.set_title(
            panel_titles[metric],
            fontsize=title_fontsize,
            fontfamily=default_font,
            fontweight=(
                "bold"
                if title_bold
                else "normal"
            ),
            pad=title_pad,
        )

        # ------------------------------------------------------------
        # Limits.
        # ------------------------------------------------------------
        if metric in ylims and ylims[metric] is not None:
            ax.set_ylim(
                ylims[metric]
            )

        if metric in xlims and xlims[metric] is not None:
            ax.set_xlim(
                xlims[metric]
            )

        # ------------------------------------------------------------
        # Spines.
        # ------------------------------------------------------------
        if remove_top_spine:
            ax.spines["top"].set_visible(False)
        else:
            ax.spines["top"].set_linewidth(
                axis_linewidth
            )

        if remove_right_spine:
            ax.spines["right"].set_visible(False)
        else:
            ax.spines["right"].set_linewidth(
                axis_linewidth
            )

        ax.spines["left"].set_linewidth(
            axis_linewidth
        )

        ax.spines["bottom"].set_linewidth(
            axis_linewidth
        )

        # ------------------------------------------------------------
        # Ticks.
        # ------------------------------------------------------------
        ax.tick_params(
            axis="both",
            which="major",
            length=tick_length,
            width=tick_width,
            direction=tick_direction,
            pad=tick_length+2,
        )

        ax.tick_params(
            axis="x",
            labelsize=xtick_labelsize,
        )

        ax.tick_params(
            axis="y",
            labelsize=ytick_labelsize,
        )

        for lbl in (
            ax.get_xticklabels()
            +ax.get_yticklabels()
        ):
            lbl.set_fontfamily(
                default_font
            )

        # ------------------------------------------------------------
        # Optional horizontal grid.
        # ------------------------------------------------------------
        if grid_alpha>0:
            ax.grid(
                True,
                axis="y",
                alpha=grid_alpha,
                linewidth=grid_linewidth,
                linestyle=grid_linestyle,
                zorder=0,
            )

    # ================================================================
    # 8. Legend occupies bottom-right grid cell
    # ================================================================
    legend_ax.axis("off")

    if show_legend:
        handles,labels=plot_axes[0].get_legend_handles_labels()

        legend_kwargs=dict(
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=legend_frame,
            fontsize=legend_fontsize,
            handlelength=legend_handlelength,
        )

        if legend_bbox is not None:
            legend_kwargs[
                "bbox_to_anchor"
            ]=legend_bbox

        legend=legend_ax.legend(
            handles,
            labels,
            **legend_kwargs,
        )

        for txt in legend.get_texts():
            txt.set_fontfamily(
                default_font
            )

    # ================================================================
    # 9. Layout
    # ================================================================
    adjust_kwargs={
        "wspace":wspace,
        "hspace":hspace,
    }

    if left_margin is not None:
        adjust_kwargs["left"]=left_margin

    if right_margin is not None:
        adjust_kwargs["right"]=right_margin

    if bottom_margin is not None:
        adjust_kwargs["bottom"]=bottom_margin

    if top_margin is not None:
        adjust_kwargs["top"]=top_margin

    fig.subplots_adjust(
        **adjust_kwargs
    )

    # ================================================================
    # 10. Save
    # ================================================================
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

    return {
        "seed_order_metrics":seed_order_metrics,
        "seed_plot_metrics":seed_plot_metrics,
        "order_plot_metrics":order_plot_metrics,
        "plot_unit_metrics":plot_unit_metrics,
        "summaries":summaries,
        "figure":fig,
        "axes":axes,
        "plot_axes":plot_axes,
        "legend_ax":legend_ax,
        "seeds":seeds,
        "mode":mode,
    }
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


def plot_seeded_cl_metrics_per_order_grid(
    seeded_allele_ap,
    task_orders,
    method_suffixes,
    base_score_col="mhcprime",

    # Which orders / metrics to plot.
    order_names=None,
    metric_order=None,

    # Display labels.
    method_labels=None,
    method_colors=None,
    dataset_labels=None,
    order_labels=None,

    # Intervals across seeds within each order.
    interval="ci95",  # {"ci95","sd","se","range",None}

    # Figure.
    figsize=None,
    dpi=300,
    font_family="Nimbus Sans",

    # Titles.
    panel_titles=None,
    title_fontsize=8,
    title_bold=False,
    title_pad=6,
    show_panel_titles=True,

    # Order labels on outer left.
    show_order_labels=True,
    order_label_fontsize=7,
    order_label_pad=0.035,
    order_label_rotation=90,
    order_label_wrap_width=28,

    # Axis labels.
    panel_ylabels=None,
    xlabel="Sequential update stage",
    xlabel_fontsize=7,
    ylabel_fontsize=7,
    xlabel_pad=3,
    ylabel_pad=3,
    ylabel_mode="top_row",   # {"top_row","all","none"}
    xlabel_mode="bottom_row",# {"bottom_row","all","none"}

    # Ticks.
    xtick_fontsize=6.5,
    ytick_fontsize=6.5,
    xtick_rotation=35,
    xtick_ha=None,
    tick_length=2.5,
    tick_width=0.5,
    tick_direction="out",
    tick_pad=2,

    # Lines / markers.
    linewidth=1.2,
    marker="o",
    markersize=3.5,
    marker_edgewidth=0,
    marker_edgecolor=None,
    line_alpha=1.0,

    # Error bars.
    error_colors=None,
    error_linewidth=0.8,
    error_capsize=2.5,
    error_capthick=0.8,
    error_alpha=1.0,
    error_zorder=2,

    # Zero reference lines.
    zero_lines=True,
    zero_line_color="black",
    zero_line_width=0.6,
    zero_line_style="--",
    zero_line_alpha=1.0,

    # Per-metric axis limits.
    ylims=None,
    xlims=None,

    # Spines.
    show_top_spine=True,
    show_right_spine=True,
    axis_linewidth=0.6,

    # Grid.
    grid=False,
    grid_axis="y",
    grid_alpha=0.15,
    grid_linewidth=0.4,
    grid_linestyle="-",

    # Legend.
    show_legend=True,
    legend_loc="upper center",
    legend_bbox=(0.5,1.01),
    legend_ncol=None,
    legend_fontsize=6.5,
    legend_frame=False,
    legend_handlelength=1.6,
    legend_columnspacing=1.0,

    # Grid spacing / margins.
    wspace=0.28,
    hspace=0.34,
    left_margin=0.09,
    right_margin=0.99,
    bottom_margin=0.08,
    top_margin=0.93,

    # Saving.
    save_path=None,
    save_pdf=True,
    save_png=True,
    bbox_inches="tight",
    show=True,
):
    """
    Plot continual learning metrics separately for each task order.

    Grid
    ----
    Rows:
        individual task orders

    Columns:
        metrics in `metric_order`

    Metrics
    -------
    seen_task_mean
        Mean current performance across non-MS assay tasks encountered
        through the current stage.

    all_domain_mean
        Mean current performance across all non-MS assay domains.

    backward_transfer
        Mean current minus acquisition-time performance across previously
        encountered non-MS assay tasks.

    maximum_forgetting
        Maximum historical-best minus current performance among previously
        encountered non-MS assay tasks.

    ms_retention
        Current pan-allelic MS mAP minus pretrained MHCPrime MS mAP.

    Statistical aggregation
    -----------------------
    Metrics are first calculated independently for every seed x task-order
    trajectory. Within each task order, plotted points are means across seeds,
    with intervals calculated across those same seed-level values.

    Returns
    -------
    dict with:
        seed_order_metrics
        summaries
        figure
        axes
        seeds
        orders
        metrics
    """

    # ================================================================
    # 0. Validation / setup
    # ================================================================
    if "ms" not in seeded_allele_ap:
        raise ValueError(
            "seeded_allele_ap must contain an 'ms' dataframe."
        )

    if interval not in {"ci95","sd","se","range",None}:
        raise ValueError(
            "interval must be 'ci95', 'sd', 'se', 'range', or None."
        )

    if ylabel_mode not in {"top_row","all","none"}:
        raise ValueError(
            "ylabel_mode must be 'top_row', 'all', or 'none'."
        )

    if xlabel_mode not in {"bottom_row","all","none"}:
        raise ValueError(
            "xlabel_mode must be 'bottom_row', 'all', or 'none'."
        )

    methods=list(method_suffixes)

    order_names=(
        list(task_orders)
        if order_names is None
        else list(order_names)
    )

    for order_name in order_names:
        if order_name not in task_orders:
            raise ValueError(
                f"Unknown task order: {order_name!r}"
            )

    valid_metrics=[
        "seen_task_mean",
        "all_domain_mean",
        "backward_transfer",
        "maximum_forgetting",
        "ms_retention",
    ]

    metric_order=(
        [
            "seen_task_mean",
            "backward_transfer",
            "maximum_forgetting",
            "ms_retention",
        ]
        if metric_order is None
        else list(metric_order)
    )

    invalid=[
        m for m in metric_order
        if m not in valid_metrics
    ]

    if invalid:
        raise ValueError(
            f"Unknown metrics: {invalid}. "
            f"Valid metrics: {valid_metrics}"
        )

    if len(set(metric_order))!=len(metric_order):
        raise ValueError(
            "metric_order contains duplicate metrics."
        )

    method_labels=(
        {m:m for m in methods}
        if method_labels is None
        else dict(method_labels)
    )

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)
    order_labels={} if order_labels is None else dict(order_labels)

    default_titles={
        "seen_task_mean":"Average performance",
        "all_domain_mean":"All domain mean performance",
        "backward_transfer":"Backward transfer",
        "maximum_forgetting":"Maximum forgetting",
        "ms_retention":"MS retention",
    }

    default_ylabels={
        "seen_task_mean":"mAP",
        "all_domain_mean":"mAP",
        "backward_transfer":"BWT (ΔmAP)",
        "maximum_forgetting":"Forgetting (mAP)",
        "ms_retention":"ΔmAP from base MS",
    }

    if panel_titles is not None:
        default_titles.update(
            panel_titles
        )

    if panel_ylabels is not None:
        default_ylabels.update(
            panel_ylabels
        )

    panel_titles=default_titles
    panel_ylabels=default_ylabels

    ylims={} if ylims is None else dict(ylims)
    xlims={} if xlims is None else dict(xlims)

    n_stages=len(
        next(iter(task_orders.values()))
    )

    if any(
        len(task_orders[o])!=n_stages
        for o in order_names
    ):
        raise ValueError(
            "All selected task orders must have the same number of stages."
        )

    assay_tasks=list(
        next(iter(task_orders.values()))
    )

    missing=set(assay_tasks)-set(seeded_allele_ap)

    if missing:
        raise ValueError(
            f"Missing assay AP dataframes: {missing}"
        )

    # ================================================================
    # 1. Infer seed IDs
    # ================================================================
    seed_pattern=re.compile(
        r"_seed_(.+)$"
    )

    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            m=seed_pattern.search(
                str(col)
            )

            if m:
                seeds.append(
                    m.group(1)
                )

    seeds=list(
        dict.fromkeys(seeds)
    )

    if not seeds:
        raise ValueError(
            "No '_seed_<seed>' score columns were found."
        )

    def _score_col(
        order,
        stage,
        method,
        seed,
    ):
        return (
            f"{order}_s{stage}"
            f"{method_suffixes[method]}"
            f"_seed_{seed}"
        )

    # ================================================================
    # 2. Validate required score columns
    # ================================================================
    required=[
        base_score_col
    ]

    for order in order_names:
        for method in methods:
            for stage in range(
                1,
                n_stages+1,
            ):
                for seed in seeds:
                    required.append(
                        _score_col(
                            order,
                            stage,
                            method,
                            seed,
                        )
                    )

    for dataset,df in seeded_allele_ap.items():
        missing=[
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset}: missing {len(missing)} required "
                f"score columns. First: {missing[:5]}"
            )

    # ================================================================
    # 3. Macro-mAP over alleles
    # ================================================================
    maps={
        dataset:df.mean(
            axis=0,
            skipna=True,
        )
        for dataset,df in seeded_allele_ap.items()
    }

    # ================================================================
    # 4. Compute seed x order trajectories
    #
    # Same metric definitions as the averaged version.
    # MS is NOT included as a continual learning task.
    # ================================================================
    rows=[]

    for order_name in order_names:
        task_order=list(
            task_orders[order_name]
        )

        acquisition={
            task:i
            for i,task in enumerate(
                task_order,
                start=1,
            )
        }

        for method in methods:
            for seed in seeds:

                perf=pd.DataFrame(
                    index=range(
                        n_stages+1
                    ),
                    columns=["ms"]+task_order,
                    dtype=float,
                )

                # Baseline.
                for task in perf.columns:
                    perf.loc[
                        0,
                        task,
                    ]=maps[task][
                        base_score_col
                    ]

                # Sequential checkpoints.
                for stage in range(
                    1,
                    n_stages+1,
                ):
                    col=_score_col(
                        order_name,
                        stage,
                        method,
                        seed,
                    )

                    for task in perf.columns:
                        perf.loc[
                            stage,
                            task,
                        ]=maps[task][col]

                # -----------------------------------------------
                # Stage metrics.
                # -----------------------------------------------
                for stage in range(
                    n_stages+1
                ):

                    # Seen-task average.
                    if stage==0:
                        seen_mean=perf.loc[
                            0,
                            task_order[0],
                        ]

                    else:
                        seen_tasks=list(
                            task_order[:stage]
                        )

                        seen_mean=perf.loc[
                            stage,
                            seen_tasks,
                        ].mean()

                    # All assay domains, excluding MS.
                    all_domain_mean=perf.loc[
                        stage,
                        task_order,
                    ].mean()

                    # Previous assay tasks only.
                    #
                    # At S1 there is no previously encountered assay,
                    # so BWT / maximum forgetting are undefined.
                    previous_tasks=(
                        []
                        if stage<2
                        else list(
                            task_order[:stage-1]
                        )
                    )

                    bwt_vals=[]
                    forgetting_vals=[]

                    for task in previous_tasks:
                        acquired=acquisition[
                            task
                        ]

                        current=perf.loc[
                            stage,
                            task,
                        ]

                        acquired_perf=perf.loc[
                            acquired,
                            task,
                        ]

                        prior_best=perf.loc[
                            acquired:stage-1,
                            task,
                        ].max()

                        bwt_vals.append(
                            current-acquired_perf
                        )

                        forgetting_vals.append(
                            prior_best-current
                        )

                    rows.append({
                        "order":order_name,
                        "method":method,
                        "seed":seed,
                        "stage":stage,

                        "seen_task_mean":
                            seen_mean,

                        "all_domain_mean":
                            all_domain_mean,

                        "backward_transfer":
                            (
                                np.mean(
                                    bwt_vals
                                )
                                if bwt_vals
                                else np.nan
                            ),

                        "maximum_forgetting":
                            (
                                np.max(
                                    forgetting_vals
                                )
                                if forgetting_vals
                                else np.nan
                            ),

                        "ms_retention":
                            (
                                perf.loc[
                                    stage,
                                    "ms",
                                ]
                                -perf.loc[
                                    0,
                                    "ms",
                                ]
                            ),
                    })

    seed_order_metrics=pd.DataFrame(
        rows
    )

    # ================================================================
    # 5. Summarize ACROSS SEEDS within each exact task order
    # ================================================================
    summary_rows=[]

    for (
        order_name,
        method,
        stage,
    ),g in seed_order_metrics.groupby(
        [
            "order",
            "method",
            "stage",
        ],
        sort=False,
    ):

        for metric in valid_metrics:
            vals=(
                g[metric]
                .dropna()
                .to_numpy(float)
            )

            n=len(vals)

            if n==0:
                mean=sd=se=lower=upper=np.nan

            else:
                mean=vals.mean()

                sd=(
                    vals.std(
                        ddof=1
                    )
                    if n>1
                    else np.nan
                )

                se=(
                    sd/np.sqrt(n)
                    if n>1
                    else np.nan
                )

                if interval is None:
                    lower=upper=np.nan

                elif interval=="sd":
                    lower=mean-sd
                    upper=mean+sd

                elif interval=="se":
                    lower=mean-se
                    upper=mean+se

                elif interval=="ci95":
                    half=(
                        student_t.ppf(
                            0.975,
                            n-1,
                        )*se
                        if n>1
                        else np.nan
                    )

                    lower=mean-half
                    upper=mean+half

                elif interval=="range":
                    lower=vals.min()
                    upper=vals.max()

            summary_rows.append({
                "order":order_name,
                "method":method,
                "stage":stage,
                "metric":metric,
                "n_seeds":n,
                "mean":mean,
                "sd":sd,
                "se":se,
                "lower":lower,
                "upper":upper,
            })

    summaries=pd.DataFrame(
        summary_rows
    )

    # ================================================================
    # 6. Plot styling
    # ================================================================
    matplotlib.rcParams[
        "font.family"
    ]="sans-serif"

    matplotlib.rcParams[
        "font.sans-serif"
    ]=[
        font_family,
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]

    matplotlib.rcParams[
        "pdf.fonttype"
    ]=42

    matplotlib.rcParams[
        "ps.fonttype"
    ]=42

    matplotlib.rcParams[
        "svg.fonttype"
    ]="none"

    default_colors=(
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    colors={
        method:(
            method_colors[method]
            if (
                method_colors is not None
                and method in method_colors
            )
            else default_colors[
                i%len(default_colors)
            ]
        )
        for i,method in enumerate(
            methods
        )
    }

    # ================================================================
    # 7. Figure / axes
    # ================================================================
    nrows=len(
        order_names
    )

    ncols=len(
        metric_order
    )

    if figsize is None:
        figsize=(
            2.55*ncols,
            1.8*nrows,
        )

    fig,axes=plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=wspace,
        hspace=hspace,
    )

    # ================================================================
    # 8. Plot every order x metric panel
    # ================================================================
    for row_i,order_name in enumerate(
        order_names
    ):
        task_order=list(
            task_orders[
                order_name
            ]
        )

        # Actual dataset names at each sequential stage.
        stage_labels=[
            "Base"
        ]+[
            dataset_labels.get(
                task,
                task,
            )
            for task in task_order
        ]

        for col_i,metric in enumerate(
            metric_order
        ):
            ax=axes[
                row_i,
                col_i,
            ]

            # -----------------------------------------------
            # Methods.
            # -----------------------------------------------
            for method in methods:

                q=(
                    summaries[
                        (
                            summaries["order"]
                            ==order_name
                        )
                        &(
                            summaries["method"]
                            ==method
                        )
                        &(
                            summaries["metric"]
                            ==metric
                        )
                    ]
                    .sort_values(
                        "stage"
                    )
                )

                color=colors[
                    method
                ]

                x=q[
                    "stage"
                ].to_numpy(float)

                y=q[
                    "mean"
                ].to_numpy(float)

                valid_line=(
                    np.isfinite(x)
                    &np.isfinite(y)
                )

                ax.plot(
                    x[valid_line],
                    y[valid_line],
                    color=color,
                    marker=marker,
                    linewidth=linewidth,
                    markersize=markersize,
                    markeredgewidth=
                        marker_edgewidth,
                    markeredgecolor=
                        marker_edgecolor,
                    alpha=line_alpha,
                    label=method_labels.get(
                        method,
                        method,
                    ),
                    zorder=3,
                )

                # -------------------------------------------
                # Interval bars across seeds.
                # -------------------------------------------
                if interval is not None:

                    low=q[
                        "lower"
                    ].to_numpy(float)

                    high=q[
                        "upper"
                    ].to_numpy(float)

                    valid=(
                        np.isfinite(x)
                        &np.isfinite(y)
                        &np.isfinite(low)
                        &np.isfinite(high)
                    )

                    if valid.any():

                        err_color=(
                            error_colors.get(
                                method,
                                color,
                            )
                            if isinstance(
                                error_colors,
                                dict,
                            )
                            else (
                                color
                                if error_colors is None
                                else error_colors
                            )
                        )

                        ax.errorbar(
                            x[valid],
                            y[valid],
                            yerr=np.vstack([
                                y[valid]
                                -low[valid],

                                high[valid]
                                -y[valid],
                            ]),
                            fmt="none",
                            ecolor=err_color,
                            elinewidth=
                                error_linewidth,
                            capsize=
                                error_capsize,
                            capthick=
                                error_capthick,
                            alpha=
                                error_alpha,
                            zorder=
                                error_zorder,
                        )

            # -----------------------------------------------
            # Zero lines.
            # -----------------------------------------------
            if (
                zero_lines
                and metric in {
                    "backward_transfer",
                    "maximum_forgetting",
                    "ms_retention",
                }
            ):
                ax.axhline(
                    0,
                    color=zero_line_color,
                    linewidth=
                        zero_line_width,
                    linestyle=
                        zero_line_style,
                    alpha=
                        zero_line_alpha,
                    zorder=1,
                )

            # -----------------------------------------------
            # X positions / labels.
            #
            # BWT and forgetting only exist from S2 onward.
            # -----------------------------------------------
            if metric in {
                "backward_transfer",
                "maximum_forgetting",
            }:
                tick_stages=np.arange(
                    2,
                    n_stages+1,
                )

                tick_labels=[
                    stage_labels[
                        int(stage)
                    ]
                    for stage in tick_stages
                ]

            else:
                tick_stages=np.arange(
                    n_stages+1
                )

                tick_labels=[
                    stage_labels[
                        int(stage)
                    ]
                    for stage in tick_stages
                ]

            ax.set_xticks(
                tick_stages
            )

            if xtick_ha is None:
                ha=(
                    "center"
                    if xtick_rotation==0
                    else "right"
                )
            else:
                ha=xtick_ha

            ax.set_xticklabels(
                tick_labels,
                rotation=
                    xtick_rotation,
                ha=ha,
                fontsize=
                    xtick_fontsize,
            )

            # -----------------------------------------------
            # Column titles: top row only.
            # -----------------------------------------------
            if (
                show_panel_titles
                and row_i==0
            ):
                ax.set_title(
                    panel_titles[
                        metric
                    ],
                    fontsize=
                        title_fontsize,
                    fontweight=(
                        "bold"
                        if title_bold
                        else "normal"
                    ),
                    pad=title_pad,
                )

            # -----------------------------------------------
            # Metric-specific y-axis labels.
            #
            # Default: first row only, so each column gets its
            # proper metric label once without repetition.
            # -----------------------------------------------
            show_ylabel=(
                ylabel_mode=="all"
                or (
                    ylabel_mode=="top_row"
                    and row_i==0
                )
            )

            if show_ylabel:
                ax.set_ylabel(
                    panel_ylabels[
                        metric
                    ],
                    fontsize=
                        ylabel_fontsize,
                    labelpad=
                        ylabel_pad,
                )
            else:
                ax.set_ylabel("")

            # -----------------------------------------------
            # X-axis label.
            # -----------------------------------------------
            show_xlabel=(
                xlabel_mode=="all"
                or (
                    xlabel_mode=="bottom_row"
                    and row_i==nrows-1
                )
            )

            if show_xlabel:
                ax.set_xlabel(
                    xlabel,
                    fontsize=
                        xlabel_fontsize,
                    labelpad=
                        xlabel_pad,
                )
            else:
                ax.set_xlabel("")

            # -----------------------------------------------
            # Limits.
            # -----------------------------------------------
            if (
                metric in ylims
                and ylims[
                    metric
                ] is not None
            ):
                ax.set_ylim(
                    ylims[
                        metric
                    ]
                )

            if (
                metric in xlims
                and xlims[
                    metric
                ] is not None
            ):
                ax.set_xlim(
                    xlims[
                        metric
                    ]
                )

            # -----------------------------------------------
            # Spines.
            # -----------------------------------------------
            ax.spines[
                "top"
            ].set_visible(
                show_top_spine
            )

            ax.spines[
                "right"
            ].set_visible(
                show_right_spine
            )

            for spine_name in [
                "left",
                "bottom",
                "top",
                "right",
            ]:
                if ax.spines[
                    spine_name
                ].get_visible():

                    ax.spines[
                        spine_name
                    ].set_linewidth(
                        axis_linewidth
                    )

            # -----------------------------------------------
            # Tick styling.
            # -----------------------------------------------
            ax.tick_params(
                axis="both",
                which="major",
                length=tick_length,
                width=tick_width,
                direction=tick_direction,
                pad=tick_pad,
            )

            ax.tick_params(
                axis="y",
                labelsize=
                    ytick_fontsize,
            )

            # -----------------------------------------------
            # Grid.
            # -----------------------------------------------
            if grid:
                ax.grid(
                    True,
                    axis=grid_axis,
                    alpha=grid_alpha,
                    linewidth=
                        grid_linewidth,
                    linestyle=
                        grid_linestyle,
                    zorder=0,
                )

    # ================================================================
    # 9. Outer-left order labels
    # ================================================================
    if show_order_labels:

        for row_i,order_name in enumerate(
            order_names
        ):

            # Use supplied label if present; otherwise build
            # an arrow-separated dataset order.
            if order_name in order_labels:
                label=order_labels[
                    order_name
                ]

            else:
                task_order=list(
                    task_orders[
                        order_name
                    ]
                )

                label=" → ".join([
                    dataset_labels.get(
                        task,
                        task,
                    )
                    for task in task_order
                ])

            if order_label_wrap_width:
                # Only useful for non-rotated labels;
                # harmless otherwise.
                words=str(label).split()

                lines=[]
                current=""

                for word in words:
                    candidate=(
                        word
                        if not current
                        else f"{current} {word}"
                    )

                    if (
                        len(candidate)
                        <=order_label_wrap_width
                    ):
                        current=candidate

                    else:
                        if current:
                            lines.append(
                                current
                            )

                        current=word

                if current:
                    lines.append(
                        current
                    )

                label="\n".join(
                    lines
                )

            pos=axes[
                row_i,
                0,
            ].get_position()

            fig.text(
                left_margin
                -order_label_pad,
                (
                    pos.y0
                    +pos.y1
                )/2,
                label,
                fontsize=
                    order_label_fontsize,
                rotation=
                    order_label_rotation,
                ha="center",
                va="center",
                fontfamily=
                    font_family,
            )

    # ================================================================
    # 10. Legend
    # ================================================================
    if show_legend:

        handles,labels=axes[
            0,
            0,
        ].get_legend_handles_labels()

        if legend_ncol is None:
            legend_ncol=len(
                methods
            )

        legend=fig.legend(
            handles,
            labels,
            loc=legend_loc,
            bbox_to_anchor=
                legend_bbox,
            ncol=legend_ncol,
            frameon=
                legend_frame,
            fontsize=
                legend_fontsize,
            handlelength=
                legend_handlelength,
            columnspacing=
                legend_columnspacing,
        )

        for txt in legend.get_texts():
            txt.set_fontfamily(
                font_family
            )

    # ================================================================
    # 11. Save / show
    # ================================================================
    if save_path is not None:

        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches=
                    bbox_inches,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                dpi=dpi,
                bbox_inches=
                    bbox_inches,
            )

    if show:
        plt.show()

    return {
        "seed_order_metrics":
            seed_order_metrics,
        "summaries":
            summaries,
        "figure":
            fig,
        "axes":
            axes,
        "seeds":
            seeds,
        "orders":
            order_names,
        "metrics":
            metric_order,
    }
import re
import textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_seeded_cl_heatmap(
    seeded_allele_ap,
    task_orders,
    method_suffixes,
    base_score_col="mhcprime",

    # What to plot.
    plot_type="complete",  # {"complete","order_average"}
    method_order=None,
    order_names=None,

    # Display names.
    method_labels=None,
    dataset_labels=None,
    order_short_labels=None,

    # Change metric.
    change_mode="delta",   # {"delta","percent"}
    change_vmin=None,
    change_vmax=None,

    # Colormaps.
    complete_cmap="viridis",
    complete_colors=None,
    raw_cmap="viridis",
    raw_colors=None,
    change_cmap="coolwarm_r",
    change_colors=None,

    # Annotation.
    annotate=True,
    decimals=2,
    annotation_fontsize=5,
    annotation_text_color="auto",
    annotation_dark_color="black",
    annotation_light_color="white",
    annotation_luminance_threshold=0.45,

    # Figure.
    figsize=None,
    dpi=300,
    font_family="Nimbus Sans",

    # Optional overall title.
    show_top_title=False,
    top_title=None,
    top_title_fontsize=12,
    top_title_y=0.98,

    # Column/order titles.
    column_title_fontsize=8,
    column_title_pad=6,
    summary_title_wrap_width=24,

    # Method labels.
    method_wrap_width=24,
    method_label_fontsize=7,
    method_label_pad=18,
    method_label_rotation=90,

    # Tick labels.
    xtick_fontsize=6,
    ytick_fontsize=6,
    x_tick_rotation=0,

    # Heatmap geometry.
    square_heatmaps=False,

    # Grid spacing.
    complete_wspace=0.42,
    complete_hspace=0.70,
    average_wspace=0.55,
    average_hspace=0.75,

    # Figure margins.
    left_margin=None,
    right_margin=None,
    bottom_margin=None,
    top_margin=None,

    # Heatmap styling.
    spine_linewidth=0.5,

    # Colorbar styling.
    colorbar_fontsize=7,
    colorbar_label_fontsize=8,

    # Output.
    save_path=None,
    save_pdf=True,
    save_png=True,
    bbox_inches="tight",
    show=True,
):
    """
    Plot one of two continual learning heatmap summaries, both including MS.

    plot_type="complete"
        Complete order-specific continual learning performance matrices.
        Rows = methods.
        Columns = task orders.
        Each heatmap contains:
            rows = Base + sequential checkpoints
            cols = MS + assay domains in that task order

    plot_type="order_average"
        Order-averaged positional continual learning matrices.
        Rows = methods.
        Columns =
            raw mean performance
            change from base
            change from previous checkpoint
            change from prior historical maximum

        Heatmap positions are aligned as:
            rows = Base, S1, S2, ...
            cols = MS, S1, S2, ...

    Mathematical behavior is preserved from the original function:
        - dataset performance is macro-mAP across alleles
        - nonlinear change calculations are performed independently
          within each seed first
        - seed aggregation occurs after those calculations
        - order averaging is performed on the resulting order-specific
          matrices

    Heatmap geometry
    ----------------
    square_heatmaps=False
        Use the subplot geometry normally.

    square_heatmaps=True
        Force each individual heatmap axes to have a physical 1:1
        height-to-width ratio. In a wider subplot cell, the x dimension
        contracts to match the existing y dimension.

    Heatmap rendering
    -----------------
    Heatmap cells are drawn with non-rasterized pcolormesh rather than
    imshow. This keeps each cell as a vector-colored quadrilateral in PDF
    output and improves compatibility when importing into Illustrator.

    Annotation colors
    -----------------
    annotation_text_color="auto"
        Annotation text switches automatically between
        annotation_dark_color and annotation_light_color according to
        the luminance of the actual rendered cell color.

    annotation_text_color="<color>"
        Use one fixed color for all annotations.

    Custom colormaps
    ----------------
    complete_colors, raw_colors, and change_colors may be lists such as:

        ["#440154","#21918c","#fde725"]

    If provided, the color list takes precedence over the corresponding
    named cmap argument and is converted into a continuous gradient.

    Returns
    -------
    dict with:
        figure
        axes
        seeds
        raw_full
        change_from_base_full
        change_from_previous_full
        change_from_historical_max_full
        order_average
    """

    # ================================================================
    # Validation / setup
    # ================================================================
    if plot_type not in {"complete","order_average"}:
        raise ValueError(
            "plot_type must be 'complete' or 'order_average'."
        )

    if change_mode not in {"delta","percent"}:
        raise ValueError(
            "change_mode must be 'delta' or 'percent'."
        )

    if (
        change_vmin is not None
        and change_vmax is not None
        and change_vmin>=change_vmax
    ):
        raise ValueError(
            "change_vmin must be smaller than change_vmax."
        )

    if not task_orders:
        raise ValueError("task_orders cannot be empty.")

    if not method_suffixes:
        raise ValueError("method_suffixes cannot be empty.")

    if not 0<=annotation_luminance_threshold<=1:
        raise ValueError(
            "annotation_luminance_threshold must be between 0 and 1."
        )

    method_order=(
        list(method_suffixes)
        if method_order is None
        else list(method_order)
    )

    order_names=(
        list(task_orders)
        if order_names is None
        else list(order_names)
    )

    method_labels={} if method_labels is None else dict(method_labels)
    dataset_labels={} if dataset_labels is None else dict(dataset_labels)
    order_short_labels={} if order_short_labels is None else dict(order_short_labels)

    for method in method_order:
        if method not in method_suffixes:
            raise ValueError(
                f"Method {method!r} is not in method_suffixes."
            )

    for order_name in order_names:
        if order_name not in task_orders:
            raise ValueError(
                f"Task order {order_name!r} is not in task_orders."
            )

    n_stages=len(next(iter(task_orders.values())))

    if any(
        len(task_orders[o])!=n_stages
        for o in order_names
    ):
        raise ValueError(
            "All task orders must contain the same number of stages."
        )

    # ================================================================
    # Matplotlib setup
    # ================================================================
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]

    # ================================================================
    # Colormap helper
    # ================================================================
    def _resolve_cmap(cmap,colors,name):
        if colors is None:
            return cmap

        if len(colors)<2:
            raise ValueError(
                f"{name} must contain at least 2 colors."
            )

        return LinearSegmentedColormap.from_list(
            name,
            list(colors),
            N=256,
        )

    complete_cmap_obj=_resolve_cmap(
        complete_cmap,
        complete_colors,
        "complete_custom",
    )

    raw_cmap_obj=_resolve_cmap(
        raw_cmap,
        raw_colors,
        "raw_custom",
    )

    change_cmap_obj=_resolve_cmap(
        change_cmap,
        change_colors,
        "change_custom",
    )

    # ================================================================
    # Label helpers
    # ================================================================
    def _display_label(x):
        s=str(x)

        if s.lower()=="ms":
            return "MS"

        if s.lower()=="base":
            return "Base"

        m=re.fullmatch(
            r"[sS](\d+)",
            s,
        )

        if m:
            return f"S{m.group(1)}"

        return dataset_labels.get(
            x,
            dataset_labels.get(
                s,
                s,
            ),
        )

    def _method_label(method):
        display=method_labels.get(
            method,
            method,
        )

        return "\n".join(
            textwrap.wrap(
                str(display),
                width=method_wrap_width,
            )
        )

    def _order_title(order_name):
        order=list(
            task_orders[order_name]
        )

        labels=[]

        for dataset in order:
            label=order_short_labels.get(
                dataset,
                dataset_labels.get(
                    dataset,
                    str(dataset),
                ),
            )

            labels.append(
                str(label)
            )

        return " → ".join(labels)

    # ================================================================
    # Detect seed IDs
    # ================================================================
    seed_pattern=re.compile(
        r"_seed_(.+)$"
    )

    seeds=[]

    for df in seeded_allele_ap.values():
        for col in df.columns:
            m=seed_pattern.search(
                str(col)
            )

            if m:
                seeds.append(
                    m.group(1)
                )

    seeds=list(
        dict.fromkeys(seeds)
    )

    if not seeds:
        raise ValueError(
            "No columns ending in '_seed_<seed>' were detected."
        )

    def _score_col(
        order_name,
        stage,
        method,
        seed,
    ):
        return (
            f"{order_name}_s{stage}"
            f"{method_suffixes[method]}"
            f"_seed_{seed}"
        )

    # ================================================================
    # Change helper
    # ================================================================
    def _change(current,reference):
        current=np.asarray(
            current,
            dtype=float,
        )

        reference=np.asarray(
            reference,
            dtype=float,
        )

        if change_mode=="delta":
            return current-reference

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            out=(
                100.0
                *(current-reference)
                /reference
            )

        out[
            ~np.isfinite(out)
        ]=np.nan

        return out

    # ================================================================
    # Dataset macro-mAP across allele rows
    # ================================================================
    maps={}

    for name,df in seeded_allele_ap.items():
        numeric=df.select_dtypes(
            include=[np.number]
        )

        if base_score_col not in numeric.columns:
            raise ValueError(
                f"{name}: base score column "
                f"{base_score_col!r} is missing or nonnumeric."
            )

        maps[name]=numeric.mean(
            axis=0,
            skipna=True,
        )

    # ================================================================
    # Build order-specific matrices WITH MS
    # ================================================================
    raw_full={}
    base_full={}
    previous_full={}
    historical_max_full={}

    for method in method_order:
        for order_name in order_names:
            order=list(
                task_orders[order_name]
            )

            row_keys=[
                "Base"
            ]+order

            col_keys=[
                "ms"
            ]+order

            seed_raw=[]
            seed_base=[]
            seed_previous=[]
            seed_historical=[]

            for seed in seeds:

                raw=np.full(
                    (
                        len(row_keys),
                        len(col_keys),
                    ),
                    np.nan,
                    dtype=float,
                )

                # Base checkpoint
                for j,dataset in enumerate(col_keys):

                    if dataset not in maps:
                        raise ValueError(
                            f"Missing dataset {dataset!r}."
                        )

                    if base_score_col not in maps[dataset].index:
                        raise ValueError(
                            f"{dataset}: missing base column "
                            f"{base_score_col!r}."
                        )

                    raw[0,j]=maps[
                        dataset
                    ][base_score_col]

                # Sequential checkpoints
                for stage in range(
                    1,
                    len(row_keys),
                ):
                    score_col=_score_col(
                        order_name,
                        stage,
                        method,
                        seed,
                    )

                    for j,dataset in enumerate(
                        col_keys
                    ):

                        if score_col not in maps[dataset].index:
                            raise ValueError(
                                f"{dataset}: missing score column "
                                f"{score_col!r}."
                            )

                        raw[stage,j]=maps[
                            dataset
                        ][score_col]

                # Change from base
                base_change=np.zeros_like(
                    raw
                )

                for stage in range(
                    1,
                    len(row_keys),
                ):
                    base_change[stage]=_change(
                        raw[stage],
                        raw[0],
                    )

                # Change from immediately previous checkpoint
                previous_change=np.zeros_like(
                    raw
                )

                for stage in range(
                    1,
                    len(row_keys),
                ):
                    previous_change[stage]=_change(
                        raw[stage],
                        raw[stage-1],
                    )

                # Change from PRIOR historical maximum
                historical_change=np.zeros_like(
                    raw
                )

                for stage in range(
                    1,
                    len(row_keys),
                ):
                    prior_max=np.nanmax(
                        raw[:stage],
                        axis=0,
                    )

                    historical_change[stage]=_change(
                        raw[stage],
                        prior_max,
                    )

                seed_raw.append(
                    raw
                )

                seed_base.append(
                    base_change
                )

                seed_previous.append(
                    previous_change
                )

                seed_historical.append(
                    historical_change
                )

            raw=np.nanmean(
                np.stack(
                    seed_raw
                ),
                axis=0,
            )

            base_change=np.nanmean(
                np.stack(
                    seed_base
                ),
                axis=0,
            )

            previous_change=np.nanmean(
                np.stack(
                    seed_previous
                ),
                axis=0,
            )

            historical_change=np.nanmean(
                np.stack(
                    seed_historical
                ),
                axis=0,
            )

            key=(
                method,
                order_name,
            )

            raw_full[key]={
                "values":raw,
                "row_keys":row_keys,
                "col_keys":col_keys,
            }

            base_full[key]={
                "values":base_change,
                "row_keys":row_keys,
                "col_keys":col_keys,
            }

            previous_full[key]={
                "values":previous_change,
                "row_keys":row_keys,
                "col_keys":col_keys,
            }

            historical_max_full[key]={
                "values":historical_change,
                "row_keys":row_keys,
                "col_keys":col_keys,
            }

    # ================================================================
    # Color limits
    # ================================================================
    def _data_limits(
        matrix_dict,
        symmetric=False,
    ):
        vals=[]

        for item in matrix_dict.values():
            x=np.asarray(
                item["values"],
                dtype=float,
            )

            x=x[
                np.isfinite(x)
            ]

            if x.size:
                vals.append(
                    x
                )

        if not vals:
            return (
                (-1,1)
                if symmetric
                else (0,1)
            )

        vals=np.concatenate(
            vals
        )

        if symmetric:
            vmax=float(
                np.nanmax(
                    np.abs(vals)
                )
            )

            if vmax==0:
                vmax=1e-12

            return -vmax,vmax

        return (
            float(np.nanmin(vals)),
            float(np.nanmax(vals)),
        )

    def _change_limits(matrix_dict):
        auto_min,auto_max=_data_limits(
            matrix_dict,
            symmetric=True,
        )

        vmin=(
            auto_min
            if change_vmin is None
            else float(change_vmin)
        )

        vmax=(
            auto_max
            if change_vmax is None
            else float(change_vmax)
        )

        if vmin>=vmax:
            raise ValueError(
                f"Invalid change color limits: "
                f"vmin={vmin}, vmax={vmax}"
            )

        return vmin,vmax

    raw_full_lim=_data_limits(
        raw_full
    )

    base_full_lim=_change_limits(
        base_full
    )

    previous_full_lim=_change_limits(
        previous_full
    )

    historical_full_lim=_change_limits(
        historical_max_full
    )

    # ================================================================
    # Order-averaged positional matrices
    # ================================================================
    def _average_orders(
        matrix_dict
    ):
        out={}

        row_keys=[
            "Base"
        ]+[
            f"S{i}"
            for i in range(
                1,
                n_stages+1,
            )
        ]

        col_keys=[
            "MS"
        ]+[
            f"S{i}"
            for i in range(
                1,
                n_stages+1,
            )
        ]

        for method in method_order:

            arrays=[
                np.asarray(
                    matrix_dict[
                        (
                            method,
                            order_name,
                        )
                    ]["values"],
                    dtype=float,
                )
                for order_name in order_names
            ]

            values=np.nanmean(
                np.stack(
                    arrays,
                    axis=0,
                ),
                axis=0,
            )

            out[method]={
                "values":values,
                "row_keys":row_keys,
                "col_keys":col_keys,
            }

        return out

    avg_raw_full=_average_orders(
        raw_full
    )

    avg_base_full=_average_orders(
        base_full
    )

    avg_previous_full=_average_orders(
        previous_full
    )

    avg_historical_full=_average_orders(
        historical_max_full
    )

    # ================================================================
    # Common cell drawing helper
    #
    # pcolormesh is used instead of imshow so each heatmap cell remains
    # a vector quadrilateral in PDF output.
    # ================================================================
    def _draw_matrix(
        ax,
        item,
        cmap,
        vmin,
        vmax,
    ):
        values=np.asarray(
            item["values"],
            dtype=float,
        )

        nrows,ncols=values.shape

        x_edges=np.arange(
            ncols+1,
            dtype=float,
        )

        y_edges=np.arange(
            nrows+1,
            dtype=float,
        )

        mesh=ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            edgecolors="none",
            linewidth=0,
            antialiased=False,
            rasterized=False,
        )

        # Match imshow's original top-to-bottom row orientation.
        ax.set_xlim(
            0,
            ncols,
        )

        ax.set_ylim(
            nrows,
            0,
        )

        ax.set_aspect(
            "auto"
        )

        # Optional physical square axes.
        if square_heatmaps:
            ax.set_box_aspect(1)
        else:
            ax.set_box_aspect(None)

        # Cell-center tick locations.
        x_centers=np.arange(
            ncols,
            dtype=float,
        )+0.5

        y_centers=np.arange(
            nrows,
            dtype=float,
        )+0.5

        ax.set_xticks(
            x_centers
        )

        xticklabels=[
            _display_label(x)
            for x in item["col_keys"]
        ]

        ax.set_xticklabels(
            xticklabels,
            rotation=x_tick_rotation,
            ha=(
                "center"
                if x_tick_rotation==0
                else "right"
            ),
            fontsize=xtick_fontsize,
        )

        ax.set_yticks(
            y_centers
        )

        ax.set_yticklabels(
            [
                _display_label(x)
                for x in item["row_keys"]
            ],
            fontsize=ytick_fontsize,
        )

        # ------------------------------------------------------------
        # Cell annotations.
        # ------------------------------------------------------------
        if annotate:
            for r in range(
                nrows
            ):
                for c in range(
                    ncols
                ):
                    value=values[r,c]

                    if not np.isfinite(value):
                        continue

                    if annotation_text_color=="auto":
                        rgba=mesh.cmap(
                            mesh.norm(value)
                        )

                        red,green,blue=rgba[:3]

                        luminance=(
                            0.2126*red
                            +0.7152*green
                            +0.0722*blue
                        )

                        text_color=(
                            annotation_light_color
                            if luminance<annotation_luminance_threshold
                            else annotation_dark_color
                        )

                    else:
                        text_color=annotation_text_color

                    ax.text(
                        c+0.5,
                        r+0.5,
                        f"{value:.{decimals}f}",
                        ha="center",
                        va="center",
                        fontsize=annotation_fontsize,
                        color=text_color,
                    )

        ax.tick_params(
            length=0
        )

        for spine in ax.spines.values():
            spine.set_linewidth(
                spine_linewidth
            )

        return mesh

    # ================================================================
    # COMPLETE ORDER-SPECIFIC GRID
    # ================================================================
    if plot_type=="complete":

        n_methods=len(
            method_order
        )

        n_orders=len(
            order_names
        )

        if figsize is None:
            figsize=(
                1.75*n_orders+2.6,
                1.42*n_methods+1.2,
            )

        fig,axes=plt.subplots(
            n_methods,
            n_orders,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
        )

        fig.subplots_adjust(
            left=(
                0.14
                if left_margin is None
                else left_margin
            ),
            right=(
                0.91
                if right_margin is None
                else right_margin
            ),
            bottom=(
                0.07
                if bottom_margin is None
                else bottom_margin
            ),
            top=(
                0.95
                if top_margin is None
                else top_margin
            ),
            wspace=complete_wspace,
            hspace=complete_hspace,
        )

        mesh=None

        for i,method in enumerate(
            method_order
        ):
            for j,order_name in enumerate(
                order_names
            ):

                ax=axes[i,j]

                item=raw_full[
                    (
                        method,
                        order_name,
                    )
                ]

                mesh=_draw_matrix(
                    ax,
                    item,
                    complete_cmap_obj,
                    raw_full_lim[0],
                    raw_full_lim[1],
                )

                if i==0:
                    ax.set_title(
                        _order_title(
                            order_name
                        ),
                        fontsize=column_title_fontsize,
                        pad=column_title_pad,
                    )

                if j==0:
                    ax.set_ylabel(
                        _method_label(
                            method
                        ),
                        fontsize=method_label_fontsize,
                        rotation=method_label_rotation,
                        ha="center",
                        va="center",
                        labelpad=method_label_pad,
                    )

        # ------------------------------------------------------------
        # External vertical colorbar.
        # ------------------------------------------------------------
        cax=fig.add_axes([
            0.93,
            0.16,
            0.014,
            0.68,
        ])

        cb=fig.colorbar(
            mesh,
            cax=cax,
        )

        if getattr(cb,"solids",None) is not None:
            cb.solids.set_edgecolor(
                "face"
            )

        cb.ax.tick_params(
            labelsize=colorbar_fontsize
        )

        cb.set_label(
            "mAP",
            fontsize=colorbar_label_fontsize,
        )

        if show_top_title:

            title=(
                "Complete continual learning performance matrices with MS"
                if top_title is None
                else top_title
            )

            fig.suptitle(
                title,
                fontsize=top_title_fontsize,
                y=top_title_y,
            )

    # ================================================================
    # ORDER-AVERAGED SUMMARY GRID
    # ================================================================
    else:

        n_methods=len(
            method_order
        )

        if figsize is None:
            figsize=(
                12,
                max(
                    8,
                    1.55*n_methods+1.7,
                ),
            )

        fig,axes=plt.subplots(
            n_methods,
            4,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
        )

        fig.subplots_adjust(
            left=(
                0.17
                if left_margin is None
                else left_margin
            ),
            right=(
                0.98
                if right_margin is None
                else right_margin
            ),
            bottom=(
                0.12
                if bottom_margin is None
                else bottom_margin
            ),
            top=(
                0.93
                if top_margin is None
                else top_margin
            ),
            wspace=average_wspace,
            hspace=average_hspace,
        )

        change_text=(
            "ΔAP"
            if change_mode=="delta"
            else "% change"
        )

        base_label=(
            f"{change_text} from base"
        )

        previous_label=(
            f"{change_text} from previous checkpoint"
        )

        historical_label=(
            f"{change_text} from prior historical maximum"
        )

        specs=[
            {
                "matrix":
                    avg_raw_full,
                "title":
                    "Raw mean performance",
                "cmap":
                    raw_cmap_obj,
                "vmin":
                    raw_full_lim[0],
                "vmax":
                    raw_full_lim[1],
                "colorbar_label":
                    "mAP",
            },

            {
                "matrix":
                    avg_base_full,
                "title":
                    base_label,
                "cmap":
                    change_cmap_obj,
                "vmin":
                    base_full_lim[0],
                "vmax":
                    base_full_lim[1],
                "colorbar_label":
                    base_label,
            },

            {
                "matrix":
                    avg_previous_full,
                "title":
                    previous_label,
                "cmap":
                    change_cmap_obj,
                "vmin":
                    previous_full_lim[0],
                "vmax":
                    previous_full_lim[1],
                "colorbar_label":
                    previous_label,
            },

            {
                "matrix":
                    avg_historical_full,
                "title":
                    historical_label,
                "cmap":
                    change_cmap_obj,
                "vmin":
                    historical_full_lim[0],
                "vmax":
                    historical_full_lim[1],
                "colorbar_label":
                    historical_label,
            },
        ]

        column_meshes=[]

        for j,spec in enumerate(
            specs
        ):

            mesh=None

            for i,method in enumerate(
                method_order
            ):

                ax=axes[i,j]

                item=spec[
                    "matrix"
                ][method]

                mesh=_draw_matrix(
                    ax,
                    item,
                    spec["cmap"],
                    spec["vmin"],
                    spec["vmax"],
                )

                if i==0:
                    wrapped_title="\n".join(
                        textwrap.wrap(
                            spec["title"],
                            width=
                                summary_title_wrap_width,
                        )
                    )

                    ax.set_title(
                        wrapped_title,
                        fontsize=column_title_fontsize,
                        pad=column_title_pad,
                    )

                if j==0:
                    ax.set_ylabel(
                        _method_label(
                            method
                        ),
                        fontsize=method_label_fontsize,
                        rotation=method_label_rotation,
                        ha="center",
                        va="center",
                        labelpad=method_label_pad,
                    )

            column_meshes.append(
                mesh
            )

        # ------------------------------------------------------------
        # One horizontal vector colorbar per metric column.
        # ------------------------------------------------------------
        for j,(spec,mesh) in enumerate(
            zip(
                specs,
                column_meshes,
            )
        ):

            cb=fig.colorbar(
                mesh,
                ax=axes[:,j],
                orientation="horizontal",
                fraction=0.025,
                pad=0.03,
                aspect=30,
            )

            if getattr(cb,"solids",None) is not None:
                cb.solids.set_edgecolor(
                    "face"
                )

            cb.ax.tick_params(
                labelsize=colorbar_fontsize
            )

            cb.set_label(
                spec["colorbar_label"],
                fontsize=colorbar_label_fontsize,
            )

        if show_top_title:

            title=(
                "Order-averaged continual learning heatmaps with MS"
                if top_title is None
                else top_title
            )

            fig.suptitle(
                title,
                fontsize=top_title_fontsize,
                y=top_title_y,
            )

    # ================================================================
    # Save / show
    # ================================================================
    if save_path is not None:

        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches=bbox_inches,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                dpi=dpi,
                bbox_inches=bbox_inches,
            )

    if show:
        plt.show()

    return {
        "figure":fig,
        "axes":axes,
        "seeds":seeds,

        "raw_full":
            raw_full,

        "change_from_base_full":
            base_full,

        "change_from_previous_full":
            previous_full,

        "change_from_historical_max_full":
            historical_max_full,

        "order_average":{
            "raw_full":
                avg_raw_full,

            "change_from_base_full":
                avg_base_full,

            "change_from_previous_full":
                avg_previous_full,

            "change_from_historical_max_full":
                avg_historical_full,
        },
    }


import re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def compute_joint_seeded_allele_ap(
    test_sets,
    method_prefixes,
    steps=(300,1200),
    base_score_col="mhcprime_abt",
    label_col="label",
    allele_col="allele",
    seed_tag="seed",
    verbose=True,
):
    """
    Compute per-allele AP for jointly updated multi-dataset models.

    Parameters
    ----------
    test_sets : dict
        {test_set_name: dataframe}

        Each dataframe contains the base score plus seed-specific jointly
        updated model score columns.

    method_prefixes : dict
        Mapping from output method name to score-column prefix.

        Example:
        {
            "FT":"model_ft_multi_v2_merged",
            "PepCL":"model_pepcl_multi_v2",
        }

    steps : iterable[int]
        Joint-training step budgets to include.

    base_score_col : str
        Deterministic pretrained reference-model score column.

    Returns
    -------
    dict
        {test_set_name: per-allele AP dataframe}

        Example output columns:
            allele
            mhcprime_abt
            model_ft_multi_v2_merged_300step_seed_0
            ...
            model_pepcl_multi_v2_1200step_seed_5

    Notes
    -----
    No averaging across seeds is performed here.
    """
    if not test_sets:
        raise ValueError("test_sets is empty.")

    if not method_prefixes:
        raise ValueError("method_prefixes is empty.")

    steps=tuple(int(s) for s in steps)

    def _find_score_cols(df):
        cols=[]

        for method,prefix in method_prefixes.items():
            for step in steps:
                pattern=re.compile(
                    rf"^{re.escape(prefix)}_{step}step_{re.escape(seed_tag)}_(\d+)$"
                )

                matched=[
                    (int(m.group(1)),c)
                    for c in df.columns
                    if (m:=pattern.match(str(c))) is not None
                ]

                if not matched:
                    raise ValueError(
                        f"No score columns found for method={method!r}, "
                        f"step={step}, prefix={prefix!r}."
                    )

                matched=sorted(matched,key=lambda x:x[0])
                cols.extend(c for _,c in matched)

        return list(dict.fromkeys(cols))

    output={}

    for test_name,df in test_sets.items():
        score_cols=_find_score_cols(df)

        required=[allele_col,label_col,base_score_col]+score_cols
        missing=[c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"{test_name}: missing {len(missing)} required columns. "
                f"First missing: {missing[:10]}"
            )

        if verbose:
            print(f"\n{'='*70}")
            print(f"Computing joint allele AP: {test_name}")
            print(f"Alleles: {df[allele_col].nunique()}")
            print(f"Model columns: {len(score_cols)}")
            print(f"{'='*70}")

        rows=[]

        for allele,g in tqdm(
            df.groupby(allele_col,sort=False),
            total=df[allele_col].nunique(),
            desc=f"{test_name} | Joint AP",
            leave=True,
        ):
            y=g[label_col].to_numpy()
            row={allele_col:allele}

            if np.sum(y==1)==0:
                row[base_score_col]=np.nan
                for col in score_cols:
                    row[col]=np.nan

            else:
                row[base_score_col]=average_precision_score(
                    y,
                    g[base_score_col].to_numpy(),
                )

                for col in score_cols:
                    row[col]=average_precision_score(
                        y,
                        g[col].to_numpy(),
                    )

            rows.append(row)

        output[test_name]=pd.DataFrame(rows)

        if verbose:
            print(
                f"{test_name}: {len(output[test_name])} alleles × "
                f"{len(output[test_name].columns)-1} AP columns"
            )

    if verbose:
        print("\nDone.")
        print("Output keys:",list(output))

    return output

def summarize_joint_vs_sequential_allele_ap(
    sequential_allele_ap,
    joint_allele_ap,
    task_orders,
    sequential_method_suffixes,
    joint_method_prefixes,
    dataset_keys=None,
    joint_steps=(300,1200),
    base_score_col="mhcprime",
    allele_col="allele",
    seed_tag="seed",
    verbose=True,
):
    """
    Build representative per-allele joint-vs-sequential comparison tables.

    IMPORTANT
    ---------
    The reference MHCPrime AP is taken ONLY from sequential_allele_ap.
    Any MHCPrime/reference column in joint_allele_ap is ignored entirely.

    Sequential aggregation
    ----------------------
    For each allele and method:
        1. take the FINAL checkpoint from every task order;
        2. average seeds within each order;
        3. average the resulting order-level means across task orders.

    Joint aggregation
    -----------------
    For each allele, method, and step budget:
        average seed-specific APs.

    Returns
    -------
    representative : dict
        {dataset: dataframe}

        Example columns:
            allele
            mhcprime
            sequential_FT
            sequential_PepCL
            joint_FT_300
            joint_PepCL_300
            joint_FT_1200
            joint_PepCL_1200

    replicate_values : dict
        Underlying values retained for later uncertainty/statistical analyses.

        Sequential:
            one column per task order after averaging seeds within that order.

        Joint:
            one column per seed.
    """
    if not sequential_allele_ap:
        raise ValueError("sequential_allele_ap is empty.")
    if not joint_allele_ap:
        raise ValueError("joint_allele_ap is empty.")
    if not task_orders:
        raise ValueError("task_orders is empty.")
    if not sequential_method_suffixes:
        raise ValueError("sequential_method_suffixes is empty.")
    if not joint_method_prefixes:
        raise ValueError("joint_method_prefixes is empty.")

    if dataset_keys is None:
        dataset_keys=[
            k for k in sequential_allele_ap
            if k in joint_allele_ap
        ]
    else:
        dataset_keys=list(dataset_keys)

    if not dataset_keys:
        raise ValueError("No dataset keys selected.")

    missing_seq=[
        k for k in dataset_keys
        if k not in sequential_allele_ap
    ]
    missing_joint=[
        k for k in dataset_keys
        if k not in joint_allele_ap
    ]

    if missing_seq:
        raise ValueError(
            f"Missing sequential AP datasets: {missing_seq}"
        )

    if missing_joint:
        raise ValueError(
            f"Missing joint AP datasets: {missing_joint}"
        )

    order_lengths={
        order_name:len(order)
        for order_name,order in task_orders.items()
    }

    if len(set(order_lengths.values()))!=1:
        raise ValueError(
            "Task orders have different numbers of stages. "
            f"Observed: {order_lengths}"
        )

    final_stage=next(iter(order_lengths.values()))
    joint_steps=tuple(int(s) for s in joint_steps)

    def _ensure_allele_column(df):
        if allele_col in df.columns:
            return df.copy()

        if df.index.name==allele_col:
            return df.reset_index()

        raise ValueError(
            f"Could not find allele column {allele_col!r}."
        )

    def _seeded_columns(df,prefix):
        pattern=re.compile(
            rf"^{re.escape(prefix)}_{re.escape(seed_tag)}_(\d+)$"
        )

        matched=[]

        for col in df.columns:
            m=pattern.match(str(col))

            if m is not None:
                matched.append(
                    (int(m.group(1)),col)
                )

        return [
            col
            for _,col in sorted(matched)
        ]

    representative={}
    replicate_values={}

    for dataset in dataset_keys:
        seq=_ensure_allele_column(
            sequential_allele_ap[dataset]
        )

        joint=_ensure_allele_column(
            joint_allele_ap[dataset]
        )

        # -------------------------------------------------------------
        # Match only on allele identity.
        # -------------------------------------------------------------
        seq_alleles=set(
            seq[allele_col].astype(str)
        )

        joint_alleles=set(
            joint[allele_col].astype(str)
        )

        common=seq_alleles & joint_alleles

        if not common:
            raise ValueError(
                f"{dataset}: no shared alleles between sequential "
                "and joint AP tables."
            )

        # Preserve sequential allele ordering.
        seq=seq[
            seq[allele_col]
            .astype(str)
            .isin(common)
        ].copy()

        joint=joint[
            joint[allele_col]
            .astype(str)
            .isin(common)
        ].copy()

        allele_order=(
            seq[allele_col]
            .astype(str)
            .tolist()
        )

        if len(allele_order)!=len(set(allele_order)):
            raise ValueError(
                f"{dataset}: duplicate alleles in sequential AP table."
            )

        joint_allele_strings=(
            joint[allele_col]
            .astype(str)
            .tolist()
        )

        if len(joint_allele_strings)!=len(set(joint_allele_strings)):
            raise ValueError(
                f"{dataset}: duplicate alleles in joint AP table."
            )

        seq=(
            seq
            .assign(__allele=seq[allele_col].astype(str))
            .set_index("__allele")
        )

        joint=(
            joint
            .assign(__allele=joint[allele_col].astype(str))
            .set_index("__allele")
            .reindex(allele_order)
        )

        # =============================================================
        # REFERENCE:
        # use ONLY the sequential AP table.
        # =============================================================
        if base_score_col not in seq.columns:
            raise ValueError(
                f"{dataset}: sequential AP table is missing "
                f"reference column {base_score_col!r}."
            )

        out=pd.DataFrame({
            allele_col:allele_order,
            base_score_col:
                seq[base_score_col].to_numpy(float),
        })

        rep={
            "sequential":{},
            "joint":{},
        }

        # =============================================================
        # SEQUENTIAL:
        # final stage only
        # -> average seeds within order
        # -> average across orders
        # =============================================================
        for method,suffix in sequential_method_suffixes.items():
            order_matrix=[]
            order_names=[]

            for order_name in task_orders:
                prefix=(
                    f"{order_name}"
                    f"_s{final_stage}"
                    f"{suffix}"
                )

                cols=_seeded_columns(
                    seq,
                    prefix,
                )

                if not cols:
                    raise ValueError(
                        f"{dataset}: no final-stage sequential "
                        f"columns found for order={order_name!r}, "
                        f"method={method!r}. "
                        f"Expected prefix {prefix!r}."
                    )

                # First average model seeds within this task order.
                order_mean=(
                    seq[cols]
                    .mean(axis=1,skipna=True)
                    .to_numpy(float)
                )

                order_matrix.append(order_mean)
                order_names.append(order_name)

            order_matrix=np.column_stack(
                order_matrix
            )

            # Then average the order-level means.
            out[
                f"sequential_{method}"
            ]=np.nanmean(
                order_matrix,
                axis=1,
            )

            rep["sequential"][method]=pd.DataFrame(
                order_matrix,
                index=allele_order,
                columns=order_names,
            )

        # =============================================================
        # JOINT:
        # average directly across joint-training seeds.
        #
        # The joint MHCPrime/reference column is intentionally ignored.
        # =============================================================
        for method,prefix in joint_method_prefixes.items():
            rep["joint"][method]={}

            for step in joint_steps:
                full_prefix=(
                    f"{prefix}_{step}step"
                )

                cols=_seeded_columns(
                    joint,
                    full_prefix,
                )

                if not cols:
                    raise ValueError(
                        f"{dataset}: no joint columns found for "
                        f"method={method!r}, step={step}. "
                        f"Expected prefix {full_prefix!r}."
                    )

                vals=(
                    joint[cols]
                    .to_numpy(float)
                )

                out[
                    f"joint_{method}_{step}"
                ]=np.nanmean(
                    vals,
                    axis=1,
                )

                rep["joint"][method][step]=pd.DataFrame(
                    vals,
                    index=allele_order,
                    columns=cols,
                )

        representative[dataset]=out
        replicate_values[dataset]=rep

        if verbose:
            print(
                f"{dataset}: {len(out)} common alleles | "
                f"{len(task_orders)} sequential orders | "
                f"final stage S{final_stage}"
            )
            print(
                f"  Reference: {base_score_col!r} taken only "
                "from sequential allele AP table"
            )

            for method in sequential_method_suffixes:
                n_orders=(
                    rep["sequential"][method]
                    .shape[1]
                )

                print(
                    f"  Sequential {method}: "
                    f"{n_orders} order-level means"
                )

            for method in joint_method_prefixes:
                for step in joint_steps:
                    n_seeds=(
                        rep["joint"][method][step]
                        .shape[1]
                    )

                    print(
                        f"  Joint {method}, {step} steps: "
                        f"{n_seeds} seeds"
                    )

    return representative,replicate_values

import re
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

def summarize_joint_vs_sequential_map(
    seeded_allele_ap,
    joint_allele_ap,
    task_orders,
    sequential_method_suffixes,
    joint_method_prefixes,
    dataset_keys=("ms","yd","episcan","ba","escape"),
    assay_keys=("yd","episcan","ba","escape"),
    joint_steps=(300,1200),
    base_score_col="mhcprime",
    allele_col="allele",
    seed_tag="seed",
    ci=0.95,
    verbose=True,
):
    """
    Summarize final sequential and joint multi-dataset updating using mAP.

    Aggregation hierarchy
    ---------------------
    Sequential:
        allele AP
        -> mAP within each order x seed x dataset
        -> delta from MHCPrime within dataset
        -> average seeds within task order
        -> summarize across task orders

    Joint:
        allele AP
        -> mAP within each seed x dataset
        -> delta from MHCPrime within dataset
        -> summarize across seeds

    Sequential models use only the FINAL stage of each task order.

    The MHCPrime reference is taken ONLY from seeded_allele_ap.

    Returns
    -------
    summary_df : pd.DataFrame
        Mean/SD/SE/CI for each approach x method x step x dataset.

    replicate_df : pd.DataFrame
        Proper replicate-level values:
            sequential -> one row per task order after seed averaging
            joint      -> one row per training seed

    raw_df : pd.DataFrame
        Before hierarchical aggregation:
            sequential -> one row per order x seed
            joint      -> one row per seed
    """
    dataset_keys=list(dataset_keys)
    assay_keys=list(assay_keys)
    joint_steps=tuple(int(x) for x in joint_steps)

    if "ms" not in dataset_keys:
        raise ValueError("dataset_keys must contain 'ms'.")

    missing_assays=[x for x in assay_keys if x not in dataset_keys]
    if missing_assays:
        raise ValueError(
            f"assay_keys contains datasets absent from dataset_keys: {missing_assays}"
        )

    order_lengths={
        order_name:len(order)
        for order_name,order in task_orders.items()
    }

    if len(set(order_lengths.values()))!=1:
        raise ValueError(
            f"Task orders have different numbers of stages: {order_lengths}"
        )

    final_stage=next(iter(order_lengths.values()))

    def _ensure_allele_col(df):
        if allele_col in df.columns:
            return df.copy()
        if df.index.name==allele_col:
            return df.reset_index()
        raise ValueError(
            f"Could not find allele column {allele_col!r}."
        )

    def _base_map(dataset):
        df=_ensure_allele_col(
            seeded_allele_ap[dataset]
        )

        if base_score_col not in df.columns:
            raise ValueError(
                f"{dataset}: missing sequential reference "
                f"column {base_score_col!r}."
            )

        return float(
            df[base_score_col].mean(skipna=True)
        )

    def _find_seed_cols(df,prefix):
        pat=re.compile(
            rf"^{re.escape(prefix)}_{re.escape(seed_tag)}_(\d+)$"
        )

        found=[]

        for col in df.columns:
            m=pat.match(str(col))
            if m is not None:
                found.append(
                    (int(m.group(1)),col)
                )

        return sorted(found)

    base_maps={
        dataset:_base_map(dataset)
        for dataset in dataset_keys
    }

    if verbose:
        print("MHCPrime mAP reference:")
        for dataset,value in base_maps.items():
            print(f"  {dataset}: {value:.5f}")

        print(f"\nSequential final stage: S{final_stage}")

    raw_rows=[]

    # ================================================================
    # SEQUENTIAL
    # ================================================================
    for method,suffix in sequential_method_suffixes.items():

        for order_name in task_orders:

            for dataset in dataset_keys:
                df=_ensure_allele_col(
                    seeded_allele_ap[dataset]
                )

                prefix=(
                    f"{order_name}"
                    f"_s{final_stage}"
                    f"{suffix}"
                )

                seed_cols=_find_seed_cols(
                    df,
                    prefix,
                )

                if not seed_cols:
                    raise ValueError(
                        f"{dataset}: no sequential columns found for "
                        f"order={order_name!r}, method={method!r}. "
                        f"Expected prefix {prefix!r}."
                    )

                for seed,col in seed_cols:
                    model_map=float(
                        df[col].mean(skipna=True)
                    )

                    raw_rows.append({
                        "approach":"Sequential",
                        "method":method,
                        "step":np.nan,
                        "order":order_name,
                        "seed":seed,
                        "dataset":dataset,
                        "base_map":base_maps[dataset],
                        "model_map":model_map,
                        "delta_map":
                            model_map-base_maps[dataset],
                    })

    # ================================================================
    # JOINT
    # ================================================================
    for method,prefix in joint_method_prefixes.items():

        for step in joint_steps:

            for dataset in dataset_keys:
                df=_ensure_allele_col(
                    joint_allele_ap[dataset]
                )

                full_prefix=(
                    f"{prefix}_{step}step"
                )

                seed_cols=_find_seed_cols(
                    df,
                    full_prefix,
                )

                if not seed_cols:
                    raise ValueError(
                        f"{dataset}: no joint columns found for "
                        f"method={method!r}, step={step}. "
                        f"Expected prefix {full_prefix!r}."
                    )

                for seed,col in seed_cols:
                    model_map=float(
                        df[col].mean(skipna=True)
                    )

                    # Deliberately use the MHCPrime reference from
                    # seeded_allele_ap, not joint_allele_ap.
                    raw_rows.append({
                        "approach":"Joint",
                        "method":method,
                        "step":step,
                        "order":None,
                        "seed":seed,
                        "dataset":dataset,
                        "base_map":base_maps[dataset],
                        "model_map":model_map,
                        "delta_map":
                            model_map-base_maps[dataset],
                    })

    raw_df=pd.DataFrame(raw_rows)

    # ================================================================
    # PROPER REPLICATE LEVEL
    #
    # Sequential:
    #     average seeds within each order first.
    #
    # Joint:
    #     seed itself is the replicate.
    # ================================================================
    sequential_rep=(
        raw_df[
            raw_df["approach"].eq("Sequential")
        ]
        .groupby(
            ["approach","method","order","dataset"],
            as_index=False,
            sort=False,
        )
        .agg(
            base_map=("base_map","first"),
            model_map=("model_map","mean"),
            delta_map=("delta_map","mean"),
            n_seeds=("seed","nunique"),
        )
    )

    sequential_rep["replicate_type"]="task_order"
    sequential_rep["replicate_id"]=(
        sequential_rep["order"]
        .astype(str)
    )
    sequential_rep["step"]=np.nan

    joint_rep=(
        raw_df[
            raw_df["approach"].eq("Joint")
        ]
        .copy()
    )

    joint_rep["replicate_type"]="seed"
    joint_rep["replicate_id"]=(
        joint_rep["seed"]
        .astype(str)
    )
    joint_rep["n_seeds"]=1

    replicate_cols=[
        "approach",
        "method",
        "step",
        "dataset",
        "replicate_type",
        "replicate_id",
        "base_map",
        "model_map",
        "delta_map",
        "n_seeds",
    ]

    replicate_df=pd.concat(
        [
            sequential_rep[replicate_cols],
            joint_rep[replicate_cols],
        ],
        ignore_index=True,
    )

    # ================================================================
    # SUMMARY ACROSS THE APPROPRIATE REPLICATES
    # ================================================================
    def _summarize(x):
        x=np.asarray(x,float)
        x=x[np.isfinite(x)]
        n=len(x)

        if n==0:
            return pd.Series({
                "mean":np.nan,
                "sd":np.nan,
                "se":np.nan,
                "ci_low":np.nan,
                "ci_high":np.nan,
                "n":0,
            })

        mean=x.mean()

        if n<2:
            return pd.Series({
                "mean":mean,
                "sd":np.nan,
                "se":np.nan,
                "ci_low":np.nan,
                "ci_high":np.nan,
                "n":n,
            })

        sd=x.std(ddof=1)
        se=sd/np.sqrt(n)
        crit=student_t.ppf(
            (1+ci)/2,
            n-1,
        )

        return pd.Series({
            "mean":mean,
            "sd":sd,
            "se":se,
            "ci_low":mean-crit*se,
            "ci_high":mean+crit*se,
            "n":n,
        })

    summary_rows=[]

    group_cols=[
        "approach",
        "method",
        "step",
        "dataset",
    ]

    # groupby normally drops NaN step, so handle sequential/joint separately.
    for approach in ["Sequential","Joint"]:
        q=replicate_df[
            replicate_df["approach"].eq(approach)
        ]

        if approach=="Sequential":
            groups=q.groupby(
                ["approach","method","dataset"],
                sort=False,
            )

            for keys,g in groups:
                a,m,dataset=keys

                for metric in [
                    "model_map",
                    "delta_map",
                ]:
                    s=_summarize(
                        g[metric]
                    )

                    row_match=[
                        r for r in summary_rows
                        if (
                            r["approach"]==a
                            and r["method"]==m
                            and r["dataset"]==dataset
                            and pd.isna(r["step"])
                        )
                    ]

                    if row_match:
                        row=row_match[0]
                    else:
                        row={
                            "approach":a,
                            "method":m,
                            "step":np.nan,
                            "dataset":dataset,
                            "base_map":
                                base_maps[dataset],
                        }
                        summary_rows.append(row)

                    for stat,val in s.items():
                        row[f"{metric}_{stat}"]=val

        else:
            groups=q.groupby(
                ["approach","method","step","dataset"],
                sort=False,
            )

            for keys,g in groups:
                a,m,step,dataset=keys

                row={
                    "approach":a,
                    "method":m,
                    "step":int(step),
                    "dataset":dataset,
                    "base_map":
                        base_maps[dataset],
                }

                for metric in [
                    "model_map",
                    "delta_map",
                ]:
                    s=_summarize(
                        g[metric]
                    )

                    for stat,val in s.items():
                        row[f"{metric}_{stat}"]=val

                summary_rows.append(row)

    summary_df=pd.DataFrame(
        summary_rows
    )

    if verbose:
        print("\nReplicate hierarchy:")
        print(
            "  Sequential: mAP within seed -> "
            "mean seeds within order -> summarize across orders"
        )
        print(
            "  Joint: mAP within seed -> summarize across seeds"
        )

        print(
            f"\nSequential task orders: {len(task_orders)}"
        )

        for method in joint_method_prefixes:
            for step in joint_steps:
                q=replicate_df[
                    replicate_df["approach"].eq("Joint")
                    &
                    replicate_df["method"].eq(method)
                    &
                    replicate_df["step"].eq(step)
                ]

                print(
                    f"  Joint {method}, {step}: "
                    f"{q['replicate_id'].nunique()} seeds"
                )

    return summary_df,replicate_df,raw_df


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_ms_vs_assay_scatter_per_assay(
    scatter_summary,
    assay_order=("yd", "episcan", "ba", "escape"),
    ms_dataset="ms",
    dataset_labels=None,
    label_order=None,
    label_map=None,
    label_colors=None,
    figsize=(12, 10),
    dpi=300,
    point_size=110,
    edgecolor="black",
    linewidth=1.0,
    alpha=0.95,
    annotate=False,
    annotation_fontsize=8,
    annotation_dx=0.0,
    annotation_dy=0.0,
    force_ms_to_zero=True,
    force_assay_from_zero=True,
    xpad_frac=0.08,
    ypad_frac=0.08,
    sharex=False,
    sharey=False,
    show_zero_lines=True,
    zero_line_color="0.5",
    zero_line_style="--",
    zero_linewidth=1.0,
    show_diag=False,
    diag_color="0.7",
    diag_style=":",
    diag_linewidth=1.0,
    square_panels=True,
    marker="o",
    legend=True,
    legend_order=None,
    legend_loc="lower center",
    legend_ncol=3,
    legend_frame=False,
    legend_fontsize=12,
    title_fontsize=16,
    axis_fontsize=12,
    tick_fontsize=10,
    save_path=None,
    save_pdf=True,
    save_png=True,
    bbox_inches="tight",
    pad_inches=0.1,
    show=True,
):
    """
    One panel per assay.
    x-axis: MS change from MHCPrime (delta mAP)
    y-axis: assay change from MHCPrime (delta mAP)
    One point per method / operating point.

    Required columns in scatter_summary:
        approach, method, step, dataset, base_map, model_map_mean
    """
    df = scatter_summary.copy()
    required = {"approach", "method", "step", "dataset", "base_map", "model_map_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"scatter_summary missing required columns: {sorted(missing)}")

    if dataset_labels is None:
        dataset_labels = {
            "ms": "MS",
            "yd": "YD",
            "episcan": "EpiScan",
            "ba": "BA",
            "escape": "ESCAPE-seq",
        }

    def _norm_step(v):
        if pd.isna(v):
            return None
        try:
            fv = float(v)
            return int(fv) if fv.is_integer() else fv
        except Exception:
            return v

    def _build_label(row):
        step = _norm_step(row["step"])
        if row["approach"] == "Sequential":
            return f"Sequential {row['method']}"
        if step is None:
            return f"{row['approach']} {row['method']}"
        return f"{row['approach']} {row['method']}, {step} steps"

    df["plot_label"] = df.apply(_build_label, axis=1)
    df["delta"] = df["model_map_mean"] - df["base_map"]

    if label_order is None:
        label_order, seen = [], set()
        for lab in df["plot_label"]:
            if lab not in seen:
                label_order.append(lab)
                seen.add(lab)

    if label_map is None:
        label_map = {lab: lab for lab in label_order}

    if label_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        label_colors = {lab: cycle[i % len(cycle)] for i, lab in enumerate(label_order)}

    panel_rows = []
    for assay in assay_order:
        assay_df = df[df["dataset"] == assay].copy()
        if assay_df.empty:
            continue

        for _, assay_row in assay_df.iterrows():
            mask = (
                (df["dataset"] == ms_dataset) &
                (df["approach"] == assay_row["approach"]) &
                (df["method"] == assay_row["method"])
            )
            assay_step = _norm_step(assay_row["step"])
            if assay_step is None:
                mask &= df["step"].isna()
            else:
                mask &= df["step"].apply(_norm_step) == assay_step

            ms_matches = df[mask]
            if len(ms_matches) == 0:
                raise ValueError(
                    f"No matching MS row for assay={assay}, approach={assay_row['approach']}, "
                    f"method={assay_row['method']}, step={assay_row['step']}"
                )
            if len(ms_matches) > 1:
                raise ValueError(
                    f"Multiple matching MS rows for assay={assay}, approach={assay_row['approach']}, "
                    f"method={assay_row['method']}, step={assay_row['step']}"
                )

            ms_row = ms_matches.iloc[0]
            panel_rows.append({
                "dataset": assay,
                "dataset_label": dataset_labels.get(assay, assay),
                "plot_label": assay_row["plot_label"],
                "display_label": label_map.get(assay_row["plot_label"], assay_row["plot_label"]),
                "x": ms_row["delta"],
                "y": assay_row["delta"],
            })

    plot_df = pd.DataFrame(panel_rows)
    if plot_df.empty:
        raise ValueError("No assay points available to plot.")

    assays_present = [a for a in assay_order if a in set(plot_df["dataset"])]
    n = len(assays_present)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False, sharex=sharex, sharey=sharey
    )
    axes = axes.ravel()

    def _lims(vals, pad_frac=0.08, force_zero_side=None):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return (-1, 1)
        vmin, vmax = np.min(vals), np.max(vals)
        if vmin == vmax:
            pad = 0.05 if vmin == 0 else abs(vmin) * pad_frac
            lo, hi = vmin - pad, vmax + pad
        else:
            pad = (vmax - vmin) * pad_frac
            lo, hi = vmin - pad, vmax + pad
        if force_zero_side == "right":
            hi = max(0.0, hi)
        elif force_zero_side == "bottom":
            lo = min(0.0, lo)
        return lo, hi

    global_xlim = global_ylim = None
    if sharex:
        global_xlim = _lims(plot_df["x"], xpad_frac, "right" if force_ms_to_zero else None)
    if sharey:
        global_ylim = _lims(plot_df["y"], ypad_frac, "bottom" if force_assay_from_zero else None)

    rank = {lab: i for i, lab in enumerate(label_order)}

    for i, assay in enumerate(assays_present):
        ax = axes[i]
        sub = plot_df[plot_df["dataset"] == assay].copy()
        sub["__rank"] = sub["plot_label"].map(lambda x: rank.get(x, 10**6))
        sub = sub.sort_values("__rank").drop(columns="__rank")

        for _, r in sub.iterrows():
            lab = r["plot_label"]
            ax.scatter(
                r["x"], r["y"],
                s=point_size,
                color=label_colors.get(lab, "#4c78a8"),
                marker=marker,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=3,
            )
            if annotate:
                ax.text(
                    r["x"] + annotation_dx,
                    r["y"] + annotation_dy,
                    label_map.get(lab, lab),
                    fontsize=annotation_fontsize,
                    ha="left",
                    va="bottom",
                )

        if show_zero_lines:
            ax.axvline(0, color=zero_line_color, ls=zero_line_style, lw=zero_linewidth, zorder=1)
            ax.axhline(0, color=zero_line_color, ls=zero_line_style, lw=zero_linewidth, zorder=1)

        if show_diag:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            lo, hi = min(x0, y0), max(x1, y1)
            ax.plot([lo, hi], [lo, hi], color=diag_color, ls=diag_style, lw=diag_linewidth, zorder=1)

        ax.set_title(dataset_labels.get(assay, assay), fontsize=title_fontsize)
        ax.set_xlabel("MS change from MHCPrime (ΔmAP)", fontsize=axis_fontsize)
        ax.set_ylabel("Assay change from MHCPrime (ΔmAP)", fontsize=axis_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

        if sharex and global_xlim is not None:
            ax.set_xlim(global_xlim)
        else:
            ax.set_xlim(_lims(sub["x"], xpad_frac, "right" if force_ms_to_zero else None))

        if sharey and global_ylim is not None:
            ax.set_ylim(global_ylim)
        else:
            ax.set_ylim(_lims(sub["y"], ypad_frac, "bottom" if force_assay_from_zero else None))

        if square_panels:
            ax.set_box_aspect(1)

    for j in range(len(assays_present), len(axes)):
        axes[j].axis("off")

    if legend:
        if legend_order is None:
            legend_order = label_order

        handles = []
        present = set(plot_df["plot_label"])
        for lab in legend_order:
            if lab not in present:
                continue
            handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    linestyle="none",
                    markerfacecolor=label_colors.get(lab, "#4c78a8"),
                    markeredgecolor=edgecolor,
                    markeredgewidth=linewidth,
                    markersize=np.sqrt(point_size),
                    label=label_map.get(lab, lab),
                )
            )

        fig.legend(
            handles=handles,
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=legend_frame,
            fontsize=legend_fontsize,
        )

    plt.tight_layout(rect=(0, 0.06 if legend else 0, 1, 1))

    if save_path is not None:
        if save_pdf:
            fig.savefig(f"{save_path}.pdf", bbox_inches=bbox_inches, pad_inches=pad_inches)
        if save_png:
            fig.savefig(f"{save_path}.png", bbox_inches=bbox_inches, pad_inches=pad_inches)

    if show:
        plt.show()

    return fig, axes, plot_df

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

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def compute_long_horizon_allele_ap(
    adaptive_test_sets, task_order, adaptive_50pct_test_sets=None,
    task_order_name="t1_allele_partition", base_score_col="mhcprime",
    adaptive_suffix="_pepcl_multi_replay_dr_full_buffer",
    adaptive_50pct_suffix="_pepcl_multi_replay_dr_full_buffer_50pct_ms",
    allele_col="allele", label_col="label", verbose=True
):
    """
    Compute per-allele AP for long-horizon continual-learning predictions.

    Supports two input modes:

    1. TWO-DICT MODE
       adaptive_test_sets contains adaptive-buffer score columns.
       adaptive_50pct_test_sets contains fixed-50%-MS score columns.

    2. SINGLE-DICT MODE
       adaptive_test_sets already contains BOTH sets of score columns.
       Set adaptive_50pct_test_sets=None.

    Only evaluation sets corresponding to the partitioned training tasks
    are retained, plus MS. Extra test sets such as cov_test are ignored.

    Returns
    -------
    dict
        {
            "ms": per-allele AP dataframe,
            "yd": per-allele AP dataframe,
            "episcan": per-allele AP dataframe,
            "ba": per-allele AP dataframe,
            "escape": per-allele AP dataframe,
        }

    Each dataframe contains:
        allele
        base_score_col
        all adaptive-buffer checkpoint AP columns
        all fixed-50%-MS checkpoint AP columns
    """
    single_dict_mode=adaptive_50pct_test_sets is None

    def _base_task_name(name):
        return re.sub(r"_\d+$","",name)

    assay_keys=list(dict.fromkeys(_base_task_name(task) for task in task_order))
    wanted_keys=["ms"]+assay_keys
    n_stages=len(task_order)

    if verbose:
        print(f"Input mode: {'single combined test-set dictionary' if single_dict_mode else 'two test-set dictionaries'}")
        print(f"Long-horizon task order ({n_stages} updates):")
        print(task_order)
        print(f"\nEvaluation sets retained: {wanted_keys}")

    missing=[k for k in wanted_keys if k not in adaptive_test_sets]
    if missing:
        raise ValueError(f"Missing test sets from adaptive_test_sets: {missing}")

    if not single_dict_mode:
        missing=[k for k in wanted_keys if k not in adaptive_50pct_test_sets]
        if missing:
            raise ValueError(f"Missing test sets from adaptive_50pct_test_sets: {missing}")

    adaptive_cols=[
        f"{task_order_name}_s{s}{adaptive_suffix}"
        for s in range(1,n_stages+1)
    ]
    adaptive_50pct_cols=[
        f"{task_order_name}_s{s}{adaptive_50pct_suffix}"
        for s in range(1,n_stages+1)
    ]

    def _compute_allele_ap(df,score_cols,desc):
        score_cols=list(dict.fromkeys(score_cols))
        required=[allele_col,label_col,base_score_col]+score_cols
        missing=[c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"{desc}: missing {len(missing)} required columns.\n"
                f"First missing columns: {missing[:10]}"
            )

        rows=[]
        grouped=df.groupby(allele_col,sort=False)

        for allele,g in tqdm(
            grouped,total=df[allele_col].nunique(),
            desc=desc,leave=True
        ):
            y=g[label_col].to_numpy()
            row={allele_col:allele}

            if np.sum(y==1)==0:
                for col in [base_score_col]+score_cols:
                    row[col]=np.nan
            else:
                for col in [base_score_col]+score_cols:
                    row[col]=average_precision_score(y,g[col].to_numpy())

            rows.append(row)

        return pd.DataFrame(rows)

    output={}

    for test_name in wanted_keys:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Computing allele AP: {test_name}")
            print(f"{'='*70}")

        # =====================================================
        # SINGLE-DICT MODE
        # =====================================================
        if single_dict_mode:
            all_score_cols=adaptive_cols+adaptive_50pct_cols

            merged=_compute_allele_ap(
                adaptive_test_sets[test_name],
                all_score_cols,
                desc=f"{test_name} | Long-horizon AP"
            )

        # =====================================================
        # TWO-DICT MODE
        # =====================================================
        else:
            ap_adaptive=_compute_allele_ap(
                adaptive_test_sets[test_name],
                adaptive_cols,
                desc=f"{test_name} | Adaptive buffer AP"
            )

            ap_50pct=_compute_allele_ap(
                adaptive_50pct_test_sets[test_name],
                adaptive_50pct_cols,
                desc=f"{test_name} | 50% MS buffer AP"
            )

            alleles_a=set(ap_adaptive[allele_col])
            alleles_b=set(ap_50pct[allele_col])

            if alleles_a!=alleles_b:
                raise ValueError(
                    f"{test_name}: allele sets differ between dictionaries.\n"
                    f"Only adaptive: {sorted(alleles_a-alleles_b)[:10]}\n"
                    f"Only 50% MS: {sorted(alleles_b-alleles_a)[:10]}"
                )

            ap_50pct=(
                ap_50pct
                .set_index(allele_col)
                .reindex(ap_adaptive[allele_col])
                .reset_index()
            )

            # Validate that the reference model produces the same
            # per-allele AP in both copies of the test set.
            a=ap_adaptive[base_score_col].to_numpy(float)
            b=ap_50pct[base_score_col].to_numpy(float)

            if not np.allclose(a,b,equal_nan=True):
                raise ValueError(
                    f"{test_name}: {base_score_col!r} allele AP differs "
                    "between the two test-set dictionaries."
                )

            ap_50pct=ap_50pct.drop(columns=base_score_col)

            merged=ap_adaptive.merge(
                ap_50pct,
                on=allele_col,
                how="inner",
                validate="one_to_one"
            )

        output[test_name]=merged

        if verbose:
            print(
                f"{test_name}: {len(merged)} alleles, "
                f"{len(merged.columns)-1} AP score columns"
            )

    if verbose:
        print("\nDone.")
        print("Output keys:",list(output))
        for key,df in output.items():
            print(f"  {key}: {df.shape}")

    return output

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_long_horizon_cl_metrics(
    allele_ap_dict, task_order, method_suffixes,
    task_order_name="t1_allele_partition", base_score_col="mhcprime",
    partition_summary=None, allele_specific_tasks=False,
    merge_partitions_by_assay=True,
    allele_col="allele", method_colors=None, dataset_labels=None,
    figsize_raw=(16,9), figsize_metrics=(16,9), dpi=150,
    linewidth=1.6, markersize=4, verbose=True, show=True
):
    """
    Plot long-horizon continual-learning performance.

    Supports:
      1. Full-assay evaluation.
      2. Allele-specific partition evaluation.
      3. Allele-specific evaluation with partitions merged back into their
         parent assay so later partitions do not increase the denominator.

    Example task order:
        yd_1, episcan_1, ba_1, escape_1,
        yd_2, episcan_2, ba_2, escape_2, ...

    Parameters
    ----------
    allele_ap_dict : dict
        Per-dataset per-allele AP dataframes:
            {"ms": df, "yd": df, "episcan": df, "ba": df, "escape": df}

    task_order : list
        Sequential partitioned updates.

    method_suffixes : dict
        {method_display_name: score_column_suffix}

    partition_summary : dict or None
        Required when allele_specific_tasks=True.
        Must contain an "alleles" list for every partition in task_order.

    allele_specific_tasks : bool
        False:
            Use complete assay test sets.

        True:
            Evaluate each partition using the intersection between its
            train-partition alleles and alleles actually present in the
            corresponding test AP dataframe.

            Missing train alleles are ignored. An error is raised only if
            none of the partition alleles are available in the test set.

    merge_partitions_by_assay : bool
        Relevant only when allele_specific_tasks=True.

        True:
            Accumulate available alleles within each assay and evaluate one
            assay-level quantity. Later partitions do not increase the number
            of tasks in aggregate averages.

        False:
            Treat each allele partition as its own CL task.

    Returns
    -------
    dict
        full_performance
        partition_performance
        assay_specific_performance
        metrics
        acquisition_stage
        partition_test_alleles
        acquired_alleles
        figures
    """
    if allele_specific_tasks and partition_summary is None:
        raise ValueError(
            "partition_summary must be supplied when allele_specific_tasks=True."
        )

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)
    methods=list(method_suffixes)

    def _base_task(name):
        return re.sub(r"_\d+$","",name)

    def _label(name):
        return dataset_labels.get(name,name)

    assay_order=list(dict.fromkeys(_base_task(x) for x in task_order))
    eval_domains=["ms"]+assay_order
    n_stages=len(task_order)

    missing=[d for d in eval_domains if d not in allele_ap_dict]
    if missing:
        raise ValueError(f"Missing evaluation datasets: {missing}")

    # ---------------------------------------------------------
    # Validate AP frames.
    # ---------------------------------------------------------
    numeric_frames={}

    for dataset in eval_domains:
        df=allele_ap_dict[dataset]

        if allele_col not in df.columns:
            raise ValueError(
                f"{dataset}: missing allele column {allele_col!r}."
            )

        numeric=df.select_dtypes(include=[np.number]).copy()

        if base_score_col not in numeric.columns:
            raise ValueError(
                f"{dataset}: base score column {base_score_col!r} "
                "is missing or nonnumeric."
            )

        numeric_frames[dataset]=numeric

    def _score_col(stage,method):
        return f"{task_order_name}_s{stage}{method_suffixes[method]}"

    for dataset in eval_domains:
        numeric=numeric_frames[dataset]

        for method in methods:
            missing_cols=[
                _score_col(stage,method)
                for stage in range(1,n_stages+1)
                if _score_col(stage,method) not in numeric.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"{dataset}: missing {len(missing_cols)} score columns "
                    f"for {method!r}. First missing: {missing_cols[:5]}"
                )

    # ---------------------------------------------------------
    # Determine usable TEST alleles for every partition.
    #
    # We use intersection(train partition alleles, test AP alleles).
    # Missing train alleles are allowed.
    # Error only if intersection is empty.
    # ---------------------------------------------------------
    partition_test_alleles={}

    if allele_specific_tasks:
        missing_summary=[
            task for task in task_order
            if task not in partition_summary
        ]

        if missing_summary:
            raise ValueError(
                f"partition_summary is missing tasks: {missing_summary}"
            )

        for task in task_order:
            assay=_base_task(task)

            requested=[
                str(a) for a in partition_summary[task]["alleles"]
            ]

            available_ordered=(
                allele_ap_dict[assay][allele_col]
                .astype(str)
                .tolist()
            )
            available_set=set(available_ordered)
            requested_set=set(requested)

            usable=[
                a for a in requested
                if a in available_set
            ]

            missing_alleles=sorted(
                requested_set-available_set
            )

            if len(usable)==0:
                raise ValueError(
                    f"{task}: none of its {len(requested)} partition alleles "
                    f"are present in the {assay!r} test AP dataframe."
                )

            partition_test_alleles[task]=usable

            if verbose:
                print(
                    f"{task}: using {len(usable)}/{len(requested)} "
                    f"partition alleles present in test"
                )

                if missing_alleles:
                    print(
                        f"  ignoring {len(missing_alleles)} train-only alleles: "
                        f"{missing_alleles[:10]}"
                        + (" ..." if len(missing_alleles)>10 else "")
                    )

    # ---------------------------------------------------------
    # Macro AP helper.
    # ---------------------------------------------------------
    def _macro(dataset,col,alleles=None):
        df=allele_ap_dict[dataset]

        if col not in df.columns:
            raise ValueError(
                f"{dataset}: missing AP column {col!r}."
            )

        if alleles is None:
            values=df[col]

        else:
            allele_set={str(a) for a in alleles}
            mask=df[allele_col].astype(str).isin(allele_set)

            if mask.sum()==0:
                raise ValueError(
                    f"{dataset}: none of the requested alleles are "
                    "present in the AP dataframe."
                )

            values=df.loc[mask,col]

        return float(values.mean(skipna=True))

    # ---------------------------------------------------------
    # First biological-assay acquisition stage.
    # ---------------------------------------------------------
    acquisition_stage={}

    for stage,task in enumerate(task_order,start=1):
        assay=_base_task(task)

        if assay not in acquisition_stage:
            acquisition_stage[assay]=stage

    # ---------------------------------------------------------
    # Accumulated AVAILABLE test alleles by assay and stage.
    # ---------------------------------------------------------
    acquired_alleles={stage:{} for stage in range(n_stages+1)}

    if allele_specific_tasks:
        running={assay:[] for assay in assay_order}

        acquired_alleles[0]={
            assay:[] for assay in assay_order
        }

        for stage,task in enumerate(task_order,start=1):
            assay=_base_task(task)
            existing=set(running[assay])

            for allele in partition_test_alleles[task]:
                if allele not in existing:
                    running[assay].append(allele)
                    existing.add(allele)

            acquired_alleles[stage]={
                assay:list(vals)
                for assay,vals in running.items()
            }

    # =========================================================
    # FULL TEST-SET PERFORMANCE
    # =========================================================
    full_performance={}

    for method in methods:
        perf=pd.DataFrame(
            index=np.arange(n_stages+1),
            columns=eval_domains,
            dtype=float
        )

        for dataset in eval_domains:
            perf.loc[0,dataset]=_macro(
                dataset,
                base_score_col
            )

        for stage in range(1,n_stages+1):
            col=_score_col(stage,method)

            for dataset in eval_domains:
                perf.loc[stage,dataset]=_macro(
                    dataset,
                    col
                )

        full_performance[method]=perf

    # =========================================================
    # FIXED PARTITION-SPECIFIC PERFORMANCE
    #
    # Every task such as ba_1 is evaluated only on the usable
    # BA test alleles belonging to ba_1.
    # =========================================================
    partition_performance={}

    if allele_specific_tasks:
        for method in methods:
            task_perf={}

            for task in task_order:
                assay=_base_task(task)
                alleles=partition_test_alleles[task]

                values=np.full(
                    n_stages+1,
                    np.nan,
                    dtype=float
                )

                values[0]=_macro(
                    assay,
                    base_score_col,
                    alleles
                )

                for stage in range(1,n_stages+1):
                    values[stage]=_macro(
                        assay,
                        _score_col(stage,method),
                        alleles
                    )

                task_perf[task]=values

            partition_performance[method]=pd.DataFrame(
                task_perf,
                index=np.arange(n_stages+1)
            )

    # =========================================================
    # ASSAY-MERGED ALLELE-SPECIFIC PERFORMANCE
    #
    # At each stage:
    #   YD = average AP over all YD test alleles encountered so far
    #   EpiScan = same
    #   BA = same
    #   ESCAPE = same
    #
    # The number of assay domains therefore stays 4, not 16.
    # =========================================================
    assay_specific_performance={}

    if allele_specific_tasks and merge_partitions_by_assay:
        first_partition={
            assay:next(
                task for task in task_order
                if _base_task(task)==assay
            )
            for assay in assay_order
        }

        for method in methods:
            perf=pd.DataFrame(
                index=np.arange(n_stages+1),
                columns=eval_domains,
                dtype=float
            )

            # MS always evaluated on full MS test set.
            perf.loc[0,"ms"]=_macro(
                "ms",
                base_score_col
            )

            for stage in range(1,n_stages+1):
                perf.loc[stage,"ms"]=_macro(
                    "ms",
                    _score_col(stage,method)
                )

            # Base values use each assay's first available partition.
            for assay in assay_order:
                alleles=partition_test_alleles[
                    first_partition[assay]
                ]

                perf.loc[0,assay]=_macro(
                    assay,
                    base_score_col,
                    alleles
                )

            # Sequential stages.
            for stage in range(1,n_stages+1):
                col=_score_col(stage,method)

                for assay in assay_order:
                    alleles=acquired_alleles[stage][assay]

                    # Before an assay first appears, evaluate on its
                    # first partition purely to retain a defined fixed
                    # reference for plotting/all-domain calculations.
                    if len(alleles)==0:
                        alleles=partition_test_alleles[
                            first_partition[assay]
                        ]

                    perf.loc[stage,assay]=_macro(
                        assay,
                        col,
                        alleles
                    )

            assay_specific_performance[method]=perf

    # ---------------------------------------------------------
    # Choose representation for CL metrics.
    # ---------------------------------------------------------
    if not allele_specific_tasks:
        metric_performance=full_performance
        metric_task_mode="assay"

    elif merge_partitions_by_assay:
        metric_performance=assay_specific_performance
        metric_task_mode="assay"

    else:
        metric_performance=partition_performance
        metric_task_mode="partition"

    # =========================================================
    # CL METRICS
    # =========================================================
    metric_rows=[]

    for include_ms in (True,False):
        mode="ms_as_task" if include_ms else "assay_tasks_only"

        for method in methods:

            # =================================================
            # ASSAY-LEVEL METRICS
            # =================================================
            if metric_task_mode=="assay":
                perf=metric_performance[method]

                for stage in range(n_stages+1):
                    acquired_assays=[
                        assay for assay in assay_order
                        if acquisition_stage[assay] <= stage
                    ]

                    if include_ms:
                        seen=["ms"]+acquired_assays

                        previous=["ms"]+[
                            assay for assay in acquired_assays
                            if acquisition_stage[assay] < stage
                        ]

                        all_domains=["ms"]+assay_order

                    else:
                        seen=acquired_assays

                        previous=[
                            assay for assay in acquired_assays
                            if acquisition_stage[assay] < stage
                        ]

                        all_domains=assay_order

                    if not include_ms and stage==0:
                        seen_mean=perf.loc[
                            0,
                            assay_order[0]
                        ]

                    else:
                        seen_mean=(
                            perf.loc[stage,seen].mean()
                            if seen else np.nan
                        )

                    all_domain_mean=perf.loc[
                        stage,
                        all_domains
                    ].mean()

                    bwt_values=[]
                    forgetting_values=[]

                    for task in previous:
                        acquired=(
                            0 if task=="ms"
                            else acquisition_stage[task]
                        )

                        bwt_values.append(
                            perf.loc[stage,task] -
                            perf.loc[acquired,task]
                        )

                        historical_best=perf.loc[
                            acquired:stage-1,
                            task
                        ].max()

                        forgetting_values.append(
                            historical_best -
                            perf.loc[stage,task]
                        )

                    bwt=(
                        np.mean(bwt_values)
                        if bwt_values else np.nan
                    )

                    max_forgetting=(
                        np.max(forgetting_values)
                        if forgetting_values else np.nan
                    )

                    ms_retention=(
                        full_performance[method].loc[stage,"ms"] -
                        full_performance[method].loc[0,"ms"]
                    )

                    metric_rows.append({
                        "mode":mode,
                        "method":method,
                        "stage":stage,
                        "update":"Base" if stage==0 else task_order[stage-1],
                        "seen_task_mean":seen_mean,
                        "all_domain_mean":all_domain_mean,
                        "backward_transfer":bwt,
                        "maximum_forgetting":max_forgetting,
                        "ms_retention":ms_retention,
                    })

            # =================================================
            # PARTITION-AS-TASK METRICS
            # =================================================
            else:
                perf=partition_performance[method]

                for stage in range(n_stages+1):
                    seen_partitions=task_order[:stage]
                    previous_partitions=task_order[:max(stage-1,0)]

                    seen_values=[]

                    if include_ms:
                        seen_values.append(
                            full_performance[method].loc[
                                stage,
                                "ms"
                            ]
                        )

                    for task in seen_partitions:
                        seen_values.append(
                            perf.loc[stage,task]
                        )

                    if not include_ms and stage==0:
                        seen_mean=perf.loc[
                            0,
                            task_order[0]
                        ]

                    else:
                        seen_mean=(
                            np.mean(seen_values)
                            if seen_values else np.nan
                        )

                    # All partition domains, whether seen yet or not.
                    all_values=[]

                    if include_ms:
                        all_values.append(
                            full_performance[method].loc[
                                stage,
                                "ms"
                            ]
                        )

                    all_values.extend(
                        perf.loc[
                            stage,
                            task_order
                        ].to_numpy(float)
                    )

                    all_domain_mean=np.nanmean(
                        all_values
                    )

                    bwt_values=[]
                    forgetting_values=[]

                    # MS
                    if include_ms and stage>0:
                        bwt_values.append(
                            full_performance[method].loc[
                                stage,
                                "ms"
                            ] -
                            full_performance[method].loc[
                                0,
                                "ms"
                            ]
                        )

                        forgetting_values.append(
                            full_performance[method]
                            .loc[:stage-1,"ms"]
                            .max() -
                            full_performance[method]
                            .loc[stage,"ms"]
                        )

                    # Prior allele-partition tasks.
                    for task in previous_partitions:
                        acquired=task_order.index(task)+1

                        bwt_values.append(
                            perf.loc[stage,task] -
                            perf.loc[acquired,task]
                        )

                        historical_best=perf.loc[
                            acquired:stage-1,
                            task
                        ].max()

                        forgetting_values.append(
                            historical_best -
                            perf.loc[stage,task]
                        )

                    bwt=(
                        np.mean(bwt_values)
                        if bwt_values else np.nan
                    )

                    max_forgetting=(
                        np.max(forgetting_values)
                        if forgetting_values else np.nan
                    )

                    ms_retention=(
                        full_performance[method].loc[stage,"ms"] -
                        full_performance[method].loc[0,"ms"]
                    )

                    metric_rows.append({
                        "mode":mode,
                        "method":method,
                        "stage":stage,
                        "update":"Base" if stage==0 else task_order[stage-1],
                        "seen_task_mean":seen_mean,
                        "all_domain_mean":all_domain_mean,
                        "backward_transfer":bwt,
                        "maximum_forgetting":max_forgetting,
                        "ms_retention":ms_retention,
                    })

    metrics=pd.DataFrame(metric_rows)

    # ---------------------------------------------------------
    # X-axis
    # ---------------------------------------------------------
    x=np.arange(n_stages+1)
    xlabels=["Base"]+list(task_order)

    # =========================================================
    # RAW / ASSAY PERFORMANCE PLOTS
    # =========================================================
    if allele_specific_tasks and merge_partitions_by_assay:
        raw_plot_performance=assay_specific_performance
        raw_title="Long-horizon allele-specific assay performance"

    else:
        raw_plot_performance=full_performance
        raw_title="Long-horizon evaluation-domain performance"

    fig_raw,axes=plt.subplots(
        2,3,
        figsize=figsize_raw,
        dpi=dpi
    )
    axes=axes.ravel()

    for ax,dataset in zip(
        axes,
        eval_domains
    ):
        for method in methods:
            color=(
                method_colors.get(method)
                if method_colors else None
            )

            ax.plot(
                x,
                raw_plot_performance[method][dataset].to_numpy(),
                marker="o",
                markersize=markersize,
                linewidth=linewidth,
                label=method,
                color=color
            )

        if dataset!="ms":
            for stage,update_name in enumerate(
                task_order,
                start=1
            ):
                if _base_task(update_name)==dataset:
                    ax.axvline(
                        stage,
                        ls=":",
                        lw=0.8,
                        alpha=0.45
                    )

        ax.set_title(_label(dataset))
        ax.set_ylabel("mAP")
        ax.set_xlabel("Sequential update")

        ax.set_xticks(x)
        ax.set_xticklabels(
            xlabels,
            rotation=45,
            ha="right",
            fontsize=7
        )

    axes[-1].axis("off")

    handles,labels=axes[0].get_legend_handles_labels()

    axes[-1].legend(
        handles,
        labels,
        loc="center"
    )

    fig_raw.suptitle(raw_title)
    fig_raw.tight_layout()

    # =========================================================
    # FIVE CL METRIC PLOTS
    # =========================================================
    specs=[
        (
            "seen_task_mean",
            "Seen-task average performance",
            "mAP"
        ),
        (
            "all_domain_mean",
            "All-domain mean performance",
            "mAP"
        ),
        (
            "backward_transfer",
            "Backward transfer",
            "BWT"
        ),
        (
            "maximum_forgetting",
            "Maximum forgetting",
            "Forgetting"
        ),
        (
            "ms_retention",
            "MS retention",
            "ΔmAP from base MS"
        ),
    ]

    figures_metrics={}

    for mode,title in [
        (
            "ms_as_task",
            "MS included as a task"
        ),
        (
            "assay_tasks_only",
            "MS excluded from task sequence"
        ),
    ]:
        fig,axes=plt.subplots(
            2,3,
            figsize=figsize_metrics,
            dpi=dpi
        )
        axes=axes.ravel()

        for ax,(metric,metric_title,ylabel) in zip(
            axes,
            specs
        ):
            for method in methods:
                q=metrics[
                    (metrics["mode"]==mode) &
                    (metrics["method"]==method)
                ].sort_values("stage")

                color=(
                    method_colors.get(method)
                    if method_colors else None
                )

                ax.plot(
                    q["stage"],
                    q[metric],
                    marker="o",
                    markersize=markersize,
                    linewidth=linewidth,
                    label=method,
                    color=color
                )

            if metric in {
                "backward_transfer",
                "maximum_forgetting",
                "ms_retention",
            }:
                ax.axhline(
                    0,
                    ls="--",
                    lw=1
                )

            ax.set_title(metric_title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Sequential update")

            ax.set_xticks(x)
            ax.set_xticklabels(
                xlabels,
                rotation=45,
                ha="right",
                fontsize=7
            )

        axes[-1].axis("off")

        handles,labels=axes[0].get_legend_handles_labels()

        axes[-1].legend(
            handles,
            labels,
            loc="center"
        )

        suffix=(
            "allele-specific, assay-merged"
            if allele_specific_tasks and merge_partitions_by_assay
            else "allele-specific partitions"
            if allele_specific_tasks
            else "full test sets"
        )

        fig.suptitle(
            f"Long-horizon continual-learning metrics - "
            f"{title} ({suffix})"
        )

        fig.tight_layout()

        figures_metrics[mode]=fig

    if show:
        plt.show()

    return {
        "full_performance":full_performance,
        "partition_performance":partition_performance,
        "assay_specific_performance":assay_specific_performance,
        "metrics":metrics,
        "acquisition_stage":acquisition_stage,
        "partition_test_alleles":partition_test_alleles,
        "acquired_alleles":acquired_alleles,
        "figures":{
            "raw_performance":fig_raw,
            **figures_metrics,
        },
    }

import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def plot_long_horizon_cl_metrics_v2(
    allele_ap_dict, task_order, method_suffixes,
    task_order_name="t1_allele_partition", base_score_col="mhcprime",
    partition_summary=None, allele_specific_tasks=False,
    merge_partitions_by_assay=True,
    allele_col="allele",

    # What to plot.
    plot_type="metrics",  # {"metrics","assay_performance"}
    include_ms_as_task=False,
    include_ms_panel=True,
    panel_order=None,

    # Labels / colors.
    method_labels=None,
    method_colors=None,
    dataset_labels=None,
    task_prefix_labels=None,
    task_tick_separator="",

    # Figure.
    figsize=(16,7),
    dpi=300,
    font_family="Nimbus Sans",

    # Lines / markers.
    linewidth=1.5,
    marker="o",
    markersize=5,
    marker_edgewidth=0,

    # Axis text.
    xlabel="Sequential update stage",
    xlabel_fontsize=None,
    ylabel_fontsize=None,
    title_fontsize=None,
    xtick_fontsize=7,
    ytick_fontsize=None,
    xtick_rotation=45,
    xtick_ha=None,

    # Spines.
    remove_top_spine=True,
    remove_right_spine=True,

    # Vertical separators.
    show_cycle_lines=False,
    cycle_line_positions=None,
    cycle_line_color="0.5",
    cycle_line_style=":",
    cycle_linewidth=0.8,
    cycle_line_alpha=0.45,

    # Original assay-specific update lines.
    show_assay_update_lines=True,
    assay_update_line_color="0.5",
    assay_update_line_style=":",
    assay_update_linewidth=0.8,
    assay_update_line_alpha=0.45,

    # Zero references for CL metrics.
    show_zero_lines=True,
    zero_line_color="black",
    zero_line_style="--",
    zero_linewidth=0.8,

    # Legend.
    show_legend=True,
    legend_loc="center",
    legend_ncol=1,
    legend_frame=False,
    legend_fontsize=None,

    # Grid geometry.
    wspace=0.30,
    hspace=0.35,
    left_margin=None,
    right_margin=None,
    bottom_margin=None,
    top_margin=None,

    # Saving.
    save_path=None,
    save_pdf=True,
    save_png=True,
    bbox_inches="tight",
    pad_inches=0.05,

    verbose=True,
    show=True,
):
    if plot_type not in {"metrics","assay_performance"}:
        raise ValueError("plot_type must be 'metrics' or 'assay_performance'.")

    if allele_specific_tasks and partition_summary is None:
        raise ValueError(
            "partition_summary must be supplied when allele_specific_tasks=True."
        )

    # -----------------------------------------------------------------
    # Publication/export settings.
    # -----------------------------------------------------------------
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,"Nimbus Sans","Nimbus Sans L",
        "Arial","Helvetica","DejaVu Sans",
    ]

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)
    task_prefix_labels={} if task_prefix_labels is None else dict(task_prefix_labels)
    method_labels={} if method_labels is None else dict(method_labels)
    methods=list(method_suffixes)

    def _base_task(name):
        return re.sub(r"_\d+$","",name)

    def _label(name):
        return dataset_labels.get(name,name)

    def _task_tick_label(name):
        m=re.match(r"^(.*?)(?:_(\d+))?$",str(name))
        base=m.group(1)
        number=m.group(2)
        prefix=task_prefix_labels.get(base,base)
        return (
            f"{prefix}{task_tick_separator}{number}"
            if number is not None else str(prefix)
        )

    assay_order=list(dict.fromkeys(_base_task(x) for x in task_order))
    eval_domains=["ms"]+assay_order
    n_stages=len(task_order)

    missing=[d for d in eval_domains if d not in allele_ap_dict]
    if missing:
        raise ValueError(f"Missing evaluation datasets: {missing}")

    # -----------------------------------------------------------------
    # Validate AP frames.
    # -----------------------------------------------------------------
    numeric_frames={}

    for dataset in eval_domains:
        df=allele_ap_dict[dataset]

        if allele_col not in df.columns:
            raise ValueError(
                f"{dataset}: missing allele column {allele_col!r}."
            )

        numeric=df.select_dtypes(include=[np.number]).copy()

        if base_score_col not in numeric.columns:
            raise ValueError(
                f"{dataset}: base score column {base_score_col!r} "
                "is missing or nonnumeric."
            )

        numeric_frames[dataset]=numeric

    def _score_col(stage,method):
        return f"{task_order_name}_s{stage}{method_suffixes[method]}"

    for dataset in eval_domains:
        numeric=numeric_frames[dataset]

        for method in methods:
            missing_cols=[
                _score_col(stage,method)
                for stage in range(1,n_stages+1)
                if _score_col(stage,method) not in numeric.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"{dataset}: missing {len(missing_cols)} score columns "
                    f"for {method!r}. First missing: {missing_cols[:5]}"
                )

    # -----------------------------------------------------------------
    # Determine usable TEST alleles for every partition.
    # -----------------------------------------------------------------
    partition_test_alleles={}

    if allele_specific_tasks:
        missing_summary=[
            task for task in task_order
            if task not in partition_summary
        ]

        if missing_summary:
            raise ValueError(
                f"partition_summary is missing tasks: {missing_summary}"
            )

        for task in task_order:
            assay=_base_task(task)

            requested=[
                str(a) for a in partition_summary[task]["alleles"]
            ]

            available_ordered=(
                allele_ap_dict[assay][allele_col]
                .astype(str)
                .tolist()
            )

            available_set=set(available_ordered)
            requested_set=set(requested)

            usable=[
                a for a in requested
                if a in available_set
            ]

            missing_alleles=sorted(
                requested_set-available_set
            )

            if len(usable)==0:
                raise ValueError(
                    f"{task}: none of its {len(requested)} partition alleles "
                    f"are present in the {assay!r} test AP dataframe."
                )

            partition_test_alleles[task]=usable

            if verbose:
                print(
                    f"{task}: using {len(usable)}/{len(requested)} "
                    "partition alleles present in test"
                )

                if missing_alleles:
                    print(
                        f"  ignoring {len(missing_alleles)} train-only alleles: "
                        f"{missing_alleles[:10]}"
                        +(" ..." if len(missing_alleles)>10 else "")
                    )

    # -----------------------------------------------------------------
    # Macro AP helper.
    # -----------------------------------------------------------------
    def _macro(dataset,col,alleles=None):
        df=allele_ap_dict[dataset]

        if col not in df.columns:
            raise ValueError(
                f"{dataset}: missing AP column {col!r}."
            )

        if alleles is None:
            values=df[col]

        else:
            allele_set={str(a) for a in alleles}
            mask=df[allele_col].astype(str).isin(allele_set)

            if mask.sum()==0:
                raise ValueError(
                    f"{dataset}: none of the requested alleles are "
                    "present in the AP dataframe."
                )

            values=df.loc[mask,col]

        return float(values.mean(skipna=True))

    # -----------------------------------------------------------------
    # First biological-assay acquisition stage.
    # -----------------------------------------------------------------
    acquisition_stage={}

    for stage,task in enumerate(task_order,start=1):
        assay=_base_task(task)

        if assay not in acquisition_stage:
            acquisition_stage[assay]=stage

    # -----------------------------------------------------------------
    # Accumulated AVAILABLE test alleles by assay and stage.
    # -----------------------------------------------------------------
    acquired_alleles={stage:{} for stage in range(n_stages+1)}

    if allele_specific_tasks:
        running={assay:[] for assay in assay_order}

        acquired_alleles[0]={
            assay:[] for assay in assay_order
        }

        for stage,task in enumerate(task_order,start=1):
            assay=_base_task(task)
            existing=set(running[assay])

            for allele in partition_test_alleles[task]:
                if allele not in existing:
                    running[assay].append(allele)
                    existing.add(allele)

            acquired_alleles[stage]={
                assay:list(vals)
                for assay,vals in running.items()
            }

    # ================================================================
    # FULL TEST-SET PERFORMANCE
    # ================================================================
    full_performance={}

    for method in methods:
        perf=pd.DataFrame(
            index=np.arange(n_stages+1),
            columns=eval_domains,
            dtype=float,
        )

        for dataset in eval_domains:
            perf.loc[0,dataset]=_macro(
                dataset,
                base_score_col,
            )

        for stage in range(1,n_stages+1):
            col=_score_col(stage,method)

            for dataset in eval_domains:
                perf.loc[stage,dataset]=_macro(
                    dataset,
                    col,
                )

        full_performance[method]=perf

    # ================================================================
    # FIXED PARTITION-SPECIFIC PERFORMANCE
    # ================================================================
    partition_performance={}

    if allele_specific_tasks:
        for method in methods:
            task_perf={}

            for task in task_order:
                assay=_base_task(task)
                alleles=partition_test_alleles[task]

                values=np.full(
                    n_stages+1,
                    np.nan,
                    dtype=float,
                )

                values[0]=_macro(
                    assay,
                    base_score_col,
                    alleles,
                )

                for stage in range(1,n_stages+1):
                    values[stage]=_macro(
                        assay,
                        _score_col(stage,method),
                        alleles,
                    )

                task_perf[task]=values

            partition_performance[method]=pd.DataFrame(
                task_perf,
                index=np.arange(n_stages+1),
            )

    # ================================================================
    # ASSAY-MERGED ALLELE-SPECIFIC PERFORMANCE
    # ================================================================
    assay_specific_performance={}

    if allele_specific_tasks and merge_partitions_by_assay:
        first_partition={
            assay:next(
                task for task in task_order
                if _base_task(task)==assay
            )
            for assay in assay_order
        }

        for method in methods:
            perf=pd.DataFrame(
                index=np.arange(n_stages+1),
                columns=eval_domains,
                dtype=float,
            )

            # MS always evaluated on full MS test set.
            perf.loc[0,"ms"]=_macro(
                "ms",
                base_score_col,
            )

            for stage in range(1,n_stages+1):
                perf.loc[stage,"ms"]=_macro(
                    "ms",
                    _score_col(stage,method),
                )

            # Base values use each assay's first available partition.
            for assay in assay_order:
                alleles=partition_test_alleles[
                    first_partition[assay]
                ]

                perf.loc[0,assay]=_macro(
                    assay,
                    base_score_col,
                    alleles,
                )

            # Sequential stages.
            for stage in range(1,n_stages+1):
                col=_score_col(stage,method)

                for assay in assay_order:
                    alleles=acquired_alleles[stage][assay]

                    if len(alleles)==0:
                        alleles=partition_test_alleles[
                            first_partition[assay]
                        ]

                    perf.loc[stage,assay]=_macro(
                        assay,
                        col,
                        alleles,
                    )

            assay_specific_performance[method]=perf

    # -----------------------------------------------------------------
    # Choose representation for CL metrics.
    # -----------------------------------------------------------------
    if not allele_specific_tasks:
        metric_performance=full_performance
        metric_task_mode="assay"

    elif merge_partitions_by_assay:
        metric_performance=assay_specific_performance
        metric_task_mode="assay"

    else:
        metric_performance=partition_performance
        metric_task_mode="partition"

    # ================================================================
    # CL METRICS
    # ================================================================
    metric_rows=[]

    for include_ms in (True,False):
        mode="ms_as_task" if include_ms else "assay_tasks_only"

        for method in methods:

            # =========================================================
            # ASSAY-LEVEL METRICS
            # =========================================================
            if metric_task_mode=="assay":
                perf=metric_performance[method]

                for stage in range(n_stages+1):
                    acquired_assays=[
                        assay for assay in assay_order
                        if acquisition_stage[assay]<=stage
                    ]

                    if include_ms:
                        seen=["ms"]+acquired_assays

                        previous=["ms"]+[
                            assay for assay in acquired_assays
                            if acquisition_stage[assay]<stage
                        ]

                        all_domains=["ms"]+assay_order

                    else:
                        seen=acquired_assays

                        previous=[
                            assay for assay in acquired_assays
                            if acquisition_stage[assay]<stage
                        ]

                        all_domains=assay_order

                    if not include_ms and stage==0:
                        seen_mean=perf.loc[
                            0,
                            assay_order[0],
                        ]

                    else:
                        seen_mean=(
                            perf.loc[stage,seen].mean()
                            if seen else np.nan
                        )

                    all_domain_mean=perf.loc[
                        stage,
                        all_domains,
                    ].mean()

                    bwt_values=[]
                    forgetting_values=[]

                    for task in previous:
                        acquired=(
                            0 if task=="ms"
                            else acquisition_stage[task]
                        )

                        bwt_values.append(
                            perf.loc[stage,task]-
                            perf.loc[acquired,task]
                        )

                        historical_best=perf.loc[
                            acquired:stage-1,
                            task,
                        ].max()

                        forgetting_values.append(
                            historical_best-
                            perf.loc[stage,task]
                        )

                    bwt=(
                        np.mean(bwt_values)
                        if bwt_values else np.nan
                    )

                    max_forgetting=(
                        np.max(forgetting_values)
                        if forgetting_values else np.nan
                    )

                    ms_retention=(
                        full_performance[method].loc[stage,"ms"]-
                        full_performance[method].loc[0,"ms"]
                    )

                    metric_rows.append({
                        "mode":mode,
                        "method":method,
                        "stage":stage,
                        "update":"Base" if stage==0 else task_order[stage-1],
                        "seen_task_mean":seen_mean,
                        "all_domain_mean":all_domain_mean,
                        "backward_transfer":bwt,
                        "maximum_forgetting":max_forgetting,
                        "ms_retention":ms_retention,
                    })

            # =========================================================
            # PARTITION-AS-TASK METRICS
            # =========================================================
            else:
                perf=partition_performance[method]

                for stage in range(n_stages+1):
                    seen_partitions=task_order[:stage]
                    previous_partitions=task_order[:max(stage-1,0)]

                    seen_values=[]

                    if include_ms:
                        seen_values.append(
                            full_performance[method].loc[
                                stage,
                                "ms",
                            ]
                        )

                    for task in seen_partitions:
                        seen_values.append(
                            perf.loc[stage,task]
                        )

                    if not include_ms and stage==0:
                        seen_mean=perf.loc[
                            0,
                            task_order[0],
                        ]

                    else:
                        seen_mean=(
                            np.mean(seen_values)
                            if seen_values else np.nan
                        )

                    all_values=[]

                    if include_ms:
                        all_values.append(
                            full_performance[method].loc[
                                stage,
                                "ms",
                            ]
                        )

                    all_values.extend(
                        perf.loc[
                            stage,
                            task_order,
                        ].to_numpy(float)
                    )

                    all_domain_mean=np.nanmean(
                        all_values
                    )

                    bwt_values=[]
                    forgetting_values=[]

                    if include_ms and stage>0:
                        bwt_values.append(
                            full_performance[method].loc[
                                stage,
                                "ms",
                            ]-
                            full_performance[method].loc[
                                0,
                                "ms",
                            ]
                        )

                        forgetting_values.append(
                            full_performance[method]
                            .loc[:stage-1,"ms"]
                            .max()-
                            full_performance[method]
                            .loc[stage,"ms"]
                        )

                    for task in previous_partitions:
                        acquired=task_order.index(task)+1

                        bwt_values.append(
                            perf.loc[stage,task]-
                            perf.loc[acquired,task]
                        )

                        historical_best=perf.loc[
                            acquired:stage-1,
                            task,
                        ].max()

                        forgetting_values.append(
                            historical_best-
                            perf.loc[stage,task]
                        )

                    bwt=(
                        np.mean(bwt_values)
                        if bwt_values else np.nan
                    )

                    max_forgetting=(
                        np.max(forgetting_values)
                        if forgetting_values else np.nan
                    )

                    ms_retention=(
                        full_performance[method].loc[stage,"ms"]-
                        full_performance[method].loc[0,"ms"]
                    )

                    metric_rows.append({
                        "mode":mode,
                        "method":method,
                        "stage":stage,
                        "update":"Base" if stage==0 else task_order[stage-1],
                        "seen_task_mean":seen_mean,
                        "all_domain_mean":all_domain_mean,
                        "backward_transfer":bwt,
                        "maximum_forgetting":max_forgetting,
                        "ms_retention":ms_retention,
                    })

    metrics=pd.DataFrame(metric_rows)

    # ================================================================
    # X AXIS
    # ================================================================
    x=np.arange(n_stages+1)

    xlabels=[
        "Base",
        *[_task_tick_label(task) for task in task_order],
    ]

    if xtick_ha is None:
        tick_ha="center" if xtick_rotation==0 else "right"
    else:
        tick_ha=xtick_ha

    # -----------------------------------------------------------------
    # Automatically determine cycle boundaries.
    #
    # For yd_1, episcan_1, ..., escape_1, yd_2, ... this gives
    # boundaries at 4.5, 8.5, ...
    # -----------------------------------------------------------------
    if cycle_line_positions is None:
        cycle_ids=[]

        for task in task_order:
            m=re.search(r"_(\d+)$",str(task))
            cycle_ids.append(int(m.group(1)) if m else None)

        auto_cycle_lines=[]

        if all(c is not None for c in cycle_ids):
            for stage in range(2,n_stages+1):
                if cycle_ids[stage-1]!=cycle_ids[stage-2]:
                    auto_cycle_lines.append(stage-0.5)

        else:
            first_assay=_base_task(task_order[0])

            for stage,task in enumerate(task_order[1:],start=2):
                if _base_task(task)==first_assay:
                    auto_cycle_lines.append(stage-0.5)

        cycle_positions=auto_cycle_lines

    else:
        cycle_positions=list(cycle_line_positions)

    # -----------------------------------------------------------------
    # Styling helpers.
    # -----------------------------------------------------------------
    default_colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]

    colors={
        method:(
            method_colors[method]
            if method_colors is not None and method in method_colors
            else default_colors[i%len(default_colors)]
        )
        for i,method in enumerate(methods)
    }

    def _style_axis(ax):
        ax.spines["top"].set_visible(not remove_top_spine)
        ax.spines["right"].set_visible(not remove_right_spine)

        if xlabel_fontsize is not None:
            ax.xaxis.label.set_size(xlabel_fontsize)

        if ylabel_fontsize is not None:
            ax.yaxis.label.set_size(ylabel_fontsize)

        if title_fontsize is not None:
            ax.title.set_size(title_fontsize)

        if ytick_fontsize is not None:
            ax.tick_params(axis="y",labelsize=ytick_fontsize)

    def _draw_cycle_lines(ax):
        if show_cycle_lines:
            for xpos in cycle_positions:
                ax.axvline(
                    xpos,
                    color=cycle_line_color,
                    linestyle=cycle_line_style,
                    linewidth=cycle_linewidth,
                    alpha=cycle_line_alpha,
                    zorder=0,
                )

    # ================================================================
    # PANEL DEFINITIONS
    # ================================================================
    metric_specs={
        "seen_task_mean":(
            "Average performance",
            "mAP",
        ),
        "ms_retention":(
            "MS retention",
            "ΔmAP from base MS",
        ),
        "all_domain_mean":(
            "All domain mean performance",
            "mAP",
        ),
        "backward_transfer":(
            "Backward transfer",
            "BWT",
        ),
        "maximum_forgetting":(
            "Maximum forgetting",
            "Forgetting",
        ),
    }

    # ================================================================
    # ASSAY PERFORMANCE FIGURE
    # ================================================================
    if plot_type=="assay_performance":

        if allele_specific_tasks and merge_partitions_by_assay:
            plot_performance=assay_specific_performance
        else:
            plot_performance=full_performance

        default_panels=list(eval_domains)

        if not include_ms_panel:
            default_panels=[
                d for d in default_panels
                if d!="ms"
            ]

        panels=default_panels if panel_order is None else list(panel_order)

        invalid=[
            p for p in panels
            if p not in eval_domains
        ]

        if invalid:
            raise ValueError(
                f"Invalid assay performance panels: {invalid}"
            )

        if not include_ms_panel and "ms" in panels:
            raise ValueError(
                "panel_order contains 'ms' but include_ms_panel=False."
            )

        fig,axes=plt.subplots(
            2,3,
            figsize=figsize,
            dpi=dpi,
        )

        axes=np.asarray(axes).ravel()

        for ax,dataset in zip(axes,panels):

            for method in methods:
                ax.plot(
                    x,
                    plot_performance[method][dataset].to_numpy(),
                    marker=marker,
                    markersize=markersize,
                    markeredgewidth=marker_edgewidth,
                    linewidth=linewidth,
                    label=method_labels.get(method,method),
                    color=colors[method],
                )

            # Preserve original assay-specific update markers.
            if show_assay_update_lines and dataset!="ms":
                for stage,update_name in enumerate(
                    task_order,
                    start=1,
                ):
                    if _base_task(update_name)==dataset:
                        ax.axvline(
                            stage,
                            color=assay_update_line_color,
                            linestyle=assay_update_line_style,
                            linewidth=assay_update_linewidth,
                            alpha=assay_update_line_alpha,
                            zorder=0,
                        )

            # Optional cycle boundaries, including MS panel.
            _draw_cycle_lines(ax)

            ax.set_title(_label(dataset))
            ax.set_ylabel("mAP")
            ax.set_xlabel(xlabel)

            ax.set_xticks(x)
            ax.set_xticklabels(
                xlabels,
                rotation=xtick_rotation,
                ha=tick_ha,
                fontsize=xtick_fontsize,
            )

            _style_axis(ax)

    # ================================================================
    # CONTINUAL LEARNING METRICS FIGURE
    # ================================================================
    else:

        mode=(
            "ms_as_task"
            if include_ms_as_task
            else "assay_tasks_only"
        )

        default_panels=[
            "seen_task_mean",
            "ms_retention",
            "all_domain_mean",
            "backward_transfer",
            "maximum_forgetting",
        ]

        panels=default_panels if panel_order is None else list(panel_order)

        invalid=[
            p for p in panels
            if p not in metric_specs
        ]

        if invalid:
            raise ValueError(
                f"Invalid metric panels: {invalid}. "
                f"Valid metrics: {list(metric_specs)}"
            )

        fig,axes=plt.subplots(
            2,3,
            figsize=figsize,
            dpi=dpi,
        )

        axes=np.asarray(axes).ravel()

        for ax,metric in zip(axes,panels):
            title,ylabel=metric_specs[metric]

            for method in methods:
                q=metrics[
                    (metrics["mode"]==mode)&
                    (metrics["method"]==method)
                ].sort_values("stage")

                ax.plot(
                    q["stage"],
                    q[metric],
                    marker=marker,
                    markersize=markersize,
                    markeredgewidth=marker_edgewidth,
                    linewidth=linewidth,
                    label=method_labels.get(method,method),
                    color=colors[method],
                )

            if show_zero_lines and metric in {
                "backward_transfer",
                "maximum_forgetting",
                "ms_retention",
            }:
                ax.axhline(
                    0,
                    color=zero_line_color,
                    linestyle=zero_line_style,
                    linewidth=zero_linewidth,
                    zorder=0,
                )

            _draw_cycle_lines(ax)

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            ax.set_xticks(x)
            ax.set_xticklabels(
                xlabels,
                rotation=xtick_rotation,
                ha=tick_ha,
                fontsize=xtick_fontsize,
            )

            _style_axis(ax)

    # ================================================================
    # LEGEND / EMPTY PANELS
    # ================================================================
    used=len(panels)

    if show_legend and used<len(axes):
        legend_ax=axes[used]
        legend_ax.axis("off")

        handles,labels=axes[0].get_legend_handles_labels()

        legend_ax.legend(
            handles,
            labels,
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=legend_frame,
            fontsize=legend_fontsize,
        )

        start_off=used+1

    else:
        start_off=used

        if show_legend:
            handles,labels=axes[0].get_legend_handles_labels()

            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=legend_ncol,
                frameon=legend_frame,
                fontsize=legend_fontsize,
            )

    for ax in axes[start_off:]:
        ax.axis("off")

    # ================================================================
    # LAYOUT
    # ================================================================
    fig.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
        **(
            {"left":left_margin}
            if left_margin is not None else {}
        ),
        **(
            {"right":right_margin}
            if right_margin is not None else {}
        ),
        **(
            {"bottom":bottom_margin}
            if bottom_margin is not None else {}
        ),
        **(
            {"top":top_margin}
            if top_margin is not None else {}
        ),
    )

    # ================================================================
    # SAVE
    # ================================================================
    if save_path is not None:

        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

    if show:
        plt.show()

    return {
        "full_performance":full_performance,
        "partition_performance":partition_performance,
        "assay_specific_performance":assay_specific_performance,
        "metrics":metrics,
        "acquisition_stage":acquisition_stage,
        "partition_test_alleles":partition_test_alleles,
        "acquired_alleles":acquired_alleles,
        "figure":fig,
        "axes":axes,
        "plot_type":plot_type,
        "panel_order":panels,
        "include_ms_as_task":include_ms_as_task,
    }

import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


def plot_long_horizon_cl_metrics_v2_multi_seeded(
    allele_ap_dict, task_order, method_suffixes,
    allele_partitions_by_seed,
    task_order_name="t1_allele_partition",
    base_score_col="mhcprime",
    allele_specific_tasks=False,
    merge_partitions_by_assay=True,
    allele_col="allele",
    seed_tag="seed",

    # NEW: evaluation / forgetting options.
    evaluate_full=False,
    show_forgetting_seed_lines=False,
    use_mean_forgetting=False,
    seed_line_alpha=0.15,
    seed_linewidth=0.8,

    # Confidence intervals.
    ci=0.95,
    show_ci=True,

    # What to plot.
    plot_type="metrics",  # {"metrics","assay_performance"}
    include_ms_as_task=False,
    include_ms_panel=True,
    panel_order=None,

    # Labels / colors.
    method_labels=None,
    method_colors=None,
    dataset_labels=None,
    task_prefix_labels=None,
    task_tick_separator="",

    # Figure.
    figsize=(16,7),
    dpi=300,
    font_family="Nimbus Sans",

    # Mean lines / markers / confidence intervals.
    linewidth=1.5,
    marker="o",
    markersize=5,
    marker_edgewidth=0,
    error_linewidth=1.0,
    error_capsize=3,
    error_capthick=1.0,

    # Axis text.
    xlabel="Sequential update stage",
    xlabel_fontsize=None,
    ylabel_fontsize=None,
    title_fontsize=None,
    xtick_fontsize=7,
    ytick_fontsize=None,
    xtick_rotation=45,
    xtick_ha=None,

    # Spines.
    remove_top_spine=True,
    remove_right_spine=True,

    # Vertical separators.
    show_cycle_lines=False,
    cycle_line_positions=None,
    cycle_line_color="0.5",
    cycle_line_style=":",
    cycle_linewidth=0.8,
    cycle_line_alpha=0.45,

    # Assay-specific update lines.
    show_assay_update_lines=True,
    assay_update_line_color="0.5",
    assay_update_line_style=":",
    assay_update_linewidth=0.8,
    assay_update_line_alpha=0.45,

    # Zero references.
    show_zero_lines=True,
    zero_line_color="black",
    zero_line_style="--",
    zero_linewidth=0.8,

    # Legend.
    show_legend=True,
    legend_loc="center",
    legend_ncol=1,
    legend_frame=False,
    legend_fontsize=None,

    # Grid geometry.
    wspace=0.30,
    hspace=0.35,
    left_margin=None,
    right_margin=None,
    bottom_margin=None,
    top_margin=None,

    # Saving.
    save_path=None,
    save_pdf=True,
    save_png=True,
    bbox_inches="tight",
    pad_inches=0.05,

    verbose=True,
    show=True,
):
    """
    Plot multi-seed long-horizon continual-learning performance.

    Important aggregation order
    ---------------------------
    1. Compute per-allele AP upstream.
    2. Within each seed/partition replicate, macro-average the appropriate
       allele set for each assay/stage.
    3. Compute CL metrics independently within each seed.
    4. Aggregate completed seed-level metrics across seeds.
    5. Confidence intervals are Student-t intervals across seed-level values.

    New options
    -----------
    evaluate_full : bool, default False
        False preserves the existing evaluation logic.

        True evaluates each assay on its full fixed test allele set at every
        stage, even when allele_specific_tasks=True. Training partitions still
        differ across seeds; only the evaluation population is fixed.

    show_forgetting_seed_lines : bool, default False
        If True, the forgetting panel shows the across-seed mean line plus
        faint individual seed trajectories and suppresses the CI bars for
        that panel only.

    use_mean_forgetting : bool, default False
        False plots maximum forgetting.

        True replaces the maximum-forgetting panel with mean forgetting.
        Mean forgetting is the mean positive decline from each previous
        task's historical best. Improvements contribute zero.
    """
    if plot_type not in {"metrics","assay_performance"}:
        raise ValueError("plot_type must be 'metrics' or 'assay_performance'.")

    if not 0<ci<1:
        raise ValueError("ci must be between 0 and 1.")

    if not allele_partitions_by_seed:
        raise ValueError("allele_partitions_by_seed is empty.")

    if not 0<=seed_line_alpha<=1:
        raise ValueError("seed_line_alpha must be between 0 and 1.")

    # -----------------------------------------------------------------
    # Publication/export settings.
    # -----------------------------------------------------------------
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"
    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,"Nimbus Sans","Nimbus Sans L",
        "Arial","Helvetica","DejaVu Sans",
    ]

    dataset_labels={} if dataset_labels is None else dict(dataset_labels)
    task_prefix_labels={} if task_prefix_labels is None else dict(task_prefix_labels)
    method_labels={} if method_labels is None else dict(method_labels)

    methods=list(method_suffixes)
    seeds=list(allele_partitions_by_seed)

    def _base_task(name):
        return re.sub(r"_\d+$","",name)

    def _label(name):
        return dataset_labels.get(name,name)

    def _task_tick_label(name):
        m=re.match(r"^(.*?)(?:_(\d+))?$",str(name))
        base=m.group(1)
        number=m.group(2)
        prefix=task_prefix_labels.get(base,base)
        return (
            f"{prefix}{task_tick_separator}{number}"
            if number is not None else str(prefix)
        )

    # -----------------------------------------------------------------
    # Extract allele partition from the structures used in this analysis.
    #
    # Supports:
    #   seed -> task -> DataFrame
    #   seed -> task -> {"alleles":[...]}
    #   seed -> task -> list / Series / array
    # -----------------------------------------------------------------
    def _partition_alleles(seed,task):
        x=allele_partitions_by_seed[seed][task]

        if isinstance(x,pd.DataFrame):
            if allele_col not in x.columns:
                raise ValueError(
                    f"Seed {seed}, task {task}: partition dataframe "
                    f"is missing allele column {allele_col!r}."
                )
            return (
                x[allele_col]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

        if isinstance(x,dict):
            if "alleles" not in x:
                raise ValueError(
                    f"Seed {seed}, task {task}: partition dictionary "
                    "must contain an 'alleles' key."
                )
            x=x["alleles"]

        return list(dict.fromkeys(str(a) for a in x))

    def _first_col(df,name):
        pos=np.flatnonzero(df.columns.to_numpy(dtype=object)==name)

        if len(pos)==0:
            raise ValueError(f"Missing column {name!r}.")

        return df.iloc[:,pos[0]]

    def _score_col(stage,method,seed):
        return (
            f"{task_order_name}_s{stage}"
            f"{method_suffixes[method]}"
            f"_{seed_tag}_{seed}"
        )

    def _mean_ci(vals):
        x=np.asarray(vals,float)
        x=x[np.isfinite(x)]
        n=len(x)

        if n==0:
            return np.nan,np.nan,np.nan,0

        mean=float(x.mean())

        if n<2:
            return mean,np.nan,np.nan,n

        se=x.std(ddof=1)/np.sqrt(n)
        crit=student_t.ppf((1+ci)/2,n-1)

        return mean,mean-crit*se,mean+crit*se,n

    assay_order=list(dict.fromkeys(_base_task(x) for x in task_order))
    eval_domains=["ms"]+assay_order
    n_stages=len(task_order)

    missing=[d for d in eval_domains if d not in allele_ap_dict]
    if missing:
        raise ValueError(f"Missing evaluation datasets: {missing}")

    for seed in seeds:
        missing_tasks=[
            task for task in task_order
            if task not in allele_partitions_by_seed[seed]
        ]

        if missing_tasks:
            raise ValueError(
                f"Seed {seed}: partition dictionary missing tasks: "
                f"{missing_tasks}"
            )

    # -----------------------------------------------------------------
    # Validate expected AP columns.
    # -----------------------------------------------------------------
    for dataset in eval_domains:
        df=allele_ap_dict[dataset]

        if allele_col not in df.columns:
            raise ValueError(
                f"{dataset}: missing allele column {allele_col!r}."
            )

        if base_score_col not in df.columns:
            raise ValueError(
                f"{dataset}: missing base score column {base_score_col!r}."
            )

        for method in methods:
            missing_cols=[
                _score_col(stage,method,seed)
                for seed in seeds
                for stage in range(1,n_stages+1)
                if _score_col(stage,method,seed) not in df.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"{dataset}: missing {len(missing_cols)} score "
                    f"columns for {method!r}. "
                    f"First missing: {missing_cols[:5]}"
                )

    # -----------------------------------------------------------------
    # Determine usable test alleles for each seed-specific partition.
    # -----------------------------------------------------------------
    partition_test_alleles={seed:{} for seed in seeds}

    if allele_specific_tasks:
        for seed in seeds:
            for task in task_order:
                assay=_base_task(task)
                requested=_partition_alleles(seed,task)

                available_ordered=(
                    allele_ap_dict[assay][allele_col]
                    .astype(str)
                    .tolist()
                )

                available_set=set(available_ordered)
                requested_set=set(requested)

                usable=[
                    allele for allele in requested
                    if allele in available_set
                ]

                missing_alleles=sorted(
                    requested_set-available_set
                )

                if len(usable)==0:
                    raise ValueError(
                        f"Seed {seed}, {task}: none of its "
                        f"{len(requested)} partition alleles are present "
                        f"in the {assay!r} test AP dataframe."
                    )

                partition_test_alleles[seed][task]=usable

                if verbose:
                    print(
                        f"Seed {seed} | {task}: using "
                        f"{len(usable)}/{len(requested)} "
                        "partition alleles present in test"
                    )

                    if missing_alleles:
                        print(
                            f"  ignoring {len(missing_alleles)} "
                            f"train-only alleles: {missing_alleles[:10]}"
                            +(" ..." if len(missing_alleles)>10 else "")
                        )

    # -----------------------------------------------------------------
    # Macro AP helper.
    # -----------------------------------------------------------------
    def _macro(dataset,col,alleles=None):
        df=allele_ap_dict[dataset]

        if col==base_score_col:
            values=_first_col(df,col)

        else:
            pos=np.flatnonzero(
                df.columns.to_numpy(dtype=object)==col
            )

            if len(pos)!=1:
                raise ValueError(
                    f"{dataset}: expected exactly one AP column "
                    f"{col!r}, found {len(pos)}."
                )

            values=df.iloc[:,pos[0]]

        if alleles is not None:
            allele_set={str(a) for a in alleles}

            mask=(
                df[allele_col]
                .astype(str)
                .isin(allele_set)
            )

            if mask.sum()==0:
                raise ValueError(
                    f"{dataset}: none of the requested alleles "
                    "are present in the AP dataframe."
                )

            values=values.loc[mask]

        return float(values.mean(skipna=True))

    # -----------------------------------------------------------------
    # First acquisition stage for each biological assay.
    # -----------------------------------------------------------------
    acquisition_stage={}

    for stage,task in enumerate(task_order,start=1):
        assay=_base_task(task)

        if assay not in acquisition_stage:
            acquisition_stage[assay]=stage

    # -----------------------------------------------------------------
    # Accumulated acquired allele sets within each seed.
    # -----------------------------------------------------------------
    acquired_alleles={
        seed:{
            stage:{}
            for stage in range(n_stages+1)
        }
        for seed in seeds
    }

    if allele_specific_tasks:
        for seed in seeds:
            running={
                assay:[]
                for assay in assay_order
            }

            acquired_alleles[seed][0]={
                assay:[]
                for assay in assay_order
            }

            for stage,task in enumerate(task_order,start=1):
                assay=_base_task(task)
                existing=set(running[assay])

                for allele in partition_test_alleles[seed][task]:
                    if allele not in existing:
                        running[assay].append(allele)
                        existing.add(allele)

                acquired_alleles[seed][stage]={
                    assay:list(vals)
                    for assay,vals in running.items()
                }

    # ================================================================
    # SEED-SPECIFIC PERFORMANCE TRAJECTORIES
    # ================================================================
    full_performance_by_seed={}
    partition_performance_by_seed={}
    assay_specific_performance_by_seed={}

    for seed in seeds:

        # -------------------------------------------------------------
        # Full fixed test-set trajectories.
        # -------------------------------------------------------------
        full_performance={}

        for method in methods:
            perf=pd.DataFrame(
                index=np.arange(n_stages+1),
                columns=eval_domains,
                dtype=float,
            )

            for dataset in eval_domains:
                perf.loc[0,dataset]=_macro(
                    dataset,
                    base_score_col,
                )

            for stage in range(1,n_stages+1):
                col=_score_col(stage,method,seed)

                for dataset in eval_domains:
                    perf.loc[stage,dataset]=_macro(
                        dataset,
                        col,
                    )

            full_performance[method]=perf

        full_performance_by_seed[seed]=full_performance

        # -------------------------------------------------------------
        # Seed-specific partition trajectories.
        # -------------------------------------------------------------
        if allele_specific_tasks:
            partition_performance={}

            for method in methods:
                task_perf={}

                for task in task_order:
                    assay=_base_task(task)
                    alleles=partition_test_alleles[seed][task]

                    values=np.full(
                        n_stages+1,
                        np.nan,
                        dtype=float,
                    )

                    values[0]=_macro(
                        assay,
                        base_score_col,
                        alleles,
                    )

                    for stage in range(1,n_stages+1):
                        values[stage]=_macro(
                            assay,
                            _score_col(stage,method,seed),
                            alleles,
                        )

                    task_perf[task]=values

                partition_performance[method]=pd.DataFrame(
                    task_perf,
                    index=np.arange(n_stages+1),
                )

            partition_performance_by_seed[seed]=(
                partition_performance
            )

            # ---------------------------------------------------------
            # Assay-level performance using accumulated partitions.
            # ---------------------------------------------------------
            if merge_partitions_by_assay:
                first_partition={
                    assay:next(
                        task for task in task_order
                        if _base_task(task)==assay
                    )
                    for assay in assay_order
                }

                assay_specific_performance={}

                for method in methods:
                    perf=pd.DataFrame(
                        index=np.arange(n_stages+1),
                        columns=eval_domains,
                        dtype=float,
                    )

                    # MS always uses its full fixed test set.
                    perf.loc[0,"ms"]=_macro(
                        "ms",
                        base_score_col,
                    )

                    for stage in range(1,n_stages+1):
                        perf.loc[stage,"ms"]=_macro(
                            "ms",
                            _score_col(stage,method,seed),
                        )

                    # Assay base value uses first available partition.
                    for assay in assay_order:
                        alleles=(
                            partition_test_alleles
                            [seed][first_partition[assay]]
                        )

                        perf.loc[0,assay]=_macro(
                            assay,
                            base_score_col,
                            alleles,
                        )

                    # Sequential accumulated allele sets.
                    for stage in range(1,n_stages+1):
                        col=_score_col(stage,method,seed)

                        for assay in assay_order:
                            alleles=(
                                acquired_alleles
                                [seed][stage][assay]
                            )

                            if len(alleles)==0:
                                alleles=(
                                    partition_test_alleles
                                    [seed][first_partition[assay]]
                                )

                            perf.loc[stage,assay]=_macro(
                                assay,
                                col,
                                alleles,
                            )

                    assay_specific_performance[method]=perf

                assay_specific_performance_by_seed[seed]=(
                    assay_specific_performance
                )

    # ================================================================
    # CL METRICS - ALWAYS COMPUTED WITHIN EACH SEED FIRST
    # ================================================================
    metric_rows=[]

    for seed in seeds:
        full_performance=full_performance_by_seed[seed]

        # -------------------------------------------------------------
        # NEW: optionally force full fixed evaluation sets.
        # -------------------------------------------------------------
        if evaluate_full:
            metric_performance=full_performance
            metric_task_mode="assay"

        elif not allele_specific_tasks:
            metric_performance=full_performance
            metric_task_mode="assay"

        elif merge_partitions_by_assay:
            metric_performance=(
                assay_specific_performance_by_seed[seed]
            )
            metric_task_mode="assay"

        else:
            metric_performance=(
                partition_performance_by_seed[seed]
            )
            metric_task_mode="partition"

        for include_ms in (True,False):
            mode=(
                "ms_as_task"
                if include_ms
                else "assay_tasks_only"
            )

            for method in methods:

                # =====================================================
                # ASSAY-LEVEL METRICS
                # =====================================================
                if metric_task_mode=="assay":
                    perf=metric_performance[method]

                    for stage in range(n_stages+1):
                        acquired_assays=[
                            assay
                            for assay in assay_order
                            if acquisition_stage[assay]<=stage
                        ]

                        if include_ms:
                            seen=[
                                "ms",
                                *acquired_assays,
                            ]

                            previous=[
                                "ms",
                                *[
                                    assay
                                    for assay in acquired_assays
                                    if acquisition_stage[assay]<stage
                                ],
                            ]

                            all_domains=[
                                "ms",
                                *assay_order,
                            ]

                        else:
                            seen=acquired_assays

                            previous=[
                                assay
                                for assay in acquired_assays
                                if acquisition_stage[assay]<stage
                            ]

                            all_domains=assay_order

                        if not include_ms and stage==0:
                            seen_mean=perf.loc[
                                0,
                                assay_order[0],
                            ]

                        else:
                            seen_mean=(
                                perf.loc[
                                    stage,
                                    seen,
                                ].mean()
                                if seen
                                else np.nan
                            )

                        all_domain_mean=(
                            perf.loc[
                                stage,
                                all_domains,
                            ].mean()
                        )

                        bwt_values=[]
                        forgetting_values=[]
                        positive_forgetting_values=[]

                        for task in previous:
                            acquired=(
                                0
                                if task=="ms"
                                else acquisition_stage[task]
                            )

                            bwt_values.append(
                                perf.loc[stage,task]
                                -perf.loc[acquired,task]
                            )

                            historical_best=(
                                perf.loc[
                                    acquired:stage-1,
                                    task,
                                ].max()
                            )

                            forgetting=(
                                historical_best
                                -perf.loc[stage,task]
                            )

                            # Preserve existing maximum-forgetting
                            # behavior exactly.
                            forgetting_values.append(
                                forgetting
                            )

                            # NEW mean forgetting:
                            # only positive losses contribute.
                            positive_forgetting_values.append(
                                max(0.0,float(forgetting))
                            )

                        bwt=(
                            np.mean(bwt_values)
                            if bwt_values
                            else np.nan
                        )

                        maximum_forgetting=(
                            np.max(forgetting_values)
                            if forgetting_values
                            else np.nan
                        )

                        mean_forgetting=(
                            np.mean(positive_forgetting_values)
                            if positive_forgetting_values
                            else np.nan
                        )

                        ms_retention=(
                            full_performance[method]
                            .loc[stage,"ms"]
                            -
                            full_performance[method]
                            .loc[0,"ms"]
                        )

                        metric_rows.append({
                            "seed":seed,
                            "mode":mode,
                            "method":method,
                            "stage":stage,
                            "update":(
                                "Base"
                                if stage==0
                                else task_order[stage-1]
                            ),
                            "seen_task_mean":seen_mean,
                            "all_domain_mean":all_domain_mean,
                            "backward_transfer":bwt,
                            "maximum_forgetting":maximum_forgetting,
                            "mean_forgetting":mean_forgetting,
                            "ms_retention":ms_retention,
                        })

                # =====================================================
                # PARTITION-AS-TASK METRICS
                # =====================================================
                else:
                    perf=metric_performance[method]

                    for stage in range(n_stages+1):
                        seen_partitions=task_order[:stage]
                        previous_partitions=(
                            task_order[:max(stage-1,0)]
                        )

                        seen_values=[]

                        if include_ms:
                            seen_values.append(
                                full_performance[method]
                                .loc[stage,"ms"]
                            )

                        for task in seen_partitions:
                            seen_values.append(
                                perf.loc[stage,task]
                            )

                        if not include_ms and stage==0:
                            seen_mean=perf.loc[
                                0,
                                task_order[0],
                            ]

                        else:
                            seen_mean=(
                                np.mean(seen_values)
                                if seen_values
                                else np.nan
                            )

                        all_values=[]

                        if include_ms:
                            all_values.append(
                                full_performance[method]
                                .loc[stage,"ms"]
                            )

                        all_values.extend(
                            perf.loc[
                                stage,
                                task_order,
                            ].to_numpy(float)
                        )

                        all_domain_mean=np.nanmean(
                            all_values
                        )

                        bwt_values=[]
                        forgetting_values=[]
                        positive_forgetting_values=[]

                        if include_ms and stage>0:
                            bwt_values.append(
                                full_performance[method]
                                .loc[stage,"ms"]
                                -
                                full_performance[method]
                                .loc[0,"ms"]
                            )

                            forgetting=(
                                full_performance[method]
                                .loc[:stage-1,"ms"]
                                .max()
                                -
                                full_performance[method]
                                .loc[stage,"ms"]
                            )

                            forgetting_values.append(
                                forgetting
                            )

                            positive_forgetting_values.append(
                                max(0.0,float(forgetting))
                            )

                        for task in previous_partitions:
                            acquired=task_order.index(task)+1

                            bwt_values.append(
                                perf.loc[stage,task]
                                -
                                perf.loc[acquired,task]
                            )

                            historical_best=(
                                perf.loc[
                                    acquired:stage-1,
                                    task,
                                ].max()
                            )

                            forgetting=(
                                historical_best
                                -perf.loc[stage,task]
                            )

                            forgetting_values.append(
                                forgetting
                            )

                            positive_forgetting_values.append(
                                max(0.0,float(forgetting))
                            )

                        bwt=(
                            np.mean(bwt_values)
                            if bwt_values
                            else np.nan
                        )

                        maximum_forgetting=(
                            np.max(forgetting_values)
                            if forgetting_values
                            else np.nan
                        )

                        mean_forgetting=(
                            np.mean(positive_forgetting_values)
                            if positive_forgetting_values
                            else np.nan
                        )

                        ms_retention=(
                            full_performance[method]
                            .loc[stage,"ms"]
                            -
                            full_performance[method]
                            .loc[0,"ms"]
                        )

                        metric_rows.append({
                            "seed":seed,
                            "mode":mode,
                            "method":method,
                            "stage":stage,
                            "update":(
                                "Base"
                                if stage==0
                                else task_order[stage-1]
                            ),
                            "seen_task_mean":seen_mean,
                            "all_domain_mean":all_domain_mean,
                            "backward_transfer":bwt,
                            "maximum_forgetting":maximum_forgetting,
                            "mean_forgetting":mean_forgetting,
                            "ms_retention":ms_retention,
                        })

    metrics_by_seed=pd.DataFrame(metric_rows)

    # ================================================================
    # ACROSS-SEED PERFORMANCE SUMMARIES
    # ================================================================
    def _aggregate_performance(perf_by_seed):
        if not perf_by_seed:
            return {}

        out={}

        for method in methods:
            template=perf_by_seed[seeds[0]][method]

            mean=pd.DataFrame(
                index=template.index,
                columns=template.columns,
                dtype=float,
            )

            lo=mean.copy()
            hi=mean.copy()
            ndf=mean.copy()

            for idx in template.index:
                for col in template.columns:
                    vals=[
                        perf_by_seed[seed][method]
                        .loc[idx,col]
                        for seed in seeds
                    ]

                    m,l,h,n=_mean_ci(vals)

                    mean.loc[idx,col]=m
                    lo.loc[idx,col]=l
                    hi.loc[idx,col]=h
                    ndf.loc[idx,col]=n

            out[method]={
                "mean":mean,
                "ci_low":lo,
                "ci_high":hi,
                "n":ndf,
            }

        return out

    full_performance_summary=(
        _aggregate_performance(
            full_performance_by_seed
        )
    )

    partition_performance_summary=(
        _aggregate_performance(
            partition_performance_by_seed
        )
    )

    assay_specific_performance_summary=(
        _aggregate_performance(
            assay_specific_performance_by_seed
        )
    )

    # ================================================================
    # ACROSS-SEED METRIC SUMMARIES
    # ================================================================
    metric_specs={
        "seen_task_mean":(
            "Average performance",
            "mAP",
        ),
        "ms_retention":(
            "MS retention",
            "ΔmAP from base MS",
        ),
        "all_domain_mean":(
            "All domain mean performance",
            "mAP",
        ),
        "backward_transfer":(
            "Backward transfer",
            "BWT",
        ),
        "maximum_forgetting":(
            "Maximum forgetting",
            "Forgetting",
        ),
        "mean_forgetting":(
            "Mean forgetting",
            "Forgetting",
        ),
    }

    metric_summary_rows=[]
    group_cols=[
        "mode",
        "method",
        "stage",
        "update",
    ]

    for keys,g in metrics_by_seed.groupby(
        group_cols,
        sort=False,
    ):
        row=dict(zip(group_cols,keys))

        for metric in metric_specs:
            m,l,h,n=_mean_ci(
                g[metric].to_numpy(float)
            )

            row[f"{metric}_mean"]=m
            row[f"{metric}_ci_low"]=l
            row[f"{metric}_ci_high"]=h
            row[f"{metric}_n"]=n

        metric_summary_rows.append(row)

    metrics=pd.DataFrame(
        metric_summary_rows
    )

    # ================================================================
    # X AXIS
    # ================================================================
    x=np.arange(n_stages+1)

    xlabels=[
        "Base",
        *[
            _task_tick_label(task)
            for task in task_order
        ],
    ]

    tick_ha=(
        "center"
        if xtick_rotation==0
        else "right"
    ) if xtick_ha is None else xtick_ha

    # ================================================================
    # CYCLE BOUNDARIES
    # ================================================================
    if cycle_line_positions is None:
        cycle_ids=[]

        for task in task_order:
            m=re.search(
                r"_(\d+)$",
                str(task),
            )

            cycle_ids.append(
                int(m.group(1))
                if m
                else None
            )

        auto_cycle_lines=[]

        if all(c is not None for c in cycle_ids):
            for stage in range(2,n_stages+1):
                if cycle_ids[stage-1]!=cycle_ids[stage-2]:
                    auto_cycle_lines.append(
                        stage-0.5
                    )

        else:
            first_assay=_base_task(
                task_order[0]
            )

            for stage,task in enumerate(
                task_order[1:],
                start=2,
            ):
                if _base_task(task)==first_assay:
                    auto_cycle_lines.append(
                        stage-0.5
                    )

        cycle_positions=auto_cycle_lines

    else:
        cycle_positions=list(
            cycle_line_positions
        )

    # ================================================================
    # STYLE
    # ================================================================
    default_colors=(
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    colors={
        method:(
            method_colors[method]
            if (
                method_colors is not None
                and method in method_colors
            )
            else default_colors[
                i%len(default_colors)
            ]
        )
        for i,method in enumerate(methods)
    }

    def _style_axis(ax):
        ax.spines["top"].set_visible(
            not remove_top_spine
        )

        ax.spines["right"].set_visible(
            not remove_right_spine
        )

        if xlabel_fontsize is not None:
            ax.xaxis.label.set_size(
                xlabel_fontsize
            )

        if ylabel_fontsize is not None:
            ax.yaxis.label.set_size(
                ylabel_fontsize
            )

        if title_fontsize is not None:
            ax.title.set_size(
                title_fontsize
            )

        if ytick_fontsize is not None:
            ax.tick_params(
                axis="y",
                labelsize=ytick_fontsize,
            )

    def _draw_cycle_lines(ax):
        if show_cycle_lines:
            for xpos in cycle_positions:
                ax.axvline(
                    xpos,
                    color=cycle_line_color,
                    linestyle=cycle_line_style,
                    linewidth=cycle_linewidth,
                    alpha=cycle_line_alpha,
                    zorder=0,
                )

    def _draw_line_with_ci(
        ax,
        xvals,
        y,
        lo,
        hi,
        method,
        draw_ci=True,
    ):
        color=colors[method]

        ax.plot(
            xvals,
            y,
            marker=marker,
            markersize=markersize,
            markeredgewidth=marker_edgewidth,
            linewidth=linewidth,
            label=method_labels.get(
                method,
                method,
            ),
            color=color,
            zorder=3,
        )

        if show_ci and draw_ci:
            y=np.asarray(y,float)
            lo=np.asarray(lo,float)
            hi=np.asarray(hi,float)

            lower=np.clip(
                y-lo,
                0,
                None,
            )

            upper=np.clip(
                hi-y,
                0,
                None,
            )

            ax.errorbar(
                xvals,
                y,
                yerr=np.vstack([
                    lower,
                    upper,
                ]),
                fmt="none",
                ecolor=color,
                elinewidth=error_linewidth,
                capsize=error_capsize,
                capthick=error_capthick,
                zorder=2,
            )

    def _draw_seed_metric_lines(
        ax,
        metric,
        method,
        mode,
    ):
        """
        Draw one faint trajectory per seed for a single metric/method.

        No cross-method connectors are drawn. Seed IDs remain paired
        internally across methods, but each method is visualized separately.
        """
        color=colors[method]

        for seed in seeds:
            q=metrics_by_seed[
                (metrics_by_seed["mode"]==mode)
                &
                (metrics_by_seed["method"]==method)
                &
                (metrics_by_seed["seed"]==seed)
            ].sort_values("stage")

            ax.plot(
                q["stage"].to_numpy(float),
                q[metric].to_numpy(float),
                color=color,
                alpha=seed_line_alpha,
                linewidth=seed_linewidth,
                zorder=1,
            )

    # ================================================================
    # ASSAY PERFORMANCE
    # ================================================================
    if plot_type=="assay_performance":

        # NEW: evaluate_full overrides partition-aware display.
        if evaluate_full:
            plot_summary=(
                full_performance_summary
            )

        elif (
            allele_specific_tasks
            and merge_partitions_by_assay
        ):
            plot_summary=(
                assay_specific_performance_summary
            )

        else:
            plot_summary=(
                full_performance_summary
            )

        default_panels=list(
            eval_domains
        )

        if not include_ms_panel:
            default_panels=[
                d for d in default_panels
                if d!="ms"
            ]

        panels=(
            default_panels
            if panel_order is None
            else list(panel_order)
        )

        invalid=[
            p for p in panels
            if p not in eval_domains
        ]

        if invalid:
            raise ValueError(
                f"Invalid assay performance panels: "
                f"{invalid}"
            )

        if (
            not include_ms_panel
            and "ms" in panels
        ):
            raise ValueError(
                "panel_order contains 'ms' but "
                "include_ms_panel=False."
            )

        fig,axes=plt.subplots(
            2,
            3,
            figsize=figsize,
            dpi=dpi,
        )

        axes=np.asarray(
            axes
        ).ravel()

        for ax,dataset in zip(
            axes,
            panels,
        ):
            for method in methods:
                s=plot_summary[method]

                _draw_line_with_ci(
                    ax,
                    x,
                    s["mean"][dataset]
                    .to_numpy(float),
                    s["ci_low"][dataset]
                    .to_numpy(float),
                    s["ci_high"][dataset]
                    .to_numpy(float),
                    method,
                )

            if (
                show_assay_update_lines
                and dataset!="ms"
            ):
                for stage,update_name in enumerate(
                    task_order,
                    start=1,
                ):
                    if (
                        _base_task(update_name)
                        ==dataset
                    ):
                        ax.axvline(
                            stage,
                            color=assay_update_line_color,
                            linestyle=assay_update_line_style,
                            linewidth=assay_update_linewidth,
                            alpha=assay_update_line_alpha,
                            zorder=0,
                        )

            _draw_cycle_lines(ax)

            ax.set_title(
                _label(dataset)
            )

            ax.set_ylabel("mAP")
            ax.set_xlabel(xlabel)

            ax.set_xticks(x)

            ax.set_xticklabels(
                xlabels,
                rotation=xtick_rotation,
                ha=tick_ha,
                fontsize=xtick_fontsize,
            )

            _style_axis(ax)

    # ================================================================
    # CONTINUAL LEARNING METRICS
    # ================================================================
    else:
        mode=(
            "ms_as_task"
            if include_ms_as_task
            else "assay_tasks_only"
        )

        # Existing default panel structure.
        default_panels=[
            "seen_task_mean",
            "ms_retention",
            "all_domain_mean",
            "backward_transfer",
            (
                "mean_forgetting"
                if use_mean_forgetting
                else "maximum_forgetting"
            ),
        ]

        if panel_order is None:
            panels=default_panels

        else:
            panels=list(panel_order)

            # NEW: if user asks to replace maximum forgetting but supplies
            # the old panel_order, transparently substitute the new metric.
            if use_mean_forgetting:
                panels=[
                    (
                        "mean_forgetting"
                        if p=="maximum_forgetting"
                        else p
                    )
                    for p in panels
                ]

        invalid=[
            p for p in panels
            if p not in metric_specs
        ]

        if invalid:
            raise ValueError(
                f"Invalid metric panels: {invalid}. "
                f"Valid metrics: {list(metric_specs)}"
            )

        fig,axes=plt.subplots(
            2,
            3,
            figsize=figsize,
            dpi=dpi,
        )

        axes=np.asarray(
            axes
        ).ravel()

        forgetting_panel=(
            "mean_forgetting"
            if use_mean_forgetting
            else "maximum_forgetting"
        )

        for ax,metric in zip(
            axes,
            panels,
        ):
            title,ylabel=(
                metric_specs[metric]
            )

            for method in methods:
                # -----------------------------------------------------
                # NEW: faint raw seed trajectories for forgetting.
                # -----------------------------------------------------
                if (
                    show_forgetting_seed_lines
                    and metric==forgetting_panel
                ):
                    _draw_seed_metric_lines(
                        ax,
                        metric,
                        method,
                        mode,
                    )

                q=metrics[
                    (metrics["mode"]==mode)
                    &
                    (metrics["method"]==method)
                ].sort_values("stage")

                # If raw seed lines are displayed for forgetting,
                # suppress the CI on that panel only.
                draw_ci=not (
                    show_forgetting_seed_lines
                    and metric==forgetting_panel
                )

                _draw_line_with_ci(
                    ax,
                    q["stage"].to_numpy(float),
                    q[f"{metric}_mean"]
                    .to_numpy(float),
                    q[f"{metric}_ci_low"]
                    .to_numpy(float),
                    q[f"{metric}_ci_high"]
                    .to_numpy(float),
                    method,
                    draw_ci=draw_ci,
                )

            if (
                show_zero_lines
                and metric in {
                    "backward_transfer",
                    "maximum_forgetting",
                    "mean_forgetting",
                    "ms_retention",
                }
            ):
                ax.axhline(
                    0,
                    color=zero_line_color,
                    linestyle=zero_line_style,
                    linewidth=zero_linewidth,
                    zorder=0,
                )

            _draw_cycle_lines(ax)

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            ax.set_xticks(x)

            ax.set_xticklabels(
                xlabels,
                rotation=xtick_rotation,
                ha=tick_ha,
                fontsize=xtick_fontsize,
            )

            _style_axis(ax)

    # ================================================================
    # LEGEND / UNUSED PANELS
    # ================================================================
    used=len(panels)

    if show_legend and used<len(axes):
        legend_ax=axes[used]
        legend_ax.axis("off")

        handles,labels=(
            axes[0]
            .get_legend_handles_labels()
        )

        legend_ax.legend(
            handles,
            labels,
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=legend_frame,
            fontsize=legend_fontsize,
        )

        start_off=used+1

    else:
        start_off=used

        if show_legend:
            handles,labels=(
                axes[0]
                .get_legend_handles_labels()
            )

            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=legend_ncol,
                frameon=legend_frame,
                fontsize=legend_fontsize,
            )

    for ax in axes[start_off:]:
        ax.axis("off")

    # ================================================================
    # LAYOUT
    # ================================================================
    fig.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
        **(
            {"left":left_margin}
            if left_margin is not None
            else {}
        ),
        **(
            {"right":right_margin}
            if right_margin is not None
            else {}
        ),
        **(
            {"bottom":bottom_margin}
            if bottom_margin is not None
            else {}
        ),
        **(
            {"top":top_margin}
            if top_margin is not None
            else {}
        ),
    )

    # ================================================================
    # SAVE
    # ================================================================
    if save_path is not None:

        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

    if verbose:
        print("\nLong-horizon plotting summary:")
        print(f"  seeds: {seeds}")
        print(f"  evaluate_full: {evaluate_full}")
        print(f"  allele_specific_tasks: {allele_specific_tasks}")
        print(f"  merge_partitions_by_assay: {merge_partitions_by_assay}")
        print(f"  use_mean_forgetting: {use_mean_forgetting}")
        print(f"  show_forgetting_seed_lines: {show_forgetting_seed_lines}")
        if show_forgetting_seed_lines:
            print(
                "  forgetting panel: mean line + individual seed "
                "trajectories; CI suppressed for this panel"
            )

    if show:
        plt.show()

    return {
        # Raw seed-level quantities.
        "seeds":seeds,
        "full_performance_by_seed":full_performance_by_seed,
        "partition_performance_by_seed":partition_performance_by_seed,
        "assay_specific_performance_by_seed":
            assay_specific_performance_by_seed,
        "metrics_by_seed":metrics_by_seed,

        # Across-seed summaries.
        "full_performance":full_performance_summary,
        "partition_performance":partition_performance_summary,
        "assay_specific_performance":
            assay_specific_performance_summary,
        "metrics":metrics,

        # Metadata.
        "acquisition_stage":acquisition_stage,
        "partition_test_alleles":partition_test_alleles,
        "acquired_alleles":acquired_alleles,

        # New settings.
        "evaluate_full":evaluate_full,
        "use_mean_forgetting":use_mean_forgetting,
        "show_forgetting_seed_lines":show_forgetting_seed_lines,

        # Figure.
        "figure":fig,
        "axes":axes,
        "plot_type":plot_type,
        "panel_order":panels,
        "include_ms_as_task":include_ms_as_task,
        "ci":ci,
    }


import itertools
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def summarize_replay_buffer_scaling(
    ms_train_data,
    train_sets,
    plot=True,

    # Plot mode.
    memory_only=False,

    # Plot colors.
    point_color="#9e9e9e",
    mean_color="#1f77b4",

    # Plot geometry.
    fig_size=(10,5),
    memory_only_fig_size=(5,5),
    dpi=300,

    # Fonts.
    font_family="Nimbus Sans",
    title_fontsize=12,
    label_fontsize=11,
    tick_fontsize=9,
    annotation_fontsize=9,

    # Annotation.
    annotate_row_increase=True,
    annotation_format="compact",

    # Output.
    save_path=None,
    show_plot=True,
):
    """
    Summarize cumulative replay-buffer size and DataFrame memory usage
    for MS alone and MS + every combination of prior assay datasets.

    If memory_only=True, plot only additional replay-buffer storage
    relative to MS alone and annotate mean points with the corresponding
    mean increase in replay-buffer rows.

    Parameters
    ----------
    ms_train_data : pd.DataFrame
        Base MS replay dataset.

    train_sets : dict
        Mapping assay name -> training dataframe.

    plot : bool
        Whether to generate plots.

    memory_only : bool
        If False, plot:
            1. total replay-buffer rows
            2. total replay-buffer memory

        If True, plot only:
            additional replay-buffer memory relative to MS alone

        Mean points can additionally be annotated with row increases.

    point_color : str
        Color for individual assay-combination points.

    mean_color : str
        Color for mean line and mean markers.

    annotation_format : {"compact","integer"}
        "compact":
            4,000,000 -> +4.0M rows

        "integer":
            4,000,000 -> +4,000,000 rows

    save_path : str or Path, optional
        Base output path without extension. Saves PDF and PNG.

    Returns
    -------
    summary_df : pd.DataFrame

    If plot=True:
        summary_df, fig, axes
    Else:
        summary_df
    """

    # ============================================================
    # FONT / VECTOR OUTPUT SETTINGS
    # ============================================================

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

    # ============================================================
    # BUILD BUFFER-SCALING SUMMARY
    # ============================================================

    rows=[]

    ms_rows=len(ms_train_data)
    ms_bytes=ms_train_data.memory_usage(
        index=True,
        deep=True,
    ).sum()

    rows.append({
        "n_added_assays":0,
        "assays":"ms",
        "n_rows":ms_rows,
        "memory_bytes":ms_bytes,
        "memory_mb":ms_bytes/1024**2,
        "memory_gb":ms_bytes/1024**3,
        "relative_to_ms":1.0,
        "additional_rows":0,
        "additional_memory_bytes":0,
        "additional_memory_mb":0,
        "additional_memory_gb":0,
    })

    assay_names=list(train_sets.keys())

    for n_assays in range(1,len(assay_names)+1):
        for combo in itertools.combinations(
            assay_names,
            n_assays,
        ):
            combo_rows=(
                ms_rows+
                sum(
                    len(train_sets[name])
                    for name in combo
                )
            )

            combo_bytes=(
                ms_bytes+
                sum(
                    train_sets[name]
                    .memory_usage(
                        index=True,
                        deep=True,
                    )
                    .sum()
                    for name in combo
                )
            )

            rows.append({
                "n_added_assays":n_assays,
                "assays":"ms+"+"+".join(combo),
                "n_rows":combo_rows,
                "memory_bytes":combo_bytes,
                "memory_mb":combo_bytes/1024**2,
                "memory_gb":combo_bytes/1024**3,
                "relative_to_ms":combo_bytes/ms_bytes,
                "additional_rows":combo_rows-ms_rows,
                "additional_memory_bytes":combo_bytes-ms_bytes,
                "additional_memory_mb":(combo_bytes-ms_bytes)/1024**2,
                "additional_memory_gb":(combo_bytes-ms_bytes)/1024**3,
            })

    summary_df=pd.DataFrame(rows)

    if not plot:
        return summary_df

    # ============================================================
    # ROW-COUNT ANNOTATION FORMATTER
    # ============================================================

    def _format_row_increase(x):
        x=float(x)

        if annotation_format=="integer":
            return f"+{int(round(x)):,} rows"

        if annotation_format!="compact":
            raise ValueError(
                "annotation_format must be 'compact' or 'integer'."
            )

        if abs(x)>=1_000_000:
            return f"+{x/1_000_000:.1f}M rows"

        if abs(x)>=1_000:
            return f"+{x/1_000:.1f}K rows"

        return f"+{int(round(x))} rows"

    # ============================================================
    # MEMORY-ONLY MODE
    # ============================================================

    if memory_only:
        fig,ax=plt.subplots(
            1,
            1,
            figsize=memory_only_fig_size,
            dpi=dpi,
        )

        # Individual combinations.
        for n_assays,group in summary_df.groupby(
            "n_added_assays"
        ):
            ax.scatter(
                [n_assays]*len(group),
                group["additional_memory_gb"],
                color=point_color,
                s=35,
                zorder=2,
            )

        # Means.
        grouped=(
            summary_df
            .groupby("n_added_assays",as_index=False)
            .agg(
                memory_gb=("additional_memory_gb","mean"),
                row_increase=("additional_rows","mean"),
            )
        )

        ax.plot(
            grouped["n_added_assays"],
            grouped["memory_gb"],
            marker="o",
            color=mean_color,
            linewidth=1.5,
            markersize=5,
            zorder=3,
        )

        # Row-increase annotations on mean points.
        if annotate_row_increase:
            for _,r in grouped.iterrows():
                ax.annotate(
                    _format_row_increase(
                        r["row_increase"]
                    ),
                    xy=(
                        r["n_added_assays"],
                        r["memory_gb"],
                    ),
                    xytext=(0,8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_fontsize,
                )

        ax.set_xlabel(
            "Number of prior non-MS assays in replay buffer",
            fontsize=label_fontsize,
        )

        ax.set_ylabel(
            "Additional replay buffer memory relative to MS only (GB)",
            fontsize=label_fontsize,
        )

        ax.set_title(
            "Replay buffer storage scaling",
            fontsize=title_fontsize,
        )

        ax.set_xticks(
            range(len(assay_names)+1)
        )

        ax.tick_params(
            axis="both",
            labelsize=tick_fontsize,
        )

        ax.set_box_aspect(1)

        fig.tight_layout()

        axes=ax

    # ============================================================
    # ORIGINAL TWO-PANEL MODE
    # ============================================================

    else:
        fig,axes=plt.subplots(
            1,
            2,
            figsize=fig_size,
            dpi=dpi,
        )

        plot_specs=[
            (
                axes[0],
                "n_rows",
                "Replay buffer rows",
                "Replay buffer dataset-size scaling",
            ),
            (
                axes[1],
                "memory_gb",
                "Replay buffer memory (GB)",
                "Replay buffer storage scaling",
            ),
        ]

        for ax,y_col,ylabel,title in plot_specs:

            # Individual combinations.
            for n_assays,group in summary_df.groupby(
                "n_added_assays"
            ):
                ax.scatter(
                    [n_assays]*len(group),
                    group[y_col],
                    color=point_color,
                    s=35,
                    zorder=2,
                )

            # Mean across combinations.
            grouped=(
                summary_df
                .groupby("n_added_assays")[y_col]
                .mean()
            )

            ax.plot(
                grouped.index,
                grouped.values,
                marker="o",
                color=mean_color,
                linewidth=1.5,
                markersize=5,
                zorder=3,
            )

            ax.set_xlabel(
                "Number of prior non-MS assays in replay buffer",
                fontsize=label_fontsize,
            )

            ax.set_ylabel(
                ylabel,
                fontsize=label_fontsize,
            )

            ax.set_title(
                title,
                fontsize=title_fontsize,
            )

            ax.set_xticks(
                range(len(assay_names)+1)
            )

            ax.tick_params(
                axis="both",
                labelsize=tick_fontsize,
            )

            ax.set_box_aspect(1)

        fig.tight_layout()

    # ============================================================
    # SAVE
    # ============================================================

    if save_path is not None:
        save_path=Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            f"{save_path}.pdf",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )

        fig.savefig(
            f"{save_path}.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )

        print(
            f"Saved:\n"
            f"  {save_path}.pdf\n"
            f"  {save_path}.png"
        )

    if show_plot:
        plt.show()

    return summary_df,fig,axes

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def plot_sequential_vs_joint(
    mean_df,ci_df,trajectories,joint_points,colors,
    group_gap=1.5,figsize=(7,4),dpi=300,
    linewidth=1.5,markersize=5,
    error_linewidth=1.0,error_capsize=3,
    ylabel="mAP",
    show_top_spine=True,show_right_spine=True,
    show_legend=True,legend_ncol=2,
    font_family="Nimbus Sans",
    save_path=None,
    save_pdf=True,save_png=True,save_svg=True,
    bbox_inches="tight",pad_inches=0.05,
    show=True,
):
    # ------------------------------------------------------------
    # Publication/export settings.
    # ------------------------------------------------------------
    matplotlib.rcParams["pdf.fonttype"]=42
    matplotlib.rcParams["ps.fonttype"]=42
    matplotlib.rcParams["svg.fonttype"]="none"

    matplotlib.rcParams["font.family"]="sans-serif"
    matplotlib.rcParams["font.sans-serif"]=[
        font_family,
        "Nimbus Sans",
        "Nimbus Sans L",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]

    mean=mean_df.iloc[0] if isinstance(mean_df,pd.DataFrame) else mean_df
    ci=ci_df.iloc[0] if isinstance(ci_df,pd.DataFrame) else ci_df

    fig,ax=plt.subplots(figsize=figsize,dpi=dpi)

    # ------------------------------------------------------------
    # Group positions.
    # ------------------------------------------------------------
    g1=np.array([0,1,2],float)

    g2_start=g1[-1]+group_gap+1
    g2=np.array([g2_start,g2_start+1,g2_start+2],float)

    joint_start=g2[-1]+group_gap+1
    gj=np.array([joint_start,joint_start+1],float)

    seq_positions=[g1,g2]
    tick_positions=[]
    tick_labels=[]

    # ------------------------------------------------------------
    # Sequential trajectories.
    # ------------------------------------------------------------
    for group_i,(_,g) in enumerate(trajectories.items()):
        x=seq_positions[group_i]

        tick_positions.extend(x)
        tick_labels.extend(["Base"]+g["stage_labels"])

        for method,cols in g["methods"].items():
            y=np.array([float(mean[c]) for c in cols])
            color=colors[method]

            ax.plot(
                x,y,
                marker="o",
                color=color,
                linewidth=linewidth,
                markersize=markersize,
                label=method if group_i==0 else "_nolegend_",
                zorder=3,
            )

            for xi,col,yi in zip(x,cols,y):
                lo=f"{col}_ci_low"
                hi=f"{col}_ci_high"

                if lo in ci.index and hi in ci.index and pd.notna(ci[lo]) and pd.notna(ci[hi]):
                    ax.errorbar(
                        xi,yi,
                        yerr=np.array([
                            [yi-float(ci[lo])],
                            [float(ci[hi])-yi],
                        ]),
                        fmt="none",
                        ecolor=color,
                        elinewidth=error_linewidth,
                        capsize=error_capsize,
                        capthick=error_linewidth,
                        zorder=2,
                    )

    # ------------------------------------------------------------
    # Joint retraining: isolated 300 / 600 points.
    # ------------------------------------------------------------
    tick_positions.extend(gj)
    tick_labels.extend(["300","600"])

    for method,cols in joint_points.items():
        if len(cols)!=2:
            raise ValueError(
                f"{method}: joint_points must contain [300step,600step]."
            )

        color=colors[method]

        for xi,col in zip(gj,cols):
            yi=float(mean[col])

            ax.plot(
                xi,yi,
                marker="o",
                linestyle="none",
                color=color,
                markersize=markersize,
                zorder=3,
            )

            lo=f"{col}_ci_low"
            hi=f"{col}_ci_high"

            if lo in ci.index and hi in ci.index and pd.notna(ci[lo]) and pd.notna(ci[hi]):
                ax.errorbar(
                    xi,yi,
                    yerr=np.array([
                        [yi-float(ci[lo])],
                        [float(ci[hi])-yi],
                    ]),
                    fmt="none",
                    ecolor=color,
                    elinewidth=error_linewidth,
                    capsize=error_capsize,
                    capthick=error_linewidth,
                    zorder=2,
                )

    # ------------------------------------------------------------
    # Formatting.
    # ------------------------------------------------------------
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_ylabel(ylabel)

    ax.set_xlim(
        g1[0]-0.35,
        gj[-1]+0.35,
    )

    ax.spines["top"].set_visible(show_top_spine)
    ax.spines["right"].set_visible(show_right_spine)

    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5,-0.14),
            ncol=legend_ncol,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.23)

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------
    if save_path is not None:
        if save_pdf:
            fig.savefig(
                f"{save_path}.pdf",
                format="pdf",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

        if save_png:
            fig.savefig(
                f"{save_path}.png",
                format="png",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

        if save_svg:
            fig.savefig(
                f"{save_path}.svg",
                format="svg",
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )

    if show:
        plt.show()

    return fig,ax

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def compute_long_horizon_allele_ap_multi_seeded(
    adaptive_test_sets, task_order, allele_partitions_by_seed,
    adaptive_50pct_test_sets=None,
    task_order_name="t1_allele_partition", base_score_col="mhcprime",
    adaptive_suffix="_pepcl_multi_replay_dr_full_buffer",
    adaptive_50pct_suffix="_pepcl_multi_replay_dr_full_buffer_50pct_ms",
    seed_tag="seed", allele_col="allele", label_col="label", verbose=True
):
    """
    Multi-seed version of compute_long_horizon_allele_ap.

    Expected score naming:
        {task_order_name}_s{stage}{method_suffix}_seed_{seed}

    allele_partitions_by_seed supplies the authoritative seed IDs:
        {
            0: partition_summary_seed0,
            1: partition_summary_seed1,
            ...
        }

    Returns raw per-allele APs only. No averaging across seeds is performed.
    """
    single_dict_mode=adaptive_50pct_test_sets is None
    seeds=list(allele_partitions_by_seed)
    if not seeds:
        raise ValueError("allele_partitions_by_seed is empty.")

    def _base_task_name(name):
        return re.sub(r"_\d+$","",name)

    def _col_count(df,name):
        return int(np.sum(df.columns.to_numpy(dtype=object)==name))

    def _series(df,name,allow_first_duplicate=False):
        pos=np.flatnonzero(df.columns.to_numpy(dtype=object)==name)
        if len(pos)==0:
            raise KeyError(name)
        if len(pos)>1 and not allow_first_duplicate:
            raise ValueError(
                f"Column {name!r} occurs {len(pos)} times; "
                "only base_score_col may be duplicated."
            )
        return df.iloc[:,pos[0]]

    # Validate that every seed has the same named task sequence available.
    for seed in seeds:
        missing_tasks=[
            task for task in task_order
            if task not in allele_partitions_by_seed[seed]
        ]
        if missing_tasks:
            raise ValueError(
                f"Seed {seed}: allele partition dictionary is missing tasks: "
                f"{missing_tasks}"
            )

    assay_keys=list(dict.fromkeys(
        _base_task_name(task) for task in task_order
    ))
    wanted_keys=["ms"]+assay_keys
    n_stages=len(task_order)

    if verbose:
        print(f"Seeds/partitions: {seeds}")
        print(
            "Input mode: "
            f"{'single combined test-set dictionary' if single_dict_mode else 'two test-set dictionaries'}"
        )
        print(f"Long-horizon task order ({n_stages} updates): {task_order}")
        print(f"Evaluation sets retained: {wanted_keys}")

    missing=[k for k in wanted_keys if k not in adaptive_test_sets]
    if missing:
        raise ValueError(
            f"Missing test sets from adaptive_test_sets: {missing}"
        )

    if not single_dict_mode:
        missing=[
            k for k in wanted_keys
            if k not in adaptive_50pct_test_sets
        ]
        if missing:
            raise ValueError(
                f"Missing test sets from adaptive_50pct_test_sets: {missing}"
            )

    adaptive_cols=[
        f"{task_order_name}_s{s}{adaptive_suffix}_{seed_tag}_{seed}"
        for seed in seeds
        for s in range(1,n_stages+1)
    ]

    adaptive_50pct_cols=[
        f"{task_order_name}_s{s}{adaptive_50pct_suffix}_{seed_tag}_{seed}"
        for seed in seeds
        for s in range(1,n_stages+1)
    ]

    def _compute_allele_ap(df,score_cols,desc):
        for c in [allele_col,label_col]:
            if _col_count(df,c)!=1:
                raise ValueError(
                    f"{desc}: expected exactly one {c!r} column, "
                    f"found {_col_count(df,c)}."
                )

        # Base may occur multiple times after upstream merges.
        # Per request, always use its first occurrence.
        if _col_count(df,base_score_col)==0:
            raise ValueError(
                f"{desc}: missing base score column {base_score_col!r}."
            )

        score_cols=list(dict.fromkeys(score_cols))

        missing=[
            c for c in score_cols
            if _col_count(df,c)==0
        ]
        if missing:
            raise ValueError(
                f"{desc}: missing {len(missing)} score columns. "
                f"First missing: {missing[:10]}"
            )

        duplicated=[
            c for c in score_cols
            if _col_count(df,c)>1
        ]
        if duplicated:
            raise ValueError(
                f"{desc}: duplicated non-base score columns found. "
                f"First: {duplicated[:10]}"
            )

        labels=_series(df,label_col).to_numpy()
        cols=[base_score_col]+score_cols

        arrays={
            base_score_col:
                _series(
                    df,
                    base_score_col,
                    allow_first_duplicate=True,
                ).to_numpy()
        }
        arrays.update({
            c:_series(df,c).to_numpy()
            for c in score_cols
        })

        grouped=df.groupby(
            allele_col,
            sort=False,
        ).indices

        rows=[]

        for allele,idx in tqdm(
            grouped.items(),
            total=len(grouped),
            desc=desc,
            leave=True,
        ):
            idx=np.asarray(idx,dtype=np.int64)
            y=labels[idx]
            row={allele_col:allele}

            if np.sum(y==1)==0:
                for c in cols:
                    row[c]=np.nan
            else:
                for c in cols:
                    row[c]=average_precision_score(
                        y,
                        arrays[c][idx],
                    )

            rows.append(row)

        return pd.DataFrame(rows)

    output={}

    for test_name in wanted_keys:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Computing allele AP: {test_name}")
            print(f"{'='*70}")

        # -----------------------------------------------------
        # SINGLE-DICT MODE
        # -----------------------------------------------------
        if single_dict_mode:
            merged=_compute_allele_ap(
                adaptive_test_sets[test_name],
                adaptive_cols+adaptive_50pct_cols,
                desc=f"{test_name} | Long-horizon seeded AP",
            )

        # -----------------------------------------------------
        # TWO-DICT MODE
        # -----------------------------------------------------
        else:
            ap_adaptive=_compute_allele_ap(
                adaptive_test_sets[test_name],
                adaptive_cols,
                desc=f"{test_name} | Adaptive buffer seeded AP",
            )

            ap_50pct=_compute_allele_ap(
                adaptive_50pct_test_sets[test_name],
                adaptive_50pct_cols,
                desc=f"{test_name} | 50% MS buffer seeded AP",
            )

            alleles_a=set(ap_adaptive[allele_col])
            alleles_b=set(ap_50pct[allele_col])

            if alleles_a!=alleles_b:
                raise ValueError(
                    f"{test_name}: allele sets differ between dictionaries.\n"
                    f"Only adaptive: "
                    f"{sorted(alleles_a-alleles_b)[:10]}\n"
                    f"Only 50% MS: "
                    f"{sorted(alleles_b-alleles_a)[:10]}"
                )

            ap_50pct=(
                ap_50pct
                .set_index(allele_col)
                .reindex(ap_adaptive[allele_col])
                .reset_index()
            )

            # Base AP should agree between the 2 copies.
            a=ap_adaptive[base_score_col].to_numpy(float)
            b=ap_50pct[base_score_col].to_numpy(float)

            if not np.allclose(a,b,equal_nan=True):
                raise ValueError(
                    f"{test_name}: {base_score_col!r} allele AP differs "
                    "between the two test-set dictionaries."
                )

            merged=ap_adaptive.merge(
                ap_50pct.drop(columns=base_score_col),
                on=allele_col,
                how="inner",
                validate="one_to_one",
            )

        output[test_name]=merged

        if verbose:
            print(
                f"{test_name}: {len(merged)} alleles, "
                f"{len(merged.columns)-1} AP score columns"
            )

    if verbose:
        print("\nDone.")
        print("Output keys:",list(output))
        for key,df in output.items():
            print(f"  {key}: {df.shape}")

    return output

def merge_two_dicts_by_keys(dict1, dict2, keys):
    out = {}
    for key in keys:
        merged = pd.concat([dict1[key], dict2[key]], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out[key] = merged
    return out