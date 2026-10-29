"""Plotting functions used to reproduce the figures in the paper."""
import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "plot_independent_symba_vs_ps",
    "plot_independent_symba_vs_pw",
    "plot_symba_vs_ps_and_pw_combined",
    "plot_diffs_analytic_shuffle",
    "plot_summary_across_nsr_four_methods",
    "plot_roc_or_pr_curves_from_summary",
    "slim_summary_data",
    "plot_runtime_scaling",
    "plot_motifs_vs_ps_full",
    "plot_score_histogram",
]

# Display names of the symbol-noise types used in the notebooks.
NOISE_LABELS = {
    "substitution": "Substitution",
    "deletion": "Deletion",
}


def _savefig(filename):
    """Saves the current figure as SVG, creating the destination folder if needed."""
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    plt.savefig(filename, format='svg', bbox_inches='tight')


def plot_independent_symba_vs_ps(aggregated_data, noise_type="deletion", save_fig=False, filename="symba_vs_ps.svg",
                                  fontsize_title=16, fontsize_label=14, fontsize_ticks=12, fontsize_legend=12):
    """
    SYMBA mean max-F1 and PR-AUC (± std across runs) vs the per-symbol noise
    probability p_s, one curve per motif length.

    aggregated_data : dict[length] with keys 'p_s_values', 'mean_f1', 'std_f1',
        'mean_auc', 'std_auc' (as saved by the noise-robustness section).
    noise_type : 'substitution' or 'deletion' (only used for the titles).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {2: 'blue', 3: 'orange', 4: 'green'}

    for length in [2, 3, 4]:
        p_s_vals = aggregated_data[length]['p_s_values']
        mean_f1  = np.array(aggregated_data[length]['mean_f1'])
        std_f1   = np.array(aggregated_data[length]['std_f1'])
        mean_auc = np.array(aggregated_data[length]['mean_auc'])
        std_auc  = np.array(aggregated_data[length]['std_auc'])

        ax1.plot(p_s_vals, mean_f1, marker='o', color=colors[length], label=f'Length {length}')
        ax1.fill_between(p_s_vals, np.clip(mean_f1 - std_f1, 0, 1), np.clip(mean_f1 + std_f1, 0, 1),
                         color=colors[length], alpha=0.2)

        ax2.plot(p_s_vals, mean_auc, marker='s', linestyle='--', color=colors[length], label=f'Length {length}')
        ax2.fill_between(p_s_vals, np.clip(mean_auc - std_auc, 0, 1), np.clip(mean_auc + std_auc, 0, 1),
                         color=colors[length], alpha=0.2)

    error_type = NOISE_LABELS.get(noise_type, noise_type)

    for ax, metric, ylabel in [
        (ax1, f'SYMBA Max F1 vs Symbol {error_type} probability ($p_s$)',   'Mean Maximum F1-Score'),
        (ax2, f'SYMBA PR-AUC vs Symbol {error_type} probability ($p_s$)',   'Mean Area Under PR Curve')
    ]:
        ax.set_title(metric, fontsize=fontsize_title)
        ax.set_xlabel('$p_s$', fontsize=fontsize_label)
        ax.set_ylabel(ylabel, fontsize=fontsize_label)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=fontsize_legend)

    plt.tight_layout()

    if save_fig:
        _savefig(filename)
        print(f"Plot saved successfully as: {filename}")

    plt.show()


def plot_independent_symba_vs_pw(aggregated_data, noise_type="deletion", save_fig=False, filename="symba_vs_pw.svg",
                                  fontsize_title=16, fontsize_label=14, fontsize_ticks=12, fontsize_legend=12):
    """
    Same as plot_independent_symba_vs_ps, but against the probability that a
    motif of length k is corrupted, p_w = 1 - (1 - p_s)^k.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {2: 'blue', 3: 'orange', 4: 'green'}

    for length in [2, 3, 4]:
        p_s_vals = aggregated_data[length]['p_s_values']
        p_w_vals = 1 - (1 - p_s_vals)**length

        mean_f1  = np.array(aggregated_data[length]['mean_f1'])
        std_f1   = np.array(aggregated_data[length]['std_f1'])
        mean_auc = np.array(aggregated_data[length]['mean_auc'])
        std_auc  = np.array(aggregated_data[length]['std_auc'])

        ax1.plot(p_w_vals, mean_f1, marker='o', color=colors[length], label=f'Length {length}')
        ax1.fill_between(p_w_vals, np.clip(mean_f1 - std_f1, 0, 1), np.clip(mean_f1 + std_f1, 0, 1),
                         color=colors[length], alpha=0.2)

        ax2.plot(p_w_vals, mean_auc, marker='s', linestyle='--', color=colors[length], label=f'Length {length}')
        ax2.fill_between(p_w_vals, np.clip(mean_auc - std_auc, 0, 1), np.clip(mean_auc + std_auc, 0, 1),
                         color=colors[length], alpha=0.2)

    error_type = NOISE_LABELS.get(noise_type, noise_type)

    for ax, metric, ylabel in [
        (ax1, f'SYMBA Max F1 vs Word Error ($p_w$) due to Symbol {error_type}',  'Mean Maximum F1-Score'),
        (ax2, f'SYMBA PR-AUC vs Word Error ($p_w$) due to Symbol {error_type}',  'Mean Area Under PR Curve')
    ]:
        ax.set_title(metric, fontsize=fontsize_title)
        ax.set_xlabel('$p_w$', fontsize=fontsize_label)
        ax.set_ylabel(ylabel, fontsize=fontsize_label)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.02, 0.95])
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=fontsize_legend)

    plt.tight_layout()

    if save_fig:
        _savefig(filename)
        print(f"Plot saved successfully as: {filename}")

    plt.show()


def plot_symba_vs_ps_and_pw_combined(aggregated_results_dict,
                                      noise_types=('substitution', 'deletion'),
                                      save_fig=False, filename="symba_combined.svg",
                                      fontsize_title=22, fontsize_label=20,
                                      fontsize_ticks=20, fontsize_legend=18):
    """
    One row per noise type, four panels per row: max-F1 and PR-AUC vs p_s and vs p_w.

    aggregated_results_dict: dict keyed by noise_type, each value is aggregated_results
    e.g. {'substitution': aggregated_results_sub, 'deletion': aggregated_results_del}
    """
    colors  = {2: 'blue', 3: 'orange', 4: 'green'}
    lengths = [2, 3, 4]

    n_rows = len(noise_types)
    n_cols = 4  # ps F1 | ps AUC | pw F1 | pw AUC

    panel_size = 5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_size * n_cols, panel_size * n_rows),
                             squeeze=False)

    # make panels square
    for ax in axes.flat:
        ax.set_aspect('auto')
    fig.set_size_inches(panel_size * n_cols, panel_size * n_rows)

    col_titles = [
        r'Max F1 vs $p_s$',
        r'PR-AUC vs $p_s$',
        r'Max F1 vs $p_w$',
        r'PR-AUC vs $p_w$',
    ]

    for row, noise_type in enumerate(noise_types):
        aggregated_data = aggregated_results_dict[noise_type]
        error_type      = NOISE_LABELS.get(noise_type, noise_type)

        for length in lengths:
            p_s_vals = aggregated_data[length]['p_s_values']
            p_w_vals = 1 - (1 - p_s_vals)**length

            mean_f1  = np.array(aggregated_data[length]['mean_f1'])
            std_f1   = np.array(aggregated_data[length]['std_f1'])
            mean_auc = np.array(aggregated_data[length]['mean_auc'])
            std_auc  = np.array(aggregated_data[length]['std_auc'])

            label = f'Length {length}'
            c     = colors[length]

            # col 0: F1 vs ps
            axes[row, 0].plot(p_s_vals, mean_f1, marker='o', color=c, label=label)
            axes[row, 0].fill_between(p_s_vals,
                np.clip(mean_f1 - std_f1, 0, 1), np.clip(mean_f1 + std_f1, 0, 1),
                color=c, alpha=0.2)

            # col 1: AUC vs ps
            axes[row, 1].plot(p_s_vals, mean_auc, marker='s', linestyle='--', color=c, label=label)
            axes[row, 1].fill_between(p_s_vals,
                np.clip(mean_auc - std_auc, 0, 1), np.clip(mean_auc + std_auc, 0, 1),
                color=c, alpha=0.2)

            # col 2: F1 vs pw
            axes[row, 2].plot(p_w_vals, mean_f1, marker='o', color=c, label=label)
            axes[row, 2].fill_between(p_w_vals,
                np.clip(mean_f1 - std_f1, 0, 1), np.clip(mean_f1 + std_f1, 0, 1),
                color=c, alpha=0.2)

            # col 3: AUC vs pw
            axes[row, 3].plot(p_w_vals, mean_auc, marker='s', linestyle='--', color=c, label=label)
            axes[row, 3].fill_between(p_w_vals,
                np.clip(mean_auc - std_auc, 0, 1), np.clip(mean_auc + std_auc, 0, 1),
                color=c, alpha=0.2)

        # format all panels in this row
        for col, (ax, xlabel) in enumerate(zip(axes[row],
                                               [r'$p_s$', r'$p_s$', r'$p_w$', r'$p_w$'])):
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlabel(xlabel, fontsize=fontsize_label)
            ax.tick_params(axis='both', labelsize=fontsize_ticks)
            ax.grid(True, linestyle=':', alpha=0.7)
            legend = ax.legend(fontsize=fontsize_legend)
            legend.get_title().set_fontsize(fontsize_legend)

            if col in [2, 3]:
                ax.set_xlim([-0.02, 0.95])

            # ylabel only on left panels
            if col == 0:
                ax.set_ylabel('Mean Maximum F1-Score', fontsize=fontsize_label)
            elif col == 1:
                ax.set_ylabel('Mean Area Under PR Curve', fontsize=fontsize_label)
            elif col == 2:
                ax.set_ylabel('Mean Maximum F1-Score', fontsize=fontsize_label)
            elif col == 3:
                ax.set_ylabel('Mean Area Under PR Curve', fontsize=fontsize_label)

            # column titles only on top row
            if row == 0:
                ax.set_title(col_titles[col], fontsize=fontsize_title)

        # noise type row label on the left
        axes[row, 0].annotate(error_type,
                              xy=(0, 0.5), xytext=(-0.35, 0.5),
                              xycoords='axes fraction', textcoords='axes fraction',
                              fontsize=fontsize_title, fontweight='bold',
                              va='center', ha='center', rotation=90)

    plt.tight_layout()

    if save_fig:
        _savefig(filename)
        print(f"Saved: {filename}")

    plt.show()


def plot_diffs_analytic_shuffle(diffs, alph_length, gamma, method="std", save_fig = False, output_dir="outputs/figures/comparison_shuffle_analytic"):
    """
    Convergence of the shuffle null model to the analytical expected probability:
    |p_exp - p_shuffle| (mean ± std, or median ± MAD) vs the number of shuffles B,
    one figure per (noise-to-signal ratio, tuple length).

    diffs : dict[B, dict[(nsr, length), np.ndarray]], as returned by
        compute_diffs_by_length_vs_B in the notebook.
    """
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)

    # Helper to extract stats
    def get_stats(diffs_by_B):
        res = {}
        for B, diffs_dict in diffs_by_B.items():
            for (param, l_val), diff_values in diffs_dict.items():
                center = np.median(diff_values) if method == "mad" else np.mean(diff_values)
                spread = np.median(np.abs(diff_values - center)) if method == "mad" else np.std(diff_values)
                res.setdefault((param, l_val), {"B": [], "center": [], "spread": []})
                res[(param, l_val)]["B"].append(B)
                res[(param, l_val)]["center"].append(center)
                res[(param, l_val)]["spread"].append(spread)
        return res

    stats_shuffle = get_stats(diffs)

    for (param, l_val) in stats_shuffle.keys():
        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=150)

        # Plotting the difference
        shuffle_data = stats_shuffle[(param, l_val)]
        idx_s = np.argsort(shuffle_data["B"])
        ax.errorbar(np.array(shuffle_data["B"])[idx_s], np.array(shuffle_data["center"])[idx_s],
                                yerr=np.array(shuffle_data["spread"])[idx_s], fmt='s--', color='#CD5C5C')


        ax.set_xscale("log")
        ax.set_xlabel(r"$B$ (shuffles)")
        ax.set_ylabel(r"$|p_{exp} - p_{shuffle}|$")
        ax.set_title(f"{l_val}-tuples (NSR={param})")
        ax.grid(True, alpha=0.3)

        if save_fig:
            plt.savefig(f"{output_dir}/layout_comp_rns{param}_l{l_val}_alphlength{alph_length}_gamma{gamma}.svg", bbox_inches='tight')
        plt.show()


def plot_summary_across_nsr_four_methods(
    summary_data, lengths=[2, 3, 4],
    metrics=('f1', 'auc_pr', 'auc_roc'),
    metric_labels=('Max F1-Score', 'PR-AUC', 'ROC-AUC'),
    legend_loc = 'outside', figsize = (6, 4),
    save_fig=False, filename="symba_vs_shuffle_nsr.svg",
    fontsize_title=16, fontsize_label=14, fontsize_ticks=12, fontsize_legend=12
):
    """
    Plot mean ± SEM for any set of scalar metrics stored in summary_data,
    one column per motif length, one row per metric.

    Parameters
    ----------
    summary_data : dict
        Output of evaluate_symba_vs_shuffle_across_nsr_full_curves.
        Must contain 'nsr_values' and 'n_runs'.
    lengths : list of int
    metrics : tuple of str
        Keys to plot, e.g. ('f1', 'auc_pr', 'auc_roc', 'precision', 'recall').
    metric_labels : tuple of str
        Display names for each metric (same order as metrics).
    """
    method_styles = {
        'symba':     {'color': (106/255, 61/255, 154/255), 'ls': '-',  'label': 'SYMBA'},           # Solid
        'shuffle_p': {'color': (235/255, 108/255, 10/255), 'ls': '--', 'label': 'shuffle, p-value'}, # Dashed
        'shuffle_z': {'color': (0/255, 114/255, 178/255),  'ls': '-.', 'label': 'shuffle, z-score'}, # Dash-dot
        'batch_z':   {'color': (27/255, 227/255, 156/255), 'ls': ':',  'label': 'batch, z-score'},   # Dotted
        }

    nsr_values = summary_data['nsr_values']
    n_runs     = summary_data['n_runs']
    n_metrics  = len(metrics)
    n_lengths  = len(lengths)

    fig, axes = plt.subplots(n_metrics, n_lengths,
                             figsize=(figsize[0] * n_lengths, figsize[1] * n_metrics),
                             squeeze=False)

    for col, length in enumerate(lengths):
        for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]

            for method, style in method_styles.items():
                mean_vals = np.array(summary_data[method][length][f'mean_{metric}'])
                sem_vals  = np.array(summary_data[method][length][f'std_{metric}'])/np.sqrt(summary_data['n_runs'])
                #sem_vals  = np.array(summary_data[method][length][f'sems_{metric}'])

                #ax.plot(nsr_values, mean_vals,
                #        marker=style['marker'], linestyle=style['linestyle'],
                #        color=style['color'], label=style['label'])
                ax.plot(nsr_values, mean_vals,
                                        linestyle=style['ls'], lw=3,
                                        color=style['color'], label=style['label'])
                ax.fill_between(nsr_values,
                                np.clip(mean_vals - sem_vals, 0, 1),
                                np.clip(mean_vals + sem_vals, 0, 1),
                                color=style['color'], alpha=0.2)

            ax.set_title(f'Length {length} — {metric_label}', fontsize=fontsize_title)
            ax.set_xlabel(r'$r_{\mathrm{ns}}$', fontsize=fontsize_label)
            ax.set_ylabel(metric_label, fontsize=fontsize_label)
            ax.set_ylim([-0.05, 1.05])
            #ax.set_xscale('log')
            ax.tick_params(axis='both', labelsize=fontsize_ticks)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(fontsize=fontsize_legend)
            if legend_loc == 'outside':
                ax.legend(fontsize=fontsize_legend,
                          bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend(fontsize=fontsize_legend,
                          loc=legend_loc)

    plt.suptitle(
        r'SYMBA vs shuffle vs batch across $r_{\mathrm{ns}}$' + f' (mean ± SEM, n={n_runs})',
        fontsize=fontsize_title + 2, y=1.02
    )


    plt.tight_layout()

    if save_fig:
        _savefig(filename)
        print(f"Plot saved as: {filename}")

    plt.show()


CURVE_GRID = np.linspace(0, 1, 200)


def _mean_curve(x_curves, y_curves, curve_type):
    """
    Averages per-run PR (curve_type="pr") or ROC curves on the common grid CURVE_GRID.

    Returns (mean, sem) arrays, or (None, None) if no run has a curve.
    """
    valid = [(x, y) for x, y in zip(x_curves, y_curves)
             if x is not None and y is not None]
    if not valid:
        return None, None

    y_interp = []
    for x_curve, y_curve in valid:
        if curve_type == "pr":
            y_interp.append(np.interp(CURVE_GRID, x_curve[::-1], y_curve[::-1]))
        else:
            y_interp.append(np.interp(CURVE_GRID, x_curve, y_curve))

    y_interp = np.array(y_interp)
    return np.mean(y_interp, axis=0), np.std(y_interp, axis=0) / np.sqrt(len(valid))


def slim_summary_data(summary_data):
    """
    Small version of the four-method results, enough to make all their figures.

    Keeps every scalar metric (per run and aggregated) and replaces the per-run
    PR/ROC curves and y_true/y_scores vectors (hundreds of MB) with the mean and
    SEM of the PR and ROC curves on CURVE_GRID, computed exactly as
    plot_roc_or_pr_curves_from_summary does.
    """
    curve_keys = {"pr": ("runs_recall_curve", "runs_precision_curve"),
                  "roc": ("runs_fpr_curve", "runs_tpr_curve")}
    slim = {}
    for key, value in summary_data.items():
        if not isinstance(value, dict):
            slim[key] = value
            continue
        slim[key] = {}
        for length, res in value.items():
            small = {k: v for k, v in res.items()
                     if not (k.startswith("runs_") and (k.endswith("_curve") or k.startswith("runs_y_")))}
            for curve_type, (x_key, y_key) in curve_keys.items():
                means, sems = [], []
                for x_curves, y_curves in zip(res[x_key], res[y_key]):
                    m, s = _mean_curve(x_curves, y_curves, curve_type)
                    means.append(m)
                    sems.append(s)
                small[f"{curve_type}_curve_mean"] = means
                small[f"{curve_type}_curve_sem"] = sems
            slim[key][length] = small
    return slim


def plot_roc_or_pr_curves_from_summary(
    summary_data,
    curve_type="pr",
    show_lengths=[2, 3, 4],
    save_fig=False,
    save_dir="outputs/figures/roc_pr_curves",
    alph_length=20, num_2words=10, num_3words=10, num_4words=10,
    rep_2words=10, rep_3words=10, rep_4words=10,
    gamma=1, ordered_words=0,
    figsize=(7, 6),
    fontsize_title=16, fontsize_label=16, fontsize_ticks=16, fontsize_legend=13,
    legend_loc='lower left',
    show_sem=False
):
    """
    Plot PR or ROC curves from summary_data (output of evaluate_symba_vs_shuffle_across_nsr_full_curves).
    One figure per (length, nsr), with one mean curve per method (curves of the
    individual runs are interpolated on a common grid and averaged).

    Parameters
    ----------
    curve_type : "pr" or "roc"
    show_lengths : list of motif lengths to plot
    figsize : tuple, figure size in inches
    legend_loc : str, legend location ('lower left', 'upper left', etc.)
                 use 'outside' to place legend to the right of the plot
    save_fig : bool
    save_dir : str, directory for saved figures
    show_sem : bool
        If True, also shade ± the standard error of the mean across runs
        (usually too narrow to be visible, hence off by default).
    """
    from matplotlib.lines import Line2D


    method_styles = {
    'symba':     {'color': (106/255, 61/255, 154/255), 'ls': '-',  'label': 'SYMBA'},           # Solid
    'shuffle_p': {'color': (235/255, 108/255, 10/255), 'ls': '--', 'label': 'shuffle, p-value'}, # Dashed
    'shuffle_z': {'color': (0/255, 114/255, 178/255),  'ls': '-.', 'label': 'shuffle, z-score'}, # Dash-dot
    'batch_z':   {'color': (27/255, 227/255, 156/255), 'ls': ':',  'label': 'batch, z-score'},   # Dotted
    }

    nsr_values = summary_data['nsr_values']
    n_runs     = summary_data['n_runs']

    if curve_type == "pr":
        x_key, y_key  = 'runs_recall_curve', 'runs_precision_curve'
        xlabel, ylabel = 'Recall', 'Precision'
        baseline       = None
    else:
        x_key, y_key  = 'runs_fpr_curve', 'runs_tpr_curve'
        xlabel, ylabel = 'False Positive Rate', 'True Positive Rate'
        baseline       = ([0, 1], [0, 1])

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)

    for length in show_lengths:
        for nsr_idx, nsr in enumerate(nsr_values):

            fig, ax = plt.subplots(figsize=figsize)
            custom_lines = []

            for method, style in method_styles.items():
                color = style['color']
                ls = style['ls']

                res = summary_data[method][length]
                if f"{curve_type}_curve_mean" in res:
                    # slim summary (see slim_summary_data): curves already averaged
                    y_mean = res[f"{curve_type}_curve_mean"][nsr_idx]
                    y_sem  = res[f"{curve_type}_curve_sem"][nsr_idx]
                else:
                    y_mean, y_sem = _mean_curve(res[x_key][nsr_idx], res[y_key][nsr_idx], curve_type)
                if y_mean is None:
                    continue

                x_grid = CURVE_GRID
                ax.plot(x_grid, y_mean, color=color, ls=ls, linewidth=3)
                if show_sem:
                    ax.fill_between(x_grid, y_mean - y_sem, y_mean + y_sem,
                                    color=color, alpha=0.2)

                custom_lines.append(Line2D(
                    [0], [0], color=color, linestyle=ls, lw=3,
                    label=f"{style['label']}"
                ))

            if baseline is not None:
                ax.plot(baseline[0], baseline[1], 'k--', linewidth=2)
                custom_lines.append(Line2D(
                    [0], [0], color='k', lw=2, linestyle='--', label='Random'
                ))

            ax.set_xlabel(xlabel, fontsize=fontsize_label)
            ax.set_ylabel(ylabel, fontsize=fontsize_label)

            ax.set_title(
                fr'$r_{{\mathrm{{ns}}}}={nsr}$',
                fontsize=fontsize_title
            )
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(axis='both', labelsize=fontsize_ticks)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(False)

            if legend_loc == 'outside':
                ax.legend(handles=custom_lines, fontsize=fontsize_legend,
                          bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend(handles=custom_lines, fontsize=fontsize_legend,
                          loc=legend_loc)

            plt.tight_layout()

            if save_fig:
                fname = (
                    f"{save_dir}/fourmethods_{curve_type}_"
                    f"motif{length}_gamma{gamma}_"
                    f"alph{alph_length}_"
                    f"n2{num_2words}_n3{num_3words}_n4{num_4words}_"
                    f"m2{rep_2words}_m3{rep_3words}_m4{rep_4words}_"
                    f"rns{nsr}_ord{ordered_words}.svg"
                )
                plt.savefig(fname, format='svg', bbox_inches='tight')
                print(f"Saved: {fname}")

            plt.show()


def plot_runtime_scaling(time_data, nsr_values, log_y=True, save_fig=False, filename="runtime_comparison.svg",
                         figsize=(5, 5), linewidth=2.5,
                         title_fontsize=14, label_fontsize=17, legend_fontsize=16, tick_fontsize=15):
    """
    Plots the mean execution time of symba vs shuffle with standard deviation error bands.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    color_symba = (106/255, 61/255, 154/255)
    color_shuffle =  (235/255, 108/255, 10/255)

    # SYMBA Data
    mean_symba = np.array(time_data['symba']['mean'])
    std_symba  = np.array(time_data['symba']['std'])

    ax.plot(nsr_values, mean_symba, marker='o', color=color_symba,
            linewidth=linewidth, label='SYMBA')
    ax.fill_between(nsr_values, np.maximum(mean_symba - std_symba, 0), mean_symba + std_symba,
                    color=color_symba, alpha=0.2, edgecolor='none')

    # SHUFFLE Data
    mean_shuffle = np.array(time_data['shuffle']['mean'])
    std_shuffle  = np.array(time_data['shuffle']['std'])

    ax.plot(nsr_values, mean_shuffle, marker='s', linestyle='--', color=color_shuffle,
            linewidth=linewidth, label='shuffle')
    ax.fill_between(nsr_values, np.maximum(mean_shuffle - std_shuffle, 0), mean_shuffle + std_shuffle,
                    color=color_shuffle, alpha=0.2, edgecolor='none')

    # Formatting
    ax.set_xlabel(r'$r_{\mathrm{ns}} \propto L$', fontsize=label_fontsize)
    ax.set_ylabel('Runtime (s)',                     fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    if log_y:
        ax.set_yscale('log')
        ax.set_xscale('log')

    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if save_fig:
        _savefig(filename)

    plt.show()


def plot_motifs_vs_ps_full(save_dict, save_fig=False, filename="motifs_vs_ps_full.svg",
                            ylim=None, plot_noise_types=None,
                            fontsize_title=22, fontsize_label=22,
                            fontsize_ticks=20, fontsize_legend=20):
    """
    Markov-sequence control: number of tuples scoring above the threshold vs
    the symbol-noise probability p_s. One row per noise type, one column per
    Markov order, one curve per tuple size (mean ± std over sequences).

    save_dict : the dictionary saved by the Markov section of the notebook.
    ylim : None, a (ymin, ymax) tuple, or {(noise_type, tuple_length): (ymin, ymax)}.
    plot_noise_types : optional subset of noise types to plot.
    """
    results       = save_dict['results']
    ps_values     = save_dict['ps_values']
    sizes         = save_dict['sizes']
    threshold     = save_dict['threshold']
    noise_types   = save_dict['noise_types']
    tuple_lengths = save_dict['tuple_lengths']
    alph_length   = save_dict['alph_length']

    # filter noise types if requested
    if plot_noise_types is not None:
        noise_types = [n for n in noise_types if n in plot_noise_types]

    colors  = plt.cm.tab10(np.linspace(0, 0.6, len(sizes)))
    markers = ['o', 's', '^', 'D', 'v']

    n_rows = len(noise_types)
    n_cols = len(tuple_lengths)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5 * n_rows),
                             squeeze=False)

    for row, noise_type in enumerate(noise_types):
        for col, tuple_length in enumerate(tuple_lengths):
            ax = axes[row, col]
            all_summaries = results[tuple_length][noise_type]

            for i, size in enumerate(sizes):
                means = np.array([all_summaries[p_s][size]['mean'] for p_s in ps_values])
                stds  = np.array([all_summaries[p_s][size]['std']  for p_s in ps_values])

                ax.plot(ps_values, means,
                        color=colors[i], marker=markers[i % len(markers)],
                        linestyle='-', label=f'Size {size}')
                ax.fill_between(ps_values,
                                np.clip(means - stds, 0, None),
                                means + stds,
                                color=colors[i], alpha=0.2)

            ax.set_title(f'Markov order {tuple_length - 2}', fontsize=fontsize_title)
            ax.set_xlabel(r'$p_s$', fontsize=fontsize_label)
            ax.set_ylabel(f'# motifs > {threshold}', fontsize=fontsize_label)
            ax.tick_params(axis='both', labelsize=fontsize_ticks)
            legend = ax.legend(title='Tuple size', fontsize=fontsize_legend)
            legend.get_title().set_fontsize(fontsize_legend)
            ax.grid(True, linestyle=':', alpha=0.7)

            # ylim: None, a single tuple (0, 20), or a dict {(noise_type, tuple_length): (0, 20)}
            if isinstance(ylim, dict):
                panel_ylim = ylim.get((noise_type, tuple_length), None)
                if panel_ylim is not None:
                    ax.set_ylim(panel_ylim)
            elif ylim is not None:
                ax.set_ylim(ylim)

        axes[row, 0].annotate(
            NOISE_LABELS.get(noise_type, noise_type.capitalize()),
            xy=(0, 0.5), xytext=(-0.35, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=fontsize_title + 2, fontweight='bold',
            va='center', ha='center', rotation=90
        )

    fig.suptitle(f'Detected motifs vs noise — alphabet size N={alph_length}',
                 fontsize=fontsize_title + 2, y=1.01)

    plt.tight_layout()

    if save_fig:
        _savefig(filename)
        print(f"Saved as {filename}")

    plt.show()


def plot_score_histogram(scores, threshold=0.6, title=None, bins=30, ylabel="counts",
                         save_fig=False, filename="score_histogram.svg"):
    """
    Histogram (log counts) of the SYMBA scores of one tuple length; bars at or
    above ``threshold`` are coloured as significant. Prints the number of tuples
    with a score at or above ``threshold``.

    scores : dict[tuple, float], e.g. js_per_size_dict[2].
    """
    count_sign_thr = sum(score >= threshold for score in scores.values())
    print("count_sign_thr", count_sign_thr)

    weights = np.array(list(scores.values())).reshape(-1, 1)

    plt.figure(figsize=(5, 3))
    _, bin_edges, patches = plt.hist(weights, bins=bins, density=False)
    plt.yscale('log')

    color_line = (27/255, 27/255, 46/255, 0.8)
    ymin, ymax = plt.ylim()
    plt.ylim([ymin, ymax])
    plt.vlines(threshold, ymin=ymin, ymax=ymax, colors=color_line, linestyles='dashed')

    # label positions chosen for threshold = 0.6
    plt.text(0.36, plt.ylim()[1] / 1.3, "non-significant", fontsize=16, ha='center', color=(120/255, 27/255, 182/255))
    plt.text(0.735, plt.ylim()[1] / 1.3, "significant", fontsize=16, ha='center', color=(255/255, 51/255, 51/255))

    nonsignificant_color = (120/255, 27/255, 182/255, 0.8)
    significant_color = (255/255, 51/255, 51/255, 0.8)
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        patch.set_facecolor(significant_color if left_edge >= threshold else nonsignificant_color)

    plt.xlabel(r"$d_{\mathrm{JS}}$", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if title:
        plt.title(title, fontsize=16)

    if save_fig:
        _savefig(filename)
        print(f"Saved: {filename}")

    plt.show()
