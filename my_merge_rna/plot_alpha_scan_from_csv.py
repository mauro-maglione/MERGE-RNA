# %%
#!/usr/bin/env python3
"""
Slide-ready plots of an alpha scan, read straight from the CSVs that run_compare_alpha_scan.py and
run_checkpoint_crossval.py already wrote -- no fitting, no merge_rna import, just pandas + matplotlib.

Plots (--plots):
  scan      OUTPUT_DIR/scan_results.csv               log-likelihood, KL divergence, total loss (training data)
  crossval  OUTPUT_DIR/crossval/crossval_results.csv  log-likelihood on the NEW (validation) experiment
  holdout   OUTPUT_DIR/holdout_results.csv            log-likelihood on masked-out / all sites (masked runs)

Method selection (--methods): any subset of the methods present in the CSVs (merge_rna, maxent_chi2,
maxent_binomial). Each method keeps the same colour + marker in every figure whatever subset is chosen.

For the held-out metrics (crossval: log_likelihood_new / _new_masked_out; holdout: _masked_out) the
best alpha of each method is marked with a star and its value is printed -- the training-side metrics
(scan) are not marked, since their optimum is trivially alpha -> 0 / infinity.

Examples (from anywhere; paths resolve against the repo root):
    python my_merge_rna/plot_alpha_scan_from_csv.py
    python my_merge_rna/plot_alpha_scan_from_csv.py --run less_coverage_10e5 --plots crossval --methods maxent_chi2
    python my_merge_rna/plot_alpha_scan_from_csv.py --run halfmask_test --plots scan holdout --format pdf svg png
    python my_merge_rna/plot_alpha_scan_from_csv.py --plots scan --metrics total_loss --theme dark --transparent
    python my_merge_rna/plot_alpha_scan_from_csv.py --layout row      # all metrics of a plot side by side
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: this script never has a display attached
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(repo_root, 'fits_paper', 'designed_sequence', 'outputs')
RUN_PREFIX = 'compare_alpha_scan_pop80_'

# %%
# =============================================================================
# Configuration
# =============================================================================
DEFAULT_RUN = 'halfmask_test'  # the subfix_out of the run to read (or pass a full directory path)

# Mirrors METHOD_LABELS in run_compare_alpha_scan.py (not imported: that module pulls in merge_rna/ViennaRNA).
METHOD_LABELS = {
    'merge_rna': 'merge-rna (native)',
    'maxent_chi2': 'maxent (chi-squared)',
    'maxent_binomial': 'maxent (binomial)',
}

# Colour follows the method, never its position in the current selection. Slots 1-3 of the reference
# categorical palette; marker shape is the second channel so identity is never colour alone.
THEMES = {
    'light': dict(surface='#ffffff', ink='#0b0b0b', ink2='#52514e', grid='#e3e2dd',
                  series=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']),
    'dark':  dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', grid='#3a3a37',
                  series=['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9']),
}
METHOD_SLOT = {'merge_rna': 0, 'maxent_chi2': 1, 'maxent_binomial': 2}
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

# (label, held_out): held_out metrics get a best-alpha star. Unknown columns fall back to their raw name.
METRIC_INFO = {
    'log_likelihood':               ('Log-likelihood (training)', False),
    'kl_divergence':                ('KL divergence', False),
    'total_loss':                   ('Total loss (NLL + $\\alpha$ KL)', False),
    'log_likelihood_new':           ('Log-likelihood (validation data)', True),
    'log_likelihood_combined':      ('Log-likelihood (training + validation)', False),
    'log_likelihood_new_trained':   ('Log-likelihood (validation, trained sites)', False),
    'log_likelihood_new_masked_out': ('Log-likelihood (validation, masked-out sites)', True),
    'log_likelihood_all_sites':     ('Log-likelihood (all sites)', False),
    'log_likelihood_masked_out':    ('Log-likelihood (masked-out sites)', True),
}

PLOTS = {
    'scan':     os.path.join('scan_results.csv'),
    'crossval': os.path.join('crossval', 'crossval_results.csv'),
    'holdout':  os.path.join('holdout_results.csv'),
}
KEY_COLS = {'method', 'alpha'}


# %%
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default=DEFAULT_RUN,
                   help=f"subfix_out of the run (dir '{RUN_PREFIX}<run>' under {os.path.relpath(RUNS_ROOT, repo_root)}) "
                        f"or a path to a run directory (default: %(default)s)")
    p.add_argument('--plots', nargs='+', choices=list(PLOTS), default=['scan', 'crossval'],
                   help='which CSV/plot families to make (default: scan crossval)')
    p.add_argument('--methods', nargs='+', default=None, metavar='METHOD',
                   help='methods to include, in legend order (default: every method found in the CSV)')
    p.add_argument('--metrics', nargs='+', default=None, metavar='COLUMN',
                   help='CSV columns to plot, e.g. total_loss log_likelihood_new (default: all of them)')
    p.add_argument('--layout', choices=['separate', 'row'], default='separate',
                   help="'separate': one figure per metric (drop straight into a slide); "
                        "'row': all metrics of a plot side by side in one figure")
    p.add_argument('--format', nargs='+', default=['png'], choices=['png', 'pdf', 'svg'],
                   help='output format(s); pdf/svg stay vector in PowerPoint/Keynote/Beamer')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--figsize', nargs=2, type=float, default=[10, 5.6], metavar=('W', 'H'),
                   help='inches, per panel (default: 10 5.6, i.e. 16:9)')
    p.add_argument('--theme', choices=list(THEMES), default='light', help='match your slide background')
    p.add_argument('--transparent', action='store_true', help='transparent figure background')
    p.add_argument('--no-title', action='store_true', help='omit the per-panel title (the slide supplies it)')
    p.add_argument('--no-mark-best', action='store_true', help='do not star the best alpha of held-out metrics')
    p.add_argument('--out-dir', default=None, help='default: <run dir>/slides')
    return p.parse_args()


def resolve_run_dir(run):
    if os.path.isdir(run):
        return run
    candidate = os.path.join(RUNS_ROOT, RUN_PREFIX + run)
    if os.path.isdir(candidate):
        return candidate
    available = sorted(d[len(RUN_PREFIX):] for d in os.listdir(RUNS_ROOT) if d.startswith(RUN_PREFIX))
    raise SystemExit(f"No run directory {candidate!r}. Available --run values: {available}")


def apply_style(theme):
    plt.rcParams.update({
        'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 20,
        'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16,
        'figure.facecolor': theme['surface'], 'axes.facecolor': theme['surface'],
        'savefig.facecolor': theme['surface'],
        'text.color': theme['ink'], 'axes.labelcolor': theme['ink'], 'axes.titlecolor': theme['ink'],
        'xtick.color': theme['ink2'], 'ytick.color': theme['ink2'],
        'axes.edgecolor': theme['ink2'], 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': theme['grid'], 'grid.linewidth': 1.0,
        'axes.axisbelow': True, 'legend.frameon': False,
        'lines.linewidth': 2.5, 'lines.markersize': 8,
    })


def method_style(method, theme, extra_methods):
    """(colour, marker) fixed per method: known methods keep their slot, unknown ones get the next free one."""
    slot = METHOD_SLOT.get(method)
    if slot is None:
        slot = len(METHOD_SLOT) + extra_methods.index(method)
    return theme['series'][slot % len(theme['series'])], MARKERS[slot % len(MARKERS)]


def select_methods(df, requested, csv_path):
    present = list(dict.fromkeys(df['method']))
    present.sort(key=lambda m: (list(METHOD_SLOT).index(m) if m in METHOD_SLOT else len(METHOD_SLOT), m))
    if requested is None:
        return present
    kept = [m for m in requested if m in present]
    missing = [m for m in requested if m not in present]
    if not kept:
        raise SystemExit(f"None of --methods {requested} are in {csv_path}. Methods there: {present}")
    if missing:
        print(f"  note: {missing} not in {os.path.basename(csv_path)}; plotting {kept}")
    return kept


def select_metrics(df, requested):
    cols = [c for c in df.columns if c not in KEY_COLS and pd.api.types.is_numeric_dtype(df[c])]
    return cols if requested is None else [c for c in requested if c in cols]


def is_higher_better(metric):
    return metric.startswith('log_likelihood')


def draw_panel(ax, df, metric, methods, theme, extra_methods, mark_best, title):
    label, held_out = METRIC_INFO.get(metric, (metric, False))
    star_handles = []
    for method in methods:
        sub = df[df['method'] == method].dropna(subset=[metric]).sort_values('alpha')
        if sub.empty:
            continue
        color, marker = method_style(method, theme, extra_methods)
        ax.plot(sub['alpha'], sub[metric], marker=marker, color=color, label=METHOD_LABELS.get(method, method))
        if mark_best and held_out:
            best = sub.loc[sub[metric].idxmax() if is_higher_better(metric) else sub[metric].idxmin()]
            ax.plot(best['alpha'], best[metric], marker='*', markersize=20, color=color,
                    markeredgecolor=theme['surface'], markeredgewidth=1.5, linestyle='', zorder=5)
            star_handles.append((Line2D([], [], marker='*', markersize=16, linestyle='', color=color),
                                 f"best $\\alpha$ = {best['alpha']:.3g}"))
    ax.set_xscale('symlog', linthresh=1e-2)
    ax.margins(y=0.08)
    ax.ticklabel_format(axis='y', useOffset=False, style='plain')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(label)
    if title:
        ax.set_title(label)
    handles, labels = ax.get_legend_handles_labels()
    handles += [h for h, _ in star_handles]
    labels += [l for _, l in star_handles]
    if len(handles) > 1:  # a single series needs no legend box
        ax.legend(handles, labels, loc='best')


def shown(path):
    return os.path.relpath(path, repo_root) if os.path.abspath(path).startswith(repo_root + os.sep) else path


def save(fig, base, formats, dpi, transparent):
    for fmt in formats:
        path = f'{base}.{fmt}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
        print(f"  saved {shown(path)}")
    plt.close(fig)


def summarize(plot, df, metrics, methods):
    rows = []
    for metric in metrics:
        for method in methods:
            sub = df[df['method'] == method].dropna(subset=[metric])
            if sub.empty:
                continue
            best = sub.loc[sub[metric].idxmax() if is_higher_better(metric) else sub[metric].idxmin()]
            rows.append(dict(plot=plot, metric=metric, method=method,
                             best='max' if is_higher_better(metric) else 'min',
                             alpha_at_best=best['alpha'], value_at_best=best[metric],
                             value_at_min_alpha=sub.loc[sub['alpha'].idxmin(), metric],
                             value_at_max_alpha=sub.loc[sub['alpha'].idxmax(), metric], n_alphas=len(sub)))
    return rows


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run)
    out_dir = args.out_dir or os.path.join(run_dir, 'slides')
    theme = THEMES[args.theme]
    apply_style(theme)
    suffix = '__' + '+'.join(args.methods) if args.methods else ''
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run: {shown(run_dir)}  ->  {shown(out_dir)}")

    summary, found_metrics = [], set()
    for plot in args.plots:
        csv_path = os.path.join(run_dir, PLOTS[plot])
        if not os.path.exists(csv_path):
            hint = ' (run my_merge_rna/run_checkpoint_crossval.py first)' if plot == 'crossval' else ''
            print(f"[{plot}] skipped: {shown(csv_path)} not found{hint}")
            continue
        df = pd.read_csv(csv_path)
        methods = select_methods(df, args.methods, csv_path)
        extra_methods = sorted(set(df['method']) - set(METHOD_SLOT))
        metrics = select_metrics(df, args.metrics)
        found_metrics.update(metrics)
        if not metrics:
            print(f"[{plot}] no requested metrics in {os.path.basename(csv_path)}")
            continue
        print(f"[{plot}] methods={methods} metrics={metrics}")
        mark_best = not args.no_mark_best
        title = not args.no_title
        w, h = args.figsize

        if args.layout == 'row':
            fig, axes = plt.subplots(1, len(metrics), figsize=(w * len(metrics), h), squeeze=False)
            for ax, metric in zip(axes[0], metrics):
                draw_panel(ax, df, metric, methods, theme, extra_methods, mark_best, title)
            fig.tight_layout()
            save(fig, os.path.join(out_dir, f'{plot}_all{suffix}'), args.format, args.dpi, args.transparent)
        else:
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(w, h))
                draw_panel(ax, df, metric, methods, theme, extra_methods, mark_best, title)
                fig.tight_layout()
                save(fig, os.path.join(out_dir, f'{plot}_{metric}{suffix}'), args.format, args.dpi, args.transparent)
        summary += summarize(plot, df, metrics, methods)

    if args.metrics:
        unknown = [m for m in args.metrics if m not in found_metrics]
        if unknown:
            print(f"warning: --metrics {unknown} not found in any selected CSV")
    if not summary:
        raise SystemExit("Nothing plotted.")

    with pd.option_context('display.width', 200, 'display.max_columns', None, 'display.float_format', '{:.6g}'.format):
        print("\nNumbers behind the plots (best = max for log-likelihoods, min otherwise):")
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()

# %%
