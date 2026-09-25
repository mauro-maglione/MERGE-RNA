# %%
#!/usr/bin/env python3
"""
Score already-fitted lambda_sc checkpoints (from a run of run_compare_alpha_scan.py) against a
NEW experiment, without re-optimizing anything.

For every (method, alpha) checkpoint under SOURCE_OUTPUT_DIR/checkpoints/:
  - if a mask.txt was saved for that run: report log-likelihood at the trained (mask=True) and
    masked-out (mask=False) positions of the NEW experiment.
  - if no mask.txt exists: report log-likelihood of the NEW experiment alone, AND of the NEW +
    original-training experiment pooled together (counts summed position-wise), since there's
    no position split to fall back on for an independent check.

Driven entirely by artifacts the source run already writes to its OUTPUT_DIR (checkpoints/*.pkl,
exp_df.csv + exp_meta.json, mask.txt if present, phys_params_only.txt) -- not by the CURRENT
config in run_compare_alpha_scan.py, since that may have changed since the source run finished.

Run from the repo root:
    python my_merge_rna/run_checkpoint_crossval.py
"""
import os
import sys
import json
import glob
import re
import pickle

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: this script never has a display attached
import matplotlib.pyplot as plt

from merge_rna import Experiment
from my_merge_rna.run_compare_alpha_scan import (
    load_reference_physical_params, build_scoring_context, combine_experiments,
    log_likelihood_on_mask, mut_rate_model_for_lambda,save_experiment
)

# %%
# =============================================================================
# Configuration
# =============================================================================
subfix_out = 'halfmask_test'  # the subdirectory of SOURCE_OUTPUT_DIR to look in for checkpoints
SOURCE_OUTPUT_DIR = os.path.join('fits_paper', 'designed_sequence', 'outputs',
                                  'compare_alpha_scan_pop80_' + subfix_out)  # the output dir of the run_compare_alpha_scan.py run whose checkpoints you want to cross-validate

# Point this at whatever you want to validate against: a real experiment's info-txt (the .txt
# file next to a .rc file), or a synthetic one saved by run_compare_alpha_scan.py's
# save_experiment() (pass its *_df.csv path and load it yourself with Experiment.from_csv).
NEW_EXPERIMENT_INFO_TXT = os.path.join(SOURCE_OUTPUT_DIR, 'valid_exp_df.csv')

RESULTS_DIR = os.path.join(SOURCE_OUTPUT_DIR, 'crossval')

_CKPT_RE = re.compile(r'^(?P<method>.+)_alpha_(?P<alpha>.+)\.pkl$')


def discover_checkpoints(checkpoint_dir):
    """Yields (method, alpha, path) for every checkpoint file actually present on disk -- this
    is the ground truth, not whatever ALPHAS/ACTIVE_METHODS run_compare_alpha_scan.py currently
    has configured (which may have changed since the source run completed)."""
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pkl'))):
        m = _CKPT_RE.match(os.path.basename(path))
        if m:
            yield m.group('method'), float(m.group('alpha')), path


def load_source_training_exp(output_dir):
    meta = json.load(open(os.path.join(output_dir, 'exp_meta.json')))
    return Experiment.from_csv(os.path.join(output_dir, 'exp_df.csv'), **meta)


def load_source_mask(output_dir):
    mask_path = os.path.join(output_dir, 'mask.txt')
    if not os.path.exists(mask_path):
        return None
    with open(mask_path) as f:
        return np.array([c == '1' for c in f.read().strip()], dtype=bool)


def main():
    # new_exp = Experiment(NEW_EXPERIMENT_INFO_TXT)
    meta = json.load(open(os.path.join(SOURCE_OUTPUT_DIR, 'exp_meta.json')))
    new_exp = Experiment.from_csv(NEW_EXPERIMENT_INFO_TXT, **meta)

    training_exp = load_source_training_exp(SOURCE_OUTPUT_DIR)
    if new_exp.seq != training_exp.seq:
        raise ValueError("New experiment's sequence doesn't match the training experiment's -- "
                          "cross-validation assumes the same underlying construct.")
    mask = load_source_mask(SOURCE_OUTPUT_DIR)

    _, _, mu_r, p_b, m0, m1, p_bind_dict = load_reference_physical_params(
        os.path.join(SOURCE_OUTPUT_DIR, 'phys_params_only.txt'))

    ctx_new = build_scoring_context([new_exp], mu_r, p_b, m0, m1, p_bind_dict, mask=mask)
    full_mask_new = np.ones(ctx_new['exp_fit'].N_seq, dtype=bool)
    if mask is None:
        combined_exp = combine_experiments(training_exp, new_exp)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_experiment(combined_exp, os.path.join(RESULTS_DIR, 'crossval_exp'))
        ctx_combined = build_scoring_context([combined_exp], mu_r, p_b, m0, m1, p_bind_dict, mask=None)
        full_mask_combined = np.ones(ctx_combined['exp_fit'].N_seq, dtype=bool)

    rows = []
    for method, alpha, path in discover_checkpoints(os.path.join(SOURCE_OUTPUT_DIR, 'checkpoints')):
        res = pickle.load(open(path, 'rb'))
        lambda_sc = res['lambda_sc']
        if lambda_sc is None:
            continue
        row = dict(method=method, alpha=alpha)
        mut_rate_new = mut_rate_model_for_lambda(ctx_new, lambda_sc)
        if mask is not None:
            row['log_likelihood_new_trained'] = log_likelihood_on_mask(ctx_new['exp_fit'], mut_rate_new, mask)
            row['log_likelihood_new_masked_out'] = log_likelihood_on_mask(ctx_new['exp_fit'], mut_rate_new, ~mask)
        else:
            row['log_likelihood_new'] = log_likelihood_on_mask(ctx_new['exp_fit'], mut_rate_new, full_mask_new)
            mut_rate_combined = mut_rate_model_for_lambda(ctx_combined, lambda_sc)
            row['log_likelihood_combined'] = log_likelihood_on_mask(
                ctx_combined['exp_fit'], mut_rate_combined, full_mask_combined)
        rows.append(row)

    crossval_df = pd.DataFrame(rows).sort_values(['method', 'alpha'])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    crossval_df.to_csv(os.path.join(RESULTS_DIR, 'crossval_results.csv'), index=False)
    print(f"Saved {len(crossval_df)} rows to {os.path.join(RESULTS_DIR, 'crossval_results.csv')}")

    metrics = [c for c in crossval_df.columns if c.startswith('log_likelihood')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9 * len(metrics), 5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for method in sorted(crossval_df['method'].unique()):
            sub = crossval_df[crossval_df['method'] == method]
            ax.plot(sub['alpha'], sub[metric], 'o-', label=method)
        ax.set_xscale('symlog', linthresh=1e-2)
        ax.set_xlabel('alpha')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'crossval_plot.png')
    plt.savefig(plot_path, dpi=120)
    print(f"Saved plot to {plot_path}")


if __name__ == '__main__':
    main()
