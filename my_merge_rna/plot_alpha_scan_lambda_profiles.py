# %%
#!/usr/bin/env python3
"""
Recover lambda_sc from the checkpoints of an alpha scan (run_compare_alpha_scan.py) and plot,
for a single chosen method, the resulting pairing-probability profile at several chosen alphas --
overlaid against the same baseline (lambda_sc=0) and target-data reference used in the maxent
notebooks.

Driven entirely by artifacts the source run already wrote to its OUTPUT_DIR (checkpoints/*.pkl,
exp_df.csv + exp_meta.json, mask.txt if present, phys_params_only.txt) -- not by the CURRENT
config in run_compare_alpha_scan.py, since that may have changed since the source run finished.

Run from the repo root:
    python my_merge_rna/plot_alpha_scan_lambda_profiles.py
"""
import os
import sys
import pickle

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: this script never has a display attached
import matplotlib.pyplot as plt

from my_merge_rna.run_compare_alpha_scan import (
    load_reference_physical_params, build_scoring_context, compute_expansion_parameters,
    METHOD_LABELS,
)
from my_merge_rna.run_checkpoint_crossval import (
    discover_checkpoints, load_source_training_exp, load_source_mask,
)

# %%
# =============================================================================
# Configuration
# =============================================================================
subfix_out = 'less_coverage_10e5'  # the subdirectory suffix of the alpha-scan run whose checkpoints to read
SOURCE_OUTPUT_DIR = os.path.join('fits_paper', 'designed_sequence', 'outputs',
                                  'compare_alpha_scan_pop80_' + subfix_out)

METHOD = 'maxent_chi2'  # which method's checkpoints to plot -- one of METHOD_LABELS' keys
                        # ('merge_rna', 'maxent_chi2', 'maxent_binomial')

# Alphas to overlay, one line per alpha. Use 'all' to plot every alpha checkpointed for METHOD,
# or a list of alpha values -- each is matched to the closest alpha actually on disk.
ALPHAS_TO_PLOT = [0.01, 1.0, 10.0, 100.0, 10000, 1000000.0]

RESULTS_DIR = os.path.join(SOURCE_OUTPUT_DIR, 'lambda_plots')


# %%
def discover_for_method(checkpoint_dir, method):
    """{alpha: path} for every checkpoint of `method` actually present on disk."""
    return {alpha: path for m, alpha, path in discover_checkpoints(checkpoint_dir) if m == method}


def resolve_alphas(by_alpha, requested):
    if requested == 'all':
        return sorted(by_alpha)
    available = np.array(sorted(by_alpha))
    resolved = []
    for alpha in requested:
        match = float(available[np.argmin(np.abs(available - alpha))])
        if not np.isclose(match, alpha, rtol=1e-6):
            print(f"  note: alpha={alpha:.6g} not checkpointed for {METHOD!r}, "
                  f"using closest available alpha={match:.6g}")
        resolved.append(match)
    return sorted(set(resolved))


def main():
    checkpoint_dir = os.path.join(SOURCE_OUTPUT_DIR, 'checkpoints')
    method_label = METHOD_LABELS.get(METHOD, METHOD)

    by_alpha = discover_for_method(checkpoint_dir, METHOD)
    if not by_alpha:
        available_methods = sorted({m for m, _, _ in discover_checkpoints(checkpoint_dir)})
        raise ValueError(f"No checkpoints found for method {METHOD!r} in {checkpoint_dir}. "
                          f"Methods available there: {available_methods}")

    training_exp = load_source_training_exp(SOURCE_OUTPUT_DIR)
    mask = load_source_mask(SOURCE_OUTPUT_DIR)

    _, _, mu_r, p_b, m0, m1, p_bind_dict = load_reference_physical_params(
        os.path.join(SOURCE_OUTPUT_DIR, 'phys_params_only.txt'))
    ctx = build_scoring_context([training_exp], mu_r, p_b, m0, m1, p_bind_dict, mask=mask)
    exp_fit, penalty, a_i, b_i = ctx['exp_fit'], ctx['penalty'], ctx['a_i'], ctx['b_i']

    alphas = resolve_alphas(by_alpha, ALPHAS_TO_PLOT)
    print(f"Plotting method={METHOD!r} ({method_label}) at alphas={[f'{a:.4g}' for a in alphas]}")

    positions = np.arange(1, exp_fit.N_seq + 1)
    mutation_counts = training_exp.df['mut_count'].values.astype(float)
    trials_counts = training_exp.df['total_count'].values.astype(float)
    target, target_var = compute_expansion_parameters(mutation_counts, trials_counts, a_i, b_i)
    target_err = np.sqrt(target_var)

    baseline_bpp = exp_fit.get_ps(penalty, np.zeros(exp_fit.N_seq))

    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    ax.plot(positions, baseline_bpp, label='baseline', linestyle=':', color='black', alpha=.6, linewidth=1.5)
    ax.plot(positions, target, marker='o', linestyle='', color='tab:blue', label='target')
    ax.fill_between(positions, target - target_err, target + target_err, facecolor='gray',
                     edgecolor='gray', alpha=0.25, linestyle='--', linewidth=1.2, label='target error band')

    cmap = plt.get_cmap('viridis')
    for i, alpha in enumerate(alphas):
        res = pickle.load(open(by_alpha[alpha], 'rb'))
        lambda_sc = res['lambda_sc']
        if lambda_sc is None:
            print(f"  skipping alpha={alpha:.4g}: checkpoint has no lambda_sc (failed fit)")
            continue
        bpp = exp_fit.get_ps(penalty, lambda_sc)
        color = cmap(i / max(len(alphas) - 1, 1))
        ax.plot(positions, bpp, linestyle='-', color=color, label=f'alpha={alpha:.4g}')

    ax.set_xlabel('Position', fontsize=20)
    ax.set_ylabel('Pairing probability', fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'{method_label}: pairing probability vs. position across alpha', fontsize=16)
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, f'{METHOD}_pairing_prob_vs_alpha.png')
    plt.savefig(plot_path, dpi=120)
    print(f"Saved plot to {plot_path}")


if __name__ == '__main__':
    main()

# %%
