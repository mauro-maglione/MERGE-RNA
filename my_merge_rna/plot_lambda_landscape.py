#!/usr/bin/env python3
"""
Visualize the merge_rna lambda_sc loss landscape (NLL-only and NLL + reg_weight*KL)
for a single free lambda_sc position (1D line plot) and two free positions (2D
contour plot), with the point MultiSystemsFitFixedLambdaPositions actually
converged to overlaid, along with the analytic gradient there.

Motivation: check by eye whether merge_rna's box constraint on lambda_sc
(bound_soft_constraints, default (-1,1)) is actively saturating -- i.e. whether
the found point sits on the edge of the box while the landscape keeps
descending beyond it. This is a diagnostic building block for the broader
maxent-vs-merge_rna box-constraint comparison in my_merge_rna/run_compare_alpha_scan.py.

The scoring/sweep/plot helpers take a plain score_fn(lambda_sc, compute_gradient)
callable rather than hardcoding the merge_rna-specific scoring path, so the same
sweep/plot code can later be pointed at the maxent dual objective
(diagnose_gamma_gradient.py's make_gamma_chi2/make_gamma_binomial) via a small
adapter, without touching this file's sweep/plot logic.

Usage:
    conda run -n merge-rna-patched-1 --no-capture-output \\
        python my_merge_rna/plot_lambda_landscape.py
"""
import os
import sys
import tempfile

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'scripts'))
os.chdir(repo_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from synthetic_lambda_utils import create_exp_synthetic_selected_nucleot, MultiSystemsFitFixedLambdaPositions
from run_compare_alpha_scan import maxent, maxent_binomial, compute_expansion_parameters
from diagnose_gamma_gradient import make_gamma_chi2, make_gamma_binomial
from merge_rna.fit import log_binomial

# =============================================================================
# Configuration
# =============================================================================
TRUE_PARAMS_PATH = ("fits_paper/structured_rnas/physical_params_only_crossval/"
                     "red_crossval_bact_RNaseP_typeA_tetrahymena_ribozyme_V_chol_gly_riboswitch/params1D.txt")
OUTPUT_DIR = os.path.join('my_merge_rna', 'outputs', 'lambda_landscape')

REG_WEIGHT = 1.0  # KL regularization strength for the "regularized" landscape panel

# Which gradient to draw on the maxent (chi2/binomial) points:
#   'own'       -- each optimizer's own dual-objective gradient (Gamma's jac at its solution),
#                  i.e. is *this point* pinned at the boundary of *its own* objective.
#   'merge_rna' -- merge_rna's NLL/KL gradient evaluated at that same lambda_sc (dimensionally
#                  consistent with the plotted curve/heatmap, so 1D draws it as a tangent line
#                  just like merge_rna's own point; 'own' draws a plain descent-direction arrow
#                  since Gamma's gradient isn't in NLL/KL units).
MAXENT_GRAD_SOURCE = 'own'

CONFIG_1D = dict(name='1d_site56', positions=np.array([120]), pops=np.array([0.8]))
CONFIG_2D = dict(name='2d_site56_57', positions=np.array([56, 120]), pops=np.array([0.8, 0.8]))

LAMBDA_RANGE = (-3.0, 3.0)   # extends past the default (-1,1) box on the saturated side
N_POINTS_1D = 61
N_POINTS_2D = 25             # 25x25 grid -> 625 ViennaRNA pf() calls per landscape


# =============================================================================
# One incrementing folder per script run (landscape_test_1, _2, ...), so PNGs,
# diagnostics .txt, and the merge_rna fit subdirectories from a given run all land
# together and past runs are never overwritten.
# =============================================================================
def next_run_dir(base_dir, prefix='landscape_test_'):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix) and d[len(prefix):].isdigit()]
    n = max((int(d[len(prefix):]) for d in existing), default=0) + 1
    run_dir = os.path.join(base_dir, f'{prefix}{n}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# =============================================================================
# Build a fit + scoring context for one (positions, pops) configuration
# =============================================================================
def build_context(name, positions, pops, run_dir):
    """Create the synthetic experiment, run the lambda-restricted fit at
    reg_weight=0 and reg_weight=REG_WEIGHT, and return everything needed to
    score arbitrary lambda_sc vectors against the same physical params."""
    exp, custom_mask, a_phys, b_phys = create_exp_synthetic_selected_nucleot(
        positions=positions, pops=pops, custom_name=name, noise=False)

    true_params_1D = np.loadtxt(TRUE_PARAMS_PATH)
    initial_full = np.concatenate([true_params_1D, np.zeros(exp.N_seq)])
    guess_fd, guess_path = tempfile.mkstemp(suffix='.txt')
    os.close(guess_fd)
    np.savetxt(guess_path, initial_full)

    free_positions = positions - 1

    def run_fit(reg_weight):
        fit = MultiSystemsFitFixedLambdaPositions(
            experiments=[exp],
            validation_exps=None,
            infer_1D_sc=True,
            fit_mode='lambda_only',
            custom_mask=custom_mask,
            free_lambda_positions={exp.system_name: free_positions},
            guess=guess_path,
            reg_weight=reg_weight,
            do_plots=False,
            print_to_std_out=False,
            root_dir=run_dir,
            output_suffix=f'{name}_reg{reg_weight:g}',
            overwrite=True,
        )
        fit.fit()
        fitted_params = fit.pack_params(fit.fit_result.x, fit.systems[0])
        return fit, fitted_params

    fit_a, params_a = run_fit(0.0)
    fit_b, params_b = run_fit(REG_WEIGHT)
    os.remove(guess_path)

    exp_fit = fit_a.systems[0].exp_fits_all[0]
    bound = fit_a.bound_soft_constraints[1]

    print(f"[{name}] reg_weight=0   lambda_sc at free positions: {params_a['lambda_sc'][free_positions]}")
    print(f"[{name}] reg_weight={REG_WEIGHT:g} lambda_sc at free positions: {params_b['lambda_sc'][free_positions]}")

    # Context needed to also run the maxent optimizers on the same problem.
    mu_r, p_b = params_a['mu_r'], params_a['p_b']
    mu_j = mu_r + exp_fit.kBT * np.log((exp_fit.conc_mM + .1) / 1000) if exp_fit.conc_mM is not None else mu_r
    penalty = exp_fit.compute_penalty_m(mu_j, p_b)

    return dict(
        name=name, exp=exp, exp_fit=exp_fit, free_positions=free_positions, bound=bound,
        mu_r=mu_r, p_b=p_b, m0=params_a['m0'], m1=params_a['m1'],
        p_bind_dict=params_a['p_bind'],
        lambda_a=params_a['lambda_sc'], lambda_b=params_b['lambda_sc'],
        a_phys=a_phys, b_phys=b_phys,
        mutation_counts=exp.df['mut_count'].values.astype(float),
        trials_counts=exp.df['total_count'].values.astype(float),
        penalty=penalty, T=exp.temp_C,
    )


# =============================================================================
# Scoring: NLL, KL, and their analytic gradients at an arbitrary lambda_sc
# =============================================================================
def score_lambda_sc(exp_fit, mu_r, p_b, p_bind_dict, m0, m1, lambda_sc, compute_gradient=False):
    mut_rate_model, grad = exp_fit.mut_rate_and_its_grad(
        mu_r=mu_r, p_b=p_b, p_bind=p_bind_dict, m0=m0, m1=m1,
        lambda_sc=lambda_sc, compute_gradient=compute_gradient)
    nll, grad_nll = exp_fit.loss_and_grad(mut_rate_model, grad)
    kl, grad_kl = exp_fit.kl_and_grad(lambda_sc, compute_gradient=compute_gradient)
    return dict(
        nll=nll, kl=kl,
        grad_nll=None if grad_nll is None else grad_nll['lambda_sc'],
        grad_kl=grad_kl,
    )


def make_merge_rna_score_fn(ctx):
    """Adapter closing over one context's physical params -> score_fn(lambda_sc, compute_gradient)."""
    def score_fn(lambda_sc, compute_gradient=False):
        return score_lambda_sc(ctx['exp_fit'], ctx['mu_r'], ctx['p_b'], ctx['p_bind_dict'],
                                ctx['m0'], ctx['m1'], lambda_sc, compute_gradient=compute_gradient)
    return score_fn


# =============================================================================
# maxent (chi-squared and binomial), restricted to the same free positions/box
# =============================================================================
def sanitized_counts(ctx):
    """trials_counts is 0 everywhere except the enforced positions (by construction of
    create_exp_synthetic_selected_nucleot); both maxent variants' closed-form target
    expressions divide by trials_counts, giving literal 0/0 there. Since those
    positions are bound-pinned to lambda=0 regardless, substitute safe placeholder
    counts there: trials=1, mutations=a_phys (i.e. "data" that exactly matches the
    model's own baseline prediction at lambda=0). This makes inverse_loss_derivative(0)
    and compute_expansion_parameters both evaluate to bpp=0 there (the correct
    lambda=0 target) with mut_prob=a_phys strictly inside (0,1) -- avoiding both the
    0/0 and the mutation_counts=0 * log(mut_prob=0) = 0*(-inf) = NaN trap that a
    plain (trials=1, mutations=0) placeholder falls into (mutation_counts=0 forces
    inverse_loss_derivative(0)=-a_phys/b_phys, which makes mut_prob land on exactly 0).
    Real data at the free positions is untouched either way."""
    trials_safe = np.where(ctx['trials_counts'] > 0, ctx['trials_counts'], 1.0)
    mut_safe = np.where(ctx['trials_counts'] > 0, ctx['mutation_counts'], ctx['a_phys'] * trials_safe)
    return trials_safe, mut_safe


def run_maxent_variants(ctx, free_idx, bound, alpha):
    """Run maxent (chi-squared) and maxent_binomial on the same problem, restricted
    to the same free lambda_sc positions and the same box as the merge_rna fit
    (via a per-position `boundaries` array: (0,0) everywhere else)."""
    n_seq = ctx['exp'].N_seq
    boundaries = np.zeros((n_seq, 2))
    for idx in np.atleast_1d(free_idx):
        boundaries[idx] = (-bound, bound)

    trials_safe, mut_safe = sanitized_counts(ctx)
    bpp_ref, sigma_sq = compute_expansion_parameters(mut_safe, trials_safe, ctx['a_phys'], ctx['b_phys'])
    res_chi2 = maxent(ctx['exp'].seq, bpp_ref, sigma_squared=sigma_sq, penalty=ctx['penalty'],
                       boundaries=boundaries, T=ctx['T'], alpha=alpha)[0]
    res_bin = maxent_binomial(ctx['exp'].seq, mutation_counts=mut_safe, trials_counts=trials_safe,
                               a_phys=ctx['a_phys'], b_phys=ctx['b_phys'], penalty=ctx['penalty'],
                               boundaries=boundaries, T=ctx['T'], alpha=alpha)[0]
    return res_chi2, res_bin


def maxent_point(score_fn, res, free_idx):
    """Build a point dict like point_and_grad(), plus the optimizer's own dual-objective
    gradient (res.jac) at the free positions -- the 'own' MAXENT_GRAD_SOURCE option."""
    point = point_and_grad(score_fn, res.x, free_idx)
    point['grad_own_free'] = res.jac[np.atleast_1d(free_idx)]
    return point


def gamma_curve_1d(ctx, free_idx, values, alpha):
    """Sweep maxent's own dual objective Gamma(lambda) along the same 1D line as the
    merge_rna NLL sweep (same free position, all others fixed at 0), for both variants,
    normalized onto a common comparable scale:

    - make_gamma_chi2's Gamma (diagnose_gamma_gradient.py:70) is raw kcal/mol -- its
      dominant term -F + dot(lambdas, bpp_ref) is not beta-scaled (only the separate
      regularization term is). make_gamma_binomial's Gamma (:133) already multiplies its
      whole dominant term by beta, i.e. is already dimensionless/adimensional. So chi2's
      curve is rescaled by the same beta here (identical formula to the one computed
      internally by both Gamma closures) to bring it onto that same adimensional scale.
    - Each curve is then shifted by its own minimum *over this swept array* (not
      necessarily Gamma's true unconstrained global minimum, which has no box baked in
      and could in principle sit outside the swept LAMBDA_RANGE -- in practice the
      converged markers sit well inside it), so both touch 0 at their own minimum and
      are visually comparable on the same twin axis.
    """
    trials_safe, mut_safe = sanitized_counts(ctx)
    bpp_ref, sigma_sq = compute_expansion_parameters(mut_safe, trials_safe, ctx['a_phys'], ctx['b_phys'])
    Gamma_chi2 = make_gamma_chi2(ctx['exp'].seq, bpp_ref, sigma_sq, ctx['penalty'], alpha, ctx['T'])
    Gamma_bin = make_gamma_binomial(ctx['exp'].seq, mut_safe, trials_safe, ctx['a_phys'], ctx['b_phys'],
                                     ctx['penalty'], alpha, ctx['T'])
    n_seq = ctx['exp'].N_seq
    gamma_chi2 = np.empty(len(values))
    gamma_bin = np.empty(len(values))
    for k, v in enumerate(values):
        lam = np.zeros(n_seq)
        lam[free_idx] = v
        gamma_chi2[k] = Gamma_chi2(lam)[0]
        gamma_bin[k] = Gamma_bin(lam)[0]

    kb = 1.98717 / 1000
    beta = 1.0 / ((ctx['T'] + 273.15) * kb)
    gamma_chi2 = gamma_chi2 * beta
    gamma_chi2 = gamma_chi2 - gamma_chi2.min()
    gamma_bin = gamma_bin - gamma_bin.min()
    return gamma_chi2, gamma_bin


# =============================================================================
# Grid sweeps (generic over any score_fn with the score_lambda_sc return shape)
# =============================================================================
def sweep_1d(score_fn, n_seq, free_idx, values):
    nll = np.empty(len(values))
    kl = np.empty(len(values))
    for k, v in enumerate(values):
        lam = np.zeros(n_seq)
        lam[free_idx] = v
        res = score_fn(lam, compute_gradient=False)
        nll[k] = res['nll']
        kl[k] = res['kl']
    return dict(values=values, nll=nll, kl=kl)


def sweep_2d(score_fn, n_seq, free_idx_i, free_idx_j, values_i, values_j):
    nll = np.empty((len(values_i), len(values_j)))
    kl = np.empty((len(values_i), len(values_j)))
    for a, vi in enumerate(values_i):
        for b, vj in enumerate(values_j):
            lam = np.zeros(n_seq)
            lam[free_idx_i] = vi
            lam[free_idx_j] = vj
            res = score_fn(lam, compute_gradient=False)
            nll[a, b] = res['nll']
            kl[a, b] = res['kl']
    return dict(values_i=values_i, values_j=values_j, nll=nll, kl=kl)


def point_and_grad(score_fn, lambda_sc, free_idx):
    """Score+gradient at an actual (fitted) lambda_sc, restricted to free_idx components."""
    res = score_fn(lambda_sc, compute_gradient=True)
    free_idx = np.atleast_1d(free_idx)
    return dict(
        lambda_free=lambda_sc[free_idx],
        nll=res['nll'], kl=res['kl'],
        grad_nll_free=res['grad_nll'][free_idx],
        grad_kl_free=res['grad_kl'][free_idx],
    )


def recovered_bpp(ctx, lambda_sc, free_idx):
    """Pairing probability that a given (full-length) lambda_sc actually produces at the
    free positions. Cheap: ExperimentFit.get_ps caches by (penalty, lambda_sc, interpolated),
    and mut_rate_and_its_grad already called it internally for this exact (penalty, lambda_sc)
    while scoring the point, so this is a cache hit, not a recompute."""
    bpp = ctx['exp_fit'].get_ps(ctx['penalty'], lambda_sc, interpolated=False)
    return bpp[np.atleast_1d(free_idx)]


def saturated_nll_lower_bound(mutation_counts, trials_counts, free_idx):
    """Exact minimum of the binomial NLL as a function of the mutation probability alone,
    i.e. if the pairing probability at each free position could be set completely freely
    (ignoring ViennaRNA/lambda_sc/the box entirely). This is the same 'loss at max'
    construction as my_merge_rna/maxent_test_2.ipynb (loss_of_maxent's log_prob_at_max):
    the binomial NLL is minimized exactly at mut_prob = mut_count/trials_count.
    Uses the real (un-sanitized) counts at the free positions only -- everywhere else
    has zero coverage and contributes 0 to both this bound and the plotted NLL. This is
    a reference floor for the alpha=0/reg_weight=0 panels only -- KL regularization has
    no equivalent simple closed form.
    """
    free_idx = np.atleast_1d(free_idx)
    k = mutation_counts[free_idx]
    n = trials_counts[free_idx]
    p_hat = k / n
    log_binom = np.array([log_binomial(ni, ki) for ni, ki in zip(n, k)])
    term_k = np.where(k > 0, k * np.log(p_hat), 0.0)
    term_nk = np.where(k < n, (n - k) * np.log(1 - p_hat), 0.0)
    return -np.sum(log_binom + term_k + term_nk)


def format_diagnostics(name, free_idx, reg_weight, saturated_nll, imposed_pops,
                        point_a, point_b, point_chi2_a, point_bin_a, point_chi2_b, point_bin_b,
                        bpp_a, bpp_b, bpp_chi2_a, bpp_bin_a, bpp_chi2_b, bpp_bin_b):
    """Human-readable dump of loss/gradient/pairing-probability values for every optimizer's
    point, at both alpha/reg_weight values -- for post-hoc inspection of exactly how each
    point compares, since near-identical *positions* can still hide different local behavior.
    bpp_* is the pairing probability that point's lambda_sc actually produces (via get_ps),
    to compare directly against imposed_pops (what create_exp_synthetic_selected_nucleot
    was asked to enforce when generating the data)."""
    free_idx = np.atleast_1d(free_idx)

    def fmt(x):
        return np.array2string(np.atleast_1d(x), precision=6, floatmode='fixed')

    def fmt_point(label, point, rw, bpp, show_own):
        total = point['nll'] + rw * point['kl']
        grad_total = point['grad_nll_free'] + rw * point['grad_kl_free']
        lines = [
            f"  {label:16s} lambda={fmt(point['lambda_free'])}  bpp={fmt(bpp)}  "
            f"NLL={point['nll']:.6f}  KL={point['kl']:.6f}  total(NLL+{rw:g}*KL)={total:.6f}",
            f"  {'':16s} true grad (merge_rna, d[NLL+{rw:g}*KL]/dlambda) = {fmt(grad_total)}",
        ]
        if show_own:
            lines.append(f"  {'':16s} maxent's own grad (res.jac at free positions)  = {fmt(point['grad_own_free'])}")
        return "\n".join(lines)

    lines = [
        f"=== {name}: lambda_sc landscape diagnostics ===",
        f"Free positions (0-indexed into lambda_sc): {list(free_idx)}",
        f"Imposed pairing probability (pops enforced by create_exp_synthetic_selected_nucleot): "
        f"{fmt(imposed_pops)}",
        f"Saturated NLL lower bound at alpha=0 (exact minimum over an unconstrained "
        f"mutation probability at the free positions, ignoring lambda_sc/folding entirely): "
        f"{saturated_nll:.6f}",
        "",
        "--- alpha=0 / reg_weight=0 ---",
        fmt_point("merge_rna", point_a, 0.0, bpp_a, show_own=False),
        fmt_point("maxent chi2", point_chi2_a, 0.0, bpp_chi2_a, show_own=True),
        fmt_point("maxent binomial", point_bin_a, 0.0, bpp_bin_a, show_own=True),
        "",
        f"--- alpha={reg_weight:g} / reg_weight={reg_weight:g} ---",
        fmt_point("merge_rna", point_b, reg_weight, bpp_b, show_own=False),
        fmt_point("maxent chi2", point_chi2_b, reg_weight, bpp_chi2_b, show_own=True),
        fmt_point("maxent binomial", point_bin_b, reg_weight, bpp_bin_b, show_own=True),
    ]
    return "\n".join(lines)


# =============================================================================
# Plotting
# =============================================================================
def _draw_tangent(ax, x0, y0, slope, xrange, frac=0.12, **kwargs):
    span = frac * (xrange[1] - xrange[0])
    xs = np.array([x0 - span, x0 + span])
    ax.plot(xs, y0 + slope * (xs - x0), **kwargs)


def _draw_direction_arrow(ax, x0, y0, grad, xrange, frac=0.06, **kwargs):
    """Descent-direction arrow along x only (for a gradient not in the plotted y's units)."""
    if grad == 0:
        return
    span = frac * (xrange[1] - xrange[0])
    dx = -np.sign(grad) * span
    ax.annotate('', xy=(x0 + dx, y0), xytext=(x0, y0), arrowprops=dict(width=2, headwidth=7, **kwargs))


def _plot_extra_point_1d(ax, point, values, *, color, marker, label, grad_source, on_total=False, reg_weight=0.0):
    x0 = point['lambda_free'][0]
    y0 = point['nll'] + (reg_weight * point['kl'] if on_total else 0.0)
    ax.plot(x0, y0, marker, color=color, markersize=9, markeredgecolor='black', label=label)
    if grad_source == 'own' and point.get('grad_own_free') is not None:
        _draw_direction_arrow(ax, x0, y0, point['grad_own_free'][0], values,
                               facecolor=color, edgecolor='black')
    elif grad_source == 'merge_rna':
        slope = point['grad_nll_free'][0] + (reg_weight * point['grad_kl_free'][0] if on_total else 0.0)
        _draw_tangent(ax, x0, y0, slope, values, color=color, linestyle='--', linewidth=1.0)


def plot_1d(sweep, point_a, point_b, point_chi2_a, point_bin_a, point_chi2_b, point_bin_b,
            bound, reg_weight, grad_source, title, save_path, saturated_nll=None,
            gamma_curves_a=None, gamma_curves_b=None):
    """gamma_curves_a/b: optional (gamma_chi2, gamma_bin) arrays (same length as sweep['values'],
    from gamma_curve_1d at alpha=0 / alpha=reg_weight respectively) -- maxent's own dual
    objective, NOT on the same scale as NLL/KL, so drawn on a twin y-axis rather than
    overlaid directly on the primary one."""
    values = sweep['values']
    nll, kl = sweep['nll'], sweep['kl']
    total_reg = nll + reg_weight * kl

    loss_a = point_a['nll']
    slope_a = point_a['grad_nll_free'][0]
    loss_b = point_b['nll'] + reg_weight * point_b['kl']
    slope_b = point_b['grad_nll_free'][0] + reg_weight * point_b['grad_kl_free'][0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(values, nll, '-', color='tab:blue')
    ax.axvline(-bound, color='gray', linestyle=':', label=f'box bound (+-{bound:g})')
    ax.axvline(bound, color='gray', linestyle=':')
    if saturated_nll is not None:
        ax.axhline(saturated_nll, color='black', linestyle='-.', linewidth=1,
                   label=f'saturated NLL (alpha=0 exact min) = {saturated_nll:.3f}')
    ax.plot(point_a['lambda_free'][0], loss_a, 'o', color='tab:red', markersize=9,
             label='merge_rna fit (reg_weight=0)')
    _draw_tangent(ax, point_a['lambda_free'][0], loss_a, slope_a, values,
                  color='tab:red', linestyle='--', linewidth=1.2)
    _plot_extra_point_1d(ax, point_chi2_a, values, color='tab:orange', marker='^',
                          label='maxent chi2 (alpha=0)', grad_source=grad_source)
    _plot_extra_point_1d(ax, point_bin_a, values, color='tab:purple', marker='D',
                          label='maxent binomial (alpha=0)', grad_source=grad_source)
    ax.set_xlabel('lambda_sc'); ax.set_ylabel('NLL'); ax.set_title('NLL only')
    if gamma_curves_a is not None:
        gamma_chi2_a, gamma_bin_a = gamma_curves_a
        ax2 = ax.twinx()
        ax2.plot(values, gamma_chi2_a, '--', color='tab:orange', linewidth=1, alpha=0.7,
                 label='Gamma chi2 (dual, alpha=0)')
        ax2.plot(values, gamma_bin_a, '--', color='tab:purple', linewidth=1, alpha=0.7,
                 label='Gamma binomial (dual, alpha=0)')
        ax2.set_ylabel('Gamma - min(Gamma)  (dimensionless, chi2 beta-scaled)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    else:
        ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(values, total_reg, '-', color='tab:blue')
    ax.axvline(-bound, color='gray', linestyle=':')
    ax.axvline(bound, color='gray', linestyle=':')
    ax.plot(point_a['lambda_free'][0], point_a['nll'] + reg_weight * point_a['kl'],
             'o', color='tab:red', markersize=7, alpha=0.5, label='reg_weight=0 fit (reference)')
    ax.plot(point_b['lambda_free'][0], loss_b, 's', color='tab:green', markersize=9,
             label=f'merge_rna fit (reg_weight={reg_weight:g})')
    _draw_tangent(ax, point_b['lambda_free'][0], loss_b, slope_b, values,
                  color='tab:green', linestyle='--', linewidth=1.2)
    _plot_extra_point_1d(ax, point_chi2_b, values, color='tab:orange', marker='^',
                          label=f'maxent chi2 (alpha={reg_weight:g})', grad_source=grad_source,
                          on_total=True, reg_weight=reg_weight)
    _plot_extra_point_1d(ax, point_bin_b, values, color='tab:purple', marker='D',
                          label=f'maxent binomial (alpha={reg_weight:g})', grad_source=grad_source,
                          on_total=True, reg_weight=reg_weight)
    ax.set_xlabel('lambda_sc'); ax.set_ylabel(f'NLL + {reg_weight:g}*KL')
    ax.set_title('NLL + KL (regularized)')
    if gamma_curves_b is not None:
        gamma_chi2_b, gamma_bin_b = gamma_curves_b
        ax2 = ax.twinx()
        ax2.plot(values, gamma_chi2_b, '--', color='tab:orange', linewidth=1, alpha=0.7,
                 label=f'Gamma chi2 (dual, alpha={reg_weight:g})')
        ax2.plot(values, gamma_bin_b, '--', color='tab:purple', linewidth=1, alpha=0.7,
                 label=f'Gamma binomial (dual, alpha={reg_weight:g})')
        ax2.set_ylabel('Gamma - min(Gamma)  (dimensionless, chi2 beta-scaled)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    else:
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def _maxent_grad_2d(point, grad_source, on_total=False, reg_weight=0.0):
    if grad_source == 'own' and point.get('grad_own_free') is not None:
        return point['grad_own_free']
    return point['grad_nll_free'] + (reg_weight * point['grad_kl_free'] if on_total else 0.0)


def plot_2d(sweep, point_a, point_b, point_chi2_a, point_bin_a, point_chi2_b, point_bin_b,
            bound, reg_weight, grad_source, labels, title, save_path, saturated_nll=None):
    values_i, values_j = sweep['values_i'], sweep['values_j']
    nll, kl = sweep['nll'], sweep['kl']
    total_reg = nll + reg_weight * kl

    range_span = max(values_i[-1] - values_i[0], values_j[-1] - values_j[0])
    arrow_len = 0.08 * range_span

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [
        (axes[0], nll, [
            ('merge_rna (reg_weight=0)', point_a, point_a['grad_nll_free'], 'red', 'o'),
            ('maxent chi2 (alpha=0)', point_chi2_a, _maxent_grad_2d(point_chi2_a, grad_source), 'tab:orange', '^'),
            ('maxent binomial (alpha=0)', point_bin_a, _maxent_grad_2d(point_bin_a, grad_source), 'tab:purple', 'D'),
        ], 'NLL only'),
        (axes[1], total_reg, [
            (f'merge_rna (reg_weight={reg_weight:g})', point_b,
             point_b['grad_nll_free'] + reg_weight * point_b['grad_kl_free'], 'green', 's'),
            (f'maxent chi2 (alpha={reg_weight:g})', point_chi2_b,
             _maxent_grad_2d(point_chi2_b, grad_source, on_total=True, reg_weight=reg_weight), 'tab:orange', '^'),
            (f'maxent binomial (alpha={reg_weight:g})', point_bin_b,
             _maxent_grad_2d(point_bin_b, grad_source, on_total=True, reg_weight=reg_weight), 'tab:purple', 'D'),
        ], f'NLL + {reg_weight:g}*KL'),
    ]
    for ax, grid, points, panel_title in panels:
        cf = ax.contourf(values_j, values_i, grid, levels=30, cmap='viridis')
        fig.colorbar(cf, ax=ax, label='loss')
        rect = plt.Rectangle((-bound, -bound), 2 * bound, 2 * bound,
                              fill=False, edgecolor='white', linestyle='--', linewidth=1.5)
        ax.add_patch(rect)
        # Arrows first (low zorder) then markers on top, so overlapping/coincident
        # points stay visible instead of being buried under crossing arrows.
        for label, point, grad_free, color, marker in points:
            li, lj = point['lambda_free']
            gi, gj = grad_free
            norm = np.hypot(gi, gj)
            if norm > 1e-12:
                di, dj = -gi / norm * arrow_len, -gj / norm * arrow_len
                ax.annotate('', xy=(lj + dj, li + di), xytext=(lj, li),
                            arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6,
                                             alpha=0.9), zorder=3)
        for label, point, grad_free, color, marker in points:
            li, lj = point['lambda_free']
            ax.plot(lj, li, marker, color=color, markersize=10, markeredgecolor='black',
                     markeredgewidth=1.2, label=label, zorder=4)
        ax.set_xlabel(labels[1]); ax.set_ylabel(labels[0]); ax.set_title(panel_title)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        if saturated_nll is not None and panel_title == 'NLL only':
            # Contour line at the exact (lambda_sc-unconstrained) NLL floor, so you can see
            # where in (lambda_i, lambda_j) space the landscape actually reaches it.
            ax.contour(values_j, values_i, grid, levels=[saturated_nll],
                       colors='black', linestyles='-.', linewidths=1)
            ax.text(0.02, 0.02, f'saturated NLL = {saturated_nll:.3f}', transform=ax.transAxes,
                    fontsize=8, va='bottom', ha='left',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# Main
# =============================================================================
def run_1d_case(run_dir):
    ctx = build_context(**CONFIG_1D, run_dir=run_dir)
    score_fn = make_merge_rna_score_fn(ctx)
    free_idx = ctx['free_positions'][0]

    values = np.linspace(*LAMBDA_RANGE, N_POINTS_1D)
    sweep = sweep_1d(score_fn, ctx['exp'].N_seq, free_idx, values)

    point_a = point_and_grad(score_fn, ctx['lambda_a'], free_idx)
    point_b = point_and_grad(score_fn, ctx['lambda_b'], free_idx)

    res_chi2_a, res_bin_a = run_maxent_variants(ctx, free_idx, ctx['bound'], alpha=0.0)
    res_chi2_b, res_bin_b = run_maxent_variants(ctx, free_idx, ctx['bound'], alpha=REG_WEIGHT)
    point_chi2_a = maxent_point(score_fn, res_chi2_a, free_idx)
    point_bin_a = maxent_point(score_fn, res_bin_a, free_idx)
    point_chi2_b = maxent_point(score_fn, res_chi2_b, free_idx)
    point_bin_b = maxent_point(score_fn, res_bin_b, free_idx)
    print(f"[{ctx['name']}] maxent chi2     alpha=0   lambda: {point_chi2_a['lambda_free']}   "
          f"alpha={REG_WEIGHT:g} lambda: {point_chi2_b['lambda_free']}")
    print(f"[{ctx['name']}] maxent binomial alpha=0   lambda: {point_bin_a['lambda_free']}   "
          f"alpha={REG_WEIGHT:g} lambda: {point_bin_b['lambda_free']}")

    bpp_a = recovered_bpp(ctx, ctx['lambda_a'], free_idx)
    bpp_b = recovered_bpp(ctx, ctx['lambda_b'], free_idx)
    bpp_chi2_a = recovered_bpp(ctx, res_chi2_a.x, free_idx)
    bpp_bin_a = recovered_bpp(ctx, res_bin_a.x, free_idx)
    bpp_chi2_b = recovered_bpp(ctx, res_chi2_b.x, free_idx)
    bpp_bin_b = recovered_bpp(ctx, res_bin_b.x, free_idx)

    saturated_nll = saturated_nll_lower_bound(ctx['mutation_counts'], ctx['trials_counts'], free_idx)
    diag_text = format_diagnostics(ctx['name'], free_idx, REG_WEIGHT, saturated_nll, CONFIG_1D['pops'],
                                    point_a, point_b, point_chi2_a, point_bin_a, point_chi2_b, point_bin_b,
                                    bpp_a, bpp_b, bpp_chi2_a, bpp_bin_a, bpp_chi2_b, bpp_bin_b)
    print(diag_text)
    diag_path = os.path.join(run_dir, f"{ctx['name']}_diagnostics.txt")
    with open(diag_path, 'w') as f:
        f.write(diag_text + "\n")
    print(f"Saved {diag_path}")

    gamma_curves_a = gamma_curve_1d(ctx, free_idx, values, alpha=0.0)
    gamma_curves_b = gamma_curve_1d(ctx, free_idx, values, alpha=REG_WEIGHT)

    plot_1d(sweep, point_a, point_b, point_chi2_a, point_bin_a, point_chi2_b, point_bin_b,
            ctx['bound'], REG_WEIGHT, MAXENT_GRAD_SOURCE,
            title=f"{ctx['name']}: lambda_sc landscape at position {CONFIG_1D['positions'][0]}",
            save_path=os.path.join(run_dir, f"{ctx['name']}.png"),
            saturated_nll=saturated_nll,
            gamma_curves_a=gamma_curves_a, gamma_curves_b=gamma_curves_b)


def run_2d_case(run_dir):
    ctx = build_context(**CONFIG_2D, run_dir=run_dir)
    score_fn = make_merge_rna_score_fn(ctx)
    free_idx_i, free_idx_j = ctx['free_positions']
    free_idx = [free_idx_i, free_idx_j]

    values_i = np.linspace(*LAMBDA_RANGE, N_POINTS_2D)
    values_j = np.linspace(*LAMBDA_RANGE, N_POINTS_2D)
    sweep = sweep_2d(score_fn, ctx['exp'].N_seq, free_idx_i, free_idx_j, values_i, values_j)

    point_a = point_and_grad(score_fn, ctx['lambda_a'], free_idx)
    point_b = point_and_grad(score_fn, ctx['lambda_b'], free_idx)

    res_chi2_a, res_bin_a = run_maxent_variants(ctx, free_idx, ctx['bound'], alpha=0.0)
    res_chi2_b, res_bin_b = run_maxent_variants(ctx, free_idx, ctx['bound'], alpha=REG_WEIGHT)
    point_chi2_a = maxent_point(score_fn, res_chi2_a, free_idx)
    point_bin_a = maxent_point(score_fn, res_bin_a, free_idx)
    point_chi2_b = maxent_point(score_fn, res_chi2_b, free_idx)
    point_bin_b = maxent_point(score_fn, res_bin_b, free_idx)
    print(f"[{ctx['name']}] maxent chi2     alpha=0   lambda: {point_chi2_a['lambda_free']}   "
          f"alpha={REG_WEIGHT:g} lambda: {point_chi2_b['lambda_free']}")
    print(f"[{ctx['name']}] maxent binomial alpha=0   lambda: {point_bin_a['lambda_free']}   "
          f"alpha={REG_WEIGHT:g} lambda: {point_bin_b['lambda_free']}")

    bpp_a = recovered_bpp(ctx, ctx['lambda_a'], free_idx)
    bpp_b = recovered_bpp(ctx, ctx['lambda_b'], free_idx)
    bpp_chi2_a = recovered_bpp(ctx, res_chi2_a.x, free_idx)
    bpp_bin_a = recovered_bpp(ctx, res_bin_a.x, free_idx)
    bpp_chi2_b = recovered_bpp(ctx, res_chi2_b.x, free_idx)
    bpp_bin_b = recovered_bpp(ctx, res_bin_b.x, free_idx)

    saturated_nll = saturated_nll_lower_bound(ctx['mutation_counts'], ctx['trials_counts'], free_idx)
    diag_text = format_diagnostics(ctx['name'], free_idx, REG_WEIGHT, saturated_nll, CONFIG_2D['pops'],
                                    point_a, point_b, point_chi2_a, point_bin_a, point_chi2_b, point_bin_b,
                                    bpp_a, bpp_b, bpp_chi2_a, bpp_bin_a, bpp_chi2_b, bpp_bin_b)
    print(diag_text)
    diag_path = os.path.join(run_dir, f"{ctx['name']}_diagnostics.txt")
    with open(diag_path, 'w') as f:
        f.write(diag_text + "\n")
    print(f"Saved {diag_path}")

    labels = (f"lambda_sc[{CONFIG_2D['positions'][0]}]", f"lambda_sc[{CONFIG_2D['positions'][1]}]")
    plot_2d(sweep, point_a, point_b, point_chi2_a, point_bin_a, point_chi2_b, point_bin_b,
            ctx['bound'], REG_WEIGHT, MAXENT_GRAD_SOURCE, labels,
            title=f"{ctx['name']}: lambda_sc landscape at positions {[int(p) for p in CONFIG_2D['positions']]}",
            save_path=os.path.join(run_dir, f"{ctx['name']}.png"),
            saturated_nll=saturated_nll)


if __name__ == "__main__":
    run_dir = next_run_dir(OUTPUT_DIR)
    print(f"Run output directory: {run_dir}")
    # Set CONFIG_1D / CONFIG_2D to None above to skip that case (faster iteration).
    if CONFIG_1D is not None:
        run_1d_case(run_dir)
    if CONFIG_2D is not None:
        run_2d_case(run_dir)
