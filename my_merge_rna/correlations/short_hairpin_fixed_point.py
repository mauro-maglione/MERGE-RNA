# %%
"""
Quick contained test on the bistable 29-nt sequence CACACAC UUUU GUGUGUG UUUU CACACAC:
  1. synthetic mutation experiment (as in generate_synth_mut_experiment.py), designed_sequence
     physical params, eps_b ~ N(mean, std) of the NewseqWT eps_b, -3.5 kcal/mol state bias
     favouring the 3' hairpin (GUGUGUG paired with the last CACACAC);
  2. maxent (chi-squared) fit of lambda_sc at alpha=1 with the physical params fixed
     (maxent from run_compare_alpha_scan.py);
  3. subopt_merge.py reweighting (alpha=1) of the ensemble corrected by the fitted lambda_sc:
     if the maxent solution is a fixed point, the optimal weights should stay W=1.
"""
import ast
import os
import pathlib
import sys

try:
    _repo_root = pathlib.Path(__file__).resolve().parents[2]
except NameError:
    _cwd = pathlib.Path.cwd().resolve()
    _repo_root = next(p for p in [_cwd, *_cwd.parents] if (p / 'merge_rna').is_dir())

os.environ.setdefault('RNA_STRUCT_HOME', str(_repo_root))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])
sys.path.insert(0, os.path.join(os.environ['RNA_STRUCT_HOME'], 'my_merge_rna'))
os.chdir(str(_repo_root))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import RNA
import bussilab.rna2d as bussi
from scipy.optimize import minimize
from scipy.special import logsumexp

from merge_rna import Experiment, MultiSystemsFit
from run_compare_alpha_scan import maxent, compute_expansion_parameters,bounds_array_with_mask

# subopt_merge.py runs a full optimization at import time: pull out only its function definitions
_subopt_src = pathlib.Path(_repo_root, 'my_merge_rna', 'correlations', 'subopt_merge.py').read_text()
_subopt_defs = [n for n in ast.parse(_subopt_src).body if isinstance(n, ast.FunctionDef)
                and n.name in ('structures_sampling', 'forward_model_fixed_params',
                               '_loss_and_der_coeff_loss', 'forward_model', 'do_optimization')]
exec(compile(ast.Module(body=_subopt_defs, type_ignores=[]), 'subopt_merge.py', 'exec'))

# %%
# =============================================================================
# Configuration
# =============================================================================
SEQ = 'CACACACUUUUGUGUGUGUUUUCACACAC'
# SEQ = 'CACACACUUUUGUGUGUGUUUUCACACACUUUUCACACAC'

CONC_MM = 80
TEMP_C = 37
N_READS = 50000
ALPHA = 1.0
STATE_BIAS1 = -2.5
STATE_BIAS2 = -3.5
STATE_POSITIONS = [[6, 11], [17, 22]]   # extremes of the 3' CACACAC and of GUGUGUG
BOUNDS_SC = 5.
DELTA_SUBOPT = 10   # kcal/mol

phase1_params = os.path.join(os.environ['RNA_STRUCT_HOME'], 'fits_paper', 'designed_sequence',
                             'experimental', 'strand_disp_turner.txt')
_eps_b_ref_path = os.path.join(os.environ['RNA_STRUCT_HOME'], 'fits', 'designed_sequence', 'outputs',
                               'experimental_turner', 'phase1', 'newseqWT', 'detailed', 'NewseqWT_eps_b.npy')
OUTPUT_DIR = os.path.join(os.environ['RNA_STRUCT_HOME'], 'my_merge_rna', 'correlations', 'outputs_short_hairpin')
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
bussi.set_random_seed(42)

# %%
# =============================================================================
# Physical model for the new sequence (dummy Experiment, only used for calculations)
# =============================================================================
N = len(SEQ)
# position_mask = np.ones(N, dtype=bool)
position_mask = np.array([nb != 'G' and nb != 'U' for nb in SEQ])
_dummy_df = pd.DataFrame({'pos': np.arange(1, N + 1), 'ref_nt': list(SEQ),
                          'mut_count': np.zeros(N, dtype=int), 'total_count': N_READS * np.ones(N, dtype=int)})
exp_dummy = Experiment.from_dataframe(_dummy_df, seq=SEQ, system_name='CAhairpin', conc_mM=CONC_MM, temp_C=TEMP_C)
fit_struct = MultiSystemsFit([exp_dummy], validation_exps=None, infer_1D_sc=False, skip_output_setup=True,
                             custom_mask=position_mask)

_eps_b_ref = np.load(_eps_b_ref_path)
eps_b = np.clip(np.random.normal(_eps_b_ref.mean(), _eps_b_ref.std(), N), 0., None)
fit_struct.systems[0].eps_b = eps_b
fit_struct.systems[0].exp_fits_train[0].eps_b = eps_b
exp_fit = fit_struct.systems[0].exp_fits_train[0]

params_dict_ref = fit_struct.pack_params(np.loadtxt(phase1_params)[:8], fit_struct.systems[0])
mu_r, p_b, m0, m1, p_bind_dict = (params_dict_ref['mu_r'], params_dict_ref['p_b'], params_dict_ref['m0'],
                                  params_dict_ref['m1'], params_dict_ref['p_bind'])
mu_j = mu_r + exp_fit.kBT * np.log((exp_fit.conc_mM + .1) / 1000)
penalty = exp_fit.compute_penalty_m(mu_j, p_b)

_, _, p_cb_given_sb_nb = exp_fit.compute_mu_dependent_quantities(mu_j, p_b, p_bind_dict)
p_react_unpaired = np.array([p_cb_given_sb_nb[(0, nb)] for nb in SEQ])
p_react_paired = np.array([p_cb_given_sb_nb[(1, nb)] for nb in SEQ])
p_cb_unpaired = exp_fit.compute_p_cb(mu_j, p_b, p_bind_dict, np.zeros(N))
p_cb_paired = exp_fit.compute_p_cb(mu_j, p_b, p_bind_dict, np.ones(N))
a_i = exp_fit.compute_mutation_profile(m0, m1, eps_b, p_cb_unpaired)
b_i = exp_fit.compute_mutation_profile(m0, m1, eps_b, p_cb_paired) - a_i
prob_fp = 1 - np.exp(-eps_b - m0)
prob_fn = np.exp(-eps_b - m1)
print(f"penalty = {penalty:.4f} kcal/mol, a_i in [{a_i.min():.4f},{a_i.max():.4f}], a_i + b_i in [{(a_i +b_i).min():.4f},{(a_i + b_i).max():.4f}]")

# %%
# =============================================================================
# 1. Synthetic experiment from the biased ensemble
# =============================================================================
bias1 = [[0, 0], [0, STATE_BIAS1]]
bias2 = [[0, 0], [0, STATE_BIAS2]]
mol_true = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N), state_positions=STATE_POSITIONS,
                          state_biases=[bias1, bias2], temperature=TEMP_C + 273.15)
mol_prior = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N), temperature=TEMP_C + 273.15)
bpp_true = mol_true.base_pairing_probability().sum(1)
bpp_prior = mol_prior.base_pairing_probability().sum(1)

sampled_states, _ = bussi.sample_to_numpy(mol_true.sample(N_READS), deduplicate=False)
paired = sampled_states != -1
reacted = np.random.random(paired.shape) < np.where(paired, p_react_paired, p_react_unpaired)
mutated = np.random.random(paired.shape) < np.where(reacted, 1 - prob_fn, prob_fp)

mut_counts = mutated.sum(0)
read_counts = N_READS * np.ones(N, dtype=int)
counts_df = pd.DataFrame({'Sample': f'synthetic_CAhairpin_{CONC_MM}mM', 'mut_count_per_site': mut_counts,
                          'number_of_reads': read_counts})
counts_df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_counts.csv'), index=False)
np.save(os.path.join(OUTPUT_DIR, 'eps_b.npy'), eps_b)

# %%

plt.plot(bpp_prior)
plt.plot(bpp_true)


# %%
# =============================================================================
# 2. maxent (chi-squared), alpha=1, physical params fixed
# =============================================================================
bpp_target, sigma_squared = compute_expansion_parameters(mut_counts.astype(float), read_counts.astype(float), a_i, b_i)
boundaries_mask = bounds_array_with_mask(N, position_mask, BOUNDS_SC)

res_chi2, bpp_maxent_out = maxent(SEQ, bpp_target, sigma_squared=sigma_squared, penalty=penalty,
                                  boundaries=boundaries_mask, alpha=ALPHA, T=TEMP_C)
lambda_sc = res_chi2.x
np.savetxt(os.path.join(OUTPUT_DIR, f'lambda_sc_maxent_chi2_alpha{ALPHA:g}.txt'), lambda_sc)
print(f"maxent: success={res_chi2.success} nit={res_chi2.nit} |grad|={np.linalg.norm(res_chi2.jac):.2e} "
      f"n_at_bound={np.sum(np.isclose(np.abs(lambda_sc), BOUNDS_SC))}")

# the fitted ensemble, recomputed with bussi (same lambdas1d convention as merge_rna/subopt_merge)
mol_fit = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N) + lambda_sc, temperature=TEMP_C + 273.15)
bpp_fit = mol_fit.base_pairing_probability().sum(1)
# maxent's second output is bpp_ref - jac + alpha*beta*sigma^2*lambda = the Vienna bpp at the solution
print(f"max |bpp(bussi, penalty+lambda_sc) - bpp(maxent vienna)| = {np.abs(bpp_fit - bpp_maxent_out).max():.2e}")

# %%
# =============================================================================
# 3. subopt_merge on the sc-corrected ensemble: is W=1 a fixed point?
# =============================================================================
_, states, logweights = structures_sampling(SEQ, penalty, DELTA_SUBOPT, lambda_sc=lambda_sc, temperature_C=TEMP_C)
coverage = mol_fit.suboptimal_coverage(DELTA_SUBOPT)
print(f"subopt: {states.shape[0]} structures within {DELTA_SUBOPT} kcal/mol, coverage of Z = {coverage:.6f}")

# objective and gradient at the starting point W=1 (same construction as do_optimization)
molecule_ref = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N))
logweights_ref = -np.apply_along_axis(molecule_ref.evaluate, 1, states) / (bussi._KB * (37 + 273.15))
logweights_ref -= logsumexp(logweights_ref)
M = forward_model_fixed_params(states, a_i, b_i, mask=position_mask)
loss_W1, grad_W1 = forward_model(mut_counts, read_counts, np.ones(len(logweights)), logweights, logweights_ref, M, alpha=ALPHA)

# same check with the per-structure gradient projected on directions that change P (gradient wrt log W,
# minus its P-weighted mean: a uniform rescaling of W leaves P unchanged)
P_fit = np.exp(logweights - logsumexp(logweights))
g_logW = grad_W1 * np.exp(logsumexp(logweights))   # W=1 so d/dlogW = d/dW, rescale P0/Z -> P
g_logW_centered = g_logW - P_fit @ g_logW

result = do_optimization(logweights, states, SEQ, penalty, a_i, b_i, counts_df, position_mask, alpha=ALPHA)
logweights_opt = np.log(result.x) + logweights
P_opt = np.exp(logweights_opt - logsumexp(logweights_opt))
pairing_states = (states != -1).astype(float)
bpp_subopt_start = P_fit @ pairing_states
bpp_subopt_opt = P_opt @ pairing_states

# reference: run the same subopt optimization starting from the prior (no sc) ensemble
_, states_0, logweights_0 = structures_sampling(SEQ, penalty, DELTA_SUBOPT, temperature_C=TEMP_C)
result_0 = do_optimization(logweights_0, states_0, SEQ, penalty, a_i, b_i, counts_df, position_mask, alpha=ALPHA)
lw0_opt = np.log(result_0.x) + logweights_0
bpp_subopt_from_prior = np.exp(lw0_opt - logsumexp(lw0_opt)) @ (states_0 != -1).astype(float)

kl_P = lambda P: P @ (np.log(P) - logweights_ref)
print(f"subopt from sc-ensemble: success={result.success} nit={result.nit} loss {loss_W1:.4f} -> {result.fun:.4f}")
print(f"  at W=1: max |dloss/dlogW| (centered, P-weighted scale) = {np.abs(P_fit * g_logW_centered).max():.2e}")
print(f"  KL(P_start||P_ref) = {kl_P(P_fit):.4f}, KL(P_opt||P_ref) = {kl_P(P_opt):.4f}, "
      f"KL(P_opt||P_start) = {P_opt @ (np.log(P_opt) - np.log(P_fit)):.2e}")
print(f"  max |bpp_opt - bpp_start| = {np.abs(bpp_subopt_opt - bpp_subopt_start).max():.4f}")
print(f"subopt from prior: success={result_0.success} nit={result_0.nit} loss={result_0.fun:.4f}, "
      f"max |bpp - bpp_maxent| = {np.abs(bpp_subopt_from_prior - bpp_fit).max():.4f}")

summary = pd.DataFrame({'seq': list(SEQ), 'eps_b': eps_b, 'a_i': a_i, 'b_i': b_i, 'mut_count': mut_counts,
                        'bpp_target': bpp_target, 'sigma_squared': sigma_squared, 'lambda_sc': lambda_sc,
                        'bpp_prior': bpp_prior, 'bpp_true': bpp_true, 'bpp_maxent': bpp_fit,
                        'bpp_subopt_start': bpp_subopt_start, 'bpp_subopt_opt': bpp_subopt_opt,
                        'bpp_subopt_from_prior': bpp_subopt_from_prior})
summary.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index_label='pos0')

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
ax = axes[0]
ax.plot(bpp_prior, 'k:', label='prior (penalty only)')
ax.plot(bpp_true, 'k-', lw=2, alpha=0.4, label=f'true (bias {STATE_BIAS1})')
ax.plot(np.clip(bpp_target, -0.2, 1.2), 'x', color='0.5', label='chi2 target (clipped)')
ax.plot(bpp_fit, 'C0o-', label='maxent chi2, alpha=1')
ax.plot(bpp_subopt_opt, 'C1s--', mfc='none', label='subopt from maxent ensemble')
ax.plot(bpp_subopt_from_prior, 'C2^:', mfc='none', label='subopt from prior')
ax.set_xticks(range(N)); ax.set_xticklabels(list(SEQ), fontsize=7)
ax.set_ylabel('pairing probability'); ax.legend(fontsize=7)
ax = axes[1]
ax.semilogy(np.sort(P_fit)[::-1], 'C0-', label='P start (maxent ensemble)')
ax.semilogy(P_opt[np.argsort(P_fit)[::-1]], 'C1.', ms=3, label='P after subopt')
ax.set_xlabel('structure rank'); ax.set_ylabel('probability'); ax.legend(fontsize=7)
ax.set_title(f'subopt: {states.shape[0]} structures within {DELTA_SUBOPT} kcal/mol, coverage of Z = {coverage:.6f}')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f'fixed_point_check_delta_{DELTA_SUBOPT}.png'), dpi=120)
plt.show()

# %%
