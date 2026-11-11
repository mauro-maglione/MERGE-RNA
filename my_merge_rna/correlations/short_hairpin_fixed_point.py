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
from collections.abc import Iterable
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
from scipy.sparse import csr_matrix

from merge_rna import Experiment, MultiSystemsFit
from run_compare_alpha_scan import maxent, compute_expansion_parameters,bounds_array_with_mask

# subopt_merge.py runs a full optimization at import time: pull out only its function definitions
_subopt_src = pathlib.Path(_repo_root, 'my_merge_rna', 'correlations', 'subopt_merge.py').read_text()
_subopt_defs = [n for n in ast.parse(_subopt_src).body if isinstance(n, ast.FunctionDef)
                and n.name in ('structures_sampling', 'forward_model_fixed_params',
                               '_loss_and_der_coeff_loss', 'forward_model', 'do_optimization')]
exec(compile(ast.Module(body=_subopt_defs, type_ignores=[]), 'subopt_merge.py', 'exec'))

# same for com_maxent.py (read-level likelihood), but in a separate namespace: some names
# (forward_model, forward_model_fixed_params) clash with the subopt_merge ones used in section 3
_cm_src = pathlib.Path(_repo_root, 'my_merge_rna', 'correlations', 'com_maxent.py').read_text()
_cm_defs = [n for n in ast.parse(_cm_src).body if isinstance(n, ast.FunctionDef)
            and n.name in ('forward_model_fixed_params', 'precompute_read_factors',
                           'precompute_log_read_likelihood', '_loss_and_ratio_sum',
                           'forward_model', 'forward_model_chunked')]
cm = {'np': np, 'logsumexp': logsumexp, 'csr_matrix': csr_matrix}
exec(compile(ast.Module(body=_cm_defs, type_ignores=[]), 'com_maxent.py', 'exec'), cm)

# %%
# =============================================================================
# Configuration
# =============================================================================
SEQ = 'CACACACUUUUGUGUGUGUUUUCACACAC'
# SEQ = 'CAGUGACUUUUGUCACUGUUUUCAGUGAC'
# SEQ = 'CACACACUUUUGUGUGUGUUUUCACACACUUUUCACACAC'

CONC_MM = 80
TEMP_C = 37
N_READS = 150000
ALPHA = 1.0
ALPHA_MAXENT = 10.0
favor_loop_ends = True   # if True, bias the 3' hairpin (GUGUGUG paired with the last CACACAC)
if favor_loop_ends:
      STATE_BIAS1 = -2.5
      STATE_BIAS2 = -3.5
      STATE_POSITIONS = [[6, 11], [17, 22]]   # extremes of the 3' CACACAC and of GUGUGUG
else:
      # STATE_BIAS1 = -2.5 # this goes to s_i*s_j for 6,11
      # STATE_BIAS2 = -3.5 # this goes to s_i*s_j for 17,22
      # STATE_BIAS3 = -10.5 # this goes to (1-s_i)*(1-s_j) for 6,11
      # STATE_POSITIONS = [[6, 11], [17, 22]]   # extremes of the 3' CACACAC and of GUGUGUG

      # PAIR_BIAS = -3.5
      # SINGLE_STATE_BIAS = 1.5
      # STATE_POSITIONS = [22, 23]  

      # STATE_BIASES = np.random.uniform(-5,5, 10)
      # twenty_rand_pos = np.random.choice(range(1, len(SEQ)-1), 20, replace=False)
      # STATE_POSITIONS = [[i, j] for i, j in zip(twenty_rand_pos[::2], twenty_rand_pos[1::2])]

      STATE_BIAS1 = 0.
      STATE_BIAS2 = 0.
      STATE_BIAS3 = -2.
      STATE_POSITIONS = [[6, 11], [17, 22], [25,26,27]]   # extremes of the 3' CACACAC and of GUGUGUG

# Optional hand-picked true ensemble: list of (dot-bracket, free energy in kcal/mol). When not None it
# replaces the biased bussi ensemble above; populations are Boltzmann weights exp(-E/kT) normalized over the list.
TRUE_STRUCTURES = None
# TRUE_STRUCTURES = [('.((......))(((((((....)))))))', -6.0063596855674355),
#                    ('((((((......))))))...........', -5.979769768943948),
#                    ('(((((((....))))))).(......)..', -5.906359780934867)]

BOUNDS_SC = 15.
DELTA_SUBOPT = 20   # kcal/mol

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
position_mask = np.array([nb != 'U' for nb in SEQ])
#position_mask = np.array([nb != 'G' and nb != 'U' for nb in SEQ])
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

if favor_loop_ends:
    bias1 = [[0, 0], [0, STATE_BIAS1]]
    bias2 = [[0, 0], [0, STATE_BIAS2]]
    mol_true = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N), state_positions=STATE_POSITIONS,
                          state_biases=[bias1, bias2], temperature=TEMP_C + 273.15)
else:
      # bias = [[[0, 0], [0, state_bias]] for state_bias in STATE_BIASES]
      # mol_true = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N), state_positions=STATE_POSITIONS,
      #                     state_biases=bias, temperature=TEMP_C + 273.15)
    
    #mol_true = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N), temperature=TEMP_C + 273.15, base_pair_biases=[(STATE_POSITIONS[0], STATE_POSITIONS[1], PAIR_BIAS)])
      bias =[[[0, 0], [0, STATE_BIAS1]],[[0, 0], [0, STATE_BIAS2]],[[[0, 0], [0, 0]],[[0, 0], [0, STATE_BIAS3]]]]
      mol_true = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N), state_positions=STATE_POSITIONS,
                          state_biases=bias, temperature=TEMP_C + 273.15)
    


mol_prior = bussi.Molecule(SEQ, lambdas1d=penalty * np.ones(N), temperature=TEMP_C + 273.15)
bpp_prior = mol_prior.base_pairing_probability().sum(1)

if TRUE_STRUCTURES is None:
    bpp_true = mol_true.base_pairing_probability().sum(1)
    corr_true = mol_true.pairing_correlation_matrix() - np.outer(bpp_true, bpp_true)
    sampled_states, _ = bussi.sample_to_numpy(mol_true.sample(N_READS), deduplicate=False)
    paired = sampled_states != -1
else:
    # finite true ensemble of hand-picked structures
    assert all(len(db) == N for db, _ in TRUE_STRUCTURES), "dot-bracket length != sequence length"
    pairing_states_true = np.array([[c != '.' for c in db] for db, _ in TRUE_STRUCTURES], dtype=float)
    _E_true = np.array([E for _, E in TRUE_STRUCTURES])
    _lw_true = -_E_true / (bussi._KB * (TEMP_C + 273.15))
    p_true = np.exp(_lw_true - logsumexp(_lw_true))
    print("true ensemble populations: " + ", ".join(f"{db} {p:.3f}" for (db, _), p in zip(TRUE_STRUCTURES, p_true)))
    bpp_true = p_true @ pairing_states_true
    corr_true = (pairing_states_true.T * p_true) @ pairing_states_true - np.outer(bpp_true, bpp_true)
    # one structure per read, drawn with probability p_true (same distribution as the inverse-CDF loop
    # u ~ U(0,1), pick the first k with u < cumsum(p_true)[k])
    true_idx = np.random.choice(len(p_true), size=N_READS, p=p_true)
    paired = pairing_states_true[true_idx].astype(bool)
    print("sampled fractions: " + ", ".join(f"{f:.3f}" for f in np.bincount(true_idx, minlength=len(p_true)) / N_READS))
reacted = np.random.random(paired.shape) < np.where(paired, p_react_paired, p_react_unpaired)
mutated = np.random.random(paired.shape) < np.where(reacted, 1 - prob_fn, prob_fp)

mut_counts = mutated.sum(0)
read_counts = N_READS * np.ones(N, dtype=int)
counts_df = pd.DataFrame({'Sample': f'synthetic_CAhairpin_{CONC_MM}mM', 'mut_count_per_site': mut_counts,
                          'number_of_reads': read_counts})
counts_df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_counts.csv'), index=False)
np.save(os.path.join(OUTPUT_DIR, 'eps_b.npy'), eps_b)

# %%
# =============================================================================
# 1.1 Plot pairing state and correlations of true and prior ensembles
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
ax.plot(bpp_prior, 'k:', label='prior (penalty only)')
ax.plot(bpp_true, 'k-', lw=2, alpha=0.4, label='true (biased at sites)')
ax.set_xticks(range(N))
ax.set_xlabel('Site')
ax.set_ylabel('Base Pairing Probability')
ax.legend()
fig.tight_layout()
plt.show()

# Figure 2: connected pairing correlation matrix (prior)
corr_prior = mol_prior.pairing_correlation_matrix() - np.matmul(np.atleast_2d(bpp_prior).T, np.atleast_2d(bpp_prior))

vmax = max(np.abs(corr_prior).max(), np.abs(corr_true).max())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, corr, title in zip(axes, (corr_prior, corr_true), ('prior', 'true')):
    im = ax.imshow(corr, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax.set_title(f'pairing correlation matrix ({title})')
    ax.set_xlabel('Site')
axes[0].set_ylabel('Site')
fig.colorbar(im, ax=axes, label='connected pairing correlation')
plt.show()

# %%
# =============================================================================
# 1.2 Save the read-level data (multiple mutations per read), com_maxent.py format
# =============================================================================
# mutations on masked (G/U) sites are ignored by the likelihood, so drop them before deduplicating
mutated_obs = mutated & position_mask[None, :]
unique_reads, read_occurrences = np.unique(mutated_obs, axis=0, return_counts=True)
reads_int_list = [np.where(r)[0].tolist() for r in unique_reads]
N_reads_unique = read_occurrences.astype(float)

reads_df = pd.DataFrame({'mut_positions': reads_int_list, 'occurrences': read_occurrences})
reads_df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_reads.csv'), index=False)
np.save(os.path.join(OUTPUT_DIR, 'synthetic_reads_mask.npy'), position_mask)

# consistency with the per-site counts at the observed sites
site_counts_from_reads = np.zeros(N)
for r, n in zip(reads_int_list, read_occurrences):
    site_counts_from_reads[r] += n
assert np.array_equal(site_counts_from_reads[position_mask], mut_counts[position_mask])
print(f"saved {len(reads_int_list)} unique reads ({N_READS} total) to synthetic_reads.csv")

# %%
# =============================================================================
# 2. maxent (chi-squared), alpha=1, physical params fixed
# =============================================================================
bpp_target, sigma_squared = compute_expansion_parameters(mut_counts.astype(float), read_counts.astype(float), a_i, b_i)
boundaries_mask = bounds_array_with_mask(N, position_mask, BOUNDS_SC)

res_chi2, bpp_maxent_out = maxent(SEQ, bpp_target, sigma_squared=sigma_squared, penalty=penalty,
                                  boundaries=boundaries_mask, alpha=ALPHA_MAXENT, T=TEMP_C)
lambda_sc = res_chi2.x
np.savetxt(os.path.join(OUTPUT_DIR, f'lambda_sc_maxent_chi2_alpha{ALPHA_MAXENT:g}.txt'), lambda_sc)
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
# =============================================================================
# 4. Read-level (multi-mutation) optimization of the maxent-corrected suboptimal ensemble
#    (com_maxent.py forward model: P = P0 W / Z, KL w.r.t. the penalized Vienna prior)
# =============================================================================
ALPHA_READS_LIST = [1, 10, 100]

S = states.shape[0]
R = len(reads_int_list)
# P0 = maxent-corrected ensemble on the suboptimal set (logweights), KL reference = penalized Vienna prior (logweights_ref)
mutation_matrix = cm['forward_model_fixed_params'](states, a_i, b_i, mask=position_mask)
X, D, base = cm['precompute_read_factors'](reads_int_list, mutation_matrix, position_mask)
USE_DENSE = R * S <= 2e7
if USE_DENSE:
    logM = cm['precompute_log_read_likelihood'](reads_int_list, mutation_matrix, position_mask)

def make_obj(alpha):
    if USE_DENSE:
        return lambda W: cm['forward_model'](N_reads_unique, W, logweights, logM, alpha=alpha, logweights_ref=logweights_ref)
    return lambda W: cm['forward_model_chunked'](N_reads_unique, W, logweights, X, D, base, alpha=alpha, logweights_ref=logweights_ref)

# finite-difference check of the gradient (wrt W) before optimizing
_rng = np.random.default_rng(0)
_W = np.exp(0.3 * _rng.standard_normal(S))
_dir = _rng.standard_normal(S)
_obj = make_obj(ALPHA_READS_LIST[1])
_h = 1e-6
_fd = (_obj(_W + _h * _dir)[0] - _obj(_W - _h * _dir)[0]) / (2 * _h)
_an = _obj(_W)[1] @ _dir
print(f"gradient check: FD={_fd:.6e} analytic={_an:.6e} rel.err={abs(_fd - _an) / abs(_an):.2e}")

bounds = [(np.finfo(np.float64).eps, None)] * S
pairing_states = (states != -1).astype(float)
bpp_reads, P_reads = {}, {}
nll_of = lambda W: make_obj(0)(W)[0]
print(f"reads: {R} unique, {S} structures, {'dense' if USE_DENSE else 'chunked'} logM; "
      f"-logL at W=1 (maxent ensemble) = {nll_of(np.ones(S)):.2f}")
for a in ALPHA_READS_LIST:
    res_r = minimize(make_obj(a), np.ones(S), jac=True, method='L-BFGS-B', bounds=bounds)
    lw_r = np.log(res_r.x) + logweights
    P_reads[a] = np.exp(lw_r - logsumexp(lw_r))
    bpp_reads[a] = P_reads[a] @ pairing_states
    print(f"alpha={a:g}: success={res_r.success} nit={res_r.nit} loss={res_r.fun:.2f} -logL={nll_of(res_r.x):.2f} "
          f"KL(P||P_ref)={kl_P(P_reads[a]):.4f} max|bpp-bpp_maxent|={np.abs(bpp_reads[a] - bpp_subopt_start).max():.4f}")

# %%
# =============================================================================
# 4.1 Compare pairing probabilities and (connected) pairing correlations with the true ensemble
# =============================================================================
def connected_corr(P):
    bpp = P @ pairing_states
    return (pairing_states.T * P) @ pairing_states - np.outer(bpp, bpp)

corr_maxent_full = mol_fit.pairing_correlation_matrix() - np.outer(bpp_fit, bpp_fit)
corr_maxent_trunc = connected_corr(P_fit)
corr_reads = {a: connected_corr(P_reads[a]) for a in ALPHA_READS_LIST}

obs = np.where(position_mask)[0]
candidates = {'prior': corr_prior, 'maxent (full ensemble)': corr_maxent_full, 'maxent (subopt set)': corr_maxent_trunc}
candidates.update({f'reads subopt, alpha={a:g}': corr_reads[a] for a in ALPHA_READS_LIST})
bpp_of = {'prior': bpp_prior, 'maxent (full ensemble)': bpp_fit, 'maxent (subopt set)': bpp_subopt_start}
bpp_of.update({f'reads subopt, alpha={a:g}': bpp_reads[a] for a in ALPHA_READS_LIST})

rows = []
for name, C in candidates.items():
    dC = C - corr_true
    rows.append({'ensemble': name,
                 'bpp_max_abs_err': np.abs(bpp_of[name] - bpp_true).max(),
                 'bpp_rmse': np.sqrt(np.mean((bpp_of[name] - bpp_true) ** 2)),
                 'corr_frob_all': np.linalg.norm(dC),
                 'corr_maxabs_all': np.abs(dC).max(),
                 'corr_frob_observed': np.linalg.norm(dC[np.ix_(obs, obs)]),
                 'corr_maxabs_observed': np.abs(dC[np.ix_(obs, obs)]).max()})
metrics = pd.DataFrame(rows).set_index('ensemble')
print(metrics.to_string(float_format=lambda v: f'{v:.4f}'))
metrics.to_csv(os.path.join(OUTPUT_DIR, 'corr_metrics.csv'))

for a in ALPHA_READS_LIST:
    summary[f'bpp_reads_alpha{a:g}'] = bpp_reads[a]
summary.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index_label='pos0')

# Figure A: pairing probabilities
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
ax.plot(bpp_prior, 'k:', label='prior (penalty only)')
ax.plot(bpp_true, 'k-', lw=2, alpha=0.4, label='true')
ax.plot(bpp_fit, 'C0o-', label='maxent chi2')
for i, a in enumerate(ALPHA_READS_LIST):
    ax.plot(bpp_reads[a], f'C{i + 1}--', marker='s', mfc='none', label=f'reads subopt, alpha={a:g}')
ax.set_xticks(range(N)); ax.set_xticklabels(list(SEQ), fontsize=7)
ax.set_ylabel('pairing probability'); ax.legend(fontsize=7)
fig.tight_layout()
plt.show()

# Figure B: correlation matrices (top) and difference to the true one (bottom), shared symmetric color scales
ncol = 2 + len(ALPHA_READS_LIST)
mats = [('true', corr_true), ('maxent (subopt set)', corr_maxent_trunc)] + \
       [(f'reads subopt, alpha={a:g}', corr_reads[a]) for a in ALPHA_READS_LIST]
vmax = max(np.abs(M_).max() for _, M_ in mats)
vmax_d = max(np.abs(M_ - corr_true).max() for _, M_ in mats[1:])
fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7), sharex=True, sharey=True)
for j, (title, M_) in enumerate(mats):
    im1 = axes[0, j].imshow(M_, cmap='bwr', vmin=-vmax, vmax=vmax)
    axes[0, j].set_title(title, fontsize=8)
    if j == 0:
        axes[1, j].axis('off')
    else:
        im2 = axes[1, j].imshow(M_ - corr_true, cmap='bwr', vmin=-vmax_d, vmax=vmax_d)
        axes[1, j].set_title(f'{title} - true', fontsize=8)
fig.colorbar(im1, ax=axes[0, :], label='connected pairing correlation', shrink=0.8)
fig.colorbar(im2, ax=axes[1, :], label='difference to true', shrink=0.8)
plt.show()

# %%
# =============================================================================
# 5. Same read-level optimization, starting from the plain Vienna suboptimal ensemble (no maxent correction)
#    P0 = KL reference = penalized Vienna prior on its own suboptimal set (states_0, logweights_0 from section 3)
# =============================================================================
def run_reads_optimization(states_, logweights0_, logweights_ref_, alphas):
    """Read-likelihood reweighting (com_maxent forward model) of a suboptimal set, one L-BFGS-B run per alpha.
    Returns {alpha: P} with P the normalized optimized probabilities of the structures."""
    S_ = states_.shape[0]
    mm = cm['forward_model_fixed_params'](states_, a_i, b_i, mask=position_mask)
    if R * S_ <= 2e7:
        logM_ = cm['precompute_log_read_likelihood'](reads_int_list, mm, position_mask)
        obj_of = lambda a: (lambda W: cm['forward_model'](N_reads_unique, W, logweights0_, logM_,
                                                           alpha=a, logweights_ref=logweights_ref_))
    else:
        X_, D_, base_ = cm['precompute_read_factors'](reads_int_list, mm, position_mask)
        obj_of = lambda a: (lambda W: cm['forward_model_chunked'](N_reads_unique, W, logweights0_, X_, D_, base_,
                                                                   alpha=a, logweights_ref=logweights_ref_))
    nll = lambda W: obj_of(0)(W)[0]
    kl = lambda P: P @ (np.log(P) - logweights_ref_)
    print(f"  {S_} structures, -logL at W=1 = {nll(np.ones(S_)):.2f}")
    out = {}
    for a in alphas:
        res_ = minimize(obj_of(a), np.ones(S_), jac=True, method='L-BFGS-B',
                        bounds=[(np.finfo(np.float64).eps, None)] * S_)
        lw_ = np.log(res_.x) + logweights0_
        out[a] = np.exp(lw_ - logsumexp(lw_))
        print(f"  alpha={a:g}: success={res_.success} nit={res_.nit} loss={res_.fun:.2f} "
              f"-logL={nll(res_.x):.2f} KL(P||P_ref)={kl(out[a]):.4f}")
    return out

logweights_0_norm = logweights_0 - logsumexp(logweights_0)
pairing_states_0 = (states_0 != -1).astype(float)
coverage_0 = mol_prior.suboptimal_coverage(DELTA_SUBOPT)
print(f"reads subopt from the Vienna (no sc) ensemble, delta={DELTA_SUBOPT} kcal/mol, coverage of Z = {coverage_0:.6f}")
P_reads_0 = run_reads_optimization(states_0, logweights_0_norm, logweights_0_norm, ALPHA_READS_LIST)
bpp_reads_0 = {a: P @ pairing_states_0 for a, P in P_reads_0.items()}

def connected_corr_of(P, pstates):
    bpp = P @ pstates
    return (pstates.T * P) @ pstates - np.outer(bpp, bpp)

corr_vienna_trunc = connected_corr_of(np.exp(logweights_0_norm), pairing_states_0)
corr_reads_0 = {a: connected_corr_of(P, pairing_states_0) for a, P in P_reads_0.items()}
corr_subopt_from_prior = connected_corr_of(np.exp(lw0_opt - logsumexp(lw0_opt)), pairing_states_0)

# %%
# =============================================================================
# 5.1 Compare: maxent-corrected start vs Vienna start (pairing probabilities and correlations)
# =============================================================================
candidates.update({'Vienna (subopt set)': corr_vienna_trunc,
                   'per-site subopt from Vienna': corr_subopt_from_prior})
candidates.update({f'reads subopt from Vienna, alpha={a:g}': corr_reads_0[a] for a in ALPHA_READS_LIST})
bpp_of.update({'Vienna (subopt set)': np.exp(logweights_0_norm) @ pairing_states_0,
               'per-site subopt from Vienna': bpp_subopt_from_prior})
bpp_of.update({f'reads subopt from Vienna, alpha={a:g}': bpp_reads_0[a] for a in ALPHA_READS_LIST})

rows = []
for name, C in candidates.items():
    dC = C - corr_true
    rows.append({'ensemble': name,
                 'bpp_max_abs_err': np.abs(bpp_of[name] - bpp_true).max(),
                 'bpp_rmse': np.sqrt(np.mean((bpp_of[name] - bpp_true) ** 2)),
                 'corr_frob_all': np.linalg.norm(dC),
                 'corr_maxabs_all': np.abs(dC).max(),
                 'corr_frob_observed': np.linalg.norm(dC[np.ix_(obs, obs)]),
                 'corr_maxabs_observed': np.abs(dC[np.ix_(obs, obs)]).max()})
metrics = pd.DataFrame(rows).set_index('ensemble')
print(metrics.to_string(float_format=lambda v: f'{v:.4f}'))
metrics.to_csv(os.path.join(OUTPUT_DIR, 'corr_metrics.csv'))

for a in ALPHA_READS_LIST:
    summary[f'bpp_reads_from_vienna_alpha{a:g}'] = bpp_reads_0[a]
summary.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index_label='pos0')

# Figure C: pairing probabilities, maxent start (left) vs Vienna start (right)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
for ax, title, bpp_start, bpp_runs in ((axes[0], 'start: maxent-corrected ensemble', bpp_fit, bpp_reads),
                                        (axes[1], 'start: Vienna ensemble (no sc)', bpp_prior, bpp_reads_0)):
    ax.plot(bpp_prior, 'k:', label='prior (penalty only)')
    ax.plot(bpp_true, 'k-', lw=2, alpha=0.4, label='true')
    ax.plot(bpp_fit, 'C0o-', ms=4, label='maxent chi2')
    for i, a in enumerate(ALPHA_READS_LIST):
        ax.plot(bpp_runs[a], f'C{i + 1}--', marker='s', mfc='none', label=f'reads subopt, alpha={a:g}')
    ax.set_xticks(range(N)); ax.set_xticklabels(list(SEQ), fontsize=7)
    ax.set_title(title)
    ax.legend(fontsize=7)
axes[0].set_ylabel('pairing probability')
fig.tight_layout()
plt.show()

# Figure D: correlations from the Vienna start (top) and difference to the true one (bottom)
mats = [('true', corr_true), ('Vienna (subopt set)', corr_vienna_trunc)] + \
       [(f'reads from Vienna, alpha={a:g}', corr_reads_0[a]) for a in ALPHA_READS_LIST]
ncol = len(mats)
vmax = max(np.abs(M_).max() for _, M_ in mats)
vmax_d = max(np.abs(M_ - corr_true).max() for _, M_ in mats[1:])
fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7), sharex=True, sharey=True)
for j, (title, M_) in enumerate(mats):
    im1 = axes[0, j].imshow(M_, cmap='bwr', vmin=-vmax, vmax=vmax)
    axes[0, j].set_title(title, fontsize=8)
    if j == 0:
        axes[1, j].axis('off')
    else:
        im2 = axes[1, j].imshow(M_ - corr_true, cmap='bwr', vmin=-vmax_d, vmax=vmax_d)
        axes[1, j].set_title(f'{title} - true', fontsize=8)
fig.colorbar(im1, ax=axes[0, :], label='connected pairing correlation', shrink=0.8)
fig.colorbar(im2, ax=axes[1, :], label='difference to true', shrink=0.8)
plt.show()

# %%
