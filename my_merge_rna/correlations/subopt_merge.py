# %%
import os
import pathlib
import sys
import json
import time
import pickle
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    _repo_root = pathlib.Path(__file__).resolve().parents[2]
except NameError:
    _cwd = pathlib.Path.cwd().resolve()
    _repo_root = next(p for p in [_cwd, *_cwd.parents] if (p / 'merge_rna').is_dir())
    
os.environ.setdefault('RNA_STRUCT_HOME', str(_repo_root))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])
os.chdir(str(_repo_root))

import numpy as np
import pandas as pd
import ast
import matplotlib
import matplotlib.pyplot as plt

from merge_rna import Experiment, MultiSystemsFit

import RNA
import bussilab.rna2d as bussi
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

# %%
# =============================================================================
# Load a single NewseqWT experiment (by concentration + replicate)
# =============================================================================
CONC_MM = 80
REP_NUMBER = 1

_PHASE1_OUTPUT_DIR = os.path.join(os.environ['RNA_STRUCT_HOME'], 'fits', 'designed_sequence',
                                   'outputs', 'experimental_turner', 'phase1', 'newseqWT')
#phase1_params = os.path.join(_PHASE1_OUTPUT_DIR, 'params1D.txt')
phase1_params = os.path.join(os.environ['RNA_STRUCT_HOME'], 'fits_paper', 'designed_sequence',
                                   'experimental', 'strand_disp_turner.txt')
_eps_b_path = os.path.join(_PHASE1_OUTPUT_DIR, 'detailed', 'NewseqWT_eps_b.npy')
_manifest_path = os.path.join(os.environ['RNA_STRUCT_HOME'], 'data', 'designed_sequence',
                               'newseq_wt_experiments_manifest.csv')


def load_single_newseq_experiment(conc_mM, rep_number, manifest_path=_manifest_path):
    """Look up one NewseqWT experiment (by concentration + replicate) in the manifest and load it."""
    manifest = pd.read_csv(manifest_path)
    match = manifest[(manifest['conc_mM'] == conc_mM) & (manifest['rep_number'] == rep_number)]
    if len(match) != 1:
        raise ValueError(f"Expected exactly one manifest row for conc_mM={conc_mM}, "
                          f"rep_number={rep_number}, found {len(match)}")
    path_txt = os.path.expandvars(match.iloc[0]['path_txt'])
    return Experiment(path_txt)


exp = load_single_newseq_experiment(CONC_MM, REP_NUMBER)
print(f"Loaded {exp.system_name} @ {exp.conc_mM}mM, {exp.temp_C}°C, rep {exp.rep_number}")

seq = exp.seq

# Build a fit struct around this one experiment purely for calculations (never calls .fit()).
fit_struct = MultiSystemsFit([exp], validation_exps=None, infer_1D_sc=False, skip_output_setup=True)

# eps_b is a System-level quantity normally derived from 0mM controls; a single non-0mM
# experiment alone would default it to zeros, so load the one already fitted/saved
# from the full 12-experiment des_seq_fits.py run instead of recomputing it.
_eps_b = np.load(_eps_b_path)
fit_struct.systems[0].eps_b = _eps_b
fit_struct.systems[0].exp_fits_train[0].eps_b = _eps_b

# %%
# =============================================================================
# Compute mutation probability params
# =============================================================================

def compute_mutation_probabilities(fit_struct, phase1_params):
    """Compute mutation probabilities parameters needed in the form M_i = a_i + b_i*s_i."""

    params_1D_ref = np.loadtxt(phase1_params)[:8]
    params_dict_ref = fit_struct.pack_params(params_1D_ref, fit_struct.systems[0])

    mu_r, p_b, m0, m1 = (params_dict_ref['mu_r'], params_dict_ref['p_b'],
                            params_dict_ref['m0'], params_dict_ref['m1'])

    p_bind_dict = params_dict_ref['p_bind']

    exp_fit = fit_struct.systems[0].exp_fits_train[0]

    mu_j = mu_r + exp_fit.kBT * np.log((exp_fit.conc_mM + .1) / 1000) if exp_fit.conc_mM is not None else mu_r
    penalty = exp_fit.compute_penalty_m(mu_j, p_b)

    # compute mu dependent quantities (and p(c_b|s_b,n_b))
    _, _, p_cb_given_sb_nb = exp_fit.compute_mu_dependent_quantities(mu_j, p_b, p_bind_dict) # should work with the p_bind dict that should have the same form of pack_params
    p_reaction_given_unpaired_nb = np.zeros(exp_fit.N_seq)
    p_reaction_given_paired_nb = np.zeros(exp_fit.N_seq)
    for i,nb in enumerate(exp_fit.seq):    # weighted sum over s_b=0,1
        p_reaction_given_unpaired_nb[i] = p_cb_given_sb_nb[(0, nb)] # must be multiplied by *(1-pairing_probs[i])
        p_reaction_given_paired_nb[i] = p_cb_given_sb_nb[(1, nb)] # must be multiplied by *pairing_probs[i] 

    p_cb_unpaired = exp_fit.compute_p_cb(mu_j, p_b, p_bind_dict, np.zeros(exp_fit.N_seq))
    p_cb_paired = exp_fit.compute_p_cb(mu_j, p_b, p_bind_dict, np.ones(exp_fit.N_seq))
    a_i = exp_fit.compute_mutation_profile(m0, m1, exp_fit.eps_b, p_cb_unpaired)
    b_i = exp_fit.compute_mutation_profile(m0, m1, exp_fit.eps_b, p_cb_paired) - a_i

    return p_reaction_given_unpaired_nb, p_reaction_given_paired_nb, a_i, b_i, penalty

p_reaction_given_unpaired_nb, p_reaction_given_paired_nb, a_i, b_i, penalty = compute_mutation_probabilities(fit_struct, phase1_params)

# %%

def structures_sampling(seq, penalty, delta, lambda_sc=None, temperature_C=37):
    """Enumerate the structures that will be reweighted, starting from the mfe up to delta_E."""
    # the starting penalty for paired sites, from DMS presence and lambda_sc if available
    if lambda_sc is not None:
        lambdas1d = penalty * np.ones(len(seq)) + lambda_sc
    else:
        lambdas1d = penalty * np.ones(len(seq))

    # define molecule used for sub_optimal structure sampling
    # default parameters should be turner 2004
    mol = bussi.Molecule(seq, lambdas1d=lambdas1d, temperature=temperature_C+273.15) 

    # coverage_of_part_func = mol.suboptimal_coverage(delta)
    coverage_of_part_func = None
    structures = mol.suboptimal_structures(delta)

    # in principle one can sample at a diff temp then the one used for weights
    states, logweights = bussi.suboptimal_to_numpy(suboptimal = structures, temperature=temperature_C+273.15) 
    return coverage_of_part_func, states, logweights

def structures_sampling_random(seq, penalty, number_samples, lambda_sc=None, temperature_C=37):
    """Sample randomly the structures that will be reweighted."""
    # the starting penalty for paired sites, from DMS presence and lambda_sc if available
    if lambda_sc is not None:
        lambdas1d = penalty * np.ones(len(seq)) + lambda_sc
    else:
        lambdas1d = penalty * np.ones(len(seq))

    # define molecule used for sub_optimal structure sampling
    # default parameters should be turner 2004
    mol = bussi.Molecule(seq, lambdas1d=lambdas1d, temperature=temperature_C+273.15) 

    structures = mol.sample(number_samples)
    energies = np.array([mol.evaluate(struct) for struct in structures])
    log_coverage = logsumexp(-(energies-mol.total_free_energy())*(1/(bussi._KB*(temperature_C+273.15))))

    # in principle one can sample at a diff temp then the one used for weights
    states, logweights = bussi.sample_to_numpy(samples = structures, deduplicate= True) 
    return np.exp(log_coverage), states, logweights

def forward_model_fixed_params(states, a_i, b_i, mask = None):
    """Compute the matrix that needs to be used for all logweights and reads,
    this stores the information about the sampled structures once.
    Returns a matrix of shape (N_samples, N_seq) with the mutation probabilities for each sampled structure and each sequence position.
    """
    if mask is None:
        mask = np.ones(len(a_i), bool)

    # unpaired (-1) -> a_i, paired -> a_i + b_i; masked columns set to 0
    mutation_probs = np.where(states == -1, a_i, a_i + b_i) * mask
    return mutation_probs

def _loss_and_der_coeff_loss(mut_i, read_i, mut_mat, log_p):
    # inputs: observed mutations per site, reads per site, mutation matrix (N_samples x N_seq), new log weights (N_samples,)
    """loss = -sum_i (m_i log M_i +(T_i-m_i) log(1-M_i)) and coeff_q = deriv_bin*(M_qi - M_i),
    with M_i expected mutation rate at site i."""
    aver_mut_rates = np.exp(log_p) @ mut_mat
    log_L = -1*np.sum(np.log(aver_mut_rates) * mut_i + np.log1p(-aver_mut_rates) * (read_i - mut_i))
    coeff_q = -1 * (mut_mat - aver_mut_rates) @ (mut_i / aver_mut_rates - (read_i - mut_i) / (1 - aver_mut_rates))
    return log_L, coeff_q

def forward_model(mut_i, read_i, logweights, logweights_0, logweights_ref, M, mask=None, alpha = 0.1):
    """Compute loss and derivatives approximately using ensemble of suboptimal structures.
    P(s) = P0(s) W(s) / Z with logweights = log(W); the gradient is taken w.r.t. W.
    logM comes from precompute_log_read_likelihood."""

    log_Z = logsumexp(logweights + logweights_0)

    # i'm considering a negative log likelihood
    likelihood, grad_coeff = _loss_and_der_coeff_loss(mut_i, read_i, M, logweights + logweights_0 - log_Z)
    if alpha > 0:
        kl = (np.exp((logweights + logweights_0 - log_Z)) @ (logweights + logweights_0-logweights_ref)) - log_Z
        loss = likelihood + alpha*kl
    else:
        loss = likelihood

    # dlogL_r/dW_j = (P0_j/Z) (M_rj/L_r - 1)
    # (multiply by W = exp(logweights - logweights_0) to get the gradient w.r.t. log W)
    grad = np.exp(logweights_0 - log_Z) * grad_coeff
    if alpha > 0:
        grad += alpha * np.exp(logweights_0 - log_Z) * (logweights + logweights_0 - log_Z - logweights_ref - kl) # logZ and kl should get promoted to be vectors

    return loss, grad

# %%

def do_optimization(logweights, states, seq, penalty, a_i, b_i,
                         data_frame_exp, position_mask,
                         alpha
                         ):

    #if states.shape[0]>200000:
    #    raise ValueError(f"Too many structures n = {states.shape[0]}, optimization is slow")
    
    molecule_ref = bussi.Molecule(seq, lambdas1d=penalty*np.ones(len(seq)))
    logweights_ref = np.apply_along_axis(molecule_ref.evaluate, 1, states)

    mutation_matrix = forward_model_fixed_params(states, a_i, b_i, mask=position_mask)

    mut_counts = data_frame_exp['mut_count_per_site'].values
    read_counts = data_frame_exp['number_of_reads'].values
    mutation_prob_mat = forward_model_fixed_params(states, a_i, b_i, mask=position_mask)

    mut_counts_masked = mut_counts[position_mask]
    read_counts_masked = read_counts[position_mask]
    mutation_prob_mat_masked = mutation_prob_mat[:, position_mask]

    obj = lambda x: forward_model(mut_counts_masked, read_counts_masked, x, logweights, logweights_ref, mutation_prob_mat_masked, alpha=alpha)

    res=minimize(obj,np.zeros(len(logweights)),jac=True,method="L-BFGS-B")
    return res

# id_str = "single_struct"
# id_str = f"r{REP_NUMBER}_true" # this is the forceded population of 80%
id_str = f"r{REP_NUMBER}_larger" # this is the forceded strand_displacement

path_synthetic_data = os.path.join(os.environ['RNA_STRUCT_HOME'], 'data', 'designed_sequence', 'synthetic_mutation_experiments')
df = pd.read_csv(os.path.join(path_synthetic_data, f"synthetic_{CONC_MM}mM_"+id_str+"_counts.csv"))
# mask saved by generate_synth_mut_experiment.py (True = observed/fitted position, kept indexing of exp.seq)
position_mask = np.load(os.path.join(path_synthetic_data, f"synthetic_{CONC_MM}mM_"+id_str+"_mask.npy"))
assert len(position_mask) == len(seq), f"mask length {len(position_mask)} != sequence length {len(seq)}"

delta_kcalpermol = 6

coverage_of_part_func, states, logweights = structures_sampling(seq, penalty, delta_kcalpermol, lambda_sc= np.loadtxt(phase1_params)[8:])
# coverage_of_part_func, states, logweights = structures_sampling(seq, penalty, delta_kcalpermol)

result = do_optimization(logweights, states, seq, penalty, a_i, b_i,
                         df, position_mask, alpha = 0.00001)
# %%

# calculate and plot the pairing probabilities
logweights_fitted = result.x + logweights
pairing_states = np.where(states == -1, 0, 1)

free_en = logsumexp(logweights_fitted)
pairing_probs = np.exp(logweights_fitted.T-free_en) @ pairing_states
pairing_probs_0 = np.exp(logweights.T) @ pairing_states

correlation_matrix = np.zeros([states.shape[1],states.shape[1]])

for struct_ind in np.arange(states.shape[0]):
    state = pairing_states[struct_ind,:][np.newaxis]
    correlation_matrix += (state.T @ state)*np.exp(logweights_fitted[struct_ind]-free_en)

unconn_correl_matrix = correlation_matrix-np.matmul(np.atleast_2d(pairing_probs).T, np.atleast_2d(pairing_probs))

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(pairing_probs, label="fitted")
ax.plot(pairing_probs_0, label="Vienna")
ax.set_xlabel('positions')
ax.set_ylabel('pairing_probability')
ax.legend()
fig.tight_layout()
# %%
