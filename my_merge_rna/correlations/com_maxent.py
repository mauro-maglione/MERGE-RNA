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

_repo_root = pathlib.Path(__file__).resolve().parents[2]
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
    """Compute the forward model for the mutation probabilities."""
    # the starting penalty for paired sites, from DMS presence and lambda_sc if available
    if lambda_sc is not None:
        lambdas1d = penalty * np.ones(len(seq)) + lambda_sc
    else:
        lambdas1d = penalty * np.ones(len(seq))

    # define molecule used for sub_optimal structure sampling
    # default parameters should be turner 2004
    mol = bussi.Molecule(seq, lambdas1d=lambdas1d, temperature=temperature_C+273.15) 

    coverage_of_part_func = mol.suboptimal_coverage(delta)
    structures = mol.suboptimal_structures(delta)

    # in principle one can sample at a diff temp then the one used for weights
    states, logweights = bussi.suboptimal_to_numpy(suboptimal = structures, temperature=temperature_C+273.15) 
    return coverage_of_part_func, states, logweights

def forward_model_fixed_params(states, a_i, b_i, mask = None):
    """Compute the matrix that needs to be used for all logweights and reads,
    this stores the information about the sampled structures once.
    Returns a matrix of shape (N_samples, N_seq) with the mutation probabilities for each sampled structure and each sequence position.
    """
    if mask is None:
        mask = np.ones(len(a_i), bool)

    mutation_probs = np.zeros(states.shape)
    for s in range(states.shape[0]):
        for site in range(states.shape[1]):
            if mask[site]:
                if states[s,site] == -1:
                    mutation_probs[s,site] = a_i[site]
                else:
                    mutation_probs[s,site] = a_i[site] + b_i[site]
        
    return mutation_probs

def forward_model(reads, N_reads, logweights, logweights_0, mutation_probs, mask = None):
    """Compute loss and derivatives approximately using ensemble of suboptimal structures.
    Masked positions (mask False) are unobserved: they contribute a factor 1 to every read."""

    # I'm assuming that reads comes in a form that has a the mutated sites -
    # - in reads and the number of reads with these mutations in N_reads
    if mask is None:
        mask = np.ones(mutation_probs.shape[1], bool)
    observed_sites = set(np.where(mask)[0])

    normalization = np.sum(np.exp(logweights))
    log_normalization = np.log(normalization)

    # compute the forward model for the mutation probabilities
    loss = 0.
    grad = np.zeros(len(logweights),float)

    for r in range(len(reads)):
        read_likelihood = 0.
        der_read_likelihood = np.zeros(len(logweights),float)
        for s in range(mutation_probs.shape[0]):
            mut_prob_per_struct = 1.
            for mut_site in reads[r]:
                if mask[mut_site]:
                    mut_prob_per_struct *= mutation_probs[s,mut_site]

            for non_mut_site in observed_sites - set(reads[r]):
                mut_prob_per_struct *= (1-mutation_probs[s,non_mut_site])

            read_likelihood += np.exp(logweights[s] - log_normalization) * mut_prob_per_struct   

            der_read_likelihood[s] += mut_prob_per_struct * np.exp(logweights_0[s]-log_normalization)
            der_read_likelihood += - mut_prob_per_struct * np.exp(logweights_0 + logweights[s] - 2*log_normalization) 

        # i'm considering a negative log likelihood

        loss += - N_reads[r]*np.log(read_likelihood)
        grad += - N_reads[r]*der_read_likelihood / read_likelihood
            
    return loss, grad
# %%

mol_test = bussi.Molecule(seq, lambdas1d=penalty*np.ones(len(seq)))
_,min_free_en = mol_test.mfe()
tot_free_en = mol_test.total_free_energy()

print(f"The minimum free energy is {min_free_en} and the total one {tot_free_en} in kcal/mol")

# %%
# delta 10 -> 145625 struct, 0.998 coverage, delta 5 -> 815 struct, 0.944 coverage
delta_kcalpermol = 4
coverage_of_part_func, states, logweights = structures_sampling(seq, penalty, delta_kcalpermol)

# %%
path_synthetic_data = os.path.join(os.environ['RNA_STRUCT_HOME'], 'data', 'designed_sequence', 'synthetic_mutation_experiments')
df = pd.read_csv(os.path.join(path_synthetic_data, f"synthetic_{CONC_MM}mM_r{REP_NUMBER}.csv"))
# mask saved by generate_synth_mut_experiment.py (True = observed/fitted position, kept indexing of exp.seq)
position_mask = np.load(os.path.join(path_synthetic_data, f"synthetic_{CONC_MM}mM_r{REP_NUMBER}_mask.npy"))
assert len(position_mask) == len(seq), f"mask length {len(position_mask)} != sequence length {len(seq)}"

mutation_matrix = forward_model_fixed_params(states, a_i, b_i, mask=position_mask)

reads_str = df['mut_positions'].values
reads_int_list = [ast.literal_eval(read) for read in reads_str]
N_reads = df['occurrences'].values

test_logcorrections = 10. * np.random.randn(len(logweights))
test_logweights = test_logcorrections+logweights
test_logweights = test_logweights-np.log(np.sum(np.exp(test_logweights)))

test_loss, test_grad = forward_model(reads_int_list, N_reads, test_logweights, logweights, mutation_matrix, mask=position_mask)
# %%
