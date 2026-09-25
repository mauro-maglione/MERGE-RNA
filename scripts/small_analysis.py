# %%
# !/u/m/mmaglion/miniconda3/envs/merge-rna-patched-1/bin/python

# %%

import os
import sys
import tempfile

import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import numpy as np
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import RNA
from scipy.optimize import minimize

from merge_rna import Experiment, create_exp_synthetic_comb, MultiSystemsFit
from merge_rna.fit import convert_p_bind_dict_to_1d
from scripts.synthetic_lambda_utils import (
    convert_p_bind_1d_to_dict,
    mut_prob_condit_contact_merge_rna,
    create_exp_synthetic_selected_nucleot,
    MultiSystemsFitFixedLambdaPositions,
)
# %%
positions = np.array([56,120])
pops = np.array([0.8,0.2])
name_exp = "test_1"
TRUE_PARAMS_PATH = "fits_paper/structured_rnas/physical_params_only_crossval/red_crossval_bact_RNaseP_typeA_tetrahymena_ribozyme_V_chol_gly_riboswitch/params1D.txt"

exp, custom_mask, a_merge, b_merge = create_exp_synthetic_selected_nucleot(positions=positions, pops=pops, custom_name=name_exp, noise= False)

# %%
# Build the initial guess: true physical params used to generate the data, plus
# zeros for the lambda_sc block. fit_mode='lambda_only' fixes the physical params
# at whatever the initial guess holds (it zeroes their gradient, it doesn't reset
# their value) so without this they would be pinned at the arbitrary default 0.1
# instead of the true generating values.
true_params_1D = np.loadtxt(TRUE_PARAMS_PATH)
initial_full = np.concatenate([true_params_1D, np.zeros(exp.N_seq)])
guess_fd, guess_path = tempfile.mkstemp(suffix='.txt')
os.close(guess_fd)
np.savetxt(guess_path, initial_full)

fit = MultiSystemsFitFixedLambdaPositions(
    experiments=[exp],
    validation_exps=None,
    infer_1D_sc=True,
    fit_mode='lambda_only',
    custom_mask=custom_mask,
    free_lambda_positions={exp.system_name: positions - 1},
    guess=guess_path,
    do_plots=False,
    print_to_std_out=True,
    root_dir='fits',
    output_suffix='test_lambda_subset',
    overwrite=True,
)
fit.fit()
os.remove(guess_path)

# %%
fitted_params = fit.pack_params(fit.fit_result.x, fit.systems[0])
lambda_sc = fitted_params['lambda_sc']

# %%
target_idx = positions - 1
non_target_mask = np.ones(exp.N_seq, bool)
non_target_mask[target_idx] = False

print("lambda_sc at target positions:", lambda_sc[target_idx])
print("max |lambda_sc| away from target positions:", np.max(np.abs(lambda_sc[non_target_mask])))
assert np.allclose(lambda_sc[non_target_mask], 0.0), "non-target lambda_sc drifted away from 0"

exp_fit = fit.systems[0].exp_fits_all[0]
mu_j = fitted_params['mu_r'] + exp_fit.kBT * np.log((exp_fit.conc_mM + .1) / 1000)
penalty = exp_fit.compute_penalty_m(mu_j, fitted_params['p_b'])
recovered_pairing_probs = exp_fit.get_ps(penalty, lambda_sc, interpolated=False)
print("requested pops:", pops)
print("recovered pairing probability at target positions:", recovered_pairing_probs[target_idx])

print("theroretical mutat probability:", exp.df['mut_rate'].to_numpy()[target_idx])
print("obtained mutat probability:", a_merge[target_idx] + b_merge[target_idx]*recovered_pairing_probs[target_idx])
print("imposed mutat probability:", exp.df['mut_count'].to_numpy()[target_idx]/exp.df['total_count'].to_numpy()[target_idx])

# %%