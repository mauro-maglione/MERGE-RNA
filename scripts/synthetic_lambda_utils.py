"""Reusable pieces for building a synthetic experiment with an enforced (exact)
pairing probability at chosen positions, and for fitting lambda_sc restricted to
those same positions.

Split out of scripts/small_analysis.py so these can be imported (e.g. from
my_merge_rna/) without executing that script's interactive `# %%` cells.
"""
import os
import sys
import math
from dataclasses import dataclass

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd

from merge_rna import Experiment, MultiSystemsFit
from merge_rna.fit import convert_p_bind_dict_to_1d


def convert_p_bind_1d_to_dict(p_bind_1d, DMS_mode=True):
    '''Convert the 1D array of p_bind to a dictionary.
    If DMS_mode is True, input is a 4-element array and will put 0A, 0C to 0 and 1G, 1U to equal to  0G, 0U.'''
    list_partial_keys = ["p_bind_A_", "p_bind_C_", "p_bind_G_", "p_bind_U_"]
    if DMS_mode:
        p_bind_dict = {incom_key+"0": value for incom_key, value in zip(list_partial_keys, p_bind_1d)}
        p_bind_dict["p_bind_A_"+"1"] = 0
        p_bind_dict["p_bind_C_"+"1"] = 0
        p_bind_dict["p_bind_G_"+"1"] = p_bind_dict["p_bind_G_"+"0"]
        p_bind_dict["p_bind_U_"+"1"] = p_bind_dict["p_bind_U_"+"0"]
    else:
        list_keys = []

        for str in ["0", "1"]:
            list_keys.append(str.join(list_partial_keys))

        p_bind_dict = {key: value for key, value in zip(list_keys, p_bind_1d)}
    return p_bind_dict


def mut_prob_condit_contact_merge_rna(seq,eps_b,*,
                            conc_mM,temp_C,
                            #p_bind_A_0, p_bind_A_1, p_bind_U_0, p_bind_U_1, p_bind_C_0, p_bind_C_1,p_bind_G_0, p_bind_G_1,
                            p_bind,
                            mu_r, p_b, m0, m1):

    kBT = (273.15 + temp_C) * (1.98717/1000) # kcal/mol, matches fit.py's self.kBT = self.temp_K * kb
    beta = 1/kBT
    mu_j = mu_r + kBT * np.log((conc_mM+.1)/1000) if conc_mM is not None else mu_r
    f_mu = np.exp(beta*mu_j)/(1-np.exp(beta*mu_j))
    f_mu_prime = np.exp(beta*(mu_j-p_b))/(1-np.exp(beta*(mu_j-p_b)))

    dict_pbind = convert_p_bind_1d_to_dict(p_bind)

    def compute_penalty_m(mu, p_b_):
        '''Compute the penalty for paired bases in kcal/mol'''
        mu_prime = mu-p_b_
        penalty_in_kbt = -math.log((1+math.exp(beta*mu_prime))/(1+math.exp(beta*mu)))
        penalty_in_kcal_per_mol = penalty_in_kbt * kBT #TBC: to be checked
        return penalty_in_kcal_per_mol

    def compute_mu_dependent_quantities(mu, p_b_):
        '''Returns some key quantities that depend on mu: e^beta*mu, e^beta*mu', p(c_b|s_b,n_b)'''
        mu_prime = mu - p_b_
        e_beta_mu = math.exp(beta*mu)
        e_beta_mu_prime = math.exp(beta*mu_prime)
        e_beta_mu_norm = e_beta_mu / (1 + e_beta_mu)
        e_beta_mu_prime_norm = e_beta_mu_prime / (1 + e_beta_mu_prime)
        # probability of chemical binding
        p_cb_given_sb_nb = dict() # p(cb=1|s_b,n_b) -> p_cb[(s_b,n_b)] , cb = 1 is implicit.
        for s_b in [0, 1]:
            physical_binding_probability = e_beta_mu_norm if s_b == 0 else e_beta_mu_prime_norm
            for n_b in ['A', 'C', 'G', 'U']:
                p_cb_given_sb_nb[(s_b, n_b)] = dict_pbind["p_bind_"+n_b+"_"+str(s_b)] * physical_binding_probability # p(cb=0|s_b,n_b) = 1 - p(cb=1|s_b,n_b)
        return e_beta_mu, e_beta_mu_prime, p_cb_given_sb_nb

    def compute_p_cb(mu, p_b_, pairing_probs):
        '''Compute p(c_b) for the whole sequence'''
        # compute mu dependent quantities (and p(c_b|s_b,n_b))
        _, _, p_cb_given_sb_nb = compute_mu_dependent_quantities(mu, p_b_)
        # apply penalty for paired bases
        penalty = compute_penalty_m(mu, p_b_)
        # compute p(s_b)
        try:
            assert 0 <= penalty <= 5
        except AssertionError:
            print('Warning, penalty is not in [0,5]')
            print(f'penalty: {penalty}, mu: {mu}, p_b: {p_b}')
        # compute p(c_b)
        p_cb = np.zeros(len(seq))
        for i,nb in enumerate(seq):    # weighted sum over s_b=0,1
            p_cb[i] = p_cb_given_sb_nb[(1, nb)]*pairing_probs[i] + p_cb_given_sb_nb[(0, nb)]*(1-pairing_probs[i])
        return p_cb

    # 2a: mutation profile model
    def compute_mutation_profile(m0, m1,eps_b, p_cb):
        '''Compute the mutation profile predicted by the model with the current parameters.
        Later: compute its derivatives'''
        # compute mutation profile
        Mb = 1 - np.exp(-eps_b) * (math.exp(-m0) +  p_cb*(math.exp(-m1) - math.exp(-m0)))
        return Mb

    a_merge = compute_mutation_profile(m0, m1, eps_b, compute_p_cb(mu_j, p_b, np.zeros(len(seq))))
    a_plus_b_merge = compute_mutation_profile(m0, m1, eps_b, compute_p_cb(mu_j, p_b, np.ones(len(seq))))
    return a_merge, a_plus_b_merge-a_merge


def create_exp_synthetic_selected_nucleot(positions, pops, custom_name,*, seq = None, params_dict = None, noise=True, covs=10000, eps_b=None, conc_mM = 50, temp_C = 37):
    """
    Create a synthetic experiment for the bistable sequence that I designed for Redmond

    seq: nucleot seq
    positions: array of positions of nucleo to which we enforce pops (actual pos starting from 1)
    pops: array of float, relative population for the corresponding posit in positions
    noise: bool, whether to add noise to the synthetic data
    """

    if seq is None:
        # seq from redmond's mail
        seq = "TAATACGACTCACTATAgggCATTATGCCACAGCCAATCCCCACTTCAACTCACAACTATTCCAAAAAATTGGAATAGTTGTGAGTTGAAGTGGGGATTAAAAAATCCCCACTTCAACTCACAACTATTCCAACCTCCAGCAGACCAT"
        seq = seq.upper().replace('T', 'U')

    if params_dict is None:
        params_1D = np.loadtxt("fits_paper/structured_rnas/physical_params_only_crossval/red_crossval_bact_RNaseP_typeA_tetrahymena_ribozyme_V_chol_gly_riboswitch/params1D.txt")
        exp = [Experiment(path_) for path_ in Experiment.paths_to_redmond_ivt_data_txt if Experiment(path_).conc_mM == 0]
        exp = exp[0]
        multi_exp = MultiSystemsFit([exp], validation_exps=None, infer_1D_sc=False, skip_output_setup=True)
        params_dict = multi_exp.pack_params(params_1D, multi_exp.systems[0])
        # pack_params converts p_bind into a dict keyed by (s_b, nt) tuples, but
        # mut_prob_condit_contact_merge_rna expects the raw 4-element 1D array.
        params_dict['p_bind'] = convert_p_bind_dict_to_1d(params_dict['p_bind'], DMS_mode=True)
        # pack_params always injects lambda_sc (None here, since infer_1D_sc=False),
        # which mut_prob_condit_contact_merge_rna's signature doesn't accept.
        del params_dict['lambda_sc']
        del params_1D
        del exp
        del multi_exp

    kwargs = {}
    kwargs['conc_mM'] = conc_mM
    kwargs['temp_C'] = temp_C
    if eps_b is None:
        eps_b = np.zeros(len(seq),float)

    a_from_merge, b_from_merge = mut_prob_condit_contact_merge_rna(seq, eps_b, **kwargs, **params_dict)

    # Create the synthetic experiment
    system_name = 'custom_synthetic_'+custom_name
    exp = Experiment(seq=seq, reagent='DMS synthetic', system_name=system_name, **kwargs)

    coverage = np.zeros(len(seq), dtype=int)
    if isinstance(covs, int):
        coverage[positions-1] = covs*np.ones(len(positions))
    elif isinstance(covs, np.ndarray):
        coverage = covs.copy()
    else:
        raise ValueError("Coverage has to be int or np.array")

    # Scatter the enforced pairing probabilities into a full-length array
    # (0 = fully unpaired background at every position we don't constrain).
    full_pops = np.zeros(len(seq))
    full_pops[positions-1] = pops
    mut_rate = a_from_merge + b_from_merge*full_pops

    # Add noise to mutation rates
    if noise:
        mut_count = np.random.binomial(coverage, mut_rate)
    else:
        mut_count = np.round(mut_rate * coverage).astype(int)

    short_description = "number of mutated sites: "+str(len(positions))
    data = {'Sample' : short_description,
            'mut_count' : mut_count,
            'wt_count' : coverage - mut_count,
            'total_count' : coverage,
            'mut_rate' : mut_rate,
            'ref_nt' : list(seq),
            'pos' : range(1, len(seq)+1)}
    df = pd.DataFrame(data)
    exp.df = df
    exp.raw_df = df.copy()

    custom_mask = np.zeros(len(seq),bool)
    for index in positions:
        custom_mask[index-1] = True

    return exp, custom_mask, a_from_merge, b_from_merge


@dataclass
class MultiSystemsFitFixedLambdaPositions(MultiSystemsFit):
    """MultiSystemsFit variant where lambda_sc is only free to vary at chosen
    positions per system; everywhere else it is pinned to 0 via zero-width bounds.

    custom_mask alone is NOT enough to achieve this: dps_dlambda_sc is a dense
    N x N Jacobian (RNA folding is globally coupled), so a masked-out lambda_sc
    still gets a nonzero gradient through its effect on pairing probabilities at
    masked-in positions (see merge_rna/fit.py loss_and_grad). Only an explicit
    bound restricts which lambda_sc entries actually move during optimization.

    free_lambda_positions: {system_name: iterable of 0-indexed sequence positions
        allowed to vary}.
    """
    free_lambda_positions: dict = None

    def initialize_guess_and_bounds(self, start_value=0.1, guess=None):
        initial_guess, bounds = super().initialize_guess_and_bounds(start_value=start_value, guess=guess)
        if self.infer_1D_sc and self.free_lambda_positions:
            for system in self.systems:
                free = set(self.free_lambda_positions.get(system.sys_name, ()))
                for local_idx, global_idx in enumerate(self.lambdas_indices[system.sys_name]):
                    if local_idx not in free:
                        bounds[global_idx] = (0.0, 0.0)
        return initial_guess, bounds


class MultiSystemsFitMaskedLambdaBounds(MultiSystemsFit):
    """MultiSystemsFit variant where lambda_sc is pinned to 0 (zero-width bounds)
    at every position excluded by custom_mask, for every system.

    custom_mask alone only affects the loss (see ExperimentFit._parse_custom_mask /
    position_mask in merge_rna/fit.py) -- it does NOT stop a masked-out lambda_sc from
    moving during optimization, since dps_dlambda_sc is a dense Jacobian (RNA folding
    is globally coupled). Only an explicit bound restricts which lambda_sc entries
    actually vary, same reasoning as MultiSystemsFitFixedLambdaPositions above.
    """

    def initialize_guess_and_bounds(self, start_value=0.1, guess=None):
        initial_guess, bounds = super().initialize_guess_and_bounds(start_value=start_value, guess=guess)
        if self.infer_1D_sc:
            for system in self.systems:
                if system.custom_mask is None:
                    continue
                mask = system.exp_fits_all[0]._parse_custom_mask(system.custom_mask)
                for local_idx, global_idx in enumerate(self.lambdas_indices[system.sys_name]):
                    if not mask[local_idx]:
                        bounds[global_idx] = (0.0, 0.0)
        return initial_guess, bounds
