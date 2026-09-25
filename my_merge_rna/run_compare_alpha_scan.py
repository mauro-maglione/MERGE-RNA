# %%
#!/usr/bin/env python3
"""
Standalone, resumable alpha scan: merge-rna native fit vs maxent (chi-squared, binomial), POP1=0.8.

Meant to be launched from a real terminal (not through an editor/agent session) with nohup, so it
survives regardless of what happens to the launching session:

    cd /u/m/mmaglion/Documents/MERGE-RNA_restored
    conda run -n merge-rna-patched-1 --no-capture-output \\
        setsid nohup python my_merge_rna/run_compare_alpha_scan.py \\
        > my_merge_rna/compare_alpha_scan_pop80.log 2>&1 < /dev/null &
    disown

Every (method, alpha) result is checkpointed to disk immediately after it's computed
(fits_paper/designed_sequence/outputs/compare_alpha_scan_pop80/checkpoints/), so re-running this
script after an interruption skips everything already done and only computes what's missing.

Check progress at any time, from another terminal, without touching the running job:

    tail -f my_merge_rna/compare_alpha_scan_pop80.log
    python my_merge_rna/run_compare_alpha_scan.py --assemble-only

The --assemble-only flag reads whatever checkpoints exist so far, writes scan_results.csv /
status_results.csv / fitted_results.pkl / scan_plot.png, and exits immediately (no fitting).
"""
import os
import sys
import json
import time
import pickle
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: this script never has a display attached
import matplotlib.pyplot as plt
import RNA
from scipy.optimize import minimize

from merge_rna import Experiment, create_exp_synthetic_comb, MultiSystemsFit
from scripts.synthetic_lambda_utils import MultiSystemsFitMaskedLambdaBounds

# %%
# =============================================================================
# Configuration
# =============================================================================
PARAMS1D_PATH = os.path.join(
    'fits_paper', 'designed_sequence', 'synthetic', 
    'reference_physical_params.txt')

POP1 = 0.8
COVERAGE = 10000
SEED = 42  # fixes the one binomial draw used across the whole scan
SEED_VAL = 43  # fixes the one binomial draw used for validation

# ALPHAS = np.concatenate([[0.0], np.logspace(-2, 3, 21)])  # 0.01 ... 1000, ~4/decade
ALPHAS = np.concatenate([np.logspace(-2, 3, 21),  np.logspace(3,6,7)]) # 0.01 ... 1000, ~4/decade

# Measured: one merge-rna native lambda_only fit with default (uncapped) tolerances took ~60 min
# (562 L-BFGS-B iterations x ~6.4s/iteration, on this 148-nt sequence). MAX_ITER bounds that cost;
# 400 iterations reaches total_loss=440.7617 vs. 440.2450 at full (562-iteration) convergence on
# this system's alpha=0 point -- close, not exact.
MAX_ITER = None
bounds_sc_tuple = (-5.,5.)

N_PARALLEL_NATIVE = 8  # out of 20 cores on this shared machine -- leaves headroom for other users

subfix_out = 'halfmask_test'
OUTPUT_DIR = os.path.join('fits_paper', 'designed_sequence', 'outputs', 'compare_alpha_scan_pop80_' + subfix_out)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

METHOD_LABELS = {
    'merge_rna': 'merge-rna (native)',
    'maxent_chi2': 'maxent (chi-squared)',
    'maxent_binomial': 'maxent (binomial)',
}

RUN_MERGE_RNA = True   # set to True to re-enable the native merge-rna fits
RUN_MAXENT_BINOMIAL = True  # set to True to re-enable the maxent binomial fits
ACTIVE_METHODS = {k: v for k, v in METHOD_LABELS.items() if k=='maxent_chi2' or (RUN_MERGE_RNA and k == 'merge_rna') or (RUN_MAXENT_BINOMIAL and k == 'maxent_binomial')}

len_seq = 148  # length of the designed sequence used in this scan
CUSTOM_MASK = np.concatenate([np.ones(len_seq//2, dtype=bool), np.zeros(len_seq//2, dtype=bool)])
#CUSTOM_MASK = np.concatenate([np.ones(50, dtype=bool), np.zeros(45, dtype=bool), np.ones(len_seq - 50 - 45, dtype=bool)])  # mask directly as array of bools, script expects file or a string with 01
#CUSTOM_MASK = None  # no masking, all positions are free to be fit

COVERAGE_FRACTION = 1.  # fraction of the total coverage to use for the synthetic data (for testing)

def checkpoint_path(method, alpha):
    return os.path.join(CHECKPOINT_DIR, f'{method}_alpha_{alpha:.6g}.pkl')


def load_checkpoint(method, alpha):
    path = checkpoint_path(method, alpha)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_checkpoint(method, alpha, result):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(checkpoint_path(method, alpha), 'wb') as f:
        pickle.dump(result, f)


# =============================================================================
# The maxent optimizers (copied from my_merge_rna/maxent_test_1.ipynb, cells 35/43/64 -- these
# implement a Lagrangian-dual maximum-entropy fit of lambda_sc and are not part of merge_rna).
# =============================================================================
def compute_expansion_parameters(z, T, a, b):
    target_contact_prob = (z - T * a) / (T * b)
    squared_confidence_interval = np.where(z>0,((T - z) / T) * (z / T**2) / b**2, T*(b**2))
    return target_contact_prob, squared_confidence_interval


def bounds_array_with_mask(n_seq, mask, bound_abs):
    """(n_seq, 2) box-constraint array: (-bound_abs, bound_abs) everywhere, pinned to
    (0.0, 0.0) at positions mask excludes (mask=None => no pinning)."""
    boundaries = bound_abs * np.ones((n_seq, 2))
    boundaries[:, 0] *= -1
    if mask is not None:
        boundaries[~mask] = (0.0, 0.0)
    return boundaries


def maxent(seq,bpp_ref,*, sigma_squared, penalty=None, boundaries=None, natural_boundaries=False, initial_guess=None, rescale=True,version=0,T=None,deriv_correct=False,alpha=0.0,tol=None):
    # seq: sequence
    # bpp_ref: array with base pairing probabilities to be enforced
    # sigma_squared error estimate, enters in derivative and regularization
    
    # boundaries: apply boundaries on lambda.
    #   a number (e.g., 2), means all lambdas are in (-2,+2)
    #   alternatively, pass a (N,2) array with minumum (:,0) and maximum (:,1)
    
    # alpha: KL regularization
    from scipy.optimize import minimize

    kb=1.98717/1000 # grepped from vienna source code
    kBT = ((RNA.cvar.temperature + 273.15) * kb)
    beta = 1.0/kBT

    if T is not None:
        save_T=RNA.cvar.temperature
        RNA.cvar.temperature=T
    
    fc=RNA.fold_compound(seq)
    if rescale:
        fc.mfe()
        fc.exp_params_rescale(1.*fc.mfe()[1])

    if natural_boundaries:
        # natural boundaries are the ones that guarantee that the pairing probability is in [0,1]
        # for a given a_phys, b_phys, and mutation counts
        natural_min_lambda=-bpp_ref/(alpha*sigma_squared)
        natural_max_lambda=(1-bpp_ref)/(alpha*sigma_squared)
        boundaries = np.array([(natural_min_lambda[i], natural_max_lambda[i]) for i in range(len(natural_min_lambda))])
        boundaries = boundaries/(beta*alpha)  # make adimensional for the optimizer
    else:
        if isinstance(boundaries,float) or isinstance(boundaries,int):
            boundaries=boundaries*np.ones((len(seq),2))
            boundaries[:,0]*=-1
    
    def Gamma(lambdas):
        
        if deriv_correct:
            lambdas_rounded=np.round(lambdas,2)
            dlambdas=lambdas-lambdas_rounded
            lambdas=lambdas_rounded
        else:
            dlambdas=lambdas*0.0
        
        if penalty is not None:
            for i in range(len(seq)):
                for j in range(i+1,len(seq)):
                    fc.sc_add_bp(i+1,j+1,2*penalty) # penalty is already in kcal/mol
        
        shift=0.0
        for i in range(len(lambdas)):
            m=lambdas[i]
            if m==0:
                continue
            elif version==1 or (version==0 and m<0):
                fc.sc_add_up(i+1,-m) # vienna takes positions starting from 1
                shift+=m
            elif version==2 or (version==0 and m>0):
                for j in range(len(lambdas)):
                    if j>i:
                        fc.sc_add_bp(i+1,j+1,m)
                    if j<i:
                        fc.sc_add_bp(j+1,i+1,m)
        F=fc.pf()[1]+shift
        gamma=-F+np.dot(lambdas,bpp_ref)
        bpp = np.array(fc.bpp())[1:,1:]
        bpp+=bpp.T
        bpp=np.sum(bpp,axis=0)
        gradients=bpp_ref-bpp
        # correction on gamma - but it does not correct the gradients...
        # this means the minimization might fail
        
        gamma+=np.dot(gradients,dlambdas)
        
        # L2 regularization
        gamma+= 0.5*alpha*beta*np.sum(sigma_squared*(lambdas+dlambdas)**2)
        gradients += alpha*beta*sigma_squared*(lambdas+dlambdas)
        
        fc.sc_init()
        
        return gamma,gradients
    
    if initial_guess is None:
        initial_guess=np.zeros(len(seq))
    else:
        initial_guess=np.array(initial_guess)
    
    if tol is None:
        res=minimize(Gamma,initial_guess,jac=True,method="L-BFGS-B",bounds=boundaries)
    else:
        res=minimize(Gamma,initial_guess,jac=True,method="L-BFGS-B",bounds=boundaries,tol=tol)
        
        
    if T is not None:
        RNA.cvar.temperature=save_T
        
    return res,bpp_ref-res.jac+alpha*beta*sigma_squared*res.x


def maxent_binomial(seq, mutation_counts, trials_counts, a_phys, b_phys,*, penalty=None, 
                    boundaries=None, formal_bounds=True, initial_guess=None, mask=None,
                    rescale=True,version=0,T=None,deriv_correction=False,alpha=0.0,tol=None):
    # seq: sequence
    # mutation_counts: array with number of mutations counted per site
    # trials_counts: array with number of attempted mutations per site

    # a_phys, b_phys: arrays of physical parameters for the mutation probability

    # boundaries: apply boundaries on lambda.
    #   a number (e.g., 2), means all lambdas are in (-2,+2)
    #   alternatively, pass a (N,2) array with minumum (:,0) and maximum (:,1)
    
    # alpha: KL regularization
    from scipy.optimize import minimize

    if T is not None:
        save_T=RNA.cvar.temperature
        RNA.cvar.temperature=T

    kb=1.98717/1000 # grepped from vienna source code
    kBT = ((RNA.cvar.temperature + 273.15) * kb)
    beta = 1.0/kBT
    #beta = 1
    
    fc=RNA.fold_compound(seq)
    if rescale:
        fc.mfe()
        fc.exp_params_rescale(1.*fc.mfe()[1])

    if mask is None:
        mask=np.ones(len(seq),dtype=bool)

    positions_with_zero_mut=np.zeros(len(seq),dtype=bool)
    positions_with_nonzero_mut=np.zeros(len(seq),dtype=bool)

    for i in range(len(seq)):
        if mask[i]:
            if mutation_counts[i]==0:
                positions_with_zero_mut[i]=True
            elif mutation_counts[i]>0:
                positions_with_nonzero_mut[i]=True

    if isinstance(boundaries,float) or isinstance(boundaries,int):
        boundaries=boundaries*np.ones((len(seq),2))
        boundaries[:,0]*=-1
    elif boundaries is None:
        boundaries = np.array([(None, None)]*len(seq))

    # when a mutation count is zero, the likelihood has a behavior that requires a lower bound on the sc
    if formal_bounds:
        for i in range(len(seq)):
            if positions_with_zero_mut[i]:
                # if trials_counts[i]*b_phys[i]/alpha <boundaries[i,1] or boundaries[i,1] is None:  # this has a problem if one imposes a lower bound that is already higher than the natural bound
                #     boundaries[i,0]=trials_counts[i]*b_phys[i]/alpha
                # else:
                #     boundaries[i,0]=trials_counts[i]*b_phys[i]/alpha # this has to be set always otherwise you get mut_prob<0 easily for small lambdas
                #     boundaries[i,1]=trials_counts[i]*b_phys[i]/alpha
                #     # alternatively one could set both bounds to the "natural" one, but it could be too large
                boundaries[i,0]=trials_counts[i]*b_phys[i]/(alpha*(1-a_phys[i]))
                boundaries[i,1]=trials_counts[i]*b_phys[i]/(alpha*(1-a_phys[i]-b_phys[i])) 

    frac_phys = a_phys/b_phys
    inv_b_phys = 1.0/b_phys

    def inverse_loss_derivative(lams):
        # lams must be adimensional
        # when lams is 0 (or alpha is 0), the results should be -frac_phys + inv_b_phys*mutation_counts/trials_counts
        lams_prime = lams*alpha/b_phys
        zero_vec = np.zeros_like(lams)

        # when mutation_counts=0, the result should be -frac_phys + inv_b_phys - T/(lambda*alpha)
        S = np.sqrt((trials_counts - lams_prime)**2 + 4*mutation_counts*lams_prime, out=zero_vec, where=(mutation_counts>0))

        result = np.where(positions_with_zero_mut, -frac_phys + inv_b_phys-trials_counts/(alpha*lams), zero_vec)
        result =  np.where(positions_with_nonzero_mut, -frac_phys + inv_b_phys*2*mutation_counts / (S - (lams_prime - trials_counts)), result)

        return result

    def loss_binomial(bpp_1):
        mut_prob = a_phys+b_phys*bpp_1
        log_mut = np.zeros_like(mut_prob)
        log_1m_mut = np.zeros_like(mut_prob)
        np.log(mut_prob, out=log_mut, where=(positions_with_zero_mut))
        np.log(1-mut_prob, out=log_1m_mut, where=(positions_with_nonzero_mut)) # one should also check if z != T
        return -mutation_counts*log_mut-(trials_counts-mutation_counts)*log_1m_mut
    
    def deriv_loss_binomial(bpp_1):
        mut_prob = a_phys+b_phys*bpp_1
        return -b_phys*mutation_counts/mut_prob+b_phys*(trials_counts-mutation_counts)/(1-mut_prob)

    def Gamma(lambdas):
        
        if deriv_correction:
            lambdas_rounded=np.round(lambdas,2)
            dlambdas=lambdas-lambdas_rounded
            lambdas=lambdas_rounded
        else:
            dlambdas=lambdas*0.0
        
        if penalty is not None:
            for i in range(len(seq)):
                for j in range(i+1,len(seq)):
                    fc.sc_add_bp(i+1,j+1,2*penalty) # penalty is already in kcal/mol
        
        shift=0.0
        for i in range(len(lambdas)):
            m=lambdas[i]
            if m==0:
                continue
            elif version==1 or (version==0 and m<0):
                fc.sc_add_up(i+1,-m) # vienna takes positions starting from 1
                shift+=m
            elif version==2 or (version==0 and m>0):
                for j in range(len(lambdas)):
                    if j>i:
                        fc.sc_add_bp(i+1,j+1,m)
                    if j<i:
                        fc.sc_add_bp(j+1,i+1,m)
        F=fc.pf()[1] + shift

        bpp_lambda = inverse_loss_derivative(beta*lambdas) # need to make lambdas adimensional
        gamma=beta*(-F+np.dot(lambdas,bpp_lambda))
        
        bpp = np.array(fc.bpp())[1:,1:]
        bpp+=bpp.T
        bpp=np.sum(bpp,axis=0)
        gradients = beta*(-bpp + bpp_lambda)
        # correction on gamma - but it does not correct the gradients...
        # this means the minimization might fail
        
        gamma+=beta*np.dot(gradients,dlambdas)
        
        # alpha regularization

        if alpha > 0:
            gamma+= -(1.0/alpha)*np.sum(loss_binomial(bpp_lambda))
        # otherwise this becomes a constant diverging term = -1/alpha * min(binmial_loss)
        
        # alpha enters in the gradients by default so no need to add a term for alpha!=0
        # but if it is equal to zero the gradients are just chasing the pairing prob that gives the exp mutation for given a_phys, b_phys
        
        fc.sc_init()
        
        return gamma,gradients
    
    if initial_guess is None:
        initial_guess=np.zeros(len(seq))
        # the minimize handles init guess out of boundaries
    else:
        initial_guess=np.array(initial_guess)
    
    if tol is None:
        res=minimize(Gamma,initial_guess,jac=True,method="L-BFGS-B",bounds=boundaries)
    else:
        res=minimize(Gamma,initial_guess,jac=True,method="L-BFGS-B",bounds=boundaries,tol=tol)
        
        
    if T is not None:
        RNA.cvar.temperature=save_T
    
    # return the result of the optimization and the average pairing probs corrected by a non zero gradient
    return res,inverse_loss_derivative(beta*res.x)-res.jac


# =============================================================================
# Shared setup (physical params, synthetic data, scoring)
# =============================================================================
def load_reference_physical_params(params_1d_path):
    """Bootstrap mu_r/p_b/m0/m1/p_bind_dict from an 8-value physical-params file, via a
    throwaway 0mM Redmond reference system (this bootstrap doesn't depend on which
    experiment will actually be fit/scored)."""
    params_1D_ref = np.loadtxt(params_1d_path)[:8]

    exp_ref_0mM = next(
        (Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt if Experiment(p).conc_mM == 0),
        None)
    if exp_ref_0mM is None:
        raise RuntimeError("Could not find a 0 mM Redmond experiment to build reference params_dict")

    multi_boot = MultiSystemsFit([exp_ref_0mM], validation_exps=None, infer_1D_sc=False, skip_output_setup=True)
    params_dict_ref = multi_boot.pack_params(params_1D_ref, multi_boot.systems[0])

    mu_r, p_b, m0, m1 = (params_dict_ref['mu_r'], params_dict_ref['p_b'],
                         params_dict_ref['m0'], params_dict_ref['m1'])
    return params_1D_ref, params_dict_ref, mu_r, p_b, m0, m1, params_dict_ref['p_bind']


def save_experiment(exp, path_prefix):
    """Save exp.df + reload metadata so Experiment.from_csv(f'{path_prefix}_df.csv', **json.load(...))
    reconstructs it later, from any script."""
    exp.df.to_csv(f'{path_prefix}_df.csv', index=False)
    meta = dict(system_name=exp.system_name, conc_mM=exp.conc_mM, temp_C=exp.temp_C,
                reagent=exp.reagent, rep_number=exp.rep_number)
    with open(f'{path_prefix}_meta.json', 'w') as f:
        json.dump(meta, f)


def save_mask(mask, path):
    """'0'/'1' string -- the exact format ExperimentFit._parse_custom_mask accepts as custom_mask=path."""
    with open(path, 'w') as f:
        f.write(''.join('1' if b else '0' for b in mask))


def combine_experiments(exp_a, exp_b, *, system_name='combined'):
    """Pool mut_count/total_count position-wise from two experiments of the SAME sequence into
    one merged Experiment -- used when there's no position mask to split trained/held-out on."""
    if exp_a.seq != exp_b.seq:
        raise ValueError(f"Cannot combine experiments with different sequences "
                          f"({exp_a.system_name!r} vs {exp_b.system_name!r})")
    if exp_a.conc_mM != exp_b.conc_mM:
        raise ValueError(f"Cannot combine experiments at different concentrations "
                          f"({exp_a.conc_mM!r} mM vs {exp_b.conc_mM!r} mM)")
    df = exp_a.df.copy()
    df['mut_count'] = exp_a.df['mut_count'].values + exp_b.df['mut_count'].values
    df['total_count'] = exp_a.df['total_count'].values + exp_b.df['total_count'].values
    df['wt_count'] = df['total_count'] - df['mut_count']
    df['mut_rate'] = df['mut_count'] / df['total_count']
    return Experiment.from_dataframe(df, seq=exp_a.seq, system_name=system_name, conc_mM=exp_a.conc_mM)


def build_scoring_context(experiments, mu_r, p_b, m0, m1, p_bind_dict, mask=None):
    """Scores/fits-shared setup for a given experiment list/mask -- generalizes what used to be
    build_shared_context()'s tail so it can also be reused to score a NEW experiment, not just
    the one being fit."""
    multi_score = MultiSystemsFit(experiments=experiments, validation_exps=None, infer_1D_sc=True,
                                   skip_output_setup=True, custom_mask=mask)
    exp_fit = multi_score.systems[0].exp_fits_train[0]

    mu_j = mu_r + exp_fit.kBT * np.log((exp_fit.conc_mM + .1) / 1000) if exp_fit.conc_mM is not None else mu_r
    penalty = exp_fit.compute_penalty_m(mu_j, p_b)

    p_cb_unpaired = exp_fit.compute_p_cb(mu_j, p_b, p_bind_dict, np.zeros(exp_fit.N_seq))
    p_cb_paired = exp_fit.compute_p_cb(mu_j, p_b, p_bind_dict, np.ones(exp_fit.N_seq))
    a_i = exp_fit.compute_mutation_profile(m0, m1, exp_fit.eps_b, p_cb_unpaired)
    b_i = exp_fit.compute_mutation_profile(m0, m1, exp_fit.eps_b, p_cb_paired) - a_i

    # Sanity check: a_i, b_i must exactly reproduce merge-rna's own affine mutation-rate model.
    rng = np.random.default_rng(0)
    lambda_test = rng.uniform(-1, 1, exp_fit.N_seq)
    bpp_test = exp_fit.get_ps(penalty, lambda_test)
    mut_rate_direct, _ = exp_fit.mut_rate_and_its_grad(
        mu_r=mu_r, p_b=p_b, p_bind=p_bind_dict, m0=m0, m1=m1,
        lambda_sc=lambda_test, compute_gradient=False)
    mut_rate_affine = a_i + b_i * bpp_test
    assert np.allclose(mut_rate_direct, mut_rate_affine, atol=1e-10), \
        "a_i, b_i do not exactly reproduce merge-rna's affine mutation-rate model"

    return dict(mu_r=mu_r, p_b=p_b, m0=m0, m1=m1, p_bind_dict=p_bind_dict,
                multi_score=multi_score, exp_fit=exp_fit, mu_j=mu_j, penalty=penalty, a_i=a_i, b_i=b_i)


def build_shared_context():
    params_1D_ref, params_dict_ref, mu_r, p_b, m0, m1, p_bind_dict = load_reference_physical_params(PARAMS1D_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    phys_only_guess_path = os.path.join(OUTPUT_DIR, 'phys_params_only.txt')
    np.savetxt(phys_only_guess_path, params_1D_ref)

    np.random.seed(SEED)

    if COVERAGE_FRACTION < 1.0:
        fit_coverage = int(COVERAGE_FRACTION * COVERAGE)
        valid_coverage = COVERAGE - fit_coverage

        exp = create_exp_synthetic_comb(pop1=POP1, params_dict=params_dict_ref, noise=True, coverage=fit_coverage)

        np.random.seed(SEED_VAL)
        valid_exp = create_exp_synthetic_comb(pop1=POP1, params_dict=params_dict_ref, noise=True, coverage=valid_coverage)

    else:
        exp = create_exp_synthetic_comb(pop1=POP1, params_dict=params_dict_ref, noise=True, coverage=COVERAGE)
        valid_exp = None

    save_experiment(exp, os.path.join(OUTPUT_DIR, 'exp'))
    if valid_exp is not None:
        save_experiment(valid_exp, os.path.join(OUTPUT_DIR, 'valid_exp'))
    if CUSTOM_MASK is not None:
        save_mask(CUSTOM_MASK, os.path.join(OUTPUT_DIR, 'mask.txt'))

    scoring_ctx = build_scoring_context([exp], mu_r, p_b, m0, m1, p_bind_dict, mask=CUSTOM_MASK)

    return dict(params_dict_ref=params_dict_ref, phys_only_guess_path=phys_only_guess_path,
                exp=exp, valid_exp=valid_exp, **scoring_ctx)


def opt_info_from_result(res, param_indices=None):
    if res is None:
        return dict(success=False, status=None,
                    message='fit() returned None (often a caught KeyboardInterrupt)',
                    nit=None, grad_norm=None)
    jac = getattr(res, 'jac', None)
    if jac is not None and param_indices is not None:
        jac = np.asarray(jac)[param_indices]
    grad_norm = float(np.linalg.norm(jac)) if jac is not None else None
    return dict(success=bool(getattr(res, 'success', False)),
                status=getattr(res, 'status', None),
                message=str(getattr(res, 'message', '')),
                nit=int(getattr(res, 'nit', -1)) if getattr(res, 'nit', None) is not None else None,
                grad_norm=grad_norm)


def score(ctx, lambda_sc, alpha, opt_info):
    exp_fit, mu_r, p_b, m0, m1, p_bind_dict, penalty = (
        ctx['exp_fit'], ctx['mu_r'], ctx['p_b'], ctx['m0'], ctx['m1'], ctx['p_bind_dict'], ctx['penalty'])
    mut_rate_model, _ = exp_fit.mut_rate_and_its_grad(
        mu_r=mu_r, p_b=p_b, p_bind=p_bind_dict, m0=m0, m1=m1,
        lambda_sc=lambda_sc, compute_gradient=False)
    nll, _ = exp_fit.loss_and_grad(mut_rate_model, None)
    kl, _ = exp_fit.kl_and_grad(lambda_sc, compute_gradient=False)
    bpp = exp_fit.get_ps(penalty, lambda_sc)
    result = dict(
        lambda_sc=np.asarray(lambda_sc, dtype=float),
        bpp=bpp,
        mut_rate_model=mut_rate_model,
        log_likelihood=-nll,
        kl_divergence=kl,
        total_loss=nll + alpha * kl,
    )
    result.update(opt_info)
    return result


def rescore_checkpoint(ctx, res, alpha):
    """Recompute a checkpoint's loss-derived fields (log_likelihood, kl_divergence,
    total_loss, bpp, mut_rate_model) from its stored lambda_sc under the CURRENT ctx/masking,
    instead of trusting whatever was cached at original fit time. Optimizer-diagnostic
    fields (success/status/message/nit/grad_norm) are carried through unchanged since they
    describe the actual optimization run, not the scoring."""
    if res['lambda_sc'] is None:
        return res
    opt_info = {k: res[k] for k in ('success', 'status', 'message', 'nit', 'grad_norm')}
    return score(ctx, res['lambda_sc'], alpha, opt_info)


def log_likelihood_on_mask(exp_fit, mut_rate_model, mask):
    """NLL under exp_fit's existing binomial loss, but restricted to `mask` instead of
    exp_fit.position_mask. Reuses loss_and_grad's exact formula by temporarily swapping
    the mask rather than re-deriving the log-likelihood."""
    original_mask = exp_fit.position_mask
    exp_fit.position_mask = mask
    try:
        nll, _ = exp_fit.loss_and_grad(mut_rate_model, None)
    finally:
        exp_fit.position_mask = original_mask
    return -nll


def mut_rate_model_for_lambda(ctx, lambda_sc):
    exp_fit, mu_r, p_b, m0, m1, p_bind_dict = (
        ctx['exp_fit'], ctx['mu_r'], ctx['p_b'], ctx['m0'], ctx['m1'], ctx['p_bind_dict'])
    mut_rate_model, _ = exp_fit.mut_rate_and_its_grad(
        mu_r=mu_r, p_b=p_b, p_bind=p_bind_dict, m0=m0, m1=m1,
        lambda_sc=lambda_sc, compute_gradient=False)
    return mut_rate_model


# =============================================================================
# Native leg worker (runs in a forked subprocess)
# =============================================================================
def run_native_fit(alpha, exp, phys_only_guess_path):
    fit_cls = MultiSystemsFitMaskedLambdaBounds if CUSTOM_MASK is not None else MultiSystemsFit
    multi_native = fit_cls(
        experiments=[exp],
        validation_exps=None,
        output_suffix=f'alpha_{alpha:.4g}',
        root_dir=OUTPUT_DIR,
        infer_1D_sc=True,
        fit_mode='lambda_only',
        guess=phys_only_guess_path,
        reg_weight=alpha,
        max_iter=MAX_ITER,
        do_plots=False,
        print_to_std_out=False,
        overwrite=True,
        bound_soft_constraints=bounds_sc_tuple,
        custom_mask=CUSTOM_MASK
    )
    result = multi_native.fit()
    lambda_idx = multi_native.systems[0].dict_with_params_pos['lambda_sc']
    info = opt_info_from_result(result, param_indices=lambda_idx)
    lambda_sc = result.x[lambda_idx] if result is not None else None
    return alpha, lambda_sc, info


def _native_worker_entry(args):
    alpha, exp, phys_only_guess_path = args
    return run_native_fit(alpha, exp, phys_only_guess_path)


# =============================================================================
# Main scan (resumable: skips any (method, alpha) that already has a checkpoint)
# =============================================================================
def run_scan():
    print(f"[{time.strftime('%H:%M:%S')}] Building shared context (physical params, synthetic data)...", flush=True)
    ctx = build_shared_context()
    exp = ctx['exp']
    print(f"[{time.strftime('%H:%M:%S')}] Sanity check passed. N_seq={exp.N_seq}, "
          f"total mut_count={exp.df['mut_count'].sum()}", flush=True)
 
    if RUN_MERGE_RNA:
        # --- merge-rna native, in parallel across whatever alphas aren't already checkpointed ---
        todo_native = [a for a in ALPHAS if load_checkpoint('merge_rna', a) is None]
        print(f"[{time.strftime('%H:%M:%S')}] merge-rna native: {len(ALPHAS) - len(todo_native)}/{len(ALPHAS)} "
              f"already done, {len(todo_native)} to run (max {N_PARALLEL_NATIVE} in parallel)", flush=True)

        if todo_native:
            t0 = time.time()
            mp_ctx = multiprocessing.get_context('fork')
            with ProcessPoolExecutor(max_workers=N_PARALLEL_NATIVE, mp_context=mp_ctx) as pool:
                futures = {pool.submit(run_native_fit, a, exp, ctx['phys_only_guess_path']): a for a in todo_native}
                for fut in as_completed(futures):
                    alpha = futures[fut]
                    _, lambda_sc, info = fut.result()
                    if lambda_sc is None:
                        print(f"  WARNING: merge-rna native fit returned None at alpha={alpha:.4g}: {info['message']}", flush=True)
                        result = dict(lambda_sc=None, bpp=None, mut_rate_model=None,
                                      log_likelihood=None, kl_divergence=None, total_loss=None, **info)
                    else:
                        result = score(ctx, lambda_sc, alpha, info)
                    save_checkpoint('merge_rna', alpha, result)
                    print(f"  [{time.strftime('%H:%M:%S')}] native alpha={alpha:.4g} done: "
                          f"success={info['success']} nit={info['nit']} grad_norm={info['grad_norm']}", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] merge-rna native leg: {time.time() - t0:.1f}s "
                  f"for {len(todo_native)} alphas ({N_PARALLEL_NATIVE} parallel workers)", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] merge-rna native: nothing to do, all checkpointed.", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Skipping merge-rna native leg (RUN_MERGE_RNA=False).", flush=True)

    # --- maxent (chi-squared) and maxent (binomial), serial, resumable per-alpha ---
    mutation_counts = exp.df['mut_count'].values.astype(float)
    trials_counts = exp.df['total_count'].values.astype(float)
    a_i, b_i, penalty = ctx['a_i'], ctx['b_i'], ctx['penalty']
    maxent_boundaries = bounds_array_with_mask(exp.N_seq, CUSTOM_MASK, bounds_sc_tuple[1])

    if not RUN_MAXENT_BINOMIAL:
        print(f"[{time.strftime('%H:%M:%S')}] Skipping maxent (binomial) leg (RUN_MAXENT_BINOMIAL=False).", flush=True)

    t0 = time.time()
    n_done = 0
    for alpha in ALPHAS:
        if load_checkpoint('maxent_chi2', alpha) is None:
            bpp_target, sigma_squared = compute_expansion_parameters(mutation_counts, trials_counts, a_i, b_i)
            res_chi2 = maxent(exp.seq, bpp_target, sigma_squared=sigma_squared, penalty=penalty,
                              boundaries=maxent_boundaries, alpha=alpha, T=ctx['exp_fit'].temp_C)[0]
            save_checkpoint('maxent_chi2', alpha, score(ctx, res_chi2.x, alpha, opt_info_from_result(res_chi2)))
            n_done += 1

        if RUN_MAXENT_BINOMIAL:
            if load_checkpoint('maxent_binomial', alpha) is None:
                res_binom = maxent_binomial(exp.seq, mutation_counts=mutation_counts, trials_counts=trials_counts,
                                                a_phys=a_i, b_phys=b_i, penalty=penalty, boundaries=maxent_boundaries,
                                                alpha=alpha, T=ctx['exp_fit'].temp_C)[0]
                save_checkpoint('maxent_binomial', alpha, score(ctx, res_binom.x, alpha, opt_info_from_result(res_binom)))
                n_done += 1

        print(f"  [{time.strftime('%H:%M:%S')}] maxent alpha={alpha:.4g} done", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] maxent legs: {time.time() - t0:.1f}s ({n_done} new checkpoints)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Scan complete.", flush=True)
    return ctx


def count_lambda_at_boundary(lambda_sc, mask, bound_abs, atol=1e-6):
    if lambda_sc is None:
        return None
    active = lambda_sc[mask] if mask is not None else lambda_sc
    return int(np.sum(np.isclose(np.abs(active), bound_abs, atol=atol)))


def assemble_results(ctx=None):
    '''Reads whatever checkpoints exist (complete or partial scan) and writes the summary
    CSV/status CSV/pickle/plot. Safe to call at any time, including while the scan is still running.'''
    fitted_results = {}
    for method in ACTIVE_METHODS:
        for alpha in ALPHAS:
            res = load_checkpoint(method, alpha)
            if res is not None:
                fitted_results[(method, alpha)] = res

    if not fitted_results:
        print("No checkpoints found yet.")
        return

    if ctx is None:
        ctx = build_shared_context()

    # Re-score every checkpoint against the current ctx/masking rather than trusting
    # whatever log_likelihood/kl_divergence/total_loss/bpp/mut_rate_model were cached at
    # original fit time -- otherwise a scoring/masking fix never takes effect for
    # already-checkpointed (method, alpha) pairs under --assemble-only.
    fitted_results = {(method, alpha): rescore_checkpoint(ctx, res, alpha)
                       for (method, alpha), res in fitted_results.items()}

    status_df = pd.DataFrame([
        dict(method=method, alpha=alpha, success=res['success'], status=res['status'],
             message=res['message'], nit=res['nit'], grad_norm=res['grad_norm'],
             n_lambda_at_boundary=count_lambda_at_boundary(res['lambda_sc'], CUSTOM_MASK, bounds_sc_tuple[1]))
        for (method, alpha), res in fitted_results.items()
    ]).sort_values(['method', 'alpha'])

    summary_df = pd.DataFrame([
        dict(method=method, alpha=alpha,
             log_likelihood=res['log_likelihood'],
             kl_divergence=res['kl_divergence'],
             total_loss=res['total_loss'])
        for (method, alpha), res in fitted_results.items()
        if res['log_likelihood'] is not None
    ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'scan_results.csv'), index=False)
    status_df.to_csv(os.path.join(OUTPUT_DIR, 'status_results.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'fitted_results.pkl'), 'wb') as f:
        pickle.dump(fitted_results, f)

    n_expected = len(ALPHAS) * len(ACTIVE_METHODS)
    n_have = len(fitted_results)
    n_failed = int((~status_df['success']).sum())
    print(f"{n_have}/{n_expected} (method, alpha) checkpoints present ({n_failed} did not report success=True)")

    if not summary_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['log_likelihood', 'kl_divergence', 'total_loss']
        titles = ['Log-likelihood', 'KL divergence', 'Total loss (NLL + alpha * KL)']
        for ax, metric, title in zip(axes, metrics, titles):
            for method, label in ACTIVE_METHODS.items():
                sub = summary_df[summary_df['method'] == method].sort_values('alpha')
                if not sub.empty:
                    ax.plot(sub['alpha'], sub[metric], 'o-', label=label)
            ax.set_xscale('symlog', linthresh=1e-2)
            ax.set_xlabel('alpha')
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'scan_plot.png')
        plt.savefig(plot_path, dpi=120)
        print(f"Saved plot to {plot_path}")

    print(f"Saved summary to {os.path.join(OUTPUT_DIR, 'scan_results.csv')}")
    print(f"Saved status to {os.path.join(OUTPUT_DIR, 'status_results.csv')}")
    print(f"Saved fitted results to {os.path.join(OUTPUT_DIR, 'fitted_results.pkl')}")

    # --- Holdout analysis: log-likelihood on masked-out sites / all sites, vs. the
    # lambda_sc=0 baseline evaluated on the same site subsets ---
    exp_fit = ctx['exp_fit']
    full_mask = np.ones(exp_fit.N_seq, dtype=bool)
    held_out_mask = ~CUSTOM_MASK if CUSTOM_MASK is not None else None

    mut_rate_zero = mut_rate_model_for_lambda(ctx, np.zeros(exp_fit.N_seq))
    ll_zero_all = log_likelihood_on_mask(exp_fit, mut_rate_zero, full_mask)
    ll_zero_masked_out = (log_likelihood_on_mask(exp_fit, mut_rate_zero, held_out_mask)
                           if held_out_mask is not None else None)

    holdout_rows = []
    for (method, alpha), res in fitted_results.items():
        if res['lambda_sc'] is None:
            continue
        mut_rate_model = res['mut_rate_model']
        row = dict(method=method, alpha=alpha,
                   log_likelihood_all_sites=log_likelihood_on_mask(exp_fit, mut_rate_model, full_mask))
        if held_out_mask is not None:
            row['log_likelihood_masked_out'] = log_likelihood_on_mask(exp_fit, mut_rate_model, held_out_mask)
        holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows).sort_values(['method', 'alpha']) if holdout_rows else pd.DataFrame()

    holdout_df.to_csv(os.path.join(OUTPUT_DIR, 'holdout_results.csv'), index=False)
    print(f"Saved holdout results to {os.path.join(OUTPUT_DIR, 'holdout_results.csv')}")

    if not holdout_df.empty:
        panels = ([('log_likelihood_masked_out', 'Log-likelihood (masked-out sites)', ll_zero_masked_out)]
                   if held_out_mask is not None else []) + \
                 [('log_likelihood_all_sites', 'Log-likelihood (all sites)', ll_zero_all)]
        fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 5), squeeze=False)
        axes = axes[0]
        for ax, (metric, title, baseline) in zip(axes, panels):
            for method, label in ACTIVE_METHODS.items():
                sub = holdout_df[holdout_df['method'] == method].sort_values('alpha')
                if not sub.empty:
                    ax.plot(sub['alpha'], sub[metric]/baseline, 'o-', label=label)
            # ax.axhline(baseline, linestyle='--', color='k', label='lambda_sc = 0 baseline')
            ax.set_xscale('symlog', linthresh=1e-2)
            ax.set_xlabel('alpha')
            ax.set_ylabel(metric)
            ax.set_title('relative ' + title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        holdout_plot_path = os.path.join(OUTPUT_DIR, 'holdout_plot.png')
        plt.savefig(holdout_plot_path, dpi=120)
        print(f"Saved holdout plot to {holdout_plot_path}")

# %%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assemble-only', action='store_true',
                        help='Skip fitting; just read existing checkpoints and (re)write the summary/plot.')
    args = parser.parse_args()

    if args.assemble_only:
        assemble_results()
    else:
        ctx = run_scan()
        assemble_results(ctx)
