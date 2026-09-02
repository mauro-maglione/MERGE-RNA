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

# =============================================================================
# Configuration
# =============================================================================
PARAMS1D_PATH = os.path.join(
    'fits_paper', 'designed_sequence', 'synthetic', 
    'reference_physical_params.txt')

POP1 = 0.8
COVERAGE = 10000
SEED = 42  # fixes the one binomial draw used across the whole scan

ALPHAS = np.concatenate([[0.0], np.logspace(-2, 3, 21)])  # 0.01 ... 1000, ~4/decade

# Measured: one merge-rna native lambda_only fit with default (uncapped) tolerances took ~60 min
# (562 L-BFGS-B iterations x ~6.4s/iteration, on this 148-nt sequence). MAX_ITER bounds that cost;
# 400 iterations reaches total_loss=440.7617 vs. 440.2450 at full (562-iteration) convergence on
# this system's alpha=0 point -- close, not exact.
MAX_ITER = None
bounds_sc_tuple = (-5.,5.)

N_PARALLEL_NATIVE = 8  # out of 20 cores on this shared machine -- leaves headroom for other users

OUTPUT_DIR = os.path.join('fits_paper', 'designed_sequence', 'outputs', 'compare_alpha_scan_pop80_new')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

METHOD_LABELS = {
    'merge_rna': 'merge-rna (native)',
    'maxent_chi2': 'maxent (chi-squared)',
    'maxent_binomial': 'maxent (binomial)',
}


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
    squared_confidence_interval = ((T - z) / T) * (z / T**2) / b**2
    return target_contact_prob, squared_confidence_interval


def maxent(seq,bpp_ref,*, sigma_squared, penalty=None, boundaries=None, initial_guess=None, rescale=True,version=0,T=None,deriv_correct=False,alpha=0.0,tol=None):
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


def maxent_binomial(seq, mutation_counts, trials_counts, a_phys, b_phys,*, penalty=None, boundaries=None, initial_guess=None, rescale=True,version=0,T=None,deriv_correction=False,alpha=0.0,tol=None):
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


    if isinstance(boundaries,float) or isinstance(boundaries,int):
        boundaries=boundaries*np.ones((len(seq),2))
        boundaries[:,0]*=-1
    elif boundaries is None:
        boundaries = np.array([(None, None)]*len(seq))

    frac_phys = a_phys/b_phys
    inv_b_phys = 1.0/b_phys

    def inverse_loss_derivative(lams):
        # lams must be adimensional
        # when lams is 0 (or alpha is 0), the results should be -frac_phys + inv_b_phys*mutation_counts/trials_counts
        lams_prime = lams*alpha/b_phys
        S = np.sqrt((trials_counts - lams_prime)**2 + 4*mutation_counts*lams_prime)
        result =  -frac_phys + inv_b_phys*2*mutation_counts / (S - (lams_prime - trials_counts))
        #print(result-(-frac_phys+inv_b_phys*mutation_counts/trials_counts))
        return result

    def loss_binomial(bpp_1):
        mut_prob = a_phys+b_phys*bpp_1
        return -mutation_counts*np.log(mut_prob)-(trials_counts-mutation_counts)*np.log(1-mut_prob)
    
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
def build_shared_context():
    params_1D_ref = np.loadtxt(PARAMS1D_PATH)[:8]

    exp_ref_0mM = next(
        (Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt if Experiment(p).conc_mM == 0),
        None)
    if exp_ref_0mM is None:
        raise RuntimeError("Could not find a 0 mM Redmond experiment to build reference params_dict")

    multi_boot = MultiSystemsFit([exp_ref_0mM], validation_exps=None, infer_1D_sc=False, skip_output_setup=True)
    params_dict_ref = multi_boot.pack_params(params_1D_ref, multi_boot.systems[0])

    mu_r, p_b, m0, m1 = (params_dict_ref['mu_r'], params_dict_ref['p_b'],
                         params_dict_ref['m0'], params_dict_ref['m1'])
    p_bind_dict = params_dict_ref['p_bind']

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    phys_only_guess_path = os.path.join(OUTPUT_DIR, 'phys_params_only.txt')
    np.savetxt(phys_only_guess_path, params_1D_ref)

    np.random.seed(SEED)
    exp = create_exp_synthetic_comb(pop1=POP1, params_dict=params_dict_ref, noise=True, coverage=COVERAGE)

    multi_score = MultiSystemsFit(experiments=[exp], validation_exps=None, infer_1D_sc=True, skip_output_setup=True)
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

    return dict(params_dict_ref=params_dict_ref, mu_r=mu_r, p_b=p_b, m0=m0, m1=m1,
                p_bind_dict=p_bind_dict, phys_only_guess_path=phys_only_guess_path,
                exp=exp, multi_score=multi_score, exp_fit=exp_fit, mu_j=mu_j,
                penalty=penalty, a_i=a_i, b_i=b_i)


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


# =============================================================================
# Native leg worker (runs in a forked subprocess)
# =============================================================================
def run_native_fit(alpha, exp, phys_only_guess_path):
    multi_native = MultiSystemsFit(
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
        bound_soft_constraints=bounds_sc_tuple
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

    # --- maxent (chi-squared) and maxent (binomial), serial, resumable per-alpha ---
    mutation_counts = exp.df['mut_count'].values.astype(float)
    trials_counts = exp.df['total_count'].values.astype(float)
    a_i, b_i, penalty = ctx['a_i'], ctx['b_i'], ctx['penalty']

    t0 = time.time()
    n_done = 0
    for alpha in ALPHAS:
        if load_checkpoint('maxent_chi2', alpha) is None:
            bpp_target, sigma_squared = compute_expansion_parameters(mutation_counts, trials_counts, a_i, b_i)
            res_chi2 = maxent(exp.seq, bpp_target, sigma_squared=sigma_squared, penalty=penalty,
                              boundaries=bounds_sc_tuple[1], alpha=alpha, T=ctx['exp_fit'].temp_C)[0]
            save_checkpoint('maxent_chi2', alpha, score(ctx, res_chi2.x, alpha, opt_info_from_result(res_chi2)))
            n_done += 1

        if load_checkpoint('maxent_binomial', alpha) is None:
            res_binom = maxent_binomial(exp.seq, mutation_counts=mutation_counts, trials_counts=trials_counts,
                                         a_phys=a_i, b_phys=b_i, penalty=penalty, boundaries=bounds_sc_tuple[1],
                                         alpha=alpha, T=ctx['exp_fit'].temp_C)[0]
            save_checkpoint('maxent_binomial', alpha, score(ctx, res_binom.x, alpha, opt_info_from_result(res_binom)))
            n_done += 1

        print(f"  [{time.strftime('%H:%M:%S')}] maxent alpha={alpha:.4g} done", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] maxent legs: {time.time() - t0:.1f}s ({n_done} new checkpoints)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Scan complete.", flush=True)


def assemble_results():
    '''Reads whatever checkpoints exist (complete or partial scan) and writes the summary
    CSV/status CSV/pickle/plot. Safe to call at any time, including while the scan is still running.'''
    fitted_results = {}
    for method in METHOD_LABELS:
        for alpha in ALPHAS:
            res = load_checkpoint(method, alpha)
            if res is not None:
                fitted_results[(method, alpha)] = res

    if not fitted_results:
        print("No checkpoints found yet.")
        return

    status_df = pd.DataFrame([
        dict(method=method, alpha=alpha, success=res['success'], status=res['status'],
             message=res['message'], nit=res['nit'], grad_norm=res['grad_norm'])
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

    n_expected = len(ALPHAS) * len(METHOD_LABELS)
    n_have = len(fitted_results)
    n_failed = int((~status_df['success']).sum())
    print(f"{n_have}/{n_expected} (method, alpha) checkpoints present ({n_failed} did not report success=True)")

    if not summary_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['log_likelihood', 'kl_divergence', 'total_loss']
        titles = ['Log-likelihood', 'KL divergence', 'Total loss (NLL + alpha * KL)']
        for ax, metric, title in zip(axes, metrics, titles):
            for method, label in METHOD_LABELS.items():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assemble-only', action='store_true',
                        help='Skip fitting; just read existing checkpoints and (re)write the summary/plot.')
    args = parser.parse_args()

    if args.assemble_only:
        assemble_results()
    else:
        run_scan()
        assemble_results()
