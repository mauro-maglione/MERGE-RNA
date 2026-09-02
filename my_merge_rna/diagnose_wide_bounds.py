#!/usr/bin/env python3
import os, sys, time
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
os.chdir(repo_root)
import numpy as np
from run_compare_alpha_scan import (build_shared_context, compute_expansion_parameters,
                                     maxent, maxent_binomial, score, opt_info_from_result,
                                     load_checkpoint)

alpha = 1.0
ctx = build_shared_context()
exp = ctx['exp']
mutation_counts = exp.df['mut_count'].values.astype(float)
trials_counts = exp.df['total_count'].values.astype(float)
a_i, b_i, penalty = ctx['a_i'], ctx['b_i'], ctx['penalty']
bpp_target, sigma_squared = compute_expansion_parameters(mutation_counts, trials_counts, a_i, b_i)

for bound in [1.0, 5.0, 20.0]:
    t0 = time.time()
    res_chi2 = maxent(exp.seq, bpp_target, sigma_squared=sigma_squared, penalty=penalty,
                      boundaries=bound, alpha=alpha, T=ctx['exp_fit'].temp_C)[0]
    sc_chi2 = score(ctx, res_chi2.x, alpha, opt_info_from_result(res_chi2))
    n_bound_chi2 = int(np.sum(np.abs(np.abs(res_chi2.x) - bound) < 1e-6))

    res_bin = maxent_binomial(exp.seq, mutation_counts=mutation_counts, trials_counts=trials_counts,
                               a_phys=a_i, b_phys=b_i, penalty=penalty, boundaries=bound,
                               alpha=alpha, T=ctx['exp_fit'].temp_C)[0]
    sc_bin = score(ctx, res_bin.x, alpha, opt_info_from_result(res_bin))
    n_bound_bin = int(np.sum(np.abs(np.abs(res_bin.x) - bound) < 1e-6))

    print(f"bound=+-{bound:<5g} "
          f"chi2: total_loss={sc_chi2['total_loss']:.4f} nll={-sc_chi2['log_likelihood']:.4f} "
          f"kl={sc_chi2['kl_divergence']:.4f} n_at_bound={n_bound_chi2}/148 success={sc_chi2['success']}  |  "
          f"binom: total_loss={sc_bin['total_loss']:.4f} nll={-sc_bin['log_likelihood']:.4f} "
          f"kl={sc_bin['kl_divergence']:.4f} n_at_bound={n_bound_bin}/148 success={sc_bin['success']}  "
          f"[{time.time()-t0:.1f}s]")

ck = load_checkpoint('merge_rna', alpha)
print(f"\nreference merge_rna (bound +-1): total_loss={ck['total_loss']:.4f} "
      f"nll={-ck['log_likelihood']:.4f} kl={ck['kl_divergence']:.4f}")
