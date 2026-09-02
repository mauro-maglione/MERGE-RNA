#!/usr/bin/env python3
"""
Step 1 diagnostic (see .claude/plans/i-have-my-main-stateful-wilkinson.md):

Finite-difference gradient check of Gamma (the dual objective inside `maxent` /
`maxent_binomial`), evaluated at merge_rna's own converged lambda_sc as well as at each
maxent variant's own converged lambda_sc, for a chosen alpha.

Does NOT modify run_compare_alpha_scan.py. It re-derives the exact same Gamma computation
(copied verbatim from maxent()/maxent_binomial() in that script) as standalone callables so
they can be evaluated at arbitrary lambda vectors -- something the nested closures in the
original functions don't expose.

Usage:
    conda run -n merge-rna-patched-1 --no-capture-output \
        python my_merge_rna/diagnose_gamma_gradient.py --alpha 1.0
"""
import os
import sys
import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_root)

import numpy as np
import RNA

from run_compare_alpha_scan import (
    build_shared_context, load_checkpoint, compute_expansion_parameters,
)


def make_gamma_chi2(seq, bpp_ref, sigma_squared, penalty, alpha, T, rescale=True, version=0):
    save_T = RNA.cvar.temperature
    RNA.cvar.temperature = T
    kb = 1.98717 / 1000
    kBT = (RNA.cvar.temperature + 273.15) * kb
    beta = 1.0 / kBT

    fc = RNA.fold_compound(seq)
    if rescale:
        fc.mfe()
        fc.exp_params_rescale(1. * fc.mfe()[1])
    RNA.cvar.temperature = save_T  # matches maxent(): only held during fc setup on this path

    def Gamma(lambdas):
        RNA.cvar.temperature = T
        if penalty is not None:
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    fc.sc_add_bp(i + 1, j + 1, 2 * penalty)
        shift = 0.0
        for i in range(len(lambdas)):
            m = lambdas[i]
            if m == 0:
                continue
            elif version == 1 or (version == 0 and m < 0):
                fc.sc_add_up(i + 1, -m)
                shift += m
            elif version == 2 or (version == 0 and m > 0):
                for j in range(len(lambdas)):
                    if j > i:
                        fc.sc_add_bp(i + 1, j + 1, m)
                    if j < i:
                        fc.sc_add_bp(j + 1, i + 1, m)
        F = fc.pf()[1] + shift
        gamma = -F + np.dot(lambdas, bpp_ref)
        bpp = np.array(fc.bpp())[1:, 1:]
        bpp += bpp.T
        bpp = np.sum(bpp, axis=0)
        gradients = bpp_ref - bpp
        gamma += 0.5 * alpha * beta * np.sum(sigma_squared * lambdas ** 2)
        gradients = gradients + alpha * beta * sigma_squared * lambdas
        fc.sc_init()
        RNA.cvar.temperature = save_T
        return gamma, gradients

    return Gamma


def make_gamma_binomial(seq, mutation_counts, trials_counts, a_phys, b_phys, penalty, alpha, T,
                         rescale=True, version=0):
    save_T = RNA.cvar.temperature
    RNA.cvar.temperature = T
    kb = 1.98717 / 1000
    kBT = (RNA.cvar.temperature + 273.15) * kb
    beta = 1.0 / kBT

    fc = RNA.fold_compound(seq)
    if rescale:
        fc.mfe()
        fc.exp_params_rescale(1. * fc.mfe()[1])
    RNA.cvar.temperature = save_T

    frac_phys = a_phys / b_phys
    inv_b_phys = 1.0 / b_phys

    def inverse_loss_derivative(lams):
        lams_prime = lams * alpha / b_phys
        S = np.sqrt((trials_counts - lams_prime) ** 2 + 4 * mutation_counts * lams_prime)
        return -frac_phys + inv_b_phys * 2 * mutation_counts / (S - (lams_prime - trials_counts))

    def loss_binomial(bpp_1):
        mut_prob = a_phys + b_phys * bpp_1
        return -mutation_counts * np.log(mut_prob) - (trials_counts - mutation_counts) * np.log(1 - mut_prob)

    def Gamma(lambdas):
        RNA.cvar.temperature = T
        if penalty is not None:
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    fc.sc_add_bp(i + 1, j + 1, 2 * penalty)
        shift = 0.0
        for i in range(len(lambdas)):
            m = lambdas[i]
            if m == 0:
                continue
            elif version == 1 or (version == 0 and m < 0):
                fc.sc_add_up(i + 1, -m)
                shift += m
            elif version == 2 or (version == 0 and m > 0):
                for j in range(len(lambdas)):
                    if j > i:
                        fc.sc_add_bp(i + 1, j + 1, m)
                    if j < i:
                        fc.sc_add_bp(j + 1, i + 1, m)
        F = fc.pf()[1] + shift

        bpp_lambda = inverse_loss_derivative(beta * lambdas)
        gamma = beta * (-F + np.dot(lambdas, bpp_lambda))

        bpp = np.array(fc.bpp())[1:, 1:]
        bpp += bpp.T
        bpp = np.sum(bpp, axis=0)
        gradients = beta * (-bpp + bpp_lambda)

        if alpha > 0:
            gamma += -(1.0 / alpha) * np.sum(loss_binomial(bpp_lambda))

        fc.sc_init()
        RNA.cvar.temperature = save_T
        return gamma, gradients

    return Gamma


def fd_check(Gamma, lam, coords, h=1e-4, label=""):
    print(f"\n--- {label}: FD vs analytic gradient at {len(coords)} coordinates (h={h}) ---")
    _, analytic = Gamma(lam)
    print(f"{'idx':>5} {'lambda_i':>10} {'analytic':>12} {'FD':>12} {'abs_diff':>10}")
    for i in coords:
        lam_p = lam.copy(); lam_p[i] += h
        lam_m = lam.copy(); lam_m[i] -= h
        g_p, _ = Gamma(lam_p)
        g_m, _ = Gamma(lam_m)
        fd = (g_p - g_m) / (2 * h)
        print(f"{i:5d} {lam[i]:10.4f} {analytic[i]:12.6f} {fd:12.6f} {abs(analytic[i]-fd):10.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-coords", type=int, default=6)
    ap.add_argument("--h", type=float, default=1e-4)
    args = ap.parse_args()
    alpha = args.alpha

    print(f"[setup] building shared context...")
    ctx = build_shared_context()
    exp = ctx["exp"]
    seq = exp.seq
    mutation_counts = exp.df["mut_count"].values.astype(float)
    trials_counts = exp.df["total_count"].values.astype(float)
    a_i, b_i, penalty = ctx["a_i"], ctx["b_i"], ctx["penalty"]
    T = ctx["exp_fit"].temp_C

    bpp_target, sigma_squared = compute_expansion_parameters(mutation_counts, trials_counts, a_i, b_i)

    ck_mergerna = load_checkpoint("merge_rna", alpha)
    ck_chi2 = load_checkpoint("maxent_chi2", alpha)
    ck_binom = load_checkpoint("maxent_binomial", alpha)
    assert ck_mergerna is not None and ck_chi2 is not None and ck_binom is not None, \
        f"missing checkpoint(s) for alpha={alpha}"

    lam_mergerna = ck_mergerna["lambda_sc"]
    lam_chi2 = ck_chi2["lambda_sc"]
    lam_binom = ck_binom["lambda_sc"]

    print(f"[info] alpha={alpha}")
    print(f"[info] total_loss   merge_rna={ck_mergerna['total_loss']:.4f}  "
          f"maxent_chi2={ck_chi2['total_loss']:.4f}  maxent_binomial={ck_binom['total_loss']:.4f}")
    print(f"[info] log_lik      merge_rna={ck_mergerna['log_likelihood']:.4f}  "
          f"maxent_chi2={ck_chi2['log_likelihood']:.4f}  maxent_binomial={ck_binom['log_likelihood']:.4f}")
    print(f"[info] kl           merge_rna={ck_mergerna['kl_divergence']:.4f}  "
          f"maxent_chi2={ck_chi2['kl_divergence']:.4f}  maxent_binomial={ck_binom['kl_divergence']:.4f}")

    for name, lam in [("merge_rna", lam_mergerna), ("maxent_chi2", lam_chi2), ("maxent_binomial", lam_binom)]:
        n_bound = int(np.sum(np.abs(np.abs(lam) - 1.0) < 1e-6))
        print(f"[info] {name}: {n_bound}/{len(lam)} lambda_i within 1e-6 of +-1")

    Gamma_chi2 = make_gamma_chi2(seq, bpp_target, sigma_squared, penalty, alpha, T)
    Gamma_binom = make_gamma_binomial(seq, mutation_counts, trials_counts, a_i, b_i, penalty, alpha, T)

    rng = np.random.default_rng(0)

    def pick_coords(lam, n):
        interior = np.where(np.abs(np.abs(lam) - 1.0) >= 1e-6)[0]
        boundary = np.where(np.abs(np.abs(lam) - 1.0) < 1e-6)[0]
        n_each = max(1, n // 2)
        picked = []
        if len(interior) > 0:
            picked += list(rng.choice(interior, size=min(n_each, len(interior)), replace=False))
        if len(boundary) > 0:
            picked += list(rng.choice(boundary, size=min(n_each, len(boundary)), replace=False))
        return picked

    print("\n" + "=" * 70)
    print("GAMMA_CHI2 at lambda_mergerna (the point maxent moved AWAY from)")
    fd_check(Gamma_chi2, lam_mergerna, pick_coords(lam_mergerna, args.n_coords), h=args.h, label="chi2 @ mergerna")

    print("\nGAMMA_CHI2 at its own converged lambda (interior vs boundary split)")
    fd_check(Gamma_chi2, lam_chi2, pick_coords(lam_chi2, args.n_coords), h=args.h, label="chi2 @ own")

    print("\n" + "=" * 70)
    print("GAMMA_BINOMIAL at lambda_mergerna (the point maxent_binomial moved AWAY from)")
    fd_check(Gamma_binom, lam_mergerna, pick_coords(lam_mergerna, args.n_coords), h=args.h, label="binom @ mergerna")

    print("\nGAMMA_BINOMIAL at its own converged lambda (interior vs boundary split)")
    fd_check(Gamma_binom, lam_binom, pick_coords(lam_binom, args.n_coords), h=args.h, label="binom @ own")


if __name__ == "__main__":
    main()
