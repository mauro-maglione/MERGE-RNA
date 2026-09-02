#!/usr/bin/env python3
import os, sys, pickle
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
os.chdir(repo_root)
import numpy as np
from run_compare_alpha_scan import ALPHAS, load_checkpoint, METHOD_LABELS

print(f"{'alpha':>10} {'merge_rna':>12} {'maxent_chi2':>12} {'maxent_binom':>13}   (n_lambda@|lam|~1 / 148)")
for a in ALPHAS:
    counts = {}
    for m in METHOD_LABELS:
        ck = load_checkpoint(m, a)
        if ck is None or ck.get('lambda_sc') is None:
            counts[m] = None
            continue
        lam = ck['lambda_sc']
        counts[m] = int(np.sum(np.abs(np.abs(lam) - 1.0) < 1e-6))
    def fmt(x): return f"{x:12d}" if x is not None else f"{'--':>12}"
    print(f"{a:10.4g} {fmt(counts['merge_rna'])} {fmt(counts['maxent_chi2'])} {fmt(counts['maxent_binomial'])}")
