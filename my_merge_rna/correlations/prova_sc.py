#  %%

import os
import pathlib
import sys

_repo_root = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('RNA_STRUCT_HOME', str(_repo_root))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])
os.chdir(str(_repo_root))

import numpy as np
import pandas as pd

import bussilab.rna2d as bussi
# %%

# import the physical params and sc from Giuses fit
phase1_params = os.path.join(os.environ['RNA_STRUCT_HOME'], 'fits_paper', 'designed_sequence',
                                   'experimental', 'strand_disp_turner.txt')

_designed_seq_dir = os.path.join(os.environ['RNA_STRUCT_HOME'], 'data', 'designed_sequence')
newseq_wt_fasta_path = os.path.join(_designed_seq_dir, 'Newseq_WT.fasta')

with open(newseq_wt_fasta_path, "r") as file:
    for i, line in enumerate(file):
        if i == 1:
            seq = line.strip()

# %%

delta_kcalpermol = 5
lambdas = np.loadtxt(phase1_params)[8:]
# obtain the suboptimal structures from the ensemble with DMS correction and sc correction
mol = bussi.Molecule(seq, lambdas1d=lambdas) 
mol.suboptimal_coverage(delta_kcalpermol)

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

    coverage_of_part_func = mol.suboptimal_coverage(delta)
    structures = mol.suboptimal_structures(delta)

    # in principle one can sample at a diff temp then the one used for weights
    states, logweights = bussi.suboptimal_to_numpy(suboptimal = structures, temperature=temperature_C+273.15) 
    return coverage_of_part_func, states, logweights

penalty = 0
coverage_of_part_func, states, logweights = structures_sampling(seq, penalty, delta_kcalpermol, lambda_sc= lambdas)
# %%
