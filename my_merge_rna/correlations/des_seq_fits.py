#!/usr/bin/env python3
"""
Paper fit: newseq WT experimental bistable data.

Two-phase fit:
  Phase 1: physical parameters only (no interpolated pairing probs).
  Phase 2: fix physical params, fit lambda_sc with use_interpolated_ps=True.

Usage:
    cd $RNA_STRUCT_HOME
    python my_merge_rna/correlations/des_seq_fits.py --phase both
    python my_merge_rna/correlations/des_seq_fits.py --phase 1 --no-plots

Can also be run cell-by-cell via the '# %%' markers in VS Code / Jupyter's
interactive window; with no CLI args, argparse falls back to the defaults
above.
"""

# %%
import argparse
import os
import pathlib
import sys

try:
    _repo_root = pathlib.Path(__file__).resolve().parents[2]
except NameError:
    _cwd = pathlib.Path.cwd().resolve()
    _repo_root = next(p for p in [_cwd, *_cwd.parents] if (p / 'merge_rna').is_dir())

os.environ.setdefault('RNA_STRUCT_HOME', str(_repo_root))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import RNA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from merge_rna import Experiment, MultiSystemsFit


# %%
# =============================================================================
# CLI
# =============================================================================
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='newseq WT experimental bistable fit')
    parser.add_argument('--phase', choices=['1', '2', 'both'], default='1')
    parser.add_argument('--soft_constraints', default=None,
                        help='Path to optional soft constraints')
    parser.add_argument('--rna-params', choices=['turner', 'andronescu'], default='turner',
                        help='ViennaRNA thermodynamic parameter set (default: turner)')
    parser.add_argument('--max-iter', type=int, default=None)
    parser.add_argument('--no-plots', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--output-dir', default=None,
                        help='Root output directory (default: fits/designed_sequence/outputs/experimental_<rna-params>)')
    args, _unknown = parser.parse_known_args(argv)
    return args

# %%

args = _parse_args() # leave this as default
# args = _parse_args(['--phase', '1'])

if args.rna_params == 'andronescu':
    RNA.params_load_RNA_Andronescu2007()
    print("Loaded Andronescu (2007) RNA parameters")
else:
    RNA.params_load_RNA_Turner2004()
    print("Loaded Turner (2004) RNA parameters")

# %%
# =============================================================================
# Configuration
# =============================================================================
_base = args.output_dir if args.output_dir else os.path.join(str(_repo_root), 'fits', 'designed_sequence', 'outputs', f'experimental_{args.rna_params}')
ROOT_DIR_PHASE1 = os.path.join(_base, 'phase1')
ROOT_DIR_PHASE2 = os.path.join(_base, 'phase2')

# %%
# =============================================================================
# Load experiments
# =============================================================================

def _resolve_newseq_paths(data_root): #  copied function to load experiments paths
    """Return (fastq_names, paths_txt, paths_txt_filter125).

    New layout: data/designed_sequence/experimental/{conc_rep}/{name}_sorted.txt
    """
    new_exp = os.path.join(data_root, 'data', 'designed_sequence', 'experimental')

    if os.path.isdir(new_exp):
        try:
            txt_paths = sorted([
                os.path.join(new_exp, d, f)
                for d in os.listdir(new_exp)
                if os.path.isdir(os.path.join(new_exp, d))
                for f in os.listdir(os.path.join(new_exp, d))
                if f.endswith('_sorted.txt')
            ])
        except OSError:
            txt_paths = []
        names = [os.path.basename(p).removesuffix('_sorted.txt') for p in txt_paths]
        return names, txt_paths  # filter_125 not available in new layout

    return [], []

_newseq_fastq_names, _paths_newseq = _resolve_newseq_paths(os.environ['RNA_STRUCT_HOME'])

experiments = [Experiment(path) for path in _paths_newseq]

print(f"Loaded {len(experiments)} newseq WT experiments:")
for exp in experiments:
    print(f"  - {exp.system} @ {exp.conc_mM}mM, {exp.temp_C}°C, rep {exp.rep_number}")

# %%
# =============================================================================
# Export WT reference FASTA + per-experiment manifest (temp/conc/rep + file paths)
# =============================================================================
def _to_rna_struct_home_path(abs_path):
    """Rewrite an absolute path as '$RNA_STRUCT_HOME/...'-relative, matching the
    portability convention already used in the *_sorted.txt info files."""
    root = os.environ['RNA_STRUCT_HOME']
    rel = os.path.relpath(os.path.abspath(abs_path), root)
    return f"$RNA_STRUCT_HOME/{rel}"


def write_newseq_wt_fasta(experiments, out_path, header='Newseq_WT'):
    """Write the single shared WT reference sequence (RNA alphabet) to a FASTA file."""
    ref_seq = experiments[0].seq
    mismatched = [getattr(exp, 'SRR', exp.short_description) for exp in experiments[1:] if exp.seq != ref_seq]
    if mismatched:
        raise ValueError(
            f"newseq WT experiments do not share an identical reference "
            f"sequence; mismatch in: {mismatched}. Refusing to write {out_path}."
        )
    with open(out_path, 'w') as fh:
        fh.write(f">{header}\n{ref_seq}\n")
    print(f"Wrote WT reference FASTA ({len(ref_seq)} nt) -> {out_path}")
    return ref_seq


def write_experiment_manifest(experiments, out_path):
    """One row per experiment: temp/conc/rep metadata + paths to its .txt/.rc/.mm files."""
    rows = []
    for exp in experiments:
        mm_path = os.path.splitext(exp.path_to_info_txt)[0] + '.mm'
        has_mm = os.path.exists(mm_path)
        rows.append({
            'system':            exp.system_name,
            'SRR':               getattr(exp, 'SRR', ''),
            'ID':                exp.ID,
            'conc_mM':           exp.conc_mM,
            'temp_C':            exp.temp_C,
            'rep_number':        exp.rep_number,
            'short_description': exp.short_description,
            'path_txt':          _to_rna_struct_home_path(exp.path_to_info_txt),
            'path_rc':           _to_rna_struct_home_path(exp.path_to_rc),
            'path_mm':           _to_rna_struct_home_path(mm_path) if has_mm else '',
        })
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(out_path, index=False)
    print(f"Wrote experiment manifest ({len(rows)} rows) -> {out_path}")
    return manifest_df


_designed_seq_dir = os.path.join(os.environ['RNA_STRUCT_HOME'], 'data', 'designed_sequence')

newseq_wt_fasta_path = os.path.join(_designed_seq_dir, 'Newseq_WT.fasta')
newseq_wt_manifest_path = os.path.join(_designed_seq_dir, 'newseq_wt_experiments_manifest.csv')

write_newseq_wt_fasta(experiments, newseq_wt_fasta_path)
write_experiment_manifest(experiments, newseq_wt_manifest_path)

# %%
_common = dict(
    experiments=experiments,
    output_suffix='newseqWT',
    strict_convergence=True,
    do_plots=not args.no_plots,
    print_to_std_out=True,
    max_iter=args.max_iter,
    overwrite=args.overwrite,
)

# %%
# =============================================================================
# Phase 1: physical parameters only, no interpolated pairing probs
# =============================================================================
soft_constraints_path = args.soft_constraints
if args.phase in ('1', 'both'):
    print(f"\n{'='*60}")
    print(f"PHASE 1: fitting physical parameters")
    print(f"{'='*60}")

    fit_struct = MultiSystemsFit(
        root_dir=ROOT_DIR_PHASE1,
        fit_mode='physical_only',
        infer_1D_sc=False,
        use_interpolated_ps=False,
        **_common,
    )
    fit_struct.fit()

    phase1_params = os.path.join(ROOT_DIR_PHASE1, 'newseqWT', 'params1D.txt')
    print(f'\nPhase 1 done. Params: {phase1_params}')


# %%

def compute_error_probabilities(fit_struct, phase1_params):
    """Compute mutation probabilities parameters needed in the form M_i = a_i + b_i*s_i."""

    params_1D_ref = np.loadtxt(phase1_params)[:8]
    params_dict_ref = fit_struct.pack_params(params_1D_ref, fit_struct.systems[0])

    mu_r, p_b, m0, m1 = (params_dict_ref['mu_r'], params_dict_ref['p_b'],
                            params_dict_ref['m0'], params_dict_ref['m1'])

    p_bind_dict = params_dict_ref['p_bind']

    eps_b = fit_struct.systems[0].eps_b
    prob_false_positive = (1 - np.exp(-eps_b-m0))
    # prob_true_positive = (1 - np.exp(-eps_b-m1))
    prob_false_negative = np.exp(-eps_b-m1)
    
    return prob_false_positive, prob_false_negative

prob_fp, prob_fn = compute_error_probabilities(fit_struct, phase1_params)

# %%
# plot the error probabilities

def plot_error_probabilities(prob_fp, prob_fn, fit_struct):
    fig, ax = plt.subplots(figsize=(15, 6))

    # plot your main series
    ax.plot(prob_fp, marker='o', linestyle='-', color='blue', label='Probability of false positive')
    ax.plot(prob_fn, marker='*', linestyle=' ', color='green', label='Probability of false negative')

    # shade regions based on the sequence
    for i, nt in enumerate(fit_struct.systems[0].seq):
        if nt in ['A', 'C']:
            ax.axvspan(i - 0.5, i + 0.5, color='green', alpha=0.15, zorder=0)

    ax.set_xlabel('Position')
    ax.set_ylabel('Error probability')
    ax.legend()

    save_dir = os.path.join(ROOT_DIR_PHASE1, 'newseqWT', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_fig_path = os.path.join(save_dir, 'error_probabilities.png')
    plt.savefig(save_fig_path, bbox_inches='tight')

    return fig, ax

plot_error_probabilities(prob_fp, prob_fn, fit_struct)

# %%
