#!/usr/bin/env python3
"""
Generate synthetic mutation experiment data.
Usefull fortesting the merge_rna parameters.
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
import bussilab.rna2d as bussi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from merge_rna import Experiment, MultiSystemsFit

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

# %%
p_reaction_given_unpaired_nb, p_reaction_given_paired_nb, a_i, b_i, penalty = compute_mutation_probabilities(fit_struct, phase1_params)

# merge_rna fits only the positions inside position_mask (first/last 25 nt excluded): the model is not
# constrained on the ends and overestimates their mutation rate, so they are treated as unobserved below.
position_mask = fit_struct.systems[0].exp_fits_train[0].position_mask
offset = exp.df.index.min()                     # kept index (exp.seq, synthetic) -> raw index (.mm file)
lo, hi = np.where(position_mask)[0][[0, -1]]    # fitted window, kept indexing
print(f"Fitted window: kept positions {lo}..{hi} (raw .mm {lo+offset}..{hi+offset})")

# %%
# =============================================================================
# Extract reaction states
# =============================================================================

ref_params_path = os.path.join(str(_repo_root), 'fits_paper', 'designed_sequence', 'experimental', 'strand_disp_turner.txt')
lambda_sc = np.loadtxt(ref_params_path)[8:]
lambda_sc = np.zeros(len(lambda_sc)) # for now, no soft constraints

sample_number = 50000

# mol = bussi.Molecule(seq, lambdas1d=lambda_sc+penalty*np.ones(len(seq)))

bias = [[0,0],[0,-4]]
mol_0 = bussi.Molecule(seq, state_positions=[[80,86],[46,52]],state_biases=[bias,bias])
mat = mol_0.pairing_correlation_matrix()
bpp = mol_0.base_pairing_probability()
bpp=np.sum(bpp,axis=0)
aver_mult = np.matmul(np.atleast_2d(bpp).T, np.atleast_2d(bpp))

plt.plot(mol_0.base_pairing_probability().sum(1))

plt.figure(figsize=(6, 5))
plt.imshow(mat-aver_mult, cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(label='Pairing correlation')

# %%

sampled_states, logweights = bussi.sample_to_numpy(mol_0.sample(sample_number), deduplicate=False)
# I do it in the stupidest way

sampled_reaction_states = np.zeros(sampled_states.shape, dtype=int)
for i in range(sample_number):
    for j in range(len(seq)):
        if sampled_states[i,j] == -1:
            sampled_reaction_states[i,j] = np.random.binomial(1, p_reaction_given_unpaired_nb[j])
        else:
            sampled_reaction_states[i,j] = np.random.binomial(1, p_reaction_given_paired_nb[j])

# %%
# =============================================================================
# Extract mutation states
# =============================================================================

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

sampled_mutation_states = np.zeros(sampled_states.shape, dtype=int)
for i in range(sample_number):
    for j in range(len(seq)):
        if sampled_reaction_states[i,j] == 0:
            sampled_mutation_states[i,j] = np.random.binomial(1, prob_fp[j])
        else:
            sampled_mutation_states[i,j] = np.random.binomial(1, 1-prob_fn[j]) # this needs the true positive probability

sampled_mutation_states_full = sampled_mutation_states.copy()  # all positions, kept only for the diagnostics
sampled_mutation_states[:, ~position_mask] = 0                  # unfitted ends: treat as unobserved
sampled_reaction_states[:, ~position_mask] = 0

# %%

reaction_counts = sampled_reaction_states.sum(axis=1)
max_k_reactions = max(reaction_counts)
bins_reactions = range(0, max_k_reactions + 2)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(reaction_counts, bins=bins_reactions, align='left', histtype='step',
            linewidth=2,
            label="Reaction counts")
ax.set_xlabel('counts')
ax.set_ylabel('population')
ax.legend()
fig.tight_layout()

# %%

mutation_counts = sampled_mutation_states.sum(axis=1)
max_k_mutations= max(mutation_counts)
bins_mutations = range(0, max_k_mutations + 2)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(mutation_counts, bins=bins_mutations, align='left', histtype='step',
            linewidth=2,
            label="Mutation counts")
ax.set_xlabel('counts')
ax.set_ylabel('population')
ax.legend()
fig.tight_layout()

# %%
# =============================================================================
# Save the synthetic mutation experiment data in .csv format
# =============================================================================

def mutation_states_to_dataframe(mutation_states, sample_name="synthetic"):
    """Convert the mutation states array to a pandas DataFrame with columns indicating
    the read index, the number of mutations per read and mutation positions."""

    unique_mut_counts_per_read = [] # initialize an array to store the mutation counts per read
    unique_mut_positions = [] # initialize a list to store the mutation positions for each read
    occurrances = [] # initialize a list to store the number of occurrences of each unique mutation state

    indices = {}
    for state in mutation_states:
        mut_count = np.sum(state)
        mut_positions = np.where(state == 1)[0].tolist() # get the positions
        mut_pos_str = str(mut_positions) # convert to string for hashing
        if mut_pos_str in indices:
            index = indices[mut_pos_str]
            occurrances[index] += 1
        else:
            indices[mut_pos_str] = len(unique_mut_positions)
            unique_mut_counts_per_read.append(mut_count)
            unique_mut_positions.append(mut_positions)
            occurrances.append(1)

    data = {'Sample' : sample_name,
            'mut_count_per_read' : unique_mut_counts_per_read,
            'mut_positions' : unique_mut_positions,
            'occurrences' : occurrances}    

    return pd.DataFrame(data)

mutation_df = mutation_states_to_dataframe(sampled_mutation_states, sample_name=f"synthetic_{CONC_MM}mM_r{REP_NUMBER}")

# %%
# save dataframe
path_synthetic_data = os.path.join(os.environ['RNA_STRUCT_HOME'], 'data', 'designed_sequence', 'synthetic_mutation_experiments')
os.makedirs(path_synthetic_data, exist_ok=True)
mutation_df.to_csv(os.path.join(path_synthetic_data, f"synthetic_{CONC_MM}mM_r{REP_NUMBER}.csv"), index=False)
# save the position mask (True = observed/fitted, kept indexing of exp.seq): masked positions carry no information
np.save(os.path.join(path_synthetic_data, f"synthetic_{CONC_MM}mM_r{REP_NUMBER}_mask.npy"), position_mask)

# %%
from merge_rna.experiment import load_rc, to_int, bytes_to_seq

# %%
from scripts.mm_file_analysis import load_mm, read_info_field
from matplotlib.patches import Patch

EXPERIMENTAL_DIR = os.path.join('data', 'designed_sequence', 'experimental')
sample = {'label': str(CONC_MM)+'mM_r'+str(REP_NUMBER),
     'mm_path': os.path.join(EXPERIMENTAL_DIR, '80mM_r1', '058_D_80mM_GiuseppeWT_1_sorted.mm'),
     'info_path': os.path.join(EXPERIMENTAL_DIR, '80mM_r1', '058_D_80mM_GiuseppeWT_1_sorted.txt'),
     'concentration_mM': 80, 'replicate': 1, 'linestyle': '-'}

df = load_mm(sample['mm_path'], sample['label'])
sample['df'] = df

filtered = df[(df['read_start'] < 20) & (df['read_end'] > 140)].copy()
filtered['read_len'] = filtered['read_end'] - filtered['read_start'] + 1
sample['df_filtered'] = filtered

total_reads = int(read_info_field(sample['info_path'], 'Total reads (rf-count paper flags)'))
mutated_reads = int(read_info_field(sample['info_path'], 'Mutated reads (rf-count paper flags)'))
assert mutated_reads == len(df), (
    f"{sample['label']}: info.txt mutated count ({mutated_reads}) != .mm read_count ({len(df)})"
)
sample['total_reads'] = total_reads
sample['zero_mut_reads'] = total_reads - mutated_reads

# count only the mutations inside the fitted window (same positions as the synthetic reads)
df['mut_count_window'] = df['mut_positions'].apply(lambda p: int(np.sum((p >= lo + offset) & (p <= hi + offset))))

fig, ax = plt.subplots(1, 1, figsize=(6 * 1, 4))
max_k = int(df['mut_count_window'].max())
counts_mutated, _ = np.histogram(df['mut_count_window'], bins=range(1, max_k + 2))
counts = np.concatenate(([total_reads - np.sum(df['mut_count_window'] > 0)], counts_mutated))
colors = ['0.7'] + ['tab:blue'] * max_k
ax.bar(range(0, max_k + 1), counts, color=colors, edgecolor='black')
ax.set_title(f"{sample['label']} (N={sample['total_reads']}), fitted window")
ax.set_xlabel('Mutations per read (fitted window)')
ax.set_ylabel('Number of reads')
ax.legend(handles=[Patch(color='0.7', label='k=0 (info.txt total - reads mutated in window)'),
                    Patch(color='tab:blue', label='k>=1 (.mm file)')], fontsize=8)
fig.tight_layout()

# %%
# =============================================================================
# Synthetic vs real: where they differ
# =============================================================================
# Note: .mm positions are raw indices, synthetic/exp.seq positions are kept indices; raw = kept + offset.

# 1) per-site mutation rates: the model overestimates only on the masked (unfitted) ends
model_rate = a_i + b_i * bpp                    # M_i = a_i + b_i <s_i>, generating ensemble (mol_0)
data_rate = exp.df['mut_rate'].values
synth_rate = sampled_mutation_states_full.mean(axis=0)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(data_rate, 'k.-', label='data (rc)')
ax.plot(model_rate, 'r-', label='model $a_i+b_i\\langle s_i\\rangle$')
ax.plot(synth_rate, 'b.', alpha=0.6, label='synthetic (before masking)')
ax.axvspan(-0.5, lo - 0.5, color='0.85', zorder=0, label='masked in merge_rna fit')
ax.axvspan(hi + 0.5, len(seq) - 0.5, color='0.85', zorder=0)
ax.set_xlabel('Position (kept index)')
ax.set_ylabel('Mutation rate')
ax.legend()
fig.tight_layout()

# 2) summary: mean mutations per read, P(k=0) and E[k|k>=1]
k_synth_full = sampled_mutation_states_full.sum(axis=1)
k_synth = sampled_mutation_states.sum(axis=1)
k_real = df['mut_count_window'].values          # mutated reads only; zeros come from total_reads
spanning = (df['read_start'] <= lo + offset) & (df['read_end'] >= hi + offset)
k_real_span = k_real[spanning.values & (k_real > 0)]
all_real_positions = np.concatenate(df["mut_positions"].values)
n_real_mut_full = np.sum((all_real_positions >= offset) & (all_real_positions < offset + len(seq)))

print(f"{'':32s}{'all 130 nt':>12s}{'fitted window':>15s}")
print(f"{'mean muts/read  model sum M_i':32s}{model_rate.sum():12.3f}{model_rate[position_mask].sum():15.3f}")
print(f"{'                data sum rate':32s}{data_rate.sum():12.3f}{data_rate[position_mask].sum():15.3f}")
print(f"{'                synthetic':32s}{k_synth_full.mean():12.3f}{k_synth.mean():15.3f}")
print(f"{'                real (.mm/total)':32s}{n_real_mut_full/total_reads:12.3f}{k_real.sum()/total_reads:15.3f}")
print(f"{'P(k=0)          synthetic':32s}{np.mean(k_synth_full == 0):12.3f}{np.mean(k_synth == 0):15.3f}")
print(f"{'                real (approx.)':32s}{sample['zero_mut_reads']/total_reads:12.3f}{1 - np.sum(k_real > 0)/total_reads:15.3f}")
print(f"{'E[k|k>=1]       synthetic':32s}{k_synth_full[k_synth_full > 0].mean():12.3f}{k_synth[k_synth > 0].mean():15.3f}")
print(f"{'                real (spanning)':32s}{'':12s}{k_real_span.mean():15.3f}")

# 3) P(k|k>=1) in the window: real reads are overdispersed wrt synthetic and factorized model
def poisson_binomial(probs):
    """Distribution of the number of successes of independent Bernoulli trials."""
    dist = np.array([1.])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return dist

pk_factorized = poisson_binomial(model_rate[position_mask])   # mergeRNA-like, no correlations

kmax = int(max(k_real_span.max(), k_synth.max()))
ks = np.arange(1, kmax + 1)
pk_real = np.array([np.mean(k_real_span == k) for k in ks])
pk_synth = np.array([np.mean(k_synth[k_synth > 0] == k) for k in ks])
pk_fact = pk_factorized[1:kmax + 1] / pk_factorized[1:].sum()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, pk_real, 'ko-', label='real (reads spanning window)')
ax.plot(ks, pk_synth, 'bs-', label='synthetic')
ax.plot(ks, pk_fact, 'r^--', label='factorized (independent sites)')
ax.set_yscale('log')
ax.set_ylim(bottom=0.5 / len(k_real_span))     # resolution of the real data
ax.set_xlabel('Mutations per read in fitted window, k')
ax.set_ylabel('P(k | k >= 1)')
ax.legend()
fig.tight_layout()



# %%
