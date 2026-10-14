# %%

# %%
import os
import sys
import pickle

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RNA_STRUCT_HOME", repo_root)
sys.path.insert(0, repo_root)
os.chdir(repo_root)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from merge_rna.experiment import load_rc, to_int, bytes_to_seq

# %%
def load_mm(filepath, sample_name, debug=False):
    """Load RNA structure data from .mm file and return it as a pandas DataFrame. Similar to load_rc from experiment.py"""
    with open(filepath,"rb") as infile:
        # Same notation as RNA framework article here
        len_transcript_id = to_int(infile.read(2)) # read in the first 4 bytes to get the length of the transcript id, needed in the next line
        transcript_id = infile.read(len_transcript_id).decode("utf8") # read in the transcript id
        len_seq = to_int(infile.read(4)) # read in the length of the sequence
        seq = bytes_to_seq(infile.read(int((len_seq+1)/2)))

        read_count = to_int(infile.read(4)) # read in the number of reads, which is 4 bytes
        read_start = np.empty(read_count, dtype=int) # initialize an array to store the read start positions
        read_end = np.empty(read_count, dtype=int) # initialize an array to store the read end positions
        mut_counts_per_read = np.empty(read_count, dtype=int) # initialize an array to store the mutation counts per read
        mut_positions = [] # initialize a list to store the mutation positions for each read

        for i in range(read_count):
            read_start[i] = to_int(infile.read(4))
            read_end[i] = to_int(infile.read(4))
            mut_counts_per_read[i] = to_int(infile.read(4))
            mut_positions.append(np.array([to_int(infile.read(4))for i in range(mut_counts_per_read[i])])) # read in the mutation positions, which are 4 bytes each

        rest_of_file = infile.read()

    if debug:
        print("rest of file", rest_of_file)
        print("number of reads", read_count)

    # Redmond's comment: correct sequence if odd length due to way bytes are read in
    seq = seq[:len_seq] # modified wrt load_rc on suggestion of Claude, not sure about it


    # Dictionary to store the data
    data = {'Sample' : sample_name,
            'mut_count_per_read' : mut_counts_per_read,
            'read_start' : read_start,
            'read_end' : read_end,
            'mut_positions' : mut_positions}

    return pd.DataFrame(data)

# %%
def read_info_field(info_path, field):
    """Read one 'Key:\tValue' field from an experiment info.txt (format parsed by
    merge_rna.experiment.Experiment.load_experiment_info: first ':' splits key/value)."""
    with open(info_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            key, _, value = line.strip().partition(':')
            if key.strip() == field:
                return value.strip()
    raise KeyError(f"Field {field!r} not found in {info_path}")

# %%
# Samples with per-read (.mm) data available today. Append more entries here once other
# concentrations/replicates have been through `rf-count -mm` (see mm_analysis_context.md) --
# nothing below needs to change to pick them up.
EXPERIMENTAL_DIR = os.path.join('data', 'designed_sequence', 'experimental')
SAMPLES = [
    {'label': '80mM_r1',
     'mm_path': os.path.join(EXPERIMENTAL_DIR, '80mM_r1', '058_D_80mM_GiuseppeWT_1_sorted.mm'),
     'info_path': os.path.join(EXPERIMENTAL_DIR, '80mM_r1', '058_D_80mM_GiuseppeWT_1_sorted.txt'),
     'concentration_mM': 80, 'replicate': 1, 'linestyle': '-'},
    {'label': '0mM_r1',
     'mm_path': os.path.join(EXPERIMENTAL_DIR, '0mM_r1', '058_E_GiuseppeWT_1_sorted.mm'),
     'info_path': os.path.join(EXPERIMENTAL_DIR, '0mM_r1', '058_E_GiuseppeWT_1_sorted.txt'),
     'concentration_mM': 0, 'replicate': 1, 'linestyle': '--'},
]
max_conc = max(s['concentration_mM'] for s in SAMPLES)

# %%
# Load every sample and apply the near-full-length filter (read_start < 20 & read_end > 140),
# which isolates the ~130nt peak that already dominates the unfiltered read-length
# distribution, so it is not a real additional selection effect.
#
# Also recover each sample's true read total from its info.txt. The .mm file only contains
# reads with >=1 mutation (see mm_analysis_context.md), so `total - mutated` gives the missing
# k=0 count -- but only for the *whole* sample, not for the near-full-length subset specifically
# (rf-count's summary isn't broken down by read length), so this is only used for the unfiltered
# per-sample histograms below, not for the near-full-length comparison.
for sample in SAMPLES:
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

# %%
# Unfiltered mutations-per-read histogram per sample, with the k=0 bar recovered from
# info.txt (grey) alongside the k>=1 bars read directly from the .mm file (blue).
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(6 * len(SAMPLES), 4))
for ax, sample in zip(np.atleast_1d(axes), SAMPLES):
    df = sample['df']
    max_k = int(df['mut_count_per_read'].max())
    counts_mutated, _ = np.histogram(df['mut_count_per_read'], bins=range(1, max_k + 2))
    counts = np.concatenate(([sample['zero_mut_reads']], counts_mutated))
    colors = ['0.7'] + ['tab:blue'] * max_k
    ax.bar(range(0, max_k + 1), counts, color=colors, edgecolor='black')
    ax.set_title(f"{sample['label']} (N={sample['total_reads']})")
    ax.set_xlabel('Mutations per read')
    ax.set_ylabel('Number of reads')
    ax.legend(handles=[Patch(color='0.7', label='k=0 (info.txt: total - mutated)'),
                        Patch(color='tab:blue', label='k>=1 (.mm file)')], fontsize=8)
fig.tight_layout()

# %%
# Near-full-length comparison across samples/concentrations. Built only from the .mm file, so
# -- unlike the unfiltered plot above -- it is conditioned on "mutated reads" and intentionally
# starts at k=1; no zero bin is added here (see note above on why that total isn't available
# for this length-restricted subset).
colors_by_label = {s['label']: plt.cm.viridis(s['concentration_mM'] / max_conc) for s in SAMPLES}
max_k_filtered = max(int(s['df_filtered']['mut_count_per_read'].max()) for s in SAMPLES)
bins = range(1, max_k_filtered + 2)

fig, ax = plt.subplots(figsize=(8, 5))
for sample in SAMPLES:
    filtered = sample['df_filtered']
    ax.hist(filtered['mut_count_per_read'], bins=bins, align='left', density=True, histtype='step',
            linewidth=2, linestyle=sample['linestyle'], color=colors_by_label[sample['label']],
            label=f"{sample['label']} (N={len(filtered)})")
ax.set_xlabel('Mutations per read (near-full-length reads, k>=1 only)')
ax.set_ylabel('Density')
ax.legend()
fig.tight_layout()

# %%
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(6 * len(SAMPLES), 4), sharex=True)
for ax, sample in zip(np.atleast_1d(axes), SAMPLES):
    filtered = sample['df_filtered']
    ax.hist(filtered['mut_count_per_read'], bins=bins, align='left',
             color=colors_by_label[sample['label']], edgecolor='black')
    ax.set_title(f"{sample['label']} (N={len(filtered)})")
    ax.set_xlabel('Mutations per read')
    ax.set_ylabel('Number of reads')
fig.tight_layout()
# %%
