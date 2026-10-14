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
import pandas as pd
from merge_rna.experiment import load_rc, to_int,bytes_to_seq

# %%
OUTPUT_DIR = os.path.join('my_merge_rna', 'outputs', 'fasta_files')

filepath = '/u/m/mmaglion/Documents/MERGE-RNA_restored/data/designed_sequence/experimental/100mM_r1/058_D_100mM_GiuseppeWT_1_sorted.rc'
df = load_rc(filepath, 'prova')

# %%
with open(filepath,"rb") as infile:
    # Same notation as RNA framework article here
    len_transcript_id = to_int(infile.read(4)) # read in the first 4 bytes to get the length of the transcript id, needed in the next line
    transcript_id = infile.read(len_transcript_id).decode("utf8") # read in the transcript id
    len_seq = to_int(infile.read(4)) # read in the length of the sequence
    seq = bytes_to_seq(infile.read(int((len_seq+1)/2)))

# %%
# bytes_to_seq decodes 2 nt per byte, so an odd-length sequence gets a trailing pad nt; drop it
seq = seq[:len_seq]

# %%
sequence_dict = {transcript_id[:len_transcript_id-1]: ''.join(seq)}

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_filename = os.path.splitext(os.path.basename(filepath))[0] + '.fasta'
output_path = os.path.join(OUTPUT_DIR, output_filename)

with open(output_path, 'w') as output_file:
    for seq_id, seq in sequence_dict.items():
        output_file.write(">" + seq_id + "\n")
        output_file.write(seq + "\n")

print(f"Wrote {len(sequence_dict)} sequence(s) to {output_path}")
# %%
