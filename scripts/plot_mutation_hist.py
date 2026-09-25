#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path, sample, out_dir):
    df = pd.read_csv(csv_path)

    if sample:
        df = df[df['Sample'] == sample]

    grouped = df.groupby('pos')['mut_count'].sum().sort_index()

    os.makedirs(out_dir, exist_ok=True)

    # Bar plot: mutation count per position
    plt.figure(figsize=(14, 4))
    plt.bar(grouped.index, grouped.values, width=1.0)
    plt.xlabel('Position')
    plt.ylabel('Mutation count (sum)')
    title = 'Mutation counts per position'
    if sample:
        title += f' — sample {sample}'
    plt.title(title)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, 'mutation_counts_per_position_bar.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()

    # Histogram: distribution of mutation counts across positions
    plt.figure(figsize=(6, 4))
    plt.hist(grouped.values, bins=50)
    plt.xlabel('Mutation count (sum per position)')
    plt.ylabel('Number of positions')
    plt.title('Histogram of mutation counts per position')
    plt.tight_layout()
    hist_path = os.path.join(out_dir, 'mutation_counts_per_position_hist.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print('Saved:', bar_path)
    print('Saved:', hist_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot mutation counts per position')
    parser.add_argument('--csv', default='/u/m/mmaglion/Documents/MERGE-RNA_restored/data/adenine_riboswitch/mutation_profiles.csv')
    parser.add_argument('--sample', default=None, help='Optional sample name to filter (e.g. SRR15560844)')
    parser.add_argument('--out', default='plots', help='Output directory for plots')
    args = parser.parse_args()
    main(args.csv, args.sample, args.out)
