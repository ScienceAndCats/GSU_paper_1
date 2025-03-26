import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys

def load_umis_per_cell(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    umis = adata.X.sum(axis=1)
    if hasattr(umis, 'A1'):  # For sparse matrix
        umis = umis.A1
    return umis

def main(file1, file2, label1, label2, ymax=None, violin_color='lightblue'):
    umis1 = load_umis_per_cell(file1)
    umis2 = load_umis_per_cell(file2)

    df = pd.DataFrame({
        'UMIs': list(umis1) + list(umis2),
        'Sample': [label1] * len(umis1) + [label2] * len(umis2)
    })

    # Calculate and print summary statistics for each sample.
    summary_stats = df.groupby('Sample')['UMIs'].agg(['median', 'mean'])
    print("Summary statistics for UMIs per cell:")
    print(summary_stats)

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Sample', y='UMIs', data=df, inner='box',
                     linewidth=1.25, color=violin_color)
    plt.title('UMIs per Cell')
    plt.ylabel('Total UMIs per Cell')
    plt.xlabel('Sample')
    if ymax is not None:
        plt.ylim(top=ymax)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # When running from the command line
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Compare UMIs per cell between two h5ad files.')
        parser.add_argument('file1', type=str, help='Path to first h5ad file')
        parser.add_argument('file2', type=str, help='Path to second h5ad file')
        parser.add_argument('--label1', type=str, default='Sample 1', help='Label for first sample')
        parser.add_argument('--label2', type=str, default='Sample 2', help='Label for second sample')
        parser.add_argument('--ymax', type=float, default=None, help='Maximum value for y-axis')
        parser.add_argument('--violin_color', type=str, default='lightblue', help='Color for violin plot infills')
        args = parser.parse_args()
        main(args.file1, args.file2, args.label1, args.label2, args.ymax, args.violin_color)
    else:
        # Default file paths and parameters for running in PyCharm
        file1 = "working_data/preprocessed_PETRI_outputs/PAcontrol_mixed_species_gene_matrix_preprocessed_100/PAcontrol_mixed_species_gene_matrix_preprocessed.h5ad"  # "working_data/preprocessed_PETRI_outputs/PAcontrol_tob_PEG_4000/PAcontrol_mixed_species_gene_matrix_preprocessed.h5ad"
        file2 = "working_data/preprocessed_PETRI_outputs/17Sep24_mixed_species_gene_matrix_preprocessed_100/17Sep24_mixed_species_gene_matrix_preprocessed.h5ad" # "working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.h5ad"
        label1 = "PA01"
        label2 = "PA01/Luz19 (rRNA depleted)"
        ymax = None           # Set this to a number (e.g., 10000) to control the max of the y-axis.
        violin_color = 'turquoise' #'#1f77b4'  # Change to your preferred color.
        main(file1, file2, label1, label2, ymax, violin_color)
