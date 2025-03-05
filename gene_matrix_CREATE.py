"""
Creates a scRNA-seq gene matrix from the filtered_mapped_UMIs txt file for processing by Scanpy.
"""


#!/usr/bin/env python3
import pandas as pd
import os

# === Hard-coded parameter for PyCharm ===

FILE_PATH = "working_files/preprocessed_PETRI_outputs"
INPUT_FILE = os.path.join(FILE_PATH, "initial_v11_threshold_0_filtered_mapped_UMIs_multihitcombo_preprocessed.txt")
OUTPUT_FILE = os.path.join(FILE_PATH, "output_test2.txt")


#sample = "MySample"  # Replace with your sample name
# ==========================================

# Read the input file that was generated earlier in the pipeline.
input_file = INPUT_FILE
table = pd.read_csv(input_file, sep='\t', index_col=0)

# Select the relevant columns and group by Cell Barcode and contig:gene.
matrix = table[['contig:gene', 'UMI']]
gene_matrix = matrix.groupby(['Cell Barcode', 'contig:gene']).count()

# Reshape the matrix: unstack, fill missing values, transpose, drop extra level.
gene_matrix = gene_matrix.unstack(level='Cell Barcode')
gene_matrix = gene_matrix.fillna(0)
gene_matrix = gene_matrix.transpose()
gene_matrix = gene_matrix.droplevel(0)
# gene_matrix = gene_matrix.loc[:, ~gene_matrix.columns.str.contains('ambiguous')] # get rid of ambiguous contig:genes

# Write the resulting matrix to a file.
output_file = OUTPUT_FILE
gene_matrix.to_csv(output_file, sep='\t')
