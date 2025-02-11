import scrublet as scr
import pandas as pd
import scanpy as sc
import numpy as np

# Define file name
file_name = "17sep2024_Luz19_20min_initial_mixed_species_gene_matrix_hans.txt"

# Read the raw data
raw_data = pd.read_csv(file_name, sep='\t', index_col=0)

# Convert raw data to an AnnData object
adata = sc.AnnData(raw_data)

# Ensure data is properly formatted (convert to dense array if needed)
adata.X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

# Filter out low-quality cells before running Scrublet
sc.pp.filter_cells(adata, min_genes=4)  # Remove cells with <200 detected genes
sc.pp.filter_genes(adata, min_cells=3)  # Remove genes detected in <3 cells

# Determine the maximum valid number of PCA components
n_prin_comps = min(adata.shape[0], adata.shape[1], 7)  # Ensures it does not exceed data dimensions

# Run Scrublet with adjusted PCA components
scrub = scr.Scrublet(adata.X)
adata.obs['doublet_score'], adata.obs['predicted_doublet'] = scrub.scrub_doublets(n_prin_comps=n_prin_comps)

# Save results
adata.write("doublet_filtered_data.h5ad")

# Print first few rows of results
print(adata.obs[['doublet_score', 'predicted_doublet']].head())

# Generate violin plot of doublet scores
sc.pl.violin(adata, ['doublet_score'])
