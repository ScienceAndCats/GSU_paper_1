import scanpy as sc
import pandas as pd

file_name = "09Oct2024_luz19_initial_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt"

# Read the raw data
raw_data = pd.read_csv(file_name, sep='\t', index_col=0)

# Convert the raw data to an AnnData object for processing
adata = sc.AnnData(raw_data)

# Compute total reads per cell
adata.obs['total_counts'] = adata.X.sum(axis=1)

# Visualize read depth distribution
sc.pl.violin(adata, ['total_counts'], jitter=0.4, log=False)

sc.pp.filter_cells(adata, min_counts=0)
sc.pp.filter_cells(adata, max_counts=150)

sc.pp.filter_genes(adata, min_counts=0)

# Visualize read depth distribution
sc.pl.violin(adata, ['total_counts'], jitter=0.4, log=False)