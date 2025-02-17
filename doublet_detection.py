import scrublet as scr
import pandas as pd
import scanpy as sc
import numpy as np

# Load data
file_name = "09Oct2024_luz19_initial_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt"
raw_data = pd.read_csv(file_name, sep='\t', index_col=0)
adata = sc.AnnData(raw_data)

# 🔹 Fix: Ensure we modify the actual object (avoids implicit view warning)
adata = adata.copy()

# Convert sparse matrix to dense if needed
if hasattr(adata.X, "toarray"):
    adata.X = adata.X.toarray()

# 🔹 STEP 1: Filter low-quality cells & genes BEFORE Scrublet
sc.pp.filter_cells(adata, min_genes=5)  # Remove cells with <5 detected genes
sc.pp.filter_genes(adata, min_cells=3)  # Remove genes detected in <3 cells
adata = adata[adata.X.sum(axis=1) > 5, :]  # Remove cells with <5 UMIs

# 🔹 STEP 2: Check if all cells were removed
if adata.shape[0] == 0:
    raise ValueError("Error: No cells remain after filtering. Try lowering min_genes or min_UMIs.")

# Print dataset size after filtering
print(f"Cells remaining after filtering: {adata.shape[0]}")
print(f"Genes remaining after filtering: {adata.shape[1]}")

# 🔹 STEP 3: Remove any NaN or infinite values
adata.X[np.isnan(adata.X)] = 0  # Replace NaN with 0
adata.X[np.isinf(adata.X)] = 0  # Replace infinite values with 0

# 🔹 STEP 4: Normalize data (optional but recommended)
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize per cell
sc.pp.log1p(adata)  # Log-transform for numerical stability

# 🔹 STEP 5: Run Scrublet First to Determine Number of Highly Variable Genes
scrub = scr.Scrublet(adata.X)
scrub.scrub_doublets(n_prin_comps=10)  # This computes gene variability without affecting final results

# Get number of highly variable genes selected by Scrublet
num_highly_variable_genes = np.count_nonzero(scrub._gene_filter)
print(f"Total genes in adata: {adata.shape[1]}")
print(f"Highly variable genes selected by Scrublet: {num_highly_variable_genes}")

# 🔹 STEP 6: Dynamically Set PCA Components Based on Available Features
n_prin_comps = max(1, min(adata.shape[0], num_highly_variable_genes, 4))  # Ensure PCA is valid

# 🔹 STEP 7: Run Scrublet Again with the Correct PCA Settings
adata.obs['doublet_score'], adata.obs['predicted_doublet'] = scrub.scrub_doublets(
    n_prin_comps=n_prin_comps, svd_solver='auto'
)

# Save results
adata.write("doublet_filtered_data.h5ad")

# Print first few rows of doublet scores
print(adata.obs[['doublet_score', 'predicted_doublet']].head())

# Generate violin plot of doublet scores
sc.pl.violin(adata, ['doublet_score'])
