import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# Replace with your file name/path
h5ad_file = "working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.h5ad"

# Load the AnnData object
adata = sc.read(h5ad_file)

# Ensure the 'bc1_selection' column exists in adata.obs
if 'bc1_selection' not in adata.obs.columns:
    raise ValueError("The 'bc1_selection' column is not found in adata.obs")

# Get unique cell groups from the 'bc1_selection' column
groups = adata.obs['bc1_selection'].unique()

# Set up subplots (one per group)
n_groups = len(groups)
fig, axs = plt.subplots(n_groups, 1, figsize=(10, 5 * n_groups))

# In case there's only one group, wrap axs in a list for consistent handling
if n_groups == 1:
    axs = [axs]

# Loop over each group
for ax, group in zip(axs, groups):
    # Subset the data to cells in the current group
    adata_group = adata[adata.obs['bc1_selection'] == group]

    # Calculate the mean expression for each gene across cells
    # adata_group.X may be sparse; convert it to a dense array if needed.
    if issparse(adata_group.X):
        # Compute mean along axis=0, then convert to 1D array
        mean_expression = np.array(adata_group.X.mean(axis=0)).flatten()
    else:
        mean_expression = np.array(adata_group.X.mean(axis=0)).flatten()

    # Create a pandas Series with gene names as index
    gene_means = pd.Series(mean_expression, index=adata.var_names)

    # Sort the genes by mean expression and take the top 20
    top20 = gene_means.sort_values(ascending=False).head(20)

    # Plot a bar chart for the top 20 genes
    ax.bar(top20.index, top20.values)
    ax.set_title(f"Top 20 Genes in Group '{group}'")
    ax.set_ylabel("Mean Expression per Cell")
    ax.set_xticklabels(top20.index, rotation=45, ha="right")

plt.tight_layout()
plt.show()
