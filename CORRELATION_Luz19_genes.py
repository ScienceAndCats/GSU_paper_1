import scanpy as sc
import numpy as np

# ---------------------------
# Step 1: Load the Data from an h5ad file
# ---------------------------
# Replace with your actual file path. Note the extension is now .h5ad.
adata = sc.read(
    "working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.h5ad")

# ---------------------------
# Step 2: Retrieve Gene Expression Data and Gene Names
# ---------------------------
# If the expression matrix is sparse, convert it to dense.
if hasattr(adata.X, "toarray"):
    expr = adata.X.toarray()
else:
    expr = adata.X

# Use gene names from the AnnData object if available; otherwise, generate generic names.
if adata.var_names is not None and len(adata.var_names) == expr.shape[1]:
    gene_names = list(adata.var_names)
else:
    gene_names = [f"Gene_{i}" for i in range(expr.shape[1])]

# ---------------------------
# Step 3: Compute Pairwise Gene Correlation
# ---------------------------
# Transpose the expression matrix so that each row corresponds to a gene.
corr_matrix = np.corrcoef(expr.T)

# ---------------------------
# Step 4: For Each 'luz19:' Gene, Find the Best Positive and Negative Correlates
# ---------------------------
print("Genes with 'luz19:' prefix and their best correlations:")

for i, gene in enumerate(gene_names):
    if gene.startswith("luz19:"):
        # Get the correlation vector for gene 'i'
        corr_row = corr_matrix[i].copy()
        # Exclude self-correlation by setting it to NaN
        corr_row[i] = np.nan

        # Find the index of the highest positive correlation
        pos_idx = np.nanargmax(corr_row)
        best_pos_corr = corr_row[pos_idx]

        # Find the index of the lowest (most negative) correlation
        neg_idx = np.nanargmin(corr_row)
        best_neg_corr = corr_row[neg_idx]

        # Print the results for this gene
        print(f"\n{gene}:")
        print(f"  Highest positive correlation: {gene_names[pos_idx]} (r = {best_pos_corr:.4f})")
        print(f"  Highest negative correlation: {gene_names[neg_idx]} (r = {best_neg_corr:.4f})")
