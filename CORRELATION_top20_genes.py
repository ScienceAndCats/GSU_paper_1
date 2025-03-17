import scanpy as sc
import numpy as np


"""
This calcs the top 20 positive and negative correlations of genes.
They're just printed out.
"""


# ---------------------------
# Step 1: Load the Data
# ---------------------------
# Replace with your actual file
adata = sc.read_csv("working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.txt", delimiter="\t")

# Convert sparse matrix to dense if necessary
if hasattr(adata.X, "toarray"):
    expr = adata.X.toarray()
else:
    expr = adata.X

# ---------------------------
# Step 2: Get Gene Names
# ---------------------------
# Try to use the gene names from the AnnData object; if not available, generate generic names.
if adata.var_names is not None and len(adata.var_names) == expr.shape[1]:
    gene_names = adata.var_names
else:
    gene_names = [f"Gene_{i}" for i in range(expr.shape[1])]

# ---------------------------
# Step 3: Compute Pairwise Gene Correlation
# ---------------------------
# Transpose the expression matrix so that each row corresponds to a gene.
corr_matrix = np.corrcoef(expr.T)

# ---------------------------
# Step 4: Extract Gene Pairs and Their Correlations
# ---------------------------
n_genes = corr_matrix.shape[0]
pairs = []
# Loop over the upper triangle of the matrix (excluding the diagonal) to avoid duplicate pairs
for i in range(n_genes):
    for j in range(i + 1, n_genes):
        pairs.append((gene_names[i], gene_names[j], corr_matrix[i, j]))

# ---------------------------
# Step 5: Sort and Select Top 20 Pairs for Positive and Negative Correlations
# ---------------------------
# Filter out pairs with positive correlations and sort them in descending order
pairs_positive = [pair for pair in pairs if pair[2] > 0]
pairs_positive_sorted = sorted(pairs_positive, key=lambda x: x[2], reverse=True)
top20_positive = pairs_positive_sorted[:20]

# Filter out pairs with negative correlations and sort them in ascending order (most negative first)
pairs_negative = [pair for pair in pairs if pair[2] < 0]
pairs_negative_sorted = sorted(pairs_negative, key=lambda x: x[2])
top20_negative = pairs_negative_sorted[:20]

# ---------------------------
# Step 6: Print the Results
# ---------------------------
print("Top 20 gene pairs with positive correlation:")
for gene1, gene2, corr_value in top20_positive:
    print(f"{gene1} - {gene2}: {corr_value:.4f}")

print("\nTop 20 gene pairs with negative correlation:")
for gene1, gene2, corr_value in top20_negative:
    print(f"{gene1} - {gene2}: {corr_value:.4f}")
