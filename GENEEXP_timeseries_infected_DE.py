import scanpy as sc
import numpy as np
from scipy import sparse
import pandas as pd

"""
Classifies cells by timepoint, filters for non-phage-expressing cells,
performs differential expression analysis per timepoint group, 
forms the union of the top DE genes from each group, and then outputs a CSV file 
that contains, for each timepoint group, the summary expression data 
(raw sum, mean, and standard deviation) for the union gene set.
"""


# -------------------- Define Helper Function -------------------- #
def classify_cell(cell_name):
    """Extracts the bc1 value from the cell name and assigns a timepoint."""
    bc1_value = int(cell_name.split('_')[2])
    if bc1_value < 25:
        return "0min"
    elif bc1_value < 49:
        return "10min"
    elif bc1_value < 73:
        return ">30min"
    else:
        return ">30min"


# -------------------- Data Loading & Filtering -------------------- #
file_path = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_mixed_species_gene_matrix_preprocessed.h5ad"  #  "working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.h5ad"
adata = sc.read(file_path, delimiter="\t")

# Remove genes with commas in their names
adata = adata[:, ~adata.var_names.str.contains(",")]

# Filter cells and genes by minimum counts
sc.pp.filter_cells(adata, min_counts=5)
sc.pp.filter_genes(adata, min_counts=5)

# -------------------- Classify Cells into Timepoints -------------------- #
adata.obs["timepoint"] = [classify_cell(cell_name) for cell_name in adata.obs_names]

# -------------------- Phage Analysis -------------------- #
phage_patterns = ["luz19:", "lkd16:"]
for phage in phage_patterns:
    # Identify phage-specific genes
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    # Sum expression across these phage genes
    if sparse.issparse(adata.X):
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1).A.flatten()
    else:
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1)

    # Save expression in obs for downstream filtering
    adata.obs[f"{phage.strip(':')}_expression"] = phage_expression

# -------------------- Filtering for Phage Expressing Cells -------------------- #
# Keep cells that have phage gene expression
filter_mask = (
        (adata.obs["luz19_expression"] > 0) &
        (adata.obs["lkd16_expression"] > 0)
)
adata_filtered = adata[filter_mask].copy()

# -------------------- Differential Expression Analysis -------------------- #
# Run DE analysis over timepoints using the Wilcoxon method (top 100 genes per group)
sc.pp.log1p(adata_filtered)

sc.tl.rank_genes_groups(adata_filtered, groupby="timepoint", method="wilcoxon", n_genes=100)

# -------------------- Form the Union of Top DE Genes -------------------- #
timepoints = adata_filtered.obs["timepoint"].unique()
union_genes = set()
for tp in timepoints:
    de_df = sc.get.rank_genes_groups_df(adata_filtered, group=tp)
    genes = de_df["names"].tolist()
    union_genes.update(genes)
union_genes = list(union_genes)

# -------------------- Compute Summary Expression Data for Union Genes -------------------- #
results = []  # list to hold summary data for each gene and each group
for tp in timepoints:
    # Subset cells for the current timepoint
    adata_group = adata_filtered[adata_filtered.obs["timepoint"] == tp]

    # Identify indices for the union genes (that are in the dataset)
    gene_indices = [adata_group.var_names.get_loc(gene) for gene in union_genes if gene in adata_group.var_names]

    # Extract expression matrix for these genes and convert to dense if needed
    if sparse.issparse(adata_group.X):
        expr_data = adata_group.X[:, gene_indices].toarray()
    else:
        expr_data = adata_group.X[:, gene_indices]

    # Compute summary statistics for each gene
    raw_sum = np.sum(expr_data, axis=0)
    mean_expr = np.mean(expr_data, axis=0)
    std_expr = np.std(expr_data, axis=0)
    total_cells = adata_group.n_obs

    # Save a row for each gene in the current group
    genes_used = [adata_group.var_names[i] for i in gene_indices]
    for i, gene in enumerate(genes_used):
        results.append({
            "gene": gene,
            "timepoint": tp,
            "raw_sum": raw_sum[i],
            "mean_expr": mean_expr[i],
            "std_expr": std_expr[i],
            "total_cells": total_cells
        })

# Combine all results into a single DataFrame and output to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("infected_union_top_DE_genes_expression_by_group.csv", index=False)

print(
    "Done! CSV file 'infected_union_top_DE_genes_expression_by_group.csv' has been created with expression data for the union of top DE genes across groups.")
