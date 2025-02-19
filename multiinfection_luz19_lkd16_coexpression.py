"""
Script for analyzing single-cell RNA-seq data from phage-infected bacterial cells.

## Functionality:
- Loads and filters gene expression data.
- Identifies and quantifies phage gene expression per cell.
- Compares phage expression levels between singly infected and coinfected cells.
- Visualizes expression distributions using violin plots.
- Performs statistical comparisons of expression using the Mann-Whitney U test.
- Computes correlation between phage gene expression in coinfected cells.

## Inputs:
- `13Nov24_RT_multi_infection_gene_matrix.txt`: Tab-separated gene expression matrix.
  - Rows: Cells
  - Columns: Genes
  - Values: Expression counts

## Outputs:
- Printed number of genes removed due to naming issues.
- Violin plots comparing expression levels of Luz19 and LKD16 between coinfected and singly infected cells.
- Mann-Whitney U test results for statistical comparison of phage expression.
- Correlation matrix for Luz19 and LKD16 expression in coinfected cells.

## Dependencies:
- scanpy, numpy, scipy, pandas, seaborn, plotly, matplotlib
"""


import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare, mannwhitneyu
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# -------------------- Data Loading & Filtering -------------------- #
file_path = "13Nov24_RT_multi_infection_gene_matrix.txt"
adata = sc.read(file_path, delimiter="\t")  # Use delimiter="\t" for tab-separated files
print(adata)

# Remove genes with commas in their names
removed_genes = adata.var_names[adata.var_names.str.contains(",")]
adata = adata[:, ~adata.var_names.str.contains(",")]
print(f"Removed {len(removed_genes)} genes with commas.")

# Set filtering thresholds
min_counts_cells = 5
min_counts_genes = 5

# Filter cells and genes by minimum counts
sc.pp.filter_cells(adata, min_counts=min_counts_cells)
sc.pp.filter_genes(adata, min_counts=min_counts_genes)
# -------------------- End Data Loading & Filtering -------------------- #

# -------------------- Phage Analysis -------------------- #
# Make metacolumns with number of each phage genes per cell
phage_patterns = ["luz19:", "lkd16:", "14one:"]
phage_gene_dict = {}

for phage in phage_patterns:
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    phage_gene_dict[phage.strip(':')] = phage_genes

    # Create a boolean array indicating phage genes
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    # Calculate the number of genes expressed per cell for each phage
    if sparse.issparse(adata.X):
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1).A.flatten()
        phage_n_genes = (adata[:, adata.var[f'{phage.strip(":")}_genes']].X > 0).sum(axis=1).A.flatten()
    else:
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1)
        phage_n_genes = (adata[:, adata.var[f'{phage.strip(":")}_genes']].X > 0).sum(axis=1)

    # Add the counts to metadata
    metadata_key = f"{phage.strip(':')}_expression"
    adata.obs[metadata_key] = phage_expression
    adata.obs[f'{phage.strip(":")}_n_genes'] = phage_n_genes


# -------------------- End Phage Analysis -------------------- #

# -------------------- (Optional) Downstream Analysis -------------------- #
# Here we compare expression patterns of luz19 and lkd16 between coinfected cells and singly infected cells.

# Define masks for subsetting cells
coinfected_mask = (adata.obs["luz19_expression"] > 0) & (adata.obs["lkd16_expression"] > 0)
luz19_only_mask = (adata.obs["luz19_expression"] > 0) & (adata.obs["lkd16_expression"] == 0)
lkd16_only_mask = (adata.obs["lkd16_expression"] > 0) & (adata.obs["luz19_expression"] == 0)

# Extract expression data
luz19_coinf = adata.obs.loc[coinfected_mask, "luz19_expression"]
luz19_single = adata.obs.loc[luz19_only_mask, "luz19_expression"]

lkd16_coinf = adata.obs.loc[coinfected_mask, "lkd16_expression"] #NOT redundant, this has only lkd16's expression levels
lkd16_single = adata.obs.loc[lkd16_only_mask, "lkd16_expression"]

# Combine into DataFrames for plotting
df_luz19 = pd.DataFrame({
    "Expression": pd.concat([luz19_coinf, luz19_single]),
    "Group": ["Coinfected"] * len(luz19_coinf) + ["Luz19 Only"] * len(luz19_single)
})

df_lkd16 = pd.DataFrame({
    "Expression": pd.concat([lkd16_coinf, lkd16_single]),
    "Group": ["Coinfected"] * len(lkd16_coinf) + ["LKD16 Only"] * len(lkd16_single)
})

# Plot violin plots comparing expression
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.violinplot(x="Group", y="Expression", data=df_luz19)
plt.title("Luz19 Expression: Coinfected vs Luz19 Only")

plt.subplot(1, 2, 2)
sns.violinplot(x="Group", y="Expression", data=df_lkd16)
plt.title("LKD16 Expression: Coinfected vs LKD16 Only")
plt.tight_layout()
plt.show()

# Statistical testing: Mann-Whitney U Test (non-parametric)
stat_luz19, p_val_luz19 = mannwhitneyu(luz19_coinf, luz19_single)
stat_lkd16, p_val_lkd16 = mannwhitneyu(lkd16_coinf, lkd16_single)
print("Mann-Whitney U Test for Luz19 Expression (Coinfected vs Luz19 Only): p-value =", p_val_luz19)
print("Mann-Whitney U Test for LKD16 Expression (Coinfected vs LKD16 Only): p-value =", p_val_lkd16)

# Correlation analysis in coinfected cells
coinf_data = adata.obs.loc[coinfected_mask, ["luz19_expression", "lkd16_expression"]]
corr_matrix = coinf_data.corr()
print("Correlation between Luz19 and LKD16 expression in coinfected cells:")
print(corr_matrix)
# -------------------- End Downstream Analysis -------------------- #

