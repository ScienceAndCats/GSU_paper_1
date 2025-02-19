"""
Script for analyzing single-cell RNA-seq data from phage-infected bacterial cells.

## Functionality:
- Loads and filters gene expression data.
- Identifies and quantifies phage gene expression per cell.
- Determines infection status based on phage expression levels.
- Identifies the top 20 most expressed genes for each infection group.
- Visualizes gene expression using bar plots.
- Compares gene expression variability using Levene’s test.

## Inputs:
- `13Nov24_RT_multi_infection_gene_matrix.txt`: Tab-separated gene expression matrix.
  - Rows: Cells
  - Columns: Genes
  - Values: Expression counts

## Outputs:
- Printed top 20 most expressed genes per infection group.
- Interactive bar plots for the top genes by infection status.
- `results_df_variance_top_genes.csv`: Variance comparison for top genes between single infections and co-infections.

## Dependencies:
- scanpy, numpy, scipy, pandas, seaborn, plotly, matplotlib, sklearn

"""


import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare, mannwhitneyu, levene
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from itertools import combinations
from sklearn.decomposition import PCA
import plotly.express as px

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
# Make metacolumns with the number of phage genes expressed per cell
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

# -------------------- Downstream Analysis: Differential Expression & Correlation -------------------- #
# Define a function to determine infection status
def determine_infection_status(row):
    if (row["luz19_expression"] > 0) and (row["lkd16_expression"] > 0):
        return "coinfected"
    elif row["luz19_expression"] > 0:
        return "luz19_only"
    elif row["lkd16_expression"] > 0:
        return "lkd16_only"
    else:
        return "uninfected"


# Create a new metadata column 'infection_status'
adata.obs["infection_status"] = adata.obs.apply(determine_infection_status, axis=1)
# -------------------- End Differential Expression & Correlation -------------------- #

# -------------------- Top 20 Most Expressed Genes by Infection Status (Printed) -------------------- #
infection_statuses = adata.obs["infection_status"].unique()
for status in infection_statuses:
    print(f"\nTop 20 most expressed genes for '{status}':")
    # Subset the data to the current infection status group
    subset = adata[adata.obs["infection_status"] == status]

    # Compute the average expression per gene across cells in this group
    if sparse.issparse(subset.X):
        avg_expression = np.array(subset.X.mean(axis=0)).flatten()
    else:
        avg_expression = subset.X.mean(axis=0)

    # Create a DataFrame with gene names and their average expression
    df_expression = pd.DataFrame({
        "gene": subset.var_names,
        "avg_expression": avg_expression
    })

    # Sort the DataFrame in descending order and take the top 20 genes
    top20_genes = df_expression.sort_values(by="avg_expression", ascending=False).head(20)
    print(top20_genes)
# -------------------- End Printed Top 20 Analysis -------------------- #

# -------------------- Subplots for Each Group: Top 20 Genes -------------------- #
# Create a subplot for each infection status.
num_groups = len(infection_statuses)
fig = make_subplots(rows=1, cols=num_groups, subplot_titles=infection_statuses)

for i, status in enumerate(infection_statuses, start=1):
    # Subset data for the current group
    subset = adata[adata.obs["infection_status"] == status]

    # Compute average expression per gene
    if sparse.issparse(subset.X):
        avg_expression = np.array(subset.X.mean(axis=0)).flatten()
    else:
        avg_expression = subset.X.mean(axis=0)

    # Create a DataFrame for gene expression
    df_expression = pd.DataFrame({
        "gene": subset.var_names,
        "avg_expression": avg_expression
    })

    # Select top 20 genes
    top20 = df_expression.sort_values(by="avg_expression", ascending=False).head(20)

    # Add bar trace to the subplot for this infection status
    fig.add_trace(
        go.Bar(x=top20["gene"], y=top20["avg_expression"], name=status),
        row=1, col=i
    )
    # Optionally update the axis titles for each subplot
    fig.update_xaxes(title_text="Gene", row=1, col=i)
    fig.update_yaxes(title_text="Avg Expression", row=1, col=i)

fig.update_layout(
    title_text="Top 20 Most Expressed Genes by Infection Status",
    showlegend=False,
    height=500,
    width=300 * num_groups  # Adjust width based on number of groups
)
fig.show()


# -------------------- End Subplots -------------------- #

# -------------------- Compare Variability for Top Genes -------------------- #
# We now focus on the top genes from the "luz19_only" and "lkd16_only" groups.
def get_top_genes(subset, top_n=20):
    if sparse.issparse(subset.X):
        avg_expr = np.array(subset.X.mean(axis=0)).flatten()
    else:
        avg_expr = subset.X.mean(axis=0)
    df = pd.DataFrame({"gene": subset.var_names, "avg_expression": avg_expr})
    top_df = df.sort_values(by="avg_expression", ascending=False).head(top_n)
    return top_df["gene"].tolist()


# Subset for the two groups of interest
subset_luz19 = adata[adata.obs["infection_status"] == "luz19_only"]
subset_lkd16 = adata[adata.obs["infection_status"] == "lkd16_only"]

top_genes_luz19 = get_top_genes(subset_luz19, top_n=20)
top_genes_lkd16 = get_top_genes(subset_lkd16, top_n=20)

# Take the union of top genes from both groups
union_top_genes = sorted(set(top_genes_luz19 + top_genes_lkd16))
print("\nUnion of top genes from 'luz19_only' and 'lkd16_only':", union_top_genes)

# For each gene in the union, compare variance between coinfected and each group using Levene's test.
results = []
for gene in union_top_genes:
    for group in ["luz19_only", "lkd16_only"]:
        # Subset expression values for the gene in each group.
        subset_group = adata[adata.obs["infection_status"] == group]
        subset_coinf = adata[adata.obs["infection_status"] == "coinfected"]

        # Extract gene expression values
        expr_group = subset_group[:, gene].X
        expr_coinf = subset_coinf[:, gene].X

        # Convert to numpy array (handle sparse matrices)
        if sparse.issparse(expr_group):
            expr_group = np.array(expr_group.toarray()).flatten()
        else:
            expr_group = np.array(expr_group).flatten()
        if sparse.issparse(expr_coinf):
            expr_coinf = np.array(expr_coinf.toarray()).flatten()
        else:
            expr_coinf = np.array(expr_coinf).flatten()

        # Calculate sample variances (ddof=1 for sample variance)
        var_group = np.var(expr_group, ddof=1)
        var_coinf = np.var(expr_coinf, ddof=1)

        # Perform Levene's test for equal variances between the two groups
        stat, p_val = levene(expr_group, expr_coinf)

        results.append({
            "gene": gene,
            "comparison": f"{group} vs coinfected",
            f"variance_{group}": var_group,
            "variance_coinfected": var_coinf,
            "levene_stat": stat,
            "p_value": p_val
        })

results_df = pd.DataFrame(results)
print("\nVariance Comparison Between Groups for Top Genes:")
print(results_df)

results_df.to_csv("results_df_variance_top_genes.csv")

# -------------------- End Variability Comparison -------------------- #
