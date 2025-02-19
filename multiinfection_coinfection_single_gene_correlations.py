"""
Script for analyzing single-cell RNA-seq data from phage-infected bacterial cells.

## Functionality:
- Loads and filters gene expression data.
- Identifies and quantifies phage gene expression per cell.
- Determines infection status based on phage expression levels.
- Identifies the top 20 most expressed genes for each infection group.
- Compares gene expression variability using Levene’s test.
- Computes Jaccard similarity for overlap of top variable genes across infection groups.
- Performs PCA on the most variable genes and visualizes the clustering.

## Inputs:
- `13Nov24_RT_multi_infection_gene_matrix.txt`: Tab-separated gene expression matrix.
  - Rows: Cells
  - Columns: Genes
  - Values: Expression counts

## Outputs:
- Printed top 20 most expressed genes per infection group.
- Interactive bar plots for the top genes by infection status.
- Levene’s test results for variance comparison across groups.
- Jaccard similarity index for gene overlap between groups.
- PCA scatter plot of most variable genes.
- `results_df_variance_top_genes.csv`: Variance comparison for top genes between single infections and co-infections.

## Dependencies:
- scanpy, numpy, scipy, pandas, seaborn, plotly, matplotlib, sklearn
"""


import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare, mannwhitneyu
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

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


from scipy.stats import levene, f_oneway
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
import plotly.express as px

# -------------------- Compute Gene Variability Per Infection Group -------------------- #
infection_statuses = adata.obs["infection_status"].unique()
gene_variance_dict = {}

for status in infection_statuses:
    subset = adata[adata.obs["infection_status"] == status]

    # Compute variance for each gene
    if sparse.issparse(subset.X):
        variance = np.array(subset.X.toarray().var(axis=0))
    else:
        variance = subset.X.var(axis=0)

    # Store as DataFrame
    gene_variance_dict[status] = pd.DataFrame({
        "gene": subset.var_names,
        "variance": variance
    }).sort_values(by="variance", ascending=False).head(100)  # Top 100 most variable genes

# -------------------- Compare Variability Using Levene's Test -------------------- #
gene_var_data = {status: df.set_index("gene")["variance"] for status, df in gene_variance_dict.items()}
gene_var_df = pd.DataFrame(gene_var_data).dropna()  # Drop genes not found in all groups

# Perform Levene's test across all groups
levene_stat, levene_p = levene(*[gene_var_df[col] for col in gene_var_df.columns])
print(f"Levene’s test for homogeneity of variance: p = {levene_p:.5f}")

# -------------------- Jaccard Similarity for Overlap of Top 100 Genes -------------------- #
jaccard_results = {}
for (status1, status2) in combinations(infection_statuses, 2):
    genes1 = set(gene_variance_dict[status1]["gene"])
    genes2 = set(gene_variance_dict[status2]["gene"])

    jaccard_index = len(genes1.intersection(genes2)) / len(genes1.union(genes2))
    jaccard_results[(status1, status2)] = jaccard_index

print("\nJaccard Similarity Index between groups:")
for pair, score in jaccard_results.items():
    print(f"{pair}: {score:.3f}")

# -------------------- PCA Visualization of Most Variable Genes -------------------- #
top_var_genes = list(set().union(*[df["gene"] for df in gene_variance_dict.values()]))  # Unique top variable genes
adata_subset = adata[:, top_var_genes]  # Subset only these genes

# Extract expression matrix
if sparse.issparse(adata_subset.X):
    expression_matrix = adata_subset.X.toarray()
else:
    expression_matrix = adata_subset.X

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(expression_matrix)

# Convert to DataFrame
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["infection_status"] = adata.obs["infection_status"].values

# Plot PCA
fig = px.scatter(pca_df, x="PC1", y="PC2", color="infection_status",
                 title="PCA of Most Variable Genes by Infection Status")
fig.show()
