"""
Script for comparing multiple single-cell RNA-seq datasets by analyzing UMI counts and gene expression FROM UMAP GRAPH SUBSETS.
NOTE: This ONLY works for h5ad files derived from selecting subsets of the UMAP graphs from another script.

## Functionality:
- Loads multiple h5ad files containing gene expression data into AnnData objects.
- Computes basic statistics for each dataset, including:
  - Total cells and genes.
  - Mean and median UMI counts per cell.
  - Top 15 most expressed genes.
- Saves computed statistics as CSV files for further analysis.
- Generates two key visualizations:
  1. **Violin Plot for UMI Count Distribution** (comparing UMI count distributions per cell across datasets).
  2. **Top 15 Expressed Genes** (bar plots per dataset).

## Inputs:
- h5ad files containing single-cell gene expression data.
  - Each file should have genes as columns and cells as rows.

## Outputs:
- CSV files:
  - `top_15_genes_<dataset>.csv`: Mean expression of top 15 genes per dataset.
  - `cell_umi_counts_<dataset>.csv`: UMI counts per cell.
- Plots:
  - Violin plot comparing UMI count distributions across datasets.
  - Bar plots showing the top 15 expressed genes in each dataset.

## Dependencies:
- scanpy, numpy, scipy, pandas, seaborn, matplotlib
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import seaborn as sns

# File paths for comparison (update these paths as needed)
file_path_1 = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_5min_infected.h5ad"
file_path_2 = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_10min_infected.h5ad"
file_path_3 = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_15min_infected.h5ad"  # New h5ad file
file_path_4 = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_20min_infected.h5ad"  # New h5ad file

def load_data(file_path):
    """Loads an h5ad file into an AnnData object."""
    adata = sc.read(file_path)
    return adata

# Collect file paths into a list
all_file_paths = [file_path_1, file_path_2, file_path_3, file_path_4]

# Try loading each file; if a file is missing, skip it
datasets = []
labels = []
for i, path in enumerate(all_file_paths, start=1):
    if path and os.path.exists(path):
        try:
            adata = load_data(path)
            datasets.append(adata)
            labels.append(f"Dataset {i}")
            print(f"Loaded {path} as Dataset {i}")
        except Exception as e:
            print(f"Error loading file {path}: {e}")
    else:
        print(f"File {path} does not exist. Skipping.")

# Exit if no datasets could be loaded
if not datasets:
    print("No datasets loaded. Exiting.")
    exit()

# Define hex colors for each dataset (adjust as needed)
default_hex_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
hex_colors = default_hex_colors[:len(datasets)]

# ------------------------------
# Compute statistics and save CSVs for each dataset
for adata, label in zip(datasets, labels):
    # Compute mean expression per gene (across all cells)
    if sparse.issparse(adata.X):
        mean_expr = np.array(adata.X.mean(axis=0)).flatten()
    else:
        mean_expr = np.mean(adata.X, axis=0)
    adata.var["mean_expression_per_gene"] = mean_expr

    # Compute UMI count per cell
    if sparse.issparse(adata.X):
        umi_count = np.array(adata.X.sum(axis=1)).flatten()
    else:
        umi_count = np.sum(adata.X, axis=1)
    adata.obs["UMI_count"] = umi_count

    print(f"\nGeneral Statistics for {label}:")
    print(f"Total Cells: {adata.n_obs}")
    print(f"Total Genes: {adata.n_vars}")
    print(f"Mean UMI Count per Cell: {adata.obs['UMI_count'].mean():.2f}")
    print(f"Median UMI Count per Cell: {adata.obs['UMI_count'].median():.2f}")
    print(f"Total UMI Count: {adata.obs['UMI_count'].sum()}")

# Identify the top 15 most expressed genes based on mean expression per gene
top_genes = {}
for adata, label in zip(datasets, labels):
    top_df = adata.var.sort_values("mean_expression_per_gene", ascending=False).head(15)
    top_genes[label] = top_df
    # Save the results to CSV files
    csv_top = f"top_15_genes_{label.replace(' ', '').lower()}.csv"
    top_df.to_csv(csv_top)
    csv_umi = f"cell_umi_counts_{label.replace(' ', '').lower()}.csv"
    adata.obs.to_csv(csv_umi)
    print(f"Saved {csv_top} and {csv_umi}")

# ------------------------------
# Figure 1: UMI Count Distribution per Cell (Violin Plot)
# Combine UMI counts from all datasets into one DataFrame for plotting.
all_data = []
for adata, label in zip(datasets, labels):
    df = pd.DataFrame({'UMI_count': adata.obs['UMI_count']})
    df['Dataset'] = label
    all_data.append(df)
all_data = pd.concat(all_data, ignore_index=True)

plt.figure(figsize=(10, 6))
sns.violinplot(x='Dataset', y='UMI_count', data=all_data, palette=hex_colors)
plt.title("UMI Count Distribution per Cell")
plt.xlabel("Dataset")
plt.ylabel("UMI Count per Cell")
plt.tight_layout()
plt.show()

# ------------------------------
# Figure 2: Top 15 Expressed Genes for each dataset
n_datasets = len(datasets)
# Dynamically determine grid layout: use 1 column if only one dataset,
# otherwise use 2 columns and enough rows to accommodate all datasets.
if n_datasets == 1:
    rows, cols = 1, 1
else:
    cols = 2
    rows = int(np.ceil(n_datasets / cols))

fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
if n_datasets == 1:
    axs = [axs]  # Make it iterable if only one axis exists
else:
    axs = axs.flatten()

# Plot each dataset's top 15 expressed genes
for ax, (label, top_df), color in zip(axs, top_genes.items(), hex_colors):
    genes = top_df.index
    expressions = top_df["mean_expression_per_gene"].values
    ax.bar(range(len(genes)), expressions, color=color, alpha=0.7)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Mean Expression per Gene")
    ax.set_title(f"Top 15 Expressed Genes in {label}")

# Hide any extra subplots if the grid is larger than needed
for ax in axs[len(top_genes):]:
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Analysis complete. Results saved as CSV files for each dataset.")
