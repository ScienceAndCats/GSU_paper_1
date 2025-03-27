"""
NOTE: This ONLY works for csv files derived from selecting subsets of the UMAP graphs from other script in this.
Script for comparing multiple single-cell RNA-seq datasets by analyzing UMI counts and gene expression FROM UMAP GRAPH
SUBSETS.

## Functionality:
- Loads multiple CSV files containing gene expression data into AnnData objects.
- Computes basic statistics for each dataset, including:
  - Total cells and genes.
  - Mean and median UMI counts per cell.
  - Top 15 most expressed genes.
- Saves computed statistics as CSV files for further analysis.
- Generates two key visualizations:
  1. **Clustered UMI Count Distribution** (histogram comparing cell distributions across datasets).
  2. **Top 15 Expressed Genes** (bar plots per dataset).

## Inputs:
- CSV files containing single-cell gene expression data.
  - Each file should have genes as columns and cells as rows.

## Outputs:
- CSV files:
  - `top_15_genes_<dataset>.csv`: Mean expression of top 15 genes per dataset.
  - `cell_umi_counts_<dataset>.csv`: UMI counts per cell.
- Plots:
  - Histogram comparing UMI count distributions across datasets.
  - Bar plots showing the top 15 expressed genes in each dataset.

## Dependencies:
- scanpy, numpy, scipy, pandas, seaborn, matplotlib
"""


import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths for comparison (update these paths as needed)
file_path_1 = "selected_points (26).csv"
file_path_2 = "selected_points (27).csv"
file_path_3 = "selected_points (28).csv"  # New CSV file
file_path_4 = "selected_points (xx).csv"  # New CSV file

def load_data(file_path):
    """Loads a CSV file into an AnnData object."""
    df = pd.read_csv(file_path, index_col=0)
    adata = sc.AnnData(df)
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
# Slice the list to match the number of datasets loaded
hex_colors = default_hex_colors[:len(datasets)]

# ------------------------------
# Compute statistics and save CSVs for each dataset
for adata, label in zip(datasets, labels):
    # Compute mean expression per gene (across all cells)
    adata.var["mean_expression_per_gene"] = np.mean(adata.X, axis=0)
    # Compute UMI count per cell
    adata.obs["UMI_count"] = np.sum(adata.X, axis=1)

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
# Figure 1: Clustered UMI Count Distribution as Percentage for all datasets
plt.figure(figsize=(12, 6))

# ---- Adjustable Histogram Parameters ----
hist_min = 0    # Minimum value for the histogram bins
hist_max = 30   # Maximum value for the histogram bins
bin_size = 1    # Size (width) of each histogram bin

# Create bin edges and centers based on the specified bin size
bins = np.arange(hist_min, hist_max + bin_size, bin_size)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Calculate the bar width for each dataset's cluster (80% of bin width divided by number of datasets)
bar_width = (bins[1] - bins[0]) * 0.8 / len(datasets)

# Compute and plot histogram data for each dataset in clusters per bin
for i, (adata, label) in enumerate(zip(datasets, labels)):
    counts, _ = np.histogram(adata.obs["UMI_count"], bins=bins)
    percentages = (counts / adata.n_obs) * 100  # Convert counts to percentage
    offset = (i - (len(datasets) - 1) / 2) * bar_width
    plt.bar(bin_centers + offset, percentages, width=bar_width, label=label,
            color=hex_colors[i], align='center')

plt.xlabel("UMI Count per Cell")
plt.ylabel("Percentage of Cells")
plt.title("Clustered UMI Count Distribution as Percentage")
plt.legend()
plt.xlim(hist_min, hist_max)
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
