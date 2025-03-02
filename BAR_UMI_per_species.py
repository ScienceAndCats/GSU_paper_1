import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# -----------------------------------------
# Configuration
# -----------------------------------------
# Path to your data file
INPUT_FILE = "17Sep24_Luz19_20min_filtered_mapped_UMIs_hans.txt"

# Predefined gene prefixes
PREFIXES = [
    "PA01:",
    "lkd16:",
    "luz19:",
    "14one:",
    "DH5alpha:",
    "MG1655:",
    "pTNS2:"
]

# Filter: Minimum UMIs per cell to include in analysis
MIN_UMIS_PER_CELL = 1

# Histogram Parameters
X_AXIS_MAX = 50  # Maximum UMIs per cell to display on the x-axis
Y_AXIS_MAX = 30  # Maximum count of cells to display on the y-axis
BIN_WIDTH = 1  # Bin width for the histogram

# -----------------------------------------
# Step 1: Read and preprocess the data
# -----------------------------------------
# Assumes a tab-delimited file
df = pd.read_csv(INPUT_FILE, sep="\t")
df.columns = ["Cell_Barcode", "UMI", "contig_gene", "total_reads"]


# -----------------------------------------
# Step 2: Identify the unique prefix for each UMI row
# -----------------------------------------
def find_prefixes_in_row(contig_genes_str):
    """
    Given a string like:
       "DH5alpha:C1467_RS02470, DH5alpha:C1467_RS06230, MG1655:rrlC"
    Returns a set of distinct prefixes found.
    """
    genes = [g.strip() for g in contig_genes_str.split(",")]
    found = set()
    for gene in genes:
        for pfx in PREFIXES:
            if gene.startswith(pfx):
                found.add(pfx)
    return found


# Create a new column with the set of prefixes for each row
df["prefix_set"] = df["contig_gene"].apply(find_prefixes_in_row)

# Keep only rows where exactly one prefix is found (per UMI)
df = df[df["prefix_set"].apply(lambda s: len(s) == 1)].copy()

# For convenience, create a 'prefix' column with the single prefix value
df["prefix"] = df["prefix_set"].apply(lambda s: list(s)[0])

# -----------------------------------------
# Step 3: Filter out cells with low total UMIs
# -----------------------------------------
# Count UMIs per cell (each row represents one UMI)
cell_umi_counts = df.groupby("Cell_Barcode")["UMI"].count()
valid_cells = cell_umi_counts[cell_umi_counts >= MIN_UMIS_PER_CELL].index
df = df[df["Cell_Barcode"].isin(valid_cells)].copy()

# -----------------------------------------
# Step 4: Create cell-level metadata with prefix combinations
# -----------------------------------------
# For each cell, determine the sorted, unique combination of prefixes (as a comma-separated string)
cell_meta = df.groupby("Cell_Barcode").agg({
    "prefix": lambda ps: ",".join(sorted(set(ps))),
    "total_reads": "sum",
    "UMI": "count"  # Count of UMIs per cell
}).reset_index()

cell_meta.rename(columns={
    "prefix": "prefix_combination",
    "UMI": "cell_umi_count",
    "total_reads": "cell_total_reads"
}, inplace=True)

# -----------------------------------------
# Step 5: Compute summary statistics for each prefix combination
# -----------------------------------------
grouped_combo = cell_meta.groupby("prefix_combination")
summary_list = []

for combo, group in grouped_combo:
    num_cells = len(group)
    total_umis = group["cell_umi_count"].sum()
    total_reads = group["cell_total_reads"].sum()
    mean_umis = total_umis / num_cells if num_cells > 0 else 0
    mean_reads = total_reads / num_cells if num_cells > 0 else 0

    summary_list.append({
        "prefix_combination": combo,
        "num_cells": num_cells,
        "total_umis": total_umis,
        "mean_umis_per_cell": mean_umis,
        "total_reads": total_reads,
        "mean_reads_per_cell": mean_reads
    })

summary_df = pd.DataFrame(summary_list).sort_values("prefix_combination")
print("\nSummary statistics by prefix combination:")
print(summary_df.to_string(index=False))

# -----------------------------------------
# Step 6: Plot histograms for each prefix combination in a single figure
# -----------------------------------------
# Define bins based on BIN_WIDTH and X_AXIS_MAX
bins = np.arange(0, X_AXIS_MAX + BIN_WIDTH, BIN_WIDTH)

# Get the unique prefix combinations
unique_combos = sorted(cell_meta["prefix_combination"].unique())
n_plots = len(unique_combos)
# Define grid: try to make it as square as possible
ncols = math.ceil(math.sqrt(n_plots))
nrows = math.ceil(n_plots / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
# Ensure axes is a flat array for easy iteration
if n_plots == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for ax, combo in zip(axes, unique_combos):
    subset = cell_meta[cell_meta["prefix_combination"] == combo]
    ax.hist(subset["cell_umi_count"], bins=bins, color='blue', alpha=0.7, edgecolor='black')
    ax.set_title(f"Combination: {combo}")
    ax.set_xlabel("UMIs per cell")
    ax.set_ylabel("Count of cells")
    ax.set_xlim(0, X_AXIS_MAX)
    ax.set_ylim(0, Y_AXIS_MAX)

# Turn off any extra subplots if the grid is larger than needed
for ax in axes[len(unique_combos):]:
    ax.axis("off")

plt.tight_layout()

# Save the figure. The output file name is derived from INPUT_FILE.
base_name = os.path.splitext(INPUT_FILE)[0]
output_file = base_name + "_histograms.png"
plt.savefig(output_file)
print(f"\nHistograms saved to {output_file}")

plt.show()
