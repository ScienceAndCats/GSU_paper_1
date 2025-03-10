"""
NOTE: THIS COUNTS THE TOTAL READS INCORRECTLY!!!
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# -----------------------------------------
# Configuration
# -----------------------------------------
# Path to your data file
INPUT_FILE = "data/18Nov2024_PAcontroltab_NH_filtered_mapped_UMIs_hans.txt"

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
BIN_WIDTH = 1    # Bin width for the histogram

# (New) How many cells to pick from each group
MG1655_SAMPLE_SIZE = 500   # <--- user-specified
PA01_SAMPLE_SIZE = 500     # <--- user-specified

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
# For each cell, determine the sorted, unique combination of prefixes
# (as a comma-separated string)
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
# NEW STEP 5: Randomly pick cells that express ONLY MG1655 or ONLY PA01
# -----------------------------------------
# 1) Identify cells that express ONLY "MG1655:" (and not "PA01:")
mg1655_cells = cell_meta[
    (cell_meta["prefix_combination"].str.contains("MG1655:")) &
    (~cell_meta["prefix_combination"].str.contains("PA01:"))
]

# 2) Identify cells that express ONLY "PA01:" (and not "MG1655:")
pa01_cells = cell_meta[
    (cell_meta["prefix_combination"].str.contains("PA01:")) &
    (~cell_meta["prefix_combination"].str.contains("MG1655:"))
]

# 3) Randomly sample the user-specified number of cells from each group
if len(mg1655_cells) < MG1655_SAMPLE_SIZE:
    print(f"Warning: Only {len(mg1655_cells)} MG1655-only cells available, sampling all.")
    mg1655_sample_size = len(mg1655_cells)
else:
    mg1655_sample_size = MG1655_SAMPLE_SIZE

if len(pa01_cells) < PA01_SAMPLE_SIZE:
    print(f"Warning: Only {len(pa01_cells)} PA01-only cells available, sampling all.")
    pa01_sample_size = len(pa01_cells)
else:
    pa01_sample_size = PA01_SAMPLE_SIZE

mg1655_sampled = mg1655_cells.sample(n=mg1655_sample_size, random_state=42)
pa01_sampled = pa01_cells.sample(n=pa01_sample_size, random_state=42)

# 4) Combine these two sets of cells (they're mutually exclusive now)
chosen_cells = pd.concat([mg1655_sampled, pa01_sampled], ignore_index=True)
chosen_cells.drop_duplicates(subset=["Cell_Barcode"], inplace=True)

# 5) Filter both df and cell_meta to include only the chosen cells
sampled_barcodes = chosen_cells["Cell_Barcode"]
df = df[df["Cell_Barcode"].isin(sampled_barcodes)].copy()
cell_meta = cell_meta[cell_meta["Cell_Barcode"].isin(sampled_barcodes)].copy()


# -----------------------------------------
# (Renumbered) Step 6: Compute summary stats
#              for each prefix combination
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
print("\nSummary statistics by prefix combination (SAMPLED CELLS):")
print(summary_df.to_string(index=False))

# -----------------------------------------
# (Renumbered) Step 7: Plot histograms
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
