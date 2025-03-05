import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# -----------------------------------------
# Configuration
# -----------------------------------------

# Path to your second data file
INPUT_FILE = "working_files/preprocessed_PETRI_outputs/PAcontrol_v11_threshold_0_filtered_mapped_UMIs_hans_preprocessed.txt"

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

# -----------------------------------------
# Function to retrieve cell names from the first file
# -----------------------------------------
def get_cell_names(file_path):
    """
    Reads a tab-delimited text file and returns a list of cell names.
    The function automatically determines if the cell name is in the first or second column.

    Parameters
    ----------
    file_path : str
        Path to the tab-delimited text file.

    Returns
    -------
    cell_names : list of str
        A list of cell names extracted from the file.
    """
    cell_names = []
    cell_column = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if cell_column is None:
                # Determine which column is the cell name
                try:
                    _ = int(parts[0])
                    cell_column = 1
                except ValueError:
                    cell_column = 0
            if len(parts) > cell_column:
                cell_names.append(parts[cell_column])
    return cell_names

# -----------------------------------------
# Step 2: Read and preprocess the second data file
# -----------------------------------------
df = pd.read_csv(INPUT_FILE, sep="\t")
df.columns = ["Cell_Barcode", "UMI", "contig_gene", "total_reads"]

# -----------------------------------------
# Step 3: Identify the unique prefix for each UMI row
# -----------------------------------------
def find_prefixes_in_row(contig_genes_str):
    """
    Given a string like:
       "DH5alpha:C1467_RS02470, DH5alpha:C1467_RS06230, MG1655:rrlC"
    Returns a set of distinct prefixes found, out of the predefined list.
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

# Keep only rows where exactly one prefix is found
df = df[df["prefix_set"].apply(lambda s: len(s) == 1)].copy()

# For convenience, create a 'prefix' column with the single prefix value
df["prefix"] = df["prefix_set"].apply(lambda s: list(s)[0])

# -----------------------------------------
# Step 4: Create cell-level metadata with prefix combinations
# -----------------------------------------
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
grouped_combo = cell_meta.groupby("prefix_combination").agg(
    num_cells=("cell_umi_count", "size"),
    total_umis=("cell_umi_count", "sum"),
    mean_umis_per_cell=("cell_umi_count", "mean"),
    median_umis_per_cell=("cell_umi_count", "median"),
    std_umis_per_cell=("cell_umi_count", "std"),
    total_reads=("cell_total_reads", "sum"),
    mean_reads_per_cell=("cell_total_reads", "mean"),
    median_reads_per_cell=("cell_total_reads", "median"),
    std_reads_per_cell=("cell_total_reads", "std")
).reset_index()

print("\nSummary statistics by prefix combination:")
print(grouped_combo.to_string(index=False))

# -----------------------------------------
# Step 6: Plot violin plots for each prefix combination
# -----------------------------------------
unique_combos = sorted(cell_meta["prefix_combination"].unique())
n_combos = len(unique_combos)
data = [cell_meta[cell_meta["prefix_combination"] == combo]["cell_umi_count"] for combo in unique_combos]

fig, ax = plt.subplots(figsize=(10, 6))
parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)

# Customize the violins
for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax.set_xlabel('Prefix Combination')
ax.set_ylabel('UMIs per Cell')
ax.set_title('Distribution of UMIs per Cell by Prefix Combination')
ax.set_xticks(np.arange(1, n_combos + 1))
ax.set_xticklabels(unique_combos, rotation=45, ha='right')

plt.tight_layout()

# Save the figure. Save to the same folder as INPUT_FILE.
folder = os.path.dirname(INPUT_FILE)
base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
output_file = os.path.join(folder, base_name + "_violin_plots.png")
plt.savefig(output_file)
print(f"\nViolin plots saved to {output_file}")

plt.show()
