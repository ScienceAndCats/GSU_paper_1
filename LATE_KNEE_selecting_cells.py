"""
SUMMARY:
This script performs the following steps:
1. Reads a first tab-delimited file (up to N lines, where N is ROW_LIMIT) to get a list of cell names.
   - The script automatically determines if the cell name is in the first or second column.
2. Reads and preprocesses a second tab-delimited file containing cell barcode, UMI, gene, and read count data.
3. Filters the second dataset to keep only the rows for cells present in the cell list.
4. Optionally filters out rows where the gene name contains a comma.
5. Identifies which predefined gene prefix each UMI row belongs to and keeps only rows with exactly one prefix.
6. Filters out cells having fewer than the defined minimum UMIs (MIN_UMIS_PER_CELL).
7. Aggregates cell-level metadata, computes summary statistics by prefix combination, and
   plots a violin plot showing the distribution of UMIs per cell.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# -----------------------------------------
# Configuration
# -----------------------------------------
# How many lines to read from the first file
ROW_LIMIT = 2000

# Path to your first file (for cell list)
PATH_TO_CELL_LIST_FILE = "data/DD2PAL_newPETRI_selected_cumulative_frequency_table.txt" # 17Sep2024_luz19_20min_selected_cumulative_frequency_table

# Path to your second data file
INPUT_FILE = "data/switched.txt" # 17Sep2024_luz19_20min_filtered_mapped_UMIs_hans

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

# Option: Filter out rows with a comma in the gene name (set to False to turn off the filter)
FILTER_GENES_WITH_COMMA = True


# -----------------------------------------
# Function to retrieve cell names from the first file
# -----------------------------------------
def get_cell_names(file_path, row_limit=4000):
    """
    Reads up to 'row_limit' lines of a tab-delimited text file
    and returns a list of cell names. This handles two formats:
      1) The first column is numeric, and the second column is the cell name.
      2) The first column itself is the cell name.

    Parameters
    ----------
    file_path : str
        Path to the tab-delimited text file.
    row_limit : int
        Maximum number of lines to read.

    Returns
    -------
    cell_names : list of str
        A list of cell names extracted from the file.
    """
    cell_names = []
    with open(file_path, 'r') as f:
        # Detect the column of the cell name from the first line
        first_line = f.readline().strip()
        if not first_line:
            # The file is empty
            return []

        parts = first_line.split('\t')
        # Decide which column is the cell name
        # Try to interpret parts[0] as an integer
        try:
            _ = int(parts[0])
            # If this succeeds, column 2 (parts[1]) is the cell name
            cell_column = 1
        except ValueError:
            # If this fails, the first token is presumably the cell name
            cell_column = 0

        # Store the cell name from the first line (if it exists)
        if len(parts) > cell_column:
            cell_names.append(parts[cell_column])

        # Now read the remaining (row_limit - 1) lines
        lines_to_read = row_limit - 1
        for _ in range(lines_to_read):
            line = f.readline().strip()
            if not line:
                # End if the file has fewer lines than row_limit
                break
            parts = line.split('\t')
            if len(parts) > cell_column:
                cell_names.append(parts[cell_column])

    return cell_names


# -----------------------------------------
# Step 1: Read cell list from the first file
# -----------------------------------------
cell_list = get_cell_names(PATH_TO_CELL_LIST_FILE, row_limit=ROW_LIMIT)
print(f"Loaded {len(cell_list)} cell names from {PATH_TO_CELL_LIST_FILE}")

# -----------------------------------------
# Step 2: Read and preprocess the second data file
# -----------------------------------------
df = pd.read_csv(INPUT_FILE, sep="\t")
df.columns = ["Cell_Barcode", "UMI", "contig_gene", "total_reads"]

# Filter the second dataset to keep only the cells present in cell_list
initial_count = len(df)
df = df[df["Cell_Barcode"].isin(cell_list)].copy()
filtered_count = len(df)
print(f"Filtered second dataset from {initial_count} to {filtered_count} rows using cell list.")

# If no rows remain, stop here
if df.empty:
    print("No matching cells found in the second dataset based on the provided cell list. Exiting.")
    sys.exit(0)

# -----------------------------------------
# (Optional) Filter out rows where the contig_gene column contains a comma
# -----------------------------------------
if FILTER_GENES_WITH_COMMA:
    # Extract the rows that would be filtered out
    filtered_df = df[df["contig_gene"].str.contains(",")].copy()
    filtered_genes = filtered_df["contig_gene"].unique().tolist()

    if len(filtered_genes) > 0:
        print("\nList of filtered genes (containing a comma):")
        for gene in filtered_genes:
            print(gene)

    # Remove those rows from the main DataFrame
    df = df[~df["contig_gene"].str.contains(",")].copy()

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
# Step 4: Filter out cells with low total UMIs
# -----------------------------------------
cell_umi_counts = df.groupby("Cell_Barcode")["UMI"].count()
valid_cells = cell_umi_counts[cell_umi_counts >= MIN_UMIS_PER_CELL].index
df = df[df["Cell_Barcode"].isin(valid_cells)].copy()

# -----------------------------------------
# Step 5: Create cell-level metadata with prefix combinations
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
# Step 6: Compute summary statistics for each prefix combination
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
# Step 7: Plot violin plots for each prefix combination
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

# Save the figure. The output file name is derived from INPUT_FILE.
base_name = os.path.splitext(INPUT_FILE)[0]
output_file = base_name + "_violin_plots.png"
# plt.savefig(output_file)
print(f"\nViolin plots saved to {output_file}")

plt.show()
