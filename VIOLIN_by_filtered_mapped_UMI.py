import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

# -----------------------------------------
# Configuration
# -----------------------------------------
# Path to your preprocessed UMI file
INPUT_FILE = "working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_Luz19_0-40min_v11_threshold_0_filtered_mapped_UMIs_multihitcombo_preprocessed.txt"

# Path to the bc1 selection pickle file generated from the preprocessing pipeline
BC1_SELECTION_PICKLE = "working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_Luz19_0-40min_v11_threshold_0_filtered_mapped_UMIs_multihitcombo_bc1_selection.pkl"
# NOTE: This will FILTER the cells from BOTH charts! Cell groups from pkl dict to display on violin graph.
# Leave empty if you want to just display everything.
SELECTED_BC1_GROUPS = [
    #"Infected"   # Replace with your desired BC1 group names
    #"Uninfected"

]



# Predefined gene prefixes
PREFIXES = [
    "PA01:",
    "lkd16:",
    "luz19:",
    "14one:",
    "DH5alpha:",
    "MG1655:",
    "pTNS2:",
    "ambiguous"
]

# -----------------------------------------
# Optional: Define which prefix combinations to analyze.

#   SELECTED_PREFIX_COMBINATIONS = [
#       ["PA01:"],
#       ["luz19:", "DH5alpha:"],
#       ["14one:"],
#       ["DH5alpha:"]
#   ]
# [['DH5alpha:'], ['DH5alpha:', 'MG1655:'], ['DH5alpha:', 'MG1655:', 'PA01:'], ['DH5alpha:', 'MG1655:', 'PA01:', 'ambiguous'], ['DH5alpha:', 'MG1655:', 'ambiguous'], ['DH5alpha:', 'PA01:'], ['DH5alpha:', 'PA01:', 'ambiguous'], ['MG1655:'], ['MG1655:', 'PA01:'], ['PA01:'], ['PA01:', 'ambiguous'], ['ambiguous']]
# If this list is empty, all prefix combinations will be analyzed.
# -----------------------------------------
SELECTED_PREFIX_COMBINATIONS = [
    ["PA01:"],
    ["luz19:"],
["PA01:", "luz19:"],
]

# -----------------------------------------
# Function to retrieve cell names from the first file (if needed)
# -----------------------------------------
def get_cell_names(file_path):
    """
    Reads a tab-delimited text file and returns a list of cell names.
    The function automatically determines if the cell name is in the first or second column.
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
                try:
                    _ = int(parts[0])
                    cell_column = 1
                except ValueError:
                    cell_column = 0
            if len(parts) > cell_column:
                cell_names.append(parts[cell_column])
    return cell_names

# -----------------------------------------
# Step 2: Read and preprocess the UMI file
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
# List all prefix combinations found
# -----------------------------------------
all_prefix_combinations = sorted(cell_meta["prefix_combination"].unique())
print("\nAll prefix combinations found (as comma-separated strings):")
print(all_prefix_combinations)
# Convert to a list-of-lists for easy copy/paste
all_prefix_combinations_list = [combo.split(",") for combo in all_prefix_combinations]
print("\nAll prefix combinations found (as a list-of-lists):")
print(all_prefix_combinations_list)

# -----------------------------------------
# Optional Filtering: Only analyze selected prefix combinations
# -----------------------------------------
if SELECTED_PREFIX_COMBINATIONS:
    # Convert each list of prefixes to a sorted, comma-separated string
    selected_prefix_strings = [",".join(sorted(combo)) for combo in SELECTED_PREFIX_COMBINATIONS]
    print("\nSelected prefix combinations for analysis:", selected_prefix_strings)
    cell_meta = cell_meta[cell_meta["prefix_combination"].isin(selected_prefix_strings)]
else:
    print("\nNo selected prefix combinations provided. Analyzing all prefix combinations.")

# -----------------------------------------
# Step 4b: Load bc1 selection pickle and add BC1 group info to cell_meta
# -----------------------------------------
with open(BC1_SELECTION_PICKLE, "rb") as f:
    bc1_dict = pickle.load(f)

# Invert the bc1_dict to create a mapping: cell barcode -> BC1 group
cell_to_bc1 = {}
for group, barcode_list in bc1_dict.items():
    for barcode in barcode_list:
        cell_to_bc1[barcode] = group

# Add a new column to cell_meta for the BC1 group; label missing entries as 'Unassigned'
cell_meta["bc1_group"] = cell_meta["Cell_Barcode"].map(cell_to_bc1).fillna("Unassigned")

# -----------------------------------------
# Optional Filtering: Only analyze selected BC1 groups
# -----------------------------------------


if SELECTED_BC1_GROUPS:
    print("\nSelected BC1 groups for analysis:", SELECTED_BC1_GROUPS)
    cell_meta = cell_meta[cell_meta["bc1_group"].isin(SELECTED_BC1_GROUPS)]
else:
    print("\nNo selected BC1 groups provided. Analyzing all BC1 groups.")

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
data_prefix = [cell_meta[cell_meta["prefix_combination"] == combo]["cell_umi_count"] for combo in unique_combos]

fig1, ax1 = plt.subplots(figsize=(10, 6))
parts1 = ax1.violinplot(data_prefix, showmeans=False, showmedians=True, showextrema=True)

# Customize the violins for prefix combination plot
for pc in parts1['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax1.set_xlabel('Prefix Combination')
ax1.set_ylabel('UMIs per Cell')
ax1.set_title('Distribution of UMIs per Cell by Prefix Combination')
ax1.set_xticks(np.arange(1, n_combos + 1))
ax1.set_xticklabels(unique_combos, rotation=45, ha='right')

plt.tight_layout()

# Save the prefix combination violin plot.
folder = os.path.dirname(INPUT_FILE)
base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
output_file_prefix = os.path.join(folder, base_name + "_violin_plots_prefix.png")
plt.savefig(output_file_prefix)
print(f"\nViolin plot by prefix combination saved to {output_file_prefix}")

# -----------------------------------------
# Step 7: Compute summary statistics for BC1 grouping
# -----------------------------------------
grouped_bc1 = cell_meta.groupby("bc1_group").agg(
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

print("\nSummary statistics by BC1 grouping:")
print(grouped_bc1.to_string(index=False))

# -----------------------------------------
# Step 8: Plot violin plots for BC1 grouping
# -----------------------------------------
unique_bc1 = sorted(cell_meta["bc1_group"].unique())
n_bc1 = len(unique_bc1)
data_bc1 = [cell_meta[cell_meta["bc1_group"] == grp]["cell_umi_count"] for grp in unique_bc1]

fig2, ax2 = plt.subplots(figsize=(10, 6))
parts2 = ax2.violinplot(data_bc1, showmeans=False, showmedians=True, showextrema=True)

# Customize the violins for BC1 grouping plot
for pc in parts2['bodies']:
    pc.set_facecolor('#4C72B0')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax2.set_xlabel('BC1 Group')
ax2.set_ylabel('UMIs per Cell')
ax2.set_title('Distribution of UMIs per Cell by BC1 Group')
ax2.set_xticks(np.arange(1, n_bc1 + 1))
ax2.set_xticklabels(unique_bc1, rotation=45, ha='right')

plt.tight_layout()

# Save the BC1 grouping violin plot.
output_file_bc1 = os.path.join(folder, base_name + "_violin_plots_bc1.png")
plt.savefig(output_file_bc1)
print(f"\nViolin plot by BC1 grouping saved to {output_file_bc1}")

plt.show()
