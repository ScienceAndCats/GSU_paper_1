import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from itertools import combinations
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
from scipy.spatial.distance import jensenshannon

# -----------------------------------------
# Configuration
# -----------------------------------------
# Paths to your two data files
INPUT_FILE1 = "DD2PAL_filtered_mapped_UMIs_multihitcombo.txt"
INPUT_FILE2 = "04DPALb_filtered_mapped_UMIs_multihitcombo.txt"

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
MIN_UMIS_PER_CELL = 5

# Histogram Parameters
X_AXIS_MAX = 50  # Maximum UMIs per cell to display on the x-axis
Y_AXIS_MAX = 50  # Maximum count of cells to display on the y-axis
BIN_WIDTH = 1  # Bin width for the histogram


# -----------------------------------------
# Functions for processing and comparing data
# -----------------------------------------
def load_and_process(filename):
    """
    Reads the file and processes it:
      - Reads tab-delimited data.
      - Extracts the prefix from each UMI (keeping only rows with exactly one prefix).
      - Filters out cells with fewer than MIN_UMIS_PER_CELL UMIs.
      - Aggregates cell-level metadata with prefix combination, total reads, and UMI count.
    Returns a DataFrame (cell_meta) with columns:
      [Cell_Barcode, prefix_combination, cell_total_reads, cell_umi_count]
    """
    df = pd.read_csv(filename, sep="\t")
    df.columns = ["Cell_Barcode", "UMI", "contig_gene", "total_reads"]

    def find_prefixes_in_row(contig_genes_str):
        genes = [g.strip() for g in contig_genes_str.split(",")]
        found = set()
        for gene in genes:
            for pfx in PREFIXES:
                if gene.startswith(pfx):
                    found.add(pfx)
        return found

    df["prefix_set"] = df["contig_gene"].apply(find_prefixes_in_row)
    # Keep only rows where exactly one prefix is found
    df = df[df["prefix_set"].apply(lambda s: len(s) == 1)].copy()
    df["prefix"] = df["prefix_set"].apply(lambda s: list(s)[0])

    # Filter cells based on total UMIs per cell
    cell_umi_counts = df.groupby("Cell_Barcode")["UMI"].count()
    valid_cells = cell_umi_counts[cell_umi_counts >= MIN_UMIS_PER_CELL].index
    df = df[df["Cell_Barcode"].isin(valid_cells)].copy()

    # Create cell-level metadata:
    cell_meta = df.groupby("Cell_Barcode").agg({
        "prefix": lambda ps: ",".join(sorted(set(ps))),
        "total_reads": "sum",
        "UMI": "count"
    }).reset_index()
    cell_meta.rename(columns={
        "prefix": "prefix_combination",
        "UMI": "cell_umi_count",
        "total_reads": "cell_total_reads"
    }, inplace=True)

    return cell_meta


def bhattacharyya_distance(hist1, hist2):
    """Calculates Bhattacharyya distance between two histograms."""
    return -np.log(np.sum(np.sqrt(hist1 * hist2)) + 1e-12)


# -----------------------------------------
# Process both files
# -----------------------------------------
cell_meta1 = load_and_process(INPUT_FILE1)
cell_meta2 = load_and_process(INPUT_FILE2)

# Get the set of prefix combinations in each file
groups1 = set(cell_meta1["prefix_combination"].unique())
groups2 = set(cell_meta2["prefix_combination"].unique())
common_groups = sorted(list(groups1.intersection(groups2)))

if not common_groups:
    print("No common prefix groups found between the two files.")
    exit()

# -----------------------------------------
# Statistical comparisons for each common group
# -----------------------------------------
bins = np.arange(0, X_AXIS_MAX + BIN_WIDTH, BIN_WIDTH)
stats_list = []

for group in common_groups:
    data1 = cell_meta1[cell_meta1["prefix_combination"] == group]["cell_umi_count"]
    data2 = cell_meta2[cell_meta2["prefix_combination"] == group]["cell_umi_count"]

    # Perform statistical tests only if both groups have data
    if len(data1) == 0 or len(data2) == 0:
        continue

    # KS Test
    ks_stat, ks_p = ks_2samp(data1, data2)
    # Mann-Whitney U Test
    u_stat, u_p = mannwhitneyu(data1, data2, alternative='two-sided')
    # T-Test (Welch's t-test)
    t_stat, t_p = ttest_ind(data1, data2, equal_var=False)

    # Calculate histograms for Jensen-Shannon and Bhattacharyya
    hist1, _ = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bins, density=True)
    jsd = jensenshannon(hist1, hist2)
    b_dist = bhattacharyya_distance(hist1, hist2)

    stats_list.append({
        "prefix_combination": group,
        "KS Stat": ks_stat,
        "KS p-value": ks_p,
        "Mann-Whitney U Stat": u_stat,
        "Mann-Whitney p-value": u_p,
        "T-Test Stat": t_stat,
        "T-Test p-value": t_p,
        "Jensen-Shannon Divergence": jsd,
        "Bhattacharyya Distance": b_dist,
        "n_file1": len(data1),
        "n_file2": len(data2)
    })

stats_df = pd.DataFrame(stats_list)
stats_df = stats_df.sort_values("prefix_combination")
print("\nStatistical Comparison for Common Prefix Groups:")
print(stats_df.to_string(index=False))

# Save the statistical comparison to a CSV file.
comparison_output_file = f"{os.path.splitext(INPUT_FILE1)[0]}_vs_{os.path.splitext(INPUT_FILE2)[0]}_stats.csv"
stats_df.to_csv(comparison_output_file, index=False)
print(f"\nStatistical comparisons saved to {comparison_output_file}")

# -----------------------------------------
# Plot combined histograms for each common group
# -----------------------------------------
n_plots = len(common_groups)
ncols = math.ceil(math.sqrt(n_plots))
nrows = math.ceil(n_plots / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
if n_plots == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for ax, group in zip(axes, common_groups):
    subset1 = cell_meta1[cell_meta1["prefix_combination"] == group]["cell_umi_count"]
    subset2 = cell_meta2[cell_meta2["prefix_combination"] == group]["cell_umi_count"]

    # Plot histograms for file1 and file2 on the same axis
    ax.hist(subset1, bins=bins, color='blue', alpha=0.6, label=os.path.basename(INPUT_FILE1), edgecolor='black')
    ax.hist(subset2, bins=bins, color='orange', alpha=0.6, label=os.path.basename(INPUT_FILE2), edgecolor='black')

    ax.set_title(f"Group: {group}")
    ax.set_xlabel("UMIs per cell")
    ax.set_ylabel("Count of cells")
    ax.set_xlim(0, X_AXIS_MAX)
    ax.set_ylim(0, Y_AXIS_MAX)
    ax.legend()

# Turn off any extra subplots
for ax in axes[len(common_groups):]:
    ax.axis("off")

plt.tight_layout()

# Save the figure. The output filename is based on the two input filenames.
base_name1 = os.path.splitext(os.path.basename(INPUT_FILE1))[0]
base_name2 = os.path.splitext(os.path.basename(INPUT_FILE2))[0]
output_fig_file = f"{base_name1}_vs_{base_name2}_histograms.png"
plt.savefig(output_fig_file)
print(f"\nCombined histograms saved to {output_fig_file}")
plt.show()
