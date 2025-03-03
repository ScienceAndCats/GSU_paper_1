import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu  # for statistical testing

# ---------------------------
# Parameters for Plotting
# ---------------------------
UMI_X_MAX = 50      # Maximum x-axis value for UMI Count histograms
UMI_Y_MAX = 1000    # Maximum y-axis value for UMI Count frequency histograms
UMI_BIN_STEP = 1    # Bin width for UMI Count histograms
UMI_BINS = np.arange(0, UMI_X_MAX + UMI_BIN_STEP, UMI_BIN_STEP)

UMI_X_MAX_percent = 20
UMI_Y_MAX_percent = 25
UMI_BIN_STEP_percent = 1
UMI_BINS_percent = np.arange(0, UMI_X_MAX_percent + UMI_BIN_STEP_percent, UMI_BIN_STEP_percent)

GENES_X_MAX = 50
GENES_Y_MAX = 1000
GENES_BIN_STEP = 1
GENES_BINS = np.arange(0, GENES_X_MAX + GENES_BIN_STEP, GENES_BIN_STEP)

GENES_X_MAX_percent = 20
GENES_Y_MAX_percent = 25
GENES_BIN_STEP_percent = 1
GENES_BINS_percent = np.arange(0, GENES_X_MAX_percent + GENES_BIN_STEP_percent, GENES_BIN_STEP_percent)

MEAN_READS_Y_MAX = 60

# ---------------------------
# File Paths and Batch Naming
# ---------------------------
batch1_file = r"04DPALb_filtered_mapped_UMIs_multihitcombo.txt"
batch2_file = r"DD2PAL_filtered_mapped_UMIs_multihitcombo.txt"

batch1_name = "04DPALb"
batch2_name = "DD2PAL"

# Define custom colors for each batch
batch_colors = {batch1_name: 'blue', batch2_name: 'red'}

# ---------------------------
# Load Data and Define Columns
# ---------------------------
columns = ["Cell Barcode", "UMI", "contig:gene", "total_reads"]

batch1 = pd.read_csv(batch1_file, sep="\t", header=0)
batch2 = pd.read_csv(batch2_file, sep="\t", header=0)

batch1.columns = columns
batch2.columns = columns

batch1["total_reads"] = pd.to_numeric(batch1["total_reads"], errors="coerce")
batch2["total_reads"] = pd.to_numeric(batch2["total_reads"], errors="coerce")

# ---------------------------
# Filter out cells with less than 5 UMIs
# ---------------------------
batch1 = batch1.groupby("Cell Barcode").filter(lambda x: len(x) >= 5)
batch2 = batch2.groupby("Cell Barcode").filter(lambda x: len(x) >= 5)

# ---------------------------
# UMI Counts per Cell
# ---------------------------
umi_counts_batch1 = batch1.groupby("Cell Barcode")["UMI"].count().reset_index()
umi_counts_batch1.columns = ["Cell Barcode", "UMI Count"]
umi_counts_batch1["Batch"] = batch1_name

umi_counts_batch2 = batch2.groupby("Cell Barcode")["UMI"].count().reset_index()
umi_counts_batch2.columns = ["Cell Barcode", "UMI Count"]
umi_counts_batch2["Batch"] = batch2_name

merged_umi_counts = pd.concat([umi_counts_batch1, umi_counts_batch2])

# ---------------------------
# Mean Reads per Cell (instead of Total Reads)
# ---------------------------
reads_batch1 = batch1.groupby("Cell Barcode")["total_reads"].mean().reset_index()
reads_batch1.columns = ["Cell Barcode", "Mean Reads"]
reads_batch1["Batch"] = batch1_name

reads_batch2 = batch2.groupby("Cell Barcode")["total_reads"].mean().reset_index()
reads_batch2.columns = ["Cell Barcode", "Mean Reads"]
reads_batch2["Batch"] = batch2_name

merged_reads = pd.concat([reads_batch1, reads_batch2])

# ---------------------------
# Detected Genes per Cell
# ---------------------------
genes_batch1 = batch1.groupby("Cell Barcode")["contig:gene"].nunique().reset_index()
genes_batch1.columns = ["Cell Barcode", "Detected Genes"]
genes_batch1["Batch"] = batch1_name

genes_batch2 = batch2.groupby("Cell Barcode")["contig:gene"].nunique().reset_index()
genes_batch2.columns = ["Cell Barcode", "Detected Genes"]
genes_batch2["Batch"] = batch2_name

merged_genes = pd.concat([genes_batch1, genes_batch2])

# ---------------------------
# Gene Dropout Rate per Cell
# ---------------------------
total_genes = len(pd.concat([batch1["contig:gene"], batch2["contig:gene"]]).unique())
merged_genes["Dropout Rate"] = (total_genes - merged_genes["Detected Genes"]) / total_genes

# ---------------------------
# Estimated Doublet Rate
# ---------------------------
doublet_threshold = merged_umi_counts["UMI Count"].quantile(0.95)
merged_umi_counts["Doublet"] = merged_umi_counts["UMI Count"] > doublet_threshold
doublet_rate = merged_umi_counts.groupby("Batch")["Doublet"].mean() * 100  # Percentage

# ---------------------------
# Custom Legend for Histogram Plots
# ---------------------------
custom_legend = [
    Line2D([0], [0], color=batch_colors[batch1_name], lw=4, label=batch1_name),
    Line2D([0], [0], color=batch_colors[batch2_name], lw=4, label=batch2_name)
]

# ---------------------------
# Visualization
# ---------------------------

# UMI Count Histogram (Frequency)
plt.figure(figsize=(10, 5))
ax = sns.histplot(
    data=merged_umi_counts,
    x="UMI Count",
    hue="Batch",
    hue_order=[batch1_name, batch2_name],
    bins=UMI_BINS,
    kde=True,
    alpha=0.6,
    palette=batch_colors
)
plt.title("UMI Counts Per Cell (Frequency)")
plt.xlabel("UMI Count per Cell")
plt.ylabel("Frequency")
plt.xlim(0, UMI_X_MAX)
plt.ylim(0, UMI_Y_MAX)
plt.legend(handles=custom_legend, title="Batch")
plt.show()

# UMI Count Histogram (Percentage)
plt.figure(figsize=(10, 5))
ax = sns.histplot(
    data=merged_umi_counts,
    x="UMI Count",
    hue="Batch",
    hue_order=[batch1_name, batch2_name],
    bins=UMI_BINS_percent,
    kde=True,
    alpha=0.6,
    stat="percent",
    palette=batch_colors
)
plt.title("UMI Counts Per Cell (Percentage)")
plt.xlabel("UMI Count per Cell")
plt.ylabel("Percentage")
plt.xlim(0, UMI_X_MAX_percent)
plt.ylim(0, UMI_Y_MAX_percent)
plt.legend(handles=custom_legend, title="Batch")
plt.show()

# Detected Genes Histogram (Frequency)
plt.figure(figsize=(10, 5))
ax = sns.histplot(
    data=merged_genes,
    x="Detected Genes",
    hue="Batch",
    hue_order=[batch1_name, batch2_name],
    bins=GENES_BINS,
    kde=True,
    alpha=0.6,
    palette=batch_colors
)
plt.title("Detected Genes Per Cell (Frequency)")
plt.xlabel("Detected Genes per Cell")
plt.ylabel("Frequency")
plt.xlim(0, GENES_X_MAX)
plt.ylim(0, GENES_Y_MAX)
plt.legend(handles=custom_legend, title="Batch")
plt.show()

# Detected Genes Histogram (Percentage)
plt.figure(figsize=(10, 5))
ax = sns.histplot(
    data=merged_genes,
    x="Detected Genes",
    hue="Batch",
    hue_order=[batch1_name, batch2_name],
    bins=GENES_BINS_percent,
    kde=True,
    alpha=0.6,
    stat="percent",
    palette=batch_colors
)
plt.title("Detected Genes Per Cell (Percentage)")
plt.xlabel("Detected Genes per Cell")
plt.ylabel("Percentage")
plt.xlim(0, GENES_X_MAX_percent)
plt.ylim(0, GENES_Y_MAX_percent)
plt.legend(handles=custom_legend, title="Batch")
plt.show()

# Mean Reads per Cell Violin Plot
plt.figure(figsize=(10, 5))
sns.violinplot(data=merged_reads, x="Batch", y="Mean Reads", inner="quartile",
               palette=batch_colors)
plt.title("Mean Reads Per Cell Distribution")
plt.xlabel("Batch")
plt.ylabel("Mean Reads per Cell")
plt.ylim(0, MEAN_READS_Y_MAX)
plt.show()

# UMI Count Violin Plot
plt.figure(figsize=(8, 5))
sns.violinplot(data=merged_umi_counts, x="Batch", y="UMI Count", inner="quartile",
               palette=batch_colors)
plt.title("UMI Count Distribution Per Cell Between Batches")
plt.xlabel("Batch")
plt.ylabel("UMI Count per Cell")
plt.show()

# ---------------------------
# Summary Statistics
# ---------------------------
print("UMI Count Summary Stats:")
print(merged_umi_counts.groupby("Batch")["UMI Count"].describe())

print("\nMean Reads Summary Stats:")
print(merged_reads.groupby("Batch")["Mean Reads"].describe())

print("\nDetected Genes Summary Stats:")
print(merged_genes.groupby("Batch")["Detected Genes"].describe())

print("\nEstimated Doublet Rates (%):")
print(doublet_rate)

print("\nGene Dropout Rate (assuming total genes = {}):".format(total_genes))
print(merged_genes.groupby("Batch")["Dropout Rate"].describe())

# ---------------------------
# Statistical Testing
# ---------------------------
# Using Mann-Whitney U test (non-parametric) to compare distributions

# UMI Count per cell
umi_vals_batch1 = merged_umi_counts[merged_umi_counts["Batch"] == batch1_name]["UMI Count"]
umi_vals_batch2 = merged_umi_counts[merged_umi_counts["Batch"] == batch2_name]["UMI Count"]
stat_umi, p_value_umi = mannwhitneyu(umi_vals_batch1, umi_vals_batch2, alternative='two-sided')
print("\nMann-Whitney U Test for UMI Count per Cell:")
print("Test Statistic: {:.3f}, p-value: {:.3e}".format(stat_umi, p_value_umi))

# Detected Genes per cell
genes_vals_batch1 = merged_genes[merged_genes["Batch"] == batch1_name]["Detected Genes"]
genes_vals_batch2 = merged_genes[merged_genes["Batch"] == batch2_name]["Detected Genes"]
stat_genes, p_value_genes = mannwhitneyu(genes_vals_batch1, genes_vals_batch2, alternative='two-sided')
print("\nMann-Whitney U Test for Detected Genes per Cell:")
print("Test Statistic: {:.3f}, p-value: {:.3e}".format(stat_genes, p_value_genes))

# Mean Reads per cell
reads_vals_batch1 = merged_reads[merged_reads["Batch"] == batch1_name]["Mean Reads"]
reads_vals_batch2 = merged_reads[merged_reads["Batch"] == batch2_name]["Mean Reads"]
stat_reads, p_value_reads = mannwhitneyu(reads_vals_batch1, reads_vals_batch2, alternative='two-sided')
print("\nMann-Whitney U Test for Mean Reads per Cell:")
print("Test Statistic: {:.3f}, p-value: {:.3e}".format(stat_reads, p_value_reads))
