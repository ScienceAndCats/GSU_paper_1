"""
This script performs quality control analysis on single-cell RNA sequencing (scRNA-seq) data
from PETRI-seq by analyzing filtered and mapped UMI files.

Key Features:
- **UMI Count Analysis:** Calculates and visualizes the distribution of UMIs per cell.
- **Mean Reads per Cell:** Computes the average number of reads per cell and displays the distribution.
- **Detected Genes per Cell:** Measures gene diversity per cell and estimates dropout rates.
- **Doublet Detection:** Identifies potential cell doublets based on high UMI counts.
- **Visualization:**
    - Histograms (Frequency & Percentage) for UMI and Gene detection per cell.
    - Violin plots for UMI distribution and mean reads per cell.
- **Customizable Plotting Parameters:** Allows control over bin size, axis limits, and display options.

The script reads two PETRI-seq output files, processes key quality metrics, and generates
various plots to compare two sequencing batches.
"""



import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------
# Parameters for Plotting
# ---------------------------
# Set axis limits and bin ranges for histograms
UMI_X_MAX = 50      # Maximum x-axis value for UMI Count histograms
UMI_Y_MAX = 200     # Maximum y-axis value for UMI Count frequency histograms
UMI_BIN_STEP = 1    # Bin width for UMI Count histograms
UMI_BINS = np.arange(0, UMI_X_MAX + UMI_BIN_STEP, UMI_BIN_STEP)

# Set axis limits and bin ranges for histograms (percentage plots)
UMI_X_MAX_percent = 20     # Maximum x-axis value for UMI Count histograms
UMI_Y_MAX_percent = 100     # Maximum y-axis value for UMI Count frequency histograms
UMI_BIN_STEP_percent = 1    # Bin width for UMI Count histograms
UMI_BINS_percent = np.arange(0, UMI_X_MAX_percent + UMI_BIN_STEP_percent, UMI_BIN_STEP_percent)


# Set stuff for genes freq plot
GENES_X_MAX = 50     # Maximum x-axis value for Detected Genes histograms
GENES_Y_MAX = 100    # Maximum y-axis value for Detected Genes frequency histograms
GENES_BIN_STEP = 1   # Bin width for Detected Genes histograms
GENES_BINS = np.arange(0, GENES_X_MAX + GENES_BIN_STEP, GENES_BIN_STEP)

# Genes percentage plot
GENES_X_MAX_percent = 20      # Maximum x-axis value for GENES Count histograms
GENES_Y_MAX_percent = 100     # Maximum y-axis value for GENES Count frequency histograms
GENES_BIN_STEP_percent = 1    # Bin width for GENES Count histograms
GENES_BINS_percent = np.arange(0, GENES_X_MAX_percent + GENES_BIN_STEP_percent, GENES_BIN_STEP_percent)

# For the Mean Reads violin plot (adjust as needed)
MEAN_READS_Y_MAX = 45

# ---------------------------
# File Paths and Batch Naming
# ---------------------------
batch1_file = r"C:\Users\hwilms2\Desktop\04DPALb\initial_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt"
batch2_file = r"C:\Users\hwilms2\Desktop\DD2PAL\initial2_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt"

# Customize batch names as needed
batch1_name = "04DPALb"
batch2_name = "DD2PAL"

# ---------------------------
# Load Data and Define Columns
# ---------------------------
# Assuming your files have header rows; use header=0 so the header isn't read as data.
columns = ["Cell Barcode", "UMI", "contig:gene", "total_reads"]

batch1 = pd.read_csv(batch1_file, sep="\t", header=0)
batch2 = pd.read_csv(batch2_file, sep="\t", header=0)

# If the file header doesn't match your desired column names, rename them:
batch1.columns = columns
batch2.columns = columns

# Convert "total_reads" to numeric (this will convert any non-numeric values to NaN)
batch1["total_reads"] = pd.to_numeric(batch1["total_reads"], errors="coerce")
batch2["total_reads"] = pd.to_numeric(batch2["total_reads"], errors="coerce")

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
# Total number of unique genes across both batches.
total_genes = len(pd.concat([batch1["contig:gene"], batch2["contig:gene"]]).unique())
# Dropout Rate = fraction of genes not detected in a cell relative to the total genes.
merged_genes["Dropout Rate"] = (total_genes - merged_genes["Detected Genes"]) / total_genes

# ---------------------------
# Estimated Doublet Rate
# ---------------------------
# Heuristic: cells with UMI counts above the 95th percentile are flagged as potential doublets.
doublet_threshold = merged_umi_counts["UMI Count"].quantile(0.95)
merged_umi_counts["Doublet"] = merged_umi_counts["UMI Count"] > doublet_threshold
doublet_rate = merged_umi_counts.groupby("Batch")["Doublet"].mean() * 100  # Percentage

# ---------------------------
# Visualization
# ---------------------------

# UMI Count Histogram (Frequency)
plt.figure(figsize=(10, 5))
ax = sns.histplot(data=merged_umi_counts, x="UMI Count", hue="Batch", bins=UMI_BINS, kde=True, alpha=0.6)
plt.title("UMI Counts Per Cell (Frequency)")
plt.xlabel("UMI Count per Cell")
plt.ylabel("Frequency")
plt.xlim(0, UMI_X_MAX)
plt.ylim(0, UMI_Y_MAX)
plt.legend(title="Batch", labels=[batch1_name, batch2_name])
plt.show()

# UMI Count Histogram (Percentage)
plt.figure(figsize=(10, 5))
ax = sns.histplot(data=merged_umi_counts, x="UMI Count", hue="Batch", bins=UMI_BINS_percent, kde=True, alpha=0.6, stat="percent")
plt.title("UMI Counts Per Cell (Percentage)")
plt.xlabel("UMI Count per Cell")
plt.ylabel("Percentage")
plt.xlim(0, UMI_X_MAX_percent)
plt.ylim(0, UMI_Y_MAX_percent)
plt.legend(title="Batch", labels=[batch1_name, batch2_name])
plt.show()

# UMI Count Violin Plot
plt.figure(figsize=(8, 5))
sns.violinplot(data=merged_umi_counts, x="Batch", y="UMI Count", inner="quartile")
plt.title("UMI Count Distribution Per Cell Between Batches")
plt.xlabel("Batch")
plt.ylabel("UMI Count per Cell")
plt.show()

# Mean Reads per Cell Violin Plot (replacing histogram)
plt.figure(figsize=(10, 5))
sns.violinplot(data=merged_reads, x="Batch", y="Mean Reads", inner="quartile")
plt.title("Mean Reads Per Cell Distribution")
plt.xlabel("Batch")
plt.ylabel("Mean Reads per Cell")
plt.ylim(0, MEAN_READS_Y_MAX)
plt.show()

# Detected Genes Histogram (Frequency)
plt.figure(figsize=(10, 5))
ax = sns.histplot(data=merged_genes, x="Detected Genes", hue="Batch", bins=GENES_BINS, kde=True, alpha=0.6)
plt.title("Detected Genes Per Cell (Frequency)")
plt.xlabel("Detected Genes per Cell")
plt.ylabel("Frequency")
plt.xlim(0, GENES_X_MAX)
plt.ylim(0, GENES_Y_MAX)
plt.legend(title="Batch", labels=[batch1_name, batch2_name])
plt.show()

# Detected Genes Histogram (Percentage)
plt.figure(figsize=(10, 5))
ax = sns.histplot(data=merged_genes, x="Detected Genes", hue="Batch", bins=GENES_BINS_percent, kde=True, alpha=0.6, stat="percent")
plt.title("Detected Genes Per Cell (Percentage)")
plt.xlabel("Detected Genes per Cell")
plt.ylabel("Percentage")
plt.xlim(0, GENES_X_MAX_percent)
plt.ylim(0, GENES_Y_MAX_percent)
plt.legend(title="Batch", labels=[batch1_name, batch2_name])
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
