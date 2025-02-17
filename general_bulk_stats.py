"""This is for when you download groups and want to explore what is different about them"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "selected_points.csv"  # Adjust path if needed
df = pd.read_csv(file_path, index_col=0)

# Convert to AnnData object
adata = sc.AnnData(df)

# Compute general statistics
adata.var["mean_expression"] = np.mean(adata.X, axis=0)  # Mean expression per gene
adata.var["total_expression"] = np.sum(adata.X, axis=0)  # Total expression per gene
adata.obs["UMI_count"] = np.sum(adata.X, axis=1)  # UMI count per cell

# Get top 15 most expressed genes
top_genes = adata.var.sort_values("total_expression", ascending=False).head(15)
print("Top 15 Most Expressed Genes:")
print(top_genes[["mean_expression", "total_expression"]])

# Summary statistics
summary_stats = {
    "Total Cells": adata.n_obs,
    "Total Genes": adata.n_vars,
    "Mean UMI Count per Cell": adata.obs["UMI_count"].mean(),
    "Median UMI Count per Cell": adata.obs["UMI_count"].median(),
    "Total UMI Count": adata.obs["UMI_count"].sum(),
}

print("\nGeneral Statistics:")
for k, v in summary_stats.items():
    print(f"{k}: {v}")

# Plot distributions
plt.figure(figsize=(12, 4))

# UMI Count Distribution
plt.subplot(1, 2, 1)
plt.hist(adata.obs["UMI_count"], bins=50, color='blue', alpha=0.7)
plt.xlabel("UMI Count per Cell")
plt.ylabel("Frequency")
plt.title("UMI Count Distribution")

# Top 15 genes bar plot
plt.subplot(1, 2, 2)
plt.bar(top_genes.index, top_genes["total_expression"], color='green', alpha=0.7)
plt.xticks(rotation=90)
plt.xlabel("Gene")
plt.ylabel("Total Expression")
plt.title("Top 15 Most Expressed Genes")

plt.tight_layout()
plt.show()

# Save results
top_genes.to_csv("top_15_genes.csv")
adata.obs.to_csv("cell_umi_counts.csv")

print("Analysis complete. Results saved as 'top_15_genes.csv' and 'cell_umi_counts.csv'.")
