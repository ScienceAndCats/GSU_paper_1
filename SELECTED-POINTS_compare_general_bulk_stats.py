import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths for comparison (update these)
file_path_1 = "selected_points (16).csv"  # Adjust path if needed
file_path_2 = "selected_points (17).csv"  # Adjust path if needed

def load_data(file_path):
    """Loads a CSV file into an AnnData object."""
    df = pd.read_csv(file_path, index_col=0)
    adata = sc.AnnData(df)
    return adata

# Load datasets
adata1 = load_data(file_path_1)
adata2 = load_data(file_path_2)

# Compute statistics
for adata, label in zip([adata1, adata2], ["Dataset 1", "Dataset 2"]):
    adata.var["mean_expression_per_cell"] = np.mean(adata.X, axis=0)  # Mean expression per cell
    adata.obs["UMI_count"] = np.sum(adata.X, axis=1)  # UMI count per cell

    print(f"\nGeneral Statistics for {label}:")
    print(f"Total Cells: {adata.n_obs}")
    print(f"Total Genes: {adata.n_vars}")
    print(f"Mean UMI Count per Cell: {adata.obs['UMI_count'].mean():.2f}")
    print(f"Median UMI Count per Cell: {adata.obs['UMI_count'].median():.2f}")
    print(f"Total UMI Count: {adata.obs['UMI_count'].sum()}")

# Identify the top 15 most expressed genes based on mean per cell expression
top_genes_1 = adata1.var.sort_values("mean_expression_per_cell", ascending=False).head(15)
top_genes_2 = adata2.var.sort_values("mean_expression_per_cell", ascending=False).head(15)

# Save results
top_genes_1.to_csv("top_15_genes_dataset1.csv")
top_genes_2.to_csv("top_15_genes_dataset2.csv")
adata1.obs.to_csv("cell_umi_counts_dataset1.csv")
adata2.obs.to_csv("cell_umi_counts_dataset2.csv")

# Plot comparisons
plt.figure(figsize=(12, 6))

# UMI Count Distribution as Percentage
plt.subplot(1, 2, 1)
bins = np.linspace(0, max(adata1.obs["UMI_count"].max(), adata2.obs["UMI_count"].max()), 50)
plt.hist(adata1.obs["UMI_count"], bins=bins, alpha=0.6, label="Dataset 1", color='blue', density=True)
plt.hist(adata2.obs["UMI_count"], bins=bins, alpha=0.6, label="Dataset 2", color='red', density=True)

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.1f}%'))  # Convert to percentage
plt.xlabel("UMI Count per Cell")
plt.ylabel("Percentage of Cells")
plt.title("UMI Count Distribution as Percentage")
plt.legend()

# Top 15 expressed genes comparison based on mean expression per cell
plt.subplot(1, 2, 2)
genes1 = top_genes_1.index
genes2 = top_genes_2.index
top_gene_exp_1 = top_genes_1["mean_expression_per_cell"].values
top_gene_exp_2 = top_genes_2["mean_expression_per_cell"].values

bar_width = 0.4
x = np.arange(len(genes1))

plt.bar(x - bar_width/2, top_gene_exp_1, width=bar_width, label="Dataset 1", color='blue', alpha=0.7)
plt.bar(x + bar_width/2, top_gene_exp_2, width=bar_width, label="Dataset 2", color='red', alpha=0.7)
plt.xticks(x, genes1, rotation=90)
plt.xlabel("Gene")
plt.ylabel("Mean Expression Per Cell")
plt.title("Top 15 Most Expressed Genes (Mean Expression Per Cell)")
plt.legend()

plt.tight_layout()
plt.show()

print("Analysis complete. Results saved as 'top_15_genes_dataset1.csv', 'top_15_genes_dataset2.csv'.")
