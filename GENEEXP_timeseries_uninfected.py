import scanpy as sc
import numpy as np
from scipy import sparse
import pandas as pd


"""
Catargorizes the cells by timeseries, and then filters for only uninfected cells.
Makes a csv of the top 50 genes expressed in those cells so you can investigate expression patterns. 
"""



# -------------------- Define Helper Function -------------------- #
def classify_cell(cell_name):
    """Extracts the bc1 value from the cell name and assigns a timepoint."""
    bc1_value = int(cell_name.split('_')[2])
    if bc1_value < 25:
        return "5min"
    elif bc1_value < 49:
        return "10min"
    elif bc1_value < 73:
        return "15min"
    else:
        return "20min"


# -------------------- Data Loading & Filtering -------------------- #
file_path = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_mixed_species_gene_matrix_preprocessed.h5ad"
adata = sc.read(file_path, delimiter="\t")

# Remove genes with commas in their names
adata = adata[:, ~adata.var_names.str.contains(",")]

# Filter cells and genes by minimum counts
sc.pp.filter_cells(adata, min_counts=5)
sc.pp.filter_genes(adata, min_counts=5)

# -------------------- Classify Cells into Timepoints -------------------- #
adata.obs["timepoint"] = [classify_cell(cell_name) for cell_name in adata.obs_names]

# -------------------- Phage Analysis -------------------- #
phage_patterns = ["luz19:", "lkd16:"]
for phage in phage_patterns:
    # Identify phage-specific genes
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    # Sum expression across these phage genes
    if sparse.issparse(adata.X):
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1).A.flatten()
    else:
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1)

    # Save expression in obs for downstream filtering
    adata.obs[f"{phage.strip(':')}_expression"] = phage_expression

# -------------------- Filtering for Non-Phage Expressing Cells -------------------- #
# Keep cells that have zero expression for *both* 'luz19:' and 'lkd16:' genes
filter_mask = (
        (adata.obs["luz19_expression"] == 0) &
        (adata.obs["lkd16_expression"] == 0)
)
adata_filtered = adata[filter_mask].copy()

# -------------------- Create CSV Files per Timepoint -------------------- #
timepoints = adata_filtered.obs["timepoint"].unique()

# For each timepoint group, compute:
#  1) Top 50 expressed genes (by total raw sum)
#  2) Raw sum, total cells, mean, and std for these 50 genes
for tp in timepoints:
    # Subset for this timepoint
    adata_tp = adata_filtered[adata_filtered.obs["timepoint"] == tp]

    # If sparse, convert to array for easy row/col ops
    if sparse.issparse(adata_tp.X):
        sums = adata_tp.X.sum(axis=0).A1
    else:
        sums = adata_tp.X.sum(axis=0)

    # Create a DataFrame with genes and their total sums
    df = pd.DataFrame({
        "gene": adata_tp.var_names,
        "sum_expr": sums
    })

    # Pick top 50 by sum
    df_top50 = df.nlargest(50, "sum_expr").copy()
    top50_genes = df_top50["gene"].values

    # Subset the AnnData to just these top 50 genes
    adata_tp_top50 = adata_tp[:, top50_genes]

    # Compute mean and std across cells
    # shape: (n_cells, n_genes)
    if sparse.issparse(adata_tp_top50.X):
        gene_means = adata_tp_top50.X.mean(axis=0).A1
        gene_stds = adata_tp_top50.X.std(axis=0).A1
    else:
        gene_means = adata_tp_top50.X.mean(axis=0)
        gene_stds = adata_tp_top50.X.std(axis=0)

    # Add mean, std, and total cell count to df_top50
    df_top50["mean_expr"] = gene_means
    df_top50["std_expr"] = gene_stds
    df_top50["total_cells"] = adata_tp.n_obs

    # Save as CSV
    output_file = f"top50_noPhage_{tp}.csv"
    df_top50.to_csv(output_file, index=False)

print("Done! CSV files with top 50 genes per timepoint (for cells not expressing phage genes) have been created.")
