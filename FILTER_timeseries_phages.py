import scanpy as sc
import numpy as np
from scipy import sparse
import pandas as pd


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
file_path = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_mixed_species_gene_matrix_preprocessed.h5ad" # "working_data/Unprocessed_data/13Nov2024RTmultiinfection/13Nov24_RT_multi_infection_gene_matrix.txt"
adata = sc.read(file_path, delimiter="\t")

# Remove genes with commas in their names
adata = adata[:, ~adata.var_names.str.contains(",")]

# Filter cells and genes by minimum counts
sc.pp.filter_cells(adata, min_counts=5)
sc.pp.filter_genes(adata, min_counts=5)

# -------------------- Classify Cells into Timepoints -------------------- #
adata.obs["timepoint"] = [classify_cell(cell_name) for cell_name in adata.obs_names]

# -------------------- Phage Analysis -------------------- #
# We'll calculate expression for both "luz19:" and "lkd16:" genes.
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

# -------------------- Filtering for Specific Cells -------------------- #
# Select cells that express both phages and belong to the "20min" timepoint group.
filter_mask = (
        (adata.obs["luz19_expression"] > 0) &
        (adata.obs["lkd16_expression"] > 0) &
        (adata.obs["timepoint"] == "20min")
)
adata_filtered = adata[filter_mask].copy()

# Print out a summary of the filtered data
print("Number of cells in the filtered dataset:", adata_filtered.n_obs)
print("Timepoints in the filtered dataset:", adata_filtered.obs["timepoint"].unique())
print("Filtered cell names:", adata_filtered.obs_names.tolist())
