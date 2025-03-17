"""
Script for analyzing single-cell RNA-seq data from phage-infected bacterial cells.

## Functionality:
- Loads and filters gene expression data.
- Identifies and quantifies phage gene expression per cell.
- Encodes infection patterns for different phage combinations.
- Computes expected infection rates based on MOI values.
- Performs a chi-square goodness-of-fit test to compare observed vs. expected infection patterns.
- Visualizes observed vs. expected infection counts using a table, bar chart, and heatmap.

## Inputs:
- `13Nov24_RT_multi_infection_gene_matrix.txt`: Tab-separated gene expression matrix.
  - Rows: Cells
  - Columns: Genes
  - Values: Expression counts

## Outputs:
- Printed number of genes removed due to naming issues.
- Printed observed vs. expected infection counts.
- Chi-square test results (statistic and p-value).
- Matplotlib table summarizing observed vs. expected infections.
- Plotly table with colorized differences between observed and expected values.
- Stacked bar chart visualizing phage co-occurrence across cells.
- `phage_cooccurrence_stacked_bar_chart.png`: Saved visualization of phage combinations.

## Dependencies:
- scanpy, numpy, scipy, pandas, seaborn, plotly, matplotlib
"""


import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# -------------------- Data Loading & Filtering -------------------- #
file_path = "../working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_mixed_species_gene_matrix_preprocessed.h5ad"
#adata = sc.read(file_path, delimiter="\t")  # Use delimiter="\t" for tab-separated files
adata = sc.read_h5ad(file_path)
print(adata)

# Remove genes with commas in their names
removed_genes = adata.var_names[adata.var_names.str.contains(",")]
adata = adata[:, ~adata.var_names.str.contains(",")]
print(f"Removed {len(removed_genes)} genes with commas.")

# Set filtering thresholds
min_counts_cells = 5
min_counts_genes = 5

# Filter cells and genes by minimum counts
sc.pp.filter_cells(adata, min_counts=min_counts_cells)
sc.pp.filter_genes(adata, min_counts=min_counts_genes)
# -------------------- End Data Loading & Filtering -------------------- #

# -------------------- Phage Analysis -------------------- #
phage_patterns = ["luz19:", "lkd16:", "14one:"]
phage_gene_dict = {}

for phage in phage_patterns:
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    phage_gene_dict[phage.strip(':')] = phage_genes

    # Create a boolean array indicating phage genes
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    # Calculate the number of genes expressed per cell for each phage
    if sparse.issparse(adata.X):
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1).A.flatten()
        phage_n_genes = (adata[:, adata.var[f'{phage.strip(":")}_genes']].X > 0).sum(axis=1).A.flatten()
    else:
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1)
        phage_n_genes = (adata[:, adata.var[f'{phage.strip(":")}_genes']].X > 0).sum(axis=1)

    # Add the counts to metadata
    metadata_key = f"{phage.strip(':')}_expression"
    adata.obs[metadata_key] = phage_expression
    adata.obs[f'{phage.strip(":")}_n_genes'] = phage_n_genes

# Encode infection patterns: luz19 -> 1, lkd16 -> 2, 14one -> 4.
adata.obs["phage_presence"] = (
    (adata.obs["luz19_expression"] > 0).astype(int) * 1 +
    (adata.obs["lkd16_expression"] > 0).astype(int) * 2 +
    (adata.obs["14one_expression"] > 0).astype(int) * 4
)

# Count cells with each combination of phages
phage_combinations = adata.obs["phage_presence"].value_counts()

# Decode combinations for visualization
combination_labels = {
    0: "No phage",
    1: "Only luz19",
    2: "Only lkd16",
    3: "luz19 and lkd16",
    4: "Only 14one",
    5: "luz19 and 14one",
    6: "lkd16 and 14one",
    7: "All phages"
}
phage_combinations.index = phage_combinations.index.map(combination_labels)
# -------------------- End Phage Analysis -------------------- #

# -------------------- Expected Infections Based on MOI -------------------- #
N_total = adata.n_obs
MOI_values = {"luz19": 0.19, "lkd16": 0.35, "14one": 0.38}
expected_infections_individual = {}
for phage, moi in MOI_values.items():
    expected_infections_individual[phage] = N_total * (1 - np.exp(-moi))

print("MOIs: ", MOI_values)
print("\nExpected number of cells infected for each individual phage (based on MOI):")
for phage, count in expected_infections_individual.items():
    print(f"{phage}: {int(count)} cells")

# -------------------- Expected Combination Counts -------------------- #
p_l = 1 - np.exp(-MOI_values["luz19"])
p_k = 1 - np.exp(-MOI_values["lkd16"])
p_14 = 1 - np.exp(-MOI_values["14one"])

expected_counts = {
    "No phage": N_total * ((1 - p_l) * (1 - p_k) * (1 - p_14)),
    "Only luz19": N_total * (p_l * (1 - p_k) * (1 - p_14)),
    "Only lkd16": N_total * ((1 - p_l) * p_k * (1 - p_14)),
    "luz19 and lkd16": N_total * (p_l * p_k * (1 - p_14)),
    "Only 14one": N_total * ((1 - p_l) * (1 - p_k) * p_14),
    "luz19 and 14one": N_total * (p_l * (1 - p_k) * p_14),
    "lkd16 and 14one": N_total * ((1 - p_l) * p_k * p_14),
    "All phages": N_total * (p_l * p_k * p_14)
}
# -------------------- End Expected Combination Counts -------------------- #

# -------------------- Chi-Square Goodness-of-Fit Test -------------------- #
categories_order = ["No phage", "Only luz19", "Only lkd16", "luz19 and lkd16",
                    "Only 14one", "luz19 and 14one", "lkd16 and 14one", "All phages"]

observed_list = [phage_combinations.get(cat, 0) for cat in categories_order]
expected_list = [expected_counts[cat] for cat in categories_order]

chi2, p_val = chisquare(f_obs=observed_list, f_exp=expected_list)
print(f"\nChi-square test results: chi2 = {chi2:.2f}, p-value = {p_val:.4f}")
# -------------------- End Chi-Square Test -------------------- #

# -------------------- Matplotlib Table (Observed vs Expected) -------------------- #
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_axis_off()
table_values = pd.DataFrame({
    "Observed": [int(x) for x in observed_list],
    "Expected": [int(round(x)) for x in expected_list],
    "Difference": [int(round(o - e)) for o, e in zip(observed_list, expected_list)]
}, index=categories_order).astype(str).values

table = ax.table(cellText=table_values,
                 colLabels=["Observed", "Expected", "Difference"],
                 rowLabels=categories_order,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2])
plt.title("Chi-Square Observed vs Expected Table", fontsize=14)
plt.show()
# -------------------- End Matplotlib Table -------------------- #

# -------------------- Helper Function for Color Mapping -------------------- #
def diff_color(val, vmin, vmax):
    """
    Maps a difference value to an RGB color.
    For negative values, interpolates from blue (at vmin) to white (0).
    For positive values, interpolates from white (0) to red (at vmax).
    """
    if val < 0:
        # For negative values: fraction = 0 at vmin, 1 at 0
        frac = (val - vmin) / (0 - vmin) if vmin != 0 else 0.5
        r = int(255 * frac)
        g = int(255 * frac)
        b = 255
    else:
        # For positive values: fraction = 0 at 0, 1 at vmax
        frac = val / vmax if vmax != 0 else 0.5
        r = 255
        g = int(255 * (1 - frac))
        b = int(255 * (1 - frac))
    return f"rgb({r},{g},{b})"

# Determine minimum and maximum of the "Difference" column for scaling
diff_vals = [int(round(o - e)) for o, e in zip(observed_list, expected_list)]
vmin = min(diff_vals)
vmax = max(diff_vals)
# Generate colors for each difference value
diff_colors = [diff_color(val, vmin, vmax) for val in diff_vals]

# -------------------- Plotly Table with Colorized "Difference" Column -------------------- #
header_values = ["Observed", "Expected", "Difference"]
# Build cell values: first column is Category, then each column from chi_square_df
cell_values = [[cat for cat in categories_order]] + [
    [int(x) for x in observed_list],
    [int(round(x)) for x in expected_list],
    diff_vals
]

# Build fill colors: use "white" for Category, Observed, Expected; use diff_colors for Difference
fill_colors = [
    ["white"] * len(categories_order),   # Category column (text only)
    ["white"] * len(categories_order),   # Observed column
    ["white"] * len(categories_order),   # Expected column
    diff_colors                          # Difference column gets our heatmapped colors
]

fig = go.Figure(data=[go.Table(
    header=dict(values=["Category"] + header_values,
                align='left',
                fill_color='lightgray'),
    cells=dict(
        values=cell_values,
        align='left',
        fill_color=fill_colors
    )
)])

fig.update_layout(title="Observed vs Expected - Chi-Square Difference", height=500)
fig.show()
# -------------------- End Plotly Table -------------------- #

# -------------------- Additional Visualization: Stacked Bar Chart -------------------- #
plt.figure(figsize=(10, 6))
bars = plt.bar(categories_order, observed_list, color="skyblue", edgecolor="black")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha="center", fontsize=12, fontweight="bold")
plt.title("Phage Co-Occurrence in Cells", fontsize=16)
plt.ylabel("Number of Cells", fontsize=14)
plt.xlabel("Phage Combination", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("phage_cooccurrence_stacked_bar_chart.png")
plt.show()

# -------------------- Print Total Number of Cells and Observed vs Expected -------------------- #
print(f"\nTotal number of cells: {N_total}")
print("\nObserved vs Expected values:")
for cat, obs, exp in zip(categories_order, observed_list, expected_list):
    print(f"{cat}: Observed = {obs}, Expected = {int(round(exp))}")
