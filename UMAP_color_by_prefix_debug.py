import scanpy as sc
import pandas as pd
import numpy as np
import plotly.express as px
import colorsys

# Helper function to convert a hue (0-360) into a hex color.
def hue_to_hex(hue, saturation=1.0, lightness=0.5):
    h = hue / 360.0
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

# Load and preprocess the dataset
def load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes):
    raw_data = pd.read_csv(file_name, sep='\t', index_col=0)

    # Optionally, the removed genes block is commented out.
    """
    removed_genes = [gene for gene in raw_data.columns if "," in gene]
    raw_data = raw_data.drop(columns=removed_genes)
    removed_genes_file = "removed_genes.txt"
    with open(removed_genes_file, "w") as f:
        for gene in removed_genes:
            f.write(gene + "\n")
    print(f"Removed {len(removed_genes)} genes with commas. Saved list to {removed_genes_file}")
    """

    raw_data_copy = raw_data.copy()
    adata = sc.AnnData(raw_data)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices, :]

    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Identify genes for each group based on prefixes.
    luz19_genes = adata.var_names[adata.var_names.str.contains("luz19:")]
    pa01_genes = adata.var_names[adata.var_names.str.contains("PA01:")]
    dh5alpha_genes = adata.var_names[adata.var_names.str.contains("DH5alpha:")]
    mg1655_genes = adata.var_names[adata.var_names.str.contains("MG1655:")]

    # Get indices for each group.
    luz19_idx = np.where(adata.var_names.isin(luz19_genes))[0]
    pa01_idx = np.where(adata.var_names.isin(pa01_genes))[0]
    dh5alpha_idx = np.where(adata.var_names.isin(dh5alpha_genes))[0]
    mg1655_idx = np.where(adata.var_names.isin(mg1655_genes))[0]

    def label_cell(cell_expression):
        labels = []
        if cell_expression[luz19_idx].sum() > 0:
            labels.append("luz19")
        if cell_expression[pa01_idx].sum() > 0:
            labels.append("PA")
        if cell_expression[dh5alpha_idx].sum() > 0:
            labels.append("DH5alpha")
        if cell_expression[mg1655_idx].sum() > 0:
            labels.append("MG1655")
        return "+".join(labels) if labels else "none"

    adata.obs['infection_status'] = [label_cell(cell) for cell in adata.X]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    return adata, raw_data_copy

# Create UMAP DataFrame
def create_umap_df(adata, n_neighbors, min_dist, n_pcs):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=min_dist, n_components=2, random_state=42)
    sc.tl.leiden(adata)
    umap_df = pd.DataFrame(
        adata.obsm['X_umap'],
        columns=['UMAP1', 'UMAP2'],
        index=adata.obs_names
    )
    umap_df['leiden'] = adata.obs['leiden']
    umap_df['infection_status'] = adata.obs['infection_status']
    umap_df['cell_name'] = umap_df.index
    return umap_df

def main():
    # Define parameters
    file_name = '17Sep2024_Luz19_20min_mixed_species_gene_matrix.txt'
    min_counts_cells = 5
    min_counts_genes = 5
    n_neighbors = 60
    min_dist = 0.3
    n_pcs = 12

    # Process data and compute UMAP
    adata, raw_data_copy = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes)
    umap_df = create_umap_df(adata, n_neighbors, min_dist, n_pcs)

    # Define preset hues for each individual group.
    group_hues = {
        "luz19": 30,      # e.g. orange
        "PA": 0,          # red
        "DH5alpha": 210,  # blue-ish
        "MG1655": 300     # magenta/purple
    }

    # Function to compute color for composite groups.
    def get_composite_color(group, group_hues):
        if group == "none":
            return "#808080"
        if group in group_hues:
            return hue_to_hex(group_hues[group])
        else:
            parts = group.split('+')
            hues = [group_hues.get(part) for part in parts if part in group_hues]
            if hues:
                avg_hue = sum(hues) / len(hues)
                return hue_to_hex(avg_hue)
            else:
                return "#808080"

    unique_groups = umap_df['infection_status'].unique()
    color_map = {group: get_composite_color(group, group_hues) for group in unique_groups}

    # Create the UMAP scatter plot using Plotly Express.
    fig = px.scatter(
        umap_df,
        x='UMAP1',
        y='UMAP2',
        color='infection_status',
        hover_data=['cell_name', 'leiden', 'infection_status'],
        custom_data=['cell_name'],
        color_discrete_map=color_map
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(dragmode='lasso')

    # Display the figure
    fig.show()

if __name__ == '__main__':
    main()
