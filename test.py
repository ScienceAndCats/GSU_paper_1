"""
Interactive Dash App for UMAP Visualization of Single-Cell RNA-seq Data

## Functionality:
- Loads and preprocesses single-cell gene expression data from a tab-delimited file.
- Filters cells and genes based on user-defined minimum count thresholds.
- Computes the number of "luz19:" and "lkd16:" genes expressed per cell.
- Performs PCA and generates UMAP embeddings for visualization.
- Allows interactive tuning of UMAP parameters (neighbors, min_dist, n_pcs).
- Displays UMAP scatter plot with points colored by the selected gene count.
- Supports lasso selection of cells for further analysis.
- Provides an option to download selected data points from UMAP.

## Inputs:
- Tab-separated gene expression matrix (CSV/TSV).
  - Rows: Cells.
  - Columns: Genes.
  - Values: Expression counts.
- User-defined parameters:
  - Minimum counts per cell and gene.
  - UMAP parameters (`n_neighbors`, `min_dist`, `n_pcs`).
  - **Plot customization:** Background color, graph size, color scale min/max.
  - **Gene selection for coloring:** "luz19:" or "lkd16:".

## Outputs:
- Interactive UMAP plot with cell points colored by the selected gene count.
- Printed preprocessing details (genes removed, filtering steps).
- CSV file containing selected data points (`selected_points.csv`).

## Dependencies:
- scanpy, pandas, numpy, dash, plotly
"""

import scanpy as sc
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes):
    # Read the raw data (tab-delimited). This DataFrame has all rows/columns unfiltered.
    raw_data = pd.read_csv(file_name, sep='\t', index_col=0)

    # Remove genes (columns) that contain a comma in their name
    removed_genes = [gene for gene in raw_data.columns if "," in gene]
    raw_data = raw_data.drop(columns=removed_genes)

    # Store removed genes separately
    removed_genes_file = "removed_genes.txt"
    with open(removed_genes_file, "w") as f:
        for gene in removed_genes:
            f.write(gene + "\n")

    print(f"Removed {len(removed_genes)} genes with commas. Saved list to {removed_genes_file}")

    # Make a copy that we'll keep for downloading later
    raw_data_copy = raw_data.copy()

    # Convert the raw data to an AnnData object for processing
    adata = sc.AnnData(raw_data)

    # Shuffle the rows of adata with a hard-coded seed for reproducibility.
    np.random.seed(42)
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices, :]

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Compute the count of "luz19:" genes expressed in each cell.
    # A gene is considered expressed if its value > 0.
    luz19_mask = adata.var_names.str.contains("luz19:")
    adata.obs['luz19_count'] = (adata.X[:, luz19_mask] > 0).sum(axis=1)

    # Compute the count of "lkd16:" genes expressed in each cell.
    lkd16_mask = adata.var_names.str.contains("lkd16:")
    adata.obs['lkd16_count'] = (adata.X[:, lkd16_mask] > 0).sum(axis=1)

    # Preprocess the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)

    # Perform PCA
    sc.tl.pca(adata, svd_solver='arpack')

    return adata, raw_data_copy

# Create UMAP DataFrame
def create_umap_df(adata, n_neighbors, min_dist, n_pcs):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    # Set random_state to ensure UMAP is reproducible.
    sc.tl.umap(adata, min_dist=min_dist, n_components=2, random_state=42)
    sc.tl.leiden(adata)

    umap_df = pd.DataFrame(
        adata.obsm['X_umap'],
        columns=['UMAP1', 'UMAP2'],
        index=adata.obs_names
    )
    umap_df['leiden'] = adata.obs['leiden']
    umap_df['luz19_count'] = adata.obs['luz19_count']
    umap_df['lkd16_count'] = adata.obs['lkd16_count']
    umap_df['cell_name'] = umap_df.index

    return umap_df

# App layout
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(
        id="file-name-input",
        type="text",
        value='working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.txt'
    ),
    html.Label("Set min_counts for cells:"),
    dcc.Input(id="min-counts-cells", type="number", value=5, step=1, min=1),
    html.Label("Set min_counts for genes:"),
    dcc.Input(id="min-counts-genes", type="number", value=5, step=1, min=1),
    html.Label("Set n_neighbors for UMAP:"),
    dcc.Input(id="n-neighbors-input", type="number", value=60, step=1, min=2, max=200),
    html.Label("Set min_dist for UMAP:"),
    dcc.Input(id="min-dist-input", type="number", value=0.1, step=0.1, min=0.0, max=1.0),
    html.Label("Set n_pcs for UMAP:"),
    dcc.Input(id="n-pcs-input", type="number", value=12, step=1, min=2),
    # New input for gene selection for coloring
    html.Label("Select gene prefix to color by:"),
    dcc.Dropdown(
        id="gene-select",
        options=[
            {"label": "luz19:", "value": "luz19"},
            {"label": "lkd16:", "value": "lkd16"}
        ],
        value="luz19",
        clearable=False
    ),
    # New customization inputs
    html.Label("Set plot background color (plot_bgcolor):"),
    dcc.Input(id="plot-bgcolor-input", type="text", value="darkgray"),
    html.Label("Set graph width:"),
    dcc.Input(id="graph-width-input", type="number", value=800, step=10),
    html.Label("Set graph height:"),
    dcc.Input(id="graph-height-input", type="number", value=600, step=10),
    html.Label("Set color scale minimum:"),
    dcc.Input(id="color-min-input", type="number", value=0, step=0.1),
    html.Label("Set color scale maximum:"),
    dcc.Input(id="color-max-input", type="number", value=10, step=0.1),
    html.Button("Update UMAP", id="update-button", n_clicks=0),
    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[dcc.Graph(id='umap-plot')]
    ),
    dcc.Store(id="umap-data"),
    dcc.Store(id="raw-data"),
    html.Button("Download Selected Points", id="download-btn", n_clicks=0),
    dcc.Download(id="download-dataframe")
])

# Callback to update UMAP and store both processed and raw data.
@app.callback(
    [Output("umap-plot", "figure"),
     Output("umap-data", "data"),
     Output("raw-data", "data")],
    [Input("update-button", "n_clicks")],
    [State("file-name-input", "value"),
     State("min-counts-cells", "value"),
     State("min-counts-genes", "value"),
     State("n-neighbors-input", "value"),
     State("min-dist-input", "value"),
     State("n-pcs-input", "value"),
     State("plot-bgcolor-input", "value"),
     State("graph-width-input", "value"),
     State("graph-height-input", "value"),
     State("color-min-input", "value"),
     State("color-max-input", "value"),
     State("gene-select", "value")],
    prevent_initial_call=True
)
def update_umap(n_clicks, file_name, min_counts_cells, min_counts_genes,
                n_neighbors, min_dist, n_pcs,
                plot_bgcolor, graph_width, graph_height, color_min, color_max,
                gene_select):
    adata, raw_data = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes)
    umap_df = create_umap_df(adata, n_neighbors, min_dist, n_pcs)

    # Select the gene count field and corresponding color scale based on the dropdown.
    if gene_select == "luz19":
        color_field = "luz19_count"
        color_scale = "Reds"
        color_title = "luz19 Count"
    else:
        color_field = "lkd16_count"
        color_scale = "Blues"
        color_title = "lkd16 Count"

    # Create the UMAP scatter plot, coloring points by the selected gene count.
    fig = px.scatter(
        umap_df,
        x='UMAP1',
        y='UMAP2',
        color=color_field,
        hover_data=['cell_name', 'leiden', color_field],
        custom_data=['cell_name'],
        color_continuous_scale=color_scale,
        range_color=[color_min, color_max],
        width=graph_width,
        height=graph_height
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(
        dragmode='lasso',
        coloraxis_colorbar=dict(title=color_title),
        paper_bgcolor="white",
        plot_bgcolor=plot_bgcolor
    )

    return fig, umap_df.to_json(date_format='iso', orient='split'), raw_data.to_json(date_format='iso', orient='split')

# Callback to download selected points using the raw_data copy.
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-btn", "n_clicks"),
    [State("umap-plot", "selectedData"),
     State("raw-data", "data")],
    prevent_initial_call=True
)
def download_selected_points(n_clicks, selectedData, raw_data_json):
    if not selectedData or "points" not in selectedData:
        return dash.no_update

    raw_df = pd.read_json(raw_data_json, orient='split')
    selected_names = [point['customdata'][0] for point in selectedData["points"] if "customdata" in point]
    selected_df = raw_df[raw_df.index.isin(selected_names)]

    if selected_df.empty:
        return dash.no_update

    return dcc.send_data_frame(selected_df.to_csv, "selected_points.csv")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
