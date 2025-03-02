import scanpy as sc
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import colorsys

# Helper function to convert a hue (0-360) into a hex color.
def hue_to_hex(hue, saturation=1.0, lightness=0.5):
    # Convert hue from degrees to fraction.
    h = hue / 360.0
    # Convert HLS (note: colorsys uses HLS instead of HSL) to RGB.
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    # Format RGB values to hex.
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

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

    # Identify uninfected cells based on "luz19:" genes
    luz19_genes = adata.var_names[adata.var_names.str.contains("luz19:")]
    luz19_gene_indices = np.where(adata.var_names.isin(luz19_genes))[0]

    def label_infection(cell_expression):
        luz19_expression = cell_expression[luz19_gene_indices]
        return "uninfected" if np.all(luz19_expression == 0) else "infected"

    adata.obs['infection_status'] = [label_infection(cell) for cell in adata.X]

    # --- New code: Remove genes with the "luz19:" prefix after labeling infection status ---
    genes_to_remove = adata.var_names[adata.var_names.str.startswith("luz19:")]
    adata = adata[:, ~adata.var_names.isin(genes_to_remove)]
    print(f"Removed {len(genes_to_remove)} genes with 'luz19:' prefix after labeling infection status.")

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

# App layout with two sliders to control the hue for infected and uninfected cells.
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(id="file-name-input", type="text", value='09Oct2024_luz19_initial_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt'),
    html.Label("Set min_counts for cells:"),
    dcc.Input(id="min-counts-cells", type="number", value=5, step=1, min=1),
    html.Label("Set min_counts for genes:"),
    dcc.Input(id="min-counts-genes", type="number", value=5, step=1, min=1),
    html.Label("Set n_neighbors for UMAP:"),
    dcc.Input(id="n-neighbors-input", type="number", value=60, step=1, min=2, max=200),
    html.Label("Set min_dist for UMAP:"),
    dcc.Input(id="min-dist-input", type="number", value=0.3, step=0.1, min=0.0, max=1.0),
    html.Label("Set n_pcs for UMAP:"),
    dcc.Input(id="n-pcs-input", type="number", value=12, step=1, min=2),
    html.Label("Infected Hue (0-360):"),
    dcc.Slider(
        id="infected-hue-slider",
        min=0, max=360, step=1, value=30,
        marks={i: str(i) for i in range(0, 361, 60)}
    ),
    html.Label("Uninfected Hue (0-360):"),
    dcc.Slider(
        id="uninfected-hue-slider",
        min=0, max=360, step=1, value=240,
        marks={i: str(i) for i in range(0, 361, 60)}
    ),
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
# This callback now also takes the slider values as inputs.
@app.callback(
    [Output("umap-plot", "figure"),
     Output("umap-data", "data"),
     Output("raw-data", "data")],
    Input("update-button", "n_clicks"),
    Input("infected-hue-slider", "value"),
    Input("uninfected-hue-slider", "value"),
    State("file-name-input", "value"),
    State("min-counts-cells", "value"),
    State("min-counts-genes", "value"),
    State("n-neighbors-input", "value"),
    State("min-dist-input", "value"),
    State("n-pcs-input", "value"),
    prevent_initial_call=True
)
def update_umap(n_clicks, infected_hue, uninfected_hue, file_name, min_counts_cells, min_counts_genes,
                n_neighbors, min_dist, n_pcs):
    adata, raw_data = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes)
    umap_df = create_umap_df(adata, n_neighbors, min_dist, n_pcs)

    # Convert slider hue values to hex colors.
    infected_color = hue_to_hex(infected_hue)
    uninfected_color = hue_to_hex(uninfected_hue)

    # Create the UMAP scatter plot using custom colors.
    fig = px.scatter(
        umap_df,
        x='UMAP1',
        y='UMAP2',
        color='infection_status',
        hover_data=['cell_name', 'leiden', 'infection_status'],
        custom_data=['cell_name'],
        color_discrete_map={'infected': infected_color, 'uninfected': uninfected_color}
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))  # All points are circles by default.
    fig.update_layout(dragmode='lasso')

    return fig, umap_df.to_json(date_format='iso', orient='split'), raw_data.to_json(date_format='iso', orient='split')

# Callback to download selected points using the raw_data copy.
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-btn", "n_clicks"),
    State("umap-plot", "selectedData"),
    State("raw-data", "data"),
    prevent_initial_call=True
)
def download_selected_points(n_clicks, selectedData, raw_data_json):
    # If no points are selected, do nothing.
    if not selectedData or "points" not in selectedData:
        return dash.no_update

    # Load the raw data copy from the stored JSON.
    raw_df = pd.read_json(raw_data_json, orient='split')

    # Extract cell names from the selected points.
    selected_names = [point['customdata'][0] for point in selectedData["points"] if "customdata" in point]

    # Filter the raw data for the selected cell names.
    selected_df = raw_df[raw_df.index.isin(selected_names)]

    if selected_df.empty:
        return dash.no_update

    # Trigger download as a CSV file.
    return dcc.send_data_frame(selected_df.to_csv, "selected_points.csv")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
