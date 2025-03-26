import scanpy as sc
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)

# Load and preprocess the dataset (up to PCA)
def load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes, gene):
    raw_data = pd.read_csv(file_name, sep='\t', index_col=0)

    # Remove genes with commas
    removed_genes = [g for g in raw_data.columns if "," in g]
    raw_data = raw_data.drop(columns=removed_genes)

    removed_genes_file = "removed_genes.txt"
    with open(removed_genes_file, "w") as f:
        for g in removed_genes:
            f.write(g + "\n")
    print(f"Removed {len(removed_genes)} genes with commas. Saved list to {removed_genes_file}")

    raw_data_copy = raw_data.copy()

    adata = sc.AnnData(raw_data)

    # For reproducibility
    np.random.seed(42)
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices, :]

    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Extract expression for the gene provided by the user
    if gene in adata.var_names:
        expr = adata[:, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().ravel()
        else:
            expr = np.ravel(expr)
        adata.obs['gene_expr'] = expr
    else:
        print(f"Warning: {gene} not found in the dataset. Setting expression to 0.")
        adata.obs['gene_expr'] = 0

    # Continue with normalization and transformation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, svd_solver='arpack')

    return adata, raw_data_copy

# Create UMAP DataFrame from the processed AnnData
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
    umap_df['gene_expr'] = adata.obs['gene_expr']
    umap_df['cell_name'] = umap_df.index

    return umap_df

# App layout with two buttons: one to update UMAP and one to update gene/coloring only.
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(
        id="file-name-input",
        type="text",
        value="working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.txt"
    ),
    html.Label("Set min_counts for cells:"),
    dcc.Input(id="min-counts-cells", type="number", value=5, step=1, min=1),
    html.Label("Set min_counts for genes:"),
    dcc.Input(id="min-counts-genes", type="number", value=5, step=1, min=1),
    html.Label("Enter gene for coloring:"),
    dcc.Input(id="gene-input", type="text", value="PA01:PA0690"),
    html.Label("Set n_neighbors for UMAP:"),
    dcc.Input(id="n-neighbors-input", type="number", value=60, step=1, min=2, max=200),
    html.Label("Set min_dist for UMAP:"),
    dcc.Input(id="min-dist-input", type="number", value=0.3, step=0.1, min=0.0, max=1.0),
    html.Label("Set n_pcs for UMAP:"),
    dcc.Input(id="n-pcs-input", type="number", value=12, step=1, min=2),
    html.Br(),
    html.Button("Update UMAP", id="update-umap-button", n_clicks=0),
    html.Button("Update Gene", id="update-gene-button", n_clicks=0),
    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[dcc.Graph(id='umap-plot')]
    ),
    # Store for the UMAP dataframe and raw data
    dcc.Store(id="umap-data"),
    dcc.Store(id="raw-data")
])

# Combined callback for both updating UMAP and updating gene/coloring only.
@app.callback(
    [Output("umap-plot", "figure"),
     Output("umap-data", "data"),
     Output("raw-data", "data")],
    [Input("update-umap-button", "n_clicks"),
     Input("update-gene-button", "n_clicks")],
    [State("file-name-input", "value"),
     State("min-counts-cells", "value"),
     State("min-counts-genes", "value"),
     State("gene-input", "value"),
     State("n-neighbors-input", "value"),
     State("min-dist-input", "value"),
     State("n-pcs-input", "value"),
     State("umap-data", "data"),
     State("raw-data", "data")]
)
def update_figure(n_clicks_umap, n_clicks_gene, file_name, min_counts_cells, min_counts_genes, gene,
                  n_neighbors, min_dist, n_pcs, stored_umap, stored_raw):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Case 1: Full UMAP update
    if button_id == "update-umap-button":
        adata, raw_data = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes, gene)
        umap_df = create_umap_df(adata, n_neighbors, min_dist, n_pcs)
        fig = px.scatter(
            umap_df,
            x='UMAP1',
            y='UMAP2',
            color='gene_expr',
            hover_data=['cell_name', 'leiden', 'gene_expr'],
            custom_data=['cell_name'],
            color_continuous_scale='Reds'
        )
        fig.update_traces(marker=dict(size=3, opacity=0.8))
        fig.update_layout(
            dragmode='lasso',
            plot_bgcolor='darkgray',
            paper_bgcolor='darkgray',
            coloraxis_colorbar=dict(title=f"{gene} Expression")
        )
        return fig, umap_df.to_json(date_format='iso', orient='split'), raw_data.to_json(date_format='iso', orient='split')

    # Case 2: Update gene (coloring) only
    elif button_id == "update-gene-button":
        # If no UMAP data is stored yet, do nothing.
        if stored_umap is None or stored_umap == "":
            raise dash.exceptions.PreventUpdate

        # Load stored UMAP dataframe
        umap_df = pd.read_json(stored_umap, orient='split')
        # Reload and preprocess data to update gene expression (without recalculating UMAP)
        adata, raw_data = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes, gene)
        # Get the new gene expression values and align them to the stored UMAP cells
        new_expr = adata.obs['gene_expr']
        new_expr = new_expr.reindex(umap_df.index, fill_value=0)
        umap_df['gene_expr'] = new_expr

        fig = px.scatter(
            umap_df,
            x='UMAP1',
            y='UMAP2',
            color='gene_expr',
            hover_data=['cell_name', 'leiden', 'gene_expr'],
            custom_data=['cell_name'],
            color_continuous_scale='Reds'
        )
        fig.update_traces(marker=dict(size=3, opacity=0.8))
        fig.update_layout(
            dragmode='lasso',
        plot_bgcolor = 'darkgray',
        paper_bgcolor = 'darkgray',
            coloraxis_colorbar=dict(title=f"{gene} Expression")
        )
        # Return updated figure; UMAP data and raw data remain unchanged.
        return fig, umap_df.to_json(date_format='iso', orient='split'), stored_raw

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
