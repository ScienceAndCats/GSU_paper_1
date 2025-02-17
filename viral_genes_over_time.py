import scanpy as sc
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, ctx, State
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)


# Load and preprocess the dataset
def load_and_preprocess_data(file_name, min_counts_cells=4, min_counts_genes=4):
    # Read the raw data (tab-delimited). This DataFrame has all rows/columns unfiltered.
    raw_data = pd.read_csv(file_name, sep='\t', index_col=0)
    # Make a copy for downloading later
    raw_data_copy = raw_data.copy()

    # Convert raw data to an AnnData object
    adata = sc.AnnData(raw_data)

    # Shuffle rows
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

    # (Normalization and scaling steps are commented out)
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    # adata = adata[:, adata.var['highly_variable']]
    # sc.pp.scale(adata, max_value=10)

    # Classify cells into groups based on barcodes
    def classify_cell(cell_name):
        bc1_value = int(cell_name.split('_')[2])
        if bc1_value < 25:
            return "Preinfection"
        elif bc1_value < 49:
            return "10min"
        elif bc1_value < 73:
            return "30min"
        else:
            return "40min"

    adata.obs['cell_group'] = adata.obs_names.map(classify_cell)

    # Perform PCA (kept for consistency)
    sc.tl.pca(adata, svd_solver='arpack')

    return adata, raw_data_copy


# Create Bulk DataFrame by aggregating single-cell data for genes with "luz19"
def create_bulk_df(adata):
    # Convert the processed data matrix into a DataFrame
    expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    # Keep only genes with "luz19" in the name
    luz19_cols = expr_df.columns[expr_df.columns.str.contains("luz19")]
    expr_df = expr_df[luz19_cols]
    # Add cell group information
    expr_df['cell_group'] = adata.obs['cell_group']
    # Group cells by their cell_group and sum the expression (i.e. create a "bulk" sample)
    bulk_df = expr_df.groupby('cell_group').sum().reset_index()
    # Melt the DataFrame to long format for Plotly
    bulk_melt = bulk_df.melt(id_vars='cell_group', var_name='gene', value_name='expression')
    return bulk_melt


# App layout with new y-axis range inputs and container for multiple plots
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(
        id="file-name-input", type="text",
        value='09Oct2024_luz19_initial_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt'
    ),
    html.Br(),
    html.Label("Set y-axis min:"),
    dcc.Input(
        id="yaxis-min-input", type="number", value=0, step=1
    ),
    html.Br(),
    html.Label("Set y-axis max:"),
    dcc.Input(
        id="yaxis-max-input", type="number", value=1000, step=1
    ),
    html.Br(),
    html.Button("Update Plot", id="update-button", n_clicks=0),
    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[html.Div(id='plots-container')]
    ),
    # Store the aggregated bulk DataFrame as JSON
    dcc.Store(id="umap-data"),
    # Store the raw (unprocessed) DataFrame as JSON
    dcc.Store(id="raw-data"),
    html.Button("Download Selected Points", id="download-btn", n_clicks=0),
    dcc.Download(id="download-dataframe")
])


# Callback to update the bar charts based on inputs.
# One graph per cell group is created and returned as a list of dcc.Graph components.
@app.callback(
    [Output("plots-container", "children"),
     Output("umap-data", "data"),
     Output("raw-data", "data")],
    Input("update-button", "n_clicks"),
    State("file-name-input", "value"),
    State("yaxis-min-input", "value"),
    State("yaxis-max-input", "value"),
    prevent_initial_call=True
)
def update_plots(n_clicks, file_name, yaxis_min, yaxis_max):
    # Load and preprocess the data using default filtering values
    adata, raw_data = load_and_preprocess_data(file_name, 4, 4)

    # Aggregate the single-cell data into bulk data (only for genes containing "luz19")
    bulk_melt = create_bulk_df(adata)

    # Create a list to hold graphs for each cell group.
    graphs = []
    # Define the expected order of groups
    expected_order = ['Preinfection', '10min', '30min', '40min']
    # Get the unique groups present and sort them based on the expected order.
    groups = sorted(bulk_melt['cell_group'].unique(), key=lambda x: expected_order.index(x))

    for group in groups:
        # Filter data for the current group
        group_data = bulk_melt[bulk_melt['cell_group'] == group]
        # Create bar chart for the current group:
        # x-axis is gene, y-axis is the summed expression.
        fig = px.bar(
            group_data,
            x='gene',
            y='expression',
            color='gene',
            title=f"Bulk Expression of 'luz19' Genes in {group}"
        )
        fig.update_layout(hovermode="closest")
        # Update the y-axis range using the provided inputs
        fig.update_yaxes(range=[yaxis_min, yaxis_max])

        # Append the graph to the list.
        graphs.append(dcc.Graph(figure=fig))

    return (
        graphs,
        bulk_melt.to_json(date_format='iso', orient='split'),
        raw_data.to_json(date_format='iso', orient='split')
    )


# Callback to download selected points (unprocessed data)
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-btn", "n_clicks"),
    State("umap-data", "data"),
    State("raw-data", "data"),
    prevent_initial_call=True
)
def download_selected_points(n_clicks, bulk_data, raw_data):
    """
    When the user clicks 'Download Selected Points', this example
    returns the entire raw data as a CSV. (Selection functionality is
    not implemented for multiple graphs.)
    """
    raw_df = pd.read_json(raw_data, orient='split')
    return dcc.send_data_frame(
        raw_df.to_csv,
        filename="selected_points.csv",
        sep='\t'
    )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
