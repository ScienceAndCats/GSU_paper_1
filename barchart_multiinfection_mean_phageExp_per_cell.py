import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import scanpy as sc

# Initialize the Dash app
app = dash.Dash(__name__)


def coexpression_counts(file_name, prefixes, min_counts_cells, min_counts_genes):
    """
    Reads a gene expression matrix from a file (cells as rows, genes as columns),
    removes genes with commas in their names (storing them in removed_genes.txt),
    converts the data to an AnnData object, and filters cells and genes using the
    provided minimum counts. Then, for each cell, it marks whether any gene with a
    given prefix is expressed (value > 0). Finally, for each prefix, it counts:
      - 'alone': cells where that prefix is expressed exclusively.
      - 'together': cells where that prefix is expressed along with at least one other.
    """
    try:
        raw_data = pd.read_csv(file_name, sep='\t', index_col=0)
    except Exception as e:
        return None, f"Error reading file: {e}"

    # Remove genes (columns) that contain a comma in their name
    removed_genes = [gene for gene in raw_data.columns if "," in gene]
    raw_data = raw_data.drop(columns=removed_genes)

    # Store removed genes separately
    removed_genes_file = "removed_genes.txt"
    with open(removed_genes_file, "w") as f:
        for gene in removed_genes:
            f.write(gene + "\n")
    print(f"Removed {len(removed_genes)} genes with commas. Saved list to {removed_genes_file}")

    # Convert the cleaned DataFrame to an AnnData object for filtering.
    adata = sc.AnnData(raw_data)

    # Filter cells and genes based on minimum counts.
    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Convert the filtered data back to a pandas DataFrame.
    filtered_data = adata.to_df()

    # Create a DataFrame to hold boolean flags for each prefix.
    # For each cell, mark True if any gene with that prefix is expressed (value > 0).
    bool_df = pd.DataFrame(index=filtered_data.index)
    for prefix in prefixes:
        matching_genes = [gene for gene in filtered_data.columns if gene.startswith(prefix)]
        if matching_genes:
            bool_series = (filtered_data[matching_genes] > 0).any(axis=1)
        else:
            bool_series = pd.Series(False, index=filtered_data.index)
        bool_df[prefix] = bool_series

    # Count, for each cell, how many gene groups are expressed.
    bool_df['total'] = bool_df.sum(axis=1)

    # For each prefix, count:
    # - 'alone': cell expresses that prefix and no other (total == 1)
    # - 'together': cell expresses that prefix along with at least one other (total > 1)
    counts = {}
    for prefix in prefixes:
        alone_count = bool_df[(bool_df[prefix]) & (bool_df['total'] == 1)].shape[0]
        together_count = bool_df[(bool_df[prefix]) & (bool_df['total'] > 1)].shape[0]
        counts[prefix] = {'alone': alone_count, 'together': together_count}

    return counts, None


# Define the app layout.
app.layout = html.Div([
    html.H2("Co-Expression: Genes Expressed Alone vs. Together in Cells"),
    html.Label("CSV/TSV File Name:"),
    dcc.Input(
        id="file-name-input",
        type="text",
        value="13Nov24_RT_multi_infection_gene_matrix.txt",
        style={"width": "50%"}
    ),
    html.Br(), html.Br(),
    html.Label("Set min_counts for cells:"),
    dcc.Input(id="min-counts-cells", type="number", value=5, step=1, min=1),
    html.Br(),
    html.Label("Set min_counts for genes:"),
    dcc.Input(id="min-counts-genes", type="number", value=5, step=1, min=1),
    html.Br(), html.Br(),
    html.Button("Calculate Co-Expression", id="update-button", n_clicks=0),
    html.Br(), html.Br(),
    dcc.Graph(id='bar-chart'),
    html.Div(id="error-message", style={"color": "red", "marginTop": "20px"})
])


# Callback to update the chart based on the calculated co-expression counts.
@app.callback(
    [Output("bar-chart", "figure"),
     Output("error-message", "children")],
    [Input("update-button", "n_clicks")],
    [State("file-name-input", "value"),
     State("min-counts-cells", "value"),
     State("min-counts-genes", "value")]
)
def update_chart(n_clicks, file_name, min_counts_cells, min_counts_genes):
    if n_clicks == 0:
        # Until the button is clicked, show an empty chart.
        return {}, ""

    prefixes = ["PA01:", "14one:", "luz19:", "lkd16:"]
    counts, error = coexpression_counts(file_name, prefixes, min_counts_cells, min_counts_genes)
    if error:
        return {}, error

    # Build a DataFrame for plotting.
    # Each row corresponds to a prefix and whether it is expressed 'Alone' or 'Together'.
    data = []
    for prefix in prefixes:
        data.append({"Prefix": prefix, "Expression": "Alone", "Count": counts[prefix]['alone']})
        data.append({"Prefix": prefix, "Expression": "Together", "Count": counts[prefix]['together']})
    df = pd.DataFrame(data)

    # Create a grouped bar chart.
    fig = px.bar(df, x="Prefix", y="Count", color="Expression", barmode="group",
                 title="Number of Cells Expressing Gene Groups Alone vs. Together")
    return fig, ""


if __name__ == '__main__':
    app.run_server(debug=True)
