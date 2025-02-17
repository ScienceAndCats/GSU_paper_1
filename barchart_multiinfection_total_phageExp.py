import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)


def count_gene_expressions(file_name, prefixes):
    """
    Reads a gene expression matrix from a file (genes as columns, cells as rows),
    removes genes with commas in their names (storing them separately),
    and for each prefix, sums the expression counts for all genes whose names
    start with that prefix.
    """
    try:
        # Read the file (adjust separator if necessary)
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

    counts_dict = {}
    for prefix in prefixes:
        # Identify genes whose name starts with the prefix.
        matching_genes = [gene for gene in raw_data.columns if gene.startswith(prefix)]
        # Sum the counts for these genes across all cells.
        # If no gene matches, the total count is 0.
        total_count = raw_data[matching_genes].sum().sum() if matching_genes else 0
        counts_dict[prefix] = total_count
    return counts_dict, None


# Define the app layout.
app.layout = html.Div([
    html.H2("Gene Expression Count by Prefix"),
    html.Label("CSV/TSV File Name:"),
    dcc.Input(
        id="file-name-input",
        type="text",
        value="13Nov24_RT_multi_infection_gene_matrix.txt",
        style={"width": "50%"}
    ),
    html.Br(), html.Br(),
    html.Button("Count Gene Expressions", id="update-button", n_clicks=0),
    html.Br(), html.Br(),
    dcc.Graph(id='bar-chart'),
    html.Div(id="error-message", style={"color": "red", "marginTop": "20px"})
])


# Callback to load data, count gene expressions, and update the bar chart.
@app.callback(
    [Output("bar-chart", "figure"),
     Output("error-message", "children")],
    [Input("update-button", "n_clicks"),
     Input("file-name-input", "value")]
)
def update_chart(n_clicks, file_name):
    if n_clicks == 0:
        # Display an empty chart until the button is clicked.
        return {}, ""

    # Define the prefixes to search for.
    prefixes = ["PA01:", "14one:", "luz19:", "lkd16:"]
    counts, error = count_gene_expressions(file_name, prefixes)
    if error:
        return {}, error

    # Convert the counts to a DataFrame for plotting.
    df = pd.DataFrame({
        "Prefix": list(counts.keys()),
        "Total Expression": list(counts.values())
    })

    # Create a bar chart.
    fig = px.bar(df, x="Prefix", y="Total Expression",
                 title="Total Gene Expression Counts by Prefix")
    return fig, ""


if __name__ == '__main__':
    app.run_server(debug=True)
