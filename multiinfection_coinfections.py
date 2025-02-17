import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import scanpy as sc

app = dash.Dash(__name__)


# ----------------------------------------------------------------------------
# 1) Data Loading & Filtering
# ----------------------------------------------------------------------------
def process_data(file_name, min_counts_cells, min_counts_genes):
    """
    Loads the gene expression matrix (cells x genes), removes any gene
    whose name contains a comma (storing them in removed_genes.txt), and
    filters cells/genes with scanpy based on the given thresholds.

    Returns (filtered_data, error_message) where filtered_data is a
    pandas DataFrame (cells x genes), or None if there's an error.
    """
    try:
        raw_data = pd.read_csv(file_name, sep='\t', index_col=0)
    except Exception as e:
        return None, f"Error reading file: {e}"

    # Remove genes with commas in their names
    removed_genes = [gene for gene in raw_data.columns if "," in gene]
    raw_data = raw_data.drop(columns=removed_genes)

    # Store removed genes separately
    removed_genes_file = "removed_genes.txt"
    with open(removed_genes_file, "w") as f:
        for gene in removed_genes:
            f.write(gene + "\n")
    print(f"Removed {len(removed_genes)} genes with commas. Saved list to {removed_genes_file}")

    # Convert to AnnData to use scanpy for filtering
    adata = sc.AnnData(raw_data)

    # Filter cells and genes by minimum counts
    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Convert back to DataFrame
    filtered_data = adata.to_df()
    return filtered_data, None


# ----------------------------------------------------------------------------
# 2) Classify Each Cell's Infection Status
# ----------------------------------------------------------------------------
def classify_infection_status(df, phage_prefixes):
    """
    For each cell, marks True/False for each phage_prefix if any gene
    with that prefix is expressed (> 0). Then returns a Series mapping
    cell -> infection category:
      - "uninfected"
      - "14one:" (or whichever single prefix)
      - "14one: + luz19:" (for double)
      - "14one: + luz19: + lkd16:" (for triple)
    """
    bool_df = pd.DataFrame(index=df.index)
    for prefix in phage_prefixes:
        matching_genes = [g for g in df.columns if g.startswith(prefix)]
        if matching_genes:
            bool_df[prefix] = (df[matching_genes] > 0).any(axis=1)
        else:
            bool_df[prefix] = False

    def get_status(row):
        active = [p for p in phage_prefixes if row[p]]
        if len(active) == 0:
            return "uninfected"
        elif len(active) == 1:
            return active[0]  # e.g. "14one:"
        elif len(active) == 2:
            return " + ".join(active)
        else:
            # all three phages
            return " + ".join(active)

    return bool_df.apply(get_status, axis=1)


# ----------------------------------------------------------------------------
# 3) Observed vs. Expected Counts
# ----------------------------------------------------------------------------
def compute_observed_expected(infection_series, bool_df):
    """
    Takes:
      - infection_series: each cell's infection status (uninfected, single, double, triple).
      - bool_df: columns = ["14one:", "luz19:", "lkd16:"], row = cell, boolean if that phage is expressed.

    Returns a list of dictionaries with these columns:
      [
        {"Infection Category": ..., "Observed (# cells)": ..., "Expected (# cells)": ...},
        ...
      ]
    for each of the 8 possible categories:
      uninfected
      single (3 ways)
      double co-infection (3 ways)
      triple co-infection
    """
    # Count total cells
    total_cells = len(infection_series)

    # Observed counts per category
    obs_counts = infection_series.value_counts().to_dict()

    # The categories we want to list explicitly:
    categories = [
        "uninfected",
        "14one:",
        "luz19:",
        "lkd16:",
        "14one: + luz19:",
        "14one: + lkd16:",
        "luz19: + lkd16:",
        "14one: + luz19: + lkd16:"
    ]

    # Fractions infected with each phage (for computing expected)
    p_14one = bool_df["14one:"].mean()  # fraction of cells infected by 14one
    p_luz19 = bool_df["luz19:"].mean()
    p_lkd16 = bool_df["lkd16:"].mean()

    # We'll build a dict of expected counts (assuming independence).
    # Probability a cell is uninfected is (1 - p_14one)*(1 - p_luz19)*(1 - p_lkd16)
    # Probability a cell is "14one: only": p_14one*(1 - p_luz19)*(1 - p_lkd16)
    # etc. Then multiply each probability by total_cells.
    exp_dict = {}
    exp_dict["uninfected"] = (1 - p_14one) * (1 - p_luz19) * (1 - p_lkd16) * total_cells
    exp_dict["14one:"] = p_14one * (1 - p_luz19) * (1 - p_lkd16) * total_cells
    exp_dict["luz19:"] = (1 - p_14one) * p_luz19 * (1 - p_lkd16) * total_cells
    exp_dict["lkd16:"] = (1 - p_14one) * (1 - p_luz19) * p_lkd16 * total_cells
    exp_dict["14one: + luz19:"] = p_14one * p_luz19 * (1 - p_lkd16) * total_cells
    exp_dict["14one: + lkd16:"] = p_14one * (1 - p_luz19) * p_lkd16 * total_cells
    exp_dict["luz19: + lkd16:"] = (1 - p_14one) * p_luz19 * p_lkd16 * total_cells
    exp_dict["14one: + luz19: + lkd16:"] = p_14one * p_luz19 * p_lkd16 * total_cells

    # Build the final list of row dicts.
    table_data = []
    for cat in categories:
        observed_val = obs_counts.get(cat, 0)
        expected_val = exp_dict[cat]
        table_data.append({
            "Infection Category": cat,
            "Observed (# cells)": observed_val,
            "Expected (# cells)": round(expected_val, 2)  # rounding to 2 decimals
        })

    return table_data


# ----------------------------------------------------------------------------
# 4) Make a Dash/HTML Table
# ----------------------------------------------------------------------------
def make_observed_expected_table(table_data):
    """
    Given a list of dicts like:
       [
         {"Infection Category": "...", "Observed (# cells)": N, "Expected (# cells)": M},
         ...
       ]
    returns a Dash/HTML table.
    """
    header = html.Thead(html.Tr([
        html.Th("Infection Category"),
        html.Th("Observed (# cells)"),
        html.Th("Expected (# cells)")
    ]))
    body_rows = []
    for row in table_data:
        body_rows.append(
            html.Tr([
                html.Td(row["Infection Category"]),
                html.Td(str(row["Observed (# cells)"])),
                html.Td(str(row["Expected (# cells)"]))
            ])
        )
    body = html.Tbody(body_rows)
    table_style = {
        'border': '1px solid black',
        'borderCollapse': 'collapse',
        'marginTop': '20px'
    }
    return html.Table([header, body], style=table_style)


# ----------------------------------------------------------------------------
# 5) Dash Layout
# ----------------------------------------------------------------------------
app.layout = html.Div([
    html.H2("Co-Infection Table: Observed vs. Expected Number of Cells"),

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

    html.Button("Calculate Co-Infection", id="update-button", n_clicks=0),
    html.Br(), html.Br(),

    # Table container
    html.Div(id="results-table"),
    html.Div(id="error-message", style={"color": "red", "marginTop": "20px"}),
])


# ----------------------------------------------------------------------------
# 6) Dash Callback
# ----------------------------------------------------------------------------
@app.callback(
    [Output("results-table", "children"),
     Output("error-message", "children")],
    [Input("update-button", "n_clicks")],
    [State("file-name-input", "value"),
     State("min-counts-cells", "value"),
     State("min-counts-genes", "value")]
)
def update_coinfection_results(n_clicks, file_name, min_counts_cells, min_counts_genes):
    if n_clicks == 0:
        # No output until the button is clicked.
        return "", ""

    # 1) Process data (load + filter)
    filtered_data, error = process_data(file_name, min_counts_cells, min_counts_genes)
    if error:
        return "", error

    # 2) Classify each cell's infection
    phage_prefixes = ["14one:", "luz19:", "lkd16:"]
    infection_series = classify_infection_status(filtered_data, phage_prefixes)

    # 3) Build bool_df for each prefix (which cells are infected by each phage?)
    bool_df = pd.DataFrame(index=filtered_data.index)
    for prefix in phage_prefixes:
        matching = [g for g in filtered_data.columns if g.startswith(prefix)]
        bool_df[prefix] = (filtered_data[matching] > 0).any(axis=1) if matching else False

    # 4) Compute Observed vs. Expected
    table_data = compute_observed_expected(infection_series, bool_df)

    # 5) Build the HTML table
    results_table = make_observed_expected_table(table_data)

    return results_table, ""


if __name__ == "__main__":
    app.run_server(debug=True)
