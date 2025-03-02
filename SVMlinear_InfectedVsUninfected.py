import scanpy as sc
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import colorsys

# Additional sklearn imports
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

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

    # Preprocess the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)

    # Perform PCA
    sc.tl.pca(adata, svd_solver='arpack')

    return adata, raw_data_copy

def create_umap(adata, n_neighbors, min_dist, n_pcs):
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

# ------------------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------------------
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(id="file-name-input", type="text",
              value='09Oct2024_luz19_initial_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt'),

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

    html.Button("Update UMAP + SVM", id="update-button", n_clicks=0),

    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[
            # 1) UMAP Plot (colored by SVM-predicted label)
            dcc.Graph(id='umap-svm-plot'),
            # 2) Confusion Matrix
            dcc.Graph(id='confusion-matrix-plot'),
            # 3) Precision-Recall Curve
            dcc.Graph(id='precision-recall-plot'),
            # 4) Feature Importance Plot
            dcc.Graph(id='feature-importance-plot'),
            # 5) Decision Boundary (PCA-reduced to 2D)
            dcc.Graph(id='decision-boundary-plot')
        ]
    ),

    dcc.Store(id="umap-data"),
    dcc.Store(id="raw-data"),

    html.Button("Download Selected Points", id="download-btn", n_clicks=0),
    dcc.Download(id="download-dataframe")
])

# ------------------------------------------------------------------------------
# CALLBACK: Main callback that updates UMAP, runs SVM, and creates all plots
# ------------------------------------------------------------------------------
@app.callback(
    [
        Output("umap-svm-plot", "figure"),
        Output("confusion-matrix-plot", "figure"),
        Output("precision-recall-plot", "figure"),
        Output("feature-importance-plot", "figure"),
        Output("decision-boundary-plot", "figure"),
        Output("umap-data", "data"),
        Output("raw-data", "data"),
    ],
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
def update_all_plots(n_clicks, infected_hue, uninfected_hue,
                     file_name, min_counts_cells, min_counts_genes,
                     n_neighbors, min_dist, n_pcs):
    """Runs the entire pipeline:
       1) Preprocess data, compute UMAP
       2) Train/test split on PCA
       3) SVM classification
       4) Generate requested plots
    """
    # -----------------------------
    # 1) Load data and compute UMAP
    # -----------------------------
    adata, raw_data = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes)
    umap_df = create_umap(adata, n_neighbors, min_dist, n_pcs)

    # Convert infection_status to numeric labels for SVM
    # e.g., infected=1, uninfected=0
    label_map = {"uninfected": 0, "infected": 1}
    y_full = adata.obs['infection_status'].map(label_map).values

    # We'll take the same principal components used for neighbors (n_pcs).
    X_full = adata.obsm["X_pca"][:, :n_pcs]

    # -----------------------------
    # 2) Train/Test Split and Train SVM
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y_full)
    svm_model = SVC(kernel="linear", random_state=42, probability=True) #svm_model = SVC(kernel="linear", random_state=42, probability=True)
    svm_model.fit(X_train, y_train)

    # Predict on the full set for UMAP coloring
    # (Alternatively, you can color only train/test subsets).
    y_pred_full = svm_model.predict(X_full)
    # Store the predicted label in the same order as adata.obs
    adata.obs['svm_pred_label'] = [int(p) for p in y_pred_full]
    adata.obs['svm_pred_class'] = ["infected" if p == 1 else "uninfected"
                                   for p in y_pred_full]

    # --------------
    # 3) Plot #1: UMAP colored by SVM-predicted label
    # --------------
    # Merge the SVM labels into umap_df
    umap_df["svm_pred_class"] = adata.obs['svm_pred_class'].values

    # Convert slider hue values to hex colors for infected/uninfected.
    infected_color = hue_to_hex(infected_hue)
    uninfected_color = hue_to_hex(uninfected_hue)
    color_discrete_map_svm = {
        "infected": infected_color,
        "uninfected": uninfected_color
    }

    fig_umap_svm = px.scatter(
        umap_df,
        x='UMAP1',
        y='UMAP2',
        color='svm_pred_class',
        hover_data=['cell_name', 'leiden', 'infection_status'],
        custom_data=['cell_name'],
        color_discrete_map=color_discrete_map_svm,
        title="UMAP Colored by SVM-Predicted Class"
    )
    fig_umap_svm.update_traces(marker=dict(size=3, opacity=0.8))
    fig_umap_svm.update_layout(dragmode='lasso')

    # -----------------------------
    # 4) Confusion Matrix (Test Set)
    # -----------------------------
    y_pred_test = svm_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    # We'll label the axes with uninfected=0, infected=1
    fig_cm = px.imshow(
        cm,
        x=["uninfected (pred)", "infected (pred)"],
        y=["uninfected (true)", "infected (true)"],
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion Matrix (Test Set)"
    )

    # -----------------------------
    # 5) Precision-Recall Curve (Test Set)
    # -----------------------------
    # Probability estimates required for precision-recall curve
    y_prob_test = svm_model.decision_function(X_test)  # or .predict_proba(...)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)
    avg_prec = average_precision_score(y_test, y_prob_test)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='Precision-Recall'
    ))
    fig_pr.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        title=f'Precision-Recall Curve (AP={avg_prec:.3f})'
    )

    # -----------------------------
    # 6) Feature Importance Plot (coefficients from linear SVM)
    # -----------------------------
    # For a linear SVM, the coefficients for each feature can be interpreted.
    # adata.var_names is shorter now (2000 HVGs) but still could be large.
    # We'll show top 10 features by absolute weight:
    coefs = svm_model.coef_[0]  # shape = (1, n_features)
    abs_coefs = np.abs(coefs)
    top_indices = np.argsort(abs_coefs)[::-1][:10]
    top_genes = adata.var_names[top_indices]
    top_importances = coefs[top_indices]

    fig_feat = px.bar(
        x=top_genes,
        y=top_importances,
        labels={"x": "Gene", "y": "Coefficient"},
        title="Top 10 Feature Importances (Linear SVM Coefficients)"
    )
    fig_feat.update_layout(xaxis_tickangle=45)

    # -----------------------------
    # 7) Decision Boundary Plot (using only first 2 PCA dims)
    # -----------------------------
    # We'll re-train an SVM on just the first 2 PCs (for visualization).
    X_2d_full = adata.obsm["X_pca"][:, :2]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X_2d_full, y_full, test_size=0.3, random_state=42, stratify=y_full
    )
    svm_2d = SVC(kernel='linear', random_state=42)
    svm_2d.fit(X2_train, y2_train)

    # Create a mesh
    x_min, x_max = X_2d_full[:, 0].min() - 1, X_2d_full[:, 0].max() + 1
    y_min, y_max = X_2d_full[:, 1].min() - 1, X_2d_full[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # We'll plot test-set points on top of the decision boundary
    fig_decision = go.Figure()
    fig_decision.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        colorscale=["blue", "red"],
        showscale=False,
        opacity=0.3,
        hoverinfo='skip'
    ))
    # Plot test-set points
    # color by actual infection_status
    for label_val, label_str, color_ in zip([0,1], ["uninfected","infected"],
                                           [uninfected_color, infected_color]):
        subset = (y2_test == label_val)
        fig_decision.add_trace(go.Scatter(
            x=X2_test[subset, 0],
            y=X2_test[subset, 1],
            mode='markers',
            marker=dict(color=color_, size=5),
            name=label_str
        ))

    fig_decision.update_layout(
        title="Decision Boundary (First 2 PCs)",
        xaxis_title="PC1",
        yaxis_title="PC2"
    )

    # Convert dataframes to JSON for storage
    return (
        fig_umap_svm,
        fig_cm,
        fig_pr,
        fig_feat,
        fig_decision,
        umap_df.to_json(date_format='iso', orient='split'),
        raw_data.to_json(date_format='iso', orient='split')
    )

# ------------------------------------------------------------------------------
# CALLBACK: Download selected points (unchanged)
# ------------------------------------------------------------------------------
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-btn", "n_clicks"),
    State("umap-svm-plot", "selectedData"),
    State("raw-data", "data"),
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

# ------------------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
