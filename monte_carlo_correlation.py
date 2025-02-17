import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Load and Prepare Data
# ---------------------------

# Replace with your actual file
adata = sc.read_csv("17sep2024_Luz19_20min_initial_mixed_species_gene_matrix_hans.txt", delimiter="\t")

# Convert sparse matrix to dense if necessary
if hasattr(adata.X, "toarray"):
    expr = adata.X.toarray()
else:
    expr = adata.X

# ---------------------------
# Step 2: Compute Pairwise Gene Correlations in Real Data
# ---------------------------
gene_corr_real = np.corrcoef(expr.T)  # Transpose to compute gene-gene correlations
mean_corr_real = np.mean(gene_corr_real[np.triu_indices_from(gene_corr_real, k=1)])  # Upper triangle mean

print("Mean pairwise gene correlation (real data):", mean_corr_real)

# ---------------------------
# Step 3: Monte Carlo Simulation (Shuffle Gene Expression in Each Cell)
# ---------------------------
n_sim = 1000  # Number of simulation iterations
simulated_means = []

for _ in range(n_sim):
    permuted_expr = expr.copy()

    # Shuffle gene expression within each cell
    for i in range(permuted_expr.shape[0]):
        np.random.shuffle(permuted_expr[i, :])

    # Compute gene-gene correlation on shuffled data
    gene_corr_shuffled = np.corrcoef(permuted_expr.T)
    mean_corr_shuffled = np.mean(gene_corr_shuffled[np.triu_indices_from(gene_corr_shuffled, k=1)])
    simulated_means.append(mean_corr_shuffled)

# ---------------------------
# Step 4: Visualize and Assess the Result
# ---------------------------
plt.hist(simulated_means, bins=30, color='lightblue', edgecolor='black')
plt.axvline(mean_corr_real, color='red', linestyle='dashed', linewidth=2, label='Observed Mean')
plt.xlabel("Mean pairwise gene correlation")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation of Gene Co-Occurrence")
plt.legend()
plt.show()

# Compute p-value
p_val = np.sum(np.array(simulated_means) >= mean_corr_real) / n_sim
print("p-value:", p_val)
