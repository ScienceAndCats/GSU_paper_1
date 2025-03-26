import os
import scanpy as sc
import pandas as pd
import pickle


"""
This script simply turns the pkl dicts and gene matrix txt files made by PREPROCESS_filtered_mapped_UMIs_to_Gene_Matrix.py 
into ScanPy anndata objects for further analysis.

I wanted to keep the txt and pkl files generated by that script in case I ever need to use a program other
than ScanPy to process the data.
"""


# Specify the folder that contains both the gene matrix file and the pkl files.
folder = "working_data/preprocessed_PETRI_outputs/PAcontrol_mixed_species_gene_matrix_preprocessed_100"  # <-- Update this to your folder path
gene_matrix_filename = "PAcontrol_mixed_species_gene_matrix_preprocessed.txt"  # Update gene matrix file name
pkl_prefix = "PAcontrol_tob_PEG_v11_threshold_0_filtered_mapped_UMIs_hans_preprocessed" # Removed from pkl file names for obs column naming. Don't forget the "_" at the end.


# 1. Load the gene expression matrix from the file
gene_matrix_path = os.path.join(folder, gene_matrix_filename)
df = pd.read_csv(gene_matrix_path, sep="\t", index_col=0)

# 2. Create an AnnData object from the DataFrame
adata = sc.AnnData(
    X=df.values,
    obs=pd.DataFrame(index=df.index),
    var=pd.DataFrame(index=df.columns)
)

# 3. Locate all pickle files in the folder
pkl_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pkl")]


# Helper function to convert group keys to a clean string
def convert_group_key(key):
    if isinstance(key, frozenset):
        # Join sorted elements so order is consistent
        return ",".join(sorted(key))
    return str(key)


# 4. Process each pickle file: assign group labels to each cell in adata.obs
for path in pkl_files:
    with open(path, "rb") as f:
        group_dict = pickle.load(f)

    # Use the file basename (without extension) as the column name in obs
    col_name = os.path.basename(path).replace(".pkl", "")
    col_name = col_name.replace(pkl_prefix, "")


    group_assignments = []
    for barcode in adata.obs_names:
        found_groups = []
        for group_key, barcodes_list in group_dict.items():
            if barcode in barcodes_list:
                found_groups.append(group_key)
        # Convert each group key using the helper function before joining
        group_assignments.append(
            ",".join(convert_group_key(g) for g in found_groups) if found_groups else None
        )

    # Insert the new column into adata.obs
    adata.obs[col_name] = group_assignments

# 5. Print the first 10 rows of the obs dataframe to check the result
print(adata.obs.head(10))

# 6. Save the complete AnnData object to a file in the same folder
adata_name = gene_matrix_filename.replace(".txt", "")
adata_filename = os.path.join(folder, f"{adata_name}.h5ad")
adata.write(adata_filename)
print(f"AnnData object saved to {adata_filename}")

# 7. Save the first 10 barcodes (obs rows) with all their metadata columns to a txt file
first10_obs_filename = os.path.join(folder, "first_10_barcodes_obs.txt")
adata.obs.head(10).to_csv(first10_obs_filename, sep="\t")
print(f"First 10 barcodes and their obs columns saved to {first10_obs_filename}")
