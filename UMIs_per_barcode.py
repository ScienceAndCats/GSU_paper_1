import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file (assuming tab-separated values)
file_path = "initial_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt"  # Change this to your actual file path

# User-defined filtering parameters
min_umi_filter = None  # Set a min UMI threshold (None = no filter, or set a value like 10)
max_umi_filter = None  # Set a max UMI threshold (None = no filter, or set a value like 100)
bin_width = 1  # Set the bin width for histogram in units of UMIs

# Read the data into a Pandas DataFrame
df = pd.read_csv(file_path, sep='\t')

# Count UMIs per Cell Barcode
umi_counts = df.groupby("Cell Barcode")["UMI"].nunique().reset_index()

# Rename columns
umi_counts.columns = ["Cell Barcode", "Total UMIs"]

# Apply min and max UMI filters if specified
if min_umi_filter is not None:
    umi_counts = umi_counts[umi_counts["Total UMIs"] >= min_umi_filter]
if max_umi_filter is not None:
    umi_counts = umi_counts[umi_counts["Total UMIs"] <= max_umi_filter]

# Compute summary statistics after filtering
umi_summary = {
    "Mean UMIs per Barcode": umi_counts["Total UMIs"].mean(),
    "Standard Deviation": umi_counts["Total UMIs"].std(),
    "Median UMIs per Barcode": umi_counts["Total UMIs"].median(),
    "Minimum UMI Count": umi_counts["Total UMIs"].min(),
    "Maximum UMI Count": umi_counts["Total UMIs"].max()
}

# Print the UMI counts per barcode
print(umi_counts)

# Print the summary statistics
print("\nSummary Statistics (After Filtering if Applied):")
for key, value in umi_summary.items():
    print(f"{key}: {value:.2f}")

# Optional: Save results to a CSV file
umi_counts.to_csv("umi_counts_per_cell_barcode_filtered.csv", index=False)
print("UMI counts saved to 'umi_counts_per_cell_barcode_filtered.csv'")

# Generate a histogram with user-defined binning
plt.figure(figsize=(8, 6))
bins = range(umi_counts["Total UMIs"].min(), umi_counts["Total UMIs"].max() + bin_width, bin_width)
sns.histplot(umi_counts["Total UMIs"], bins=bins, kde=True, color="blue", edgecolor="black")

# Customize the plot
plt.xlabel("Total UMIs per Barcode")
plt.ylabel("Frequency")
plt.title("Distribution of UMIs per Cell Barcode")
plt.show()
