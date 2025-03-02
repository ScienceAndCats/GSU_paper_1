"""
UMI Counts Analysis and Histogram Plotter

- Loads a tab-delimited file containing UMI data.
- Groups data by 'Cell Barcode' to count unique UMIs per cell.
- Applies optional minimum and maximum UMI filters.
- Computes summary statistics (mean, std, median, min, max) for the filtered counts.
- Prints UMI counts and summary statistics.
- Saves filtered UMI counts to a CSV file.
- Plots a histogram (with KDE) of UMIs per barcode using seaborn,
  with user-defined bin width and axis limits.
"""




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file (assuming tab-separated values)
file_path = "PMP2_initial2_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt"  # Change this to your actual file path

# User-defined filtering parameters
min_umi_filter = None  # Set a min UMI threshold (None = no filter, or set a value like 10)
max_umi_filter = None  # Set a max UMI threshold (None = no filter, or set a value like 100)
bin_width = 1  # Set the bin width for histogram in units of UMIs
y_axis_max = 100  # Set the maximum y-axis value for the histogram (None = auto-scale)
x_axis_max = 80  # Set the maximum x-axis value for the histogram (None = auto-scale)

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

# Define bins with respect to x_axis_max if provided
if x_axis_max is not None:
    bins = range(umi_counts["Total UMIs"].min(), min(umi_counts["Total UMIs"].max(), x_axis_max) + bin_width, bin_width)
else:
    bins = range(umi_counts["Total UMIs"].min(), umi_counts["Total UMIs"].max() + bin_width, bin_width)

sns.histplot(umi_counts["Total UMIs"], bins=bins, kde=True, color="blue", edgecolor="black")

# Customize the plot
plt.xlabel("Total UMIs per Barcode")
plt.ylabel("Frequency")
plt.title("Distribution of UMIs per Cell Barcode")

# Set axis limits if specified
if y_axis_max is not None:
    plt.ylim(0, y_axis_max)
if x_axis_max is not None:
    plt.xlim(umi_counts["Total UMIs"].min(), x_axis_max)

plt.show()
