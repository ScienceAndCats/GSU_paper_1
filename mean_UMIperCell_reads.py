#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set parameters (adjust these as needed)
input_file = "L50_initial_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt"  # Replace with your actual file path

# Read the tab-delimited file.
df = pd.read_csv(input_file, sep='\t')

# Calculate unique UMIs per cell.
umi_counts_unique = df.groupby("Cell Barcode")["UMI"].nunique()

# Calculate and print the mean unique UMIs per cell.
mean_unique = umi_counts_unique.mean()
print("Mean unique UMIs per cell:", mean_unique)

# ------------------------------------------------------------------
# Option 1: Control bins by specifying a bin range.
# Define the lower bound, upper bound, and bin width.
bin_lower = 0      # Lower bound for bins
bin_upper = 20   # Upper bound for bins
bin_width = 1     # Width of each bin
bins_range = np.arange(bin_lower, bin_upper + bin_width, bin_width)

# ------------------------------------------------------------------
# Option 2: Control bins by specifying a fixed number of bins.
num_bins = 20

# ------------------------------------------------------------------
# Option 3: Control bins by specifying custom bin edges.
custom_bins = [0, 50, 100, 200, 500, 1000]

# ------------------------------------------------------------------
# Choose one of the binning options by setting 'bins_to_use':
# Uncomment the option you wish to use.

bins_to_use = bins_range   # Option 1: Use bin range
# bins_to_use = num_bins       # Option 2: Use a fixed number of bins
# bins_to_use = custom_bins      # Option 3: Use custom bin edges

# Plot histogram for unique UMIs per cell.
plt.figure(figsize=(10, 6))
plt.hist(umi_counts_unique, bins=bins_to_use, color='skyblue', edgecolor='black')
plt.xlabel('Unique UMIs per Cell')
plt.ylabel('Number of Cells')
plt.title('Histogram of Unique UMIs per Cell')
plt.tight_layout()
plt.show()
