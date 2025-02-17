#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set parameters (adjust these as needed)
input_file = "multipass_v11_threshold_0_filtered_mapped_UMIs_hans.txt"  # Replace with the actual file path
bin_lower = 0     # Lower bound of the bins
bin_upper = 20  # Upper bound of the bins
bin_width = 1    # Width of each bin

# Compute bin edges from lower bound to upper bound.
bins = np.arange(bin_lower, bin_upper + bin_width, bin_width)

# Read the tab-delimited file.
df = pd.read_csv(input_file, sep='\t')

# Convert total_reads to numeric and drop any non-numeric values.
df['total_reads'] = pd.to_numeric(df['total_reads'], errors='coerce')
total_reads = df['total_reads'].dropna()

# Create the histogram using the computed bin edges.
plt.figure(figsize=(10, 6))
plt.hist(total_reads, bins=bins, color='skyblue', edgecolor='black')
plt.xlabel('Total Reads')
plt.ylabel('Frequency')
plt.title('Histogram of Total Reads')
plt.tight_layout()
plt.show()
