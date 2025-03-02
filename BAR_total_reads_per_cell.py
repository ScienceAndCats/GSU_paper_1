"""
Histogram Plotter for Total Reads

- Loads a tab-delimited file containing a 'total_reads' column.
- Converts the 'total_reads' values to numeric, dropping non-numeric entries.
- Defines bins from 0 to 100 (with a bin width of 1) using numpy.
- Plots a histogram with skyblue bars and black edges.
- Sets axis labels and a plot title, then displays the plot.
"""





#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set parameters (adjust these as needed)
input_file = "2PMP_initial_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt"  # Replace with the actual file path
bin_lower = 0     # Lower bound of the bins
bin_upper = 100  # Upper bound of the bins
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
