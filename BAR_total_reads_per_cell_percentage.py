"""
Histogram Generator for Total Reads

- Reads a tab-delimited file containing read counts.
- Converts the 'total_reads' column to numeric, omitting non-numeric entries.
- Bins the data into defined ranges (default: 0-100 with a bin width of 1).
- Computes percentage weights for each bin and plots a histogram.
- Annotates each bar with its corresponding percentage.
- Customizes axis labels, tick formatting, and axis limits.
- Displays the final histogram plot.
"""




#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# Set parameters (adjust these as needed)
input_file = "2PMP_initial_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt"  # Replace with the actual file path
bin_lower = 0     # Lower bound of the bins
bin_upper = 100   # Upper bound of the bins
bin_width = 1     # Width of each bin

# Axis limits (set to None for auto-scaling)
x_axis_max = 30  # For example, set to 80 to limit the x-axis to 80, or None to auto-scale
y_axis_max = 50  # For example, set to 30 for a maximum of 30%, or None to auto-scale

# Compute bin edges from lower bound to upper bound.
bins = np.arange(bin_lower, bin_upper + bin_width, bin_width)

# Read the tab-delimited file.
df = pd.read_csv(input_file, sep='\t')

# Convert total_reads to numeric and drop any non-numeric values.
df['total_reads'] = pd.to_numeric(df['total_reads'], errors='coerce')
total_reads = df['total_reads'].dropna()

# Create the histogram using weights to convert counts to percentages.
plt.figure(figsize=(10, 6))
weights = np.ones_like(total_reads) / len(total_reads)
counts, bins, patches = plt.hist(total_reads, bins=bins, weights=weights,
                                 color='skyblue', edgecolor='black')

# Annotate each bar with its percentage (converted from fraction to percentage)
for count, patch in zip(counts, patches):
    if count > 0:  # Only annotate bars with a non-zero count.
        x = patch.get_x() + patch.get_width() / 2  # Center of the bin
        y = patch.get_height()
        plt.text(x, y, f'{count * 100:.1f}%', ha='center', va='bottom', fontsize=8)

plt.xlabel('Total Reads')
plt.ylabel('Percentage')
plt.title('Histogram of Total Reads')

# Format y-axis ticks as percentages
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Set x and y axis limits if specified.
if x_axis_max is not None:
    plt.xlim(bin_lower, x_axis_max)
if y_axis_max is not None:
    plt.ylim(0, y_axis_max / 100)

plt.tight_layout()
plt.show()
