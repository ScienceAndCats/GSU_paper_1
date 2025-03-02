import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from multiprocessing import Pool

# -----------------------------------------
# Configuration
# -----------------------------------------
INPUT_FILE = "17Sep24_Luz19_20min_filtered_mapped_UMIs_hans.txt"

PREFIXES = [
    "PA01:",
    "lkd16:",
    "luz19:",
    "14one:",
    "DH5alpha:",
    "MG1655:",
    "pTNS2:"
]

MIN_UMIS_PER_CELL = 2

X_AXIS_MAX = 50
Y_AXIS_MAX = 30
BIN_WIDTH = 1

NUM_ROW_SAMPLES = 5000  # Total number of sampling events

# -----------------------------------------
# Step 1: Read and preprocess the data
# -----------------------------------------
df = pd.read_csv(INPUT_FILE, sep="\t")
df.columns = ["Cell_Barcode", "UMI", "contig_gene", "total_reads"]


# -----------------------------------------
# Step 2: Identify the unique prefix for each UMI row
# -----------------------------------------
def find_prefixes_in_row(contig_genes_str):
    genes = [g.strip() for g in contig_genes_str.split(",")]
    found = set()
    for gene in genes:
        for pfx in PREFIXES:
            if gene.startswith(pfx):
                found.add(pfx)
    return found


df["prefix_set"] = df["contig_gene"].apply(find_prefixes_in_row)
df = df[df["prefix_set"].apply(lambda s: len(s) == 1)].copy()
df["prefix"] = df["prefix_set"].apply(lambda s: list(s)[0])


# Note: We are not filtering by MIN_UMIS_PER_CELL here;
# that filter is applied after sampling.

# -----------------------------------------
# Step 3: Partition-based sampling with random selection (no weighting)
# -----------------------------------------
def sample_partition(df_part, num_samples):
    """
    Given a DataFrame partition and a target number of sampling events,
    perform sampling by randomly selecting rows (uniformly) that have available reads.
    For each event, the selected row's 'total_reads' is decremented by 1.
    If the Cell_Barcode is already in the sampled results, increment its total_reads by 1.
    Otherwise, add the sampled row (with total_reads set to 1) to the results.
    """
    local_df = df_part.copy()
    sampled_dict = {}  # keys: Cell_Barcode, values: row dict with aggregated total_reads
    for _ in range(num_samples):
        available = local_df[local_df["total_reads"] > 0]
        if available.empty:
            break  # No more rows available in this partition.
        # Uniformly select a row from those available.
        chosen_idx = np.random.choice(available.index)
        barcode = local_df.loc[chosen_idx, "Cell_Barcode"]
        # Decrement available reads for the chosen row.
        local_df.at[chosen_idx, "total_reads"] -= 1
        # Check if we've already recorded a sample for this barcode.
        if barcode in sampled_dict:
            sampled_dict[barcode]["total_reads"] += 1
        else:
            # Create a new sampled event for this barcode.
            # Make a copy of the row; override total_reads to 1 for the sampled event.
            sampled_copy = local_df.loc[chosen_idx].copy()
            sampled_copy["total_reads"] = 1
            sampled_dict[barcode] = sampled_copy.to_dict()
    if sampled_dict:
        return pd.DataFrame(list(sampled_dict.values()))
    else:
        return pd.DataFrame(columns=local_df.columns)



if __name__ == '__main__':
    # -----------------------------------------
    # Randomly shuffle the DataFrame to distribute rows randomly.
    # -----------------------------------------
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Determine the number of available cores.
    num_workers = os.cpu_count() or 1
    print(f"Using {num_workers} cores for parallel sampling.")

    # Partition the shuffled DataFrame into random subsets (one per worker).
    df_parts = np.array_split(df_shuffled, num_workers)

    # Allocate the total sampling events equally among partitions.
    samples_per_worker = NUM_ROW_SAMPLES // num_workers
    extra = NUM_ROW_SAMPLES % num_workers
    allocated_samples = [samples_per_worker + (1 if i < extra else 0) for i in range(num_workers)]
    print("Allocated sampling events per partition:", allocated_samples)

    # Run sampling on each partition in parallel.
    pool_args = [(df_parts[i], allocated_samples[i]) for i in range(num_workers)]
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(sample_partition, pool_args)

    # Combine the results from all partitions.
    df_sampled = pd.concat(results, ignore_index=True)
    print(f"Total sampled events: {len(df_sampled)}")

    # -----------------------------------------
    # Step 4: Filter out cells with low total UMIs (after sampling)
    # -----------------------------------------
    cell_umi_counts = df_sampled.groupby("Cell_Barcode")["UMI"].count()
    valid_cells = cell_umi_counts[cell_umi_counts >= MIN_UMIS_PER_CELL].index
    df_sampled = df_sampled[df_sampled["Cell_Barcode"].isin(valid_cells)].copy()

    # -----------------------------------------
    # Step 5: Create cell-level metadata with prefix combinations
    # -----------------------------------------
    cell_meta = df_sampled.groupby("Cell_Barcode").agg({
        "prefix": lambda ps: ",".join(sorted(set(ps))),
        "total_reads": "sum",
        "UMI": "count"  # Each sampled event counts as one UMI.
    }).reset_index()

    cell_meta.rename(columns={
        "prefix": "prefix_combination",
        "UMI": "cell_umi_count",
        "total_reads": "cell_total_reads"
    }, inplace=True)

    # -----------------------------------------
    # Step 6: Compute summary statistics for each prefix combination
    # -----------------------------------------
    grouped_combo = cell_meta.groupby("prefix_combination")
    summary_list = []
    for combo, group in grouped_combo:
        num_cells = len(group)
        total_umis = group["cell_umi_count"].sum()
        total_reads = group["cell_total_reads"].sum()
        mean_umis = total_umis / num_cells if num_cells > 0 else 0
        mean_reads = total_reads / num_cells if num_cells > 0 else 0
        summary_list.append({
            "prefix_combination": combo,
            "num_cells": num_cells,
            "total_umis": total_umis,
            "mean_umis_per_cell": mean_umis,
            "total_reads": total_reads,
            "mean_reads_per_cell": mean_reads
        })

    summary_df = pd.DataFrame(summary_list).sort_values("prefix_combination")
    print("\nSummary statistics by prefix combination (SAMPLED ROWS):")
    print(summary_df.to_string(index=False))

    # -----------------------------------------
    # Step 7: Plot histograms for each prefix combination in a single figure
    # -----------------------------------------
    bins = np.arange(0, X_AXIS_MAX + BIN_WIDTH, BIN_WIDTH)

    unique_combos = sorted(cell_meta["prefix_combination"].unique())
    n_plots = len(unique_combos)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, combo in zip(axes, unique_combos):
        subset = cell_meta[cell_meta["prefix_combination"] == combo]
        ax.hist(subset["cell_umi_count"], bins=bins, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f"Combination: {combo}")
        ax.set_xlabel("UMIs per cell")
        ax.set_ylabel("Count of cells")
        ax.set_xlim(0, X_AXIS_MAX)
        ax.set_ylim(0, Y_AXIS_MAX)

    for ax in axes[len(unique_combos):]:
        ax.axis("off")

    plt.tight_layout()

    base_name = os.path.splitext(INPUT_FILE)[0]
    output_file = base_name + "_histograms.png"
    plt.savefig(output_file)
    print(f"\nHistograms saved to {output_file}")

    plt.show()
