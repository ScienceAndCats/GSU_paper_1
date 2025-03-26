#!/usr/bin/env python3

import os
import sys
import re
import pickle
import pandas as pd

"""
PETRI-seq UMI Preprocessing Pipeline

This script preprocesses a PETRI-seq UMI table through multiple optional filtering
and grouping steps to clean and organize single-cell RNA data. The result is a
filtered UMI table and a corresponding gene expression matrix.

--------------------
UPDATED USAGE:
--------------------
    python script_name.py <FILE_PATH> <INPUT_UMI_FILENAME> <SELECTED_CELL_FILENAME> [run_rna_filter]

EXAMPLE:
    python script_name.py /my/full/path \
        dd2pal_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt \
        DD2PAL_selected_cumulative_frequency_table.txt \
        True

Where:
  - <FILE_PATH> is the directory containing your input files.
  - <INPUT_UMI_FILENAME> is the actual name of your UMI table file in that directory.
  - <SELECTED_CELL_FILENAME> is the name of the file listing selected cell barcodes.
  - The optional fourth argument ('True' or 'False') controls whether to run the rRNA/multihit filter.
    Defaults to the internal toggle if not provided.

--------------------
INPUTS:
--------------------
- INPUT_FILE: UMI table with columns [Cell Barcode, UMI, contig:gene, total_reads].
- SELECTED_CELL_FILE: List of barcodes ranked by read count, used for filtering the top cells.
- USER CONFIGURATIONS: Located within the script to toggle filters and define groupings.

--------------------
OUTPUTS:
--------------------
- Preprocessed UMI table (saved to OUTPUT_FOLDER).
- Gene expression matrix (cells x genes) in tab-separated format.
- Pickle files for each dictionary produced by the grouping/assignment functions.

--------------------
FILTERING OPTIONS:
--------------------
- RUN_FUNCTION_RNA_FILTER: Remove or clean rRNA and multihit entries.
- REMOVE_rRNA: Exclude single rRNA hits.
- RUN_REMOVE_COMMAS_FILTER: Remove rows containing commas.
- RUN_BARCODE_FILTER: Keep only the top N barcodes from SELECTED_CELL_FILE.
- RUN_BC1_SELECTION: Group barcodes by bc1 ranges.
- RUN_CONTIG_GROUPS: Group barcodes based on shared contig sets.
- RUN_CONTIG_ASSIGNMENT: Assign barcodes to predefined contig groups (e.g., all Luz19 cells are labeled infected)
- RUN_MIN_UMI_FILTER: Remove barcodes with fewer than MIN_UMI_COUNT UMIs.
- RUN_MIN_GENE_FILTER: Remove barcodes with fewer than MIN_GENE_COUNT unique genes.

--------------------
FINAL OUTPUTS:
--------------------
- <input_file>_preprocessed.txt: Filtered UMI table.
- <sample_name>_mixed_species_gene_matrix_preprocessed.txt: Gene expression matrix.
- <input_file>_<dictionary_name>.pkl: Pickle files for dictionaries.
"""


# --------------------
# USER CONFIGURATIONS
# --------------------
TROUBLESHOOTING = False

# Default values (will be overridden by command-line arguments)
FILE_PATH = "working_data/Unprocessed_data/PAcontrol_tob_PEG"
INPUT_FILE = "PAcontrol_tob_PEG_v11_threshold_0_filtered_mapped_UMIs_hans.txt"
OUTPUT_FOLDER = os.path.join(FILE_PATH, "preprocessed_PETRI_outputs")
SELECTED_CELL_FILE = "PAcontrol_tob_PEG_selected_cumulative_frequency_table.txt"

# Toggle for rRNA / ribo filtering
RUN_FUNCTION_RNA_FILTER = True # this is also for multihits, need to update the name
REMOVE_rRNA = False

# Toggle for removing rows with commas
RUN_REMOVE_COMMAS_FILTER = True

# Toggle for barcode-based filtering from a selected cell file
# Selects the top n number of barcodes with the highest amount of reads, NOT UMIs.
RUN_BARCODE_FILTER = True
NUM_BARCODES = 100

# Toggle for bc1 grouping
RUN_BC1_SELECTION = False
BC1_GROUPS = [
    ["Untreated", 1, 48],
    ["Treated", 49, 96]
]

# Toggle for contig grouping
RUN_CONTIG_GROUPS = True

# Toggle for new contig assignment
RUN_CONTIG_ASSIGNMENT = True
CONTIG_ASSIGNMENT_GROUPS = [
    ["luz19", "infected"],
    ["lkd16", "infected"],
    ["14one", "infected"],
    ["PA01", "pseudo"],
    ["MG1655", "ecoli"],
    ["DH5alpha", "ecoli"],
    ["pTNS2", "plasmid"]
]

# Toggle and parameter for filtering by minimum UMI count
RUN_MIN_UMI_FILTER = True
MIN_UMI_COUNT = 1

# Toggle and parameter for filtering by minimum unique gene count
RUN_MIN_GENE_FILTER = True
MIN_GENE_COUNT = 1


# --------------------
# HELPER FUNCTIONS
# --------------------
def filter_ribo_and_multihits(rows, remove_rRNA=False):
    """
    Removes rows with multiple hits while cleaning up rRNA hits.

    Processing logic:
      - If "rna-" appears 0 times in the contig:gene field, keep the row.
      - If "rna-" appears exactly once, replace the contig:gene with a single entry
        (extracting the contig from the first colon-containing chunk).
      - If there are >=2 'rna-' or >=2 contig markers, discard the row.
      - If remove_rRNA is True, skip any row that contains a single 'rna-' hit.
    """
    processed_rows = []
    for row in rows:
        if len(row) < 4:
            print("Malformed row (less than 4 columns), filtered out:", row)
            continue

        cell_barcode, umi, contig_gene, total_reads = row
        count_rna = contig_gene.count("rna-")
        count_contigs = contig_gene.count(":")
        if count_contigs >= 2 or count_rna >= 2:
            continue  # discard multiple-hit occurrences
        elif count_rna == 0:
            processed_rows.append(row)
        else:  # correct the rRNA hits formatting
            if remove_rRNA:
                continue
            gene_chunks = [g.strip() for g in contig_gene.split(",")]
            selected = None
            current_contig = None
            for g in gene_chunks:
                if ":" in g:
                    current_contig = g.split(":")[0]
                    break
            for g in gene_chunks:
                if "rna-" in g:
                    if ":" in g:
                        selected = g.split(":")[1]
                    else:
                        selected = g
                    break
            if selected and current_contig:
                new_gene_field = f"{current_contig}:{selected}"
                processed_rows.append([cell_barcode, umi, new_gene_field, total_reads])
    return processed_rows


def remove_rows_with_commas(rows):
    """Return only rows that do NOT contain a comma in any field."""
    return [row for row in rows if all(',' not in field for field in row)]


def get_barcode_list(selected_cell_file, num_barcodes):
    """
    Reads the selected cell file and returns a list of barcodes
    from its first `num_barcodes` rows.
    """
    barcodes = []
    with open(selected_cell_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[:num_barcodes]:
        fields = line.strip().split("\t")
        if fields:
            barcodes.append(fields[0])
    return barcodes


def filter_by_barcode_order(rows, barcode_order):
    """
    Keep and reorder rows by the list of barcodes in `barcode_order`.
    """
    barcode_to_rows = {}
    for row in rows:
        barcode = row[0]
        barcode_to_rows.setdefault(barcode, []).append(row)
    filtered_rows = []
    for bc in barcode_order:
        if bc in barcode_to_rows:
            filtered_rows.extend(barcode_to_rows[bc])
    return filtered_rows


def barcode_1_selection(rows, bc1_groups):
    """
    Group barcodes based on the integer after '_bc1_' in their name,
    using user-defined ranges in bc1_groups (list of [group_name, lower, upper]).
    Returns a dict: group_name -> list of barcodes.
    """
    group_dict = {group_name: [] for (group_name, _, _) in bc1_groups}
    bc1_pattern = re.compile(r"_bc1_(\d+)_")
    for row in rows:
        if len(row) < 4:
            continue
        cell_barcode = row[0]
        match = bc1_pattern.search(cell_barcode)
        if match:
            bc1_value = int(match.group(1))
            for group_name, lower_bound, upper_bound in bc1_groups:
                if lower_bound <= bc1_value <= upper_bound:
                    if cell_barcode not in group_dict[group_name]:
                        group_dict[group_name].append(cell_barcode)
                    break
    return group_dict


def group_barcodes_by_contigs(rows):
    """
    Creates a dictionary grouping barcodes by the set of contigs they contain.
    For each row, parse the contig from 'contig:gene' (everything before ':')
    and accumulate these in a set per barcode.
    Then invert that mapping to produce a dictionary of
      frozenset(contigs) -> list of barcodes.
    """
    barcode_to_contigs = {}
    for row in rows:
        if len(row) < 4:
            continue
        barcode = row[0]
        contig_gene = row[2]
        contig = contig_gene.split(":")[0].strip()
        barcode_to_contigs.setdefault(barcode, set()).add(contig)
    contig_dict = {}
    for barcode, contig_set in barcode_to_contigs.items():
        key = frozenset(contig_set)
        contig_dict.setdefault(key, []).append(barcode)
    return contig_dict


def assign_barcodes_to_contig_groups(rows, contig_assignments):
    """
    Using a user-defined list of pairs [<contig>, <group>], assign barcodes
    to groups based on whether any of their rows have that contig (in the contig:gene field).
    A barcode may be assigned to multiple groups (but is not duplicated within any group).

    Returns a dictionary mapping group names to lists of unique barcodes.
    """
    group_dict = {}
    for contig, group in contig_assignments:
        group_dict.setdefault(group, set())
    for row in rows:
        if len(row) < 4:
            continue
        barcode = row[0]
        contig_gene = row[2]
        contig = contig_gene.split(":")[0].strip()
        for assign_contig, group in contig_assignments:
            if contig == assign_contig:
                group_dict[group].add(barcode)
    # Convert sets to lists
    for group in group_dict:
        group_dict[group] = list(group_dict[group])
    return group_dict


def filter_by_min_umi(rows, min_count):
    """
    Count the number of rows (UMIs) for each Cell Barcode.
    Remove all rows belonging to any barcode that has fewer than min_count rows.
    """
    barcode_counts = {}
    for row in rows:
        barcode = row[0]
        barcode_counts[barcode] = barcode_counts.get(barcode, 0) + 1
    filtered_rows = [row for row in rows if barcode_counts.get(row[0], 0) >= min_count]
    return filtered_rows


def filter_by_min_gene_count(rows, min_count):
    """
    For each Cell Barcode, count the number of unique genes in the contig:gene column.
    Gene is taken as the substring after the first colon in the contig:gene field.
    Remove all rows for barcodes that have fewer than min_count unique genes.
    """
    barcode_to_genes = {}
    for row in rows:
        if len(row) < 3:
            continue
        barcode = row[0]
        contig_gene = row[2]
        parts = contig_gene.split(":")
        gene = parts[1].strip() if len(parts) > 1 else contig_gene.strip()
        barcode_to_genes.setdefault(barcode, set()).add(gene)
    filtered_rows = [row for row in rows if len(barcode_to_genes.get(row[0], set())) >= min_count]
    return filtered_rows


def process_file(input_path, output_path, run_rna_filter):
    """
    Main pipeline:
      1. Read UMI file, detect header if present.
      2. If toggled, apply ribo/multihit filter.
      3. If toggled, remove rows that contain commas.
      4. If toggled, reorder rows by barcodes from SELECTED_CELL_FILE.
      5. If toggled, group barcodes by bc1 range (prints summary and outputs pickle).
      6. If toggled, group barcodes by contig sets (prints summary and outputs pickle).
      7. If toggled, assign barcodes to contig groups (prints summary and outputs pickle).
      8. If toggled, remove barcodes with fewer than MIN_UMI_COUNT UMIs.
      9. If toggled, remove barcodes with fewer than MIN_GENE_COUNT unique genes.
     10. Write processed rows to output.
    """
    # 1) Read file
    with open(input_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()
    rows = [line.rstrip("\n").split("\t") for line in lines]

    header = None
    data_rows = rows
    if rows and "Cell Barcode" in rows[0][0]:
        header = rows[0]
        data_rows = rows[1:]

    # 2) Ribo/multihit filter
    if run_rna_filter:
        data_rows = filter_ribo_and_multihits(data_rows, remove_rRNA=REMOVE_rRNA)

    # 3) Remove rows with commas
    if RUN_REMOVE_COMMAS_FILTER:
        data_rows = remove_rows_with_commas(data_rows)

    # 4) Barcode reordering using selected cell file
    if RUN_BARCODE_FILTER:
        barcode_list = get_barcode_list(SELECTED_CELL_FILE, NUM_BARCODES)
        data_rows = filter_by_barcode_order(data_rows, barcode_list)

    # Determine base name for pickle file naming
    base = os.path.basename(input_path)
    name_part, _ = os.path.splitext(base)

    # 5) bc1 grouping
    if RUN_BC1_SELECTION:
        groups_dict = barcode_1_selection(data_rows, BC1_GROUPS)
        print("\nBC1 GROUP SELECTION SUMMARY:")
        for group_name, bc_list in groups_dict.items():
            print(f"  {group_name} => {len(bc_list)} barcodes")
        if TROUBLESHOOTING:
            for group in groups_dict:
                print(groups_dict[group])
        # Save bc1 selection dictionary as pickle
        pickle_file_path = os.path.join(OUTPUT_FOLDER, f"{name_part}_bc1_selection.pkl")
        with open(pickle_file_path, "wb") as f_pickle:
            pickle.dump(groups_dict, f_pickle)

    # 6) Contig grouping by set
    if RUN_CONTIG_GROUPS:
        contig_dict = group_barcodes_by_contigs(data_rows)
        print("\nCONTIG GROUPING SUMMARY:")
        for contig_set, bc_list in contig_dict.items():
            contig_str = "+".join(sorted(contig_set))
            print(f"  [{contig_str}] => {len(bc_list)} barcodes")
            if TROUBLESHOOTING:
                print(f"    Barcodes: {bc_list}")
        # Save contig grouping dictionary as pickle (frozenset keys are preserved)
        pickle_file_path = os.path.join(OUTPUT_FOLDER, f"{name_part}_contig_groups.pkl")
        with open(pickle_file_path, "wb") as f_pickle:
            pickle.dump(contig_dict, f_pickle)

    # 7) Assign barcodes to contig groups
    if RUN_CONTIG_ASSIGNMENT:
        assignment_dict = assign_barcodes_to_contig_groups(data_rows, CONTIG_ASSIGNMENT_GROUPS)
        print("\nCONTIG ASSIGNMENT SUMMARY:")
        for group, bc_list in assignment_dict.items():
            print(f"  {group} => {len(bc_list)} barcodes")
            if TROUBLESHOOTING:
                print(f"    Barcodes: {bc_list}")
        pickle_file_path = os.path.join(OUTPUT_FOLDER, f"{name_part}_contig_assignment.pkl")
        with open(pickle_file_path, "wb") as f_pickle:
            pickle.dump(assignment_dict, f_pickle)

    # 8) Filter out barcodes with fewer than MIN_UMI_COUNT UMIs
    if RUN_MIN_UMI_FILTER:
        data_rows = filter_by_min_umi(data_rows, MIN_UMI_COUNT)
        print(f"\nAfter filtering by minimum UMI count (min={MIN_UMI_COUNT}), {len(data_rows)} rows remain.")

    # 9) Filter out barcodes with fewer than MIN_GENE_COUNT unique genes
    if RUN_MIN_GENE_FILTER:
        data_rows = filter_by_min_gene_count(data_rows, MIN_GENE_COUNT)
        print(f"\nAfter filtering by minimum gene count (min={MIN_GENE_COUNT}), {len(data_rows)} rows remain.")

    # 10) Write processed output
    output_lines = []
    if header:
        output_lines.append("\t".join(header) + "\n")
    for row in data_rows:
        output_lines.append("\t".join(row) + "\n")
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(output_lines)
    print(f"\nFile has been processed and saved to: {output_path}")


# --------------------
# MAIN GATE
# --------------------
def main():
    """
    Main entry point.
    Usage: python script_name.py <FILE_PATH> <INPUT_UMI_FILENAME> <SELECTED_CELL_FILENAME> [run_rna_filter]
    If no arguments are provided, the configuration defaults are used.
    """
    global FILE_PATH, INPUT_FILE, SELECTED_CELL_FILE

    run_rna_filter = RUN_FUNCTION_RNA_FILTER

    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        INPUT_FILE = sys.argv[2]
    if len(sys.argv) > 3:
        SELECTED_CELL_FILE = sys.argv[3]
    if len(sys.argv) > 4:
        run_rna_filter = sys.argv[4].lower() == "true"

    # Construct the full paths to input files
    input_path = os.path.join(FILE_PATH, INPUT_FILE)
    selected_cell_file_path = os.path.join(FILE_PATH, SELECTED_CELL_FILE)

    # Update the global SELECTED_CELL_FILE so that 'process_file' can read it
    SELECTED_CELL_FILE = selected_cell_file_path

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    base_name = os.path.basename(input_path)
    name_part, ext_part = os.path.splitext(base_name)
    if not ext_part:
        ext_part = ".txt"
    output_file_name = f"{name_part}_preprocessed{ext_part}"
    output_path = os.path.join(OUTPUT_FOLDER, output_file_name)

    # Run the main pipeline
    process_file(input_path, output_path, run_rna_filter)




    # === Make the Gene Matrix ===
    # Read the input file that was generated earlier in the pipeline.
    table = pd.read_csv(output_path, sep='\t', index_col=0)

    # Select the relevant columns and group by Cell Barcode and contig:gene.
    matrix = table[['contig:gene', 'UMI']]
    gene_matrix = matrix.groupby(['Cell Barcode', 'contig:gene']).count()

    # Reshape the matrix: unstack, fill missing values, transpose, drop extra level.
    gene_matrix = gene_matrix.unstack(level='Cell Barcode')
    gene_matrix = gene_matrix.fillna(0)
    gene_matrix = gene_matrix.transpose()
    gene_matrix = gene_matrix.droplevel(0)
    gene_matrix = gene_matrix.loc[:, ~gene_matrix.columns.str.contains('ambiguous')] # get rid of ambiguous contig:genes

    # Write the resulting matrix to a file.
    sample_name = os.path.basename(INPUT_FILE).split("_")[0]
    gm_output_file = f"{sample_name}_mixed_species_gene_matrix_preprocessed.txt"
    GM_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, gm_output_file)

    gene_matrix.to_csv(GM_OUTPUT_PATH, sep='\t')
    print(f"Gene expression matrix saved to: {GM_OUTPUT_PATH}")



if __name__ == "__main__":
    main()
