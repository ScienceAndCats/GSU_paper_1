import os
import subprocess
import glob
import multiprocessing

# Detect the number of available CPU cores
num_cores = multiprocessing.cpu_count()

# Set your folder containing FASTQ files (update to your folder path)
fastq_folder = "fastq"
assembly_output = "assembly_results"

# Define the adapter sequence for Cutadapt (your custom adapter)
adapter_seq = "AGAATACACGACGCTCTTCCGATCTNNNNNNNNNNNNNNGGTCCTTGGCTTCGCNNNNNNNCCTCCTACGCCAGANNNNNNN"

# Define the Nextera transposase adapter sequences to remove
nextera_adapter1 = "TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG"
nextera_adapter2 = "GTCTCGTGGGCTCGGAGATGTGTATAAGAGACAG"

# Find all .fastq.gz files in the folder
fastq_files = sorted(glob.glob(os.path.join(fastq_folder, "*.fastq.gz")))

# Limit to a maximum of 8 files if necessary
max_files = 8
if len(fastq_files) > max_files:
    print(f"Warning: More than {max_files} FASTQ files found. Only processing the first {max_files} files.")
    fastq_files = fastq_files[:max_files]

# Automatically pair R1 and R2 files based on filename convention (_R1 and _R2)
r1_files = [f for f in fastq_files if "_R1" in f]
r2_files = [f.replace("_R1", "_R2") for f in r1_files if f.replace("_R1", "_R2") in fastq_files]

# Check that each R1 file has a corresponding R2 file
if len(r1_files) != len(r2_files):
    raise ValueError("Mismatch in paired-end files. Ensure every R1 file has a corresponding R2 file.")

print("Detected paired FASTQ files:")
for r1, r2 in zip(r1_files, r2_files):
    print(f"  R1: {r1}")
    print(f"  R2: {r2}")

# Merge all R1 files into one file and all R2 files into another file.
merged_r1 = os.path.join(fastq_folder, "merged_R1.fastq.gz")
merged_r2 = os.path.join(fastq_folder, "merged_R2.fastq.gz")

# Merge the files using the cat command (works for gzipped files)
with open(merged_r1, "wb") as outfile:
    for fname in r1_files:
        with open(fname, "rb") as infile:
            outfile.write(infile.read())

with open(merged_r2, "wb") as outfile:
    for fname in r2_files:
        with open(fname, "rb") as infile:
            outfile.write(infile.read())

print(f"\nMerged {len(r1_files)} R1 files into {merged_r1}")
print(f"Merged {len(r2_files)} R2 files into {merged_r2}")

# Define output file names for trimmed reads
trimmed_r1 = os.path.join(fastq_folder, "trimmed_R1.fastq.gz")
trimmed_r2 = os.path.join(fastq_folder, "trimmed_R2.fastq.gz")

# Build the Cutadapt command.
# Now we supply only the merged R1 and R2 files.
cmd_cutadapt = [
    "python3", "-m", "cutadapt",
    "-a", adapter_seq,
    "-a", nextera_adapter1,
    "-a", nextera_adapter2,
    "-A", adapter_seq,
    "-A", nextera_adapter1,
    "-A", nextera_adapter2,
    "--cut", "75",
    "-o", trimmed_r1,
    "-p", trimmed_r2,
    "-j", str(num_cores),  # Use all available CPU cores
    merged_r1,
    merged_r2
]

print(f"\nRunning Cutadapt using {num_cores} cores...")
subprocess.run(cmd_cutadapt, check=True)
print("Cutadapt trimming complete!")

# Create assembly output directory if it doesn't exist
os.makedirs(assembly_output, exist_ok=True)

# Build the SPAdes command.
cmd_spades = [
    "SPAdes-4.1.0-Linux/bin/spades.py",
    "--only-assembler",
    "-1", trimmed_r1,
    "-2", trimmed_r2,
    "-o", assembly_output,
    "-t", str(num_cores)  # Use all available CPU cores
]

print(f"\nRunning SPAdes using {num_cores} cores...")
subprocess.run(cmd_spades, check=True)
print(f"SPAdes assembly complete! Output saved in {assembly_output}")
