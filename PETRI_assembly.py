import os
import subprocess
import glob
import gzip

# ======= Configuration =======
# Folder containing your gzipped FASTQ files
fastq_folder = r"C:\Users\hanst\PycharmProjects\GSU_paper_1\fastq"

# Where to store trimmed outputs (in the same folder)
trimmed_r1_gz = os.path.join(fastq_folder, "trimmed_R1.fastq.gz")
trimmed_r2_gz = os.path.join(fastq_folder, "trimmed_R2.fastq.gz")

# Where to write decompressed trimmed FASTQ files (for Velvet)
trimmed_r1 = os.path.join(fastq_folder, "trimmed_R1.fastq")
trimmed_r2 = os.path.join(fastq_folder, "trimmed_R2.fastq")

# Velvet assembly output directory
velvet_output = os.path.join(fastq_folder, "velvet_assembly")
kmer = "31"  # Adjust k-mer size as needed

# Adapter sequence to trim (with ambiguous bases as 'N')
adapter_seq = "AGAATACACGACGCTCTTCCGATCTNNNNNNNNNNNNNNGGTCCTTGGCTTCGCNNNNNNNCCTCCTACGCCAGANNNNNNN"

# ======= Step 1: Discover and Pair FASTQ Files =======
fastq_files = sorted(glob.glob(os.path.join(fastq_folder, "*.fastq.gz")))
r1_files = [f for f in fastq_files if "_R1" in f]
r2_files = [f.replace("_R1", "_R2") for f in r1_files if f.replace("_R1", "_R2") in fastq_files]

if len(r1_files) != len(r2_files):
    raise ValueError("Mismatch in paired-end files. Ensure every R1 file has a corresponding R2 file.")

print("Detected paired FASTQ files:")
for r1, r2 in zip(r1_files, r2_files):
    print(f"  R1: {r1}")
    print(f"  R2: {r2}")

# ======= Step 2: Run Cutadapt for Trimming =======
# Build the Cutadapt command. We call it as a Python module so Windows can find it.
cmd_cutadapt = [
                   "python", "-m", "cutadapt",
                   "-a", adapter_seq,  # Adapter for R1
                   "-A", adapter_seq,  # Adapter for R2
                   "--crop", "75",  # Crop reads to a maximum of 75bp
                   "-o", trimmed_r1_gz,  # Output for trimmed R1
                   "-p", trimmed_r2_gz,  # Output for trimmed R2
               ] + r1_files + r2_files

print("\nRunning Cutadapt...")
subprocess.run(cmd_cutadapt, check=True)
print("Cutadapt trimming complete!")


# ======= Step 3: Decompress Trimmed FASTQ Files =======
def gunzip_file(gz_file, out_file):
    with gzip.open(gz_file, 'rt') as fin, open(out_file, 'w') as fout:
        fout.write(fin.read())


print("\nDecompressing trimmed FASTQ files for Velvet...")
gunzip_file(trimmed_r1_gz, trimmed_r1)
gunzip_file(trimmed_r2_gz, trimmed_r2)
print("Decompression complete!")

# ======= Step 4: Run Velvet for Assembly =======
# Velvet requires two steps: velveth then velvetg.
# First, create the hash table with velveth.
cmd_velveth = [
    "velveth", velvet_output, kmer,
    "-fastq", "-shortPaired", "-separate",
    trimmed_r1, trimmed_r2
]
print("\nRunning velveth...")
subprocess.run(cmd_velveth, check=True)

# Next, run velvetg to perform the assembly.
cmd_velvetg = [
    "velvetg", velvet_output,
    "-exp_cov", "auto",
    "-cov_cutoff", "auto"
]
print("\nRunning velvetg...")
subprocess.run(cmd_velvetg, check=True)
print(f"Velvet assembly complete! Assembly output is in: {velvet_output}")
