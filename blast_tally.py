import subprocess
from collections import defaultdict
from Bio import SeqIO


make_database = True

if make_database:
    fasta_file = "luz19_lkd16_14one_pa01_pTNS2_ECs.fa"
    db_name = "genomes_db"

    # Build the command as a list of arguments
    cmd = ["makeblastdb", "-in", fasta_file, "-dbtype", "nucl", "-out", db_name]

    # Run the command
    subprocess.run(cmd, check=True)

    print("BLAST database created successfully.")




# Files and parameters
fastq_file = "JRG08-DD2PAL_S3_L001_R1_001.fastq"            # Input FASTQ file with reads
fasta_query_file = "reads.fasta"        # Temporary FASTA file
blast_db = "genomes_db"                 # Name of your BLAST database (created from your FASTA file with genomes)
blast_output = "blast_results.tsv"      # Output file for BLAST results

# Step 1: Convert FASTQ to FASTA (BLAST requires FASTA format)
with open(fasta_query_file, "w") as fasta_out:
    SeqIO.convert(fastq_file, "fastq", fasta_out, "fasta")

# Step 2: Run BLAST (blastn in this example)
blast_command = [
    "blastn",
    "-query", fasta_query_file,
    "-db", blast_db,
    "-outfmt", "6 qseqid sseqid pident length evalue bitscore",
    "-out", blast_output
]

# Execute the BLAST command
subprocess.run(blast_command, check=True)
print("BLAST search completed.")

# Step 3: Parse the BLAST output and tally hits per genome
# Here we assume that sseqid is the genome identifier in the BLAST output
hit_counts = defaultdict(int)
with open(blast_output) as result_file:
    for line in result_file:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        qseqid, sseqid = parts[0], parts[1]
        hit_counts[sseqid] += 1

# Step 4: Print the tally
print("Tally of reads aligning to each genome:")
for genome, count in hit_counts.items():
    print(f"{genome}: {count}")
