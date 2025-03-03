from Bio import SeqIO
import sys


def fastq_to_fasta(fastq_file, fasta_file):
    """Converts a FASTQ file to a FASTA file."""
    with open(fastq_file, "r") as fq, open(fasta_file, "w") as fa:
        SeqIO.convert(fq, "fastq", fa, "fasta")
    print(f"Conversion complete: {fastq_file} -> {fasta_file}")


fastq_to_fasta("JRG08-DD2PAL_S3_L001_R2_001.fastq", "JRG08-DD2PAL_S3_L001_R2_001.fa")