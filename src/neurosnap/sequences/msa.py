"""
Provides functions and classes related to processing protein sequence data.
TODO: Move some stuff from proteins.py over here
"""
import io
import os
import shutil
import subprocess
import tempfile

from Bio import SearchIO

from neurosnap.log import logger
from neurosnap.sequences.proteins import read_msa, write_msa


def run_phmmer(query, database, evalue=10, cpu=2):
  """
  -------------------------------------------------------
  Run phmmer using a query sequence against a database and
  return all the sequences that are considered as hits.
  Shameless stolen and adapted from https://github.com/seanrjohnson/protein_gibbs_sampler/blob/a5de349d5f6a474407fc0f19cecf39a0447a20a6/src/pgen/utils.py#L263
  -------------------------------------------------------
  Parameters:
    query......: Amino acid sequence of the protein you want to find hits for (str).
    database...: Path to reference database of sequences you want to search for hits and create and alignment with, must be a protein fasta file (str)
    evalue.....: The threshold E value for the phmmer hit to be reported (float)
    cpu........: The number of CPU cores to be used to run phmmer (str)
  Returns:
    hits: List of hits ranked by how good the hits are (list<str>)
  """
  assert shutil.which("phmmer") is not None, Exception("Cannot find phmmer. Please ensure phmmer is installed and added to your PATH.")

  # Create a fasta file containing the query protein sequence. The fasta file name is based on input genbank file name
  with tempfile.TemporaryDirectory() as tmp:
    queryfa_path = os.path.join(tmp, "query.fa")
    with open(queryfa_path, "w") as tmpfasta:
      print(f">QUERY\n{query}", file=tmpfasta)

    search_args = ["phmmer", "--noali", "--notextw", "--cpu", str(cpu), "-E", str(evalue), queryfa_path, database]
    out = subprocess.run(search_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    
    if out.returncode != 0:
      raise Exception(f"Error in hmmer execution: \n{out.stdout}\n{out.stderr}")

    hits = SearchIO.read(io.StringIO(out.stdout), "hmmer3-text")
    hit_names = [x.id for x in hits]
  
  return hit_names


def alignment_mafft(seqs, ep=0.0, op=1.53):
  """
  -------------------------------------------------------
  Generates an alignment using mafft.
  -------------------------------------------------------
  Parameters:
    seqs: Can be fasta file path, list of sequences, or dictionary where values are AA sequences and keys are their corresponding names/IDs.
    ep..: ep value for mafft, default is 0.00 (float)
    op..: op value for mafft, default is 1.53 (float)
  Returns:
    out_names: List of aligned protein names (list<str>)
    out_seqs.: List of corresponding protein sequences (list<str>)
  """
  # check if mafft is actually present
  assert shutil.which("mafft") is not None, Exception("Cannot create alignment without mafft being installed. Please install mafft either using a package manager or from https://mafft.cbrc.jp/alignment/software/")
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_fasta_path =  f"{tmp_dir}/tmp.fasta"
    if isinstance(seqs, str):
      tmp_fasta_path = seqs
    elif isinstance(seqs, list):
      with open(tmp_fasta_path, "w") as f:
        for i, seq in enumerate(seqs):
          f.write(f">seq_{i}\n{seq}\n")
    elif isinstance(seqs, dict):
      with open(tmp_fasta_path, "w") as f:
        for name, seq in seqs.items():
          f.write(f">{name}\n{seq}\n")
    else:
      raise ValueError(f"Input seqs cannot be of type {type(seqs)}. Can be fasta file path, list of sequences, or dictionary where values are AA sequences and keys are their corresponding names/IDs.")

    # logger.info(f"[*] Generating alignment with {len(seqs)} using mafft.")
    align_out = subprocess.run(["mafft", "--thread", "8", "--maxiterate", "1000", "--globalpair", "--ep", str(ep), "--op", str(op), tmp_fasta_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
      align_out.check_returncode()
    except:
      logger.error(align_out.stderr)
      raise Exception()

  return read_msa(io.StringIO(align_out.stdout.decode("utf-8")), allow_chars="-")

def generate_MSA_mafft(query, ref_db_path, size=float("inf"), in_name="input_sequence"):
  """
  -------------------------------------------------------
  Generate MSA using phmmer and mafft from reference sequences.
  -------------------------------------------------------
  Parameters:
    query......: Amino acid sequence of the protein you want to create an MSA of (str).
    ref_db_path: Path to reference database of sequences you want to search for hits and create and alignment with (str)
    size.......: Top n number of sequences to keep (int)
    in_name....: Optional name for input sequence to put in the output (str)
  Returns:
    out_names: List of aligned protein names (list<str>)
    out_seqs.: List of corresponding protein sequences (list<str>)
  """
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_fasta_path =  f"{tmp_dir}/tmp.fasta"
    # clean input fasta
    names, seqs = read_msa(ref_db_path, remove_chars="*-", drop_chars="X")
    # ensure no duplicate IDs
    reference_seqs = {}
    for i, (name, seq) in enumerate(zip(names, seqs)):
      if name not in seq:
        reference_seqs[name] = seq
      else:
        reference_seqs[f"{name}_{i}"] = seq
    # write cleaned fasta
    write_msa(tmp_fasta_path, reference_seqs.keys(), reference_seqs.values())
    
    # find hits
    hits = run_phmmer(query, tmp_fasta_path)
    logger.info(f"Found {len(hits)}/{len(names)} in reference DB for query.")
    unaligned_seqs = {in_name:query} # keep target sequence at the top
    found_names = set(in_name)
    found_seqs = set(query)
    for i in range(min(size, len(hits)-1)):
      hit_name = hits[i]
      hit_seq = reference_seqs[hit_name]
      if hit_name not in found_names and hit_seq not in found_seqs:
        found_names.add(hit_name)
        found_seqs.add(hit_seq)
        unaligned_seqs[hit_name] = hit_seq

  # generate alignment and return
  return alignment_mafft(unaligned_seqs)
