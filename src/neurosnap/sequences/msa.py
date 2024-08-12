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
      raise Exception(f"Error in hmmer execution: \n{out.stdout}\n{out.stderr}", file=os.sys.stderr)

    hits = SearchIO.read(io.StringIO(out.stdout), "hmmer3-text")
    hit_names = [x.id for x in hits]
  
  return hit_names