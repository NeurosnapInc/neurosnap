"""
Provides functions and classes related to processing protein sequence data.
"""

import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple, Union

import requests
from Bio import SearchIO

from neurosnap.api import USER_AGENT
from neurosnap.constants import STANDARD_AAs
from neurosnap.log import logger

### CONSTANTS ###


### FUNCTIONS ###
def read_msa(
  input_fasta: Union[str, io.TextIOBase],
  *,
  size: float = float("inf"),
  allow_chars: str = "",
  drop_chars: str = "",
  remove_chars: str = "*",
  uppercase: bool = True,
  name_allow_all_chars: bool = False,
) -> Iterator[Tuple[str, str]]:
  """Reads an MSA, a3m, or fasta file and yields (name, seq) pairs as a stream.
  Returned headers will consist of all characters up until the first space with
  the "|" character replaced with an underscore.

  Parameters:
    input_fasta: Path to read input a3m file, fasta as a raw string, or a file-handle like object to read
    size: Number of rows to read
    allow_chars: Sequences that contain characters not included within STANDARD_AAs+allow_chars will throw an exception
    drop_chars: Drop sequences that contain these characters. For example, ``"-X"``
    remove_chars: Removes these characters from sequences. For example, ``"*-X"``
    uppercase: Converts all amino acid chars to uppercase when True
    name_allow_all_chars: Uses the entire header string for names instead of the standard regex pattern

  Yields:
    Tuples of the form ``(name, seq)``

    - ``name``: protein name from the a3m file, including gaps
    - ``seq``: protein sequence from the a3m file, including gaps

  """
  allow_chars = allow_chars.replace("-", "\\-")
  drop_chars = drop_chars.replace("-", "\\-")
  remove_chars = remove_chars.replace("-", "\\-")

  # compile regular expressions
  if name_allow_all_chars:
    reg_name = re.compile(r"^>(.*)$")
  else:
    reg_name = re.compile(r"^>([\w_-\|]*)")

  if remove_chars:
    reg_rc = re.compile(f"[{remove_chars}\\s]")
  if drop_chars:
    reg_dc = re.compile(f"[{drop_chars}]")
  reg_ac = re.compile(f"^[{''.join(STANDARD_AAs)+allow_chars}]*$")

  if isinstance(input_fasta, str):
    if os.path.exists(input_fasta):
      f = open(input_fasta)
    else:
      f = io.StringIO(input_fasta)
  elif isinstance(input_fasta, io.TextIOBase):
    f = input_fasta
  else:
    raise ValueError(f"Invalid input for input_fasta, {type(input_fasta)} is not a valid type.")

  current_name = None
  current_seq = ""
  dropped = False
  yielded = 0

  try:
    for i, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      if line.startswith(">"):
        if current_name is not None:
          if not dropped and current_seq == "":
            raise ValueError(f"Invalid MSA/fasta. Header {current_name} is missing a sequence.")
          if not dropped:
            yield current_name, current_seq
            yielded += 1
            if yielded >= size:
              return
        match = reg_name.search(line)
        assert match is not None, f"Invalid MSA/fasta. {line} is not a valid header."
        name = match.group(1)
        name = name.replace("|", "_")
        assert len(name), f"Invalid MSA/fasta. line {i} has an empty header."
        current_name = name
        current_seq = ""
        dropped = False
      else:
        assert current_name is not None, f"Invalid MSA/fasta. line {i} has sequence data before a header."
        if uppercase:
          line = line.upper()
        # remove whitespace and remove_chars
        if remove_chars:
          line = reg_rc.sub("", line)
        # drop chars
        if drop_chars:
          match = reg_dc.search(line)
          if match is not None:
            dropped = True
            continue

        if not dropped:
          match = reg_ac.search(line)
          if match is None:
            raise ValueError(
              f"Sequence on line {i} contains an invalid character. Please specify whether you would like drop or replace characters in sequences like these. Sequence='{line}'"
            )
          current_seq += line
  finally:
    f.close()

  if current_name is not None:
    if not dropped and current_seq == "":
      assert len(current_seq), f"Invalid sequence for entry with name {current_name}. Sequence is empty."
    if not dropped:
      yield current_name, current_seq


def write_msa(output_path: str, names: List[str], seqs: List[str]):
  """Writes an MSA, a3m, or fasta to a file.
  Makes no assumptions about the validity of names or
  sequences. Will throw an exception if ``len(names) != len(seqs)``

  Parameters:
    output_path: Path to output file to write, will overwrite existing files
    names: List of proteins names from the file
    seqs: List of proteins sequences from the file

  """
  assert len(names) == len(seqs), "The number of names and sequences do not match."
  with open(output_path, "w") as f:
    for name, seq in zip(names, seqs):
      f.write(f">{name}\n{seq}\n")


def pad_seqs(seqs: List[str], char: str = "-", truncate: Union[bool, int] = False) -> List[str]:
  """Pads all sequences to the longest sequences length using a character from the right side.

  Parameters:
    seqs: List of sequences to pad
    chars: The character to perform the padding with, default is "-"
    truncate: When set to True will truncate all sequences to the length of the first, set to integer to truncate sequence to that length

  Returns:
    The padded sequences

  """
  if truncate is True:
    longest_seq = len(seqs[0])
  elif type(truncate) is int:
    assert truncate >= 1, "truncate must be either a boolean value or an integer greater than or equal to 1."
    longest_seq = truncate
  else:
    longest_seq = max(len(x) for x in seqs)

  for i, seq in enumerate(seqs):
    seqs[i] = seq.ljust(longest_seq, "-")
    seqs[i] = seqs[i][:longest_seq]
  return seqs


def seqid(seq1: str, seq2: str, *, count_gaps: bool = False) -> float:
  """Calculate the pairwise sequence identity of two same length sequences or alignments.
  Will not perform any alignment steps.

  Args:
      seq1: The 1st sequence / aligned sequence.
      seq2: The 2nd sequence / aligned sequence.
      count_gaps: When True, include gap positions in the identity calculation.

  Returns:
      The pairwise sequence identity, 0 means no matches found, 100 means sequences were identical.
  """
  assert len(seq1) == len(seq2), "Sequences are not the same length."
  assert len(seq1) > 0, "Sequence cannot have a length of 0."
  num_matches = 0
  denom = 0
  if count_gaps:
    for a, b in zip(seq1, seq2):
      if a == b:
        num_matches += 1
    denom = len(seq1)
  else:
    for a, b in zip(seq1, seq2):
      if a == "-" or b == "-":
        continue
      denom += 1
      if a == b:
        num_matches += 1
  return 100 * num_matches / denom if denom else 0.0


def alignment_coverage(seq1: str, seq2: str) -> float:
  """Calculate alignment coverage (%) for two aligned sequences. First sequence should the query sequence in most cases

  Args:
    seq1: Query aligned sequence (with gaps).
    seq2: Subject aligned sequence (with gaps).

  Returns:
    Percentage of non-gap positions in the query sequence.
  """
  aligned_query_positions = sum(1 for c1, c2 in zip(seq1, seq2) if c1 != "-")
  query_length = len(seq1.replace("-", ""))

  return aligned_query_positions / query_length * 100


def filter_msa(
  input_fasta: Union[str, io.TextIOBase],
  output_path: Union[str, os.PathLike],
  *,
  query: Optional[str] = None,
  cov: int = 50,
  id: int = 90,
  max_seqs: Union[int, float] = float("inf"),
) -> Tuple[List[str], List[str]]:
  """Filter an MSA based on sequence coverage and identity against a query.

  Parameters:
    input_fasta: Path to read input a3m file, fasta as a raw string, or a file-handle like object to read.
    output_path: Path to output file to write, will overwrite existing files.
    query: Query amino acid sequence. If not provided, the first sequence in the MSA is used.
    cov: Minimum percentage of query sequence coverage required to keep a sequence. It measures the
      proportion of non-gap positions in the query that are aligned to non-gap positions in the candidate.
      For example, with a 50% threshold, at least half of the query positions must align. Raising this
      value filters out shorter/partial matches and increases overall overlap.
    id: Minimum percentage of sequence identity required to keep a sequence. Identity is the exact match
      rate at aligned positions between the query and candidate. Higher values keep only close homologs;
      lower values allow more diverse sequences.
    max_seqs: Maximum number of sequences to write to the output. Sequences beyond this limit are dropped.

  Returns:
    A tuple of the form ``(names, seqs)`` for the filtered MSA.
  """
  # TODO: Possibly merge with read_msa
  names = []
  seqs = []
  for name, seq in read_msa(input_fasta):
    names.append(name)
    seqs.append(seq)
  assert len(seqs) > 0, "MSA is empty."

  if query is None:
    query_aligned = seqs[0]
  else:
    if len(query) == len(seqs[0]):
      query_aligned = query
    else:
      q_ungapped = query.replace("-", "")
      query_aligned = None
      for seq in seqs:
        if seq.replace("-", "") == q_ungapped:
          query_aligned = seq
          break
      if query_aligned is None:
        raise ValueError("Query sequence length does not match MSA and was not found in the MSA.")

  q = query_aligned
  assert len(q) > 0, "Query sequence cannot be empty."
  q_positions = [i for i, c in enumerate(q) if c != "-"]
  q_non_gap_count = len(q_positions)
  assert q_non_gap_count > 0, "Query sequence cannot be all gaps."

  kept_names = []
  kept_seqs = []

  for name, seq in zip(names, seqs):
    aligned = 0
    matches = 0
    for i in q_positions:
      qc = q[i]
      sc = seq[i]
      if sc != "-":
        aligned += 1
        if sc == qc:
          matches += 1
    coverage = 100 * aligned / q_non_gap_count
    identity = 100 * matches / aligned if aligned else 0.0
    if coverage >= cov and identity >= id:
      kept_names.append(name)
      kept_seqs.append(seq)

  if max_seqs != float("inf"):
    kept_names = kept_names[: int(max_seqs)]
    kept_seqs = kept_seqs[: int(max_seqs)]

  with open(output_path, "w") as f:
    for name, seq in zip(kept_names, kept_seqs):
      f.write(f">{name}\n{seq}\n")

  return kept_names, kept_seqs


def consensus_sequence(sequences: List[str]) -> str:
  """Generates the consensus sequence from a list of aligned sequences.

  The consensus is formed by taking the most common character at each position.

  Args:
      sequences: A list of equal-length sequences (e.g., amino acid or DNA).

  Returns:
      The consensus sequence.

  Raises:
      ValueError: If the sequence list is empty or sequences are of unequal lengths.
  """
  if not sequences:
    raise ValueError("The sequence list is empty.")

  seq_length = len(sequences[0])
  if any(len(seq) != seq_length for seq in sequences):
    raise ValueError("All sequences must be of the same length.")

  consensus = []
  for i in range(seq_length):
    column = [seq[i] for seq in sequences]
    most_common, _ = Counter(column).most_common(1)[0]
    consensus.append(most_common)

  return "".join(consensus)


def run_phmmer(query: str, database: str, evalue: float = 10.0, cpu: int = 2) -> List[str]:
  """Run phmmer using a query sequence against a database and
  return all the sequences that are considered as hits.
  Shamelessly stolen and adapted from https://github.com/seanrjohnson/protein_gibbs_sampler/blob/a5de349d5f6a474407fc0f19cecf39a0447a20a6/src/pgen/utils.py#L263

  Parameters:
    query: Amino acid sequence of the protein you want to find hits for
    database: Path to reference database of sequences you want to search for hits and create and alignment with, must be a protein fasta file
    evalue: The threshold E value for the phmmer hit to be reported
    cpu: The number of CPU cores to be used to run phmmer

  Returns:
    List of hits ranked by how good the hits are

  """
  assert shutil.which("phmmer") is not None, "Cannot find phmmer. Please ensure phmmer is installed and added to your PATH."

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


def align_mafft(seqs: Union[str, List[str], Dict[str, str]], ep: float = 0.0, op: float = 1.53, threads: int = 8) -> Tuple[List[str], List[str]]:
  """Generates an alignment using mafft.

  Parameters:
    seqs: Can be:

      - fasta file path,
      - list of sequences, or
      - dictionary where values are AA sequences and keys are their corresponding names/IDs
    ep: ep value for mafft, default is 0.00
    op: op value for mafft, default is 1.53
    threads: Number of MAFFT threads to use (default: 8)

  Returns:
    A tuple of the form ``(out_names, out_seqs)``

    - ``out_names``: list of aligned protein names
    - ``out_seqs``: list of corresponding protein sequences
  """
  assert (
    shutil.which("mafft") is not None
  ), "Cannot create alignment without mafft being installed. Please install mafft either using a package manager or from https://mafft.cbrc.jp/alignment/software/"

  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_fasta_path = f"{tmp_dir}/tmp.fasta"
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
      raise ValueError(
        f"Input seqs cannot be of type {type(seqs)}. Can be fasta file path, list of sequences, or dictionary where values are AA sequences and keys are their corresponding names/IDs."
      )

    align_out = subprocess.run(
      ["mafft", "--thread", str(threads), "--maxiterate", "1000", "--globalpair", "--ep", str(ep), "--op", str(op), tmp_fasta_path],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    try:
      align_out.check_returncode()
    except:  # noqa: E722
      # Keep behavior similar to original; surface stderr
      raise Exception(align_out.stderr.decode("utf-8", errors="ignore"))

  names = []
  seqs = []
  for name, seq in read_msa(io.StringIO(align_out.stdout.decode("utf-8")), allow_chars="-"):
    names.append(name)
    seqs.append(seq)
  return names, seqs


def run_phmmer_mafft(
  query: str, ref_db_path: str, size: Optional[int] = None, in_name: str = "input_sequence", phmmer_cpu: int = 2, mafft_threads: int = 8
) -> Tuple[List[str], List[str]]:
  """Generate MSA using phmmer and mafft from reference sequences.

  Parameters:
    query: Amino acid sequence of the protein whose MSA you want to create
    ref_db_path: Path to reference database of sequences with which you want to search for hits and create and alignment
    size: Total number of sequences to return, including the query. Use None to include all hits. Must be a positive integer greater than 1.
    in_name: Optional name for input sequence to put in the output
    phmmer_cpu: The number of CPU cores to be used to run phmmer (default: 2)
    mafft_threads: Number of MAFFT threads to use (default: 8)

  Returns:
    A tuple of the form ``(out_names, out_seqs)``

    - ``out_names``: list of aligned protein names
    - ``out_seqs``: list of corresponding protein sequences
  """
  assert size is None or (isinstance(size, int) and size > 1), "size must be None or a positive integer greater than 1."
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_fasta_path = f"{tmp_dir}/tmp.fasta"

    # clean input fasta
    names = []
    seqs = []
    for name, seq in read_msa(ref_db_path, remove_chars="*-", drop_chars="X"):
      names.append(name)
      seqs.append(seq)

    # ensure no duplicate IDs
    reference_seqs = {}
    for i, (name, seq) in enumerate(zip(names, seqs)):
      key = name if name not in reference_seqs else f"{name}_{i}"
      reference_seqs[key] = seq

    # write cleaned fasta
    write_msa(tmp_fasta_path, list(reference_seqs.keys()), list(reference_seqs.values()))

    # find hits
    hits = run_phmmer(query, tmp_fasta_path, cpu=phmmer_cpu)
    logger.info(f"Found {len(hits)}/{len(names)} in reference DB for query.")
    unaligned_seqs = {in_name: query}  # keep target sequence at the top
    found_names = {in_name}
    found_seqs = {query}

    # choose up to `size` hits
    chosen = hits if size is None else hits[: size - 1]

    for hit_name in chosen:
      hit_seq = reference_seqs.get(hit_name)
      if hit_seq is None:
        continue
      if (hit_name not in found_names) and (hit_seq not in found_seqs):
        found_names.add(hit_name)
        found_seqs.add(hit_seq)
        unaligned_seqs[hit_name] = hit_seq

  # generate alignment and return
  return align_mafft(unaligned_seqs, threads=mafft_threads)


def run_mmseqs2(
  seqs: Union[str, List[str]],
  output: str,
  database: str = "mmseqs2_uniref_env",
  use_filter: bool = True,
  use_templates: bool = False,
  pairing: Optional[str] = None,
) -> Tuple[List[str], Optional[List[str]]]:
  """
  Submits amino acid sequences to the ColabFold MMseqs2 API to generate multiple sequence alignments (MSAs),
  optionally downloading template structures. Results are written to the specified output directory, including:
  - One combined A3M file per input sequence (named `combined_{i}.a3m`)
  - Optional structure templates (if `use_templates=True` and supported)

  Parameters:
    seqs (str or List[str]):
        One or more amino acid sequences. If a single string is provided, it is treated as one sequence.
        Duplicates are automatically de-duplicated to reduce redundant API calls.

    output (str):
        Path to an output directory. If it exists, it will be deleted and recreated.

    database (str):
        MMseqs2 search database to use. Must be one of:
        - "mmseqs2_uniref_env" (environmental sequences + UniRef)
        - "mmseqs2_uniref" (UniRef only)

    use_filter (bool):
        Whether to apply diversity/length filtering to limit the size of the resulting MSA.
        Recommended for performance and downstream quality. If False, may yield larger but noisier MSAs.

    use_templates (bool):
        Whether to fetch structural templates for each input sequence using hits from the `pdb70.m8` file.
        Automatically disabled if `pairing` is set.

    pairing (str or None):
        If specified, activates MSA pairing mode. Must be one of:
        - "greedy": fast pairing using best bidirectional hits
        - "complete": exhaustive pairing of all hits
        - None: disables pairing (default)

        Note: Only one MSA is generated per pair using `pair.a3m` when pairing is enabled.
        If this is set, `use_templates` will be ignored.

  Returns:
    Tuple[List[str], Optional[List[str]]]:
      - a3m_lines: A list of strings, each representing the combined MSA (in A3M format) for each input sequence,
                   in the same order as provided. These are also written as `combined_{i}.a3m` files.
      - template_paths: If `use_templates=True`, returns a list of template directory paths (or None for sequences with no templates found).
                        Otherwise, returns None.

  Notes:
    - Internally deduplicates sequences but returns results in original input order.
    - Implements robust retry logic for ColabFold’s unstable API endpoints, including long-lived 502 errors.
    - Null bytes in A3M files are stripped to avoid downstream parsing issues.
    - Original code adapted from ColabFold: https://github.com/sokrypton/ColabFold/
    - Please cite ColabFold if using this in research: https://colabfold.mmseqs.com/
  """
  ### Constants
  ## Settings
  host_url = "https://api.colabfold.com"
  submission_endpoint = "ticket/pair" if pairing else "ticket/msa"
  headers = {}
  headers["User-Agent"] = USER_AGENT
  # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
  # "good practice to set connect timeouts to slightly larger than a multiple of 3"
  timeout = 6.02
  # Number of seconds between subsequent failed request
  request_delay = 15
  # The maximum error count to tolerate
  max_error_count = 50  # NOTE: This is intentionally set very high since the mmseqs2 API has this disgusting flaw where it will return a 502 error while it is compiling the output tar file once everything is complete. Extremely cringe!

  ## Other constants
  start = datetime.now()

  # set the mode
  assert database in ["mmseqs2_uniref_env", "mmseqs2_uniref"], 'database must be either "mmseqs2_uniref_env" or "mmseqs2_uniref"'
  if use_filter:
    mode = "env" if database == "mmseqs2_uniref_env" else "all"
  else:
    mode = "env-nofilter" if database == "mmseqs2_uniref_env" else "nofilter"

  if pairing:
    # greedy is default, complete was the previous behavior
    assert pairing in ["greedy", "complete"], 'pairing must be either "greedy", "complete", or None'
    if pairing == "greedy":
      mode = "pairgreedy"
    elif pairing == "complete":
      mode = "paircomplete"

  # define path
  if os.path.isdir(output):
    shutil.rmtree(output)
  os.mkdir(output)
  temp_dir = tempfile.mkdtemp()

  ### Functions
  def retry_request(method, url, json_mode=True, **kwargs):
    """Helper function for retrying requests until success or an error threshold is reached"""
    error_count = 0
    while error_count <= max_error_count:
      try:
        r = method(url, timeout=timeout, headers=headers, **kwargs)
        r.raise_for_status()
        if not json_mode:
          return r
        try:
          out = r.json()
        except Exception as _:
          raise Exception(f"Server didn't reply with json: {r.text}")

        if out["status"] in ["UNKNOWN", "RATELIMIT"]:
          raise Exception(f"Server failed to produce desired result (status: {out['status']})")
        elif out["status"] == "ERROR":
          raise Exception(
            "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later."
          )
        elif out["status"] == "MAINTENANCE":
          raise Exception("MMseqs2 API is undergoing maintenance. Please try again in a few minutes.")
        return out
      except Exception as e:
        error_count += 1
        logger.warning(f"Error contacting MSA server. Retrying... ({error_count}/{max_error_count})")
        logger.warning(f"Error: {e}")
        time.sleep(request_delay)
    raise Exception("Too many failed attempts at MSA generation. Please review your inputs or try again in a few hours.")

  ### Call mmseqs2 api
  ## Perform initial submission
  seqs = [seqs] if isinstance(seqs, str) else seqs
  if pairing:
    # In pairing mode, do not deduplicate; MMseqs2 uses positional headers (>1, >2, ...) for each input.
    seqs_unique = list(seqs)
    sequence_ids = list(range(1, len(seqs) + 1))
  else:
    seqs_unique = []
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    sequence_ids = [1 + seqs_unique.index(seq) for seq in seqs]

  query = ""
  for i, seq in enumerate(seqs, start=1):
    query += f">{i}\n{seq}\n"
  out = retry_request(requests.post, f"{host_url}/{submission_endpoint}", json_mode=True, data={"q": query, "mode": mode})

  ID = out["id"]
  logger.info(f"Successfully submitted mmseqs2 API request with job ID {ID}")

  ## Wait for job to finish
  # possible status' that won't trigger an exception in retry_request include ["RUNNING", "PENDING", "COMPLETE"]
  for i in range(50):
    out = retry_request(requests.get, f"{host_url}/ticket/{ID}", json_mode=True)
    if out["status"] == "COMPLETE":
      break
    logger.info(f"Checking status for MSA with ID {ID} in {request_delay}s (current: {out['status']}).")
    time.sleep(request_delay)

  ## Download results
  r = retry_request(requests.get, f"{host_url}/result/download/{ID}", json_mode=False, stream=True)

  # extract files
  with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
    tar.extractall(path=temp_dir)
  for file in ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m", "pair.a3m", "pdb70.m8"]:
    src_path = os.path.join(temp_dir, file)
    if os.path.exists(src_path):
      shutil.move(src_path, os.path.join(output, file))

  shutil.rmtree(temp_dir)

  if pairing:
    a3m_files = [f"{output}/pair.a3m"]
  else:
    a3m_files = [f"{output}/uniref.a3m"]
    if mode == "env":
      a3m_files.append(f"{output}/bfd.mgnify30.metaeuk30.smag30.a3m")

  ## Combine a3m lines
  a3m_lines = {seq_id: "" for seq_id in sequence_ids}
  for a3m_file in a3m_files:
    update_M, seq_id = True, None
    for line in open(a3m_file, "r"):
      if len(line) > 0:
        if "\x00" in line:
          line = line.replace("\x00", "")
          update_M = True
        if line.startswith(">") and update_M:
          seq_id = int(line[1:].rstrip())
          update_M = False
        a3m_lines[seq_id] += line

  a3m_lines = [a3m_lines[n] for n in sequence_ids]

  # remove null bytes from all files including pair files
  for fname in ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m", "pair.a3m"]:
    if os.path.exists(f"{output}/{fname}"):
      with open(f"{output}/{fname}", "r") as fin, open(f"{output}/{fname}.tmp", "w") as fout:
        for line in fin:
          fout.write(line.replace("\x00", ""))
      shutil.move(f"{output}/{fname}.tmp", f"{output}/{fname}")

  # write combined MSAs too
  for i, msa in enumerate(a3m_lines, start=1):
    with open(f"{output}/combined_{i}.a3m", "w") as f:
      f.write(msa)

  ## fetch templates if applicable
  if not pairing and use_templates:
    # keys are input sequence IDs and values are lists of PDB IDs
    templates = {}
    for line in open(f"{output}/pdb70.m8", "r"):
      p = line.rstrip().split()
      seq_id, pdb, qid, e_value = p[0], p[1], p[2], p[10]
      seq_id = int(seq_id)
      if seq_id not in templates:
        templates[seq_id] = []
      templates[seq_id].append(pdb)

    # keys are input sequence IDs and values file paths for the directory containing the templates
    template_paths = {seq_id: None for seq_id in sequence_ids}
    for seq_id, pdb_ids in templates.items():
      template_fpath = f"{output}/templates_{seq_id}"
      os.makedirs(template_fpath, exist_ok=True)
      r = retry_request(requests.get, f"{host_url}/template/{','.join(pdb_ids[:20])}", json_mode=False, stream=True)
      with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
        tar.extractall(path=template_fpath, filter="data")
      os.remove(f"{template_fpath}/pdb70_a3m.ffindex")
      template_paths[seq_id] = template_fpath

    # log missing templates
    for seq_id, template_fpath in template_paths.items():
      if template_fpath is None:
        logger.warning(f"No templates found for {seqs_unique[seq_id]}")

  logger.info(f"Finished generating MSA, took {datetime.now()-start}")

  if not pairing and use_templates:
    return a3m_lines, template_paths.values()
  else:
    return a3m_lines, None


def run_mmseqs2_modes(
  seq: Union[str, List[str]],
  output: str,
  cov: int = 50,
  id: int = 90,
  max_msa: int = 2048,
  mode: str = "unpaired_paired",
):
  """
  Generate a multiple sequence alignment (MSA) for the given sequence(s)
  using Colabfold's API. Key difference between this function and
  run_mmseqs2 is that this function supports different modes.
  The final a3m and most useful a3m file will be written as "output/final.a3m".
  Code originally adapted from: https://github.com/sokrypton/ColabFold/

  Parameters:
    seq: Sequence(s) to generate the MSA for. If a list of sequences is
      provided, they will be considered as a single protein for the MSA.
    output: Output directory path, will overwrite existing results.
    cov: Coverage of the MSA
    id: Identity threshold for the MSA
    max_msa: Maximum number of sequences in the MSA
    mode: Mode to run the MSA generation in. Must be in ``["unpaired", "paired", "unpaired_paired"]``

  """
  # Check if HH-suite is installed and available
  hhfilter_path = shutil.which("hhfilter")
  assert hhfilter_path is not None, (
    "HH-suite not found. Please ensure it is installed and available in your PATH. "
    "For installation instructions, visit: https://github.com/soedinglab/hh-suite"
  )
  # Validate the mode
  assert mode in ["unpaired", "paired", "unpaired_paired"], "Invalid mode"

  seqs = [seq] if isinstance(seq, str) else seq
  # Collapse homooligomeric sequences
  counts = Counter(seqs)
  u_seqs = list(counts.keys())
  u_nums = list(counts.values())

  # Expand homooligomeric sequences
  first_seq = "/".join(sum([[x] * n for x, n in zip(u_seqs, u_nums)], []))
  msa = [first_seq]

  # Handle paired MSA if applicable
  if mode in ["paired", "unpaired_paired"] and len(u_seqs) > 1:
    print("Getting paired MSA")
    out_paired, _ = run_mmseqs2(u_seqs, output, pairing="greedy")
    headers, sequences = [], []
    for a3m_lines in out_paired:
      n = -1
      for line in a3m_lines.split("\n"):
        if len(line) > 0:
          if line.startswith(">"):
            n += 1
            if len(headers) < (n + 1):
              headers.append([])
              sequences.append([])
            headers[n].append(line)
          else:
            sequences[n].append(line)
    # Filter MSA
    paired_in_fpath = os.path.join(output, "paired_in.a3m")
    paired_out_fpath = os.path.join(output, "paired_out.a3m")
    with open(paired_in_fpath, "w") as f:
      for n, sequence in enumerate(sequences):
        f.write(f">n{n}\n{''.join(sequence)}\n")

    os.system(f"hhfilter -i {paired_in_fpath} -id {id} -cov {cov} -o {paired_out_fpath}")

    with open(paired_out_fpath, "r") as f:
      for line in f:
        if line.startswith(">"):
          n = int(line[2:])
          xs = sequences[n]
          # Expand homooligomeric sequences
          xs = ["/".join([x] * num) for x, num in zip(xs, u_nums)]
          msa.append("/".join(xs))

  # Handle unpaired MSA if applicable
  if len(msa) < max_msa and (mode in ["unpaired", "unpaired_paired"] or len(u_seqs) == 1):
    print("Getting unpaired MSA")
    out, _ = run_mmseqs2(u_seqs, output, pairing=None)
    Ls = [len(seq) for seq in u_seqs]
    sub_idx = []
    sub_msa = []
    sub_msa_num = 0
    for n, a3m_lines in enumerate(out):
      sub_msa.append([])
      in_fpath = os.path.join(output, f"in_{n}.a3m")
      out_fpath = os.path.join(output, f"out_{n}.a3m")
      with open(in_fpath, "w") as f:
        f.write(a3m_lines)

      # Filter
      os.system(f"hhfilter -i {in_fpath} -id {id} -cov {cov} -o {out_fpath}")

      with open(out_fpath, "r") as f:
        for line in f:
          if not line.startswith(">"):
            xs = ["-" * l for l in Ls]
            xs[n] = line.rstrip()
            # Expand homooligomeric sequences
            xs = ["/".join([x] * num) for x, num in zip(xs, u_nums)]
            sub_msa[-1].append("/".join(xs))
            sub_msa_num += 1
        sub_idx.append(list(range(len(sub_msa[-1]))))

    while len(msa) < max_msa and sub_msa_num > 0:
      for n in range(len(sub_idx)):
        if len(sub_idx[n]) > 0:
          msa.append(sub_msa[n][sub_idx[n].pop(0)])
          sub_msa_num -= 1
        if len(msa) == max_msa:
          break

  # Write final MSA to file
  with open(os.path.join(output, "final.a3m"), "w") as f:
    for n, sequence in enumerate(msa):
      f.write(f">n{n}\n{sequence}\n")
