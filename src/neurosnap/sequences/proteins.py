"""
Provides functions and classes related to processing protein sequence data.
"""
### IMPORTS ###
import os, requests, time, tarfile, re, io


### CONSTANTS ###
# Standard amino acids excluding the unknown character ("X")
STANDARD_AAs = "ARNDBCEQZGHILKMFPSTWYV"


### FUNCTIONS ###
def generate_msa(seq, output_path, mode="all", max_retries=10):
  """
  -------------------------------------------------------
  Generate an a3m MSA using the ColabFold API.
  -------------------------------------------------------
  Parameters:
    seq........: Amino acid sequence for protein to generate an MSA of (str)
    output_path: Path to where the MSA will be downloaded, should end with .a3m (str)
    mode.......: Supports modes like "all" or "env" (str)
    max_retries: Maximum number of retries (int)
  """
  print("[*] This function uses ColabFold mmseqs2 API (https://colabfold.mmseqs.com/). This API is made freely available so consider citing the authors (https://doi.org/10.1038/s41592-022-01488-1).")
  def get_status(ID):
    r = requests.get(f'https://api.colabfold.com/ticket/{ID}')
    try:
      out = r.json()
    except ValueError:
      print(f"Server didn't reply with json: {res.text}")
      out = {"status":"ERROR"}
    return out

  query = f">input_seq\n{seq}"
  ID = None
  while ID is None:
    r = requests.post(f'https://api.colabfold.com/ticket/msa', data={'q':query,'mode': mode}, timeout=6.02)
    data = r.json()
    if "id" in data:
      ID = data["id"]
      print(data)
    else:
      print(f"[-] Failed {data['status']} {proxy}")
      print(data)
      proxy_index += 1

  for _ in range(max_retries):
    status = get_status(ID)["status"]
    if status == "COMPLETE":
      break
    elif status == "ERROR":
      raise
    else:
      time.sleep(5)
  
  # download
  error_count = 0
  while True:
    try:
      r = requests.get(f'https://api.colabfold.com/result/download/{ID}', timeout=6.02)
    except requests.exceptions.Timeout:
      print("Timeout while fetching result from MSA server. Retrying...")
      continue
    except Exception as e:
      error_count += 1
      print(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
      print(f"Error: {e}")
      time.sleep(5)
      if error_count > max_retries:
        raise
      continue
    break
  # write tar file
  tar_path = f"{ID}.tar.gz"
  with open(tar_path, "wb") as f:
    f.write(r.content)
  # extract a3m only
  with tarfile.open(tar_path) as tar:
    for member in tar.getmembers():
      if member.name.endswith(".a3m"):
        with open(output_path, "wb") as f:
          f.write(tar.extractfile(member).read())
        break
  # clean
  os.remove(tar_path)


def read_msa(input_fasta, size=float("inf"), allow_chars="", drop_chars="", remove_chars="*", uppercase=True):
  """
  -------------------------------------------------------
  Reads an MSA, a3m, or fasta file and returns an array of names and seqs.
  -------------------------------------------------------
  Parameters:
    input_fasta.: Path to read input a3m file or a file-handle like object to read (str|io.TextIOBase)
    size........: Number of rows to read (int)
    allow_chars.: Sequences that contain characters not included within STANDARD_AAs+allow_chars will throw an exception (str)
    drop_chars..: Drop sequences that contain these characters e.g., "-X" (str)
    remove_chars: Removes these characters from sequences e.g., "*-X" (str)
    uppercase...: Converts all amino acid chars to uppercase when True (bool)
  Returns:
    names: List of proteins names from the a3m file including gaps (list<str>)
    seqs.: List of proteins sequences from the a3m file including gaps (list<str>)
  """
  names = []
  seqs = []
  char_whitelist = STANDARD_AAs + allow_chars + drop_chars + remove_chars

  if isinstance(input_fasta, str):
    if os.path.exists(input_fasta):
      f = open(input_fasta)
  elif isinstance(input_fasta, io.TextIOBase):
    f = input_fasta
  else:
    raise ValueError(f"Invalid input for input_fasta, {type(input_fasta)} is not a valid type.")

  for i, line in enumerate(f, start=1):
    line = line.strip()
    if line:
      if line.startswith(">"):
        if seqs and seqs[-1] == "":
          raise ValueError(f"Invalid MSA/fasta. Header {names[-1]} is missing a sequence.")
        if len(seqs) >= size+1:
          break
        match = re.search(r">([\w\-_]*)", line)
        assert match is not None, ValueError(f"Invalid MSA/fasta. {line} is not a valid header.")
        name = match.group(1)
        assert len(name), ValueError(f"Invalid MSA/fasta. line {i} has an empty header.")
        names.append(name)
        seqs.append("")
      else:
        if uppercase:
          line = line.upper()
        # remove whitespace and remove_chars
        if remove_chars:
          line = re.sub(f"[{remove_chars}\s]", "", line)
        # drop chars
        if drop_chars:
          match = re.search(f"[{drop_chars}]", line)
          if match is not None:
            names.pop()
            seqs.pop()
            continue
        
        match = re.search(f"^[{STANDARD_AAs+allow_chars}]*$", line)
        if match is None:
          raise ValueError(f"Sequence on line {i} contains an invalid character. Please specify whether you would like drop or replace characters in sequences like these. Sequence='{line}'")
        seqs[-1] += line

  f.close()
  assert len(names) == len(seqs), ValueError("Invalid MSA/fasta. The number sequences and headers found do not match.")
  return names, seqs


def write_msa(output_path, names, seqs):
  """
  -------------------------------------------------------
  Writes an MSA, a3m, or fasta to a file.
  Makes no assumptions about the validity of names or
  sequences. Will throw an exception if len(names) != len(seqs)
  -------------------------------------------------------
  Parameters:
    output_path: Path to output file to write, will overwrite existing files (str)
    names......: List of proteins names from the file (list<str>)
    seqs.......: List of proteins sequences from the file (list<str>)
  """
  assert len(names) == len(seqs), ValueError("The number of names and sequences do not match.")
  with open(output_path, "w") as f:
    for name, seq in zip(names, seqs):
      f.write(f">{name}\n{seq}\n")


def pad_seqs(seqs, char="-", truncate=False):
  """
  -------------------------------------------------------
  Pads all sequences to the longest sequences length
  using a character from the right side.
  -------------------------------------------------------
  Parameters:
    seqs......: List of sequences to pad (list<str>)
    chars.....: The character to perform the padding with, default is "-" (str)
    truncate..: When set to True will truncate all sequences to the length of the first, set to integer to truncate sequence to that length (bool/int)
  Returns:
    seqs_padded: The padded sequences (list<str>)
  """
  if truncate is True:
    longest_seq = len(seqs[0])
  elif type(truncate) is int:
    assert truncate >= 1, ValueError("truncate must be either a boolean value or an integer greater than or equal to 1.")
    longest_seq = truncate
  else:
    longest_seq = max(len(x) for x in seqs)

  for i, seq in enumerate(seqs):
    seqs[i] = seq.ljust(longest_seq, "-")
    seqs[i] = seqs[i][:longest_seq]
  return seqs


def get_seqid(seq1, seq2, align=False):
  """
  -------------------------------------------------------
  Calculate the pairwise sequence identity of two same length sequences or alignments.
  -------------------------------------------------------
  Parameters:
    seq1: The 1st sequence / aligned sequence. (str)
    seq2: The 2nd sequence / aligned sequence. (str)
  Returns:
    seq_id: The pairwise sequence identity. Will return None  (float)
  """
  assert len(seq1) == len(seq2), ValueError("Sequences are not the same length.")
  num_matches = 0
  for a,b in zip(seq1, seq2):
    if a == b:
      num_matches += 1
  return 100 * num_matches / len(seq1)