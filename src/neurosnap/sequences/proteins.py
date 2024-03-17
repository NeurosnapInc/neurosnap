"""
Provides functions and classes related to processing protein sequence data.
"""
### IMPORTS ###
import os, requests, time, tarfile, re, io, shutil


### CONSTANTS ###
## Standard amino acids excluding the unknown character ("X")
## Currently excludes
# B | asx | asparagine/aspartic acid
# Z | glx | glutamine/glutamic acid
# J | xle | leucine/isoleucine
# O | pyl | pyrrolysine
# U | sec | selenocysteine
# X | UNK | unknown codon
# * | TRM | termination codon
STANDARD_AAs = "ARNDCEQGHILKMFPSTWYV"

## Full standard amino acids table
AAs_FULL_TABLE = [
  ['Ala', 'A', 'alanine'],
  ['Arg', 'R', 'arginine'],
  ['Asn', 'N', 'asparagine'],
  ['Asp', 'D', 'aspartic acid'],
  ['Cys', 'C', 'cysteine'],
  ['Gln', 'Q', 'glutamine'],
  ['Glu', 'E', 'glutamic acid'],
  ['Gly', 'G', 'glycine'],
  ['His', 'H', 'histidine'],
  ['Ile', 'I', 'isoleucine'],
  ['Leu', 'L', 'leucine'],
  ['Lys', 'K', 'lysine'],
  ['Met', 'M', 'methionine'],
  ['Phe', 'F', 'phenylalanine'],
  ['Pro', 'P', 'proline'],
  ['Ser', 'S', 'serine'],
  ['Thr', 'T', 'threonine'],
  ['Trp', 'W', 'tryptophan'],
  ['Tyr', 'Y', 'tyrosine'],
  ['Val', 'V', 'valine'],
  ['Pyl', 'O', 'pyrrolysine'],
  ['Sec', 'U', 'selenocysteine'],
]

### CLASSES ###
class GetAA:
  """
  -------------------------------------------------------
  Efficiently get any amino acid using either their 1 letter code,
  3 letter abbreviation, or full name.
  -------------------------------------------------------
  Parameters:
    seq..........: Amino acid sequence for protein to generate an MSA of (str)
    output.......: Output directory path, will overwrite existing results (str)
    database.....: Choose the database to use, must be either "mmseqs2_uniref_env" or "mmseqs2_uniref" (str)
    use_filter...: Enables the diversity and msa filtering steps that ensures the MSA will not become enormously large (described in manuscript methods section of ColabFold paper) (bool)
    use_templates: Download templates as well using the mmseqs2 results (bool)
    pairing......: Can be set to either "greedy", "complete", or None for no pairing (str)
  """


### FUNCTIONS ###
def run_mmseqs2(seq, output, database="mmseqs2_uniref_env", use_filter=True, use_templates=False, pairing=None):
  """
  -------------------------------------------------------
  Generate an a3m MSA using the ColabFold API. Will write
  all results to the output directory including templates,
  MSAs, and accompanying files.
  Code originally adapted from: https://github.com/sokrypton/ColabFold/
  -------------------------------------------------------
  Parameters:
    seq..........: Amino acid sequence for protein to generate an MSA of (str)
    output.......: Output directory path, will overwrite existing results (str)
    database.....: Choose the database to use, must be either "mmseqs2_uniref_env" or "mmseqs2_uniref" (str)
    use_filter...: Enables the diversity and msa filtering steps that ensures the MSA will not become enormously large (described in manuscript methods section of ColabFold paper) (bool)
    use_templates: Download templates as well using the mmseqs2 results (bool)
    pairing......: Can be set to either "greedy", "complete", or None for no pairing (str)
  """
  # API settings
  user_agent = "Neurosnap-OSS-Tools/v1" #TODO: Use actual version
  host_url = "https://api.colabfold.com"
  submission_endpoint = "ticket/pair" if pairing else "ticket/msa"
  headers = {}
  headers['User-Agent'] = user_agent
  timeout = 6.02

  # set the mode
  assert database in ["mmseqs2_uniref_env", "mmseqs2_uniref"], ValueError('database must be either "mmseqs2_uniref_env" or "mmseqs2_uniref"')
  if use_filter:
    mode = "env" if database == "mmseqs2_uniref_env" else "all"
  else:
    mode = "env-nofilter" if database == "mmseqs2_uniref_env" else "nofilter"

  if pairing:
    use_templates = False
    # greedy is default, complete was the previous behavior
    assert pairing in ["greedy", "complete"], ValueError('pairing must be either "greedy", "complete", or None')
    if pairing == "greedy":
      mode = "pairgreedy"
    elif pairing == "complete":
      mode = "paircomplete"

  # define path
  if os.path.isdir(output):
    shutil.rmtree(output)
  os.mkdir(output)

  def submit(seq, mode):
    query = f">query\n{seq}\n"
    while True:
      error_count = 0
      try:
        r = requests.post(f'{host_url}/{submission_endpoint}', data={'q': query, 'mode':mode}, timeout=timeout, headers=headers)
      except requests.exceptions.Timeout:
        print("Timeout while submitting to MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        print(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        print(f"Error: {e}")
        time.sleep(timeout)
        if error_count > 5:
          raise
        continue
      break

    try:
      out = r.json()
    except ValueError:
      print(f"Server didn't reply with json: {r.text}")
      out = {"status":"ERROR"}
    return out

  def status(ID):
    while True:
      error_count = 0
      try:
        r = requests.get(f'{host_url}/ticket/{ID}', timeout=timeout, headers=headers)
      except requests.exceptions.Timeout:
        print("Timeout while fetching status from MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        print(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        print(f"Error: {e}")
        time.sleep(timeout)
        if error_count > 5:
          raise
        continue
      break
    try:
      out = r.json()
    except ValueError:
      print(f"Server didn't reply with json: {r.text}")
      out = {"status":"ERROR"}
    return out

  ## call mmseqs2 api
  # Resubmit job until it goes through
  out = submit(seq, mode)
  while out["status"] in ["UNKNOWN", "RATELIMIT"]:
    print(f"Sleeping for {timeout}s. Reason: {out['status']}")
    # resubmit
    time.sleep(timeout)
    out = submit(seq, mode)

  if out["status"] == "ERROR":
    raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

  if out["status"] == "MAINTENANCE":
    raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

  # wait for job to finish
  ID = out["id"]
  while out["status"] in ["UNKNOWN","RUNNING","PENDING"]:
    print(f"Sleeping for {timeout}s. Reason: {out['status']}")
    time.sleep(timeout)
    out = status(ID)

  if out["status"] == "ERROR" or out["status"] != "COMPLETE":
    print(out)
    raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

  # Download results
  error_count = 0
  while True:
    try:
      r = requests.get(f'{host_url}/result/download/{ID}', stream=True, timeout=timeout, headers=headers)
    except requests.exceptions.Timeout:
      print("Timeout while fetching result from MSA server. Retrying...")
      continue
    except Exception as e:
      error_count += 1
      print(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
      print(f"Error: {e}")
      time.sleep(timeout)
      if error_count > 5:
        raise
      continue
    break
  # extract files
  with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
    tar.extractall(path=output, filter="data")

  # remove null bytes from all files including pair files
  for fname in os.listdir(output):
    if fname in ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m", "pair.a3m"]:
      with open(f"{output}/{fname}", "r") as fin:
        with open(f"{output}/{fname}.tmp", "w") as fout:
          for line in fin:
            fout.write(line.replace("\x00",""))
      shutil.move(f"{output}/{fname}.tmp", f"{output}/{fname}")

  # templates
  if use_templates:
    templates = []
    # .m8 file description: https://linsalrob.github.io/ComputationalGenomicsManual/SequenceFileFormats/#blast-m8
    # print("seq\tpdb\tcid\tevalue")
    for line in open(f"{output}/pdb70.m8","r"):
      line = line.rstrip().split()
      name, pdb, qid, e_value = line[0], line[1], line[2], line[10]
      templates.append(pdb)
      # if len(templates) <= 20:
      #  print(f"{name}\t{pdb}\t{qid}\t{e_value}")

    template_path = f"{output}/templates"
    os.mkdir(f"{output}/templates")
    for template in templates:
      error_count = 0
      while True:
        try:
          r = requests.get(f"{host_url}/template/{template[:20]}", stream=True, timeout=timeout, headers=headers)
        except requests.exceptions.Timeout:
          print("Timeout while submitting to template server. Retrying...")
          continue
        except Exception as e:
          error_count += 1
          print(f"Error while fetching result from template server. Retrying... ({error_count}/5)")
          print(f"Error: {e}")
          time.sleep(timeout)
          if error_count > 5:
            raise
          continue
        break
      with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
        tar.extractall(path=template_path, filter="data")


def read_msa(input_fasta, size=float("inf"), allow_chars="", drop_chars="", remove_chars="*", uppercase=True):
  """
  -------------------------------------------------------
  Reads an MSA, a3m, or fasta file and returns an array of names and seqs.
  -------------------------------------------------------
  Parameters:
    input_fasta.: Path to read input a3m file, fasta as a raw string, or a file-handle like object to read (str|io.TextIOBase)
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
  allow_chars = allow_chars.replace("-", "\-")
  drop_chars = drop_chars.replace("-", "\-")
  remove_chars = remove_chars.replace("-", "\-")
  char_whitelist = STANDARD_AAs + allow_chars + drop_chars + remove_chars

  if isinstance(input_fasta, str):
    if os.path.exists(input_fasta):
      f = open(input_fasta)
    else:
      f = io.StringIO(input_fasta)
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