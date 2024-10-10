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

import requests
from Bio import SearchIO

from neurosnap.log import logger
from neurosnap.api import USER_AGENT
from neurosnap.protein import STANDARD_AAs


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
  allow_chars = allow_chars.replace("-", "\\-")
  drop_chars = drop_chars.replace("-", "\\-")
  remove_chars = remove_chars.replace("-", "\\-")

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
        match = re.search(r"^>([\w-]*)", line)
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
          line = re.sub(f"[{remove_chars}\\s]", "", line)
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


def get_seqid(seq1, seq2):
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


def align_mafft(seqs, ep=0.0, op=1.53):
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
    except:  # noqa: E722
      logger.error(align_out.stderr)
      raise Exception()

  return read_msa(io.StringIO(align_out.stdout.decode("utf-8")), allow_chars="-")


def run_phmmer_mafft(query, ref_db_path, size=float("inf"), in_name="input_sequence"):
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
  return align_mafft(unaligned_seqs)


def run_mmseqs2(seqs, output, database="mmseqs2_uniref_env", use_filter=True, use_templates=False, pairing=None):
  """
  -------------------------------------------------------
  Generate an a3m MSA using the ColabFold API. Will write
  all results to the output directory including templates,
  MSAs, and accompanying files.
  Code originally adapted from: https://github.com/sokrypton/ColabFold/
  -------------------------------------------------------
  Parameters:
    seqs..........: Amino acid sequences for protein to generate an MSA of (str)
    output.......: Output directory path, will overwrite existing results (str)
    database.....: Choose the database to use, must be either "mmseqs2_uniref_env" or "mmseqs2_uniref" (str)
    use_filter...: Enables the diversity and msa filtering steps that ensures the MSA will not become enormously large (described in manuscript methods section of ColabFold paper) (bool)
    use_templates: Download templates as well using the mmseqs2 results (bool)
    pairing......: Can be set to either "greedy", "complete", or None for no pairing (str)
  """
  print("""The MMseqs2 webserver used to generate this MSA is provided as a free service. Please help keep the authors of this service keep things free by appropriately citing them as follows:
    @article{Mirdita2019,
      title        = {{MMseqs2 desktop and local web server app for fast, interactive sequence searches}},
      author       = {Mirdita, Milot and Steinegger, Martin and S{"{o}}ding, Johannes},
      year         = 2019,
      journal      = {Bioinformatics},
      volume       = 35,
      number       = 16,
      pages        = {2856--2858},
      doi          = {10.1093/bioinformatics/bty1057},
      pmid         = 30615063,
      comment      = {MMseqs2 search server}
    }
    @article{Mirdita2017,
      title        = {{Uniclust databases of clustered and deeply annotated protein sequences and alignments}},
      author       = {Mirdita, Milot and von den Driesch, Lars and Galiez, Clovis and Martin, Maria J. and S{"{o}}ding, Johannes and Steinegger, Martin},
      year         = 2017,
      journal      = {Nucleic Acids Res.},
      volume       = 45,
      number       = {D1},
      pages        = {D170--D176},
      doi          = {10.1093/nar/gkw1081},
      pmid         = 27899574,
      comment      = {Uniclust30/UniRef30 database}
    }
    @article{Mitchell2019,
      title        = {{MGnify: the microbiome analysis resource in 2020}},
      author       = {Mitchell, Alex L and Almeida, Alexandre and Beracochea, Martin and Boland, Miguel and Burgin, Josephine and Cochrane, Guy and Crusoe, Michael R and Kale, Varsha and Potter, Simon C and Richardson, Lorna J and Sakharova, Ekaterina and Scheremetjew, Maxim and Korobeynikov, Anton and Shlemov, Alex and Kunyavskaya, Olga and Lapidus, Alla and Finn, Robert D},
      year         = 2019,
      journal      = {Nucleic Acids Res.},
      doi          = {10.1093/nar/gkz1035},
      comment      = {MGnify database}
    }
    @article{Mirdita2022,
      title        = {{ColabFold: making protein folding accessible to all}},
      author       = {Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
      year         = 2022,
      journal      = {Nature Methods},
      doi          = {10.1038/s41592-022-01488-1},
      comment      = {ColabFold API}
    }
  """)
  # API settings
  host_url = "https://api.colabfold.com"
  submission_endpoint = "ticket/pair" if pairing else "ticket/msa"
  headers = {}
  headers["User-Agent"] = USER_AGENT
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

  def submit(seqs, mode, N=101):
    n, query = N, ""
    for seq in seqs:
      query += f">{n}\n{seq}\n"
      n += 1
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
  seqs = [seqs] if isinstance(seqs, str) else seqs
  seqs_unique = []
  [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
  Ms = [101 + seqs_unique.index(seq) for seq in seqs]

  out = submit(seqs_unique, mode)
  while out["status"] in ["UNKNOWN", "RATELIMIT"]:
    print(f"Sleeping for {timeout}s. Reason: {out['status']}")
    # resubmit
    time.sleep(timeout)
    out = submit(seqs_unique, mode)

  if out["status"] == "ERROR":
    raise Exception('MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

  if out["status"] == "MAINTENANCE":
    raise Exception('MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

  # wait for job to finish
  ID = out["id"]
  while out["status"] in ["UNKNOWN","RUNNING","PENDING"]:
    print(f"Sleeping for {timeout}s. Reason: {out['status']}")
    time.sleep(timeout)
    out = status(ID)

  if out["status"] == "ERROR" or out["status"] != "COMPLETE":
    print(out)
    raise Exception('MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

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

  if pairing:
    a3m_files = [f"{output}/pair.a3m"]
  else:
    a3m_files = [f"{output}/uniref.a3m"]
    if mode == "env": a3m_files.append(f"{output}/bfd.mgnify30.metaeuk30.smag30.a3m")
  
   # gather a3m lines
  a3m_lines = {}
  for a3m_file in a3m_files:
    update_M,M = True,None
    for line in open(a3m_file,"r"):
      if len(line) > 0:
        if "\x00" in line:
          line = line.replace("\x00","")
          update_M = True
        if line.startswith(">") and update_M:
          M = int(line[1:].rstrip())
          update_M = False
          if M not in a3m_lines: a3m_lines[M] = []
        a3m_lines[M].append(line)
  
  a3m_lines = ["".join(a3m_lines[n]) for n in Ms]
 
  # remove null bytes from all files including pair files
  for fname in os.listdir(output):
    if fname in ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m", "pair.a3m"]:
      with open(f"{output}/{fname}", "r") as fin:
        with open(f"{output}/{fname}.tmp", "w") as fout:
          for line in fin:
            fout.write(line.replace("\x00", ""))
      shutil.move(f"{output}/{fname}.tmp", f"{output}/{fname}")


  if pairing is None:
    # Concatenate to create combined file
    with open(f"{output}/combined.a3m", "w") as fout:
      with open(f"{output}/uniref.a3m") as f:
          for line in f:
              fout.write(line)

      with open(f"{output}/bfd.mgnify30.metaeuk30.smag30.a3m") as f:
        # skip first two lines
        f.readline()
        f.readline()
        for line in f:
          fout.write(line)
  
  # templates
  if use_templates:
    templates = {}
    #print("seq\tpdb\tcid\tevalue")
    for line in open(f"{output}/pdb70.m8","r"):
      p = line.rstrip().split()
      M,pdb,qid,e_value = p[0],p[1],p[2],p[10]
      M = int(M)
      if M not in templates: templates[M] = []
      templates[M].append(pdb)
      #if len(templates[M]) <= 20:
      #  print(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}")

    template_paths = {}
    for k,TMPL in templates.items():
      TMPL_PATH = f"{output}/templates_{k}"
      if not os.path.isdir(TMPL_PATH):
        os.mkdir(TMPL_PATH)
        TMPL_LINE = ",".join(TMPL[:20])
        response = None
        while True:
          error_count = 0
          try:
            # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
            # "good practice to set connect timeouts to slightly larger than a multiple of 3"
            response = requests.get(f"{host_url}/template/{TMPL_LINE}", stream=True, timeout=6.02, headers=headers)
          except requests.exceptions.Timeout:
            logger.warning("Timeout while submitting to template server. Retrying...")
            continue
          except Exception as e:
            error_count += 1
            logger.warning(f"Error while fetching result from template server. Retrying... ({error_count}/5)")
            logger.warning(f"Error: {e}")
            time.sleep(5)
            if error_count > 5:
              raise
            continue
          break
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
          tar.extractall(path=TMPL_PATH, filter="data")
        os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
        with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
          f.write("")
      template_paths[k] = TMPL_PATH
  template_paths_ = []
  for n in Ms:
    if n not in template_paths:
      template_paths_.append(None)
      #print(f"{n-N}\tno_templates_found")
    else:
      template_paths_.append(template_paths[n])
  template_paths = template_paths_

  return (a3m_lines, template_paths) if use_templates else a3m_lines