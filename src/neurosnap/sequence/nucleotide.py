"""
Provides functions and classes related to processing nucleotide data.
"""

import gzip
import random
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Optional, TextIO, Tuple, Union

from neurosnap.log import logger


def generate_random_rna_sequence(
  length: int = 100,
  gc_content: int = 50,
  *,
  add_start_codon: bool = True,
  add_stop_codon: bool = True,
  add_5_utr: bool = False,
  add_3_utr: bool = False,
  poly_a_tail_length: int = 0,
  stem_loops: int = 0,
  include_pseudouridine: bool = False,
  rng: random.Random | None = None,
) -> str:
  """Generate one RNA sequence with optional transcript-like features.

  ``length`` includes all selected features. Randomized positions are adjusted
  toward the requested GC percentage; fixed codons, motifs, and poly-A can
  prevent the complete sequence from reaching that percentage.

  Raises:
      ValueError: If numeric settings are invalid or features exceed length.
  """
  if isinstance(length, bool) or not isinstance(length, int) or length < 1:
    raise ValueError("Sequence length must be a positive whole number.")
  if isinstance(gc_content, bool) or not isinstance(gc_content, int) or not 0 <= gc_content <= 100:
    raise ValueError("GC content must be a whole percentage from 0 to 100.")
  if isinstance(poly_a_tail_length, bool) or not isinstance(poly_a_tail_length, int) or poly_a_tail_length < 0:
    raise ValueError("Poly-A tail length must be a non-negative whole number.")
  if isinstance(stem_loops, bool) or not isinstance(stem_loops, int) or stem_loops < 0:
    raise ValueError("Stem-loop count must be a non-negative whole number.")

  source = rng if rng is not None else random.Random()
  weights = [(100 - gc_content) / 2, (100 - gc_content) / 2, gc_content / 2, gc_content / 2]
  alphabet = "AUGC"
  prefix = ""
  if add_5_utr:
    prefix += "".join(source.choices(alphabet, weights=weights, k=24)) + "GCCACC"
  if add_start_codon:
    prefix += "AUG"
  suffix = source.choice(("UAA", "UAG", "UGA")) if add_stop_codon else ""
  if add_3_utr:
    suffix += "".join(source.choices(alphabet, weights=weights, k=30))
  suffix += "A" * poly_a_tail_length

  hairpins = []
  pairs = str.maketrans("AUGC", "UACG")
  for _ in range(stem_loops):
    stem = "".join(source.choices(alphabet, weights=weights, k=4))
    loop = "".join(source.choices(alphabet, weights=weights, k=4))
    hairpins.append(stem + loop + stem.translate(pairs)[::-1])
  fixed_body = "".join(hairpins)
  filler_length = length - len(prefix) - len(fixed_body) - len(suffix)
  if filler_length < 0:
    raise ValueError("Sequence length is too short for the selected RNA features.")

  fixed_gc = (prefix + fixed_body + suffix).count("G") + (prefix + fixed_body + suffix).count("C")
  gc_bases = max(0, min(filler_length, round(length * gc_content / 100) - fixed_gc))
  filler = [source.choice("GC") for _ in range(gc_bases)]
  filler.extend(source.choice("AU") for _ in range(filler_length - gc_bases))
  source.shuffle(filler)
  if include_pseudouridine:
    uridines = [index for index, base in enumerate(filler) if base == "U"]
    if uridines:
      for index in source.sample(uridines, max(1, round(len(uridines) * 0.1))):
        filler[index] = "Ψ"

  return prefix + fixed_body + "".join(filler) + suffix


def get_reverse_complement(seq: str) -> str:
  """
  Generate the complementary strand of a DNA or RNA sequence in reverse order.

  Args:
      seq (str): A string representing the nucleotide sequence.
                  Valid characters are 'A', 'T', 'C', 'G' for DNA and 'A', 'U', 'C', 'G' for RNA.

  Returns:
      str: A string representing the reverse complementary strand of the input sequence.

  Raises:
      KeyError: If the input sequence contains invalid nucleotide characters.
  """
  complement = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}
  return "".join([complement[base] for base in seq[::-1]])


def split_interleaved_fastq(
  fn_in: Union[str, Path],
  output_dir: Union[str, Path],
  *,
  preserve_identifier_names: bool = False,
) -> Tuple[Path, Path]:
  """
  Split an interleaved FASTQ into left/right FASTQ files.

  Assumes pairs are adjacent (left read followed by right read) and rewrites
  headers as "@<index>/1" and "@<index>/2".

  Supports gzip-compressed inputs with filenames ending in ".fastq.gz" or ".fq.gz".
  Compression is detected by filename and streamed during read.

  Parameters:
      fn_in: Path to the interleaved FASTQ file.
      output_dir: Directory to write outputs into.
      preserve_identifier_names: If True, preserve the input read identifiers
          (normalizing mate suffix to "/1" or "/2"). If False, rewrite
          identifiers as "@<index>/1" and "@<index>/2".

  Returns:
      Tuple[Path, Path]: Paths to the left and right FASTQ output files.
  """
  fn_in_path = Path(fn_in)
  output_dir_path = Path(output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)

  left_path = output_dir_path / "split_left.fq"
  right_path = output_dir_path / "split_right.fq"
  is_gz = fn_in_path.name.endswith((".fastq.gz", ".fq.gz"))
  open_fn = gzip.open if is_gz else open
  open_mode = "rt" if is_gz else "r"

  # Status corresponds to the expected type of line to read next
  #  - "@"  = Header for NT sequence
  #  - "BP" = NT sequence
  #  - "+"  = Header for quality assurance sequence
  #  - "QA" = Assurance sequence
  status = "@"
  current_len = None
  read_direction = 1  # 1 corresponds to left read (first), 2 corresponds to right read.
  index = 1  # Actual index
  with open_fn(fn_in_path, open_mode) as fin:
    with open(left_path, "w") as fout_l:
      with open(right_path, "w") as fout_r:
        for i, line in enumerate(fin, start=1):
          line = line.strip()
          if status == "@" and line.startswith("@"):
            if preserve_identifier_names:
              prefix, suffix = (line.split(" ", 1) + [""])[:2]
              base = re.sub(r"[/.][12]$", "", prefix)  # strip existing mate suffixes
              output = f"{base}/{read_direction}" + (f" {suffix}" if suffix else "")
            else:
              output = f"@{index}/{read_direction}"
            status = "BP"
          elif status == "BP" and re.search(r"^([GUATNC]*)$", line):
            output = line
            status = "+"
            current_len = len(line)
          elif status == "+" and line.startswith("+"):
            if line == "+":
              output = "+"
            elif preserve_identifier_names:
              prefix, suffix = (line.split(" ", 1) + [""])[:2]
              base = re.sub(r"[/.][12]$", "", prefix)  # strip existing mate suffixes
              output = f"{base}/{read_direction}" + (f" {suffix}" if suffix else "")
            else:
              output = "+"
            status = "QA"
          elif status == "QA" and re.search(
            r"^([ !\"#$%&'()*+,-.\/0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~])*$", line
          ):
            output = line
            status = "@"
            if current_len != len(line):
              raise ValueError(f"Sequence length does not match for line {i + 1}:\n{line}")
          else:
            print(status)
            raise ValueError(f"Unknown parsing error for line {i + 1}:\n{line}")

          # write to corresponding output file
          if read_direction == 1:
            fout_l.write(f"{output}\n")
          else:
            fout_r.write(f"{output}\n")

          # change read_direction
          if status == "@":
            if read_direction == 1:
              read_direction = 2
            else:
              read_direction = 1
              index += 1

  assert read_direction == 1, "Uneven number of reads in both files"
  logger.info(f"Found total of {index:,} syntactically valid reads")
  return left_path, right_path


def interleave_fastq(
  fn_left: Union[str, Path],
  fn_right: Union[str, Path],
  output_dir: Union[str, Path],
  *,
  preserve_identifier_names: bool = False,
) -> Path:
  """Interleave matched R1/R2 FASTQ records into ``interleaved.fq``.

  Streams plain or gzip-compressed ``.fastq.gz``/``.fq.gz`` inputs, validating
  four-line records, sequence/quality lengths, matching identifiers, and equal
  nonzero record counts. Inputs must already have mates in the same order.
  Mate suffixes ``/1``, ``/2``, ``.1``, and ``.2`` are normalized for comparison.

  Args:
      fn_left: Forward (R1) FASTQ path.
      fn_right: Reverse (R2) FASTQ path.
      output_dir: Directory for the uncompressed interleaved FASTQ.
      preserve_identifier_names: Preserve identifiers with normalized /1 and /2
          suffixes; otherwise rewrite them as @<index>/1 and @<index>/2.

  Returns:
      Path to the interleaved FASTQ, with R1 followed by R2 for each pair.

  Raises:
      ValueError: If records are malformed, empty, unequal in count, or unpaired.
  """
  inputs = (Path(fn_left), Path(fn_right))
  output_dir = Path(output_dir)
  output_path = output_dir / "interleaved.fq"
  if output_path.resolve() in {path.resolve() for path in inputs}:
    raise ValueError("Output file must be different from the input files.")
  output_dir.mkdir(parents=True, exist_ok=True)

  def read_record(handle: TextIO, path: Path, index: int) -> Optional[Tuple[str, str, str, str]]:
    header = handle.readline()
    if not header:
      return None
    lines = [header] + [handle.readline() for _ in range(3)]
    if any(not line for line in lines):
      raise ValueError(f"Incomplete FASTQ record {index} in {path.name}.")
    header, sequence, separator, quality = (line.rstrip("\r\n") for line in lines)
    if not header.startswith("@") or not header[1:].split():
      raise ValueError(f"Invalid FASTQ header at record {index} in {path.name}.")
    if not sequence or not re.fullmatch(r"[GUATNC]+", sequence):
      raise ValueError(f"Invalid FASTQ sequence at record {index} in {path.name}.")
    if not separator.startswith("+"):
      raise ValueError(f"Invalid FASTQ separator at record {index} in {path.name}.")
    if len(sequence) != len(quality):
      raise ValueError(f"Sequence/quality length mismatch at record {index} in {path.name}.")
    if any(not 33 <= ord(char) <= 126 for char in quality):
      raise ValueError(f"Invalid FASTQ quality at record {index} in {path.name}.")
    return header, sequence, separator, quality

  temporary_path = None
  try:
    with ExitStack() as stack:
      handles = [stack.enter_context((gzip.open if path.name.lower().endswith((".fastq.gz", ".fq.gz")) else open)(path, "rt")) for path in inputs]
      output = stack.enter_context(tempfile.NamedTemporaryFile(mode="w", dir=output_dir, suffix=".tmp", delete=False))
      temporary_path = Path(output.name)
      index = 1
      while True:
        records = [read_record(handle, path, index) for handle, path in zip(handles, inputs)]
        if all(record is None for record in records):
          if index == 1:
            raise ValueError("FASTQ inputs must contain at least one read pair.")
          break
        if any(record is None for record in records):
          raise ValueError(f"Unequal number of reads in R1 and R2 at pair {index}.")
        names = [record[0].split()[0][1:] for record in records]
        # SRA spot identifiers can end in .1 or .2 in both files. Equal names
        # identify the same spot; only strip dot suffixes when names differ.
        identifiers = names if names[0] == names[1] else [re.sub(r"[/.][12]$", "", name) for name in names]
        if identifiers[0] != identifiers[1]:
          raise ValueError(f"Read identifiers do not match at pair {index}: {identifiers[0]} and {identifiers[1]}.")
        for direction, record in enumerate(records, start=1):
          header, sequence, separator, quality = record
          parts = header.split(maxsplit=1)
          suffix_match = re.search(r"[/.]([12])$", parts[0])
          if suffix_match and ("/" in suffix_match[0] or names[0] != names[1]) and int(suffix_match[1]) != direction:
            raise ValueError(f"Wrong mate suffix in R{direction} at pair {index}.")
          if preserve_identifier_names:
            header = f"@{identifiers[direction - 1]}/{direction}" + (f" {parts[1]}" if len(parts) == 2 else "")
            if separator != "+":
              separator = "+" + header[1:]
          else:
            header, separator = f"@{index}/{direction}", "+"
          output.write(f"{header}\n{sequence}\n{separator}\n{quality}\n")
        index += 1
    temporary_path.replace(output_path)
  finally:
    if temporary_path is not None:
      temporary_path.unlink(missing_ok=True)
  logger.info(f"Interleaved {index - 1:,} read pairs")
  return output_path
