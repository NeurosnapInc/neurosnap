"""
Provides functions and classes related to processing nucleotide data.
"""

import re
from pathlib import Path
from typing import Tuple, Union

from neurosnap.log import logger


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


def split_interleaved_fastq(fn_in: Union[str, Path], fn_out: Union[str, Path]) -> Tuple[Path, Path]:
  """
  Split an interleaved FASTQ into left/right FASTQ files.

  Assumes pairs are adjacent (left read followed by right read) and rewrites
  headers as "@<index>/1" and "@<index>/2".

  Parameters:
      fn_in: Path to the interleaved FASTQ file.
      fn_out: Output prefix or directory. If a directory is provided, the input
          stem is used as the prefix. If a filename is provided, any suffix is
          stripped before appending "_l.fq" and "_r.fq".

  Returns:
      Tuple[Path, Path]: Paths to the left and right FASTQ output files.
  """
  fn_in_path = Path(fn_in)
  fn_out_path = Path(fn_out)
  if fn_out_path.is_dir():
    prefix = fn_out_path / fn_in_path.stem
  else:
    prefix = fn_out_path.with_suffix("") if fn_out_path.suffix else fn_out_path

  left_path = prefix.parent / f"{prefix.name}_l.fq"
  right_path = prefix.parent / f"{prefix.name}_r.fq"
  left_path.parent.mkdir(parents=True, exist_ok=True)

  # Status corresponds to the expected type of line to read next
  #  - "@"  = Header for NT sequence
  #  - "BP" = NT sequence
  #  - "+"  = Header for quality assurance sequence
  #  - "QA" = Assurance sequence
  status = "@"
  current_len = None
  read_direction = 1  # 1 corresponds to left read (first), 2 corresponds to right read.
  with open(fn_in_path) as fin:
    with open(left_path, "w") as fout_l:
      with open(right_path, "w") as fout_r:
        for index, line in enumerate(fin, start=1):
          line = line.strip()
          if status == "@" and re.search(r"@.*?\s", line):
            output = f"@{index}/{read_direction}"
            status = "BP"
          elif status == "BP" and re.search(r"^([GUATNC]*)$", line):
            output = line
            status = "+"
            current_len = len(line)
          elif status == "+" and line.startswith("+"):
            output = line
            status = "QA"
          elif status == "QA" and re.search(
            r"^([ !\"#$%&'()*+,-.\/0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~])*$", line
          ):
            output = line
            status = "@"
            if current_len != len(line):
              raise ValueError(f"Sequence length does not match for line {index}:\n{line}")
          else:
            print(status)
            raise ValueError(f"Unknown parsing error for line {index}:\n{line}")

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