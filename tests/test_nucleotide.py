# tests/test_nucleotide.py
import gzip
from pathlib import Path

import pytest

from neurosnap.nucleotide import get_reverse_complement, split_interleaved_fastq


def test_reverse_complement_dna_basic():
  # ATCG -> CGAT
  assert get_reverse_complement("ATCG") == "CGAT"


def test_reverse_complement_rna_basic():
  # AUCG -> CGAT (U treated as A's complement)
  assert get_reverse_complement("AUCG") == "CGAT"


def test_reverse_complement_mixed_case_and_length():
  # Handles longer sequence
  seq = "AATTCCGG"
  rc = get_reverse_complement(seq)
  assert rc == "CCGGAATT"
  assert len(rc) == len(seq)


def test_reverse_complement_invalid_char_raises():
  with pytest.raises(KeyError):
    get_reverse_complement("AXTG")  # X not valid


def _normalize_interleaved_headers(lines):
  normalized = []
  for line in lines:
    if not line:
      normalized.append(line)
      continue
    if line[0] in {"@", "+"}:
      parts = line.split(" ", 1)
      head = parts[0]
      if head.endswith(".1"):
        head = head[:-2] + "/1"
      elif head.endswith(".2"):
        head = head[:-2] + "/2"
      line = head if len(parts) == 1 else f"{head} {parts[1]}"
    normalized.append(line)
  return normalized


def test_split_interleaved_fastq_transcript_assembly(tmp_path):
  src_path = Path(__file__).resolve().parent / "files" / "transcript_assembly.fastq"
  lines = src_path.read_text().splitlines()
  assert len(lines) % 4 == 0

  interleaved_path = tmp_path / "transcript_assembly_interleaved.fastq"
  interleaved_path.write_text("\n".join(_normalize_interleaved_headers(lines)) + "\n")

  left_path, right_path = split_interleaved_fastq(interleaved_path, tmp_path)
  assert left_path.exists()
  assert right_path.exists()

  left_lines = left_path.read_text().splitlines()
  right_lines = right_path.read_text().splitlines()
  assert len(left_lines) % 4 == 0
  assert len(right_lines) % 4 == 0

  total_reads = len(lines) // 4
  assert total_reads % 2 == 0
  assert len(left_lines) // 4 == total_reads // 2
  assert len(right_lines) // 4 == total_reads // 2
  assert left_lines[0] == "@1/1"
  assert right_lines[0] == "@1/2"


def test_split_interleaved_fastq_uneven_reads_raises(tmp_path):
  interleaved_path = tmp_path / "uneven.fastq"
  interleaved_path.write_text(
    "\n".join(
      [
        "@read1/1",
        "ACGT",
        "+",
        "!!!!",
        "@read1/2",
        "TGCA",
        "+",
        "!!!!",
        "@read2/1",
        "CCCC",
        "+",
        "####",
      ]
    )
    + "\n"
  )

  with pytest.raises(AssertionError, match="Uneven number of reads in both files"):
    split_interleaved_fastq(interleaved_path, tmp_path)


@pytest.mark.parametrize("suffix", [".fastq.gz", ".fq.gz"])
def test_split_interleaved_fastq_gz_support(tmp_path, suffix):
  src_path = Path(__file__).resolve().parent / "files" / "transcript_assembly.fastq"
  lines = src_path.read_text().splitlines()
  assert len(lines) % 4 == 0

  interleaved_path = tmp_path / f"transcript_assembly_interleaved{suffix}"
  with gzip.open(interleaved_path, "wt") as fout:
    fout.write("\n".join(_normalize_interleaved_headers(lines)) + "\n")

  left_path, right_path = split_interleaved_fastq(interleaved_path, tmp_path)
  assert left_path.exists()
  assert right_path.exists()

  left_lines = left_path.read_text().splitlines()
  right_lines = right_path.read_text().splitlines()
  assert left_lines[0] == "@1/1"
  assert right_lines[0] == "@1/2"
