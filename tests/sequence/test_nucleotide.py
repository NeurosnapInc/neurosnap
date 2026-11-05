# tests/sequence/test_nucleotide.py
import gzip
import random
from pathlib import Path

import pytest

from neurosnap.sequence.nucleotide import generate_random_rna_sequence, get_reverse_complement, interleave_fastq, split_interleaved_fastq


def test_random_rna_length_features_and_composition():
  sequence = generate_random_rna_sequence(
    length=300,
    gc_content=50,
    add_5_utr=True,
    add_3_utr=True,
    poly_a_tail_length=25,
    stem_loops=2,
    include_pseudouridine=True,
    rng=random.Random(42),
  )
  assert len(sequence) == 300
  assert sequence[24:33] == "GCCACCAUG"
  assert sequence[-25:] == "A" * 25
  assert sequence[-58:-55] in ("UAA", "UAG", "UGA")
  assert "Ψ" in sequence
  assert abs(100 * (sequence.count("G") + sequence.count("C")) / len(sequence) - 50) < 0.5


def test_random_rna_short_preset_and_impossible_features():
  sequence = generate_random_rna_sequence(length=22, add_start_codon=False, add_stop_codon=False, stem_loops=1, rng=random.Random(1))
  assert len(sequence) == 22
  motif = sequence[:12]
  assert motif[-4:] == motif[:4].translate(str.maketrans("AUGC", "UACG"))[::-1]
  with pytest.raises(ValueError, match="too short"):
    generate_random_rna_sequence(length=22, poly_a_tail_length=200)


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
  src_path = Path(__file__).resolve().parents[1] / "files" / "transcript_assembly.fastq"
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


def test_split_interleaved_fastq_preserves_bare_separator_lines(tmp_path):
  interleaved_path = tmp_path / "interleaved.fastq"
  interleaved_path.write_text("@read.1\nACGT\n+\n!!!!\n@read.2\nTGCA\n+\n####\n")

  left_path, right_path = split_interleaved_fastq(
    interleaved_path,
    tmp_path,
    preserve_identifier_names=True,
  )

  assert left_path.read_text().splitlines() == ["@read/1", "ACGT", "+", "!!!!"]
  assert right_path.read_text().splitlines() == ["@read/2", "TGCA", "+", "####"]


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
  src_path = Path(__file__).resolve().parents[1] / "files" / "transcript_assembly.fastq"
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


@pytest.mark.parametrize("suffixes", [(".fastq", ".fq"), (".fastq.gz", ".fq.gz"), (".FASTQ.GZ", ".fastq")])
@pytest.mark.parametrize("preserve", [False, True])
def test_interleave_fastq_round_trip(tmp_path, suffixes, preserve):
  paths = [tmp_path / f"r{direction}{suffix}" for direction, suffix in enumerate(suffixes, 1)]
  for direction, path in enumerate(paths, 1):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "wt") as handle:
      handle.write(f"@readA.{direction} description\nACGT\n+readA.{direction} description\n!!!!\n")
      handle.write(f"@readB/{direction}\nTGCA\n+\n####\n")
  output = interleave_fastq(*paths, tmp_path / "out", preserve_identifier_names=preserve)
  lines = output.read_text().splitlines()
  expected = ["@readA/1 description", "@readA/2 description", "@readB/1", "@readB/2"] if preserve else ["@1/1", "@1/2", "@2/1", "@2/2"]
  assert lines[::4] == expected
  assert lines[1::4] == ["ACGT", "ACGT", "TGCA", "TGCA"]
  left, right = split_interleaved_fastq(output, tmp_path / "split", preserve_identifier_names=True)
  assert left.read_text().splitlines()[::4] == expected[::2]
  assert right.read_text().splitlines()[::4] == expected[1::2]


@pytest.mark.parametrize(
  "left,right,message",
  [
    ("", "", "at least one"),
    ("@read/1\nACGT\n+\n!!!!\n", "", "Unequal"),
    ("", "@read/2\nACGT\n+\n!!!!\n", "Unequal"),
    ("@read/1\nACGT\n+\n", "@read/2\nACGT\n+\n!!!!\n", "Incomplete"),
    ("read/1\nACGT\n+\n!!!!\n", "@read/2\nACGT\n+\n!!!!\n", "header"),
    ("@read/1\nAXGT\n+\n!!!!\n", "@read/2\nACGT\n+\n!!!!\n", "sequence"),
    ("@read/1\nACGT\n-\n!!!!\n", "@read/2\nACGT\n+\n!!!!\n", "separator"),
    ("@read/1\nACGT\n+\n!!!\n", "@read/2\nACGT\n+\n!!!!\n", "length mismatch"),
    ("@read/1\nACGT\n+\n!!! \n", "@read/2\nACGT\n+\n!!!!\n", "quality"),
    ("@read/1\nACGT\n+\n!!!!\n", "@other/2\nACGT\n+\n!!!!\n", "identifiers"),
    ("@read/2\nACGT\n+\n!!!!\n", "@read/1\nACGT\n+\n!!!!\n", "mate suffix"),
  ],
)
def test_interleave_fastq_rejects_bad_pairs(tmp_path, left, right, message):
  paths = [tmp_path / "r1.fq", tmp_path / "r2.fq"]
  for path, data in zip(paths, (left, right)):
    path.write_text(data)
  with pytest.raises(ValueError, match=message):
    interleave_fastq(*paths, tmp_path / "out")
  assert not list((tmp_path / "out").iterdir())


def test_interleave_fastq_does_not_overwrite_input(tmp_path):
  path = tmp_path / "interleaved.fq"
  path.write_text("input")
  with pytest.raises(ValueError, match="different"):
    interleave_fastq(path, path, tmp_path)
  assert path.read_text() == "input"


def test_interleave_fastq_illumina_headers(tmp_path):
  paths = [tmp_path / "r1.fq", tmp_path / "r2.fq"]
  for direction, path in enumerate(paths, 1):
    path.write_text(f"@instrument:read {direction}:N:0:INDEX\nACGT\n+\n!!!!\n")
  output = interleave_fastq(*paths, tmp_path, preserve_identifier_names=True)
  assert output.read_text().splitlines()[::4] == ["@instrument:read/1 1:N:0:INDEX", "@instrument:read/2 2:N:0:INDEX"]


@pytest.mark.parametrize("spot", ["SRR123.1", "SRR123.2"])
def test_interleave_fastq_preserves_sra_spot_ids(tmp_path, spot):
  paths = [tmp_path / "r1.fq", tmp_path / "r2.fq"]
  for path in paths:
    path.write_text(f"@{spot}\nACGT\n+\n!!!!\n")
  output = interleave_fastq(*paths, tmp_path, preserve_identifier_names=True)
  assert output.read_text().splitlines()[::4] == [f"@{spot}/1", f"@{spot}/2"]


def test_interleave_fastq_cleans_up_corrupt_gzip(tmp_path):
  left, right = tmp_path / "left.fastq.gz", tmp_path / "right.fq"
  left.write_bytes(b"not gzip")
  right.write_text("@read/2\nACGT\n+\n!!!!\n")
  with pytest.raises(gzip.BadGzipFile):
    interleave_fastq(left, right, tmp_path / "out")
  assert not list((tmp_path / "out").iterdir())
