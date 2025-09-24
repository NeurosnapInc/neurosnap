# tests/test_nucleotide.py
import pytest

from neurosnap.nucleotide import get_reverse_complement


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