"""Regression tests for landscape edge cases found by adversarial stress runs."""

from __future__ import annotations

import bz2

import numpy as np
import pytest

from neurosnap.chemistry.landscape import (
  ChemicalLandscape,
  FingerprintConfig,
  decompose_molecules,
  minhash_signatures,
  stream_chunks,
)


def test_minhash_accepts_empty_tail_rows():
  offsets = np.array([0, 1, 1], dtype=np.int64)
  indices = np.array([7], dtype=np.int64)
  a = np.array([1], dtype=np.uint64)
  b = np.array([0], dtype=np.uint64)

  signatures = minhash_signatures(offsets, indices, a, b)

  assert signatures.tolist() == [[7], [0xFFFFFFFF]]


def test_duplicate_compound_ids_are_rejected():
  with pytest.raises(ValueError, match="compound IDs must be unique"):
    ChemicalLandscape(["CCOC", "CCOC"], compound_ids=["dup", "dup"])


@pytest.mark.parametrize("smiles", [[], ["not-a-molecule"]])
def test_empty_or_all_invalid_libraries_build_a_empty_graph(smiles):
  library = ChemicalLandscape(smiles).build_all()

  assert len(library) == 0
  assert library.fingerprints is not None
  assert library.fingerprints.n_mols == 0
  assert library.characterize().counts["n_compounds"] == 0


def test_nested_config_overrides_are_converted():
  library = ChemicalLandscape(["CCO"], fingerprints={"n_bits": 256}).load()

  assert isinstance(library.config.fingerprints, FingerprintConfig)
  assert library.fingerprints.n_bits == 256


def test_rgroup_failure_count_includes_invalid_smiles():
  result = decompose_molecules(["CCO", "not-a-molecule"], "CC")

  assert len(result.rows) == 1
  assert result.n_failed == 1


def test_smi_bz2_reader(tmp_path):
  path = tmp_path / "library.smi.bz2"
  with bz2.open(path, "wt", encoding="utf-8") as handle:
    handle.write("CCO ethanol\n")

  chunks = list(stream_chunks(path))

  assert [chunk.compound_ids for chunk in chunks] == [["ethanol"]]
  assert [chunk.smiles for chunk in chunks] == [["CCO"]]
