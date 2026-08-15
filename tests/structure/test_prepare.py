"""Preparation helper tests for structure objects."""

import numpy as np
import pytest

from neurosnap.structure import (
  Structure,
  add_hydrogens_with_pdb2pqr,
  has_hydrogens,
  optimize_hydrogens_with_pdb2pqr,
  strip_hydrogens,
)
from tests._structure_test_utils import make_structure


def test_has_hydrogens_detects_present_and_absent_hydrogens():
  no_h = make_structure([("CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")])
  with_h = make_structure(
    [
      ("CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("H", "GLY", "A", 1, 0.2, 0.1, 0.0, "H"),
    ]
  )

  assert has_hydrogens(no_h) is False
  assert has_hydrogens(with_h) is True


def test_strip_hydrogens_returns_independent_subset_and_remaps_bonds():
  structure = make_structure(
    [
      ("N", "GLY", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("H", "GLY", "A", 1, 0.2, 0.0, 0.0, "H"),
      ("CA", "GLY", "A", 1, 1.4, 0.0, 0.0, "C"),
    ]
  )
  structure.bonds = np.array([(0, 1, 1, 0), (0, 2, 1, 0)], dtype=structure._dtype_bond)

  stripped = strip_hydrogens(structure)

  assert len(stripped) == 2
  assert list(stripped.atom_annotations["element"]) == ["N", "C"]
  assert len(stripped.bonds) == 1
  assert tuple(stripped.bonds[0]) == (0, 1, 1, 0)


def test_prepare_helpers_require_structure():
  with pytest.raises(TypeError):
    has_hydrogens(object())
  with pytest.raises(TypeError):
    strip_hydrogens(object())
  with pytest.raises(TypeError):
    add_hydrogens_with_pdb2pqr(object())
  with pytest.raises(TypeError):
    optimize_hydrogens_with_pdb2pqr(object())


def test_add_hydrogens_with_pdb2pqr_delegates_to_assign_pqr(monkeypatch):
  structure = make_structure([("CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")])
  sentinel = Structure(remove_annotations=False)
  captured = {}

  def fake_assign_pqr(*args, **kwargs):
    captured["args"] = args
    captured["kwargs"] = kwargs
    return sentinel

  monkeypatch.setattr("neurosnap.algos.pdb2pqr.assign_pqr", fake_assign_pqr)

  result = add_hydrogens_with_pdb2pqr(
    structure,
    forcefield="AMBER",
    ffout="PARSE",
    neutraln=True,
    neutralc=False,
    debump=False,
  )

  assert result is sentinel
  assert captured["args"] == (structure,)
  assert captured["kwargs"] == {
    "forcefield": "AMBER",
    "ffout": "PARSE",
    "neutraln": True,
    "neutralc": False,
    "assign_only": False,
    "debump": False,
    "optimize": False,
  }


def test_optimize_hydrogens_with_pdb2pqr_delegates_to_assign_pqr(monkeypatch):
  structure = make_structure([("CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")])
  sentinel = Structure(remove_annotations=False)
  captured = {}

  def fake_assign_pqr(*args, **kwargs):
    captured["args"] = args
    captured["kwargs"] = kwargs
    return sentinel

  monkeypatch.setattr("neurosnap.algos.pdb2pqr.assign_pqr", fake_assign_pqr)

  result = optimize_hydrogens_with_pdb2pqr(structure, forcefield="CHARMM", debump=True)

  assert result is sentinel
  assert captured["args"] == (structure,)
  assert captured["kwargs"] == {
    "forcefield": "CHARMM",
    "ffout": None,
    "neutraln": False,
    "neutralc": False,
    "assign_only": False,
    "debump": True,
    "optimize": True,
  }

