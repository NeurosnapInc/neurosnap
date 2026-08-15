"""Preparation helper tests for structure objects."""

import numpy as np
import pytest

from neurosnap.structure import (
  Structure,
  add_terminal_capping_groups,
  add_hydrogens_with_pdb2pqr,
  has_hydrogens,
  optimize_hydrogens_with_pdb2pqr,
  strip_hydrogens,
)
from tests._structure_test_utils import PROTEIN_BACKBONE_ATOMS, replace_chain, make_structure


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
    add_terminal_capping_groups(object())
  with pytest.raises(TypeError):
    add_hydrogens_with_pdb2pqr(object())
  with pytest.raises(TypeError):
    optimize_hydrogens_with_pdb2pqr(object())


def test_add_terminal_capping_groups_adds_ace_and_nme_caps():
  structure = make_structure(PROTEIN_BACKBONE_ATOMS, bonds=[(2, 3, 1, 0)], interactions=[(1, 4, 127)])

  capped = add_terminal_capping_groups(structure)

  assert len(structure) == 6
  assert len(capped) == 11
  assert capped.metadata == structure.metadata
  res_names = capped.atom_annotations["res_name"].tolist()
  assert res_names.count("ACE") == 3
  assert res_names.count("NME") == 2
  assert capped.atom_annotations["res_name"].tolist() == ["ACE", "ACE", "ACE", "ALA", "ALA", "ALA", "GLY", "GLY", "GLY", "NME", "NME"]
  assert capped.atom_annotations["atom_name"].tolist() == ["CH3", "C", "O", "N", "CA", "C", "N", "CA", "C", "N", "CH3"]
  assert capped.atom_annotations["element"].tolist() == ["C", "C", "O", "N", "C", "C", "N", "C", "C", "N", "C"]
  assert capped.atom_annotations["res_id"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
  assert capped.atom_annotations["chain_id"].tolist() == ["A"] * 11

  bond_rows = {tuple(row) for row in capped.bonds.tolist()}
  assert (0, 1, 1, 0) in bond_rows
  assert (1, 2, 2, 0) in bond_rows
  assert (1, 3, 1, 0) in bond_rows
  assert (5, 6, 1, 0) in bond_rows
  assert (8, 9, 1, 0) in bond_rows
  assert (9, 10, 1, 0) in bond_rows
  assert capped.interactions.tolist() == [(4, 7, 127)]


def test_add_terminal_capping_groups_filters_chains_and_can_disable_one_end():
  atom_defs = PROTEIN_BACKBONE_ATOMS + tuple(replace_chain(PROTEIN_BACKBONE_ATOMS, "B"))
  structure = make_structure(atom_defs)

  capped = add_terminal_capping_groups(structure, chains=["B"], c_terminal=False)

  assert len(capped) == len(structure) + 3
  cap_rows = capped.atom_annotations[6:9]
  assert cap_rows["res_name"].tolist() == ["ACE", "ACE", "ACE"]
  assert cap_rows["chain_id"].tolist() == ["B", "B", "B"]
  assert capped.atom_annotations["res_name"][9] == "ALA"
  assert "NME" not in capped.atom_annotations["res_name"].tolist()


def test_add_terminal_capping_groups_skips_existing_caps():
  atom_defs = (
    ("CH3", "ACE", "A", 0, -2.8, 0.0, 0.0, "C"),
    ("C", "ACE", "A", 0, -1.3, 0.0, 0.0, "C"),
    ("O", "ACE", "A", 0, -1.3, 1.2, 0.0, "O"),
    *PROTEIN_BACKBONE_ATOMS,
    ("N", "NME", "A", 3, 6.5, 2.8, 0.0, "N"),
    ("CH3", "NME", "A", 3, 7.9, 2.8, 0.0, "C"),
  )
  structure = make_structure(atom_defs)

  capped = add_terminal_capping_groups(structure)

  assert len(capped) == len(structure)
  assert capped.atom_annotations["res_name"].tolist().count("ACE") == 3
  assert capped.atom_annotations["res_name"].tolist().count("NME") == 2


def test_add_terminal_capping_groups_rejects_missing_chain():
  structure = make_structure(PROTEIN_BACKBONE_ATOMS)

  with pytest.raises(ValueError, match="Chain\\(s\\) not found"):
    add_terminal_capping_groups(structure, chains=["Z"])


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
