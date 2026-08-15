"""Preparation helper tests for structure objects."""

from pathlib import Path

import numpy as np
import pytest

from neurosnap.algos import evoef2
from neurosnap.io.pdb import save_pdb
from neurosnap.structure import (
  Structure,
  add_hydrogens_with_pdb2pqr,
  has_hydrogens,
  optimize_hydrogens_with_pdb2pqr,
  rebuild_missing_atoms_with_evoef2,
  strip_hydrogens,
)
from tests._structure_test_utils import FILES, make_structure, parse_single_model


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
  with pytest.raises(TypeError):
    rebuild_missing_atoms_with_evoef2(object())


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


def test_rebuild_missing_atoms_with_evoef2_delegates_to_local_backend(monkeypatch):
  structure = make_structure([("CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")])
  sentinel = Structure(remove_annotations=False)
  captured = []

  def fake_rebuild_missing_atoms_backend(*args, **kwargs):
    captured.append((args, kwargs))
    return sentinel

  monkeypatch.setattr("neurosnap.structure.prepare.rebuild_missing_atoms_with_evoef2_backend", fake_rebuild_missing_atoms_backend)

  param_path = Path("/tmp/params.prm")
  topo_path = Path("/tmp/topology.rtf")

  result_missing = rebuild_missing_atoms_with_evoef2(structure, param_path=param_path, topo_path=topo_path)

  assert result_missing is sentinel
  assert captured == [((structure,), {"param_path": param_path, "topo_path": topo_path})]


def test_evoef2_atom_params_provide_native_elements():
  params = evoef2.load_atom_params()

  assert params["THR"]["HG1"].element == "H"
  assert params["HSD"]["HD1"].element == "H"
  assert params["HSE"]["HE2"].element == "H"
  assert params["ILE"]["CD"].element == "C"


def test_rebuild_missing_atoms_with_evoef2_exports_rebuilt_hydrogens_with_h_element(tmp_path):
  rebuilt = rebuild_missing_atoms_with_evoef2(parse_single_model(FILES / "1MAL.pdb"))
  hydrogen_like_indices = [
    idx
    for idx, atom_name in enumerate(rebuilt.atom_annotations["atom_name"])
    if str(atom_name).strip().upper().startswith("H")
  ]

  assert hydrogen_like_indices
  assert {str(rebuilt.atom_annotations["element"][idx]).strip().upper() for idx in hydrogen_like_indices} == {"H"}
  ile_cd_indices = [
    idx
    for idx, row in enumerate(rebuilt.atom_annotations)
    if str(row["res_name"]).strip().upper() == "ILE" and str(row["atom_name"]).strip().upper() == "CD"
  ]
  assert ile_cd_indices
  assert {str(rebuilt.atom_annotations["element"][idx]).strip().upper() for idx in ile_cd_indices} == {"C"}

  output_path = tmp_path / "rebuilt.pdb"
  save_pdb(rebuilt, output_path)
  for line in output_path.read_text().splitlines():
    if line.startswith(("ATOM", "HETATM")) and line[12:16].strip().upper().startswith("H"):
      assert line[76:78].strip().upper() == "H"
    if line.startswith(("ATOM", "HETATM")) and line[12:16].strip().upper() == "CD" and line[17:20].strip().upper() == "ILE":
      assert line[76:78].strip().upper() == "C"
