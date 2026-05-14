"""Tests for the structure-first PDB2PQR adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from neurosnap.algos.pdb2pqr import _run_vendor_pdb2pqr, assign_pqr
from tests._structure_test_utils import make_structure


class _FakeVendorAtom:
  def __init__(
    self,
    *,
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_id: int,
    coord,
    element: str,
    partial_charge: float,
    radius: float,
    record_type: str = "ATOM",
  ):
    self.serial = serial
    self.name = atom_name
    self.res_name = res_name
    self.chain_id = chain_id
    self.res_seq = res_id
    self.ins_code = ""
    self.x = float(coord[0])
    self.y = float(coord[1])
    self.z = float(coord[2])
    self.element = element
    self.type = record_type
    self.occupancy = 1.0
    self.temp_factor = 12.5
    self.charge = 0
    self.ffcharge = partial_charge
    self.radius = radius
    self.residue = SimpleNamespace(name=res_name, res_seq=res_id)
    self.bonds = []


def test_assign_pqr_requires_structure():
  with pytest.raises(TypeError):
    assign_pqr(object())


def test_assign_pqr_returns_updated_structure(monkeypatch):
  input_structure = make_structure([("CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")])
  atom1 = _FakeVendorAtom(
    serial=1,
    atom_name="N",
    res_name="GLY",
    chain_id="A",
    res_id=1,
    coord=(1.0, 2.0, 3.0),
    element="N",
    partial_charge=0.1234,
    radius=1.55,
  )
  atom2 = _FakeVendorAtom(
    serial=2,
    atom_name="H",
    res_name="GLY",
    chain_id="A",
    res_id=1,
    coord=(1.5, 2.2, 3.1),
    element="H",
    partial_charge=0.4321,
    radius=1.05,
  )
  atom1.bonds = [atom2]
  atom2.bonds = [atom1]

  monkeypatch.setattr(
    "neurosnap.algos.pdb2pqr._run_vendor_pdb2pqr",
    lambda *args, **kwargs: (
      [atom1, atom2],
      "REMARK   1 Test header\n",
      {
        "pdb2pqr_forcefield": "AMBER",
        "pdb2pqr_missing_atoms": [],
      },
    ),
  )

  updated = assign_pqr(input_structure, forcefield="AMBER")

  assert len(updated) == 2
  assert updated.atom_annotations.dtype.names[-2:] == ("partial_charge", "radius")
  assert np.allclose(updated.atom_annotations["partial_charge"], np.array([0.1234, 0.4321], dtype=np.float32))
  assert np.allclose(updated.atom_annotations["radius"], np.array([1.55, 1.05], dtype=np.float32))
  assert updated.metadata["pdb2pqr_forcefield"] == "AMBER"
  assert updated.metadata["pdb2pqr_header"].startswith("REMARK")
  assert len(updated.bonds) == 1
  assert tuple(updated.bonds[0]) == (0, 1, 1)


def test_run_vendor_pdb2pqr_retries_assign_only_histidine_failure(monkeypatch):
  structure = make_structure(
    [
      ("ND1", "HIS", "A", 10, 0.0, 0.0, 0.0, "N"),
      ("NE2", "HIS", "A", 10, 1.0, 0.0, 0.0, "N"),
      ("HD1", "HIS", "A", 10, 0.0, 1.0, 0.0, "H"),
    ]
  )

  atom = _FakeVendorAtom(
    serial=1,
    atom_name="ND1",
    res_name="HIS",
    chain_id="A",
    res_id=10,
    coord=(0.0, 0.0, 0.0),
    element="N",
    partial_charge=0.1,
    radius=1.5,
  )
  calls = []

  def fake_once(run_structure, **kwargs):
    calls.append((run_structure, kwargs))
    if kwargs["assign_only"]:
      raise TypeError(
        "Invalid type for HIS A 10! Missing both HD1 and HE2 atoms. "
        "If you receive this error while using the --assign-only option you can only resolve it by adding HD1, HE2 or both to this residue."
      )
    return [atom], "REMARK   1 Retry\n", {"pdb2pqr_forcefield": "AMBER", "pdb2pqr_missing_atoms": []}

  monkeypatch.setattr("neurosnap.algos.pdb2pqr._run_vendor_pdb2pqr_once", fake_once)

  _, _, metadata = _run_vendor_pdb2pqr(
    structure,
    forcefield_name="amber",
    ffout_name=None,
    neutraln=False,
    neutralc=False,
    assign_only=True,
    debump_requested=True,
    optimize_requested=True,
    effective_debump=False,
    effective_optimize=False,
  )

  assert len(calls) == 2
  assert calls[0][1]["assign_only"] is True
  assert calls[1][1]["assign_only"] is False
  assert "assign_only_histidine_fallback" in metadata["pdb2pqr_remediations"]
  assert len(calls[1][0]) == 2


def test_run_vendor_pdb2pqr_strips_hydrogens_before_full_rebuild(monkeypatch):
  structure = make_structure(
    [
      ("N", "GLY", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("H", "GLY", "A", 1, 0.0, 1.0, 0.0, "H"),
      ("CA", "GLY", "A", 1, 1.0, 0.0, 0.0, "C"),
    ]
  )
  atom = _FakeVendorAtom(
    serial=1,
    atom_name="N",
    res_name="GLY",
    chain_id="A",
    res_id=1,
    coord=(0.0, 0.0, 0.0),
    element="N",
    partial_charge=0.1,
    radius=1.5,
  )
  observed_lengths = []

  def fake_once(run_structure, **kwargs):
    observed_lengths.append(len(run_structure))
    return [atom], "REMARK   1 Stripped\n", {"pdb2pqr_forcefield": "AMBER", "pdb2pqr_missing_atoms": []}

  monkeypatch.setattr("neurosnap.algos.pdb2pqr._run_vendor_pdb2pqr_once", fake_once)

  _, _, metadata = _run_vendor_pdb2pqr(
    structure,
    forcefield_name="amber",
    ffout_name=None,
    neutraln=False,
    neutralc=False,
    assign_only=False,
    debump_requested=False,
    optimize_requested=False,
    effective_debump=False,
    effective_optimize=False,
  )

  assert observed_lengths == [2]
  assert metadata["pdb2pqr_remediations"] == ["stripped_input_hydrogens"]
