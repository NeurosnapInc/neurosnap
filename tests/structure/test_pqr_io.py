"""Tests for PQR output from annotated Structures."""

from io import StringIO

import pytest

from neurosnap.io.pqr import save_pqr
from tests._structure_test_utils import make_structure


def _annotated_structure():
  structure = make_structure(
    [
      ("N", "GLY", "A", 1, 1.0, 2.0, 3.0, "N"),
      ("CA", "GLY", "A", 1, 1.5, 2.4, 3.2, "C"),
    ]
  )
  structure.add_annotation("partial_charge", "f4", values=[0.1234, -0.5678])
  structure.add_annotation("radius", "f4", values=[1.55, 1.70])
  structure.metadata["pdb2pqr_header"] = "REMARK   1 Test header\n"
  return structure


def test_save_pqr_writes_header_and_atom_records():
  structure = _annotated_structure()
  buffer = StringIO()

  save_pqr(structure, buffer, include_header=True)

  text = buffer.getvalue().splitlines()
  assert text[0] == "REMARK   1 Test header"
  assert text[1].startswith("ATOM")
  assert "0.1234" in text[1]
  assert "1.5500" in text[1]
  assert text[-2] == "TER"
  assert text[-1] == "END"


def test_save_pqr_requires_charge_and_radius_annotations():
  structure = make_structure([("CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")])

  with pytest.raises(ValueError, match="partial_charge"):
    save_pqr(structure, StringIO())
