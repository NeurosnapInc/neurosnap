"""Missing PDB elements use exact protein definitions before atom alignment."""

import io
from pathlib import Path

import numpy as np
import pytest

from neurosnap.constants.chemistry import ATOMIC_MASSES
from neurosnap.io.pdb import parse_pdb, save_pdb


def atom_line(atom_name: str, residue: str, element: str = "", *, hetero: bool = False, serial: int = 1, x: float = 0.0) -> str:
  """Build a fixed-column PDB record, preserving atom-name alignment."""
  record = "HETATM" if hetero else "ATOM  "
  return (
    f"{record}{serial:5d} {atom_name:4s} {residue:3s} A{serial:4d}    "
    f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00          {element:>2s}\n"
  )


@pytest.mark.parametrize(("residue", "atom_name", "expected"), [
  ("ALA", "CA", "C"),
  ("LEU", "CD1", "C"),
  ("LYS", "CE", "C"),
  ("ARG", "HE", "H"),
  ("CYS", "HG", "H"),
  ("LYS", "HG1", "H"),
  ("HIS", "ND1", "N"),
  ("HIS", "NE2", "N"),
  ("ARG", "NH1", "N"),
  ("SER", "OG", "O"),
  ("CYS", "SG", "S"),
  ("LEU", "1HD1", "H"),
  ("LYS", "1HG", "H"),
  ("HID", "ND1", "N"),
  ("HIE", "NE2", "N"),
  ("CYX", "SG", "S"),
])
def test_known_protein_atom_elements(residue, atom_name, expected, tmp_path):
  structure = parse_pdb(io.StringIO(atom_line(atom_name, residue)), return_type="ensemble").first()
  assert list(structure.atom_annotations["element"]) == [expected]
  assert structure.to_dataframe().iloc[0]["mass"] == pytest.approx(ATOMIC_MASSES[expected])

  output = tmp_path / "roundtrip.pdb"
  save_pdb(structure, output)
  reloaded = parse_pdb(output, return_type="ensemble").first()
  assert list(reloaded.atom_annotations["element"]) == [expected]


@pytest.mark.parametrize(("residue", "atom_name", "expected"), [
  ("CA", "CA", "CA"),
  ("HG", "HG", "HG"),
  ("LIG", "CA", "CA"),
  ("MSE", "SE", "SE"),
  ("SEC", "SE", "SE"),
  # A recognized residue alone does not justify overriding alignment.
  ("ALA", "HG", "HG"),
  ("LIG", " CA", "C"),
  ("LIG", "1HG", "H"),
])
def test_unknown_pairs_keep_pdb_alignment(residue, atom_name, expected):
  structure = parse_pdb(io.StringIO(atom_line(atom_name, residue, hetero=True)), return_type="ensemble").first()
  assert list(structure.atom_annotations["element"]) == [expected]
  assert structure.to_dataframe().iloc[0]["mass"] == pytest.approx(ATOMIC_MASSES[expected.capitalize()])


@pytest.mark.parametrize("hetero", [False, True])
def test_explicit_element_is_preserved(hetero):
  structure = parse_pdb(io.StringIO(atom_line("CA", "ALA", "O", hetero=hetero)), return_type="ensemble").first()
  assert list(structure.atom_annotations["element"]) == ["O"]


def test_contextual_inference_gives_correct_center_of_mass():
  pdb = atom_line("CA", "ALA", serial=1) + atom_line("HG", "CYS", serial=2, x=10.0)
  structure = parse_pdb(io.StringIO(pdb), return_type="ensemble").first()
  expected_x = 10.0 * ATOMIC_MASSES["H"] / (ATOMIC_MASSES["C"] + ATOMIC_MASSES["H"])
  np.testing.assert_allclose(structure.calculate_center_of_mass(), [expected_x, 0.0, 0.0], rtol=1e-6)


def test_processed_protein_has_no_spurious_elements():
  path = Path(__file__).resolve().parents[1] / "files" / "protein_with_zinc_ions_missing_elements_gmx.pdb"
  structure = parse_pdb(path, return_type="ensemble").first()
  assert set(structure.atom_annotations["element"]) <= {"C", "H", "N", "O", "S", "ZN"}
  assert np.isfinite(structure.calculate_center_of_mass()).all()
