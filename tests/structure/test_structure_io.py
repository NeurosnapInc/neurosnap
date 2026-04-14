"""Structure I/O and ligand-extraction tests."""

import io
from pathlib import Path

import numpy as np
import pytest

from neurosnap.io.mmcif import save_cif
from neurosnap.io.pdb import parse_pdb, save_pdb
from neurosnap.io.sdf import parse_sdf, save_sdf
from neurosnap.structure import StructureEnsemble, StructureStack, extract_non_biopolymers, fix_nucleic_termini

from tests._structure_test_utils import FILES as TEST_FILES, PDB_NO_H, make_structure, parse_single_model

TESTS_DIR = Path(__file__).resolve().parents[1]
FILES = TESTS_DIR / "files"


def test_save_and_reload_pdb(tmp_path):
  structure = parse_single_model(PDB_NO_H)
  output_pdb = tmp_path / "out.pdb"

  save_pdb(structure, output_pdb)
  assert output_pdb.exists() and output_pdb.stat().st_size > 0

  reloaded = parse_single_model(output_pdb)
  assert len(reloaded) == len(structure)
  assert [chain.chain_id for chain in reloaded.chains()] == [chain.chain_id for chain in structure.chains()]


def test_fix_nucleic_termini_noop_roundtrips_protein_with_zinc(tmp_path, caplog):
  structure = parse_pdb(TEST_FILES / "protein_with_zinc_ions.pdb", return_type="ensemble").first()
  output_pdb = tmp_path / "zn_noop.pdb"

  fix_nucleic_termini(structure)
  save_pdb(structure, output_pdb)
  reloaded = parse_pdb(output_pdb, return_type="ensemble").first()

  assert len(reloaded) == len(structure)
  assert list(reloaded.atom_annotations["element"]) == list(structure.atom_annotations["element"])
  assert any("No nucleotide residues were found while running fix_nucleic_termini()" in message for message in caplog.messages)


def test_parse_pdb_infers_missing_hydrogen_elements_from_processed_pdb(caplog):
  structure = parse_pdb(TEST_FILES / "protein_with_zinc_ions_missing_elements_gmx.pdb", return_type="ensemble").first()

  assert len(structure) > 0
  assert any(element == "H" for element in structure.atom_annotations["element"])
  assert any("Missing element assignment at line 4; inferred element \"H\"" in message for message in caplog.messages)


def test_save_and_reload_mmcif(tmp_path):
  structure = parse_single_model(PDB_NO_H)
  output_cif = tmp_path / "out.cif"

  save_cif(structure, output_cif)
  assert output_cif.exists() and output_cif.stat().st_size > 0

  reloaded = parse_single_model(output_cif)
  assert len(reloaded) == len(structure)
  assert [chain.chain_id for chain in reloaded.chains()] == [chain.chain_id for chain in structure.chains()]


def _ligand_structure():
  structure = make_structure(
    [
      ("C1", "LIG", "B", 1, 10.0, 0.0, 0.0, "C"),
      ("C2", "LIG", "B", 1, 11.4, 0.0, 0.0, "C"),
      ("O1", "LIG", "B", 1, 12.2, 1.0, 0.0, "O"),
      ("N1", "LIG", "B", 1, 11.8, -1.1, 0.0, "N"),
      ("S1", "LIG", "B", 1, 13.0, 0.0, 0.0, "S"),
    ]
  )
  structure.atom_annotations["hetero"][:] = True
  structure.bonds = np.array(
    [(0, 1, 1), (1, 2, 2), (1, 3, 1), (0, 4, 1)],
    dtype=structure._dtype_bond,
  )
  structure.metadata["title"] = "Ligand"
  return structure


def test_save_and_reload_sdf(tmp_path):
  structure = _ligand_structure()
  output_sdf = tmp_path / "ligand.sdf"

  save_sdf(structure, output_sdf)
  assert output_sdf.exists() and output_sdf.stat().st_size > 0

  reloaded = parse_single_model(output_sdf)
  assert len(reloaded) == len(structure)
  assert np.array_equal(reloaded.atom_annotations["element"], structure.atom_annotations["element"])
  assert np.allclose(
    np.column_stack((reloaded.atoms["x"], reloaded.atoms["y"], reloaded.atoms["z"])),
    np.column_stack((structure.atoms["x"], structure.atoms["y"], structure.atoms["z"])),
  )
  assert len(reloaded.bonds) == len(structure.bonds)
  assert reloaded.metadata["title"] == "Ligand"


def test_parse_sdf_multi_record_auto_returns_stack(tmp_path):
  structure_one = _ligand_structure()
  structure_two = _ligand_structure()
  structure_two.metadata["model_id"] = 2
  structure_two.translate(x=1.0, y=2.0, z=3.0)

  ensemble = StructureEnsemble([structure_one], model_ids=[1])
  ensemble.append(structure_two, model_id=2)
  output_sdf = tmp_path / "ensemble.sdf"

  save_sdf(ensemble, output_sdf)
  parsed = parse_sdf(output_sdf, return_type="auto")

  assert isinstance(parsed, StructureStack)
  assert parsed.model_ids == [1, 2]
  assert len(parsed[1]) == len(structure_one)
  assert np.allclose(parsed[2].atoms["x"], structure_two.atoms["x"])


def test_extract_non_biopolymers(tmp_path):
  structure = make_structure(
    [
      ("N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("CA", "ALA", "A", 1, 1.4, 0.0, 0.0, "C"),
      ("C", "ALA", "A", 1, 2.0, 1.4, 0.0, "C"),
      ("C1", "LIG", "B", 1, 10.0, 0.0, 0.0, "C"),
      ("C2", "LIG", "B", 1, 11.4, 0.0, 0.0, "C"),
      ("O1", "LIG", "B", 1, 12.2, 1.0, 0.0, "O"),
      ("N1", "LIG", "B", 1, 11.8, -1.1, 0.0, "N"),
      ("S1", "LIG", "B", 1, 13.0, 0.0, 0.0, "S"),
    ]
  )
  structure.atom_annotations["hetero"][3:] = True
  structure.bonds = np.array(
    [(3, 4, 1), (4, 5, 1), (4, 6, 1), (5, 7, 1)],
    dtype=structure._dtype_bond,
  )
  output_dir = tmp_path / "ligands"

  extract_non_biopolymers(structure, str(output_dir), min_atoms=5)

  sdf_files = list(output_dir.glob("*.sdf"))
  assert output_dir.exists()
  assert len(sdf_files) == 1
  assert sdf_files[0].stat().st_size > 0


def test_parse_pdb_malformed_conect_strict_invalid_serial():
  pdb_text = (
    "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
    "ATOM      2  C2  LIG A   1       1.400   0.000   0.000  1.00 20.00           C  \n"
    "CONECT    1XXXX\n"
    "END\n"
  )

  with pytest.raises(ValueError, match='Invalid CONECT serial "XXXX"'):
    parse_pdb(io.StringIO(pdb_text), return_type="ensemble", malformed_conect="strict")


def test_parse_pdb_malformed_conect_warn_invalid_serial(caplog):
  pdb_text = (
    "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
    "ATOM      2  C2  LIG A   1       1.400   0.000   0.000  1.00 20.00           C  \n"
    "CONECT    1XXXX\n"
    "END\n"
  )

  ensemble = parse_pdb(io.StringIO(pdb_text), return_type="ensemble", malformed_conect="warn")
  assert len(ensemble.first().bonds) == 0
  assert any('Invalid CONECT serial "XXXX"' in message for message in caplog.messages)


def test_parse_pdb_malformed_conect_ignore_unknown_reference(caplog):
  pdb_text = (
    "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
    "ATOM      2  C2  LIG A   1       1.400   0.000   0.000  1.00 20.00           C  \n"
    "CONECT    1    9\n"
    "END\n"
  )

  ensemble = parse_pdb(io.StringIO(pdb_text), return_type="ensemble", malformed_conect="ignore")
  assert len(ensemble.first().bonds) == 0
  assert not caplog.messages


def test_parse_pdb_preserves_duplicate_ligand_atom_names_with_renaming(caplog):
  pdb_text = (
    "HETATM    1  C   UNK     0       0.000   0.000   0.000  0.00  0.00           C  \n"
    "HETATM    2  C   UNK     0       1.000   0.000   0.000  0.00  0.00           C  \n"
    "HETATM    3  C   UNK     0       2.000   0.000   0.000  0.00  0.00           C  \n"
    "HETATM    4  O   UNK     0       3.000   0.000   0.000  0.00  0.00           O  \n"
    "HETATM    5  O   UNK     0       4.000   0.000   0.000  0.00  0.00           O  \n"
    "END\n"
  )
  structure = parse_pdb(io.StringIO(pdb_text), return_type="ensemble").first()
  ligand = structure.to_dataframe().query("chain == '' and res_id == 0 and res_name == 'UNK'")

  assert len(ligand) == 5
  assert ligand["atom_name"].nunique() == 5
  assert any('Duplicate atom name "C"' in message for message in caplog.messages)
