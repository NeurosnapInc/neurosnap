"""PDB I/O and ligand-extraction tests for structure objects."""

import numpy as np
import pytest

from neurosnap.io.pdb import save_pdb
from neurosnap.structure import extract_non_biopolymers

from ._structure_test_utils import PDB_NO_H, make_structure, parse_single_model


def test_save_and_reload_pdb(tmp_path):
  structure = parse_single_model(PDB_NO_H)
  output_pdb = tmp_path / "out.pdb"

  save_pdb(structure, output_pdb)
  assert output_pdb.exists() and output_pdb.stat().st_size > 0

  reloaded = parse_single_model(output_pdb)
  assert len(reloaded) == len(structure)
  assert [chain.chain_id for chain in reloaded.chains()] == [chain.chain_id for chain in structure.chains()]


def test_save_and_reload_mmcif(tmp_path):
  pytest.xfail("mmCIF read/write support has not been migrated to the structure layer yet.")


@pytest.mark.xfail(reason="The structure layer does not expose a to_sdf() export helper yet.", strict=True)
def test_to_sdf_writes(tmp_path):
  structure = parse_single_model(PDB_NO_H)
  output_sdf = tmp_path / "prot.sdf"
  structure.to_sdf(output_sdf)


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
