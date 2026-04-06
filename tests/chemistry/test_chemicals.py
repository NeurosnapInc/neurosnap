from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from neurosnap.chemicals import (
  get_mol_center,
  move_ligand_to_center,
  sdf_to_smiles,
  smiles_to_sdf,
  validate_smiles,
)
from tests._structure_test_utils import parse_single_model

TESTS_DIR = Path(__file__).resolve().parents[1]
FILES = TESTS_DIR / "files"


# -----------------------------
# smiles_to_sdf
# -----------------------------


def test_smiles_to_sdf_valid_roundtrip(tmp_path: Path):
  out = tmp_path / "ethanol.sdf"
  smiles_to_sdf("CCO", str(out))
  assert out.exists()
  suppl = Chem.SDMolSupplier(str(out), removeHs=False)
  mol = suppl[0]
  assert mol is not None
  # sanity: generated smiles from SDF parses
  gen = Chem.MolToSmiles(mol)
  assert isinstance(gen, str) and len(gen) > 0


def test_smiles_to_sdf_invalid_raises(tmp_path: Path):
  out = tmp_path / "bad.sdf"
  with pytest.raises(ValueError):
    smiles_to_sdf("not-a-smiles", str(out))


# -----------------------------
# sdf_to_smiles
# -----------------------------


def test_sdf_to_smiles_reads_valid_list():
  sdf = FILES / "Structure2D_COMPOUND_CID_3345.sdf"
  assert sdf.exists(), "Fixture SDF missing"
  smiles_list = sdf_to_smiles(str(sdf))
  assert isinstance(smiles_list, list) and len(smiles_list) > 0
  # each returned string should be a valid SMILES (rdkit-parseable)
  for s in smiles_list:
    assert validate_smiles(s)


# -----------------------------
# validate_smiles
# -----------------------------


@pytest.mark.parametrize(
  "txt,expected",
  [
    ("CCO", True),
    ("c1ccccc1O", True),
    ("N[N+](N)N", True),
    ("XYZ", False),
    ("", False),
  ],
)
def test_validate_smiles_various(txt, expected):
  assert validate_smiles(txt) is expected


# -----------------------------
# get_mol_center
# -----------------------------


def test_get_mol_center_geom_and_mass():
  sdf = FILES / "input_ligand.sdf"
  assert sdf.exists(), "Fixture ligand SDF missing"
  suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
  mol = suppl[0]
  assert mol is not None

  c_geom = get_mol_center(mol, use_mass=False)
  c_mass = get_mol_center(mol, use_mass=True)

  assert isinstance(c_geom, np.ndarray) and c_geom.shape == (3,)
  assert isinstance(c_mass, np.ndarray) and c_mass.shape == (3,)
  # centers should be finite numbers
  assert np.isfinite(c_geom).all()
  assert np.isfinite(c_mass).all()


# -----------------------------
# move_ligand_to_center
# -----------------------------


def test_move_ligand_to_center_aligns_centers(tmp_path: Path):
  ligand = FILES / "input_ligand.sdf"
  receptor = FILES / "1BTL.pdb"
  out = tmp_path / "centered_ligand.sdf"

  # run function
  res_path = move_ligand_to_center(str(ligand), str(receptor), str(out), use_mass=False)
  assert Path(res_path).exists()

  # compute centers to verify alignment
  r_center = parse_single_model(receptor).calculate_center_of_mass()

  # ligand new center via RDKit
  suppl = Chem.SDMolSupplier(str(out), removeHs=False)
  mol = suppl[0]
  assert mol is not None
  l_center = get_mol_center(mol, use_mass=False)

  # centers should match closely
  assert np.linalg.norm(r_center - l_center) < 1e-3
