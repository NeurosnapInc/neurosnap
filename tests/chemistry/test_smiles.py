from pathlib import Path

import pytest
from rdkit import Chem

from neurosnap.chemistry import (
  canonicalize_smiles,
  largest_fragment,
  neutralize_molecule,
  remove_salts,
  sdf_to_smiles,
  smiles_to_sdf,
  standardize_molecule,
  validate_smiles,
)

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


def test_canonicalize_smiles_returns_canonical_form():
  assert canonicalize_smiles("OC(C)C") == "CC(C)O"


def test_canonicalize_smiles_invalid_raises():
  with pytest.raises(ValueError):
    canonicalize_smiles("not-a-smiles")


def test_standardize_molecule_returns_rdkit_mol():
  mol = Chem.MolFromSmiles("C[N+](=O)[O-]")
  standardized = standardize_molecule(mol)
  assert isinstance(standardized, Chem.Mol)
  assert Chem.MolToSmiles(standardized) == "C[N+](=O)[O-]"
  assert standardized is not mol


def test_neutralize_molecule_removes_supported_charge():
  mol = Chem.MolFromSmiles("C[NH+](C)C")
  neutral = neutralize_molecule(mol)
  assert Chem.MolToSmiles(neutral) == "CN(C)C"


def test_largest_fragment_keeps_main_component():
  mol = Chem.MolFromSmiles("CCO.[Na+]")
  fragment = largest_fragment(mol)
  assert Chem.MolToSmiles(fragment) == "CCO"


def test_remove_salts_removes_common_counterion():
  mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]")
  stripped = remove_salts(mol)
  assert Chem.MolToSmiles(stripped) == "CC(=O)[O-]"
