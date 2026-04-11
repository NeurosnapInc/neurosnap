from pathlib import Path

import pytest
from rdkit import Chem

from neurosnap.chemistry import sdf_to_smiles, smiles_to_sdf, validate_smiles

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
