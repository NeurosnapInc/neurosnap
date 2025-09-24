# tests/test_chemicals.py
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from neurosnap.chemicals import (
  fetch_ccd,
  get_ccds,
  get_mol_center,
  move_ligand_to_center,
  sdf_to_smiles,
  smiles_to_sdf,
  validate_smiles,
)
from neurosnap.protein import Protein

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"


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
# get_ccds (network mocked)
# -----------------------------


class _MockResp:
  def __init__(self, text: str = "", content: bytes = b"", status: int = 200):
    self.text = text
    self.content = content
    self.status_code = status

  def raise_for_status(self):
    if not (200 <= self.status_code < 300):
      raise Exception(f"HTTP {self.status_code}")


def test_get_ccds_downloads_and_caches(monkeypatch, tmp_path: Path):
  cache = tmp_path / "ccd_codes.json"

  # Mock response: include real-like tokens
  cif_text = "\n".join(
    [
      "data_something",
      "_chem_comp.three_letter_code ATP",
      "_chem_comp.three_letter_code ?",
      "_chem_comp.three_letter_code NAD",
    ]
  )

  called = {"count": 0}

  def fake_get(url):
    called["count"] += 1
    # ensure hitting the expected endpoint
    assert "components.cif" in url
    return _MockResp(text=cif_text)

  monkeypatch.setattr("neurosnap.chemicals.requests.get", fake_get)

  # First call: should download and cache
  codes = get_ccds(str(cache))
  assert codes == {"ATP", "NAD"}
  assert cache.exists()
  # cache content should be a JSON list of codes
  on_disk = set(json.loads(cache.read_text()))
  assert on_disk == {"ATP", "NAD"}
  assert called["count"] == 1

  # Second call: should read from cache, not re-download
  def exploding_get(_):
    raise AssertionError("requests.get should not be called when cache exists")

  monkeypatch.setattr("neurosnap.chemicals.requests.get", exploding_get)

  codes2 = get_ccds(str(cache))
  assert codes2 == {"ATP", "NAD"}


# -----------------------------
# fetch_ccd (network mocked)
# -----------------------------


def test_fetch_ccd_writes_file_uppercases(monkeypatch, tmp_path: Path):
  out = tmp_path / "ATP_ideal.sdf"

  seen = {"url": None}

  def fake_get(url):
    seen["url"] = url
    return _MockResp(content=b"SDFDATA")

  monkeypatch.setattr("neurosnap.chemicals.requests.get", fake_get)

  fetch_ccd("atp", str(out))
  assert out.exists()
  assert out.read_bytes() == b"SDFDATA"
  assert seen["url"] is not None and seen["url"].endswith("/ATP_ideal.sdf")


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
  # receptor center via Protein helper
  r_center = Protein(str(receptor)).calculate_center_of_mass()

  # ligand new center via RDKit
  suppl = Chem.SDMolSupplier(str(out), removeHs=False)
  mol = suppl[0]
  assert mol is not None
  l_center = get_mol_center(mol, use_mass=False)

  # centers should match closely
  assert np.linalg.norm(r_center - l_center) < 1e-3
