from pathlib import Path

import pytest
from rdkit import Chem

from neurosnap.database.ccd import CCD, get_ccd, get_ccd_entries, get_ccd_rcsb


class _MockCCDResponse:
  def __init__(self, payload):
    self._payload = payload
    self.status_code = 200

  def raise_for_status(self):
    return None

  def json(self):
    return self._payload


@pytest.fixture
def ccd_payload():
  return {
    "created_at": 1775430114,
    "entries": {
      "000": {"name": "methyl hydrogen carbonate", "smiles": "COC(O)=O"},
      "ATP": {"name": "ATP", "smiles": "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"},
      "EOH": {"name": "ethanol", "smiles": "CCO"},
    },
  }


@pytest.mark.integration
def test_get_ccd_entries_downloads_and_uses_cache(tmp_path: Path):
  # NOTE:
  # This test intentionally performs a real network request instead of mocking
  # requests.get. The goal is to catch upstream payload/schema changes from the
  # live CCD entries endpoint. Do not replace this with a fake response.
  cache = tmp_path / "ccd_entries.json"
  entries = get_ccd_entries(cache_path=str(cache))
  cache_mtime_ns = cache.stat().st_mtime_ns

  assert "ATP" in entries
  assert "EOH" in entries
  assert isinstance(entries["ATP"], CCD)
  assert cache.exists()

  entries_2 = get_ccd_entries(cache_path=str(cache))
  assert entries_2["ATP"].code == "ATP"
  assert entries_2["EOH"].smiles
  assert cache.stat().st_mtime_ns == cache_mtime_ns


def test_get_ccd_entries_returns_ccd_objects(monkeypatch, tmp_path: Path, ccd_payload):
  cache = tmp_path / "ccd_entries.json"
  monkeypatch.setattr("neurosnap.database.ccd.requests.get", lambda url, timeout=None: _MockCCDResponse(ccd_payload))

  entries = get_ccd_entries(cache_path=str(cache))
  assert sorted(entries) == ["000", "ATP", "EOH"]
  assert entries["ATP"].name == "ATP"


def test_get_ccd_by_code(monkeypatch, tmp_path: Path, ccd_payload):
  cache = tmp_path / "ccd_entries.json"
  monkeypatch.setattr("neurosnap.database.ccd.requests.get", lambda url, timeout=None: _MockCCDResponse(ccd_payload))

  atp = get_ccd("atp", cache_path=str(cache))
  assert atp is not None
  assert atp.code == "ATP"
  assert atp.smiles == ccd_payload["entries"]["ATP"]["smiles"]
  assert get_ccd("missing", cache_path=str(cache)) is None


def test_get_ccd_rcsb_downloads_sdf(tmp_path: Path):
  out = tmp_path / "ATP_ideal.sdf"
  get_ccd_rcsb("atp", str(out))
  assert out.exists()
  content = out.read_text()
  assert len(content) > 0
  assert "V2000" in content or "V3000" in content


def test_ccd_smiles_canonical_and_to_mol():
  ccd = CCD(code="EOH", name="ethanol", smiles="OCC")
  assert ccd.smiles_canonical() == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"), canonical=True)
  mol = ccd.to_mol()
  assert mol is not None
  assert Chem.MolToSmiles(mol, canonical=True) == "CCO"


def test_ccd_smiles_canonical_rejects_invalid_smiles():
  ccd = CCD(code="BAD", name="broken", smiles="not-a-smiles")
  with pytest.raises(ValueError):
    ccd.smiles_canonical()
