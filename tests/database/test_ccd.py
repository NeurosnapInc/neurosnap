from pathlib import Path

import pytest
from rdkit import Chem

from neurosnap.database.ccd import CCD, get_ccd, get_ccd_entries


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


def test_get_ccd_entries_downloads_and_uses_cache(monkeypatch, tmp_path: Path, ccd_payload):
  cache = tmp_path / "ccd_entries.json"
  called = {"count": 0}

  def fake_get(url, timeout=None):
    called["count"] += 1
    assert "entries.json" in url
    return _MockCCDResponse(ccd_payload)

  monkeypatch.setattr("neurosnap.database.ccd.requests.get", fake_get)

  entries = get_ccd_entries(cache_path=str(cache))
  assert set(entries) == {"000", "ATP", "EOH"}
  assert isinstance(entries["ATP"], CCD)
  assert cache.exists()
  assert called["count"] == 1

  entries_2 = get_ccd_entries(cache_path=str(cache))
  assert entries_2["EOH"].smiles == "CCO"
  assert called["count"] == 1


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
