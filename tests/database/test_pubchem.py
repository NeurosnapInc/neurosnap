"""Tests for the PubChem database helpers."""

from typing import Any, Dict, List, Optional

import pytest
import requests

from neurosnap.database.pubchem import (
  PubChemCompound,
  PubChemHit,
  fetch_pubchem_sdfs,
  get_pubchem_compound,
  search_pubchem_similar,
)

ETHANOL_PROPERTIES = {
  "CID": 702,
  "MolecularFormula": "C2H6O",
  "MolecularWeight": "46.07",
  "SMILES": "CCO",
  "IUPACName": "ethanol",
}

ASPIRIN_PROPERTIES = {
  "CID": 2244,
  "MolecularFormula": "C9H8O4",
  "MolecularWeight": "180.16",
  "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "IUPACName": "aspirin",
}


class _MockResponse:
  """Minimal stand-in for ``requests.Response``."""

  def __init__(self, *, json_data: Optional[Any] = None, text: str = "", status_code: int = 200):
    self._json_data = json_data
    self.text = text
    self.status_code = status_code

  def raise_for_status(self) -> None:
    if self.status_code >= 400:
      raise requests.HTTPError(f"HTTP {self.status_code}")

  def json(self) -> Any:
    return self._json_data


def _install_routes(monkeypatch, routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Install fake request methods that match URLs against ``routes``.

  Each route is a dict with ``match`` (substring) and ``response`` (a
  ``_MockResponse`` or callable). Returns the recorded list of calls.
  """
  calls: List[Dict[str, Any]] = []

  def fake_request(method: str, url: str, params: Optional[dict], data: Optional[dict], timeout: Optional[int]):
    calls.append({"method": method, "url": url, "params": params, "data": data, "timeout": timeout})
    for route in routes:
      if route["match"] in url:
        response = route["response"]
        return response(params) if callable(response) else response
    raise AssertionError(f"Unexpected request URL: {url}")

  def fake_get(url: str, params: Optional[dict] = None, timeout: Optional[int] = None):
    return fake_request("get", url, params, None, timeout)

  def fake_post(url: str, params: Optional[dict] = None, data: Optional[dict] = None, timeout: Optional[int] = None):
    return fake_request("post", url, params, data, timeout)

  monkeypatch.setattr("neurosnap.database.pubchem.requests.get", fake_get)
  monkeypatch.setattr("neurosnap.database.pubchem.requests.post", fake_post)
  return calls


def test_get_pubchem_compound_by_name(monkeypatch):
  _install_routes(
    monkeypatch,
    [
      {"match": "/compound/name/ethanol/cids/TXT", "response": _MockResponse(text="702\n")},
      {"match": "/compound/cid/702/property/", "response": _MockResponse(json_data={"PropertyTable": {"Properties": [ETHANOL_PROPERTIES]}})},
    ],
  )

  compound = get_pubchem_compound("ethanol")

  assert isinstance(compound, PubChemCompound)
  assert compound.cid == 702
  assert compound.name == "ethanol"
  assert compound.smiles == "CCO"
  assert compound.molecular_formula == "C2H6O"
  assert compound.molecular_weight == pytest.approx(46.07)
  assert compound.url == "https://pubchem.ncbi.nlm.nih.gov/compound/702"


def test_get_pubchem_compound_by_cid(monkeypatch):
  calls = _install_routes(
    monkeypatch,
    [
      {"match": "/compound/name/", "response": _MockResponse(status_code=404)},
      {"match": "/compound/cid/2244/property/", "response": _MockResponse(json_data={"PropertyTable": {"Properties": [ASPIRIN_PROPERTIES]}})},
    ],
  )

  compound = get_pubchem_compound(2244)

  assert compound.cid == 2244
  assert compound.name == "aspirin"
  assert not any("/compound/name/" in call["url"] for call in calls)


def test_get_pubchem_compound_unknown_name_raises(monkeypatch):
  _install_routes(monkeypatch, [{"match": "/compound/name/unknown/cids/TXT", "response": _MockResponse(status_code=404)}])

  with pytest.raises(ValueError, match="Could not find"):
    get_pubchem_compound("unknown")


def test_search_pubchem_similar_orders_by_local_similarity(monkeypatch):
  calls = _install_routes(
    monkeypatch,
    [
      {"match": "/fastsimilarity_2d/", "response": _MockResponse(json_data={"IdentifierList": {"CID": [2244, 702]}})},
      {
        "match": "/property/",
        "response": _MockResponse(json_data={"PropertyTable": {"Properties": [ASPIRIN_PROPERTIES, ETHANOL_PROPERTIES]}}),
      },
    ],
  )

  hits = search_pubchem_similar("CCO", threshold=0.9, max_records=10)

  assert [hit.cid for hit in hits] == [702, 2244]
  assert hits[0].rank == 2
  assert hits[0].similarity == pytest.approx(1.0)
  assert hits[1].rank == 1
  assert hits[1].similarity < hits[0].similarity
  assert all(isinstance(hit, PubChemHit) for hit in hits)

  similarity_request = next(call for call in calls if "/fastsimilarity_2d/" in call["url"])
  assert similarity_request["params"] == {"Threshold": 90, "MaxRecords": 10}


def test_search_pubchem_similar_empty_on_not_found(monkeypatch):
  _install_routes(monkeypatch, [{"match": "/fastsimilarity_2d/", "response": _MockResponse(status_code=404)}])

  assert search_pubchem_similar("CCO") == []


def test_search_pubchem_similar_posts_stereochemical_smiles(monkeypatch):
  query = "O=C1C2=C(C=CS2)/C(C3=CC=CC=C3C1)=C4CCN(C(OCCN(C)C)=O)CC/4"
  calls = _install_routes(
    monkeypatch,
    [{"match": "/fastsimilarity_2d/smiles/cids/JSON", "response": _MockResponse(json_data={"IdentifierList": {"CID": []}})}],
  )

  assert search_pubchem_similar(query) == []

  assert calls == [
    {
      "method": "post",
      "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/cids/JSON",
      "params": {"Threshold": 90, "MaxRecords": 100},
      "data": {"smiles": query},
      "timeout": 30,
    }
  ]


def test_search_pubchem_similar_rejects_invalid_smiles(monkeypatch):
  with pytest.raises(ValueError, match="not a valid SMILES"):
    search_pubchem_similar("this-is-not-smiles")


def test_fetch_pubchem_sdfs_prefers_3d_with_2d_fallback(monkeypatch):
  three_d = "702\n  3D record\n$$$$\n"
  two_d = "2244\n  2D record\n$$$$\n"

  def sdf_response(params: Optional[dict]) -> _MockResponse:
    return _MockResponse(text=three_d if params and params.get("record_type") == "3d" else two_d)

  calls = _install_routes(monkeypatch, [{"match": "/SDF", "response": sdf_response}])

  records = fetch_pubchem_sdfs([702, 2244])

  assert set(records) == {702, 2244}
  assert records[702].startswith("702")
  assert "3D record" in records[702]
  assert "2D record" in records[2244]
  assert [call["params"]["record_type"] for call in calls] == ["3d", "2d"]


@pytest.mark.integration
def test_pubchem_live_roundtrip():
  compound = get_pubchem_compound("ethanol")
  assert compound.cid == 702
  assert compound.smiles

  hits = search_pubchem_similar("CCO", threshold=0.95, max_records=5)
  assert hits
  assert all(hit.cid > 0 and 0.0 <= hit.similarity <= 1.0 for hit in hits)
