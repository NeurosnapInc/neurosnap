"""PubChem compound lookup and similarity-search helpers."""

import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
from urllib.parse import quote

import requests
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

from neurosnap._compat import compat_dataclass

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_COMPOUND_URL = "https://pubchem.ncbi.nlm.nih.gov/compound"

# PubChem accepts up to a few hundred records per request, but shorter lists keep
# the request URLs comfortably small.
_PROPERTY_CHUNK_SIZE = 50
_SDF_CHUNK_SIZE = 50
_PROPERTY_FIELDS = ("MolecularFormula", "MolecularWeight", "SMILES", "IUPACName")
_SMILES_FIELDS = ("SMILES", "ConnectivitySMILES", "CanonicalSMILES", "IsomericSMILES")

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@compat_dataclass(frozen=True, slots=True)
class PubChemCompound:
  """Minimal PubChem compound record.

  Attributes:
    cid: PubChem Compound Identifier.
    name: Preferred IUPAC name (or the best available record title).
    smiles: Isomeric SMILES, falling back to the connectivity SMILES.
    molecular_formula: Hill-notation molecular formula.
    molecular_weight: Molecular weight in g/mol.
  """

  cid: int
  name: str
  smiles: str
  molecular_formula: str
  molecular_weight: float

  @property
  def url(self) -> str:
    """Return the public PubChem page URL for the compound."""
    return f"{PUBCHEM_COMPOUND_URL}/{self.cid}"


@compat_dataclass(frozen=True, slots=True)
class PubChemHit:
  """A single compound returned by a PubChem similarity search.

  Attributes:
    query: SMILES string the hit was found for.
    cid: PubChem Compound Identifier.
    rank: PubChem's 1-based similarity rank for the hit.
    name: Preferred IUPAC name (or the best available record title).
    smiles: Isomeric SMILES, falling back to the connectivity SMILES.
    molecular_formula: Hill-notation molecular formula.
    molecular_weight: Molecular weight in g/mol.
    similarity: Local RDKit Morgan/Tanimoto score in the ``0-1`` range.
  """

  query: str
  cid: int
  rank: int
  name: str
  smiles: str
  molecular_formula: str
  molecular_weight: float
  similarity: float

  @property
  def url(self) -> str:
    """Return the public PubChem page URL for the compound."""
    return f"{PUBCHEM_COMPOUND_URL}/{self.cid}"


def get_pubchem_compound(query: Union[str, int], *, timeout: int = 30) -> PubChemCompound:
  """Fetch a PubChem compound by name or Compound Identifier (CID).

  Args:
    query: Compound name or numeric PubChem CID.
    timeout: HTTP timeout in seconds.

  Returns:
    The resolved :class:`PubChemCompound` record.

  Raises:
    ValueError: If the compound cannot be found or its record is incomplete.
  """
  cid = _resolve_cid(query, timeout=timeout)
  properties = _fetch_properties([cid], timeout=timeout)
  entry = properties.get(cid)
  if entry is None:
    raise ValueError(f"Could not retrieve compound data for PubChem CID {cid}.")
  return _record_from_properties(cid, entry)


def search_pubchem_similar(
  smiles: str,
  *,
  threshold: float = 0.9,
  max_records: int = 100,
  timeout: int = 30,
) -> List[PubChemHit]:
  """Find PubChem compounds structurally similar to a query SMILES.

  Results are ordered by the local RDKit Tanimoto score (descending) while
  retaining PubChem's own similarity rank for each hit. The local score uses
  RDKit Morgan fingerprints and approximates PubChem's ranking; the two
  fingerprints are not identical.

  Args:
    smiles: Query SMILES string.
    threshold: Similarity threshold in the ``0-1`` range, scaled to PubChem's
      ``1-100`` ``Threshold`` parameter.
    max_records: Maximum number of hits to request from PubChem.
    timeout: HTTP timeout in seconds.

  Returns:
    A list of :class:`PubChemHit` records, possibly empty.

  Raises:
    ValueError: If ``smiles`` is empty or cannot be parsed.
  """
  query_smiles = str(smiles).strip()
  if not query_smiles:
    raise ValueError("A SMILES string is required to run a PubChem similarity search.")
  if Chem.MolFromSmiles(query_smiles) is None:
    raise ValueError(f'"{query_smiles}" is not a valid SMILES string.')

  pubchem_threshold = _scale_threshold(threshold)
  pubchem_max_records = max(1, int(max_records))
  url = f"{PUBCHEM_BASE_URL}/compound/fastsimilarity_2d/smiles/{quote(query_smiles, safe='')}/cids/JSON"
  response = requests.get(
    url,
    params={"Threshold": pubchem_threshold, "MaxRecords": pubchem_max_records},
    timeout=timeout,
  )
  if response.status_code == 404:
    return []
  response.raise_for_status()

  cids = [int(cid) for cid in response.json().get("IdentifierList", {}).get("CID", [])]
  if not cids:
    return []

  properties = _fetch_properties(cids, timeout=timeout)
  query_fingerprint = _fingerprint(query_smiles)

  hits: List[PubChemHit] = []
  for rank, cid in enumerate(cids, start=1):
    entry = properties.get(cid, {})
    hit_smiles = _extract_smiles(entry)
    similarity = _similarity(query_fingerprint, hit_smiles)
    hits.append(
      PubChemHit(
        query=query_smiles,
        cid=cid,
        rank=rank,
        name=_extract_name(entry),
        smiles=hit_smiles,
        molecular_formula=str(entry.get("MolecularFormula", "")),
        molecular_weight=_extract_weight(entry),
        similarity=similarity,
      )
    )

  # Stable sort keeps PubChem's rank order for hits with equal local scores.
  hits.sort(key=lambda hit: hit.similarity, reverse=True)
  return hits


def fetch_pubchem_sdfs(
  cids: Iterable[int],
  *,
  prefer_3d: bool = True,
  timeout: int = 30,
) -> Dict[int, str]:
  """Download SDF records for one or more PubChem CIDs.

  Records are retrieved in batches. When ``prefer_3d`` is set, 3D conformers are
  requested first and any CID without a 3D record falls back to its 2D record.

  Args:
    cids: PubChem Compound Identifiers to download.
    prefer_3d: If ``True``, request 3D SDF records before falling back to 2D.
    timeout: HTTP timeout in seconds per batch request.

  Returns:
    Dictionary mapping each resolved CID to its SDF record text.
  """
  unique_cids = list(dict.fromkeys(int(cid) for cid in cids))
  records: Dict[int, str] = {}

  if prefer_3d and unique_cids:
    for chunk in _chunked(unique_cids, _SDF_CHUNK_SIZE):
      records.update(_fetch_sdf_chunk(chunk, record_type="3d", timeout=timeout))

  missing = [cid for cid in unique_cids if cid not in records]
  for chunk in _chunked(missing, _SDF_CHUNK_SIZE):
    records.update(_fetch_sdf_chunk(chunk, record_type="2d", timeout=timeout))

  return records


def _resolve_cid(query: Union[str, int], *, timeout: int) -> int:
  """Resolve a name or CID string to a numeric PubChem CID."""
  text = str(query).strip()
  if not text:
    raise ValueError("A PubChem compound name or CID is required.")

  if text.isdigit():
    return int(text)

  response = requests.get(f"{PUBCHEM_BASE_URL}/compound/name/{quote(text, safe='')}/cids/TXT", timeout=timeout)
  if response.status_code == 404:
    raise ValueError(f'Could not find "{text}" in PubChem. Please check the compound name and try again.')
  response.raise_for_status()

  cids = [line.strip() for line in response.text.splitlines() if line.strip()]
  if not cids:
    raise ValueError(f'Could not find "{text}" in PubChem. Please check the compound name and try again.')
  return int(cids[0])


def _fetch_properties(cids: List[int], *, timeout: int) -> Dict[int, Dict[str, Any]]:
  """Fetch property rows for one or more CIDs in batches."""
  properties: Dict[int, Dict[str, Any]] = {}
  for chunk in _chunked(cids, _PROPERTY_CHUNK_SIZE):
    id_list = ",".join(str(cid) for cid in chunk)
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{id_list}/property/{','.join(_PROPERTY_FIELDS)}/JSON"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
      continue
    response.raise_for_status()
    for entry in response.json().get("PropertyTable", {}).get("Properties", []):
      properties[int(entry["CID"])] = entry
  return properties


def _fetch_sdf_chunk(cids: List[int], *, record_type: str, timeout: int) -> Dict[int, str]:
  """Fetch a batch of SDF records and split them into per-CID entries."""
  id_list = ",".join(str(cid) for cid in cids)
  url = f"{PUBCHEM_BASE_URL}/compound/cid/{id_list}/SDF"
  response = requests.get(url, params={"record_type": record_type}, timeout=timeout)
  if response.status_code == 404:
    return {}
  response.raise_for_status()

  records: Dict[int, str] = {}
  for raw in response.text.split("$$$$"):
    record = raw.strip("\n")
    if not record.strip():
      continue
    # PubChem writes the CID as the first line of each SDF record.
    first_line = record.splitlines()[0].strip()
    if not first_line.isdigit():
      continue
    records[int(first_line)] = record + "\n$$$$\n"
  return records


def _record_from_properties(cid: int, entry: Dict[str, Any]) -> PubChemCompound:
  """Build a :class:`PubChemCompound` from a PubChem property row."""
  return PubChemCompound(
    cid=cid,
    name=_extract_name(entry),
    smiles=_extract_smiles(entry),
    molecular_formula=str(entry.get("MolecularFormula", "")),
    molecular_weight=_extract_weight(entry),
  )


def _extract_name(entry: Dict[str, Any]) -> str:
  """Return the best available display name from a property row."""
  return str(entry.get("IUPACName") or entry.get("Title") or "")


def _extract_smiles(entry: Dict[str, Any]) -> str:
  """Return the best available SMILES value from a property row."""
  for field in _SMILES_FIELDS:
    value = entry.get(field)
    if value:
      return str(value)
  return ""


def _extract_weight(entry: Dict[str, Any]) -> float:
  """Return the molecular weight as a float, defaulting to ``0.0``."""
  try:
    return float(entry.get("MolecularWeight", 0.0))
  except (TypeError, ValueError):
    return 0.0


def _fingerprint(smiles: str) -> Optional[Any]:
  """Return a Morgan fingerprint for a SMILES string, or ``None`` if invalid."""
  if not smiles:
    return None
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  return _MORGAN_GENERATOR.GetFingerprint(mol)


def _similarity(query_fingerprint: Optional[Any], smiles: str) -> float:
  """Return the local Tanimoto similarity between a query and hit SMILES."""
  if query_fingerprint is None:
    return 0.0
  hit_fingerprint = _fingerprint(smiles)
  if hit_fingerprint is None:
    return 0.0
  return float(TanimotoSimilarity(query_fingerprint, hit_fingerprint))


def _scale_threshold(threshold: float) -> int:
  """Scale a ``0-1`` similarity threshold to PubChem's ``1-100`` range."""
  try:
    value = float(threshold)
  except (TypeError, ValueError):
    value = 0.9
  if not math.isfinite(value):
    value = 0.9
  value = min(max(value, 0.0), 1.0)
  return max(1, min(round(value * 100), 100))


def _chunked(values: List[int], size: int) -> Iterator[List[int]]:
  """Yield successive fixed-size slices of a list."""
  for start in range(0, len(values), size):
    yield values[start : start + size]


__all__ = [
  "PUBCHEM_BASE_URL",
  "PUBCHEM_COMPOUND_URL",
  "PubChemCompound",
  "PubChemHit",
  "fetch_pubchem_sdfs",
  "get_pubchem_compound",
  "search_pubchem_similar",
]
