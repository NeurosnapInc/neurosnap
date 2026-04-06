"""Chemical Component Dictionary metadata helpers."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

import requests
from rdkit import Chem

from neurosnap.log import logger

CCD_ENTRIES_URL = "https://neurosnap.ai/assets/ccd/entries.json"
DEFAULT_CCD_CACHE = "~/.cache/neurosnap/ccd_entries.json"

_MEMORY_INDEX_CACHE: Dict[str, Tuple[str, Dict[str, "CCD"], Dict[str, "CCD"]]] = {}


@dataclass(frozen=True, slots=True)
class CCD:
  """Minimal Chemical Component Dictionary entry.

  Attributes:
    code: CCD identifier, typically 1-5 characters.
    name: Human-readable component name.
    smiles: SMILES string for the component (technically canonicalized but the canonicalization algorithm used by wwPDB is inconsistent with that of RDkit).
  """

  code: str
  name: str
  smiles: str

  def to_mol(self) -> Chem.Mol:
    """Return an RDKit molecule parsed from the canonical SMILES string.

    Returns:
      RDKit molecule for the CCD entry.

    Raises:
      ValueError: If the stored canonical SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(self.smiles_canonical)
    if mol is None:
      raise ValueError(f'Could not parse canonical SMILES for CCD "{self.code}".')
    return mol

  def smiles_canonical(self) -> Optional[str]:
    """Return a canonical RDKit SMILES string or raises a ValueError exception if invalid."""
    mol = Chem.MolFromSmiles(self.smiles)
    if mol is None:
      raise ValueError("Failed to canonicalize CCD entry.")
    return Chem.MolToSmiles(mol, canonical=True)


def get_ccd_entries(
  *,
  cache_path: str = DEFAULT_CCD_CACHE,
  overwrite: bool = False,
  max_age_days: int = 7,
  timeout: int = 30,
) -> Dict[str, CCD]:
  """Fetch and cache CCD metadata entries.

  The CCD payload is cached locally and refreshed when the cached payload
  exceeds ``max_age_days`` based on its embedded ``created_at`` timestamp.

  Parameters:
    cache_path: Local cache file path for the raw JSON payload.
    overwrite: If ``True``, force a fresh download.
    max_age_days: Maximum accepted payload age in days.
    timeout: HTTP timeout in seconds for the download request.

  Returns:
    Dictionary mapping CCD code to :class:`CCD`.
  """
  resolved_cache_path = str(Path(cache_path).expanduser().resolve())
  payload = _load_ccd_payload(resolved_cache_path, overwrite=overwrite, max_age_days=max_age_days, timeout=timeout)
  created_at = str(payload["created_at"])

  cached = _MEMORY_INDEX_CACHE.get(resolved_cache_path)
  if cached is not None and cached[0] == created_at:
    return cached[1]

  code_map: Dict[str, CCD] = {}
  smiles_map: Dict[str, CCD] = {}
  for code, entry in payload["entries"].items():
    ccd = CCD(
      code=str(code).upper(),
      name=str(entry.get("name", "")),
      smiles_canonical=str(entry.get("smiles_canonical", "")),
    )
    code_map[ccd.code] = ccd

  _MEMORY_INDEX_CACHE[resolved_cache_path] = (created_at, code_map, smiles_map)
  return code_map


def get_ccd(
  code: str,
  cache_path: str = DEFAULT_CCD_CACHE,
  *,
  overwrite: bool = False,
  max_age_days: int = 7,
  timeout: int = 30,
) -> Optional[CCD]:
  """Return a CCD entry by its component code."""
  return get_ccd_entries(cache_path, overwrite=overwrite, max_age_days=max_age_days, timeout=timeout).get(str(code).upper().strip())


def _load_ccd_payload(cache_path: str, *, overwrite: bool, max_age_days: int, timeout: int) -> dict:
  """Load a cached CCD payload or refresh it from the remote endpoint."""
  path = Path(cache_path)
  if not overwrite and path.exists():
    try:
      payload = json.loads(path.read_text())
      if _payload_is_fresh(payload, max_age_days=max_age_days):
        return payload
      logger.info("Cached CCD entries are stale; refreshing.")
    except (json.JSONDecodeError, OSError, ValueError):
      logger.warning("Could not read cached CCD entries; refreshing from remote source.")

  response = requests.get(CCD_ENTRIES_URL, timeout=timeout)
  response.raise_for_status()
  payload = response.json()
  if "entries" not in payload or "created_at" not in payload:
    raise ValueError("CCD entries payload is missing required keys.")

  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload))
  return payload


def _payload_is_fresh(payload: dict, *, max_age_days: int) -> bool:
  """Return ``True`` if a CCD payload is recent enough to reuse."""
  created_at = payload.get("created_at")
  if not created_at:
    return False

  created_at_dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
  return datetime.now(timezone.utc) - created_at_dt <= timedelta(days=max_age_days)


__all__ = [
  "CCD",
  "CCD_ENTRIES_URL",
  "DEFAULT_CCD_CACHE",
  "get_ccd_entries",
  "get_ccd",
]
