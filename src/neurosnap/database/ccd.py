"""Chemical Component Dictionary metadata helpers."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import requests
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

from neurosnap._compat import compat_dataclass
from neurosnap.constants.sequence import AA_RECORDS_CANONICAL, AA_RECORDS_FORCEFIELD_VARIANTS, AARecord
from neurosnap.log import logger

CCD_ENTRIES_URL = "https://neurosnap.ai/assets/ccd/entries.json"
DEFAULT_CCD_CACHE = "~/.cache/neurosnap/ccd_entries.json"

_MEMORY_INDEX_CACHE: Dict[str, Tuple[str, Dict[str, "CCD"]]] = {}


@compat_dataclass(frozen=True, slots=True)
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
    mol = Chem.MolFromSmiles(self.smiles)
    if mol is None:
      raise ValueError(f'Could not parse canonical SMILES for CCD "{self.code}".')
    return mol

  def smiles_canonical(self) -> str:
    """Return the RDKit-canonicalized SMILES string for this CCD entry."""
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
  payload = _load_ccd_payload(cache_path=resolved_cache_path, overwrite=overwrite, max_age_days=max_age_days, timeout=timeout)
  created_at = str(payload["created_at"])

  cached = _MEMORY_INDEX_CACHE.get(resolved_cache_path)
  if cached is not None and cached[0] == created_at:
    return cached[1]

  code_map: Dict[str, CCD] = {}
  for code, entry in payload["entries"].items():
    ccd = CCD(
      code=str(code).upper(),
      name=str(entry.get("name", "")),
      smiles=str(entry.get("smiles", "")),
    )
    code_map[ccd.code] = ccd

  _MEMORY_INDEX_CACHE[resolved_cache_path] = (created_at, code_map)
  return code_map


def get_ccd(
  code: str,
  *,
  cache_path: str = DEFAULT_CCD_CACHE,
  overwrite: bool = False,
  max_age_days: int = 7,
  timeout: int = 30,
) -> Optional[CCD]:
  """Return a CCD entry by its component code."""
  return get_ccd_entries(cache_path=cache_path, overwrite=overwrite, max_age_days=max_age_days, timeout=timeout).get(str(code).upper().strip())


def get_ccd_canonical_aa(
  ccd: Union[str, CCD],
  *,
  standard: bool = False,
  cache_path: str = DEFAULT_CCD_CACHE,
  overwrite: bool = False,
  max_age_days: int = 7,
  timeout: int = 30,
) -> AARecord:
  """Return the most similar canonical or standard amino acid for a CCD entry.

  If the input CCD code already has an explicit amino-acid mapping in the
  amino-acid residue tables, that mapping is reused directly. Otherwise, the
  CCD entry is compared against canonical amino-acid CCD entries using RDKit
  Morgan fingerprints and the highest-similarity amino acid is
  returned. Selenium-containing residues are also compared after conservative
  selenium-to-sulfur normalization so common analogs such as MSE can map to
  their sulfur parent without being hard-coded into the amino-acid constants.

  Parameters:
    ccd: CCD code string or a :class:`CCD` instance.
    standard: If ``True``, collapse results to the 20 standard amino
      acids. If ``False`` (default), allow all canonical residues, including ``SEC`` and
      ``PYL``.
    cache_path: Local cache file path for the CCD JSON payload.
    overwrite: If ``True``, force a fresh CCD payload download.
    max_age_days: Maximum accepted payload age in days.
    timeout: HTTP timeout in seconds for CCD payload downloads.

  Returns:
    The best-matching amino-acid record under the requested mode.

  Raises:
    TypeError: If ``ccd`` is not a string or :class:`CCD`.
    ValueError: If the CCD code is unknown or its SMILES cannot be parsed.
  """
  if isinstance(ccd, str):
    query_code = ccd.upper().strip()
    ccd_entry = get_ccd(query_code, cache_path=cache_path, overwrite=overwrite, max_age_days=max_age_days, timeout=timeout)
    if ccd_entry is None:
      raise ValueError(f'Unknown CCD code "{query_code}".')
  elif isinstance(ccd, CCD):
    query_code = ccd.code.upper().strip()
    ccd_entry = ccd
  else:
    raise TypeError(f"Expected `ccd` to be a str or CCD, found {type(ccd).__name__}.")

  aa_record = AA_RECORDS_CANONICAL.get_by_abr(query_code) or AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr(
    query_code
  )
  if aa_record is not None:
    mapped_record = AA_RECORDS_CANONICAL.get_standard_record(aa_record) if standard else AA_RECORDS_CANONICAL.get_canonical_record(aa_record)
    if mapped_record is not None:
      return mapped_record

  entries = get_ccd_entries(cache_path=cache_path, overwrite=overwrite, max_age_days=max_age_days, timeout=timeout)
  fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
  query_mol = ccd_entry.to_mol()
  query_fps = [fp_generator.GetFingerprint(query_mol)]
  if any(atom.GetAtomicNum() == 34 for atom in query_mol.GetAtoms()):
    sulfur_query_mol = Chem.RWMol(query_mol)
    for atom in sulfur_query_mol.GetAtoms():
      if atom.GetAtomicNum() == 34:
        atom.SetAtomicNum(16)
    normalized_query_mol = sulfur_query_mol.GetMol()
    try:
      Chem.SanitizeMol(normalized_query_mol)
      query_fps.append(fp_generator.GetFingerprint(normalized_query_mol))
    except Exception:
      logger.debug('Failed to sanitize selenium-normalized CCD "%s"; using original molecule only.', query_code)

  best_match: Optional[AARecord] = None
  best_similarity = -1.0
  for aa in AA_RECORDS_CANONICAL.values():
    if standard and aa.standard_equiv_abr is not None:
      continue
    standard_ccd = entries.get(aa.abr)
    if standard_ccd is None:
      continue

    standard_fp = fp_generator.GetFingerprint(standard_ccd.to_mol())
    similarity = max(TanimotoSimilarity(query_fp, standard_fp) for query_fp in query_fps)
    if similarity > best_similarity:
      best_similarity = similarity
      best_match = aa

  if best_match is None:
    raise ValueError("Could not compare CCD entry against amino acids because no matching canonical amino-acid CCD entries were available.")

  return best_match


def get_ccd_rcsb(ccd_code: str, fpath: str):
  """
  Fetches the ideal SDF (Structure Data File) for a given CCD (Chemical Component Dictionary) code
  and saves it to the specified file path.

  This function retrieves the idealized structure of a chemical component from the RCSB Protein
  Data Bank (PDB) by downloading the corresponding SDF file. The downloaded file is then saved
  to the specified location.

  Parameters:
      ccd_code (str): The three-letter CCD code representing the chemical component (e.g., "ATP").
      fpath (str): The file path where the downloaded SDF file will be saved.

  Raises:
      HTTPError: If the request to fetch the SDF file fails (e.g., 404 or connection error).
      IOError: If there is an issue saving the SDF file to the specified file path.

  Example:
      >>> fetch_ccd("ATP", "ATP_ideal.sdf")
      Fetches the ideal SDF file for the ATP molecule and saves it as "ATP_ideal.sdf".

  External Resources:
      - SDF File Download: https://files.rcsb.org/ligands/download/{CCD_CODE}_ideal.sdf
  """
  ccd_code = ccd_code.upper()
  logger.info(f"Fetching CCD with code {ccd_code} from rcsb.org...")
  r = requests.get(f"https://files.rcsb.org/ligands/download/{ccd_code}_ideal.sdf")
  r.raise_for_status()
  with open(fpath, "wb") as f:
    f.write(r.content)


def _load_ccd_payload(*, cache_path: str, overwrite: bool, max_age_days: int, timeout: int) -> dict:
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

  logger.info("Fetching and caching CCD entries.")
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
  "get_ccd_canonical_aa",
  "get_ccd_entries",
  "get_ccd",
]
