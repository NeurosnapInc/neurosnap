"""
Code for LDDT (Local Distance Difference Test) calculation, adapted from https://github.com/ba-lab/disteval/blob/main/LDDT.ipynb
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from neurosnap.protein import Protein

valid_amino_acids = {
  "LLP": "K",
  "TPO": "T",
  "CSS": "C",
  "OCS": "C",
  "CSO": "C",
  "PCA": "E",
  "KCX": "K",
  "CME": "C",
  "MLY": "K",
  "SEP": "S",
  "CSX": "C",
  "CSD": "C",
  "MSE": "M",
  "ALA": "A",
  "ASN": "N",
  "CYS": "C",
  "GLN": "Q",
  "HIS": "H",
  "LEU": "L",
  "MET": "M",
  "MHO": "M",
  "PRO": "P",
  "THR": "T",
  "TYR": "Y",
  "ARG": "R",
  "ASP": "D",
  "GLU": "E",
  "GLY": "G",
  "ILE": "I",
  "LYS": "K",
  "PHE": "F",
  "SER": "S",
  "TRP": "W",
  "VAL": "V",
  "SEC": "U",
}


# Helpers for metrics calculated using numpy scheme
def _get_flattened(dmap):
  if dmap.ndim == 1:
    return dmap
  elif dmap.ndim == 2:
    return dmap[np.triu_indices_from(dmap, k=1)]
  else:
    assert False, "ERROR: the passes array has dimension not equal to 2 or 1!"


def _get_separations(dmap):
  t_indices = np.triu_indices_from(dmap, k=1)
  separations = np.abs(t_indices[0] - t_indices[1])
  return separations


# return a 1D boolean array indicating where the sequence separation in the
# upper triangle meets the threshold comparison
def _get_sep_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {"gt", "lt", "ge", "le"}, "ERROR: Unknown comparator for thresholding!"
  separations = _get_separations(dmap)
  if comparator == "gt":
    threshed = separations > thresh
  elif comparator == "lt":
    threshed = separations < thresh
  elif comparator == "ge":
    threshed = separations >= thresh
  elif comparator == "le":
    threshed = separations <= thresh
  return threshed


# return a 1D boolean array indicating where the distance in the
# upper triangle meets the threshold comparison
def _get_dist_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {"gt", "lt", "ge", "le"}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = _get_flattened(dmap)
  if comparator == "gt":
    threshed = dmap_flat > thresh
  elif comparator == "lt":
    threshed = dmap_flat < thresh
  elif comparator == "ge":
    threshed = dmap_flat >= thresh
  elif comparator == "le":
    threshed = dmap_flat <= thresh
  return threshed


def _aa3_to_aa1(resname: str) -> Optional[str]:
  """Map 3-letter aa (incl. many non-standards you listed) to 1-letter; returns None if unknown."""
  return valid_amino_acids.get(resname)


def _extract_cb_coords_from_protein(
  prot: Protein,
  *,
  model: Optional[int] = None,
  chains: Optional[List[str]] = None,
  require_standard_aa: bool = True,
) -> Dict[Tuple[str, int], Tuple[float, float, float]]:
  """Collect per-residue coordinates (Cβ, fallback Cα) for amino acids.

  Returns:
    dict keyed by (chain_id, res_id) -> (x,y,z)
  """
  if model is None:
    model = prot.models()[0]
  assert model in prot.models(), f"Model {model} not found in protein {prot.title}"

  coords: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
  model_obj = prot.structure[model]

  # Decide which chains to traverse
  chain_ids = [c.id for c in model_obj] if not chains else chains
  for cid in chain_ids:
    if cid not in prot.chains(model):
      # Skip silently if a requested chain isn't present
      continue
    chain = model_obj[cid]
    for res in chain:
      # Only amino acids
      if getattr(res, "resname", None) is None:
        continue
      aa1 = _aa3_to_aa1(res.resname)
      if require_standard_aa and aa1 is None:
        continue

      # Prefer CB, except GLY (no CB) -> CA
      atom = None
      if res.resname != "GLY" and "CB" in res:
        atom = res["CB"]
      elif "CA" in res:
        atom = res["CA"]
      if atom is None:
        continue

      key = (cid, res.id[1])  # (chain, residue sequence number)
      coords[key] = (float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2]))
  return coords


def _coords_to_distmat(ordered_keys: List[Tuple[str, int]], coord_map: Dict[Tuple[str, int], Tuple[float, float, float]]) -> np.ndarray:
  """Build an NxN Euclidean distance matrix from an ordered list of residue keys and a coord map."""
  if not ordered_keys:
    return np.empty((0, 0))
  pts = np.array([coord_map[k] for k in ordered_keys], dtype=float)  # (N,3)
  # Pairwise distances with broadcasting
  diff = pts[:, None, :] - pts[None, :, :]
  dist = np.sqrt(np.sum(diff * diff, axis=-1))
  return dist


def _calc_lddt_from_maps(
  true_map: np.ndarray,
  pred_map: np.ndarray,
  *,
  R: float = 15.0,
  sep_thresh: int = -1,
  T_set: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
  precision: int = 4,
) -> float:
  """
  Mariani V, Biasini M, Barbato A, Schwede T.
  lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests.
  Bioinformatics. 2013 Nov 1;29(21):2722-8.
  doi: 10.1093/bioinformatics/btt473.
  Epub 2013 Aug 27.
  PMID: 23986568; PMCID: PMC3799472.
  """

  # Helper for number preserved in a threshold
  def get_n_preserved(ref_flat, mod_flat, thresh):
    err = np.abs(ref_flat - mod_flat)
    n_preserved = (err < thresh).sum()
    return n_preserved

  # flatten upper triangles
  true_flat_map = _get_flattened(true_map)
  pred_flat_map = _get_flattened(pred_map)

  # Find set L
  S_thresh_indices = _get_sep_thresh_b_indices(true_map, sep_thresh, "gt")
  R_thresh_indices = _get_dist_thresh_b_indices(true_flat_map, R, "lt")

  L_indices = S_thresh_indices & R_thresh_indices

  L_n = L_indices.sum()
  if L_n == 0:
    return float("nan")

  true_flat_in_L = true_flat_map[L_indices]
  pred_flat_in_L = pred_flat_map[L_indices]

  # Calculated LDDT
  preserved_fractions = []
  for _thresh in T_set:
    _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)
    _f_preserved = _n_preserved / L_n
    preserved_fractions.append(_f_preserved)

  lddt = np.mean(preserved_fractions)
  if precision > 0:
    lddt = round(lddt, precision)
  return lddt


def calc_lddt(
  reference: Union[np.ndarray, Protein],
  prediction: Union[np.ndarray, Protein],
  *,
  model_ref: Optional[int] = None,
  model_pred: Optional[int] = None,
  chains_ref: Optional[List[str]] = None,
  chains_pred: Optional[List[str]] = None,
  R: float = 15.0,
  sep_thresh: int = -1,
  T_set: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
  precision: int = 4,
  require_standard_aa: bool = True,
) -> float:
  """Compute lDDT from distance maps or Protein structures.

  Args:
    reference: Distance map or Protein used as the ground truth.
    prediction: Distance map or Protein to compare against the reference.
    model_ref: Model index to select when `reference` is a Protein.
    model_pred: Model index to select when `prediction` is a Protein.
    chains_ref: Chain identifiers to include when `reference` is a Protein.
    chains_pred: Chain identifiers to include when `prediction` is a Protein.
    R: Maximum reference distance to consider when defining the L set.
    sep_thresh: Minimum sequence separation between residue pairs.
    T_set: Error thresholds used to compute preserved distance fractions.
    precision: Decimal precision of the reported score; use 0 or negative to skip rounding.
    require_standard_aa: Skip residues with unknown amino-acid codes when True.

  Returns:
    lDDT score between the reference and prediction.

  Raises:
    ValueError: If distance maps do not share the same shape.
    TypeError: If the inputs are not both arrays or both Protein instances.
  """
  is_ref_array = isinstance(reference, np.ndarray)
  is_pred_array = isinstance(prediction, np.ndarray)
  is_ref_protein = isinstance(reference, Protein)
  is_pred_protein = isinstance(prediction, Protein)

  if is_ref_array and is_pred_array:
    if reference.shape != prediction.shape:
      raise ValueError("Distance maps must share the same shape to compute lDDT.")
    return _calc_lddt_from_maps(reference, prediction, R=R, sep_thresh=sep_thresh, T_set=T_set, precision=precision)

  if is_ref_protein and is_pred_protein:
    # Extract per-residue coordinates for both proteins
    ref_cb = _extract_cb_coords_from_protein(reference, model=model_ref, chains=chains_ref, require_standard_aa=require_standard_aa)
    pred_cb = _extract_cb_coords_from_protein(prediction, model=model_pred, chains=chains_pred, require_standard_aa=require_standard_aa)

    # Intersect keys to ensure 1:1 residue correspondence
    common_keys = sorted(set(ref_cb.keys()).intersection(pred_cb.keys()), key=lambda x: (x[0], x[1]))
    if len(common_keys) < 2:
      # Not enough residues/pairs to compute meaningful lDDT
      return float("nan")

    # Build distance maps on the common residue set
    true_map = _coords_to_distmat(common_keys, ref_cb)
    pred_map = _coords_to_distmat(common_keys, pred_cb)
    return _calc_lddt_from_maps(true_map, pred_map, R=R, sep_thresh=sep_thresh, T_set=T_set, precision=precision)

  raise TypeError("calc_lddt expects both inputs to be either numpy.ndarray distance maps or Protein objects.")
