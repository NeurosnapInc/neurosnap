"""Code for LDDT (Local Distance Difference Test) calculation."""

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from neurosnap.constants.structure import (
  BACKBONE_ATOMS_AA,
  BACKBONE_ATOMS_DNA,
  BACKBONE_ATOMS_RNA,
  STANDARD_NUCLEOTIDES,
)
from neurosnap.sequence.protein import getAA
from neurosnap.structure import Atom, Residue, Structure
from neurosnap.structure._common import classify_polymer_residue

_PROTEIN_BACKBONE_FALLBACK = tuple(atom for atom in ("CA", "N", "C") if atom in BACKBONE_ATOMS_AA)

_NUCLEOTIDE_BACKBONE_ATOMS = set(BACKBONE_ATOMS_DNA).union(BACKBONE_ATOMS_RNA)
_NUCLEOTIDE_PREF_BASES = ("C4'", "C3'", "C1'", "C2'", "P", "O4'", "O3'", "O5'")
_NUCLEOTIDE_ATOM_PRIORITY: List[Tuple[str, ...]] = []
_seen_nuc_bases = set()
for base in _NUCLEOTIDE_PREF_BASES:
  if base in _NUCLEOTIDE_BACKBONE_ATOMS and base not in _seen_nuc_bases:
    names = tuple(dict.fromkeys((base, base.replace("'", "*"))))
    _NUCLEOTIDE_ATOM_PRIORITY.append(names)
    _seen_nuc_bases.add(base)
for base in sorted(_NUCLEOTIDE_BACKBONE_ATOMS):
  if base not in _seen_nuc_bases:
    names = tuple(dict.fromkeys((base, base.replace("'", "*"))))
    _NUCLEOTIDE_ATOM_PRIORITY.append(names)
    _seen_nuc_bases.add(base)

_WATER_RESIDUE_NAMES = {"HOH", "WAT"}
_PROTEIN_BACKBONE_NAMES = {atom.upper() for atom in BACKBONE_ATOMS_AA}
SiteKey = Tuple[str, int, str, str, str, str]


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


def _is_hydrogen_atom(atom: Atom) -> bool:
  element = str(atom.element).strip().upper()
  if element == "H":
    return True
  return atom.atom_name.strip().upper().startswith("H")


def _classify_residue_for_lddt(residue: Residue) -> Optional[Literal["protein", "dna", "rna"]]:
  """Classify standard and modified polymer residues for lDDT site selection."""
  polymer_type = classify_polymer_residue(residue)
  if polymer_type is not None:
    return polymer_type

  atom_names = {atom.atom_name.strip().upper() for atom in residue.atoms()}
  if len(atom_names.intersection(_PROTEIN_BACKBONE_NAMES)) >= 2:
    return "protein"
  return None


def _get_atom(residue: Residue, name: str) -> Optional[Atom]:
  """Fetch an atom by name, handling historical prime markers (* vs ')."""
  atom_lookup = {atom.atom_name.strip().upper(): atom for atom in residue.atoms()}
  normalized = name.strip().upper()
  if normalized in atom_lookup:
    return atom_lookup[normalized]
  alt = normalized.replace("'", "*") if "'" in normalized else normalized.replace("*", "'")
  if alt != normalized and alt in atom_lookup:
    return atom_lookup[alt]
  return None


def _select_polymer_representative_atom(residue: Residue, polymer_type: Literal["protein", "dna", "rna"]) -> Optional[Atom]:
  """Return a deterministic representative atom for a polymer residue."""
  if polymer_type == "protein":
    if residue.res_name.strip().upper() != "GLY":
      atom = _get_atom(residue, "CB")
      if atom is not None:
        return atom
    for atom_name in _PROTEIN_BACKBONE_FALLBACK:
      atom = _get_atom(residue, atom_name)
      if atom is not None:
        return atom
    return None

  for atom_names in _NUCLEOTIDE_ATOM_PRIORITY:
    for atom_name in atom_names:
      atom = _get_atom(residue, atom_name)
      if atom is not None:
        return atom
  return None


def _extract_cb_coords_from_structure(
  structure: Structure,
  *,
  chains: Optional[List[str]] = None,
  require_standard_aa: bool = False,
) -> Dict[SiteKey, Tuple[float, float, float]]:
  """Collect aligned analysis-site coordinates from a structure.

  Returns:
    dict keyed by a stable site identifier -> (x,y,z)
  """
  coords: Dict[SiteKey, Tuple[float, float, float]] = {}
  chain_lookup = {chain.chain_id: chain for chain in structure.chains()}
  chain_ids = [chain.chain_id for chain in structure.chains()] if not chains else list(chains)
  for cid in chain_ids:
    chain = chain_lookup.get(cid)
    if chain is None:
      continue
    for residue in chain.residues():
      resname = residue.res_name.strip().upper()
      if resname in _WATER_RESIDUE_NAMES:
        continue

      polymer_type = _classify_residue_for_lddt(residue)
      if polymer_type is not None:
        if require_standard_aa:
          if polymer_type == "protein":
            try:
              getAA(resname, non_standard="convert")
            except ValueError:
              continue
          elif resname not in STANDARD_NUCLEOTIDES:
            continue

        atom = _select_polymer_representative_atom(residue, polymer_type)
        if atom is None:
          continue

        key = (cid, int(residue.res_id), residue.ins_code, "polymer", "", "")
        coords[key] = (float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2]))
        continue

      if require_standard_aa:
        continue

      for atom in residue.atoms():
        if _is_hydrogen_atom(atom):
          continue
        key = (cid, int(residue.res_id), residue.ins_code, "nonpolymer", resname, atom.atom_name.strip().upper())
        coords[key] = (float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2]))
  return coords


def _coords_to_distmat(ordered_keys: List[SiteKey], coord_map: Dict[SiteKey, Tuple[float, float, float]]) -> np.ndarray:
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
  return lddt


def calc_lddt(
  reference: Union[np.ndarray, Structure],
  prediction: Union[np.ndarray, Structure],
  *,
  chains_ref: Optional[List[str]] = None,
  chains_pred: Optional[List[str]] = None,
  R: float = 15.0,
  sep_thresh: int = -1,
  T_set: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
  require_standard_aa: bool = False,
) -> float:
  """Compute lDDT from distance maps or Neurosnap structure containers.

  Args:
    reference: Distance map or single-model structure used as the ground truth.
    prediction: Distance map or single-model structure to compare against the reference.
    chains_ref: Chain identifiers to include when `reference` is a structure.
    chains_pred: Chain identifiers to include when `prediction` is a structure.
    R: Maximum reference distance to consider when defining the L set.
    sep_thresh: Minimum sequence separation between residue pairs.
    T_set: Error thresholds used to compute preserved distance fractions.
    require_standard_aa: Restrict structure inputs to canonical amino acids and
      standard nucleotides when True. When False, modified polymer residues are
      included using polymer-aware proxy atoms and non-polymer residues such as
      ligands contribute heavy-atom sites.

  Returns:
    lDDT score between the reference and prediction. Typical range is [0.0, 1.0],
    where 1.0 indicates perfect local distance agreement and 0.0 indicates no
    preserved distances under the selected thresholds. Returns NaN when no
    residue pairs satisfy the L-set criteria (for example, no pairs within `R`
    and above `sep_thresh`).

  Raises:
    ValueError: If distance maps do not share the same shape.
    TypeError: If the inputs are not both arrays or both structure containers.
  """
  is_ref_array = isinstance(reference, np.ndarray)
  is_pred_array = isinstance(prediction, np.ndarray)
  is_ref_structure = isinstance(reference, Structure)
  is_pred_structure = isinstance(prediction, Structure)

  if is_ref_array and is_pred_array:
    if reference.shape != prediction.shape:
      raise ValueError("Distance maps must share the same shape to compute lDDT.")
    return _calc_lddt_from_maps(reference, prediction, R=R, sep_thresh=sep_thresh, T_set=T_set)

  if is_ref_structure and is_pred_structure:
    ref_cb = _extract_cb_coords_from_structure(reference, chains=chains_ref, require_standard_aa=require_standard_aa)
    pred_cb = _extract_cb_coords_from_structure(prediction, chains=chains_pred, require_standard_aa=require_standard_aa)

    # Intersect keys to ensure 1:1 residue correspondence
    common_keys = sorted(set(ref_cb.keys()).intersection(pred_cb.keys()), key=lambda x: (x[0], x[1]))
    if len(common_keys) < 2:
      raise Exception("Not enough residues/pairs to compute meaningful lDDT")

    # Build distance maps on the common residue set
    true_map = _coords_to_distmat(common_keys, ref_cb)
    pred_map = _coords_to_distmat(common_keys, pred_cb)
    return _calc_lddt_from_maps(true_map, pred_map, R=R, sep_thresh=sep_thresh, T_set=T_set)

  raise TypeError("calc_lddt expects both inputs to be either numpy.ndarray distance maps or single-model Neurosnap Structure objects.")
