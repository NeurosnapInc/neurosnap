"""Pairwise comparison and alignment functions for Neurosnap structures."""

from typing import Optional, Sequence

import numpy as np

from ._common import available_chain_ids, backbone_atom_order, coord_matrix
from .structure import Structure


def _matched_backbone_coords(
  reference: Structure,
  mobile: Structure,
  chains1: Optional[Sequence[str]] = None,
  chains2: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Return matched backbone coordinates for two structures.

  Matching is done by chain mapping plus residue/atom identity, not by raw
  atom-table order. This keeps pairwise comparisons stable even when the two
  files store chains in different orders or one file includes extra
  non-backbone atoms such as hydrogens.
  """
  chains1 = list(chains1 or [])
  chains2 = list(chains2 or [])
  chains1_provided = bool(chains1)
  chains2_provided = bool(chains2)

  available_reference_chains = available_chain_ids(reference)
  available_mobile_chains = available_chain_ids(mobile)

  if chains1:
    for chain_id in chains1:
      if chain_id not in available_reference_chains:
        raise ValueError(f"Chain {chain_id} was not found in the reference structure.")
  else:
    chains1 = available_reference_chains

  if chains2:
    for chain_id in chains2:
      if chain_id not in available_mobile_chains:
        raise ValueError(f"Chain {chain_id} was not found in the mobile structure.")
  else:
    chains2 = available_mobile_chains

  chain_mapping_mode = chains1_provided and chains2_provided
  if chain_mapping_mode and len(chains1) != len(chains2):
    raise ValueError("chains1 and chains2 must contain the same number of chains for pairwise mapping.")

  if chain_mapping_mode:
    reference_chain_specs = [(pair_index, chain_id) for pair_index, chain_id in enumerate(chains1)]
    mobile_chain_specs = [(pair_index, chain_id) for pair_index, chain_id in enumerate(chains2)]
  else:
    # In the default mode, chains must line up by their identifiers in both
    # structures so the residue/atom keys can be matched directly.
    reference_chain_specs = [(chain_id, chain_id) for chain_id in chains1]
    mobile_chain_specs = [(chain_id, chain_id) for chain_id in chains2]

  def backbone_atom_map(structure_model, chain_specs):
    """Build a residue-aware backbone lookup for one structure."""
    chain_lookup = {chain.chain_id: chain for chain in structure_model.chains()}
    atom_map = {}
    for map_key, chain_id in chain_specs:
      chain = chain_lookup.get(chain_id)
      if chain is None:
        continue
      for residue in chain.residues():
        atom_order = backbone_atom_order(residue, include_nucleotides=True)
        if not atom_order:
          continue
        residue_atoms = {atom.atom_name.strip().upper(): atom.coord for atom in residue.atoms()}
        for atom_name in atom_order:
          if atom_name in residue_atoms:
            atom_map[(map_key, residue.res_id, residue.ins_code, atom_name)] = residue_atoms[atom_name]
    return atom_map

  reference_atom_map = backbone_atom_map(reference, reference_chain_specs)
  mobile_atom_map = backbone_atom_map(mobile, mobile_chain_specs)
  if not reference_atom_map:
    raise ValueError("Reference structure does not contain any backbone atoms to align.")
  if not mobile_atom_map:
    raise ValueError("Mobile structure does not contain any backbone atoms to align.")

  common_keys = sorted(reference_atom_map.keys() & mobile_atom_map.keys())
  if not common_keys:
    raise ValueError("Structures do not share common backbone atoms to align.")
  if len(common_keys) != len(reference_atom_map) or len(common_keys) != len(mobile_atom_map):
    raise ValueError("Backbone atom mismatch between structures.")

  reference_coords = np.asarray([reference_atom_map[key] for key in common_keys], dtype=np.float32)
  mobile_coords = np.asarray([mobile_atom_map[key] for key in common_keys], dtype=np.float32)
  return reference_coords, mobile_coords


def align(
  reference: Structure,
  mobile: Structure,
  chains1: Optional[Sequence[str]] = None,
  chains2: Optional[Sequence[str]] = None,
):
  """Align a mobile structure onto a reference structure using polymer backbone atoms.

  When both ``chains1`` and ``chains2`` are provided, they are interpreted as
  explicit pairwise chain mappings in matching order.

  Parameters:
    reference: Reference single-model :class:`Structure`.
    mobile: Mobile single-model :class:`Structure` to transform in-place.
    chains1: Optional reference chain IDs to include in the alignment.
    chains2: Optional mobile chain IDs to include in the alignment.

  Returns:
    ``None``. The mobile structure is transformed in-place.
  """
  if not isinstance(reference, Structure):
    raise TypeError(f"align() expects reference to be a Structure, found {type(reference).__name__}.")
  if not isinstance(mobile, Structure):
    raise TypeError(f"align() expects mobile to be a Structure, found {type(mobile).__name__}.")

  reference_coords, mobile_coords = _matched_backbone_coords(reference, mobile, chains1=chains1, chains2=chains2)
  # Standard Kabsch alignment on the matched backbone coordinates.
  reference_center = reference_coords.mean(axis=0)
  mobile_center = mobile_coords.mean(axis=0)
  centered_reference = reference_coords - reference_center
  centered_mobile = mobile_coords - mobile_center
  covariance = centered_mobile.T @ centered_reference
  u_matrix, _, vt_matrix = np.linalg.svd(covariance)
  rotation = u_matrix @ vt_matrix
  if np.linalg.det(rotation) < 0:
    u_matrix[:, -1] *= -1
    rotation = u_matrix @ vt_matrix
  translation = reference_center - (mobile_center @ rotation)

  all_mobile_coords = coord_matrix(mobile)
  aligned_coords = all_mobile_coords @ rotation.astype(np.float32) + translation.astype(np.float32)
  mobile.atoms["x"] = aligned_coords[:, 0]
  mobile.atoms["y"] = aligned_coords[:, 1]
  mobile.atoms["z"] = aligned_coords[:, 2]


def calculate_rmsd(
  reference: Structure,
  mobile: Structure,
  chains1: Optional[Sequence[str]] = None,
  chains2: Optional[Sequence[str]] = None,
  align_structures: bool = True,
) -> float:
  """Calculate backbone RMSD between two structures.

  Parameters:
    reference: Reference single-model :class:`Structure`.
    mobile: Mobile single-model :class:`Structure`.
    chains1: Optional reference chain IDs to include.
    chains2: Optional mobile chain IDs to include.
    align_structures: If ``True``, align the mobile structure before computing
      RMSD.

  Returns:
    Backbone RMSD in Å using the same residue/atom correspondence as
    :func:`align`.
  """
  if not isinstance(reference, Structure):
    raise TypeError(f"calculate_rmsd() expects reference to be a Structure, found {type(reference).__name__}.")
  if not isinstance(mobile, Structure):
    raise TypeError(f"calculate_rmsd() expects mobile to be a Structure, found {type(mobile).__name__}.")

  if align_structures:
    align(reference, mobile, chains1=chains1, chains2=chains2)

  reference_coords, mobile_coords = _matched_backbone_coords(reference, mobile, chains1=chains1, chains2=chains2)
  if reference_coords.size == 0:
    return 0.0

  diff = reference_coords - mobile_coords
  return float(np.sqrt(np.sum(diff**2) / reference_coords.shape[0]))
