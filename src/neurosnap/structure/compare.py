"""Pairwise comparison and alignment functions for Neurosnap structures."""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from ._common import available_chain_ids, backbone_atom_order, coord_matrix, resolve_model, resolve_model_id, set_model_coordinates
from .analysis import get_backbone


def align(
  reference,
  mobile,
  chains1: Optional[Sequence[str]] = None,
  chains2: Optional[Sequence[str]] = None,
  model1: Optional[int] = None,
  model2: Optional[int] = None,
):
  """Align a mobile model onto a reference model using polymer backbone atoms.

  When both ``chains1`` and ``chains2`` are provided, they are interpreted as
  explicit pairwise chain mappings in matching order.
  """
  reference_model = resolve_model(reference, model=model1)
  mobile_model = resolve_model(mobile, model=model2)

  chains1 = list(chains1 or [])
  chains2 = list(chains2 or [])
  chains1_provided = bool(chains1)
  chains2_provided = bool(chains2)

  available_reference_chains = available_chain_ids(reference_model)
  available_mobile_chains = available_chain_ids(mobile_model)

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
    reference_chain_specs = [(chain_id, chain_id) for chain_id in chains1]
    mobile_chain_specs = [(chain_id, chain_id) for chain_id in chains2]

  reference_atom_map = _backbone_atom_map(reference_model, reference_chain_specs)
  mobile_atom_map = _backbone_atom_map(mobile_model, mobile_chain_specs)
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
  rotation, translation = _kabsch_transform(reference_coords, mobile_coords)

  all_mobile_coords = coord_matrix(mobile_model)
  aligned_coords = all_mobile_coords @ rotation + translation
  set_model_coordinates(mobile, aligned_coords, model=resolve_model_id(mobile, model=model2))


def calculate_rmsd(
  reference,
  mobile,
  chains1: Optional[Sequence[str]] = None,
  chains2: Optional[Sequence[str]] = None,
  model1: Optional[int] = None,
  model2: Optional[int] = None,
  align_structures: bool = True,
) -> float:
  """Calculate backbone RMSD between two structures."""
  if align_structures:
    align(reference, mobile, chains1=chains1, chains2=chains2, model1=model1, model2=model2)

  reference_coords = get_backbone(reference, chains=chains1, model=model1, include_nucleotides=True)
  mobile_coords = get_backbone(mobile, chains=chains2, model=model2, include_nucleotides=True)
  if reference_coords.shape != mobile_coords.shape:
    raise ValueError("Structures must have the same number of backbone atoms for RMSD calculation.")
  if reference_coords.size == 0:
    return 0.0

  diff = reference_coords - mobile_coords
  return float(np.sqrt(np.sum(diff**2) / reference_coords.shape[0]))


def _backbone_atom_map(structure_model, chain_specs: Iterable[Tuple[object, str]]) -> dict[tuple[object, int, str, str], np.ndarray]:
  """Return a backbone-atom lookup keyed by chain mapping and residue identity."""
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
      residue_key = (map_key, residue.res_id, residue.ins_code)
      for atom_name in atom_order:
        if atom_name in residue_atoms:
          atom_map[(residue_key[0], residue_key[1], residue_key[2], atom_name)] = residue_atoms[atom_name]
  return atom_map


def _kabsch_transform(reference_coords: np.ndarray, mobile_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Return the optimal rotation and translation for mobile-to-reference alignment."""
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
  return rotation.astype(np.float32), translation.astype(np.float32)
