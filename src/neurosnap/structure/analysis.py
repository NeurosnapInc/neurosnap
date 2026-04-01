"""General analysis helpers for Neurosnap structures."""

from typing import Optional

import numpy as np

from ._common import resolve_model


def calculate_distance_matrix(structure, model: Optional[int] = None, chain: Optional[str] = None) -> np.ndarray:
  """Calculate the CA-atom distance matrix for a selected model and chain set.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or :class:`StructureStack`.
    model: Optional model ID when selecting from an ensemble or stack.
    chain: Optional chain ID to restrict the calculation to.

  Returns:
    A square NumPy array of pairwise CA distances in Å.
  """
  structure_model = resolve_model(structure, model=model)
  ca_coords = []

  for chain_view in structure_model.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      for atom in residue.atoms():
        if atom.atom_name == "CA":
          ca_coords.append(atom.coord)
          break

  if not ca_coords:
    return np.zeros((0, 0), dtype=np.float32)

  coord = np.asarray(ca_coords, dtype=np.float32)
  return np.linalg.norm(coord[:, np.newaxis, :] - coord[np.newaxis, :, :], axis=-1)


def ca_distance_matrix(structure, model: Optional[int] = None, chain: Optional[str] = None) -> np.ndarray:
  """Alias for :func:`calculate_distance_matrix`."""
  return calculate_distance_matrix(structure, model=model, chain=chain)
