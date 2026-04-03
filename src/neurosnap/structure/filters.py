"""Convenience structure-filtering functions."""

from typing import Callable, Optional

import numpy as np

from ._common import classify_polymer_residue, filter_structure_atoms, residue_index_groups
from .structure import Structure


def remove_waters(structure: Structure, chain: Optional[str] = None):
  """Remove water residues from a structure in-place.

  Parameters:
    structure: Input :class:`Structure`.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The input structure is modified in-place.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"remove_waters() expects a Structure, found {type(structure).__name__}.")
  _remove_residues(structure, lambda residue: residue.res_name.strip().upper() in {"WAT", "HOH"}, chain=chain)


def remove_nucleotides(structure: Structure, chain: Optional[str] = None):
  """Remove DNA and RNA residues from a structure in-place.

  Parameters:
    structure: Input :class:`Structure`.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The input structure is modified in-place.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"remove_nucleotides() expects a Structure, found {type(structure).__name__}.")
  _remove_residues(structure, lambda residue: classify_polymer_residue(residue) in {"dna", "rna"}, chain=chain)


def remove_non_biopolymers(structure: Structure, chain: Optional[str] = None):
  """Remove non-protein and non-nucleotide residues from a structure in-place.

  Parameters:
    structure: Input :class:`Structure`.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The input structure is modified in-place.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"remove_non_biopolymers() expects a Structure, found {type(structure).__name__}.")
  _remove_residues(structure, lambda residue: classify_polymer_residue(residue) is None, chain=chain)


def _remove_residues(structure: Structure, predicate: Callable, chain: Optional[str]):
  """Remove residues that satisfy a predicate from a single-model structure."""
  keep_mask = np.ones(len(structure), dtype=bool)
  residue_groups = residue_index_groups(structure)
  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if not predicate(residue):
        continue
      residue_id = (residue.chain_id, residue.res_id, residue.ins_code, residue.res_name, residue.hetero)
      for atom_index in residue_groups.get(residue_id, []):
        keep_mask[atom_index] = False

  filter_structure_atoms(structure, keep_mask)
