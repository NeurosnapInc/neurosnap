"""Convenience structure-filtering functions."""

from typing import Callable, Optional

import numpy as np

from ._common import (
  classify_polymer_residue,
  filter_stack_atoms,
  filter_structure_atoms,
  residue_index_groups,
  resolve_model,
)
from .structure import Structure, StructureEnsemble


def remove_waters(structure, model: Optional[int] = None, chain: Optional[str] = None):
  """Remove water residues from a structure-like object in-place.

  For :class:`StructureStack`, atom removal is applied across all models because
  atom annotations are shared by the stack.
  """
  _remove_residues(structure, _is_water_residue, model=model, chain=chain)


def remove_nucleotides(structure, model: Optional[int] = None, chain: Optional[str] = None):
  """Remove DNA and RNA residues from a structure-like object in-place."""
  _remove_residues(structure, lambda residue: classify_polymer_residue(residue) in {"dna", "rna"}, model=model, chain=chain)


def remove_non_biopolymers(structure, model: Optional[int] = None, chain: Optional[str] = None):
  """Remove non-protein and non-nucleotide residues from a structure-like object in-place."""
  _remove_residues(structure, lambda residue: classify_polymer_residue(residue) is None, model=model, chain=chain)


def _remove_residues(structure, predicate: Callable, model: Optional[int], chain: Optional[str]):
  """Remove residues that satisfy a predicate from a structure-like object."""
  if isinstance(structure, Structure):
    keep_mask = _residue_keep_mask(structure, predicate, chain=chain)
    filter_structure_atoms(structure, keep_mask)
    return

  if isinstance(structure, StructureEnsemble):
    if model is None:
      for model_structure in structure.models():
        keep_mask = _residue_keep_mask(model_structure, predicate, chain=chain)
        filter_structure_atoms(model_structure, keep_mask)
      return

    target_model = resolve_model(structure, model=model)
    keep_mask = _residue_keep_mask(target_model, predicate, chain=chain)
    filter_structure_atoms(target_model, keep_mask)
    return

  target_model = resolve_model(structure, model=model)
  keep_mask = _residue_keep_mask(target_model, predicate, chain=chain)
  filter_stack_atoms(structure, keep_mask)


def _residue_keep_mask(structure: Structure, predicate: Callable, chain: Optional[str] = None) -> np.ndarray:
  """Return an atom keep-mask from a residue predicate."""
  keep_mask = np.ones(len(structure), dtype=bool)
  residue_groups = residue_index_groups(structure)

  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if not predicate(residue):
        continue
      for atom_index in residue_groups.get((residue.chain_id, residue.res_id, residue.ins_code, residue.res_name, residue.hetero), []):
        keep_mask[atom_index] = False

  return keep_mask


def _is_water_residue(residue) -> bool:
  """Return ``True`` for common water residue names."""
  return residue.res_name.strip().upper() in {"WAT", "HOH"}
