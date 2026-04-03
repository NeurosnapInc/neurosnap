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

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    model: Optional model ID when selecting from an ensemble or stack.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The selected structure object is modified in-place.
  """
  _remove_residues(structure, lambda residue: residue.res_name.strip().upper() in {"WAT", "HOH"}, model=model, chain=chain)


def remove_nucleotides(structure, model: Optional[int] = None, chain: Optional[str] = None):
  """Remove DNA and RNA residues from a structure-like object in-place.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    model: Optional model ID when selecting from an ensemble or stack.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The selected structure object is modified in-place.
  """
  _remove_residues(structure, lambda residue: classify_polymer_residue(residue) in {"dna", "rna"}, model=model, chain=chain)


def remove_non_biopolymers(structure, model: Optional[int] = None, chain: Optional[str] = None):
  """Remove non-protein and non-nucleotide residues from a structure-like object in-place.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    model: Optional model ID when selecting from an ensemble or stack.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The selected structure object is modified in-place.
  """
  _remove_residues(structure, lambda residue: classify_polymer_residue(residue) is None, model=model, chain=chain)


def _remove_residues(structure, predicate: Callable, model: Optional[int], chain: Optional[str]):
  """Remove residues that satisfy a predicate from a structure-like object.

  For stacks, residue removal is driven by the selected model but applied to
  the shared atom table because every model in a stack shares the same
  annotations and bond topology.
  """
  def residue_keep_mask(model_structure: Structure) -> np.ndarray:
    """Build an atom keep-mask by evaluating the predicate residue-by-residue."""
    keep_mask = np.ones(len(model_structure), dtype=bool)
    residue_groups = residue_index_groups(model_structure)

    for chain_view in model_structure.chains():
      if chain is not None and chain_view.chain_id != chain:
        continue
      for residue in chain_view.residues():
        if not predicate(residue):
          continue
        residue_id = (residue.chain_id, residue.res_id, residue.ins_code, residue.res_name, residue.hetero)
        for atom_index in residue_groups.get(residue_id, []):
          keep_mask[atom_index] = False
    return keep_mask

  if isinstance(structure, Structure):
    filter_structure_atoms(structure, residue_keep_mask(structure))
    return

  if isinstance(structure, StructureEnsemble):
    if model is None:
      for model_structure in structure.models():
        filter_structure_atoms(model_structure, residue_keep_mask(model_structure))
      return

    target_model = resolve_model(structure, model=model)
    filter_structure_atoms(target_model, residue_keep_mask(target_model))
    return

  target_model = resolve_model(structure, model=model)
  filter_stack_atoms(structure, residue_keep_mask(target_model))
