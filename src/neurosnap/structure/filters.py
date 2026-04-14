"""Convenience structure-filtering functions."""

from typing import Callable, Optional

import numpy as np

from neurosnap.constants.structure import FIVE_PRIME_TERMINAL_ATOMS, THREE_PRIME_TERMINAL_ATOMS
from neurosnap.log import logger

from ._common import classify_polymer_residue, filter_structure_atoms
from .structure import Structure

_PHOSPHATE_RENAME_MAP = {"O1P": "OP1", "O2P": "OP2"}


def remove_chains(structure: Structure, predicate: Callable):
  """Remove chains from a structure in-place when they match a predicate.

  Parameters:
    structure: Input :class:`Structure`.
    predicate: Callable that accepts a chain view and returns ``True`` when
      that chain should be removed.

  Returns:
    ``None``. The input structure is modified in-place.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"remove_chains() expects a Structure, found {type(structure).__name__}.")

  keep_mask = np.ones(len(structure), dtype=bool)
  for chain_view in structure.chains():
    if not predicate(chain_view):
      continue
    for residue in chain_view.residues():
      for atom_index in residue.atom_indices():
        keep_mask[atom_index] = False

  filter_structure_atoms(structure, keep_mask)


def remove_residues(structure: Structure, predicate: Callable, chain: Optional[str]):
  """Remove residues from a structure in-place when they match a predicate.

  Parameters:
    structure: Input :class:`Structure`.
    predicate: Callable that accepts a residue view and returns ``True`` when
      that residue should be removed.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The input structure is modified in-place.
  """
  keep_mask = np.ones(len(structure), dtype=bool)
  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if not predicate(residue):
        continue
      for atom_index in residue.atom_indices():
        keep_mask[atom_index] = False

  filter_structure_atoms(structure, keep_mask)


def remove_atoms(structure: Structure, predicate: Callable, chain: Optional[str] = None):
  """Remove atoms from a structure in-place when they match a predicate.

  Parameters:
    structure: Input :class:`Structure`.
    predicate: Callable that accepts an atom view and returns ``True`` when
      that atom should be removed.
    chain: Optional chain ID to restrict atom removal to.

  Returns:
    ``None``. The input structure is modified in-place.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"remove_atoms() expects a Structure, found {type(structure).__name__}.")
  if chain is not None and chain not in structure.chain_ids():
    raise ValueError(f'Chain "{chain}" was not found in the structure.')

  keep_mask = np.ones(len(structure), dtype=bool)
  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      for atom, atom_index in zip(residue.atoms(), residue.atom_indices()):
        if predicate(atom):
          keep_mask[atom_index] = False

  filter_structure_atoms(structure, keep_mask)


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
  remove_residues(structure, lambda residue: residue.res_name.strip().upper() in {"WAT", "HOH"}, chain=chain)


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
  remove_residues(structure, lambda residue: classify_polymer_residue(residue) in {"dna", "rna"}, chain=chain)


def remove_non_biopolymers(structure: Structure, chain: Optional[str] = None):
  """Remove non-protein and non-nucleotide residues from a structure in-place.

  Parameters:
    structure: Input :class:`Structure`.
    chain: Optional chain ID to restrict residue removal to.

  Returns:
    ``None``. The input structure is modified in-place.

  Notes:
    Hetero residues are always removed by this filter, even if their residue
    names overlap with amino-acid or nucleotide dictionaries such as ``UNK``.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"remove_non_biopolymers() expects a Structure, found {type(structure).__name__}.")
  remove_residues(structure, lambda residue: residue.hetero or classify_polymer_residue(residue) is None, chain=chain)


def fix_nucleic_termini(structure: Structure, *, strip_3prime: bool = False, chain: Optional[str] = None):
  """Normalize nucleotide phosphate names and strip terminal phosphate atoms.

  Parameters:
    structure: Input :class:`Structure`.
    strip_3prime: If ``True``, also remove ``O3P`` and ``OP3`` from 3' termini.
    chain: Optional chain ID to restrict processing to.

  Returns:
    ``None``. The input structure is modified in-place.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"fix_nucleic_termini() expects a Structure, found {type(structure).__name__}.")
  if chain is not None and chain not in structure.chain_ids():
    raise ValueError(f'Chain "{chain}" was not found in the structure.')

  nucleic_residues_found = False
  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if classify_polymer_residue(residue) in {"dna", "rna"}:
        nucleic_residues_found = True
        break
    if nucleic_residues_found:
      break

  if not nucleic_residues_found:
    logger.warning("No nucleotide residues were found while running fix_nucleic_termini(); leaving the structure unchanged.")
    return

  keep_mask = np.ones(len(structure), dtype=bool)

  for atom_index in range(len(structure)):
    atom_name = structure.atom_annotations["atom_name"][atom_index].strip().upper()
    renamed_atom_name = _PHOSPHATE_RENAME_MAP.get(atom_name)
    if renamed_atom_name is not None:
      structure.atom_annotations["atom_name"][atom_index] = renamed_atom_name

  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue

    segment_start = None
    segment_end = None
    previous_nucleic_residue_id = None

    for residue in chain_view.residues():
      is_nucleic = classify_polymer_residue(residue) in {"dna", "rna"}
      if not is_nucleic:
        if segment_start is not None:
          for atom_index in segment_start.atom_indices():
            atom_name = structure.atom_annotations["atom_name"][atom_index].strip().upper()
            if atom_name in FIVE_PRIME_TERMINAL_ATOMS:
              keep_mask[atom_index] = False
          if strip_3prime and segment_end is not None:
            for atom_index in segment_end.atom_indices():
              atom_name = structure.atom_annotations["atom_name"][atom_index].strip().upper()
              if atom_name in THREE_PRIME_TERMINAL_ATOMS:
                keep_mask[atom_index] = False

        segment_start = None
        segment_end = None
        previous_nucleic_residue_id = None
        continue

      residue_id = residue.res_id
      if segment_start is None or previous_nucleic_residue_id is None or residue_id - previous_nucleic_residue_id > 1:
        if segment_start is not None:
          for atom_index in segment_start.atom_indices():
            atom_name = structure.atom_annotations["atom_name"][atom_index].strip().upper()
            if atom_name in FIVE_PRIME_TERMINAL_ATOMS:
              keep_mask[atom_index] = False
          if strip_3prime and segment_end is not None:
            for atom_index in segment_end.atom_indices():
              atom_name = structure.atom_annotations["atom_name"][atom_index].strip().upper()
              if atom_name in THREE_PRIME_TERMINAL_ATOMS:
                keep_mask[atom_index] = False
        segment_start = residue

      segment_end = residue
      previous_nucleic_residue_id = residue_id

    if segment_start is not None:
      for atom_index in segment_start.atom_indices():
        atom_name = structure.atom_annotations["atom_name"][atom_index].strip().upper()
        if atom_name in FIVE_PRIME_TERMINAL_ATOMS:
          keep_mask[atom_index] = False
      if strip_3prime and segment_end is not None:
        for atom_index in segment_end.atom_indices():
          atom_name = structure.atom_annotations["atom_name"][atom_index].strip().upper()
          if atom_name in THREE_PRIME_TERMINAL_ATOMS:
            keep_mask[atom_index] = False

  if not np.all(keep_mask):
    filter_structure_atoms(structure, keep_mask)
