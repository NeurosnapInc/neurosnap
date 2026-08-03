"""Shared helpers for standalone structure analysis functions."""

from typing import Literal, Optional, Tuple

import numpy as np

from neurosnap.constants.sequence import AA_ALIASES, AA_RECORDS
from neurosnap.constants.structure import BACKBONE_ATOMS_AA, BACKBONE_ATOMS_DNA, BACKBONE_ATOMS_RNA, NA_DNA_CODES, NA_RNA_CODES

from .structure import Residue, Structure, StructureStack

PolymerType = Literal["protein", "dna", "rna"]


def coord_matrix(structure: Structure) -> np.ndarray:
  """Return an ``(n_atoms, 3)`` coordinate matrix for a structure."""
  if len(structure) == 0:
    return np.zeros((0, 3), dtype=np.float32)
  return np.column_stack((structure.atoms["x"], structure.atoms["y"], structure.atoms["z"])).astype(np.float32, copy=False)


def classify_polymer_residue(residue: Residue) -> Optional[PolymerType]:
  """Classify a residue as protein, DNA, RNA, or non-polymer."""
  residue_name = residue.res_name.strip().upper()
  canonical_residue_name = AA_ALIASES.get(residue_name, residue_name)
  if canonical_residue_name in AA_RECORDS:
    return "protein"
  if residue_name in NA_DNA_CODES:
    return "dna"
  if residue_name in NA_RNA_CODES:
    return "rna"

  atom_names = {atom.atom_name.strip().upper() for atom in residue.atoms()}
  if "O2'" in atom_names:
    if len(atom_names.intersection({atom_name.upper() for atom_name in BACKBONE_ATOMS_RNA})) >= 3:
      return "rna"
  if len(atom_names.intersection({atom_name.upper() for atom_name in BACKBONE_ATOMS_DNA})) >= 3:
    return "dna"
  return None


def backbone_atom_order(residue: Residue, include_nucleotides: bool = True) -> Optional[Tuple[str, ...]]:
  """Return backbone atom names in deterministic per-residue order."""
  polymer_type = classify_polymer_residue(residue)
  if polymer_type == "protein":
    return BACKBONE_ATOMS_AA
  if not include_nucleotides:
    return None
  if polymer_type == "dna":
    return BACKBONE_ATOMS_DNA
  if polymer_type == "rna":
    return BACKBONE_ATOMS_RNA
  return None


def filter_structure_atoms(structure: Structure, keep_mask: np.ndarray):
  """Apply an atom keep-mask to a structure and reindex its atom-pair tables."""
  keep_mask = np.asarray(keep_mask, dtype=bool)
  if keep_mask.ndim != 1 or len(keep_mask) != len(structure):
    raise ValueError("Atom keep-mask must be a one-dimensional boolean array with one entry per atom.")

  index_map = np.full(len(structure), -1, dtype=np.int32)
  kept_indices = np.flatnonzero(keep_mask)
  index_map[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)

  bond_keep_mask = np.ones(len(structure.bonds), dtype=bool)
  if len(structure.bonds):
    bond_keep_mask = keep_mask[structure.bonds["atom_i"]] & keep_mask[structure.bonds["atom_j"]]
  new_bonds = structure.bonds[bond_keep_mask].copy()
  if len(new_bonds):
    new_bonds["atom_i"] = index_map[new_bonds["atom_i"]]
    new_bonds["atom_j"] = index_map[new_bonds["atom_j"]]

  interaction_keep_mask = np.ones(len(structure.interactions), dtype=bool)
  if len(structure.interactions):
    interaction_keep_mask = keep_mask[structure.interactions["atom_i"]] & keep_mask[structure.interactions["atom_j"]]
  new_interactions = structure.interactions[interaction_keep_mask].copy()
  if len(new_interactions):
    new_interactions["atom_i"] = index_map[new_interactions["atom_i"]]
    new_interactions["atom_j"] = index_map[new_interactions["atom_j"]]

  structure.atoms = structure.atoms[keep_mask].copy()
  structure.atom_annotations = structure.atom_annotations[keep_mask].copy()
  structure.bonds = new_bonds
  structure.interactions = new_interactions


def filter_stack_atoms(stack: StructureStack, keep_mask: np.ndarray):
  """Apply an atom keep-mask to every model in a stack and reindex atom-pair tables."""
  keep_mask = np.asarray(keep_mask, dtype=bool)
  if keep_mask.ndim != 1 or len(keep_mask) != stack.atom_count:
    raise ValueError("Atom keep-mask must be a one-dimensional boolean array with one entry per shared atom.")

  index_map = np.full(stack.atom_count, -1, dtype=np.int32)
  kept_indices = np.flatnonzero(keep_mask)
  index_map[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)

  bond_keep_mask = np.ones(len(stack.bonds), dtype=bool)
  if len(stack.bonds):
    bond_keep_mask = keep_mask[stack.bonds["atom_i"]] & keep_mask[stack.bonds["atom_j"]]
  new_bonds = stack.bonds[bond_keep_mask].copy()
  if len(new_bonds):
    new_bonds["atom_i"] = index_map[new_bonds["atom_i"]]
    new_bonds["atom_j"] = index_map[new_bonds["atom_j"]]

  # Each model owns its own interaction table, so every one is remapped.
  new_interaction_tables = []
  for table in stack.interactions:
    if len(table):
      table_keep_mask = keep_mask[table["atom_i"]] & keep_mask[table["atom_j"]]
      remapped = table[table_keep_mask].copy()
      if len(remapped):
        remapped["atom_i"] = index_map[remapped["atom_i"]]
        remapped["atom_j"] = index_map[remapped["atom_j"]]
    else:
      remapped = table.copy()
    new_interaction_tables.append(remapped)

  stack.coord = stack.coord[:, keep_mask, :].copy()
  stack.atom_annotations = stack.atom_annotations[keep_mask].copy()
  stack.bonds = new_bonds
  stack.interactions = new_interaction_tables
