"""Shared helpers for standalone structure analysis functions."""

from typing import Literal, Optional, Tuple

import numpy as np

from neurosnap.constants.sequence import AA_RECORDS
from neurosnap.constants.structure import BACKBONE_ATOMS_DNA, BACKBONE_ATOMS_RNA, NUC_DNA_CODES, NUC_RNA_CODES

from .structure import Residue, Structure, StructureStack

PolymerType = Literal["protein", "dna", "rna"]


# Backbone extraction needs a deterministic atom order so RMSD/alignment
# calculations do not depend on the atom order used in the parsed file.
_PROTEIN_BACKBONE_ATOMS = ("N", "CA", "C")
_DNA_BACKBONE_ATOMS = ("P", "O1P", "O2P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C1'", "C2'", "C3'", "O3'")
_RNA_BACKBONE_ATOMS = ("P", "O1P", "O2P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C1'", "C2'", "O2'", "C3'", "O3'")


def coord_matrix(structure: Structure) -> np.ndarray:
  """Return an ``(n_atoms, 3)`` coordinate matrix for a structure."""
  if len(structure) == 0:
    return np.zeros((0, 3), dtype=np.float32)
  return np.column_stack((structure.atoms["x"], structure.atoms["y"], structure.atoms["z"])).astype(np.float32, copy=False)


def classify_polymer_residue(residue: Residue) -> Optional[PolymerType]:
  """Classify a residue as protein, DNA, RNA, or non-polymer."""
  residue_name = residue.res_name.strip().upper()
  if residue_name in AA_RECORDS:
    return "protein"
  if residue_name in NUC_DNA_CODES:
    return "dna"
  if residue_name in NUC_RNA_CODES:
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
    return _PROTEIN_BACKBONE_ATOMS
  if not include_nucleotides:
    return None
  if polymer_type == "dna":
    return _DNA_BACKBONE_ATOMS
  if polymer_type == "rna":
    return _RNA_BACKBONE_ATOMS
  return None


def filter_structure_atoms(structure: Structure, keep_mask: np.ndarray):
  """Apply an atom keep-mask to a structure and reindex its bonds."""
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

  structure.atoms = structure.atoms[keep_mask].copy()
  structure.atom_annotations = structure.atom_annotations[keep_mask].copy()
  structure.bonds = new_bonds


def filter_stack_atoms(stack: StructureStack, keep_mask: np.ndarray):
  """Apply an atom keep-mask to every model in a stack and reindex bonds."""
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

  stack.coord = stack.coord[:, keep_mask, :].copy()
  stack.atom_annotations = stack.atom_annotations[keep_mask].copy()
  stack.bonds = new_bonds
