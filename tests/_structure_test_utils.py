"""Shared helpers for structure-layer tests."""

from pathlib import Path

import numpy as np

from neurosnap.io.pdb import parse_pdb
from neurosnap.structure import Structure

FILES = Path("tests/files")
PDB_MONO = FILES / "1BTL.pdb"
PDB_DIMER = FILES / "dimer_af2.pdb"
PDB_LIG = FILES / "1MAL.pdb"
AF2_RANK1 = FILES / "4AOW_af2_rank_1.pdb"
AF2_RANK2 = FILES / "4AOW_af2_rank_2.pdb"
PDB_WITH_H = FILES / "1nkp_mycmax_with_hydrogens.pdb"
PDB_NO_H = FILES / "1nkp_mycmax.pdb"
RNA_CIF_1 = FILES / "rna_monomer_1.cif"
RNA_CIF_2 = FILES / "rna_monomer_2.cif"

ROT_Z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
TRANSLATION_VECTOR = np.array([10.0, -5.0, 3.0], dtype=np.float32)

PROTEIN_BACKBONE_ATOMS = (
  ("N", "ALA", "A", 1, 0.000, 0.000, 0.000, "N"),
  ("CA", "ALA", "A", 1, 1.458, 0.000, 0.000, "C"),
  ("C", "ALA", "A", 1, 1.958, 1.410, 0.000, "C"),
  ("N", "GLY", "A", 2, 3.300, 1.410, 0.000, "N"),
  ("CA", "GLY", "A", 2, 3.800, 2.820, 0.000, "C"),
  ("C", "GLY", "A", 2, 5.200, 2.820, 0.000, "C"),
)

DNA_BACKBONE_ATOMS = (
  ("P", "DA", "A", 1, 0.000, 0.000, 0.000, "P"),
  ("O1P", "DA", "A", 1, -0.800, 1.000, 0.000, "O"),
  ("O2P", "DA", "A", 1, -0.800, -1.000, 0.000, "O"),
  ("O5'", "DA", "A", 1, 1.200, 0.200, 0.000, "O"),
  ("C5'", "DA", "A", 1, 2.200, 0.700, 0.200, "C"),
  ("C4'", "DA", "A", 1, 2.800, 1.800, 0.300, "C"),
  ("O4'", "DA", "A", 1, 2.300, 2.900, 0.400, "O"),
  ("C3'", "DA", "A", 1, 3.600, 1.700, 1.500, "C"),
  ("O3'", "DA", "A", 1, 4.600, 1.700, 1.800, "O"),
  ("C1'", "DA", "A", 1, 1.800, 3.000, -0.500, "C"),
  ("C2'", "DA", "A", 1, 2.300, 2.100, -1.500, "C"),
)

MIXED_BACKBONE_ATOMS = PROTEIN_BACKBONE_ATOMS + tuple(
  (atom_name, resname, "B", resid, x, y, z, element) for atom_name, resname, _chain_id, resid, x, y, z, element in DNA_BACKBONE_ATOMS
)
def make_structure(atom_defs):
  """Build a synthetic single-model Structure directly from atom tuples."""
  structure = Structure(remove_annotations=False)
  structure.metadata["model_id"] = 1
  structure.atoms = np.array([(x, y, z) for _atom_name, _resname, _chain_id, _resid, x, y, z, _element in atom_defs], dtype=structure._dtype_atoms)
  structure.atom_annotations = np.zeros(len(atom_defs), dtype=structure._dtype_atom_annotations)

  for atom_index, (atom_name, resname, chain_id, resid, _x, _y, _z, element) in enumerate(atom_defs):
    structure.atom_annotations["chain_id"][atom_index] = chain_id
    structure.atom_annotations["res_id"][atom_index] = resid
    structure.atom_annotations["ins_code"][atom_index] = ""
    structure.atom_annotations["res_name"][atom_index] = resname
    structure.atom_annotations["hetero"][atom_index] = False
    structure.atom_annotations["atom_name"][atom_index] = atom_name
    structure.atom_annotations["element"][atom_index] = element
    structure.atom_annotations["atom_id"][atom_index] = atom_index + 1
    structure.atom_annotations["b_factor"][atom_index] = 20.0
    structure.atom_annotations["occupancy"][atom_index] = 1.0
    structure.atom_annotations["charge"][atom_index] = 0
    structure.atom_annotations["sym_id"][atom_index] = ""

  structure.bonds = np.zeros(0, dtype=structure._dtype_bond)
  structure._remove_empty_annotations()
  return structure


def parse_single_model(path):
  """Parse a single-model structure file and return the first model."""
  ensemble = parse_pdb(path, return_type="ensemble")
  return ensemble[ensemble.model_ids[0]]


def parse_ensemble(path):
  """Parse a structure file into a StructureEnsemble."""
  return parse_pdb(path, return_type="ensemble")


def transform_atoms(atom_defs, rotation, translation):
  """Apply a rigid transform to atom tuples."""
  transformed = []
  for atom_name, resname, chain_id, resid, x, y, z, element in atom_defs:
    coord = np.array([x, y, z], dtype=np.float32)
    new_coord = rotation @ coord + translation
    transformed.append((atom_name, resname, chain_id, resid, float(new_coord[0]), float(new_coord[1]), float(new_coord[2]), element))
  return transformed


def replace_chain(atom_defs, chain_id):
  """Replace the chain ID for every atom tuple."""
  return [(atom_name, resname, chain_id, resid, x, y, z, element) for atom_name, resname, _old_chain, resid, x, y, z, element in atom_defs]


def coords_from_atom_defs(atom_defs):
  """Return a coordinate matrix from atom tuples."""
  return np.asarray([[x, y, z] for _atom_name, _resname, _chain_id, _resid, x, y, z, _element in atom_defs], dtype=np.float32)
