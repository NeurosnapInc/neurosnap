"""Geometry helpers for small molecules."""

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from neurosnap.io.pdb import parse_pdb


def get_mol_center(mol, use_mass=False):
  """Computes the geometric center or center of mass of a molecule.

  Args:
      mol (Mol): An RDKit molecule object with 3D coordinates.
      use_mass (bool, optional): If True, computes the center of mass using atomic masses.
                                  If False, computes the simple geometric center. Defaults to False.

  Returns:
      np.ndarray: A NumPy array of shape (3,) representing the [x, y, z] center coordinates.
                  Returns None if the molecule has no conformers.

  Raises:
      ValueError: If no conformer is found in the molecule.
  """
  conf = mol.GetConformer()
  coords = []
  masses = []

  for atom in mol.GetAtoms():
    pos = conf.GetAtomPosition(atom.GetIdx())
    coords.append([pos.x, pos.y, pos.z])
    if use_mass:
      masses.append(atom.GetMass())

  coords = np.array(coords)
  if use_mass:
    masses = np.array(masses)
    return np.average(coords, axis=0, weights=masses)
  else:
    return np.mean(coords, axis=0)


def move_ligand_to_center(ligand_sdf_path, receptor_pdb_path, output_sdf_path, use_mass=False):
  """Moves the center of a ligand in an SDF file to match the center of a receptor in a PDB file.

  This function reads a ligand from an SDF file and a receptor from a PDB file, calculates
  their respective centers (center of mass or geometric center), and translates the ligand
  such that its center aligns with the receptor's center. The modified ligand is then saved
  to a new SDF file.

  Args:
      ligand_sdf_path (str): Path to the input ligand SDF file.
      receptor_pdb_path (str): Path to the input receptor PDB file.
      output_sdf_path (str): Path where the adjusted ligand SDF will be saved.
      use_mass (bool, optional): If True, compute center of mass; otherwise use geometric center. Defaults to False.

  Returns:
      str: Path to the output SDF file with the translated ligand.

  Raises:
      ValueError: If the ligand cannot be parsed from the input SDF file.
  """
  suppl = Chem.SDMolSupplier(ligand_sdf_path, removeHs=False)
  mol = suppl[0]
  if mol is None:
    raise ValueError("Could not parse ligand SDF.")

  receptor = parse_pdb(receptor_pdb_path, return_type="ensemble").models()[0]
  receptor_center = receptor.calculate_center_of_mass()
  ligand_center = get_mol_center(mol, use_mass=use_mass)
  shift = receptor_center - ligand_center

  conf = mol.GetConformer()
  for i in range(conf.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    new_pos = np.array([pos.x, pos.y, pos.z]) + shift
    conf.SetAtomPosition(i, new_pos)

  writer = Chem.SDWriter(output_sdf_path)
  writer.write(mol)
  writer.close()

  return output_sdf_path


def calculate_distance_matrix(mol: Chem.Mol) -> np.ndarray:
  """Calculates the pairwise 3D distance matrix for a molecule.

  Distances are computed from the atomic coordinates stored in the
  molecule's active conformer. The returned matrix is square with one
  row and column per atom.

  Args:
    mol (Chem.Mol): Input RDKit molecule with at least one conformer.

  Returns:
    np.ndarray: A square NumPy array of shape ``(n_atoms, n_atoms)``
      containing pairwise Euclidean distances in Angstroms.

  Raises:
    ValueError: If the input molecule is ``None`` or has no conformers.
  """
  if mol is None:
    raise ValueError("Input molecule cannot be None")
  if not mol.GetNumConformers():
    raise ValueError("Input molecule must have at least one conformer")
  return np.array(Chem.Get3DDistanceMatrix(mol))


def calculate_rmsd(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
  """Calculates the best-fit RMSD between two molecules.

  This function uses RDKit's alignment-based RMSD calculation, meaning
  the molecules are optimally superimposed before the RMSD value is
  reported. As a result, pure rigid-body translations and rotations do
  not by themselves increase the returned RMSD.

  Args:
    mol_a (Chem.Mol): First RDKit molecule with at least one conformer.
    mol_b (Chem.Mol): Second RDKit molecule with at least one conformer.

  Returns:
    float: Best-fit root-mean-square deviation between the two molecules.

  Raises:
    ValueError: If either molecule is ``None`` or lacks conformers.
  """
  if mol_a is None or mol_b is None:
    raise ValueError("Input molecules cannot be None")
  if not mol_a.GetNumConformers() or not mol_b.GetNumConformers():
    raise ValueError("Both molecules must have at least one conformer")
  return float(rdMolAlign.GetBestRMS(mol_a, mol_b))


def translate_molecule(mol: Chem.Mol, vector) -> Chem.Mol:
  """Translates all atomic coordinates in a molecule by a vector.

  The input molecule is not modified in place. Instead, a copy is made
  and every atom position in the first conformer is shifted by the
  provided ``[x, y, z]`` vector.

  Args:
    mol (Chem.Mol): Input RDKit molecule with at least one conformer.
    vector: Translation vector of length 3 containing the x, y, and z shifts.

  Returns:
    Chem.Mol: A translated copy of the input molecule.

  Raises:
    ValueError: If the molecule is ``None``, has no conformers, or the
      translation vector is not length 3.
  """
  if mol is None:
    raise ValueError("Input molecule cannot be None")
  if not mol.GetNumConformers():
    raise ValueError("Input molecule must have at least one conformer")

  shift = np.asarray(vector, dtype=float)
  if shift.shape != (3,):
    raise ValueError("Translation vector must be length 3")

  translated = Chem.Mol(mol)
  conf = translated.GetConformer()
  for i in range(conf.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    conf.SetAtomPosition(i, np.array([pos.x, pos.y, pos.z]) + shift)
  return translated


def align_molecule_to_reference(mol: Chem.Mol, ref_mol: Chem.Mol) -> Chem.Mol:
  """Aligns a molecule to a reference molecule and returns the aligned copy.

  The alignment is performed using RDKit's coordinate-based molecular
  alignment routine. The input molecule is copied before alignment, so
  the original object remains unchanged.

  Args:
    mol (Chem.Mol): Molecule to align, with at least one conformer.
    ref_mol (Chem.Mol): Reference molecule defining the target orientation.

  Returns:
    Chem.Mol: A copy of ``mol`` aligned to ``ref_mol``.

  Raises:
    ValueError: If either molecule is ``None`` or lacks conformers.
  """
  if mol is None or ref_mol is None:
    raise ValueError("Input molecules cannot be None")
  if not mol.GetNumConformers() or not ref_mol.GetNumConformers():
    raise ValueError("Both molecules must have at least one conformer")

  aligned = Chem.Mol(mol)
  rdMolAlign.AlignMol(aligned, ref_mol)
  return aligned
