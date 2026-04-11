"""Geometry helpers for small molecules."""

import numpy as np

from rdkit import Chem

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
