"""
Provides functions and classes related to processing chemical data.
"""

from typing import List, Optional

import numpy as np
from rdkit import Chem

from neurosnap.io.pdb import parse_pdb
from neurosnap.log import logger


### FUNCTIONS ###
def smiles_to_sdf(smiles: str, output_path: str) -> None:
  """Converts a SMILES string to an sdf file.
  Will overwrite existing results.

  NOTE: This function does the bare minimum in terms of
  generating the SDF molecule. The :obj:`neurosnap.conformers` module
  should be used in most cases.

  Parameters:
    smiles: Smiles string to parse and convert
    output_path: Path to output SDF file, should end with .sdf

  """
  try:
    m = Chem.MolFromSmiles(smiles)
    # m = Chem.AddHs(m)
    with Chem.SDWriter(output_path) as w:
      w.write(m)
  except Exception as e:
    logger.error(f"Exception {e}")
    raise ValueError(f'"{smiles}" is not a valid SMILES string, please follow the input instructions')


def sdf_to_smiles(fpath: str) -> List[str]:
  """
  Converts molecules in an SDF file to SMILES strings.

  Reads an input SDF file and extracts SMILES strings from its molecules.
  Invalid or unreadable molecules are skipped, with warnings logged.

  Args:
    fpath (str): Path to the input SDF file.

  Returns:
    List[str]: A list of SMILES strings corresponding to valid molecules in the SDF file.

  Raises:
    FileNotFoundError: If the SDF file cannot be found.
    IOError: If the file cannot be read.
  """
  output_smiles = []
  supplier = Chem.SDMolSupplier(fpath, removeHs=False)  # Keep hydrogens if present
  for mol in supplier:
    if mol is None:
      logger.warning("Skipped an invalid molecule.")
      continue
    try:
      smiles = Chem.MolToSmiles(mol)
      output_smiles.append(smiles)
    except Exception as e:
      logger.error(f"Error converting molecule to SMILES: {e}")
  return output_smiles


def validate_smiles(smiles: str) -> bool:
  """
  Validates a SMILES (Simplified Molecular Input Line Entry System) string.

  Args:
    smiles (str): The SMILES string to validate.

  Returns:
    bool: True if the SMILES string is valid, False otherwise.

  Raises:
    Exception: Logs any exception encountered during validation.
  """
  if not smiles:
    logger.error("Invalid input SMILES provided. SMILES string cannot be empty.")
    return False
  try:
    mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles)
    return mol is not None
  except Exception as e:
    logger.error(f"Error validating SMILES: {e}")
    return False


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
  mol = suppl[0]  # Assume single ligand
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
