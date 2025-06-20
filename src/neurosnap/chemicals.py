"""
Provides functions and classes related to processing chemical data.
"""

import json
import os
from typing import List, Optional, Set

import numpy as np
import requests
from rdkit import Chem

from neurosnap.log import logger
from neurosnap.protein import Protein


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
  try:
    mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles)
    return mol is not None
  except Exception as e:
    logger.error(f"Error validating SMILES: {e}")
    return False


def get_ccds(fpath: str = "~/.cache/ccd_codes.json") -> Set[str]:
  """
  Retrieves a set of all CCD (Chemical Component Dictionary) codes from the PDB.

  This function checks for a locally cached JSON file with the CCD codes.
  - If the file exists, it reads and returns the set of codes from the cache.
  - If the file does not exist, it downloads the full Chemical Component Dictionary
    (in mmCIF format) from the Protein Data Bank (PDB), extracts the CCD codes,
    and caches them in a JSON file for future use.

  Parameters:
      fpath: The path to store / cache all the stored ccd_codes as a JSON file.
              Default is "~/.cache/ccd_codes.json"

  Returns:
      set: A set of all CCD codes (three-letter codes representing small molecules,
            ligands, and post-translational modifications).

  Raises:
      HTTPError: If the request to the PDB server fails.
      JSONDecodeError: If the cached JSON file is corrupted.

  File Cache:
      - Cached file path: ".cache/ccd_codes.json"
      - The cache is automatically updated if it does not exist.

  External Resources:
      - CCD information: https://www.wwpdb.org/data/ccd
      - CCD data download: https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
  """
  fpath = os.path.expanduser(fpath)
  if os.path.exists(fpath):
    logger.debug("Found CCD codes cached locally.")
    with open(fpath) as f:
      return set(json.load(f))
  else:
    logger.info("No CCD codes cached locally, downloading now.")
    r = requests.get("https://files.wwpdb.org/pub/pdb/data/monomers/components.cif")
    r.raise_for_status()
    logger.debug("Finished downloading CCD codes.")

    codes = []
    for line in r.text.split("\n"):
      if "_chem_comp.three_letter_code" in line:
        code = line.split()[-1]
        if code != "?":
          codes.append(code)

    with open(fpath, "w") as f:
      json.dump(codes, f)

    return set(codes)


def fetch_ccd(ccd_code: str, fpath: str):
  """
  Fetches the ideal SDF (Structure Data File) for a given CCD (Chemical Component Dictionary) code
  and saves it to the specified file path.

  This function retrieves the idealized structure of a chemical component from the RCSB Protein
  Data Bank (PDB) by downloading the corresponding SDF file. The downloaded file is then saved
  to the specified location.

  Parameters:
      ccd_code (str): The three-letter CCD code representing the chemical component (e.g., "ATP").
      fpath (str): The file path where the downloaded SDF file will be saved.

  Raises:
      HTTPError: If the request to fetch the SDF file fails (e.g., 404 or connection error).
      IOError: If there is an issue saving the SDF file to the specified file path.

  Example:
      >>> fetch_ccd("ATP", "ATP_ideal.sdf")
      Fetches the ideal SDF file for the ATP molecule and saves it as "ATP_ideal.sdf".

  External Resources:
      - CCD Information: https://www.wwpdb.org/data/ccd
      - SDF File Download: https://files.rcsb.org/ligands/download/{CCD_CODE}_ideal.sdf
  """
  logger.info(f"Fetching CCD with code {ccd_code} from rcsb.org...")
  r = requests.get(f"https://files.rcsb.org/ligands/download/{ccd_code}_ideal.sdf")
  r.raise_for_status()
  with open(fpath, "wb") as f:
    f.write(r.content)


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

  receptor_center = Protein(receptor_pdb_path).calculate_center_of_mass()
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
