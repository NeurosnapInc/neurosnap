"""
Provides functions and classes related to processing chemical data.
"""

import json
import os
from typing import List, Optional, Set

import requests
from rdkit import Chem

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
