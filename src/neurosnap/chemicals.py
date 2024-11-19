"""
Provides functions and classes related to processing chemical data.
"""

from typing import List, Optional

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
