"""
Provides functions and classes related to processing chemical data.
"""
from rdkit import Chem
from neurosnap.log import logger


### FUNCTIONS ###
def smiles_to_sdf(smiles, output_path):
  """
  -------------------------------------------------------
  Converts a SMILES string to an sdf file.
  Will overwrite existing results.
  NOTE: This function does the bare minimum in terms of
  generating the SDF molecule. The conformers.py module
  should be used in most cases.
  -------------------------------------------------------
  Parameters:
    smiles.....: Smiles string to parse and convert (str)
    output_path: Path to output SDF file, should end with .sdf (str)
  """
  try:
    m = Chem.MolFromSmiles(smiles)
    # m = Chem.AddHs(m)
    with Chem.SDWriter(output_path) as w:
      w.write(m)
  except Exception as e:
    logger.error(f"Exception {e}")
    raise ValueError(f'"{smiles}" is not a valid SMILES string, please follow the input instructions')