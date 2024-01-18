"""
Provides functions and classes related to processing chemical data.
"""
### IMPORTS ###
import os
from rdkit import Chem
from openbabel import pybel


### FUNCTIONS ###
def smiles_to_sdf(smiles, output_path):
  """
  -------------------------------------------------------
  Converts a SMILES string to an sdf file.
  Will overwrite existing results.
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
    raise ValueError(f'"{smiles}" is not a valid SMILES string, please follow the input instructions. Specific error: {e}')