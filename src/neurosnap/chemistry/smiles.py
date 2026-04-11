"""Utilities for SMILES and SDF conversion."""

from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

from neurosnap.log import logger


def smiles_to_sdf(smiles: str, output_path: str) -> None:
  """Converts a SMILES string to an sdf file.
  Will overwrite existing results.

  NOTE: This function does the bare minimum in terms of
  generating the SDF molecule. The :obj:`neurosnap.chemistry.conformers` module
  should be used in most cases.

  Parameters:
    smiles: Smiles string to parse and convert
    output_path: Path to output SDF file, should end with .sdf

  """
  try:
    m = Chem.MolFromSmiles(smiles)
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
  supplier = Chem.SDMolSupplier(fpath, removeHs=False)
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


def canonicalize_smiles(smiles: str) -> str:
  """Converts a SMILES string into its canonical RDKit representation.

  This is useful for normalizing equivalent SMILES strings into a
  stable text form for storage, comparison, or deduplication.

  Args:
    smiles (str): Input SMILES string to canonicalize.

  Returns:
    str: Canonical SMILES string produced by RDKit.

  Raises:
    ValueError: If the input string cannot be parsed as a valid SMILES.
  """
  mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles)
  if mol is None:
    raise ValueError(f'"{smiles}" is not a valid SMILES string')
  return Chem.MolToSmiles(mol, canonical=True)


def standardize_molecule(mol: Chem.Mol) -> Chem.Mol:
  """Standardizes a molecule using RDKit's cleanup workflow.

  The standardization process applies RDKit's built-in molecular cleanup
  rules, which can normalize representations such as functional groups,
  charges, and related valence patterns into a more consistent form.

  Args:
    mol (Chem.Mol): Input RDKit molecule to standardize.

  Returns:
    Chem.Mol: A standardized copy of the input molecule.

  Raises:
    ValueError: If the input molecule is ``None``.
  """
  if mol is None:
    raise ValueError("Input molecule cannot be None")
  return rdMolStandardize.Cleanup(Chem.Mol(mol))


def neutralize_molecule(mol: Chem.Mol) -> Chem.Mol:
  """Neutralizes formal charges in a molecule where chemically supported.

  This function uses RDKit's uncharging logic to neutralize ionized
  atoms when a valid neutral form can be produced. Charges that cannot
  be safely neutralized are preserved.

  Args:
    mol (Chem.Mol): Input RDKit molecule to neutralize.

  Returns:
    Chem.Mol: A copy of the molecule with reducible charges neutralized.

  Raises:
    ValueError: If the input molecule is ``None``.
  """
  if mol is None:
    raise ValueError("Input molecule cannot be None")
  uncharger = rdMolStandardize.Uncharger()
  return uncharger.uncharge(Chem.Mol(mol))


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
  """Selects the largest fragment from a multi-component molecule.

  This is typically useful for salts, mixtures, or counterion-containing
  inputs where only the primary chemical component should be retained.

  Args:
    mol (Chem.Mol): Input RDKit molecule, which may contain multiple fragments.

  Returns:
    Chem.Mol: A copy containing only the largest fragment.

  Raises:
    ValueError: If the input molecule is ``None``.
  """
  if mol is None:
    raise ValueError("Input molecule cannot be None")
  chooser = rdMolStandardize.LargestFragmentChooser()
  return chooser.choose(Chem.Mol(mol))


def remove_salts(mol: Chem.Mol) -> Chem.Mol:
  """Removes common salt fragments while retaining the main molecular component.

  The function first strips recognized salts and small counterions using
  RDKit's salt remover, then selects the largest remaining fragment to
  produce a single primary molecule.

  Args:
    mol (Chem.Mol): Input RDKit molecule that may contain salts or counterions.

  Returns:
    Chem.Mol: A desalted copy of the molecule.

  Raises:
    ValueError: If the input molecule is ``None``.
  """
  if mol is None:
    raise ValueError("Input molecule cannot be None")
  remover = SaltRemover()
  stripped = remover.StripMol(Chem.Mol(mol), dontRemoveEverything=True)
  return largest_fragment(stripped)
