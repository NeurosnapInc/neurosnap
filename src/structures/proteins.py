"""
Provides functions and classes related to processing protein structure data.
"""
### IMPORTS ###
import numpy as np
import Bio.PDB
from biotite.structure.io import pdb
from biotite.structure.residues import get_residues
from biotite.sequence import ProteinSequence


### FUNCTIONS ###
def read_chains(pdb_path):
  """
  -------------------------------------------------------
  Reads the chains in PDB file and returns a set of their names/IDs.
  -------------------------------------------------------
  Parameters:
    pdb_path: Input PDB file path (str)
  Returns:
    chains: Chain names/IDs found within the PDB file (set<str>)
  """
  parser = Bio.PDB.PDBParser()
  structure = parser.get_structure("pdb", pdb_path)
  return set(chain.id for chain in structure[0])


def calc_pdm(pdb_path, chain=None):
  """
  -------------------------------------------------------
  Calculates distance matrix for a given input protein
  -------------------------------------------------------
  Parameters:
    pdb_path: Path to PDB file you want to calculate the distance matrix of (str)
    chain...: The chain to use. By default will just use the longest chain (str)
  Returns:
    dm: Distance matrix of the PDB file (np.array)
  """
  def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

  # load structure and load first model
  # NOTE: If a PDB contains multiple models this solution will just use the first model
  structure = Bio.PDB.PDBParser().get_structure("null", pdb_path)
  model = structure[0]
  
  # get the desired chain
  max_len = float("-inf")
  if chain is None:
    for x in model.get_chains():
      if len(x) > max_len:
        max_len = len(x)
        chain = x.id

  chain = model[chain]

  # calculate distance matrix
  dm = np.zeros((len(chain), len(chain)), float)
  for row, residue_one in enumerate(chain) :
    for col, residue_two in enumerate(chain) :
      dm[row, col] = calc_residue_dist(residue_one, residue_two)
  return dm


def pdb_to_AA(pdb_path):
  """
  -------------------------------------------------------
  Reads a PDB file to and gets its corresponding amino acid sequence.
  Current implementation uses biotite and not biopython. 
  -------------------------------------------------------
  Parameters:
    pdb_path: Path to input PDB file to read (str)
  Returns:
    seq: Corresponding amino acid sequence of PDB file (str)
  """
  with open(pdb_path) as f:
    pdb_file = pdb.PDBFile.read(pdb_path)
    atoms  = pdb_file.get_structure()
    residues = get_residues(atoms)[1]
  return ''.join([ProteinSequence.convert_letter_3to1(r) for r in residues])


def pdb_to_sdf(pdb_path, output_path, max_residues=50):
  """
  -------------------------------------------------------
  Converts a protein/peptide in a PDB file to an SDF.
  PDB file can only include a single entry.
  Will overwrite existing results.
  Validates the SDF file with RDkit on completion
  -------------------------------------------------------
  Parameters:
    pdb_path....: Path to input PDB file to convert (str)
    output_path.: Path to output SDF file, should end with .sdf (str)
    max_residues: Maximum number of residues, default=50 (int)
  """
  # delete output if exists already
  if os.path.exists(output_path):
    os.remove(output_path)
  # stupidly doesn't return an exception if invalid file but will return empty list
  molecules = list(pybel.readfile("pdb", pdb_path))
  assert len(molecules), ValueError(f"Invalid input PDB file.")
  
  found = 0
  for mol in molecules:
    if mol.atoms: # don't include empty molecules
      assert len(mol.residues) <= max_residues, ValueError(f"PDB file is too large and exceeds the maximum number of {max_residues} residues.")
      # print(mol.write("sdf")) #print to stdout
      mol.write("sdf", filename=output_path)
      found += 1
      assert found <= 1, ValueError(f"Invalid input PDB file. Contains more than one chain or molecule.")
  
  if Chem.MolFromMolFile(output_path) is None:
    raise ValueError("Invalid input PDB file. Could not convert properly into an SDF.")