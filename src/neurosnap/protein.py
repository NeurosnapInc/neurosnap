"""
Provides functions and classes related to processing protein data as well as
a feature rich wrapper around protein structures using BioPython.
# TODO:
- Easy way to get backbone coordinates
"""
import io
import os
import tempfile

import pandas as pd
import requests
from Bio.PDB import PDBIO, PDBParser, PPBuilder

from neurosnap.log import logger

### CONSTANTS ###
## Standard amino acids excluding the unknown character ("X")
## Currently excludes
# O | pyl | pyrrolysine
# U | sec | selenocysteine
# B | asx | asparagine/aspartic acid
# Z | glx | glutamine/glutamic acid
# J | xle | leucine/isoleucine
# X | UNK | unknown codon
# * | TRM | termination codon
STANDARD_AAs = "ACDEFGHIKLMNPQRSTVWY"

## Full amino acids table
AAs_FULL_TABLE = [
  ['A', 'ALA', 'ALANINE'],
  ['R', 'ARG', 'ARGININE'],
  ['N', 'ASN', 'ASPARAGINE'],
  ['D', 'ASP', 'ASPARTIC ACID'],
  ['C', 'CYS', 'CYSTEINE'],
  ['Q', 'GLN', 'GLUTAMINE'],
  ['E', 'GLU', 'GLUTAMIC ACID'],
  ['G', 'GLY', 'GLYCINE'],
  ['H', 'HIS', 'HISTIDINE'],
  ['I', 'ILE', 'ISOLEUCINE'],
  ['L', 'LEU', 'LEUCINE'],
  ['K', 'LYS', 'LYSINE'],
  ['M', 'MET', 'METHIONINE'],
  ['F', 'PHE', 'PHENYLALANINE'],
  ['P', 'PRO', 'PROLINE'],
  ['S', 'SER', 'SERINE'],
  ['T', 'THR', 'THREONINE'],
  ['W', 'TRP', 'TRYPTOPHAN'],
  ['Y', 'TYR', 'TYROSINE'],
  ['V', 'VAL', 'VALINE'],
  ['O', 'PYL', 'PYRROLYSINE'],
  ['U', 'SEC', 'SELENOCYSTEINE'],
  ['B', 'ASX', 'ASPARAGINE/ASPARTIC ACID'],
  ['Z', 'GLX', 'GLUTAMINE/GLUTAMIC ACID'],
  ['J', 'XLE', 'LEUCINE/ISOLEUCINE'],
  ['X', 'UNK', 'UNKNOWN CODON'],
]
AA_CODE_TO_ABR = {}
AA_CODE_TO_NAME = {}
AA_ABR_TO_CODE = {}
AA_ABR_TO_NAME = {}
AA_NAME_TO_CODE = {}
AA_NAME_TO_ABR = {}
for code,abr,name in AAs_FULL_TABLE:
  AA_CODE_TO_ABR[code] = abr
  AA_CODE_TO_NAME[code] = name
  AA_ABR_TO_CODE[abr] = code
  AA_ABR_TO_NAME[abr] = name
  AA_NAME_TO_ABR[name] = abr
  AA_NAME_TO_CODE[name] = code


### CLASSES ###
class Protein():
  def __init__(self, pdb):
    """
    -------------------------------------------------------
    Class that wraps around a protein structure.
    Utilizes the biopython protein structure under the hood.
    -------------------------------------------------------
    Parameters:
      pdb: Can be either a file handle, PDB filepath, PDB ID, or UniProt ID (str|io.IOBase)
    """
    if isinstance(pdb, io.IOBase):
      pass
    elif isinstance(pdb, str):
      if not os.path.exists(pdb):
        if len(pdb) == 4: # check if a valid PDB ID
          r = requests.get(f"https://files.rcsb.org/download/{pdb.upper()}.pdb")
          r.raise_for_status()
          with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as temp_file:
            temp_file.write(r.content)
            pdb = temp_file.name
            logger.info("Found matching structure in RCSB PDB, downloading and using that.")
        else: # check if provided a uniprot ID and fetch from AF2
          r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{pdb.upper()}")
          r = requests.get(r.json()[0]["pdbUrl"])
          r.raise_for_status()
          with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as temp_file:
            temp_file.write(r.content)
            pdb = temp_file.name
            logger.info("Found matching structure in AF-DB, downloading and using that.")
    else:
      raise ValueError("Invalid input type provided. Can be either a file handle, PDB filepath, PDB ID, or UniProt ID")

    # load structure
    parser = PDBParser()
    self.structure = parser.get_structure("structure", pdb)
    assert len(self.structure), ValueError("No models found. Structure appears to be empty.")

    # create a pandas dataframe similar to that of biopandas (https://biopandas.github.io/biopandas/)
    df = {
      "model": [],
      "chain": [],
      "res_id": [],
      "res_name": [],
      "atom": [],
      "atom_name": [],
      "bfactor": [],
      "x": [],
      "y": [],
      "z": [],
      "mass": [],
    }
    for model in self.structure:
      for chain in model:
        for res in chain:
          for atom in res:
            df["model"].append(model.id)
            df["chain"].append(chain.id)
            df["res_id"].append(res.id[1])
            df["res_name"].append(res.resname)
            df["atom"].append(atom.serial_number)
            df["atom_name"].append(atom.name)
            df["bfactor"].append(atom.bfactor)
            df["x"].append(atom.coord[0])
            df["y"].append(atom.coord[1])
            df["z"].append(atom.coord[2])
            df["mass"].append(atom.mass)
    self.df = pd.DataFrame(df)

  def models(self):
    """
    -------------------------------------------------------
    Returns a list of all the model names/IDs.
    -------------------------------------------------------
    Returns:
      models: Chain names/IDs found within the PDB file (list<int>)
    """
    return [model.id for model in self.structure]

  def chains(self, model = 0):
    """
    -------------------------------------------------------
    Returns a list of all the chain names/IDs.
    Assumes first model of 0 if not provided.
    -------------------------------------------------------
    Parameters:
      model: The ID of the model you want to fetch the chains of, defaults to 0 (int)
    Returns:
      chains: Chain names/IDs found within the PDB file (list<str>)
    """
    return [chain.id for chain in self.structure[model] if chain.id.strip()]
  
  def get_aas(self, model, chain):
    """
    -------------------------------------------------------
    Returns the amino acid sequence of a target chain.
    Ligands, small molecules, and nucleotides are ignored.#TODO CHeck nucleotides
    -------------------------------------------------------
    Parameters:
      model: The ID of the model containing the target chain (int)
      chain: The ID of the chain you want to fetch the AA sequence of (str)
    Returns:
      seq: The amino acid sequence of the found chain (str)
    """
    assert model in self.structure, ValueError(f'Protein does not contain model "{model}"')
    assert chain in self.structure[model], ValueError(f'Model {model} does not contain chain "{chain}"')

    ppb = PPBuilder()
    return ppb.build_peptides(self.structure[model][chain])[0].get_sequence()

  def save(self, fpath):
    """
    -------------------------------------------------------
    Save the PDB as a file. Will overwrite existing file.
    -------------------------------------------------------
    Parameters:
      fpath: File path to where you want to save the PDB (str)
    """
    io = PDBIO()
    io.set_structure(self.structure)
    io.save(fpath, preserve_atom_numbering = True)


### FUNCTIONS ###
def getAA(query):
  """
  -------------------------------------------------------
  Efficiently get any amino acid using either their 1 letter code,
  3 letter abbreviation, or full name. See AAs_FULL_TABLE
  for a list of all supported amino acids and codes.
  -------------------------------------------------------
  Parameters:
    query: Amino acid code, abbreviation, or name (str)
  Returns:
    code: Amino acid 1 letter abbreviation / code (str)
    abr.: Amino acid 3 letter abbreviation / code (str)
    name: Amino acid full name (str)
  """
  query = query.upper()
  try:
    if len(query) == 1:
      return query, AA_CODE_TO_ABR[query], AA_CODE_TO_NAME[query]
    elif len(query) == 3:
      return AA_ABR_TO_CODE[query], query, AA_ABR_TO_NAME[query]
    else:
      return AA_NAME_TO_CODE[query], AA_NAME_TO_ABR[query], query
  except KeyError:
    raise ValueError(f"Unknown amino acid for {query}")