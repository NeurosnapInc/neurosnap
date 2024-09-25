"""
Provides functions and classes related to processing protein structure data.
"""
import os
import numpy as np
import Bio.PDB
from Bio.PDB import PDBParser, PPBuilder
from openbabel import pybel
from rdkit import Chem
import neurosnap.structures.lDDT as lDDT


### FUNCTIONS ###
def read_chains(pdb_path):
  """
  -------------------------------------------------------
  Reads the chains in PDB file and returns a list of their names/IDs.
  Only does so for the first model within the pdb file.
  -------------------------------------------------------
  Parameters:
    pdb_path: Input PDB file path (str)
  Returns:
    chains: Chain names/IDs found within the PDB file (list<str>)
  """
  parser = Bio.PDB.PDBParser()
  structure = parser.get_structure("pdb", pdb_path)
  return list(set(chain.id for chain in structure[0] if chain.id.strip()))


def read_pdb(pdb_path):
  """
  -------------------------------------------------------
  Reads a protein and returns the IDs
  -------------------------------------------------------
  Parameters:
    pdb_path: Input PDB file path (str)
  Returns:
    protein: Dictionary where keys are chain IDs and values are lists of residue IDs (dict<str:[str]>)
  """
  parser = Bio.PDB.PDBParser()
  structure = parser.get_structure("pdb", pdb_path)
  # assume everything is in the first model
  protein = {}
  for chain in structure[0]:
    protein[chain.id] = []
    for resi in chain:
      if resi.id[0] == " ": #ensure is not heteroatom
        protein[chain.id].append(resi.id[1])
    protein[chain.id] = sorted(set(protein[chain.id])) # in case the PDB is really weird
  return protein


def calc_pdm(pdb_path, chain=None):
  """
  -------------------------------------------------------
  Calculates distance matrix for a given input protein using
  the C-Alpha distances between residues.
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
  # NOTE: This is redundant since the upper triangle is identical to the lower one.
  dm = np.zeros((len(chain), len(chain)), float)
  for row, residue_one in enumerate(chain):
    for col, residue_two in enumerate(chain):
      dm[row, col] = calc_residue_dist(residue_one, residue_two)
  return dm


def pdb_to_aa(pdb_path):
  """
  -------------------------------------------------------
  Reads a PDB file to and gets its corresponding amino acid sequence.
  Current implementation uses biopython and ignores all non-standard AA molecules.
  All chains on all models are concatenated together. 
  -------------------------------------------------------
  Parameters:
    pdb_path: Path to input PDB file to read (str)
  Returns:
    seq: Corresponding amino acid sequence of PDB file (str)
  """
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure("structure", pdb_path)

  # Create a Polypeptide builder
  ppb = PPBuilder()
  
  # Extract the amino acid sequence
  sequences = []
  for pp in ppb.build_peptides(structure):
    sequences.append(pp.get_sequence())
  
  # Concatenate sequences in case there are multiple chains
  full_sequence = ''.join(map(str, sequences))
  
  return full_sequence


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
  assert len(molecules), ValueError("Invalid input PDB file.")
  
  found = 0
  for mol in molecules:
    if mol.atoms: # don't include empty molecules
      assert len(mol.residues) <= max_residues, ValueError(f"PDB file is too large and exceeds the maximum number of {max_residues} residues.")
      # print(mol.write("sdf")) #print to stdout
      mol.write("sdf", filename=output_path)
      found += 1
      assert found <= 1, ValueError("Invalid input PDB file. Contains more than one chain or molecule.")
  
  if Chem.MolFromMolFile(output_path) is None:
    raise ValueError("Invalid input PDB file. Could not convert properly into an SDF.")


def align_pdbs(ref_pdb, sample_pdb):
  """
  -------------------------------------------------------
  Aligns two pdb structures by their longest chain using the first pdb as the reference.
  Reference pdb is not modified or overwritten while the sample pdb is overwritten.
  -------------------------------------------------------
  Parameters:
    ref_pdb...: Filepath for reference protein to align to (str)
    sample_pdb: Filepath for sample protein to be modified and aligned to the reference (str)
  """
  # Start the parser
  pdb_parser = Bio.PDB.PDBParser(QUIET=True)

  # Get the structures
  ref_structure = pdb_parser.get_structure("reference", ref_pdb)
  sample_structure = pdb_parser.get_structure("sample", sample_pdb)

  # Use the first model in the pdb-files for alignment
  # Change the number 0 if you want to align to another structure
  ref_model = ref_structure[0]
  sample_model = sample_structure[0]

  # Select what residues numbers you wish to align
  # and put them in a list
  start_id = 1
  end_id = max(len(chain) for chain in ref_structure[0])
  atoms_to_be_aligned = range(start_id, end_id + 1)

  # Make a list of the atoms (in the structures) you wish to align.
  # In this case we use CA atoms whose index is in the specified range
  ref_atoms = []
  sample_atoms = []

  # Iterate of all chains in the model in order to find all residues
  for ref_chain in ref_model:
    # Iterate of all residues in each model in order to find proper atoms
    for ref_res in ref_chain:
      # Check if residue number ( .get_id() ) is in the list
      if ref_res.get_id()[1] in atoms_to_be_aligned:
        if Bio.PDB.Polypeptide.is_aa(ref_res, standard=False): # check if residue is standard AA
          # Append CA atom to list
          ref_atoms.append(ref_res['CA'])

  # Do the same for the sample structure
  for sample_chain in sample_model:
    for sample_res in sample_chain:
      if sample_res.get_id()[1] in atoms_to_be_aligned:
        if Bio.PDB.Polypeptide.is_aa(sample_res, standard=False): # check if residue is standard AA
          sample_atoms.append(sample_res['CA'])

  # Now we initiate the superimposer:
  super_imposer = Bio.PDB.Superimposer()
  super_imposer.set_atoms(ref_atoms, sample_atoms)
  super_imposer.apply(sample_model.get_atoms())

  # Print RMSD:
  # print(super_imposer.rms)

  # Save the aligned version of 1UBQ.pdb
  io = Bio.PDB.PDBIO()
  io.set_structure(sample_structure) 
  io.save(sample_pdb)


def calc_lDDT(ref_pdb, sample_pdb):
  """
  -------------------------------------------------------
  Calculates the lDDT (Local Distance Difference Test) between two proteins.
  -------------------------------------------------------
  Parameters:
    ref_pdb...: Filepath for reference protein (str)
    sample_pdb: Filepath for sample protein (str)
  Returns:
    lDDT: The lDDT score of the two proteins which ranges between 0-1 (float)
  """
  ref_L, ref_dmap, ref_rnames = lDDT.pdb2dmap(ref_pdb)
  mod_L, mod_dmap, mod_rnames = lDDT.pdb2dmap(sample_pdb)
  return lDDT.get_LDDT(ref_dmap, mod_dmap)