"""
NOTE: THIS MODULE IS DEPRECATED AND MOST OF IT'S FUNCTIONALITY HAS BEEN MOVED TO neurosnap.protein MODULE.
Provides functions and classes related to processing protein structure data.
TODO: Refactor like the rest for consistency or integrate into protein.py.
"""
import Bio.PDB
from Bio.PDB import PDBParser, PPBuilder


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


def align_pdbs(ref_pdb, sample_pdb):
  """
  -------------------------------------------------------
  Aligns two pdb structures by their longest chain using the first pdb as the reference.
  Reference pdb is not modified or overwritten while the sample pdb is overwritten.
  TODO: REMOVE, replaced by Protein.calculate_rmsd()
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