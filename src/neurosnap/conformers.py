"""
Provides functions and classes related to processing and generating conformers.
"""
import os

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdFMCS, rdMolAlign
from rdkit.ML.Cluster import Butina

from neurosnap.log import logger


def find_LCS(mol):
  """
  -------------------------------------------------------
  Find the largest common substructure (LCS) between a
  set of conformers and aligns all conformers to the LCS.
  Raises an exception if no LCS detected.
  -------------------------------------------------------
  Parameters:
    mol...: Input RDkit molecule object, must already have conformers present (rdkit.Chem.rdchem.Mol)
  Returns:
    mol_aligned: Resultant molecule object with all conformers aligned to the LCS (rdkit.Chem.rdchem.Mol)
  """
  logger.info("Finding largest common substructure (LCS) for clustering")
  # Convert the molecule to a list of molecules (each conformer treated as a separate molecule)
  conformer_mols = []
  for conf in mol.GetConformers():
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(conf, assignId=True)
    conformer_mols.append(new_mol)

  # Perform MCS (Maximum Common Substructure) search
  mcs_result = rdFMCS.FindMCS(conformer_mols, completeRingsOnly=True, ringMatchesRingOnly=True)
  core = Chem.MolFromSmarts(mcs_result.smartsString)

  # Check if a core substructure was found
  assert core is not None and core.GetNumAtoms(), ValueError("No core substructure detected. Aligning using first conformer.")
  
  logger.debug("Core substructure detected. Aligning using core.")
  AllChem.AlignMolConformers(mol, mol.GetSubstructMatch(core))
  return mol


def minimize(mol, method="MMFF94", e_delta=5):
  """
  -------------------------------------------------------
  Minimize conformer energy (kcal/mol) using RDkit
  and filter out conformers below a certain threshold.
  -------------------------------------------------------
  Parameters:
    mol....: RDkit mol object containing the conformers you want to minimize. (rdkit.Chem.rdchem.Mol)
    method.: Can be either UFF, MMFF94, or MMFF94s (str)
    e_delta: Filters out conformers that are above a certain energy threshold in kcal/mol. Formula used is >= min_conformer_energy + e_delta (float)
  Returns:
    mol_filtered: The pairwise sequence identity. Will return None (float)
    energies....: Dictionary where keys are conformer IDs and values are calculated energies in kcal/mol (dict<int:float>)
  """
  logger.info(f"Minimizing energy of all conformers using {method}")
  for i in range(mol.GetNumConformers()):
    if method == "UFF":
      AllChem.UFFOptimizeMolecule(mol, confId=i)
    elif method == "MMFF94":
      AllChem.MMFFOptimizeMolecule(mol, confId=i, mmffVariant="MMFF94")
    elif method == "MMFF94s":
      AllChem.MMFFOptimizeMolecule(mol, confId=i, mmffVariant="MMFF94s")
    else:
      raise ValueError(f'Invalid minimization method passed {method}. Must be either "UFF", "MMFF94", or "MMFF94s".')

  ## Energy Filters
  energies = {} # keys are conformer IDs, values are calculated energies.
  energies_filtered = {}
  for i in range(mol.GetNumConformers()):
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=i)
    energies[i] = ff.CalcEnergy()

  # Filter out conformers above a certain energy threshold (e.g., 5 kcal/mol above the lowest energy conformer)
  energy_threshold = min(energies.values()) + e_delta # kcal/mol
  num_filtered = 0

  # Create a new molecule object with only the filtered conformers
  mol_filtered = Chem.Mol(mol) # Copy the molecule structure
  mol_filtered.RemoveAllConformers() # Remove all conformers from the copy
  for conf_id, energy in energies.items():
    if energy <= energy_threshold:
      num_filtered += 1
      conformer = mol.GetConformer(conf_id)
      new_conf_id = mol_filtered.AddConformer(conformer, assignId=True)  # Add the filtered conformer to the new molecule
      energies_filtered[new_conf_id] = energy
      # mol_filtered.GetConformer(new_conf_id).SetId(conf_id) # Optionally, assign the same ID

  logger.info(f"Filtered {num_filtered:,} conformers within 0 to {energy_threshold:.2f} kcal/mol.")
  logger.info(f"min: {min(energies.values()):.2f}, max: {max(energies.values()):.2f}, avg: {sum(energies.values())/len(energies.values()):.2f}")
  return mol_filtered, energies_filtered


def generate(input_mol, output_name="unique_conformers", write_multi=False, num_confs=1000, min_method="MMFF94", max_atoms=500):
  """
  -------------------------------------------------------
  Generate conformers for an input molecule.
  Performs the following actions in order:
  1. Generate conformers using ETKDG method
  2. Minimize energy of all conformers and remove those below a dynamic threshold
  3. Align & create RMSD matrix of all conformers
  4. Clusters using Butina method to remove structurally redundant conformers
  5. Return most energetically favorable conformers in each cluster
  -------------------------------------------------------
  Parameters:
    input_mol..: Input molecule can be a path to a molecule file, a SMILES string, or an instance of rdkit.Chem.rdchem.Mol (any)
    output_name: Output to write SDF files of passing conformers (str)
    write_multi: If True will write all unique conformers to a single SDF file, if False will write all unique conformers in separate SDF files in output_name (bool)
    num_confs..: Number of conformers to generate (int)
    min_method.: Method for minimization, can be either UFF, MMFF94, MMFF94s, or None for no minimization (str)
    max_atoms..: Maximum number of atoms allowed for the input molecule (int)
  Returns:
    df_out....: Output pandas dataframe with all conformer statistics (pandas.core.frame.DataFrame)
  """
  ### parse input and construct corresponding RDkit mol object
  my_mol = None
  if isinstance(input_mol, rdkit.Chem.rdchem.Mol):
    my_mol = input_mol
  if isinstance(input_mol, str):
    if input_mol.endswith(".pdb"):
      my_mol = Chem.MolFromPDBFile(input_mol)
    elif input_mol.endswith(".sdf"):
      for mol in Chem.SDMolSupplier(input_mol): # get first valid molecule if exists
        if mol is not None:
          my_mol = mol
          break
      assert my_mol is not None, ValueError("Unable to find a valid molecule in the provided SDF file.")
    else: # assume smiles
      my_mol = Chem.MolFromSmiles(input_mol)
      assert my_mol is not None, ValueError(f"Invalid smiles string provided {input_mol}")
  else:
    raise ValueError("Invalid input type for input_mol. Input molecule can be a path to a molecule file, a SMILES string, or an instance of rdkit.Chem.rdchem.Mol")

  # Add hydrogens to molecule to generate a more accurate 3D structure
  my_mol = Chem.AddHs(my_mol)

  ### Generate 3D conformers
  params = AllChem.ETKDGv3()
  params.numThreads = 0
  rdDistGeom.EmbedMultipleConfs(my_mol, numConfs=num_confs, params=params) # NOTE: maxAttempts=100 was removed as does not work with ETKDGv3
  logger.info(f"Generated {my_mol.GetNumConformers():,} 3D conformers.")

  ### Minimize energy of each conformer
  if min_method:
    my_mol, energies = minimize(my_mol, method=min_method)

  ### Clustering
  ## Calculate the RMSD matrix between all pairs of conformers
  dists = []
  num_conformers = my_mol.GetNumConformers()
  my_mol_nh = Chem.RemoveHs(my_mol) # remove hydrogens as it interferes with following steps
  for i in range(num_conformers):
    for j in range(i):
      dists.append(rdMolAlign.GetBestRMS(my_mol_nh, my_mol_nh, i, j))

  ## Perform clustering using Butina approach
  # Calculate the dynamic threshold based on molecule size (distance threshold in Å for Butina)
  num_heavy_atoms = my_mol.GetNumHeavyAtoms()
  c = 0.2  # Constant, can be adjusted based on needs
  threshold = c * np.sqrt(num_heavy_atoms)
  logger.debug(f"Dynamic RMSD threshold: {threshold:.2f} Å")

  # perform clustering
  clusters = Butina.ClusterData(dists, num_conformers, threshold, isDistData=True, reordering=True)

  # get most favorable representatives for each cluster using calculated energy
  output = {
    "conformer_id": [],
    "cluster_id": [],
    "energy": [],
    "cluster_size": [],
  }
  for i, cluster in enumerate(clusters, start=1):
    best_energy = float("inf")
    best_cid = None
    for cid in cluster:
      if energies[cid] < best_energy:
        best_cid = cid
        best_energy = energies[cid]
    logger.debug(f"Cluster ID: {i}, Best Conformer: {best_cid} ({best_energy:.2f}), Conformers {cluster}")
    output["conformer_id"].append(best_cid)
    output["cluster_id"].append(i)
    output["energy"].append(best_energy)
    output["cluster_size"].append(len(cluster))

  df = pd.DataFrame(output)
  logger.info(f"Selected {len(df)} unique conformers.")
  print(df)

  ## Write output
  if write_multi: # write to single SDF file
    with Chem.SDWriter(f"{output_name}.sdf") as w:
      for conf_id in output["conformer_id"]:
        w.write(my_mol, confId=conf_id)
  else: # write to single separate SDF files
    os.makedirs(output_name, exist_ok=True)
    for conf_id in output["conformer_id"]:
      with Chem.SDWriter(os.path.join(output_name, f"conformer_{conf_id}.sdf")) as w:
        w.write(my_mol, confId=conf_id)

  return df