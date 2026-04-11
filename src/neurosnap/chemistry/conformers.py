"""Provides functions and classes related to processing and generating conformers."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdFMCS
from rdkit.ML.Cluster import Butina

from neurosnap.log import logger


def find_LCS(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
  """Find the largest common substructure (LCS) between a
  set of conformers and aligns all conformers to the LCS.

  Parameters:
    mol: Input RDkit molecule object, must already have conformers present

  Returns:
    Resultant molecule object with all conformers aligned to the LCS

  Raises:
    Exception: if no LCS is detected

  """
  logger.info("Finding largest common substructure (LCS) for clustering")
  conformer_mols = []
  for conf in mol.GetConformers():
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(conf, assignId=True)
    conformer_mols.append(new_mol)

  mcs_result = rdFMCS.FindMCS(conformer_mols, completeRingsOnly=True, ringMatchesRingOnly=True)
  core = Chem.MolFromSmarts(mcs_result.smartsString)
  assert core is not None and core.GetNumAtoms(), "No core substructure detected. Aligning using first conformer."

  logger.debug("Core substructure detected. Aligning using core.")
  AllChem.AlignMolConformers(mol, mol.GetSubstructMatch(core))
  return mol


def minimize(mol: Chem.rdchem.Mol, method: str = "MMFF94", percentile: float = 100.0) -> Tuple[float, Dict[int, float]]:
  """Minimize conformer energy (kcal/mol) using RDkit
  and filter out conformers based on energy percentile.

  Parameters:
      mol: RDkit mol object containing the conformers you want to minimize. (rdkit.Chem.rdchem.Mol)
      method: Can be either UFF, MMFF94, or MMFF94s (str)
      percentile: Filters out conformers above a given energy percentile (0 to 100). For example, 10.0 will retain conformers within the lowest 10% energy. (float)

  Returns:
      A tuple of the form ``(mol_filtered, energies)``
      - ``mol_filtered``: Molecule object with filtered conformers.
      - ``energies``: Dictionary where keys are conformer IDs and values are calculated energies in kcal/mol.

  """
  logger.info(f"Minimizing energy of all conformers using {method}")
  energies = {}
  for i in range(mol.GetNumConformers()):
    if method == "UFF":
      AllChem.UFFOptimizeMolecule(mol, confId=i)
      ff = AllChem.UFFGetMoleculeForceField(mol, confId=i)
    elif method in ["MMFF94", "MMFF94s"]:
      AllChem.MMFFOptimizeMolecule(mol, confId=i, mmffVariant=method)
      props = AllChem.MMFFGetMoleculeProperties(mol)
      if props is None:
        logger.warning(f"MMFF properties could not be initialized for molecule: {Chem.MolToSmiles(mol)}")
        energies[i] = 0
        continue
      ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=i)
    else:
      raise ValueError(f"Invalid minimization method: {method}")

    if ff is None:
      logger.warning(f"Force field could not be initialized for conformer {i}, setting value to 0.")
      energies[i] = 0
      continue
    energies[i] = ff.CalcEnergy()

  energy_values = list(energies.values())
  energy_threshold = np.percentile(energy_values, percentile)

  mol_filtered = Chem.Mol(mol)
  mol_filtered.RemoveAllConformers()
  energies_filtered = {}
  num_filtered = 0
  for conf_id, energy in energies.items():
    if energy <= energy_threshold:
      num_filtered += 1
      conformer = mol.GetConformer(conf_id)
      new_conf_id = mol_filtered.AddConformer(conformer, assignId=True)
      energies_filtered[new_conf_id] = energy

  logger.info(f"Filtered {num_filtered:,}/{len(energy_values):,} conformers within the lowest {percentile}% of energies.")
  logger.info(f"min: {min(energy_values):.2f}, max: {max(energy_values):.2f}, avg: {sum(energy_values)/len(energy_values):.2f}")
  return mol_filtered, energies_filtered


def generate(
  input_mol: Any,
  output_name: str = "unique_conformers",
  write_multi: bool = False,
  num_confs: int = 1000,
  min_method: Optional[str] = "auto",
  max_atoms: int = 500,
) -> pd.DataFrame:
  """Generate conformers for an input molecule.

  Performs the following actions in order:
  1. Generate conformers using ETKDG method
  2. Minimize energy of all conformers and remove those below a dynamic threshold
  3. Align & create RMSD matrix of all conformers
  4. Clusters using Butina method to remove structurally redundant conformers
  5. Return most energetically favorable conformers in each cluster

  Parameters:
      input_mol: Input molecule can be a path to a molecule file, a SMILES string, or an instance of rdkit.Chem.rdchem.Mol
      output_name: Output to write SDF files of passing conformers
      write_multi: If True will write all unique conformers to a single SDF file, if False will write all unique conformers in separate SDF files in output_name
      num_confs: Number of conformers to generate
      min_method: Method for minimization, can be either "auto", "UFF", "MMFF94", "MMFF94s", or None for no minimization
      max_atoms: Maximum number of atoms allowed for the input molecule

  Returns:
      A dataframe with all conformer statistics. Note if energy minimization is disabled or fails then energy column will consist of None values.

  """
  my_mol = None
  if isinstance(input_mol, rdkit.Chem.rdchem.Mol):
    my_mol = input_mol
  if isinstance(input_mol, str):
    if input_mol.endswith(".pdb"):
      my_mol = Chem.MolFromPDBFile(input_mol)
    elif input_mol.endswith(".sdf"):
      for mol in Chem.SDMolSupplier(input_mol):
        if mol is not None:
          my_mol = mol
          break
      assert my_mol is not None, "Unable to find a valid molecule in the provided SDF file."
    else:
      my_mol = Chem.MolFromSmiles(input_mol)
      assert my_mol is not None, f"Invalid smiles string provided {input_mol}"
  else:
    raise ValueError(
      "Invalid input type for input_mol. Input molecule can be a path to a molecule file, a SMILES string, or an instance of rdkit.Chem.rdchem.Mol"
    )

  my_mol = Chem.AddHs(my_mol)

  if my_mol.GetNumAtoms() > max_atoms:
    raise ValueError(
      f"Input molecule exceeds the maximum allowed atoms ({max_atoms}). It has {my_mol.GetNumAtoms()} atoms. For proteins try using a different service such as the AFcluster and AlphaFlow tools on Neurosnap."
    )

  logger.debug(f"Generating {num_confs:,} 3D conformers.")
  params = AllChem.ETKDGv3()
  params.numThreads = 0
  params.optimizerForceTol = 0.0001
  params.useRandomCoords = True
  params.pruneRmsThresh = 0.1
  rdDistGeom.EmbedMultipleConfs(my_mol, numConfs=num_confs, params=params)
  logger.info(f"Generated {my_mol.GetNumConformers():,} 3D conformers.")

  energies = {}
  if min_method:
    if min_method == "auto":
      for method in ["MMFF94", "UFF", "MMFF94s"]:
        try:
          my_mol, energies = minimize(my_mol, method=method)
          break
        except Exception:
          pass
    else:
      my_mol, energies = minimize(my_mol, method=min_method)

  logger.debug("Calculating RMSD matrix between all conformer pairs")
  num_conformers = my_mol.GetNumConformers()
  my_mol_nh = Chem.RemoveHs(my_mol)
  rmslist = AllChem.GetConformerRMSMatrix(my_mol_nh)
  print(rmslist)

  num_heavy_atoms = my_mol.GetNumHeavyAtoms()
  c = 0.2
  threshold = c * np.sqrt(num_heavy_atoms)
  logger.debug(f"Dynamic RMSD threshold: {threshold:.2f} Å")

  logger.debug(f"Clustering {num_conformers:,} conformers.")
  clusters = Butina.ClusterData(rmslist, num_conformers, threshold, isDistData=True, reordering=True)
  logger.debug(f"Produced {len(clusters):,} clusters.")

  output = {
    "conformer_id": [],
    "cluster_id": [],
    "energy": [],
    "cluster_size": [],
  }
  for i, cluster in enumerate(clusters, start=1):
    best_energy = float("inf")
    best_cid = None
    if energies:
      for cid in cluster:
        if energies[cid] < best_energy:
          best_cid = cid
          best_energy = energies[cid]
    else:
      best_cid = cluster[0]
      best_energy = None

    logger.debug(f"Cluster ID: {i}, Best Conformer: {best_cid} ({float('nan') if best_energy is None else best_energy:.2f}), Conformers {cluster}")
    output["conformer_id"].append(best_cid)
    output["cluster_id"].append(i)
    output["energy"].append(best_energy)
    output["cluster_size"].append(len(cluster))

  df = pd.DataFrame(output)
  logger.info(f"Selected {len(df)} unique conformers.")
  print(df)

  if write_multi:
    with Chem.SDWriter(f"{output_name}.sdf") as w:
      for conf_id in output["conformer_id"]:
        w.write(my_mol, confId=conf_id)
  else:
    os.makedirs(output_name, exist_ok=True)
    for conf_id in output["conformer_id"]:
      with Chem.SDWriter(os.path.join(output_name, f"conformer_{conf_id}.sdf")) as w:
        w.write(my_mol, confId=conf_id)

  return df
