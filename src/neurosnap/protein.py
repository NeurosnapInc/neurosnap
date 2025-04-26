"""
Provides functions and classes related to processing protein data as well as
a feature rich wrapper around protein structures using BioPython.
"""

import io
import json
import os
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBIO, MMCIFParser, PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Superimposer import Superimposer
from matplotlib import collections as mcoll
from rdkit import Chem
from requests_toolbelt.multipart.encoder import MultipartEncoder
from scipy.special import expit as sigmoid
from tqdm import tqdm

import neurosnap.algos.lDDT as lDDT
from neurosnap.api import USER_AGENT
from neurosnap.constants import (
  AA_ABR_TO_CODE,
  AA_ABR_TO_NAME,
  AA_CODE_TO_ABR,
  AA_CODE_TO_NAME,
  AA_NAME_TO_ABR,
  AA_NAME_TO_CODE,
  BACKBONE_ATOMS,
  HYDROPHOBIC_RESIDUES,
  STANDARD_NUCLEOTIDES,
  STANDARD_AAs_ABR,
)
from neurosnap.log import logger
from neurosnap.msa import read_msa


### CLASSES ###
class Protein:
  def __init__(self, pdb: Union[str, io.IOBase], format: str = "auto"):
    """Class that wraps around a protein structure.

    Utilizes the biopython protein structure under the hood.
    Atoms that are not part of a chain will automatically be
    added to a new chain that does not overlap with any
    existing chains.

    Parameters:
      pdb: Can be either a file handle, PDB or mmCIF filepath, PDB ID, or UniProt ID
      format: File format of the input ("pdb", "mmcif", or "auto" to infer format from extension)
    """
    self.title = "Untitled Protein"
    if isinstance(pdb, io.IOBase):
      pass
    elif isinstance(pdb, str):
      if os.path.exists(pdb):
        self.title = pdb.split("/")[-1]
      else:
        self.title = pdb.upper()
        if len(pdb) == 4:  # check if a valid PDB ID
          r = requests.get(f"https://files.rcsb.org/download/{pdb.upper()}.pdb")
          r.raise_for_status()
          with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
            temp_file.write(r.content)
            pdb = temp_file.name
            logger.info("Found matching structure in RCSB PDB, downloading and using that.")
        else:  # check if provided a uniprot ID and fetch from AF2
          r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{pdb.upper()}")
          if r.status_code != 200:
            raise ValueError("Invalid input type provided. Can be either a file handle, PDB filepath, PDB ID, or UniProt ID")
          r = requests.get(r.json()[0]["pdbUrl"])
          if r.status_code != 200:
            raise ValueError("Invalid input type provided. Can be either a file handle, PDB filepath, PDB ID, or UniProt ID")
          with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
            temp_file.write(r.content)
            pdb = temp_file.name
            logger.info("Found matching structure in AF-DB, downloading and using that.")
    else:
      raise ValueError("Invalid input type provided. Can be either a file handle, PDB filepath, PDB ID, or UniProt ID")

    # Infer format if set to auto
    if format == "auto":
      if str(pdb).lower().endswith(".pdb"):
        format = "pdb"
      elif str(pdb).lower().endswith(".cif") or str(pdb).lower().endswith(".mmcif"):
        format = "mmcif"
      else:
        raise ValueError("Failed to infer format. Please specify format explicitly as 'pdb' or 'mmcif'.")

    # load structure
    if format == "pdb":
      parser = PDBParser()
    elif format == "mmcif":
      parser = MMCIFParser()
    else:
      raise ValueError("Invalid format specified. Supported formats are 'pdb' or 'mmcif'.")

    self.structure = parser.get_structure("structure", pdb)
    assert len(self.structure), "No models found. Structure appears to be empty."

    # Adds any missing chains to the structure.
    # Ensures new chain IDs do not overlap with existing ones.
    existing_chains = self.chains()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numeric_suffix = 1
    new_chain_id = None

    # Find an available chain ID that does not overlap with existing ones
    for char in alphabet:
      if char not in existing_chains:
        new_chain_id = char
        break
    if new_chain_id is None:
      # If all single-letter chain IDs are used, append a numeric suffix
      while new_chain_id is None:
        for char in alphabet:
          candidate_id = f"{char}{numeric_suffix}"
          if candidate_id not in existing_chains:
            new_chain_id = candidate_id
            break
        numeric_suffix += 1

    if new_chain_id:
      # rename chain
      for model in self.structure:
        for chain in model:
          if chain.id is None or chain.id == " ":
            chain.id = new_chain_id
            logger.info(f"Assigned ID '{new_chain_id}' to chain with missing ID in model '{model.id}'.")
    else:
      logger.warning(f"No chain IDs available. Could not rename chain with missing ID in model '{model.id}'.")

    # generate the pandas dataframe similar to that of biopandas
    self.generate_df()

  def __repr__(self):
    return f"<Neurosnap Protein: Title={self.title} Models={self.models()}, Chains=[{', '.join(self.chains())}], Atoms={len(self.df)}>"

  def __call__(self, model: Optional[int] = None, chain: Optional[int] = None, res_type: Optional[int] = None) -> pd.DataFrame:
    """Returns a selection of a copy of the internal dataframe
    that matches the provided query. If no queries are
    provided, will return a copy of the internal dataframe.

    Parameters:
      model: If provided, returned atoms must match this model
      chain: If provided, returned atoms must match this chain
      res_type: If provided, returned atoms must match this res_type

    Returns:
      Copy of the internal dataframe that matches the input query

    """
    df = self.df.copy()
    if model is not None:
      df = df.loc[df.model == model]
    if chain is not None:
      df = df.loc[df.chain == chain]
    if res_type is not None:
      df = df.loc[df.res_type == res_type]
    return df

  def __sub__(self, other_protein: "Protein") -> pd.DataFrame:
    """Automatically calculate the RMSD of two proteins.
    Model used will naively be the first models that have
    identical backbone shapes.
    Essentially just wraps around :obj:`self.calculate_rmsd()`

    Parameters:
      other_protein: Another Protein object to compare against

    Returns:
      Copy of the internal dataframe that matches the input query

    """
    # Get models for each structure using backbone shape
    model1 = None
    model2 = None
    for m1 in self.structure:
      backbone1 = self.get_backbone(model=m1.id)
      for m2 in other_protein.structure:
        backbone2 = other_protein.get_backbone(model=m2.id)
        if backbone1.shape == backbone2.shape:
          model1 = m1.id
          model2 = m2.id
          break

    assert model1 is not None, (
      "Could not find any matching matching models to calculate RMSD for. Please ensure at least two models with matching backbone shapes are provided."
    )
    return self.calculate_rmsd(other_protein, model1=model1, model2=model2)

  def models(self) -> List[int]:
    """Returns a list of all the model names/IDs.

    Returns:
      models: Chain names/IDs found within the PDB file

    """
    return [model.id for model in self.structure]

  def chains(self, model: int = 0) -> List[str]:
    """Returns a list of all the chain names/IDs.

    Parameters:
      model: The ID of the model you want to fetch the chains of, defaults to 0

    Returns:
      Chain names/IDs found within the PDB file

    """
    return [chain.id for chain in self.structure[model] if chain.id.strip()]

  def generate_df(self):
    """Generate the biopandas-like dataframe and update the
    value of self.df to the new dataframe.
    This method should be called whenever the internal
    protein structure is modified or has a transformation
    applied to it.

    Inspired by: https://biopandas.github.io/biopandas

    """
    df = {
      "model": [],
      "chain": [],
      "res_id": [],
      "res_name": [],
      "res_type": [],
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
            # get residue type
            res_type = "HETEROGEN"
            if res.id[0] == " " and res.resname in AA_ABR_TO_CODE:
              res_type = "AMINO_ACID"
            elif res.id[0] == " " and res.resname in STANDARD_NUCLEOTIDES:
              res_type = "NUCLEOTIDE"
            df["model"].append(model.id)
            df["chain"].append(chain.id)
            df["res_id"].append(res.id[1])
            df["res_name"].append(res.resname)
            df["res_type"].append(res_type)
            df["atom"].append(atom.serial_number)
            df["atom_name"].append(atom.name)
            df["bfactor"].append(atom.bfactor)
            df["x"].append(atom.coord[0])
            df["y"].append(atom.coord[1])
            df["z"].append(atom.coord[2])
            df["mass"].append(atom.mass)
    self.df = pd.DataFrame(df)

  def get_aas(self, chain: str, model: Optional[int] = None) -> str:
    """Returns the amino acid sequence of a target chain.
    Ligands, small molecules, and nucleotides are ignored.

    Parameters:
      chain: The ID of the chain you want to fetch the AA sequence of
      model: The ID of the model containing the target chain, if set to None will default to first model

    Returns:
      The amino acid sequence of the found chain

    """
    if model is None:
      model = self.models()[0]
    assert model in self.structure, f'Protein does not contain model "{model}"'
    assert chain in self.structure[model], f'Model {model} does not contain chain "{chain}"'

    seq = ""
    for res in self.structure[model][chain]:
      resn = res.get_resname()
      if resn in STANDARD_AAs_ABR:
        seq += getAA(resn)[0]

    return seq

  def select_residues(self, selectors: str, model: Optional[int] = None) -> Dict[str, List[int]]:
    """Select residues from a protein structure using a string selector.

    This method allows for flexible selection of residues in a protein structure
    based on a string query. The query must be a comma-delimited list of selectors
    following these patterns:

    - "C": Select all residues in chain C.
    - "B1": Select residue with identifier 1 in chain B only.
    - "A10-20": Select residues with identifiers 10 to 20 (inclusive) in chain A.
    - "A15,A20-23,B": Select residues 15, 20, 21, 22, 23, and all residues in chain B.

    If any selector does not match residues in the structure, an exception is raised.

    Parameters:
        selectors: A string specifying the residue selection query.
        model: The ID of the model to select from. If None, the first model is used.

    Returns:
        dict: A dictionary where keys are chain IDs and values are sorted
              lists of residue sequence numbers that match the query.

    Raises:
        ValueError: If a specified chain or residue in the selector does not exist in the structure.
    """
    if model is None:
      model = self.models().pop(0)
    elif model not in self.structure:
      raise ValueError(f'Protein does not contain model "{model}"')

    # get chains and create output object
    chains = self.chains()
    output = {}
    for chain in chains:
      output[chain] = set()

    # compile regular expressions
    pattern_res_single = re.compile(r"^[A-Za-z](\d{1,})$")
    pattern_res_range = re.compile(r"^[A-Za-z](\d{1,})-(\d{1,})$")

    # remove white space
    selectors = re.sub(r"\s", "", selectors)
    # remove any leading or trailing commas
    selectors = selectors.strip(",")
    # remove ",," if present
    while ",," in selectors:
      selectors = selectors.replace(",,", ",")

    # get selection
    for selector in selectors.split(","):
      # get and validate chain
      chain = selector[0]
      if chain not in chains:
        raise ValueError(f'Chain "{chain}" in selector "{selector}" does not exist in the specified structure.')

      # if select entire chain
      if len(selector) == 1:
        self.df[(self.df["chain"] == chain)]
        output[chain] = output[chain].union(self.df[(self.df["chain"] == chain)]["res_id"].to_list())
        continue

      # if select single residue
      found = pattern_res_single.search(selector)
      if found:
        resi = int(found.group(1))
        if self.df[(self.df["chain"] == chain) & (self.df["res_id"] == resi)].empty:
          raise ValueError(f'Residue "{resi}" in selector "{selector}" does not exist in the specified chain.')
        else:
          output[chain].add(resi)
        continue

      # if select residue range
      found = pattern_res_range.search(selector)
      if found:
        resi_start = int(found.group(1))
        resi_end = int(found.group(2))
        if resi_start > resi_end:
          raise ValueError(f'Invalid residue range selector "{selector}". The starting residue cannot be greater than the ending residue.')
        for resi in range(resi_start, resi_end + 1):
          if self.df[(self.df["chain"] == chain) & (self.df["res_id"] == resi)].empty:
            raise ValueError(f'Residue "{resi}" in selector "{selector}" does not exist in the specified chain.')
          else:
            output[chain].add(resi)
        continue

    # remove empty chains and convert to sorted array
    empty = []
    for chain, resis in output.items():
      if resis:
        output[chain] = sorted(list(resis))
      else:
        empty.append(chain)
    for chain in empty:
      del output[chain]

    return output

  def renumber(self, model: Optional[int] = None, chain: Optional[int] = None, start: int = 1):
    """Renumbers all selected residues. If selection does not
    exist this function will do absolutely nothing.

    Parameters:
      model: The model ID to renumber. If ``None``, will use all models.
      chain: The chain ID to renumber. If ``None``, will use all models.
      start: Starting value to increment from, defaults to 1.

    """

    def aux(start):
      for m in self.structure:
        if model is None or m.id == model:
          for c in m:
            if chain is None or c.id == chain:
              for res in c:
                # Renumber residue
                res.id = (res.id[0], start, res.id[2])
                start += 1

    # perform initial renumbering to avoid collisions
    aux(-100000)
    # perform actual renumbering
    aux(start)
    # update the pandas dataframe
    self.generate_df()

  def remove_waters(self):
    """Removes all water molecules (residues named 'WAT' or 'HOH')
    from the structure. It is suggested to call :obj:`.renumber()`
    afterwards as well.

    """
    for model in self.structure:
      for chain in model:
        # Identify water molecules and mark them for removal
        residues_to_remove = [res for res in chain if res.get_resname() in ["WAT", "HOH"]]

        # Remove water molecules
        for res in residues_to_remove:
          chain.detach_child(res.id)
    # update the pandas dataframe
    self.generate_df()

  def remove_nucleotides(self, model: Optional[int] = None, chain: Optional[str] = None):
    """Removes all nucleotides (DNA and RNA) from the structure.
    If no model or chain is provided, it will remove nucleotides
    from the entire structure.

    Parameters:
      model: The model ID to process. If ``None``, will use all models.
      chain: The chain ID to process. If ``None``, will use all chains.

    """
    for m in self.structure:
      if model is None or m.id == model:
        for c in m:
          if chain is None or c.id == chain:
            # Identify nucleotide residues (both RNA and DNA)
            residues_to_remove = [res for res in c if res.get_resname() in STANDARD_NUCLEOTIDES]

            # Remove nucleotide residues
            for res in residues_to_remove:
              c.detach_child(res.id)

    # update the pandas dataframe
    self.generate_df()

  def remove_non_biopolymers(self, model: Optional[int] = None, chain: Optional[str] = None):
    """Removes all ligands, heteroatoms, and non-biopolymer
    residues from the selected structure. Non-biopolymer
    residues are considered to be any residues that are not
    standard amino acids or standard nucleotides (DNA/RNA).
    If no model or chain is provided, it will remove from
    the entire structure.

    Parameters:
      model: The model ID to process. If ``None``, will use all models.
      chain: The chain ID to process. If ``None``, will use all chains.

    """
    # List of standard amino acids and nucleotides (biopolymer residues)
    biopolymer_residues = set(AA_ABR_TO_CODE.keys()).union(STANDARD_NUCLEOTIDES)
    biopolymer_residues.remove("UNK")

    for m in self.structure:
      if model is None or m.id == model:
        for c in m:
          if chain is None or c.id == chain:
            # Identify non-biopolymer residues (ligands, heteroatoms, etc.)
            residues_to_remove = [res for res in c if res.get_resname() not in biopolymer_residues]

            # Remove non-biopolymer residues
            for res in residues_to_remove:
              c.detach_child(res.id)
    # update the pandas dataframe
    self.generate_df()

  def get_backbone(self, model: Optional[int] = None, chain: Optional[str] = None) -> np.ndarray:
    """Extract backbone atoms (N, CA, C) from the structure.
    If model or chain is not provided, extracts from all models/chains.

    Parameters:
      model: Model ID to extract from. If ``None``, all models are included.
      chain: Chain ID to extract from. If ``None``, all chains are included.

    Returns:
      A numpy array of backbone coordinates (Nx3)

    """
    backbone_coords = []

    for m in self.structure:
      if model is None or m.id == model:
        for c in m:
          if chain is None or c.id == chain:
            for res in c:
              for atom in res:
                if atom.name in BACKBONE_ATOMS:
                  backbone_coords.append(atom.coord)

    return np.array(backbone_coords)

  def find_disulfide_bonds(self, chain: Optional[str] = None, model: Optional[int] = None, threshold: float = 2.05) -> List[Tuple]:
    """Find disulfide bonds between Cysteine residues in the structure.
    Looks for SG-SG bonds within a threshold distance.

    Parameters:
      chain: Chain ID to search. If ``None``, all chains are searched.
      model: Model ID to search, If ``None``, the first available model is searched.
      threshold: Maximum distance to consider a bond between SG atoms, in angstroms.
        Default is 2.05 Å.

    Returns:
      List of tuples of residue pairs forming disulfide bonds

    """
    if model is None:
      model = self.models().pop(0)

    disulfide_pairs = []

    for c in self.structure[model]:
      if chain is None or c.id == chain:
        cysteines = [res for res in c if res.get_resname() == "CYS"]
        for i, res1 in enumerate(cysteines):
          for res2 in cysteines[i + 1 :]:
            try:
              sg1 = res1["SG"]
              sg2 = res2["SG"]
              distance = sg1 - sg2
              if distance < threshold:
                disulfide_pairs.append((res1, res2))
            except KeyError:
              pass  # Skip if no SG atom found
    return disulfide_pairs

  def find_salt_bridges(self, chain: Optional[str] = None, model: Optional[int] = None, cutoff: float = 4.0) -> List[Tuple]:
    """Identify salt bridges between oppositely charged residues.
    A salt bridge is defined as an interaction between
    a positively charged residue (Lys, Arg) and a negatively
    charged residue (Asp, Glu) within a given cutoff distance.

    Parameters:
      chain: Chain ID to search. If ``None``, all chains are searched.
      model: Model ID to search, If ``None``, the first available model is searched.
      cutoff: Maximum distance for a salt bridge (float)

    Returns:
      List of residue pairs forming salt bridges

    """
    positive_residues = {"LYS", "ARG"}
    negative_residues = {"ASP", "GLU"}
    salt_bridges = []

    if model is None:
      model = self.models().pop(0)

    for c in self.structure[model]:
      if chain is None or c.id == chain:
        pos_residues = [res for res in c if res.get_resname() in positive_residues]
        neg_residues = [res for res in c if res.get_resname() in negative_residues]
        for pos_res in pos_residues:
          for neg_res in neg_residues:
            dist = pos_res["CA"] - neg_res["CA"]  # Use alpha-carbon distance as a proxy
            if dist < cutoff:
              salt_bridges.append((pos_res, neg_res))
    return salt_bridges

  def find_hydrophobic_residues(self, chain: Optional[str] = None, model: Optional[int] = None) -> List[Tuple]:
    """Identify hydrophobic residues in the structure.

    Parameters:
      chain: Chain ID to extract from. If ``None``, all chains are checked.
      model: Model ID to extract from. If ``None``, the first available model is searched.

    Returns:
      List of tuples ``(chain_id, residue)`` for hydrophobic residues

    """
    hydrophobic_residues = []

    if model is None:
      model = self.models().pop(0)

    for c in self.structure[model]:
      if chain is None or c.id == chain:
        for res in c:
          if res.get_resname() in HYDROPHOBIC_RESIDUES:
            hydrophobic_residues.append((c.id, res))

    return hydrophobic_residues

  def find_missing_residues(self, chain: Optional[str] = None) -> List[int]:
    """Identify missing residues in the structure based on residue numbering.
    Useful for identifying gaps in the structure.

    Parameters:
      chain: Chain ID to inspect. If ``None``, all chains are inspected.

    Returns:
      missing_residues: List of missing residue positions

    """
    missing_residues = []

    for model in self.structure:
      for chain in model:
        residues = sorted(res.id[1] for res in chain)
        for i in range(len(residues) - 1):
          if residues[i + 1] != residues[i] + 1:
            missing_residues.extend(range(residues[i] + 1, residues[i + 1]))

    return missing_residues

  def align(self, other_protein: "Protein", model1: int = 0, model2: int = 0, chain1: List[str] = [], chain2: List[str] = []):
    """Align another Protein object's structure to the self.structure
    of the current object. The other Protein will be transformed
    and aligned. Only compares backbone atoms (N, CA, C).

    Parameters:
      other_protein: Another Neurosnap Protein object to compare against
      model1: Model ID of reference protein to align to
      model2: Model ID of other protein to transform and align to reference
      chain1: The chain(s) you want to include in the alignment within the reference protein, set to an empty list to use all chains.
      chain2: The chain(s) you want to include in the alignment within the other protein, set to an empty list to use all chains.

    """
    assert model1 in self.models(), "Specified model needs to be present in the reference structure."
    assert model2 in other_protein.models(), "Specified model needs to be present in the other structure."
    # validate chains
    avail_chains = self.chains(model1)
    for chain in chain1:
      assert chain in avail_chains, f"Chain {chain} was not found in the reference protein. Found chains include {', '.join(avail_chains)}."
    avail_chains = other_protein.chains(model2)
    for chain in chain2:
      assert chain in avail_chains, f"Chain {chain} was not found in the other protein. Found chains include {', '.join(avail_chains)}."

    # Use the Superimposer to align the structures
    def aux_get_atoms(sample_model, chains):
      atoms = []
      for sample_chain in sample_model:
        if not chains or sample_chain.id in chains:
          for res in sample_chain:
            for atom in res:
              if atom.name in BACKBONE_ATOMS:
                atoms.append(atom)
      return atoms

    sup = Superimposer()
    sup.set_atoms(aux_get_atoms(self.structure[model1], chain1), aux_get_atoms(other_protein.structure[model2], chain2))
    sup.apply(other_protein.structure[model2])  # Apply the transformation to the other protein
    # update the pandas dataframe
    other_protein.generate_df()

  def calculate_rmsd(
    self, other_protein: "Protein", model1: int = 0, model2: int = 0, chain1: Optional[str] = None, chain2: Optional[str] = None, align: bool = True
  ) -> float:
    """Calculate RMSD between the current structure and another protein.
    Only compares backbone atoms (N, CA, C). RMSD is in angstroms (Å).

    Parameters:
      other_protein: Another Protein object to compare against
      model1: Model ID of original protein to compare
      model2: Model ID of other protein to compare
      chain1: Chain ID of original protein, if not provided compares all chains
      chain2: Chain ID of other protein, if not provided compares all chains
      align: Whether to align the structures first using Superimposer

    Returns:
      The root-mean-square deviation between the two structures

    """
    # ensure models are present
    assert model1 in self.models(), f"Model {model1} was not found in current protein."
    assert model2 in other_protein.models(), f"Model {model2} was not found in other protein."

    # Get backbone coordinates of both structures
    backbone1 = self.get_backbone(model=model1, chain=chain1)
    backbone2 = other_protein.get_backbone(model=model2, chain=chain2)
    assert backbone1.shape == backbone2.shape, "Structures must have the same number of backbone atoms for RMSD calculation."

    if align:
      self.align(other_protein, model1=model1, model2=model2)

    # Get new backbone coordinates of both structures
    backbone1 = self.get_backbone(model=model1, chain=chain1)
    backbone2 = other_protein.get_backbone(model=model2, chain=chain2)

    diff = backbone1 - backbone2
    rmsd = np.sqrt(np.sum(diff**2) / backbone1.shape[0])
    return rmsd

  def calculate_distance_matrix(self, model: Optional[int] = None, chain: Optional[str] = None) -> np.ndarray:
    """Calculate the distance matrix for all alpha-carbon (CA) atoms in the chain.
    Useful for creating contact maps or proximity analyses.

    Parameters:
      model: The model ID to calculate the distance matrix for, if not provided will use first model found
      chain: The chain ID to calculate, if not provided calculates for all chains

    Returns:
      A 2D numpy array representing the distance matrix

    """
    model = self.models()[0]
    ca_atoms = []

    for m in self.structure:
      if m.id == model:
        for c in m:
          if chain is None or c.id == chain:
            for res in c:
              if "CA" in res:
                ca_atoms.append(res["CA"].coord)

    ca_atoms = np.array(ca_atoms)
    dist_matrix = np.sqrt(np.sum((ca_atoms[:, np.newaxis] - ca_atoms[np.newaxis, :]) ** 2, axis=-1))
    return dist_matrix

  def calculate_center_of_mass(self, model: Optional[int] = 0, chains: Optional[List[str]] = None) -> np.ndarray:
    """Calculate the center of mass of the protein.
    Considers only atoms with defined masses.

    Parameters:
      model: Model ID to calculate for, if not provided defaults to 0
      chains: List of chain IDs to calculate for, if not provided calculates for all chains

    Returns:
      center_of_mass: A 3D numpy array representing the center of mass

    """
    total_mass = 0
    weighted_coords = np.zeros(3)

    for m in self.structure:
      if m.id == model:
        for c in m:
          # Check if the current chain should be included
          if chains is None or c.id in chains:
            for res in c:
              for atom in res:
                if atom.mass is not None:
                  total_mass += atom.mass
                  weighted_coords += atom.mass * atom.coord

    if total_mass == 0:
      raise ValueError("No atoms with mass found in the selected structure.")

    return weighted_coords / total_mass

  def distances_from_com(self, model: Optional[int] = 0, chains: Optional[List[str]] = None) -> np.ndarray:
    """Calculate the distances of all atoms from the center of mass (COM) of the protein.

    This method computes the Euclidean distance between the coordinates of each atom
    and the center of mass of the structure. The center of mass is calculated for the
    specified model and chain, or for all models and chains if none are provided.

    Parameters:
      model: The model ID to calculate for. If not provided, defaults to 0.
      chains: List of chain IDs to calculate for. If not provided, calculates for all chains.

    Returns:
      A 1D NumPy array containing the distances (in Ångströms) between each atom and the center of mass.

    """
    com = self.calculate_center_of_mass(model=model, chains=chains)
    distances = []

    for m in self.structure:
      if m.id == model:
        for c in m:
          # Check if the current chain should be included
          if chains is None or c.id in chains:
            for res in c:
              for atom in res:
                distance = np.linalg.norm(atom.coord - com)
                distances.append(distance)

    return np.array(distances)  # Convert the list of distances to a NumPy array

  def calculate_rog(self, model: Optional[int] = 0, chains: Optional[List[str]] = None) -> float:
    """Calculate the radius of gyration (Rg) of the protein.

    The radius of gyration measures the overall size and compactness of the protein structure.
    It is calculated based on the distances of atoms from the center of mass (COM).

    Parameters:
      model: The model ID to calculate for. If not provided, defaults to 0.
      chains: List of chain IDs to calculate for. If not provided, calculates for all chains.

    Returns:
      The radius of gyration (Rg) in Ångströms. Returns 0.0 if no atoms are found.

    """
    distances_sq = self.distances_from_com(model=model, chains=chains) ** 2

    if distances_sq.size == 0:
      logger.warning("No atoms found for the specified model/chains. Returning Rg = 0.0")
      return 0.0

    rg = np.sqrt(np.sum(distances_sq) / distances_sq.size)
    return float(rg)

  def calculate_surface_area(self, model: int = 0, level: str = "R") -> float:
    """Calculate the solvent-accessible surface area (SASA) of the protein.
    Utilizes Biopython's SASA module.

    Parameters:
      model: The model ID to calculate SASA for, defaults to 0.
      level: The level at which ASA values are assigned, which can be one of "A" (Atom), "R" (Residue), "C" (Chain), "M" (Model), or "S" (Structure). The ASA value of an entity is the sum of all ASA values of its children.

    Returns:
      Solvent-accessible surface area in Å²

    """
    from Bio.PDB import SASA  # NOTE: SASA isn't imported the same depending on biopython version so import it here to prevent errors

    assert model in self.models(), f"Model {model} is not currently present."
    structure_model = self.structure[model]
    sasa_calculator = SASA.ShrakeRupley()
    sasa_calculator.compute(structure_model, level=level)
    total_sasa = sum([residue.sasa for residue in structure_model.get_residues() if residue.sasa])
    return total_sasa

  def calculate_protein_volume(self, model: int = 0, chain: Optional[str] = None) -> float:
    """Compute an estimate of the protein volume using the van der Waals radii.
    Uses the sum of atom radii to compute the volume.

    Parameters:
      model: Model ID to compute volume for, defaults to 0
      chain: Chain ID to compute, if not provided computes for all chains

    Returns:
      Estimated volume in Å³

    """
    assert model in self.models(), f"Model {model} is not currently present."
    vdw_radii = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "P": 1.8, "S": 1.8}  # Example radii in Å
    volume = 0

    for m in self.structure:
      for c in m:
        if chain is None or c.id == chain:
          for res in c:
            if is_aa(res):
              for atom in res:
                element = atom.element
                if element in vdw_radii:
                  radius = vdw_radii[element]
                  volume += (4 / 3) * np.pi * (radius**3)
    return volume

  def calculate_hydrogen_bonds(
    self,
    model: Optional[int] = None,
    chain: Optional[str] = None,
    chain_other: Optional[str] = None,
    donor_acceptor_cutoff: float = 3.5,
    angle_cutoff: float = 120.0,
  ) -> int:
    """Calculate the number of hydrogen bonds in the protein structure.
    Hydrogen atoms must be explicitly defined within the structure as implicit hydrogens
    will not computed. We recommend using a tool like reduce to add missing hydrogens.

    Hydrogen bonds are detected based on distance and angle criteria:
    - Distance between donor and acceptor must be less than `donor_acceptor_cutoff`.
    - The angle formed by donor-hydrogen-acceptor must be greater than `angle_cutoff`.

    If `model` is set to `None`, hydrogen bonds are calculated only for the first model in the structure.

    If `chain_other` is `None`:
      - Hydrogen bonds are calculated for the specified `chain` or all chains if `chain` is also `None`.
    If `chain_other` is set to a specific chain:
      - Hydrogen bonds are calculated only between atoms of `chain` and `chain_other`.
    If `chain_other` is specified but `chain` is not, an exception is raised.

    Parameters:
      model: Model ID to calculate for. If None, only the first model is considered.
      chain: Chain ID to calculate for. If None, all chains in the selected model are considered.
      chain_other: Secondary chain ID for inter-chain hydrogen bonds. If None, intra-chain bonds are calculated.
      donor_acceptor_cutoff: Maximum distance between donor and acceptor (in Å). Default is 3.5 Å.
      angle_cutoff: Minimum angle for a hydrogen bond (in degrees). Default is 120°.

    Returns:
      The total number of hydrogen bonds in the structure.

    Raises:
      ValueError: If `chain_other` is specified but `chain` is not.

    """
    hydrogen_bonds = 0
    hydrogen_distance_cutoff = 1.2  # Typical bond length between H and donor atom (in Å)

    # Default to the first model if model is None
    if model is None:
      model = self.models()[0]
    model = self.structure[model]

    # input validation
    if chain_other is not None and chain is None:
      raise ValueError("`chain_other` is specified, but `chain` is not. Both must be provided for inter-chain hydrogen bond calculation.")
    if chain is not None and chain not in self.chains():
      raise ValueError(f"Chain {chain} does not exist within the input structure.")
    if chain_other is not None and chain_other not in self.chains():
      raise ValueError(f"Chain {chain_other} does not exist within the input structure.")

    for c in model:
      if chain is not None and c.id != chain:
        continue

      for donor_res in c:
        for donor_atom in donor_res:
          if donor_atom.element not in ["N", "O"]:
            continue  # Only consider N or O as donors

          # Identify hydrogen atoms bonded to the donor
          bonded_hydrogens = [
            atom for atom in donor_res if atom.element == "H" and np.linalg.norm(donor_atom.coord - atom.coord) <= hydrogen_distance_cutoff
          ]
          if not bonded_hydrogens:
            continue

          # Determine chains to search for acceptors
          acceptor_chains = [chain_other] if chain_other else [c.id for c in model if chain is None or c.id == chain]

          for acceptor_chain in acceptor_chains:
            for acceptor_res in model[acceptor_chain]:
              for acceptor_atom in acceptor_res:
                if acceptor_atom.element not in ["N", "O"]:
                  continue  # Only consider N or O as acceptors
                if acceptor_atom is donor_atom:
                  continue  # Skip self-bonding

                # Calculate the distance between donor and acceptor
                distance = np.linalg.norm(donor_atom.coord - acceptor_atom.coord)
                if distance > donor_acceptor_cutoff:
                  continue

                # Calculate the angle formed by donor-hydrogen-acceptor for each bonded hydrogen
                for hydrogen in bonded_hydrogens:
                  donor_h_vector = hydrogen.coord - donor_atom.coord
                  acceptor_donor_vector = acceptor_atom.coord - donor_atom.coord
                  angle = np.arccos(
                    np.dot(donor_h_vector, acceptor_donor_vector) / (np.linalg.norm(donor_h_vector) * np.linalg.norm(acceptor_donor_vector))
                  )
                  angle = np.degrees(angle)

                  if angle >= angle_cutoff:
                    hydrogen_bonds += 1

    return hydrogen_bonds

  def to_sdf(self, fpath: str):
    """Save the current protein structure as an SDF file.
    Will export all models and chains. Use :obj:`.remove()`
    method to get rid of undesired regions.

    Parameters:
      fpath: Path to the output SDF file

    """
    # Write the current protein structure to a temporary PDB file
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdb") as temp_pdb:
      self.save(temp_pdb.name, format="pdb")
      pdb_fpath = temp_pdb.name

      # Read the PDB file
      mol = Chem.MolFromPDBFile(pdb_fpath, sanitize=False)

      if mol is None:
        raise ValueError("Unable to parse the PDB file.")

      # Write the molecule to SDF
      writer = Chem.SDWriter(fpath)
      writer.write(mol)
      writer.close()
      logger.info(f"Successfully wrote SDF file to {fpath}.")

  def remove(self, model: int, chain: Optional[str] = None, resi_start: Optional[int] = None, resi_end: Optional[int] = None):
    """Completely removes all parts of a selection from
    self.structure. If a residue range is provided then all
    residues between resi_start and resi_end will be removed
    from the structure (inclusively). If a residue range is
    not provided then all residues in a chain will be removed.

    Parameters:
      model: ID of model to remove from
      chain: ID of chain to remove from, if not provided will remove all chains in the model
      resi_start: Index of first residue in the range you want to remove
      resi_end: Index of last residues in the range you want to remove

    """
    # validate input query
    assert model in self.models(), f"Model ID {model} does not exist in your structure. Found models include {self.models()}."
    if chain is not None:
      assert chain in self.chains(model), f"Chain ID {chain} does not exist in your structure. Found chains include {self.chains(model)}."
    if resi_start is not None or resi_end is not None:
      assert chain is not None, "Chain needs to specified if you want to remove residues"
      assert resi_start is not None and resi_end is not None, "Both resi_start and resi_end must be provided"
      assert isinstance(resi_start, int) and isinstance(resi_end, int), "Both resi_start and resi_end must be valid integers"
      assert resi_end >= resi_start, "resi_start start must be less than resi_end"
      assert resi_start in self.structure[model][chain], f"Residue {resi_start} does not exist in the specified part of your structure."
      assert resi_end in self.structure[model][chain], f"Residue {resi_end} does not exist in the specified part of your structure."

    # Perform the removal
    if resi_start is not None and resi_end is not None:
      # Remove residues in the specified range
      chain_obj = self.structure[model][chain]
      residues_to_remove = [res for res in chain_obj if resi_start <= res.id[1] <= resi_end]
      for res in residues_to_remove:
        chain_obj.detach_child(res.id)
    elif chain is not None:
      # Remove the entire chain
      self.structure[model].detach_child(chain)
    else:
      # Remove the entire model
      self.structure.detach_child(model)

    # Update the pandas dataframe to reflect the changes
    self.generate_df()

  def save(self, fpath: str, format: str = "auto"):
    """Save the structure as a PDB or mmCIF file.
    Will overwrite any existing files.

    Parameters:
      fpath: File path where you want to save the structure
      format: File format to save in, either 'pdb' or 'mmcif', set to 'auto' to infer format from extension.

    """
    format = format.lower()
    # infer format from extension if not provided
    if format == "auto":
      if str(fpath).endswith(".pdb"):
        format = "pdb"
      elif str(fpath).endswith(".mmcif") or str(fpath).endswith(".cif"):
        format = "mmcif"
      else:
        raise ValueError("Failed to infer format for file. Supported extensions consist of '.pdb' or '.mmcif'.")

    # save file
    if format == "pdb":
      io = PDBIO()
      io.set_structure(self.structure)
      io.save(fpath, preserve_atom_numbering=True)
    elif format == "mmcif":
      mmcif_io = MMCIFIO()
      mmcif_io.set_structure(self.structure)
      mmcif_io.save(fpath)
    else:
      raise ValueError("Format must be 'pdb' or 'mmcif'.")


### FUNCTIONS ###
def getAA(query: str) -> Tuple[str, str, str]:
  """Efficiently get any amino acid using either their 1 letter code,
  3 letter abbreviation, or full name. See AAs_FULL_TABLE
  for a list of all supported amino acids and codes.

  Parameters:
    query: Amino acid code, abbreviation, or name

  Returns:
    A triple of the form ``(code, abr, name)``.

    - ``code`` is the amino acid 1 letter abbreviation / code
    - ``abr`` is the amino acid 3 letter abbreviation / code
    - ``name`` is the amino acid full name

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


def extract_non_biopolymers(pdb_file: str, output_dir: str, min_atoms: int = 0):
  """
  Extracts all non-biopolymer molecules (ligands, heteroatoms, etc.)
  from the specified PDB file and writes them to SDF files.
  Each molecule is saved as a separate SDF file in the output directory.
  Automatically adds hydrogens to molecules. Attempts to sanitize
  the molecule if possible; logs a warning if sanitization fails.

  Parameters:
      pdb_file: Path to the input PDB file.
      output_dir: Directory where the SDF files will be saved. Will overwrite existing directory.
      min_atoms: Minimum number of atoms a molecule must have to be saved. Molecules with fewer atoms are skipped.
  """

  def is_biopolymer(molecule):
    """
    Determines if a molecule is a biopolymer (protein or nucleotide) based on specific characteristics.
    Returns True if it is a biopolymer; False otherwise.
    """
    # Check for peptide bonds or nucleotide backbones
    # Simplified logic: exclude molecules with standard amino acids or nucleotide bases
    biopolymer_keywords = [
      "GLY",
      "ALA",
      "VAL",
      "LEU",
      "ILE",
      "MET",
      "PHE",
      "TYR",
      "TRP",
      "SER",
      "THR",
      "CYS",
      "PRO",
      "ASN",
      "GLN",
      "ASP",
      "GLU",
      "LYS",
      "ARG",
      "HIS",  # Amino acids
      "DA",
      "DT",
      "DG",
      "DC",
      "A",
      "T",
      "G",
      "C",
      "U",  # Nucleotide bases
    ]
    for atom in molecule.GetAtoms():
      residue_info = atom.GetPDBResidueInfo()
      if residue_info:
        res_name = residue_info.GetResidueName().strip()
        if res_name in biopolymer_keywords:
          return True
    return False

  # Create output directory if it doesn't exist
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  # Read the PDB file
  mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
  if mol is None:
    raise ValueError(f"Failed to read PDB file: {pdb_file}.")

  # Split the molecule into fragments (separate entities in the PDB)
  fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
  molecule_count = 1

  for i, frag in enumerate(fragments):
    if frag is None:
      logger.warning(f"Skipping fragment {i} due to processing failure.")
      continue

    try:
      # Add hydrogens and sanitize molecule
      Chem.SanitizeMol(frag)
    except Exception as e:
      logger.warning(f"Failed to sanitize fragment {i}: {e}")
      continue

    # Check if the fragment is a biopolymer
    if is_biopolymer(frag):
      logger.info(f"Skipping biopolymer fragment {i}.")
      continue

    # Skip small molecules based on atom count
    if frag.GetNumAtoms() < min_atoms:
      logger.info(f"Skipping small molecule fragment {i} (atom count: {frag.GetNumAtoms()}).")
      continue

    # Save fragment to SDF
    sdf_file = os.path.join(output_dir, f"ligand_{molecule_count}.sdf")
    writer = Chem.SDWriter(sdf_file)
    writer.write(frag)
    writer.close()
    molecule_count += 1

  logger.info(f"Extracted {molecule_count - 1} non-biopolymer molecules to {output_dir}.")


def calc_lDDT(ref_pdb: str, sample_pdb: str) -> float:
  """Calculates the lDDT (Local Distance Difference Test) between two proteins.

  Parameters:
    ref_pdb: Filepath for reference protein
    sample_pdb: Filepath for sample protein

  Returns:
    The lDDT score of the two proteins which ranges between 0-1

  """
  ref_L, ref_dmap, ref_rnames = lDDT.pdb2dmap(ref_pdb)
  mod_L, mod_dmap, mod_rnames = lDDT.pdb2dmap(sample_pdb)
  return lDDT.get_LDDT(ref_dmap, mod_dmap)


def fetch_accessions(accessions: Iterable[str], batch_size: int = 150) -> Dict[str, Union[str, None]]:
  """
  Fetch protein sequences corresponding to a list of UniProt accession numbers.

  This function retrieves sequences from the UniProt API, checking first the UniParc database and
  then UniProtKB if sequences are missing. Accessions are processed in batches to handle large lists efficiently.

  Args:
      accessions: A list of UniProt accession numbers. Duplicate accessions will be automatically removed.
      batch_size: Size of each batch of sequences to fetch from uniprot API per request.

  Returns:
      dict: A dictionary where keys are accession numbers and values are the corresponding protein sequences. Missing sequences will have the value None.

  Raises:
      requests.exceptions.HTTPError: If the API request fails and raises an HTTP error.

  Notes:
      - Batching is performed with a default batch size of 150, which was determined to be optimal during testing.
      - The function first queries the UniParc API and then queries the UniProtKB API for any missing accessions.

  Example:
      >>> accessions = ["P12345", "Q67890", "A1B2C3"]
      >>> sequences = fetch_accessions(accessions)
      >>> print(sequences["P12345"])
      "MEEPQSDPSV...GDE"

  Steps:
      1. Deduplicate the input list of accessions.
      2. Split the accessions into batches to query UniParc.
      3. Query the UniParc API for each batch and store results.
      4. Identify missing accessions and query UniProtKB.
      5. Validate that all input accessions were retrieved successfully.
  """
  accessions = list(set(str(x).strip() for x in accessions))

  # chunk into fragments to run separately
  batches = [accessions[i : i + batch_size] for i in range(0, len(accessions), batch_size)]

  output = {}
  for batch in tqdm(batches, desc="Fetching sequences from uniprot.org", total=len(batches)):
    query = " OR ".join([f"isoform:{x}" if "-" in x else f"accession:{x}" for x in batch])
    r = requests.get(f"https://rest.uniprot.org/uniparc/search?fields=accession,sequence&format=tsv&query=({query})&size=500")  # max size is 500
    if r.status_code == 200:
      df = pd.read_csv(io.StringIO(r.text), sep="\t")
      for _, row in df.iterrows():
        for acc in row.UniProtKB.split("; "):
          if acc in batch and acc not in output:
            output[acc] = df.Sequence[0]
            break
    else:
      logger.error(f"[{r.status_code}] {r.text}")
      r.raise_for_status()

  # get missing accessions and try looking for them in uniprotkb
  accessions_missing = [acc for acc in accessions if acc not in output]
  batches = [accessions_missing[i : i + batch_size] for i in range(0, len(accessions_missing), batch_size)]
  for batch in tqdm(batches, desc="Fetching sequences from uniprot.org", total=len(batches)):
    query = " OR ".join([f"accession:{x}" for x in batch])
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/search?fields=accession,sequence&format=tsv&query=({query})&size=500")  # max size is 500
    if r.status_code == 200:
      df = pd.read_csv(io.StringIO(r.text), sep="\t")
      for _, row in df.iterrows():
        if row.Entry in batch and row.Entry not in output:
          output[row.Entry] = df.Sequence[0]
    else:
      logger.error(f"[{r.status_code}] {r.text}")
      r.raise_for_status()

  # check if all accessions are present
  for acc in accessions:
    if acc not in output:
      output[acc] = None
      logger.warning(f"Could not find a sequence for accession: {acc}")

  return output


def fetch_uniprot(uniprot_id: str, head: bool = False) -> Union[str, bool]:
  """
  Fetches a UniProt or UniParc FASTA entry by its identifier.

  This function retrieves a protein sequence in FASTA format using the UniProt REST API.
  If the given UniProt ID is not found in UniProtKB, the function will attempt to fetch
  it from UniParc. The function can either fetch the header information (if `head` is True)
  or the full sequence.

  Args:
      uniprot_id (str): The UniProt or UniParc accession ID for the protein sequence.
      head (bool, optional): If True, performs a HEAD request to check if the entry exists
          without downloading the sequence. Defaults to False.

  Returns:
      Union[str, bool]:
          - If `head` is True: Returns `True` if the ID exists.
          - If `head` is False: Returns the protein sequence as a string if successfully fetched.

  Raises:
      Exception: If the UniProt or UniParc ID is not found in either database.
      ValueError: If the retrieved FASTA contains too many or no sequences.

  Example:
      ```python
      try:
          sequence = fetch_uniprot("P12345")
          print(sequence)
      except Exception as e:
          print(f"Error: {e}")
      ```

  """
  method = requests.head if head else requests.get
  logger.debug(f"Fetching uniprot entry with ID {uniprot_id}")
  r = method(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
  if r.status_code != 200:
    r = method(f"https://rest.uniprot.org/uniparc/{uniprot_id}.fasta")
    if r.status_code != 200:
      raise Exception(
        f'Could not find UniProt accession "{uniprot_id}" in either UniProtKB or UniParc. Please ensure that IDs are correct and refer to actual proteins.'
      )

  if head:
    return True

  _, seqs = read_msa(r.text)
  if len(seqs) > 1:
    print(r.text)
    raise ValueError("Too many sequences returned")
  elif len(seqs) < 1:
    print(r.text)
    raise ValueError("No sequence returned")

  return seqs[0]


def foldseek_search(
  protein: Union["Protein", str],
  mode: str = "3diaa",
  databases: List[str] = None,
  max_retries: int = 10,
  retry_interval: int = 5,
  output_format: str = "json",
) -> Union[str, pd.DataFrame]:
  """Perform a protein structure search using the Foldseek API.

  Parameters:
      protein: Either a Protein object or a path to a PDB file.
      mode: Search mode. Must be on of "3diaa" or "tm-align".
      databases: List of databases to search. Defaults to a predefined list if not provided.
      max_retries: Maximum number of retries to check the job status.
      retry_interval: Time in seconds between retries for checking job status.
      output_format: Format of the output, either "json" or "dataframe".

  Returns:
      Search results in the specified format (JSON string or pandas DataFrame).

  Raises:
      RuntimeError: If the job fails.
      TimeoutError: If the job does not complete within the allotted retries.
      ValueError: If an invalid output_format is specified.

  """

  BASE_URL = "https://search.foldseek.com/api"

  # Default databases to search
  if databases is None:
    databases = ["afdb50", "afdb-swissprot", "afdb-proteome", "bfmd", "cath50", "mgnify_esm30", "pdb100", "gmgcl_id", "bfvd"]

  # Handle file input (Protein object or file path)
  if isinstance(protein, Protein):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
      protein.save(temp_file.name)
      file_path = temp_file.name
  else:
    file_path = protein

  # Submit the job to the Foldseek API
  data = {"mode": mode, "database[]": databases}
  try:
    with open(file_path, "rb") as file:
      files = {"q": file}
      response = requests.post(f"{BASE_URL}/ticket", data=data, files=files)
    response.raise_for_status()
    job_id = response.json()["id"]
  except requests.RequestException as e:
    raise RuntimeError(f"Failed to submit job: {e}")

  # Poll for job status until complete or max retries are reached
  for attempt in range(max_retries):
    try:
      status_response = requests.get(f"{BASE_URL}/ticket/{job_id}")
      status_response.raise_for_status()
      status = status_response.json().get("status", "ERROR")
    except requests.RequestException as e:
      raise RuntimeError(f"Failed to retrieve job status: {e}")

    if status == "COMPLETE":
      break
    elif status == "ERROR":
      raise RuntimeError("Job failed")

    time.sleep(retry_interval)
  else:
    raise TimeoutError(f"Job did not complete within {max_retries * retry_interval} seconds")

  # Retrieve and accumulate results
  results = []
  entry = 0
  while True:
    try:
      result_response = requests.get(f"{BASE_URL}/result/{job_id}/{entry}")
      result_response.raise_for_status()
      result = result_response.json()
    except requests.RequestException as e:
      raise RuntimeError(f"Failed to retrieve results: {e}")

    if not result or all(len(db_result["alignments"]) == 0 for db_result in result["results"]):
      break

    results.append(result)
    entry += 1

  # Clean up temporary file if it was created
  if isinstance(protein, Protein):
    os.remove(file_path)

  # Return results based on the output format
  if output_format == "json":
    return json.dumps(results, indent=2)
  elif output_format == "dataframe":
    rows = []
    for result in results:
      for db_result in result["results"]:
        alignments = db_result["alignments"]
        for alignment in alignments[0]:
          rows.append(
            {
              "target": alignment["target"],
              "db": db_result["db"],
              "seqId": alignment.get("seqId", ""),
              "alnLength": alignment.get("alnLength", ""),
              "missmatches": alignment.get("missmatches", ""),
              "gapsopened": alignment.get("gapsopened", ""),
              "qStartPos": alignment.get("qStartPos", ""),
              "qEndPos": alignment.get("qEndPos", ""),
              "dbStartPos": alignment.get("dbStartPos", ""),
              "dbEndPos": alignment.get("dbEndPos", ""),
              "eval": alignment.get("eval", ""),
              "score": alignment.get("score", ""),
              "qLen": alignment.get("qLen", ""),
              "dbLen": alignment.get("dbLen", ""),
              "seq": alignment.get("tSeq", ""),
            }
          )
    return pd.DataFrame(rows)
  else:
    raise ValueError("Invalid output_format. Choose 'json' or 'dataframe'.")


def run_blast(
  sequence: Union[str, "Protein"],
  email: str,
  matrix: str = "BLOSUM62",
  alignments: int = 250,
  scores: int = 250,
  evalue: float = 10.0,
  filter: bool = False,
  gapalign: bool = True,
  database: str = "uniprotkb_refprotswissprot",
  output_format: Optional[str] = None,
  output_path: Optional[str] = None,
  return_df: bool = True,
) -> Optional[pd.DataFrame]:
  """Submits a BLAST job to the EBI NCBI BLAST web service, checks the status periodically, and retrieves the result.
  The result can be saved either as an XML or FASTA file. Optionally, a DataFrame with alignment details can be returned.

  Parameters:
    sequence: The input amino acid sequence as a string or a Protein object.

      If a Protein object is provided with multiple chains, an error will be raised, and the user will be prompted to provide a single chain sequence using the :obj:`Protein.get_aas` method.
    email: The email address to use for communication if there is a problem.
    matrix: The scoring matrix to use, default is ``"BLOSUM62"``.

      Must be one of::

      ["BLOSUM45", "BLOSUM62", "BLOSUM80", "PAM30", "PAM70"].
    alignments: The number of alignments to display in the result (default is 250). the number alignments must be one of the following::

      [50, 100, 250, 500, 750, 1000]
    scores: The number of scores to display in the result, default is ``250``.
    evalue: The E threshold for alignments (default is 10.0). Must be one of::

      [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    filter: Whether to filter low complexity regions (default is False).
    gapalign: Whether to allow gap alignments (default is True).
    database: The database to search in, default is ``"uniprotkb_refprotswissprot"``.

      Must be one of::

      ["uniprotkb_refprotswissprot", "uniprotkb_pdb", "uniprotkb", "afdb", "uniprotkb_reference_proteomes", "uniprotkb_swissprot", "uniref100", "uniref90", "uniref50", "uniparc"]
    output_format: The format in which to save the result, either ``"xml"`` or ``"fasta"``. If ``None``, which is the default, no file will be saved.
    output_path: The file path to save the output. This is required if `output_format` is specified.
    return_df: Whether to return a DataFrame with alignment details, default is ``True``.

  Returns:
    A pandas DataFrame with BLAST hit and alignment information, if `return_df` is True.

    The DataFrame contains the following columns:
    - "Hit ID": The identifier of the hit sequence.
    - "Accession": The accession number of the hit sequence.
    - "Description": The description of the hit sequence.
    - "Length": The length of the hit sequence.
    - "Score": The score of the alignment.
    - "Bits": The bit score of the alignment.
    - "Expectation": The E-value of the alignment.
    - "Identity (%)": The percentage identity of the alignment.
    - "Gaps": The number of gaps in the alignment.
    - "Query Sequence": The query sequence in the alignment.
    - "Match Sequence": The matched sequence in the alignment.

  Raises:
    AssertionError: If ``sequence`` is provided as a Protein object with multiple chains.
  """

  def parse_xml_to_fasta_and_dataframe(xml_content, job_id, output_format=None, output_path=None, return_df=True):
    """Parses the XML content, saves it as a FASTA file, or returns a DataFrame if requested."""
    root = ET.fromstring(xml_content)
    hits = []
    fasta_content = ""

    for hit in root.findall(".//{http://www.ebi.ac.uk/schema}hit"):
      hit_id = hit.attrib["id"]
      hit_ac = hit.attrib["ac"]
      hit_description = hit.attrib["description"]
      hit_length = hit.attrib["length"]

      for alignment in hit.findall(".//{http://www.ebi.ac.uk/schema}alignment"):
        score = alignment.find("{http://www.ebi.ac.uk/schema}score").text
        bits = alignment.find("{http://www.ebi.ac.uk/schema}bits").text
        expectation = alignment.find("{http://www.ebi.ac.uk/schema}expectation").text
        identity = alignment.find("{http://www.ebi.ac.uk/schema}identity").text
        gaps = alignment.find("{http://www.ebi.ac.uk/schema}gaps").text
        query_seq = alignment.find("{http://www.ebi.ac.uk/schema}querySeq").text
        match_seq = alignment.find("{http://www.ebi.ac.uk/schema}matchSeq").text

        fasta_content += (
          f">{hit_id} | Accession: {hit_ac} | Description: {hit_description} | "
          f"Length: {hit_length} | Score: {score} | Bits: {bits} | "
          f"Expectation: {expectation} | Identity: {identity}% | Gaps: {gaps}\n"
          f"{match_seq}\n\n"
        )

        hits.append(
          {
            "Hit ID": hit_id,
            "Accession": hit_ac,
            "Description": hit_description,
            "Length": hit_length,
            "Score": score,
            "Bits": bits,
            "Expectation": expectation,
            "Identity (%)": identity,
            "Gaps": gaps,
            "Query Sequence": query_seq,
            "Match Sequence": match_seq,
          }
        )

    # Step 4: Save or return results
    if output_format == "fasta" and output_path:
      with open(output_path, "w") as fasta_file:
        fasta_file.write(fasta_content)
      print(f"FASTA result saved as {output_path}")

    if return_df:
      df = pd.DataFrame(hits)
      return df

  valid_databases = [
    "uniprotkb_refprotswissprot",
    "uniprotkb_pdb",
    "uniprotkb",
    "afdb",
    "uniprotkb_reference_proteomes",
    "uniprotkb_swissprot",
    "uniref100",
    "uniref90",
    "uniref50",
    "uniparc",
  ]
  if database not in valid_databases:
    raise ValueError(f"Database must be one of the following {valid_databases}")

  valid_matrices = ["BLOSUM45", "BLOSUM62", "BLOSUM80", "PAM30", "PAM70"]
  if matrix not in valid_matrices:
    raise ValueError(f"Matrix must be one of the following {valid_matrices}")

  valid_evalues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
  if evalue not in valid_evalues:
    raise ValueError(f"E-threshold must be one of the following {valid_evalues}")

  # the api is very delicate, we need specific parameters
  if evalue > 1:
    evalue = int(evalue)

  valid_alignments = [50, 100, 250, 500, 750, 1000]
  if alignments not in valid_alignments:
    raise ValueError(f"Alignments must be one of the following {valid_alignments}")

  valid_output_formats = ["xml", "fasta", None]
  if output_format not in valid_output_formats:
    raise ValueError(f"Output format must be one of the following {valid_output_formats}")

  # Handle Protein object input
  if isinstance(sequence, Protein):
    if len(sequence.chains()) > 1:
      raise AssertionError("The protein has multiple chains. Use '.get_aas(chain)' to obtain the sequence for a specific chain.")

    chain = sequence.chains()[0]
    sequence = sequence.get_aas(chain)

  # Step 1: Submit the BLAST job
  url = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run"

  multipart_data = MultipartEncoder(
    fields={
      "email": email,
      "program": "blastp",
      "matrix": matrix,
      "alignments": str(alignments),
      "scores": str(scores),
      "exp": str(evalue),
      "filter": "T" if filter else "F",
      "gapalign": str(gapalign).lower(),
      "stype": "protein",
      "sequence": sequence,
      "database": database,
    }
  )

  headers = {
    "User-Agent": USER_AGENT,
    "Accept": "text/plain,application/json",
    "Accept-Language": "en-US,en;q=0.5",
    "Content-Type": multipart_data.content_type,
  }

  response = requests.post(url, headers=headers, data=multipart_data)

  if response.status_code == 200:
    job_id = response.text.strip()
    print(f"Job submitted successfully. Job ID: {job_id}")
  else:
    response.raise_for_status()

  # Step 2: Check job status
  status_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/status/{job_id}"

  while True:
    status_response = requests.get(status_url)
    status = status_response.text.strip()

    if status_response.status_code == 200:
      print(f"Job status: {status}")
      if status == "FINISHED":
        break
      elif status in ["RUNNING", "PENDING", "QUEUED"]:
        time.sleep(20)
      else:
        raise Exception(f"Job failed with status: {status}")
    else:
      status_response.raise_for_status()

  # Step 3: Retrieve XML result
  xml_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/{job_id}/xml"
  xml_response = requests.get(xml_url)

  if xml_response.status_code == 200:
    xml_content = xml_response.text
    # Save XML if output format is XML
    if output_format == "xml" and output_path:
      with open(output_path, "w") as xml_file:
        xml_file.write(xml_content)
      print(f"XML result saved as {output_path}")
    elif output_format == "fasta" and output_path:
      return parse_xml_to_fasta_and_dataframe(xml_content, job_id, output_format, output_path, return_df)
    elif return_df:
      return parse_xml_to_fasta_and_dataframe(xml_content, job_id, output_format, output_path, return_df)
  else:
    xml_response.raise_for_status()


def plot_pseudo_3D(
  xyz: Union[np.ndarray, pd.DataFrame],
  c: np.ndarray = None,
  ax: matplotlib.axes.Axes = None,
  chainbreak: int = 5,
  Ls: Optional[List] = None,
  cmap: str = "gist_rainbow",
  line_w: float = 2.0,
  cmin: Optional[float] = None,
  cmax: Optional[float] = None,
  zmin: Optional[float] = None,
  zmax: Optional[float] = None,
  shadow: float = 0.95,
) -> matplotlib.collections.LineCollection:
  """Plot the famous Pseudo 3D projection of a protein.

  Algorithm originally written By Dr. Sergey Ovchinnikov.
  Adapted from https://github.com/sokrypton/ColabDesign/blob/16e03c23f2a30a3dcb1775ac25e107424f9f7352/colabdesign/shared/plot.py

  Parameters:
    xyz: XYZ coordinates of the protein
    c: 1D array of all the values to use to color the protein, defaults to residue index
    ax: Matplotlib axes object to add the figure to
    chainbreak: Minimum distance in angstroms between chains / segments before being considered a chain break (int)
    Ls: Allows handling multiple chains or segments by providing the lengths of each chain, ensuring that chains are visualized separately without unwanted connections
    cmap: Matplotlib color map to use for coloring the protein
    line_w: Line width
    cmin: Minimum value for coloring, automatically calculated if None
    cmax: Maximum value for coloring, automatically calculated if None
    zmin: Minimum z coordinate values, automatically calculated if None
    zmax: Maximum z coordinate values, automatically calculated if None
    shadow: Shadow intensity between 0 and 1 inclusive, lower numbers mean darker more intense shadows

  Returns:
    LineCollection object of what's been drawn

  """

  def rescale(a, amin=None, amax=None):
    a = np.copy(a)
    if amin is None:
      amin = a.min()
    if amax is None:
      amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin) / (amax - amin)

  # clip color values and produce warning if necesarry
  if c is not None and cmin is not None and cmax is not None:
    if np.any(c < cmin):
      logger.warn(f"The provided c colors array contains values that are less than cmin ({cmin}). Out of range values will be clipped into range.")
    if np.any(c > cmax):
      logger.warn(f"The provided c colors array contains values that are greater than cmax ({cmax}). Out of range values will be clipped into range.")
    c = np.clip(c, a_min=cmin, a_max=cmax)

  # make segments and colors for each segment
  xyz = np.asarray(xyz)
  if Ls is None:
    seg = np.concatenate([xyz[:, None], np.roll(xyz, 1, 0)[:, None]], axis=1)
    c_seg = np.arange(len(seg))[::-1] if c is None else (c + np.roll(c, 1, 0)) / 2
  else:
    Ln = 0
    seg = []
    c_seg = []
    for L in Ls:
      sub_xyz = xyz[Ln : Ln + L]
      seg.append(np.concatenate([sub_xyz[:, None], np.roll(sub_xyz, 1, 0)[:, None]], axis=1))
      if c is not None:
        sub_c = c[Ln : Ln + L]
        c_seg.append((sub_c + np.roll(sub_c, 1, 0)) / 2)
      Ln += L
    seg = np.concatenate(seg, 0)
    c_seg = np.arange(len(seg))[::-1] if c is None else np.concatenate(c_seg, 0)

  # set colors
  c_seg = rescale(c_seg, cmin, cmax)
  if isinstance(cmap, str):
    if cmap == "gist_rainbow":
      c_seg *= 0.75
    colors = matplotlib.colormaps[cmap](c_seg)
  else:
    colors = cmap(c_seg)

  # remove segments that aren't connected
  seg_len = np.sqrt(np.square(seg[:, 0] - seg[:, 1]).sum(-1))
  if chainbreak is not None:
    idx = seg_len < chainbreak
    seg = seg[idx]
    seg_len = seg_len[idx]
    colors = colors[idx]

  seg_mid = seg.mean(1)
  seg_xy = seg[..., :2]
  seg_z = seg[..., 2].mean(-1)
  order = seg_z.argsort()

  # add shade/tint based on z-dimension
  z = rescale(seg_z, zmin, zmax)[:, None]

  # add shadow (make lines darker if they are behind other lines)
  seg_len_cutoff = (seg_len[:, None] + seg_len[None, :]) / 2
  seg_mid_z = seg_mid[:, 2]
  seg_mid_dist = np.sqrt(np.square(seg_mid[:, None] - seg_mid[None, :]).sum(-1))
  shadow_mask = sigmoid(seg_len_cutoff * 2.0 - seg_mid_dist) * (seg_mid_z[:, None] < seg_mid_z[None, :])
  np.fill_diagonal(shadow_mask, 0.0)
  shadow_mask = shadow ** shadow_mask.sum(-1, keepdims=True)

  seg_mid_xz = seg_mid[:, :2]
  seg_mid_xydist = np.sqrt(np.square(seg_mid_xz[:, None] - seg_mid_xz[None, :]).sum(-1))
  tint_mask = sigmoid(seg_len_cutoff / 2 - seg_mid_xydist) * (seg_mid_z[:, None] < seg_mid_z[None, :])
  np.fill_diagonal(tint_mask, 0.0)
  tint_mask = 1 - tint_mask.max(-1, keepdims=True)

  colors[:, :3] = colors[:, :3] + (1 - colors[:, :3]) * (0.50 * z + 0.50 * tint_mask) / 3
  colors[:, :3] = colors[:, :3] * (0.20 + 0.25 * z + 0.55 * shadow_mask)

  set_lim = False
  if ax is None:
    fig, ax = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    set_lim = True
  else:
    fig = ax.get_figure()
    if ax.get_xlim() == (0, 1):
      set_lim = True

  if set_lim:
    xy_min = xyz[:, :2].min() - line_w
    xy_max = xyz[:, :2].max() + line_w
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

  ax.set_aspect("equal")
  ax.set_xlabel("Distance (Å)", fontsize=12)
  ax.set_ylabel("Distance (Å)", fontsize=12)

  # determine linewidths
  width = fig.bbox_inches.width * ax.get_position().width
  linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

  lines = mcoll.LineCollection(
    seg_xy[order], colors=colors[order], linewidths=linewidths, path_effects=[matplotlib.patheffects.Stroke(capstyle="round")]
  )
  return ax.add_collection(lines)


def animate_pseudo_3D(
  fig: matplotlib.figure.Figure,
  ax: matplotlib.axes.Axes,
  frames: matplotlib.collections.LineCollection,
  titles: Union[str, List[str]] = "Protein Animation",
  interval: int = 200,
  repeat_delay: int = 0,
  repeat: bool = True,
) -> matplotlib.animation.ArtistAnimation:
  """Animate multiple Pseudo 3D LineCollection objects.

  Parameters:
    fig: Matplotlib figure that contains all the frames
    ax: Matplotlib axes for the figure that contains all the frames
    frames: List of LineCollection objects
    titles: Single title or list of titles corresponding to each frame
    interval: Delay between frames in milliseconds
    repeat_delay: The delay in milliseconds between consecutive animation runs, if repeat is True
    repeat: Whether the animation repeats when the sequence of frames is completed

  Returns:
    Animation of all the different frames

  """
  # check titles
  if isinstance(titles, str):
    titles = [f"{titles} ({i + 1}/{len(frames)})" for i in range(len(frames))]
  elif len(titles) == len(frames):
    ValueError(f"The number of titles ({len(titles)}) does not match the number of frames ({len(frames)}).")

  if isinstance(titles, str):
    titles = [f"{titles} ({i + 1}/{len(frames)})" for i in range(len(frames))]
  elif len(titles) == len(frames):
    ValueError(f"The number of titles ({len(titles)}) does not match the number of frames ({len(frames)}).")

  # Gather all vertices safely across multiple paths per frame
  all_x = np.concatenate([path.vertices[:, 0] for frame in frames for path in frame.get_paths()])
  all_y = np.concatenate([path.vertices[:, 1] for frame in frames for path in frame.get_paths()])
  # Calculate limits with optional padding
  x_min, x_max = all_x.min(), all_x.max()
  y_min, y_max = all_y.min(), all_y.max()
  # Apply padding to the limits
  padding = 4
  ax.set_xlim(x_min - padding, x_max + padding)
  ax.set_ylim(y_min - padding, y_max + padding)
  # disable axes
  ax.axis("off")

  def init():
    # hide all frames
    for frame in frames:
      frame.set_visible(False)
    # Initialize the plot with the first frame
    collection = frames[0]
    ax.add_collection(collection)
    ax.set_title(titles[0])
    return (collection,)

  def animate(i):
    frames[i - 1].set_visible(False)
    frames[i].set_visible(True)
    ax.add_collection(frames[i])
    ax.set_title(titles[i])
    return (frames[i],)

  ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames), interval=interval, repeat_delay=repeat_delay, repeat=repeat)
  return ani
