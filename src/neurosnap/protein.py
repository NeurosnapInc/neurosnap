"""
Provides functions and classes related to processing protein data as well as
a feature rich wrapper around protein structures using BioPython.
"""

import io
import json
import os
import pathlib
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBIO, MMCIFParser, NeighborSearch, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue as ResidueType
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Superimposer import Superimposer
from rdkit import Chem
from requests_toolbelt.multipart.encoder import MultipartEncoder
from tqdm import tqdm

from neurosnap.api import USER_AGENT
from neurosnap.constants import (
  AA_ALIASES,
  AA_RECORDS,
  AA_WEIGHTS_PROTEIN_AVG,
  BACKBONE_ATOMS_AA,
  BACKBONE_ATOMS_DNA,
  BACKBONE_ATOMS_RNA,
  DEFAULT_PKA,
  HYDROPHOBIC_RESIDUES,
  NUC_DNA_CODES,
  NUC_RNA_CODES,
  STANDARD_NUCLEOTIDES,
  AARecord,
  STANDARD_AAs,
)
from neurosnap.log import logger
from neurosnap.msa import read_msa


### CLASSES ###
class Protein:
  def __init__(self, pdb: Union[str, pathlib.Path, io.IOBase], format: str = "auto"):
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
    elif isinstance(pdb, (str, pathlib.Path)):
      pdb = str(pdb)  # converts pathlib paths to string
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
            if res.id[0] == " ":
              if res.resname in STANDARD_NUCLEOTIDES:
                res_type = "NUCLEOTIDE"
              else:
                try:
                  getAA(res.resname, non_standard="allow")
                  res_type = "AMINO_ACID"
                except:
                  pass
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
      try:
        seq += getAA(resn).code
      except:
        pass

    return seq

  def select_residues(self, selectors: str, invert: bool = False, model: Optional[int] = None) -> Dict[str, List[int]]:
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
        invert: Whether to invert the selection.
        model: The ID of the model to select from. If None, the first model is used.

    Returns:
        dict: A dictionary where keys are chain IDs and values are sorted
              lists of residue sequence numbers that match the query.

    Raises:
        ValueError: If a specified chain or residue in the selector does not exist in the structure.
        ValueError: If selector string is empty.
    """
    if model is None:
      model = self.models().pop(0)
    elif model not in self.structure:
      raise ValueError(f'Protein does not contain model "{model}"')

    # prepare model dataframe
    dfm = self.df[self.df["model"] == model]

    # get chains and create output object
    chains = self.chains(model)
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

    assert selectors, "Provided selectors string is empty"

    # get selection
    for selector in selectors.split(","):
      # get and validate chain
      chain = selector[0]
      if chain not in chains:
        raise ValueError(f'Chain "{chain}" in selector "{selector}" does not exist in the specified structure.')

      # if select entire chain
      if len(selector) == 1:
        output[chain] = output[chain].union(dfm[(dfm["chain"] == chain)]["res_id"].to_list())
        continue

      # if select single residue
      found = pattern_res_single.search(selector)
      if found:
        resi = int(found.group(1))
        if dfm[(dfm["chain"] == chain) & (dfm["res_id"] == resi)].empty:
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
          if dfm[(dfm["chain"] == chain) & (dfm["res_id"] == resi)].empty:
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

    if invert:
      output_inverted = {}
      for chain in chains:
        avail_resis = set(dfm[(dfm["chain"] == chain)]["res_id"].to_list())
        if chain in output:  # chain is in output so get all residues not specified
          diff_resis = avail_resis - set(output[chain])
          if diff_resis:
            output_inverted[chain] = diff_resis
        else:  # chain is not in output so the entire chain is selected
          output_inverted[chain] = avail_resis
      output = output_inverted

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
    biopolymer_residues = set(AA_RECORDS.keys()).union(STANDARD_NUCLEOTIDES)
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

  def center(
    self,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    model: Optional[int] = 0,
    chains: Optional[List[str]] = None,
  ):
    """Translate the structure so its center of mass matches the target coordinates.

    Parameters:
      x: Target x-coordinate to center on, defaults to 0.0
      y: Target y-coordinate to center on, defaults to 0.0
      z: Target z-coordinate to center on, defaults to 0.0
      model: Model ID to center. If ``None``, all models are centered individually.
      chains: Optional list of chain IDs to center. If ``None``, use all chains in the model.

    """
    target = np.array([x, y, z], dtype=float)
    if model is None:
      models_to_center = self.models()
    else:
      assert model in self.models(), f"Model {model} was not found in current protein."
      models_to_center = [model]

    for model_id in models_to_center:
      com = self.calculate_center_of_mass(model=model_id, chains=chains)
      translation = target - com
      for chain in self.structure[model_id]:
        if chains is None or chain.id in chains:
          for residue in chain:
            for atom in residue:
              atom.coord = atom.coord + translation
    # regenerate dataframe to reflect updated coordinates
    self.generate_df()

  def get_backbone(
    self,
    chains: Optional[Tuple[List[str], Set[str]]] = None,
    model: Optional[int] = None,
  ) -> np.ndarray:
    """Extract backbone atoms (N, CA, C) from the structure.
    If model or chain is not provided, extracts from all models/chains.

    Parameters:
      chains: Chain ID(s) to extract from. If ``None``, all chains are included.
      model: Model ID to extract from. If ``None``, all models are included.

    Returns:
      A numpy array of backbone coordinates (Nx3)

    """
    backbone_coords = []

    for m in self.structure:
      if model is None or m.id == model:
        for c in m:
          if chains is None or c.id in chains:
            for res in c:
              for atom in res:
                if atom.name in BACKBONE_ATOMS_AA:
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

  def find_interface_contacts(
    self, chain1: str, chain2: str, *, model: Optional[int] = None, cutoff: float = 4.5, hydrogens: bool = True
  ) -> List[Tuple[Atom, Atom]]:
    """
    Identify interface atoms between two chains using a distance cutoff.

    Parameters:
        chain1: ID of the binder chain.
        chain2: ID of the target chain.
        cutoff: Distance cutoff in Ångströms for defining an interface contact (default 4.5 Å).
        hydrogens: Whether to keep hydrogen atoms when evaluating contacts (set False to exclude them).

    Returns:
        List[Tuple[Atom, Atom]]: Paired atoms from the binder and target chains that are within the cutoff distance.
    """
    # Default to the first model if model is None
    if model is None:
      model = self.models()[0]

    assert chain1 in self.chains(model), f"Chain {chain1} was not found."
    assert chain2 in self.chains(model), f"Chain {chain2} was not found."

    chain1 = self.structure[model][chain1]
    chain2 = self.structure[model][chain2]
    atoms1 = [a for a in chain1.get_atoms() if hydrogens or a.element != "H"]
    atoms2 = [a for a in chain2.get_atoms() if hydrogens or a.element != "H"]

    return find_contacts(atoms1, atoms2, cutoff=cutoff)

  def find_interface_residues(
    self, chain1: str, chain2: str, *, model: Optional[int] = None, cutoff: float = 4.5, hydrogens: bool = True
  ) -> List[Tuple[ResidueType, ResidueType]]:
    """
    Identify residue-residue contacts between two chains using a distance cutoff.

    A residue pair is included when any atom in the binder residue is within `cutoff`
    Å of any atom in the target residue.

    Parameters:
        chain1: ID of the binder chain.
        chain2: ID of the target chain.
        model: Index of the model to use (defaults to the first).
        cutoff: Distance cutoff in Ångströms for defining an interface contact (default 4.5 Å).
        hydrogens: Whether to keep hydrogen atoms when evaluating contacts (set False to exclude them).

    Returns:
        List[Tuple[Residue, Residue]]: Unique residue pairs (binder residue, target residue) that are in contact.
    """
    # Default to the first model if model is None
    if model is None:
      model = self.models()[0]

    assert chain1 in self.chains(model), f"Chain {chain1} was not found."
    assert chain2 in self.chains(model), f"Chain {chain2} was not found."

    chain1_obj = self.structure[model][chain1]
    chain2_obj = self.structure[model][chain2]
    atoms1 = [a for a in chain1_obj.get_atoms() if hydrogens or a.element != "H"]
    atoms2 = [a for a in chain2_obj.get_atoms() if hydrogens or a.element != "H"]

    residue_pairs: List[Tuple[ResidueType, ResidueType]] = []
    seen = set()
    for atom1, atom2 in find_contacts(atoms1, atoms2, cutoff=cutoff):
      res1, res2 = atom1.get_parent(), atom2.get_parent()
      key = (res1.get_parent().id, res1.id, res2.get_parent().id, res2.id)
      if key in seen:
        continue
      seen.add(key)
      residue_pairs.append((res1, res2))

    return residue_pairs

  def find_non_interface_hydrophobic_patches(
    self,
    chain_pairs: Iterable[Tuple[str, str]],
    target_chains: Optional[Iterable[str]] = None,
    *,
    model: Optional[int] = None,
    cutoff_interface: float = 4.5,
    hydrogens: bool = True,
    patch_cutoff: float = 6.0,
    min_patch_area: float = 40.0,
  ) -> List[Set[ResidueType]]:
    """
    Identify solvent-exposed hydrophobic patches that are not part of specified interfaces.

    Parameters:
        chain_pairs: Iterable of chain ID pairs that define interface contacts (e.g. [('A', 'B')]).
        target_chains: Optional iterable of chain IDs to restrict the patch search to. If None, all chains are considered.
        model: Model index to use. Defaults to the first available model.
        cutoff_interface: Atom–atom distance cutoff (Å) for defining an interface contact.
        hydrogens: Whether to include hydrogens when identifying interface contacts.
        patch_cutoff: CA–CA distance cutoff (Å) for linking hydrophobic residues into the same patch.
        min_patch_area: Minimum total solvent-accessible surface area (Å²) for a patch to be counted.

    Returns:
        List of patches, where each patch is a set of residues belonging to that solvent-exposed, non-interface hydrophobic cluster.
    """
    if model is None:
      model = self.models()[0]
    assert model in self.structure, f"Model {model} was not found."

    chain_pairs = [(c1.strip(), c2.strip()) for c1, c2 in chain_pairs]
    available_chains = set(self.chains(model))
    for chain_a, chain_b in chain_pairs:
      assert chain_a in available_chains, f"Chain {chain_a} was not found."
      assert chain_b in available_chains, f"Chain {chain_b} was not found."

    chain_set = None
    if target_chains is not None:
      chain_set = {c.strip() for c in target_chains}
      missing = chain_set - available_chains
      assert not missing, f"Chain(s) {', '.join(sorted(missing))} were not found."

    # Compute interface residues using the same logic as find_interface_contacts.
    interface_res: Set[Tuple[str, int]] = set()
    for chainA, chainB in chain_pairs:
      contacts = self.find_interface_contacts(chainA, chainB, model=model, cutoff=cutoff_interface, hydrogens=hydrogens)
      for atomA, atomB in contacts:
        resA = atomA.get_parent()
        resB = atomB.get_parent()
        interface_res.add((resA.get_parent().id.strip(), resA.id[1]))
        interface_res.add((resB.get_parent().id.strip(), resB.id[1]))

    # Compute per-residue SASA and build a lookup.
    from Bio.PDB import SASA

    structure_model = self.structure[model]
    sasa_calculator = SASA.ShrakeRupley()
    sasa_calculator.compute(structure_model, level="R")
    per_res_sasa = {}
    for chain in structure_model:
      for res in chain:
        if not hasattr(res, "sasa"):
          continue
        per_res_sasa[(chain.id.strip(), res.id[1])] = float(res.sasa or 0.0)

    # Collect solvent-exposed hydrophobic residues not at the interface.
    hydrophobic_atoms: List[Atom] = []
    hydrophobic_res_keys: List[Tuple[str, int]] = []
    hydrophobic_res_objs: List[ResidueType] = []
    for chain in structure_model:
      chain_id = chain.id.strip()
      if chain_set is not None and chain_id not in chain_set:
        continue
      for res in chain:
        if res.id[0] != " ":
          continue
        if res.get_resname() not in HYDROPHOBIC_RESIDUES:
          continue
        resid_key = (chain_id, res.id[1])
        if resid_key in interface_res:
          continue
        if per_res_sasa.get(resid_key, 0.0) <= 0.01:
          continue
        if "CA" not in res:
          continue
        hydrophobic_atoms.append(res["CA"])
        hydrophobic_res_keys.append(resid_key)
        hydrophobic_res_objs.append(res)

    # Build adjacency between hydrophobic residues based on CA distance.
    coords = np.asarray([atom.coord for atom in hydrophobic_atoms])
    if not len(coords):
      return []

    n = len(coords)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
      for j in range(i + 1, n):
        if np.linalg.norm(coords[i] - coords[j]) <= patch_cutoff:
          neighbors[i].append(j)
          neighbors[j].append(i)

    # Find connected components and collect qualifying patches.
    patches: List[Set[ResidueType]] = []
    visited = [False] * n
    for idx in range(n):
      if visited[idx]:
        continue
      stack = [idx]
      component = []
      while stack:
        cur = stack.pop()
        if visited[cur]:
          continue
        visited[cur] = True
        component.append(cur)
        stack.extend(neighbors[cur])

      if len(component) <= 1:
        continue

      comp_area = 0.0
      for comp_idx in component:
        resid_key = hydrophobic_res_keys[comp_idx]
        comp_area += per_res_sasa.get(resid_key, 0.0)
      if comp_area >= min_patch_area:
        patches.append({hydrophobic_res_objs[i] for i in component})

    return patches

  def align(self, other_protein: "Protein", chains1: List[str] = [], chains2: List[str] = [], model1: int = 0, model2: int = 0):
    """Align another Protein object's structure to the self.structure
    of the current object. The other Protein will be transformed
    and aligned. Uses protein and nucleotide backbone atoms.

    Parameters:
      other_protein: Another Neurosnap Protein object to compare against
      chains1: The chain(s) you want to include in the alignment within the reference protein, set to an empty list to use all chains.
      chains2: The chain(s) you want to include in the alignment within the other protein, set to an empty list to use all chains.
      model1: Model ID of reference protein to align to
      model2: Model ID of other protein to transform and align to reference

    """
    assert model1 in self.models(), "Specified model needs to be present in the reference structure."
    assert model2 in other_protein.models(), "Specified model needs to be present in the other structure."
    # validate chains1
    avail_chains = self.chains(model1)
    if chains1:
      for chain in chains1:
        assert chain in avail_chains, f"Chain {chain} was not found in the reference protein. Found chains include {', '.join(avail_chains)}."
    else:
      chains1 = avail_chains
    # validate chains2
    avail_chains = other_protein.chains(model2)
    if chains2:
      for chain in chains2:
        assert chain in avail_chains, f"Chain {chain} was not found in the other protein. Found chains include {', '.join(avail_chains)}."
    else:
      chains2 = avail_chains

    # Use the Superimposer to align the structures
    def _allowed_backbone_atoms(residue) -> Optional[Set[str]]:
      """Return the backbone atom names to consider for the residue."""
      if is_aa(residue, standard=False):
        return BACKBONE_ATOMS_AA
      resname = residue.get_resname().strip().upper()
      if resname in NUC_DNA_CODES:
        return BACKBONE_ATOMS_DNA
      if resname in NUC_RNA_CODES:
        return BACKBONE_ATOMS_RNA
      if resname in STANDARD_NUCLEOTIDES:
        # Fall back to RNA atoms; these cover the shared sugar/phosphate backbone.
        return BACKBONE_ATOMS_RNA
      return None

    def aux_get_atom_map(sample_model, chains):
      if chains:
        chain_order = list(chains)
      else:
        chain_order = [chain.id for chain in sample_model if chain.id.strip()]
      chain_lookup = {chain.id: chain for chain in sample_model}
      atoms = {}
      for chain_id in chain_order:
        chain = chain_lookup.get(chain_id)
        if chain is None:
          continue
        for residue in chain:
          backbone_atoms = _allowed_backbone_atoms(residue)
          if not backbone_atoms:
            continue
          het_flag, seq_id, icode = residue.id
          resid_key = (chain_id, het_flag, seq_id, (icode or "").strip())
          for atom in residue:
            if atom.name in backbone_atoms:
              atoms[(resid_key, atom.name)] = atom
      return atoms

    ref_atom_map = aux_get_atom_map(self.structure[model1], chains1)
    mov_atom_map = aux_get_atom_map(other_protein.structure[model2], chains2)
    assert ref_atom_map, "Reference protein does not contain any backbone atoms to align."
    assert mov_atom_map, "Other protein does not contain any backbone atoms to align."

    common_keys = sorted(ref_atom_map.keys() & mov_atom_map.keys())
    assert common_keys, "Proteins do not share common backbone atoms to align."
    if len(common_keys) != len(ref_atom_map) or len(common_keys) != len(mov_atom_map):
      missing_ref = sorted(k for k in ref_atom_map.keys() - mov_atom_map.keys())
      missing_mov = sorted(k for k in mov_atom_map.keys() - ref_atom_map.keys())
      raise AssertionError(
        f"Backbone atom mismatch between structures. Reference-only atoms: {len(missing_ref)}, other-only atoms: {len(missing_mov)}."
      )

    ref_atoms = [ref_atom_map[key] for key in common_keys]
    mov_atoms = [mov_atom_map[key] for key in common_keys]
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mov_atoms)
    sup.apply(other_protein.structure[model2])  # Apply the transformation to the other protein
    # update the pandas dataframe
    other_protein.generate_df()

  def calculate_rmsd(
    self,
    other_protein: "Protein",
    chains1: Optional[Tuple[List[str], Set[str]]] = None,
    chains2: Optional[Tuple[List[str], Set[str]]] = None,
    model1: int = 0,
    model2: int = 0,
    align: bool = True,
  ) -> float:
    """Calculate RMSD between the current structure and another protein.
    Only compares backbone atoms (N, CA, C). RMSD is in angstroms (Å).

    Parameters:
      other_protein: Another Protein object to compare against
      chains1: Chain ID of original protein, if not provided compares all chains
      chains2: Chain ID of other protein, if not provided compares all chains
      model1: Model ID of original protein to compare
      model2: Model ID of other protein to compare
      align: Whether to align the structures first using Superimposer

    Returns:
      The root-mean-square deviation between the two structures

    """
    # ensure models are present
    assert model1 in self.models(), f"Model {model1} was not found in current protein."
    assert model2 in other_protein.models(), f"Model {model2} was not found in other protein."

    # Get backbone coordinates of both structures
    backbone1 = self.get_backbone(chains=chains1, model=model1)
    backbone2 = other_protein.get_backbone(chains=chains2, model=model2)
    assert backbone1.shape == backbone2.shape, "Structures must have the same number of backbone atoms for RMSD calculation."

    if align:
      self.align(other_protein, model1=model1, model2=model2, chains1=chains1, chains2=chains2)

    # Get new backbone coordinates of both structures
    backbone1 = self.get_backbone(chains=chains1, model=model1)
    backbone2 = other_protein.get_backbone(chains=chains2, model=model2)

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
      center_of_mass: A numpy array with shape (3,) representing the center of mass (x, y, z)

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

  def distances_from_com(self, model: Optional[int] = 0, chains: Optional[List[str]] = None, com: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate the distances of all atoms from the center of mass (COM) of the protein.

    This method computes the Euclidean distance between the coordinates of each atom
    and the center of mass of the structure. The center of mass is calculated for the
    specified model and chain, or for all models and chains if none are provided.

    Parameters:
      model: The model ID to calculate for. If not provided, defaults to 0.
      chains: List of chain IDs to calculate for. If not provided, calculates for all chains.
      com: Center of mass (com) to use, must be a numpy array with shape (3,) representing the center of mass (x, y, z). Set to None to calculate the com automatically

    Returns:
      A 1D NumPy array containing the distances (in Ångströms) between each atom and the center of mass.

    """
    if com is not None:
      assert len(com) == 3, "Input center of mass must be a numpy array with shape (3,) representing the center of mass (x, y, z)."
    else:
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

  def calculate_rog(self, model: Optional[int] = 0, chains: Optional[List[str]] = None, distances_from_com: Optional[np.ndarray] = None) -> float:
    """Calculate the radius of gyration (Rg) of the protein.

    The radius of gyration measures the overall size and compactness of the protein structure.
    It is calculated based on the distances of atoms from the center of mass (COM).

    Parameters:
      model: The model ID to calculate for. If not provided, defaults to 0.
      chains: List of chain IDs to calculate for. If not provided, calculates for all chains.
      distances_from_com: A 1D NumPy array containing the distances (in Ångströms) between each atom and the center of mass. Set to None to automatically calculate this.

    Returns:
      The radius of gyration (Rg) in Ångströms. Returns 0.0 if no atoms are found.

    """
    if distances_from_com is not None:
      distances_sq = distances_from_com**2
    else:
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
    return float(total_sasa)

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
    chain: Optional[str] = None,
    chain_other: Optional[str] = None,
    *,
    model: Optional[int] = None,
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

  def calculate_interface_hydrogen_bonding_residues(
    self,
    chain: Optional[str] = None,
    chain_other: Optional[str] = None,
    *,
    model: Optional[int] = None,
    donor_acceptor_cutoff: float = 3.5,
    angle_cutoff: float = 120.0,
  ) -> int:
    """
    Count the number of unique residues that participate in hydrogen bonds at an interface.

    A residue is considered hydrogen-bonding if at least one of its atoms participates in a
    hydrogen bond that satisfies both:
      - donor–acceptor distance <= donor_acceptor_cutoff (Å)
      - donor–H–acceptor angle >= angle_cutoff (degrees)

    Hydrogen atoms must be explicitly present; add missing hydrogens with a tool like 'reduce'.

    If 'chain_other' is None:
      - Hydrogen bonds are evaluated within 'chain' (intra-chain), or across all chains if 'chain' is None.
    If 'chain_other' is provided:
      - Only inter-chain hydrogen bonds between 'chain' and 'chain_other' are considered.
      - If 'chain_other' is provided but 'chain' is not, an exception is raised.

    Parameters:
      model: Model index to evaluate. If None, defaults to the first model.
      chain: Primary chain ID to evaluate. If None and 'chain_other' is also None, all chains are considered.
      chain_other: Secondary chain ID for inter-chain evaluation. If None, intra-chain bonds are considered.
      donor_acceptor_cutoff: Maximum donor–acceptor distance (Å). Default 3.5.
      angle_cutoff: Minimum donor–H–acceptor angle (degrees). Default 120.0.

    Returns:
      Integer count of unique residues (from all evaluated chains) that participate in at least one hydrogen bond.

    Raises:
      ValueError if 'chain_other' is provided but 'chain' is not, or if a specified chain does not exist.
    """
    hydrogen_distance_cutoff = 1.2  # Typical donor–H bond length (Å)

    # Default to the first model if model is None
    if model is None:
      model = self.models()[0]
    model_obj = self.structure[model]

    # Input validation
    if chain_other is not None and chain is None:
      raise ValueError("`chain_other` is specified, but `chain` is not. Provide both for inter-chain evaluation.")
    if chain is not None and chain not in self.chains():
      raise ValueError(f"Chain {chain} does not exist within the input structure.")
    if chain_other is not None and chain_other not in self.chains():
      raise ValueError(f"Chain {chain_other} does not exist within the input structure.")

    # Helper to create a stable residue identifier
    def resid_tuple(res):
      # res.get_id() returns a tuple like (' ', seq_id, insertion_code)
      return (res.get_parent().id, res.get_id())

    hb_residues = set()  # set of (chain_id, res_id_tuple)

    # Determine which chains to iterate as "donor side"
    donor_chains = [chain] if chain is not None else [c.id for c in model_obj]

    for donor_chain_id in donor_chains:
      donor_chain = model_obj[donor_chain_id]

      # Determine which chains to search as "acceptor side"
      if chain_other:
        acceptor_chain_ids = [chain_other]
      else:
        # Intra-chain if chain is specified; else across all chains (including same chain)
        acceptor_chain_ids = [donor_chain_id] if chain is not None else [c.id for c in model_obj]

      for donor_res in donor_chain:
        for donor_atom in donor_res:
          # Only N/O donors
          if getattr(donor_atom, "element", None) not in ("N", "O"):
            continue

          # Collect explicit hydrogens bonded to this donor
          bonded_hydrogens = [
            atom
            for atom in donor_res
            if getattr(atom, "element", None) == "H" and np.linalg.norm(donor_atom.coord - atom.coord) <= hydrogen_distance_cutoff
          ]
          if not bonded_hydrogens:
            continue

          # Scan acceptor chains
          for acc_chain_id in acceptor_chain_ids:
            acc_chain = model_obj[acc_chain_id]

            # If inter-chain only, skip same-chain pairs
            if chain_other and acc_chain_id == donor_chain_id:
              continue

            for acceptor_res in acc_chain:
              for acceptor_atom in acceptor_res:
                # Only N/O acceptors and not the same atom
                if getattr(acceptor_atom, "element", None) not in ("N", "O"):
                  continue
                if acceptor_atom is donor_atom:
                  continue

                # Distance check (donor–acceptor)
                da_dist = np.linalg.norm(donor_atom.coord - acceptor_atom.coord)
                if da_dist > donor_acceptor_cutoff:
                  continue

                # Angle check for any bonded hydrogen
                for hydrogen in bonded_hydrogens:
                  v_dh = hydrogen.coord - donor_atom.coord
                  v_da = acceptor_atom.coord - donor_atom.coord
                  # Guard against zero-length vectors
                  if np.linalg.norm(v_dh) == 0 or np.linalg.norm(v_da) == 0:
                    continue

                  cos_theta = np.dot(v_dh, v_da) / (np.linalg.norm(v_dh) * np.linalg.norm(v_da))
                  # Numerical stability
                  cos_theta = max(min(cos_theta, 1.0), -1.0)
                  angle = np.degrees(np.arccos(cos_theta))

                  if angle >= angle_cutoff:
                    hb_residues.add(resid_tuple(donor_res))
                    hb_residues.add(resid_tuple(acceptor_res))
                    # Once accepted for this acceptor/donor pair, no need to keep checking more Hs
                    break

    return len(hb_residues)

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
      # TODO: Known bug where multi-model PDB files result in all models being written as 0 (biopython sucks)
    elif format == "mmcif":
      mmcif_io = MMCIFIO()
      mmcif_io.set_structure(self.structure)
      mmcif_io.save(fpath)
    else:
      raise ValueError("Format must be 'pdb' or 'mmcif'.")


### FUNCTIONS ###
def getAA(query: str, *, non_standard: str = "reject") -> AARecord:
  """Resolve an amino acid identifier to a canonical record.

  This function accepts either a **1-letter code**, **3-letter abbreviation**,
  or **full name** (case-insensitive) and returns the corresponding `AARecord`.

  Parameters
  ----------
  query : str
      Amino acid identifier (1-letter code, 3-letter CCD abbreviation,
      or full name).
  non_standard : {"reject", "convert", "allow"}, optional
      Policy for handling non-standard amino acids (default: "reject"):

      - "reject": Raise an error if the amino acid is non-standard.
      - "convert": Map non-standard amino acids to their closest
        standard equivalent (e.g., MSE → MET).
      - "allow": Return the non-standard amino acid unchanged.

  Returns
  -------
  AARecord
      A record containing:
      - `code`: 1-letter code (may be "?" if unavailable for non-standard AAs).
      - `abr`: 3-letter abbreviation.
      - `name`: Full amino acid name.
      - `is_standard`: Whether the residue is one of the 20 canonical amino acids.
      - `standard_equiv_abr`: 3-letter abbreviation of the standard equivalent
        (if applicable).

  Raises
  ------
  ValueError
      If `query` does not match any supported amino acid identifier.
      If `non_standard="reject"` and the amino acid is non-standard.
      If `non_standard="convert"` but no standard equivalent is defined.
  """
  query = query.upper()
  try:
    abr = AA_ALIASES[query]
    rec = AA_RECORDS[abr]
  except KeyError:
    raise ValueError(f"Unknown amino acid identifier: '{query}'. Expected a 1-letter code, 3-letter code, or full name.")

  if not rec.is_standard:
    if non_standard == "reject":
      raise ValueError(
        f"Encountered non-standard amino acid '{rec.abr}' ({rec.name}). "
        "To handle these, set `non_standard='allow'` to keep them "
        "or `non_standard='convert'` to map them to a standard equivalent."
      )
    elif non_standard == "convert":
      if not rec.standard_equiv_abr:
        raise ValueError(f"Non-standard amino acid '{rec.abr}' ({rec.name}) does not have a standard equivalent and cannot be converted.")
      rec = AA_RECORDS[rec.standard_equiv_abr]
  return rec


def sanitize_aa_seq(seq: str, *, non_standard: str = "reject", trim_term: bool = True, uppercase=True, clean_whitespace: bool = True) -> str:
  """
  Validates and sanitizes an amino acid sequence string.

  Parameters:
      seq: The input amino acid sequence.
      non_standard: How to handle non-standard amino acids.
          Must be one of:
          - "reject": Raise an error if any non-standard residue is found (default).
          - "convert": Replace non-standard residues with standard equivalents, if possible.
          - "allow": Keep non-standard residues unchanged.
      trim_term: If True, trims terminal stop codons ("*") from the end of the sequence. Default is True.
      uppercase: If True, converts the sequence to uppercase before processing. Default is True.
      clean_whitespace: If True, removes all whitespace characters from the sequence. Default is True.

  Returns:
      The sanitized amino acid sequence.

  Raises:
      ValueError: If an invalid residue is found and `non_standard` is set to "reject",
                  or if a residue cannot be converted when `non_standard` is "convert".
      AssertionError: If `non_standard` is not one of "allow", "convert", or "reject".
  """
  assert non_standard in ("allow", "convert", "reject"), f'Unknown value of "{non_standard}" supplied for non_standard parameter.'

  if uppercase:
    seq = seq.upper()

  if clean_whitespace:
    seq = re.sub(r"\s", "", seq)

  if trim_term:
    seq = seq.rstrip("*")

  new_seq = ""
  for i, x in enumerate(seq, start=1):
    if x not in STANDARD_AAs:
      if non_standard == "allow":
        pass
      elif non_standard == "convert":
        x = getAA(x, non_standard="convert").code
      else:
        raise ValueError(f'Invalid amino acid "{x}" specified at position {i}.')
    new_seq += x
  return new_seq


def molecular_weight(sequence: str, aa_mws: Dict[str, float] = AA_WEIGHTS_PROTEIN_AVG) -> float:
  """
  Calculate the molecular weight of a protein or peptide sequence.

  This function computes the molecular weight by summing the residue
  masses for each amino acid in the input sequence. By default, it uses
  average amino acid residue masses (`AA_WEIGHTS_PROTEIN_AVG`), but you
  can provide a custom mass dictionary (e.g., monoisotopic or free amino
  acid masses).

  The calculation accounts for the loss of one water molecule (H₂O,
  18.015 Da) for each peptide bond formed. For a sequence of length n,
  (n - 1) * 18.015 Da is subtracted from the total.

  Args:
      sequence: Amino acid sequence (one-letter codes).
      aa_mws: Dictionary mapping amino acid one-letter codes to molecular
          weights. Defaults to `AA_WEIGHTS_PROTEIN_AVG`.

  Returns:
      Estimated molecular weight of the protein or peptide in Daltons (Da).

  Raises:
      ValueError: If the sequence contains an invalid or unsupported
      amino acid code.

  Notes:
      - Use `AA_WEIGHTS_PROTEIN_MONO` for monoisotopic mass calculations,
        typically used in mass spectrometry.
      - Use `AA_WEIGHTS_PROTEIN_AVG` (default) for average residue masses,
        appropriate for bulk molecular weight estimation.
      - For free amino acids (not incorporated in peptides), use
        `AA_WEIGHTS_FREE`.
      - Weight dictionaries are defined in `constants.py`.
  """
  # Remove whitespace and convert to uppercase
  sequence = sequence.strip().upper()

  # Sum molecular weights
  weight = 0.0
  for aa in sequence:
    if aa not in aa_mws:
      raise ValueError(f"Invalid amino acid: {aa}")
    weight += aa_mws[aa]

  # Adjust for water loss during peptide bond formation
  # Each peptide bond loses one H2O (18.015 Da)
  if len(sequence) > 1:
    weight -= (len(sequence) - 1) * 18.015

  return weight


def _fraction_protonated_basic(pH: float, pKa: float) -> float:
  """For BH+ <-> B + H+, returns fraction in the protonated (+1) form."""
  return 1.0 / (1.0 + 10.0 ** (pH - pKa))


def _fraction_deprotonated_acidic(pH: float, pKa: float) -> float:
  """For HA <-> A- + H+, returns fraction in the deprotonated (-1) form."""
  return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def net_charge(sequence: str, pH: float, pKa: Dict[str, float] = DEFAULT_PKA) -> float:
  """
  Calculate the net charge of a protein or peptide sequence at a given pH.

  This function applies the Henderson–Hasselbalch equation to estimate
  the protonation state of titratable groups (N-terminus, C-terminus,
  and ionizable side chains) and computes the overall net charge.

  Args:
      sequence: Amino acid sequence in one-letter codes. Supports the 20
          canonical residues and optionally 'U' (selenocysteine).
          Non-ionizable residues are ignored.
      pH: The solution pH at which to evaluate the net charge.
      pKa: Dictionary of pKa values for titratable groups. Must include
          keys "N_TERMINUS", "C_TERMINUS", "D", "E", "C", "Y",
          "H", "K", and "R". If 'U' is present in the sequence, it
          should also include "U".

  Returns:
      Estimated net charge of the sequence at the given pH.

  Notes:
      Positive charges come from protonated groups:
      - N-terminus
      - Lysine (K)
      - Arginine (R)
      - Histidine (H)

      Negative charges come from deprotonated groups:
      - C-terminus
      - Aspartic acid (D)
      - Glutamic acid (E)
      - Cysteine (C)
      - Tyrosine (Y)
      - Selenocysteine (U), if included

      The calculation assumes independent ionization equilibria and does
      not account for local environment or structural effects. It is best
      interpreted as an approximate charge profile.
  """
  seq = sequence.strip().upper()
  if not seq:
    return 0.0

  counts = Counter(seq)

  # N-terminus (+1 when protonated)
  nterm = _fraction_protonated_basic(pH, pKa["N_TERMINUS"])

  # C-terminus (-1 when deprotonated)
  cterm = _fraction_deprotonated_acidic(pH, pKa["C_TERMINUS"])

  # Side chains
  pos = (
    counts.get("K", 0) * _fraction_protonated_basic(pH, pKa["K"])
    + counts.get("R", 0) * _fraction_protonated_basic(pH, pKa["R"])
    + counts.get("H", 0) * _fraction_protonated_basic(pH, pKa["H"])
  )

  neg = (
    counts.get("D", 0) * _fraction_deprotonated_acidic(pH, pKa["D"])
    + counts.get("E", 0) * _fraction_deprotonated_acidic(pH, pKa["E"])
    + counts.get("C", 0) * _fraction_deprotonated_acidic(pH, pKa["C"])
    + counts.get("Y", 0) * _fraction_deprotonated_acidic(pH, pKa["Y"])
    + counts.get("U", 0) * _fraction_deprotonated_acidic(pH, pKa["U"])  # optional
  )

  return (nterm + pos) - (cterm + neg)


def isoelectric_point(
  sequence: str, pKa: Dict[str, float] = DEFAULT_PKA, *, pH_low: float = 0.0, pH_high: float = 14.0, tol: float = 1e-4, max_iter: int = 100
) -> float:
  """
  Estimate the isoelectric point (pI) of a protein or peptide.

  The pI is the pH at which the net charge of the molecule is zero.
  This function computes the net charge across pH and uses a bisection
  search to find the root.

  Args:
      sequence: Amino acid sequence (one-letter codes). Supports the 20
          canonical residues and optionally 'U' (selenocysteine).
          Non-titratable residues contribute no charge.
      pKa: Dictionary of pKa values for titratable groups. Must include
          keys "N_TERMINUS", "C_TERMINUS", and for side chains
          "D", "E", "C", "Y", "H", "K", "R". If 'U' appears in the
          sequence, include "U" (default ~5.2, approximate).
      pH_low: Lower bound of the bracketing interval for the bisection
          search (default 0.0).
      pH_high: Upper bound of the bracketing interval for the bisection
          search (default 14.0).
      tol: Target absolute net charge tolerance at the solution
          (default 1e-4).
      max_iter: Maximum iterations for the bisection search (default 100).

  Returns:
      Estimated pI.

  Notes:
      - Results depend on the chosen pKa set. For consistency with common
        tools, you may substitute a different pKa dictionary
        (e.g., Bjellqvist or IPC sets).
      - Pyrrolysine ('O') is not included by default due to scarce
        consensus pKa data; it is treated as non-titratable here.
        You can add an entry if you have a value.
      - This model ignores sequence-context and microenvironment effects
        (local shifts in pKa due to neighbors or structure). It’s a good
        heuristic, not a guarantee.
  """
  seq = sequence.strip().upper()
  if not seq:
    return 0.0

  # Validate sequence characters
  valid = STANDARD_AAs | {"U", "O"}
  for aa in seq:
    if aa not in valid:
      raise ValueError(f"Invalid amino acid: {aa}")

  # Bisection search
  lo, hi = pH_low, pH_high
  q_lo = net_charge(seq, lo, pKa)
  q_hi = net_charge(seq, hi, pKa)

  # If the bracket doesn't change sign, still proceed but clamp toward the side
  # where |charge| is smaller to avoid errors on unusual sequences.
  if q_lo == 0:
    return lo
  if q_hi == 0:
    return hi
  if q_lo * q_hi > 0:
    # No sign change; do a guarded search by nudging bounds inward.
    # This keeps the function robust for edge cases (e.g., extremely acidic/basic sequences).
    for _ in range(20):
      mid = (lo + hi) / 2.0
      q_mid = net_charge(seq, mid, pKa)
      if abs(q_mid) < abs(q_lo) and abs(q_mid) < abs(q_hi):
        # Use the best we have if we can't bracket
        best = mid
      lo += 0.1
      hi -= 0.1
    return best if "best" in locals() else (lo + hi) / 2.0

  for _ in range(max_iter):
    mid = (lo + hi) / 2.0
    q_mid = net_charge(seq, mid, pKa)

    if abs(q_mid) <= tol:
      return mid
    # Decide which subinterval keeps the sign change
    if q_lo * q_mid < 0:
      hi, q_hi = mid, q_mid
    else:
      lo, q_lo = mid, q_mid

  # Fallback if not converged within max_iter
  return (lo + hi) / 2.0


def calculate_bsa(
  protein_complex: Union[str, "Protein"], chain_group_1: List[str], chain_group_2: List[str], *, model: int = 0, level: str = "R"
) -> float:
  """Calculate the buried surface area (BSA) between two groups of chains.

  The buried surface area is computed as the difference in solvent-accessible
  surface area (SASA) when the two groups of chains are in complex versus separate.
  The BSA is defined as:
    BSA = (SASA(group 1) + SASA(group 2)) − SASA(complex)

  Parameters:
    complex: Complex to calculate BSA for.
    chain_group_1: List of chain IDs for the first group.
    chain_group_2: List of chain IDs for the second group.
    model: Model ID to calculate BSA for, defaults to 0.
    level: The level at which ASA values are assigned, which can be one of "A" (Atom), "R" (Residue), "C" (Chain), "M" (Model), or "S" (Structure).

  Returns:
    The buried surface area (BSA) in Å².
  """
  # Validate input
  if isinstance(protein_complex, str):
    protein_complex = Protein(protein_complex)
  assert model in protein_complex.models(), f"Model {model} not found."
  all_chains = set(protein_complex.chains(model))
  assert len(chain_group_1) > 0 and len(chain_group_2) > 0, "Chain groups cannot be empty."
  assert set(chain_group_1).isdisjoint(chain_group_2), "Chain groups must not overlap."
  assert set(chain_group_1).union(set(chain_group_2)) == all_chains, "Chain groups must cover all chains."

  sasa_complex = protein_complex.calculate_surface_area(model=model, level=level)

  with tempfile.TemporaryDirectory() as tmp_dir:
    complex_pdb = os.path.join(tmp_dir, "complex.pdb")
    protein_complex.save(complex_pdb)

    prot_group1 = Protein(complex_pdb)
    for chain in chain_group_2:
      prot_group1.remove(model=model, chain=chain)

    prot_group2 = Protein(complex_pdb)
    for chain in chain_group_1:
      prot_group2.remove(model=model, chain=chain)

    sasa_group1 = prot_group1.calculate_surface_area(model=model, level=level)
    sasa_group2 = prot_group2.calculate_surface_area(model=model, level=level)

  return (sasa_group1 + sasa_group2) - sasa_complex


def find_contacts(atoms1: List[Atom], atoms2: List[Atom], cutoff: float = 4.5) -> List[Tuple[Atom, Atom]]:
  """
  Identifies atomic contacts between two sets of atoms within a distance threshold.

  Parameters:
      atoms1: A list of Bio.PDB.Atom objects from the binder chain.
      atoms2: A list of Bio.PDB.Atom objects from the target chain.
      cutoff: Distance cutoff (in Å) to consider a contact (default is 4.5 Å).

  Returns:
      List[Tuple[Atom, Atom]]: Atom pairs (binder atom, target atom) within the cutoff.
  """
  ns = NeighborSearch(atoms2)
  contacts: List[Tuple[Atom, Atom]] = []
  contacts_append = contacts.append
  for atom in atoms1:
    for neighbor in ns.search(atom.coord, cutoff, level="A"):
      contacts_append((atom, neighbor))
  return contacts


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
  biopolymer_keywords = set(AA_RECORDS.keys()).union(STANDARD_NUCLEOTIDES)
  biopolymer_keywords.remove("UNK")

  def is_biopolymer(molecule):
    """
    Determines if a molecule is a biopolymer (protein or nucleotide) based on specific characteristics.
    Returns True if it is a biopolymer; False otherwise.
    """
    # Check for peptide bonds or nucleotide backbones
    # Simplified logic: exclude molecules with standard amino acids or nucleotide bases
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
      logger.warning(f"Skipping fragment {i} due to processing failure (frag is None).")
      continue

    # Check if the fragment is a biopolymer
    if is_biopolymer(frag):
      logger.info(f"Skipping biopolymer fragment {i}.")
      continue

    try:
      # Add hydrogens and sanitize molecule
      Chem.SanitizeMol(frag)
    except Exception as e:
      raise ValueError(f"Failed to sanitize fragment {i}: {e}")

    # Skip small molecules based on atom count
    if frag.GetNumAtoms() < min_atoms:
      logger.info(f"Skipping small molecule fragment {i} (atom count: {frag.GetNumAtoms()}).")
      continue

    # Save fragment to SDF
    sdf_file = os.path.join(output_dir, f"mol_{molecule_count}.sdf")
    writer = Chem.SDWriter(sdf_file)
    writer.write(frag)
    writer.close()
    molecule_count += 1

  logger.info(f"Extracted {molecule_count - 1} non-biopolymer molecules to {output_dir}.")


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
            output[acc] = row.Sequence
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
          output[row.Entry] = row.Sequence
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

  seqs = [seq for _, seq in read_msa(r.text)]
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
