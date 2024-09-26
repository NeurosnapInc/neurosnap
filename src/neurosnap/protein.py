"""
Provides functions and classes related to processing protein data as well as
a feature rich wrapper around protein structures using BioPython.
"""
import io
import os
import tempfile

import matplotlib
import matplotlib.animation as animation
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBIO, PDBParser, PPBuilder
from Bio.PDB.Superimposer import Superimposer
from matplotlib import collections as mcoll
from scipy.special import expit as sigmoid

from neurosnap.log import logger

### CONSTANTS ###
# Codes for both RNA and DNA
STANDARD_NUCLEOTIDES = {'A', 'T', 'C', 'G', 'U', 'DA', 'DT', 'DC', 'DG', 'DU'}
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

  def __repr__(self):
    return f"<Neurosnap Protein: Models={self.models()}, Chains=[{', '.join(self.chains())}], Atoms={len(self.df)}>"

  def __call__(self, model = None, chain = None, res_type = None):
    """
    -------------------------------------------------------
    Returns a selection of a copy of the internal dataframe
    that matches the provided query. If no queries are
    provided, will return a copy of the internal dataframe.
    -------------------------------------------------------
    Parameters:
      model...: If provided, returned atoms must match this model (int)
      chain...: If provided, returned atoms must match this chain (int)
      res_type: If provided, returned atoms must match this res_type (int)
    Returns:
      df: Copy of the internal dataframe that matches the input query (pandas.core.frame.DataFrame)
    """
    df = self.df.copy()
    if model is not None:
      df = df.loc[df.model == model]
    if chain is not None:
      df = df.loc[df.chain == chain]
    if res_type is not None:
      df = df.loc[df.res_type == res_type]
    return df

  def __sub__(self, other_protein):
    """
    -------------------------------------------------------
    Automatically calculate the RMSD of two proteins.
    Model used will naively be the first models that have
    identical backbone shapes.
    Essentially just wraps around self.calculate_rmsd()
    -------------------------------------------------------
    Parameters:
      other_protein: Another Protein object to compare against (Protein)
    Returns:
      df: Copy of the internal dataframe that matches the input query (pandas.core.frame.DataFrame)
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
    
    assert model1 is not None, ValueError("Could not find any matching matching models to calculate RMSD for. Please ensure at least two models with matching backbone shapes are provided.")
    return self.calculate_rmsd(other_protein, model1=model1, model2=model2)

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
    Ligands, small molecules, and nucleotides are ignored.
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

  def renumber(self, model=None, chain=None, start=1):
    """
    -------------------------------------------------------
    Renumbers all selected residues. If selection does not
    exist this function will do absolutely nothing.
    -------------------------------------------------------
    Parameters:
      model: The model ID to renumber, if not provided will use all models (int)
      chain: The chain ID to renumber, if not provided will use all chains (str)
      start: Starting value to increment from, default 1 (int)
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

  def remove_waters(self):
    """
    -------------------------------------------------------
    Removes all water molecules (residues named 'WAT' or 'HOH') 
    from the structure. It is suggested to call .renumber()
    afterwards as well.
    -------------------------------------------------------
    """
    for model in self.structure:
      for chain in model:
        # Identify water molecules and mark them for removal
        residues_to_remove = [res for res in chain if res.get_resname() in ['WAT', 'HOH']]

        # Remove water molecules
        for res in residues_to_remove:
          chain.detach_child(res.id)

  def remove_non_biopolymers(self, model=None, chain=None):
    """
    -------------------------------------------------------
    Removes all ligands, heteroatoms, and non-biopolymer
    residues from the selected structure. Non-biopolymer
    residues are considered to be any residues that are not
    standard amino acids or standard nucleotides (DNA/RNA).
    If no model or chain is provided, it will remove from
    the entire structure.
    -------------------------------------------------------
    Parameters:
      model: The model ID to process, if not provided will use all models (int)
      chain: The chain ID to process, if not provided will use all chains (str)
    -------------------------------------------------------
    """
    # List of standard amino acids and nucleotides (biopolymer residues)
    biopolymer_residues = set(AA_ABR_TO_CODE.keys()).union(STANDARD_NUCLEOTIDES)
    
    for m in self.structure:
      if model is None or m.id == model:
        for c in m:
          if chain is None or c.id == chain:
            # Identify non-biopolymer residues (ligands, heteroatoms, etc.)
            residues_to_remove = [
              res for res in c
              if res.get_resname() not in biopolymer_residues
            ]

            # Remove non-biopolymer residues
            for res in residues_to_remove:
              c.detach_child(res.id)

  def get_backbone(self, model=None, chain=None):
    """
    -------------------------------------------------------
    Extract backbone atoms (N, CA, C) from the structure.
    If model or chain is not provided, extracts from all models/chains.
    -------------------------------------------------------
    Parameters:
      model: Model ID to extract from, if not provided, all models are included (int)
      chain: Chain ID to extract from, if not provided, all chains are included (str)
    Returns:
      backbone: A numpy array of backbone coordinates (Nx3) (numpy.ndarray)
    """
    backbone_atoms = ["N", "CA", "C"]
    backbone_coords = []
    
    for m in self.structure:
      if model is None or m.id == model:
        for c in m:
          if chain is None or c.id == chain:
            for res in c:
              for atom in res:
                if atom.name in backbone_atoms:
                  backbone_coords.append(atom.coord)

    return np.array(backbone_coords)

  def find_disulfide_bonds(self, threshold=2.05):
    """
    -------------------------------------------------------
    Find disulfide bonds between Cysteine residues in the structure.
    Looks for SG-SG bonds within a threshold distance (default 2.05 Å).
    -------------------------------------------------------
    Parameters:
      threshold: Maximum distance to consider a bond between SG atoms (float)
    Returns:
      disulfide_pairs: List of tuples of residue pairs forming disulfide bonds (list<tuple>)
    -------------------------------------------------------
    """
    disulfide_pairs = []
    
    for model in self.structure:
      for chain in model:
        cysteines = [res for res in chain if res.get_resname() == 'CYS']
        for i, res1 in enumerate(cysteines):
          for res2 in cysteines[i+1:]:
            try:
              sg1 = res1['SG']
              sg2 = res2['SG']
              distance = sg1 - sg2
              if distance < threshold:
                disulfide_pairs.append((res1, res2))
            except KeyError:
              pass  # Skip if no SG atom found
    return disulfide_pairs

  def calculate_rmsd(self, other_protein, model1=0, model2=0, chain1=None, chain2=None, align=True):
    """
    -------------------------------------------------------
    Calculate RMSD between the current structure and another protein.
    Only compares backbone atoms (N, CA, C). RMSD is in angstroms (Å).
    -------------------------------------------------------
    Parameters:
      other_protein: Another Protein object to compare against (Protein)
      model1......: Model ID of original protein to compare (int)
      model2......: Model ID of other protein to compare (int)
      chain1......: Chain ID of original protein, if not provided compares all chains (str)
      chain2......: Chain ID of other protein, if not provided compares all chains (str)
      align.......: Whether to align the structures first using Superimposer (bool)
    Returns:
      rmsd: The root-mean-square deviation between the two structures (float)
    -------------------------------------------------------
    """
    backbone_atoms = ["N", "CA", "C"]
    # ensure models are present
    assert model1 in self.models(), ValueError(f"Model {model1} was not found in current protein.")
    assert model2 in other_protein.models(), ValueError(f"Model {model2} was not found in other protein.")

    # Get backbone coordinates of both structures
    backbone1 = self.get_backbone(model=model1, chain=chain1)
    backbone2 = other_protein.get_backbone(model=model2, chain=chain2)
    assert backbone1.shape == backbone2.shape, "Structures must have the same number of backbone atoms for RMSD calculation."

    # Use the Superimposer to align the structures
    if align:
      def aux(sample_model):
        atoms = []
        for sample_chain in sample_model:
          for res in sample_chain:
            for atom in res:
              if atom.name in backbone_atoms:
                atoms.append(atom)
        return atoms

      sup = Superimposer()
      sup.set_atoms(aux(self.structure[model1]), aux(other_protein.structure[model2]))
      sup.apply(other_protein.structure)  # Apply the transformation to the other protein

    # Get new backbone coordinates of both structures
    backbone1 = self.get_backbone(model=model1, chain=chain1)
    backbone2 = other_protein.get_backbone(model=model2, chain=chain2)
    
    diff = backbone1 - backbone2
    rmsd = np.sqrt(np.sum(diff ** 2) / backbone1.shape[0])
    return rmsd

  def calculate_distance_matrix(self, model=None, chain=None):
    """
    -------------------------------------------------------
    Calculate the distance matrix for all alpha-carbon (CA) atoms in the chain.
    Useful for creating contact maps or proximity analyses.
    -------------------------------------------------------
    Parameters:
      model: The model ID to calculate the distance matrix for, default 0 (int)
      chain: The chain ID to calculate, if not provided calculates for all chains (str)
    Returns:
      dist_matrix: A 2D numpy array representing the distance matrix (numpy.ndarray)
    -------------------------------------------------------
    """
    ca_atoms = []

    for m in self.structure:
      if model is None or m.id == model:
        for c in m:
          if chain is None or c.id == chain:
            for res in c:
              if 'CA' in res:
                ca_atoms.append(res['CA'].coord)

    ca_atoms = np.array(ca_atoms)
    dist_matrix = np.sqrt(np.sum((ca_atoms[:, np.newaxis] - ca_atoms[np.newaxis, :]) ** 2, axis=-1))
    return dist_matrix

  def find_missing_residues(self):
    """
    -------------------------------------------------------
    Identify missing residues in the structure based on residue numbering.
    Useful for identifying gaps in the structure.
    -------------------------------------------------------
    Parameters:
      chain: The chain ID to inspect, if not provided inspects all chains (str)
    Returns:
      missing_residues: List of missing residue positions (list<int>)
    -------------------------------------------------------
    """
    missing_residues = []
    
    for model in self.structure:
      for chain in model:
        residues = sorted(res.id[1] for res in chain)
        for i in range(len(residues) - 1):
          if residues[i+1] != residues[i] + 1:
            missing_residues.extend(range(residues[i] + 1, residues[i+1]))
    
    return missing_residues

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


def plot_pseudo_3D(xyz, c=None, ax=None, chainbreak=5, Ls=None, cmap="gist_rainbow", line_w=2.0, cmin=None, cmax=None, zmin=None, zmax=None, shadow=0.95):
  """
  -------------------------------------------------------
  Plot the famous Pseudo 3D projection of a protein.
  Algorithm originally written By Dr. Sergey Ovchinnikov.
  Adapted from https://github.com/sokrypton/ColabDesign/blob/16e03c23f2a30a3dcb1775ac25e107424f9f7352/colabdesign/shared/plot.py
  -------------------------------------------------------
  Parameters:
    xyz.......: XYZ coordinates of the protein (numpy.ndarray|pandas.core.frame.DataFrame)
    c.........: 1D array of all the values to use to color the protein, defaults to residue index (numpy.ndarray)
    ax........: Matplotlib axes object to add the figure to (matplotlib.axes._axes.Axes)
    chainbreak: Minimum distance in angstroms between chains / segments before being considered a chain break (int)
    Ls........: Allows handling multiple chains or segments by providing the lengths of each chain, ensuring that chains are visualized separately without unwanted connections (list)
    cmap......: Matplotlib color map to use for coloring the protein (str)
    line_w....: Line width (float)
    cmin......: Minimum value for coloring, automatically calculated if None (float)
    cmax......: Maximum value for coloring, automatically calculated if None (float)
    zmin......: Minimum z coordinate values, automatically calculated if None (float)
    zmax......: Maximum z coordinate values, automatically calculated if None (float)
    shadow....: Shadow intensity between 0 and 1 inclusive, lower numbers mean darker more intense shadows (float)
  Returns:
    lc: LineCollection object of whats been drawn (matplotlib.collections.LineCollection)
  """
  def rescale(a, amin=None, amax=None):
    a = np.copy(a)
    if amin is None:
      amin = a.min()
    if amax is None:
      amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin)/(amax - amin)

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
    seg = np.concatenate([xyz[:,None],np.roll(xyz,1,0)[:,None]],axis=1)
    c_seg = np.arange(len(seg))[::-1] if c is None else (c + np.roll(c,1,0))/2
  else:
    Ln = 0
    seg = []
    c_seg = []
    for L in Ls:
      sub_xyz = xyz[Ln:Ln+L]
      seg.append(np.concatenate([sub_xyz[:,None],np.roll(sub_xyz,1,0)[:,None]],axis=1))
      if c is not None:
        sub_c = c[Ln:Ln+L]
        c_seg.append((sub_c + np.roll(sub_c,1,0))/2)
      Ln += L
    seg = np.concatenate(seg,0)
    c_seg = np.arange(len(seg))[::-1] if c is None else np.concatenate(c_seg,0)
  
  # set colors
  c_seg = rescale(c_seg,cmin,cmax)  
  if isinstance(cmap, str):
    if cmap == "gist_rainbow": 
      c_seg *= 0.75
    colors = matplotlib.colormaps[cmap](c_seg)
  else:
    colors = cmap(c_seg)
  
  # remove segments that aren't connected
  seg_len = np.sqrt(np.square(seg[:,0] - seg[:,1]).sum(-1))
  if chainbreak is not None:
    idx = seg_len < chainbreak
    seg = seg[idx]
    seg_len = seg_len[idx]
    colors = colors[idx]

  seg_mid = seg.mean(1)
  seg_xy = seg[...,:2]
  seg_z = seg[...,2].mean(-1)
  order = seg_z.argsort()

  # add shade/tint based on z-dimension
  z = rescale(seg_z,zmin,zmax)[:,None]

  # add shadow (make lines darker if they are behind other lines)
  seg_len_cutoff = (seg_len[:,None] + seg_len[None,:]) / 2
  seg_mid_z = seg_mid[:,2]
  seg_mid_dist = np.sqrt(np.square(seg_mid[:,None] - seg_mid[None,:]).sum(-1))
  shadow_mask = sigmoid(seg_len_cutoff * 2.0 - seg_mid_dist) * (seg_mid_z[:,None] < seg_mid_z[None,:])
  np.fill_diagonal(shadow_mask,0.0)
  shadow_mask = shadow ** shadow_mask.sum(-1,keepdims=True)

  seg_mid_xz = seg_mid[:,:2]
  seg_mid_xydist = np.sqrt(np.square(seg_mid_xz[:,None] - seg_mid_xz[None,:]).sum(-1))
  tint_mask = sigmoid(seg_len_cutoff/2 - seg_mid_xydist) * (seg_mid_z[:,None] < seg_mid_z[None,:])
  np.fill_diagonal(tint_mask,0.0)
  tint_mask = 1 - tint_mask.max(-1,keepdims=True)

  colors[:,:3] = colors[:,:3] + (1 - colors[:,:3]) * (0.50 * z + 0.50 * tint_mask) / 3
  colors[:,:3] = colors[:,:3] * (0.20 + 0.25 * z + 0.55 * shadow_mask)

  set_lim = False
  if ax is None:
    fig, ax = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    set_lim = True
  else:
    fig = ax.get_figure()
    if ax.get_xlim() == (0,1):
      set_lim = True
      
  if set_lim:
    xy_min = xyz[:,:2].min() - line_w
    xy_max = xyz[:,:2].max() + line_w
    ax.set_xlim(xy_min,xy_max)
    ax.set_ylim(xy_min,xy_max)

  ax.set_aspect("equal")
    
  # determine linewidths
  width = fig.bbox_inches.width * ax.get_position().width
  linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

  lines = mcoll.LineCollection(seg_xy[order], colors=colors[order], linewidths=linewidths, path_effects=[matplotlib.patheffects.Stroke(capstyle="round")])
  return ax.add_collection(lines)


def animate_pseudo_3D(fig, frames, interval=200, repeat_delay=0, repeat=True):
  """
  -------------------------------------------------------
  Animate multiple Pseudo 3D LineCollection objects.
  -------------------------------------------------------
  Parameters:
    fig.........: Matplotlib figure that contains all the frames (matplotlib.figure.Figure)
    frames......: List of LineCollection objects (matplotlib.collections.LineCollection)
    interval....: Delay between frames in milliseconds (int)
    repeat_delay: The delay in milliseconds between consecutive animation runs, if repeat is True (int)
    repeat......: Whether the animation repeats when the sequence of frames is completed (bool)
  Returns:
    ani: Animation of all the different frames (matplotlib.animation.ArtistAnimation)
  """
  frames = [[frame] for frame in frames]
  ani = animation.ArtistAnimation(fig, frames, interval=interval, repeat_delay=repeat_delay, repeat=repeat, blit=True)
  return ani