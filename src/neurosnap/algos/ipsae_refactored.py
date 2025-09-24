# Originally written by https://github.com/DunbrackLab/IPSAE
# Heavily modified in this work to make easier to use
#
# Script for calculating the ipSAE score for scoring pairwise protein-protein interactions
# in AlphaFold2 and AlphaFold3 models
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1
#
# Also calculates:
#    pDockQ: Bryant, Pozotti, and Eloffson. https://www.nature.com/articles/s41467-022-28865-w
#    pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
#    LIS: Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def ptm_func(x: np.ndarray | float, d0: float) -> np.ndarray | float:
  return 1.0 / (1.0 + (x / d0) ** 2.0)


ptm_func_vec = np.vectorize(ptm_func)


def calc_d0(L: float, pair_type: str) -> float:
  L = float(max(L, 27.0))
  min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
  d0 = 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8
  return max(min_value, d0)


def calc_d0_array(L: np.ndarray, pair_type: str) -> np.ndarray:
  L = np.asarray(L, dtype=float)
  L = np.maximum(27.0, L)
  min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
  return np.maximum(min_value, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def contiguous_ranges(numbers: set[int]) -> str:
  if not numbers:
    return ""
  s = sorted(numbers)
  out = []
  a = b = s[0]
  for x in s[1:]:
    if x == b + 1:
      b = x
    else:
      out.append(f"{a}" if a == b else f"{a}-{b}")
      a = b = x
  out.append(f"{a}" if a == b else f"{a}-{b}")
  return "+".join(out)


def init_pairdict_scalar(chains: np.ndarray, init_val: float = 0.0) -> Dict[str, Dict[str, float]]:
  uniq = np.unique(chains)
  return {c1: {c2: init_val for c2 in uniq if c2 != c1} for c1 in uniq}


def init_pairdict_array(chains: np.ndarray, size: int) -> Dict[str, Dict[str, np.ndarray]]:
  uniq = np.unique(chains)
  return {c1: {c2: np.zeros(size, dtype=float) for c2 in uniq if c2 != c1} for c1 in uniq}


def init_pairdict_set(chains: np.ndarray) -> Dict[str, Dict[str, set]]:
  uniq = np.unique(chains)
  return {c1: {c2: set() for c2 in uniq if c2 != c1} for c1 in uniq}


# ------------------ derive residue-level arrays from Protein ------------------

AA_SET = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
NA_SET = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}  # DNA/RNA (common 3-letter)

# choose a single coordinate per residue:
#   - proteins: Cβ (GLY -> CA, else CA fallback if CB missing)
#   - nucleic acids: C3' (or C3*), else C1' (or C1*), else P as last resort
NA_PRIORITIES = (("C3'", "C3*"), ("C1'", "C1*"), ("P",))


def _pick_atom_coord(res, is_protein: bool) -> Optional[np.ndarray]:
  if is_protein:
    if "CB" in res:
      return res["CB"].coord
    if res.get_resname() == "GLY" and "CA" in res:
      return res["CA"].coord
    if "CA" in res:
      return res["CA"].coord
    return None
  # nucleic acids
  for tier in NA_PRIORITIES:
    for name in tier:
      if name in res:
        return res[name].coord
  return None


def _protein_to_residue_arrays(protein, model: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Returns arrays aligned across residues:
    residue_names (N,), chains (N,), residue_numbers (N,), cb_like_coords (N,3)
  Only biopolymer residues (standard AAs or nucleotides) with a valid proxy atom are included.
  """
  if model is None:
    model = protein.models()[0]
  assert model in protein.models(), f"Model {model} not present."

  residue_names: List[str] = []
  chains: List[str] = []
  residue_numbers: List[int] = []
  coords: List[np.ndarray] = []

  mdl = protein.structure[model]
  # iterate in structure order to keep stable alignment
  for chain in mdl:
    ch_id = chain.id
    for res in chain:
      resname = res.get_resname()
      is_aa = (res.id[0] == " ") and (resname in AA_SET)
      is_na = (res.id[0] == " ") and (resname in NA_SET)
      if not (is_aa or is_na):
        continue
      coord = _pick_atom_coord(res, is_protein=is_aa)
      if coord is None:
        continue  # skip residues without a good representative atom
      residue_names.append(resname)
      chains.append(ch_id)
      residue_numbers.append(int(res.id[1]))
      coords.append(coord.astype(float))

  if not residue_names:
    raise ValueError("No usable biopolymer residues with representative atoms were found.")

  return (
    np.array(residue_names, dtype=object),
    np.array(chains, dtype=object),
    np.array(residue_numbers, dtype=int),
    np.vstack(coords).astype(float),
  )


def _classify_chains(chains: np.ndarray, residue_names: np.ndarray) -> Dict[str, str]:
  chain_types: Dict[str, str] = {}
  uniq = np.unique(chains)
  for ch in uniq:
    idx = np.where(chains == ch)[0]
    names = residue_names[idx]
    is_na = any(r in NA_SET for r in names)
    chain_types[ch] = "nucleic_acid" if is_na else "protein"
  return chain_types


def calculate_ipSAE(
  protein,  # Neurosnap Protein
  plddt: np.ndarray,  # (N,) per-residue pLDDT aligned to derived residues
  pae_matrix: np.ndarray,  # (N,N) PAE aligned to derived residues
  *,
  model: Optional[int] = None,
  pae_cutoff: float = 10.0,
  dist_cutoff: float = 10.0,
  pDockQ_cutoff: float = 8.0,
  return_pml: bool = False,
) -> Dict[str, Any]:
  """
  Derives residue ordering from `Protein` and computes ipSAE, ipTM (d0chn),
  pDockQ, pDockQ2, and LIS for each chain pair.

  Alignment contract:
    - We produce an ordered list of residues (biopolymers with a valid proxy atom).
    - `plddt` and `pae_matrix` **must** be in that exact order.
      If you built them from AF outputs, map them to this residue list first.
  """
  residue_names, chains, residue_nums, coords_cb = _protein_to_residue_arrays(protein, model=model)
  N = len(chains)

  # shape checks
  if plddt.shape != (N,):
    raise ValueError(f"plddt shape {plddt.shape} does not match derived residue count {N}.")
  if pae_matrix.shape != (N, N):
    raise ValueError(f"pae_matrix shape {pae_matrix.shape} does not match ({N},{N}).")

  uniq_chains = np.unique(chains)
  chain_type = _classify_chains(chains, residue_names)
  pair_type = {
    c1: {
      c2: ("nucleic_acid" if (chain_type[c1] == "nucleic_acid" or chain_type[c2] == "nucleic_acid") else "protein") for c2 in uniq_chains if c2 != c1
    }
    for c1 in uniq_chains
  }

  # distance matrix from CB / NA proxy coordinates
  diffs = coords_cb[:, None, :] - coords_cb[None, :, :]
  distances = np.sqrt((diffs**2).sum(axis=2))

  # -------- allocate (same layout as previous version) --------
  iptm_d0chn_byres = init_pairdict_array(chains, N)
  ipsae_d0chn_byres = init_pairdict_array(chains, N)
  ipsae_d0dom_byres = init_pairdict_array(chains, N)
  ipsae_d0res_byres = init_pairdict_array(chains, N)

  iptm_d0chn_asym = init_pairdict_scalar(chains)
  ipsae_d0chn_asym = init_pairdict_scalar(chains)
  ipsae_d0dom_asym = init_pairdict_scalar(chains)
  ipsae_d0res_asym = init_pairdict_scalar(chains)

  iptm_d0chn_asymres = init_pairdict_scalar(chains, init_val=0.0)
  ipsae_d0chn_asymres = init_pairdict_scalar(chains, init_val=0.0)
  ipsae_d0dom_asymres = init_pairdict_scalar(chains, init_val=0.0)
  ipsae_d0res_asymres = init_pairdict_scalar(chains, init_val=0.0)

  iptm_d0chn_max = init_pairdict_scalar(chains)
  ipsae_d0chn_max = init_pairdict_scalar(chains)
  ipsae_d0dom_max = init_pairdict_scalar(chains)
  ipsae_d0res_max = init_pairdict_scalar(chains)

  iptm_d0chn_maxres = init_pairdict_scalar(chains, init_val=0.0)
  ipsae_d0chn_maxres = init_pairdict_scalar(chains, init_val=0.0)
  ipsae_d0dom_maxres = init_pairdict_scalar(chains, init_val=0.0)
  ipsae_d0res_maxres = init_pairdict_scalar(chains, init_val=0.0)

  n0chn = init_pairdict_scalar(chains)
  n0dom = init_pairdict_scalar(chains)
  n0dom_max = init_pairdict_scalar(chains)
  n0res = init_pairdict_scalar(chains)
  n0res_max = init_pairdict_scalar(chains)

  d0chn = init_pairdict_scalar(chains)
  d0dom = init_pairdict_scalar(chains)
  d0dom_max = init_pairdict_scalar(chains)
  d0res = init_pairdict_scalar(chains)
  d0res_max = init_pairdict_scalar(chains)

  n0res_byres = init_pairdict_array(chains, N)
  d0res_byres = init_pairdict_array(chains, N)

  valid_pair_counts = init_pairdict_scalar(chains, init_val=0.0)
  dist_valid_pair_counts = init_pairdict_scalar(chains, init_val=0.0)

  uniq_res_chain1 = init_pairdict_set(chains)
  uniq_res_chain2 = init_pairdict_set(chains)
  dist_uniq_res_chain1 = init_pairdict_set(chains)
  dist_uniq_res_chain2 = init_pairdict_set(chains)
  pdockq_uniq_res = init_pairdict_set(chains)

  pDockQ = init_pairdict_scalar(chains)
  pDockQ2 = init_pairdict_scalar(chains)
  LIS = init_pairdict_scalar(chains)

  # -------- pDockQ --------
  for c1 in uniq_chains:
    for c2 in uniq_chains:
      if c1 == c2:
        continue
      npairs = 0
      for i in range(N):
        if chains[i] != c1:
          continue
        near = (chains == c2) & (distances[i] <= pDockQ_cutoff)
        npairs += int(near.sum())
        if near.any():
          pdockq_uniq_res[c1][c2].add(i)
          for j in np.where(near)[0]:
            pdockq_uniq_res[c1][c2].add(j)
      if npairs > 0:
        idx = list(pdockq_uniq_res[c1][c2])
        mean_plddt = float(plddt[idx].mean()) if idx else 0.0
        x = mean_plddt * math.log10(npairs)
        pDockQ[c1][c2] = 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018
      else:
        pDockQ[c1][c2] = 0.0

  # -------- pDockQ2 --------
  for c1 in uniq_chains:
    for c2 in uniq_chains:
      if c1 == c2:
        continue
      npairs = 0
      s = 0.0
      for i in range(N):
        if chains[i] != c1:
          continue
        near = (chains == c2) & (distances[i] <= pDockQ_cutoff)
        if near.any():
          npairs += int(near.sum())
          s += ptm_func_vec(pae_matrix[i][near], 10.0).sum()
      if npairs > 0:
        idx = list(pdockq_uniq_res[c1][c2])
        mean_plddt = float(plddt[idx].mean()) if idx else 0.0
        mean_ptm = s / npairs
        x = mean_plddt * mean_ptm
        pDockQ2[c1][c2] = 1.31 / (1.0 + math.exp(-0.075 * (x - 84.733))) + 0.005
      else:
        pDockQ2[c1][c2] = 0.0

  # -------- LIS --------
  for c1 in uniq_chains:
    for c2 in uniq_chains:
      if c1 == c2:
        continue
      mask = (chains[:, None] == c1) & (chains[None, :] == c2)
      sel = pae_matrix[mask]
      if sel.size:
        valid = sel[sel <= 12.0]
        LIS[c1][c2] = float(((12.0 - valid) / 12.0).mean()) if valid.size else 0.0
      else:
        LIS[c1][c2] = 0.0

  # -------- ipTM/ipSAE by-residue --------
  for c1 in uniq_chains:
    for c2 in uniq_chains:
      if c1 == c2:
        continue

      n0chn[c1][c2] = int((chains == c1).sum() + (chains == c2).sum())
      d0chn[c1][c2] = calc_d0(n0chn[c1][c2], pair_type[c1][c2])

      ptm_d0chn = ptm_func_vec(pae_matrix, d0chn[c1][c2])

      valid_col_c2 = chains == c2

      for i in range(N):
        if chains[i] != c1:
          continue

        # ipTM_d0chn (interchain, no PAE cutoff on column)
        iptm_d0chn_byres[c1][c2][i] = float(ptm_d0chn[i, valid_col_c2].mean()) if valid_col_c2.any() else 0.0

        # ipSAE_d0chn (apply PAE cutoff)
        mask_ipsae = (chains == c2) & (pae_matrix[i] < pae_cutoff)
        ipsae_d0chn_byres[c1][c2][i] = float(ptm_d0chn[i, mask_ipsae].mean()) if mask_ipsae.any() else 0.0

        # track counts for domain-level
        valid_pair_counts[c1][c2] += float(mask_ipsae.sum())
        if mask_ipsae.any():
          uniq_res_chain1[c1][c2].add(int(residue_nums[i]))
          for j in np.where(mask_ipsae)[0]:
            uniq_res_chain2[c1][c2].add(int(residue_nums[j]))

        # distance-restricted (for interface counts)
        mask_if = (chains == c2) & (pae_matrix[i] < pae_cutoff) & (distances[i] < dist_cutoff)
        dist_valid_pair_counts[c1][c2] += float(mask_if.sum())
        if mask_if.any():
          dist_uniq_res_chain1[c1][c2].add(int(residue_nums[i]))
          for j in np.where(mask_if)[0]:
            dist_uniq_res_chain2[c1][c2].add(int(residue_nums[j]))

  # -------- ipSAE_d0dom & ipSAE_d0res --------
  for c1 in uniq_chains:
    for c2 in uniq_chains:
      if c1 == c2:
        continue
      nres_1 = len(uniq_res_chain1[c1][c2])
      nres_2 = len(uniq_res_chain2[c1][c2])
      n0dom[c1][c2] = int(nres_1 + nres_2)
      d0dom[c1][c2] = calc_d0(n0dom[c1][c2], pair_type[c1][c2])

      ptm_d0dom = ptm_func_vec(pae_matrix, d0dom[c1][c2])
      valid_mat = (chains[None, :] == c2) & (pae_matrix < pae_cutoff)

      n0res_row = valid_mat.sum(axis=1)
      d0res_row = calc_d0_array(n0res_row, pair_type[c1][c2])

      n0res_byres[c1][c2] = n0res_row
      d0res_byres[c1][c2] = d0res_row

      for i in range(N):
        if chains[i] != c1:
          continue
        row_mask = (chains == c2) & (pae_matrix[i] < pae_cutoff)
        ipsae_d0dom_byres[c1][c2][i] = float(ptm_d0dom[i, row_mask].mean()) if row_mask.any() else 0.0
        ptm_row_d0res = ptm_func_vec(pae_matrix[i], d0res_row[i])
        ipsae_d0res_byres[c1][c2][i] = float(ptm_row_d0res[row_mask].mean()) if row_mask.any() else 0.0

  # -------- asym maxima & symmetric maxima --------
  for c1 in uniq_chains:
    for c2 in uniq_chains:
      if c1 == c2:
        continue

      def set_asym(arr_byres, target_asym, target_asymres):
        vals = arr_byres[c1][c2]
        if vals.size:
          k = int(np.argmax(vals))
          target_asym[c1][c2] = float(vals[k])
          target_asymres[c1][c2] = f"{residue_names[k]:3}   {chains[k]:1} {int(residue_nums[k]):4d}"
        else:
          target_asym[c1][c2] = 0.0
          target_asymres[c1][c2] = "None"

      set_asym(iptm_d0chn_byres, iptm_d0chn_asym, iptm_d0chn_asymres)
      set_asym(ipsae_d0chn_byres, ipsae_d0chn_asym, ipsae_d0chn_asymres)
      set_asym(ipsae_d0dom_byres, ipsae_d0dom_asym, ipsae_d0dom_asymres)
      set_asym(ipsae_d0res_byres, ipsae_d0res_asym, ipsae_d0res_asymres)

      k = int(np.argmax(ipsae_d0res_byres[c1][c2])) if N else 0
      n0res[c1][c2] = float(n0res_byres[c1][c2][k]) if N else 0.0
      d0res[c1][c2] = float(d0res_byres[c1][c2][k]) if N else 0.0

  for c1 in uniq_chains:
    for c2 in uniq_chains:
      if c1 >= c2:
        continue

      def set_pair_max(asym, asym_res, out_max, out_max_res):
        v12 = asym[c1][c2]
        v21 = asym[c2][c1]
        if v12 >= v21:
          out_max[c1][c2] = out_max[c2][c1] = v12
          out_max_res[c1][c2] = out_max_res[c2][c1] = asym_res[c1][c2]
        else:
          out_max[c1][c2] = out_max[c2][c1] = v21
          out_max_res[c1][c2] = out_max_res[c2][c1] = asym_res[c2][c1]

      set_pair_max(iptm_d0chn_asym, iptm_d0chn_asymres, iptm_d0chn_max, iptm_d0chn_maxres)
      set_pair_max(ipsae_d0chn_asym, ipsae_d0chn_asymres, ipsae_d0chn_max, ipsae_d0chn_maxres)
      set_pair_max(ipsae_d0dom_asym, ipsae_d0dom_asymres, ipsae_d0dom_max, ipsae_d0dom_maxres)
      set_pair_max(ipsae_d0res_asym, ipsae_d0res_asymres, ipsae_d0res_max, ipsae_d0res_maxres)

      # carry n0/d0 companions from the winning directions
      if ipsae_d0dom_max[c1][c2] == ipsae_d0dom_asym[c1][c2]:
        n0dom_max[c1][c2] = n0dom_max[c2][c1] = n0dom[c1][c2]
        d0dom_max[c1][c2] = d0dom_max[c2][c1] = d0dom[c1][c2]
      else:
        n0dom_max[c1][c2] = n0dom_max[c2][c1] = n0dom[c2][c1]
        d0dom_max[c1][c2] = d0dom_max[c2][c1] = d0dom[c2][c1]

      if ipsae_d0res_max[c1][c2] == ipsae_d0res_asym[c1][c2]:
        n0res_max[c1][c2] = n0res_max[c2][c1] = n0res[c1][c2]
        d0res_max[c1][c2] = d0res_max[c2][c1] = d0res[c1][c2]
      else:
        n0res_max[c1][c2] = n0res_max[c2][c1] = n0res[c2][c1]
        d0res_max[c1][c2] = d0res_max[c2][c1] = d0res[c2][c1]

  # optional PyMOL script (same coloring scheme)
  pml = None
  if return_pml:
    chaincolor = {
      "A": "magenta",
      "B": "marine",
      "C": "lime",
      "D": "orange",
      "E": "yellow",
      "F": "cyan",
      "G": "lightorange",
      "H": "pink",
      "I": "deepteal",
      "J": "forest",
      "K": "lightblue",
      "L": "slate",
      "M": "violet",
      "N": "arsenic",
      "O": "iodine",
      "P": "silver",
      "Q": "red",
      "R": "sulfur",
      "S": "purple",
      "T": "olive",
      "U": "palegreen",
      "V": "green",
      "W": "blue",
      "X": "palecyan",
      "Y": "limon",
      "Z": "chocolate",
    }
    lines = []
    for c1 in uniq_chains:
      for c2 in uniq_chains:
        if c1 == c2:
          continue
        col1 = chaincolor.get(c1, "magenta")
        col2 = chaincolor.get(c2, "marine")
        res1 = contiguous_ranges(uniq_res_chain1[c1][c2])
        res2 = contiguous_ranges(uniq_res_chain2[c1][c2])
        sel1 = f"chain {c1} and resi {res1}" if res1 else f"chain {c1}"
        sel2 = f"chain {c2} and resi {res2}" if res2 else f"chain {c2}"
        alias = f"color_{c1}_{c2}"
        lines.append(f"alias {alias}, color gray80, all; color {col1}, {sel1}; color {col2}, {sel2}")
    pml = "\n".join(lines) + "\n"

  return {
    "by_residue": {
      "iptm_d0chn": iptm_d0chn_byres,
      "ipsae_d0chn": ipsae_d0chn_byres,
      "ipsae_d0dom": ipsae_d0dom_byres,
      "ipsae_d0res": ipsae_d0res_byres,
      "n0res_byres": n0res_byres,
      "d0res_byres": d0res_byres,
    },
    "asym": {
      "iptm_d0chn": iptm_d0chn_asym,
      "ipsae_d0chn": ipsae_d0chn_asym,
      "ipsae_d0dom": ipsae_d0dom_asym,
      "ipsae_d0res": ipsae_d0res_asym,
      "iptm_d0chn_res": iptm_d0chn_asymres,
      "ipsae_d0chn_res": ipsae_d0chn_asymres,
      "ipsae_d0dom_res": ipsae_d0dom_asymres,
      "ipsae_d0res_res": ipsae_d0res_asymres,
    },
    "max": {
      "iptm_d0chn": iptm_d0chn_max,
      "ipsae_d0chn": ipsae_d0chn_max,
      "ipsae_d0dom": ipsae_d0dom_max,
      "ipsae_d0res": ipsae_d0res_max,
      "iptm_d0chn_res": iptm_d0chn_maxres,
      "ipsae_d0chn_res": ipsae_d0chn_maxres,
      "ipsae_d0dom_res": ipsae_d0dom_maxres,
      "ipsae_d0res_res": ipsae_d0res_maxres,
    },
    "counts": {
      "n0chn": n0chn,
      "d0chn": d0chn,
      "n0dom": n0dom,
      "d0dom": d0dom,
      "n0dom_max": n0dom_max,
      "d0dom_max": d0dom_max,
      "n0res": n0res,
      "d0res": d0res,
      "n0res_max": n0res_max,
      "d0res_max": d0res_max,
      "pairs_with_pae_lt_cutoff": valid_pair_counts,
      "pairs_with_pae_lt_cutoff_and_dist": dist_valid_pair_counts,
      "unique_res_chain1": uniq_res_chain1,
      "unique_res_chain2": uniq_res_chain2,
      "dist_unique_res_chain1": dist_uniq_res_chain1,
      "dist_unique_res_chain2": dist_uniq_res_chain2,
    },
    "scores": {"pDockQ": pDockQ, "pDockQ2": pDockQ2, "LIS": LIS},
    "params": {
      "pae_cutoff": float(pae_cutoff),
      "dist_cutoff": float(dist_cutoff),
      "pDockQ_cutoff": float(pDockQ_cutoff),
      "model": model if model is not None else protein.models()[0],
    },
    "pml": pml,
    "residue_order": {
      "names": residue_names,
      "chains": chains,
      "numbers": residue_nums,
    },
  }
