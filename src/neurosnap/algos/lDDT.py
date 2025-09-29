"""
Code for lDDT (Local Distance Difference Test) calculation, adapted from https://github.com/ba-lab/disteval/blob/main/LDDT.ipynb
"""

import numpy as np
from math import sqrt

valid_amino_acids = {
  "LLP": "K",
  "TPO": "T",
  "CSS": "C",
  "OCS": "C",
  "CSO": "C",
  "PCA": "E",
  "KCX": "K",
  "CME": "C",
  "MLY": "K",
  "SEP": "S",
  "CSX": "C",
  "CSD": "C",
  "MSE": "M",
  "ALA": "A",
  "ASN": "N",
  "CYS": "C",
  "GLN": "Q",
  "HIS": "H",
  "LEU": "L",
  "MET": "M",
  "MHO": "M",
  "PRO": "P",
  "THR": "T",
  "TYR": "Y",
  "ARG": "R",
  "ASP": "D",
  "GLU": "E",
  "GLY": "G",
  "ILE": "I",
  "LYS": "K",
  "PHE": "F",
  "SER": "S",
  "TRP": "W",
  "VAL": "V",
  "SEC": "U",
}


def check_pdb_valid_row(valid_amino_acids, l):
  if (get_pdb_rname(l) in valid_amino_acids.keys()) and (l.startswith("ATOM") or l.startswith("HETA")):
    return True
  return False


def get_pdb_atom_name(l):
  return l[12:16].strip()


def get_pdb_rnum(l):
  return int(l[22:27].strip())


def get_pdb_rname(l):
  return l[17:20].strip()


def get_pdb_xyz_cb(lines):
  xyz = {}
  for l in lines:
    if get_pdb_atom_name(l) == "CB":
      xyz[get_pdb_rnum(l)] = (float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip()))
  for l in lines:
    if (get_pdb_rnum(l) not in xyz) and get_pdb_atom_name(l) == "CA":
      xyz[get_pdb_rnum(l)] = (float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip()))
  return xyz


def get_pdb_xyz_ca(lines):
  xyz = {}
  for l in lines:
    if get_pdb_atom_name(l) == "CA":
      xyz[get_pdb_rnum(l)] = (float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip()))
  return xyz


def pdb2dmap(pdbfile):
  f = open(pdbfile, mode="r")
  flines = f.read()
  f.close()
  lines = flines.splitlines()
  templines = flines.splitlines()
  for l in templines:
    if not l.startswith("ATOM"):
      lines.remove(l)
  # We have filtered out all non ATOMs at this point
  rnum_rnames = {}
  for l in lines:
    atom = get_pdb_atom_name(l)
    if atom != "CA":
      continue
    if not get_pdb_rname(l) in valid_amino_acids.keys():
      print("" + get_pdb_rname(l) + " is unknown amino acid in " + l)
      return
    rnum_rnames[int(get_pdb_rnum(l))] = valid_amino_acids[get_pdb_rname(l)]
  seq = ""
  for i in range(max(rnum_rnames.keys())):
    if i + 1 not in rnum_rnames:
      # print (rnum_rnames)
      # print ('Warning! residue not defined for rnum = ' + str(i+1))
      seq += "-"
    else:
      seq += rnum_rnames[i + 1]
  L = len(seq)
  xyz_cb = get_pdb_xyz_cb(lines)
  total_valid_residues = len(xyz_cb)
  if len(xyz_cb) != L:
    print(rnum_rnames)
    for i in range(L):
      if i + 1 not in xyz_cb:
        print("XYZ not defined for " + str(i + 1))
    print("Warning! Something went wrong - len of cbxyz != seqlen!! " + str(len(xyz_cb)) + " " + str(L))
  cb_map = np.full((L, L), np.nan)
  for r1 in sorted(xyz_cb):
    (a, b, c) = xyz_cb[r1]
    for r2 in sorted(xyz_cb):
      (p, q, r) = xyz_cb[r2]
      cb_map[r1 - 1, r2 - 1] = sqrt((a - p) ** 2 + (b - q) ** 2 + (c - r) ** 2)
  return (total_valid_residues, cb_map, rnum_rnames)


# Helpers for metrics calculated using numpy scheme
def get_flattened(dmap):
  if dmap.ndim == 1:
    return dmap
  elif dmap.ndim == 2:
    return dmap[np.triu_indices_from(dmap, k=1)]
  else:
    assert False, "ERROR: the passes array has dimension not equal to 2 or 1!"


def get_separations(dmap):
  t_indices = np.triu_indices_from(dmap, k=1)
  separations = np.abs(t_indices[0] - t_indices[1])
  return separations


# return a 1D boolean array indicating where the sequence separation in the
# upper triangle meets the threshold comparison
def get_sep_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {"gt", "lt", "ge", "le"}, "ERROR: Unknown comparator for thresholding!"
  separations = get_separations(dmap)
  if comparator == "gt":
    threshed = separations > thresh
  elif comparator == "lt":
    threshed = separations < thresh
  elif comparator == "ge":
    threshed = separations >= thresh
  elif comparator == "le":
    threshed = separations <= thresh
  return threshed


# return a 1D boolean array indicating where the distance in the
# upper triangle meets the threshold comparison
def get_dist_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {"gt", "lt", "ge", "le"}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = get_flattened(dmap)
  if comparator == "gt":
    threshed = dmap_flat > thresh
  elif comparator == "lt":
    threshed = dmap_flat < thresh
  elif comparator == "ge":
    threshed = dmap_flat >= thresh
  elif comparator == "le":
    threshed = dmap_flat <= thresh
  return threshed


# Calculate lDDT using numpy scheme
def get_lDDT(true_map, pred_map, R=15, sep_thresh=-1, T_set=[0.5, 1, 2, 4], precision=4):
  """
  Mariani V, Biasini M, Barbato A, Schwede T.
  lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests.
  Bioinformatics. 2013 Nov 1;29(21):2722-8.
  doi: 10.1093/bioinformatics/btt473.
  Epub 2013 Aug 27.
  PMID: 23986568; PMCID: PMC3799472.
  """

  # Helper for number preserved in a threshold
  def get_n_preserved(ref_flat, mod_flat, thresh):
    err = np.abs(ref_flat - mod_flat)
    n_preserved = (err < thresh).sum()
    return n_preserved

  # flatten upper triangles
  true_flat_map = get_flattened(true_map)
  pred_flat_map = get_flattened(pred_map)

  # Find set L
  S_thresh_indices = get_sep_thresh_b_indices(true_map, sep_thresh, "gt")
  R_thresh_indices = get_dist_thresh_b_indices(true_flat_map, R, "lt")

  L_indices = S_thresh_indices & R_thresh_indices

  true_flat_in_L = true_flat_map[L_indices]
  pred_flat_in_L = pred_flat_map[L_indices]

  # Number of pairs in L
  L_n = L_indices.sum()

  # Calculated lDDT
  preserved_fractions = []
  for _thresh in T_set:
    _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)
    _f_preserved = _n_preserved / L_n
    preserved_fractions.append(_f_preserved)

  lDDT = np.mean(preserved_fractions)
  if precision > 0:
    lDDT = round(lDDT, precision)
  return lDDT
