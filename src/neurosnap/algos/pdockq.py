from typing import Tuple, Union

import numpy as np

from neurosnap.protein import Protein


def _chain_cb_or_gly_ca(prot: Protein, chain_id: str) -> Tuple[np.ndarray, np.ndarray]:
  """
  Extract per-residue coordinates and plDDT for a chain using CB atoms,
  or CA for GLY residues (matching the original pDockQ convention).

  Returns:
      coords: (N, 3) array
      plddt: (N,) array aligned with coords
  """
  df = prot.df
  sel = df[(df["chain"] == chain_id) & ((df["atom_name"] == "CB") | ((df["atom_name"] == "CA") & (df["res_name"] == "GLY")))].copy()

  # Ensure stable ordering along the sequence
  # (sort by residue number, then by atom name to keep deterministic order)
  sel.sort_values(["res_id", "atom_name"], inplace=True)

  if sel.empty:
    # Fall back to CA-only if nothing found (e.g., all-GLY chains without CB records)
    sel = df[(df["chain"] == chain_id) & (df["atom_name"] == "CA")].copy()
    sel.sort_values(["res_id", "atom_name"], inplace=True)

  coords = sel[["x", "y", "z"]].to_numpy(dtype=float)
  plddt = sel["bfactor"].to_numpy(dtype=float)  # AlphaFold stores pLDDT in B-factor
  return coords, plddt


def _calc_pdockq_from_arrays(chain_coords: dict, chain_plddt: dict, dist_thresh: float) -> Tuple[float, float]:
  """
  Reproduce the original pDockQ mapping given pre-extracted arrays.
  Returns (pDockQ, PPV).
  """
  # Unpack the two chains
  ch1, ch2 = list(chain_coords.keys())
  coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
  plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

  if coords1.size == 0 or coords2.size == 0:
    return 0.0, 0.0

  # Pairwise distances between residues (CB/CA(GLY) representatives)
  mat = np.vstack([coords1, coords2])
  diff = mat[:, None, :] - mat[None, :, :]
  dists = np.sqrt(np.sum(diff**2, axis=-1))

  l1 = len(coords1)
  contact_dists = dists[:l1, l1:]  # distances between chain1 (rows) and chain2 (cols)
  contacts = np.argwhere(contact_dists <= dist_thresh)

  if contacts.shape[0] < 1:
    return 0.0, 0.0

  # Average interface pLDDT (unique residues on either side that form a contact)
  if_idx1 = np.unique(contacts[:, 0])
  if_idx2 = np.unique(contacts[:, 1])
  avg_if_plddt = np.average(np.concatenate([plddt1[if_idx1], plddt2[if_idx2]]))

  # Number of interface contacts
  n_if_contacts = contacts.shape[0]

  # Logistic mapping (original constants)
  x = avg_if_plddt * np.log10(max(n_if_contacts, 1))
  pdockq = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

  # PPV calibration (original look-up)
  PPV = np.array(
    [
      0.98128027,
      0.96322524,
      0.95333044,
      0.9400192,
      0.93172991,
      0.92420274,
      0.91629946,
      0.90952562,
      0.90043139,
      0.8919553,
      0.88570037,
      0.87822061,
      0.87116417,
      0.86040801,
      0.85453785,
      0.84294946,
      0.83367787,
      0.82238224,
      0.81190228,
      0.80223507,
      0.78549007,
      0.77766077,
      0.75941223,
      0.74006263,
      0.73044282,
      0.71391784,
      0.70615739,
      0.68635536,
      0.66728511,
      0.63555449,
      0.55890174,
    ]
  )
  pdockq_thresholds = np.array(
    [
      0.67333079,
      0.65666073,
      0.63254566,
      0.62604391,
      0.60150931,
      0.58313803,
      0.5647381,
      0.54122438,
      0.52314392,
      0.49659878,
      0.4774676,
      0.44661346,
      0.42628389,
      0.39990988,
      0.38479715,
      0.3649393,
      0.34526004,
      0.3262589,
      0.31475668,
      0.29750023,
      0.26673725,
      0.24561247,
      0.21882689,
      0.19651314,
      0.17606258,
      0.15398168,
      0.13927677,
      0.12024131,
      0.09996019,
      0.06968505,
      0.02946438,
    ]
  )
  inds = np.argwhere(pdockq_thresholds >= pdockq)
  if len(inds) > 0:
    ppv = float(PPV[inds[-1]][0])
  else:
    ppv = float(PPV[0])

  return float(pdockq), float(ppv)


def calculate_pDockQ(structure: Union[str, Protein], chain_id1: str, chain_id2: str, dist_thresh: float = 8.0) -> Tuple[float, float]:
  """
  Calculate the predicted DockQ (pDockQ) score and corresponding PPV
  for a two-chain complex.

  Args:
      structure: Protein object or a path/ID string that can be parsed by neurosnap.protein.Protein
      chain_id1: Chain ID for the first chain (e.g., "A")
      chain_id2: Chain ID for the second chain (e.g., "B")
      dist_thresh: Distance threshold (Å) for interface contacts, default 8.0

  Returns:
      (pdockq, ppv)
  """
  # Normalize/obtain a Protein object
  prot = structure if isinstance(structure, Protein) else Protein(structure, format="auto")

  # Basic chain validation
  chains = set(prot.chains(prot.models()[0]))
  if chain_id1 not in chains:
    raise ValueError(f'Chain "{chain_id1}" not found. Available chains: {sorted(chains)}')
  if chain_id2 not in chains:
    raise ValueError(f'Chain "{chain_id2}" not found. Available chains: {sorted(chains)}')
  if chain_id1 == chain_id2:
    raise ValueError("chain_id1 and chain_id2 must be different.")

  # Extract chain representatives and plDDT
  coords1, plddt1 = _chain_cb_or_gly_ca(prot, chain_id1)
  coords2, plddt2 = _chain_cb_or_gly_ca(prot, chain_id2)

  chain_coords = {chain_id1: coords1, chain_id2: coords2}
  chain_plddt = {chain_id1: plddt1, chain_id2: plddt2}

  # Compute and return
  return _calc_pdockq_from_arrays(chain_coords, chain_plddt, dist_thresh)
