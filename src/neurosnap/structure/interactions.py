"""Interaction analysis helpers for Neurosnap structures."""

from typing import List, Optional, Tuple

import numpy as np

from neurosnap.constants.structure import HYDROPHOBIC_RESIDUES

from .interaction_report import InteractionEntity, InteractionReport
from .structure import Residue, Structure


def find_disulfide_bonds(structure: Structure, chain: Optional[str] = None, threshold: float = 2.05) -> List[Tuple[Residue, Residue]]:
  """Find disulfide bonds between cysteine residues using SG-SG distance.

  This is a legacy helper maintained alongside the
  :func:`analyze_interactions` engine.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional chain ID to restrict the search to.
    threshold: Maximum SG-SG distance in Å used to classify a disulfide bond.

  Returns:
    List of ``(residue1, residue2)`` cysteine pairs that satisfy the distance
    cutoff.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"find_disulfide_bonds() expects a Structure, found {type(structure).__name__}.")

  report = analyze_interactions(structure, interaction_types=["disulfide"], disulfide_cutoff=threshold)

  atom_to_residue = {}
  for chain_view in structure.chains():
    for residue in chain_view.residues():
      for idx in residue.atom_indices():
        atom_to_residue[idx] = residue

  disulfide_pairs = []
  seen = set()
  for rec in report.records:
    if rec.interaction_type == "disulfide":
      res1 = atom_to_residue.get(rec.atom_index1)
      res2 = atom_to_residue.get(rec.atom_index2)
      if res1 is not None and res2 is not None:
        if chain is not None:
          if res1.chain_id != chain or res2.chain_id != chain:
            continue
        pair = (res1, res2)
        pair_key = (res1.key(), res2.key())
        if pair_key not in seen:
          seen.add(pair_key)
          disulfide_pairs.append(pair)

  return disulfide_pairs


def find_salt_bridges(structure: Structure, chain: Optional[str] = None, cutoff: float = 4.0) -> List[Tuple[Residue, Residue]]:
  """Identify salt bridges using charged side-chain atoms/groups.

  This is a legacy helper maintained alongside the
  :func:`analyze_interactions` engine.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional chain ID to restrict the search to. When None, both intra-
      and inter-chain salt bridges are returned. When specified, only salt
      bridges where both residues are within the specified chain are returned.
    cutoff: Maximum atom/group distance in Å used to classify a salt bridge.

  Returns:
    List of ``(positive_residue, negative_residue)`` pairs that satisfy the
    ionic contact rules.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"find_salt_bridges() expects a Structure, found {type(structure).__name__}.")

  report = analyze_interactions(structure, interaction_types=["salt_bridge"], salt_bridge_cutoff=cutoff)

  atom_to_residue = {}
  for chain_view in structure.chains():
    for residue in chain_view.residues():
      for idx in residue.atom_indices():
        atom_to_residue[idx] = residue

  salt_bridges = []
  seen = set()
  for rec in report.records:
    if rec.interaction_type == "salt_bridge":
      res1 = atom_to_residue.get(rec.atom_index1)
      res2 = atom_to_residue.get(rec.atom_index2)
      if res1 is not None and res2 is not None:
        if chain is not None:
          if res1.chain_id != chain or res2.chain_id != chain:
            continue
        if rec.role1 == "positive":
          pos_res, neg_res = res1, res2
        else:
          pos_res, neg_res = res2, res1

        pair_key = (pos_res.key(), neg_res.key())
        if pair_key not in seen:
          seen.add(pair_key)
          salt_bridges.append((pos_res, neg_res))

  return salt_bridges


def find_hydrophobic_residues(structure: Structure, chain: Optional[str] = None) -> List[Tuple[str, Residue]]:
  """Return hydrophobic residues from a single structure.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional chain ID to restrict the search to.

  Returns:
    List of ``(chain_id, residue)`` tuples for residues classified as
    hydrophobic.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"find_hydrophobic_residues() expects a Structure, found {type(structure).__name__}.")
  hydrophobic = []

  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if residue.res_name.strip().upper() in HYDROPHOBIC_RESIDUES:
        hydrophobic.append((chain_view.chain_id, residue))

  return hydrophobic


def calculate_hydrogen_bonds(
  structure: Structure,
  chain: Optional[str] = None,
  chain_other: Optional[str] = None,
  *,
  donor_acceptor_cutoff: float = 3.5,
  angle_cutoff: float = 120.0,
) -> int:
  """Count hydrogen bonds using explicit hydrogens and simple geometric cutoffs.

  This is a legacy helper maintained alongside the
  :func:`analyze_interactions` engine.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional donor-chain ID. When omitted, all chains are searched.
    chain_other: Optional acceptor-chain ID for inter-chain counting. Both must
      be provided if chain_other is specified.
    donor_acceptor_cutoff: Maximum donor-acceptor distance in Å.
    angle_cutoff: Minimum donor-H-acceptor angle in degrees.

  Returns:
    Total number of hydrogen bonds that satisfy the geometric cutoffs.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"calculate_hydrogen_bonds() expects a Structure, found {type(structure).__name__}.")
  _validate_hydrogen_bond_inputs(structure, chain=chain, chain_other=chain_other)

  report = analyze_interactions(
    structure, interaction_types=["hydrogen_bond"], hbond_donor_acceptor_cutoff=donor_acceptor_cutoff, hbond_angle_cutoff=angle_cutoff
  )

  hydrogen_bond_count = 0
  for rec in report.records:
    if rec.interaction_type == "hydrogen_bond" and rec.evidence == "detected":
      d_chain = rec.chain1 if rec.role1 == "donor" else rec.chain2
      a_chain = rec.chain2 if rec.role1 == "donor" else rec.chain1

      if chain is not None:
        if d_chain != chain:
          continue
        if chain_other is not None:
          if a_chain != chain_other:
            continue
        else:
          if a_chain != chain:
            continue

      hydrogen_bond_count += 1

  return hydrogen_bond_count


def calculate_interface_hydrogen_bonding_residues(
  structure: Structure,
  chain: Optional[str] = None,
  chain_other: Optional[str] = None,
  *,
  donor_acceptor_cutoff: float = 3.5,
  angle_cutoff: float = 120.0,
) -> int:
  """Count unique residues that participate in inter- or intra-chain hydrogen bonds.

  This is a legacy helper maintained alongside the
  :func:`analyze_interactions` engine.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional donor-chain ID. When omitted, all chains are searched.
    chain_other: Optional acceptor-chain ID for inter-chain counting.
    donor_acceptor_cutoff: Maximum donor-acceptor distance in Å.
    angle_cutoff: Minimum donor-H-acceptor angle in degrees.

  Returns:
    Number of unique residues that participate in at least one qualifying
    hydrogen bond.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"calculate_interface_hydrogen_bonding_residues() expects a Structure, found {type(structure).__name__}.")
  _validate_hydrogen_bond_inputs(structure, chain=chain, chain_other=chain_other)

  report = analyze_interactions(
    structure, interaction_types=["hydrogen_bond"], hbond_donor_acceptor_cutoff=donor_acceptor_cutoff, hbond_angle_cutoff=angle_cutoff
  )

  atom_to_residue = {}
  for chain_view in structure.chains():
    for residue in chain_view.residues():
      for idx in residue.atom_indices():
        atom_to_residue[idx] = residue

  hydrogen_bonding_residues = set()
  for rec in report.records:
    if rec.interaction_type == "hydrogen_bond" and rec.evidence == "detected":
      d_chain = rec.chain1 if rec.role1 == "donor" else rec.chain2
      a_chain = rec.chain2 if rec.role1 == "donor" else rec.chain1

      if chain is not None:
        if d_chain != chain:
          continue
        if chain_other is not None:
          if a_chain != chain_other:
            continue
        else:
          if a_chain != chain:
            continue

      if chain_other is not None and d_chain == a_chain:
        continue

      res1 = atom_to_residue.get(rec.atom_index1)
      res2 = atom_to_residue.get(rec.atom_index2)
      if res1 is not None:
        hydrogen_bonding_residues.add(res1)
      if res2 is not None:
        hydrogen_bonding_residues.add(res2)

  return len(hydrogen_bonding_residues)


def _validate_hydrogen_bond_inputs(structure: Structure, chain: Optional[str], chain_other: Optional[str]):
  """Validate hydrogen-bond chain inputs against a structure."""
  available_chains = set(structure.chain_ids())
  if chain_other is not None and chain is None:
    raise ValueError("`chain_other` is specified, but `chain` is not. Both must be provided for inter-chain calculation.")
  if chain is not None and chain not in available_chains:
    raise ValueError(f"Chain {chain} does not exist within the input structure.")
  if chain_other is not None and chain_other not in available_chains:
    raise ValueError(f"Chain {chain_other} does not exist within the input structure.")


def _find_neighbor_candidates(
  coords: np.ndarray,
  indices1: np.ndarray,
  indices2: np.ndarray,
  cutoff: float,
) -> List[Tuple[int, int, float]]:
  """Find neighbor candidates between two disjoint sets of atom indices using cKDTree."""
  if len(indices1) == 0 or len(indices2) == 0:
    return []

  c1 = coords[indices1]
  c2 = coords[indices2]

  # Reject NaN/Inf coordinates before building the tree
  if not np.all(np.isfinite(c1)) or not np.all(np.isfinite(c2)):
    import logging

    logging.getLogger("neurosnap").warning("Non-finite coordinates detected in neighbor search. Filtering them out.")
    finite1 = np.all(np.isfinite(c1), axis=1)
    finite2 = np.all(np.isfinite(c2), axis=1)
    indices1 = indices1[finite1]
    indices2 = indices2[finite2]
    c1 = c1[finite1]
    c2 = c2[finite2]
    if len(indices1) == 0 or len(indices2) == 0:
      return []

  from scipy.spatial import cKDTree

  tree1 = cKDTree(c1)
  tree2 = cKDTree(c2)

  # Dual-tree query
  pairs_list = tree1.query_ball_tree(tree2, cutoff)

  results = []
  for i, neighbors in enumerate(pairs_list):
    idx1 = indices1[i]
    for j in neighbors:
      idx2 = indices2[j]
      dist = float(np.linalg.norm(c1[i] - c2[j]))
      u, v = sorted((idx1, idx2))
      results.append((u, v, dist))

  # Deduplicate
  seen = set()
  unique_results = []
  for u, v, dist in results:
    if (u, v) not in seen:
      seen.add((u, v))
      unique_results.append((u, v, dist))

  # Deterministic sort
  unique_results.sort(key=lambda x: (x[0], x[1]))
  return unique_results


def analyze_interactions(
  structure: Structure,
  *entities: "InteractionEntity",
  interaction_types: Optional[List[str]] = None,
  contact_cutoff_a: float = 4.5,
  vdw_tolerance_a: float = 0.5,
  clash_overlap_a: float = 0.4,
  include_hydrogens: bool = False,
  covalent_candidates: bool = False,
  covalent_lower_factor: float = 0.8,
  covalent_upper_factor: float = 1.2,
  disulfide_cutoff: float = 2.2,
  salt_bridge_cutoff: float = 4.0,
  hbond_donor_acceptor_cutoff: float = 3.5,
  hbond_angle_cutoff: float = 130.0,
  metal_coordination_cutoff: float = 2.8,
  include_candidates: bool = False,
) -> "InteractionReport":
  """High-level interaction analyzer orchestrator.

  Parameters:
    structure: Input single-model :class:`Structure`.
    *entities: Positional InteractionEntity objects to analyze.
    interaction_types: Interaction types to analyze. Defaults to conservative
      ["contact", "covalent"].
    contact_cutoff_a: Maximum contact cutoff distance.
    vdw_tolerance_a: Tolerance added to VDW radii sum.
    clash_overlap_a: Overlap distance to classify VDW clash.
    include_hydrogens: Whether to include hydrogens.
    covalent_candidates: Whether to calculate covalent candidate interactions.
    covalent_lower_factor: Lower factor for covalent candidate bond distance.
    covalent_upper_factor: Upper factor for covalent candidate bond distance.
    disulfide_cutoff: Cutoff distance for disulfide bonds.
    salt_bridge_cutoff: Cutoff distance for salt bridges.
    hbond_donor_acceptor_cutoff: Cutoff distance for hydrogen bonds.
    hbond_angle_cutoff: Minimum angle for hydrogen bonds.
    metal_coordination_cutoff: Cutoff distance for metal coordination.
    include_candidates: Whether to include candidate interactions.

  Returns:
    InteractionReport containing deterministically sorted records and center summaries.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"analyze_interactions() expects a Structure, found {type(structure).__name__}.")

  # Validate options
  if interaction_types is None:
    interaction_types = ["contact", "covalent"]

  return structure._analyze_interactions(
    interaction_types=interaction_types,
    entities=entities if entities else None,
    contact_cutoff_a=contact_cutoff_a,
    vdw_tolerance_a=vdw_tolerance_a,
    clash_overlap_a=clash_overlap_a,
    include_hydrogens=include_hydrogens,
    covalent_candidates=covalent_candidates,
    covalent_lower_factor=covalent_lower_factor,
    covalent_upper_factor=covalent_upper_factor,
    disulfide_cutoff=disulfide_cutoff,
    salt_bridge_cutoff=salt_bridge_cutoff,
    hbond_donor_acceptor_cutoff=hbond_donor_acceptor_cutoff,
    hbond_angle_cutoff=hbond_angle_cutoff,
    metal_coordination_cutoff=metal_coordination_cutoff,
    include_candidates=include_candidates,
  )
