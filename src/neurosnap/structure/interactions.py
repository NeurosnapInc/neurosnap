"""Interaction analysis helpers for Neurosnap structures."""

from typing import List, Optional, Set, Tuple

import numpy as np

from neurosnap.constants import HYDROPHOBIC_RESIDUES

from ._common import available_chain_ids, resolve_model
from .structure import Atom, Residue


def find_disulfide_bonds(
  structure, chain: Optional[str] = None, model: Optional[int] = None, threshold: float = 2.05
) -> List[Tuple[Residue, Residue]]:
  """Find disulfide bonds between cysteine residues using SG-SG distance.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    chain: Optional chain ID to restrict the search to.
    model: Optional model ID when selecting from an ensemble or stack.
    threshold: Maximum SG-SG distance in Å used to classify a disulfide bond.

  Returns:
    List of ``(residue1, residue2)`` cysteine pairs that satisfy the distance
    cutoff.
  """
  structure_model = resolve_model(structure, model=model)
  disulfide_pairs = []

  for chain_view in structure_model.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    cysteines = [residue for residue in chain_view.residues() if residue.res_name.strip().upper() == "CYS"]
    for index, residue1 in enumerate(cysteines):
      residue1_sg = _atom_by_name(residue1, "SG")
      if residue1_sg is None:
        continue
      for residue2 in cysteines[index + 1 :]:
        residue2_sg = _atom_by_name(residue2, "SG")
        if residue2_sg is None:
          continue
        if np.linalg.norm(residue1_sg.coord - residue2_sg.coord) < threshold:
          disulfide_pairs.append((residue1, residue2))

  return disulfide_pairs


def find_salt_bridges(structure, chain: Optional[str] = None, model: Optional[int] = None, cutoff: float = 4.0) -> List[Tuple[Residue, Residue]]:
  """Identify salt bridges using CA-CA distance as a simple proxy.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    chain: Optional chain ID to restrict the search to.
    model: Optional model ID when selecting from an ensemble or stack.
    cutoff: Maximum CA-CA distance in Å used to classify a salt bridge.

  Returns:
    List of ``(positive_residue, negative_residue)`` pairs that satisfy the
    distance cutoff.
  """
  structure_model = resolve_model(structure, model=model)
  positive_residues = {"LYS", "ARG"}
  negative_residues = {"ASP", "GLU"}
  salt_bridges = []

  for chain_view in structure_model.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    positive = [residue for residue in chain_view.residues() if residue.res_name.strip().upper() in positive_residues]
    negative = [residue for residue in chain_view.residues() if residue.res_name.strip().upper() in negative_residues]
    for positive_residue in positive:
      positive_ca = _atom_by_name(positive_residue, "CA")
      if positive_ca is None:
        continue
      for negative_residue in negative:
        negative_ca = _atom_by_name(negative_residue, "CA")
        if negative_ca is None:
          continue
        if np.linalg.norm(positive_ca.coord - negative_ca.coord) < cutoff:
          salt_bridges.append((positive_residue, negative_residue))

  return salt_bridges


def find_hydrophobic_residues(structure, chain: Optional[str] = None, model: Optional[int] = None) -> List[Tuple[str, Residue]]:
  """Return hydrophobic residues from a selected model.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    chain: Optional chain ID to restrict the search to.
    model: Optional model ID when selecting from an ensemble or stack.

  Returns:
    List of ``(chain_id, residue)`` tuples for residues classified as
    hydrophobic.
  """
  structure_model = resolve_model(structure, model=model)
  hydrophobic = []

  for chain_view in structure_model.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if residue.res_name.strip().upper() in HYDROPHOBIC_RESIDUES:
        hydrophobic.append((chain_view.chain_id, residue))

  return hydrophobic


def calculate_hydrogen_bonds(
  structure,
  chain: Optional[str] = None,
  chain_other: Optional[str] = None,
  *,
  model: Optional[int] = None,
  donor_acceptor_cutoff: float = 3.5,
  angle_cutoff: float = 120.0,
) -> int:
  """Count hydrogen bonds using explicit hydrogens and simple geometric cutoffs.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    chain: Optional donor-chain ID. When omitted, all chains are searched.
    chain_other: Optional acceptor-chain ID for inter-chain counting.
    model: Optional model ID when selecting from an ensemble or stack.
    donor_acceptor_cutoff: Maximum donor-acceptor distance in Å.
    angle_cutoff: Minimum donor-H-acceptor angle in degrees.

  Returns:
    Total number of hydrogen bonds that satisfy the geometric cutoffs.
  """
  structure_model = resolve_model(structure, model=model)
  _validate_hydrogen_bond_inputs(structure_model, chain=chain, chain_other=chain_other)

  hydrogen_distance_cutoff = 1.2
  hydrogen_bond_count = 0
  chain_lookup = {chain_view.chain_id: chain_view for chain_view in structure_model.chains()}
  donor_chain_ids = [chain] if chain is not None else list(chain_lookup)

  for donor_chain_id in donor_chain_ids:
    donor_chain = chain_lookup[donor_chain_id]
    acceptor_chain_ids = [chain_other] if chain_other else ([donor_chain_id] if chain is not None else list(chain_lookup))

    for donor_residue in donor_chain.residues():
      for donor_atom in donor_residue.atoms():
        if donor_atom.element not in {"N", "O"}:
          continue
        bonded_hydrogens = [
          atom for atom in donor_residue.atoms() if atom.element == "H" and np.linalg.norm(donor_atom.coord - atom.coord) <= hydrogen_distance_cutoff
        ]
        if not bonded_hydrogens:
          continue

        for acceptor_chain_id in acceptor_chain_ids:
          acceptor_chain = chain_lookup[acceptor_chain_id]
          for acceptor_residue in acceptor_chain.residues():
            for acceptor_atom in acceptor_residue.atoms():
              if acceptor_atom.element not in {"N", "O"}:
                continue
              if acceptor_atom == donor_atom:
                continue
              if np.linalg.norm(donor_atom.coord - acceptor_atom.coord) > donor_acceptor_cutoff:
                continue

              for hydrogen in bonded_hydrogens:
                if _hydrogen_bond_angle(donor_atom, hydrogen, acceptor_atom) >= angle_cutoff:
                  hydrogen_bond_count += 1
                  break

  return hydrogen_bond_count


def calculate_interface_hydrogen_bonding_residues(
  structure,
  chain: Optional[str] = None,
  chain_other: Optional[str] = None,
  *,
  model: Optional[int] = None,
  donor_acceptor_cutoff: float = 3.5,
  angle_cutoff: float = 120.0,
) -> int:
  """Count unique residues that participate in inter- or intra-chain hydrogen bonds.

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or
      :class:`StructureStack`.
    chain: Optional donor-chain ID. When omitted, all chains are searched.
    chain_other: Optional acceptor-chain ID for inter-chain counting.
    model: Optional model ID when selecting from an ensemble or stack.
    donor_acceptor_cutoff: Maximum donor-acceptor distance in Å.
    angle_cutoff: Minimum donor-H-acceptor angle in degrees.

  Returns:
    Number of unique residues that participate in at least one qualifying
    hydrogen bond.
  """
  structure_model = resolve_model(structure, model=model)
  _validate_hydrogen_bond_inputs(structure_model, chain=chain, chain_other=chain_other)

  hydrogen_distance_cutoff = 1.2
  chain_lookup = {chain_view.chain_id: chain_view for chain_view in structure_model.chains()}
  donor_chain_ids = [chain] if chain is not None else list(chain_lookup)
  hydrogen_bonding_residues: Set[Tuple[str, int, str, str, bool]] = set()

  for donor_chain_id in donor_chain_ids:
    donor_chain = chain_lookup[donor_chain_id]
    acceptor_chain_ids = [chain_other] if chain_other else ([donor_chain_id] if chain is not None else list(chain_lookup))

    for donor_residue in donor_chain.residues():
      for donor_atom in donor_residue.atoms():
        if donor_atom.element not in {"N", "O"}:
          continue
        bonded_hydrogens = [
          atom for atom in donor_residue.atoms() if atom.element == "H" and np.linalg.norm(donor_atom.coord - atom.coord) <= hydrogen_distance_cutoff
        ]
        if not bonded_hydrogens:
          continue

        for acceptor_chain_id in acceptor_chain_ids:
          acceptor_chain = chain_lookup[acceptor_chain_id]
          if chain_other and acceptor_chain_id == donor_chain_id:
            continue
          for acceptor_residue in acceptor_chain.residues():
            for acceptor_atom in acceptor_residue.atoms():
              if acceptor_atom.element not in {"N", "O"}:
                continue
              if acceptor_atom == donor_atom:
                continue
              if np.linalg.norm(donor_atom.coord - acceptor_atom.coord) > donor_acceptor_cutoff:
                continue

              for hydrogen in bonded_hydrogens:
                if _hydrogen_bond_angle(donor_atom, hydrogen, acceptor_atom) >= angle_cutoff:
                  hydrogen_bonding_residues.add(
                    (donor_residue.chain_id, donor_residue.res_id, donor_residue.ins_code, donor_residue.res_name, donor_residue.hetero)
                  )
                  hydrogen_bonding_residues.add(
                    (
                      acceptor_residue.chain_id,
                      acceptor_residue.res_id,
                      acceptor_residue.ins_code,
                      acceptor_residue.res_name,
                      acceptor_residue.hetero,
                    )
                  )
                  break

  return len(hydrogen_bonding_residues)


def _atom_by_name(residue: Residue, atom_name: str) -> Optional[Atom]:
  """Return an atom from a residue by name."""
  atom_name = atom_name.strip().upper()
  for atom in residue.atoms():
    if atom.atom_name.strip().upper() == atom_name:
      return atom
  return None


def _validate_hydrogen_bond_inputs(structure_model, chain: Optional[str], chain_other: Optional[str]):
  """Validate hydrogen-bond chain inputs against a selected model."""
  available_chains = set(available_chain_ids(structure_model))
  if chain_other is not None and chain is None:
    raise ValueError("`chain_other` is specified, but `chain` is not. Both must be provided for inter-chain calculation.")
  if chain is not None and chain not in available_chains:
    raise ValueError(f"Chain {chain} does not exist within the input structure.")
  if chain_other is not None and chain_other not in available_chains:
    raise ValueError(f"Chain {chain_other} does not exist within the input structure.")


def _hydrogen_bond_angle(donor_atom: Atom, hydrogen_atom: Atom, acceptor_atom: Atom) -> float:
  """Return the donor-H-acceptor angle in degrees."""
  donor_h_vector = hydrogen_atom.coord - donor_atom.coord
  donor_acceptor_vector = acceptor_atom.coord - donor_atom.coord
  donor_h_norm = np.linalg.norm(donor_h_vector)
  donor_acceptor_norm = np.linalg.norm(donor_acceptor_vector)
  if donor_h_norm == 0 or donor_acceptor_norm == 0:
    return 0.0
  cos_theta = np.dot(donor_h_vector, donor_acceptor_vector) / (donor_h_norm * donor_acceptor_norm)
  cos_theta = max(min(float(cos_theta), 1.0), -1.0)
  return float(np.degrees(np.arccos(cos_theta)))
