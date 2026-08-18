"""Structure-preparation helpers.

This module provides small structure-level preparation utilities plus thin
wrappers around the existing PDB2PQR preparation backend.

The wrappers intentionally do not reimplement that engine. They expose a
structure-oriented API while delegating the underlying chemistry logic to the
existing algorithm module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from neurosnap.constants.sequence import AA_RECORDS_CANONICAL, AA_RECORDS_FORCEFIELD_VARIANTS

from ._common import filter_structure_atoms
from .structure import BondType, Residue, Structure

__all__ = [
  "has_hydrogens",
  "strip_hydrogens",
  "remove_altlocs_and_duplicate_atoms",
  "add_terminal_capping_groups",
  "add_hydrogens_with_pdb2pqr",
  "optimize_hydrogens_with_pdb2pqr",
]


def has_hydrogens(structure: Structure) -> bool:
  """Return ``True`` if the structure currently contains hydrogen atoms.

  Parameters:
    structure: Input single-model structure.

  Returns:
    ``True`` when any atom has element ``H`` after simple normalization.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"has_hydrogens() expects a Structure, found {type(structure).__name__}.")
  if len(structure) == 0:
    return False
  elements = np.char.upper(np.char.strip(structure.atom_annotations["element"].astype("U2")))
  return bool(np.any(elements == "H"))


def strip_hydrogens(structure: Structure) -> Structure:
  """Return a copy of the structure with hydrogen atoms removed.

  Atom-level connectivity tables are subsetted and remapped automatically via
  :meth:`Structure.select`, so both bonds and interactions remain consistent
  with the returned atom table.

  Parameters:
    structure: Input single-model structure.

  Returns:
    New :class:`Structure` without hydrogen atoms.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"strip_hydrogens() expects a Structure, found {type(structure).__name__}.")
  return structure.select(predicate=lambda atom: atom.element.strip().upper() != "H")


def remove_altlocs_and_duplicate_atoms(structure: Structure) -> Structure:
  """Return a copy with one atom retained for each residue atom site.

  Atom sites are identified by chain ID, residue ID, insertion code, residue
  name, hetero flag, and atom name. When multiple atoms share a site, the atom
  with the highest occupancy is retained. Ties prefer a blank alternate
  location, then alternate location ``A``, then the first atom in input order.

  Neurosnap's PDB/mmCIF parsers already collapse alternate locations while
  loading. This helper is mainly useful for structures built manually or from
  workflows that add an ``altloc``-style annotation column.

  Parameters:
    structure: Input single-model structure.

  Returns:
    New :class:`Structure` with duplicate atom sites removed. If an optional
    alternate-location annotation column is present, it is removed from the
    returned structure.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"remove_altlocs_and_duplicate_atoms() expects a Structure, found {type(structure).__name__}.")

  deduplicated = structure.select()
  altloc_annotation_names = ("altloc", "alt_loc", "label_alt_id")

  def remove_altloc_annotations() -> None:
    for field_name in altloc_annotation_names:
      if field_name in deduplicated.atom_annotations.dtype.names:
        deduplicated.remove_annotation(field_name)

  if len(deduplicated) == 0:
    remove_altloc_annotations()
    return deduplicated

  annotation_names = deduplicated.atom_annotations.dtype.names

  def atom_site_key(atom_index: int) -> tuple:
    annotations = deduplicated.atom_annotations
    return (
      str(annotations["chain_id"][atom_index]).strip(),
      int(annotations["res_id"][atom_index]),
      str(annotations["ins_code"][atom_index]).strip(),
      str(annotations["res_name"][atom_index]).strip().upper(),
      bool(annotations["hetero"][atom_index]),
      str(annotations["atom_name"][atom_index]).strip().upper(),
    )

  def atom_occupancy(atom_index: int) -> float:
    if "occupancy" not in annotation_names:
      return 1.0
    return float(deduplicated.atom_annotations["occupancy"][atom_index])

  def atom_altloc(atom_index: int) -> str:
    for field_name in altloc_annotation_names:
      if field_name in annotation_names:
        return str(deduplicated.atom_annotations[field_name][atom_index]).strip()
    return ""

  def altloc_rank(atom_index: int) -> tuple[int, str]:
    altloc = atom_altloc(atom_index)
    if not altloc:
      return (0, "")
    if altloc.upper() == "A":
      return (1, "")
    return (2, "")

  def prefer_atom_site(previous_index: int, candidate_index: int) -> bool:
    previous_occupancy = atom_occupancy(previous_index)
    candidate_occupancy = atom_occupancy(candidate_index)
    if candidate_occupancy > previous_occupancy:
      return True
    if candidate_occupancy < previous_occupancy:
      return False
    return altloc_rank(candidate_index) < altloc_rank(previous_index)

  selected_by_site: dict[tuple, int] = {}
  for atom_index in range(len(deduplicated)):
    atom_key = atom_site_key(atom_index)
    selected_index = selected_by_site.get(atom_key)
    if selected_index is None or prefer_atom_site(selected_index, atom_index):
      selected_by_site[atom_key] = atom_index

  keep_mask = np.zeros(len(deduplicated), dtype=bool)
  keep_mask[list(selected_by_site.values())] = True
  if not np.all(keep_mask):
    filter_structure_atoms(deduplicated, keep_mask)

  remove_altloc_annotations()
  return deduplicated


def add_terminal_capping_groups(
  structure: Structure,
  *,
  chains: Optional[Sequence[str]] = None,
  n_terminal: bool = True,
  c_terminal: bool = True,
) -> Structure:
  """Return a copy with ACE/NME caps added to protein chain termini.

  Adds an ``ACE`` heavy-atom cap to each selected chain N-terminus and an
  ``NME`` heavy-atom cap to each selected chain C-terminus. Coordinates are
  placed from terminal backbone geometry and should be relaxed by a molecular
  mechanics tool before downstream workflows that require optimized cap
  conformations.

  Parameters:
    structure: Input single-model structure.
    chains: Optional chain IDs to cap. If ``None``, all protein-containing
      chains are considered.
    n_terminal: Whether to add N-terminal ``ACE`` caps.
    c_terminal: Whether to add C-terminal ``NME`` caps.

  Returns:
    New :class:`Structure` containing the original atoms plus cap atoms and
    covalent cap bonds.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"add_terminal_capping_groups() expects a Structure, found {type(structure).__name__}.")
  if not isinstance(n_terminal, bool):
    raise TypeError(f"n_terminal must be a bool, found {type(n_terminal).__name__}.")
  if not isinstance(c_terminal, bool):
    raise TypeError(f"c_terminal must be a bool, found {type(c_terminal).__name__}.")
  if chains is not None:
    if isinstance(chains, str):
      chains = (chains,)
    else:
      chains = tuple(str(chain_id) for chain_id in chains)
    missing = sorted(set(chains) - set(structure.chain_ids()))
    if missing:
      raise ValueError(f"Chain(s) not found in structure: {', '.join(missing)}.")

  capped = Structure.empty_like(structure, copy_metadata=True)
  old_to_new: dict[int, int] = {}
  extra_bonds: list[tuple[int, int, int, BondType]] = []
  next_atom_id = _next_atom_id(structure)
  selected_chains = set(chains) if chains is not None else None
  for chain in structure.chains():
    chain_residues = chain.residues()
    protein_residues = [residue for residue in chain_residues if _is_protein_residue(residue)]
    should_cap_chain = selected_chains is None or chain.chain_id in selected_chains
    residue_names = {residue.res_name.strip().upper() for residue in chain_residues}
    n_terminal_residue = protein_residues[0] if protein_residues and should_cap_chain and n_terminal and "ACE" not in residue_names else None
    c_terminal_residue = protein_residues[-1] if protein_residues and should_cap_chain and c_terminal and "NME" not in residue_names else None

    for residue in chain_residues:
      if n_terminal_residue is not None and residue == n_terminal_residue:
        next_atom_id, ace_carbon_index, terminal_n_index = _append_ace_cap(capped, residue, next_atom_id, extra_bonds)
      _append_original_residue(capped, structure, residue, old_to_new)
      if c_terminal_residue is not None and residue == c_terminal_residue:
        next_atom_id = _append_nme_cap(capped, residue, next_atom_id, old_to_new, extra_bonds)
      if n_terminal_residue is not None and residue == n_terminal_residue:
        extra_bonds.append((ace_carbon_index, old_to_new[terminal_n_index], 1, BondType.COVALENT))

  _append_remapped_topology(capped, structure, old_to_new, extra_bonds)
  return capped


def _append_original_residue(capped: Structure, source: Structure, residue: Residue, old_to_new: dict[int, int]) -> None:
  atom_indices = residue.atom_indices()
  coords = [(float(source.atoms["x"][idx]), float(source.atoms["y"][idx]), float(source.atoms["z"][idx])) for idx in atom_indices]
  annotations = [_annotation_row_to_dict(source, idx) for idx in atom_indices]
  for old_index, new_index in zip(atom_indices, capped.add_atoms(coords, annotations)):
    old_to_new[old_index] = new_index


def _annotation_row_to_dict(structure: Structure, atom_index: int) -> dict:
  return {name: structure.atom_annotations[name][atom_index].item() for name in structure.atom_annotations.dtype.names}


def _append_remapped_topology(
  capped: Structure,
  source: Structure,
  old_to_new: dict[int, int],
  extra_bonds: list[tuple[int, int, int, BondType]],
) -> None:
  bond_rows = []
  for bond in source.bonds:
    atom_i = old_to_new.get(int(bond["atom_i"]))
    atom_j = old_to_new.get(int(bond["atom_j"]))
    if atom_i is None or atom_j is None:
      continue
    bond_rows.append((atom_i, atom_j, int(bond["bond_order"]), int(bond["bond_type"])))
  bond_rows.extend((atom_i, atom_j, bond_order, int(bond_type)) for atom_i, atom_j, bond_order, bond_type in extra_bonds)
  capped.add_bonds(bond_rows)

  interaction_rows = []
  for interaction in source.interactions:
    atom_i = old_to_new.get(int(interaction["atom_i"]))
    atom_j = old_to_new.get(int(interaction["atom_j"]))
    if atom_i is None or atom_j is None:
      continue
    interaction_rows.append((atom_i, atom_j, int(interaction["interaction_type"])))
  capped.add_interactions(interaction_rows)


def _next_atom_id(structure: Structure) -> int:
  if len(structure) == 0 or "atom_id" not in structure.atom_annotations.dtype.names:
    return 1
  return int(np.max(structure.atom_annotations["atom_id"])) + 1


def _is_protein_residue(residue: Residue) -> bool:
  res_name = residue.res_name.strip().upper()
  return AA_RECORDS_CANONICAL.get_by_abr(res_name) is not None or AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr(res_name) is not None


def _residue_atom_map(residue: Residue) -> dict[str, tuple[int, np.ndarray]]:
  atoms = residue.atoms()
  indices = residue.atom_indices()
  return {atom.atom_name.strip().upper(): (atom_index, atom.coord.astype(np.float32)) for atom_index, atom in zip(indices, atoms)}


def _unit_vector(vector: np.ndarray) -> np.ndarray:
  norm = float(np.linalg.norm(vector))
  if norm < 1e-6:
    raise ValueError("Cannot place terminal capping group from degenerate backbone coordinates.")
  return (vector / norm).astype(np.float32)


def _perpendicular_unit(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
  perp = reference - axis * float(np.dot(reference, axis))
  if float(np.linalg.norm(perp)) < 1e-6:
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, fallback))) > 0.9:
      fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    perp = fallback - axis * float(np.dot(fallback, axis))
  return _unit_vector(perp)


def _base_cap_annotations(structure: Structure, residue: Residue, res_name: str, atom_name: str, element: str, atom_id: int, res_id: int) -> dict:
  annotations = {
    "chain_id": residue.chain_id,
    "res_id": res_id,
    "ins_code": "",
    "res_name": res_name,
    "hetero": False,
    "atom_name": atom_name,
    "element": element,
    "atom_id": atom_id,
    "b_factor": 0.0,
    "occupancy": 1.0,
    "charge": 0,
  }
  return {name: value for name, value in annotations.items() if name in structure.atom_annotations.dtype.names}


def _append_ace_cap(
  structure: Structure,
  residue: Residue,
  next_atom_id: int,
  extra_bonds: list[tuple[int, int, int, BondType]],
) -> tuple[int, int, int]:
  atom_map = _residue_atom_map(residue)
  if "N" not in atom_map or "CA" not in atom_map:
    raise ValueError(f'Cannot add ACE cap to chain "{residue.chain_id}" residue {residue.res_id}: missing N or CA atom.')

  n_index, n_coord = atom_map["N"]
  _ca_index, ca_coord = atom_map["CA"]
  c_ref = atom_map.get("C", (None, ca_coord))[1]
  axis = _unit_vector(n_coord - ca_coord)
  perp = _perpendicular_unit(axis, c_ref - ca_coord)
  cap_c = n_coord + axis * 1.33
  cap_o = cap_c + perp * 1.23
  cap_ch3 = cap_c + axis * 1.50
  res_id = int(residue.res_id) - 1

  cap_indices = structure.add_atoms(
    [cap_ch3, cap_c, cap_o],
    [
      _base_cap_annotations(structure, residue, "ACE", "CH3", "C", next_atom_id, res_id),
      _base_cap_annotations(structure, residue, "ACE", "C", "C", next_atom_id + 1, res_id),
      _base_cap_annotations(structure, residue, "ACE", "O", "O", next_atom_id + 2, res_id),
    ],
  )
  extra_bonds.extend(
    (
      (cap_indices[0], cap_indices[1], 1, BondType.COVALENT),
      (cap_indices[1], cap_indices[2], 2, BondType.COVALENT),
    )
  )
  return next_atom_id + 3, cap_indices[1], n_index


def _append_nme_cap(
  structure: Structure,
  residue: Residue,
  next_atom_id: int,
  old_to_new: dict[int, int],
  extra_bonds: list[tuple[int, int, int, BondType]],
) -> int:
  atom_map = _residue_atom_map(residue)
  if "C" not in atom_map or "CA" not in atom_map:
    raise ValueError(f'Cannot add NME cap to chain "{residue.chain_id}" residue {residue.res_id}: missing C or CA atom.')

  c_index, c_coord = atom_map["C"]
  _ca_index, ca_coord = atom_map["CA"]
  axis = _unit_vector(c_coord - ca_coord)
  cap_n = c_coord + axis * 1.33
  cap_ch3 = cap_n + axis * 1.45
  res_id = int(residue.res_id) + 1

  cap_indices = structure.add_atoms(
    [cap_n, cap_ch3],
    [
      _base_cap_annotations(structure, residue, "NME", "N", "N", next_atom_id, res_id),
      _base_cap_annotations(structure, residue, "NME", "CH3", "C", next_atom_id + 1, res_id),
    ],
  )
  extra_bonds.extend(
    (
      (old_to_new[c_index], cap_indices[0], 1, BondType.COVALENT),
      (cap_indices[0], cap_indices[1], 1, BondType.COVALENT),
    )
  )
  return next_atom_id + 2


def add_hydrogens_with_pdb2pqr(
  structure: Structure,
  *,
  forcefield: str = "PARSE",
  ffout: Optional[str] = None,
  neutraln: bool = False,
  neutralc: bool = False,
  debump: bool = True,
) -> Structure:
  """Add hydrogens using the PDB2PQR preparation backend.

  This wrapper delegates to :func:`neurosnap.algos.pdb2pqr.assign_pqr` with
  ``assign_only=False`` and ``optimize=False``. PDB2PQR may still perform its
  internal water-specific hydrogen handling, but it skips the full optimization
  path used by :func:`optimize_hydrogens_with_pdb2pqr`.

  The returned structure is the PDB2PQR-rebuilt structure, so it also carries
  any charge/radius annotations and provenance metadata that backend emits.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"add_hydrogens_with_pdb2pqr() expects a Structure, found {type(structure).__name__}.")

  from neurosnap.algos.pdb2pqr import assign_pqr

  return assign_pqr(
    structure,
    forcefield=forcefield,
    ffout=ffout,
    neutraln=neutraln,
    neutralc=neutralc,
    assign_only=False,
    debump=debump,
    optimize=False,
  )


def optimize_hydrogens_with_pdb2pqr(
  structure: Structure,
  *,
  forcefield: str = "PARSE",
  ffout: Optional[str] = None,
  neutraln: bool = False,
  neutralc: bool = False,
  debump: bool = True,
) -> Structure:
  """Add and optimize hydrogens using the PDB2PQR preparation backend.

  This wrapper delegates to :func:`neurosnap.algos.pdb2pqr.assign_pqr` with
  ``assign_only=False`` and ``optimize=True``.

  The returned structure is the PDB2PQR-rebuilt structure, so it also carries
  any charge/radius annotations and provenance metadata that backend emits.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"optimize_hydrogens_with_pdb2pqr() expects a Structure, found {type(structure).__name__}.")

  from neurosnap.algos.pdb2pqr import assign_pqr

  return assign_pqr(
    structure,
    forcefield=forcefield,
    ffout=ffout,
    neutraln=neutraln,
    neutralc=neutralc,
    assign_only=False,
    debump=debump,
    optimize=True,
  )
