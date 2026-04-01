"""Chain-interface analysis helpers for Neurosnap structures."""

from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from neurosnap.constants import HYDROPHOBIC_RESIDUES

from ._common import available_chain_ids, resolve_model, residue_key
from .properties import _residue_surface_area_map
from .structure import Atom, Residue


def find_interface_contacts(
  structure,
  chain1: str,
  chain2: str,
  *,
  model: Optional[int] = None,
  cutoff: float = 4.5,
  hydrogens: bool = True,
) -> List[Tuple[Atom, Atom]]:
  """Identify atom-atom contacts between two chains using a distance cutoff."""
  structure_model = resolve_model(structure, model=model)
  _validate_chain_pair(structure_model, chain1, chain2)

  chain_lookup = {chain.chain_id: chain for chain in structure_model.chains()}
  atoms1 = [atom for residue in chain_lookup[chain1].residues() for atom in residue.atoms() if hydrogens or atom.element != "H"]
  atoms2 = [atom for residue in chain_lookup[chain2].residues() for atom in residue.atoms() if hydrogens or atom.element != "H"]
  return _find_contacts(atoms1, atoms2, cutoff=cutoff)


def find_interface_residues(
  structure,
  chain1: str,
  chain2: str,
  *,
  model: Optional[int] = None,
  cutoff: float = 4.5,
  hydrogens: bool = True,
) -> List[Tuple[Residue, Residue]]:
  """Identify unique residue-residue contacts between two chains."""
  structure_model = resolve_model(structure, model=model)
  _validate_chain_pair(structure_model, chain1, chain2)

  chain_lookup = {chain.chain_id: chain for chain in structure_model.chains()}
  residue_lookup1 = _residue_lookup(chain_lookup[chain1])
  residue_lookup2 = _residue_lookup(chain_lookup[chain2])

  residue_pairs = []
  seen = set()
  for atom1, atom2 in find_interface_contacts(structure_model, chain1, chain2, cutoff=cutoff, hydrogens=hydrogens):
    residue_key1 = (atom1.chain_id, atom1.res_id, atom1.ins_code)
    residue_key2 = (atom2.chain_id, atom2.res_id, atom2.ins_code)
    pair_key = residue_key1 + residue_key2
    if pair_key in seen:
      continue
    seen.add(pair_key)
    residue_pairs.append((residue_lookup1[residue_key1], residue_lookup2[residue_key2]))

  return residue_pairs


def find_non_interface_hydrophobic_patches(
  structure,
  chain_pairs: Iterable[Tuple[str, str]],
  target_chains: Optional[Iterable[str]] = None,
  *,
  model: Optional[int] = None,
  cutoff_interface: float = 4.5,
  hydrogens: bool = True,
  patch_cutoff: float = 6.0,
  min_patch_area: float = 40.0,
) -> List[Set[Residue]]:
  """Identify solvent-exposed hydrophobic patches outside specified interfaces."""
  structure_model = resolve_model(structure, model=model)
  chain_pairs = [(chain1.strip(), chain2.strip()) for chain1, chain2 in chain_pairs]
  available_chains = set(available_chain_ids(structure_model))
  for chain1, chain2 in chain_pairs:
    if chain1 not in available_chains or chain2 not in available_chains:
      raise ValueError(f"Chain pair ({chain1}, {chain2}) is not present in the selected model.")

  target_chain_set = None
  if target_chains is not None:
    target_chain_set = {chain_id.strip() for chain_id in target_chains}
    missing = target_chain_set - available_chains
    if missing:
      raise ValueError(f'Chain(s) {", ".join(sorted(missing))} were not found.')

  interface_residue_keys = set()
  for chain1, chain2 in chain_pairs:
    for atom1, atom2 in find_interface_contacts(
      structure_model,
      chain1,
      chain2,
      cutoff=cutoff_interface,
      hydrogens=hydrogens,
    ):
      interface_residue_keys.add((atom1.chain_id, atom1.res_id, atom1.ins_code, atom1.res_name, atom1.hetero))
      interface_residue_keys.add((atom2.chain_id, atom2.res_id, atom2.ins_code, atom2.res_name, atom2.hetero))

  residue_sasa = _residue_surface_area_map(structure_model)
  hydrophobic_residues = []
  hydrophobic_keys = []
  hydrophobic_ca_coords = []

  for chain in structure_model.chains():
    if target_chain_set is not None and chain.chain_id not in target_chain_set:
      continue
    for residue in chain.residues():
      residue_name = residue.res_name.strip().upper()
      if residue.hetero or residue_name not in HYDROPHOBIC_RESIDUES:
        continue
      key = residue_key(residue)
      if key in interface_residue_keys:
        continue
      if residue_sasa.get(key, 0.0) <= 0.01:
        continue
      ca_atom = _atom_by_name(residue, "CA")
      if ca_atom is None:
        continue
      hydrophobic_residues.append(residue)
      hydrophobic_keys.append(key)
      hydrophobic_ca_coords.append(ca_atom.coord)

  if not hydrophobic_ca_coords:
    return []

  coord = np.asarray(hydrophobic_ca_coords, dtype=np.float32)
  neighbor_lists = [[] for _ in range(len(coord))]
  for atom_index in range(len(coord)):
    for neighbor_index in range(atom_index + 1, len(coord)):
      if np.linalg.norm(coord[atom_index] - coord[neighbor_index]) <= patch_cutoff:
        neighbor_lists[atom_index].append(neighbor_index)
        neighbor_lists[neighbor_index].append(atom_index)

  patches = []
  visited = [False] * len(coord)
  for atom_index in range(len(coord)):
    if visited[atom_index]:
      continue
    stack = [atom_index]
    component = []
    while stack:
      current_index = stack.pop()
      if visited[current_index]:
        continue
      visited[current_index] = True
      component.append(current_index)
      stack.extend(neighbor_lists[current_index])

    if len(component) <= 1:
      continue

    component_area = sum(float(residue_sasa.get(hydrophobic_keys[index], 0.0)) for index in component)
    if component_area >= min_patch_area:
      patches.append({hydrophobic_residues[index] for index in component})

  return patches


def _validate_chain_pair(structure_model, chain1: str, chain2: str):
  """Validate that both chains are present in a selected model."""
  available_chains = set(available_chain_ids(structure_model))
  if chain1 not in available_chains:
    raise ValueError(f"Chain {chain1} was not found.")
  if chain2 not in available_chains:
    raise ValueError(f"Chain {chain2} was not found.")


def _find_contacts(atoms1: List[Atom], atoms2: List[Atom], cutoff: float) -> List[Tuple[Atom, Atom]]:
  """Return atom pairs within a cutoff distance."""
  if not atoms1 or not atoms2:
    return []

  coords2 = np.asarray([atom.coord for atom in atoms2], dtype=np.float32)
  contacts = []
  for atom1 in atoms1:
    distances = np.linalg.norm(coords2 - atom1.coord, axis=1)
    for atom2_index in np.where(distances <= cutoff)[0]:
      contacts.append((atom1, atoms2[atom2_index]))
  return contacts


def _residue_lookup(chain) -> dict[tuple[str, int, str], Residue]:
  """Return residue lookup for a chain keyed by chain ID, residue ID, and insertion code."""
  return {(residue.chain_id, residue.res_id, residue.ins_code): residue for residue in chain.residues()}


def _atom_by_name(residue: Residue, atom_name: str) -> Optional[Atom]:
  """Return an atom from a residue by atom name."""
  atom_name = atom_name.strip().upper()
  for atom in residue.atoms():
    if atom.atom_name.strip().upper() == atom_name:
      return atom
  return None
