"""Chain-interface analysis helpers for Neurosnap structures."""

import copy
from typing import Iterable, List, Optional, Tuple

import numpy as np

from neurosnap.constants import HYDROPHOBIC_RESIDUES

from ._common import available_chain_ids, filter_structure_atoms, residue_key
from .analysis import _residue_surface_area_map, calculate_surface_area
from .structure import Atom, Residue, Structure


def find_contacts(atoms1: List[Atom], atoms2: List[Atom], cutoff: float = 4.5) -> List[Tuple[Atom, Atom]]:
  """Identify atom-atom contacts between two atom sets using a distance cutoff.

  Parameters:
    atoms1: First set of atoms.
    atoms2: Second set of atoms.
    cutoff: Distance cutoff in Å.

  Returns:
    List of ``(atom1, atom2)`` pairs within the cutoff distance.
  """
  if not atoms1 or not atoms2:
    return []

  coords2 = np.asarray([np.asarray(atom.coord, dtype=np.float32) for atom in atoms2], dtype=np.float32)
  contacts = []
  for atom1 in atoms1:
    distances = np.linalg.norm(coords2 - np.asarray(atom1.coord, dtype=np.float32), axis=1)
    for atom2_index in np.where(distances <= cutoff)[0]:
      contacts.append((atom1, atoms2[atom2_index]))
  return contacts


def calculate_bsa(
  structure: Structure,
  chain_group_1: List[str],
  chain_group_2: List[str],
  *,
  level: str = "R",
) -> float:
  """Calculate buried surface area between two chain groups.

  The buried surface area (BSA) is computed as:
    ``(SASA(group 1) + SASA(group 2)) - SASA(complex)``

  Parameters:
    structure: Input complex as a single-model :class:`Structure`.
    chain_group_1: Chain IDs for the first group.
    chain_group_2: Chain IDs for the second group.
    level: Surface-area aggregation level forwarded to
      :func:`calculate_surface_area`.

  Returns:
    Buried surface area in Å².
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"calculate_bsa() expects a Structure, found {type(structure).__name__}.")

  all_chains = {chain.chain_id for chain in structure.chains()}
  if not chain_group_1 or not chain_group_2:
    raise ValueError("Chain groups cannot be empty.")
  if not set(chain_group_1).isdisjoint(chain_group_2):
    raise ValueError("Chain groups must not overlap.")
  if set(chain_group_1).union(set(chain_group_2)) != all_chains:
    raise ValueError("Chain groups must cover all chains.")

  sasa_complex = calculate_surface_area(structure, level=level)
  group_structures = []
  for keep_chains in (set(chain_group_1), set(chain_group_2)):
    # BSA is defined against the isolated partners, so each group gets a
    # filtered copy of the complex with only its chains retained.
    group_structure = Structure(remove_annotations=False)
    group_structure._dtype_atoms = structure._dtype_atoms
    group_structure._dtype_atom_annotations = structure._dtype_atom_annotations
    group_structure._dtype_bond = structure._dtype_bond
    group_structure.atoms = structure.atoms.copy()
    group_structure.atom_annotations = structure.atom_annotations.copy()
    group_structure.bonds = structure.bonds.copy()
    group_structure.metadata = copy.deepcopy(structure.metadata)
    filter_structure_atoms(group_structure, np.isin(group_structure.atom_annotations["chain_id"], list(keep_chains)))
    group_structures.append(group_structure)

  group1_structure, group2_structure = group_structures
  sasa_group1 = calculate_surface_area(group1_structure, level=level)
  sasa_group2 = calculate_surface_area(group2_structure, level=level)
  return float((sasa_group1 + sasa_group2) - sasa_complex)


def find_interface_contacts(
  structure: Structure,
  chain1: str,
  chain2: str,
  *,
  cutoff: float = 4.5,
  hydrogens: bool = True,
) -> List[Tuple[Atom, Atom]]:
  """Identify atom-atom contacts between two chains using a distance cutoff.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain1: First chain ID.
    chain2: Second chain ID.
    cutoff: Contact cutoff distance in Å.
    hydrogens: Whether hydrogen atoms should be included.

  Returns:
    List of contacting ``(atom1, atom2)`` pairs.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"find_interface_contacts() expects a Structure, found {type(structure).__name__}.")

  available_chains = {chain.chain_id for chain in structure.chains()}
  if chain1 not in available_chains:
    raise ValueError(f"Chain {chain1} was not found.")
  if chain2 not in available_chains:
    raise ValueError(f"Chain {chain2} was not found.")

  chain_lookup = {chain.chain_id: chain for chain in structure.chains()}
  atoms1 = [atom for residue in chain_lookup[chain1].residues() for atom in residue.atoms() if hydrogens or atom.element != "H"]
  atoms2 = [atom for residue in chain_lookup[chain2].residues() for atom in residue.atoms() if hydrogens or atom.element != "H"]
  return find_contacts(atoms1, atoms2, cutoff=cutoff)


def find_interface_residues(
  structure: Structure,
  chain1: str,
  chain2: str,
  *,
  cutoff: float = 4.5,
  hydrogens: bool = True,
) -> List[Tuple[Residue, Residue]]:
  """Identify unique residue-residue contacts between two chains.

  Multiple atom-atom contacts between the same residue pair are collapsed into
  one output pair.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain1: First chain ID.
    chain2: Second chain ID.
    cutoff: Contact cutoff distance in Å.
    hydrogens: Whether hydrogen atoms should be included in the contact search.

  Returns:
    List of unique contacting ``(residue1, residue2)`` pairs.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"find_interface_residues() expects a Structure, found {type(structure).__name__}.")

  available_chains = {chain.chain_id for chain in structure.chains()}
  if chain1 not in available_chains:
    raise ValueError(f"Chain {chain1} was not found.")
  if chain2 not in available_chains:
    raise ValueError(f"Chain {chain2} was not found.")

  chain_lookup = {chain.chain_id: chain for chain in structure.chains()}
  residue_lookup1 = {(residue.chain_id, residue.res_id, residue.ins_code): residue for residue in chain_lookup[chain1].residues()}
  residue_lookup2 = {(residue.chain_id, residue.res_id, residue.ins_code): residue for residue in chain_lookup[chain2].residues()}

  residue_pairs = []
  seen = set()
  for atom1, atom2 in find_interface_contacts(structure, chain1, chain2, cutoff=cutoff, hydrogens=hydrogens):
    residue_key1 = (atom1.chain_id, atom1.res_id, atom1.ins_code)
    residue_key2 = (atom2.chain_id, atom2.res_id, atom2.ins_code)
    pair_key = residue_key1 + residue_key2
    if pair_key in seen:
      continue
    seen.add(pair_key)
    residue_pairs.append((residue_lookup1[residue_key1], residue_lookup2[residue_key2]))

  return residue_pairs


def find_non_interface_hydrophobic_patches(
  structure: Structure,
  chain_pairs: Iterable[Tuple[str, str]],
  target_chains: Optional[Iterable[str]] = None,
  *,
  cutoff_interface: float = 4.5,
  hydrogens: bool = True,
  patch_cutoff: float = 6.0,
  min_patch_area: float = 40.0,
) -> List[List[Residue]]:
  """Identify solvent-exposed hydrophobic patches outside specified interfaces.

  Hydrophobic residues are first filtered to remove interface residues and
  buried residues, then clustered by CA-CA proximity into connected
  components.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain_pairs: Iterable of chain-ID pairs whose interfaces should be excluded
      from patch detection.
    target_chains: Optional chain IDs to search for patches. If ``None``, all
      chains are considered.
    cutoff_interface: Distance cutoff in Å used to classify interface contacts.
    hydrogens: Whether hydrogen atoms should be included in the interface
      contact search.
    patch_cutoff: CA-CA distance cutoff in Å used to connect hydrophobic
      residues into the same patch.
    min_patch_area: Minimum summed SASA in Å² required for a connected
      component to be returned.

  Returns:
    List of residue lists, where each list represents one hydrophobic patch.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"find_non_interface_hydrophobic_patches() expects a Structure, found {type(structure).__name__}.")

  chain_pairs = [(chain1.strip(), chain2.strip()) for chain1, chain2 in chain_pairs]
  available_chains = set(available_chain_ids(structure))
  for chain1, chain2 in chain_pairs:
    if chain1 not in available_chains or chain2 not in available_chains:
      raise ValueError(f"Chain pair ({chain1}, {chain2}) is not present in the structure.")

  target_chain_set = None
  if target_chains is not None:
    target_chain_set = {chain_id.strip() for chain_id in target_chains}
    missing = target_chain_set - available_chains
    if missing:
      raise ValueError(f'Chain(s) {", ".join(sorted(missing))} were not found.')

  interface_residue_keys = set()
  for chain1, chain2 in chain_pairs:
    for atom1, atom2 in find_interface_contacts(
      structure,
      chain1,
      chain2,
      cutoff=cutoff_interface,
      hydrogens=hydrogens,
    ):
      interface_residue_keys.add((atom1.chain_id, atom1.res_id, atom1.ins_code, atom1.res_name, atom1.hetero))
      interface_residue_keys.add((atom2.chain_id, atom2.res_id, atom2.ins_code, atom2.res_name, atom2.hetero))

  residue_sasa = _residue_surface_area_map(structure)
  hydrophobic_residues = []
  hydrophobic_keys = []
  hydrophobic_ca_coords = []

  for chain in structure.chains():
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
      ca_atom = next((atom for atom in residue.atoms() if atom.atom_name.strip().upper() == "CA"), None)
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
    # Simple depth-first search over the residue-contact graph.
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
      patches.append([hydrophobic_residues[index] for index in component])

  return patches
