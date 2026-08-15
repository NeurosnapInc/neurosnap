"""Native Structure-first EvoEF2-style missing-atom reconstruction.

This backend rebuilds missing heavy atoms and hydrogens directly against the
native :class:`Structure` representation. It reuses EvoEF2's bundled topology
and parameter readers, but it does not construct or depend on EvoEF2's runtime
Atom/Residue/Chain compatibility model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from neurosnap.constants.chemistry import ATOMIC_MASSES
from neurosnap.constants.sequence import AA_RECORDS_CANONICAL, AA_RECORDS_FORCEFIELD_VARIANTS
from neurosnap.constants.structure import NA_ALL_CODES, NA_RESIDUE_MAP
from neurosnap.log import logger

from .structure import BondType, Structure


def rebuild_missing_atoms_with_evoef2_backend(
  structure: Structure,
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
) -> Structure:
  """Rebuild missing heavy atoms and hydrogens using bundled EvoEF2 data."""
  if not isinstance(structure, Structure):
    raise TypeError(f"rebuild_missing_atoms_with_evoef2() expects a Structure, found {type(structure).__name__}.")

  source_df = structure.to_dataframe()
  if source_df.empty:
    empty = Structure(remove_annotations=False)
    empty.metadata = dict(structure.metadata)
    return empty

  from neurosnap.algos import evoef2

  params = evoef2.load_atom_params(param_path)
  topologies = evoef2.load_topology(topo_path)
  if any(_is_nucleic_res_name(str(name)) for name in source_df["res_name"].unique()):
    na_params, na_topologies = evoef2._load_na_bundle()
    params = {**params, **na_params}
    topologies = {**topologies, **na_topologies}

  chains = _build_residue_state(source_df, params, topologies)
  _rebuild_missing_coordinates(chains, topologies)
  rebuilt = _build_native_structure(chains, structure, source_df)
  rebuilt.metadata = dict(structure.metadata)
  return rebuilt


def _is_nucleic_res_name(res_name: str) -> bool:
  return res_name in NA_RESIDUE_MAP or res_name in NA_ALL_CODES


def _normalize_internal_residue_name(res_name: str) -> str:
  """Map public residue names onto EvoEF2/CHARMM residue identifiers."""
  if res_name in {"HIE", "HSE", "HISE"}:
    return "HSE"
  if res_name in {"HIP", "HSP", "HISH"}:
    return "HSP"
  if res_name in {"HIS", "HID", "HSD", "HISD"}:
    return "HSD"
  if res_name in {"ASH", "ASPH", "ASPP"}:
    return "ASP"
  if res_name in {"GLH", "GLUH", "GLUP"}:
    return "GLU"
  if res_name in {"CYSH", "CYM", "CYX", "CYS2"}:
    return "CYS"
  if res_name in {"LYN", "LYSN"}:
    return "LYS"
  if res_name in {"ARN", "ARGN"}:
    return "ARG"
  if res_name == "CSE":
    return "SEC"
  return NA_RESIDUE_MAP.get(res_name, res_name)


def _protein_one_letter_code(res_name: str) -> Optional[str]:
  """Return the canonical one-letter code for a protein residue name."""
  record = AA_RECORDS_CANONICAL.get_by_abr(res_name) or AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr(res_name)
  if record is None or record.code is None:
    return None
  return record.code


def _xyz_angle(v1: np.ndarray, v2: np.ndarray) -> float:
  norm = np.linalg.norm(v1) * np.linalg.norm(v2)
  if norm < 1e-12:
    return 1000.0
  cos_value = float(np.dot(v1, v2) / norm)
  return math.acos(max(-1.0, min(1.0, cos_value)))


def _xyz_rotate_around(point: np.ndarray, axis_from: np.ndarray, axis_to: np.ndarray, angle: float) -> np.ndarray:
  """Rotate a point around an axis defined by two points."""
  s = math.sin(angle)
  c = math.cos(angle)
  axis = axis_from - axis_to
  norm = np.linalg.norm(axis)
  if norm < 1e-12:
    return point.copy()
  axis = axis / norm
  translated = point - axis_from
  a00 = axis[0] * axis[0] + (1 - axis[0] * axis[0]) * c
  a01 = axis[0] * axis[1] * (1 - c) - axis[2] * s
  a02 = axis[0] * axis[2] * (1 - c) + axis[1] * s
  a10 = axis[0] * axis[1] * (1 - c) + axis[2] * s
  a11 = axis[1] * axis[1] + (1 - axis[1] * axis[1]) * c
  a12 = axis[1] * axis[2] * (1 - c) - axis[0] * s
  a20 = axis[0] * axis[2] * (1 - c) - axis[1] * s
  a21 = axis[1] * axis[2] * (1 - c) + axis[0] * s
  a22 = axis[2] * axis[2] + (1 - axis[2] * axis[2]) * c
  matrix = np.array([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]], dtype=float)
  return translated @ matrix + axis_from


def _get_fourth_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray, ic_param: Sequence[float]) -> np.ndarray:
  """Compute atom D coordinates from three references and one CHARMM IC row."""
  ba = b - a
  bc = b - c
  ba_x_bc = np.cross(ba, bc)
  if np.linalg.norm(ba_x_bc) < 1e-12:
    raise ValueError("Zero division in GetFourthAtom")
  angle_abc = _xyz_angle(ba, bc)
  d = _xyz_rotate_around(a, b, ba_x_bc + b, ic_param[3] - (math.pi - angle_abc))
  d = _xyz_rotate_around(d, b, c, ic_param[2])
  d = d - b
  d = d / np.linalg.norm(d) * ic_param[4]
  return d + c


def _build_residue_state(source_df, params, topologies) -> List[dict]:
  """Build ordered chain/residue state directly from the native atom dataframe."""
  chains: List[dict] = []
  source_df = source_df.reset_index(drop=True)
  source_df["source_index"] = np.arange(len(source_df), dtype=int)

  for chain_id in source_df["chain"].drop_duplicates().tolist():
    chain_df = source_df[source_df["chain"] == chain_id]
    chain_state = {"chain_id": str(chain_id), "residues": []}
    current_key = None
    current_rows = []
    for row in chain_df.itertuples(index=False):
      row_key = (int(row.res_id), str(row.ins_code), str(row.res_name), bool(row.hetero))
      if current_key is None:
        current_key = row_key
      if row_key != current_key:
        chain_state["residues"].append(_build_residue(current_rows, params, topologies))
        current_rows = []
        current_key = row_key
      current_rows.append(row)
    if current_rows:
      chain_state["residues"].append(_build_residue(current_rows, params, topologies))

    if chain_state["residues"]:
      _apply_chain_terminus_patches(chain_state, params, topologies)
      chains.append(chain_state)
  return chains


def _build_residue(current_rows, params, topologies) -> dict:
  """Build one native residue state dictionary from source atom rows."""
  first = current_rows[0]
  output_name = str(first.res_name)
  internal_name = _normalize_internal_residue_name(output_name)
  residue = {
    "chain_id": str(first.chain),
    "res_id": int(first.res_id),
    "ins_code": str(first.ins_code),
    "output_name": output_name,
    "internal_name": internal_name,
    "hetero": bool(first.hetero),
    "is_protein": _protein_one_letter_code(internal_name) is not None,
    "is_nucleic": _is_nucleic_res_name(internal_name),
    "patches": [],
    "atoms": {},
    "bonds": [],
  }

  if internal_name in params:
    for atom_name, param in params[internal_name].items():
      residue["atoms"][atom_name] = {"param": param, "xyz": None, "is_xyz_valid": False, "source_index": None}
  elif output_name != "HOH":
    logger.warning("No EvoEF2 parameters for residue %s; only existing atoms will be preserved.", output_name)

  topology = topologies.get(internal_name)
  if topology is not None:
    residue["bonds"] = list(topology.bonds)

  for row in current_rows:
    atom_name = str(row.atom_name)
    atom_state = residue["atoms"].setdefault(atom_name, {"param": None, "xyz": None, "is_xyz_valid": False, "source_index": None})
    if atom_state["param"] is None and internal_name in params and atom_name in params[internal_name]:
      atom_state["param"] = params[internal_name][atom_name]
    atom_state["xyz"] = np.array([float(row.x), float(row.y), float(row.z)], dtype=float)
    atom_state["is_xyz_valid"] = True
    atom_state["source_index"] = int(row.source_index)
  return residue


def _apply_chain_terminus_patches(chain_state: dict, params, topologies) -> None:
  """Apply EvoEF2-like protein terminus patches to a chain."""
  residues = chain_state["residues"]
  if not residues or not residues[0]["is_protein"]:
    return
  _patch_terminus(residues[0], params, topologies, "NTER")
  if "HT1" in residues[0]["atoms"] or "HN1" in residues[0]["atoms"]:
    _patch_terminus(residues[-1], params, topologies, "CTER")


def _apply_patch(residue: dict, patch_name: str, params, topologies, *, delete_o: bool = True) -> None:
  """Apply one EvoEF2 topology patch to a residue state."""
  topology = topologies.get(patch_name)
  if topology is None:
    raise ValueError(f"Missing topology for patch {patch_name}")

  for atom_name in topology.deletes:
    if not delete_o and atom_name == "O":
      continue
    residue["atoms"].pop(atom_name, None)

  if patch_name in params:
    for atom_name, param in params[patch_name].items():
      atom_state = residue["atoms"].get(atom_name)
      if atom_state is None:
        residue["atoms"][atom_name] = {"param": param, "xyz": None, "is_xyz_valid": False, "source_index": None}
      else:
        atom_state["param"] = param

  residue["patches"].insert(0, patch_name)
  residue["bonds"].extend(topology.bonds)
  filtered_bonds = []
  for bond in residue["bonds"]:
    if not (bond.a.startswith(("+", "-")) or bond.a in residue["atoms"]):
      continue
    if not (bond.b.startswith(("+", "-")) or bond.b in residue["atoms"]):
      continue
    filtered_bonds.append(bond)
  residue["bonds"] = filtered_bonds


def _patch_terminus(residue: dict, params, topologies, terminus: str) -> None:
  """Apply the bundled N- or C-terminus patch rules."""
  if terminus == "NTER":
    if residue["internal_name"] == "GLY":
      _apply_patch(residue, "GLYP", params, topologies)
    elif residue["internal_name"] == "PRO":
      _apply_patch(residue, "PROP", params, topologies)
    else:
      _apply_patch(residue, "NTER", params, topologies)
    delete_prefix = "-"
  elif terminus == "CTER":
    _apply_patch(residue, "CTER", params, topologies, delete_o=False)
    delete_prefix = "+"
  else:
    raise ValueError(f"Unknown terminus patch {terminus}")

  removed = False
  filtered_bonds = []
  for bond in residue["bonds"]:
    if not removed and (bond.a.startswith(delete_prefix) or bond.b.startswith(delete_prefix)):
      removed = True
      continue
    filtered_bonds.append(bond)
  residue["bonds"] = filtered_bonds


def _find_ic_for_atom(residue: dict, topologies, atom_name: str):
  """Return the IC row that builds ``atom_name`` if one exists."""
  for patch_name in residue["patches"]:
    patch_topology = topologies.get(patch_name)
    if patch_topology is None:
      continue
    for ic in patch_topology.ics:
      if ic.atom_d == atom_name:
        return ic
  topology = topologies.get(residue["internal_name"])
  if topology is None:
    return None
  for ic in topology.ics:
    if ic.atom_d == atom_name:
      return ic
  return None


def _get_atom_xyz(residue: dict, atom_name: str) -> Optional[np.ndarray]:
  atom_state = residue["atoms"].get(atom_name)
  if atom_state is None or not atom_state["is_xyz_valid"]:
    return None
  return atom_state["xyz"]


def _calc_atom_xyz(residue: dict, topologies, prev_residue: Optional[dict], next_residue: Optional[dict], atom_name: str) -> Optional[np.ndarray]:
  """Compute one missing atom from its CHARMM IC definition."""
  ic = _find_ic_for_atom(residue, topologies, atom_name)
  if ic is None:
    return None

  coords = []
  for ref_name in (ic.atom_a, ic.atom_b, ic.atom_c):
    if ref_name.startswith("-"):
      xyz = None if prev_residue is None else _get_atom_xyz(prev_residue, ref_name[1:])
    elif ref_name.startswith("+"):
      xyz = None if next_residue is None else _get_atom_xyz(next_residue, ref_name[1:])
    else:
      xyz = _get_atom_xyz(residue, ref_name)
    if xyz is None:
      return None
    coords.append(xyz)

  try:
    return _get_fourth_atom(coords[0], coords[1], coords[2], ic.ic_param)
  except ValueError:
    return None


def _rebuild_missing_coordinates(chains: List[dict], topologies) -> None:
  """Iteratively rebuild all missing atoms in chain order."""
  for chain in chains:
    residues = chain["residues"]
    for residue_index, residue in enumerate(residues):
      prev_residue = residues[residue_index - 1] if residue_index > 0 else None
      next_residue = residues[residue_index + 1] if residue_index + 1 < len(residues) else None
      _rebuild_residue_atoms(residue, topologies, prev_residue, next_residue)


def _rebuild_residue_atoms(residue: dict, topologies, prev_residue: Optional[dict], next_residue: Optional[dict]) -> None:
  """Repeatedly try to fill in every missing atom for one residue."""
  made_progress = True
  while made_progress:
    made_progress = False
    for atom_name, atom_state in residue["atoms"].items():
      if atom_state["is_xyz_valid"]:
        continue
      new_xyz = _calc_atom_xyz(residue, topologies, prev_residue, next_residue, atom_name)
      if new_xyz is None:
        continue
      atom_state["xyz"] = new_xyz
      atom_state["is_xyz_valid"] = True
      made_progress = True


def _element_from_atom_state(atom_state: dict, fallback: str = "") -> str:
  """Return a normalized element symbol from EvoEF2 atom parameters."""
  param = atom_state.get("param")
  if param is not None and param.element:
    return param.element
  if fallback:
    normalized = fallback.strip().title()
    if normalized in ATOMIC_MASSES:
      return normalized.upper()
  return ""


def _build_native_structure(chains: List[dict], source_structure: Structure, source_df) -> Structure:
  """Convert rebuilt residue state back into a native Structure."""
  result = Structure.empty_like(source_structure)

  source_df = source_df.reset_index(drop=True)
  source_keys = [
    (
      str(source_df.iloc[idx]["chain"]),
      int(source_df.iloc[idx]["res_id"]),
      str(source_df.iloc[idx]["ins_code"]),
      bool(source_df.iloc[idx]["hetero"]),
      str(source_df.iloc[idx]["atom_name"]),
    )
    for idx in range(len(source_df))
  ]
  next_atom_id = 1
  if len(source_structure) and "atom_id" in source_structure.atom_annotations.dtype.names:
    next_atom_id = int(np.max(source_structure.atom_annotations["atom_id"])) + 1

  atom_index_by_key: Dict[Tuple[str, int, str, bool, str], int] = {}
  residue_atom_names: Dict[Tuple[str, int, str, bool, str], set] = {}
  coord_rows = []
  annotation_rows = []
  atom_keys = []

  for chain in chains:
    for residue in chain["residues"]:
      residue_key = (residue["chain_id"], residue["res_id"], residue["ins_code"], residue["hetero"], residue["output_name"])
      residue_atom_names[residue_key] = set()
      for atom_name, atom_state in residue["atoms"].items():
        if not atom_state["is_xyz_valid"]:
          continue

        source_index = atom_state["source_index"]
        fallback_element = ""
        if source_index is None:
          annotations = {"atom_id": next_atom_id} if "atom_id" in result.annotation_names else {}
          if annotations:
            next_atom_id += 1
        else:
          row = source_structure.atom_annotations[source_index].copy()
          annotations = {name: row[name] for name in result.annotation_names}
          fallback_element = str(row["element"])

        element = _element_from_atom_state(atom_state, fallback_element)
        annotations.update(
          {
            "chain_id": residue["chain_id"],
            "res_id": residue["res_id"],
            "ins_code": residue["ins_code"],
            "res_name": residue["output_name"],
            "hetero": residue["hetero"],
            "atom_name": atom_name,
            "element": element,
          }
        )

        coord_rows.append((float(atom_state["xyz"][0]), float(atom_state["xyz"][1]), float(atom_state["xyz"][2])))
        annotation_rows.append(annotations)
        atom_key = (residue["chain_id"], residue["res_id"], residue["ins_code"], residue["hetero"], atom_name)
        atom_keys.append(atom_key)
        residue_atom_names[residue_key].add(atom_name)

  for atom_key, atom_index in zip(atom_keys, result.add_atoms(coord_rows, annotation_rows)):
    atom_index_by_key[atom_key] = atom_index

  seen_bonds = set()
  bond_rows = []
  for chain in chains:
    for residue in chain["residues"]:
      residue_key = (residue["chain_id"], residue["res_id"], residue["ins_code"], residue["hetero"], residue["output_name"])
      present_names = residue_atom_names.get(residue_key, set())
      for bond in residue["bonds"]:
        if bond.a.startswith(("+", "-")) or bond.b.startswith(("+", "-")):
          continue
        if bond.a not in present_names or bond.b not in present_names:
          continue
        key_a = (residue["chain_id"], residue["res_id"], residue["ins_code"], residue["hetero"], bond.a)
        key_b = (residue["chain_id"], residue["res_id"], residue["ins_code"], residue["hetero"], bond.b)
        atom_i = atom_index_by_key.get(key_a)
        atom_j = atom_index_by_key.get(key_b)
        if atom_i is None or atom_j is None:
          continue
        atom_i, atom_j = sorted((atom_i, atom_j))
        if atom_i == atom_j or (atom_i, atom_j) in seen_bonds:
          continue
        bond_rows.append((atom_i, atom_j, 1, BondType.COVALENT))
        seen_bonds.add((atom_i, atom_j))

  for bond in source_structure.bonds:
    atom_i = int(bond["atom_i"])
    atom_j = int(bond["atom_j"])
    if atom_i >= len(source_keys) or atom_j >= len(source_keys):
      continue
    remapped_i = atom_index_by_key.get(source_keys[atom_i])
    remapped_j = atom_index_by_key.get(source_keys[atom_j])
    if remapped_i is None or remapped_j is None:
      continue
    remapped_i, remapped_j = sorted((remapped_i, remapped_j))
    if remapped_i == remapped_j or (remapped_i, remapped_j) in seen_bonds:
      continue
    bond_rows.append((remapped_i, remapped_j, int(bond["bond_order"]), int(bond["bond_type"])))
    seen_bonds.add((remapped_i, remapped_j))
  result.add_bonds(bond_rows)

  seen_interactions = set()
  interaction_rows = []
  for interaction in source_structure.interactions:
    atom_i = int(interaction["atom_i"])
    atom_j = int(interaction["atom_j"])
    if atom_i >= len(source_keys) or atom_j >= len(source_keys):
      continue
    remapped_i = atom_index_by_key.get(source_keys[atom_i])
    remapped_j = atom_index_by_key.get(source_keys[atom_j])
    if remapped_i is None or remapped_j is None:
      continue
    remapped_i, remapped_j = sorted((remapped_i, remapped_j))
    interaction_type = int(interaction["interaction_type"])
    row_key = (remapped_i, remapped_j, interaction_type)
    if remapped_i == remapped_j or row_key in seen_interactions:
      continue
    interaction_rows.append((remapped_i, remapped_j, interaction_type))
    seen_interactions.add(row_key)
  result.add_interactions(interaction_rows)

  result._remove_empty_annotations()
  return result
