"""Parser for mmCIF coordinate files.

This module provides a Neurosnap-native :func:`parse_mmcif` helper for reading
mmCIF files into :class:`~neurosnap.structure.structure.Structure`,
:class:`~neurosnap.structure.structure.StructureEnsemble`, and
:class:`~neurosnap.structure.structure.StructureStack` objects.

Parsing follows the atom-site driven approach used by BioPython's
``MMCIFParser`` while building Neurosnap structures directly in a more
Biotite-like array-oriented form.
"""

import io
import pathlib
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from neurosnap.log import logger
from neurosnap.structure.structure import Structure, StructureEnsemble, StructureStack

__all__ = ["parse_mmcif"]

ReturnType = Literal["ensemble", "stack", "auto"]

_MISSING_VALUES = {"", ".", "?"}


def _normalize_mmcif_value(value: object) -> str:
  """Return a stripped mmCIF token with missing-value sentinels collapsed."""
  text = "" if value is None else str(value).strip()
  return "" if text in _MISSING_VALUES else text


def _parse_mmcif_int(value: object, field_name: str, row_number: int, *, required: bool) -> Optional[int]:
  """Parse an integer token from an mmCIF atom-site column."""
  text = _normalize_mmcif_value(value)
  if not text:
    if required:
      raise ValueError(f'Missing required mmCIF field "{field_name}" at atom row {row_number}.')
    return None
  try:
    return int(float(text))
  except ValueError as exc:
    raise ValueError(f'Invalid integer value "{text}" for mmCIF field "{field_name}" at atom row {row_number}.') from exc


def _parse_mmcif_float(value: object, field_name: str, row_number: int, *, required: bool, default: float = 0.0) -> float:
  """Parse a float token from an mmCIF atom-site column."""
  text = _normalize_mmcif_value(value)
  if not text:
    if required:
      raise ValueError(f'Missing required mmCIF field "{field_name}" at atom row {row_number}.')
    return float(default)
  try:
    return float(text)
  except ValueError as exc:
    raise ValueError(f'Invalid float value "{text}" for mmCIF field "{field_name}" at atom row {row_number}.') from exc


def parse_mmcif(
  mmcif: Union[str, pathlib.Path, io.IOBase],
  return_type: ReturnType = "auto",
) -> Union[StructureEnsemble, StructureStack]:
  """Parse an mmCIF file into Neurosnap structure containers.

  The parser reads the ``_atom_site`` loop directly, using author-provided
  chain IDs and residue numbering when available and falling back to label
  identifiers otherwise. Parsed models are first collected into a
  :class:`StructureEnsemble` and are optionally converted into a
  :class:`StructureStack` at the end.

  Alternate locations are always ignored. When alternate locations are present,
  the parser keeps only the highest-occupancy conformer for each atom site and
  emits a :func:`logger.warning` so the user knows this happened.

  Unlike :func:`neurosnap.io.pdb.parse_pdb`, this first mmCIF implementation
  does not currently parse explicit bond tables such as ``_struct_conn``.
  Parsed structures therefore keep full atom annotations but default to an
  empty bond list unless bonds are added later by other code.

  Parameters:
    mmcif: mmCIF filepath or open file handle.
    return_type: Output container type. ``"ensemble"`` always returns a
      :class:`StructureEnsemble`, ``"stack"`` requires stack-compatible models,
      and ``"auto"`` returns a :class:`StructureStack` when possible or falls
      back to a :class:`StructureEnsemble`.

  Returns:
    A :class:`StructureEnsemble` or :class:`StructureStack` depending on
    ``return_type`` and model compatibility.
  """
  if return_type not in {"ensemble", "stack", "auto"}:
    raise ValueError('return_type must be one of "ensemble", "stack", or "auto".')

  mmcif_dict = MMCIF2Dict(mmcif)
  atom_groups = mmcif_dict.get("_atom_site.group_PDB")
  if not atom_groups:
    raise ValueError('No "_atom_site" coordinate records were found in the mmCIF file.')

  required_columns = {
    "_atom_site.id",
    "_atom_site.type_symbol",
    "_atom_site.label_atom_id",
    "_atom_site.label_comp_id",
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
  }
  missing_columns = sorted(column for column in required_columns if column not in mmcif_dict)
  if missing_columns:
    raise ValueError(f'Missing required mmCIF atom-site column(s): {", ".join(missing_columns)}.')

  row_count = len(atom_groups)
  for column_name, values in mmcif_dict.items():
    if column_name.startswith("_atom_site.") and len(values) != row_count:
      raise ValueError(f'MMCIF atom-site column "{column_name}" has {len(values)} values, but "_atom_site.group_PDB" has {row_count}.')

  def column(name: str, default: Optional[str] = None) -> List[str]:
    if name in mmcif_dict:
      return list(mmcif_dict[name])
    fallback = "" if default is None else default
    return [fallback] * row_count

  atom_ids = column("_atom_site.id")
  elements = column("_atom_site.type_symbol")
  atom_names = column("_atom_site.label_atom_id")
  altlocs = column("_atom_site.label_alt_id")
  label_res_names = column("_atom_site.label_comp_id")
  auth_res_names = column("_atom_site.auth_comp_id")
  label_seq_ids = column("_atom_site.label_seq_id")
  auth_seq_ids = column("_atom_site.auth_seq_id")
  insertion_codes = column("_atom_site.pdbx_PDB_ins_code")
  label_chain_ids = column("_atom_site.label_asym_id")
  auth_chain_ids = column("_atom_site.auth_asym_id")
  xs = column("_atom_site.Cartn_x")
  ys = column("_atom_site.Cartn_y")
  zs = column("_atom_site.Cartn_z")
  occupancies = column("_atom_site.occupancy", default="1.0")
  b_factors = column("_atom_site.B_iso_or_equiv", default="0.0")
  model_ids = column("_atom_site.pdbx_PDB_model_num", default="1")
  charges = column("_atom_site.pdbx_formal_charge", default="0")

  altloc_sites: set[tuple[int, tuple[str, int, str, str, bool, str]]] = set()
  model_order: List[int] = []
  model_builders: Dict[int, Dict[str, object]] = {}
  implicit_residue_state: Dict[int, Dict[str, object]] = {}

  def get_builder(model_id: int) -> Dict[str, object]:
    builder = model_builders.get(model_id)
    if builder is not None:
      return builder

    model_order.append(model_id)
    builder = {
      "atoms": [],
      "annotations": {
        "chain_id": [],
        "res_id": [],
        "ins_code": [],
        "res_name": [],
        "hetero": [],
        "atom_name": [],
        "element": [],
        "atom_id": [],
        "b_factor": [],
        "occupancy": [],
        "charge": [],
        "sym_id": [],
      },
      "atom_key_to_index": {},
      "selected_altloc": {},
    }
    model_builders[model_id] = builder
    return builder

  for atom_row_index in range(row_count):
    row_number = atom_row_index + 1
    group_pdb = _normalize_mmcif_value(atom_groups[atom_row_index]).upper()
    if group_pdb not in {"ATOM", "HETATM"}:
      continue

    model_id = _parse_mmcif_int(model_ids[atom_row_index], "_atom_site.pdbx_PDB_model_num", row_number, required=False)
    if model_id is None:
      model_id = 1

    chain_id = _normalize_mmcif_value(auth_chain_ids[atom_row_index]) or _normalize_mmcif_value(label_chain_ids[atom_row_index])
    res_name = _normalize_mmcif_value(auth_res_names[atom_row_index]) or _normalize_mmcif_value(label_res_names[atom_row_index])
    atom_name = _normalize_mmcif_value(atom_names[atom_row_index])
    element = _normalize_mmcif_value(elements[atom_row_index]).upper()
    insertion_code = _normalize_mmcif_value(insertion_codes[atom_row_index])
    altloc = _normalize_mmcif_value(altlocs[atom_row_index])
    hetero = group_pdb == "HETATM"

    if not chain_id:
      raise ValueError(f"Missing chain identifier in mmCIF atom row {row_number}.")
    if not res_name:
      raise ValueError(f"Missing residue name in mmCIF atom row {row_number}.")
    if not atom_name:
      raise ValueError(f"Missing atom name in mmCIF atom row {row_number}.")
    if not element:
      raise ValueError(f"Missing element in mmCIF atom row {row_number}.")
    if len(chain_id) > 4:
      raise ValueError(f'Chain ID "{chain_id}" exceeds the supported 4-character Neurosnap limit.')
    if len(res_name) > 5:
      raise ValueError(f'Residue name "{res_name}" exceeds the supported 5-character Neurosnap limit.')
    if len(atom_name) > 6:
      raise ValueError(f'Atom name "{atom_name}" exceeds the supported 6-character Neurosnap limit.')
    if len(element) > 2:
      raise ValueError(f'Element "{element}" exceeds the supported 2-character Neurosnap limit.')
    if len(insertion_code) > 1:
      raise ValueError(f'Insertion code "{insertion_code}" exceeds the supported 1-character Neurosnap limit.')

    res_id = _parse_mmcif_int(auth_seq_ids[atom_row_index], "_atom_site.auth_seq_id", row_number, required=False)
    if res_id is None:
      res_id = _parse_mmcif_int(label_seq_ids[atom_row_index], "_atom_site.label_seq_id", row_number, required=False)
    if res_id is None:
      # Non-polymer mmCIF rows can omit both auth/label residue numbers. In
      # that case preserve residue blocks by assigning synthetic IDs in row
      # order for each model.
      residue_signature = (
        chain_id,
        _normalize_mmcif_value(label_chain_ids[atom_row_index]),
        res_name,
        insertion_code,
        hetero,
      )
      state = implicit_residue_state.setdefault(model_id, {"counter": 0, "last_signature": None})
      if state["last_signature"] != residue_signature:
        state["counter"] += 1
        state["last_signature"] = residue_signature
      res_id = int(state["counter"])

    atom_id = _parse_mmcif_int(atom_ids[atom_row_index], "_atom_site.id", row_number, required=True)
    occupancy = _parse_mmcif_float(occupancies[atom_row_index], "_atom_site.occupancy", row_number, required=False, default=1.0)
    b_factor = _parse_mmcif_float(b_factors[atom_row_index], "_atom_site.B_iso_or_equiv", row_number, required=False, default=0.0)
    charge = _parse_mmcif_int(charges[atom_row_index], "_atom_site.pdbx_formal_charge", row_number, required=False)
    x = _parse_mmcif_float(xs[atom_row_index], "_atom_site.Cartn_x", row_number, required=True)
    y = _parse_mmcif_float(ys[atom_row_index], "_atom_site.Cartn_y", row_number, required=True)
    z = _parse_mmcif_float(zs[atom_row_index], "_atom_site.Cartn_z", row_number, required=True)

    atom_key = (chain_id, res_id, insertion_code, res_name, hetero, atom_name)
    if altloc:
      altloc_sites.add((model_id, atom_key))

    builder = get_builder(model_id)
    atom_index = builder["atom_key_to_index"].get(atom_key)
    if atom_index is None:
      atom_index = len(builder["atoms"])
      builder["atom_key_to_index"][atom_key] = atom_index
      builder["selected_altloc"][atom_key] = (occupancy, altloc)
      builder["atoms"].append((x, y, z))
      builder["annotations"]["chain_id"].append(chain_id)
      builder["annotations"]["res_id"].append(res_id)
      builder["annotations"]["ins_code"].append(insertion_code)
      builder["annotations"]["res_name"].append(res_name)
      builder["annotations"]["hetero"].append(hetero)
      builder["annotations"]["atom_name"].append(atom_name)
      builder["annotations"]["element"].append(element)
      builder["annotations"]["atom_id"].append(atom_id)
      builder["annotations"]["b_factor"].append(b_factor)
      builder["annotations"]["occupancy"].append(occupancy)
      builder["annotations"]["charge"].append(0 if charge is None else charge)
      builder["annotations"]["sym_id"].append("")
      continue

    previous_occupancy, previous_altloc = builder["selected_altloc"][atom_key]
    should_replace = occupancy > previous_occupancy or (occupancy == previous_occupancy and previous_altloc and not altloc)
    if not should_replace:
      continue

    builder["selected_altloc"][atom_key] = (occupancy, altloc)
    builder["atoms"][atom_index] = (x, y, z)
    builder["annotations"]["chain_id"][atom_index] = chain_id
    builder["annotations"]["res_id"][atom_index] = res_id
    builder["annotations"]["ins_code"][atom_index] = insertion_code
    builder["annotations"]["res_name"][atom_index] = res_name
    builder["annotations"]["hetero"][atom_index] = hetero
    builder["annotations"]["atom_name"][atom_index] = atom_name
    builder["annotations"]["element"][atom_index] = element
    builder["annotations"]["atom_id"][atom_index] = atom_id
    builder["annotations"]["b_factor"][atom_index] = b_factor
    builder["annotations"]["occupancy"][atom_index] = occupancy
    builder["annotations"]["charge"][atom_index] = 0 if charge is None else charge
    builder["annotations"]["sym_id"][atom_index] = ""

  if not model_order:
    raise ValueError("No models or atoms were found in the mmCIF file.")

  ensemble = StructureEnsemble()
  for model_id in model_order:
    builder = model_builders[model_id]
    structure = Structure(remove_annotations=False)
    structure.metadata = {"model_id": model_id}

    if builder["atoms"]:
      structure.atoms = np.array(builder["atoms"], dtype=structure._dtype_atoms)
      structure.atom_annotations = np.empty(len(builder["atoms"]), dtype=structure._dtype_atom_annotations)
      for field_name, values in builder["annotations"].items():
        field_dtype = structure._dtype_atom_annotations.fields[field_name][0]
        structure.atom_annotations[field_name] = np.asarray(values, dtype=field_dtype)
    else:
      structure.atoms = np.zeros(0, dtype=structure._dtype_atoms)
      structure.atom_annotations = np.zeros(0, dtype=structure._dtype_atom_annotations)

    structure.bonds = np.zeros(0, dtype=structure._dtype_bond)
    structure._remove_empty_annotations()
    ensemble.append(structure, model_id=model_id)

  ensemble.metadata["source_format"] = "mmcif"

  if altloc_sites:
    logger.warning(
      "Ignoring alternate locations for %d atom site(s); using the highest-occupancy conformer for each.",
      len(altloc_sites),
    )

  if return_type == "ensemble":
    return ensemble
  if return_type == "stack":
    return StructureStack.from_ensemble(ensemble)

  try:
    return StructureStack.from_ensemble(ensemble)
  except ValueError:
    return ensemble
