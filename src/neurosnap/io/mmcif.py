"""Parser and writer for mmCIF coordinate files.

This module provides Neurosnap-native :func:`parse_mmcif` and
:func:`save_cif` helpers for reading and writing
:class:`~neurosnap.structure.structure.Structure`,
:class:`~neurosnap.structure.structure.StructureEnsemble`, and
:class:`~neurosnap.structure.structure.StructureStack` objects.

Parsing follows the atom-site driven mmCIF loop structure while building
Neurosnap structures directly in an array-oriented form.
"""

import io
import pathlib
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np

from neurosnap.log import logger
from neurosnap.structure.structure import Structure, StructureEnsemble, StructureStack

__all__ = ["parse_mmcif", "save_cif"]

ReturnType = Literal["ensemble", "stack", "auto"]

_MISSING_VALUES = {"", ".", "?"}


def _read_mmcif_text(mmcif: Union[str, pathlib.Path, io.IOBase]) -> str:
  """Return mmCIF text from a filepath or open file handle."""
  if isinstance(mmcif, io.IOBase):
    content = mmcif.read()
  else:
    with open(mmcif, encoding="utf-8") as handle:
      content = handle.read()

  if isinstance(content, bytes):
    content = content.decode("utf-8")
  if not content:
    raise ValueError("Empty file.")
  return content


def _split_mmcif_line(line: str) -> Iterator[str]:
  """Yield mmCIF tokens from a single line, handling quotes and comments."""
  quote_chars = {"'", '"'}
  whitespace_chars = {" ", "\t"}
  in_token = False
  quote_open_char = None
  start_index = 0

  for index, char in enumerate(line):
    if char in whitespace_chars:
      if in_token and quote_open_char is None:
        in_token = False
        yield line[start_index:index]
    elif char in quote_chars:
      if quote_open_char is None and not in_token:
        quote_open_char = char
        in_token = True
        start_index = index + 1
      elif char == quote_open_char and (index + 1 == len(line) or line[index + 1] in whitespace_chars):
        quote_open_char = None
        in_token = False
        yield line[start_index:index]
    elif char == "#" and not in_token:
      return
    elif not in_token:
      in_token = True
      start_index = index

  if in_token:
    yield line[start_index:]
  if quote_open_char is not None:
    raise ValueError(f"Line ended with quote open: {line}")


def _tokenize_mmcif(text: str) -> Iterator[str]:
  """Yield tokens from mmCIF text, including loop blocks and multiline values."""
  lines = io.StringIO(text)
  empty = True
  for line in lines:
    empty = False
    if line.startswith("#"):
      continue
    if line.startswith(";"):
      token_buffer = [line[1:].rstrip()]
      for line in lines:
        line = line.rstrip()
        if line.startswith(";"):
          yield "\n".join(token_buffer)
          line = line[1:]
          if line and line[0] not in {" ", "\t"}:
            raise ValueError("Missing whitespace after closing semicolon for multiline mmCIF value.")
          break
        token_buffer.append(line)
      else:
        raise ValueError("Missing closing semicolon for multiline mmCIF value.")

    yield from _split_mmcif_line(line.strip())

  if empty:
    raise ValueError("Empty file.")


def _parse_mmcif_dict(text: str) -> Dict[str, Union[str, List[str]]]:
  """Parse mmCIF text into a dictionary of scalar values and loop columns."""
  tokens = _tokenize_mmcif(text)
  try:
    first_token = next(tokens)
  except StopIteration as exc:
    raise ValueError("Empty file.") from exc

  if not first_token.startswith("data_"):
    raise ValueError("The input mmCIF file must begin with a 'data_' directive.")

  mmcif_dict: Dict[str, Union[str, List[str]]] = {first_token[0:5]: first_token[5:]}
  loop_flag = False
  pending_key: Optional[str] = None
  loop_keys: List[str] = []
  loop_key_count = 0
  loop_value_index = 0

  for token in tokens:
    if token.lower() == "loop_":
      loop_flag = True
      loop_keys = []
      loop_key_count = 0
      loop_value_index = 0
      pending_key = None
      continue

    if loop_flag:
      if token.startswith("_") and (loop_key_count == 0 or loop_value_index % loop_key_count == 0):
        if loop_value_index > 0:
          loop_flag = False
        else:
          mmcif_dict[token] = []
          loop_keys.append(token)
          loop_key_count += 1
          continue

      if loop_flag:
        if loop_key_count == 0:
          raise ValueError("mmCIF loop_ block does not define any keys before values.")
        column_key = loop_keys[loop_value_index % loop_key_count]
        mmcif_dict[column_key].append(token)
        loop_value_index += 1
        continue

    if pending_key is None:
      pending_key = token
    else:
      mmcif_dict[pending_key] = [token]
      pending_key = None

  if pending_key is not None:
    raise ValueError(f'mmCIF key "{pending_key}" is missing a value.')
  return mmcif_dict


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

  mmcif_dict = _parse_mmcif_dict(_read_mmcif_text(mmcif))
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


def save_cif(structure: Union[Structure, StructureEnsemble, StructureStack], cif: Union[str, pathlib.Path, io.IOBase]):
  """Save a Neurosnap structure container as an mmCIF file.

  Parameters:
    structure: Structure container to write.
    cif: Output filepath or open file handle.

  Notes:
    The writer emits a compact ``_atom_site`` loop that round-trips through
    :func:`parse_mmcif`. Multi-model outputs are represented using the
    ``_atom_site.pdbx_PDB_model_num`` column. This initial writer does not yet
    export bond tables such as ``_struct_conn``.
  """
  models = _models_for_cif_output(structure)
  if not models:
    raise ValueError("No models are available for mmCIF output.")

  lines = [
    "data_neurosnap",
    "#",
    "loop_",
    "_atom_site.group_PDB",
    "_atom_site.id",
    "_atom_site.type_symbol",
    "_atom_site.label_atom_id",
    "_atom_site.label_alt_id",
    "_atom_site.label_comp_id",
    "_atom_site.label_seq_id",
    "_atom_site.auth_seq_id",
    "_atom_site.pdbx_PDB_ins_code",
    "_atom_site.label_asym_id",
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
    "_atom_site.occupancy",
    "_atom_site.label_entity_id",
    "_atom_site.auth_asym_id",
    "_atom_site.auth_comp_id",
    "_atom_site.B_iso_or_equiv",
    "_atom_site.pdbx_PDB_model_num",
    "_atom_site.pdbx_formal_charge",
  ]

  entity_ids: Dict[str, int] = {}
  next_entity_id = 1
  for _model_id, model in models:
    for chain_id in model.atom_annotations["chain_id"]:
      chain_id = str(chain_id)
      if chain_id not in entity_ids:
        entity_ids[chain_id] = next_entity_id
        next_entity_id += 1

  for model_id, model in models:
    atom_ids = _atom_ids_for_model(model)
    for atom_index in range(len(model)):
      chain_id = str(model.atom_annotations["chain_id"][atom_index])
      res_id = int(model.atom_annotations["res_id"][atom_index])
      ins_code = _annotation_value_for_cif(model, "ins_code", atom_index, "")
      res_name = _annotation_value_for_cif(model, "res_name", atom_index, "")
      atom_name = _annotation_value_for_cif(model, "atom_name", atom_index, "")
      element = str(_annotation_value_for_cif(model, "element", atom_index, "")).upper()
      occupancy = float(_annotation_value_for_cif(model, "occupancy", atom_index, 1.0))
      b_factor = float(_annotation_value_for_cif(model, "b_factor", atom_index, 0.0))
      charge = _annotation_value_for_cif(model, "charge", atom_index, None)
      hetero = bool(model.atom_annotations["hetero"][atom_index])

      lines.append(
        " ".join(
          [
            "HETATM" if hetero else "ATOM",
            str(int(atom_ids[atom_index])),
            _format_mmcif_token(element),
            _format_mmcif_token(atom_name),
            ".",
            _format_mmcif_token(res_name),
            str(res_id),
            str(res_id),
            _format_mmcif_token(ins_code or "?"),
            _format_mmcif_token(chain_id),
            f"{float(model.atoms['x'][atom_index]):.6f}",
            f"{float(model.atoms['y'][atom_index]):.6f}",
            f"{float(model.atoms['z'][atom_index]):.6f}",
            f"{occupancy:.3f}",
            str(entity_ids[chain_id]),
            _format_mmcif_token(chain_id),
            _format_mmcif_token(res_name),
            f"{b_factor:.3f}",
            str(model_id),
            _format_mmcif_token("?" if charge is None else int(charge)),
          ]
        )
      )

  lines.append("#")
  _write_cif_lines(cif, lines)


def _models_for_cif_output(structure: Union[Structure, StructureEnsemble, StructureStack]) -> List[Tuple[int, Structure]]:
  """Return a normalized list of ``(model_id, model)`` pairs for writing."""
  if isinstance(structure, Structure):
    model_id = int(structure.metadata.get("model_id", 1))
    return [(model_id, structure)]
  if isinstance(structure, StructureEnsemble):
    return list(zip(structure.model_ids, structure.models()))
  if isinstance(structure, StructureStack):
    return list(zip(structure.model_ids, structure.models()))
  raise TypeError(f"Unsupported structure type for mmCIF output: {type(structure).__name__}.")


def _atom_ids_for_model(model: Structure) -> np.ndarray:
  """Return atom IDs for a model, preserving them when possible."""
  if "atom_id" in model.atom_annotations.dtype.names:
    atom_ids = np.asarray(model.atom_annotations["atom_id"], dtype=np.int32)
    if atom_ids.size and np.all(atom_ids > 0) and len(np.unique(atom_ids)) == len(atom_ids):
      return atom_ids.copy()
  return np.arange(1, len(model) + 1, dtype=np.int32)


def _annotation_value_for_cif(model: Structure, name: str, atom_index: int, default):
  """Return an annotation value with a fallback default for mmCIF output."""
  if name not in model.atom_annotations.dtype.names:
    return default
  value = model.atom_annotations[name][atom_index]
  if isinstance(value, np.generic):
    return value.item()
  return value


def _format_mmcif_token(value: object) -> str:
  """Return a safely tokenized mmCIF value."""
  text = "" if value is None else str(value)
  if not text or text in _MISSING_VALUES:
    return "?"
  if any(char.isspace() for char in text) or "'" in text or '"' in text or text.startswith("_") or text.startswith("#") or text.startswith(";"):
    if '"' not in text:
      return f'"{text}"'
    if "'" not in text:
      return f"'{text}'"
  return text


def _write_cif_lines(cif: Union[str, pathlib.Path, io.IOBase], lines: List[str]):
  """Write text lines to a filepath or file-like object."""
  text = "\n".join(lines) + "\n"
  if isinstance(cif, io.IOBase):
    cif.write(text)
    return

  with open(cif, "w", encoding="utf-8") as handle:
    handle.write(text)
