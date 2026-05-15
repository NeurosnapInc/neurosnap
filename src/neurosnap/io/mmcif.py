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
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union, Set

import numpy as np

from neurosnap.log import logger
from neurosnap.structure.structure import Structure, StructureEnsemble, StructureStack, _classify_polymer_residue

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

  altloc_sites: Set[Tuple[int, Tuple[str, int, str, str, bool, str]]] = set()
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


def save_cif(structure: Union[Structure, StructureEnsemble, StructureStack], cif: Union[str, pathlib.Path, io.IOBase], minimal: bool = False):
  """Save a Neurosnap structure container as an mmCIF file.

  Parameters:
    structure: Structure container to write.
    cif: Output filepath or open file handle.
    minimal: If ``True``, emit the legacy compact atom-site-only mmCIF output.
      If ``False`` (default), include entity/polymer/subchain metadata for
      broader downstream parser compatibility.

  Notes:
    Multi-model outputs are represented using the
    ``_atom_site.pdbx_PDB_model_num`` column. The writer does not yet export
    bond tables such as ``_struct_conn``.
  """
  models = _models_for_cif_output(structure)
  if not models:
    raise ValueError("No models are available for mmCIF output.")

  chain_metadata = _build_cif_chain_metadata(models)
  lines = ["data_neurosnap", "#"]

  if not minimal:
    _append_cif_entity_metadata(lines, chain_metadata)

  lines.extend(
    [
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
  )

  for model_id, model in models:
    atom_ids = _atom_ids_for_model(model)
    residue_label_seq_ids = _residue_label_seq_ids_for_model(model)
    for atom_index in range(len(model)):
      chain_id = str(model.atom_annotations["chain_id"][atom_index])
      chain_info = chain_metadata[chain_id]
      res_id = int(model.atom_annotations["res_id"][atom_index])
      ins_code = _annotation_value_for_cif(model, "ins_code", atom_index, "")
      res_name = _annotation_value_for_cif(model, "res_name", atom_index, "")
      atom_name = _annotation_value_for_cif(model, "atom_name", atom_index, "")
      element = str(_annotation_value_for_cif(model, "element", atom_index, "")).upper()
      occupancy = float(_annotation_value_for_cif(model, "occupancy", atom_index, 1.0))
      b_factor = float(_annotation_value_for_cif(model, "b_factor", atom_index, 0.0))
      charge = _annotation_value_for_cif(model, "charge", atom_index, None)
      hetero = bool(model.atom_annotations["hetero"][atom_index])
      residue_key = (chain_id, res_id, str(ins_code), str(res_name), hetero)
      label_seq_id = res_id if minimal else residue_label_seq_ids.get(residue_key, ".")

      lines.append(
        " ".join(
          [
            "HETATM" if hetero else "ATOM",
            str(int(atom_ids[atom_index])),
            _format_mmcif_token(element),
            _format_mmcif_token(atom_name),
            ".",
            _format_mmcif_token(res_name),
            str(label_seq_id),
            str(res_id),
            _format_mmcif_token(ins_code or "?"),
            _format_mmcif_token(chain_info["label_asym_id"]),
            f"{float(model.atoms['x'][atom_index]):.6f}",
            f"{float(model.atoms['y'][atom_index]):.6f}",
            f"{float(model.atoms['z'][atom_index]):.6f}",
            f"{occupancy:.3f}",
            str(chain_info["entity_id"]),
            _format_mmcif_token(chain_info["auth_asym_id"]),
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


def _build_cif_chain_metadata(models: List[Tuple[int, Structure]]) -> Dict[str, Dict[str, object]]:
  """Return per-chain metadata used by full mmCIF output."""
  chain_metadata: Dict[str, Dict[str, object]] = {}
  next_entity_id = 1
  generated_chain_index = 1

  for _model_id, model in models:
    for chain in model.chains():
      if chain.chain_id in chain_metadata:
        continue

      label_asym_id = chain.chain_id
      auth_asym_id = chain.chain_id
      if not label_asym_id:
        label_asym_id = f"CHAIN{generated_chain_index}"
        auth_asym_id = label_asym_id
        generated_chain_index += 1

      polymer_type = _chain_polymer_type(chain)
      chain_metadata[chain.chain_id] = {
        "entity_id": next_entity_id,
        "label_asym_id": label_asym_id,
        "auth_asym_id": auth_asym_id,
        "polymer_type": polymer_type,
        "polymer_residues": _chain_polymer_residues(chain, polymer_type),
      }
      next_entity_id += 1

  return chain_metadata


def _append_cif_entity_metadata(lines: List[str], chain_metadata: Dict[str, Dict[str, object]]):
  """Append full entity/polymer/asym metadata blocks to an mmCIF output."""
  lines.extend(
    [
      "loop_",
      "_entity.id",
      "_entity.type",
      "_entity.src_method",
      "_entity.pdbx_description",
      "_entity.formula_weight",
      "_entity.pdbx_number_of_molecules",
      "_entity.details",
    ]
  )
  for chain_info in chain_metadata.values():
    chain_label = str(chain_info["auth_asym_id"])
    entity_type = "polymer" if chain_info["polymer_type"] is not None else "non-polymer"
    lines.append(f'{chain_info["entity_id"]} {entity_type} man {_format_mmcif_token(f"Chain {chain_label}")} . 1 .')
  lines.append("#")

  polymer_entities = [chain_info for chain_info in chain_metadata.values() if chain_info["polymer_type"] is not None]
  if polymer_entities:
    lines.extend(
      [
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
        "_entity_poly.nstd_linkage",
        "_entity_poly.nstd_monomer",
        "_entity_poly.pdbx_strand_id",
        "_entity_poly.pdbx_seq_one_letter_code",
        "_entity_poly.pdbx_seq_one_letter_code_can",
      ]
    )
    for chain_info in polymer_entities:
      sequence_code = _entity_poly_sequence_code(chain_info["polymer_residues"], str(chain_info["polymer_type"]))
      lines.extend(
        [
          (
            f'{chain_info["entity_id"]} {_mmcif_entity_poly_type(str(chain_info["polymer_type"]))} '
            f'no no {_format_mmcif_token(str(chain_info["auth_asym_id"]))}'
          ),
          f";{sequence_code}",
          ";",
          f";{sequence_code}",
          ";",
        ]
      )
    lines.append("#")

    lines.extend(
      [
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.hetero",
      ]
    )
    for chain_info in polymer_entities:
      for seq_index, residue in enumerate(chain_info["polymer_residues"], start=1):
        lines.append(f'{chain_info["entity_id"]} {seq_index} {_format_mmcif_token(residue.res_name)} .')
    lines.append("#")

  lines.extend(
    [
      "loop_",
      "_struct_asym.id",
      "_struct_asym.entity_id",
      "_struct_asym.details",
    ]
  )
  for chain_info in chain_metadata.values():
    chain_label = str(chain_info["auth_asym_id"])
    lines.append(f'{_format_mmcif_token(str(chain_info["label_asym_id"]))} {chain_info["entity_id"]} {_format_mmcif_token(f"Chain {chain_label}")}')
  lines.append("#")


def _chain_polymer_type(chain) -> Optional[str]:
  """Return a normalized polymer type for a chain."""
  polymer_types = {polymer_type for residue in chain.residues() if not residue.hetero for polymer_type in [_classify_polymer_residue(residue)] if polymer_type is not None}
  if not polymer_types:
    return None
  if len(polymer_types) > 1:
    chain_label = chain.chain_id or "<blank>"
    raise ValueError(f'Chain "{chain_label}" mixes incompatible polymer residue types for mmCIF output.')
  return next(iter(polymer_types))


def _chain_polymer_residues(chain, polymer_type: Optional[str]):
  """Return polymer residues in atom-table order for a chain."""
  if polymer_type is None:
    return []
  return [residue for residue in chain.residues() if not residue.hetero and _classify_polymer_residue(residue) == polymer_type]


def _entity_poly_sequence_code(polymer_residues, polymer_type: str) -> str:
  """Return a conservative one-letter-style sequence code for ``_entity_poly``."""
  residue_tokens: List[str] = []
  for residue in polymer_residues:
    residue_name = residue.res_name.strip().upper()
    if polymer_type == "protein":
      residue_tokens.append(_protein_sequence_token(residue_name))
    elif polymer_type == "dna":
      residue_tokens.append(_dna_sequence_token(residue_name))
    else:
      residue_tokens.append(_rna_sequence_token(residue_name))
  return "".join(residue_tokens) or "?"


def _protein_sequence_token(residue_name: str) -> str:
  """Return a one-letter or CCD token for a protein residue."""
  if residue_name == "ALA":
    return "A"
  if residue_name == "ARG":
    return "R"
  if residue_name == "ASN":
    return "N"
  if residue_name == "ASP":
    return "D"
  if residue_name == "CYS":
    return "C"
  if residue_name == "GLN":
    return "Q"
  if residue_name == "GLU":
    return "E"
  if residue_name == "GLY":
    return "G"
  if residue_name == "HIS":
    return "H"
  if residue_name == "ILE":
    return "I"
  if residue_name == "LEU":
    return "L"
  if residue_name == "LYS":
    return "K"
  if residue_name == "MET":
    return "M"
  if residue_name == "PHE":
    return "F"
  if residue_name == "PRO":
    return "P"
  if residue_name == "SER":
    return "S"
  if residue_name == "THR":
    return "T"
  if residue_name == "TRP":
    return "W"
  if residue_name == "TYR":
    return "Y"
  if residue_name == "VAL":
    return "V"
  return f"({residue_name})"


def _dna_sequence_token(residue_name: str) -> str:
  """Return a one-letter or CCD token for a DNA residue."""
  if residue_name == "DA":
    return "A"
  if residue_name == "DC":
    return "C"
  if residue_name == "DG":
    return "G"
  if residue_name == "DT":
    return "T"
  return f"({residue_name})"


def _rna_sequence_token(residue_name: str) -> str:
  """Return a one-letter or CCD token for an RNA residue."""
  if residue_name in {"A", "C", "G", "U"}:
    return residue_name
  return f"({residue_name})"


def _mmcif_entity_poly_type(polymer_type: str) -> str:
  """Return the mmCIF ``_entity_poly.type`` label for a polymer."""
  if polymer_type == "protein":
    return "polypeptide(L)"
  if polymer_type == "dna":
    return "polydeoxyribonucleotide"
  if polymer_type == "rna":
    return "polyribonucleotide"
  raise ValueError(f'Unsupported polymer type "{polymer_type}".')


def _residue_label_seq_ids_for_model(model: Structure) -> Dict[Tuple[str, int, str, str, bool], int]:
  """Return ``_atom_site.label_seq_id`` values keyed by residue identity."""
  label_seq_ids: Dict[Tuple[str, int, str, str, bool], int] = {}
  for chain in model.chains():
    seq_index = 1
    for residue in chain.residues():
      if residue.hetero or _classify_polymer_residue(residue) is None:
        continue
      label_seq_ids[residue.key()] = seq_index
      seq_index += 1
  return label_seq_ids


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
