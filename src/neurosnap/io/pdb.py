"""Parser and writer for PDB files.

This module provides Neurosnap-native :func:`parse_pdb` and :func:`save_pdb`
helpers for reading and writing
:class:`~neurosnap.structure.structure.Structure`,
:class:`~neurosnap.structure.structure.StructureEnsemble`, and
:class:`~neurosnap.structure.structure.StructureStack` objects.
"""

import io
from collections import Counter
from dataclasses import dataclass, field
import pathlib
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np

from neurosnap.log import logger
from neurosnap.structure.structure import Structure, StructureEnsemble, StructureStack

__all__ = ["parse_pdb", "save_pdb"]

ReturnType = Literal["ensemble", "stack", "auto"]
AtomKey = Tuple[str, int, str, str, bool, str]
AltlocSite = Tuple[int, AtomKey]
ConectRecord = Tuple[int, List[int], int]


@dataclass(slots=True)
class _AtomRecord:
  """Internal representation of a parsed PDB atom record."""

  x: float
  y: float
  z: float
  chain_id: str
  res_id: int
  ins_code: str
  res_name: str
  hetero: bool
  atom_name: str
  element: str
  atom_id: int
  b_factor: float
  occupancy: float
  charge: int
  altloc: str
  atom_key: AtomKey


@dataclass(slots=True)
class _ModelAccumulator:
  """Mutable builder used while parsing a single PDB model."""

  model_id: int
  atoms: List[Tuple[float, float, float]] = field(default_factory=list)
  annotations: Dict[str, List[object]] = field(
    default_factory=lambda: {
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
    }
  )
  serial_to_index: Dict[int, int] = field(default_factory=dict)
  _atom_key_to_index: Dict[AtomKey, int] = field(default_factory=dict)
  _selected_altloc: Dict[AtomKey, Tuple[float, str, int]] = field(default_factory=dict)
  directed_bonds: Counter[Tuple[int, int]] = field(default_factory=Counter)

  def add_atom(self, atom: _AtomRecord):
    """Add or replace a selected atom record in the current model.

    Alternate locations collapse onto the same ``atom_key`` so the model keeps
    only one conformer per atom site.
    """
    atom_index = self._atom_key_to_index.get(atom.atom_key)
    if atom_index is None:
      atom_index = len(self.atoms)
      self._atom_key_to_index[atom.atom_key] = atom_index
      self.atoms.append((atom.x, atom.y, atom.z))
      self.annotations["chain_id"].append(atom.chain_id)
      self.annotations["res_id"].append(atom.res_id)
      self.annotations["ins_code"].append(atom.ins_code)
      self.annotations["res_name"].append(atom.res_name)
      self.annotations["hetero"].append(atom.hetero)
      self.annotations["atom_name"].append(atom.atom_name)
      self.annotations["element"].append(atom.element)
      self.annotations["atom_id"].append(atom.atom_id)
      self.annotations["b_factor"].append(atom.b_factor)
      self.annotations["occupancy"].append(atom.occupancy)
      self.annotations["charge"].append(atom.charge)
      self.annotations["sym_id"].append("")
      self.serial_to_index[atom.atom_id] = atom_index
      self._selected_altloc[atom.atom_key] = (atom.occupancy, atom.altloc, atom.atom_id)
      return

    prev_occupancy, prev_altloc, prev_serial = self._selected_altloc[atom.atom_key]
    if not self._should_replace_atom(prev_occupancy, prev_altloc, atom.occupancy, atom.altloc):
      return

    self.atoms[atom_index] = (atom.x, atom.y, atom.z)
    self.annotations["chain_id"][atom_index] = atom.chain_id
    self.annotations["res_id"][atom_index] = atom.res_id
    self.annotations["ins_code"][atom_index] = atom.ins_code
    self.annotations["res_name"][atom_index] = atom.res_name
    self.annotations["hetero"][atom_index] = atom.hetero
    self.annotations["atom_name"][atom_index] = atom.atom_name
    self.annotations["element"][atom_index] = atom.element
    self.annotations["atom_id"][atom_index] = atom.atom_id
    self.annotations["b_factor"][atom_index] = atom.b_factor
    self.annotations["occupancy"][atom_index] = atom.occupancy
    self.annotations["charge"][atom_index] = atom.charge
    self.annotations["sym_id"][atom_index] = ""
    if prev_serial in self.serial_to_index and self.serial_to_index[prev_serial] == atom_index:
      del self.serial_to_index[prev_serial]
    self.serial_to_index[atom.atom_id] = atom_index
    self._selected_altloc[atom.atom_key] = (atom.occupancy, atom.altloc, atom.atom_id)

  def add_directed_bond(self, source_serial: int, target_serial: int) -> bool:
    """Register a directed bond if both atom serials exist in this model."""
    source_index = self.serial_to_index.get(source_serial)
    target_index = self.serial_to_index.get(target_serial)
    if source_index is None or target_index is None or source_index == target_index:
      return False

    self.directed_bonds[(source_index, target_index)] += 1
    return True

  def to_structure(self) -> Structure:
    """Finalize the current model into a :class:`Structure`."""
    structure = Structure(remove_annotations=False)
    structure.metadata = {"model_id": self.model_id}

    if self.atoms:
      structure.atoms = np.array(self.atoms, dtype=structure._dtype_atoms)
      structure.atom_annotations = np.empty(len(self.atoms), dtype=structure._dtype_atom_annotations)
      for field_name, values in self.annotations.items():
        structure.atom_annotations[field_name] = np.array(values, dtype=structure._dtype_atom_annotations.fields[field_name][0])
    else:
      structure.atoms = np.zeros(0, dtype=structure._dtype_atoms)
      structure.atom_annotations = np.zeros(0, dtype=structure._dtype_atom_annotations)

    bond_rows = []
    undirected_bonds: Dict[Tuple[int, int], int] = {}
    for (atom_i, atom_j), count in self.directed_bonds.items():
      # PDB ``CONECT`` records may appear in both directions, so bond order is
      # determined from the strongest directed count for each undirected pair.
      pair = (min(atom_i, atom_j), max(atom_i, atom_j))
      undirected_bonds[pair] = max(undirected_bonds.get(pair, 0), count)

    for (atom_i, atom_j), bond_type in sorted(undirected_bonds.items()):
      bond_rows.append((atom_i, atom_j, bond_type))

    if bond_rows:
      structure.bonds = np.array(bond_rows, dtype=structure._dtype_bond)
    else:
      structure.bonds = np.zeros(0, dtype=structure._dtype_bond)

    structure._remove_empty_annotations()
    return structure

  @staticmethod
  def _should_replace_atom(prev_occupancy: float, prev_altloc: str, new_occupancy: float, new_altloc: str) -> bool:
    """Return ``True`` if a newly seen altloc atom should replace the current one."""
    if new_occupancy > prev_occupancy:
      return True
    if new_occupancy < prev_occupancy:
      return False
    if prev_altloc and not new_altloc:
      return True
    return False


def parse_pdb(
  pdb: Union[str, pathlib.Path, io.IOBase],
  return_type: ReturnType = "auto",
) -> Union[StructureEnsemble, StructureStack]:
  """Parse a PDB file into Neurosnap structure containers.

  Parsing follows the fixed-width PDB record layout used by BioPython's parser
  but builds Neurosnap :class:`Structure` entities.Parsed models are first
  collected into a :class:`StructureEnsemble` and are optionally converted into a
  :class:`StructureStack` at the end.

  HETATM records and ``CONECT`` records are parsed directly so ligands and
  custom covalent bonds are preserved more faithfully than in the old
  BioPython-backed path.

  Alternate locations are always ignored. When alternate locations are present,
  the parser keeps only the highest-occupancy conformer for each atom site and
  emits a :func:`logger.warning` so the user knows this happened.

  Repeated ``CONECT`` records are interpreted as higher bond order using a
  directed-count collapse:
    bond_type = max(count(atom_i -> atom_j), count(atom_j -> atom_i))
  This means repeated records in one direction can encode double or triple
  bonds, while mirrored ``CONECT`` entries from both atoms do not artificially
  inflate bond order.

  Parameters:
    pdb: PDB filepath or open file handle.
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
  lines = _read_lines(pdb)
  if not lines:
    raise ValueError("Empty file.")

  altloc_sites: set[AltlocSite] = set()
  ensemble = _parse_pdb_models(lines, altloc_sites)
  ensemble.metadata["source_format"] = "pdb"

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


def save_pdb(structure: Union[Structure, StructureEnsemble, StructureStack], pdb: Union[str, pathlib.Path, io.IOBase]):
  """Save a Neurosnap structure container as a PDB file.

  Parameters:
    structure: Structure container to write.
    pdb: Output filepath or open file handle.

  Notes:
    Multi-model outputs are written using ``MODEL`` / ``ENDMDL`` records.
    ``CONECT`` records are written for single-model outputs and for multi-model
    outputs only when all models share identical bonds and atom serials so the
    topology can be represented unambiguously in PDB format.
  """
  models = _models_for_pdb_output(structure)
  shared_conect_lines: Optional[List[str]] = None
  if len(models) > 1:
    shared_conect_lines = _shared_conect_lines(models)

  lines: List[str] = []
  for model_position, (model_id, model) in enumerate(models):
    serials = _atom_serials_for_model(model)
    if len(models) > 1:
      lines.append(_format_model_record(model_id))

    lines.extend(_atom_record_lines(model, serials))
    if len(model) > 0:
      lines.append("TER")

    if len(models) == 1:
      lines.extend(_conect_lines_for_model(model, serials))
    elif shared_conect_lines is None and len(model.bonds) > 0 and model_position == 0:
      logger.warning("Omitting CONECT records for multi-model PDB output because the models do not share identical bonds and atom serials.")

    if len(models) > 1:
      lines.append("ENDMDL")

  if shared_conect_lines is not None:
    lines.extend(shared_conect_lines)
  lines.append("END")
  _write_pdb_lines(pdb, lines)


def _parse_pdb_models(
  lines: Iterable[str],
  altloc_sites: set[AltlocSite],
) -> StructureEnsemble:
  """Parse PDB coordinate records into a :class:`StructureEnsemble`.

  The parser accumulates each model independently and applies ``CONECT``
  records only after all atoms are known so serial-number lookups are complete.
  """
  pending_conect: List[ConectRecord] = []
  models: List[_ModelAccumulator] = []
  current_model: Optional[_ModelAccumulator] = None
  implicit_model_id = 1

  for line_number, raw_line in enumerate(lines, start=1):
    if not raw_line.strip():
      continue

    padded_line = raw_line.ljust(80)
    record_type = padded_line[0:6]

    if record_type == "MODEL ":
      model_id = _parse_model_id(padded_line, line_number)
      current_model = _ModelAccumulator(model_id=model_id)
      models.append(current_model)
      implicit_model_id += 1
      continue

    if record_type == "ENDMDL":
      current_model = None
      continue

    if record_type in ("ATOM  ", "HETATM"):
      if current_model is None:
        current_model = _ModelAccumulator(model_id=implicit_model_id)
        models.append(current_model)
        implicit_model_id += 1

      atom = _parse_atom_record(padded_line, record_type, line_number)
      if atom.altloc:
        # Altlocs are dropped later inside ``add_atom()``, but the parser keeps
        # track of affected sites so it can emit one summary warning.
        altloc_sites.add((current_model.model_id, atom.atom_key))
      current_model.add_atom(atom)
      continue

    if record_type == "CONECT":
      conect = _parse_conect_record(padded_line, line_number)
      if conect is not None:
        pending_conect.append(conect)
      continue

    if record_type == "TER   ":
      continue

  if not models:
    raise ValueError("No models or atoms were found in the PDB file.")

  _apply_conect_records(models, pending_conect)
  ensemble = StructureEnsemble()
  for model in models:
    ensemble.append(model.to_structure(), model_id=model.model_id)
  return ensemble


def _models_for_pdb_output(
  structure: Union[Structure, StructureEnsemble, StructureStack],
) -> List[Tuple[int, Structure]]:
  """Return a normalized list of ``(model_id, model)`` pairs for writing."""
  if isinstance(structure, Structure):
    model_id = int(structure.metadata.get("model_id", 1))
    return [(model_id, structure)]
  if isinstance(structure, StructureEnsemble):
    return list(zip(structure.model_ids, structure.models()))
  if isinstance(structure, StructureStack):
    return list(zip(structure.model_ids, structure.models()))
  raise TypeError(f"Unsupported structure type for PDB output: {type(structure).__name__}.")


def _shared_conect_lines(models: List[Tuple[int, Structure]]) -> Optional[List[str]]:
  """Return shared ``CONECT`` lines for compatible multi-model outputs."""
  if not models:
    return []

  serial_maps = []
  for _, model in models:
    serials = _atom_serials_for_model(model)
    serial_maps.append(serials)

  reference_model = models[0][1]
  reference_serials = serial_maps[0]
  for (_, model), serials in zip(models[1:], serial_maps[1:]):
    if len(model.bonds) != len(reference_model.bonds):
      return None
    if not np.array_equal(model.bonds, reference_model.bonds):
      return None
    if not np.array_equal(serials, reference_serials):
      return None

  return _conect_lines_for_model(reference_model, reference_serials)


def _atom_serials_for_model(model: Structure) -> np.ndarray:
  """Return atom serial numbers for a model, preserving them when possible."""
  if len(model) > 99999:
    raise ValueError("PDB output supports at most 99999 atoms per model.")

  if "atom_id" in model.atom_annotations.dtype.names:
    serials = np.asarray(model.atom_annotations["atom_id"], dtype=np.int32)
    if serials.size and np.all(serials > 0) and len(np.unique(serials)) == len(serials) and np.max(serials) <= 99999:
      return serials.copy()

  return np.arange(1, len(model) + 1, dtype=np.int32)


def _format_model_record(model_id: int) -> str:
  """Return a ``MODEL`` record."""
  if model_id < 1 or model_id > 9999:
    raise ValueError(f'MODEL serial "{model_id}" is outside the supported PDB range 1-9999.')
  return f"MODEL     {model_id:4d}"


def _atom_record_lines(model: Structure, serials: np.ndarray) -> List[str]:
  """Return ``ATOM`` / ``HETATM`` lines for a model."""
  chain_ids = model.atom_annotations["chain_id"]
  lines = []
  previous_chain_id = None
  for atom_index in range(len(model)):
    chain_id = str(chain_ids[atom_index])
    if previous_chain_id is not None and chain_id != previous_chain_id:
      lines.append("TER")
    lines.append(_format_atom_record(model, atom_index, int(serials[atom_index])))
    previous_chain_id = chain_id
  return lines


def _format_atom_record(model: Structure, atom_index: int, serial: int) -> str:
  """Format one ``ATOM`` or ``HETATM`` record."""
  if serial < 1 or serial > 99999:
    raise ValueError(f'Atom serial "{serial}" is outside the supported PDB range 1-99999.')

  atom_name = str(model.atom_annotations["atom_name"][atom_index]).strip().upper()
  residue_name = str(model.atom_annotations["res_name"][atom_index]).strip().upper()
  chain_id = str(model.atom_annotations["chain_id"][atom_index]).strip()
  insertion_code = str(model.atom_annotations["ins_code"][atom_index]).strip()
  element = str(model.atom_annotations["element"][atom_index]).strip().upper()
  hetero = bool(model.atom_annotations["hetero"][atom_index])
  residue_id = int(model.atom_annotations["res_id"][atom_index])

  if not atom_name:
    raise ValueError(f"Atom {atom_index + 1} is missing an atom_name and cannot be written to PDB.")
  if len(atom_name) > 4:
    raise ValueError(f'Atom name "{atom_name}" exceeds the 4-character PDB limit.')
  if not residue_name:
    raise ValueError(f"Atom {atom_index + 1} is missing a res_name and cannot be written to PDB.")
  if len(residue_name) > 3:
    raise ValueError(f'Residue name "{residue_name}" exceeds the 3-character PDB limit.')
  if len(chain_id) > 1:
    raise ValueError(f'Chain ID "{chain_id}" exceeds the 1-character PDB limit.')
  if len(insertion_code) > 1:
    raise ValueError(f'Insertion code "{insertion_code}" exceeds the 1-character PDB limit.')
  if not element:
    raise ValueError(f"Atom {atom_index + 1} is missing an element and cannot be written to PDB.")
  if len(element) > 2:
    raise ValueError(f'Element "{element}" exceeds the 2-character PDB limit.')

  if len(f"{residue_id:d}") > 4:
    raise ValueError(f'Residue ID "{residue_id}" exceeds the 4-character PDB limit.')

  occupancy = _annotation_value_for_pdb(model, "occupancy", atom_index, 1.0)
  b_factor = _annotation_value_for_pdb(model, "b_factor", atom_index, 0.0)
  charge = _annotation_value_for_pdb(model, "charge", atom_index, 0)

  atom_name_field = _format_atom_name(atom_name, element)
  charge_field = _format_charge_field(int(charge))
  record_name = "HETATM" if hetero else "ATOM  "
  return (
    f"{record_name}{serial:5d} {atom_name_field} "
    f"{residue_name:>3} {chain_id[:1]:1}{residue_id:4d}{insertion_code[:1]:1}   "
    f"{float(model.atoms['x'][atom_index]):8.3f}"
    f"{float(model.atoms['y'][atom_index]):8.3f}"
    f"{float(model.atoms['z'][atom_index]):8.3f}"
    f"{float(occupancy):6.2f}"
    f"{float(b_factor):6.2f}"
    f"          {element:>2}{charge_field:>2}"
  )


def _annotation_value_for_pdb(model: Structure, name: str, atom_index: int, default):
  """Return an annotation value with a fallback default for PDB output."""
  if name not in model.atom_annotations.dtype.names:
    return default
  value = model.atom_annotations[name][atom_index]
  if isinstance(value, np.generic):
    return value.item()
  return value


def _format_atom_name(atom_name: str, element: str) -> str:
  """Return a 4-character atom name field."""
  if len(atom_name) == 4:
    return atom_name
  if len(element) == 1 and atom_name and atom_name[0].isalpha():
    return f" {atom_name:<3}"
  return f"{atom_name:>4}"


def _format_charge_field(charge: int) -> str:
  """Return a 2-character PDB charge field."""
  if charge == 0:
    return ""
  if abs(charge) > 9:
    raise ValueError(f'Atom charge "{charge}" exceeds the 1-digit PDB charge limit.')
  sign = "+" if charge > 0 else "-"
  return f"{abs(charge)}{sign}"


def _conect_lines_for_model(model: Structure, serials: np.ndarray) -> List[str]:
  """Return ``CONECT`` lines for a model."""
  if len(model.bonds) == 0:
    return []

  atom_index_to_serial = {atom_index: int(serial) for atom_index, serial in enumerate(serials)}
  directed_counts: Counter[Tuple[int, int]] = Counter()
  for bond in model.bonds:
    atom_i = int(bond["atom_i"])
    atom_j = int(bond["atom_j"])
    bond_type = max(1, int(bond["bond_type"]))
    if atom_i not in atom_index_to_serial or atom_j not in atom_index_to_serial:
      raise ValueError("Bond table contains atom indices outside the atom table.")
    directed_counts[(atom_i, atom_j)] += bond_type

  lines = []
  for (atom_i, atom_j), count in sorted(directed_counts.items()):
    source_serial = atom_index_to_serial[atom_i]
    target_serial = atom_index_to_serial[atom_j]
    lines.extend(_format_conect_records(source_serial, target_serial, count))
  return lines


def _format_conect_records(source_serial: int, target_serial: int, count: int) -> List[str]:
  """Return one or more ``CONECT`` records for a repeated bond."""
  if source_serial > 99999 or target_serial > 99999:
    raise ValueError("CONECT atom serial exceeds the 5-character PDB limit.")

  repeated_targets = [target_serial] * count
  lines = []
  for start in range(0, len(repeated_targets), 4):
    chunk = repeated_targets[start : start + 4]
    fields = "".join(f"{serial:5d}" for serial in chunk)
    lines.append(f"CONECT{source_serial:5d}{fields}")
  return lines


def _apply_conect_records(models: List[_ModelAccumulator], pending_conect: List[ConectRecord]):
  """Apply stored ``CONECT`` records to all models that contain the referenced serials."""
  for source_serial, target_serials, line_number in pending_conect:
    found_match = False
    for model in models:
      for target_serial in target_serials:
        if model.add_directed_bond(source_serial, target_serial):
          found_match = True

    if not found_match:
      raise ValueError(f"CONECT record references unknown atom serials for source atom {source_serial} at line {line_number}.")


def _parse_model_id(line: str, line_number: int) -> int:
  """Parse the serial number from a ``MODEL`` record."""
  model_field = line[10:14].strip()
  if not model_field:
    raise ValueError(f"Missing MODEL serial number at line {line_number}.")

  try:
    return int(model_field)
  except ValueError:
    raise ValueError(f'Invalid MODEL serial number "{model_field}" at line {line_number}.')


def _parse_atom_record(line: str, record_type: str, line_number: int) -> _AtomRecord:
  """Parse a single ``ATOM`` or ``HETATM`` record.

  The parser is intentionally strict about required fields such as residue
  names and element assignments so ambiguous files fail early.
  """
  try:
    atom_id = int(line[6:11].strip())
    atom_name = line[12:16].strip()
    altloc = line[16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21].strip()
    res_id = int(line[22:26].strip())
    ins_code = line[26].strip()
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    occupancy = _parse_float_field(line[54:60], default=1.0, line_number=line_number, label="occupancy")
    b_factor = _parse_float_field(line[60:66], default=0.0, line_number=line_number, label="B-factor")
    element = line[76:78].strip().upper()
  except ValueError as exc:
    raise ValueError(f"Invalid atom record ({exc}) at line {line_number}.")

  if not atom_name:
    raise ValueError(f"Missing atom name at line {line_number}.")
  if not res_name:
    raise ValueError(f"Missing residue name at line {line_number}.")
  if not element:
    raise ValueError(f"Missing element assignment at line {line_number}.")

  charge = _parse_charge(line[78:80], line_number)
  hetero = record_type == "HETATM"
  # Altloc is excluded from the identity key so alternate conformers collapse
  # onto a single atom site during model accumulation.
  atom_key = (chain_id, res_id, ins_code, res_name, hetero, atom_name)
  return _AtomRecord(
    x=x,
    y=y,
    z=z,
    chain_id=chain_id,
    res_id=res_id,
    ins_code=ins_code,
    res_name=res_name,
    hetero=hetero,
    atom_name=atom_name,
    element=element,
    atom_id=atom_id,
    b_factor=b_factor,
    occupancy=occupancy,
    charge=charge,
    altloc=altloc,
    atom_key=atom_key,
  )


def _parse_conect_record(line: str, line_number: int) -> Optional[Tuple[int, List[int], int]]:
  """Parse a ``CONECT`` record into source and target serial numbers."""
  serials: List[int] = []
  for start in range(6, len(line), 5):
    chunk = line[start : start + 5].strip()
    if not chunk:
      continue
    try:
      serials.append(int(chunk))
    except ValueError:
      raise ValueError(f'Invalid CONECT serial "{chunk}" at line {line_number}.')

  if len(serials) < 2:
    return None
  return serials[0], serials[1:], line_number


def _parse_float_field(field: str, *, default: float, line_number: int, label: str) -> float:
  """Parse a float field with fixed defaults for blank values."""
  field = field.strip()
  if not field:
    return default

  try:
    return float(field)
  except ValueError:
    raise ValueError(f'Invalid {label} value "{field}" at line {line_number}.')


def _parse_charge(field: str, line_number: int) -> int:
  """Parse the PDB atom charge field into a signed integer."""
  field = field.strip()
  if not field:
    return 0

  if len(field) == 2 and field[0].isdigit() and field[1] in "+-":
    return int(field[0]) * (1 if field[1] == "+" else -1)
  if len(field) == 2 and field[1].isdigit() and field[0] in "+-":
    return int(field[1]) * (1 if field[0] == "+" else -1)
  if field.endswith(("+", "-")) and field[:-1].isdigit():
    return int(field[:-1]) * (1 if field[-1] == "+" else -1)
  if field.lstrip("+-").isdigit():
    return int(field)

  raise ValueError(f'Invalid atom charge "{field}" at line {line_number}.')


def _read_lines(file: Union[str, pathlib.Path, io.IOBase]) -> List[str]:
  """Read all lines from a filepath or file handle.

  Text and binary file handles are both accepted so the parser works with
  the same broad range of inputs as the rest of the I/O layer.
  """
  if isinstance(file, (str, pathlib.Path)):
    with open(file, "rt", encoding="utf-8") as handle:
      return handle.read().splitlines()

  content = file.read()
  if isinstance(content, bytes):
    content = content.decode("utf-8")
  return content.splitlines()


def _write_pdb_lines(file: Union[str, pathlib.Path, io.IOBase], lines: List[str]):
  """Write PDB lines to a filepath or file handle."""
  content = "\n".join(lines) + "\n"
  if isinstance(file, (str, pathlib.Path)):
    with open(file, "wt", encoding="utf-8", newline="\n") as handle:
      handle.write(content)
    return

  if isinstance(file, io.TextIOBase):
    file.write(content)
    return
  try:
    file.write(content)
  except TypeError:
    file.write(content.encode("utf-8"))
