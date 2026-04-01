"""Parser for PDB files.

This module provides a Neurosnap-native :func:`parse_pdb` function that reads
PDB files into :class:`~neurosnap.structure.structure.StructureEnsemble` or
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

__all__ = ["parse_pdb"]

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
  format: str = "auto",
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
    format: Input format. ``"auto"`` and ``"pdb"`` are accepted. Since this
      parser only supports PDB files, ``"auto"`` resolves to ``"pdb"``.
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
  if format not in {"auto", "pdb"}:
    raise ValueError('format must be either "auto" or "pdb".')

  # ``auto`` is accepted to match the higher-level Protein API even though
  # this parser only supports PDB input.
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

  Text and binary file handles are both accepted to match the existing
  ``Protein`` loader conventions.
  """
  if isinstance(file, (str, pathlib.Path)):
    with open(file, "rt", encoding="utf-8") as handle:
      return handle.read().splitlines()

  content = file.read()
  if isinstance(content, bytes):
    content = content.decode("utf-8")
  return content.splitlines()
