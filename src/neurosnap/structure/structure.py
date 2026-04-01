"""
# TODO: Full desc
Universal unit is Å.
# TODO: Include some examples:
# ca_atoms = atoms[atoms["atom_name"] == "CA"]

"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

### IMPORTANT NOTES
# Universal unit is Å.
# This new Structure object does not care about altlocs and will automatically drop them
# Hetatoms are stored with proper bond information
# In PDB files repeated bonds will correspond to bonds being interpreted at a higher order. For instance if atom i and j have two records for bonds in a PDB file this will be interpreted as them having a double bond.
# Each structure corresponds to a single model ONLY, the StructureEnsemble object should be used instead for an ordered collection of models (OR optional later: StructureStack = shared-annotation multi-model fast path, only when all models have identical atoms/bonds)


class Structure:
  # These fields define atom identity and basic PDB semantics, so they must
  # always be present even if every value is currently the default.
  _MANDATORY_ANNOTATIONS = ("chain_id", "res_id", "ins_code", "res_name", "hetero", "atom_name", "element")
  _ANNOTATION_DEFAULTS = {
    "chain_id": "",
    "res_id": 0,
    "ins_code": "",
    "res_name": "",
    "hetero": False,
    "atom_name": "",
    "element": "",
    "atom_id": 0,
    "b_factor": 0.0,
    "occupancy": 1.0,
    "charge": 0,
    "sym_id": "",
  }

  def __init__(self, remove_annotations: bool = True):
    # keys are metadata field names / titles and values are the corresponding values
    # TODO: Read metadata from file and add it as needed
    self.metadata: Dict[str, Any] = {}

    # create dtype for atoms array
    self._dtype_atoms = np.dtype(
      [
        ("x", "f4"),  # x coordinate
        ("y", "f4"),  # y coordinate
        ("z", "f4"),  # z coordinate
      ]
    )

    # create dtype for annotations array
    self._dtype_atom_annotations = np.dtype(
      [
        ("chain_id", "U4"),  # chain ID
        ("res_id", "i4"),  # residue number
        ("ins_code", "U1"),  # insertion code
        ("res_name", "U5"),  # residue name
        ("hetero", "?"),  # ATOM vs HETATM
        ("atom_name", "U6"),  # atom name
        ("element", "U2"),  # chemical element
        ("atom_id", "i4"),  # atom serial number
        ("b_factor", "f4"),  # temperature factor
        ("occupancy", "f4"),  # occupancy
        ("charge", "i1"),  # small int (-128 to 127 is enough)
        ("sym_id", "U4"),  # symmetry ID (string, often small)
      ]
    )

    # create dtype for bonds array
    self._dtype_bond = np.dtype(
      [
        ("atom_i", np.int32),
        ("atom_j", np.int32),
        ("bond_type", np.int8),
      ]
    )

    # Coordinates and annotations are stored separately so annotation schema
    # changes only require rebuilding the annotation table.
    self.atoms = np.zeros(0, dtype=self._dtype_atoms)
    self.atom_annotations = np.zeros(0, dtype=self._dtype_atom_annotations)
    self.bonds = np.zeros(0, dtype=self._dtype_bond)

    if remove_annotations:
      self._remove_empty_annotations()

  def __len__(self) -> int:
    return len(self.atoms)

  def chains(self) -> List["Chain"]:
    """Return the chains in this structure as immutable hierarchy views."""
    chain_map: Dict[str, Dict[Tuple[int, str, str, bool], List[int]]] = {}

    for atom_index in range(len(self)):
      chain_id = self._annotation_value("chain_id", atom_index)
      residue_key = (
        self._annotation_value("res_id", atom_index),
        self._annotation_value("ins_code", atom_index),
        self._annotation_value("res_name", atom_index),
        self._annotation_value("hetero", atom_index),
      )

      if chain_id not in chain_map:
        chain_map[chain_id] = {}
      if residue_key not in chain_map[chain_id]:
        chain_map[chain_id][residue_key] = []
      chain_map[chain_id][residue_key].append(atom_index)

    chains: List[Chain] = []
    for chain_id, residue_map in chain_map.items():
      residues: List[Residue] = []
      for residue_key, atom_indices in residue_map.items():
        res_id, ins_code, res_name, hetero = residue_key
        residues.append(
          Residue(
            chain_id=chain_id,
            res_id=res_id,
            ins_code=ins_code,
            res_name=res_name,
            hetero=hetero,
            _atoms=tuple(self._atom_view(atom_index) for atom_index in atom_indices),
          )
        )
      chains.append(Chain(chain_id=chain_id, _residues=tuple(residues)))
    return chains

  def add_annotation(
    self,
    name: str,
    dtype: Any,
    values: Any = None,
    *,
    fill_value: Any = None,
    overwrite: bool = False,
  ):
    """Add a new per-atom annotation column."""
    if not isinstance(name, str) or not name:
      raise ValueError("Annotation name must be a non-empty string.")
    if name in self._dtype_atoms.names:
      raise ValueError(f'"{name}" is reserved for coordinate storage.')

    annotation_dtype = np.dtype(dtype)
    if annotation_dtype.names is not None:
      raise TypeError("Annotation dtype must describe a single scalar value per atom.")

    if name in self._dtype_atom_annotations.names:
      if not overwrite:
        raise ValueError(f'Annotation "{name}" already exists.')
      if name in self._MANDATORY_ANNOTATIONS:
        raise ValueError(f'Cannot overwrite mandatory annotation "{name}".')
      self.remove_annotation(name)

    atom_count = len(self)
    if values is not None:
      # Annotation columns are always per-atom, so the incoming data must line
      # up exactly with the current atom table.
      values_array = np.asarray(values, dtype=annotation_dtype)
      if values_array.ndim != 1:
        raise ValueError("Annotation values must be one-dimensional.")
      if len(values_array) != atom_count:
        raise ValueError(f'Annotation "{name}" must contain exactly {atom_count} values; found {len(values_array)}.')
    else:
      if fill_value is None:
        fill_value = self._default_fill_value(name, annotation_dtype)
      values_array = np.full(atom_count, fill_value, dtype=annotation_dtype)

    # NumPy structured dtypes are immutable, so adding a field means rebuilding
    # the annotation table with the old columns copied over.
    new_dtype = np.dtype(list(self._dtype_atom_annotations.descr) + [(name, annotation_dtype)])
    new_annotations = np.empty(atom_count, dtype=new_dtype)
    for field_name in self._dtype_atom_annotations.names:
      new_annotations[field_name] = self.atom_annotations[field_name]
    new_annotations[name] = values_array

    self._dtype_atom_annotations = new_dtype
    self.atom_annotations = new_annotations

  def remove_annotation(self, name: str):
    """Remove a non-mandatory annotation column and return its values."""
    if not isinstance(name, str) or not name:
      raise ValueError("Annotation name must be a non-empty string.")
    if name not in self._dtype_atom_annotations.names:
      raise KeyError(f'Annotation "{name}" was not found.')
    if name in self._MANDATORY_ANNOTATIONS:
      raise ValueError(f'Cannot remove mandatory annotation "{name}".')

    removed_values = self.atom_annotations[name].copy()
    # Removing a field uses the same rebuild path as add_annotation(), but with
    # the target column omitted from the new dtype.
    remaining_fields = [
      (field_name, self._dtype_atom_annotations.fields[field_name][0]) for field_name in self._dtype_atom_annotations.names if field_name != name
    ]
    new_dtype = np.dtype(remaining_fields)
    new_annotations = np.empty(len(self), dtype=new_dtype)
    for field_name in new_dtype.names:
      new_annotations[field_name] = self.atom_annotations[field_name]

    self._dtype_atom_annotations = new_dtype
    self.atom_annotations = new_annotations
    return removed_values

  def _remove_empty_annotations(self):
    """Drop optional annotations that do not currently carry information."""
    # This is most useful after parsing/loading when optional columns were kept
    # available during construction but ended up containing only defaults.
    for name in list(self._dtype_atom_annotations.names):
      if name in self._MANDATORY_ANNOTATIONS:
        continue
      if self._annotation_is_empty(name):
        self.remove_annotation(name)

  def _annotation_is_empty(self, name: str) -> bool:
    values = self.atom_annotations[name]
    if values.size == 0:
      return True
    # A column that is still entirely filled with its default sentinel value
    # does not contain any structure-specific information yet.
    default = self._default_fill_value(name, values.dtype)
    return bool(np.all(values == default))

  def _default_fill_value(self, name: str, dtype: np.dtype):
    if name in self._ANNOTATION_DEFAULTS:
      return self._ANNOTATION_DEFAULTS[name]

    if dtype.kind in {"U", "S"}:
      return ""
    if dtype.kind == "b":
      return False
    if dtype.kind in {"i", "u"}:
      return 0
    if dtype.kind == "f":
      return 0.0

    raise TypeError(f'No default fill value is defined for dtype "{dtype}" of annotation "{name}".')

  def _annotation_value(self, name: str, atom_index: int):
    value = self.atom_annotations[name][atom_index]
    if isinstance(value, np.generic):
      return value.item()
    return value

  def _atom_view(self, atom_index: int) -> "Atom":
    coord = self.atoms[atom_index]
    extra_annotations = {
      name: self._annotation_value(name, atom_index) for name in self._dtype_atom_annotations.names if name not in self._MANDATORY_ANNOTATIONS
    }
    return Atom(
      x=float(coord["x"]),
      y=float(coord["y"]),
      z=float(coord["z"]),
      chain_id=self._annotation_value("chain_id", atom_index),
      res_id=self._annotation_value("res_id", atom_index),
      ins_code=self._annotation_value("ins_code", atom_index),
      res_name=self._annotation_value("res_name", atom_index),
      hetero=self._annotation_value("hetero", atom_index),
      atom_name=self._annotation_value("atom_name", atom_index),
      element=self._annotation_value("element", atom_index),
      annotations=MappingProxyType(extra_annotations),
    )


@dataclass(frozen=True, slots=True)
class Atom:
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
  annotations: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

  @property
  def coord(self) -> np.ndarray:
    return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Residue:
  chain_id: str
  res_id: int
  ins_code: str
  res_name: str
  hetero: bool
  _atoms: Tuple[Atom, ...] = field(repr=False)

  def atoms(self) -> List[Atom]:
    return list(self._atoms)


@dataclass(frozen=True, slots=True)
class Chain:
  chain_id: str
  _residues: Tuple[Residue, ...] = field(repr=False)

  def residues(self) -> List[Residue]:
    return list(self._residues)
