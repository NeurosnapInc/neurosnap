"""Data structures for representing molecular coordinates and annotations.

This module provides a single-model :class:`Structure`, immutable hierarchy
views (:class:`Chain`, :class:`Residue`, and :class:`Atom`), an ordered
multi-model container (:class:`StructureEnsemble`), and a shared-annotation
multi-model fast path (:class:`StructureStack`).

The universal length unit is Å.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

### IMPORTANT NOTES
# Universal unit is Å.
# This new Structure object does not care about altlocs and will automatically drop them
# Hetatoms are stored with proper bond information
# In PDB files repeated bonds will correspond to bonds being interpreted at a higher order. For instance if atom i and j have two records for bonds in a PDB file this will be interpreted as them having a double bond.
# Each structure corresponds to a single model ONLY, the StructureEnsemble object should be used instead for an ordered collection of models (OR optional later: StructureStack = shared-annotation multi-model fast path, only when all models have identical atoms/bonds)


class Structure:
  """Single-model molecular structure container.

  Coordinates are stored separately from per-atom annotations so geometry-heavy
  operations can work on compact numeric arrays while annotation schemas remain
  flexible.

  Parameters:
    remove_annotations: If ``True``, optional annotation columns that only
      contain default values are removed after initialization.
  """

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
    """Initialize an empty single-model structure."""
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
    """Return the number of atoms in the structure."""
    return len(self.atoms)

  def chains(self) -> List["Chain"]:
    """Return all chains in the structure as immutable hierarchy views.

    Returns:
      List of :class:`Chain` objects in atom-table order.
    """
    chain_map: Dict[str, Dict[Tuple[int, str, str, bool], List[int]]] = {}

    for atom_index in range(len(self)):
      chain_id = self._annotation_value("chain_id", atom_index)
      residue_key = (
        self._annotation_value("res_id", atom_index),
        self._annotation_value("ins_code", atom_index),
        self._annotation_value("res_name", atom_index),
        self._annotation_value("hetero", atom_index),
      )

      # Chain and residue ordering follow the original atom table order so
      # hierarchy views remain stable with respect to the parsed file.
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
    """Add a new per-atom annotation column.

    Parameters:
      name: Annotation name to add.
      dtype: NumPy-compatible scalar dtype for the annotation values.
      values: Optional per-atom values for the annotation.
      fill_value: Optional default value used when ``values`` is not supplied.
      overwrite: Whether to replace an existing optional annotation of the same
        name.

    Raises:
      ValueError: If the name is invalid, reserved, already present, or the
        supplied values do not match the atom count.
      TypeError: If the supplied dtype is not a scalar per-atom dtype.
    """
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
    """Remove a non-mandatory annotation column and return its values.

    Parameters:
      name: Annotation name to remove.

    Returns:
      Copy of the removed annotation values.

    Raises:
      KeyError: If the annotation does not exist.
      ValueError: If the name is invalid or refers to a mandatory annotation.
    """
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
    """Return ``True`` if an annotation contains only default values."""
    values = self.atom_annotations[name]
    if values.size == 0:
      return True
    # A column that is still entirely filled with its default sentinel value
    # does not contain any structure-specific information yet.
    default = self._default_fill_value(name, values.dtype)
    return bool(np.all(values == default))

  def _default_fill_value(self, name: str, dtype: np.dtype):
    """Return the default fill value for an annotation dtype."""
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
    """Return a Python scalar for a single annotation value."""
    value = self.atom_annotations[name][atom_index]
    if isinstance(value, np.generic):
      return value.item()
    return value

  def _atom_view(self, atom_index: int) -> "Atom":
    """Create an immutable :class:`Atom` view for one atom index."""
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
  """Immutable atom-level hierarchy view."""

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
    """Return the atom coordinates as a length-3 NumPy array."""
    return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Residue:
  """Immutable residue-level hierarchy view."""

  chain_id: str
  res_id: int
  ins_code: str
  res_name: str
  hetero: bool
  _atoms: Tuple[Atom, ...] = field(repr=False)

  def atoms(self) -> List[Atom]:
    """Return all atoms in this residue."""
    return list(self._atoms)


@dataclass(frozen=True, slots=True)
class Chain:
  """Immutable chain-level hierarchy view."""

  chain_id: str
  _residues: Tuple[Residue, ...] = field(repr=False)

  def residues(self) -> List[Residue]:
    """Return all residues in this chain."""
    return list(self._residues)


def _validate_structure_model(model: Structure):
  """Validate that an object is a structurally consistent ``Structure``.

  Parameters:
    model: Candidate structure to validate.

  Raises:
    TypeError: If the input is not a :class:`Structure`.
    ValueError: If the coordinate and annotation tables are inconsistent.
  """
  if not isinstance(model, Structure):
    raise TypeError(f"Expected a Structure instance, found {type(model).__name__}.")
  if len(model.atoms) != len(model.atom_annotations):
    raise ValueError("Structure atoms and atom_annotations must have the same length.")
  if model.atoms.dtype.names != ("x", "y", "z"):
    raise ValueError('Structure atoms dtype must contain the coordinate fields "x", "y", and "z".')


class StructureEnsemble:
  """Ordered collection of independent ``Structure`` models.

  Unlike :class:`StructureStack`, models in an ensemble do not need to have the
  same atoms, annotations, or bonds.

  Parameters:
    models: Optional initial list of models.
    model_ids: Optional identifiers corresponding to ``models``.
    metadata: Optional ensemble-level metadata dictionary.
  """

  def __init__(
    self,
    models: Optional[List[Structure]] = None,
    *,
    model_ids: Optional[List[int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
  ):
    """Initialize an ordered collection of independent models."""
    self.metadata: Dict[str, Any] = dict(metadata or {})
    self._models: List[Structure] = []
    self.model_ids: List[int] = []

    models = models or []
    if model_ids is not None and len(model_ids) != len(models):
      raise ValueError("model_ids must have the same length as models.")

    for index, model in enumerate(models):
      model_id = None if model_ids is None else model_ids[index]
      self.append(model, model_id=model_id)

  def __len__(self) -> int:
    """Return the number of models in the ensemble."""
    return len(self._models)

  def __iter__(self) -> Iterator[Structure]:
    """Iterate over the stored models in order."""
    return iter(self._models)

  def __getitem__(self, index):
    """Return a model or sliced sub-ensemble."""
    if isinstance(index, slice):
      return StructureEnsemble(self._models[index], model_ids=self.model_ids[index], metadata=self.metadata)
    return self._models[index]

  def append(self, model: Structure, *, model_id: Optional[int] = None):
    """Append a validated model to the ensemble.

    Parameters:
      model: Model to append.
      model_id: Optional model identifier. Defaults to the next sequential index.
    """
    _validate_structure_model(model)
    self._models.append(model)
    self.model_ids.append(len(self.model_ids) if model_id is None else int(model_id))

  def models(self) -> List[Structure]:
    """Return the models as a shallow copied list."""
    return list(self._models)

  def to_stack(self) -> "StructureStack":
    """Convert the ensemble into a ``StructureStack``.

    Raises:
      ValueError: If the models are not stack-compatible.
    """
    return StructureStack(self._models, model_ids=self.model_ids, metadata=self.metadata)


class StructureStack:
  """Shared-annotation, shared-bond multi-model fast path.

  All models in a stack must share the same atom ordering, per-atom annotations,
  and bonds. Only the coordinates vary between models.

  Parameters:
    models: Optional initial list of stack-compatible models.
    model_ids: Optional identifiers corresponding to ``models``.
    metadata: Optional stack-level metadata dictionary.
  """

  def __init__(
    self,
    models: Optional[List[Structure]] = None,
    *,
    model_ids: Optional[List[int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
  ):
    """Initialize an empty or pre-populated stack of compatible models."""
    self.metadata: Dict[str, Any] = dict(metadata or {})
    self.model_ids: List[int] = []

    # Use an empty Structure instance to seed the default dtypes for an empty
    # stack before the first model is added.
    template = Structure(remove_annotations=False)
    self._dtype_atoms = template._dtype_atoms
    self._dtype_atom_annotations = template._dtype_atom_annotations
    self._dtype_bond = template._dtype_bond
    self.coord = np.zeros((0, 0, 3), dtype=np.float32)
    self.atom_annotations = np.zeros(0, dtype=self._dtype_atom_annotations)
    self.bonds = np.zeros(0, dtype=self._dtype_bond)

    models = models or []
    if model_ids is not None and len(model_ids) != len(models):
      raise ValueError("model_ids must have the same length as models.")

    for index, model in enumerate(models):
      model_id = None if model_ids is None else model_ids[index]
      self.append(model, model_id=model_id)

  def __len__(self) -> int:
    """Return the number of models in the stack."""
    return self.coord.shape[0]

  def __iter__(self) -> Iterator[Structure]:
    """Iterate over the stack as materialized ``Structure`` models."""
    for model_index in range(len(self)):
      yield self[model_index]

  def __getitem__(self, index):
    """Return a materialized model or sliced sub-stack."""
    if isinstance(index, slice):
      return StructureStack._from_parts(
        self.coord[index].copy(),
        self.atom_annotations.copy(),
        self.bonds.copy(),
        model_ids=self.model_ids[index],
        metadata=self.metadata,
      )
    return self._model_to_structure(index)

  @property
  def atom_count(self) -> int:
    """Return the number of shared atoms per model."""
    return self.coord.shape[1]

  def append(self, model: Structure, *, model_id: Optional[int] = None):
    """Append a stack-compatible model.

    Parameters:
      model: Model to append.
      model_id: Optional model identifier. Defaults to the next sequential index.

    Raises:
      ValueError: If the candidate model is not compatible with the existing
        stack.
    """
    _validate_structure_model(model)
    coord = self._coord_matrix_from_structure(model)

    if len(self) == 0:
      # The first model defines the shared annotation and bond schema for the
      # entire stack.
      self._dtype_atoms = model._dtype_atoms
      self._dtype_atom_annotations = model._dtype_atom_annotations
      self._dtype_bond = model._dtype_bond
      self.coord = coord[np.newaxis, ...]
      self.atom_annotations = np.array(model.atom_annotations, dtype=model.atom_annotations.dtype, copy=True)
      self.bonds = np.array(model.bonds, dtype=model.bonds.dtype, copy=True)
    else:
      reference = self._model_to_structure(0)
      self._ensure_stack_compatible(reference, model)
      self.coord = np.concatenate((self.coord, coord[np.newaxis, ...]), axis=0)

    self.model_ids.append(len(self.model_ids) if model_id is None else int(model_id))

  def models(self) -> List[Structure]:
    """Materialize and return all models in the stack."""
    return [self[index] for index in range(len(self))]

  def to_ensemble(self) -> StructureEnsemble:
    """Convert the stack into an independent ``StructureEnsemble``."""
    return StructureEnsemble(self.models(), model_ids=self.model_ids, metadata=self.metadata)

  @classmethod
  def from_ensemble(cls, ensemble: StructureEnsemble) -> "StructureStack":
    """Build a stack from an ensemble of compatible models."""
    return cls(ensemble.models(), model_ids=ensemble.model_ids, metadata=ensemble.metadata)

  @classmethod
  def _from_parts(
    cls,
    coord: np.ndarray,
    atom_annotations: np.ndarray,
    bonds: np.ndarray,
    *,
    model_ids: Optional[List[int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
  ) -> "StructureStack":
    """Construct a stack directly from shared coordinates and annotations."""
    coord = np.asarray(coord, dtype=np.float32)
    if coord.ndim != 3 or coord.shape[2] != 3:
      raise ValueError("StructureStack coordinates must have shape (n_models, n_atoms, 3).")
    if len(atom_annotations) != coord.shape[1]:
      raise ValueError("Shared atom annotations must match the atom dimension of coord.")

    stack = cls(metadata=metadata)
    stack.coord = coord.copy()
    stack._dtype_atom_annotations = atom_annotations.dtype
    stack.atom_annotations = np.array(atom_annotations, dtype=atom_annotations.dtype, copy=True)
    stack._dtype_bond = bonds.dtype
    stack.bonds = np.array(bonds, dtype=bonds.dtype, copy=True)
    stack.model_ids = list(range(coord.shape[0])) if model_ids is None else [int(x) for x in model_ids]
    if len(stack.model_ids) != coord.shape[0]:
      raise ValueError("model_ids must match the number of models in coord.")
    return stack

  def _model_to_structure(self, model_index: int) -> Structure:
    """Materialize a single model from the shared stack representation."""
    atoms = self._atoms_from_coord_matrix(self.coord[model_index], self._dtype_atoms)
    metadata = dict(self.metadata)
    metadata["model_id"] = self.model_ids[model_index]
    return self._structure_from_parts(atoms, self.atom_annotations, self.bonds, metadata=metadata)

  @staticmethod
  def _coord_matrix_from_structure(model: Structure) -> np.ndarray:
    """Extract an ``(n_atoms, 3)`` coordinate matrix from a structure."""
    if len(model.atoms) == 0:
      return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack((model.atoms["x"], model.atoms["y"], model.atoms["z"])).astype(np.float32, copy=False)

  @staticmethod
  def _atoms_from_coord_matrix(coord: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Create a coordinate structured array from an ``(n_atoms, 3)`` matrix."""
    coord = np.asarray(coord, dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] != 3:
      raise ValueError("Coordinate matrix must have shape (n_atoms, 3).")

    atoms = np.empty(coord.shape[0], dtype=dtype)
    atoms["x"] = coord[:, 0]
    atoms["y"] = coord[:, 1]
    atoms["z"] = coord[:, 2]
    return atoms

  @staticmethod
  def _structure_from_parts(
    atoms: np.ndarray,
    atom_annotations: np.ndarray,
    bonds: np.ndarray,
    metadata: Optional[Mapping[str, Any]] = None,
  ) -> Structure:
    """Build an independent ``Structure`` from array components."""
    model = Structure(remove_annotations=False)
    model._dtype_atoms = atoms.dtype
    model._dtype_atom_annotations = atom_annotations.dtype
    model._dtype_bond = bonds.dtype
    model.atoms = np.array(atoms, dtype=atoms.dtype, copy=True)
    model.atom_annotations = np.array(atom_annotations, dtype=atom_annotations.dtype, copy=True)
    model.bonds = np.array(bonds, dtype=bonds.dtype, copy=True)
    model.metadata = dict(metadata or {})
    return model

  @staticmethod
  def _ensure_stack_compatible(reference: Structure, candidate: Structure):
    """Validate that two structures can coexist in the same stack."""
    if len(reference) != len(candidate):
      raise ValueError("StructureStack requires each model to have the same number of atoms.")
    if reference._dtype_atom_annotations != candidate._dtype_atom_annotations:
      raise ValueError("StructureStack requires identical annotation schemas across all models.")
    if not np.array_equal(reference.atom_annotations, candidate.atom_annotations):
      raise ValueError("StructureStack requires identical atom annotations across all models.")
    if reference.bonds.dtype != candidate.bonds.dtype or not np.array_equal(reference.bonds, candidate.bonds):
      raise ValueError("StructureStack requires identical bonds across all models.")
