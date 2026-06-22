"""Data structures for representing molecular coordinates and annotations.

This module provides a single-model :class:`Structure`, immutable hierarchy
views (:class:`Chain`, :class:`Residue`, and :class:`Atom`), an ordered
multi-model container (:class:`StructureEnsemble`), and a shared-annotation
multi-model fast path (:class:`StructureStack`).

The universal length unit is Å.
"""

from collections.abc import Callable, Sequence
from dataclasses import field
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from neurosnap._compat import compat_dataclass
from neurosnap.constants.chemistry import ATOMIC_MASSES
from neurosnap.constants.sequence import AA_RECORDS
from neurosnap.constants.structure import BACKBONE_ATOMS_DNA, BACKBONE_ATOMS_RNA, NUC_DNA_CODES, NUC_RNA_CODES, STANDARD_NUCLEOTIDES
from neurosnap.log import logger

### IMPORTANT NOTES
# Universal unit is Å.
# This new Structure object does not care about altlocs and will automatically drop them
# Hetatoms are stored with proper bond information
# In PDB files repeated bonds will correspond to bonds being interpreted at a higher order. For instance if atom i and j have two records for bonds in a PDB file this will be interpreted as them having a double bond.
# Each structure corresponds to a single model ONLY, the StructureEnsemble object should be used instead for an ordered collection of models (OR optional later: StructureStack = shared-annotation multi-model fast path, only when all models have identical atoms/bonds)

_STRUCTURE_DATAFRAME_COLUMNS = (
  "chain",
  "res_id",
  "ins_code",
  "res_name",
  "hetero",
  "res_type",
  "atom",
  "atom_name",
  "element",
  "bfactor",
  "occupancy",
  "charge",
  "sym_id",
  "x",
  "y",
  "z",
  "mass",
)


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

  def __init__(self, *, remove_annotations: bool = True):
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

    assert isinstance(remove_annotations, bool) # prevent users from mistakenly initialize the class with the incorrect type
    if remove_annotations is True:
      self._remove_empty_annotations()

  def __len__(self) -> int:
    """Return the number of atoms in the structure."""
    return len(self.atoms)

  def __repr__(self) -> str:
    """Return a compact string summary of the structure."""
    chains = self.chains()
    chain_ids = [chain.chain_id if chain.chain_id else "<blank>" for chain in chains]
    residue_count = sum(len(chain.residues()) for chain in chains)
    return f"<Structure: Chains=[{','.join(chain_ids)}] Residues={residue_count} Atoms={len(self)}>"

  def __iter__(self) -> Iterator["Chain"]:
    """Iterate over chains in atom-table order."""
    return iter(self.chains())

  def __getitem__(self, chain_id: str) -> "Chain":
    """Return a chain view by chain ID.

    Parameters:
      chain_id: Chain identifier to retrieve.

    Returns:
      The matching :class:`Chain` view.

    Raises:
      TypeError: If ``chain_id`` is not a string.
      KeyError: If the requested chain is not present in the structure.
    """
    if not isinstance(chain_id, str):
      raise TypeError("Structure indices must be chain IDs as strings.")

    for chain in self.chains():
      if chain.chain_id == chain_id:
        return chain
    raise KeyError(f'Chain "{chain_id}" was not found.')

  def select(
    self,
    *,
    chains: Optional[Sequence[str]] = None,
    residues: Optional[Sequence[Union[int, "Residue", Tuple[Any, ...]]]] = None,
    predicate: Optional[Callable[["Atom"], bool]] = None,
  ) -> "Structure":
    """Return an independent atom-level subset of the structure.

    The returned structure preserves the selected atoms exactly as parsed:
    coordinates, atom serials, residue identifiers, optional annotations, and
    any bonds whose endpoints remain in the subset. Bond indices are remapped
    onto the new atom table automatically so the subset can be exported
    directly with :meth:`save_pdb` or :meth:`save_cif`.

    Parameters:
      chains: Optional chain IDs to keep. If ``None``, atoms from all chains
        remain eligible for selection.
      residues: Optional residue selectors to keep. Supported selector forms
        are:
          - integer residue IDs, matched across all selected chains
          - :class:`Residue` objects
          - ``(chain_id, res_id)`` tuples
          - ``(chain_id, res_id, ins_code)`` tuples
          - full residue-key tuples
            ``(chain_id, res_id, ins_code, res_name, hetero)``
      predicate: Optional atom-level predicate. When provided, each atom is
        exposed as an immutable :class:`Atom` view and kept only if the
        predicate returns a truthy value.

    Returns:
      A new :class:`Structure` containing only atoms that satisfy every
      provided filter.

    Raises:
      ValueError: If a requested chain or residue selector is not present in
        the structure.
      TypeError: If ``predicate`` is not callable or a residue selector has an
        unsupported type/shape.
    """
    atom_mask = _structure_selection_mask(self, chains=chains, residues=residues, predicate=predicate)
    return _subset_structure(self, atom_mask)

  def save_pdb(self, pdb):
    """Write the structure directly as a PDB file.

    This is a convenience wrapper around
    :func:`neurosnap.io.pdb.save_pdb`. It is especially useful after
    :meth:`select`, because the selected structure can be exported without
    rebuilding a new container manually.

    Parameters:
      pdb: Output filepath or open writable file handle.
    """
    from neurosnap.io.pdb import save_pdb

    save_pdb(self, pdb)

  def save_cif(self, cif, *, minimal: bool = False):
    """Write the structure directly as an mmCIF file.

    This is a convenience wrapper around
    :func:`neurosnap.io.mmcif.save_cif`. It preserves the current atom table
    and metadata exactly as stored on the structure.

    Parameters:
      cif: Output filepath or open writable file handle.
      minimal: If ``True``, emit compact atom-site-only mmCIF output. If
        ``False`` (default), include entity/polymer/subchain metadata.
    """
    from neurosnap.io.mmcif import save_cif

    save_cif(self, cif, minimal=minimal)

  def to_dataframe(self) -> pd.DataFrame:
    """Export the structure as a pandas dataframe.

    This dataframe is derived on demand from the current atom table and is
    never cached on the structure.
    """
    atom_count = len(self)
    data = {
      "chain": self._annotation_export("chain_id"),
      "res_id": self._annotation_export("res_id"),
      "ins_code": self._annotation_export("ins_code"),
      "res_name": self._annotation_export("res_name"),
      "hetero": self._annotation_export("hetero"),
      "res_type": self._residue_types(),
      "atom": self._annotation_export("atom_id"),
      "atom_name": self._annotation_export("atom_name"),
      "element": self._annotation_export("element"),
      "bfactor": self._annotation_export("b_factor"),
      "occupancy": self._annotation_export("occupancy"),
      "charge": self._annotation_export("charge"),
      "sym_id": self._annotation_export("sym_id"),
      "x": self.atoms["x"].copy() if atom_count else np.zeros(0, dtype=np.float32),
      "y": self.atoms["y"].copy() if atom_count else np.zeros(0, dtype=np.float32),
      "z": self.atoms["z"].copy() if atom_count else np.zeros(0, dtype=np.float32),
      "mass": self._atom_masses(),
    }
    return pd.DataFrame(data, columns=_STRUCTURE_DATAFRAME_COLUMNS)

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
            _atom_indices=tuple(atom_indices),
          )
        )
      chains.append(Chain(chain_id=chain_id, _residues=tuple(residues)))
    return chains

  def chain_ids(self) -> List[str]:
    """Return all chains IDs found in the structure.

    Returns:
      List of strings for each chain.
    """
    return [str(x) for x in np.unique(self.atom_annotations["chain_id"])]

  def renumber(self, chain: Optional[str] = None, start: int = 1):
    """Renumber residues in-place.

    Parameters:
      chain: Chain ID to renumber. If ``None``, all chains are renumbered in
        chain order using one continuous counter.
      start: Starting residue number.

    Notes:
      Renumbering treats inserted residues as ordinary sequential residues and
      clears their insertion codes. For example, residues ``10``, ``10A``, and
      ``10B`` become ``1``, ``2``, and ``3`` (with empty insertion codes) when
      renumbered with ``start=1``.
    """
    if chain is not None and chain not in self.chain_ids():
      raise ValueError(f'Chain "{chain}" was not found in the structure.')

    residue_number = int(start)
    residue_map: Dict[Tuple[str, int, str, str, bool], int] = {}
    for chain_view in self.chains():
      if chain is not None and chain_view.chain_id != chain:
        continue
      for residue in chain_view.residues():
        residue_key = (residue.chain_id, residue.res_id, residue.ins_code, residue.res_name, residue.hetero)
        residue_map[residue_key] = residue_number
        residue_number += 1

    if not residue_map:
      return

    for atom_index in range(len(self)):
      residue_key = (
        self._annotation_value("chain_id", atom_index),
        self._annotation_value("res_id", atom_index),
        self._annotation_value("ins_code", atom_index),
        self._annotation_value("res_name", atom_index),
        self._annotation_value("hetero", atom_index),
      )
      if residue_key in residue_map:
        self.atom_annotations["res_id"][atom_index] = residue_map[residue_key]
        self.atom_annotations["ins_code"][atom_index] = ""

  def translate(
    self,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    chains: Optional[List[str]] = None,
  ):
    """Translate selected atoms in-place by a fixed vector.

    Parameters:
      x: Translation along the x-axis.
      y: Translation along the y-axis.
      z: Translation along the z-axis.
      chains: Optional chain IDs to translate. If ``None``, all atoms are
        translated.
    """
    atom_mask = self._atom_mask(chains=chains)
    self.atoms["x"][atom_mask] += float(x)
    self.atoms["y"][atom_mask] += float(y)
    self.atoms["z"][atom_mask] += float(z)

  def center_at(
    self,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    chains: Optional[List[str]] = None,
  ):
    """Translate selected atoms so their center of mass matches a target point.

    Parameters:
      x: Target x-coordinate for the center of mass.
      y: Target y-coordinate for the center of mass.
      z: Target z-coordinate for the center of mass.
      chains: Optional chain IDs to center. If ``None``, all atoms are used.
    """
    target = np.array([x, y, z], dtype=np.float32)
    center_of_mass = self.calculate_center_of_mass(chains=chains)
    translation = target - center_of_mass
    self.translate(x=float(translation[0]), y=float(translation[1]), z=float(translation[2]), chains=chains)

  def calculate_center_of_mass(self, chains: Optional[List[str]] = None) -> np.ndarray:
    """Calculate the center of mass for the selected atoms.

    Parameters:
      chains: Optional chain IDs to include. If ``None``, all atoms are used.

    Returns:
      A length-3 NumPy array containing the center of mass in Å.

    Raises:
      ValueError: If no atoms are found in the selected structure or if any
        selected atom has an unknown element mass.
    """
    atom_mask = self._atom_mask(chains=chains)
    if not np.any(atom_mask):
      raise ValueError("No atoms were found in the selected structure.")

    coord = self._coord_matrix(atom_mask=atom_mask)
    masses = self._atom_masses(atom_mask=atom_mask)
    return np.average(coord, axis=0, weights=masses)

  def calculate_geometric_center(self, chains: Optional[List[str]] = None) -> np.ndarray:
    """Calculate the geometric center for the selected atoms.

    Parameters:
      chains: Optional chain IDs to include. If ``None``, all atoms are used.

    Returns:
      A length-3 NumPy array containing the arithmetic mean of the selected
      atom coordinates in Å.

    Raises:
      ValueError: If no atoms are found in the selected structure.
    """
    atom_mask = self._atom_mask(chains=chains)
    if not np.any(atom_mask):
      raise ValueError("No atoms were found in the selected structure.")

    coord = self._coord_matrix(atom_mask=atom_mask)
    return coord.mean(axis=0)

  def distances_from(self, point: np.ndarray, chains: Optional[List[str]] = None) -> np.ndarray:
    """Calculate distances from a point for the selected atoms.

    Parameters:
      point: Reference point as an array-like object with shape ``(3,)``.
      chains: Optional chain IDs to include. If ``None``, all atoms are used.

    Returns:
      A 1D NumPy array containing Euclidean distances in atom-table order.
    """
    point = np.asarray(point, dtype=np.float32)
    if point.shape != (3,):
      raise ValueError("Point must be an array-like object with shape (3,).")

    atom_mask = self._atom_mask(chains=chains)
    coord = self._coord_matrix(atom_mask=atom_mask)
    if coord.size == 0:
      return np.zeros(0, dtype=np.float32)

    return np.linalg.norm(coord - point, axis=1)

  def calculate_rog(self, chains: Optional[List[str]] = None, center: Optional[np.ndarray] = None) -> float:
    """Calculate the radius of gyration for the selected atoms.

    Parameters:
      chains: Optional chain IDs to include. If ``None``, all atoms are used.
      center: Optional reference point. If ``None``, the center of mass is used.

    Returns:
      Radius of gyration in Å.
    """
    atom_mask = self._atom_mask(chains=chains)
    if not np.any(atom_mask):
      return 0.0

    if center is None:
      center = self.calculate_center_of_mass(chains=chains)
    distances = self.distances_from(center, chains=chains)
    if distances.size == 0:
      return 0.0

    return float(np.sqrt(np.mean(distances**2)))

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

  def _annotation_export(self, name: str) -> np.ndarray:
    """Return an annotation column or a default-filled export column."""
    if name in self._dtype_atom_annotations.names:
      return self.atom_annotations[name].copy()

    default_value = self._ANNOTATION_DEFAULTS[name]
    return np.full(len(self), default_value)

  def _residue_types(self) -> np.ndarray:
    """Return per-atom residue-type labels for dataframe export."""
    residue_types = np.full(len(self), "HETEROGEN", dtype="U12")
    if len(self) == 0:
      return residue_types

    hetero = self._annotation_export("hetero")
    res_name = self._annotation_export("res_name")
    for atom_index in range(len(self)):
      if bool(hetero[atom_index]):
        continue
      residue_name = str(res_name[atom_index])
      if residue_name in STANDARD_NUCLEOTIDES:
        residue_types[atom_index] = "NUCLEOTIDE"
      elif residue_name in AA_RECORDS:
        residue_types[atom_index] = "AMINO_ACID"
    return residue_types

  def _atom_mask(self, chains: Optional[List[str]] = None) -> np.ndarray:
    """Return a boolean mask for atoms in the selected chains."""
    if chains is None:
      return np.ones(len(self), dtype=bool)

    selected_chains = [str(chain_id) for chain_id in chains]
    missing_chains = sorted(set(selected_chains) - set(self.chain_ids()))
    if missing_chains:
      raise ValueError(f'Chain(s) {", ".join(missing_chains)} were not found in the structure.')

    return np.isin(self.atom_annotations["chain_id"], selected_chains)

  def _coord_matrix(self, atom_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return atom coordinates as an ``(n_atoms, 3)`` matrix."""
    if atom_mask is None:
      atom_mask = np.ones(len(self), dtype=bool)
    return np.column_stack((self.atoms["x"][atom_mask], self.atoms["y"][atom_mask], self.atoms["z"][atom_mask])).astype(np.float32, copy=False)

  def _atom_masses(self, atom_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return per-atom masses derived from element symbols."""
    if atom_mask is None:
      atom_mask = np.ones(len(self), dtype=bool)

    selected_indices = np.flatnonzero(atom_mask)
    masses = np.zeros(len(selected_indices), dtype=np.float32)
    unknown_elements = set()

    for index, atom_index in enumerate(selected_indices):
      element = str(self._annotation_value("element", atom_index)).strip()
      if element not in ATOMIC_MASSES:
        unknown_elements.add(element or "<blank>")
        continue
      masses[index] = float(ATOMIC_MASSES[element])

    if unknown_elements:
      message = (
        "Unknown element mass for atom selection: " + ", ".join(sorted(unknown_elements)) + ". This is likely an error in the input structure."
      )
      logger.warning(message)
      raise ValueError(message)

    return masses

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


@compat_dataclass(frozen=True, slots=True)
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


@compat_dataclass(frozen=True, slots=True)
class Residue:
  """Immutable residue-level hierarchy view.

  A :class:`Residue` groups atoms that share the same chain identifier,
  residue number, insertion code, residue name, and hetero flag. The object is
  a lightweight read-only view over the parsed atom table, intended for
  traversal and analysis rather than in-place editing.

  Attributes:
    chain_id: Chain identifier containing the residue.
    res_id: Residue sequence number.
    ins_code: PDB insertion code for the residue.
    res_name: Residue name / CCD code.
    hetero: ``True`` for heterogens and ``False`` for polymer ``ATOM`` records.
  """

  chain_id: str
  res_id: int
  ins_code: str
  res_name: str
  hetero: bool
  _atoms: Tuple[Atom, ...] = field(repr=False)
  _atom_indices: Tuple[int, ...] = field(repr=False)

  def atoms(self) -> List[Atom]:
    """Return the atoms that belong to this residue.

    Returns:
      List of immutable :class:`Atom` views in atom-table order.
    """
    return list(self._atoms)

  def atom_indices(self) -> List[int]:
    """Return atom-table indices for the atoms in this residue.

    Returns:
      List of integer atom indices in atom-table order.
    """
    return list(self._atom_indices)

  def key(self) -> Tuple[str, int, str, str, bool]:
    """Return a stable residue identity tuple.

    The returned key is suitable for dictionary/set membership when residue
    identity needs to be tracked outside the object itself.

    Returns:
      ``(chain_id, res_id, ins_code, res_name, hetero)``
    """
    return (self.chain_id, self.res_id, self.ins_code, self.res_name, self.hetero)

  def __hash__(self):
    """Return a hash derived from :meth:`key`."""
    return hash(self.key())

  def __eq__(self, other):
    """Compare two residue views by stable identity."""
    if not isinstance(other, Residue):
      return NotImplemented
    return self.key() == other.key()


@compat_dataclass(frozen=True, slots=True)
class Chain:
  """Immutable chain-level hierarchy view.

  A :class:`Chain` is a read-only hierarchy view over the residues associated
  with one chain identifier in a single :class:`Structure`. It provides chain-
  level traversal plus convenience helpers for sequence extraction and simple
  residue-number gap detection.

  Attributes:
    chain_id: Chain identifier represented by this view.
  """

  chain_id: str
  _residues: Tuple[Residue, ...] = field(repr=False)

  def __iter__(self) -> Iterator[Residue]:
    """Iterate over residues in residue order."""
    return iter(self._residues)

  def residues(self) -> List[Residue]:
    """Return the residues that belong to this chain.

    Returns:
      List of immutable :class:`Residue` views in residue order.
    """
    return list(self._residues)

  def __getitem__(self, res_id: int) -> Residue:
    """Return a residue view by residue ID, not by positional index.

    Parameters:
      res_id: Residue sequence number to retrieve.

    Returns:
      The first :class:`Residue` in this chain with the requested residue ID.

    Raises:
      TypeError: If ``res_id`` is not an integer residue ID.
      KeyError: If no residue with the requested ID is present in the chain.

    Notes:
      This method looks up residues by their residue ID rather than by list
      position. If multiple residues share the same residue ID, such as
      inserted residues distinguished by insertion codes, the first matching
      residue is returned and a warning is emitted.
    """
    if not isinstance(res_id, (int, np.integer)):
      raise TypeError("Chain indices must be residue IDs as integers.")

    matches = [residue for residue in self._residues if residue.res_id == int(res_id)]
    if not matches:
      raise KeyError(f'Residue ID {res_id} was not found in chain "{self.chain_id}".')
    if len(matches) > 1:
      logger.warning(
        'Chain "%s" contains multiple residues with residue ID %d; returning the first match.',
        self.chain_id,
        int(res_id),
      )
    return matches[0]

  def sequence(
    self,
    polymer_type: Literal["auto", "protein", "dna", "rna", "nucleotide"] = "auto",
    include_modifications: bool = False,
    modification_mode: Literal["inline", "parent"] = "inline",
    on_unknown_modified: Literal["raise", "unknown"] = "raise",
  ) -> str:
    """Return the polymer sequence for this chain.

    Protein, DNA, and RNA sequences are supported. Small molecules and other
    non-polymer residues in the chain are ignored. Modified residues can either
    be skipped, emitted inline as ``(CCD)``, or mapped to their parent
    sequence code when available.

    Parameters:
      polymer_type: Polymer family to extract. ``"auto"`` infers the family
        from the chain contents. ``"nucleotide"`` accepts either DNA or RNA,
        but raises if both are present.
      include_modifications: Whether modified residues should contribute to the
        sequence. If ``False``, modified residues are skipped entirely.
      modification_mode: How included modifications are emitted. ``"inline"``
        inserts ``(CCD)`` tokens, while ``"parent"`` uses the inferred parent
        residue code.
      on_unknown_modified: Behavior when ``modification_mode="parent"`` is
        requested but no parent code can be inferred. ``"raise"`` raises a
        :class:`ValueError`; ``"unknown"`` inserts ``"X"``.

    Returns:
      Sequence string for the selected polymer family. Returns an empty string
      if the chain contains no residues from the requested polymer family.

    Raises:
      ValueError: If the chain mixes polymer families in a way that conflicts
        with ``polymer_type`` or if an unknown modified residue cannot be
        mapped in ``"parent"`` mode.
    """
    if polymer_type not in {"auto", "protein", "dna", "rna", "nucleotide"}:
      raise ValueError('polymer_type must be one of "auto", "protein", "dna", "rna", or "nucleotide".')
    if modification_mode not in {"inline", "parent"}:
      raise ValueError('modification_mode must be either "inline" or "parent".')
    if on_unknown_modified not in {"raise", "unknown"}:
      raise ValueError('on_unknown_modified must be either "raise" or "unknown".')

    detected_polymer_type = None if polymer_type == "auto" else polymer_type
    sequence_parts = []

    for residue in self._residues:
      residue_polymer_type = _classify_polymer_residue(residue)
      if residue_polymer_type is None:
        continue

      # Auto-detection locks onto the first polymer family encountered and then
      # rejects incompatible mixtures later in the chain.
      if detected_polymer_type is None:
        detected_polymer_type = residue_polymer_type
      elif not _polymer_types_compatible(detected_polymer_type, residue_polymer_type):
        raise ValueError(f'Chain "{self.chain_id}" mixes polymer residue types; found both "{detected_polymer_type}" and "{residue_polymer_type}".')

      if polymer_type != "auto" and not _polymer_types_compatible(polymer_type, residue_polymer_type):
        raise ValueError(
          f'Chain "{self.chain_id}" contains "{residue_polymer_type}" residues, which are incompatible with polymer_type="{polymer_type}".'
        )

      residue_name = residue.res_name.strip().upper()
      if residue_polymer_type == "protein":
        # Standard amino acids map directly to one-letter codes. Modified amino
        # acids either get skipped, emitted inline as CCD tokens, or mapped to
        # their declared parent residue when available.
        residue_record = AA_RECORDS[residue_name]
        if residue_record.code is not None:
          residue_token = residue_record.code
        elif not include_modifications:
          residue_token = None
        elif modification_mode == "inline":
          residue_token = f"({residue_record.abr})"
        else:
          residue_token = None
          if residue_record.standard_equiv_abr is not None:
            parent_record = AA_RECORDS.get(residue_record.standard_equiv_abr)
            if parent_record is not None:
              residue_token = parent_record.code
          if residue_token is None:
            if on_unknown_modified == "unknown":
              residue_token = "X"
            else:
              raise ValueError(f'Could not infer a parent sequence code for modified residue "{residue_name}".')
      else:
        # Nucleotide handling mirrors the protein path but uses canonical
        # one-letter base codes and a simple parent-code inference fallback for
        # modified residues.
        if residue_name in NUC_RNA_CODES:
          residue_token = residue_name
        elif residue_name in NUC_DNA_CODES:
          residue_token = residue_name[1]
        elif not include_modifications:
          residue_token = None
        elif modification_mode == "inline":
          residue_token = f"({residue_name})"
        else:
          allowed_codes = {"A", "C", "G", "T"} if residue_polymer_type == "dna" else {"A", "C", "G", "U"}
          residue_token = next((char for char in reversed(residue_name) if char in allowed_codes), None)
          if residue_token is None:
            if on_unknown_modified == "unknown":
              residue_token = "X"
            else:
              raise ValueError(f'Could not infer a parent sequence code for modified residue "{residue_name}".')
      if residue_token is not None:
        sequence_parts.append(residue_token)

    return "".join(sequence_parts)

  def missing_residue_ids(self) -> List[int]:
    """Return missing residue numbers inferred from gaps in the chain.

    Hetero residues are ignored so ligand or solvent numbering does not create
    artificial gaps in the polymer residue sequence.

    Returns:
      Sorted list of integer residue IDs that are absent between observed
      non-hetero residue numbers.
    """
    residue_ids = sorted({residue.res_id for residue in self._residues if not residue.hetero})
    missing_ids = []
    for index in range(len(residue_ids) - 1):
      current_residue_id = residue_ids[index]
      next_residue_id = residue_ids[index + 1]
      if next_residue_id > current_residue_id + 1:
        missing_ids.extend(range(current_residue_id + 1, next_residue_id))
    return missing_ids


def _normalize_requested_model_ids(existing_model_ids: Sequence[int], requested_model_ids: Optional[Sequence[int]]) -> set[int]:
  """Return validated model identifiers requested for selection.

  Parameters:
    existing_model_ids: Model IDs currently present in an ensemble or stack.
    requested_model_ids: Optional model IDs requested by the caller. ``None``
      means all existing models should be kept.

  Returns:
    Set of integer model IDs to retain.

  Raises:
    ValueError: If any requested model ID does not exist.
  """
  if requested_model_ids is None:
    return {int(model_id) for model_id in existing_model_ids}

  selected_model_ids = {int(model_id) for model_id in requested_model_ids}
  missing_model_ids = sorted(selected_model_ids - {int(model_id) for model_id in existing_model_ids})
  if missing_model_ids:
    raise ValueError(f"Model ID(s) {', '.join(str(model_id) for model_id in missing_model_ids)} were not found.")
  return selected_model_ids


def _normalize_residue_selector(selector: Union[int, Residue, Tuple[Any, ...]]) -> Tuple[Optional[str], int, Optional[str], Optional[str], Optional[bool]]:
  """Normalize one residue selector into comparable components.

  Parameters:
    selector: One supported residue selector supplied to :meth:`Structure.select`.

  Returns:
    Tuple ``(chain_id, res_id, ins_code, res_name, hetero)`` where each field
    may be ``None`` when the selector intentionally leaves that part
    unspecified.

  Raises:
    TypeError: If the selector does not match one of the supported forms.
  """
  if isinstance(selector, Residue):
    return selector.chain_id, int(selector.res_id), selector.ins_code, selector.res_name, bool(selector.hetero)
  if isinstance(selector, int):
    return None, int(selector), None, None, None
  if isinstance(selector, tuple):
    if len(selector) == 2:
      chain_id, res_id = selector
      return str(chain_id), int(res_id), None, None, None
    if len(selector) == 3:
      chain_id, res_id, ins_code = selector
      return str(chain_id), int(res_id), str(ins_code), None, None
    if len(selector) == 5:
      chain_id, res_id, ins_code, res_name, hetero = selector
      return str(chain_id), int(res_id), str(ins_code), str(res_name), bool(hetero)
  raise TypeError(
    "Residue selectors must be integers, Residue objects, "
    "(chain_id, res_id) tuples, (chain_id, res_id, ins_code) tuples, or full residue-key tuples."
  )


def _residue_matches_selector(
  residue_key: Tuple[str, int, str, str, bool],
  selector: Tuple[Optional[str], int, Optional[str], Optional[str], Optional[bool]],
) -> bool:
  """Return ``True`` when a residue key matches a normalized selector.

  Parameters:
    residue_key: Full residue identity tuple from a parsed structure.
    selector: Normalized selector tuple produced by
      :func:`_normalize_residue_selector`.
  """
  chain_id, res_id, ins_code, res_name, hetero = selector
  return (
    residue_key[1] == res_id
    and (chain_id is None or residue_key[0] == chain_id)
    and (ins_code is None or residue_key[2] == ins_code)
    and (res_name is None or residue_key[3] == res_name)
    and (hetero is None or residue_key[4] == hetero)
  )


def _selected_residue_keys(
  structure: Structure,
  residues: Sequence[Union[int, Residue, Tuple[Any, ...]]],
  chain_filter: Optional[set[str]] = None,
) -> set[Tuple[str, int, str, str, bool]]:
  """Return validated residue keys selected from a structure.

  Parameters:
    structure: Source structure being filtered.
    residues: Residue selectors requested by the caller.
    chain_filter: Optional set of chain IDs already selected upstream. When
      provided, residue matching is restricted to those chains.

  Returns:
    Set of full residue-key tuples present in ``structure``.

  Raises:
    ValueError: If any requested residue selector does not match at least one
      residue in the filtered structure view.
  """
  available_residue_keys = [residue.key() for chain in structure.chains() for residue in chain.residues()]
  if chain_filter is not None:
    available_residue_keys = [residue_key for residue_key in available_residue_keys if residue_key[0] in chain_filter]

  selected_residue_keys: set[Tuple[str, int, str, str, bool]] = set()
  unmatched_selectors = []
  for selector in residues:
    normalized_selector = _normalize_residue_selector(selector)
    matches = {residue_key for residue_key in available_residue_keys if _residue_matches_selector(residue_key, normalized_selector)}
    if not matches:
      unmatched_selectors.append(selector)
      continue
    selected_residue_keys.update(matches)

  if unmatched_selectors:
    formatted = ", ".join(repr(selector) for selector in unmatched_selectors)
    raise ValueError(f"Residue selector(s) were not found in the structure: {formatted}.")
  return selected_residue_keys


def _structure_selection_mask(
  structure: Structure,
  *,
  chains: Optional[Sequence[str]] = None,
  residues: Optional[Sequence[Union[int, Residue, Tuple[Any, ...]]]] = None,
  predicate: Optional[Callable[[Atom], bool]] = None,
) -> np.ndarray:
  """Return a boolean atom mask for a structure selection.

  The mask applies chain, residue, and predicate filters cumulatively. Each
  filter narrows the selection further; an atom must satisfy all supplied
  constraints to be retained.

  Parameters:
    structure: Source structure being filtered.
    chains: Optional chain IDs to keep.
    residues: Optional residue selectors to keep.
    predicate: Optional atom-level predicate applied to immutable
      :class:`Atom` views.

  Returns:
    One-dimensional boolean mask with one entry per atom in ``structure``.

  Raises:
    ValueError: If a requested chain or residue selector is not present.
    TypeError: If ``predicate`` is not callable.
  """
  atom_mask = np.ones(len(structure), dtype=bool)
  selected_chains: Optional[set[str]] = None

  if chains is not None:
    selected_chains = {str(chain_id) for chain_id in chains}
    missing_chains = sorted(selected_chains - set(structure.chain_ids()))
    if missing_chains:
      raise ValueError(f'Chain(s) {", ".join(missing_chains)} were not found in the structure.')
    atom_mask &= np.isin(structure.atom_annotations["chain_id"], list(selected_chains))

  if residues is not None:
    residue_keys = _selected_residue_keys(structure, residues, chain_filter=selected_chains)
    residue_mask = np.zeros(len(structure), dtype=bool)
    for atom_index in range(len(structure)):
      residue_key = (
        str(structure.atom_annotations["chain_id"][atom_index]),
        int(structure.atom_annotations["res_id"][atom_index]),
        str(structure.atom_annotations["ins_code"][atom_index]),
        str(structure.atom_annotations["res_name"][atom_index]),
        bool(structure.atom_annotations["hetero"][atom_index]),
      )
      if residue_key in residue_keys:
        residue_mask[atom_index] = True
    atom_mask &= residue_mask

  if predicate is not None:
    if not callable(predicate):
      raise TypeError("predicate must be callable.")
    predicate_mask = np.zeros(len(structure), dtype=bool)
    for atom_index in range(len(structure)):
      predicate_mask[atom_index] = bool(predicate(structure._atom_view(atom_index)))
    atom_mask &= predicate_mask

  return atom_mask


def _subset_structure(structure: Structure, atom_mask: np.ndarray) -> Structure:
  """Return a structure subset with bonds remapped onto the selected atoms.

  Parameters:
    structure: Source structure to subset.
    atom_mask: One-dimensional boolean array indicating which atoms to keep.

  Returns:
    A new independent :class:`Structure` with copied coordinates, copied atom
    annotations, copied metadata, and a bond table containing only bonds whose
    two endpoints are both retained.

  Raises:
    ValueError: If ``atom_mask`` does not have exactly one boolean entry per
      atom in ``structure``.
  """
  atom_mask = np.asarray(atom_mask, dtype=bool)
  if atom_mask.ndim != 1 or len(atom_mask) != len(structure):
    raise ValueError("Atom selection mask must be a one-dimensional boolean array with one entry per atom.")

  selected_indices = np.flatnonzero(atom_mask)
  index_map = {int(atom_index): new_index for new_index, atom_index in enumerate(selected_indices)}
  bond_rows = []
  for bond in structure.bonds:
    atom_i = int(bond["atom_i"])
    atom_j = int(bond["atom_j"])
    if atom_i in index_map and atom_j in index_map:
      bond_rows.append((index_map[atom_i], index_map[atom_j], int(bond["bond_type"])))

  subset = Structure(remove_annotations=False)
  subset.metadata = dict(structure.metadata)
  subset._dtype_atoms = structure._dtype_atoms
  subset._dtype_atom_annotations = structure._dtype_atom_annotations
  subset._dtype_bond = structure._dtype_bond
  subset.atoms = np.array(structure.atoms[atom_mask], dtype=structure._dtype_atoms, copy=True)
  subset.atom_annotations = np.array(structure.atom_annotations[atom_mask], dtype=structure._dtype_atom_annotations, copy=True)
  if bond_rows:
    subset.bonds = np.array(bond_rows, dtype=structure._dtype_bond)
  else:
    subset.bonds = np.zeros(0, dtype=structure._dtype_bond)
  return subset


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


def _structure_chain_ids(structure: Structure) -> List[str]:
  """Return chain identifiers from a structure in hierarchy order."""
  return [chain.chain_id if chain.chain_id else "<blank>" for chain in structure.chains()]


def _model_position_from_id(model_ids: List[int], model_id: int) -> int:
  """Return the positional index for a model identifier.

  Raises:
    KeyError: If the requested model identifier is not present.
  """
  try:
    return model_ids.index(int(model_id))
  except ValueError:
    raise KeyError(f"Model ID {model_id} was not found.")


def _polymer_types_compatible(requested_polymer_type: str, residue_polymer_type: str) -> bool:
  """Return ``True`` if a residue polymer family matches a requested family."""
  if requested_polymer_type == "nucleotide":
    return residue_polymer_type in {"dna", "rna"}
  return requested_polymer_type == residue_polymer_type


def _classify_polymer_residue(residue: Residue) -> Optional[str]:
  """Classify a residue as protein, DNA, RNA, or non-polymer."""
  residue_name = residue.res_name.strip().upper()
  if residue_name in AA_RECORDS:
    return "protein"
  if residue_name in NUC_DNA_CODES:
    return "dna"
  if residue_name in NUC_RNA_CODES:
    return "rna"

  atom_names = {atom.atom_name.strip().upper() for atom in residue._atoms}
  if "O2'" in atom_names:
    backbone_matches = len(atom_names.intersection({atom_name.upper() for atom_name in BACKBONE_ATOMS_RNA}))
    if backbone_matches >= 3:
      return "rna"
  backbone_matches = len(atom_names.intersection({atom_name.upper() for atom_name in BACKBONE_ATOMS_DNA}))
  if backbone_matches >= 3:
    return "dna"
  return None


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

  def __repr__(self) -> str:
    """Return a compact string summary of the ensemble."""
    seen = set()
    chain_ids = []
    for model in self._models:
      for chain_id in _structure_chain_ids(model):
        if chain_id in seen:
          continue
        seen.add(chain_id)
        chain_ids.append(chain_id)
    atom_count = sum(len(model) for model in self._models)
    return f"<Structure Ensemble: Models={len(self)} Chains=[{', '.join(chain_ids)}] Atoms={atom_count}>"

  def to_dataframe(self) -> pd.DataFrame:
    """Export the ensemble as a pandas dataframe with a ``model`` column.

    This dataframe is derived on demand from the current models and is never
    cached on the ensemble.
    """
    frames = []
    for model_id, model in zip(self.model_ids, self._models):
      frame = model.to_dataframe()
      frame.insert(0, "model", model_id)
      frames.append(frame)

    if not frames:
      return pd.DataFrame(columns=("model",) + _STRUCTURE_DATAFRAME_COLUMNS)
    return pd.concat(frames, ignore_index=True)

  def __iter__(self) -> Iterator[Structure]:
    """Iterate over the stored models in order."""
    return iter(self._models)

  def __getitem__(self, index):
    """Return a model by model ID or a sliced sub-ensemble by position.

    Integer access uses ``model_id`` lookup rather than positional indexing, so
    ``ensemble[5]`` returns the model whose ID is ``5``. Slice access keeps
    normal positional semantics to preserve standard Python iteration and
    slicing behavior.

    Raises:
      KeyError: If an integer model ID is requested but not present.
    """
    if isinstance(index, slice):
      return StructureEnsemble(self._models[index], model_ids=self.model_ids[index], metadata=self.metadata)
    model_position = _model_position_from_id(self.model_ids, index)
    return self._models[model_position]

  def append(self, model: Structure, *, model_id: Optional[int] = None):
    """Append a validated model to the ensemble.

    Parameters:
      model: Model to append.
      model_id: Optional model identifier. Defaults to the next sequential
        model ID starting at ``1``.
    """
    _validate_structure_model(model)
    assigned_model_id = len(self.model_ids) + 1 if model_id is None else int(model_id)
    self._models.append(model)
    self.model_ids.append(assigned_model_id)
    model.metadata["model_id"] = assigned_model_id

  def remove_model(self, model_id: int) -> Structure:
    """Remove and return a model by model ID.

    Parameters:
      model_id: Model identifier to remove.

    Returns:
      The removed :class:`Structure`.

    Raises:
      KeyError: If the requested model ID is not present.
    """
    model_position = _model_position_from_id(self.model_ids, model_id)
    self.model_ids.pop(model_position)
    return self._models.pop(model_position)

  def renumber(self, start: int = 1):
    """Renumber model identifiers in-place.

    Parameters:
      start: Starting model ID. Defaults to ``1``.
    """
    self.model_ids = list(range(int(start), int(start) + len(self)))
    for model_id, model in zip(self.model_ids, self._models):
      model.metadata["model_id"] = model_id

  def models(self) -> List[Structure]:
    """Return the models as a shallow copied list."""
    return list(self._models)

  def first(self) -> Structure:
    """Return the first model in the ensemble.

    Returns:
      The first :class:`Structure` in stored order.

    Raises:
      IndexError: If the ensemble is empty.
    """
    if not self._models:
      raise IndexError("Cannot fetch the first model from an empty StructureEnsemble.")
    return self._models[0]

  def select(
    self,
    *,
    models: Optional[Sequence[int]] = None,
    chains: Optional[Sequence[str]] = None,
    residues: Optional[Sequence[Union[int, Residue, Tuple[Any, ...]]]] = None,
    predicate: Optional[Callable[[Atom], bool]] = None,
  ) -> "StructureEnsemble":
    """Return a filtered ensemble of independently subsetted models.

    Parameters:
      models: Optional model IDs to keep. If ``None``, all models are
        considered.
      chains: Optional chain IDs to keep within each selected model.
      residues: Optional residue selectors to keep within each selected model.
      predicate: Optional atom-level predicate applied independently inside
        each selected model.

    Returns:
      A new :class:`StructureEnsemble` whose model IDs match the selected
      source models and whose per-model contents are the corresponding
      :meth:`Structure.select` subsets.

    Raises:
      ValueError: If any requested model ID, chain, or residue selector is not
        present.
      TypeError: If ``predicate`` is not callable or a residue selector is
        malformed.
    """
    selected_model_ids = _normalize_requested_model_ids(self.model_ids, models)
    subset = StructureEnsemble(metadata=self.metadata)
    for model_id, model in zip(self.model_ids, self._models):
      if model_id not in selected_model_ids:
        continue
      subset.append(model.select(chains=chains, residues=residues, predicate=predicate), model_id=model_id)
    return subset

  def save_pdb(self, pdb):
    """Write the ensemble directly as a PDB file.

    Parameters:
      pdb: Output filepath or open writable file handle.
    """
    from neurosnap.io.pdb import save_pdb

    save_pdb(self, pdb)

  def save_cif(self, cif, *, minimal: bool = False):
    """Write the ensemble directly as an mmCIF file.

    Parameters:
      cif: Output filepath or open writable file handle.
      minimal: If ``True``, emit compact atom-site-only mmCIF output. If
        ``False`` (default), include entity/polymer/subchain metadata.
    """
    from neurosnap.io.mmcif import save_cif

    save_cif(self, cif, minimal=minimal)

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

  def __repr__(self) -> str:
    """Return a compact string summary of the stack."""
    chain_ids = [] if len(self) == 0 else _structure_chain_ids(self._model_to_structure(0))
    atom_count = len(self) * self.atom_count
    return f"<Structure Stack: Models={len(self)} Chains=[{', '.join(chain_ids)}] Atoms={atom_count}>"

  def to_dataframe(self) -> pd.DataFrame:
    """Export the stack as a pandas dataframe with a ``model`` column.

    This dataframe is derived on demand from the current stack contents and is
    never cached on the stack.
    """
    frames = []
    for model_index, model_id in enumerate(self.model_ids):
      frame = self._model_to_structure(model_index).to_dataframe()
      frame.insert(0, "model", model_id)
      frames.append(frame)

    if not frames:
      return pd.DataFrame(columns=("model",) + _STRUCTURE_DATAFRAME_COLUMNS)
    return pd.concat(frames, ignore_index=True)

  def __iter__(self) -> Iterator[Structure]:
    """Iterate over the stack as materialized ``Structure`` models."""
    for model_index in range(len(self)):
      yield self._model_to_structure(model_index)

  def first(self) -> Structure:
    """Return the first model in the stack.

    Returns:
      The first :class:`Structure` in stored order.

    Raises:
      IndexError: If the stack is empty.
    """
    if len(self) == 0:
      raise IndexError("Cannot fetch the first model from an empty StructureStack.")
    return self._model_to_structure(0)

  def __getitem__(self, index):
    """Return a materialized model by model ID or a sliced sub-stack by position.

    Integer access uses ``model_id`` lookup rather than positional indexing, so
    ``stack[5]`` returns the model whose ID is ``5``. Slice access keeps normal
    positional semantics to preserve standard Python slicing behavior.

    Raises:
      KeyError: If an integer model ID is requested but not present.
    """
    if isinstance(index, slice):
      return StructureStack._from_parts(
        self.coord[index].copy(),
        self.atom_annotations.copy(),
        self.bonds.copy(),
        model_ids=self.model_ids[index],
        metadata=self.metadata,
      )
    model_position = _model_position_from_id(self.model_ids, index)
    return self._model_to_structure(model_position)

  @property
  def atom_count(self) -> int:
    """Return the number of shared atoms per model."""
    return self.coord.shape[1]

  def append(self, model: Structure, *, model_id: Optional[int] = None):
    """Append a stack-compatible model.

    Parameters:
      model: Model to append.
      model_id: Optional model identifier. Defaults to the next sequential
        model ID starting at ``1``.

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

    self.model_ids.append(len(self.model_ids) + 1 if model_id is None else int(model_id))

  def remove_model(self, model_id: int) -> Structure:
    """Remove and return a model by model ID.

    Parameters:
      model_id: Model identifier to remove.

    Returns:
      The removed :class:`Structure`.

    Raises:
      KeyError: If the requested model ID is not present.
    """
    model_position = _model_position_from_id(self.model_ids, model_id)
    removed_model = self._model_to_structure(model_position)
    self.coord = np.delete(self.coord, model_position, axis=0)
    self.model_ids.pop(model_position)
    return removed_model

  def renumber(self, start: int = 1):
    """Renumber model identifiers in-place.

    Parameters:
      start: Starting model ID. Defaults to ``1``.
    """
    self.model_ids = list(range(int(start), int(start) + len(self)))

  def models(self) -> List[Structure]:
    """Materialize and return all models in the stack."""
    return [self._model_to_structure(index) for index in range(len(self))]

  def select(
    self,
    *,
    models: Optional[Sequence[int]] = None,
    chains: Optional[Sequence[str]] = None,
    residues: Optional[Sequence[Union[int, Residue, Tuple[Any, ...]]]] = None,
    predicate: Optional[Callable[[Atom], bool]] = None,
  ) -> Union["StructureStack", StructureEnsemble]:
    """Return a filtered multi-model subset of the stack.

    The selection is executed through the ensemble path so each chosen model is
    subsetted with the same semantics as :meth:`Structure.select`. If the
    resulting models still share identical atom annotations and bonds, the
    return value is a :class:`StructureStack`; otherwise it falls back to a
    :class:`StructureEnsemble`.

    Parameters:
      models: Optional model IDs to keep. If ``None``, all models are
        considered.
      chains: Optional chain IDs to keep within each selected model.
      residues: Optional residue selectors to keep within each selected model.
      predicate: Optional atom-level predicate applied independently inside
        each selected model.

    Returns:
      A :class:`StructureStack` when the subset remains stack-compatible,
      otherwise a :class:`StructureEnsemble`.
    """
    subset_ensemble = self.to_ensemble().select(models=models, chains=chains, residues=residues, predicate=predicate)
    try:
      return StructureStack.from_ensemble(subset_ensemble)
    except ValueError:
      return subset_ensemble

  def save_pdb(self, pdb):
    """Write the stack directly as a PDB file.

    Parameters:
      pdb: Output filepath or open writable file handle.
    """
    from neurosnap.io.pdb import save_pdb

    save_pdb(self, pdb)

  def save_cif(self, cif, *, minimal: bool = False):
    """Write the stack directly as an mmCIF file.

    Parameters:
      cif: Output filepath or open writable file handle.
      minimal: If ``True``, emit compact atom-site-only mmCIF output. If
        ``False`` (default), include entity/polymer/subchain metadata.
    """
    from neurosnap.io.mmcif import save_cif

    save_cif(self, cif, minimal=minimal)

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
    stack.model_ids = list(range(1, coord.shape[0] + 1)) if model_ids is None else [int(x) for x in model_ids]
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
