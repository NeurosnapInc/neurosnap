"""Data structures for representing molecular coordinates and annotations.

This module provides a single-model :class:`Structure`, immutable hierarchy
views (:class:`Chain`, :class:`Residue`, and :class:`Atom`), an ordered
multi-model container (:class:`StructureEnsemble`), and a shared-annotation
multi-model fast path (:class:`StructureStack`).

The universal length unit is Å.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from neurosnap.constants import AA_MASS_PROTEIN_AVG, AA_RECORDS, BACKBONE_ATOMS_DNA, BACKBONE_ATOMS_RNA, NUC_DNA_CODES, NUC_RNA_CODES, STANDARD_NUCLEOTIDES

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

  def __repr__(self) -> str:
    """Return a compact string summary of the structure."""
    chain_ids = _structure_chain_ids(self)
    return f"<Structure: Models=1 Chains=[{', '.join(chain_ids)}] Atoms={len(self)}>"

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
          )
        )
      chains.append(Chain(chain_id=chain_id, _residues=tuple(residues)))
    return chains

  def renumber(self, chain: Optional[str] = None, start: int = 1):
    """Renumber residues in-place.

    Parameters:
      chain: Chain ID to renumber. If ``None``, all chains are renumbered in
        chain order using one continuous counter.
      start: Starting residue number.
    """
    if chain is not None and chain not in self._available_chain_ids():
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
      ValueError: If no selected atoms have a known amino-acid residue mass.
    """
    atom_mask = self._atom_mask(chains=chains)
    if not np.any(atom_mask):
      raise ValueError("No atoms were found in the selected structure.")

    coord = self._coord_matrix(atom_mask=atom_mask)
    masses = self._atom_masses(atom_mask=atom_mask)
    known_mass_mask = masses > 0.0
    if not np.any(known_mass_mask):
      raise ValueError("No atoms with known amino-acid residue masses were found in the selected structure.")

    return np.average(coord[known_mass_mask], axis=0, weights=masses[known_mass_mask])

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

  def _available_chain_ids(self) -> List[str]:
    """Return chain identifiers present in the structure in first-seen order."""
    chain_ids = []
    seen = set()
    for chain_id in self.atom_annotations["chain_id"]:
      chain_id = str(chain_id)
      if chain_id in seen:
        continue
      seen.add(chain_id)
      chain_ids.append(chain_id)
    return chain_ids

  def _atom_mask(self, chains: Optional[List[str]] = None) -> np.ndarray:
    """Return a boolean mask for atoms in the selected chains."""
    if chains is None:
      return np.ones(len(self), dtype=bool)

    selected_chains = [str(chain_id) for chain_id in chains]
    missing_chains = sorted(set(selected_chains) - set(self._available_chain_ids()))
    if missing_chains:
      raise ValueError(f'Chain(s) {", ".join(missing_chains)} were not found in the structure.')

    return np.isin(self.atom_annotations["chain_id"], selected_chains)

  def _coord_matrix(self, atom_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return atom coordinates as an ``(n_atoms, 3)`` matrix."""
    if atom_mask is None:
      atom_mask = np.ones(len(self), dtype=bool)
    return np.column_stack((self.atoms["x"][atom_mask], self.atoms["y"][atom_mask], self.atoms["z"][atom_mask])).astype(np.float32, copy=False)

  def _atom_masses(self, atom_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return per-atom pseudo-masses derived from amino-acid residue weights.

    Residue weights from :data:`AA_MASS_PROTEIN_AVG` are distributed evenly
    across the atoms present in each amino-acid residue. Non-amino-acid
    residues receive a mass of zero.
    """
    if atom_mask is None:
      atom_mask = np.ones(len(self), dtype=bool)

    selected_indices = np.flatnonzero(atom_mask)
    masses = np.zeros(len(selected_indices), dtype=np.float32)
    residue_counts: Dict[Tuple[str, int, str, str, bool], int] = {}
    residue_masses: Dict[Tuple[str, int, str, str, bool], float] = {}

    for atom_index in selected_indices:
      residue_key = (
        self._annotation_value("chain_id", atom_index),
        self._annotation_value("res_id", atom_index),
        self._annotation_value("ins_code", atom_index),
        self._annotation_value("res_name", atom_index),
        self._annotation_value("hetero", atom_index),
      )
      residue_counts[residue_key] = residue_counts.get(residue_key, 0) + 1

    for residue_key, atom_count in residue_counts.items():
      residue_name = residue_key[3]
      residue_record = AA_RECORDS.get(residue_name)
      residue_mass = 0.0
      if residue_record is not None:
        residue_code = residue_record.code
        if residue_code is None and residue_record.standard_equiv_abr is not None:
          residue_code = AA_RECORDS[residue_record.standard_equiv_abr].code
        if residue_code is not None:
          residue_mass = float(AA_MASS_PROTEIN_AVG.get(residue_code, 0.0))
      residue_masses[residue_key] = residue_mass / atom_count if atom_count else 0.0

    for index, atom_index in enumerate(selected_indices):
      residue_key = (
        self._annotation_value("chain_id", atom_index),
        self._annotation_value("res_id", atom_index),
        self._annotation_value("ins_code", atom_index),
        self._annotation_value("res_name", atom_index),
        self._annotation_value("hetero", atom_index),
      )
      masses[index] = residue_masses[residue_key]

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
        raise ValueError(
          f'Chain "{self.chain_id}" mixes polymer residue types; found both "{detected_polymer_type}" and "{residue_polymer_type}".'
        )

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
    """
    residue_ids = sorted({residue.res_id for residue in self._residues if not residue.hetero})
    missing_ids = []
    for index in range(len(residue_ids) - 1):
      current_residue_id = residue_ids[index]
      next_residue_id = residue_ids[index + 1]
      if next_residue_id > current_residue_id + 1:
        missing_ids.extend(range(current_residue_id + 1, next_residue_id))
    return missing_ids


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
