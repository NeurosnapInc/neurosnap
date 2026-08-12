"""Topology-aware chemical-library characterization.

Reimplementation of the ChemManifold chemical-landscape algorithm by Danial
Gharaie Amirabadi. A molecular library is represented as a *multi-resolution
traversable chemical graph*:

- nodes: compounds, Bemis-Murcko scaffolds, fragments
- edges: compound -> scaffold, compound -> fragment, compound <-> compound
  Tanimoto similarity, scaffold hierarchy (general -> specific), fragment
  sharing through a common ring system

from neurosnap.chemistry import ChemicalLandscape

landscape = ChemicalLandscape("library.csv", smiles_column="smiles")
landscape.build_all()
report = landscape.characterize()
print(report.summary())
landscape.path_between("aspirin", "naproxen")
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import lzma
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem.Scaffolds import MurckoScaffold, rdScaffoldNetwork
from scipy.sparse import csgraph

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- taxonomy


class NodeType(IntEnum):
  """Node types of the heterogeneous chemical graph."""

  COMPOUND = 0
  SCAFFOLD = 1
  FRAGMENT = 2


class EdgeType(IntEnum):
  """Edge types of the heterogeneous chemical graph."""

  COMPOUND_SCAFFOLD = 1
  COMPOUND_FRAGMENT = 2
  COMPOUND_SIMILARITY = 3
  SCAFFOLD_HIERARCHY = 4  # parent (more general) -> child (more specific)
  FRAGMENT_SHARED = 5  # fragments sharing a ring system


class FragmentMethod(IntEnum):
  """Provenance of a fragment node."""

  UNKNOWN = 0
  BRICS = 1
  ROTATABLE_BOND = 2
  LINKER = 3


NODE_LABELS = {0: "Compound", 1: "Scaffold", 2: "Fragment"}
EDGE_LABELS = {
  1: "compound_scaffold",
  2: "compound_fragment",
  3: "compound_similarity",
  4: "scaffold_hierarchy",
  5: "fragment_shared",
}


# ------------------------------------------------------------------ configs


@dataclass
class FingerprintConfig:
  """Morgan fingerprint settings.

  ``radii`` may hold several radii; bits of all radii are OR-ed into one
  packed vector, which keeps one fingerprint per compound while still
  covering multiple resolutions (radius 2 and 3 by default).
  """

  radii: tuple = (2, 3)
  n_bits: int = 2048
  use_chirality: bool = False
  use_features: bool = False

  def __post_init__(self) -> None:
    if isinstance(self.radii, int):
      self.radii = (self.radii,)
    self.radii = tuple(int(r) for r in self.radii)
    if not self.radii:
      raise ValueError("radii must not be empty")
    if min(self.radii) < 0:
      raise ValueError("radii must be non-negative")
    if self.n_bits <= 0 or self.n_bits % 64 != 0:
      raise ValueError("n_bits must be a positive multiple of 64")

  @property
  def n_words(self) -> int:
    return self.n_bits // 64


@dataclass
class ScaffoldConfig:
  """Scaffold network settings (Bemis-Murcko is only the entry point)."""

  max_level: int = 6  # max ring count expanded into a hierarchy
  include_generic: bool = False  # element/bond-flattened scaffolds
  flatten_chirality: bool = True
  keep_only_first_fragment: bool = True
  strip_attachments: bool = True  # merge '*'-decorated variants into ring scaffolds
  max_nodes_per_molecule: int = 64


@dataclass
class FragmentConfig:
  """Fragmentation settings, in priority order."""

  use_brics: bool = True
  use_rotatable_bonds: bool = True
  use_linkers: bool = True
  min_fragment_atoms: int = 3
  max_fragments_per_molecule: int = 32
  max_rotatable_cuts: int = 8
  shared_links_per_fragment: int = 4
  reversible: bool = True  # keep attachment points so fragments rebuild into molecules


@dataclass
class SimilarityConfig:
  """Sparse similarity graph settings (never all-vs-all above a size cap)."""

  threshold: float = 0.55
  k: int = 8  # neighbours kept per compound
  n_permutations: int = 128
  n_bands: int = 32
  bucket_cap: int = 64
  max_candidate_pairs: int = 20_000_000
  mutual_only: bool = False
  seed: int = 0xC0FFEE
  metric: str = "tanimoto"
  exact_below: int = 2000  # brute-force (blocked) below this many compounds

  def __post_init__(self) -> None:
    if not 0.0 <= self.threshold <= 1.0:
      raise ValueError("threshold must be within [0, 1]")
    if self.n_permutations % self.n_bands != 0:
      raise ValueError("n_permutations must be divisible by n_bands")


@dataclass
class LandscapeConfig:
  """Top-level build settings."""

  smiles_column: str = "smiles"
  id_column: str | None = None
  delimiter: str | None = None
  chunk_size: int = 20_000
  workers: int = 1
  limit: int | None = None
  fingerprints: FingerprintConfig = field(default_factory=FingerprintConfig)
  scaffolds: ScaffoldConfig = field(default_factory=ScaffoldConfig)
  fragments: FragmentConfig = field(default_factory=FragmentConfig)
  similarity: SimilarityConfig = field(default_factory=SimilarityConfig)

  def __post_init__(self) -> None:
    if isinstance(self.fingerprints, dict):
      self.fingerprints = FingerprintConfig(**self.fingerprints)
    if isinstance(self.scaffolds, dict):
      self.scaffolds = ScaffoldConfig(**self.scaffolds)
    if isinstance(self.fragments, dict):
      self.fragments = FragmentConfig(**self.fragments)
    if isinstance(self.similarity, dict):
      self.similarity = SimilarityConfig(**self.similarity)
    if self.chunk_size <= 0:
      raise ValueError("chunk_size must be positive")

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict) -> LandscapeConfig:
    known = set(cls.__dataclass_fields__.keys())
    return cls(**{k: v for k, v in data.items() if k in known and v is not None})


# ------------------------------------------------------------------ readers


class RecordChunk:
  """A batch of ``(compound_id, smiles)`` records."""

  def __init__(self, compound_ids, smiles, index_offset=0):
    self.compound_ids = list(compound_ids)
    self.smiles = list(smiles)
    self.index_offset = int(index_offset)

  @property
  def size(self) -> int:
    return len(self.smiles)

  def __len__(self) -> int:
    return self.size


_ID_CANDIDATES = ("compound_id", "id", "name", "molecule_id", "mol_id", "title", "idnumber")
_ID_TEMPLATE = "mol-{:07d}"
_CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
_SMI_SUFFIXES = {".smi", ".ism", ".smiles"}
_SDF_SUFFIXES = {".sdf", ".sd", ".mol"}
_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zst"}


def _looks_like_smiles(token: str) -> bool:
  token = token.strip()
  if not token:
    return False
  return any(ch in token for ch in "cCONS[]()=#") and " " not in token


def _open_text(path: Path) -> io.TextIOBase:
  suffix = path.suffix.lower()
  if suffix == ".gz":
    return gzip.open(path, "rt", encoding="utf-8", newline="")
  if suffix == ".bz2":
    return bz2.open(path, "rt", encoding="utf-8", newline="")
  if suffix == ".xz":
    return lzma.open(path, "rt", encoding="utf-8", newline="")
  if suffix == ".zst":
    raise ValueError("zstandard-compressed inputs require the optional zstandard package")
  return open(path, encoding="utf-8", newline="")


def _open_binary(path: Path):
  """Open an SDF as binary, transparently handling supported compression."""
  suffix = path.suffix.lower()
  if suffix == ".gz":
    return gzip.open(path, "rb")
  if suffix == ".bz2":
    return bz2.open(path, "rb")
  if suffix == ".xz":
    return lzma.open(path, "rb")
  if suffix == ".zst":
    raise ValueError("zstandard-compressed inputs require the optional zstandard package")
  return open(path, "rb")


def _sniff_delimiter(header: str) -> str:
  counts = {d: header.count(d) for d in (",", "\t", ";", "|")}
  best = max(counts, key=lambda d: counts[d])
  return best if counts[best] > 0 else ","


def _resolve_column(fieldnames, wanted: str):
  for name in fieldnames:
    if name and name.lower() == wanted.lower():
      return name
  return None


def _pick_id_column(fieldnames, smiles_column: str):
  lowered = {name.lower(): name for name in fieldnames if name}
  for candidate in _ID_CANDIDATES:
    if candidate in lowered and lowered[candidate].lower() != smiles_column.lower():
      return lowered[candidate]
  return None


def _chunks_from_pairs(pairs, chunk_size: int, limit):
  ids, smis, offset = [], [], 0
  for emitted, (cid, smi) in enumerate(pairs, start=1):
    ids.append(cid)
    smis.append(smi)
    if len(smis) >= chunk_size:
      yield RecordChunk(ids, smis, offset)
      offset += len(smis)
      ids, smis = [], []
    if limit is not None and emitted >= limit:
      break
  if smis:
    yield RecordChunk(ids, smis, offset)


def _csv_pairs(path, smiles_column, id_column, delimiter):
  with _open_text(path) as fh:
    first = fh.readline()
    if not first.strip():
      return
    sep = delimiter or _sniff_delimiter(first)
    header = next(csv.reader([first], delimiter=sep))
    reader = csv.reader(fh, delimiter=sep)

    smi_name = _resolve_column(header, smiles_column)
    if smi_name is None:
      raise ValueError(f"column {smiles_column!r} not found in {path.name}; columns are {header!r}")
    smi_idx = header.index(smi_name)

    resolved_id = id_column and _resolve_column(header, id_column)
    if id_column and resolved_id is None:
      raise ValueError(f"id column {id_column!r} not found in {path.name}")
    if resolved_id is None:
      resolved_id = _pick_id_column(header, smi_name)
    id_idx = header.index(resolved_id) if resolved_id else -1

    row_index = 0
    for row in reader:
      if not row or len(row) <= smi_idx:
        continue
      smi = row[smi_idx].strip()
      if not smi:
        continue
      if id_idx >= 0 and id_idx < len(row) and row[id_idx].strip():
        cid = row[id_idx].strip()
      else:
        cid = _ID_TEMPLATE.format(row_index)
      row_index += 1
      yield cid, smi


def _smi_pairs(path, id_column):
  row_index = 0
  with _open_text(path) as fh:
    for raw in fh:
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split()
      smi = parts[0]
      if row_index == 0 and not _looks_like_smiles(smi):
        continue  # header line such as "smiles id"
      cid = parts[1] if len(parts) > 1 else _ID_TEMPLATE.format(row_index)
      row_index += 1
      yield cid, smi


def _sdf_pairs(path, id_column):
  if path.suffix.lower() in _COMPRESSED_SUFFIXES:
    with _open_binary(path) as stream:
      yield from _sdf_pairs_from_supplier(Chem.ForwardSDMolSupplier(stream), id_column)
  else:
    yield from _sdf_pairs_from_supplier(Chem.ForwardSDMolSupplier(str(path)), id_column)


def _sdf_pairs_from_supplier(supplier, id_column):
  row_index = 0
  for mol in supplier:
    if mol is None:
      continue
    smi = Chem.MolToSmiles(mol)
    if not smi:
      continue
    cid = ""
    if id_column and mol.HasProp(id_column):
      cid = mol.GetProp(id_column).strip()
    if not cid and mol.HasProp("_Name"):
      cid = mol.GetProp("_Name").strip()
    if not cid:
      cid = _ID_TEMPLATE.format(row_index)
    row_index += 1
    yield cid, smi


def stream_chunks(
  path: str | Path,
  smiles_column: str = "smiles",
  id_column: str | None = None,
  chunk_size: int = 20_000,
  limit: int | None = None,
  delimiter: str | None = None,
) -> Iterator[RecordChunk]:
  """Stream a molecular library in bounded chunks.

  Args:
    path: CSV, TSV, SMI, or SDF input path, optionally compressed.
    smiles_column: Name of the SMILES column for delimited inputs.
    id_column: Optional compound identifier column.
    chunk_size: Maximum number of records yielded per chunk.
    limit: Optional maximum number of records to read.
    delimiter: Optional delimiter override for delimited inputs.

  Yields:
    :class:`RecordChunk` instances containing compound IDs and SMILES.

  Raises:
    FileNotFoundError: If ``path`` does not exist.
    ValueError: If the input format or chunk size is invalid.
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(str(p))
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")
  suffixes = [s.lower() for s in p.suffixes]
  if suffixes and suffixes[-1] in _COMPRESSED_SUFFIXES:
    suffixes = suffixes[:-1]
  suffix = suffixes[-1] if suffixes else ""
  if suffix in _SDF_SUFFIXES:
    pairs = _sdf_pairs(p, id_column)
  elif suffix in _SMI_SUFFIXES:
    pairs = _smi_pairs(p, id_column)
  elif suffix in _CSV_SUFFIXES:
    pairs = _csv_pairs(p, smiles_column, id_column, delimiter)
  else:
    raise ValueError(f"unsupported input format for {p.name!r}")
  yield from _chunks_from_pairs(pairs, chunk_size, limit)


# ------------------------------------------------------------- fingerprints


def pack_bits(dense: np.ndarray) -> np.ndarray:
  """Pack a binary matrix into uint64 words.

  Args:
    dense: ``(n, n_bits)`` binary matrix whose bit width is divisible by 64.

  Returns:
    Packed ``(n, n_bits // 64)`` uint64 words.
  """
  dense = np.ascontiguousarray(dense, dtype=np.uint8)
  n, n_bits = dense.shape
  if n_bits % 64 != 0:
    raise ValueError("n_bits must be a multiple of 64")
  view = dense.reshape(n, n_bits // 64, 64).astype(np.uint64)
  weights = (1 << np.arange(64, dtype=np.uint64)).astype(np.uint64)
  return (view * weights).sum(axis=2, dtype=np.uint64)


def unpack_bits(packed: np.ndarray, n_bits: int) -> np.ndarray:
  """Unpack uint64 words into a binary matrix.

  Args:
    packed: ``(n, n_words)`` packed uint64 words.
    n_bits: Number of output bits per row.

  Returns:
    A ``(n, n_bits)`` uint8 matrix.
  """
  out = (packed[:, :, None] >> np.arange(64, dtype=np.uint64)[None, None, :]) & np.uint64(1)
  return out.astype(np.uint8).reshape(packed.shape[0], n_bits)


def popcount_words(words: np.ndarray) -> np.ndarray:
  """Count set bits in uint64 values.

  Args:
    words: NumPy array of values to count.

  Returns:
    An array with the population count of each input value.
  """
  words = np.ascontiguousarray(words, dtype=np.uint64)
  bc = getattr(np, "bitwise_count", None)
  if bc is not None:  # NumPy >= 2.1
    return bc(words).astype(np.int64)
  # 16-bit lookup table fallback (older NumPy)
  lut = np.zeros(1 << 16, dtype=np.int64)
  for i in range(1, 1 << 16):
    lut[i] = lut[i >> 1] + (i & 1)
  w = words.reshape(-1)
  counts = (
    lut[(w & 0xFFFF).astype(np.uint16)]
    + lut[((w >> 16) & 0xFFFF).astype(np.uint16)]
    + lut[((w >> 32) & 0xFFFF).astype(np.uint16)]
    + lut[((w >> 48) & 0xFFFF).astype(np.uint16)]
  )
  return counts.reshape(words.shape)


def popcount_rows(packed: np.ndarray) -> np.ndarray:
  """Count set bits row-wise in a packed fingerprint matrix.

  Args:
    packed: ``(n, n_words)`` uint64 fingerprint matrix.

  Returns:
    One population count per row.
  """
  packed = np.ascontiguousarray(packed, dtype=np.uint64)
  if packed.size == 0:
    return np.zeros(packed.shape[0], dtype=np.int64)
  return popcount_words(packed).sum(axis=1)


class FingerprintBlock:
  """Packed Morgan fingerprints for the whole library."""

  def __init__(self, packed: np.ndarray, popcounts: np.ndarray, n_bits: int):
    self.packed = np.ascontiguousarray(packed, dtype=np.uint64)
    self.popcounts = np.ascontiguousarray(popcounts, dtype=np.int64)
    self.n_bits = int(n_bits)

  @property
  def n_mols(self) -> int:
    return self.packed.shape[0]

  @property
  def n_words(self) -> int:
    return self.packed.shape[1]

  def dense(self, start: int = 0, stop: int | None = None) -> np.ndarray:
    stop = self.n_mols if stop is None else stop
    return unpack_bits(self.packed[start:stop], self.n_bits)

  def onbits_csr(self) -> tuple:
    """CSR of set bit positions: ``(offsets, indices)``."""
    if self.n_mols == 0:
      return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64)
    dense = unpack_bits(self.packed, self.n_bits)
    rows, cols = np.nonzero(dense)
    counts = np.bincount(rows, minlength=self.n_mols)
    offsets = np.zeros(self.n_mols + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, cols.astype(np.int64)


def morgan_packed(smiles: Sequence[str], cfg: FingerprintConfig) -> FingerprintBlock:
  """Generate packed Morgan fingerprints.

  Args:
    smiles: Molecule SMILES strings.
    cfg: Fingerprint settings.

  Returns:
    A packed fingerprint block. Unparsable SMILES produce all-zero rows.
  """
  n = len(smiles)
  dense = np.zeros((n, cfg.n_bits), dtype=np.uint8)
  if n:
    kwargs = {"fpSize": cfg.n_bits, "includeChirality": cfg.use_chirality}
    gens = []
    for radius in sorted(set(cfg.radii)):
      if cfg.use_features:
        inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        gens.append(rdFingerprintGenerator.GetMorganGenerator(radius=radius, atomInvariantsGenerator=inv, **kwargs))
      else:
        gens.append(rdFingerprintGenerator.GetMorganGenerator(radius=radius, **kwargs))
    for i, smi in enumerate(smiles):
      mol = Chem.MolFromSmiles(smi) if smi else None
      if mol is None:
        continue
      acc = dense[i]
      for gen in gens:
        np.bitwise_or(acc, gen.GetFingerprintAsNumPy(mol).astype(np.uint8), out=acc)
  packed = pack_bits(dense)
  return FingerprintBlock(packed, popcount_rows(packed), cfg.n_bits)


# ----------------------------------------------------------------- scaffolds


def _as_mol(smiles_or_mol):
  if isinstance(smiles_or_mol, Chem.Mol):
    return smiles_or_mol
  if not smiles_or_mol:
    return None
  return Chem.MolFromSmiles(smiles_or_mol)


def murcko_smiles(smiles_or_mol: str | Chem.Mol) -> str:
  """Return the canonical Bemis-Murcko scaffold SMILES.

  Args:
    smiles_or_mol: SMILES string or RDKit molecule.

  Returns:
    The scaffold SMILES, or ``""`` for acyclic or invalid input.
  """
  mol = _as_mol(smiles_or_mol)
  if mol is None:
    return ""
  try:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
  except (TypeError, ValueError, RuntimeError):
    return ""
  if scaffold is None or scaffold.GetNumAtoms() == 0:
    return ""
  return Chem.MolToSmiles(scaffold)


def generic_scaffold_smiles(smiles_or_mol: str | Chem.Mol) -> str:
  """Return the generic Bemis-Murcko graph framework.

  Args:
    smiles_or_mol: SMILES string or RDKit molecule.

  Returns:
    A scaffold with generic atoms and single bonds, or ``""`` if invalid.
  """
  mol = _as_mol(smiles_or_mol)
  if mol is None:
    return ""
  try:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
      return ""
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    Chem.SanitizeMol(generic)
  except (TypeError, ValueError, RuntimeError):
    return ""
  return Chem.MolToSmiles(generic)


def _strip_dummies(smiles: str) -> str:
  """Remove attachment-point dummy atoms, returning a plain scaffold SMILES."""
  if "*" not in smiles:
    return smiles
  mol = Chem.MolFromSmiles(smiles, sanitize=False)
  if mol is None:
    return ""
  editable = Chem.RWMol(mol)
  for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
    editable.RemoveAtom(idx)
  stripped = editable.GetMol()
  if stripped.GetNumAtoms() == 0:
    return ""
  try:
    Chem.SanitizeMol(stripped)
  except (TypeError, ValueError, RuntimeError):
    return ""
  return Chem.MolToSmiles(stripped)


_RING_CACHE: dict = {}


def _ring_count(smiles: str) -> int:
  cached = _RING_CACHE.get(smiles)
  if cached is None:
    mol = Chem.MolFromSmiles(smiles)
    cached = int(rdMolDescriptors.CalcNumRings(mol)) if mol is not None else 0
    if len(_RING_CACHE) < 1_000_000:
      _RING_CACHE[smiles] = cached
  return cached


_RELATION_BY_NAME = {
  "Fragment": 1,
  "Generic": 2,
  "GenericBond": 2,
  "RemoveAttachment": 3,
  "Initialize": 4,
}


def _network_params(cfg: ScaffoldConfig):
  params = rdScaffoldNetwork.ScaffoldNetworkParams()
  params.includeGenericScaffolds = False
  params.includeGenericBondScaffolds = False
  params.includeScaffoldsWithoutAttachments = True
  params.includeScaffoldsWithAttachments = not cfg.strip_attachments
  params.keepOnlyFirstFragment = bool(cfg.keep_only_first_fragment)
  params.pruneBeforeFragmenting = True
  params.flattenChirality = bool(cfg.flatten_chirality)
  params.flattenIsotopes = True
  return params


class ScaffoldNetworkResult:
  """Scaffold nodes, per-compound Murcko links and hierarchy edges."""

  def __init__(
    self,
    scaffolds=None,
    levels=None,
    compound_scaffold=None,
    hierarchy_parent=None,
    hierarchy_child=None,
    hierarchy_relation=None,
    murcko=None,
  ):
    self.scaffolds = list(scaffolds) if scaffolds is not None else []
    self.levels = np.asarray(levels if levels is not None else [], dtype=np.int32)
    self.compound_scaffold = np.asarray(compound_scaffold if compound_scaffold is not None else [], dtype=np.int64)
    self.hierarchy_parent = np.asarray(hierarchy_parent if hierarchy_parent is not None else [], dtype=np.int64)
    self.hierarchy_child = np.asarray(hierarchy_child if hierarchy_child is not None else [], dtype=np.int64)
    self.hierarchy_relation = np.asarray(hierarchy_relation if hierarchy_relation is not None else [], dtype=np.int8)
    self.murcko = list(murcko) if murcko is not None else []

  @property
  def n_scaffolds(self) -> int:
    return len(self.scaffolds)

  @property
  def n_hierarchy_edges(self) -> int:
    return int(self.hierarchy_parent.size)

  @classmethod
  def merge(cls, results):
    """Merge chunk results, re-indexing scaffolds into one global list."""
    smiles, index, levels, compound_scaffold, murcko, edges = [], {}, [], [], [], {}
    for res in results:
      local_to_global = []
      for smi, level in zip(res.scaffolds, res.levels.tolist()):
        gid = index.get(smi)
        if gid is None:
          gid = len(smiles)
          index[smi] = gid
          smiles.append(smi)
          levels.append(int(level))
        local_to_global.append(gid)
      for local in res.compound_scaffold.tolist():
        compound_scaffold.append(local_to_global[local] if local >= 0 else -1)
      murcko.extend(res.murcko)
      for p, c, rel in zip(
        res.hierarchy_parent.tolist(),
        res.hierarchy_child.tolist(),
        res.hierarchy_relation.tolist(),
      ):
        key = (local_to_global[p], local_to_global[c])
        if key[0] != key[1]:
          edges.setdefault(key, int(rel))
    return cls(
      scaffolds=smiles,
      levels=np.asarray(levels, dtype=np.int32),
      compound_scaffold=np.asarray(compound_scaffold, dtype=np.int64),
      hierarchy_parent=np.fromiter((k[0] for k in edges), dtype=np.int64, count=len(edges)),
      hierarchy_child=np.fromiter((k[1] for k in edges), dtype=np.int64, count=len(edges)),
      hierarchy_relation=np.fromiter(edges.values(), dtype=np.int8, count=len(edges)),
      murcko=murcko,
    )


def _expand_scaffold(murcko_smi, cfg, params):
  """Expand one Murcko scaffold into (nodes, parent->child edges)."""
  mol = Chem.MolFromSmiles(murcko_smi)
  if mol is None:
    return [], []
  n_rings = rdMolDescriptors.CalcNumRings(mol)
  if n_rings > cfg.max_level:
    return [murcko_smi], []  # too many rings: annotation only, no expansion
  try:
    net = rdScaffoldNetwork.CreateScaffoldNetwork([mol], params)
  except (TypeError, ValueError, RuntimeError):
    return [murcko_smi], []

  raw_nodes = list(net.nodes)
  keys = []
  for smi in raw_nodes:
    key = _strip_dummies(smi) if cfg.strip_attachments else smi
    keys.append(key if key and _ring_count(key) >= 1 else "")

  nodes = {murcko_smi}
  edges = []
  for edge in net.edges:
    parent = keys[edge.endIdx]  # end node is the more general one
    child = keys[edge.beginIdx]
    if not parent or not child or parent == child:
      continue
    relation = int(_RELATION_BY_NAME.get(str(edge.type).split(".")[-1], 0))
    nodes.add(parent)
    nodes.add(child)
    edges.append((parent, child, relation))
  for key in keys:
    if key:
      nodes.add(key)

  if cfg.include_generic:
    for smi in sorted(nodes):
      generic = generic_scaffold_smiles(smi)
      if generic and generic != smi and _ring_count(generic) >= 1:
        nodes.add(generic)
        edges.append((generic, smi, 2))

  if len(nodes) > cfg.max_nodes_per_molecule:
    ranked = sorted(nodes, key=lambda s: (-_ring_count(s), s))
    keep = {murcko_smi, *ranked[: cfg.max_nodes_per_molecule]}
    nodes = keep
    edges = [e for e in edges if e[0] in keep and e[1] in keep]
  return sorted(nodes), edges


def scaffold_network(smiles: Sequence[str], cfg: ScaffoldConfig | None = None) -> ScaffoldNetworkResult:
  """Build the scaffold network for a list of compounds.

  Unique Murcko scaffolds are expanded once, so cost scales with the number
  of *distinct* scaffolds rather than the number of compounds.

  Args:
    smiles: Compound SMILES strings.
    cfg: Optional scaffold network settings.

  Returns:
    Scaffold nodes, compound links, and hierarchy edges.
  """
  cfg = cfg or ScaffoldConfig()
  params = _network_params(cfg)

  murckos = [murcko_smiles(s) for s in smiles]
  unique = sorted({m for m in murckos if m})

  scaffold_index: dict = {}
  scaffold_list: list = []
  edge_map: dict = {}

  def node_id(smi):
    gid = scaffold_index.get(smi)
    if gid is None:
      gid = len(scaffold_list)
      scaffold_index[smi] = gid
      scaffold_list.append(smi)
    return gid

  for murcko_smi in unique:
    nodes, edges = _expand_scaffold(murcko_smi, cfg, params)
    for smi in nodes:
      node_id(smi)
    for parent, child, relation in edges:
      key = (node_id(parent), node_id(child))
      if key[0] != key[1]:
        edge_map.setdefault(key, relation)

  compound_scaffold = np.asarray([scaffold_index.get(m, -1) if m else -1 for m in murckos], dtype=np.int64)
  levels = np.asarray([_ring_count(s) for s in scaffold_list], dtype=np.int32)
  parents = np.fromiter((k[0] for k in edge_map), dtype=np.int64, count=len(edge_map))
  children = np.fromiter((k[1] for k in edge_map), dtype=np.int64, count=len(edge_map))
  relations = np.fromiter(edge_map.values(), dtype=np.int8, count=len(edge_map))

  # keep hierarchy oriented general -> specific
  swap = levels[parents] > levels[children]
  if np.any(swap):
    parents, children = (
      np.where(swap, children, parents),
      np.where(swap, parents, children),
    )

  return ScaffoldNetworkResult(
    scaffolds=scaffold_list,
    levels=levels,
    compound_scaffold=compound_scaffold,
    hierarchy_parent=parents,
    hierarchy_child=children,
    hierarchy_relation=relations,
    murcko=murckos,
  )


# ---------------------------------------------------------------- fragments


def _clean_fragment(smiles: str, min_atoms: int) -> str:
  """Canonicalise a fragment: drop dummy atoms, enforce a size floor."""
  if not smiles:
    return ""
  mol = Chem.MolFromSmiles(smiles, sanitize=False)
  if mol is None:
    return ""
  if mol.HasSubstructMatch(Chem.MolFromSmarts("[#0]")):
    editable = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
      editable.RemoveAtom(idx)
    mol = editable.GetMol()
  if mol.GetNumAtoms() == 0:
    return ""
  try:
    Chem.SanitizeMol(mol)
  except (TypeError, ValueError, RuntimeError):
    return ""
  if mol.GetNumHeavyAtoms() < min_atoms:
    return ""
  return Chem.MolToSmiles(mol)


def _brics_fragments(mol, cfg):
  try:
    pieces = BRICS.BRICSDecompose(mol, returnMols=False, keepNonLeafNodes=False)
  except (TypeError, ValueError, RuntimeError):
    return []
  out = []
  for smi in pieces:
    cleaned = _clean_fragment(smi, cfg.min_fragment_atoms)
    if cleaned:
      out.append(cleaned)
  return out


def _fragment_on_bonds(mol, bond_indices):
  if not bond_indices:
    return []
  try:
    exploded = Chem.FragmentOnBonds(mol, list(bond_indices), addDummies=False)
    pieces = Chem.GetMolFrags(exploded, asMols=True, sanitizeFrags=True)
  except (TypeError, ValueError, RuntimeError):
    return []
  return [Chem.MolToSmiles(p) for p in pieces]


def _rotatable_fragments(mol, cfg):
  matches = mol.GetSubstructMatches(RotatableBondSmarts)
  bonds = []
  for a1, a2 in matches:
    bond = mol.GetBondBetweenAtoms(a1, a2)
    if bond is not None and not bond.IsInRing():
      bonds.append(bond.GetIdx())
  bonds = sorted(set(bonds))[: cfg.max_rotatable_cuts]
  out = []
  for smi in _fragment_on_bonds(mol, bonds):
    cleaned = _clean_fragment(smi, cfg.min_fragment_atoms)
    if cleaned:
      out.append(cleaned)
  return out


def _linker_fragments(mol, cfg):
  """Ring-free bridges between ring systems, taken from the Murcko scaffold."""
  scaffold_smi = murcko_smiles(mol)
  if not scaffold_smi:
    return []
  scaffold = Chem.MolFromSmiles(scaffold_smi)
  if scaffold is None:
    return []
  bonds = [bond.GetIdx() for bond in scaffold.GetBonds() if not bond.IsInRing() and (bond.GetBeginAtom().IsInRing() != bond.GetEndAtom().IsInRing())]
  out = []
  for smi in _fragment_on_bonds(scaffold, bonds):
    piece = Chem.MolFromSmiles(smi)
    if piece is None or piece.GetRingInfo().NumRings() > 0:
      continue  # ring systems are covered by the scaffold layer
    cleaned = _clean_fragment(smi, cfg.min_fragment_atoms)
    if cleaned:
      out.append(cleaned)
  return out


def fragment_molecule(smiles: str | Chem.Mol, cfg: FragmentConfig | None = None) -> list[tuple[str, int]]:
  """Fragment one molecule into ``(fragment_smiles, method)`` pairs.

  Args:
    smiles: SMILES string or RDKit molecule to fragment.
    cfg: Optional fragmentation settings.

  Returns:
    Fragment SMILES paired with their :class:`FragmentMethod` value. Invalid
    molecules return an empty list.
  """
  cfg = cfg or FragmentConfig()
  mol = smiles if isinstance(smiles, Chem.Mol) else Chem.MolFromSmiles(smiles)
  if mol is None:
    return []

  seen: dict = {}
  stages = (
    (cfg.use_brics, FragmentMethod.BRICS, _brics_fragments),
    (cfg.use_rotatable_bonds, FragmentMethod.ROTATABLE_BOND, _rotatable_fragments),
    (cfg.use_linkers, FragmentMethod.LINKER, _linker_fragments),
  )
  for enabled, method, fn in stages:
    if not enabled or len(seen) >= cfg.max_fragments_per_molecule:
      continue
    for smi in fn(mol, cfg):
      if smi in seen:
        continue
      seen[smi] = int(method)
      if len(seen) >= cfg.max_fragments_per_molecule:
        break
  return list(seen.items())


class FragmentResult:
  """Fragment nodes plus compound->fragment edges for a chunk."""

  def __init__(
    self,
    fragments=None,
    methods=None,
    frequencies=None,
    ring_systems=None,
    compound_fragment_src=None,
    compound_fragment_dst=None,
    compound_fragment_method=None,
    n_compounds=0,
  ):
    self.fragments = list(fragments) if fragments is not None else []
    self.methods = np.asarray(methods if methods is not None else [], dtype=np.int8)
    self.frequencies = np.asarray(frequencies if frequencies is not None else [], dtype=np.int64)
    self.ring_systems = list(ring_systems) if ring_systems is not None else []
    self.compound_fragment_src = np.asarray(compound_fragment_src if compound_fragment_src is not None else [], dtype=np.int64)
    self.compound_fragment_dst = np.asarray(compound_fragment_dst if compound_fragment_dst is not None else [], dtype=np.int64)
    self.compound_fragment_method = np.asarray(compound_fragment_method if compound_fragment_method is not None else [], dtype=np.int8)
    self.n_compounds = int(n_compounds)

  @property
  def n_fragments(self) -> int:
    return len(self.fragments)

  @classmethod
  def merge(cls, results):
    fragments, index, methods, ring_systems, freq = [], {}, [], [], []
    src, dst, meth = [], [], []
    offset = 0
    for res in results:
      mapping = np.empty(res.n_fragments, dtype=np.int64)
      for local, smi in enumerate(res.fragments):
        gid = index.get(smi)
        if gid is None:
          gid = len(fragments)
          index[smi] = gid
          fragments.append(smi)
          methods.append(int(res.methods[local]))
          ring_systems.append(res.ring_systems[local])
          freq.append(0)
        mapping[local] = gid
        freq[gid] += int(res.frequencies[local])
      src.append(res.compound_fragment_src + offset)
      dst.append(mapping[res.compound_fragment_dst] if res.n_fragments else res.compound_fragment_dst)
      meth.append(res.compound_fragment_method)
      offset += res.n_compounds
    return cls(
      fragments=fragments,
      methods=np.asarray(methods, dtype=np.int8),
      frequencies=np.asarray(freq, dtype=np.int64),
      ring_systems=ring_systems,
      compound_fragment_src=np.concatenate(src) if src else np.empty(0, dtype=np.int64),
      compound_fragment_dst=np.concatenate(dst) if dst else np.empty(0, dtype=np.int64),
      compound_fragment_method=np.concatenate(meth) if meth else np.empty(0, dtype=np.int8),
      n_compounds=offset,
    )


def fragment_library(smiles: Sequence[str], cfg: FragmentConfig | None = None) -> FragmentResult:
  """Fragment a chunk of compounds into deduplicated fragment nodes.

  Args:
    smiles: Compound SMILES strings.
    cfg: Optional fragmentation settings.

  Returns:
    Fragment nodes and compound-to-fragment edges for the input chunk.
  """
  cfg = cfg or FragmentConfig()
  index, fragments, methods, freq, src, dst, meth = {}, [], [], [], [], [], []
  for i, smi in enumerate(smiles):
    for frag_smi, method in fragment_molecule(smi, cfg):
      gid = index.get(frag_smi)
      if gid is None:
        gid = len(fragments)
        index[frag_smi] = gid
        fragments.append(frag_smi)
        methods.append(method)
        freq.append(0)
      freq[gid] += 1
      src.append(i)
      dst.append(gid)
      meth.append(method)

  ring_systems = [murcko_smiles(smi) for smi in fragments]
  return FragmentResult(
    fragments=fragments,
    methods=np.asarray(methods, dtype=np.int8),
    frequencies=np.asarray(freq, dtype=np.int64),
    ring_systems=ring_systems,
    compound_fragment_src=np.asarray(src, dtype=np.int64),
    compound_fragment_dst=np.asarray(dst, dtype=np.int64),
    compound_fragment_method=np.asarray(meth, dtype=np.int8),
    n_compounds=len(smiles),
  )


def shared_fragment_edges(
  ring_systems: Sequence[str],
  frequencies: Sequence[int] | np.ndarray,
  links_per_fragment: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
  """Fragment-fragment edges for fragments sharing a ring system.

  Each fragment links to the ``links_per_fragment`` most frequent other
  fragments carrying the same ring system, keeping the edge count linear.

  Args:
    ring_systems: Ring-system SMILES parallel to the fragment list.
    frequencies: Fragment frequencies parallel to ``ring_systems``.
    links_per_fragment: Maximum number of neighbours per fragment.

  Returns:
    Two arrays containing the source and destination fragment indices.
  """
  groups: dict = {}
  for idx, key in enumerate(ring_systems):
    if key:
      groups.setdefault(key, []).append(idx)

  freq = np.asarray(frequencies, dtype=np.int64)
  pairs = set()
  for members in groups.values():
    if len(members) < 2:
      continue
    ordered = sorted(members, key=lambda i: (-int(freq[i]), i))
    for i in members:
      linked = 0
      for j in ordered:
        if j == i:
          continue
        pairs.add((min(i, j), max(i, j)))
        linked += 1
        if linked >= links_per_fragment:
          break

  if not pairs:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
  arr = np.asarray(sorted(pairs), dtype=np.int64)
  return arr[:, 0], arr[:, 1]


# ------------------------------------------------------- reversible fragments


class ReverseFragmentRecord:
  """A compound's fragments kept with their attachment points so the molecule
  can be rebuilt.

  Cutting a set of bonds produces pieces whose cut ends are capped with dummy
  atoms carrying a unique *cut-id* isotope.  ``pieces`` stores those reactive
  SMILES; ``cut_orders[k]`` is the bond order of the cut that produced the
  ``k``-th isotope pair.  Reassembly matches the two dummies of each cut id
  and reconnects their neighbours with the recorded bond order.
  """

  def __init__(self, pieces=None, cut_orders=None, methods=None):
    self.pieces = list(pieces) if pieces is not None else []
    self.cut_orders = list(cut_orders) if cut_orders is not None else []
    self.methods = list(methods) if methods is not None else []

  @property
  def n_cuts(self) -> int:
    return len(self.cut_orders)

  def to_dict(self) -> dict:
    return {"pieces": self.pieces, "cut_orders": list(self.cut_orders), "methods": list(self.methods)}

  @classmethod
  def from_dict(cls, data: dict) -> ReverseFragmentRecord:
    return cls(data.get("pieces", []), data.get("cut_orders", []), data.get("methods", []))


def _brics_cut_bonds(mol, cfg):
  """Bond indices BRICS would cut (the disconnection rules)."""
  out = []
  try:
    for (a1, a2), _ in BRICS.FindBRICSBonds(mol):
      bond = mol.GetBondBetweenAtoms(a1, a2)
      if bond is not None:
        out.append(bond.GetIdx())
  except (TypeError, ValueError, RuntimeError):
    return out
  return out


def _rotatable_cut_bonds(mol, cfg):
  """Bond indices of rotatable (non-ring) single bonds, capped."""
  out = []
  for a1, a2 in mol.GetSubstructMatches(RotatableBondSmarts):
    bond = mol.GetBondBetweenAtoms(a1, a2)
    if bond is not None and not bond.IsInRing():
      out.append(bond.GetIdx())
  return sorted(set(out))[: cfg.max_rotatable_cuts]


def _linker_cut_bonds(mol, cfg):
  """Ring-free bonds bridging two ring systems, taken from the Murcko scaffold."""
  scaffold_smi = murcko_smiles(mol)
  if not scaffold_smi:
    return []
  scaffold = Chem.MolFromSmiles(scaffold_smi)
  if scaffold is None:
    return []
  return [bond.GetIdx() for bond in scaffold.GetBonds() if not bond.IsInRing() and (bond.GetBeginAtom().IsInRing() != bond.GetEndAtom().IsInRing())]


def fragment_cut_bonds(mol: Chem.Mol, cfg: FragmentConfig | None = None) -> list[tuple[int, FragmentMethod]]:
  """Bond indices each enabled method would cut, with their FragmentMethod.

  Returns ``[(bond_idx, method), ...]`` with duplicates removed and the
  per-molecule budget respected.

  Args:
    mol: RDKit molecule to inspect.
    cfg: Optional fragmentation settings.

  Returns:
    Unique bond indices paired with their cut method.
  """
  cfg = cfg or FragmentConfig()
  gathered = []
  if cfg.use_brics:
    gathered += [(b, FragmentMethod.BRICS) for b in _brics_cut_bonds(mol, cfg)]
  if cfg.use_rotatable_bonds:
    gathered += [(b, FragmentMethod.ROTATABLE_BOND) for b in _rotatable_cut_bonds(mol, cfg)]
  if cfg.use_linkers:
    gathered += [(b, FragmentMethod.LINKER) for b in _linker_cut_bonds(mol, cfg)]
  seen, out = set(), []
  for b, method in gathered:
    if b in seen:
      continue
    seen.add(b)
    # respect the per-molecule fragment budget (a cut yields ~2 pieces)
    if len(out) >= max(cfg.max_fragments_per_molecule // 2, 1):
      break
    out.append((b, method))
  return out


def apply_reversible_cut(mol: Chem.Mol, bond_ids: Sequence[int]) -> tuple[list[str], list[int]]:
  """Cut ``bond_ids`` and cap both ends with cut-id dummy atoms.

  Returns ``(pieces, cut_orders)`` where ``pieces`` are reactive SMILES in
  which cut ``k`` explains isotope ``k + 1`` and ``cut_orders[k]`` holds the
  original bond order of that cut.
  """
  rw = Chem.RWMol(mol)
  orders = []
  for k, bid in enumerate(bond_ids):
    bond = mol.GetBondWithIdx(bid)
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    orders.append(int(bond.GetBondType()))
    rw.RemoveBond(a1, a2)
    d1 = rw.AddAtom(Chem.Atom(0))
    d2 = rw.AddAtom(Chem.Atom(0))
    rw.GetAtomWithIdx(d1).SetIsotope(k + 1)
    rw.GetAtomWithIdx(d2).SetIsotope(k + 1)
    rw.AddBond(a1, d1, Chem.BondType.SINGLE)
    rw.AddBond(a2, d2, Chem.BondType.SINGLE)
  fm = rw.GetMol()
  frags = Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=False)
  return [Chem.MolToSmiles(f) for f in frags], orders


def reversible_fragment_record(
  smiles: str | Chem.Mol,
  cfg: FragmentConfig | None = None,
) -> ReverseFragmentRecord | None:
  """Fragment a molecule reversibly.

  Args:
    smiles: SMILES string or RDKit molecule to fragment.
    cfg: Optional fragmentation settings.

  Returns:
    A reversible fragment record, or ``None`` when no reversible cut applies.
  """
  cfg = cfg or FragmentConfig()
  if not cfg.reversible:
    return None
  mol = smiles if isinstance(smiles, Chem.Mol) else Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  cuts = fragment_cut_bonds(mol, cfg)
  if not cuts:
    return None
  bond_ids = [b for b, _ in cuts]
  methods = [int(m) for _, m in cuts]
  pieces, orders = apply_reversible_cut(mol, bond_ids)
  return ReverseFragmentRecord(pieces, orders, methods)


def reassemble_fragments(pieces: Sequence[str], cut_orders: Sequence[int]) -> str:
  """Join reactive fragment pieces back into the original molecule.

  Two dummies sharing a cut-id isotope are reconnected with the recorded bond
  order, then all dummies are removed and the result is sanitized.

  Args:
    pieces: Reactive fragment SMILES.
    cut_orders: Original bond-order values, indexed by cut isotope.

  Returns:
    The reassembled canonical SMILES, or ``""`` if reassembly fails.
  """
  if not pieces:
    return ""
  mols = [Chem.MolFromSmiles(s, sanitize=False) for s in pieces]
  combined = mols[0]
  for m in mols[1:]:
    combined = Chem.CombineMols(combined, m)
  rw = Chem.RWMol(combined)
  for k in range(len(cut_orders)):
    iso = k + 1
    real = []
    for a in rw.GetAtoms():
      if a.GetAtomicNum() == 0 and a.GetIsotope() == iso:
        nbrs = list(a.GetNeighbors())
        if nbrs:
          real.append(nbrs[0].GetIdx())
    if len(real) == 2:
      rw.AddBond(real[0], real[1], Chem.BondType.values[int(cut_orders[k])])
  mol = rw.GetMol()
  mol = _remove_dummies(mol, None)
  try:
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)
  except (TypeError, ValueError, RuntimeError):
    return ""


# ------------------------------------------------------------ cores & rgroups


@dataclass
class RGroupDecompositionResult:
  """A library decomposed into a core plus per-position R-groups.

  ``rows`` is a list of per-molecule dicts keyed by R-group label (``'Core'``,
  ``'R1'``, ``'R2'``, ...) mapping to the group SMILES.
  """

  core_smiles: str = ""
  rows: list = field(default_factory=list)
  n_failed: int = 0

  @property
  def positions(self):
    """Sorted integer attachment labels present (excluding the core)."""
    seen = set()
    for row in self.rows:
      for k in row:
        if k != "Core":
          seen.add(_rgroup_number(k))
    return sorted(seen)

  def to_dict(self) -> dict:
    return {
      "core_smiles": self.core_smiles,
      "rows": self.rows,
      "n_failed": self.n_failed,
    }


def _rgroup_number(key: str) -> int:
  """Convert an R-group key like ``'R3'`` to its integer label ``3``."""
  if key.startswith("R") and key[1:].isdigit():
    return int(key[1:])
  raise KeyError(key)


def _dummy_neighbor(mol, label):
  """Return ``(dummy_idx, neighbour_idx)`` for a dummy labelled ``label``
  (matching either the isotope form ``[k*]`` or the map-number form ``[*:k]``)."""
  for a in mol.GetAtoms():
    if a.GetAtomicNum() == 0 and (a.GetIsotope() == label or a.GetAtomMapNum() == label):
      nbrs = list(a.GetNeighbors())
      return a.GetIdx(), (nbrs[0].GetIdx() if nbrs else None)
  return None, None


def decompose_molecules(
  mol_smiles: Sequence[str],
  core_smiles: str,
  *,
  params: Any = None,
) -> RGroupDecompositionResult:
  """Decompose a set of molecules into a core + R-groups (R-group linkage).

  ``core_smiles`` may be labeled (``[*:1]``, ``[*:2]``, ...) or unlabeled; an
  unlabeled core is matched and its attachment points detected automatically.

  Args:
    mol_smiles: Molecules to decompose.
    core_smiles: Labeled or unlabeled core SMILES.
    params: Optional RDKit R-group decomposition parameters.

  Returns:
    Decomposition rows and the matched core SMILES.
  """
  from rdkit.Chem import rdRGroupDecomposition as rgd

  mol_smiles = list(mol_smiles)
  core = Chem.MolFromSmiles(core_smiles)
  if core is None:
    raise ValueError(f"invalid core SMILES: {core_smiles!r}")
  p = params if params is not None else rgd.RGroupDecompositionParameters()
  rd = rgd.RGroupDecomposition([core], p)
  mols = []
  for smi in mol_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
      rd.Add(mol)
      mols.append(mol)
  rd.Process()
  rows = []
  core_smiles_out = ""
  for row in rd.GetRGroupsAsRows():
    entry = {}
    for label in row:
      entry[label] = Chem.MolToSmiles(row[label])
    if "Core" in entry:
      core_smiles_out = entry["Core"]
    rows.append(entry)
  return RGroupDecompositionResult(
    core_smiles=core_smiles_out,
    rows=rows,
    n_failed=len(mol_smiles) - len(rows),
  )


def _delete_labeled_dummies(mol, label: int):
  """Delete every dummy atom carrying ``label`` in either labeled form.

  Supports the isotope form ``[k*]`` and the map-number form ``[*:k]`` used by
  R-group decomposition.  Only dummy atoms (atomic number 0) are matched.
  """
  mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts(f"[{label}*]"))  # isotope [k*]
  mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts(f"[#0:{label}]"))  # map [*:k]
  return mol


def _remove_dummies(mol, labels=None):
  """Remove dummy atoms, optionally only those carrying any of ``labels``.

  ``labels`` is a set of ints matched against isotope or map-number labels; if
  ``None`` every dummy is removed.  Atoms are removed by index in descending
  order so indices of surviving atoms are stable.
  """
  to_remove = []
  for a in mol.GetAtoms():
    if a.GetAtomicNum() == 0 and (labels is None or (a.GetIsotope() in labels or a.GetAtomMapNum() in labels)):
      to_remove.append(a.GetIdx())
  if not to_remove:
    return mol
  rw = Chem.RWMol(mol)
  for idx in sorted(to_remove, reverse=True):
    rw.RemoveAtom(idx)
  return rw.GetMol()


def _attach_one(core_smiles, rgroup_smiles, label: int) -> str:
  """Attach a single R-group to a core at a labeled attachment point.

  Combines the core and R-group, bonds the two real atoms adjacent to the
  matching labeled dummies, then removes every dummy.  Works with both the
  isotope form ``[k*]`` and the map-number form ``[*:k]``.
  """
  core = Chem.MolFromSmiles(core_smiles)
  rg = Chem.MolFromSmiles(rgroup_smiles)
  if core is None or rg is None:
    return ""
  _, c_real = _dummy_neighbor(core, label)
  _, r_real = _dummy_neighbor(rg, label)
  if c_real is None or r_real is None:
    return ""
  combined = Chem.CombineMols(core, rg)
  off = core.GetNumAtoms()
  rw = Chem.RWMol(combined)
  rw.AddBond(c_real, r_real + off, Chem.BondType.SINGLE)
  mol = rw.GetMol()
  # remove only the consumed dummy pair (leave other attachment labels intact)
  mol = _remove_dummies(mol, {label})
  try:
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)
  except (TypeError, ValueError, RuntimeError):
    return ""


def _attach_many(core_smiles, rgroups_by_label):
  """Attach every R-group to a labeled core in one pass.

  ``rgroups_by_label`` maps an attachment label to its R-group SMILES.  All
  fragments are combined, all attachment bonds are added, then every dummy is
  removed.  Returns the fully assembled product SMILES (or "" on failure).
  """
  core = Chem.MolFromSmiles(core_smiles)
  if core is None:
    return ""
  combined = core
  rgs = {}
  offsets = {}
  for lbl, rsmiles in rgroups_by_label.items():
    rg = Chem.MolFromSmiles(rsmiles)
    if rg is None:
      return ""
    offsets[lbl] = combined.GetNumAtoms()
    combined = Chem.CombineMols(combined, rg)
    rgs[lbl] = rg
  rw = Chem.RWMol(combined)
  for lbl, rg in rgs.items():
    c_real = None
    for a in core.GetAtoms():
      if a.GetAtomicNum() == 0 and (a.GetIsotope() == lbl or a.GetAtomMapNum() == lbl):
        nbrs = list(a.GetNeighbors())
        if nbrs:
          c_real = nbrs[0].GetIdx()
    r_real = None
    for a in rg.GetAtoms():
      if a.GetAtomicNum() == 0 and (a.GetIsotope() == lbl or a.GetAtomMapNum() == lbl):
        nbrs = list(a.GetNeighbors())
        if nbrs:
          r_real = nbrs[0].GetIdx()
    if c_real is None or r_real is None:
      return ""
    rw.AddBond(c_real, r_real + offsets[lbl], Chem.BondType.SINGLE)
  mol = rw.GetMol()
  mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#0]"))
  try:
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)
  except (TypeError, ValueError, RuntimeError):
    return ""


def attach_rgroup(core_smiles: str, rgroup_smiles: str, label: int) -> str:
  """Connect an R-group onto a core at the labeled attachment point.

  Both the core and the R-group carry a matching labeled dummy (isotope or
  map-number form).  The two dummies are removed and a single bond joins their
  neighbours.  Returns the product SMILES (still carrying any other labels).

  Args:
    core_smiles: Core containing the labeled attachment point.
    rgroup_smiles: R-group containing the matching label.
    label: Attachment-point label.

  Returns:
    Product SMILES, or ``""`` if the attachment is invalid.
  """
  out = _attach_one(core_smiles, rgroup_smiles, label)
  return out


def enumerate_core(
  core_smiles: str,
  rgroups_by_label: Mapping[int, Sequence[str]],
  *,
  max_products: int = 100_000,
  dedupe: bool = True,
) -> list[str]:
  """Enumerate all products of a labeled core with lists of R-groups per label.

  ``rgroups_by_label`` maps an attachment label to a list of R-group SMILES
  (each carrying a matching dummy).  Returns the cartesian product as SMILES.

  Args:
    core_smiles: Core containing labeled attachment points.
    rgroups_by_label: R-group choices keyed by attachment label.
    max_products: Maximum number of products to return.
    dedupe: Whether to remove duplicate canonical products.

  Returns:
    Enumerated product SMILES.
  """
  import itertools

  labels = sorted(rgroups_by_label.keys())
  if not labels:
    return [core_smiles] if core_smiles else []
  products = []
  seen = set()
  for combo in itertools.product(*[rgroups_by_label[label] for label in labels]):
    out = _attach_many(core_smiles, dict(zip(labels, combo)))
    if not out:
      continue
    if dedupe:
      if out in seen:
        continue
      seen.add(out)
    products.append(out)
    if len(products) >= max_products:
      break
  return products


# ---------------------------------------------------------------- similarity
# MinHash + LSH + exact Tanimoto rescoring, all vectorized with NumPy/SciPy
# (no Numba).  The exact all-pairs path uses scipy.spatial.distance for small
# libraries; larger libraries go through MinHash/LSH candidate generation.

_PRIME = np.uint64(4294967291)  # largest prime below 2**32
_EMPTY_SIG = 0xFFFFFFFF
_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)


def make_permutations(
  n_permutations: int,
  seed: int = 0,
  n_bits: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Draw coefficients for MinHash permutations.

  Args:
    n_permutations: Number of hash functions.
    seed: Random generator seed.
    n_bits: Optional fingerprint width retained for API compatibility.

  Returns:
    Two uint64 coefficient arrays ``(a, b)``.
  """
  rng = np.random.default_rng(seed)
  a = rng.integers(1, int(_PRIME) - 1, size=n_permutations, dtype=np.uint64)
  b = rng.integers(0, int(_PRIME) - 1, size=n_permutations, dtype=np.uint64)
  return a, b


def minhash_signatures(
  offsets: np.ndarray,
  indices: np.ndarray,
  a: np.ndarray,
  b: np.ndarray,
) -> np.ndarray:
  """Compute MinHash signatures over on-bit CSR data.

  Args:
    offsets: CSR row offsets.
    indices: CSR set-bit indices.
    a: MinHash multiplier coefficients.
    b: MinHash offset coefficients.

  Returns:
    A uint32 signature matrix with one row per molecule.
  """
  n = offsets.shape[0] - 1
  n_perm = a.shape[0]
  sig = np.empty((n, n_perm), dtype=np.uint32)
  counts = np.diff(offsets)
  empty_rows = counts == 0
  nonempty_rows = np.flatnonzero(~empty_rows)
  for p in range(n_perm):
    h = (a[p] * indices + b[p]) % _PRIME
    col = np.full(n, _EMPTY_SIG, dtype=np.uint32)
    if nonempty_rows.size:
      col[nonempty_rows] = np.minimum.reduceat(h, offsets[nonempty_rows]).astype(np.uint32)
    sig[:, p] = col
  return sig


def band_hashes(signatures: np.ndarray, n_bands: int) -> np.ndarray:
  """Hash each signature band into one uint64 value per molecule."""
  n, n_perm = signatures.shape
  rows = n_perm // n_bands
  out = np.empty((n, n_bands), dtype=np.uint64)
  for band in range(n_bands):
    h = np.full(n, _FNV_OFFSET, dtype=np.uint64)
    base = band * rows
    for r in range(rows):
      h = (h ^ signatures[:, base + r].astype(np.uint64)) * _FNV_PRIME
    out[:, band] = h
  return out


def signature_hashes(signatures: np.ndarray) -> np.ndarray:
  """Hash each complete MinHash signature into one uint64 value."""
  n, n_perm = signatures.shape
  h = np.full(n, _FNV_OFFSET, dtype=np.uint64)
  for p in range(n_perm):
    h = (h ^ signatures[:, p].astype(np.uint64)) * _FNV_PRIME
  return h


def _band_candidates(keys, order, bucket_cap, n_nodes, capacity):
  """Emit up to ``bucket_cap`` neighbours per bucket member as packed keys."""
  ks = keys[order]
  boundaries = np.flatnonzero(ks[1:] != ks[:-1]) + 1
  starts = np.concatenate([[0], boundaries])
  ends = np.concatenate([boundaries, [len(ks)]])
  out = []
  count = 0
  nn = np.int64(n_nodes)
  for s, e in zip(starts, ends):
    size = int(e - s)
    if size < 2:
      continue
    members = order[s:e]
    for pos in range(size):
      i = np.int64(members[pos])
      steps = np.arange(1, min(bucket_cap, size) + 1, dtype=np.int64)
      others = members[(pos + steps) % size].astype(np.int64)
      others = others[others != i]  # never pair a node with itself
      if others.size == 0:
        continue
      a = np.minimum(i, others)
      b = np.maximum(i, others)
      packed = a * nn + b
      out.append(packed)
      count += packed.size
      if count >= capacity:
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)
  return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def lsh_candidate_pairs(
  signatures: np.ndarray,
  n_bands: int = 32,
  bucket_cap: int = 64,
  max_pairs: int = 40_000_000,
) -> tuple[np.ndarray, np.ndarray]:
  """Generate deduplicated LSH candidate pairs.

  Args:
    signatures: MinHash signature matrix.
    n_bands: Number of bands used for candidate generation.
    bucket_cap: Maximum candidates contributed by one bucket member.
    max_pairs: Maximum number of candidate pairs.

  Returns:
    Source and destination arrays with ``source < destination``.
  """
  signatures = np.ascontiguousarray(signatures, dtype=np.uint32)
  n, n_perm = signatures.shape
  if n < 2:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
  if n_perm % n_bands != 0:
    raise ValueError("n_permutations must be divisible by n_bands")

  hashes = band_hashes(signatures, n_bands)
  full = signature_hashes(signatures)
  capacity = int(min(max_pairs, n * bucket_cap))
  if capacity <= 0:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

  accumulated = []
  total = 0
  for band in range(n_bands):
    keys = hashes[:, band]
    order = np.lexsort((full, keys)).astype(np.int64)
    band_keys = _band_candidates(keys, order, bucket_cap, n, capacity - total)
    if band_keys.size == 0:
      continue
    band_keys = np.unique(band_keys)
    accumulated.append(band_keys)
    total += band_keys.size
    if total >= max_pairs:
      break

  if not accumulated:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
  keys = np.unique(np.concatenate(accumulated))
  if keys.size > max_pairs:
    keys = keys[:max_pairs]
  nn = np.int64(n)
  return keys // nn, keys % nn


def tanimoto_for_pairs(
  packed: np.ndarray,
  popcounts: np.ndarray,
  src: np.ndarray,
  dst: np.ndarray,
) -> np.ndarray:
  """Compute exact Tanimoto similarity for explicit fingerprint pairs.

  Args:
    packed: Packed fingerprint matrix.
    popcounts: Population counts for each fingerprint row.
    src: Source row indices.
    dst: Destination row indices.

  Returns:
    Float32 similarity scores.
  """
  inter = popcount_rows(packed[src] & packed[dst])
  union = popcounts[src] + popcounts[dst] - inter
  sim = np.zeros(src.size, dtype=np.float32)
  pos = union > 0
  sim[pos] = (inter[pos].astype(np.float64) / union[pos]).astype(np.float32)
  return sim


def knn_arrays_to_pairs(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Flatten a neighbour table into deduplicated undirected pairs.

  Args:
    idx: ``(n, k)`` neighbour indices, using negative values for empty slots.
    val: ``(n, k)`` neighbour scores.

  Returns:
    Source indices, destination indices, and scores.
  """
  n, _ = idx.shape
  rows = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
  cols = idx.reshape(-1)
  scores = val.reshape(-1)
  mask = cols >= 0
  rows, cols, scores = rows[mask], cols[mask], scores[mask]
  if rows.size == 0:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
  a = np.minimum(rows, cols)
  b = np.maximum(rows, cols)
  keys = a * np.int64(n) + b
  order = np.argsort(keys, kind="stable")
  keys, a, b, scores = keys[order], a[order], b[order], scores[order]
  unique_mask = np.empty(keys.size, dtype=bool)
  unique_mask[0] = True
  np.not_equal(keys[1:], keys[:-1], out=unique_mask[1:])
  return a[unique_mask], b[unique_mask], scores[unique_mask]


def _brute_force_knn(dense, k: int, threshold: float):
  """Exact top-k neighbours by pairwise Jaccard — for small libraries only."""
  from scipy.spatial.distance import pdist, squareform

  n = dense.shape[0]
  if n < 2:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
  dist = squareform(pdist(dense.astype(bool), metric="jaccard"))
  sim = 1.0 - dist
  np.fill_diagonal(sim, -1.0)
  idx = np.full((n, k), -1, dtype=np.int64)
  val = np.zeros((n, k), dtype=np.float32)
  for i in range(n):
    row = sim[i]
    cand = np.flatnonzero(row >= threshold)
    if cand.size == 0:
      continue
    top = cand[np.argsort(-row[cand], kind="stable")[:k]]
    idx[i, : top.size] = top
    val[i, : top.size] = row[top]
  return knn_arrays_to_pairs(idx, val)


def top_k_filter(
  src: np.ndarray,
  dst: np.ndarray,
  score: np.ndarray,
  n_nodes: int,
  k: int,
  mutual_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Keep the strongest ``k`` neighbours per compound.

  Args:
    src: Similarity-edge source indices.
    dst: Similarity-edge destination indices.
    score: Similarity scores parallel to ``src`` and ``dst``.
    n_nodes: Number of compounds.
    k: Maximum neighbours per compound.
    mutual_only: Keep only edges selected by both endpoints.

  Returns:
    Filtered source indices, destination indices, and scores.
  """
  if src.size == 0:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

  u = np.concatenate([src, dst])
  v = np.concatenate([dst, src])
  s = np.concatenate([score, score]).astype(np.float32)

  order = np.lexsort((-s, u))
  u, v, s = u[order], v[order], s[order]
  starts = np.searchsorted(u, np.arange(n_nodes, dtype=np.int64), side="left")
  rank = np.arange(u.size, dtype=np.int64) - starts[u]
  keep = rank < k
  u, v, s = u[keep], v[keep], s[keep]

  a = np.minimum(u, v)
  b = np.maximum(u, v)
  keys = a * np.int64(n_nodes) + b
  order = np.argsort(keys, kind="stable")
  keys, s = keys[order], s[order]
  unique_keys, first_idx, counts = np.unique(keys, return_index=True, return_counts=True)
  if mutual_only:
    selected = counts >= 2
    unique_keys, first_idx = unique_keys[selected], first_idx[selected]
  scores = s[first_idx]
  return (
    unique_keys // np.int64(n_nodes),
    unique_keys % np.int64(n_nodes),
    scores.astype(np.float32),
  )


def build_similarity_edges(
  fps: FingerprintBlock,
  cfg: SimilarityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Build compound-compound similarity edges ``(src, dst, score)``.

  Exact scan for small libraries, MinHash/LSH + exact rescoring above
  ``cfg.exact_below``.  Returned pairs satisfy ``src < dst`` and are unique.

  Args:
    fps: Packed fingerprints for the library.
    cfg: Optional similarity graph settings.

  Returns:
    Unique source indices, destination indices, and Tanimoto scores.
  """
  cfg = cfg or SimilarityConfig()
  n = fps.n_mols
  empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
  if n < 2:
    return empty

  packed = np.ascontiguousarray(fps.packed, dtype=np.uint64)
  popcounts = np.ascontiguousarray(fps.popcounts, dtype=np.int64)

  if n <= cfg.exact_below:
    src, dst, score = _brute_force_knn(fps.dense(), cfg.k, cfg.threshold)
    if cfg.mutual_only:
      return top_k_filter(src, dst, score, n, cfg.k, mutual_only=True)
    return src, dst, score

  offsets, indices = fps.onbits_csr()
  a, b = make_permutations(cfg.n_permutations, seed=cfg.seed, n_bits=fps.n_bits)
  signatures = minhash_signatures(offsets, indices, a, b)
  src, dst = lsh_candidate_pairs(
    signatures,
    n_bands=cfg.n_bands,
    bucket_cap=cfg.bucket_cap,
    max_pairs=cfg.max_candidate_pairs,
  )
  if src.size == 0:
    return empty
  score = tanimoto_for_pairs(packed, popcounts, src, dst)
  keep = score >= np.float32(cfg.threshold)
  src, dst, score = src[keep], dst[keep], score[keep]
  if src.size == 0:
    return empty
  return top_k_filter(src, dst, score, n, cfg.k, mutual_only=cfg.mutual_only)


# --------------------------------------------------------------------- graph


class ChemicalGraph:
  """Column-store heterogeneous chemical graph.

  Nodes: compounds, scaffolds, fragments.  Edges: compound-scaffold,
  compound-fragment, similarity, scaffold hierarchy, fragment sharing.
  Flat NumPy arrays plus a cached SciPy CSR adjacency for graph algorithms.
  """

  def __init__(
    self,
    node_type,
    smiles,
    compound_id,
    level,
    frequency,
    n_atoms,
    n_rings,
    mw,
    method,
    murcko,
    src,
    dst,
    edge_type,
    weight,
  ):
    self.node_type = np.ascontiguousarray(node_type, dtype=np.int8)
    self.smiles = list(smiles)
    self.compound_id = list(compound_id)
    self.level = np.ascontiguousarray(level, dtype=np.int32)
    self.frequency = np.ascontiguousarray(frequency, dtype=np.int64)
    self.n_atoms = np.ascontiguousarray(n_atoms, dtype=np.int32)
    self.n_rings = np.ascontiguousarray(n_rings, dtype=np.int32)
    self.mw = np.ascontiguousarray(mw, dtype=np.float32)
    self.method = np.ascontiguousarray(method, dtype=np.int8)
    self.murcko = list(murcko)
    self.src = np.ascontiguousarray(src, dtype=np.int64)
    self.dst = np.ascontiguousarray(dst, dtype=np.int64)
    self.edge_type = np.ascontiguousarray(edge_type, dtype=np.int8)
    self.weight = np.ascontiguousarray(weight, dtype=np.float32)
    self._csr_cache = {}
    self._compound_index = None
    self._smiles_index = None

  # ------------------------------------------------------------- basics
  @property
  def n_nodes(self) -> int:
    return int(self.node_type.size)

  @property
  def n_edges(self) -> int:
    return int(self.src.size)

  def count_nodes(self, node_type) -> int:
    return int(np.count_nonzero(self.node_type == int(node_type)))

  def count_edges(self, edge_type) -> int:
    return int(np.count_nonzero(self.edge_type == int(edge_type)))

  def nodes_of_type(self, node_type) -> np.ndarray:
    return np.flatnonzero(self.node_type == int(node_type)).astype(np.int64)

  def edges_of_type(self, edge_type) -> np.ndarray:
    return np.flatnonzero(self.edge_type == int(edge_type)).astype(np.int64)

  def node_id_of_compound(self, compound_id: str) -> int:
    if self._compound_index is None:
      self._compound_index = {cid: i for i, cid in enumerate(self.compound_id) if cid}
    return int(self._compound_index[compound_id])

  def node_id_of_smiles(self, smiles: str, node_type) -> int:
    key = (int(node_type), smiles)
    if self._smiles_index is None:
      self._smiles_index = {(int(self.node_type[i]), smi): i for i, smi in enumerate(self.smiles)}
    return int(self._smiles_index[key])

  def label(self, node: int) -> str:
    cid = self.compound_id[node]
    return cid if cid else self.smiles[node]

  # --------------------------------------------------------------- views
  def adjacency(self, edge_types=None) -> sp.csr_matrix:
    """Unweighted undirected CSR adjacency, optionally restricted to types."""
    key = None if edge_types is None else tuple(sorted(int(t) for t in edge_types))
    cached = self._csr_cache.get(key)
    if cached is not None:
      return cached
    if key is None:
      src, dst = self.src, self.dst
    else:
      mask = np.isin(self.edge_type, np.asarray(key, dtype=np.int8))
      src, dst = self.src[mask], self.dst[mask]
    n = self.n_nodes
    if src.size == 0:
      A = sp.csr_matrix((n, n), dtype=np.float32)
    else:
      # deduplicate (a compound may appear in several edge families)
      a = np.minimum(src, dst)
      b = np.maximum(src, dst)
      keys = a * np.int64(n) + b
      keys = np.unique(keys)
      a, b = keys // np.int64(n), keys % np.int64(n)
      rows = np.concatenate([a, b])
      cols = np.concatenate([b, a])
      A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
      A.data[:] = 1.0
    self._csr_cache[key] = A
    return A

  def weighted_adjacency(self, edge_types=None) -> sp.csr_matrix:
    """Undirected CSR with edge weights (used by community detection)."""
    key = ("w",) if edge_types is None else ("w",) + tuple(sorted(int(t) for t in edge_types))
    cached = self._csr_cache.get(key)
    if cached is not None:
      return cached
    if edge_types is None:
      src, dst, w = self.src, self.dst, self.weight
    else:
      mask = np.isin(self.edge_type, np.asarray([int(t) for t in edge_types], dtype=np.int8))
      src, dst, w = self.src[mask], self.dst[mask], self.weight[mask]
    n = self.n_nodes
    if src.size == 0:
      A = sp.csr_matrix((n, n), dtype=np.float32)
    else:
      rows = np.concatenate([src, dst])
      cols = np.concatenate([dst, src])
      data = np.concatenate([w, w]).astype(np.float64)
      A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    self._csr_cache[key] = A
    return A

  def degrees(self, edge_types=None) -> np.ndarray:
    A = self.adjacency(edge_types)
    return np.diff(A.indptr).astype(np.int64)

  def neighbors(self, node: int, edge_types=None) -> np.ndarray:
    A = self.adjacency(edge_types)
    return A.indices[A.indptr[node] : A.indptr[node + 1]]

  def neighborhood(self, node: int, radius: int = 1, edge_types=None) -> np.ndarray:
    A = self.adjacency(edge_types)
    dist = csgraph.dijkstra(A, directed=False, unweighted=True, indices=int(node))
    return np.flatnonzero((dist >= 0) & (dist <= radius)).astype(np.int64)

  def shortest_path(self, source: int, target: int, edge_types=None) -> np.ndarray:
    """One shortest hop path ``source -> target`` (empty if none)."""
    A = self.adjacency(edge_types)
    if source == target:
      return np.asarray([source], dtype=np.int64)
    dist, pred = csgraph.dijkstra(A, directed=False, unweighted=True, indices=int(source), return_predecessors=True)
    if not np.isfinite(dist[int(target)]):
      return np.empty(0, dtype=np.int64)
    path = [int(target)]
    node = int(target)
    while pred[node] != -9999 and pred[node] != source:
      node = int(pred[node])
      path.append(node)
    path.append(int(source))
    return np.asarray(path[::-1], dtype=np.int64)

  def connected_components(self, edge_types=None) -> np.ndarray:
    A = self.adjacency(edge_types)
    _, labels = csgraph.connected_components(A, directed=False)
    return labels.astype(np.int64)

  def pagerank(self, damping: float = 0.85, n_iter: int = 100, tol: float = 1e-8, edge_types=None) -> np.ndarray:
    A = self.adjacency(edge_types)
    n = A.shape[0]
    if n == 0:
      return np.empty(0, dtype=np.float64)
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(out_deg)
    pos = out_deg > 0
    inv[pos] = 1.0 / out_deg[pos]
    G = A.multiply(inv[:, None]).tocsr()
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    dangling = np.flatnonzero(out_deg == 0)
    for _ in range(n_iter):
      new = np.asarray(G.T.dot(rank)).ravel()
      d = float(rank[dangling].sum()) if dangling.size else 0.0
      new = damping * new + (1.0 - damping) / n + damping * d / n
      delta = np.abs(new - rank).sum()
      rank = new
      if delta < tol:
        break
    total = rank.sum()
    if total > 0:
      rank = rank / total
    return rank

  def average_path_length(self, sources: np.ndarray):
    """Average shortest-path length estimated from BFS trees of ``sources``."""
    A = self.adjacency()
    sources = np.ascontiguousarray(sources, dtype=np.int64)
    if sources.size == 0 or A.nnz == 0:
      return 0.0, 0
    dist = csgraph.shortest_path(A, directed=False, unweighted=True, indices=sources)
    finite = np.isfinite(dist)
    n_pairs = int(finite.sum()) - finite.shape[0]  # exclude the source itself
    if n_pairs <= 0:
      return 0.0, 0
    return float(dist[finite].sum() / n_pairs), n_pairs

  # ------------------------------------------------------------- export
  def to_dict(self) -> dict:
    return {
      "n_nodes": self.n_nodes,
      "n_edges": self.n_edges,
      "node_type": self.node_type.tolist(),
      "smiles": self.smiles,
      "compound_id": self.compound_id,
      "level": self.level.tolist(),
      "frequency": self.frequency.tolist(),
      "n_atoms": self.n_atoms.tolist(),
      "n_rings": self.n_rings.tolist(),
      "mw": self.mw.tolist(),
      "method": self.method.tolist(),
      "murcko": self.murcko,
      "src": self.src.tolist(),
      "dst": self.dst.tolist(),
      "edge_type": self.edge_type.tolist(),
      "weight": self.weight.tolist(),
    }

  @classmethod
  def from_dict(cls, data: dict) -> ChemicalGraph:
    return cls(
      node_type=data["node_type"],
      smiles=data["smiles"],
      compound_id=data["compound_id"],
      level=data["level"],
      frequency=data["frequency"],
      n_atoms=data["n_atoms"],
      n_rings=data["n_rings"],
      mw=data["mw"],
      method=data["method"],
      murcko=data["murcko"],
      src=data["src"],
      dst=data["dst"],
      edge_type=data["edge_type"],
      weight=data["weight"],
    )


class GraphBuilder:
  """Incremental builder that deduplicates scaffold and fragment nodes."""

  def __init__(self) -> None:
    self._node_type = []
    self._smiles = []
    self._compound_id = []
    self._level = []
    self._frequency = []
    self._n_atoms = []
    self._n_rings = []
    self._mw = []
    self._method = []
    self._murcko = []
    self._key_index = {}
    self._src = []
    self._dst = []
    self._etype = []
    self._weight = []

  def _add_node(
    self,
    node_type,
    smiles,
    compound_id="",
    level=-1,
    n_atoms=0,
    n_rings=0,
    mw=0.0,
    method=0,
    murcko="",
    dedupe=True,
  ) -> int:
    key = (int(node_type), smiles if not compound_id else compound_id)
    if dedupe:
      existing = self._key_index.get(key)
      if existing is not None:
        self._frequency[existing] += 1
        return existing
    node_id = len(self._node_type)
    self._key_index[key] = node_id
    self._node_type.append(int(node_type))
    self._smiles.append(smiles)
    self._compound_id.append(compound_id)
    self._level.append(level)
    self._frequency.append(1)
    self._n_atoms.append(n_atoms)
    self._n_rings.append(n_rings)
    self._mw.append(float(mw))
    self._method.append(int(method))
    self._murcko.append(murcko)
    return node_id

  def add_compound(self, smiles, compound_id, n_atoms=0, n_rings=0, mw=0.0, murcko=""):
    return self._add_node(
      NodeType.COMPOUND,
      smiles,
      compound_id=compound_id,
      level=0,
      n_atoms=n_atoms,
      n_rings=n_rings,
      mw=mw,
      murcko=murcko,
    )

  def add_scaffold(self, smiles, level=1, n_atoms=0, mw=0.0, frequency=1):
    return self._add_node(NodeType.SCAFFOLD, smiles, level=level, n_atoms=n_atoms, n_rings=level, mw=mw)

  def add_fragment(self, smiles, method=0, n_atoms=0, n_rings=0, mw=0.0):
    return self._add_node(NodeType.FRAGMENT, smiles, level=-1, n_atoms=n_atoms, n_rings=n_rings, mw=mw, method=method)

  def set_frequency(self, node_id: int, frequency: int) -> None:
    self._frequency[node_id] = int(frequency)

  def add_edge(self, src: int, dst: int, edge_type, weight: float = 1.0):
    """Add a single edge (convenience wrapper around :meth:`add_edges`)."""
    self.add_edges(
      np.asarray([src], dtype=np.int64),
      np.asarray([dst], dtype=np.int64),
      edge_type,
      np.asarray([weight], dtype=np.float32),
    )

  def add_edges(self, src, dst, edge_type, weight=1.0):
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    if src.size != dst.size:
      raise ValueError("src and dst must have equal length")
    if src.size == 0:
      return
    if np.isscalar(weight):
      weights = np.full(src.size, float(weight), dtype=np.float32)
    else:
      weights = np.ascontiguousarray(weight, dtype=np.float32)
    self._src.append(src)
    self._dst.append(dst)
    self._etype.append(np.full(src.size, int(edge_type), dtype=np.int8))
    self._weight.append(weights)

  def build(self) -> ChemicalGraph:
    def cat(parts, dtype):
      if not parts:
        return np.empty(0, dtype=dtype)
      return np.concatenate(parts).astype(dtype, copy=False)

    return ChemicalGraph(
      node_type=np.asarray(self._node_type, dtype=np.int8),
      smiles=list(self._smiles),
      compound_id=list(self._compound_id),
      level=np.asarray(self._level, dtype=np.int32),
      frequency=np.asarray(self._frequency, dtype=np.int64),
      n_atoms=np.asarray(self._n_atoms, dtype=np.int32),
      n_rings=np.asarray(self._n_rings, dtype=np.int32),
      mw=np.asarray(self._mw, dtype=np.float32),
      method=np.asarray(self._method, dtype=np.int8),
      murcko=list(self._murcko),
      src=cat(self._src, np.int64),
      dst=cat(self._dst, np.int64),
      edge_type=cat(self._etype, np.int8),
      weight=cat(self._weight, np.float32),
    )


# ---------------------------------------------------------------- analysis


def shannon_entropy(counts: Sequence[float] | np.ndarray) -> float:
  """Compute Shannon entropy in bits.

  Args:
    counts: Count vector; zero entries are ignored.

  Returns:
    Shannon entropy in bits.
  """
  counts = np.asarray(counts, dtype=np.float64)
  counts = counts[counts > 0]
  total = counts.sum()
  if total <= 0 or counts.size <= 1:
    return 0.0
  p = counts / total
  return float(-(p * np.log2(p)).sum())


def normalized_entropy(counts: Sequence[float] | np.ndarray) -> float:
  """Normalize Shannon entropy to the ``[0, 1]`` range.

  Args:
    counts: Count vector.

  Returns:
    Entropy divided by the maximum entropy for the observed support.
  """
  counts = np.asarray(counts, dtype=np.float64)
  k = int(np.count_nonzero(counts > 0))
  if k <= 1:
    return 0.0
  return shannon_entropy(counts) / np.log2(k)


def gini(counts: Sequence[float] | np.ndarray) -> float:
  """Compute the Gini concentration coefficient.

  Args:
    counts: Count vector.

  Returns:
    Gini coefficient where 0 is even and 1 is fully concentrated.
  """
  counts = np.sort(np.asarray(counts, dtype=np.float64))
  n = counts.size
  total = counts.sum()
  if n == 0 or total <= 0:
    return 0.0
  index = np.arange(1, n + 1, dtype=np.float64)
  return float((2.0 * (index * counts).sum()) / (n * total) - (n + 1.0) / n)


def _support(graph: ChemicalGraph, edge_type, node_type) -> np.ndarray:
  """Number of compound links per node of ``node_type``."""
  mask = graph.edge_type == int(edge_type)
  counts = np.zeros(graph.n_nodes, dtype=np.int64)
  if np.any(mask):
    endpoints = np.concatenate([graph.src[mask], graph.dst[mask]])
    target = endpoints[graph.node_type[endpoints] == int(node_type)]
    np.add.at(counts, target, 1)
  return counts


def scaffold_support(graph: ChemicalGraph) -> np.ndarray:
  """Count compound-to-scaffold links for every graph node."""
  return _support(graph, EdgeType.COMPOUND_SCAFFOLD, NodeType.SCAFFOLD)


def fragment_support(graph: ChemicalGraph) -> np.ndarray:
  """Count compound-to-fragment links for every graph node."""
  return _support(graph, EdgeType.COMPOUND_FRAGMENT, NodeType.FRAGMENT)


@dataclass
class DiversityMetrics:
  """Diversity summary of a library."""

  n_compounds: int = 0
  n_scaffold_nodes: int = 0
  n_populated_scaffolds: int = 0
  n_murcko_scaffolds: int = 0
  n_fragments: int = 0
  scaffold_entropy: float = 0.0
  scaffold_entropy_normalized: float = 0.0
  scaffold_gini: float = 0.0
  fragment_entropy: float = 0.0
  fragment_entropy_normalized: float = 0.0
  chemical_coverage: float = 0.0
  scaffold_redundancy: float = 0.0
  fragment_coverage: float = 0.0
  singleton_scaffold_fraction: float = 0.0
  compounds_per_scaffold: float = 0.0
  scaffold_levels: dict = field(default_factory=dict)
  top_scaffolds: list = field(default_factory=list)
  top_fragments: list = field(default_factory=list)

  def to_dict(self) -> dict:
    return asdict(self)


def diversity_metrics(graph: ChemicalGraph, top_n: int = 10) -> DiversityMetrics:
  """Compute the diversity block of a characterization report.

  Args:
    graph: Chemical graph to summarize.
    top_n: Number of top scaffolds and fragments to retain.

  Returns:
    Diversity metrics for the graph.
  """
  compounds = graph.nodes_of_type(NodeType.COMPOUND)
  scaffolds = graph.nodes_of_type(NodeType.SCAFFOLD)
  fragments = graph.nodes_of_type(NodeType.FRAGMENT)
  n_compounds = int(compounds.size)

  scaf_support = _support(graph, EdgeType.COMPOUND_SCAFFOLD, NodeType.SCAFFOLD)[scaffolds]
  frag_support = _support(graph, EdgeType.COMPOUND_FRAGMENT, NodeType.FRAGMENT)[fragments]
  populated = scaf_support[scaf_support > 0]

  murcko = {m for m in (graph.murcko[int(i)] for i in compounds) if m}
  levels = {}
  for level in graph.level[scaffolds].tolist():
    levels[int(level)] = levels.get(int(level), 0) + 1

  order = np.argsort(-scaf_support, kind="stable")[:top_n]
  top_scaffolds = [(graph.smiles[int(scaffolds[i])], int(scaf_support[i])) for i in order if scaf_support[i] > 0]
  frag_order = np.argsort(-frag_support, kind="stable")[:top_n]
  top_fragments = [(graph.smiles[int(fragments[i])], int(frag_support[i])) for i in frag_order if frag_support[i] > 0]

  n_populated = int(populated.size)
  return DiversityMetrics(
    n_compounds=n_compounds,
    n_scaffold_nodes=int(scaffolds.size),
    n_populated_scaffolds=n_populated,
    n_murcko_scaffolds=len(murcko),
    n_fragments=int(fragments.size),
    scaffold_entropy=shannon_entropy(populated),
    scaffold_entropy_normalized=normalized_entropy(populated),
    scaffold_gini=gini(populated),
    fragment_entropy=shannon_entropy(frag_support),
    fragment_entropy_normalized=normalized_entropy(frag_support),
    chemical_coverage=(n_populated / n_compounds) if n_compounds else 0.0,
    scaffold_redundancy=(1.0 - n_populated / n_compounds) if n_compounds else 0.0,
    fragment_coverage=(int(fragments.size) / n_compounds) if n_compounds else 0.0,
    singleton_scaffold_fraction=(float(np.count_nonzero(populated == 1) / n_populated) if n_populated else 0.0),
    compounds_per_scaffold=(float(populated.mean()) if n_populated else 0.0),
    scaffold_levels=dict(sorted(levels.items())),
    top_scaffolds=top_scaffolds,
    top_fragments=top_fragments,
  )


def _reachable_support(parents, children, candidates, support, n_nodes):
  """Sum ``support`` over distinct descendants of each candidate (self excluded)."""
  if parents.size == 0 or candidates.size == 0:
    return np.zeros(candidates.size, dtype=np.int64)
  A = sp.csr_matrix((np.ones(parents.size, dtype=np.float32), (parents, children)), shape=(n_nodes, n_nodes))
  sums = np.zeros(candidates.size, dtype=np.int64)
  for idx, c in enumerate(candidates):
    dist = csgraph.dijkstra(A, directed=True, unweighted=True, indices=int(c))
    reachable = np.flatnonzero(np.isfinite(dist))
    reachable = reachable[reachable != c]
    sums[idx] = int(support[reachable].sum())
  return sums


def frontier_scaffolds(
  graph: ChemicalGraph,
  min_support: int = 1,
  limit: int = 20,
) -> list[dict[str, Any]]:
  """Unexplored regions: general scaffolds whose descendants are populated.

  A frontier scaffold has at most ``min_support`` compounds of its own while
  its children in the hierarchy carry many compounds.  Ranked by descendant
  support.

  Args:
    graph: Chemical graph to inspect.
    min_support: Maximum direct support for a frontier scaffold.
    limit: Maximum number of frontier records.

  Returns:
    Ranked frontier scaffold records.
  """
  scaffolds = graph.nodes_of_type(NodeType.SCAFFOLD)
  if scaffolds.size == 0:
    return []
  support = _support(graph, EdgeType.COMPOUND_SCAFFOLD, NodeType.SCAFFOLD)

  hier = graph.edge_type == int(EdgeType.SCAFFOLD_HIERARCHY)
  parents = graph.src[hier]
  children = graph.dst[hier]

  n_children = np.zeros(graph.n_nodes, dtype=np.int64)
  if parents.size:
    np.add.at(n_children, parents, 1)

  candidates = np.asarray(
    [int(node) for node in scaffolds if support[node] <= min_support and n_children[node] > 0],
    dtype=np.int64,
  )
  if candidates.size == 0:
    return []

  sums = _reachable_support(parents, children, candidates, support, graph.n_nodes)
  descendant_support = dict(zip(candidates.tolist(), sums.tolist()))

  ranked = [node for node in candidates.tolist() if descendant_support[node] > 0]
  ranked.sort(key=lambda node: (-descendant_support[node], int(support[node]), graph.smiles[node]))
  return [
    {
      "node_id": node,
      "smiles": graph.smiles[node],
      "level": int(graph.level[node]),
      "support": int(support[node]),
      "n_children": int(n_children[node]),
      "descendant_support": int(descendant_support[node]),
    }
    for node in ranked[:limit]
  ]


@dataclass
class NetworkMetrics:
  """Topology summary of the chemical graph."""

  n_nodes: int = 0
  n_edges: int = 0
  density: float = 0.0
  mean_degree: float = 0.0
  median_degree: float = 0.0
  max_degree: int = 0
  degree_histogram: dict = field(default_factory=dict)
  n_components: int = 0
  largest_component_size: int = 0
  largest_component_fraction: float = 0.0
  n_singletons: int = 0
  component_size_distribution: list = field(default_factory=list)
  average_path_length: float = 0.0
  path_sample_pairs: int = 0
  path_length_exact: bool = False
  edge_type_counts: dict = field(default_factory=dict)
  node_type_counts: dict = field(default_factory=dict)
  central_nodes: list = field(default_factory=list)

  def to_dict(self) -> dict:
    return asdict(self)


_DEGREE_EDGES = [0, 1, 2, 3, 5, 9, 17, 33, 65]


def _degree_histogram(degrees: np.ndarray) -> dict:
  if degrees.size == 0:
    return {}
  hist = {}
  for i, low in enumerate(_DEGREE_EDGES):
    if i + 1 < len(_DEGREE_EDGES):
      high = _DEGREE_EDGES[i + 1] - 1
      count = int(np.count_nonzero((degrees >= low) & (degrees <= high)))
      key = f"{low}-{high}" if high > low else f"{low}"
    else:
      count = int(np.count_nonzero(degrees >= low))
      key = f"{low}+"
    if count:
      hist[key] = count
  return hist


def network_metrics(graph: ChemicalGraph, n_samples: int = 512, seed: int = 0, top_central: int = 10) -> NetworkMetrics:
  """Compute the network block of a characterization report.

  Args:
    graph: Chemical graph to summarize.
    n_samples: Maximum number of source nodes for path-length sampling.
    seed: Random generator seed.
    top_central: Number of PageRank-central nodes to retain.

  Returns:
    Network metrics for the graph.
  """
  n = graph.n_nodes
  A = graph.adjacency()
  degrees = np.diff(A.indptr).astype(np.int64)
  n_edges = int(A.nnz // 2)

  labels = graph.connected_components()
  if labels.size:
    _, sizes = np.unique(labels, return_counts=True)
    sizes = np.sort(sizes)[::-1]
  else:
    sizes = np.empty(0, dtype=np.int64)

  rng = np.random.default_rng(seed)
  if n > 0:
    sample_size = min(n_samples, n)
    sources = rng.choice(n, size=sample_size, replace=False).astype(np.int64)
    apl, pairs = graph.average_path_length(sources)
    exact = sample_size == n
  else:
    apl, pairs, exact = 0.0, 0, False

  pr = graph.pagerank() if n else np.empty(0)
  central = []
  if n:
    order = np.argsort(-pr, kind="stable")[:top_central]
    for node in order.tolist():
      central.append(
        {
          "node_id": int(node),
          "label": graph.label(node),
          "node_type": NODE_LABELS[int(graph.node_type[node])],
          "degree": int(degrees[node]),
          "pagerank": float(pr[node]),
        }
      )

  edge_counts = {EDGE_LABELS[t]: graph.count_edges(t) for t in sorted({int(x) for x in graph.edge_type.tolist()})}
  node_counts = {NODE_LABELS[t]: graph.count_nodes(t) for t in sorted({int(x) for x in graph.node_type.tolist()})}

  return NetworkMetrics(
    n_nodes=n,
    n_edges=n_edges,
    density=(2.0 * n_edges / (n * (n - 1))) if n > 1 else 0.0,
    mean_degree=float(degrees.mean()) if n else 0.0,
    median_degree=float(np.median(degrees)) if n else 0.0,
    max_degree=int(degrees.max()) if n else 0,
    degree_histogram=_degree_histogram(degrees),
    n_components=int(sizes.size),
    largest_component_size=int(sizes[0]) if sizes.size else 0,
    largest_component_fraction=float(sizes[0] / n) if sizes.size and n else 0.0,
    n_singletons=int(np.count_nonzero(sizes == 1)),
    component_size_distribution=[int(s) for s in sizes[:20]],
    average_path_length=float(apl),
    path_sample_pairs=int(pairs),
    path_length_exact=bool(exact),
    edge_type_counts=edge_counts,
    node_type_counts=node_counts,
    central_nodes=central,
  )


def louvain_local_moving(A: sp.csr_matrix, resolution: float = 1.0, n_iter: int = 20) -> np.ndarray:
  """Modularity local-moving (first Louvain phase) in pure NumPy.

  Community-detection fallback used because igraph/leidenalg are not
  Neurosnap dependencies.  Returns a community label per node.

  Args:
    A: Weighted sparse adjacency matrix.
    resolution: Modularity resolution parameter.
    n_iter: Maximum local-moving iterations.

  Returns:
    Dense community label for every node.
  """
  n = A.shape[0]
  if n == 0:
    return np.empty(0, dtype=np.int64)
  k = np.asarray(A.sum(axis=1)).ravel()  # weighted degree
  two_m = k.sum()
  if two_m <= 0.0:
    return np.arange(n, dtype=np.int64)

  community = np.arange(n, dtype=np.int64)
  tot = k.copy()

  for _ in range(n_iter):
    moves = 0
    for i in range(n):
      start, end = A.indptr[i], A.indptr[i + 1]
      if start == end:
        continue
      nbrs = A.indices[start:end]
      weights = A.data[start:end]
      nbr_comm = community[nbrs]
      uniq, inv = np.unique(nbr_comm, return_inverse=True)
      acc = np.bincount(inv, weights=weights, minlength=uniq.size)

      current = int(community[i])
      tot[current] -= k[i]
      best = current
      best_gain = -resolution * tot[current] * k[i] / two_m
      for t, c in enumerate(uniq.tolist()):
        gain = acc[t] - resolution * tot[c] * k[i] / two_m
        if gain > best_gain:
          best_gain = gain
          best = c
      tot[best] += k[i]
      community[i] = best
      if best != current:
        moves += 1
    if moves == 0:
      break

  # relabel densely
  mapping = {}
  out = np.empty(n, dtype=np.int64)
  for i in range(n):
    c = int(community[i])
    if c not in mapping:
      mapping[c] = len(mapping)
    out[i] = mapping[c]
  return out


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
  """Relabel communities so island 0 is the largest."""
  if labels.size == 0:
    return labels
  unique, counts = np.unique(labels, return_counts=True)
  order = unique[np.lexsort((unique, -counts))]
  mapping = np.empty(unique.max() + 1, dtype=np.int64)
  for new, old in enumerate(order.tolist()):
    mapping[old] = new
  return mapping[labels]


@dataclass
class IslandResult:
  """Community structure of the compound similarity graph."""

  labels: np.ndarray
  compound_nodes: np.ndarray
  islands: list = field(default_factory=list)
  bridges: list = field(default_factory=list)
  method: str = "louvain_local_moving"
  modularity: float = 0.0
  resolution: float = 1.0

  @property
  def n_islands(self) -> int:
    return len(self.islands)

  @property
  def n_bridges(self) -> int:
    return len(self.bridges)

  def to_dict(self) -> dict:
    return {
      "n_islands": self.n_islands,
      "method": self.method,
      "modularity": self.modularity,
      "resolution": self.resolution,
      "islands": self.islands,
      "n_bridges": self.n_bridges,
      "bridges": self.bridges[:50],
    }


def detect_islands(graph: ChemicalGraph, resolution: float = 1.0, top_scaffolds: int = 3) -> IslandResult:
  """Detect chemical islands and describe their chemistry.

  Args:
    graph: Chemical graph to analyze.
    resolution: Local-moving community resolution.
    top_scaffolds: Number of representative scaffolds per island.

  Returns:
    Island labels and per-island summaries.
  """
  compounds = graph.nodes_of_type(NodeType.COMPOUND)
  n = int(compounds.size)
  if n == 0:
    return IslandResult(np.empty(0, dtype=np.int64), compounds, [], [], "none", 0.0, resolution)

  local = np.full(graph.n_nodes, -1, dtype=np.int64)
  local[compounds] = np.arange(n, dtype=np.int64)

  sim = graph.edge_type == int(EdgeType.COMPOUND_SIMILARITY)
  src = local[graph.src[sim]]
  dst = local[graph.dst[sim]]
  weight = graph.weight[sim].astype(np.float64)
  valid = (src >= 0) & (dst >= 0) & (src != dst)
  src, dst, weight = src[valid], dst[valid], weight[valid]

  method = "none"
  if src.size == 0:
    labels = np.arange(n, dtype=np.int64)
    method = "singletons"
  else:
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    A = sp.csr_matrix((np.concatenate([weight, weight]), (rows, cols)), shape=(n, n))
    labels = louvain_local_moving(A, resolution)
    method = "louvain_local_moving"

  labels = _relabel_by_size(labels)

  # scaffold membership per compound
  scaf_mask = graph.edge_type == int(EdgeType.COMPOUND_SCAFFOLD)
  scaf_src, scaf_dst = graph.src[scaf_mask], graph.dst[scaf_mask]
  compound_side = np.where(graph.node_type[scaf_src] == int(NodeType.COMPOUND), scaf_src, scaf_dst)
  scaffold_side = np.where(graph.node_type[scaf_src] == int(NodeType.COMPOUND), scaf_dst, scaf_src)
  compound_island = np.full(graph.n_nodes, -1, dtype=np.int64)
  compound_island[compounds] = labels

  n_islands = int(labels.max()) + 1 if labels.size else 0
  island_scaffolds = [set() for _ in range(n_islands)]
  for c_node, s_node in zip(compound_side.astype(np.int64).tolist(), scaffold_side.astype(np.int64).tolist()):
    island = int(compound_island[c_node])
    if island >= 0:
      island_scaffolds[island].add(int(s_node))

  scaffold_owner_count = {}
  for members in island_scaffolds:
    for s in members:
      scaffold_owner_count[s] = scaffold_owner_count.get(s, 0) + 1

  # internal / bridging similarity edges
  island_src = labels[src]
  island_dst = labels[dst]
  internal = island_src == island_dst
  internal_sum = np.zeros(n_islands, dtype=np.float64)
  internal_count = np.zeros(n_islands, dtype=np.int64)
  np.add.at(internal_sum, island_src[internal], weight[internal])
  np.add.at(internal_count, island_src[internal], 1)

  islands = []
  for island in range(n_islands):
    members = compounds[labels == island]
    scaffolds = island_scaffolds[island]
    exclusive = sum(1 for s in scaffolds if scaffold_owner_count.get(s, 0) == 1)
    scaffold_counts = {}
    for c_node, s_node in zip(compound_side.astype(np.int64).tolist(), scaffold_side.astype(np.int64).tolist()):
      if int(compound_island[c_node]) == island:
        smi = graph.smiles[int(s_node)]
        scaffold_counts[smi] = scaffold_counts.get(smi, 0) + 1
    top = sorted(scaffold_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_scaffolds]
    mean_sim = float(internal_sum[island] / internal_count[island]) if internal_count[island] else 0.0
    n_members = int(members.size)
    islands.append(
      {
        "island_id": island,
        "n_compounds": n_members,
        "n_scaffolds": len(scaffolds),
        "n_exclusive_scaffolds": exclusive,
        "scaffold_exclusivity": (exclusive / len(scaffolds)) if scaffolds else 0.0,
        "compounds_per_scaffold": (n_members / len(scaffolds)) if scaffolds else 0.0,
        "internal_edges": int(internal_count[island]),
        "mean_internal_similarity": mean_sim,
        "mean_mw": float(graph.mw[members].mean()) if n_members else 0.0,
        "representative": graph.smiles[int(members[0])] if n_members else "",
        "top_scaffolds": top,
        "unique_chemistry": bool(scaffolds) and exclusive == len(scaffolds),
      }
    )

  bridge_idx = np.flatnonzero(~internal)
  bridges = []
  if bridge_idx.size:
    order = bridge_idx[np.argsort(-weight[bridge_idx], kind="stable")]
    for e in order.tolist():
      bridges.append(
        {
          "island_a": int(island_src[e]),
          "island_b": int(island_dst[e]),
          "compound_a": graph.compound_id[int(compounds[src[e]])],
          "compound_b": graph.compound_id[int(compounds[dst[e]])],
          "weight": float(weight[e]),
        }
      )

  return IslandResult(
    labels=labels,
    compound_nodes=compounds,
    islands=islands,
    bridges=bridges,
    method=method,
    modularity=0.0,  # local-moving does not report a global modularity
    resolution=resolution,
  )


@dataclass
class CharacterizationReport:
  """Everything the engine can say about a library."""

  diversity: DiversityMetrics
  network: NetworkMetrics
  islands: IslandResult
  frontier: list = field(default_factory=list)
  counts: dict = field(default_factory=dict)
  metadata: dict = field(default_factory=dict)

  def to_dict(self) -> dict:
    return {
      "counts": self.counts,
      "diversity": self.diversity.to_dict(),
      "network": self.network.to_dict(),
      "islands": self.islands.to_dict(),
      "frontier": self.frontier,
      "metadata": self.metadata,
    }

  def summary(self) -> str:
    """Human-readable answer to 'does this library contain real diversity?'"""
    d = self.diversity
    n = self.network
    islands = sorted(self.islands.islands, key=lambda x: -x["n_compounds"])
    lines = []
    add = lines.append

    add("Chemical landscape summary")
    add("=" * 60)
    add(f"Compounds            {d.n_compounds:>12,}")
    add(f"Scaffold nodes       {d.n_scaffold_nodes:>12,}  (populated {d.n_populated_scaffolds:,})")
    add(f"Fragment nodes       {d.n_fragments:>12,}")
    add(f"Edges                {n.n_edges:>12,}")
    for label, count in n.edge_type_counts.items():
      add(f"  {label:<26} {count:>10,}")
    add("")
    add("Diversity")
    add(f"  scaffold entropy          {d.scaffold_entropy:8.3f} bits (normalized {d.scaffold_entropy_normalized:.3f})")
    add(f"  fragment entropy          {d.fragment_entropy:8.3f} bits (normalized {d.fragment_entropy_normalized:.3f})")
    add(f"  chemical coverage         {d.chemical_coverage:8.3f} (scaffolds per compound)")
    add(f"  scaffold redundancy       {d.scaffold_redundancy:8.3f}")
    add(f"  compounds per scaffold    {d.compounds_per_scaffold:8.2f}")
    add(f"  singleton scaffolds       {d.singleton_scaffold_fraction:8.3f} of populated")
    add(f"  scaffold concentration    {d.scaffold_gini:8.3f} (Gini)")
    if d.top_scaffolds:
      add("  most populated scaffolds:")
      for smi, count in d.top_scaffolds[:5]:
        add(f"    {count:>8,}  {smi}")
    add("")
    add("Topology")
    add(f"  density                   {n.density:.6f}")
    add(f"  mean degree               {n.mean_degree:8.2f} (max {n.max_degree:,})")
    add(f"  components                {n.n_components:>8,} (largest {n.largest_component_fraction:.1%}, singletons {n.n_singletons:,})")
    suffix = "exact" if n.path_length_exact else f"sampled over {n.path_sample_pairs:,} pairs"
    add(f"  average path length       {n.average_path_length:8.2f} ({suffix})")
    if n.central_nodes:
      add("  central nodes:")
      for node in n.central_nodes[:5]:
        add(f"    {node['node_type']:<9} deg {node['degree']:>6,}  {node['label']}")
    add("")
    add(f"Islands ({self.islands.n_islands:,} chemical islands, method={self.islands.method})")
    for island in islands[:10]:
      add(
        f"  Island {island['island_id']:<4} "
        f"{island['n_compounds']:>8,} compounds | "
        f"{island['n_scaffolds']:>6,} scaffolds | "
        f"mean sim {island['mean_internal_similarity']:.2f} | "
        f"exclusive {island['scaffold_exclusivity']:.0%}" + ("  [unique chemistry]" if island["unique_chemistry"] else "")
      )
      if island["representative"]:
        add(f"           e.g. {island['representative']}")
    if self.islands.n_bridges:
      add(f"  bridges between islands   {self.islands.n_bridges:,}")
    add("")
    if self.frontier:
      add("Unexplored frontier (general scaffolds with little direct support)")
      for entry in self.frontier[:5]:
        add(
          f"  support {entry['support']:>4,} | descendants {entry['descendant_support']:>8,} | children {entry['n_children']:>4,}  {entry['smiles']}"
        )
    else:
      add("Unexplored frontier: none detected")
    return "\n".join(lines)


def characterize(
  graph: ChemicalGraph,
  n_samples: int = 512,
  seed: int = 0,
  resolution: float = 1.0,
  min_frontier_support: int = 1,
  metadata: Mapping[str, Any] | None = None,
) -> CharacterizationReport:
  """Run every analysis block over a built chemical graph.

  Args:
    graph: Built chemical graph.
    n_samples: Maximum number of path-length source nodes.
    seed: Random generator seed.
    resolution: Island community resolution.
    min_frontier_support: Maximum direct support for frontier scaffolds.
    metadata: Optional metadata copied into the report.

  Returns:
    Complete characterization report.
  """
  diversity = diversity_metrics(graph)
  network = network_metrics(graph, n_samples=n_samples, seed=seed)
  islands = detect_islands(graph, resolution=resolution)
  frontier = frontier_scaffolds(graph, min_support=min_frontier_support)
  counts = {
    "n_nodes": graph.n_nodes,
    "n_edges": graph.n_edges,
    "n_compounds": graph.count_nodes(NodeType.COMPOUND),
    "n_scaffolds": graph.count_nodes(NodeType.SCAFFOLD),
    "n_fragments": graph.count_nodes(NodeType.FRAGMENT),
    "n_similarity_edges": graph.count_edges(EdgeType.COMPOUND_SIMILARITY),
    "n_hierarchy_edges": graph.count_edges(EdgeType.SCAFFOLD_HIERARCHY),
    "n_compound_scaffold_edges": graph.count_edges(EdgeType.COMPOUND_SCAFFOLD),
    "n_compound_fragment_edges": graph.count_edges(EdgeType.COMPOUND_FRAGMENT),
    "n_fragment_shared_edges": graph.count_edges(EdgeType.FRAGMENT_SHARED),
  }
  return CharacterizationReport(
    diversity=diversity,
    network=network,
    islands=islands,
    frontier=frontier,
    counts=counts,
    metadata=metadata or {},
  )


# ------------------------------------------------------------- persistence


def save_landscape(
  graph: ChemicalGraph,
  fingerprints: FingerprintBlock | None,
  config: LandscapeConfig | None,
  path: str | Path,
  extra: Mapping[str, Any] | None = None,
) -> Path:
  """Persist the landscape as JSON plus an optional NPZ fingerprint store.

  Args:
    graph: Chemical graph to persist.
    fingerprints: Optional packed fingerprint block.
    config: Optional landscape configuration.
    path: Destination directory.
    extra: Optional JSON-serializable metadata.

  Returns:
    The destination directory.
  """
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  payload = {
    "graph": graph.to_dict(),
    "config": config.to_dict() if config is not None else {},
    "extra": extra or {},
  }
  (path / "landscape.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
  if fingerprints is not None:
    np.savez_compressed(
      path / "fingerprints.npz",
      packed=fingerprints.packed,
      popcounts=fingerprints.popcounts,
      n_bits=fingerprints.n_bits,
    )
  return path


def load_landscape(path: str | Path) -> tuple[ChemicalGraph, FingerprintBlock | None, LandscapeConfig, dict[str, Any]]:
  """Load a landscape written by :func:`save_landscape`.

  Args:
    path: Landscape directory.

  Returns:
    Graph, optional fingerprints, configuration, and extra metadata.
  """
  path = Path(path)
  payload = json.loads((path / "landscape.json").read_text(encoding="utf-8"))
  graph = ChemicalGraph.from_dict(payload["graph"])
  config = LandscapeConfig.from_dict(payload.get("config", {}))
  fingerprints = None
  fp_path = path / "fingerprints.npz"
  if fp_path.exists():
    with np.load(fp_path) as data:
      fingerprints = FingerprintBlock(data["packed"], data["popcounts"], int(data["n_bits"]))
  extra = payload.get("extra", {})
  return graph, fingerprints, config, extra


# ---------------------------------------------------------------- exports


def to_json_graph(
  graph: ChemicalGraph,
  node_types: Sequence[NodeType] | None = None,
  edge_types: Sequence[EdgeType] | None = None,
  max_nodes: int | None = None,
) -> dict[str, Any]:
  """Build a D3/Cytoscape-friendly node-link dictionary.

  Args:
    graph: Chemical graph to serialize.
    node_types: Optional node-type filter.
    edge_types: Optional edge-type filter.
    max_nodes: Optional cap on retained nodes.

  Returns:
    JSON-serializable nodes, links, and metadata.
  """
  if node_types is None:
    keep_nodes = np.ones(graph.n_nodes, dtype=bool)
  else:
    keep_nodes = np.isin(graph.node_type, np.asarray([int(t) for t in node_types], dtype=np.int8))
  if max_nodes is not None:
    allowed = np.flatnonzero(keep_nodes)[:max_nodes]
    keep_nodes = np.zeros(graph.n_nodes, dtype=bool)
    keep_nodes[allowed] = True
  if edge_types is None:
    keep_edges = np.ones(graph.n_edges, dtype=bool)
  else:
    keep_edges = np.isin(graph.edge_type, np.asarray([int(t) for t in edge_types], dtype=np.int8))
  keep_edges &= keep_nodes[graph.src] & keep_nodes[graph.dst]

  nodes = []
  for node in np.flatnonzero(keep_nodes).tolist():
    entry = {
      "id": node,
      "node_type": NODE_LABELS[int(graph.node_type[node])],
      "smiles": graph.smiles[node],
      "level": int(graph.level[node]),
      "frequency": int(graph.frequency[node]),
      "n_atoms": int(graph.n_atoms[node]),
      "n_rings": int(graph.n_rings[node]),
      "mw": float(graph.mw[node]),
    }
    if graph.compound_id[node]:
      entry["compound_id"] = graph.compound_id[node]
    if graph.murcko[node]:
      entry["murcko"] = graph.murcko[node]
    nodes.append(entry)

  links = [
    {
      "source": int(graph.src[e]),
      "target": int(graph.dst[e]),
      "edge_type": EDGE_LABELS[int(graph.edge_type[e])],
      "weight": round(float(graph.weight[e]), 6),
    }
    for e in np.flatnonzero(keep_edges).tolist()
  ]
  return {
    "nodes": nodes,
    "links": links,
    "metadata": {
      "directed": False,
      "multigraph": False,
      "n_nodes": len(nodes),
      "n_links": len(links),
      "node_types": sorted({n["node_type"] for n in nodes}),
      "edge_types": sorted({link["edge_type"] for link in links}),
    },
  }


def export_json(
  graph: ChemicalGraph,
  path: str | Path,
  indent: int | None = None,
  **kwargs: Any,
) -> Path:
  """Write a node-link JSON representation of ``graph``."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(to_json_graph(graph, **kwargs), indent=indent), encoding="utf-8")
  return path


def export_graphml(
  graph: ChemicalGraph,
  path: str | Path,
  node_types: Sequence[NodeType] | None = None,
  edge_types: Sequence[EdgeType] | None = None,
) -> Path:
  """Streaming GraphML export (no NetworkX)."""
  from xml.sax.saxutils import escape

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if node_types is None:
    keep_nodes = np.ones(graph.n_nodes, dtype=bool)
  else:
    keep_nodes = np.isin(graph.node_type, np.asarray([int(t) for t in node_types], dtype=np.int8))
  if edge_types is None:
    keep_edges = np.ones(graph.n_edges, dtype=bool)
  else:
    keep_edges = np.isin(graph.edge_type, np.asarray([int(t) for t in edge_types], dtype=np.int8))
  keep_edges &= keep_nodes[graph.src] & keep_nodes[graph.dst]

  with path.open("w", encoding="utf-8") as fh:
    fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    fh.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
    for name, kind in (
      ("node_type", "string"),
      ("smiles", "string"),
      ("compound_id", "string"),
      ("level", "int"),
      ("frequency", "long"),
      ("n_atoms", "int"),
      ("n_rings", "int"),
      ("mw", "double"),
      ("murcko", "string"),
    ):
      fh.write(f'  <key id="n_{name}" for="node" attr.name="{name}" attr.type="{kind}"/>\n')
    fh.write('  <key id="e_edge_type" for="edge" attr.name="edge_type" attr.type="string"/>\n')
    fh.write('  <key id="e_weight" for="edge" attr.name="weight" attr.type="double"/>\n')
    fh.write('  <graph id="chemical_landscape" edgedefault="undirected">\n')
    for node in np.flatnonzero(keep_nodes).tolist():
      fh.write(f'    <node id="n{node}">\n')
      fh.write(f'      <data key="n_node_type">{NODE_LABELS[int(graph.node_type[node])]}</data>\n')
      fh.write(f'      <data key="n_smiles">{escape(graph.smiles[node])}</data>\n')
      if graph.compound_id[node]:
        fh.write(f'      <data key="n_compound_id">{escape(graph.compound_id[node])}</data>\n')
      fh.write(f'      <data key="n_level">{int(graph.level[node])}</data>\n')
      fh.write(f'      <data key="n_frequency">{int(graph.frequency[node])}</data>\n')
      fh.write(f'      <data key="n_n_atoms">{int(graph.n_atoms[node])}</data>\n')
      fh.write(f'      <data key="n_n_rings">{int(graph.n_rings[node])}</data>\n')
      fh.write(f'      <data key="n_mw">{float(graph.mw[node]):.4f}</data>\n')
      if graph.murcko[node]:
        fh.write(f'      <data key="n_murcko">{escape(graph.murcko[node])}</data>\n')
      fh.write("    </node>\n")
    for e in np.flatnonzero(keep_edges).tolist():
      src, dst = int(graph.src[e]), int(graph.dst[e])
      label = EDGE_LABELS[int(graph.edge_type[e])]
      fh.write(
        f'    <edge id="e{e}" source="n{src}" target="n{dst}">\n'
        f'      <data key="e_edge_type">{label}</data>\n'
        f'      <data key="e_weight">{float(graph.weight[e]):.6f}</data>\n'
        "    </edge>\n"
      )
    fh.write("  </graph>\n</graphml>\n")
  return path


# ------------------------------------------------------------------- plots


def _agg_backend():
  """Force the headless Agg backend regardless of MPLBACKEND env vars."""
  import os as _os

  _os.environ["MPLBACKEND"] = "Agg"
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  return plt


def plot_scaffold_map(graph: ChemicalGraph, path: str | Path) -> Path:
  """Scaffold support vs. hierarchy level, plus the dominant scaffolds.

  Args:
    graph: Chemical graph to visualize.
    path: Output image path.

  Returns:
    The output path.
  """
  plt = _agg_backend()

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
  scaffolds = graph.nodes_of_type(NodeType.SCAFFOLD)
  if scaffolds.size == 0:
    for ax in axes:
      ax.text(0.5, 0.5, "no scaffolds", ha="center", va="center")
      ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

  support = scaffold_support(graph)[scaffolds]
  levels = graph.level[scaffolds]
  rng = np.random.default_rng(0)
  jitter = rng.normal(0.0, 0.06, size=levels.size)

  ax = axes[0]
  sizes = 12 + 40 * np.log1p(support) / max(np.log1p(support.max()), 1e-9)
  ax.scatter(np.log10(support + 1), levels + jitter, s=sizes, c=support, cmap="viridis", alpha=0.75)
  ax.set_xlabel("log10(compound support + 1)")
  ax.set_ylabel("scaffold level (ring count)")
  ax.set_title(f"Scaffold map — {scaffolds.size:,} scaffold nodes")
  ax.set_yticks(sorted({int(v) for v in levels.tolist()}))

  ax = axes[1]
  top_n = 15
  order = np.argsort(-support)[:top_n]
  labels = [graph.smiles[int(scaffolds[i])][:27] for i in order][::-1]
  values = support[order][::-1]
  ax.barh(np.arange(len(values)), values, color="#b5563f", alpha=0.85)
  ax.set_yticks(np.arange(len(values)))
  ax.set_yticklabels(labels, fontsize=7)
  ax.set_xlabel("compounds")
  ax.set_title("Most populated scaffolds")
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_islands(islands: IslandResult, path: str | Path) -> Path:
  """Plot island size and scaffold exclusivity.

  Args:
    islands: Island analysis result.
    path: Output image path.

  Returns:
    The output path.
  """
  plt = _agg_backend()

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(7, 4.6))
  data = islands.islands
  if not data:
    ax.text(0.5, 0.5, "no islands", ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
  n_compounds = np.asarray([d["n_compounds"] for d in data], dtype=float)
  n_scaffolds = np.asarray([d["n_scaffolds"] for d in data], dtype=float)
  exclusivity = np.asarray([d["scaffold_exclusivity"] for d in data], dtype=float)
  ax.scatter(
    n_compounds,
    np.maximum(n_scaffolds, 0.5),
    s=25 + 200 * n_compounds / n_compounds.max(),
    c=exclusivity,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    alpha=0.85,
    edgecolors="k",
    linewidths=0.3,
  )
  ax.set_xscale("log")
  ax.set_xlabel("compounds per island")
  ax.set_ylabel("scaffolds per island")
  ax.set_title(f"Chemical islands — {islands.n_islands:,} islands, {islands.method}")
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
  return path


def plot_diversity_report(report: CharacterizationReport, path: str | Path) -> Path:
  """Write a compact image summary of a characterization report.

  Args:
    report: Characterization report to render.
    path: Output image path.

  Returns:
    The output path.
  """
  plt = _agg_backend()

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(9, 4.6))
  d = report.diversity
  n = report.network
  ax.axis("off")
  lines = [
    f"n compounds: {d.n_compounds:,}",
    f"scaffold nodes: {d.n_scaffold_nodes:,} (populated {d.n_populated_scaffolds:,})",
    f"fragment nodes: {d.n_fragments:,}",
    f"edges: {n.n_edges:,}",
    f"chemical coverage: {d.chemical_coverage:.3f}",
    f"scaffold entropy: {d.scaffold_entropy:.3f} bits (norm {d.scaffold_entropy_normalized:.3f})",
    f"scaffold redundancy: {d.scaffold_redundancy:.3f}",
    f"islands: {report.islands.n_islands:,}",
    f"components: {n.n_components:,} (largest {n.largest_component_fraction:.1%})",
    f"average path length: {n.average_path_length:.2f}",
  ]
  ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
  ax.set_title("Diversity report")
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
  return path


# ---------------------------------------------------------------- facade


class _Compounds:
  """Loaded compound layer (arrays only)."""

  def __init__(self):
    self.ids = []
    self.smiles = []
    self.n_atoms = []
    self.n_rings = []
    self.mw = []
    self.failures = []
    self.reverse = []  # list[ReverseFragmentRecord | None], parallel to smiles

  @property
  def n(self) -> int:
    return len(self.smiles)


class ChemicalLandscape:
  """A molecular library and the chemical landscape built from it.

  The source may be a path or an in-memory SMILES sequence:

      landscape = ChemicalLandscape("library.csv", smiles_column="smiles")
      landscape.build_all()
      report = landscape.characterize()
      print(report.summary())
      landscape.save("out/")
      reloaded = ChemicalLandscape.from_store("out/")

  Args:
    source: Input path or in-memory SMILES sequence.
    smiles_column: SMILES column for delimited input files.
    id_column: Optional compound identifier column.
    config: Optional complete landscape configuration.
    compound_ids: IDs corresponding to an in-memory SMILES sequence.
    **overrides: Configuration field overrides.
  """

  def __init__(
    self,
    source: str | Path | Sequence[str] | None = None,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    config: LandscapeConfig | None = None,
    *,
    compound_ids: Sequence[str] | None = None,
    **overrides: Any,
  ) -> None:
    self.config = config or LandscapeConfig(smiles_column=smiles_column, id_column=id_column)
    if config is None:
      self.config.smiles_column = smiles_column
      self.config.id_column = id_column
    nested_configs = {
      "fingerprints": FingerprintConfig,
      "scaffolds": ScaffoldConfig,
      "fragments": FragmentConfig,
      "similarity": SimilarityConfig,
    }
    for key, value in overrides.items():
      if not hasattr(self.config, key):
        raise TypeError(f"unknown config option {key!r}")
      if key in nested_configs and isinstance(value, dict):
        value = nested_configs[key](**value)
      setattr(self.config, key, value)

    self.source = source
    self._compounds = _Compounds()
    self._fingerprints = None
    self._scaffolds = None
    self._fragments = None
    self._similarity = None
    self._graph = None
    self._report = None
    self.timings = {}

    if source is not None and not isinstance(source, (str, Path)):
      self._load_from_sequence(list(source), compound_ids)

  # ------------------------------------------------------------------ load
  def _load_from_sequence(self, smiles, compound_ids=None):
    ids = list(compound_ids) if compound_ids else [f"mol-{i:07d}" for i in range(len(smiles))]
    if len(ids) != len(smiles):
      raise ValueError("compound_ids must have the same length as the SMILES sequence")
    self._consume_chunks([RecordChunk(ids, list(smiles), 0)])

  def _consume_chunks(self, chunks):
    started = time.perf_counter()
    c = self._compounds
    seen_ids = {cid for cid in c.ids}
    seen_ids.update(cid for cid, _ in c.failures)
    for chunk in chunks:
      ids, smis = chunk.compound_ids, chunk.smiles
      if len(ids) != len(smis):
        raise ValueError("compound IDs and SMILES must have equal lengths")
      for cid in ids:
        if cid in seen_ids:
          raise ValueError(f"compound IDs must be unique; duplicate {cid!r}")
        seen_ids.add(cid)
      for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None or mol.GetNumAtoms() == 0:
          c.failures.append((ids[i], smi))
          continue
        c.ids.append(ids[i])
        c.smiles.append(Chem.MolToSmiles(mol))
        c.n_atoms.append(mol.GetNumHeavyAtoms())
        c.n_rings.append(int(rdMolDescriptors.CalcNumRings(mol)))
        c.mw.append(float(Descriptors.MolWt(mol)))
    self._fingerprints = morgan_packed(c.smiles, self.config.fingerprints)
    self._invalidate()
    self.timings["load"] = time.perf_counter() - started

  def load(self) -> ChemicalLandscape:
    """Parse the input, canonicalize SMILES and build fingerprints."""
    if self._compounds.n:
      return self
    if self.source is None:
      raise ValueError("no source given")
    if isinstance(self.source, (str, Path)):
      chunks = list(
        stream_chunks(
          self.source,
          smiles_column=self.config.smiles_column,
          id_column=self.config.id_column,
          chunk_size=self.config.chunk_size,
          limit=self.config.limit,
          delimiter=self.config.delimiter,
        )
      )
      self._consume_chunks(chunks)
    return self

  # ---------------------------------------------------------------- stages
  def build_scaffolds(self) -> ChemicalLandscape:
    """Annotate Bemis-Murcko scaffolds and build the scaffold network."""
    self.load()
    started = time.perf_counter()
    self._scaffolds = scaffold_network(self._compounds.smiles, self.config.scaffolds)
    self._invalidate()
    self.timings["scaffolds"] = time.perf_counter() - started
    return self

  def build_fragments(self) -> ChemicalLandscape:
    """Fragment every compound (BRICS, rotatable bonds, linkers).

    When ``FragmentConfig.reversible`` is enabled, each compound also keeps
    a :class:`ReverseFragmentRecord` (attachment points + cut orders) so it
    can be rebuilt with :meth:`reassemble`.
    """
    self.load()
    started = time.perf_counter()
    cfg = self.config.fragments
    self._fragments = fragment_library(self._compounds.smiles, cfg)
    if cfg.reversible:
      self._compounds.reverse = [reversible_fragment_record(smi, cfg) for smi in self._compounds.smiles]
    self._invalidate()
    self.timings["fragments"] = time.perf_counter() - started
    return self

  def build_similarity_graph(self) -> ChemicalLandscape:
    """Build the sparse compound-compound similarity graph."""
    self.load()
    started = time.perf_counter()
    if self._fingerprints is None:
      raise RuntimeError("fingerprints are missing; call load() first")
    self._similarity = build_similarity_edges(self._fingerprints, self.config.similarity)
    self._invalidate()
    self.timings["similarity"] = time.perf_counter() - started
    return self

  def build_all(self) -> ChemicalLandscape:
    """Run every stage."""
    return self.load().build_scaffolds().build_fragments().build_similarity_graph()

  def _invalidate(self) -> None:
    self._graph = None
    self._report = None

  # ----------------------------------------------------------------- graph
  @property
  def graph(self) -> ChemicalGraph:
    """The heterogeneous chemical graph (assembled on first access)."""
    if self._graph is None:
      self._graph = self._assemble()
    return self._graph

  @property
  def fingerprints(self) -> FingerprintBlock | None:
    """Packed Morgan fingerprints, if the library has been loaded."""
    return self._fingerprints

  @property
  def compound_ids(self) -> list[str]:
    """Canonical compound identifiers in library order."""
    return list(self._compounds.ids)

  @property
  def smiles(self) -> list[str]:
    """Canonical SMILES in library order."""
    return list(self._compounds.smiles)

  @property
  def failures(self) -> list[tuple[str, str]]:
    """``(compound_id, smiles)`` pairs RDKit could not parse."""
    return list(self._compounds.failures)

  def __len__(self) -> int:
    return self._compounds.n

  def _assemble(self) -> ChemicalGraph:
    started = time.perf_counter()
    builder = GraphBuilder()
    c = self._compounds
    murcko = self._scaffolds.murcko if self._scaffolds is not None else None

    for i in range(c.n):
      builder.add_compound(
        c.smiles[i],
        compound_id=c.ids[i],
        n_atoms=int(c.n_atoms[i]),
        n_rings=int(c.n_rings[i]),
        mw=float(c.mw[i]),
        murcko=murcko[i] if murcko else "",
      )

    if self._scaffolds is not None and self._scaffolds.n_scaffolds:
      scaf = self._scaffolds
      scaffold_nodes = np.empty(scaf.n_scaffolds, dtype=np.int64)
      for local, smi in enumerate(scaf.scaffolds):
        scaffold_nodes[local] = builder.add_scaffold(smi, level=int(scaf.levels[local]))
      linked = scaf.compound_scaffold >= 0
      compound_side = np.flatnonzero(linked).astype(np.int64)
      builder.add_edges(
        compound_side,
        scaffold_nodes[scaf.compound_scaffold[linked]],
        EdgeType.COMPOUND_SCAFFOLD,
      )
      if scaf.n_hierarchy_edges:
        builder.add_edges(
          scaffold_nodes[scaf.hierarchy_parent],
          scaffold_nodes[scaf.hierarchy_child],
          EdgeType.SCAFFOLD_HIERARCHY,
        )
      support = np.bincount(scaf.compound_scaffold[linked], minlength=scaf.n_scaffolds).astype(np.int64)
      for local in range(scaf.n_scaffolds):
        builder.set_frequency(int(scaffold_nodes[local]), int(support[local]))

    if self._fragments is not None and self._fragments.n_fragments:
      frag = self._fragments
      fragment_nodes = np.empty(frag.n_fragments, dtype=np.int64)
      for local, smi in enumerate(frag.fragments):
        fragment_nodes[local] = builder.add_fragment(smi, method=int(frag.methods[local]))
        builder.set_frequency(int(fragment_nodes[local]), int(frag.frequencies[local]))
      builder.add_edges(
        frag.compound_fragment_src,
        fragment_nodes[frag.compound_fragment_dst],
        EdgeType.COMPOUND_FRAGMENT,
      )
      shared_a, shared_b = shared_fragment_edges(
        frag.ring_systems,
        frag.frequencies,
        links_per_fragment=self.config.fragments.shared_links_per_fragment,
      )
      if shared_a.size:
        builder.add_edges(fragment_nodes[shared_a], fragment_nodes[shared_b], EdgeType.FRAGMENT_SHARED)

    if self._similarity is not None:
      src, dst, score = self._similarity
      builder.add_edges(src, dst, EdgeType.COMPOUND_SIMILARITY, score)

    graph = builder.build()
    self.timings["assemble"] = time.perf_counter() - started
    return graph

  # -------------------------------------------------------------- analysis
  def characterize(self, n_samples: int = 512, seed: int = 0, resolution: float = 1.0) -> CharacterizationReport:
    """Run the full characterization and cache the report."""
    started = time.perf_counter()
    report = characterize(
      self.graph,
      n_samples=n_samples,
      seed=seed,
      resolution=resolution,
      metadata={
        "source": str(self.source) if isinstance(self.source, (str, Path)) else "sequence",
        "n_input_failures": len(self._compounds.failures),
        "timings": dict(self.timings),
        "config": self.config.to_dict(),
      },
    )
    self._report = report
    self.timings["characterize"] = time.perf_counter() - started
    return report

  @property
  def report(self) -> CharacterizationReport:
    """Return the cached characterization report, building it if needed."""
    if self._report is None:
      return self.characterize()
    return self._report

  # ------------------------------------------------------------ traversal
  def node_of(self, compound_id: str) -> int:
    """Return the graph node ID for a compound identifier."""
    return self.graph.node_id_of_compound(compound_id)

  def neighbors(self, compound_id: str, edge_types: Sequence[EdgeType] | None = None) -> list[str]:
    """Return labels of the direct neighbours of a compound.

    Args:
      compound_id: Compound identifier to query.
      edge_types: Optional edge-type filter.

    Returns:
      Labels of directly connected nodes.
    """
    graph = self.graph
    return [graph.label(int(n)) for n in graph.neighbors(self.node_of(compound_id), edge_types)]

  def path_between(self, compound_a: str, compound_b: str) -> list[str]:
    """Return a traversal path between two compounds.

    Args:
      compound_a: First compound identifier.
      compound_b: Second compound identifier.

    Returns:
      Labels along the shortest graph path.
    """
    graph = self.graph
    path = graph.shortest_path(self.node_of(compound_a), self.node_of(compound_b))
    return [graph.label(int(n)) for n in path]

  def island_of(self, compound_id: str) -> int:
    """Island id of a compound (-1 if it was not part of the analysis)."""
    report = self.report
    node = self.node_of(compound_id)
    nodes = report.islands.compound_nodes
    match = np.flatnonzero(nodes == node)
    return int(report.islands.labels[match[0]]) if match.size else -1

  # ------------------------------------------------------- fragments & cores

  def reassemble(self, compound_id: str) -> str:
    """Rebuild a compound's canonical SMILES from its fragments.

    Uses the reversible fragment record (attachment points + cut orders)
    captured during :meth:`build_fragments`.  Returns ``""`` if the compound
    was not fragmented reversibly.
    """
    idx = self.node_of(compound_id)
    rec = self._compounds.reverse[idx] if idx < len(self._compounds.reverse) else None
    if rec is None or rec.n_cuts == 0:
      return ""
    return reassemble_fragments(rec.pieces, rec.cut_orders)

  def common_cores(self, n: int = 10) -> list[str]:
    """Return the most frequently shared scaffold SMILES.

    Args:
      n: Maximum number of scaffold SMILES.

    Returns:
      Scaffold SMILES ordered by support.
    """
    if self._scaffolds is None:
      self.build_scaffolds()
    g = self.graph
    scaffolds = g.nodes_of_type(NodeType.SCAFFOLD)
    support = scaffold_support(g)[scaffolds]
    order = np.argsort(-support, kind="stable")
    out = []
    for i in order:
      if support[i] <= 0:
        break
      out.append(g.smiles[int(scaffolds[i])])
      if len(out) >= n:
        break
    return out

  def decompose(self, core_smiles: str | None = None, *, params: Any = None) -> RGroupDecompositionResult:
    """Decompose the library into a core + per-position R-groups.

    If ``core_smiles`` is omitted, the most frequent scaffold is used.

    Args:
      core_smiles: Optional labeled or unlabeled core SMILES.
      params: Optional RDKit R-group decomposition parameters.

    Returns:
      Per-molecule R-group decomposition.
    """
    self.load()
    if core_smiles is None:
      cores = self.common_cores(1)
      if not cores:
        raise ValueError("no shared scaffold found to use as a core")
      core_smiles = cores[0]
    return decompose_molecules(self._compounds.smiles, core_smiles, params=params)

  def enumerate(
    self,
    core_smiles: str,
    rgroups_by_label: Mapping[int, Sequence[str]],
    *,
    max_products: int = 100_000,
  ) -> list[str]:
    """Enumerate all products of a labeled core with R-groups per position.

    ``rgroups_by_label`` maps an attachment label to a list of R-group
    SMILES (each carrying a matching labeled dummy, e.g. ``CO[*:1]``).

    Args:
      core_smiles: Labeled core SMILES.
      rgroups_by_label: R-group choices keyed by attachment label.
      max_products: Maximum products to return.

    Returns:
      Enumerated product SMILES.
    """
    return enumerate_core(core_smiles, rgroups_by_label, max_products=max_products)

  def rgroups_at(self, compound_id: str, core_smiles: str | None = None) -> dict[str, str]:
    """Return one compound's R-groups for a core.

    Args:
      compound_id: Compound identifier to decompose.
      core_smiles: Optional labeled or unlabeled core SMILES.

    Returns:
      Mapping of R-group labels to SMILES.
    """
    self.load()
    if core_smiles is None:
      cores = self.common_cores(1)
      if not cores:
        raise ValueError("no shared scaffold found to use as a core")
      core_smiles = cores[0]
    idx = self.node_of(compound_id)
    decomp = decompose_molecules([self._compounds.smiles[idx]], core_smiles)
    return decomp.rows[0] if decomp.rows else {}

  def swap_rgroup(self, compound_id: str, position: int, new_rgroup: str, core_smiles: str | None = None) -> str:
    """Replace an R-group at ``position`` and return the new molecule.

    ``new_rgroup`` should carry a dummy labeled for ``position`` (e.g.
    ``CO[*:1]`` for position 1).  The compound is decomposed against a core,
    the labelled position is substituted, and the product is reassembled
    with the other R-groups left in place.
    """
    self.load()
    if core_smiles is None:
      cores = self.common_cores(1)
      if not cores:
        raise ValueError("no shared scaffold found to use as a core")
      core_smiles = cores[0]
    idx = self.node_of(compound_id)
    decomp = decompose_molecules([self._compounds.smiles[idx]], core_smiles)
    if not decomp.rows:
      return ""
    row = decomp.rows[0]
    key = f"R{position}"
    if key not in row:
      raise KeyError(f"compound has no R-group at position {position}")
    row[key] = new_rgroup
    # rebuild from the labeled core + all R-groups
    return _attach_many(decomp.core_smiles, {_rgroup_number(k): v for k, v in row.items() if k != "Core"})

  # ---------------------------------------------------------------- store
  def save(self, path: str | Path) -> Path:
    """Persist the landscape as JSON plus an optional NPZ fingerprint store."""
    return save_landscape(
      self.graph,
      fingerprints=self._fingerprints,
      config=self.config,
      path=path,
      extra={
        "timings": dict(self.timings),
        "input": {
          "source": str(self.source) if isinstance(self.source, (str, Path)) else "sequence",
          "n_records": self._compounds.n,
          "n_failures": len(self._compounds.failures),
        },
        "reverse": [rec.to_dict() if rec is not None else None for rec in self._compounds.reverse],
      },
    )

  @classmethod
  def from_store(cls, path: str | Path) -> ChemicalLandscape:
    """Load a previously persisted landscape.

    Args:
      path: Landscape directory created by :meth:`save`.

    Returns:
      Restored :class:`ChemicalLandscape` instance.
    """
    graph, fingerprints, config, extra = load_landscape(path)
    library = cls(source=None, config=config)
    library._graph = graph
    compounds = graph.nodes_of_type(NodeType.COMPOUND)
    library._compounds.ids = [graph.compound_id[int(i)] for i in compounds]
    library._compounds.smiles = [graph.smiles[int(i)] for i in compounds]
    library._compounds.n_atoms = [int(graph.n_atoms[int(i)]) for i in compounds]
    library._compounds.n_rings = [int(graph.n_rings[int(i)]) for i in compounds]
    library._compounds.mw = [float(graph.mw[int(i)]) for i in compounds]
    rev = extra.get("reverse", [])
    library._compounds.reverse = [ReverseFragmentRecord.from_dict(r) if r is not None else None for r in rev]
    library._fingerprints = fingerprints
    library.timings = dict(extra.get("timings", {}))
    return library

  # --------------------------------------------------------------- export
  def export_graphml(self, path: str | Path, **kwargs: Any) -> Path:
    """Export the assembled graph as GraphML."""
    return export_graphml(self.graph, path, **kwargs)

  def export_json(self, path: str | Path, **kwargs: Any) -> Path:
    """Export the assembled graph as node-link JSON."""
    return export_json(self.graph, path, **kwargs)

  def plot(self, outdir: str | Path) -> list[Path]:
    """Write the scaffold map, island plot, and diversity report.

    Args:
      outdir: Destination directory for the three image files.

    Returns:
      Paths to the generated image files.
    """
    from pathlib import Path as _Path

    outdir = _Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = self.report
    return [
      plot_scaffold_map(self.graph, outdir / "scaffold_map.png"),
      plot_islands(report.islands, outdir / "chemical_islands.png"),
      plot_diversity_report(report, outdir / "diversity_report.png"),
    ]


__all__ = [
  "EDGE_LABELS",
  "NODE_LABELS",
  "CharacterizationReport",
  "ChemicalGraph",
  "ChemicalLandscape",
  "DiversityMetrics",
  "EdgeType",
  "FingerprintBlock",
  "FingerprintConfig",
  "FragmentConfig",
  "FragmentMethod",
  "FragmentResult",
  "GraphBuilder",
  "IslandResult",
  "LandscapeConfig",
  "NetworkMetrics",
  "NodeType",
  "RGroupDecompositionResult",
  "ReverseFragmentRecord",
  "ScaffoldConfig",
  "ScaffoldNetworkResult",
  "SimilarityConfig",
  "apply_reversible_cut",
  "attach_rgroup",
  "build_similarity_edges",
  "characterize",
  "decompose_molecules",
  "detect_islands",
  "diversity_metrics",
  "enumerate_core",
  "export_graphml",
  "export_json",
  "fragment_cut_bonds",
  "fragment_library",
  "fragment_molecule",
  "frontier_scaffolds",
  "load_landscape",
  "louvain_local_moving",
  "minhash_signatures",
  "morgan_packed",
  "murcko_smiles",
  "network_metrics",
  "popcount_rows",
  "popcount_words",
  "reassemble_fragments",
  "reversible_fragment_record",
  "save_landscape",
  "scaffold_network",
  "shannon_entropy",
  "shared_fragment_edges",
  "stream_chunks",
]
