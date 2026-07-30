"""Interaction detection rules and geometry helpers for structure analysis.

This module holds the rule implementations behind
:meth:`neurosnap.structure.structure.Structure.get_interactions`. Callers first
build an :class:`InteractionContext` with :func:`build_context`, then invoke the
individual ``detect_*`` rules against it. Each rule returns
:class:`~neurosnap.structure.interaction_report.InteractionRecord` objects and
owns exactly one interaction family, which keeps the rules independently
testable and lets the public analyzer stay a thin orchestrator.

The universal length unit is Å.
"""

import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from neurosnap._compat import compat_dataclass
from neurosnap.constants.chemistry import METAL_ELEMENTS, VDW_RADII_BONDI

from .interaction_report import AtomReference, CoordinationCenterRecord, InteractionRecord
from .interaction_rules import (
  CANONICAL_AMINO_ACIDS,
  METAL_DONOR_ELEMENTS,
  PROTEIN_IONIC_NEGATIVE,
  PROTEIN_IONIC_POSITIVE,
  PROTEIN_SIDECHAIN_ACCEPTORS,
  PROTEIN_SIDECHAIN_DONORS,
  RULE_SET,
  RULE_VERSION,
)
from .structure import BondType, InteractionType

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for annotations only
  from .structure import Structure

#: Interaction families this engine can *detect* from geometry and chemistry.
#:
#: Distinct from :class:`~neurosnap.structure.structure.InteractionType`, which
#: labels the noncovalent interactions a source file *declares*. This set is
#: wider because it also covers families that are only ever derived, such as
#: ``contact`` and ``clash``.
DETECTABLE_INTERACTION_TYPES = frozenset(
  {
    "hydrogen_bond",
    "disulfide",
    "salt_bridge",
    "vdw_clash",
    "metal_coordination",
    "contact",
    "vdw_contact",
    "clash",
    "covalent",
  }
)

#: Residue names treated as solvent rather than as a non-polymer ligand.
WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "H2O", "SOL", "TIP"})

#: Cysteine residue names searched for disulfide ``SG`` pairs.
CYSTEINE_RESIDUE_NAMES = frozenset({"CYS", "CYX"})

#: Backbone and terminal oxygens treated as hydrogen-bond acceptors.
BACKBONE_ACCEPTOR_ATOM_NAMES = frozenset({"O", "OXT", "O1", "O2"})

#: Elements accepted as the hydrogen of a donor-H-acceptor triplet.
HBOND_HYDROGEN_ELEMENTS = frozenset({"H", "D"})

#: Elements treated as hydrogen when filtering plain contacts.
CONTACT_HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})

#: Evidence labels that rank as directly observed rather than explicit or candidate.
DETECTED_EVIDENCE_LABELS = frozenset({"detected", "vdw_overlap", "vdw_contact", "distance_cutoff"})

#: Fallback van der Waals radius for elements missing from the Bondi table.
DEFAULT_VDW_RADIUS = 1.7

#: Reference sulfur-sulfur van der Waals sum used for disulfide gap reporting.
DISULFIDE_REFERENCE_VDW_SUM = 3.6

#: Maximum donor-hydrogen distance used when a residue carries no explicit bonds.
IMPLICIT_HYDROGEN_MAX_DISTANCE = 1.2

#: Coordination numbers for which an idealized geometry template exists.
MIN_GEOMETRY_DONORS = 2
MAX_GEOMETRY_DONORS = 6


class LazyDistanceMatrix:
  """Sparse pairwise distance cache seeded from a KD-tree radius query.

  Pairs within ``threshold`` are precomputed in bulk and exposed through
  :meth:`seeded_pairs` and :meth:`seeded_items`. Lookups outside that radius are
  computed on demand and memoized separately, so a rule that scans the seeded
  set always sees the same pairs no matter which other rules ran first.

  Parameters:
    coords: Atom coordinates of shape ``(n_atoms, 3)``.
    tree: A ``scipy.spatial.KDTree`` already built over ``coords``.
    threshold: Radius in Å to precompute pairs for.
  """

  def __init__(self, coords, tree, threshold=4.5):
    self.coords = coords
    self.tree = tree
    self.threshold = threshold
    self._seeded = {}
    self._memo = {}
    pairs = tree.query_pairs(threshold, output_type="ndarray")
    if len(pairs) > 0:
      diff = coords[pairs[:, 0]] - coords[pairs[:, 1]]
      dvals = np.linalg.norm(diff, axis=-1)
      for (u, v), d in zip(pairs, dvals):
        self._seeded[(u, v)] = float(d)

  def seeded_items(self):
    """Return ``((low, high), distance)`` for every pair within ``threshold``."""
    return self._seeded.items()

  def __getitem__(self, key):
    u, v = key
    if u == v:
      return 0.0
    k = (u, v) if u < v else (v, u)
    if k in self._seeded:
      return self._seeded[k]
    if k in self._memo:
      return self._memo[k]
    val = float(np.linalg.norm(self.coords[u] - self.coords[v]))
    self._memo[k] = val
    return val


@functools.lru_cache(maxsize=1)
def _periodic_table():
  """Return the cached RDKit periodic table."""
  from rdkit import Chem

  return Chem.GetPeriodicTable()


@functools.lru_cache(maxsize=1)
def _feature_factory():
  """Return the cached RDKit chemical feature factory built from ``BaseFeatures.fdef``."""
  import os

  from rdkit import RDConfig
  from rdkit.Chem import ChemicalFeatures

  return ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))


def get_vdw_radius(element: str) -> float:
  """Return the Bondi van der Waals radius for an element symbol.

  Parameters:
    element: Element symbol in any casing.

  Returns:
    Radius in Å, falling back to :data:`DEFAULT_VDW_RADIUS` for unknown elements.
  """
  return VDW_RADII_BONDI.get(element.title(), DEFAULT_VDW_RADIUS)


def get_covalent_radius(element: str) -> Optional[float]:
  """Return the RDKit covalent radius for an element symbol.

  Parameters:
    element: Element symbol in any casing.

  Returns:
    Radius in Å, or ``None`` when the element is empty or unknown to RDKit.
  """
  if not element:
    return None
  try:
    return float(_periodic_table().GetRcovalent(element.title()))
  except Exception:
    return None


def is_non_polymer_entity(entity: Any, atom_annotations: np.ndarray) -> bool:
  """Report whether an entity contains at least one non-solvent heterogen atom.

  Parameters:
    entity: Entity whose ``atom_indices`` are checked.
    atom_annotations: Structured annotation array for the parent structure.

  Returns:
    ``True`` when the entity holds a heterogen atom that is not solvent.
  """
  if "hetero" not in atom_annotations.dtype.names:
    return False
  hetero_mask = atom_annotations["hetero"]
  res_names = atom_annotations["res_name"]
  for idx in entity.atom_indices:
    if hetero_mask[idx] and res_names[idx].strip().upper() not in WATER_RESIDUE_NAMES:
      return True
  return False


def get_rdkit_donors_and_acceptors(mol: Any) -> Tuple[Set[int], Set[int]]:
  """Return hydrogen-bond donor and acceptor atom ids for an RDKit molecule.

  Parameters:
    mol: Aligned RDKit molecule, or ``None``.

  Returns:
    Tuple of ``(donor_ids, acceptor_ids)`` as molecule-local atom indices.
  """
  donors: Set[int] = set()
  acceptors: Set[int] = set()
  if mol is None:
    return donors, acceptors
  for feat in _feature_factory().GetFeaturesForMol(mol):
    family = feat.GetFamily()
    if family == "Donor":
      donors.update(feat.GetAtomIds())
    elif family == "Acceptor":
      acceptors.update(feat.GetAtomIds())
  return donors, acceptors


def evidence_priority(evidence: str) -> int:
  """Rank an evidence label so stronger evidence survives deduplication.

  Parameters:
    evidence: Evidence label from a detected record.

  Returns:
    ``3`` for explicit, ``2`` for directly observed, ``1`` for candidate, ``0`` otherwise.
  """
  evidence = (evidence or "").lower()
  if evidence == "explicit":
    return 3
  if evidence in DETECTED_EVIDENCE_LABELS:
    return 2
  if evidence == "candidate":
    return 1
  return 0


def resolve_interaction_types(interaction_types: Optional[Sequence[str]]) -> List[str]:
  """Validate requested interaction types, defaulting to every supported family.

  Parameters:
    interaction_types: Requested family names, or ``None`` for all of them.

  Returns:
    List of validated family names.

  Raises:
    ValueError: If any requested name is not in :data:`DETECTABLE_INTERACTION_TYPES`.
  """
  if interaction_types is None:
    return list(DETECTABLE_INTERACTION_TYPES)
  resolved = []
  for name in interaction_types:
    if name not in DETECTABLE_INTERACTION_TYPES:
      raise ValueError(f"unsupported interaction type: {name}")
    resolved.append(name)
  return resolved


def validate_ligand_topology(entities: Sequence[Any], atom_annotations: np.ndarray, types_to_run: Sequence[str]) -> None:
  """Require aligned RDKit molecules when chemistry-typed rules involve ligands.

  Parameters:
    entities: Entities participating in the analysis.
    atom_annotations: Structured annotation array for the parent structure.
    types_to_run: Interaction families that will be evaluated.

  Raises:
    ValueError: If a non-polymer entity lacks an RDKit molecule or its atom
      count disagrees with the entity size.
  """
  requires_typing = {"hydrogen_bond", "salt_bridge"}
  if not any(name in requires_typing for name in types_to_run):
    return
  if not any(is_non_polymer_entity(entity, atom_annotations) for entity in entities):
    return
  for entity in entities:
    if not is_non_polymer_entity(entity, atom_annotations):
      continue
    if entity.rdkit_mol is None:
      raise ValueError(f"Entity {entity.name} is missing an aligned RDKit molecule (mismatch).")
    if entity.rdkit_mol.GetNumAtoms() != len(entity.atom_indices):
      raise ValueError(
        f"RDKit molecule atom count ({entity.rdkit_mol.GetNumAtoms()}) does not match entity atom count ({len(entity.atom_indices)}) (mismatch)."
      )


@compat_dataclass()
class InteractionContext:
  """Precomputed structure views shared by every interaction rule.

  Building this once avoids recomputing atom references, entity lookups, bond
  adjacency, and the distance cache for each rule.

  Attributes:
    atom_refs: Immutable per-atom metadata in atom-index order.
    coords: Atom coordinates of shape ``(n_atoms, 3)``.
    entities: Entities participating in the analysis.
    atom_annotations: Structured annotation array for the parent structure.
    bonds: Structured bond array for the parent structure.
    interactions: Noncovalent interactions declared by the source file.
    pairwise_dist: Distance cache, or ``None`` for an empty structure.
    atom_to_entity_name: Map of atom index to owning entity name.
    atom_to_entity_idx: Map of atom index to owning entity position.
    atom_to_residue: Map of atom index to its :class:`Residue` view.
    n_terminal_residues: Residues that begin a chain.
    bonded_pairs: Sorted ``(atom_i, atom_j)`` pairs that are bonded.
    bonded_adjacency: Adjacency list built from ``bonded_pairs``.
    model_id: Model identifier stamped onto emitted records.
  """

  atom_refs: List[AtomReference]
  coords: np.ndarray
  entities: List[Any]
  atom_annotations: np.ndarray
  bonds: np.ndarray
  interactions: np.ndarray
  pairwise_dist: Optional[LazyDistanceMatrix]
  atom_to_entity_name: Dict[int, str]
  atom_to_entity_idx: Dict[int, int]
  atom_to_residue: Dict[int, Any]
  n_terminal_residues: Set[Any]
  bonded_pairs: Set[Tuple[int, int]]
  bonded_adjacency: Dict[int, List[int]]
  model_id: int

  def __len__(self) -> int:
    return len(self.atom_refs)

  def entity_name(self, atom_index: int) -> str:
    """Return the entity name owning an atom, or an empty string."""
    return self.atom_to_entity_name.get(atom_index, "")

  def declared_bond_pairs(self, bond_type: BondType) -> Set[Tuple[int, int]]:
    """Return atom pairs the source file declared as bonds of one category.

    Parameters:
      bond_type: Bond category to select.

    Returns:
      Set of ``(low, high)`` atom-index pairs.
    """
    return {
      (min(int(bond["atom_i"]), int(bond["atom_j"])), max(int(bond["atom_i"]), int(bond["atom_j"])))
      for bond in self.bonds
      if int(bond["bond_type"]) == int(bond_type)
    }

  def declared_interaction_pairs(self, interaction_type: InteractionType) -> Set[Tuple[int, int]]:
    """Return atom pairs the source file declared as one noncovalent interaction.

    Parameters:
      interaction_type: Interaction category to select.

    Returns:
      Set of ``(low, high)`` atom-index pairs.
    """
    return {
      (min(int(row["atom_i"]), int(row["atom_j"])), max(int(row["atom_i"]), int(row["atom_j"])))
      for row in self.interactions
      if int(row["interaction_type"]) == int(interaction_type)
    }


def build_context(structure: "Structure", entities: Sequence[Any], largest_cutoff: float) -> InteractionContext:
  """Assemble the shared :class:`InteractionContext` for a structure.

  Parameters:
    structure: Single-model structure to analyze.
    entities: Entities participating in the analysis.
    largest_cutoff: Radius in Å to seed the distance cache with.

  Returns:
    A populated :class:`InteractionContext`.
  """
  annotations = structure.atom_annotations
  names = annotations.dtype.names

  atom_refs = []
  for idx in range(len(structure)):
    atom_refs.append(
      AtomReference(
        atom_index=idx,
        chain_id=str(annotations["chain_id"][idx]) if "chain_id" in names else "",
        res_id=int(annotations["res_id"][idx]) if "res_id" in names else 0,
        ins_code=str(annotations["ins_code"][idx]) if "ins_code" in names else "",
        res_name=str(annotations["res_name"][idx]) if "res_name" in names else "",
        atom_name=str(annotations["atom_name"][idx]) if "atom_name" in names else "",
        element=str(annotations["element"][idx]).strip().upper() if "element" in names else "",
      )
    )

  atom_to_entity_name: Dict[int, str] = {}
  atom_to_entity_idx: Dict[int, int] = {}
  for entity_idx, entity in enumerate(entities):
    for idx in entity.atom_indices:
      atom_to_entity_name[idx] = entity.name
      atom_to_entity_idx[idx] = entity_idx

  coords = np.column_stack([structure.atoms["x"], structure.atoms["y"], structure.atoms["z"]])

  pairwise_dist = None
  if len(structure) > 0:
    from scipy.spatial import KDTree

    pairwise_dist = LazyDistanceMatrix(coords, KDTree(coords), threshold=largest_cutoff)

  bonded_pairs: Set[Tuple[int, int]] = set()
  bonded_adjacency: Dict[int, List[int]] = {}
  for bond in structure.bonds:
    u, v = sorted((bond["atom_i"], bond["atom_j"]))
    bonded_pairs.add((u, v))
    bonded_adjacency.setdefault(u, []).append(v)
    bonded_adjacency.setdefault(v, []).append(u)

  atom_to_residue: Dict[int, Any] = {}
  n_terminal_residues: Set[Any] = set()
  for chain_view in structure.chains():
    residues = chain_view.residues()
    if residues:
      n_terminal_residues.add(residues[0])
    for residue in residues:
      for idx in residue.atom_indices():
        atom_to_residue[idx] = residue

  return InteractionContext(
    atom_refs=atom_refs,
    coords=coords,
    entities=list(entities),
    atom_annotations=annotations,
    bonds=structure.bonds,
    interactions=np.array(structure.interactions, copy=True),
    pairwise_dist=pairwise_dist,
    atom_to_entity_name=atom_to_entity_name,
    atom_to_entity_idx=atom_to_entity_idx,
    atom_to_residue=atom_to_residue,
    n_terminal_residues=n_terminal_residues,
    bonded_pairs=bonded_pairs,
    bonded_adjacency=bonded_adjacency,
    model_id=structure.metadata.get("model_id", 1),
  )


def atom_elements(structure: "Structure") -> List[str]:
  """Return normalized element symbols for every atom in a structure.

  Parameters:
    structure: Single-model structure to read annotations from.

  Returns:
    Uppercased element symbols in atom-index order, empty strings when the
    structure carries no element annotation.
  """
  if "element" not in structure.atom_annotations.dtype.names:
    return [""] * len(structure)
  return [str(value).strip().upper() for value in structure.atom_annotations["element"]]


def compute_largest_cutoff(
  ctx_atom_elements: Sequence[str],
  types_to_run: Sequence[str],
  *,
  contact_cutoff_a: float,
  vdw_tolerance_a: float,
  disulfide_cutoff: float,
  salt_bridge_cutoff: float,
  hbond_donor_acceptor_cutoff: float,
  metal_coordination_cutoff: float,
  covalent_candidates: bool,
  covalent_upper_factor: float,
) -> float:
  """Return the widest radius any enabled rule can require.

  Parameters:
    ctx_atom_elements: Element symbols for every atom in the structure.
    types_to_run: Interaction families that will be evaluated.
    contact_cutoff_a: Plain contact cutoff in Å.
    vdw_tolerance_a: Slack added to the van der Waals radius sum.
    disulfide_cutoff: Maximum SG-SG distance in Å.
    salt_bridge_cutoff: Maximum charge-center distance in Å.
    hbond_donor_acceptor_cutoff: Maximum donor-acceptor distance in Å.
    metal_coordination_cutoff: Maximum metal-donor distance in Å.
    covalent_candidates: Whether geometric covalent candidates are requested.
    covalent_upper_factor: Upper multiplier on the covalent radius sum.

  Returns:
    Radius in Å, defaulting to ``4.5`` when no rule constrains it.
  """
  largest = 0.0
  per_type = {
    "contact": contact_cutoff_a,
    "disulfide": disulfide_cutoff,
    "salt_bridge": salt_bridge_cutoff,
    "hydrogen_bond": hbond_donor_acceptor_cutoff,
    "metal_coordination": metal_coordination_cutoff,
  }
  for name, cutoff in per_type.items():
    if name in types_to_run:
      largest = max(largest, cutoff)

  if ctx_atom_elements:
    if "vdw_contact" in types_to_run or "clash" in types_to_run or "vdw_clash" in types_to_run:
      largest = max(largest, max(get_vdw_radius(element) for element in ctx_atom_elements) * 2 + vdw_tolerance_a)
    if "covalent" in types_to_run and covalent_candidates:
      radii = [radius for radius in (get_covalent_radius(element) for element in ctx_atom_elements) if radius is not None]
      if radii:
        largest = max(largest, max(radii) * 2 * covalent_upper_factor)

  return largest if largest > 0.0 else 4.5


def make_record(
  ctx: InteractionContext,
  interaction_type: str,
  evidence: str,
  idx1: int,
  idx2: int,
  *,
  distance_a: float,
  vdw_gap_a: Optional[float] = None,
  angle_deg: Optional[float] = None,
  role1: str = "",
  role2: str = "",
  details: Optional[dict] = None,
) -> InteractionRecord:
  """Build an :class:`InteractionRecord` from two atom indices.

  Endpoint metadata, entity names, model id, and rule versioning are copied from
  ``ctx`` so individual rules only supply what is specific to them. Endpoints
  are used in the order given; callers decide the ordering and matching roles.

  Parameters:
    ctx: Shared analysis context.
    interaction_type: Interaction family name.
    evidence: Evidence label for this observation.
    idx1: First endpoint atom index.
    idx2: Second endpoint atom index.
    distance_a: Endpoint separation in Å.
    vdw_gap_a: Van der Waals gap in Å. Defaults to the Bondi radius-sum gap.
    angle_deg: Interaction angle in degrees when the rule measures one.
    role1: Role label for the first endpoint.
    role2: Role label for the second endpoint.
    details: Rule-specific extras merged into the record.

  Returns:
    A populated record with an empty ``interaction_id``, assigned downstream.
  """
  ref1 = ctx.atom_refs[idx1]
  ref2 = ctx.atom_refs[idx2]
  if vdw_gap_a is None:
    vdw_gap_a = float(distance_a - (get_vdw_radius(ref1.element) + get_vdw_radius(ref2.element)))
  return InteractionRecord(
    interaction_id="",
    interaction_type=interaction_type,
    evidence=evidence,
    entity1=ctx.entity_name(idx1),
    atom_index1=idx1,
    chain1=ref1.chain_id,
    res_id1=ref1.res_id,
    ins_code1=ref1.ins_code,
    res_name1=ref1.res_name,
    atom_name1=ref1.atom_name,
    element1=ref1.element,
    role1=role1,
    entity2=ctx.entity_name(idx2),
    atom_index2=idx2,
    chain2=ref2.chain_id,
    res_id2=ref2.res_id,
    ins_code2=ref2.ins_code,
    res_name2=ref2.res_name,
    atom_name2=ref2.atom_name,
    element2=ref2.element,
    role2=role2,
    distance_a=float(distance_a),
    angle_deg=angle_deg,
    vdw_gap_a=float(vdw_gap_a),
    model_id=ctx.model_id,
    rule_set=RULE_SET,
    rule_version=RULE_VERSION,
    details=details if details is not None else {},
  )


def detect_disulfides(ctx: InteractionContext, *, disulfide_cutoff: float) -> List[InteractionRecord]:
  """Detect cysteine SG-SG disulfide bonds.

  Evidence is ``explicit`` for a declared disulfide connection, ``detected``
  when the structure carries an SG-SG bond, and ``candidate`` for geometry only.

  Parameters:
    ctx: Shared analysis context.
    disulfide_cutoff: Maximum SG-SG distance in Å.

  Returns:
    One record per qualifying SG pair.
  """
  if len(ctx) == 0:
    return []

  sg_indices = [
    idx for idx, ref in enumerate(ctx.atom_refs) if ref.res_name.strip().upper() in CYSTEINE_RESIDUE_NAMES and ref.atom_name.strip().upper() == "SG"
  ]
  declared_disulfides = ctx.declared_bond_pairs(BondType.DISULFIDE)
  bonded_sg_pairs = {
    (min(bond["atom_i"], bond["atom_j"]), max(bond["atom_i"], bond["atom_j"]))
    for bond in ctx.bonds
    if ctx.atom_refs[bond["atom_i"]].atom_name.strip().upper() == "SG" and ctx.atom_refs[bond["atom_j"]].atom_name.strip().upper() == "SG"
  }

  records = []
  for position, u in enumerate(sg_indices):
    for v in sg_indices[position + 1 :]:
      distance = ctx.pairwise_dist[u, v]
      if distance > disulfide_cutoff:
        continue
      pair = (min(u, v), max(u, v))
      if pair in declared_disulfides:
        evidence = "explicit"
      elif pair in bonded_sg_pairs:
        evidence = "detected"
      else:
        evidence = "candidate"
      records.append(
        make_record(
          ctx,
          "disulfide",
          evidence,
          u,
          v,
          distance_a=distance,
          vdw_gap_a=float(distance - DISULFIDE_REFERENCE_VDW_SUM),
        )
      )
  return records


def _charged_atom_indices(ctx: InteractionContext) -> Tuple[Set[int], Set[int]]:
  """Return positively and negatively charged atom indices.

  Canonical residues are typed from the ionic rule tables, histidine is gated on
  a positive formal charge annotation, and ligand entities are typed from the
  formal charges of their aligned RDKit molecule.

  Parameters:
    ctx: Shared analysis context.

  Returns:
    Tuple of ``(positive_indices, negative_indices)``.
  """
  positive: Set[int] = set()
  negative: Set[int] = set()
  has_charge = "charge" in ctx.atom_annotations.dtype.names

  for idx, ref in enumerate(ctx.atom_refs):
    res_name = ref.res_name.strip().upper()
    atom_name = ref.atom_name.strip().upper()
    if res_name not in CANONICAL_AMINO_ACIDS:
      continue
    if atom_name in PROTEIN_IONIC_POSITIVE.get(res_name, ()):
      positive.add(idx)
    elif res_name == "HIS" and atom_name in {"ND1", "NE2"}:
      if has_charge and ctx.atom_annotations["charge"][idx] > 0:
        positive.add(idx)
    elif atom_name in PROTEIN_IONIC_NEGATIVE.get(res_name, ()):
      negative.add(idx)

  for entity in ctx.entities:
    if not is_non_polymer_entity(entity, ctx.atom_annotations) or entity.rdkit_mol is None:
      continue
    for atom_id in range(entity.rdkit_mol.GetNumAtoms()):
      formal_charge = entity.rdkit_mol.GetAtomWithIdx(atom_id).GetFormalCharge()
      if formal_charge > 0:
        positive.add(entity.atom_indices[atom_id])
      elif formal_charge < 0:
        negative.add(entity.atom_indices[atom_id])

  return positive, negative


def detect_salt_bridges(ctx: InteractionContext, *, salt_bridge_cutoff: float) -> List[InteractionRecord]:
  """Detect ionic contacts between oppositely charged atoms.

  Evidence is ``explicit`` when the source file already declared the pair as a
  salt bridge or ionic interaction, and ``detected`` otherwise.

  Parameters:
    ctx: Shared analysis context.
    salt_bridge_cutoff: Maximum charge-center distance in Å.

  Returns:
    One record per qualifying opposite-charge pair.
  """
  if len(ctx) == 0:
    return []

  positive, negative = _charged_atom_indices(ctx)
  declared = ctx.declared_interaction_pairs(InteractionType.SALT_BRIDGE) | ctx.declared_interaction_pairs(InteractionType.IONIC)
  records = []
  for u in positive:
    for v in negative:
      distance = ctx.pairwise_dist[u, v]
      if distance > salt_bridge_cutoff:
        continue
      idx1, idx2 = sorted((u, v))
      records.append(
        make_record(
          ctx,
          "salt_bridge",
          "explicit" if (idx1, idx2) in declared else "detected",
          idx1,
          idx2,
          distance_a=distance,
          role1="positive" if idx1 == u else "negative",
          role2="negative" if idx1 == u else "positive",
        )
      )
  return records


def _hbond_donors_and_acceptors(ctx: InteractionContext) -> Tuple[Set[int], Set[int]]:
  """Return hydrogen-bond donor and acceptor atom indices.

  Canonical residues are typed from the sidechain rule tables plus the backbone
  amide and carbonyl, with proline donating only at a chain N-terminus. Water
  oxygens act as both. Ligand entities are typed by RDKit feature definitions.

  Parameters:
    ctx: Shared analysis context.

  Returns:
    Tuple of ``(donor_indices, acceptor_indices)``.
  """
  donors: Set[int] = set()
  acceptors: Set[int] = set()

  for idx, ref in enumerate(ctx.atom_refs):
    res_name = ref.res_name.strip().upper()
    atom_name = ref.atom_name.strip().upper()
    if res_name in CANONICAL_AMINO_ACIDS:
      if atom_name == "N":
        residue = ctx.atom_to_residue.get(idx)
        if res_name != "PRO" or (residue is not None and residue in ctx.n_terminal_residues):
          donors.add(idx)
      elif atom_name in PROTEIN_SIDECHAIN_DONORS.get(res_name, set()):
        donors.add(idx)

      if atom_name in BACKBONE_ACCEPTOR_ATOM_NAMES:
        acceptors.add(idx)
      elif atom_name in PROTEIN_SIDECHAIN_ACCEPTORS.get(res_name, set()):
        acceptors.add(idx)
    elif res_name in WATER_RESIDUE_NAMES and ref.element == "O":
      donors.add(idx)
      acceptors.add(idx)

  for entity in ctx.entities:
    if not is_non_polymer_entity(entity, ctx.atom_annotations) or entity.rdkit_mol is None:
      continue
    ligand_donors, ligand_acceptors = get_rdkit_donors_and_acceptors(entity.rdkit_mol)
    for atom_id in ligand_donors:
      donors.add(entity.atom_indices[atom_id])
    for atom_id in ligand_acceptors:
      acceptors.add(entity.atom_indices[atom_id])

  return donors, acceptors


def _attached_hydrogens(ctx: InteractionContext, donor_index: int) -> List[int]:
  """Return hydrogen atoms bonded to a donor.

  Explicit bonds are preferred. When the donor's residue carries no bonds at
  all, nearby hydrogens within :data:`IMPLICIT_HYDROGEN_MAX_DISTANCE` are used.

  Parameters:
    ctx: Shared analysis context.
    donor_index: Atom index of the donor.

  Returns:
    Atom indices of the attached hydrogens, empty when none can be assigned.
  """
  attached = [idx for idx in ctx.bonded_adjacency.get(donor_index, []) if ctx.atom_refs[idx].element in HBOND_HYDROGEN_ELEMENTS]
  if attached:
    return attached

  if ctx.atom_refs[donor_index].res_name.strip().upper() not in CANONICAL_AMINO_ACIDS:
    return attached
  residue = ctx.atom_to_residue.get(donor_index)
  if residue is None:
    return attached
  if any(idx in ctx.bonded_adjacency for idx in residue.atom_indices()):
    return attached

  for idx in residue.atom_indices():
    if idx == donor_index or ctx.atom_refs[idx].element not in HBOND_HYDROGEN_ELEMENTS:
      continue
    if ctx.pairwise_dist[donor_index, idx] <= IMPLICIT_HYDROGEN_MAX_DISTANCE:
      attached.append(idx)
  return attached


def detect_hydrogen_bonds(
  ctx: InteractionContext,
  *,
  hbond_donor_acceptor_cutoff: float,
  hbond_angle_cutoff: float,
  include_candidates: bool,
) -> List[InteractionRecord]:
  """Detect hydrogen bonds between typed donors and acceptors.

  When a hydrogen is present the donor-H-acceptor angle must reach
  ``hbond_angle_cutoff``. When no hydrogen can be assigned and
  ``include_candidates`` is set, a distance-only candidate is emitted instead.
  Evidence is raised to ``explicit`` when the source file already declared the
  pair as a hydrogen bond.

  Parameters:
    ctx: Shared analysis context.
    hbond_donor_acceptor_cutoff: Maximum donor-acceptor distance in Å.
    hbond_angle_cutoff: Minimum donor-H-acceptor angle in degrees.
    include_candidates: Whether to emit distance-only candidates.

  Returns:
    One record per qualifying donor-acceptor pair.
  """
  if len(ctx) == 0:
    return []

  donors, acceptors = _hbond_donors_and_acceptors(ctx)
  declared = ctx.declared_interaction_pairs(InteractionType.HYDROGEN_BOND)
  records = []
  for donor_index in donors:
    for acceptor_index in acceptors:
      if donor_index == acceptor_index:
        continue
      distance = ctx.pairwise_dist[donor_index, acceptor_index]
      if distance > hbond_donor_acceptor_cutoff:
        continue

      idx1, idx2 = sorted((donor_index, acceptor_index))
      role1 = "donor" if idx1 == donor_index else "acceptor"
      role2 = "acceptor" if idx1 == donor_index else "donor"

      hydrogens = _attached_hydrogens(ctx, donor_index)
      if hydrogens:
        for hydrogen_index in hydrogens:
          v_hd = ctx.coords[donor_index] - ctx.coords[hydrogen_index]
          v_ha = ctx.coords[acceptor_index] - ctx.coords[hydrogen_index]
          norm_hd = np.linalg.norm(v_hd)
          norm_ha = np.linalg.norm(v_ha)
          if norm_hd == 0 or norm_ha == 0:
            continue
          cos_theta = np.clip(np.dot(v_hd, v_ha) / (norm_hd * norm_ha), -1.0, 1.0)
          angle = float(np.degrees(np.arccos(cos_theta)))
          if angle < hbond_angle_cutoff:
            continue
          records.append(
            make_record(
              ctx,
              "hydrogen_bond",
              "explicit" if (idx1, idx2) in declared else "detected",
              idx1,
              idx2,
              distance_a=distance,
              angle_deg=angle,
              role1=role1,
              role2=role2,
              details={
                "hydrogen_index": hydrogen_index,
                "h_index": hydrogen_index,
                "donor_h_distance_a": float(norm_hd),
                "h_acceptor_distance_a": float(norm_ha),
              },
            )
          )
          break
      elif include_candidates:
        records.append(
          make_record(
            ctx,
            "hydrogen_bond",
            "candidate",
            idx1,
            idx2,
            distance_a=distance,
            angle_deg=None,
            role1=role1,
            role2=role2,
            details={"geometry": "distance_only"},
          )
        )
  return records


def detect_vdw_clashes(ctx: InteractionContext, *, clash_overlap_a: float) -> List[InteractionRecord]:
  """Detect non-bonded van der Waals overlaps across the whole structure.

  Unlike ``clash``, this rule is not restricted to cross-entity pairs; any
  non-bonded pair in the structure qualifies.

  Parameters:
    ctx: Shared analysis context.
    clash_overlap_a: Minimum overlap in Å reported as a clash. A pair qualifies
      when its van der Waals gap is below ``-clash_overlap_a``.

  Returns:
    One record per non-bonded pair that overlaps by more than ``clash_overlap_a``.
  """
  if len(ctx) == 0:
    return []

  # This rule queries its own radius rather than reusing the shared seed radius,
  # so its output does not depend on which other interaction types were requested.
  gap_cutoff = -clash_overlap_a
  max_radius_sum = 2 * max(get_vdw_radius(ref.element) for ref in ctx.atom_refs)
  search_radius = max_radius_sum + gap_cutoff
  if search_radius <= 0:
    return []

  pairs = ctx.pairwise_dist.tree.query_pairs(search_radius, output_type="ndarray")
  records = []
  for u, v in sorted((int(a), int(b)) for a, b in pairs):
    if (u, v) in ctx.bonded_pairs:
      continue
    distance = ctx.pairwise_dist[u, v]
    gap = distance - (get_vdw_radius(ctx.atom_refs[u].element) + get_vdw_radius(ctx.atom_refs[v].element))
    if gap < gap_cutoff:
      records.append(make_record(ctx, "vdw_clash", "vdw_overlap", u, v, distance_a=distance, vdw_gap_a=float(gap)))
  return records


def classify_coordination_geometry(metal_coord: np.ndarray, donor_coords: Sequence[np.ndarray]) -> Tuple[str, Optional[float]]:
  """Classify a metal coordination shell against idealized geometry templates.

  Parameters:
    metal_coord: Coordinate of the metal center.
    donor_coords: Coordinates of the coordinating donor atoms.

  Returns:
    Tuple of ``(geometry_name, rmsd_deviation_deg)``, or ``("unknown", None)``
    when the coordination number is unsupported or the shell is degenerate.
  """
  count = len(donor_coords)
  if count < MIN_GEOMETRY_DONORS or count > MAX_GEOMETRY_DONORS:
    return "unknown", None

  vectors = []
  for donor_coord in donor_coords:
    vector = donor_coord - metal_coord
    norm = np.linalg.norm(vector)
    if norm < 1e-5:
      return "unknown", None
    vectors.append(vector / norm)

  angles = []
  for i in range(count):
    for j in range(i + 1, count):
      angle = np.degrees(np.arccos(np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)))
      if angle < 5.0:  # Degeneracy check
        return "unknown", None
      angles.append(angle)
  sorted_angles = np.sort(angles)

  templates = {}
  if count == 2:
    templates["linear"] = np.array([180.0])
  elif count == 3:
    templates["trigonal planar"] = np.array([120.0, 120.0, 120.0])
  elif count == 4:
    ideal_tetrahedral = np.degrees(np.arccos(-1.0 / 3.0))
    templates["tetrahedral"] = np.array([ideal_tetrahedral] * 6)
    templates["square planar"] = np.sort(np.array([90.0, 90.0, 90.0, 90.0, 180.0, 180.0]))
  elif count == 5:
    templates["trigonal bipyramidal"] = np.sort(np.array([90.0] * 6 + [120.0] * 3 + [180.0]))
    templates["square pyramidal"] = np.sort(np.array([90.0] * 8 + [180.0] * 2))
  elif count == 6:
    templates["octahedral"] = np.sort(np.array([90.0] * 12 + [180.0] * 3))

  best_geometry = "unknown"
  min_rmsd = float("inf")
  for name, ideal_angles in templates.items():
    rmsd = float(np.sqrt(np.mean((sorted_angles - ideal_angles) ** 2)))
    if rmsd < min_rmsd:
      min_rmsd = rmsd
      best_geometry = name
  return best_geometry, min_rmsd


def detect_metal_coordination(
  ctx: InteractionContext,
  *,
  metal_coordination_cutoff: float,
  include_candidates: bool,
  emit_records: bool,
) -> Tuple[List[InteractionRecord], List[CoordinationCenterRecord]]:
  """Detect metal coordination shells and classify their geometry.

  Coordination centers are always returned so callers can report them even when
  ``metal_coordination`` records were not requested.

  Parameters:
    ctx: Shared analysis context.
    metal_coordination_cutoff: Maximum metal-donor distance in Å.
    include_candidates: Whether to keep untyped ``candidate`` donors.
    emit_records: Whether to also emit per-donor interaction records.

  Returns:
    Tuple of ``(records, coordination_centers)``.
  """
  if len(ctx) == 0:
    return [], []

  declared_coordination = ctx.declared_bond_pairs(BondType.METAL_COORDINATION)
  records: List[InteractionRecord] = []
  centers: List[CoordinationCenterRecord] = []

  for u, metal_ref in enumerate(ctx.atom_refs):
    if metal_ref.element not in METAL_ELEMENTS:
      continue

    donors = []
    for v, donor_ref in enumerate(ctx.atom_refs):
      if u == v:
        continue
      evidence = None
      distance = ctx.pairwise_dist[u, v]
      if (min(u, v), max(u, v)) in declared_coordination:
        evidence = "explicit"
      elif distance <= metal_coordination_cutoff and donor_ref.element in METAL_DONOR_ELEMENTS:
        donor_entity_idx = ctx.atom_to_entity_idx.get(v)
        is_typed_protein = donor_ref.res_name.strip().upper() in CANONICAL_AMINO_ACIDS
        is_aligned_ligand = (
          donor_entity_idx is not None
          and is_non_polymer_entity(ctx.entities[donor_entity_idx], ctx.atom_annotations)
          and ctx.entities[donor_entity_idx].rdkit_mol is not None
        )
        evidence = "detected" if (is_typed_protein or is_aligned_ligand) else "candidate"

      if evidence is None:
        continue
      if evidence == "candidate" and not include_candidates:
        continue
      donors.append({"atom_index": v, "element": donor_ref.element, "distance": distance, "evidence": evidence})

    if not donors:
      continue

    donors.sort(key=lambda donor: donor["atom_index"])
    donor_indices = [donor["atom_index"] for donor in donors]
    geometry, deviation = classify_coordination_geometry(ctx.coords[u], [ctx.coords[idx] for idx in donor_indices])

    evidence_labels = {donor["evidence"] for donor in donors}
    if "explicit" in evidence_labels:
      center_evidence = "explicit"
    elif "detected" in evidence_labels:
      center_evidence = "detected"
    else:
      center_evidence = "candidate"

    centers.append(
      CoordinationCenterRecord(
        center_id="",
        metal_atom_index=u,
        entity=ctx.entity_name(u),
        chain=metal_ref.chain_id,
        res_id=metal_ref.res_id,
        ins_code=metal_ref.ins_code,
        res_name=metal_ref.res_name,
        atom_name=metal_ref.atom_name,
        element=metal_ref.element,
        coordination_number=len(donors),
        donor_atom_indices=donor_indices,
        donor_elements=[donor["element"] for donor in donors],
        geometry=geometry,
        geometry_deviation_deg=deviation,
        evidence=center_evidence,
        model_id=ctx.model_id,
        rule_set=RULE_SET,
        rule_version=RULE_VERSION,
      )
    )

    if not emit_records:
      continue
    for donor in donors:
      v = donor["atom_index"]
      idx1, idx2 = sorted((u, v))
      records.append(
        make_record(
          ctx,
          "metal_coordination",
          donor["evidence"],
          idx1,
          idx2,
          distance_a=donor["distance"],
          role1="metal" if idx1 == u else "coordinating_donor",
          role2="coordinating_donor" if idx1 == u else "metal",
        )
      )

  return records, centers


def _entity_pair_crossings(ctx: InteractionContext, explicit_covalent: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
  """Group bonded and explicit covalent pairs that cross entity boundaries.

  Parameters:
    ctx: Shared analysis context.
    explicit_covalent: Atom pairs declared as covalent bonds by the source file.

  Returns:
    Map of sorted entity-index pair to the crossing atom pairs between them.
  """
  crossings: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
  for pairs in (ctx.bonded_pairs, explicit_covalent):
    for u, v in pairs:
      entity_u = ctx.atom_to_entity_idx.get(u)
      entity_v = ctx.atom_to_entity_idx.get(v)
      if entity_u is None or entity_v is None or entity_u == entity_v:
        continue
      key = (min(entity_u, entity_v), max(entity_u, entity_v))
      crossings.setdefault(key, set()).add((u, v))
  return crossings


def detect_cross_entity_candidates(
  ctx: InteractionContext,
  types_to_run: Sequence[str],
  *,
  contact_cutoff_a: float,
  vdw_tolerance_a: float,
  clash_overlap_a: float,
  include_hydrogens: bool,
  covalent_candidates: bool,
  covalent_lower_factor: float,
  covalent_upper_factor: float,
) -> List[InteractionRecord]:
  """Detect contacts, van der Waals contacts, clashes, and covalent bonds between entities.

  Only atom pairs whose endpoints sit in two different entities are considered.

  Parameters:
    ctx: Shared analysis context.
    types_to_run: Interaction families that will be evaluated.
    contact_cutoff_a: Plain contact cutoff in Å.
    vdw_tolerance_a: Slack added to the van der Waals radius sum.
    clash_overlap_a: Minimum overlap in Å reported as a clash.
    include_hydrogens: Whether hydrogen-involving contacts are kept.
    covalent_candidates: Whether to emit geometric covalent candidates.
    covalent_lower_factor: Lower multiplier on the covalent radius sum.
    covalent_upper_factor: Upper multiplier on the covalent radius sum.

  Returns:
    Records for every qualifying cross-entity pair.
  """
  run_contact = "contact" in types_to_run
  run_vdw = "vdw_contact" in types_to_run
  run_clash = "clash" in types_to_run
  run_covalent = "covalent" in types_to_run
  if len(ctx) == 0 or not (run_contact or run_vdw or run_clash or run_covalent):
    return []

  explicit_covalent = ctx.declared_bond_pairs(BondType.COVALENT)
  crossings_by_entity_pair = _entity_pair_crossings(ctx, explicit_covalent)

  entity_max_vdw = []
  entity_max_covalent = []
  for entity in ctx.entities:
    vdw_values = [VDW_RADII_BONDI.get(ctx.atom_refs[idx].element.title(), 0.0) for idx in entity.atom_indices]
    entity_max_vdw.append(max(vdw_values) if vdw_values else 0.0)
    covalent_values = [radius for radius in (get_covalent_radius(ctx.atom_refs[idx].element) for idx in entity.atom_indices) if radius is not None]
    entity_max_covalent.append(max(covalent_values) if covalent_values else 0.0)

  records: List[InteractionRecord] = []
  for i_entity, entity1 in enumerate(ctx.entities):
    indices1 = set(entity1.atom_indices)
    for j_entity in range(i_entity + 1, len(ctx.entities)):
      indices2 = set(ctx.entities[j_entity].atom_indices)
      crossings = crossings_by_entity_pair.get((i_entity, j_entity), set())

      if run_covalent:
        for u, v in sorted(crossings):
          distance = float(np.linalg.norm(ctx.coords[u] - ctx.coords[v]))
          records.append(make_record(ctx, "covalent", "explicit", u, v, distance_a=distance, role1="explicit", role2="explicit"))

      max_cutoff = 0.0
      if run_contact:
        max_cutoff = max(max_cutoff, contact_cutoff_a)
      if (run_vdw or run_clash) and entity_max_vdw[i_entity] > 0 and entity_max_vdw[j_entity] > 0:
        max_cutoff = max(max_cutoff, entity_max_vdw[i_entity] + entity_max_vdw[j_entity] + vdw_tolerance_a)
      if run_covalent and covalent_candidates and entity_max_covalent[i_entity] > 0 and entity_max_covalent[j_entity] > 0:
        max_cutoff = max(max_cutoff, (entity_max_covalent[i_entity] + entity_max_covalent[j_entity]) * covalent_upper_factor)
      if max_cutoff <= 0.0:
        continue

      candidates = [
        (u, v, distance)
        for (u, v), distance in ctx.pairwise_dist.seeded_items()
        if distance <= max_cutoff and ((u in indices1 and v in indices2) or (v in indices1 and u in indices2))
      ]
      candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))

      for u, v, distance in candidates:
        ref1 = ctx.atom_refs[u]
        ref2 = ctx.atom_refs[v]

        if run_contact and distance <= contact_cutoff_a:
          involves_hydrogen = ref1.element in CONTACT_HYDROGEN_ELEMENTS or ref2.element in CONTACT_HYDROGEN_ELEMENTS
          if include_hydrogens or not involves_hydrogen:
            records.append(make_record(ctx, "contact", "distance_cutoff", u, v, distance_a=distance))

        bondi1 = VDW_RADII_BONDI.get(ref1.element.title())
        bondi2 = VDW_RADII_BONDI.get(ref2.element.title())

        if run_vdw and bondi1 is not None and bondi2 is not None and distance <= bondi1 + bondi2 + vdw_tolerance_a:
          records.append(make_record(ctx, "vdw_contact", "vdw_contact", u, v, distance_a=distance, vdw_gap_a=float(distance - (bondi1 + bondi2))))

        if run_clash and bondi1 is not None and bondi2 is not None:
          overlap = bondi1 + bondi2 - distance
          if overlap >= clash_overlap_a:
            records.append(make_record(ctx, "clash", "vdw_overlap", u, v, distance_a=distance, vdw_gap_a=float(-overlap)))

        if run_covalent and covalent_candidates and (u, v) not in crossings:
          if ref1.element.upper() in METAL_ELEMENTS or ref2.element.upper() in METAL_ELEMENTS:
            continue
          covalent1 = get_covalent_radius(ref1.element)
          covalent2 = get_covalent_radius(ref2.element)
          if covalent1 is None or covalent2 is None:
            continue
          radius_sum = covalent1 + covalent2
          if radius_sum * covalent_lower_factor <= distance <= radius_sum * covalent_upper_factor:
            records.append(
              make_record(
                ctx,
                "covalent",
                "candidate",
                u,
                v,
                distance_a=distance,
                role1="candidate",
                role2="candidate",
                details={
                  "covalent_lower_factor": covalent_lower_factor,
                  "covalent_upper_factor": covalent_upper_factor,
                },
              )
            )
  return records


def deduplicate_records(records: Sequence[InteractionRecord], params: dict) -> List[InteractionRecord]:
  """Collapse records describing the same atom pair and interaction type.

  The strongest evidence wins, per :func:`evidence_priority`. Complementary
  details, measurements, and role labels from weaker duplicates are merged into
  the survivor, and the analysis parameters are attached to every record.

  Parameters:
    records: Records emitted by the individual rules, in rule order.
    params: Analysis parameters recorded under ``details["params"]``.

  Returns:
    Deduplicated records in order of first appearance.
  """
  import dataclasses

  grouped: Dict[Tuple[int, int, str], List[InteractionRecord]] = {}
  for record in records:
    key = (min(record.atom_index1, record.atom_index2), max(record.atom_index1, record.atom_index2), record.interaction_type)
    grouped.setdefault(key, []).append(record)

  deduped = []
  for group in grouped.values():
    group.sort(key=lambda record: evidence_priority(record.evidence), reverse=True)
    strongest = group[0]

    merged_details: Dict[str, Any] = {}
    for record in reversed(group):
      if record.details:
        merged_details.update(record.details)
    merged_details["params"] = params

    merged_fields: Dict[str, Any] = {}
    for name in ("distance_a", "angle_deg", "vdw_gap_a"):
      value = getattr(strongest, name)
      if value is None or (isinstance(value, float) and np.isnan(value)):
        for record in group:
          candidate = getattr(record, name)
          if candidate is not None and not (isinstance(candidate, float) and np.isnan(candidate)):
            merged_fields[name] = candidate
            break

    for name in ("role1", "role2"):
      if not getattr(strongest, name):
        for record in group:
          candidate = getattr(record, name)
          if candidate:
            merged_fields[name] = candidate
            break

    deduped.append(dataclasses.replace(strongest, details=merged_details, **merged_fields))
  return deduped
