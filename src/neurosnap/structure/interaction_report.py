"""Data models and reports for structural interactions and coordination centers."""

import json
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
import numpy as np
import pandas as pd

EXPECTED_INTERACTION_COLUMNS = [
  "interaction_id",
  "interaction_type",
  "evidence",
  "entity1",
  "atom_index1",
  "chain1",
  "res_id1",
  "ins_code1",
  "res_name1",
  "atom_name1",
  "element1",
  "role1",
  "entity2",
  "atom_index2",
  "chain2",
  "res_id2",
  "ins_code2",
  "res_name2",
  "atom_name2",
  "element2",
  "role2",
  "distance_a",
  "angle_deg",
  "vdw_gap_a",
  "source",
  "rule_set",
  "rule_version",
  "model_id",
  "details",
]

EXPECTED_COORDINATION_COLUMNS = [
  "center_id",
  "metal_atom_index",
  "entity",
  "chain",
  "res_id",
  "ins_code",
  "res_name",
  "atom_name",
  "element",
  "coordination_number",
  "donor_atom_indices",
  "donor_elements",
  "geometry",
  "geometry_deviation_deg",
  "evidence",
  "rule_set",
  "rule_version",
  "model_id",
]


class InteractionEntity:
  """Representation of a molecular entity participating in interactions."""

  def __init__(self, name: str, atom_indices: Sequence[int], rdkit_mol: Optional[Any] = None):
    self.name = name
    self.atom_indices = tuple(sorted(set(atom_indices)))
    self.rdkit_mol = rdkit_mol

  def __repr__(self) -> str:
    return f"InteractionEntity(name={self.name!r}, atom_indices={self.atom_indices!r})"


@dataclass(frozen=True)
class AtomReference:
  """A reference to a specific atom in a Structure with copied metadata."""

  atom_index: int
  chain_id: str
  res_id: int
  ins_code: str
  res_name: str
  atom_name: str
  element: str


@dataclass(frozen=True)
class InteractionRecord:
  """The exact fields representing a detected structural interaction."""

  interaction_id: str
  interaction_type: str
  evidence: str
  entity1: str
  atom_index1: int
  chain1: str
  res_id1: int
  ins_code1: str
  res_name1: str
  atom_name1: str
  element1: str
  role1: str
  entity2: str
  atom_index2: int
  chain2: str
  res_id2: int
  ins_code2: str
  res_name2: str
  atom_name2: str
  element2: str
  role2: str
  distance_a: Optional[float] = None
  angle_deg: Optional[float] = None
  vdw_gap_a: Optional[float] = None
  source: str = "geometric_rules"
  rule_set: str = "default"
  rule_version: str = "1"
  model_id: int = 1
  details: Optional[dict] = None


@dataclass(frozen=True)
class CoordinationCenterRecord:
  """The exact fields representing a metal coordination center."""

  center_id: str
  metal_atom_index: int
  entity: str
  chain: str
  res_id: int
  ins_code: str
  res_name: str
  atom_name: str
  element: str
  coordination_number: int
  donor_atom_indices: Sequence[int]
  donor_elements: Sequence[str]
  geometry: Optional[str] = None
  geometry_deviation_deg: Optional[float] = None
  evidence: str = "distance_cutoff"
  rule_set: str = "default"
  rule_version: str = "1"
  model_id: int = 1


class InteractionReport:
  """Immutable collection of structural interaction records and coordination centers."""

  def __init__(self, records: Sequence[InteractionRecord], coordination_centers: Sequence[CoordinationCenterRecord], metadata: Optional[dict] = None):
    self.metadata = metadata or {}
    # Sort interaction records by: interaction_type, atom_index1, atom_index2, role1, role2
    sorted_records = sorted(records, key=lambda r: (r.interaction_type or "", r.atom_index1 or 0, r.atom_index2 or 0, r.role1 or "", r.role2 or ""))
    # Re-assign interaction_id deterministically
    self.records = tuple(dataclasses.replace(r, interaction_id=f"int_{i + 1}") for i, r in enumerate(sorted_records))

    # Sort coordination centers by metal_atom_index
    sorted_coord = sorted(coordination_centers, key=lambda c: c.metal_atom_index or 0)
    # Re-assign center_id deterministically
    self.coordination_centers = tuple(dataclasses.replace(c, center_id=f"coord_{i + 1}") for i, c in enumerate(sorted_coord))

  def filter(
    self,
    *,
    interaction_types: Optional[Sequence[str]] = None,
    entities: Optional[Sequence[str]] = None,
    chains: Optional[Sequence[str]] = None,
    record_indices: Optional[Sequence[int]] = None,
    predicate: Optional[Callable[[InteractionRecord], bool]] = None,
  ) -> "InteractionReport":
    """Filter interactions and return a new InteractionReport with relevant coordination centers."""
    filtered_records = []
    itypes = set(interaction_types) if interaction_types is not None else None
    ents = set(entities) if entities is not None else None
    chns = set(chains) if chains is not None else None
    rec_indices = set(record_indices) if record_indices is not None else None

    for idx, rec in enumerate(self.records):
      if rec_indices is not None and idx not in rec_indices:
        continue
      if itypes is not None and rec.interaction_type not in itypes:
        continue
      if ents is not None and rec.entity1 not in ents and rec.entity2 not in ents:
        continue
      if chns is not None and rec.chain1 not in chns and rec.chain2 not in chns:
        continue
      if predicate is not None and not predicate(rec):
        continue
      filtered_records.append(rec)

    # Re-evaluate coordination centers: keep only those whose metal_atom_index is involved
    # in at least one of the remaining/filtered interactions, or matches the chain/entity filters.
    active_atom_indices = set()
    for rec in filtered_records:
      active_atom_indices.add(rec.atom_index1)
      active_atom_indices.add(rec.atom_index2)

    filtered_coord = []
    for cc in self.coordination_centers:
      if chns is not None and cc.chain not in chns:
        continue
      if ents is not None and cc.entity not in ents:
        continue
      # If we filtered by interaction types, predicate, or record indices, only keep if metal is involved
      if (itypes is not None or predicate is not None or rec_indices is not None) and cc.metal_atom_index not in active_atom_indices:
        continue
      filtered_coord.append(cc)

    # Return a new report, which will sort and re-index the IDs deterministically
    return InteractionReport(filtered_records, filtered_coord, metadata=self.metadata)

  def to_dataframe(self) -> pd.DataFrame:
    """Convert interaction records to a pandas DataFrame with the expected columns."""
    data = []
    for rec in self.records:
      data.append(
        {
          "interaction_id": rec.interaction_id,
          "interaction_type": rec.interaction_type,
          "evidence": rec.evidence,
          "entity1": rec.entity1,
          "atom_index1": rec.atom_index1,
          "chain1": rec.chain1,
          "res_id1": rec.res_id1,
          "ins_code1": rec.ins_code1,
          "res_name1": rec.res_name1,
          "atom_name1": rec.atom_name1,
          "element1": rec.element1,
          "role1": rec.role1,
          "entity2": rec.entity2,
          "atom_index2": rec.atom_index2,
          "chain2": rec.chain2,
          "res_id2": rec.res_id2,
          "ins_code2": rec.ins_code2,
          "res_name2": rec.res_name2,
          "atom_name2": rec.atom_name2,
          "element2": rec.element2,
          "role2": rec.role2,
          "distance_a": rec.distance_a if rec.distance_a is not None else np.nan,
          "angle_deg": rec.angle_deg if rec.angle_deg is not None else np.nan,
          "vdw_gap_a": rec.vdw_gap_a if rec.vdw_gap_a is not None else np.nan,
          "source": rec.source,
          "rule_set": rec.rule_set,
          "rule_version": rec.rule_version,
          "model_id": rec.model_id,
          "details": rec.details,
        }
      )
    frame = pd.DataFrame(data, columns=EXPECTED_INTERACTION_COLUMNS)
    if "rule_set" in frame.columns:
      frame["rule_set"] = frame["rule_set"].astype("string")
    if "rule_version" in frame.columns:
      frame["rule_version"] = frame["rule_version"].astype("string")
    return frame

  def coordination_centers_dataframe(self) -> pd.DataFrame:
    """Convert coordination centers to a pandas DataFrame with the expected columns."""
    data = []
    for rec in self.coordination_centers:
      data.append(
        {
          "center_id": rec.center_id,
          "metal_atom_index": rec.metal_atom_index,
          "entity": rec.entity,
          "chain": rec.chain,
          "res_id": rec.res_id,
          "ins_code": rec.ins_code,
          "res_name": rec.res_name,
          "atom_name": rec.atom_name,
          "element": rec.element,
          "coordination_number": rec.coordination_number,
          "donor_atom_indices": rec.donor_atom_indices,
          "donor_elements": rec.donor_elements,
          "geometry": rec.geometry,
          "geometry_deviation_deg": rec.geometry_deviation_deg if rec.geometry_deviation_deg is not None else np.nan,
          "evidence": rec.evidence,
          "rule_set": rec.rule_set,
          "rule_version": rec.rule_version,
          "model_id": rec.model_id,
        }
      )
    frame = pd.DataFrame(data, columns=EXPECTED_COORDINATION_COLUMNS)
    if "rule_set" in frame.columns:
      frame["rule_set"] = frame["rule_set"].astype("string")
    if "rule_version" in frame.columns:
      frame["rule_version"] = frame["rule_version"].astype("string")
    return frame

  def to_csv(self, **kwargs) -> str:
    """Export the interaction records DataFrame as a CSV string."""
    return self.to_dataframe().to_csv(index=False, **kwargs)

  def to_json(self, indent: Optional[int] = None) -> str:
    """Export the report in record-oriented JSON format with deterministic key order."""

    def clean_val(v):
      if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
      return v

    def serialize_details(d):
      if d is None:
        return None
      # Sort keys deterministically
      return {k: clean_val(d[k]) for k in sorted(d.keys())}

    interactions_list = []
    for rec in self.records:
      interactions_list.append(
        {
          "interaction_id": rec.interaction_id,
          "interaction_type": rec.interaction_type,
          "evidence": rec.evidence,
          "entity1": rec.entity1,
          "atom_index1": rec.atom_index1,
          "chain1": rec.chain1,
          "res_id1": rec.res_id1,
          "ins_code1": rec.ins_code1,
          "res_name1": rec.res_name1,
          "atom_name1": rec.atom_name1,
          "element1": rec.element1,
          "role1": rec.role1,
          "entity2": rec.entity2,
          "atom_index2": rec.atom_index2,
          "chain2": rec.chain2,
          "res_id2": rec.res_id2,
          "ins_code2": rec.ins_code2,
          "res_name2": rec.res_name2,
          "atom_name2": rec.atom_name2,
          "element2": rec.element2,
          "role2": rec.role2,
          "distance_a": clean_val(rec.distance_a),
          "angle_deg": clean_val(rec.angle_deg),
          "vdw_gap_a": clean_val(rec.vdw_gap_a),
          "source": rec.source,
          "rule_set": rec.rule_set,
          "rule_version": rec.rule_version,
          "model_id": rec.model_id,
          "details": serialize_details(rec.details),
        }
      )

    coordination_list = []
    for rec in self.coordination_centers:
      coordination_list.append(
        {
          "center_id": rec.center_id,
          "metal_atom_index": rec.metal_atom_index,
          "entity": rec.entity,
          "chain": rec.chain,
          "res_id": rec.res_id,
          "ins_code": rec.ins_code,
          "res_name": rec.res_name,
          "atom_name": rec.atom_name,
          "element": rec.element,
          "coordination_number": rec.coordination_number,
          "donor_atom_indices": list(rec.donor_atom_indices),
          "donor_elements": list(rec.donor_elements),
          "geometry": rec.geometry,
          "geometry_deviation_deg": clean_val(rec.geometry_deviation_deg),
          "evidence": rec.evidence,
          "rule_set": rec.rule_set,
          "rule_version": rec.rule_version,
          "model_id": rec.model_id,
        }
      )

    data = {"interactions": interactions_list, "coordination_centers": coordination_list}
    return json.dumps(data, indent=indent)
