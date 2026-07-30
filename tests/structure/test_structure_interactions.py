import pytest
import numpy as np
import pandas as pd
import json
from io import StringIO

from neurosnap.structure import BondType
from tests._structure_test_utils import make_structure

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


def test_interaction_dataframe_columns():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")])
  df = struct.detect_interactions()
  assert list(df.columns) == EXPECTED_INTERACTION_COLUMNS


def test_coordination_dataframe_columns():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")])
  df = struct.detect_coordination_centers()
  assert list(df.columns) == EXPECTED_COORDINATION_COLUMNS


def test_stable_ordering():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CB", "ALA", "A", 1, 1.0, 0.0, 0.0, "C")])
  df1 = struct.detect_interactions()
  df2 = struct.detect_interactions()
  assert df1.equals(df2)
  # verify sorting is by interaction type, atom indices, then roles
  assert list(df1.sort_values(["interaction_type", "atom_index1", "atom_index2", "role1", "role2"]).index) == list(df1.index)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  string_cols = [
    "interaction_id",
    "interaction_type",
    "evidence",
    "entity1",
    "chain1",
    "ins_code1",
    "res_name1",
    "atom_name1",
    "element1",
    "role1",
    "entity2",
    "chain2",
    "ins_code2",
    "res_name2",
    "atom_name2",
    "element2",
    "role2",
    "source",
    "rule_set",
    "rule_version",
    "center_id",
    "entity",
    "chain",
    "res_name",
    "atom_name",
    "element",
    "geometry",
  ]
  for col in string_cols:
    if col in df.columns:
      df[col] = df[col].fillna("").astype(str)
  return df


def test_json_csv_round_trip():
  # Create a structure that generates a contact/interaction
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C")])
  df = struct.detect_interactions()

  # Ensure there's a row to test scalar values and details
  assert len(df) > 0, "Expected at least one interaction to test serialization"

  # JSON round-trip
  json_str = df.to_json(orient="records")
  df_json = pd.read_json(StringIO(json_str), orient="records")
  pd.testing.assert_frame_equal(normalize_dataframe(df), normalize_dataframe(df_json), check_dtype=False)

  # CSV round-trip
  df_csv_out = df.copy()
  if "details" in df_csv_out.columns:
    df_csv_out["details"] = df_csv_out["details"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
  csv_str = df_csv_out.to_csv(index=False)

  df_csv_in = pd.read_csv(StringIO(csv_str))
  if "details" in df_csv_in.columns:
    df_csv_in["details"] = df_csv_in["details"].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("{") else x)
  pd.testing.assert_frame_equal(normalize_dataframe(df), normalize_dataframe(df_csv_in), check_dtype=False)


def test_d_h_a_geometry():
  # D=(0,0,0), H=(0,0,1), A=(0,0,2.8) passes
  struct_pass = make_structure(
    [
      ("N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),  # Donor
      ("H", "ALA", "A", 1, 0.0, 0.0, 1.0, "H"),  # Hydrogen
      ("O", "GLY", "B", 2, 0.0, 0.0, 2.8, "O"),  # Acceptor
    ]
  )
  df_pass = struct_pass.detect_interactions()
  assert len(df_pass) > 0
  assert "hydrogen_bond" in df_pass["interaction_type"].values

  # H=(0,0,-1) fails the angle rule
  struct_fail = make_structure(
    [("N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"), ("H", "ALA", "A", 1, 0.0, 0.0, -1.0, "H"), ("O", "GLY", "B", 2, 0.0, 0.0, 2.8, "O")]
  )
  df_fail = struct_fail.detect_interactions()
  if len(df_fail) > 0:
    assert "hydrogen_bond" not in df_fail["interaction_type"].values


def test_inter_chain_cys_sg_pair():
  struct = make_structure([("SG", "CYS", "A", 1, 0.0, 0.0, 0.0, "S"), ("SG", "CYS", "B", 1, 2.03, 0.0, 0.0, "S")])
  df = struct.detect_interactions()
  assert len(df) > 0
  assert "disulfide" in df["interaction_type"].values


def test_salt_bridge_distance_logic():
  # LYS-NZ and ASP-OD1/OD2 while their C-alpha atoms are farther than the ionic cutoff
  struct = make_structure(
    [
      ("CA", "LYS", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("NZ", "LYS", "A", 1, 5.0, 0.0, 0.0, "N"),
      ("CA", "ASP", "B", 2, 12.0, 0.0, 0.0, "C"),
      ("OD1", "ASP", "B", 2, 7.0, 0.0, 0.0, "O"),
    ]
  )
  df = struct.detect_interactions()
  assert len(df) > 0
  assert "salt_bridge" in df["interaction_type"].values


def test_raw_contacts():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C")])
  df = struct.get_raw_contacts()
  assert len(df) == 1
  assert df.iloc[0]["distance_a"] == pytest.approx(1.5)
  assert df.iloc[0]["atom_index1"] == 0
  assert df.iloc[0]["atom_index2"] == 1


def test_entity_validation_bounds():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")], entity_ids=[999])
  with pytest.raises(ValueError, match="bounds"):
    struct.validate_entities()


def test_entity_validation_duplicate_indices():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CB", "ALA", "A", 1, 1.0, 0.0, 0.0, "C")])
  struct.atom_annotations["atom_id"][1] = struct.atom_annotations["atom_id"][0]
  with pytest.raises(ValueError, match="duplicate"):
    struct.validate_entities()


def test_entity_validation_rdkit_mismatch():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")])
  with pytest.raises(ValueError, match="mismatch"):
    struct.validate_entities(check_rdkit=True)


def test_entity_validation_overlapping_entities():
  struct = make_structure(
    [("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "GLY", "B", 2, 1.0, 0.0, 0.0, "C"), ("CB", "ALA", "A", 1, 2.0, 0.0, 0.0, "C")],
    entity_ids=[1, 2, 1],
  )
  with pytest.raises(ValueError, match="overlap|interleave|continuous|contiguous"):
    struct.validate_entities()


def test_entity_validation_unsupported_interaction():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")])
  with pytest.raises(ValueError, match="unsupported"):
    struct.detect_interactions(interaction_types=["magic_bond"])


def test_entity_validation_non_finite_coordinates():
  struct = make_structure([("CA", "ALA", "A", 1, np.nan, 0.0, 0.0, "C")])
  with pytest.raises(ValueError, match="finite"):
    struct.validate_entities()


def test_missing_ligand_topology():
  struct = make_structure([("C1", "LIG", "A", 1, 0.0, 0.0, 0.0, "C"), ("C2", "LIG", "A", 1, 1.5, 0.0, 0.0, "C")], hetero=[True, True])
  with pytest.raises(ValueError):
    struct.detect_interactions()

  # Requesting only distance-based interactions should succeed without requiring topology
  df_distance = struct.detect_interactions(interaction_types=["vdw_clash"])
  assert isinstance(df_distance, pd.DataFrame)

  # Raw contacts should still work
  df = struct.get_raw_contacts()
  assert len(df) > 0


def test_empty_results_return_empty_tables():
  struct = make_structure([])
  df_interactions = struct.detect_interactions()
  assert len(df_interactions) == 0
  assert list(df_interactions.columns) == EXPECTED_INTERACTION_COLUMNS

  df_coord = struct.detect_coordination_centers()
  assert len(df_coord) == 0
  assert list(df_coord.columns) == EXPECTED_COORDINATION_COLUMNS


def test_empty_results_with_atoms_returns_empty_tables():
  # Structure has atoms, but they are too far apart for interactions
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "ALA", "B", 2, 100.0, 0.0, 0.0, "C")])
  df_interactions = struct.detect_interactions()
  assert len(df_interactions) == 0
  assert list(df_interactions.columns) == EXPECTED_INTERACTION_COLUMNS

  df_coord = struct.detect_coordination_centers()
  assert len(df_coord) == 0
  assert list(df_coord.columns) == EXPECTED_COORDINATION_COLUMNS


def test_interaction_report_filter():
  struct = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
      ("SG", "CYS", "B", 2, 10.0, 10.0, 10.0, "S"),
      ("SG", "CYS", "B", 3, 11.2, 10.0, 10.0, "S"),  # Disulfide candidate (1.2 A)
    ]
  )
  report = struct._analyze_interactions()

  # Ensure we have at least 2 interactions (one vdw_clash, one disulfide)
  assert len(report.records) >= 2

  # Filter by record_indices
  filtered_indices = report.filter(record_indices=[0])
  assert len(filtered_indices.records) == 1

  # Filter by interaction_types
  filtered_types = report.filter(interaction_types=["disulfide"])
  assert len(filtered_types.records) > 0
  for r in filtered_types.records:
    assert r.interaction_type == "disulfide"

  # Filter by predicate
  filtered_pred = report.filter(predicate=lambda r: r.interaction_type == "vdw_clash")
  assert len(filtered_pred.records) > 0
  for r in filtered_pred.records:
    assert r.interaction_type == "vdw_clash"


def test_interaction_report_direct_serialization():
  from neurosnap.structure.interaction_report import InteractionReport, InteractionRecord, CoordinationCenterRecord

  rec = InteractionRecord(
    interaction_id="",
    interaction_type="vdw_clash",
    evidence="distance_cutoff",
    entity1="Chain_A",
    atom_index1=1,
    chain1="A",
    res_id1=1,
    ins_code1="",
    res_name1="ALA",
    atom_name1="CA",
    element1="C",
    role1="clash",
    entity2="Chain_A",
    atom_index2=2,
    chain2="A",
    res_id2=1,
    ins_code2="",
    res_name2="ALA",
    atom_name2="CB",
    element2="C",
    role2="clash",
    distance_a=1.5,
    details={"key": "val"},
  )

  cc = CoordinationCenterRecord(
    center_id="",
    metal_atom_index=3,
    entity="Chain_A",
    chain="A",
    res_id=2,
    ins_code="",
    res_name="ZN",
    atom_name="ZN",
    element="ZN",
    coordination_number=1,
    donor_atom_indices=[1],
    donor_elements=["C"],
  )

  report = InteractionReport([rec], [cc])

  # Test to_json() on InteractionReport directly
  json_str = report.to_json()
  parsed = json.loads(json_str)
  assert "interactions" in parsed
  assert "coordination_centers" in parsed
  assert parsed["interactions"][0]["interaction_type"] == "vdw_clash"
  assert parsed["interactions"][0]["details"] == {"key": "val"}
  assert parsed["coordination_centers"][0]["metal_atom_index"] == 3

  # Test to_csv() on InteractionReport directly
  csv_str = report.to_csv()
  assert "vdw_clash" in csv_str
  assert "Chain_A" in csv_str


def test_interaction_report_filter_coordination_rec_indices():
  from neurosnap.structure.interaction_report import InteractionReport, InteractionRecord, CoordinationCenterRecord

  rec1 = InteractionRecord(
    interaction_id="",
    interaction_type="vdw_clash",
    evidence="distance_cutoff",
    entity1="Chain_A",
    atom_index1=1,
    chain1="A",
    res_id1=1,
    ins_code1="",
    res_name1="ALA",
    atom_name1="CA",
    element1="C",
    role1="clash",
    entity2="Chain_A",
    atom_index2=2,
    chain2="A",
    res_id2=1,
    ins_code2="",
    res_name2="ALA",
    atom_name2="CB",
    element2="C",
    role2="clash",
    distance_a=1.5,
  )

  rec2 = InteractionRecord(
    interaction_id="",
    interaction_type="metal_coordination",
    evidence="distance_cutoff",
    entity1="Chain_A",
    atom_index1=3,
    chain1="A",
    res_id1=2,
    ins_code1="",
    res_name1="ZN",
    atom_name1="ZN",
    element1="ZN",
    role1="metal",
    entity2="Chain_A",
    atom_index2=4,
    chain2="A",
    res_id2=3,
    ins_code2="",
    res_name2="HIS",
    atom_name2="NE2",
    element2="N",
    role2="donor",
    distance_a=2.0,
  )

  cc = CoordinationCenterRecord(
    center_id="",
    metal_atom_index=3,
    entity="Chain_A",
    chain="A",
    res_id=2,
    ins_code="",
    res_name="ZN",
    atom_name="ZN",
    element="ZN",
    coordination_number=1,
    donor_atom_indices=[4],
    donor_elements=["N"],
  )

  report = InteractionReport([rec1, rec2], [cc])
  assert len(report.records) == 2
  assert len(report.coordination_centers) == 1

  # Filter by rec_indices to keep only rec1 (vdw_clash, which does NOT involve the metal atom 3)
  # sorted_records sorts by (interaction_type, atom_index1, atom_index2, role1, role2)
  # "metal_coordination" comes before "vdw_clash" alphabetically.
  # So rec2 (metal_coordination) will be index 0, and rec1 (vdw_clash) will be index 1.
  # Let's keep only rec1 (index 1) which does NOT involve metal atom 3.
  filtered = report.filter(record_indices=[1])
  assert len(filtered.records) == 1
  assert filtered.records[0].interaction_type == "vdw_clash"
  # Since the metal coordination is filtered out and the remaining record doesn't involve ZN (atom 3),
  # the coordination center CC (with metal_atom_index=3) should also be filtered out.
  assert len(filtered.coordination_centers) == 0

  # If we filter to keep rec2 (index 0), which does involve the metal, coordination center CC should be kept.
  filtered_keep = report.filter(record_indices=[0])
  assert len(filtered_keep.records) == 1
  assert filtered_keep.records[0].interaction_type == "metal_coordination"
  assert len(filtered_keep.coordination_centers) == 1


def test_spatial_contacts_vdw_clash():
  struct1 = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "ALA", "B", 1, 0.0, 0.0, 4.2, "C")])
  df1 = struct1.detect_interactions(interaction_types=["contact"])
  assert len(df1) == 1
  assert df1.iloc[0]["interaction_type"] == "contact"
  assert df1.iloc[0]["distance_a"] == pytest.approx(4.2)

  struct2 = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "ALA", "B", 1, 0.0, 0.0, 3.6, "C")])
  df2 = struct2.detect_interactions(interaction_types=["vdw_contact"])
  assert len(df2) == 1
  assert df2.iloc[0]["interaction_type"] == "vdw_contact"
  assert df2.iloc[0]["distance_a"] == pytest.approx(3.6)

  struct3 = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "ALA", "B", 1, 0.0, 0.0, 2.9, "C")])
  df3 = struct3.detect_interactions(interaction_types=["clash"])
  assert len(df3) == 1
  assert df3.iloc[0]["interaction_type"] == "clash"
  assert df3.iloc[0]["distance_a"] == pytest.approx(2.9)


def test_spatial_contacts_hydrogens():
  struct = make_structure([("H", "ALA", "A", 1, 0.0, 0.0, 0.0, "H"), ("H", "ALA", "B", 1, 0.0, 0.0, 2.0, "H")])
  df_no_h = struct.detect_interactions(interaction_types=["contact"], include_hydrogens=False)
  assert len(df_no_h) == 0

  df_with_h = struct.detect_interactions(interaction_types=["contact"], include_hydrogens=True)
  assert len(df_with_h) == 1
  assert df_with_h.iloc[0]["interaction_type"] == "contact"


def test_spatial_contacts_covalent():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "ALA", "B", 1, 0.0, 0.0, 1.5, "C")])
  struct.bonds = np.array([(0, 1, 1, int(BondType.COVALENT))], dtype=struct._dtype_bond)

  df_explicit = struct.detect_interactions(interaction_types=["covalent"], covalent_candidates=False)
  assert len(df_explicit) == 1
  assert df_explicit.iloc[0]["interaction_type"] == "covalent"
  assert df_explicit.iloc[0]["role1"] == "explicit"
  assert df_explicit.iloc[0]["evidence"] == "explicit"

  struct_cand = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "ALA", "B", 1, 0.0, 0.0, 1.5, "C")])
  df_cand = struct_cand.detect_interactions(interaction_types=["covalent"], covalent_candidates=True)
  assert len(df_cand) == 1
  assert df_cand.iloc[0]["interaction_type"] == "covalent"
  assert df_cand.iloc[0]["role1"] == "candidate"
  assert df_cand.iloc[0]["evidence"] == "candidate"
  assert df_cand.iloc[0]["details"]["covalent_lower_factor"] == 0.8


def test_non_finite_coordinates_rejection_candidate_engine(caplog):
  from neurosnap.structure.interactions import _find_neighbor_candidates
  import logging

  coords = np.array([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
  indices1 = np.array([0], dtype=np.int32)
  indices2 = np.array([1], dtype=np.int32)
  with caplog.at_level(logging.WARNING):
    res = _find_neighbor_candidates(coords, indices1, indices2, 4.5)
  assert len(res) == 0
  assert any("Non-finite coordinates detected" in record.message for record in caplog.records)


def test_ckdtree_brute_force_comparison():
  np.random.seed(42)
  size1 = 100
  size2 = 120
  coords1 = np.random.uniform(-10.0, 10.0, (size1, 3)).astype(np.float32)
  coords2 = np.random.uniform(-10.0, 10.0, (size2, 3)).astype(np.float32)

  coords = np.vstack([coords1, coords2])
  indices1 = np.arange(size1, dtype=np.int32)
  indices2 = np.arange(size1, size1 + size2, dtype=np.int32)
  cutoff = 3.5

  from neurosnap.structure.interactions import _find_neighbor_candidates

  ckd_results = _find_neighbor_candidates(coords, indices1, indices2, cutoff)

  bf_results = []
  for i in indices1:
    for j in indices2:
      dist = float(np.linalg.norm(coords[i] - coords[j]))
      if dist <= cutoff:
        u, v = sorted((i, j))
        bf_results.append((u, v, dist))
  bf_results.sort(key=lambda x: (x[0], x[1]))

  assert len(ckd_results) == len(bf_results)
  for r1, r2 in zip(ckd_results, bf_results):
    assert r1[0] == r2[0]
    assert r1[1] == r2[1]
    assert r1[2] == pytest.approx(r2[2])


def test_disulfide_evidence_categories():
  # 1. Geometry-only CYS SG pair -> candidate
  struct_cand = make_structure([("SG", "CYS", "A", 1, 0.0, 0.0, 0.0, "S"), ("SG", "CYS", "B", 1, 2.1, 0.0, 0.0, "S")])
  df = struct_cand.detect_interactions(interaction_types=["disulfide"])
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "candidate"

  # 2. CYS SG pair with covalent bond in bonds -> detected
  struct_det = make_structure([("SG", "CYS", "A", 1, 0.0, 0.0, 0.0, "S"), ("SG", "CYS", "B", 1, 2.1, 0.0, 0.0, "S")])
  struct_det.bonds = np.array([(0, 1, 1, int(BondType.COVALENT))], dtype=struct_det._dtype_bond)
  df = struct_det.detect_interactions(interaction_types=["disulfide"])
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "detected"

  # 3. CYS SG pair with explicit disulfide connection -> explicit
  struct_exp = make_structure([("SG", "CYS", "A", 1, 0.0, 0.0, 0.0, "S"), ("SG", "CYS", "B", 1, 2.1, 0.0, 0.0, "S")])
  struct_exp.bonds = np.array([(0, 1, 1, int(BondType.DISULFIDE))], dtype=struct_exp._dtype_bond)
  df = struct_exp.detect_interactions(interaction_types=["disulfide"])
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "explicit"


def test_hbond_candidates_and_detected():
  # 1. With hydrogen coordinates -> detected
  struct_det = make_structure(
    [
      ("N", "SER", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("H", "SER", "A", 1, 0.0, 0.0, 1.0, "H"),
      ("O", "ASP", "B", 2, 0.0, 0.0, 2.8, "O"),
    ]
  )
  df = struct_det.detect_interactions(interaction_types=["hydrogen_bond"], include_candidates=False)
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "detected"
  assert df.iloc[0]["angle_deg"] == pytest.approx(180.0)
  assert df.iloc[0]["details"]["hydrogen_index"] == 1

  # 2. No hydrogen coordinates, include_candidates=False -> no rows
  struct_no_h = make_structure(
    [
      ("N", "SER", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("O", "ASP", "B", 2, 0.0, 0.0, 2.8, "O"),
    ]
  )
  df = struct_no_h.detect_interactions(interaction_types=["hydrogen_bond"], include_candidates=False)
  assert len(df) == 0

  # 3. No hydrogen coordinates, include_candidates=True -> candidate
  df = struct_no_h.detect_interactions(interaction_types=["hydrogen_bond"], include_candidates=True)
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "candidate"
  assert pd.isna(df.iloc[0]["angle_deg"]) or df.iloc[0]["angle_deg"] is None
  assert df.iloc[0]["details"]["geometry"] == "distance_only"


def test_ionic_contacts_protein_and_ligand():
  # 1. Protein-protein ionic contact (LYS NZ and ASP OD1) -> detected
  struct_prot = make_structure(
    [
      ("NZ", "LYS", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("OD1", "ASP", "B", 2, 3.8, 0.0, 0.0, "O"),
    ]
  )
  df = struct_prot.detect_interactions(interaction_types=["salt_bridge"])
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "detected"
  assert df.iloc[0]["role1"] == "positive"
  assert df.iloc[0]["role2"] == "negative"

  # 2. HIS ionic contact sensitive to positive charge
  # HIS ND1 without positive charge -> no rows
  struct_his = make_structure(
    [
      ("ND1", "HIS", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("OD1", "ASP", "B", 2, 3.8, 0.0, 0.0, "O"),
    ]
  )
  df = struct_his.detect_interactions(interaction_types=["salt_bridge"])
  assert len(df) == 0

  # HIS ND1 with positive charge -> detected
  struct_his_pos = make_structure(
    [
      ("ND1", "HIS", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("OD1", "ASP", "B", 2, 3.8, 0.0, 0.0, "O"),
    ]
  )
  struct_his_pos.add_annotation("charge", "i1", np.array([1, 0], dtype=np.int8))

  df = struct_his_pos.detect_interactions(interaction_types=["salt_bridge"])
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "detected"
  assert df.iloc[0]["role1"] == "positive"

  # 3. Protein-ligand ionic contact
  from rdkit import Chem
  from neurosnap.structure.interaction_report import InteractionEntity

  # Create a dummy structure with a protein LYS NZ and a ligand atom with formal charge -1
  struct_lig = make_structure(
    [
      ("NZ", "LYS", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("O1", "LIG", "B", 2, 3.8, 0.0, 0.0, "O"),
    ],
    hetero=[False, True],
  )

  # Set RDKit molecule for the ligand entity
  mol = Chem.MolFromSmiles("[O-]")
  assert mol is not None
  struct_lig.entities = [InteractionEntity(name="Chain_A", atom_indices=[0]), InteractionEntity(name="Ligand_B", atom_indices=[1], rdkit_mol=mol)]
  df = struct_lig.detect_interactions(interaction_types=["salt_bridge"])
  assert len(df) == 1
  assert df.iloc[0]["evidence"] == "detected"
  assert df.iloc[0]["role1"] == "positive"
  assert df.iloc[0]["role2"] == "negative"


def test_metal_coordination_geometries():
  # 1. Linear (Ideal)
  # Metal at 0,0,0; Donors at 0,0,2.0 and 0,0,-2.0
  struct_linear = make_structure(
    [("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"), ("O", "ALA", "A", 2, 0.0, 0.0, 2.0, "O"), ("O", "ALA", "A", 3, 0.0, 0.0, -2.0, "O")]
  )
  df = struct_linear.detect_coordination_centers()
  assert len(df) == 1
  assert df.iloc[0]["coordination_number"] == 2
  assert df.iloc[0]["geometry"] == "linear"
  assert df.iloc[0]["geometry_deviation_deg"] == pytest.approx(0.0, abs=1e-3)

  # 2. Linear (Distorted)
  struct_linear_dist = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 0.0, 0.0, 2.0, "O"),
      ("O", "ALA", "A", 3, 0.1, 0.0, -2.0, "O"),  # slightly tilted
    ]
  )
  df = struct_linear_dist.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "linear"
  assert df.iloc[0]["geometry_deviation_deg"] > 0.1
  assert df.iloc[0]["geometry_deviation_deg"] < 10.0

  # 3. Trigonal Planar (Ideal)
  struct_trig = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 3, -1.0, np.sqrt(3), 0.0, "O"),
      ("O", "ALA", "A", 4, -1.0, -np.sqrt(3), 0.0, "O"),
    ]
  )
  df = struct_trig.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "trigonal planar"
  assert df.iloc[0]["geometry_deviation_deg"] == pytest.approx(0.0, abs=1e-3)

  # 4. Tetrahedral (Ideal)
  # Ideal tetrahedral vectors normalized to distance 2.0
  s = 2.0 / np.sqrt(3.0)
  struct_tet = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, s, s, s, "O"),
      ("O", "ALA", "A", 3, -s, -s, s, "O"),
      ("O", "ALA", "A", 4, -s, s, -s, "O"),
      ("O", "ALA", "A", 5, s, -s, -s, "O"),
    ]
  )
  df = struct_tet.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "tetrahedral"
  assert df.iloc[0]["geometry_deviation_deg"] == pytest.approx(0.0, abs=1e-3)

  # 5. Tetrahedral (Distorted)
  struct_tet_dist = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, s + 0.1, s, s, "O"),
      ("O", "ALA", "A", 3, -s, -s, s, "O"),
      ("O", "ALA", "A", 4, -s, s, -s, "O"),
      ("O", "ALA", "A", 5, s, -s, -s, "O"),
    ]
  )
  df = struct_tet_dist.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "tetrahedral"
  assert df.iloc[0]["geometry_deviation_deg"] > 0.1

  # 6. Square Planar (Ideal)
  struct_sq = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 3, 0.0, 2.0, 0.0, "O"),
      ("O", "ALA", "A", 4, -2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 5, 0.0, -2.0, 0.0, "O"),
    ]
  )
  df = struct_sq.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "square planar"
  assert df.iloc[0]["geometry_deviation_deg"] == pytest.approx(0.0, abs=1e-3)

  # 7. Square Planar (Distorted)
  struct_sq_dist = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.1, 0.0, "O"),
      ("O", "ALA", "A", 3, 0.0, 2.0, 0.0, "O"),
      ("O", "ALA", "A", 4, -2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 5, 0.0, -2.0, 0.0, "O"),
    ]
  )
  df = struct_sq_dist.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "square planar"
  assert df.iloc[0]["geometry_deviation_deg"] > 0.1

  # 8. Trigonal Bipyramidal (Ideal)
  struct_tb = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 3, -1.0, np.sqrt(3), 0.0, "O"),
      ("O", "ALA", "A", 4, -1.0, -np.sqrt(3), 0.0, "O"),
      ("O", "ALA", "A", 5, 0.0, 0.0, 2.0, "O"),
      ("O", "ALA", "A", 6, 0.0, 0.0, -2.0, "O"),
    ]
  )
  df = struct_tb.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "trigonal bipyramidal"
  assert df.iloc[0]["geometry_deviation_deg"] == pytest.approx(0.0, abs=1e-3)

  # 9. Square Pyramidal (Ideal)
  struct_sp = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 3, 0.0, 2.0, 0.0, "O"),
      ("O", "ALA", "A", 4, -2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 5, 0.0, -2.0, 0.0, "O"),
      ("O", "ALA", "A", 6, 0.0, 0.0, 2.0, "O"),
    ]
  )
  df = struct_sp.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "square pyramidal"
  assert df.iloc[0]["geometry_deviation_deg"] == pytest.approx(0.0, abs=1e-3)

  # 10. Octahedral (Ideal)
  struct_oct = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 3, -2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 4, 0.0, 2.0, 0.0, "O"),
      ("O", "ALA", "A", 5, 0.0, -2.0, 0.0, "O"),
      ("O", "ALA", "A", 6, 0.0, 0.0, 2.0, "O"),
      ("O", "ALA", "A", 7, 0.0, 0.0, -2.0, "O"),
    ]
  )
  df = struct_oct.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "octahedral"
  assert df.iloc[0]["geometry_deviation_deg"] == pytest.approx(0.0, abs=1e-3)

  # 11. Octahedral (Distorted)
  struct_oct_dist = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.1, 0.0, "O"),
      ("O", "ALA", "A", 3, -2.0, 0.0, 0.0, "O"),
      ("O", "ALA", "A", 4, 0.0, 2.0, 0.0, "O"),
      ("O", "ALA", "A", 5, 0.0, -2.0, 0.0, "O"),
      ("O", "ALA", "A", 6, 0.0, 0.0, 2.0, "O"),
      ("O", "ALA", "A", 7, 0.0, 0.0, -2.0, "O"),
    ]
  )
  df = struct_oct_dist.detect_coordination_centers()
  assert df.iloc[0]["geometry"] == "octahedral"
  assert df.iloc[0]["geometry_deviation_deg"] > 0.1


def test_metal_coordination_configurability_and_evidence():
  # 1. Test distance configurability
  struct = make_structure([("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"), ("O", "ALA", "A", 2, 0.0, 0.0, 2.9, "O")])
  # Default is 2.8, so it shouldn't detect
  df_default = struct.detect_interactions(interaction_types=["metal_coordination"])
  assert len(df_default) == 0

  # Cutoff increased to 3.0, it should detect
  df_high = struct.detect_interactions(interaction_types=["metal_coordination"], metal_coordination_cutoff=3.0)
  assert len(df_high) == 1
  assert df_high.iloc[0]["evidence"] == "detected"

  # 2. Test candidate vs detected evidence
  struct_mixed = make_structure(
    [
      ("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"),
      ("O", "ALA", "A", 2, 2.0, 0.0, 0.0, "O"),
      ("O", "HOH", "A", 3, 0.0, 2.0, 0.0, "O"),
      ("O", "XYZ", "A", 4, -2.0, 0.0, 0.0, "O"),
    ]
  )

  # Without candidates (default): only the protein donor ALA should be reported/coordinating
  df_cc_default = struct_mixed.detect_coordination_centers(include_candidates=False)
  assert len(df_cc_default) == 1
  assert df_cc_default.iloc[0]["coordination_number"] == 1
  assert list(df_cc_default.iloc[0]["donor_elements"]) == ["O"]
  assert df_cc_default.iloc[0]["evidence"] == "detected"

  # With candidates: all three donors should be reported
  df_cc_candidates = struct_mixed.detect_coordination_centers(include_candidates=True)
  assert len(df_cc_candidates) == 1
  assert df_cc_candidates.iloc[0]["coordination_number"] == 3
  assert sorted(list(df_cc_candidates.iloc[0]["donor_atom_indices"])) == [1, 2, 3]
  assert df_cc_candidates.iloc[0]["evidence"] == "detected"  # Highest priority

  # If we only have HOH (water), without candidates -> 0 coordination centers
  struct_water_only = make_structure([("ZN", "ZN", "A", 1, 0.0, 0.0, 0.0, "ZN"), ("O", "HOH", "A", 2, 0.0, 2.0, 0.0, "O")])
  assert len(struct_water_only.detect_coordination_centers(include_candidates=False)) == 0

  # With candidates -> 1 coordination center of type candidate
  df_cc_water = struct_water_only.detect_coordination_centers(include_candidates=True)
  assert len(df_cc_water) == 1
  assert df_cc_water.iloc[0]["coordination_number"] == 1
  assert df_cc_water.iloc[0]["evidence"] == "candidate"


def test_analyze_interactions_orchestrator():
  from neurosnap.structure.interactions import analyze_interactions
  from neurosnap.structure.interaction_report import InteractionReport
  from neurosnap.structure import StructureEnsemble

  # 1. Require a single Structure, not ensemble or stack
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CA", "ALA", "B", 1, 1.5, 0.0, 0.0, "C")])
  ensemble = StructureEnsemble()
  with pytest.raises(TypeError, match="expects a Structure"):
    analyze_interactions(ensemble)

  # 2. Default types should be conservative (contact, covalent)
  report = analyze_interactions(struct)
  assert isinstance(report, InteractionReport)
  for rec in report.records:
    assert rec.interaction_type in {"contact", "covalent"}

  # 3. Running only requested interaction types
  report_clash = analyze_interactions(struct, interaction_types=["vdw_clash"])
  for rec in report_clash.records:
    assert rec.interaction_type == "vdw_clash"

  # 4. Deduplication by semantic key (atom_index1, atom_index2, interaction_type)
  # retaining explicit > detected > candidate, and preserving complementary details
  struct_cys = make_structure([("SG", "CYS", "A", 1, 0.0, 0.0, 0.0, "S"), ("SG", "CYS", "B", 1, 2.03, 0.0, 0.0, "S")])
  # a disulfide the source file declared outright
  struct_cys.bonds = np.array([(0, 1, 1, int(BondType.DISULFIDE))], dtype=struct_cys._dtype_bond)
  report_dedup = analyze_interactions(struct_cys, interaction_types=["disulfide"])
  assert len(report_dedup.records) == 1
  assert report_dedup.records[0].evidence == "explicit"

  # 5. Metadata and row details include all cutoff/rule parameters
  report_meta = analyze_interactions(struct, interaction_types=["contact"])
  assert "params" in report_meta.metadata
  assert report_meta.metadata["params"]["contact_cutoff_a"] == 4.5
  assert len(report_meta.records) > 0
  assert "params" in report_meta.records[0].details
  assert report_meta.records[0].details["params"]["contact_cutoff_a"] == 4.5

  # 6. Sorting is deterministic
  report1 = analyze_interactions(struct, interaction_types=["contact", "vdw_clash"])
  report2 = analyze_interactions(struct, interaction_types=["contact", "vdw_clash"])
  assert [r.interaction_id for r in report1.records] == [r.interaction_id for r in report2.records]


def test_ckdtree_performance_and_equivalence(monkeypatch):
  from scipy.spatial import cKDTree as real_cKDTree
  import scipy.spatial
  from neurosnap.structure.interface import find_interface_contacts

  # We want to spy on cKDTree construction
  call_count = 0

  class SpyKDTree(real_cKDTree):
    def __init__(self, *args, **kwargs):
      nonlocal call_count
      call_count += 1
      super().__init__(*args, **kwargs)

  # Monkeypatch cKDTree in scipy.spatial
  monkeypatch.setattr(scipy.spatial, "cKDTree", SpyKDTree)

  # Generate at least hundreds of atoms per entity (e.g. 200 atoms for chain A, 200 atoms for chain B)
  atom_defs = []
  np.random.seed(42)
  coords_a = np.random.uniform(0, 10, size=(200, 3))
  # Place B nearby A so we get contacts
  coords_b = np.random.uniform(2, 12, size=(200, 3))

  for idx, coord in enumerate(coords_a):
    atom_defs.append(("CA", "ALA", "A", idx + 1, float(coord[0]), float(coord[1]), float(coord[2]), "C"))

  for idx, coord in enumerate(coords_b):
    atom_defs.append(("CA", "ALA", "B", idx + 1, float(coord[0]), float(coord[1]), float(coord[2]), "C"))

  structure = make_structure(atom_defs)

  # Run interface contact detection which calls find_contacts, which uses cKDTree via _find_neighbor_candidates
  contacts = find_interface_contacts(structure, "A", "B", cutoff=4.5, hydrogens=True)

  # Verify cKDTree path was used (tree1 and tree2 built)
  assert call_count >= 2

  # Verify equivalence with brute-force pairwise calculation
  brute_force_contacts = []
  for idx_a, coord_a in enumerate(coords_a):
    for idx_b, coord_b in enumerate(coords_b):
      dist = np.linalg.norm(coord_a - coord_b)
      if dist <= 4.5:
        # Save indices relative to the structure:
        # Chain A starts at 0, Chain B starts at 200
        brute_force_contacts.append((idx_a, 200 + idx_b))

  # Gather structure-based contact indices
  detected_contacts = []
  for atom1, atom2 in contacts:
    coord1 = atom1.coord
    coord2 = atom2.coord
    idx1 = np.argmin(np.linalg.norm(coords_a - coord1, axis=-1))
    idx2 = np.argmin(np.linalg.norm(coords_b - coord2, axis=-1))
    detected_contacts.append((min(idx1, 200 + idx2), max(idx1, 200 + idx2)))

  brute_force_contacts = [tuple(sorted(p)) for p in brute_force_contacts]
  brute_force_contacts.sort()
  detected_contacts = [tuple(sorted(p)) for p in detected_contacts]
  detected_contacts.sort()

  # Check equivalence
  assert len(detected_contacts) == len(brute_force_contacts)
  for p1, p2 in zip(detected_contacts, brute_force_contacts):
    assert p1 == p2


def test_get_coordination_centers_works_with_untyped_ligand_present():
  """Coordination centers must not require RDKit typing of unrelated ligands.

  Regression: this method used to request every interaction type, so it inherited
  the aligned-RDKit-molecule requirement of the hydrogen-bond and salt-bridge
  rules and raised on any structure containing a heterogen.
  """
  from tests._structure_test_utils import FILES
  from neurosnap.io.pdb import parse_pdb

  structure = parse_pdb(str(FILES / "protein_with_zinc_ions.pdb"), return_type="ensemble").first()

  # the zinc ions are non-polymer atoms with no aligned RDKit molecule
  assert any(structure.atom_annotations["hetero"])
  assert all(entity.rdkit_mol is None for entity in structure.entities)

  centers = structure.detect_coordination_centers()
  assert len(centers) == 2
  assert set(centers["element"]) == {"ZN"}
