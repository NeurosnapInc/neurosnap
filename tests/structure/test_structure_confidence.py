import pytest
import numpy as np

from neurosnap.structure import Structure
from tests._structure_test_utils import make_structure


def test_confidence_metric_not_labeled_as_bond_probabilities():
  struct = make_structure([("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"), ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C")])
  df = struct.detect_interactions()
  assert len(df) > 0, "Expected at least one interaction"

  # verify that confidence metrics aren't called probabilities in columns
  for col in df.columns:
    assert "probability" not in col.lower()

  # verify that confidence metrics aren't called probabilities inside details
  if "details" in df.columns:
    for details_dict in df["details"]:
      if isinstance(details_dict, dict):
        assert "probability" not in [str(k).lower() for k in details_dict.keys()]


def test_summarize_plddt_auto_scale_detection():
  from neurosnap.structure import summarize_plddt

  struct = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
      ("CA", "GLY", "B", 1, 10.0, 0.0, 0.0, "C"),
    ]
  )

  # Case 1: scale="auto", values in [0, 1] -> detected as 1.0 -> multiplied by 100
  plddt_01 = [0.2, 0.5, 0.8]
  report = summarize_plddt(struct, plddt=plddt_01, scale="auto")
  assert report.input_scale == 1.0
  assert np.allclose(report.atom["plddt"].values, [20.0, 50.0, 80.0])

  # Case 2: scale="auto", values in [0, 100] but > 1 -> detected as 100.0 -> no multiplier
  plddt_0100 = [20.0, 50.0, 80.0]
  report = summarize_plddt(struct, plddt=plddt_0100, scale="auto")
  assert report.input_scale == 100.0
  assert np.allclose(report.atom["plddt"].values, [20.0, 50.0, 80.0])


def test_summarize_plddt_explicit_scale():
  from neurosnap.structure import summarize_plddt

  struct = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
    ]
  )

  # Explicitly specify scale=1.0
  report = summarize_plddt(struct, plddt=[0.5, 0.6], scale=1.0)
  assert report.input_scale == 1.0
  assert np.allclose(report.atom["plddt"].values, [50.0, 60.0])

  # Explicitly specify scale=100.0
  report = summarize_plddt(struct, plddt=[50.0, 60.0], scale=100.0)
  assert report.input_scale == 100.0
  assert np.allclose(report.atom["plddt"].values, [50.0, 60.0])

  # Mismatched explicit scale raises error
  with pytest.raises(ValueError):
    summarize_plddt(struct, plddt=[50.0, 60.0], scale=1.0)


def test_summarize_plddt_ambiguous_raises():
  from neurosnap.structure import summarize_plddt

  struct = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
    ]
  )

  # All zero raises when scale="auto"
  with pytest.raises(ValueError, match="Ambiguous all-zero/all-one"):
    summarize_plddt(struct, plddt=[0.0, 0.0], scale="auto")

  # All one raises when scale="auto"
  with pytest.raises(ValueError, match="Ambiguous all-zero/all-one"):
    summarize_plddt(struct, plddt=[1.0, 1.0], scale="auto")

  # Works fine with explicit scale
  report = summarize_plddt(struct, plddt=[1.0, 1.0], scale=1.0)
  assert report.input_scale == 1.0
  assert np.allclose(report.atom["plddt"].values, [100.0, 100.0])


def test_summarize_plddt_source_b_factor():
  from neurosnap.structure import summarize_plddt

  struct = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
    ]
  )

  # make_structure sets b_factor to 20.0
  report = summarize_plddt(struct, plddt=None, source="b_factor", scale="auto")
  assert report.input_scale == 100.0
  assert np.allclose(report.atom["plddt"].values, [20.0, 20.0])

  # Invalid source raises ValueError
  with pytest.raises(ValueError):
    summarize_plddt(struct, plddt=None, source="unsupported_source")


def test_summarize_plddt_aggregations_and_distribution():
  from neurosnap.structure import summarize_plddt

  # A: res_id 1 (3 atoms), B: res_id 2 (1 atom)
  struct = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
      ("N", "ALA", "A", 1, -1.0, 0.0, 0.0, "N"),
      ("CA", "GLY", "B", 2, 10.0, 0.0, 0.0, "C"),
    ]
  )

  plddt = [30.0, 60.0, 90.0, 85.0]
  report = summarize_plddt(struct, plddt=plddt, scale=100.0)

  # Check residue aggregation
  # ALA key: ('A', 1, '', 'ALA', False) -> mean of [30, 60, 90] is 60, min is 30, max is 90, median is 60
  ala_key = ("A", 1, "", "ALA", False)
  ala_row = report.residue.loc[[ala_key]].iloc[0]
  assert ala_row["count"] == 3
  assert ala_row["mean"] == 60.0
  assert ala_row["min"] == 30.0
  assert ala_row["max"] == 90.0
  assert ala_row["median"] == 60.0

  # Check chain aggregation
  # Chain A: [30, 60, 90], Chain B: [85]
  row_a = report.chain.loc["A"]
  assert row_a["count"] == 3
  assert row_a["mean"] == 60.0
  assert row_a["min"] == 30.0
  assert row_a["max"] == 90.0
  assert row_a["median"] == 60.0
  assert row_a["q25"] == 45.0
  assert row_a["q75"] == 75.0

  # Check distribution
  # <50: [30] -> 1 atom (25%)
  # 50-70: [60] -> 1 atom (25%)
  # 70-90: [85] -> 1 atom (25%)
  # >=90: [90] -> 1 atom (25%)
  dist = report.distribution
  assert dist.loc["<50", "count"] == 1
  assert dist.loc["<50", "percentage"] == 25.0
  assert dist.loc["50-70", "count"] == 1
  assert dist.loc["50-70", "percentage"] == 25.0
  assert dist.loc["70-90", "count"] == 1
  assert dist.loc["70-90", "percentage"] == 25.0
  assert dist.loc[">=90", "count"] == 1
  assert dist.loc[">=90", "percentage"] == 25.0

  # Percentages sum to 100
  assert np.isclose(dist["percentage"].sum(), 100.0)


def test_summarize_plddt_empty_structure():
  from neurosnap.structure import summarize_plddt

  struct = Structure(remove_annotations=False)
  report = summarize_plddt(struct, plddt=[], scale=100.0)

  assert len(report.atom) == 0
  assert len(report.residue) == 0
  assert len(report.chain) == 0
  assert len(report.distribution) == 4
  assert (report.distribution["count"] == 0).all()
  assert (report.distribution["percentage"] == 0.0).all()
