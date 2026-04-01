"""Protein sequence utility tests."""

import pytest

from neurosnap.sequence.protein import getAA, isoelectric_point, molecular_weight, net_charge, sanitize_aa_seq


def test_getAA_and_sanitize_and_mw_and_charge_and_pi():
  record = getAA("A")
  assert (record.code, record.abr, record.name) == ("A", "ALA", "ALANINE")

  record = getAA("ala")
  assert (record.code, record.abr, record.name) == ("A", "ALA", "ALANINE")

  with pytest.raises(ValueError, match=r"Unknown amino acid identifier"):
    getAA("???")

  sequence = sanitize_aa_seq(" a c d e f * \n", non_standard="reject", trim_term=True)
  assert sequence == "ACDEF"
  assert sanitize_aa_seq("ACDZX", non_standard="allow") == "ACDZX"
  with pytest.raises(ValueError):
    sanitize_aa_seq("ACDZ?", non_standard="reject")

  from neurosnap.constants import AA_MASS_PROTEIN_AVG as aa_mass_average

  molecular_weight_gly = molecular_weight("G")
  assert abs(molecular_weight_gly - aa_mass_average["G"]) < 1e-6

  molecular_weight_ag = molecular_weight("AG")
  assert abs(molecular_weight_ag - (aa_mass_average["A"] + aa_mass_average["G"] - 18.015)) < 1e-6

  acidic_charge = net_charge("DE", pH=7.0)
  basic_charge = net_charge("KR", pH=7.0)
  assert acidic_charge < 0 and basic_charge > 0

  pi = isoelectric_point("ACDEFGHIKLMNPQRSTVWY")
  assert 0.0 <= pi <= 14.0


def test_getAA_non_standard_handling():
  with pytest.raises(ValueError, match=r"Encountered non-standard amino acid"):
    getAA("MSE", non_standard="reject")

  record = getAA("MSE", non_standard="allow")
  assert record.abr == "MSE"
  assert record.name.upper().startswith("SELENO")
  assert record.code in (None, "?")

  converted = getAA("MSE", non_standard="convert")
  assert (converted.code, converted.abr, converted.name) == ("M", "MET", "METHIONINE")
