"""Protein sequence utility tests."""

import pytest

from neurosnap.constants.sequence import AA_RECORDS_AMBIGUOUS, AA_RECORDS_CANONICAL, AA_RECORDS_FORCEFIELD_VARIANTS
from neurosnap.sequence.protein import isoelectric_point, molecular_weight, net_charge, sanitize_aa_seq


def test_aa_records_and_sanitize_and_mw_and_charge_and_pi():
  record = AA_RECORDS_CANONICAL.get_by_code("A")
  assert (record.code, record.abr, record.name) == ("A", "ALA", "ALANINE")

  record = AA_RECORDS_CANONICAL.get_by_abr("ala")
  assert (record.code, record.abr, record.name) == ("A", "ALA", "ALANINE")

  record = AA_RECORDS_CANONICAL.get_by_name("alanine")
  assert (record.code, record.abr, record.name) == ("A", "ALA", "ALANINE")

  assert AA_RECORDS_CANONICAL.get_by_abr("???") is None
  assert AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr("???") is None

  sequence = sanitize_aa_seq(" a c d e f * \n", non_standard="reject", trim_term=True)
  assert sequence == "ACDEF"
  assert sanitize_aa_seq("ACDZX", non_standard="allow") == "ACDZX"
  assert sanitize_aa_seq("ABZJ", non_standard="convert") == "ADEL"
  with pytest.raises(ValueError):
    sanitize_aa_seq("ACDZ?", non_standard="reject")
  with pytest.raises(ValueError):
    sanitize_aa_seq("AX", non_standard="convert")
  with pytest.raises(ValueError):
    sanitize_aa_seq("*", non_standard="convert", trim_term=False)

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


def test_aa_record_tables_separate_forcefield_variants_from_ambiguous_records():
  record = AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr("HID")
  assert record.abr == "HID"
  assert record.name.upper().startswith("HISTIDINE")
  assert record.code == "H"

  converted = AA_RECORDS_CANONICAL.get_canonical_record(record)
  assert (converted.code, converted.abr, converted.name) == ("H", "HIS", "HISTIDINE")

  assert AA_RECORDS_CANONICAL.get_by_abr("ASX") is None
  assert AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr("ASX") is None
  assert AA_RECORDS_AMBIGUOUS.get_by_abr("ASX").standard_equiv_abr == "ASP"


def test_aa_records_handles_protonation_variants():
  standard = AA_RECORDS_CANONICAL.get_by_code("H")
  assert standard.abr == "HIS"

  histidine_alias = AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr("HID")
  assert (histidine_alias.code, histidine_alias.abr, histidine_alias.standard_equiv_abr) == ("H", "HID", "HIS")

  aspartate_alias = AA_RECORDS_CANONICAL.get_canonical_record(AA_RECORDS_FORCEFIELD_VARIANTS.get_by_abr("ASH"))
  assert (aspartate_alias.code, aspartate_alias.abr) == ("D", "ASP")
