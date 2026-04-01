"""Alignment and RMSD tests for structure objects."""

import numpy as np
import pytest

from neurosnap.structure import align, calculate_rmsd, get_backbone

from tests._structure_test_utils import (
  AF2_RANK1,
  AF2_RANK2,
  DNA_BACKBONE_ATOMS,
  MIXED_BACKBONE_ATOMS,
  PDB_NO_H,
  PDB_WITH_H,
  PROTEIN_BACKBONE_ATOMS,
  RNA_CIF_1,
  RNA_CIF_2,
  ROT_Z_90,
  TRANSLATION_VECTOR,
  coords_from_atom_defs,
  make_structure,
  parse_single_model,
  replace_chain,
  transform_atoms,
)


def test_rmsd_align_af2():
  reference = parse_single_model(AF2_RANK1)
  mobile = parse_single_model(AF2_RANK2)
  rmsd = calculate_rmsd(reference, mobile, align_structures=True)
  assert rmsd < 0.45


@pytest.mark.xfail(reason="Backbone RMSD does not currently reconcile hydrogenated and non-hydrogenated structures.", strict=True)
def test_rmsd_align_hydrogens():
  reference = parse_single_model(PDB_NO_H)
  mobile = parse_single_model(PDB_WITH_H)
  rmsd = calculate_rmsd(reference, mobile, align_structures=True)
  assert rmsd < 1e-2


def test_align_backbone_protein_only():
  reference = make_structure(list(PROTEIN_BACKBONE_ATOMS))
  mobile = make_structure(transform_atoms(PROTEIN_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR))

  expected = coords_from_atom_defs(PROTEIN_BACKBONE_ATOMS)
  before = get_backbone(mobile, include_nucleotides=False)
  assert expected.shape == before.shape
  assert not np.allclose(expected, before)

  align(reference, mobile)
  after = get_backbone(mobile, include_nucleotides=False)
  assert np.allclose(expected, after, atol=1e-3)


def test_align_backbone_pairwise_chain_mapping():
  reference = make_structure(replace_chain(PROTEIN_BACKBONE_ATOMS, "B"))
  mobile = make_structure(replace_chain(transform_atoms(PROTEIN_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR), "A"))

  expected = coords_from_atom_defs(replace_chain(PROTEIN_BACKBONE_ATOMS, "B"))
  before = get_backbone(mobile, include_nucleotides=False)
  assert expected.shape == before.shape
  assert not np.allclose(expected, before)

  align(reference, mobile, chains1=["B"], chains2=["A"])
  after = get_backbone(mobile, include_nucleotides=False)
  assert np.allclose(expected, after, atol=1e-3)


def test_align_backbone_default_mode_requires_matching_chain_ids():
  reference = make_structure(replace_chain(PROTEIN_BACKBONE_ATOMS, "B"))
  mobile = make_structure(replace_chain(transform_atoms(PROTEIN_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR), "A"))

  with pytest.raises(ValueError, match="do not share common backbone atoms"):
    align(reference, mobile)


def test_rmsd_pairwise_chain_mapping():
  reference = make_structure(replace_chain(PROTEIN_BACKBONE_ATOMS, "B"))
  mobile = make_structure(replace_chain(transform_atoms(PROTEIN_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR), "A"))

  rmsd = calculate_rmsd(reference, mobile, chains1=["B"], chains2=["A"], align_structures=True)
  assert rmsd < 1e-3


def test_align_pairwise_chain_mapping_requires_equal_lengths():
  reference = make_structure(list(MIXED_BACKBONE_ATOMS))
  mobile = make_structure(transform_atoms(MIXED_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR))

  with pytest.raises(ValueError, match="same number of chains"):
    align(reference, mobile, chains1=["A", "B"], chains2=["A"])


def test_align_backbone_nucleotide_only():
  reference = make_structure(list(DNA_BACKBONE_ATOMS))
  mobile = make_structure(transform_atoms(DNA_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR))

  expected = get_backbone(reference)
  before = get_backbone(mobile)
  assert expected.shape == before.shape
  assert not np.allclose(expected, before)

  align(reference, mobile)
  after = get_backbone(mobile)
  assert np.allclose(expected, after, atol=1e-3)


@pytest.mark.xfail(reason="mmCIF parsing is not implemented in the structure layer yet.", strict=True)
def test_align_backbone_rna_only_real_structures():
  reference = parse_single_model(RNA_CIF_1)
  mobile = parse_single_model(RNA_CIF_2)

  before = get_backbone(mobile)
  expected = get_backbone(reference)
  rmsd_before = float(np.sqrt(np.mean(np.sum((before - expected) ** 2, axis=1))))

  align(reference, mobile)
  after = get_backbone(mobile)
  rmsd_after = float(np.sqrt(np.mean(np.sum((after - expected) ** 2, axis=1))))

  assert rmsd_after < rmsd_before
  assert rmsd_after < 2.0


def test_align_backbone_mixed_protein_and_nucleotide():
  reference = make_structure(list(MIXED_BACKBONE_ATOMS))
  mobile = make_structure(transform_atoms(MIXED_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR))

  expected = get_backbone(reference)
  before = get_backbone(mobile)
  assert expected.shape == before.shape
  assert not np.allclose(expected, before)

  align(reference, mobile)
  after = get_backbone(mobile)
  assert np.allclose(expected, after, atol=1e-3)
