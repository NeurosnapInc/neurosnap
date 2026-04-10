"""Tests for the neurosnap.algos.lddt module."""

import numpy as np
import pytest

from neurosnap.algos.lddt import calc_lddt
from neurosnap.io.mmcif import parse_mmcif
from neurosnap.structure import Structure, StructureEnsemble
from tests._structure_test_utils import FILES, parse_ensemble, parse_single_model


@pytest.fixture(scope="module")
def rank1_protein():
  return parse_single_model(FILES / "4AOW_af2_rank_1.pdb")


@pytest.fixture(scope="module")
def rank2_protein():
  return parse_single_model(FILES / "4AOW_af2_rank_2.pdb")


@pytest.fixture(scope="module")
def rna_model_one():
  return parse_single_model(FILES / "rna_monomer_1.cif")


@pytest.fixture(scope="module")
def rna_model_two():
  return parse_single_model(FILES / "rna_monomer_2.cif")


def _make_structure(atom_defs):
  """Build a synthetic single-model structure with optional hetero residues."""
  structure = Structure(remove_annotations=False)
  structure.metadata["model_id"] = 1
  structure.atoms = np.array([(x, y, z) for _atom_name, _resname, _chain_id, _resid, _ins_code, _hetero, x, y, z, _element in atom_defs], dtype=structure._dtype_atoms)
  structure.atom_annotations = np.zeros(len(atom_defs), dtype=structure._dtype_atom_annotations)

  for atom_index, (atom_name, resname, chain_id, resid, ins_code, hetero, _x, _y, _z, element) in enumerate(atom_defs):
    structure.atom_annotations["chain_id"][atom_index] = chain_id
    structure.atom_annotations["res_id"][atom_index] = resid
    structure.atom_annotations["ins_code"][atom_index] = ins_code
    structure.atom_annotations["res_name"][atom_index] = resname
    structure.atom_annotations["hetero"][atom_index] = hetero
    structure.atom_annotations["atom_name"][atom_index] = atom_name
    structure.atom_annotations["element"][atom_index] = element
    structure.atom_annotations["atom_id"][atom_index] = atom_index + 1
    structure.atom_annotations["b_factor"][atom_index] = 20.0
    structure.atom_annotations["occupancy"][atom_index] = 1.0
    structure.atom_annotations["charge"][atom_index] = 0
    structure.atom_annotations["sym_id"][atom_index] = ""

  structure.bonds = np.zeros(0, dtype=structure._dtype_bond)
  structure._remove_empty_annotations()
  return structure


def test_calc_lddt_identical_proteins_returns_one(rank1_protein):
  score = calc_lddt(rank1_protein, rank1_protein)
  assert score == 1.0


def test_calc_lddt_variant_models_close_but_not_identical(rank1_protein, rank2_protein):
  score = calc_lddt(rank1_protein, rank2_protein)
  assert score < 1.0
  assert score == pytest.approx(0.982843137254902, rel=1e-6)


def test_calc_lddt_distance_map_shape_mismatch_raises():
  reference = np.zeros((2, 2))
  prediction = np.zeros((3, 3))
  with pytest.raises(ValueError):
    calc_lddt(reference, prediction)


def test_calc_lddt_mixed_input_types_raises(rank1_protein):
  with pytest.raises(TypeError):
    calc_lddt(rank1_protein, np.zeros((1, 1)))


def test_calc_lddt_rejects_multi_model_structure_inputs():
  model = parse_single_model(FILES / "4AOW_af2_rank_1.pdb")
  ensemble = StructureEnsemble([model, model])
  with pytest.raises(TypeError):
    calc_lddt(ensemble, ensemble)


def test_calc_lddt_identical_nucleic_acids_returns_one(rna_model_one):
  score = calc_lddt(rna_model_one, rna_model_one)
  assert score == 1.0


def test_calc_lddt_nucleic_acids_not_nan(rna_model_one, rna_model_two):
  score = calc_lddt(rna_model_one, rna_model_two)
  assert not np.isnan(score)


def test_calc_lddt_phosphotyrosine_peptide_and_grb2_models():
  struct1 = parse_mmcif(FILES / "phosphotyrosine_peptide_and_GRB2_chai_1_rank1.cif")
  struct2 = parse_mmcif(FILES / "phosphotyrosine_peptide_and_GRB2_chai_1_rank2.cif")

  score = calc_lddt(struct1[1], struct2[1])

  assert score == pytest.approx(0.9369076482256997)


def test_calc_lddt_includes_modified_polymer_residues_by_default():
  structure = _make_structure(
    (
      ("N", "ALA", "A", 1, "", False, 0.0, 0.0, 0.0, "N"),
      ("CA", "ALA", "A", 1, "", False, 1.5, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, "", False, 1.8, -1.2, 0.0, "C"),
      ("C", "ALA", "A", 1, "", False, 2.2, 1.1, 0.0, "C"),
      ("N", "PTM", "A", 2, "", True, 3.4, 1.2, 0.0, "N"),
      ("CA", "PTM", "A", 2, "", True, 4.2, 2.3, 0.0, "C"),
      ("CB", "PTM", "A", 2, "", True, 5.5, 1.9, 0.0, "C"),
      ("C", "PTM", "A", 2, "", True, 4.8, 3.7, 0.0, "C"),
    )
  )

  assert calc_lddt(structure, structure) == 1.0


def test_calc_lddt_includes_small_molecule_heavy_atoms_by_default():
  structure = _make_structure(
    (
      ("N", "ALA", "A", 1, "", False, 0.0, 0.0, 0.0, "N"),
      ("CA", "ALA", "A", 1, "", False, 1.5, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, "", False, 1.8, -1.2, 0.0, "C"),
      ("C", "ALA", "A", 1, "", False, 2.2, 1.1, 0.0, "C"),
      ("C1", "LIG", "A", 2, "", True, 3.0, 0.5, 0.0, "C"),
      ("O1", "LIG", "A", 2, "", True, 4.2, 0.5, 0.0, "O"),
      ("H1", "LIG", "A", 2, "", True, 2.6, 1.4, 0.0, "H"),
    )
  )

  assert calc_lddt(structure, structure) == 1.0
