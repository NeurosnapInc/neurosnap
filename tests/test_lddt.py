"""
Tests for the neurosnap.algos.LDDT module.
"""

from pathlib import Path

import numpy as np
import pytest

from neurosnap.algos.LDDT import calc_lddt
from neurosnap.protein import Protein

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"


@pytest.fixture(scope="module")
def rank1_protein():
  return Protein(str(FILES / "4AOW_af2_rank_1.pdb"))


@pytest.fixture(scope="module")
def rank2_protein():
  return Protein(str(FILES / "4AOW_af2_rank_2.pdb"))


def test_calc_lddt_identical_proteins_returns_one(rank1_protein):
  score = calc_lddt(rank1_protein, rank1_protein)
  assert score == 1.0


def test_calc_lddt_variant_models_close_but_not_identical(rank1_protein, rank2_protein):
  score = calc_lddt(rank1_protein, rank2_protein, precision=0)
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
