from pathlib import Path

import pytest

from neurosnap.algos.LDDT import calc_lddt
from neurosnap.protein import Protein

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"


def test_calc_lddt_protein_inputs_consistent_across_alignment():
  prot1 = Protein(str(FILES / "4AOW_af2_rank_1.pdb"))
  prot2 = Protein(str(FILES / "4AOW_af2_rank_2.pdb"))

  # Identical structures should score perfectly.
  assert calc_lddt(prot1, prot1) == pytest.approx(1.0, abs=1e-6)

  # Closely related structures should match historical expectations.
  score = calc_lddt(prot1, prot2)
  assert score == pytest.approx(0.9828, abs=1e-4)

  # lDDT is superposition-free, so aligning structures must not change the score.
  prot1_ref = Protein(str(FILES / "4AOW_af2_rank_1.pdb"))
  prot2_aligned = Protein(str(FILES / "4AOW_af2_rank_2.pdb"))
  prot1_ref.align(prot2_aligned)

  aligned_score = calc_lddt(prot1_ref, prot2_aligned)
  assert aligned_score == pytest.approx(score, abs=1e-6)
