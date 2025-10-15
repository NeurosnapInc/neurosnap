# tests/test_ipsae.py
import io
import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder

from neurosnap.algos.ipsae import (
  calc_d0,
  calc_d0_array,
  calculate_ipSAE,
  contiguous_ranges,
  init_pairdict_array,
  init_pairdict_scalar,
  init_pairdict_set,
  ptm_func,
  ptm_func_vec,
)
from neurosnap.protein import Protein

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"


# ----------------------------
# simple/unit helpers
# ----------------------------


def test_contiguous_ranges():
  assert contiguous_ranges(set()) == ""
  assert contiguous_ranges({5}) == "5"
  assert contiguous_ranges({1, 2, 3}) == "1-3"
  assert contiguous_ranges({1, 3, 4, 7, 8, 10}) == "1+3-4+7-8+10"


def test_ptm_func_scalar_and_vectorized():
  # scalar
  assert ptm_func(0.0, 5.0) == 1.0
  v = ptm_func(5.0, 5.0)
  assert 0.49 < v < 0.51  # ~0.5
  # vectorized
  x = np.array([0.0, 5.0, 10.0])
  y = ptm_func_vec(x, 5.0)
  assert y.shape == (3,)
  assert y[0] == 1.0 and 0.49 < y[1] < 0.51 and y[2] < 0.3


def test_calc_d0_floor_and_mode():
  # L is clamped to >=27
  d0_prot = calc_d0(10.0, "protein")
  d0_na = calc_d0(10.0, "nucleic_acid")
  assert d0_prot >= 1.0
  assert d0_na >= 2.0
  # monotonic-ish with L
  assert calc_d0(27.0, "protein") <= calc_d0(100.0, "protein")


def test_calc_d0_array_matches_scalar():
  Ls = np.array([10.0, 27.0, 100.0])
  arr = calc_d0_array(Ls, "protein")
  exp = np.array([calc_d0(L, "protein") for L in Ls])
  # elementwise same
  assert np.allclose(arr, exp)


def test_init_pairdict_variants():
  chains = np.array(list("AABBC"))
  d_scalar = init_pairdict_scalar(chains, init_val=-1.0)
  d_array = init_pairdict_array(chains, size=4)
  d_set = init_pairdict_set(chains)

  uniq = set("ABC")
  # keys are chain IDs; no self-pairs
  for d in (d_scalar, d_array, d_set):
    assert set(d.keys()) == uniq
    for k, inner in d.items():
      assert set(inner.keys()) == (uniq - {k})

  # array sizes correct
  for inner in d_array.values():
    for v in inner.values():
      assert isinstance(v, np.ndarray) and v.shape == (4,)


# ----------------------------
# end-to-end: provided fixtures
# ----------------------------


def _load_plddt_pae(json_path: Path):
  with open(json_path) as f:
    score = json.load(f)
  plddt = np.asarray(score["plddt"])
  pae = np.asarray(score["pae"])
  # ensure square pae
  assert pae.ndim == 2 and pae.shape[0] == pae.shape[1]
  return plddt, pae


@pytest.mark.parametrize(
  "struct_path,score_path,expect_na_pair",
  [
    (FILES / "orf1_boltz1.cif", FILES / "orf1_boltz1.json", False),
    (FILES / "dimer_af2.pdb", FILES / "dimer_af2.json", False),
  ],
)
def test_calculate_ipsae_basic(struct_path: Path, score_path: Path, expect_na_pair: bool):
  assert struct_path.exists(), f"Missing structure fixture: {struct_path}"
  assert score_path.exists(), f"Missing score fixture: {score_path}"

  prot = Protein(str(struct_path))
  plddt, pae = _load_plddt_pae(score_path)

  # Should succeed and return the full result dict
  res = calculate_ipSAE(prot, plddt=plddt, pae_matrix=pae, return_pml=True)

  # sanity: expected top-level keys
  for k in ["by_residue", "asym", "max", "min", "counts", "scores", "params", "pml", "residue_order"]:
    assert k in res

  # chain sanity: expect at least two chains like 'A' and 'B'
  chains = np.unique(res["residue_order"]["chains"])
  assert len(chains) >= 2
  c1, c2 = chains[0], chains[1]

  # by_residue arrays align with residue count
  N = len(res["residue_order"]["chains"])
  for k in ["iptm_d0chn", "ipsae_d0chn", "ipsae_d0dom", "ipsae_d0res", "n0res_byres", "d0res_byres"]:
    arr = res["by_residue"][k][c1][c2]
    assert isinstance(arr, np.ndarray) and arr.shape == (N,)

  # asym metrics are floats in [0,1]
  for k in ["iptm_d0chn", "ipsae_d0chn", "ipsae_d0dom", "ipsae_d0res"]:
    v12 = res["asym"][k][c1][c2]
    v21 = res["asym"][k][c2][c1]
    assert 0.0 <= v12 <= 1.0 and 0.0 <= v21 <= 1.0

  # symmetric max/min present for both directions and within [0,1]
  for bucket in ["max", "min"]:
    for k in ["iptm_d0chn", "ipsae_d0chn", "ipsae_d0dom", "ipsae_d0res"]:
      v12 = res[bucket][k][c1][c2]
      v21 = res[bucket][k][c2][c1]
      assert v12 == v21  # symmetry
      assert 0.0 <= v12 <= 1.0

  # auxiliary scores
  for k in ["pDockQ", "pDockQ2", "LIS"]:
    s12 = res["scores"][k][c1][c2]
    s21 = res["scores"][k][c2][c1]
    assert 0.0 <= s12 <= 1.0 and 0.0 <= s21 <= 1.0

  # pml should contain aliases for each ordered pair
  if res["pml"]:
    # e.g., "alias color_A_B"
    assert f"color_{c1}_{c2}" in res["pml"] or f"color_{c2}_{c1}" in res["pml"]


def test_calculate_ipsae_shape_mismatch_raises():
  prot = Protein(str(FILES / "dimer_af2.pdb"))
  plddt, pae = _load_plddt_pae(FILES / "dimer_af2.json")
  # break plddt length
  with pytest.raises(ValueError):
    calculate_ipSAE(prot, plddt=plddt[:-1], pae_matrix=pae)
  # break pae shape
  with pytest.raises(ValueError):
    calculate_ipSAE(prot, plddt=plddt, pae_matrix=pae[:-1, :])


def test_calculate_ipsae_reports_pairs_and_counts():
  prot = Protein(str(FILES / "dimer_af2.pdb"))
  plddt, pae = _load_plddt_pae(FILES / "dimer_af2.json")
  res = calculate_ipSAE(prot, plddt=plddt, pae_matrix=pae, pae_cutoff=10.0, dist_cutoff=10.0)

  chains = np.unique(res["residue_order"]["chains"])
  c1, c2 = chains[0], chains[1]

  # counts presence
  for k in ["n0chn", "d0chn", "n0dom", "d0dom", "pairs_with_pae_lt_cutoff"]:
    assert k in res["counts"]
    _ = res["counts"][k][c1][c2]  # access should not KeyError

  # valid pairs counts are non-negative
  assert res["counts"]["pairs_with_pae_lt_cutoff"][c1][c2] >= 0.0
  assert res["counts"]["pairs_with_pae_lt_cutoff_and_dist"][c1][c2] >= 0.0


def test_calculate_ipsae_accepts_nucleic_acids():
  builder = StructureBuilder()
  builder.init_structure("NA")
  builder.init_model(0)
  chain_offsets = {"A": np.array([0.0, 0.0, 0.0]), "B": np.array([8.0, 0.0, 0.0])}
  residues = {"A": [("DA", 1), ("A", 2)], "B": [("DG", 1), ("U", 2)]}

  for chain_id, res_list in residues.items():
    builder.init_chain(chain_id)
    builder.init_seg("    ")
    for resname, resseq in res_list:
      builder.init_residue(resname, " ", resseq, " ")
      base = chain_offsets[chain_id] + np.array([0.0, 0.0, float(resseq)])
      builder.init_atom("C3'", base, 1.0, 10.0, " ", "C3'", element="C")
      builder.init_atom("C1'", base + np.array([0.5, 0.5, 0.0]), 1.0, 10.0, " ", "C1'", element="C")

  structure = builder.get_structure()
  handle = io.StringIO()
  pdbio = PDBIO()
  pdbio.set_structure(structure)
  pdbio.save(handle)
  handle.seek(0)

  prot = Protein(handle, format="pdb")

  plddt = np.full(4, 90.0, dtype=float)
  pae = np.full((4, 4), 5.0, dtype=float)
  np.fill_diagonal(pae, 0.0)

  res = calculate_ipSAE(prot, plddt=plddt, pae_matrix=pae, pae_cutoff=10.0)

  names = set(res["residue_order"]["names"].tolist())
  assert {"DA", "A", "DG", "U"}.issubset(names)

  # chain pair should be treated as nucleic acid (min d0 of 2.0)
  assert res["counts"]["d0chn"]["A"]["B"] >= 2.0

  # by-residue arrays still align with residue count
  assert res["by_residue"]["ipsae_d0chn"]["A"]["B"].shape == (4,)

  # ensure we recorded at least one valid inter-chain pair
  assert res["counts"]["pairs_with_pae_lt_cutoff"]["A"]["B"] > 0.0
