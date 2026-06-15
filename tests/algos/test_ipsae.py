import json
from pathlib import Path

import numpy as np
import pytest

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
from tests._structure_test_utils import make_structure, parse_ensemble, parse_single_model

TESTS_DIR = Path(__file__).resolve().parents[1]
FILES = TESTS_DIR / "files"


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
  "struct_path,score_path",
  [
    (FILES / "orf1_boltz1.cif", FILES / "orf1_boltz1.json"),
    (FILES / "dimer_af2.pdb", FILES / "dimer_af2.json"),
  ],
)
def test_calculate_ipsae_basic(struct_path: Path, score_path: Path):
  assert struct_path.exists(), f"Missing structure fixture: {struct_path}"
  assert score_path.exists(), f"Missing score fixture: {score_path}"

  structure = parse_single_model(struct_path)
  plddt, pae = _load_plddt_pae(score_path)

  # Should succeed and return the full result dict
  res = calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae, return_pml=True)

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
  structure = parse_single_model(FILES / "dimer_af2.pdb")
  plddt, pae = _load_plddt_pae(FILES / "dimer_af2.json")
  # break plddt length
  with pytest.raises(ValueError):
    calculate_ipSAE(structure, plddt=plddt[:-1], pae_matrix=pae)
  # break pae shape
  with pytest.raises(ValueError):
    calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae[:-1, :])


def test_calculate_ipsae_reports_pairs_and_counts():
  structure = parse_single_model(FILES / "dimer_af2.pdb")
  plddt, pae = _load_plddt_pae(FILES / "dimer_af2.json")
  res = calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae, pae_cutoff=10.0, dist_cutoff=10.0)

  chains = np.unique(res["residue_order"]["chains"])
  c1, c2 = chains[0], chains[1]

  # counts presence
  for k in ["n0chn", "d0chn", "n0dom", "d0dom", "pairs_with_pae_lt_cutoff"]:
    assert k in res["counts"]
    _ = res["counts"][k][c1][c2]  # access should not KeyError

  # valid pairs counts are non-negative
  assert res["counts"]["pairs_with_pae_lt_cutoff"][c1][c2] >= 0.0
  assert res["counts"]["pairs_with_pae_lt_cutoff_and_dist"][c1][c2] >= 0.0


def test_calculate_ipsae_requires_structure():
  structure = parse_ensemble(FILES / "dimer_af2.pdb")
  plddt, pae = _load_plddt_pae(FILES / "dimer_af2.json")
  with pytest.raises(TypeError):
    calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae)


def test_calculate_ipsae_accepts_nucleic_acids():
  chain_offsets = {"A": np.array([0.0, 0.0, 0.0]), "B": np.array([8.0, 0.0, 0.0])}
  residues = {"A": [("DA", 1), ("A", 2)], "B": [("DG", 1), ("U", 2)]}
  atom_defs = []
  for chain_id, res_list in residues.items():
    for resname, resseq in res_list:
      base = chain_offsets[chain_id] + np.array([0.0, 0.0, float(resseq)])
      atom_defs.append(("C3'", resname, chain_id, resseq, float(base[0]), float(base[1]), float(base[2]), "C"))
      atom_defs.append(("C1'", resname, chain_id, resseq, float(base[0] + 0.5), float(base[1] + 0.5), float(base[2]), "C"))

  structure = make_structure(atom_defs)

  plddt = np.full(4, 90.0, dtype=float)
  pae = np.full((4, 4), 5.0, dtype=float)
  np.fill_diagonal(pae, 0.0)

  res = calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae, pae_cutoff=10.0)

  names = set(res["residue_order"]["names"].tolist())
  assert {"DA", "A", "DG", "U"}.issubset(names)

  # chain pair should be treated as nucleic acid (min d0 of 2.0)
  assert res["counts"]["d0chn"]["A"]["B"] >= 2.0

  # by-residue arrays still align with residue count
  assert res["by_residue"]["ipsae_d0chn"]["A"]["B"].shape == (4,)

  # ensure we recorded at least one valid inter-chain pair
  assert res["counts"]["pairs_with_pae_lt_cutoff"]["A"]["B"] > 0.0


def test_calculate_ipsae_prunes_nonstandard_residues():
  structure = make_structure(
    [
      ("CA", "MSE", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "MSE", "A", 1, 1.5, 0.0, 0.0, "C"),
      ("CA", "ALA", "A", 2, 3.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 2, 4.0, 0.0, 0.0, "C"),
      ("C3'", "DG", "B", 1, 0.0, 5.0, 0.0, "C"),
      ("C1'", "DG", "B", 1, 0.5, 5.5, 0.0, "C"),
    ]
  )

  # pLDDT/PAE include the non-standard residue; ipSAE should prune it automatically
  plddt = np.array([50.0, 90.0, 80.0], dtype=float)
  pae = np.full((3, 3), 5.0, dtype=float)
  np.fill_diagonal(pae, 0.0)

  res = calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae, pae_cutoff=10.0)

  names = res["residue_order"]["names"].tolist()
  assert names == ["ALA", "DG"]
  assert "MSE" not in names

  # by-residue arrays should match the filtered residue count
  assert res["by_residue"]["ipsae_d0chn"]["A"]["B"].shape == (2,)


def test_calculate_ipsae_accepts_token_expanded_nonstandard_residues():
  structure = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
      ("N", "MSE", "A", 2, 2.0, 0.0, 0.0, "N"),
      ("CA", "MSE", "A", 2, 3.0, 0.0, 0.0, "C"),
      ("SE", "MSE", "A", 2, 4.0, 0.0, 0.0, "SE"),
      ("CA", "ALA", "B", 1, 0.0, 5.0, 0.0, "C"),
      ("CB", "ALA", "B", 1, 1.0, 5.0, 0.0, "C"),
    ]
  )

  # Token-expanded payload:
  #   ALA(1) -> 1 representative token
  #   MSE(2) -> 3 atom-level tokens
  #   ALA(B1) -> 1 representative token
  # total = 5
  plddt = np.array([80.0, 70.0, 60.0, 50.0, 90.0], dtype=float)
  pae = np.full((5, 5), 5.0, dtype=float)
  np.fill_diagonal(pae, 0.0)

  res = calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae, pae_cutoff=10.0)

  names = res["residue_order"]["names"].tolist()
  numbers = res["residue_order"]["numbers"].tolist()
  chains = res["residue_order"]["chains"].tolist()

  assert len(names) == 5
  assert names.count("MSE") == 3
  assert sum(1 for ch, n, nm in zip(chains, numbers, names) if ch == "A" and n == 2 and nm == "MSE") == 3
  assert res["by_residue"]["ipsae_d0chn"]["A"]["B"].shape == (5,)


def test_calculate_ipsae_accepts_token_expanded_hetero_residues():
  structure = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
      ("CA", "GLY", "B", 1, 0.0, 5.0, 0.0, "C"),
      ("C1", "ATP", "C", 1, 2.0, 2.0, 0.0, "C"),
      ("N1", "ATP", "C", 1, 3.0, 2.0, 0.0, "N"),
      ("MG", "MG", "D", 1, 4.0, 2.0, 0.0, "MG"),
    ]
  )
  structure.atom_annotations["hetero"][3:] = True

  plddt = np.array([90.0, 85.0, 70.0, 68.0, 60.0], dtype=float)
  pae = np.full((5, 5), 5.0, dtype=float)
  np.fill_diagonal(pae, 0.0)

  res = calculate_ipSAE(structure, plddt=plddt, pae_matrix=pae, pae_cutoff=10.0)

  assert res["residue_order"]["names"].tolist() == ["ALA", "GLY", "ATP", "ATP", "MG"]
  assert res["residue_order"]["chains"].tolist() == ["A", "B", "C", "C", "D"]
  assert set(res["asym"]["ipsae_d0chn"].keys()) == {"A", "B", "C", "D"}
  assert res["by_residue"]["ipsae_d0chn"]["A"]["B"].shape == (5,)
  assert "C" in res["scores"]["pDockQ"]


def test_calculate_ipsae_min_metric_with_ptm_fixture():
  prot = parse_single_model(FILES / "chai1_dimer_ptm_protein_with_nanobody.cif")
  plddt, pae = _load_plddt_pae(FILES / "chai1_dimer_ptm_protein_with_nanobody.json")
  res = calculate_ipSAE(prot, plddt=plddt, pae_matrix=pae)

  assert "min" in res
  assert "ipsae_d0res" in res["min"]
  min_ab = res["min"]["ipsae_d0res"]["A"]["B"]
  min_ba = res["min"]["ipsae_d0res"]["B"]["A"]
  assert np.isfinite(min_ab)
  assert np.isfinite(min_ba)
  assert min_ab == min_ba
