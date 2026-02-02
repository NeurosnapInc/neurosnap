"""
Tests for EvoEF2 scoring parity checks against reference outputs.
"""

from pathlib import Path

import pytest

import numpy as np

from neurosnap.algos import evoef2
from neurosnap.algos.evoef2 import calculate_binding, calculate_interface_energy, calculate_stability, rebuild_missing_atoms

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"


def _compare_terms(actual, expected, *, abs_tol=0.1, rel_tol=0.01):
  bad_terms = []
  for key, exp in expected.items():
    if key not in actual:
      bad_terms.append(key)
      continue
    act = float(actual[key])
    delta = act - exp
    if abs(delta) > abs_tol and (abs(exp) < 1e-8 or abs(delta / exp) > rel_tol):
      bad_terms.append(key)
  return bad_terms


EVOEF2_REFERENCE_DIMER_AF2 = {
  "reference_ALA": -4.08,
  "reference_CYS": -0.0,
  "reference_ASP": -3.21,
  "reference_GLU": -17.15,
  "reference_PHE": 1.36,
  "reference_GLY": -20.93,
  "reference_HIS": -1.18,
  "reference_ILE": 37.28,
  "reference_LYS": -7.5,
  "reference_LEU": 16.13,
  "reference_MET": 1.52,
  "reference_ASN": -0.0,
  "reference_PRO": -2.59,
  "reference_GLN": -7.74,
  "reference_ARG": -10.58,
  "reference_SER": -23.74,
  "reference_THR": -2.5,
  "reference_VAL": 10.2,
  "reference_TRP": 0.0,
  "reference_TYR": 0.0,
  "intraR_vdwatt": -21.04,
  "intraR_vdwrep": 5.0,
  "intraR_electr": -0.26,
  "intraR_deslvP": 0.0,
  "intraR_deslvH": -2.8,
  "intraR_hbscbb_dis": -1.66,
  "intraR_hbscbb_the": -0.32,
  "intraR_hbscbb_phi": -0.0,
  "aapropensity": -11.36,
  "ramachandran": 268.22,
  "dunbrack": 46.69,
  "interS_vdwatt": -553.25,
  "interS_vdwrep": 84.96,
  "interS_electr": -35.97,
  "interS_deslvP": 368.03,
  "interS_deslvH": -251.21,
  "interS_ssbond": 0.0,
  "interS_hbbbbb_dis": -66.43,
  "interS_hbbbbb_the": -58.44,
  "interS_hbbbbb_phi": -66.84,
  "interS_hbscbb_dis": -3.26,
  "interS_hbscbb_the": -3.43,
  "interS_hbscbb_phi": -0.76,
  "interS_hbscsc_dis": -2.98,
  "interS_hbscsc_the": -1.07,
  "interS_hbscsc_phi": -0.0,
  "interD_vdwatt": -47.03,
  "interD_vdwrep": 1.97,
  "interD_electr": -2.8,
  "interD_deslvP": 28.44,
  "interD_deslvH": -43.48,
  "interD_ssbond": 0.0,
  "interD_hbbbbb_dis": -5.68,
  "interD_hbbbbb_the": -4.72,
  "interD_hbbbbb_phi": -5.77,
  "interD_hbscbb_dis": -0.06,
  "interD_hbscbb_the": -0.67,
  "interD_hbscbb_phi": -0.42,
  "interD_hbscsc_dis": 0.0,
  "interD_hbscsc_the": 0.0,
  "interD_hbscsc_phi": 0.0,
  "total": -423.11,
}


EVOEF2_REFERENCE_4AOW_AF2 = {
  "reference_ALA": -6.12,
  "reference_CYS": -0.89,
  "reference_ASP": -16.84,
  "reference_GLU": -11.03,
  "reference_PHE": 5.43,
  "reference_GLY": -54.42,
  "reference_HIS": -2.36,
  "reference_ILE": 41.94,
  "reference_LYS": -20.0,
  "reference_LEU": 45.16,
  "reference_MET": 3.04,
  "reference_ASN": -30.17,
  "reference_PRO": -6.47,
  "reference_GLN": -23.23,
  "reference_ARG": -17.19,
  "reference_SER": -53.41,
  "reference_THR": -11.65,
  "reference_VAL": 35.7,
  "reference_TRP": 26.05,
  "reference_TYR": 4.2,
  "intraR_vdwatt": -55.12,
  "intraR_vdwrep": 10.39,
  "intraR_electr": -0.51,
  "intraR_deslvP": 0.0,
  "intraR_deslvH": -5.7,
  "intraR_hbscbb_dis": -9.12,
  "intraR_hbscbb_the": -0.26,
  "intraR_hbscbb_phi": -0.0,
  "aapropensity": -47.4,
  "ramachandran": 675.73,
  "dunbrack": 124.34,
  "interS_vdwatt": -1939.27,
  "interS_vdwrep": 196.76,
  "interS_electr": -87.9,
  "interS_deslvP": 1182.12,
  "interS_deslvH": -805.85,
  "interS_ssbond": 0.0,
  "interS_hbbbbb_dis": -148.31,
  "interS_hbbbbb_the": -115.7,
  "interS_hbbbbb_phi": -163.12,
  "interS_hbscbb_dis": -44.66,
  "interS_hbscbb_the": -32.2,
  "interS_hbscbb_phi": -9.7,
  "interS_hbscsc_dis": -40.52,
  "interS_hbscsc_the": -13.68,
  "interS_hbscsc_phi": -0.0,
  "interD_vdwatt": 0.0,
  "interD_vdwrep": 0.0,
  "interD_electr": 0.0,
  "interD_deslvP": 0.0,
  "interD_deslvH": 0.0,
  "interD_ssbond": 0.0,
  "interD_hbbbbb_dis": 0.0,
  "interD_hbbbbb_the": 0.0,
  "interD_hbbbbb_phi": 0.0,
  "interD_hbscbb_dis": 0.0,
  "interD_hbscbb_the": 0.0,
  "interD_hbscbb_phi": 0.0,
  "interD_hbscsc_dis": 0.0,
  "interD_hbscsc_the": 0.0,
  "interD_hbscsc_phi": 0.0,
  "total": -1421.94,
}


@pytest.mark.parametrize(
  "pdb_name,reference,total_delta_limit",
  [
    ("dimer_af2.pdb", EVOEF2_REFERENCE_DIMER_AF2, 4.0),
    ("4AOW_af2_rank_1.pdb", EVOEF2_REFERENCE_4AOW_AF2, 31.0),
  ],
)
def test_evoef2_stability_matches_reference(pdb_name, reference, total_delta_limit):
  actual = calculate_stability(str(FILES / pdb_name))
  bad_terms = _compare_terms(actual, reference, abs_tol=0.1, rel_tol=0.01)
  total_delta = abs(float(actual["total"]) - float(reference["total"]))
  assert total_delta <= total_delta_limit, (
    f"Total delta {total_delta:.3f} exceeds limit {total_delta_limit:.3f}. "
    f"Terms outside tolerance: {', '.join(bad_terms) if bad_terms else 'none'}"
  )


def test_energy_term_weighting_sets_total():
  terms = evoef2._energy_term_initialize()
  terms[1] = 1.0
  terms[21] = -2.0
  weights = [1.0] * evoef2.MAX_EVOEF_ENERGY_TERM_NUM
  weights[21] = 2.0
  weighted = evoef2._energy_term_weighting(terms, weights)
  expected_total = weighted[1] + weighted[21]
  assert weighted[0] == pytest.approx(expected_total)


def test_load_tables_shapes():
  aap = evoef2.load_aapropensity()
  rama = evoef2.load_ramachandran()
  dun = evoef2.load_dunbrack()
  assert aap.aap.shape == (36, 36, 20)
  assert rama.rama.shape == (36, 36, 20)
  assert len(dun.bins) == 36 * 36


def test_rebuild_missing_atoms_produces_valid_atoms():
  structure = rebuild_missing_atoms(str(FILES / "1nkp_mycmax.pdb"))
  assert structure.chains
  valid_atoms = sum(
    1
    for chain in structure.chains
    for res in chain.residues
    for atom in res.atoms.values()
    if atom.is_xyz_valid
  )
  assert valid_atoms > 0


def test_calc_phi_psi_assigns_values():
  structure = rebuild_missing_atoms(str(FILES / "dimer_af2.pdb"))
  for chain in structure.chains:
    if chain.is_protein:
      evoef2._calc_phi_psi(chain)
      for res in chain.residues:
        phi, psi = res.phipsi
        assert np.isfinite(phi)
        assert np.isfinite(psi)


def test_interface_and_binding_have_expected_keys():
  pdb_path = str(FILES / "dimer_af2.pdb")
  interface = calculate_interface_energy(pdb_path, split1=["A"], split2=["B"])
  binding = calculate_binding(pdb_path, split1=["A"], split2=["B"])
  assert "total" in interface
  assert "dg_bind" in binding
  assert "stability_complex" in binding
  assert "stability_split1" in binding
  assert "stability_split2" in binding
  assert np.isfinite(interface["total"])
  assert np.isfinite(binding["dg_bind"]["total"])


def test_dg_bind_matches_subtraction():
  pdb_path = str(FILES / "dimer_af2.pdb")
  binding = calculate_binding(pdb_path, split1=["A"], split2=["B"])
  full = binding["stability_complex"]
  s1 = binding["stability_split1"]
  s2 = binding["stability_split2"]
  dg_bind = binding["dg_bind"]
  for key in full.keys():
    expected = full.get(key, 0.0) - s1.get(key, 0.0) - s2.get(key, 0.0)
    assert dg_bind[key] == pytest.approx(expected)


def test_debug_structure_smoke():
  stats = _debug_evoef2_structure(str(FILES / "dimer_af2.pdb"))
  assert stats["total_atoms"] > 0
  assert stats["valid_atoms"] > 0
  assert stats["protein_residues"] > 0


def test_na_binding_dna_dna_smoke():
  data = calculate_binding(
    str(FILES / "1nkp_mycmax_with_hydrogens.pdb"),
    split1=["H"],
    split2=["J"],
  )
  assert "interface" in data and "total" in data["interface"]
  assert np.isfinite(data["interface"]["total"])


def test_na_binding_protein_dna_smoke():
  data = calculate_binding(
    str(FILES / "1nkp_mycmax_with_hydrogens.pdb"),
    split1=["D"],
    split2=["J"],
  )
  assert "interface" in data and "total" in data["interface"]
  assert np.isfinite(data["interface"]["total"])


def test_rna_stability_smoke():
  data = calculate_stability(str(FILES / "rna_monomer_1.cif"))
  assert "total" in data
  assert np.isfinite(data["total"])


def _debug_evoef2_structure(
  structure,
  *,
  param_path=None,
  topo_path=None,
  dunbrack_path=None,
):
  topologies = evoef2.load_topology(topo_path)
  dun = evoef2.load_dunbrack(dunbrack_path)
  evo_struct = evoef2.rebuild_missing_atoms(structure, param_path=param_path, topo_path=topo_path)
  for chain in evo_struct.chains:
    if chain.is_protein:
      evoef2._calc_phi_psi(chain)
      for res in chain.residues:
        evoef2._residue_calc_sidechain_torsions(res, topologies)

  total_atoms = 0
  valid_atoms = 0
  missing_atoms = 0
  missing_h_atoms = 0
  hb_h_atoms = 0
  hb_a_atoms = 0
  residues_with_default_phipsi = 0
  protein_residues = 0
  torsion_expected = 0
  torsion_missing = 0
  dunbrack_bins = 0
  dunbrack_missing = 0

  for chain in evo_struct.chains:
    for res in chain.residues:
      if res.is_protein:
        protein_residues += 1
        if res.phipsi == (-60.0, 60.0):
          residues_with_default_phipsi += 1
        expected = evoef2._DUNBRACK_TORSION_COUNT.get(res.name, 0)
        torsion_expected += expected
        if expected > 0 and len(res.xtorsions) == 0:
          torsion_missing += 1
        phi = int(res.phipsi[0])
        psi = int(res.phipsi[1])
        bin_index = ((phi + 180) // 10) * 36 + ((psi + 180) // 10)
        if 0 <= bin_index < len(dun.bins):
          if res.name in dun.bins[bin_index].by_residue:
            dunbrack_bins += 1
          else:
            dunbrack_missing += 1

      for atom in res.atoms.values():
        total_atoms += 1
        if atom.is_xyz_valid:
          valid_atoms += 1
        else:
          missing_atoms += 1
          if atom.is_h:
            missing_h_atoms += 1
        if atom.is_hbond_h:
          hb_h_atoms += 1
        if atom.is_hbond_a:
          hb_a_atoms += 1

  return {
    "total_atoms": total_atoms,
    "valid_atoms": valid_atoms,
    "missing_atoms": missing_atoms,
    "missing_h_atoms": missing_h_atoms,
    "hb_h_atoms": hb_h_atoms,
    "hb_a_atoms": hb_a_atoms,
    "protein_residues": protein_residues,
    "default_phipsi_residues": residues_with_default_phipsi,
    "torsion_expected_total": torsion_expected,
    "torsion_missing_residues": torsion_missing,
    "dunbrack_bins_with_residue": dunbrack_bins,
    "dunbrack_bins_missing_residue": dunbrack_missing,
  }
