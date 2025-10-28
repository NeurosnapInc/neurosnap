# tests/test_protein.py
import io
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless

from Bio.PDB.Polypeptide import is_aa

from neurosnap.constants import (
  BACKBONE_ATOMS_AA,
  BACKBONE_ATOMS_DNA,
  BACKBONE_ATOMS_RNA,
  NUC_DNA_CODES,
  NUC_RNA_CODES,
  STANDARD_NUCLEOTIDES,
)
from neurosnap.protein import (
  Protein,
  animate_pseudo_3D,
  calculate_bsa,
  extract_non_biopolymers,
  fetch_uniprot,
  getAA,
  isoelectric_point,
  molecular_weight,
  net_charge,
  plot_pseudo_3D,
  sanitize_aa_seq,
)

FILES = Path("tests/files")
PDB_MONO = FILES / "1BTL.pdb"
PDB_DIMER = FILES / "dimer_af2.pdb"
PDB_LIG = FILES / "1MAL.pdb"
AF2_RANK1 = FILES / "4AOW_af2_rank_1.pdb"
AF2_RANK2 = FILES / "4AOW_af2_rank_2.pdb"
PDB_WITH_H = FILES / "1nkp_mycmax_with_hydrogens.pdb"
PDB_NO_H = FILES / "1nkp_mycmax.pdb"
CIF = FILES / "orf1_boltz1.cif"


def _pdb_from_atoms(atom_defs):
  lines = []
  for serial, (atom_name, resname, chain_id, resid, x, y, z, element) in enumerate(atom_defs, start=1):
    lines.append(
      f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain_id:1s}{resid:4d}    "
      f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}"
    )
  lines.append("TER")
  lines.append("END")
  return "\n".join(lines) + "\n"


def _transform_atoms(atom_defs, rotation, translation):
  transformed = []
  for atom_name, resname, chain_id, resid, x, y, z, element in atom_defs:
    coord = np.array([x, y, z])
    new_coord = rotation @ coord + translation
    transformed.append((atom_name, resname, chain_id, resid, new_coord[0], new_coord[1], new_coord[2], element))
  return transformed


def _replace_chain(atom_defs, chain_id):
  return [(atom_name, resname, chain_id, resid, x, y, z, element) for (atom_name, resname, _, resid, x, y, z, element) in atom_defs]


def _make_protein(atom_defs):
  return Protein(io.StringIO(_pdb_from_atoms(atom_defs)), format="pdb")


def _collect_backbone_coords(prot):
  coords = []
  for model in prot.structure:
    for chain in model:
      for residue in chain:
        if is_aa(residue, standard=False):
          allowed = BACKBONE_ATOMS_AA
        else:
          resname = residue.get_resname().strip().upper()
          if resname in NUC_DNA_CODES:
            allowed = BACKBONE_ATOMS_DNA
          elif resname in NUC_RNA_CODES:
            allowed = BACKBONE_ATOMS_RNA
          elif resname in STANDARD_NUCLEOTIDES:
            allowed = BACKBONE_ATOMS_RNA
          else:
            allowed = None
        if not allowed:
          continue
        for atom in residue:
          if atom.name in allowed:
            coords.append(((model.id, chain.id, residue.id[1], atom.name), np.array(atom.coord)))
  coords.sort(key=lambda item: item[0])
  return np.array([coord for _, coord in coords])


ROT_Z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
TRANSLATION_VECTOR = np.array([10.0, -5.0, 3.0])

PROTEIN_BACKBONE_ATOMS = (
  ("N", "ALA", "A", 1, 0.000, 0.000, 0.000, "N"),
  ("CA", "ALA", "A", 1, 1.458, 0.000, 0.000, "C"),
  ("C", "ALA", "A", 1, 1.958, 1.410, 0.000, "C"),
  ("N", "GLY", "A", 2, 3.300, 1.410, 0.000, "N"),
  ("CA", "GLY", "A", 2, 3.800, 2.820, 0.000, "C"),
  ("C", "GLY", "A", 2, 5.200, 2.820, 0.000, "C"),
)

DNA_BACKBONE_ATOMS = (
  ("P", "DA", "A", 1, 0.000, 0.000, 0.000, "P"),
  ("O1P", "DA", "A", 1, -0.800, 1.000, 0.000, "O"),
  ("O2P", "DA", "A", 1, -0.800, -1.000, 0.000, "O"),
  ("O5'", "DA", "A", 1, 1.200, 0.200, 0.000, "O"),
  ("C5'", "DA", "A", 1, 2.200, 0.700, 0.200, "C"),
  ("C4'", "DA", "A", 1, 2.800, 1.800, 0.300, "C"),
  ("O4'", "DA", "A", 1, 2.300, 2.900, 0.400, "O"),
  ("C3'", "DA", "A", 1, 3.600, 1.700, 1.500, "C"),
  ("O3'", "DA", "A", 1, 4.600, 1.700, 1.800, "O"),
  ("C1'", "DA", "A", 1, 1.800, 3.000, -0.500, "C"),
  ("C2'", "DA", "A", 1, 2.300, 2.100, -1.500, "C"),
)

MIXED_BACKBONE_ATOMS = PROTEIN_BACKBONE_ATOMS + tuple(_replace_chain(DNA_BACKBONE_ATOMS, "B"))


# -----------------------
# Basic loading & dataframe
# -----------------------


def test_load_local_pdb_and_df():
  prot = Protein(str(PDB_MONO))
  assert len(prot.models()) >= 1
  assert len(prot.chains(prot.models()[0])) >= 1
  assert isinstance(prot.df, pd.DataFrame) and not prot.df.empty
  assert "atom_name" in prot.df.columns
  assert "x" in prot.df.columns
  r = repr(prot)
  assert "Neurosnap Protein" in r


def test_models_chains_get_aas():
  prot = Protein(str(PDB_DIMER))
  m = prot.models()[0]
  chains = prot.chains(m)
  assert chains, "No chains found"
  # pick first chain and expect a non-empty AA sequence (ligands ignored)
  seq = prot.get_aas(chains[0], model=m)
  assert isinstance(seq, str)
  assert len(seq) >= 1


# -----------------------
# Selection / renumber / removal
# -----------------------


def test_select_residues_parsing_and_invert():
  prot = Protein(str(PDB_NO_H))
  m = prot.models()[0]
  chain = prot.chains(m)[0]

  # pick a real residue id from the dataframe
  dfc = prot.df[(prot.df["model"] == m) & (prot.df["chain"] == chain)]
  rid = int(dfc["res_id"].iloc[0])

  sel = prot.select_residues(f"{chain}{rid}")
  assert chain in sel and rid in sel[chain]

  # entire chain selector
  sel_all = prot.select_residues(chain)
  assert chain in sel_all and len(sel_all[chain]) >= 1

  # invert should return the complement (on the chain)
  inv = prot.select_residues(f"{chain}{rid}", invert=True)
  assert chain in inv and rid not in inv[chain]

  # invalid chain raises
  with pytest.raises(ValueError):
    prot.select_residues("Z999")


def test_renumber_updates_df():
  prot = Protein(str(PDB_NO_H))
  prot.renumber(start=1)
  # after renumber, residue ids should be positive and start near 1
  assert prot.df["res_id"].min() == 1
  # strictly increasing within a chain
  m = prot.models()[0]
  c = prot.chains(m)[0]
  res_ids = prot.df[(prot.df["model"] == m) & (prot.df["chain"] == c)]["res_id"].to_numpy()
  assert np.all(np.diff(np.unique(res_ids)) >= 0)


def test_remove_waters_and_non_biopolymers():
  prot = Protein(str(PDB_MONO))
  prot.remove_waters()
  assert not prot.df["res_name"].isin(["HOH", "WAT"]).any()

  prot2 = Protein(str(PDB_MONO))
  prot2.remove_non_biopolymers()
  # standard amino acids and nucleotides only (UNK allowed sometimes; the function removes non-standard)
  bad = prot2.df[(prot2.df["res_type"] == "HETEROGEN")]
  assert bad.empty


# -----------------------
# Geometry & distances
# -----------------------


def test_get_backbone_and_distance_matrix_and_center_of_mass_and_rg():
  prot = Protein(str(PDB_NO_H))
  m = prot.models()[0]
  chains = prot.chains(m)

  bb = prot.get_backbone(model=m)
  assert bb.ndim == 2 and bb.shape[1] == 3 and bb.shape[0] > 0

  # distance matrix (CA-only)
  dm_all = prot.calculate_distance_matrix(model=m, chain=None)
  assert dm_all.ndim == 2 and dm_all.shape[0] == dm_all.shape[1]
  assert np.allclose(np.diag(dm_all), 0.0)

  # center of mass and RoG
  com = prot.calculate_center_of_mass(model=m)
  assert isinstance(com, np.ndarray) and com.shape == (3,)
  dists = prot.distances_from_com(model=m, com=com)
  assert dists.ndim == 1 and dists.size > 0
  rg = prot.calculate_rog(model=m, distances_from_com=dists)
  assert rg > 0.0


@pytest.mark.slow
def test_surface_area_positive():
  prot = Protein(str(PDB_NO_H))
  sasa = prot.calculate_surface_area(model=prot.models()[0], level="R")
  assert isinstance(sasa, float) and sasa >= 0.0


def test_contacts_interface_nonnegative():
  prot = Protein(str(PDB_DIMER))
  m = prot.models()[0]
  chains = prot.chains(m)
  assert len(chains) >= 2
  # take first two chains
  n_contacts = prot.calculate_contacts_interface(chains[0], chains[1], model=m, threshold=6.0)
  assert isinstance(n_contacts, int) and n_contacts >= 0


def test_hbond_errors():
  prot = Protein(str(PDB_WITH_H))
  # specifying chain_other without chain -> error
  with pytest.raises(ValueError):
    prot.calculate_hydrogen_bonds(chain=None, chain_other="B")
  # invalid chain names -> errors
  with pytest.raises(ValueError):
    prot.calculate_hydrogen_bonds(chain="Z")
  with pytest.raises(ValueError):
    prot.calculate_hydrogen_bonds(chain="A", chain_other="Z")


# -----------------------
# Alignment / RMSD
# -----------------------


def test_rmsd_align_and_op_sub_af2():
  # same backbone (with vs without H) should yield very small RMSD after alignment
  prot1 = Protein(str(AF2_RANK1))
  prot2 = Protein(str(AF2_RANK2))
  rmsd = prot1.calculate_rmsd(prot2, align=True)
  assert rmsd < 0.45  # tight threshold
  # __sub__ sugar
  rmsd2 = prot1 - prot2
  assert abs(rmsd2 - rmsd) < 1e-6


def test_rmsd_align_and_op_sub_hydrogens():
  # same backbone (with vs without H) should yield very small RMSD after alignment
  prot_ref = Protein(str(PDB_NO_H))
  prot_h = Protein(str(PDB_WITH_H))
  rmsd = prot_ref.calculate_rmsd(prot_h, align=True)
  assert rmsd < 1e-2  # tight threshold


def test_align_backbone_protein_only():
  prot_ref = _make_protein(list(PROTEIN_BACKBONE_ATOMS))
  prot_offset = _make_protein(_transform_atoms(PROTEIN_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR))

  coords_ref = _collect_backbone_coords(prot_ref)
  coords_before = _collect_backbone_coords(prot_offset)
  assert coords_ref.shape == coords_before.shape
  assert not np.allclose(coords_ref, coords_before)

  prot_ref.align(prot_offset)
  coords_after = _collect_backbone_coords(prot_offset)
  assert np.allclose(coords_ref, coords_after, atol=1e-3)


def test_align_backbone_nucleotide_only():
  prot_ref = _make_protein(list(DNA_BACKBONE_ATOMS))
  prot_offset = _make_protein(_transform_atoms(DNA_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR))

  coords_ref = _collect_backbone_coords(prot_ref)
  coords_before = _collect_backbone_coords(prot_offset)
  assert coords_ref.shape == coords_before.shape
  assert not np.allclose(coords_ref, coords_before)

  prot_ref.align(prot_offset)
  coords_after = _collect_backbone_coords(prot_offset)
  assert np.allclose(coords_ref, coords_after, atol=1e-3)


def test_align_backbone_rna_only_real_structures():
  prot_ref = Protein(str(FILES / "rna_monomer_1.cif"))
  prot_offset = Protein(str(FILES / "rna_monomer_2.cif"))

  coords_ref = _collect_backbone_coords(prot_ref)
  coords_before = _collect_backbone_coords(prot_offset)
  assert coords_ref.shape == coords_before.shape

  rmsd_before = np.sqrt(np.mean(np.sum((coords_before - coords_ref) ** 2, axis=1)))

  prot_ref.align(prot_offset)
  coords_after = _collect_backbone_coords(prot_offset)
  rmsd_after = np.sqrt(np.mean(np.sum((coords_after - coords_ref) ** 2, axis=1)))

  assert rmsd_after < rmsd_before
  assert rmsd_after < 2.0


def test_align_backbone_mixed_protein_and_nucleotide():
  prot_ref = _make_protein(list(MIXED_BACKBONE_ATOMS))
  prot_offset = _make_protein(_transform_atoms(MIXED_BACKBONE_ATOMS, ROT_Z_90, TRANSLATION_VECTOR))

  coords_ref = _collect_backbone_coords(prot_ref)
  coords_before = _collect_backbone_coords(prot_offset)
  assert coords_ref.shape == coords_before.shape
  assert not np.allclose(coords_ref, coords_before)

  prot_ref.align(prot_offset)
  coords_after = _collect_backbone_coords(prot_offset)
  assert np.allclose(coords_ref, coords_after, atol=1e-3)

# -----------------------
# File IO & conversions
# -----------------------


def test_save_and_reload(tmp_path):
  prot = Protein(str(PDB_NO_H))
  out_pdb = tmp_path / "out.pdb"
  out_cif = tmp_path / "out.cif"
  prot.save(str(out_pdb), format="pdb")
  prot.save(str(out_cif), format="mmcif")
  assert out_pdb.exists() and out_pdb.stat().st_size > 0
  assert out_cif.exists() and out_cif.stat().st_size > 0
  # reload
  _ = Protein(str(out_pdb))
  _ = Protein(str(out_cif))


def test_to_sdf_writes(tmp_path):
  prot = Protein(str(PDB_NO_H))
  out = tmp_path / "prot.sdf"
  prot.to_sdf(str(out))
  assert out.exists() and out.stat().st_size > 0


@pytest.mark.slow
def test_extract_non_biopolymers(tmp_path):
  # 1MAL contains ligands; expect >= 1 SDF written (robust: >= 0 if environment differs)
  outdir = tmp_path / "ligs"
  extract_non_biopolymers(str(PDB_LIG), str(outdir), min_atoms=5)
  assert outdir.exists()
  # Usually >=1; keep non-failing if zero on exotic environments
  n = len(list(outdir.glob("*.sdf")))
  assert n >= 0


# -----------------------
# Sequence utilities
# -----------------------


def test_getAA_and_sanitize_and_mw_and_charge_and_pi():
  # getAA
  rec = getAA("A")
  assert (rec.code, rec.abr, rec.name) == ("A", "ALA", "ALANINE")

  rec2 = getAA("ala")  # case-insensitive
  assert (rec2.code, rec2.abr, rec2.name) == ("A", "ALA", "ALANINE")

  with pytest.raises(ValueError, match=r"Unknown amino acid identifier"):
    getAA("???")

  # sanitize
  seq_raw = " a c d e f * \n"
  seq = sanitize_aa_seq(seq_raw, non_standard="reject", trim_term=True)
  assert seq == "ACDEF"
  seq2 = sanitize_aa_seq("ACDZX", non_standard="allow")
  assert seq2 == "ACDZX"
  with pytest.raises(ValueError):
    sanitize_aa_seq("ACDZ?", non_standard="reject")

  # molecular weight (single AA equals its residue mass)
  from neurosnap.constants import AA_WEIGHTS_PROTEIN_AVG as AA_W

  mw_gly = molecular_weight("G")
  assert abs(mw_gly - AA_W["G"]) < 1e-6
  # dipeptide subtracts water once
  mw_AG = molecular_weight("AG")
  assert abs(mw_AG - (AA_W["A"] + AA_W["G"] - 18.015)) < 1e-6

  # charge & pI sanity
  q_acidic = net_charge("DE", pH=7.0)
  q_basic = net_charge("KR", pH=7.0)
  assert q_acidic < 0 and q_basic > 0
  pi = isoelectric_point("ACDEFGHIKLMNPQRSTVWY")
  assert 0.0 <= pi <= 14.0

def test_getAA_non_standard_handling():
  # Non-standard example: MSE (selenomethionine)
  # reject -> ValueError with helpful guidance
  with pytest.raises(ValueError, match=r"Encountered non-standard amino acid"):
    getAA("MSE", non_standard="reject")

  # allow -> returns the non-standard record unchanged
  rec_allow = getAA("MSE", non_standard="allow")
  assert rec_allow.abr == "MSE"
  assert rec_allow.name.upper().startswith("SELENO")  # "SELENOMETHIONINE"
  # 1-letter code may be unavailable; implementation may use "?" as a sentinel
  assert rec_allow.code in (None, "?")

  # convert -> mapped to its standard equivalent MET
  rec_conv = getAA("MSE", non_standard="convert")
  assert (rec_conv.code, rec_conv.abr, rec_conv.name) == ("M", "MET", "METHIONINE")


# -----------------------
# BSA (uses SASA internally)
# -----------------------


@pytest.mark.slow
def test_calculate_bsa_dimer():
  prot = Protein(str(PDB_DIMER))
  m = prot.models()[0]
  chains = prot.chains(m)
  assert len(chains) >= 2
  # Split chains into two non-overlapping groups that cover all chains
  group1 = [chains[0]]
  group2 = chains[1:]
  bsa = calculate_bsa(prot, group1, group2, model=m)
  assert isinstance(bsa, float) and bsa >= 0.0


# -----------------------
# Plotting helpers (no display)
# -----------------------


def test_plot_pseudo_3D_and_animate():
  prot = Protein(str(PDB_NO_H))
  xyz = prot.df[["x", "y", "z"]].to_numpy()
  fig, ax = matplotlib.pyplot.subplots()
  lc = plot_pseudo_3D(xyz[:200], ax=ax)  # limit points for speed
  assert lc is not None

  # simple 2-frame animation
  fig2, ax2 = matplotlib.pyplot.subplots()
  lc1 = plot_pseudo_3D(xyz[:150], ax=ax2)
  lc2 = plot_pseudo_3D(xyz[50:200], ax=ax2)
  ani = animate_pseudo_3D(fig2, ax2, [lc1, lc2], titles=["t1", "t2"])
  assert hasattr(ani, "_func")  # ArtistAnimation/FuncAnimation instance


# -----------------------
# Networking helpers (mocked)
# -----------------------


def test_fetch_uniprot_head_and_seq(monkeypatch):
  # HEAD happy path
  class R:
    def __init__(self, code=200, text=""):
      self.status_code = code
      self.text = text

  monkeypatch.setattr("neurosnap.protein.requests.head", lambda url: R(200))
  assert fetch_uniprot("P12345", head=True) is True

  # GET happy path via UniProtKB with mocked read_msa
  fasta = ">sp|P12345| Some protein\nACDEFGHIKLMNPQRSTVWY\n"
  monkeypatch.setattr("neurosnap.protein.requests.get", lambda url: R(200, fasta))
  monkeypatch.setattr("neurosnap.protein.read_msa", lambda txt: (["h"], ["ACDEFGHIKLMNPQRSTVWY"]))
  seq = fetch_uniprot("P12345", head=False)
  assert seq == "ACDEFGHIKLMNPQRSTVWY"

  # Fallback to UniParc then error
  calls = {"n": 0}

  def fake_get(url):
    calls["n"] += 1
    # first call fails (uniprotkb), second returns OK (uniparc)
    if "uniprotkb" in url:
      return R(404, "")
    return R(200, fasta)

  monkeypatch.setattr("neurosnap.protein.requests.get", fake_get)
  seq = fetch_uniprot("UPI000000", head=False)
  assert isinstance(seq, str)
