"""Interface and interaction tests for structure objects."""

import pytest

from neurosnap.structure import (
  calculate_bsa,
  calculate_hydrogen_bonds,
  calculate_interface_hydrogen_bonding_residues,
  find_disulfide_bonds,
  find_hydrophobic_residues,
  find_interface_contacts,
  find_interface_residues,
  find_non_interface_hydrophobic_patches,
  find_salt_bridges,
)

from tests._structure_test_utils import PDB_DIMER, PDB_MONO, PDB_WITH_H, make_structure, parse_ensemble, parse_single_model


def test_find_interface_contacts_hydrogen_filtering():
  structure = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("H", "ALA", "A", 1, 0.5, 0.0, 0.0, "H"),
      ("CA", "ALA", "B", 1, 0.0, 0.0, 3.0, "C"),
      ("O", "ALA", "B", 1, 10.0, 0.0, 0.0, "O"),
    ]
  )

  contacts_with_h = find_interface_contacts(structure, "A", "B", cutoff=4.5, hydrogens=True)
  contacts_no_h = find_interface_contacts(structure, "A", "B", cutoff=4.5, hydrogens=False)

  assert len(contacts_with_h) == 2
  assert len(contacts_no_h) == 1
  assert all(atom1.element != "H" and atom2.element != "H" for atom1, atom2 in contacts_no_h)
  assert {(atom1.chain_id, atom2.chain_id) for atom1, atom2 in contacts_with_h} == {("A", "B")}


def test_find_interface_residues_returns_pairs():
  structure = make_structure(
    [
      ("CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CB", "ALA", "A", 1, 0.5, 0.0, 0.0, "C"),
      ("CA", "ALA", "B", 1, 0.0, 0.0, 3.0, "C"),
      ("O", "ALA", "B", 1, 0.0, 0.5, 3.0, "O"),
      ("CA", "GLY", "A", 2, 0.0, 0.0, 10.0, "C"),
      ("CA", "GLY", "B", 2, 0.0, 0.0, 12.0, "C"),
      ("CA", "SER", "A", 3, 20.0, 0.0, 0.0, "C"),
      ("H", "SER", "B", 3, 20.0, 0.0, 2.0, "H"),
    ]
  )

  pairs_with_h = find_interface_residues(structure, "A", "B", cutoff=4.5, hydrogens=True)
  pairs_no_h = find_interface_residues(structure, "A", "B", cutoff=4.5, hydrogens=False)

  def keys(pairs):
    return {(residue_a.chain_id, residue_a.res_id, residue_b.chain_id, residue_b.res_id) for residue_a, residue_b in pairs}

  assert keys(pairs_with_h) == {("A", 1, "B", 1), ("A", 2, "B", 2), ("A", 3, "B", 3)}
  assert keys(pairs_no_h) == {("A", 1, "B", 1), ("A", 2, "B", 2)}


@pytest.mark.xfail(reason="Hydrophobic patch detection currently returns unhashable Residue views.", strict=True)
def test_find_non_interface_hydrophobic_patches_expected_sets_chain_a():
  structure = parse_single_model(PDB_DIMER)
  patches = find_non_interface_hydrophobic_patches(structure, [("A", "B")], target_chains=["A"])
  expected = {
    frozenset({("A", "LEU", 8), ("A", "MET", 45), ("A", "ALA", 46)}),
    frozenset({("A", "VAL", 38), ("A", "ALA", 33), ("A", "LEU", 35), ("A", "PRO", 34)}),
    frozenset({("A", "VAL", 40), ("A", "ILE", 41), ("A", "ILE", 42)}),
    frozenset({("A", "LEU", 56), ("A", "ILE", 52), ("A", "ALA", 57)}),
  }
  actual = {frozenset((residue.chain_id, residue.res_name, residue.res_id) for residue in patch) for patch in patches}
  assert len(patches) == 4
  assert actual == expected


@pytest.mark.xfail(reason="Hydrophobic patch detection currently returns unhashable Residue views.", strict=True)
def test_find_non_interface_hydrophobic_patches_expected_sets_chain_b():
  structure = parse_single_model(PDB_DIMER)
  patches = find_non_interface_hydrophobic_patches(structure, [("A", "B")], target_chains=["B"])
  expected = {
    frozenset({("B", "LEU", 8), ("B", "MET", 45), ("B", "ALA", 46)}),
    frozenset({("B", "VAL", 38), ("B", "ALA", 33), ("B", "LEU", 35), ("B", "PRO", 34)}),
    frozenset({("B", "VAL", 40), ("B", "ILE", 41), ("B", "ILE", 42)}),
    frozenset({("B", "LEU", 56), ("B", "ILE", 52), ("B", "ALA", 57)}),
  }
  actual = {frozenset((residue.chain_id, residue.res_name, residue.res_id) for residue in patch) for patch in patches}
  assert len(patches) == 4
  assert actual == expected


@pytest.mark.slow
def test_surface_area_positive():
  from neurosnap.structure import calculate_surface_area

  structure = parse_single_model(PDB_WITH_H)
  sasa = calculate_surface_area(structure, level="R")
  assert isinstance(sasa, float) and sasa >= 0.0


def test_hbond_errors():
  structure = parse_single_model(PDB_WITH_H)

  with pytest.raises(ValueError):
    calculate_hydrogen_bonds(structure, chain=None, chain_other="B")
  with pytest.raises(ValueError):
    calculate_hydrogen_bonds(structure, chain="Z")
  with pytest.raises(ValueError):
    calculate_hydrogen_bonds(structure, chain="A", chain_other="Z")


def test_interface_helpers_require_structure():
  ensemble = parse_ensemble(PDB_DIMER)

  with pytest.raises(TypeError):
    calculate_bsa(ensemble, ["A"], ["B"])
  with pytest.raises(TypeError):
    find_interface_contacts(ensemble, "A", "B")
  with pytest.raises(TypeError):
    find_interface_residues(ensemble, "A", "B")
  with pytest.raises(TypeError):
    find_non_interface_hydrophobic_patches(ensemble, [("A", "B")])


def test_interaction_helpers_require_structure():
  ensemble = parse_ensemble(PDB_MONO)

  with pytest.raises(TypeError):
    find_disulfide_bonds(ensemble)
  with pytest.raises(TypeError):
    find_salt_bridges(ensemble)
  with pytest.raises(TypeError):
    find_hydrophobic_residues(ensemble)
  with pytest.raises(TypeError):
    calculate_hydrogen_bonds(ensemble)
  with pytest.raises(TypeError):
    calculate_interface_hydrogen_bonding_residues(ensemble)


def test_find_disulfide_bonds_detects_close_cysteines():
  structure = make_structure(
    [
      ("SG", "CYS", "A", 1, 0.0, 0.0, 0.0, "S"),
      ("SG", "CYS", "A", 2, 1.9, 0.0, 0.0, "S"),
      ("SG", "CYS", "A", 3, 5.0, 0.0, 0.0, "S"),
    ]
  )

  bonds = find_disulfide_bonds(structure)
  assert {(res1.res_id, res2.res_id) for res1, res2 in bonds} == {(1, 2)}


def test_find_salt_bridges_and_hydrophobic_residues():
  structure = make_structure(
    [
      ("CA", "LYS", "A", 1, 0.0, 0.0, 0.0, "C"),
      ("CA", "ASP", "A", 2, 3.0, 0.0, 0.0, "C"),
      ("CA", "VAL", "A", 3, 10.0, 0.0, 0.0, "C"),
      ("CA", "SER", "B", 1, 20.0, 0.0, 0.0, "C"),
    ]
  )

  bridges = find_salt_bridges(structure)
  assert {(pos.res_id, neg.res_id) for pos, neg in bridges} == {(1, 2)}

  hydrophobic = find_hydrophobic_residues(structure)
  assert [(chain_id, residue.res_name, residue.res_id) for chain_id, residue in hydrophobic] == [("A", "VAL", 3)]


def test_calculate_interface_hydrogen_bonding_residues_counts_unique_residues():
  structure = make_structure(
    [
      ("N", "SER", "A", 1, 0.0, 0.0, 0.0, "N"),
      ("H", "SER", "A", 1, 0.0, 0.0, -1.0, "H"),
      ("O", "ASP", "B", 2, 0.0, 0.0, 2.8, "O"),
    ]
  )

  assert calculate_hydrogen_bonds(structure, chain="A", chain_other="B") == 1
  assert calculate_interface_hydrogen_bonding_residues(structure, chain="A", chain_other="B") == 2


@pytest.mark.slow
def test_calculate_bsa_dimer():
  structure = parse_single_model(PDB_DIMER)
  chains = [chain.chain_id for chain in structure.chains()]
  assert len(chains) >= 2

  buried_surface_area = calculate_bsa(structure, [chains[0]], chains[1:])
  assert isinstance(buried_surface_area, float) and buried_surface_area >= 0.0
