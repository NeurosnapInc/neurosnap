"""Core container, selector, and geometry tests for structure objects."""

import numpy as np
import pandas as pd
import pytest

from neurosnap.structure.structure import Structure
from neurosnap.structure import (
  ca_distance_matrix,
  calculate_distance_matrix,
  calculate_protein_volume,
  calculate_surface_area,
  fix_nucleic_termini,
  get_backbone,
  remove_atoms,
  remove_chains,
  remove_non_biopolymers,
  remove_waters,
  select_residues,
)

from tests._structure_test_utils import PDB_DIMER, PDB_MONO, PDB_NO_H, parse_ensemble, parse_single_model


def _make_two_atom_structure(elements, coords, hetero=None):
  structure = Structure(remove_annotations=False)
  atom_count = len(elements)
  structure.atoms = np.zeros(atom_count, dtype=structure._dtype_atoms)
  structure.atom_annotations = np.zeros(atom_count, dtype=structure._dtype_atom_annotations)
  hetero = [False] * atom_count if hetero is None else hetero

  for atom_index, ((x, y, z), element, is_hetero) in enumerate(zip(coords, elements, hetero), start=1):
    structure.atoms["x"][atom_index - 1] = x
    structure.atoms["y"][atom_index - 1] = y
    structure.atoms["z"][atom_index - 1] = z
    structure.atom_annotations["chain_id"][atom_index - 1] = "A"
    structure.atom_annotations["res_id"][atom_index - 1] = atom_index
    structure.atom_annotations["ins_code"][atom_index - 1] = ""
    structure.atom_annotations["res_name"][atom_index - 1] = "LIG" if is_hetero else "GLY"
    structure.atom_annotations["hetero"][atom_index - 1] = is_hetero
    structure.atom_annotations["atom_name"][atom_index - 1] = f"{element}{atom_index}"
    structure.atom_annotations["element"][atom_index - 1] = element
    structure.atom_annotations["atom_id"][atom_index - 1] = atom_index
    structure.atom_annotations["b_factor"][atom_index - 1] = 0.0
    structure.atom_annotations["occupancy"][atom_index - 1] = 1.0
    structure.atom_annotations["charge"][atom_index - 1] = 0
    structure.atom_annotations["sym_id"][atom_index - 1] = ""

  structure.bonds = np.zeros(0, dtype=structure._dtype_bond)
  return structure


def _make_structure_from_records(records):
  structure = Structure(remove_annotations=False)
  atom_count = len(records)
  structure.atoms = np.zeros(atom_count, dtype=structure._dtype_atoms)
  structure.atom_annotations = np.zeros(atom_count, dtype=structure._dtype_atom_annotations)

  for atom_index, record in enumerate(records, start=1):
    structure.atoms["x"][atom_index - 1] = float(atom_index)
    structure.atoms["y"][atom_index - 1] = 0.0
    structure.atoms["z"][atom_index - 1] = 0.0
    structure.atom_annotations["chain_id"][atom_index - 1] = record.get("chain_id", "A")
    structure.atom_annotations["res_id"][atom_index - 1] = record["res_id"]
    structure.atom_annotations["ins_code"][atom_index - 1] = record.get("ins_code", "")
    structure.atom_annotations["res_name"][atom_index - 1] = record["res_name"]
    structure.atom_annotations["hetero"][atom_index - 1] = record.get("hetero", False)
    structure.atom_annotations["atom_name"][atom_index - 1] = record["atom_name"]
    structure.atom_annotations["element"][atom_index - 1] = record.get("element", record["atom_name"][0])
    structure.atom_annotations["atom_id"][atom_index - 1] = atom_index
    structure.atom_annotations["b_factor"][atom_index - 1] = 0.0
    structure.atom_annotations["occupancy"][atom_index - 1] = 1.0
    structure.atom_annotations["charge"][atom_index - 1] = 0
    structure.atom_annotations["sym_id"][atom_index - 1] = ""

  structure.bonds = np.zeros(0, dtype=structure._dtype_bond)
  return structure


def test_parse_local_pdb_and_dataframe():
  ensemble = parse_ensemble(PDB_MONO)
  model_id = ensemble.model_ids[0]
  structure = ensemble[model_id]

  assert ensemble.model_ids
  assert structure.chains()
  dataframe = structure.to_dataframe()
  assert isinstance(dataframe, pd.DataFrame) and not dataframe.empty
  assert "atom_name" in dataframe.columns
  assert "x" in dataframe.columns
  assert "<Structure Ensemble:" in repr(ensemble)
  assert "<Structure:" in repr(structure)


def test_models_chains_and_sequence():
  ensemble = parse_ensemble(PDB_DIMER)
  structure = ensemble[ensemble.model_ids[0]]
  chains = structure.chains()

  assert chains
  sequence = chains[0].sequence(polymer_type="protein")
  assert isinstance(sequence, str)
  assert len(sequence) >= 1


def test_structure_is_subscriptable_by_chain_id():
  structure = parse_single_model(PDB_MONO)
  chain = structure.chains()[0]

  selected_chain = structure[chain.chain_id]
  assert selected_chain == chain
  assert selected_chain.chain_id == chain.chain_id

  with pytest.raises(KeyError):
    structure["missing"]


def test_chain_is_subscriptable_by_residue_id(caplog):
  structure = Structure(remove_annotations=False)
  structure.atoms = np.zeros(3, dtype=structure._dtype_atoms)
  structure.atom_annotations = np.zeros(3, dtype=structure._dtype_atom_annotations)
  structure.bonds = np.zeros(0, dtype=structure._dtype_bond)

  for atom_index, (res_id, ins_code) in enumerate([(10, ""), (10, "A"), (11, "")], start=1):
    structure.atoms["x"][atom_index - 1] = float(atom_index)
    structure.atoms["y"][atom_index - 1] = 0.0
    structure.atoms["z"][atom_index - 1] = 0.0
    structure.atom_annotations["chain_id"][atom_index - 1] = "A"
    structure.atom_annotations["res_id"][atom_index - 1] = res_id
    structure.atom_annotations["ins_code"][atom_index - 1] = ins_code
    structure.atom_annotations["res_name"][atom_index - 1] = "GLY"
    structure.atom_annotations["hetero"][atom_index - 1] = False
    structure.atom_annotations["atom_name"][atom_index - 1] = f"C{atom_index}"
    structure.atom_annotations["element"][atom_index - 1] = "C"
    structure.atom_annotations["atom_id"][atom_index - 1] = atom_index
    structure.atom_annotations["b_factor"][atom_index - 1] = 0.0
    structure.atom_annotations["occupancy"][atom_index - 1] = 1.0
    structure.atom_annotations["charge"][atom_index - 1] = 0
    structure.atom_annotations["sym_id"][atom_index - 1] = ""

  chain = structure["A"]
  residue = chain[10]
  assert residue.res_id == 10
  assert residue.ins_code == ""
  assert any("multiple residues with residue ID 10" in message for message in caplog.messages)

  assert chain[11].res_id == 11
  with pytest.raises(KeyError):
    chain[99]


def test_structure_and_chain_iteration_still_work():
  structure = parse_single_model(PDB_MONO)

  chains = list(structure)
  assert chains
  assert chains[0] == structure[chains[0].chain_id]

  residues = list(chains[0])
  assert residues
  assert residues[0] == chains[0][residues[0].res_id]


def test_select_residues_parsing_and_invert():
  structure = parse_single_model(PDB_NO_H)
  chain = structure.chains()[0].chain_id
  dataframe = structure.to_dataframe()
  residue_id = int(dataframe[dataframe["chain"] == chain]["res_id"].iloc[0])

  selection = select_residues(structure, f"{chain}{residue_id}")
  assert chain in selection and residue_id in selection[chain]

  chain_selection = select_residues(structure, chain)
  assert chain in chain_selection and chain_selection[chain]

  inverted = select_residues(structure, f"{chain}{residue_id}", invert=True)
  assert chain in inverted and residue_id not in inverted[chain]

  with pytest.raises(ValueError):
    select_residues(structure, "Z999")


def test_select_residues_requires_structure():
  ensemble = parse_ensemble(PDB_MONO)
  with pytest.raises(TypeError):
    select_residues(ensemble, "A1")


def test_renumber_updates_dataframe():
  structure = parse_single_model(PDB_NO_H)
  structure.renumber(start=1)

  dataframe = structure.to_dataframe()
  assert dataframe["res_id"].min() == 1
  chain = structure.chains()[0].chain_id
  residue_ids = dataframe[dataframe["chain"] == chain]["res_id"].to_numpy()
  assert np.all(np.diff(np.unique(residue_ids)) >= 0)


def test_remove_waters_and_non_biopolymers():
  structure = parse_single_model(PDB_MONO)
  remove_waters(structure)
  assert not structure.to_dataframe()["res_name"].isin(["HOH", "WAT"]).any()

  structure = parse_single_model(PDB_MONO)
  remove_non_biopolymers(structure)
  heterogens = structure.to_dataframe().query("res_type == 'HETEROGEN'")
  assert heterogens.empty


def test_remove_chains_removes_matching_chain():
  structure = _make_structure_from_records(
    [
      {"chain_id": "A", "res_id": 1, "res_name": "GLY", "atom_name": "CA", "element": "C"},
      {"chain_id": "A", "res_id": 1, "res_name": "GLY", "atom_name": "N", "element": "N"},
      {"chain_id": "B", "res_id": 1, "res_name": "GLY", "atom_name": "CA", "element": "C"},
      {"chain_id": "B", "res_id": 1, "res_name": "GLY", "atom_name": "N", "element": "N"},
    ]
  )

  remove_chains(structure, lambda chain_view: chain_view.chain_id == "B")

  assert structure.chain_ids() == ["A"]
  assert set(structure.to_dataframe()["chain"]) == {"A"}


def test_remove_atoms_removes_matching_atoms_with_optional_chain_scope():
  structure = _make_structure_from_records(
    [
      {"chain_id": "A", "res_id": 1, "res_name": "GLY", "atom_name": "N", "element": "N"},
      {"chain_id": "A", "res_id": 1, "res_name": "GLY", "atom_name": "CA", "element": "C"},
      {"chain_id": "B", "res_id": 1, "res_name": "GLY", "atom_name": "N", "element": "N"},
      {"chain_id": "B", "res_id": 1, "res_name": "GLY", "atom_name": "CA", "element": "C"},
    ]
  )

  remove_atoms(structure, lambda atom: atom.atom_name.strip().upper() == "N", chain="A")

  dataframe = structure.to_dataframe()
  assert dataframe.query("chain == 'A'")["atom_name"].tolist() == ["CA"]
  assert sorted(dataframe.query("chain == 'B'")["atom_name"].tolist()) == ["CA", "N"]


def test_fix_nucleic_termini_normalizes_and_strips_terminal_atoms():
  structure = _make_structure_from_records(
    [
      {"res_id": 1, "res_name": "A", "atom_name": "P", "element": "P"},
      {"res_id": 1, "res_name": "A", "atom_name": "O1P", "element": "O"},
      {"res_id": 1, "res_name": "A", "atom_name": "O2P", "element": "O"},
      {"res_id": 1, "res_name": "A", "atom_name": "C1'", "element": "C"},
      {"res_id": 2, "res_name": "U", "atom_name": "P", "element": "P"},
      {"res_id": 2, "res_name": "U", "atom_name": "O1P", "element": "O"},
      {"res_id": 2, "res_name": "U", "atom_name": "C1'", "element": "C"},
    ]
  )

  fix_nucleic_termini(structure)

  atom_names = structure.to_dataframe()["atom_name"].tolist()
  assert "O1P" not in atom_names
  assert "O2P" not in atom_names
  assert atom_names.count("P") == 1
  assert atom_names.count("OP1") == 1
  assert atom_names.count("C1'") == 2


def test_fix_nucleic_termini_optionally_strips_3prime_and_gap_starts_new_segment():
  structure = _make_structure_from_records(
    [
      {"res_id": 1, "res_name": "A", "atom_name": "C1'", "element": "C"},
      {"res_id": 2, "res_name": "U", "atom_name": "O3P", "element": "O"},
      {"res_id": 2, "res_name": "U", "atom_name": "C1'", "element": "C"},
      {"res_id": 4, "res_name": "G", "atom_name": "P", "element": "P"},
      {"res_id": 4, "res_name": "G", "atom_name": "OP3", "element": "O"},
      {"res_id": 4, "res_name": "G", "atom_name": "C1'", "element": "C"},
    ]
  )

  fix_nucleic_termini(structure, strip_3prime=True)

  atom_names = structure.to_dataframe()["atom_name"].tolist()
  assert "P" not in atom_names
  assert "OP3" not in atom_names
  assert "O3P" not in atom_names
  assert atom_names.count("C1'") == 3


def test_fix_nucleic_termini_warns_and_leaves_non_nucleic_structure_unchanged(caplog):
  structure = _make_structure_from_records(
    [
      {"chain_id": "A", "res_id": 1, "res_name": "GLY", "atom_name": "N", "element": "N"},
      {"chain_id": "A", "res_id": 1, "res_name": "GLY", "atom_name": "CA", "element": "C"},
      {"chain_id": "A", "res_id": 201, "res_name": "ZN", "atom_name": "ZN", "element": "ZN", "hetero": True},
    ]
  )
  original_atom_names = structure.atom_annotations["atom_name"].copy()
  original_elements = structure.atom_annotations["element"].copy()

  fix_nucleic_termini(structure)

  assert np.array_equal(structure.atom_annotations["atom_name"], original_atom_names)
  assert np.array_equal(structure.atom_annotations["element"], original_elements)
  assert any("No nucleotide residues were found while running fix_nucleic_termini()" in message for message in caplog.messages)


def test_remove_non_biopolymers_removes_hetero_unk_ligand():
  structure = parse_single_model("tests/files/dimer_1lig_malformed.pdb")
  assert not structure.to_dataframe().query("chain == '' and res_id == 0 and res_name == 'UNK'").empty

  remove_non_biopolymers(structure)
  assert structure.to_dataframe().query("chain == '' and res_id == 0 and res_name == 'UNK'").empty


def test_filters_require_structure():
  ensemble = parse_ensemble(PDB_MONO)
  with pytest.raises(TypeError):
    remove_waters(ensemble)
  with pytest.raises(TypeError):
    remove_non_biopolymers(ensemble)
  with pytest.raises(TypeError):
    fix_nucleic_termini(ensemble)
  with pytest.raises(TypeError):
    remove_chains(ensemble, lambda chain_view: True)
  with pytest.raises(TypeError):
    remove_atoms(ensemble, lambda atom: True)


def test_get_backbone_and_distance_matrix_and_center_of_mass_and_rg():
  structure = parse_single_model(PDB_NO_H)

  backbone = get_backbone(structure)
  assert backbone.ndim == 2 and backbone.shape[1] == 3 and backbone.shape[0] > 0

  distance_matrix = calculate_distance_matrix(structure)
  assert distance_matrix.ndim == 2 and distance_matrix.shape[0] == distance_matrix.shape[1]
  assert np.allclose(np.diag(distance_matrix), 0.0)
  assert np.allclose(distance_matrix, ca_distance_matrix(structure))

  surface_area = calculate_surface_area(structure, level="R")
  assert isinstance(surface_area, float) and surface_area >= 0.0

  volume = calculate_protein_volume(structure)
  assert isinstance(volume, float) and volume >= 0.0

  center_of_mass = structure.calculate_center_of_mass()
  assert isinstance(center_of_mass, np.ndarray) and center_of_mass.shape == (3,)
  geometric_center = structure.calculate_geometric_center()
  assert isinstance(geometric_center, np.ndarray) and geometric_center.shape == (3,)
  distances = structure.distances_from(center_of_mass)
  assert distances.ndim == 1 and distances.size > 0
  radius_of_gyration = structure.calculate_rog(center=center_of_mass)
  assert radius_of_gyration > 0.0

  first_chain = structure.chains()[0].chain_id
  chain_center = structure.calculate_geometric_center(chains=[first_chain])
  dataframe = structure.to_dataframe().query("chain == @first_chain")
  expected_center = dataframe[["x", "y", "z"]].to_numpy(dtype=np.float32).mean(axis=0)
  assert np.allclose(chain_center, expected_center)


def test_calculate_center_of_mass_includes_small_molecules():
  structure = _make_two_atom_structure(
    elements=["C", "Fe"],
    coords=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
    hetero=[False, True],
  )

  center_of_mass = structure.calculate_center_of_mass()
  expected_x = (12.011 * 0.0 + 55.845 * 10.0) / (12.011 + 55.845)
  assert np.allclose(center_of_mass, np.array([expected_x, 0.0, 0.0], dtype=np.float32))


def test_calculate_center_of_mass_warns_on_unknown_element(caplog):
  structure = _make_two_atom_structure(
    elements=["C", "Xx"],
    coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    hetero=[False, True],
  )

  with pytest.raises(ValueError, match="Unknown element mass"):
    structure.calculate_center_of_mass()

  assert any("Unknown element mass" in message for message in caplog.messages)


def test_analysis_helpers_require_structure():
  ensemble = parse_ensemble(PDB_MONO)
  with pytest.raises(TypeError):
    get_backbone(ensemble)
  with pytest.raises(TypeError):
    calculate_distance_matrix(ensemble)
  with pytest.raises(TypeError):
    ca_distance_matrix(ensemble)
  with pytest.raises(TypeError):
    calculate_surface_area(ensemble)
  with pytest.raises(TypeError):
    calculate_protein_volume(ensemble)
