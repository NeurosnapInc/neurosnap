"""Core container, selector, and geometry tests for structure objects."""

import numpy as np
import pandas as pd
import pytest

from neurosnap.structure import (
  ca_distance_matrix,
  calculate_distance_matrix,
  calculate_protein_volume,
  calculate_surface_area,
  get_backbone,
  remove_non_biopolymers,
  remove_waters,
  select_residues,
)

from tests._structure_test_utils import PDB_DIMER, PDB_MONO, PDB_NO_H, parse_ensemble, parse_single_model


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


def test_filters_require_structure():
  ensemble = parse_ensemble(PDB_MONO)
  with pytest.raises(TypeError):
    remove_waters(ensemble)
  with pytest.raises(TypeError):
    remove_non_biopolymers(ensemble)


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
  distances = structure.distances_from(center_of_mass)
  assert distances.ndim == 1 and distances.size > 0
  radius_of_gyration = structure.calculate_rog(center=center_of_mass)
  assert radius_of_gyration > 0.0


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
