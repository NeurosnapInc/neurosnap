from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from neurosnap.chemistry import (
  align_molecule_to_reference,
  calculate_distance_matrix,
  calculate_rmsd,
  get_mol_center,
  move_ligand_to_center,
  translate_molecule,
)
from tests._structure_test_utils import parse_single_model

TESTS_DIR = Path(__file__).resolve().parents[1]
FILES = TESTS_DIR / "files"

# -----------------------------
# get_mol_center
# -----------------------------


def test_get_mol_center_geom_and_mass():
  sdf = FILES / "input_ligand.sdf"
  assert sdf.exists(), "Fixture ligand SDF missing"
  suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
  mol = suppl[0]
  assert mol is not None

  c_geom = get_mol_center(mol, use_mass=False)
  c_mass = get_mol_center(mol, use_mass=True)

  assert isinstance(c_geom, np.ndarray) and c_geom.shape == (3,)
  assert isinstance(c_mass, np.ndarray) and c_mass.shape == (3,)
  # centers should be finite numbers
  assert np.isfinite(c_geom).all()
  assert np.isfinite(c_mass).all()


# -----------------------------
# move_ligand_to_center
# -----------------------------


def test_move_ligand_to_center_aligns_centers(tmp_path: Path):
  ligand = FILES / "input_ligand.sdf"
  receptor = FILES / "1BTL.pdb"
  out = tmp_path / "centered_ligand.sdf"

  # run function
  res_path = move_ligand_to_center(str(ligand), str(receptor), str(out), use_mass=False)
  assert Path(res_path).exists()

  # compute centers to verify alignment
  r_center = parse_single_model(receptor).calculate_center_of_mass()

  # ligand new center via RDKit
  suppl = Chem.SDMolSupplier(str(out), removeHs=False)
  mol = suppl[0]
  assert mol is not None
  l_center = get_mol_center(mol, use_mass=False)

  # centers should match closely
  assert np.linalg.norm(r_center - l_center) < 1e-3


def test_calculate_distance_matrix_returns_square_symmetric_matrix():
  mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
  AllChem.EmbedMolecule(mol, randomSeed=7)

  distance_matrix = calculate_distance_matrix(mol)

  assert distance_matrix.shape == (mol.GetNumAtoms(), mol.GetNumAtoms())
  assert np.allclose(distance_matrix, distance_matrix.T)
  assert np.allclose(np.diag(distance_matrix), 0.0)


def test_calculate_rmsd_is_zero_for_translated_identical_molecule():
  mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
  AllChem.EmbedMolecule(mol, randomSeed=11)

  translated = translate_molecule(mol, [3.0, -2.0, 1.5])

  assert calculate_rmsd(mol, translated) == pytest.approx(0.0, abs=1e-6)


def test_translate_molecule_returns_shifted_copy():
  mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
  AllChem.EmbedMolecule(mol, randomSeed=13)
  shift = np.array([1.0, 2.0, -0.5])

  translated = translate_molecule(mol, shift)

  orig_conf = mol.GetConformer()
  moved_conf = translated.GetConformer()
  for i in range(mol.GetNumAtoms()):
    orig = np.array(orig_conf.GetAtomPosition(i))
    moved = np.array(moved_conf.GetAtomPosition(i))
    assert np.allclose(moved - orig, shift)


def test_align_molecule_to_reference_matches_reference_coordinates():
  ref = Chem.AddHs(Chem.MolFromSmiles("CCO"))
  AllChem.EmbedMolecule(ref, randomSeed=17)

  moved = translate_molecule(ref, [8.0, -4.0, 2.0])
  aligned = align_molecule_to_reference(moved, ref)

  aligned_conf = aligned.GetConformer()
  ref_conf = ref.GetConformer()
  for i in range(ref.GetNumAtoms()):
    aligned_pos = np.array(aligned_conf.GetAtomPosition(i))
    ref_pos = np.array(ref_conf.GetAtomPosition(i))
    assert np.allclose(aligned_pos, ref_pos, atol=1e-4)
