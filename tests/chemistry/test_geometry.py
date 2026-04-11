from pathlib import Path

import numpy as np
from rdkit import Chem

from neurosnap.chemistry import get_mol_center, move_ligand_to_center
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
