# tests/test_pdockq.py
from pathlib import Path

import numpy as np
import pytest

from neurosnap.algos.pdockq import (
  _calc_pdockq_from_arrays,
  _chain_cb_or_gly_ca,
  calculate_pDockQ,
)
from neurosnap.protein import Protein

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"


# -------------------------
# Unit tests for internals
# -------------------------


def test__chain_cb_or_gly_ca_returns_coords_and_plddt():
  prot = Protein(str(FILES / "1BTL.pdb"))  # single-chain protein (A)
  coords, plddt = _chain_cb_or_gly_ca(prot, "A")
  assert isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3
  assert isinstance(plddt, np.ndarray) and plddt.ndim == 1
  assert len(coords) == len(plddt)
  assert len(coords) > 0
  # pLDDT is stored in B-factor, should be finite
  assert np.isfinite(plddt).all()


def test__calc_pdockq_from_arrays_monotonic_in_plddt():
  # two residues per chain, very close => contacts present
  coords1 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
  coords2 = np.array([[0.0, 0.0, 6.0], [5.0, 0.0, 6.0]])
  chain_coords = {"A": coords1, "B": coords2}

  low = np.array([50.0, 50.0])
  high = np.array([90.0, 90.0])
  pd1, ppv1 = _calc_pdockq_from_arrays(chain_coords, {"A": low, "B": low}, dist_thresh=8.0)
  pd2, ppv2 = _calc_pdockq_from_arrays(chain_coords, {"A": high, "B": high}, dist_thresh=8.0)

  assert 0.0 <= pd1 <= 1.0 and 0.0 <= pd2 <= 1.0
  assert pd2 > pd1  # higher plDDT -> higher pDockQ
  assert 0.55 <= ppv1 <= 0.99 and 0.55 <= ppv2 <= 0.99
  assert ppv2 >= ppv1  # PPV should not decrease when pDockQ increases


def test__calc_pdockq_from_arrays_no_contacts_returns_zero():
  coords1 = np.array([[0.0, 0.0, 0.0]])
  coords2 = np.array([[100.0, 0.0, 0.0]])  # far apart
  pd, ppv = _calc_pdockq_from_arrays({"A": coords1, "B": coords2}, {"A": np.array([80.0]), "B": np.array([80.0])}, dist_thresh=8.0)
  assert pd == 0.0 and ppv == 0.0


# -------------------------
# Public API tests
# -------------------------


def test_calculate_pdockq_autodetect_two_chains_and_ranges():
  prot_path = FILES / "dimer_af2.pdb"  # has exactly two chains
  prot = Protein(str(prot_path))

  pd, ppv = calculate_pDockQ(prot)  # auto-detect A/B
  assert 0.0 <= pd <= 1.0
  # PPV table spans ~0.56..0.98; allow 0 if no contacts in this model
  assert 0.0 <= ppv <= 0.99

  # If we make the threshold tiny, likely no contacts -> (0, 0)
  pd0, ppv0 = calculate_pDockQ(prot, dist_thresh=0.1)
  assert pd0 == 0.0 and ppv0 == 0.0


def test_calculate_pdockq_requires_two_chains_or_explicit_ids():
  # Single chain structure: should raise when chain IDs omitted
  single = Protein(str(FILES / "1BTL.pdb"))
  with pytest.raises(ValueError):
    _ = calculate_pDockQ(single)

  # Supplying the same chain twice is invalid
  with pytest.raises(ValueError):
    _ = calculate_pDockQ(single, chain1="A", chain2="A")

  # Wrong chain name should raise
  with pytest.raises(ValueError):
    _ = calculate_pDockQ(single, chain1="A", chain2="Z")