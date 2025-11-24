import math
from types import SimpleNamespace

import numpy as np
import pytest

from neurosnap.algos import ec_interface


class FakeAtom:
  def __init__(self, coord):
    self.coord = np.asarray(coord, dtype=float)


def test_compute_ec_returns_nan_without_contacts(monkeypatch):
  monkeypatch.setattr(ec_interface, "find_interface_atoms", lambda *args, **kwargs: ([], []))

  result = ec_interface.compute_ec_for_pair(
    SimpleNamespace(structure=None),
    "A",
    "B",
    cutoff=4.5,
    forcefield="AMBER",
    pdb2pqr="pdb2pqr",
    apbs="apbs",
  )

  assert all(math.isnan(v) for v in result)


def test_compute_ec_requires_sufficient_samples(monkeypatch):
  atoms = [FakeAtom((i, 0.0, 0.0)) for i in range(5)]
  monkeypatch.setattr(ec_interface, "find_interface_atoms", lambda *args, **kwargs: (atoms, atoms))

  monkeypatch.setattr(ec_interface, "write_single_chain_pdb", lambda structure, chain_id, outfile: outfile.write_text(f"{chain_id}\n"))
  prep_calls = {"count": 0}
  apbs_calls = {"count": 0}

  def fake_prepare(pdb_path, pqr_path, pdb2pqr_bin, forcefield):
    prep_calls["count"] += 1

  def fake_run_apbs(pqr_path, dx_out, apbs_bin):
    apbs_calls["count"] += 1

  monkeypatch.setattr(ec_interface, "_prepare_pqr", fake_prepare)
  monkeypatch.setattr(ec_interface, "_run_apbs", fake_run_apbs)
  monkeypatch.setattr(ec_interface, "_parse_dx", lambda path: (np.zeros(3), (1, 1, 1), np.zeros((2, 2, 2))))
  monkeypatch.setattr(ec_interface, "_sample_potential", lambda coords, origin, delta, grid: np.arange(len(coords), dtype=float))

  result = ec_interface.compute_ec_for_pair(
    SimpleNamespace(structure=None),
    "A",
    "B",
    cutoff=4.5,
    forcefield="AMBER",
    pdb2pqr="pdb2pqr",
    apbs="apbs",
  )

  assert all(math.isnan(v) for v in result)
  assert prep_calls["count"] == 2  # binder and target
  assert apbs_calls["count"] == 2


def test_compute_ec_uses_pearson_correlations(monkeypatch):
  binder_atoms = [FakeAtom((i, 0.0, 0.0)) for i in range(12)]
  target_atoms = [FakeAtom((i + 100, 0.0, 0.0)) for i in range(12)]
  monkeypatch.setattr(ec_interface, "find_interface_atoms", lambda *args, **kwargs: (binder_atoms, target_atoms))

  written_chains = []

  def fake_write(structure, chain_id, outfile):
    written_chains.append(chain_id)
    outfile.write_text(f"{chain_id}\n")

  monkeypatch.setattr(ec_interface, "write_single_chain_pdb", fake_write)
  monkeypatch.setattr(ec_interface, "_prepare_pqr", lambda *args, **kwargs: None)
  monkeypatch.setattr(ec_interface, "_run_apbs", lambda *args, **kwargs: None)
  monkeypatch.setattr(ec_interface, "_parse_dx", lambda path: (np.zeros(3), (1, 1, 1), np.zeros((2, 2, 2))))

  potentials = [
    np.arange(12, dtype=float),  # V_b_on_b
    np.arange(12, dtype=float)[::-1],  # V_t_on_b (perfect negative correlation)
    np.linspace(0.0, 11.0, 12),  # V_b_on_t
    np.linspace(0.0, 11.0, 12),  # V_t_on_t (perfect positive correlation)
  ]

  def fake_sample_potential(coords, origin, delta, grid):
    return potentials.pop(0)

  monkeypatch.setattr(ec_interface, "_sample_potential", fake_sample_potential)

  ec, r_b, r_t = ec_interface.compute_ec_for_pair(
    SimpleNamespace(structure=None),
    "A",
    "B",
    cutoff=4.5,
    forcefield="AMBER",
    pdb2pqr="pdb2pqr",
    apbs="apbs",
  )

  assert ec == pytest.approx(0.0)
  assert r_b == pytest.approx(-1.0)
  assert r_t == pytest.approx(1.0)
  assert written_chains == ["A", "B"]


def test_compute_ec_real_structure_with_stubbed_io(monkeypatch):
  target_r_b = -0.009667953397034084
  target_r_t = -0.007880445653751258

  # real interface detection on the AF2 dimer
  protein = ec_interface.Protein("tests/files/dimer_af2.pdb")

  ib_atoms, it_atoms = ec_interface.find_interface_atoms(protein, "A", "B", 4.5)

  def make_correlated(length, target_r, rng):
    x = rng.normal(size=length)
    x = x - x.mean()
    x_hat = x / np.linalg.norm(x)
    y = rng.normal(size=length)
    y = y - y.mean()
    y = y - x_hat * (y @ x_hat)  # enforce orthogonality to control correlation
    y_hat = y / np.linalg.norm(y)
    z = target_r * x_hat + math.sqrt(1 - target_r**2) * y_hat
    return x_hat, z

  rng = np.random.default_rng(1234)
  V_b_on_b, V_t_on_b = make_correlated(len(ib_atoms), target_r_b, rng)
  V_b_on_t, V_t_on_t = make_correlated(len(it_atoms), target_r_t, rng)

  monkeypatch.setattr(ec_interface, "_prepare_pqr", lambda *args, **kwargs: None)
  monkeypatch.setattr(ec_interface, "_run_apbs", lambda *args, **kwargs: None)
  monkeypatch.setattr(ec_interface, "_parse_dx", lambda path: (np.zeros(3), (1, 1, 1), np.zeros((2, 2, 2))))

  calls = {"i": 0}

  def fake_sample(coords, origin, delta, grid):
    idx = calls["i"]
    calls["i"] += 1
    mapping = {0: V_b_on_b, 1: V_t_on_b, 2: V_b_on_t, 3: V_t_on_t}
    arr = mapping[idx]
    return arr[: len(coords)]

  monkeypatch.setattr(ec_interface, "_sample_potential", fake_sample)

  data = ec_interface.compute_ec_for_pair(
    protein,
    "A",
    "B",
  )

  expected = (
    np.float64(0.00877419952539267),
    np.float64(-0.009667953397034084),
    np.float64(-0.007880445653751258),
  )
  assert data == pytest.approx(expected)
