import math

import numpy as np
import pytest

from neurosnap.algos import ec_interface
from tests._structure_test_utils import parse_single_model


class FakeAtom:
  def __init__(self, coord, *, chain_id="A", res_id=1, atom_name="CA", res_name="GLY", ins_code="", hetero=False):
    self.coord = np.asarray(coord, dtype=float)
    self.chain_id = chain_id
    self.res_id = res_id
    self.atom_name = atom_name
    self.res_name = res_name
    self.ins_code = ins_code
    self.hetero = hetero


def test_compute_ec_returns_nan_without_contacts(monkeypatch):
  monkeypatch.setattr(ec_interface, "resolve_model", lambda structure, model=None: structure)
  monkeypatch.setattr(ec_interface, "find_interface_contacts", lambda *args, **kwargs: [])

  result = ec_interface.compute_ec(object(), "A", "B", cutoff=4.5, forcefield="AMBER", pdb2pqr="pdb2pqr", apbs="apbs")

  assert all(math.isnan(v) for v in result)


def test_compute_ec_requires_sufficient_samples(monkeypatch):
  chain1_atoms = [FakeAtom((i, 0.0, 0.0), chain_id="A", res_id=i + 1, atom_name=f"C{i}") for i in range(5)]
  chain2_atoms = [FakeAtom((i + 10, 0.0, 0.0), chain_id="B", res_id=i + 1, atom_name=f"N{i}") for i in range(5)]
  monkeypatch.setattr(ec_interface, "resolve_model", lambda structure, model=None: structure)
  monkeypatch.setattr(ec_interface, "find_interface_contacts", lambda *args, **kwargs: list(zip(chain1_atoms, chain2_atoms)))

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

  result = ec_interface.compute_ec(object(), "A", "B", cutoff=4.5, forcefield="AMBER", pdb2pqr="pdb2pqr", apbs="apbs")

  assert all(math.isnan(v) for v in result)
  assert prep_calls["count"] == 2  # both interface chains
  assert apbs_calls["count"] == 2


def test_compute_ec_uses_pearson_correlations(monkeypatch):
  chain1_atoms = [FakeAtom((i, 0.0, 0.0), chain_id="A", res_id=i + 1, atom_name=f"C{i}") for i in range(12)]
  chain2_atoms = [FakeAtom((i + 100, 0.0, 0.0), chain_id="B", res_id=i + 1, atom_name=f"N{i}") for i in range(12)]
  monkeypatch.setattr(ec_interface, "resolve_model", lambda structure, model=None: structure)
  monkeypatch.setattr(ec_interface, "find_interface_contacts", lambda *args, **kwargs: list(zip(chain1_atoms, chain2_atoms)))

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

  ec, r_b, r_t = ec_interface.compute_ec(object(), "A", "B", cutoff=4.5, forcefield="AMBER", pdb2pqr="pdb2pqr", apbs="apbs")

  assert ec == pytest.approx(0.0)
  assert r_b == pytest.approx(-1.0)
  assert r_t == pytest.approx(1.0)
  assert written_chains == ["A", "B"]


def test_compute_ec_real_structure_with_stubbed_io(monkeypatch):
  target_r_b = -0.009667953397034084
  target_r_t = -0.007880445653751258

  # real interface detection on the AF2 dimer
  structure = parse_single_model("tests/files/dimer_af2.pdb")

  contacts = ec_interface.find_interface_contacts(structure, "A", "B", cutoff=4.5, hydrogens=False)
  ib_atoms = []
  ib_seen = set()
  it_atoms = []
  it_seen = set()
  for atom1, atom2 in contacts:
    key1 = (atom1.chain_id, atom1.res_id, atom1.ins_code, atom1.res_name, atom1.atom_name)
    key2 = (atom2.chain_id, atom2.res_id, atom2.ins_code, atom2.res_name, atom2.atom_name)
    if key1 not in ib_seen:
      ib_seen.add(key1)
      ib_atoms.append(atom1)
    if key2 not in it_seen:
      it_seen.add(key2)
      it_atoms.append(atom2)

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

  data = ec_interface.compute_ec(structure, "A", "B")

  expected = (
    np.float64(0.00877419952539267),
    np.float64(-0.009667953397034084),
    np.float64(-0.007880445653751258),
  )
  assert data == pytest.approx(expected)
