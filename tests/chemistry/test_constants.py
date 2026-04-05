# tests/chemistry/test_constants.py

from neurosnap.constants import ATOMIC_MASSES


def test_atomic_masses_cover_all_elements():
  assert len(ATOMIC_MASSES) == 118


def test_atomic_masses_include_expected_reference_values():
  assert ATOMIC_MASSES["H"] == 1.008
  assert ATOMIC_MASSES["C"] == 12.011
  assert ATOMIC_MASSES["Fe"] == 55.845
  assert ATOMIC_MASSES["U"] == 238.02891
  assert ATOMIC_MASSES["Og"] == 294.0
