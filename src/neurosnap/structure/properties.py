"""Physical property calculations for Neurosnap structures."""

from math import pi
from typing import Dict, Optional

import numpy as np

from neurosnap.constants import VDW_RADII_BONDI

from ._common import classify_polymer_residue, coord_matrix, residue_key, resolve_model


def calculate_surface_area(
  structure,
  model: Optional[int] = None,
  level: str = "R",
  probe_radius: float = 1.4,
  n_sphere_points: int = 96,
) -> float:
  """Estimate solvent-accessible surface area using a simple Shrake-Rupley approximation.

  The returned total SASA is the same regardless of ``level``; the parameter is
  kept for compatibility with the public surface-area API.
  """
  if level not in {"A", "R", "C", "M", "S"}:
    raise ValueError('level must be one of "A", "R", "C", "M", or "S".')
  structure_model = resolve_model(structure, model=model)
  atom_areas = _atom_surface_areas(structure_model, probe_radius=probe_radius, n_sphere_points=n_sphere_points)
  return float(atom_areas.sum())


def calculate_protein_volume(structure, model: Optional[int] = None, chain: Optional[str] = None) -> float:
  """Estimate protein volume from atom van der Waals spheres."""
  structure_model = resolve_model(structure, model=model)
  volume = 0.0

  for chain_view in structure_model.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if classify_polymer_residue(residue) != "protein":
        continue
      for atom in residue.atoms():
        radius = _vdw_radius(atom.element)
        volume += (4.0 / 3.0) * pi * (radius**3)

  return float(volume)


def _atom_surface_areas(structure_model, probe_radius: float = 1.4, n_sphere_points: int = 96) -> np.ndarray:
  """Return per-atom solvent-accessible surface areas."""
  coord = coord_matrix(structure_model).astype(np.float32, copy=False)
  atom_count = len(coord)
  if atom_count == 0:
    return np.zeros(0, dtype=np.float32)

  elements = np.asarray([str(element).strip() for element in structure_model.atom_annotations["element"]], dtype=object)
  radii = np.asarray([_vdw_radius(element) + probe_radius for element in elements], dtype=np.float32)
  sphere_points = _unit_sphere_points(n_sphere_points).astype(np.float32, copy=False)
  center_distances = np.linalg.norm(coord[:, np.newaxis, :] - coord[np.newaxis, :, :], axis=-1)
  atom_areas = np.zeros(atom_count, dtype=np.float32)

  for atom_index in range(atom_count):
    candidate_neighbors = np.where((np.arange(atom_count) != atom_index) & (center_distances[atom_index] < (radii[atom_index] + radii + 1e-6)))[0]
    if len(candidate_neighbors) == 0:
      atom_areas[atom_index] = 4.0 * pi * (radii[atom_index] ** 2)
      continue

    points = coord[atom_index] + sphere_points * radii[atom_index]
    blocked = np.zeros(n_sphere_points, dtype=bool)
    for neighbor_index in candidate_neighbors:
      squared_distance = np.sum((points - coord[neighbor_index]) ** 2, axis=1)
      blocked |= squared_distance < (radii[neighbor_index] ** 2)
      if blocked.all():
        break

    accessible_fraction = float((~blocked).sum()) / float(n_sphere_points)
    atom_areas[atom_index] = accessible_fraction * (4.0 * pi * (radii[atom_index] ** 2))

  return atom_areas


def _unit_sphere_points(n_sphere_points: int) -> np.ndarray:
  """Return approximately uniform unit-sphere points by golden-section spiral."""
  indices = np.arange(n_sphere_points, dtype=np.float32)
  offset = 2.0 / float(n_sphere_points)
  y_coord = ((indices * offset) - 1.0) + (offset / 2.0)
  radius = np.sqrt(np.clip(1.0 - y_coord**2, 0.0, None))
  phi = indices * (pi * (3.0 - np.sqrt(5.0)))
  x_coord = np.cos(phi) * radius
  z_coord = np.sin(phi) * radius
  return np.column_stack((x_coord, y_coord, z_coord))


def _vdw_radius(element: str) -> float:
  """Return a van der Waals radius with a conservative fallback."""
  element = str(element).strip()
  if not element:
    return 1.8
  normalized = element[0].upper() + element[1:].lower()
  return float(VDW_RADII_BONDI.get(normalized, 1.8))
