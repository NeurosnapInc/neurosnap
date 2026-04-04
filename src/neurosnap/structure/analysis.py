"""General analysis helpers & physical property calculations for Neurosnap structures."""

import os
import shutil
import tempfile
from math import pi
from typing import Dict, Optional, Sequence

import numpy as np
from rdkit import Chem

from neurosnap.constants import AA_RECORDS, STANDARD_NUCLEOTIDES, VDW_RADII_BONDI
from neurosnap.log import logger

from ._common import backbone_atom_order, classify_polymer_residue, coord_matrix, filter_structure_atoms, residue_key
from .structure import Structure


def get_backbone(
  structure: Structure,
  chains: Optional[Sequence[str]] = None,
  *,
  include_nucleotides: bool = True,
) -> np.ndarray:
  """Extract ordered backbone coordinates from a single structure.

  Protein residues contribute ``N``, ``CA``, and ``C`` atoms. When
  ``include_nucleotides`` is enabled, DNA and RNA residues contribute their
  sugar-phosphate backbone atoms in a deterministic order. Non-polymers are
  ignored.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chains: Optional chain IDs to include. If ``None``, all chains are used.
    include_nucleotides: If ``True``, include DNA and RNA backbone atoms in
      addition to protein backbone atoms.

  Returns:
    A NumPy array of backbone coordinates with shape ``(n_atoms, 3)``.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"get_backbone() expects a Structure, found {type(structure).__name__}.")
  selected_chain_ids = None if chains is None else {str(chain_id) for chain_id in chains}
  backbone_coords = []

  for chain_view in structure.chains():
    if selected_chain_ids is not None and chain_view.chain_id not in selected_chain_ids:
      continue
    for residue in chain_view.residues():
      atom_order = backbone_atom_order(residue, include_nucleotides=include_nucleotides)
      if not atom_order:
        continue

      residue_atoms = {atom.atom_name.strip().upper(): atom.coord for atom in residue.atoms()}
      for atom_name in atom_order:
        if atom_name in residue_atoms:
          backbone_coords.append(residue_atoms[atom_name])

  if not backbone_coords:
    return np.zeros((0, 3), dtype=np.float32)
  return np.asarray(backbone_coords, dtype=np.float32)


def calculate_distance_matrix(structure: Structure, chain: Optional[str] = None) -> np.ndarray:
  """Calculate the CA-atom distance matrix for a single structure.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional chain ID to restrict the calculation to.

  Returns:
    A square NumPy array of pairwise CA distances in Å.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"calculate_distance_matrix() expects a Structure, found {type(structure).__name__}.")
  ca_coords = []

  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      for atom in residue.atoms():
        if atom.atom_name == "CA":
          ca_coords.append(atom.coord)
          break

  if not ca_coords:
    return np.zeros((0, 0), dtype=np.float32)

  coord = np.asarray(ca_coords, dtype=np.float32)
  return np.linalg.norm(coord[:, np.newaxis, :] - coord[np.newaxis, :, :], axis=-1)


def ca_distance_matrix(structure: Structure, chain: Optional[str] = None) -> np.ndarray:
  """Alias for :func:`calculate_distance_matrix`.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional chain ID to restrict the calculation to.

  Returns:
    A square NumPy array of pairwise CA distances in Å.
  """
  return calculate_distance_matrix(structure, chain=chain)


def calculate_surface_area(
  structure: Structure,
  level: str = "R",
  probe_radius: float = 1.4,
  n_sphere_points: int = 96,
) -> float:
  """Estimate solvent-accessible surface area using a simple Shrake-Rupley approximation.

  The returned total SASA is the same regardless of ``level``; the parameter is
  kept for compatibility with the public surface-area API.

  Parameters:
    structure: Input single-model :class:`Structure`.
    level: Compatibility flag matching the historical public API. The returned
      total SASA is always a structure-level scalar, regardless of this value.
      Must be one of ``"A"``, ``"R"``, ``"C"``, ``"M"``, or ``"S"``.
    probe_radius: Solvent probe radius in Å used to inflate atom radii during
      the accessibility calculation.
    n_sphere_points: Number of surface points sampled per atom for the
      Shrake-Rupley approximation.

  Returns:
    Estimated solvent-accessible surface area in Å².
  """
  if level not in {"A", "R", "C", "M", "S"}:
    raise ValueError('level must be one of "A", "R", "C", "M", or "S".')
  if not isinstance(structure, Structure):
    raise TypeError(f"calculate_surface_area() expects a Structure, found {type(structure).__name__}.")
  atom_areas = _atom_surface_areas(structure, probe_radius=probe_radius, n_sphere_points=n_sphere_points)
  return float(atom_areas.sum())


def calculate_protein_volume(structure: Structure, chain: Optional[str] = None) -> float:
  """Estimate protein volume from atom van der Waals spheres.

  The calculation sums the volumes of van der Waals spheres for atoms belonging
  to residues classified as protein. It is therefore a simple geometric
  estimate rather than an excluded-volume or solvent-corrected measurement.

  Parameters:
    structure: Input single-model :class:`Structure`.
    chain: Optional chain ID to restrict the calculation to.

  Returns:
    Estimated protein volume in Å³.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"calculate_protein_volume() expects a Structure, found {type(structure).__name__}.")
  volume = 0.0

  for chain_view in structure.chains():
    if chain is not None and chain_view.chain_id != chain:
      continue
    for residue in chain_view.residues():
      if classify_polymer_residue(residue) != "protein":
        continue
      for atom in residue.atoms():
        radius = _vdw_radius(atom.element)
        volume += (4.0 / 3.0) * pi * (radius**3)

  return float(volume)


def _residue_surface_area_map(structure_model, probe_radius: float = 1.4, n_sphere_points: int = 96) -> Dict[tuple, float]:
  """Return per-residue solvent-accessible surface areas for one model."""
  atom_areas = _atom_surface_areas(structure_model, probe_radius=probe_radius, n_sphere_points=n_sphere_points)
  residue_areas: Dict[tuple, float] = {}

  atom_index = 0
  for chain_view in structure_model.chains():
    for residue in chain_view.residues():
      key = residue_key(residue)
      residue_areas[key] = residue_areas.get(key, 0.0)
      for _atom in residue.atoms():
        residue_areas[key] += float(atom_areas[atom_index])
        atom_index += 1

  return residue_areas


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


def extract_non_biopolymers(structure: Structure, output_dir: str, min_atoms: int = 0):
  """Extract non-biopolymer fragments from a structure and write them as SDF files.

  Biopolymer residues are removed using the same residue-name logic as the old
  implementation: any residue present in ``AA_RECORDS`` or
  ``STANDARD_NUCLEOTIDES`` is treated as part of a protein or nucleotide
  polymer, except ``UNK`` which is preserved. The remaining atoms are written to
  a temporary PDB, read into RDKit, split into disconnected fragments, and then
  exported as individual SDF files.

  Parameters:
    structure: Input single-model :class:`Structure`.
    output_dir: Directory where SDF files will be written. Any existing
      directory at that path is replaced.
    min_atoms: Minimum fragment atom count required for export.

  Returns:
    ``None``. Matching fragments are written to ``output_dir`` as ``.sdf``
    files.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"extract_non_biopolymers() expects a Structure, found {type(structure).__name__}.")

  biopolymer_keywords = set(AA_RECORDS.keys()).union(STANDARD_NUCLEOTIDES)
  biopolymer_keywords.discard("UNK")

  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  ligand_structure = Structure(remove_annotations=False)
  ligand_structure._dtype_atoms = structure._dtype_atoms
  ligand_structure._dtype_atom_annotations = structure._dtype_atom_annotations
  ligand_structure._dtype_bond = structure._dtype_bond
  ligand_structure.atoms = structure.atoms.copy()
  ligand_structure.atom_annotations = structure.atom_annotations.copy()
  ligand_structure.bonds = structure.bonds.copy()
  ligand_structure.metadata = dict(structure.metadata)

  keep_mask = ~np.isin(ligand_structure.atom_annotations["res_name"], list(biopolymer_keywords))
  filter_structure_atoms(ligand_structure, keep_mask)

  if len(ligand_structure) == 0:
    logger.info("Extracted 0 non-biopolymer molecules to %s.", output_dir)
    return

  from neurosnap.io.pdb import save_pdb

  with tempfile.NamedTemporaryFile(suffix=".pdb") as temp_pdb:
    save_pdb(ligand_structure, temp_pdb.name)
    molecule = Chem.MolFromPDBFile(temp_pdb.name, removeHs=False, sanitize=False)

  if molecule is None:
    raise ValueError("Failed to convert structure into an RDKit molecule.")

  fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
  molecule_count = 1
  for fragment_index, fragment in enumerate(fragments):
    if fragment is None:
      logger.warning("Skipping fragment %s due to processing failure (fragment is None).", fragment_index)
      continue

    try:
      Chem.SanitizeMol(fragment)
    except Exception as exc:
      raise ValueError(f"Failed to sanitize fragment {fragment_index}: {exc}")

    if fragment.GetNumAtoms() < min_atoms:
      logger.info("Skipping small molecule fragment %s (atom count: %s).", fragment_index, fragment.GetNumAtoms())
      continue

    sdf_file = os.path.join(output_dir, f"mol_{molecule_count}.sdf")
    writer = Chem.SDWriter(sdf_file)
    writer.write(fragment)
    writer.close()
    molecule_count += 1

  logger.info("Extracted %s non-biopolymer molecules to %s.", molecule_count - 1, output_dir)
