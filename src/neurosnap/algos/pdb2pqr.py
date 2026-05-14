"""Structure-first PDB2PQR integration for Neurosnap."""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from neurosnap.constants.chemistry import ATOMIC_MASSES
from neurosnap.io.pdb import save_pdb
from neurosnap.log import logger
from neurosnap.structure import Structure

from ._pdb2pqr_vendor import biomolecule as vendor_biomolecule
from ._pdb2pqr_vendor import debump as vendor_debump
from ._pdb2pqr_vendor import forcefield as vendor_forcefield
from ._pdb2pqr_vendor import hydrogens as vendor_hydrogens
from ._pdb2pqr_vendor import io as vendor_io
from ._pdb2pqr_vendor import pdb as vendor_pdb
from ._pdb2pqr_vendor.config import FORCE_FIELDS, REPAIR_LIMIT
from ._pdb2pqr_vendor.utilities import noninteger_charge

__all__ = ["PDB2PQR_FORCE_FIELDS", "assign_pqr"]

PDB2PQR_FORCE_FIELDS = tuple(forcefield.upper() for forcefield in FORCE_FIELDS)
_VENDORED_LOGGER_NAMES = (
  "neurosnap.algos._pdb2pqr_vendor.aa",
  "neurosnap.algos._pdb2pqr_vendor.biomolecule",
  "neurosnap.algos._pdb2pqr_vendor.cells",
  "neurosnap.algos._pdb2pqr_vendor.debump",
  "neurosnap.algos._pdb2pqr_vendor.definitions",
  "neurosnap.algos._pdb2pqr_vendor.forcefield",
  "neurosnap.algos._pdb2pqr_vendor.hydrogens",
  "neurosnap.algos._pdb2pqr_vendor.hydrogens.optimize",
  "neurosnap.algos._pdb2pqr_vendor.hydrogens.structures",
  "neurosnap.algos._pdb2pqr_vendor.pdb",
  "neurosnap.algos._pdb2pqr_vendor.residue",
  "neurosnap.algos._pdb2pqr_vendor.utilities",
)


def assign_pqr(
  structure: Structure,
  *,
  forcefield: str = "PARSE",
  ffout: Optional[str] = None,
  neutraln: bool = False,
  neutralc: bool = False,
  assign_only: bool = False,
  debump: bool = True,
  optimize: bool = True,
) -> Structure:
  """Assign PDB2PQR radii and partial charges to a local Structure.

  The returned structure is rebuilt from the PDB2PQR-updated atom table and
  includes two float annotations:
    - ``partial_charge``
    - ``radius``

  Parameters:
    structure: Input single-model structure.
    forcefield: PDB2PQR forcefield name.
    ffout: Optional alternate output naming scheme.
    neutraln: Make the N-terminus neutral. Only supported with PARSE.
    neutralc: Make the C-terminus neutral. Only supported with PARSE.
    assign_only: Assign parameters without repair, debumping, or hydrogen optimization.
    debump: Run the PDB2PQR debumping routines.
    optimize: Optimize hydrogens when ``assign_only`` is false.

  Returns:
    A new :class:`Structure` carrying PDB2PQR geometry updates and annotations.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"assign_pqr() expects a Structure, found {type(structure).__name__}.")

  forcefield_name = _normalize_forcefield_name(forcefield)
  ffout_name = None if ffout is None else _normalize_forcefield_name(ffout)
  if neutraln and forcefield_name != "parse":
    raise ValueError("neutraln only works with the PARSE forcefield.")
  if neutralc and forcefield_name != "parse":
    raise ValueError("neutralc only works with the PARSE forcefield.")

  effective_debump = bool(debump) and not assign_only
  effective_optimize = bool(optimize) and not assign_only

  matched_atoms, header, metadata_updates = _run_vendor_pdb2pqr(
    structure,
    forcefield_name=forcefield_name,
    ffout_name=ffout_name,
    neutraln=neutraln,
    neutralc=neutralc,
    assign_only=assign_only,
    debump_requested=bool(debump),
    optimize_requested=bool(optimize),
    effective_debump=effective_debump,
    effective_optimize=effective_optimize,
  )
  updated_structure = _build_structure_from_vendor_atoms(matched_atoms, source_metadata=structure.metadata)
  updated_structure.metadata["pdb2pqr_header"] = header
  updated_structure.metadata.update(metadata_updates)
  return updated_structure
def _normalize_forcefield_name(forcefield: str) -> str:
  if not isinstance(forcefield, str) or not forcefield:
    raise ValueError("forcefield must be a non-empty string.")
  normalized = forcefield.strip().lower()
  if normalized not in FORCE_FIELDS:
    raise ValueError(f'forcefield must be one of {", ".join(PDB2PQR_FORCE_FIELDS)}.')
  return normalized


def _run_vendor_pdb2pqr(
  structure: Structure,
  *,
  forcefield_name: str,
  ffout_name: Optional[str],
  neutraln: bool,
  neutralc: bool,
  assign_only: bool,
  debump_requested: bool,
  optimize_requested: bool,
  effective_debump: bool,
  effective_optimize: bool,
) -> Tuple[Sequence[Any], str, Dict[str, Any]]:
  metadata_updates: Dict[str, Any] = {"pdb2pqr_remediations": []}
  working_structure = structure

  if not assign_only and _structure_has_hydrogens(working_structure):
    working_structure = _strip_hydrogens(working_structure)
    metadata_updates["pdb2pqr_remediations"].append("stripped_input_hydrogens")
    logger.info("PDB2PQR: stripped input hydrogens before rebuilding protonation and parameters.")

  try:
    with _vendored_logging_context():
      matched_atoms, header, run_metadata = _run_vendor_pdb2pqr_once(
        working_structure,
        forcefield_name=forcefield_name,
        ffout_name=ffout_name,
        neutraln=neutraln,
        neutralc=neutralc,
        assign_only=assign_only,
        effective_debump=effective_debump,
        effective_optimize=effective_optimize,
      )
  except TypeError as exc:
    if not assign_only or not _is_assign_only_histidine_error(exc):
      raise
    fallback_structure = _strip_hydrogens(structure)
    metadata_updates["pdb2pqr_remediations"].extend(
      ["assign_only_histidine_fallback", "stripped_input_hydrogens"]
    )
    logger.info("PDB2PQR: input protonation was ambiguous for assign-only mode; retrying with a full hydrogen rebuild.")
    with _vendored_logging_context():
      matched_atoms, header, run_metadata = _run_vendor_pdb2pqr_once(
        fallback_structure,
        forcefield_name=forcefield_name,
        ffout_name=ffout_name,
        neutraln=neutraln,
        neutralc=neutralc,
        assign_only=False,
        effective_debump=debump_requested,
        effective_optimize=optimize_requested,
      )

  metadata_updates.update(run_metadata)
  if not metadata_updates["pdb2pqr_remediations"]:
    metadata_updates.pop("pdb2pqr_remediations")
  return matched_atoms, header, metadata_updates


def _run_vendor_pdb2pqr_once(
  structure: Structure,
  *,
  forcefield_name: str,
  ffout_name: Optional[str],
  neutraln: bool,
  neutralc: bool,
  assign_only: bool,
  effective_debump: bool,
  effective_optimize: bool,
) -> Tuple[Sequence[Any], str, Dict[str, Any]]:
  definition = vendor_io.get_definitions()
  pdblist = _structure_to_vendor_pdblist(structure)

  biomolecule = vendor_biomolecule.Biomolecule(pdblist, definition)
  logger.info("PDB2PQR: assigning %s parameters for %d residues and %d atoms.", forcefield_name.upper(), len(biomolecule.residues), len(biomolecule.atoms))
  biomolecule.set_termini(neutraln=neutraln, neutralc=neutralc)
  biomolecule.update_bonds()

  if assign_only:
    biomolecule.set_hip()
  else:
    if _is_repairable(biomolecule):
      biomolecule.repair_heavy()
    biomolecule.update_ss_bridges()
    debumper = vendor_debump.Debump(biomolecule)
    if effective_debump:
      debumper.debump_biomolecule()
    biomolecule.add_hydrogens()
    if effective_debump:
      debumper.debump_biomolecule()
    hydrogen_handler = vendor_hydrogens.create_handler()
    hydrogen_routines = vendor_hydrogens.HydrogenRoutines(debumper, hydrogen_handler)
    if effective_optimize:
      hydrogen_routines.set_optimizeable_hydrogens()
      biomolecule.hold_residues(None)
      hydrogen_routines.initialize_full_optimization()
    else:
      hydrogen_routines.initialize_wat_optimization()
    hydrogen_routines.optimize_hydrogens()
    hydrogen_routines.cleanup()

  biomolecule.set_states()
  active_forcefield = vendor_forcefield.Forcefield(forcefield_name, definition, None, None)
  matched_atoms, missing_atoms = biomolecule.apply_force_field(active_forcefield)

  total_charge = 0.0
  residue_charge_warnings = []
  for residue in biomolecule.residues:
    charge = residue.charge
    charge_error = noninteger_charge(charge)
    if charge_error:
      residue_charge_warnings.append((str(residue), charge_error))
    total_charge += charge

  if residue_charge_warnings:
    sample_count = min(3, len(residue_charge_warnings))
    sample_text = "; ".join(f"{residue}: {message}" for residue, message in residue_charge_warnings[:sample_count])
    if len(residue_charge_warnings) > sample_count:
      sample_text += f"; +{len(residue_charge_warnings) - sample_count} more"
    logger.warning("PDB2PQR found %d residues with non-integral charges: %s", len(residue_charge_warnings), sample_text)

  total_charge_error = noninteger_charge(total_charge)
  if total_charge_error:
    logger.warning("PDB2PQR returned a non-integral total charge: %s", total_charge_error)

  if ffout_name is not None:
    output_forcefield = active_forcefield if ffout_name == forcefield_name else vendor_forcefield.Forcefield(ffout_name, definition, None, None)
    biomolecule.apply_name_scheme(output_forcefield)

  reslist, net_charge = biomolecule.charge
  header = vendor_io.print_pqr_header(
    biomolecule.pdblist,
    missing_atoms,
    reslist,
    net_charge,
    forcefield_name,
    None,
    None,
    ffout_name,
    include_old_header=False,
  )
  metadata_updates = {
    "pdb2pqr_forcefield": forcefield_name.upper(),
    "pdb2pqr_missing_atoms": [
      {
        "serial": int(atom.serial),
        "atom_name": str(atom.name),
        "res_name": str(atom.residue.name),
        "res_id": int(atom.residue.res_seq),
        "chain_id": str(atom.chain_id),
      }
      for atom in missing_atoms
    ],
  }
  if total_charge_error:
    metadata_updates["pdb2pqr_charge_warning"] = total_charge_error
  if ffout_name is not None:
    metadata_updates["pdb2pqr_ffout"] = ffout_name.upper()
  return matched_atoms, header, metadata_updates


def _structure_to_vendor_pdblist(structure: Structure):
  buffer = io.StringIO()
  save_pdb(structure, buffer)
  buffer.seek(0)
  pdblist, errlist = vendor_pdb.read_pdb(buffer)
  if errlist:
    logger.warning("Vendored PDB2PQR parser reported non-standard record types: %s", ", ".join(sorted(set(errlist))))
  return pdblist


@contextmanager
def _vendored_logging_context():
  previous_levels = []
  try:
    for logger_name in _VENDORED_LOGGER_NAMES:
      vendored_logger = logging.getLogger(logger_name)
      previous_levels.append((vendored_logger, vendored_logger.level))
      vendored_logger.setLevel(logging.ERROR)
    yield
  finally:
    for vendored_logger, previous_level in previous_levels:
      vendored_logger.setLevel(previous_level)


def _structure_has_hydrogens(structure: Structure) -> bool:
  if len(structure) == 0:
    return False
  elements = np.char.upper(np.char.strip(structure.atom_annotations["element"].astype("U2")))
  return bool(np.any(elements == "H"))


def _strip_hydrogens(structure: Structure) -> Structure:
  if not _structure_has_hydrogens(structure):
    return structure

  elements = np.char.upper(np.char.strip(structure.atom_annotations["element"].astype("U2")))
  keep_mask = elements != "H"
  kept_indices = np.flatnonzero(keep_mask)
  index_map = {int(old_index): new_index for new_index, old_index in enumerate(kept_indices)}

  stripped = Structure(remove_annotations=False)
  stripped.metadata = dict(structure.metadata)
  stripped.atoms = structure.atoms[keep_mask].copy()
  stripped.atom_annotations = structure.atom_annotations[keep_mask].copy()

  bond_rows = []
  for bond in structure.bonds:
    atom_i = int(bond["atom_i"])
    atom_j = int(bond["atom_j"])
    if atom_i not in index_map or atom_j not in index_map:
      continue
    bond_rows.append((index_map[atom_i], index_map[atom_j], int(bond["bond_type"])))

  if bond_rows:
    stripped.bonds = np.array(bond_rows, dtype=structure._dtype_bond)
  else:
    stripped.bonds = np.zeros(0, dtype=structure._dtype_bond)
  return stripped


def _is_assign_only_histidine_error(exc: Exception) -> bool:
  message = str(exc)
  return "Missing both HD1 and HE2 atoms" in message and "assign-only" in message


def _is_repairable(biomolecule) -> bool:
  num_heavy = biomolecule.num_heavy
  num_missing = biomolecule.num_missing_heavy
  if num_heavy == 0:
    raise ValueError("No biomolecule heavy atoms were found by the vendored PDB2PQR engine.")
  if num_missing == 0:
    return False
  missing_fraction = float(num_missing) / float(num_heavy)
  if missing_fraction > REPAIR_LIMIT:
    raise ValueError(
      f"This structure is missing too many heavy atoms for repair ({num_missing} of {num_heavy}, fraction={missing_fraction:g}; limit={REPAIR_LIMIT:g})."
    )
  return True


def _build_structure_from_vendor_atoms(vendor_atoms: Sequence[Any], *, source_metadata: Dict[str, Any]) -> Structure:
  structure = Structure(remove_annotations=False)
  structure.metadata = dict(source_metadata)
  structure.metadata["source_format"] = "pqr"
  atom_count = len(vendor_atoms)
  structure.atoms = np.zeros(atom_count, dtype=structure._dtype_atoms)
  structure.atom_annotations = np.zeros(atom_count, dtype=structure._dtype_atom_annotations)

  partial_charges = np.zeros(atom_count, dtype=np.float32)
  radii = np.zeros(atom_count, dtype=np.float32)
  bond_pairs = set()
  atom_to_index = {id(atom): atom_index for atom_index, atom in enumerate(vendor_atoms)}

  for atom_index, atom in enumerate(vendor_atoms):
    structure.atoms["x"][atom_index] = float(atom.x)
    structure.atoms["y"][atom_index] = float(atom.y)
    structure.atoms["z"][atom_index] = float(atom.z)
    structure.atom_annotations["chain_id"][atom_index] = atom.chain_id or ""
    structure.atom_annotations["res_id"][atom_index] = int(atom.res_seq)
    structure.atom_annotations["ins_code"][atom_index] = atom.ins_code or ""
    structure.atom_annotations["res_name"][atom_index] = atom.res_name or ""
    structure.atom_annotations["hetero"][atom_index] = bool(atom.type == "HETATM")
    structure.atom_annotations["atom_name"][atom_index] = atom.name or ""
    structure.atom_annotations["element"][atom_index] = _infer_element(atom.element, atom.name)
    structure.atom_annotations["atom_id"][atom_index] = int(atom.serial or (atom_index + 1))
    structure.atom_annotations["b_factor"][atom_index] = float(atom.temp_factor) if atom.temp_factor is not None else 0.0
    structure.atom_annotations["occupancy"][atom_index] = float(atom.occupancy) if atom.occupancy is not None else 1.0
    structure.atom_annotations["charge"][atom_index] = _normalize_integer_charge(getattr(atom, "charge", None))
    structure.atom_annotations["sym_id"][atom_index] = ""
    partial_charges[atom_index] = float(atom.ffcharge) if atom.ffcharge is not None else 0.0
    radii[atom_index] = float(atom.radius) if atom.radius is not None else 0.0

    for bonded_atom in getattr(atom, "bonds", []):
      bonded_index = atom_to_index.get(id(bonded_atom))
      if bonded_index is None or bonded_index == atom_index:
        continue
      bond_pairs.add((min(atom_index, bonded_index), max(atom_index, bonded_index)))

  if bond_pairs:
    structure.bonds = np.array([(atom_i, atom_j, 1) for atom_i, atom_j in sorted(bond_pairs)], dtype=structure._dtype_bond)
  else:
    structure.bonds = np.zeros(0, dtype=structure._dtype_bond)

  structure.add_annotation("partial_charge", np.float32, values=partial_charges)
  structure.add_annotation("radius", np.float32, values=radii)
  structure._remove_empty_annotations()
  return structure


def _infer_element(raw_element: Any, atom_name: Any) -> str:
  element = "" if raw_element is None else str(raw_element).strip().upper()
  if element in ATOMIC_MASSES:
    return element

  atom_name_text = "" if atom_name is None else str(atom_name).strip()
  if not atom_name_text:
    return ""

  candidate = atom_name_text[:2].strip().title()
  if candidate in ATOMIC_MASSES:
    return candidate.upper()

  candidate = atom_name_text[:1].strip().title()
  if candidate in ATOMIC_MASSES:
    return candidate.upper()
  return ""


def _normalize_integer_charge(value: Any) -> int:
  if value is None:
    return 0
  if isinstance(value, (int, np.integer)):
    return int(value)
  text = str(value).strip()
  if not text:
    return 0
  try:
    return int(float(text))
  except ValueError:
    sign = -1 if text.endswith("-") else 1
    digits = "".join(char for char in text if char.isdigit())
    if digits:
      return sign * int(digits)
    return 0
