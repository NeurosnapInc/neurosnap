"""
Amber-style relaxation / energy minimization utilities built on OpenMM.

This module loads a PDB, cleans it with PDBFixer (adds missing atoms/hydrogens,
replaces nonstandard residues), removes waters and non-biopolymers, and then
performs energy minimization (“relaxation”) using the Amber14 force field
(`amber14-all.xml` with `amber14/tip3p.xml`). By default it uses HBond
constraints, no PME (NoCutoff), and a Langevin integrator (300 K, 2 fs) for
context initialization; only the minimizer is used to relax coordinates.

Requirements
------------
- **OpenMM** (python package `openmm`) must be installed to use any function here.
- **pdbfixer** is required for structure fixing.
- **NumPy** is used for RMSD calculations.
- Optional: CUDA-compatible GPU for faster context creation/minimization
  (set `use_gpu=True`, uses mixed precision and disables PME stream).

Key functions
-------------
- `minimize(...)`: Run energy minimization on a PDB file. Accepts a tolerance
  in kcal/mol (clamped to [0.1, 10.0]), supports GPU/CPU selection, and returns
  a dict with `rmsd` (Å), `initial_energy` and `final_energy` (kJ/mol).
  B-factors from the input PDB are preserved and copied to the minimized output.
- `_compute_rmsd(...)`: Internal RMSD helper (nm in OpenMM units; returned as Å).
- `_load_bfactors_from_pdb(...)` / `_apply_bfactors_to_pdb(...)`: Internal
  helpers to snapshot and restore B-factors on ATOM/HETATM records.

Notes & caveats
---------------
- Uses `NoCutoff` for nonbonded interactions; adapt if you need PME/periodic
  systems. Hydrogens are constrained via `HBonds`.
- Minimization stops when the maximum force falls below the specified tolerance
  (converted to kJ/mol) or when `max_iterations` is reached.

Example
-------
>>> minimize("input.pdb", output_minimized_pdb="minimized.pdb",
...          tolerance=1.0, max_iterations=1000, use_gpu=True)
"""

from pathlib import Path
from typing import Any

import numpy as np
from openmm import LangevinIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer

from neurosnap.log import logger
from neurosnap.protein import Protein


def _load_bfactors_from_pdb(pdb_path):
  """
  Read ATOM/HETATM records and return a map:
    (chainID, resSeq, iCode, resName, atomName, altLoc) -> bfactor(float)
  altLoc is kept to disambiguate; if blank, store as ''.
  """
  bmap = {}
  with open(pdb_path, "r") as fh:
    for line in fh:
      if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
      # PDB fixed columns
      atom_name = line[12:16].strip()
      alt_loc = line[16].strip() or ""
      res_name = line[17:20].strip()
      chain_id = line[21].strip() or ""
      res_seq = line[22:26].strip()
      i_code = line[26].strip() or ""
      try:
        bfac = float(line[60:66])
      except ValueError:
        # Some files may leave it blank; treat as 0.0
        bfac = 0.0
      key = (chain_id, res_seq, i_code, res_name, atom_name, alt_loc)
      bmap[key] = bfac
  return bmap


def _apply_bfactors_to_pdb(template_bmap, in_path, out_path):
  """
  Copy B-factors from template_bmap into ATOM/HETATM lines of in_path,
  write to out_path. If an atom key isn't found, leave as-is.
  """
  with open(in_path, "r") as fh:
    lines = fh.readlines()

  new_lines = []
  for line in lines:
    if line.startswith("ATOM") or line.startswith("HETATM"):
      atom_name = line[12:16].strip()
      alt_loc = line[16].strip() or ""
      res_name = line[17:20].strip()
      chain_id = line[21].strip() or ""
      res_seq = line[22:26].strip()
      i_code = line[26].strip() or ""
      key = (chain_id, res_seq, i_code, res_name, atom_name, alt_loc)

      if key in template_bmap:
        bfac = template_bmap[key]
        # Replace columns 61–66 (1-indexed) i.e. line[60:66] in 0-indexed Python
        # Keep occupancy (cols 55–60) untouched.
        line = f"{line[:60]}{bfac:6.2f}{line[66:]}"
    new_lines.append(line)

  with open(out_path, "w") as fh:
    fh.writelines(new_lines)


def _compute_rmsd(ref_positions, new_positions):
  """Compute RMSD between two sets of positions."""
  ref = np.array(ref_positions.value_in_unit(unit.nanometers))
  new = np.array(new_positions.value_in_unit(unit.nanometers))

  # Center molecules
  ref -= ref.mean(axis=0)
  new -= new.mean(axis=0)

  # Compute RMSD
  diff = ref - new
  rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
  return rmsd * unit.nanometers


def minimize(
  pdb_file: str,
  output_minimized_pdb: str = "minimized.pdb",
  max_iterations: int = 1000,
  tolerance: float = 1.0,  # Default tolerance of 1.0 Kcal/mol
  use_gpu: bool = True,
  properties: dict[str, Any] = {},
) -> dict[str, float]:
  """
  Loads a PDB file and performs energy minimization with a specified tolerance.

  Args:
      pdb_file: Path to input PDB file.
      output_minimized_pdb: Path to output minimized PDB file.
      max_iterations: Number of minimization iterations.
      tolerance: Convergence tolerance in Kcal/mol (range: 0.1 - 10.0).
      use_gpu: Whether to use GPU (CUDA). Defaults to True.
      properties: Platform properties to provide to OpenMM if needed.

  Returns:
      dict: Contains RMSD, initial energy, and final energy.
  """
  # Ensure tolerance is within the acceptable range
  tolerance = max(0.1, min(10.0, tolerance))  # Clamp between 0.1 and 10.0 Kcal/mol
  tolerance_kj = tolerance * 4.184  # Convert Kcal/mol to kJ/mol

  # Extract & remove non-biopolymers
  prot = Protein(pdb_file)
  prot.remove_waters()
  prot.remove_non_biopolymers()
  prot.save(pdb_file)

  # snapshot B-factors from the cleaned input
  input_bfactor_map = _load_bfactors_from_pdb(pdb_file)

  # Fix only what's necessary with PDBFixer (nonstandard residues)
  fixer = PDBFixer(filename=pdb_file)
  fixer.findNonstandardResidues()
  fixer.replaceNonstandardResidues()
  fixer.findMissingResidues()
  fixer.addMissingAtoms()

  # Set up force field and Modeller-based fixes
  forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")

  # Build modeller from the (possibly still incomplete) topology/positions
  modeller = Modeller(fixer.topology, fixer.positions)

  # Order matters to avoid redundancy and to let FF drive what's added
  modeller.addHydrogens(forcefield, pH=7.0)  # then add Hs based on FF/pH
  modeller.addExtraParticles(forcefield)  # e.g., lone pairs/virtual sites; safe no-op for TIP3P

  # Create OpenMM system
  system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

  # Set up integrator
  integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)

  # Select GPU or CPU
  if use_gpu:
    platform = Platform.getPlatformByName("CUDA")
    properties = {**properties, "CudaPrecision": "mixed"}
    # properties["DisablePmeStream"] = "true"  # optional for certain GPUs
  else:
    platform = Platform.getPlatformByName("CPU")
    properties = {}

  # Create Simulation object
  simulation = Simulation(modeller.topology, system, integrator, platform, properties)
  simulation.context.setPositions(modeller.positions)

  # Compute initial energy
  state = simulation.context.getState(getEnergy=True)
  initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

  # Minimize energy with specified tolerance
  logger.info(f"Minimizing energy (tolerance: {tolerance} Kcal/mol, {tolerance_kj:.2f} kJ/mol)...")
  simulation.minimizeEnergy(tolerance=tolerance_kj, maxIterations=max_iterations)

  # Get minimized positions and compute final energy
  state = simulation.context.getState(getPositions=True, getEnergy=True)
  final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  minimized_positions = state.getPositions()

  # Compute RMSD
  rmsd = _compute_rmsd(modeller.positions, minimized_positions).value_in_unit(unit.angstroms)

  # Save minimized structure to a temp path first
  tmp_minimized = str(Path(output_minimized_pdb).with_suffix(".tmp.pdb"))
  with open(tmp_minimized, "w") as f:
    PDBFile.writeFile(modeller.topology, minimized_positions, f)

  # copy B-factors from input snapshot into the minimized PDB
  _apply_bfactors_to_pdb(input_bfactor_map, tmp_minimized, output_minimized_pdb)
  Path(tmp_minimized).unlink(missing_ok=True)

  logger.info(f"Minimization complete, minimized structure saved to {output_minimized_pdb}")

  return {"rmsd": rmsd, "final_energy": final_energy, "initial_energy": initial_energy}
