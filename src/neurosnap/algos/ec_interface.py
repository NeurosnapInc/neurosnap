"""
ec_interface.py
===============

Quantify electrostatic complementarity (EC) for binder–target chain
pairs in a protein complex.

Key features
------------
* Biopython-based interface detection (distance-cutoff, heavy atoms).
* PDB2PQR for protonation / charge & radius assignment.
* APBS continuum electrostatics – works with *all* 1.x-to-3.x builds:
  - auto builds a padding box around the molecule,
  - supplies mandatory legacy keywords (glen/gcent/sdens),
  - handles .dx.gz, rank-tagged PE-files, and sub-dir output.
* Step-by-step logging for debugging and data provenance.
* Cleans all temporary files automatically.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from scipy.stats import pearsonr

from neurosnap.log import logger


# ---------------------------------------------------------------------
#  Section 1 – Utility helpers
# ---------------------------------------------------------------------
def parse_chain_pairs(spec: str) -> List[Tuple[str, str]]:
  """Convert 'A:B,B:C' → [('A','B'), ('B','C')]."""
  pairs = []
  for token in spec.split(","):
    if ":" not in token:
      raise ValueError(f"Malformed pair '{token}'. Use 'X:Y'.")
    binder, target = token.split(":")
    pairs.append((binder.strip(), target.strip()))
  return pairs


def write_single_chain_pdb(structure, chain_id: str, outfile: Path) -> None:
  """Write only *chain_id* to PDB."""
  from Bio.PDB import PDBIO, Select

  class ChainSelect(Select):
    def accept_chain(self, chain):  # noqa: D401  (Biopython naming)
      return chain.id == chain_id

  io = PDBIO()
  io.set_structure(structure)
  io.save(str(outfile), ChainSelect())  # Path → str avoids .tell() bug


def find_interface_atoms(binder_chain, target_chain, cutoff: float = 4.5):
  """Return (binder_atoms, target_atoms) at the interface (Å cutoff)."""
  binder_atoms = [a for a in binder_chain.get_atoms() if a.element != "H"]
  target_atoms = [a for a in target_chain.get_atoms() if a.element != "H"]

  ns = NeighborSearch(binder_atoms + target_atoms)
  ib, it = set(), set()
  for atom in binder_atoms:
    for nb in ns.search(atom.coord, cutoff, level="A"):
      if nb in target_atoms:
        ib.add(atom)
        it.add(nb)
  return list(ib), list(it)


# ---------------------------------------------------------------------
#  Section 2 – DX parsing & interpolation
# ---------------------------------------------------------------------
_DX_HDR = re.compile(r"^object 1 class gridpositions counts (\d+) (\d+) (\d+)")
_DX_DXY = re.compile(r"^delta\s+([+-]?\d+\.\d+).*")


def _parse_dx(dx_path: Path):
  """
  Parse an APBS/OpenDX potential map.

  Returns
  -------
  origin : (3,) np.ndarray
  delta  : (dx, dy, dz) tuple of floats
  grid   : ndarray [nx, ny, nz]  (kT/e)
  """
  header, numbers = [], []
  with dx_path.open() as fh:
    # ---------------- read header ----------------
    for line in fh:
      header.append(line.rstrip())
      if line.startswith("object 3 class array"):
        break  # start of numeric block declared

    # ---------------- read numeric block ---------
    for line in fh:
      # ignore comment / attribute / component lines
      if not line[0].isdigit() and line[0] not in "+-.":
        continue
      numbers.extend(map(float, line.split()))

  # ---- extract grid size, origin, delta ----------
  for h in header:
    m = _DX_HDR.match(h)
    if m:
      nx, ny, nz = map(int, m.groups())
    if h.startswith("origin"):
      origin = np.fromstring(h.split("origin")[1], sep=" ")

  deltas = []
  for h in header:
    m = _DX_DXY.match(h)
    if m:
      v = list(map(float, h.split()[1:]))
      deltas.append(next(val for val in v if abs(val) > 1e-6))
  if len(deltas) != 3:
    raise RuntimeError(f"Could not parse grid spacing in {dx_path}")

  grid = np.asarray(numbers, dtype=float).reshape((nz, ny, nx)).transpose(2, 1, 0)
  return origin, tuple(deltas), grid


def _sample_potential(coords, origin, delta, grid):
  """Trilinear interpolation (vectorised). Returns potentials[N]."""
  nx, ny, nz = grid.shape
  inv = 1.0 / np.asarray(delta)
  gi = (coords - origin) * inv
  i0 = np.floor(gi).astype(int)
  f = gi - i0
  invalid = (i0[:, 0] < 0) | (i0[:, 1] < 0) | (i0[:, 2] < 0) | (i0[:, 0] >= nx - 1) | (i0[:, 1] >= ny - 1) | (i0[:, 2] >= nz - 1)
  pot = np.full(len(coords), np.nan)
  v = np.where(~invalid)[0]
  if v.size:
    x, y, z = i0[v].T
    fx, fy, fz = f[v].T
    c000 = grid[x, y, z]
    c100 = grid[x + 1, y, z]
    c010 = grid[x, y + 1, z]
    c001 = grid[x, y, z + 1]
    c110 = grid[x + 1, y + 1, z]
    c101 = grid[x + 1, y, z + 1]
    c011 = grid[x, y + 1, z + 1]
    c111 = grid[x + 1, y + 1, z + 1]
    pot[v] = (
      c000 * (1 - fx) * (1 - fy) * (1 - fz)
      + c100 * fx * (1 - fy) * (1 - fz)
      + c010 * (1 - fx) * fy * (1 - fz)
      + c001 * (1 - fx) * (1 - fy) * fz
      + c110 * fx * fy * (1 - fz)
      + c101 * fx * (1 - fy) * fz
      + c011 * (1 - fx) * fy * fz
      + c111 * fx * fy * fz
    )
  return pot


# ---------------------------------------------------------------------
#  Section 3 – External program wrappers
# ---------------------------------------------------------------------
def _run_subprocess(cmd: Sequence[str], cwd: Path, label: str):
  logger.info("%s ▸ %s", label, " ".join(cmd))
  try:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  except subprocess.CalledProcessError as e:
    logger.error("%s failed\nStdout:\n%s\nStderr:\n%s", label, e.stdout.decode(), e.stderr.decode())
    raise


def _prepare_pqr(pdb_path: Path, pqr_path: Path, pdb2pqr_bin: str, forcefield: str):
  cmd = [pdb2pqr_bin, "--ff", forcefield, "--with-ph", "7.0", str(pdb_path), str(pqr_path)]
  _run_subprocess(cmd, cwd=pdb_path.parent, label="PDB2PQR")


# ---------------------------------------------------------------------
#  Section 4 – Bounding-box & APBS runner (robust)
# ---------------------------------------------------------------------
def _bbox_from_pqr(pqr_path: Path, pad: float = 10.0):
  """Return (centre_xyz, length_xyz) of padded bounding box for atoms."""
  coords = []
  with pqr_path.open() as fh:
    for line in fh:
      if line.startswith(("ATOM", "HETATM")):
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
  if not coords:
    raise RuntimeError(f"No atom coords parsed in {pqr_path}")
  xyz = np.asarray(coords)
  xyz_min = xyz.min(axis=0) - pad
  xyz_max = xyz.max(axis=0) + pad
  lengths = xyz_max - xyz_min
  centre = (xyz_max + xyz_min) / 2.0
  return centre, lengths


def _run_apbs(pqr_path: Path, dx_out: Path, apbs_bin: str):
  """Run APBS and guarantee an uncompressed single-rank DX file."""
  centre, lengths = _bbox_from_pqr(pqr_path)
  cx, cy, cz = centre
  lx, ly, lz = lengths
  dime = "161 161 161"

  in_text = f"""
read
    mol pqr {pqr_path.name}
end
elec
    mg-manual
    dime {dime}
    glen {lx:.1f} {ly:.1f} {lz:.1f}
    gcent {cx:.3f} {cy:.3f} {cz:.3f}
    mol 1
    lpbe
    bcfl sdh
    pdie 2.0
    sdie 78.0
    chgm spl2
    srfm smol
    srad 1.4
    swin 0.3
    sdens 10.0
    temp 298.15
    calcenergy no
    calcforce no
    write pot dx {dx_out.stem}
end
quit
"""
  in_path = pqr_path.with_suffix(".in")
  in_path.write_text(in_text)

  _run_subprocess([apbs_bin, in_path.name], cwd=pqr_path.parent, label="APBS")

  # ── locate output ────────────────────────────────────────────────
  pattern = str(pqr_path.parent / f"**/{dx_out.stem}*.dx*")
  candidates = glob.glob(pattern, recursive=True)
  if not candidates:
    raise FileNotFoundError(f"APBS produced no DX file matching '{dx_out.stem}*.dx*'")

  def _pick_best(paths):
    for p in paths:
      if p.endswith(".dx") and "PE" not in p:
        return p
    for p in paths:
      if p.endswith(".dx") and "PE0" in p:
        return p
    return paths[0]

  dx_path = Path(_pick_best(candidates))

  # gunzip if needed
  if dx_path.suffix == ".gz":
    tmp = dx_path.with_suffix("")  # strip .gz
    with gzip.open(dx_path, "rb") as fin, tmp.open("wb") as fout:
      shutil.copyfileobj(fin, fout)
    dx_path = tmp

  # copy / rename to expected name
  if dx_out.resolve() != dx_path.resolve():
    shutil.copyfile(dx_path, dx_out)

  if not dx_out.exists():
    raise FileNotFoundError(f"Unable to materialise '{dx_out}' DX map.")


# ---------------------------------------------------------------------
#  Section 5 – Per-pair EC computation
# ---------------------------------------------------------------------
def compute_ec_for_pair(
  structure,
  binder_id: str,
  target_id: str,
  *,
  cutoff: float,
  forcefield: str,
  pdb2pqr: str,
  apbs: str,
  workdir: Path,
) -> Tuple[float, float, float]:
  """
  Compute electrostatic complementarity (EC) and Pearson correlations (r_b, r_t)
  for a binder–target chain pair in a protein structure.

  Parameters
  ----------
  structure : Bio.PDB.Structure.Structure
    Parsed protein complex containing the binder and target chains.
  binder_id : str
    Chain identifier for the binder.
  target_id : str
    Chain identifier for the target.
  cutoff : float
    Heavy-atom distance cutoff (Å) used to define interface atoms.
  forcefield : str
    PDB2PQR force field name (e.g., AMBER).
  pdb2pqr : str
    Path to the pdb2pqr executable.
  apbs : str
    Path to the apbs executable.
  workdir : Path
    Directory used for temporary PDB/PQR/DX files.

  Returns
  -------
  tuple[float, float, float]
    (ec, r_b, r_t), or (nan, nan, nan) when insufficient interface samples.
  """
  binder_chain = structure[0][binder_id]
  target_chain = structure[0][target_id]

  ib_atoms, it_atoms = find_interface_atoms(binder_chain, target_chain, cutoff)
  if not ib_atoms or not it_atoms:
    logger.warning(f"No inter-chain contacts for {binder_id}:{target_id}, skipping.")
    return np.nan, np.nan, np.nan

  logger.info(
    "Pair %s:%s – %d binder + %d target interface atoms",
    binder_id,
    target_id,
    len(ib_atoms),
    len(it_atoms),
  )

  binder_pdb = workdir / f"binder_{binder_id}.pdb"
  target_pdb = workdir / f"target_{target_id}.pdb"
  write_single_chain_pdb(structure, binder_id, binder_pdb)
  write_single_chain_pdb(structure, target_id, target_pdb)

  binder_pqr = binder_pdb.with_suffix(".pqr")
  target_pqr = target_pdb.with_suffix(".pqr")
  _prepare_pqr(binder_pdb, binder_pqr, pdb2pqr, forcefield)
  _prepare_pqr(target_pdb, target_pqr, pdb2pqr, forcefield)

  binder_dx = binder_pqr.with_suffix(".dx")
  target_dx = target_pqr.with_suffix(".dx")
  _run_apbs(binder_pqr, binder_dx, apbs)
  _run_apbs(target_pqr, target_dx, apbs)

  # ── load potentials
  o_b, d_b, grid_b = _parse_dx(binder_dx)
  o_t, d_t, grid_t = _parse_dx(target_dx)

  V_b_on_b = _sample_potential(np.array([a.coord for a in ib_atoms]), o_b, d_b, grid_b)
  V_t_on_b = _sample_potential(np.array([a.coord for a in ib_atoms]), o_t, d_t, grid_t)
  V_b_on_t = _sample_potential(np.array([a.coord for a in it_atoms]), o_b, d_b, grid_b)
  V_t_on_t = _sample_potential(np.array([a.coord for a in it_atoms]), o_t, d_t, grid_t)

  mask_b = ~np.isnan(V_b_on_b) & ~np.isnan(V_t_on_b)
  mask_t = ~np.isnan(V_b_on_t) & ~np.isnan(V_t_on_t)
  if mask_b.sum() < 10 or mask_t.sum() < 10:
    logger.warning(f"Too few interface samples for {binder_id}:{target_id}, skipping.")
    return np.nan, np.nan, np.nan

  r_b, _ = pearsonr(V_b_on_b[mask_b], V_t_on_b[mask_b])
  r_t, _ = pearsonr(V_b_on_t[mask_t], V_t_on_t[mask_t])
  ec = -(r_b + r_t) / 2.0  # negative correlation = complementarity

  logger.info("Pair %s:%s – EC = %.3f  (r_b=%.3f, r_t=%.3f)", binder_id, target_id, ec, r_b, r_t)
  return ec, r_b, r_t


# ---------------------------------------------------------------------
#  Section 6 – CLI
# ---------------------------------------------------------------------
def main(argv: Sequence[str] | None = None):
  cli = argparse.ArgumentParser(description="Electrostatic complementarity via APBS")
  cli.add_argument("pdb", type=Path, help="Input PDB (protein complex)")
  cli.add_argument("pairs", help="Comma list of binder:target chain IDs (e.g. A:B,B:C)")
  cli.add_argument("--cutoff", type=float, default=4.5, help="Contact cutoff Å (default 4.5)")
  cli.add_argument("--ff", dest="forcefield", default="AMBER", help="PDB2PQR force field (default AMBER)")
  cli.add_argument("--pdb2pqr", default="pdb2pqr", help="Path to pdb2pqr executable")
  cli.add_argument("--apbs", default="apbs", help="Path to apbs executable")
  cli.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
  args = cli.parse_args(argv)

  logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

  if not args.pdb.exists():
    sys.exit(f"PDB file '{args.pdb}' not found.")

  parser = PDBParser(QUIET=True)
  structure = parser.get_structure("complex", str(args.pdb))

  all_chains = {c.id for c in structure.get_chains()}
  chain_pairs = parse_chain_pairs(args.pairs)
  for b, t in chain_pairs:
    if b not in all_chains or t not in all_chains:
      sys.exit(f"Chain '{b}' or '{t}' not present in PDB file.")

  with tempfile.TemporaryDirectory() as td:
    workdir = Path(td)
    results: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for b, t in chain_pairs:
      try:
        ec, r_b, r_t = compute_ec_for_pair(
          structure,
          b,
          t,
          cutoff=args.cutoff,
          forcefield=args.forcefield,
          pdb2pqr=args.pdb2pqr,
          apbs=args.apbs,
          workdir=workdir,
        )
        results[(b, t)] = (ec, r_b, r_t)
      except Exception as exc:
        logger.error("Failed for pair %s:%s – %s", b, t, exc)

  if not results:
    sys.exit("No EC scores computed (see errors above).")

  print("# Electrostatic complementarity (McCoy –r)")
  for (b, t), (ec, r_b, r_t) in results.items():
    print(f"{b}:{t}\t{ec:8.4f}\t{r_b:8.4f}\t{r_t:8.4f}")


if __name__ == "__main__":
  main()