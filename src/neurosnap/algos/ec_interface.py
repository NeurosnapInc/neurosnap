"""
Electrostatic complementarity (EC) quantification for binder-target chain
pairs in protein complexes. Written by Danial Gharaie Amirabadi and Keaun Amani, based on the algorithm from:
Airlie J. McCoy, V. Chandana Epa, and Peter M. Colman. Electrostatic
complementarity at protein/protein interfaces (edited by B. Honig).
Journal of Molecular Biology. 1997;268(2):570–584.

EC is calculated as EC = -(r_b + r_t)/2 where r_b and r_t are the Pearson
correlations of the two partners' electrostatic potentials on the buried
surfaces of the binder and target, respectively. Because of the leading minus
sign, larger (more-positive) EC values correspond to stronger
complementarity (perfect complementarity gives +1, identical surfaces -1).

Key features
------------
* Biopython interface detection using a heavy-atom distance cutoff.
* PDB2PQR protonation plus charge and radius assignment.
* APBS continuum electrostatics (compatible with all 1.x–3.x builds):
  - auto-builds a padded box,
  - supplies required legacy keywords (glen/gcent/sdens),
  - handles .dx.gz, rank-tagged PE files, and subdir output.
* Stepwise logging for debugging and provenance.
* Automatic cleanup of temporary files.
"""

from __future__ import annotations

import glob
import gzip
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr

from neurosnap.log import logger
from neurosnap.protein import Protein

_METRIC_DEFINITIONS_LOGGED = False


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
def _run_subprocess(cmd: Sequence[str], cwd: Path, label: str, verbosity=0):
  if verbosity > 0:
    logger.debug("%s ▸ %s", label, " ".join(cmd))
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
def compute_ec(
  protein: Protein,
  chain1: str,
  chain2: str,
  *,
  cutoff: float = 4.5,
  forcefield: str = "AMBER",
  pdb2pqr: str = "pdb2pqr",
  apbs: str = "apbs",
  verbosity: int = 1,
) -> Tuple[float, float, float]:
  """
  Compute electrostatic complementarity (EC) and Pearson correlations (r_b, r_t)
  for an order-invariant interface chain pair in a Protein.

  Parameters
  ----------
  protein
    Protein containing the interface chains.
  chain1
    Chain identifier for the first interface chain.
  chain2
    Chain identifier for the second interface chain (order does not matter).
  cutoff
    Heavy-atom distance cutoff (Å) used to define interface atoms.
  forcefield
    PDB2PQR force field name (e.g., AMBER).
  pdb2pqr
    Path to the pdb2pqr executable.
  apbs
    Path to the apbs executable.
  verbosity
    Set to 1 for normal verbosity, set to 0 to mute info logs.

  Returns
  -------
  tuple[float, float, float]
    (ec, r_b, r_t), or (nan, nan, nan) when insufficient interface samples.
  """
  contacts = protein.find_interface_contacts(chain1, chain2, cutoff=cutoff, hydrogens=False)
  ib_atoms = list({a for a, _ in contacts})
  it_atoms = list({b for _, b in contacts})
  if not ib_atoms or not it_atoms:
    logger.warning(f"No inter-chain contacts for {chain1}:{chain2}, skipping.")
    return np.nan, np.nan, np.nan

  if verbosity > 0:
    logger.info(
      "Pair %s:%s – %d atoms on chain1 + %d on chain2",
      chain1,
      chain2,
      len(ib_atoms),
      len(it_atoms),
    )

  with tempfile.TemporaryDirectory() as td:
    workdir = Path(td)
    chain1_pdb = workdir / f"chain1_{chain1}.pdb"
    chain2_pdb = workdir / f"chain2_{chain2}.pdb"
    write_single_chain_pdb(protein.structure, chain1, chain1_pdb)
    write_single_chain_pdb(protein.structure, chain2, chain2_pdb)

    chain1_pqr = chain1_pdb.with_suffix(".pqr")
    chain2_pqr = chain2_pdb.with_suffix(".pqr")
    _prepare_pqr(chain1_pdb, chain1_pqr, pdb2pqr, forcefield)
    _prepare_pqr(chain2_pdb, chain2_pqr, pdb2pqr, forcefield)

    chain1_dx = chain1_pqr.with_suffix(".dx")
    chain2_dx = chain2_pqr.with_suffix(".dx")
    _run_apbs(chain1_pqr, chain1_dx, apbs)
    _run_apbs(chain2_pqr, chain2_dx, apbs)

    # ── load potentials
    o_b, d_b, grid_b = _parse_dx(chain1_dx)
    o_t, d_t, grid_t = _parse_dx(chain2_dx)

    V_b_on_b = _sample_potential(np.array([a.coord for a in ib_atoms]), o_b, d_b, grid_b)
    V_t_on_b = _sample_potential(np.array([a.coord for a in ib_atoms]), o_t, d_t, grid_t)
    V_b_on_t = _sample_potential(np.array([a.coord for a in it_atoms]), o_b, d_b, grid_b)
    V_t_on_t = _sample_potential(np.array([a.coord for a in it_atoms]), o_t, d_t, grid_t)

    mask_b = ~np.isnan(V_b_on_b) & ~np.isnan(V_t_on_b)
    mask_t = ~np.isnan(V_b_on_t) & ~np.isnan(V_t_on_t)
    if mask_b.sum() < 10 or mask_t.sum() < 10:
      logger.warning(f"Too few interface samples for {chain1}:{chain2}, skipping.")
      return np.nan, np.nan, np.nan

    r_b, _ = pearsonr(V_b_on_b[mask_b], V_t_on_b[mask_b])
    r_t, _ = pearsonr(V_b_on_t[mask_t], V_t_on_t[mask_t])
    ec = -(r_b + r_t) / 2.0  # negative correlation = complementarity

    if verbosity > 0:
      logger.info("Pair %s:%s | EC=%.4f, RB=%.4f, RT=%.4f", chain1, chain2, ec, r_b, r_t)
      global _METRIC_DEFINITIONS_LOGGED
      if not _METRIC_DEFINITIONS_LOGGED:
        logger.info("Definitions:")
        logger.info(
          "EC: Electrostatic complementarity on the buried surfaces of chain1 and chain2; the minus sign means more-positive EC values indicating stronger complementarity (perfect complementarity = +1, identical surfaces = -1)."
        )
        logger.info("RB: correlation of chain1 potential vs chain2 potential on chain1 interface atoms")
        logger.info("RT: correlation of chain1 potential vs chain2 potential on chain2 interface atoms (Pearson(V_b_on_t, V_t_on_t)).")
        _METRIC_DEFINITIONS_LOGGED = True
    return ec, r_b, r_t
