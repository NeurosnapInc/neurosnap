"""
A Python implementation of the EvoEF2 protein scoring function / force field.
Ported from the native EvoEF2 reference implementation.
Original Implementation: https://github.com/tommyhuangthu/EvoEF2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from neurosnap.algos.evoef2_lib.weights import get_weights
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from neurosnap.log import logger
from neurosnap.protein import Protein

# -----------------------------
# Constants (mirrors EvoEF2)
# -----------------------------
MAX_EVOEF_ENERGY_TERM_NUM = 100

ENERGY_DISTANCE_CUTOFF = 6.0
ENERGY_SCALE_FACTOR_BOND_123 = 0.0
ENERGY_SCALE_FACTOR_BOND_14 = 0.2
ENERGY_SCALE_FACTOR_BOND_15 = 1.0
RADIUS_SCALE_FOR_VDW = 0.95

HBOND_DISTANCE_CUTOFF_MAX = 3.0
HBOND_WELL_DEPTH = 1.0
HBOND_OPTIMAL_DISTANCE = 1.9
HBOND_LOCAL_REDUCE = 0.5

ELEC_DISTANCE_CUTOFF = 6.0
COULOMB_CONSTANT = 332.0
DIELECTRIC_CONST_PROTEIN = 8.0
DIELECTRIC_CONSTANT_WATER = 80.0
DIELECTRIC_CONST_PROTEIN_AVE = 20.0
IONIC_STRENGTH = 0.05
PROTEIN_DESIGN_TEMPERATURE = 298

LK_SOLV_DISTANCE_CUTOFF = 6.0
RADIUS_SCALE_FOR_DESOLV = 1.00

SSBOND_DISTANCE = 2.03
SSBOND_ANGLE = 105.0
SSBOND_TORSION = 90.0
SSBOND_CUTOFF_MAX = 2.15
SSBOND_CUTOFF_MIN = 1.95

PI = math.pi

# Energy term indices and names
ENERGY_TERM_NAMES = {
  0: "total",
  1: "reference_ALA",
  2: "reference_CYS",
  3: "reference_ASP",
  4: "reference_GLU",
  5: "reference_PHE",
  6: "reference_GLY",
  7: "reference_HIS",
  8: "reference_ILE",
  9: "reference_LYS",
  10: "reference_LEU",
  11: "reference_MET",
  12: "reference_ASN",
  13: "reference_PRO",
  14: "reference_GLN",
  15: "reference_ARG",
  16: "reference_SER",
  17: "reference_THR",
  18: "reference_VAL",
  19: "reference_TRP",
  20: "reference_TYR",
  21: "intraR_vdwatt",
  22: "intraR_vdwrep",
  23: "intraR_electr",
  24: "intraR_deslvP",
  25: "intraR_deslvH",
  26: "intraR_hbscbb_dis",
  27: "intraR_hbscbb_the",
  28: "intraR_hbscbb_phi",
  31: "interS_vdwatt",
  32: "interS_vdwrep",
  33: "interS_electr",
  34: "interS_deslvP",
  35: "interS_deslvH",
  36: "interS_ssbond",
  41: "interS_hbbbbb_dis",
  42: "interS_hbbbbb_the",
  43: "interS_hbbbbb_phi",
  44: "interS_hbscbb_dis",
  45: "interS_hbscbb_the",
  46: "interS_hbscbb_phi",
  47: "interS_hbscsc_dis",
  48: "interS_hbscsc_the",
  49: "interS_hbscsc_phi",
  51: "interD_vdwatt",
  52: "interD_vdwrep",
  53: "interD_electr",
  54: "interD_deslvP",
  55: "interD_deslvH",
  56: "interD_ssbond",
  61: "interD_hbbbbb_dis",
  62: "interD_hbbbbb_the",
  63: "interD_hbbbbb_phi",
  64: "interD_hbscbb_dis",
  65: "interD_hbscbb_the",
  66: "interD_hbscbb_phi",
  67: "interD_hbscsc_dis",
  68: "interD_hbscsc_the",
  69: "interD_hbscsc_phi",
  71: "ligand_vdwatt",
  72: "ligand_vdwrep",
  73: "ligand_electr",
  74: "ligand_deslvP",
  75: "ligand_deslvH",
  81: "ligand_hbscbb_dis_raw",
  82: "ligand_hbscbb_the_raw",
  83: "ligand_hbscbb_phi_raw",
  84: "ligand_hbscbb_dis",
  85: "ligand_hbscbb_the",
  86: "ligand_hbscbb_phi",
  87: "ligand_hbscsc_dis",
  88: "ligand_hbscsc_the",
  89: "ligand_hbscsc_phi",
  91: "aapropensity",
  92: "ramachandran",
  93: "dunbrack",
}

ENERGY_TERM_ORDER = [k for k in sorted(ENERGY_TERM_NAMES.keys()) if k != 0]

AA_ONE_LETTER = "ACDEFGHIKLMNPQRSTVWY"
AA_THREE_TO_ONE = {
  "ALA": "A",
  "CYS": "C",
  "ASP": "D",
  "GLU": "E",
  "PHE": "F",
  "GLY": "G",
  "HIS": "H",
  "HSE": "H",
  "HSD": "H",
  "HSP": "H",
  "ILE": "I",
  "LYS": "K",
  "LEU": "L",
  "MET": "M",
  "ASN": "N",
  "PRO": "P",
  "GLN": "Q",
  "ARG": "R",
  "SER": "S",
  "THR": "T",
  "VAL": "V",
  "TRP": "W",
  "TYR": "Y",
}

# -----------------------------
# Data structures
# -----------------------------


@dataclass
class AtomParam:
  """Per-atom parameter record loaded from the EvoEF2 parameter library.

  Attributes:
    name: Atom name as defined in topology.
    type: Atom type in the parameter file.
    is_bb: Whether the atom is considered backbone for scoring.
    polarity: Polarity class used in desolvation.
    epsilon: Lennard-Jones well depth (kcal/mol).
    radius: Lennard-Jones radius (Angstrom).
    charge: Partial charge (e).
    hb_h_or_a: Hydrogen-bond role flag: H (hydrogen) or A (acceptor).
    hb_d_or_b: Hydrogen-bond donor/base classification.
    hb_b2: Secondary hydrogen-bond base flag.
    hybrid: Hybridization class (e.g., sp2/sp3).
    eef1_free_dg: EEF1 free energy parameter.
    eef1_volume: EEF1 volume parameter.
    eef1_lambda: EEF1 lambda parameter.
  """
  name: str
  type: str
  is_bb: bool
  polarity: str
  epsilon: float
  radius: float
  charge: float
  hb_h_or_a: str
  hb_d_or_b: str
  hb_b2: str
  hybrid: str
  eef1_free_dg: float
  eef1_volume: float
  eef1_lambda: float

  @property
  def is_hbond_h(self) -> bool:
    return self.hb_h_or_a == "H"

  @property
  def is_hbond_a(self) -> bool:
    return self.hb_h_or_a == "A"


@dataclass
class Atom:
  """Atom instance with coordinates, parameters, and per-structure state.

  Attributes:
    name: Atom name (e.g., CA, CB, O).
    param: Parameter record used for scoring.
    chain: Chain identifier for the parent residue.
    pos: Residue index within chain (0-based in this structure).
    res: Parent residue reference (for H-bond donor/base lookup).
    xyz: Cartesian coordinate in Angstrom.
    is_xyz_valid: Whether the coordinate is present or reconstructed.
    is_in_hbond: Flag used during H-bond detection to avoid double counting.
  """
  name: str
  param: AtomParam
  chain: str
  pos: int
  res: Optional["Residue"] = None
  xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
  is_xyz_valid: bool = False
  is_in_hbond: bool = False

  @property
  def vdw_epsilon(self) -> float:
    return self.param.epsilon

  @property
  def vdw_radius(self) -> float:
    return self.param.radius

  @property
  def charge(self) -> float:
    return self.param.charge

  @property
  def eef1_free_dg(self) -> float:
    return self.param.eef1_free_dg

  @property
  def eef1_volume(self) -> float:
    return self.param.eef1_volume

  @property
  def eef1_lambda(self) -> float:
    return self.param.eef1_lambda

  @property
  def hb_h_or_a(self) -> str:
    return self.param.hb_h_or_a

  @property
  def hb_d_or_b(self) -> str:
    return self.param.hb_d_or_b

  @property
  def hb_b2(self) -> str:
    return self.param.hb_b2

  @property
  def hybrid(self) -> str:
    return self.param.hybrid

  @property
  def is_bb(self) -> bool:
    return self.param.is_bb

  @property
  def polarity(self) -> str:
    return self.param.polarity

  @property
  def is_h(self) -> bool:
    return self.name.startswith("H")

  @property
  def is_hbond_h(self) -> bool:
    return self.param.is_hbond_h

  @property
  def is_hbond_a(self) -> bool:
    return self.param.is_hbond_a


@dataclass
class Bond:
  """Covalent bond between two atoms inside a residue."""
  a: str
  b: str
  bond_type: int = 1


@dataclass
class CharmmIC:
  """CHARMM internal coordinate (IC) entry used to build missing atoms."""
  atom_a: str
  atom_b: str
  atom_c: str
  atom_d: str
  ic_param: List[float]
  torsion_proper: bool


@dataclass
class ResidueTopology:
  """Residue topology template with atoms, bonds, and ICs from library."""
  name: str
  atoms: List[str] = field(default_factory=list)
  deletes: List[str] = field(default_factory=list)
  bonds: List[Bond] = field(default_factory=list)
  ics: List[CharmmIC] = field(default_factory=list)


@dataclass
class Residue:
  """Residue instance with atoms, bonds, and cached torsions/geometry."""
  name: str
  chain: str
  pos: int
  atoms: Dict[str, Atom] = field(default_factory=dict)
  bonds: List[Bond] = field(default_factory=list)
  patches: List[str] = field(default_factory=list)
  phipsi: Tuple[float, float] = (0.0, 0.0)
  n_cb_in_8a: int = 0
  is_protein: bool = True
  xtorsions: List[float] = field(default_factory=list)

  def get_atom(self, name: str) -> Optional[Atom]:
    """Return an atom by name if present."""
    return self.atoms.get(name)


@dataclass
class Chain:
  """Chain container for residues with an is_protein flag."""
  name: str
  residues: List[Residue] = field(default_factory=list)
  is_protein: bool = True


@dataclass
class Structure:
  """Structure container holding all chains."""
  chains: List[Chain] = field(default_factory=list)

  def all_residues(self) -> Iterable[Residue]:
    """Iterate over all residues across all chains."""
    for chain in self.chains:
      for res in chain.residues:
        yield res


@dataclass
class AAppTable:
  """Amino-acid propensity table indexed by phi/psi bins."""
  aap: np.ndarray  # shape (36,36,20)


@dataclass
class RamaTable:
  """Ramachandran probability table indexed by phi/psi bins."""
  rama: np.ndarray  # shape (36,36,20)


_AAPP_TABLE: Optional[AAppTable] = None
_RAMA_TABLE: Optional[RamaTable] = None


# -----------------------------
# Geometry utilities
# -----------------------------


def safe_acos(cos_value: float) -> float:
  """Return acos with inputs clamped to [-1, 1] to avoid numeric blowups.

  Args:
    cos_value: Cosine value to clamp.

  Returns:
    acos(cos_value) in radians.
  """
  if cos_value > 1.0:
    cos_value = 1.0
  elif cos_value < -1.0:
    cos_value = -1.0
  return math.acos(cos_value)


def rad_to_deg(rad: float) -> float:
  """Convert radians to degrees.

  Args:
    rad: Angle in radians.

  Returns:
    Angle in degrees.
  """
  return rad * 180.0 / PI


def deg_to_rad(deg: float) -> float:
  """Convert degrees to radians.

  Args:
    deg: Angle in degrees.

  Returns:
    Angle in radians.
  """
  return deg * PI / 180.0


def xyz_distance(a: np.ndarray, b: np.ndarray) -> float:
  """Return Euclidean distance between two points.

  Args:
    a: First point.
    b: Second point.

  Returns:
    Distance in Angstrom.
  """
  return float(np.linalg.norm(a - b))


def xyz_angle(v1: np.ndarray, v2: np.ndarray) -> float:
  """Return the angle between two vectors in radians.

  Args:
    v1: First vector.
    v2: Second vector.

  Returns:
    Angle in radians.
  """
  norm = np.linalg.norm(v1) * np.linalg.norm(v2)
  if norm < 1e-12:
    return 1000.0
  return safe_acos(float(np.dot(v1, v2) / norm))


def xyz_rotate_around(p: np.ndarray, axis_from: np.ndarray, axis_to: np.ndarray, angle: float) -> np.ndarray:
  """Rotate a point around an axis defined by two points.

  Args:
    p: Point to rotate.
    axis_from: First point on rotation axis.
    axis_to: Second point on rotation axis.
    angle: Rotation angle in radians.

  Returns:
    Rotated point in Cartesian space.
  """
  s = math.sin(angle)
  c = math.cos(angle)
  n = axis_from - axis_to
  norm = np.linalg.norm(n)
  if norm < 1e-12:
    return p.copy()
  n = n / norm
  result = p - axis_from
  a00 = n[0] * n[0] + (1 - n[0] * n[0]) * c
  a01 = n[0] * n[1] * (1 - c) - n[2] * s
  a02 = n[0] * n[2] * (1 - c) + n[1] * s
  a10 = n[0] * n[1] * (1 - c) + n[2] * s
  a11 = n[1] * n[1] + (1 - n[1] * n[1]) * c
  a12 = n[1] * n[2] * (1 - c) - n[0] * s
  a20 = n[0] * n[2] * (1 - c) - n[1] * s
  a21 = n[1] * n[2] * (1 - c) + n[0] * s
  a22 = n[2] * n[2] + (1 - n[2] * n[2]) * c
  m = np.array([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]], dtype=float)
  result = result @ m
  return result + axis_from


def get_fourth_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray, ic_param: Sequence[float]) -> np.ndarray:
  """Compute the fourth atom position using internal coordinates (IC).

  Args:
    a: Atom A coordinates.
    b: Atom B coordinates.
    c: Atom C coordinates.
    ic_param: IC parameters (angle/torsion/length) from topology.

  Returns:
    Coordinates for atom D implied by IC parameters.
  """
  ba = b - a
  bc = b - c
  ba_x_bc = np.cross(ba, bc)
  if np.linalg.norm(ba_x_bc) < 1e-12:
    raise ValueError("Zero division in GetFourthAtom")
  angle_abc = xyz_angle(ba, bc)
  d = xyz_rotate_around(a, b, ba_x_bc + b, ic_param[3] - (PI - angle_abc))
  d = xyz_rotate_around(d, b, c, ic_param[2])
  d = d - b
  d = d / np.linalg.norm(d) * ic_param[4]
  d = d + c
  return d


def get_torsion_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
  """Return signed torsion angle for four points.

  The sign convention is matched to the EvoEF2 / Bio.PDB implementation.
  """
  r_ab = a - b
  r_bc = b - c
  r_cd = c - d
  r_ab_x_rbc = np.cross(r_ab, r_bc)
  r_bc_x_rcd = np.cross(r_bc, r_cd)
  r_cd_x_rab = np.cross(r_cd, r_ab)
  norm1 = np.linalg.norm(r_ab_x_rbc)
  norm2 = np.linalg.norm(r_bc_x_rcd)
  if norm1 < 1e-12 or norm2 < 1e-12:
    return 0.0
  cos_value = float(np.dot(r_ab_x_rbc, r_bc_x_rcd) / norm1 / norm2)
  sin_value = float(np.dot(r_bc, r_cd_x_rab))
  angle = safe_acos(cos_value)
  if sin_value < 0:
    angle = -angle
  return -angle


# -----------------------------
# Parsing of EvoEF2 libraries
# -----------------------------


def _default_evoef2_root() -> Path:
  """Return the default path to the bundled EvoEF2 data directory."""
  return Path(__file__).resolve().parent / "evoef2_lib"


def load_atom_params(param_path: Optional[Path] = None) -> Dict[str, Dict[str, AtomParam]]:
  """Load atom parameters from the EvoEF2 CHARMM19+LK parameter file.

  Args:
    param_path: Optional explicit path to the parameter file.

  Returns:
    Mapping of residue name to atom name to AtomParam.
  """
  if param_path is None:
    param_path = _default_evoef2_root() / "param_charmm19_lk.prm"
  param_map: Dict[str, Dict[str, AtomParam]] = {}
  with open(param_path, "r") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("!"):
        continue
      parts = line.split()
      if len(parts) < 15:
        continue
      res_name = parts[0]
      atom_name = parts[1]
      atom_type = parts[2]
      is_bb = parts[3] == "Y"
      polar = parts[4]
      epsilon = float(parts[5])
      radius = float(parts[6])
      charge = float(parts[7])
      hb_h_or_a = parts[8]
      hb_d_or_b = parts[9]
      hb_b2 = parts[10]
      hybrid = parts[11]
      dg_free = float(parts[12])
      volume = float(parts[13])
      lam = float(parts[14]) if len(parts) > 14 else 0.0
      param = AtomParam(
        name=atom_name,
        type=atom_type,
        is_bb=is_bb,
        polarity=polar,
        epsilon=epsilon,
        radius=radius,
        charge=charge,
        hb_h_or_a=hb_h_or_a,
        hb_d_or_b=hb_d_or_b,
        hb_b2=hb_b2,
        hybrid=hybrid,
        eef1_free_dg=dg_free,
        eef1_volume=volume,
        eef1_lambda=lam,
      )
      param_map.setdefault(res_name, {})[atom_name] = param
  return param_map


def load_topology(top_path: Optional[Path] = None) -> Dict[str, ResidueTopology]:
  """Load CHARMM topology definitions for residues and patches.

  Args:
    top_path: Optional explicit path to the topology file.

  Returns:
    Mapping of residue/patch name to topology template.
  """
  if top_path is None:
    top_path = _default_evoef2_root() / "top_polh19_prot.inp"
  topologies: Dict[str, ResidueTopology] = {}
  current: Optional[ResidueTopology] = None
  with open(top_path, "r") as f:
    for raw in f:
      line = raw.strip()
      if "!" in line:
        line = line.split("!", 1)[0].strip()
      if not line or line.startswith("!") or line.startswith("*"):
        continue
      parts = line.split()
      if not parts:
        continue
      keyword = parts[0]
      if keyword in {"RESI", "PRES"}:
        if len(parts) >= 2:
          name = parts[1]
          current = ResidueTopology(name=name)
          topologies[name] = current
        continue
      if current is None:
        continue
      if keyword == "ATOM" and len(parts) >= 2:
        current.atoms.append(parts[1])
      elif keyword == "DELETE" and len(parts) >= 3 and parts[1] == "ATOM":
        current.deletes.append(parts[2])
      elif keyword == "BOND":
        atoms = parts[1:]
        for i in range(0, len(atoms) - 1, 2):
          current.bonds.append(Bond(atoms[i], atoms[i + 1]))
      elif keyword == "IC":
        if len(parts) >= 10:
          atom_a = parts[1]
          atom_b = parts[2]
          atom_c = parts[3]
          atom_d = parts[4]
          torsion_proper = True
          if atom_c.startswith("*"):
            torsion_proper = False
            atom_c = atom_c[1:]
          ic_param = [0.0] * 5
          ic_param[0] = float(parts[5])
          ic_param[1] = deg_to_rad(float(parts[6]))
          ic_param[2] = deg_to_rad(float(parts[7]))
          ic_param[3] = deg_to_rad(float(parts[8]))
          ic_param[4] = float(parts[9])
          current.ics.append(
            CharmmIC(
              atom_a=atom_a,
              atom_b=atom_b,
              atom_c=atom_c,
              atom_d=atom_d,
              ic_param=ic_param,
              torsion_proper=torsion_proper,
            )
          )
  return topologies


def load_aapropensity(path: Optional[Path] = None) -> AAppTable:
  """Load amino-acid propensity table for phi/psi bins.

  Args:
    path: Optional explicit path to the table file.

  Returns:
    AAppTable instance with a [36, 36, 20] tensor.
  """
  global _AAPP_TABLE
  if _AAPP_TABLE is not None and path is None:
    return _AAPP_TABLE
  if path is None:
    path = _default_evoef2_root() / "aapropensity.npz"
  data = np.load(path)["data"]
  table = AAppTable(aap=data)
  if path == _default_evoef2_root() / "aapropensity.npz":
    _AAPP_TABLE = table
  return table


def load_ramachandran(path: Optional[Path] = None) -> RamaTable:
  """Load Ramachandran probability table for phi/psi bins.

  Args:
    path: Optional explicit path to the table file.

  Returns:
    RamaTable instance with a [36, 36, 20] tensor.
  """
  global _RAMA_TABLE
  if _RAMA_TABLE is not None and path is None:
    return _RAMA_TABLE
  if path is None:
    path = _default_evoef2_root() / "ramachandran.npz"
  data = np.load(path)["data"]
  table = RamaTable(rama=data)
  if path == _default_evoef2_root() / "ramachandran.npz":
    _RAMA_TABLE = table
  return table


def _default_dunbrack_path() -> Path:
  """Return the default Dunbrack library path."""
  return _default_evoef2_root() / "dun2010bb3per.lib"


@dataclass
class DunbrackRotamer:
  """Single Dunbrack rotamer entry with chi statistics."""
  torsions: List[float]
  deviations: List[float]
  probability: float


@dataclass
class DunbrackBin:
  """Dunbrack bin for a phi/psi region with rotamer list."""
  by_residue: Dict[str, List[DunbrackRotamer]] = field(default_factory=dict)


@dataclass
class DunbrackLibrary:
  """Full Dunbrack library indexed by residue and phi/psi bins."""
  bins: List[DunbrackBin]


_DUNBRACK_TORSION_COUNT = {
  "ALA": 0,
  "ARG": 4,
  "ASN": 2,
  "ASP": 2,
  "CYS": 1,
  "GLN": 3,
  "GLU": 3,
  "GLY": 0,
  "HSD": 2,
  "HSE": 2,
  "ILE": 2,
  "LEU": 2,
  "LYS": 4,
  "MET": 3,
  "PHE": 2,
  "PRO": 2,
  "SER": 1,
  "THR": 1,
  "TRP": 2,
  "TYR": 2,
  "VAL": 1,
}


def load_dunbrack(path: Optional[Path] = None) -> DunbrackLibrary:
  """Load Dunbrack rotamer library from the EvoEF2 distribution.

  Args:
    path: Optional explicit path to the Dunbrack library file.

  Returns:
    DunbrackLibrary with bins indexed by phi/psi.
  """
  if path is None:
    path = _default_dunbrack_path()
  bins = [DunbrackBin() for _ in range(36 * 36)]
  with open(path, "r") as f:
    for raw in f:
      line = raw.strip()
      if not line or line.startswith("#") or line.startswith(" "):
        continue
      parts = line.split()
      if len(parts) < 17:
        continue
      resname = parts[0]
      phi = int(parts[1])
      psi = int(parts[2])
      if phi == 180 and psi == 180:
        continue
      prob = float(parts[8])
      x = [float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12])]
      s = [float(parts[13]), float(parts[14]), float(parts[15]), float(parts[16])]
      xcount = _DUNBRACK_TORSION_COUNT.get(resname, 0)
      torsions = [deg_to_rad(v) for v in x[:xcount]]
      deviations = [deg_to_rad(v) for v in s[:xcount]]
      bin_index = ((phi + 180) // 10) * 36 + ((psi + 180) // 10)
      if bin_index < 0 or bin_index >= len(bins):
        continue
      bins[bin_index].by_residue.setdefault(resname, []).append(
        DunbrackRotamer(torsions=torsions, deviations=deviations, probability=prob)
      )
  return DunbrackLibrary(bins=bins)


# -----------------------------
# Topology and atom reconstruction
# -----------------------------


def _residue_intra_bond_12(atom1: str, atom2: str, bonds: List[Bond]) -> bool:
  """Return True if atom1-atom2 is a direct covalent bond.

  Args:
    atom1: Atom name.
    atom2: Atom name.
    bonds: Residue bond list.

  Returns:
    True if atom1 and atom2 are bonded (1-2).
  """
  for bond in bonds:
    if (atom1 == bond.a and atom2 == bond.b) or (atom2 == bond.a and atom1 == bond.b):
      return True
  return False


def _residue_intra_bond_13(atom1: str, atom2: str, bonds: List[Bond]) -> bool:
  """Return True if atom1-atom2 is separated by two covalent bonds (1-3).

  Args:
    atom1: Atom name.
    atom2: Atom name.
    bonds: Residue bond list.

  Returns:
    True if atom1 and atom2 are 1-3 connected.
  """
  for bond in bonds:
    if atom1 == bond.a:
      if _residue_intra_bond_12(bond.b, atom2, bonds):
        return True
    elif atom1 == bond.b:
      if _residue_intra_bond_12(bond.a, atom2, bonds):
        return True
  return False


def _residue_intra_bond_14(atom1: str, atom2: str, bonds: List[Bond]) -> bool:
  """Return True if atom1-atom2 is separated by three covalent bonds (1-4).

  Args:
    atom1: Atom name.
    atom2: Atom name.
    bonds: Residue bond list.

  Returns:
    True if atom1 and atom2 are 1-4 connected.
  """
  for bond in bonds:
    if atom1 == bond.a:
      if _residue_intra_bond_13(bond.b, atom2, bonds):
        return True
    elif atom1 == bond.b:
      if _residue_intra_bond_13(bond.a, atom2, bonds):
        return True
  return False


def _residue_intra_bond_connection(atom1: str, atom2: str, bonds: List[Bond]) -> int:
  """Return bond connection category: 12, 13, 14, or 15 (nonbonded).

  Args:
    atom1: Atom name.
    atom2: Atom name.
    bonds: Residue bond list.

  Returns:
    Integer category (12/13/14/15).
  """
  if _residue_intra_bond_12(atom1, atom2, bonds):
    return 12
  if _residue_intra_bond_13(atom1, atom2, bonds):
    return 13
  if _residue_intra_bond_14(atom1, atom2, bonds):
    return 14
  return 15


_ATOM_ORDER_SEQUENCE = "ABGDEZ"


def _protein_atom_order(atom_name: str) -> int:
  """Return atom ordering index used to detect sidechain torsions.

  Args:
    atom_name: Atom name.

  Returns:
    Order index or -1 if not part of the sidechain torsion sequence.
  """
  if atom_name.startswith("H"):
    return -1
  can_be = atom_name[-1]
  if can_be.isdigit():
    if can_be == "1" and len(atom_name) >= 2:
      can_be = atom_name[-2]
    else:
      return -1
  try:
    return _ATOM_ORDER_SEQUENCE.index(can_be)
  except ValueError:
    return -1


def residue_calc_sidechain_torsions(res: Residue, topologies: Dict[str, ResidueTopology]) -> None:
  """Compute and cache sidechain torsion angles for a residue.

  Args:
    res: Residue to annotate.
    topologies: Residue topologies used to identify torsions.
  """
  res.xtorsions.clear()
  torsion_count = _DUNBRACK_TORSION_COUNT.get(res.name, 0)
  if torsion_count == 0:
    return
  topo = topologies.get(res.name)
  if topo is None:
    return
  for torsion_index in range(torsion_count):
    desired_b = torsion_index
    desired_c = torsion_index + 1
    ic_found = None
    for ic in topo.ics:
      atom_b_order = _protein_atom_order(ic.atom_b)
      atom_c_order = _protein_atom_order(ic.atom_c)
      if atom_b_order == desired_b and atom_c_order == desired_c:
        ic_found = ic
        break
    if ic_found is None:
      return
    a = res.get_atom(ic_found.atom_a)
    b = res.get_atom(ic_found.atom_b)
    c = res.get_atom(ic_found.atom_c)
    d = res.get_atom(ic_found.atom_d)
    if a is None or b is None or c is None or d is None:
      return
    if not (a.is_xyz_valid and b.is_xyz_valid and c.is_xyz_valid and d.is_xyz_valid):
      return
    torsion = get_torsion_angle(a.xyz, b.xyz, c.xyz, d.xyz)
    res.xtorsions.append(torsion)


def _residue_and_next_residue_bond_type(atom_pre: str, atom_next: str, next_res_name: str) -> int:
  """Return bond connection category for atoms in adjacent residues.

  Args:
    atom_pre: Atom name in previous residue.
    atom_next: Atom name in next residue.
    next_res_name: Next residue name for special PRO handling.

  Returns:
    Integer category (12/13/14/15).
  """
  # charmm19 rules from EvoEF2
  if atom_pre == "C":
    if atom_next == "N":
      return 12
    if atom_next in {"CA", "H"} or (atom_next == "CD" and next_res_name == "PRO"):
      return 13
    if atom_next in {"CB", "C"} or (atom_next == "CG" and next_res_name == "PRO"):
      return 14
  elif atom_pre in {"O", "CA"}:
    if atom_next == "N":
      return 13
    if atom_next in {"CA", "H"} or (atom_next == "CD" and next_res_name == "PRO"):
      return 14
  elif atom_pre in {"CB", "N"}:
    if atom_next == "N":
      return 14
  return 15


def _find_ic_for_atom(res: Residue, topologies: Dict[str, ResidueTopology], atom_name: str) -> Optional[CharmmIC]:
  """Find the IC entry that defines how to build an atom.

  Args:
    res: Residue to search.
    topologies: Topology templates.
    atom_name: Target atom name.

  Returns:
    Matching IC entry or None.
  """
  # check patches first
  for patch in res.patches:
    topo = topologies.get(patch)
    if topo is None:
      continue
    for ic in topo.ics:
      if ic.atom_d == atom_name:
        return ic
  topo = topologies.get(res.name)
  if topo is None:
    return None
  for ic in topo.ics:
    if ic.atom_d == atom_name:
      return ic
  return None


def _get_atom_xyz(res: Residue, name: str) -> Optional[np.ndarray]:
  """Return atom coordinates if present and valid.

  Args:
    res: Residue containing the atom.
    name: Atom name.

  Returns:
    Coordinate array if present and valid, otherwise None.
  """
  atom = res.get_atom(name)
  if atom is None or not atom.is_xyz_valid:
    return None
  return atom.xyz


def _calc_atom_xyz(res: Residue, topologies: Dict[str, ResidueTopology], prev_res: Optional[Residue], next_res: Optional[Residue], atom_name: str) -> Optional[np.ndarray]:
  """Compute coordinates for a missing atom using ICs and neighbors.

  Args:
    res: Residue owning the atom.
    topologies: Topology templates.
    prev_res: Previous residue (if any).
    next_res: Next residue (if any).
    atom_name: Atom name to reconstruct.

  Returns:
    Computed coordinates or None if insufficient references.
  """
  ic = _find_ic_for_atom(res, topologies, atom_name)
  if ic is None:
    return None
  names = [ic.atom_a, ic.atom_b, ic.atom_c]
  coords: List[np.ndarray] = []
  for n in names:
    if n.startswith("-") and prev_res is not None:
      xyz = _get_atom_xyz(prev_res, n[1:])
    elif n.startswith("+") and next_res is not None:
      xyz = _get_atom_xyz(next_res, n[1:])
    else:
      xyz = _get_atom_xyz(res, n)
    if xyz is None:
      return None
    coords.append(xyz)
  try:
    return get_fourth_atom(coords[0], coords[1], coords[2], ic.ic_param)
  except ValueError:
    return None


def residue_calc_all_atom_xyz(res: Residue, topologies: Dict[str, ResidueTopology], prev_res: Optional[Residue], next_res: Optional[Residue]) -> None:
  """Rebuild all missing atoms in a residue using IC parameters.

  Args:
    res: Residue to rebuild.
    topologies: Topology templates.
    prev_res: Previous residue (if any).
    next_res: Next residue (if any).
  """
  done = False
  while not done:
    done = True
    for atom in res.atoms.values():
      if atom.is_xyz_valid:
        continue
      new_xyz = _calc_atom_xyz(res, topologies, prev_res, next_res, atom.name)
      if new_xyz is None:
        continue
      atom.xyz = new_xyz
      atom.is_xyz_valid = True
      done = False


def chain_calc_all_atom_xyz(chain: Chain, topologies: Dict[str, ResidueTopology]) -> None:
  """Rebuild missing atoms for every residue in a chain.

  Args:
    chain: Chain to rebuild.
    topologies: Topology templates.
  """
  for i, res in enumerate(chain.residues):
    prev_res = chain.residues[i - 1] if i > 0 else None
    next_res = chain.residues[i + 1] if i < len(chain.residues) - 1 else None
    residue_calc_all_atom_xyz(res, topologies, prev_res, next_res)


def _apply_patch(res: Residue, patch_name: str, params: Dict[str, Dict[str, AtomParam]], topologies: Dict[str, ResidueTopology], delete_o: bool = True) -> None:
  """Apply a topology patch (NTER/CTER/disulfide) to a residue.

  Args:
    res: Residue to modify.
    patch_name: Patch name in topology (e.g., NTER, CTER, GLYP).
    params: Parameter table for atoms.
    topologies: Topology templates.
    delete_o: Whether to delete terminal O atom when patching.
  """
  topo = topologies.get(patch_name)
  if topo is None:
    raise ValueError(f"Missing topology for patch {patch_name}")
  # delete atoms
  for atom_name in topo.deletes:
    if not delete_o and atom_name == "O":
      continue
    res.atoms.pop(atom_name, None)
  # add atoms from patch params
  if patch_name in params:
    for atom_name, param in params[patch_name].items():
      if atom_name in res.atoms:
        atom = res.atoms[atom_name]
        xyz = atom.xyz.copy()
        valid = atom.is_xyz_valid
        res.atoms[atom_name] = Atom(
          name=atom_name,
          param=param,
          chain=res.chain,
          pos=res.pos,
          res=res,
          xyz=xyz,
          is_xyz_valid=valid,
        )
      else:
        res.atoms[atom_name] = Atom(name=atom_name, param=param, chain=res.chain, pos=res.pos, res=res)
  # record patch order (head)
  res.patches.insert(0, patch_name)
  # add bonds
  for bond in topo.bonds:
    res.bonds.append(bond)
  # remove bonds to deleted atoms (unless +/-)
  new_bonds: List[Bond] = []
  for bond in res.bonds:
    a = bond.a
    b = bond.b
    if not (a.startswith("+") or a.startswith("-")) and a not in res.atoms:
      continue
    if not (b.startswith("+") or b.startswith("-")) and b not in res.atoms:
      continue
    new_bonds.append(bond)
  res.bonds = new_bonds


def _patch_nter_or_cter(res: Residue, params: Dict[str, Dict[str, AtomParam]], topologies: Dict[str, ResidueTopology], terminus: str) -> None:
  """Apply N- or C-terminus patching rules to a residue.

  Args:
    res: Residue to modify.
    params: Parameter table for atoms.
    topologies: Topology templates.
    terminus: Either "NTER" or "CTER".
  """
  if terminus == "NTER":
    if res.name == "GLY":
      _apply_patch(res, "GLYP", params, topologies)
    elif res.name == "PRO":
      _apply_patch(res, "PROP", params, topologies)
    else:
      _apply_patch(res, "NTER", params, topologies)
    delete_prefix = "-"
  elif terminus == "CTER":
    _apply_patch(res, "CTER", params, topologies, delete_o=False)
    delete_prefix = "+"
  else:
    raise ValueError(f"Unknown terminus patch: {terminus}")
  # remove bond to previous or next residue
  new_bonds = []
  removed = False
  for bond in res.bonds:
    if (bond.a.startswith(delete_prefix) or bond.b.startswith(delete_prefix)) and not removed:
      removed = True
      continue
    new_bonds.append(bond)
  res.bonds = new_bonds


def _add_atoms_from_params(res: Residue, params: Dict[str, Dict[str, AtomParam]]) -> None:
  """Populate residue atoms from parameter tables.

  Args:
    res: Residue to populate.
    params: Parameter table for atoms.
  """
  if res.name not in params:
    return
  for atom_name, param in params[res.name].items():
    if atom_name not in res.atoms:
      res.atoms[atom_name] = Atom(name=atom_name, param=param, chain=res.chain, pos=res.pos, res=res)
    else:
      atom = res.atoms[atom_name]
      xyz = atom.xyz.copy()
      valid = atom.is_xyz_valid
      res.atoms[atom_name] = Atom(
        name=atom_name,
        param=param,
        chain=res.chain,
        pos=res.pos,
        res=res,
        xyz=xyz,
        is_xyz_valid=valid,
      )


def _add_bonds_from_topology(res: Residue, topologies: Dict[str, ResidueTopology]) -> None:
  """Populate residue bonds from topology templates.

  Args:
    res: Residue to populate.
    topologies: Topology templates.
  """
  topo = topologies.get(res.name)
  if topo is None:
    return
  res.bonds = list(topo.bonds)


def rebuild_missing_atoms(
  structure: Union[Protein, str, Path],
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
) -> Structure:
  """Rebuild missing heavy atoms and hydrogens using EvoEF2 topology.

  Args:
    structure: Protein object, Structure, or PDB/mmCIF path.
    param_path: Optional parameter file override.
    topo_path: Optional topology file override.

  Returns:
    Structure with missing atoms reconstructed where possible.
  """
  params = load_atom_params(param_path)
  topologies = load_topology(topo_path)
  # Normalize input into a Protein object for consistent parsing.
  protein = structure if isinstance(structure, Protein) else Protein(structure, format="auto")
  df = protein.df
  df = df[df["model"] == protein.models()[0]]

  chains: List[Chain] = []
  for chain_id in sorted(df["chain"].unique()):
    df_chain = df[df["chain"] == chain_id].reset_index(drop=True)
    protein_residues: List[Residue] = []
    ligand_residues: List[Residue] = []
    current_key = None
    current_rows = []
    # Build residues from PDB rows while preserving atom coordinates.
    for _, row in df_chain.iterrows():
      key = (row["res_id"], row["res_name"])
      if current_key is None:
        current_key = key
      if key != current_key:
        res_id, res_name = current_key
        # Normalize histidine names to EvoEF2 protonation variants.
        if res_name == "HIS":
          res_name = "HSD"
        elif res_name == "HIE":
          res_name = "HSE"
        elif res_name == "HIP":
          res_name = "HSP"
        is_protein = res_name in AA_THREE_TO_ONE
        res = Residue(name=res_name, chain=chain_id, pos=int(res_id), is_protein=is_protein)
        # Initialize residue with full atom set and topology bonds.
        _add_atoms_from_params(res, params)
        _add_bonds_from_topology(res, topologies)
        if res_name not in params and current_rows:
          logger.warning("No EvoEF2 parameters for residue %s; skipping parameterization for its atoms.", res_name)
        for r in current_rows:
          atom_name = r["atom_name"]
          if atom_name not in res.atoms:
            if res_name in params and atom_name in params[res_name]:
              res.atoms[atom_name] = Atom(
                name=atom_name,
                param=params[res_name][atom_name],
                chain=chain_id,
                pos=int(res_id),
                res=res,
              )
            else:
              continue
          atom = res.atoms[atom_name]
          atom.xyz = np.array([r["x"], r["y"], r["z"]], dtype=float)
          atom.is_xyz_valid = True
        if is_protein:
          protein_residues.append(res)
        else:
          ligand_residues.append(res)
        current_key = key
        current_rows = []
      current_rows.append(row)
    if current_key is not None:
      res_id, res_name = current_key
      if res_name == "HIS":
        res_name = "HSD"
      elif res_name == "HIE":
        res_name = "HSE"
      elif res_name == "HIP":
        res_name = "HSP"
      is_protein = res_name in AA_THREE_TO_ONE
      res = Residue(name=res_name, chain=chain_id, pos=int(res_id), is_protein=is_protein)
      _add_atoms_from_params(res, params)
      _add_bonds_from_topology(res, topologies)
      if res_name not in params and current_rows:
        logger.warning("No EvoEF2 parameters for residue %s; skipping parameterization for its atoms.", res_name)
      for r in current_rows:
        atom_name = r["atom_name"]
        if atom_name not in res.atoms:
          if res_name in params and atom_name in params[res_name]:
            res.atoms[atom_name] = Atom(
              name=atom_name,
              param=params[res_name][atom_name],
              chain=chain_id,
              pos=int(res_id),
              res=res,
            )
          else:
            continue
        atom = res.atoms[atom_name]
        atom.xyz = np.array([r["x"], r["y"], r["z"]], dtype=float)
        atom.is_xyz_valid = True
      if is_protein:
        protein_residues.append(res)
      else:
        ligand_residues.append(res)

    if protein_residues:
      chain = Chain(name=chain_id, residues=protein_residues, is_protein=True)
      # Apply termini patches before rebuilding missing atoms.
      _patch_nter_or_cter(protein_residues[0], params, topologies, "NTER")
      if protein_residues[0].get_atom("HT1") is not None or protein_residues[0].get_atom("HN1") is not None:
        _patch_nter_or_cter(protein_residues[-1], params, topologies, "CTER")
      # Ensure atoms reference their parent residue (used in H-bond lookups).
      for res in chain.residues:
        for atom in res.atoms.values():
          atom.res = res
      # Rebuild missing heavy atoms and hydrogens from ICs.
      chain_calc_all_atom_xyz(chain, topologies)
      # Cache sidechain torsions for Dunbrack scoring.
      for res in chain.residues:
        residue_calc_sidechain_torsions(res, topologies)
      chains.append(chain)

    if ligand_residues:
      lig_chain = Chain(name=f"{chain_id}_L", residues=ligand_residues, is_protein=False)
      for res in lig_chain.residues:
        for atom in res.atoms.values():
          atom.res = res
      chain_calc_all_atom_xyz(lig_chain, topologies)
      chains.append(lig_chain)

  return Structure(chains=chains)


# -----------------------------
# Energies
# -----------------------------


def energy_term_initialize() -> List[float]:
  """Allocate a zeroed energy term vector.

  Returns:
    List sized to MAX_EVOEF_ENERGY_TERM_NUM.
  """
  return [0.0] * MAX_EVOEF_ENERGY_TERM_NUM


def energy_term_weighting(energy_terms: List[float], weights: List[float]) -> List[float]:
  """Apply weights to energy terms and populate the total term.

  Args:
    energy_terms: Unweighted energy terms.
    weights: Weight vector indexed by EvoEF2 term index.

  Returns:
    Weighted energy terms with total at index 0.
  """
  weighted = energy_terms[:]
  total = 0.0
  for i in range(1, MAX_EVOEF_ENERGY_TERM_NUM):
    weighted[i] *= weights[i]
    total += weighted[i]
  weighted[0] = total
  return weighted


def _vdw_att(atom1: Atom, atom2: Atom, distance: float, bond_type: int) -> float:
  """Attractive part of the VDW potential with EvoEF2 smoothing.

  Args:
    atom1: First atom.
    atom2: Second atom.
    distance: Inter-atomic distance (Angstrom).
    bond_type: Connection category (12/13/14/15).

  Returns:
    VDW attractive energy contribution.
  """
  if distance >= ENERGY_DISTANCE_CUTOFF or bond_type in (12, 13):
    return 0.0
  if atom1.is_h or atom2.is_h:
    return 0.0
  rmin = RADIUS_SCALE_FOR_VDW * (atom1.vdw_radius + atom2.vdw_radius)
  ratio = distance / rmin
  scale = ENERGY_SCALE_FACTOR_BOND_14 if bond_type == 14 else ENERGY_SCALE_FACTOR_BOND_15
  if ratio < 0.8909:
    return 0.0
  if distance <= 5.0:
    epsilon = math.sqrt(atom1.vdw_epsilon * atom2.vdw_epsilon)
    b6 = (1 / ratio) ** 6
    a12 = b6 * b6
    energy = epsilon * (a12 - 2.0 * b6)
  else:
    epsilon = math.sqrt(atom1.vdw_epsilon * atom2.vdw_epsilon)
    b6 = (rmin / 5.0) ** 6
    a12 = b6 * b6
    m = epsilon * (a12 - 2.0 * b6)
    n = 2.4 * epsilon * (b6 - a12)
    a = 2 * m + n
    b = -33 * m - 17 * n
    c = 180 * m + 96 * n
    d = -324 * m - 180 * n
    energy = a * distance**3 + b * distance**2 + c * distance + d
  return energy * scale


def _vdw_rep(atom1: Atom, atom2: Atom, distance: float, bond_type: int) -> float:
  """Repulsive part of the VDW potential with EvoEF2 smoothing.

  Args:
    atom1: First atom.
    atom2: Second atom.
    distance: Inter-atomic distance (Angstrom).
    bond_type: Connection category (12/13/14/15).

  Returns:
    VDW repulsive energy contribution.
  """
  if bond_type in (12, 13):
    return 0.0
  rmin = RADIUS_SCALE_FOR_VDW * (atom1.vdw_radius + atom2.vdw_radius)
  ratio = distance / rmin
  epsilon = math.sqrt(atom1.vdw_epsilon * atom2.vdw_epsilon)
  scale = ENERGY_SCALE_FACTOR_BOND_14 if bond_type == 14 else ENERGY_SCALE_FACTOR_BOND_15
  ratio_cutoff = 0.70
  if ratio > 0.8909:
    energy = 0.0
  elif ratio >= ratio_cutoff:
    b6 = (1 / ratio) ** 6
    a12 = b6 * b6
    energy = epsilon * (a12 - 2.0 * b6)
  else:
    b6_0 = (1 / ratio_cutoff) ** 6
    a = epsilon * (b6_0 * b6_0 - 2.0 * b6_0)
    b = epsilon * 12.0 * (b6_0 / ratio_cutoff - b6_0 * b6_0 / ratio_cutoff)
    y0 = a * epsilon
    k = b * epsilon
    energy = k * (ratio - ratio_cutoff) + y0
  return energy * scale


def _hbond(atom_h: Atom, atom_a: Atom, atom_d: Atom, atom_b: Atom, distance_ha: float, bond_type: int) -> Tuple[float, float, float, float]:
  """Compute hydrogen-bond energy and its geometric components.

  Args:
    atom_h: Hydrogen atom (donor).
    atom_a: Acceptor atom.
    atom_d: Donor heavy atom.
    atom_b: Acceptor base atom.
    distance_ha: H...A distance.
    bond_type: Connection category (12/13/14/15).

  Returns:
    Tuple of (total, distance, theta, phi) energy components.
  """
  if bond_type in (12, 13):
    return 0.0, 0.0, 0.0, 0.0
  if distance_ha > HBOND_DISTANCE_CUTOFF_MAX:
    return 0.0, 0.0, 0.0, 0.0
  xyz_dh = atom_d.xyz - atom_h.xyz
  xyz_ha = atom_h.xyz - atom_a.xyz
  xyz_ab = atom_a.xyz - atom_b.xyz
  angle_theta = PI - xyz_angle(xyz_dh, xyz_ha)
  if rad_to_deg(angle_theta) < 90:
    return 0.0, 0.0, 0.0, 0.0
  angle_phi = PI - xyz_angle(xyz_ha, xyz_ab)
  if rad_to_deg(angle_phi) < 80:
    return 0.0, 0.0, 0.0, 0.0

  if distance_ha < HBOND_OPTIMAL_DISTANCE:
    energy_r = -1.0 * HBOND_WELL_DEPTH * math.cos((distance_ha - HBOND_OPTIMAL_DISTANCE) * PI)
  else:
    energy_r = -0.5 * math.cos(PI / (HBOND_DISTANCE_CUTOFF_MAX - HBOND_OPTIMAL_DISTANCE) * (distance_ha - HBOND_OPTIMAL_DISTANCE)) - 0.5
  if energy_r > 0.0:
    energy_r = 0.0

  energy_theta = -1.0 * (math.cos(angle_theta) ** 4)
  energy_phi = 0.0
  if atom_h.is_bb and atom_a.is_bb:
    energy_phi = -1.0 * (math.cos(angle_phi - deg_to_rad(150)) ** 4)
  else:
    if atom_a.hybrid == "SP3":
      energy_phi = -1.0 * (math.cos(angle_phi - deg_to_rad(135)) ** 4)
    elif atom_a.hybrid == "SP2":
      energy_phi = -1.0 * (math.cos(angle_phi - deg_to_rad(150)) ** 4)

  energy = 0.0
  if rad_to_deg(angle_theta) >= 90 and rad_to_deg(angle_phi) >= 80 and distance_ha < HBOND_DISTANCE_CUTOFF_MAX:
    energy = energy_r + energy_theta + energy_phi
    if energy > 0.0:
      energy = 0.0
  return energy, energy_r, energy_theta, energy_phi


def _electro(atom1: Atom, atom2: Atom, distance: float, bond_type: int) -> float:
  """Coulombic electrostatics with EvoEF2 cutoffs and scaling.

  Args:
    atom1: First atom.
    atom2: Second atom.
    distance: Inter-atomic distance (Angstrom).
    bond_type: Connection category (12/13/14/15).

  Returns:
    Electrostatic energy contribution.
  """
  if bond_type in (12, 13) or distance > ELEC_DISTANCE_CUTOFF:
    return 0.0
  if abs(atom1.charge) < 1e-2 or abs(atom2.charge) < 1e-2:
    return 0.0
  min_dist = 0.8 * (atom1.vdw_radius + atom2.vdw_radius)
  if distance < min_dist:
    distance = min_dist
  energy = COULOMB_CONSTANT * atom1.charge * atom2.charge / distance / distance / 40.0
  scale = ENERGY_SCALE_FACTOR_BOND_14 if bond_type == 14 else ENERGY_SCALE_FACTOR_BOND_15
  return energy * scale


def _lk_desolv(atom1: Atom, atom2: Atom, distance: float, bond_type: int) -> Tuple[float, float]:
  """Lazaridis-Karplus desolvation split into polar/hydrophobic parts.

  Args:
    atom1: First atom.
    atom2: Second atom.
    distance: Inter-atomic distance (Angstrom).
    bond_type: Connection category (12/13/14/15).

  Returns:
    Tuple of (polar, hydrophobic) desolvation energies.
  """
  if bond_type in (12, 13):
    return 0.0, 0.0
  if atom1.is_h or atom2.is_h:
    return 0.0, 0.0
  if distance > ENERGY_DISTANCE_CUTOFF:
    return 0.0, 0.0
  volume1 = atom1.eef1_volume
  volume2 = atom2.eef1_volume
  dg1 = atom1.eef1_free_dg
  dg2 = atom2.eef1_free_dg
  coeff = -0.089793561062582974
  r1 = atom1.vdw_radius * RADIUS_SCALE_FOR_DESOLV
  r2 = atom2.vdw_radius * RADIUS_SCALE_FOR_DESOLV
  r12 = r1 + r2
  distance = max(distance, r12)
  lam1 = atom1.eef1_lambda * distance * distance
  lam2 = atom2.eef1_lambda * distance * distance
  x1 = (distance - r1) / atom1.eef1_lambda
  x2 = (distance - r2) / atom2.eef1_lambda
  desolv12 = coeff * volume2 * dg1 / lam1
  desolv12 *= math.exp(-1.0 * x1 * x1)
  desolv21 = coeff * volume1 * dg2 / lam2
  desolv21 *= math.exp(-1.0 * x2 * x2)

  energy_p = 0.0
  energy_h = 0.0
  if atom1.polarity in {"P", "C"}:
    energy_p += desolv12
  else:
    energy_h += desolv12
  if atom2.polarity in {"P", "C"}:
    energy_p += desolv21
  else:
    energy_h += desolv21
  return energy_p, energy_h


def _cell_index(xyz: np.ndarray, cell_size: float) -> Tuple[int, int, int]:
  return (
    int(math.floor(xyz[0] / cell_size)),
    int(math.floor(xyz[1] / cell_size)),
    int(math.floor(xyz[2] / cell_size)),
  )


def _build_cell_list(atoms: Iterable[Atom], cell_size: float) -> Dict[Tuple[int, int, int], List[Atom]]:
  grid: Dict[Tuple[int, int, int], List[Atom]] = {}
  for atom in atoms:
    if not atom.is_xyz_valid:
      continue
    key = _cell_index(atom.xyz, cell_size)
    grid.setdefault(key, []).append(atom)
  return grid


def _iter_neighbor_atoms(atom: Atom, grid: Dict[Tuple[int, int, int], List[Atom]], cell_size: float) -> Iterable[Atom]:
  key = _cell_index(atom.xyz, cell_size)
  for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
      for dz in (-1, 0, 1):
        cell = (key[0] + dx, key[1] + dy, key[2] + dz)
        for other in grid.get(cell, []):
          yield other


def _collect_atoms(chain: Chain, *, protein_only: Optional[bool] = None) -> List[Atom]:
  atoms: List[Atom] = []
  for res in chain.residues:
    if protein_only is not None and res.is_protein != protein_only:
      continue
    for atom in res.atoms.values():
      if atom.is_xyz_valid:
        atoms.append(atom)
  return atoms


def _inter_chain_energy(chain_a: Chain, chain_b: Chain, terms: List[float]) -> None:
  """Compute inter-chain protein-protein energies using a spatial grid."""
  atoms_a = _collect_atoms(chain_a, protein_only=True)
  atoms_b = _collect_atoms(chain_b, protein_only=True)
  if not atoms_a or not atoms_b:
    return
  cell_size = ENERGY_DISTANCE_CUTOFF
  grid_b = _build_cell_list(atoms_b, cell_size)
  for a1 in atoms_a:
    for a2 in _iter_neighbor_atoms(a1, grid_b, cell_size):
      dist = xyz_distance(a1.xyz, a2.xyz)
      if dist > ENERGY_DISTANCE_CUTOFF:
        continue
      bond_type = 15
      terms[51] += _vdw_att(a1, a2, dist, bond_type)
      terms[52] += _vdw_rep(a1, a2, dist, bond_type)
      terms[53] += _electro(a1, a2, dist, bond_type)
      des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
      terms[54] += des_p
      terms[55] += des_h
      if dist < HBOND_DISTANCE_CUTOFF_MAX:
        hbd = hbt = hbp = 0.0
        if a1.is_hbond_h and a2.is_hbond_a and a1.res and a2.res:
          atom_d = a1.res.get_atom(a1.hb_d_or_b)
          atom_b = a2.res.get_atom(a2.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
        elif a2.is_hbond_h and a1.is_hbond_a and a1.res and a2.res:
          atom_d = a2.res.get_atom(a2.hb_d_or_b)
          atom_b = a1.res.get_atom(a1.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
        if a1.is_bb and a2.is_bb:
          terms[61] += hbd
          terms[62] += hbt
          terms[63] += hbp
        elif not a1.is_bb and not a2.is_bb:
          terms[67] += hbd
          terms[68] += hbt
          terms[69] += hbp
        else:
          terms[64] += hbd
          terms[65] += hbt
          terms[66] += hbp
      if a1.name == "SG" and a2.name == "SG" and a1.res and a2.res:
        if a1.res.name == "CYS" and a2.res.name == "CYS":
          if SSBOND_CUTOFF_MIN < dist < SSBOND_CUTOFF_MAX:
            cb1 = a1.res.get_atom("CB")
            cb2 = a2.res.get_atom("CB")
            ca1 = a1.res.get_atom("CA")
            ca2 = a2.res.get_atom("CA")
            if cb1 and cb2 and ca1 and ca2:
              terms[56] += _ssbond(a1, a2, cb1, cb2, ca1, ca2)


def _protein_ligand_energy(protein_chain: Chain, ligand_chain: Chain, terms: List[float]) -> None:
  """Compute protein-ligand energies using a spatial grid."""
  atoms_p = _collect_atoms(protein_chain, protein_only=True)
  atoms_l = _collect_atoms(ligand_chain, protein_only=False)
  if not atoms_p or not atoms_l:
    return
  cell_size = ENERGY_DISTANCE_CUTOFF
  grid_l = _build_cell_list(atoms_l, cell_size)
  for a1 in atoms_p:
    for a2 in _iter_neighbor_atoms(a1, grid_l, cell_size):
      dist = xyz_distance(a1.xyz, a2.xyz)
      if dist > ENERGY_DISTANCE_CUTOFF:
        continue
      bond_type = 15
      terms[71] += _vdw_att(a1, a2, dist, bond_type)
      terms[72] += _vdw_rep(a1, a2, dist, bond_type)
      terms[73] += _electro(a1, a2, dist, bond_type)
      des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
      terms[74] += des_p
      terms[75] += des_h
      if dist < HBOND_DISTANCE_CUTOFF_MAX:
        hbd = hbt = hbp = 0.0
        if a1.is_hbond_h and a2.is_hbond_a and a1.res and a2.res:
          atom_d = a1.res.get_atom(a1.hb_d_or_b)
          atom_b = a2.res.get_atom(a2.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
        elif a2.is_hbond_h and a1.is_hbond_a and a1.res and a2.res:
          atom_d = a2.res.get_atom(a2.hb_d_or_b)
          atom_b = a1.res.get_atom(a1.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
        if not a1.is_bb and not a2.is_bb:
          terms[84] += hbd
          terms[85] += hbt
          terms[86] += hbp
        else:
          terms[81] += hbd
          terms[82] += hbt
          terms[83] += hbp


def _ssbond(atom_s1: Atom, atom_s2: Atom, atom_cb1: Atom, atom_cb2: Atom, atom_ca1: Atom, atom_ca2: Atom) -> float:
  """Compute disulfide bond energy using EvoEF2 geometry terms.

  Args:
    atom_s1: SG atom from residue 1.
    atom_s2: SG atom from residue 2.
    atom_cb1: CB atom from residue 1.
    atom_cb2: CB atom from residue 2.
    atom_ca1: CA atom from residue 1.
    atom_ca2: CA atom from residue 2.

  Returns:
    Disulfide energy contribution (non-positive).
  """
  dss = xyz_distance(atom_s1.xyz, atom_s2.xyz)
  a_c1s1s2 = rad_to_deg(PI - xyz_angle(atom_s1.xyz - atom_s2.xyz, atom_s2.xyz - atom_cb2.xyz))
  a_c2s2s1 = rad_to_deg(PI - xyz_angle(atom_cb1.xyz - atom_s1.xyz, atom_s1.xyz - atom_s2.xyz))
  x_c1s1s2c2 = get_torsion_angle(atom_cb1.xyz, atom_s1.xyz, atom_s2.xyz, atom_cb2.xyz)
  x_ca1cb1sg1sg2 = get_torsion_angle(atom_ca1.xyz, atom_cb1.xyz, atom_s1.xyz, atom_s2.xyz)
  x_ca2cb2sg2sg1 = get_torsion_angle(atom_ca2.xyz, atom_cb2.xyz, atom_s2.xyz, atom_s1.xyz)
  sse = (
    0.8 * (1 - math.exp(-10.0 * (dss - SSBOND_DISTANCE))) ** 2
    + 0.005 * (a_c1s1s2 - SSBOND_ANGLE) ** 2
    + 0.005 * (a_c2s2s1 - SSBOND_ANGLE) ** 2
    + math.cos(2.0 * x_c1s1s2c2)
    + 1.0
    + 1.25 * math.sin(x_ca1cb1sg1sg2 + 2.0 * PI / 3.0)
    - 1.75
    + 1.25 * math.sin(x_ca2cb2sg2sg1 + 2.0 * PI / 3.0)
    - 1.75
  )
  return min(0.0, sse)


def _aa_reference_energy(res: Residue, terms: List[float]) -> None:
  """Accumulate amino-acid reference counts for a residue.

  Args:
    res: Residue to score.
    terms: Energy term accumulator.
  """
  name = res.name
  if name == "ALA":
    terms[1] += 1.0
  elif name == "CYS":
    terms[2] += 1.0
  elif name == "ASP":
    terms[3] += 1.0
  elif name == "GLU":
    terms[4] += 1.0
  elif name == "PHE":
    terms[5] += 1.0
  elif name == "GLY":
    terms[6] += 1.0
  elif name in {"HIS", "HSE", "HSD", "HSP"}:
    terms[7] += 1.0
  elif name == "ILE":
    terms[8] += 1.0
  elif name == "LYS":
    terms[9] += 1.0
  elif name == "LEU":
    terms[10] += 1.0
  elif name == "MET":
    terms[11] += 1.0
  elif name == "ASN":
    terms[12] += 1.0
  elif name == "PRO":
    terms[13] += 1.0
  elif name == "GLN":
    terms[14] += 1.0
  elif name == "ARG":
    terms[15] += 1.0
  elif name == "SER":
    terms[16] += 1.0
  elif name == "THR":
    terms[17] += 1.0
  elif name == "VAL":
    terms[18] += 1.0
  elif name == "TRP":
    terms[19] += 1.0
  elif name == "TYR":
    terms[20] += 1.0


def _aa_propensity_ramachandran(res: Residue, aap: AAppTable, rama: RamaTable, terms: List[float]) -> None:
  """Accumulate AA propensity and Ramachandran energies for a residue.

  Args:
    res: Residue with phi/psi assigned.
    aap: Amino-acid propensity table.
    rama: Ramachandran table.
    terms: Energy term accumulator.
  """
  aa1 = AA_THREE_TO_ONE.get(res.name)
  if aa1 is None:
    return
  aa_index = AA_ONE_LETTER.index(aa1)
  phi = int(res.phipsi[0])
  psi = int(res.phipsi[1])
  phi_index = (phi + 180) // 10
  psi_index = (psi + 180) // 10
  phi_index = max(0, min(35, phi_index))
  psi_index = max(0, min(35, psi_index))
  terms[91] += aap.aap[phi_index, psi_index, aa_index]
  terms[92] += rama.rama[phi_index, psi_index, aa_index]
  terms[93] += 0.0


def _aa_dunbrack(res: Residue, dun: DunbrackLibrary, terms: List[float]) -> None:
  """Accumulate Dunbrack rotamer energy for a residue.

  Args:
    res: Residue with sidechain torsions computed.
    dun: Dunbrack library.
    terms: Energy term accumulator.
  """
  if res.name in {"ALA", "GLY"}:
    terms[93] += 0.0
    return
  phi = int(res.phipsi[0])
  psi = int(res.phipsi[1])
  bin_index = ((phi + 180) // 10) * 36 + ((psi + 180) // 10)
  if bin_index < 0 or bin_index >= len(dun.bins):
    return
  rotamers = dun.bins[bin_index].by_residue.get(res.name)
  if not rotamers:
    return
  match_index = -1
  delta_prob = 1e-7
  for i, rot in enumerate(rotamers):
    match = True
    for j, mean in enumerate(rot.torsions):
      min_v = mean - deg_to_rad(30)
      max_v = mean + deg_to_rad(30)
      torsion = res.xtorsions[j] if j < len(res.xtorsions) else 0.0
      torsion_m2pi = torsion - 2 * PI
      torsion_p2pi = torsion + 2 * PI
      torsion2 = torsion
      if (res.name in {"PHE", "TYR", "ASP"} and j == 1) or (res.name == "GLU" and j == 2):
        torsion2 = torsion + PI
        torsion2 = torsion - PI if torsion > 0 else torsion2
      torsion2_m2pi = torsion2 - 2 * PI
      torsion2_p2pi = torsion2 + 2 * PI
      if not (
        (min_v <= torsion <= max_v)
        or (min_v <= torsion_m2pi <= max_v)
        or (min_v <= torsion_p2pi <= max_v)
        or (min_v <= torsion2 <= max_v)
        or (min_v <= torsion2_m2pi <= max_v)
        or (min_v <= torsion2_p2pi <= max_v)
      ):
        match = False
        break
    if match:
      match_index = i
      break
  if match_index != -1:
    prob = rotamers[match_index].probability
  else:
    prob = rotamers[-1].probability
  terms[93] += -1.0 * math.log(prob + delta_prob)


def _calc_phi_psi(chain: Chain) -> None:
  """Compute phi/psi angles for each residue in a chain.

  Args:
    chain: Chain to annotate with torsion angles.
  """
  for i, res in enumerate(chain.residues):
    phi = -60.0
    psi = 60.0
    n = res.get_atom("N")
    ca = res.get_atom("CA")
    c = res.get_atom("C")
    if n is None or ca is None or c is None:
      res.phipsi = (phi, psi)
      continue
    if i > 0:
      prev_c = chain.residues[i - 1].get_atom("C")
      if prev_c is not None:
        dist_c0_n1 = xyz_distance(prev_c.xyz, n.xyz)
        if 1.25 < dist_c0_n1 < 1.45:
          phi = rad_to_deg(get_torsion_angle(prev_c.xyz, n.xyz, ca.xyz, c.xyz))
    if i < len(chain.residues) - 1:
      next_n = chain.residues[i + 1].get_atom("N")
      if next_n is not None:
        dist_c1_n2 = xyz_distance(c.xyz, next_n.xyz)
        if 1.25 < dist_c1_n2 < 1.45:
          psi = rad_to_deg(get_torsion_angle(n.xyz, ca.xyz, c.xyz, next_n.xyz))
    res.phipsi = (phi, psi)


def _residue_intra_energy(res: Residue, terms: List[float]) -> None:
  """Compute intra-residue nonbonded and H-bond terms.

  Args:
    res: Residue to score.
    terms: Energy term accumulator.
  """
  atoms = list(res.atoms.values())
  for i, a1 in enumerate(atoms):
    if not a1.is_xyz_valid:
      continue
    for j in range(i + 1, len(atoms)):
      a2 = atoms[j]
      if not a2.is_xyz_valid:
        continue
      dist = xyz_distance(a1.xyz, a2.xyz)
      if dist > ENERGY_DISTANCE_CUTOFF:
        continue
      if a1.is_bb and a2.is_bb:
        continue
      if not a1.is_bb and not a2.is_bb:
        if res.name in {"ILE", "MET", "GLN", "GLU", "LYS", "ARG"}:
          bond_type = _residue_intra_bond_connection(a1.name, a2.name, res.bonds)
          if bond_type in (12, 13):
            continue
          terms[21] += _vdw_att(a1, a2, dist, bond_type)
          terms[22] += _vdw_rep(a1, a2, dist, bond_type)
          des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
          terms[24] += des_p
          terms[25] += des_h
      else:
        if a1.name == "CB" or a2.name == "CB":
          continue
        bond_type = _residue_intra_bond_connection(a1.name, a2.name, res.bonds)
        if bond_type in (12, 13):
          continue
        terms[21] += _vdw_att(a1, a2, dist, bond_type)
        terms[22] += _vdw_rep(a1, a2, dist, bond_type)
        terms[23] += _electro(a1, a2, dist, bond_type)
        des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
        terms[24] += des_p
        terms[25] += des_h
        if dist < HBOND_DISTANCE_CUTOFF_MAX:
          hb_tot = hb_dist = hb_theta = hb_phi = 0.0
          if a1.is_hbond_h and a2.is_hbond_a:
            atom_d = res.get_atom(a1.hb_d_or_b)
            atom_b = res.get_atom(a2.hb_d_or_b)
            if atom_d and atom_b:
              hb_tot, hb_dist, hb_theta, hb_phi = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
          elif a2.is_hbond_h and a1.is_hbond_a:
            atom_d = res.get_atom(a2.hb_d_or_b)
            atom_b = res.get_atom(a1.hb_d_or_b)
            if atom_d and atom_b:
              hb_tot, hb_dist, hb_theta, hb_phi = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
          terms[26] += hb_dist
          terms[27] += hb_theta
          terms[28] += hb_phi


def _residue_and_next_energy(res: Residue, other: Residue, terms: List[float]) -> None:
  """Compute energy between sequential residues in a chain.

  Args:
    res: Current residue.
    other: Next residue in the chain.
    terms: Energy term accumulator.
  """
  for a1 in res.atoms.values():
    if not a1.is_xyz_valid:
      continue
    for a2 in other.atoms.values():
      if not a2.is_xyz_valid:
        continue
      dist = xyz_distance(a1.xyz, a2.xyz)
      if dist > ENERGY_DISTANCE_CUTOFF:
        continue
      if a1.is_bb and a2.is_bb:
        continue
      if not a1.is_bb and not a2.is_bb:
        bond_type = 15
        terms[31] += _vdw_att(a1, a2, dist, bond_type)
        terms[32] += _vdw_rep(a1, a2, dist, bond_type)
        terms[33] += _electro(a1, a2, dist, bond_type)
        des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
        terms[34] += des_p
        terms[35] += des_h
        if dist < HBOND_DISTANCE_CUTOFF_MAX:
          hbd = hbt = hbp = 0.0
          if a1.is_hbond_h and a2.is_hbond_a:
            atom_d = res.get_atom(a1.hb_d_or_b)
            atom_b = other.get_atom(a2.hb_d_or_b)
            if atom_d and atom_b:
              _, hbd, hbt, hbp = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
          elif a2.is_hbond_h and a1.is_hbond_a:
            atom_d = other.get_atom(a2.hb_d_or_b)
            atom_b = res.get_atom(a1.hb_d_or_b)
            if atom_d and atom_b:
              _, hbd, hbt, hbp = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
          if abs(res.pos - other.pos) <= 2:
            hbd *= HBOND_LOCAL_REDUCE
            hbt *= HBOND_LOCAL_REDUCE
            hbp *= HBOND_LOCAL_REDUCE
          if a1.is_bb and a2.is_bb:
            terms[41] += hbd
            terms[42] += hbt
            terms[43] += hbp
          elif not a1.is_bb and not a2.is_bb:
            terms[47] += hbd
            terms[48] += hbt
            terms[49] += hbp
          else:
            terms[44] += hbd
            terms[45] += hbt
            terms[46] += hbp
      else:
        bond_type = _residue_and_next_residue_bond_type(a1.name, a2.name, other.name)
        if bond_type in (12, 13):
          continue
        terms[31] += _vdw_att(a1, a2, dist, bond_type)
        terms[32] += _vdw_rep(a1, a2, dist, bond_type)
        terms[33] += _electro(a1, a2, dist, bond_type)
        des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
        terms[34] += des_p
        terms[35] += des_h
        if dist < HBOND_DISTANCE_CUTOFF_MAX:
          hbd = hbt = hbp = 0.0
          if a1.is_hbond_h and a2.is_hbond_a:
            atom_d = res.get_atom(a1.hb_d_or_b)
            atom_b = other.get_atom(a2.hb_d_or_b)
            if atom_d and atom_b:
              _, hbd, hbt, hbp = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
          elif a2.is_hbond_h and a1.is_hbond_a:
            atom_d = other.get_atom(a2.hb_d_or_b)
            atom_b = res.get_atom(a1.hb_d_or_b)
            if atom_d and atom_b:
              _, hbd, hbt, hbp = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
          if a1.is_bb and a2.is_bb:
            terms[41] += hbd
            terms[42] += hbt
            terms[43] += hbp
          elif not a1.is_bb and not a2.is_bb:
            terms[47] += hbd
            terms[48] += hbt
            terms[49] += hbp
          else:
            terms[44] += hbd
            terms[45] += hbt
            terms[46] += hbp


def _residue_other_same_chain(res: Residue, other: Residue, terms: List[float]) -> None:
  """Compute energy between non-adjacent residues in the same chain.

  Args:
    res: First residue.
    other: Second residue (non-adjacent).
    terms: Energy term accumulator.
  """
  for a1 in res.atoms.values():
    if not a1.is_xyz_valid:
      continue
    for a2 in other.atoms.values():
      if not a2.is_xyz_valid:
        continue
      dist = xyz_distance(a1.xyz, a2.xyz)
      if dist > ENERGY_DISTANCE_CUTOFF:
        continue
      bond_type = 15
      terms[31] += _vdw_att(a1, a2, dist, bond_type)
      terms[32] += _vdw_rep(a1, a2, dist, bond_type)
      terms[33] += _electro(a1, a2, dist, bond_type)
      des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
      terms[34] += des_p
      terms[35] += des_h
      if dist < HBOND_DISTANCE_CUTOFF_MAX:
        hbd = hbt = hbp = 0.0
        if a1.is_hbond_h and a2.is_hbond_a:
          atom_d = res.get_atom(a1.hb_d_or_b)
          atom_b = other.get_atom(a2.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
        elif a2.is_hbond_h and a1.is_hbond_a:
          atom_d = other.get_atom(a2.hb_d_or_b)
          atom_b = res.get_atom(a1.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
        if abs(res.pos - other.pos) <= 2:
          hbd *= HBOND_LOCAL_REDUCE
          hbt *= HBOND_LOCAL_REDUCE
          hbp *= HBOND_LOCAL_REDUCE
        if a1.is_bb and a2.is_bb:
          terms[41] += hbd
          terms[42] += hbt
          terms[43] += hbp
        elif not a1.is_bb and not a2.is_bb:
          terms[47] += hbd
          terms[48] += hbt
          terms[49] += hbp
        else:
          terms[44] += hbd
          terms[45] += hbt
          terms[46] += hbp

  if res.name == "CYS" and other.name == "CYS":
    sg1 = res.get_atom("SG")
    sg2 = other.get_atom("SG")
    cb1 = res.get_atom("CB")
    cb2 = other.get_atom("CB")
    ca1 = res.get_atom("CA")
    ca2 = other.get_atom("CA")
    if sg1 and sg2 and cb1 and cb2 and ca1 and ca2:
      dist = xyz_distance(sg1.xyz, sg2.xyz)
      if SSBOND_CUTOFF_MIN < dist < SSBOND_CUTOFF_MAX:
        terms[36] += _ssbond(sg1, sg2, cb1, cb2, ca1, ca2)


def _residue_other_diff_chain(res: Residue, other: Residue, terms: List[float]) -> None:
  """Compute inter-chain residue-residue energy terms.

  Args:
    res: Residue in chain A.
    other: Residue in chain B.
    terms: Energy term accumulator.
  """
  for a1 in res.atoms.values():
    if not a1.is_xyz_valid:
      continue
    for a2 in other.atoms.values():
      if not a2.is_xyz_valid:
        continue
      dist = xyz_distance(a1.xyz, a2.xyz)
      if dist > ENERGY_DISTANCE_CUTOFF:
        continue
      bond_type = 15
      terms[51] += _vdw_att(a1, a2, dist, bond_type)
      terms[52] += _vdw_rep(a1, a2, dist, bond_type)
      terms[53] += _electro(a1, a2, dist, bond_type)
      des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
      terms[54] += des_p
      terms[55] += des_h
      if dist < HBOND_DISTANCE_CUTOFF_MAX:
        hbd = hbt = hbp = 0.0
        if a1.is_hbond_h and a2.is_hbond_a:
          atom_d = res.get_atom(a1.hb_d_or_b)
          atom_b = other.get_atom(a2.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
        elif a2.is_hbond_h and a1.is_hbond_a:
          atom_d = other.get_atom(a2.hb_d_or_b)
          atom_b = res.get_atom(a1.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
        if a1.is_bb and a2.is_bb:
          terms[61] += hbd
          terms[62] += hbt
          terms[63] += hbp
        elif not a1.is_bb and not a2.is_bb:
          terms[67] += hbd
          terms[68] += hbt
          terms[69] += hbp
        else:
          terms[64] += hbd
          terms[65] += hbt
          terms[66] += hbp

  if res.name == "CYS" and other.name == "CYS":
    sg1 = res.get_atom("SG")
    sg2 = other.get_atom("SG")
    cb1 = res.get_atom("CB")
    cb2 = other.get_atom("CB")
    ca1 = res.get_atom("CA")
    ca2 = other.get_atom("CA")
    if sg1 and sg2 and cb1 and cb2 and ca1 and ca2:
      dist = xyz_distance(sg1.xyz, sg2.xyz)
      if SSBOND_CUTOFF_MIN < dist < SSBOND_CUTOFF_MAX:
        terms[56] += _ssbond(sg1, sg2, cb1, cb2, ca1, ca2)


def _residue_and_ligand_energy(res: Residue, ligand: Residue, terms: List[float]) -> None:
  """Compute protein-ligand energy terms for a residue/ligand pair.

  Args:
    res: Protein residue.
    ligand: Ligand residue.
    terms: Energy term accumulator.
  """
  for a1 in res.atoms.values():
    if not a1.is_xyz_valid:
      continue
    for a2 in ligand.atoms.values():
      if not a2.is_xyz_valid:
        continue
      dist = xyz_distance(a1.xyz, a2.xyz)
      if dist > ENERGY_DISTANCE_CUTOFF:
        continue
      bond_type = 15
      terms[71] += _vdw_att(a1, a2, dist, bond_type)
      terms[72] += _vdw_rep(a1, a2, dist, bond_type)
      terms[73] += _electro(a1, a2, dist, bond_type)
      des_p, des_h = _lk_desolv(a1, a2, dist, bond_type)
      terms[74] += des_p
      terms[75] += des_h
      if dist < HBOND_DISTANCE_CUTOFF_MAX:
        hbd = hbt = hbp = 0.0
        if a1.is_hbond_h and a2.is_hbond_a:
          atom_d = res.get_atom(a1.hb_d_or_b)
          atom_b = ligand.get_atom(a2.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a1, a2, atom_d, atom_b, dist, bond_type)
        elif a2.is_hbond_h and a1.is_hbond_a:
          atom_d = ligand.get_atom(a2.hb_d_or_b)
          atom_b = res.get_atom(a1.hb_d_or_b)
          if atom_d and atom_b:
            _, hbd, hbt, hbp = _hbond(a2, a1, atom_d, atom_b, dist, bond_type)
        if not a1.is_bb and not a2.is_bb:
          terms[84] += hbd
          terms[85] += hbt
          terms[86] += hbp
        else:
          terms[81] += hbd
          terms[82] += hbt
          terms[83] += hbp


# -----------------------------
# Public API
# -----------------------------


def calculate_stability(
  structure: Union[Protein, str, Path],
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
  weight_dict: Optional[Dict[str, float]] = None,
  aapropensity_path: Optional[Path] = None,
  ramachandran_path: Optional[Path] = None,
  dunbrack_path: Optional[Path] = None,
) -> Dict[str, float]:
  """Compute EvoEF2 stability energy for a structure.

  Args:
    structure: Protein object or PDB/mmCIF path.
    param_path: Optional parameter file override.
    topo_path: Optional topology file override.
    weight_dict: Weights dictionary to use.
    aapropensity_path: Optional AA propensity table override.
    ramachandran_path: Optional Ramachandran table override.
    dunbrack_path: Optional Dunbrack library override.

  Returns:
    Dict of all weighted energy terms plus the total.
  """
  weights = get_weights(weight_dict)
  aap = load_aapropensity(aapropensity_path)
  rama = load_ramachandran(ramachandran_path)
  dun = load_dunbrack(dunbrack_path)

  evo_struct = rebuild_missing_atoms(structure, param_path=param_path, topo_path=topo_path)
  for chain in evo_struct.chains:
    if chain.is_protein:
      # Phi/psi angles are required for Ramachandran and Dunbrack terms.
      _calc_phi_psi(chain)

  terms = energy_term_initialize()
  # Compute stability across the whole structure.
  for chain in evo_struct.chains:
    for i, res in enumerate(chain.residues):
      if not res.is_protein:
        continue
      # Per-residue reference and internal terms.
      _aa_reference_energy(res, terms)
      _residue_intra_energy(res, terms)
      _aa_propensity_ramachandran(res, aap, rama, terms)
      _aa_dunbrack(res, dun, terms)
      # same-chain pairs
      for j in range(i + 1, len(chain.residues)):
        other = chain.residues[j]
        if j == i + 1:
          # Adjacent residues use special 1-4 scaling rules.
          _residue_and_next_energy(res, other, terms)
        else:
          _residue_other_same_chain(res, other, terms)
  # different chains (avoid double counting by index)
  for i, chain_i in enumerate(evo_struct.chains):
    for k in range(i + 1, len(evo_struct.chains)):
      chain_k = evo_struct.chains[k]
      if chain_i.is_protein and chain_k.is_protein:
        _inter_chain_energy(chain_i, chain_k, terms)
      elif chain_i.is_protein and not chain_k.is_protein:
        _protein_ligand_energy(chain_i, chain_k, terms)
      elif not chain_i.is_protein and chain_k.is_protein:
        _protein_ligand_energy(chain_k, chain_i, terms)

  weighted = energy_term_weighting(terms, weights)
  return _energy_terms_to_dict(weighted)


def calculate_interface_energy(
  structure: Union[Protein, str, Path],
  split1: Sequence[str],
  split2: Sequence[str],
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
  weight_dict: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
  """Compute interface energy between two chain groups.

  Args:
    structure: Protein object or PDB/mmCIF path.
    split1: Chain IDs for group 1.
    split2: Chain IDs for group 2.
    param_path: Optional parameter file override.
    topo_path: Optional topology file override.
    weight_dict: Weights dictionary to use.

  Returns:
    Dict of weighted inter-chain energy terms plus the total.
  """
  weights = get_weights(weight_dict)
  evo_struct = rebuild_missing_atoms(structure, param_path=param_path, topo_path=topo_path)
  terms = energy_term_initialize()
  set1 = set(split1)
  set2 = set(split2)
  for i, chain_i in enumerate(evo_struct.chains):
    for k in range(i + 1, len(evo_struct.chains)):
      chain_k = evo_struct.chains[k]
      if not ((chain_i.name in set1 and chain_k.name in set2) or (chain_i.name in set2 and chain_k.name in set1)):
        continue
      if chain_i.is_protein and chain_k.is_protein:
        _inter_chain_energy(chain_i, chain_k, terms)
      elif chain_i.is_protein and not chain_k.is_protein:
        _protein_ligand_energy(chain_i, chain_k, terms)
      elif not chain_i.is_protein and chain_k.is_protein:
        _protein_ligand_energy(chain_k, chain_i, terms)
  weighted = energy_term_weighting(terms, weights)
  return _energy_terms_to_dict(weighted)


def calculate_binding(
  structure: Union[Protein, str, Path],
  split1: Sequence[str],
  split2: Sequence[str],
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
  weight_path: Optional[Path] = None,
  aapropensity_path: Optional[Path] = None,
  ramachandran_path: Optional[Path] = None,
  dunbrack_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
  """Compute interface energy and DG_bind for two chain groups.

  Args:
    structure: Protein object or PDB/mmCIF path.
    split1: Chain IDs for group 1.
    split2: Chain IDs for group 2.
    param_path: Optional parameter file override.
    topo_path: Optional topology file override.
    weight_dict: Weights dictionary to use.
    aapropensity_path: Optional AA propensity table override.
    ramachandran_path: Optional Ramachandran table override.
    dunbrack_path: Optional Dunbrack library override.

  Returns:
    Dict with interface energy, complex stability, split stabilities, and DG_bind.
  """
  # returns interface energy and DG_bind by stability difference
  interface = calculate_interface_energy(
    structure,
    split1,
    split2,
    param_path=param_path,
    topo_path=topo_path,
    weight_path=weight_path,
  )
  # compute stability of complex and each split independently
  full = calculate_stability(
    structure,
    param_path=param_path,
    topo_path=topo_path,
    weight_path=weight_path,
    aapropensity_path=aapropensity_path,
    ramachandran_path=ramachandran_path,
    dunbrack_path=dunbrack_path,
  )
  # compute stability by filtering chains in a single rebuilt structure
  evo_struct = rebuild_missing_atoms(structure, param_path=param_path, topo_path=topo_path)
  split1_struct = Structure(chains=[c for c in evo_struct.chains if c.name in split1])
  split2_struct = Structure(chains=[c for c in evo_struct.chains if c.name in split2])
  split1_energy = _calculate_stability_from_structure(
    split1_struct,
    weight_path=weight_path,
    aapropensity_path=aapropensity_path,
    ramachandran_path=ramachandran_path,
    dunbrack_path=dunbrack_path,
  )
  split2_energy = _calculate_stability_from_structure(
    split2_struct,
    weight_path=weight_path,
    aapropensity_path=aapropensity_path,
    ramachandran_path=ramachandran_path,
    dunbrack_path=dunbrack_path,
  )
  dg_bind = _subtract_energy_dicts(full, split1_energy, split2_energy)
  return {
    "interface": interface,
    "stability_complex": full,
    "stability_split1": split1_energy,
    "stability_split2": split2_energy,
    "dg_bind": dg_bind,
  }


def _calculate_stability_from_structure(
  evo_struct: Structure,
  *,
  weight_dict: Optional[Dict[str, float]] = None,
  aapropensity_path: Optional[Path] = None,
  ramachandran_path: Optional[Path] = None,
  dunbrack_path: Optional[Path] = None,
) -> Dict[str, float]:
  """Compute stability energy from a pre-built Structure.

  Args:
    evo_struct: Structure with atoms already reconstructed.
    weight_dict: Weights dictionary to use.
    aapropensity_path: Optional AA propensity table override.
    ramachandran_path: Optional Ramachandran table override.
    dunbrack_path: Optional Dunbrack library override.

  Returns:
    Dict of weighted energy terms plus the total.
  """
  weights = get_weights(weight_dict)
  aap = load_aapropensity(aapropensity_path)
  rama = load_ramachandran(ramachandran_path)
  dun = load_dunbrack(dunbrack_path)
  for chain in evo_struct.chains:
    if chain.is_protein:
      _calc_phi_psi(chain)
  terms = energy_term_initialize()
  for chain in evo_struct.chains:
    for i, res in enumerate(chain.residues):
      if not res.is_protein:
        continue
      _aa_reference_energy(res, terms)
      _residue_intra_energy(res, terms)
      _aa_propensity_ramachandran(res, aap, rama, terms)
      _aa_dunbrack(res, dun, terms)
      for j in range(i + 1, len(chain.residues)):
        other = chain.residues[j]
        if j == i + 1:
          _residue_and_next_energy(res, other, terms)
        else:
          _residue_other_same_chain(res, other, terms)
  for i, chain_i in enumerate(evo_struct.chains):
    for k in range(i + 1, len(evo_struct.chains)):
      chain_k = evo_struct.chains[k]
      if chain_i.is_protein and chain_k.is_protein:
        _inter_chain_energy(chain_i, chain_k, terms)
      elif chain_i.is_protein and not chain_k.is_protein:
        _protein_ligand_energy(chain_i, chain_k, terms)
      elif not chain_i.is_protein and chain_k.is_protein:
        _protein_ligand_energy(chain_k, chain_i, terms)
  weighted = energy_term_weighting(terms, weights)
  return _energy_terms_to_dict(weighted)


def _energy_terms_to_dict(energy_terms: List[float]) -> Dict[str, float]:
  """Convert the energy term vector to a stable, named dict.

  Args:
    energy_terms: Vector indexed by EvoEF2 term indices.

  Returns:
    Dict ordered by EvoEF2 term names, including total.
  """
  result: Dict[str, float] = {}
  result[ENERGY_TERM_NAMES[0]] = energy_terms[0]
  for idx in ENERGY_TERM_ORDER:
    result[ENERGY_TERM_NAMES[idx]] = energy_terms[idx]
  return result


def _subtract_energy_dicts(full: Dict[str, float], a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
  """Compute DG_bind-style subtraction: full - a - b.

  Args:
    full: Energy dict for the complex.
    a: Energy dict for chain group 1.
    b: Energy dict for chain group 2.

  Returns:
    Dict with per-term subtraction results.
  """
  result: Dict[str, float] = {}
  for key in full.keys():
    result[key] = full.get(key, 0.0) - a.get(key, 0.0) - b.get(key, 0.0)
  return result


def debug_evoef2_structure(
  structure: Union[Protein, str, Path],
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
  dunbrack_path: Optional[Path] = None,
) -> Dict[str, float]:
  """Collect reconstruction and torsion diagnostics for a structure.

  Args:
    structure: Protein object or PDB/mmCIF path.
    param_path: Optional parameter file override.
    topo_path: Optional topology file override.
    dunbrack_path: Optional Dunbrack library override.

  Returns:
    Dict with counts for missing atoms, torsions, and Dunbrack coverage.
  """
  topologies = load_topology(topo_path)
  dun = load_dunbrack(dunbrack_path)
  evo_struct = rebuild_missing_atoms(structure, param_path=param_path, topo_path=topo_path)
  for chain in evo_struct.chains:
    if chain.is_protein:
      _calc_phi_psi(chain)
      for res in chain.residues:
        residue_calc_sidechain_torsions(res, topologies)

  total_atoms = 0
  valid_atoms = 0
  missing_atoms = 0
  missing_h_atoms = 0
  hb_h_atoms = 0
  hb_a_atoms = 0
  residues_with_default_phipsi = 0
  protein_residues = 0
  torsion_expected = 0
  torsion_missing = 0
  dunbrack_bins = 0
  dunbrack_missing = 0

  for chain in evo_struct.chains:
    for res in chain.residues:
      if res.is_protein:
        protein_residues += 1
        if res.phipsi == (-60.0, 60.0):
          residues_with_default_phipsi += 1
        expected = _DUNBRACK_TORSION_COUNT.get(res.name, 0)
        torsion_expected += expected
        if expected > 0 and len(res.xtorsions) == 0:
          torsion_missing += 1
        phi = int(res.phipsi[0])
        psi = int(res.phipsi[1])
        bin_index = ((phi + 180) // 10) * 36 + ((psi + 180) // 10)
        if 0 <= bin_index < len(dun.bins):
          if res.name in dun.bins[bin_index].by_residue:
            dunbrack_bins += 1
          else:
            dunbrack_missing += 1

      for atom in res.atoms.values():
        total_atoms += 1
        if atom.is_xyz_valid:
          valid_atoms += 1
        else:
          missing_atoms += 1
          if atom.is_h:
            missing_h_atoms += 1
        if atom.is_hbond_h:
          hb_h_atoms += 1
        if atom.is_hbond_a:
          hb_a_atoms += 1

  return {
    "total_atoms": total_atoms,
    "valid_atoms": valid_atoms,
    "missing_atoms": missing_atoms,
    "missing_h_atoms": missing_h_atoms,
    "hb_h_atoms": hb_h_atoms,
    "hb_a_atoms": hb_a_atoms,
    "protein_residues": protein_residues,
    "default_phipsi_residues": residues_with_default_phipsi,
    "torsion_expected_total": torsion_expected,
    "torsion_missing_residues": torsion_missing,
    "dunbrack_bins_with_residue": dunbrack_bins,
    "dunbrack_bins_missing_residue": dunbrack_missing,
  }
