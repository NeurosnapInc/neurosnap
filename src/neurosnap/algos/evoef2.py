"""
A Python implementation of the EvoEF2 protein scoring function / force field.
Ported from the native EvoEF2 reference implementation.
Original Implementation: https://github.com/tommyhuangthu/EvoEF2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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

WEIGHT_KEY_TO_INDEX = {
  "reference_ALA": 1,
  "reference_CYS": 2,
  "reference_ASP": 3,
  "reference_GLU": 4,
  "reference_PHE": 5,
  "reference_GLY": 6,
  "reference_HIS": 7,
  "reference_ILE": 8,
  "reference_LYS": 9,
  "reference_LEU": 10,
  "reference_MET": 11,
  "reference_ASN": 12,
  "reference_PRO": 13,
  "reference_GLN": 14,
  "reference_ARG": 15,
  "reference_SER": 16,
  "reference_THR": 17,
  "reference_VAL": 18,
  "reference_TRP": 19,
  "reference_TYR": 20,
  "intraR_vdwatt": 21,
  "intraR_vdwrep": 22,
  "intraR_electr": 23,
  "intraR_deslvP": 24,
  "intraR_deslvH": 25,
  "intraR_hbscbb_dis": 26,
  "intraR_hbscbb_the": 27,
  "intraR_hbscbb_phi": 28,
  "aapropensity": 91,
  "ramachandran": 92,
  "dunbrack": 93,
  "interS_vdwatt": 31,
  "interS_vdwrep": 32,
  "interS_electr": 33,
  "interS_deslvP": 34,
  "interS_deslvH": 35,
  "interS_ssbond": 36,
  "interS_hbbbbb_dis": 41,
  "interS_hbbbbb_the": 42,
  "interS_hbbbbb_phi": 43,
  "interS_hbscbb_dis": 44,
  "interS_hbscbb_the": 45,
  "interS_hbscbb_phi": 46,
  "interS_hbscsc_dis": 47,
  "interS_hbscsc_the": 48,
  "interS_hbscsc_phi": 49,
  "interD_vdwatt": 51,
  "interD_vdwrep": 52,
  "interD_electr": 53,
  "interD_deslvP": 54,
  "interD_deslvH": 55,
  "interD_ssbond": 56,
  "interD_hbbbbb_dis": 61,
  "interD_hbbbbb_the": 62,
  "interD_hbbbbb_phi": 63,
  "interD_hbscbb_dis": 64,
  "interD_hbscbb_the": 65,
  "interD_hbscbb_phi": 66,
  "interD_hbscsc_dis": 67,
  "interD_hbscsc_the": 68,
  "interD_hbscsc_phi": 69,
  "ligand_vdwatt": 71,
  "ligand_vdwrep": 72,
  "ligand_electr": 73,
  "ligand_deslvP": 74,
  "ligand_deslvH": 75,
  "ligand_hbscbb_dis": 84,
  "ligand_hbscbb_the": 85,
  "ligand_hbscbb_phi": 86,
  "ligand_hbscsc_dis": 87,
  "ligand_hbscsc_the": 88,
  "ligand_hbscsc_phi": 89,
}

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
  name: str
  param: AtomParam
  chain: str
  pos: int
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
  a: str
  b: str
  bond_type: int = 1


@dataclass
class CharmmIC:
  atom_a: str
  atom_b: str
  atom_c: str
  atom_d: str
  ic_param: List[float]
  torsion_proper: bool


@dataclass
class ResidueTopology:
  name: str
  atoms: List[str] = field(default_factory=list)
  deletes: List[str] = field(default_factory=list)
  bonds: List[Bond] = field(default_factory=list)
  ics: List[CharmmIC] = field(default_factory=list)


@dataclass
class Residue:
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
    return self.atoms.get(name)


@dataclass
class Chain:
  name: str
  residues: List[Residue] = field(default_factory=list)
  is_protein: bool = True


@dataclass
class Structure:
  chains: List[Chain] = field(default_factory=list)

  def all_residues(self) -> Iterable[Residue]:
    for chain in self.chains:
      for res in chain.residues:
        yield res


@dataclass
class AAppTable:
  aap: np.ndarray  # shape (36,36,20)


@dataclass
class RamaTable:
  rama: np.ndarray  # shape (36,36,20)


# -----------------------------
# Geometry utilities
# -----------------------------


def safe_acos(cos_value: float) -> float:
  if cos_value > 1.0:
    cos_value = 1.0
  elif cos_value < -1.0:
    cos_value = -1.0
  return math.acos(cos_value)


def rad_to_deg(rad: float) -> float:
  return rad * 180.0 / PI


def deg_to_rad(deg: float) -> float:
  return deg * PI / 180.0


def xyz_distance(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.linalg.norm(a - b))


def xyz_angle(v1: np.ndarray, v2: np.ndarray) -> float:
  norm = np.linalg.norm(v1) * np.linalg.norm(v2)
  if norm < 1e-12:
    return 1000.0
  return safe_acos(float(np.dot(v1, v2) / norm))


def xyz_rotate_around(p: np.ndarray, axis_from: np.ndarray, axis_to: np.ndarray, angle: float) -> np.ndarray:
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
  result = result @ m.T
  return result + axis_from


def get_fourth_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray, ic_param: Sequence[float]) -> np.ndarray:
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
  if sin_value < 0:
    return -safe_acos(cos_value)
  return safe_acos(cos_value)


# -----------------------------
# Parsing of EvoEF2 libraries
# -----------------------------


def _default_evoef2_root() -> Path:
  return Path(__file__).resolve().parents[3] / "EvoEF2"


def load_atom_params(param_path: Optional[Path] = None) -> Dict[str, Dict[str, AtomParam]]:
  if param_path is None:
    param_path = _default_evoef2_root() / "library" / "param_charmm19_lk.prm"
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
  if top_path is None:
    top_path = _default_evoef2_root() / "library" / "top_polh19_prot.inp"
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


def load_weights(weight_path: Optional[Path] = None) -> List[float]:
  if weight_path is None:
    weight_path = _default_evoef2_root() / "wread" / "weight_EvoEF2.txt"
  weights = [1.0] * MAX_EVOEF_ENERGY_TERM_NUM
  if not weight_path.exists():
    logger.warning("EvoEF2 weight file not found at %s; using unit weights.", weight_path)
    return weights
  with open(weight_path, "r") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("!"):
        continue
      parts = line.split()
      if len(parts) < 2:
        continue
      term = parts[0]
      val = float(parts[1])
      if term in WEIGHT_KEY_TO_INDEX:
        weights[WEIGHT_KEY_TO_INDEX[term]] = val
  return weights


def load_aapropensity(path: Optional[Path] = None) -> AAppTable:
  if path is None:
    path = _default_evoef2_root() / "library" / "aapropensity.txt"
  aap = np.zeros((36, 36, 20), dtype=float)
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split()
      if len(parts) != 22:
        continue
      phi = int(parts[0])
      psi = int(parts[1])
      phi_index = (phi + 180) // 10
      psi_index = (psi + 180) // 10
      for j in range(20):
        aap[phi_index, psi_index, j] = float(parts[j + 2])
  return AAppTable(aap=aap)


def load_ramachandran(path: Optional[Path] = None) -> RamaTable:
  if path is None:
    path = _default_evoef2_root() / "library" / "ramachandran.txt"
  rama = np.zeros((36, 36, 20), dtype=float)
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split()
      if len(parts) != 22:
        continue
      phi = int(parts[0])
      psi = int(parts[1])
      phi_index = (phi + 180) // 10
      psi_index = (psi + 180) // 10
      for j in range(20):
        rama[phi_index, psi_index, j] = float(parts[j + 2])
  return RamaTable(rama=rama)


def _default_dunbrack_path() -> Path:
  return _default_evoef2_root() / "library" / "dun2010bb3per.lib"


@dataclass
class DunbrackRotamer:
  torsions: List[float]
  deviations: List[float]
  probability: float


@dataclass
class DunbrackBin:
  by_residue: Dict[str, List[DunbrackRotamer]] = field(default_factory=dict)


@dataclass
class DunbrackLibrary:
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
  for bond in bonds:
    if (atom1 == bond.a and atom2 == bond.b) or (atom2 == bond.a and atom1 == bond.b):
      return True
  return False


def _residue_intra_bond_13(atom1: str, atom2: str, bonds: List[Bond]) -> bool:
  for bond in bonds:
    if atom1 == bond.a:
      if _residue_intra_bond_12(bond.b, atom2, bonds):
        return True
    elif atom1 == bond.b:
      if _residue_intra_bond_12(bond.a, atom2, bonds):
        return True
  return False


def _residue_intra_bond_14(atom1: str, atom2: str, bonds: List[Bond]) -> bool:
  for bond in bonds:
    if atom1 == bond.a:
      if _residue_intra_bond_13(bond.b, atom2, bonds):
        return True
    elif atom1 == bond.b:
      if _residue_intra_bond_13(bond.a, atom2, bonds):
        return True
  return False


def _residue_intra_bond_connection(atom1: str, atom2: str, bonds: List[Bond]) -> int:
  if _residue_intra_bond_12(atom1, atom2, bonds):
    return 12
  if _residue_intra_bond_13(atom1, atom2, bonds):
    return 13
  if _residue_intra_bond_14(atom1, atom2, bonds):
    return 14
  return 15


_ATOM_ORDER_SEQUENCE = "ABGDEZ"


def _protein_atom_order(atom_name: str) -> int:
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
  atom = res.get_atom(name)
  if atom is None or not atom.is_xyz_valid:
    return None
  return atom.xyz


def _calc_atom_xyz(res: Residue, topologies: Dict[str, ResidueTopology], prev_res: Optional[Residue], next_res: Optional[Residue], atom_name: str) -> Optional[np.ndarray]:
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
  for i, res in enumerate(chain.residues):
    prev_res = chain.residues[i - 1] if i > 0 else None
    next_res = chain.residues[i + 1] if i < len(chain.residues) - 1 else None
    residue_calc_all_atom_xyz(res, topologies, prev_res, next_res)


def _apply_patch(res: Residue, patch_name: str, params: Dict[str, Dict[str, AtomParam]], topologies: Dict[str, ResidueTopology], delete_o: bool = True) -> None:
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
        res.atoms[atom_name] = Atom(name=atom_name, param=param, chain=res.chain, pos=res.pos, xyz=xyz, is_xyz_valid=valid)
      else:
        res.atoms[atom_name] = Atom(name=atom_name, param=param, chain=res.chain, pos=res.pos)
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
  if res.name not in params:
    return
  for atom_name, param in params[res.name].items():
    if atom_name not in res.atoms:
      res.atoms[atom_name] = Atom(name=atom_name, param=param, chain=res.chain, pos=res.pos)
    else:
      atom = res.atoms[atom_name]
      xyz = atom.xyz.copy()
      valid = atom.is_xyz_valid
      res.atoms[atom_name] = Atom(name=atom_name, param=param, chain=res.chain, pos=res.pos, xyz=xyz, is_xyz_valid=valid)


def _add_bonds_from_topology(res: Residue, topologies: Dict[str, ResidueTopology]) -> None:
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
  params = load_atom_params(param_path)
  topologies = load_topology(topo_path)
  protein = structure if isinstance(structure, Protein) else Protein(structure, format="auto")
  df = protein.df
  df = df[df["model"] == protein.models()[0]]

  chains: List[Chain] = []
  for chain_id in sorted(df["chain"].unique()):
    df_chain = df[df["chain"] == chain_id]
    residues: List[Residue] = []
    for (res_id, res_name), df_res in df_chain.groupby(["res_id", "res_name"], sort=True):
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
      # set coordinates from PDB
      if res_name not in params and not df_res.empty:
        logger.warning("No EvoEF2 parameters for residue %s; skipping parameterization for its atoms.", res_name)
      for _, row in df_res.iterrows():
        atom_name = row["atom_name"]
        if atom_name not in res.atoms:
          if res_name in params and atom_name in params[res_name]:
            res.atoms[atom_name] = Atom(name=atom_name, param=params[res_name][atom_name], chain=chain_id, pos=int(res_id))
          else:
            continue
        atom = res.atoms[atom_name]
        atom.xyz = np.array([row["x"], row["y"], row["z"]], dtype=float)
        atom.is_xyz_valid = True
      residues.append(res)
    chain_is_protein = any(r.is_protein for r in residues)
    chain = Chain(name=chain_id, residues=residues, is_protein=chain_is_protein)
    # patch NTER and CTER for protein chains
    if chain.is_protein and residues:
      _patch_nter_or_cter(residues[0], params, topologies, "NTER")
      if residues[0].get_atom("HT1") is not None or residues[0].get_atom("HN1") is not None:
        _patch_nter_or_cter(residues[-1], params, topologies, "CTER")
    chain_calc_all_atom_xyz(chain, topologies)
    if chain.is_protein:
      for res in chain.residues:
        if res.is_protein:
          residue_calc_sidechain_torsions(res, topologies)
    chains.append(chain)

  return Structure(chains=chains)


# -----------------------------
# Energies
# -----------------------------


def energy_term_initialize() -> List[float]:
  return [0.0] * MAX_EVOEF_ENERGY_TERM_NUM


def energy_term_weighting(energy_terms: List[float], weights: List[float]) -> List[float]:
  weighted = energy_terms[:]
  total = 0.0
  for i in range(1, MAX_EVOEF_ENERGY_TERM_NUM):
    weighted[i] *= weights[i]
    total += weighted[i]
  weighted[0] = total
  return weighted


def _vdw_att(atom1: Atom, atom2: Atom, distance: float, bond_type: int) -> float:
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


def _ssbond(atom_s1: Atom, atom_s2: Atom, atom_cb1: Atom, atom_cb2: Atom, atom_ca1: Atom, atom_ca2: Atom) -> float:
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
  weight_path: Optional[Path] = None,
  aapropensity_path: Optional[Path] = None,
  ramachandran_path: Optional[Path] = None,
) -> Dict[str, float]:
  params = load_atom_params(param_path)
  topologies = load_topology(topo_path)
  weights = load_weights(weight_path)
  aap = load_aapropensity(aapropensity_path)
  rama = load_ramachandran(ramachandran_path)

  evo_struct = rebuild_missing_atoms(structure, param_path=param_path, topo_path=topo_path)
  for chain in evo_struct.chains:
    if chain.is_protein:
      _calc_phi_psi(chain)

  terms = energy_term_initialize()
  # compute stability across the whole structure
  for chain in evo_struct.chains:
    for i, res in enumerate(chain.residues):
      if not res.is_protein:
        continue
      _aa_reference_energy(res, terms)
      _residue_intra_energy(res, terms)
      _aa_propensity_ramachandran(res, aap, rama, terms)
      # same-chain pairs
      for j in range(i + 1, len(chain.residues)):
        other = chain.residues[j]
        if j == i + 1:
          _residue_and_next_energy(res, other, terms)
        else:
          _residue_other_same_chain(res, other, terms)
  # different chains (avoid double counting by index)
  for i, chain_i in enumerate(evo_struct.chains):
    for k in range(i + 1, len(evo_struct.chains)):
      chain_k = evo_struct.chains[k]
      for res_i in chain_i.residues:
        for res_k in chain_k.residues:
          if res_i.is_protein and res_k.is_protein:
            _residue_other_diff_chain(res_i, res_k, terms)
          elif res_i.is_protein and not res_k.is_protein:
            _residue_and_ligand_energy(res_i, res_k, terms)
          elif not res_i.is_protein and res_k.is_protein:
            _residue_and_ligand_energy(res_k, res_i, terms)

  weighted = energy_term_weighting(terms, weights)
  return _energy_terms_to_dict(weighted)


def calculate_interface_energy(
  structure: Union[Protein, str, Path],
  split1: Sequence[str],
  split2: Sequence[str],
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
  weight_path: Optional[Path] = None,
) -> Dict[str, float]:
  weights = load_weights(weight_path)
  evo_struct = rebuild_missing_atoms(structure, param_path=param_path, topo_path=topo_path)
  terms = energy_term_initialize()
  set1 = set(split1)
  set2 = set(split2)
  for i, chain_i in enumerate(evo_struct.chains):
    for k in range(i + 1, len(evo_struct.chains)):
      chain_k = evo_struct.chains[k]
      if not ((chain_i.name in set1 and chain_k.name in set2) or (chain_i.name in set2 and chain_k.name in set1)):
        continue
      for res_i in chain_i.residues:
        for res_k in chain_k.residues:
          if res_i.is_protein and res_k.is_protein:
            _residue_other_diff_chain(res_i, res_k, terms)
          elif res_i.is_protein and not res_k.is_protein:
            _residue_and_ligand_energy(res_i, res_k, terms)
          elif not res_i.is_protein and res_k.is_protein:
            _residue_and_ligand_energy(res_k, res_i, terms)
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
) -> Dict[str, Dict[str, float]]:
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
  )
  split2_energy = _calculate_stability_from_structure(
    split2_struct,
    weight_path=weight_path,
    aapropensity_path=aapropensity_path,
    ramachandran_path=ramachandran_path,
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
  weight_path: Optional[Path] = None,
  aapropensity_path: Optional[Path] = None,
  ramachandran_path: Optional[Path] = None,
) -> Dict[str, float]:
  weights = load_weights(weight_path)
  aap = load_aapropensity(aapropensity_path)
  rama = load_ramachandran(ramachandran_path)
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
      for j in range(i + 1, len(chain.residues)):
        other = chain.residues[j]
        if j == i + 1:
          _residue_and_next_energy(res, other, terms)
        else:
          _residue_other_same_chain(res, other, terms)
  for i, chain_i in enumerate(evo_struct.chains):
    for k in range(i + 1, len(evo_struct.chains)):
      chain_k = evo_struct.chains[k]
      for res_i in chain_i.residues:
        for res_k in chain_k.residues:
          if res_i.is_protein and res_k.is_protein:
            _residue_other_diff_chain(res_i, res_k, terms)
          elif res_i.is_protein and not res_k.is_protein:
            _residue_and_ligand_energy(res_i, res_k, terms)
          elif not res_i.is_protein and res_k.is_protein:
            _residue_and_ligand_energy(res_k, res_i, terms)
  weighted = energy_term_weighting(terms, weights)
  return _energy_terms_to_dict(weighted)


def _energy_terms_to_dict(energy_terms: List[float]) -> Dict[str, float]:
  result: Dict[str, float] = {}
  result[ENERGY_TERM_NAMES[0]] = energy_terms[0]
  for idx in ENERGY_TERM_ORDER:
    result[ENERGY_TERM_NAMES[idx]] = energy_terms[idx]
  return result


def _subtract_energy_dicts(full: Dict[str, float], a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
  result: Dict[str, float] = {}
  for key in full.keys():
    result[key] = full.get(key, 0.0) - a.get(key, 0.0) - b.get(key, 0.0)
  return result
