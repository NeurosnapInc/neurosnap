"""
Shared constants for EvoEF2 scoring.
"""

from __future__ import annotations

import math

# -----------------------------
# Core constants (mirrors EvoEF2)
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
  94: "rna_suite",
  95: "dna_bibii",
  96: "dna_chi",
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
  "rna_suite": 94,
  "dna_bibii": 95,
  "dna_chi": 96,
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

# Nucleic acid residues (common PDB names and CHARMM names)
NA_RNA_RESIDUES = {"A", "C", "G", "U", "I"}
NA_DNA_RESIDUES = {"DA", "DC", "DG", "DT", "DI"}
NA_CHARMm_RESIDUES = {"ADE", "GUA", "CYT", "THY", "URA"}
NA_RESIDUES = NA_RNA_RESIDUES | NA_DNA_RESIDUES | NA_CHARMm_RESIDUES

# Map common PDB names to CHARMM NA residue names.
NA_RESIDUE_MAP = {
  "A": "ADE",
  "G": "GUA",
  "C": "CYT",
  "U": "URA",
  "DA": "ADE",
  "DG": "GUA",
  "DC": "CYT",
  "DT": "THY",
  "I": "GUA",
  "DI": "GUA",
}

# Backbone atoms for nucleic acids (sugar/phosphate)
NA_BACKBONE_ATOMS = {
  "P",
  "OP1",
  "OP2",
  "OP3",
  "O1P",
  "O2P",
  "O5'",
  "C5'",
  "C4'",
  "O4'",
  "C3'",
  "O3'",
  "C2'",
  "O2'",
  "C1'",
}

# Nucleic acid torsion suite reference:
# "RNA backbone is rotameric" (PNAS, 2003) Table 3
# Table 3: δ-1, ε-1, ζ-1, α, β, γ, δ (degrees)
# https://www.pnas.org/doi/full/10.1073/pnas.1835769100
RNA_SUITE_CONFORMERS = [
  ("3' e p p t p 3'", [84, -150, 60, 65, 180, 55, 84]),
  ("3' e -140 p t p 3'", [84, -125, -140, 65, 180, 55, 84]),
  ("3' e m p t p 3'", [84, -120, -100, 70, 180, 55, 84]),
  ("2' e p p t p 3'", [147, -100, 85, 65, 180, 55, 84]),
  ("2' e t p t p 3'", [147, -120, 175, 65, 180, 55, 84]),
  ("2' e m p t p 3'", [147, -100, -100, 65, 180, 55, 84]),
  ("3' e p p 110 t 3'", [84, -150, 60, 65, 110, 180, 84]),
  ("2' e p p t t 3'", [147, -100, 85, 70, -170, 180, 84]),
  ("3' e m t t p 3'", [84, -150, -75, 165, 165, 55, 84]),
  ("2' e p t t p 3'", [147, -100, 85, 165, 165, 55, 84]),
  ("2' e t t t p 3'", [147, -120, 175, 165, 165, 55, 84]),
  ("2' e m t t p 3'", [147, -100, -70, 165, 165, 55, 84]),
  ("3' e m t 135 t 3'", [84, -150, -75, 170, 135, 175, 84]),
  ("3' e m t t t 3'", [84, -150, -75, 155, -175, 180, 84]),
  ("3' e t m t p 3'", [84, -140, 175, -65, 175, 55, 84]),
  ("3' e -140 m t p 3'", [84, -125, -140, -65, 175, 55, 84]),
  ("3' e m m t p 3'", [84, -150, -75, -65, 175, 55, 84]),
  ("2' e t m t p 3'", [147, -120, 175, -65, 175, 55, 84]),
  ("2' e m m t p 3'", [147, -100, -70, -65, 175, 55, 84]),
  ("3' e m m -135 p 3'", [84, -150, -75, -65, -135, 55, 84]),
  ("3' e m -110 80 t 3'", [84, -150, -75, -110, 80, 170, 84]),
  ("2' e m m t t 3'", [147, -100, -70, -65, 170, 180, 84]),
  ("3' e p p t p 2'", [84, -150, 60, 65, 180, 55, 147]),
  ("3' e t p t p 2'", [84, -140, 175, 65, 180, 55, 147]),
  ("3' e -140 p t p 2'", [84, -125, -140, 65, 180, 55, 147]),
  ("2' e p p t p 2'", [147, -100, 85, 65, 180, 55, 147]),
  ("2' e t p t p 2'", [147, -120, 175, 65, 180, 55, 147]),
  ("3' e p p 110 t 2'", [84, -150, 60, 65, 110, 180, 147]),
  ("3' e -140 p t m 2'", [84, -125, -140, 65, 170, -60, 147]),
  ("3' e p t t p 2'", [84, -150, 45, 155, 150, 55, 147]),
  ("3' e m t t p 2'", [84, -150, -75, 165, 165, 55, 147]),
  ("3' e m t t t 2'", [84, -165, -75, 180, -175, 180, 147]),
  ("3' e t m t p 2'", [84, -140, 175, -65, 175, 55, 147]),
  ("3' e -140 m t p 2'", [84, -125, -140, -65, 175, 55, 147]),
  ("3' e m m t p 2'", [84, -150, -100, -65, 180, 55, 147]),
  ("2' e p m t p 2'", [147, -115, 85, -85, 175, 55, 147]),
  ("2' e t m t p 2'", [147, -120, 175, -65, 175, 55, 147]),
  ("2' e m m t p 2'", [147, -100, -70, -65, 175, 55, 147]),
  ("3' e m m -135 p 2'", [84, -150, -75, -65, -135, 55, 147]),
  ("2' e m m -135 p 2'", [147, -100, -70, -65, -135, 55, 147]),
  ("3' e m m t m 2'", [84, -150, -75, -65, -160, -65, 147]),
  ("2' e m m t m 2'", [147, -100, -70, -65, -160, -65, 147]),
]

# DNA BI/BII and chi (degrees); TODO: replace with curated distributions.
DNA_BI_CENTER = (-85.0, -65.0)   # (epsilon, zeta)
DNA_BII_CENTER = (-160.0, -85.0)
DNA_CHI_PURINE_CENTERS = (-120.0, 60.0)  # anti/syn
DNA_CHI_PYRIMIDINE_CENTERS = (-120.0,)   # anti

_ATOM_ORDER_SEQUENCE = "ABGDEZ"

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
