"""Sequence- and amino-acid-related constants."""

from dataclasses import dataclass
from typing import Dict, Optional

## Amino Acid Codes and Properties
# Codes for standard amino acids
STANDARD_AAs = set("ACDEFGHIKLMNPQRSTVWY")


# Amino acid Record class
@dataclass(frozen=True)
class AARecord:
  code: Optional[str]  # 1-letter code; None for if unavailable
  abr: str  # 3-letter abbreviation or CCD code
  name: str  # full name (upper-cased)
  is_standard: bool  # True for the 20 canonical AAs
  standard_equiv_abr: Optional[str]  # e.g., "LYS" for KCX; None if standard or unknown


## Amino acids keyed by ABR
AA_RECORDS: Dict[str, AARecord] = {
  "ALA": AARecord("A", "ALA", "ALANINE", True, None),
  "ARG": AARecord("R", "ARG", "ARGININE", True, None),
  "ASN": AARecord("N", "ASN", "ASPARAGINE", True, None),
  "ASP": AARecord("D", "ASP", "ASPARTIC ACID", True, None),
  "CYS": AARecord("C", "CYS", "CYSTEINE", True, None),
  "GLN": AARecord("Q", "GLN", "GLUTAMINE", True, None),
  "GLU": AARecord("E", "GLU", "GLUTAMIC ACID", True, None),
  "GLY": AARecord("G", "GLY", "GLYCINE", True, None),
  "HIS": AARecord("H", "HIS", "HISTIDINE", True, None),
  "ILE": AARecord("I", "ILE", "ISOLEUCINE", True, None),
  "LEU": AARecord("L", "LEU", "LEUCINE", True, None),
  "LYS": AARecord("K", "LYS", "LYSINE", True, None),
  "MET": AARecord("M", "MET", "METHIONINE", True, None),
  "PHE": AARecord("F", "PHE", "PHENYLALANINE", True, None),
  "PRO": AARecord("P", "PRO", "PROLINE", True, None),
  "SER": AARecord("S", "SER", "SERINE", True, None),
  "THR": AARecord("T", "THR", "THREONINE", True, None),
  "TRP": AARecord("W", "TRP", "TRYPTOPHAN", True, None),
  "TYR": AARecord("Y", "TYR", "TYROSINE", True, None),
  "VAL": AARecord("V", "VAL", "VALINE", True, None),
  "PYL": AARecord("O", "PYL", "PYRROLYSINE", False, "LYS"),
  "SEC": AARecord("U", "SEC", "SELENOCYSTEINE", False, "CYS"),
  "ASX": AARecord("B", "ASX", "ASPARAGINE/ASPARTIC ACID", False, "ASP"),
  "GLX": AARecord("Z", "GLX", "GLUTAMINE/GLUTAMIC ACID", False, "GLU"),
  "XLE": AARecord("J", "XLE", "LEUCINE/ISOLEUCINE", False, "LEU"),
  "UNK": AARecord("X", "UNK", "UNKNOWN", False, None),
  "TRM": AARecord("*", "TRM", "TERMINATION", False, None),
  "LLP": AARecord(None, "LLP", "Nε-LIPOYL-LYSINE", False, "LYS"),
  "TPO": AARecord(None, "TPO", "O-PHOSPHOTHREONINE", False, "THR"),
  "CSS": AARecord(None, "CSS", "SULFONATED CYSTEINE", False, "CYS"),
  "OCS": AARecord(None, "OCS", "CYSTEINE-S-SULFONIC ACID", False, "CYS"),
  "CSO": AARecord(None, "CSO", "S-HYDROXYCYSTEINE (CYSTEINE SULFINIC ACID)", False, "CYS"),
  "PCA": AARecord(None, "PCA", "PYROGLUTAMIC ACID", False, "GLU"),
  "KCX": AARecord(None, "KCX", "CARBOXYLYSINE", False, "LYS"),
  "CME": AARecord(None, "CME", "S-METHYLCYSTEINE", False, "CYS"),
  "MLY": AARecord(None, "MLY", "Nε-METHYLLYSINE", False, "LYS"),
  "SEP": AARecord(None, "SEP", "O-PHOSPHOSERINE", False, "SER"),
  "CSX": AARecord(None, "CSX", "CYSTEINE OXIDATION PRODUCT (UNSPECIFIED)", False, "CYS"),
  "CSD": AARecord(None, "CSD", "CYSTEINE DISULFIDE", False, "CYS"),
  "MSE": AARecord(None, "MSE", "SELENOMETHIONINE", False, "MET"),
  "MHO": AARecord(None, "MHO", "METHIONINE SULFOXIDE", False, "MET"),
}

# Alias map: every searchable token → ABR
# (1-letter codes, 3-letter codes, and names)
AA_ALIASES: Dict[str, str] = {}
for abr, rec in AA_RECORDS.items():
  if rec.code is not None:
    AA_ALIASES[rec.code] = abr
  AA_ALIASES[abr] = abr
  AA_ALIASES[rec.name] = abr

## Amino acid molecular masses
# Average residue masses (in Daltons) for amino acids *as incorporated into peptides/proteins*.
# These values already account for the loss of one H2O molecule during peptide bond formation,
# so they represent the contribution of each amino acid *residue* in a chain.
# Source: https://proteomicsresource.washington.edu/protocols06/masses.php (Average masses)
AA_MASS_PROTEIN_AVG = {
  "A": 71.07790000,
  "R": 156.1856800,
  "N": 114.1026400,
  "D": 115.0874000,
  "C": 103.1429000,
  "E": 129.1139800,
  "Q": 128.1292200,
  "G": 57.05132000,
  "H": 137.1392800,
  "I": 113.1576400,
  "L": 113.1576400,
  "K": 128.1722800,
  "M": 131.1960600,
  "F": 147.1738600,
  "P": 97.11518000,
  "S": 87.07730000,
  "T": 101.1038800,
  "W": 186.2099000,
  "Y": 163.1732600,
  "V": 99.13106000,
  "O": 237.2981600,
  "U": 150.0379000,
}

# Monoisotopic residue masses (in Daltons) for amino acids *as incorporated into peptides/proteins*.
# These use the exact mass of the most abundant isotope of each element (e.g., 12C, 1H, 16O, 14N).
# Like the average masses above, these are residue contributions (with H2O already removed).
# Source: https://proteomicsresource.washington.edu/protocols06/masses.php (Monoisotopic masses)
AA_MASS_PROTEIN_MONO = {
  "A": 71.0371138050,
  "R": 156.101111050,
  "N": 114.042927470,
  "D": 115.026943065,
  "C": 103.009184505,
  "E": 129.042593135,
  "Q": 128.058577540,
  "G": 57.0214637350,
  "H": 137.058911875,
  "I": 113.084064015,
  "L": 113.084064015,
  "K": 128.094963050,
  "M": 131.040484645,
  "F": 147.068413945,
  "P": 97.0527638750,
  "S": 87.0320284350,
  "T": 101.047678505,
  "W": 186.079312980,
  "Y": 163.063328575,
  "V": 99.0684139450,
  "O": 237.147726925,
  "U": 150.953633405,
}

# Average molecular masses (in Daltons) of *free amino acids* (not incorporated into a chain).
# These values include the full amino acid with terminal H and OH groups, i.e. before peptide bond formation.
# Often used for small-molecule calculations or educational purposes, but not for intact peptides/proteins.
AA_MASS_FREE = {
  "A": 89.090,
  "R": 174.20,
  "N": 132.12,
  "D": 133.10,
  "C": 121.15,
  "E": 147.13,
  "Q": 146.15,
  "G": 75.070,
  "H": 155.16,
  "I": 131.17,
  "L": 131.17,
  "K": 146.19,
  "M": 149.21,
  "F": 165.19,
  "P": 115.13,
  "S": 105.09,
  "T": 119.12,
  "W": 204.23,
  "Y": 181.19,
  "V": 117.15,
  "O": 255.31,
  "U": 168.06,
}

## pKa Values
# Default pKa set (EMBOSS-like). Values are typical textbook approximations.
# You can swap these for another set (e.g., Bjellqvist, IPC) if desired.
DEFAULT_PKA = {
  "N_TERMINUS": 8.6,
  "C_TERMINUS": 3.6,
  "C": 8.50,
  "D": 3.90,
  "E": 4.10,
  "Y": 10.1,
  "H": 6.50,
  "K": 10.8,
  "R": 12.5,
  "U": 5.20,
}

__all__ = [
  "AARecord",
  "AA_ALIASES",
  "AA_MASS_FREE",
  "AA_MASS_PROTEIN_AVG",
  "AA_MASS_PROTEIN_MONO",
  "AA_RECORDS",
  "DEFAULT_PKA",
  "STANDARD_AAs",
]
