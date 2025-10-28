"""
This file contains constants.
"""

from dataclasses import dataclass
from typing import Dict, Optional

## Backbone Atoms
# Names of atoms that are part of a protein's backbone structure
BACKBONE_ATOMS_AA = {"N", "CA", "C"}
# Names of atoms that are part of a DNA backbone structure
BACKBONE_ATOMS_DNA = {
  # Phosphorus
  "P",
  # Phosphate oxygens (sometimes labeled OP1, OP2)
  "O1P",
  "O2P",
  # Alternate naming convention
  "OP1",
  "OP2",
  # Bridging oxygens between sugar and phosphate
  "O3'",
  "O5'",
  # Sugar atoms
  "C3'",
  "C4'",
  "C5'",
  "O4'",
  "C1'",
  "C2'",
}
# Names of atoms that are part of an RNA backbone structure
# (Same as DNA but includes the 2'-OH group)
BACKBONE_ATOMS_RNA = BACKBONE_ATOMS_DNA.union({"O2'"})

## Nucleotide Codes
# Single-letter PDB residue codes for standard DNA residues
NUC_DNA_CODES = {"DA", "DT", "DC", "DG"}
# Single-letter PDB residue codes for standard RNA residues
NUC_RNA_CODES = {"A", "U", "C", "G"}
# Codes for standard nucleotides (both RNA and DNA)
STANDARD_NUCLEOTIDES = NUC_DNA_CODES.union(NUC_RNA_CODES)

## Amino Acid Codes and Properties
# Codes for standard amino acids
STANDARD_AAs = set("ACDEFGHIKLMNPQRSTVWY")
# List of hydrophobic residues
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}


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
  # STANDARD AMINO ACIDS
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
  # NON-STANDARD / SPECIAL AMINO ACIDS (sequence-level)
  "PYL": AARecord("O", "PYL", "PYRROLYSINE", False, "LYS"),
  "SEC": AARecord("U", "SEC", "SELENOCYSTEINE", False, "CYS"),
  "ASX": AARecord("B", "ASX", "ASPARAGINE/ASPARTIC ACID", False, "ASP"),
  "GLX": AARecord("Z", "GLX", "GLUTAMINE/GLUTAMIC ACID", False, "GLU"),
  "XLE": AARecord("J", "XLE", "LEUCINE/ISOLEUCINE", False, "LEU"),
  "UNK": AARecord("X", "UNK", "UNKNOWN", False, None),
  "TRM": AARecord("*", "TRM", "TERMINATION", False, None),
  # NON-STANDARD / MODIFIED (from CCD)
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

## Amino acid molecular weights
# Average residue masses (in Daltons) for amino acids *as incorporated into peptides/proteins*.
# These values already account for the loss of one H2O molecule during peptide bond formation,
# so they represent the contribution of each amino acid *residue* in a chain.
# Source: https://proteomicsresource.washington.edu/protocols06/masses.php (Average masses)
AA_WEIGHTS_PROTEIN_AVG = {
  "A": 71.07790000,  # Alanine
  "R": 156.1856800,  # Arginine
  "N": 114.1026400,  # Asparagine
  "D": 115.0874000,  # Aspartic acid
  "C": 103.1429000,  # Cysteine
  "E": 129.1139800,  # Glutamic acid
  "Q": 128.1292200,  # Glutamine
  "G": 57.05132000,  # Glycine
  "H": 137.1392800,  # Histidine
  "I": 113.1576400,  # Isoleucine
  "L": 113.1576400,  # Leucine
  "K": 128.1722800,  # Lysine
  "M": 131.1960600,  # Methionine
  "F": 147.1738600,  # Phenylalanine
  "P": 97.11518000,  # Proline
  "S": 87.07730000,  # Serine
  "T": 101.1038800,  # Threonine
  "W": 186.2099000,  # Tryptophan
  "Y": 163.1732600,  # Tyrosine
  "V": 99.13106000,  # Valine
  "O": 237.2981600,  # pyrrolysine
  "U": 150.0379000,  # selenocysteine
}

# Monoisotopic residue masses (in Daltons) for amino acids *as incorporated into peptides/proteins*.
# These use the exact mass of the most abundant isotope of each element (e.g., 12C, 1H, 16O, 14N).
# Like the average masses above, these are residue contributions (with H2O already removed).
# Source: https://proteomicsresource.washington.edu/protocols06/masses.php (Monoisotopic masses)
AA_WEIGHTS_PROTEIN_MONO = {
  "A": 71.0371138050,  # Alanine
  "R": 156.101111050,  # Arginine
  "N": 114.042927470,  # Asparagine
  "D": 115.026943065,  # Aspartic acid
  "C": 103.009184505,  # Cysteine
  "E": 129.042593135,  # Glutamic acid
  "Q": 128.058577540,  # Glutamine
  "G": 57.0214637350,  # Glycine
  "H": 137.058911875,  # Histidine
  "I": 113.084064015,  # Isoleucine
  "L": 113.084064015,  # Leucine
  "K": 128.094963050,  # Lysine
  "M": 131.040484645,  # Methionine
  "F": 147.068413945,  # Phenylalanine
  "P": 97.0527638750,  # Proline
  "S": 87.0320284350,  # Serine
  "T": 101.047678505,  # Threonine
  "W": 186.079312980,  # Tryptophan
  "Y": 163.063328575,  # Tyrosine
  "V": 99.0684139450,  # Valine
  "O": 237.147726925,  # pyrrolysine
  "U": 150.953633405,  # selenocysteine
}

# Average molecular weights (in Daltons) of *free amino acids* (not incorporated into a chain).
# These values include the full amino acid with terminal H and OH groups, i.e. before peptide bond formation.
# Often used for small-molecule calculations or educational purposes, but not for intact peptides/proteins.
AA_WEIGHTS_FREE = {
  "A": 89.090,  # Alanine
  "R": 174.20,  # Arginine
  "N": 132.12,  # Asparagine
  "D": 133.10,  # Aspartic acid
  "C": 121.15,  # Cysteine
  "E": 147.13,  # Glutamic acid
  "Q": 146.15,  # Glutamine
  "G": 75.070,  # Glycine
  "H": 155.16,  # Histidine
  "I": 131.17,  # Isoleucine
  "L": 131.17,  # Leucine
  "K": 146.19,  # Lysine
  "M": 149.21,  # Methionine
  "F": 165.19,  # Phenylalanine
  "P": 115.13,  # Proline
  "S": 105.09,  # Serine
  "T": 119.12,  # Threonine
  "W": 204.23,  # Tryptophan
  "Y": 181.19,  # Tyrosine
  "V": 117.15,  # Valine
  "O": 255.31,  # Pyrrolysine (free)
  "U": 168.06,  # Selenocysteine (free)
}

## pKa Values
# Default pKa set (EMBOSS-like). Values are typical textbook approximations.
# You can swap these for another set (e.g., Bjellqvist, IPC) if desired.
DEFAULT_PKA = {
  # termini
  "N_TERMINUS": 8.6,
  "C_TERMINUS": 3.6,
  # acidic side chains (deprotonate to -1)
  "C": 8.50,  # Cys
  "D": 3.90,  # Asp
  "E": 4.10,  # Glu
  "Y": 10.1,  # Tyr
  # basic side chains (protonate to +1)
  "H": 6.50,  # His
  "K": 10.8,  # Lys
  "R": 12.5,  # Arg
  # optional uncommon residue
  "U": 5.20,  # Selenocysteine (approx.; behaves like acidic thiol/selenol)
}
