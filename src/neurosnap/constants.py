"""
This file contains constants.
"""

# Names of atoms that are part of a protein's backbone structure
BACKBONE_ATOMS = {"N", "CA", "C"}
# Codes for standard nucleotides (both RNA and DNA)
STANDARD_NUCLEOTIDES = {"A", "T", "C", "G", "U", "DA", "DT", "DC", "DG", "DU"}
# Codes for standard amino acids
STANDARD_AAs = set("ACDEFGHIKLMNPQRSTVWY")
# Maps non-standard amino acids to equivalent standard amino acids (if possible)
NON_STANDARD_AAs_TO_STANDARD_AAs = {
  "O": "K",  # Pyrrolysine → closest to Lysine
  "U": "C",  # Selenocysteine → closest to Cysteine
  "B": "D",  # Asparagine/Aspartic Acid → map to Aspartic Acid
  "Z": "E",  # Glutamine/Glutamic Acid → map to Glutamic Acid
  "J": "L",  # Leucine/Isoleucine → map to Leucine
}
# List of hydrophobic residues
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

## Full amino acids table
AAs_FULL_TABLE = [
  ["A", "ALA", "ALANINE"],
  ["R", "ARG", "ARGININE"],
  ["N", "ASN", "ASPARAGINE"],
  ["D", "ASP", "ASPARTIC ACID"],
  ["C", "CYS", "CYSTEINE"],
  ["Q", "GLN", "GLUTAMINE"],
  ["E", "GLU", "GLUTAMIC ACID"],
  ["G", "GLY", "GLYCINE"],
  ["H", "HIS", "HISTIDINE"],
  ["I", "ILE", "ISOLEUCINE"],
  ["L", "LEU", "LEUCINE"],
  ["K", "LYS", "LYSINE"],
  ["M", "MET", "METHIONINE"],
  ["F", "PHE", "PHENYLALANINE"],
  ["P", "PRO", "PROLINE"],
  ["S", "SER", "SERINE"],
  ["T", "THR", "THREONINE"],
  ["W", "TRP", "TRYPTOPHAN"],
  ["Y", "TYR", "TYROSINE"],
  ["V", "VAL", "VALINE"],
  # NON-STANDARD AMINO ACIDS
  ["O", "PYL", "PYRROLYSINE"],
  ["U", "SEC", "SELENOCYSTEINE"],
  ["B", "ASX", "ASPARAGINE/ASPARTIC ACID"],
  ["Z", "GLX", "GLUTAMINE/GLUTAMIC ACID"],
  ["J", "XLE", "LEUCINE/ISOLEUCINE"],
  ["X", "UNK", "UNKNOWN"],
  ["*", "TRM", "TERMINATION"],
]
AA_CODE_TO_ABR = {}
AA_CODE_TO_NAME = {}
AA_ABR_TO_CODE = {}
AA_ABR_TO_NAME = {}
AA_NAME_TO_CODE = {}
AA_NAME_TO_ABR = {}
for code, abr, name in AAs_FULL_TABLE:
  AA_CODE_TO_ABR[code] = abr
  AA_CODE_TO_NAME[code] = name
  AA_ABR_TO_CODE[abr] = code
  AA_ABR_TO_NAME[abr] = name
  AA_NAME_TO_ABR[name] = abr
  AA_NAME_TO_CODE[name] = code
