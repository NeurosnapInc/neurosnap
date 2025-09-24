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