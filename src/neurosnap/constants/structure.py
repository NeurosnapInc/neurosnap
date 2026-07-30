"""Structure- and residue-level constants."""

### Nucleotide Constants
# 5' terminal atoms
FIVE_PRIME_TERMINAL_ATOMS = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}
# 3' terminal atoms
THREE_PRIME_TERMINAL_ATOMS = {"O3P", "OP3"}

## Backbone Atoms
# Names of atoms that are part of a DNA backbone structure
BACKBONE_ATOMS_DNA = (
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
)
# Names of atoms that are part of an RNA backbone structure
# (Same as DNA but includes the 2'-OH group)
BACKBONE_ATOMS_RNA = BACKBONE_ATOMS_DNA + ("O2'",)
# Backbone atoms for nucleic acids across the common DNA/RNA naming variants
# used in structure parsing and topology lookup. This includes OP3, which is a
# terminal phosphate atom name encountered in some files.
BACKBONE_ATOMS_NUCLEOTIDE = BACKBONE_ATOMS_RNA + ("OP3",)

## Nucleotide Codes
# Single-letter PDB residue codes for standard DNA residues
NA_DNA_CODES = {"DA", "DT", "DC", "DG", "DI"}
# Single-letter PDB residue codes for standard RNA residues
NA_RNA_CODES = {"A", "U", "C", "G", "I"}
# Codes for standard nucleotides (both RNA and DNA)
STANDARD_NUCLEOTIDES = NA_DNA_CODES.union(NA_RNA_CODES)
# Canonical CHARMM residue names for standard nucleic acid bases.
NA_CHARMM_RESIDUES = {"ADE", "GUA", "CYT", "THY", "URA"} # TODO: Kinda redundant with NA_RESIDUE_MAP.values()
# Combined set of standard nucleic-acid residue names across common PDB and
# CHARMM conventions.
NA_ALL_CODES = STANDARD_NUCLEOTIDES.union(NA_CHARMM_RESIDUES)
# Map common PDB nucleic acid residue names to their corresponding CHARMM residue names for topology and parameter lookup.
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


### Protein Constants
## Backbone Atoms
# Names of atoms that are part of a protein's backbone structure
BACKBONE_ATOMS_AA = ("N", "CA", "C")

# List of hydrophobic residues
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

__all__ = [
  "BACKBONE_ATOMS_AA",
  "BACKBONE_ATOMS_DNA",
  "BACKBONE_ATOMS_NUCLEOTIDE",
  "BACKBONE_ATOMS_RNA",
  "HYDROPHOBIC_RESIDUES",
  "NA_ALL_CODES",
  "NA_CHARMM_RESIDUES",
  "NA_DNA_CODES",
  "NA_RNA_CODES",
  "NA_RESIDUE_MAP",
  "STANDARD_NUCLEOTIDES",
  "FIVE_PRIME_TERMINAL_ATOMS",
  "THREE_PRIME_TERMINAL_ATOMS",
]
