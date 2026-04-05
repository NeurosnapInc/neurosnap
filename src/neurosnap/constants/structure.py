"""Structure- and residue-level constants."""

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

# List of hydrophobic residues
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

__all__ = [
  "BACKBONE_ATOMS_AA",
  "BACKBONE_ATOMS_DNA",
  "BACKBONE_ATOMS_RNA",
  "HYDROPHOBIC_RESIDUES",
  "NUC_DNA_CODES",
  "NUC_RNA_CODES",
  "STANDARD_NUCLEOTIDES",
]
