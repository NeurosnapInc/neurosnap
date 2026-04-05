"""Structure- and residue-level constants."""

BACKBONE_ATOMS_AA = {"N", "CA", "C"}
BACKBONE_ATOMS_DNA = {
  "P",
  "O1P",
  "O2P",
  "OP1",
  "OP2",
  "O3'",
  "O5'",
  "C3'",
  "C4'",
  "C5'",
  "O4'",
  "C1'",
  "C2'",
}
BACKBONE_ATOMS_RNA = BACKBONE_ATOMS_DNA.union({"O2'"})

NUC_DNA_CODES = {"DA", "DT", "DC", "DG"}
NUC_RNA_CODES = {"A", "U", "C", "G"}
STANDARD_NUCLEOTIDES = NUC_DNA_CODES.union(NUC_RNA_CODES)

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
