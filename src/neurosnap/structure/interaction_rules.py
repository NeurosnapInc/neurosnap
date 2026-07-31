"""Residue and atom tables backing the interaction detection rules.

Metal identity is not defined here: coordination centres and covalent-candidate
exclusions both draw on :data:`neurosnap.constants.chemistry.METAL_ELEMENTS` so
there is a single list of metals in the package.
"""

RULE_SET = "default"
RULE_VERSION = "1"

# Canonical amino acids
CANONICAL_AMINO_ACIDS = {
  "ALA",
  "ARG",
  "ASN",
  "ASP",
  "CYS",
  "CYX",
  "GLN",
  "GLU",
  "GLY",
  "HIS",
  "ILE",
  "LEU",
  "LYS",
  "MET",
  "PHE",
  "PRO",
  "SER",
  "THR",
  "TRP",
  "TYR",
  "VAL",
}

# Hydrogen bond donor/acceptor definitions for canonical protein residues
PROTEIN_SIDECHAIN_DONORS = {
  "ARG": {"NE", "NH1", "NH2"},
  "ASN": {"ND2"},
  "GLN": {"NE2"},
  "HIS": {"ND1", "NE2"},
  "LYS": {"NZ"},
  "SER": {"OG"},
  "THR": {"OG1"},
  "TRP": {"NE1"},
  "TYR": {"OH"},
}

PROTEIN_SIDECHAIN_ACCEPTORS = {
  "ASN": {"OD1"},
  "ASP": {"OD1", "OD2"},
  "GLN": {"OE1"},
  "GLU": {"OE1", "OE2"},
  "HIS": {"ND1", "NE2"},
  "SER": {"OG"},
  "THR": {"OG1"},
  "TYR": {"OH"},
}

# Ionic rules for canonical proteins
PROTEIN_IONIC_POSITIVE = {
  "LYS": {"NZ"},
  "ARG": {"NH1", "NH2"},
}

PROTEIN_IONIC_NEGATIVE = {
  "ASP": {"OD1", "OD2"},
  "GLU": {"OE1", "OE2"},
}

#: Elements accepted as coordinating donors around a metal centre.
#:
#: Oxygen, nitrogen, and sulfur cover the protein and nucleic-acid donors, and
#: the halides are included because they are common inorganic ligands. Insulin
#: zinc sites, for example, are chloride-coordinated.
METAL_DONOR_ELEMENTS = {"O", "N", "S", "F", "CL", "BR", "I"}
