"""Python-native hydrogen optimization tables."""

# Generated from legacy vendored PDB2PQR data files.
# Keep values identical; only the on-disk representation has changed.

HYDROGEN_DATA = {
  "HIS": {
    "name": "HIS",
    "opttype": "Flip",
    "optangle": "CA CB CG ND1",
    "atoms": [{"name": "HD1"}, {"name": "HD2"}, {"name": "HE1"}, {"name": "HE2"}],
  },
  "ASN": {"name": "ASN", "opttype": "Flip", "optangle": "CA CB CG OD1", "atoms": [{"name": "HD21"}, {"name": "HD22"}]},
  "GLN": {"name": "GLN", "opttype": "Flip", "optangle": "CB CG CD OE1", "atoms": [{"name": "HE21"}, {"name": "HE22"}]},
  "ASH": {
    "name": "ASH",
    "opttype": "Carboxylic",
    "optangle": "CA CB CG OD1",
    "atoms": [{"name": "HD2", "bond": "OD2"}, {"name": "HD1", "bond": "OD1"}],
  },
  "GLH": {
    "name": "GLH",
    "opttype": "Carboxylic",
    "optangle": "CB CG CD OE1",
    "atoms": [{"name": "HE2", "bond": "OE2"}, {"name": "HE1", "bond": "OE1"}],
  },
  "CYS": {"name": "CYS", "opttype": "Alcoholic", "optangle": "", "atoms": [{"name": "HG"}]},
  "SER": {"name": "SER", "opttype": "Alcoholic", "optangle": "", "atoms": [{"name": "HG"}]},
  "THR": {"name": "THR", "opttype": "Alcoholic", "optangle": "", "atoms": [{"name": "HG1"}]},
  "TYR": {"name": "TYR", "opttype": "Alcoholic", "optangle": "", "atoms": [{"name": "HH"}]},
  "WAT": {"name": "WAT", "opttype": "Water", "optangle": "", "atoms": []},
  "ARG": {"name": "ARG", "opttype": "Generic", "optangle": "", "atoms": []},
  "LYS": {"name": "LYS", "opttype": "Generic", "optangle": "", "atoms": []},
  "NTR": {"name": "NTR", "opttype": "Generic", "optangle": "", "atoms": []},
  "CTR": {"name": "CTR", "opttype": "Carboxylic", "optangle": "", "atoms": []},
}
