"""Sequence- and amino-acid-related constants."""

from typing import List, Optional

from neurosnap._compat import compat_dataclass

## Amino Acid Codes and Properties
# Codes for standard amino acids
STANDARD_AAs = set("ACDEFGHIKLMNPQRSTVWY")


# Amino acid Record class
@compat_dataclass(frozen=True, slots=True)
class AARecord:
  code: Optional[str]  # 1-letter code; None for if unavailable
  abr: str  # 3-letter abbreviation or CCD code
  name: str  # full name (upper-cased)
  standard_equiv_abr: Optional[str]  # Standard parent residue when one exists (e.g., "HIS" for HID or "LYS" for PYL); otherwise None.


class AARecordTable:
  """Indexed lookup wrapper for a single amino-acid record table.

  The table preserves the input record order for iteration while also exposing
  direct lookups by one-letter code, three-letter/CCD abbreviation, and full
  residue name.
  """

  def __init__(self, records: List[AARecord]):
    """Build indexed lookups for a single amino-acid record table.

    Parameters:
      records: Ordered amino-acid records to include in the table.
    """
    self.records = list(records)
    self.code_to_rec = {record.code: record for record in self.records if record.code is not None}
    self.abr_to_rec = {record.abr: record for record in self.records}
    self.name_to_rec = {record.name: record for record in self.records}

  def __iter__(self):
    """Iterate records in their original table order.

    Returns:
      Iterator over :class:`AARecord` objects.
    """
    yield from self.records

  def items(self):
    """Return abbreviation-keyed record pairs.

    Returns:
      A dynamic view of ``(abr, record)`` pairs for the table.
    """
    return self.abr_to_rec.items()

  def values(self):
    """Return the records keyed by residue abbreviation.

    Returns:
      A dynamic view of the table's :class:`AARecord` objects.
    """
    return self.abr_to_rec.values()

  def get_by_code(self, code: str) -> Optional[AARecord]:
    """Look up a record by one-letter amino-acid code.

    Parameters:
      code: One-letter residue code.

    Returns:
      Matching :class:`AARecord`, or ``None`` when the code is absent.
    """
    query = str(code).strip().upper()
    if len(query) != 1:
      return None
    return self.code_to_rec.get(query)

  def get_by_abr(self, abr: str) -> Optional[AARecord]:
    """Look up a record by residue abbreviation.

    Parameters:
      abr: Three-letter or CCD-style residue abbreviation.

    Returns:
      Matching :class:`AARecord`, or ``None`` when the abbreviation is absent.
    """
    return self.abr_to_rec.get(str(abr).strip().upper())

  def get_by_name(self, name: str) -> Optional[AARecord]:
    """Look up a record by full residue name.

    Parameters:
      name: Full residue name.

    Returns:
      Matching :class:`AARecord`, or ``None`` when the name is absent.
    """
    return self.name_to_rec.get(str(name).strip().upper())

  def get_canonical_record(self, record_or_abr: str | AARecord) -> Optional[AARecord]:
    """Resolve a record or abbreviation to its canonical-table record.

    Parameters:
      record_or_abr: Either an :class:`AARecord` instance already associated
        with this table family, or a residue abbreviation to resolve against
        the table.

    Returns:
      The canonical-table :class:`AARecord` when one can be determined. If the
      supplied record is already present in this table, that record is returned
      unchanged. Returns ``None`` when the residue is unknown to the table or
      does not declare a standard parent that exists in this table.
    """
    record = record_or_abr if isinstance(record_or_abr, AARecord) else self.get_by_abr(record_or_abr)
    if record is None:
      return None
    if self.get_by_abr(record.abr) is not None:
      return record
    if record.standard_equiv_abr is None:
      return None
    return self.get_by_abr(record.standard_equiv_abr)

  def get_standard_record(self, record_or_abr: str | AARecord) -> Optional[AARecord]:
    """Resolve a record or abbreviation to its standard parent record.

    Parameters:
      record_or_abr: Either an :class:`AARecord` instance or a residue
        abbreviation to resolve.

    Returns:
      The standard-parent :class:`AARecord` when one can be determined.
      Returns the input record unchanged when it is already a standard amino
      acid. Returns ``None`` when the residue is unknown or has no standard
      parent mapping.
    """
    record = record_or_abr if isinstance(record_or_abr, AARecord) else self.get_by_abr(record_or_abr)
    if record is None:
      return None
    if self.get_by_abr(record.abr) is not None and record.standard_equiv_abr is None:
      return record
    if record.standard_equiv_abr is None:
      return None
    return self.get_by_abr(record.standard_equiv_abr)


## Canonical / standard amino acids keyed by ABR.
# This is the primary amino-acid reference table used for one-letter-code
# lookup, canonical parent resolution, and sequence-facing normalization.
# It contains the standard residues plus pyrrolysine (PYL) and
# selenocysteine (SEC), which are treated as first-class canonical records here
# because they have stable one-letter codes even though they still map to
# conventional parent residues through ``standard_equiv_abr``.
AA_RECORDS_CANONICAL = AARecordTable([
  AARecord("A", "ALA", "ALANINE", None),
  AARecord("R", "ARG", "ARGININE", None),
  AARecord("N", "ASN", "ASPARAGINE", None),
  AARecord("D", "ASP", "ASPARTIC ACID", None),
  AARecord("C", "CYS", "CYSTEINE", None),
  AARecord("Q", "GLN", "GLUTAMINE", None),
  AARecord("E", "GLU", "GLUTAMIC ACID", None),
  AARecord("G", "GLY", "GLYCINE", None),
  AARecord("H", "HIS", "HISTIDINE", None),
  AARecord("I", "ILE", "ISOLEUCINE", None),
  AARecord("L", "LEU", "LEUCINE", None),
  AARecord("K", "LYS", "LYSINE", None),
  AARecord("M", "MET", "METHIONINE", None),
  AARecord("F", "PHE", "PHENYLALANINE", None),
  AARecord("P", "PRO", "PROLINE", None),
  AARecord("S", "SER", "SERINE", None),
  AARecord("T", "THR", "THREONINE", None),
  AARecord("W", "TRP", "TRYPTOPHAN", None),
  AARecord("Y", "TYR", "TYROSINE", None),
  AARecord("V", "VAL", "VALINE", None),
  AARecord("O", "PYL", "PYRROLYSINE", "LYS"),
  AARecord("U", "SEC", "SELENOCYSTEINE", "CYS"),
])

# Ambiguous amino-acid placeholders are tracked separately so callers can make
# an explicit decision about whether to support them. These codes do not denote
# a concrete residue identity in structure-preparation or MD workflows:
# ``ASX`` and ``GLX`` collapse acidic/amido side-chain identities, ``XLE``
# merges leucine/isoleucine, ``UNK`` means the residue identity is unknown, and
# ``TRM`` is a translation termination token rather than a physical residue.
# They are intentionally excluded from standard amino-acid lookup and from
# protein-residue classification helpers.
AA_RECORDS_AMBIGUOUS = AARecordTable([
  AARecord("B", "ASX", "ASPARAGINE/ASPARTIC ACID", "ASP"),
  AARecord("Z", "GLX", "GLUTAMINE/GLUTAMIC ACID", "GLU"),
  AARecord("J", "XLE", "LEUCINE/ISOLEUCINE", "LEU"),
  AARecord("X", "UNK", "UNKNOWN", None),
  AARecord("*", "TRM", "TERMINATION", None),
])

# Force-field and protonation-state residue names represent concrete protein
# residues that appear in prepared structures and MD/topology workflows. These
# entries should remain valid for protein-residue classification and sequence
# parent mapping, but they are not treated as canonical residue names.
AA_RECORDS_FORCEFIELD_VARIANTS = AARecordTable([
  AARecord("H", "HID", "HISTIDINE DELTA-PROTONATED", "HIS"),
  AARecord("H", "HIE", "HISTIDINE EPSILON-PROTONATED", "HIS"),
  AARecord("H", "HIP", "HISTIDINE DOUBLY PROTONATED", "HIS"),
  AARecord("H", "HSD", "HISTIDINE DELTA-PROTONATED", "HIS"),
  AARecord("H", "HSE", "HISTIDINE EPSILON-PROTONATED", "HIS"),
  AARecord("H", "HSP", "HISTIDINE DOUBLY PROTONATED", "HIS"),
  AARecord("H", "HISD", "HISTIDINE DELTA-PROTONATED", "HIS"),
  AARecord("H", "HISE", "HISTIDINE EPSILON-PROTONATED", "HIS"),
  AARecord("H", "HISH", "HISTIDINE DOUBLY PROTONATED", "HIS"),
  AARecord("D", "ASH", "ASPARTIC ACID PROTONATED", "ASP"),
  AARecord("D", "ASPH", "ASPARTIC ACID PROTONATED", "ASP"),
  AARecord("D", "ASPP", "ASPARTIC ACID PROTONATED", "ASP"),
  AARecord("E", "GLH", "GLUTAMIC ACID PROTONATED", "GLU"),
  AARecord("E", "GLUH", "GLUTAMIC ACID PROTONATED", "GLU"),
  AARecord("E", "GLUP", "GLUTAMIC ACID PROTONATED", "GLU"),
  AARecord("C", "CYSH", "CYSTEINE THIOL", "CYS"),
  AARecord("C", "CYM", "CYSTEINE THIOLATE", "CYS"),
  AARecord("C", "CYX", "CYSTEINE DISULFIDE", "CYS"),
  AARecord("C", "CYS2", "CYSTEINE DISULFIDE", "CYS"),
  AARecord("K", "LYN", "LYSINE NEUTRAL", "LYS"),
  AARecord("K", "LYSN", "LYSINE NEUTRAL", "LYS"),
  AARecord("R", "ARN", "ARGININE NEUTRAL", "ARG"),
  AARecord("R", "ARGN", "ARGININE NEUTRAL", "ARG"),
])


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
  "AARecordTable",
  "AA_MASS_FREE",
  "AA_MASS_PROTEIN_AVG",
  "AA_MASS_PROTEIN_MONO",
  "AA_RECORDS_AMBIGUOUS",
  "AA_RECORDS_CANONICAL",
  "AA_RECORDS_FORCEFIELD_VARIANTS",
  "DEFAULT_PKA",
  "STANDARD_AAs",
]
