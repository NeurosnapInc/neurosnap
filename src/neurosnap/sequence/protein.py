import re
from collections import Counter
from types import Dict

from neurosnap.constants import (
  AA_ALIASES,
  AA_MASS_PROTEIN_AVG,
  AA_RECORDS,
  DEFAULT_PKA,
  AARecord,
  STANDARD_AAs,
)


def getAA(query: str, *, non_standard: str = "reject") -> AARecord:
  """Resolve an amino acid identifier to a canonical record.

  This function accepts either a **1-letter code**, **3-letter abbreviation**,
  or **full name** (case-insensitive) and returns the corresponding `AARecord`.

  Parameters
  ----------
  query : str
      Amino acid identifier (1-letter code, 3-letter CCD abbreviation,
      or full name).
  non_standard : {"reject", "convert", "allow"}, optional
      Policy for handling non-standard amino acids (default: "reject"):

      - "reject": Raise an error if the amino acid is non-standard.
      - "convert": Map non-standard amino acids to their closest
        standard equivalent (e.g., MSE → MET).
      - "allow": Return the non-standard amino acid unchanged.

  Returns
  -------
  AARecord
      A record containing:
      - `code`: 1-letter code (may be "?" if unavailable for non-standard AAs).
      - `abr`: 3-letter abbreviation.
      - `name`: Full amino acid name.
      - `is_standard`: Whether the residue is one of the 20 canonical amino acids.
      - `standard_equiv_abr`: 3-letter abbreviation of the standard equivalent
        (if applicable).

  Raises
  ------
  ValueError
      If `query` does not match any supported amino acid identifier.
      If `non_standard="reject"` and the amino acid is non-standard.
      If `non_standard="convert"` but no standard equivalent is defined.
  """
  query = query.upper()
  try:
    abr = AA_ALIASES[query]
    rec = AA_RECORDS[abr]
  except KeyError:
    raise ValueError(f"Unknown amino acid identifier: '{query}'. Expected a 1-letter code, 3-letter code, or full name.")

  if not rec.is_standard:
    if non_standard == "reject":
      raise ValueError(
        f"Encountered non-standard amino acid '{rec.abr}' ({rec.name}). "
        "To handle these, set `non_standard='allow'` to keep them "
        "or `non_standard='convert'` to map them to a standard equivalent."
      )
    elif non_standard == "convert":
      if not rec.standard_equiv_abr:
        raise ValueError(f"Non-standard amino acid '{rec.abr}' ({rec.name}) does not have a standard equivalent and cannot be converted.")
      rec = AA_RECORDS[rec.standard_equiv_abr]
  return rec


def sanitize_aa_seq(seq: str, *, non_standard: str = "reject", trim_term: bool = True, uppercase=True, clean_whitespace: bool = True) -> str:
  """
  Validates and sanitizes an amino acid sequence string.

  Parameters:
      seq: The input amino acid sequence.
      non_standard: How to handle non-standard amino acids.
          Must be one of:
          - "reject": Raise an error if any non-standard residue is found (default).
          - "convert": Replace non-standard residues with standard equivalents, if possible.
          - "allow": Keep non-standard residues unchanged.
      trim_term: If True, trims terminal stop codons ("*") from the end of the sequence. Default is True.
      uppercase: If True, converts the sequence to uppercase before processing. Default is True.
      clean_whitespace: If True, removes all whitespace characters from the sequence. Default is True.

  Returns:
      The sanitized amino acid sequence.

  Raises:
      ValueError: If an invalid residue is found and `non_standard` is set to "reject",
                  or if a residue cannot be converted when `non_standard` is "convert".
      AssertionError: If `non_standard` is not one of "allow", "convert", or "reject".
  """
  assert non_standard in ("allow", "convert", "reject"), f'Unknown value of "{non_standard}" supplied for non_standard parameter.'

  if uppercase:
    seq = seq.upper()

  if clean_whitespace:
    seq = re.sub(r"\s", "", seq)

  if trim_term:
    seq = seq.rstrip("*")

  new_seq = ""
  for i, x in enumerate(seq, start=1):
    if x not in STANDARD_AAs:
      if non_standard == "allow":
        pass
      elif non_standard == "convert":
        x = getAA(x, non_standard="convert").code
      else:
        raise ValueError(f'Invalid amino acid "{x}" specified at position {i}.')
    new_seq += x
  return new_seq


def molecular_weight(sequence: str, aa_mws: Dict[str, float] = AA_MASS_PROTEIN_AVG) -> float:
  """
  Calculate the molecular weight of a protein or peptide sequence.

  This function computes the molecular weight by summing the residue
  masses for each amino acid in the input sequence. By default, it uses
  average amino acid residue masses (`AA_MASS_PROTEIN_AVG`), but you
  can provide a custom mass dictionary (e.g., monoisotopic or free amino
  acid masses).

  The calculation accounts for the loss of one water molecule (H₂O,
  18.015 Da) for each peptide bond formed. For a sequence of length n,
  (n - 1) * 18.015 Da is subtracted from the total.

  Args:
      sequence: Amino acid sequence (one-letter codes).
      aa_mws: Dictionary mapping amino acid one-letter codes to molecular
          weights. Defaults to `AA_MASS_PROTEIN_AVG`.

  Returns:
      Estimated molecular weight of the protein or peptide in Daltons (Da).

  Raises:
      ValueError: If the sequence contains an invalid or unsupported
      amino acid code.

  Notes:
      - Use `AA_MASS_PROTEIN_MONO` for monoisotopic mass calculations,
        typically used in mass spectrometry.
      - Use `AA_MASS_PROTEIN_AVG` (default) for average residue masses,
        appropriate for bulk molecular weight estimation.
      - For free amino acids (not incorporated in peptides), use
        `AA_MASS_FREE`.
      - Weight dictionaries are defined in `constants.py`.
  """
  # Remove whitespace and convert to uppercase
  sequence = sequence.strip().upper()

  # Sum molecular weights
  weight = 0.0
  for aa in sequence:
    if aa not in aa_mws:
      raise ValueError(f"Invalid amino acid: {aa}")
    weight += aa_mws[aa]

  # Adjust for water loss during peptide bond formation
  # Each peptide bond loses one H2O (18.015 Da)
  if len(sequence) > 1:
    weight -= (len(sequence) - 1) * 18.015

  return weight


def _fraction_protonated_basic(pH: float, pKa: float) -> float:
  """For BH+ <-> B + H+, returns fraction in the protonated (+1) form."""
  return 1.0 / (1.0 + 10.0 ** (pH - pKa))


def _fraction_deprotonated_acidic(pH: float, pKa: float) -> float:
  """For HA <-> A- + H+, returns fraction in the deprotonated (-1) form."""
  return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def net_charge(sequence: str, pH: float, pKa: Dict[str, float] = DEFAULT_PKA) -> float:
  """
  Calculate the net charge of a protein or peptide sequence at a given pH.

  This function applies the Henderson–Hasselbalch equation to estimate
  the protonation state of titratable groups (N-terminus, C-terminus,
  and ionizable side chains) and computes the overall net charge.

  Args:
      sequence: Amino acid sequence in one-letter codes. Supports the 20
          canonical residues and optionally 'U' (selenocysteine).
          Non-ionizable residues are ignored.
      pH: The solution pH at which to evaluate the net charge.
      pKa: Dictionary of pKa values for titratable groups. Must include
          keys "N_TERMINUS", "C_TERMINUS", "D", "E", "C", "Y",
          "H", "K", and "R". If 'U' is present in the sequence, it
          should also include "U".

  Returns:
      Estimated net charge of the sequence at the given pH.

  Notes:
      Positive charges come from protonated groups:
      - N-terminus
      - Lysine (K)
      - Arginine (R)
      - Histidine (H)

      Negative charges come from deprotonated groups:
      - C-terminus
      - Aspartic acid (D)
      - Glutamic acid (E)
      - Cysteine (C)
      - Tyrosine (Y)
      - Selenocysteine (U), if included

      The calculation assumes independent ionization equilibria and does
      not account for local environment or structural effects. It is best
      interpreted as an approximate charge profile.
  """
  seq = sequence.strip().upper()
  if not seq:
    return 0.0

  counts = Counter(seq)

  # N-terminus (+1 when protonated)
  nterm = _fraction_protonated_basic(pH, pKa["N_TERMINUS"])

  # C-terminus (-1 when deprotonated)
  cterm = _fraction_deprotonated_acidic(pH, pKa["C_TERMINUS"])

  # Side chains
  pos = (
    counts.get("K", 0) * _fraction_protonated_basic(pH, pKa["K"])
    + counts.get("R", 0) * _fraction_protonated_basic(pH, pKa["R"])
    + counts.get("H", 0) * _fraction_protonated_basic(pH, pKa["H"])
  )

  neg = (
    counts.get("D", 0) * _fraction_deprotonated_acidic(pH, pKa["D"])
    + counts.get("E", 0) * _fraction_deprotonated_acidic(pH, pKa["E"])
    + counts.get("C", 0) * _fraction_deprotonated_acidic(pH, pKa["C"])
    + counts.get("Y", 0) * _fraction_deprotonated_acidic(pH, pKa["Y"])
    + counts.get("U", 0) * _fraction_deprotonated_acidic(pH, pKa["U"])  # optional
  )
  return (nterm + pos) - (cterm + neg)


def isoelectric_point(
  sequence: str, pKa: Dict[str, float] = DEFAULT_PKA, *, pH_low: float = 0.0, pH_high: float = 14.0, tol: float = 1e-4, max_iter: int = 100
) -> float:
  """
  Estimate the isoelectric point (pI) of a protein or peptide.

  The pI is the pH at which the net charge of the molecule is zero.
  This function computes the net charge across pH and uses a bisection
  search to find the root.

  Args:
      sequence: Amino acid sequence (one-letter codes). Supports the 20
          canonical residues and optionally 'U' (selenocysteine).
          Non-titratable residues contribute no charge.
      pKa: Dictionary of pKa values for titratable groups. Must include
          keys "N_TERMINUS", "C_TERMINUS", and for side chains
          "D", "E", "C", "Y", "H", "K", "R". If 'U' appears in the
          sequence, include "U" (default ~5.2, approximate).
      pH_low: Lower bound of the bracketing interval for the bisection
          search (default 0.0).
      pH_high: Upper bound of the bracketing interval for the bisection
          search (default 14.0).
      tol: Target absolute net charge tolerance at the solution
          (default 1e-4).
      max_iter: Maximum iterations for the bisection search (default 100).

  Returns:
      Estimated pI.

  Notes:
      - Results depend on the chosen pKa set. For consistency with common
        tools, you may substitute a different pKa dictionary
        (e.g., Bjellqvist or IPC sets).
      - Pyrrolysine ('O') is not included by default due to scarce
        consensus pKa data; it is treated as non-titratable here.
        You can add an entry if you have a value.
      - This model ignores sequence-context and microenvironment effects
        (local shifts in pKa due to neighbors or structure). It’s a good
        heuristic, not a guarantee.
  """
  seq = sequence.strip().upper()
  if not seq:
    return 0.0

  # Validate sequence characters
  valid = STANDARD_AAs | {"U", "O"}
  for aa in seq:
    if aa not in valid:
      raise ValueError(f"Invalid amino acid: {aa}")

  # Bisection search
  lo, hi = pH_low, pH_high
  q_lo = net_charge(seq, lo, pKa)
  q_hi = net_charge(seq, hi, pKa)

  # If the bracket doesn't change sign, still proceed but clamp toward the side
  # where |charge| is smaller to avoid errors on unusual sequences.
  if q_lo == 0:
    return lo
  if q_hi == 0:
    return hi
  if q_lo * q_hi > 0:
    # No sign change; do a guarded search by nudging bounds inward.
    # This keeps the function robust for edge cases (e.g., extremely acidic/basic sequences).
    for _ in range(20):
      mid = (lo + hi) / 2.0
      q_mid = net_charge(seq, mid, pKa)
      if abs(q_mid) < abs(q_lo) and abs(q_mid) < abs(q_hi):
        # Use the best we have if we can't bracket
        best = mid
      lo += 0.1
      hi -= 0.1
    return best if "best" in locals() else (lo + hi) / 2.0

  for _ in range(max_iter):
    mid = (lo + hi) / 2.0
    q_mid = net_charge(seq, mid, pKa)

    if abs(q_mid) <= tol:
      return mid
    # Decide which subinterval keeps the sign change
    if q_lo * q_mid < 0:
      hi, q_hi = mid, q_mid
    else:
      lo, q_lo = mid, q_mid

  # Fallback if not converged within max_iter
  return (lo + hi) / 2.0
