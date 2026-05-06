from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from neurosnap._compat import compat_dataclass

# Keep the exported feature order identical to the bundled model files.
STANDARD_AMINO_ACIDS = tuple("ARNDCQEGHILKMFPSTWYV")
FEATURE_NAMES = [
  "Acont",
  "Rcont",
  "Ncont",
  "Dcont",
  "Ccont",
  "Qcont",
  "Econt",
  "Gcont",
  "Hcont",
  "Icont",
  "Lcont",
  "Kcont",
  "Mcont",
  "Fcont",
  "Pcont",
  "Scont",
  "Tcont",
  "Wcont",
  "Ycont",
  "Vcont",
  "length",
  "act",
  "alm",
  "averageNegativeCharge0_24",
  "bac",
  "caa",
  "dna",
  "erl",
  "erm",
  "gpi",
  "gvh",
  "leu",
  "m1a",
  "m1b",
  "m2",
  "m3a",
  "m3b",
  "mNt",
  "m_",
  "maxHydropathy0_29_12",
  "maxHydropathy5_29_11",
  "maxNegativeCharge0_19_12",
  "mip",
  "mit",
  "myr",
  "nuc",
  "pox",
  "psg",
  "px2",
  "rib",
  "rnp",
  "tms",
  "top",
  "tyr",
  "vac",
  "yqr",
]

# These index tables are lifted from the original Perl modules so the Python
# port produces feature values on the same scale as the reference pipeline.
AMINO_ACID_INDEXES: Dict[str, Dict[str, float]] = {
  "Engelman GES 1986": {
    "A": -6.7,
    "R": 51.5,
    "N": 20.1,
    "D": 38.5,
    "C": -8.4,
    "Q": 17.2,
    "E": 34.3,
    "G": -4.2,
    "H": 12.6,
    "I": -13.0,
    "L": -11.7,
    "K": 36.8,
    "M": -14.2,
    "F": -15.5,
    "P": 0.8,
    "S": -2.5,
    "T": -5.0,
    "W": -7.9,
    "Y": 2.9,
    "V": -10.9,
    "B": 30.5,
    "Z": 27.8,
    "X": 5.8,
  },
  "KYTJ820101": {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "B": -3.5,
    "Z": -3.5,
    "X": -0.19,
  },
  "KLEP840101": {
    "A": 0.0,
    "R": 1.0,
    "N": 0.0,
    "D": -1.0,
    "C": 0.0,
    "Q": 0.0,
    "E": -1.0,
    "G": 0.0,
    "H": 0.0,
    "I": 0.0,
    "L": 0.0,
    "K": 1.0,
    "M": 0.0,
    "F": 0.0,
    "P": 0.0,
    "S": 0.0,
    "T": 0.0,
    "W": 0.0,
    "Y": 0.0,
    "V": 0.0,
    "B": -0.56,
    "Z": -0.62,
    "X": -0.01,
  },
}

AA_GVH_INDEX = {
  "A": 0,
  "C": 1,
  "D": 2,
  "E": 3,
  "F": 4,
  "G": 5,
  "H": 6,
  "I": 7,
  "K": 8,
  "L": 9,
  "M": 10,
  "N": 11,
  "P": 12,
  "Q": 13,
  "R": 14,
  "S": 15,
  "T": 16,
  "V": 17,
  "W": 18,
  "Y": 19,
}

GVH_PRK = (
  (1.14, 0.92, 0.92, 1.03, 0.63, 0.78, 0.45, 0.63, 0.78, 0.78, 2.01, -0.47, 2.27, 1.73, 0.22),
  (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -3.58, 0.00, -3.58, 0.00, 0.00),
  (-0.69, -0.69, -0.69, -0.69, -0.69, -0.69, -0.69, -0.69, -0.69, -0.69, -3.58, -0.69, -3.58, 0.00, 1.39),
  (-0.79, -0.79, -0.79, -0.79, -0.79, -0.79, -0.79, -0.79, -0.79, -0.79, -3.58, -0.79, -3.58, 0.60, 1.29),
  (0.43, 1.12, 0.84, 1.12, -0.26, -0.26, 1.82, -0.26, 1.12, -0.26, -3.58, 1.68, -3.58, -0.26, -0.26),
  (0.39, -0.30, -0.30, -0.30, 0.11, 0.62, -0.30, 0.39, -0.30, -0.30, -3.58, -0.30, -0.30, -0.99, -0.99),
  (0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, -3.58, 2.17, -3.58, 0.22, 0.22),
  (0.57, -0.53, 1.08, -0.53, 1.08, -0.53, -0.53, 0.57, -0.53, -0.53, -3.58, -0.53, -3.58, -0.53, 0.16),
  (-0.92, -0.92, -0.92, -0.92, -0.92, -0.92, -0.92, -0.92, -0.92, -0.92, -3.58, -0.22, -3.58, 0.18, -0.92),
  (1.09, 1.40, 1.20, 1.09, 1.20, 1.57, -0.99, -0.99, -0.30, -0.30, -0.99, -0.30, -3.58, -0.99, -0.99),
  (0.51, 1.20, 0.51, 0.51, 1.61, 1.20, 1.61, 0.51, 0.51, 1.20, -3.58, 1.90, -3.58, 0.51, 0.51),
  (-0.47, -0.47, -0.47, -0.47, -0.47, -0.47, -0.47, -0.47, -0.47, -0.47, -3.58, 0.63, -3.58, -0.47, 0.92),
  (-0.53, -0.53, -0.53, -0.53, -0.53, -0.53, 0.16, 0.57, 1.08, 0.16, -3.58, -0.53, -3.58, -0.53, 1.08),
  (-0.34, -0.34, -0.34, -0.34, -0.34, -0.34, -0.34, -0.34, 0.36, 0.36, -3.58, 0.76, -3.58, -0.34, -0.34),
  (-0.53, -0.53, -0.53, -0.53, -0.53, -0.53, -0.53, -0.53, -0.53, -0.53, -3.58, -0.53, -3.58, -0.53, -0.53),
  (-0.96, -0.96, -0.96, 0.43, 0.43, -0.96, 0.65, 1.75, 0.65, 1.12, 0.65, -0.26, -0.26, -0.96, -0.96),
  (-0.10, -0.79, 0.60, -0.10, -0.10, -0.10, -0.10, -0.10, 0.82, -0.79, 0.31, -0.79, -0.79, -0.79, -0.10),
  (0.69, 1.03, -0.92, 0.18, -0.92, 0.47, 1.03, -0.92, -0.92, 0.47, 0.18, -0.92, -3.58, -0.22, -0.92),
  (0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, -3.58, 0.92, -3.58, 0.92, 0.92),
  (-0.26, -0.26, -0.26, -0.26, -0.26, -0.26, -0.26, -0.26, -0.26, 0.84, -3.58, -0.26, -3.58, -0.26, -0.26),
)

GVH_EUK = (
  (0.10, -0.11, -0.04, 0.03, 0.32, 0.22, 0.22, 0.16, 0.54, 0.03, 1.18, -0.88, 1.71, 0.22, -0.88),
  (-0.41, 0.29, 0.69, 0.44, 0.69, 1.13, 0.29, 0.58, 0.11, 0.29, 1.44, -0.41, 0.69, 0.58, -0.41),
  (-2.19, -2.19, -2.19, -2.19, -2.19, -2.19, -2.19, -2.19, -0.58, -1.09, -5.08, -0.58, -5.08, 0.12, 0.21),
  (-2.30, -2.30, -2.30, -2.30, -2.30, -2.30, -2.30, -2.30, -1.20, -0.36, -5.08, -0.36, -5.08, 0.26, 0.34),
  (0.84, 0.47, 0.68, 0.68, 0.07, 0.22, 1.17, 0.84, -0.34, -0.11, -5.08, 0.84, -5.08, 0.07, -0.34),
  (-1.11, -1.11, -1.39, -0.70, -1.39, 0.07, -1.39, -1.80, 0.45, 1.03, -0.88, -0.55, 1.17, -0.19, -0.55),
  (-1.22, -1.22, -1.22, -1.22, -1.22, -1.22, -1.22, -1.22, 0.39, -1.22, -5.08, 0.57, -5.08, 0.16, -0.53),
  (0.71, 0.71, 0.08, -0.21, 0.40, -0.39, -0.62, 0.08, -0.39, -2.00, 0.30, -0.39, -5.08, 0.08, -0.06),
  (-2.42, -2.42, -2.42, -2.42, -2.42, -2.42, -2.42, -2.42, -2.42, -1.04, -5.08, -1.73, -5.08, -0.03, -0.23),
  (1.77, 1.73, 1.78, 1.88, 1.86, 1.31, 1.67, 1.40, -0.19, 0.64, -0.41, 0.50, -2.49, -0.41, -1.11),
  (-0.99, 0.11, 0.95, 0.39, -0.99, 0.80, -0.30, -0.30, -0.99, -0.99, -5.08, -0.99, -5.08, -0.99, -0.30),
  (-1.96, -1.96, -1.96, -1.96, -1.96, -1.96, -1.96, -1.96, -0.86, -0.86, -5.08, 0.34, -5.08, -0.57, -0.01),
  (-1.31, -2.00, -1.31, -2.00, -2.00, -0.62, -2.00, 0.08, 0.99, 0.64, -5.08, -2.00, -0.90, -2.00, 1.09),
  (-1.84, -1.84, -1.84, -1.84, -1.84, -0.05, -1.84, -1.84, 0.46, 0.24, -5.08, 1.05, -0.74, 1.10, 0.46),
  (-1.34, -2.03, -2.03, -2.03, -2.03, -2.03, -2.03, -2.03, -0.08, -0.64, -5.08, 0.68, -5.08, 0.46, 0.17),
  (-0.24, -1.34, -0.35, -0.64, 0.13, -0.13, 0.27, 0.34, 0.82, -0.04, 0.70, 0.40, 0.56, 0.27, -0.13),
  (-1.58, 0.03, -0.66, -0.89, -0.66, 0.29, -0.33, -0.33, 0.21, -0.48, 0.56, -0.19, -0.48, -1.17, 0.03),
  (0.59, 0.81, 0.30, 0.48, 0.16, 0.30, -0.01, 0.89, -2.41, 0.08, 1.06, -1.31, -5.08, -0.33, 0.43),
  (0.80, 0.51, 0.51, -0.59, -0.59, 0.11, 1.20, 0.51, -0.59, 0.51, -5.08, 1.61, -5.08, 0.11, -0.59),
  (-1.72, -1.72, -0.34, -1.72, -1.72, -1.72, -0.62, -1.72, -1.72, -1.03, -5.08, -0.11, -5.08, -1.72, 0.22),
)


def _resource_path() -> Path:
  """Return the package directory that contains bundled model resources.

  Args:
    None.

  Returns:
    Absolute path to the local WoLF PSORT package directory.
  """
  return Path(__file__).resolve().parent


@compat_dataclass(frozen=True, slots=True)
class SequenceRecord:
  """Normalized sequence input record used by the feature extractor."""

  identifier: str
  sequence: str
  label: str = "unknown"


def _normalize_sequence(identifier: str, sequence: str) -> str:
  """Normalize and validate a sequence using PSORT-compatible residue rules.

  Args:
    identifier: Sequence identifier used in validation errors.
    sequence: Raw protein sequence text.

  Returns:
    Uppercase validated sequence with PSORT-style residue substitutions applied.
  """
  seq = re.sub(r"\s+", "", sequence).upper()
  # The upstream code coerces these ambiguous or uncommon residues before
  # feature extraction, so the port must do the same for parity.
  seq = seq.replace("X", "G").replace("B", "D").replace("J", "C").replace("U", "C")
  bad = sorted(set(re.sub(r"[ABRNDCQEGHIJLKMFPSTUWXYV]", "", seq)))
  if bad:
    raise ValueError(f"{identifier} contains unsupported amino acid characters: {''.join(bad)}")
  return seq


def records_from_sequence_iterator(sequences: Iterator[Tuple[str, str]]) -> List[SequenceRecord]:
  """Build normalized sequence records from an ``(id, sequence)`` iterator.

  Args:
    sequences: Iterator yielding ``(identifier, sequence)`` tuples.

  Returns:
    List of normalized :class:`SequenceRecord` objects.
  """
  records: List[SequenceRecord] = []
  for identifier, sequence in sequences:
    normalized_identifier = str(identifier)
    records.append(SequenceRecord(normalized_identifier, _normalize_sequence(normalized_identifier, str(sequence))))
  return records


def _seq_sum(seq: Sequence[str], index_name: str, start: int = 0, length: int | None = None) -> float:
  """Sum an amino-acid index over a contiguous sequence window.

  Args:
    seq: Sequence represented as one residue per element.
    index_name: Amino-acid index table name from :data:`AMINO_ACID_INDEXES`.
    start: Zero-based start position.
    length: Optional number of residues to include. When omitted, uses the
      suffix from ``start`` to sequence end.

  Returns:
    Sum of index values over the requested window.
  """
  values = AMINO_ACID_INDEXES[index_name]
  end = len(seq) if length is None else start + length
  return sum(values[aa] for aa in seq[start:end])


def _max_window_sum(seq: Sequence[str], index_name: str, window_length: int, start: int, length: int) -> float:
  """Return the maximum sliding-window index sum over a search region.

  Args:
    seq: Sequence represented as one residue per element.
    index_name: Amino-acid index table name from :data:`AMINO_ACID_INDEXES`.
    window_length: Sliding-window length.
    start: Zero-based search start position.
    length: Number of residues in the search interval.

  Returns:
    Maximum window sum observed within the requested interval.
  """
  values = AMINO_ACID_INDEXES[index_name]
  end = start + length
  num_positions = end - start - window_length + 1
  if num_positions < 0:
    raise ValueError("Invalid window parameters")
  current = sum(values[aa] for aa in seq[start : start + window_length])
  maximum = current
  for pos in range(start, num_positions):
    current -= values[seq[pos]]
    current += values[seq[pos + window_length]]
    if current > maximum:
      maximum = current
  return maximum


class MotifLibraries:
  """Lazy loader for bundled regex-motif libraries used by WoLF PSORT."""

  def __init__(self, root: Path | None = None) -> None:
    """Initialize the motif-library loader.

    Args:
      root: Optional directory override containing ``*.json`` motif files.

    Returns:
      None. The loader is initialized in place.
    """
    self.root = root or (_resource_path() / "motif_libraries")
    self._patterns: Dict[str, List[re.Pattern[str]]] = {}

  def count_matches(self, library_name: str, sequence: str) -> int:
    """Count non-overlapping motif matches for one named library.

    Args:
      library_name: Motif-library base name without the ``.json`` suffix.
      sequence: Sequence string to scan.

    Returns:
      Total number of non-overlapping matches across every regex in the library.
    """
    if library_name not in self._patterns:
      path = self.root / f"{library_name}.json"
      payload = json.loads(path.read_text())
      patterns = [re.compile(motif["pattern"]) for motif in payload["motifs"]]
      self._patterns[library_name] = patterns
    return sum(len(pattern.findall(sequence)) for pattern in self._patterns[library_name])


class FeatureExtractor:
  """Pure Python port of the WoLF PSORT / PSORT feature generator."""

  LSEG = 17
  LSEG2 = 8
  MAX_POSITION_OF_CR_END = 10

  def __init__(self, motif_libraries: MotifLibraries | None = None) -> None:
    """Initialize the feature extractor.

    Args:
      motif_libraries: Optional preloaded motif-library helper.

    Returns:
      None. The extractor is initialized in place.
    """
    self.motif_libraries = motif_libraries or MotifLibraries()
    # These are scratch values populated by the ALOM membrane scan and then
    # consumed by the membrane-type assignment routine.
    self._last_tms_positions: Dict[int, float] = {}
    self._last_mostn = -1

  def compute(self, record: SequenceRecord) -> Dict[str, float]:
    """Compute the full WoLF PSORT feature vector for one sequence.

    Args:
      record: Normalized input sequence record.

    Returns:
      Dictionary of WoLF PSORT feature values in bundled model order.
    """
    seq = record.sequence
    if len(seq) < 30:
      raise ValueError(f"{record.identifier} is too short for WoLF PSORT: {len(seq)} aa")
    seq_list = list(seq)

    features: Dict[str, float] = {}
    standard_count = len(seq)
    counts = {aa: seq.count(aa) for aa in STANDARD_AMINO_ACIDS}
    for aa in STANDARD_AMINO_ACIDS:
      features[f"{aa}cont"] = counts[aa] / standard_count
    features["length"] = float(len(seq))

    # Recreate the original PSORT control flow: signal peptide checks feed the
    # membrane scan, which then feeds the topology-specific heuristics.
    psg = self._psg(seq_list)
    gvh, gvh_pos = self._gvh(seq_list, "eukaryote")
    if psg > 0 and gvh > -2.1:
      sig = "cleavable"
      tms, alm, npos = self._alom2(seq_list, gvh_pos + 1, 0.5, -2.0)
      middle = int(gvh_pos / 2)
    else:
      sig = "nosignal"
      tms, alm, npos = self._alom2(seq_list, 1, 0.5, -2.0)
      middle = npos + 7

    if psg <= 0 and tms == 0:
      top = 0.0
    else:
      top = self._mtop(seq_list, middle)

    mtype, start, end = self._mtype_assign(seq_list, tms, sig, top)
    gpi = 1.0 if tms > 0 and mtype == "1a" and (end - start < 10) else 0.0

    # Populate the final flat feature dictionary expected by the bundled model
    # files. The feature names and order must stay exactly aligned.
    features["act"] = float(self._actin(seq))
    features["alm"] = float(alm)
    features["averageNegativeCharge0_24"] = float(_seq_sum(seq_list, "KLEP840101", 0, 24))
    features["bac"] = float(self.motif_libraries.count_matches("bacterialDnaBinding", seq))
    features["caa"] = float(self._isoprenyl(seq))
    features["dna"] = float(self.motif_libraries.count_matches("dnaBinding", seq))
    features["erl"] = float(self._hdel(seq))
    features["erm"] = float(self._erm(seq, mtype))
    features["gpi"] = float(gpi)
    features["gvh"] = float(gvh)
    features["leu"] = float(self._dileu(seq, tms, start, end))
    for membrane_type in ("m1a", "m1b", "m2", "m3a", "m3b", "mNt", "m_"):
      features[membrane_type] = 0.0
    features[self._mtype_feature_name(mtype)] = 1.0
    features["maxHydropathy0_29_12"] = float(_max_window_sum(seq_list, "KYTJ820101", 12, 0, 29))
    features["maxHydropathy5_29_11"] = float(_max_window_sum(seq_list, "KYTJ820101", 11, 5, 24))
    features["maxNegativeCharge0_19_12"] = float(_max_window_sum(seq_list, "KLEP840101", 7, 0, 19))
    features["mip"], _ = self._gavel(seq_list)
    features["mit"] = float(self._mitdisc(seq_list))
    features["myr"] = float(self._nmyr(seq))
    features["nuc"] = float(self._nucdisc(seq_list))
    features["pox"] = float(self._pts1(seq))
    features["psg"] = float(psg)
    features["px2"] = float(self._pts2(seq))
    features["rib"] = float(self.motif_libraries.count_matches("ribosomal", seq))
    features["rnp"] = float(self._rnp1(seq))
    features["tms"] = float(tms)
    features["top"] = float(top)
    features["tyr"] = float(self._tyros(seq, tms, start, end))
    features["vac"] = float(self._vaccalc(seq))
    features["yqr"] = float(self._yqrl(seq, tms))

    return {name: features[name] for name in FEATURE_NAMES}

  def _regions(self, seq: Sequence[str]) -> Tuple[int, int, int, int]:
    """Locate the N-terminal charged and uncharged signal-peptide regions.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Tuple of charged-region length, positive-charge count, negative-charge
      count, and following uncharged-region length.
    """
    i = self.MAX_POSITION_OF_CR_END
    while i >= 0:
      aa = seq[i]
      if aa in {"R", "D", "E", "K"}:
        break
      i -= 1
    cr_end = i
    poschg = sum(1 for aa in seq[: cr_end + 1] if aa in {"R", "K"})
    negchg = sum(1 for aa in seq[: cr_end + 1] if aa in {"D", "E"})
    ur_len = 0
    for aa in seq[cr_end + 1 :]:
      if aa in {"R", "D", "E", "K"}:
        break
      ur_len += 1
    return cr_end + 1, poschg, negchg, ur_len

  def _hydseg2(self, seq: Sequence[str], seglen: int, start: int, ur_len: int) -> float:
    """Compute the best hydrophobic segment score in a constrained window.

    Args:
      seq: Sequence represented as one residue per element.
      seglen: Sliding-window length.
      start: One-based start position used by the original PSORT routine.
      ur_len: Search span length after the charged region.

    Returns:
      Maximum average hydrophobicity score across the search interval.
    """
    if start + seglen - 1 > len(seq):
      raise ValueError("Sequence too short")
    start -= 1
    search_range = 0 if ur_len - seglen < 0 else ur_len - seglen
    aas = seq[start : start + seglen + search_range]
    sum_value = 0.0
    hmax = sum_value
    ges = AMINO_ACID_INDEXES["Engelman GES 1986"]
    for i in range(1, search_range + 1):
      sum_value = sum_value + ges[aas[i - 1]] - ges[aas[i + seglen - 1]]
      if sum_value > hmax:
        hmax = sum_value
    return hmax / seglen

  def _psg(self, seq: Sequence[str]) -> float:
    """Score the McGeoch-style signal peptide heuristic.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Signal-peptide score used by the original PSORT workflow.
    """
    seglen = 8
    threshold = 4.4
    cr_len, poschg, negchg, ur_len = self._regions(seq)
    ur_len2 = 30 - cr_len if cr_len + ur_len > 30 else ur_len
    if cr_len + seglen > len(seq):
      return -10000.0
    ur_peak = self._hydseg2(seq, seglen, cr_len + 1, ur_len2)
    if ur_len >= 60 or (poschg - negchg) < 0:
      ur_peak = 0.0
    return ur_peak - threshold

  def _gvh(self, seq: Sequence[str], organism_type: str) -> Tuple[float, int]:
    """Run the von Heijne cleavage-site scoring model.

    Args:
      seq: Sequence represented as one residue per element.
      organism_type: Either ``"prokaryote"`` or ``"eukaryote"``.

    Returns:
      Tuple of GvH score and predicted cleavage position.
    """
    params = GVH_PRK if organism_type == "prokaryote" else GVH_EUK
    end = 48 if len(seq) - 15 > 48 else len(seq) - 15
    imax = -1
    smax = -1000.0
    for i in range(end + 1):
      score = 0.0
      for j in range(15):
        aa = seq[i + j]
        idx = AA_GVH_INDEX.get(aa)
        if idx is not None:
          score += params[idx][j]
      if score > smax:
        smax = score
        imax = i
    return smax - 7.5, imax + 13

  def _alom2(self, seq: Sequence[str], start: int, threshold_loose: float, threshold_strict: float) -> Tuple[int, float, int]:
    """Run the ALOM transmembrane-segment detector.

    Args:
      seq: Sequence represented as one residue per element.
      start: One-based start position for the scan.
      threshold_loose: Initial permissive threshold.
      threshold_strict: Secondary stricter threshold.

    Returns:
      Tuple of TMS count, best ALOM score, and most N-terminal TMS position.
    """
    start -= 1
    if len(seq) < 20 or start > len(seq) - self.LSEG:
      self._last_tms_positions = {}
      self._last_mostn = -1
      return -1, 999.0, -1
    hyd = [AMINO_ACID_INDEXES["KYTJ820101"][aa] for aa in seq]
    a1, a0 = -9.02, 14.27
    tms: Dict[int, float] = {}
    count = 0
    xmax = 0.0
    while True:
      # Repeatedly extract the best remaining segment, then mask it out before
      # searching again, matching the original Perl implementation.
      current = sum(hyd[start : start + self.LSEG])
      imax = start
      hmax = current
      for i in range(start + 1, len(seq) - self.LSEG + 1):
        current = current - hyd[i - 1] + hyd[i + self.LSEG - 1]
        if current > hmax:
          hmax = current
          imax = i
      hmax /= self.LSEG
      x0 = a1 * hmax + a0
      if count == 0:
        xmax = x0
      if x0 > threshold_loose:
        break
      tms[imax + 1] = x0
      for i in range(self.LSEG):
        hyd[imax + i] = -10.0
      count += 1
    if count > 0:
      count2 = sum(1 for score in tms.values() if score <= threshold_strict)
      if count2 < 2:
        tms = {pos: score for pos, score in tms.items() if score <= threshold_strict}
        count = count2
    mostn = min(tms) if tms else -1
    self._last_tms_positions = dict(tms)
    self._last_mostn = mostn
    return count, xmax, mostn

  def _mtop(self, seq: Sequence[str], center_position: int) -> float:
    """Estimate membrane topology from flank charge asymmetry.

    Args:
      seq: Sequence represented as one residue per element.
      center_position: Anchor position used for the flank charge comparison.

    Returns:
      C-terminal minus N-terminal flank charge score.
    """
    if len(seq) < 20 or center_position < 1 or center_position > len(seq) - 1:
      return -100.0
    start = center_position
    while start >= 0 and seq[start] not in {"R", "D", "E", "H", "K"}:
      start -= 1
    end = center_position
    while end < len(seq) and seq[end] not in {"R", "D", "E", "H", "K"}:
      end += 1
    flank_len = 15
    n_charge = 1.0 if start <= flank_len else 0.0
    for pos in range(start, max(-1, start - flank_len), -1):
      aa = seq[pos]
      if aa in {"D", "E"}:
        n_charge -= 1.0
      elif aa in {"R", "K"}:
        n_charge += 1.0
      elif aa == "H":
        n_charge += 0.5
    c_charge = 0.0
    for pos in range(end, min(len(seq), end + flank_len)):
      aa = seq[pos]
      if aa in {"D", "E"}:
        c_charge -= 1.0
      elif aa in {"R", "K"}:
        c_charge += 1.0
      elif aa == "H":
        c_charge += 0.5
    return c_charge - n_charge

  def _mtype_assign(self, seq: Sequence[str], tmsnum: int, sig: str, mtop: float) -> Tuple[str, int, int]:
    """Assign the discrete membrane-topology class used by WoLF PSORT.

    Args:
      seq: Sequence represented as one residue per element.
      tmsnum: Number of predicted transmembrane segments.
      sig: Signal-peptide state from the signal-peptide detectors.
      mtop: Membrane-topology charge score.

    Returns:
      Tuple of membrane-type label and the associated tail bounds.
    """
    if tmsnum == 0:
      return "__", 0, 0
    if tmsnum == 1:
      if sig == "cleavable":
        pos = min(len(seq), next(iter(self._last_tms_positions)) + self.LSEG)
        return "1a", pos, len(seq)
      mostn = self._last_mostn
      if (mostn / len(seq)) > 0.8:
        return "Nt", 1, max(0, mostn - 1)
      if mtop > 0:
        return "1b", mostn, len(seq)
      return "2 ", 1, mostn
    return ("3b", 0, 0) if mtop > 0 else ("3a", 0, 0)

  @staticmethod
  def _mtype_feature_name(mtype: str) -> str:
    """Map an internal membrane-type code to its feature name.

    Args:
      mtype: Internal membrane-type label.

    Returns:
      Feature name corresponding to the supplied membrane type.
    """
    return {
      "1a": "m1a",
      "1b": "m1b",
      "2 ": "m2",
      "3a": "m3a",
      "3b": "m3b",
      "Nt": "mNt",
      "__": "m_",
    }[mtype]

  def _getrange(self, seq: str) -> int:
    """Return the mitochondrial-presequence search range used by MITDISC.

    Args:
      seq: Sequence string.

    Returns:
      One-based end position of the acidic-residue-delimited search range.
    """
    pos1 = 15
    if len(seq) < pos1:
      return 0
    pos = pos1
    for i in range(1, pos1):
      if seq[i] in {"D", "E"}:
        pos = i + 1
        break
    for i in range(pos, len(seq)):
      if seq[i] in {"D", "E"}:
        return i + 1
    return len(seq)

  def _hmom(self, deg: int, segment: Sequence[str]) -> float:
    """Compute the hydrophobic moment of a short N-terminal segment.

    Args:
      deg: Helical rotation angle in degrees.
      segment: Sequence segment represented as one residue per element.

    Returns:
      Hydrophobic moment score normalized by the original window length.
    """
    import math

    radian_angle = 3.14159 * deg / 180.0
    ges = AMINO_ACID_INDEXES["Engelman GES 1986"]
    sum_sin = 0.0
    sum_cos = 0.0
    for i, aa in enumerate(segment):
      sum_sin += ges[aa] * math.sin(radian_angle * i)
      sum_cos += ges[aa] * math.cos(radian_angle * i)
    return math.sqrt(sum_sin**2 + sum_cos**2) / self.LSEG2

  def _mitdisc(self, seq: Sequence[str]) -> float:
    """Score the mitochondrial targeting sequence discriminator.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      MITDISC score used by the original PSORT workflow.
    """
    freq = {aa: 0 for aa in AMINO_ACID_INDEXES["Engelman GES 1986"]}
    search_range = self._getrange("".join(seq))
    for aa in seq[:search_range]:
      freq[aa] += 1
    limit = self.LSEG2 if search_range > self.LSEG2 else search_range
    segment = seq[:limit]
    m95 = self._hmom(95, segment)
    m75 = self._hmom(75, segment)
    coeffs = (
      2.0102 - 1.0973,
      0.3922 - 0.2538,
      0.4737 - 0.3566,
      -0.7417 + 0.2655,
      9.2710 - 11.3204,
      0.7522 - 0.4503,
      -17.5993 + 13.0901,
    )
    return (
      coeffs[0] * freq["R"]
      + coeffs[1] * m75
      + coeffs[2] * m95
      + coeffs[3] * freq["G"]
      + coeffs[4] * (freq["D"] + freq["E"])
      + coeffs[5] * (freq["S"] + freq["T"])
      + coeffs[6]
    )

  def _r3(self, seq: Sequence[str]) -> int:
    """Find the best R-3 mitochondrial cleavage motif position.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Zero-based motif anchor position or a negative sentinel when absent.
    """
    max_pos = len(seq) - 10 if len(seq) < 67 else 67
    ipos = -1000
    for i in range(11, max_pos - 2):
      if seq[i] == "R" and seq[i + 2] == "Y" and seq[i + 3] in {"S", "A"}:
        if abs(i - 23) < abs(ipos - 23):
          ipos = i
    return ipos

  def _r10(self, seq: Sequence[str]) -> int:
    """Find the best R-10 mitochondrial cleavage motif position.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Zero-based motif anchor position or a negative sentinel when absent.
    """
    max_pos = len(seq) - 10 if len(seq) < 60 else 60
    ipos = -1000
    for i in range(4, max_pos - 2):
      if seq[i] == "R" and seq[i + 2] == "F" and seq[i + 3] == "S":
        if abs(i - 23) < abs(ipos - 23):
          ipos = i
    return ipos

  def _r2(self, seq: Sequence[str]) -> int:
    """Find the fallback R-2 cleavage-associated transition motif.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Zero-based arginine position or a negative sentinel when absent.
    """
    ipos = -1000
    found = False
    for i in range(len(seq) - 12):
      if seq[i] in {"D", "E"}:
        for j in range(i + 1, i + 13):
          if seq[j] in {"D", "E"}:
            ipos = i
            found = True
            break
      if found:
        break
    if found:
      for k in range(ipos - 2, 0, -1):
        if seq[k] == "R":
          return k
    return -1000

  def _gavel(self, seq: Sequence[str]) -> Tuple[int, str]:
    """Predict a mitochondrial cleavage site using Gavel-style motifs.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Tuple of cleavage-site position and motif label.
    """
    ipos = self._r3(seq)
    if ipos > 0:
      return ipos + 4, "R-3"
    ipos = self._r10(seq)
    if ipos > 0:
      return ipos + 11, "R-10"
    ipos = self._r2(seq)
    if ipos > 0:
      return ipos + 11, "R-2"
    return 0, "___"

  def _nls1(self, seq: Sequence[str]) -> int:
    """Score the short basic nuclear-localization motif.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Aggregate NLS1 motif score across the full sequence.
    """
    nbas = np = nh = 0
    for aa in seq[:4]:
      if aa in {"R", "K"}:
        nbas += 1
      elif aa == "P":
        np += 1
      elif aa == "H":
        nh += 1
    score = 5 if nbas == 4 else 4 if nbas == 3 and np == 1 else 3 if nbas == 3 and nh == 1 else 0
    for i in range(len(seq) - 4):
      nxt = seq[i + 4]
      if nxt in {"R", "K"}:
        nbas += 1
      elif nxt == "P":
        np += 1
      elif nxt == "H":
        nh += 1
      prev = seq[i]
      if prev in {"R", "K"}:
        nbas -= 1
      elif prev == "P":
        np -= 1
      elif prev == "H":
        nh -= 1
      current = 5 if nbas == 4 else 4 if nbas == 3 and np == 1 else 3 if nbas == 3 and nh == 1 else 0
      score += current
    return score

  def _scr7(self, seq: Sequence[str], i: int) -> int:
    """Score one seven-residue NLS2 motif candidate window.

    Args:
      seq: Sequence represented as one residue per element.
      i: Zero-based window start index.

    Returns:
      Motif score or ``-1`` when the window cannot match.
    """
    if seq[i] != "P":
      return -1
    max_score = -1
    for k in range(3):
      nbas = 0
      for j in range(1, 5):
        aa = seq[i + j + k]
        if aa in {"R", "K"}:
          nbas += 1
      if nbas == 4:
        max_score = 5
      elif nbas == 3:
        max_score = max(max_score, 5 - k)
    return max_score

  def _nls2(self, seq: Sequence[str]) -> int:
    """Score the longer proline-anchored nuclear-localization motif.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Aggregate NLS2 motif score across the full sequence.
    """
    score = 0
    for i in range(len(seq) - 7):
      if seq[i] == "P":
        hit = self._scr7(seq, i)
        if hit != -1:
          score += hit
    return score

  def _bipartite(self, seq: str) -> int:
    """Score the bipartite nuclear-localization motif.

    Args:
      seq: Sequence string.

    Returns:
      Bipartite NLS score, including the original double-counting behavior.
    """
    score = 0
    for i in range(len(seq) - 16):
      if re.match(r"[RK][RK]", seq[i : i + 2]) is None:
        continue
      cnt = sum(1 for aa in seq[i + 12 : i + 17] if aa in {"R", "K"})
      # Preserve the historical double-add bug because the bundled training
      # data and model weights were learned against that behavior.
      if cnt >= 3:
        score += cnt
      if cnt >= 3:
        score += cnt
    return score

  def _nucaa(self, seq: Sequence[str]) -> int:
    """Score global basic-residue enrichment for nuclear localization.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Discretized basic-residue enrichment score.
    """
    count = sum(1 for aa in seq if aa in {"R", "K"}) / len(seq)
    return int(count * 10 - 1) if count > 0.2 else 0

  def _nucdisc(self, seq: Sequence[str]) -> float:
    """Combine nuclear-localization heuristics into the NUCDISC score.

    Args:
      seq: Sequence represented as one residue per element.

    Returns:
      Combined nuclear-localization discrimination score.
    """
    return (
      (0.0901 - 0.0274) * self._nls1(seq)
      + (0.1648 - 0.0786) * self._nls2(seq)
      + (0.1063 - 0.0241) * self._bipartite("".join(seq))
      + (1.1162 - 0.1665) * self._nucaa(seq)
      + (-1.2642 + 0.7904)
    )

  @staticmethod
  def _rnp1(seq: str) -> int:
    """Count RNA-binding RNP1 motif matches.

    Args:
      seq: Sequence string.

    Returns:
      Number of RNP1 motif matches.
    """
    return len(re.findall(r"[RK]G[^EDRKHPCG][AGSCI][FY][LIVA].[FYM]", seq))

  @staticmethod
  def _actin(seq: str) -> int:
    """Count actinin-type actin-binding motif matches.

    Args:
      seq: Sequence string.

    Returns:
      Total number of type-1 and type-2 actin-binding motif matches.
    """
    return len(re.findall(r"[EQ]..[ATV]F..W.N", seq)) + len(
      re.findall(r"[LIVM].[SGN][LIVM][DAGHE][SAG].[DEAG][LIVM].[DEAG]....[LIVM].L[SAG][LIVM][LIVM]W.[LIVM][LIVM]", seq)
    )

  @staticmethod
  def _hdel(seq: str) -> int:
    """Detect the C-terminal ER-retention motif.

    Args:
      seq: Sequence string.

    Returns:
      ``1`` when the sequence ends with ``HDEL`` or ``KDEL``, else ``0``.
    """
    return 1 if seq[-4:] in {"HDEL", "KDEL"} else 0

  @staticmethod
  def _pts1(seq: str) -> float:
    """Score the C-terminal peroxisomal targeting signal PTS1.

    Args:
      seq: Sequence string.

    Returns:
      PTS1 heuristic score.
    """
    if seq.endswith("SKL"):
      return 10 / 12
    if seq.endswith("SKF") or seq.endswith("AKL"):
      return 0.5
    return 0.25 if re.search(r"([SAGCN][RKH][LIVMAF])$", seq) else 0.0

  @staticmethod
  def _pts2(seq: str) -> int:
    """Count PTS2 motif matches.

    Args:
      seq: Sequence string.

    Returns:
      Number of PTS2 motif matches.
    """
    return len(re.findall(r"[RK][LI].....[HQ]L", seq))

  @staticmethod
  def _vaccalc(seq: str) -> int:
    """Count candidate vacuolar targeting motifs.

    Args:
      seq: Sequence string.

    Returns:
      Number of vacuolar motif matches.
    """
    return len(re.findall(r"[TIK]LP[NKI]", seq))

  @staticmethod
  def _nmyr(seq: str) -> int:
    """Score the N-myristoylation motif heuristic.

    Args:
      seq: Sequence string.

    Returns:
      N-myristoylation score.
    """
    match = re.search(r"^(M?G[^EDRKHPFYW]..[STAGCN][^P])", seq)
    if not match:
      return 0
    score = 1
    if re.search(r"^M?GC", seq):
      score += 1
    score += 1 if sum(1 for aa in seq[3:10] if aa == "K") >= 2 else 0
    return score

  @staticmethod
  def _isoprenyl(seq: str) -> int:
    """Score C-terminal isoprenylation motifs.

    Args:
      seq: Sequence string.

    Returns:
      Farnesylation / geranylgeranylation motif score.
    """
    if re.search(r"(C[^DENQ][LIVM].)$", seq):
      return 2
    if re.search(r"(C.C)$", seq) or re.search(r"(CC..)$", seq):
      return 1
    return 0

  @staticmethod
  def _erm(seq: str, mtype: str) -> int:
    """Score ER membrane retention motifs with topology-aware bonuses.

    Args:
      seq: Sequence string.
      mtype: Internal membrane-topology label.

    Returns:
      ER membrane retention score.
    """
    score = 0
    nseg = seq[1:5]
    cseg = seq[len(seq) - 5 : len(seq) - 1]
    r_count = nseg.count("R")
    if r_count:
      score += r_count
      if mtype in {"2 ", "Nt"}:
        score += 2
    k_count = cseg.count("K")
    if k_count:
      score += k_count
      if mtype in {"1a", "1b"}:
        score += 2
    return score

  @staticmethod
  def _yqrl(seq: str, tmsnum: int) -> int:
    """Score the YQRL endocytic transport motif.

    Args:
      seq: Sequence string.
      tmsnum: Number of predicted transmembrane segments.

    Returns:
      YQRL transport motif score.
    """
    matches = len(re.findall(r"YQRL", seq))
    if tmsnum == 0:
      return 0
    if tmsnum == 1:
      return matches + 3
    return matches

  @staticmethod
  def _tyros(seq: str, tmsnum: int, start: int, end: int) -> float:
    """Score tyrosine enrichment in the membrane tail.

    Args:
      seq: Sequence string.
      tmsnum: Number of predicted transmembrane segments.
      start: One-based tail start position.
      end: One-based tail end position.

    Returns:
      Tail tyrosine score normalized by tail length.
    """
    if tmsnum != 1 or (end - start + 1) > 50:
      return 0.0
    count = sum(1 for aa in seq[start - 1 : end] if aa == "Y")
    return 0.0 if count == 0 else 10.0 * count / (end - start + 1)

  @staticmethod
  def _dileu(seq: str, tmsnum: int, start: int, end: int) -> float:
    """Score dileucine motif density in the membrane tail.

    Args:
      seq: Sequence string.
      tmsnum: Number of predicted transmembrane segments.
      start: One-based tail start position.
      end: One-based tail end position.

    Returns:
      Dileucine density score normalized by tail length.
    """
    if tmsnum != 1:
      return 0.0
    tail = seq[start - 1 : end]
    matches = len(re.findall(r"LL", tail))
    return 10.0 * matches / (end - start + 1)
