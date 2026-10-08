"""Protein position-specific scoring matrices and affine-gap profile alignment.

Scores use base-2 log odds (bits), not PSI-BLAST's integer score scale.
Only NumPy and the SDK's existing alignment workflows are used.
"""

import io
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from neurosnap.sequence.align import align_mafft, read_msa, run_mmseqs2, run_phmmer_mafft

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AMBIGUOUS = {"B": "DN", "Z": "EQ", "J": "IL", "X": AA_ORDER}


def _background(values=None):
  if values is None:
    result = np.full(20, 1.0 / 20)
  elif isinstance(values, Mapping):
    if set(values) != set(AA_ORDER):
      raise ValueError("Background must specify all 20 standard amino acids.")
    result = np.array([values[a] for a in AA_ORDER], dtype=float)
  else:
    result = np.asarray(values, dtype=float).copy()
  if result.shape != (20,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
    raise ValueError("Background must contain 20 finite, strictly positive frequencies.")
  result /= result.max()
  return result / result.sum()


def _distribution(residue, background, ambiguous):
  result = np.zeros(20)
  if residue in AA_ORDER:
    result[AA_ORDER.index(residue)] = 1
  elif residue in "-.":
    pass
  elif ambiguous == "ignore":
    pass
  elif ambiguous == "distribute" and residue in _AMBIGUOUS:
    indices = [AA_ORDER.index(a) for a in _AMBIGUOUS[residue]]
    result[indices] = background[indices] / background[indices].sum()
  else:
    raise ValueError(f"Unsupported amino acid {residue!r}.")
  return result


@dataclass(frozen=True)
class PSSM:
  """A protein profile with rows in query/MSA order and columns in :data:`AA_ORDER`.

  Stored profile data:

    * ``probabilities``: Smoothed residue probabilities, shape (length, 20).
    * ``background``: Positive background frequencies, shape (20,).
    * ``counts``: Weighted residue counts before smoothing, shape (length, 20).
    * ``sequence_weights``: Weights used for the input MSA, normalized to sum to its row count.
    * ``msa_columns``: Zero-based original match-column indices (A3M insertions excluded).
    * ``query_positions``: Zero-based ungapped query indices; -1 denotes a query gap.
    * ``query``: Aligned query characters for retained rows, or None without a query row.

  Arrays are copied and made read-only. Direct construction requires all metadata;
  use :func:`pssm_from_msa` for routine construction.
  """

  probabilities: np.ndarray
  background: np.ndarray
  counts: np.ndarray
  sequence_weights: np.ndarray
  msa_columns: np.ndarray
  query_positions: np.ndarray
  query: Optional[str] = None

  def __post_init__(self):
    for name in ("probabilities", "background", "counts", "sequence_weights", "msa_columns", "query_positions"):
      dtype = int if name in ("msa_columns", "query_positions") else float
      value = np.array(getattr(self, name), dtype=dtype, copy=True)
      value.setflags(write=False)
      object.__setattr__(self, name, value)
    p = self.probabilities
    if p.ndim != 2 or p.shape[1] != 20 or len(p) == 0 or not np.all(np.isfinite(p)) or np.any(p < 0):
      raise ValueError("Probabilities must be a nonempty, finite, nonnegative (length, 20) matrix.")
    if not np.allclose(p.sum(axis=1), 1):
      raise ValueError("Each probability row must sum to one.")
    if not np.allclose(self.background, _background(self.background)):
      raise ValueError("Background frequencies must sum to one.")
    if self.counts.shape != p.shape or not np.all(np.isfinite(self.counts)) or np.any(self.counts < 0):
      raise ValueError("Counts must be finite, nonnegative, and match the probability matrix.")
    if self.sequence_weights.ndim != 1 or not len(self.sequence_weights) or not np.all(np.isfinite(self.sequence_weights)):
      raise ValueError("Sequence weights must be a nonempty finite vector.")
    if np.any(self.sequence_weights < 0) or self.sequence_weights.sum() <= 0:
      raise ValueError("Sequence weights must be nonnegative with positive total weight.")
    if self.msa_columns.shape != (len(p),) or self.query_positions.shape != (len(p),):
      raise ValueError("Coordinate arrays must have one entry per profile row.")
    if np.any(self.msa_columns < 0) or np.any(np.diff(self.msa_columns) <= 0) or np.any(self.query_positions < -1):
      raise ValueError("Invalid profile coordinates.")
    if self.query is not None and len(self.query) != len(p):
      raise ValueError("Query must have one character per profile row.")

  def __len__(self):
    return len(self.probabilities)

  @property
  def scores(self):
    """Return log2(probability/background); unobserved residues may score -inf."""
    with np.errstate(divide="ignore"):
      return np.log2(self.probabilities / self.background)

  @property
  def consensus(self):
    """Most probable standard residue per position, ties resolved by AA_ORDER."""
    return "".join(AA_ORDER[i] for i in self.probabilities.argmax(axis=1))

  def to_dataframe(self, kind="scores"):
    """Return a labeled pandas DataFrame for scores, probabilities, or counts.

    The index contains original MSA match-column indices. Pandas is an existing
    SDK dependency and is imported only when this method is called.
    """
    import pandas as pd

    if kind not in ("scores", "probabilities", "counts"):
      raise ValueError("kind must be scores, probabilities, or counts.")
    return pd.DataFrame(getattr(self, kind).copy(), index=self.msa_columns, columns=list(AA_ORDER)).rename_axis("msa_column")


def pssm_from_msa(
  msa: Union[str, io.TextIOBase, Sequence[str], Mapping[str, str]],
  *,
  background=None,
  pseudocount: float = 1.0,
  weighting: str = "henikoff",
  weights=None,
  ambiguous: str = "distribute",
  query_index: Optional[int] = 0,
  drop_query_gaps: bool = True,
  a3m: bool = False,
) -> PSSM:
  """Build a PSSM from aligned sequences, FASTA text/path, or a text handle.

  Parameters:
    msa: Equal-length aligned rows, a name-to-sequence mapping, or FASTA input.
    background: Mapping or vector in AA_ORDER; defaults to uniform frequencies.
    pseudocount: Total prior mass per column, distributed using the background.
      Zero is allowed and may produce -inf scores. Empty columns use background.
    weighting: ``"henikoff"`` (position-based) or ``"uniform"``. Henikoff weighting
      counts fractional ambiguous residues and excludes gaps. All-gap rows get
      zero weight unless every row is all gaps, when uniform weights are used.
    weights: Optional explicit nonnegative row weights; overrides weighting.
      Weights are normalized to sum to the number of input sequences.
    ambiguous: ``"distribute"`` assigns B/Z/J/X across their possible standard
      residues in background proportions; ``"ignore"`` skips nonstandard letters;
      ``"error"`` rejects them. U/O are rejected unless ignored.
    query_index: Query row index, default first row. None retains all columns
      and records -1 for every query position.
    drop_query_gaps: Remove columns that are gaps in the query row.
    a3m: Strip lowercase insertion residues and dots BEFORE uppercasing. False
      treats lowercase FASTA residues as ordinary amino acids.

  Query coordinates are zero-based in the original ungapped query, counting
  lowercase A3M insertion residues even though those residues have no profile row. This is a transparent log-odds model, not PSI-BLAST emulation.
  """
  bg = _background(background)
  if not np.isfinite(pseudocount) or pseudocount < 0:
    raise ValueError("pseudocount must be finite and nonnegative.")
  if ambiguous not in ("distribute", "ignore", "error"):
    raise ValueError("ambiguous must be distribute, ignore, or error.")
  if weighting not in ("henikoff", "uniform"):
    raise ValueError("weighting must be henikoff or uniform.")
  if isinstance(msa, (str, io.TextIOBase)):
    rows = [seq for _, seq in read_msa(msa, allow_chars="-XBZJUO.", sequence_mode="a3m_preserve" if a3m else "uppercase")]
  elif isinstance(msa, Mapping):
    rows = list(msa.values())
  else:
    rows = list(msa)
  if not rows or any(not isinstance(seq, str) for seq in rows):
    raise ValueError("MSA must contain at least one sequence string.")
  raw_rows = rows
  rows = [(re.sub(r"[a-z.]", "", seq) if a3m else seq).upper() for seq in rows]
  length = len(rows[0])
  if not length or any(len(seq) != length for seq in rows):
    raise ValueError("MSA match columns must be nonempty and equal in length.")
  if any(re.search(r"[^A-Z.\-]", seq) for seq in rows):
    raise ValueError("MSA contains invalid sequence characters.")
  if query_index is not None and (isinstance(query_index, bool) or not isinstance(query_index, int) or not 0 <= query_index < len(rows)):
    raise ValueError("query_index must be a valid nonnegative row index or None.")
  query = rows[query_index] if query_index is not None else None
  keep = np.array([i for i in range(length) if query is None or not drop_query_gaps or query[i] not in "-."])
  if not len(keep):
    raise ValueError("No profile positions remain after removing query gaps.")
  # Work one column at a time rather than allocating an N x L x 20 array.
  row_weights = np.zeros(len(rows))
  if weights is not None:
    row_weights = np.asarray(weights, dtype=float).copy()
    if row_weights.shape != (len(rows),) or not np.all(np.isfinite(row_weights)) or np.any(row_weights < 0) or not np.any(row_weights > 0):
      raise ValueError("weights must have one finite nonnegative value per row and a positive total.")
  elif weighting == "uniform":
    row_weights.fill(1)
  else:
    for col in keep:
      contributions = np.array([_distribution(seq[col], bg, ambiguous) for seq in rows])
      totals = contributions.sum(axis=0)
      present = totals > 0
      if present.any():
        row_weights += (contributions[:, present] / totals[present]).sum(axis=1) / present.sum()
    if row_weights.sum() == 0:
      row_weights.fill(1)
  row_weights /= row_weights.max()
  row_weights *= len(rows) / row_weights.sum()
  counts = np.zeros((len(keep), 20))
  for pos, col in enumerate(keep):
    for weight, seq in zip(row_weights, rows):
      counts[pos] += weight * _distribution(seq[col], bg, ambiguous)
  smoothed = counts + pseudocount * bg
  totals = smoothed.sum(axis=1)
  probabilities = np.broadcast_to(bg, counts.shape).copy()
  np.divide(smoothed, totals[:, None], out=probabilities, where=totals[:, None] > 0)
  positions = np.full(length, -1, dtype=int)
  if query is not None:
    position, col = 0, 0
    for residue in raw_rows[query_index]:
      if a3m and residue == ".":
        continue
      if a3m and residue.islower():
        position += 1
        continue
      if residue not in "-.":
        positions[col] = position
        position += 1
      col += 1
  return PSSM(probabilities, bg, counts, row_weights, keep, positions[keep], None if query is None else "".join(query[i] for i in keep))


def generate_pssm(sequences, *, method="mafft", alignment_kwargs=None, pssm_kwargs=None) -> PSSM:
  """Generate an MSA through an existing workflow and build its profile.

  Parameters:
    sequences: MAFFT sequences/path, or a single query string for MMseqs2 or
      phmmer_mafft. A single sequence alone supplies no evolutionary evidence.
    method: ``"mafft"``, ``"mmseqs2"``, or ``"phmmer_mafft"``.
    alignment_kwargs: Keyword arguments forwarded to the selected workflow.
      MMseqs2 requires ``output`` and network access; phmmer_mafft requires
      ``ref_db_path``. MAFFT/phmmer retain their existing executable requirements.
    pssm_kwargs: Options forwarded to pssm_from_msa. MMseqs2 input is always A3M.

  No new dependencies are installed. Only MSA-to-PSSM calculation and profile
  alignment run entirely in Python/NumPy; workflow prerequisites still apply.
  """
  align_options = dict(alignment_kwargs or {})
  profile_options = dict(pssm_kwargs or {})
  if method == "mafft":
    _, rows = align_mafft(sequences, **align_options)
  elif method == "phmmer_mafft":
    if not isinstance(sequences, str):
      raise ValueError("phmmer_mafft requires a single query string.")
    _, rows = run_phmmer_mafft(sequences, **align_options)
  elif method == "mmseqs2":
    if not isinstance(sequences, str):
      raise ValueError("mmseqs2 requires a single query string.")
    msas, _ = run_mmseqs2(sequences, **align_options)
    if len(msas) != 1:
      raise ValueError("Expected exactly one query MSA.")
    rows = msas[0]
    profile_options["a3m"] = True
  else:
    raise ValueError("method must be mafft, mmseqs2, or phmmer_mafft.")
  return pssm_from_msa(rows, **profile_options)


def _sequence_indices(sequence):
  if not isinstance(sequence, str) or not sequence or re.search(r"[^A-Za-z]", sequence):
    raise ValueError("Expected a nonempty ungapped amino acid sequence.")
  sequence = sequence.upper()
  indices = []
  for residue in sequence:
    if residue in AA_ORDER:
      indices.append([AA_ORDER.index(residue)])
    elif residue in _AMBIGUOUS:
      indices.append([AA_ORDER.index(a) for a in _AMBIGUOUS[residue]])
    else:
      raise ValueError(f"Unsupported amino acid {residue!r}.")
  return sequence, indices


def _sequence_scores(profile, sequence):
  sequence, columns = _sequence_indices(sequence)
  matrix = np.empty((len(profile), len(sequence)))
  for j, indices in enumerate(columns):
    # Marginal log odds: X is uninformative (0 bits), B/Z/J use possible residues.
    with np.errstate(divide="ignore"):
      matrix[:, j] = np.log2(profile.probabilities[:, indices].sum(axis=1) / profile.background[indices].sum())
  return sequence, matrix


def score_sequence(profile: PSSM, sequence: str) -> float:
  """Score a same-length ungapped sequence in bits, marginalizing B/Z/J/X.

  Runtime is linear in profile length and no pairwise score matrix is allocated.
  """
  sequence, columns = _sequence_indices(sequence)
  if len(sequence) != len(profile):
    raise ValueError("Sequence length must match profile length.")
  with np.errstate(divide="ignore"):
    return float(sum(np.log2(profile.probabilities[i, indices].sum() / profile.background[indices].sum()) for i, indices in enumerate(columns)))


def scan_sequence(profile: PSSM, sequence: str) -> np.ndarray:
  """Return ungapped window scores in bits, indexed by zero-based start position.

  A sequence shorter than the profile yields an empty array.
  """
  _, matrix = _sequence_scores(profile, sequence)
  return np.array([sum(matrix[i, start + i] for i in range(len(profile))) for start in range(len(sequence) - len(profile) + 1)])


@dataclass(frozen=True)
class ProfileAlignment:
  """Alignment score in bits, display strings, and zero-based coordinates.

  ``pairs`` contains profile/sequence indices, with None for a gap. Profile
  indices address PSSM rows; use msa_columns/query_positions to map them back.
  start/end are half-open spans in each input. Display strings use consensus
  residues for profiles. Local alignments with no positive score have empty
  strings/pairs and zero spans. Scores are not calibrated E-values.
  """

  score: float
  aligned_a: str
  aligned_b: str
  pairs: Tuple[Tuple[Optional[int], Optional[int]], ...]
  start_a: int
  end_a: int
  start_b: int
  end_b: int
  mode: str


def _align(matrix, a, b, mode, gap_open, gap_extend):
  if mode not in ("local", "global"):
    raise ValueError("mode must be local or global.")
  if not np.isfinite(gap_open) or not np.isfinite(gap_extend) or gap_open < 0 or gap_extend < 0:
    raise ValueError("Gap penalties must be finite and nonnegative.")
  n, m = matrix.shape
  # M: matched pair, X: gap in b, Y: gap in a. A length-k gap costs
  # gap_open + (k-1)*gap_extend. Opposite gaps cannot directly transition.
  scores = np.full((3, n + 1, m + 1), -np.inf)
  trace = np.full((3, n + 1, m + 1), -1, dtype=np.int8)
  scores[0, 0, 0] = 0
  if mode == "global":
    for i in range(1, n + 1):
      scores[1, i, 0] = -gap_open - (i - 1) * gap_extend
      trace[1, i, 0] = 0 if i == 1 else 1
    for j in range(1, m + 1):
      scores[2, 0, j] = -gap_open - (j - 1) * gap_extend
      trace[2, 0, j] = 0 if j == 1 else 2
  else:
    scores[0, :, 0] = 0
    scores[0, 0, :] = 0
  best, endpoint = 0.0, (0, 0, 0)
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      previous = scores[:, i - 1, j - 1]
      state = int(np.argmax(previous))
      scores[0, i, j] = previous[state] + matrix[i - 1, j - 1]
      trace[0, i, j] = state
      for target, pi, pj in ((1, i - 1, j), (2, i, j - 1)):
        opened = scores[0, pi, pj] - gap_open
        extended = scores[target, pi, pj] - gap_extend
        parent = 0 if opened >= extended else target
        scores[target, i, j] = opened if parent == 0 else extended
        trace[target, i, j] = parent
      if mode == "local":
        for state in range(3):
          if scores[state, i, j] <= 0:
            scores[state, i, j] = 0
            trace[state, i, j] = -1
          if scores[state, i, j] > best:
            best = float(scores[state, i, j])
            endpoint = (state, i, j)
  if mode == "global":
    state = int(np.argmax(scores[:, n, m]))
    best, endpoint = float(scores[state, n, m]), (state, n, m)
    if not np.isfinite(best):
      raise ValueError("No finite global alignment exists; use positive pseudocounts or local alignment.")
  state, i, j = endpoint
  end_a, end_b = i, j
  pairs = []
  while i > 0 or j > 0:
    parent = int(trace[state, i, j])
    if parent == -1:
      break
    if state == 0:
      pairs.append((i - 1, j - 1))
      i, j = i - 1, j - 1
    elif state == 1:
      pairs.append((i - 1, None))
      i -= 1
    else:
      pairs.append((None, j - 1))
      j -= 1
    state = parent
  pairs.reverse()
  aligned_a = "".join("-" if x is None else a[x] for x, _ in pairs)
  aligned_b = "".join("-" if y is None else b[y] for _, y in pairs)
  return ProfileAlignment(best, aligned_a, aligned_b, tuple(pairs), i, end_a, j, end_b, mode)


def align_sequence_to_pssm(profile: PSSM, sequence: str, *, mode="local", gap_open=5.0, gap_extend=1.0) -> ProfileAlignment:
  """Align an ungapped protein sequence to a PSSM using affine gap penalties.

  Penalties are positive costs in bits. A k-residue gap costs
  gap_open + (k-1)*gap_extend. Time and memory are O(profile length * sequence
  length); this implementation is intended for individual proteins, not database
  searches. B/Z/J/X use marginal log odds as in score_sequence.
  """
  sequence, matrix = _sequence_scores(profile, sequence)
  return _align(matrix, profile.consensus, sequence, mode, gap_open, gap_extend)


def align_pssms(a: PSSM, b: PSSM, *, mode="local", gap_open=5.0, gap_extend=1.0, background=None) -> ProfileAlignment:
  """Align two profiles using log-odds probability overlap and affine gaps.

  Position score: log2(sum_r p_a(r) * p_b(r) / background(r)). This symmetric
  score compares residue overlap with independent background draws. It is a
  simple profile similarity model, not an HHsearch/HMM alignment or a calibrated
  significance score. Uniform/background-only columns score zero when profiles
  share that background. Different backgrounds require an explicit common
  background argument. Gap semantics and O(L_a * L_b) costs match sequence
  alignment. Output strings display profile consensus residues.
  """
  if background is None:
    if not np.allclose(a.background, b.background, rtol=1e-10, atol=0):
      raise ValueError("Profiles have different backgrounds; supply a common background explicitly.")
    bg = a.background
  else:
    bg = _background(background)
  with np.errstate(divide="ignore"):
    matrix = np.log2((a.probabilities / bg) @ b.probabilities.T)
  return _align(matrix, a.consensus, b.consensus, mode, gap_open, gap_extend)
