"""Numerical references, coordinate checks, and exhaustive affine alignment checks."""

import io
import itertools

import numpy as np
import pytest

from neurosnap.sequence.align import read_msa
from neurosnap.sequence.pssm import (
  AA_ORDER,
  align_pssms,
  align_sequence_to_pssm,
  generate_pssm,
  pssm_from_msa,
  scan_sequence,
  score_sequence,
)


def test_uniform_counts_and_log_odds():
  profile = pssm_from_msa(["AC", "AD"], weighting="uniform", pseudocount=2)
  a, c, d = map(AA_ORDER.index, "ACD")
  assert profile.counts[0, a] == 2
  assert profile.counts[1, c] == profile.counts[1, d] == 1
  assert profile.probabilities[0, a] == pytest.approx(2.1 / 4)
  assert profile.scores[0, a] == pytest.approx(np.log2(10.5))
  np.testing.assert_allclose(profile.probabilities.sum(axis=1), 1)
  assert profile.consensus == "AC"
  assert list(profile.to_dataframe().columns) == list(AA_ORDER)
  with pytest.raises(ValueError):
    profile.to_dataframe("bad")
  with pytest.raises(ValueError):
    profile.probabilities[0, 0] = 0


def test_henikoff_reduces_duplicate_bias():
  profile = pssm_from_msa(["AA", "AA", "CC"], pseudocount=0)
  np.testing.assert_allclose(profile.sequence_weights, [0.75, 0.75, 1.5])
  assert profile.probabilities[0, AA_ORDER.index("A")] == pytest.approx(0.5)
  assert profile.probabilities[0, AA_ORDER.index("C")] == pytest.approx(0.5)


def test_weight_scaling_gaps_and_ambiguous_residues():
  bg = dict.fromkeys(AA_ORDER, 1)
  bg["D"] = 3
  profile = pssm_from_msa(["BZXJ-", "-----"], background=bg, weights=[2, 0], query_index=None, pseudocount=0)
  assert profile.counts[0, AA_ORDER.index("D")] == pytest.approx(1.5)
  assert profile.counts[0, AA_ORDER.index("N")] == pytest.approx(0.5)
  np.testing.assert_allclose(profile.probabilities[2], profile.background)
  np.testing.assert_allclose(profile.probabilities[4], profile.background)
  np.testing.assert_allclose(profile.sequence_weights, [2, 0])
  np.testing.assert_array_equal(profile.query_positions, [-1] * 5)
  ignored = pssm_from_msa(["UO"], ambiguous="ignore", pseudocount=0)
  np.testing.assert_allclose(ignored.probabilities, np.full((2, 20), 0.05))


def test_coordinates_and_a3m_input_variants(tmp_path):
  text = ">query\nA-CdE\n>hit\nATCggE\n"
  path = tmp_path / "msa.a3m"
  path.write_text(text)
  for msa in (text, str(path), io.StringIO(text), ["A-CdE", "ATCggE"], {"q": "A-CdE", "h": "ATCggE"}):
    profile = pssm_from_msa(msa, a3m=True)
    assert profile.query == "ACE"
    np.testing.assert_array_equal(profile.msa_columns, [0, 2, 3])
    np.testing.assert_array_equal(profile.query_positions, [0, 1, 3])
  profile = pssm_from_msa(text, a3m=True, drop_query_gaps=False)
  np.testing.assert_array_equal(profile.query_positions, [0, -1, 1, 3])
  assert pssm_from_msa(["acde"]).query == "ACDE"


@pytest.mark.parametrize(
  "options",
  [
    {"pseudocount": -1},
    {"pseudocount": np.nan},
    {"weighting": "bad"},
    {"ambiguous": "bad"},
    {"background": [0] * 20},
    {"background": {"A": 1}},
    {"weights": [1, 2]},
    {"weights": [0]},
    {"weights": [-1]},
    {"query_index": -1},
    {"query_index": True},
    {"query_index": 2},
  ],
)
def test_invalid_options(options):
  with pytest.raises(ValueError):
    pssm_from_msa(["AC"], **options)


@pytest.mark.parametrize(
  "rows,options",
  [
    ([], {}),
    ([""], {}),
    (["AC", "A"], {}),
    (["---"], {}),
    (["A?"], {}),
    (["AU"], {}),
    (["AX"], {"ambiguous": "error"}),
    (["abc"], {"a3m": True}),
  ],
)
def test_invalid_msa(rows, options):
  with pytest.raises(ValueError):
    pssm_from_msa(rows, **options)


def test_sequence_scoring_and_scanning():
  p = pssm_from_msa(["ACD"], pseudocount=1)
  expected = sum(p.scores[i, AA_ORDER.index(a)] for i, a in enumerate("ACD"))
  assert score_sequence(p, "acd") == pytest.approx(expected)
  assert score_sequence(p, "XXX") == pytest.approx(0)
  scores = scan_sequence(p, "EACDE")
  assert len(scores) == 3
  assert scores[1] == pytest.approx(expected)
  assert len(scan_sequence(p, "AC")) == 0
  with pytest.raises(ValueError):
    score_sequence(p, "AC")
  with pytest.raises(ValueError):
    score_sequence(p, "A-C")
  with pytest.raises(ValueError):
    align_sequence_to_pssm(p, "AU")
  sharp = pssm_from_msa(["ACD"], pseudocount=0)
  assert score_sequence(sharp, "ACE") == -np.inf


def test_sequence_alignment_local_global_and_coordinates():
  p = pssm_from_msa(["ACD"], pseudocount=0.01)
  local = align_sequence_to_pssm(p, "WWACDWW")
  assert local.aligned_a == local.aligned_b == "ACD"
  assert (local.start_a, local.end_a, local.start_b, local.end_b) == (0, 3, 2, 5)
  assert local.pairs == ((0, 2), (1, 3), (2, 4))
  global_result = align_sequence_to_pssm(p, "ACWWD", mode="global", gap_open=2, gap_extend=0.5)
  assert global_result.aligned_a == "AC--D"
  assert global_result.aligned_b == "ACWWD"
  assert global_result.score == pytest.approx(score_sequence(p, "ACD") - 2.5)
  assert global_result.pairs == ((0, 0), (1, 1), (None, 2), (None, 3), (2, 4))
  deletion = align_sequence_to_pssm(p, "AD", mode="global", gap_open=2)
  assert deletion.aligned_b == "A-D"
  no_hit = align_sequence_to_pssm(p, "WWW")
  assert no_hit.score == 0 and no_hit.pairs == () and no_hit.aligned_a == ""
  for options in ({"mode": "bad"}, {"gap_open": -1}, {"gap_extend": np.nan}):
    with pytest.raises(ValueError):
      align_sequence_to_pssm(p, "ACD", **options)
  with pytest.raises(ValueError, match="No finite"):
    align_sequence_to_pssm(pssm_from_msa(["A"], pseudocount=0), "C", mode="global")


def test_profile_alignment_symmetric_and_background_handling():
  a = pssm_from_msa(["ACD"], pseudocount=0.01)
  b = pssm_from_msa(["ACWWD"], pseudocount=0.01)
  result = align_pssms(a, b, mode="global", gap_open=2, gap_extend=0.5)
  assert result.aligned_a == "AC--D" and result.aligned_b == "ACWWD"
  assert result.score == pytest.approx(align_pssms(b, a, mode="global", gap_open=2, gap_extend=0.5).score)
  expected = sum(np.log2(np.sum(a.probabilities[i] * b.probabilities[j] / a.background)) for i, j in ((0, 0), (1, 1), (2, 4))) - 2.5
  assert result.score == pytest.approx(expected)
  bg = np.arange(1, 21)
  c = pssm_from_msa(["ACD"], background=bg)
  with pytest.raises(ValueError, match="different backgrounds"):
    align_pssms(a, c)
  assert np.isfinite(align_pssms(a, c, background=bg).score)
  uniform = pssm_from_msa(["XXX"])
  assert align_pssms(uniform, uniform).score == pytest.approx(0)


def _all_global_scores(p, sequence, gap_open, gap_extend):
  # Independently enumerate every legal affine alignment for tiny inputs.
  def visit(i, j, previous, total):
    if i == len(p) and j == len(sequence):
      yield total
    if i < len(p) and j < len(sequence):
      score = p.scores[i, AA_ORDER.index(sequence[j])]
      yield from visit(i + 1, j + 1, "M", total + score)
    if i < len(p) and previous != "Y":
      yield from visit(i + 1, j, "X", total - (gap_extend if previous == "X" else gap_open))
    if j < len(sequence) and previous != "X":
      yield from visit(i, j + 1, "Y", total - (gap_extend if previous == "Y" else gap_open))

  return list(visit(0, 0, "M", 0))


def test_affine_dp_against_exhaustive_global_and_local_reference():
  strings = ["".join(chars) for n in range(1, 4) for chars in itertools.product("AC", repeat=n)]
  for query in strings:
    p = pssm_from_msa([query], pseudocount=1)
    for sequence in strings:
      for gap_open, gap_extend in ((2, 0.5), (0.5, 2), (0, 0)):
        result = align_sequence_to_pssm(p, sequence, mode="global", gap_open=gap_open, gap_extend=gap_extend)
        assert result.score == pytest.approx(max(_all_global_scores(p, sequence, gap_open, gap_extend)))
        # Local optimum equals the best global alignment of any pair of substrings (or empty).
        expected = 0
        for i in range(len(query)):
          for end_i in range(i + 1, len(query) + 1):
            subprofile = pssm_from_msa([query[i:end_i]], pseudocount=1)
            for j in range(len(sequence)):
              for end_j in range(j + 1, len(sequence) + 1):
                expected = max(expected, max(_all_global_scores(subprofile, sequence[j:end_j], gap_open, gap_extend)))
        local = align_sequence_to_pssm(p, sequence, gap_open=gap_open, gap_extend=gap_extend)
        assert local.score == pytest.approx(expected)
        assert len(local.aligned_a) == len(local.aligned_b) == len(local.pairs)


def test_generation_workflows(monkeypatch):
  import neurosnap.sequence.pssm as module

  calls = []

  def mafft(sequences, **kwargs):
    calls.append((sequences, kwargs))
    return ["q", "h"], ["AC", "AD"]

  monkeypatch.setattr(module, "align_mafft", mafft)
  assert generate_pssm(["AC", "AD"], alignment_kwargs={"threads": 1}).query == "AC"
  assert calls == [(["AC", "AD"], {"threads": 1})]
  monkeypatch.setattr(module, "run_phmmer_mafft", mafft)
  assert generate_pssm("AC", method="phmmer_mafft", alignment_kwargs={"ref_db_path": "db.fa"}).query == "AC"

  def mmseqs(query, **kwargs):
    calls.append((query, kwargs))
    return [">q\nAC\n>h\nAgC\n"], None

  monkeypatch.setattr(module, "run_mmseqs2", mmseqs)
  assert generate_pssm("AC", method="mmseqs2", alignment_kwargs={"output": "msa"}).query == "AC"
  assert calls[-1] == ("AC", {"output": "msa"})
  for method in ("bad", "mmseqs2", "phmmer_mafft"):
    with pytest.raises(ValueError):
      generate_pssm(["AC"], method=method)


def test_a3m_reader_preserves_or_strips_insertions_and_filters_match_columns():
  text = ">q\nACdE\n>h\nACggE\n>different\nACtD\n"
  assert list(read_msa(text))[0][1] == "ACDE"
  preserved = list(read_msa(text, a3m_insertions="preserve", id=100))
  assert preserved == [("q", "ACdE"), ("h", "ACggE")]
  assert list(read_msa(text, a3m_insertions="strip", id=100)) == [("q", "ACE"), ("h", "ACE")]
  assert list(read_msa(text, a3m_insertions="preserve", query="ACE", id=100)) == preserved
  assert list(read_msa(">q\nAC.dE\n", a3m_insertions="preserve"))[0][1] == "AC.dE"
  assert list(read_msa(">q\nAC.dE\n", a3m_insertions="strip"))[0][1] == "ACE"
  assert list(read_msa(">q\nACdE\n", uppercase=False))[0][1] == "ACdE"
  with pytest.raises(ValueError):
    list(read_msa(text, a3m_insertions="bad"))


def test_reader_handles_borrowed_stream_and_explicit_unaligned_query():
  stream = io.StringIO(">q\nA-CD\n>h\nATCD\n")
  assert len(list(read_msa(stream, allow_chars="-", query="ACD", id=100))) == 2
  assert not stream.closed
  with pytest.raises(ValueError, match="equal-length"):
    list(read_msa(">q\nACD\n>h\nAC\n", id=10))


def _trace_score(profile, sequence, alignment, gap_open, gap_extend):
  score, previous = 0, "M"
  for i, j in alignment.pairs:
    if i is not None and j is not None:
      score += profile.scores[i, AA_ORDER.index(sequence[j])]
      previous = "M"
    else:
      state = "X" if j is None else "Y"
      score -= gap_extend if previous == state else gap_open
      previous = state
  return score


def test_traceback_scores_and_spans_for_boundary_gaps_and_local_hits():
  rng = np.random.default_rng(37)
  for _ in range(50):
    query = "".join(rng.choice(list("ACDE"), size=5))
    sequence = "".join(rng.choice(list("ACDE"), size=7))
    profile = pssm_from_msa([query], pseudocount=1)
    for mode in ("global", "local"):
      result = align_sequence_to_pssm(profile, sequence, mode=mode, gap_open=1, gap_extend=0.25)
      assert result.score == pytest.approx(_trace_score(profile, sequence, result, 1, 0.25))
      assert result.aligned_a.replace("-", "") == query[result.start_a : result.end_a]
      assert result.aligned_b.replace("-", "") == sequence[result.start_b : result.end_b]
  profile = pssm_from_msa(["ACD"], pseudocount=0.01)
  result = align_sequence_to_pssm(profile, "WWACDWW", mode="global", gap_open=1, gap_extend=0.25)
  assert result.aligned_a == "--ACD--"
  assert result.score == pytest.approx(score_sequence(profile, "ACD") - 2.5)


@pytest.mark.integration
def test_generate_pssm_with_real_local_workflows(tmp_path):
  import shutil

  if shutil.which("mafft") is None:
    pytest.skip("MAFFT not available")
  sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEYGHIKLMNPQRSTVWY"]
  p = generate_pssm(sequences, alignment_kwargs={"threads": 1})
  assert p.query == sequences[0]
  assert p.counts.shape == (20, 20)
  if shutil.which("phmmer") is None:
    return
  pytest.importorskip("Bio")
  db = tmp_path / "homologs.fasta"
  db.write_text(f">q\n{sequences[0]}\n>h\n{sequences[1]}\n")
  p = generate_pssm(sequences[0], method="phmmer_mafft", alignment_kwargs={"ref_db_path": str(db), "phmmer_cpu": 1, "mafft_threads": 1})
  assert p.query == sequences[0]
  np.testing.assert_array_equal(p.query_positions, np.arange(20))


def test_all_gap_fasta_without_query_and_alternative_query_row():
  p = pssm_from_msa(">gaps\n---\n", query_index=None, pseudocount=0)
  np.testing.assert_allclose(p.probabilities, np.full((3, 20), 0.05))
  p = pssm_from_msa(">gaps\n---\n>query\nACD\n", query_index=1)
  assert p.query == "ACD"
  np.testing.assert_array_equal(p.query_positions, [0, 1, 2])
  with pytest.raises(AssertionError, match="all gaps"):
    list(read_msa(">gaps\n---\n", allow_chars="-", cov=10))


def test_a3m_query_coordinates_include_leading_and_internal_insertions():
  p = pssm_from_msa(">q\nxxA-Cdd.E\n>h\nATC.E\n", a3m=True, drop_query_gaps=False)
  assert p.query == "A-CE"
  np.testing.assert_array_equal(p.query_positions, [2, -1, 3, 6])
  np.testing.assert_array_equal(p.msa_columns, [0, 1, 2, 3])
