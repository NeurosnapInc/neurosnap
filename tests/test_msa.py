# tests/test_msa.py
import os
import shutil
from pathlib import Path

import pytest

from neurosnap.msa import (
  align_mafft,
  alignment_coverage,
  consensus_sequence,
  filter_msa,
  pad_seqs,
  read_msa,
  run_mmseqs2,
  run_mmseqs2_modes,
  run_phmmer,
  run_phmmer_mafft,
  seqid,
  write_msa,
)

# ---------- Paths & fixtures ----------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "unit" else HERE
DATA = ROOT  # your current artifacts are already under ./tests/
FASTA_DB = DATA / "test.fasta"  # existing in your repo


@pytest.fixture(scope="session")
def data_dir() -> Path:
  assert DATA.exists(), "tests directory not found"
  return DATA


@pytest.fixture(scope="session")
def has_phmmer() -> bool:
  return shutil.which("phmmer") is not None


@pytest.fixture(scope="session")
def has_mafft() -> bool:
  return shutil.which("mafft") is not None


@pytest.fixture(scope="session")
def has_hhfilter() -> bool:
  return shutil.which("hhfilter") is not None


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
  """Run test in an isolated CWD."""
  monkeypatch.chdir(tmp_path)
  return tmp_path


# ---------- read_msa ----------


def collect_msa(pairs_iter):
  pairs = list(pairs_iter)
  return [n for n, _ in pairs], [s for _, s in pairs]


def test_read_msa_from_string_basic():
  a3m = ">sp|Q9|name one\nACDEx-\n>seq2 desc\nACDE-\n"
  names, seqs = collect_msa(
    read_msa(
      a3m,
      allow_chars="-X",  # X allowed
      drop_chars="",  # nothing dropped
      remove_chars="*",  # nothing to remove in this string
      uppercase=True,
    )
  )
  # '|' becomes '_' and header stops at first space (default behavior)
  assert names == ["sp_Q9_name", "seq2"]
  assert seqs == ["ACDEX-", "ACDE-"]


def test_read_msa_from_file_and_truncation(tmp_path):
  fp = tmp_path / "toy.fasta"
  fp.write_text(">a\nAAAA\n>c\nCCCC\n")
  names, seqs = collect_msa(read_msa(str(fp), size=2))
  assert names == ["a", "c"]
  assert seqs == ["AAAA", "CCCC"]


def test_read_msa_drop_and_remove_and_uppercase():
  # drop 'X' rows, remove '-' and '*', uppercase on
  s = ">ok\nac-de*\n>bad\nAXC\n>ok2\nacde\n"
  names, seqs = collect_msa(read_msa(s, allow_chars="", drop_chars="X", remove_chars="-*", uppercase=True))
  assert names == ["ok", "ok2"]
  assert seqs == ["ACDE", "ACDE"]


def test_read_msa_name_allow_all_chars():
  s = ">we keep the whole header | yes\nACD\n"
  names, seqs = collect_msa(read_msa(s, name_allow_all_chars=True))
  assert names == ["we keep the whole header _ yes"]  # nothing stripped/replaced
  assert seqs == ["ACD"]


def test_read_msa_invalid_chars_raises():
  s = ">h\nACDEZ\n"  # 'Z' not in STANDARD_AAs and not allowed
  with pytest.raises(ValueError):
    list(read_msa(s, allow_chars=""))  # do not allow Z


def test_read_msa_header_without_sequence_raises():
  s = ">only\n"
  with pytest.raises(AssertionError):
    list(read_msa(s))


# ---------- write_msa ----------


def test_write_msa_roundtrip(tmp_path):
  out = tmp_path / "out.fasta"
  names = ["n1", "n2"]
  seqs = ["AAAA", "CCCC"]
  write_msa(str(out), names, seqs)
  rn, rs = collect_msa(read_msa(str(out)))
  assert rn == names and rs == seqs


def test_write_msa_len_mismatch_asserts(tmp_path):
  out = tmp_path / "out.fasta"
  with pytest.raises(AssertionError):
    write_msa(str(out), ["n1"], ["AA", "BB"])


# ---------- pad_seqs ----------


def test_pad_seqs_default_right_pad():
  seqs = ["AA", "AAAA", "A"]
  out = pad_seqs(seqs[:])  # copy
  assert out == ["AA--", "AAAA", "A---"]


def test_pad_seqs_truncate_true():
  seqs = ["AAAA", "AA--", "A----"]
  out = pad_seqs(seqs[:], truncate=True)
  assert out == ["AAAA", "AA--", "A---"]  # truncated to len(first)==4


def test_pad_seqs_truncate_len():
  seqs = ["AAAAAA", "BB"]
  out = pad_seqs(seqs[:], truncate=3)
  assert out == ["AAA", "BB-"]


def test_pad_seqs_truncate_len_invalid():
  with pytest.raises(AssertionError):
    pad_seqs(["AA"], truncate=0)


# ---------- seqid ----------


def test_seqid_basic_and_bounds():
  assert seqid("AAAA", "AAAA") == 100.0
  assert seqid("ABCD", "ABCE") == 75.0


def test_seqid_unequal_length_asserts():
  with pytest.raises(AssertionError):
    seqid("AAA", "AA")


# ---------- alignment_coverage ----------


def test_alignment_coverage_basic():
  assert alignment_coverage("A-CD", "AB-D") == 100.0


# ---------- filter_msa ----------


def test_filter_msa_identity_and_query_default(tmp_path):
  a3m = ">q\nACDE\n>hit1\nACDE\n>hit2\nACDD\n"
  out = tmp_path / "filtered.a3m"
  names, seqs = filter_msa(a3m, str(out), cov=50, id=90)
  assert names == ["q", "hit1"]
  assert seqs == ["ACDE", "ACDE"]
  rn, rs = collect_msa(read_msa(str(out)))
  assert rn == names and rs == seqs


def test_filter_msa_max_seqs(tmp_path):
  a3m = ">q\nACDE\n>hit1\nACDE\n>hit2\nACDD\n"
  out = tmp_path / "filtered_max.a3m"
  names, seqs = filter_msa(a3m, str(out), cov=0, id=0, max_seqs=2)
  assert names == ["q", "hit1"]
  assert seqs == ["ACDE", "ACDE"]


# ---------- consensus_sequence ----------


def test_consensus_sequence_basic():
  seqs = ["ACDE", "ACDF", "BCDF"]
  # per-position most common: A/C/D/F => "ACDF"
  assert consensus_sequence(seqs) == "ACDF"


def test_consensus_sequence_errors():
  with pytest.raises(ValueError):
    consensus_sequence([])
  with pytest.raises(ValueError):
    consensus_sequence(["AAA", "AA"])


# ---------- align_mafft (external tool) ----------


@pytest.mark.slow
@pytest.mark.integration
def test_align_mafft_list_input(has_mafft):
  if not has_mafft:
    pytest.skip("mafft not available on PATH")
  names, seqs = align_mafft(["ACDE", "ACDF"])
  assert len(names) == 2 and len(seqs) == 2
  assert len(seqs[0]) == len(seqs[1])  # aligned => equal lengths
  # names are auto-generated seq_0, seq_1 when list input is used
  assert names == ["seq_0", "seq_1"]


@pytest.mark.slow
@pytest.mark.integration
def test_align_mafft_dict_input(has_mafft):
  if not has_mafft:
    pytest.skip("mafft not available on PATH")
  names, seqs = align_mafft({"a": "ACDE", "b": "ACDF"})
  assert names == ["a", "b"]
  assert len(seqs[0]) == len(seqs[1])


# ---------- run_phmmer (external tool) ----------
@pytest.mark.slow
@pytest.mark.integration
def test_run_phmmer_self_hit(has_phmmer, tmp_path: Path):
  if not has_phmmer:
    pytest.skip("phmmer not available on PATH")

  # create a tiny FASTA in a tmp dir
  fasta = tmp_path / "test.fasta"
  fasta.write_text(">seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nACDEYGHIKLMNPQRSTVWY\n")

  # use first sequence as query
  query = "ACDEFGHIKLMNPQRSTVWY"
  hits = run_phmmer(query, str(fasta), evalue=10.0, cpu=1)

  assert isinstance(hits, list)
  assert any("seq1" in h or "seq2" in h for h in hits)


# ---------- run_phmmer_mafft (external tools) ----------
@pytest.mark.slow
@pytest.mark.integration
def test_run_phmmer_mafft_pipeline(has_phmmer, has_mafft, tmp_path: Path):
  if not has_phmmer:
    pytest.skip("phmmer not available on PATH")
  if not has_mafft:
    pytest.skip("mafft not available on PATH")

  # create a tiny FASTA
  fasta = tmp_path / "test.fasta"
  fasta.write_text(">seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nACDEYGHIKLMNPQRSTVWY\n>seq3\nACDEFGHIKLMNPQRSAVWY\n")

  query = "ACDEFGHIKLMNPQRSTVWY"

  names, seqs = run_phmmer_mafft(
    query=query,
    ref_db_path=str(fasta),
    size=3,
    in_name="input_sequence",
    phmmer_cpu=1,
    mafft_threads=1,
  )

  assert "input_sequence" in names
  # all sequences should be aligned => equal lengths
  assert len(set(map(len, seqs))) == 1


# ---------- run_mmseqs2 (network/API) ----------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.mmseqs2
def test_run_mmseqs2_env_unfiltered_roundtrip(chdir_tmp: Path):
  # NOTE: this hits ColabFold API; skip in fast CI
  seq = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
  outdir = Path("msa_out")
  a3m_lines, template_paths = run_mmseqs2(
    seqs=seq,
    output=str(outdir),
    database="mmseqs2_uniref_env",
    use_filter=True,
    use_templates=False,
    pairing=None,
  )
  # Files we expect in 'env' mode (uniref + bfd)
  expected = {"uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m", "pdb70.m8", "combined_1.a3m"}
  assert expected.issubset(set(os.listdir(outdir)))
  assert isinstance(a3m_lines, list) and len(a3m_lines) == 1
  assert template_paths is None


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.mmseqs2
def test_run_mmseqs2_pairing_greedy(chdir_tmp: Path):
  seq = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
  outdir = Path("msa_pair")
  a3m_lines, template_paths = run_mmseqs2(
    seqs=[seq, seq],  # pairing mode expects list
    output=str(outdir),
    use_templates=True,  # ignored in pairing mode
    pairing="greedy",
  )
  assert "pair.a3m" in set(os.listdir(outdir))
  assert "combined_1.a3m" in set(os.listdir(outdir))
  assert template_paths is None


# ---------- run_mmseqs2_modes (HH-suite + network) ----------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.mmseqs2
def test_run_mmseqs2_modes_unpaired_paired(has_hhfilter, chdir_tmp: Path):
  if not has_hhfilter:
    pytest.skip("hhfilter (HH-suite) not available on PATH")
  seqs = ["ACDEFGHIKLMNPQRSTVWY", "ACDEYGHIKLMNPQRSTVWY"]  # tiny toy pair
  outdir = "modes_out"
  run_mmseqs2_modes(
    seq=seqs,
    output=outdir,
    cov=50,
    id=90,
    max_msa=32,
    mode="unpaired_paired",
  )
  assert Path(outdir, "final.a3m").exists()
