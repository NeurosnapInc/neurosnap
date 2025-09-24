# tests/test_kluster.py
from pathlib import Path

import numpy as np
import pytest

# Import module under test
from neurosnap.algos.kluster import (
  check_alignment_tool,
  cluster_projection,
  compute_distance_matrix,
  reduce_dimensions,
  run_alignment,
  visualize_projection,
  visualize_projection_interactive,
)

HERE = Path(__file__).resolve().parent


# -----------------------
# Dependency checks
# -----------------------


@pytest.fixture(scope="session")
def has_sklearn():
  try:
    import sklearn  # noqa: F401

    return True
  except Exception:
    return False


@pytest.fixture(scope="session")
def has_umap():
  try:
    from umap import UMAP  # noqa: F401

    return True
  except Exception:
    return False


@pytest.fixture(scope="session")
def has_plotly():
  try:
    import plotly  # noqa: F401
    import plotly.express as px  # noqa: F401
    import plotly.graph_objects as go  # noqa: F401

    return True
  except Exception:
    return False


# -----------------------
# check_alignment_tool
# -----------------------


def test_check_alignment_tool_happy(monkeypatch):
  # Simulate a valid executable path
  def fake_which(name):
    return "/usr/local/bin/TMalign"

  monkeypatch.setattr("neurosnap.algos.kluster.shutil.which", fake_which)
  monkeypatch.setattr("neurosnap.algos.kluster.os.access", lambda p, mode: True)
  out = check_alignment_tool("TMalign")
  assert out.endswith("TMalign")


def test_check_alignment_tool_missing(monkeypatch):
  monkeypatch.setattr("neurosnap.algos.kluster.shutil.which", lambda name: None)
  with pytest.raises(FileNotFoundError):
    check_alignment_tool("USalign")


def test_check_alignment_tool_not_executable(monkeypatch):
  monkeypatch.setattr("neurosnap.algos.kluster.shutil.which", lambda name: "/bin/tool")
  monkeypatch.setattr("neurosnap.algos.kluster.os.access", lambda p, mode: False)
  with pytest.raises(FileNotFoundError):
    check_alignment_tool("tool")


# -----------------------
# run_alignment (mocked subprocess)
# -----------------------


def test_run_alignment_parsing(monkeypatch, tmp_path):
  # Short-circuit the tool check
  monkeypatch.setattr("neurosnap.algos.kluster.check_alignment_tool", lambda name: "/bin/TMalign")

  # Fake output containing both TM-score and RMSD tokens
  fake_out = "Some header\nTM-score= 0.6789 (normalized) blah\nRMSD= 2.34 something\n"

  def fake_check_output(cmd, text=True, stderr=None):
    assert cmd[:1] == ["/bin/TMalign"]
    return fake_out

  monkeypatch.setattr("neurosnap.algos.kluster.subprocess.check_output", fake_check_output)

  feats = run_alignment("a.pdb", "b.pdb", alignment_tool="TMalign", use_tmscore=True, use_rmsd=True)
  assert feats["tma_score"] == pytest.approx(0.6789, rel=1e-6)
  assert feats["rmsd"] == pytest.approx(2.34, rel=1e-6)


def test_run_alignment_handles_fail(monkeypatch):
  monkeypatch.setattr("neurosnap.algos.kluster.check_alignment_tool", lambda name: "/bin/TMalign")

  def boom(cmd, text=True, stderr=None):
    raise Exception("boom")

  monkeypatch.setattr("neurosnap.algos.kluster.subprocess.check_output", boom)
  feats = run_alignment("a.pdb", "b.pdb", alignment_tool="TMalign", use_tmscore=True, use_rmsd=True)
  assert feats == {}


# -----------------------
# compute_distance_matrix (mock Pool + alignment)
# -----------------------


class FakePool:
  """Minimal serial Pool stub with context manager and imap."""

  def __init__(self, processes=None):
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False

  def imap(self, fn, iterable):
    for item in iterable:
      yield fn(item)


def test_compute_distance_matrix_builds_features(monkeypatch):
  # Proteins dict: ids -> fake file paths
  proteins = {"P1": "p1.pdb", "P2": "p2.pdb", "P3": "p3.pdb"}

  # Force serial "Pool"
  monkeypatch.setattr("neurosnap.algos.kluster.Pool", FakePool)

  # Fake pairwise alignment results: return both tma_score and rmsd varying by pair index
  def fake_run_alignment_pair(args):
    # args: (proteins, protein_ids, combo, alignment_tool, use_tmscore, use_rmsd)
    _, protein_ids, combo, *_ = args
    i, j = combo
    # contrive deterministic values
    return {"tma_score": 0.5 + 0.1 * (i + j), "rmsd": 1.0 * (i + j)}

  monkeypatch.setattr("neurosnap.algos.kluster._run_alignment_pair", fake_run_alignment_pair)

  mat, ids = compute_distance_matrix(
    proteins=proteins,
    alignment_tool="TMalign",
    use_tmscore=True,
    use_rmsd=True,
    num_processes=1,
  )
  # Expect n x (n*m) where m=2 features, n=len(proteins)=3
  assert mat.shape == (3, 3 * 2)
  assert ids == sorted(proteins.keys())
  # Values should be finite (some zeros on diagonal slots OK)
  assert np.isfinite(mat).all()


# -----------------------
# reduce_dimensions
# -----------------------


@pytest.mark.slow
@pytest.mark.integration
def test_reduce_dimensions_umap_and_pca(has_umap, has_sklearn):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  # Fake flattened feature matrix (n, n*m) — small but non-trivial
  rng = np.random.default_rng(0)
  mat = rng.normal(size=(10, 40))

  proj_umap_2d = reduce_dimensions(mat, method="UMAP", dimensions=2, scale=True, n_neighbors=5, min_dist=0.05)
  proj_pca_3d = reduce_dimensions(mat, method="PCA", dimensions=3, scale=False)

  assert proj_umap_2d.shape == (10, 2)
  assert proj_pca_3d.shape == (10, 3)


@pytest.mark.slow
@pytest.mark.integration
def test_reduce_dimensions_tsne(has_sklearn):
  if not has_sklearn:
    pytest.skip("Requires scikit-learn")

  rng = np.random.default_rng(1)
  mat = rng.normal(size=(8, 24))
  proj_tsne = reduce_dimensions(mat, method="TSNE", dimensions=2, scale=True, perplexity=5.0)
  assert proj_tsne.shape == (8, 2)


# -----------------------
# cluster_projection
# -----------------------


def test_cluster_projection_errors_and_defaults():
  # wrong shape
  with pytest.raises(ValueError):
    cluster_projection(np.array([1, 2, 3]))
  # too few points
  with pytest.raises(ValueError):
    cluster_projection(np.array([[0.0, 0.0]]))

  # Works on simple grid; exercise auto eps/min_samples
  pts = np.array([[0.0, 0.0], [0.01, 0.01], [5.0, 5.0], [5.02, 5.01]])
  labels = cluster_projection(pts, eps=None, min_samples=None, scaling_factor=0.2)
  assert labels.shape == (4,)
  # DBSCAN produces -1 for noise or 0/1 for clusters; we just check type/size
  assert all(isinstance(int(l), int) for l in labels)


# -----------------------
# visualize_projection (matplotlib)
# -----------------------


def test_visualize_projection_static_png(tmp_path):
  proj = np.array([[0, 0], [1, 1], [2, 0.5]])
  labels = np.array([0, 0, -1])
  out = tmp_path / "proj.png"
  visualize_projection(proj, protein_ids=["A", "B", "C"], output_file=str(out), method="UMAP", dimensions=2, cluster_labels=labels)
  assert out.exists() and out.stat().st_size > 0


def test_visualize_projection_static_png_3d(tmp_path):
  proj = np.array([[0, 0, 0], [1, 1, 1], [2, 0.5, 1.5]])
  labels = np.array([1, 1, -1])
  out = tmp_path / "proj3d.png"
  visualize_projection(proj, protein_ids=["A", "B", "C"], output_file=str(out), method="PCA", dimensions=3, cluster_labels=labels)
  assert out.exists() and out.stat().st_size > 0


# -----------------------
# visualize_projection_interactive (plotly)
# -----------------------


@pytest.mark.slow
def test_visualize_projection_interactive_returns_figure(has_plotly):
  if not has_plotly:
    pytest.skip("Requires plotly")

  proj = np.array([[0, 0], [1, 1], [2, 0.5], [3, 1.5]])
  labels = np.array([0, 1, -1, 1])
  fig = visualize_projection_interactive(proj, ["p1", "p2", "p3", "p4"], method="UMAP", cluster_labels=labels)
  # basic sanity
  import plotly.graph_objects as go

  assert isinstance(fig, go.Figure)
  # should have as many traces as there are unique clusters
  assert len(fig.data) == len(np.unique(labels))