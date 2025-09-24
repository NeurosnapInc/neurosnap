# tests/test_clusterprot.py
import types
from pathlib import Path

import numpy as np
import pytest

from neurosnap.algos.clusterprot import ClusterProt, animate_results, create_figure_plotly
from neurosnap.protein import Protein

HERE = Path(__file__).resolve().parent
CLUSTER_DIR = HERE / "files" / "proteins_clustering"


# -----------------------
# Dependency checks
# -----------------------


@pytest.fixture(scope="session")
def has_umap():
  try:
    from umap import UMAP  # noqa

    return True
  except Exception:
    return False


@pytest.fixture(scope="session")
def has_sklearn():
  try:
    from sklearn.cluster import DBSCAN  # noqa
    from sklearn.decomposition import PCA  # noqa

    return True
  except Exception:
    return False


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture(scope="session")
def cluster_files():
  assert CLUSTER_DIR.exists() and CLUSTER_DIR.is_dir(), f"Missing fixture dir: {CLUSTER_DIR}"
  files = sorted([p for p in CLUSTER_DIR.glob("*.pdb") if p.is_file()])
  # per implementation: requires at least 5 proteins
  assert len(files) >= 5, "ClusterProt requires >=5 PDBs in tests/files/proteins_clustering"
  return files


@pytest.fixture
def proteins_from_paths(cluster_files):
  return [str(p) for p in cluster_files]


@pytest.fixture
def protein_objects(cluster_files):
  return [Protein(str(p)) for p in cluster_files]


# -----------------------
# Core algorithm
# -----------------------


@pytest.mark.slow
@pytest.mark.integration
def test_clusterprot_with_paths_umap_1d(has_umap, has_sklearn, proteins_from_paths):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  res = ClusterProt(
    proteins=proteins_from_paths,  # file paths
    model=0,
    chain=None,
    umap_n_neighbors=0,  # auto
    proj_1d_algo="umap",  # 1D: UMAP
    dbscan_eps=0,  # auto
    dbscan_min_samples=0,  # auto
    eps_scale_factor=0.10,
  )
  n = len(proteins_from_paths)
  # keys present
  for k in ["proteins", "projection_1d", "projection_2d", "cluster_labels"]:
    assert k in res
  # types & lengths
  assert len(res["proteins"]) == n
  assert all(isinstance(p, Protein) for p in res["proteins"])
  assert len(res["projection_1d"]) == n
  assert len(res["projection_2d"]) == n
  assert len(res["cluster_labels"]) == n
  # shapes
  p1 = np.asarray(res["projection_1d"])
  p2 = np.asarray(res["projection_2d"])
  assert p1.ndim == 2 and p1.shape[1] == 1
  assert p2.ndim == 2 and p2.shape[1] == 2
  # labels are ints (DBSCAN, −1 allowed)
  assert all(isinstance(int(lbl), int) for lbl in res["cluster_labels"])


@pytest.mark.slow
@pytest.mark.integration
def test_clusterprot_with_objects_pca_1d(has_umap, has_sklearn, protein_objects):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  res = ClusterProt(
    proteins=protein_objects,  # Protein instances
    proj_1d_algo="pca",  # 1D: PCA
    dbscan_eps=0,  # auto
    dbscan_min_samples=0,  # auto
    eps_scale_factor=0.05,
  )
  n = len(protein_objects)
  p1 = np.asarray(res["projection_1d"])
  assert p1.shape == (n, 1)


def test_clusterprot_invalid_1d_algo_raises(proteins_from_paths, has_umap, has_sklearn):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  with pytest.raises(ValueError):
    ClusterProt(proteins=proteins_from_paths, proj_1d_algo="tsne?!")


def test_clusterprot_minimum_count_enforced(cluster_files, has_umap, has_sklearn):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  few = [str(p) for p in cluster_files[:3]]
  with pytest.raises(AssertionError):
    ClusterProt(proteins=few)


# -----------------------
# Animation (monkeypatched)
# -----------------------


@pytest.mark.slow
def test_animate_results_monkeypatched(tmp_path, proteins_from_paths, has_umap, has_sklearn, monkeypatch):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  # small run to produce results (use PCA for faster 1D if desired)
  res = ClusterProt(proteins=proteins_from_paths, proj_1d_algo="pca")

  # Patch out heavy animation bits: return a dummy object with save()
  class DummyAnim:
    def __init__(self, outpath: Path):
      self.outpath = outpath

    def save(self, path, writer=None, fps=None):
      Path(path).write_bytes(b"GIF89a")  # minimal marker so file exists

  def fake_plot_pseudo_3D(df, ax=None):
    return None  # a "frame" placeholder

  def fake_animate_pseudo_3D(fig, ax, frames, titles):
    return DummyAnim(outpath=tmp_path / "dummy.gif")

  # Patch the symbols inside the module under test
  import neurosnap.algos.clusterprot as cpmod

  monkeypatch.setattr(cpmod, "plot_pseudo_3D", fake_plot_pseudo_3D)
  monkeypatch.setattr(cpmod, "animate_pseudo_3D", fake_animate_pseudo_3D)

  out = tmp_path / "cluster_prot.gif"
  animate_results(res, animation_fpath=str(out))
  assert out.exists() and out.read_bytes().startswith(b"GIF89a")


# -----------------------
# Plotly figure (mocked)
# -----------------------


def test_create_figure_plotly_with_fake_module(monkeypatch, proteins_from_paths, has_umap, has_sklearn):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  res = ClusterProt(proteins=proteins_from_paths, proj_1d_algo="pca")

  # Build a fake plotly.express module
  class FakeFig:
    def __init__(self):
      self.updated = False
      self.shown = False

    def update_layout(self, **kwargs):
      self.updated = True

    def show(self):
      self.shown = True

  def fake_scatter(*args, **kwargs):
    return FakeFig()

  fake_px = types.ModuleType("plotly.express")
  fake_px.scatter = fake_scatter

  fake_plotly = types.ModuleType("plotly")
  fake_plotly.express = fake_px

  # Inject into sys.modules so import inside function resolves to our fake
  monkeypatch.setitem(__import__("sys").modules, "plotly", fake_plotly)
  monkeypatch.setitem(__import__("sys").modules, "plotly.express", fake_px)

  # Should run without raising and call into our fakes
  create_figure_plotly(res)  # no asserts needed; would raise if broken
