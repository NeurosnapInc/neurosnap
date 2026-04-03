# tests/test_clusterprot.py
import types
from pathlib import Path

import numpy as np
import pytest

from neurosnap.algos.clusterprot import ClusterProt, animate_results, create_figure_plotly
from neurosnap.io.pdb import parse_pdb
from neurosnap.structure import Structure, StructureEnsemble

TESTS_DIR = Path(__file__).resolve().parents[1]
CLUSTER_DIR = TESTS_DIR / "files" / "proteins_clustering"


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
def structure_list(cluster_files):
  return [parse_pdb(str(p), return_type="ensemble").first() for p in cluster_files]


@pytest.fixture
def structure_ensemble(cluster_files):
  ensemble = StructureEnsemble()
  for structure in [parse_pdb(str(p), return_type="ensemble").first() for p in cluster_files]:
    ensemble.append(structure)
  return ensemble


# -----------------------
# Core algorithm
# -----------------------


@pytest.mark.slow
@pytest.mark.integration
def test_clusterprot_with_structure_list_umap_1d(has_umap, has_sklearn, structure_list):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  res = ClusterProt(
    proteins=structure_list,
    chain=None,
    umap_n_neighbors=0,  # auto
    proj_1d_algo="umap",  # 1D: UMAP
    dbscan_eps=0,  # auto
    dbscan_min_samples=0,  # auto
    eps_scale_factor=0.10,
  )
  n = len(structure_list)
  # keys present
  for k in ["structures", "titles", "projection_1d", "projection_2d", "cluster_labels"]:
    assert k in res
  # types & lengths
  assert len(res["structures"]) == n
  assert len(res["titles"]) == n
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
def test_clusterprot_with_ensemble_pca_1d(has_umap, has_sklearn, structure_ensemble):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  res = ClusterProt(
    proteins=structure_ensemble,
    proj_1d_algo="pca",  # 1D: PCA
    dbscan_eps=0,  # auto
    dbscan_min_samples=0,  # auto
    eps_scale_factor=0.05,
  )
  n = len(structure_ensemble)
  assert all(isinstance(s, Structure) for s in res["structures"])
  p1 = np.asarray(res["projection_1d"])
  assert p1.shape == (n, 1)


def test_clusterprot_invalid_1d_algo_raises(structure_list, has_umap, has_sklearn):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  with pytest.raises(ValueError):
    ClusterProt(proteins=structure_list, proj_1d_algo="tsne?!")


def test_clusterprot_minimum_count_enforced(cluster_files, has_umap, has_sklearn):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  few = [parse_pdb(str(p), return_type="ensemble").first() for p in cluster_files[:3]]
  with pytest.raises(AssertionError):
    ClusterProt(proteins=few)


def test_clusterprot_rejects_non_structure_list_inputs():
  with pytest.raises(TypeError):
    ClusterProt(proteins=["not", "structures"])


# -----------------------
# Animation (monkeypatched)
# -----------------------


@pytest.mark.slow
def test_animate_results_monkeypatched(tmp_path, structure_list, has_umap, has_sklearn, monkeypatch):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  # small run to produce results (use PCA for faster 1D if desired)
  res = ClusterProt(proteins=structure_list, proj_1d_algo="pca")

  # Patch rendering + animation to avoid heavy work
  import neurosnap.algos.clusterprot as cpmod

  def fake_render_structure_pseudo3D(structure, **kwargs):
    return np.zeros((4, 4, 3), dtype=np.uint8)

  def fake_animate_frames(frames, output_fpath, **kwargs):
    Path(output_fpath).write_bytes(b"GIF89a")  # minimal marker so file exists

  monkeypatch.setattr(cpmod, "render_structure_pseudo3D", fake_render_structure_pseudo3D)
  monkeypatch.setattr(cpmod, "animate_frames", fake_animate_frames)

  out = tmp_path / "cluster_prot.gif"
  animate_results(res, animation_fpath=str(out))
  assert out.exists() and out.read_bytes().startswith(b"GIF89a")


# -----------------------
# Plotly figure (mocked)
# -----------------------


def test_create_figure_plotly_with_fake_module(monkeypatch, structure_list, has_umap, has_sklearn):
  if not (has_umap and has_sklearn):
    pytest.skip("Requires umap-learn and scikit-learn")

  res = ClusterProt(proteins=structure_list, proj_1d_algo="pca")

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
