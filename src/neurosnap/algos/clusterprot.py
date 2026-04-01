"""
Implementation of the ClusterProt algorithm from https://neurosnap.ai/service/ClusterProt.
ClusterProt is an algorithm for clustering proteins by their structure similarity.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from neurosnap.io.pdb import parse_pdb
from neurosnap.log import logger
from neurosnap.rendering import animate_frames, render_protein_pseudo3D
from neurosnap.structure import Structure, StructureEnsemble, StructureStack, align, calculate_distance_matrix

try:
  from sklearn.cluster import DBSCAN
  from sklearn.decomposition import PCA
  from umap import UMAP
except Exception as e:
  logger.critical(
    "Unable to import sklearn and umap. The ClusterProt algorithm depends on these two packages to function. Please add them to your environment using something like $ pip install umap-learn scikit-learn"
  )
  raise e


def ClusterProt(
  proteins: List[Union[Structure, StructureEnsemble, StructureStack, str, Path]],
  model: Optional[int] = None,
  chain: Optional[str] = None,
  umap_n_neighbors: int = 0,
  proj_1d_algo: str = "umap",
  dbscan_eps: float = 0,
  dbscan_min_samples: int = 0,
  eps_scale_factor: float = 0.05,
) -> Dict[str, Any]:
  """Run the ClusterProt algorithm on some input proteins.

  Clusters proteins using their structural similarity.

  Algorithm Description:
    1. Ensure all protein structures are fully loaded
    2. Compute the distance matrices of using the alpha carbons of all the loaded proteins from the selected regions
    3. Get the flattened upper triangle of the of the distance matrices excluding the diagonal.
    4. Align all the proteins to the reference protein (optional but useful for analysis like the animation)
    5. Create the 2D projection using UMAP
    6. Create clusters for the 2D projection using DBSCAN
    7. Create the 1D projection using either UMAP or PCA (optional but useful for organizing proteins 1-dimensionally)

  Parameters:
    proteins: List of structures to cluster, as Neurosnap structure containers or PDB filepaths.
    model: Model ID for ClusterProt to use (must be consistent across all structures). Defaults to the first available model in each input.
    chain: Chain ID to for ClusterProt to use (must be consistent across all structures), if not provided calculates for all chains
    umap_n_neighbors: The ``n_neighbors`` value to provide to UMAP for the main projection. Leave as 0 to automatically calculate optimal value. Prior to the 2024-06-14 update this values was left as ``7``.
    proj_1d_algo: Algorithm to use for the 1D projection. Can be either ``"umap"`` or ``"pca"``
    dbscan_eps: The ``eps`` value to provide to DBSCAN. Leave as 0 to automatically calculate optimal value. Prior to the 2024-04-15 update this values was left as ``0.5``.
    dbscan_min_samples: The ``min_samples`` value to provide to DBSCAN. Leave as 0 to automatically calculate optimal value. Prior to the 2024-04-15 update this values was left as ``5``.
    eps_scale_factor: Fraction of the 2D data's diagonal range used to set DBSCAN's eps. Recommended: 0.05-0.10 for larger datasets or finer clusters; 0.15 for smaller datasets or broader clustering.

  Returns:
    A dictionary containing the results from the algorithm:

      - structures (list): Sorted list of all the Neurosnap structures aligned by the reference structure.
      - titles (list<str>): Display labels for each structure.
      - projection_2d (list<list<float>>): Generated 2D projection of all the structures.
      - cluster_labels (list<float>): List of the labels for each of the structures.

  """
  structures = []
  titles = []
  logger.debug(f"Loading {len(proteins)} structures for clustering")
  for index, structure in enumerate(proteins, start=1):
    if isinstance(structure, (Structure, StructureEnsemble, StructureStack)):
      structures.append(structure)
      titles.append(str(structure.metadata.get("title", f"structure_{index}")))
    else:
      structure_path = Path(structure)
      structures.append(parse_pdb(structure_path, return_type="ensemble"))
      titles.append(structure_path.stem)

  proteins_vects = []
  logger.debug(f"Clustering {len(structures)} structures")

  assert len(structures) >= 5, "ClusterProt requires at least 5 structures in order to work."

  # compute distance matrices
  logger.debug("Computing distance matrices")
  prot_ref = structures[0]
  protein_length = len(calculate_distance_matrix(prot_ref, model=model, chain=chain))
  for index, prot in enumerate(structures):
    dm = calculate_distance_matrix(prot, model=model, chain=chain)
    assert len(dm) == protein_length, (
      f"All structures need to have the same number of residues. Structures {titles[0]} and {titles[index]} have different lengths."
    )
    # get the upper triangle without the diagonal as a flattened vector
    triu_vect = dm[np.triu_indices(len(dm), k=1)]
    proteins_vects.append(triu_vect)

  proteins_vects = np.array(proteins_vects)

  # align all proteins
  logger.debug("Aligning all structures")
  for prot in structures:
    align(prot_ref, prot, model1=model, model2=model)

  # 2D projection and cluster it using DBSCAN
  if umap_n_neighbors == 0:
    umap_n_neighbors = min(15, max(5, round(len(structures) * 0.01)))
  logger.debug(f"Creating 2D projection using UMAP (n_neighbors={umap_n_neighbors})")
  proj_2d = UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=0.04).fit_transform(proteins_vects)

  # cluster using DBSCAN and optionally calculate ideal DBSCAN params
  logger.debug("Creating cluster labels using DBSCAN")
  if dbscan_eps <= 0:
    xmin, ymin = np.min(proj_2d, axis=0)
    xmax, ymax = np.max(proj_2d, axis=0)
    data_range = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    dbscan_eps = eps_scale_factor * data_range  # n% of diagonal range
    dbscan_eps = max(dbscan_eps, 1e-4)  # clip value to ensure it doesn't get too small

  if dbscan_min_samples <= 0:
    dbscan_min_samples = int(np.log(len(structures))) + 1

  # calculate DBSCAN labels
  cluster_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(proj_2d)

  # compute 1D projection for animation
  logger.debug("Computing 1D projection for animation")
  proj_1d_algo = proj_1d_algo.lower()
  if proj_1d_algo == "umap":
    proj_1d = UMAP(n_components=1, init="random").fit_transform(proteins_vects)
  elif proj_1d_algo == "pca":
    proj_1d = PCA(n_components=1).fit_transform(proteins_vects)
  else:
    raise ValueError("Invalid 1D projection method provided for the animation. Must be either umap or pca.")

  # return results
  return {
    "structures": structures,
    "titles": titles,
    "model_id": model,
    "projection_1d": proj_1d.tolist(),
    "projection_2d": proj_2d.tolist(),
    "cluster_labels": cluster_labels.tolist(),
  }


def animate_results(cp_results: Dict, animation_fpath: str = "cluster_prot.gif"):
  """Animate the ClusterProt results using the aligned proteins and 1D projections.

  Parameters:
    cp_results: Results object from ClusterProt run
    animation_fpath: Output filepath for the animation of all the proteins

  """
  structures = cp_results["structures"]
  titles = cp_results.get("titles", [f"structure_{index + 1}" for index in range(len(structures))])
  model_id = cp_results.get("model_id")
  projection_1d = np.squeeze(np.asarray(cp_results["projection_1d"], dtype=float))
  order = np.argsort(projection_1d)
  frames = []
  subtitles = []
  total = len(order)
  for i, idx in enumerate(tqdm(order, desc="Rendering frames", unit="frame"), start=1):
    frames.append(render_protein_pseudo3D(structures[idx], model=model_id, image_size=(800, 580)))
    subtitles.append(f"{titles[idx]} ({i}/{total})")
  animate_frames(frames, animation_fpath, title="ClusterProt Animation", subtitles=subtitles, interval=150, repeat=True)


def create_figure_plotly(cp_results: Dict):
  """Create a scatter plot of the 2D projection from ClusterProt using plotly express.

  NOTE: The plotly package will need to be installed for this

  Parameters:
    cp_results: Results object from ClusterProt run

  """
  try:
    import plotly.express as px
  except Exception as e:
    logger.critical(
      "Unable to import plotly. This function depends on plotly express. Please add it to your environment using something like $ pip install plotly"
    )
    raise e

  titles = cp_results.get("titles", [f"structure_{index + 1}" for index in range(len(cp_results["structures"]))])
  fig = px.scatter(
    cp_results["projection_2d"],
    x=0,
    y=1,
    # z=2,
    title="ClusterProt: Protein Clustering by Structural Similarity",
    labels={"color": "Conformation Clusters", "0": "x", "1": "y", "2": "z"},
    color=["outlier" if x == -1 else f"cluster {x}" for x in cp_results["cluster_labels"]],
    hover_name=titles,
    # text=titles,
  )
  fig.update_layout(xaxis_title="", yaxis_title="")
  fig.show()
