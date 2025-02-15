"""
Implementation of the ClusterProt algorithm from https://neurosnap.ai/service/ClusterProt.
ClusterProt is an algorithm for clustering proteins by their structure similarity.
"""

import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional, Union, Dict, Any

from neurosnap.log import logger
from neurosnap.protein import Protein, animate_pseudo_3D, plot_pseudo_3D

try:
  from sklearn.cluster import DBSCAN
  from sklearn.decomposition import PCA
  from umap import UMAP
except Exception as e:
  logger.critical(
    "Unable to import sklearn and umap. The ClusterProt algorithm depends on these two packages to function. Please add them to your environment using something like $ pip install umap-learn scikit-learn"
  )
  raise e


def ClusterProt(proteins: List[Union["Protein", str]], model: int = 0, chain: Optional[str] = None, proj_1d_algo: str = "umap") -> Dict[str, Any]:
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
    proteins: List of proteins to cluster, can be either neurosnap Protein objects of filepaths to proteins that will get loaded as Protein objects
    model: Model ID to for ClusterProt to use (must be consistent across all structures)
    chain: Chain ID to for ClusterProt to use (must be consistent across all structures), if not provided calculates for all chains
    proj_1d_algo: Algorithm to use for the 1D projection. Can be either ``"umap"`` or ``"pca"``

  Returns:
    A dictionary containing the results from the algorithm:

      - proteins (list<Protein>): Sorted list of all the neurosnap Proteins aligned by the reference protein.
      - projection_2d (list<list<float>>): Generated 2D projection of all the proteins.
      - cluster_labels (list<float>): List of the labels for each of the proteins.

  """
  # ensure input data is valid
  logger.debug(f"Loading {len(proteins)} for clustering")
  for i, protein in enumerate(proteins):
    if isinstance(protein, Protein):
      pass
    elif isinstance(protein, str):
      proteins[i] = Protein(protein)
    else:
      raise f"One of the provided proteins was not Neurosnap Protein object or a path to an existing PDB file.\nprotein: {protein}"

  proteins_vects = []
  logger.debug(f"Clustering {len(proteins)} proteins")

  if len(proteins) < 5:
    raise "ClusterProt requires at least 5 proteins in order to work."

  # compute distance matrices
  logger.debug("Computing distance matrices")
  prot_ref = proteins[0]
  protein_length = len(prot_ref.calculate_distance_matrix(model=model, chain=chain))
  for prot in proteins:
    dm = prot.calculate_distance_matrix()  # compute protein distance matrix
    assert (
      len(dm) == protein_length
    ), f"All proteins need to have the same number of residues. Proteins {proteins[0].title} and {prot.title} have different lengths."
    # get the upper triangle without the diagonal as a flattened vector
    triu_vect = dm[np.triu_indices(len(dm), k=1)]
    proteins_vects.append(triu_vect)

  proteins_vects = np.array(proteins_vects)

  # align all proteins
  logger.debug("Aligning all proteins")
  for prot in proteins:
    prot_ref.align(prot, model1=model, model2=model)

  # 2D projection and cluster it using DBSCAN
  logger.debug("Creating 2D projection and clustering it using DBSCAN")
  proj_2d = UMAP(n_components=2, init="random", n_neighbors=7).fit_transform(proteins_vects)
  cluster_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(proj_2d)

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
    "proteins": proteins,
    "projection_1d": proj_1d.tolist(),
    "projection_2d": proj_2d.tolist(),
    "cluster_labels": cluster_labels.tolist(),
  }


def animate_results(cp_results: Dict, animation_fpath: str="cluster_prot.gif"):
  """Animate the ClusterProt results using the aligned proteins and 1D projections.

  Parameters:
    cp_results: Results object from ClusterProt run
    animation_fpath: Output filepath for the animation of all the proteins

  """
  logger.debug("Drawing animation frames")
  fig, ax = plt.subplots()
  frames = []
  titles = []
  cp_results["proteins"]
  sorted_proteins = sorted([[prot, x] for prot, x in zip(cp_results["proteins"], cp_results["projection_1d"])], key=lambda x: x[1])
  for i, (prot, x) in enumerate(sorted_proteins, start=1):
    print(f"Creating animation frames for {i}/{len(sorted_proteins)} proteins\r", end="", flush=True)
    frame = plot_pseudo_3D(prot.df[["x", "y", "z"]], ax=ax)
    frames.append(frame)
    titles.append(f"{prot.title} ({i}/{len(sorted_proteins)})")
  print()
  ani = animate_pseudo_3D(fig, ax, frames, titles)
  ani.save(animation_fpath, writer="ffmpeg", fps=7)


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

  titles = [prot.title for prot in cp_results["proteins"]]
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
