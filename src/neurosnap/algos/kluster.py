"""
Implementation of the Kluster algorithm by Danial Gharaie.

This clustering algorithm is adapted from:
Amani, K., Shivnauth, V., & Castroverde, C. D. M. (2023). CBP60‐DB: An AlphaFold‐predicted plant kingdom‐wide database of the CALMODULIN‐BINDING PROTEIN 60 protein family with a novel structural clustering algorithm. Plant Direct, 7(7). https://doi.org/10.1002/pld3.509
"""

import math
import os
import re
import subprocess
from itertools import combinations_with_replacement as cwr
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP


def check_alignment_tool(tool_name: str) -> str:
  """Check if the specified alignment tool exists and is executable."""
  tool_path = os.path.join(os.path.dirname(__file__), "bin", tool_name)
  if not os.path.exists(tool_path) or not os.access(tool_path, os.X_OK):
    raise FileNotFoundError(f"{tool_name} executable not found at {tool_path}")
  return tool_path


def run_alignment(pdb_f1: str, pdb_f2: str, alignment_tool: str, use_tmscore: bool, use_rmsd: bool) -> Dict[str, Any]:
  """Run structural alignment and extract features."""
  features = {}
  cmd = [
    check_alignment_tool(alignment_tool),
    pdb_f1,
    pdb_f2,
  ]
  try:
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    print(f"Alignment failed: {str(e)}")
    return features
  except Exception as e:
    print(f"Unexpected error: {str(e)}")
    return features

  if use_tmscore:
    if m := re.search(r"TM-score=\s*([0-9.]+)", output):
      features["tma_score"] = float(m.group(1))
  if use_rmsd:
    if m := re.search(r"RMSD=\s*([0-9.]+)", output):
      features["rmsd"] = float(m.group(1))

  return features


def _run_alignment_pair(args):
  """Helper function for parallel alignment."""
  proteins, protein_ids, combo, alignment_tool, use_tmscore, use_rmsd = args
  return run_alignment(
    proteins[protein_ids[combo[0]]],
    proteins[protein_ids[combo[1]]],
    alignment_tool,
    use_tmscore,
    use_rmsd,
  )


def compute_distance_matrix(
  proteins: Dict[str, str],
  alignment_tool: str,
  use_tmscore: bool,
  use_rmsd: bool,
  num_processes: int,
) -> Tuple[np.ndarray, List[str]]:
  """Compute pairwise distance matrix using multiprocessing.

  Returns:
      tuple: A tuple containing:
          - np.ndarray: A flattened feature matrix of shape (n, n*m) where:
              n is the number of proteins
              m is the number of features (TM-score and/or RMSD)
          - List[str]: Sorted list of protein IDs corresponding to matrix rows/columns
  """
  protein_ids = sorted(proteins.keys())
  combos = list(cwr(range(len(protein_ids)), 2))

  # Prepare arguments for parallel processing
  parallel_args = [(proteins, protein_ids, combo, alignment_tool, use_tmscore, use_rmsd) for combo in combos]

  with Pool(processes=num_processes) as pool:
    scores = list(
      tqdm(
        pool.imap(_run_alignment_pair, parallel_args),
        total=len(combos),
        desc="Aligning structures",
      )
    )

  # Build feature tensors
  n = len(protein_ids)
  features = []

  if use_tmscore:
    tm_matrix = np.zeros((n, n))
    for idx, (i, j) in enumerate(combos):
      if "tma_score" in scores[idx]:
        tm_matrix[i, j] = tm_matrix[j, i] = scores[idx]["tma_score"]
    features.append(tm_matrix)

  if use_rmsd:
    rmsd_matrix = np.zeros((n, n))
    for idx, (i, j) in enumerate(combos):
      if "rmsd" in scores[idx]:
        rmsd_matrix[i, j] = rmsd_matrix[j, i] = scores[idx]["rmsd"]
    features.append(rmsd_matrix)

  # Stack features into a tensor and flatten
  feature_tensor = np.stack(features, axis=-1)  # Shape: (n, n, m)
  flattened_matrix = feature_tensor.reshape(n, -1)  # Shape: (n, n*m)

  return flattened_matrix, protein_ids


def reduce_dimensions(
  matrix: np.ndarray,
  method: str,
  dimensions: int,
  scale: bool,
  perplexity: float = 30.0,
  n_neighbors: int = 15,
  min_dist: float = 0.1,
) -> np.ndarray:
  """Perform dimensionality reduction on the flattened feature matrix.

  Args:
      matrix: Flattened feature matrix of shape (n, n*m)
      method: Dimensionality reduction method (UMAP, TSNE, or PCA)
      dimensions: Output dimensions (2 or 3)
      scale: Whether to scale features before reduction
      perplexity: t-SNE perplexity parameter
      n_neighbors: UMAP n_neighbors parameter
      min_dist: UMAP min_dist parameter

  Returns:
      np.ndarray: Reduced dimensional representation of shape (n, dimensions)
  """
  import warnings

  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", category=UserWarning)

  # Handle missing values
  imputer = SimpleImputer(strategy="mean")
  matrix_imputed = imputer.fit_transform(matrix)

  # Scale features if requested
  matrix_processed = matrix_imputed
  if scale:
    scaler = StandardScaler()
    matrix_processed = scaler.fit_transform(matrix_imputed)

  # Perform dimensionality reduction
  if method == "TSNE":
    proj = TSNE(
      n_components=dimensions,
      perplexity=perplexity,
      init="random",
    ).fit_transform(matrix_processed)
  elif method == "PCA":
    proj = PCA(
      n_components=dimensions,
    ).fit_transform(matrix_processed)
  elif method == "UMAP":
    proj = UMAP(
      n_components=dimensions,
      n_neighbors=n_neighbors,
      min_dist=min_dist,
      init="random",
    ).fit_transform(matrix_processed)

  return proj


def cluster_projection(
  proj: np.ndarray,
  eps: float | None = None,
  min_samples: int | None = None,
  scaling_factor: float = 0.05,
  eps_floor: float = 1e-4,
) -> np.ndarray:
  """
  Cluster the projection using DBSCAN and raise error on too small input.

  Expects proj to be the two dimensional output of reduce_dimensions
  array shape is (n_samples, n_dims)

  Args:
      proj: Array with shape (n_samples, n_dims)
      eps: If provided use directly otherwise estimate as scaling_factor times the range
      min_samples: If provided use directly otherwise estimate as max(1, int(log(n_samples)) plus one)
      scaling_factor: Fraction of the projection range for eps
      eps_floor: Minimum eps to avoid zero when all points are the same

  Returns:
      labels: Cluster labels with negative one for noise

  Raises:
      ValueError: If proj is not a two dimensional array or has fewer than two samples
  """
  # verify the input is two dimensional
  if proj.ndim != 2:
    raise ValueError(f"Expected two dimensional input from reduce_dimensions got shape {proj.shape}")

  n_points, n_dims = proj.shape

  # require at least two samples
  if n_points < 2:
    raise ValueError(f"Need at least two samples to cluster got {n_points}")

  # estimate eps if not provided
  if eps is None:
    if n_dims == 1:
      data_range = float(np.ptp(proj))
    else:
      ranges = np.ptp(proj, axis=0)
      data_range = float(np.linalg.norm(ranges))
    final_eps = scaling_factor * data_range
    final_eps = max(final_eps, eps_floor)
  else:
    final_eps = eps

  # estimate min samples if not provided
  if min_samples is None:
    final_min = max(1, int(math.log(n_points)) + 1)
  else:
    final_min = max(1, min_samples)

  # fit DBSCAN and return labels
  clustering = DBSCAN(eps=final_eps, min_samples=final_min).fit(proj)
  return clustering.labels_


def visualize_projection(
  proj: np.ndarray,
  protein_ids: List[str],
  output_file: str,
  method: str,
  dimensions: int,
  cluster_labels: np.ndarray,
) -> None:
  """Generate 2D/3D visualization of the projection with cluster coloring."""
  # Set up colors for clusters
  unique_clusters = np.unique(cluster_labels)
  n_clusters = len(unique_clusters)

  # Get colors from tableau palette, fall back to tab20 if more needed
  colors = list(mcolors.TABLEAU_COLORS.values())
  if n_clusters > len(colors):
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

  # Create figure with appropriate size
  plt.figure(figsize=(12, 8))

  if dimensions == 2:
    # Plot each cluster
    for i, cluster in enumerate(unique_clusters):
      mask = cluster_labels == cluster
      if cluster == -1:
        # Plot outlier points in gray
        plt.scatter(proj[mask, 0], proj[mask, 1], c="gray", label="Outlier", alpha=0.5)
      else:
        plt.scatter(
          proj[mask, 0],
          proj[mask, 1],
          c=[colors[i]],
          label=f"Cluster {cluster}",
          alpha=0.7,
        )

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
  else:  # 3D plot
    ax = plt.axes(projection="3d")
    for i, cluster in enumerate(unique_clusters):
      mask = cluster_labels == cluster
      if cluster == -1:
        # Plot Outlier points in gray
        ax.scatter(
          proj[mask, 0],
          proj[mask, 1],
          proj[mask, 2],
          c="gray",
          label="Outlier",
          alpha=0.5,
        )
      else:
        ax.scatter(
          proj[mask, 0],
          proj[mask, 1],
          proj[mask, 2],
          c=[colors[i]],
          label=f"Cluster {cluster}",
          alpha=0.7,
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

  plt.title(f"{method} projection of protein structures\n({n_clusters - 1} clusters)")

  # Add legend outside the plot
  plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

  # Adjust layout to prevent legend cutoff
  plt.tight_layout()

  # Save with high DPI for better quality
  plt.savefig(output_file, dpi=300, bbox_inches="tight")
  plt.close()


def visualize_projection_interactive(
  proj: np.ndarray,
  protein_ids: List[str],
  method: str,
  cluster_labels: np.ndarray,
  width: int = 900,
  height: int = 600,
) -> go.Figure:
  """Generate interactive visualization of the projection using Plotly.

  This function is designed for use in Jupyter notebooks and returns a Plotly figure
  that can be displayed with rich interactive features like zooming, panning, and
  hovering information.

  Args:
      proj: Projection matrix of shape (n, dimensions)
      protein_ids: List of protein identifiers
      method: Name of dimensionality reduction method used
      cluster_labels: Cluster assignments from DBSCAN
      width: Width of the plot in pixels
      height: Height of the plot in pixels

  Returns:
      go.Figure: Interactive Plotly figure

  Example:
      >>> fig = visualize_projection_interactive(proj, protein_ids, 'UMAP', cluster_labels)
      >>> fig.show()  # Display in notebook
  """

  # Get unique clusters and set up colors
  unique_clusters = np.unique(cluster_labels)
  n_clusters = len(unique_clusters)

  # Get a qualitative color sequence
  colors = px.colors.qualitative.Set3[:n_clusters]

  # Create figure
  fig = go.Figure()

  # Add traces for each cluster
  for i, cluster in enumerate(unique_clusters):
    mask = cluster_labels == cluster
    cluster_points = proj[mask]
    cluster_ids = np.array(protein_ids)[mask]

    if cluster == -1:
      # outlier points in gray
      color = "gray"
      name = "Outlier"
    else:
      color = colors[i]
      name = f"Cluster {cluster}"

    if proj.shape[1] == 3:
      # 3D scatter plot
      fig.add_trace(
        go.Scatter3d(
          x=cluster_points[:, 0],
          y=cluster_points[:, 1],
          z=cluster_points[:, 2],
          mode="markers",
          marker=dict(size=6, color=color, opacity=0.7),
          text=cluster_ids,  # This will show on hover
          name=name,
          hovertemplate=(
            "Protein: %{text}<br>" "x: %{x:.3f}<br>" "y: %{y:.3f}<br>" "z: %{z:.3f}<br>" "<extra></extra>"  # This removes the secondary box
          ),
        )
      )
    else:
      # 2D scatter plot
      fig.add_trace(
        go.Scatter(
          x=cluster_points[:, 0],
          y=cluster_points[:, 1],
          mode="markers",
          marker=dict(size=8, color=color, opacity=0.7),
          text=cluster_ids,  # This will show on hover
          name=name,
          hovertemplate=(
            "Protein: %{text}<br>" "x: %{x:.3f}<br>" "y: %{y:.3f}<br>" "<extra></extra>"  # This removes the secondary box
          ),
        )
      )

  # Update layout
  layout_args = dict(
    title=dict(
      text=f"{method} projection of protein structures<br>({n_clusters - 1} clusters)",
      x=0.5,  # Center the title
      y=0.95,
    ),
    width=width,
    height=height,
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, title=dict(text="Clusters")),
    margin=dict(l=0, r=0, t=50, b=0),  # Tight margins
    hovermode="closest",
  )

  if proj.shape[1] == 3:
    layout_args.update(
      scene=dict(
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        zaxis_title="Component 3",
      )
    )
  else:
    layout_args.update(
      xaxis_title="Component 1",
      yaxis_title="Component 2",
    )

  fig.update_layout(**layout_args)

  return fig
