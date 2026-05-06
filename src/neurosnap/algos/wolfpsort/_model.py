from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from neurosnap._compat import compat_dataclass


@compat_dataclass(frozen=True, slots=True)
class UniformScaler:
  """Piecewise-linear percentile scaler used by the original classifier."""

  xs: Sequence[float]
  ys: Sequence[float]

  def transform(self, value: float) -> float:
    """Scale a single feature value onto the learned uniform axis.

    Args:
      value: Raw feature value in the original feature space.

    Returns:
      Piecewise-linear transformed value in the scaled feature space.
    """
    if len(self.xs) == 1:
      return self.ys[0]
    if value <= self.xs[0]:
      index = 0
    elif value >= self.xs[-1]:
      index = len(self.xs) - 2
    else:
      index = bisect.bisect_right(self.xs, value) - 1
      if self.xs[index] == value:
        return self.ys[index]
    x0, x1 = self.xs[index], self.xs[index + 1]
    y0, y1 = self.ys[index], self.ys[index + 1]
    if x1 == x0:
      return y0
    return y0 + (value - x0) * (y1 - y0) / (x1 - x0)

  @classmethod
  def from_values(cls, values: Sequence[float], scale_max: float = 100.0) -> "UniformScaler":
    """Build a scaler from one training feature column.

    Args:
      values: Raw training values for one feature across all training examples.
      scale_max: Maximum output scale value used for the percentile mapping.

    Returns:
      :class:`UniformScaler` fitted to the supplied training values.
    """
    xs = sorted(values)
    if not xs:
      raise ValueError("Scaler requires at least one value")
    if len(xs) == 1:
      return cls([xs[0]], [0.0])
    unique_xs: List[float] = []
    unique_ys: List[float] = []
    n = len(xs)
    i = 0
    while i < n:
      x = xs[i]
      j = i
      while j < n and xs[j] == x:
        j += 1
      midrank = (i + j - 1) / 2.0
      unique_xs.append(x)
      unique_ys.append(scale_max * midrank / (n - 1))
      i = j
    return cls(unique_xs, unique_ys)


@dataclass
class ModelData:
  """Parsed WoLF PSORT model assets and cached scaled training data."""

  feature_names: List[str]
  w1: List[float]
  w2: List[float]
  k_max: int
  training_ids: List[str]
  training_classes: List[str]
  training_matrix: List[List[float]]
  utility_classes: List[str]
  utility_matrix: Dict[str, Dict[str, float]]
  scalers: List[UniformScaler]
  scaled_training_matrix: List[List[float]]

  @classmethod
  def load(cls, root: Path, organism_type: str) -> "ModelData":
    """Load one bundled organism model from package data files.

    Args:
      root: Directory containing the bundled ``.weights.json``,
        ``.training.csv``, and ``.utility.json`` files.
      organism_type: Model prefix to load, such as ``"fungi"``.

    Returns:
      Fully parsed :class:`ModelData` instance with cached scaled training rows.
    """
    feature_names, w1, w2, k_max = _read_weights(root / f"{organism_type}.weights.json")
    train_features, training_ids, training_classes, training_matrix = _read_training(root / f"{organism_type}.training.csv")
    utility_classes, utility_matrix = _read_utility(root / f"{organism_type}.utility.json")
    if feature_names != train_features:
      raise ValueError("Training and weight feature order differ")
    columns = list(zip(*training_matrix))
    # The original classifier scales each feature independently from the
    # training distribution before computing weighted similarity.
    scalers = [UniformScaler.from_values(column) for column in columns]
    return cls(
      feature_names=feature_names,
      w1=w1,
      w2=w2,
      k_max=k_max,
      training_ids=training_ids,
      training_classes=training_classes,
      training_matrix=training_matrix,
      utility_classes=utility_classes,
      utility_matrix=utility_matrix,
      scalers=scalers,
      # Cache scaled rows once so repeated predictions stay cheap.
      scaled_training_matrix=[[scaler.transform(value) for scaler, value in zip(scalers, row)] for row in training_matrix],
    )

  def transform_feature_vector(self, values: Sequence[float]) -> List[float]:
    """Apply the learned per-feature scalers to one feature vector.

    Args:
      values: Raw WoLF PSORT feature vector in bundled feature order.

    Returns:
      Scaled feature vector aligned to the model's transformed training space.
    """
    return [scaler.transform(value) for scaler, value in zip(self.scalers, values)]

  def predict(self, feature_vector: Sequence[float], include_neighbors: bool = False) -> Dict[str, object]:
    """Run the weighted kNN localization model on one feature vector.

    Args:
      feature_vector: Raw WoLF PSORT feature vector in bundled feature order.
      include_neighbors: When ``True``, include the ranked neighbors used in the
        score calculation.

    Returns:
      Dictionary containing the predicted class, ranked class scores, and best
      ``k`` values discovered during cumulative utility scoring.
    """
    query_scaled = self.transform_feature_vector(feature_vector)
    neighbors = []
    for index, (identifier, label, row) in enumerate(zip(self.training_ids, self.training_classes, self.scaled_training_matrix)):
      # The upstream similarity is a negative weighted L1 plus L2-hybrid norm,
      # so larger values are better because they are closer to zero.
      l1 = sum(weight * abs(q - r) for weight, q, r in zip(self.w1, query_scaled, row))
      l2 = sum(weight * (q - r) ** 2 for weight, q, r in zip(self.w2, query_scaled, row))
      similarity = -(l1 + math.sqrt(l2))
      neighbors.append((similarity, index, identifier, label))
    neighbors.sort(key=lambda item: (-item[0], item[1]))

    cumulative_scores = {name: 0.0 for name in self.utility_classes}
    best_scores = {name: 0.0 for name in self.utility_classes}
    best_k_by_class = {name: 0 for name in self.utility_classes}
    best_overall_class = None
    best_overall_k = 0
    best_overall_score = -math.inf

    for k, (_, _, _, neighbor_class) in enumerate(neighbors[: self.k_max], start=1):
      # WoLF PSORT keeps the best cumulative score each class reaches over the
      # full path from k=1 through k_max instead of only reporting the final k.
      for class_name in self.utility_classes:
        cumulative_scores[class_name] += self.utility_matrix[neighbor_class][class_name]
        if cumulative_scores[class_name] > best_scores[class_name]:
          best_scores[class_name] = cumulative_scores[class_name]
          best_k_by_class[class_name] = k
      top_class, top_score = max(cumulative_scores.items(), key=lambda item: (item[1], item[0]))
      if top_score > best_overall_score:
        best_overall_score = top_score
        best_overall_class = top_class
        best_overall_k = k

    ranked_classes = sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
    result: Dict[str, object] = {
      "predicted_class": best_overall_class,
      "best_overall_k": best_overall_k,
      "k_max": self.k_max,
      "scores_by_class": best_scores,
      "best_k_by_class": best_k_by_class,
      "ranked_classes": ranked_classes,
    }
    if include_neighbors:
      result["neighbors"] = [
        {"rank": rank, "similarity": similarity, "training_id": identifier, "training_class": label}
        for rank, (similarity, _, identifier, label) in enumerate(neighbors[: self.k_max], start=1)
      ]
    return result


def _read_weights(path: Path) -> Tuple[List[str], List[float], List[float], int]:
  """Read a WoLF PSORT weights JSON file.

  Args:
    path: Path to a bundled ``.weights.json`` file.

  Returns:
    Tuple of feature names, L1 weights, L2 weights, and maximum ``k``.
  """
  payload = json.loads(path.read_text())
  return payload["feature_names"], payload["w1"], payload["w2"], payload["k_max"]


def _read_training(path: Path) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
  """Read the bundled WoLF PSORT training table CSV.

  Args:
    path: Path to a bundled ``.training.csv`` file.

  Returns:
    Tuple of feature names, training IDs, training classes, and feature rows.
  """
  with path.open(newline="") as handle:
    reader = csv.reader(handle)
    header = next(reader)
    feature_names = header[2:]
    training_ids: List[str] = []
    training_classes: List[str] = []
    matrix: List[List[float]] = []
    for parts in reader:
      if not parts:
        continue
      training_ids.append(parts[0])
      training_classes.append(parts[1])
      matrix.append([float(value) for value in parts[2:]])
  return feature_names, training_ids, training_classes, matrix


def _read_utility(path: Path) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
  """Read the class-utility matrix JSON used for cumulative kNN scoring.

  Args:
    path: Path to a bundled ``.utility.json`` file.

  Returns:
    Tuple of class names and a nested class-to-class utility mapping.
  """
  payload = json.loads(path.read_text())
  return payload["class_names"], payload["matrix"]
