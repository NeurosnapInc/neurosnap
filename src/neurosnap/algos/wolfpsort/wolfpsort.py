"""Structured WoLF PSORT localization prediction API.

This submodule provides the public Python interface for the WoLF PSORT-style
localization workflow bundled in :mod:`neurosnap.algos.wolfpsort`. It exposes
helpers for computing the model feature vector and for running the bundled
fungi, animal, and plant localization models with dictionary or DataFrame
outputs.

This implementation was developed as a distinct Python reimplementation for the
academic community, while drawing technical reference and attribution from the
original WoLF PSORT project by Paul Horton and Kenta Nakai. The referenced
project materials consulted during development are available from the public
WoLF PSORT source distribution rehosted at:

  https://github.com/fmaguire/WoLFPSort

That distribution includes the historical PSORT / WoLF PSORT command-line code,
model assets, and accompanying project notices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pandas as pd

from ._features import FEATURE_NAMES, FeatureExtractor, records_from_sequence_iterator
from ._model import ModelData


class WoLFPSortPredictor:
  """Pure Python WoLF PSORT port with structured outputs."""

  VALID_ORGANISMS = {"fungi", "animal", "plant"}

  def __init__(self, organism_type: str = "fungi") -> None:
    """Initialize a WoLF PSORT predictor for one bundled organism model.

    Args:
      organism_type: Bundled model to use. Supported values are ``"fungi"``,
        ``"animal"``, and ``"plant"``.

    Returns:
      None. The predictor is initialized in place.
    """
    if organism_type not in self.VALID_ORGANISMS:
      raise ValueError(f"organism_type must be one of {sorted(self.VALID_ORGANISMS)}")
    self.organism_type = organism_type
    self._feature_extractor = FeatureExtractor()
    # Keep the model files package-local so the port is independent of the
    # temporary upstream checkout once installed.
    data_root = Path(__file__).resolve().parent / "data"
    self._model = ModelData.load(data_root, organism_type)
    if self._model.feature_names != FEATURE_NAMES:
      raise ValueError("Bundled feature names do not match the ported extractor")

  def compute_features(self, sequences: Iterator[Tuple[str, str]]) -> List[Dict[str, object]]:
    """Compute WoLF PSORT features for one or more protein sequences.

    Args:
      sequences: Iterator yielding ``(identifier, sequence)`` tuples.

    Returns:
      List of dictionaries containing ``id`` and every WoLF PSORT feature used
      by the bundled models.
    """
    records = records_from_sequence_iterator(sequences)
    rows = []
    for record in records:
      features = self._feature_extractor.compute(record)
      # Keep the feature payload flat so it converts cleanly into both JSON-like
      # dicts and pandas rows.
      rows.append(
        {
          "id": record.identifier,
          **features,
        }
      )
    return rows

  def compute_features_dataframe(self, sequences: Iterator[Tuple[str, str]]) -> pd.DataFrame:
    """Compute WoLF PSORT features and return them as a DataFrame.

    Args:
      sequences: Iterator yielding ``(identifier, sequence)`` tuples.

    Returns:
      DataFrame with one row per sequence and one column per feature.
    """
    rows = self.compute_features(sequences)
    columns = ["id", *FEATURE_NAMES]
    return pd.DataFrame(rows, columns=columns)

  def predict(self, sequences: Iterator[Tuple[str, str]], include_features: bool = False, include_neighbors: bool = False) -> List[Dict[str, object]]:
    """Predict localization scores for one or more protein sequences.

    Args:
      sequences: Iterator yielding ``(identifier, sequence)`` tuples.
      include_features: When ``True``, attach the computed feature dictionary to
        each prediction record.
      include_neighbors: When ``True``, include the ranked training neighbors
        used during kNN scoring.

    Returns:
      List of dictionaries containing the predicted class, ranked class scores,
      best ``k`` value, and optional feature / neighbor details.
    """
    records = records_from_sequence_iterator(sequences)
    results: List[Dict[str, object]] = []
    for record in records:
      features = self._feature_extractor.compute(record)
      feature_vector = [features[name] for name in FEATURE_NAMES]
      prediction = self._model.predict(feature_vector, include_neighbors=include_neighbors)
      row: Dict[str, object] = {
        "id": record.identifier,
        "predicted_class": prediction["predicted_class"],
        "ranked_classes": prediction["ranked_classes"],
        "scores_by_class": prediction["scores_by_class"],
        "best_overall_k": prediction["best_overall_k"],
      }
      if include_features:
        row["features"] = features
      if include_neighbors:
        row["neighbors"] = prediction["neighbors"]
      results.append(row)
    return results

  def predict_dataframe(self, sequences: Iterator[Tuple[str, str]], include_features: bool = False) -> pd.DataFrame:
    """Predict localization scores and return a tabular summary.

    Args:
      sequences: Iterator yielding ``(identifier, sequence)`` tuples.
      include_features: When ``True``, expand the computed features into
        additional DataFrame columns.

    Returns:
      DataFrame with one row per sequence plus the top prediction metadata and
      full score dictionary.
    """
    rows = []
    for prediction in self.predict(sequences, include_features=include_features, include_neighbors=False):
      ranked = prediction["ranked_classes"]
      top_class, top_score = ranked[0]
      # Preserve the full class-score mapping even in the tabular API so users
      # can inspect tied or secondary localizations without recomputing.
      row = {
        "id": prediction["id"],
        "predicted_class": prediction["predicted_class"],
        "top_class": top_class,
        "top_score": top_score,
        "best_overall_k": prediction["best_overall_k"],
        "scores_by_class": prediction["scores_by_class"],
      }
      if include_features:
        row.update(prediction["features"])
      rows.append(row)
    return pd.DataFrame(rows)


def compute_features(sequences: Iterator[Tuple[str, str]]) -> List[Dict[str, object]]:
  """Compute WoLF PSORT features using the fungi feature definition.

  Args:
    sequences: Iterator yielding ``(identifier, sequence)`` tuples.

  Returns:
    List of dictionaries containing structured feature values.
  """
  return WoLFPSortPredictor("fungi").compute_features(sequences)


def compute_features_dataframe(sequences: Iterator[Tuple[str, str]]) -> pd.DataFrame:
  """Compute WoLF PSORT features and return them in DataFrame form.

  Args:
    sequences: Iterator yielding ``(identifier, sequence)`` tuples.

  Returns:
    DataFrame with one row per sequence and one column per feature.
  """
  return WoLFPSortPredictor("fungi").compute_features_dataframe(sequences)


def predict_localization(
  sequences: Iterator[Tuple[str, str]],
  organism_type: str = "fungi",
  include_features: bool = False,
  include_neighbors: bool = False,
  as_dataframe: bool = True,
):
  """Predict WoLF PSORT localization scores for one or more sequences.

  Args:
    sequences: Iterator yielding ``(identifier, sequence)`` tuples.
    organism_type: Bundled organism model to use. Supported values are
      ``"fungi"``, ``"animal"``, and ``"plant"``.
    include_features: When ``True``, include the computed feature dictionary in
      dictionary output or expand feature columns in DataFrame output.
    include_neighbors: When ``True``, include the ranked training neighbors in
      dictionary output. This is not supported in DataFrame mode.
    as_dataframe: When ``True``, return a DataFrame summary. When ``False``,
      return a list of dictionaries.

  Returns:
    DataFrame or list of dictionaries, depending on ``as_dataframe``.
  """
  predictor = WoLFPSortPredictor(organism_type)
  if as_dataframe:
    if include_neighbors:
      raise ValueError("Neighbors are only available in dictionary output")
    return predictor.predict_dataframe(sequences, include_features=include_features)
  return predictor.predict(sequences, include_features=include_features, include_neighbors=include_neighbors)
