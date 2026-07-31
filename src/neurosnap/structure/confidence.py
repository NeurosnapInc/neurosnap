"""Granular pLDDT confidence metric exports and analysis."""

from dataclasses import dataclass
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd

from neurosnap.structure.structure import Structure


@dataclass(frozen=True)
class PLDDTReport:
  """Report container for pLDDT confidence metrics.

  Exposes DataFrames for atom, residue, chain, and distribution summaries.
  """

  atom: pd.DataFrame
  residue: pd.DataFrame
  chain: pd.DataFrame
  distribution: pd.DataFrame
  input_scale: float
  metadata: dict

  def __repr__(self) -> str:
    return f"PLDDTReport(input_scale={self.input_scale})"


def _get_distribution_df(plddt_values: np.ndarray, boundaries: Sequence[float]) -> pd.DataFrame:
  total = len(plddt_values)
  sorted_bounds = sorted(boundaries)

  labels = []
  # First bin: < boundaries[0]
  labels.append(f"<{sorted_bounds[0]}")
  # Intermediate bins: boundaries[i] - boundaries[i+1]
  for i in range(len(sorted_bounds) - 1):
    labels.append(f"{sorted_bounds[i]}-{sorted_bounds[i + 1]}")
  # Last bin: >= boundaries[-1]
  labels.append(f">={sorted_bounds[-1]}")

  counts = []
  percentages = []

  if total == 0:
    for _ in labels:
      counts.append(0)
      percentages.append(0.0)
  else:
    # count for < sorted_bounds[0]
    c = int(np.sum(plddt_values < sorted_bounds[0]))
    counts.append(c)

    for i in range(len(sorted_bounds) - 1):
      c = int(np.sum((plddt_values >= sorted_bounds[i]) & (plddt_values < sorted_bounds[i + 1])))
      counts.append(c)

    c = int(np.sum(plddt_values >= sorted_bounds[-1]))
    counts.append(c)

    percentages = [float(cnt) / total * 100.0 for cnt in counts]

  return pd.DataFrame({"count": counts, "percentage": percentages}, index=labels)


def summarize_plddt(
  structure: Structure,
  plddt: Optional[Union[np.ndarray, Sequence[float]]] = None,
  source: str = "b_factor",
  scale: Union[str, float] = "auto",
  boundaries: Sequence[float] = (50, 70, 90),
) -> PLDDTReport:
  """Summarize pLDDT metrics for a structure.

  Experimental B-factors are not pLDDT, and callers are responsible for source correctness.

  Parameters:
    structure: The input molecular structure.
    plddt: One finite pLDDT value per atom. If ``None``, values are read from ``source``.
    source: Source of pLDDT values when reading from the structure. Must be ``"b_factor"``.
    scale: Scale of the input values: ``"auto"``, ``1.0`` for ``[0, 1]``, or ``100.0`` for ``[0, 100]``.
    boundaries: Bin boundaries for distribution calculation.

  Returns:
    The calculated pLDDT confidence report.
  """
  if plddt is None:
    if source != "b_factor":
      raise ValueError(f"Unsupported source: {source}. Only 'b_factor' is supported for reading from structure.")
    plddt_vals = structure._annotation_export("b_factor")
    plddt_vals = np.asarray(plddt_vals, dtype=float)
    if not np.isfinite(plddt_vals).all():
      raise ValueError("B-factors in structure contain non-finite values (NaN or Inf).")
  else:
    plddt_vals = np.asarray(plddt, dtype=float)
    if len(plddt_vals) != len(structure):
      raise ValueError(f"plddt length ({len(plddt_vals)}) does not match structure atom count ({len(structure)}).")
    if not np.isfinite(plddt_vals).all():
      raise ValueError("plddt contains non-finite values (NaN or Inf).")

  # Scale detection and validation
  if scale not in ("auto", 1.0, 100.0, 1, 100):
    raise ValueError(f"Invalid scale: {scale}. Must be 'auto', 1.0, or 100.0.")

  if len(plddt_vals) == 0:
    detected_scale = 100.0
  else:
    is_all_zero = np.allclose(plddt_vals, 0.0)
    is_all_one = np.allclose(plddt_vals, 1.0)

    if scale == "auto":
      if is_all_zero or is_all_one:
        raise ValueError("Ambiguous all-zero/all-one pLDDT data requires an explicit scale (1.0 or 100.0) instead of guessing.")

      # Determine scale automatically based on range [0, 1] vs [0, 100]
      if np.all((plddt_vals >= 0.0) & (plddt_vals <= 1.0)):
        detected_scale = 1.0
      else:
        if np.all((plddt_vals >= 0.0) & (plddt_vals <= 100.0)):
          detected_scale = 100.0
        else:
          raise ValueError("pLDDT values are outside [0, 100] range.")
    else:
      detected_scale = float(scale)
      if detected_scale == 1.0:
        if not np.all((plddt_vals >= 0.0) & (plddt_vals <= 1.0)):
          raise ValueError("pLDDT values are outside [0, 1] range but scale=1.0 was specified.")
      elif detected_scale == 100.0:
        if not np.all((plddt_vals >= 0.0) & (plddt_vals <= 100.0)):
          raise ValueError("pLDDT values are outside [0, 100] range but scale=100.0 was specified.")

  # Normalize values to 0-100
  if detected_scale == 1.0:
    normalized_plddt = plddt_vals * 100.0
  else:
    normalized_plddt = plddt_vals.copy()

  if len(structure) == 0:
    atom_df = pd.DataFrame(columns=["atom_index", "chain_id", "res_id", "ins_code", "res_name", "atom_name", "element", "plddt"])
    residue_df = pd.DataFrame(columns=["count", "mean", "min", "max", "median"])
    chain_df = pd.DataFrame(columns=["count", "mean", "min", "max", "median", "q25", "q75", "25%", "50%", "75%"])
    distribution_df = _get_distribution_df(np.array([], dtype=float), boundaries)
  else:
    chain_ids = structure._annotation_export("chain_id")
    res_ids = structure._annotation_export("res_id")
    ins_codes = structure._annotation_export("ins_code")
    res_names = structure._annotation_export("res_name")
    hetero = structure._annotation_export("hetero")
    atom_names = structure._annotation_export("atom_name")
    elements = structure._annotation_export("element")

    atom_df = pd.DataFrame(
      {
        "atom_index": np.arange(len(structure)),
        "chain_id": [str(x) for x in chain_ids],
        "res_id": [int(x) for x in res_ids],
        "ins_code": [str(x) for x in ins_codes],
        "res_name": [str(x) for x in res_names],
        "atom_name": [str(x) for x in atom_names],
        "element": [str(x) for x in elements],
        "plddt": normalized_plddt,
      }
    )

    # stable residue key tuple (chain_id, res_id, ins_code, res_name, hetero)
    res_keys = [(str(chain_ids[i]), int(res_ids[i]), str(ins_codes[i]), str(res_names[i]), bool(hetero[i])) for i in range(len(structure))]
    atom_df["residue_key"] = res_keys

    # Group by stable residue key
    res_grouped = atom_df.groupby("residue_key", sort=False)["plddt"]
    residue_df = res_grouped.agg(count="count", mean="mean", min="min", max="max", median="median")

    # Group by chain_id
    chain_grouped = atom_df.groupby("chain_id", sort=False)["plddt"]
    chain_df = chain_grouped.agg(count="count", mean="mean", min="min", max="max", median="median")
    chain_df["q25"] = chain_grouped.quantile(0.25)
    chain_df["q75"] = chain_grouped.quantile(0.75)
    chain_df["25%"] = chain_df["q25"]
    chain_df["50%"] = chain_df["median"]
    chain_df["75%"] = chain_df["q75"]

    distribution_df = _get_distribution_df(normalized_plddt, boundaries)

    # Drop the internal column helper from atom_df to keep it clean
    atom_df = atom_df.drop(columns=["residue_key"])

  metadata = {"input_scale": detected_scale}

  return PLDDTReport(
    atom=atom_df,
    residue=residue_df,
    chain=chain_df,
    distribution=distribution_df,
    input_scale=detected_scale,
    metadata=metadata,
  )
