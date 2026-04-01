"""Foldseek API helpers."""

import json
import os
import tempfile
import time
from typing import List, Optional, TYPE_CHECKING, Union

import pandas as pd
import requests

if TYPE_CHECKING:
  from neurosnap.protein import Protein


def foldseek_search(
  protein: Union["Protein", str],
  mode: str = "3diaa",
  databases: Optional[List[str]] = None,
  max_retries: int = 10,
  retry_interval: int = 5,
  output_format: str = "json",
) -> Union[str, pd.DataFrame]:
  """Perform a protein structure search using the Foldseek API.

  Parameters:
      protein: Either a :class:`Protein` object or a path to a PDB file.
      mode: Search mode. Must be one of ``"3diaa"`` or ``"tm-align"``.
      databases: Databases to search. Defaults to Foldseek's common public
          structure databases.
      max_retries: Maximum number of retries when polling job status.
      retry_interval: Seconds between job-status polls.
      output_format: Output format, either ``"json"`` or ``"dataframe"``.

  Returns:
      Search results in the requested format.

  Raises:
      RuntimeError: If job submission or retrieval fails.
      TimeoutError: If the job does not complete before retries are exhausted.
      ValueError: If ``output_format`` is invalid.
  """
  base_url = "https://search.foldseek.com/api"

  if databases is None:
    databases = ["afdb50", "afdb-swissprot", "afdb-proteome", "bfmd", "cath50", "mgnify_esm30", "pdb100", "gmgcl_id", "bfvd"]

  created_temp_file = False
  if _is_protein_instance(protein):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
      protein.save(temp_file.name)
      file_path = temp_file.name
      created_temp_file = True
  else:
    file_path = protein

  data = {"mode": mode, "database[]": databases}
  try:
    with open(file_path, "rb") as file_handle:
      files = {"q": file_handle}
      response = requests.post(f"{base_url}/ticket", data=data, files=files)
    response.raise_for_status()
    job_id = response.json()["id"]
  except requests.RequestException as exc:
    raise RuntimeError(f"Failed to submit job: {exc}")

  try:
    for _attempt in range(max_retries):
      try:
        status_response = requests.get(f"{base_url}/ticket/{job_id}")
        status_response.raise_for_status()
        status = status_response.json().get("status", "ERROR")
      except requests.RequestException as exc:
        raise RuntimeError(f"Failed to retrieve job status: {exc}")

      if status == "COMPLETE":
        break
      if status == "ERROR":
        raise RuntimeError("Job failed")
      time.sleep(retry_interval)
    else:
      raise TimeoutError(f"Job did not complete within {max_retries * retry_interval} seconds")

    results = []
    entry = 0
    while True:
      try:
        result_response = requests.get(f"{base_url}/result/{job_id}/{entry}")
        result_response.raise_for_status()
        result = result_response.json()
      except requests.RequestException as exc:
        raise RuntimeError(f"Failed to retrieve results: {exc}")

      if not result or all(len(db_result["alignments"]) == 0 for db_result in result["results"]):
        break

      results.append(result)
      entry += 1
  finally:
    if created_temp_file:
      os.remove(file_path)

  if output_format == "json":
    return json.dumps(results, indent=2)
  if output_format == "dataframe":
    rows = []
    for result in results:
      for db_result in result["results"]:
        alignments = db_result["alignments"]
        for alignment in alignments[0]:
          rows.append(
            {
              "target": alignment["target"],
              "db": db_result["db"],
              "seqId": alignment.get("seqId", ""),
              "alnLength": alignment.get("alnLength", ""),
              "missmatches": alignment.get("missmatches", ""),
              "gapsopened": alignment.get("gapsopened", ""),
              "qStartPos": alignment.get("qStartPos", ""),
              "qEndPos": alignment.get("qEndPos", ""),
              "dbStartPos": alignment.get("dbStartPos", ""),
              "dbEndPos": alignment.get("dbEndPos", ""),
              "eval": alignment.get("eval", ""),
              "score": alignment.get("score", ""),
              "qLen": alignment.get("qLen", ""),
              "dbLen": alignment.get("dbLen", ""),
              "seq": alignment.get("tSeq", ""),
            }
          )
    return pd.DataFrame(rows)

  raise ValueError("Invalid output_format. Choose 'json' or 'dataframe'.")


def _is_protein_instance(value) -> bool:
  """Return ``True`` when a value is a Neurosnap Protein instance."""
  try:
    from neurosnap.protein import Protein
  except ImportError:
    return False
  return isinstance(value, Protein)
