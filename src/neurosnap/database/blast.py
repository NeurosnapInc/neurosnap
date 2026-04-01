"""BLAST search helpers for sequence queries."""

import time
import xml.etree.ElementTree as ET
from typing import Optional, Union

import pandas as pd
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from neurosnap.api import USER_AGENT
from neurosnap.structure._common import resolve_model
from neurosnap.structure.structure import Structure, StructureEnsemble, StructureStack

StructureLike = Union[Structure, StructureEnsemble, StructureStack]


def run_blast(
  sequence: Union[str, StructureLike],
  email: str,
  matrix: str = "BLOSUM62",
  alignments: int = 250,
  scores: int = 250,
  evalue: float = 10.0,
  filter: bool = False,
  gapalign: bool = True,
  database: str = "uniprotkb_refprotswissprot",
  output_format: Optional[str] = None,
  output_path: Optional[str] = None,
  return_df: bool = True,
) -> Optional[pd.DataFrame]:
  """Submit a BLASTP job to the EBI service and optionally return hits as a dataframe.

  When a structure container is provided, the sequence is derived from the
  first model and requires that model to contain exactly one chain.
  """
  valid_databases = [
    "uniprotkb_refprotswissprot",
    "uniprotkb_pdb",
    "uniprotkb",
    "afdb",
    "uniprotkb_reference_proteomes",
    "uniprotkb_swissprot",
    "uniref100",
    "uniref90",
    "uniref50",
    "uniparc",
  ]
  if database not in valid_databases:
    raise ValueError(f"Database must be one of the following {valid_databases}")

  valid_matrices = ["BLOSUM45", "BLOSUM62", "BLOSUM80", "PAM30", "PAM70"]
  if matrix not in valid_matrices:
    raise ValueError(f"Matrix must be one of the following {valid_matrices}")

  valid_evalues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
  if evalue not in valid_evalues:
    raise ValueError(f"E-threshold must be one of the following {valid_evalues}")
  if evalue > 1:
    evalue = int(evalue)

  valid_alignments = [50, 100, 250, 500, 750, 1000]
  if alignments not in valid_alignments:
    raise ValueError(f"Alignments must be one of the following {valid_alignments}")

  valid_output_formats = ["xml", "fasta", None]
  if output_format not in valid_output_formats:
    raise ValueError(f"Output format must be one of the following {valid_output_formats}")

  if isinstance(sequence, (Structure, StructureEnsemble, StructureStack)):
    structure_model = resolve_model(sequence)
    chains = structure_model.chains()
    if len(chains) > 1:
      raise AssertionError("The structure has multiple chains. Extract a single chain sequence before calling run_blast().")
    if not chains:
      raise ValueError("The structure does not contain any chains.")
    sequence = chains[0].sequence(polymer_type="protein")
    if not sequence:
      raise ValueError("Could not derive a protein sequence from the provided structure.")

  url = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run"
  multipart_data = MultipartEncoder(
    fields={
      "email": email,
      "program": "blastp",
      "matrix": matrix,
      "alignments": str(alignments),
      "scores": str(scores),
      "exp": str(evalue),
      "filter": "T" if filter else "F",
      "gapalign": str(gapalign).lower(),
      "stype": "protein",
      "sequence": sequence,
      "database": database,
    }
  )

  headers = {
    "User-Agent": USER_AGENT,
    "Accept": "text/plain,application/json",
    "Accept-Language": "en-US,en;q=0.5",
    "Content-Type": multipart_data.content_type,
  }

  response = requests.post(url, headers=headers, data=multipart_data)
  if response.status_code == 200:
    job_id = response.text.strip()
    print(f"Job submitted successfully. Job ID: {job_id}")
  else:
    response.raise_for_status()

  status_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/status/{job_id}"
  while True:
    status_response = requests.get(status_url)
    status = status_response.text.strip()

    if status_response.status_code == 200:
      print(f"Job status: {status}")
      if status == "FINISHED":
        break
      if status in ["RUNNING", "PENDING", "QUEUED"]:
        time.sleep(20)
      else:
        raise Exception(f"Job failed with status: {status}")
    else:
      status_response.raise_for_status()

  xml_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/{job_id}/xml"
  xml_response = requests.get(xml_url)

  if xml_response.status_code == 200:
    xml_content = xml_response.text
    if output_format == "xml" and output_path:
      with open(output_path, "w") as xml_file:
        xml_file.write(xml_content)
      print(f"XML result saved as {output_path}")
    elif output_format == "fasta" and output_path:
      return _parse_xml_to_fasta_and_dataframe(xml_content, output_format=output_format, output_path=output_path, return_df=return_df)
    elif return_df:
      return _parse_xml_to_fasta_and_dataframe(xml_content, output_format=output_format, output_path=output_path, return_df=return_df)
  else:
    xml_response.raise_for_status()

  return None


def _parse_xml_to_fasta_and_dataframe(
  xml_content: str,
  output_format: Optional[str] = None,
  output_path: Optional[str] = None,
  return_df: bool = True,
) -> Optional[pd.DataFrame]:
  """Parse EBI BLAST XML into FASTA output and/or a dataframe."""
  root = ET.fromstring(xml_content)
  hits = []
  fasta_content = ""

  for hit in root.findall(".//{http://www.ebi.ac.uk/schema}hit"):
    hit_id = hit.attrib["id"]
    hit_accession = hit.attrib["ac"]
    hit_description = hit.attrib["description"]
    hit_length = hit.attrib["length"]

    for alignment in hit.findall(".//{http://www.ebi.ac.uk/schema}alignment"):
      score = alignment.find("{http://www.ebi.ac.uk/schema}score").text
      bits = alignment.find("{http://www.ebi.ac.uk/schema}bits").text
      expectation = alignment.find("{http://www.ebi.ac.uk/schema}expectation").text
      identity = alignment.find("{http://www.ebi.ac.uk/schema}identity").text
      gaps = alignment.find("{http://www.ebi.ac.uk/schema}gaps").text
      query_seq = alignment.find("{http://www.ebi.ac.uk/schema}querySeq").text
      match_seq = alignment.find("{http://www.ebi.ac.uk/schema}matchSeq").text

      fasta_content += (
        f">{hit_id} | Accession: {hit_accession} | Description: {hit_description} | "
        f"Length: {hit_length} | Score: {score} | Bits: {bits} | "
        f"Expectation: {expectation} | Identity: {identity}% | Gaps: {gaps}\n"
        f"{match_seq}\n\n"
      )

      hits.append(
        {
          "Hit ID": hit_id,
          "Accession": hit_accession,
          "Description": hit_description,
          "Length": hit_length,
          "Score": score,
          "Bits": bits,
          "Expectation": expectation,
          "Identity (%)": identity,
          "Gaps": gaps,
          "Query Sequence": query_seq,
          "Match Sequence": match_seq,
        }
      )

  if output_format == "fasta" and output_path:
    with open(output_path, "w") as fasta_file:
      fasta_file.write(fasta_content)
    print(f"FASTA result saved as {output_path}")

  if return_df:
    return pd.DataFrame(hits)
  return None
