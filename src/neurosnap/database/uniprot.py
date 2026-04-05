"""UniProt and UniParc sequence retrieval helpers."""

import io
from typing import Dict, Iterable, Union

import pandas as pd
import requests
from tqdm import tqdm

from neurosnap.log import logger
from neurosnap.sequence.align import read_msa


def fetch_accessions(accessions: Iterable[str], batch_size: int = 150) -> Dict[str, Union[str, None]]:
  """Fetch sequences corresponding to a list of UniProt accession numbers.

  This function queries UniParc first and then UniProtKB for any missing
  accessions. Accessions are processed in batches to handle large lists
  efficiently.

  Args:
      accessions: A list of UniProt accession numbers. Duplicate accessions
          are removed automatically.
      batch_size: Number of accessions to query per request.

  Returns:
      Dictionary mapping accession numbers to protein sequences. Missing
      accessions are assigned ``None``.

  Raises:
      requests.exceptions.HTTPError: If an API request fails.
  """
  accessions = list(set(str(x).strip() for x in accessions))

  batches = [accessions[i : i + batch_size] for i in range(0, len(accessions), batch_size)]

  output = {}
  for batch in tqdm(batches, desc="Fetching sequences from uniprot.org", total=len(batches)):
    query = " OR ".join([f"isoform:{accession}" if "-" in accession else f"accession:{accession}" for accession in batch])
    response = requests.get(f"https://rest.uniprot.org/uniparc/search?fields=accession,sequence&format=tsv&query=({query})&size=500")
    if response.status_code == 200:
      dataframe = pd.read_csv(io.StringIO(response.text), sep="\t")
      for _, row in dataframe.iterrows():
        for accession in row.UniProtKB.split("; "):
          if accession in batch and accession not in output:
            output[accession] = row.Sequence
            break
    else:
      logger.error(f"[{response.status_code}] {response.text}")
      response.raise_for_status()

  accessions_missing = [accession for accession in accessions if accession not in output]
  batches = [accessions_missing[i : i + batch_size] for i in range(0, len(accessions_missing), batch_size)]
  for batch in tqdm(batches, desc="Fetching sequences from uniprot.org", total=len(batches)):
    query = " OR ".join([f"accession:{accession}" for accession in batch])
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/search?fields=accession,sequence&format=tsv&query=({query})&size=500")
    if response.status_code == 200:
      dataframe = pd.read_csv(io.StringIO(response.text), sep="\t")
      for _, row in dataframe.iterrows():
        if row.Entry in batch and row.Entry not in output:
          output[row.Entry] = row.Sequence
    else:
      logger.error(f"[{response.status_code}] {response.text}")
      response.raise_for_status()

  for accession in accessions:
    if accession not in output:
      output[accession] = None
      logger.warning(f"Could not find a sequence for accession: {accession}")

  return output


def fetch_uniprot(uniprot_id: str, head: bool = False) -> Union[str, bool]:
  """Fetch a UniProtKB or UniParc FASTA entry by identifier.

  Args:
      uniprot_id: UniProtKB or UniParc accession ID.
      head: If ``True``, perform a HEAD request and return whether the entry
          exists.

  Returns:
      ``True`` when ``head`` is enabled and the accession exists, otherwise the
      fetched protein sequence.

  Raises:
      Exception: If the accession is not found in UniProtKB or UniParc.
      ValueError: If the returned FASTA does not contain exactly one sequence.
  """
  method = requests.head if head else requests.get
  logger.debug(f"Fetching uniprot entry with ID {uniprot_id}")
  response = method(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
  if response.status_code != 200:
    response = method(f"https://rest.uniprot.org/uniparc/{uniprot_id}.fasta")
    if response.status_code != 200:
      raise Exception(
        f'Could not find UniProt accession "{uniprot_id}" in either UniProtKB or UniParc. Please ensure that IDs are correct and refer to actual proteins.'
      )

  if head:
    return True

  sequences = [sequence for _, sequence in read_msa(response.text)]
  if len(sequences) > 1:
    print(response.text)
    raise ValueError("Too many sequences returned")
  if len(sequences) < 1:
    print(response.text)
    raise ValueError("No sequence returned")

  return sequences[0]
