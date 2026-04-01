"""Residue-selection utilities for Neurosnap structures."""

import re
from typing import Dict, List, Optional

from ._common import available_chain_ids, resolve_model


def select_residues(structure, selectors: str, invert: bool = False, model: Optional[int] = None) -> Dict[str, List[int]]:
  """Select residues from a structure using a chain/residue selector string.

  Supported selector forms include:
    - ``"A"`` for an entire chain
    - ``"A10"`` or ``"A10-20"`` for legacy single-character chain IDs
    - ``"AB:10"`` or ``"AB:10-20"`` for multi-character chain IDs

  Parameters:
    structure: Input :class:`Structure`, :class:`StructureEnsemble`, or :class:`StructureStack`.
    selectors: Comma-delimited selector string.
    invert: Whether to invert the selection within each chain.
    model: Optional model ID when selecting from an ensemble or stack.

  Returns:
    Dictionary mapping chain IDs to sorted residue numbers.
  """
  structure_model = resolve_model(structure, model=model)
  chain_ids = available_chain_ids(structure_model)
  output = {chain_id: set() for chain_id in chain_ids}

  selectors = re.sub(r"\s", "", selectors).strip(",")
  while ",," in selectors:
    selectors = selectors.replace(",,", ",")
  if not selectors:
    raise ValueError("Provided selectors string is empty.")

  residue_ids_by_chain = {}
  for chain in structure_model.chains():
    residue_ids_by_chain[chain.chain_id] = {residue.res_id for residue in chain.residues()}

  for selector in selectors.split(","):
    chain_id, residue_spec = _parse_selector(selector, chain_ids)
    if chain_id not in residue_ids_by_chain:
      raise ValueError(f'Chain "{chain_id}" in selector "{selector}" does not exist in the specified structure.')

    if residue_spec is None:
      output[chain_id].update(residue_ids_by_chain[chain_id])
      continue

    if "-" in residue_spec:
      start_text, end_text = residue_spec.split("-", maxsplit=1)
      resi_start = int(start_text)
      resi_end = int(end_text)
      if resi_start > resi_end:
        raise ValueError(f'Invalid residue range selector "{selector}". The starting residue cannot be greater than the ending residue.')
      residue_ids = range(resi_start, resi_end + 1)
    else:
      residue_ids = [int(residue_spec)]

    for residue_id in residue_ids:
      if residue_id not in residue_ids_by_chain[chain_id]:
        raise ValueError(f'Residue "{residue_id}" in selector "{selector}" does not exist in the specified chain.')
      output[chain_id].add(residue_id)

  output = {chain_id: sorted(residue_ids) for chain_id, residue_ids in output.items() if residue_ids}
  if not invert:
    return output

  inverted_output = {}
  for chain_id in chain_ids:
    available_residue_ids = residue_ids_by_chain[chain_id]
    selected_residue_ids = set(output.get(chain_id, []))
    inverted_ids = sorted(available_residue_ids - selected_residue_ids)
    if inverted_ids:
      inverted_output[chain_id] = inverted_ids
  return inverted_output


def _parse_selector(selector: str, chain_ids: List[str]) -> tuple[str, Optional[str]]:
  """Parse a selector token into ``(chain_id, residue_spec)``."""
  if selector in chain_ids:
    return selector, None

  if ":" in selector:
    chain_id, residue_spec = selector.split(":", maxsplit=1)
    if not residue_spec:
      raise ValueError(f'Invalid selector "{selector}".')
    if chain_id not in chain_ids:
      raise ValueError(f'Chain "{chain_id}" in selector "{selector}" does not exist in the specified structure.')
    _validate_residue_spec(selector, residue_spec)
    return chain_id, residue_spec

  for chain_id in sorted(chain_ids, key=len, reverse=True):
    if selector.startswith(chain_id):
      residue_spec = selector[len(chain_id) :]
      if residue_spec:
        _validate_residue_spec(selector, residue_spec)
        return chain_id, residue_spec
  raise ValueError(f'Invalid selector "{selector}".')


def _validate_residue_spec(selector: str, residue_spec: str):
  """Validate the residue portion of a selector token."""
  if re.fullmatch(r"\d+", residue_spec):
    return
  if re.fullmatch(r"\d+-\d+", residue_spec):
    return
  raise ValueError(f'Invalid selector "{selector}".')
