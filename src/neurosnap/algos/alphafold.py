"""
Variables, functions, and classes associated with AlphaFold2 and whatnot.
"""
import numpy as np


def score_af2m_binding(af2m_dict: str, binder_len: int, target_len: int = None) -> dict:
  """
  -------------------------------------------------------
  Calculate binding scores from AlphaFold2 multimer prediction results.
  The binder is assumed to be the first part of the sequence up to `binder_len`,
  with the target being the remainder, unless otherwise specified.
  Adapted from: https://github.com/hgbrian/biomodals/blob/990c010e711c1e8a7221294e0370c6f37927eae6/modal_alphafold.py#L33
  -------------------------------------------------------
  Parameters:
    af_multimer_dict: From AlphaFold2 multimer JSON file (str)
    binder_len: Length of the binder protein sequence (int)
    target_len: Length of the target protein sequence (int)
  Returns:
    dict: A dictionary containing the following scores:
      - plddt_binder (float): Average pLDDT score for the binder.
      - plddt_target (float): Average pLDDT score for the target.
      - pae_binder (float): Average PAE score within the binder.
      - pae_target (float): Average PAE score within the target.
      - ipae (float): Average PAE score for the binder-target interaction.
  """
  target_end = (binder_len + target_len) if target_len is not None else None

  # pLDDT
  plddt_array = np.array(af2m_dict["plddt"])
  plddt_binder = np.mean(plddt_array[:binder_len])
  plddt_target = np.mean(plddt_array[binder_len:target_end])

  # PAE
  pae_array = np.array(af2m_dict["pae"])

  pae_binder = np.mean(pae_array[:binder_len, :binder_len])
  pae_target = np.mean(pae_array[binder_len:target_end, binder_len:target_end])
  ipae = np.mean([
    np.mean(pae_array[:binder_len, binder_len:target_end]),
    np.mean(pae_array[binder_len:target_end, :binder_len]),
  ])

  return {
    "plddt_binder": float(plddt_binder),
    "plddt_target": float(plddt_target),
    "pae_binder": float(pae_binder),
    "pae_target": float(pae_target),
    "ipae": float(ipae),
  }