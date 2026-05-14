"""PQR writer for Structures annotated with partial charge and radius."""

from __future__ import annotations

import io
import pathlib
from typing import Union

from neurosnap.structure.structure import Structure

__all__ = ["save_pqr"]


def save_pqr(
  structure: Structure,
  pqr: Union[str, pathlib.Path, io.IOBase],
  *,
  include_header: bool = False,
  keep_chain: bool = True,
):
  """Save an annotated Structure as a PQR file.

  Parameters:
    structure: Single-model structure annotated with ``partial_charge`` and ``radius``.
    pqr: Output filepath or open file handle.
    include_header: Whether to include the stored ``pdb2pqr_header`` metadata when present.
    keep_chain: Whether to include chain IDs in the written PQR records.

  Example:
    Save a structure returned by :func:`neurosnap.algos.pdb2pqr.assign_pqr`::

      from neurosnap.io.pdb import parse_pdb
      from neurosnap.algos.pdb2pqr import assign_pqr
      from neurosnap.io.pqr import save_pqr

      structure = parse_pdb("tests/files/1BTL.pdb", return_type="ensemble").first()
      pqr_structure = assign_pqr(structure, forcefield="AMBER", assign_only=True)
      save_pqr(pqr_structure, "test_output.pqr", include_header=True)
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"save_pqr() expects a Structure, found {type(structure).__name__}.")
  if "partial_charge" not in structure.atom_annotations.dtype.names:
    raise ValueError('Structure is missing required "partial_charge" annotation for PQR output.')
  if "radius" not in structure.atom_annotations.dtype.names:
    raise ValueError('Structure is missing required "radius" annotation for PQR output.')

  lines = []
  header = structure.metadata.get("pdb2pqr_header")
  if include_header and isinstance(header, str) and header:
    lines.append(header if header.endswith("\n") else f"{header}\n")

  current_chain = None
  for atom_index in range(len(structure)):
    chain_id = str(structure.atom_annotations["chain_id"][atom_index])
    if current_chain is None:
      current_chain = chain_id
    elif chain_id != current_chain:
      current_chain = chain_id
      lines.append("TER\n")
    lines.append(_format_pqr_atom_record(structure, atom_index, keep_chain=keep_chain))

  lines.append("TER\n")
  lines.append("END\n")
  text = "".join(lines)

  if isinstance(pqr, io.IOBase):
    pqr.write(text)
    return

  with open(pqr, "w", encoding="utf-8") as handle:
    handle.write(text)


def _format_pqr_atom_record(structure: Structure, atom_index: int, *, keep_chain: bool) -> str:
  record = "HETATM" if bool(structure.atom_annotations["hetero"][atom_index]) else "ATOM  "
  atom_id = int(structure.atom_annotations["atom_id"][atom_index]) or (atom_index + 1)
  atom_name = str(structure.atom_annotations["atom_name"][atom_index])
  res_name = str(structure.atom_annotations["res_name"][atom_index])
  chain_id = str(structure.atom_annotations["chain_id"][atom_index]) if keep_chain else ""
  res_id = int(structure.atom_annotations["res_id"][atom_index])
  ins_code = str(structure.atom_annotations["ins_code"][atom_index])
  x = float(structure.atoms["x"][atom_index])
  y = float(structure.atoms["y"][atom_index])
  z = float(structure.atoms["z"][atom_index])
  partial_charge = float(structure.atom_annotations["partial_charge"][atom_index])
  radius = float(structure.atom_annotations["radius"][atom_index])

  line = f"{record}{atom_id:5d} "
  if len(atom_name) == 4 or len(atom_name.strip('FLIP')) == 4:
    line += f"{atom_name:<4}"[:4]
  else:
    line += f" {atom_name:<3}"[:4]
  if len(res_name) == 4:
    line += f"{res_name:<4}"[:4]
  else:
    line += f" {res_name:<3}"[:4]
  line += " "
  line += f"{chain_id:<1}"[:1]
  line += f"{res_id:4d}"[:4]
  line += f"{ins_code}   " if ins_code else "    "
  line += f"{x:8.3f}"[:8]
  line += f"{y:8.3f}"[:8]
  line += f"{z:8.3f}"[:8]
  line += f"{partial_charge:8.4f}"[-8:]
  line += f"{radius:7.4f}"[-7:]
  return f"{line}\n"
