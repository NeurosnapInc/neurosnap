"""Structure-preparation helpers.

This module provides small structure-level preparation utilities plus thin
wrappers around the existing PDB2PQR and EvoEF2 preparation backends.

The wrappers intentionally do not reimplement those engines. They expose a
structure-oriented API while delegating the underlying chemistry logic to the
existing algorithm modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ._prepare_evoef2 import rebuild_missing_atoms_with_evoef2_backend
from .structure import Structure

__all__ = [
  "has_hydrogens",
  "strip_hydrogens",
  "add_hydrogens_with_pdb2pqr",
  "optimize_hydrogens_with_pdb2pqr",
  "rebuild_missing_atoms_with_evoef2",
]


def has_hydrogens(structure: Structure) -> bool:
  """Return ``True`` if the structure currently contains hydrogen atoms.

  Parameters:
    structure: Input single-model structure.

  Returns:
    ``True`` when any atom has element ``H`` after simple normalization.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"has_hydrogens() expects a Structure, found {type(structure).__name__}.")
  if len(structure) == 0:
    return False
  elements = np.char.upper(np.char.strip(structure.atom_annotations["element"].astype("U2")))
  return bool(np.any(elements == "H"))


def strip_hydrogens(structure: Structure) -> Structure:
  """Return a copy of the structure with hydrogen atoms removed.

  Atom-level connectivity tables are subsetted and remapped automatically via
  :meth:`Structure.select`, so both bonds and interactions remain consistent
  with the returned atom table.

  Parameters:
    structure: Input single-model structure.

  Returns:
    New :class:`Structure` without hydrogen atoms.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"strip_hydrogens() expects a Structure, found {type(structure).__name__}.")
  return structure.select(predicate=lambda atom: atom.element.strip().upper() != "H")


def add_hydrogens_with_pdb2pqr(
  structure: Structure,
  *,
  forcefield: str = "PARSE",
  ffout: Optional[str] = None,
  neutraln: bool = False,
  neutralc: bool = False,
  debump: bool = True,
) -> Structure:
  """Add hydrogens using the PDB2PQR preparation backend.

  This wrapper delegates to :func:`neurosnap.algos.pdb2pqr.assign_pqr` with
  ``assign_only=False`` and ``optimize=False``. PDB2PQR may still perform its
  internal water-specific hydrogen handling, but it skips the full optimization
  path used by :func:`optimize_hydrogens_with_pdb2pqr`.

  The returned structure is the PDB2PQR-rebuilt structure, so it also carries
  any charge/radius annotations and provenance metadata that backend emits.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"add_hydrogens_with_pdb2pqr() expects a Structure, found {type(structure).__name__}.")

  from neurosnap.algos.pdb2pqr import assign_pqr

  return assign_pqr(
    structure,
    forcefield=forcefield,
    ffout=ffout,
    neutraln=neutraln,
    neutralc=neutralc,
    assign_only=False,
    debump=debump,
    optimize=False,
  )


def optimize_hydrogens_with_pdb2pqr(
  structure: Structure,
  *,
  forcefield: str = "PARSE",
  ffout: Optional[str] = None,
  neutraln: bool = False,
  neutralc: bool = False,
  debump: bool = True,
) -> Structure:
  """Add and optimize hydrogens using the PDB2PQR preparation backend.

  This wrapper delegates to :func:`neurosnap.algos.pdb2pqr.assign_pqr` with
  ``assign_only=False`` and ``optimize=True``.

  The returned structure is the PDB2PQR-rebuilt structure, so it also carries
  any charge/radius annotations and provenance metadata that backend emits.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"optimize_hydrogens_with_pdb2pqr() expects a Structure, found {type(structure).__name__}.")

  from neurosnap.algos.pdb2pqr import assign_pqr

  return assign_pqr(
    structure,
    forcefield=forcefield,
    ffout=ffout,
    neutraln=neutraln,
    neutralc=neutralc,
    assign_only=False,
    debump=debump,
    optimize=True,
  )

def rebuild_missing_atoms_with_evoef2(
  structure: Structure,
  *,
  param_path: Optional[Path] = None,
  topo_path: Optional[Path] = None,
) -> Structure:
  """Rebuild missing heavy atoms and hydrogens using bundled EvoEF2 topology data.

  Unlike the scoring implementation in ``neurosnap.algos.evoef2``, this
  structure-level wrapper uses a local reconstruction backend and returns a
  native :class:`Structure` directly.
  """
  if not isinstance(structure, Structure):
    raise TypeError(f"rebuild_missing_atoms_with_evoef2() expects a Structure, found {type(structure).__name__}.")
  return rebuild_missing_atoms_with_evoef2_backend(structure, param_path=param_path, topo_path=topo_path)
