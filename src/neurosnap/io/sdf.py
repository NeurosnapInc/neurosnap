"""Parser and writer for SDF files.

This module provides Neurosnap-native :func:`parse_sdf` and :func:`save_sdf`
helpers for reading and writing
:class:`~neurosnap.structure.structure.Structure`,
:class:`~neurosnap.structure.structure.StructureEnsemble`, and
:class:`~neurosnap.structure.structure.StructureStack` objects.

The implementation intentionally follows RDKit's own SDF reading and writing
logic as closely as possible: molecules are parsed through RDKit suppliers,
sanitized using RDKit defaults, and written back using RDKit's SD writer.
"""

import io
import pathlib
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D

from neurosnap.structure.structure import Structure, StructureEnsemble, StructureStack

__all__ = ["parse_sdf", "save_sdf"]

ReturnType = Literal["ensemble", "stack", "auto"]

_RDKIT_BOND_TO_INT = {
  rdchem.BondType.SINGLE: 1,
  rdchem.BondType.DOUBLE: 2,
  rdchem.BondType.TRIPLE: 3,
  rdchem.BondType.AROMATIC: 4,
  rdchem.BondType.QUADRUPLE: 5,
}
_INT_TO_RDKIT_BOND = {
  1: rdchem.BondType.SINGLE,
  2: rdchem.BondType.DOUBLE,
  3: rdchem.BondType.TRIPLE,
  4: rdchem.BondType.AROMATIC,
  5: rdchem.BondType.QUADRUPLE,
}


def parse_sdf(
  sdf: Union[str, pathlib.Path, io.IOBase],
  return_type: ReturnType = "auto",
) -> Union[StructureEnsemble, StructureStack]:
  """Parse an SDF file into Neurosnap structure containers.

  Each SDF record is parsed with RDKit and converted into one Neurosnap
  :class:`Structure` model. Multi-record SDF files therefore map naturally to
  a :class:`StructureEnsemble`, and ``return_type="auto"`` will return a
  :class:`StructureStack` when all records share identical atom annotations and
  bonds.

  Because SDF is a small-molecule format, chain and residue hierarchy are not
  natively represented. Parsed structures therefore default to a single
  heterogen residue ``LIG`` in chain ``A`` unless RDKit monomer information is
  present on the atoms.

  Parameters:
    sdf: SDF filepath or open file handle.
    return_type: Output container type. ``"ensemble"`` always returns a
      :class:`StructureEnsemble`, ``"stack"`` requires stack-compatible models,
      and ``"auto"`` returns a :class:`StructureStack` when possible or falls
      back to a :class:`StructureEnsemble`.

  Returns:
    A :class:`StructureEnsemble` or :class:`StructureStack` depending on
    ``return_type`` and model compatibility.
  """
  if return_type not in {"ensemble", "stack", "auto"}:
    raise ValueError('return_type must be one of "ensemble", "stack", or "auto".')

  supplier = _rdkit_supplier_from_sdf(sdf)
  ensemble = StructureEnsemble()
  for record_index, mol in enumerate(supplier, start=1):
    if mol is None:
      raise ValueError(f"Failed to parse SDF record {record_index}.")

    structure = _structure_from_rdkit_mol(mol, model_id=record_index)
    ensemble.append(structure, model_id=record_index)

  if len(ensemble) == 0:
    raise ValueError("No molecules were found in the SDF file.")

  ensemble.metadata["source_format"] = "sdf"
  if return_type == "ensemble":
    return ensemble
  if return_type == "stack":
    return StructureStack.from_ensemble(ensemble)

  try:
    return StructureStack.from_ensemble(ensemble)
  except ValueError:
    return ensemble


def save_sdf(
  structure: Union[Structure, StructureEnsemble, StructureStack],
  sdf: Union[str, pathlib.Path, io.IOBase],
):
  """Save a Neurosnap structure container as an SDF file.

  Parameters:
    structure: Structure container to write.
    sdf: Output filepath or open file handle.

  Notes:
    SDF is a small-molecule format, so chain and residue hierarchy are flattened
    during output. Each model is written as a separate SDF record using RDKit's
    own SD writer. Structure metadata is exported as SDF molecule properties
    when the values are scalar.
  """
  models = _models_for_sdf_output(structure)
  if not models:
    raise ValueError("No models are available for SDF output.")

  writer = Chem.SDWriter(str(sdf) if not isinstance(sdf, io.IOBase) else sdf)
  try:
    for model_id, model in models:
      mol = _rdkit_mol_from_structure(model, model_id=model_id)
      writer.write(mol)
  finally:
    writer.close()


def _rdkit_supplier_from_sdf(sdf: Union[str, pathlib.Path, io.IOBase]) -> Chem.SDMolSupplier:
  """Return an RDKit SDF supplier for a filepath or file-like object."""
  if isinstance(sdf, io.IOBase):
    content = sdf.read()
    if isinstance(content, bytes):
      content = content.decode("utf-8")
    supplier = Chem.SDMolSupplier()
    supplier.SetData(content, sanitize=True, removeHs=False, strictParsing=True)
    return supplier

  return Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False, strictParsing=True)


def _structure_from_rdkit_mol(mol: Chem.Mol, model_id: int) -> Structure:
  """Convert one RDKit molecule into a Neurosnap structure model."""
  if mol.GetNumAtoms() == 0:
    raise ValueError("SDF record contains no atoms.")

  if mol.GetNumConformers() == 0:
    raise ValueError("SDF record does not contain 3D coordinates.")

  conformer = mol.GetConformer()
  structure = Structure(remove_annotations=False)
  structure.metadata = {"model_id": model_id}
  if mol.HasProp("_Name"):
    structure.metadata["title"] = mol.GetProp("_Name")

  for prop_name in mol.GetPropNames(includePrivate=False, includeComputed=False):
    if prop_name == "_Name":
      continue
    structure.metadata[prop_name] = mol.GetProp(prop_name)

  atom_defs = []
  bond_rows = []
  element_counts: Dict[str, int] = {}

  for atom_index, atom in enumerate(mol.GetAtoms()):
    atom_info = atom.GetMonomerInfo()
    if isinstance(atom_info, Chem.AtomPDBResidueInfo):
      chain_id = atom_info.GetChainId().strip() or "A"
      res_id = atom_info.GetResidueNumber()
      ins_code = atom_info.GetInsertionCode().strip()
      res_name = atom_info.GetResidueName().strip() or "LIG"
      hetero = atom_info.GetIsHeteroAtom()
      atom_name = atom_info.GetName().strip() or atom.GetSymbol()
      atom_id = atom_info.GetSerialNumber() or (atom_index + 1)
    else:
      chain_id = "A"
      res_id = 1
      ins_code = ""
      res_name = "LIG"
      hetero = True
      element = atom.GetSymbol().upper()
      element_counts[element] = element_counts.get(element, 0) + 1
      atom_name = f"{element}{element_counts[element]}"
      atom_id = atom_index + 1

    position = conformer.GetAtomPosition(atom_index)
    atom_defs.append(
      (
        float(position.x),
        float(position.y),
        float(position.z),
        chain_id,
        int(res_id),
        ins_code,
        res_name,
        bool(hetero),
        atom_name,
        atom.GetSymbol().upper(),
        int(atom_id),
        0.0,
        1.0,
        int(atom.GetFormalCharge()),
        "",
      )
    )

  for bond in mol.GetBonds():
    bond_type = _RDKIT_BOND_TO_INT.get(bond.GetBondType())
    if bond_type is None:
      if bond.GetIsAromatic():
        bond_type = 4
      else:
        raise ValueError(f"Unsupported RDKit bond type {bond.GetBondType()} in SDF record.")
    bond_rows.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type))

  structure.atoms = np.array([(x, y, z) for x, y, z, *_rest in atom_defs], dtype=structure._dtype_atoms)
  structure.atom_annotations = np.zeros(len(atom_defs), dtype=structure._dtype_atom_annotations)
  for atom_index, (_x, _y, _z, chain_id, res_id, ins_code, res_name, hetero, atom_name, element, atom_id, b_factor, occupancy, charge, sym_id) in enumerate(atom_defs):
    structure.atom_annotations["chain_id"][atom_index] = chain_id
    structure.atom_annotations["res_id"][atom_index] = res_id
    structure.atom_annotations["ins_code"][atom_index] = ins_code
    structure.atom_annotations["res_name"][atom_index] = res_name
    structure.atom_annotations["hetero"][atom_index] = hetero
    structure.atom_annotations["atom_name"][atom_index] = atom_name
    structure.atom_annotations["element"][atom_index] = element
    structure.atom_annotations["atom_id"][atom_index] = atom_id
    structure.atom_annotations["b_factor"][atom_index] = b_factor
    structure.atom_annotations["occupancy"][atom_index] = occupancy
    structure.atom_annotations["charge"][atom_index] = charge
    structure.atom_annotations["sym_id"][atom_index] = sym_id

  if bond_rows:
    structure.bonds = np.array(bond_rows, dtype=structure._dtype_bond)
  else:
    structure.bonds = np.zeros(0, dtype=structure._dtype_bond)

  structure._remove_empty_annotations()
  return structure


def _models_for_sdf_output(structure: Union[Structure, StructureEnsemble, StructureStack]) -> List[Tuple[int, Structure]]:
  """Return a normalized list of ``(model_id, model)`` pairs for writing."""
  if isinstance(structure, Structure):
    model_id = int(structure.metadata.get("model_id", 1))
    return [(model_id, structure)]
  if isinstance(structure, StructureEnsemble):
    return list(zip(structure.model_ids, structure.models()))
  if isinstance(structure, StructureStack):
    return list(zip(structure.model_ids, structure.models()))
  raise TypeError(f"Unsupported structure type for SDF output: {type(structure).__name__}.")


def _rdkit_mol_from_structure(structure: Structure, model_id: int) -> Chem.Mol:
  """Convert one Neurosnap structure model into an RDKit molecule."""
  rw_mol = Chem.RWMol()
  conformer = Chem.Conformer(len(structure))
  aromatic_atoms: set[int] = set()

  for atom_index in range(len(structure)):
    element = str(structure.atom_annotations["element"][atom_index]).strip().upper()
    if not element:
      raise ValueError(f"Atom {atom_index + 1} is missing an element and cannot be written to SDF.")

    rd_atom = Chem.Atom(element)
    charge = _annotation_value_for_sdf(structure, "charge", atom_index, 0)
    rd_atom.SetFormalCharge(int(charge))
    rw_mol.AddAtom(rd_atom)
    conformer.SetAtomPosition(
      atom_index,
      Point3D(
        float(structure.atoms["x"][atom_index]),
        float(structure.atoms["y"][atom_index]),
        float(structure.atoms["z"][atom_index]),
      ),
    )

  for bond in structure.bonds:
    atom_i = int(bond["atom_i"])
    atom_j = int(bond["atom_j"])
    bond_type_value = int(bond["bond_type"])
    rd_bond_type = _INT_TO_RDKIT_BOND.get(bond_type_value)
    if rd_bond_type is None:
      raise ValueError(f"Unsupported bond_type {bond_type_value} for SDF output.")

    rw_mol.AddBond(atom_i, atom_j, rd_bond_type)
    if rd_bond_type == rdchem.BondType.AROMATIC:
      aromatic_atoms.add(atom_i)
      aromatic_atoms.add(atom_j)

  mol = rw_mol.GetMol()
  mol.AddConformer(conformer, assignId=True)

  for atom_index in aromatic_atoms:
    atom = mol.GetAtomWithIdx(atom_index)
    atom.SetIsAromatic(True)

  for bond in mol.GetBonds():
    if bond.GetBondType() == rdchem.BondType.AROMATIC:
      bond.SetIsAromatic(True)

  title = structure.metadata.get("title") or structure.metadata.get("name") or f"model_{model_id}"
  mol.SetProp("_Name", str(title))
  mol.SetIntProp("model_id", int(model_id))
  for key, value in structure.metadata.items():
    if key in {"title", "name", "model_id"}:
      continue
    if isinstance(value, (str, int, float, bool)):
      mol.SetProp(str(key), str(value))

  return mol


def _annotation_value_for_sdf(structure: Structure, name: str, atom_index: int, default):
  """Return an annotation value with a fallback default for SDF output."""
  if name not in structure.atom_annotations.dtype.names:
    return default
  value = structure.atom_annotations[name][atom_index]
  if isinstance(value, np.generic):
    return value.item()
  return value
