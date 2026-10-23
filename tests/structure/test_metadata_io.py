"""Native descriptive metadata survives structure file round trips."""

import io

import pytest

from neurosnap.io.mmcif import parse_mmcif, save_cif
from neurosnap.io.pdb import parse_pdb, save_pdb
from neurosnap.structure import StructureEnsemble
from tests._structure_test_utils import MIXED_BACKBONE_ATOMS, make_structure


def _two_models():
  return StructureEnsemble([make_structure(MIXED_BACKBONE_ATOMS), make_structure(MIXED_BACKBONE_ATOMS)], model_ids=[1, 2])


@pytest.mark.parametrize("return_type", ["ensemble", "stack", "auto"])
@pytest.mark.parametrize("single_model", [False, True])
def test_pdb_metadata_round_trip(return_type, single_model):
  records = [
    "HEADER    TEST STRUCTURE                           23-OCT-26   TEST",
    "TITLE     FIRST TITLE LINE",
    "TITLE    2 SECOND TITLE LINE",
    "AUTHOR    A.RESEARCHER",
    "REMARK   2 RESOLUTION.    2.00 ANGSTROMS.",
    "REMARK 999 CUSTOM METADATA",
  ]
  coordinates = io.StringIO()
  save_pdb(_two_models(), coordinates)
  parsed = parse_pdb(io.StringIO("\n".join(records) + "\n" + coordinates.getvalue()), return_type=return_type)

  assert parsed.metadata["pdb_records"] == records
  model = parsed.models()[0]
  assert model.metadata["pdb_records"] == records
  target = model if single_model else parsed
  target.metadata["pdb_records"] = records + ["REMARK 999 ADDED BY CALLER"]
  output = io.StringIO()
  save_pdb(target, output)
  assert output.getvalue().startswith("\n".join(target.metadata["pdb_records"]) + "\n")
  reloaded = parse_pdb(io.StringIO(output.getvalue()), return_type=return_type)
  assert len(reloaded) == (1 if single_model else 2)
  assert reloaded.models()[0].metadata["pdb_records"] == target.metadata["pdb_records"]
  assert len(reloaded.models()[0]) == len(model)


@pytest.mark.parametrize("return_type", ["ensemble", "stack", "auto"])
@pytest.mark.parametrize("single_model", [False, True])
@pytest.mark.parametrize("minimal", [False, True])
def test_mmcif_metadata_round_trip(return_type, single_model, minimal):
  coordinates = io.StringIO()
  save_cif(_two_models(), coordinates)
  metadata_text = """_entry.id TEST
_struct.title
;A title with 'single' and "double" quotes
and a second line
;
loop_
_audit_author.name
_audit_author.pdbx_ordinal
'First, Author' 1
'Second, Author' 2
_exptl.method 'X-RAY DIFFRACTION'
_custom.unknown ?
_custom.inapplicable .
"""
  source = coordinates.getvalue().replace("data_neurosnap", "data_test\n" + metadata_text, 1)
  parsed = parse_mmcif(io.StringIO(source), return_type=return_type)
  fields = parsed.metadata["mmcif_fields"]
  assert parsed.metadata["mmcif_data_name"] == "test"
  assert fields["_entry.id"] == ["TEST"]
  assert fields["_audit_author.name"] == ["First, Author", "Second, Author"]
  assert fields["_struct.title"] == ['A title with \'single\' and "double" quotes\nand a second line']
  assert not any(key.startswith(("_atom_site.", "_entity.", "_struct_asym.")) for key in fields)
  model = parsed.models()[0]
  assert model.metadata["mmcif_fields"] == fields
  target = model if single_model else parsed
  target.metadata["mmcif_fields"]["_entry.id"] = ["EDITED"]
  output = io.StringIO()
  save_cif(target, output, minimal=minimal)
  reloaded = parse_mmcif(io.StringIO(output.getvalue()), return_type=return_type)
  assert len(reloaded) == (1 if single_model else 2)
  assert reloaded.metadata["mmcif_data_name"] == "test"
  assert reloaded.metadata["mmcif_fields"] == target.metadata["mmcif_fields"]
  assert reloaded.models()[0].metadata["mmcif_fields"] == target.metadata["mmcif_fields"]
  assert len(reloaded.models()[0]) == len(model)


def test_mmcif_metadata_rejects_inconsistent_loop_columns():
  structure = make_structure(MIXED_BACKBONE_ATOMS)
  structure.metadata["mmcif_fields"] = {"_audit_author.name": ["A", "B"], "_audit_author.pdbx_ordinal": ["1"]}
  with pytest.raises(ValueError, match="must have the same length"):
    save_cif(structure, io.StringIO())
