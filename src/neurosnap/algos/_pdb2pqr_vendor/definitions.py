"""Python-native biomolecular residue topology definitions.

.. codeauthor::  Jens Erik Nielsen
.. codeauthor::  Todd Dolinsky
.. codeauthor::  Yong Huang
"""

import copy
import re

from . import residue, structures
from .dat.definitions_data import DEFINITION_DATA


class Definition:
    """Force field topology definitions.

    The Definition class contains the structured definitions found in the files
    and several mappings for easy access to the information.
    """

    def __init__(self, aa_data=None, na_data=None, patch_data=None):
        """Initialize object from Python-native definition data."""
        self.map = {}
        self.patches = {}
        aa_data = DEFINITION_DATA["aa"] if aa_data is None else aa_data
        na_data = DEFINITION_DATA["na"] if na_data is None else na_data
        patch_data = DEFINITION_DATA["patches"] if patch_data is None else patch_data
        for definition_map in (aa_data, na_data):
            for residue_name, residue_data in definition_map.items():
                self.map[residue_name] = _build_definition_residue(residue_data)
        # Apply specific patches to the reference object, allowing users
        # to specify protonation states in the PDB file
        for patch_data_entry in patch_data:
            patch = _build_patch(patch_data_entry)
            if patch.newname != "":
                # Find all residues matching applyto
                resnames = list(self.map.keys())
                for name in resnames:
                    regexp = re.compile(patch.applyto).match(name)
                    if not regexp:
                        continue
                    newname = patch.newname.replace("*", name)
                    self.add_patch(patch, name, newname)
            # Either way, make sure the main patch name is available
            self.add_patch(patch, patch.applyto, patch.name)

    def add_patch(self, patch, refname, newname):
        """Add a patch to a topology definition residue.

        :param patch:  the patch object to add
        :type patch:  Patch
        :param refname:  the name of the object to add the patch to
        :type refname:  str
        :param newname:  the name of the new (patched) object
        :type newname:  str
        """
        try:
            aadef = self.map[refname]  # The reference
            patch_residue = copy.deepcopy(aadef)
            # Add atoms from patch
            for atomname in patch.map:
                patch_residue.map[atomname] = patch.map[atomname]
                for bond in patch.map[atomname].bonds:
                    if bond not in patch_residue.map:
                        continue
                    if atomname not in patch_residue.map[bond].bonds:
                        patch_residue.map[bond].bonds.append(atomname)
            # Rename atoms as directed
            for key in patch.altnames:
                patch_residue.altnames[key] = patch.altnames[key]
            # Remove atoms as directed
            for remove in patch.remove:
                if not patch_residue.has_atom(remove):
                    continue
                removebonds = patch_residue.map[remove].bonds
                del patch_residue.map[remove]
                for bond in removebonds:
                    if remove in patch_residue.map[bond].bonds:
                        patch_residue.map[bond].bonds.remove(remove)
            # Add the new dihedrals
            for dihedral in patch.dihedrals:
                patch_residue.dihedrals.append(dihedral)
            # Point at the new reference
            self.map[newname] = patch_residue
            # Store the patch
            self.patches[newname] = patch
        except KeyError:  # Just store the patch
            self.patches[newname] = patch


class Patch:
    """Residue patches for structure topologies."""

    def __init__(self):
        self.name = ""
        self.applyto = ""
        self.map = {}
        self.remove = []
        self.altnames = {}
        self.dihedrals = []
        self.newname = ""

    def __str__(self):
        text = f"{self.name}\n"
        text += f"Apply to: {self.applyto}\n"
        text += "Atoms to add: \n"
        for atom in self.map:
            text += f"\t{self.map[atom]!s}\n"
        text += "Atoms to remove: \n"
        for remove in self.remove:
            text += f"\t{remove}\n"
        text += "Alternate naming map: \n"
        text += f"\t{self.altnames}\n"
        return text


class DefinitionResidue(residue.Residue):
    """Force field toplogy representation for a residue."""

    def __init__(self):
        self.name = ""
        self.dihedrals = []
        self.map = {}
        self.altnames = {}

    def __str__(self):
        text = f"{self.name}\n"
        text += "Atoms: \n"
        for atom in self.map:
            text += f"\t{self.map[atom]!s}\n"
        text += "Dihedrals: \n"
        for dihedral in self.dihedrals:
            text += f"\t{dihedral}\n"
        text += "Alternate naming map: \n"
        text += f"\t{self.altnames}\n"
        return text

    def get_nearest_bonds(self, atomname):
        """Get bonded atoms near a given atom.

        :param atomname:  name of specific atom
        :type atomname:  str
        :return:  list of nearby bonded atom names
        :rtype:  [str]
        """
        bonds = []
        lev2bonds = []
        atom = self.map[atomname]
        # Get directly bonded (length = 1) atoms
        for bondedatom in atom.bonds:
            if bondedatom not in bonds:
                bonds.append(bondedatom)
        # Get bonded atoms 2 bond lengths away
        for bondedatom in atom.bonds:
            for bond2 in self.map[bondedatom].bonds:
                if bond2 not in bonds and bond2 != atomname:
                    bonds.append(bond2)
                    lev2bonds.append(bond2)
        # Get bonded atoms 3 bond lengths away
        for lev2atom in lev2bonds:
            for bond3 in self.map[lev2atom].bonds:
                if bond3 not in bonds:
                    bonds.append(bond3)
        return bonds


class DefinitionAtom(structures.Atom):
    """Store force field atom topology definitions."""

    def __init__(self, name=None, x=None, y=None, z=None):
        """Initialize class.

        :param name:  atom name
        :type name:  str
        :param x:  x-coordinate
        :type x:  float
        :param y:  y-coordinate
        :type y:  float
        :param z:  z-coordinate
        :type z:  float
        """
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        if name is None:
            self.name = ""
        if x is None:
            self.x = 0.0
        if y is None:
            self.y = 0.0
        if z is None:
            self.z = 0.0
        self.bonds = []

    def __str__(self):
        text = f"{self.name}: {self.x:.3f} {self.y:.3f} {self.z:.3f}"
        for bond in self.bonds:
            text += f" {bond}"
        return text

    @property
    def is_backbone(self):
        """Identify whether atom is in backbone.

        :return:  true if atom name is in backbone, otherwise false
        :rtype:  bool
        """
        return self.name in structures.BACKBONE


def _build_definition_residue(data):
    residue_obj = DefinitionResidue()
    residue_obj.name = data["name"]
    residue_obj.dihedrals = list(data.get("dihedrals", []))
    residue_obj.altnames = dict(data.get("altnames", {}))
    for atom_name, atom_data in data.get("atoms", {}).items():
        residue_obj.map[atom_name] = _build_definition_atom(atom_data)
    return residue_obj


def _build_patch(data):
    patch_obj = Patch()
    patch_obj.name = data.get("name", "")
    patch_obj.applyto = data.get("applyto", "")
    patch_obj.newname = data.get("newname", "")
    patch_obj.remove = list(data.get("remove", []))
    patch_obj.altnames = dict(data.get("altnames", {}))
    patch_obj.dihedrals = list(data.get("dihedrals", []))
    for atom_name, atom_data in data.get("atoms", {}).items():
        patch_obj.map[atom_name] = _build_definition_atom(atom_data)
    return patch_obj


def _build_definition_atom(data):
    atom = DefinitionAtom(
        name=data.get("name", ""),
        x=float(data.get("x", 0.0)),
        y=float(data.get("y", 0.0)),
        z=float(data.get("z", 0.0)),
    )
    atom.bonds = list(data.get("bonds", []))
    return atom
