Structure Interactions & Confidence Metrics
===========================================

The Neurosnap SDK provides a robust and flexible framework for identifying physical interactions within and between molecular entities, and for analyzing predicted structure confidence metrics.

These features are exposed through high-level orchestrators, legacy compatibility wrappers, and dedicated report containers.

Overview
--------

Declared versus detected
~~~~~~~~~~~~~~~~~~~~~~~~

A structure carries two tables of connectivity that a file may state outright:

* ``Structure.bonds`` holds persistent topology, each row tagged with a
  :class:`~neurosnap.structure.structure.BondType` of ``COVALENT``,
  ``DISULFIDE``, ``METAL_COORDINATION``, or ``OTHER``.
* ``Structure.interactions`` holds noncovalent contacts a file declared, each
  row tagged with an :class:`~neurosnap.structure.structure.InteractionType` of
  ``IONIC``, ``HYDROGEN_BOND``, ``SALT_BRIDGE``, or ``OTHER_NONCOVALENT``.

The analysis described on this page is separate from both. It *detects*
interactions from geometry and chemistry and returns them as a report, leaving
the two tables untouched. Where a detected interaction was also declared by the
source file, its ``evidence`` is reported as ``explicit``. The methods are named
``detect_*`` for that reason: ``structure.detect_interactions()`` computes a
result, whereas ``structure.interactions`` is a stored table.

Interaction analysis allows you to detect various structural contacts and bonds based on classical, geometric, and heuristic rules:

* **Hydrogen Bonds**: Identifies donor-acceptor pairings and attached hydrogens using distance and optional angle cutoffs.
* **Salt Bridges**: Identifies ionic contacts between oppositely charged residue centers or formal charges on aligned ligands.
* **Disulfide Bonds**: Detects covalent bonds between cysteine sulfur (SG) atoms.
* **Metal Coordination**: Analyzes coordination geometry and deviations around metal centers.
* **Contacts & Clashes**: Detects standard spatial proximity and steric clashes.
* **Covalent Candidates**: Identifies cross-entity covalent bond candidates.

Complete Local Example
----------------------

The following example demonstrates how to parse a synthetic PDB file from an in-memory string, define interaction entities, perform the interaction analysis, and extract a pandas DataFrame of the results.

.. code-block:: python

   import io
   import pandas as pd
   from neurosnap.io.pdb import parse_pdb
   from neurosnap.structure import analyze_interactions, InteractionEntity

   # 1. Define synthetic PDB data in memory
   pdb_data = """
   ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
   ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
   ATOM      3  C   ALA A   1       1.958   1.410   0.000  1.00 20.00           C
   ATOM      4  O   ALA A   1       1.200   2.200   0.000  1.00 20.00           O
   HETATM    5  C1  BEN B   2       5.000   0.000   0.000  1.00 20.00           C
   HETATM    6  C2  BEN B   2       6.400   0.000   0.000  1.00 20.00           C
   """

   # 2. Parse the PDB string into a Structure object
   ensemble = parse_pdb(io.StringIO(pdb_data), return_type="ensemble")
   structure = ensemble.first()

   # 3. Define the interacting entities using structure atom indices
   # Entity 1: The ALA residue (atoms 0 to 3)
   entity1 = InteractionEntity(name="protein_residue", atom_indices=[0, 1, 2, 3])
   # Entity 2: The BEN ligand (atoms 4 and 5)
   entity2 = InteractionEntity(name="ligand", atom_indices=[4, 5])

   # 4. Analyze interactions between the two entities
   report = analyze_interactions(structure, entity1, entity2, interaction_types=["contact"])

   # 5. Export findings to a pandas DataFrame
   df = report.to_dataframe()
   print(df[["interaction_id", "interaction_type", "entity1", "entity2", "distance_a"]])

.. note::
   The ``parse_pdb`` function with ``return_type="ensemble"`` returns a ``StructureEnsemble`` object. Calling ``ensemble.first()`` retrieves the first ``Structure`` model from this ensemble for downstream single-model interaction analysis.

Supported Interaction Types
---------------------------

The interaction detection rules, default cutoffs, required topology, and possible evidence levels are summarized below:

.. list-table:: Supported Interaction Types
   :widths: 20 20 30 30
   :header-rows: 1

   * - Interaction Type
     - Default Cutoff
     - Required Topology
     - Possible Evidence Levels
   * - ``hydrogen_bond``
     - 3.5 Å (donor-acceptor distance), >= 130.0° (donor-H-acceptor angle)
     - Nitrogen, Oxygen, or Sulfur donors/acceptors. Uses RDKit chemical feature factory definitions for non-polymer ligands.
     - ``explicit`` (declared in ``interactions``), ``detected`` (hydrogen found), ``candidate`` (distance-only cutoff when ``include_candidates=True``)
   * - ``disulfide``
     - 2.2 Å
     - Cysteine residues with sulfur atoms (SG).
     - ``explicit`` (``BondType.DISULFIDE`` in ``bonds``), ``detected`` (any SG-SG bond), ``candidate`` (distance-only)
   * - ``salt_bridge``
     - 4.0 Å
     - Charged residue centers (NZ for LYS, NH1/NH2 for ARG, positively charged HIS; OD1/OD2 for ASP, OE1/OE2 for GLU) or formal charges on aligned ligands.
     - ``explicit`` (declared in ``interactions``), ``detected``
   * - ``metal_coordination``
     - 2.8 Å
     - Any element in :data:`neurosnap.constants.chemistry.METAL_ELEMENTS`, with coordinating donor atoms (O, N, S).
     - ``explicit``, ``detected``, ``candidate``
   * - ``contact``
     - 4.5 Å
     - Any non-hydrogen atom (unless ``include_hydrogens=True``).
     - ``distance_cutoff``
   * - ``vdw_contact``
     - Sum of Bondi VDW radii + 0.5 Å
     - Any atom pairs.
     - ``vdw_contact``
   * - ``clash``
     - Bondi VDW overlap >= 0.4 Å
     - Any cross-entity atom pairs.
     - ``vdw_overlap``
   * - ``vdw_clash``
     - Gap < -0.4 Å
     - Any non-bonded atom pairs across the structure.
     - ``vdw_overlap``
   * - ``covalent``
     - [0.8, 1.2] * sum of covalent radii
     - Any non-metal atom pairs.
     - ``explicit``, ``candidate``

Normalized Output Columns
-------------------------

The pandas DataFrame returned by calling ``InteractionReport.to_dataframe()`` contains the following normalized columns:

.. list-table:: Interaction DataFrame Columns
   :widths: 25 20 55
   :header-rows: 1

   * - Column Name
     - Type
     - Description
   * - ``interaction_id``
     - string
     - Unique, deterministically-assigned interaction identifier (e.g., ``int_1``).
   * - ``interaction_type``
     - string
     - The type of interaction (e.g., ``hydrogen_bond``, ``salt_bridge``).
   * - ``evidence``
     - string
     - Detection confidence level (e.g., ``explicit``, ``detected``, ``candidate``).
   * - ``entity1`` / ``entity2``
     - string
     - The names of the interacting entities (as defined in `InteractionEntity`).
   * - ``atom_index1`` / ``atom_index2``
     - integer
     - 0-indexed structure atom indices for the interacting atoms.
   * - ``chain1`` / ``chain2``
     - string
     - Chain identifiers.
   * - ``res_id1`` / ``res_id2``
     - integer
     - Residue sequence numbers.
   * - ``ins_code1`` / ``ins_code2``
     - string
     - Insertion codes.
   * - ``res_name1`` / ``res_name2``
     - string
     - Residue names (e.g., ``ALA``, ``LYS``).
   * - ``atom_name1`` / ``atom_name2``
     - string
     - Atom names (e.g., ``CA``, ``NZ``).
   * - ``element1`` / ``element2``
     - string
     - Chemical elements (e.g., ``C``, ``N``, ``O``).
   * - ``role1`` / ``role2``
     - string
     - The role of each atom in the interaction (e.g., ``donor``, ``acceptor``, ``positive``, ``negative``, ``metal``).
   * - ``distance_a``
     - float
     - The distance between the interacting atoms in Å.
   * - ``angle_deg``
     - float
     - The angle of the interaction in degrees (e.g., donor-H-acceptor angle).
   * - ``vdw_gap_a``
     - float
     - The Van der Waals gap/overlap in Å.
   * - ``source``
     - string
     - The module name or engine source generating the interaction (e.g., ``geometric_rules``).
   * - ``rule_set``
     - string
     - The name of the rule set (defaults to ``default``).
   * - ``rule_version``
     - string
     - The version of the rule set (defaults to ``1.0``).
   * - ``model_id``
     - integer
     - The model index within the ensemble (1-indexed).
   * - ``details``
     - dict
     - Extra metadata (e.g. hydrogen atom index, donor-H and H-acceptor distance).

The pandas DataFrame returned by calling ``InteractionReport.coordination_centers_dataframe()`` contains the following columns:

.. list-table:: Coordination Center DataFrame Columns
   :widths: 25 20 55
   :header-rows: 1

   * - Column Name
     - Type
     - Description
   * - ``center_id``
     - string
     - Unique, deterministically-assigned center identifier (e.g., ``coord_1``).
   * - ``metal_atom_index``
     - integer
     - 0-indexed structure atom index of the coordinating metal.
   * - ``entity``
     - string
     - The name of the entity containing the metal.
   * - ``chain``
     - string
     - Chain identifier of the metal.
   * - ``res_id``
     - integer
     - Residue sequence number of the metal.
   * - ``ins_code``
     - string
     - Insertion code.
   * - ``res_name``
     - string
     - Residue name of the metal.
   * - ``atom_name``
     - string
     - Atom name of the metal.
   * - ``element``
     - string
     - Chemical element of the metal.
   * - ``coordination_number``
     - integer
     - Number of coordinating donor atoms.
   * - ``donor_atom_indices``
     - sequence
     - List of 0-indexed structure atom indices of the coordinating donor atoms.
   * - ``donor_elements``
     - sequence
     - List of chemical elements of the donor atoms.
   * - ``geometry``
     - string
     - Classified coordination geometry (e.g., ``tetrahedral``, ``octahedral``, ``linear``, ``trigonal planar``, ``square planar``).
   * - ``geometry_deviation_deg``
     - float
     - The root-mean-square deviation (RMSD) of pairwise donor-metal-donor angles from the ideal geometry in degrees.
   * - ``evidence``
     - string
     - Best evidence level among coordinating donors (e.g., ``explicit``, ``detected``, ``candidate``).
   * - ``rule_set``
     - string
     - The name of the rule set.
   * - ``rule_version``
     - string
     - The version of the rule set.
   * - ``model_id``
     - integer
     - The model index within the ensemble (1-indexed).

Ligand Templates and Atom Ordering
----------------------------------

When performing interaction analysis involving non-polymer ligands, the SDK relies on RDKit for chemical typing (such as determining formal charges, donor/acceptor roles, and hybridization). 

.. important::
   The 0-indexed atom order of the RDKit molecule supplied as ``rdkit_mol`` must match the 1-to-1 order of the structure atom indices specified in ``InteractionEntity.atom_indices``.

If the atom indices are sorted as ``[100, 101, 102, 103]``, then RDKit atom index ``0`` corresponds to structure atom index ``100``, RDKit atom index ``1`` corresponds to structure atom index ``101``, and so on. Mismatched ordering will result in incorrect interaction typing.

.. code-block:: python

   from rdkit import Chem
   from neurosnap.structure import InteractionEntity

   # Assume ligand atoms are at indices 50 to 55 in the Structure
   atom_indices = list(range(50, 56))

   # The SMILES string must be constructed so that its atom order
   # matches the sequential atom positions in the PDB structure.
   rdkit_mol = Chem.MolFromSmiles("CCOCCO")

   ligand_entity = InteractionEntity(
       name="ligand_1",
       atom_indices=atom_indices,
       rdkit_mol=rdkit_mol
   )

Metal Coordination and Geometry Deviation
-----------------------------------------

The coordination geometry of metal centers is classified for coordination numbers between 2 and 6. The SDK compares the distribution of pairwise donor-metal-donor angles against ideal geometric templates:

* **CN=2**: ``linear`` (ideal: 180.0°)
* **CN=3**: ``trigonal planar`` (ideal: 120.0° x 3)
* **CN=4**: ``tetrahedral`` (ideal: 109.5° x 6) or ``square planar`` (ideal: 90.0° x 4, 180.0° x 2)
* **CN=5**: ``trigonal bipyramidal`` (ideal: 90.0° x 6, 120.0° x 3, 180.0° x 1) or ``square pyramidal`` (ideal: 90.0° x 8, 180.0° x 2)
* **CN=6**: ``octahedral`` (ideal: 90.0° x 12, 180.0° x 3)

The best-fitting template is determined by minimizing the root-mean-square deviation (RMSD) between the actual sorted angles and the ideal sorted angles. This RMSD value is returned in the ``geometry_deviation_deg`` column.

.. code-block:: python

   from neurosnap.structure import analyze_interactions

   # Retrieve metal coordination reports
   report = analyze_interactions(structure, interaction_types=["metal_coordination"])
   coord_df = report.coordination_centers_dataframe()
   for idx, row in coord_df.iterrows():
       print(f"Metal: {row['element']} | Geometry: {row['geometry']} | Dev: {row['geometry_deviation_deg']:.2f}°")

Explicit Records in PDB Files
-----------------------------

``SSBOND`` and ``LINK`` records are read into ``Structure.bonds`` with the
matching :class:`~neurosnap.structure.structure.BondType`, and written back out
on save, so a bond classification survives a round trip:

* ``SSBOND`` maps to ``BondType.DISULFIDE`` with ``bond_order = 1``.
* ``LINK`` maps to ``BondType.METAL_COORDINATION`` with ``bond_order = 0`` when
  either endpoint is a metal, and to ``BondType.COVALENT`` otherwise.

``CONECT`` records cannot express either category: they carry no bond-type field
and cannot represent ``bond_order = 0``. Metal-coordination bonds are therefore
written as ``LINK`` and omitted from ``CONECT``, so re-reading a saved file does
not silently downgrade them to ordinary covalent bonds.

A file may repeat the same ``LINK`` pair several times. Because a bond table
holds one row per atom pair, those repeats collapse into a single bond.

pLDDT Distribution Summaries
----------------------------

The SDK provides granular summaries of the predicted Local Distance Difference Test (pLDDT) confidence metric. Calling ``summarize_plddt`` automatically detects the input scale ([0, 1] vs [0, 100]) and returns a ``PLDDTReport`` with pre-computed summaries.

.. code-block:: python

   from neurosnap.structure.confidence import summarize_plddt

   # Compute the pLDDT confidence report
   report = summarize_plddt(structure, source="b_factor", scale="auto", boundaries=(50, 70, 90))

   # The 'distribution' DataFrame contains counts and percentages for each bin
   print(report.distribution)

Example distribution output:

.. code-block:: text

            count  percentage
   <50          5        5.00
   50-70       15       15.00
   70-90       30       30.00
   >=90        50       50.00

Explicit Limitations
--------------------

When using the Neurosnap SDK for interaction and confidence analysis, keep the following limitations in mind:

1. **No Bond Probability**: The interaction engine relies on binary geometric and topological thresholds to classify contacts. It does not predict continuous bond probabilities or confidence levels for covalent/non-covalent bonds.
2. **No Binding Affinity**: The engine identifies physical contacts and geometric coordination, but does not calculate binding free energies (:math:`\Delta G`), dissociation constants (:math:`K_d`), or half-maximal inhibitory concentrations (:math:`IC_{50}`).
3. **No Automatic Protonation/Tautomer/Oxidation-State Inference**: The user is responsible for preparing the input structure (e.g. adding hydrogens, checking protonation and tautomer states). No automatic correction of chemical states is performed.
4. **No QM/MM or Energy Calculations**: All calculations are purely geometric and heuristic. No quantum mechanical (QM) or molecular mechanics (MM) force fields are evaluated.
5. **No Job or Download Functionality**: The SDK operates entirely on local coordinate files and in-memory structures. It does not interface with remote queues, hosted job schedulers, or external structural databases.
6. **Ligand Chemistry Requires a Template**: ``hydrogen_bond`` and ``salt_bridge`` type ligand donors, acceptors, and formal charges from an aligned RDKit molecule. A ligand without one is left untyped and a warning names it; polymer chemistry is still evaluated, so the analysis proceeds rather than failing.
7. **Hydrogen Bonds Need Hydrogens**: Donor-H-acceptor angles cannot be measured without explicit hydrogens. On a structure with none, the rule reports nothing and warns; pass ``include_candidates=True`` for distance-only candidates.
