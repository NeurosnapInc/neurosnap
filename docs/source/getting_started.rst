Getting Started
===============

This page is a practical quickstart for the Neurosnap SDK.

Why Neurosnap SDK?
------------------

The Neurosnap SDK gives you a Python-native workflow for structural biology,
bioinformatics, and cheminformatics tasks. For many common sequence and
structure workflows, it can act as a reliable, free, and open alternative to
PyRosetta.

Install
-------

.. code-block:: bash

   # stable release
   pip install -U neurosnap

   # latest version from GitHub
   pip install -U git+https://github.com/NeurosnapInc/neurosnap.git

   # optional extras for structure clustering workflows
   pip install -U "neurosnap[clusterprot]"

Structure Basics
----------------

The :mod:`neurosnap.io` and :mod:`neurosnap.structure` modules provide the
core structure workflow. Parse local coordinate files into a
:class:`~neurosnap.structure.structure.StructureEnsemble` or
:class:`~neurosnap.structure.structure.StructureStack`, then work with
single-model :class:`~neurosnap.structure.structure.Structure` objects.

.. code-block:: python

   from neurosnap.io.pdb import parse_pdb

   ensemble = parse_pdb("example.pdb", return_type="ensemble")
   structure = ensemble.first()

   print(ensemble)
   print(structure)
   print("Model IDs:", ensemble.model_ids)
   print("Chains:", [chain.chain_id for chain in structure.chains()])

You can derive sequences directly from chain views.

.. code-block:: python

   first_chain = structure.chains()[0]
   print("Protein sequence:", first_chain.sequence(polymer_type="protein"))

Useful structure-level calculations are available directly on the
:class:`~neurosnap.structure.structure.Structure` object and in
:mod:`neurosnap.structure`.

.. code-block:: python

   from neurosnap.structure import calculate_distance_matrix, calculate_surface_area

   dmat = calculate_distance_matrix(structure)
   com = structure.calculate_center_of_mass()
   geom = structure.calculate_geometric_center()
   rog = structure.calculate_rog()
   sasa = calculate_surface_area(structure)

   print("Distance matrix shape:", dmat.shape)
   print("Center of mass:", com)
   print("Geometric center:", geom)
   print("Radius of gyration:", rog)
   print("Surface area:", sasa)

Structure I/O Basics
--------------------

Neurosnap supports PDB, mmCIF, and SDF as first-class parsers/writers.

.. code-block:: python

   from neurosnap.io.mmcif import parse_mmcif, save_cif
   from neurosnap.io.pdb import save_pdb
   from neurosnap.io.sdf import parse_sdf, save_sdf

   cif_structure = parse_mmcif("example.cif").first()
   ligand = parse_sdf("ligand.sdf").first()

   save_pdb(cif_structure, "structure_out.pdb")
   save_cif(cif_structure, "structure_out.cif")
   save_sdf(ligand, "ligand_out.sdf")

Rendering Basics
----------------

The :mod:`neurosnap.structure.rendering` module provides fast static rendering
for notebook workflows.

.. code-block:: python

   import matplotlib.pyplot as plt
   from neurosnap.structure import render_structure_pseudo3D

   image = render_structure_pseudo3D(structure, image_size=(600, 450), style="chain_id")
   plt.imshow(image)
   plt.axis("off")
   plt.show()

Sequence Alignment Basics
-------------------------

The :mod:`neurosnap.sequence.align` module covers sequence-alignment I/O and
common MSA operations.

.. code-block:: python

   from neurosnap.sequence.align import consensus_sequence, read_msa

   records = list(read_msa("example.a3m", size=500))
   names, seqs = zip(*records)

   print("Loaded sequences:", len(records))
   print("Consensus:", consensus_sequence(list(seqs)))

Nucleotide Utilities
--------------------

The :mod:`neurosnap.sequence.nucleotide` module provides basic nucleotide
helpers.

.. code-block:: python

   from neurosnap.sequence.nucleotide import get_reverse_complement

   print(get_reverse_complement("ATGGCC"))

Algorithm Highlights
--------------------

Beyond core modules, the SDK includes production-ready algorithm components.

ClusterProt
^^^^^^^^^^^

:mod:`neurosnap.algos.clusterprot` clusters proteins by structural similarity
and produces 2D/1D projections for analysis.

.. code-block:: python

   from neurosnap.algos.clusterprot import ClusterProt
   from neurosnap.io.pdb import parse_pdb

   structures = [
     parse_pdb("a.pdb", return_type="ensemble").first(),
     parse_pdb("b.pdb", return_type="ensemble").first(),
     parse_pdb("c.pdb", return_type="ensemble").first(),
   ]

   results = ClusterProt(structures, proj_1d_algo="umap")
   print("Cluster labels:", results["cluster_labels"])

EvoEF2
^^^^^^

:mod:`neurosnap.algos.evoef2` provides a fast Python implementation of EvoEF2
for stability and interface scoring. For many scoring-driven workflows, this is
a powerful free/open alternative to PyRosetta-based setups.

.. code-block:: python

   from neurosnap.algos.evoef2 import calculate_binding, calculate_stability
   from neurosnap.io.pdb import parse_pdb

   structure = parse_pdb("complex.pdb", return_type="ensemble").first()
   chain_ids = [chain.chain_id for chain in structure.chains()]
   print("Available chains:", chain_ids)

   stability = calculate_stability(structure)
   binding = calculate_binding(structure, split1=["A"], split2=["B"])

   print("Stability total:", stability["total"])
   print("Binding DG:", binding["dg_bind"]["total"])

Next Steps
----------

* Continue with :doc:`modules` for full API/module reference.
* Explore additional examples in the
  `example_notebooks directory <https://github.com/NeurosnapInc/neurosnap/tree/main/example_notebooks>`_.
