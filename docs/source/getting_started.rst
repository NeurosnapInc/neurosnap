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

   # optional extras for structure clustering workflows
   pip install -U "neurosnap[clusterprot]"

Protein Class Basics
--------------------

The :mod:`neurosnap.protein` module provides a feature-rich ``Protein`` class
for loading, inspecting, and analyzing structures.

.. code-block:: python

   from neurosnap.protein import Protein

   # Accepts local files, PDB IDs, and UniProt IDs
   prot = Protein("1CRN")

   print(prot)
   print("Models:", prot.models())
   print("Chains:", prot.chains())

   # Example structure-aware calculations
   dmat = prot.calculate_distance_matrix()
   rog = prot.calculate_rog()
   print("Distance matrix shape:", dmat.shape)
   print("Radius of gyration:", rog)

MSA Module Basics
-----------------

The :mod:`neurosnap.msa` module covers sequence-alignment I/O and common MSA
operations.

.. code-block:: python

   from neurosnap.msa import read_msa, consensus_sequence

   # Stream records from FASTA/A3M
   records = list(read_msa("example.a3m", size=500))
   names, seqs = zip(*records)

   print("Loaded sequences:", len(records))
   print("Consensus:", consensus_sequence(list(seqs)))

Algorithm Highlights
--------------------

Beyond core modules, the SDK includes production-ready algorithm components.

ClusterProt
^^^^^^^^^^^

:mod:`neurosnap.algos.clusterprot` clusters proteins by structural similarity
and produces 2D/1D projections for analysis.

.. code-block:: python

   from neurosnap.algos.clusterprot import ClusterProt

   results = ClusterProt(
     ["a.pdb", "b.pdb", "c.pdb", "d.pdb", "e.pdb"],
     proj_1d_algo="umap",
   )
   print("Cluster labels:", results["cluster_labels"])

EvoEF2
^^^^^^

:mod:`neurosnap.algos.evoef2` provides a fast Python implementation of EvoEF2
for stability and interface scoring. For many scoring-driven workflows, this is
a powerful free/open alternative to PyRosetta-based setups.

.. code-block:: python

   from neurosnap.algos.evoef2 import calculate_stability, calculate_binding

   stability = calculate_stability("complex.pdb")
   binding = calculate_binding("complex.pdb", split1=["A"], split2=["B"])

   print("Stability total:", stability["total"])
   print("Binding DG:", binding["DG_bind"])

Next Steps
----------

* Continue with :doc:`modules` for full API/module reference.
* Explore additional examples in the
  `example_notebooks directory <https://github.com/NeurosnapInc/neurosnap/tree/main/example_notebooks>`_.
