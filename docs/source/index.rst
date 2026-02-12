Neurosnap Documentation
=======================

Neurosnap combines a production web application for AI-native molecular discovery
with a Python package for scripting, automation, and custom analysis workflows.
This documentation covers both.

Neurosnap Platform
------------------

`Neurosnap.ai <https://neurosnap.ai/>`_ is a browser-based platform for
computational biology and cheminformatics, with support for structure prediction,
binder design, docking, screening, and downstream analysis workflows.

Core platform areas:

* `Services <https://neurosnap.ai/services>`_: hosted tools and models for
  protein and small-molecule workflows.
* `Use Cases <https://neurosnap.ai/#use-cases>`_: antibody engineering, peptide discovery,
  enzyme engineering, and small-molecule discovery.
* `NeuroFold <https://neurosnap.ai/neurofold>`_: structure prediction and
  design-focused workflows.
* `API <https://neurosnap.ai/blog/post/full-neurosnap-api-tutorial-the-quick-easy-api-for-bioinformatics/66b00dacec3f2aa9b4be703a>`_: programmatic access for automation and
  integration into internal pipelines.
* `Security <https://neurosnap.ai/security>`_: platform security posture and
  data-handling commitments.

Neurosnap SDK (Python Package)
------------------------------

`Neurosnap SDK <https://github.com/NeurosnapInc/neurosnap>`_ is the official
Python package for working with Neurosnap programmatically. It includes:

* A lightweight SDK wrapper for Neurosnap API endpoints.
* Utility modules for common bioinformatics and cheminformatics tasks.
* Building blocks for sequence-centric and structure-aware pipeline development.

Use this package when you need reproducible scripts, batch processing, notebook
workflows, or integration with lab and MLOps infrastructure.

Documentation Map
-----------------

**Core SDK modules:**

* `API client <neurosnap.api.html>`_
* `Protein utilities <neurosnap.protein.html>`_
* `MSA utilities <neurosnap.msa.html>`_
* `Chemical utilities <neurosnap.chemicals.html>`_
* `Conformer generation <neurosnap.conformers.html>`_
* `Nucleotide utilities <neurosnap.nucleotide.html>`_
* `Constants and residue metadata <neurosnap.constants.html>`_
* `Rendering utilities <neurosnap.rendering.html>`_
* `Logging utilities <neurosnap.log.html>`_

**Algorithm modules:**

* `AlphaFold metrics <neurosnap.algos.alphafold.html>`_
* `pDockQ scoring <neurosnap.algos.pdockq.html>`_
* `ipSAE metrics <neurosnap.algos.ipsae.html>`_
* `LDDT scoring <neurosnap.algos.LDDT.html>`_
* `Electrostatic interface scoring <neurosnap.algos.ec_interface.html>`_
* `EvoEF2 force field <neurosnap.algos.evoef2.html>`_
* `ClusterProt clustering <neurosnap.algos.clusterprot.html>`_
* `Kluster workflow <neurosnap.algos.kluster.html>`_
* `Amber relax <neurosnap.algos.amber_relax.html>`_

.. toctree::
   :hidden:
   :maxdepth: 2

   modules

Learning Resources
------------------

* `Example notebooks <https://github.com/NeurosnapInc/neurosnap/tree/main/example_notebooks>`_
* `Neurosnap Blog <https://neurosnap.ai/blog>`_
* `Discord community <https://discord.gg/2yDZX6rTh4>`_

.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
