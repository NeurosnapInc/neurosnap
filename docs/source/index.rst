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

Neurosnap Tools (Python SDK)
----------------------------

`Neurosnap Tools <https://github.com/NeurosnapInc/neurosnap>`_ is the official
Python package for working with Neurosnap programmatically. It includes:

* A lightweight SDK wrapper for Neurosnap API endpoints.
* Utility modules for common bioinformatics and cheminformatics tasks.
* Building blocks for sequence-centric and structure-aware pipeline development.

Use this package when you need reproducible scripts, batch processing, notebook
workflows, or integration with lab and MLOps infrastructure.

Documentation Map
-----------------

Core SDK modules:

* `API client <neurosnap.html#module-neurosnap.api>`_
* `Protein utilities <neurosnap.html#module-neurosnap.protein>`_
* `MSA utilities <neurosnap.html#module-neurosnap.msa>`_
* `Chemical utilities <neurosnap.html#module-neurosnap.chemicals>`_
* `Conformer generation <neurosnap.html#module-neurosnap.conformers>`_
* `Nucleotide utilities <neurosnap.html#module-neurosnap.nucleotide>`_
* `Constants and residue metadata <neurosnap.html#module-neurosnap.constants>`_
* `Rendering utilities <neurosnap.html#module-neurosnap.rendering>`_
* `Logging utilities <neurosnap.html#module-neurosnap.log>`_

Algorithm modules:

* `AlphaFold metrics <neurosnap.html#module-neurosnap.algos.alphafold>`_
* `pDockQ scoring <neurosnap.html#module-neurosnap.algos.pdockq>`_
* `ipSAE metrics <neurosnap.html#module-neurosnap.algos.ipsae>`_
* `LDDT scoring <neurosnap.html#module-neurosnap.algos.LDDT>`_
* `Electrostatic interface scoring <neurosnap.html#module-neurosnap.algos.ec_interface>`_
* `EvoEF2 force field <neurosnap.html#module-neurosnap.algos.evoef2>`_
* `ClusterProt clustering <neurosnap.html#module-neurosnap.algos.clusterprot>`_
* `Kluster workflow <neurosnap.html#module-neurosnap.algos.kluster>`_
* `Amber relax <neurosnap.html#module-neurosnap.algos.amber_relax>`_

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
