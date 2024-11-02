Neurosnap Tools
===============

.. image:: https://img.shields.io/github/license/KeaunAmani/neurosnap?color=%234361EE
   :target: https://github.com/KeaunAmani/neurosnap/blob/master/LICENSE
   :alt: GitHub License

.. image:: https://img.shields.io/github/created-at/KeaunAmani/neurosnap?color=%234361EE
   :target: https://img.shields.io/github/created-at/KeaunAmani/neurosnap?color=%234361EE
   :alt: GitHub Created At

.. image:: https://img.shields.io/github/last-commit/KeaunAmani/neurosnap?color=%234361EE
   :target: https://img.shields.io/github/last-commit/KeaunAmani/neurosnap?color=%234361EE
   :alt: GitHub Last Commit

.. image:: https://img.shields.io/discord/1014770343883309076
   :target: https://discord.gg/2yDZX6rTh4
   :alt: Discord


Collection of useful bioinformatic functions and tools for various computational biology pipelines. Primarily designed for working with amino acid sequences and chemical structures.

This a package developed by Keaun Amani at `<https://neurosnap.ai>`_. You are welcome to use this code and contribute as you see fit. We are currently working on expanding this package as well to add support for more common functions.

..
    reference link for how to write .rst files, very useful
    https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   modules


Contributing
=============

We welcome contributions to this package. If you have a feature that you want to code or have added, submit a pull request or an issue.


.. code-block:: shell

   # set up a virtualenv
   python3.8 -m venv venv

   # this step might differ depending on your shell
   source venv/bin/activate

   pip install --editable .[dev]

Building documentation
----------------------

To build documentation, enter your virtual environment and run :code:`make docs` from the root of the repository.

Then, open :code:`docs/build/html/index.html` in a web browser.

Installation
============

.. code-block:: sh

   pip install -U --no-cache-dir git+https://github.com/NeurosnapInc/neurosnap.git

Tutorials
=========

Various interactive jupyter notebooks can be found in the `example_notebooks directory <https://github.com/NeurosnapInc/neurosnap/tree/main/example_notebooks>`_ of this repository. For additional tutorials check out the `Official Neurosnap Blog <https://neurosnap.ai/blog>`_ or `join our discord server <https://discord.gg/2yDZX6rTh4>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

..
   this is a comment block btw
   the search page above doesn't work due to a bug in the theme:
   https://github.com/readthedocs/sphinx_rtd_theme/issues/998

   its not terrible because the rtd_theme has a search bar in the top left
   already. However, if we switch to a different theme, then it might not be the
   case. Also, the search page is pretty standard in sphinx so we should leave
   it in, in case we switch themes or anything.
