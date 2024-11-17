[![GitHub license](https://img.shields.io/github/license/KeaunAmani/neurosnap?color=%234361EE)](https://github.com/KeaunAmani/neurosnap/blob/master/LICENSE)
![GitHub Created At](https://img.shields.io/github/created-at/KeaunAmani/neurosnap?color=%234361EE)
![GitHub last commit](https://img.shields.io/github/last-commit/KeaunAmani/neurosnap?color=%234361EE)
[![Discord](https://img.shields.io/discord/1014770343883309076)](https://discord.gg/2yDZX6rTh4)

# Neurosnap Tools
[![Neurosnap Header](https://raw.githubusercontent.com/NeurosnapInc/neurosnap/refs/heads/main/assets/header.webp)](https://neurosnap.ai/)

Collection of useful bioinformatic functions and tools for various computational biology pipelines. Primarily designed for working with amino acid sequences and chemical structures.

This a package developed by Keaun Amani at [neurosnap.ai](https://neurosnap.ai/). You are welcome to use this code and contribute as you see fit. We are currently working on expanding this package as well to add support for more common functions.

# Contributions
We welcome contributions to this package. If you have a feature that you want to code or have added, submit a pull request or an issue.

```sh
# set up a virtualenv
python -m venv .venv

# this step might differ depending on your shell
source .venv/bin/activate

pip install --editable .[dev]
```

## Building documentation
To build documentation, enter your virtual environment and run `make docs` from
the root of the repository.

Then, open `docs/build/html/index.html` in a web browser.

# Installation
```sh
# current stable version
pip install -U --no-cache-dir neurosnap

# latest version
pip install -U --no-cache-dir git+https://github.com/NeurosnapInc/neurosnap.git
```

# Tutorials
Various interactive jupyter notebooks can be found in the [example_notebooks directory](https://github.com/NeurosnapInc/neurosnap/tree/main/example_notebooks) of this repository. For additional tutorials check out the [Official Neurosnap Blog](https://neurosnap.ai/blog) or [join our discord server](https://discord.gg/2yDZX6rTh4).
<!-- TODO: Add a list of all available tutorials here -->

<!-- # Usage
Note that all functions have their own documentation within the code. We recommend checking those documentation blocks when confused. -->
