[![GitHub license](https://img.shields.io/github/license/NeurosnapInc/neurosnap?color=%234361EE)](https://github.com/NeurosnapInc/neurosnap/blob/master/LICENSE)
![GitHub Created At](https://img.shields.io/github/created-at/NeurosnapInc/neurosnap?color=%234361EE)
![GitHub last commit](https://img.shields.io/github/last-commit/NeurosnapInc/neurosnap?color=%234361EE)
[![Discord](https://img.shields.io/discord/1014770343883309076)](https://discord.gg/2yDZX6rTh4)

# Neurosnap SDK
[![Neurosnap Header](https://raw.githubusercontent.com/NeurosnapInc/neurosnap/refs/heads/main/assets/header.webp)](https://neurosnap.ai/)

Neurosnap SDK is a collection of utilities for bioinformatics, structural biology, and cheminformatics workflows, with strong support for amino acid sequences and molecular structures.

This a package developed by Keaun Amani at [neurosnap.ai](https://neurosnap.ai/). You are welcome to use this code and contribute as you see fit. We are currently working on expanding this package as well to add support for more common functions.

# Installation
```sh
# current stable version
pip install -U --no-cache-dir neurosnap

# latest version
pip install -U --no-cache-dir git+https://github.com/NeurosnapInc/neurosnap.git

# latest version + ClusterProt dependencies
pip install -U --no-cache-dir "neurosnap @ git+https://github.com/NeurosnapInc/neurosnap.git#egg=neurosnap[clusterprot]"

# latest version + Kluster dependencies
pip install -U --no-cache-dir "neurosnap @ git+https://github.com/NeurosnapInc/neurosnap.git#egg=neurosnap[kluster]"

# latest version + development dependencies
pip install -U --no-cache-dir "neurosnap @ git+https://github.com/NeurosnapInc/neurosnap.git#egg=neurosnap[dev]"
```

# Documentation
Official documentation can be found here: [https://neurosnap.ai/docs/](https://neurosnap.ai/docs/).

## Building documentation
To build documentation, enter your virtual environment and run `make docs` from
the root of the repository.

Then, open `docs/build/html/index.html` in a web browser.

# Tutorials
Various interactive jupyter notebooks can be found in the [example_notebooks directory](https://github.com/NeurosnapInc/neurosnap/tree/main/example_notebooks) of this repository. For additional tutorials check out the [Official Neurosnap Blog](https://neurosnap.ai/blog) or [join our discord server](https://discord.gg/2yDZX6rTh4).

# Contributions
We welcome contributions to this package. If you have a feature that you want to code or have added, submit a pull request or an issue.

```sh
# set up a virtualenv
python -m venv .venv

# this step might differ depending on your shell
source .venv/bin/activate

pip install --editable .[dev]
```

# Citations
If you found this SDK helpful, please feel free to cite it using the following.

```BibTeX
@misc{amani-2024,
	author = {Amani, Keaun and Amirabadi, Danial Gharaie},
	title = {{Neurosnap SDK}},
	year = {2024},
	url = {https://github.com/NeurosnapInc/neurosnap},
}
```
