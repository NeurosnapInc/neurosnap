# Neurosnap Tools
Collection of useful bioinformatic functions and tools for various computational biology pipelines. Primarily designed for working with amino acid sequences and chemical structures.

This a package developed by Keaun Amani at [neurosnap.ai](https://neurosnap.ai/). You are welcome to use this code and contribute as you see fit. We are currently working on expanding this package as well to add support for more common functions.

## Installation
```sh
## ensure you have openbabel installed for the python bindings
# debian/ubuntu
sudo apt-get install openbabel
# arch
sudo pacman -S python-openbabel

## install the package
pip install -U --no-cache-dir --force-reinstall git+https://github.com/KeaunAmani/neurosnap.git
```

## Usage
Note that all functions have their own documentation within the code. We recommend checking those documentation blocks when confused.
### Sequence tools
Generate an a3m MSA using the ColabFold API.
```py
generate_MSA(seq, output_path, mode="all", max_retries=10)
```

<!-- ## Package Structure
This package is organized into the following sections:
```
neurosnap/
├── sequences
├── pyproject.toml
├── README.md
├── src/
│   └── example_package_YOUR_USERNAME_HERE/
│       ├── __init__.py
│       └── example.py
└── tests/
``` -->