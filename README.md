# Neurosnap Tools
Collection of useful bioinformatic functions and tools for various computational biology pipelines. Primarily designed for working with amino acid sequences and chemical structures.

This a package developed by Keaun Amani at [neurosnap.ai](https://neurosnap.ai/). You are welcome to use this code and contribute as you see fit. We are currently working on expanding this package as well to add support for more common functions.

# Installation
```sh
## ensure you have openbabel installed for the python bindings
# debian/ubuntu
sudo apt-get install openbabel
# arch
sudo pacman -S python-openbabel

## install the package
pip install -U --no-cache-dir --force-reinstall git+https://github.com/KeaunAmani/neurosnap.git
```

# Usage
Note that all functions have their own documentation within the code. We recommend checking those documentation blocks when confused.

## Sequence tools
This modules contains utility functions for working with sequence data such as protein sequences, alignments, etc.

### Proteins
To access the following functions import the following:
```py
from neurosnap.sequences.proteins import *
```

#### run_mmseqs2
##### Usage
```py
run_mmseqs2(seq, output, database="mmseqs2_uniref_env", use_filter=True, use_templates=False, pairing=None)
```
##### Description:
Generate an a3m MSA using the ColabFold API. Will writeall results to the output directory including templates,MSAs, and accompanying files.Code originally from https://github.com/sokrypton/ColabFold/.
##### Parameters:
- seq..........: Amino acid sequence for protein to generate an MSA of (str)
- output.......: Output directory path, will overwrite existing results (str)
- database.....: Choose the database to use, must be either "mmseqs2_uniref_env" or "mmseqs2_uniref" (str)
- use_filter...: Enables the diversity and msa filtering steps that ensures the MSA will not become enormously large (described in manuscript methods section of ColabFold paper) (bool)
- use_templates: Download templates as well using the mmseqs2 results (bool)
- pairing......: Can be set to either "greedy", "complete", or None for no pairing (str)

#### read_msa
##### Usage
```py
read_msa(input_fasta, size=float("inf"), allow_chars="", drop_chars="", remove_chars="*", uppercase=True)
```
##### Description:
Reads an MSA, a3m, or fasta file and returns an array of names and seqs.
##### Parameters:
- input_fasta.: Path to read input a3m file, fasta as a raw string, or a file-handle like object to read (str|io.TextIOBase)
- size........: Number of rows to read (int)
- allow_chars.: Sequences that contain characters not included within STANDARD_AAs+allow_chars will throw an exception (str)
- drop_chars..: Drop sequences that contain these characters e.g., "-X" (str)
- remove_chars: Removes these characters from sequences e.g., "*-X" (str)
- uppercase...: Converts all amino acid chars to uppercase when True (bool)
##### Returns:
- names: List of proteins names from the a3m file including gaps (list<str>)
- seqs.: List of proteins sequences from the a3m file including gaps (list<str>)

#### write_msa
##### Usage
```py
write_msa(output_path, names, seqs)
```
##### Description:
Writes an MSA, a3m, or fasta to a file.Makes no assumptions about the validity of names orsequences. Will throw an exception if len(names) != len(seqs)
##### Parameters:
- output_path: Path to output file to write, will overwrite existing files (str)
- names......: List of proteins names from the file (list<str>)
- seqs.......: List of proteins sequences from the file (list<str>)

#### pad_seqs
##### Usage
```py
pad_seqs(seqs, char="-", truncate=False)
```
##### Description:
Pads all sequences to the longest sequences lengthusing a character from the right side.
##### Parameters:
- seqs......: List of sequences to pad (list<str>)
- chars.....: The character to perform the padding with, default is "-" (str)
- truncate..: When set to True will truncate all sequences to the length of the first, set to integer to truncate sequence to that length (bool/int)
##### Returns:
- seqs_padded: The padded sequences (list<str>)

#### get_seqid
##### Usage
```py
get_seqid(seq1, seq2, align=False)
```
##### Description:
Calculate the pairwise sequence identity of two same length sequences or alignments.
##### Parameters:
- seq1: The 1st sequence / aligned sequence. (str)
- seq2: The 2nd sequence / aligned sequence. (str)
##### Returns:
- seq_id: The pairwise sequence identity. Will return None  (float)

## Structure tools
This modules contains utility functions for working with structural data such as protein structures, small molecules, ligands, and more.

### Chemicals
To access the following functions import the following:
```py
from neurosnap.structures.chemicals import *
```

#### `smiles_to_sdf(smiles, output_path)`

**Description:**
Converts a SMILES string to an sdf file.Will overwrite existing results.

**Parameters:**
- smiles.....: Smiles string to parse and convert (str)
- output_path: Path to output SDF file, should end with .sdf (str)


### Proteins
To access the following functions import the following:
```py
from neurosnap.structures.proteins import *
```

#### `read_chains(pdb_path)`

**Description:**
Reads the chains in PDB file and returns a set of their names/IDs.

**Parameters:**
- pdb_path: Input PDB file path (str)

**Returns:**
- chains: Chain names/IDs found within the PDB file (set<str>)

#### `calc_pdm(pdb_path, chain=None)`

**Description:**
Calculates distance matrix for a given input protein usingthe C-Alpha distances between residues.

**Parameters:**
- pdb_path: Path to PDB file you want to calculate the distance matrix of (str)
- chain...: The chain to use. By default will just use the longest chain (str)

**Returns:**
- dm: Distance matrix of the PDB file (np.array)

#### `pdb_to_aa(pdb_path)`

**Description:**
Reads a PDB file to and gets its corresponding amino acid sequence.Current implementation uses biotite and not biopython.

**Parameters:**
- pdb_path: Path to input PDB file to read (str)

**Returns:**
- seq: Corresponding amino acid sequence of PDB file (str)

#### `pdb_to_sdf(pdb_path, output_path, max_residues=50)`

**Description:**
Converts a protein/peptide in a PDB file to an SDF.PDB file can only include a single entry.Will overwrite existing results.Validates the SDF file with RDkit on completion

**Parameters:**
- pdb_path....: Path to input PDB file to convert (str)
- output_path.: Path to output SDF file, should end with .sdf (str)
- max_residues: Maximum number of residues, default=50 (int)


#### `align_pdbs(ref_pdb, sample_pdb)`
**Description:**
Aligns two pdb structures by their longest chain using the first pdb as the reference.Reference pdb is not modified while the sample pdb is

**Parameters:**
- ref_pdb...: Reference protein to align to (str)
- sample_pdb: Sample protein to be modified and aligned to the reference (str)



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