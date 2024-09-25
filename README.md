# Neurosnap Tools

[![Neurosnap Header](https://raw.githubusercontent.com/NeurosnapInc/neurosnap/refs/heads/main/assets/header.webp)](https://neurosnap.ai/)

Collection of useful bioinformatic functions and tools for various computational biology pipelines. Primarily designed for working with amino acid sequences and chemical structures.

This a package developed by Keaun Amani at [neurosnap.ai](https://neurosnap.ai/). You are welcome to use this code and contribute as you see fit. We are currently working on expanding this package as well to add support for more common functions.

# Contributions
We welcome contributions to this package. If you have a feature that you want to code or have added, submit a pull request or an issue.

# Installation
```sh
## ensure you have openbabel installed for the python bindings
# debian/ubuntu
sudo apt-get install openbabel
# arch
sudo pacman -S python-openbabel

## install the package
pip install -U --no-cache-dir git+https://github.com/NeurosnapInc/neurosnap.git
```

# Usage
Note that all functions have their own documentation within the code. We recommend checking those documentation blocks when confused.

### CHEMICALS MODULE
Provides functions and classes related to processing chemical data.

#### smiles_to_sdf
```py
from neurosnap import chemicals
chemicals.smiles_to_sdf(smiles, output_path)
```

**Description:**

Converts a SMILES string to an sdf file.Will overwrite existing results.NOTE: This function does the bare minimum in terms ofgenerating the SDF molecule. The conformers.py moduleshould be used in most cases.

**Parameters:**

- **smiles**:  Smiles string to parse and convert (str)
- **output_path**:  Path to output SDF file, should end with .sdf (str)

### PROTEIN MODULE
Provides functions and classes related to processing protein data as well as

a feature rich wrapper around protein structures using BioPython.

TODO:

- Easy way to get backbone coordinates

#### getAA
```py
from neurosnap import protein
protein.getAA(query)
```

**Description:**

Efficiently get any amino acid using either their 1 letter code,3 letter abbreviation, or full name. See AAs_FULL_TABLEfor a list of all supported amino acids and codes.

**Parameters:**

- **query**:  Amino acid code, abbreviation, or name (str)

**Returns:**

- **code**:  Amino acid 1 letter abbreviation / code (str)
- **abr**:  Amino acid 3 letter abbreviation / code (str)
- **name**:  Amino acid full name (str)

### CONFORMERS MODULE
Provides functions and classes related to processing and generating conformers.

#### find_LCS
```py
from neurosnap import conformers
conformers.find_LCS(mol)
```

**Description:**

Find the largest common substructure (LCS) between aset of conformers and aligns all conformers to the LCS.Raises an exception if no LCS detected.

**Parameters:**

- **mol**:  Input RDkit molecule object, must already have conformers present (rdkit.Chem.rdchem.Mol)

**Returns:**

- **mol_aligned**:  Resultant molecule object with all conformers aligned to the LCS (rdkit.Chem.rdchem.Mol)

#### minimize
```py
from neurosnap import conformers
conformers.minimize(mol, method="MMFF94", e_delta=5)
```

**Description:**

Minimize conformer energy (kcal/mol) using RDkitand filter out conformers below a certain threshold.

**Parameters:**

- **mol**:  RDkit mol object containing the conformers you want to minimize. (rdkit.Chem.rdchem.Mol)
- **method**:  Can be either UFF, MMFF94, or MMFF94s (str)
- **e_delta**:  Filters out conformers that are above a certain energy threshold in kcal/mol. Formula used is >= min_conformer_energy + e_delta (float)

**Returns:**

- **mol_filtered**:  The pairwise sequence identity. Will return None (float)
- **energies**:  Dictionary where keys are conformer IDs and values are calculated energies in kcal/mol (dict<int:float>)

#### generate
```py
from neurosnap import conformers
conformers.generate(input_mol, output_name="unique_conformers", write_multi=False, num_confs=1000, min_method="MMFF94")
```

**Description:**

Generate conformers for an input molecule.Performs the following actions in order:1. Generate conformers using ETKDG method2. Minimize energy of all conformers and remove those below a dynamic threshold3. Align & create RMSD matrix of all conformers4. Clusters using Butina method to remove structurally redundant conformers5. Return most energetically favorable conformers in each cluster

**Parameters:**

- **input_mol**:  Input molecule can be a path to a molecule file, a SMILES string, or an instance of rdkit.Chem.rdchem.Mol (any)
- **output_name**:  Output to write SDF files of passing conformers (str)
- **write_multi**:  If True will write all unique conformers to a single SDF file, if False will write all unique conformers in separate SDF files in output_name (bool)
- **num_confs**:  Number of conformers to generate (int)
- **min_method**:  Method for minimization, can be either UFF, MMFF94, MMFF94s, or None for no minimization (str)

**Returns:**

- **df_out**:  Output pandas dataframe with all conformer statistics (pandas.core.frame.DataFrame)

### PROTEINS MODULE
Provides functions and classes related to processing protein structure data.

TODO: Refactor like the rest for consistency or integrate into protein.py.

#### read_chains
```py
from neurosnap.structures import proteins
proteins.read_chains(pdb_path)
```

**Description:**

Reads the chains in PDB file and returns a list of their names/IDs.Only does so for the first model within the pdb file.

**Parameters:**

- **pdb_path**:  Input PDB file path (str)

**Returns:**

- **chains**:  Chain names/IDs found within the PDB file (list<str>)

#### read_pdb
```py
from neurosnap.structures import proteins
proteins.read_pdb(pdb_path)
```

**Description:**

Reads a protein and returns the IDs

**Parameters:**

- **pdb_path**:  Input PDB file path (str)

**Returns:**

- **protein**:  Dictionary where keys are chain IDs and values are lists of residue IDs (dict<str:[str]>)

#### calc_pdm
```py
from neurosnap.structures import proteins
proteins.calc_pdm(pdb_path, chain=None)
```

**Description:**

Calculates distance matrix for a given input protein usingthe C-Alpha distances between residues.

**Parameters:**

- **pdb_path**:  Path to PDB file you want to calculate the distance matrix of (str)
- **chain**:  The chain to use. By default will just use the longest chain (str)

**Returns:**

- **dm**:  Distance matrix of the PDB file (np.array)

#### pdb_to_aa
```py
from neurosnap.structures import proteins
proteins.pdb_to_aa(pdb_path)
```

**Description:**

Reads a PDB file to and gets its corresponding amino acid sequence.Current implementation uses biopython and ignores all non-standard AA molecules.All chains on all models are concatenated together.

**Parameters:**

- **pdb_path**:  Path to input PDB file to read (str)

**Returns:**

- **seq**:  Corresponding amino acid sequence of PDB file (str)

#### pdb_to_sdf
```py
from neurosnap.structures import proteins
proteins.pdb_to_sdf(pdb_path, output_path, max_residues=50)
```

**Description:**

Converts a protein/peptide in a PDB file to an SDF.PDB file can only include a single entry.Will overwrite existing results.Validates the SDF file with RDkit on completion

**Parameters:**

- **pdb_path**:  Path to input PDB file to convert (str)
- **output_path**:  Path to output SDF file, should end with .sdf (str)
- **max_residues**:  Maximum number of residues, default=50 (int)

#### align_pdbs
```py
from neurosnap.structures import proteins
proteins.align_pdbs(ref_pdb, sample_pdb)
```

**Description:**

Aligns two pdb structures by their longest chain using the first pdb as the reference.Reference pdb is not modified or overwritten while the sample pdb is overwritten.

**Parameters:**

- **ref_pdb**:  Filepath for reference protein to align to (str)
- **sample_pdb**:  Filepath for sample protein to be modified and aligned to the reference (str)

#### calc_lDDT
```py
from neurosnap.structures import proteins
proteins.calc_lDDT(ref_pdb, sample_pdb)
```

**Description:**

Calculates the lDDT (Local Distance Difference Test) between two proteins.

**Parameters:**

- **ref_pdb**:  Filepath for reference protein (str)
- **sample_pdb**:  Filepath for sample protein (str)

**Returns:**

- **lDDT**:  The lDDT score of the two proteins which ranges between 0-1 (float)

### MSA MODULE
Provides functions and classes related to processing protein sequence data.

#### read_msa
```py
from neurosnap import msa
msa.read_msa(input_fasta, size=float("inf"), allow_chars="", drop_chars="", remove_chars="*", uppercase=True)
```

**Description:**

Reads an MSA, a3m, or fasta file and returns an array of names and seqs.

**Parameters:**

- **input_fasta**:  Path to read input a3m file, fasta as a raw string, or a file-handle like object to read (str|io.TextIOBase)
- **size**:  Number of rows to read (int)
- **allow_chars**:  Sequences that contain characters not included within STANDARD_AAs+allow_chars will throw an exception (str)
- **drop_chars**:  Drop sequences that contain these characters e.g., "-X" (str)
- **remove_chars**:  Removes these characters from sequences e.g., "*-X" (str)
- **uppercase**:  Converts all amino acid chars to uppercase when True (bool)

**Returns:**

- **names**:  List of proteins names from the a3m file including gaps (list<str>)
- **seqs**:  List of proteins sequences from the a3m file including gaps (list<str>)

#### write_msa
```py
from neurosnap import msa
msa.write_msa(output_path, names, seqs)
```

**Description:**

Writes an MSA, a3m, or fasta to a file.Makes no assumptions about the validity of names orsequences. Will throw an exception if len(names) != len(seqs)

**Parameters:**

- **output_path**:  Path to output file to write, will overwrite existing files (str)
- **names**:  List of proteins names from the file (list<str>)
- **seqs**:  List of proteins sequences from the file (list<str>)

#### pad_seqs
```py
from neurosnap import msa
msa.pad_seqs(seqs, char="-", truncate=False)
```

**Description:**

Pads all sequences to the longest sequences lengthusing a character from the right side.

**Parameters:**

- **seqs**:  List of sequences to pad (list<str>)
- **chars**:  The character to perform the padding with, default is "-" (str)
- **truncate**:  When set to True will truncate all sequences to the length of the first, set to integer to truncate sequence to that length (bool/int)

**Returns:**

- **seqs_padded**:  The padded sequences (list<str>)

#### get_seqid
```py
from neurosnap import msa
msa.get_seqid(seq1, seq2)
```

**Description:**

Calculate the pairwise sequence identity of two same length sequences or alignments.

**Parameters:**

- **seq1**:  The 1st sequence / aligned sequence. (str)
- **seq2**:  The 2nd sequence / aligned sequence. (str)

**Returns:**

- **seq_id**:  The pairwise sequence identity. Will return None  (float)

#### run_phmmer
```py
from neurosnap import msa
msa.run_phmmer(query, database, evalue=10, cpu=2)
```

**Description:**

Run phmmer using a query sequence against a database andreturn all the sequences that are considered as hits.Shameless stolen and adapted from https://github.com/seanrjohnson/protein_gibbs_sampler/blob/a5de349d5f6a474407fc0f19cecf39a0447a20a6/src/pgen/utils.py#L263

**Parameters:**

- **query**:  Amino acid sequence of the protein you want to find hits for (str).
- **database**:  Path to reference database of sequences you want to search for hits and create and alignment with, must be a protein fasta file (str)
- **evalue**:  The threshold E value for the phmmer hit to be reported (float)
- **cpu**:  The number of CPU cores to be used to run phmmer (str)

**Returns:**

- **hits**:  List of hits ranked by how good the hits are (list<str>)

#### align_mafft
```py
from neurosnap import msa
msa.align_mafft(seqs, ep=0.0, op=1.53)
```

**Description:**

Generates an alignment using mafft.

**Parameters:**

- **seqs**:  Can be fasta file path, list of sequences, or dictionary where values are AA sequences and keys are their corresponding names/IDs.
- **ep**:  ep value for mafft, default is 0.00 (float)
- **op**:  op value for mafft, default is 1.53 (float)

**Returns:**

- **out_names**:  List of aligned protein names (list<str>)
- **out_seqs**:  List of corresponding protein sequences (list<str>)

#### run_phmmer_mafft
```py
from neurosnap import msa
msa.run_phmmer_mafft(query, ref_db_path, size=float("inf"), in_name="input_sequence")
```

**Description:**

Generate MSA using phmmer and mafft from reference sequences.

**Parameters:**

- **query**:  Amino acid sequence of the protein you want to create an MSA of (str).
- **ref_db_path**:  Path to reference database of sequences you want to search for hits and create and alignment with (str)
- **size**:  Top n number of sequences to keep (int)
- **in_name**:  Optional name for input sequence to put in the output (str)

**Returns:**

- **out_names**:  List of aligned protein names (list<str>)
- **out_seqs**:  List of corresponding protein sequences (list<str>)

#### run_mmseqs2
```py
from neurosnap import msa
msa.run_mmseqs2(seq, output, database="mmseqs2_uniref_env", use_filter=True, use_templates=False, pairing=None)
```

**Description:**

Generate an a3m MSA using the ColabFold API. Will writeall results to the output directory including templates,MSAs, and accompanying files.Code originally adapted from: https://github.com/sokrypton/ColabFold/

**Parameters:**

- **seq**:  Amino acid sequence for protein to generate an MSA of (str)
- **output**:  Output directory path, will overwrite existing results (str)
- **database**:  Choose the database to use, must be either "mmseqs2_uniref_env" or "mmseqs2_uniref" (str)
- **use_filter**:  Enables the diversity and msa filtering steps that ensures the MSA will not become enormously large (described in manuscript methods section of ColabFold paper) (bool)
- **use_templates**:  Download templates as well using the mmseqs2 results (bool)
- **pairing**:  Can be set to either "greedy", "complete", or None for no pairing (str)

