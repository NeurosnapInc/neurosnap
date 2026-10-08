Protein profiles and PSSM alignment
===================================

Protein profiles and alignment
------------------------------

Build a profile from an existing alignment without external tools::

   from neurosnap.sequence.pssm import pssm_from_msa, score_sequence

   profile = pssm_from_msa(["ACDE", "ACDF", "AC-E"])
   print(profile.to_dataframe())
   print(score_sequence(profile, "ACDE"))

Profiles use a fixed ``ACDEFGHIKLMNPQRSTVWY`` column order. Counts, smoothed
probabilities, background frequencies, sequence weights, original MSA columns,
and query residue indices are available on the returned object. Scores are
base-2 log odds in bits. The default weighting is position-based Henikoff
weighting and the default prior is one total pseudocount per position,
distributed over a uniform background. Explicit weights and backgrounds are
supported. The prior uses a simple background model; scores are not intended
to reproduce PSI-BLAST output.

Gaps do not contribute residue counts. By default, columns with gaps in the
first (query) row are removed. Set ``drop_query_gaps=False`` to retain them,
``query_index`` to select another row, or ``query_index=None`` to retain all
columns without assigning query coordinates. All-gap columns fall back to the
background. B/Z/J/X are distributed over their possible residues in background
proportions. U/O are rejected unless ``ambiguous="ignore"``; use
``ambiguous="error"`` to reject every nonstandard residue.

A3M insertion handling
----------------------

Lowercase A3M residues are insertions, not aligned match columns. Tell the
profile builder explicitly when input is A3M::

   profile = pssm_from_msa("alignment.a3m", a3m=True)

The reader can preserve these insertions for other consumers or strip them::

   from neurosnap.sequence.align import read_msa

   raw_rows = list(read_msa("alignment.a3m", allow_chars="-X", sequence_mode="a3m_preserve"))
   match_rows = list(read_msa("alignment.a3m", allow_chars="-X", sequence_mode="a3m_strip"))

``a3m_preserve`` retains lowercase residues and dots. ``a3m_strip`` removes
these insertions before uppercase conversion. Both A3M modes calculate coverage
and identity over match columns only. The default ``sequence_mode="uppercase"``
uppercases all residues as before. ``sequence_mode="preserve_case"`` retains
ordinary sequence casing without interpreting lowercase letters as insertions.
Do not use A3M modes for lowercase FASTA input: those letters can be ordinary
sequence residues. Preserved A3M rows may have different lengths; pass
``a3m=True`` when building their profile.

Migrating existing reader calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``read_msa`` now uses a single ``sequence_mode`` parameter. The previous
``uppercase`` and ``a3m_insertions`` parameters have been removed:

* ``uppercase=True`` (or no options): omit the option or use
  ``sequence_mode="uppercase"``.
* ``uppercase=False``: use ``sequence_mode="preserve_case"``.
* ``a3m_insertions="preserve"``: use ``sequence_mode="a3m_preserve"``.
* ``a3m_insertions="strip"``: use ``sequence_mode="a3m_strip"``.

If both previous options were set, choose the A3M mode. Calls without either
option are unchanged. The ``pssm_from_msa(..., a3m=True)`` interface is also
unchanged. Passing the removed keywords now raises ``TypeError``.

``msa_columns`` indexes original match columns, excluding A3M insertion
residues and dots. ``query_positions`` indexes the original ungapped query, including lowercase
A3M insertion residues in its numbering, with -1 for retained query gaps.
Insertion residues have no profile row, so these indices can skip positions. Every index is zero-based.

Generating profiles through alignment workflows
-----------------------------------------------

The existing MAFFT, phmmer/MAFFT, and MMseqs2 workflows can generate profiles::

   from neurosnap.sequence.pssm import generate_pssm

   profile = generate_pssm(
       ["ACDE", "ACDF"],
       method="mafft",
       alignment_kwargs={"threads": 2},
       pssm_kwargs={"pseudocount": 2.0},
   )

   profile = generate_pssm(
       "ACDEFGHIKLMNPQRSTVWY",
       method="mmseqs2",
       alignment_kwargs={"output": "msa_output", "use_templates": False},
   )

   profile = generate_pssm(
       "ACDEFGHIKLMNPQRSTVWY",
       method="phmmer_mafft",
       alignment_kwargs={"ref_db_path": "homologs.fasta", "mafft_threads": 2},
   )

The MMseqs2 and phmmer/MAFFT methods accept a single query. MMseqs2 output is
always processed as A3M. Workflow arguments and prerequisites remain those of
the existing functions: MAFFT/phmmer require their executables, the existing
phmmer workflow uses Biopython, and MMseqs2 requires network access. No new
package dependencies are added. PSSM calculation, scoring, and alignment use
NumPy and run locally. A single input sequence alone cannot provide evidence
about evolutionary residue preferences.

Sequence and profile alignment
------------------------------

Score every ungapped window, align a sequence, or align two profiles::

   from neurosnap.sequence.pssm import scan_sequence, align_sequence_to_pssm, align_pssms

   window_scores = scan_sequence(profile, "WWACDEWW")
   hit = align_sequence_to_pssm(profile, "WWACDEWW", mode="local")
   print(hit.score, hit.aligned_a, hit.aligned_b)
   full = align_sequence_to_pssm(profile, "ACDE", mode="global")
   other = pssm_from_msa(["ACDE", "ACDF"])
   comparison = align_pssms(profile, other, mode="global")

Local/global alignments use affine gaps: a gap of length k costs
``gap_open + (k - 1) * gap_extend`` bits. Defaults are 5 bits to open and 1 bit
to extend. Sequence B/Z/J/X scores marginalize over their possible residues;
X contributes zero bits. Alignment results include display strings, input
index pairs (``None`` for a gap), and half-open input spans. Profiles are
displayed using their consensus sequences. Map a profile index back through
``msa_columns`` or ``query_positions`` to recover source coordinates.

Profile-to-profile position scores are
``log2(sum(p_a * p_b / background))``. This symmetric overlap model requires
a shared background, or an explicit ``background`` argument if profiles were
built with different backgrounds. It is not an HMM alignment model and does
not return calibrated E-values. Unsmoothed profiles can contain impossible
matches; global alignment raises an error if no finite path exists. Both
alignment methods require quadratic time and memory in input lengths and are
intended for individual proteins, not large database searches.

API reference
-------------

See :mod:`neurosnap.sequence.pssm` for the complete function and result API.
