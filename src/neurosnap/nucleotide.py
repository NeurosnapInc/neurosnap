"""
Provides functions and classes related to processing nucleotide data.
"""

def get_reverse_complement(seq: str) -> str:
  """
  Generate the complementary strand of a DNA or RNA sequence in reverse order.

  Args:
      seq (str): A string representing the nucleotide sequence. 
                  Valid characters are 'A', 'T', 'C', 'G' for DNA and 'A', 'U', 'C', 'G' for RNA.

  Returns:
      str: A string representing the reverse complementary strand of the input sequence.

  Raises:
      KeyError: If the input sequence contains invalid nucleotide characters.
  """
  complement = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}
  return "".join([complement[base] for base in seq[::-1]])