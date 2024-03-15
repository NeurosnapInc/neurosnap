import re


# filename = "src/neurosnap/structures/proteins.py"
# filename = "src/neurosnap/structures/chemicals.py"
filename = "src/neurosnap/sequences/proteins.py"

with open(filename) as f:
  func = None
  for line in f:
    match = re.search(r"^def (.*)\:", line)
    if match:
      func = match.group(1)
      desc = ""
      comment_border1 = False
      comment_border2 = False
      desc_border1 = False
      desc_border2 = False
      params_border = False
      returns_border = False
      done = False
      params = []
      returns = []
    elif func is None:
      continue
    elif line.strip() == '"""' and comment_border1 == False:
      comment_border1 = True
    elif line.strip() == '"""' and comment_border1 == True:
      comment_border2 = True
      print(f"#### {func.split('(')[0]}")
      print(f"##### Usage")
      print(f"```py\n{func}\n```")
      print(f"##### Description:")
      print(f"{desc}")
      if params:
        print(f"##### Parameters:")
        for x in params:
          t,d = x.split(":", 1)
          print(f"- **{t.strip('.')}**: {d}")
      if returns:
        print(f"##### Returns:")
        for x in returns:
          t,d = x.split(":", 1)
          print(f"- **{t.strip('.')}**: {d}")
      print()
    elif comment_border1 == True and comment_border2 == False:
      if line.strip() == "-------------------------------------------------------":
        if desc_border1 == False:
          desc_border1 = True
        else:
          desc_border2 = True
      elif desc_border1 == True and desc_border2 == False:
        desc += line.strip()
      elif line.strip() == "Parameters:":
        params_border = True
      elif line.strip() == "Returns:":
        returns_border = True
      elif params_border == True and returns_border == False:
        params.append(line.strip())
      elif returns_border == True:
        returns.append(line.strip())
    



"""
-------------------------------------------------------
Reads the chains in PDB file and returns a set of their names/IDs.
-------------------------------------------------------
Parameters:
  pdb_path: Input PDB file path (str)
Returns:
  chains: Chain names/IDs found within the PDB file (set<str>)
"""