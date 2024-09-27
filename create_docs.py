import re

filenames = [
  "src/neurosnap/chemicals.py",
  # "src/neurosnap/lDDT.py",
  # "src/neurosnap/log.py",
  "src/neurosnap/protein.py",
  "src/neurosnap/conformers.py",
  "src/neurosnap/structures/proteins.py",
  "src/neurosnap/msa.py",
]

print() #  required

for filename in filenames:
  module_doc_start = False
  module_doc_end = False
  with open(filename) as f:
    # get import path
    filename_clean = filename[4:-3].split("/")
    import_statement = f"from {".".join(filename_clean[:-1])} import {filename_clean[-1]}"
    # print module title
    print(f"### {filename_clean[-1].upper()} MODULE")
    
    func = None
    for line in f:
      # get module doc
      if module_doc_start == False and line.strip() == '"""':
        module_doc_start = True
      elif module_doc_start == True and module_doc_end == False and line.strip() == '"""':
        module_doc_end = True
      elif module_doc_start == True and module_doc_end == False:
        print(line)
      
      # function doc
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
        print(f"```py\n{import_statement}\n{filename_clean[-1]}.{func}\n```")
        print(f"##### Description:")
        print(desc.strip())
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
          desc += line.strip() + " "
        elif line.strip() == "Parameters:":
          params_border = True
        elif line.strip() == "Returns:":
          returns_border = True
        elif params_border == True and returns_border == False:
          params.append(line.strip())
        elif returns_border == True:
          returns.append(line.strip())