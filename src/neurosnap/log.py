import logging


### CLASSES ###
class c:
  """Terminal colors class"""

  _ = "\033[0m"  # reset terminal
  p = "\033[38;5;204m"  # pink
  o = "\033[38;5;208m"  # orange
  b = "\033[38;5;295m"  # blue
  c = "\033[38;5;299m"  # cyan
  g = "\033[38;5;47m"  # green
  grey = "\033[90m"  # grey
  r = "\033[38;5;1m"  # red
  br = "\x1b[31;1m"  # boldred
  y = "\033[38;5;226m"  # yellow


class CustomLogger(logging.Formatter):
  """Custom logger with specialized formatting.

  NOTE:
    ``[+] logging.DEBUG``: Used for all general info

    ``[*] logging.INFO``: Used for more important key info that isn't negative

    ``[-] logging.WARNING``: Used for non-severe info that is negative

    ``[!] logging.ERROR``: Used for errors that require attention but are super concerning

    ``[!] logging.CRITICAL``: Used for very severe errors that require immediate attention and are concerning
  """

  log_format_detailed = f"{c.grey}%(asctime)s{c._} %(message)s {c.p}(%(filename)s:%(lineno)d){c._}"
  log_format_basic = "%(message)s"

  FORMATS = {
    logging.DEBUG: f"{c.g}[+]{c._} {log_format_basic}",
    logging.INFO: f"{c.b}[*]{c._} {log_format_basic}",
    logging.WARNING: f"{c.y}[-]{c._} {log_format_detailed}",
    logging.ERROR: f"{c.r}[!]{c._} {log_format_detailed}",
    logging.CRITICAL: f"{c.br}[!]{c._} {log_format_detailed}",
  }
  """:meta private:"""
  # marking this as private so sphinx doesn't try to document it

  def format(self, record):
    log_fmt = self.FORMATS.get(record.levelno)
    formatter = logging.Formatter(log_fmt)
    return formatter.format(record)


logger = logging.getLogger("neurosnap")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomLogger())
logger.addHandler(ch)
