"""API-related constants."""

# try to get the version of the user agent
version = "unknown"
try:
  from importlib.metadata import version as _get_version

  version = _get_version("neurosnap")
except Exception:  # noqa: BLE001
  pass

# User agent to use throughout the application
USER_AGENT = f"Neurosnap-OSS-Tools/v-{version}"

__all__ = ["USER_AGENT"]
