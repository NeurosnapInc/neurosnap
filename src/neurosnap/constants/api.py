"""API-related constants."""

version = "unknown"
try:
  from importlib.metadata import version as _get_version

  version = _get_version("neurosnap")
except Exception:  # noqa: BLE001
  pass

USER_AGENT = f"Neurosnap-OSS-Tools/v-{version}"

__all__ = ["USER_AGENT"]
