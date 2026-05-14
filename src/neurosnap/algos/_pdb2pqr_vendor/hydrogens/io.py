"""Shim module so vendored hydrogen code can reuse the parent IO helpers."""

from ..io import DuplicateFilter

__all__ = ["DuplicateFilter"]
