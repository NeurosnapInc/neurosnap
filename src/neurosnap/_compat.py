"""Private package-wide compatibility helpers.

This module contains small internal shims that smooth over Python-version
differences without spreading conditional logic throughout the codebase. It is
intended for internal use across ``neurosnap`` rather than as part of the
public API surface.
"""

import sys
from dataclasses import dataclass


def compat_dataclass(**kwargs):
  """Return a version-compatible wrapper around ``dataclasses.dataclass``.

  This helper exists to make dataclass declarations portable across Python
  versions that differ in support for the ``slots`` keyword argument.
  Python 3.10 and newer support ``@dataclass(slots=True)``, which generates
  ``__slots__`` for the class and can reduce per-instance memory usage while
  preventing arbitrary new attributes from being attached. Python 3.9 and
  earlier do not support the ``slots`` argument and will raise a
  ``TypeError`` if it is passed directly to ``dataclass``.

  To avoid that version-specific failure, this function removes the
  ``slots`` keyword when the interpreter is older than Python 3.10, then
  forwards the remaining keyword arguments unchanged to
  ``dataclasses.dataclass``. On Python 3.10 and newer, all keyword
  arguments, including ``slots``, are passed through as provided.

  This allows code such as ``@compat_dataclass(frozen=True, slots=True)``
  to behave correctly on both older and newer Python versions without
  requiring duplicate class definitions or conditional decorator logic at
  each call site.

  Args:
    **kwargs: Arbitrary keyword arguments accepted by
      ``dataclasses.dataclass``, such as ``frozen``, ``order``, ``eq``,
      ``repr``, and optionally ``slots``.

  Returns:
    Callable: The class decorator returned by ``dataclasses.dataclass``,
    with the ``slots`` argument omitted automatically on Python versions
    earlier than 3.10.

  Example:
    ```python
    @compat_dataclass(frozen=True, slots=True)
    class CCDEntry:
      code: str
      name: str
    ```

    On Python 3.10+, this behaves like
    ``@dataclass(frozen=True, slots=True)``. On Python 3.9 and earlier,
    it behaves like ``@dataclass(frozen=True)``.
  """
  if sys.version_info < (3, 10):
    kwargs.pop("slots", None)
  return dataclass(**kwargs)
