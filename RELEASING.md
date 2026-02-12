# Releasing to PyPI

Use this checklist to publish a new `neurosnap` release to PyPI.

## Prerequisites

- Ensure the package version in `pyproject.toml` has been bumped.
- Ensure you can authenticate with PyPI (API token recommended).
- Ensure release tooling is installed:

```sh
python -m pip install -U build twine pkginfo
```

## 1. Run tests

```sh
pytest
```

## 2. Clean old build artifacts

```sh
rm -rf dist/ build/ src/neurosnap.egg-info
```

## 3. Build sdist and wheel

```sh
python -m build
```

## 4. Validate built distributions

```sh
twine check dist/*
```

## 5. (Recommended) Upload to TestPyPI first

```sh
twine upload --repository testpypi dist/*
```

## 6. Upload to PyPI

```sh
# This prompts for credentials unless configured via .pypirc or environment variables.
twine upload dist/*
```

## Notes

- PyPI token auth uses username `__token__` and password `pypi-...`.
- If upload fails because a file already exists, confirm version bump or use:

```sh
twine upload --skip-existing dist/*
```
