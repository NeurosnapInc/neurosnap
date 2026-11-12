"""Publish a Sphinx build as an immutable platform SDK-reference release.

Usage: python scripts/publish_docs.py docs/build/html ../neurosnap-backend/templates/sdk-reference
Never clears the destination. A new pointer takes effect on backend restart.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


def publish(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and destination must be separate directory trees")
    manifest = json.loads((source / "neurosnap-reference.json").read_text(encoding="utf-8"))
    version = manifest.get("version", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+\-]*", version):
        raise ValueError("Missing or invalid package version")
    pages = manifest.get("pages", [])
    names = set()
    for page in pages:
        name = page.get("name", "")
        if not name or Path(name).name != name or not name.endswith(".html") or name in names:
            raise ValueError("Invalid or duplicate SDK page: " + name)
        if not page.get("html") or not page.get("title") or not (source / name).is_file():
            raise ValueError("Incomplete SDK page: " + name)
        names.add(name)
    if "index.html" not in names or not (source / "_static").is_dir() or not (source / "objects.inv").is_file():
        raise ValueError("Incomplete Sphinx HTML build")
    digest = hashlib.sha256()
    for entry in sorted(source.rglob("*")):
        if entry.is_symlink():
            raise ValueError("Build contains a symlink: " + str(entry))
        if entry.is_file():
            digest.update(entry.relative_to(source).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(entry.read_bytes()).digest())
    release = version + "-" + digest.hexdigest()[:16]
    releases = target / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    destination = releases / release
    if not destination.exists():
        stage = Path(tempfile.mkdtemp(prefix=".publishing-", dir=releases))
        try:
            shutil.copytree(source, stage, dirs_exist_ok=True)
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    pointer = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target, prefix=".current-", delete=False, encoding="utf-8") as stream:
            pointer = Path(stream.name)
            json.dump({"release": release}, stream)
            stream.write("\n")
        os.replace(pointer, target / "current.json")
    finally:
        if pointer is not None and pointer.exists():
            pointer.unlink()
    print("Published SDK " + version + " to " + str(destination))
    print("Restart the backend to load this reference and its matching assets.")
    return release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    publish(args.source, args.target)
