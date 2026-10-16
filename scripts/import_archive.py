#!/usr/bin/env python3
"""Import the verified UTP v0.3 source archive (no downloaded corpora/models)."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_NAME = "Ukrainian-Token-Project-v0.3.zip"
PREFIX = "ukrainian-token-project-v3/"
# Keep bootstrap GitHub settings and paired translation benchmark dependencies.
SKIP = {".gitignore", "requirements.txt", ".github/workflows/ci.yml"}


def import_archive(archive_path: Path, destination: Path) -> list[str]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Upload {ARCHIVE_NAME} to the repository root first.")
    written: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        manifest_name = PREFIX + "RELEASE_MANIFEST.json"
        if manifest_name not in names:
            raise ValueError("Missing v0.3 manifest. This is not the expected archive.")
        manifest = json.loads(archive.read(manifest_name))
        if manifest.get("release") != "v0.3":
            raise ValueError("Archive does not identify itself as v0.3")
        expected = manifest["files"]
        # Verify *all* manifest entries, even assets we intentionally skip.
        for relative, digest in expected.items():
            safe = PurePosixPath(relative)
            if (safe.is_absolute() or ".." in safe.parts or
                    safe.as_posix() != relative or not relative):
                raise ValueError(f"Unsafe archive entry: {relative}")
            archive_entry = PREFIX + relative
            if archive_entry not in names:
                raise ValueError(f"Missing archive entry: {relative}")
            data = archive.read(archive_entry)
            actual = hashlib.sha256(data).hexdigest()
            if actual != digest:
                raise ValueError(f"Checksum mismatch: {relative}")
        for relative in sorted(expected):
            # Never commit trained model binaries, demo runs, third-party corpora,
            # or overwrite the CI workflow and dependencies already on GitHub.
            if (relative.startswith("runs/") or relative.startswith("data/") or
                    relative in SKIP):
                continue
            if relative.startswith(".github/"):
                continue
            out_path = destination.joinpath(*PurePosixPath(relative).parts)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(archive.read(PREFIX + relative))
            written.append(relative)
        (destination / "RELEASE_MANIFEST.json").write_bytes(
            archive.read(manifest_name)
        )
        written.append("RELEASE_MANIFEST.json")
    return written


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    imported = import_archive(root / ARCHIVE_NAME, root)
    print(f"Verified source import: {len(imported)} files")
    for path in imported:
        print(path)
