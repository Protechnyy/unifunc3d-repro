#!/usr/bin/env python3
"""Regenerate the release's content checksums after an intentional edit."""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_lib.common import ROOT, sha256, write_json

if __name__ == "__main__":
    # Honor .gitignore instead of traversing local weights, datasets and runs.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    paths = sorted({ROOT / name for name in names if name and (ROOT / name).is_file()
                    and Path(name).name not in {"checksums.json", ".DS_Store"}
                    and Path(name).suffix not in {".log", ".pyc"}})
    write_json(ROOT / "checksums.json", {str(p.relative_to(ROOT)): sha256(p) for p in paths}, pretty=True)
    print(f"Recorded {len(paths)} SHA-256 hashes")
