"""Audit source release contents without opening any file outside this repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".local-data", "data", "artifacts", "outputs", "output", "logs", "build", "dist",
           ".cache", ".pytest_cache", ".ruff_cache", "__pycache__", "htmlcov"}
ASSET_SUFFIXES = {".bvh", ".npz", ".npy", ".csv", ".stl", ".obj", ".dae", ".pt",
                  ".mp4", ".gif", ".png", ".jpg", ".jpeg", ".zip", ".tar", ".gz"}
SECRET_NAMES = {"id_rsa", "id_ed25519", "id_ed25519_github", "authorized_keys", "known_hosts", ".env"}
ORIGINAL_MEDIA = {"docs/assets/original_q1_demo.mp4", "docs/assets/original_q1_demo.gif"}
PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    ("GitHub access token", re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("machine-specific home path", re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")),
]


def candidates(tracked: bool):
    if tracked:
        output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        return [ROOT / name.decode() for name in output.split(b"\0") if name]
    return [p for p in ROOT.rglob("*") if p.is_file() and not any(
        part in IGNORED or part.startswith(".venv") or part.endswith(".egg-info")
        for part in p.relative_to(ROOT).parts)]


def audit(paths):
    failures = []
    count = 0
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            failures.append((str(relative), "symlink is not allowed in source release"))
            continue
        if relative.as_posix() in ORIGINAL_MEDIA:
            try:
                provenance = json.loads((ROOT/"docs/assets/demo.json").read_text())
                valid = (provenance["origin"] == "original_synthetic" and provenance["uses_lafan"] is False
                         and provenance["generator"] == "tools/make_demo.py"
                         and provenance["generator_sha256"] == hashlib.sha256((ROOT/"tools/make_demo.py").read_bytes()).hexdigest()
                         and provenance["files"][path.name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                         and 0 < path.stat().st_size <= 12*1024*1024)
                if not valid:
                    raise ValueError("Original media provenance mismatch")
            except (OSError, ValueError, KeyError):
                failures.append((str(relative), "original-demo provenance, hash, or size check failed"))
            continue
        if path.suffix.lower() in ASSET_SUFFIXES or path.name in SECRET_NAMES or path.suffix in {".pem", ".key"}:
            failures.append((str(relative), "asset, generated data, or credential filename"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            failures.append((str(relative), "unreadable or non-text source content"))
            continue
        count += 1
        for label, pattern in PATTERNS:
            if pattern.search(content):
                # Never print the matched material, even if it is a credential.
                failures.append((str(relative), label))
    return count, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="Audit exactly git ls-files before publication")
    args = parser.parse_args()
    count, failures = audit(candidates(args.tracked))
    if count == 0:
        failures.append((".", "empty source release"))
    for path, reason in failures:
        print(f"FAIL {path}: {reason}", file=sys.stderr)
    print(f"Source release audit: {count} text files; {len(failures)} problems")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
