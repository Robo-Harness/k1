"""Fail-closed source-tree hygiene checks; never prints matching secret values."""

import argparse
import hashlib
import json
from pathlib import Path
import re

PATTERNS = {
    "credential": re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|ghp_[A-Za-z0-9]{20,})"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "embedded_auth": re.compile(r"https?://[^\s/]+:[^\s/]+@"),
    "private_absolute_path": re.compile(r"(?<!\w)/(?:mnt|home|Users|root)/[^\s\"']+"),
    "nonlocal_ipv4": re.compile(r"\b(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
}
ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".example"}
ALLOWED_NAMES = {"LICENSE", ".gitignore", "MANIFEST.in"}
# Only reviewed documentation images are allowed, not arbitrary binary artifacts.
# Replacing an image requires a new content review and checksum update.
APPROVED_ASSETS = {
    "assets/logo.png": "e40e0e5a55a742e84807323b25be375f91deb350b7573cdf3b09583257f405ed",
    "assets/perception-tools.png": "f14005565f00888b923ba200806078cb8bf4c8961857dfbda348cfb958272430",
}
GENERATED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


def audit(root):
    findings, files = [], 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append({"file": str(relative), "rule": "symlink"})
            continue
        if not path.is_file():
            continue
        files += 1
        if any(p in GENERATED_PARTS or p.endswith(".egg-info") for p in relative.parts):
            findings.append({"file": str(relative), "rule": "generated_or_repository_metadata"})
            continue
        approved_hash = APPROVED_ASSETS.get(relative.as_posix())
        if approved_hash is not None:
            if hashlib.sha256(path.read_bytes()).hexdigest() != approved_hash:
                findings.append({"file": str(relative), "rule": "unreviewed_asset_content"})
            continue
        if path.name not in ALLOWED_NAMES and path.suffix not in ALLOWED_SUFFIXES:
            findings.append({"file": str(relative), "rule": "unexpected_file_type"})
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            findings.append({"file": str(relative), "rule": "local_environment_file"})
        try:
            text = path.read_text()
        except UnicodeError:
            findings.append({"file": str(relative), "rule": "binary_file"})
            continue
        for line, value in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(value):
                    findings.append({"file": str(relative), "line": line, "rule": name})
    return {
        "files_checked": files,
        "passed": not findings,
        "findings": findings,
        "scope": "Heuristic scan, not a proof of absence of all sensitive information",
    }


def main():
    parser = argparse.ArgumentParser(description="Audit a source tree before sharing")
    parser.add_argument("path", type=Path)
    result = audit(parser.parse_args().path.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
