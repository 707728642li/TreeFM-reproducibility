#!/usr/bin/env python3
"""Validate the methods-and-code-only repository policy."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKABLE_BYTES = 5 * 1024 * 1024

FORBIDDEN_SUFFIXES = {
    ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".tif", ".tiff", ".pptx", ".zip"
}
FORBIDDEN_TOP_LEVEL = {
    "data", "results", "output", "outputs", "figures", "images", "models", "checkpoints",
    "embeddings", "manuscript", "submission", "supplement", "review", "reports", "tmp",
}
TEXT_SUFFIXES = {".py", ".R", ".r", ".sh", ".ps1", ".json", ".md", ".yml", ".yaml", ".txt", ".tsv"}
FORBIDDEN_PATTERNS = {
    "local Windows project root": re.compile(r"[A-Za-z]:\\(?:Users|01_CC_dir|00_Codex_dir)\\", re.I),
    "private server project root": re.compile(r"/data/codexli/", re.I),
    "institutional genome mirror": re.compile(r"/nas_data/NFS/", re.I),
    "private IPv4 host": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "Gmail address": re.compile(r"\b[A-Z0-9._%+-]+@gmail\.com\b", re.I),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]+\b"),
    "generic secret assignment": re.compile(r"(?i)\b(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*['\"][^'\"]+['\"]"),
}


def tracked_candidates() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    )


def main() -> int:
    errors: list[str] = []
    files = tracked_candidates()

    for path in files:
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0].lower() in FORBIDDEN_TOP_LEVEL:
            errors.append(f"forbidden top-level directory: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden publication artifact: {rel}")
        if path.stat().st_size > MAX_TRACKABLE_BYTES:
            errors.append(f"file exceeds 5 MiB policy: {rel}")

        if path.suffix in TEXT_SUFFIXES or path.suffix.lower() in {s.lower() for s in TEXT_SUFFIXES}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"non-UTF-8 text file: {rel}")
                continue
            # The policy implementation necessarily contains the forbidden
            # patterns as literals; scan all other repository text.
            if rel.as_posix() not in {"tools/validate_repository.py", "tests/test_repository_policy.py"}:
                for label, pattern in FORBIDDEN_PATTERNS.items():
                    if pattern.search(text):
                        errors.append(f"{label} found in {rel}")

        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python syntax/encoding error in {rel}: {exc}")

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Repository validation passed: {len(files)} files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
