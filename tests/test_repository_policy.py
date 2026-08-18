from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def repository_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    ]


def test_no_publication_artifacts() -> None:
    forbidden = {".docx", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".tif", ".tiff", ".pptx", ".zip"}
    offenders = [path.relative_to(ROOT).as_posix() for path in repository_files() if path.suffix.lower() in forbidden]
    assert offenders == []


def test_no_forbidden_top_level_directories() -> None:
    forbidden = {"data", "results", "figures", "images", "manuscript", "submission", "review", "reports"}
    offenders = sorted({path.relative_to(ROOT).parts[0] for path in repository_files() if path.relative_to(ROOT).parts[0].lower() in forbidden})
    assert offenders == []


def test_no_private_machine_paths_or_gmail_addresses() -> None:
    pattern = re.compile(
        r"[A-Za-z]:\\(?:Users|01_CC_dir|00_Codex_dir)\\|/data/codexli/|/nas_data/NFS/|"
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b|"
        r"\b[A-Z0-9._%+-]+@gmail\.com\b",
        re.I,
    )
    offenders: list[str] = []
    for path in repository_files():
        if path.suffix.lower() not in {".py", ".r", ".sh", ".ps1", ".json", ".md", ".yml", ".yaml", ".txt", ".tsv"}:
            continue
        if path.relative_to(ROOT).as_posix() in {"tools/validate_repository.py", "tests/test_repository_policy.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_all_python_files_parse() -> None:
    failures: list[str] = []
    for path in repository_files():
        if path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    assert failures == []
