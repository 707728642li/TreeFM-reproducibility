#!/usr/bin/env python3
"""Run corrected Tier-A extraction and an unbiased full-Pfam hmmscan."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROFILE_DIR = Path("data/reference/pfam_full_v3")
PROFILE_GZ = PROFILE_DIR / "Pfam-A.hmm.gz"
PROFILE_DAT_GZ = PROFILE_DIR / "Pfam-A.hmm.dat.gz"
PROFILE_HMM = PROFILE_DIR / "Pfam-A.hmm"
PROFILE_DAT = PROFILE_DIR / "Pfam-A.hmm.dat"
FASTA = Path(
    "data/processed/publication_v3_tier_a_annotation/"
    "tier_a_candidate_and_anchor_proteins.fa"
)
DOMAIN_DIR = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/domains"
)
DOMTBL = DOMAIN_DIR / "Pfam-A.domtblout"
TEXT_OUTPUT = DOMAIN_DIR / "Pfam-A.hmmscan.txt"
SUMMARY = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/domain_summary.json"
)
ANNOTATION_FREEZE = Path(
    "config/publication_v3_tier_a_postselection_annotation_freeze.json"
)
CANDIDATE_FREEZE = Path(
    "config/publication_v3_crossgenus_candidate_catalog_freeze.json"
)
CONTRACT = Path(
    "docs/publication_v3_tier_a_postselection_annotation_contract_v2.md"
)
OUT = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "pipeline_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def gunzip_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with gzip.open(source, "rb") as compressed, partial.open("wb") as output:
        shutil.copyfileobj(compressed, output, length=8 * 1024 * 1024)
    os.replace(partial, destination)


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def verify_freeze(root: Path, hmmscan: Path, hmmpress: Path) -> dict[str, object]:
    freeze_path = root / ANNOTATION_FREEZE
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("freeze_version") != "2.0":
        raise RuntimeError("Tier-A annotation freeze v2 is not active")
    pinned = freeze["pinned_sha256"]
    observed_paths = {
        str(CONTRACT): root / CONTRACT,
        str(CANDIDATE_FREEZE): root / CANDIDATE_FREEZE,
        str(PROFILE_GZ): root / PROFILE_GZ,
        str(PROFILE_DAT_GZ): root / PROFILE_DAT_GZ,
        "scripts/247_prepare_publication_v3_tier_a_annotation.py": (
            root / "scripts/247_prepare_publication_v3_tier_a_annotation.py"
        ),
        "scripts/248_summarize_publication_v3_tier_a_domains.py": (
            root / "scripts/248_summarize_publication_v3_tier_a_domains.py"
        ),
        "scripts/249_run_publication_v3_tier_a_annotation.py": (
            root / "scripts/249_run_publication_v3_tier_a_annotation.py"
        ),
        str(hmmscan): hmmscan,
        str(hmmpress): hmmpress,
    }
    failures = []
    for label, path in observed_paths.items():
        expected = pinned.get(label)
        observed = sha256(path) if path.is_file() else None
        if observed != expected:
            failures.append(
                {"path": label, "expected": expected, "observed": observed}
            )
    if failures:
        raise RuntimeError(
            "annotation freeze hash failures: " + json.dumps(failures)
        )
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--hmmscan", type=Path, required=True)
    parser.add_argument("--hmmpress", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=64)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.cpu < 1:
        raise ValueError("--cpu must be positive")
    hmmscan = args.hmmscan.resolve()
    hmmpress = args.hmmpress.resolve()
    for path in [hmmscan, hmmpress, root / PROFILE_GZ, root / PROFILE_DAT_GZ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze = verify_freeze(root, hmmscan, hmmpress)

    run(
        [
            sys.executable,
            "scripts/247_prepare_publication_v3_tier_a_annotation.py",
            "--project-root",
            str(root),
        ],
        root,
    )
    gunzip_atomic(root / PROFILE_GZ, root / PROFILE_HMM)
    gunzip_atomic(root / PROFILE_DAT_GZ, root / PROFILE_DAT)

    profile_hmm = root / PROFILE_HMM
    run([str(hmmpress), "-f", str(profile_hmm)], root)
    domain_dir = root / DOMAIN_DIR
    domain_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(hmmscan),
        "--cpu",
        str(args.cpu),
        "--cut_ga",
        "--noali",
        "--domtblout",
        str(root / DOMTBL),
        "-o",
        str(root / TEXT_OUTPUT),
        str(profile_hmm),
        str(root / FASTA),
    ]
    run(command, root)
    run(
        [
            sys.executable,
            "scripts/248_summarize_publication_v3_tier_a_domains.py",
            "--project-root",
            str(root),
        ],
        root,
    )
    summary = json.loads((root / SUMMARY).read_text(encoding="utf-8"))
    if summary.get("status") != "pass":
        raise RuntimeError("full-Pfam summary is not passing")
    pressed = {
        str(Path(str(PROFILE_HMM) + suffix)): sha256(
            Path(str(root / PROFILE_HMM) + suffix)
        )
        for suffix in (".h3f", ".h3i", ".h3m", ".h3p")
    }
    manifest = {
        "status": "pass",
        "scope": "retrospective_corrected_tier_a_full_pfam_pipeline",
        "selection_authority": False,
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_freeze_version": freeze["freeze_version"],
        "annotation_freeze_sha256": sha256(root / ANNOTATION_FREEZE),
        "hmmscan": str(hmmscan),
        "hmmscan_sha256": sha256(hmmscan),
        "hmmpress": str(hmmpress),
        "hmmpress_sha256": sha256(hmmpress),
        "cpu": args.cpu,
        "command": command,
        "pfam_gzip_sha256": sha256(root / PROFILE_GZ),
        "pfam_dat_gzip_sha256": sha256(root / PROFILE_DAT_GZ),
        "pfam_hmm_sha256": sha256(profile_hmm),
        "pfam_dat_sha256": sha256(root / PROFILE_DAT),
        "pressed_database_sha256": pressed,
        "scan_fasta_sha256": sha256(root / FASTA),
        "raw_output_sha256": {
            str(DOMTBL): sha256(root / DOMTBL),
            str(TEXT_OUTPUT): sha256(root / TEXT_OUTPUT),
        },
        "domain_summary_sha256": sha256(root / SUMMARY),
        "domain_summary_status": summary["status"],
        "candidate_orthogroups": summary["candidate_orthogroups"],
        "extracted_proteins": summary["extracted_proteins"],
        "violations": [],
    }
    atomic_json(root / OUT, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
