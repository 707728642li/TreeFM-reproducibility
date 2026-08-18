#!/usr/bin/env python3
"""Run offline protein-mode BUSCO for every source-complete matched-control candidate."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SUMMARY_PATTERN = re.compile(
    r"C:([0-9.]+)%\[S:([0-9.]+)%,D:([0-9.]+)%\],"
    r"F:([0-9.]+)%,M:([0-9.]+)%,n:(\d+)"
)


def locate_summary(out_root: Path, slug: str) -> Path | None:
    candidates = sorted((out_root / slug).rglob("short_summary*.json"))
    return candidates[0] if candidates else None


def remove_incomplete_output(out_root: Path, slug: str) -> None:
    """Remove only an incomplete direct child so interrupted BUSCO jobs can resume."""
    target = out_root / slug
    if not target.exists():
        return
    allowed_root = out_root.resolve(strict=True)
    resolved = target.resolve(strict=True)
    if resolved.parent != allowed_root or target.is_symlink() or not target.is_dir():
        raise RuntimeError(f"refusing to remove unsafe BUSCO output path: {target}")
    shutil.rmtree(target)


def prepare_protein(root: Path, slug: str) -> Path:
    source = (
        root
        / "data/raw/publication_v3_phylogc_candidates"
        / slug
        / "protein.fa.gz"
    )
    destination = (
        root
        / "data/interim/publication_v3_phylogc_candidates"
        / slug
        / "primary.protein.fa"
    )
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    with gzip.open(source, "rb") as input_handle, partial.open("wb") as output:
        shutil.copyfileobj(input_handle, output, length=8 * 1024 * 1024)
    if partial.stat().st_size == 0:
        raise RuntimeError(f"empty decompressed proteome: {slug}")
    partial.replace(destination)
    return destination


def run_one(
    root: Path,
    slug: str,
    busco: Path,
    lineage: Path,
    out_root: Path,
    log_root: Path,
    cpus: int,
) -> dict[str, object]:
    protein = prepare_protein(root, slug)
    existing = locate_summary(out_root, slug)
    if existing is None:
        remove_incomplete_output(out_root, slug)
        command = [
            str(busco),
            "--in",
            str(protein),
            "--mode",
            "proteins",
            "--lineage_dataset",
            str(lineage),
            "--out",
            slug,
            "--out_path",
            str(out_root),
            "--cpu",
            str(cpus),
            "--offline",
        ]
        environment = os.environ.copy()
        environment["PATH"] = f"{busco.parent}:{environment['PATH']}"
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        log_root.mkdir(parents=True, exist_ok=True)
        (log_root / f"{slug}.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(f"BUSCO failed for {slug}")
        existing = locate_summary(out_root, slug)
    if existing is None:
        raise RuntimeError(f"BUSCO summary absent for {slug}")
    payload = json.loads(existing.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    one_line = str(
        results.get("one_line_summary")
        or payload.get("one_line_summary_raw")
        or payload.get("one_line_summary")
        or ""
    )
    match = SUMMARY_PATTERN.search(one_line)
    if not match:
        raise RuntimeError(f"unparsed BUSCO summary for {slug}: {one_line}")
    complete, single, duplicated, fragmented, missing, markers = match.groups()
    return {
        "slug": slug,
        "complete_pct": float(complete),
        "single_copy_pct": float(single),
        "duplicated_pct": float(duplicated),
        "fragmented_pct": float(fragmented),
        "missing_pct": float(missing),
        "n_markers": int(markers),
        "busco_gate_pass": float(complete) >= 90.0,
        "protein": str(protein.relative_to(root)),
        "summary_json": str(existing.relative_to(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--cpus-per-job", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    discovery = pd.read_csv(
        root / "metadata/publication_v3_phylogc_source_discovery.tsv",
        sep="\t",
        dtype=str,
    ).fillna("")
    eligible = discovery.loc[
        discovery["source_status"].eq("complete_same_release")
    ].copy()
    busco = root / "envs/treefm-match/bin/busco"
    lineage = root / "data/raw/busco_downloads/lineages/embryophyta_odb12"
    out_root = root / "results/busco_publication_v3_phylogc_candidates"
    log_root = root / "logs/busco_publication_v3_phylogc_candidates"
    if not busco.is_file() or not lineage.is_dir():
        raise FileNotFoundError("BUSCO executable or offline lineage is absent")

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_one,
                root,
                row.slug,
                busco,
                lineage,
                out_root,
                log_root,
                args.cpus_per_job,
            ): row.slug
            for row in eligible.itertuples(index=False)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"{record['slug']}\tBUSCO={record['complete_pct']:.1f}\t"
                f"pass={record['busco_gate_pass']}",
                flush=True,
            )
    summary = eligible[
        ["scientific_name", "slug", "order", "family", "life_form"]
    ].merge(pd.DataFrame(records), on="slug", how="left", validate="one_to_one")
    output_path = root / "metadata/publication_v3_phylogc_busco.tsv"
    summary.to_csv(output_path, sep="\t", index=False)
    audit = {
        "status": "pass" if summary["complete_pct"].notna().all() else "fail",
        "threshold_complete_pct": 90.0,
        "source_complete_candidates": len(summary),
        "busco_pass_candidates": int(summary["busco_gate_pass"].sum()),
        "busco_fail_candidates": summary.loc[
            ~summary["busco_gate_pass"].fillna(False), "slug"
        ].tolist(),
        "output": str(output_path.relative_to(root)),
    }
    audit_path = root / "metadata/publication_v3_phylogc_busco.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
