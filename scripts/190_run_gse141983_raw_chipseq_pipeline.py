#!/usr/bin/env python3
"""Frozen controller for the post-hoc GSE141983 raw ChIP-seq sensitivity."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], stdout: Path, stderr: Path) -> None:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        completed = subprocess.run(command, stdout=out, stderr=err, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; "
            f"stdout={stdout}; stderr={stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-alignment-workers", type=int, default=12)
    parser.add_argument("--threads-per-alignment", type=int, default=7)
    args = parser.parse_args()

    root = args.project_root.resolve()
    freeze_path = root / "config/publication_v3_gse141983_raw_sensitivity_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    freeze = read_json(freeze_path)
    if freeze.get("status") != "frozen" or freeze.get("malus_accessed") is not False:
        raise RuntimeError("raw sensitivity freeze is not valid")
    amendment_path = (
        root
        / "config/publication_v3_gse141983_raw_readcount_implementation_amendment.json"
    )
    if not amendment_path.is_file():
        raise FileNotFoundError(amendment_path)
    amendment = read_json(amendment_path)
    if (
        amendment.get("status") != "frozen_implementation_correction"
        or amendment.get("malus_accessed") is not False
        or amendment.get("alignment_started_before_amendment") is not False
        or amendment.get("label_data_read_before_amendment") is not False
    ):
        raise RuntimeError("paired-end read-count implementation amendment is invalid")
    if sha256_file(freeze_path) != amendment.get("parent_freeze_sha256"):
        raise RuntimeError("implementation amendment parent-freeze hash differs")
    replacements = amendment.get("replacement_artifact_sha256", {})
    for relative, expected in freeze["artifact_sha256"].items():
        path = root / relative
        active_expected = replacements.get(relative, expected)
        if relative in replacements and amendment.get("parent_artifact_sha256", {}).get(
            relative
        ) != expected:
            raise RuntimeError(f"amendment parent artifact differs: {relative}")
        if not path.is_file() or sha256_file(path) != active_expected:
            raise RuntimeError(f"freeze artifact mismatch: {relative}")
    for relative, expected in amendment.get("supporting_artifact_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"amendment supporting artifact mismatch: {relative}")

    output_root = root / "results/biological_cases/prunus_publication_v3_gse141983_raw"
    log_root = root / "logs/gse141983_raw_sensitivity/controller"
    controller_path = output_root / "pipeline_controller.json"
    state: dict[str, Any] = {
        "status": "waiting_for_fastq",
        "scope": "posthoc_gse141983_raw_h3_h3k4me3_pipeline",
        "malus_accessed": False,
        "max_alignment_workers": args.max_alignment_workers,
        "threads_per_alignment": args.threads_per_alignment,
        "completed_alignment_runs": [],
    }
    write_json(controller_path, state)

    fastq_controller_path = output_root / "fastq_controller.json"
    while True:
        if fastq_controller_path.is_file():
            fastq = read_json(fastq_controller_path)
            state["fastq_status"] = fastq.get("status")
            state["fastq_completed_runs"] = fastq.get("completed_runs", 0)
            state["fastq_failed_runs"] = fastq.get("failed_runs", 0)
            write_json(controller_path, state)
            if fastq.get("status") == "failed":
                state["status"] = "failed_fastq_transport"
                write_json(controller_path, state)
                return 2
            if fastq.get("status") == "complete":
                break
        time.sleep(args.poll_seconds)

    with (
        root
        / "metadata/publication_v3/gse141983_raw_sensitivity/"
        "eligible_h3_h3k4me3_runs.tsv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    if len(manifest) != 12:
        raise RuntimeError("frozen sample manifest does not contain 12 rows")

    state["status"] = "building_bowtie2_index"
    write_json(controller_path, state)
    chip_bin = root / "envs/treefm-chip/bin"
    bio_bin = root / "envs/treefm-bio/bin"
    reference_root = root / "data/raw/functional_v3/GSE141983_raw_sensitivity/reference"
    genome = reference_root / "Prunus_persica_NCBIv2.fa"
    index_root = reference_root / "bowtie2"
    index_root.mkdir(parents=True, exist_ok=True)
    index_prefix = index_root / "Prunus_persica_NCBIv2"
    if not all(
        (index_root / f"Prunus_persica_NCBIv2.{suffix}.bt2").is_file()
        for suffix in ("1", "2", "3", "4", "rev.1", "rev.2")
    ):
        run_logged(
            [
                str(chip_bin / "bowtie2-build"),
                "--threads",
                "32",
                str(genome),
                str(index_prefix),
            ],
            log_root / "bowtie2_build.out.log",
            log_root / "bowtie2_build.err.log",
        )

    state["status"] = "aligning"
    write_json(controller_path, state)

    def align(row: dict[str, str]) -> tuple[str, int]:
        run = row["run_accession"]
        sample_qc = output_root / "alignment" / run / "sample_qc.json"
        if sample_qc.is_file() and read_json(sample_qc).get("status") == "complete":
            return run, 0
        command = [
            str(bio_bin / "python"),
            str(root / "scripts/188_process_gse141983_raw_chipseq_sample.py"),
            "--project-root",
            str(root),
            "--run-accession",
            run,
            "--threads",
            str(args.threads_per_alignment),
        ]
        out = log_root / f"alignment_{run}.out.log"
        err = log_root / f"alignment_{run}.err.log"
        with out.open("wb") as stdout, err.open("wb") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        return run, completed.returncode

    failed: list[str] = []
    completed_runs: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_alignment_workers
    ) as pool:
        future_by_run = {pool.submit(align, row): row["run_accession"] for row in manifest}
        for future in concurrent.futures.as_completed(future_by_run):
            run, return_code = future.result()
            if return_code == 0:
                completed_runs.append(run)
            else:
                failed.append(run)
            state["completed_alignment_runs"] = sorted(completed_runs)
            state["failed_alignment_runs"] = sorted(failed)
            write_json(controller_path, state)
    if failed:
        state["status"] = "failed_alignment"
        write_json(controller_path, state)
        return 3

    state["status"] = "counting_tss_fragments"
    write_json(controller_path, state)
    counts_root = output_root / "counts"
    counts_root.mkdir(parents=True, exist_ok=True)
    counts_path = counts_root / "tss_counts.txt"
    saf = reference_root / "Prunus_persica_NCBIv2_TSS_minus1000_plus999.saf"
    bams = [
        output_root / "alignment" / row["run_accession"] / f"{row['run_accession']}.mapq30.dedup.bam"
        for row in manifest
    ]
    run_logged(
        [
            str(bio_bin / "featureCounts"),
            "-T",
            "64",
            "-p",
            "--countReadPairs",
            "-B",
            "-C",
            "-F",
            "SAF",
            "-a",
            str(saf),
            "-o",
            str(counts_path),
            *map(str, bams),
        ],
        log_root / "featurecounts.out.log",
        log_root / "featurecounts.err.log",
    )

    state["status"] = "technical_gate_and_analysis"
    write_json(controller_path, state)
    run_logged(
        [
            str(bio_bin / "python"),
            str(root / "scripts/189_analyze_gse141983_raw_h3k4me3.py"),
            "--project-root",
            str(root),
        ],
        log_root / "analysis.out.log",
        log_root / "analysis.err.log",
    )
    run_logged(
        [
            str(bio_bin / "python"),
            str(root / "scripts/191_verify_gse141983_raw_h3k4me3.py"),
            "--project-root",
            str(root),
        ],
        log_root / "verification.out.log",
        log_root / "verification.err.log",
    )
    summary = read_json(output_root / "summary.json")
    verification = read_json(
        root / "results/metrics/publication_v3_gse141983_raw_h3k4me3_verification.json"
    )
    if verification.get("status") != "pass":
        raise RuntimeError("independent GSE141983 raw-sensitivity verification failed")
    state["status"] = "complete"
    state["analysis_status"] = summary.get("status")
    state["verification_status"] = verification.get("status")
    state["malus_accessed"] = False
    write_json(controller_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
