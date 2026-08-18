#!/usr/bin/env python3
"""QC and align one frozen GSE141983 H3/H3K4me3 paired-end run."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log: Path) -> None:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; log={log}"
        )


def find_row(path: Path, accession: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["run_accession"] == accession
        ]
    if len(matches) != 1:
        raise RuntimeError(f"manifest match count for {accession}: {len(matches)}")
    return matches[0]


def pipeline(commands: list[list[str]], stderr_paths: list[Path]) -> None:
    processes: list[subprocess.Popen[bytes]] = []
    stderr_handles = []
    previous = None
    try:
        for command, stderr_path in zip(commands, stderr_paths):
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_handle = stderr_path.open("wb")
            stderr_handles.append(stderr_handle)
            process = subprocess.Popen(
                command,
                stdin=previous.stdout if previous else None,
                stdout=subprocess.PIPE if len(processes) < len(commands) - 1 else None,
                stderr=stderr_handle,
            )
            if previous and previous.stdout:
                previous.stdout.close()
            processes.append(process)
            previous = process
        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"alignment pipeline failed: return_codes={return_codes}")
    finally:
        for handle in stderr_handles:
            handle.close()


def count_alignments(samtools: Path, bam: Path) -> int:
    completed = subprocess.run(
        [str(samtools), "view", "-c", "-F", "2304", str(bam)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return int(completed.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-accession", required=True)
    parser.add_argument(
        "--manifest",
        default="metadata/publication_v3/gse141983_raw_sensitivity/"
        "eligible_h3_h3k4me3_runs.tsv",
    )
    parser.add_argument("--bio-env-bin", default="envs/treefm-bio/bin")
    parser.add_argument("--chip-env-bin", default="envs/treefm-chip/bin")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    root = args.project_root.resolve()
    run = args.run_accession
    row = find_row(root / args.manifest, run)
    bio_env_bin = root / args.bio_env_bin
    chip_env_bin = root / args.chip_env_bin
    fastp = bio_env_bin / "fastp"
    bowtie2 = chip_env_bin / "bowtie2"
    samtools = bio_env_bin / "samtools"
    for binary in (fastp, bowtie2, samtools):
        if not binary.is_file():
            raise FileNotFoundError(binary)

    raw_dir = root / "data/raw/functional_v3/GSE141983_raw_sensitivity/fastq" / run
    fq1 = raw_dir / f"{run}_1.fastq.gz"
    fq2 = raw_dir / f"{run}_2.fastq.gz"
    prefetch_status = json.loads(
        (raw_dir / "prefetch_status.json").read_text(encoding="utf-8-sig")
    )
    if prefetch_status.get("status") != "complete":
        raise RuntimeError(f"FASTQ prefetch is not complete for {run}")

    output_root = (
        root
        / "results/biological_cases/prunus_publication_v3_gse141983_raw/alignment"
        / run
    )
    log_root = root / "logs/gse141983_raw_sensitivity/alignment" / run
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sample_qc.json"
    started = time.time()
    status: dict[str, Any] = {
        "status": "running",
        "run_accession": run,
        "stage": row["stage"],
        "replicate": int(row["replicate"]),
        "mark": row["mark"],
        "malus_accessed": False,
    }
    write_json(status_path, status)

    trimmed1 = output_root / f"{run}.trimmed_1.fastq.gz"
    trimmed2 = output_root / f"{run}.trimmed_2.fastq.gz"
    fastp_json = output_root / "fastp.json"
    fastp_html = output_root / "fastp.html"
    name_bam = output_root / f"{run}.name_sorted.bam"
    fixmate_bam = output_root / f"{run}.fixmate.bam"
    coordinate_bam = output_root / f"{run}.coordinate_sorted.bam"
    final_bam = output_root / f"{run}.mapq30.dedup.bam"
    index_prefix = (
        root
        / "data/raw/functional_v3/GSE141983_raw_sensitivity/reference/bowtie2/"
        "Prunus_persica_NCBIv2"
    )
    try:
        run_logged(
            [
                str(fastp),
                "--in1",
                str(fq1),
                "--in2",
                str(fq2),
                "--out1",
                str(trimmed1),
                "--out2",
                str(trimmed2),
                "--thread",
                str(min(args.threads, 16)),
                "--detect_adapter_for_pe",
                "--qualified_quality_phred",
                "20",
                "--unqualified_percent_limit",
                "40",
                "--n_base_limit",
                "5",
                "--length_required",
                "30",
                "--json",
                str(fastp_json),
                "--html",
                str(fastp_html),
            ],
            log_root / "fastp.log",
        )
        fastp_payload = json.loads(fastp_json.read_text(encoding="utf-8"))
        before = fastp_payload["summary"]["before_filtering"]
        after = fastp_payload["summary"]["after_filtering"]
        observed_reads = int(before["total_reads"])
        expected_ena_read_pairs = int(row["read_count"])
        expected_fastp_total_reads = 2 * expected_ena_read_pairs
        if observed_reads != expected_fastp_total_reads:
            raise RuntimeError(
                "paired-end FASTQ read count differs from twice the ENA pair count: "
                f"{observed_reads}/{expected_fastp_total_reads}"
            )

        pipeline(
            [
                [
                    str(bowtie2),
                    "-x",
                    str(index_prefix),
                    "-1",
                    str(trimmed1),
                    "-2",
                    str(trimmed2),
                    "--very-sensitive",
                    "--no-mixed",
                    "--no-discordant",
                    "-X",
                    "2000",
                    "-p",
                    str(args.threads),
                ],
                [
                    str(samtools),
                    "view",
                    "-@",
                    "2",
                    "-b",
                    "-q",
                    "30",
                    "-f",
                    "2",
                    "-",
                ],
                [
                    str(samtools),
                    "sort",
                    "-@",
                    "2",
                    "-n",
                    "-o",
                    str(name_bam),
                    "-",
                ],
            ],
            [
                log_root / "bowtie2.log",
                log_root / "samtools_view.log",
                log_root / "samtools_name_sort.log",
            ],
        )
        run_logged(
            [str(samtools), "fixmate", "-@", "2", "-m", str(name_bam), str(fixmate_bam)],
            log_root / "samtools_fixmate.log",
        )
        run_logged(
            [
                str(samtools),
                "sort",
                "-@",
                str(min(args.threads, 8)),
                "-o",
                str(coordinate_bam),
                str(fixmate_bam),
            ],
            log_root / "samtools_coordinate_sort.log",
        )
        run_logged(
            [
                str(samtools),
                "markdup",
                "-@",
                "2",
                "-r",
                "-s",
                str(coordinate_bam),
                str(final_bam),
            ],
            log_root / "samtools_markdup.log",
        )
        run_logged(
            [str(samtools), "index", "-@", "2", str(final_bam)],
            log_root / "samtools_index.log",
        )
        run_logged(
            [str(samtools), "flagstat", "-@", "2", str(final_bam)],
            output_root / "samtools_flagstat.txt",
        )
        retained_alignments = count_alignments(samtools, final_bam)
        if retained_alignments % 2 != 0:
            raise RuntimeError(f"odd retained paired alignment count: {retained_alignments}")
        retained_pairs = retained_alignments // 2
        filtered_pairs = int(after["total_reads"]) // 2
        mapping_fraction = retained_pairs / max(1, filtered_pairs)
        status.update(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "expected_ena_read_pairs": expected_ena_read_pairs,
                "expected_fastp_total_reads": expected_fastp_total_reads,
                "observed_fastq_total_reads": observed_reads,
                "post_fastp_total_reads": int(after["total_reads"]),
                "post_fastp_q30_rate": float(after["q30_rate"]),
                "post_fastp_pairs": filtered_pairs,
                "mapq30_nonduplicate_proper_pairs": retained_pairs,
                "mapq30_nonduplicate_pair_fraction": mapping_fraction,
                "bam": str(final_bam.relative_to(root)),
                "bam_bytes": final_bam.stat().st_size,
            }
        )
        for path in (name_bam, fixmate_bam, coordinate_bam):
            path.unlink(missing_ok=True)
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "elapsed_seconds": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    write_json(status_path, status)
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
