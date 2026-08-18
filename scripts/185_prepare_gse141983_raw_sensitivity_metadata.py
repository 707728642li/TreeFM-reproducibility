#!/usr/bin/env python3
"""Create the outcome-free GSE141983 raw ChIP-seq sample manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ELIGIBLE_TITLE = re.compile(
    r"^Flower buds, ChIP-seq_date(?P<stage>[123])_(?P<replicate>[23])_"
    r"(?P<mark>H3|K4)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_soft(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current:
                    records.append(current)
                current = {"gsm": line.split(" = ", 1)[1]}
                continue
            if current is None:
                continue
            if line.startswith("!Sample_title = "):
                current["title"] = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_library_layout = "):
                current["geo_library_layout"] = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_library_strategy = "):
                current["geo_library_strategy"] = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_characteristics_ch1 = time-point: "):
                current["collection_date"] = line.rsplit(": ", 1)[1]
            elif line.startswith("!Sample_characteristics_ch1 = chip antibody: "):
                current["antibody"] = line.split("chip antibody: ", 1)[1]
            elif line.startswith("!Sample_relation = SRA: "):
                match = re.search(r"(SRX\d+)", line)
                if match:
                    current["experiment_accession"] = match.group(1)
        if current:
            records.append(current)
    return records


def read_runinfo(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--soft",
        default="metadata/publication_v3/gse141983_raw_sensitivity/"
        "GSE141983_family.soft.gz",
    )
    parser.add_argument(
        "--runinfo",
        default="metadata/publication_v3/gse141983_raw_sensitivity/"
        "SRP237509_ENA_runinfo.tsv",
    )
    parser.add_argument(
        "--output",
        default="metadata/publication_v3/gse141983_raw_sensitivity/"
        "eligible_h3_h3k4me3_runs.tsv",
    )
    parser.add_argument(
        "--audit-output",
        default="results/metrics/"
        "publication_v3_gse141983_raw_metadata_audit.json",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    soft_path = root / args.soft
    runinfo_path = root / args.runinfo
    soft = parse_soft(soft_path)
    runinfo = read_runinfo(runinfo_path)
    by_experiment = {row["experiment_accession"]: row for row in runinfo}
    violations: list[str] = []
    if len(soft) != 27:
        violations.append(f"unexpected_geo_sample_count:{len(soft)}")
    if len(runinfo) != 27:
        violations.append(f"unexpected_ena_run_count:{len(runinfo)}")
    if len(by_experiment) != len(runinfo):
        violations.append("nonunique_experiment_to_run_mapping")

    eligible: list[dict[str, Any]] = []
    for sample in soft:
        match = ELIGIBLE_TITLE.match(sample.get("title", ""))
        if not match:
            continue
        experiment = sample.get("experiment_accession", "")
        row = by_experiment.get(experiment)
        if row is None:
            violations.append(f"missing_runinfo:{experiment}")
            continue
        urls = [f"https://{item}" for item in row["fastq_ftp"].split(";")]
        md5s = row["fastq_md5"].split(";")
        byte_values = [int(item) for item in row["fastq_bytes"].split(";")]
        if len(urls) != 2 or len(md5s) != 2 or len(byte_values) != 2:
            violations.append(f"not_two_fastq_files:{row['run_accession']}")
            continue
        groups = match.groupdict()
        mark = "H3K4me3" if groups["mark"] == "K4" else "H3"
        eligible.append(
            {
                "run_accession": row["run_accession"],
                "experiment_accession": experiment,
                "geo_accession": sample["gsm"],
                "title": sample["title"],
                "stage": f"date{groups['stage']}",
                "collection_date": sample.get("collection_date", ""),
                "replicate": int(groups["replicate"]),
                "mark": mark,
                "antibody": sample.get("antibody", ""),
                "library_layout": row["library_layout"],
                "library_strategy": row["library_strategy"],
                "instrument_model": row["instrument_model"],
                "read_count": int(row["read_count"]),
                "base_count": int(row["base_count"]),
                "fastq_1_url": urls[0],
                "fastq_2_url": urls[1],
                "fastq_1_md5": md5s[0],
                "fastq_2_md5": md5s[1],
                "fastq_1_bytes": byte_values[0],
                "fastq_2_bytes": byte_values[1],
            }
        )

    eligible.sort(key=lambda row: (row["stage"], row["replicate"], row["mark"]))
    if len(eligible) != 12:
        violations.append(f"unexpected_eligible_count:{len(eligible)}")
    cells = {(row["stage"], row["replicate"], row["mark"]) for row in eligible}
    expected_cells = {
        (f"date{stage}", replicate, mark)
        for stage in (1, 2, 3)
        for replicate in (2, 3)
        for mark in ("H3", "H3K4me3")
    }
    if cells != expected_cells:
        violations.append("eligible_stage_replicate_mark_cells_differ")
    if any(row["library_layout"] != "PAIRED" for row in eligible):
        violations.append("nonpaired_eligible_library")
    if any(row["library_strategy"] != "ChIP-Seq" for row in eligible):
        violations.append("non_chipseq_eligible_library")

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(eligible[0]) if eligible else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(eligible)

    total_fastq_bytes = sum(
        row["fastq_1_bytes"] + row["fastq_2_bytes"] for row in eligible
    )
    audit = {
        "status": "pass" if not violations else "fail",
        "scope": "outcome_free_gse141983_raw_chipseq_metadata_gate",
        "scientific_decision_authority": False,
        "malus_accessed": False,
        "violations": violations,
        "source_sha256": {
            args.soft: sha256_file(soft_path),
            args.runinfo: sha256_file(runinfo_path),
        },
        "geo_samples": len(soft),
        "ena_runs": len(runinfo),
        "eligible_h3_h3k4me3_runs": len(eligible),
        "eligible_cells": len(cells),
        "total_fastq_files": len(eligible) * 2,
        "total_fastq_bytes": total_fastq_bytes,
        "total_fastq_gib": total_fastq_bytes / (1024**3),
        "run_manifest": args.output,
        "run_manifest_sha256": sha256_file(output),
    }
    audit_path = root / args.audit_output
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
