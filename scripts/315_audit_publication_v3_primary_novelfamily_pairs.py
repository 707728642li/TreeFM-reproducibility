#!/usr/bin/env python3
"""Audit pair integrity and leakage in the frozen primary NovelFamily panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


SLUGS = ("hevea_brasiliensis", "prunus_persica", "pyrus_pyrifolia")
TASKS = ("tis", "tts", "splice_donor", "splice_acceptor")
COLUMNS = (
    "pair_id",
    "slug",
    "task",
    "label",
    "family_transfer_class",
    "pair_has_exact_task_train_sequence",
    "pair_has_near_task_train_sequence_ge_0_90",
    "pair_has_near_task_train_sequence_ge_0_95",
    "maximum_task_train_identity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_count(data: pd.DataFrame, column: str) -> int:
    return int(data.loc[data[column].fillna(False), "pair_id"].nunique())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    benchmark_root = (
        root / "data/processed/technical_benchmarks_publication_v3_26"
    )

    records: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    integrity_errors: list[str] = []
    for slug in SLUGS:
        source = benchmark_root / f"{slug}.parquet"
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
        input_hashes[str(source.relative_to(root))] = sha256(source)
        data = pd.read_parquet(source, columns=list(COLUMNS))
        if set(data["slug"].unique()) != {slug}:
            integrity_errors.append(f"{slug}: source contains another slug")
        for task in TASKS:
            scope = data.loc[
                data["task"].eq(task)
                & data["family_transfer_class"].eq("logo_novel_family")
            ].copy()
            grouped = scope.groupby("pair_id", sort=False)
            size_failures = int(grouped.size().ne(2).sum())
            label_failures = int(grouped["label"].nunique().ne(2).sum())
            positive_rows = int(scope["label"].eq(1).sum())
            negative_rows = int(scope["label"].eq(0).sum())
            pairs = int(scope["pair_id"].nunique())
            if size_failures or label_failures:
                integrity_errors.append(
                    f"{slug}/{task}: pair_size={size_failures}, pair_label={label_failures}"
                )
            if positive_rows != pairs or negative_rows != pairs:
                integrity_errors.append(
                    f"{slug}/{task}: positives={positive_rows}, negatives={negative_rows}, pairs={pairs}"
                )
            records.append(
                {
                    "slug": slug,
                    "task": task,
                    "rows": int(len(scope)),
                    "pairs": pairs,
                    "positive_rows": positive_rows,
                    "negative_rows": negative_rows,
                    "pair_size_failures": size_failures,
                    "pair_label_failures": label_failures,
                    "exact_pairs": pair_count(
                        scope, "pair_has_exact_task_train_sequence"
                    ),
                    "near_0_90_pairs": pair_count(
                        scope, "pair_has_near_task_train_sequence_ge_0_90"
                    ),
                    "near_0_95_pairs": pair_count(
                        scope, "pair_has_near_task_train_sequence_ge_0_95"
                    ),
                    "maximum_task_train_identity": float(
                        scope["maximum_task_train_identity"].max()
                    ),
                }
            )

    scopes = pd.DataFrame(records)
    expected_scope_count = len(SLUGS) * len(TASKS)
    observed = {
        "scopes": int(len(scopes)),
        "rows": int(scopes["rows"].sum()),
        "pairs": int(scopes["pairs"].sum()),
        "minimum_pairs_per_scope": int(scopes["pairs"].min()),
        "maximum_pairs_per_scope": int(scopes["pairs"].max()),
        "exact_pairs": int(scopes["exact_pairs"].sum()),
        "near_0_90_pairs": int(scopes["near_0_90_pairs"].sum()),
        "near_0_95_pairs": int(scopes["near_0_95_pairs"].sum()),
        "maximum_task_train_identity": float(
            scopes["maximum_task_train_identity"].max()
        ),
        "pair_size_failures": int(scopes["pair_size_failures"].sum()),
        "pair_label_failures": int(scopes["pair_label_failures"].sum()),
    }
    expected = {
        "scopes": expected_scope_count,
        "pairs": 14791,
        "minimum_pairs_per_scope": 636,
        "maximum_pairs_per_scope": 2589,
        "exact_pairs": 0,
        "near_0_90_pairs": 1,
        "near_0_95_pairs": 1,
        "maximum_task_train_identity": 0.984,
        "pair_size_failures": 0,
        "pair_label_failures": 0,
    }
    for key, value in expected.items():
        matches = (
            math.isclose(observed[key], value, rel_tol=0.0, abs_tol=1e-6)
            if key == "maximum_task_train_identity"
            else observed[key] == value
        )
        if not matches:
            integrity_errors.append(
                f"frozen expectation mismatch for {key}: {observed[key]} versus {value}"
            )

    output_table = (
        root / "metadata/publication_v3_primary_novelfamily_pair_summary.tsv"
    )
    output_table.parent.mkdir(parents=True, exist_ok=True)
    scopes.to_csv(output_table, sep="\t", index=False)

    payload = {
        "status": "pass" if not integrity_errors else "fail",
        "scope": "frozen_seed23_primary_novelfamily_pair_integrity",
        "species": list(SLUGS),
        "tasks": list(TASKS),
        "family_transfer_class": "logo_novel_family",
        "observed": observed,
        "expected": expected,
        "integrity_errors": integrity_errors,
        "input_sha256": input_hashes,
        "artifacts": {
            "scope_table": str(output_table.relative_to(root)),
        },
        "malus_accessed": False,
    }
    output_json = (
        root / "results/metrics/publication_v3_primary_novelfamily_pair_audit.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if integrity_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
