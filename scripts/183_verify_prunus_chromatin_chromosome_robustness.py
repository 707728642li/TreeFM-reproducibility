#!/usr/bin/env python3
"""Verify post-hoc Prunus chromosome robustness artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


def null_metrics(null: np.ndarray, observed: float) -> tuple[float, float]:
    return float(null.mean()), float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    result_root = root / "results/biological_cases/prunus_publication_v3_chromatin_replication"
    summary_path = result_root / "posthoc_chromosome_robustness_summary.json"
    conditioned_path = result_root / "posthoc_chromosome_conditioned_null.tsv.gz"
    jackknife_path = result_root / "posthoc_leave_one_chromosome_nulls.tsv.gz"
    gene_path = result_root / "gse190586_binary_gene_calls.tsv.gz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    genes = pd.read_csv(gene_path, sep="\t")
    conditioned_null = pd.read_csv(conditioned_path, sep="\t")["null_fraction"].to_numpy(float)
    jackknife_null = pd.read_csv(jackknife_path, sep="\t")
    violations: list[str] = []

    if len(conditioned_null) != 10_000:
        violations.append("conditioned_null_not_10000")
    expected_chromosomes = [f"Pp{index:02d}" for index in range(1, 9)]
    observed_chromosomes = sorted(jackknife_null["excluded_chromosome"].unique())
    if observed_chromosomes != expected_chromosomes:
        violations.append("jackknife_chromosome_set_mismatch")
    if not all(
        len(jackknife_null.loc[jackknife_null["excluded_chromosome"].eq(chromosome)]) == 10_000
        for chromosome in expected_chromosomes
    ):
        violations.append("jackknife_null_not_10000_each")

    category = genes["category"]
    call = genes["chromatin_call"].astype(int)
    positive = category.ne("negative")
    concordant = (
        (category.eq("positive_up") & call.eq(1))
        | (category.eq("positive_down") & call.eq(-1))
    )
    observed = float(concordant.sum() / positive.sum())
    reported_conditioned = summary["conditioned_permutation"]
    conditioned_mean, conditioned_p = null_metrics(conditioned_null, observed)
    if not close(observed, reported_conditioned["observed_fraction"]):
        violations.append("conditioned_observed_mismatch")
    if not close(conditioned_mean, reported_conditioned["null_mean"]):
        violations.append("conditioned_null_mean_mismatch")
    if not close(conditioned_p, reported_conditioned["one_sided_empirical_p"]):
        violations.append("conditioned_p_mismatch")

    reported_jackknife = {
        item["excluded_chromosome"]: item for item in summary["leave_one_chromosome_out"]
    }
    recalculated_jackknife: list[dict[str, object]] = []
    for chromosome in expected_chromosomes:
        keep = genes["chromosome"].ne(chromosome)
        retained_positive = positive & keep
        retained_observed = float((concordant & keep).sum() / retained_positive.sum())
        null = jackknife_null.loc[
            jackknife_null["excluded_chromosome"].eq(chromosome), "null_fraction"
        ].to_numpy(float)
        mean, pvalue = null_metrics(null, retained_observed)
        reported = reported_jackknife[chromosome]
        if not close(retained_observed, reported["observed_fraction"]):
            violations.append(f"{chromosome}_observed_mismatch")
        if not close(mean, reported["null_mean"]):
            violations.append(f"{chromosome}_null_mean_mismatch")
        if not close(pvalue, reported["one_sided_empirical_p"]):
            violations.append(f"{chromosome}_p_mismatch")
        recalculated_jackknife.append(
            {
                "excluded_chromosome": chromosome,
                "observed_fraction": retained_observed,
                "null_mean": mean,
                "enrichment_ratio": float(retained_observed / mean),
                "one_sided_empirical_p": pvalue,
            }
        )

    pass_rule = bool(
        observed > conditioned_mean
        and all(item["observed_fraction"] > item["null_mean"] for item in recalculated_jackknife)
    )
    if pass_rule != summary["descriptive_robustness_pass"]:
        violations.append("descriptive_robustness_rule_mismatch")
    if summary.get("malus_accessed") is not False:
        violations.append("malus_access_flag_not_false")

    output = {
        "status": "pass" if not violations else "fail",
        "scope": "independent_posthoc_chromosome_output_recalculation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "violations": violations,
        "conditioned": {
            "observed_fraction": observed,
            "null_mean": conditioned_mean,
            "enrichment_ratio": float(observed / conditioned_mean),
            "one_sided_empirical_p": conditioned_p,
        },
        "leave_one_chromosome_out": recalculated_jackknife,
        "descriptive_robustness_pass": pass_rule,
        "artifact_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in (summary_path, conditioned_path, jackknife_path, gene_path)
        },
        "malus_accessed": False,
    }
    output_path = root / "results/metrics/publication_v3_prunus_chromatin_chromosome_verification.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

