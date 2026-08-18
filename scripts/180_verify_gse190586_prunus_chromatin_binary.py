#!/usr/bin/env python3
"""Independently verify frozen GSE190586 binary-sensitivity outputs."""

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


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    result_root = root / "results/biological_cases/prunus_publication_v3_chromatin_replication"
    summary_path = result_root / "summary.json"
    gene_path = result_root / "gse190586_binary_gene_calls.tsv.gz"
    null_path = result_root / "gse190586_binary_permutation_null.tsv.gz"
    bootstrap_path = result_root / "gse190586_binary_bootstrap.tsv.gz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    genes = pd.read_csv(gene_path, sep="\t")
    null = pd.read_csv(null_path, sep="\t")["null_fraction"].to_numpy(float)
    bootstrap = pd.read_csv(bootstrap_path, sep="\t")["fraction"].to_numpy(float)

    violations: list[str] = []
    if len(genes) != 2492 or genes["gene_id"].duplicated().any():
        violations.append("gene_table_not_2492_unique_genes")
    expected_categories = {"positive_up": 153, "positive_down": 334, "negative": 2005}
    observed_categories = genes["category"].value_counts().to_dict()
    if observed_categories != expected_categories:
        violations.append(f"category_counts_mismatch:{observed_categories}")
    if set(genes["chromatin_call"].astype(int).unique()) - {-1, 0, 1}:
        violations.append("invalid_chromatin_call")
    if genes["stratum"].nunique() != 100:
        violations.append("not_100_joint_strata")
    if len(null) != 10_000 or not np.isfinite(null).all():
        violations.append("invalid_permutation_null")
    if len(bootstrap) != 2_000 or not np.isfinite(bootstrap).all():
        violations.append("invalid_bootstrap")

    positive = genes["category"].isin(["positive_up", "positive_down"])
    concordant = (
        (genes["category"].eq("positive_up") & genes["chromatin_call"].eq(1))
        | (genes["category"].eq("positive_down") & genes["chromatin_call"].eq(-1))
    )
    observed = float(concordant.loc[positive].mean())
    empirical_p = float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
    ci = np.quantile(bootstrap, [0.025, 0.975])
    reported = summary["binary_analysis"]
    comparisons = {
        "observed_fraction": close(observed, reported["observed_concordant_positive_fraction"]),
        "concordant_genes": int(concordant.loc[positive].sum()) == reported["concordant_positive_genes"],
        "positive_genes": int(positive.sum()) == reported["positive_genes"],
        "permutation_p": close(empirical_p, reported["one_sided_empirical_p"]),
        "null_mean": close(float(null.mean()), reported["permutation_null_mean"]),
        "bootstrap_ci_low": close(float(ci[0]), reported["bootstrap_95_interval"][0]),
        "bootstrap_ci_high": close(float(ci[1]), reported["bootstrap_95_interval"][1]),
    }
    violations.extend(key for key, passed in comparisons.items() if not passed)
    if summary.get("malus_accessed") is not False:
        violations.append("malus_access_flag_not_false")
    if summary.get("pyrus_rescue_allowed") is not False:
        violations.append("pyrus_rescue_flag_not_false")

    output = {
        "status": "pass" if not violations else "fail",
        "scope": "independent_output_recalculation_and_integrity_check",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "violations": violations,
        "recalculated": {
            "positive_genes": int(positive.sum()),
            "concordant_positive_genes": int(concordant.loc[positive].sum()),
            "observed_fraction": observed,
            "permutation_null_mean": float(null.mean()),
            "null_enrichment_ratio": float(observed / null.mean()),
            "absolute_excess_fraction": float(observed - null.mean()),
            "one_sided_empirical_p": empirical_p,
            "bootstrap_95_interval": [float(ci[0]), float(ci[1])],
        },
        "comparisons": comparisons,
        "artifact_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in (summary_path, gene_path, null_path, bootstrap_path)
        },
        "malus_accessed": False,
    }
    output_path = root / "results/metrics/publication_v3_prunus_chromatin_binary_verification.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

