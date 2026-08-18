#!/usr/bin/env python3
"""Independently verify the frozen GSE141983 raw-sensitivity outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def statistic(effect: np.ndarray, categories: np.ndarray) -> float:
    up = categories == "positive_up"
    down = categories == "positive_down"
    negative = categories == "negative"
    positive_count = int(up.sum() + down.sum())
    positive_oriented = (effect[up].sum() - effect[down].sum()) / positive_count
    mixture = (up.sum() - down.sum()) / positive_count
    return float(positive_oriented - mixture * effect[negative].mean())


def oriented_effect(effect: np.ndarray, categories: np.ndarray) -> np.ndarray:
    up = categories == "positive_up"
    down = categories == "positive_down"
    positive_count = int(up.sum() + down.sum())
    mixture = (up.sum() - down.sum()) / positive_count
    result = effect * mixture
    result[up] = effect[up]
    result[down] = -effect[down]
    return result


def ols_hc3(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ inverse) * x, axis=1)
    adjusted = residual / np.maximum(1e-10, 1 - leverage)
    covariance = inverse @ (x.T @ ((adjusted**2)[:, None] * x)) @ inverse
    return beta, np.sqrt(np.maximum(0, np.diag(covariance)))


def isclose(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--result-root",
        default="results/biological_cases/prunus_publication_v3_gse141983_raw",
    )
    parser.add_argument(
        "--output",
        default="results/metrics/"
        "publication_v3_gse141983_raw_h3k4me3_verification.json",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    result_root = root / args.result_root
    summary_path = result_root / "summary.json"
    gate_path = result_root / "technical_gate.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    violations: list[str] = []
    if summary.get("malus_accessed") is not False or gate.get("malus_accessed") is not False:
        violations.append("malus_seal_invariant_failed")
    for row in gate.get("sample_qc", []):
        expected = (
            int(row["post_fastp_pairs"]) >= 10_000_000
            and float(row["post_fastp_q30_rate"]) >= 0.80
            and float(row["mapq30_nonduplicate_pair_fraction"]) >= 0.50
            and int(row["assigned_tss_fragments"]) >= 1_000_000
            and int(row["nonzero_tss_genes"]) >= 10_000
        )
        if bool(row["pass"]) != expected:
            violations.append(f"sample_gate_recalculation_failed:{row['run_accession']}")
    for row in gate.get("replicate_qc", []):
        expected = int(row["genes"]) >= 10_000 and float(row["spearman_rho"]) >= 0.70
        if bool(row["pass"]) != expected:
            violations.append(f"replicate_gate_recalculation_failed:{row['stage']}")

    recalculated: dict[str, Any] = {}
    artifact_paths = [summary_path, gate_path]
    if summary.get("status", "").startswith("complete_omitted"):
        if gate.get("status") == "pass" and summary.get("status") == "complete_omitted_technical_gate_failed":
            violations.append("omitted_technical_status_but_gate_passed")
        if (result_root / "analysis/label_chromatin_gene_results.tsv.gz").exists():
            violations.append("label_results_exist_after_technical_omission")
    else:
        analysis_root = result_root / "analysis"
        gene_path = analysis_root / "label_chromatin_gene_results.tsv.gz"
        null_path = analysis_root / "continuous_permutation_null.tsv.gz"
        bootstrap_path = analysis_root / "continuous_bootstrap.tsv.gz"
        gbox_null_path = analysis_root / "exact_gbox_interaction_permutation_null.tsv.gz"
        artifact_paths.extend([gene_path, null_path, bootstrap_path, gbox_null_path])
        genes = pd.read_csv(gene_path, sep="\t")
        effect = genes["chromatin_effect"].to_numpy(dtype=float)
        categories = genes["category"].astype(str).to_numpy()
        observed = statistic(effect, categories)
        null = pd.read_csv(null_path, sep="\t")["statistic"].to_numpy(dtype=float)
        bootstrap = pd.read_csv(bootstrap_path, sep="\t")["statistic"].to_numpy(dtype=float)
        empirical_p = (1 + int(np.sum(null >= observed))) / (len(null) + 1)
        interval = np.quantile(bootstrap, [0.025, 0.975])
        reported = summary["continuous_label_chromatin"]
        comparisons = {
            "statistic": isclose(observed, reported["statistic"]),
            "permutation_null_mean": isclose(null.mean(), reported["permutation_null_mean"]),
            "permutation_p": isclose(empirical_p, reported["one_sided_empirical_p"]),
            "bootstrap_low": isclose(interval[0], reported["bootstrap_95_interval"][0]),
            "bootstrap_high": isclose(interval[1], reported["bootstrap_95_interval"][1]),
        }
        for name, passed in comparisons.items():
            if not passed:
                violations.append(f"continuous_recalculation_failed:{name}")

        label_status = (categories != "negative").astype(float)
        gbox = genes["exact_gbox"].astype(float).to_numpy()
        gc = genes["gc_fraction"].to_numpy(dtype=float)
        cpg = genes["cpg_fraction"].to_numpy(dtype=float)
        expression = genes["baseline_expression_rank_score"].to_numpy(dtype=float)
        y = oriented_effect(effect, categories)
        x = np.column_stack(
            [np.ones(len(genes)), label_status, gbox, label_status * gbox, gc, gc**2, cpg, expression]
        )
        beta, se = ols_hc3(x, y)
        gbox_null = pd.read_csv(gbox_null_path, sep="\t")["interaction_beta"].to_numpy(dtype=float)
        gbox_p = (1 + int(np.sum(gbox_null >= beta[3]))) / (len(gbox_null) + 1)
        reported_gbox = summary["exact_gbox_interaction"]
        gbox_comparisons = {
            "interaction_beta": isclose(beta[3], reported_gbox["interaction_beta"]),
            "hc3_standard_error": isclose(se[3], reported_gbox["hc3_standard_error"]),
            "permutation_p": isclose(gbox_p, reported_gbox["one_sided_permutation_p"]),
        }
        for name, passed in gbox_comparisons.items():
            if not passed:
                violations.append(f"gbox_recalculation_failed:{name}")
        recalculated = {
            "continuous_statistic": observed,
            "continuous_null_mean": float(null.mean()),
            "continuous_empirical_p": empirical_p,
            "continuous_bootstrap_95_interval": interval.tolist(),
            "continuous_comparisons": comparisons,
            "gbox_interaction_beta": float(beta[3]),
            "gbox_hc3_standard_error": float(se[3]),
            "gbox_permutation_p": gbox_p,
            "gbox_comparisons": gbox_comparisons,
        }

    verification = {
        "status": "pass" if not violations else "fail",
        "scope": "independent_gse141983_raw_h3k4me3_output_recalculation",
        "scientific_decision_authority": False,
        "malus_accessed": False,
        "violations": violations,
        "analysis_status": summary.get("status"),
        "recalculated": recalculated,
        "artifact_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in artifact_paths
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
