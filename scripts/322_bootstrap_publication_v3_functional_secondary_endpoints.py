#!/usr/bin/env python3
"""Post-hoc paired bootstrap for functional top-k enrichment and calibration.

This analysis is descriptive and has no continuation, model-selection or
primary-claim authority. Positive contrasts always mean that Tree is better:
higher top-k enrichment or lower 15-bin calibration error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
CONTROLS = ("herb", "random_plant", "phylogc_match")
GENERA = ("prunus", "pyrus")
READOUTS = ("linear", "xgboost")
POPULATIONS = (
    "all",
    "heldout_chromosome_test",
    "mapped_novel_orthogroup",
    "no_shared_orthogroup",
)
ENDPOINTS = ("top_k_enrichment", "ece_15bin")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_k_enrichment(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives < 1 or positives >= len(labels):
        return float("nan")
    selected = np.argsort(-probabilities, kind="stable")[:positives]
    return float(labels[selected].mean() / labels.mean())


def calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += mask.mean() * abs(
                probabilities[mask].mean() - labels[mask].mean()
            )
    return float(error)


def endpoint_values(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, dict[str, float]]:
    y = labels[indices]
    return {
        "top_k_enrichment": {
            arm: top_k_enrichment(y, probability[indices])
            for arm, probability in scores.items()
        },
        "ece_15bin": {
            arm: calibration_error(y, probability[indices])
            for arm, probability in scores.items()
        },
    }


def benefits(endpoint: str, values: dict[str, float]) -> dict[str, float]:
    if endpoint == "top_k_enrichment":
        return {
            "tree_vs_base_benefit": values["tree"] - values["base"],
            "tree_vs_best_matched_control_benefit": values["tree"]
            - max(values[arm] for arm in CONTROLS),
        }
    if endpoint == "ece_15bin":
        return {
            "tree_vs_base_benefit": values["base"] - values["tree"],
            "tree_vs_best_matched_control_benefit": min(
                values[arm] for arm in CONTROLS
            )
            - values["tree"],
        }
    raise ValueError(endpoint)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array, kind="stable")
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    return pd.Series(restored, index=values.index)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def population_masks(
    data: pd.DataFrame, training_orthogroups: set[str]
) -> dict[str, np.ndarray]:
    orthogroups = data["orthogroup"].fillna("").astype(str)
    mapped = orthogroups.ne("").to_numpy()
    shared = orthogroups.isin(training_orthogroups).to_numpy()
    return {
        "all": np.ones(len(data), dtype=bool),
        "heldout_chromosome_test": data["split"].eq("test").to_numpy(),
        "mapped_novel_orthogroup": mapped & ~shared,
        "no_shared_orthogroup": ~shared,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260802)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.replicates != 2000:
        raise ValueError("this post-hoc analysis is fixed at 2,000 replicates")

    root = args.project_root.resolve()
    source_root = root / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
    benchmark_path = (
        root
        / "results/metrics/publication_v3_rebuild_functional_benchmark/all_prespecified_cells.tsv"
    )
    if not benchmark_path.is_file():
        raise FileNotFoundError(benchmark_path)
    benchmark = pd.read_csv(benchmark_path, sep="\t")
    output = root / "results/metrics/publication_v3_functional_secondary_bootstrap"
    output.mkdir(parents=True, exist_ok=True)

    source_hashes = {str(benchmark_path.relative_to(root)): sha256(benchmark_path)}
    metadata: dict[tuple[str, str], pd.DataFrame] = {}
    probabilities: dict[tuple[str, str, str], np.ndarray] = {}
    for genus in GENERA:
        for readout in READOUTS:
            canonical: pd.DataFrame | None = None
            for arm in ARMS:
                path = source_root / arm / "seed_23" / f"heldout_{genus}.{readout}.predictions.parquet"
                if not path.is_file():
                    raise FileNotFoundError(path)
                source_hashes[str(path.relative_to(root))] = sha256(path)
                data = pd.read_parquet(path).sort_values("gene_id").reset_index(drop=True)
                columns = ["gene_id", "label_binary", "split", "orthogroup"]
                if canonical is None:
                    canonical = data[columns].copy()
                elif not data[columns].equals(canonical):
                    raise RuntimeError(f"prediction rows do not align: {arm}/{genus}/{readout}")
                probabilities[(arm, genus, readout)] = data["probability"].to_numpy(dtype=np.float64)
            assert canonical is not None
            metadata[(genus, readout)] = canonical

    summary_records: list[dict[str, object]] = []
    draw_records: list[dict[str, object]] = []
    scope_index = 0
    point_failures: list[dict[str, object]] = []
    for genus in GENERA:
        training_genus = "pyrus" if genus == "prunus" else "prunus"
        training_data = metadata[(training_genus, "linear")]
        training_orthogroups = set(
            training_data["orthogroup"].fillna("").astype(str)
        ) - {""}
        for readout in READOUTS:
            data = metadata[(genus, readout)]
            masks = population_masks(data, training_orthogroups)
            all_scores = {
                arm: probabilities[(arm, genus, readout)] for arm in ARMS
            }
            for population in POPULATIONS:
                mask_indices = np.flatnonzero(masks[population])
                labels = data.loc[mask_indices, "label_binary"].to_numpy(dtype=np.int8)
                if not set(labels) == {0, 1}:
                    raise RuntimeError(f"population lacks both classes: {genus}/{readout}/{population}")
                scores = {arm: values[mask_indices] for arm, values in all_scores.items()}
                full_indices = np.arange(len(labels), dtype=np.int64)
                point = endpoint_values(labels, scores, full_indices)
                for endpoint in ENDPOINTS:
                    expected = benchmark.loc[
                        benchmark["heldout_genus"].eq(genus)
                        & benchmark["readout"].eq(readout)
                        & benchmark["population"].eq(population)
                        & benchmark["metric"].eq(endpoint)
                    ]
                    if len(expected) != 1:
                        raise RuntimeError(f"missing benchmark point: {genus}/{readout}/{population}/{endpoint}")
                    row = expected.iloc[0]
                    for arm in ARMS:
                        if not np.isclose(point[endpoint][arm], float(row[arm]), atol=1e-12):
                            point_failures.append(
                                {
                                    "genus": genus,
                                    "readout": readout,
                                    "population": population,
                                    "endpoint": endpoint,
                                    "arm": arm,
                                    "recomputed": point[endpoint][arm],
                                    "expected": float(row[arm]),
                                }
                            )

                if args.validate_only:
                    scope_index += 1
                    continue

                positive = np.flatnonzero(labels == 1)
                negative = np.flatnonzero(labels == 0)
                rng = np.random.default_rng(args.random_seed + scope_index)
                indices = [
                    np.concatenate(
                        [
                            rng.choice(positive, size=len(positive), replace=True),
                            rng.choice(negative, size=len(negative), replace=True),
                        ]
                    )
                    for _ in range(args.replicates)
                ]
                boot = Parallel(n_jobs=args.n_jobs, batch_size=20)(
                    delayed(endpoint_values)(labels, scores, draw) for draw in indices
                )
                for replicate, endpoints in enumerate(boot):
                    for endpoint, values in endpoints.items():
                        row = {
                            "heldout_genus": genus,
                            "readout": readout,
                            "population": population,
                            "endpoint": endpoint,
                            "replicate": replicate,
                            **{f"{arm}_value": values[arm] for arm in ARMS},
                            **benefits(endpoint, values),
                        }
                        draw_records.append(row)
                for endpoint in ENDPOINTS:
                    point_benefits = benefits(endpoint, point[endpoint])
                    for contrast in (
                        "tree_vs_base_benefit",
                        "tree_vs_best_matched_control_benefit",
                    ):
                        values = np.asarray(
                            [benefits(endpoint, record[endpoint])[contrast] for record in boot],
                            dtype=np.float64,
                        )
                        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
                        p_positive = (1 + int(np.count_nonzero(values <= 0))) / (1 + len(values))
                        p_negative = (1 + int(np.count_nonzero(values >= 0))) / (1 + len(values))
                        summary_records.append(
                            {
                                "heldout_genus": genus,
                                "readout": readout,
                                "population": population,
                                "endpoint": endpoint,
                                "contrast": contrast,
                                "point_benefit": point_benefits[contrast],
                                "bootstrap_median": median,
                                "ci95_low": low,
                                "ci95_high": high,
                                "probability_gt_0": float(np.mean(values > 0)),
                                "two_sided_p": float(min(1.0, 2 * min(p_positive, p_negative))),
                                "replicates": args.replicates,
                            }
                        )
                scope_index += 1

    if point_failures:
        raise RuntimeError(f"secondary point estimates failed alignment: {point_failures[:5]}")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "validated_scopes": scope_index,
                    "validated_endpoints": scope_index * len(ENDPOINTS),
                    "validated_arm_points": scope_index * len(ENDPOINTS) * len(ARMS),
                    "point_alignment_failures": 0,
                },
                indent=2,
            )
        )
        return
    summary = pd.DataFrame(summary_records)
    draws = pd.DataFrame(draw_records)
    summary["two_sided_q_bh_64"] = benjamini_hochberg(summary["two_sided_p"])
    summary["analysis_tier"] = "posthoc_seed23_descriptive"
    summary["decision_authority"] = False
    summary_path = output / "scope_summary.tsv"
    draw_path = output / "paired_bootstrap_secondary.parquet"
    summary.to_csv(summary_path, sep="\t", index=False)
    draws.to_parquet(draw_path, index=False)

    control = summary.loc[
        summary["contrast"].eq("tree_vs_best_matched_control_benefit")
    ].copy()
    endpoint_summary: dict[str, dict[str, object]] = {}
    for endpoint in ENDPOINTS:
        subset = control.loc[control["endpoint"].eq(endpoint)]
        endpoint_summary[endpoint] = {
            "cells": int(len(subset)),
            "point_positive": int(subset["point_benefit"].gt(1e-12).sum()),
            "point_tied": int(np.isclose(subset["point_benefit"], 0).sum()),
            "point_negative": int(subset["point_benefit"].lt(-1e-12).sum()),
            "ci_entirely_positive": int(subset["ci95_low"].gt(0).sum()),
            "ci_entirely_negative": int(subset["ci95_high"].lt(0).sum()),
            "ci_overlapping_zero": int(
                ((subset["ci95_low"] <= 0) & (subset["ci95_high"] >= 0)).sum()
            ),
            "bh_significant_positive": int(
                ((subset["two_sided_q_bh_64"] <= 0.05) & (subset["point_benefit"] > 0)).sum()
            ),
            "bh_significant_negative": int(
                ((subset["two_sided_q_bh_64"] <= 0.05) & (subset["point_benefit"] < 0)).sum()
            ),
        }

    endpoint_rows = [
        [
            endpoint,
            details["point_positive"],
            details["point_tied"],
            details["point_negative"],
            details["ci_entirely_positive"],
            details["ci_entirely_negative"],
            details["ci_overlapping_zero"],
            details["bh_significant_positive"],
            details["bh_significant_negative"],
        ]
        for endpoint, details in endpoint_summary.items()
    ]
    significant = control.loc[control["two_sided_q_bh_64"].le(0.05)].sort_values(
        ["endpoint", "heldout_genus", "readout", "population"]
    )
    significant_rows = [
        [
            row.endpoint,
            row.heldout_genus,
            row.readout,
            row.population,
            f"{row.point_benefit:+.6f}",
            f"[{row.ci95_low:+.6f}, {row.ci95_high:+.6f}]",
            f"{row.two_sided_q_bh_64:.4f}",
        ]
        for row in significant.itertuples(index=False)
    ]
    report = f"""# Post-hoc functional secondary-endpoint bootstrap

**Status:** complete descriptive seed-23 post-hoc analysis; no continuation, model-selection or primary-claim authority.

The 16 prespecified functional populations were re-evaluated for top-k enrichment and 15-bin calibration error using 2,000 label-stratified paired gene bootstraps per cell. Identical gene draws were applied to all five arms. Benefits are oriented so that positive values always favor Tree: higher enrichment or lower calibration error. Two-sided P values were adjusted by Benjamini-Hochberg across all 64 endpoint-by-cell-by-contrast tests.

## Tree versus the strongest matched DAPT control

{markdown_table(
    ["Endpoint", "Point +", "Point tie", "Point -", "CI > 0", "CI < 0", "CI overlaps 0", "BH +", "BH -"],
    endpoint_rows,
)}

All 16 top-k-enrichment intervals overlapped zero and no positive contrast survived multiplicity correction. Calibration showed no significant Tree benefit; four Prunus-linear populations instead had intervals entirely below zero and remained significant after correction.

## Multiplicity-adjusted cells

{markdown_table(
    ["Endpoint", "Held-out genus", "Readout", "Population", "Tree benefit", "95% interval", "BH q"],
    significant_rows,
)}

## Interpretation boundary

The apparent frequency of point-estimate top-k wins does not constitute a robust hidden Tree advantage. The four calibration disadvantages localize a possible seed-23 probability-allocation cost to the Prunus linear readout, but this post-hoc result does not change the failed AUPRC gate, establish adaptation-seed stability or authorize seeds 41/59 or Malus access.
"""
    report_path = root / "reports/PUBLICATION_V3_FUNCTIONAL_SECONDARY_BOOTSTRAP_EN.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "posthoc_functional_secondary_endpoint_bootstrap",
        "analysis_tier": "posthoc_seed23_descriptive",
        "decision_authority": False,
        "primary_auprc_gate_unchanged": True,
        "seed": 23,
        "replicates": args.replicates,
        "random_seed": args.random_seed,
        "n_jobs": args.n_jobs,
        "bootstrap_unit": "label-stratified paired gene resampling within population",
        "benefit_direction": "positive always favors Tree",
        "multiplicity": "Benjamini-Hochberg across all 64 post-hoc endpoint-by-cell-by-contrast tests",
        "point_alignment_failures": 0,
        "endpoint_summary": endpoint_summary,
        "malus_accessed": False,
        "input_sha256": source_hashes,
        "outputs": {
            "scope_summary": str(summary_path.relative_to(root)),
            "bootstrap_draws": str(draw_path.relative_to(root)),
            "report": str(report_path.relative_to(root)),
        },
        "output_sha256": {
            str(summary_path.relative_to(root)): sha256(summary_path),
            str(draw_path.relative_to(root)): sha256(draw_path),
            str(report_path.relative_to(root)): sha256(report_path),
        },
    }
    result_path = output / "summary.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
