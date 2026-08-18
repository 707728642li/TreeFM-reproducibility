#!/usr/bin/env python3
"""Exploratory fixed G-box/ABRE-like integration with Pyrus H3K4me3."""

from __future__ import annotations

import argparse
import os
import gzip
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm


MOTIFS = ("CACGTG", "ACGTG", "ACGT", "CACG")
DAYS = (0, 10, 20, 30, 40, 45, 50)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def overlapping_count(sequence: str, motif: str) -> int:
    return len(re.findall(f"(?={re.escape(motif)})", sequence))


def motif_features(sequence: str) -> dict[str, object]:
    sequence = sequence.upper()
    result: dict[str, object] = {}
    total = 0
    for motif in MOTIFS:
        variants = {motif, reverse_complement(motif)}
        count = sum(overlapping_count(sequence, item) for item in variants)
        result[f"motif_{motif}_count"] = count
        total += count
    valid = sum(sequence.count(base) for base in "ACGT")
    result["fixed_motif_family_count"] = total
    result["fixed_motif_family_present"] = int(total > 0)
    result["cpg_fraction"] = (
        sequence.count("CG") / max(len(sequence) - 1, 1)
    )
    result["valid_base_fraction"] = valid / max(len(sequence), 1)
    return result


def hc3_fit(
    matrix: np.ndarray,
    response: np.ndarray,
) -> dict[str, np.ndarray]:
    inverse = np.linalg.pinv(matrix.T @ matrix)
    coefficients = inverse @ matrix.T @ response
    residuals = response - matrix @ coefficients
    leverage = np.einsum("ij,jk,ik->i", matrix, inverse, matrix)
    scaled = residuals / np.clip(1 - leverage, 1e-8, None)
    meat = matrix.T @ ((scaled**2)[:, None] * matrix)
    covariance = inverse @ meat @ inverse
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    z_values = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2 * norm.sf(np.abs(z_values))
    return {
        "beta": coefficients,
        "se": standard_errors,
        "z": z_values,
        "p": p_values,
    }


def orient_changes(
    categories: np.ndarray,
    changes: np.ndarray,
) -> np.ndarray:
    positive_up = categories == "positive_up"
    positive_down = categories == "positive_down"
    positive_count = int(positive_up.sum() + positive_down.sum())
    if positive_count == 0:
        raise RuntimeError("no positive genes for direction mixture")
    mixture = (positive_up.sum() - positive_down.sum()) / positive_count
    directions = np.full(len(categories), mixture, dtype=np.float64)
    directions[positive_up] = 1.0
    directions[positive_down] = -1.0
    return directions * changes


def design_matrix(frame: pd.DataFrame, categories: np.ndarray) -> np.ndarray:
    positive = np.isin(categories, ["positive_up", "positive_down"]).astype(
        np.float64
    )
    motif = frame["fixed_motif_family_present"].to_numpy(dtype=np.float64)
    gc = frame["gc_fraction"].to_numpy(dtype=np.float64)
    cpg = frame["cpg_fraction"].to_numpy(dtype=np.float64)
    expression = frame["baseline_log2_mean_cpm"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.ones(len(frame)),
            positive,
            motif,
            positive * motif,
            gc,
            gc**2,
            cpg,
            expression,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstraps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-wait-hours", type=float, default=168)
    args = parser.parse_args()
    if min(
        args.permutations,
        args.bootstraps,
        args.poll_seconds,
        args.maximum_wait_hours,
    ) <= 0:
        raise ValueError("analysis and wait settings must be positive")
    root = args.project_root.resolve()
    expected = Path(os.environ["TREEFM_ROOT"]).resolve() if os.environ.get("TREEFM_ROOT") else None
    if expected is not None and root != expected:
        raise SystemExit(f"refusing to run outside {expected}: {root}")
    result_root = (
        root
        / "results/biological_cases/"
        "publication_v3_exploratory_gbox_chromatin"
    )
    controller_path = result_root / "controller.json"
    state: dict[str, object] = {
        "status": "waiting_for_primary_pyrus_chromatin_analysis",
        "started_utc": utc_now(),
        "contract": "docs/publication_v3_exploratory_gbox_chromatin_plan.md",
    }
    write_json(controller_path, state)
    primary_root = (
        root / "results/biological_cases/pyrus_publication_v3_chipseq"
    )
    try:
        started = time.time()
        while True:
            primary = read_json(primary_root / "mechanistic_controller.json") or {}
            status = str(primary.get("status", "absent"))
            state.update(
                {
                    "status": "waiting_for_primary_pyrus_chromatin_analysis",
                    "primary_status": status,
                    "wait_elapsed_seconds": round(time.time() - started, 3),
                }
            )
            write_json(controller_path, state)
            if status in {"complete", "complete_omitted", "failed"}:
                break
            if time.time() - started > args.maximum_wait_hours * 3600:
                raise TimeoutError("timed out waiting for primary chromatin analysis")
            time.sleep(args.poll_seconds)
        if status != "complete":
            summary = {
                "status": "omitted",
                "reason": f"primary_status={status}",
                "contract": state["contract"],
            }
            write_json(result_root / "summary.json", summary)
            state.update({"status": "complete_omitted", "finished_utc": utc_now()})
            write_json(controller_path, state)
            return

        state["status"] = "building_fixed_motif_gene_table"
        write_json(controller_path, state)
        gene_table = pd.read_csv(
            root
            / "data/processed/functional/Pyrus_PRJNA669907/"
            "chipseq_mechanistic_gene_table.tsv.gz",
            sep="\t",
        )
        promoters = pd.read_parquet(
            root
            / "data/processed/functional/Pyrus_PRJNA669907/"
            "promoter_labels.parquet",
            columns=["gene_id", "promoter_2048"],
        )
        features = pd.DataFrame(
            [
                {"gene_id": gene_id, **motif_features(sequence)}
                for gene_id, sequence in promoters[
                    ["gene_id", "promoter_2048"]
                ].itertuples(index=False, name=None)
            ]
        )
        frame = gene_table.merge(features, on="gene_id", validate="one_to_one")
        if len(frame) != len(gene_table):
            raise RuntimeError("motif merge lost mechanistic genes")
        if not frame["valid_base_fraction"].eq(1.0).all():
            raise RuntimeError("fixed motif analysis encountered ambiguous promoter")
        categories = frame["category"].to_numpy(dtype=str)
        changes = frame[
            "nonduplicate_score_change_day50_minus_day0"
        ].to_numpy(dtype=np.float64)
        response = orient_changes(categories, changes)
        matrix = design_matrix(frame, categories)
        fit = hc3_fit(matrix, response)
        interaction_index = 3
        observed = float(fit["beta"][interaction_index])

        rng = np.random.default_rng(args.seed)
        strata = [
            group.index.to_numpy()
            for _, group in frame.groupby("permutation_stratum", sort=True)
        ]
        null = np.empty(args.permutations, dtype=np.float64)
        for repeat in range(args.permutations):
            permuted = categories.copy()
            for indices in strata:
                permuted[indices] = rng.permutation(permuted[indices])
            permuted_response = orient_changes(permuted, changes)
            null[repeat] = hc3_fit(
                design_matrix(frame, permuted),
                permuted_response,
            )["beta"][interaction_index]
        empirical_p = float(
            (1 + np.count_nonzero(null >= observed))
            / (1 + args.permutations)
        )
        pd.DataFrame(
            {
                "permutation": np.arange(1, args.permutations + 1),
                "positive_by_motif_beta": null,
            }
        ).to_csv(result_root / "permutation_null.tsv", sep="\t", index=False)

        category_indices = {
            category: np.flatnonzero(categories == category)
            for category in ("positive_up", "positive_down", "negative")
        }
        bootstrap = np.empty(args.bootstraps, dtype=np.float64)
        for repeat in range(args.bootstraps):
            sampled = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in category_indices.values()
                ]
            )
            sampled_frame = frame.iloc[sampled].reset_index(drop=True)
            sampled_categories = sampled_frame["category"].to_numpy(dtype=str)
            sampled_changes = sampled_frame[
                "nonduplicate_score_change_day50_minus_day0"
            ].to_numpy(dtype=np.float64)
            bootstrap[repeat] = hc3_fit(
                design_matrix(sampled_frame, sampled_categories),
                orient_changes(sampled_categories, sampled_changes),
            )["beta"][interaction_index]
        bootstrap_ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()

        positive = frame["category"].isin(["positive_up", "positive_down"])
        positive_frame = frame.loc[positive].reset_index(drop=True)
        positive_direction = (
            positive_frame["category"].eq("positive_up").astype(float) * 2 - 1
        ).to_numpy()
        positive_response = (
            positive_direction
            * positive_frame[
                "nonduplicate_score_change_day50_minus_day0"
            ].to_numpy(dtype=np.float64)
        )
        positive_matrix = np.column_stack(
            [
                np.ones(len(positive_frame)),
                positive_frame["fixed_motif_family_present"],
                positive_direction,
                positive_frame["gc_fraction"],
                positive_frame["gc_fraction"] ** 2,
                positive_frame["cpg_fraction"],
                positive_frame["baseline_log2_mean_cpm"],
            ]
        ).astype(np.float64)
        positive_fit = hc3_fit(positive_matrix, positive_response)

        acquired = (
            frame["promoter_broad_peak_day50"].astype(bool)
            & ~frame["promoter_broad_peak_day0"].astype(bool)
        )
        motif = frame["fixed_motif_family_present"].astype(bool)
        table = np.array(
            [
                [(acquired & motif).sum(), ((~acquired) & motif).sum()],
                [(acquired & ~motif).sum(), ((~acquired) & ~motif).sum()],
            ],
            dtype=int,
        )
        peak_odds_ratio, peak_p = fisher_exact(
            table, alternative="greater"
        )

        trajectory_records = []
        for category in ("positive_up", "positive_down", "negative"):
            for motif_present in (0, 1):
                selected = (
                    frame["category"].eq(category)
                    & frame["fixed_motif_family_present"].eq(motif_present)
                )
                for day in DAYS:
                    values = frame.loc[
                        selected, f"nonduplicate_score_day{day}"
                    ]
                    trajectory_records.append(
                        {
                            "category": category,
                            "fixed_motif_family_present": motif_present,
                            "day": day,
                            "genes": int(selected.sum()),
                            "mean_score": float(values.mean()),
                            "median_score": float(values.median()),
                        }
                    )
        pd.DataFrame(trajectory_records).to_csv(
            result_root / "trajectory_summary.tsv", sep="\t", index=False
        )
        with gzip.open(
            result_root / "gbox_chromatin_gene_table.tsv.gz",
            "wt",
            encoding="utf-8",
        ) as handle:
            frame.to_csv(handle, sep="\t", index=False)

        names = (
            "intercept",
            "positive",
            "motif",
            "positive_by_motif",
            "gc",
            "gc_squared",
            "cpg",
            "baseline_expression",
        )
        regression = {
            name: {
                "beta": float(fit["beta"][index]),
                "hc3_se": float(fit["se"][index]),
                "z": float(fit["z"][index]),
                "two_sided_p": float(fit["p"][index]),
            }
            for index, name in enumerate(names)
        }
        summary = {
            "status": "complete",
            "scope": "exploratory_nonconfirmatory",
            "contract": state["contract"],
            "genes": len(frame),
            "motif_family": list(MOTIFS),
            "motif_positive_genes": int(
                frame["fixed_motif_family_present"].sum()
            ),
            "primary_statistic": "positive_by_motif_beta",
            "primary_beta": observed,
            "primary_hc3_se": float(fit["se"][interaction_index]),
            "primary_two_sided_p": float(fit["p"][interaction_index]),
            "primary_one_sided_permutation_p": empirical_p,
            "primary_bootstrap_ci95": bootstrap_ci,
            "regression": regression,
            "positive_only_motif_beta": float(positive_fit["beta"][1]),
            "positive_only_motif_hc3_se": float(positive_fit["se"][1]),
            "positive_only_motif_two_sided_p": float(positive_fit["p"][1]),
            "peak_acquisition_table": table.tolist(),
            "peak_acquisition_odds_ratio": float(peak_odds_ratio),
            "peak_acquisition_one_sided_fisher_p": float(peak_p),
            "interpretation_limit": (
                "motif family was discovered from the same non-Malus label "
                "resource; this is mechanistic triangulation, not independent "
                "confirmation or a causal test"
            ),
        }
        write_json(result_root / "summary.json", summary)
        state.update(
            {
                "status": "complete",
                "finished_utc": utc_now(),
                "summary": str(
                    (result_root / "summary.json").relative_to(root)
                ),
            }
        )
        write_json(controller_path, state)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(controller_path, state)
        raise


if __name__ == "__main__":
    main()
