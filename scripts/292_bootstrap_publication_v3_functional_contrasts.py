#!/usr/bin/env python3
"""Run the frozen paired label-stratified functional bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
CONTROLS = ("herb", "random_plant", "phylogc_match")
CONTRASTS = (
    "tree_minus_base",
    "tree_minus_herb",
    "tree_minus_random_plant",
    "tree_minus_phylogc_match",
    "tree_minus_max_matched_control",
)
GENERA = ("prunus", "pyrus")
READOUTS = ("linear", "xgboost")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate(scores: dict[str, np.ndarray], labels: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    values = {
        arm: float(average_precision_score(labels[indices], probability[indices]))
        for arm, probability in scores.items()
    }
    return {
        "tree_minus_base": values["tree"] - values["base"],
        "tree_minus_herb": values["tree"] - values["herb"],
        "tree_minus_random_plant": values["tree"] - values["random_plant"],
        "tree_minus_phylogc_match": values["tree"] - values["phylogc_match"],
        "tree_minus_max_matched_control": values["tree"]
        - max(values[control] for control in CONTROLS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260801)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.replicates != 2000:
        raise ValueError("the frozen contract requires exactly 2,000 replicates")
    root = args.project_root.resolve()
    source_root = root / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
    output = root / "results/metrics/publication_v3_rebuild_functional_bootstrap"
    output.mkdir(parents=True, exist_ok=True)

    source_hashes: dict[str, str] = {}
    replicate_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for genus_index, genus in enumerate(GENERA):
        canonical: pd.DataFrame | None = None
        draws: list[np.ndarray] | None = None
        for readout in READOUTS:
            scores: dict[str, np.ndarray] = {}
            for arm in ARMS:
                path = source_root / arm / "seed_23" / f"heldout_{genus}.{readout}.predictions.parquet"
                if not path.is_file():
                    raise FileNotFoundError(path)
                source_hashes[str(path.relative_to(root))] = sha256(path)
                data = pd.read_parquet(path).sort_values("gene_id").reset_index(drop=True)
                if canonical is None:
                    canonical = data[["gene_id", "label_binary"]].copy()
                    labels = canonical["label_binary"].to_numpy(dtype=np.int8)
                    positive = np.flatnonzero(labels == 1)
                    negative = np.flatnonzero(labels == 0)
                    rng = np.random.default_rng(args.random_seed + genus_index)
                    draws = [
                        np.concatenate(
                            (
                                rng.choice(positive, size=len(positive), replace=True),
                                rng.choice(negative, size=len(negative), replace=True),
                            )
                        )
                        for _ in range(args.replicates)
                    ]
                elif not data[["gene_id", "label_binary"]].equals(canonical):
                    raise RuntimeError(f"prediction rows do not align for {arm}/{genus}/{readout}")
                scores[arm] = data["probability"].to_numpy(dtype=np.float64)

            assert canonical is not None and draws is not None
            labels = canonical["label_binary"].to_numpy(dtype=np.int8)
            full_indices = np.arange(len(labels), dtype=np.int64)
            point = calculate(scores, labels, full_indices)
            boot = Parallel(n_jobs=args.n_jobs, batch_size=20)(
                delayed(calculate)(scores, labels, indices) for indices in draws
            )
            for replicate, values in enumerate(boot):
                for contrast, delta in values.items():
                    replicate_records.append(
                        {
                            "heldout_genus": genus,
                            "readout": readout,
                            "replicate": replicate,
                            "contrast": contrast,
                            "delta_auprc": delta,
                        }
                    )
            for contrast in CONTRASTS:
                values = np.asarray([row[contrast] for row in boot], dtype=np.float64)
                q025, q05, q50, q95, q975 = np.quantile(
                    values, [0.025, 0.05, 0.5, 0.95, 0.975]
                )
                summary_records.append(
                    {
                        "heldout_genus": genus,
                        "readout": readout,
                        "contrast": contrast,
                        "point_delta_auprc": point[contrast],
                        "bootstrap_median": q50,
                        "ci95_low": q025,
                        "ci95_high": q975,
                        "ci90_low": q05,
                        "ci90_high": q95,
                        "probability_gt_0": float(np.mean(values > 0.0)),
                        "probability_ge_0_02": float(np.mean(values >= 0.02)),
                        "equivalent_within_0_02_by_90ci": bool(
                            q05 > -0.02 and q95 < 0.02
                        ),
                        "replicates": args.replicates,
                    }
                )

    replicates = pd.DataFrame(replicate_records)
    summary = pd.DataFrame(summary_records)
    replicates.to_parquet(output / "paired_bootstrap_replicates.parquet", index=False)
    summary.to_csv(output / "contrast_summary.tsv", sep="\t", index=False)
    woody = summary.loc[summary["contrast"].eq("tree_minus_max_matched_control")].copy()
    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "docs/publication_v3_functional_analysis_contract.md",
        "method": "label-stratified paired gene bootstrap; identical draws across arms",
        "replicates": args.replicates,
        "random_seed": args.random_seed,
        "n_jobs": args.n_jobs,
        "woody_control_cells": woody.to_dict(orient="records"),
        "all_woody_control_point_estimates_negative": bool(
            woody["point_delta_auprc"].lt(0).all()
        ),
        "woody_control_cells_ci95_excluding_zero": int(
            ((woody["ci95_low"] > 0) | (woody["ci95_high"] < 0)).sum()
        ),
        "malus_accessed": False,
        "input_prediction_sha256": source_hashes,
        "tables": {
            "summary": "contrast_summary.tsv",
            "replicates": "paired_bootstrap_replicates.parquet",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
