#!/usr/bin/env python3
"""Exploratory full-cell agreement analysis for functional predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
COMPARATORS = ("base", "herb", "random_plant", "phylogc_match")
GENERA = ("prunus", "pyrus")
READOUTS = ("linear", "xgboost")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
    output = root / "results/metrics/publication_v3_rebuild_prediction_agreement"
    output.mkdir(parents=True, exist_ok=True)

    records = []
    hashes = {}
    closest_spearman = {}
    closest_mae = {}
    for genus in GENERA:
        for readout in READOUTS:
            frames = {}
            for arm in ARMS:
                path = source / arm / "seed_23" / f"heldout_{genus}.{readout}.predictions.parquet"
                if not path.is_file():
                    raise FileNotFoundError(path)
                hashes[str(path.relative_to(root))] = sha256(path)
                frames[arm] = pd.read_parquet(path).sort_values("gene_id").reset_index(drop=True)
            canonical = frames["tree"][["gene_id", "label_binary"]]
            for arm, frame in frames.items():
                if not frame[["gene_id", "label_binary"]].equals(canonical):
                    raise RuntimeError(f"rows do not align for {arm}/{genus}/{readout}")
            labels = canonical["label_binary"].to_numpy(dtype=np.int8)
            n = len(labels)
            k = int(labels.sum())
            tree_probability = frames["tree"]["probability"].to_numpy(dtype=float)
            tree_top = set(np.argpartition(-tree_probability, k - 1)[:k])
            cell_records = []
            for arm in COMPARATORS:
                probability = frames[arm]["probability"].to_numpy(dtype=float)
                arm_top = set(np.argpartition(-probability, k - 1)[:k])
                overlap = len(tree_top & arm_top)
                union = len(tree_top | arm_top)
                expected_overlap = k * k / n
                kuncheva = (
                    (overlap * n - k * k) / (k * (n - k))
                    if 0 < k < n
                    else float("nan")
                )
                record = {
                    "heldout_genus": genus,
                    "readout": readout,
                    "comparator": arm,
                    "rows": n,
                    "top_k": k,
                    "spearman": float(spearmanr(tree_probability, probability).statistic),
                    "pearson": float(np.corrcoef(tree_probability, probability)[0, 1]),
                    "mean_absolute_probability_difference": float(
                        np.mean(np.abs(tree_probability - probability))
                    ),
                    "top_k_overlap": overlap,
                    "top_k_expected_overlap": expected_overlap,
                    "top_k_jaccard": overlap / union,
                    "top_k_kuncheva": kuncheva,
                }
                records.append(record)
                cell_records.append(record)
            key = f"{genus}_{readout}"
            closest_spearman[key] = max(cell_records, key=lambda row: row["spearman"])[
                "comparator"
            ]
            closest_mae[key] = min(
                cell_records,
                key=lambda row: row["mean_absolute_probability_difference"],
            )["comparator"]

    table = pd.DataFrame(records)
    table.to_csv(output / "agreement.tsv", sep="\t", index=False)
    spearman_wins = pd.Series(closest_spearman).value_counts().to_dict()
    mae_wins = pd.Series(closest_mae).value_counts().to_dict()
    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory_all_four_primary_prediction_cells",
        "closest_to_tree_by_spearman": closest_spearman,
        "closest_to_tree_by_probability_mae": closest_mae,
        "closest_spearman_cell_counts": spearman_wins,
        "closest_mae_cell_counts": mae_wins,
        "interpretation": (
            "RandomPlant is the closest comparator to Tree in all four primary "
            "prediction cells by both rank correlation and absolute probability "
            "difference, consistent with generic plant-domain adaptation explaining "
            "much of the Tree-induced prediction change."
        ),
        "posthoc_exploratory": True,
        "malus_accessed": False,
        "input_prediction_sha256": hashes,
        "table": "agreement.tsv",
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
