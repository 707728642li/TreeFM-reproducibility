#!/usr/bin/env python3
"""Aggregate fixed-G-box attribution across arms, seeds and held-out genera."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
CONTROLS = ("base", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)
GENERA = ("prunus", "pyrus")
READOUTS = ("linear", "xgboost")
FREEZE = Path("config/publication_v3_gbox_model_attribution_freeze.json")
RESULT_ROOT = Path(
    "results/biological_cases/publication_v3_gbox_model_attribution"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validate_freeze(root: Path) -> dict[str, object]:
    path = root / FREEZE
    if not path.is_file():
        raise FileNotFoundError(path)
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        raise RuntimeError("G-box model-attribution freeze is not active")
    if freeze.get("malus_accessed") is not False:
        raise RuntimeError("G-box attribution freeze does not preserve Malus")
    for relative, expected in freeze.get("artifact_sha256", {}).items():
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if sha256(artifact) != expected:
            raise RuntimeError(
                f"G-box attribution artifact changed after freeze: {relative}"
            )
    return freeze


def paired_cell_bootstrap(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise RuntimeError("invalid paired attribution cells")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(values), size=len(values))
        estimates[replicate] = values[sampled].mean()
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    p_two_sided = min(
        1.0,
        2
        * min(
            (np.count_nonzero(estimates <= 0) + 1)
            / (len(estimates) + 1),
            (np.count_nonzero(estimates >= 0) + 1)
            / (len(estimates) + 1),
        ),
    )
    return {
        "mean_tree_minus_control_interaction": float(values.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "bootstrap_p_two_sided": float(p_two_sided),
        "positive_cell_fraction": float(np.mean(values > 0)),
        "minimum_cell_effect": float(values.min()),
        "maximum_cell_effect": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    root = args.project_root.resolve()
    freeze = validate_freeze(root)
    result_root = root / RESULT_ROOT
    input_paths: list[Path] = []
    summaries: list[pd.DataFrame] = []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = result_root / arm / f"seed_{seed}"
            summary_path = run_dir / "summary.tsv"
            effects_path = run_dir / "gene_effects.parquet"
            run_spec_path = run_dir / "run_spec.json"
            for path in (summary_path, effects_path, run_spec_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
                input_paths.append(path)
            frame = pd.read_csv(summary_path, sep="\t")
            expected = {
                (readout, genus)
                for readout in READOUTS
                for genus in GENERA
            }
            observed = set(
                zip(frame["readout"], frame["heldout_genus"])
            )
            if len(frame) != 4 or observed != expected:
                raise RuntimeError(
                    f"incomplete G-box summary: {summary_path}"
                )
            if set(frame["arm"]) != {arm} or set(frame["seed"]) != {
                seed
            }:
                raise RuntimeError(
                    f"G-box summary identifiers differ: {summary_path}"
                )
            summaries.append(frame)
    all_summaries = pd.concat(summaries, ignore_index=True)
    if len(all_summaries) != len(ARMS) * len(SEEDS) * 4:
        raise RuntimeError("unexpected aggregate G-box summary row count")

    key = ["seed", "readout", "heldout_genus"]
    tree = all_summaries[all_summaries["arm"].eq("tree")][
        key + ["positive_minus_negative_interaction"]
    ].rename(
        columns={
            "positive_minus_negative_interaction": "tree_interaction"
        }
    )
    contrast_frames: list[pd.DataFrame] = []
    aggregate_records: list[dict[str, object]] = []
    for control in CONTROLS:
        control_frame = all_summaries[
            all_summaries["arm"].eq(control)
        ][key + ["positive_minus_negative_interaction"]].rename(
            columns={
                "positive_minus_negative_interaction": (
                    "control_interaction"
                )
            }
        )
        cells = tree.merge(
            control_frame, on=key, how="inner", validate="one_to_one"
        )
        if len(cells) != len(SEEDS) * len(GENERA) * len(READOUTS):
            raise RuntimeError(f"incomplete Tree-{control} pairing")
        cells["control_arm"] = control
        cells["tree_minus_control_interaction"] = (
            cells["tree_interaction"] - cells["control_interaction"]
        )
        contrast_frames.append(cells)
        values = cells["tree_minus_control_interaction"].to_numpy(
            dtype=np.float64
        )
        aggregate_records.append(
            {
                "contrast": f"tree_minus_{control}",
                "control_arm": control,
                "paired_cells": len(cells),
                **paired_cell_bootstrap(
                    values,
                    args.bootstrap_replicates,
                    stable_seed(
                        "publication_v3_gbox_tree_control_bootstrap_v1",
                        control,
                    ),
                ),
            }
        )
    contrast_cells = pd.concat(contrast_frames, ignore_index=True)
    aggregate = pd.DataFrame(aggregate_records)

    input_sha = {
        str(path.relative_to(root)): sha256(path) for path in input_paths
    }
    fingerprint_payload = {
        "freeze_input_fingerprint": freeze["input_fingerprint"],
        "input_sha256": input_sha,
        "bootstrap_replicates": args.bootstrap_replicates,
        "aggregation": (
            "paired bootstrap over seed x readout x heldout-genus cells"
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output_dir = result_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary_path = output_dir / "all_arm_summaries.tsv"
    cells_path = output_dir / "tree_control_cells.tsv"
    aggregate_path = output_dir / "tree_control_aggregate.tsv"
    all_summaries.to_csv(all_summary_path, sep="\t", index=False)
    contrast_cells.to_csv(cells_path, sep="\t", index=False)
    aggregate.to_csv(aggregate_path, sep="\t", index=False)

    result = {
        "status": "complete",
        **fingerprint_payload,
        "input_fingerprint": fingerprint,
        "arm_summary_rows": len(all_summaries),
        "paired_contrast_cells": len(contrast_cells),
        "aggregate_contrasts": aggregate.to_dict(orient="records"),
        "all_arm_summaries_sha256": sha256(all_summary_path),
        "tree_control_cells_sha256": sha256(cells_path),
        "tree_control_aggregate_sha256": sha256(aggregate_path),
        "confirmatory_endpoints_changed": False,
        "malus_accessed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
