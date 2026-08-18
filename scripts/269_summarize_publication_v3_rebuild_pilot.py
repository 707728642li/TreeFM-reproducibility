#!/usr/bin/env python3
"""Summarize the prospectively defined seed-23 non-Malus rebuild pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
MATCHED_CONTROLS = ("herb", "random_plant", "phylogc_match")
PRIMARY_TECHNICAL_SLUGS = (
    "hevea_brasiliensis",
    "prunus_persica",
    "pyrus_pyrifolia",
)
TASKS = ("tis", "tts", "splice_donor", "splice_acceptor")
READOUTS = ("linear", "xgboost")
SEED = 23


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metrics(root: Path, track: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if track == "technical":
        base = root / "results/metrics/plantcad_dapt_publication_v3_probes"
    else:
        base = root / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
    blocks = []
    hashes: dict[str, str] = {}
    for arm in ARMS:
        path = base / arm / f"seed_{SEED}" / "metrics.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        block = pd.read_csv(path, sep="\t")
        if set(block["arm"].astype(str)) != {arm}:
            raise RuntimeError(f"arm mismatch in {path}")
        blocks.append(block)
        hashes[str(path.relative_to(root))] = sha256(path)
    return pd.concat(blocks, ignore_index=True), hashes


def add_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["strongest_matched_control_auprc"] = result[
        list(MATCHED_CONTROLS)
    ].max(axis=1)
    result["strongest_matched_control"] = result[
        list(MATCHED_CONTROLS)
    ].idxmax(axis=1)
    result["tree_minus_base"] = result["tree"] - result["base"]
    result["tree_minus_strongest_matched_control"] = (
        result["tree"] - result["strongest_matched_control_auprc"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "results/metrics/publication_v3_rebuild_pilot_summary"
    output.mkdir(parents=True, exist_ok=True)

    technical, technical_hashes = load_metrics(root, "technical")
    technical = technical.loc[
        technical["scope"].eq("species")
        & technical["slug"].isin(PRIMARY_TECHNICAL_SLUGS)
        & technical["task"].isin(TASKS)
        & technical["readout"].isin(READOUTS)
        & technical["family_transfer_class"].eq("logo_novel_family")
        & technical["identity_population"].eq("all")
    ].copy()
    technical_wide = technical.pivot(
        index=["slug", "task", "readout"], columns="arm", values="auprc"
    ).reset_index()
    if len(technical_wide) != 24 or technical_wide[list(ARMS)].isna().any().any():
        raise RuntimeError(
            f"expected 24 complete primary technical cells, observed {len(technical_wide)}"
        )
    technical_wide = add_contrasts(technical_wide)
    technical_wide.to_csv(
        output / "technical_primary_cell_effects.tsv", sep="\t", index=False
    )
    technical_tasks = (
        technical_wide.groupby("task", sort=True)[
            ["tree_minus_base", "tree_minus_strongest_matched_control"]
        ]
        .agg(["mean", "min", "max"])
    )
    technical_tasks.columns = ["_".join(column) for column in technical_tasks]
    technical_tasks = technical_tasks.reset_index()
    technical_tasks["positive_vs_matched_control"] = technical_tasks[
        "tree_minus_strongest_matched_control_mean"
    ].gt(0.0)
    technical_tasks["material_reversal"] = technical_tasks[
        "tree_minus_strongest_matched_control_mean"
    ].lt(-0.01)
    technical_tasks.to_csv(
        output / "technical_task_summary.tsv", sep="\t", index=False
    )

    functional, functional_hashes = load_metrics(root, "functional")
    functional = functional.loc[
        functional["population"].eq("all")
        & functional["heldout_genus"].isin(("prunus", "pyrus"))
        & functional["readout"].isin(READOUTS)
    ].copy()
    functional_wide = functional.pivot(
        index=["heldout_genus", "training_genus", "readout"],
        columns="arm",
        values="auprc",
    ).reset_index()
    if len(functional_wide) != 4 or functional_wide[list(ARMS)].isna().any().any():
        raise RuntimeError(
            f"expected four complete primary functional cells, observed {len(functional_wide)}"
        )
    functional_wide = add_contrasts(functional_wide)
    functional_wide.to_csv(
        output / "functional_primary_cell_effects.tsv", sep="\t", index=False
    )

    positive_technical_tasks = int(
        technical_tasks["positive_vs_matched_control"].sum()
    )
    technical_material_reversals = int(
        technical_tasks["material_reversal"].sum()
    )
    maximum_functional_gain = float(
        functional_wide["tree_minus_strongest_matched_control"].max()
    )
    functional_cells_ge_002 = int(
        functional_wide["tree_minus_strongest_matched_control"].ge(0.02).sum()
    )
    continue_full = bool(
        maximum_functional_gain >= 0.02
        and positive_technical_tasks >= 2
        and technical_material_reversals < 2
    )
    payload = {
        "status": "complete",
        "scope": "exploratory_seed23_nonmalus_rebuild_pilot",
        "seed": SEED,
        "decision": (
            "continue_full_multiseed" if continue_full else "stop_and_reassess"
        ),
        "continuation_rule": {
            "functional_threshold": (
                "at least one primary heldout-genus/readout Tree-minus-best-"
                "matched-control AUPRC >= 0.02"
            ),
            "technical_positive_threshold": (
                "mean primary NovelFamily Tree-minus-best-matched-control "
                "AUPRC > 0 in at least two of four tasks"
            ),
            "systematic_reversal_definition": (
                "material reversal is task-mean gain < -0.01; fewer than two "
                "material reversals required"
            ),
        },
        "observed": {
            "maximum_functional_woody_control_gain": maximum_functional_gain,
            "functional_primary_cells_ge_0_02": functional_cells_ge_002,
            "positive_technical_tasks": positive_technical_tasks,
            "technical_material_reversals": technical_material_reversals,
        },
        "malus_accessed": False,
        "input_sha256": {**technical_hashes, **functional_hashes},
        "tables": {
            "technical_cells": "technical_primary_cell_effects.tsv",
            "technical_tasks": "technical_task_summary.tsv",
            "functional_cells": "functional_primary_cell_effects.tsv",
        },
    }
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
