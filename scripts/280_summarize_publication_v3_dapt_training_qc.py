#!/usr/bin/env python3
"""Summarize publication-v3 DAPT optimization dynamics as QC-only artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARMS = ("tree", "herb", "random_plant", "phylogc_match")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_state(run_dir: Path) -> Path | None:
    final_state = run_dir / "trainer_state.json"
    if final_state.is_file():
        return final_state
    checkpoints = []
    for path in run_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            step = int(path.parent.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(None, None))[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--expected-steps", type=int, default=15000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / "results"
    )
    metric_dir = output_root / "metrics/publication_v3_rebuild_pilot_training_qc"
    figure_dir = output_root / "figures"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    arms_summary: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    incomplete: list[str] = []
    for arm in ARMS:
        run_dir = (
            root
            / "results/models/plantcad_dapt_publication_v3"
            / arm
            / f"seed_{args.seed}"
        )
        state_path = resolve_state(run_dir)
        if state_path is None:
            missing.append(arm)
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        global_step = int(state.get("global_step", 0))
        if global_step != args.expected_steps:
            incomplete.append(arm)
        for entry in state.get("log_history", []):
            if "loss" not in entry and "eval_loss" not in entry:
                continue
            rows.append(
                {
                    "arm": arm,
                    "seed": args.seed,
                    "step": int(entry["step"]),
                    "epoch": entry.get("epoch"),
                    "record_type": "eval" if "eval_loss" in entry else "train",
                    "loss": entry.get("eval_loss", entry.get("loss")),
                    "learning_rate": entry.get("learning_rate"),
                    "grad_norm": entry.get("grad_norm"),
                }
            )
        arm_rows = [row for row in rows if row["arm"] == arm]
        eval_rows = [row for row in arm_rows if row["record_type"] == "eval"]
        train_rows = [row for row in arm_rows if row["record_type"] == "train"]
        final_model = run_dir / "final/pytorch_model.bin"
        arms_summary[arm] = {
            "state_path": str(state_path.relative_to(root)),
            "state_sha256": sha256(state_path),
            "global_step": global_step,
            "train_records": len(train_rows),
            "eval_records": len(eval_rows),
            "minimum_eval_loss": (
                float(min(row["loss"] for row in eval_rows)) if eval_rows else None
            ),
            "minimum_eval_step": (
                int(min(eval_rows, key=lambda row: row["loss"])["step"])
                if eval_rows
                else None
            ),
            "final_eval_loss": (
                float(max(eval_rows, key=lambda row: row["step"])["loss"])
                if eval_rows
                else None
            ),
            "final_model_present": final_model.is_file(),
            "final_model_sha256": sha256(final_model) if final_model.is_file() else None,
        }

    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"missing trainer states for arms: {missing}")
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(
            f"arms did not reach exactly {args.expected_steps} steps: {incomplete}"
        )
    if not rows:
        raise RuntimeError("no train/eval history records found")

    table = pd.DataFrame(rows).sort_values(["arm", "step", "record_type"])
    if not np.isfinite(table["loss"].astype(float)).all():
        raise RuntimeError("non-finite training/evaluation loss detected")
    table_path = metric_dir / "history.tsv"
    table.to_csv(table_path, sep="\t", index=False)

    colors = {
        "tree": "#2b8c6b",
        "herb": "#9c6ade",
        "random_plant": "#de8f05",
        "phylogc_match": "#377eb8",
    }
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
    for arm in ARMS:
        subset = table[table["arm"] == arm]
        train = subset[subset["record_type"] == "train"].copy()
        evaluate = subset[subset["record_type"] == "eval"]
        if not train.empty:
            train["loss_smooth"] = train["loss"].rolling(5, min_periods=1).mean()
            axes[0].plot(
                train["step"], train["loss_smooth"], label=arm,
                color=colors[arm], linewidth=1.6,
            )
            axes[2].plot(
                train["step"], train["learning_rate"], label=arm,
                color=colors[arm], linewidth=1.4,
            )
        if not evaluate.empty:
            axes[1].plot(
                evaluate["step"], evaluate["loss"], marker="o", label=arm,
                color=colors[arm], linewidth=1.6, markersize=3.5,
            )
    axes[0].set_ylabel("Train loss\n(5-point mean)")
    axes[1].set_ylabel("Validation loss")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Optimizer step")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("Publication-v3 DAPT optimization QC (not downstream evidence)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure_path = figure_dir / "publication_v3_rebuild_pilot_training_qc.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    status = "pass" if not missing and not incomplete else "incomplete_qa"
    payload = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "optimization_QC_only_not_downstream_evidence",
        "seed": args.seed,
        "expected_steps": args.expected_steps,
        "missing_arms": missing,
        "incomplete_arms": incomplete,
        "arms": arms_summary,
        "history_tsv": str(table_path),
        "figure_png": str(figure_path),
    }
    summary_path = metric_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
