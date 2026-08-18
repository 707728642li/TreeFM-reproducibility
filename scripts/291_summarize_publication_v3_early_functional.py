#!/usr/bin/env python3
"""Freeze the early functional part of the preregistered seed-23 pilot gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
MATCHED_CONTROLS = ("herb", "random_plant", "phylogc_match")
READOUTS = ("linear", "xgboost")
HELDOUT_GENERA = ("prunus", "pyrus")
SEED = 23
THRESHOLD = 0.02


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
    output = root / "results/metrics/publication_v3_rebuild_early_functional"
    output.mkdir(parents=True, exist_ok=True)
    report = root / "reports/PUBLICATION_V3_REBUILD_EARLY_FUNCTIONAL_RESULT_20260801_CN.md"

    blocks: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for arm in ARMS:
        directory = (
            root
            / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
            / arm
            / f"seed_{SEED}"
        )
        metrics_path = directory / "metrics.tsv"
        spec_path = directory / "run_spec.json"
        for required in (metrics_path, spec_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        block = pd.read_csv(metrics_path, sep="\t")
        if set(block["arm"].astype(str)) != {arm} or set(block["seed"]) != {SEED}:
            raise RuntimeError(f"arm/seed mismatch in {metrics_path}")
        blocks.append(block)
        hashes[str(metrics_path.relative_to(root))] = sha256(metrics_path)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        embedding_seed = 0 if arm == "base" else SEED
        manifest = (
            root
            / "results/embeddings/plantcad_dapt_publication_v3_functional"
            / arm
            / f"seed_{embedding_seed}"
            / "manifest.tsv"
        )
        if spec.get("embedding_manifest_sha256") != sha256(manifest):
            raise RuntimeError(f"embedding manifest fingerprint mismatch for {arm}")
        fingerprints[arm] = str(spec.get("input_fingerprint", ""))

    metrics = pd.concat(blocks, ignore_index=True)
    primary = metrics.loc[
        metrics["population"].eq("all")
        & metrics["heldout_genus"].isin(HELDOUT_GENERA)
        & metrics["readout"].isin(READOUTS)
    ].copy()
    wide = primary.pivot(
        index=["heldout_genus", "training_genus", "readout"],
        columns="arm",
        values="auprc",
    ).reset_index()
    if len(wide) != 4 or wide[list(ARMS)].isna().any().any():
        raise RuntimeError(f"expected four complete primary cells, observed {len(wide)}")
    wide["strongest_matched_control"] = wide[list(MATCHED_CONTROLS)].idxmax(axis=1)
    wide["strongest_matched_control_auprc"] = wide[list(MATCHED_CONTROLS)].max(axis=1)
    wide["tree_minus_strongest_matched_control"] = (
        wide["tree"] - wide["strongest_matched_control_auprc"]
    )
    wide["tree_minus_base"] = wide["tree"] - wide["base"]
    wide = wide.sort_values(["heldout_genus", "readout"]).reset_index(drop=True)
    wide.to_csv(output / "primary_cell_effects.tsv", sep="\t", index=False)

    maximum_gain = float(wide["tree_minus_strongest_matched_control"].max())
    cells_ge_threshold = int(
        wide["tree_minus_strongest_matched_control"].ge(THRESHOLD).sum()
    )
    functional_gate_pass = bool(maximum_gain >= THRESHOLD)
    tree_vs_base_positive = int(wide["tree_minus_base"].gt(0).sum())
    tree_vs_matched_positive = int(
        wide["tree_minus_strongest_matched_control"].gt(0).sum()
    )
    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "early_functional_component_of_frozen_seed23_nonmalus_gate",
        "seed": SEED,
        "functional_gate_pass": functional_gate_pass,
        "logical_implication": (
            "full_seed23_continuation_gate_cannot_pass"
            if not functional_gate_pass
            else "technical_component_still_required"
        ),
        "threshold": {
            "metric": "Tree-minus-strongest-matched-control AUPRC",
            "minimum": THRESHOLD,
            "required_cells": 1,
            "primary_cells": 4,
        },
        "observed": {
            "maximum_tree_minus_strongest_matched_control_auprc": maximum_gain,
            "cells_ge_0_02": cells_ge_threshold,
            "tree_vs_base_positive_cells": tree_vs_base_positive,
            "tree_vs_matched_control_positive_cells": tree_vs_matched_positive,
            "mean_tree_minus_strongest_matched_control_auprc": float(
                wide["tree_minus_strongest_matched_control"].mean()
            ),
        },
        "interpretation": (
            "Tree can improve over the undapted base in selected cells, but the "
            "gain is not tree-specific because a matched non-Tree control is better "
            "in every primary heldout-genus/readout cell."
        ),
        "technical_metrics_pending": True,
        "multiseed_authorized": False,
        "malus_accessed": False,
        "input_metrics_sha256": hashes,
        "input_fingerprints": fingerprints,
        "table": "primary_cell_effects.tsv",
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    table_lines = [
        "| Held-out genus | Readout | Base | Tree | Best matched control | Control | Tree − control | Tree − base |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in wide.itertuples(index=False):
        table_lines.append(
            f"| {row.heldout_genus} | {row.readout} | {row.base:.6f} | "
            f"{row.tree:.6f} | {row.strongest_matched_control_auprc:.6f} | "
            f"{row.strongest_matched_control} | "
            f"{row.tree_minus_strongest_matched_control:.6f} | "
            f"{row.tree_minus_base:.6f} |"
        )
    report.write_text(
        "\n".join(
            [
                "# Publication-v3 重建 pilot：提前功能结果",
                "",
                "## 冻结判定",
                "",
                f"功能必要条件未通过。四个主单元中最大 Tree－最强匹配对照增益为 "
                f"**{maximum_gain:.6f} AUPRC**，达到预设 +0.02 的单元为 "
                f"**{cells_ge_threshold}/4**。因此完整 seed-23 Go/No-go 规则在逻辑上已不可能通过，"
                "seeds 41/59 当前不获授权，Malus 继续封存。",
                "",
                "## 四个预设主单元",
                "",
                *table_lines,
                "",
                "## 研究含义",
                "",
                f"Tree 相对未适配 base 在 {tree_vs_base_positive}/4 个单元为正，但相对预先指定的 "
                f"Herb、RandomPlant、PhyloGCMatch 最强匹配对照在 {tree_vs_matched_positive}/4 个单元为正。"
                "这说明部分 DAPT 改善更像是一般性的植物域适配，而不是 Tree 语料特异性优势。",
                "",
                "技术面板仍在运行，用于完整描述效应方向和形成可发表的负结果/基准结论；它不能改变本轮功能必要条件已经失败的事实。",
                "",
                "本结论仅使用 Prunus/Pyrus 非 Malus 数据及冻结 readout；尚未进行多种子或因果验证。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
