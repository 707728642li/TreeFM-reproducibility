#!/usr/bin/env python3
"""Summarize all prespecified functional cells and endpoints without selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
CONTROLS = ("herb", "random_plant", "phylogc_match")
METRICS = {
    "auprc": True,
    "auroc": True,
    "top_k_enrichment": True,
    "ece_15bin": False,
}
INDEX = ("heldout_genus", "training_genus", "readout", "population")


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
    output = root / "results/metrics/publication_v3_rebuild_functional_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    report = root / "reports/PUBLICATION_V3_REBUILD_FUNCTIONAL_BENCHMARK_20260801_CN.md"

    blocks = []
    hashes = {}
    for arm in ARMS:
        path = source / arm / "seed_23" / "metrics.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        block = pd.read_csv(path, sep="\t")
        if len(block) != 16 or set(block["arm"].astype(str)) != {arm}:
            raise RuntimeError(f"unexpected functional cells in {path}")
        blocks.append(block)
        hashes[str(path.relative_to(root))] = sha256(path)
    metrics = pd.concat(blocks, ignore_index=True)

    cell_frames = []
    aggregate_records = []
    for metric, higher_is_better in METRICS.items():
        wide = metrics.pivot(index=list(INDEX), columns="arm", values=metric).reset_index()
        if len(wide) != 16 or wide[list(ARMS)].isna().any().any():
            raise RuntimeError(f"expected 16 complete cells for {metric}")
        values = wide[list(ARMS)]
        ranks = values.rank(axis=1, ascending=not higher_is_better, method="average")
        optimum = values.max(axis=1) if higher_is_better else values.min(axis=1)
        best_control = (
            wide[list(CONTROLS)].max(axis=1)
            if higher_is_better
            else wide[list(CONTROLS)].min(axis=1)
        )
        wide["metric"] = metric
        wide["higher_is_better"] = higher_is_better
        wide["tree_rank"] = ranks["tree"]
        wide["tree_is_inclusive_winner"] = np.isclose(wide["tree"], optimum)
        if higher_is_better:
            wide["tree_advantage_over_best_matched_control"] = wide["tree"] - best_control
            wide["tree_advantage_over_base"] = wide["tree"] - wide["base"]
        else:
            wide["tree_advantage_over_best_matched_control"] = best_control - wide["tree"]
            wide["tree_advantage_over_base"] = wide["base"] - wide["tree"]
        wide["tree_better_than_all_matched_controls"] = wide[
            "tree_advantage_over_best_matched_control"
        ].gt(1e-12)
        wide["tree_tied_best_matched_control"] = np.isclose(
            wide["tree_advantage_over_best_matched_control"], 0.0
        )
        cell_frames.append(wide)

        for arm in ARMS:
            aggregate_records.append(
                {
                    "metric": metric,
                    "arm": arm,
                    "mean_rank": float(ranks[arm].mean()),
                    "inclusive_wins": int(np.isclose(wide[arm], optimum).sum()),
                    "cells": len(wide),
                }
            )
        aggregate_records.append(
            {
                "metric": metric,
                "arm": "tree_vs_best_matched_control",
                "mean_rank": float(wide["tree_rank"].mean()),
                "inclusive_wins": int(wide["tree_better_than_all_matched_controls"].sum()),
                "ties": int(wide["tree_tied_best_matched_control"].sum()),
                "cells": len(wide),
                "mean_advantage": float(
                    wide["tree_advantage_over_best_matched_control"].mean()
                ),
            }
        )

    cells = pd.concat(cell_frames, ignore_index=True)
    aggregate = pd.DataFrame(aggregate_records)
    cells.to_csv(output / "all_prespecified_cells.tsv", sep="\t", index=False)
    aggregate.to_csv(output / "endpoint_rank_summary.tsv", sep="\t", index=False)

    auprc = cells.loc[cells["metric"].eq("auprc")].copy()
    win_counts = {
        arm: int(np.isclose(auprc[arm], auprc[list(ARMS)].max(axis=1)).sum())
        for arm in ARMS
    }
    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all_prespecified_functional_populations_and_endpoints",
        "cells_per_endpoint": 16,
        "auprc_inclusive_win_counts": win_counts,
        "tree_auprc_positive_vs_base_cells": int(
            auprc["tree_advantage_over_base"].gt(0).sum()
        ),
        "tree_auprc_positive_vs_best_matched_control_cells": int(
            auprc["tree_better_than_all_matched_controls"].sum()
        ),
        "tree_auprc_mean_rank": float(auprc["tree_rank"].mean()),
        "interpretation": (
            "Tree DAPT improves over Base in selected cells but never exceeds the "
            "best matched DAPT control on prespecified AUPRC cells. Secondary retrieval "
            "and calibration behavior is heterogeneous and is not a substitute for the "
            "failed primary ranking endpoint."
        ),
        "malus_accessed": False,
        "input_metrics_sha256": hashes,
        "tables": {
            "cells": "all_prespecified_cells.tsv",
            "ranks": "endpoint_rank_summary.tsv",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    def metric_line(metric: str) -> str:
        subset = aggregate.loc[
            (aggregate["metric"].eq(metric))
            & (aggregate["arm"].eq("tree_vs_best_matched_control"))
        ].iloc[0]
        return (
            f"| {metric} | {int(subset.inclusive_wins)}/16 | "
            f"{int(subset.get('ties', 0))}/16 | {subset.mean_advantage:.6f} |"
        )

    report.write_text(
        "\n".join(
            [
                "# Publication-v3 功能基准全量汇总",
                "",
                "## 不选择子集的结果",
                "",
                "全部 16 个预设功能单元（2 个属 × 2 个 readout × 4 个人群）均纳入。AUPRC 获胜次数为："
                f"Base {win_counts['base']}、Tree {win_counts['tree']}、Herb {win_counts['herb']}、"
                f"RandomPlant {win_counts['random_plant']}、PhyloGCMatch {win_counts['phylogc_match']}。",
                "",
                f"Tree 相对 Base 的 AUPRC 在 {payload['tree_auprc_positive_vs_base_cells']}/16 个单元为正，"
                f"但相对最强匹配对照为 {payload['tree_auprc_positive_vs_best_matched_control_cells']}/16。"
                "这将一般域适配收益与 Tree 特异收益明确区分开。",
                "",
                "## Tree 相对最强匹配对照",
                "",
                "正优势按指标方向统一定义为 Tree 更好；ECE 越低越好。",
                "",
                "| Endpoint | Tree 严格更好 | 并列 | 平均优势 |",
                "|---|---:|---:|---:|",
                *(metric_line(metric) for metric in METRICS),
                "",
                "top-k 富集和校准存在部分有利单元，但方向并不一致，不能替代失败的主 AUPRC 端点。",
                "",
                "Malus 未访问；所有结果均来自冻结 Prunus/Pyrus 分析。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
