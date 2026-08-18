#!/usr/bin/env python3
"""Smoke-test the seed-23 technical figure and manuscript finalizer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd


def load_finalizer(path: Path):
    spec = importlib.util.spec_from_file_location("publication_v3_finalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    module = load_finalizer(
        root / "scripts/272_write_publication_v3_rebuild_pilot_report_cn.py"
    )
    temporary_root = root / "results/tmp_publication_v3_finalizer_smoke"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    try:
        (temporary_root / "reports").mkdir(parents=True)
        shutil.copy2(
            root
            / "reports/PUBLICATION_V3_REBUILD_INTEGRATED_MANUSCRIPT_SKELETON_EN.md",
            temporary_root
            / "reports/PUBLICATION_V3_REBUILD_INTEGRATED_MANUSCRIPT_SKELETON_EN.md",
        )
        technical_rows = []
        bootstrap_rows = []
        cell_index = 0
        for slug in module.SPECIES_ORDER:
            for task in module.TASK_ORDER:
                for readout in module.READOUT_ORDER:
                    point = -0.012 + cell_index * 0.001
                    technical_rows.append(
                        {
                            "slug": slug,
                            "task": task,
                            "readout": readout,
                            "tree_minus_base": point + 0.004,
                            "tree_minus_strongest_matched_control": point,
                        }
                    )
                    bootstrap_rows.append(
                        {
                            "seed": 23,
                            "slug": slug,
                            "task": task,
                            "readout": readout,
                            "family_transfer_class": "logo_novel_family",
                            "identity_population": "all",
                            "delta_vs_base": point + 0.004,
                            "delta_vs_base_ci_low": point - 0.001,
                            "delta_vs_base_ci_high": point + 0.009,
                            "woody_control_gain": point,
                            "woody_control_gain_ci_low": point - 0.005,
                            "woody_control_gain_ci_high": point + 0.005,
                        }
                    )
                    cell_index += 1
        technical_cells = pd.DataFrame(technical_rows)
        bootstrap = pd.DataFrame(bootstrap_rows)
        aligned = module.validate_technical_bootstrap(
            technical_cells, bootstrap
        )
        summary, hashes = module.write_technical_figure(
            temporary_root, aligned
        )
        tasks = pd.DataFrame(
            {
                "task": list(module.TASK_ORDER),
                "tree_minus_strongest_matched_control_mean": [-0.01, -0.005, 0.002, 0.006],
                "tree_minus_strongest_matched_control_min": [-0.02, -0.01, -0.004, 0.001],
                "tree_minus_strongest_matched_control_max": [0.001, 0.002, 0.008, 0.012],
                "positive_vs_matched_control": [False, False, True, True],
                "material_reversal": [False, False, False, False],
            }
        )
        module.write_technical_manuscript_artifacts(
            temporary_root, tasks, aligned, summary
        )
        manuscript = (
            temporary_root
            / "reports/PUBLICATION_V3_REBUILD_INTEGRATED_MANUSCRIPT_SKELETON_EN.md"
        ).read_text(encoding="utf-8")
        required = {
            "aligned_cells": len(aligned) == 24,
            "three_figure_formats": len(hashes) == 3,
            "abstract_filled": "<!-- TECHNICAL_SENTENCE_START -->" in manuscript,
            "section_filled": "<!-- TECHNICAL_SECTION_START -->" in manuscript,
            "placeholder_removed": "TECHNICAL-PANEL SENTENCE TO BE INSERTED" not in manuscript,
            "pending_removed": "Pending frozen outputs" not in manuscript,
        }
        if not all(required.values()):
            raise RuntimeError(f"finalizer smoke checks failed: {required}")
        print(json.dumps({"status": "pass", "checks": required}, indent=2))
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


if __name__ == "__main__":
    main()
