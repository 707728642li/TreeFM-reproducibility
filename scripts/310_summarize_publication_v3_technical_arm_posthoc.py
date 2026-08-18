#!/usr/bin/env python3
"""Post-hoc arm-versus-Base summary from the frozen technical bootstrap draws.

This analysis has no continuation, model-selection, or primary-claim authority.
It reuses the paired bootstrap samples produced by script 118 and therefore
does not rerun or redefine the frozen Tree attribution analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("tree", "herb", "random_plant", "phylogc_match")
TASKS = ("tis", "tts", "splice_donor", "splice_acceptor")
READOUTS = ("linear", "xgboost")
SLUGS = ("hevea_brasiliensis", "prunus_persica", "pyrus_pyrifolia")
ARM_LABELS = {
    "tree": "Tree",
    "herb": "Herb",
    "random_plant": "RandomPlant",
    "phylogc_match": "PhyloGCMatch",
}
TASK_LABELS = {
    "tis": "TIS",
    "tts": "TTS",
    "splice_donor": "Splice donor",
    "splice_acceptor": "Splice acceptor",
}
SCOPE_KEYS = (
    "seed",
    "readout",
    "task",
    "slug",
    "family_transfer_class",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    if np.isnan(array).any() or ((array < 0) | (array > 1)).any():
        raise ValueError("P values must be finite and lie in [0, 1]")
    order = np.argsort(array, kind="stable")
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    return pd.Series(restored, index=values.index)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def validate_inputs(points: pd.DataFrame, draws: pd.DataFrame, replicates: int) -> None:
    required_points = set(SCOPE_KEYS) | {
        "base_auprc",
        *(f"{arm}_auprc" for arm in ARMS),
    }
    required_draws = required_points | {"replicate"}
    missing_points = required_points - set(points.columns)
    missing_draws = required_draws - set(draws.columns)
    if missing_points:
        raise RuntimeError(f"point table missing columns: {sorted(missing_points)}")
    if missing_draws:
        raise RuntimeError(f"bootstrap table missing columns: {sorted(missing_draws)}")
    expected_scopes = len(TASKS) * len(READOUTS) * len(SLUGS)
    if len(points) != expected_scopes:
        raise RuntimeError(
            f"expected {expected_scopes} NovelFamily point scopes, observed {len(points)}"
        )
    if points.duplicated(list(SCOPE_KEYS)).any():
        raise RuntimeError("point table contains duplicate technical scopes")
    observed = draws.groupby(list(SCOPE_KEYS), sort=False)["replicate"].nunique()
    if len(observed) != expected_scopes or not observed.eq(replicates).all():
        raise RuntimeError(
            f"expected {replicates} unique draws in each of {expected_scopes} scopes"
        )
    if set(points["seed"].astype(int)) != {23}:
        raise RuntimeError("post-hoc summary is restricted to stopped-pilot seed 23")
    if set(points["task"]) != set(TASKS):
        raise RuntimeError("technical task set is incomplete")
    if set(points["readout"]) != set(READOUTS):
        raise RuntimeError("technical readout set is incomplete")
    if set(points["slug"]) != set(SLUGS):
        raise RuntimeError("technical species set is incomplete")
    if set(points["family_transfer_class"]) != {"logo_novel_family"}:
        raise RuntimeError("unexpected family-transfer population")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--comparison-subdir",
        default="plantcad_dapt_publication_v3_seed23_comparison",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    comparison = root / "results/metrics" / args.comparison_subdir
    point_path = comparison / "bootstrap_scope_effects.tsv"
    draw_path = comparison / "paired_bootstrap_effects.parquet"
    for path in (point_path, draw_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    points = pd.read_csv(point_path, sep="\t")
    draws = pd.read_parquet(draw_path)
    validate_inputs(points, draws, args.bootstrap_replicates)

    records: list[dict[str, object]] = []
    for point in points.sort_values(list(SCOPE_KEYS)).itertuples(index=False):
        mask = np.ones(len(draws), dtype=bool)
        for key in SCOPE_KEYS:
            mask &= draws[key].eq(getattr(point, key)).to_numpy()
        scope_draws = draws.loc[mask]
        if len(scope_draws) != args.bootstrap_replicates:
            raise RuntimeError(f"bootstrap alignment failed for {point}")
        for arm in ARMS:
            distribution = (
                scope_draws[f"{arm}_auprc"].to_numpy(dtype=float)
                - scope_draws["base_auprc"].to_numpy(dtype=float)
            )
            lower, upper = np.quantile(distribution, [0.025, 0.975])
            p_positive = (1 + int(np.count_nonzero(distribution <= 0))) / (
                1 + len(distribution)
            )
            p_negative = (1 + int(np.count_nonzero(distribution >= 0))) / (
                1 + len(distribution)
            )
            records.append(
                {
                    **{key: getattr(point, key) for key in SCOPE_KEYS},
                    "arm": arm,
                    "arm_auprc": float(getattr(point, f"{arm}_auprc")),
                    "base_auprc": float(point.base_auprc),
                    "arm_minus_base": float(
                        getattr(point, f"{arm}_auprc") - point.base_auprc
                    ),
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                    "one_sided_p_positive": float(p_positive),
                    "two_sided_p": float(min(1.0, 2 * min(p_positive, p_negative))),
                    "ci_direction": (
                        "positive" if lower > 0 else "negative" if upper < 0 else "overlap"
                    ),
                }
            )

    scopes = pd.DataFrame(records)
    scopes["two_sided_q_bh_96"] = benjamini_hochberg(scopes["two_sided_p"])
    scopes["analysis_tier"] = "posthoc_seed23_descriptive"
    scopes["decision_authority"] = False

    output = root / "results/metrics/publication_v3_technical_arm_posthoc"
    output.mkdir(parents=True, exist_ok=True)
    scope_path = output / "arm_vs_base_scope_effects.tsv"
    scopes.to_csv(scope_path, sep="\t", index=False)

    task_summary = (
        scopes.groupby(["arm", "task"], sort=True)
        .agg(
            cells=("arm_minus_base", "size"),
            positive_cells=("arm_minus_base", lambda values: int((values > 0).sum())),
            mean_effect=("arm_minus_base", "mean"),
            min_effect=("arm_minus_base", "min"),
            max_effect=("arm_minus_base", "max"),
            ci_positive_cells=("ci_direction", lambda values: int((values == "positive").sum())),
            ci_negative_cells=("ci_direction", lambda values: int((values == "negative").sum())),
            bh_significant_cells=("two_sided_q_bh_96", lambda values: int((values <= 0.05).sum())),
        )
        .reset_index()
    )
    task_path = output / "arm_vs_base_task_summary.tsv"
    task_summary.to_csv(task_path, sep="\t", index=False)

    arm_summary = (
        scopes.groupby("arm", sort=True)
        .agg(
            cells=("arm_minus_base", "size"),
            positive_cells=("arm_minus_base", lambda values: int((values > 0).sum())),
            mean_effect=("arm_minus_base", "mean"),
            min_effect=("arm_minus_base", "min"),
            max_effect=("arm_minus_base", "max"),
            ci_positive_cells=("ci_direction", lambda values: int((values == "positive").sum())),
            ci_negative_cells=("ci_direction", lambda values: int((values == "negative").sum())),
            bh_significant_cells=("two_sided_q_bh_96", lambda values: int((values <= 0.05).sum())),
        )
        .reset_index()
    )
    arm_path = output / "arm_vs_base_summary.tsv"
    arm_summary.to_csv(arm_path, sep="\t", index=False)

    arm_rows: list[list[object]] = []
    for arm in ARMS:
        row = arm_summary.loc[arm_summary["arm"].eq(arm)].iloc[0]
        arm_rows.append(
            [
                ARM_LABELS[arm],
                f"{int(row.positive_cells)}/{int(row.cells)}",
                f"{float(row.mean_effect):+.6f}",
                f"{float(row.min_effect):+.6f}",
                f"{float(row.max_effect):+.6f}",
                int(row.ci_positive_cells),
                int(row.ci_negative_cells),
                int(row.bh_significant_cells),
            ]
        )
    task_rows: list[list[object]] = []
    for arm in ARMS:
        for task in TASKS:
            row = task_summary.loc[
                task_summary["arm"].eq(arm) & task_summary["task"].eq(task)
            ].iloc[0]
            task_rows.append(
                [
                    ARM_LABELS[arm],
                    TASK_LABELS[task],
                    f"{int(row.positive_cells)}/{int(row.cells)}",
                    f"{float(row.mean_effect):+.6f}",
                    f"{float(row.min_effect):+.6f}",
                    f"{float(row.max_effect):+.6f}",
                    int(row.ci_positive_cells),
                    int(row.ci_negative_cells),
                    int(row.bh_significant_cells),
                ]
            )
    report = f"""# Post-hoc technical arm-versus-Base heterogeneity

**Status:** complete descriptive seed-23 post-hoc analysis; no continuation, model-selection or primary-claim authority.

The frozen 2,000-replicate positive-negative pair-block bootstrap was reused without changing the prespecified Tree attribution analysis. Each DAPT arm was contrasted with Base in the same 24 NovelFamily species-by-task-by-readout cells. Cell-wise two-sided P values were adjusted by Benjamini-Hochberg across all 96 post-hoc contrasts. Means below are unweighted descriptive averages across cells; the CI columns count cell-specific intervals and are not confidence intervals for the displayed means.

## Arm-level summary

{markdown_table(
    ["Arm", "Positive cells", "Mean effect", "Minimum", "Maximum", "CI > 0", "CI < 0", "BH q <= 0.05"],
    arm_rows,
)}

## Task-by-arm summary

{markdown_table(
    ["Arm", "Task", "Positive cells", "Mean effect", "Minimum", "Maximum", "CI > 0", "CI < 0", "BH q <= 0.05"],
    task_rows,
)}

## Interpretation boundary

These results can describe whether generic DAPT effects vary by corpus, task or readout. They cannot rescue the failed prospective functional necessary condition, establish cross-seed stability, authorize seeds 41/59, change the frozen Tree-versus-strongest-control estimand or support a Malus/pan-woody claim.
"""
    report_path = root / "reports/PUBLICATION_V3_TECHNICAL_ARM_POSTHOC_EN.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "status": "complete",
        "analysis_tier": "posthoc_seed23_descriptive",
        "decision_authority": False,
        "primary_tree_attribution_unchanged": True,
        "seed": 23,
        "family_transfer_class": "logo_novel_family",
        "scope_contrasts": int(len(scopes)),
        "scope_count": int(len(points)),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "positive-negative pair block",
        "multiplicity": "Benjamini-Hochberg across all 96 post-hoc arm-by-scope contrasts",
        "malus_accessed": False,
        "input_sha256": {
            str(point_path.relative_to(root)): sha256(point_path),
            str(draw_path.relative_to(root)): sha256(draw_path),
        },
        "outputs": {
            "scope_effects": str(scope_path.relative_to(root)),
            "task_summary": str(task_path.relative_to(root)),
            "arm_summary": str(arm_path.relative_to(root)),
            "report": str(report_path.relative_to(root)),
        },
        "output_sha256": {
            str(scope_path.relative_to(root)): sha256(scope_path),
            str(task_path.relative_to(root)): sha256(task_path),
            str(arm_path.relative_to(root)): sha256(arm_path),
            str(report_path.relative_to(root)): sha256(report_path),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
