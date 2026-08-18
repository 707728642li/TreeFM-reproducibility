#!/usr/bin/env python3
"""Independent audit for publication-v4 technical and leakage-QC figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


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
    metric = root / "results/metrics/publication_v4_technical_qc"
    summary = json.loads((metric / "summary.json").read_text(encoding="utf-8"))
    scope96 = pd.read_csv(metric / "technical_arm_vs_base_all96.tsv", sep="\t")
    tree24 = pd.read_csv(metric / "technical_tree_contrasts_all24.tsv", sep="\t")
    arm_summary = pd.read_csv(metric / "technical_arm_summary.tsv", sep="\t")
    leakage = pd.read_csv(metric / "sequence_leakage_all28.tsv", sep="\t")
    frozen96 = pd.read_csv(root / "results/metrics/publication_v3_technical_arm_posthoc/arm_vs_base_scope_effects.tsv", sep="\t")
    frozen24 = pd.read_csv(root / "results/metrics/plantcad_dapt_publication_v3_seed23_comparison/bootstrap_scope_effects.tsv", sep="\t")
    frozen_leakage = pd.read_csv(root / "metadata/publication_v3_sequence_leakage_summary.tsv", sep="\t")

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("summary_complete", summary.get("status") == "complete", summary.get("status"))
    check("no_decision_authority", summary.get("scientific_decision_authority") is False and not scope96["decision_authority"].astype(bool).any(), summary.get("analysis_tier"))
    check("malus_not_accessed", summary.get("malus_accessed") is False, summary.get("malus_accessed"))
    check("complete_96_contrasts", len(scope96) == 96 and scope96.groupby(["readout", "task", "slug"])["arm"].nunique().eq(4).all(), len(scope96))
    check("four_dapt_arms", set(scope96["arm"]) == {"tree", "herb", "random_plant", "phylogc_match"}, scope96["arm"].value_counts().to_dict())
    check("complete_24_tree_cells", len(tree24) == 24 and not tree24.duplicated(["readout", "task", "slug"]).any(), len(tree24))
    check("bootstrap_intervals_ordered", bool((tree24["delta_vs_base_ci_low"] <= tree24["delta_vs_base"]).all() and (tree24["delta_vs_base"] <= tree24["delta_vs_base_ci_high"]).all() and (tree24["woody_control_gain_ci_low"] <= tree24["woody_control_gain"]).all() and (tree24["woody_control_gain"] <= tree24["woody_control_gain_ci_high"]).all()), "all 48 intervals")
    check("tree_positive_counts", int((tree24["delta_vs_base"] > 0).sum()) == 16 == summary["tree_vs_base_positive_cells"], int((tree24["delta_vs_base"] > 0).sum()))
    check("matched_positive_counts", int((tree24["woody_control_gain"] > 0).sum()) == 6 == summary["tree_vs_strongest_control_positive_cells"], int((tree24["woody_control_gain"] > 0).sum()))
    check("matched_ci_positive_zero", int((tree24["woody_control_gain_ci_low"] > 0).sum()) == 0 == summary["tree_vs_strongest_control_ci_positive"], int((tree24["woody_control_gain_ci_low"] > 0).sum()))
    check("matched_ci_negative_five", int((tree24["woody_control_gain_ci_high"] < 0).sum()) == 5 == summary["tree_vs_strongest_control_ci_negative"], int((tree24["woody_control_gain_ci_high"] < 0).sum()))
    check("arm_summary_four", len(arm_summary) == 4 and set(arm_summary["arm"]) == set(scope96["arm"]), arm_summary.to_dict(orient="records"))

    sort96 = ["readout", "task", "slug", "arm"]
    obs96 = scope96.sort_values(sort96).reset_index(drop=True)
    src96 = frozen96.sort_values(sort96).reset_index(drop=True)
    check("technical_values_frozen", np.allclose(obs96[["arm_minus_base", "ci_low", "ci_high", "two_sided_q_bh_96"]], src96[["arm_minus_base", "ci_low", "ci_high", "two_sided_q_bh_96"]], rtol=0, atol=1e-12), "all 384 values")
    sort24 = ["readout", "task", "slug"]
    obs24 = tree24.sort_values(sort24).reset_index(drop=True)
    src24 = frozen24.sort_values(sort24).reset_index(drop=True)
    cols24 = ["delta_vs_base", "delta_vs_base_ci_low", "delta_vs_base_ci_high", "woody_control_gain", "woody_control_gain_ci_low", "woody_control_gain_ci_high"]
    check("tree_values_frozen", np.allclose(obs24[cols24], src24[cols24], rtol=0, atol=1e-12), "all 144 values")

    check("all_28_leakage_strata", len(leakage) == 28 and not leakage.duplicated(["slug", "task"]).any(), leakage.groupby("task").size().to_dict())
    check("zero_exact_matches", int(leakage["exact_rows"].sum()) == 0 and int(leakage["exact_pairs"].sum()) == 0, [int(leakage["exact_rows"].sum()), int(leakage["exact_pairs"].sum())])
    check("nested_near_identity_counts", bool((leakage["near_0_95_pairs"] <= leakage["near_0_90_pairs"]).all() and (leakage["near_0_90_pairs"] <= leakage["pairs"]).all()), "all strata")
    check("censoring_bounded", bool((leakage["identity_search_censored_rows"] <= leakage["rows"]).all()), float(leakage["identity_search_censored_rows"].sum() / leakage["rows"].sum()))
    source_leak = frozen_leakage.sort_values(["slug", "task"]).reset_index(drop=True)
    observed_leak = leakage.sort_values(["slug", "task"]).reset_index(drop=True)
    source_cols = frozen_leakage.columns.tolist()
    check("leakage_counts_frozen", observed_leak[source_cols].equals(source_leak[source_cols]), "all source columns")
    check("near90_fraction_recomputed", np.isclose(summary["near90_pair_fraction"], leakage["near_0_90_pairs"].sum() / leakage["pairs"].sum()), summary["near90_pair_fraction"])

    figures = [root / f"results/figures/publication_v4_technical_effects.{ext}" for ext in ["png", "pdf", "svg"]] + [root / f"results/figures/publication_v4_sequence_leakage_qc.{ext}" for ext in ["png", "pdf", "svg"]]
    check("all_figure_formats_exist", all(p.exists() and p.stat().st_size > 10_000 for p in figures), {p.name: p.stat().st_size if p.exists() else None for p in figures})
    hash_ok = True
    for path in figures:
        key = str(path.relative_to(root)).replace("\\", "/")
        hash_ok &= key in summary["outputs"] and summary["outputs"][key]["sha256"] == sha256(path)
    check("figure_hashes_match_metadata", hash_ok, len(figures))
    forbidden = [key for key in summary.get("inputs", {}) if "malus" in key.lower()]
    check("no_malus_input_paths", not forbidden, forbidden)

    passed = sum(bool(x["passed"]) for x in checks)
    audit = {"status": "pass" if passed == len(checks) else "fail", "created_utc": datetime.now(timezone.utc).isoformat(), "checks_passed": passed, "checks_total": len(checks), "checks": checks}
    out = root / "results/metrics/publication_v4_technical_qc_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["status", "checks_passed", "checks_total"]}, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
