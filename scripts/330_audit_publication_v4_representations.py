#!/usr/bin/env python3
"""Independently audit the publication-v4 representation analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ARM_ORDER = ["base", "tree", "herb", "random_plant", "phylogc_match"]
CONTROL_ARMS = ["herb", "random_plant", "phylogc_match"]
COMPARISONS = [(arm, "base") for arm in ARM_ORDER[1:]] + [("tree", arm) for arm in CONTROL_ARMS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_seed(arm: str) -> int:
    return 0 if arm == "base" else 23


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    metric = root / "results/metrics/publication_v4_representation"
    summary = json.loads((metric / "summary.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []

    def check(name: str, condition: bool, observed: object, expected: object) -> None:
        records.append({"name": name, "passed": bool(condition), "observed": observed, "expected": expected})

    check("status", summary.get("status") == "complete", summary.get("status"), "complete")
    check("analysis_tier", summary.get("analysis_tier") == "posthoc_seed23_descriptive", summary.get("analysis_tier"), "posthoc_seed23_descriptive")
    check("decision_authority", summary.get("decision_authority") is False, summary.get("decision_authority"), False)
    check("wsl_gpu0_only", summary.get("device") == "cuda" and summary.get("cuda_visible_devices") == "0", {"device": summary.get("device"), "visible": summary.get("cuda_visible_devices")}, {"device": "cuda", "visible": "0"})
    check("arms", summary.get("arms") == ARM_ORDER, summary.get("arms"), ARM_ORDER)
    check("scope_shape", summary.get("species") == 26 and summary.get("dimensions") == 384 and len(summary.get("tasks", [])) == 4, {key: summary.get(key) for key in ("species", "dimensions", "tasks")}, {"species": 26, "dimensions": 384, "tasks": 4})
    check("malus_sealed", summary.get("malus_outcomes_accessed") is False, summary.get("malus_outcomes_accessed"), False)

    selected = pd.read_csv(metric / "selected_rows.tsv", sep="\t")
    group_counts = selected.groupby(["slug", "task", "label"]).size()
    expected_rows = 26 * 4 * 2 * 128
    check("sample_rows", len(selected) == expected_rows and selected[["slug", "task", "label"]].drop_duplicates().shape[0] == 26 * 4 * 2, {"rows": len(selected), "strata": len(group_counts)}, {"rows": expected_rows, "strata": 208})
    check("sample_balance", bool((group_counts == 128).all()), group_counts.value_counts().to_dict(), {128: 208})
    check("sample_unique", not selected.duplicated(["slug", "benchmark_row_index"]).any(), int(selected.duplicated(["slug", "benchmark_row_index"]).sum()), 0)
    check("sample_hash", sha256(metric / "selected_rows.tsv") == summary.get("selected_rows_sha256"), sha256(metric / "selected_rows.tsv"), summary.get("selected_rows_sha256"))

    # Independently prove that every retained row belongs to the smallest 128
    # contract hashes in its full species/task/label stratum.
    selection_failures: list[str] = []
    for slug in selected["slug"].drop_duplicates():
        benchmark = pd.read_parquet(
            root / f"data/processed/technical_benchmarks_publication_v3_26/{slug}.parquet",
            columns=["pair_id", "task", "label", "benchmark_row_index"],
        )
        for (task, label), stratum in benchmark.groupby(["task", "label"], sort=True):
            hashes = [
                hashlib.sha256(
                    f"publication-v4|{slug}|{task}|{int(label)}|{pair_id}|{int(row_index)}".encode("utf-8")
                ).hexdigest()
                for pair_id, row_index in zip(stratum["pair_id"], stratum["benchmark_row_index"])
            ]
            expected = set(
                stratum.assign(selection_sha256=hashes)
                .sort_values(["selection_sha256", "benchmark_row_index"], kind="mergesort")
                .head(128)["benchmark_row_index"]
                .astype(int)
            )
            observed = set(
                selected.loc[
                    (selected["slug"] == slug)
                    & (selected["task"] == task)
                    & (selected["label"] == label),
                    "benchmark_row_index",
                ].astype(int)
            )
            if expected != observed:
                selection_failures.append(f"{slug}|{task}|{label}")
    check("deterministic_selection_rebuilt", not selection_failures, selection_failures, [])

    cka = pd.read_csv(metric / "linear_cka.tsv", sep="\t")
    expected_cka_rows = (1 + 26 + 4) * 25
    check("cka_rows", len(cka) == expected_cka_rows, len(cka), expected_cka_rows)
    check("cka_finite_range", np.isfinite(cka["linear_cka"]).all() and (cka["linear_cka"] >= 0).all() and (cka["linear_cka"] <= 1.0001).all(), [float(cka["linear_cka"].min()), float(cka["linear_cka"].max())], [0.0, 1.0001])
    symmetry_failures = []
    for (scope_type, scope), part in cka.groupby(["scope_type", "scope"]):
        matrix = part.pivot(index="arm_a", columns="arm_b", values="linear_cka").loc[ARM_ORDER, ARM_ORDER]
        if not np.allclose(matrix.to_numpy(), matrix.to_numpy().T, atol=2e-6) or not np.allclose(np.diag(matrix), 1.0, atol=2e-5):
            symmetry_failures.append(f"{scope_type}|{scope}")
    check("cka_symmetry_diagonal", not symmetry_failures, symmetry_failures, [])
    global_matrix = cka.loc[cka["scope_type"] == "global"].pivot(index="arm_a", columns="arm_b", values="linear_cka")
    nearest_cka = max(CONTROL_ARMS, key=lambda arm: float(global_matrix.loc["tree", arm]))
    check("nearest_control_cka", nearest_cka == summary.get("nearest_control_to_tree_by_global_cka"), nearest_cka, summary.get("nearest_control_to_tree_by_global_cka"))

    displacement = pd.read_parquet(metric / "paired_displacements.parquet")
    expected_displacement_rows = len(COMPARISONS) * expected_rows
    check("displacement_rows", len(displacement) == expected_displacement_rows, len(displacement), expected_displacement_rows)
    observed_comparisons = set(zip(displacement["comparison_arm"], displacement["reference_arm"]))
    check("displacement_comparisons", observed_comparisons == set(COMPARISONS), sorted(observed_comparisons), sorted(COMPARISONS))
    check("displacement_finite", np.isfinite(displacement[["cosine_distance", "relative_l2_displacement"]]).all().all() and (displacement[["cosine_distance", "relative_l2_displacement"]] >= 0).all().all(), displacement[["cosine_distance", "relative_l2_displacement"]].min().to_dict(), ">=0 finite")

    # Numerically reproduce direct distances for a deterministic 128-row subset.
    check_rows = selected.iloc[np.linspace(0, len(selected) - 1, 128, dtype=int)]
    numerical_failures = []
    for arm_a, arm_b in COMPARISONS:
        observed = displacement.loc[
            (displacement["comparison_arm"] == arm_a)
            & (displacement["reference_arm"] == arm_b)
        ].set_index(["slug", "benchmark_row_index"])
        for slug, part in check_rows.groupby("slug"):
            indices = part["benchmark_row_index"].to_numpy(dtype=int)
            a = np.asarray(np.load(root / f"results/embeddings/plantcad_dapt_publication_v3/{arm_a}/seed_{arm_seed(arm_a)}/{slug}.npy", mmap_mode="r")[indices], dtype=np.float32)
            b = np.asarray(np.load(root / f"results/embeddings/plantcad_dapt_publication_v3/{arm_b}/seed_{arm_seed(arm_b)}/{slug}.npy", mmap_mode="r")[indices], dtype=np.float32)
            cosine = 1.0 - np.sum(a * b, axis=1) / np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12, None)
            rel_l2 = np.linalg.norm(a - b, axis=1) / np.clip(np.linalg.norm(b, axis=1), 1e-12, None)
            for row_index, expected_cosine, expected_l2 in zip(indices, cosine, rel_l2):
                row = observed.loc[(slug, row_index)]
                if not np.isclose(float(row["cosine_distance"]), float(expected_cosine), atol=2e-6) or not np.isclose(float(row["relative_l2_displacement"]), float(expected_l2), atol=2e-6):
                    numerical_failures.append(f"{arm_a}|{arm_b}|{slug}|{row_index}")
    check("paired_distance_numeric_rebuild", not numerical_failures, numerical_failures[:20], [])

    displacement_summary = pd.read_csv(metric / "displacement_summary.tsv", sep="\t")
    tree_control = displacement_summary.loc[
        (displacement_summary["scope_type"] == "global")
        & (displacement_summary["comparison_arm"] == "tree")
        & (displacement_summary["reference_arm"].isin(CONTROL_ARMS))
        & (displacement_summary["metric"] == "cosine_distance")
    ]
    nearest_cosine = str(tree_control.sort_values("median").iloc[0]["reference_arm"])
    check("nearest_control_cosine", nearest_cosine == summary.get("nearest_control_to_tree_by_median_cosine_distance"), nearest_cosine, summary.get("nearest_control_to_tree_by_median_cosine_distance"))
    check("nearest_control_concordance", summary.get("nearest_control_concordant") == (nearest_cka == nearest_cosine), {"cka": nearest_cka, "cosine": nearest_cosine}, summary.get("nearest_control_concordant"))

    bootstrap = pd.read_parquet(metric / "displacement_bootstrap.parquet")
    expected_bootstrap_rows = len(COMPARISONS) * 2 * 2000
    check("bootstrap_rows", len(bootstrap) == expected_bootstrap_rows and bootstrap["replicate"].nunique() == 2000, {"rows": len(bootstrap), "replicates": bootstrap["replicate"].nunique()}, {"rows": expected_bootstrap_rows, "replicates": 2000})

    procrustes = pd.read_csv(metric / "procrustes_residuals.tsv", sep="\t")
    check("procrustes_rows", len(procrustes) == len(COMPARISONS) * 5 and np.isfinite(procrustes["procrustes_residual"]).all(), len(procrustes), len(COMPARISONS) * 5)

    pca = pd.read_parquet(metric / "pca_coordinates.parquet")
    pca_counts = pca.groupby("arm").size().to_dict()
    check("pca_rows", pca_counts == {arm: 2000 for arm in ARM_ORDER}, pca_counts, {arm: 2000 for arm in ARM_ORDER})
    identity_counts = pca.groupby(["slug", "task", "label", "pair_id", "benchmark_row_index"])['arm'].nunique()
    check("pca_common_rows", len(identity_counts) == 2000 and (identity_counts == 5).all(), {"identities": len(identity_counts), "arms_per_identity": identity_counts.value_counts().to_dict()}, {"identities": 2000, "arms_per_identity": {5: 2000}})

    output_failures = []
    for rel, expected in summary.get("output_sha256", {}).items():
        path = root / rel
        if not path.exists() or sha256(path) != expected:
            output_failures.append(rel)
    check("output_fingerprints", not output_failures and len(summary.get("output_sha256", {})) == 8, output_failures, "eight matching output hashes")

    manifest_failures = []
    for arm, expected in summary.get("manifest_sha256", {}).items():
        path = root / f"results/embeddings/plantcad_dapt_publication_v3/{arm}/seed_{arm_seed(arm)}/manifest.tsv"
        if sha256(path) != expected:
            manifest_failures.append(arm)
    check("manifest_fingerprints", not manifest_failures, manifest_failures, [])

    failures = [record for record in records if not record["passed"]]
    audit = {
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "publication_v4_seed23_representation",
        "checks": len(records),
        "passed_checks": len(records) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "records": records,
        "analysis_tier": "posthoc_seed23_descriptive",
        "decision_authority": False,
        "malus_outcomes_accessed": False,
    }
    output = root / "results/metrics/publication_v4_representation_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
