#!/usr/bin/env python3
"""Integrate fixed-G-box model dependence with Pyrus H3K4me3 change."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
CONTROLS = ("base", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)
READOUTS = ("linear", "xgboost")
CONTRACT = Path(
    "docs/"
    "publication_v3_exploratory_gbox_model_chromatin_integration_plan.md"
)
FREEZE = Path(
    "config/publication_v3_gbox_model_chromatin_integration_freeze.json"
)
ATTRIBUTION_ROOT = Path(
    "results/biological_cases/publication_v3_gbox_model_attribution"
)
CHROMATIN_ROOT = Path(
    "results/biological_cases/"
    "publication_v3_exploratory_gbox_chromatin"
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
        raise RuntimeError("model-chromatin integration is not frozen")
    if freeze.get("malus_accessed") is not False:
        raise RuntimeError("integration freeze does not preserve Malus")
    for relative, expected in freeze.get("artifact_sha256", {}).items():
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if sha256(artifact) != expected:
            raise RuntimeError(
                "model-chromatin integration artifact changed: "
                f"{relative}"
            )
    return freeze


def standardized(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def covariate_matrix(frame: pd.DataFrame) -> np.ndarray:
    gc = frame["gc_fraction"].to_numpy(dtype=np.float64)
    direction = frame["category"].eq("positive_up").to_numpy(dtype=float)
    return np.column_stack(
        [
            np.ones(len(frame)),
            standardized(
                frame["baseline_log2_mean_cpm"].to_numpy(dtype=float)
            ),
            standardized(gc),
            standardized(gc**2),
            standardized(
                np.log1p(
                    frame["fixed_motif_family_count"].to_numpy(
                        dtype=float
                    )
                )
            ),
            standardized(
                frame["motif_CACGTG_count"].to_numpy(dtype=float)
            ),
            direction,
        ]
    )


def residualize(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average").astype(np.float64)
    coefficients = np.linalg.pinv(matrix) @ ranked
    return ranked - matrix @ coefficients


def partial_spearman(
    frame: pd.DataFrame,
    model_values: np.ndarray | None = None,
    chromatin_column: str = "oriented_nonduplicate_change",
) -> float:
    if model_values is None:
        model_values = frame["gbox_dependence"].to_numpy(dtype=float)
    chromatin = frame[chromatin_column].to_numpy(dtype=float)
    finite = np.isfinite(model_values) & np.isfinite(chromatin)
    work = frame.loc[finite].reset_index(drop=True)
    model_values = np.asarray(model_values, dtype=float)[finite]
    chromatin = chromatin[finite]
    if len(work) < 20:
        raise RuntimeError("too few genes for model-chromatin integration")
    matrix = covariate_matrix(work)
    model_residual = residualize(model_values, matrix)
    chromatin_residual = residualize(chromatin, matrix)
    if (
        model_residual.std(ddof=0) == 0
        or chromatin_residual.std(ddof=0) == 0
    ):
        return float("nan")
    return float(np.corrcoef(model_residual, chromatin_residual)[0, 1])


def chromosome_bootstrap(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
    chromatin_column: str,
) -> np.ndarray:
    chromosomes = sorted(frame["chromosome"].astype(str).unique())
    blocks = {
        chromosome: frame[
            frame["chromosome"].astype(str).eq(chromosome)
        ]
        for chromosome in chromosomes
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    attempts = 0
    while len(estimates) < replicates and attempts < replicates * 20:
        attempts += 1
        sampled = rng.choice(
            chromosomes, size=len(chromosomes), replace=True
        )
        boot = pd.concat(
            [blocks[str(chromosome)] for chromosome in sampled],
            ignore_index=True,
        )
        estimate = partial_spearman(
            boot, chromatin_column=chromatin_column
        )
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) != replicates:
        raise RuntimeError("chromosome bootstrap did not converge")
    return np.asarray(estimates, dtype=np.float64)


def stratified_permutation(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
    chromatin_column: str,
) -> np.ndarray:
    values = frame["gbox_dependence"].to_numpy(dtype=np.float64)
    strata = [
        group.index.to_numpy()
        for _, group in frame.groupby("permutation_stratum", sort=True)
    ]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        permuted = values.copy()
        for indices in strata:
            permuted[indices] = rng.permutation(permuted[indices])
        estimates[replicate] = partial_spearman(
            frame,
            model_values=permuted,
            chromatin_column=chromatin_column,
        )
    return estimates


def paired_correlation_bootstrap(
    merged: pd.DataFrame,
    replicates: int,
    seed: int,
) -> np.ndarray:
    chromosomes = sorted(merged["chromosome"].astype(str).unique())
    blocks = {
        chromosome: merged[
            merged["chromosome"].astype(str).eq(chromosome)
        ]
        for chromosome in chromosomes
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    attempts = 0
    while len(estimates) < replicates and attempts < replicates * 20:
        attempts += 1
        sampled = rng.choice(
            chromosomes, size=len(chromosomes), replace=True
        )
        boot = pd.concat(
            [blocks[str(chromosome)] for chromosome in sampled],
            ignore_index=True,
        )
        tree = partial_spearman(
            boot,
            model_values=boot["tree_gbox_dependence"].to_numpy(
                dtype=float
            ),
        )
        control = partial_spearman(
            boot,
            model_values=boot["control_gbox_dependence"].to_numpy(
                dtype=float
            ),
        )
        difference = tree - control
        if np.isfinite(difference):
            estimates.append(difference)
    if len(estimates) != replicates:
        raise RuntimeError("paired correlation bootstrap did not converge")
    return np.asarray(estimates, dtype=np.float64)


def empirical_two_sided(null: np.ndarray, observed: float) -> float:
    return float(
        (1 + np.count_nonzero(np.abs(null) >= abs(observed)))
        / (1 + len(null))
    )


def bootstrap_sign_p(draws: np.ndarray) -> float:
    return float(
        min(
            1.0,
            2
            * min(
                (1 + np.count_nonzero(draws <= 0)) / (1 + len(draws)),
                (1 + np.count_nonzero(draws >= 0)) / (1 + len(draws)),
            ),
        )
    )


def orient_chromatin(frame: pd.DataFrame, column: str) -> np.ndarray:
    categories = frame["category"].astype(str)
    positive_up = categories.eq("positive_up")
    positive_down = categories.eq("positive_down")
    positive_count = int(positive_up.sum() + positive_down.sum())
    if not positive_count:
        raise RuntimeError("chromatin table lacks positive genes")
    mixture = (
        int(positive_up.sum()) - int(positive_down.sum())
    ) / positive_count
    direction = np.full(len(frame), mixture, dtype=np.float64)
    direction[positive_up.to_numpy()] = 1.0
    direction[positive_down.to_numpy()] = -1.0
    return direction * frame[column].to_numpy(dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--cell-bootstraps", type=int, default=10000)
    args = parser.parse_args()
    root = args.project_root.resolve()
    freeze = validate_freeze(root)
    chromatin_path = (
        root / CHROMATIN_ROOT / "gbox_chromatin_gene_table.tsv.gz"
    )
    chromatin_summary = root / CHROMATIN_ROOT / "summary.json"
    for path in (chromatin_path, chromatin_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary_payload = json.loads(
        chromatin_summary.read_text(encoding="utf-8")
    )
    if summary_payload.get("status") != "complete":
        raise RuntimeError("Pyrus G-box chromatin analysis is incomplete")

    chromatin = pd.read_csv(chromatin_path, sep="\t")
    required_chromatin = {
        "gene_id",
        "chromosome",
        "category",
        "permutation_stratum",
        "baseline_log2_mean_cpm",
        "gc_fraction",
        "fixed_motif_family_count",
        "motif_CACGTG_count",
        "nonduplicate_score_change_day50_minus_day0",
        "with_duplicates_score_change_day50_minus_day0",
    }
    missing = sorted(required_chromatin - set(chromatin.columns))
    if missing:
        raise RuntimeError(f"chromatin table lacks columns: {missing}")
    if chromatin["gene_id"].duplicated().any():
        raise RuntimeError("chromatin table contains duplicate genes")
    chromatin["oriented_nonduplicate_change"] = orient_chromatin(
        chromatin, "nonduplicate_score_change_day50_minus_day0"
    )
    chromatin["oriented_duplicate_change"] = orient_chromatin(
        chromatin, "with_duplicates_score_change_day50_minus_day0"
    )

    input_paths = [chromatin_path, chromatin_summary, root / CONTRACT]
    effect_frames: list[pd.DataFrame] = []
    for arm in ARMS:
        for seed in SEEDS:
            path = (
                root
                / ATTRIBUTION_ROOT
                / arm
                / f"seed_{seed}"
                / "gene_effects.parquet"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            input_paths.append(path)
            frame = pd.read_parquet(path)
            frame = frame[
                frame["heldout_genus"].eq("pyrus")
            ].copy()
            if len(frame) == 0:
                raise RuntimeError(f"no Pyrus attribution rows: {path}")
            effect_frames.append(frame)
    effects = pd.concat(effect_frames, ignore_index=True)
    expected_groups = len(ARMS) * len(SEEDS) * len(READOUTS)
    if effects.groupby(["arm", "seed", "readout"]).ngroups != expected_groups:
        raise RuntimeError("incomplete attribution arm/seed/readout groups")

    group_frames: dict[tuple[str, int, str], pd.DataFrame] = {}
    summary_records: list[dict[str, object]] = []
    for keys, group in effects.groupby(
        ["arm", "seed", "readout"], sort=True
    ):
        arm, seed, readout = keys
        merged = group.merge(
            chromatin, on=["gene_id", "chromosome"], validate="one_to_one"
        )
        if not merged["gbox_count"].eq(
            merged["motif_CACGTG_count"]
        ).all():
            raise RuntimeError(
                f"exact G-box counts differ for {arm}/{seed}/{readout}"
            )
        positive = merged[
            merged["category"].isin(["positive_up", "positive_down"])
        ].reset_index(drop=True)
        negative = merged[
            merged["category"].eq("negative")
        ].reset_index(drop=True)
        if len(positive) < 100 or len(negative) < 20:
            raise RuntimeError(
                f"insufficient integration population: {keys}"
            )
        observed = partial_spearman(positive)
        bootstrap = chromosome_bootstrap(
            positive,
            args.bootstraps,
            stable_seed("gbox_model_chromatin_bootstrap_v1", *keys),
            "oriented_nonduplicate_change",
        )
        permutation = stratified_permutation(
            positive,
            args.permutations,
            stable_seed("gbox_model_chromatin_permutation_v1", *keys),
            "oriented_nonduplicate_change",
        )
        duplicate_observed = partial_spearman(
            positive, chromatin_column="oriented_duplicate_change"
        )
        negative_reference = partial_spearman(negative)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        summary_records.append(
            {
                "arm": arm,
                "seed": seed,
                "readout": readout,
                "positive_genes": len(positive),
                "negative_genes": len(negative),
                "primary_partial_spearman": observed,
                "primary_ci_low": float(lower),
                "primary_ci_high": float(upper),
                "primary_permutation_p_two_sided": empirical_two_sided(
                    permutation, observed
                ),
                "duplicate_retaining_partial_spearman": (
                    duplicate_observed
                ),
                "negative_reference_partial_spearman": (
                    negative_reference
                ),
            }
        )
        group_frames[(arm, int(seed), readout)] = positive
        print(
            f"{arm} seed={seed} {readout}: n={len(positive)} "
            f"partial_rho={observed:.6g}",
            flush=True,
        )

    cell_records: list[dict[str, object]] = []
    aggregate_records: list[dict[str, object]] = []
    for control in CONTROLS:
        control_cells: list[float] = []
        for seed in SEEDS:
            for readout in READOUTS:
                tree = group_frames[("tree", seed, readout)]
                other = group_frames[(control, seed, readout)]
                merge_columns = [
                    "gene_id",
                    "chromosome",
                    "category",
                    "permutation_stratum",
                    "baseline_log2_mean_cpm",
                    "gc_fraction",
                    "fixed_motif_family_count",
                    "motif_CACGTG_count",
                    "oriented_nonduplicate_change",
                    "gbox_dependence",
                ]
                paired = tree[merge_columns].merge(
                    other[["gene_id", "gbox_dependence"]],
                    on="gene_id",
                    suffixes=("_tree", "_control"),
                    validate="one_to_one",
                )
                paired = paired.rename(
                    columns={
                        "gbox_dependence_tree": "tree_gbox_dependence",
                        "gbox_dependence_control": (
                            "control_gbox_dependence"
                        ),
                    }
                )
                tree_rho = partial_spearman(
                    paired,
                    model_values=paired[
                        "tree_gbox_dependence"
                    ].to_numpy(dtype=float),
                )
                control_rho = partial_spearman(
                    paired,
                    model_values=paired[
                        "control_gbox_dependence"
                    ].to_numpy(dtype=float),
                )
                difference = tree_rho - control_rho
                draws = paired_correlation_bootstrap(
                    paired,
                    args.bootstraps,
                    stable_seed(
                        "gbox_model_chromatin_paired_bootstrap_v1",
                        control,
                        seed,
                        readout,
                    ),
                )
                lower, upper = np.quantile(draws, [0.025, 0.975])
                cell_records.append(
                    {
                        "control_arm": control,
                        "seed": seed,
                        "readout": readout,
                        "genes": len(paired),
                        "tree_partial_spearman": tree_rho,
                        "control_partial_spearman": control_rho,
                        "tree_minus_control": difference,
                        "ci_low": float(lower),
                        "ci_high": float(upper),
                        "bootstrap_p_two_sided": bootstrap_sign_p(draws),
                    }
                )
                control_cells.append(difference)
        values = np.asarray(control_cells, dtype=np.float64)
        rng = np.random.default_rng(
            stable_seed(
                "gbox_model_chromatin_cell_bootstrap_v1", control
            )
        )
        draws = np.empty(args.cell_bootstraps, dtype=np.float64)
        for repeat in range(args.cell_bootstraps):
            selected = rng.integers(0, len(values), size=len(values))
            draws[repeat] = values[selected].mean()
        observed = float(values.mean())
        lower, upper = np.quantile(draws, [0.025, 0.975])
        aggregate_records.append(
            {
                "contrast": f"tree_minus_{control}",
                "paired_seed_readout_cells": len(values),
                "mean_correlation_difference": observed,
                "ci_low": float(lower),
                "ci_high": float(upper),
                "bootstrap_p_two_sided": bootstrap_sign_p(draws),
                "positive_cell_fraction": float(np.mean(values > 0)),
                "minimum_cell_difference": float(values.min()),
            }
        )

    output_root = (
        root
        / "results/biological_cases/"
        "publication_v3_gbox_model_chromatin_integration"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_table = pd.DataFrame(summary_records)
    cell_table = pd.DataFrame(cell_records)
    aggregate_table = pd.DataFrame(aggregate_records)
    summary_path = output_root / "arm_seed_readout_associations.tsv"
    cell_path = output_root / "tree_control_cells.tsv"
    aggregate_path = output_root / "tree_control_aggregate.tsv"
    summary_table.to_csv(summary_path, sep="\t", index=False)
    cell_table.to_csv(cell_path, sep="\t", index=False)
    aggregate_table.to_csv(aggregate_path, sep="\t", index=False)

    input_sha = {
        str(path.relative_to(root)): sha256(path) for path in input_paths
    }
    fingerprint_payload = {
        "freeze_input_fingerprint": freeze["input_fingerprint"],
        "input_sha256": input_sha,
        "bootstraps": args.bootstraps,
        "permutations": args.permutations,
        "cell_bootstraps": args.cell_bootstraps,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "status": "complete",
        **fingerprint_payload,
        "input_fingerprint": fingerprint,
        "association_rows": len(summary_table),
        "tree_control_cell_rows": len(cell_table),
        "aggregate_contrasts": aggregate_table.to_dict(
            orient="records"
        ),
        "association_sha256": sha256(summary_path),
        "tree_control_cells_sha256": sha256(cell_path),
        "tree_control_aggregate_sha256": sha256(aggregate_path),
        "confirmatory_endpoints_changed": False,
        "malus_accessed": False,
    }
    partial = output_root / "summary.json.partial"
    partial.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(partial, output_root / "summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
