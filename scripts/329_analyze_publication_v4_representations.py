#!/usr/bin/env python3
"""Quantify seed-23 representation relationships under the frozen v4 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA


ARM_ORDER = ["base", "tree", "herb", "random_plant", "phylogc_match"]
DAPT_ARMS = ARM_ORDER[1:]
CONTROL_ARMS = ["herb", "random_plant", "phylogc_match"]
CONTRACT_SEED = 20260803


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_seed(arm: str) -> int:
    return 0 if arm == "base" else 23


def manifest_path(root: Path, arm: str) -> Path:
    return root / f"results/embeddings/plantcad_dapt_publication_v3/{arm}/seed_{arm_seed(arm)}/manifest.tsv"


def embedding_path(root: Path, arm: str, slug: str) -> Path:
    return root / f"results/embeddings/plantcad_dapt_publication_v3/{arm}/seed_{arm_seed(arm)}/{slug}.npy"


def centered_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = torch.sum(cross * cross)
    denominator = torch.sqrt(torch.sum(xx * xx) * torch.sum(yy * yy))
    return float((numerator / denominator.clamp_min(1e-30)).detach().cpu())


def procrustes_residual(reference: np.ndarray, target: np.ndarray) -> float:
    x = reference.astype(np.float64, copy=True)
    y = target.astype(np.float64, copy=True)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0 or y_norm == 0:
        return float("nan")
    x /= x_norm
    y /= y_norm
    rotation, _ = orthogonal_procrustes(y, x)
    aligned = y @ rotation
    return float(np.linalg.norm(x - aligned))


def hierarchical_bootstrap_median(
    values: np.ndarray,
    species: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = CONTRACT_SEED,
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(species.tolist())))
    by_species = {slug: np.flatnonzero(species == slug) for slug in unique}
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_species = rng.choice(unique, size=len(unique), replace=True)
        chunks = []
        for slug in sampled_species:
            idx = by_species[slug]
            chunks.append(rng.choice(values[idx], size=len(idx), replace=True))
        draws[replicate] = float(np.median(np.concatenate(chunks)))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-per-stratum", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.sample_per_stratum != 128:
        raise ValueError("Final contract requires exactly 128 rows per species/task/label stratum")
    if args.bootstrap_replicates != 2000:
        raise ValueError("Final contract requires exactly 2,000 bootstrap replicates")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != "0":
            raise RuntimeError("WSL representation analysis is restricted to physical GPU0: set CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    manifests: dict[str, pd.DataFrame] = {}
    manifest_hashes: dict[str, str] = {}
    for arm in ARM_ORDER:
        path = manifest_path(root, arm)
        manifests[arm] = pd.read_csv(path, sep="\t")
        manifest_hashes[arm] = sha256(path)
    base = manifests["base"].copy()
    expected_species = base["slug"].tolist()
    if len(expected_species) != 26 or len(set(expected_species)) != 26:
        raise ValueError("Base manifest must contain 26 unique species")
    for arm in ARM_ORDER:
        current = manifests[arm]
        if current["slug"].tolist() != expected_species:
            raise ValueError(f"Species order mismatch for {arm}")
        aligned = base[["slug", "rows", "dimensions", "row_hash"]].merge(
            current[["slug", "rows", "dimensions", "row_hash"]],
            on="slug",
            suffixes=("_base", "_arm"),
            validate="one_to_one",
        )
        for column in ("rows", "dimensions", "row_hash"):
            if not (aligned[f"{column}_base"] == aligned[f"{column}_arm"]).all():
                raise ValueError(f"Manifest row alignment failed for {arm}: {column}")
        if not (current["dimensions"] == 384).all():
            raise ValueError(f"Unexpected embedding dimension for {arm}")

    sample_parts: list[pd.DataFrame] = []
    benchmark_hashes: dict[str, str] = {}
    for slug in expected_species:
        path = root / f"data/processed/technical_benchmarks_publication_v3_26/{slug}.parquet"
        benchmark_hashes[slug] = sha256(path)
        frame = pd.read_parquet(
            path,
            columns=["pair_id", "task", "label", "benchmark_row_index", "family_transfer_class"],
        )
        expected_rows = int(base.set_index("slug").loc[slug, "rows"])
        if len(frame) != expected_rows:
            raise ValueError(f"Benchmark/embedding row mismatch for {slug}")
        if not np.array_equal(frame["benchmark_row_index"].to_numpy(), np.arange(len(frame))):
            raise ValueError(f"benchmark_row_index is not contiguous for {slug}")
        frame.insert(0, "slug", slug)
        selected_strata = []
        for (task, label), stratum in frame.groupby(["task", "label"], sort=True):
            hashes = [
                hashlib.sha256(
                    f"publication-v4|{slug}|{task}|{int(label)}|{pair_id}|{int(row_index)}".encode("utf-8")
                ).hexdigest()
                for pair_id, row_index in zip(stratum["pair_id"], stratum["benchmark_row_index"])
            ]
            ranked = stratum.assign(selection_sha256=hashes).sort_values(
                ["selection_sha256", "benchmark_row_index"], kind="mergesort"
            )
            kept = ranked.head(args.sample_per_stratum).copy()
            kept["stratum_rows"] = len(stratum)
            kept["selected_rows"] = len(kept)
            kept["selection_rank"] = np.arange(1, len(kept) + 1)
            selected_strata.append(kept)
        sample_parts.append(pd.concat(selected_strata, ignore_index=True))
    sample = pd.concat(sample_parts, ignore_index=True)
    sample = sample.sort_values(["slug", "task", "label", "selection_rank"], kind="mergesort").reset_index(drop=True)
    if sample.duplicated(["slug", "benchmark_row_index"]).any():
        raise ValueError("Deterministic sample contains duplicated rows")
    expected_max = 26 * 4 * 2 * args.sample_per_stratum
    if len(sample) > expected_max:
        raise ValueError("Deterministic sample exceeds contract maximum")

    metric_dir = root / "results/metrics/publication_v4_representation"
    metric_dir.mkdir(parents=True, exist_ok=True)
    sample_path = metric_dir / "selected_rows.tsv"
    sample.to_csv(sample_path, sep="\t", index=False)
    sample_hash = sha256(sample_path)

    arm_arrays: dict[str, np.ndarray] = {}
    species_slices: dict[str, np.ndarray] = {}
    offset = 0
    for slug, part in sample.groupby("slug", sort=False):
        species_slices[slug] = np.arange(offset, offset + len(part))
        offset += len(part)
    for arm in ARM_ORDER:
        chunks = []
        for slug, part in sample.groupby("slug", sort=False):
            path = embedding_path(root, arm, slug)
            matrix = np.load(path, mmap_mode="r")
            if matrix.shape != (
                int(base.set_index("slug").loc[slug, "rows"]),
                384,
            ):
                raise ValueError(f"Embedding shape mismatch: {arm}/{slug}: {matrix.shape}")
            chunks.append(np.asarray(matrix[part["benchmark_row_index"].to_numpy()], dtype=np.float32))
        arm_arrays[arm] = np.concatenate(chunks, axis=0)
        if arm_arrays[arm].shape != (len(sample), 384):
            raise ValueError(f"Concatenated representation shape mismatch for {arm}")

    tensors = {arm: torch.from_numpy(values).to(device) for arm, values in arm_arrays.items()}
    cka_rows: list[dict[str, object]] = []

    def record_cka(scope_type: str, scope: str, indices: np.ndarray) -> None:
        index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        scoped = {arm: tensors[arm].index_select(0, index_tensor) for arm in ARM_ORDER}
        for arm_a in ARM_ORDER:
            for arm_b in ARM_ORDER:
                cka_rows.append(
                    {
                        "scope_type": scope_type,
                        "scope": scope,
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "rows": len(indices),
                        "linear_cka": centered_linear_cka(scoped[arm_a], scoped[arm_b]),
                    }
                )

    all_indices = np.arange(len(sample))
    record_cka("global", "all", all_indices)
    for slug in expected_species:
        record_cka("species", slug, np.flatnonzero(sample["slug"].to_numpy() == slug))
    for task in sorted(sample["task"].unique()):
        record_cka("task", str(task), np.flatnonzero(sample["task"].to_numpy() == task))
    cka = pd.DataFrame(cka_rows)
    cka.to_csv(metric_dir / "linear_cka.tsv", sep="\t", index=False)

    # Row-wise direct displacement. Coordinates correspond because every arm starts
    # from the same architecture and checkpoint; Procrustes is also reported.
    displacement_rows: list[pd.DataFrame] = []
    comparisons = [(arm, "base") for arm in DAPT_ARMS] + [("tree", arm) for arm in CONTROL_ARMS]
    for arm_a, arm_b in comparisons:
        a = arm_arrays[arm_a]
        b = arm_arrays[arm_b]
        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)
        cosine = 1.0 - np.sum(a * b, axis=1) / np.clip(a_norm * b_norm, 1e-12, None)
        rel_l2 = np.linalg.norm(a - b, axis=1) / np.clip(b_norm, 1e-12, None)
        out = sample[["slug", "task", "label", "pair_id", "benchmark_row_index"]].copy()
        out.insert(0, "reference_arm", arm_b)
        out.insert(0, "comparison_arm", arm_a)
        out["cosine_distance"] = cosine.astype(np.float32)
        out["relative_l2_displacement"] = rel_l2.astype(np.float32)
        displacement_rows.append(out)
    displacement = pd.concat(displacement_rows, ignore_index=True)
    displacement_path = metric_dir / "paired_displacements.parquet"
    displacement.to_parquet(displacement_path, index=False)

    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for (arm_a, arm_b), part in displacement.groupby(["comparison_arm", "reference_arm"], sort=False):
        for metric_name in ("cosine_distance", "relative_l2_displacement"):
            values = part[metric_name].to_numpy(dtype=float)
            species = part["slug"].to_numpy()
            low, high, draws = hierarchical_bootstrap_median(
                values,
                species,
                replicates=args.bootstrap_replicates,
                seed=CONTRACT_SEED + sum(ord(char) for char in f"{arm_a}|{arm_b}|{metric_name}"),
            )
            summary_rows.append(
                {
                    "scope_type": "global",
                    "scope": "all",
                    "comparison_arm": arm_a,
                    "reference_arm": arm_b,
                    "metric": metric_name,
                    "rows": len(part),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "bootstrap_low": low,
                    "bootstrap_high": high,
                    "bootstrap_replicates": args.bootstrap_replicates,
                }
            )
            bootstrap_rows.extend(
                {
                    "comparison_arm": arm_a,
                    "reference_arm": arm_b,
                    "metric": metric_name,
                    "replicate": int(i),
                    "median": float(value),
                }
                for i, value in enumerate(draws, start=1)
            )
        for scope_type, column in (("species", "slug"), ("task", "task")):
            for scope, sub in part.groupby(column, sort=True):
                for metric_name in ("cosine_distance", "relative_l2_displacement"):
                    values = sub[metric_name].to_numpy(dtype=float)
                    summary_rows.append(
                        {
                            "scope_type": scope_type,
                            "scope": scope,
                            "comparison_arm": arm_a,
                            "reference_arm": arm_b,
                            "metric": metric_name,
                            "rows": len(sub),
                            "median": float(np.median(values)),
                            "mean": float(np.mean(values)),
                            "bootstrap_low": np.nan,
                            "bootstrap_high": np.nan,
                            "bootstrap_replicates": 0,
                        }
                    )
    displacement_summary = pd.DataFrame(summary_rows)
    displacement_summary.to_csv(metric_dir / "displacement_summary.tsv", sep="\t", index=False)
    pd.DataFrame(bootstrap_rows).to_parquet(metric_dir / "displacement_bootstrap.parquet", index=False)

    procrustes_rows = []
    for arm_a, arm_b in comparisons:
        procrustes_rows.append(
            {
                "scope_type": "global",
                "scope": "all",
                "comparison_arm": arm_a,
                "reference_arm": arm_b,
                "rows": len(sample),
                "procrustes_residual": procrustes_residual(arm_arrays[arm_b], arm_arrays[arm_a]),
            }
        )
        for task in sorted(sample["task"].unique()):
            idx = np.flatnonzero(sample["task"].to_numpy() == task)
            procrustes_rows.append(
                {
                    "scope_type": "task",
                    "scope": task,
                    "comparison_arm": arm_a,
                    "reference_arm": arm_b,
                    "rows": len(idx),
                    "procrustes_residual": procrustes_residual(arm_arrays[arm_b][idx], arm_arrays[arm_a][idx]),
                }
            )
    procrustes = pd.DataFrame(procrustes_rows)
    procrustes.to_csv(metric_dir / "procrustes_residuals.tsv", sep="\t", index=False)

    # PCA is descriptive. Fit to the complete deterministic five-arm sample and
    # retain the same 2,000 sequence rows from every arm for plotting.
    concatenated = np.concatenate([arm_arrays[arm] for arm in ARM_ORDER], axis=0)
    pca = PCA(n_components=50, svd_solver="randomized", random_state=CONTRACT_SEED)
    scores = pca.fit_transform(concatenated)
    plot_common = np.argsort(sample["selection_sha256"].to_numpy(), kind="mergesort")[: min(2000, len(sample))]
    pca_parts = []
    for arm_index, arm in enumerate(ARM_ORDER):
        offset = arm_index * len(sample)
        coords = scores[offset + plot_common, :2]
        meta = sample.iloc[plot_common][["slug", "task", "label", "pair_id", "benchmark_row_index", "selection_sha256"]].reset_index(drop=True)
        meta.insert(0, "arm", arm)
        meta["pc1"] = coords[:, 0]
        meta["pc2"] = coords[:, 1]
        pca_parts.append(meta)
    pca_coordinates = pd.concat(pca_parts, ignore_index=True)
    pca_coordinates.to_parquet(metric_dir / "pca_coordinates.parquet", index=False)
    pca_variance = pd.DataFrame(
        {
            "component": np.arange(1, 51),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    pca_variance.to_csv(metric_dir / "pca_explained_variance.tsv", sep="\t", index=False)

    global_cka = cka.loc[cka["scope_type"] == "global"].pivot(index="arm_a", columns="arm_b", values="linear_cka")
    nearest_by_cka = max(CONTROL_ARMS, key=lambda arm: float(global_cka.loc["tree", arm]))
    tree_dist = displacement_summary.loc[
        (displacement_summary["scope_type"] == "global")
        & (displacement_summary["comparison_arm"] == "tree")
        & (displacement_summary["reference_arm"].isin(CONTROL_ARMS))
        & (displacement_summary["metric"] == "cosine_distance")
    ]
    nearest_by_cosine = str(tree_dist.sort_values("median").iloc[0]["reference_arm"])
    closest_concordant = nearest_by_cka == nearest_by_cosine

    del tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_paths = [
        sample_path,
        metric_dir / "linear_cka.tsv",
        displacement_path,
        metric_dir / "displacement_summary.tsv",
        metric_dir / "displacement_bootstrap.parquet",
        metric_dir / "procrustes_residuals.tsv",
        metric_dir / "pca_coordinates.parquet",
        metric_dir / "pca_explained_variance.tsv",
    ]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "publication_v4_seed23_technical_representation",
        "analysis_tier": "posthoc_seed23_descriptive",
        "decision_authority": False,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arms": ARM_ORDER,
        "species": len(expected_species),
        "tasks": sorted(sample["task"].unique().tolist()),
        "dimensions": 384,
        "sample_per_species_task_label": args.sample_per_stratum,
        "sample_rows": len(sample),
        "selected_rows_sha256": sample_hash,
        "bootstrap_replicates": args.bootstrap_replicates,
        "nearest_control_to_tree_by_global_cka": nearest_by_cka,
        "nearest_control_to_tree_by_median_cosine_distance": nearest_by_cosine,
        "nearest_control_concordant": closest_concordant,
        "global_tree_control_cka": {arm: float(global_cka.loc["tree", arm]) for arm in CONTROL_ARMS},
        "pca_components": 50,
        "pca_plot_rows_per_arm": len(plot_common),
        "pca_first_two_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "manifest_sha256": manifest_hashes,
        "benchmark_sha256": benchmark_hashes,
        "output_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in output_paths},
        "malus_outcomes_accessed": False,
    }
    summary_path = metric_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
