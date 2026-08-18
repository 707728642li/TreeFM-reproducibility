#!/usr/bin/env python3
"""Exploratory GC-adjusted cross-genus promoter k-mer meta-analysis."""

from __future__ import annotations

import argparse
import os
import gzip
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "16")

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm


BASES = "ACGT"
BASE_CODE = {base: index for index, base in enumerate(BASES)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def decode(code: int, k: int) -> str:
    chars = ["A"] * k
    for index in range(k - 1, -1, -1):
        chars[index] = BASES[code & 3]
        code >>= 2
    return "".join(chars)


def reverse_complement_code(code: int, k: int) -> int:
    result = 0
    for _ in range(k):
        result = (result << 2) | (3 - (code & 3))
        code >>= 2
    return result


def feature_maps(
    ks: tuple[int, ...],
) -> tuple[list[str], dict[int, tuple[np.ndarray, int, int]]]:
    names: list[str] = []
    maps: dict[int, tuple[np.ndarray, int, int]] = {}
    offset = 0
    for k in ks:
        canonical_codes = sorted(
            {
                min(code, reverse_complement_code(code, k))
                for code in range(4**k)
            }
        )
        canonical_to_index = {
            code: offset + index
            for index, code in enumerate(canonical_codes)
        }
        code_map = np.empty(4**k, dtype=np.int32)
        for code in range(4**k):
            canonical = min(code, reverse_complement_code(code, k))
            code_map[code] = canonical_to_index[canonical]
        names.extend(f"{k}mer_{decode(code, k)}" for code in canonical_codes)
        maps[k] = (code_map, offset, len(canonical_codes))
        offset += len(canonical_codes)
    return names, maps


def sequence_features(
    sequences: list[str],
    ks: tuple[int, ...],
    maps: dict[int, tuple[np.ndarray, int, int]],
    total_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.zeros((len(sequences), total_features), dtype=np.float32)
    gc = np.zeros(len(sequences), dtype=np.float64)
    cpg = np.zeros(len(sequences), dtype=np.float64)
    for row_index, raw in enumerate(sequences):
        sequence = raw.upper()
        valid_bases = sum(sequence.count(base) for base in BASES)
        gc[row_index] = (
            (sequence.count("G") + sequence.count("C")) / valid_bases
            if valid_bases
            else np.nan
        )
        cpg[row_index] = (
            sequence.count("CG") / max(len(sequence) - 1, 1)
        )
        encoded = [BASE_CODE.get(base, -1) for base in sequence]
        for k in ks:
            code_map, offset, count = maps[k]
            local_counts = np.zeros(count, dtype=np.float32)
            code = 0
            valid_run = 0
            mask = (1 << (2 * k)) - 1
            valid_windows = 0
            for base in encoded:
                if base < 0:
                    code = 0
                    valid_run = 0
                    continue
                code = ((code << 2) | base) & mask
                valid_run += 1
                if valid_run >= k:
                    global_index = int(code_map[code])
                    local_counts[global_index - offset] += 1
                    valid_windows += 1
            if valid_windows:
                matrix[
                    row_index, offset : offset + count
                ] = local_counts / valid_windows
    return matrix, gc, cpg


def fit_null(covariates: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros(covariates.shape[1], dtype=np.float64)
    for _ in range(100):
        eta = np.clip(covariates @ coefficients, -30, 30)
        mu = expit(eta)
        weights = np.clip(mu * (1 - mu), 1e-8, None)
        working = eta + (y - mu) / weights
        information = covariates.T @ (weights[:, None] * covariates)
        information.flat[:: information.shape[0] + 1] += 1e-8
        updated = np.linalg.solve(
            information,
            covariates.T @ (weights * working),
        )
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    mu = expit(np.clip(covariates @ coefficients, -30, 30))
    return mu, coefficients


def score_features(
    matrix: np.ndarray,
    gc: np.ndarray,
    cpg: np.ndarray,
    y: np.ndarray,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if center is None:
        center = np.nanmean(matrix, axis=0)
    if scale is None:
        scale = np.nanstd(matrix, axis=0, ddof=1)
    valid_scale = np.isfinite(scale) & (scale > 0)
    safe_scale = np.where(valid_scale, scale, 1.0)
    standardized = (matrix - center) / safe_scale
    standardized[:, ~valid_scale] = 0.0
    covariates = np.column_stack(
        [
            np.ones(len(y)),
            gc,
            gc**2,
            cpg,
        ]
    )
    mu, null_coefficients = fit_null(covariates, y)
    weights = np.clip(mu * (1 - mu), 1e-8, None)
    residual = y - mu
    information = covariates.T @ (weights[:, None] * covariates)
    information.flat[:: information.shape[0] + 1] += 1e-8
    inverse = np.linalg.inv(information)
    cross = covariates.T @ (weights[:, None] * standardized)
    raw_variance = np.sum(
        weights[:, None] * standardized**2,
        axis=0,
    )
    adjustment = np.sum(cross * (inverse @ cross), axis=0)
    variance = raw_variance - adjustment
    score = standardized.T @ residual
    valid = valid_scale & np.isfinite(variance) & (variance > 1e-10)
    beta = np.full(matrix.shape[1], np.nan)
    standard_error = np.full(matrix.shape[1], np.nan)
    z_value = np.full(matrix.shape[1], np.nan)
    p_value = np.full(matrix.shape[1], np.nan)
    beta[valid] = score[valid] / variance[valid]
    standard_error[valid] = 1 / np.sqrt(variance[valid])
    z_value[valid] = score[valid] / np.sqrt(variance[valid])
    p_value[valid] = 2 * norm.sf(np.abs(z_value[valid]))
    return {
        "beta": beta,
        "se": standard_error,
        "z": z_value,
        "p": p_value,
        "center": center,
        "scale": scale,
        "null_coefficients": null_coefficients,
    }


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    result = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return result
    order = valid[np.argsort(p_values[valid], kind="stable")]
    ranked = p_values[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def load_genus(
    path: Path,
    genus: str,
    ks: tuple[int, ...],
    maps: dict[int, tuple[np.ndarray, int, int]],
    total_features: int,
) -> dict[str, object]:
    frame = pd.read_parquet(path)
    required = {"promoter_2048", "label_binary", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{genus} promoter data lack columns: {missing}")
    frame = frame.loc[
        frame["label_binary"].isin([0, 1])
        & frame["promoter_2048"].notna()
    ].reset_index(drop=True)
    matrix, gc, cpg = sequence_features(
        frame["promoter_2048"].astype(str).tolist(),
        ks,
        maps,
        total_features,
    )
    if not np.all(np.isfinite(gc)):
        raise RuntimeError(f"{genus} contains promoter with no valid bases")
    y = frame["label_binary"].to_numpy(dtype=np.float64)
    discovery = frame["split"].astype(str).ne("test").to_numpy()
    test = frame["split"].astype(str).eq("test").to_numpy()
    if min(y[discovery].sum(), len(y[discovery]) - y[discovery].sum()) < 50:
        raise RuntimeError(f"{genus} discovery split has too few labels")
    if min(y[test].sum(), len(y[test]) - y[test].sum()) < 20:
        raise RuntimeError(f"{genus} test split has too few labels")
    discovery_result = score_features(
        matrix[discovery],
        gc[discovery],
        cpg[discovery],
        y[discovery],
    )
    test_result = score_features(
        matrix[test],
        gc[test],
        cpg[test],
        y[test],
        center=discovery_result["center"],
        scale=discovery_result["scale"],
    )
    return {
        "frame": frame,
        "matrix": matrix,
        "gc": gc,
        "cpg": cpg,
        "y": y,
        "discovery": discovery,
        "test": test,
        "discovery_result": discovery_result,
        "test_result": test_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    root = args.project_root.resolve()
    expected = Path(os.environ["TREEFM_ROOT"]).resolve() if os.environ.get("TREEFM_ROOT") else None
    if expected is not None and root != expected:
        raise SystemExit(f"refusing to run outside {expected}: {root}")
    output_root = (
        root
        / "results/biological_cases/publication_v3_cross_genus_promoter_kmers"
    )
    controller = output_root / "controller.json"
    state: dict[str, object] = {
        "status": "running",
        "started_utc": utc_now(),
        "contract": (
            "docs/publication_v3_exploratory_cross_genus_kmer_plan.md"
        ),
        "exploratory_only": True,
    }
    write_json(controller, state)
    try:
        ks = (4, 5, 6)
        names, maps = feature_maps(ks)
        paths = {
            "Prunus": root
            / "data/processed/functional/Prunus_publication_v3/"
            "promoter_labels.parquet",
            "Pyrus": root
            / "data/processed/functional/Pyrus_PRJNA669907/"
            "promoter_labels.parquet",
        }
        genera = {
            genus: load_genus(
                path,
                genus,
                ks,
                maps,
                len(names),
            )
            for genus, path in paths.items()
        }
        prunus = genera["Prunus"]["discovery_result"]
        pyrus = genera["Pyrus"]["discovery_result"]
        weights_prunus = 1 / np.square(prunus["se"])
        weights_pyrus = 1 / np.square(pyrus["se"])
        total_weight = weights_prunus + weights_pyrus
        meta_beta = (
            weights_prunus * prunus["beta"]
            + weights_pyrus * pyrus["beta"]
        ) / total_weight
        meta_se = 1 / np.sqrt(total_weight)
        meta_z = meta_beta / meta_se
        meta_p = 2 * norm.sf(np.abs(meta_z))
        meta_q = bh_adjust(meta_p)
        q_statistic = (
            weights_prunus * np.square(prunus["beta"] - meta_beta)
            + weights_pyrus * np.square(pyrus["beta"] - meta_beta)
        )
        q_p = chi2.sf(q_statistic, df=1)
        i_squared = np.maximum(0, (q_statistic - 1) / q_statistic)
        i_squared[~np.isfinite(i_squared)] = 0

        records: dict[str, object] = {
            "feature": names,
            "k": [int(name.split("mer_", 1)[0]) for name in names],
            "canonical_kmer": [
                name.split("mer_", 1)[1] for name in names
            ],
            "prunus_discovery_beta": prunus["beta"],
            "prunus_discovery_se": prunus["se"],
            "prunus_discovery_p": prunus["p"],
            "pyrus_discovery_beta": pyrus["beta"],
            "pyrus_discovery_se": pyrus["se"],
            "pyrus_discovery_p": pyrus["p"],
            "meta_beta": meta_beta,
            "meta_se": meta_se,
            "meta_z": meta_z,
            "meta_p": meta_p,
            "meta_q": meta_q,
            "heterogeneity_q": q_statistic,
            "heterogeneity_p": q_p,
            "heterogeneity_i2": i_squared,
            "prunus_test_beta": genera["Prunus"]["test_result"]["beta"],
            "prunus_test_p": genera["Prunus"]["test_result"]["p"],
            "pyrus_test_beta": genera["Pyrus"]["test_result"]["beta"],
            "pyrus_test_p": genera["Pyrus"]["test_result"]["p"],
        }
        result = pd.DataFrame(records)
        result["discovery_same_direction"] = (
            np.sign(result["prunus_discovery_beta"])
            == np.sign(result["pyrus_discovery_beta"])
        )
        result["prunus_test_direction_match"] = (
            np.sign(result["prunus_test_beta"])
            == np.sign(result["meta_beta"])
        )
        result["pyrus_test_direction_match"] = (
            np.sign(result["pyrus_test_beta"])
            == np.sign(result["meta_beta"])
        )
        result["replicated_signature"] = (
            result["meta_q"].le(0.05)
            & result["discovery_same_direction"]
            & result["prunus_test_direction_match"]
            & result["pyrus_test_direction_match"]
        )
        observed_replicated = int(result["replicated_signature"].sum())
        rng = np.random.default_rng(args.seed)
        null_replicated_counts: list[int] = []
        null_max_abs_meta_z: list[float] = []
        for _ in range(args.permutations):
            permuted_discovery = {}
            permuted_test = {}
            for genus, data in genera.items():
                discovery = data["discovery"]
                test = data["test"]
                discovery_y = data["y"][discovery].copy()
                test_y = data["y"][test].copy()
                rng.shuffle(discovery_y)
                rng.shuffle(test_y)
                observed_scaling = data["discovery_result"]
                permuted_discovery[genus] = score_features(
                    data["matrix"][discovery],
                    data["gc"][discovery],
                    data["cpg"][discovery],
                    discovery_y,
                    center=observed_scaling["center"],
                    scale=observed_scaling["scale"],
                )
                permuted_test[genus] = score_features(
                    data["matrix"][test],
                    data["gc"][test],
                    data["cpg"][test],
                    test_y,
                    center=observed_scaling["center"],
                    scale=observed_scaling["scale"],
                )
            null_prunus = permuted_discovery["Prunus"]
            null_pyrus = permuted_discovery["Pyrus"]
            null_w_prunus = 1 / np.square(null_prunus["se"])
            null_w_pyrus = 1 / np.square(null_pyrus["se"])
            null_total_weight = null_w_prunus + null_w_pyrus
            null_meta_beta = (
                null_w_prunus * null_prunus["beta"]
                + null_w_pyrus * null_pyrus["beta"]
            ) / null_total_weight
            null_meta_se = 1 / np.sqrt(null_total_weight)
            null_meta_z = null_meta_beta / null_meta_se
            null_meta_p = 2 * norm.sf(np.abs(null_meta_z))
            null_meta_q = bh_adjust(null_meta_p)
            null_same_discovery = (
                np.sign(null_prunus["beta"])
                == np.sign(null_pyrus["beta"])
            )
            null_prunus_test_match = (
                np.sign(permuted_test["Prunus"]["beta"])
                == np.sign(null_meta_beta)
            )
            null_pyrus_test_match = (
                np.sign(permuted_test["Pyrus"]["beta"])
                == np.sign(null_meta_beta)
            )
            null_replicated = (
                (null_meta_q <= 0.05)
                & null_same_discovery
                & null_prunus_test_match
                & null_pyrus_test_match
            )
            null_replicated_counts.append(int(null_replicated.sum()))
            null_max_abs_meta_z.append(
                float(np.nanmax(np.abs(null_meta_z)))
            )
        result = result.sort_values(
            ["replicated_signature", "meta_q", "meta_p"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        output_root.mkdir(parents=True, exist_ok=True)
        table_path = output_root / "cross_genus_kmer_meta.tsv.gz"
        with gzip.open(table_path, "wt", encoding="utf-8") as handle:
            result.to_csv(handle, sep="\t", index=False)
        top = result.head(50)
        top.to_csv(
            output_root / "top_50_cross_genus_kmers.tsv",
            sep="\t",
            index=False,
        )
        genus_summary = {}
        for genus, data in genera.items():
            frame = data["frame"]
            discovery = data["discovery"]
            test = data["test"]
            y = data["y"]
            genus_summary[genus] = {
                "promoters": int(len(frame)),
                "discovery_promoters": int(discovery.sum()),
                "discovery_positives": int(y[discovery].sum()),
                "discovery_negatives": int(
                    discovery.sum() - y[discovery].sum()
                ),
                "test_promoters": int(test.sum()),
                "test_positives": int(y[test].sum()),
                "test_negatives": int(test.sum() - y[test].sum()),
                "null_coefficients": [
                    float(value)
                    for value in data["discovery_result"][
                        "null_coefficients"
                    ]
                ],
            }
        summary = {
            "status": "complete",
            "finished_utc": utc_now(),
            "contract": state["contract"],
            "exploratory_only": True,
            "k_values": list(ks),
            "features_tested": int(len(result)),
            "meta_fdr_005": int(result["meta_q"].le(0.05).sum()),
            "discovery_same_direction_fdr_005": int(
                (
                    result["meta_q"].le(0.05)
                    & result["discovery_same_direction"]
                ).sum()
            ),
            "replicated_signatures": observed_replicated,
            "null_permutations": args.permutations,
            "null_seed": args.seed,
            "null_replicated_median": float(
                np.median(null_replicated_counts)
            ),
            "null_replicated_maximum": int(max(null_replicated_counts)),
            "replicated_count_empirical_p": float(
                (
                    1
                    + sum(
                        count >= observed_replicated
                        for count in null_replicated_counts
                    )
                )
                / (args.permutations + 1)
            ),
            "observed_max_abs_meta_z": float(
                np.nanmax(np.abs(result["meta_z"]))
            ),
            "null_max_abs_meta_z_median": float(
                np.median(null_max_abs_meta_z)
            ),
            "null_max_abs_meta_z_maximum": float(
                max(null_max_abs_meta_z)
            ),
            "genus_summary": genus_summary,
            "top_replicated": (
                result.loc[result["replicated_signature"]]
                .head(20)[
                    [
                        "canonical_kmer",
                        "k",
                        "meta_beta",
                        "meta_q",
                        "heterogeneity_i2",
                        "prunus_test_beta",
                        "pyrus_test_beta",
                    ]
                ]
                .to_dict(orient="records")
            ),
            "artifacts": {
                "full_table": str(table_path.relative_to(root)),
                "top_50": str(
                    (
                        output_root / "top_50_cross_genus_kmers.tsv"
                    ).relative_to(root)
                ),
            },
            "null_replicated_counts": null_replicated_counts,
        }
        write_json(output_root / "summary.json", summary)
        state.update(summary)
        write_json(controller, state)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "error": str(error),
            }
        )
        write_json(controller, state)
        raise


if __name__ == "__main__":
    main()
