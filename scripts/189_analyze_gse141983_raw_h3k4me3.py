#!/usr/bin/env python3
"""QC and analyze the frozen post-hoc GSE141983 raw H3K4me3 sensitivity."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tmm_factors(counts: np.ndarray) -> tuple[np.ndarray, int]:
    """Faithful edgeR-style weighted trimmed-mean-of-M-values factors."""
    library_sizes = counts.sum(axis=0).astype(float)
    if np.any(library_sizes <= 0):
        raise RuntimeError("zero count-library size")
    scaled = counts / library_sizes
    upper_quartiles = np.quantile(scaled, 0.75, axis=0)
    reference_index = int(np.argmin(np.abs(upper_quartiles - upper_quartiles.mean())))
    ref = counts[:, reference_index].astype(float)
    ref_library = library_sizes[reference_index]
    factors = np.ones(counts.shape[1], dtype=float)
    for column in range(counts.shape[1]):
        if column == reference_index:
            continue
        obs = counts[:, column].astype(float)
        library = library_sizes[column]
        keep = (obs > 0) & (ref > 0)
        obs = obs[keep]
        ref_keep = ref[keep]
        if len(obs) == 0:
            raise RuntimeError(f"no common positive genes for TMM column {column}")
        log_ratio = np.log2((obs / library) / (ref_keep / ref_library))
        absolute_expression = 0.5 * np.log2(
            (obs / library) * (ref_keep / ref_library)
        )
        variance = (library - obs) / (library * obs) + (
            ref_library - ref_keep
        ) / (ref_library * ref_keep)
        finite = np.isfinite(log_ratio) & np.isfinite(absolute_expression) & (variance > 0)
        log_ratio = log_ratio[finite]
        absolute_expression = absolute_expression[finite]
        variance = variance[finite]
        n = len(log_ratio)
        if n < 100:
            raise RuntimeError(f"too few TMM genes for column {column}: {n}")
        m_rank = stats.rankdata(log_ratio, method="average")
        a_rank = stats.rankdata(absolute_expression, method="average")
        lower_m = math.floor(n * 0.30) + 1
        upper_m = n + 1 - lower_m
        lower_a = math.floor(n * 0.05) + 1
        upper_a = n + 1 - lower_a
        retained = (
            (m_rank >= lower_m)
            & (m_rank <= upper_m)
            & (a_rank >= lower_a)
            & (a_rank <= upper_a)
        )
        if retained.sum() < 100:
            raise RuntimeError(f"too few retained TMM genes for column {column}")
        mean_log_ratio = np.average(log_ratio[retained], weights=1 / variance[retained])
        factors[column] = 2**mean_log_ratio
    factors /= np.exp(np.mean(np.log(factors)))
    return factors, reference_index


def statistic(effect: np.ndarray, categories: np.ndarray) -> float:
    up = categories == "positive_up"
    down = categories == "positive_down"
    negative = categories == "negative"
    positive_count = int(up.sum() + down.sum())
    if positive_count == 0 or negative.sum() == 0:
        return float("nan")
    positive_oriented = (effect[up].sum() - effect[down].sum()) / positive_count
    mixture = (up.sum() - down.sum()) / positive_count
    negative_reference = mixture * effect[negative].mean()
    return float(positive_oriented - negative_reference)


def oriented_effect(effect: np.ndarray, categories: np.ndarray) -> np.ndarray:
    up = categories == "positive_up"
    down = categories == "positive_down"
    positive_count = int(up.sum() + down.sum())
    mixture = (up.sum() - down.sum()) / max(1, positive_count)
    result = effect * mixture
    result[up] = effect[up]
    result[down] = -effect[down]
    return result


def permute_categories(
    categories: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    permuted = categories.copy()
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        permuted[indices] = categories[indices][rng.permutation(len(indices))]
    return permuted


def ols_hc3(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ xtx_inverse) * x, axis=1)
    adjusted = residual / np.maximum(1e-10, 1 - leverage)
    meat = x.T @ ((adjusted**2)[:, None] * x)
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_errors = np.sqrt(np.maximum(0, np.diag(covariance)))
    return beta, standard_errors


def cpg_fraction(sequence: str) -> float:
    sequence = str(sequence).upper()
    bases = sum(sequence.count(base) for base in "ACGT")
    return sequence.count("CG") / max(1, bases - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        default="metadata/publication_v3/gse141983_raw_sensitivity/"
        "eligible_h3_h3k4me3_runs.tsv",
    )
    parser.add_argument(
        "--featurecounts",
        default="results/biological_cases/prunus_publication_v3_gse141983_raw/"
        "counts/tss_counts.txt",
    )
    parser.add_argument(
        "--label-covariates",
        default="results/biological_cases/prunus_publication_v3_chromatin_replication/"
        "gse190586_binary_gene_calls.tsv.gz",
    )
    parser.add_argument(
        "--promoter-labels",
        default="data/processed/functional/Prunus_publication_v3/"
        "promoter_labels.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="results/biological_cases/prunus_publication_v3_gse141983_raw",
    )
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstraps", type=int, default=2_000)
    parser.add_argument("--permutation-seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--gbox-permutation-seed", type=int, default=20260730)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_root = root / args.output_root
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    with (root / args.manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    if len(manifest) != 12:
        raise RuntimeError(f"expected 12 samples, found {len(manifest)}")

    counts_raw = pd.read_csv(root / args.featurecounts, sep="\t", comment="#")
    sample_columns = list(counts_raw.columns[6:])
    run_by_column: dict[str, str] = {}
    for column in sample_columns:
        basename = Path(column).name
        matches = [row["run_accession"] for row in manifest if basename.startswith(row["run_accession"])]
        if len(matches) != 1:
            raise RuntimeError(f"cannot resolve featureCounts sample column: {column}")
        run_by_column[column] = matches[0]
    if len(run_by_column) != 12:
        raise RuntimeError("featureCounts sample cardinality differs")
    counts = counts_raw.set_index("Geneid")[sample_columns].astype(np.int64)
    counts.columns = [run_by_column[column] for column in sample_columns]
    counts = counts[[row["run_accession"] for row in manifest]]
    count_matrix_path = analysis_root / "tss_counts_matrix.tsv.gz"
    deterministic_gzip = {"method": "gzip", "mtime": 0}
    counts.to_csv(count_matrix_path, sep="\t", compression=deterministic_gzip)

    factors, reference_index = tmm_factors(counts.to_numpy(dtype=float))
    library_sizes = counts.sum(axis=0).to_numpy(dtype=float)
    effective_sizes = library_sizes * factors
    cpm = counts.to_numpy(dtype=float) / effective_sizes[None, :] * 1_000_000
    log_cpm = pd.DataFrame(
        np.log2(cpm + 0.5), index=counts.index, columns=counts.columns
    )
    tmm_frame = pd.DataFrame(
        {
            "run_accession": counts.columns,
            "library_size": library_sizes.astype(np.int64),
            "tmm_factor": factors,
            "effective_library_size": effective_sizes,
            "reference_library": [index == reference_index for index in range(len(factors))],
        }
    )
    tmm_path = analysis_root / "tmm_factors.tsv"
    tmm_frame.to_csv(tmm_path, sep="\t", index=False)
    log_cpm_path = analysis_root / "tmm_log2_cpm.tsv.gz"
    log_cpm.to_csv(log_cpm_path, sep="\t", compression=deterministic_gzip)

    manifest_by_cell = {
        (row["stage"], int(row["replicate"]), row["mark"]): row["run_accession"]
        for row in manifest
    }
    signals: dict[str, pd.Series] = {}
    for stage in ("date1", "date2", "date3"):
        for replicate in (2, 3):
            k4 = manifest_by_cell[(stage, replicate, "H3K4me3")]
            h3 = manifest_by_cell[(stage, replicate, "H3")]
            signals[f"{stage}_rep{replicate}"] = log_cpm[k4] - log_cpm[h3]
    signal_frame = pd.DataFrame(signals)
    signal_path = analysis_root / "h3_normalized_signal.tsv.gz"
    signal_frame.to_csv(signal_path, sep="\t", compression=deterministic_gzip)
    effect = (
        signal_frame["date3_rep2"]
        + signal_frame["date3_rep3"]
        - 0.5
        * (
            signal_frame["date1_rep2"]
            + signal_frame["date1_rep3"]
            + signal_frame["date2_rep2"]
            + signal_frame["date2_rep3"]
        )
    ) / 2
    effect.name = "january_minus_equal_weight_october_december"
    effect_path = analysis_root / "gene_effects.tsv.gz"
    effect.to_csv(effect_path, sep="\t", compression=deterministic_gzip, header=True)

    feature_summary = pd.read_csv(
        Path(str(root / args.featurecounts) + ".summary"), sep="\t"
    ).set_index("Status")
    assigned_by_run: dict[str, int] = {}
    for column in feature_summary.columns:
        basename = Path(column).name
        run = next(row["run_accession"] for row in manifest if basename.startswith(row["run_accession"]))
        assigned_by_run[run] = int(feature_summary.loc["Assigned", column])

    sample_qc = []
    technical_violations: list[str] = []
    for row in manifest:
        run = row["run_accession"]
        payload = json.loads(
            (
                output_root / "alignment" / run / "sample_qc.json"
            ).read_text(encoding="utf-8-sig")
        )
        nonzero = int((counts[run] > 0).sum())
        qc = {
            "run_accession": run,
            "stage": row["stage"],
            "replicate": int(row["replicate"]),
            "mark": row["mark"],
            "post_fastp_pairs": int(payload["post_fastp_pairs"]),
            "post_fastp_q30_rate": float(payload["post_fastp_q30_rate"]),
            "mapq30_nonduplicate_pair_fraction": float(
                payload["mapq30_nonduplicate_pair_fraction"]
            ),
            "assigned_tss_fragments": assigned_by_run[run],
            "nonzero_tss_genes": nonzero,
        }
        qc["pass"] = (
            qc["post_fastp_pairs"] >= 10_000_000
            and qc["post_fastp_q30_rate"] >= 0.80
            and qc["mapq30_nonduplicate_pair_fraction"] >= 0.50
            and qc["assigned_tss_fragments"] >= 1_000_000
            and qc["nonzero_tss_genes"] >= 10_000
        )
        if not qc["pass"]:
            technical_violations.append(f"sample_gate_failed:{run}")
        sample_qc.append(qc)

    replicate_qc = []
    for stage in ("date1", "date2", "date3"):
        required_runs = [
            manifest_by_cell[(stage, replicate, mark)]
            for replicate in (2, 3)
            for mark in ("H3", "H3K4me3")
        ]
        common_nonzero = (counts[required_runs] > 0).all(axis=1)
        rho = float(
            stats.spearmanr(
                signal_frame.loc[common_nonzero, f"{stage}_rep2"],
                signal_frame.loc[common_nonzero, f"{stage}_rep3"],
            ).statistic
        )
        record = {
            "stage": stage,
            "genes": int(common_nonzero.sum()),
            "spearman_rho": rho,
            "pass": int(common_nonzero.sum()) >= 10_000 and rho >= 0.70,
        }
        if not record["pass"]:
            technical_violations.append(f"replicate_gate_failed:{stage}")
        replicate_qc.append(record)

    technical_gate = {
        "status": "pass" if not technical_violations else "fail",
        "scope": "posthoc_gse141983_raw_h3_h3k4me3_technical_gate",
        "malus_accessed": False,
        "violations": technical_violations,
        "sample_thresholds": {
            "post_fastp_pairs_min": 10_000_000,
            "post_fastp_q30_rate_min": 0.80,
            "mapq30_nonduplicate_pair_fraction_min": 0.50,
            "assigned_tss_fragments_min": 1_000_000,
            "nonzero_tss_genes_min": 10_000,
        },
        "replicate_thresholds": {
            "common_nonzero_genes_min": 10_000,
            "h3_normalized_signal_spearman_min": 0.70,
        },
        "sample_qc": sample_qc,
        "replicate_qc": replicate_qc,
        "normalization": {
            "method": "weighted_TMM_edgeR_style",
            "reference_run": counts.columns[reference_index],
            "log_signal": "log2(TMM_CPM_H3K4me3+0.5)-log2(TMM_CPM_H3+0.5)",
        },
        "output_sha256": {
            str(count_matrix_path.relative_to(root)): sha256_file(count_matrix_path),
            str(tmm_path.relative_to(root)): sha256_file(tmm_path),
            str(log_cpm_path.relative_to(root)): sha256_file(log_cpm_path),
            str(signal_path.relative_to(root)): sha256_file(signal_path),
            str(effect_path.relative_to(root)): sha256_file(effect_path),
        },
    }
    technical_gate_path = output_root / "technical_gate.json"
    write_json(technical_gate_path, technical_gate)
    if technical_violations:
        summary = {
            "status": "complete_omitted_technical_gate_failed",
            "scope": "posthoc_gse141983_raw_continuous_sensitivity",
            "technical_gate": "fail",
            "label_data_read": False,
            "malus_accessed": False,
            "violations": technical_violations,
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0

    # The label and promoter files are intentionally opened only after every
    # outcome-free technical gate has passed.
    covariates = pd.read_csv(root / args.label_covariates, sep="\t")
    promoters = pd.read_parquet(
        root / args.promoter_labels,
        columns=["gene_id", "promoter_sequence", "label", "endpoint_direction"],
    ).drop_duplicates("gene_id")
    joined = covariates.merge(
        effect.rename("chromatin_effect"), left_on="gene_id", right_index=True, how="inner"
    )
    joined = joined.merge(
        promoters[["gene_id", "promoter_sequence"]], on="gene_id", how="left", validate="one_to_one"
    )
    expected_positive = int((covariates["label"] == "positive").sum())
    expected_negative = int((covariates["label"] == "negative").sum())
    mapped_positive = int((joined["label"] == "positive").sum())
    mapped_negative = int((joined["label"] == "negative").sum())
    positive_mapping_fraction = mapped_positive / max(1, expected_positive)
    negative_mapping_fraction = mapped_negative / max(1, expected_negative)
    if positive_mapping_fraction < 0.70 or negative_mapping_fraction < 0.70:
        summary = {
            "status": "complete_omitted_mapping_gate_failed",
            "scope": "posthoc_gse141983_raw_continuous_sensitivity",
            "technical_gate": "pass",
            "label_data_read": True,
            "positive_mapping_fraction": positive_mapping_fraction,
            "negative_mapping_fraction": negative_mapping_fraction,
            "malus_accessed": False,
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0

    categories = np.where(
        joined["label"].to_numpy() == "negative",
        "negative",
        np.where(
            joined["endpoint_direction"].to_numpy() == "up",
            "positive_up",
            "positive_down",
        ),
    )
    effects = joined["chromatin_effect"].to_numpy(dtype=float)
    strata = joined["stratum"].astype(str).to_numpy()
    observed = statistic(effects, categories)
    permutation_rng = np.random.default_rng(args.permutation_seed)
    null = np.empty(args.permutations, dtype=float)
    for index in range(args.permutations):
        null[index] = statistic(
            effects, permute_categories(categories, strata, permutation_rng)
        )
    empirical_p = (1 + int(np.sum(null >= observed))) / (args.permutations + 1)

    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)
    group_indices = {
        category: np.flatnonzero(categories == category)
        for category in ("positive_up", "positive_down", "negative")
    }
    bootstrap = np.empty(args.bootstraps, dtype=float)
    for index in range(args.bootstraps):
        sampled_effects = []
        sampled_categories = []
        for category, indices in group_indices.items():
            sampled = bootstrap_rng.choice(indices, size=len(indices), replace=True)
            sampled_effects.append(effects[sampled])
            sampled_categories.extend([category] * len(indices))
        bootstrap[index] = statistic(
            np.concatenate(sampled_effects), np.asarray(sampled_categories)
        )
    bootstrap_interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()

    joined["category"] = categories
    joined["direction_oriented_chromatin_effect"] = oriented_effect(effects, categories)
    joined["exact_gbox"] = joined["promoter_sequence"].str.upper().str.contains("CACGTG", regex=False)
    joined["cpg_fraction"] = joined["promoter_sequence"].map(cpg_fraction)
    gene_results_path = analysis_root / "label_chromatin_gene_results.tsv.gz"
    joined.drop(columns=["promoter_sequence"]).to_csv(
        gene_results_path, sep="\t", index=False, compression=deterministic_gzip
    )
    null_path = analysis_root / "continuous_permutation_null.tsv.gz"
    pd.DataFrame({"permutation": np.arange(1, len(null) + 1), "statistic": null}).to_csv(
        null_path, sep="\t", index=False, compression=deterministic_gzip
    )
    bootstrap_path = analysis_root / "continuous_bootstrap.tsv.gz"
    pd.DataFrame(
        {"bootstrap": np.arange(1, len(bootstrap) + 1), "statistic": bootstrap}
    ).to_csv(bootstrap_path, sep="\t", index=False, compression=deterministic_gzip)

    # Frozen exact-G-box interaction, with HC3 uncertainty for the observed fit
    # and a one-sided category permutation value for a positive interaction.
    label_status = (categories != "negative").astype(float)
    gbox = joined["exact_gbox"].astype(float).to_numpy()
    gc = joined["gc_fraction"].to_numpy(dtype=float)
    cpg = joined["cpg_fraction"].to_numpy(dtype=float)
    expression = joined["baseline_expression_rank_score"].to_numpy(dtype=float)
    y = oriented_effect(effects, categories)
    x = np.column_stack(
        [np.ones(len(joined)), label_status, gbox, label_status * gbox, gc, gc**2, cpg, expression]
    )
    beta, standard_error = ols_hc3(x, y)
    interaction_beta = float(beta[3])
    interaction_se = float(standard_error[3])
    interaction_z = interaction_beta / interaction_se
    interaction_hc3_p = float(2 * stats.norm.sf(abs(interaction_z)))
    gbox_rng = np.random.default_rng(args.gbox_permutation_seed)
    gbox_null = np.empty(args.permutations, dtype=float)
    for index in range(args.permutations):
        permuted = permute_categories(categories, strata, gbox_rng)
        perm_label = (permuted != "negative").astype(float)
        perm_y = oriented_effect(effects, permuted)
        perm_x = np.column_stack(
            [
                np.ones(len(joined)),
                perm_label,
                gbox,
                perm_label * gbox,
                gc,
                gc**2,
                cpg,
                expression,
            ]
        )
        gbox_null[index] = float(np.linalg.lstsq(perm_x, perm_y, rcond=None)[0][3])
    gbox_empirical_p = (1 + int(np.sum(gbox_null >= interaction_beta))) / (
        args.permutations + 1
    )
    gbox_null_path = analysis_root / "exact_gbox_interaction_permutation_null.tsv.gz"
    pd.DataFrame(
        {"permutation": np.arange(1, len(gbox_null) + 1), "interaction_beta": gbox_null}
    ).to_csv(gbox_null_path, sep="\t", index=False, compression=deterministic_gzip)

    summary = {
        "status": "complete_supportive" if observed > 0 and empirical_p <= 0.05 else "complete_nonsupportive",
        "scope": "posthoc_gse141983_raw_continuous_h3k4me3_sensitivity",
        "posthoc": True,
        "primary_gse190586_reclassified": False,
        "pyrus_primary_mechanism_rescued": False,
        "dapt_model_selection_allowed": False,
        "malus_accessed": False,
        "technical_gate": "pass",
        "label_mapping_gate": {
            "positive_mapping_fraction": positive_mapping_fraction,
            "negative_mapping_fraction": negative_mapping_fraction,
            "threshold": 0.70,
            "pass": True,
        },
        "continuous_label_chromatin": {
            "statistic": observed,
            "bootstrap_95_interval": bootstrap_interval,
            "permutation_null_mean": float(null.mean()),
            "permutation_null_95_interval": np.quantile(null, [0.025, 0.975]).tolist(),
            "one_sided_empirical_p": empirical_p,
            "within_family_bh_q": empirical_p,
            "permutations": args.permutations,
            "bootstraps": args.bootstraps,
            "permutation_seed": args.permutation_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "positive_genes": mapped_positive,
            "negative_genes": mapped_negative,
        },
        "exact_gbox_interaction": {
            "interaction_beta": interaction_beta,
            "hc3_standard_error": interaction_se,
            "hc3_two_sided_p": interaction_hc3_p,
            "one_sided_permutation_p": gbox_empirical_p,
            "within_family_bh_q": gbox_empirical_p,
            "permutations": args.permutations,
            "permutation_seed": args.gbox_permutation_seed,
            "exact_gbox_positive_genes": int(
                ((joined["label"] == "positive") & joined["exact_gbox"]).sum()
            ),
            "exact_gbox_negative_genes": int(
                ((joined["label"] == "negative") & joined["exact_gbox"]).sum()
            ),
        },
        "model_chromatin_integration": "pending_frozen_model_attribution_then_run_or_omit",
        "output_sha256": {
            str(technical_gate_path.relative_to(root)): sha256_file(technical_gate_path),
            str(gene_results_path.relative_to(root)): sha256_file(gene_results_path),
            str(null_path.relative_to(root)): sha256_file(null_path),
            str(bootstrap_path.relative_to(root)): sha256_file(bootstrap_path),
            str(gbox_null_path.relative_to(root)): sha256_file(gbox_null_path),
        },
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
