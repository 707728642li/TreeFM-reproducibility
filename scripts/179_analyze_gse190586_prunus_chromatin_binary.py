#!/usr/bin/env python3
"""Run the frozen exploratory GSE190586 binary H3K4me3 sensitivity."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PERMUTATION_SEED = 20260717
BOOTSTRAP_SEED = 20260718
PERMUTATIONS = 10_000
BOOTSTRAPS = 2_000
GENE_RE = re.compile(r"^(?:PRUPE_|Prupe\.)([1-8])G(\d{6})$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_gene(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    match = GENE_RE.fullmatch(str(value).strip())
    if not match:
        return None
    return f"Prupe.{match.group(1)}G{match.group(2)}"


def verify_freeze(root: Path) -> dict[str, object]:
    path = root / "config/publication_v3_prunus_chromatin_binary_implementation_freeze.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("malus_accessed") is not False:
        raise RuntimeError("implementation freeze is not valid")
    for relative, expected in manifest["artifact_sha256"].items():
        actual = sha256(root / relative)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {relative}: {actual} != {expected}")
    return manifest


def read_direction_sheet(path: Path, sheet: str, direction: int) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_excel(path, sheet_name=sheet, header=1, engine="openpyxl")
    required = {"Distance to TSS", "Entrez ID", "Gene Name", "Peak Score"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise RuntimeError(f"{sheet} is missing columns: {missing}")
    distance = pd.to_numeric(raw["Distance to TSS"], errors="coerce")
    raw = raw.loc[distance.between(-100, 1000, inclusive="both")].copy()
    raw["entrez_gene"] = raw["Entrez ID"].map(canonical_gene)
    raw["name_gene"] = raw["Gene Name"].map(canonical_gene)
    conflicting = (
        raw["entrez_gene"].notna()
        & raw["name_gene"].notna()
        & raw["entrez_gene"].ne(raw["name_gene"])
    )
    accepted = raw.loc[~conflicting].copy()
    accepted["gene_id"] = accepted["entrez_gene"].fillna(accepted["name_gene"])
    accepted = accepted.loc[accepted["gene_id"].notna()].copy()
    accepted["call"] = direction
    accepted["peak_score"] = pd.to_numeric(accepted["Peak Score"], errors="coerce")
    accepted["distance_to_tss"] = pd.to_numeric(accepted["Distance to TSS"], errors="coerce")
    stats = {
        "input_rows": int(len(distance)),
        "promoter_rows": int(len(raw)),
        "conflicting_identifier_rows": int(conflicting.sum()),
        "accepted_rows": int(len(accepted)),
        "accepted_unique_genes": int(accepted["gene_id"].nunique()),
    }
    return accepted[["gene_id", "call", "peak_score", "distance_to_tss"]], stats


def aggregate_calls(up: pd.DataFrame, down: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    combined = pd.concat([up, down], ignore_index=True)
    grouped = combined.groupby("gene_id", sort=True)
    records: list[dict[str, object]] = []
    ambiguous = 0
    for gene, frame in grouped:
        calls = sorted(set(int(value) for value in frame["call"]))
        if len(calls) != 1:
            ambiguous += 1
            continue
        call = calls[0]
        records.append(
            {
                "gene_id": gene,
                "chromatin_call": call,
                "eligible_peak_count": int(len(frame)),
                "max_peak_score": float(frame["peak_score"].max()),
                "min_abs_distance_to_tss": float(frame["distance_to_tss"].abs().min()),
            }
        )
    output = pd.DataFrame.from_records(records)
    return output, {
        "genes_with_any_promoter_call": int(combined["gene_id"].nunique()),
        "ambiguous_bidirectional_genes_excluded": int(ambiguous),
        "unambiguous_genes": int(len(output)),
    }


def expression_rank_score(root: Path, genes: Iterable[str]) -> tuple[pd.Series, dict[str, object]]:
    genes = list(genes)
    study_scores: list[pd.Series] = []
    diagnostics: dict[str, object] = {}
    interim = root / "data/interim/functional_v3/Prunus_publication_v3"
    for accession in ("GSE130426", "GSE138792", "GSE298924"):
        counts = pd.read_csv(interim / f"{accession}_selected_counts.tsv.gz", sep="\t")
        design = pd.read_csv(interim / f"{accession}_selected_design.tsv", sep="\t")
        if counts.columns[0] != "gene_id" or counts["gene_id"].duplicated().any():
            raise RuntimeError(f"invalid {accession} selected count table")
        baseline_samples = design.loc[design["condition"].eq("baseline"), "sample_id"].tolist()
        if not baseline_samples or not set(baseline_samples).issubset(counts.columns):
            raise RuntimeError(f"invalid {accession} baseline design")
        matrix = counts.set_index("gene_id")[baseline_samples].astype(np.float64)
        library_sizes = matrix.sum(axis=0)
        if (library_sizes <= 0).any():
            raise RuntimeError(f"nonpositive {accession} library size")
        log_cpm = np.log2(matrix.divide(library_sizes, axis=1) * 1_000_000.0 + 0.5)
        baseline = log_cpm.median(axis=1).reindex(genes)
        ranked = baseline.rank(method="average", pct=True)
        ranked.name = accession
        study_scores.append(ranked)
        diagnostics[accession] = {
            "baseline_libraries": len(baseline_samples),
            "label_genes_present": int(baseline.notna().sum()),
            "label_genes_missing": int(baseline.isna().sum()),
            "library_size_min": int(library_sizes.min()),
            "library_size_max": int(library_sizes.max()),
        }
    scores = pd.concat(study_scores, axis=1)
    return scores.mean(axis=1, skipna=False), diagnostics


def ten_equal_rank_bins(values: pd.Series, gene_ids: pd.Series) -> pd.Series:
    order = pd.DataFrame({"value": values.to_numpy(), "gene_id": gene_ids.to_numpy()})
    order = order.sort_values(["value", "gene_id"], kind="mergesort").reset_index()
    bins = np.floor(np.arange(len(order)) * 10 / len(order)).astype(int)
    result = pd.Series(index=order["index"], data=bins, dtype=int)
    return result.sort_index()


def permuted_statistic(frame: pd.DataFrame, rng: np.random.Generator) -> float:
    assigned = np.empty(len(frame), dtype=object)
    categories = frame["category"].to_numpy(dtype=object)
    for indices in frame.groupby("stratum", sort=True).indices.values():
        indices = np.asarray(indices, dtype=int)
        assigned[indices] = rng.permutation(categories[indices])
    calls = frame["chromatin_call"].to_numpy(dtype=int)
    positives = (assigned == "positive_up") | (assigned == "positive_down")
    concordant = ((assigned == "positive_up") & (calls == 1)) | (
        (assigned == "positive_down") & (calls == -1)
    )
    return float(concordant.sum() / positives.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    freeze = verify_freeze(root)

    labels = pd.read_parquet(
        root / "data/processed/functional/Prunus_publication_v3/promoter_labels.parquet"
    ).copy()
    if labels["gene_id"].duplicated().any() or set(labels["label"]) != {"positive", "negative"}:
        raise RuntimeError("invalid frozen Prunus label table")
    labels["category"] = np.where(
        labels["label"].eq("negative"),
        "negative",
        "positive_" + labels["endpoint_direction"].astype(str),
    )
    expected_categories = {"positive_up", "positive_down", "negative"}
    if set(labels["category"]) != expected_categories:
        raise RuntimeError("unexpected frozen label categories")
    invalid_label_ids = int(labels["gene_id"].map(canonical_gene).isna().sum())
    if invalid_label_ids:
        raise RuntimeError(f"{invalid_label_ids} frozen label IDs do not map canonically")

    workbook = root / "data/raw/functional_v3/GSE190586/NPH-236-974-s005.xlsx"
    endpoint, endpoint_stats = read_direction_sheet(
        workbook, "0CU vs 770CU_770CU", 1
    )
    baseline, baseline_stats = read_direction_sheet(
        workbook, "0CU vs 770CU_0CU ", -1
    )
    calls, aggregate_stats = aggregate_calls(endpoint, baseline)

    labels["gc_fraction"] = labels["promoter_2048"].str.upper().map(
        lambda sequence: (
            (sequence.count("G") + sequence.count("C"))
            / sum(sequence.count(base) for base in "ACGT")
        )
    )
    expression_score, expression_diagnostics = expression_rank_score(root, labels["gene_id"])
    labels["baseline_expression_rank_score"] = labels["gene_id"].map(expression_score)
    if labels[["gc_fraction", "baseline_expression_rank_score"]].isna().any().any():
        raise RuntimeError("missing frozen nuisance-stratum variable")
    labels = labels.sort_values("gene_id", kind="mergesort").reset_index(drop=True)
    labels["expression_decile"] = ten_equal_rank_bins(
        labels["baseline_expression_rank_score"], labels["gene_id"]
    )
    labels["promoter_gc_decile"] = ten_equal_rank_bins(labels["gc_fraction"], labels["gene_id"])
    labels["stratum"] = (
        labels["expression_decile"].astype(str) + "|" + labels["promoter_gc_decile"].astype(str)
    )
    frame = labels.merge(calls, on="gene_id", how="left", validate="one_to_one")
    frame["chromatin_call"] = frame["chromatin_call"].fillna(0).astype(int)
    frame["eligible_peak_count"] = frame["eligible_peak_count"].fillna(0).astype(int)
    frame["concordant_positive"] = (
        (frame["category"].eq("positive_up") & frame["chromatin_call"].eq(1))
        | (frame["category"].eq("positive_down") & frame["chromatin_call"].eq(-1))
    )
    positive = frame["category"].ne("negative")
    observed = float(frame.loc[positive, "concordant_positive"].mean())

    permutation_rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.fromiter(
        (permuted_statistic(frame, permutation_rng) for _ in range(PERMUTATIONS)),
        dtype=np.float64,
        count=PERMUTATIONS,
    )
    empirical_p = float((1 + np.count_nonzero(null >= observed)) / (PERMUTATIONS + 1))

    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    up_values = frame.loc[frame["category"].eq("positive_up"), "concordant_positive"].to_numpy(float)
    down_values = frame.loc[frame["category"].eq("positive_down"), "concordant_positive"].to_numpy(float)
    bootstrap = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        up_sample = bootstrap_rng.choice(up_values, size=len(up_values), replace=True)
        down_sample = bootstrap_rng.choice(down_values, size=len(down_values), replace=True)
        bootstrap[index] = float((up_sample.sum() + down_sample.sum()) / (len(up_sample) + len(down_sample)))
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    output_root = root / "results/biological_cases/prunus_publication_v3_chromatin_replication"
    output_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "gene_id", "category", "label", "endpoint_direction", "chromosome", "split",
        "baseline_expression_rank_score", "gc_fraction", "expression_decile",
        "promoter_gc_decile", "stratum", "chromatin_call", "concordant_positive",
        "eligible_peak_count", "max_peak_score", "min_abs_distance_to_tss",
    ]
    frame[columns].to_csv(
        output_root / "gse190586_binary_gene_calls.tsv.gz",
        sep="\t", index=False, compression="gzip", na_rep="NA",
    )
    pd.DataFrame({"permutation": np.arange(1, PERMUTATIONS + 1), "null_fraction": null}).to_csv(
        output_root / "gse190586_binary_permutation_null.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )
    pd.DataFrame({"bootstrap": np.arange(1, BOOTSTRAPS + 1), "fraction": bootstrap}).to_csv(
        output_root / "gse190586_binary_bootstrap.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )

    category_counts = frame.groupby("category", sort=True).agg(
        genes=("gene_id", "size"),
        plus_calls=("chromatin_call", lambda x: int((x == 1).sum())),
        minus_calls=("chromatin_call", lambda x: int((x == -1).sum())),
        no_calls=("chromatin_call", lambda x: int((x == 0).sum())),
        concordant_positive=("concordant_positive", "sum"),
    )
    summary = {
        "status": "complete_supportive" if observed > 0 and empirical_p <= 0.05 else "complete_nonsupportive",
        "scope": "exploratory_supplementary_binary_sensitivity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accession": "GSE190586",
        "endpoint": "H3K4me3_770_minus_0_chilling_units",
        "malus_accessed": False,
        "pyrus_rescue_allowed": False,
        "continuous_analysis": "omitted_missing_adjusted_p_se_and_continuous_effect",
        "binary_analysis": {
            "observed_concordant_positive_fraction": observed,
            "concordant_positive_genes": int(frame.loc[positive, "concordant_positive"].sum()),
            "positive_genes": int(positive.sum()),
            "bootstrap_95_interval": [float(ci_low), float(ci_high)],
            "permutation_null_mean": float(null.mean()),
            "permutation_null_95_interval": [float(x) for x in np.quantile(null, [0.025, 0.975])],
            "one_sided_empirical_p": empirical_p,
            "within_family_bh_q": empirical_p,
            "permutations": PERMUTATIONS,
            "bootstraps": BOOTSTRAPS,
            "permutation_seed": PERMUTATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "mapping_gate": {
            "frozen_positive_mapping_fraction": 1.0,
            "frozen_negative_mapping_fraction": 1.0,
            "gate_threshold": 0.70,
            "pass": True,
            "basis": "same_PrUNUS_persica_v2_canonical_gene_space_and_anchored_one_to_one_ID_transform",
        },
        "endpoint_sheet": endpoint_stats,
        "baseline_sheet": baseline_stats,
        "gene_call_aggregation": aggregate_stats,
        "expression_strata": expression_diagnostics,
        "strata": int(frame["stratum"].nunique()),
        "category_counts": category_counts.reset_index().to_dict(orient="records"),
        "implementation_freeze_input_fingerprint": freeze["input_fingerprint"],
        "output_files": {
            "gene_calls": "gse190586_binary_gene_calls.tsv.gz",
            "permutation_null": "gse190586_binary_permutation_null.tsv.gz",
            "bootstrap": "gse190586_binary_bootstrap.tsv.gz",
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

