#!/usr/bin/env python3
"""Post-hoc chromosome robustness checks for the Prunus chromatin result."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PERMUTATIONS = 10_000
CONDITIONED_SEED = 20260719
JACKKNIFE_SEED = 20260720


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze(root: Path) -> dict[str, object]:
    path = root / "config/publication_v3_prunus_chromatin_posthoc_chromosome_freeze.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or not manifest.get("posthoc", False):
        raise RuntimeError("post-hoc chromosome freeze is invalid")
    for relative, expected in manifest["artifact_sha256"].items():
        actual = sha256(root / relative)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {relative}: {actual} != {expected}")
    return manifest


def equal_rank_bins(values: pd.Series, genes: pd.Series, bins: int) -> pd.Series:
    order = pd.DataFrame({"value": values.to_numpy(), "gene_id": genes.to_numpy()})
    order = order.sort_values(["value", "gene_id"], kind="mergesort").reset_index()
    assignments = np.floor(np.arange(len(order)) * bins / len(order)).astype(int)
    output = pd.Series(index=order["index"], data=assignments, dtype=int)
    return output.sort_index()


def observed_statistic(frame: pd.DataFrame) -> float:
    categories = frame["category"].to_numpy(dtype=object)
    calls = frame["chromatin_call"].to_numpy(dtype=int)
    positive = (categories == "positive_up") | (categories == "positive_down")
    concordant = ((categories == "positive_up") & (calls == 1)) | (
        (categories == "positive_down") & (calls == -1)
    )
    return float(concordant.sum() / positive.sum())


def permutation_null(frame: pd.DataFrame, stratum_column: str, seed: int) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    categories = frame["category"].to_numpy(dtype=object)
    calls = frame["chromatin_call"].to_numpy(dtype=int)
    groups = [np.asarray(indices, dtype=int) for indices in frame.groupby(stratum_column, sort=True).indices.values()]
    positive_total = int(np.isin(categories, ["positive_up", "positive_down"]).sum())
    rng = np.random.default_rng(seed)
    null = np.empty(PERMUTATIONS, dtype=np.float64)
    assigned = np.empty(len(frame), dtype=object)
    for iteration in range(PERMUTATIONS):
        for indices in groups:
            assigned[indices] = rng.permutation(categories[indices])
        concordant = ((assigned == "positive_up") & (calls == 1)) | (
            (assigned == "positive_down") & (calls == -1)
        )
        null[iteration] = float(concordant.sum() / positive_total)
    return null


def summarize(frame: pd.DataFrame, stratum: str, seed: int) -> tuple[dict[str, object], np.ndarray]:
    observed = observed_statistic(frame)
    null = permutation_null(frame, stratum, seed)
    empirical_p = float((1 + np.count_nonzero(null >= observed)) / (PERMUTATIONS + 1))
    summary = {
        "genes": int(len(frame)),
        "positive_genes": int(frame["category"].ne("negative").sum()),
        "concordant_positive_genes": int(round(observed * frame["category"].ne("negative").sum())),
        "strata": int(frame[stratum].nunique()),
        "observed_fraction": observed,
        "null_mean": float(null.mean()),
        "null_95_interval": [float(value) for value in np.quantile(null, [0.025, 0.975])],
        "enrichment_ratio": float(observed / null.mean()),
        "absolute_excess_fraction": float(observed - null.mean()),
        "one_sided_empirical_p": empirical_p,
        "seed": seed,
    }
    return summary, null


def jackknife_task(payload: tuple[str, pd.DataFrame, int]) -> tuple[str, dict[str, object], np.ndarray]:
    chromosome, frame, seed = payload
    retained = frame.loc[frame["chromosome"].ne(chromosome)].copy()
    summary, null = summarize(retained, "stratum", seed)
    summary["excluded_chromosome"] = chromosome
    return chromosome, summary, null


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    freeze = verify_freeze(root)
    result_root = root / "results/biological_cases/prunus_publication_v3_chromatin_replication"
    gene_path = result_root / "gse190586_binary_gene_calls.tsv.gz"
    frame = pd.read_csv(gene_path, sep="\t")
    required = {
        "gene_id", "chromosome", "category", "chromatin_call", "stratum",
        "baseline_expression_rank_score", "gc_fraction",
    }
    if not required.issubset(frame.columns) or frame["gene_id"].duplicated().any():
        raise RuntimeError("invalid frozen gene-call input")
    chromosomes = sorted(frame["chromosome"].unique())
    if chromosomes != [f"Pp{index:02d}" for index in range(1, 9)]:
        raise RuntimeError(f"unexpected chromosome set: {chromosomes}")

    frame = frame.sort_values("gene_id", kind="mergesort").reset_index(drop=True)
    frame["expression_tertile"] = equal_rank_bins(
        frame["baseline_expression_rank_score"], frame["gene_id"], 3
    )
    frame["gc_tertile"] = equal_rank_bins(frame["gc_fraction"], frame["gene_id"], 3)
    frame["chromosome_conditioned_stratum"] = (
        frame["chromosome"].astype(str) + "|"
        + frame["expression_tertile"].astype(str) + "|"
        + frame["gc_tertile"].astype(str)
    )
    conditioned, conditioned_null = summarize(
        frame, "chromosome_conditioned_stratum", CONDITIONED_SEED
    )

    payloads = [
        (chromosome, frame, JACKKNIFE_SEED + index)
        for index, chromosome in enumerate(chromosomes)
    ]
    jackknife: list[dict[str, object]] = []
    jackknife_nulls: list[pd.DataFrame] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, 8)) as executor:
        for chromosome, summary, null in executor.map(jackknife_task, payloads):
            jackknife.append(summary)
            jackknife_nulls.append(
                pd.DataFrame(
                    {
                        "excluded_chromosome": chromosome,
                        "permutation": np.arange(1, PERMUTATIONS + 1),
                        "null_fraction": null,
                    }
                )
            )
    jackknife.sort(key=lambda item: str(item["excluded_chromosome"]))
    chromosome_counts = []
    for chromosome, group in frame.groupby("chromosome", sort=True):
        positive = group["category"].ne("negative")
        concordant = (
            (group["category"].eq("positive_up") & group["chromatin_call"].eq(1))
            | (group["category"].eq("positive_down") & group["chromatin_call"].eq(-1))
        )
        chromosome_counts.append(
            {
                "chromosome": chromosome,
                "genes": int(len(group)),
                "positive_genes": int(positive.sum()),
                "concordant_positive_genes": int(concordant.sum()),
                "concordant_fraction": float(concordant.sum() / positive.sum()),
            }
        )

    robustness_pass = bool(
        conditioned["observed_fraction"] > conditioned["null_mean"]
        and all(item["observed_fraction"] > item["null_mean"] for item in jackknife)
    )
    output = {
        "status": "pass" if robustness_pass else "nonsupportive",
        "scope": "posthoc_chromosome_robustness_sensitivity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "posthoc": True,
        "primary_result_reclassified": False,
        "malus_accessed": False,
        "conditioned_permutation": conditioned,
        "leave_one_chromosome_out": jackknife,
        "chromosome_counts": chromosome_counts,
        "descriptive_robustness_rule": (
            "conditioned observed>null mean and every leave-one-chromosome-out observed>null mean"
        ),
        "descriptive_robustness_pass": robustness_pass,
        "freeze_input_fingerprint": freeze["input_fingerprint"],
    }
    summary_path = result_root / "posthoc_chromosome_robustness_summary.json"
    summary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "permutation": np.arange(1, PERMUTATIONS + 1),
            "null_fraction": conditioned_null,
        }
    ).to_csv(
        result_root / "posthoc_chromosome_conditioned_null.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )
    pd.concat(jackknife_nulls, ignore_index=True).to_csv(
        result_root / "posthoc_leave_one_chromosome_nulls.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

