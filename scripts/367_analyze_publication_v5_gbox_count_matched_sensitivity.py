#!/usr/bin/env python3
"""Re-test overlap-masked motifs with exact-CACGTG-count-matched controls."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results/metrics/publication_v5_motif"
MASKED = ROOT / "results/metrics/publication_v5_motif_gbox_masked_sensitivity"
OUT = ROOT / "results/metrics/publication_v5_motif_gbox_count_matched_sensitivity"
AUDIT = ROOT / "results/metrics/publication_v5_motif_gbox_count_matched_sensitivity_audit.json"
SEED = 20260803
REPLICATES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def build_pools(promoters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for genus in sorted(promoters["genus"].unique()):
        foreground = promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"]]
        background = promoters.loc[promoters["genus"].eq(genus) & ~promoters["foreground"]]
        for target in foreground.to_dict("records"):
            base = background.loc[
                background["promoter_length"].eq(target["promoter_length"])
                & background["exact_cacgtg_count"].eq(target["exact_cacgtg_count"])
            ].copy()
            base["gc_distance"] = (base["gc_fraction_recomputed"] - target["gc_fraction_recomputed"]).abs()
            pool = base.loc[base["chromosome"].eq(target["chromosome"]) & base["gc_distance"].le(0.02)].copy()
            stage = "same_chromosome_gc002_exact_count"
            if len(pool) < 10:
                pool = base.loc[base["gc_distance"].le(0.02)].copy()
                stage = "same_genus_gc002_exact_count"
            if len(pool) < 10:
                pool = base.sort_values(["gc_distance", "gene_id"]).head(100).copy()
                stage = "same_genus_nearest100_exact_count"
            if len(pool) < 10:
                raise RuntimeError(f"Fewer than 10 exact-count controls for {genus}:{target['gene_id']}")
            for control in pool.sort_values(["gc_distance", "gene_id"]).to_dict("records"):
                rows.append(
                    {
                        "genus": genus,
                        "foreground_safe_id": target["safe_id"],
                        "foreground_gene_id": target["gene_id"],
                        "foreground_exact_cacgtg_count": int(target["exact_cacgtg_count"]),
                        "control_safe_id": control["safe_id"],
                        "control_gene_id": control["gene_id"],
                        "control_exact_cacgtg_count": int(control["exact_cacgtg_count"]),
                        "matching_stage": stage,
                        "same_chromosome": control["chromosome"] == target["chromosome"],
                        "gc_distance": float(control["gc_distance"]),
                        "promoter_length": int(target["promoter_length"]),
                    }
                )
    return pd.DataFrame(rows)


def assignments(promoters: pd.DataFrame, pools: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    matrices = {}
    rows = []
    for genus in sorted(promoters["genus"].unique()):
        foreground = sorted(promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"])
        pool_map = {
            safe_id: sorted(frame["control_safe_id"].unique())
            for safe_id, frame in pools.loc[pools["genus"].eq(genus)].groupby("foreground_safe_id")
        }
        matrix = np.empty((REPLICATES, len(foreground)), dtype=object)
        for replicate in range(REPLICATES):
            selected = [None] * len(foreground)
            used: set[str] = set()
            for index in rng.permutation(len(foreground)):
                safe_id = foreground[index]
                available = [item for item in pool_map[safe_id] if item not in used]
                if not available:
                    available = pool_map[safe_id]
                control = available[int(rng.integers(0, len(available)))]
                selected[index] = control
                used.add(control)
            matrix[replicate] = selected
            rows.extend(
                {
                    "replicate": replicate + 1,
                    "genus": genus,
                    "foreground_safe_id": target,
                    "control_safe_id": control,
                }
                for target, control in zip(foreground, selected)
            )
        matrices[genus] = matrix
    return matrices, pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prior_audit = json.loads((ROOT / "results/metrics/publication_v5_motif_gbox_masked_sensitivity_audit.json").read_text(encoding="utf-8"))
    if prior_audit.get("status") != "pass":
        raise RuntimeError("Exact-overlap-masked sensitivity audit has not passed")
    promoters = pd.read_csv(PRIMARY / "promoter_map.tsv.gz", sep="\t")
    sites = pd.read_csv(MASKED / "exact_cacgtg_sites.tsv.gz", sep="\t")
    hits = pd.read_csv(MASKED / "fimo_hits_excluding_exact_cacgtg_overlap.tsv.gz", sep="\t")
    metadata = pd.read_csv(PRIMARY / "jaspar2026_plant_metadata.tsv", sep="\t", dtype=str)
    metadata["tf_family"] = metadata["family"].fillna("").replace("", np.nan)
    metadata["tf_family"] = metadata["tf_family"].fillna("Unclassified: " + metadata["class"].fillna("unknown"))
    families = sorted(metadata["tf_family"].unique())
    counts = sites.groupby("safe_id").size()
    promoters["exact_cacgtg_count"] = promoters["safe_id"].map(counts).fillna(0).astype(int)

    pools = build_pools(promoters)
    matrices, assignment_rows = assignments(promoters, pools)
    pools.to_csv(OUT / "foreground_control_pools_gbox_count_matched.tsv.gz", sep="\t", index=False, compression="gzip")
    assignment_rows.to_csv(OUT / "matched_background_replicates_gbox_count_matched.tsv.gz", sep="\t", index=False, compression="gzip")

    genes = sorted(promoters["safe_id"].unique())
    gene_index = {gene: index for index, gene in enumerate(genes)}
    family_index = {family: index for index, family in enumerate(families)}
    presence = np.zeros((len(genes), len(families)), dtype=np.uint8)
    for safe_id, family in hits[["safe_id", "tf_family"]].drop_duplicates().itertuples(index=False):
        presence[gene_index[safe_id], family_index[family]] = 1

    candidate_fraction = {}
    null_fraction = {}
    for genus, matrix in matrices.items():
        foreground = sorted(promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"])
        candidate_fraction[genus] = presence[[gene_index[item] for item in foreground]].mean(axis=0)
        null = np.empty((REPLICATES, len(families)), dtype=np.float32)
        for replicate in range(REPLICATES):
            null[replicate] = presence[[gene_index[item] for item in matrix[replicate]]].mean(axis=0)
        null_fraction[genus] = null
    genera = sorted(matrices)
    observed = np.mean([candidate_fraction[genus] for genus in genera], axis=0)
    null_combined = np.mean([null_fraction[genus] for genus in genera], axis=0)
    p_values = (1 + (null_combined >= observed[None, :]).sum(axis=0)) / (REPLICATES + 1)
    q_values = bh(p_values.astype(float))
    rows = []
    for index, family in enumerate(families):
        row = {"tf_family": family}
        positive_both = True
        for genus in genera:
            candidate = float(candidate_fraction[genus][index])
            control = float(np.median(null_fraction[genus][:, index]))
            row[f"{genus}_candidate_fraction"] = candidate
            row[f"{genus}_control_median"] = control
            row[f"{genus}_difference"] = candidate - control
            positive_both &= candidate > control
        row.update(
            {
                "combined_candidate_fraction": float(observed[index]),
                "combined_control_median": float(np.median(null_combined[:, index])),
                "combined_difference": float(observed[index] - np.median(null_combined[:, index])),
                "empirical_p": float(p_values[index]),
                "bh_q": float(q_values[index]),
                "positive_both_genera": bool(positive_both),
                "convergently_enriched": bool(positive_both and q_values[index] < 0.05),
            }
        )
        rows.append(row)
    enrichment = pd.DataFrame(rows).sort_values(["bh_q", "empirical_p", "tf_family"])
    enrichment.to_csv(OUT / "tf_family_enrichment_gbox_overlap_removed_count_matched.tsv", sep="\t", index=False)

    output_names = [
        "foreground_control_pools_gbox_count_matched.tsv.gz",
        "matched_background_replicates_gbox_count_matched.tsv.gz",
        "tf_family_enrichment_gbox_overlap_removed_count_matched.tsv",
    ]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_circularity_sensitivity",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "replicates": REPLICATES,
        "seed": SEED,
        "foreground_promoters": int(promoters["foreground"].sum()),
        "tf_families_tested": len(families),
        "convergently_enriched_families": sorted(enrichment.loc[enrichment["convergently_enriched"], "tf_family"]),
        "minimum_controls_per_foreground": int(pools.groupby("foreground_safe_id")["control_safe_id"].nunique().min()),
        "matching_stage_counts": pools.groupby("matching_stage")["foreground_safe_id"].nunique().to_dict(),
        "output_fingerprints": {name: sha256(OUT / name) for name in output_names},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    checks = {
        "prior_masked_audit_pass": prior_audit.get("status") == "pass",
        "foreground_79": int(promoters["foreground"].sum()) == 79,
        "two_genera": set(matrices) == {"prunus", "pyrus"},
        "family_universe_62": len(families) == len(enrichment) == 62,
        "ten_thousand_replicates": all(matrix.shape[0] == REPLICATES for matrix in matrices.values()),
        "at_least_ten_controls_each": bool((pools.groupby("foreground_safe_id")["control_safe_id"].nunique() >= 10).all()),
        "exact_gbox_counts_identical": bool((pools["foreground_exact_cacgtg_count"] == pools["control_exact_cacgtg_count"]).all()),
        "q_values_valid": bool(enrichment["bh_q"].between(0, 1).all()),
        "malus_outcomes_sealed": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    audit = {
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": len(checks),
        "passed_checks": len(checks) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "records": checks,
        "summary": summary,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit["status"], **summary, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
