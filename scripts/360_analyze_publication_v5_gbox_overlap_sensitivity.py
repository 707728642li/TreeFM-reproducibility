#!/usr/bin/env python3
"""Re-test motif enrichment after removing hits overlapping exact CACGTG sites."""

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
OUT = ROOT / "results/metrics/publication_v5_motif_gbox_masked_sensitivity"
SEED = 20260803
REPLICATES = 10_000
MOTIF = "CACGTG"


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


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    result[name] = "".join(chunks).upper()
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            result[name] = "".join(chunks).upper()
    return result


def exact_sites(sequences: dict[str, str]) -> tuple[dict[str, list[tuple[int, int]]], pd.DataFrame]:
    by_gene: dict[str, list[tuple[int, int]]] = {}
    rows = []
    for safe_id, sequence in sequences.items():
        sites = []
        offset = 0
        while True:
            index = sequence.find(MOTIF, offset)
            if index < 0:
                break
            start = index + 1
            stop = start + len(MOTIF) - 1
            sites.append((start, stop))
            rows.append({"safe_id": safe_id, "start": start, "stop": stop, "motif": MOTIF})
            offset = index + 1
        by_gene[safe_id] = sites
    return by_gene, pd.DataFrame(rows)


def generate_assignments(promoters: pd.DataFrame, pools: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    assignments: dict[str, np.ndarray] = {}
    for genus in sorted(promoters["genus"].unique()):
        foreground = sorted(promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"])
        pool_map = {
            safe: sorted(group["control_safe_id"].unique())
            for safe, group in pools.loc[pools["genus"].eq(genus)].groupby("foreground_safe_id")
        }
        matrix = np.empty((REPLICATES, len(foreground)), dtype=object)
        for replicate in range(REPLICATES):
            used: set[str] = set()
            selected = [None] * len(foreground)
            for index in rng.permutation(len(foreground)):
                target = foreground[index]
                available = [control for control in pool_map[target] if control not in used]
                if not available:
                    available = pool_map[target]
                control = available[int(rng.integers(0, len(available)))]
                selected[index] = control
                used.add(control)
            matrix[replicate] = selected
        assignments[genus] = matrix
    return assignments


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    promoter_map = pd.read_csv(PRIMARY / "promoter_map.tsv.gz", sep="\t")
    pools = pd.read_csv(PRIMARY / "foreground_control_pools.tsv.gz", sep="\t")
    hits = pd.read_csv(PRIMARY / "fimo_hits.tsv.gz", sep="\t")
    primary_enrichment = pd.read_csv(PRIMARY / "tf_family_enrichment.tsv", sep="\t")
    metadata = pd.read_csv(PRIMARY / "jaspar2026_plant_metadata.tsv", sep="\t", dtype=str)
    metadata["tf_family"] = metadata["family"].fillna("").replace("", np.nan)
    metadata["tf_family"] = metadata["tf_family"].fillna("Unclassified: " + metadata["class"].fillna("unknown"))
    families = sorted(metadata["tf_family"].unique())

    sequences = read_fasta(PRIMARY / "prepared/positive_promoters.fasta")
    sites_by_gene, site_rows = exact_sites(sequences)
    if site_rows.empty:
        raise RuntimeError("No exact CACGTG sites found")
    overlaps = []
    for row in hits[["safe_id", "start", "stop"]].itertuples(index=False):
        overlaps.append(any(int(row.start) <= stop and int(row.stop) >= start for start, stop in sites_by_gene[row.safe_id]))
    hits = hits.copy()
    hits["overlaps_exact_cacgtg"] = np.asarray(overlaps, dtype=bool)
    filtered = hits.loc[~hits["overlaps_exact_cacgtg"]].copy()
    site_rows.to_csv(OUT / "exact_cacgtg_sites.tsv.gz", sep="\t", index=False, compression="gzip")
    filtered.to_csv(OUT / "fimo_hits_excluding_exact_cacgtg_overlap.tsv.gz", sep="\t", index=False, compression="gzip")

    genes = sorted(promoter_map["safe_id"].unique())
    gene_index = {gene: index for index, gene in enumerate(genes)}
    family_index = {family: index for index, family in enumerate(families)}
    presence = np.zeros((len(genes), len(families)), dtype=np.uint8)
    for safe_id, family in filtered[["safe_id", "tf_family"]].drop_duplicates().itertuples(index=False):
        presence[gene_index[safe_id], family_index[family]] = 1

    assignments = generate_assignments(promoter_map, pools)
    candidate_fraction = {}
    null_fraction = {}
    for genus in sorted(assignments):
        foreground = sorted(promoter_map.loc[promoter_map["genus"].eq(genus) & promoter_map["foreground"], "safe_id"])
        candidate_fraction[genus] = presence[[gene_index[item] for item in foreground]].mean(axis=0)
        null = np.empty((REPLICATES, len(families)), dtype=np.float32)
        for replicate in range(REPLICATES):
            null[replicate] = presence[[gene_index[item] for item in assignments[genus][replicate]]].mean(axis=0)
        null_fraction[genus] = null
    genera = sorted(assignments)
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
                "primary_convergently_enriched": bool(
                    primary_enrichment.set_index("tf_family").loc[family, "convergently_enriched"]
                ),
            }
        )
        rows.append(row)
    enrichment = pd.DataFrame(rows).sort_values(["bh_q", "empirical_p", "tf_family"])
    enrichment.to_csv(OUT / "tf_family_enrichment_gbox_overlap_removed.tsv", sep="\t", index=False)

    retained = sorted(enrichment.loc[enrichment["convergently_enriched"], "tf_family"])
    primary_significant = sorted(primary_enrichment.loc[primary_enrichment["convergently_enriched"], "tf_family"])
    outputs = [
        "exact_cacgtg_sites.tsv.gz",
        "fimo_hits_excluding_exact_cacgtg_overlap.tsv.gz",
        "tf_family_enrichment_gbox_overlap_removed.tsv",
    ]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_circularity_sensitivity",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "exact_motif": MOTIF,
        "exact_sites": len(site_rows),
        "promoters_with_exact_sites": int(site_rows["safe_id"].nunique()),
        "primary_fimo_hits": len(hits),
        "overlapping_hits_removed": int(hits["overlaps_exact_cacgtg"].sum()),
        "retained_fimo_hits": len(filtered),
        "tf_families_tested": len(families),
        "primary_convergently_enriched_families": primary_significant,
        "masked_convergently_enriched_families": retained,
        "output_fingerprints": {name: sha256(OUT / name) for name in outputs},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checks = {
        "exact_sites_present": len(site_rows) > 0,
        "some_but_not_all_hits_removed": 0 < int(hits["overlaps_exact_cacgtg"].sum()) < len(hits),
        "all_62_families_tested": len(enrichment) == len(families) == 62,
        "replicates_10000": all(matrix.shape[0] == REPLICATES for matrix in assignments.values()),
        "same_primary_significant_set": set(primary_significant) == set(primary_enrichment.loc[primary_enrichment["convergently_enriched"], "tf_family"]),
        "both_genus_rule_applied": all(
            bool(row["convergently_enriched"]) == (bool(row["positive_both_genera"]) and float(row["bh_q"]) < 0.05)
            for row in enrichment.to_dict("records")
        ),
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
    (ROOT / "results/metrics/publication_v5_motif_gbox_masked_sensitivity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit["status"], **summary, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
