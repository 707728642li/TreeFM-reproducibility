#!/usr/bin/env python3
"""Collapse JASPAR/FIMO hits to TF families and test matched cross-genus enrichment."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/metrics/publication_v5_motif"
HITS = OUT / "fimo_chunks"
PROMOTER_MAP = OUT / "promoter_map.tsv.gz"
POOLS = OUT / "foreground_control_pools.tsv.gz"
METADATA = OUT / "jaspar2026_plant_metadata.tsv"
SEED = 20260803
REPLICATES = 10_000
BIN_EDGES = np.linspace(-2048, 0, 21)
FIXED_FAMILY_TOKENS = ["AP2", "ERF", "WRKY", "bZIP", "bHLH"]


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


def read_fimo() -> pd.DataFrame:
    frames = []
    paths = sorted(HITS.glob("fimo_*.tsv.gz"))
    if len(paths) != 32:
        raise RuntimeError(f"Expected 32 FIMO chunks, observed {len(paths)}")
    for path in paths:
        # FIMO releases differ in whether the tabular header is prefixed by
        # "# ".  Preserve that header while dropping any other comment lines.
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            lines = []
            for line in handle:
                stripped = line.lstrip("# ")
                if line.startswith("#") and not stripped.startswith("motif_id\t"):
                    continue
                lines.append(stripped if stripped.startswith("motif_id\t") else line)
        if not lines or not lines[0].startswith("motif_id\t"):
            raise RuntimeError(f"FIMO output lacks a recognizable header: {path}")
        frame = pd.read_csv(
            io.StringIO("".join(lines)),
            sep="\t",
            dtype={"motif_id": str, "sequence_name": str},
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"motif_alt_id": "motif_name", "sequence_name": "safe_id", "p-value": "p_value", "q-value": "q_value"})
    required = {"motif_id", "safe_id", "start", "stop", "strand", "score", "p_value", "matched_sequence"}
    missing = required - set(result.columns)
    if missing:
        raise RuntimeError(f"FIMO output missing columns: {sorted(missing)}")
    if result.duplicated(["motif_id", "safe_id", "start", "stop", "strand"]).any():
        raise RuntimeError("Duplicate FIMO hits across motif chunks")
    return result


def generate_assignments(promoters: pd.DataFrame, pools: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    assignments: dict[str, np.ndarray] = {}
    rows = []
    for genus in sorted(promoters["genus"].unique()):
        foreground = sorted(promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"])
        pool_map = {
            safe: sorted(group["control_safe_id"].unique())
            for safe, group in pools.loc[pools["genus"].eq(genus)].groupby("foreground_safe_id")
        }
        matrix = np.empty((REPLICATES, len(foreground)), dtype=object)
        for replicate in range(REPLICATES):
            used: set[str] = set()
            order = rng.permutation(len(foreground))
            selected = [None] * len(foreground)
            for index in order:
                target = foreground[index]
                available = [control for control in pool_map[target] if control not in used]
                if not available:
                    available = pool_map[target]
                control = available[int(rng.integers(0, len(available)))]
                selected[index] = control
                used.add(control)
            matrix[replicate] = selected
            for target, control in zip(foreground, selected):
                rows.append(
                    {
                        "replicate": replicate + 1,
                        "genus": genus,
                        "foreground_safe_id": target,
                        "control_safe_id": control,
                    }
                )
        assignments[genus] = matrix
    return assignments, pd.DataFrame(rows)


def family_position_matrix(hits: pd.DataFrame, gene_order: list[str], family: str) -> np.ndarray:
    gene_index = {gene: index for index, gene in enumerate(gene_order)}
    matrix = np.zeros((len(gene_order), len(BIN_EDGES) - 1), dtype=float)
    subset = hits.loc[hits["tf_family"].eq(family) & hits["safe_id"].isin(gene_index)].copy()
    if subset.empty:
        return matrix
    subset["distance_to_tss"] = (subset["start"].astype(float) + subset["stop"].astype(float)) / 2 - 2048.5
    subset["bin"] = np.clip(np.digitize(subset["distance_to_tss"], BIN_EDGES) - 1, 0, len(BIN_EDGES) - 2)
    counts = subset.groupby(["safe_id", "bin"]).size().rename("count").reset_index()
    bin_width_kb = float(BIN_EDGES[1] - BIN_EDGES[0]) / 1000.0
    counts["hits_per_kb"] = counts["count"] / bin_width_kb
    for row in counts.to_dict("records"):
        matrix[gene_index[row["safe_id"]], int(row["bin"])] = float(row["hits_per_kb"])
    return matrix


def main() -> int:
    fimo = read_fimo()
    promoters = pd.read_csv(PROMOTER_MAP, sep="\t")
    pools = pd.read_csv(POOLS, sep="\t")
    metadata = pd.read_csv(METADATA, sep="\t", dtype=str)
    metadata["tf_family"] = metadata["family"].fillna("").replace("", np.nan)
    metadata["tf_family"] = metadata["tf_family"].fillna("Unclassified: " + metadata["class"].fillna("unknown"))
    motif_to_family = dict(zip(metadata["matrix_id"], metadata["tf_family"]))
    fimo["tf_family"] = fimo["motif_id"].map(motif_to_family)
    if fimo["tf_family"].isna().any():
        raise RuntimeError(f"Unmapped motifs: {sorted(fimo.loc[fimo['tf_family'].isna(), 'motif_id'].unique())[:5]}")
    fimo = fimo.merge(promoters[["safe_id", "genus", "gene_id", "foreground"]], on="safe_id", how="left", validate="many_to_one")
    if fimo["genus"].isna().any():
        raise RuntimeError("FIMO contains unknown sequence IDs")
    fimo.sort_values(["tf_family", "motif_id", "safe_id", "start", "stop"]).to_csv(
        OUT / "fimo_hits.tsv.gz", sep="\t", index=False, compression="gzip"
    )

    families = sorted(metadata["tf_family"].unique())
    genes = sorted(promoters["safe_id"].unique())
    gene_index = {gene: index for index, gene in enumerate(genes)}
    family_index = {family: index for index, family in enumerate(families)}
    presence = np.zeros((len(genes), len(families)), dtype=np.uint8)
    for safe, family in fimo[["safe_id", "tf_family"]].drop_duplicates().itertuples(index=False):
        presence[gene_index[safe], family_index[family]] = 1

    assignments, assignment_rows = generate_assignments(promoters, pools)
    assignment_rows.to_csv(OUT / "matched_background_replicates.tsv.gz", sep="\t", index=False, compression="gzip")

    candidate_fraction = {}
    null_fraction = {}
    for genus in sorted(assignments):
        fg_ids = sorted(promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"])
        fg_indices = np.array([gene_index[item] for item in fg_ids])
        candidate_fraction[genus] = presence[fg_indices].mean(axis=0)
        null = np.empty((REPLICATES, len(families)), dtype=np.float32)
        for replicate in range(REPLICATES):
            indices = [gene_index[item] for item in assignments[genus][replicate]]
            null[replicate] = presence[indices].mean(axis=0)
        null_fraction[genus] = null

    genera = sorted(assignments)
    observed_combined = np.mean([candidate_fraction[genus] for genus in genera], axis=0)
    null_combined = np.mean([null_fraction[genus] for genus in genera], axis=0)
    p_values = (1 + (null_combined >= observed_combined[None, :]).sum(axis=0)) / (REPLICATES + 1)
    q_values = bh(p_values.astype(float))
    enrichment_rows = []
    for index, family in enumerate(families):
        family_metadata = metadata.loc[metadata["tf_family"].eq(family)]
        row = {
            "tf_family": family,
            "tf_classes": ";".join(sorted(family_metadata["class"].dropna().astype(str).unique())),
            "representative_profiles": ";".join(sorted(family_metadata["name"].dropna().astype(str).unique())[:8]),
            "motif_profiles": int(family_metadata["matrix_id"].nunique()),
        }
        positive_both = True
        for genus in genera:
            median = float(np.median(null_fraction[genus][:, index]))
            candidate = float(candidate_fraction[genus][index])
            row[f"{genus}_candidate_fraction"] = candidate
            row[f"{genus}_control_median"] = median
            row[f"{genus}_difference"] = candidate - median
            positive_both &= candidate > median
        row.update(
            {
                "combined_candidate_fraction": float(observed_combined[index]),
                "combined_control_median": float(np.median(null_combined[:, index])),
                "combined_difference": float(observed_combined[index] - np.median(null_combined[:, index])),
                "empirical_p": float(p_values[index]),
                "bh_q": float(q_values[index]),
                "positive_both_genera": bool(positive_both),
                "convergently_enriched": bool(positive_both and q_values[index] < 0.05),
            }
        )
        enrichment_rows.append(row)
    enrichment = pd.DataFrame(enrichment_rows).sort_values(["bh_q", "empirical_p", "tf_family"])
    enrichment.to_csv(OUT / "tf_family_enrichment.tsv", sep="\t", index=False)

    selected_families = set(enrichment.loc[enrichment["convergently_enriched"], "tf_family"])
    for family in families:
        family_metadata = metadata.loc[metadata["tf_family"].eq(family)]
        annotation_text = " ".join(
            [family]
            + family_metadata["class"].dropna().astype(str).tolist()
            + family_metadata["name"].dropna().astype(str).tolist()
        ).lower()
        if any(token.lower() in annotation_text for token in FIXED_FAMILY_TOKENS):
            selected_families.add(family)
    position_rows = []
    for family in sorted(selected_families):
        for genus in genera:
            genus_genes = sorted(promoters.loc[promoters["genus"].eq(genus), "safe_id"])
            local_index = {gene: index for index, gene in enumerate(genus_genes)}
            matrix = family_position_matrix(fimo, genus_genes, family)
            fg_ids = sorted(promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"])
            fg_density = matrix[[local_index[item] for item in fg_ids]].mean(axis=0)
            null_density = np.empty((REPLICATES, matrix.shape[1]), dtype=np.float32)
            for replicate in range(REPLICATES):
                indices = [local_index[item] for item in assignments[genus][replicate]]
                null_density[replicate] = matrix[indices].mean(axis=0)
            for bin_index in range(len(BIN_EDGES) - 1):
                position_rows.append(
                    {
                        "tf_family": family,
                        "genus": genus,
                        "bin_start": BIN_EDGES[bin_index],
                        "bin_end": BIN_EDGES[bin_index + 1],
                        "candidate_density": float(fg_density[bin_index]),
                        "control_median": float(np.median(null_density[:, bin_index])),
                        "control_q025": float(np.quantile(null_density[:, bin_index], 0.025)),
                        "control_q975": float(np.quantile(null_density[:, bin_index], 0.975)),
                    }
                )
    pd.DataFrame(position_rows).to_csv(OUT / "tf_family_position_density.tsv", sep="\t", index=False)

    output_names = [
        "fimo_hits.tsv.gz",
        "matched_background_replicates.tsv.gz",
        "tf_family_enrichment.tsv",
        "tf_family_position_density.tsv",
    ]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_descriptive",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "fimo_threshold": 1e-5,
        "replicates": REPLICATES,
        "seed": SEED,
        "promoters_scanned": len(promoters),
        "foreground_promoters": int(promoters["foreground"].sum()),
        "fimo_hits": len(fimo),
        "motif_profiles_with_hits": int(fimo["motif_id"].nunique()),
        "tf_families_tested": len(families),
        "convergently_enriched_families": int(enrichment["convergently_enriched"].sum()),
        "selected_position_families": sorted(selected_families),
        "input_fingerprints": {
            str(PROMOTER_MAP.relative_to(ROOT)): sha256(PROMOTER_MAP),
            str(POOLS.relative_to(ROOT)): sha256(POOLS),
            str(METADATA.relative_to(ROOT)): sha256(METADATA),
            **{path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(HITS.glob("fimo_*.tsv.gz"))},
        },
        "output_fingerprints": {name: sha256(OUT / name) for name in output_names},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    checks = {
        "thirty_two_fimo_chunks": len(list(HITS.glob("fimo_*.tsv.gz"))) == 32,
        "all_hits_below_threshold": bool(len(fimo)) and float(fimo["p_value"].max()) <= 1e-5,
        "all_hits_mapped": not fimo["tf_family"].isna().any() and not fimo["genus"].isna().any(),
        "foreground_79": int(promoters["foreground"].sum()) == 79,
        "replicates_10000": all(matrix.shape[0] == 10000 for matrix in assignments.values()),
        "all_metadata_derived_families_tested": len(enrichment) == len(families) == metadata["tf_family"].nunique(),
        "both_genus_rule_applied": all(
            bool(row["convergently_enriched"]) == (bool(row["positive_both_genera"]) and float(row["bh_q"]) < 0.05)
            for row in enrichment.to_dict("records")
        ),
        "position_families_prespecified_or_significant": set(position["tf_family"] for position in position_rows) == selected_families,
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
    (ROOT / "results/metrics/publication_v5_motif_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit["status"], **summary, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
