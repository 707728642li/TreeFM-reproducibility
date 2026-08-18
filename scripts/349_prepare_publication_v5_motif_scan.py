#!/usr/bin/env python3
"""Prepare frozen Tier-A promoter foreground, matched pools, and JASPAR chunks."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/metrics/publication_v5_motif"
PREPARED = OUT / "prepared"
CANDIDATES = ROOT / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv"
MOTIFS = ROOT / "data/raw/publication_v5_jaspar2026/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt"
METADATA = ROOT / "data/raw/publication_v5_jaspar2026/ultimate_metadata_table_CORE.tsv"
PROMOTERS = {
    "prunus": ROOT / "data/processed/functional/Prunus_publication_v3/promoter_labels.parquet",
    "pyrus": ROOT / "data/processed/functional/Pyrus_PRJNA669907/promoter_labels.parquet",
}
CHUNKS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_id(genus: str, gene: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene).strip("_")
    digest = hashlib.sha256(f"{genus}|{gene}".encode()).hexdigest()[:10]
    return f"{genus}__{token}__{digest}"


def split_meme() -> list[dict]:
    text = MOTIFS.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^MOTIF\s+", text))
    if len(matches) != 927:
        raise RuntimeError(f"Expected 927 motif blocks, observed {len(matches)}")
    header = text[: matches[0].start()]
    blocks = [
        text[match.start() : (matches[index + 1].start() if index + 1 < len(matches) else len(text))]
        for index, match in enumerate(matches)
    ]
    chunk_dir = PREPARED / "motif_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for chunk in range(CHUNKS):
        selected = blocks[chunk::CHUNKS]
        path = chunk_dir / f"jaspar_plants_{chunk + 1:02d}.meme"
        path.write_text(header + "".join(selected), encoding="utf-8")
        ids = [re.match(r"MOTIF\s+(\S+)", block).group(1) for block in selected]
        rows.append(
            {
                "chunk": chunk + 1,
                "path": path.relative_to(ROOT).as_posix(),
                "motif_count": len(ids),
                "first_motif": ids[0],
                "last_motif": ids[-1],
                "sha256": sha256(path),
            }
        )
    return rows


def main() -> int:
    PREPARED.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(CANDIDATES, sep="\t", dtype=str)
    foreground = {
        "prunus": {gene for value in candidates["prunus_gene_ids"].fillna("") for gene in value.split(";") if gene},
        "pyrus": {gene for value in candidates["pyrus_gene_ids"].fillna("") for gene in value.split(";") if gene},
    }
    frames = []
    for genus, path in PROMOTERS.items():
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["label_binary"].eq(1) & frame["promoter_2048"].notna()].copy()
        frame["gene_id"] = frame["gene_id"].astype(str)
        frame["chromosome"] = frame["chromosome"].astype(str)
        frame["sequence"] = frame["promoter_2048"].astype(str).str.upper()
        frame["promoter_length"] = frame["sequence"].str.len()
        valid = frame["sequence"].str.count("[ACGT]")
        frame["gc_fraction_recomputed"] = (
            frame["sequence"].str.count("[GC]") / valid.replace(0, np.nan)
        )
        frame["genus"] = genus
        frame["foreground"] = frame["gene_id"].isin(foreground[genus])
        frame["safe_id"] = [safe_id(genus, gene) for gene in frame["gene_id"]]
        frames.append(
            frame[
                [
                    "safe_id",
                    "genus",
                    "gene_id",
                    "chromosome",
                    "promoter_length",
                    "gc_fraction_recomputed",
                    "foreground",
                    "sequence",
                ]
            ]
        )
    promoters = pd.concat(frames, ignore_index=True)
    if promoters["safe_id"].duplicated().any():
        raise RuntimeError("Nonunique promoter safe IDs")
    observed_foreground = {
        genus: set(promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "gene_id"])
        for genus in PROMOTERS
    }
    if observed_foreground != foreground:
        raise RuntimeError(
            f"Foreground mismatch: { {g: sorted(foreground[g] - observed_foreground[g]) for g in PROMOTERS} }"
        )

    map_columns = [column for column in promoters.columns if column != "sequence"]
    promoters[map_columns].to_csv(OUT / "promoter_map.tsv.gz", sep="\t", index=False, compression="gzip")
    fasta_path = PREPARED / "positive_promoters.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in promoters.sort_values(["genus", "gene_id"]).to_dict("records"):
            sequence = re.sub(r"[^ACGT]", "N", row["sequence"])
            handle.write(f">{row['safe_id']}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")

    pool_rows = []
    for genus in PROMOTERS:
        fg = promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"]].copy()
        bg = promoters.loc[promoters["genus"].eq(genus) & ~promoters["foreground"]].copy()
        for target in fg.to_dict("records"):
            exact_length = bg["promoter_length"].eq(target["promoter_length"])
            gc_distance = (bg["gc_fraction_recomputed"] - target["gc_fraction_recomputed"]).abs()
            same_chromosome = bg["chromosome"].eq(target["chromosome"])
            pool = bg.loc[exact_length & (gc_distance <= 0.02) & same_chromosome].copy()
            stage = "same_chromosome_gc002"
            if len(pool) < 20:
                pool = bg.loc[exact_length & (gc_distance <= 0.02)].copy()
                stage = "same_genus_gc002"
            if len(pool) < 20:
                pool = bg.loc[exact_length].copy()
                pool["_distance"] = (pool["gc_fraction_recomputed"] - target["gc_fraction_recomputed"]).abs()
                pool = pool.sort_values(["_distance", "gene_id"]).head(100)
                stage = "same_genus_nearest100"
            if len(pool) < 20:
                raise RuntimeError(f"Insufficient controls for {genus}:{target['gene_id']}")
            for control in pool.to_dict("records"):
                pool_rows.append(
                    {
                        "genus": genus,
                        "foreground_safe_id": target["safe_id"],
                        "foreground_gene_id": target["gene_id"],
                        "control_safe_id": control["safe_id"],
                        "control_gene_id": control["gene_id"],
                        "matching_stage": stage,
                        "same_chromosome": control["chromosome"] == target["chromosome"],
                        "gc_distance": abs(control["gc_fraction_recomputed"] - target["gc_fraction_recomputed"]),
                        "promoter_length": target["promoter_length"],
                    }
                )
    pd.DataFrame(pool_rows).to_csv(OUT / "foreground_control_pools.tsv.gz", sep="\t", index=False, compression="gzip")

    metadata = pd.read_csv(METADATA, sep="\t", dtype=str)
    motif_ids = set(re.findall(r"(?m)^MOTIF\s+(\S+)", MOTIFS.read_text(encoding="utf-8")))
    plants = metadata.loc[metadata["tax_group"].eq("plants") & metadata["matrix_id"].isin(motif_ids)].copy()
    if len(plants) != 927 or plants["matrix_id"].nunique() != 927:
        raise RuntimeError(f"Unexpected JASPAR plant metadata: {plants.shape}")
    plants.to_csv(OUT / "jaspar2026_plant_metadata.tsv", sep="\t", index=False)
    chunks = split_meme()
    pd.DataFrame(chunks).to_csv(OUT / "motif_chunk_manifest.tsv", sep="\t", index=False)

    pool_frame = pd.DataFrame(pool_rows)
    stage_counts = {
        genus: (
            pool_frame.loc[pool_frame["genus"].eq(genus)]
            .groupby("matching_stage")["foreground_gene_id"]
            .nunique()
            .to_dict()
        )
        for genus in PROMOTERS
    }
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_descriptive",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "foreground_counts": {genus: len(values) for genus, values in foreground.items()},
        "foreground_total": sum(len(values) for values in foreground.values()),
        "positive_promoter_counts": promoters.groupby("genus").size().to_dict(),
        "motif_count": len(plants),
        "motif_family_count": int(
            plants["family"]
            .fillna("")
            .replace("", pd.NA)
            .fillna("Unclassified: " + plants["class"].fillna("unknown"))
            .nunique()
        ),
        "motif_chunks": CHUNKS,
        "matching_stage_counts": stage_counts,
        "input_fingerprints": {
            str(CANDIDATES.relative_to(ROOT)): sha256(CANDIDATES),
            str(MOTIFS.relative_to(ROOT)): sha256(MOTIFS),
            str(METADATA.relative_to(ROOT)): sha256(METADATA),
            **{str(path.relative_to(ROOT)): sha256(path) for path in PROMOTERS.values()},
        },
        "output_fingerprints": {
            "promoter_map.tsv.gz": sha256(OUT / "promoter_map.tsv.gz"),
            "foreground_control_pools.tsv.gz": sha256(OUT / "foreground_control_pools.tsv.gz"),
            "jaspar2026_plant_metadata.tsv": sha256(OUT / "jaspar2026_plant_metadata.tsv"),
            "motif_chunk_manifest.tsv": sha256(OUT / "motif_chunk_manifest.tsv"),
            "positive_promoters.fasta": sha256(fasta_path),
        },
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
