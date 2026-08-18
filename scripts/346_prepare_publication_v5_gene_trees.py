#!/usr/bin/env python3
"""Extract and relabel all proteins for the 14 frozen Tier-A gene families."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "results/metrics/publication_v5_resource_audit/tier_a_member_mapping.tsv.gz"
RESOURCE_AUDIT = ROOT / "results/metrics/publication_v5_resource_audit.json"
CANDIDATES = ROOT / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv"
BRIDGE = ROOT / "metadata/publication_v3_prunus_v21_gene_id_bridge.tsv"
SPECIES_TREE = ROOT / "results/metrics/publication_v4_corpus_phylogeny/species_tree_named_rooted.nwk"
OUT = ROOT / "results/metrics/publication_v5_gene_trees"
PREPARED = OUT / "prepared"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protein_path(species: str) -> Path:
    if species == "prunus_persica":
        return ROOT / "data/interim/functional_genomes/prunus_persica/primary.protein.fa"
    return ROOT / f"data/processed/orthofinder_benchmark_publication_v3/{species}.fa"


def safe_identifier(species: str, gene: str, member: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene).strip("_")
    digest = hashlib.sha256(member.encode("utf-8")).hexdigest()[:10]
    return f"{species}__{base}__{digest}"


def read_fasta(path: Path, wanted: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    name = None
    chunks: list[str] = []

    def commit() -> None:
        if name is None:
            return
        canonical = name.replace(":", "_")
        if canonical not in wanted:
            return
        sequence = "".join(chunks).upper().replace("*", "")
        sequence = re.sub(r"[^ABCDEFGHIKLMNPQRSTVWXYZ]", "X", sequence)
        if not sequence:
            raise RuntimeError(f"Empty sequence for {canonical} in {path}")
        result[canonical] = sequence

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                commit()
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        commit()
    return result


def write_fasta(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in rows:
            handle.write(f">{identifier}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def main() -> int:
    audit = json.loads(RESOURCE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Publication-v5 resource gate has not passed")
    PREPARED.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(RESOURCE, sep="\t", dtype=str)
    if mapping["orthogroup"].nunique() != 14 or mapping["species"].nunique() != 26 or len(mapping) != 2836:
        raise RuntimeError("Frozen member mapping has unexpected dimensions")

    bridge = pd.read_csv(BRIDGE, sep="\t", dtype=str)
    source_to_technical = dict(zip(bridge["source_gene_id"], bridge["technical_gene_id"]))
    candidates = pd.read_csv(CANDIDATES, sep="\t", dtype=str)
    candidate_keys: set[tuple[str, str, str]] = set()
    source_labels: dict[tuple[str, str], str] = {}
    for row in candidates.to_dict("records"):
        og = row["orthogroup"]
        for genus, species, field in [
            ("prunus", "prunus_persica", "prunus_gene_ids"),
            ("pyrus", "pyrus_pyrifolia", "pyrus_gene_ids"),
        ]:
            for source_gene in str(row.get(field, "")).split(";"):
                source_gene = source_gene.strip()
                if not source_gene:
                    continue
                technical = source_to_technical.get(source_gene, source_gene) if genus == "prunus" else source_gene
                candidate_keys.add((og, species, technical))
                source_labels[(species, technical)] = source_gene

    wanted_by_species: dict[str, set[str]] = defaultdict(set)
    for row in mapping.to_dict("records"):
        wanted_by_species[row["species"]].add(row["member_id"])
    sequences: dict[str, str] = {}
    protein_fingerprints = {}
    for species in sorted(wanted_by_species):
        path = protein_path(species)
        extracted = read_fasta(path, wanted_by_species[species])
        missing = sorted(wanted_by_species[species] - set(extracted))
        if missing:
            raise RuntimeError(f"Missing {len(missing)} proteins for {species}: {missing[:3]}")
        sequences.update(extracted)
        protein_fingerprints[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    id_rows = []
    family_rows = []
    for og, frame in mapping.groupby("orthogroup", sort=True):
        fasta_rows = []
        smap_rows = []
        lengths = []
        for row in frame.sort_values(["species", "gene_id", "member_id"]).to_dict("records"):
            safe = safe_identifier(row["species"], row["gene_id"], row["member_id"])
            sequence = sequences[row["member_id"]]
            fasta_rows.append((safe, sequence))
            smap_rows.append((safe, row["species"]))
            lengths.append(len(sequence))
            is_candidate = (og, row["species"], row["gene_id"]) in candidate_keys
            id_rows.append(
                {
                    "orthogroup": og,
                    "safe_id": safe,
                    "species": row["species"],
                    "gene_id": row["gene_id"],
                    "source_gene_id": source_labels.get((row["species"], row["gene_id"]), ""),
                    "original_member_id": row["member_id"],
                    "sequence_length": len(sequence),
                    "tier_a_positive_candidate": is_candidate,
                }
            )
        if len({identifier for identifier, _ in fasta_rows}) != len(fasta_rows):
            raise RuntimeError(f"Nonunique safe IDs in {og}")
        write_fasta(PREPARED / f"{og}.raw.faa", fasta_rows)
        with (PREPARED / f"{og}.smap.tsv").open("w", encoding="utf-8", newline="\n") as handle:
            for gene, species in smap_rows:
                handle.write(f"{gene} {species}\n")
        family_rows.append(
            {
                "orthogroup": og,
                "sequences": len(fasta_rows),
                "species": frame["species"].nunique(),
                "candidate_sequences": sum(
                    (og, row["species"], row["gene_id"]) in candidate_keys for row in frame.to_dict("records")
                ),
                "min_length": min(lengths),
                "median_length": float(pd.Series(lengths).median()),
                "max_length": max(lengths),
                "fasta_sha256": sha256(PREPARED / f"{og}.raw.faa"),
                "smap_sha256": sha256(PREPARED / f"{og}.smap.tsv"),
            }
        )

    pd.DataFrame(id_rows).to_csv(OUT / "id_map.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(family_rows).to_csv(OUT / "family_input_summary.tsv", sep="\t", index=False)
    tree_text = SPECIES_TREE.read_text(encoding="utf-8").strip()
    (PREPARED / "species_tree.nwk").write_text(tree_text + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_descriptive",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "orthogroups": sorted(mapping["orthogroup"].unique()),
        "orthogroup_count": 14,
        "species_count": 26,
        "sequence_count": len(id_rows),
        "candidate_sequence_count": sum(bool(row["tier_a_positive_candidate"]) for row in id_rows),
        "display_families": ["OG0000025", "OG0000413", "OG0000277"],
        "species_tree_sha256": sha256(PREPARED / "species_tree.nwk"),
        "protein_fingerprints": protein_fingerprints,
        "output_fingerprints": {
            "id_map.tsv.gz": sha256(OUT / "id_map.tsv.gz"),
            "family_input_summary.tsv": sha256(OUT / "family_input_summary.tsv"),
        },
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "orthogroup_count", "species_count", "sequence_count", "candidate_sequence_count", "display_families")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
