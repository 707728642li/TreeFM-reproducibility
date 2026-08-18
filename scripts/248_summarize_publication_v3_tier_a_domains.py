#!/usr/bin/env python3
"""Summarize an unbiased full-Pfam scan of corrected frozen Tier-A proteins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MAP = Path(
    "data/processed/publication_v3_tier_a_annotation/"
    "tier_a_candidate_and_anchor_proteins.tsv"
)
FASTA = Path(
    "data/processed/publication_v3_tier_a_annotation/"
    "tier_a_candidate_and_anchor_proteins.fa"
)
EXTRACTION_MANIFEST = Path(
    "data/processed/publication_v3_tier_a_annotation/extraction_manifest.json"
)
DOMTBL = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "domains/Pfam-A.domtblout"
)
OUT_SEQUENCE = Path(
    "metadata/publication_v3_tier_a_candidate_domain_annotation.tsv"
)
OUT_HITS = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/pfam_hits.tsv"
)
OUT_CONSENSUS = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "orthogroup_domain_consensus.tsv"
)
OUT_JSON = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "domain_summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def atomic_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header: str | None = None
    sequence: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    if header in records:
                        raise RuntimeError(f"duplicate FASTA identifier: {header}")
                    records[header] = "".join(sequence)
                header = line[1:].split()[0]
                sequence = []
            else:
                if header is None:
                    raise RuntimeError("sequence before first FASTA header")
                sequence.append(line)
    if header is not None:
        if header in records:
            raise RuntimeError(f"duplicate FASTA identifier: {header}")
        records[header] = "".join(sequence)
    return records


def parse_domtbl(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    hits: list[dict[str, object]] = []
    headers: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("#"):
                headers.append(raw.rstrip())
                continue
            if not raw.strip():
                continue
            fields = raw.split(maxsplit=22)
            if len(fields) < 22:
                raise RuntimeError(f"malformed hmmscan domtblout row: {raw}")
            accession = fields[1].split(".")[0]
            if not accession.startswith("PF"):
                raise RuntimeError(f"non-Pfam accession in hmmscan output: {fields[1]}")
            hit = {
                "pfam_name": fields[0],
                "pfam_accession": accession,
                "pfam_accession_versioned": fields[1],
                "profile_length": int(fields[2]),
                "sequence_id": fields[3],
                "query_length": int(fields[5]),
                "full_evalue": float(fields[6]),
                "full_score": float(fields[7]),
                "full_bias": float(fields[8]),
                "domain_index": int(fields[9]),
                "domain_total": int(fields[10]),
                "domain_c_evalue": float(fields[11]),
                "domain_i_evalue": float(fields[12]),
                "domain_score": float(fields[13]),
                "domain_bias": float(fields[14]),
                "hmm_from": int(fields[15]),
                "hmm_to": int(fields[16]),
                "ali_from": int(fields[17]),
                "ali_to": int(fields[18]),
                "env_from": int(fields[19]),
                "env_to": int(fields[20]),
                "accuracy": float(fields[21]),
                "description": fields[22] if len(fields) == 23 else "",
            }
            numeric = [
                hit["full_evalue"],
                hit["full_score"],
                hit["full_bias"],
                hit["domain_c_evalue"],
                hit["domain_i_evalue"],
                hit["domain_score"],
                hit["domain_bias"],
                hit["accuracy"],
            ]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise RuntimeError("non-finite statistic in hmmscan output")
            if not (
                1
                <= int(hit["hmm_from"])
                <= int(hit["hmm_to"])
                <= int(hit["profile_length"])
            ):
                raise RuntimeError("invalid HMM coordinates in hmmscan output")
            if not (
                1
                <= int(hit["ali_from"])
                <= int(hit["ali_to"])
                <= int(hit["query_length"])
            ):
                raise RuntimeError("invalid alignment coordinates in hmmscan output")
            if not (
                1
                <= int(hit["env_from"])
                <= int(hit["env_to"])
                <= int(hit["query_length"])
            ):
                raise RuntimeError("invalid envelope coordinates in hmmscan output")
            hit["profile_coverage"] = (
                int(hit["hmm_to"]) - int(hit["hmm_from"]) + 1
            ) / int(hit["profile_length"])
            hit["sequence_coverage"] = (
                int(hit["ali_to"]) - int(hit["ali_from"]) + 1
            ) / int(hit["query_length"])
            hits.append(hit)
    return hits, headers


def support_label(anchor: int, prunus: int, pyrus: int) -> str:
    if anchor and prunus and pyrus:
        return "cross_genus_anchor_supported"
    if prunus and pyrus:
        return "candidate_cross_genus"
    if anchor and (prunus or pyrus):
        return "anchor_plus_single_genus"
    if anchor:
        return "anchor_only"
    if prunus or pyrus:
        return "single_genus_candidate_only"
    raise RuntimeError("empty domain support pattern")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    paths = [MAP, FASTA, EXTRACTION_MANIFEST, DOMTBL]
    for relative in paths:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)

    with (root / MAP).open("r", encoding="utf-8", newline="") as handle:
        mapping = list(csv.DictReader(handle, delimiter="\t"))
    if not mapping:
        raise RuntimeError("empty Tier-A protein map")
    map_by_id = {row["sequence_id"]: row for row in mapping}
    if len(map_by_id) != len(mapping):
        raise RuntimeError("duplicate sequence identifier in Tier-A protein map")
    sequences = read_fasta(root / FASTA)
    if set(sequences) != set(map_by_id):
        raise RuntimeError("FASTA and protein-map sequence inventories differ")
    for sequence_id, sequence in sequences.items():
        source = map_by_id[sequence_id]
        if len(sequence) != int(source["protein_length"]):
            raise RuntimeError(f"protein length mismatch for {sequence_id}")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != source[
            "sequence_sha256"
        ]:
            raise RuntimeError(f"protein sequence hash mismatch for {sequence_id}")

    raw_hits, domtbl_headers = parse_domtbl(root / DOMTBL)
    for hit in raw_hits:
        sequence_id = str(hit["sequence_id"])
        if sequence_id not in map_by_id:
            raise RuntimeError(f"hmmscan hit has unknown sequence: {sequence_id}")
        if int(hit["query_length"]) != len(sequences[sequence_id]):
            raise RuntimeError(f"hmmscan query length mismatch: {sequence_id}")

    hits_by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    hit_rows: list[dict[str, object]] = []
    hit_fields = list(mapping[0]) + [
        "pfam_name",
        "pfam_accession",
        "pfam_accession_versioned",
        "description",
        "profile_length",
        "query_length",
        "full_evalue",
        "full_score",
        "full_bias",
        "domain_index",
        "domain_total",
        "domain_c_evalue",
        "domain_i_evalue",
        "domain_score",
        "domain_bias",
        "hmm_from",
        "hmm_to",
        "ali_from",
        "ali_to",
        "env_from",
        "env_to",
        "accuracy",
        "profile_coverage",
        "sequence_coverage",
        "selection_authority",
    ]
    for hit in sorted(
        raw_hits,
        key=lambda row: (
            int(map_by_id[str(row["sequence_id"])]["catalog_rank"]),
            str(row["sequence_id"]),
            int(row["ali_from"]),
            str(row["pfam_accession"]),
            int(row["domain_index"]),
        ),
    ):
        sequence_id = str(hit["sequence_id"])
        hits_by_sequence[sequence_id].append(hit)
        hit_rows.append(
            {
                **map_by_id[sequence_id],
                **hit,
                "selection_authority": False,
            }
        )

    sequence_rows: list[dict[str, object]] = []
    for source in mapping:
        sequence_id = source["sequence_id"]
        seq_hits = sorted(
            hits_by_sequence.get(sequence_id, []),
            key=lambda row: (
                int(row["ali_from"]),
                int(row["ali_to"]),
                str(row["pfam_accession"]),
            ),
        )
        accessions = sorted({str(row["pfam_accession"]) for row in seq_hits})
        names = sorted({str(row["pfam_name"]) for row in seq_hits})
        architecture = ";".join(
            f"{row['pfam_accession']}:{row['ali_from']}-{row['ali_to']}"
            for row in seq_hits
        )
        sequence_rows.append(
            {
                **source,
                "has_pfam_domain": bool(seq_hits),
                "pfam_domain_hit_count": len(seq_hits),
                "pfam_distinct_accession_count": len(accessions),
                "pfam_accessions": ";".join(accessions),
                "pfam_names": ";".join(names),
                "pfam_architecture": architecture,
                "best_domain_i_evalue": (
                    min(float(row["domain_i_evalue"]) for row in seq_hits)
                    if seq_hits
                    else ""
                ),
                "selection_authority": False,
            }
        )

    totals: dict[tuple[str, str], int] = defaultdict(int)
    for source in mapping:
        totals[(source["orthogroup"], source["genus"])] += 1
    domain_sequences: dict[
        tuple[str, str, str], set[str]
    ] = defaultdict(set)
    domain_meta: dict[tuple[str, str], tuple[str, str]] = {}
    for hit in raw_hits:
        sequence_id = str(hit["sequence_id"])
        source = map_by_id[sequence_id]
        key = (
            source["orthogroup"],
            str(hit["pfam_accession"]),
            source["genus"],
        )
        domain_sequences[key].add(sequence_id)
        domain_meta[(source["orthogroup"], str(hit["pfam_accession"]))] = (
            str(hit["pfam_name"]),
            str(hit["description"]),
        )

    consensus_rows: list[dict[str, object]] = []
    for orthogroup, accession in sorted(
        domain_meta,
        key=lambda key: (
            min(
                int(row["catalog_rank"])
                for row in mapping
                if row["orthogroup"] == key[0]
            ),
            key[1],
        ),
    ):
        counts = {
            genus: len(domain_sequences[(orthogroup, accession, genus)])
            for genus in ("arabidopsis", "prunus", "pyrus")
        }
        name, description = domain_meta[(orthogroup, accession)]
        consensus_rows.append(
            {
                "catalog_rank": min(
                    int(row["catalog_rank"])
                    for row in mapping
                    if row["orthogroup"] == orthogroup
                ),
                "orthogroup": orthogroup,
                "pfam_accession": accession,
                "pfam_name": name,
                "description": description,
                "arabidopsis_detected": counts["arabidopsis"],
                "arabidopsis_total": totals[(orthogroup, "arabidopsis")],
                "arabidopsis_fraction": counts["arabidopsis"]
                / totals[(orthogroup, "arabidopsis")],
                "prunus_detected": counts["prunus"],
                "prunus_total": totals[(orthogroup, "prunus")],
                "prunus_fraction": counts["prunus"] / totals[(orthogroup, "prunus")],
                "pyrus_detected": counts["pyrus"],
                "pyrus_total": totals[(orthogroup, "pyrus")],
                "pyrus_fraction": counts["pyrus"] / totals[(orthogroup, "pyrus")],
                "support_label": support_label(
                    counts["arabidopsis"], counts["prunus"], counts["pyrus"]
                ),
                "selection_authority": False,
            }
        )

    atomic_tsv(root / OUT_SEQUENCE, sequence_rows, list(sequence_rows[0]))
    atomic_tsv(root / OUT_HITS, hit_rows, hit_fields)
    consensus_fields = [
        "catalog_rank",
        "orthogroup",
        "pfam_accession",
        "pfam_name",
        "description",
        "arabidopsis_detected",
        "arabidopsis_total",
        "arabidopsis_fraction",
        "prunus_detected",
        "prunus_total",
        "prunus_fraction",
        "pyrus_detected",
        "pyrus_total",
        "pyrus_fraction",
        "support_label",
        "selection_authority",
    ]
    atomic_tsv(root / OUT_CONSENSUS, consensus_rows, consensus_fields)

    orthogroup_summary: dict[str, dict[str, object]] = {}
    for orthogroup in dict.fromkeys(row["orthogroup"] for row in mapping):
        group_sequences = [
            row["sequence_id"] for row in mapping if row["orthogroup"] == orthogroup
        ]
        group_consensus = [
            row for row in consensus_rows if row["orthogroup"] == orthogroup
        ]
        orthogroup_summary[orthogroup] = {
            "catalog_rank": min(
                int(row["catalog_rank"])
                for row in mapping
                if row["orthogroup"] == orthogroup
            ),
            "proteins_total": len(group_sequences),
            "proteins_with_pfam_hit": sum(
                bool(hits_by_sequence.get(sequence_id))
                for sequence_id in group_sequences
            ),
            "distinct_pfam_accessions": len(
                {row["pfam_accession"] for row in group_consensus}
            ),
            "cross_genus_anchor_supported_domains": [
                row["pfam_accession"]
                for row in group_consensus
                if row["support_label"] == "cross_genus_anchor_supported"
            ],
        }

    extraction = json.loads(
        (root / EXTRACTION_MANIFEST).read_text(encoding="utf-8")
    )
    if extraction.get("status") != "pass":
        raise RuntimeError("protein extraction manifest is not passing")
    summary = {
        "status": "pass",
        "scope": "retrospective_corrected_tier_a_full_pfam_annotation",
        "selection_authority": False,
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hmm_threshold": "model_specific_gathering_threshold",
        "scan_program": "hmmscan",
        "scan_header": domtbl_headers,
        "candidate_orthogroups": len(orthogroup_summary),
        "extracted_proteins": len(mapping),
        "proteins_with_pfam_hit": sum(bool(hits_by_sequence.get(key)) for key in map_by_id),
        "proteins_without_pfam_hit": sum(
            not bool(hits_by_sequence.get(key)) for key in map_by_id
        ),
        "domain_hit_rows": len(hit_rows),
        "orthogroup_domain_rows": len(consensus_rows),
        "cross_genus_anchor_supported_domain_rows": sum(
            row["support_label"] == "cross_genus_anchor_supported"
            for row in consensus_rows
        ),
        "by_orthogroup": orthogroup_summary,
        "technical_gates": {
            "mapping_nonempty": True,
            "mapping_sequence_ids_unique": True,
            "fasta_inventory_exact": True,
            "sequence_lengths_and_hashes_exact": True,
            "all_hits_map_to_queries": True,
            "all_hit_coordinates_valid": True,
            "all_hit_statistics_finite": True,
            "no_biological_result_gate": True,
            "model_outputs_accessed": False,
            "malus_accessed": False,
        },
        "inputs": {
            str(relative): sha256(root / relative) for relative in paths
        },
        "outputs": {
            str(OUT_SEQUENCE): sha256(root / OUT_SEQUENCE),
            str(OUT_HITS): sha256(root / OUT_HITS),
            str(OUT_CONSENSUS): sha256(root / OUT_CONSENSUS),
        },
        "violations": [],
    }
    atomic_json(root / OUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
