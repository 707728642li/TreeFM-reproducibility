#!/usr/bin/env python3
"""Independently verify the corrected Tier-A full-Pfam annotation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("TREEFM_ROOT", Path(__file__).resolve().parents[1])).resolve()
FREEZE = Path("config/publication_v3_tier_a_postselection_annotation_freeze.json")
CANDIDATE_FREEZE = Path(
    "config/publication_v3_crossgenus_candidate_catalog_freeze.json"
)
CANDIDATES = Path(
    "results/biological_cases/publication_v3_crossgenus_candidates/"
    "tier_a_candidates.tsv"
)
MAP = Path(
    "data/processed/publication_v3_tier_a_annotation/"
    "tier_a_candidate_and_anchor_proteins.tsv"
)
FASTA = Path(
    "data/processed/publication_v3_tier_a_annotation/"
    "tier_a_candidate_and_anchor_proteins.fa"
)
DOMTBL = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "domains/Pfam-A.domtblout"
)
SEQUENCE_TABLE = Path(
    "metadata/publication_v3_tier_a_candidate_domain_annotation.tsv"
)
HITS = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/pfam_hits.tsv"
)
CONSENSUS = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "orthogroup_domain_consensus.tsv"
)
SUMMARY = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/domain_summary.json"
)
PIPELINE = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "pipeline_manifest.json"
)
OUT = Path(
    "results/biological_cases/publication_v3_tier_a_annotation/"
    "independent_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def split_ids(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    name: str | None = None
    sequence: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    if name in result:
                        raise RuntimeError(f"duplicate FASTA name: {name}")
                    result[name] = "".join(sequence)
                name = line[1:].split()[0]
                sequence = []
            else:
                if name is None:
                    raise RuntimeError("sequence before FASTA header")
                sequence.append(line)
    if name is not None:
        if name in result:
            raise RuntimeError(f"duplicate FASTA name: {name}")
        result[name] = "".join(sequence)
    return result


def parse_domtbl(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.split(maxsplit=22)
            if len(fields) < 22:
                raise RuntimeError("malformed independent domtbl parse")
            row = {
                "pfam_name": fields[0],
                "pfam_accession": fields[1].split(".")[0],
                "sequence_id": fields[3],
                "profile_length": int(fields[2]),
                "query_length": int(fields[5]),
                "domain_i_evalue": float(fields[12]),
                "domain_score": float(fields[13]),
                "domain_index": int(fields[9]),
                "hmm_from": int(fields[15]),
                "hmm_to": int(fields[16]),
                "ali_from": int(fields[17]),
                "ali_to": int(fields[18]),
                "env_from": int(fields[19]),
                "env_to": int(fields[20]),
            }
            if not str(row["pfam_accession"]).startswith("PF"):
                raise RuntimeError("non-Pfam accession")
            if not all(
                math.isfinite(float(row[key]))
                for key in ("domain_i_evalue", "domain_score")
            ):
                raise RuntimeError("non-finite domain statistic")
            if not (
                1
                <= int(row["hmm_from"])
                <= int(row["hmm_to"])
                <= int(row["profile_length"])
            ):
                raise RuntimeError("invalid independent HMM coordinates")
            if not (
                1
                <= int(row["ali_from"])
                <= int(row["ali_to"])
                <= int(row["query_length"])
            ):
                raise RuntimeError("invalid independent alignment coordinates")
            result.append(row)
    return result


def support_label(anchor: int, prunus: int, pyrus: int) -> str:
    if anchor and prunus and pyrus:
        return "cross_genus_anchor_supported"
    if prunus and pyrus:
        return "candidate_cross_genus"
    if anchor and (prunus or pyrus):
        return "anchor_plus_single_genus"
    if anchor:
        return "anchor_only"
    return "single_genus_candidate_only"


def main() -> None:
    root = Path.cwd().resolve()
    if root != ROOT:
        raise SystemExit(f"refusing to verify outside {ROOT}: {root}")
    required = [
        FREEZE,
        CANDIDATE_FREEZE,
        CANDIDATES,
        MAP,
        FASTA,
        DOMTBL,
        SEQUENCE_TABLE,
        HITS,
        CONSENSUS,
        SUMMARY,
        PIPELINE,
    ]
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)

    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    freeze = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    candidate_freeze = json.loads(
        (root / CANDIDATE_FREEZE).read_text(encoding="utf-8")
    )
    summary = json.loads((root / SUMMARY).read_text(encoding="utf-8"))
    pipeline = json.loads((root / PIPELINE).read_text(encoding="utf-8"))
    check(freeze.get("status") == "frozen", "annotation_freeze_not_frozen")
    check(freeze.get("freeze_version") == "2.0", "annotation_freeze_not_v2")
    check(
        candidate_freeze.get("status") == "pass"
        and candidate_freeze.get("freeze_version") == "2.0",
        "candidate_freeze_not_corrected_v2",
    )
    check(
        pipeline.get("annotation_freeze_sha256") == sha256(root / FREEZE),
        "pipeline_annotation_freeze_hash_mismatch",
    )
    check(
        datetime.fromisoformat(freeze["created_utc"])
        < datetime.fromisoformat(pipeline["created_utc"]),
        "annotation_freeze_did_not_precede_scan",
    )
    freeze_hash_failures = []
    for label, expected in freeze["pinned_sha256"].items():
        path = Path(label)
        actual_path = path if path.is_absolute() else root / path
        observed = sha256(actual_path) if actual_path.is_file() else None
        if observed != expected:
            freeze_hash_failures.append(
                {"path": label, "expected": expected, "observed": observed}
            )
    check(not freeze_hash_failures, "annotation_freeze_hash_failure")

    candidates = read_tsv(root / CANDIDATES)
    frozen_order = candidate_freeze["result_summary"]["tier_a_families"]
    check(
        [row["orthogroup"] for row in candidates] == frozen_order,
        "candidate_order_or_membership_mismatch",
    )
    check(
        len(candidates)
        == int(candidate_freeze["result_summary"]["tier_counts"]["A"]),
        "candidate_count_mismatch",
    )

    mapping = read_tsv(root / MAP)
    map_by_id = {row["sequence_id"]: row for row in mapping}
    check(len(map_by_id) == len(mapping), "duplicate_map_sequence_id")
    expected_by_og: dict[str, dict[str, int]] = {}
    for row in candidates:
        expected_by_og[row["orthogroup"]] = {
            "arabidopsis": len(split_ids(row["arabidopsis_gene_ids"])),
            "prunus": len(split_ids(row["prunus_gene_ids"])),
            "pyrus": len(split_ids(row["pyrus_gene_ids"])),
        }
    observed_by_og: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in mapping:
        observed_by_og[row["orthogroup"]][row["genus"]] += 1
    check(
        all(
            dict(observed_by_og[og]) == expected
            for og, expected in expected_by_og.items()
        ),
        "extracted_genus_counts_mismatch",
    )

    sequences = read_fasta(root / FASTA)
    check(set(sequences) == set(map_by_id), "fasta_map_inventory_mismatch")
    sequence_integrity = all(
        len(sequences[sequence_id]) == int(row["protein_length"])
        and hashlib.sha256(sequences[sequence_id].encode("ascii")).hexdigest()
        == row["sequence_sha256"]
        for sequence_id, row in map_by_id.items()
    )
    check(sequence_integrity, "fasta_length_or_hash_mismatch")

    raw_hits = parse_domtbl(root / DOMTBL)
    check(
        all(row["sequence_id"] in map_by_id for row in raw_hits),
        "unknown_domtbl_query",
    )
    check(
        all(
            int(row["query_length"]) == len(sequences[str(row["sequence_id"])])
            for row in raw_hits
        ),
        "domtbl_query_length_mismatch",
    )

    hit_table = read_tsv(root / HITS)
    raw_keys = sorted(
        (
            str(row["sequence_id"]),
            str(row["pfam_accession"]),
            int(row["domain_index"]),
            int(row["hmm_from"]),
            int(row["hmm_to"]),
            int(row["ali_from"]),
            int(row["ali_to"]),
            float(row["domain_i_evalue"]),
            float(row["domain_score"]),
        )
        for row in raw_hits
    )
    table_keys = sorted(
        (
            row["sequence_id"],
            row["pfam_accession"],
            int(row["domain_index"]),
            int(row["hmm_from"]),
            int(row["hmm_to"]),
            int(row["ali_from"]),
            int(row["ali_to"]),
            float(row["domain_i_evalue"]),
            float(row["domain_score"]),
        )
        for row in hit_table
    )
    check(raw_keys == table_keys, "long_form_hit_table_not_exact")

    raw_by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_hits:
        raw_by_sequence[str(row["sequence_id"])].append(row)
    sequence_table = read_tsv(root / SEQUENCE_TABLE)
    sequence_by_id = {row["sequence_id"]: row for row in sequence_table}
    check(
        len(sequence_by_id) == len(mapping)
        and set(sequence_by_id) == set(map_by_id),
        "sequence_annotation_inventory_mismatch",
    )
    sequence_counts_exact = all(
        int(sequence_by_id[sequence_id]["pfam_domain_hit_count"])
        == len(raw_by_sequence.get(sequence_id, []))
        and int(sequence_by_id[sequence_id]["pfam_distinct_accession_count"])
        == len(
            {
                str(hit["pfam_accession"])
                for hit in raw_by_sequence.get(sequence_id, [])
            }
        )
        for sequence_id in map_by_id
    )
    check(sequence_counts_exact, "sequence_domain_counts_mismatch")

    totals: dict[tuple[str, str], int] = defaultdict(int)
    domain_sequences: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    domain_name: dict[tuple[str, str], str] = {}
    for row in mapping:
        totals[(row["orthogroup"], row["genus"])] += 1
    for hit in raw_hits:
        source = map_by_id[str(hit["sequence_id"])]
        key = (
            source["orthogroup"],
            str(hit["pfam_accession"]),
            source["genus"],
        )
        domain_sequences[key].add(str(hit["sequence_id"]))
        domain_name[(source["orthogroup"], str(hit["pfam_accession"]))] = str(
            hit["pfam_name"]
        )
    reconstructed_consensus = {}
    for og, accession in domain_name:
        counts = {
            genus: len(domain_sequences[(og, accession, genus)])
            for genus in ("arabidopsis", "prunus", "pyrus")
        }
        reconstructed_consensus[(og, accession)] = {
            "pfam_name": domain_name[(og, accession)],
            "arabidopsis_detected": counts["arabidopsis"],
            "arabidopsis_total": totals[(og, "arabidopsis")],
            "prunus_detected": counts["prunus"],
            "prunus_total": totals[(og, "prunus")],
            "pyrus_detected": counts["pyrus"],
            "pyrus_total": totals[(og, "pyrus")],
            "support_label": support_label(
                counts["arabidopsis"], counts["prunus"], counts["pyrus"]
            ),
        }
    consensus = read_tsv(root / CONSENSUS)
    consensus_by_key = {
        (row["orthogroup"], row["pfam_accession"]): row for row in consensus
    }
    check(
        set(consensus_by_key) == set(reconstructed_consensus),
        "consensus_domain_inventory_mismatch",
    )
    consensus_exact = True
    for key, expected in reconstructed_consensus.items():
        observed = consensus_by_key.get(key)
        if observed is None:
            consensus_exact = False
            continue
        for field, value in expected.items():
            if field.endswith("_detected") or field.endswith("_total"):
                equal = int(observed[field]) == int(value)
            else:
                equal = observed[field] == value
            if not equal:
                consensus_exact = False
    check(consensus_exact, "consensus_fields_mismatch")

    reconstructed = {
        "candidate_orthogroups": len(candidates),
        "extracted_proteins": len(mapping),
        "proteins_with_pfam_hit": sum(
            bool(raw_by_sequence.get(sequence_id)) for sequence_id in map_by_id
        ),
        "proteins_without_pfam_hit": sum(
            not bool(raw_by_sequence.get(sequence_id)) for sequence_id in map_by_id
        ),
        "domain_hit_rows": len(raw_hits),
        "orthogroup_domain_rows": len(reconstructed_consensus),
        "cross_genus_anchor_supported_domain_rows": sum(
            row["support_label"] == "cross_genus_anchor_supported"
            for row in reconstructed_consensus.values()
        ),
    }
    check(
        all(int(summary[key]) == value for key, value in reconstructed.items()),
        "summary_counts_mismatch",
    )

    pipeline_hash_failures = []
    for label, expected in {
        **pipeline["raw_output_sha256"],
        str(SUMMARY): pipeline["domain_summary_sha256"],
    }.items():
        path = root / label
        observed = sha256(path) if path.is_file() else None
        if observed != expected:
            pipeline_hash_failures.append(
                {"path": label, "expected": expected, "observed": observed}
            )
    check(not pipeline_hash_failures, "pipeline_output_hash_failure")
    check(
        summary.get("status") == "pass"
        and pipeline.get("status") == "pass"
        and not summary.get("violations")
        and not pipeline.get("violations"),
        "summary_or_pipeline_not_passing",
    )
    check(
        not any(
            payload.get("model_outputs_accessed") or payload.get("malus_accessed")
            for payload in (freeze, candidate_freeze, summary, pipeline)
        ),
        "model_or_malus_access_flag",
    )

    audit = {
        "status": "pass" if not failures else "fail",
        "scope": "independent_corrected_tier_a_full_pfam_audit",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_independent_of_summarizer": True,
        "selection_authority": False,
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "reconstructed": reconstructed,
        "candidate_order": frozen_order,
        "freeze_hash_failures": freeze_hash_failures,
        "pipeline_hash_failures": pipeline_hash_failures,
        "failure_count": len(failures),
        "failures": failures,
        "audited_output_sha256": {
            str(path): sha256(root / path)
            for path in (
                MAP,
                FASTA,
                DOMTBL,
                SEQUENCE_TABLE,
                HITS,
                CONSENSUS,
                SUMMARY,
                PIPELINE,
            )
        },
    }
    output = root / OUT
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
