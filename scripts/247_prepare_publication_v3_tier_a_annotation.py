#!/usr/bin/env python3
"""Extract frozen Tier-A candidate and Arabidopsis anchor proteins.

This is a retrospective annotation utility. It never reads model outputs or
the sealed Malus endpoint and cannot change the frozen candidate catalog.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CANDIDATES = Path(
    "results/biological_cases/publication_v3_crossgenus_candidates/"
    "tier_a_candidates.tsv"
)
CANDIDATE_FREEZE = Path(
    "config/publication_v3_crossgenus_candidate_catalog_freeze.json"
)
CONTRACT = Path(
    "docs/publication_v3_tier_a_postselection_annotation_contract_v2.md"
)
BRIDGE = Path("metadata/publication_v3_prunus_v21_gene_id_bridge.tsv")
PROTEIN_FASTA = {
    "arabidopsis": Path(
        "data/processed/orthofinder_benchmark_publication_v3/"
        "arabidopsis_thaliana.fa"
    ),
    "prunus": Path(
        "data/processed/orthofinder_benchmark_publication_v3/"
        "prunus_persica.fa"
    ),
    "pyrus": Path(
        "data/processed/orthofinder_benchmark_publication_v3/"
        "pyrus_pyrifolia.fa"
    ),
}
OUT_DIR = Path("data/processed/publication_v3_tier_a_annotation")
OUT_FASTA = OUT_DIR / "tier_a_candidate_and_anchor_proteins.fa"
OUT_MAP = OUT_DIR / "tier_a_candidate_and_anchor_proteins.tsv"
OUT_MANIFEST = OUT_DIR / "extraction_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def split_ids(value: str) -> list[str]:
    return [item for item in str(value).split(";") if item]


def fasta_records(path: Path) -> Iterable[tuple[str, str]]:
    header: str | None = None
    sequence: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence).upper()
                header = line[1:]
                sequence = []
            else:
                if header is None:
                    raise RuntimeError(f"sequence before header in {path}")
                sequence.append(line)
    if header is not None:
        yield header, "".join(sequence).upper()


def build_index(path: Path) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for header, sequence in fasta_records(path):
        tokens = set(header.split("|"))
        tokens.add(header.split()[0])
        for token in tokens:
            if token and token not in index:
                index[token] = (header, sequence)
    return index


def load_bridge(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        bridge = {
            row["source_gene_id"]: row["technical_gene_id"] for row in rows
        }
    if len(bridge) != len(set(bridge)):
        raise RuntimeError("Prunus bridge has duplicate source identifiers")
    return bridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()

    candidate_path = root / CANDIDATES
    bridge_path = root / BRIDGE
    fasta_paths = {key: root / value for key, value in PROTEIN_FASTA.items()}
    freeze_path = root / CANDIDATE_FREEZE
    contract_path = root / CONTRACT
    for path in [
        candidate_path,
        freeze_path,
        contract_path,
        bridge_path,
        *fasta_paths.values(),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "pass" or freeze.get("freeze_version") != "2.0":
        raise RuntimeError("corrected candidate freeze v2 is not passing")
    if freeze.get("model_outputs_accessed") or freeze.get("malus_accessed"):
        raise RuntimeError("candidate freeze violates post-selection isolation")
    frozen_tier_a = list(freeze["result_summary"]["tier_a_families"])
    frozen_tier_counts = freeze["result_summary"]["tier_counts"]
    pinned_candidate = freeze["frozen_artifacts"][str(CANDIDATES)]["sha256"]
    if sha256(candidate_path) != pinned_candidate:
        raise RuntimeError("Tier-A candidate table hash differs from freeze v2")

    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))
    observed_tier_a = [row["orthogroup"] for row in candidates]
    if observed_tier_a != frozen_tier_a:
        raise RuntimeError("Tier-A order or membership differs from freeze v2")
    if len(candidates) != int(frozen_tier_counts["A"]):
        raise RuntimeError("Tier-A count differs from freeze v2")
    if any(row["tier"] != "A" for row in candidates):
        raise RuntimeError("non-Tier-A row in frozen Tier-A input")

    bridge = load_bridge(bridge_path)
    indexes = {key: build_index(path) for key, path in fasta_paths.items()}

    records: list[dict[str, object]] = []
    fasta_lines: list[str] = []
    missing: list[str] = []

    for row in candidates:
        orthogroup = row["orthogroup"]
        rank = int(row["catalog_rank"])
        requests: list[tuple[str, str, str]] = []
        for gene_id in split_ids(row["arabidopsis_gene_ids"]):
            requests.append(("arabidopsis", gene_id, gene_id))
        for gene_id in split_ids(row["prunus_gene_ids"]):
            technical = bridge.get(gene_id, "")
            requests.append(("prunus", gene_id, technical))
        for gene_id in split_ids(row["pyrus_gene_ids"]):
            requests.append(("pyrus", gene_id, gene_id))

        for genus, reported_gene_id, fasta_gene_id in requests:
            if not fasta_gene_id:
                missing.append(f"{orthogroup}:{genus}:{reported_gene_id}:bridge")
                continue
            hit = indexes[genus].get(fasta_gene_id)
            if hit is None:
                # Some FASTA headers use the gene identifier without a version.
                candidates_for_id = [
                    value
                    for key, value in indexes[genus].items()
                    if key.split(".v1.")[0] == fasta_gene_id.split(".v1.")[0]
                ]
                unique = {
                    (header, sequence) for header, sequence in candidates_for_id
                }
                hit = next(iter(unique)) if len(unique) == 1 else None
            if hit is None:
                missing.append(
                    f"{orthogroup}:{genus}:{reported_gene_id}:{fasta_gene_id}"
                )
                continue
            source_header, sequence = hit
            sequence_id = "|".join(
                [orthogroup, genus, reported_gene_id, fasta_gene_id]
            )
            fasta_lines.extend([f">{sequence_id}", sequence])
            records.append(
                {
                    "catalog_rank": rank,
                    "orthogroup": orthogroup,
                    "tier": "A",
                    "role": (
                        "arabidopsis_anchor"
                        if genus == "arabidopsis"
                        else "labeled_candidate"
                    ),
                    "genus": genus,
                    "reported_gene_id": reported_gene_id,
                    "fasta_gene_id": fasta_gene_id,
                    "sequence_id": sequence_id,
                    "source_header": source_header,
                    "protein_length": len(sequence),
                    "sequence_sha256": hashlib.sha256(
                        sequence.encode("ascii")
                    ).hexdigest(),
                }
            )

    if missing:
        raise RuntimeError("missing protein sequences: " + "; ".join(missing))
    sequence_ids = [str(record["sequence_id"]) for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise RuntimeError("duplicate extracted sequence identifier")
    expected_candidates = sum(
        len(split_ids(row["prunus_gene_ids"]))
        + len(split_ids(row["pyrus_gene_ids"]))
        for row in candidates
    )
    observed_candidates = sum(
        record["role"] == "labeled_candidate" for record in records
    )
    if observed_candidates != expected_candidates:
        raise RuntimeError(
            f"candidate protein count mismatch: {observed_candidates} "
            f"!= {expected_candidates}"
        )

    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(root / OUT_FASTA, "\n".join(fasta_lines))
    fieldnames = list(records[0])
    map_path = root / OUT_MAP
    partial = map_path.with_name(map_path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    os.replace(partial, map_path)

    counts: dict[str, dict[str, int]] = {}
    for orthogroup in [row["orthogroup"] for row in candidates]:
        group = [row for row in records if row["orthogroup"] == orthogroup]
        counts[orthogroup] = {
            "total": len(group),
            "labeled_candidates": sum(
                row["role"] == "labeled_candidate" for row in group
            ),
            "arabidopsis_anchors": sum(
                row["role"] == "arabidopsis_anchor" for row in group
            ),
        }
    manifest = {
        "status": "pass",
        "scope": "retrospective_corrected_tier_a_protein_extraction",
        "selection_authority": False,
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_freeze_version": freeze["freeze_version"],
        "candidate_count": len(candidates),
        "candidate_order": observed_tier_a,
        "extracted_sequence_count": len(records),
        "counts": counts,
        "inputs": {
            str(CONTRACT): sha256(contract_path),
            str(CANDIDATE_FREEZE): sha256(freeze_path),
            str(CANDIDATES): sha256(candidate_path),
            str(BRIDGE): sha256(bridge_path),
            **{
                str(PROTEIN_FASTA[key]): sha256(path)
                for key, path in fasta_paths.items()
            },
        },
        "outputs": {
            str(OUT_FASTA): sha256(root / OUT_FASTA),
            str(OUT_MAP): sha256(map_path),
        },
        "violations": [],
    }
    atomic_json(root / OUT_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
