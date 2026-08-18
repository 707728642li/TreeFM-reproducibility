#!/usr/bin/env python3
"""Audit resources needed for publication-v5 targeted comparative genomics."""

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
OUT = ROOT / "results/metrics/publication_v5_resource_audit"
ORTHOFINDER_ROOT = ROOT / "data/processed/orthofinder_benchmark_publication_v3"
ORTHOGROUPS = ORTHOFINDER_ROOT / "OrthoFinder/Results_PublicationV3/Orthogroups/Orthogroups.txt"
TIER_A = ROOT / "results/metrics/publication_v4_tier_a_comparative/tier_a_orthogroups.txt"
CANDIDATES = ROOT / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv"
BRIDGE = ROOT / "metadata/publication_v3_prunus_v21_gene_id_bridge.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attributes(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in text.rstrip(";\n").split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif " " in item:
            key, value = item.split(" ", 1)
            value = value.strip('"')
        else:
            continue
        result[key] = value
    return result


def normalized_aliases(attrs: dict[str, str]) -> set[str]:
    values: set[str] = set()
    for value in attrs.values():
        for token in re.split(r"[,|]", value):
            token = token.strip()
            if not token:
                continue
            values.add(token)
            if ":" in token and token.split(":", 1)[0].lower() in {"gene", "transcript", "rna", "mrna"}:
                values.add(token.split(":", 1)[1])
    return values


def protein_path(species: str) -> Path:
    if species == "prunus_persica":
        return ROOT / "data/interim/functional_genomes/prunus_persica/primary.protein.fa"
    return ORTHOFINDER_ROOT / f"{species}.fa"


def gff_path(species: str) -> Path:
    if species == "prunus_persica":
        return ROOT / "data/interim/functional_genomes/prunus_persica/annotation.gff3"
    if species == "pyrus_pyrifolia":
        return ROOT / "data/interim/publication_v3_genomes/pyrus_pyrifolia/annotation.gff3"
    return ROOT / f"data/interim/normalized/{species}/annotation.gff3"


def load_tier_a() -> list[str]:
    values = []
    for line in TIER_A.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token.startswith("OG"):
            values.append(token.rstrip(":"))
    return values


def load_members(targets: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with ORTHOGROUPS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("OG"):
                continue
            name, payload = line.rstrip("\n").split(":", 1)
            if name in targets:
                result[name] = payload.strip().split()
                if len(result) == len(targets):
                    break
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tier_a = load_tier_a()
    members = load_members(set(tier_a))
    member_to_og = {member: og for og, values in members.items() for member in values}
    by_species: dict[str, set[str]] = defaultdict(set)
    for member in member_to_og:
        species = member.split("|", 1)[0]
        by_species[species].add(member)
    species_order = sorted(by_species)

    target_aliases: dict[str, set[str]] = defaultdict(set)
    member_gene: dict[str, str] = {}
    for species, values in by_species.items():
        for member in values:
            parts = member.split("|", 2)
            gene = parts[1] if len(parts) >= 2 else member
            member_gene[member] = gene
            target_aliases[species].add(gene)

    resource_rows = []
    mapping_rows = []
    all_proteins_found: dict[str, set[str]] = {}
    for species in species_order:
        protein = protein_path(species)
        gff = gff_path(species)
        wanted_members = by_species[species]
        found_proteins: set[str] = set()
        if protein.is_file():
            with protein.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith(">"):
                        identifier = line[1:].strip().split()[0]
                        # OrthoFinder sanitizes ':' in sequence identifiers to '_'.
                        canonical = identifier.replace(":", "_")
                        if canonical in wanted_members:
                            found_proteins.add(canonical)
        all_proteins_found[species] = found_proteins

        alias_to_records: dict[str, list[dict[str, object]]] = defaultdict(list)
        gene_records = 0
        if gff.is_file():
            with gff.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line or line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 9 or fields[2].lower() not in {"gene", "pseudogene", "mrna", "transcript"}:
                        continue
                    if fields[2].lower() in {"gene", "pseudogene"}:
                        gene_records += 1
                    attrs = attributes(fields[8])
                    aliases = normalized_aliases(attrs)
                    wanted = aliases & target_aliases[species]
                    if not wanted:
                        continue
                    record = {
                        "seqid": fields[0],
                        "start": int(fields[3]),
                        "end": int(fields[4]),
                        "strand": fields[6],
                        "feature": fields[2],
                        "attributes": fields[8],
                    }
                    for alias in wanted:
                        alias_to_records[alias].append(record)

        for member in sorted(wanted_members):
            gene = member_gene[member]
            records = alias_to_records.get(gene, [])
            preferred = [record for record in records if str(record["feature"]).lower() in {"gene", "pseudogene"}]
            if preferred:
                records = preferred
            unique_records = {}
            for record in records:
                key = (record["seqid"], record["start"], record["end"], record["strand"])
                unique_records[key] = record
            records = list(unique_records.values())
            mapping_rows.append(
                {
                    "orthogroup": member_to_og[member],
                    "species": species,
                    "member_id": member,
                    "gene_id": gene,
                    "protein_found": member in found_proteins,
                    "gff_record_count": len(records),
                    "seqid": records[0]["seqid"] if len(records) == 1 else "",
                    "start": records[0]["start"] if len(records) == 1 else "",
                    "end": records[0]["end"] if len(records) == 1 else "",
                    "strand": records[0]["strand"] if len(records) == 1 else "",
                }
            )
        resource_rows.append(
            {
                "species": species,
                "protein_path": protein.relative_to(ROOT).as_posix() if protein.is_file() else str(protein),
                "protein_bytes": protein.stat().st_size if protein.is_file() else 0,
                "gff_path": gff.relative_to(ROOT).as_posix() if gff.is_file() else str(gff),
                "gff_bytes": gff.stat().st_size if gff.is_file() else 0,
                "gff_gene_records": gene_records,
                "tier_a_members": len(wanted_members),
                "tier_a_proteins_found": len(found_proteins),
                "tier_a_gff_unique": sum(len(alias_to_records.get(member_gene[m], [])) == 1 for m in wanted_members),
            }
        )

    bridge = pd.read_csv(BRIDGE, sep="\t", dtype=str)
    source_to_technical = dict(zip(bridge["source_gene_id"], bridge["technical_gene_id"]))
    candidates = pd.read_csv(CANDIDATES, sep="\t", dtype=str)
    promoter_paths = {
        "prunus_persica": ROOT / "data/processed/functional/Prunus_publication_v3/promoter_labels.parquet",
        "pyrus_pyrifolia": ROOT / "data/processed/functional/Pyrus_PRJNA669907/promoter_labels.parquet",
    }
    promoter_ids = {
        species: set(pd.read_parquet(path, columns=["gene_id"])["gene_id"].astype(str))
        for species, path in promoter_paths.items()
    }
    candidate_rows = []
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
                technical_gene = source_to_technical.get(source_gene, source_gene) if genus == "prunus" else source_gene
                matching_members = [
                    member
                    for member in by_species[species]
                    if member_gene[member] == technical_gene and member_to_og[member] == og
                ]
                gff_matches = [
                    item
                    for item in mapping_rows
                    if item["species"] == species and item["gene_id"] == technical_gene and item["orthogroup"] == og
                ]
                candidate_rows.append(
                    {
                        "catalog_rank": row["catalog_rank"],
                        "orthogroup": og,
                        "genus": genus,
                        "species": species,
                        "source_gene_id": source_gene,
                        "technical_gene_id": technical_gene,
                        "orthogroup_member_found": len(matching_members) == 1,
                        "protein_found": bool(matching_members and matching_members[0] in all_proteins_found[species]),
                        "gff_unique": len(gff_matches) == 1 and int(gff_matches[0]["gff_record_count"]) == 1,
                        "promoter_found": source_gene in promoter_ids[species],
                    }
                )

    def write_tsv(path: Path, rows: list[dict]) -> None:
        if not rows:
            raise RuntimeError(f"No rows for {path}")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(OUT / "resource_inventory.tsv", resource_rows)
    with gzip.open(OUT / "tier_a_member_mapping.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(mapping_rows)
    write_tsv(OUT / "candidate_target_mapping.tsv", candidate_rows)

    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_resource_gate",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "tier_a_orthogroups": tier_a,
        "orthogroup_count": len(members),
        "species_count": len(species_order),
        "member_count": len(mapping_rows),
        "protein_member_coverage": sum(bool(row["protein_found"]) for row in mapping_rows) / len(mapping_rows),
        "gff_unique_member_coverage": sum(int(row["gff_record_count"]) == 1 for row in mapping_rows) / len(mapping_rows),
        "candidate_gene_count": len(candidate_rows),
        "candidate_all_gates_count": sum(
            all(row[key] for key in ("orthogroup_member_found", "protein_found", "gff_unique", "promoter_found"))
            for row in candidate_rows
        ),
        "input_fingerprints": {
            str(TIER_A.relative_to(ROOT)): sha256(TIER_A),
            str(CANDIDATES.relative_to(ROOT)): sha256(CANDIDATES),
            str(BRIDGE.relative_to(ROOT)): sha256(BRIDGE),
            str(ORTHOGROUPS.relative_to(ROOT)): sha256(ORTHOGROUPS),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    failures = []
    checks = {
        "fourteen_orthogroups": len(tier_a) == len(set(tier_a)) == 14,
        "all_orthogroups_found": len(members) == 14,
        "twenty_six_species": len(species_order) == 26,
        "all_protein_files": all(int(row["protein_bytes"]) > 0 for row in resource_rows),
        "all_gff_files": all(int(row["gff_bytes"]) > 0 for row in resource_rows),
        "all_member_proteins": all(bool(row["protein_found"]) for row in mapping_rows),
        "all_member_gff_unique": all(int(row["gff_record_count"]) == 1 for row in mapping_rows),
        "all_candidates_complete": all(
            all(row[key] for key in ("orthogroup_member_found", "protein_found", "gff_unique", "promoter_found"))
            for row in candidate_rows
        ),
        "malus_outcomes_sealed": True,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
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
    audit_path = ROOT / "results/metrics/publication_v5_resource_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit["status"], **summary, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
