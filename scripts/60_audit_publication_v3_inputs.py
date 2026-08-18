#!/usr/bin/env python3
"""Audit publication-v3 frozen panel, genome copies, and functional inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--require-copied-genomes", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("metadata/publication_v3_predata_audit.json"),
    )
    parser.add_argument(
        "--asset-output",
        type=Path,
        default=Path("metadata/publication_v3_genome_asset_audit.tsv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    panel_path = root / "config/publication_v3_panel.tsv"
    v2_path = root / "config/mpu_panel_frozen_v2.tsv"
    accessions_path = root / "config/publication_v3_functional_accessions.tsv"
    genomes_path = root / "config/publication_v3_genome_sources.tsv"

    for path in (panel_path, v2_path, accessions_path, genomes_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    panel = pd.read_csv(panel_path, sep="\t", dtype=str).fillna("")
    v2 = pd.read_csv(v2_path, sep="\t", dtype=str).fillna("")
    accessions = pd.read_csv(accessions_path, sep="\t", dtype=str).fillna("")
    genomes = pd.read_csv(genomes_path, sep="\t", dtype=str).fillna("")

    checks: dict[str, dict[str, object]] = {}

    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"passed": bool(passed), "evidence": evidence}

    check(
        "panel_28_unique_species",
        len(panel) == 28 and panel["slug"].nunique() == 28,
        {"rows": len(panel), "unique_slugs": panel["slug"].nunique()},
    )
    role_counts = panel.groupby("dapt_role")["slug"].count().to_dict()
    check(
        "inherited_dapt_counts",
        role_counts.get("Tree", 0) == 13 and role_counts.get("Herb", 0) == 6,
        role_counts,
    )
    primary = sorted(panel.loc[panel["analysis_tier"] == "primary_test", "slug"])
    check(
        "three_primary_genera_frozen",
        primary == ["hevea_brasiliensis", "prunus_persica", "vitis_vinifera"],
        primary,
    )
    blind = panel.loc[panel["analysis_tier"] == "external_blind", "slug"].tolist()
    check("single_publication_blind_genus", blind == ["malus_domestica"], blind)

    inherited = panel.loc[panel["slug"].isin(v2["slug"])].copy()
    role_columns = [
        "slug",
        "life_form",
        "dapt_role",
        "downstream_role",
        "include",
        "analysis_tier",
        "primary_inference",
    ]
    v2_roles = v2[role_columns].sort_values("slug").reset_index(drop=True)
    inherited_roles = inherited[role_columns].sort_values("slug").reset_index(drop=True)
    unchanged = inherited_roles.equals(v2_roles)
    changed_rows = []
    if not unchanged:
        merged = v2_roles.merge(
            inherited_roles,
            on="slug",
            how="outer",
            suffixes=("_v2", "_v3"),
            indicator=True,
        )
        for _, row in merged.iterrows():
            differences = [
                column
                for column in role_columns[1:]
                if row.get(f"{column}_v2", "") != row.get(f"{column}_v3", "")
            ]
            if differences or row["_merge"] != "both":
                changed_rows.append({"slug": row["slug"], "differences": differences})
    check("v2_roles_unchanged", unchanged, changed_rows)

    required_genera = {"Prunus", "Vitis", "Malus"}
    accession_genera = set(accessions["genus"])
    check(
        "functional_three_genera",
        required_genera.issubset(accession_genera),
        sorted(accession_genera),
    )
    check(
        "functional_accessions_unique",
        accessions["accession"].nunique() == len(accessions),
        {"rows": len(accessions), "unique": accessions["accession"].nunique()},
    )

    processed_rows = accessions[accessions["current_state"] == "processed"]
    processed_evidence = []
    processed_ok = True
    for row in processed_rows.itertuples(index=False):
        directory = root / row.server_path
        counts = directory / "counts.parquet"
        metadata = directory / "sample_metadata.tsv"
        exists = directory.is_dir() and counts.is_file() and metadata.is_file()
        evidence: dict[str, object] = {
            "accession": row.accession,
            "directory": str(directory),
            "exists": exists,
        }
        if exists:
            count_frame = pd.read_parquet(counts)
            sample_frame = pd.read_csv(metadata, sep="\t", dtype=str)
            evidence.update(
                {
                    "count_rows": int(count_frame.shape[0]),
                    "count_columns": int(count_frame.shape[1]),
                    "metadata_rows": int(sample_frame.shape[0]),
                    "counts_sha256": sha256(counts),
                    "metadata_sha256": sha256(metadata),
                }
            )
            plausible = count_frame.shape[0] > 1000 and sample_frame.shape[0] > 10
            processed_ok &= plausible
            evidence["plausible"] = plausible
        else:
            processed_ok = False
        processed_evidence.append(evidence)
    check("existing_processed_functional_inputs", processed_ok, processed_evidence)

    asset_records = []
    asset_ok = True
    copied_ok = True
    for row in genomes.itertuples(index=False):
        source = Path(row.nas_source)
        destination = root / row.project_destination
        source_exists = source.is_file()
        destination_exists = destination.is_file()
        asset_ok &= source_exists
        copied_ok &= destination_exists
        record = {
            "scientific_name": row.scientific_name,
            "slug": row.slug,
            "assembly": row.assembly,
            "asset_type": row.asset_type,
            "nas_source": str(source),
            "project_destination": str(destination),
            "source_exists": source_exists,
            "destination_exists": destination_exists,
            "source_bytes": source.stat().st_size if source_exists else 0,
            "destination_bytes": destination.stat().st_size if destination_exists else 0,
            "source_sha256": sha256(source) if source_exists else "",
            "destination_sha256": sha256(destination) if destination_exists else "",
        }
        record["copy_matches"] = bool(
            source_exists
            and destination_exists
            and record["source_bytes"] == record["destination_bytes"]
            and record["source_sha256"] == record["destination_sha256"]
        )
        if destination_exists:
            copied_ok &= record["copy_matches"]
        asset_records.append(record)
    check("nas_genome_assets_present", asset_ok, {"assets": len(asset_records)})
    check(
        "genomes_copied_and_verified",
        copied_ok,
        {
            "required_now": args.require_copied_genomes,
            "copied": sum(record["destination_exists"] for record in asset_records),
            "matching": sum(record["copy_matches"] for record in asset_records),
            "total": len(asset_records),
        },
    )

    asset_output = root / args.asset_output
    asset_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(asset_records).to_csv(asset_output, sep="\t", index=False)

    required_check_names = [name for name in checks if name != "genomes_copied_and_verified"]
    if args.require_copied_genomes:
        required_check_names.append("genomes_copied_and_verified")
    passed = all(checks[name]["passed"] for name in required_check_names)
    payload = {
        "protocol_version": "publication-v3-predata-0.1",
        "passed": passed,
        "required_checks": required_check_names,
        "checks": checks,
    }
    json_output = root / args.json_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": passed, "checks": len(required_check_names)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
