#!/usr/bin/env python3
"""Build and audit the label-independent Prunus v2.1 gene-ID bridge."""

from __future__ import annotations

import argparse
import os
import gzip
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SOURCE_PATTERN = re.compile(r"^(Prupe\.\d+G\d+)\.v2\.1$")
TECHNICAL_PATTERN = re.compile(r"^(Pper\d+G\d+\.v1\.442)$")
MIN_FUNCTIONAL_MAPPING_FRACTION = 0.98
MIN_ORTHOGROUP_MAPPING_FRACTION = 0.95


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def parse_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for field in text.strip().strip(";").split(";"):
        if "=" in field:
            key, value = field.split("=", 1)
            attributes[key] = value
    return attributes


def parse_gene_bridge(gff_path: Path) -> pd.DataFrame:
    opener = gzip.open if gff_path.suffix == ".gz" else open
    records: list[dict[str, object]] = []
    with opener(gff_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2].lower() != "gene":
                continue
            attributes = parse_attributes(fields[8])
            technical_match = TECHNICAL_PATTERN.fullmatch(
                attributes.get("ID", "")
            )
            source_match = SOURCE_PATTERN.fullmatch(
                attributes.get("Source_ID", "")
            )
            if technical_match is None or source_match is None:
                continue
            records.append(
                {
                    "source_gene_id": source_match.group(1),
                    "technical_gene_id": technical_match.group(1),
                    "chromosome": fields[0],
                    "start_1based": int(fields[3]),
                    "end_1based": int(fields[4]),
                    "strand": fields[6],
                    "mapping_authority": (
                        "Prunus persica v2.1 GFF gene ID/Source_ID"
                    ),
                }
            )
    bridge = pd.DataFrame(records).sort_values(
        ["chromosome", "start_1based", "technical_gene_id"], kind="stable"
    )
    if bridge.empty:
        raise RuntimeError("no canonical Prunus gene ID pairs were parsed")
    return bridge.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--gff",
        type=Path,
        default=Path(
            "data/raw/functional_genomes/prunus_persica/annotation.gff.gz"
        ),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    expected = Path(os.environ["TREEFM_ROOT"]).resolve() if os.environ.get("TREEFM_ROOT") else None
    if expected is not None and root != expected:
        raise SystemExit(f"refusing to run outside {expected}: {root}")
    gff_path = args.gff if args.gff.is_absolute() else root / args.gff
    functional_path = (
        root
        / "data/processed/functional/Prunus_publication_v3/"
        "promoter_labels.parquet"
    )
    technical_path = (
        root
        / "data/processed/technical_benchmarks_publication_v3_26/"
        "prunus_persica.parquet"
    )
    for path in (gff_path, functional_path, technical_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    bridge = parse_gene_bridge(gff_path)
    source_duplicate_rows = int(
        bridge["source_gene_id"].duplicated(keep=False).sum()
    )
    technical_duplicate_rows = int(
        bridge["technical_gene_id"].duplicated(keep=False).sum()
    )

    technical = pd.read_parquet(
        technical_path, columns=["gene_id", "chromosome", "orthogroup"]
    )
    technical["orthogroup"] = technical["orthogroup"].fillna("").astype(str)
    orthogroup_counts = technical.groupby("gene_id")["orthogroup"].nunique()
    technical_gene_has_multiple_orthogroups = int(
        orthogroup_counts.gt(1).sum()
    )
    technical = technical.drop_duplicates("gene_id", keep="first")
    bridge_qc = bridge.merge(
        technical,
        left_on="technical_gene_id",
        right_on="gene_id",
        how="left",
        validate="one_to_one",
        suffixes=("_gff", "_technical"),
    )

    functional = pd.read_parquet(
        functional_path,
        columns=["gene_id", "chromosome", "label_binary"],
    )
    functional_qc = functional.merge(
        bridge_qc,
        left_on="gene_id",
        right_on="source_gene_id",
        how="left",
        validate="one_to_one",
        suffixes=("_functional", "_bridge"),
    )
    mapped = functional_qc["technical_gene_id"].notna()
    orthogroup_mapped = functional_qc["orthogroup"].fillna("").ne("")
    functional_mapping_fraction = float(mapped.mean())
    orthogroup_mapping_fraction = float(orthogroup_mapped.mean())
    class_qc: dict[str, dict[str, object]] = {}
    for label, data in functional_qc.groupby("label_binary", observed=True):
        class_qc[str(int(label))] = {
            "rows": int(len(data)),
            "bridge_mapped": int(data["technical_gene_id"].notna().sum()),
            "bridge_mapping_fraction": float(
                data["technical_gene_id"].notna().mean()
            ),
            "orthogroup_mapped": int(
                data["orthogroup"].fillna("").ne("").sum()
            ),
            "orthogroup_mapping_fraction": float(
                data["orthogroup"].fillna("").ne("").mean()
            ),
        }

    checks = {
        "canonical_gene_pairs_present": bool(len(bridge) >= 25_000),
        "source_ids_are_one_to_one": source_duplicate_rows == 0,
        "technical_ids_are_one_to_one": technical_duplicate_rows == 0,
        "technical_genes_have_at_most_one_orthogroup": (
            technical_gene_has_multiple_orthogroups == 0
        ),
        "functional_bridge_mapping_at_least_0_98": (
            functional_mapping_fraction >= MIN_FUNCTIONAL_MAPPING_FRACTION
        ),
        "functional_orthogroup_mapping_at_least_0_95": (
            orthogroup_mapping_fraction >= MIN_ORTHOGROUP_MAPPING_FRACTION
        ),
        "functional_to_gff_chromosomes_agree": bool(
            functional_qc.loc[mapped, "chromosome"].eq(
                functional_qc.loc[mapped, "chromosome_gff"]
            ).all()
        ),
        "gff_to_technical_chromosomes_agree": bool(
            functional_qc.loc[
                orthogroup_mapped, "chromosome_gff"
            ].eq(
                functional_qc.loc[
                    orthogroup_mapped, "chromosome_technical"
                ]
            ).all()
        ),
        "both_functional_label_classes_reported": set(class_qc) == {"0", "1"},
    }
    passed = bool(all(checks.values()))

    output = (
        root / "metadata/publication_v3_prunus_v21_gene_id_bridge.tsv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    bridge.to_csv(partial, sep="\t", index=False)
    os.replace(partial, output)
    audit = {
        "status": "pass" if passed else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "docs/publication_v3_prunus_gene_id_bridge_contract.md",
        "mapping_construction_used_functional_outcomes": False,
        "mapping_authority": "Prunus persica v2.1 GFF gene ID/Source_ID",
        "thresholds": {
            "minimum_functional_mapping_fraction": (
                MIN_FUNCTIONAL_MAPPING_FRACTION
            ),
            "minimum_orthogroup_mapping_fraction": (
                MIN_ORTHOGROUP_MAPPING_FRACTION
            ),
        },
        "inputs": {
            "gff": {
                "path": str(gff_path.relative_to(root)),
                "sha256": sha256(gff_path),
            },
            "functional_dataset": {
                "path": str(functional_path.relative_to(root)),
                "sha256": sha256(functional_path),
            },
            "technical_benchmark": {
                "path": str(technical_path.relative_to(root)),
                "sha256": sha256(technical_path),
            },
        },
        "bridge": {
            "path": str(output.relative_to(root)),
            "sha256": sha256(output),
            "rows": int(len(bridge)),
            "source_unique": int(bridge["source_gene_id"].nunique()),
            "technical_unique": int(bridge["technical_gene_id"].nunique()),
            "source_duplicate_rows": source_duplicate_rows,
            "technical_duplicate_rows": technical_duplicate_rows,
        },
        "functional_qc": {
            "rows": int(len(functional_qc)),
            "bridge_mapped": int(mapped.sum()),
            "bridge_mapping_fraction": functional_mapping_fraction,
            "orthogroup_mapped": int(orthogroup_mapped.sum()),
            "orthogroup_mapping_fraction": orthogroup_mapping_fraction,
            "class_qc": class_qc,
        },
        "checks": checks,
    }
    audit_path = (
        root
        / "results/metrics/publication_v3_prunus_gene_id_bridge_audit.json"
    )
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
