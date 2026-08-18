#!/usr/bin/env python3
"""Audit non-blind functional schemas without constructing Malus outcomes."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd


def delimited_number_sum(value: object) -> int:
    total = 0
    for token in str(value).split(";"):
        token = token.strip()
        if token and token.lower() != "nan":
            total += int(token)
    return total


def count_gzip_lines(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metadata/publication_v3_functional_schema_audit.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    audit: dict[str, object] = {"blind_outcomes_opened": False}

    existing = root / "data/processed/functional/GSE124820"
    counts = pd.read_parquet(existing / "counts.parquet")
    metadata = pd.read_csv(existing / "sample_metadata.tsv", sep="\t", dtype=str)
    days = metadata["title"].str.extract(r"(?i)day\s*(\d+)", expand=False)
    audit["GSE124820"] = {
        "gene_rows": int(counts.shape[0]),
        "samples": int(counts.shape[1]),
        "metadata_rows": int(len(metadata)),
        "species_counts": metadata["organism_ch1"].value_counts().to_dict(),
        "genotypes": sorted(metadata["genotype"].dropna().unique().tolist()),
        "days": sorted(pd.to_numeric(days, errors="coerce").dropna().astype(int).unique().tolist()),
        "explicit_gene_ids": not pd.RangeIndex(len(counts)).equals(counts.index),
        "first_gene_ids": [str(value) for value in counts.index[:5]],
    }

    gse232 = root / "data/raw/functional_v3/GSE232062/GSE232062_ControlAcclimation_gene_count_geo.csv.gz"
    gse232_head = pd.read_csv(gse232, nrows=5)
    audit["GSE232062"] = {
        "data_rows": count_gzip_lines(gse232) - 1,
        "sample_columns": int(gse232_head.shape[1]),
        "first_columns": list(gse232_head.columns[:5]),
        "explicit_gene_id_column": any(
            re.search(r"gene|id|locus", str(column), flags=re.IGNORECASE)
            for column in gse232_head.columns
        ),
        "status": "raw_reprocessing_or_author_mapping_required",
    }

    gse771 = root / "data/raw/functional_v3/GSE77119/GSE77119_ED_SD_PD.txt.gz"
    gse771_frame = pd.read_csv(
        gse771,
        sep="\t",
        usecols=["Symbol", "Transcripts", "GeneID", "Endodormancy", "Summer_Buds", "Paradormancy"],
        low_memory=False,
    )
    audit["GSE77119"] = {
        "rows": int(len(gse771_frame)),
        "columns": list(gse771_frame.columns),
        "gene_id_nonmissing": int(gse771_frame.get("GeneID", pd.Series(dtype=object)).notna().sum()),
        "stage_columns": [
            column
            for column in gse771_frame.columns
            if str(column).lower() in {"endodormancy", "summer_buds", "paradormancy"}
        ],
        "first_gene_ids": [str(value) for value in gse771_frame["GeneID"].head(5)],
    }
    gse771_sequences = pd.read_csv(
        gse771,
        sep="\t",
        usecols=["GeneID", "Transcripts", "sequence"],
        low_memory=False,
    )
    sequence_lengths = gse771_sequences["sequence"].fillna("").astype(str).str.removeprefix(">").str.len()
    audit["GSE77119"]["sequence_nonempty"] = int((sequence_lengths > 30).sum())
    audit["GSE77119"]["sequence_length_median"] = float(sequence_lengths[sequence_lengths > 30].median())

    gse127 = root / "data/raw/functional_v3/GSE127322/GSE127322_processed_data.xlsx"
    workbook = pd.ExcelFile(gse127)
    sheet_records = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(gse127, sheet_name=sheet)
        record = {
            "sheet": sheet,
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
        }
        if "Gene_id" in frame.columns:
            record["first_gene_ids"] = [str(value) for value in frame["Gene_id"].head(5)]
            identifiers = frame["Gene_id"].fillna("").astype(str)
            record["gene_id_prefix_counts"] = {
                "Novel": int(identifiers.str.startswith("Novel").sum()),
                "Vitvi": int(identifiers.str.startswith("Vitvi").sum()),
                "VIT": int(identifiers.str.startswith("VIT_").sum()),
                "other": int(
                    (~identifiers.str.startswith(("Novel", "Vitvi", "VIT_"))).sum()
                ),
            }
        sheet_records.append(record)
    audit["GSE127322"] = {"sheets": sheet_records}

    ena_path = root / "data/blind/publication_v3/malus/PRJNA374502/PRJNA374502_ena_run_manifest.tsv"
    ena = pd.read_csv(ena_path, sep="\t", dtype=str)
    audit["PRJNA374502_blind_metadata_only"] = {
        "runs": int(len(ena)),
        "scientific_names": sorted(ena["scientific_name"].dropna().unique().tolist()),
        "library_strategies": sorted(ena["library_strategy"].dropna().unique().tolist()),
        "library_layouts": sorted(ena["library_layout"].dropna().unique().tolist()),
        "fastq_bytes": int(ena["fastq_bytes"].map(delimited_number_sum).sum()),
        "outcome_fields_read": [],
    }

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "datasets": len(audit) - 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
