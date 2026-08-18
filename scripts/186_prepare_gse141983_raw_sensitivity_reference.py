#!/usr/bin/env python3
"""Prepare the frozen NCBIv2 peach reference and 2-kb TSS windows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd


CANONICAL_GENE = re.compile(r"^(Prupe\.[1-8]G\d{6})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decompress(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with gzip.open(source, "rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    temporary.replace(destination)


def fasta_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    name: str | None = None
    length = 0
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = length
                name = line[1:].split()[0]
                length = 0
            else:
                length += len(line.strip())
    if name is not None:
        lengths[name] = length
    return lengths


def attributes(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in text.rstrip().split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--genome-source",
        default="data/raw/functional_genomes/prunus_persica/genome.fa.gz",
    )
    parser.add_argument(
        "--annotation-source",
        default="data/raw/functional_genomes/prunus_persica/annotation.gff.gz",
    )
    parser.add_argument(
        "--label-parquet",
        default="data/processed/functional/Prunus_publication_v3/"
        "promoter_labels.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/functional_v3/GSE141983_raw_sensitivity/reference",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    genome_source = root / args.genome_source
    annotation_source = root / args.annotation_source
    label_path = root / args.label_parquet
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    genome = output_root / "Prunus_persica_NCBIv2.fa"
    annotation = output_root / "Prunus_persica_NCBIv2.gff"
    saf = output_root / "Prunus_persica_NCBIv2_TSS_minus1000_plus999.saf"

    decompress(genome_source, genome)
    decompress(annotation_source, annotation)
    lengths = fasta_lengths(genome)
    violations: list[str] = []
    expected_chromosomes = {f"Pp{index:02d}" for index in range(1, 9)}
    if not expected_chromosomes.issubset(lengths):
        violations.append("canonical_chromosomes_missing_from_fasta")

    rows: dict[str, tuple[str, int, int, str]] = {}
    conflicting: set[str] = set()
    with annotation.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chromosome, _, _, start_text, end_text, _, strand, _, attr_text = fields
            attr = attributes(attr_text)
            match = CANONICAL_GENE.match(attr.get("Source_ID", ""))
            if not match or chromosome not in lengths:
                continue
            gene = match.group(1)
            start = int(start_text)
            end = int(end_text)
            tss = start if strand == "+" else end
            window_start = max(1, tss - 1000)
            window_end = min(lengths[chromosome], tss + 999)
            record = (chromosome, window_start, window_end, strand)
            if gene in rows and rows[gene] != record:
                conflicting.add(gene)
            else:
                rows[gene] = record
    for gene in conflicting:
        rows.pop(gene, None)
    if len(rows) < 20_000:
        violations.append(f"too_few_unique_tss_windows:{len(rows)}")

    with saf.open("w", encoding="utf-8", newline="") as handle:
        handle.write("GeneID\tChr\tStart\tEnd\tStrand\n")
        for gene in sorted(rows):
            chromosome, start, end, strand = rows[gene]
            handle.write(f"{gene}\t{chromosome}\t{start}\t{end}\t{strand}\n")

    labels = pd.read_parquet(label_path, columns=["gene_id", "label", "endpoint_direction"])
    labels = labels.drop_duplicates("gene_id")
    label_ids = set(labels["gene_id"].astype(str))
    positive_ids = set(labels.loc[labels["label"] == "positive", "gene_id"].astype(str))
    negative_ids = set(labels.loc[labels["label"] == "negative", "gene_id"].astype(str))
    mapped = label_ids.intersection(rows)
    mapped_positive = positive_ids.intersection(rows)
    mapped_negative = negative_ids.intersection(rows)
    positive_fraction = len(mapped_positive) / max(1, len(positive_ids))
    negative_fraction = len(mapped_negative) / max(1, len(negative_ids))
    if positive_fraction < 0.70:
        violations.append("positive_label_reference_mapping_below_0.70")
    if negative_fraction < 0.70:
        violations.append("negative_label_reference_mapping_below_0.70")

    manifest = {
        "status": "pass" if not violations else "fail",
        "scope": "outcome_free_gse141983_raw_reference_and_tss_window_gate",
        "scientific_decision_authority": False,
        "malus_accessed": False,
        "violations": violations,
        "reference": "Prunus_persica_NCBIv2_Phytozome_v2.1",
        "tss_window": {"relative_start": -1000, "relative_end": 999},
        "source_sha256": {
            args.genome_source: sha256_file(genome_source),
            args.annotation_source: sha256_file(annotation_source),
            args.label_parquet: sha256_file(label_path),
        },
        "output_sha256": {
            str(genome.relative_to(root)): sha256_file(genome),
            str(annotation.relative_to(root)): sha256_file(annotation),
            str(saf.relative_to(root)): sha256_file(saf),
        },
        "fasta_sequences": len(lengths),
        "canonical_chromosomes": sorted(expected_chromosomes.intersection(lengths)),
        "unique_gene_windows": len(rows),
        "conflicting_gene_windows_excluded": len(conflicting),
        "frozen_label_genes": len(label_ids),
        "mapped_label_genes": len(mapped),
        "positive_mapping_fraction": positive_fraction,
        "negative_mapping_fraction": negative_fraction,
    }
    manifest_path = output_root / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
