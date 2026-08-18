#!/usr/bin/env python3
"""Materialize and audit the three new publication-v3 genome resources."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd


def opener(path: Path, mode: str):
    return gzip.open(path, mode) if path.suffix == ".gz" else path.open(mode)


def materialize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent, delete=False
    ) as temporary_handle:
        temporary = Path(temporary_handle.name)
    try:
        with opener(source, "rb") as source_handle, temporary.open("wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=8 * 1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def fasta_records(path: Path):
    name = None
    description = ""
    chunks: list[str] = []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield name, description, "".join(chunks).upper()
                description = line[1:].strip()
                name = description.split(None, 1)[0]
                chunks = []
            elif name is not None:
                chunks.append(line.strip())
        if name is not None:
            yield name, description, "".join(chunks).upper()


def genome_stats(path: Path) -> dict[str, object]:
    lengths = []
    base_counts = Counter()
    for _, _, sequence in fasta_records(path):
        lengths.append(len(sequence))
        base_counts.update(sequence)
    total = sum(lengths)
    threshold = total / 2
    cumulative = 0
    n50 = 0
    for length in sorted(lengths, reverse=True):
        cumulative += length
        if cumulative >= threshold:
            n50 = length
            break
    acgt = sum(base_counts[base] for base in "ACGT")
    return {
        "sequence_count": len(lengths),
        "genome_bp": total,
        "n50_bp": n50,
        "n_fraction": base_counts["N"] / total if total else 1.0,
        "gc_fraction": (base_counts["G"] + base_counts["C"]) / acgt if acgt else 0.0,
    }


def parse_attributes(text: str) -> dict[str, str]:
    attributes = {}
    for field in text.strip().strip(";").split(";"):
        if not field:
            continue
        if "=" in field:
            key, value = field.split("=", 1)
            attributes[key.strip()] = value.strip()
    return attributes


def annotation_stats(path: Path) -> dict[str, object]:
    features = Counter()
    seqids = set()
    gene_ids = set()
    transcript_ids = set()
    with opener(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            feature = fields[2].lower()
            features[feature] += 1
            seqids.add(fields[0])
            attributes = parse_attributes(fields[8])
            identifier = attributes.get("ID", "")
            if feature == "gene" and identifier:
                gene_ids.add(identifier.removeprefix("gene:"))
            if feature in {"mrna", "transcript", "primary_transcript"} and identifier:
                transcript_ids.add(identifier.removeprefix("transcript:"))
    return {
        "annotation_seqids": len(seqids),
        "genes": len(gene_ids) or features["gene"],
        "transcripts": len(transcript_ids) or features["mrna"] + features["transcript"],
        "cds_features": features["cds"],
    }


def protein_gene(description: str, record_id: str) -> tuple[str, str]:
    gene_match = re.search(r"(?:^|\s)gene:([^\s]+)", description)
    transcript_match = re.search(r"(?:^|\s)transcript:([^\s]+)", description)
    if gene_match is None:
        gene_match = re.search(r"(?:^|\s)Gene=([^\s]+)", description)
    if transcript_match is None:
        transcript_match = re.search(r"(?:^|\s)mRNA=([^\s]+)", description)
    gene = gene_match.group(1) if gene_match else record_id
    transcript = transcript_match.group(1) if transcript_match else record_id
    return gene, transcript


def select_primary_proteins(source: Path, destination: Path, slug: str) -> dict[str, object]:
    selected: dict[str, tuple[str, str]] = {}
    all_records = 0
    rejected_short = 0
    rejected_stop = 0
    for record_id, description, sequence in fasta_records(source):
        all_records += 1
        sequence = sequence.rstrip("*")
        if len(sequence) < 30:
            rejected_short += 1
            continue
        if "*" in sequence:
            rejected_stop += 1
            continue
        gene, transcript = protein_gene(description, record_id)
        incumbent = selected.get(gene)
        if incumbent is None or (len(sequence), transcript) > (len(incumbent[1]), incumbent[0]):
            selected[gene] = (transcript, sequence)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="ascii", newline="\n", prefix=destination.name + ".",
        suffix=".partial", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for gene in sorted(selected):
            transcript, sequence = selected[gene]
            handle.write(f">{slug}|{gene}|{transcript}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    temporary.replace(destination)
    return {
        "all_protein_records": all_records,
        "primary_proteins": len(selected),
        "rejected_short_proteins": rejected_short,
        "rejected_internal_stop": rejected_stop,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("config/publication_v3_genome_sources.tsv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("metadata/publication_v3_genome_basic_qc.tsv")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    sources = pd.read_csv(root / args.config, sep="\t", dtype=str).fillna("")
    records = []
    for slug, frame in sources.groupby("slug", sort=True):
        assets = {
            row.asset_type: (root / row.project_destination).resolve()
            for row in frame.itertuples(index=False)
        }
        if set(assets) != {"genome", "annotation", "protein"}:
            raise ValueError(f"incomplete asset set for {slug}: {sorted(assets)}")
        output_root = root / "data/interim/publication_v3_genomes" / slug
        genome = output_root / "genome.fa"
        annotation = output_root / "annotation.gff3"
        primary = output_root / "primary.protein.fa"
        materialize(assets["genome"], genome)
        materialize(assets["annotation"], annotation)
        record = {
            "scientific_name": frame.iloc[0]["scientific_name"],
            "slug": slug,
            "assembly": frame.iloc[0]["assembly"],
            **genome_stats(genome),
            **annotation_stats(annotation),
            **select_primary_proteins(assets["protein"], primary, slug),
            "genome_fasta": str(genome),
            "annotation_gff3": str(annotation),
            "primary_proteome": str(primary),
        }
        record["gene_protein_mapping_fraction"] = (
            record["primary_proteins"] / record["genes"] if record["genes"] else 0.0
        )
        records.append(record)
        print(
            f"{slug}\tgenome_bp={record['genome_bp']}\tgenes={record['genes']}\t"
            f"primary_proteins={record['primary_proteins']}",
            flush=True,
        )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {output} with {len(records)} genomes")


if __name__ == "__main__":
    main()
