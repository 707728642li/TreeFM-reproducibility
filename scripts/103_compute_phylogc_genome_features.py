#!/usr/bin/env python3
"""Compute outcome-independent genome, repeat, region, and taxonomy features."""

from __future__ import annotations

import argparse
import os
import gzip
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


TARGET_SPECIES = (
    "Hevea brasiliensis",
    "Prunus persica",
    "Pyrus pyrifolia",
    "Malus domestica",
)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def open_binary(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


VALID_DNA_RUN = re.compile(b"[ACGT]+")


def fasta_composition(path: Path) -> dict[str, int | float]:
    counts = {base: 0 for base in "ACGTN"}
    other = 0
    lowercase_non_n = 0
    valid_21mer_occurrences = 0
    current_valid_run = 0
    with open_binary(path) as handle:
        for line in handle:
            if line.startswith(b">"):
                current_valid_run = 0
                continue
            sequence = line.strip()
            lowercase_non_n += sum(sequence.count(base) for base in b"acgt")
            upper = sequence.upper()
            line_counts = {
                base: upper.count(base.encode("ascii")) for base in counts
            }
            for base, count in line_counts.items():
                counts[base] += count
            other += len(upper) - sum(line_counts.values())

            matches = list(VALID_DNA_RUN.finditer(upper))
            if not matches:
                current_valid_run = 0
                continue
            first = matches[0]
            if first.start() == 0:
                combined = current_valid_run + len(first.group())
                valid_21mer_occurrences += max(0, combined - 20) - max(
                    0, current_valid_run - 20
                )
            else:
                valid_21mer_occurrences += max(0, len(first.group()) - 20)
            for match in matches[1:]:
                valid_21mer_occurrences += max(0, len(match.group()) - 20)
            last = matches[-1]
            current_valid_run = len(last.group()) if last.end() == len(upper) else 0
    genome_bp = sum(counts.values()) + other
    valid = sum(counts[base] for base in "ACGT")
    return {
        "genome_bp": genome_bp,
        "a_bp": counts["A"],
        "c_bp": counts["C"],
        "g_bp": counts["G"],
        "t_bp": counts["T"],
        "n_bp": counts["N"] + other,
        "gc_fraction": (counts["G"] + counts["C"]) / valid if valid else 0.0,
        "n_fraction": (counts["N"] + other) / genome_bp if genome_bp else 1.0,
        "softmask_fraction": lowercase_non_n / valid if valid else 0.0,
        "valid_21mer_occurrences": valid_21mer_occurrences,
    }


def fasta_softmask_fraction(path: Path) -> float:
    lowercase_non_n = 0
    valid = 0
    with open_binary(path) as handle:
        for line in handle:
            if line.startswith(b">"):
                continue
            sequence = line.strip()
            lowercase_non_n += sum(sequence.count(base) for base in b"acgt")
            upper = sequence.upper()
            valid += sum(upper.count(base) for base in (b"A", b"C", b"G", b"T"))
    return lowercase_non_n / valid if valid else 0.0


def union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    return total + current_end - current_start + 1


def annotation_composition(path: Path) -> dict[str, int]:
    intervals: dict[str, dict[str, list[tuple[int, int]]]] = {
        "gene": defaultdict(list),
        "CDS": defaultdict(list),
    }
    feature_counts = {"gene": 0, "CDS": 0}
    with open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in intervals:
                continue
            feature = fields[2]
            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError:
                continue
            if start > end or start < 1:
                continue
            intervals[feature][fields[0]].append((start, end))
            feature_counts[feature] += 1
    gene_union = sum(
        union_length(seq_intervals)
        for seq_intervals in intervals["gene"].values()
    )
    cds_union = sum(
        union_length(seq_intervals)
        for seq_intervals in intervals["CDS"].values()
    )
    return {
        "gene_features": feature_counts["gene"],
        "cds_features": feature_counts["CDS"],
        "gene_union_bp": gene_union,
        "cds_union_bp": cds_union,
    }


def safe_rmtree(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve(strict=False)
    allowed = allowed_root.resolve(strict=True)
    if not resolved.is_relative_to(allowed):
        raise RuntimeError(f"refusing removal outside KMC temp root: {path}")
    if path.exists():
        shutil.rmtree(path)


def kmc_repeat_fraction(
    slug: str,
    genome: Path,
    kmc: Path,
    kmc_tools: Path,
    tmp_root: Path,
    threads: int,
    memory_gb: int,
    log_root: Path,
    valid_21mer_occurrences: int,
) -> dict[str, int | float]:
    work = tmp_root / slug
    safe_rmtree(work, tmp_root)
    work.mkdir()
    database = work / "database"
    kmc_tmp = work / "tmp"
    kmc_tmp.mkdir()
    log_path = log_root / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        counted = subprocess.run(
            [
                str(kmc),
                "-k21",
                "-ci1",
                "-cs10000",
                "-fm",
                f"-t{threads}",
                f"-m{memory_gb}",
                str(genome),
                str(database),
                str(kmc_tmp),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if counted.returncode != 0:
            raise RuntimeError(f"KMC failed for {slug}; see {log_path}")
        histogram = work / "histogram.tsv"
        transformed = subprocess.run(
            [
                str(kmc_tools),
                "transform",
                str(database),
                "histogram",
                str(histogram),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if transformed.returncode != 0 or not histogram.is_file():
        raise RuntimeError(f"KMC histogram failed for {slug}; see {log_path}")
    low_copy_occurrences = 0
    distinct_total = 0
    with histogram.open(encoding="utf-8") as handle:
        for line in handle:
            count_text, distinct_text = line.split()
            count = int(count_text)
            distinct = int(distinct_text)
            occurrences = count * distinct
            distinct_total += distinct
            if count < 10:
                low_copy_occurrences += occurrences
    if valid_21mer_occurrences <= 0:
        raise RuntimeError(f"no valid 21-mer occurrences for {slug}")
    repetitive_occurrences = valid_21mer_occurrences - low_copy_occurrences
    if repetitive_occurrences < 0:
        raise RuntimeError(
            f"KMC low-copy occurrences exceed FASTA positions for {slug}: "
            f"{low_copy_occurrences}>{valid_21mer_occurrences}"
        )
    result = {
        "canonical_21mers_distinct": distinct_total,
        "canonical_21mer_occurrences": valid_21mer_occurrences,
        "canonical_21mer_occurrences_count_1_to_9": low_copy_occurrences,
        "canonical_21mer_occurrences_count_ge_10": repetitive_occurrences,
        "repetitive_21mer_fraction": (
            repetitive_occurrences / valid_21mer_occurrences
        ),
    }
    safe_rmtree(work, tmp_root)
    return result


def analyze_species(task: dict[str, object]) -> dict[str, object]:
    genome = Path(str(task["genome"]))
    annotation = Path(str(task["annotation"]))
    composition = fasta_composition(genome)
    annotation_stats = annotation_composition(annotation)
    softmask_path = str(task.get("softmask", ""))
    softmask_fraction = None
    if softmask_path and Path(softmask_path).is_file():
        softmask_fraction = fasta_softmask_fraction(Path(softmask_path))
    repeat = kmc_repeat_fraction(
        str(task["slug"]),
        genome,
        Path(str(task["kmc"])),
        Path(str(task["kmc_tools"])),
        Path(str(task["tmp_root"])),
        int(task["kmc_threads"]),
        int(task["kmc_memory_gb"]),
        Path(str(task["log_root"])),
        int(composition["valid_21mer_occurrences"]),
    )
    genome_bp = int(composition["genome_bp"])
    gene_fraction = annotation_stats["gene_union_bp"] / genome_bp
    cds_fraction = annotation_stats["cds_union_bp"] / genome_bp
    record = {
        "scientific_name": task["scientific_name"],
        "slug": task["slug"],
        "feature_role": task["feature_role"],
        "order": task["order"],
        "family": task["family"],
        "life_form": task["life_form"],
        "genome": str(genome),
        "annotation": str(annotation),
        **composition,
        **annotation_stats,
        **repeat,
        "source_softmask_fraction": softmask_fraction,
        "gene_fraction": gene_fraction,
        "cds_fraction": cds_fraction,
        "intergenic_fraction": max(0.0, 1.0 - gene_fraction),
    }
    checkpoint_root = Path(str(task["checkpoint_root"]))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / f"{task['slug']}.json"
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    payload = {
        "checkpoint_version": 1,
        "signature": task_signature(task),
        "record": record,
    }
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, checkpoint)
    return record


def task_signature(task: dict[str, object]) -> dict[str, object]:
    assets: dict[str, object] = {}
    for key in ("genome", "annotation", "softmask"):
        value = str(task.get(key, ""))
        path = Path(value) if value else None
        assets[key] = (
            {"path": value, "bytes": path.stat().st_size}
            if path is not None and path.is_file()
            else {"path": value, "bytes": 0}
        )
    return {
        "checkpoint_version": 1,
        "slug": str(task["slug"]),
        "scientific_name": str(task["scientific_name"]),
        "feature_role": str(task["feature_role"]),
        "assets": assets,
        "kmer": 21,
        "repetitive_count_threshold": 10,
    }


def load_taxonomy(
    names_path: Path, nodes_path: Path
) -> tuple[dict[str, int], dict[int, int]]:
    taxids_by_name: dict[str, set[int]] = defaultdict(set)
    with names_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = [field.strip() for field in line.split("|")]
            if len(fields) >= 4 and fields[3] in {
                "scientific name",
                "synonym",
                "equivalent name",
            }:
                taxids_by_name[fields[1]].add(int(fields[0]))
    names = {
        name: next(iter(taxids))
        for name, taxids in taxids_by_name.items()
        if len(taxids) == 1
    }
    parents: dict[int, int] = {}
    with nodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = [field.strip() for field in line.split("|")]
            parents[int(fields[0])] = int(fields[1])
    return names, parents


def lineage(taxid: int, parents: dict[int, int]) -> list[int]:
    result = [taxid]
    while parents[result[-1]] != result[-1]:
        result.append(parents[result[-1]])
    return result


def taxonomy_distance(left: int, right: int, parents: dict[int, int]) -> int:
    left_lineage = lineage(left, parents)
    right_positions = {
        taxid: index for index, taxid in enumerate(lineage(right, parents))
    }
    for left_steps, taxid in enumerate(left_lineage):
        if taxid in right_positions:
            return left_steps + right_positions[taxid]
    raise RuntimeError(f"taxonomy lineages do not meet: {left}, {right}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kmc-threads", type=int, default=8)
    parser.add_argument("--kmc-memory-gb", type=int, default=24)
    args = parser.parse_args()
    root = args.project_root.resolve()
    env = root / "envs/treefm-match/bin"
    kmc = env / "kmc"
    kmc_tools = env / "kmc_tools"
    for required in (kmc, kmc_tools):
        if not required.is_file():
            raise FileNotFoundError(required)
    tmp_root = root / "results/tmp/phylogc_kmc"
    tmp_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = root / "results/tmp/phylogc_feature_records"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root = root / "logs/phylogc_kmc"

    panel = pd.read_csv(
        root / "config/publication_v3_panel.tsv", sep="\t", dtype=str
    ).fillna("")
    tree = panel.loc[
        panel["include"].eq("1") & panel["dapt_role"].eq("Tree")
    ].copy()
    seed_taxonomy = pd.read_csv(
        root / "config/species_panel_seed.tsv", sep="\t", dtype=str
    ).set_index("slug")
    file_manifest = pd.read_csv(
        root / "metadata/file_manifest.tsv", sep="\t", dtype=str
    ).fillna("")
    source_genome_by_name = (
        file_manifest.loc[file_manifest["file_type"].eq("genome")]
        .set_index("scientific_name")["source_path"]
        .to_dict()
    )
    tasks: list[dict[str, object]] = []
    for row in tree.to_dict(orient="records"):
        taxonomy = seed_taxonomy.loc[row["slug"]]
        source = Path(source_genome_by_name[row["scientific_name"]])
        softmask = ""
        if source.name.endswith(".dna.toplevel.fa.gz"):
            candidate = source.with_name(
                source.name.replace(".dna.toplevel.fa.gz", ".dna_sm.toplevel.fa.gz")
            )
            if candidate.is_file():
                softmask = str(candidate)
        tasks.append(
            {
                "scientific_name": row["scientific_name"],
                "slug": row["slug"],
                "feature_role": "tree_target",
                "order": taxonomy["order"],
                "family": taxonomy["family"],
                "life_form": row["life_form"],
                "genome": root / "data/interim/normalized" / row["slug"] / "genome.fa",
                "annotation": root
                / "data/interim/normalized"
                / row["slug"]
                / "annotation.gff3",
                "softmask": softmask,
            }
        )

    discovery = pd.read_csv(
        root / "metadata/publication_v3_phylogc_source_discovery.tsv",
        sep="\t",
        dtype=str,
    ).fillna("")
    discovery = discovery.loc[
        discovery["source_status"].eq("complete_same_release")
    ]
    for row in discovery.to_dict(orient="records"):
        raw = root / "data/raw/publication_v3_phylogc_candidates" / row["slug"]
        tasks.append(
            {
                "scientific_name": row["scientific_name"],
                "slug": row["slug"],
                "feature_role": "candidate",
                "order": row["order"],
                "family": row["family"],
                "life_form": row["life_form"],
                "genome": raw / "genome.fa.gz",
                "annotation": raw / "annotation.gff3.gz",
                "softmask": (
                    str(raw / "softmask.fa.gz")
                    if (raw / "softmask.fa.gz").is_file()
                    else ""
                ),
            }
        )
    for task in tasks:
        for key in ("genome", "annotation"):
            if not Path(str(task[key])).is_file():
                raise FileNotFoundError(task[key])
        task.update(
            {
                "kmc": kmc,
                "kmc_tools": kmc_tools,
                "tmp_root": tmp_root,
                "kmc_threads": args.kmc_threads,
                "kmc_memory_gb": args.kmc_memory_gb,
                "log_root": log_root,
                "checkpoint_root": checkpoint_root,
            }
        )

    records: list[dict[str, object]] = []
    pending_tasks: list[dict[str, object]] = []
    for task in tasks:
        checkpoint = checkpoint_root / f"{task['slug']}.json"
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pending_tasks.append(task)
            continue
        if (
            payload.get("checkpoint_version") == 1
            and payload.get("signature") == task_signature(task)
            and isinstance(payload.get("record"), dict)
        ):
            record = payload["record"]
            records.append(record)
            print(
                f"checkpoint\t{record['feature_role']}\t{record['slug']}",
                flush=True,
            )
        else:
            pending_tasks.append(task)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyze_species, task): str(task["slug"])
            for task in pending_tasks
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"{record['feature_role']}\t{record['slug']}\t"
                f"GC={record['gc_fraction']:.4f}\trepeat21={record['repetitive_21mer_fraction']:.4f}",
                flush=True,
            )

    public_genome_root = Path(os.environ.get("PUBLIC_GENOME_ROOT", "public_genomes"))
    names, parents = load_taxonomy(
        public_genome_root / "species_info/NCBI/names.dmp",
        public_genome_root / "species_info/NCBI/nodes.dmp",
    )
    requested_names = set(
        str(record["scientific_name"]) for record in records
    ) | set(TARGET_SPECIES)
    missing_names = sorted(requested_names - set(names))
    if missing_names:
        raise RuntimeError(f"scientific names absent from NCBI taxonomy: {missing_names}")
    target_taxids = {name: names[name] for name in TARGET_SPECIES}
    for record in records:
        taxid = names[str(record["scientific_name"])]
        record["ncbi_taxid"] = taxid
        for target_name, target_taxid in target_taxids.items():
            key = "taxonomy_distance_" + target_name.lower().replace(" ", "_")
            record[key] = taxonomy_distance(taxid, target_taxid, parents)

    features = pd.DataFrame(records).sort_values(["feature_role", "slug"])
    output_path = root / "metadata/publication_v3_phylogc_genome_features.tsv"
    features.to_csv(output_path, sep="\t", index=False)
    summary = {
        "status": "pass",
        "tree_species": int(features["feature_role"].eq("tree_target").sum()),
        "candidate_species": int(features["feature_role"].eq("candidate").sum()),
        "kmer": 21,
        "repetitive_count_threshold": 10,
        "taxonomy_targets": list(TARGET_SPECIES),
        "output": str(output_path.relative_to(root)),
    }
    summary_path = root / "metadata/publication_v3_phylogc_genome_features.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
