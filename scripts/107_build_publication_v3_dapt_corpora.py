#!/usr/bin/env python3
"""Build the four frozen, equal-budget publication-v3 DAPT corpora."""

from __future__ import annotations

import argparse
import os
import bisect
import gzip
import hashlib
import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta


WINDOW_COLUMNS = (
    "corpus",
    "scientific_name",
    "slug",
    "life_form",
    "chromosome",
    "start_0based",
    "end_0based",
    "gc_fraction",
    "n_fraction",
    "sequence",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, corpus: str, slug: str) -> int:
    payload = f"{seed}|{corpus}|{slug}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def allocate(total: int, slugs: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(slugs))
    return {
        slug: base + int(index < remainder)
        for index, slug in enumerate(sorted(slugs))
    }


def prepare_matched_genome(root: Path, slug: str) -> Path:
    source = (
        root
        / "data/raw/publication_v3_phylogc_candidates"
        / slug
        / "genome.fa.gz"
    )
    destination = (
        root
        / "data/interim/normalized_publication_v3_phylogc"
        / slug
        / "genome.fa"
    )
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    with gzip.open(source, "rb") as input_handle, partial.open("wb") as output:
        shutil.copyfileobj(input_handle, output, length=8 * 1024 * 1024)
    if partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"empty decompressed genome for {slug}")
    partial.replace(destination)
    return destination


def audit_existing(
    path: Path, target: int, window_length: int
) -> pd.DataFrame | None:
    frame = pd.read_parquet(path)
    if (
        len(frame) != target
        or list(frame.columns) != list(WINDOW_COLUMNS)
        or not frame["sequence"].str.len().eq(window_length).all()
        or frame.duplicated(["chromosome", "start_0based"]).any()
        or frame["sequence"].str.contains(r"[^ACGTN]", regex=True).any()
    ):
        return None
    return frame


def sample_species(
    row: dict[str, str],
    corpus: str,
    target: int,
    genome_path: str,
    output_root: str,
    window_length: int,
    max_n_fraction: float,
    seed: int,
) -> dict[str, object]:
    slug = row["slug"]
    output_path = Path(output_root) / corpus / f"{slug}.parquet"
    frame = (
        audit_existing(output_path, target, window_length)
        if output_path.is_file()
        else None
    )
    if frame is not None:
        attempts = target
        rejected_n = 0
        rejected_ambiguous = 0
        rejected_duplicate = 0
        provenance = "existing_audited"
    else:
        rng = random.Random(stable_seed(seed, corpus, slug))
        fasta = Fasta(
            genome_path,
            as_raw=True,
            sequence_always_upper=True,
            rebuild=False,
        )
        contigs = [
            (name, len(fasta[name]))
            for name in fasta.keys()
            if len(fasta[name]) >= window_length
        ]
        total_positions = sum(
            length - window_length + 1 for _, length in contigs
        )
        if total_positions <= 0:
            raise RuntimeError(f"no eligible contigs for {slug}")
        cumulative: list[int] = []
        running = 0
        for _, length in contigs:
            running += length - window_length + 1
            cumulative.append(running)

        records: list[dict[str, object]] = []
        seen: set[tuple[str, int]] = set()
        attempts = 0
        rejected_n = 0
        rejected_ambiguous = 0
        rejected_duplicate = 0
        max_attempts = max(target * 100, target + 1_000_000)
        while len(records) < target and attempts < max_attempts:
            attempts += 1
            draw = rng.randrange(total_positions)
            index = bisect.bisect_right(cumulative, draw)
            previous = cumulative[index - 1] if index else 0
            chromosome = contigs[index][0]
            start0 = draw - previous
            key = (chromosome, start0)
            if key in seen:
                rejected_duplicate += 1
                continue
            seen.add(key)
            sequence = str(
                fasta[chromosome][start0 : start0 + window_length]
            ).upper()
            n_count = sequence.count("N")
            valid_base_count = n_count + sum(
                sequence.count(base) for base in "ACGT"
            )
            if valid_base_count != len(sequence):
                rejected_ambiguous += 1
                continue
            n_fraction = n_count / window_length
            if len(sequence) != window_length or n_fraction > max_n_fraction:
                rejected_n += 1
                continue
            records.append(
                {
                    "corpus": corpus,
                    "scientific_name": row["scientific_name"],
                    "slug": slug,
                    "life_form": row["life_form"],
                    "chromosome": chromosome,
                    "start_0based": start0,
                    "end_0based": start0 + window_length,
                    "gc_fraction": (
                        sequence.count("G") + sequence.count("C")
                    )
                    / window_length,
                    "n_fraction": n_fraction,
                    "sequence": sequence,
                }
            )
        fasta.close()
        if len(records) != target:
            raise RuntimeError(
                f"{slug}/{corpus}: sampled {len(records)} of {target} "
                f"after {attempts} attempts"
            )
        frame = pd.DataFrame(records, columns=WINDOW_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(output_path.name + ".partial")
        partial.unlink(missing_ok=True)
        frame.to_parquet(partial, compression="zstd", index=False)
        partial.replace(output_path)
        provenance = "sampled"

    return {
        "corpus": corpus,
        "scientific_name": row["scientific_name"],
        "slug": slug,
        "life_form": row["life_form"],
        "target_windows": target,
        "written_windows": len(frame),
        "base_budget": len(frame) * window_length,
        "mean_gc_fraction": float(frame["gc_fraction"].mean()),
        "mean_n_fraction": float(frame["n_fraction"].mean()),
        "attempts": attempts,
        "rejected_n": rejected_n,
        "rejected_ambiguous": rejected_ambiguous,
        "rejected_duplicate": rejected_duplicate,
        "provenance": provenance,
        "output_parquet": str(output_path),
        "sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=46)
    parser.add_argument("--windows-per-corpus", type=int, default=1_000_000)
    parser.add_argument("--window-length", type=int, default=512)
    parser.add_argument("--max-n-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    root = args.project_root.resolve()
    expected = Path(os.environ["TREEFM_ROOT"]).resolve() if os.environ.get("TREEFM_ROOT") else None
    if expected is not None and root != expected:
        raise SystemExit(f"refusing to run outside {expected}: {root}")

    panel = pd.read_csv(
        root / "config/publication_v3_panel.tsv", sep="\t", dtype=str
    ).fillna("")
    panel = panel.loc[panel["include"].eq("1")]
    tree = panel.loc[panel["dapt_role"].eq("Tree")].copy()
    herb = panel.loc[panel["dapt_role"].eq("Herb")].copy()
    random_plant = pd.concat([tree, herb], ignore_index=True)
    selected = pd.read_csv(
        root / "config/publication_v3_phylogc_match_selected.tsv",
        sep="\t",
        dtype=str,
    ).fillna("")
    if len(tree) != 13 or len(herb) != 6 or len(selected) != 8:
        raise RuntimeError(
            "unexpected frozen cohort sizes: "
            f"Tree={len(tree)}, Herb={len(herb)}, PhyloGCMatch={len(selected)}"
        )
    if set(selected["slug"]) & set(panel["slug"]):
        raise RuntimeError("PhyloGCMatch is not species-disjoint from v3 panel")

    genome_paths: dict[str, Path] = {}
    for row in pd.concat([tree, herb], ignore_index=True).itertuples(index=False):
        path = root / "data/interim/normalized" / row.slug / "genome.fa"
        if not path.is_file():
            raise FileNotFoundError(path)
        genome_paths[row.slug] = path
    selected_rows = list(selected.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=min(8, len(selected_rows))) as executor:
        futures = {
            executor.submit(prepare_matched_genome, root, row.slug): row.slug
            for row in selected_rows
        }
        for future in as_completed(futures):
            slug = futures[future]
            genome_paths[slug] = future.result()

    # Build any missing FASTA indices serially before process-level sampling to
    # avoid multiple workers racing to create the same .fai.
    for path in sorted(set(genome_paths.values())):
        fasta = Fasta(str(path), as_raw=True, sequence_always_upper=True)
        fasta.close()

    cohorts = {
        "tree": tree,
        "herb": herb,
        "random_plant": random_plant,
        "phylogc_match": selected,
    }
    output_root = root / "data/processed/publication_v3_dapt_corpora"
    tasks: list[tuple[dict[str, str], str, int]] = []
    for corpus, cohort in cohorts.items():
        if corpus == "phylogc_match":
            allocations = {slug: 125_000 for slug in cohort["slug"]}
        else:
            allocations = allocate(
                args.windows_per_corpus, cohort["slug"].tolist()
            )
        if sum(allocations.values()) != args.windows_per_corpus:
            raise RuntimeError(f"unequal pre-sampling budget for {corpus}")
        tasks.extend(
            (row, corpus, allocations[row["slug"]])
            for row in cohort.to_dict(orient="records")
        )
    workers = min(args.workers, len(tasks))

    summaries: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                sample_species,
                row,
                corpus,
                target,
                str(genome_paths[row["slug"]]),
                str(output_root),
                args.window_length,
                args.max_n_fraction,
                args.seed,
            ): (corpus, row["slug"])
            for row, corpus, target in tasks
        }
        for future in as_completed(futures):
            record = future.result()
            summaries.append(record)
            print(
                f"{record['corpus']}\t{record['slug']}\t"
                f"windows={record['written_windows']}\t"
                f"GC={record['mean_gc_fraction']:.4f}",
                flush=True,
            )

    summary = pd.DataFrame(summaries).sort_values(["corpus", "slug"])
    summary_path = (
        root / "metadata/publication_v3_dapt_corpus_shards.tsv"
    )
    summary.to_csv(summary_path, sep="\t", index=False)
    corpus_audit = (
        summary.groupby("corpus", as_index=False)
        .agg(
            species=("slug", "nunique"),
            windows=("written_windows", "sum"),
            bases=("base_budget", "sum"),
            mean_gc_fraction=("mean_gc_fraction", "mean"),
            maximum_mean_n_fraction=("mean_n_fraction", "max"),
        )
        .sort_values("corpus")
    )
    expected_corpora = set(cohorts)
    passed = (
        set(corpus_audit["corpus"]) == expected_corpora
        and corpus_audit["windows"].eq(args.windows_per_corpus).all()
        and corpus_audit["bases"]
        .eq(args.windows_per_corpus * args.window_length)
        .all()
    )
    audit_path = root / "metadata/publication_v3_dapt_corpus_audit.tsv"
    corpus_audit.to_csv(audit_path, sep="\t", index=False)
    payload = {
        "status": "pass" if passed else "fail",
        "seed": args.seed,
        "window_length": args.window_length,
        "windows_per_corpus": args.windows_per_corpus,
        "corpora": sorted(expected_corpora),
        "shard_manifest": str(summary_path.relative_to(root)),
        "corpus_audit": str(audit_path.relative_to(root)),
    }
    payload_path = root / "metadata/publication_v3_dapt_corpus_gate.json"
    payload_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
