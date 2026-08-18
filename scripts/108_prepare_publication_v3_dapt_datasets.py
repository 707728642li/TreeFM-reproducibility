#!/usr/bin/env python3
"""Consolidate the four publication-v3 DAPT corpora into HF-ready splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORPORA = ("tree", "herb", "random_plant", "phylogc_match")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_buckets(data: pd.DataFrame, modulus: int) -> np.ndarray:
    keys = zip(
        data["slug"],
        data["chromosome"],
        data["start_0based"],
        data["end_0based"],
    )

    def bucket(parts: tuple[object, object, object, object]) -> int:
        key = "|".join(map(str, parts))
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % modulus

    return np.fromiter(
        (bucket(parts) for parts in keys),
        dtype=np.uint16,
        count=len(data),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("data/processed/publication_v3_dapt_corpora"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/publication_v3_dapt_hf"),
    )
    parser.add_argument("--validation-modulus", type=int, default=100)
    parser.add_argument("--validation-remainder", type=int, default=0)
    args = parser.parse_args()
    root = args.project_root.resolve()
    corpus_root = (
        args.corpus_root
        if args.corpus_root.is_absolute()
        else root / args.corpus_root
    )
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else root / args.output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    integrity_records: list[dict[str, object]] = []
    for corpus in CORPORA:
        paths = sorted((corpus_root / corpus).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no corpus shards for {corpus}")
        columns = [
            "slug",
            "chromosome",
            "start_0based",
            "end_0based",
            "sequence",
        ]
        data = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in paths],
            ignore_index=True,
        )
        duplicate_coordinates = int(
            data.duplicated(
                ["slug", "chromosome", "start_0based", "end_0based"]
            ).sum()
        )
        invalid_lengths = int((~data["sequence"].str.len().eq(512)).sum())
        invalid_bases = int(
            data["sequence"].str.contains(r"[^ACGTN]", regex=True).sum()
        )
        if (
            len(data) != 1_000_000
            or duplicate_coordinates
            or invalid_lengths
            or invalid_bases
        ):
            raise RuntimeError(
                f"{corpus} integrity failure: rows={len(data)}, "
                f"duplicate_coordinates={duplicate_coordinates}, "
                f"invalid_lengths={invalid_lengths}, invalid_bases={invalid_bases}"
            )
        buckets = split_buckets(data, args.validation_modulus)
        data = data.rename(
            columns={
                "slug": "assembly",
                "chromosome": "chrom",
                "start_0based": "start",
                "end_0based": "end",
                "sequence": "seq",
            }
        )
        data["strand"] = "+"
        output_dir = output_root / corpus
        output_dir.mkdir(parents=True, exist_ok=True)
        split_rows: dict[str, int] = {}
        for split, mask in (
            ("validation", buckets == args.validation_remainder),
            ("train", buckets != args.validation_remainder),
        ):
            output = output_dir / f"{split}.parquet"
            frame = data.loc[
                mask, ["assembly", "chrom", "start", "end", "strand", "seq"]
            ]
            partial = output.with_name(output.name + ".partial")
            partial.unlink(missing_ok=True)
            frame.to_parquet(partial, compression="zstd", index=False)
            partial.replace(output)
            split_rows[split] = len(frame)
            manifest_records.append(
                {
                    "corpus": corpus,
                    "split": split,
                    "rows": len(frame),
                    "bases": len(frame) * 512,
                    "species": frame["assembly"].nunique(),
                    "sha256": sha256(output),
                    "path": str(output.relative_to(root)),
                }
            )
            print(
                f"{corpus}\t{split}\trows={len(frame)}\t"
                f"sha256={manifest_records[-1]['sha256']}",
                flush=True,
            )
        integrity_records.append(
            {
                "corpus": corpus,
                "rows": len(data),
                "species": data["assembly"].nunique(),
                "duplicate_coordinates": duplicate_coordinates,
                "invalid_lengths": invalid_lengths,
                "invalid_bases": invalid_bases,
                "train_rows": split_rows["train"],
                "validation_rows": split_rows["validation"],
                "split_rows_sum": sum(split_rows.values()),
            }
        )

    manifest = pd.DataFrame(manifest_records)
    manifest_path = output_root / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    integrity = pd.DataFrame(integrity_records)
    integrity_path = output_root / "integrity.tsv"
    integrity.to_csv(integrity_path, sep="\t", index=False)
    policy = {
        "status": "pass",
        "corpora": list(CORPORA),
        "rows_per_corpus": 1_000_000,
        "bases_per_corpus": 512_000_000,
        "hash": "blake2b-64",
        "key": "slug|chromosome|start_0based|end_0based",
        "validation_modulus": args.validation_modulus,
        "validation_remainder": args.validation_remainder,
        "manifest": str(manifest_path.relative_to(root)),
        "integrity": str(integrity_path.relative_to(root)),
    }
    (output_root / "split_policy.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
