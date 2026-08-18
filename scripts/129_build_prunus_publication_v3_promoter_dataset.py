#!/usr/bin/env python3
"""Attach frozen robust labels to the Prunus 2,048-bp promoter dataset."""

from __future__ import annotations

import argparse
import os
import gzip
import json
from pathlib import Path

import pandas as pd


CHUNKS = tuple(f"promoter_chunk_{index}" for index in range(1, 5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    expected = Path(os.environ["TREEFM_ROOT"]).resolve() if os.environ.get("TREEFM_ROOT") else None
    if expected is not None and root != expected:
        raise SystemExit(f"refusing to run outside {expected}: {root}")

    source = pd.read_parquet(
        root
        / "data/processed/functional/"
        "prunus_dormancy_sequence_targets.parquet"
    )
    labels = pd.read_csv(
        root
        / "results/biological_cases/prunus_publication_v3/"
        "prunus_publication_v3_robust_labels.tsv.gz",
        sep="\t",
    )
    labels = labels.loc[labels["label"].isin(["positive", "negative"])].copy()
    if labels.empty or set(labels["label"]) != {"positive", "negative"}:
        raise RuntimeError("both robust Prunus label classes are required")
    if source["gene_id"].duplicated().any() or labels["gene_id"].duplicated().any():
        raise RuntimeError("Prunus sequence or label genes are duplicated")
    keep_label_columns = [
        "gene_id",
        "label",
        "endpoint_direction",
        "positive_supporting_studies",
        "equivalent_studies",
        "median_absolute_study_effect",
    ]
    frame = source.drop(
        columns=[
            "robust_consensus_target",
            "consensus_median_log2cpm",
            "sign_concordance",
        ],
        errors="ignore",
    ).merge(
        labels[keep_label_columns],
        on="gene_id",
        how="inner",
        validate="one_to_one",
    )
    frame = frame.loc[
        frame["chromosome"].isin([f"Pp0{index}" for index in range(1, 9)])
    ].copy()
    frame["label_binary"] = frame["label"].eq("positive").astype("int8")
    frame["split"] = "test"
    frame.loc[
        frame["chromosome"].isin(
            [f"Pp0{index}" for index in range(1, 6)]
        ),
        "split",
    ] = "train"
    frame.loc[frame["chromosome"].eq("Pp06"), "split"] = "development"
    for column in ("promoter_2048", *CHUNKS):
        expected_length = 2048 if column == "promoter_2048" else 512
        if not frame[column].str.len().eq(expected_length).all():
            raise RuntimeError(f"invalid Prunus sequence length: {column}")
    if frame["promoter_2048"].str.count("N").div(2048).gt(0.10).any():
        raise RuntimeError("Prunus promoter ambiguity exceeds the frozen gate")
    split_counts = (
        frame.groupby(["split", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    if not all(
        split in split_counts.index
        and split_counts.loc[split].get("positive", 0) > 0
        and split_counts.loc[split].get("negative", 0) > 0
        for split in ("train", "development", "test")
    ):
        raise RuntimeError("a Prunus chromosome split lacks one label class")
    frame = frame.sort_values(["chromosome", "gene_id"], kind="stable")

    output_root = (
        root / "data/processed/functional/Prunus_publication_v3"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = output_root / "promoter_labels.parquet"
    tsv_path = output_root / "promoter_labels.tsv.gz"
    frame.to_parquet(parquet_path, compression="zstd", index=False)
    with gzip.open(tsv_path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False)
    summary = {
        "status": "pass",
        "contract": "docs/publication_v3_prunus_functional_label_contract.md",
        "input_labeled_genes": len(labels),
        "promoters_retained": len(frame),
        "positive_promoters": int(frame["label_binary"].sum()),
        "negative_promoters": int((frame["label_binary"] == 0).sum()),
        "split_class_counts": {
            split: {
                label: int(value)
                for label, value in split_counts.loc[split].to_dict().items()
            }
            for split in split_counts.index
        },
        "parquet": str(parquet_path.relative_to(root)),
        "tsv_gz": str(tsv_path.relative_to(root)),
    }
    result_root = (
        root / "results/biological_cases/prunus_publication_v3"
    )
    (result_root / "promoter_dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
