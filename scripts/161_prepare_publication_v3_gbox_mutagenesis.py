#!/usr/bin/env python3
"""Prepare deterministic fixed-G-box and matched-control promoter variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd


MOTIF = "CACGTG"
DISRUPTED_MOTIF = "CAGCTG"
WINDOW = len(MOTIF)
CONTROL_REPLICATES = 10
EXCLUSION_FLANK = 6
CONTRACT = Path(
    "docs/publication_v3_exploratory_gbox_model_attribution_plan.md"
)
DATASETS = {
    "prunus": Path(
        "data/processed/functional/Prunus_publication_v3/"
        "promoter_labels.parquet"
    ),
    "pyrus": Path(
        "data/processed/functional/Pyrus_PRJNA669907/"
        "promoter_labels.parquet"
    ),
}
OUTPUT_DIR = Path(
    "data/processed/functional/publication_v3_gbox_mutagenesis"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def exact_nonoverlapping_starts(sequence: str, motif: str) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while True:
        start = sequence.find(motif, cursor)
        if start < 0:
            return starts
        starts.append(start)
        cursor = start + len(motif)


def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def swap_middle(window: str) -> str:
    if len(window) != WINDOW:
        raise ValueError("unexpected perturbation-window length")
    chars = list(window)
    chars[2], chars[3] = chars[3], chars[2]
    return "".join(chars)


def mutate_windows(
    sequence: str,
    starts: list[int],
    expected: str | None = None,
) -> str:
    chars = list(sequence)
    for start in starts:
        window = sequence[start : start + WINDOW]
        if expected is not None and window != expected:
            raise RuntimeError(
                f"expected {expected} at {start}, observed {window}"
            )
        replacement = swap_middle(window)
        chars[start : start + WINDOW] = replacement
    return "".join(chars)


def control_candidates(
    sequence: str, motif_starts: list[int]
) -> list[int]:
    forbidden = [
        (
            max(0, start - EXCLUSION_FLANK),
            min(len(sequence), start + WINDOW + EXCLUSION_FLANK),
        )
        for start in motif_starts
    ]
    candidates: list[int] = []
    for start in range(0, len(sequence) - WINDOW + 1):
        span = (start, start + WINDOW)
        if any(overlaps(span, blocked) for blocked in forbidden):
            continue
        window = sequence[start : start + WINDOW]
        if set(window) - set("ACGT"):
            continue
        if sum(base in "GC" for base in window) != 4:
            continue
        if window == MOTIF or window[2] == window[3]:
            continue
        if swap_middle(window) == MOTIF:
            continue
        candidates.append(start)
    return candidates


def select_control_starts(
    candidates: list[int],
    required: int,
    seed: int,
) -> list[int] | None:
    ordered = candidates.copy()
    random.Random(seed).shuffle(ordered)
    selected: list[int] = []
    for start in ordered:
        span = (start, start + WINDOW)
        if any(
            overlaps(span, (chosen, chosen + WINDOW))
            for chosen in selected
        ):
            continue
        selected.append(start)
        if len(selected) == required:
            return sorted(selected)
    return None


def normalize_source(data: pd.DataFrame, genus: str) -> pd.DataFrame:
    required = {
        "gene_id",
        "chromosome",
        "label",
        "label_binary",
        "split",
        "promoter_2048",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"{genus} dataset lacks columns: {missing}")
    data = data.copy()
    data["promoter_2048"] = data["promoter_2048"].str.upper()
    if data["gene_id"].duplicated().any():
        raise RuntimeError(f"{genus} contains duplicate gene identifiers")
    if not data["promoter_2048"].str.len().eq(2048).all():
        raise RuntimeError(f"{genus} promoters are not all 2,048 bp")
    if set(data["label_binary"].astype(int)) != {0, 1}:
        raise RuntimeError(f"{genus} lacks both label classes")
    return data.sort_values(
        ["chromosome", "gene_id"], kind="stable"
    ).reset_index(drop=True)


def row_hash(data: pd.DataFrame) -> str:
    columns = [
        "variant_id",
        "genus",
        "gene_id",
        "chromosome",
        "label_binary",
        "split",
        "variant_type",
        "control_replicate",
        "gbox_count",
        "mutated_starts",
        "promoter_2048",
    ]
    values = pd.util.hash_pandas_object(
        data[columns], index=False
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    contract = root / CONTRACT
    script = Path(__file__).resolve()
    if not contract.is_file():
        raise FileNotFoundError(contract)

    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "promoter_variants.parquet"
    excluded_output = output_dir / "excluded_genes.tsv"
    manifest_path = output_dir / "manifest.json"

    source_hashes = {
        genus: sha256(root / path) for genus, path in DATASETS.items()
    }
    fingerprint_payload = {
        "contract_sha256": sha256(contract),
        "script_sha256": sha256(script),
        "source_sha256": source_hashes,
        "motif": MOTIF,
        "disrupted_motif": DISRUPTED_MOTIF,
        "control_replicates": CONTROL_REPLICATES,
        "exclusion_flank": EXCLUSION_FLANK,
        "control_rule": (
            "same count of nonoverlapping 6-bp GC=4 windows; "
            "swap positions 3 and 4"
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if output.is_file() and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("input_fingerprint") == fingerprint
            and previous.get("output_sha256") == sha256(output)
        ):
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "reused": True,
                        "rows": previous["variant_rows"],
                        "retained_genes": previous["retained_genes"],
                        "output": str(output.relative_to(root)),
                    },
                    indent=2,
                )
            )
            return

    records: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    eligible_before = 0
    source_rows: dict[str, int] = {}
    for genus, relative_path in DATASETS.items():
        source = root / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        data = normalize_source(pd.read_parquet(source), genus)
        source_rows[genus] = len(data)
        for source_index, row in data.iterrows():
            sequence = str(row["promoter_2048"])
            motif_starts = exact_nonoverlapping_starts(sequence, MOTIF)
            if not motif_starts:
                continue
            eligible_before += 1
            candidates = control_candidates(sequence, motif_starts)
            controls: list[list[int]] = []
            for replicate in range(CONTROL_REPLICATES):
                selected = select_control_starts(
                    candidates,
                    len(motif_starts),
                    stable_seed(
                        "publication_v3_gbox_control_v1",
                        genus,
                        row["gene_id"],
                        replicate,
                    ),
                )
                if selected is None:
                    excluded.append(
                        {
                            "genus": genus,
                            "gene_id": row["gene_id"],
                            "gbox_count": len(motif_starts),
                            "candidate_windows": len(candidates),
                            "failed_replicate": replicate,
                            "reason": "insufficient_nonoverlapping_controls",
                        }
                    )
                    controls = []
                    break
                controls.append(selected)
            if not controls:
                continue

            common = {
                "genus": genus,
                "gene_id": str(row["gene_id"]),
                "chromosome": str(row["chromosome"]),
                "label": str(row["label"]),
                "label_binary": int(row["label_binary"]),
                "split": str(row["split"]),
                "source_index": int(source_index),
                "gbox_count": len(motif_starts),
            }
            disrupted = mutate_windows(
                sequence, motif_starts, expected=MOTIF
            )
            if any(
                disrupted[start : start + WINDOW] == MOTIF
                for start in motif_starts
            ):
                raise RuntimeError("G-box disruption left an exact motif")
            records.append(
                {
                    **common,
                    "variant_id": (
                        f"{genus}|{row['gene_id']}|motif_disruption"
                    ),
                    "variant_type": "motif_disruption",
                    "control_replicate": -1,
                    "mutated_starts": ",".join(map(str, motif_starts)),
                    "promoter_2048": disrupted,
                }
            )
            for replicate, starts in enumerate(controls):
                records.append(
                    {
                        **common,
                        "variant_id": (
                            f"{genus}|{row['gene_id']}|"
                            f"matched_control_{replicate:02d}"
                        ),
                        "variant_type": "matched_control",
                        "control_replicate": replicate,
                        "mutated_starts": ",".join(map(str, starts)),
                        "promoter_2048": mutate_windows(sequence, starts),
                    }
                )

    variants = pd.DataFrame(records).sort_values(
        ["variant_id"], kind="stable"
    ).reset_index(drop=True)
    if variants.empty:
        raise RuntimeError("no G-box perturbation variants were generated")
    if variants["variant_id"].duplicated().any():
        raise RuntimeError("duplicate perturbation variant identifiers")
    per_gene = variants.groupby(["genus", "gene_id"]).size()
    if not per_gene.eq(CONTROL_REPLICATES + 1).all():
        raise RuntimeError("retained genes lack the exact variant count")
    motif_counts = (
        variants["variant_type"].eq("motif_disruption")
        .groupby([variants["genus"], variants["gene_id"]])
        .sum()
    )
    if not motif_counts.eq(1).all():
        raise RuntimeError("retained genes lack one motif disruption")

    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    variants.to_parquet(
        partial, compression="zstd", index=False
    )
    os.replace(partial, output)
    excluded_frame = pd.DataFrame(
        excluded,
        columns=[
            "genus",
            "gene_id",
            "gbox_count",
            "candidate_windows",
            "failed_replicate",
            "reason",
        ],
    )
    excluded_frame.to_csv(excluded_output, sep="\t", index=False)

    gene_table = variants.drop_duplicates(["genus", "gene_id"])
    by_genus = {}
    for genus in DATASETS:
        subset = gene_table[gene_table["genus"].eq(genus)]
        by_genus[genus] = {
            "retained_genes": len(subset),
            "positives": int(subset["label_binary"].sum()),
            "negatives": int(subset["label_binary"].eq(0).sum()),
            "variant_rows": int(variants["genus"].eq(genus).sum()),
        }
    manifest = {
        "status": "complete",
        **fingerprint_payload,
        "input_fingerprint": fingerprint,
        "source_rows": source_rows,
        "eligible_genes_before_control_gate": eligible_before,
        "retained_genes": int(len(gene_table)),
        "excluded_genes": int(len(excluded_frame)),
        "variant_rows": int(len(variants)),
        "by_genus": by_genus,
        "row_hash": row_hash(variants),
        "output": str(output.relative_to(root)),
        "output_sha256": sha256(output),
        "excluded_output": str(excluded_output.relative_to(root)),
        "excluded_sha256": sha256(excluded_output),
        "malus_accessed": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
