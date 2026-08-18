#!/usr/bin/env python3
"""Prepare frozen study-specific Prunus count designs for publication-v3."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


GENE_PATTERN = re.compile(r"^Prupe\.\d+G\d+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_gse130426(metadata: pd.DataFrame) -> pd.DataFrame:
    contrasts = (
        (
            "CRISTOBALINA",
            "December 1, 2015",
            "December 15, 2015",
        ),
        ("GARNET", "January 26, 2016", "February 23, 2016"),
        ("REGINA", "February 23, 2016", "March 8, 2016"),
    )
    rows = []
    for cultivar, baseline, endpoint in contrasts:
        for condition, date in (("baseline", baseline), ("endpoint", endpoint)):
            selected = metadata.loc[
                metadata["cultivar"].eq(cultivar)
                & metadata["time_point"].eq(date)
            ].copy()
            for sample_id in selected["sample_id"]:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subgroup": cultivar,
                        "condition": condition,
                        "source_stage": date,
                    }
                )
    return pd.DataFrame(rows)


def select_gse138792(metadata: pd.DataFrame) -> pd.DataFrame:
    eligible = (
        ("Prunus armeniaca", "A1956", "petal visible"),
        ("Prunus armeniaca", "A660", "petal visible"),
        ("Prunus persica", "A209", "Pre-bloom"),
        ("Prunus persica", "A318", "Pre-bloom"),
        ("Prunus persica", "A323", "Pre-bloom"),
        ("Prunus persica", "A340", "Pre-bloom"),
    )
    rows = []
    for organism, cultivar, endpoint in eligible:
        subgroup = f"{organism.replace(' ', '_')}__{cultivar}"
        for condition, stage in (("baseline", "0 CH"), ("endpoint", endpoint)):
            selected = metadata.loc[
                metadata["organism_ch1"].eq(organism)
                & metadata["cultivar_number"].eq(cultivar)
                & metadata["stage"].eq(stage)
            ].copy()
            for sample_id in selected["sample_id"]:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subgroup": subgroup,
                        "condition": condition,
                        "source_stage": stage,
                    }
                )
    return pd.DataFrame(rows)


def select_gse298924(metadata: pd.DataFrame) -> pd.DataFrame:
    contrasts = (
        ("flower2022", "CUb0", "CUe930"),
        ("veg2021", "CU0", "CU770"),
        ("veg2022", "PreCU0", "CU930"),
    )
    rows = []
    for cohort, baseline, endpoint in contrasts:
        for condition, stage in (("baseline", baseline), ("endpoint", endpoint)):
            selected = metadata.loc[
                metadata["cohort"].eq(cohort)
                & metadata["stage"].eq(stage)
            ].copy()
            for sample_id in selected["sample_id"]:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subgroup": cohort,
                        "condition": condition,
                        "source_stage": stage,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    expected = Path(os.environ["TREEFM_ROOT"]).resolve() if os.environ.get("TREEFM_ROOT") else None
    if expected is not None and root != expected:
        raise SystemExit(f"refusing to run outside {expected}: {root}")

    source_root = root / "data/processed/functional"
    output_root = (
        root / "data/interim/functional_v3/Prunus_publication_v3"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    selectors = {
        "GSE130426": select_gse130426,
        "GSE138792": select_gse138792,
        "GSE298924": select_gse298924,
    }
    counts_by_study: dict[str, pd.DataFrame] = {}
    metadata_by_study: dict[str, pd.DataFrame] = {}
    source_hashes = {}
    for accession, selector in selectors.items():
        counts_path = source_root / accession / "counts.parquet"
        metadata_path = source_root / accession / "sample_metadata.tsv"
        counts = pd.read_parquet(counts_path)
        counts.index = counts.index.astype(str)
        counts = counts.loc[
            [bool(GENE_PATTERN.fullmatch(value)) for value in counts.index]
        ].copy()
        if counts.index.duplicated().any():
            raise RuntimeError(f"canonical gene IDs duplicated: {accession}")
        metadata = pd.read_csv(metadata_path, sep="\t", dtype=str).fillna("")
        design = selector(metadata)
        if design.empty or design["sample_id"].duplicated().any():
            raise RuntimeError(f"invalid frozen sample design: {accession}")
        cell_counts = design.groupby(["subgroup", "condition"]).size()
        if cell_counts.empty or int(cell_counts.min()) < 2:
            raise RuntimeError(
                f"a frozen subgroup cell has fewer than two samples: "
                f"{accession}: {cell_counts.to_dict()}"
            )
        if set(design["condition"]) != {"baseline", "endpoint"}:
            raise RuntimeError(f"both conditions are required: {accession}")
        missing = set(design["sample_id"]) - set(counts.columns)
        if missing:
            raise RuntimeError(
                f"selected samples absent from counts: {accession}: "
                f"{sorted(missing)}"
            )
        design.insert(0, "accession", accession)
        design["condition"] = pd.Categorical(
            design["condition"],
            categories=["baseline", "endpoint"],
            ordered=True,
        )
        design = design.sort_values(
            ["subgroup", "condition", "sample_id"], kind="stable"
        ).reset_index(drop=True)
        counts_by_study[accession] = counts.loc[:, design["sample_id"]]
        metadata_by_study[accession] = design
        source_hashes[accession] = {
            "counts_parquet_sha256": sha256(counts_path),
            "sample_metadata_sha256": sha256(metadata_path),
        }

    common_genes = sorted(
        set.intersection(
            *(set(counts.index) for counts in counts_by_study.values())
        )
    )
    if len(common_genes) < 25_000:
        raise RuntimeError(f"too few common Prunus genes: {len(common_genes)}")
    study_records = {}
    for accession in selectors:
        counts = counts_by_study[accession].loc[common_genes]
        design = metadata_by_study[accession]
        count_path = output_root / f"{accession}_selected_counts.tsv.gz"
        design_path = output_root / f"{accession}_selected_design.tsv"
        counts.to_csv(
            count_path,
            sep="\t",
            index=True,
            index_label="gene_id",
            compression="gzip",
        )
        design.to_csv(design_path, sep="\t", index=False)
        study_records[accession] = {
            "genes": len(counts),
            "samples": len(design),
            "subgroups": int(design["subgroup"].nunique()),
            "cell_counts": {
                f"{subgroup}|{condition}": int(value)
                for (subgroup, condition), value in design.groupby(
                    ["subgroup", "condition"], observed=True
                ).size().items()
            },
            "counts": str(count_path.relative_to(root)),
            "counts_sha256": sha256(count_path),
            "design": str(design_path.relative_to(root)),
            "design_sha256": sha256(design_path),
            **source_hashes[accession],
        }

    summary = {
        "status": "pass",
        "scope": "sample/design freeze; no expression effects inspected",
        "contract": "docs/publication_v3_prunus_functional_label_contract.md",
        "contract_sha256": sha256(
            root / "docs/publication_v3_prunus_functional_label_contract.md"
        ),
        "common_genes": len(common_genes),
        "studies": study_records,
    }
    summary_path = (
        root / "metadata/publication_v3_prunus_label_design.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
