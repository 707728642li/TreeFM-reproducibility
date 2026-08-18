#!/usr/bin/env python3
"""Discover same-release Ensembl assets for the frozen Phylo/GC-Match pool."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import pandas as pd


def release_prefix(genome: Path) -> str:
    suffix = ".dna.toplevel.fa.gz"
    if not genome.name.endswith(suffix):
        raise ValueError(f"unexpected toplevel genome name: {genome}")
    return genome.name[: -len(suffix)]


def choose_gff(gff_root: Path, prefix: str) -> Path | None:
    candidates = []
    for path in gff_root.glob(f"{prefix}*.gff3.gz"):
        lowered = path.name.lower()
        if any(
            marker in lowered
            for marker in (
                ".chr.",
                ".chromosome.",
                ".primary_assembly.",
                ".nonchromosomal.",
                ".scaffold.",
            )
        ):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (len(path.name), path.name))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--ensembl-root",
        type=Path,
        default=Path(os.environ.get("PUBLIC_GENOME_ROOT", "public_genomes")) / "Ensemble",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    candidate_path = root / "config/publication_v3_phylogc_candidate_pool.tsv"
    candidates = pd.read_csv(candidate_path, sep="\t", dtype=str).fillna("")
    publication_panel = pd.read_csv(
        root / "config/publication_v3_panel.tsv", sep="\t", dtype=str
    )
    frozen_panel = pd.read_csv(
        root / "config/mpu_panel_frozen_v2.tsv", sep="\t", dtype=str
    )
    prohibited = set(publication_panel["slug"]) | set(frozen_panel["slug"])
    overlap = sorted(set(candidates["slug"]) & prohibited)
    if overlap:
        raise RuntimeError(f"candidate pool overlaps frozen/evaluation species: {overlap}")
    if not candidates["plantcad_v1_scope"].eq("unseen").all():
        raise RuntimeError("all candidates must be outside the known PlantCAD-v1 scope")

    records: list[dict[str, object]] = []
    for row in candidates.to_dict(orient="records"):
        slug = row["slug"]
        fasta_root = args.ensembl_root / "fasta" / slug
        dna_root = fasta_root / "dna"
        genomes = sorted(
            path
            for path in dna_root.glob("*.dna.toplevel.fa.gz")
            if ".dna_sm." not in path.name and ".dna_rm." not in path.name
        )
        if not genomes:
            genomes = sorted(
                path
                for path in (fasta_root / "dna_index").glob("*.dna.toplevel.fa.gz")
                if ".dna_sm." not in path.name and ".dna_rm." not in path.name
            )
        release_records: list[dict[str, object]] = []
        for genome in genomes:
            prefix = release_prefix(genome)
            softmask_candidates = list(
                dna_root.glob(f"{prefix}.dna_sm.toplevel.fa.gz")
            )
            softmask = softmask_candidates[0] if softmask_candidates else None
            protein = fasta_root / "pep" / f"{prefix}.pep.all.fa.gz"
            gff = choose_gff(args.ensembl_root / "gff3" / slug, prefix)
            complete = bool(
                genome.is_file()
                and protein.is_file()
                and gff is not None
                and gff.is_file()
            )
            release_records.append(
                {
                    "prefix": prefix,
                    "genome": genome,
                    "softmask": softmask,
                    "annotation": gff,
                    "protein": protein,
                    "complete": complete,
                }
            )
        complete_releases = [
            record for record in release_records if bool(record["complete"])
        ]
        selected = (
            sorted(complete_releases, key=lambda record: str(record["prefix"]))[-1]
            if complete_releases
            else None
        )
        record: dict[str, object] = {
            **row,
            "source_status": "complete_same_release" if selected else "ineligible_missing_asset",
            "release_prefix": selected["prefix"] if selected else "",
            "genome_source": str(selected["genome"]) if selected else "",
            "softmask_source": (
                str(selected["softmask"])
                if selected and selected["softmask"] is not None
                else ""
            ),
            "annotation_source": str(selected["annotation"]) if selected else "",
            "protein_source": str(selected["protein"]) if selected else "",
            "available_release_count": len(release_records),
            "complete_release_count": len(complete_releases),
        }
        for key in ("genome", "softmask", "annotation", "protein"):
            source = Path(str(record[f"{key}_source"])) if record[f"{key}_source"] else None
            record[f"{key}_bytes"] = source.stat().st_size if source else 0
        records.append(record)

    output = pd.DataFrame(records).sort_values("slug")
    output_path = root / "metadata/publication_v3_phylogc_source_discovery.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False)
    summary = {
        "status": "pass"
        if output["source_status"].eq("complete_same_release").sum() >= 8
        else "fail",
        "candidate_pool": str(candidate_path.relative_to(root)),
        "candidate_count": len(output),
        "complete_same_release": int(
            output["source_status"].eq("complete_same_release").sum()
        ),
        "ineligible": output.loc[
            ~output["source_status"].eq("complete_same_release"), "slug"
        ].tolist(),
        "prohibited_overlap": overlap,
        "output": str(output_path.relative_to(root)),
    }
    summary_path = root / "metadata/publication_v3_phylogc_source_discovery.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
