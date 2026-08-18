#!/usr/bin/env python3
"""Deterministically select the frozen eight-species Phylo/GC-Match corpus."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = (
    "taxonomy_distance_hevea_brasiliensis",
    "taxonomy_distance_prunus_persica",
    "taxonomy_distance_pyrus_pyrifolia",
    "taxonomy_distance_malus_domestica",
    "gc_fraction",
    "repetitive_21mer_fraction",
    "gene_fraction",
    "cds_fraction",
    "intergenic_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--selected-species", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    features = pd.read_csv(
        root / "metadata/publication_v3_phylogc_genome_features.tsv",
        sep="\t",
    )
    busco = pd.read_csv(
        root / "metadata/publication_v3_phylogc_busco.tsv", sep="\t"
    )
    tree = features.loc[features["feature_role"].eq("tree_target")].copy()
    candidates = features.loc[features["feature_role"].eq("candidate")].merge(
        busco[["slug", "complete_pct", "busco_gate_pass"]],
        on="slug",
        how="left",
        validate="one_to_one",
    )
    if len(tree) != 13:
        raise RuntimeError(f"Tree target must contain 13 species, observed {len(tree)}")
    candidates["assembly_gate_pass"] = (
        candidates["n_fraction"].le(0.10)
        & candidates["gene_features"].ge(15000)
    )
    candidates["busco_gate_pass"] = candidates["complete_pct"].ge(90.0)
    candidates["eligible"] = (
        candidates["assembly_gate_pass"]
        & candidates["busco_gate_pass"]
    )
    candidates["exclusion_reason"] = ""
    candidates.loc[
        candidates["n_fraction"].gt(0.10), "exclusion_reason"
    ] += "N_fraction_above_0.10;"
    candidates.loc[
        candidates["gene_features"].lt(15000), "exclusion_reason"
    ] += "gene_count_below_15000;"
    candidates.loc[
        ~candidates["busco_gate_pass"],
        "exclusion_reason",
    ] += "BUSCO_below_90;"
    eligible = candidates.loc[candidates["eligible"]].sort_values("slug").copy()
    if len(eligible) < args.selected_species:
        raise RuntimeError(
            f"only {len(eligible)} candidates pass fixed eligibility gates"
        )

    combined = pd.concat([tree, eligible], ignore_index=True)
    scales = combined[list(FEATURES)].std(ddof=0)
    if scales.le(0).any() or scales.isna().any():
        raise RuntimeError(f"invalid feature scales: {scales.to_dict()}")
    target_mean = tree[list(FEATURES)].mean()

    scores: list[dict[str, object]] = []
    best_key: tuple[float, tuple[str, ...]] | None = None
    best_slugs: tuple[str, ...] | None = None
    eligible_by_slug = eligible.set_index("slug")
    for slugs in itertools.combinations(
        eligible["slug"].tolist(), args.selected_species
    ):
        subset = eligible_by_slug.loc[list(slugs)]
        order_counts = subset["order"].value_counts()
        feasible = subset["order"].nunique() >= 5 and int(order_counts.max()) <= 2
        if feasible:
            standardized_difference = (
                subset[list(FEATURES)].mean() - target_mean
            ) / scales
            objective = float(np.square(standardized_difference).sum())
        else:
            objective = float("inf")
        record = {
            "slugs": ",".join(slugs),
            "orders": ",".join(sorted(set(subset["order"]))),
            "order_count": int(subset["order"].nunique()),
            "maximum_species_per_order": int(order_counts.max()),
            "feasible": feasible,
            "objective": objective,
        }
        scores.append(record)
        if feasible:
            key = (objective, slugs)
            if best_key is None or key < best_key:
                best_key = key
                best_slugs = slugs
    if best_key is None or best_slugs is None:
        raise RuntimeError("no feasible eight-species matched subset")

    score_table = pd.DataFrame(scores).sort_values(
        ["feasible", "objective", "slugs"],
        ascending=[False, True, True],
    )
    score_table["selected"] = score_table["slugs"].eq(",".join(best_slugs))
    scores_path = root / "metadata/publication_v3_phylogc_subset_scores.tsv"
    score_table.to_csv(scores_path, sep="\t", index=False)

    selected = eligible_by_slug.loc[list(best_slugs)].reset_index()
    selected["dapt_role"] = "PhyloGCMatch"
    selected["include"] = 1
    selected["windows"] = 125000
    selected["window_length"] = 512
    selected["selection_objective"] = best_key[0]
    selected_path = root / "config/publication_v3_phylogc_match_selected.tsv"
    selected[
        [
            "scientific_name",
            "slug",
            "order",
            "family",
            "life_form",
            "dapt_role",
            "include",
            "windows",
            "window_length",
            "selection_objective",
        ]
    ].to_csv(selected_path, sep="\t", index=False)

    comparison_rows = []
    selected_mean = selected[list(FEATURES)].mean()
    for feature in FEATURES:
        comparison_rows.append(
            {
                "feature": feature,
                "tree_mean": float(target_mean[feature]),
                "selected_mean": float(selected_mean[feature]),
                "raw_difference": float(selected_mean[feature] - target_mean[feature]),
                "standardization_scale": float(scales[feature]),
                "standardized_difference": float(
                    (selected_mean[feature] - target_mean[feature]) / scales[feature]
                ),
            }
        )
    comparison_path = (
        root / "metadata/publication_v3_phylogc_selected_feature_match.tsv"
    )
    pd.DataFrame(comparison_rows).to_csv(
        comparison_path, sep="\t", index=False
    )
    eligibility_path = (
        root / "metadata/publication_v3_phylogc_candidate_eligibility.tsv"
    )
    candidates.sort_values("slug").to_csv(eligibility_path, sep="\t", index=False)
    summary = {
        "status": "pass",
        "tree_species": len(tree),
        "source_feature_candidates": len(candidates),
        "eligible_candidates": len(eligible),
        "selected_species": list(best_slugs),
        "selected_orders": sorted(set(selected["order"])),
        "objective": best_key[0],
        "features": list(FEATURES),
        "constraints": {
            "selected_species": args.selected_species,
            "minimum_orders": 5,
            "maximum_species_per_order": 2,
        },
        "artifacts": {
            "selected_panel": str(selected_path.relative_to(root)),
            "subset_scores": str(scores_path.relative_to(root)),
            "feature_match": str(comparison_path.relative_to(root)),
            "eligibility": str(eligibility_path.relative_to(root)),
        },
    }
    summary_path = root / "metadata/publication_v3_phylogc_selection.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
