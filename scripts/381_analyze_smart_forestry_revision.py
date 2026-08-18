#!/usr/bin/env python3
"""Build the scientific correction tables required for the Smart Forestry revision.

This script does not overwrite the frozen publication-v5 results.  It harmonizes
three inconsistent JASPAR family labels, re-runs the primary/masked/strict motif
tests from frozen hits and assignments, and prepares functional-context tables
for corpus overlap, no-skill AUPRC and species-disjoint controls.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metrics" / "smart_forestry_revision"
MOTIF_OUT = OUT / "motif_harmonized"
FUNCTIONAL_OUT = OUT / "functional_context"

PRIMARY = ROOT / "results" / "metrics" / "publication_v5_motif"
MASKED = ROOT / "results" / "metrics" / "publication_v5_motif_gbox_masked_sensitivity"
STRICT = ROOT / "results" / "metrics" / "publication_v5_motif_gbox_count_matched_sensitivity"
FUNCTIONAL = ROOT / "results" / "metrics" / "publication_v4_functional_conclusion"
CORPUS = ROOT / "results" / "metrics" / "publication_v4_corpus_phylogeny"
TECHNICAL = ROOT / "results" / "metrics" / "publication_v4_technical_qc"
BOOTSTRAP = ROOT / "results" / "metrics" / "publication_v3_rebuild_functional_bootstrap"

REPLICATES = 10_000
BIN_EDGES = np.linspace(-2048, 0, 21)
SYNONYMS = {"D": "Group D", "S": "Group S", "group A": "Group A"}
FIXED_FAMILY_TOKENS = ("AP2", "ERF", "WRKY", "bZIP", "bHLH")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def metadata_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    metadata = pd.read_csv(PRIMARY / "jaspar2026_plant_metadata.tsv", sep="\t", dtype=str)
    metadata["tf_family_raw"] = metadata["family"].fillna("").replace("", np.nan)
    metadata["tf_family_raw"] = metadata["tf_family_raw"].fillna(
        "Unclassified: " + metadata["class"].fillna("unknown")
    )
    metadata["tf_family"] = metadata["tf_family_raw"].replace(SYNONYMS)
    mapping = (
        metadata.groupby(["tf_family_raw", "tf_family"], as_index=False)
        .agg(
            motif_profiles=("matrix_id", "nunique"),
            tf_classes=("class", lambda x: ";".join(sorted(set(x.dropna())))),
            species=("species", lambda x: ";".join(sorted(set(x.dropna())))),
        )
        .sort_values(["tf_family", "tf_family_raw"])
    )
    motif_to_family = dict(zip(metadata["matrix_id"], metadata["tf_family"]))
    return metadata, mapping, motif_to_family


def load_hits(path: Path, motif_to_family: dict[str, str]) -> pd.DataFrame:
    hits = pd.read_csv(path, sep="\t", low_memory=False)
    hits["tf_family"] = hits["motif_id"].map(motif_to_family)
    if hits["tf_family"].isna().any():
        missing = sorted(hits.loc[hits["tf_family"].isna(), "motif_id"].unique())[:10]
        raise RuntimeError(f"Unmapped motifs in {path}: {missing}")
    return hits


def assignment_indices(
    assignment_path: Path,
    promoters: pd.DataFrame,
    gene_index: dict[str, int],
) -> dict[str, np.ndarray]:
    assignments = pd.read_csv(assignment_path, sep="\t")
    result: dict[str, np.ndarray] = {}
    for genus in sorted(promoters["genus"].unique()):
        foreground = sorted(
            promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"]
        )
        subset = assignments.loc[assignments["genus"].eq(genus)].copy()
        pivot = subset.pivot(
            index="replicate", columns="foreground_safe_id", values="control_safe_id"
        )
        pivot = pivot.reindex(columns=foreground).sort_index()
        if pivot.shape != (REPLICATES, len(foreground)) or pivot.isna().any().any():
            raise RuntimeError(
                f"Malformed assignments for {genus}: {pivot.shape}, expected {(REPLICATES, len(foreground))}"
            )
        result[genus] = np.vectorize(gene_index.__getitem__)(pivot.to_numpy())
    return result


def family_annotations(metadata: pd.DataFrame) -> dict[str, dict[str, object]]:
    annotations = {}
    for family, frame in metadata.groupby("tf_family"):
        annotations[family] = {
            "tf_classes": ";".join(sorted(set(frame["class"].dropna()))),
            "raw_family_labels": ";".join(sorted(set(frame["tf_family_raw"]))),
            "representative_profiles": ";".join(sorted(set(frame["name"].dropna()))[:8]),
            "motif_profiles": int(frame["matrix_id"].nunique()),
        }
    return annotations


def evaluate(
    hits: pd.DataFrame,
    promoters: pd.DataFrame,
    metadata: pd.DataFrame,
    assignment_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, int], np.ndarray, list[str]]:
    families = sorted(metadata["tf_family"].unique())
    genes = sorted(promoters["safe_id"].unique())
    gene_index = {gene: i for i, gene in enumerate(genes)}
    family_index = {family: i for i, family in enumerate(families)}
    presence = np.zeros((len(genes), len(families)), dtype=np.uint8)
    for safe_id, family in hits[["safe_id", "tf_family"]].drop_duplicates().itertuples(index=False):
        presence[gene_index[safe_id], family_index[family]] = 1

    controls = assignment_indices(assignment_path, promoters, gene_index)
    genera = sorted(promoters["genus"].unique())
    candidate_fraction: dict[str, np.ndarray] = {}
    null_fraction: dict[str, np.ndarray] = {}
    for genus in genera:
        foreground = sorted(
            promoters.loc[promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"]
        )
        candidate_fraction[genus] = presence[[gene_index[x] for x in foreground]].mean(axis=0)
        null_fraction[genus] = presence[controls[genus]].mean(axis=1)

    observed = np.mean([candidate_fraction[g] for g in genera], axis=0)
    null_combined = np.mean([null_fraction[g] for g in genera], axis=0)
    p_values = (1 + (null_combined >= observed[None, :]).sum(axis=0)) / (REPLICATES + 1)
    q_values = bh(p_values.astype(float))
    annotations = family_annotations(metadata)
    rows = []
    for index, family in enumerate(families):
        row = {"tf_family": family, **annotations[family]}
        positive_both = True
        for genus in genera:
            candidate = float(candidate_fraction[genus][index])
            control = float(np.median(null_fraction[genus][:, index]))
            row[f"{genus}_candidate_fraction"] = candidate
            row[f"{genus}_control_median"] = control
            row[f"{genus}_difference"] = candidate - control
            positive_both &= candidate > control
        row.update(
            {
                "combined_candidate_fraction": float(observed[index]),
                "combined_control_median": float(np.median(null_combined[:, index])),
                "combined_difference": float(
                    observed[index] - np.median(null_combined[:, index])
                ),
                "empirical_p": float(p_values[index]),
                "bh_q": float(q_values[index]),
                "positive_both_genera": bool(positive_both),
                "convergently_enriched": bool(positive_both and q_values[index] < 0.05),
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["bh_q", "empirical_p", "tf_family"])
    return table, controls, gene_index, presence, families


def position_density(
    hits: pd.DataFrame,
    promoters: pd.DataFrame,
    controls: dict[str, np.ndarray],
    gene_index: dict[str, int],
    selected_families: set[str],
) -> pd.DataFrame:
    genes = sorted(promoters["safe_id"].unique())
    bin_width_kb = float(BIN_EDGES[1] - BIN_EDGES[0]) / 1000.0
    rows = []
    for family in sorted(selected_families):
        subset = hits.loc[hits["tf_family"].eq(family)].copy()
        subset["distance_to_tss"] = (
            subset["start"].astype(float) + subset["stop"].astype(float)
        ) / 2 - 2048.5
        subset["bin"] = np.clip(
            np.digitize(subset["distance_to_tss"], BIN_EDGES) - 1,
            0,
            len(BIN_EDGES) - 2,
        )
        counts = subset.groupby(["safe_id", "bin"]).size()
        matrix = np.zeros((len(genes), len(BIN_EDGES) - 1), dtype=np.float32)
        for (safe_id, bin_index), count in counts.items():
            matrix[gene_index[safe_id], int(bin_index)] = float(count) / bin_width_kb
        for genus in sorted(promoters["genus"].unique()):
            foreground = sorted(
                promoters.loc[
                    promoters["genus"].eq(genus) & promoters["foreground"], "safe_id"
                ]
            )
            candidate = matrix[[gene_index[x] for x in foreground]].mean(axis=0)
            null = matrix[controls[genus]].mean(axis=1)
            for b in range(len(BIN_EDGES) - 1):
                rows.append(
                    {
                        "tf_family": family,
                        "genus": genus,
                        "bin_start": float(BIN_EDGES[b]),
                        "bin_end": float(BIN_EDGES[b + 1]),
                        "candidate_density": float(candidate[b]),
                        "control_median": float(np.median(null[:, b])),
                        "control_q025": float(np.quantile(null[:, b], 0.025)),
                        "control_q975": float(np.quantile(null[:, b], 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def old_hit_sets() -> dict[str, list[str]]:
    files = {
        "primary": PRIMARY / "tf_family_enrichment.tsv",
        "masked": MASKED / "tf_family_enrichment_gbox_overlap_removed.tsv",
        "strict": STRICT / "tf_family_enrichment_gbox_overlap_removed_count_matched.tsv",
    }
    result = {}
    for stage, path in files.items():
        table = pd.read_csv(path, sep="\t")
        result[stage] = sorted(
            table.loc[bool_series(table["convergently_enriched"]), "tf_family"].tolist()
        )
    return result


def build_motif() -> dict:
    MOTIF_OUT.mkdir(parents=True, exist_ok=True)
    metadata, mapping, motif_to_family = metadata_tables()
    mapping.to_csv(MOTIF_OUT / "tf_family_harmonization.tsv", sep="\t", index=False)
    promoters = pd.read_csv(PRIMARY / "promoter_map.tsv.gz", sep="\t")
    promoters["foreground"] = bool_series(promoters["foreground"])
    primary_hits = load_hits(PRIMARY / "fimo_hits.tsv.gz", motif_to_family)
    masked_hits = load_hits(
        MASKED / "fimo_hits_excluding_exact_cacgtg_overlap.tsv.gz", motif_to_family
    )

    primary, primary_controls, gene_index, _, families = evaluate(
        primary_hits,
        promoters,
        metadata,
        PRIMARY / "matched_background_replicates.tsv.gz",
    )
    masked, _, _, _, _ = evaluate(
        masked_hits,
        promoters,
        metadata,
        PRIMARY / "matched_background_replicates.tsv.gz",
    )
    strict, _, _, _, _ = evaluate(
        masked_hits,
        promoters,
        metadata,
        STRICT / "matched_background_replicates_gbox_count_matched.tsv.gz",
    )

    primary.to_csv(MOTIF_OUT / "primary_59families.tsv", sep="\t", index=False)
    masked.to_csv(MOTIF_OUT / "gbox_overlap_masked_59families.tsv", sep="\t", index=False)
    strict.to_csv(
        MOTIF_OUT / "gbox_overlap_masked_exact_count_matched_59families.tsv",
        sep="\t",
        index=False,
    )
    # Compatibility names allow the frozen plotting code to be reused with
    # only its input/output constants redirected to the revision namespace.
    primary.to_csv(MOTIF_OUT / "tf_family_enrichment.tsv", sep="\t", index=False)
    compatibility_drop = [
        "tf_classes",
        "raw_family_labels",
        "representative_profiles",
        "motif_profiles",
    ]
    masked.drop(columns=compatibility_drop).to_csv(
        MOTIF_OUT / "tf_family_enrichment_gbox_overlap_removed.tsv",
        sep="\t",
        index=False,
    )
    strict.drop(columns=compatibility_drop).to_csv(
        MOTIF_OUT / "tf_family_enrichment_gbox_overlap_removed_count_matched.tsv",
        sep="\t",
        index=False,
    )
    primary_hits.to_csv(
        MOTIF_OUT / "fimo_hits.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    masked_hits.to_csv(
        MOTIF_OUT / "fimo_hits_excluding_exact_cacgtg_overlap.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    annotations = family_annotations(metadata)
    new_sets = {
        "primary": sorted(
            primary.loc[bool_series(primary["convergently_enriched"]), "tf_family"]
        ),
        "masked": sorted(
            masked.loc[bool_series(masked["convergently_enriched"]), "tf_family"]
        ),
        "strict": sorted(
            strict.loc[bool_series(strict["convergently_enriched"]), "tf_family"]
        ),
    }
    selected = set().union(*map(set, new_sets.values()))
    for family, annotation in annotations.items():
        text = f"{family} {annotation['tf_classes']}".lower()
        if any(token.lower() in text for token in FIXED_FAMILY_TOKENS):
            selected.add(family)
    position = position_density(
        primary_hits, promoters, primary_controls, gene_index, selected
    )
    position.to_csv(MOTIF_OUT / "primary_position_density_59families.tsv", sep="\t", index=False)
    position.to_csv(MOTIF_OUT / "tf_family_position_density.tsv", sep="\t", index=False)

    strict_group_d = strict.loc[strict["tf_family"].eq("Group D")].iloc[0]
    checks = {
        "metadata_profiles_927": len(metadata) == 927,
        "raw_family_universe_62": metadata["tf_family_raw"].nunique() == 62,
        "harmonized_family_universe_59": len(families) == 59,
        "mapping_only_three_aliases": set(
            mapping.loc[mapping["tf_family_raw"] != mapping["tf_family"], "tf_family_raw"]
        )
        == set(SYNONYMS),
        "primary_assignments_10000": all(x.shape[0] == REPLICATES for x in primary_controls.values()),
        "strict_only_group_d": new_sets["strict"] == ["Group D"],
        "strict_group_d_positive_both": bool(strict_group_d["positive_both_genera"]),
        "strict_group_d_q_expected": abs(float(strict_group_d["bh_q"]) - 0.0058994100589941)
        < 1e-12,
        "malus_not_used": not any("malus" in str(path).lower() for path in [PRIMARY, MASKED, STRICT]),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "smart_forestry_revision_posthoc_label_harmonization",
        "decision_authority": False,
        "source_results_unchanged": True,
        "synonyms": SYNONYMS,
        "old_universe": 62,
        "harmonized_universe": 59,
        "old_convergent_sets": old_hit_sets(),
        "harmonized_convergent_sets": new_sets,
        "strict_group_d": {
            "prunus_difference": float(strict_group_d["prunus_difference"]),
            "pyrus_difference": float(strict_group_d["pyrus_difference"]),
            "empirical_p": float(strict_group_d["empirical_p"]),
            "bh_q": float(strict_group_d["bh_q"]),
            "motif_profiles": int(strict_group_d["motif_profiles"]),
            "raw_labels_merged": strict_group_d["raw_family_labels"],
        },
        "checks": checks,
        "output_sha256": {
            path.name: sha256(path)
            for path in sorted(MOTIF_OUT.glob("*.tsv"))
        },
    }
    (MOTIF_OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if summary["status"] != "pass":
        raise RuntimeError(f"Harmonized motif checks failed: {checks}")
    return summary


def build_functional() -> dict:
    FUNCTIONAL_OUT.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(FUNCTIONAL / "primary_arm_performance.tsv", sep="\t")
    boot = pd.read_parquet(BOOTSTRAP / "paired_bootstrap_replicates.parquet")

    # No-skill AUPRC equals the positive prevalence for each held-out genus.
    metric_paths = {
        "prunus": ROOT
        / "results/metrics/plantcad_dapt_publication_v3_functional_probes/base/seed_23/metrics.tsv",
        "pyrus": ROOT
        / "results/metrics/plantcad_dapt_publication_v3_functional_probes/base/seed_23/metrics.tsv",
    }
    metrics = pd.read_csv(metric_paths["prunus"], sep="\t")
    prevalence = (
        metrics.loc[metrics["population"].eq("all")]
        .groupby("heldout_genus")["prevalence"]
        .first()
        .to_dict()
    )

    context_rows = []
    disjoint_rows = []
    for row in primary.itertuples(index=False):
        no_skill = float(prevalence[row.heldout_genus])
        arm_values = {
            arm: float(getattr(row, arm))
            for arm in ["base", "tree", "herb", "random_plant", "phylogc_match"]
        }
        for arm, value in arm_values.items():
            context_rows.append(
                {
                    "heldout_genus": row.heldout_genus,
                    "readout": row.readout,
                    "arm": arm,
                    "auprc": value,
                    "no_skill_auprc": no_skill,
                    "absolute_gain_over_no_skill": value - no_skill,
                    "relative_gain_over_no_skill": (value - no_skill) / no_skill,
                }
            )
        best_disjoint = max(arm_values["herb"], arm_values["phylogc_match"])
        point = arm_values["tree"] - best_disjoint
        subset = boot.loc[
            boot["heldout_genus"].eq(row.heldout_genus)
            & boot["readout"].eq(row.readout)
            & boot["contrast"].isin(["tree_minus_herb", "tree_minus_phylogc_match"])
        ]
        pivot = subset.pivot(index="replicate", columns="contrast", values="delta_auprc")
        delta = pivot.min(axis=1)  # Tree - max(Herb, PhyloGCMatch)
        disjoint_rows.append(
            {
                "heldout_genus": row.heldout_genus,
                "readout": row.readout,
                "tree_auprc": arm_values["tree"],
                "best_species_disjoint_control_auprc": best_disjoint,
                "point_delta_auprc": point,
                "bootstrap_median": float(delta.median()),
                "ci95_low": float(delta.quantile(0.025)),
                "ci95_high": float(delta.quantile(0.975)),
                "ci90_low": float(delta.quantile(0.05)),
                "ci90_high": float(delta.quantile(0.95)),
                "probability_gt_0": float((delta > 0).mean()),
                "probability_ge_0_02": float((delta >= 0.02).mean()),
                "replicates": len(delta),
            }
        )
    context = pd.DataFrame(context_rows)
    disjoint = pd.DataFrame(disjoint_rows)
    context.to_csv(FUNCTIONAL_OUT / "primary_auprc_with_no_skill.tsv", sep="\t", index=False)
    disjoint.to_csv(
        FUNCTIONAL_OUT / "tree_minus_best_species_disjoint_control.tsv",
        sep="\t",
        index=False,
    )

    corpus = pd.read_csv(CORPUS / "corpus_species_source.tsv", sep="\t")
    species_sets = {
        arm: set(corpus.loc[corpus["corpus"].eq(arm), "slug"])
        for arm in ["tree", "herb", "random_plant", "phylogc_match"]
    }
    random_rows = corpus.loc[corpus["corpus"].eq("random_plant")].copy()
    random_rows["source_panel"] = np.where(
        random_rows["slug"].isin(species_sets["tree"]), "tree", "herb"
    )
    overlap = (
        random_rows.groupby("source_panel", as_index=False)
        .agg(species=("slug", "nunique"), windows=("written_windows", "sum"))
        .sort_values("source_panel")
    )
    overlap["fraction_of_random_plant_windows"] = overlap["windows"] / overlap["windows"].sum()
    overlap.to_csv(FUNCTIONAL_OUT / "random_plant_overlap.tsv", sep="\t", index=False)

    technical = pd.read_csv(TECHNICAL / "technical_tree_contrasts_all24.tsv", sep="\t")
    task = (
        technical.groupby("task", as_index=False)
        .agg(
            mean_tree_minus_strongest_control=("woody_control_gain", "mean"),
            minimum=("woody_control_gain", "min"),
            maximum=("woody_control_gain", "max"),
            cells=("woody_control_gain", "size"),
        )
        .sort_values("task")
    )
    task["mean_positive"] = task["mean_tree_minus_strongest_control"] > 0
    task.to_csv(FUNCTIONAL_OUT / "technical_continuation_conditions.tsv", sep="\t", index=False)

    expected_points = [-0.0030942472176362, 0.0012357970016437, -0.0068850849090102, -0.0108936138388411]
    checks = {
        "randomplant_equals_tree_union_herb": species_sets["random_plant"]
        == species_sets["tree"] | species_sets["herb"],
        "tree_herb_disjoint": species_sets["tree"].isdisjoint(species_sets["herb"]),
        "phylogc_disjoint_all_other": species_sets["phylogc_match"].isdisjoint(
            species_sets["tree"] | species_sets["herb"] | species_sets["random_plant"]
        ),
        "randomplant_684211_tree_windows": int(
            overlap.loc[overlap["source_panel"].eq("tree"), "windows"].iloc[0]
        )
        == 684_211,
        "randomplant_315789_herb_windows": int(
            overlap.loc[overlap["source_panel"].eq("herb"), "windows"].iloc[0]
        )
        == 315_789,
        "prevalence_prunus": abs(prevalence["prunus"] - 0.1954253611556982) < 1e-12,
        "prevalence_pyrus": abs(prevalence["pyrus"] - 0.7672028596961573) < 1e-12,
        "disjoint_points_expected": np.allclose(
            disjoint["point_delta_auprc"].to_numpy(), expected_points, atol=1e-12
        ),
        "disjoint_threshold_wins_zero_of_four": int(
            (disjoint["point_delta_auprc"] >= 0.02).sum()
        )
        == 0,
        "technical_positive_tasks_one_of_four": int(task["mean_positive"].sum()) == 1,
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "smart_forestry_revision_descriptive_disclosure",
        "decision_authority": False,
        "seeds_41_59_authorized": False,
        "malus_accessed": False,
        "no_skill_auprc": prevalence,
        "random_plant_overlap": overlap.to_dict("records"),
        "species_disjoint_control_point_deltas": disjoint[
            ["heldout_genus", "readout", "point_delta_auprc"]
        ].to_dict("records"),
        "species_disjoint_cells_at_least_0_02": int(
            (disjoint["point_delta_auprc"] >= 0.02).sum()
        ),
        "technical_tasks_with_positive_mean": int(task["mean_positive"].sum()),
        "checks": checks,
        "output_sha256": {
            path.name: sha256(path)
            for path in sorted(FUNCTIONAL_OUT.glob("*.tsv"))
        },
    }
    (FUNCTIONAL_OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if summary["status"] != "pass":
        raise RuntimeError(f"Functional context checks failed: {checks}")
    return summary


def main() -> int:
    motif = build_motif()
    functional = build_functional()
    combined = {
        "status": "pass" if motif["status"] == functional["status"] == "pass" else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Smart Forestry scientific revisions A1-A4",
        "motif": motif,
        "functional_context": functional,
        "boundaries": {
            "frozen_publication_v5_overwritten": False,
            "seeds_41_59_run": False,
            "malus_accessed": False,
            "new_model_training": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "revision_analysis_audit.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0 if combined["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
