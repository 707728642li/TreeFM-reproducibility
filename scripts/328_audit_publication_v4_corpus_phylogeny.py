#!/usr/bin/env python3
"""Independently audit publication-v4 corpus and species-phylogeny outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from Bio import Phylo


ARM_ORDER = ["tree", "herb", "random_plant", "phylogc_match"]
EXPECTED_SPECIES = {"tree": 13, "herb": 6, "random_plant": 19, "phylogc_match": 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    metric = root / "results/metrics/publication_v4_corpus_phylogeny"
    summary_path = metric / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    records: list[dict[str, object]] = []

    def check(name: str, condition: bool, observed: object, expected: object) -> None:
        records.append({"name": name, "passed": bool(condition), "observed": observed, "expected": expected})

    check("summary_status", summary.get("status") == "complete", summary.get("status"), "complete")
    check("analysis_tier", summary.get("analysis_tier") == "posthoc_seed23_descriptive", summary.get("analysis_tier"), "posthoc_seed23_descriptive")
    check("decision_authority", summary.get("decision_authority") is False, summary.get("decision_authority"), False)
    check("malus_outcomes_sealed", summary.get("malus_outcomes_accessed") is False, summary.get("malus_outcomes_accessed"), False)

    corpus = pd.read_csv(metric / "corpus_equal_budget_summary.tsv", sep="\t").set_index("corpus")
    check("four_corpus_rows", set(corpus.index) == set(ARM_ORDER) and len(corpus) == 4, sorted(corpus.index.tolist()), ARM_ORDER)
    check("equal_windows", bool((corpus.loc[ARM_ORDER, "windows"] == 1_000_000).all()), corpus.loc[ARM_ORDER, "windows"].to_dict(), {arm: 1_000_000 for arm in ARM_ORDER})
    check("equal_bases", bool((corpus.loc[ARM_ORDER, "bases"] == 512_000_000).all()), corpus.loc[ARM_ORDER, "bases"].to_dict(), {arm: 512_000_000 for arm in ARM_ORDER})
    observed_species = {arm: int(corpus.loc[arm, "species"]) for arm in ARM_ORDER}
    check("corpus_species", observed_species == EXPECTED_SPECIES, observed_species, EXPECTED_SPECIES)

    membership = pd.read_csv(metric / "technical_species_corpus_membership.tsv", sep="\t")
    check("membership_rows", len(membership) == 26 and membership["slug"].nunique() == 26, {"rows": len(membership), "species": membership["slug"].nunique()}, 26)
    binary_ok = all(set(membership[arm].dropna().astype(int).unique()).issubset({0, 1}) for arm in ARM_ORDER)
    check("membership_binary", binary_ok, {arm: sorted(membership[arm].dropna().unique().tolist()) for arm in ARM_ORDER}, [0, 1])

    annotations = pd.read_csv(metric / "technical_species_annotations.tsv", sep="\t")
    check("annotation_rows", len(annotations) == 26 and annotations["slug"].nunique() == 26, {"rows": len(annotations), "species": annotations["slug"].nunique()}, 26)
    check("annotations_complete", not annotations[["scientific_name", "order", "family", "life_form", "analysis_tier"]].isna().any().any(), int(annotations.isna().sum().sum()), 0)

    tree = Phylo.read(metric / "species_tree_named_rooted.nwk", "newick")
    tree_tips = {tip.name for tip in tree.get_terminals()}
    check("tree_tip_count", len(tree_tips) == 26, len(tree_tips), 26)
    check("tree_annotation_identity", tree_tips == set(annotations["slug"]), sorted(tree_tips), sorted(annotations["slug"].tolist()))
    root_children = [[tip.name for tip in child.get_terminals()] for child in tree.root.clades]
    magnolia_child_present = any(tips == ["magnolia_biondii"] for tips in root_children)
    check("magnolia_display_root", magnolia_child_present, root_children, "one root child containing only magnolia_biondii")

    features = pd.read_csv(metric / "phylogc_feature_match.tsv", sep="\t")
    check("matching_feature_rows", len(features) == 9 and features["feature"].nunique() == 9, {"rows": len(features), "features": features["feature"].nunique()}, 9)
    max_abs = float(features["standardized_difference"].abs().max())
    check("matching_max_reproduced", abs(max_abs - float(summary["maximum_absolute_standardized_difference"])) < 1e-12, max_abs, summary["maximum_absolute_standardized_difference"])

    output_hash_failures = []
    for rel, expected in summary.get("output_sha256", {}).items():
        path = root / rel
        if not path.exists() or sha256(path) != expected:
            output_hash_failures.append(rel)
    check("figure_fingerprints", not output_hash_failures and len(summary.get("output_sha256", {})) == 3, output_hash_failures, "three matching PNG/PDF/SVG hashes")

    input_hash_failures = []
    for key, expected in summary.get("input_sha256", {}).items():
        source_map = {
            "tree": root / "results/metrics/publication_v4_species_phylogeny/species_tree_ids.nwk",
            "species_ids": root / "metadata/publication_v4_species_ids.txt",
            "species_panel": root / "config/species_panel_seed.tsv",
            "pyrus_source": root / "config/publication_v3_pyrus_genome_sources.tsv",
            "embedding_manifest": root / "results/embeddings/plantcad_dapt_publication_v3/base/seed_0/manifest.tsv",
            "corpus_shards": root / "metadata/publication_v3_dapt_corpus_shards.tsv",
            "corpus_audit": root / "metadata/publication_v3_dapt_corpus_audit.tsv",
            "feature_match": root / "metadata/publication_v3_phylogc_selected_feature_match.tsv",
            "candidate_features": root / "metadata/publication_v3_phylogc_genome_features.tsv",
            "selected_phylogc": root / "config/publication_v3_phylogc_match_selected.tsv",
        }
        path = source_map[key]
        if not path.exists() or sha256(path) != expected:
            input_hash_failures.append(key)
    check("input_fingerprints", not input_hash_failures, input_hash_failures, [])

    failures = [record for record in records if not record["passed"]]
    audit = {
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "publication_v4_corpus_phylogeny",
        "checks": len(records),
        "passed_checks": len(records) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "records": records,
        "analysis_tier": "posthoc_seed23_descriptive",
        "decision_authority": False,
        "malus_outcomes_accessed": False,
    }
    output = root / "results/metrics/publication_v4_corpus_phylogeny_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
