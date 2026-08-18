#!/usr/bin/env python3
"""Audit the rebuilt biological, functional, and technical evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-technical-pending", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    methods_path = root / "reports/PUBLICATION_V3_REBUILD_INTEGRATED_METHODS_EN.md"
    methods_text = methods_path.read_text(encoding="utf-8") if methods_path.is_file() else ""
    required_methods_headings = {
        "## Study design, evidence roles and prospective stopping rule",
        "## Backbone and equal-budget DAPT",
        "## Model-independent cross-genus GO convergence",
        "## Reciprocal functional transfer",
        "## Technical 26-species transfer panel",
        "## Computational execution and reproducibility",
        "## Interpretation constraints",
    }
    check(
        "executed_methods_complete",
        methods_path.is_file()
        and len(methods_text) > 10000
        and required_methods_headings.issubset(set(methods_text.splitlines())),
        {
            "bytes": methods_path.stat().st_size if methods_path.is_file() else 0,
            "headings": sorted(
                required_methods_headings.intersection(methods_text.splitlines())
            ),
        },
        {"bytes": ">10000", "headings": sorted(required_methods_headings)},
    )
    check(
        "executed_methods_stopping_boundary",
        "Seeds 41 and 59 could run only if" in methods_text
        and "Malus remained sealed throughout" in methods_text
        and "completed three-seed confirmatory" not in methods_text
        and "Malus was unlocked" not in methods_text,
        "stopped_seed23_and_malus_sealed",
        "stopped_seed23_and_malus_sealed",
    )
    check(
        "executed_methods_primary_novelfamily_counts",
        "14,791 balanced positive-negative pairs" in methods_text
        and "636–2,589 pairs per species-task scope" in methods_text
        and "no exact task-training sequence match" in methods_text
        and "maximum identity 0.984" in methods_text,
        "primary_pair_counts_and_leakage_reported",
        "primary_pair_counts_and_leakage_reported",
    )
    references_path = root / "reports/PUBLICATION_V3_REBUILD_REFERENCE_BACKBONE_EN.md"
    references_text = (
        references_path.read_text(encoding="utf-8")
        if references_path.is_file()
        else ""
    )
    numbered_references = [
        line
        for line in references_text.splitlines()
        if line[:1].isdigit() and ". " in line[:4]
    ]
    required_reference_dois = {
        "10.1073/pnas.2421738122",
        "10.1038/s42003-024-06465-2",
        "10.1101/2025.08.27.672609",
        "10.1093/molbev/msag139",
        "10.1111/1365-2745.13888",
        "10.1111/nph.18393",
    }
    check(
        "verified_reference_backbone",
        references_path.is_file()
        and len(numbered_references) == 20
        and all(doi in references_text for doi in required_reference_dois),
        {
            "references": len(numbered_references),
            "required_dois_present": sorted(
                doi for doi in required_reference_dois if doi in references_text
            ),
        },
        {
            "references": 20,
            "required_dois_present": sorted(required_reference_dois),
        },
    )
    skeleton_path = (
        root / "reports/PUBLICATION_V3_REBUILD_INTEGRATED_MANUSCRIPT_SKELETON_EN.md"
    )
    skeleton_text = (
        skeleton_path.read_text(encoding="utf-8") if skeleton_path.is_file() else ""
    )
    cited_reference_numbers: set[int] = set()
    for match in re.finditer(r"\[(\d+)(?:[–-](\d+))?\]", skeleton_text + "\n" + methods_text):
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        cited_reference_numbers.update(range(start, stop + 1))
    expected_reference_numbers = set(range(1, 21))
    check(
        "all_numbered_references_cited",
        cited_reference_numbers == expected_reference_numbers,
        sorted(cited_reference_numbers),
        sorted(expected_reference_numbers),
    )
    legends_path = root / "reports/PUBLICATION_V3_REBUILD_FIGURE_LEGENDS_EN.md"
    legends_text = (
        legends_path.read_text(encoding="utf-8") if legends_path.is_file() else ""
    )
    required_legend_headings = {
        "## Figure 1. Frozen biological benchmark, equal-budget representation interventions and prospective pilot decision",
        "## Figure 2. Corrected cross-genus biological-process convergence",
        "## Figure 3. Complete model-independent cross-genus candidate catalog",
        "## Figure 4. Matched domain-adaptation controls eliminate the apparent Tree-specific functional advantage",
        "## Figure 5. Technical NovelFamily transfer under matched DAPT controls",
        "## Supplementary Figure S1. Tier-A mechanism evidence matrix",
        "## Supplementary Figure S2. Frozen Prunus RNA direction is associated with independent promoter H3K4me3 change",
    }
    check(
        "figure_legends_complete_and_corrected",
        legends_path.is_file()
        and required_legend_headings.issubset(set(legends_text.splitlines()))
        and "observed count of 52 exceeded the null maximum of 12" in legends_text
        and "111 shared positive-only candidate orthogroups" in legends_text
        and "A–D (14/38/49/10)" in legends_text
        and "observed count was 0/4" in legends_text
        and "14,791 balanced positive-negative pairs" in legends_text
        and "no exact task-training sequence match" in legends_text,
        {
            "headings": len(required_legend_headings.intersection(legends_text.splitlines())),
            "bytes": legends_path.stat().st_size if legends_path.is_file() else 0,
        },
        {"headings": len(required_legend_headings), "corrected_values": True},
    )
    pair_audit_path = (
        root / "results/metrics/publication_v3_primary_novelfamily_pair_audit.json"
    )
    pair_table_path = (
        root / "metadata/publication_v3_primary_novelfamily_pair_summary.tsv"
    )
    pair_audit = read_json(pair_audit_path)
    pair_table = pd.read_csv(pair_table_path, sep="\t")
    pair_observed = pair_audit.get("observed", {})
    check(
        "primary_novelfamily_pair_audit_status",
        pair_audit.get("status") == "pass"
        and pair_audit.get("malus_accessed") is False
        and pair_audit.get("integrity_errors") == [],
        {
            "status": pair_audit.get("status"),
            "errors": pair_audit.get("integrity_errors"),
            "malus_accessed": pair_audit.get("malus_accessed"),
        },
        {"status": "pass", "errors": [], "malus_accessed": False},
    )
    check(
        "primary_novelfamily_pair_counts",
        len(pair_table) == 12
        and pair_observed.get("pairs") == 14791
        and pair_observed.get("minimum_pairs_per_scope") == 636
        and pair_observed.get("maximum_pairs_per_scope") == 2589
        and pair_observed.get("pair_size_failures") == 0
        and pair_observed.get("pair_label_failures") == 0,
        pair_observed,
        {
            "scopes": 12,
            "pairs": 14791,
            "minimum_pairs_per_scope": 636,
            "maximum_pairs_per_scope": 2589,
            "pair_size_failures": 0,
            "pair_label_failures": 0,
        },
    )
    check(
        "primary_novelfamily_leakage_counts",
        pair_observed.get("exact_pairs") == 0
        and pair_observed.get("near_0_90_pairs") == 1
        and pair_observed.get("near_0_95_pairs") == 1
        and abs(
            float(pair_observed.get("maximum_task_train_identity", 0))
            - 0.984
        )
        < 1e-6,
        {
            key: pair_observed.get(key)
            for key in (
                "exact_pairs",
                "near_0_90_pairs",
                "near_0_95_pairs",
                "maximum_task_train_identity",
            )
        },
        {
            "exact_pairs": 0,
            "near_0_90_pairs": 1,
            "near_0_95_pairs": 1,
            "maximum_task_train_identity": 0.984,
        },
    )
    pair_input_hashes = pair_audit.get("input_sha256", {})
    pair_hashes_match = bool(pair_input_hashes) and all(
        (root / relative).is_file()
        and sha256(root / relative) == expected_hash
        for relative, expected_hash in pair_input_hashes.items()
    )
    check(
        "primary_novelfamily_pair_input_hashes",
        pair_hashes_match and len(pair_input_hashes) == 3,
        {"files": len(pair_input_hashes), "hashes_match": pair_hashes_match},
        {"files": 3, "hashes_match": True},
    )
    design_metadata = read_json(
        root / "results/figures/publication_v3_rebuild_study_design.metadata.json"
    )
    design_values = design_metadata.get("scientific_values", {})
    check(
        "study_design_figure_values",
        design_metadata.get("status") == "complete"
        and design_metadata.get("malus_accessed") is False
        and design_values
        == {
            "prunus_labels": 2492,
            "pyrus_labels": 2238,
            "replicated_go_terms": 52,
            "candidate_orthogroups": 111,
            "tier_a_families": 14,
            "h3k4me3_fold_enrichment": 3.81,
            "functional_cells_ge_0_02": 0,
            "functional_primary_cells": 4,
        },
        design_values,
        "exact frozen study-design values",
    )
    for suffix in ("png", "pdf", "svg"):
        design_path = root / f"results/figures/publication_v3_rebuild_study_design.{suffix}"
        design_hash = design_metadata.get("outputs_sha256", {}).get(
            str(design_path.relative_to(root)).replace("\\", "/")
        )
        check(
            f"study_design_figure_{suffix}",
            design_path.is_file()
            and design_path.stat().st_size > 0
            and design_hash == sha256(design_path),
            sha256(design_path) if design_path.is_file() else None,
            design_hash,
        )
    training_qc = read_json(
        root / "results/metrics/publication_v3_rebuild_pilot_training_qc/summary.json"
    )
    training_steps = {
        arm: details.get("global_step")
        for arm, details in training_qc.get("arms", {}).items()
    }
    check(
        "seed23_training_qc",
        training_qc.get("status") == "pass"
        and training_qc.get("expected_steps") == 15000
        and training_steps
        == {
            "tree": 15000,
            "herb": 15000,
            "random_plant": 15000,
            "phylogc_match": 15000,
        },
        {"status": training_qc.get("status"), "steps": training_steps},
        {
            "status": "pass",
            "steps": {
                "tree": 15000,
                "herb": 15000,
                "random_plant": 15000,
                "phylogc_match": 15000,
            },
        },
    )
    for arm in ("tree", "herb", "random_plant", "phylogc_match"):
        run_spec = read_json(
            root
            / f"results/models/plantcad_dapt_publication_v3/{arm}/seed_23/run_spec.json"
        )
        observed_spec = {
            "max_steps": run_spec.get("max_steps"),
            "sequence_length": run_spec.get("sequence_length"),
            "mlm_probability": run_spec.get("mlm_probability"),
            "learning_rate": run_spec.get("learning_rate"),
            "effective_batch_size": run_spec.get("effective_batch_size"),
            "runtime_world_size": run_spec.get("runtime_world_size"),
            "trained_sequences": run_spec.get("trained_sequences"),
            "trained_bases": run_spec.get("trained_bases"),
        }
        expected_spec = {
            "max_steps": 15000,
            "sequence_length": 512,
            "mlm_probability": 0.15,
            "learning_rate": 1e-5,
            "effective_batch_size": 64,
            "runtime_world_size": 2,
            "trained_sequences": 960000,
            "trained_bases": 491520000,
        }
        check(
            f"seed23_run_spec:{arm}",
            observed_spec == expected_spec,
            observed_spec,
            expected_spec,
        )
    premetric = read_json(
        root
        / "results/metrics/publication_v3_rebuild_pre_metric_runtime_amendment_audit.json"
    )
    check(
        "premetric_amendment_timing",
        premetric.get("status")
        == "captured_before_any_rebuilt_downstream_metric"
        and premetric.get("downstream_metric_file_count") == 0
        and premetric.get("malus_accessed") is False,
        {
            "status": premetric.get("status"),
            "downstream_metrics": premetric.get("downstream_metric_file_count"),
            "malus_accessed": premetric.get("malus_accessed"),
        },
        {
            "status": "captured_before_any_rebuilt_downstream_metric",
            "downstream_metrics": 0,
            "malus_accessed": False,
        },
    )

    go = read_json(
        root / "results/biological_cases/publication_v3_crossgenus_go/summary.json"
    )
    go_audit = read_json(
        root
        / "results/biological_cases/publication_v3_crossgenus_go/independent_audit.json"
    )
    check("go_status", go.get("status") == "complete", go.get("status"), "complete")
    check("go_audit", go_audit.get("status") == "pass", go_audit.get("status"), "pass")
    check("go_primary_terms", go.get("primary_robust_replicated_terms") == 52, go.get("primary_robust_replicated_terms"), 52)
    go_primary = go.get("layers", {}).get("curated_no_iea", {})
    check("go_permutations", go_primary.get("permutations") == 10000, go_primary.get("permutations"), 10000)
    check("go_null_maximum", go_primary.get("null_maximum_hits") == 12, go_primary.get("null_maximum_hits"), 12)

    candidates = read_json(
        root
        / "results/biological_cases/publication_v3_crossgenus_candidates/summary.json"
    )
    candidate_audit = read_json(
        root
        / "results/biological_cases/publication_v3_crossgenus_candidates/independent_audit.json"
    )
    check("candidate_status", candidates.get("status") == "pass", candidates.get("status"), "pass")
    check("candidate_audit", candidate_audit.get("status") == "pass", candidate_audit.get("status"), "pass")
    check("candidate_count", candidates.get("candidate_orthogroups") == 111, candidates.get("candidate_orthogroups"), 111)
    check("candidate_tiers", candidates.get("candidate_counts_by_tier") == {"A": 14, "B": 38, "C": 49, "D": 10}, candidates.get("candidate_counts_by_tier"), {"A": 14, "B": 38, "C": 49, "D": 10})

    domains = read_json(
        root
        / "results/biological_cases/publication_v3_tier_a_annotation/domain_summary.json"
    )
    domain_audit = read_json(
        root
        / "results/biological_cases/publication_v3_tier_a_annotation/independent_audit.json"
    )
    mechanism_audit = read_json(
        root
        / "results/biological_cases/publication_v3_tier_a_mechanism_evidence/independent_audit.json"
    )
    check("domain_status", domains.get("status") == "pass", domains.get("status"), "pass")
    check("domain_audit", domain_audit.get("status") == "pass", domain_audit.get("status"), "pass")
    check("tier_a_count", domains.get("candidate_orthogroups") == 14, domains.get("candidate_orthogroups"), 14)
    support = mechanism_audit.get("reconstructed_pfam_support_counts", {})
    check("tier_a_crossgenus_pfam", support.get("cross_genus_anchor_supported") == 12, support, {"cross_genus_anchor_supported": 12})

    chromatin = read_json(
        root
        / "results/metrics/publication_v3_prunus_chromatin_complete_evidence_manifest.json"
    )
    check("chromatin_status", chromatin.get("status") == "pass", chromatin.get("status"), "pass")
    chromatin_result = chromatin.get("gse190586", {})
    check("chromatin_concordant", chromatin_result.get("direction_consistent_calls") == 54, chromatin_result.get("direction_consistent_calls"), 54)
    check("chromatin_positive", chromatin_result.get("positive_genes") == 487, chromatin_result.get("positive_genes"), 487)
    check("chromatin_enrichment", abs(float(chromatin_result.get("enrichment_ratio", 0)) - 3.8143405075898316) < 1e-12, chromatin_result.get("enrichment_ratio"), 3.8143405075898316)

    tier_a_chromatin = read_json(
        root
        / "results/biological_cases/publication_v3_tier_a_chromatin_overlap/summary.json"
    )
    check(
        "tier_a_chromatin_posthoc",
        tier_a_chromatin.get("status") == "complete_posthoc_descriptive"
        and tier_a_chromatin.get("posthoc") is True
        and tier_a_chromatin.get("selection_authority") is False,
        {
            "status": tier_a_chromatin.get("status"),
            "posthoc": tier_a_chromatin.get("posthoc"),
            "selection_authority": tier_a_chromatin.get("selection_authority"),
        },
        {
            "status": "complete_posthoc_descriptive",
            "posthoc": True,
            "selection_authority": False,
        },
    )
    check(
        "tier_a_chromatin_counts",
        tier_a_chromatin.get("tier_a_orthogroups") == 14
        and tier_a_chromatin.get("tier_a_prunus_genes") == 34
        and tier_a_chromatin.get(
            "tier_a_direction_concordant_h3k4me3_genes"
        )
        == 4,
        {
            "orthogroups": tier_a_chromatin.get("tier_a_orthogroups"),
            "genes": tier_a_chromatin.get("tier_a_prunus_genes"),
            "concordant": tier_a_chromatin.get(
                "tier_a_direction_concordant_h3k4me3_genes"
            ),
        },
        {"orthogroups": 14, "genes": 34, "concordant": 4},
    )
    check(
        "tier_a_chromatin_no_set_enrichment",
        tier_a_chromatin.get("enrichment_claim_supported") is False,
        tier_a_chromatin.get("enrichment_claim_supported"),
        False,
    )
    supported_tier_a_genes = {
        row.get("gene_id")
        for row in tier_a_chromatin.get("supported_exact_genes", [])
    }
    check(
        "tier_a_chromatin_exact_supported_genes",
        supported_tier_a_genes
        == {
            "Prupe.4G176200",
            "Prupe.7G220600",
            "Prupe.5G087600",
            "Prupe.5G155800",
        },
        sorted(supported_tier_a_genes),
        sorted(
            {
                "Prupe.4G176200",
                "Prupe.7G220600",
                "Prupe.5G087600",
                "Prupe.5G155800",
            }
        ),
    )
    tier_a_hashes = {
        **tier_a_chromatin.get("input_sha256", {}),
        **tier_a_chromatin.get("output_sha256", {}),
    }
    for relative, expected in tier_a_hashes.items():
        path = root / relative
        check(
            f"sha256:{relative}",
            path.is_file() and sha256(path) == expected,
            sha256(path) if path.is_file() else None,
            expected,
        )

    functional = read_json(
        root / "results/metrics/publication_v3_rebuild_early_functional/summary.json"
    )
    functional_bootstrap = read_json(
        root
        / "results/metrics/publication_v3_rebuild_functional_bootstrap/summary.json"
    )
    functional_benchmark = read_json(
        root
        / "results/metrics/publication_v3_rebuild_functional_benchmark/summary.json"
    )
    agreement = read_json(
        root
        / "results/metrics/publication_v3_rebuild_prediction_agreement/summary.json"
    )
    secondary_bootstrap = read_json(
        root
        / "results/metrics/publication_v3_functional_secondary_bootstrap/summary.json"
    )
    check("functional_gate_failed", functional.get("functional_gate_pass") is False, functional.get("functional_gate_pass"), False)
    check("functional_primary_vs_control", functional.get("observed", {}).get("tree_vs_matched_control_positive_cells") == 0, functional.get("observed", {}).get("tree_vs_matched_control_positive_cells"), 0)
    check("functional_all_vs_control", functional_benchmark.get("tree_auprc_positive_vs_best_matched_control_cells") == 0, functional_benchmark.get("tree_auprc_positive_vs_best_matched_control_cells"), 0)
    check("functional_bootstrap_replicates", functional_bootstrap.get("replicates") == 2000, functional_bootstrap.get("replicates"), 2000)
    check("functional_bootstrap_negative_points", functional_bootstrap.get("all_woody_control_point_estimates_negative") is True, functional_bootstrap.get("all_woody_control_point_estimates_negative"), True)
    check("prediction_agreement_posthoc", agreement.get("posthoc_exploratory") is True, agreement.get("posthoc_exploratory"), True)
    check("prediction_agreement_randomplant", agreement.get("closest_spearman_cell_counts") == {"random_plant": 4} and agreement.get("closest_mae_cell_counts") == {"random_plant": 4}, {"spearman": agreement.get("closest_spearman_cell_counts"), "mae": agreement.get("closest_mae_cell_counts")}, {"random_plant": 4})
    secondary_endpoints = secondary_bootstrap.get("endpoint_summary", {})
    secondary_topk = secondary_endpoints.get("top_k_enrichment", {})
    secondary_ece = secondary_endpoints.get("ece_15bin", {})
    check(
        "functional_secondary_bootstrap_contract",
        secondary_bootstrap.get("status") == "complete"
        and secondary_bootstrap.get("analysis_tier") == "posthoc_seed23_descriptive"
        and secondary_bootstrap.get("decision_authority") is False
        and secondary_bootstrap.get("primary_auprc_gate_unchanged") is True
        and secondary_bootstrap.get("replicates") == 2000
        and secondary_bootstrap.get("point_alignment_failures") == 0,
        {
            "status": secondary_bootstrap.get("status"),
            "tier": secondary_bootstrap.get("analysis_tier"),
            "decision_authority": secondary_bootstrap.get("decision_authority"),
            "replicates": secondary_bootstrap.get("replicates"),
            "alignment_failures": secondary_bootstrap.get("point_alignment_failures"),
        },
        {
            "status": "complete",
            "tier": "posthoc_seed23_descriptive",
            "decision_authority": False,
            "replicates": 2000,
            "alignment_failures": 0,
        },
    )
    check(
        "functional_secondary_topk_no_robust_benefit",
        secondary_topk.get("cells") == 16
        and secondary_topk.get("ci_overlapping_zero") == 16
        and secondary_topk.get("bh_significant_positive") == 0
        and secondary_topk.get("bh_significant_negative") == 0,
        secondary_topk,
        {
            "cells": 16,
            "ci_overlapping_zero": 16,
            "bh_significant_positive": 0,
            "bh_significant_negative": 0,
        },
    )
    check(
        "functional_secondary_ece_localized_cost",
        secondary_ece.get("cells") == 16
        and secondary_ece.get("ci_entirely_positive") == 0
        and secondary_ece.get("ci_entirely_negative") == 4
        and secondary_ece.get("bh_significant_positive") == 0
        and secondary_ece.get("bh_significant_negative") == 4,
        secondary_ece,
        {
            "cells": 16,
            "ci_entirely_positive": 0,
            "ci_entirely_negative": 4,
            "bh_significant_positive": 0,
            "bh_significant_negative": 4,
        },
    )
    secondary_hash_failures = []
    for relative, expected in secondary_bootstrap.get("output_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            secondary_hash_failures.append(relative)
    check(
        "functional_secondary_output_fingerprints",
        set(secondary_bootstrap.get("output_sha256", {}))
        == {
            "results/metrics/publication_v3_functional_secondary_bootstrap/scope_summary.tsv",
            "results/metrics/publication_v3_functional_secondary_bootstrap/paired_bootstrap_secondary.parquet",
            "reports/PUBLICATION_V3_FUNCTIONAL_SECONDARY_BOOTSTRAP_EN.md",
        }
        and not secondary_hash_failures,
        secondary_hash_failures,
        [],
    )

    # Recheck every frozen functional input fingerprint against the current file.
    for summary in (functional, functional_bootstrap, functional_benchmark, agreement, secondary_bootstrap):
        hashes = summary.get("input_metrics_sha256", {}) | summary.get(
            "input_prediction_sha256", {}
        ) | summary.get("input_sha256", {})
        for relative, expected in hashes.items():
            path = root / relative
            check(
                f"sha256:{relative}",
                path.is_file() and sha256(path) == expected,
                sha256(path) if path.is_file() else None,
                expected,
            )

    decision_path = (
        root / "results/metrics/publication_v3_rebuild_pilot_summary/decision.json"
    )
    comparison_root = (
        root / "results/metrics/plantcad_dapt_publication_v3_seed23_comparison"
    )
    technical_complete = decision_path.is_file() and (
        comparison_root / "gate.json"
    ).is_file()
    supplementary_path = (
        root / "reports/PUBLICATION_V3_REBUILD_SUPPLEMENTARY_DATA_MAP_EN.md"
    )
    supplementary_text = (
        supplementary_path.read_text(encoding="utf-8")
        if supplementary_path.is_file()
        else ""
    )
    supplementary_headings = [
        line
        for line in supplementary_text.splitlines()
        if line.startswith("## Supplementary Data ")
    ]
    technical_pending_marker = "**Status:** pending technical output"
    technical_complete_marker = "**Status:** complete; all five frozen arms"
    check(
        "supplementary_data_inventory",
        supplementary_path.is_file()
        and len(supplementary_headings) == 10
        and "results/biological_cases/publication_v3_crossgenus_go/"
        in supplementary_text
        and "results/metrics/publication_v3_rebuild_functional_bootstrap/"
        in supplementary_text
        and "results/metrics/publication_v3_functional_secondary_bootstrap/"
        in supplementary_text
        and "Malus-access flags must remain false throughout" in supplementary_text
        and (
            technical_complete_marker in supplementary_text
            if technical_complete
            else technical_pending_marker in supplementary_text
        )
        and not (
            technical_complete and technical_pending_marker in supplementary_text
        ),
        {
            "headings": len(supplementary_headings),
            "technical_marker": "complete"
            if technical_complete_marker in supplementary_text
            else "pending"
            if technical_pending_marker in supplementary_text
            else "missing",
        },
        {
            "headings": 10,
            "technical_marker": "complete" if technical_complete else "pending",
        },
    )
    assembled_draft_path = (
        root / "reports/PUBLICATION_V3_REBUILD_FULL_MANUSCRIPT_DRAFT_EN.md"
    )
    assembled_draft_text = (
        assembled_draft_path.read_text(encoding="utf-8")
        if assembled_draft_path.is_file()
        else ""
    )
    required_claim_boundaries = {
        "does not establish across-seed equivalence",
        "Sample-level bootstrap intervals do not estimate adaptation-seed variation",
        "Malus remains sealed",
        "do not prove binding or causal regulation",
        "explicitly post hoc prediction-agreement analysis",
        "all 16 Tree-minus-strongest-control intervals overlapped zero",
        "four Prunus-linear calibration disadvantages",
    }
    prohibited_positive_claims = {
        "Tree DAPT significantly outperformed",
        "Tree-specific superiority was established",
        "Malus validation showed",
        "a causal chromatin mechanism was established",
        "This is a completed three-seed confirmatory study",
    }
    check(
        "manuscript_claim_boundaries",
        assembled_draft_path.is_file()
        and required_claim_boundaries.issubset(
            {
                phrase
                for phrase in required_claim_boundaries
                if phrase in assembled_draft_text
            }
        )
        and not any(
            phrase in assembled_draft_text for phrase in prohibited_positive_claims
        ),
        {
            "required_present": sorted(
                phrase
                for phrase in required_claim_boundaries
                if phrase in assembled_draft_text
            ),
            "prohibited_present": sorted(
                phrase
                for phrase in prohibited_positive_claims
                if phrase in assembled_draft_text
            ),
        },
        {
            "required_present": sorted(required_claim_boundaries),
            "prohibited_present": [],
        },
    )
    if technical_complete:
        decision = read_json(decision_path)
        gate = read_json(comparison_root / "gate.json")
        scopes = pd.read_csv(comparison_root / "bootstrap_scope_effects.tsv", sep="\t")
        check("pilot_decision", decision.get("decision") == "stop_and_reassess", decision.get("decision"), "stop_and_reassess")
        check("technical_comparison_tier", gate.get("analysis_tier") == "pilot_direction_only", gate.get("analysis_tier"), "pilot_direction_only")
        check("technical_seed", gate.get("seeds") == [23], gate.get("seeds"), [23])
        check("technical_bootstrap_scopes", len(scopes) == 24, len(scopes), 24)
        check("generic_claim_disabled", gate.get("generic_woody_claim_screen") is False, gate.get("generic_woody_claim_screen"), False)
        check("technical_malus_sealed", decision.get("malus_accessed") is False, decision.get("malus_accessed"), False)
        technical_figure_summary = read_json(
            root
            / "results/metrics/publication_v3_rebuild_technical_figure/summary.json"
        )
        expected_positive = int(scopes["woody_control_gain"].gt(0).sum())
        expected_ci_positive = int(
            scopes["woody_control_gain_ci_low"].gt(0).sum()
        )
        expected_ci_negative = int(
            scopes["woody_control_gain_ci_high"].lt(0).sum()
        )
        check(
            "technical_figure_summary_status",
            technical_figure_summary.get("status") == "complete",
            technical_figure_summary.get("status"),
            "complete",
        )
        check(
            "technical_figure_summary_cells",
            technical_figure_summary.get("primary_cells") == 24,
            technical_figure_summary.get("primary_cells"),
            24,
        )
        check(
            "technical_figure_summary_counts",
            technical_figure_summary.get(
                "tree_minus_strongest_control_positive_cells"
            )
            == expected_positive
            and technical_figure_summary.get(
                "control_ci_entirely_above_zero"
            )
            == expected_ci_positive
            and technical_figure_summary.get(
                "control_ci_entirely_below_zero"
            )
            == expected_ci_negative,
            {
                "positive": technical_figure_summary.get(
                    "tree_minus_strongest_control_positive_cells"
                ),
                "ci_positive": technical_figure_summary.get(
                    "control_ci_entirely_above_zero"
                ),
                "ci_negative": technical_figure_summary.get(
                    "control_ci_entirely_below_zero"
                ),
            },
            {
                "positive": expected_positive,
                "ci_positive": expected_ci_positive,
                "ci_negative": expected_ci_negative,
            },
        )
        for suffix in ("png", "pdf", "svg"):
            figure_path = (
                root
                / f"results/figures/publication_v3_rebuild_technical_panel.{suffix}"
            )
            relative = str(figure_path.relative_to(root))
            expected_hash = technical_figure_summary.get(
                "figure_sha256", {}
            ).get(relative)
            check(
                f"technical_figure_{suffix}",
                figure_path.is_file()
                and figure_path.stat().st_size > 0
                and expected_hash == sha256(figure_path),
                sha256(figure_path) if figure_path.is_file() else None,
                expected_hash,
            )
        integrated_report = (
            root / "reports/PUBLICATION_V3_REBUILD_PILOT_RESULT_20260801_CN.md"
        )
        technical_block = (
            root
            / "reports/PUBLICATION_V3_REBUILD_TECHNICAL_MANUSCRIPT_BLOCK_EN.md"
        )
        manuscript = (
            root
            / "reports/PUBLICATION_V3_REBUILD_INTEGRATED_MANUSCRIPT_SKELETON_EN.md"
        )
        manuscript_text = (
            manuscript.read_text(encoding="utf-8") if manuscript.is_file() else ""
        )
        check(
            "integrated_seed23_report",
            integrated_report.is_file() and integrated_report.stat().st_size > 0,
            integrated_report.stat().st_size if integrated_report.is_file() else 0,
            ">0 bytes",
        )
        check(
            "technical_manuscript_block",
            technical_block.is_file() and technical_block.stat().st_size > 0,
            technical_block.stat().st_size if technical_block.is_file() else 0,
            ">0 bytes",
        )
        check(
            "integrated_manuscript_technical_fill",
            "[TECHNICAL-PANEL SENTENCE TO BE INSERTED AFTER AUDIT.]"
            not in manuscript_text
            and "Pending frozen outputs" not in manuscript_text
            and "<!-- TECHNICAL_SENTENCE_START -->" in manuscript_text
            and "<!-- TECHNICAL_SECTION_START -->" in manuscript_text,
            "filled" if "<!-- TECHNICAL_SECTION_START -->" in manuscript_text else "pending",
            "filled",
        )
        manuscript_assembly = read_json(
            root / "results/metrics/publication_v3_rebuild_manuscript_assembly.json"
        )
        full_manuscript = root / manuscript_assembly.get("output", "")
        check(
            "full_manuscript_assembled",
            manuscript_assembly.get("status") == "complete"
            and manuscript_assembly.get("technical_filled") is True
            and manuscript_assembly.get("word_count", 0) > 2500
            and manuscript_assembly.get("source_sha256", {}).get(
                str(legends_path.relative_to(root)).replace("\\", "/")
            )
            == sha256(legends_path)
            and manuscript_assembly.get("source_sha256", {}).get(
                str(supplementary_path.relative_to(root)).replace("\\", "/")
            )
            == sha256(supplementary_path)
            and full_manuscript.is_file()
            and sha256(full_manuscript)
            == manuscript_assembly.get("output_sha256"),
            {
                "status": manuscript_assembly.get("status"),
                "technical_filled": manuscript_assembly.get("technical_filled"),
                "word_count": manuscript_assembly.get("word_count"),
                "sha256": sha256(full_manuscript)
                if full_manuscript.is_file()
                else None,
            },
            {
                "status": "complete",
                "technical_filled": True,
                "word_count": ">2500",
                "sha256": manuscript_assembly.get("output_sha256"),
            },
        )
    else:
        check(
            "technical_complete",
            args.allow_technical_pending,
            "pending",
            "complete" if not args.allow_technical_pending else "pending_allowed",
        )

    malus_values = [
        go.get("malus_accessed"),
        go_audit.get("malus_accessed"),
        candidates.get("malus_accessed"),
        candidate_audit.get("malus_accessed"),
        domains.get("malus_accessed"),
        domain_audit.get("malus_accessed"),
        mechanism_audit.get("malus_accessed"),
        chromatin.get("malus_accessed"),
        functional.get("malus_accessed"),
        functional_bootstrap.get("malus_accessed"),
        functional_benchmark.get("malus_accessed"),
        agreement.get("malus_accessed"),
        secondary_bootstrap.get("malus_accessed"),
        tier_a_chromatin.get("malus_accessed"),
    ]
    check("malus_sealed_across_modules", all(value is False for value in malus_values), malus_values, [False] * len(malus_values))

    failures = [row for row in checks if not row["passed"]]
    payload = {
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "integrated_rebuilt_seed23_biology_functional_technical_evidence",
        "technical_complete": technical_complete,
        "allow_technical_pending": args.allow_technical_pending,
        "checks": len(checks),
        "passed_checks": len(checks) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "malus_accessed": False,
    }
    output = (
        root
        / "results/metrics/publication_v3_rebuild_integrated_evidence_audit.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
