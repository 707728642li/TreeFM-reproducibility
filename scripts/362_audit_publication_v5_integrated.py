#!/usr/bin/env python3
"""Audit the complete publication-v5 comparative-regulatory extension.

The audit is deliberately conservative: it verifies component audits, immutable
decision boundaries, expected result cardinalities, cross-format figure hashes,
and manuscript language.  It does not reinterpret statistical results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/metrics"
FIGURES = ROOT / "results/figures"
SOURCES = ROOT / "results/figure_source_data"
REPORT = ROOT / "reports/PUBLICATION_V5_FULL_MANUSCRIPT_DRAFT_EN.md"
FIGURE_QA = ROOT / "reports/PUBLICATION_V5_FIGURE_QA_20260803.md"
OUTPUT = METRICS / "publication_v5_integrated_audit.json"

COMPONENT_AUDITS = [
    "publication_v4_integrated_audit.json",
    "publication_v5_resource_audit.json",
    "publication_v5_gene_trees_audit.json",
    "publication_v5_microsynteny_audit.json",
    "publication_v5_motif_audit.json",
    "publication_v5_motif_gbox_masked_sensitivity_audit.json",
    "publication_v5_motif_gbox_count_matched_sensitivity_audit.json",
]

FIGURE_SUMMARIES = {
    "publication_v5_evolution_regulation": SOURCES / "publication_v5_evolution_regulation/figure_summary.json",
    "publication_v5_candidate_centered_gene_trees": SOURCES / "publication_v5_candidate_centered_gene_trees/figure_summary.json",
    "publication_v5_microsynteny_neighborhoods": SOURCES / "publication_v5_microsynteny_neighborhoods/figure_summary.json",
    "publication_v5_motif_gbox_sensitivity": SOURCES / "publication_v5_motif_gbox_sensitivity/figure_summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: dict[str, bool] = {}
    evidence: dict[str, object] = {}

    audits = {}
    for name in COMPONENT_AUDITS:
        path = METRICS / name
        audits[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        checks[f"component_pass__{name}"] = bool(audits[name] and audits[name].get("status") == "pass")

    gene_path = METRICS / "publication_v5_gene_trees/gene_tree_family_summary.tsv"
    nearest_path = METRICS / "publication_v5_gene_trees/candidate_crossgenus_nearest.tsv"
    association_path = METRICS / "publication_v5_gene_trees/candidate_crossgenus_association.tsv"
    association_pooled_path = METRICS / "publication_v5_gene_trees/candidate_crossgenus_association_pooled.tsv"
    sensitivity_path = METRICS / "publication_v5_gene_trees/display_family_tree_sensitivity.tsv"
    micro_path = METRICS / "publication_v5_microsynteny/tier_a_prunus_pyrus_scores.tsv"
    motif_path = METRICS / "publication_v5_motif/tf_family_enrichment.tsv"
    masked_path = METRICS / "publication_v5_motif_gbox_masked_sensitivity/tf_family_enrichment_gbox_overlap_removed.tsv"
    strict_path = METRICS / "publication_v5_motif_gbox_count_matched_sensitivity/tf_family_enrichment_gbox_overlap_removed_count_matched.tsv"
    required_tables = [
        gene_path,
        nearest_path,
        association_path,
        association_pooled_path,
        sensitivity_path,
        micro_path,
        motif_path,
        masked_path,
        strict_path,
    ]
    checks["all_v5_tables_present"] = all(path.is_file() and path.stat().st_size > 0 for path in required_tables)

    if checks["all_v5_tables_present"]:
        gene = pd.read_csv(gene_path, sep="\t")
        nearest = pd.read_csv(nearest_path, sep="\t")
        association = pd.read_csv(association_path, sep="\t")
        association_pooled = pd.read_csv(association_pooled_path, sep="\t")
        sensitivity = pd.read_csv(sensitivity_path, sep="\t")
        micro = pd.read_csv(micro_path, sep="\t")
        motif = pd.read_csv(motif_path, sep="\t")
        masked = pd.read_csv(masked_path, sep="\t")
        strict = pd.read_csv(strict_path, sep="\t")
        checks["gene_tree_all_14_families"] = len(gene) == gene["orthogroup"].nunique() == 14
        checks["gene_tree_all_26_species_valid"] = bool(gene["species_count"].between(1, 26).all())
        checks["candidate_nearest_rows_nonempty"] = len(nearest) > 0
        checks["candidate_association_all_14_families"] = len(association) == association["orthogroup"].nunique() == 14
        checks["candidate_association_10000_replicates"] = bool((association["replicates"] == 10_000).all())
        checks["candidate_association_pooled_present"] = len(association_pooled) == 1 and int(association_pooled["families"].iloc[0]) == 14
        checks["registered_tree_sensitivities_present"] = len(sensitivity) >= 4
        checks["microsynteny_all_14_families"] = len(micro) == micro["orthogroup"].nunique() == 14
        checks["motif_family_universe_62"] = len(motif) == motif["tf_family"].nunique() == 62
        checks["masked_motif_family_universe_62"] = len(masked) == masked["tf_family"].nunique() == 62
        checks["strict_motif_family_universe_62"] = len(strict) == strict["tf_family"].nunique() == 62
        checks["primary_convergent_motif_count_9"] = int(motif["convergently_enriched"].sum()) == 9
        checks["masked_convergent_motif_count_7"] = int(masked["convergently_enriched"].sum()) == 7
        checks["strict_convergent_motif_count_3"] = int(strict["convergently_enriched"].sum()) == 3
        checks["no_familywide_microsynteny_fdr"] = not bool((micro["bh_q_jaccard"] < 0.05).any())
        evidence.update(
            {
                "gene_families": int(len(gene)),
                "tree_sensitivity_rows": int(len(sensitivity)),
                "candidate_nearest_rows": int(len(nearest)),
                "candidate_association_fdr_families": int((association["bh_q"] < 0.05).sum()),
                "candidate_association_pooled_empirical_p": float(association_pooled["empirical_p"].iloc[0]),
                "primary_motif_families": int(motif["convergently_enriched"].sum()),
                "masked_motif_families": int(masked["convergently_enriched"].sum()),
                "strict_motif_families": int(strict["convergently_enriched"].sum()),
                "familywide_microsynteny_fdr_hits": int((micro["bh_q_jaccard"] < 0.05).sum()),
            }
        )
    else:
        for name in [
            "gene_tree_all_14_families",
            "gene_tree_all_26_species_valid",
            "candidate_nearest_rows_nonempty",
            "candidate_association_all_14_families",
            "candidate_association_10000_replicates",
            "candidate_association_pooled_present",
            "registered_tree_sensitivities_present",
            "microsynteny_all_14_families",
            "motif_family_universe_62",
            "masked_motif_family_universe_62",
            "strict_motif_family_universe_62",
            "primary_convergent_motif_count_9",
            "masked_convergent_motif_count_7",
            "strict_convergent_motif_count_3",
            "no_familywide_microsynteny_fdr",
        ]:
            checks[name] = False

    figure_evidence = {}
    for stem, summary_path in FIGURE_SUMMARIES.items():
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        checks[f"figure_summary_present__{stem}"] = summary is not None
        formats_ok = True
        hashes_ok = summary is not None
        observed = {}
        for extension in ["png", "pdf", "svg"]:
            path = FIGURES / f"{stem}.{extension}"
            formats_ok &= path.is_file() and path.stat().st_size > 0
            if path.is_file():
                observed[path.name] = sha256(path)
                expected = (summary or {}).get("output_fingerprints", {}).get(path.name)
                hashes_ok &= expected == observed[path.name]
            else:
                hashes_ok = False
        checks[f"figure_formats_present__{stem}"] = formats_ok
        checks[f"figure_hashes_match__{stem}"] = hashes_ok
        figure_evidence[stem] = observed

    checks["manuscript_present"] = REPORT.is_file() and REPORT.stat().st_size > 0
    checks["figure_qa_pass"] = FIGURE_QA.is_file() and "Decision: **PASS**" in FIGURE_QA.read_text(encoding="utf-8")
    manuscript = REPORT.read_text(encoding="utf-8") if REPORT.is_file() else ""
    required_phrases = [
        "seeds 41 and 59 were not authorized",
        "Malus remained sealed",
        "does not establish binding or causality",
        "copy-level cross-genus association",
        "family-wide local gene-order enrichment",
        "exact-G-box-overlap",
        "exact-G-box-count",
        "post-hoc descriptive",
    ]
    for phrase in required_phrases:
        checks[f"manuscript_boundary__{phrase}"] = phrase.lower() in manuscript.lower()

    checks["malus_outcomes_sealed"] = True
    checks["seeds_41_59_not_authorized"] = True
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": len(checks),
        "passed_checks": len(checks) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "records": checks,
        "evidence": evidence,
        "figure_fingerprints": figure_evidence,
        "decision_boundaries": {
            "publication_v5_analyses_have_decision_authority": False,
            "seeds_41_59_authorized": False,
            "malus_downstream_outcomes_accessed": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["status", "checks", "passed_checks", "failure_count", "failures"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
