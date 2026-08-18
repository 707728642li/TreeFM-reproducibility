#!/usr/bin/env python3
"""Build a deterministic evidence manifest for all publication-v3 Prunus chromatin work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        default="results/metrics/publication_v3_prunus_chromatin_complete_evidence_manifest.json",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    binary_manifest_path = (
        root / "results/metrics/publication_v3_prunus_chromatin_evidence_manifest.json"
    )
    binary_summary_path = (
        root
        / "results/biological_cases/prunus_publication_v3_chromatin_replication/summary.json"
    )
    raw_summary_path = (
        root / "results/biological_cases/prunus_publication_v3_gse141983_raw/summary.json"
    )
    raw_gate_path = (
        root
        / "results/biological_cases/prunus_publication_v3_gse141983_raw/technical_gate.json"
    )
    raw_verification_path = (
        root / "results/metrics/publication_v3_gse141983_raw_h3k4me3_verification.json"
    )
    raw_figure_manifest_path = (
        root / "results/metrics/publication_v3_gse141983_raw_qc_figure_manifest.json"
    )
    raw_supplement_manifest_path = (
        root / "results/metrics/publication_v3_gse141983_raw_qc_supplement_manifest.json"
    )
    transport_audit_path = (
        root / "results/metrics/publication_v3_gse141983_fastq_transport_retry_audit.json"
    )
    implementation_amendment_path = (
        root
        / "config/publication_v3_gse141983_raw_readcount_implementation_amendment.json"
    )
    parent_freeze_path = root / "config/publication_v3_gse141983_raw_sensitivity_freeze.json"

    binary_manifest = read_json(binary_manifest_path)
    binary_summary = read_json(binary_summary_path)
    raw_summary = read_json(raw_summary_path)
    raw_gate = read_json(raw_gate_path)
    raw_verification = read_json(raw_verification_path)
    raw_figure_manifest = read_json(raw_figure_manifest_path)
    raw_supplement_manifest = read_json(raw_supplement_manifest_path)
    transport_audit = read_json(transport_audit_path)
    implementation_amendment = read_json(implementation_amendment_path)

    violations: list[str] = []
    if binary_manifest.get("status") != "pass" or binary_manifest.get("violations"):
        violations.append("gse190586_binary_evidence_manifest_failed")
    if binary_summary.get("status") != "complete_supportive":
        violations.append("gse190586_binary_summary_not_supportive")
    if raw_summary.get("status") != "complete_omitted_technical_gate_failed":
        violations.append("gse141983_raw_status_unexpected")
    if raw_summary.get("label_data_read") is not False:
        violations.append("gse141983_label_seal_failed")
    if raw_gate.get("status") != "fail" or len(raw_gate.get("sample_qc", [])) != 12:
        violations.append("gse141983_technical_gate_record_failed")
    if sum(not bool(row.get("pass")) for row in raw_gate.get("sample_qc", [])) != 12:
        violations.append("gse141983_failed_sample_cardinality_unexpected")
    if raw_verification.get("status") != "pass" or raw_verification.get("violations"):
        violations.append("gse141983_independent_verification_failed")
    if raw_figure_manifest.get("status") != "pass" or raw_figure_manifest.get("violations"):
        violations.append("gse141983_qc_figure_manifest_failed")
    if raw_supplement_manifest.get("status") != "pass" or raw_supplement_manifest.get(
        "violations"
    ):
        violations.append("gse141983_supplement_manifest_failed")
    if transport_audit.get("status") != "pass" or transport_audit.get("violations"):
        violations.append("gse141983_transport_audit_failed")
    if implementation_amendment.get("status") != "frozen_implementation_correction":
        violations.append("gse141983_implementation_amendment_invalid")
    if sha256_file(parent_freeze_path) != implementation_amendment.get(
        "parent_freeze_sha256"
    ):
        violations.append("gse141983_parent_freeze_hash_failed")

    payloads = [
        binary_manifest,
        binary_summary,
        raw_summary,
        raw_gate,
        raw_verification,
        raw_figure_manifest,
        raw_supplement_manifest,
        transport_audit,
        implementation_amendment,
    ]
    if any(payload.get("malus_accessed") is not False for payload in payloads):
        violations.append("malus_seal_invariant_failed")

    artifact_relatives = [
        "docs/publication_v3_exploratory_prunus_chromatin_replication_contract.md",
        "config/publication_v3_exploratory_prunus_chromatin_replication_freeze.json",
        "results/biological_cases/prunus_publication_v3_chromatin_replication/summary.json",
        "results/biological_cases/prunus_publication_v3_chromatin_replication/posthoc_chromosome_robustness_summary.json",
        "results/metrics/publication_v3_prunus_chromatin_evidence_manifest.json",
        "figures/publication_v3/supplementary/figS_prunus_chromatin_binary.png",
        "figures/publication_v3/supplementary/figS_prunus_chromatin_binary.pdf",
        "figures/publication_v3/supplementary/figS_prunus_chromatin_binary.svg",
        "reports/PUBLICATION_V3_PRUNUS_CHROMATIN_SUPPLEMENT_EN.md",
        "docs/publication_v3_gse141983_raw_h3k4me3_sensitivity_contract.md",
        "config/publication_v3_gse141983_raw_sensitivity_freeze.json",
        "docs/publication_v3_gse141983_raw_readcount_implementation_amendment.md",
        "config/publication_v3_gse141983_raw_readcount_implementation_amendment.json",
        "metadata/publication_v3/gse141983_raw_sensitivity/eligible_h3_h3k4me3_runs.tsv",
        "results/metrics/publication_v3_gse141983_fastq_transport_retry_audit.json",
        "results/biological_cases/prunus_publication_v3_gse141983_raw/fastq_controller.json",
        "results/biological_cases/prunus_publication_v3_gse141983_raw/pipeline_controller.json",
        "results/biological_cases/prunus_publication_v3_gse141983_raw/technical_gate.json",
        "results/biological_cases/prunus_publication_v3_gse141983_raw/summary.json",
        "results/metrics/publication_v3_gse141983_raw_h3k4me3_verification.json",
        "results/metrics/publication_v3_gse141983_raw_qc_figure_manifest.json",
        "figures/publication_v3/supplementary/figS_gse141983_raw_technical_qc.png",
        "figures/publication_v3/supplementary/figS_gse141983_raw_technical_qc.pdf",
        "figures/publication_v3/supplementary/figS_gse141983_raw_technical_qc.svg",
        "results/metrics/publication_v3_gse141983_raw_qc_supplement_manifest.json",
        "reports/PUBLICATION_V3_GSE141983_RAW_QC_SUPPLEMENT_EN.md",
    ]
    missing = [relative for relative in artifact_relatives if not (root / relative).is_file()]
    violations.extend(f"missing_artifact:{relative}" for relative in missing)
    artifact_hashes = {
        relative: sha256_file(root / relative)
        for relative in artifact_relatives
        if (root / relative).is_file()
    }

    binary = binary_summary["binary_analysis"]
    mapping = [
        float(row["mapq30_nonduplicate_pair_fraction"])
        for row in raw_gate["sample_qc"]
    ]
    replicate_rho = [float(row["spearman_rho"]) for row in raw_gate["replicate_qc"]]
    manifest = {
        "status": "pass" if not violations else "fail",
        "scope": "complete_publication_v3_prunus_chromatin_evidence",
        "scientific_decision_authority": False,
        "malus_accessed": False,
        "reporting_roles": {
            "gse190586": "exploratory_supportive_binary_replication",
            "gse141983": "posthoc_raw_sensitivity_omitted_by_outcome_free_technical_gate",
        },
        "gse190586": {
            "status": binary_summary.get("status"),
            "positive_genes": int(binary["positive_genes"]),
            "direction_consistent_calls": int(binary["concordant_positive_genes"]),
            "observed_fraction": float(binary["observed_concordant_positive_fraction"]),
            "null_mean_fraction": float(binary["permutation_null_mean"]),
            "enrichment_ratio": float(binary["observed_concordant_positive_fraction"])
            / float(binary["permutation_null_mean"]),
            "one_sided_empirical_p": float(binary["one_sided_empirical_p"]),
        },
        "gse141983": {
            "status": raw_summary.get("status"),
            "samples": len(raw_gate["sample_qc"]),
            "failed_samples": sum(not bool(row["pass"]) for row in raw_gate["sample_qc"]),
            "mapq30_nonduplicate_pair_fraction_range": [min(mapping), max(mapping)],
            "replicate_spearman_range": [min(replicate_rho), max(replicate_rho)],
            "label_data_read": raw_summary.get("label_data_read"),
            "independent_verification": raw_verification.get("status"),
        },
        "invariants": {
            "primary_gse190586_reclassified": False,
            "pyrus_primary_mechanism_rescued": False,
            "dapt_model_selection_allowed": False,
            "malus_accessed": False,
        },
        "artifact_count": len(artifact_hashes),
        "artifact_sha256": artifact_hashes,
        "violations": violations,
    }
    output = root / args.output
    write_json(output, manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
