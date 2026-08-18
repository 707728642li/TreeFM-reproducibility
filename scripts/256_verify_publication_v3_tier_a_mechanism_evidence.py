#!/usr/bin/env python3
"""Independent verifier for the fixed Tier-A mechanism evidence layer.

This script intentionally does not import the builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ORDER = [
    "OG0000025",
    "OG0000191",
    "OG0000413",
    "OG0000350",
    "OG0004506",
    "OG0000215",
    "OG0000277",
    "OG0001301",
    "OG0001507",
    "OG0000208",
    "OG0000742",
    "OG0000139",
    "OG0000692",
    "OG0000083",
]

MODULE = {
    "OG0000025": "transcriptional relay",
    "OG0000191": "transcriptional relay",
    "OG0000413": "ABA metabolism/signaling",
    "OG0000350": "transcriptional relay",
    "OG0004506": "unresolved stress protein",
    "OG0000215": "ABA metabolism/signaling",
    "OG0000277": "ABA metabolism/signaling",
    "OG0001301": "transcriptional relay",
    "OG0001507": "transcriptional relay",
    "OG0000208": "receptor/transport/metabolism",
    "OG0000742": "transcriptional relay",
    "OG0000139": "receptor/transport/metabolism",
    "OG0000692": "receptor/transport/metabolism",
    "OG0000083": "receptor/transport/metabolism",
}

LIT_LEVEL = {
    "direct_same_family_same_process": 4,
    "same_pathway_same_process": 3,
    "family_expression_same_process": 2,
    "direct_same_family_related_process": 1,
    "annotation_only": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def is_close(a: object, b: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    candidate_freeze_path = root / "config/publication_v3_crossgenus_candidate_catalog_freeze.json"
    annotation_freeze_path = root / "config/publication_v3_tier_a_postselection_annotation_freeze.json"
    tier_path = root / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv"
    consensus_path = (
        root / "results/biological_cases/publication_v3_tier_a_annotation/orthogroup_domain_consensus.tsv"
    )
    domain_summary_path = root / "results/biological_cases/publication_v3_tier_a_annotation/domain_summary.json"
    domain_audit_path = (
        root / "results/biological_cases/publication_v3_tier_a_annotation/independent_audit.json"
    )
    literature_path = root / "metadata/publication_v3_tier_a_literature_evidence.tsv"
    contract_path = root / "docs/publication_v3_tier_a_mechanism_evidence_contract_v1.md"
    matrix_path = root / "metadata/publication_v3_tier_a_evidence_matrix.tsv"
    report_path = root / "reports/PUBLICATION_V3_TIER_A_MECHANISM_EVIDENCE_20260717_CN.md"
    png_path = root / "figures/publication_v3_tier_a_evidence_matrix.png"
    pdf_path = root / "figures/publication_v3_tier_a_evidence_matrix.pdf"
    svg_path = root / "figures/publication_v3_tier_a_evidence_matrix.svg"
    out_dir = root / "results/biological_cases/publication_v3_tier_a_mechanism_evidence"
    provenance_path = out_dir / "provenance.json"
    audit_path = out_dir / "independent_audit.json"

    required = [
        candidate_freeze_path,
        annotation_freeze_path,
        tier_path,
        consensus_path,
        domain_summary_path,
        domain_audit_path,
        literature_path,
        contract_path,
        matrix_path,
        report_path,
        png_path,
        pdf_path,
        svg_path,
        provenance_path,
    ]
    for path in required:
        check(path.is_file(), f"missing required file: {path.relative_to(root).as_posix()}")
    if failures:
        out_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "status": "fail",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "failure_count": len(failures),
            "failures": failures,
        }
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(json.dumps(audit, indent=2))

    candidate_freeze = json.loads(candidate_freeze_path.read_text(encoding="utf-8"))
    annotation_freeze = json.loads(annotation_freeze_path.read_text(encoding="utf-8"))
    domain_summary = json.loads(domain_summary_path.read_text(encoding="utf-8"))
    domain_audit = json.loads(domain_audit_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    tier = load_tsv(tier_path)
    consensus = load_tsv(consensus_path)
    literature = load_tsv(literature_path)
    matrix = load_tsv(matrix_path)

    check(candidate_freeze.get("status") == "pass", "candidate freeze status is not pass")
    check(candidate_freeze.get("freeze_version") == "2.0", "candidate freeze version is not 2.0")
    check(
        candidate_freeze.get("result_summary", {}).get("tier_a_families") == ORDER,
        "candidate freeze order differs from corrected fixed order",
    )
    check(not candidate_freeze.get("model_outputs_accessed"), "candidate freeze reports model access")
    check(not candidate_freeze.get("malus_accessed"), "candidate freeze reports Malus access")
    check(annotation_freeze.get("status") == "frozen", "annotation freeze is not frozen")
    check(annotation_freeze.get("freeze_version") == "2.0", "annotation freeze version is not 2.0")
    check(
        annotation_freeze.get("candidate_population", {}).get("candidate_freeze_sha256")
        == sha256(candidate_freeze_path),
        "annotation freeze does not pin current candidate freeze",
    )
    check(
        annotation_freeze.get("candidate_population", {}).get("tier_a_families") == ORDER,
        "annotation freeze order differs from corrected fixed order",
    )
    check(not annotation_freeze.get("model_outputs_accessed"), "annotation freeze reports model access")
    check(not annotation_freeze.get("malus_accessed"), "annotation freeze reports Malus access")
    check(domain_audit.get("status") == "pass", "full-Pfam audit status is not pass")
    check(domain_audit.get("failure_count", -1) == 0, "full-Pfam audit has failures")
    check(not domain_audit.get("model_outputs_accessed"), "full-Pfam audit reports model access")
    check(not domain_audit.get("malus_accessed"), "full-Pfam audit reports Malus access")

    frozen_tier = candidate_freeze.get("frozen_artifacts", {}).get(
        tier_path.relative_to(root).as_posix(), {}
    ).get("sha256")
    check(frozen_tier == sha256(tier_path), "Tier-A TSV hash differs from candidate freeze")

    check(len(tier) == 14, "Tier-A TSV does not contain 14 rows")
    check([row.get("orthogroup") for row in tier] == ORDER, "Tier-A TSV order is not exact")
    check([int(row["catalog_rank"]) for row in tier] == list(range(1, 15)), "Tier-A ranks are not 1..14")
    check(all(row.get("tier") == "A" for row in tier), "Tier-A TSV contains non-A rows")
    check(len(literature) == 14, "literature table does not contain 14 rows")
    check([row.get("orthogroup") for row in literature] == ORDER, "literature order is not exact")
    check(
        [int(row["catalog_rank"]) for row in literature] == list(range(1, 15)),
        "literature ranks are not 1..14",
    )
    check(
        all(not as_bool(row.get("selection_authority")) for row in literature),
        "literature table claims selection authority",
    )
    check(
        all(row.get("evidence_grade") in LIT_LEVEL for row in literature),
        "literature table contains unknown evidence grade",
    )

    lit_by_og = {row["orthogroup"]: row for row in literature}
    check(
        lit_by_og.get("OG0000191", {}).get("doi") == "10.1007/s00438-016-1171-6",
        "corrected peach WRKY DOI is absent for OG0000191",
    )
    check(
        lit_by_og.get("OG0000350", {}).get("doi") == "10.1007/s00438-016-1171-6",
        "corrected peach WRKY DOI is absent for OG0000350",
    )
    check(
        "10.3390/ijms19010310" in lit_by_og.get("OG0000413", {}).get("doi", ""),
        "corrected pear ABA DOI is absent for OG0000413",
    )
    check(
        "10.3390/ijms19010310" in lit_by_og.get("OG0000277", {}).get("doi", ""),
        "corrected pear ABA DOI is absent for OG0000277",
    )
    all_dois = ";".join(row.get("doi", "") for row in literature)
    check("10.1186/s12870-016-0808-0" not in all_dois, "obsolete incorrect WRKY DOI remains")
    check("10.3389/fpls.2017.02210" not in all_dois, "obsolete incorrect pear ABA DOI remains")
    for row in literature:
        for doi in filter(None, row.get("doi", "").split(";")):
            check(bool(re.fullmatch(r"10\.\d{4,9}/\S+", doi)), f"invalid DOI syntax: {doi}")

    consensus_by_og: dict[str, list[dict[str, str]]] = {og: [] for og in ORDER}
    for row in consensus:
        if row.get("orthogroup") in consensus_by_og:
            consensus_by_og[row["orthogroup"]].append(row)
    tier_by_og = {row["orthogroup"]: row for row in tier}
    matrix_by_og = {row["orthogroup"]: row for row in matrix}
    check(len(matrix) == 14, "evidence matrix does not contain 14 rows")
    check([row.get("orthogroup") for row in matrix] == ORDER, "evidence matrix order is not exact")
    check(len(matrix_by_og) == 14, "evidence matrix orthogroups are not unique")

    reconstructed_status: dict[str, str] = {}
    reconstructed_rows = 0
    matched_controls = 0
    for rank, og in enumerate(ORDER, start=1):
        candidate = tier_by_og[og]
        lit = lit_by_og[og]
        observed = matrix_by_og.get(og, {})
        detail = domain_summary.get("by_orthogroup", {}).get(og)
        check(detail is not None, f"domain summary missing {og}")
        if detail is None:
            continue
        three_way = sorted(
            {
                row["pfam_accession"]
                for row in consensus_by_og[og]
                if row.get("support_label") == "cross_genus_anchor_supported"
            }
        )
        three_way_names = sorted(
            {
                row["pfam_name"]
                for row in consensus_by_og[og]
                if row.get("support_label") == "cross_genus_anchor_supported"
            }
        )
        all_accessions = sorted({row["pfam_accession"] for row in consensus_by_og[og]})
        all_names = sorted({row["pfam_name"] for row in consensus_by_og[og]})
        if three_way:
            status = "cross_genus_anchor_supported"
            expected_accessions = three_way
            expected_names = three_way_names
        elif int(detail["proteins_with_pfam_hit"]) > 0:
            status = "partial"
            expected_accessions = all_accessions
            expected_names = all_names
        else:
            status = "unresolved"
            expected_accessions = []
            expected_names = []
        reconstructed_status[og] = status

        min_gbox = min(
            float(candidate["prunus_gbox_promoter_fraction"]),
            float(candidate["pyrus_gbox_promoter_fraction"]),
        )
        has_control = bool(candidate.get("matched_control_orthogroup"))
        matched_controls += int(has_control)

        expected_strings = {
            "catalog_rank": str(rank),
            "orthogroup": og,
            "retrospective_family_label": lit["retrospective_family_label"],
            "mechanism_module": MODULE[og],
            "endpoint_direction_prunus": candidate["prunus_directions"],
            "endpoint_direction_pyrus": candidate["pyrus_directions"],
            "leaf_go_term_count": candidate["leaf_go_term_count"],
            "leaf_go_term_ids": candidate["leaf_go_term_ids"],
            "leaf_go_term_names": candidate["leaf_go_term_names"],
            "prunus_gbox_gene_count": candidate["prunus_gbox_gene_count"],
            "prunus_labeled_gene_count": candidate["prunus_labeled_gene_count"],
            "pyrus_gbox_gene_count": candidate["pyrus_gbox_gene_count"],
            "pyrus_labeled_gene_count": candidate["pyrus_labeled_gene_count"],
            "matched_control_orthogroup": candidate["matched_control_orthogroup"],
            "pfam_support_status": status,
            "supported_pfam_accessions": ";".join(expected_accessions),
            "supported_pfam_names": ";".join(expected_names),
            "proteins_total": str(detail["proteins_total"]),
            "proteins_with_pfam_hit": str(detail["proteins_with_pfam_hit"]),
            "literature_evidence_grade": lit["evidence_grade"],
            "literature_evidence_level": str(LIT_LEVEL[lit["evidence_grade"]]),
            "prior_evidence_note": lit["prior_evidence_note"],
            "source_url": lit["source_url"],
            "doi": lit["doi"],
        }
        for field, expected in expected_strings.items():
            check(observed.get(field) == expected, f"{og}: field {field} differs")

        expected_floats = {
            "prunus_gbox_promoter_fraction": candidate["prunus_gbox_promoter_fraction"],
            "pyrus_gbox_promoter_fraction": candidate["pyrus_gbox_promoter_fraction"],
            "min_genus_gbox_promoter_fraction": min_gbox,
        }
        for field, expected in expected_floats.items():
            check(is_close(observed.get(field), expected), f"{og}: float field {field} differs")

        check(
            as_bool(observed.get("strict_matched_control")) == has_control,
            f"{og}: strict control flag differs",
        )
        check(not as_bool(observed.get("selection_authority")), f"{og}: selection authority is true")
        check(not as_bool(observed.get("model_outputs_accessed")), f"{og}: model access is true")
        check(not as_bool(observed.get("malus_accessed")), f"{og}: Malus access is true")
        check(candidate["prunus_directions"] == "down", f"{og}: Prunus endpoint is not down")
        check(candidate["pyrus_directions"] == "down", f"{og}: Pyrus endpoint is not down")
        reconstructed_rows += 1

    status_counts = Counter(reconstructed_status.values())
    check(reconstructed_rows == 14, "did not reconstruct all 14 matrix rows")
    check(status_counts == Counter({"cross_genus_anchor_supported": 12, "partial": 1, "unresolved": 1}),
          f"unexpected Pfam support counts: {dict(status_counts)}")
    check(matched_controls == 6, f"strict matched-control count is {matched_controls}, expected 6")

    report = report_path.read_text(encoding="utf-8")
    for og in ORDER:
        check(og in report, f"report omits {og}")
    for phrase in [
        "没有因为结构域或文献结果进行事后删除或重排",
        "不证明本研究",
        "G-box 位于候选基因启动子",
        "不使用放宽后对照补齐",
    ]:
        check(phrase in report, f"report missing boundary statement: {phrase}")

    check(png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature is invalid")
    check(pdf_path.read_bytes()[:4] == b"%PDF", "PDF signature is invalid")
    check("<svg" in svg_path.read_text(encoding="utf-8", errors="replace")[:5000], "SVG signature is invalid")
    check(png_path.stat().st_size > 50_000, "PNG is unexpectedly small")
    check(pdf_path.stat().st_size > 20_000, "PDF is unexpectedly small")
    check(svg_path.stat().st_size > 20_000, "SVG is unexpectedly small")

    check(provenance.get("status") == "pass", "provenance status is not pass")
    check(provenance.get("candidate_count") == 14, "provenance candidate count is not 14")
    check(provenance.get("candidate_order") == ORDER, "provenance order is not exact")
    check(not provenance.get("selection_authority"), "provenance claims selection authority")
    check(not provenance.get("candidate_reranking_performed"), "provenance reports re-ranking")
    check(not provenance.get("model_outputs_accessed"), "provenance reports model access")
    check(not provenance.get("malus_accessed"), "provenance reports Malus access")
    check(provenance.get("violations") == [], "provenance has violations")
    accessed_paths = provenance.get("accessed_paths", [])
    prohibited_tokens = ["/models/", "embedding", "probe", "malus"]
    for path in accessed_paths:
        lower = f"/{path.lower()}"
        for token in prohibited_tokens:
            check(token not in lower, f"provenance accessed prohibited path: {path}")

    expected_inputs = [
        candidate_freeze_path,
        annotation_freeze_path,
        tier_path,
        domain_summary_path,
        consensus_path,
        domain_audit_path,
        literature_path,
        contract_path,
    ]
    for path in expected_inputs:
        rel = path.relative_to(root).as_posix()
        check(provenance.get("inputs", {}).get(rel) == sha256(path), f"input hash mismatch: {rel}")
    expected_outputs = [matrix_path, report_path, png_path, pdf_path, svg_path]
    for path in expected_outputs:
        rel = path.relative_to(root).as_posix()
        check(provenance.get("outputs", {}).get(rel) == sha256(path), f"output hash mismatch: {rel}")

    audit = {
        "status": "pass" if not failures else "fail",
        "scope": "independent_reimplementation_tier_a_mechanism_evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_authority": False,
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "reconstructed_candidate_rows": reconstructed_rows,
        "reconstructed_candidate_order": ORDER,
        "reconstructed_pfam_support_counts": dict(status_counts),
        "reconstructed_strict_matched_controls": matched_controls,
        "verified_field_count_per_candidate": 31,
        "gates": {
            "candidate_and_annotation_freezes_current": not any(
                "freeze" in item.lower() for item in failures
            ),
            "all_candidate_fields_reimplemented": not any(": field " in item or ": float field " in item for item in failures),
            "corrected_dois_present": not any("DOI" in item for item in failures),
            "all_14_candidates_retained": len(matrix) == 14,
            "rank_order_preserved": [row.get("orthogroup") for row in matrix] == ORDER,
            "strict_controls_not_relaxed": matched_controls == 6,
            "figure_file_signatures_and_sizes_pass": not any(
                token in item for item in failures for token in ["PNG", "PDF", "SVG"]
            ),
            "provenance_hashes_exact": not any("hash mismatch" in item for item in failures),
            "no_model_access": True,
            "no_malus_access": True,
        },
        "hashes": {
            path.relative_to(root).as_posix(): sha256(path)
            for path in [matrix_path, report_path, png_path, pdf_path, svg_path, provenance_path]
        },
        "failure_count": len(failures),
        "failures": failures,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
