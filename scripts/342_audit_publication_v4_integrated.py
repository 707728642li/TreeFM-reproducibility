#!/usr/bin/env python3
"""Integrated, read-only audit of the publication-v4 evidence and figure system."""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metrics" / "publication_v4_integrated_audit.json"
MANUSCRIPT = ROOT / "reports" / "PUBLICATION_V4_FULL_MANUSCRIPT_DRAFT_EN.md"
QA_RECORD = ROOT / "reports" / "PUBLICATION_V4_FIGURE_QA_20260803.md"
INVENTORY = ROOT / "metadata" / "publication_v4_figure_inventory.tsv"

COMPONENT_AUDITS = [
    ("corpus_phylogeny", ROOT / "results/metrics/publication_v4_corpus_phylogeny_audit.json", 19),
    ("representation", ROOT / "results/metrics/publication_v4_representation_audit.json", 28),
    ("go_stability", ROOT / "results/metrics/publication_v4_go_stability_audit.json", 23),
    ("tier_a_comparative", ROOT / "results/metrics/publication_v4_tier_a_comparative_audit.json", 22),
    ("tier_a_mechanism_map", ROOT / "results/metrics/publication_v4_tier_a_mechanism_map_audit.json", 18),
    ("functional_conclusion", ROOT / "results/metrics/publication_v4_functional_conclusion_audit.json", 20),
    ("technical_qc", ROOT / "results/metrics/publication_v4_technical_qc_audit.json", 23),
]

FIGURES = [
    "publication_v4_corpus_phylogeny",
    "publication_v4_go_robustness",
    "publication_v4_go_loo_all52",
    "publication_v4_tier_a_comparative",
    "publication_v4_tier_a_mechanism_map",
    "publication_v4_functional_conclusion_reversal",
    "publication_v4_functional_secondary_qc",
    "publication_v4_representation_similarity",
    "publication_v4_technical_effects",
    "publication_v4_sequence_leakage_qc",
]


records: list[dict] = []


def check(name: str, passed: bool, observed, expected) -> None:
    records.append(
        {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}
    )


def audit_pass_count(payload: dict) -> int:
    if "passed_checks" in payload:
        return int(payload["passed_checks"])
    return int(payload.get("checks_passed", 0))


def audit_total_count(payload: dict) -> int:
    value = payload.get("checks_total", payload.get("checks", 0))
    return len(value) if isinstance(value, list) else int(value)


def main() -> int:
    component_summary = {}
    component_passed = 0
    component_total = 0
    for label, path, expected_checks in COMPONENT_AUDITS:
        exists = path.is_file()
        check(f"component_{label}_exists", exists, str(path.relative_to(ROOT)) if exists else None, True)
        if not exists:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        passed = audit_pass_count(payload)
        total = audit_total_count(payload)
        status = str(payload.get("status", "")).lower()
        component_summary[label] = {"status": status, "passed": passed, "total": total}
        check(
            f"component_{label}_pass",
            status == "pass" and passed == total == expected_checks,
            component_summary[label],
            {"status": "pass", "passed": expected_checks, "total": expected_checks},
        )
        component_passed += passed
        component_total += total

    check(
        "component_check_total",
        component_passed == component_total == 153,
        {"passed": component_passed, "total": component_total},
        {"passed": 153, "total": 153},
    )

    manuscript_exists = MANUSCRIPT.is_file()
    check("manuscript_exists", manuscript_exists, str(MANUSCRIPT.relative_to(ROOT)), True)
    manuscript = MANUSCRIPT.read_text(encoding="utf-8") if manuscript_exists else ""
    word_count = len(re.findall(r"\b[\w][\w'–-]*\b", manuscript, flags=re.UNICODE))
    check("manuscript_word_count", word_count >= 6500, word_count, ">=6500")

    required_sections = [
        "## Abstract",
        "## Introduction",
        "## Results",
        "## Discussion",
        "## Methods",
        "## Data and code availability",
        "## Figure legends",
        "## References",
    ]
    missing_sections = [section for section in required_sections if section not in manuscript]
    check("manuscript_sections", not missing_sections, missing_sections, [])

    missing_main_legends = [f"### Figure {i}." for i in range(1, 8) if f"### Figure {i}." not in manuscript]
    missing_supp_legends = [
        f"### Supplementary Figure S{i}."
        for i in range(1, 7)
        if f"### Supplementary Figure S{i}." not in manuscript
    ]
    check("main_figure_legends", not missing_main_legends, missing_main_legends, [])
    check("supplementary_figure_legends", not missing_supp_legends, missing_supp_legends, [])

    unresolved = sorted(set(re.findall(r"\b(?:TODO|TBD|FIXME|XXX)\b", manuscript, flags=re.IGNORECASE)))
    check("no_unresolved_markers", not unresolved, unresolved, [])

    required_claims = {
        "go_terms_and_permutations": ["52", "10,000", "1/10,001"],
        "tier_a": ["14 Tier-A", "Twelve Tier-A", "ten were present in all 26 species"],
        "representation": ["0.998125", "25 of 26 species", "all four tasks"],
        "technical": ["16 of 24", "only six cells", "zero 95% intervals"],
        "stopping_boundary": ["seeds 41 and 59 were not authorized", "Malus remained sealed"],
    }
    for label, needles in required_claims.items():
        missing = [needle for needle in needles if needle not in manuscript]
        check(f"claim_{label}", not missing, missing, [])

    figure_results = {}
    for basename in FIGURES:
        formats = {}
        for extension in ("png", "pdf", "svg"):
            path = ROOT / "results" / "figures" / f"{basename}.{extension}"
            ok = path.is_file() and path.stat().st_size > 10_000
            detail = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
            if ok and extension == "png":
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        detail["dimensions"] = list(image.size)
                        ok = image.width >= 1200 and image.height >= 700
                except Exception as exc:  # pragma: no cover - diagnostic path
                    detail["error"] = repr(exc)
                    ok = False
            elif ok and extension == "svg":
                try:
                    root = ElementTree.parse(path).getroot()
                    detail["root_tag"] = root.tag
                    ok = root.tag.endswith("svg")
                except Exception as exc:  # pragma: no cover - diagnostic path
                    detail["error"] = repr(exc)
                    ok = False
            elif ok and extension == "pdf":
                raw = path.read_bytes()
                detail["pdf_header"] = raw[:5].decode("ascii", errors="replace")
                detail["has_eof"] = b"%%EOF" in raw[-2048:]
                ok = raw.startswith(b"%PDF-") and detail["has_eof"]
            formats[extension] = {"passed": ok, **detail}
            check(f"figure_{basename}_{extension}", ok, detail, "valid publication artifact")
        figure_results[basename] = formats

    qa_exists = QA_RECORD.is_file()
    qa_text = QA_RECORD.read_text(encoding="utf-8") if qa_exists else ""
    check("manual_qa_record_exists", qa_exists, str(QA_RECORD.relative_to(ROOT)), True)
    missing_qa = [basename for basename in FIGURES if basename not in qa_text]
    check("manual_qa_covers_all_figures", not missing_qa and "Decision: **PASS**" in qa_text, missing_qa, [])

    inventory_exists = INVENTORY.is_file()
    inventory_rows = []
    if inventory_exists:
        with INVENTORY.open("r", encoding="utf-8", newline="") as handle:
            inventory_rows = list(csv.DictReader(handle, delimiter="\t"))
    check("figure_inventory_exists", inventory_exists, str(INVENTORY.relative_to(ROOT)), True)
    check(
        "figure_inventory_complete",
        len(inventory_rows) == 10 and {row.get("basename") for row in inventory_rows} == set(FIGURES),
        {"rows": len(inventory_rows), "basenames": sorted(row.get("basename", "") for row in inventory_rows)},
        {"rows": 10, "basenames": sorted(FIGURES)},
    )
    check(
        "figure_inventory_all_pass",
        bool(inventory_rows) and all(row.get("manual_qa") == "PASS" for row in inventory_rows),
        sorted(set(row.get("manual_qa", "") for row in inventory_rows)),
        ["PASS"],
    )

    expected_scripts = [
        "327_analyze_and_plot_publication_v4_corpus_phylogeny.py",
        "328_audit_publication_v4_corpus_phylogeny.py",
        "329_analyze_publication_v4_representations.py",
        "330_audit_publication_v4_representations.py",
        "331_plot_publication_v4_representations.py",
        "332_analyze_and_plot_publication_v4_go_stability.py",
        "333_audit_publication_v4_go_stability.py",
        "334_analyze_and_plot_publication_v4_tier_a_comparative.py",
        "335_audit_publication_v4_tier_a_comparative.py",
        "336_plot_publication_v4_tier_a_mechanism_map.py",
        "337_audit_publication_v4_tier_a_mechanism_map.py",
        "338_plot_publication_v4_functional_conclusion.py",
        "339_audit_publication_v4_functional_conclusion.py",
        "340_plot_publication_v4_technical_and_leakage.py",
        "341_audit_publication_v4_technical_and_leakage.py",
        "342_audit_publication_v4_integrated.py",
    ]
    missing_scripts = [name for name in expected_scripts if not (ROOT / "scripts" / name).is_file()]
    check("analysis_scripts_complete", not missing_scripts, missing_scripts, [])

    failures = [record for record in records if not record["passed"]]
    payload = {
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "publication_v4_integrated_scientific_and_figure_audit",
        "checks": len(records),
        "passed_checks": len(records) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "component_checks": {"passed": component_passed, "total": component_total},
        "manuscript_word_count": word_count,
        "figure_count": len(FIGURES),
        "figure_format_artifacts": len(FIGURES) * 3,
        "component_summary": component_summary,
        "figure_results": figure_results,
        "records": records,
        "analysis_tier": "posthoc_seed23_descriptive_plus_frozen_primary",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "seeds_41_59_authorized": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "checks", "passed_checks", "failure_count", "component_checks", "manuscript_word_count", "figure_count")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
