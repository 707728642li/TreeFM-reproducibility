#!/usr/bin/env python3
"""Independent audit for the publication-v4 GO robustness artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_LEAF = {
    "GO:0006970", "GO:0042742", "GO:1901701", "GO:0009409", "GO:0009414",
    "GO:0009611", "GO:0009411", "GO:0071456", "GO:0009737", "GO:0032870",
    "GO:0071396", "GO:0009753", "GO:0009751", "GO:0071669", "GO:0042545",
    "GO:0006721", "GO:0008299", "GO:0009699",
}


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
    metric = root / "results/metrics/publication_v4_go_stability"
    source = root / "results/biological_cases/publication_v3_crossgenus_go"
    summary = json.loads((metric / "summary.json").read_text(encoding="utf-8"))
    leaf = pd.read_csv(metric / "go_leaf_forest.tsv", sep="\t")
    all52 = pd.read_csv(metric / "go_loo_stability_all52.tsv", sep="\t")
    leaf18 = pd.read_csv(metric / "go_loo_stability_leaf18.tsv", sep="\t")
    edges = pd.read_csv(metric / "go_ontology_edges.tsv", sep="\t")
    null = pd.read_csv(metric / "go_permutation_null.tsv.gz", sep="\t")
    frozen_terms_all = pd.read_csv(source / "robust_replicated_terms.tsv", sep="\t")
    frozen_terms = frozen_terms_all.loc[frozen_terms_all["evidence_layer"] == "curated_no_iea"].copy()
    frozen_loo = pd.read_csv(source / "curated_no_iea_leave_one_chromosome.tsv", sep="\t")

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("summary_complete", summary.get("status") == "complete", summary.get("status"))
    check("posthoc_no_authority", summary.get("scientific_decision_authority") is False, summary.get("analysis_tier"))
    check("malus_not_accessed", summary.get("malus_accessed") is False, summary.get("malus_accessed"))
    check("frozen_term_count", len(frozen_terms) == 52 == summary.get("robust_replicated_terms"), len(frozen_terms))
    check("fixed_leaf_identity", set(leaf["term_id"]) == EXPECTED_LEAF and len(leaf) == 18, sorted(leaf["term_id"].tolist()))
    check("four_fixed_modules", leaf["module"].nunique() == 4 and not leaf["module"].isna().any(), leaf["module"].value_counts().to_dict())
    check("permutation_count", len(null) == 10_000, len(null))
    check("null_maximum", int(null["replicated_fdr_hits"].max()) == 12, int(null["replicated_fdr_hits"].max()))
    check("observed_exceeds_null", summary.get("observed_hits") == 52 and 52 > int(null["replicated_fdr_hits"].max()), summary.get("observed_hits"))
    check("empirical_p_exact", np.isclose(float(summary.get("empirical_p")), 1 / 10001), summary.get("empirical_p"))
    check("all52_refit_rows", len(all52) == 52 * 25 == 1300, len(all52))
    per_term = all52.groupby("term_id")["left_out_chromosome"].nunique()
    check("every_term_all_chromosomes", len(per_term) == 52 and bool((per_term == 25).all()), per_term.value_counts().to_dict())
    per_genus = all52.groupby("genus")["left_out_chromosome"].nunique().to_dict()
    check("genus_chromosome_coverage", per_genus == {"prunus": 8, "pyrus": 17}, per_genus)
    check("leaf_refit_rows", len(leaf18) == 18 * 25, len(leaf18))
    check("loo_numerical_match", np.allclose(all52["odds_ratio_haldane"], frozen_loo["odds_ratio_haldane"], rtol=0, atol=1e-12), "all rows")
    check("loo_log_transform", np.allclose(all52["log2_odds_ratio"], np.log2(all52["odds_ratio_haldane"]), rtol=0, atol=1e-12), "all rows")
    check("all_directions_positive", bool(all52["direction_positive"].astype(bool).all()), int(all52["direction_positive"].astype(bool).sum()))
    check("edge_endpoints_fixed", set(edges["source"]).union(edges["target"]).issubset(EXPECTED_LEAF), len(edges))
    check("edge_effect_independent_label", bool(edges["edge_basis"].str.contains("ancestor", case=False).all()), edges["edge_basis"].unique().tolist())
    check("edge_weights_valid", bool(edges["ancestor_jaccard"].between(0, 1, inclusive="both").all()), [float(edges["ancestor_jaccard"].min()), float(edges["ancestor_jaccard"].max())])

    expected_outputs = [
        root / f"results/figures/publication_v4_go_robustness.{ext}" for ext in ["png", "pdf", "svg"]
    ] + [root / f"results/figures/publication_v4_go_loo_all52.{ext}" for ext in ["png", "pdf", "svg"]]
    check("all_figure_formats_exist", all(p.exists() and p.stat().st_size > 10_000 for p in expected_outputs), {p.name: p.stat().st_size if p.exists() else None for p in expected_outputs})
    hash_ok = True
    for path in expected_outputs:
        key = str(path.relative_to(root)).replace("\\", "/")
        hash_ok &= key in summary["outputs"] and summary["outputs"][key]["sha256"] == sha256(path)
    check("figure_hashes_match_metadata", hash_ok, len(expected_outputs))
    forbidden = [key for key in summary.get("inputs", {}) if "malus" in key.lower()]
    check("no_malus_input_paths", not forbidden, forbidden)

    passed = sum(bool(x["passed"]) for x in checks)
    audit = {
        "status": "pass" if passed == len(checks) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    out = root / "results/metrics/publication_v4_go_stability_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["status", "checks_passed", "checks_total"]}, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
