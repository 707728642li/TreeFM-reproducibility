#!/usr/bin/env python3
"""Independent audit for Tier-A comparative-genomics artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from Bio import Phylo


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
    metric = root / "results/metrics/publication_v4_tier_a_comparative"
    summary = json.loads((metric / "summary.json").read_text(encoding="utf-8"))
    copy_number = pd.read_csv(metric / "tier_a_copy_number.tsv", sep="\t")
    candidates = pd.read_csv(metric / "tier_a_candidate_summary.tsv", sep="\t")
    evidence = pd.read_csv(metric / "tier_a_evidence_layers.tsv", sep="\t")
    species_order = pd.read_csv(metric / "tier_a_species_order.tsv", sep="\t")
    tier_frozen = pd.read_csv(root / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv", sep="\t")
    tree = Phylo.read(root / "results/metrics/publication_v4_corpus_phylogeny/species_tree_named_rooted.nwk", "newick")

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("summary_complete", summary.get("status") == "complete", summary.get("status"))
    check("no_decision_authority", summary.get("scientific_decision_authority") is False and summary.get("candidate_selection_allowed") is False, summary.get("analysis_tier"))
    check("no_causal_claim", summary.get("causal_claim") is False, summary.get("causal_claim"))
    check("malus_not_accessed", summary.get("malus_accessed") is False, summary.get("malus_accessed"))
    check("frozen_14_identity", len(candidates) == 14 and set(candidates["orthogroup"]) == set(tier_frozen["orthogroup"]), candidates["orthogroup"].tolist())
    check("frozen_catalog_order", candidates.sort_values("catalog_rank")["orthogroup"].tolist() == tier_frozen.sort_values("catalog_rank")["orthogroup"].tolist(), candidates.sort_values("catalog_rank")["orthogroup"].tolist())
    tree_species = [tip.name for tip in tree.get_terminals()]
    check("tree_26_species", len(tree_species) == 26 and len(set(tree_species)) == 26, tree_species)
    check("species_order_matches_tree", species_order.sort_values("tree_order")["slug"].tolist() == tree_species, species_order.sort_values("tree_order")["slug"].tolist())
    check("complete_26x14_matrix", len(copy_number) == 26 * 14 and not copy_number.duplicated(["species", "orthogroup"]).any(), len(copy_number))
    check("copy_numbers_nonnegative_integer", bool((copy_number["copy_number"] >= 0).all()) and bool((copy_number["copy_number"] % 1 == 0).all()), [int(copy_number["copy_number"].min()), int(copy_number["copy_number"].max())])

    # Independently parse the minimal server extract and recompute every cell.
    parsed: dict[str, Counter] = {}
    extract = metric / "tier_a_orthogroups.txt"
    for line in extract.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        orthogroup, raw = line.split(":", 1)
        parsed[orthogroup] = Counter(member.split("|", 1)[0] for member in raw.strip().split())
    recomputed = []
    for orthogroup in tier_frozen.sort_values("catalog_rank")["orthogroup"]:
        for species in tree_species:
            recomputed.append((orthogroup, species, int(parsed[orthogroup][species])))
    expected = pd.DataFrame(recomputed, columns=["orthogroup", "species", "copy_number"]).sort_values(["orthogroup", "species"]).reset_index(drop=True)
    observed = copy_number[["orthogroup", "species", "copy_number"]].sort_values(["orthogroup", "species"]).reset_index(drop=True)
    check("all_copy_cells_recomputed", expected.equals(observed), int((expected["copy_number"] != observed["copy_number"]).sum()))
    check("member_total_recomputed", int(expected["copy_number"].sum()) == int(summary["member_gene_representatives"]), int(expected["copy_number"].sum()))
    breadth = expected.groupby("orthogroup")["copy_number"].apply(lambda x: int((x > 0).sum())).to_dict()
    summary_breadth = candidates.set_index("orthogroup")["species_present"].astype(int).to_dict()
    check("breadth_recomputed", breadth == summary_breadth, breadth)
    check("four_fixed_modules", set(candidates["mechanism_module"]) == {"transcriptional relay", "ABA metabolism/signaling", "receptor/transport/metabolism", "unresolved stress protein"}, candidates["mechanism_module"].value_counts().to_dict())
    check("evidence_complete", len(evidence) == 14 and set(evidence["orthogroup"]) == set(candidates["orthogroup"]), len(evidence))
    bool_cols = ["leaf_go", "direction_conserved", "gbox_both_genera", "pfam_three_way", "h3k4me3", "strict_matched_control"]
    check("evidence_fields_binary", all(set(evidence[col].astype(str).str.lower()).issubset({"true", "false", "0", "1"}) for col in bool_cols), {col: evidence[col].value_counts().to_dict() for col in bool_cols})
    check("three_way_pfam_count", int(evidence["pfam_three_way"].astype(bool).sum()) == int(summary["three_way_pfam_supported"]), int(evidence["pfam_three_way"].astype(bool).sum()))
    check("h3k4me3_count", int(evidence["h3k4me3"].astype(bool).sum()) == int(summary["h3k4me3_intersecting"]), int(evidence["h3k4me3"].astype(bool).sum()))
    check("literature_levels_valid", bool(evidence["literature_evidence_level"].between(0, 4, inclusive="both").all()), evidence["literature_evidence_level"].value_counts().to_dict())

    figures = [root / f"results/figures/publication_v4_tier_a_comparative.{ext}" for ext in ["png", "pdf", "svg"]]
    check("all_figure_formats_exist", all(p.exists() and p.stat().st_size > 10_000 for p in figures), {p.name: p.stat().st_size if p.exists() else None for p in figures})
    hash_ok = True
    for path in figures:
        key = str(path.relative_to(root)).replace("\\", "/")
        hash_ok &= key in summary["outputs"] and summary["outputs"][key]["sha256"] == sha256(path)
    check("figure_hashes_match_metadata", hash_ok, len(figures))
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
    out = root / "results/metrics/publication_v4_tier_a_comparative_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["status", "checks_passed", "checks_total"]}, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
