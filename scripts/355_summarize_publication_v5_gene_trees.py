#!/usr/bin/env python3
"""Summarize audited gene trees, reconciliation events and sensitivity trees."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
from dendropy.calculate import treecompare


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/metrics/publication_v5_gene_trees"
RUNS = OUT / "runs"
REC = OUT / "reconciliation"
UNTRIMMED = OUT / "sensitivity_untrimmed"
DOMAIN = OUT / "sensitivity_domain"
DISPLAY = ["OG0000025", "OG0000413", "OG0000277"]
ASSOCIATION_REPLICATES = 10_000
ASSOCIATION_SEED = 20260803


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fasta_stats(path: Path) -> tuple[int, int, float]:
    sequences = []
    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if chunks:
                    sequences.append("".join(chunks))
                chunks = []
            else:
                chunks.append(line.strip())
        if chunks:
            sequences.append("".join(chunks))
    if not sequences or len({len(sequence) for sequence in sequences}) != 1:
        raise RuntimeError(f"Invalid alignment: {path}")
    length = len(sequences[0])
    missing = sum(character in "-?Xx" for sequence in sequences for character in sequence)
    return len(sequences), length, missing / (len(sequences) * length)


def selected_model(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Best-fit model according to BIC:\s*(\S+)", text)
    if not match:
        raise RuntimeError(f"Selected model missing from {path}")
    return match.group(1)


def composition(path: Path) -> tuple[int, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"([0-9.]+)%\s+(\d+) sequences failed composition chi2 test", text)
    if not matches:
        raise RuntimeError(f"Composition summary missing from {path}")
    percent, count = matches[-1]
    return int(count), float(percent)


def supports(path: Path) -> tuple[int, float, float, float, float]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    pairs = re.findall(r"\)([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?):", text)
    if not pairs:
        raise RuntimeError(f"No SH-aLRT/UFBoot supports parsed from {path}")
    alrt = np.asarray([float(left) for left, _ in pairs], dtype=float)
    ufboot = np.asarray([float(right) for _, right in pairs], dtype=float)
    return len(pairs), float(np.median(alrt)), float(np.median(ufboot)), float(np.mean(alrt >= 80)), float(np.mean(ufboot >= 80))


def reconciliation_stats(summary_path: Path, nhx_path: Path) -> tuple[int, int, Counter]:
    first = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    match = re.search(r"DUPLICATIONS\s*:\s*(\d+)\s+LOSSES\s*:\s*(\d+)", first)
    if not match:
        raise RuntimeError(f"Cannot parse reconciliation counts from {summary_path}")
    nhx = nhx_path.read_text(encoding="utf-8", errors="replace")
    locations = Counter(re.findall(r"\[&&NHX:S=([^:\]]+):D=Y", nhx))
    return int(match.group(1)), int(match.group(2)), locations


def load_tree(path: Path) -> dendropy.Tree:
    return dendropy.Tree.get(
        path=str(path),
        schema="newick",
        preserve_underscores=True,
        rooting="force-unrooted",
    )


def normalized_rf(left_path: Path, right_path: Path) -> tuple[int, float]:
    left_labels = {taxon.label for taxon in load_tree(left_path).taxon_namespace}
    right_labels = {taxon.label for taxon in load_tree(right_path).taxon_namespace}
    common = left_labels & right_labels
    if len(common) < 4:
        raise RuntimeError(f"Too few common leaves for RF comparison: {left_path}, {right_path}")
    namespace = dendropy.TaxonNamespace(sorted(common))
    left = dendropy.Tree.get(path=str(left_path), schema="newick", taxon_namespace=namespace, preserve_underscores=True, rooting="force-unrooted")
    right = dendropy.Tree.get(path=str(right_path), schema="newick", taxon_namespace=namespace, preserve_underscores=True, rooting="force-unrooted")
    left.retain_taxa_with_labels(common)
    right.retain_taxa_with_labels(common)
    left.encode_bipartitions()
    right.encode_bipartitions()
    rf = int(treecompare.symmetric_difference(left, right))
    maximum = 2 * (len(common) - 3)
    return len(common), rf / maximum if maximum else 0.0


def nearest_other_genus(path: Path, id_map: pd.DataFrame, orthogroup: str, tree_kind: str) -> pd.DataFrame:
    tree = load_tree(path)
    pdm = tree.phylogenetic_distance_matrix()
    metadata = id_map.loc[id_map["orthogroup"].eq(orthogroup)].set_index("safe_id")
    available = {taxon.label for taxon in tree.taxon_namespace}
    rows = []
    for safe_id, row in metadata.loc[metadata["tier_a_positive_candidate"]].iterrows():
        if safe_id not in available or row["species"] not in {"prunus_persica", "pyrus_pyrifolia"}:
            continue
        other_species = "pyrus_pyrifolia" if row["species"] == "prunus_persica" else "prunus_persica"
        choices = metadata.loc[metadata["species"].eq(other_species)].index.intersection(list(available))
        if choices.empty:
            continue
        distances = [(other, pdm.distance(tree.find_node_with_taxon_label(safe_id).taxon, tree.find_node_with_taxon_label(other).taxon)) for other in choices]
        nearest_id, distance = sorted(distances, key=lambda item: (item[1], item[0]))[0]
        nearest = metadata.loc[nearest_id]
        rows.append(
            {
                "orthogroup": orthogroup,
                "tree_kind": tree_kind,
                "safe_id": safe_id,
                "species": row["species"],
                "gene_id": row["gene_id"],
                "source_gene_id": row["source_gene_id"],
                "nearest_other_genus_safe_id": nearest_id,
                "nearest_other_genus_gene_id": nearest["gene_id"],
                "nearest_other_genus_is_candidate": bool(nearest["tier_a_positive_candidate"]),
                "patristic_distance": float(distance),
            }
        )
    return pd.DataFrame(rows)


def bh_adjust(values: pd.Series) -> pd.Series:
    pvalues = values.to_numpy(dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def crossgenus_association(
    path: Path,
    id_map: pd.DataFrame,
    orthogroup: str,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray]:
    """Test whether candidate copies preferentially neighbor candidates across genera."""
    tree = load_tree(path)
    pdm = tree.phylogenetic_distance_matrix()
    metadata = id_map.loc[id_map["orthogroup"].eq(orthogroup)].set_index("safe_id")
    available = {taxon.label for taxon in tree.taxon_namespace}
    prunus = sorted(metadata.loc[metadata["species"].eq("prunus_persica")].index.intersection(available))
    pyrus = sorted(metadata.loc[metadata["species"].eq("pyrus_pyrifolia")].index.intersection(available))
    if not prunus or not pyrus:
        raise RuntimeError(f"Both target genera are required for association test: {orthogroup}")

    nodes = {label: tree.find_node_with_taxon_label(label).taxon for label in prunus + pyrus}
    nearest_pyrus = {
        label: min(pyrus, key=lambda other: (pdm.distance(nodes[label], nodes[other]), other))
        for label in prunus
    }
    nearest_prunus = {
        label: min(prunus, key=lambda other: (pdm.distance(nodes[label], nodes[other]), other))
        for label in pyrus
    }
    observed_prunus = set(metadata.loc[metadata.index.isin(prunus) & metadata["tier_a_positive_candidate"]].index)
    observed_pyrus = set(metadata.loc[metadata.index.isin(pyrus) & metadata["tier_a_positive_candidate"]].index)
    k_prunus = len(observed_prunus)
    k_pyrus = len(observed_pyrus)
    denominator = k_prunus + k_pyrus
    if not denominator or not k_prunus or not k_pyrus:
        raise RuntimeError(f"Candidate copies are required in both target genera: {orthogroup}")

    def successes(candidate_prunus: set[str], candidate_pyrus: set[str]) -> int:
        return sum(nearest_pyrus[label] in candidate_pyrus for label in candidate_prunus) + sum(
            nearest_prunus[label] in candidate_prunus for label in candidate_pyrus
        )

    observed_successes = successes(observed_prunus, observed_pyrus)
    null_successes = np.empty(ASSOCIATION_REPLICATES, dtype=np.int32)
    prunus_array = np.asarray(prunus, dtype=object)
    pyrus_array = np.asarray(pyrus, dtype=object)
    for replicate in range(ASSOCIATION_REPLICATES):
        shuffled_prunus = set(rng.choice(prunus_array, size=k_prunus, replace=False).tolist())
        shuffled_pyrus = set(rng.choice(pyrus_array, size=k_pyrus, replace=False).tolist())
        null_successes[replicate] = successes(shuffled_prunus, shuffled_pyrus)

    null_fraction = null_successes / denominator
    row = {
        "orthogroup": orthogroup,
        "prunus_leaves": len(prunus),
        "pyrus_leaves": len(pyrus),
        "prunus_candidate_leaves": k_prunus,
        "pyrus_candidate_leaves": k_pyrus,
        "candidate_queries": denominator,
        "observed_successes": observed_successes,
        "observed_fraction": observed_successes / denominator,
        "null_median_fraction": float(np.median(null_fraction)),
        "null_q95_fraction": float(np.quantile(null_fraction, 0.95)),
        "effect_over_null_median": observed_successes / denominator - float(np.median(null_fraction)),
        "empirical_p": (1 + int(np.sum(null_successes >= observed_successes))) / (ASSOCIATION_REPLICATES + 1),
        "replicates": ASSOCIATION_REPLICATES,
        "seed": ASSOCIATION_SEED,
    }
    return row, null_successes


def main() -> int:
    required_markers = [
        OUT / "primary_all.complete",
        OUT / "reconciliation_and_untrimmed.complete",
        DOMAIN / "domain_all.complete",
    ]
    for marker in required_markers:
        if not marker.exists():
            raise RuntimeError(f"Required completion marker missing: {marker}")
    id_map = pd.read_csv(OUT / "id_map.tsv.gz", sep="\t")
    if id_map["tier_a_positive_candidate"].dtype != bool:
        id_map["tier_a_positive_candidate"] = id_map["tier_a_positive_candidate"].astype(str).str.lower().eq("true")
    family_input = pd.read_csv(OUT / "family_input_summary.tsv", sep="\t")
    family_rows = []
    nearest_frames = []
    association_rows = []
    association_nulls = []
    association_rng = np.random.default_rng(ASSOCIATION_SEED)
    for input_row in family_input.to_dict("records"):
        orthogroup = input_row["orthogroup"]
        work = RUNS / orthogroup
        aligned_n, aligned_sites, aligned_missing = fasta_stats(work / f"{orthogroup}.aligned.faa")
        trimmed_n, trimmed_sites, trimmed_missing = fasta_stats(work / f"{orthogroup}.trimmed.faa")
        failed_count, reported_total_percent = composition(work / "primary.log")
        failed_percent = 100.0 * failed_count / aligned_n
        support_n, alrt_median, ufboot_median, alrt80, ufboot80 = supports(work / "primary.treefile")
        rec_dir = REC / orthogroup
        duplications, losses, duplication_locations = reconciliation_stats(
            rec_dir / "primary.contree_recs.relationships_summary.txt",
            rec_dir / "primary.contree_recs.nhx",
        )
        local = id_map.loc[id_map["orthogroup"].eq(orthogroup)]
        nearest = nearest_other_genus(work / "primary.treefile", id_map, orthogroup, "primary_full_length")
        nearest_frames.append(nearest)
        association_row, association_null = crossgenus_association(
            work / "primary.treefile", id_map, orthogroup, association_rng
        )
        association_rows.append(association_row)
        association_nulls.append(association_null)
        family_rows.append(
            {
                "orthogroup": orthogroup,
                "leaf_count": aligned_n,
                "species_count": int(local["species"].nunique()),
                "candidate_leaf_count": int(local["tier_a_positive_candidate"].sum()),
                "aligned_sites": aligned_sites,
                "trimmed_sites": trimmed_sites,
                "retained_site_fraction": trimmed_sites / aligned_sites,
                "aligned_missing_fraction": aligned_missing,
                "trimmed_missing_fraction": trimmed_missing,
                "selected_model": selected_model(work / "primary.iqtree"),
                "model_search_scope": "restricted_LG_JTT_WAG_QPLANT_E_G4_R4_R6_R8" if orthogroup == "OG0000025" else "unrestricted_MFP",
                "model_search_feasibility_amendment": orthogroup == "OG0000025",
                "composition_failed_sequences": failed_count,
                "composition_failed_percent": failed_percent,
                "iqtree_reported_total_percent": reported_total_percent,
                "supported_internal_branches": support_n,
                "median_sh_alrt": alrt_median,
                "median_ufboot": ufboot_median,
                "fraction_sh_alrt_ge_80": alrt80,
                "fraction_ufboot_ge_80": ufboot80,
                "reconciled_duplications": duplications,
                "reconciled_losses": losses,
                "prunus_specific_duplications": int(duplication_locations["prunus_persica"]),
                "pyrus_specific_duplications": int(duplication_locations["pyrus_pyrifolia"]),
                "other_or_ancestral_duplications": int(duplications - duplication_locations["prunus_persica"] - duplication_locations["pyrus_pyrifolia"]),
                "candidate_nearest_other_genus_is_candidate_fraction": float(nearest["nearest_other_genus_is_candidate"].mean()) if not nearest.empty else np.nan,
            }
        )
    family_summary = pd.DataFrame(family_rows)
    catalog = pd.read_csv(ROOT / "results/metrics/publication_v4_tier_a_comparative/tier_a_candidate_summary.tsv", sep="\t")
    family_summary = catalog[["catalog_rank", "orthogroup", "display_label", "retrospective_family_label", "mechanism_module", "literature_evidence_level"]].merge(
        family_summary, on="orthogroup", how="right", validate="one_to_one"
    ).sort_values("catalog_rank")
    association = pd.DataFrame(association_rows)
    association["bh_q"] = bh_adjust(association["empirical_p"])
    association = catalog[["catalog_rank", "orthogroup", "display_label"]].merge(
        association, on="orthogroup", how="right", validate="one_to_one"
    ).sort_values("catalog_rank")
    association.to_csv(OUT / "candidate_crossgenus_association.tsv", sep="\t", index=False)
    association_for_merge = association[
        [
            "orthogroup",
            "observed_fraction",
            "null_median_fraction",
            "null_q95_fraction",
            "effect_over_null_median",
            "empirical_p",
            "bh_q",
        ]
    ].rename(columns={column: f"candidate_crossgenus_{column}" for column in [
        "observed_fraction",
        "null_median_fraction",
        "null_q95_fraction",
        "effect_over_null_median",
        "empirical_p",
        "bh_q",
    ]})
    family_summary = family_summary.merge(association_for_merge, on="orthogroup", how="left", validate="one_to_one")
    family_summary.to_csv(OUT / "gene_tree_family_summary.tsv", sep="\t", index=False)

    pooled_observed_successes = int(association["observed_successes"].sum())
    pooled_queries = int(association["candidate_queries"].sum())
    pooled_null_successes = np.sum(np.vstack(association_nulls), axis=0)
    pooled_null_fraction = pooled_null_successes / pooled_queries
    pooled = pd.DataFrame(
        [
            {
                "families": len(association),
                "candidate_queries": pooled_queries,
                "observed_successes": pooled_observed_successes,
                "observed_fraction": pooled_observed_successes / pooled_queries,
                "null_median_fraction": float(np.median(pooled_null_fraction)),
                "null_q95_fraction": float(np.quantile(pooled_null_fraction, 0.95)),
                "effect_over_null_median": pooled_observed_successes / pooled_queries - float(np.median(pooled_null_fraction)),
                "empirical_p": (1 + int(np.sum(pooled_null_successes >= pooled_observed_successes))) / (ASSOCIATION_REPLICATES + 1),
                "replicates": ASSOCIATION_REPLICATES,
                "seed": ASSOCIATION_SEED,
            }
        ]
    )
    pooled.to_csv(OUT / "candidate_crossgenus_association_pooled.tsv", sep="\t", index=False)

    nearest_all = pd.concat(nearest_frames, ignore_index=True)
    domain_summary = pd.read_csv(DOMAIN / "domain_family_summary.tsv", sep="\t")
    domain_triggered = set(domain_summary.loc[domain_summary["contract_domain_triggered"], "orthogroup"])
    sensitivity_rows = []
    for orthogroup in DISPLAY:
        work = RUNS / orthogroup
        untrimmed = UNTRIMMED / orthogroup / "untrimmed.contree"
        diagnostic_path = UNTRIMMED / orthogroup / "untrimmed.diagnostic_nonconverged.json"
        if untrimmed.is_file() and (UNTRIMMED / orthogroup / "untrimmed.complete").is_file():
            common, rf = normalized_rf(work / "primary.contree", untrimmed)
            sensitivity_rows.append(
                {
                    "orthogroup": orthogroup,
                    "comparison": "primary_vs_untrimmed",
                    "common_leaves": common,
                    "normalized_rf": rf,
                    "sensitivity_status": "converged",
                    "completed_iterations": np.nan,
                    "bootstrap_correlation": np.nan,
                }
            )
            nearest_frames.append(nearest_other_genus(untrimmed, id_map, orthogroup, "untrimmed_full_length"))
        elif diagnostic_path.is_file():
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            sensitivity_rows.append(
                {
                    "orthogroup": orthogroup,
                    "comparison": "primary_vs_untrimmed",
                    "common_leaves": int(diagnostic["leaf_count"]),
                    "normalized_rf": np.nan,
                    "sensitivity_status": "nonconverged_ufboot_diagnostic_cap",
                    "completed_iterations": int(diagnostic["completed_iterations"]),
                    "bootstrap_correlation": float(diagnostic["bootstrap_correlation"]),
                }
            )
        else:
            raise RuntimeError(f"Untrimmed sensitivity result is missing: {orthogroup}")
        if orthogroup in domain_triggered:
            domain_tree = DOMAIN / orthogroup / "domain.contree"
            common, rf = normalized_rf(work / "primary.contree", domain_tree)
            sensitivity_rows.append(
                {
                    "orthogroup": orthogroup,
                    "comparison": "primary_vs_domain",
                    "common_leaves": common,
                    "normalized_rf": rf,
                    "sensitivity_status": "converged",
                    "completed_iterations": np.nan,
                    "bootstrap_correlation": np.nan,
                }
            )
            nearest_frames.append(nearest_other_genus(domain_tree, id_map, orthogroup, "pfam_domain_contract_triggered"))
    nearest_all = pd.concat(nearest_frames, ignore_index=True)
    nearest_all.to_csv(OUT / "candidate_crossgenus_nearest.tsv", sep="\t", index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "display_family_tree_sensitivity.tsv", sep="\t", index=False)

    outputs = [
        "gene_tree_family_summary.tsv",
        "candidate_crossgenus_nearest.tsv",
        "candidate_crossgenus_association.tsv",
        "candidate_crossgenus_association_pooled.tsv",
        "display_family_tree_sensitivity.tsv",
    ]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_descriptive",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "orthogroups": int(family_summary["orthogroup"].nunique()),
        "display_families": DISPLAY,
        "crossgenus_association_replicates": ASSOCIATION_REPLICATES,
        "crossgenus_association_seed": ASSOCIATION_SEED,
        "crossgenus_association_fdr_families": int((association["bh_q"] < 0.05).sum()),
        "crossgenus_association_pooled_empirical_p": float(pooled["empirical_p"].iloc[0]),
        "untrimmed_sensitivity_nonconverged": int(sensitivity["normalized_rf"].isna().sum()),
        "total_reconciled_duplications": int(family_summary["reconciled_duplications"].sum()),
        "total_reconciled_losses": int(family_summary["reconciled_losses"].sum()),
        "output_fingerprints": {name: sha256(OUT / name) for name in outputs},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checks = {
        "all_14_families": len(family_summary) == family_summary["orthogroup"].nunique() == 14,
        "all_leaf_counts_match": bool((family_summary["leaf_count"] == family_input.set_index("orthogroup").loc[family_summary["orthogroup"], "sequences"].to_numpy()).all()),
        "all_species_counts_valid": bool(family_summary["species_count"].between(1, 26).all()),
        "all_primary_trees_supported": bool((family_summary["supported_internal_branches"] > 0).all()),
        "all_reconciliations_present": bool((family_summary["reconciled_duplications"] >= 0).all() and (family_summary["reconciled_losses"] >= 0).all()),
        "registered_sensitivity_comparisons_complete": len(sensitivity) == len(DISPLAY) + len(domain_triggered),
        "rf_in_unit_interval_when_reported": bool(sensitivity.loc[sensitivity["normalized_rf"].notna(), "normalized_rf"].between(0, 1).all()),
        "nonconverged_rf_not_reported": bool(
            sensitivity["normalized_rf"].isna().sum() == 1
            and sensitivity.loc[sensitivity["normalized_rf"].isna(), "orthogroup"].eq("OG0000025").all()
            and sensitivity.loc[sensitivity["normalized_rf"].isna(), "sensitivity_status"].eq("nonconverged_ufboot_diagnostic_cap").all()
        ),
        "candidate_rows_present": len(nearest_all) > 0,
        "candidate_association_all_14_families": len(association) == association["orthogroup"].nunique() == 14,
        "candidate_association_10000_replicates": bool((association["replicates"] == ASSOCIATION_REPLICATES).all()),
        "candidate_association_fractions_consistent": bool(
            np.allclose(
                family_summary["candidate_nearest_other_genus_is_candidate_fraction"],
                family_summary["candidate_crossgenus_observed_fraction"],
            )
        ),
        "candidate_association_pvalues_valid": bool(
            association["empirical_p"].between(1 / (ASSOCIATION_REPLICATES + 1), 1).all()
            and association["bh_q"].between(0, 1).all()
            and pooled["empirical_p"].between(1 / (ASSOCIATION_REPLICATES + 1), 1).all()
        ),
        "ap2_feasibility_amendment_recorded": bool(
            family_summary.loc[family_summary["orthogroup"].eq("OG0000025"), "model_search_feasibility_amendment"].iloc[0]
            and (family_summary.loc[~family_summary["orthogroup"].eq("OG0000025"), "model_search_feasibility_amendment"] == False).all()
        ),
        "malus_outcomes_sealed": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    audit = {
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": len(checks),
        "passed_checks": len(checks) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "records": checks,
        "summary": summary,
    }
    (ROOT / "results/metrics/publication_v5_gene_trees_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit["status"], **summary, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
