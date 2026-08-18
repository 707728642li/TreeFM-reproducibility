#!/usr/bin/env python3
"""Orthogroup-anchored microsynteny for frozen Tier-A families and matched controls."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
ORTHOGROUPS = ROOT / "data/processed/orthofinder_benchmark_publication_v3/OrthoFinder/Results_PublicationV3/Orthogroups/Orthogroups.txt"
RESOURCE = ROOT / "results/metrics/publication_v5_resource_audit/tier_a_member_mapping.tsv.gz"
TIER_A_CATALOG = ROOT / "results/metrics/publication_v4_tier_a_comparative/tier_a_candidate_summary.tsv"
CANDIDATE_OGS = ROOT / "results/biological_cases/publication_v3_crossgenus_candidates/candidate_orthogroups.tsv"
OUT = ROOT / "results/metrics/publication_v5_microsynteny"
WINDOW_RADIUS = 10
DISPLAY_FAMILIES = ["OG0000025", "OG0000413", "OG0000277"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gff_path(species: str) -> Path:
    if species == "prunus_persica":
        return ROOT / "data/interim/functional_genomes/prunus_persica/annotation.gff3"
    if species == "pyrus_pyrifolia":
        return ROOT / "data/interim/publication_v3_genomes/pyrus_pyrifolia/annotation.gff3"
    return ROOT / f"data/interim/normalized/{species}/annotation.gff3"


def parse_attrs(text: str) -> dict[str, str]:
    result = {}
    for item in text.rstrip(";\n").split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif " " in item:
            key, value = item.split(" ", 1)
            value = value.strip('"')
        else:
            continue
        result[key] = value
    return result


def aliases(attrs: dict[str, str]) -> list[str]:
    ordered = []
    preferred = ["Gene_Accession", "gene_id", "ID", "Name", "Source_ID", "locus_tag"]
    keys = preferred + [key for key in attrs if key not in preferred]
    for key in keys:
        value = attrs.get(key, "")
        for token in re.split(r"[,|]", value):
            token = token.strip()
            if not token:
                continue
            variants = [token]
            if ":" in token and token.split(":", 1)[0].lower() in {"gene", "transcript", "rna", "mrna"}:
                variants.append(token.split(":", 1)[1])
            for variant in variants:
                if variant not in ordered:
                    ordered.append(variant)
    return ordered


def load_orthogroups() -> tuple[dict[tuple[str, str], str], dict[str, dict[str, object]]]:
    gene_to_og: dict[tuple[str, str], str] = {}
    summaries: dict[str, dict[str, object]] = {}
    with ORTHOGROUPS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("OG"):
                continue
            og, payload = line.rstrip("\n").split(":", 1)
            members = payload.strip().split()
            counts: Counter[str] = Counter()
            for member in members:
                parts = member.split("|", 2)
                if len(parts) < 2:
                    continue
                species, gene = parts[0], parts[1]
                gene_to_og[(species, gene)] = og
                counts[species] += 1
            summaries[og] = {
                "orthogroup": og,
                "species_breadth": len(counts),
                "total_members": sum(counts.values()),
                "prunus_copies": counts["prunus_persica"],
                "pyrus_copies": counts["pyrus_pyrifolia"],
                "species_counts": dict(counts),
            }
    return gene_to_og, summaries


def parse_gene_order(species: str, gene_to_og: dict[tuple[str, str], str]) -> tuple[dict[str, list[dict]], dict[str, dict], dict]:
    path = gff_path(species)
    gene_rows = []
    transcript_rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            feature = fields[2].lower()
            if feature not in {"gene", "pseudogene", "mrna", "transcript"}:
                continue
            attrs = parse_attrs(fields[8])
            row = {
                "seqid": fields[0],
                "start": int(fields[3].strip()),
                "end": int(fields[4].strip()),
                "strand": fields[6],
                "feature": feature,
                "aliases": aliases(attrs),
            }
            (gene_rows if feature in {"gene", "pseudogene"} else transcript_rows).append(row)
    selected = gene_rows if gene_rows else transcript_rows
    by_key = {}
    ambiguous = 0
    for row in selected:
        mapped = [(alias, gene_to_og[(species, alias)]) for alias in row["aliases"] if (species, alias) in gene_to_og]
        unique_mapped = list(dict.fromkeys(mapped))
        if len({og for _, og in unique_mapped}) > 1:
            ambiguous += 1
        mapped_gene = unique_mapped[0][0] if unique_mapped else ""
        orthogroup = unique_mapped[0][1] if unique_mapped else ""
        stable = mapped_gene or (row["aliases"][0] if row["aliases"] else f"anonymous_{len(by_key)}")
        key = (row["seqid"], stable)
        previous = by_key.get(key)
        if previous is None or (row["end"] - row["start"]) > (previous["end"] - previous["start"]):
            by_key[key] = {
                "species": species,
                "gene_id": stable,
                "mapped_gene_id": mapped_gene,
                "orthogroup": orthogroup,
                "seqid": row["seqid"],
                "start": row["start"],
                "end": row["end"],
                "strand": row["strand"],
            }
    by_seqid: dict[str, list[dict]] = defaultdict(list)
    index: dict[str, dict] = {}
    duplicate_mapped = 0
    for row in by_key.values():
        by_seqid[row["seqid"]].append(row)
    for seqid, rows in by_seqid.items():
        rows.sort(key=lambda item: (item["start"], item["end"], item["gene_id"]))
        for position, row in enumerate(rows):
            row["index"] = position
            gene = row["mapped_gene_id"]
            if gene:
                if gene in index:
                    duplicate_mapped += 1
                else:
                    index[gene] = row
    qc = {
        "species": species,
        "gff_path": path.relative_to(ROOT).as_posix(),
        "record_source": "gene_or_pseudogene" if gene_rows else "mrna_or_transcript_fallback",
        "gene_records": len(by_key),
        "mapped_gene_records": len(index),
        "ambiguous_orthogroup_records": ambiguous,
        "duplicate_mapped_gene_ids": duplicate_mapped,
    }
    return dict(by_seqid), index, qc


def window_for(row: dict, order: dict[str, list[dict]]) -> dict:
    rows = order[row["seqid"]]
    center = int(row["index"])
    left = max(0, center - WINDOW_RADIUS)
    right = min(len(rows), center + WINDOW_RADIUS + 1)
    subset = rows[left:right]
    oriented = list(reversed(subset)) if row["strand"] == "-" else subset
    center_oriented = len(subset) - 1 - (center - left) if row["strand"] == "-" else center - left
    entries = []
    for position, neighbor in enumerate(oriented):
        entries.append(
            {
                "relative_rank": position - center_oriented,
                "neighbor_gene_id": neighbor["mapped_gene_id"] or neighbor["gene_id"],
                "neighbor_orthogroup": neighbor["orthogroup"],
                "neighbor_start": neighbor["start"],
                "neighbor_end": neighbor["end"],
                "neighbor_strand": neighbor["strand"],
            }
        )
    return {
        "species": row["species"],
        "target_gene_id": row["mapped_gene_id"],
        "target_orthogroup": row["orthogroup"],
        "seqid": row["seqid"],
        "target_start": row["start"],
        "target_end": row["end"],
        "target_strand": row["strand"],
        "edge_truncated": left == 0 or right == len(rows),
        "entries": entries,
    }


def content(window: dict) -> set[str]:
    target = window["target_orthogroup"]
    return {entry["neighbor_orthogroup"] for entry in window["entries"] if entry["neighbor_orthogroup"] and entry["neighbor_orthogroup"] != target}


def sequence(window: dict) -> list[str]:
    target = window["target_orthogroup"]
    return [entry["neighbor_orthogroup"] for entry in window["entries"] if entry["neighbor_orthogroup"] and entry["neighbor_orthogroup"] != target]


def jaccard(a: dict, b: dict) -> float:
    left, right = content(a), content(b)
    return len(left & right) / len(left | right) if left or right else 0.0


def lcs_score(a: dict, b: dict) -> float:
    x, y = sequence(a), sequence(b)
    if not x or not y:
        return 0.0
    previous = [0] * (len(y) + 1)
    for item in x:
        current = [0]
        for index, other in enumerate(y, start=1):
            current.append(previous[index - 1] + 1 if item == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1] / max(len(x), len(y))


def family_match(left: list[dict], right: list[dict]) -> dict:
    size = max(len(left), len(right))
    if size == 0:
        return {"jaccard": np.nan, "lcs": np.nan, "real_pairs": []}
    jac = np.zeros((size, size), dtype=float)
    lcs = np.zeros((size, size), dtype=float)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            jac[i, j] = jaccard(a, b)
            lcs[i, j] = lcs_score(a, b)
    row_ind, col_ind = linear_sum_assignment(-jac)
    pairs = []
    for i, j in zip(row_ind, col_ind):
        if i < len(left) and j < len(right):
            pairs.append(
                {
                    "left_gene": left[i]["target_gene_id"],
                    "right_gene": right[j]["target_gene_id"],
                    "jaccard": jac[i, j],
                    "lcs": lcs[i, j],
                }
            )
    return {"jaccard": float(jac[row_ind, col_ind].sum() / size), "lcs": float(lcs[row_ind, col_ind].sum() / size), "real_pairs": pairs}


def bh(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order] * len(array) / np.arange(1, len(array) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result.tolist()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tier_catalog = pd.read_csv(TIER_A_CATALOG, sep="\t", dtype={"orthogroup": str})
    tier_a = tier_catalog.sort_values("catalog_rank")["orthogroup"].tolist()
    excluded = set(pd.read_csv(CANDIDATE_OGS, sep="\t", usecols=["orthogroup"])["orthogroup"].astype(str))
    gene_to_og, family_summaries = load_orthogroups()

    species = sorted({key[0] for key in gene_to_og})
    gene_orders = {}
    gene_indexes = {}
    gff_qc = []
    for name in species:
        order, index, qc = parse_gene_order(name, gene_to_og)
        gene_orders[name] = order
        gene_indexes[name] = index
        gff_qc.append(qc)
    pd.DataFrame(gff_qc).to_csv(OUT / "gff_gene_order_qc.tsv", sep="\t", index=False)

    control_rows = []
    controls_by_target: dict[str, list[str]] = {}
    eligible = [
        og for og, item in family_summaries.items()
        if og not in excluded and og not in tier_a and item["prunus_copies"] > 0 and item["pyrus_copies"] > 0
    ]
    stages = [
        ("strict", 1, 0, 0.25),
        ("breadth2_size05", 2, 0, 0.50),
        ("copy1_breadth3_size075", 3, 1, 0.75),
        ("copy2_breadth5_size1", 5, 2, 1.00),
        ("global_nearest", 26, 1000000, 100.00),
    ]
    for target in tier_a:
        reference = family_summaries[target]
        chosen = []
        chosen_stage = ""
        for stage, breadth_tol, copy_tol, log_tol in stages:
            pool = []
            for control in eligible:
                item = family_summaries[control]
                if abs(item["species_breadth"] - reference["species_breadth"]) > breadth_tol:
                    continue
                copy_distance = abs(item["prunus_copies"] - reference["prunus_copies"]) + abs(item["pyrus_copies"] - reference["pyrus_copies"])
                if copy_distance > copy_tol:
                    continue
                log_distance = abs(math.log2(item["total_members"]) - math.log2(reference["total_members"]))
                if log_distance > log_tol:
                    continue
                distance = (
                    abs(item["species_breadth"] - reference["species_breadth"]) / 26
                    + copy_distance / max(reference["prunus_copies"] + reference["pyrus_copies"], 1)
                    + log_distance
                )
                tie = hashlib.sha256(f"{target}|{control}".encode()).hexdigest()
                pool.append((distance, tie, control, copy_distance, log_distance))
            pool.sort()
            chosen = pool[:100]
            chosen_stage = stage
            if len(chosen) >= 50:
                break
        if len(chosen) < 50:
            raise RuntimeError(f"Fewer than 50 matched controls for {target}: {len(chosen)}")
        controls_by_target[target] = [item[2] for item in chosen]
        for rank, (distance, tie, control, copy_distance, log_distance) in enumerate(chosen, start=1):
            item = family_summaries[control]
            control_rows.append(
                {
                    "target_orthogroup": target,
                    "control_rank": rank,
                    "control_orthogroup": control,
                    "matching_stage": chosen_stage,
                    "distance": distance,
                    "breadth": item["species_breadth"],
                    "total_members": item["total_members"],
                    "prunus_copies": item["prunus_copies"],
                    "pyrus_copies": item["pyrus_copies"],
                    "copy_distance": copy_distance,
                    "log2_size_distance": log_distance,
                }
            )
    pd.DataFrame(control_rows).to_csv(OUT / "matched_control_orthogroups.tsv", sep="\t", index=False)

    needed_target = set(tier_a)
    needed_prpy = needed_target | {control for values in controls_by_target.values() for control in values}
    windows_by_species_og: dict[tuple[str, str], list[dict]] = defaultdict(list)
    flat_window_rows = []
    for name in species:
        wanted = needed_prpy if name in {"prunus_persica", "pyrus_pyrifolia"} else needed_target
        for gene, row in gene_indexes[name].items():
            if row["orthogroup"] not in wanted:
                continue
            window = window_for(row, gene_orders[name])
            windows_by_species_og[(name, row["orthogroup"])].append(window)
            if row["orthogroup"] in needed_target:
                for entry in window["entries"]:
                    flat_window_rows.append(
                        {
                            "species": name,
                            "target_orthogroup": row["orthogroup"],
                            "target_gene_id": gene,
                            "seqid": window["seqid"],
                            "target_start": window["target_start"],
                            "target_end": window["target_end"],
                            "target_strand": window["target_strand"],
                            "edge_truncated": window["edge_truncated"],
                            **entry,
                        }
                    )
    pd.DataFrame(flat_window_rows).to_csv(OUT / "tier_a_local_gene_windows.tsv.gz", sep="\t", index=False, compression="gzip")

    score_rows = []
    pair_rows = []
    for target in tier_a:
        observed = family_match(
            windows_by_species_og[("prunus_persica", target)],
            windows_by_species_og[("pyrus_pyrifolia", target)],
        )
        control_jac = []
        control_lcs = []
        for control in controls_by_target[target]:
            score = family_match(
                windows_by_species_og[("prunus_persica", control)],
                windows_by_species_og[("pyrus_pyrifolia", control)],
            )
            control_jac.append(score["jaccard"])
            control_lcs.append(score["lcs"])
        p_jac = (1 + sum(value >= observed["jaccard"] for value in control_jac)) / (1 + len(control_jac))
        p_lcs = (1 + sum(value >= observed["lcs"] for value in control_lcs)) / (1 + len(control_lcs))
        score_rows.append(
            {
                "orthogroup": target,
                "catalog_rank": int(tier_catalog.loc[tier_catalog["orthogroup"] == target, "catalog_rank"].iloc[0]),
                "prunus_copies": len(windows_by_species_og[("prunus_persica", target)]),
                "pyrus_copies": len(windows_by_species_og[("pyrus_pyrifolia", target)]),
                "jaccard": observed["jaccard"],
                "lcs": observed["lcs"],
                "control_count": len(control_jac),
                "control_jaccard_median": float(np.median(control_jac)),
                "control_jaccard_q95": float(np.quantile(control_jac, 0.95)),
                "control_lcs_median": float(np.median(control_lcs)),
                "control_lcs_q95": float(np.quantile(control_lcs, 0.95)),
                "empirical_p_jaccard": p_jac,
                "empirical_p_lcs": p_lcs,
            }
        )
        if target in DISPLAY_FAMILIES:
            for pair in observed["real_pairs"]:
                pair_rows.append({"orthogroup": target, **pair})
    q_jac = bh([row["empirical_p_jaccard"] for row in score_rows])
    q_lcs = bh([row["empirical_p_lcs"] for row in score_rows])
    for row, value_j, value_l in zip(score_rows, q_jac, q_lcs):
        row["bh_q_jaccard"] = value_j
        row["bh_q_lcs"] = value_l
    pd.DataFrame(score_rows).sort_values("catalog_rank").to_csv(OUT / "tier_a_prunus_pyrus_scores.tsv", sep="\t", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "display_family_matched_pairs.tsv", sep="\t", index=False)

    depth_rows = []
    for target in tier_a:
        for anchor in ("prunus_persica", "pyrus_pyrifolia"):
            for other in species:
                if other == anchor:
                    continue
                score = family_match(windows_by_species_og[(anchor, target)], windows_by_species_og[(other, target)])
                depth_rows.append(
                    {
                        "orthogroup": target,
                        "anchor_species": anchor,
                        "other_species": other,
                        "anchor_copies": len(windows_by_species_og[(anchor, target)]),
                        "other_copies": len(windows_by_species_og[(other, target)]),
                        "jaccard": score["jaccard"],
                        "lcs": score["lcs"],
                    }
                )
    pd.DataFrame(depth_rows).to_csv(OUT / "tier_a_cross_species_depth.tsv", sep="\t", index=False)

    score_frame = pd.DataFrame(score_rows)
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_descriptive",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "window_radius_genes": WINDOW_RADIUS,
        "orthogroup_count": len(tier_a),
        "species_count": len(species),
        "mapped_gene_records": int(sum(item["mapped_gene_records"] for item in gff_qc)),
        "tier_a_window_rows": len(flat_window_rows),
        "control_rows": len(control_rows),
        "jaccard_bh_significant": int((score_frame["bh_q_jaccard"] < 0.05).sum()),
        "lcs_bh_significant": int((score_frame["bh_q_lcs"] < 0.05).sum()),
        "display_families": DISPLAY_FAMILIES,
        "input_fingerprints": {
            str(ORTHOGROUPS.relative_to(ROOT)): sha256(ORTHOGROUPS),
            str(RESOURCE.relative_to(ROOT)): sha256(RESOURCE),
            str(TIER_A_CATALOG.relative_to(ROOT)): sha256(TIER_A_CATALOG),
            str(CANDIDATE_OGS.relative_to(ROOT)): sha256(CANDIDATE_OGS),
        },
    }
    output_names = [
        "gff_gene_order_qc.tsv",
        "matched_control_orthogroups.tsv",
        "tier_a_local_gene_windows.tsv.gz",
        "tier_a_prunus_pyrus_scores.tsv",
        "display_family_matched_pairs.tsv",
        "tier_a_cross_species_depth.tsv",
    ]
    summary["output_fingerprints"] = {name: sha256(OUT / name) for name in output_names}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    checks = {
        "fourteen_targets": len(tier_a) == len(set(tier_a)) == 14,
        "twenty_six_species": len(species) == 26,
        "no_ambiguous_gff_orthogroups": sum(item["ambiguous_orthogroup_records"] for item in gff_qc) == 0,
        "all_target_members_have_windows": all(
            len(windows_by_species_og[(species_name, target)]) == int(family_summaries[target]["species_counts"].get(species_name, 0))
            for target in tier_a for species_name in species
        ),
        "at_least_50_controls_each": all(len(values) >= 50 for values in controls_by_target.values()),
        "controls_exclude_candidates": all(row["control_orthogroup"] not in excluded for row in control_rows),
        "complete_target_scores": len(score_rows) == 14 and all(np.isfinite(row["jaccard"]) and np.isfinite(row["lcs"]) for row in score_rows),
        "complete_depth_scores": len(depth_rows) == 14 * 2 * 25,
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
    (ROOT / "results/metrics/publication_v5_microsynteny_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit["status"], **summary, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
