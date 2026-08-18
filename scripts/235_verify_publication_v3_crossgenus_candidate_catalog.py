#!/usr/bin/env python3
"""Independent audit of the frozen cross-genus candidate catalog.

This verifier does not import the catalog builder.  It reconstructs mappings,
orthogroup states, GO propagation, motif/GC summaries, priority tiers, order,
and negative-control matching directly from the frozen inputs.
"""

from __future__ import annotations

import argparse
import os
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = {
    "prunus": Path(
        "data/processed/functional/Prunus_publication_v3/"
        "promoter_labels.parquet"
    ),
    "pyrus": Path(
        "data/processed/functional/Pyrus_PRJNA669907/"
        "promoter_labels.parquet"
    ),
}
BRIDGE = Path("metadata/publication_v3_prunus_v21_gene_id_bridge.tsv")
ORTHOGROUPS = Path(
    "data/processed/orthofinder_benchmark_publication_v3/OrthoFinder/"
    "Results_PublicationV3/Orthogroups/Orthogroups.txt"
)
GAF = Path("data/raw/publication_v3_go/ARATH-mod.gaf.gz")
OBO = Path("data/raw/publication_v3_go/go-basic.obo")
LEAF = Path("results/tables/publication_v3_crossgenus_go_leaf_terms.tsv")
CONTRACT = Path(
    "docs/publication_v3_crossgenus_candidate_catalog_contract_v2.md"
)
OUT = Path(
    "results/biological_cases/publication_v3_crossgenus_candidates"
)
CATALOG = OUT / "candidate_orthogroups.tsv"
TIER_A = OUT / "tier_a_candidates.tsv"
CONTROLS = OUT / "matched_negative_controls.tsv"
SUMMARY = OUT / "summary.json"
AUDIT = OUT / "independent_audit.json"
AGI = re.compile(r"(?i)(AT[1-5MC]G\d{5})")
GO_ROOT = "GO:0008150"
MOTIF = "CACGTG"
TIERS = {3: "A", 2: "B", 1: "C", 0: "D"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def parse_obo(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, frozenset[str]]]:
    terms: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None

    def retain(record: dict[str, Any] | None) -> None:
        if record is not None and "id" in record:
            terms[str(record["id"])] = record

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                retain(current)
                current = {"parents": set(), "obsolete": False}
            elif line.startswith("["):
                retain(current)
                current = None
            elif current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:]
                elif line.startswith("is_a: "):
                    current["parents"].add(line[6:].split()[0])
                elif line.startswith("relationship: part_of "):
                    current["parents"].add(line.split()[2])
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
    retain(current)

    @lru_cache(maxsize=None)
    def ancestors(term_id: str) -> frozenset[str]:
        values = {term_id}
        for parent in terms.get(term_id, {}).get("parents", set()):
            values.update(ancestors(str(parent)))
        return frozenset(values)

    return terms, {term_id: ancestors(term_id) for term_id in terms}


def parse_gaf(
    path: Path,
    terms: dict[str, dict[str, Any]],
    ancestors: dict[str, frozenset[str]],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    annotations: dict[str, set[str]] = defaultdict(set)
    symbols: dict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                continue
            match = AGI.search(" ".join(fields[index] for index in (1, 2, 10)))
            if match is None:
                continue
            gene = match.group(1).upper()
            symbol = fields[2].strip()
            if symbol and symbol.upper() != gene:
                symbols[gene].add(symbol)
            if "NOT" in fields[3].split("|") or fields[6] in {"IEA", "ND"}:
                continue
            term_id = fields[4]
            term = terms.get(term_id)
            if (
                term is None
                or term.get("obsolete")
                or term.get("namespace") != "biological_process"
            ):
                continue
            annotations[gene].update(
                ancestor
                for ancestor in ancestors.get(term_id, frozenset({term_id}))
                if ancestor != GO_ROOT
                and ancestor in terms
                and not terms[ancestor].get("obsolete")
                and terms[ancestor].get("namespace") == "biological_process"
            )
    return dict(annotations), {
        gene: ";".join(sorted(values, key=lambda value: value.lower()))
        for gene, values in symbols.items()
    }


def parse_orthogroups(path: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            orthogroup, raw_members = line.rstrip("\n").split(": ", 1)
            members: set[str] = set()
            for token in raw_members.split():
                if not token.startswith("arabidopsis_thaliana|"):
                    continue
                match = AGI.search(token)
                if match is not None:
                    members.add(match.group(1).upper())
            if members:
                groups[orthogroup] = members
    return groups


def motif_count(sequence: object) -> int:
    sequence = str(sequence).upper()
    return sum(
        sequence[index : index + len(MOTIF)] == MOTIF
        for index in range(max(0, len(sequence) - len(MOTIF) + 1))
    )


def promoter_gc(sequence: object) -> float:
    sequence = str(sequence).upper()
    denominator = sum(base in "ACGT" for base in sequence)
    if denominator == 0:
        raise RuntimeError("promoter contains no canonical bases")
    return sum(base in "GC" for base in sequence) / denominator


def parse_common_gene_to_group(
    path: Path, slugs: set[str]
) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if ": " not in line:
                raise RuntimeError(
                    f"malformed Orthogroups line {line_number}"
                )
            orthogroup, members = line.split(": ", 1)
            for member in members.split():
                parts = member.split("|", 2)
                if len(parts) < 2 or parts[0] not in slugs:
                    continue
                key = (parts[0], parts[1])
                previous = output.setdefault(key, orthogroup)
                if previous != orthogroup:
                    raise RuntimeError(f"multiple common groups for {key}")
    return output


def load_genes(
    root: Path,
    genus: str,
    common_gene_to_group: dict[tuple[str, str], str],
) -> pd.DataFrame:
    labels = pd.read_parquet(
        root / LABELS[genus],
        columns=[
            "gene_id",
            "chromosome",
            "promoter_2048",
            "endpoint_direction",
            "label_binary",
        ],
    ).copy()
    labels["functional_gene_id"] = labels["gene_id"].astype(str)
    if genus == "prunus":
        bridge = pd.read_csv(root / BRIDGE, sep="\t", dtype=str)
        if bridge["source_gene_id"].duplicated().any():
            raise RuntimeError("duplicate Prunus bridge source identifier")
        bridge = bridge.set_index("source_gene_id")["technical_gene_id"]
        labels["technical_gene_id"] = labels["functional_gene_id"].map(bridge)
        slug = "prunus_persica"
    else:
        labels["technical_gene_id"] = labels["functional_gene_id"]
        slug = "pyrus_pyrifolia"
    labels["orthogroup"] = labels["technical_gene_id"].map(
        lambda gene_id: (
            common_gene_to_group.get((slug, str(gene_id)), "")
            if pd.notna(gene_id)
            else ""
        )
    )
    labels["orthogroup"] = labels["orthogroup"].fillna("").astype(str)
    labels["label_binary"] = labels["label_binary"].astype(np.int64)
    labels["direction"] = (
        labels["endpoint_direction"].fillna("").astype(str).str.strip().str.lower()
    )
    labels.loc[
        labels["direction"].isin(
            ["", "none", "nan", "na", "n/a", "null", "unknown"]
        ),
        "direction",
    ] = ""
    labels["gbox_count"] = labels["promoter_2048"].map(motif_count)
    labels["gc"] = labels["promoter_2048"].map(promoter_gc)
    return labels


def collapse(genes: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    mapped = genes.loc[genes["orthogroup"].ne("")]
    for orthogroup, group in mapped.groupby("orthogroup", sort=True):
        positives = group.loc[group["label_binary"].eq(1)]
        positive_count = int(len(positives))
        negative_count = int(len(group) - positive_count)
        if positive_count and not negative_count:
            state = "positive_only"
        elif negative_count and not positive_count:
            state = "negative_only"
        else:
            state = "mixed"
        directions = sorted(
            value for value in positives["direction"].unique() if value
        )
        output[orthogroup] = {
            "state": state,
            "labeled_gene_count": int(len(group)),
            "positive_gene_count": positive_count,
            "negative_gene_count": negative_count,
            "gene_ids": ";".join(
                sorted(group["functional_gene_id"].astype(str))
            ),
            "directions": ";".join(directions),
            "direction_unique_non_none": bool(
                positive_count
                and positives["direction"].ne("").all()
                and len(directions) == 1
            ),
            "gbox_gene_count": int(group["gbox_count"].gt(0).sum()),
            "gbox_total_count": int(group["gbox_count"].sum()),
            "gbox_promoter_fraction": float(group["gbox_count"].gt(0).mean()),
            "gbox_present": bool(group["gbox_count"].gt(0).any()),
            "mean_promoter_gc": float(group["gc"].mean()),
            "chromosomes": ";".join(
                sorted(group["chromosome"].astype(str).unique())
            ),
        }
    return output


def stratum(value: int) -> str:
    return "1" if value == 1 else "2" if value == 2 else "3+"


def reconstruct(
    groups: dict[str, dict[str, dict[str, Any]]],
    ath: dict[str, set[str]],
    annotations: dict[str, set[str]],
    symbols: dict[str, str],
    leaf_names: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    shared = set(groups["prunus"]) & set(groups["pyrus"]) & set(ath)
    positive = sorted(
        og
        for og in shared
        if groups["prunus"][og]["state"] == "positive_only"
        and groups["pyrus"][og]["state"] == "positive_only"
    )
    negative = sorted(
        og
        for og in shared
        if groups["prunus"][og]["state"] == "negative_only"
        and groups["pyrus"][og]["state"] == "negative_only"
    )
    records: list[dict[str, Any]] = []
    for og in positive:
        prunus = groups["prunus"][og]
        pyrus = groups["pyrus"][og]
        leaf = sorted(set(annotations.get(og, set())) & set(leaf_names))
        direction = bool(
            prunus["direction_unique_non_none"]
            and pyrus["direction_unique_non_none"]
            and prunus["directions"] == pyrus["directions"]
        )
        both_gbox = bool(prunus["gbox_present"] and pyrus["gbox_present"])
        count = int(bool(leaf)) + int(direction) + int(both_gbox)
        ath_ids = sorted(ath[og])
        record: dict[str, Any] = {
            "orthogroup": og,
            "tier": TIERS[count],
            "component_count": count,
            "leaf_go": bool(leaf),
            "direction_conserved": direction,
            "gbox_both_genera": both_gbox,
            "leaf_go_term_count": len(leaf),
            "leaf_go_term_ids": ";".join(leaf),
            "leaf_go_term_names": ";".join(leaf_names[value] for value in leaf),
            "arabidopsis_gene_ids": ";".join(ath_ids),
            "arabidopsis_symbols": ";".join(
                f"{gene}:{symbols.get(gene, '')}".rstrip(":")
                for gene in ath_ids
            ),
            "min_genus_gbox_promoter_fraction": min(
                prunus["gbox_promoter_fraction"],
                pyrus["gbox_promoter_fraction"],
            ),
        }
        for genus, values in (("prunus", prunus), ("pyrus", pyrus)):
            for key, value in values.items():
                record[f"{genus}_{key}"] = value
        records.append(record)
    records.sort(
        key=lambda row: (
            row["tier"],
            -row["leaf_go_term_count"],
            -row["min_genus_gbox_promoter_fraction"],
            row["orthogroup"],
        )
    )
    for index, record in enumerate(records, start=1):
        record["catalog_rank"] = index

    matches: list[dict[str, Any]] = []
    used: set[str] = set()
    for candidate in (row for row in records if row["tier"] == "A"):
        strata = (
            stratum(candidate["prunus_labeled_gene_count"]),
            stratum(candidate["pyrus_labeled_gene_count"]),
        )
        pattern = (
            candidate["prunus_gbox_present"],
            candidate["pyrus_gbox_present"],
        )
        eligible: list[tuple[float, str]] = []
        for control in negative:
            if control in used:
                continue
            prunus = groups["prunus"][control]
            pyrus = groups["pyrus"][control]
            if (
                stratum(prunus["labeled_gene_count"]),
                stratum(pyrus["labeled_gene_count"]),
            ) != strata:
                continue
            if (prunus["gbox_present"], pyrus["gbox_present"]) != pattern:
                continue
            distance = abs(
                candidate["prunus_mean_promoter_gc"]
                - prunus["mean_promoter_gc"]
            ) + abs(
                candidate["pyrus_mean_promoter_gc"]
                - pyrus["mean_promoter_gc"]
            )
            eligible.append((distance, control))
        eligible.sort()
        selected = eligible[0] if eligible else None
        if selected:
            used.add(selected[1])
        matches.append(
            {
                "candidate_orthogroup": candidate["orthogroup"],
                "candidate_prunus_gene_count_stratum": strata[0],
                "candidate_pyrus_gene_count_stratum": strata[1],
                "candidate_prunus_gbox_present": pattern[0],
                "candidate_pyrus_gbox_present": pattern[1],
                "eligible_controls_before_selection": len(eligible),
                "matched": bool(selected),
                "control_orthogroup": selected[1] if selected else "",
                "gc_distance": selected[0] if selected else float("nan"),
            }
        )
    selected_by_candidate = {
        match["candidate_orthogroup"]: match["control_orthogroup"]
        for match in matches
    }
    for record in records:
        record["matched_control_orthogroup"] = selected_by_candidate.get(
            record["orthogroup"], ""
        )
    counts = {
        "shared": len(shared),
        "positive": len(positive),
        "negative": len(negative),
    }
    return records, matches, counts


def equal_value(observed: object, expected: object) -> bool:
    if isinstance(expected, (bool, np.bool_)):
        if isinstance(observed, str):
            return observed.strip().lower() == str(bool(expected)).lower()
        return bool(observed) == bool(expected)
    if isinstance(expected, (float, np.floating)):
        if math.isnan(float(expected)):
            return pd.isna(observed) or str(observed).strip() == ""
        return bool(np.isclose(float(observed), float(expected), atol=1e-12, rtol=1e-10))
    if isinstance(expected, (int, np.integer)):
        return int(observed) == int(expected)
    observed_text = "" if pd.isna(observed) else str(observed)
    return observed_text == str(expected)


def compare_records(
    observed: pd.DataFrame,
    expected: list[dict[str, Any]],
    key: str,
    columns: list[str],
) -> list[str]:
    failures: list[str] = []
    if len(observed) != len(expected):
        failures.append(f"{key}: row count {len(observed)} != {len(expected)}")
        return failures
    if observed[key].tolist() != [record[key] for record in expected]:
        failures.append(f"{key}: row order or membership differs")
    observed_index = observed.set_index(key, drop=False)
    for record in expected:
        identifier = record[key]
        if identifier not in observed_index.index:
            continue
        row = observed_index.loc[identifier]
        for column in columns:
            if column not in observed.columns:
                failures.append(f"{identifier}: missing column {column}")
            elif not equal_value(row[column], record[column]):
                failures.append(
                    f"{identifier}.{column}: {row[column]!r} != {record[column]!r}"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("TREEFM_ROOT", Path(__file__).resolve().parents[1])),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    required = [
        *LABELS.values(),
        BRIDGE,
        ORTHOGROUPS,
        GAF,
        OBO,
        LEAF,
        CONTRACT,
        CATALOG,
        TIER_A,
        CONTROLS,
        SUMMARY,
    ]
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("missing audit inputs: " + ", ".join(missing))

    terms, ancestors = parse_obo(root / OBO)
    gene_terms, symbols = parse_gaf(root / GAF, terms, ancestors)
    ath = parse_orthogroups(root / ORTHOGROUPS)
    og_annotations = {
        og: set().union(*(gene_terms.get(gene, set()) for gene in genes))
        for og, genes in ath.items()
    }
    leaf = pd.read_csv(root / LEAF, sep="\t", dtype=str)
    leaf_names = dict(zip(leaf["term_id"], leaf["term_name"]))
    common_gene_to_group = parse_common_gene_to_group(
        root / ORTHOGROUPS,
        {"prunus_persica", "pyrus_pyrifolia"},
    )
    genes = {
        genus: load_genes(root, genus, common_gene_to_group)
        for genus in ("prunus", "pyrus")
    }
    groups = {genus: collapse(table) for genus, table in genes.items()}
    expected, expected_matches, counts = reconstruct(
        groups, ath, og_annotations, symbols, leaf_names
    )

    catalog = pd.read_csv(root / CATALOG, sep="\t")
    tier_a = pd.read_csv(root / TIER_A, sep="\t")
    controls = pd.read_csv(root / CONTROLS, sep="\t", keep_default_na=False)
    summary = json.loads((root / SUMMARY).read_text(encoding="utf-8"))

    candidate_columns = list(expected[0])
    failures = compare_records(
        catalog, expected, "orthogroup", candidate_columns
    )
    expected_tier_a = [record for record in expected if record["tier"] == "A"]
    failures.extend(
        compare_records(
            tier_a, expected_tier_a, "orthogroup", candidate_columns
        )
    )
    match_columns = list(expected_matches[0]) if expected_matches else []
    failures.extend(
        compare_records(
            controls,
            expected_matches,
            "candidate_orthogroup",
            match_columns,
        )
    )

    expected_tiers = Counter(record["tier"] for record in expected)
    summary_checks = {
        "summary_status_pass": summary.get("status") == "pass",
        "summary_gate_status_pass": summary.get("gate_status") == "pass",
        "summary_failed_gate_count_zero": summary.get("failed_gate_count") == 0,
        "summary_model_outputs_false": summary.get("model_outputs_accessed") is False,
        "summary_malus_false": summary.get("malus_accessed") is False,
        "contract_hash_exact": summary.get("contract_sha256")
        == sha256(root / CONTRACT),
        "candidate_count_exact": summary.get("candidate_orthogroups")
        == len(expected),
        "tier_counts_exact": summary.get("candidate_counts_by_tier")
        == {tier: expected_tiers.get(tier, 0) for tier in "ABCD"},
        "shared_population_exact": summary.get(
            "shared_labeled_orthogroups_with_arabidopsis"
        )
        == counts["shared"],
        "positive_population_exact": summary.get(
            "shared_positive_only_orthogroups"
        )
        == counts["positive"],
        "negative_pool_exact": summary.get(
            "shared_negative_only_control_pool"
        )
        == counts["negative"],
        "matched_count_exact": summary.get("matched_controls")
        == sum(record["matched"] for record in expected_matches),
        "input_hashes_exact": all(
            (root / path).is_file() and sha256(root / path) == digest
            for path, digest in summary.get("input_sha256", {}).items()
        ),
        "no_model_or_malus_paths_in_manifest": all(
            "model" not in path.lower() and "malus" not in path.lower()
            for path in summary.get("input_sha256", {})
        ),
        "single_common_26species_namespace_reconstructed": bool(
            summary.get("orthogroup_namespace", {}).get(
                "both_target_genera_mapped_directly"
            )
            and summary.get("orthogroup_namespace", {}).get(
                "arabidopsis_anchors_from_same_file"
            )
        ),
    }
    failures.extend(
        key for key, passed in summary_checks.items() if not passed
    )
    audit = {
        "status": "pass" if not failures else "fail",
        "scope": (
            "independent_crossgenus_candidate_catalog_audit_"
            "common_26species_namespace"
        ),
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_independent_of_builder": True,
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "reconstructed_candidate_orthogroups": len(expected),
        "reconstructed_tier_counts": {
            tier: expected_tiers.get(tier, 0) for tier in "ABCD"
        },
        "reconstructed_control_pool": counts["negative"],
        "reconstructed_match_records": len(expected_matches),
        "reconstructed_matched_controls": sum(
            record["matched"] for record in expected_matches
        ),
        "candidate_fields_reconstructed": len(candidate_columns),
        "summary_checks": summary_checks,
        "failure_count": len(failures),
        "failures": failures[:200],
        "audited_output_sha256": {
            str(path): sha256(root / path)
            for path in (CATALOG, TIER_A, CONTROLS, SUMMARY)
        },
    }
    write_json(root / AUDIT, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
