#!/usr/bin/env python3
"""Exploratory orthogroup-level GO enrichment across Prunus and Pyrus."""

from __future__ import annotations

import argparse
import os
import gzip
import hashlib
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, hypergeom


AGI_PATTERN = re.compile(r"(?i)(AT[1-5MC]G\d{5})")
GENERA = ("prunus", "pyrus")
DATASETS = {
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
NAMESPACE_AUDIT = Path(
    "results/metrics/publication_v3_orthogroup_namespace_audit.json"
)
ROOT_TERM = "GO:0008150"
MIN_TERM_ORTHOGROUPS = 5
MAX_TERM_FRACTION = 0.80
MIN_COMPLEMENT_ORTHOGROUPS = 5
PRIMARY_EXCLUDED_EVIDENCE = {"IEA", "ND"}
ALL_EXCLUDED_EVIDENCE = {"ND"}
PERMUTATION_SEED = 20260717
GAF_URL = (
    "https://current.geneontology.org/annotations/gaf/ARATH-mod.gaf.gz"
)
OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=np.float64)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def parse_obo(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, set[str]]]:
    terms: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                if current and "id" in current:
                    terms[str(current["id"])] = current
                current = {"parents": set(), "obsolete": False}
            elif line.startswith("["):
                if current and "id" in current:
                    terms[str(current["id"])] = current
                current = None
            elif current is not None and ": " in line:
                key, value = line.split(": ", 1)
                if key == "id":
                    current["id"] = value
                elif key == "name":
                    current["name"] = value
                elif key == "namespace":
                    current["namespace"] = value
                elif key == "is_a":
                    current["parents"].add(value.split()[0])
                elif key == "relationship" and value.startswith("part_of "):
                    current["parents"].add(value.split()[1])
                elif key == "is_obsolete" and value == "true":
                    current["obsolete"] = True
        if current and "id" in current:
            terms[str(current["id"])] = current

    cache: dict[str, set[str]] = {}

    def ancestors(term_id: str, active: set[str] | None = None) -> set[str]:
        if term_id in cache:
            return cache[term_id]
        active = set() if active is None else active
        if term_id in active:
            raise RuntimeError(f"GO parent cycle detected at {term_id}")
        active.add(term_id)
        result = {term_id}
        for parent in terms.get(term_id, {}).get("parents", set()):
            result.update(ancestors(str(parent), active.copy()))
        cache[term_id] = result
        return result

    for term_id in terms:
        ancestors(term_id)
    return terms, cache


def parse_gaf(
    path: Path,
    terms: dict[str, dict[str, object]],
    ancestors: dict[str, set[str]],
    excluded_evidence: set[str],
) -> dict[str, set[str]]:
    annotations: dict[str, set[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                continue
            qualifier = fields[3]
            term_id = fields[4]
            evidence = fields[6]
            if "NOT" in qualifier.split("|") or evidence in excluded_evidence:
                continue
            term = terms.get(term_id)
            if (
                not term
                or term.get("obsolete") is True
                or term.get("namespace") != "biological_process"
            ):
                continue
            identifiers = " ".join(
                fields[index] for index in (1, 2, 10) if index < len(fields)
            )
            match = AGI_PATTERN.search(identifiers)
            if match is None:
                continue
            gene = match.group(1).upper()
            propagated = {
                parent
                for parent in ancestors.get(term_id, {term_id})
                if parent != ROOT_TERM
                and parent in terms
                and terms[parent].get("namespace") == "biological_process"
                and terms[parent].get("obsolete") is not True
            }
            annotations.setdefault(gene, set()).update(propagated)
    return annotations


def parse_orthogroup_arabidopsis(path: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            orthogroup, members = line.rstrip("\n").split(": ", 1)
            genes: set[str] = set()
            for member in members.split():
                if not member.startswith("arabidopsis_thaliana|"):
                    continue
                match = AGI_PATTERN.search(member)
                if match:
                    genes.add(match.group(1).upper())
            if genes:
                output[orthogroup] = genes
    return output


def parse_common_gene_to_group(
    path: Path, slugs: set[str]
) -> dict[tuple[str, str], str]:
    """Parse target-species membership from one OrthoFinder namespace."""
    output: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if ": " not in line:
                raise RuntimeError(
                    f"malformed Orthogroups line {line_number}"
                )
            orthogroup, members = line.rstrip("\n").split(": ", 1)
            for member in members.split():
                parts = member.split("|", 2)
                if len(parts) < 2 or parts[0] not in slugs:
                    continue
                key = (parts[0], parts[1])
                previous = output.setdefault(key, orthogroup)
                if previous != orthogroup:
                    raise RuntimeError(
                        f"gene occurs in multiple orthogroups: {key}"
                    )
    return output


def gene_level_frame(
    root: Path,
    genus: str,
    common_gene_to_group: dict[tuple[str, str], str],
) -> pd.DataFrame:
    labels = pd.read_parquet(
        root / DATASETS[genus],
        columns=["gene_id", "chromosome", "label_binary"],
    ).copy()
    labels["functional_gene_id"] = labels["gene_id"].astype(str)
    if genus == "prunus":
        bridge = pd.read_csv(root / BRIDGE, sep="\t", dtype=str)
        if bridge["source_gene_id"].duplicated().any():
            raise RuntimeError("Prunus bridge has duplicate source identifiers")
        bridge_map = bridge.set_index("source_gene_id")["technical_gene_id"]
        labels["technical_gene_id"] = labels["functional_gene_id"].map(
            bridge_map
        )
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
    labels["genus"] = genus
    return labels[
        ["genus", "functional_gene_id", "chromosome", "label_binary", "orthogroup"]
    ]


def collapse_orthogroups(genes: pd.DataFrame) -> pd.DataFrame:
    mapped = genes.loc[genes["orthogroup"].ne("")].copy()
    rows: list[dict[str, object]] = []
    for orthogroup, group in mapped.groupby("orthogroup", sort=True):
        positives = int(group["label_binary"].astype(int).sum())
        negatives = int(len(group) - positives)
        state = (
            "positive_only"
            if positives and not negatives
            else "negative_only"
            if negatives and not positives
            else "mixed"
        )
        rows.append(
            {
                "orthogroup": orthogroup,
                "state": state,
                "positive_genes": positives,
                "negative_genes": negatives,
                "labeled_genes": len(group),
                "chromosomes": ",".join(
                    sorted(group["chromosome"].astype(str).unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def orthogroup_terms(
    orthogroup_genes: dict[str, set[str]],
    gene_terms: dict[str, set[str]],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for orthogroup, genes in orthogroup_genes.items():
        terms: set[str] = set()
        for gene in genes:
            terms.update(gene_terms.get(gene, set()))
        if terms:
            output[orthogroup] = terms
    return output


def genus_term_statistics(
    collapsed: pd.DataFrame,
    annotations: dict[str, set[str]],
    term_ids: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[np.ndarray]]:
    universe = collapsed.loc[
        collapsed["state"].isin(["positive_only", "negative_only"])
        & collapsed["orthogroup"].isin(annotations)
    ].copy()
    universe = universe.sort_values("orthogroup", kind="stable").reset_index(drop=True)
    if len(universe) < 40 or universe["state"].nunique() != 2:
        raise RuntimeError("GO universe is too small or lacks a label class")
    # The annotation count for a GO term can exceed 127 orthogroups.  Keep both
    # operands of the matrix products at int32 so NumPy cannot silently wrap
    # observed counts (or raise during permutation assignment) at the int8
    # boundary.
    state = universe["state"].eq("positive_only").to_numpy(dtype=np.int32)
    orthogroups = universe["orthogroup"].tolist()
    matrix = np.zeros((len(term_ids), len(universe)), dtype=np.int32)
    for column, orthogroup in enumerate(orthogroups):
        for term_id in annotations[orthogroup]:
            if term_id in TERM_INDEX:
                matrix[TERM_INDEX[term_id], column] = 1
    n_total = len(universe)
    n_positive = int(state.sum())
    term_total = matrix.sum(axis=1).astype(int)
    term_positive = (matrix @ state).astype(int)
    records = []
    for index, term_id in enumerate(term_ids):
        total = int(term_total[index])
        positive = int(term_positive[index])
        negative = total - positive
        positive_not = n_positive - positive
        negative_total = n_total - n_positive
        negative_not = negative_total - negative
        eligible = bool(
            total >= MIN_TERM_ORTHOGROUPS
            and total / n_total <= MAX_TERM_FRACTION
            and n_total - total >= MIN_COMPLEMENT_ORTHOGROUPS
        )
        odds = float(
            ((positive + 0.5) * (negative_not + 0.5))
            / ((positive_not + 0.5) * (negative + 0.5))
        )
        p_value = float(
            hypergeom.sf(positive - 1, n_total, total, n_positive)
        )
        records.append(
            {
                "term_id": term_id,
                "universe_orthogroups": n_total,
                "positive_orthogroups": n_positive,
                "negative_orthogroups": negative_total,
                "term_orthogroups": total,
                "term_positive": positive,
                "term_negative": negative,
                "odds_ratio_haldane": odds,
                "p_one_sided": p_value,
                "eligible": eligible,
            }
        )
    strata = [
        np.flatnonzero(
            np.where(universe["labeled_genes"].to_numpy() >= 3, 3, universe["labeled_genes"])
            == value
        )
        for value in (1, 2, 3)
    ]
    return pd.DataFrame(records), matrix, state, strata


def crossgenus_table(
    genus_tables: dict[str, pd.DataFrame],
    terms: dict[str, dict[str, object]],
) -> pd.DataFrame:
    first = genus_tables["prunus"].add_prefix("prunus_")
    second = genus_tables["pyrus"].add_prefix("pyrus_")
    merged = first.merge(
        second,
        left_on="prunus_term_id",
        right_on="pyrus_term_id",
        validate="one_to_one",
    )
    merged["term_id"] = merged["prunus_term_id"]
    merged["term_name"] = merged["term_id"].map(
        lambda value: str(terms.get(value, {}).get("name", ""))
    )
    merged["eligible_both"] = (
        merged["prunus_eligible"] & merged["pyrus_eligible"]
    )
    eligible = merged["eligible_both"].to_numpy()
    p1 = np.maximum(merged["prunus_p_one_sided"].to_numpy(float), 1e-300)
    p2 = np.maximum(merged["pyrus_p_one_sided"].to_numpy(float), 1e-300)
    merged["fisher_statistic"] = -2.0 * (np.log(p1) + np.log(p2))
    merged["p_crossgenus"] = chi2.sf(merged["fisher_statistic"], df=4)
    merged["q_crossgenus"] = 1.0
    merged.loc[eligible, "q_crossgenus"] = bh_adjust(
        merged.loc[eligible, "p_crossgenus"].to_numpy(float)
    )
    merged["replicated_fdr_hit"] = (
        merged["eligible_both"]
        & merged["prunus_odds_ratio_haldane"].gt(1.0)
        & merged["pyrus_odds_ratio_haldane"].gt(1.0)
        & merged["prunus_p_one_sided"].le(0.05)
        & merged["pyrus_p_one_sided"].le(0.05)
        & merged["q_crossgenus"].le(0.05)
    )
    return merged.sort_values(
        ["q_crossgenus", "p_crossgenus", "term_id"], kind="stable"
    ).reset_index(drop=True)


def leave_one_chromosome(
    genes: pd.DataFrame,
    annotations: dict[str, set[str]],
    hits: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    chromosome_counts = genes.groupby("chromosome").size()
    chromosomes = sorted(chromosome_counts.loc[chromosome_counts >= 20].index)
    for chromosome in chromosomes:
        collapsed = collapse_orthogroups(
            genes.loc[genes["chromosome"].ne(chromosome)]
        )
        universe = collapsed.loc[
            collapsed["state"].isin(["positive_only", "negative_only"])
            & collapsed["orthogroup"].isin(annotations)
        ]
        positive_set = set(
            universe.loc[
                universe["state"].eq("positive_only"), "orthogroup"
            ]
        )
        negative_set = set(
            universe.loc[
                universe["state"].eq("negative_only"), "orthogroup"
            ]
        )
        for term_id in hits:
            annotated = {
                orthogroup
                for orthogroup in positive_set | negative_set
                if term_id in annotations.get(orthogroup, set())
            }
            a = len(positive_set & annotated)
            b = len(positive_set - annotated)
            c = len(negative_set & annotated)
            d = len(negative_set - annotated)
            odds = float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
            p_value = float(fisher_exact([[a, b], [c, d]], alternative="greater").pvalue)
            rows.append(
                {
                    "left_out_chromosome": chromosome,
                    "term_id": term_id,
                    "positive_orthogroups": len(positive_set),
                    "negative_orthogroups": len(negative_set),
                    "term_positive": a,
                    "term_negative": c,
                    "odds_ratio_haldane": odds,
                    "p_one_sided": p_value,
                    "direction_positive": odds > 1.0,
                }
            )
    return pd.DataFrame(rows)


def _permutation_chunk(args: tuple[object, ...]) -> list[int]:
    (
        start,
        stop,
        matrices,
        states,
        strata,
        term_totals,
        totals,
        positives,
    ) = args
    outputs: list[int] = []
    for permutation in range(int(start), int(stop)):
        genus_p: list[np.ndarray] = []
        genus_or: list[np.ndarray] = []
        for genus_index in range(2):
            rng = np.random.default_rng(
                PERMUTATION_SEED
                + permutation * 104729
                + genus_index * 1_000_003
            )
            shuffled = states[genus_index].copy()
            for indices in strata[genus_index]:
                shuffled[indices] = rng.permutation(shuffled[indices])
            matrix = matrices[genus_index]
            k = matrix @ shuffled
            n_total = totals[genus_index]
            n_positive = positives[genus_index]
            k_total = term_totals[genus_index]
            p = hypergeom.sf(k - 1, n_total, k_total, n_positive)
            negative_hits = k_total - k
            positive_not = n_positive - k
            negative_not = (n_total - n_positive) - negative_hits
            odds = (
                (k + 0.5) * (negative_not + 0.5)
                / ((positive_not + 0.5) * (negative_hits + 0.5))
            )
            genus_p.append(np.maximum(p, 1e-300))
            genus_or.append(odds)
        combined = chi2.sf(
            -2.0 * (np.log(genus_p[0]) + np.log(genus_p[1])), df=4
        )
        q_values = bh_adjust(combined)
        hits = (
            (genus_or[0] > 1.0)
            & (genus_or[1] > 1.0)
            & (genus_p[0] <= 0.05)
            & (genus_p[1] <= 0.05)
            & (q_values <= 0.05)
        )
        outputs.append(int(hits.sum()))
    return outputs


def permutation_null(
    matrices: list[np.ndarray],
    states: list[np.ndarray],
    strata: list[list[np.ndarray]],
    permutations: int,
    workers: int,
) -> np.ndarray:
    chunks = min(max(1, workers), permutations)
    boundaries = np.linspace(0, permutations, chunks + 1, dtype=int)
    term_totals = [matrix.sum(axis=1).astype(int) for matrix in matrices]
    totals = [len(state) for state in states]
    positives = [int(state.sum()) for state in states]
    arguments = [
        (
            int(boundaries[index]),
            int(boundaries[index + 1]),
            matrices,
            states,
            strata,
            term_totals,
            totals,
            positives,
        )
        for index in range(chunks)
        if boundaries[index] < boundaries[index + 1]
    ]
    with ProcessPoolExecutor(max_workers=chunks) as pool:
        blocks = list(pool.map(_permutation_chunk, arguments))
    return np.asarray([value for block in blocks for value in block], dtype=int)


TERM_INDEX: dict[str, int] = {}


def run_layer(
    layer: str,
    genes: dict[str, pd.DataFrame],
    collapsed: dict[str, pd.DataFrame],
    orthogroup_genes: dict[str, set[str]],
    gene_terms: dict[str, set[str]],
    terms: dict[str, dict[str, object]],
    permutations: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    annotations = orthogroup_terms(orthogroup_genes, gene_terms)
    term_ids = sorted({term for values in annotations.values() for term in values})
    global TERM_INDEX
    TERM_INDEX = {term_id: index for index, term_id in enumerate(term_ids)}
    genus_tables: dict[str, pd.DataFrame] = {}
    matrices: list[np.ndarray] = []
    states: list[np.ndarray] = []
    strata: list[list[np.ndarray]] = []
    for genus in GENERA:
        table, matrix, state, genus_strata = genus_term_statistics(
            collapsed[genus], annotations, term_ids
        )
        genus_tables[genus] = table
        matrices.append(matrix)
        states.append(state)
        strata.append(genus_strata)
    cross = crossgenus_table(genus_tables, terms)
    eligible_terms = cross.loc[cross["eligible_both"], "term_id"].tolist()
    eligible_indices = [TERM_INDEX[term_id] for term_id in eligible_terms]
    matrices_eligible = [matrix[eligible_indices] for matrix in matrices]
    observed_candidates = cross.loc[
        cross["replicated_fdr_hit"], "term_id"
    ].tolist()
    loo_blocks = []
    for genus in GENERA:
        loo = leave_one_chromosome(
            genes[genus], annotations, observed_candidates
        )
        if not loo.empty:
            loo.insert(0, "genus", genus)
            loo_blocks.append(loo)
    loo_table = (
        pd.concat(loo_blocks, ignore_index=True)
        if loo_blocks
        else pd.DataFrame(
            columns=[
                "genus",
                "left_out_chromosome",
                "term_id",
                "odds_ratio_haldane",
                "p_one_sided",
                "direction_positive",
            ]
        )
    )
    robustness: dict[tuple[str, str], float] = {}
    if not loo_table.empty:
        robustness = (
            loo_table.groupby(["genus", "term_id"])["direction_positive"]
            .mean()
            .to_dict()
        )
    cross["prunus_loo_direction_fraction"] = cross["term_id"].map(
        lambda term_id: robustness.get(("prunus", term_id), 0.0)
    )
    cross["pyrus_loo_direction_fraction"] = cross["term_id"].map(
        lambda term_id: robustness.get(("pyrus", term_id), 0.0)
    )
    cross["robust_replicated_hit"] = (
        cross["replicated_fdr_hit"]
        & cross["prunus_loo_direction_fraction"].ge(0.80)
        & cross["pyrus_loo_direction_fraction"].ge(0.80)
    )
    observed_candidate_count = int(cross["replicated_fdr_hit"].sum())
    robust_count = int(cross["robust_replicated_hit"].sum())
    null = (
        permutation_null(
            matrices_eligible, states, strata, permutations, workers
        )
        if permutations > 0 and eligible_terms
        else np.zeros(0, dtype=int)
    )
    empirical_p = (
        float(
            (1 + int((null >= observed_candidate_count).sum()))
            / (len(null) + 1)
        )
        if len(null)
        else None
    )
    null_table = pd.DataFrame(
        {"permutation": np.arange(len(null)), "replicated_fdr_hits": null}
    )
    summary = {
        "layer": layer,
        "arabidopsis_genes_with_bp": len(gene_terms),
        "orthogroups_with_arabidopsis_bp": len(annotations),
        "terms_total": len(term_ids),
        "terms_eligible_both": len(eligible_terms),
        "replicated_fdr_candidates": len(observed_candidates),
        "robust_replicated_terms": robust_count,
        "permutations": len(null),
        "null_median_hits": float(np.median(null)) if len(null) else None,
        "null_maximum_hits": int(null.max()) if len(null) else None,
        "empirical_p_replicated_candidate_count": empirical_p,
    }
    return cross, loo_table, null_table, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--gaf",
        type=Path,
        default=Path("data/raw/publication_v3_go/ARATH-mod.gaf.gz"),
    )
    parser.add_argument(
        "--obo",
        type=Path,
        default=Path("data/raw/publication_v3_go/go-basic.obo"),
    )
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root = args.project_root.resolve()
    expected = Path(os.environ["TREEFM_ROOT"]).resolve() if os.environ.get("TREEFM_ROOT") else None
    if expected is not None and root != expected:
        raise SystemExit(f"refusing to run outside {expected}: {root}")
    if args.permutations < 0 or args.workers < 1 or args.workers > 140:
        raise ValueError("invalid permutation or worker count")
    gaf_path = args.gaf if args.gaf.is_absolute() else root / args.gaf
    obo_path = args.obo if args.obo.is_absolute() else root / args.obo
    required = [
        gaf_path,
        obo_path,
        root / BRIDGE,
        root / ORTHOGROUPS,
        root / NAMESPACE_AUDIT,
        *(root / DATASETS[genus] for genus in GENERA),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    namespace_audit = json.loads(
        (root / NAMESPACE_AUDIT).read_text(encoding="utf-8")
    )
    if namespace_audit.get("status") != "namespace_mismatch_confirmed":
        raise RuntimeError("orthogroup namespace root cause is not confirmed")
    terms, ancestors = parse_obo(obo_path)
    orthogroup_genes = parse_orthogroup_arabidopsis(root / ORTHOGROUPS)
    common_gene_to_group = parse_common_gene_to_group(
        root / ORTHOGROUPS,
        {"prunus_persica", "pyrus_pyrifolia"},
    )
    genes = {
        genus: gene_level_frame(root, genus, common_gene_to_group)
        for genus in GENERA
    }
    collapsed = {
        genus: collapse_orthogroups(genes[genus]) for genus in GENERA
    }
    output_root = (
        root
        / "results/biological_cases/publication_v3_crossgenus_go"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    robust_blocks = []
    for layer, excluded in (
        ("curated_no_iea", PRIMARY_EXCLUDED_EVIDENCE),
        ("all_evidence", ALL_EXCLUDED_EVIDENCE),
    ):
        gene_terms = parse_gaf(gaf_path, terms, ancestors, excluded)
        layer_permutations = args.permutations if layer == "curated_no_iea" else 0
        cross, loo, null, summary = run_layer(
            layer,
            genes,
            collapsed,
            orthogroup_genes,
            gene_terms,
            terms,
            layer_permutations,
            args.workers,
        )
        cross.to_csv(
            output_root / f"{layer}_term_enrichment.tsv",
            sep="\t",
            index=False,
        )
        loo.to_csv(
            output_root / f"{layer}_leave_one_chromosome.tsv",
            sep="\t",
            index=False,
        )
        if layer_permutations:
            null.to_csv(
                output_root / f"{layer}_permutation_null.tsv.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )
        selected = cross.loc[cross["robust_replicated_hit"]].copy()
        if not selected.empty:
            selected.insert(0, "evidence_layer", layer)
            robust_blocks.append(selected)
        summaries[layer] = summary

    robust = (
        pd.concat(robust_blocks, ignore_index=True)
        if robust_blocks
        else pd.DataFrame(columns=["evidence_layer", "term_id", "term_name"])
    )
    robust.to_csv(
        output_root / "robust_replicated_terms.tsv", sep="\t", index=False
    )
    primary_count = int(
        (robust.get("evidence_layer", pd.Series(dtype=str)) == "curated_no_iea").sum()
    )
    gates = {
        "prunus_bridge_mapping_at_least_0_98": bool(
            genes["prunus"]["orthogroup"].ne("").mean() >= 0.98
        ),
        "pyrus_mapping_at_least_0_90": bool(
            genes["pyrus"]["orthogroup"].ne("").mean() >= 0.90
        ),
        "single_common_26species_orthogroup_namespace": True,
        "primary_permutations_complete": (
            summaries["curated_no_iea"]["permutations"] == args.permutations
        ),
        "malus_not_used": True,
    }
    summary = {
        "status": "complete" if all(gates.values()) else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "exploratory_crossgenus_orthogroup_go_"
            "corrected_common_26species_namespace"
        ),
        "scientific_decision_authority": False,
        "model_selection_allowed": False,
        "malus_accessed": False,
        "orthogroup_namespace": {
            "source": str(ORTHOGROUPS),
            "source_sha256": sha256(root / ORTHOGROUPS),
            "both_target_genera_mapped_directly": True,
            "arabidopsis_anchors_from_same_file": True,
            "supersedes_namespace_mismatched_result": True,
            "root_cause_audit": str(NAMESPACE_AUDIT),
        },
        "contract": (
            "docs/publication_v3_crossgenus_go_exploratory_contract_v2.md"
        ),
        "inputs": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                **(
                    {"source_url": GAF_URL}
                    if path == gaf_path
                    else {"source_url": OBO_URL}
                    if path == obo_path
                    else {}
                ),
            }
            for path in required
        },
        "mapping_qc": {
            genus: {
                "genes": len(genes[genus]),
                "mapped_genes": int(genes[genus]["orthogroup"].ne("").sum()),
                "mapping_fraction": float(
                    genes[genus]["orthogroup"].ne("").mean()
                ),
                "orthogroups": int(
                    genes[genus].loc[
                        genes[genus]["orthogroup"].ne(""), "orthogroup"
                    ].nunique()
                ),
                "positive_only": int(
                    collapsed[genus]["state"].eq("positive_only").sum()
                ),
                "negative_only": int(
                    collapsed[genus]["state"].eq("negative_only").sum()
                ),
                "mixed": int(collapsed[genus]["state"].eq("mixed").sum()),
            }
            for genus in GENERA
        },
        "layers": summaries,
        "primary_robust_replicated_terms": primary_count,
        "gates": gates,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
