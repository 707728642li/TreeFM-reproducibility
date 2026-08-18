#!/usr/bin/env python3
"""Independent audit of the publication-v3 exploratory cross-genus GO result.

This verifier deliberately does not import the analysis implementation.  It
reconstructs orthogroup labels and GO propagation from the frozen inputs using
set arithmetic and int64 counts, recalculates every observed test and
leave-one-chromosome result, and reproduces a deterministic sample of the
stored permutation null.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, hypergeom


ROOT = Path(os.environ.get("TREEFM_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = Path("results/biological_cases/publication_v3_crossgenus_go")
GAF = Path("data/raw/publication_v3_go/ARATH-mod.gaf.gz")
OBO = Path("data/raw/publication_v3_go/go-basic.obo")
ORTHOGROUPS = Path(
    "data/processed/orthofinder_benchmark_publication_v3/OrthoFinder/"
    "Results_PublicationV3/Orthogroups/Orthogroups.txt"
)
BRIDGE = Path("metadata/publication_v3_prunus_v21_gene_id_bridge.tsv")
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
AGI = re.compile(r"(?i)(AT[1-5MC]G\d{5})")
GO_ROOT = "GO:0008150"
PERMUTATION_SEED = 20260717


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty(len(values), dtype=np.float64)
    output[order] = np.minimum(ranked, 1.0)
    return output


def parse_ontology(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, set[str]]]:
    terms: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None

    def retain(record: dict[str, object] | None) -> None:
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
    def lineage(term_id: str) -> frozenset[str]:
        result = {term_id}
        for parent in terms.get(term_id, {}).get("parents", set()):
            result.update(lineage(str(parent)))
        return frozenset(result)

    ancestors = {term_id: set(lineage(term_id)) for term_id in terms}
    return terms, ancestors


def parse_gaf(
    path: Path,
    terms: dict[str, dict[str, object]],
    ancestors: dict[str, set[str]],
    excluded: set[str],
) -> dict[str, set[str]]:
    gene_terms: dict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                continue
            if "NOT" in fields[3].split("|") or fields[6] in excluded:
                continue
            term_id = fields[4]
            term = terms.get(term_id)
            if (
                term is None
                or term.get("obsolete")
                or term.get("namespace") != "biological_process"
            ):
                continue
            match = AGI.search(" ".join(fields[index] for index in (1, 2, 10)))
            if match is None:
                continue
            propagated = {
                ancestor
                for ancestor in ancestors.get(term_id, {term_id})
                if ancestor != GO_ROOT
                and ancestor in terms
                and terms[ancestor].get("namespace") == "biological_process"
                and not terms[ancestor].get("obsolete")
            }
            gene_terms[match.group(1).upper()].update(propagated)
    return dict(gene_terms)


def arabidopsis_orthogroups(path: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            orthogroup, members = line.rstrip("\n").split(": ", 1)
            genes = {
                match.group(1).upper()
                for member in members.split()
                if member.startswith("arabidopsis_thaliana|")
                for match in [AGI.search(member)]
                if match is not None
            }
            if genes:
                output[orthogroup] = genes
    return output


def direct_common_gene_to_group(
    path: Path, slugs: set[str]
) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if ": " not in line:
                raise RuntimeError(
                    f"malformed Orthogroups line {line_number}"
                )
            orthogroup, member_text = line.split(": ", 1)
            for member in member_text.split():
                parts = member.split("|", 2)
                if len(parts) < 2 or parts[0] not in slugs:
                    continue
                key = (parts[0], parts[1])
                previous = mapping.setdefault(key, orthogroup)
                if previous != orthogroup:
                    raise RuntimeError(f"duplicate membership: {key}")
    return mapping


def labeled_genes(
    root: Path,
    genus: str,
    common_gene_to_group: dict[tuple[str, str], str],
) -> pd.DataFrame:
    labels = pd.read_parquet(
        root / LABELS[genus],
        columns=["gene_id", "chromosome", "label_binary"],
    ).copy()
    labels["functional_gene_id"] = labels["gene_id"].astype(str)
    if genus == "prunus":
        bridge = pd.read_csv(root / BRIDGE, sep="\t", dtype=str)
        if bridge["source_gene_id"].duplicated().any():
            raise RuntimeError("duplicate Prunus bridge source identifier")
        bridge_map = bridge.set_index("source_gene_id")["technical_gene_id"]
        labels["technical_gene_id"] = labels["functional_gene_id"].map(bridge_map)
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
    return labels[
        [
            "functional_gene_id",
            "chromosome",
            "label_binary",
            "orthogroup",
        ]
    ]


def collapse(genes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    mapped = genes.loc[genes["orthogroup"].ne("")]
    for orthogroup, group in mapped.groupby("orthogroup", sort=True):
        positives = int(group["label_binary"].sum())
        negatives = int(len(group) - positives)
        state = (
            "positive_only"
            if positives > 0 and negatives == 0
            else "negative_only"
            if negatives > 0 and positives == 0
            else "mixed"
        )
        records.append(
            {
                "orthogroup": orthogroup,
                "state": state,
                "labeled_genes": int(len(group)),
            }
        )
    return pd.DataFrame(records)


def orthogroup_annotations(
    orthogroup_genes: dict[str, set[str]],
    gene_terms: dict[str, set[str]],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for orthogroup, genes in orthogroup_genes.items():
        annotations: set[str] = set()
        for gene in genes:
            annotations.update(gene_terms.get(gene, set()))
        if annotations:
            output[orthogroup] = annotations
    return output


def reconstructed_layer(
    collapsed: dict[str, pd.DataFrame],
    annotations: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    term_ids = sorted({term for values in annotations.values() for term in values})
    term_orthogroups: dict[str, set[str]] = defaultdict(set)
    for orthogroup, values in annotations.items():
        for term_id in values:
            term_orthogroups[term_id].add(orthogroup)
    genus_tables: dict[str, pd.DataFrame] = {}
    records: dict[str, list[dict[str, object]]] = {"prunus": [], "pyrus": []}
    universes: dict[str, pd.DataFrame] = {}
    for genus in ("prunus", "pyrus"):
        universe = collapsed[genus].loc[
            collapsed[genus]["state"].isin(["positive_only", "negative_only"])
            & collapsed[genus]["orthogroup"].isin(annotations)
        ].sort_values("orthogroup", kind="stable").reset_index(drop=True)
        universes[genus] = universe
        universe_set = set(universe["orthogroup"])
        positive_set = set(
            universe.loc[universe["state"].eq("positive_only"), "orthogroup"]
        )
        n_total = len(universe)
        n_positive = len(positive_set)
        n_negative = n_total - n_positive
        for term_id in term_ids:
            annotated = universe_set & term_orthogroups[term_id]
            total = len(annotated)
            positive = len(positive_set & annotated)
            negative = total - positive
            eligible = (
                total >= 5
                and total / n_total <= 0.80
                and n_total - total >= 5
            )
            odds = (
                (positive + 0.5) * (n_negative - negative + 0.5)
                / ((n_positive - positive + 0.5) * (negative + 0.5))
            )
            records[genus].append(
                {
                    "term_id": term_id,
                    "universe_orthogroups": n_total,
                    "positive_orthogroups": n_positive,
                    "negative_orthogroups": n_negative,
                    "term_orthogroups": total,
                    "term_positive": positive,
                    "term_negative": negative,
                    "odds_ratio_haldane": odds,
                    "p_one_sided": hypergeom.sf(
                        positive - 1, n_total, total, n_positive
                    ),
                    "eligible": eligible,
                }
            )
        genus_tables[genus] = pd.DataFrame(records[genus]).set_index("term_id")

    table = genus_tables["prunus"].add_prefix("prunus_").join(
        genus_tables["pyrus"].add_prefix("pyrus_"), how="inner"
    )
    table["eligible_both"] = table["prunus_eligible"] & table["pyrus_eligible"]
    p1 = np.maximum(table["prunus_p_one_sided"].to_numpy(float), 1e-300)
    p2 = np.maximum(table["pyrus_p_one_sided"].to_numpy(float), 1e-300)
    table["fisher_statistic"] = -2 * (np.log(p1) + np.log(p2))
    table["p_crossgenus"] = chi2.sf(table["fisher_statistic"], df=4)
    table["q_crossgenus"] = 1.0
    eligible = table["eligible_both"].to_numpy(bool)
    table.loc[eligible, "q_crossgenus"] = bh(
        table.loc[eligible, "p_crossgenus"].to_numpy(float)
    )
    table["replicated_fdr_hit"] = (
        table["eligible_both"]
        & table["prunus_odds_ratio_haldane"].gt(1)
        & table["pyrus_odds_ratio_haldane"].gt(1)
        & table["prunus_p_one_sided"].le(0.05)
        & table["pyrus_p_one_sided"].le(0.05)
        & table["q_crossgenus"].le(0.05)
    )
    return table, universes


def assert_frame_matches(observed: pd.DataFrame, expected: pd.DataFrame) -> None:
    observed = observed.set_index("term_id").sort_index()
    expected = expected.sort_index()
    if not observed.index.equals(expected.index):
        raise AssertionError("GO term index differs from independent reconstruction")
    integer_columns = [
        f"{genus}_{name}"
        for genus in ("prunus", "pyrus")
        for name in (
            "universe_orthogroups",
            "positive_orthogroups",
            "negative_orthogroups",
            "term_orthogroups",
            "term_positive",
            "term_negative",
        )
    ]
    boolean_columns = [
        "prunus_eligible",
        "pyrus_eligible",
        "eligible_both",
        "replicated_fdr_hit",
    ]
    float_columns = [
        f"{genus}_{name}"
        for genus in ("prunus", "pyrus")
        for name in ("odds_ratio_haldane", "p_one_sided")
    ] + ["fisher_statistic", "p_crossgenus", "q_crossgenus"]
    for column in integer_columns + boolean_columns:
        if not np.array_equal(
            observed[column].to_numpy(), expected[column].to_numpy()
        ):
            raise AssertionError(f"observed column mismatch: {column}")
    for column in float_columns:
        if not np.allclose(
            observed[column].to_numpy(float),
            expected[column].to_numpy(float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise AssertionError(f"observed numeric mismatch: {column}")


def verify_leave_one_chromosome(
    genes: dict[str, pd.DataFrame],
    annotations: dict[str, set[str]],
    candidates: list[str],
    observed_path: Path,
) -> dict[tuple[str, str], float]:
    records: list[dict[str, object]] = []
    for genus in ("prunus", "pyrus"):
        chromosome_counts = genes[genus].groupby("chromosome").size()
        for chromosome in sorted(chromosome_counts[chromosome_counts >= 20].index):
            reduced = collapse(
                genes[genus].loc[genes[genus]["chromosome"].ne(chromosome)]
            )
            universe = reduced.loc[
                reduced["state"].isin(["positive_only", "negative_only"])
                & reduced["orthogroup"].isin(annotations)
            ]
            positives = set(
                universe.loc[universe["state"].eq("positive_only"), "orthogroup"]
            )
            negatives = set(
                universe.loc[universe["state"].eq("negative_only"), "orthogroup"]
            )
            for term_id in candidates:
                annotated = {
                    orthogroup
                    for orthogroup in positives | negatives
                    if term_id in annotations.get(orthogroup, set())
                }
                a = len(positives & annotated)
                b = len(positives - annotated)
                c = len(negatives & annotated)
                d = len(negatives - annotated)
                odds = (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))
                records.append(
                    {
                        "genus": genus,
                        "left_out_chromosome": chromosome,
                        "term_id": term_id,
                        "positive_orthogroups": len(positives),
                        "negative_orthogroups": len(negatives),
                        "term_positive": a,
                        "term_negative": c,
                        "odds_ratio_haldane": odds,
                        "p_one_sided": fisher_exact(
                            [[a, b], [c, d]], alternative="greater"
                        ).pvalue,
                        "direction_positive": odds > 1,
                    }
                )
    expected = pd.DataFrame(records).sort_values(
        ["genus", "left_out_chromosome", "term_id"], kind="stable"
    ).reset_index(drop=True)
    observed = pd.read_csv(observed_path, sep="\t").sort_values(
        ["genus", "left_out_chromosome", "term_id"], kind="stable"
    ).reset_index(drop=True)
    key_and_count = [
        "genus",
        "left_out_chromosome",
        "term_id",
        "positive_orthogroups",
        "negative_orthogroups",
        "term_positive",
        "term_negative",
        "direction_positive",
    ]
    if not expected[key_and_count].equals(observed[key_and_count]):
        raise AssertionError("leave-one-chromosome identities/counts differ")
    for column in ("odds_ratio_haldane", "p_one_sided"):
        if not np.allclose(
            expected[column].to_numpy(float),
            observed[column].to_numpy(float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise AssertionError(f"leave-one-chromosome mismatch: {column}")
    return (
        expected.groupby(["genus", "term_id"])["direction_positive"]
        .mean()
        .to_dict()
    )


def sampled_null_counts(
    table: pd.DataFrame,
    universes: dict[str, pd.DataFrame],
    annotations: dict[str, set[str]],
    permutation_indices: np.ndarray,
) -> dict[int, int]:
    eligible_terms = sorted(
        table.index[table["eligible_both"]],
        key=lambda term_id: (
            table.at[term_id, "q_crossgenus"],
            table.at[term_id, "p_crossgenus"],
            term_id,
        ),
    )
    matrices: list[np.ndarray] = []
    states: list[np.ndarray] = []
    strata: list[list[np.ndarray]] = []
    for genus in ("prunus", "pyrus"):
        universe = universes[genus].reset_index(drop=True)
        orthogroups = universe["orthogroup"].tolist()
        matrix = np.asarray(
            [
                [
                    int(term_id in annotations.get(orthogroup, set()))
                    for orthogroup in orthogroups
                ]
                for term_id in eligible_terms
            ],
            dtype=np.int64,
        )
        state = universe["state"].eq("positive_only").to_numpy(dtype=np.int64)
        bins = np.minimum(
            universe["labeled_genes"].to_numpy(dtype=np.int64), 3
        )
        matrices.append(matrix)
        states.append(state)
        strata.append([np.flatnonzero(bins == value) for value in (1, 2, 3)])
    totals = [len(state) for state in states]
    positives = [int(state.sum()) for state in states]
    term_totals = [matrix.sum(axis=1, dtype=np.int64) for matrix in matrices]
    output: dict[int, int] = {}
    for permutation in permutation_indices:
        genus_p: list[np.ndarray] = []
        genus_odds: list[np.ndarray] = []
        for genus_index in range(2):
            rng = np.random.default_rng(
                PERMUTATION_SEED
                + int(permutation) * 104729
                + genus_index * 1_000_003
            )
            shuffled = states[genus_index].copy()
            for indices in strata[genus_index]:
                shuffled[indices] = rng.permutation(shuffled[indices])
            hits = matrices[genus_index] @ shuffled
            term_total = term_totals[genus_index]
            n_total = totals[genus_index]
            n_positive = positives[genus_index]
            negative_hits = term_total - hits
            positive_not = n_positive - hits
            negative_not = (n_total - n_positive) - negative_hits
            genus_p.append(
                np.maximum(
                    hypergeom.sf(
                        hits - 1, n_total, term_total, n_positive
                    ),
                    1e-300,
                )
            )
            genus_odds.append(
                (hits + 0.5) * (negative_not + 0.5)
                / ((positive_not + 0.5) * (negative_hits + 0.5))
            )
        combined = chi2.sf(
            -2 * (np.log(genus_p[0]) + np.log(genus_p[1])), df=4
        )
        q_values = bh(combined)
        output[int(permutation)] = int(
            (
                (genus_odds[0] > 1)
                & (genus_odds[1] > 1)
                & (genus_p[0] <= 0.05)
                & (genus_p[1] <= 0.05)
                & (q_values <= 0.05)
            ).sum()
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sampled-permutations", type=int, default=64)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT:
        raise SystemExit(f"refusing to audit outside {ROOT}: {root}")
    if args.sampled_permutations < 8 or args.sampled_permutations > 512:
        raise ValueError("sampled-permutations must be in [8, 512]")
    output_root = root / OUT
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates: dict[str, bool] = {
        "analysis_declares_no_malus_access": (
            summary.get("malus_accessed") is False
            and summary.get("scientific_decision_authority") is False
            and summary.get("model_selection_allowed") is False
        ),
        "reported_input_hashes_match": True,
    }
    for relative, record in summary["inputs"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            gates["reported_input_hashes_match"] = False

    terms, ancestors = parse_ontology(root / OBO)
    orthogroup_genes = arabidopsis_orthogroups(root / ORTHOGROUPS)
    common_gene_to_group = direct_common_gene_to_group(
        root / ORTHOGROUPS,
        {"prunus_persica", "pyrus_pyrifolia"},
    )
    genes = {
        genus: labeled_genes(root, genus, common_gene_to_group)
        for genus in ("prunus", "pyrus")
    }
    gates["single_common_26species_namespace_reconstructed"] = bool(
        summary.get("orthogroup_namespace", {}).get(
            "both_target_genera_mapped_directly"
        )
        and summary.get("orthogroup_namespace", {}).get(
            "arabidopsis_anchors_from_same_file"
        )
    )
    collapsed = {genus: collapse(genes[genus]) for genus in genes}
    layers: dict[str, dict[str, object]] = {}
    primary_table: pd.DataFrame | None = None
    primary_annotations: dict[str, set[str]] | None = None
    primary_universes: dict[str, pd.DataFrame] | None = None
    for layer, excluded in (
        ("curated_no_iea", {"IEA", "ND"}),
        ("all_evidence", {"ND"}),
    ):
        gene_terms = parse_gaf(root / GAF, terms, ancestors, excluded)
        annotations = orthogroup_annotations(orthogroup_genes, gene_terms)
        reconstructed, universes = reconstructed_layer(collapsed, annotations)
        observed = pd.read_csv(
            output_root / f"{layer}_term_enrichment.tsv", sep="\t"
        )
        assert_frame_matches(observed, reconstructed)
        candidate_count = int(reconstructed["replicated_fdr_hit"].sum())
        gates[f"{layer}_all_observed_statistics_reproduced"] = True
        gates[f"{layer}_candidate_count_matches"] = (
            candidate_count
            == summary["layers"][layer]["replicated_fdr_candidates"]
        )
        layers[layer] = {
            "genes_with_bp": len(gene_terms),
            "orthogroups_with_bp": len(annotations),
            "terms": len(reconstructed),
            "eligible_terms": int(reconstructed["eligible_both"].sum()),
            "replicated_fdr_candidates": candidate_count,
        }
        if layer == "curated_no_iea":
            primary_table = reconstructed
            primary_annotations = annotations
            primary_universes = universes

    assert primary_table is not None
    assert primary_annotations is not None
    assert primary_universes is not None
    candidates = primary_table.index[
        primary_table["replicated_fdr_hit"]
    ].tolist()
    loo_fraction = verify_leave_one_chromosome(
        genes,
        primary_annotations,
        candidates,
        output_root / "curated_no_iea_leave_one_chromosome.tsv",
    )
    robust = [
        term_id
        for term_id in candidates
        if loo_fraction.get(("prunus", term_id), 0) >= 0.80
        and loo_fraction.get(("pyrus", term_id), 0) >= 0.80
    ]
    gates["primary_leave_one_chromosome_reproduced"] = True
    gates["primary_robust_count_matches"] = (
        len(robust) == summary["primary_robust_replicated_terms"]
    )

    null_path = output_root / "curated_no_iea_permutation_null.tsv.gz"
    null = pd.read_csv(null_path, sep="\t")
    gates["permutation_file_complete"] = (
        len(null) == 10_000
        and np.array_equal(null["permutation"].to_numpy(), np.arange(10_000))
    )
    indices = np.unique(
        np.linspace(
            0, 9_999, args.sampled_permutations, dtype=np.int64
        )
    )
    reproduced = sampled_null_counts(
        primary_table,
        primary_universes,
        primary_annotations,
        indices,
    )
    observed_counts = null.set_index("permutation")["replicated_fdr_hits"]
    gates["sampled_permutation_counts_reproduced_int64"] = all(
        int(observed_counts.loc[index]) == count
        for index, count in reproduced.items()
    )
    observed_candidate_count = len(candidates)
    empirical_p = (
        1
        + int(
            (
                null["replicated_fdr_hits"].to_numpy()
                >= observed_candidate_count
            ).sum()
        )
    ) / 10_001
    gates["permutation_summary_matches"] = (
        float(np.median(null["replicated_fdr_hits"]))
        == summary["layers"]["curated_no_iea"]["null_median_hits"]
        and int(null["replicated_fdr_hits"].max())
        == summary["layers"]["curated_no_iea"]["null_maximum_hits"]
        and np.isclose(
            empirical_p,
            summary["layers"]["curated_no_iea"][
                "empirical_p_replicated_candidate_count"
            ],
            rtol=0,
            atol=1e-15,
        )
    )

    artifacts = [
        summary_path,
        output_root / "curated_no_iea_term_enrichment.tsv",
        output_root / "curated_no_iea_leave_one_chromosome.tsv",
        null_path,
        output_root / "all_evidence_term_enrichment.tsv",
        output_root / "all_evidence_leave_one_chromosome.tsv",
        output_root / "robust_replicated_terms.tsv",
    ]
    # NumPy comparisons can yield np.bool_; normalize the persisted audit to
    # JSON-native booleans without weakening any gate.
    gates = {name: bool(value) for name, value in gates.items()}
    audit = {
        "status": "pass" if all(gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "independent_crossgenus_go_reconstruction_"
            "common_26species_namespace"
        ),
        "analysis_module_imported": False,
        "count_dtype": "int64_and_python_integer",
        "malus_accessed": False,
        "layers": layers,
        "primary_robust_terms": robust,
        "sampled_permutation_indices": [int(value) for value in indices],
        "sampled_permutation_counts": reproduced,
        "empirical_p": empirical_p,
        "gates": gates,
        "artifacts": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        },
    }
    destination = output_root / "independent_audit.json"
    partial = destination.with_suffix(".json.partial")
    partial.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, destination)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
