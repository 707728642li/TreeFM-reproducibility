#!/usr/bin/env python3
"""Paired comparison of Base and four publication-v3 DAPT arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)
TASKS = ("tis", "tts", "splice_donor", "splice_acceptor")
READOUTS = ("linear", "xgboost")
PRIMARY_SLUGS = (
    "hevea_brasiliensis",
    "prunus_persica",
    "pyrus_pyrifolia",
)
BOOTSTRAP_FAMILIES = ("logo_novel_family",)


def load_predictions(
    root: Path, seed: int, task: str, readout: str
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    keys = ["pair_id", "slug", "task", "gene_id", "label"]
    metadata_columns = [
        "family_transfer_class",
        "pair_has_exact_task_train_sequence",
        "pair_has_near_task_train_sequence_ge_0_90",
        "pair_has_near_task_train_sequence_ge_0_95",
        "maximum_task_train_identity",
    ]
    for arm in ARMS:
        path = (
            root
            / arm
            / f"seed_{seed}"
            / f"{task}.{readout}.evaluation_predictions.parquet"
        )
        data = pd.read_parquet(path)
        keep = keys + (
            [column for column in metadata_columns if column in data.columns]
            if merged is None
            else []
        )
        keep.append("probability")
        data = data[keep].rename(columns={"probability": arm})
        if merged is None:
            merged = data
        else:
            merged = merged.merge(
                data,
                on=keys,
                how="inner",
                validate="one_to_one",
            )
    assert merged is not None
    return merged


def effects_from_scores(scores: dict[str, float]) -> dict[str, float]:
    return {
        "delta_vs_base": scores["tree"] - scores["base"],
        "delta_vs_random_plant": scores["tree"] - scores["random_plant"],
        "delta_vs_herb": scores["tree"] - scores["herb"],
        "delta_vs_phylogc_match": scores["tree"] - scores["phylogc_match"],
        "woody_control_gain": scores["tree"]
        - max(
            scores["herb"],
            scores["random_plant"],
            scores["phylogc_match"],
        ),
        **{f"{arm}_auprc": scores[arm] for arm in ARMS},
    }


def point_effects(data: pd.DataFrame) -> dict[str, float]:
    labels = data["label"].to_numpy(dtype=np.int8)
    scores = {
        arm: float(average_precision_score(labels, data[arm]))
        for arm in ARMS
    }
    return effects_from_scores(scores)


def ap_from_pair_counts(
    labels: np.ndarray,
    probabilities: np.ndarray,
    pair_codes: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    sorted_pairs = pair_codes[order]
    group_ends = np.flatnonzero(
        np.r_[
            sorted_probabilities[:-1] != sorted_probabilities[1:],
            True,
        ]
    )
    weights = counts[:, sorted_pairs]
    positive_weights = weights * sorted_labels
    cumulative_positive = np.cumsum(positive_weights, axis=1)[:, group_ends]
    cumulative_total = np.cumsum(weights, axis=1)[:, group_ends]
    previous_positive = np.pad(
        cumulative_positive[:, :-1], ((0, 0), (1, 0))
    )
    increments = cumulative_positive - previous_positive
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=np.float64),
        where=cumulative_total > 0,
    )
    total_positive = positive_weights.sum(axis=1)
    return np.divide(
        (increments * precision).sum(axis=1),
        total_positive,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=total_positive > 0,
    )


def paired_bootstrap(
    data: pd.DataFrame,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    data = data.reset_index(drop=True)
    grouped = data.groupby("pair_id", sort=False)
    if not grouped.size().eq(2).all() or grouped["label"].nunique().ne(2).any():
        raise RuntimeError("pair-block bootstrap requires balanced two-row pairs")
    pair_codes, pair_ids = pd.factorize(data["pair_id"], sort=False)
    labels = data["label"].to_numpy(dtype=np.int8)
    probabilities = {
        arm: data[arm].to_numpy(dtype=np.float64) for arm in ARMS
    }
    rng = np.random.default_rng(seed)
    draws: list[pd.DataFrame] = []
    n_pairs = len(pair_ids)
    probabilities_uniform = np.full(n_pairs, 1 / n_pairs)
    completed = 0
    while completed < replicates:
        size = min(chunk_size, replicates - completed)
        counts = rng.multinomial(
            n_pairs, probabilities_uniform, size=size
        ).astype(np.float64)
        scores = {
            arm: ap_from_pair_counts(
                labels, probabilities[arm], pair_codes, counts
            )
            for arm in ARMS
        }
        frame = pd.DataFrame(
            {
                "replicate": np.arange(completed + 1, completed + size + 1),
                "delta_vs_base": scores["tree"] - scores["base"],
                "delta_vs_random_plant": scores["tree"]
                - scores["random_plant"],
                "delta_vs_herb": scores["tree"] - scores["herb"],
                "delta_vs_phylogc_match": scores["tree"]
                - scores["phylogc_match"],
                "woody_control_gain": scores["tree"]
                - np.maximum.reduce(
                    [
                        scores["herb"],
                        scores["random_plant"],
                        scores["phylogc_match"],
                    ]
                ),
                **{f"{arm}_auprc": values for arm, values in scores.items()},
            }
        )
        draws.append(frame)
        completed += size
    bootstrap = pd.concat(draws, ignore_index=True)
    point = point_effects(data)
    for metric in (
        "delta_vs_base",
        "delta_vs_random_plant",
        "delta_vs_herb",
        "delta_vs_phylogc_match",
        "woody_control_gain",
    ):
        point[f"{metric}_ci_low"] = float(
            bootstrap[metric].quantile(0.025)
        )
        point[f"{metric}_ci_high"] = float(
            bootstrap[metric].quantile(0.975)
        )
        point[f"{metric}_one_sided_p"] = float(
            (1 + bootstrap[metric].le(0).sum()) / (1 + len(bootstrap))
        )
    return point, bootstrap


def identity_masks(data: pd.DataFrame) -> dict[str, np.ndarray]:
    all_rows = np.ones(len(data), dtype=bool)
    return {
        "all": all_rows,
        "exclude_exact": ~data[
            "pair_has_exact_task_train_sequence"
        ].fillna(False).to_numpy(),
        "exclude_near_0_90": ~data[
            "pair_has_near_task_train_sequence_ge_0_90"
        ].fillna(False).to_numpy(),
        "exclude_near_0_95": ~data[
            "pair_has_near_task_train_sequence_ge_0_95"
        ].fillna(False).to_numpy(),
    }


def stable_scope_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=4).digest(), "little"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=25)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Evaluated seeds; use '--seeds 23' for the stopped pilot.",
    )
    parser.add_argument(
        "--output-subdir",
        default="plantcad_dapt_publication_v3_comparison",
        help="Subdirectory under results/metrics for comparison artifacts.",
    )
    args = parser.parse_args()
    seeds = tuple(dict.fromkeys(args.seeds))
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError("--seeds must contain one or more positive integers")
    root = args.project_root.resolve()
    metrics_root = (
        root / "results/metrics/plantcad_dapt_publication_v3_probes"
    )
    output_root = (
        root / "results/metrics" / args.output_subdir
    )
    output_root.mkdir(parents=True, exist_ok=True)
    point_records: list[dict[str, object]] = []
    bootstrap_records: list[pd.DataFrame] = []

    for seed in seeds:
        for readout in READOUTS:
            for task in TASKS:
                data = load_predictions(metrics_root, seed, task, readout)
                data = data.loc[data["slug"].isin(PRIMARY_SLUGS)].copy()
                for slug, species in data.groupby("slug", sort=True):
                    families = ["all"] + sorted(
                        species["family_transfer_class"].dropna().unique()
                    )
                    for identity_population, identity_mask in identity_masks(
                        species
                    ).items():
                        for family in families:
                            family_mask = (
                                np.ones(len(species), dtype=bool)
                                if family == "all"
                                else species[
                                    "family_transfer_class"
                                ].eq(family).to_numpy()
                            )
                            subset = species.loc[
                                identity_mask & family_mask
                            ].copy()
                            if (
                                len(subset) < 4
                                or subset["label"].nunique() < 2
                            ):
                                continue
                            record = {
                                "seed": seed,
                                "readout": readout,
                                "task": task,
                                "slug": slug,
                                "family_transfer_class": family,
                                "identity_population": identity_population,
                                "rows": len(subset),
                                "pairs": subset["pair_id"].nunique(),
                                **point_effects(subset),
                            }
                            point_records.append(record)
                            if (
                                identity_population == "all"
                                and family in BOOTSTRAP_FAMILIES
                            ):
                                point, draws = paired_bootstrap(
                                    subset,
                                    args.bootstrap_replicates,
                                    stable_scope_seed(
                                        seed, readout, task, slug, family
                                    ),
                                    args.bootstrap_chunk_size,
                                )
                                record.update(point)
                                draws.insert(0, "seed", seed)
                                draws.insert(1, "readout", readout)
                                draws.insert(2, "task", task)
                                draws.insert(3, "slug", slug)
                                draws.insert(
                                    4, "family_transfer_class", family
                                )
                                bootstrap_records.append(draws)
                print(
                    f"compared seed={seed} readout={readout} task={task}",
                    flush=True,
                )

    points = pd.DataFrame(point_records)
    points.to_csv(
        output_root / "point_effects_all_populations.tsv",
        sep="\t",
        index=False,
    )
    bootstraps = pd.concat(bootstrap_records, ignore_index=True)
    bootstraps.to_parquet(
        output_root / "paired_bootstrap_effects.parquet",
        compression="zstd",
        index=False,
    )
    bootstrap_scopes = points.loc[
        points["identity_population"].eq("all")
        & points["family_transfer_class"].isin(BOOTSTRAP_FAMILIES)
    ].copy()
    bootstrap_scopes.to_csv(
        output_root / "bootstrap_scope_effects.tsv",
        sep="\t",
        index=False,
    )

    primary = points.loc[
        points["identity_population"].eq("all")
        & points["family_transfer_class"].eq("logo_novel_family")
    ].copy()
    aggregation_rows: list[dict[str, object]] = []
    for keys, group in primary.groupby(
        ["readout", "task", "slug"], sort=True
    ):
        if sorted(group["seed"].tolist()) != sorted(seeds):
            raise RuntimeError(f"seed alignment failed for {keys}")
        record = dict(zip(["readout", "task", "slug"], keys))
        for metric in (
            "delta_vs_base",
            "delta_vs_random_plant",
            "delta_vs_herb",
            "delta_vs_phylogc_match",
            "woody_control_gain",
        ):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
            record[f"{metric}_positive_seed_fraction"] = float(
                group[metric].gt(0).mean()
            )
            record[f"{metric}_worst_seed"] = float(group[metric].min())
        aggregation_rows.append(record)
    aggregation = pd.DataFrame(aggregation_rows)
    aggregation_name = (
        "novel_family_multiseed_effects.tsv"
        if len(seeds) >= 3
        else "novel_family_seed23_effects.tsv"
    )
    aggregation.to_csv(
        output_root / aggregation_name,
        sep="\t",
        index=False,
    )
    readout_gates = {}
    for readout, group in aggregation.groupby("readout", sort=True):
        stable = group.loc[
            group["woody_control_gain_mean"].gt(0)
            & group["woody_control_gain_positive_seed_fraction"].ge(2 / 3)
        ]
        tasks_all_three = [
            task
            for task, task_group in stable.groupby("task")
            if set(task_group["slug"]) == set(PRIMARY_SLUGS)
        ]
        readout_gates[readout] = {
            "positive_genus_task_scopes": len(stable),
            "tasks_positive_in_all_three_genera": sorted(tasks_all_three),
        }
    direction_screen = bool(
        all(
            len(details["tasks_positive_in_all_three_genera"]) >= 2
            for details in readout_gates.values()
        )
    )
    gate = {
        "status": "complete",
        "seeds": list(seeds),
        "analysis_tier": (
            "confirmatory_multiseed" if len(seeds) >= 3 else "pilot_direction_only"
        ),
        "primary_slugs": list(PRIMARY_SLUGS),
        "primary_endpoint": "NovelFamily AUPRC",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "positive-negative pair block",
        "readout_gates": readout_gates,
        "seed23_direction_screen": direction_screen,
        "generic_woody_claim_screen": bool(
            len(seeds) >= 3 and direction_screen
        ),
        "aggregation_table": aggregation_name,
        "claim_note": (
            "This screen summarizes direction stability only. A one-seed run "
            "is a stopped-pilot analysis and cannot establish cross-seed "
            "stability; final claims also require the prespecified functional "
            "meta-effect and hierarchical uncertainty analysis."
        ),
        "malus_accessed": False,
    }
    (output_root / "gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
