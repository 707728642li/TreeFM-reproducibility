#!/usr/bin/env python3
"""Paired cross-genus comparison of publication-v3 functional predictions."""

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
READOUTS = ("linear", "xgboost")
GENERA = ("prunus", "pyrus")
POPULATIONS = (
    "all",
    "heldout_chromosome_test",
    "no_shared_orthogroup",
    "mapped_novel_orthogroup",
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=4).digest(), "little"
    )


def load_predictions(
    root: Path, seed: int, readout: str, genus: str
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    keys = [
        "gene_id",
        "chromosome",
        "label",
        "label_binary",
        "split",
        "genus",
        "orthogroup",
        "training_genus",
        "heldout_genus",
    ]
    for arm in ARMS:
        path = (
            root
            / arm
            / f"seed_{seed}"
            / f"heldout_{genus}.{readout}.predictions.parquet"
        )
        data = pd.read_parquet(path)
        keep = keys + ["probability"]
        data = data[keep].rename(columns={"probability": arm})
        if merged is None:
            merged = data
        else:
            merged = merged.merge(
                data, on=keys, how="inner", validate="one_to_one"
            )
    assert merged is not None
    return merged


def effects(scores: dict[str, float]) -> dict[str, float]:
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
    labels = data["label_binary"].to_numpy(dtype=np.int8)
    scores = {
        arm: float(average_precision_score(labels, data[arm]))
        for arm in ARMS
    }
    return effects(scores)


def stratified_paired_bootstrap(
    data: pd.DataFrame, replicates: int, seed: int
) -> tuple[dict[str, float], pd.DataFrame]:
    data = data.reset_index(drop=True)
    labels = data["label_binary"].to_numpy(dtype=np.int8)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise RuntimeError("functional bootstrap requires both label classes")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        indices = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        sampled_labels = labels[indices]
        scores = {
            arm: float(
                average_precision_score(
                    sampled_labels,
                    data[arm].to_numpy(dtype=np.float64)[indices],
                )
            )
            for arm in ARMS
        }
        rows.append({"replicate": replicate + 1, **effects(scores)})
    bootstrap = pd.DataFrame(rows)
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


def population_mask(data: pd.DataFrame, population: str) -> np.ndarray:
    if population == "all":
        return np.ones(len(data), dtype=bool)
    if population == "heldout_chromosome_test":
        return data["split"].eq("test").to_numpy()
    training_groups = {
        value
        for value in data.attrs.get("training_orthogroups", set())
        if value
    }
    orthogroups = data["orthogroup"].fillna("").astype(str)
    mapped = orthogroups.ne("").to_numpy()
    shared = orthogroups.isin(training_groups).to_numpy()
    if population == "no_shared_orthogroup":
        return ~shared
    if population == "mapped_novel_orthogroup":
        return mapped & ~shared
    raise ValueError(population)


def infer_training_groups(
    root: Path, training_genus: str
) -> set[str]:
    slug = (
        "prunus_persica"
        if training_genus == "prunus"
        else "pyrus_pyrifolia"
    )
    path = (
        root.parent.parent.parent
        / "data/processed/functional"
        / (
            "Prunus_publication_v3/promoter_labels.parquet"
            if training_genus == "prunus"
            else "Pyrus_PRJNA669907/promoter_labels.parquet"
        )
    )
    benchmark = (
        root.parent.parent.parent
        / "data/processed/technical_benchmarks_publication_v3_26"
        / f"{slug}.parquet"
    )
    genes = pd.read_parquet(path, columns=["gene_id"])["gene_id"]
    groups = pd.read_parquet(
        benchmark, columns=["gene_id", "orthogroup"]
    ).drop_duplicates("gene_id")
    return {
        value
        for value in groups.loc[
            groups["gene_id"].isin(set(genes)), "orthogroup"
        ].fillna("").astype(str)
        if value
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    root = args.project_root.resolve()
    metrics_root = (
        root
        / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
    )
    output_root = (
        root
        / "results/metrics/plantcad_dapt_publication_v3_functional_comparison"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    point_records: list[dict[str, object]] = []
    bootstrap_records: list[pd.DataFrame] = []

    for seed in SEEDS:
        for readout in READOUTS:
            for genus in GENERA:
                data = load_predictions(metrics_root, seed, readout, genus)
                training_genus = str(data["training_genus"].iloc[0])
                data.attrs["training_orthogroups"] = infer_training_groups(
                    metrics_root, training_genus
                )
                for population in POPULATIONS:
                    subset = data.loc[
                        population_mask(data, population)
                    ].reset_index(drop=True)
                    if len(subset) < 10 or subset["label_binary"].nunique() < 2:
                        continue
                    record: dict[str, object] = {
                        "seed": seed,
                        "readout": readout,
                        "heldout_genus": genus,
                        "training_genus": training_genus,
                        "population": population,
                        "rows": len(subset),
                        "positives": int(subset["label_binary"].sum()),
                        "negatives": int(
                            subset["label_binary"].eq(0).sum()
                        ),
                        **point_effects(subset),
                    }
                    if population == "all":
                        point, draws = stratified_paired_bootstrap(
                            subset,
                            args.bootstrap_replicates,
                            stable_seed(seed, readout, genus, population),
                        )
                        record.update(point)
                        draws.insert(0, "seed", seed)
                        draws.insert(1, "readout", readout)
                        draws.insert(2, "heldout_genus", genus)
                        draws.insert(3, "population", population)
                        bootstrap_records.append(draws)
                    point_records.append(record)
                print(
                    f"compared seed={seed} readout={readout} "
                    f"heldout={genus}",
                    flush=True,
                )

    points = pd.DataFrame(point_records)
    points.to_csv(
        output_root / "point_effects.tsv", sep="\t", index=False
    )
    bootstraps = pd.concat(bootstrap_records, ignore_index=True)
    bootstraps.to_parquet(
        output_root / "paired_gene_bootstrap.parquet",
        compression="zstd",
        index=False,
    )

    primary = points.loc[points["population"].eq("all")].copy()
    aggregation_rows: list[dict[str, object]] = []
    for keys, group in primary.groupby(
        ["readout", "heldout_genus"], sort=True
    ):
        if sorted(group["seed"].tolist()) != list(SEEDS):
            raise RuntimeError(f"functional seed alignment failed: {keys}")
        record = dict(zip(["readout", "heldout_genus"], keys))
        for metric in (
            "delta_vs_base",
            "delta_vs_random_plant",
            "delta_vs_herb",
            "delta_vs_phylogc_match",
            "woody_control_gain",
        ):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            record[f"{metric}_positive_seed_fraction"] = float(
                group[metric].gt(0).mean()
            )
            record[f"{metric}_worst_seed"] = float(group[metric].min())
        aggregation_rows.append(record)
    aggregation = pd.DataFrame(aggregation_rows)
    aggregation.to_csv(
        output_root / "multiseed_genus_effects.tsv",
        sep="\t",
        index=False,
    )

    readout_gates: dict[str, object] = {}
    for readout, group in aggregation.groupby("readout", sort=True):
        stable = group.loc[
            group["woody_control_gain_mean"].gt(0)
            & group["woody_control_gain_positive_seed_fraction"].ge(2 / 3)
        ]
        readout_gates[readout] = {
            "stable_genera": sorted(stable["heldout_genus"].tolist()),
            "both_nonblind_genera_stable": set(stable["heldout_genus"])
            == set(GENERA),
        }
    gate = {
        "status": "complete",
        "contract": "docs/publication_v3_functional_analysis_contract.md",
        "seeds": list(SEEDS),
        "heldout_genera": list(GENERA),
        "primary_endpoint": "leave-one-genus-out robust-label AUPRC",
        "readout_gates": readout_gates,
        "pre_malus_directionally_stable": all(
            details["both_nonblind_genera_stable"]
            for details in readout_gates.values()
        ),
        "claim_note": (
            "A generic woody conclusion remains prohibited until the sealed "
            "Malus evaluation and hierarchical synthesis are complete."
        ),
    }
    (output_root / "gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()

