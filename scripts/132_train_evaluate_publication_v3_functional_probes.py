#!/usr/bin/env python3
"""Fit frozen reciprocal Prunus-Pyrus leave-one-genus-out probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)
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
TECHNICAL_SLUGS = {
    "prunus": "prunus_persica",
    "pyrus": "pyrus_pyrifolia",
}
PRUNUS_GENE_ID_BRIDGE = Path(
    "metadata/publication_v3_prunus_v21_gene_id_bridge.tsv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_dataset(data: pd.DataFrame, genus: str) -> pd.DataFrame:
    required = {
        "gene_id",
        "chromosome",
        "label_binary",
        "label",
        "split",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"{genus} dataset lacks columns: {missing}")
    data = data.copy().sort_values(
        ["chromosome", "gene_id"], kind="stable"
    ).reset_index(drop=True)
    data["genus"] = genus
    if data["gene_id"].duplicated().any():
        raise RuntimeError(f"{genus} dataset contains duplicate genes")
    if set(data["label_binary"].astype(int)) != {0, 1}:
        raise RuntimeError(f"{genus} dataset lacks both classes")
    return data


def load_orthogroups(root: Path, genus: str) -> dict[str, str]:
    path = (
        root
        / "data/processed/technical_benchmarks_publication_v3_26"
        / f"{TECHNICAL_SLUGS[genus]}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_parquet(path, columns=["gene_id", "orthogroup"])
    data["orthogroup"] = data["orthogroup"].fillna("").astype(str)
    uniqueness = data.groupby("gene_id")["orthogroup"].nunique()
    if uniqueness.gt(1).any():
        raise RuntimeError(f"{genus} genes map to multiple orthogroups")
    data = data.drop_duplicates("gene_id")
    if genus != "prunus":
        return data.set_index("gene_id")["orthogroup"].to_dict()

    bridge_path = root / PRUNUS_GENE_ID_BRIDGE
    if not bridge_path.is_file():
        raise FileNotFoundError(bridge_path)
    bridge = pd.read_csv(bridge_path, sep="\t", dtype=str)
    required = {"source_gene_id", "technical_gene_id", "chromosome"}
    missing = sorted(required - set(bridge.columns))
    if missing:
        raise RuntimeError(f"Prunus gene-ID bridge lacks columns: {missing}")
    if (
        bridge["source_gene_id"].duplicated().any()
        or bridge["technical_gene_id"].duplicated().any()
    ):
        raise RuntimeError("Prunus gene-ID bridge is not one-to-one")
    mapped = bridge.merge(
        data[["gene_id", "orthogroup"]],
        left_on="technical_gene_id",
        right_on="gene_id",
        how="left",
        validate="one_to_one",
    )
    mapped["orthogroup"] = mapped["orthogroup"].fillna("").astype(str)
    return mapped.set_index("source_gene_id")["orthogroup"].to_dict()


def calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(
        np.digitize(probabilities, edges[1:-1]), bins - 1
    )
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += mask.mean() * abs(
                probabilities[mask].mean() - labels[mask].mean()
            )
    return float(error)


def top_k_enrichment(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[int, float]:
    positives = int(labels.sum())
    if positives < 1 or positives >= len(labels):
        return positives, float("nan")
    order = np.argsort(-probabilities, kind="stable")[:positives]
    precision = float(labels[order].mean())
    return positives, precision / float(labels.mean())


def metric_row(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    **identifiers: object,
) -> dict[str, object]:
    labels = data["label_binary"].to_numpy(dtype=np.int8)
    k, enrichment = top_k_enrichment(labels, probabilities)
    return {
        **identifiers,
        "rows": len(labels),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "prevalence": float(labels.mean()),
        "top_k": k,
        "auprc": average_precision_score(labels, probabilities),
        "auroc": roc_auc_score(labels, probabilities),
        "top_k_enrichment": enrichment,
        "ece_15bin": calibration_error(labels, probabilities),
    }


def population_masks(
    data: pd.DataFrame, training_orthogroups: set[str]
) -> dict[str, np.ndarray]:
    orthogroups = data["orthogroup"].fillna("").astype(str)
    mapped = orthogroups.ne("").to_numpy()
    shared = orthogroups.isin(training_orthogroups).to_numpy()
    return {
        "all": np.ones(len(data), dtype=bool),
        "heldout_chromosome_test": data["split"].eq("test").to_numpy(),
        "no_shared_orthogroup": ~shared,
        "mapped_novel_orthogroup": mapped & ~shared,
    }


def genus_class_balanced_weights(
    datasets: dict[str, pd.DataFrame]
) -> np.ndarray:
    blocks = []
    for genus in GENERA:
        labels = datasets[genus]["label_binary"].to_numpy(dtype=np.int8)
        weights = np.zeros(len(labels), dtype=np.float64)
        for label in (0, 1):
            mask = labels == label
            if not mask.any():
                raise RuntimeError(f"{genus} lacks label class {label}")
            weights[mask] = 0.25 / mask.sum()
        blocks.append(weights)
    combined = np.concatenate(blocks)
    if not np.isclose(combined.sum(), 1.0):
        raise RuntimeError("genus/class weights do not sum to one")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    root = args.project_root.resolve()
    embedding_seed = 0 if args.arm == "base" else args.seed
    embedding_dir = (
        root
        / "results/embeddings/plantcad_dapt_publication_v3_functional"
        / args.arm
        / f"seed_{embedding_seed}"
    )
    output_dir = (
        root
        / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
        / args.arm
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = embedding_dir / "manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    datasets: dict[str, pd.DataFrame] = {}
    embeddings: dict[str, np.ndarray] = {}
    dataset_hashes: dict[str, str] = {}
    for genus in GENERA:
        source = root / DATASETS[genus]
        if not source.is_file():
            raise FileNotFoundError(source)
        datasets[genus] = normalize_dataset(pd.read_parquet(source), genus)
        embeddings[genus] = np.load(
            embedding_dir / f"{genus}.npy", mmap_mode="r"
        )
        if embeddings[genus].shape != (len(datasets[genus]), 1536):
            raise RuntimeError(f"{genus} functional embedding rows differ")
        orthogroups = load_orthogroups(root, genus)
        datasets[genus]["orthogroup"] = (
            datasets[genus]["gene_id"].map(orthogroups).fillna("")
        )
        dataset_hashes[genus] = sha256(source)

    fingerprint_payload = {
        "contract": "docs/publication_v3_functional_analysis_contract.md",
        "arm": args.arm,
        "seed": args.seed,
        "embedding_manifest_sha256": sha256(manifest_path),
        "dataset_sha256": dataset_hashes,
        "prunus_gene_id_bridge_sha256": sha256(
            root / PRUNUS_GENE_ID_BRIDGE
        ),
        "linear": {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "average": True,
            "max_iter": 2000,
            "tol": 1e-4,
        },
        "xgboost": {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    run_spec_path = output_dir / "run_spec.json"
    previous = (
        json.loads(run_spec_path.read_text(encoding="utf-8"))
        if run_spec_path.is_file()
        else {}
    )
    reusable = previous.get("input_fingerprint") == fingerprint
    run_spec = {
        **fingerprint_payload,
        "input_fingerprint": fingerprint,
        "primary_endpoint": "leave-one-genus-out AUPRC",
        "training_policy": "all robust labels in the non-heldout genus",
        "heldout_policy": "no heldout-genus labels enter fitting",
    }
    run_spec_path.write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )

    metric_records: list[dict[str, object]] = []
    for heldout_genus in GENERA:
        training_genus = next(
            genus for genus in GENERA if genus != heldout_genus
        )
        train_data = datasets[training_genus]
        heldout_data = datasets[heldout_genus].copy()
        train_x = np.asarray(embeddings[training_genus], dtype=np.float32)
        train_y = train_data["label_binary"].to_numpy(dtype=np.int8)
        heldout_x = np.asarray(embeddings[heldout_genus], dtype=np.float32)
        training_groups = {
            value
            for value in train_data["orthogroup"].astype(str)
            if value
        }
        probabilities: dict[str, np.ndarray] = {}

        linear_path = output_dir / f"heldout_{heldout_genus}.linear.joblib"
        if linear_path.is_file() and reusable:
            bundle = joblib.load(linear_path)
            scaler = bundle["scaler"]
            linear = bundle["model"]
        else:
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_x)
            linear = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-4,
                class_weight="balanced",
                average=True,
                random_state=args.seed,
                n_jobs=args.n_jobs,
            )
            linear.fit(scaled_train, train_y)
            joblib.dump(
                {"scaler": scaler, "model": linear},
                linear_path,
                compress=3,
            )
        probabilities["linear"] = linear.predict_proba(
            scaler.transform(heldout_x)
        )[:, 1]

        negatives = int((train_y == 0).sum())
        positives = int(train_y.sum())
        xgb_path = output_dir / f"heldout_{heldout_genus}.xgboost.json"
        classifier = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=negatives / positives,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            tree_method="hist",
            eval_metric="logloss",
        )
        if xgb_path.is_file() and reusable:
            classifier.load_model(xgb_path)
        else:
            classifier.fit(train_x, train_y, verbose=False)
            classifier.save_model(xgb_path)
        probabilities["xgboost"] = classifier.predict_proba(heldout_x)[:, 1]

        for readout, probability in probabilities.items():
            predictions = heldout_data[
                [
                    "gene_id",
                    "chromosome",
                    "label",
                    "label_binary",
                    "split",
                    "genus",
                    "orthogroup",
                ]
                + (
                    ["endpoint_direction"]
                    if "endpoint_direction" in heldout_data
                    else []
                )
            ].copy()
            predictions["training_genus"] = training_genus
            predictions["heldout_genus"] = heldout_genus
            predictions["arm"] = args.arm
            predictions["seed"] = args.seed
            predictions["readout"] = readout
            predictions["probability"] = probability
            predictions.to_parquet(
                output_dir
                / f"heldout_{heldout_genus}.{readout}.predictions.parquet",
                compression="zstd",
                index=False,
            )
            for population, mask in population_masks(
                predictions, training_groups
            ).items():
                subset = predictions.loc[mask].reset_index(drop=True)
                subset_probability = probability[mask]
                if len(subset) < 10 or subset["label_binary"].nunique() < 2:
                    continue
                metric_records.append(
                    metric_row(
                        subset,
                        subset_probability,
                        arm=args.arm,
                        seed=args.seed,
                        readout=readout,
                        training_genus=training_genus,
                        heldout_genus=heldout_genus,
                        population=population,
                    )
                )
            print(
                f"{args.arm} seed={args.seed} {readout}: "
                f"{training_genus}->{heldout_genus} "
                f"train={len(train_data)} heldout={len(heldout_data)}",
                flush=True,
            )

    combined_x = np.concatenate(
        [
            np.asarray(embeddings[genus], dtype=np.float32)
            for genus in GENERA
        ]
    )
    combined_y = np.concatenate(
        [
            datasets[genus]["label_binary"].to_numpy(dtype=np.int8)
            for genus in GENERA
        ]
    )
    combined_weights = genus_class_balanced_weights(datasets)
    blind_linear_path = output_dir / "malus_blind_training_only.linear.joblib"
    blind_xgb_path = output_dir / "malus_blind_training_only.xgboost.json"
    if blind_linear_path.is_file() and blind_xgb_path.is_file() and reusable:
        blind_bundle = joblib.load(blind_linear_path)
        blind_scaler = blind_bundle["scaler"]
        blind_linear = blind_bundle["model"]
        blind_xgb = xgb.XGBClassifier()
        blind_xgb.load_model(blind_xgb_path)
    else:
        blind_scaler = StandardScaler()
        scaled_combined = blind_scaler.fit_transform(
            combined_x, sample_weight=combined_weights
        )
        blind_linear = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            average=True,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        blind_linear.fit(
            scaled_combined,
            combined_y,
            sample_weight=combined_weights,
        )
        joblib.dump(
            {"scaler": blind_scaler, "model": blind_linear},
            blind_linear_path,
            compress=3,
        )
        blind_xgb = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            tree_method="hist",
            eval_metric="logloss",
        )
        blind_xgb.fit(
            combined_x,
            combined_y,
            sample_weight=combined_weights,
            verbose=False,
        )
        blind_xgb.save_model(blind_xgb_path)
    blind_training_spec = {
        "status": "frozen_without_malus_access",
        "arm": args.arm,
        "seed": args.seed,
        "training_genera": list(GENERA),
        "training_rows": {
            genus: len(datasets[genus]) for genus in GENERA
        },
        "weighting": (
            "total weight 0.5 per genus and 0.25 per genus-label cell"
        ),
        "linear_model": blind_linear_path.name,
        "xgboost_model": blind_xgb_path.name,
        "malus_data_accessed": False,
        "input_fingerprint": fingerprint,
    }
    (output_dir / "malus_blind_training_only.json").write_text(
        json.dumps(blind_training_spec, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = pd.DataFrame(metric_records)
    metrics.to_csv(output_dir / "metrics.tsv", sep="\t", index=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": args.arm,
                "seed": args.seed,
                "metric_rows": len(metrics),
                "input_fingerprint": fingerprint,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
