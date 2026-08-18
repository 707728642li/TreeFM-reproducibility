#!/usr/bin/env python3
"""Train frozen linear and XGBoost probes on publication-v3 embeddings."""

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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)
TASKS = ("tis", "tts", "splice_donor", "splice_acceptor")
READOUTS = ("linear", "xgboost")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task(
    slugs: list[str],
    task: str,
    benchmark_root: Path,
    embedding_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    metadata_columns = (
        "pair_id",
        "slug",
        "life_form",
        "task",
        "gene_id",
        "label",
        "family_transfer_class",
        "orthogroup",
        "exact_task_train_sequence",
        "pair_has_exact_task_train_sequence",
        "maximum_task_train_identity",
        "maximum_task_train_identity_same_label",
        "maximum_task_train_identity_opposite_label",
        "pair_has_near_task_train_sequence_ge_0_90",
        "pair_has_near_task_train_sequence_ge_0_95",
    )
    for slug in slugs:
        data = pd.read_parquet(benchmark_root / f"{slug}.parquet")
        embedding = np.load(
            embedding_dir / f"{slug}.npy", mmap_mode="r"
        )
        if len(data) != len(embedding):
            raise RuntimeError(
                f"embedding row mismatch for {slug}: {len(data)} != {len(embedding)}"
            )
        mask = data["task"].eq(task).to_numpy()
        feature_blocks.append(np.asarray(embedding[mask], dtype=np.float32))
        label_blocks.append(data.loc[mask, "label"].to_numpy(dtype=np.int8))
        keep = [column for column in metadata_columns if column in data.columns]
        metadata_blocks.append(data.loc[mask, keep].copy())
    return (
        np.concatenate(feature_blocks),
        np.concatenate(label_blocks),
        pd.concat(metadata_blocks, ignore_index=True),
    )


def select_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    positives = labels.sum()
    negatives = len(labels) - positives
    tp = tpr * positives
    fn = positives - tp
    fp = fpr * negatives
    tn = negatives - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )
    finite = np.isfinite(thresholds)
    return float(thresholds[np.flatnonzero(finite)[np.argmax(mcc[finite])]])


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


def metric_row(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    **identifiers: object,
) -> dict[str, object]:
    predicted = probabilities >= threshold
    return {
        **identifiers,
        "rows": len(labels),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "threshold": threshold,
        "auprc": average_precision_score(labels, probabilities),
        "auroc": roc_auc_score(labels, probabilities),
        "mcc": matthews_corrcoef(labels, predicted),
        "balanced_accuracy": balanced_accuracy_score(labels, predicted),
        "ece_15bin": calibration_error(labels, probabilities),
    }


def population_masks(data: pd.DataFrame) -> dict[str, np.ndarray]:
    size = len(data)
    all_rows = np.ones(size, dtype=bool)
    return {
        "all": all_rows,
        "exclude_exact": (
            ~data["pair_has_exact_task_train_sequence"].fillna(False).to_numpy()
            if "pair_has_exact_task_train_sequence" in data
            else all_rows
        ),
        "exclude_near_0_90": (
            ~data[
                "pair_has_near_task_train_sequence_ge_0_90"
            ].fillna(False).to_numpy()
            if "pair_has_near_task_train_sequence_ge_0_90" in data
            else all_rows
        ),
        "exclude_near_0_95": (
            ~data[
                "pair_has_near_task_train_sequence_ge_0_95"
            ].fillna(False).to_numpy()
            if "pair_has_near_task_train_sequence_ge_0_95" in data
            else all_rows
        ),
    }


def add_scope_metrics(
    records: list[dict[str, object]],
    data: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    identifiers: dict[str, object],
) -> None:
    labels = data["label"].to_numpy(dtype=np.int8)
    families = ["all"]
    if "family_transfer_class" in data:
        families.extend(sorted(data["family_transfer_class"].dropna().unique()))
    for population, population_mask in population_masks(data).items():
        for family in families:
            if family == "all":
                mask = population_mask
            else:
                mask = population_mask & data[
                    "family_transfer_class"
                ].eq(family).to_numpy()
            if mask.sum() < 4 or np.unique(labels[mask]).size < 2:
                continue
            records.append(
                metric_row(
                    labels[mask],
                    probabilities[mask],
                    threshold,
                    **identifiers,
                    family_transfer_class=family,
                    identity_population=population,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=1000)
    args = parser.parse_args()
    root = args.project_root.resolve()
    panel_path = root / "config/publication_v3_technical_panel_26.tsv"
    benchmark_root = (
        root / "data/processed/technical_benchmarks_publication_v3_26"
    )
    embedding_seed = 0 if args.arm == "base" else args.seed
    embedding_dir = (
        root
        / "results/embeddings/plantcad_dapt_publication_v3"
        / args.arm
        / f"seed_{embedding_seed}"
    )
    output_dir = (
        root
        / "results/metrics/plantcad_dapt_publication_v3_probes"
        / args.arm
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_manifest = embedding_dir / "manifest.tsv"
    benchmark_manifest = (
        root / "metadata/publication_v3_technical_benchmark_manifest.tsv"
    )
    for required in (panel_path, embedding_manifest, benchmark_manifest):
        if not required.is_file():
            raise FileNotFoundError(required)
    panel = pd.read_csv(panel_path, sep="\t", dtype=str).query("include == '1'")
    train_slugs = panel.loc[
        panel["downstream_role"].eq("supervised_train"), "slug"
    ].tolist()
    dev_slugs = panel.loc[
        panel["downstream_role"].eq("development_holdout"), "slug"
    ].tolist()
    eval_panel = panel.loc[
        ~panel["downstream_role"].isin(
            ["supervised_train", "development_holdout"]
        )
    ].copy()
    eval_slugs = eval_panel["slug"].tolist()
    if len(train_slugs) != 19 or len(dev_slugs) != 1:
        raise RuntimeError(
            f"unexpected train/dev sizes: {len(train_slugs)}/{len(dev_slugs)}"
        )

    fingerprint_payload = {
        "arm": args.arm,
        "seed": args.seed,
        "embedding_manifest_sha256": sha256(embedding_manifest),
        "benchmark_manifest_sha256": sha256(benchmark_manifest),
        "panel_sha256": sha256(panel_path),
        "tasks": TASKS,
        "readouts": {
            "linear": {
                "model": "SGDClassifier_log_loss",
                "alpha": 1e-4,
                "penalty": "l2",
                "average": True,
                "max_iter": 2000,
                "tol": 1e-4,
            },
            "xgboost": {
                "n_estimators": args.n_estimators,
                "max_depth": 6,
                "learning_rate": 0.1,
                "tree_method": "hist",
            },
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
        "train_slugs": train_slugs,
        "development_slugs": dev_slugs,
        "evaluation_slugs": eval_slugs,
        "threshold_policy": "Castanea development maximum MCC per arm/task/readout",
        "benchmark_root": str(benchmark_root),
        "embedding_dir": str(embedding_dir),
    }
    run_spec_path.write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )

    metrics: list[dict[str, object]] = []
    for task in TASKS:
        train_x, train_y, _ = load_task(
            train_slugs, task, benchmark_root, embedding_dir
        )
        dev_x, dev_y, dev_meta = load_task(
            dev_slugs, task, benchmark_root, embedding_dir
        )
        eval_x, _, eval_meta = load_task(
            eval_slugs, task, benchmark_root, embedding_dir
        )
        readout_probabilities: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

        linear_path = output_dir / f"{task}.linear.joblib"
        if linear_path.is_file() and reusable:
            linear_bundle = joblib.load(linear_path)
            scaler = linear_bundle["scaler"]
            linear = linear_bundle["model"]
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
            del scaled_train
        linear_dev = linear.predict_proba(scaler.transform(dev_x))[:, 1]
        linear_eval = linear.predict_proba(scaler.transform(eval_x))[:, 1]
        linear_threshold = select_mcc_threshold(dev_y, linear_dev)
        readout_probabilities["linear"] = (
            linear_dev,
            linear_eval,
            linear_threshold,
        )

        xgboost_path = output_dir / f"{task}.xgboost.json"
        classifier = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=6,
            learning_rate=0.1,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            tree_method="hist",
            eval_metric="logloss",
        )
        if xgboost_path.is_file() and reusable:
            classifier.load_model(xgboost_path)
        else:
            classifier.fit(
                train_x,
                train_y,
                eval_set=[(dev_x, dev_y)],
                verbose=False,
            )
            classifier.save_model(xgboost_path)
        xgb_dev = classifier.predict_proba(dev_x)[:, 1]
        xgb_eval = classifier.predict_proba(eval_x)[:, 1]
        xgb_threshold = select_mcc_threshold(dev_y, xgb_dev)
        readout_probabilities["xgboost"] = (
            xgb_dev,
            xgb_eval,
            xgb_threshold,
        )

        for readout, (
            dev_probability,
            eval_probability,
            threshold,
        ) in readout_probabilities.items():
            dev_predictions = dev_meta.copy()
            dev_predictions["probability"] = dev_probability
            dev_predictions["threshold"] = threshold
            dev_predictions["readout"] = readout
            dev_predictions.to_parquet(
                output_dir
                / f"{task}.{readout}.development_predictions.parquet",
                compression="zstd",
                index=False,
            )
            add_scope_metrics(
                metrics,
                dev_predictions,
                dev_probability,
                threshold,
                {
                    "arm": args.arm,
                    "seed": args.seed,
                    "readout": readout,
                    "task": task,
                    "scope": "development",
                    "slug": ",".join(dev_slugs),
                    "analysis_tier": "development_reference",
                },
            )

            predictions = eval_meta.copy()
            predictions["probability"] = eval_probability
            predictions["threshold"] = threshold
            predictions["readout"] = readout
            predictions.to_parquet(
                output_dir / f"{task}.{readout}.evaluation_predictions.parquet",
                compression="zstd",
                index=False,
            )
            for slug, group in predictions.groupby("slug", sort=True):
                panel_row = eval_panel.loc[eval_panel["slug"].eq(slug)].iloc[0]
                indices = group.index.to_numpy()
                add_scope_metrics(
                    metrics,
                    group.reset_index(drop=True),
                    eval_probability[indices],
                    threshold,
                    {
                        "arm": args.arm,
                        "seed": args.seed,
                        "readout": readout,
                        "task": task,
                        "scope": "species",
                        "slug": slug,
                        "analysis_tier": panel_row["analysis_tier"],
                    },
                )
            primary_slugs = eval_panel.loc[
                eval_panel["primary_inference"].eq("1"), "slug"
            ].tolist()
            primary_mask = predictions["slug"].isin(primary_slugs).to_numpy()
            add_scope_metrics(
                metrics,
                predictions.loc[primary_mask].reset_index(drop=True),
                eval_probability[primary_mask],
                threshold,
                {
                    "arm": args.arm,
                    "seed": args.seed,
                    "readout": readout,
                    "task": task,
                    "scope": "primary_pooled",
                    "slug": ",".join(primary_slugs),
                    "analysis_tier": "primary_test",
                },
            )
            print(
                f"{args.arm} seed={args.seed} {task} {readout}: "
                f"train={len(train_y)} dev={len(dev_y)} eval={len(eval_meta)} "
                f"threshold={threshold:.6f}",
                flush=True,
            )
        del train_x, train_y, dev_x, dev_y, eval_x

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output_dir / "metrics.tsv", sep="\t", index=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": args.arm,
                "seed": args.seed,
                "metric_rows": len(metric_frame),
            }
        )
    )


if __name__ == "__main__":
    main()
