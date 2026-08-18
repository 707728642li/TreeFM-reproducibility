#!/usr/bin/env python3
"""Score fixed-G-box perturbations with frozen cross-genus probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)
GENERA = ("prunus", "pyrus")
READOUTS = ("linear", "xgboost")
VARIANTS = Path(
    "data/processed/functional/publication_v3_gbox_mutagenesis/"
    "promoter_variants.parquet"
)
VARIANT_MANIFEST = VARIANTS.with_name("manifest.json")
FREEZE = Path("config/publication_v3_gbox_model_attribution_freeze.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validate_freeze(root: Path) -> dict[str, object]:
    path = root / FREEZE
    if not path.is_file():
        raise FileNotFoundError(path)
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        raise RuntimeError("G-box model-attribution freeze is not active")
    if freeze.get("malus_accessed") is not False:
        raise RuntimeError("G-box attribution freeze does not preserve Malus")
    for relative, expected in freeze.get("artifact_sha256", {}).items():
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if sha256(artifact) != expected:
            raise RuntimeError(
                f"G-box attribution artifact changed after freeze: {relative}"
            )
    return freeze


def load_linear_probability(
    model_path: Path, embeddings: np.ndarray
) -> np.ndarray:
    bundle = joblib.load(model_path)
    if set(bundle) != {"scaler", "model"}:
        raise RuntimeError(f"unexpected linear bundle: {model_path}")
    return bundle["model"].predict_proba(
        bundle["scaler"].transform(embeddings)
    )[:, 1]


def load_xgboost_probability(
    model_path: Path, embeddings: np.ndarray, n_jobs: int
) -> np.ndarray:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(n_jobs=n_jobs)
    return model.predict_proba(embeddings)[:, 1]


def chromosome_block_bootstrap(
    effects: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    positive = effects.loc[
        effects["label_binary"].eq(1), "gbox_dependence"
    ].to_numpy(dtype=np.float64)
    negative = effects.loc[
        effects["label_binary"].eq(0), "gbox_dependence"
    ].to_numpy(dtype=np.float64)
    if not len(positive) or not len(negative):
        raise RuntimeError("attribution population lacks a label class")
    observed = float(positive.mean() - negative.mean())
    chromosomes = sorted(effects["chromosome"].astype(str).unique())
    chromosome_blocks = {
        chromosome: effects[
            effects["chromosome"].astype(str).eq(chromosome)
        ]
        for chromosome in chromosomes
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    attempts = 0
    while len(estimates) < replicates and attempts < replicates * 20:
        attempts += 1
        sampled = rng.choice(
            chromosomes, size=len(chromosomes), replace=True
        )
        boot = pd.concat(
            [chromosome_blocks[str(chromosome)] for chromosome in sampled],
            ignore_index=True,
        )
        pos = boot.loc[
            boot["label_binary"].eq(1), "gbox_dependence"
        ]
        neg = boot.loc[
            boot["label_binary"].eq(0), "gbox_dependence"
        ]
        if len(pos) and len(neg):
            estimates.append(float(pos.mean() - neg.mean()))
    if len(estimates) != replicates:
        raise RuntimeError("chromosome-block bootstrap did not converge")
    array = np.asarray(estimates, dtype=np.float64)
    lower, upper = np.quantile(array, [0.025, 0.975])
    p_two_sided = min(
        1.0,
        2
        * min(
            (np.count_nonzero(array <= 0) + 1) / (len(array) + 1),
            (np.count_nonzero(array >= 0) + 1) / (len(array) + 1),
        ),
    )
    return {
        "positive_mean_gbox_dependence": float(positive.mean()),
        "negative_mean_gbox_dependence": float(negative.mean()),
        "positive_minus_negative_interaction": observed,
        "interaction_ci_low": float(lower),
        "interaction_ci_high": float(upper),
        "interaction_bootstrap_p_two_sided": float(p_two_sided),
    }


def build_gene_effects(
    variants: pd.DataFrame,
    probabilities: np.ndarray,
    original_probabilities: dict[str, float],
    *,
    arm: str,
    seed: int,
    genus: str,
    readout: str,
) -> pd.DataFrame:
    scored = variants.copy()
    scored["mutated_probability"] = probabilities
    scored["original_probability"] = scored["gene_id"].map(
        original_probabilities
    )
    if scored["original_probability"].isna().any():
        raise RuntimeError("missing original prediction for a variant gene")
    scored["delta_probability"] = (
        scored["mutated_probability"] - scored["original_probability"]
    )
    records: list[dict[str, object]] = []
    for gene_id, group in scored.groupby("gene_id", sort=True):
        motif = group[group["variant_type"].eq("motif_disruption")]
        controls = group[group["variant_type"].eq("matched_control")]
        if len(motif) != 1 or len(controls) != 10:
            raise RuntimeError(f"invalid perturbation bundle: {gene_id}")
        motif_row = motif.iloc[0]
        control_delta = controls["delta_probability"].to_numpy(
            dtype=np.float64
        )
        motif_delta = float(motif_row["delta_probability"])
        control_median = float(np.median(control_delta))
        records.append(
            {
                "arm": arm,
                "seed": seed,
                "readout": readout,
                "heldout_genus": genus,
                "gene_id": gene_id,
                "chromosome": motif_row["chromosome"],
                "label": motif_row["label"],
                "label_binary": int(motif_row["label_binary"]),
                "split": motif_row["split"],
                "gbox_count": int(motif_row["gbox_count"]),
                "original_probability": float(
                    motif_row["original_probability"]
                ),
                "motif_mutated_probability": float(
                    motif_row["mutated_probability"]
                ),
                "motif_delta": motif_delta,
                "control_delta_mean": float(control_delta.mean()),
                "control_delta_median": control_median,
                "control_delta_sd": float(
                    control_delta.std(ddof=1)
                ),
                "gbox_dependence": control_median - motif_delta,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    freeze = validate_freeze(root)
    variant_path = root / VARIANTS
    variant_manifest_path = root / VARIANT_MANIFEST
    if not variant_path.is_file() or not variant_manifest_path.is_file():
        raise FileNotFoundError("G-box perturbation inputs are absent")
    variant_manifest = json.loads(
        variant_manifest_path.read_text(encoding="utf-8")
    )
    variants = pd.read_parquet(variant_path).sort_values(
        ["variant_id"], kind="stable"
    ).reset_index(drop=True)
    if len(variants) != int(variant_manifest["variant_rows"]):
        raise RuntimeError("G-box perturbation row count differs")

    embedding_seed = 0 if args.arm == "base" else args.seed
    embedding_dir = (
        root
        / "results/embeddings/"
        "plantcad_dapt_publication_v3_gbox_mutagenesis"
        / args.arm
        / f"seed_{embedding_seed}"
    )
    embedding_path = embedding_dir / "variants.npy"
    embedding_spec = embedding_dir / "run_spec.json"
    if not embedding_path.is_file() or not embedding_spec.is_file():
        raise FileNotFoundError(embedding_path)
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (len(variants), 1536):
        raise RuntimeError("G-box perturbation embedding shape differs")

    probe_dir = (
        root
        / "results/metrics/"
        "plantcad_dapt_publication_v3_functional_probes"
        / args.arm
        / f"seed_{args.seed}"
    )
    required_paths: list[Path] = [
        embedding_path,
        embedding_spec,
        probe_dir / "run_spec.json",
    ]
    for genus in GENERA:
        required_paths.extend(
            [
                probe_dir / f"heldout_{genus}.linear.joblib",
                probe_dir / f"heldout_{genus}.xgboost.json",
                probe_dir
                / f"heldout_{genus}.linear.predictions.parquet",
                probe_dir
                / f"heldout_{genus}.xgboost.predictions.parquet",
            ]
        )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    fingerprint_payload = {
        "freeze_input_fingerprint": freeze["input_fingerprint"],
        "variant_sha256": sha256(variant_path),
        "embedding_sha256": sha256(embedding_path),
        "embedding_spec_sha256": sha256(embedding_spec),
        "probe_artifact_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in required_paths[2:]
        },
        "arm": args.arm,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "n_jobs": args.n_jobs,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output_dir = (
        root
        / "results/biological_cases/"
        "publication_v3_gbox_model_attribution"
        / args.arm
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    effects_path = output_dir / "gene_effects.parquet"
    summary_path = output_dir / "summary.tsv"
    run_spec_path = output_dir / "run_spec.json"
    if (
        effects_path.is_file()
        and summary_path.is_file()
        and run_spec_path.is_file()
    ):
        previous = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if previous.get("input_fingerprint") == fingerprint:
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "reused": True,
                        "arm": args.arm,
                        "seed": args.seed,
                        "gene_effects": previous["gene_effect_rows"],
                    },
                    indent=2,
                )
            )
            return

    effect_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for genus in GENERA:
        mask = variants["genus"].eq(genus).to_numpy()
        genus_variants = variants.loc[mask].reset_index(drop=True)
        genus_embeddings = np.asarray(
            embeddings[mask], dtype=np.float32
        )
        for readout in READOUTS:
            prediction_path = (
                probe_dir
                / f"heldout_{genus}.{readout}.predictions.parquet"
            )
            original = pd.read_parquet(
                prediction_path, columns=["gene_id", "probability"]
            )
            if original["gene_id"].duplicated().any():
                raise RuntimeError(
                    f"duplicate original predictions: {prediction_path}"
                )
            original_map = original.set_index("gene_id")[
                "probability"
            ].astype(float).to_dict()
            if readout == "linear":
                probability = load_linear_probability(
                    probe_dir / f"heldout_{genus}.linear.joblib",
                    genus_embeddings,
                )
            else:
                probability = load_xgboost_probability(
                    probe_dir / f"heldout_{genus}.xgboost.json",
                    genus_embeddings,
                    args.n_jobs,
                )
            effects = build_gene_effects(
                genus_variants,
                probability,
                original_map,
                arm=args.arm,
                seed=args.seed,
                genus=genus,
                readout=readout,
            )
            effect_frames.append(effects)
            stats = chromosome_block_bootstrap(
                effects,
                args.bootstrap_replicates,
                stable_seed(
                    "publication_v3_gbox_attribution_bootstrap_v1",
                    args.arm,
                    args.seed,
                    genus,
                    readout,
                ),
            )
            summaries.append(
                {
                    "arm": args.arm,
                    "seed": args.seed,
                    "readout": readout,
                    "heldout_genus": genus,
                    "genes": len(effects),
                    "positives": int(effects["label_binary"].sum()),
                    "negatives": int(
                        effects["label_binary"].eq(0).sum()
                    ),
                    **stats,
                }
            )
            print(
                f"{args.arm} seed={args.seed} {readout} "
                f"heldout={genus}: genes={len(effects)} "
                f"interaction="
                f"{stats['positive_minus_negative_interaction']:.6g}",
                flush=True,
            )

    all_effects = pd.concat(effect_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    effects_partial = effects_path.with_name(
        effects_path.name + ".partial"
    )
    effects_partial.unlink(missing_ok=True)
    all_effects.to_parquet(
        effects_partial, compression="zstd", index=False
    )
    effects_partial.replace(effects_path)
    summary.to_csv(summary_path, sep="\t", index=False)
    run_spec = {
        "status": "complete",
        **fingerprint_payload,
        "input_fingerprint": fingerprint,
        "gene_effect_rows": len(all_effects),
        "summary_rows": len(summary),
        "effects_sha256": sha256(effects_path),
        "summary_sha256": sha256(summary_path),
        "primary_quantity": (
            "median(matched control probability delta) minus "
            "exact CACGTG disruption probability delta"
        ),
        "malus_accessed": False,
    }
    run_spec_path.write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_spec, indent=2))


if __name__ == "__main__":
    main()
