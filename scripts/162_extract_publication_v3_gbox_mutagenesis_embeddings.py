#!/usr/bin/env python3
"""Extract frozen embeddings for fixed-G-box promoter perturbations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)
CHUNKS = tuple(f"promoter_chunk_{index}" for index in range(1, 5))
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


def resolve_model(root: Path, arm: str, seed: int) -> Path:
    if arm == "base":
        return root / "models/external_registry/PlantCaduceus_l20"
    return (
        root
        / "results/models/plantcad_dapt_publication_v3"
        / arm
        / f"seed_{seed}"
        / "final"
    )


def normalize_variants(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant_id",
        "genus",
        "gene_id",
        "chromosome",
        "label_binary",
        "variant_type",
        "control_replicate",
        "gbox_count",
        "promoter_2048",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"perturbation dataset lacks columns: {missing}")
    data = data.copy().sort_values(
        ["variant_id"], kind="stable"
    ).reset_index(drop=True)
    if data["variant_id"].duplicated().any():
        raise RuntimeError("perturbation variant identifiers are not unique")
    if set(data["genus"]) != {"prunus", "pyrus"}:
        raise RuntimeError("perturbation dataset has unexpected genera")
    if not data["promoter_2048"].str.len().eq(2048).all():
        raise RuntimeError("perturbed promoters are not all 2,048 bp")
    for index, column in enumerate(CHUNKS):
        data[column] = data["promoter_2048"].str.slice(
            index * 512, (index + 1) * 512
        )
    return data


def extract_chunk(
    model: torch.nn.Module,
    tokenizer: object,
    sequences: pd.Series,
    batch_size: int,
    token_index: int,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences.iloc[start : start + batch_size].tolist()
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = encoded["input_ids"].to(
                "cuda", non_blocking=True
            )
            if tuple(input_ids.shape)[1] != 512:
                raise RuntimeError(
                    f"unexpected perturbation token shape: "
                    f"{tuple(input_ids.shape)}"
                )
            hidden = model(
                input_ids=input_ids, output_hidden_states=True
            ).hidden_states[-1][:, token_index, :].float()
            if hidden.shape[-1] % 2:
                raise RuntimeError("RCPS hidden dimension is not even")
            half = hidden.shape[-1] // 2
            pooled = (
                hidden[:, :half] + hidden[:, half:].flip(dims=(-1,))
            ) / 2
            blocks.append(pooled.cpu().numpy().astype(np.float16))
    return np.concatenate(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--token-index", type=int, default=255)
    args = parser.parse_args()
    if args.arm == "base" and args.seed != 0:
        raise ValueError("Base perturbation embeddings require seed 0")
    if args.arm != "base" and args.seed not in SEEDS:
        raise ValueError("DAPT seed must be 23, 41 or 59")
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        raise RuntimeError("G-box embedding extraction is local-GPU-only")
    root = args.project_root.resolve()
    if (root / "config/GPU_PAUSED").exists():
        raise RuntimeError("GPU work is paused by config/GPU_PAUSED")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one local GPU per extractor")
    freeze = validate_freeze(root)

    variant_path = root / VARIANTS
    manifest_path = root / VARIANT_MANIFEST
    if not variant_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("G-box perturbation inputs are absent")
    variant_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    data = normalize_variants(pd.read_parquet(variant_path))
    if len(data) != int(variant_manifest["variant_rows"]):
        raise RuntimeError("perturbation row count differs from manifest")
    if sha256(variant_path) != variant_manifest["output_sha256"]:
        raise RuntimeError("perturbation parquet hash differs from manifest")

    model_dir = resolve_model(root, args.arm, args.seed)
    for required in (
        model_dir / "config.json",
        model_dir / "pytorch_model.bin",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    model_sha = sha256(model_dir / "pytorch_model.bin")
    output_dir = (
        root
        / "results/embeddings/"
        "plantcad_dapt_publication_v3_gbox_mutagenesis"
        / args.arm
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "variants.npy"
    sidecar = output_dir / "variants.json"

    reusable = False
    if output.is_file() and sidecar.is_file():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        array = np.load(output, mmap_mode="r")
        reusable = bool(
            saved.get("variant_row_hash") == variant_manifest["row_hash"]
            and saved.get("model_weight_sha256") == model_sha
            and saved.get("freeze_input_fingerprint")
            == freeze["input_fingerprint"]
            and array.shape == (len(data), 1536)
        )
    if not reusable:
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForMaskedLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        ).to("cuda")
        chunks = []
        for column in CHUNKS:
            print(
                f"{args.arm} seed={args.seed} {column}: rows={len(data)}",
                flush=True,
            )
            chunks.append(
                extract_chunk(
                    model,
                    tokenizer,
                    data[column],
                    args.batch_size,
                    args.token_index,
                )
            )
        embeddings = np.concatenate(chunks, axis=1)
        if (
            embeddings.shape != (len(data), 1536)
            or not np.isfinite(embeddings).all()
        ):
            raise RuntimeError(
                f"invalid perturbation embeddings: {embeddings.shape}"
            )
        partial = output.with_name(output.name + ".partial.npy")
        partial.unlink(missing_ok=True)
        np.save(partial, embeddings, allow_pickle=False)
        os.replace(partial, output)
        del model
        torch.cuda.empty_cache()
        sidecar.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "arm": args.arm,
                    "seed": args.seed,
                    "rows": len(data),
                    "dimensions": 1536,
                    "variant_row_hash": variant_manifest["row_hash"],
                    "variant_sha256": sha256(variant_path),
                    "model_weight_sha256": model_sha,
                    "freeze_input_fingerprint": freeze[
                        "input_fingerprint"
                    ],
                    "token_index": args.token_index,
                    "pooling": (
                        "four 512-bp chunks; center-token RCPS half "
                        "average; concatenated"
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    result = {
        "status": "complete",
        "reused": reusable,
        "arm": args.arm,
        "seed": args.seed,
        "rows": len(data),
        "dimensions": 1536,
        "embedding": str(output.relative_to(root)),
        "embedding_sha256": sha256(output),
        "model_weight_sha256": model_sha,
        "malus_accessed": False,
    }
    (output_dir / "run_spec.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
