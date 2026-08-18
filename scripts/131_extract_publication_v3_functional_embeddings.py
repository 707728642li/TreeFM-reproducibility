#!/usr/bin/env python3
"""Extract frozen four-chunk functional embeddings for Prunus and Pyrus."""

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
CONFIRMATORY_SEEDS = (23, 41, 59)
GENUS_DATASETS = {
    "prunus": Path(
        "data/processed/functional/Prunus_publication_v3/"
        "promoter_labels.parquet"
    ),
    "pyrus": Path(
        "data/processed/functional/Pyrus_PRJNA669907/"
        "promoter_labels.parquet"
    ),
}
CHUNKS = tuple(f"promoter_chunk_{index}" for index in range(1, 5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def normalize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "gene_id",
        "chromosome",
        "promoter_2048",
        "label_binary",
        "label",
        "split",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"functional dataset lacks columns: {missing}")
    data = data.copy()
    if data["gene_id"].duplicated().any():
        raise RuntimeError("functional dataset contains duplicate genes")
    if set(data["label_binary"].dropna().astype(int)) != {0, 1}:
        raise RuntimeError("functional dataset requires both binary classes")
    if not data["promoter_2048"].str.len().eq(2048).all():
        raise RuntimeError("functional promoters must be exactly 2,048 bp")
    for index, column in enumerate(CHUNKS):
        expected = data["promoter_2048"].str.slice(
            index * 512, (index + 1) * 512
        )
        if column in data:
            if not data[column].eq(expected).all():
                raise RuntimeError(f"{column} disagrees with promoter_2048")
        else:
            data[column] = expected
    return data.sort_values(
        ["chromosome", "gene_id"], kind="stable"
    ).reset_index(drop=True)


def row_hash(data: pd.DataFrame) -> str:
    columns = [
        "gene_id",
        "chromosome",
        "label_binary",
        "label",
        "split",
        *CHUNKS,
    ]
    values = pd.util.hash_pandas_object(
        data[columns], index=False
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


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
            input_ids = encoded["input_ids"].to("cuda", non_blocking=True)
            if tuple(input_ids.shape)[1] != 512:
                raise RuntimeError(
                    f"unexpected functional token shape: {tuple(input_ids.shape)}"
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
        raise ValueError("Base functional embeddings use seed 0")
    if args.arm != "base" and args.seed not in CONFIRMATORY_SEEDS:
        raise ValueError("DAPT seed must be 23, 41 or 59")
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        raise RuntimeError("functional embedding extraction is local-GPU-only")
    kernel_release = Path("/proc/sys/kernel/osrelease")
    if (
        kernel_release.is_file()
        and "microsoft" in kernel_release.read_text(encoding="utf-8").lower()
        and os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise RuntimeError(
            "WSL functional embedding extraction is restricted to physical GPU0; "
            "set CUDA_VISIBLE_DEVICES=0"
        )
    root = args.project_root.resolve()
    if (root / "config/GPU_PAUSED").exists():
        raise RuntimeError("GPU work is paused by config/GPU_PAUSED")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one local GPU per extractor")

    model_dir = resolve_model(root, args.arm, args.seed)
    output_dir = (
        root
        / "results/embeddings/plantcad_dapt_publication_v3_functional"
        / args.arm
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (
        model_dir / "config.json",
        model_dir / "pytorch_model.bin",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

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
    model_sha = sha256(model_dir / "pytorch_model.bin")
    records: list[dict[str, object]] = []
    for genus, relative_source in GENUS_DATASETS.items():
        source = root / relative_source
        if not source.is_file():
            raise FileNotFoundError(source)
        data = normalize_dataset(pd.read_parquet(source))
        expected_hash = row_hash(data)
        output = output_dir / f"{genus}.npy"
        sidecar = output.with_suffix(".json")
        reusable = False
        if output.is_file() and sidecar.is_file():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            array = np.load(output, mmap_mode="r")
            reusable = bool(
                saved.get("row_hash") == expected_hash
                and saved.get("model_weight_sha256") == model_sha
                and array.shape == (len(data), 1536)
            )
        if not reusable:
            chunks = []
            for column in CHUNKS:
                print(
                    f"{args.arm} seed={args.seed} {genus} {column}: "
                    f"rows={len(data)}",
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
                    f"invalid functional embeddings for {genus}: "
                    f"{embeddings.shape}"
                )
            partial = output.with_name(output.name + ".partial.npy")
            partial.unlink(missing_ok=True)
            np.save(partial, embeddings, allow_pickle=False)
            os.replace(partial, output)
            sidecar.write_text(
                json.dumps(
                    {
                        "genus": genus,
                        "rows": len(data),
                        "dimensions": 1536,
                        "row_hash": expected_hash,
                        "model_weight_sha256": model_sha,
                        "source": str(relative_source),
                        "chunks": list(CHUNKS),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        records.append(
            {
                "arm": args.arm,
                "seed": args.seed,
                "genus": genus,
                "rows": len(data),
                "dimensions": 1536,
                "row_hash": expected_hash,
                "embedding_file": str(output.relative_to(root)),
                "embedding_sha256": sha256(output),
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "manifest.tsv", sep="\t", index=False)
    run_spec = {
        "status": "complete",
        "contract": "docs/publication_v3_functional_analysis_contract.md",
        "arm": args.arm,
        "seed": args.seed,
        "model_dir": str(model_dir.relative_to(root)),
        "model_weight_sha256": model_sha,
        "token_index": args.token_index,
        "dimensions": 1536,
        "pooling": (
            "four 512-bp chunks; center-token RCPS half average; concatenated"
        ),
        "visible_gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (output_dir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_spec, indent=2))


if __name__ == "__main__":
    main()
