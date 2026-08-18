#!/usr/bin/env python3
"""Extract frozen center-token embeddings for the 26-species v3 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
CONFIRMATORY_SEEDS = (23, 41, 59)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_hash(data: pd.DataFrame) -> str:
    preferred = [
        "pair_id",
        "slug",
        "task",
        "gene_id",
        "label",
        "sequence",
        "family_transfer_class",
        "orthogroup",
        "exact_task_train_sequence",
        "maximum_task_train_identity",
    ]
    columns = [column for column in preferred if column in data.columns]
    hashed = pd.util.hash_pandas_object(
        data[columns], index=False
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


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


def extract(
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
                raise RuntimeError(f"unexpected token shape: {tuple(input_ids.shape)}")
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][:, token_index, :].float()
            if embedding.shape[-1] % 2:
                raise RuntimeError("RCPS hidden dimension is not even")
            half = embedding.shape[-1] // 2
            pooled = (
                embedding[:, :half] + embedding[:, half:].flip(dims=(-1,))
            ) / 2
            blocks.append(pooled.cpu().numpy().astype(np.float16))
    return np.concatenate(blocks)


def embedding_is_reusable(
    output: Path,
    sidecar: Path,
    expected_row_hash: str,
    model_sha: str,
    expected_rows: int,
) -> bool:
    if not output.is_file() or not sidecar.is_file():
        return False
    try:
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        array = np.load(output, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        saved.get("row_hash") == expected_row_hash
        and saved.get("model_weight_sha256") == model_sha
        and array.shape == (expected_rows, 384)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--token-index", type=int, default=255)
    parser.add_argument(
        "--reverse-panel",
        action="store_true",
        help="Process species in reverse panel order for cooperative dual-GPU extraction.",
    )
    parser.add_argument(
        "--cooperative-worker",
        action="store_true",
        help="Do not overwrite the primary worker's run specification.",
    )
    args = parser.parse_args()
    if args.arm == "base" and args.seed != 0:
        raise ValueError("the base embedding is stored once with seed 0")
    if args.arm != "base" and args.seed not in CONFIRMATORY_SEEDS:
        raise ValueError("DAPT embedding seed must be 23, 41 or 59")
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        raise RuntimeError("embedding extraction is local-GPU-only")
    kernel_release = Path("/proc/sys/kernel/osrelease")
    if (
        kernel_release.is_file()
        and "microsoft" in kernel_release.read_text(encoding="utf-8").lower()
        and os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise RuntimeError(
            "WSL embedding extraction is restricted to physical GPU0; "
            "set CUDA_VISIBLE_DEVICES=0"
        )
    root = args.project_root.resolve()
    if (root / "config/GPU_PAUSED").exists():
        raise RuntimeError("GPU work is paused by config/GPU_PAUSED")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one local GPU for each extractor")

    panel_path = root / "config/publication_v3_technical_panel_26.tsv"
    benchmark_root = (
        root / "data/processed/technical_benchmarks_publication_v3_26"
    )
    model_dir = resolve_model(root, args.arm, args.seed)
    output_dir = (
        root
        / "results/embeddings/plantcad_dapt_publication_v3"
        / args.arm
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (
        panel_path,
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
    panel = pd.read_csv(panel_path, sep="\t", dtype=str).query("include == '1'")
    if len(panel) != 26:
        raise RuntimeError(f"expected 26 technical species, observed {len(panel)}")
    model_sha = sha256(model_dir / "pytorch_model.bin")
    run_spec = {
        "arm": args.arm,
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_weight_sha256": model_sha,
        "benchmark_root": str(benchmark_root),
        "token_index": args.token_index,
        "batch_size": args.batch_size,
        "rcps_pooling": "average_forward_and_reversed_reverse_half",
        "cooperative_species_locking": True,
        "storage_dtype": "float16",
        "visible_gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if not args.cooperative_worker:
        (output_dir / "run_spec.json").write_text(
            json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
        )

    panel_order = {slug: index for index, slug in enumerate(panel["slug"])}
    panel_items = list(panel.itertuples(index=False))
    if args.reverse_panel:
        panel_items.reverse()
    records: list[dict[str, object]] = []
    for item in panel_items:
        source = benchmark_root / f"{item.slug}.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        data = pd.read_parquet(source)
        expected_row_hash = row_hash(data)
        output = output_dir / f"{item.slug}.npy"
        sidecar = output.with_suffix(".json")
        reusable = embedding_is_reusable(
            output, sidecar, expected_row_hash, model_sha, len(data)
        )
        computed = False
        lock = output.with_suffix(".lock")
        while not reusable:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
            except FileExistsError:
                # A single 200k-row species can take tens of minutes. Only
                # clear genuinely stale locks left by an interrupted process.
                try:
                    lock_age = time.time() - lock.stat().st_mtime
                except FileNotFoundError:
                    lock_age = 0
                if lock_age > 2 * 60 * 60:
                    lock.unlink(missing_ok=True)
                elif lock.is_file():
                    time.sleep(5)
                reusable = embedding_is_reusable(
                    output, sidecar, expected_row_hash, model_sha, len(data)
                )
                continue
            try:
                # Recheck after acquiring the lock because another worker may
                # have completed this species while we were waiting.
                reusable = embedding_is_reusable(
                    output, sidecar, expected_row_hash, model_sha, len(data)
                )
                if reusable:
                    continue
                embeddings = extract(
                    model,
                    tokenizer,
                    data["sequence"],
                    args.batch_size,
                    args.token_index,
                )
                if (
                    embeddings.shape != (len(data), 384)
                    or not np.isfinite(embeddings).all()
                ):
                    raise RuntimeError(
                        f"invalid embedding matrix for {item.slug}: {embeddings.shape}"
                    )
                partial = output.with_name(output.name + ".partial.npy")
                partial.unlink(missing_ok=True)
                np.save(partial, embeddings, allow_pickle=False)
                os.replace(partial, output)
                sidecar.write_text(
                    json.dumps(
                        {
                            "slug": item.slug,
                            "rows": len(data),
                            "dimensions": 384,
                            "row_hash": expected_row_hash,
                            "model_weight_sha256": model_sha,
                            "source": str(source),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                computed = True
                reusable = True
            finally:
                lock.unlink(missing_ok=True)
        action = "extracted" if computed else "reused"
        print(f"{item.slug}: {action} {len(data)} rows", flush=True)
        array = np.load(output, mmap_mode="r")
        records.append(
            {
                "arm": args.arm,
                "seed": args.seed,
                "slug": item.slug,
                "downstream_role": item.downstream_role,
                "analysis_tier": item.analysis_tier,
                "rows": array.shape[0],
                "dimensions": array.shape[1],
                "row_hash": expected_row_hash,
                "embedding_file": str(output.relative_to(root)),
                "embedding_sha256": sha256(output),
            }
        )
    manifest = pd.DataFrame(records)
    manifest["_panel_order"] = manifest["slug"].map(panel_order)
    manifest = manifest.sort_values("_panel_order").drop(columns="_panel_order")
    manifest.to_csv(output_dir / "manifest.tsv", sep="\t", index=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": args.arm,
                "seed": args.seed,
                "species": len(records),
                "rows": int(manifest["rows"].sum()),
                "model_sha256": model_sha,
            }
        )
    )


if __name__ == "__main__":
    main()
