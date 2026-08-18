#!/usr/bin/env python3
"""Frozen two-GPU PlantCAD DAPT for publication-v3 confirmatory arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


ARMS = ("tree", "herb", "random_plant", "phylogc_match")
SEEDS = (23, 41, 59)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunSpec:
    arm: str
    seed: int
    model_dir: str
    train_file: str
    train_sha256: str
    validation_file: str
    validation_sha256: str
    output_dir: str
    max_steps: int
    sequence_length: int
    mlm_probability: float
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    required_world_size: int
    runtime_world_size: int
    effective_batch_size: int
    trained_sequences: int
    trained_bases: int
    release_gate: str


class FixedLengthMLMCollator(DataCollatorForLanguageModeling):
    """PlantCAD-compatible collator without unsupported attention masks."""

    def torch_call(
        self, examples: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        batch = {
            key: torch.stack(
                [torch.as_tensor(example[key]) for example in examples], dim=0
            )
            for key in examples[0]
        }
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"],
            special_tokens_mask=special_tokens_mask,
        )
        return batch


def absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/external_registry/PlantCaduceus_l20"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/publication_v3_dapt_hf"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/models/plantcad_dapt_publication_v3"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/plantcad_dapt_publication_v3"),
    )
    parser.add_argument(
        "--release-gate",
        type=Path,
        default=Path("config/publication_v3_dapt_pretraining_release.json"),
    )
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=2_500)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = args.project_root.resolve()
    if (root / "config/GPU_PAUSED").exists():
        raise SystemExit("GPU work is paused by config/GPU_PAUSED")
    model_dir = absolute(root, args.model_dir).resolve()
    data_root = absolute(root, args.data_root).resolve()
    output_dir = (
        absolute(root, args.output_root).resolve()
        / args.arm
        / f"seed_{args.seed}"
    )
    cache_dir = absolute(root, args.cache_dir).resolve()
    release_gate_path = absolute(root, args.release_gate).resolve()
    train_file = data_root / args.arm / "train.parquet"
    validation_file = data_root / args.arm / "validation.parquet"
    for required in (model_dir / "config.json", train_file, validation_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not args.validate_only:
        if not release_gate_path.is_file():
            raise FileNotFoundError(release_gate_path)
        release_gate = json.loads(release_gate_path.read_text(encoding="utf-8"))
        contract = release_gate.get("training_contract") or {}
        pretraining_release = bool(
            release_gate.get("status") == "pass"
            and release_gate.get("scope") == "unsupervised_dapt_only"
            and release_gate.get("gpu_pretraining_released") is True
            and release_gate.get("dapt_corpus_qc") is True
            and release_gate.get("outcome_data_permitted") is False
            and release_gate.get("analysis_freeze_still_required") is True
            and contract.get("backbone") == "PlantCaduceus_l20"
            and tuple(contract.get("arms", [])) == ARMS
            and tuple(contract.get("seeds", [])) == SEEDS
            and contract.get("max_steps_per_run") == 15_000
            and contract.get("world_size") == 2
            and contract.get("local_gpu_only") is True
            and contract.get("sequence_length") == 512
            and contract.get("mlm_probability") == args.mlm_probability
            and contract.get("learning_rate") == args.learning_rate
            and contract.get("warmup_ratio") == args.warmup_ratio
            and contract.get("weight_decay") == args.weight_decay
            and contract.get("per_device_batch_size")
            == args.per_device_batch_size
            and contract.get("gradient_accumulation_steps")
            == args.gradient_accumulation_steps
            and args.max_steps == contract.get("max_steps_per_run")
            and (
                release_gate.get("artifacts", {})
                .get("training_script", {})
                .get("sha256")
                == sha256(Path(__file__).resolve())
            )
        )
        required_full_gates = (
            "genome_qc",
            "label_qc",
            "task_qc",
            "family_qc",
            "integrity_qc",
            "dapt_corpus_qc",
            "analysis_qc",
        )
        full_release = bool(
            release_gate.get("status") == "pass"
            and release_gate.get("gpu_training_released") is True
            and all(
                release_gate.get(key) is True for key in required_full_gates
            )
        )
        if not (pretraining_release or full_release):
            raise RuntimeError(
                f"publication-v3 GPU release gate is closed: {release_gate}"
            )

    runtime_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    required_world_size = 2
    effective_batch_size = (
        args.per_device_batch_size
        * args.gradient_accumulation_steps
        * required_world_size
    )
    spec = RunSpec(
        arm=args.arm,
        seed=args.seed,
        model_dir=str(model_dir),
        train_file=str(train_file),
        train_sha256=sha256(train_file),
        validation_file=str(validation_file),
        validation_sha256=sha256(validation_file),
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        sequence_length=512,
        mlm_probability=args.mlm_probability,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        required_world_size=required_world_size,
        runtime_world_size=runtime_world_size,
        effective_batch_size=effective_batch_size,
        trained_sequences=args.max_steps * effective_batch_size,
        trained_bases=args.max_steps * effective_batch_size * 512,
        release_gate=str(release_gate_path),
    )
    if runtime_world_size not in (1, 2):
        raise RuntimeError(f"unexpected world size: {runtime_world_size}")
    if not args.validate_only:
        if runtime_world_size != required_world_size:
            raise RuntimeError("production DAPT requires exactly two DDP ranks")
        if torch.cuda.device_count() != required_world_size:
            raise RuntimeError("production DAPT must expose two local GPUs")

    set_seed(args.seed)
    raw = load_dataset(
        "parquet",
        data_files={
            "train": str(train_file),
            "validation": str(validation_file),
        },
        cache_dir=str(cache_dir / "datasets"),
    )
    if args.validate_only:
        for split in raw:
            raw[split] = raw[split].select(range(min(128, len(raw[split]))))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["seq"],
            return_special_tokens_mask=True,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )

    # torchrun starts two independent Python processes before Trainer creates
    # the distributed process group. If both ranks use the same datasets cache
    # filename, their num_proc workers can replace one another's Arrow shards.
    # Rank-specific deterministic caches avoid that race while retaining safe
    # restart reuse for each rank.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cache_mode = "validation" if args.validate_only else "production"
    token_cache_dir = cache_dir / "tokenized" / args.arm / cache_mode
    token_cache_dir.mkdir(parents=True, exist_ok=True)
    tokenized = raw.map(
        tokenize,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=raw["train"].column_names,
        cache_file_names={
            split: str(
                token_cache_dir / f"{split}_rank_{local_rank}.arrow"
            )
            for split in raw
        },
        desc=f"Tokenizing publication-v3 {args.arm}",
    )
    lengths = {
        len(tokenized[split][0]["input_ids"])
        for split in ("train", "validation")
    }
    if lengths != {512}:
        raise RuntimeError(f"PlantCAD token length mismatch: {sorted(lengths)}")
    collator = FixedLengthMLMCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    if args.validate_only:
        test_batch = collator([tokenized["train"][0], tokenized["train"][1]])
        if (
            set(test_batch) != {"input_ids", "labels"}
            or tuple(test_batch["input_ids"].shape) != (2, 512)
        ):
            raise RuntimeError(
                f"unexpected batch: keys={sorted(test_batch)}, "
                f"shape={tuple(test_batch['input_ids'].shape)}"
            )
        print(
            json.dumps(
                {"status": "validated", "run_spec": asdict(spec)}, indent=2
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        fp16=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        save_safetensors=False,
        logging_steps=args.logging_steps,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    if trainer.is_world_process_zero():
        (output_dir / "run_spec.json").write_text(
            json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8"
        )
    checkpoint = get_last_checkpoint(str(output_dir))
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        for source in model_dir.glob("*.py"):
            shutil.copy2(source, final_dir / source.name)
    metrics = dict(train_result.metrics)
    metrics.update(trainer.evaluate())
    trainer.log_metrics("final", metrics)
    trainer.save_metrics("final", metrics)
    trainer.save_state()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
