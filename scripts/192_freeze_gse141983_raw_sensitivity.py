#!/usr/bin/env python3
"""Freeze GSE141983 raw-sensitivity code, inputs, references and environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    "docs/publication_v3_gse141983_raw_h3k4me3_sensitivity_contract.md",
    "docs/publication_v3_exploratory_prunus_chromatin_replication_contract.md",
    "config/publication_v3_exploratory_prunus_chromatin_replication_freeze.json",
    "config/publication_v3_prunus_chromatin_binary_implementation_freeze.json",
    "metadata/publication_v3/gse141983_raw_sensitivity/GSE141983_family.soft.gz",
    "metadata/publication_v3/gse141983_raw_sensitivity/SRP237509_ENA_runinfo.tsv",
    "metadata/publication_v3/gse141983_raw_sensitivity/eligible_h3_h3k4me3_runs.tsv",
    "results/metrics/publication_v3_gse141983_raw_metadata_audit.json",
    "data/raw/functional_v3/GSE141983_raw_sensitivity/reference/"
    "Prunus_persica_NCBIv2.fa",
    "data/raw/functional_v3/GSE141983_raw_sensitivity/reference/"
    "Prunus_persica_NCBIv2.gff",
    "data/raw/functional_v3/GSE141983_raw_sensitivity/reference/"
    "Prunus_persica_NCBIv2_TSS_minus1000_plus999.saf",
    "data/raw/functional_v3/GSE141983_raw_sensitivity/reference/reference_manifest.json",
    "data/processed/functional/Prunus_publication_v3/promoter_labels.parquet",
    "results/biological_cases/prunus_publication_v3_chromatin_replication/"
    "gse190586_binary_gene_calls.tsv.gz",
    "results/biological_cases/prunus_publication_v3_chromatin_replication/summary.json",
    "scripts/185_prepare_gse141983_raw_sensitivity_metadata.py",
    "scripts/186_prepare_gse141983_raw_sensitivity_reference.py",
    "scripts/187_prefetch_gse141983_raw_chipseq.py",
    "scripts/188_process_gse141983_raw_chipseq_sample.py",
    "scripts/189_analyze_gse141983_raw_h3k4me3.py",
    "scripts/190_run_gse141983_raw_chipseq_pipeline.py",
    "scripts/191_verify_gse141983_raw_h3k4me3.py",
    "scripts/192_freeze_gse141983_raw_sensitivity.py",
    "scripts/193_prefetch_gse141983_ena_fastq_local.py",
    "scripts/194_watch_relay_gse141983_fastq.ps1",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw_root = root / "data/raw/functional_v3/GSE141983_raw_sensitivity"
    forbidden_patterns = ("*.fastq", "*.fastq.gz", "*.sra", "*.bam", "*.cram")
    forbidden = sorted(
        str(path.relative_to(root))
        for pattern in forbidden_patterns
        for path in raw_root.rglob(pattern)
    )
    if forbidden:
        raise RuntimeError(
            "quantitative GSE141983 raw files existed before freeze: " + ",".join(forbidden)
        )
    downstream_root = (
        root / "results/biological_cases/prunus_publication_v3_gse141983_raw"
    )
    if downstream_root.exists() and any(downstream_root.rglob("*.bam")):
        raise RuntimeError("GSE141983 raw alignment existed before freeze")

    metadata_audit = json.loads(
        (root / "results/metrics/publication_v3_gse141983_raw_metadata_audit.json").read_text(
            encoding="utf-8-sig"
        )
    )
    reference_manifest = json.loads(
        (
            raw_root / "reference/reference_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    if metadata_audit.get("status") != "pass" or reference_manifest.get("status") != "pass":
        raise RuntimeError("outcome-free metadata/reference gate did not pass")
    if metadata_audit.get("malus_accessed") is not False or reference_manifest.get(
        "malus_accessed"
    ) is not False:
        raise RuntimeError("Malus seal invariant failed before raw-sensitivity freeze")

    conda = Path("/home/codexli/miniconda3/bin/conda")
    environment_root = root / "metadata/publication_v3/gse141983_raw_sensitivity"
    environment_root.mkdir(parents=True, exist_ok=True)
    environment_locks: dict[str, str] = {}
    for name in ("treefm-bio", "treefm-chip"):
        prefix = root / "envs" / name
        lock = environment_root / f"{name}.explicit.txt"
        lock.write_text(
            capture([str(conda), "list", "-p", str(prefix), "--explicit"]) + "\n",
            encoding="utf-8",
        )
        environment_locks[str(lock.relative_to(root))] = sha256_file(lock)

    bio = root / "envs/treefm-bio/bin"
    chip = root / "envs/treefm-chip/bin"
    versions: dict[str, Any] = {
        "treefm_bio_python": capture([str(bio / "python"), "--version"]),
        "sra_tools": capture([str(bio / "prefetch"), "--version"]),
        "samtools": capture([str(bio / "samtools"), "--version"]).splitlines()[0],
        "bedtools": capture([str(bio / "bedtools"), "--version"]),
        "fastp": capture([str(bio / "fastp"), "--version"]),
        "featureCounts": capture([str(bio / "featureCounts"), "-v"]),
        "bowtie2": capture([str(chip / "bowtie2"), "--version"]).splitlines()[0],
        "pigz": capture(["/usr/bin/pigz", "--version"]),
        "python_packages": capture(
            [
                str(bio / "python"),
                "-c",
                "import numpy,pandas,pyarrow,scipy; "
                "print('numpy='+numpy.__version__); "
                "print('pandas='+pandas.__version__); "
                "print('pyarrow='+pyarrow.__version__); "
                "print('scipy='+scipy.__version__)",
            ]
        ).splitlines(),
    }
    versions_path = environment_root / "software_versions.json"
    versions_path.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")

    artifacts = {relative: sha256_file(root / relative) for relative in ARTIFACTS}
    artifacts.update(environment_locks)
    artifacts[str(versions_path.relative_to(root))] = sha256_file(versions_path)
    fingerprint_payload = {
        "artifacts": artifacts,
        "samples": 12,
        "permutations": 10_000,
        "bootstraps": 2_000,
        "seeds": [20260728, 20260729, 20260730],
        "technical_thresholds": [10_000_000, 0.80, 0.50, 1_000_000, 10_000, 0.70],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    freeze = {
        "status": "frozen",
        "manifest_version": 1,
        "scope": "posthoc_gse141983_raw_h3_h3k4me3_sensitivity",
        "posthoc": True,
        "artifact_sha256": artifacts,
        "input_fingerprint": fingerprint,
        "eligible_runs": 12,
        "eligible_fastq_files": 24,
        "expected_fastq_bytes_from_ena": metadata_audit["total_fastq_bytes"],
        "reference": reference_manifest["reference"],
        "technical_thresholds": {
            "post_fastp_pairs_min": 10_000_000,
            "post_fastp_q30_rate_min": 0.80,
            "mapq30_nonduplicate_pair_fraction_min": 0.50,
            "assigned_tss_fragments_min": 1_000_000,
            "nonzero_tss_genes_min": 10_000,
            "replicate_common_nonzero_genes_min": 10_000,
            "replicate_spearman_min": 0.70,
            "label_mapping_fraction_min": 0.70,
        },
        "permutations": 10_000,
        "bootstraps": 2_000,
        "permutation_seed": 20260728,
        "bootstrap_seed": 20260729,
        "gbox_permutation_seed": 20260730,
        "primary_gse190586_reclassification_allowed": False,
        "pyrus_primary_mechanism_rescue_allowed": False,
        "dapt_model_selection_allowed": False,
        "server_gpu_allowed": False,
        "malus_accessed": False,
        "malus_access_allowed": False,
        "raw_quantitative_files_present_before_freeze": False,
        "gse141983_label_association_inspected_before_freeze": False,
        "allowed_transport": [
            "exact_ENA_gzip_local_relay_with_pre_and_post_MD5",
            "NCBI_SRA_prefetch_vdb_validate_fasterq_fallback",
        ],
    }
    output = root / "config/publication_v3_gse141983_raw_sensitivity_freeze.json"
    output.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
