#!/usr/bin/env python3
"""Extract the best frozen Pfam domain from each display-family protein."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TREE_OUT = ROOT / "results/metrics/publication_v5_gene_trees"
RUNS = TREE_OUT / "runs"
OUT = TREE_OUT / "sensitivity_domain"
HMM_DIR = ROOT / "data/raw/publication_v5_pfam_domains"
FAMILIES = {
    "OG0000025": "PF00847",
    "OG0000413": "PF00481",
    "OG0000277": "PF03055",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks)
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            records[name] = "".join(chunks)
    return records


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def composition_failure(path: Path) -> tuple[int, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"([0-9.]+)%\s+(\d+) sequences failed composition chi2 test", text)
    if not matches:
        raise RuntimeError(f"Composition summary missing from {path}")
    reported_total_percent, count = matches[-1]
    return int(count), float(reported_total_percent)


def parse_domtbl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split(maxsplit=22)
            if len(fields) < 22:
                raise RuntimeError(f"Malformed HMMER domtbl line in {path}")
            rows.append(
                {
                    "safe_id": fields[0],
                    "target_length": int(fields[2]),
                    "query_name": fields[3],
                    "query_accession": fields[4],
                    "full_evalue": float(fields[6]),
                    "full_score": float(fields[7]),
                    "domain_index": int(fields[9]),
                    "domain_count": int(fields[10]),
                    "conditional_evalue": float(fields[11]),
                    "independent_evalue": float(fields[12]),
                    "domain_score": float(fields[13]),
                    "hmm_from": int(fields[15]),
                    "hmm_to": int(fields[16]),
                    "ali_from": int(fields[17]),
                    "ali_to": int(fields[18]),
                    "env_from": int(fields[19]),
                    "env_to": int(fields[20]),
                    "accuracy": float(fields[21]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    family_rows = []
    for orthogroup, pfam in FAMILIES.items():
        work = RUNS / orthogroup
        raw_fasta = work / f"{orthogroup}.raw.faa"
        failed_count, reported_total_percent = composition_failure(work / "primary.log")

        family_out = OUT / orthogroup
        family_out.mkdir(parents=True, exist_ok=True)
        hmm_gz = HMM_DIR / f"{pfam}.hmm.gz"
        hmm = family_out / f"{pfam}.hmm"
        with gzip.open(hmm_gz, "rb") as source, hmm.open("wb") as dest:
            dest.write(source.read())
        domtbl = family_out / f"{orthogroup}.domtblout"
        stdout = family_out / "hmmsearch.stdout.log"
        stderr = family_out / "hmmsearch.stderr.log"
        with stdout.open("w", encoding="utf-8") as out_handle, stderr.open("w", encoding="utf-8") as err_handle:
            subprocess.run(
                [
                    "hmmsearch",
                    "--cut_ga",
                    "--noali",
                    "--cpu",
                    "2",
                    "--domtblout",
                    str(domtbl),
                    str(hmm),
                    str(raw_fasta),
                ],
                check=True,
                stdout=out_handle,
                stderr=err_handle,
            )

        sequences = read_fasta(raw_fasta)
        failed_percent = 100.0 * failed_count / len(sequences)
        contract_triggered = failed_percent > 20.0
        hits = parse_domtbl(domtbl)
        if hits.empty:
            raise RuntimeError(f"No Pfam hits for {orthogroup}/{pfam}")
        hits = hits.sort_values(
            ["safe_id", "independent_evalue", "domain_score", "accuracy", "ali_from"],
            ascending=[True, True, False, False, True],
        )
        best = hits.drop_duplicates("safe_id", keep="first").copy()
        extracted = []
        for row in best.to_dict("records"):
            sequence = sequences[row["safe_id"]]
            start = int(row["ali_from"])
            stop = int(row["ali_to"])
            domain_sequence = sequence[start - 1 : stop]
            if not domain_sequence:
                raise RuntimeError(f"Empty domain for {row['safe_id']}")
            extracted.append((row["safe_id"], domain_sequence))
            all_rows.append(
                {
                    "orthogroup": orthogroup,
                    "pfam": pfam,
                    **row,
                    "extracted_length": len(domain_sequence),
                }
            )
        domain_fasta = family_out / f"{orthogroup}.domain.faa"
        write_fasta(domain_fasta, extracted)
        family_rows.append(
            {
                "orthogroup": orthogroup,
                "pfam": pfam,
                "composition_failed_sequences": failed_count,
                "composition_failed_percent": failed_percent,
                "iqtree_reported_total_percent": reported_total_percent,
                "contract_domain_triggered": contract_triggered,
                "input_sequences": len(sequences),
                "sequences_with_pfam_domain": len(best),
                "domain_coverage_fraction": len(best) / len(sequences),
                "median_extracted_length": float(best.eval("ali_to - ali_from + 1").median()),
                "hmm_gz_sha256": sha256(hmm_gz),
                "domain_fasta_sha256": sha256(domain_fasta),
            }
        )

    pd.DataFrame(all_rows).to_csv(OUT / "domain_extraction.tsv", sep="\t", index=False)
    families = pd.DataFrame(family_rows).sort_values("orthogroup")
    families.to_csv(OUT / "domain_family_summary.tsv", sep="\t", index=False)
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "publication_v5_posthoc_descriptive_sensitivity",
        "decision_authority": False,
        "malus_downstream_outcomes_accessed": False,
        "trigger_percent": 20.0,
        "families_scanned": sorted(FAMILIES),
        "families_triggered": sorted(row["orthogroup"] for row in family_rows if row["contract_domain_triggered"]),
        "family_records": family_rows,
        "output_fingerprints": {
            "domain_extraction.tsv": sha256(OUT / "domain_extraction.tsv"),
            "domain_family_summary.tsv": sha256(OUT / "domain_family_summary.tsv"),
        },
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
