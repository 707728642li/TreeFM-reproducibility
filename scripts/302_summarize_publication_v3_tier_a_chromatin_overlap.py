#!/usr/bin/env python3
"""Post hoc exact-gene overlap of frozen Tier-A candidates and H3K4me3 calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import beta, fisher_exact


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binomial_exact_interval(successes: int, total: int) -> list[float]:
    low = 0.0 if successes == 0 else float(
        beta.ppf(0.025, successes, total - successes + 1)
    )
    high = 1.0 if successes == total else float(
        beta.ppf(0.975, successes + 1, total - successes)
    )
    return [low, high]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    candidate_path = (
        root
        / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv"
    )
    chromatin_path = (
        root
        / "results/biological_cases/prunus_publication_v3_chromatin_replication/gse190586_binary_gene_calls.tsv.gz"
    )
    candidates = pd.read_csv(candidate_path, sep="\t")
    calls = pd.read_csv(chromatin_path, sep="\t")
    if len(candidates) != 14 or set(candidates["tier"].astype(str)) != {"A"}:
        raise RuntimeError("expected the 14 frozen Tier-A orthogroups")
    genes = (
        candidates[
            [
                "catalog_rank",
                "orthogroup",
                "arabidopsis_symbols",
                "prunus_gene_ids",
                "prunus_directions",
            ]
        ]
        .assign(gene_id=lambda frame: frame["prunus_gene_ids"].str.split(";"))
        .explode("gene_id")
        .drop(columns="prunus_gene_ids")
    )
    if len(genes) != 34 or genes["gene_id"].nunique() != 34:
        raise RuntimeError(
            f"expected 34 unique Tier-A Prunus genes, observed {len(genes)}"
        )
    joined = genes.merge(
        calls,
        on="gene_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"].ne("both"), "gene_id"].tolist()
        raise RuntimeError(f"Tier-A genes missing from chromatin table: {missing}")
    joined = joined.drop(columns="_merge")
    if set(joined["label"].astype(str)) != {"positive"}:
        raise RuntimeError("all Tier-A genes must retain the frozen positive label")
    if not joined["prunus_directions"].eq("down").all():
        raise RuntimeError("all Tier-A Prunus endpoint directions must be down")

    positives = calls.loc[calls["label"].eq("positive")].copy()
    positive_total = len(positives)
    positive_concordant = int(positives["concordant_positive"].astype(bool).sum())
    if positive_total != 487 or positive_concordant != 54:
        raise RuntimeError(
            "frozen chromatin totals changed: "
            f"{positive_concordant}/{positive_total}"
        )
    tier_total = len(joined)
    tier_concordant = int(joined["concordant_positive"].astype(bool).sum())
    other_total = positive_total - tier_total
    other_concordant = positive_concordant - tier_concordant
    odds_ratio, fisher_p = fisher_exact(
        [
            [tier_concordant, tier_total - tier_concordant],
            [other_concordant, other_total - other_concordant],
        ],
        alternative="two-sided",
    )

    output = (
        root
        / "results/biological_cases/publication_v3_tier_a_chromatin_overlap"
    )
    output.mkdir(parents=True, exist_ok=True)
    gene_columns = [
        "catalog_rank",
        "orthogroup",
        "arabidopsis_symbols",
        "gene_id",
        "endpoint_direction",
        "chromatin_call",
        "concordant_positive",
        "eligible_peak_count",
        "max_peak_score",
        "min_abs_distance_to_tss",
    ]
    gene_table = joined[gene_columns].sort_values(
        ["catalog_rank", "gene_id"]
    )
    gene_path = output / "tier_a_gene_chromatin_overlap.tsv"
    gene_table.to_csv(gene_path, sep="\t", index=False)
    orthogroup_table = (
        joined.groupby(
            ["catalog_rank", "orthogroup", "arabidopsis_symbols"],
            sort=True,
            as_index=False,
        )
        .agg(
            prunus_genes=("gene_id", "size"),
            direction_concordant_h3k4me3=("concordant_positive", "sum"),
            supported_gene_ids=(
                "gene_id",
                lambda values: ";".join(
                    joined.loc[
                        joined["gene_id"].isin(values)
                        & joined["concordant_positive"].astype(bool),
                        "gene_id",
                    ].astype(str)
                ),
            ),
        )
    )
    orthogroup_path = output / "tier_a_orthogroup_chromatin_overlap.tsv"
    orthogroup_table.to_csv(orthogroup_path, sep="\t", index=False)

    supported = gene_table.loc[gene_table["concordant_positive"].astype(bool)]
    summary = {
        "status": "complete_posthoc_descriptive",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen_tier_a_exact_prunus_gene_by_independent_h3k4me3_overlap",
        "posthoc": True,
        "selection_authority": False,
        "tier_a_orthogroups": 14,
        "tier_a_prunus_genes": tier_total,
        "tier_a_direction_concordant_h3k4me3_genes": tier_concordant,
        "tier_a_fraction": tier_concordant / tier_total,
        "tier_a_fraction_exact_95ci": binomial_exact_interval(
            tier_concordant, tier_total
        ),
        "all_positive_genes": positive_total,
        "all_positive_direction_concordant_h3k4me3_genes": positive_concordant,
        "all_positive_fraction": positive_concordant / positive_total,
        "tier_a_vs_other_positive_fisher_odds_ratio": float(odds_ratio),
        "tier_a_vs_other_positive_fisher_two_sided_p": float(fisher_p),
        "enrichment_claim_supported": bool(fisher_p < 0.05 and odds_ratio > 1),
        "supported_exact_genes": supported[
            ["catalog_rank", "orthogroup", "gene_id", "arabidopsis_symbols"]
        ].to_dict(orient="records"),
        "interpretation": (
            "Four exact Tier-A Prunus genes have independent direction-concordant "
            "promoter H3K4me3 support. The Tier-A fraction is not enriched versus "
            "other positive genes, so this is candidate-level prioritization, not "
            "validation of the Tier-A set as a whole."
        ),
        "input_sha256": {
            str(candidate_path.relative_to(root)): sha256(candidate_path),
            str(chromatin_path.relative_to(root)): sha256(chromatin_path),
        },
        "output_sha256": {
            str(gene_path.relative_to(root)): sha256(gene_path),
            str(orthogroup_path.relative_to(root)): sha256(orthogroup_path),
        },
        "malus_accessed": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    supported_lines = [
        f"- Rank {int(row.catalog_rank)} `{row.orthogroup}` / `{row.gene_id}`"
        for row in supported.itertuples(index=False)
    ]
    report = f"""# Tier-A 候选与独立 H3K4me3 的精确基因交集

生成时间：{summary['created_utc']}

## 结论

这是候选冻结后的**事后描述性分析**，不参与候选选择或模型门槛。14 个 Tier-A orthogroup 包含 34 个 Prunus 阳性基因，其中 {tier_concordant} 个具有方向一致的独立启动子 H3K4me3 call（{100*tier_concordant/tier_total:.2f}%），精确二项 95% CI 为 {100*summary['tier_a_fraction_exact_95ci'][0]:.2f}%–{100*summary['tier_a_fraction_exact_95ci'][1]:.2f}%。

全部 487 个阳性基因中有 54 个获得相同支持（11.09%）。Tier-A 与其余阳性基因相比的 Fisher 双侧检验 OR={odds_ratio:.3f}、P={fisher_p:.4g}，因此**不支持 Tier-A 集合整体额外富集**。它的价值是为 4 个精确候选提供独立优先级，而不是再次验证 Tier-A 选择规则。

## 获得精确基因支持的候选

{chr(10).join(supported_lines)}

这些基因分别涉及 AP2/ERF、PP2C、bHLH 和 GAPDH 家族。家族注释与染色质方向的一致仍不等同于结合、调控或休眠因果作用。
"""
    report_path = root / "reports/PUBLICATION_V3_TIER_A_CHROMATIN_OVERLAP_20260802_CN.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
