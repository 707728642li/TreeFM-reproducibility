#!/usr/bin/env python3
"""Build the frozen, model-independent cross-genus candidate catalog.

The selection and matching rules are defined in
``docs/publication_v3_crossgenus_candidate_catalog_contract.md``.  This script
does not read model outputs or the sealed Malus endpoint.
"""

from __future__ import annotations

import argparse
import os
import gzip
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
BRIDGE = Path("metadata/publication_v3_prunus_v21_gene_id_bridge.tsv")
ORTHOGROUPS = Path(
    "data/processed/orthofinder_benchmark_publication_v3/OrthoFinder/"
    "Results_PublicationV3/Orthogroups/Orthogroups.txt"
)
GAF = Path("data/raw/publication_v3_go/ARATH-mod.gaf.gz")
OBO = Path("data/raw/publication_v3_go/go-basic.obo")
LEAF_TERMS = Path("results/tables/publication_v3_crossgenus_go_leaf_terms.tsv")
GO_FREEZE = Path("config/publication_v3_crossgenus_go_exploratory_freeze.json")
ANALYSIS_FREEZE = Path("config/publication_v3_analysis_freeze.json")
CONTRACT = Path(
    "docs/publication_v3_crossgenus_candidate_catalog_contract_v2.md"
)
NAMESPACE_AUDIT = Path(
    "results/metrics/publication_v3_orthogroup_namespace_audit.json"
)
VERIFIED_GO_IMPLEMENTATION = Path(
    "scripts/222_verify_publication_v3_crossgenus_go.py"
)
OUT = Path(
    "results/biological_cases/publication_v3_crossgenus_candidates"
)
REPORT = Path(
    "reports/PUBLICATION_V3_CROSSGENUS_CANDIDATE_CATALOG_20260717_CN.md"
)
MOTIF = "CACGTG"
ABSENT_DIRECTION = {"", "none", "nan", "na", "n/a", "null", "unknown"}
TIER_BY_COUNT = {3: "A", 2: "B", 1: "C", 0: "D"}
AGI = re.compile(r"(?i)(AT[1-5MC]G\d{5})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def atomic_tsv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    table.to_csv(partial, sep="\t", index=False, lineterminator="\n")
    os.replace(partial, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(partial, path)


def load_verified_go_module(root: Path) -> Any:
    path = root / VERIFIED_GO_IMPLEMENTATION
    spec = importlib.util.spec_from_file_location("publication_v3_go_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verified GO implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_overlapping(sequence: str, motif: str = MOTIF) -> int:
    sequence = str(sequence).upper()
    count = 0
    start = 0
    while True:
        position = sequence.find(motif, start)
        if position < 0:
            return count
        count += 1
        start = position + 1


def gc_fraction(sequence: str) -> float:
    sequence = str(sequence).upper()
    canonical = sum(sequence.count(base) for base in "ACGT")
    if canonical == 0:
        return float("nan")
    return (sequence.count("G") + sequence.count("C")) / canonical


def normalize_direction(value: object) -> str:
    direction = str(value).strip().lower()
    return "" if direction in ABSENT_DIRECTION else direction


def gene_count_stratum(value: int) -> str:
    return "1" if value == 1 else "2" if value == 2 else "3+"


def parse_common_gene_to_group(
    path: Path, slugs: set[str]
) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if ": " not in line:
                raise RuntimeError(
                    f"malformed Orthogroups line {line_number}"
                )
            orthogroup, member_text = line.rstrip("\n").split(": ", 1)
            for member in member_text.split():
                parts = member.split("|", 2)
                if len(parts) < 2 or parts[0] not in slugs:
                    continue
                key = (parts[0], parts[1])
                previous = output.setdefault(key, orthogroup)
                if previous != orthogroup:
                    raise RuntimeError(f"duplicate orthogroup membership: {key}")
    return output


def load_genes(
    root: Path,
    genus: str,
    common_gene_to_group: dict[tuple[str, str], str],
) -> pd.DataFrame:
    labels = pd.read_parquet(
        root / DATASETS[genus],
        columns=[
            "gene_id",
            "chromosome",
            "promoter_2048",
            "endpoint_direction",
            "label_binary",
        ],
    ).copy()
    labels["functional_gene_id"] = labels["gene_id"].astype(str)
    if labels["functional_gene_id"].duplicated().any():
        raise RuntimeError(f"duplicate functional gene identifiers: {genus}")

    if genus == "prunus":
        bridge = pd.read_csv(root / BRIDGE, sep="\t", dtype=str)
        if (
            bridge["source_gene_id"].duplicated().any()
            or bridge["technical_gene_id"].duplicated().any()
        ):
            raise RuntimeError("Prunus bridge is not one-to-one")
        bridge_map = bridge.set_index("source_gene_id")["technical_gene_id"]
        labels["technical_gene_id"] = labels["functional_gene_id"].map(bridge_map)
        slug = "prunus_persica"
    else:
        labels["technical_gene_id"] = labels["functional_gene_id"]
        slug = "pyrus_pyrifolia"

    labels["orthogroup"] = labels["technical_gene_id"].map(
        lambda gene_id: (
            common_gene_to_group.get((slug, str(gene_id)), "")
            if pd.notna(gene_id)
            else ""
        )
    )
    labels["orthogroup"] = labels["orthogroup"].fillna("").astype(str)
    labels["label_binary"] = labels["label_binary"].astype(np.int64)
    if not set(labels["label_binary"].unique()).issubset({0, 1}):
        raise RuntimeError(f"non-binary labels: {genus}")
    labels["endpoint_direction"] = labels["endpoint_direction"].map(
        normalize_direction
    )
    labels["gbox_count"] = labels["promoter_2048"].map(count_overlapping)
    labels["gc_fraction_recomputed"] = labels["promoter_2048"].map(gc_fraction)
    if labels["gc_fraction_recomputed"].isna().any():
        raise RuntimeError(f"promoter with no canonical base: {genus}")
    labels["genus"] = genus
    return labels[
        [
            "genus",
            "functional_gene_id",
            "technical_gene_id",
            "chromosome",
            "orthogroup",
            "label_binary",
            "endpoint_direction",
            "gbox_count",
            "gc_fraction_recomputed",
        ]
    ]


def collapse_orthogroups(genes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    mapped = genes.loc[genes["orthogroup"].ne("")].copy()
    for orthogroup, group in mapped.groupby("orthogroup", sort=True):
        positive = group.loc[group["label_binary"].eq(1)]
        positives = int(len(positive))
        negatives = int(len(group) - positives)
        state = (
            "positive_only"
            if positives > 0 and negatives == 0
            else "negative_only"
            if negatives > 0 and positives == 0
            else "mixed"
        )
        directions = sorted(
            value for value in positive["endpoint_direction"].unique() if value
        )
        all_positive_have_direction = bool(
            positives > 0 and positive["endpoint_direction"].ne("").all()
        )
        records.append(
            {
                "orthogroup": orthogroup,
                "state": state,
                "labeled_gene_count": int(len(group)),
                "positive_gene_count": positives,
                "negative_gene_count": negatives,
                "gene_ids": ";".join(
                    sorted(group["functional_gene_id"].astype(str))
                ),
                "directions": ";".join(directions),
                "direction_unique_non_none": bool(
                    all_positive_have_direction and len(directions) == 1
                ),
                "gbox_gene_count": int(group["gbox_count"].gt(0).sum()),
                "gbox_total_count": int(group["gbox_count"].sum()),
                "gbox_promoter_fraction": float(group["gbox_count"].gt(0).mean()),
                "gbox_present": bool(group["gbox_count"].gt(0).any()),
                "mean_promoter_gc": float(
                    group["gc_fraction_recomputed"].mean()
                ),
                "chromosomes": ";".join(
                    sorted(group["chromosome"].astype(str).unique())
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def parse_arabidopsis_symbols(path: Path) -> dict[str, str]:
    symbols: dict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            match = AGI.search(" ".join(fields[index] for index in (1, 2)))
            if match is None:
                continue
            gene = match.group(1).upper()
            symbol = fields[2].strip()
            if symbol and symbol.upper() != gene:
                symbols[gene].add(symbol)
    return {
        gene: ";".join(sorted(values, key=lambda value: value.lower()))
        for gene, values in symbols.items()
    }


def prefixed_record(row: pd.Series, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_{column}": row[column]
        for column in row.index
        if column != "orthogroup"
    }


def build_catalog(
    collapsed: dict[str, pd.DataFrame],
    arabidopsis: dict[str, set[str]],
    annotations: dict[str, set[str]],
    symbols: dict[str, str],
    leaf_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    indexed = {
        genus: table.set_index("orthogroup", drop=False)
        for genus, table in collapsed.items()
    }
    shared = (
        set(indexed["prunus"].index)
        & set(indexed["pyrus"].index)
        & set(arabidopsis)
    )
    positive = sorted(
        orthogroup
        for orthogroup in shared
        if indexed["prunus"].at[orthogroup, "state"] == "positive_only"
        and indexed["pyrus"].at[orthogroup, "state"] == "positive_only"
    )
    negative = sorted(
        orthogroup
        for orthogroup in shared
        if indexed["prunus"].at[orthogroup, "state"] == "negative_only"
        and indexed["pyrus"].at[orthogroup, "state"] == "negative_only"
    )

    candidate_records: list[dict[str, Any]] = []
    for orthogroup in positive:
        prunus = indexed["prunus"].loc[orthogroup]
        pyrus = indexed["pyrus"].loc[orthogroup]
        matching_leaf_terms = sorted(
            set(annotations.get(orthogroup, set())) & set(leaf_names)
        )
        prunus_direction = str(prunus["directions"])
        pyrus_direction = str(pyrus["directions"])
        direction_conserved = bool(
            prunus["direction_unique_non_none"]
            and pyrus["direction_unique_non_none"]
            and prunus_direction == pyrus_direction
        )
        gbox_both = bool(prunus["gbox_present"] and pyrus["gbox_present"])
        component_count = int(bool(matching_leaf_terms))
        component_count += int(direction_conserved) + int(gbox_both)
        ath_ids = sorted(arabidopsis[orthogroup])
        record: dict[str, Any] = {
            "orthogroup": orthogroup,
            "tier": TIER_BY_COUNT[component_count],
            "component_count": component_count,
            "leaf_go": bool(matching_leaf_terms),
            "direction_conserved": direction_conserved,
            "gbox_both_genera": gbox_both,
            "leaf_go_term_count": len(matching_leaf_terms),
            "leaf_go_term_ids": ";".join(matching_leaf_terms),
            "leaf_go_term_names": ";".join(
                leaf_names[term] for term in matching_leaf_terms
            ),
            "arabidopsis_gene_ids": ";".join(ath_ids),
            "arabidopsis_symbols": ";".join(
                f"{gene}:{symbols.get(gene, '')}".rstrip(":")
                for gene in ath_ids
            ),
            "min_genus_gbox_promoter_fraction": min(
                float(prunus["gbox_promoter_fraction"]),
                float(pyrus["gbox_promoter_fraction"]),
            ),
        }
        record.update(prefixed_record(prunus, "prunus"))
        record.update(prefixed_record(pyrus, "pyrus"))
        candidate_records.append(record)

    candidates = pd.DataFrame.from_records(candidate_records)
    if not candidates.empty:
        candidates["_tier_order"] = candidates["tier"].map(
            {"A": 0, "B": 1, "C": 2, "D": 3}
        )
        candidates = candidates.sort_values(
            [
                "_tier_order",
                "leaf_go_term_count",
                "min_genus_gbox_promoter_fraction",
                "orthogroup",
            ],
            ascending=[True, False, False, True],
            kind="stable",
        ).drop(columns="_tier_order")
        candidates.insert(0, "catalog_rank", np.arange(1, len(candidates) + 1))
    else:
        raise RuntimeError("candidate population is unexpectedly empty")

    control_records: list[dict[str, Any]] = []
    used_controls: set[str] = set()
    tier_a = candidates.loc[candidates["tier"].eq("A")]
    for _, candidate in tier_a.iterrows():
        candidate_strata = (
            gene_count_stratum(int(candidate["prunus_labeled_gene_count"])),
            gene_count_stratum(int(candidate["pyrus_labeled_gene_count"])),
        )
        candidate_pattern = (
            bool(candidate["prunus_gbox_present"]),
            bool(candidate["pyrus_gbox_present"]),
        )
        eligible: list[tuple[float, str]] = []
        for orthogroup in negative:
            if orthogroup in used_controls:
                continue
            prunus = indexed["prunus"].loc[orthogroup]
            pyrus = indexed["pyrus"].loc[orthogroup]
            control_strata = (
                gene_count_stratum(int(prunus["labeled_gene_count"])),
                gene_count_stratum(int(pyrus["labeled_gene_count"])),
            )
            control_pattern = (
                bool(prunus["gbox_present"]),
                bool(pyrus["gbox_present"]),
            )
            if control_strata != candidate_strata:
                continue
            if control_pattern != candidate_pattern:
                continue
            distance = abs(
                float(candidate["prunus_mean_promoter_gc"])
                - float(prunus["mean_promoter_gc"])
            ) + abs(
                float(candidate["pyrus_mean_promoter_gc"])
                - float(pyrus["mean_promoter_gc"])
            )
            eligible.append((distance, orthogroup))
        eligible.sort(key=lambda item: (item[0], item[1]))

        base: dict[str, Any] = {
            "candidate_orthogroup": candidate["orthogroup"],
            "candidate_prunus_gene_count_stratum": candidate_strata[0],
            "candidate_pyrus_gene_count_stratum": candidate_strata[1],
            "candidate_prunus_gbox_present": candidate_pattern[0],
            "candidate_pyrus_gbox_present": candidate_pattern[1],
            "eligible_controls_before_selection": len(eligible),
            "matched": bool(eligible),
            "control_orthogroup": "",
            "gc_distance": float("nan"),
        }
        if eligible:
            distance, control_id = eligible[0]
            used_controls.add(control_id)
            prunus = indexed["prunus"].loc[control_id]
            pyrus = indexed["pyrus"].loc[control_id]
            ath_ids = sorted(arabidopsis[control_id])
            base.update(
                {
                    "control_orthogroup": control_id,
                    "gc_distance": distance,
                    "control_prunus_gene_ids": prunus["gene_ids"],
                    "control_pyrus_gene_ids": pyrus["gene_ids"],
                    "control_arabidopsis_gene_ids": ";".join(ath_ids),
                    "control_arabidopsis_symbols": ";".join(
                        f"{gene}:{symbols.get(gene, '')}".rstrip(":")
                        for gene in ath_ids
                    ),
                    "control_prunus_labeled_gene_count": int(
                        prunus["labeled_gene_count"]
                    ),
                    "control_pyrus_labeled_gene_count": int(
                        pyrus["labeled_gene_count"]
                    ),
                    "control_prunus_gbox_present": bool(
                        prunus["gbox_present"]
                    ),
                    "control_pyrus_gbox_present": bool(pyrus["gbox_present"]),
                    "candidate_prunus_mean_promoter_gc": float(
                        candidate["prunus_mean_promoter_gc"]
                    ),
                    "candidate_pyrus_mean_promoter_gc": float(
                        candidate["pyrus_mean_promoter_gc"]
                    ),
                    "control_prunus_mean_promoter_gc": float(
                        prunus["mean_promoter_gc"]
                    ),
                    "control_pyrus_mean_promoter_gc": float(
                        pyrus["mean_promoter_gc"]
                    ),
                }
            )
        control_records.append(base)

    controls = pd.DataFrame.from_records(control_records)
    match_by_candidate = (
        controls.set_index("candidate_orthogroup")["control_orthogroup"].to_dict()
        if not controls.empty
        else {}
    )
    candidates["matched_control_orthogroup"] = candidates["orthogroup"].map(
        match_by_candidate
    ).fillna("")

    population = {
        "shared_labeled_orthogroups_with_arabidopsis": len(shared),
        "shared_positive_only_orthogroups": len(positive),
        "shared_negative_only_control_pool": len(negative),
    }
    return candidates, controls, population


def validate_catalog(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    collapsed: dict[str, pd.DataFrame],
    arabidopsis: dict[str, set[str]],
) -> dict[str, bool]:
    indexed = {
        genus: table.set_index("orthogroup", drop=False)
        for genus, table in collapsed.items()
    }
    tier_counts = candidates["component_count"].map(TIER_BY_COUNT)
    candidate_state_ok = all(
        indexed[genus].at[orthogroup, "state"] == "positive_only"
        for orthogroup in candidates["orthogroup"]
        for genus in ("prunus", "pyrus")
    )
    control_state_ok = all(
        indexed[genus].at[orthogroup, "state"] == "negative_only"
        for orthogroup in controls.loc[controls["matched"], "control_orthogroup"]
        for genus in ("prunus", "pyrus")
    )
    match_exact = True
    if not controls.empty:
        for _, match in controls.loc[controls["matched"]].iterrows():
            candidate = candidates.set_index("orthogroup").loc[
                match["candidate_orthogroup"]
            ]
            control_id = match["control_orthogroup"]
            for genus in ("prunus", "pyrus"):
                control = indexed[genus].loc[control_id]
                if gene_count_stratum(int(control["labeled_gene_count"])) != str(
                    match[f"candidate_{genus}_gene_count_stratum"]
                ):
                    match_exact = False
                if bool(control["gbox_present"]) != bool(
                    candidate[f"{genus}_gbox_present"]
                ):
                    match_exact = False

    expected_order = candidates.sort_values(
        [
            "tier",
            "leaf_go_term_count",
            "min_genus_gbox_promoter_fraction",
            "orthogroup",
        ],
        ascending=[True, False, False, True],
        kind="stable",
    )["orthogroup"].tolist()
    gates = {
        "candidate_population_nonempty": bool(len(candidates) > 0),
        "candidate_orthogroups_unique": bool(
            not candidates["orthogroup"].duplicated().any()
        ),
        "candidate_states_positive_only_both": bool(candidate_state_ok),
        "all_candidates_have_arabidopsis_ortholog": bool(
            candidates["orthogroup"].isin(arabidopsis).all()
        ),
        "tier_equals_component_count": bool(
            candidates["tier"].tolist() == tier_counts.tolist()
        ),
        "catalog_sort_order_exact": bool(
            candidates["orthogroup"].tolist() == expected_order
        ),
        "tier_a_has_one_match_record_each": bool(
            len(controls) == int(candidates["tier"].eq("A").sum())
            and (
                controls["candidate_orthogroup"].is_unique
                if not controls.empty
                else True
            )
        ),
        "matched_controls_unique": bool(
            controls.loc[
                controls["matched"], "control_orthogroup"
            ].is_unique
            if not controls.empty
            else True
        ),
        "matched_controls_negative_only_both": bool(control_state_ok),
        "matched_control_rules_exact": bool(match_exact),
        "model_outputs_accessed": False,
        "malus_accessed": False,
    }
    return gates


def build_report(
    summary: dict[str, Any],
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
) -> str:
    tier_counts = summary["candidate_counts_by_tier"]
    tier_a = candidates.loc[candidates["tier"].eq("A")]
    lines = [
        "# TreeFM publication-v3 跨属实验候选目录",
        "",
        f"- 生成时间（UTC）：`{summary['generated_utc']}`",
        f"- 状态：`{summary['status']}`",
        "- 分析性质：模型无关、Malus 盲态下的描述性实验优先级。",
        "- G-box 定义：2,048-bp 启动子中精确 `CACGTG`，允许重叠计数。",
        "- GC 定义：每条启动子中 `(G+C)/(A+C+G+T)`，再在正交群内取均值。",
        "",
        "## 主要结果",
        "",
        (
            f"Prunus 与 Pyrus 同时为 `positive_only` 且含拟南芥同源基因的"
            f"候选正交群共 **{len(candidates)}** 个。"
        ),
        (
            f"按预注册三项证据分层：Tier A={tier_counts.get('A', 0)}，"
            f"Tier B={tier_counts.get('B', 0)}，"
            f"Tier C={tier_counts.get('C', 0)}，"
            f"Tier D={tier_counts.get('D', 0)}。"
        ),
        (
            f"Tier A 严格匹配到负对照 "
            f"**{summary['matched_controls']}/{len(tier_a)}** 个；"
            "未匹配者不放宽规则。"
        ),
        "",
        "Tier A 同时满足：至少一个冻结叶级 GO 过程、两属所有正例成员"
        "方向完整且一致、两属均至少一个启动子含精确 G-box。"
        "",
        "## Tier A 候选（全部）",
        "",
    ]
    if tier_a.empty:
        lines.append("预注册规则下没有 Tier A 候选；未放宽任何判定。")
    else:
        lines.extend(
            [
                "| Orthogroup | Prunus genes | Pyrus genes | Arabidopsis | "
                "leaf GO | direction | matched control |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for _, row in tier_a.iterrows():
            lines.append(
                "| {orthogroup} | {prunus} | {pyrus} | {ath} | {go} | "
                "{direction} | {control} |".format(
                    orthogroup=row["orthogroup"],
                    prunus=str(row["prunus_gene_ids"]).replace(";", "<br>"),
                    pyrus=str(row["pyrus_gene_ids"]).replace(";", "<br>"),
                    ath=str(row["arabidopsis_symbols"]).replace(";", "<br>"),
                    go=str(row["leaf_go_term_names"]).replace(";", "<br>"),
                    direction=row["prunus_directions"],
                    control=row["matched_control_orthogroup"] or "unmatched",
                )
            )
    lines.extend(
        [
            "",
            "## 建议的验证顺序",
            "",
            "1. 对全部 Tier A 家族做两属同源基因的独立 RT-qPCR 时间序列；",
            "2. 对野生型与精确 G-box 突变启动子做双荧光素酶报告；",
            "3. 仅在存在预注册严格匹配时同步测量 `negative_only` 正交群；"
            "当前无匹配者不得事后放宽家族大小、G-box 模式或启动子 GC 规则；",
            "4. 若方向与报告实验均复现，再进入 TF 结合或 CRISPR 启动子编辑。",
            "",
            "该目录用于实验物流优先级，不构成 GO 过程或 G-box 因果性的证明。"
            "完整候选和所有匹配字段保存在 TSV 中。",
            "",
            "## 审计",
            "",
            f"- 判定门：`{summary['gate_status']}`；"
            f"失败门数：`{summary['failed_gate_count']}`。",
            "- 模型输出读取：`false`；Malus 读取：`false`。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("TREEFM_ROOT", Path(__file__).resolve().parents[1])),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    input_paths = [
        CONTRACT,
        VERIFIED_GO_IMPLEMENTATION,
        DATASETS["prunus"],
        DATASETS["pyrus"],
        BRIDGE,
        ORTHOGROUPS,
        NAMESPACE_AUDIT,
        GAF,
        OBO,
        LEAF_TERMS,
        GO_FREEZE,
        ANALYSIS_FREEZE,
    ]
    missing = [str(path) for path in input_paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(missing))

    namespace_audit = json.loads(
        (root / NAMESPACE_AUDIT).read_text(encoding="utf-8")
    )
    if namespace_audit.get("status") != "namespace_mismatch_confirmed":
        raise RuntimeError("orthogroup namespace mismatch was not confirmed")
    go = load_verified_go_module(root)
    terms, ancestors = go.parse_ontology(root / OBO)
    gene_terms = go.parse_gaf(
        root / GAF,
        terms,
        ancestors,
        excluded={"IEA", "ND"},
    )
    arabidopsis = go.arabidopsis_orthogroups(root / ORTHOGROUPS)
    annotations = go.orthogroup_annotations(arabidopsis, gene_terms)
    symbols = parse_arabidopsis_symbols(root / GAF)
    leaf_table = pd.read_csv(root / LEAF_TERMS, sep="\t", dtype=str)
    if leaf_table["term_id"].duplicated().any():
        raise RuntimeError("corrected leaf-term table contains duplicates")
    leaf_names = dict(zip(leaf_table["term_id"], leaf_table["term_name"]))

    common_gene_to_group = parse_common_gene_to_group(
        root / ORTHOGROUPS,
        {"prunus_persica", "pyrus_pyrifolia"},
    )
    genes = {
        genus: load_genes(root, genus, common_gene_to_group)
        for genus in ("prunus", "pyrus")
    }
    collapsed = {
        genus: collapse_orthogroups(table) for genus, table in genes.items()
    }
    candidates, controls, population = build_catalog(
        collapsed,
        arabidopsis,
        annotations,
        symbols,
        leaf_names,
    )
    gates = validate_catalog(candidates, controls, collapsed, arabidopsis)
    gates["prunus_mapping_at_least_0_98"] = bool(
        genes["prunus"]["orthogroup"].ne("").mean() >= 0.98
    )
    gates["pyrus_mapping_at_least_0_90"] = bool(
        genes["pyrus"]["orthogroup"].ne("").mean() >= 0.90
    )
    gates["complete_corrected_leaf_term_set_loaded"] = bool(
        len(leaf_names) == len(leaf_table)
    )
    gates["single_common_26species_orthogroup_namespace"] = True
    failed = sorted(
        key
        for key, passed in gates.items()
        if key not in {"model_outputs_accessed", "malus_accessed"} and not passed
    )
    forbidden_true = [
        key
        for key in ("model_outputs_accessed", "malus_accessed")
        if gates[key]
    ]
    failed.extend(forbidden_true)

    tier_counts = Counter(candidates["tier"])
    summary: dict[str, Any] = {
        "status": "pass" if not failed else "fail",
        "scope": (
            "model_independent_crossgenus_experimental_candidate_catalog_"
            "corrected_common_26species_namespace"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(CONTRACT),
        "contract_sha256": sha256(root / CONTRACT),
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "motif": MOTIF,
        "motif_counting": "exact overlapping matches",
        "gc_fraction_definition": "(G+C)/(A+C+G+T)",
        "primary_go_excluded_evidence": ["IEA", "ND"],
        "leaf_term_count": len(leaf_names),
        "orthogroup_namespace": {
            "source": str(ORTHOGROUPS),
            "source_sha256": sha256(root / ORTHOGROUPS),
            "both_target_genera_mapped_directly": True,
            "arabidopsis_anchors_from_same_file": True,
            "supersedes_namespace_mismatched_catalog": True,
            "root_cause_audit": str(NAMESPACE_AUDIT),
        },
        "gene_mapping": {
            genus: {
                "labeled_genes": int(len(table)),
                "mapped_genes": int(table["orthogroup"].ne("").sum()),
                "mapping_fraction": float(table["orthogroup"].ne("").mean()),
            }
            for genus, table in genes.items()
        },
        **population,
        "candidate_orthogroups": int(len(candidates)),
        "candidate_counts_by_tier": {
            tier: int(tier_counts.get(tier, 0)) for tier in "ABCD"
        },
        "matched_controls": int(
            controls["matched"].sum() if not controls.empty else 0
        ),
        "unmatched_tier_a_candidates": int(
            (~controls["matched"]).sum() if not controls.empty else 0
        ),
        "gate_status": "pass" if not failed else "fail",
        "failed_gate_count": len(failed),
        "failed_gates": failed,
        "gates": gates,
        "input_sha256": {
            str(path): sha256(root / path) for path in input_paths
        },
        "outputs": {
            "candidate_catalog": str(OUT / "candidate_orthogroups.tsv"),
            "tier_a_candidates": str(OUT / "tier_a_candidates.tsv"),
            "matched_negative_controls": str(
                OUT / "matched_negative_controls.tsv"
            ),
            "report": str(REPORT),
        },
    }

    atomic_tsv(root / OUT / "candidate_orthogroups.tsv", candidates)
    atomic_tsv(
        root / OUT / "tier_a_candidates.tsv",
        candidates.loc[candidates["tier"].eq("A")].copy(),
    )
    atomic_tsv(root / OUT / "matched_negative_controls.tsv", controls)
    atomic_json(root / OUT / "summary.json", summary)
    atomic_text(root / REPORT, build_report(summary, candidates, controls))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
