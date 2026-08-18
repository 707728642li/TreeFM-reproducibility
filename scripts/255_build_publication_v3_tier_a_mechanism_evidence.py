#!/usr/bin/env python3
"""Build the retrospective, non-selective Tier-A mechanism evidence layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


EXPECTED_OG_ORDER = [
    "OG0000025",
    "OG0000191",
    "OG0000413",
    "OG0000350",
    "OG0004506",
    "OG0000215",
    "OG0000277",
    "OG0001301",
    "OG0001507",
    "OG0000208",
    "OG0000742",
    "OG0000139",
    "OG0000692",
    "OG0000083",
]

MECHANISM_MODULE = {
    "OG0000025": "transcriptional relay",
    "OG0000191": "transcriptional relay",
    "OG0000413": "ABA metabolism/signaling",
    "OG0000350": "transcriptional relay",
    "OG0004506": "unresolved stress protein",
    "OG0000215": "ABA metabolism/signaling",
    "OG0000277": "ABA metabolism/signaling",
    "OG0001301": "transcriptional relay",
    "OG0001507": "transcriptional relay",
    "OG0000208": "receptor/transport/metabolism",
    "OG0000742": "transcriptional relay",
    "OG0000139": "receptor/transport/metabolism",
    "OG0000692": "receptor/transport/metabolism",
    "OG0000083": "receptor/transport/metabolism",
}

MODULE_COLOR = {
    "transcriptional relay": "#6A51A3",
    "ABA metabolism/signaling": "#D95F0E",
    "receptor/transport/metabolism": "#238B45",
    "unresolved stress protein": "#636363",
}

LITERATURE_SHORT = {
    "direct_same_family_same_process": "same-process\nfamily perturb.",
    "same_pathway_same_process": "same-process\npathway",
    "family_expression_same_process": "same-process\nexpression",
    "direct_same_family_related_process": "related-process\nmechanism",
    "annotation_only": "annotation\nonly",
}

LITERATURE_LEVEL = {
    "direct_same_family_same_process": 4,
    "same_pathway_same_process": 3,
    "family_expression_same_process": 2,
    "direct_same_family_related_process": 1,
    "annotation_only": 0,
}

LITERATURE_COLOR = {
    4: "#2166AC",
    3: "#4393C3",
    2: "#92C5DE",
    1: "#D1E5F0",
    0: "#E0E0E0",
}

DOMAIN_COLOR = {
    "cross_genus_anchor_supported": "#1B9E77",
    "partial": "#E6AB02",
    "unresolved": "#BDBDBD",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_frozen_inputs(root: Path) -> tuple[dict, dict, list[Path]]:
    candidate_freeze_path = root / "config/publication_v3_crossgenus_candidate_catalog_freeze.json"
    annotation_freeze_path = root / "config/publication_v3_tier_a_postselection_annotation_freeze.json"
    candidate_freeze = json.loads(candidate_freeze_path.read_text(encoding="utf-8"))
    annotation_freeze = json.loads(annotation_freeze_path.read_text(encoding="utf-8"))

    if candidate_freeze.get("status") != "pass" or candidate_freeze.get("freeze_version") != "2.0":
        fail("candidate catalog freeze v2 is not passing")
    frozen_order = candidate_freeze["result_summary"]["tier_a_families"]
    if frozen_order != EXPECTED_OG_ORDER:
        fail("candidate freeze Tier-A order differs from the fixed corrected order")
    if candidate_freeze.get("model_outputs_accessed") or candidate_freeze.get("malus_accessed"):
        fail("candidate freeze reports prohibited access")

    candidate_freeze_hash = sha256(candidate_freeze_path)
    population = annotation_freeze.get("candidate_population", {})
    if annotation_freeze.get("status") != "frozen" or annotation_freeze.get("freeze_version") != "2.0":
        fail("annotation freeze v2 is not frozen")
    if population.get("candidate_freeze_sha256") != candidate_freeze_hash:
        fail("annotation freeze does not pin the current candidate freeze")
    if population.get("tier_a_families") != EXPECTED_OG_ORDER:
        fail("annotation freeze Tier-A order differs from the corrected catalog")
    if annotation_freeze.get("model_outputs_accessed") or annotation_freeze.get("malus_accessed"):
        fail("annotation freeze reports prohibited access")

    tier_path = root / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv"
    frozen_tier = candidate_freeze["frozen_artifacts"][tier_path.relative_to(root).as_posix()]["sha256"]
    if sha256(tier_path) != frozen_tier:
        fail("Tier-A candidate table hash differs from candidate freeze")

    domain_audit_path = root / "results/biological_cases/publication_v3_tier_a_annotation/independent_audit.json"
    domain_audit = json.loads(domain_audit_path.read_text(encoding="utf-8"))
    if domain_audit.get("status") != "pass" or domain_audit.get("failure_count", 0) != 0:
        fail("full-Pfam independent audit is not passing")
    if domain_audit.get("model_outputs_accessed") or domain_audit.get("malus_accessed"):
        fail("full-Pfam audit reports prohibited access")

    accessed = [
        candidate_freeze_path,
        annotation_freeze_path,
        tier_path,
        root / "results/biological_cases/publication_v3_tier_a_annotation/domain_summary.json",
        root / "results/biological_cases/publication_v3_tier_a_annotation/orthogroup_domain_consensus.tsv",
        domain_audit_path,
        root / "metadata/publication_v3_tier_a_literature_evidence.tsv",
        root / "docs/publication_v3_tier_a_mechanism_evidence_contract_v1.md",
    ]
    return candidate_freeze, annotation_freeze, accessed


def reconstruct_matrix(root: Path) -> pd.DataFrame:
    tier = pd.read_csv(
        root / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    literature = pd.read_csv(
        root / "metadata/publication_v3_tier_a_literature_evidence.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    consensus = pd.read_csv(
        root / "results/biological_cases/publication_v3_tier_a_annotation/orthogroup_domain_consensus.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    domain_summary = json.loads(
        (root / "results/biological_cases/publication_v3_tier_a_annotation/domain_summary.json").read_text(
            encoding="utf-8"
        )
    )

    if tier["orthogroup"].tolist() != EXPECTED_OG_ORDER or tier["catalog_rank"].astype(int).tolist() != list(
        range(1, 15)
    ):
        fail("Tier-A table order/ranks are not exact")
    if set(tier["tier"]) != {"A"}:
        fail("Tier-A table contains a non-A row")
    if literature["orthogroup"].tolist() != EXPECTED_OG_ORDER:
        fail("literature table order is not exact")
    if literature["catalog_rank"].astype(int).tolist() != list(range(1, 15)):
        fail("literature table ranks are not exact")
    if any(parse_bool(value) for value in literature["selection_authority"]):
        fail("literature metadata claims selection authority")
    if set(literature["evidence_grade"]) - set(LITERATURE_LEVEL):
        fail("unknown literature evidence grade")

    merged = tier.merge(
        literature,
        on=["catalog_rank", "orthogroup"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        og = row["orthogroup"]
        sub = consensus.loc[consensus["orthogroup"] == og].copy()
        supported = sub.loc[sub["support_label"] == "cross_genus_anchor_supported"]
        detail = domain_summary["by_orthogroup"][og]
        if not supported.empty:
            domain_status = "cross_genus_anchor_supported"
            selected = supported
        elif int(detail["proteins_with_pfam_hit"]) > 0:
            domain_status = "partial"
            selected = sub
        else:
            domain_status = "unresolved"
            selected = sub.iloc[0:0]

        accessions = ";".join(sorted(set(selected["pfam_accession"]))) if not selected.empty else ""
        names = ";".join(sorted(set(selected["pfam_name"]))) if not selected.empty else ""
        prunus_direction = row["prunus_directions"]
        pyrus_direction = row["pyrus_directions"]
        if prunus_direction != "down" or pyrus_direction != "down":
            fail(f"{og} does not have the expected conserved down endpoint direction")

        matched_control = bool(row["matched_control_orthogroup"])
        min_gbox = min(float(row["prunus_gbox_promoter_fraction"]), float(row["pyrus_gbox_promoter_fraction"]))
        if abs(min_gbox - float(row["min_genus_gbox_promoter_fraction"])) > 1e-12:
            fail(f"{og} min-genus G-box fraction is inconsistent")

        rows.append(
            {
                "catalog_rank": int(row["catalog_rank"]),
                "orthogroup": og,
                "retrospective_family_label": row["retrospective_family_label"],
                "mechanism_module": MECHANISM_MODULE[og],
                "endpoint_direction_prunus": prunus_direction,
                "endpoint_direction_pyrus": pyrus_direction,
                "leaf_go_term_count": int(row["leaf_go_term_count"]),
                "leaf_go_term_ids": row["leaf_go_term_ids"],
                "leaf_go_term_names": row["leaf_go_term_names"],
                "prunus_gbox_gene_count": int(row["prunus_gbox_gene_count"]),
                "prunus_labeled_gene_count": int(row["prunus_labeled_gene_count"]),
                "prunus_gbox_promoter_fraction": float(row["prunus_gbox_promoter_fraction"]),
                "pyrus_gbox_gene_count": int(row["pyrus_gbox_gene_count"]),
                "pyrus_labeled_gene_count": int(row["pyrus_labeled_gene_count"]),
                "pyrus_gbox_promoter_fraction": float(row["pyrus_gbox_promoter_fraction"]),
                "min_genus_gbox_promoter_fraction": min_gbox,
                "strict_matched_control": matched_control,
                "matched_control_orthogroup": row["matched_control_orthogroup"],
                "pfam_support_status": domain_status,
                "supported_pfam_accessions": accessions,
                "supported_pfam_names": names,
                "proteins_total": int(detail["proteins_total"]),
                "proteins_with_pfam_hit": int(detail["proteins_with_pfam_hit"]),
                "literature_evidence_grade": row["evidence_grade"],
                "literature_evidence_level": LITERATURE_LEVEL[row["evidence_grade"]],
                "prior_evidence_note": row["prior_evidence_note"],
                "source_url": row["source_url"],
                "doi": row["doi"],
                "selection_authority": False,
                "model_outputs_accessed": False,
                "malus_accessed": False,
            }
        )

    matrix = pd.DataFrame(rows)
    if matrix["orthogroup"].tolist() != EXPECTED_OG_ORDER:
        fail("reconstructed matrix order is not exact")
    if int(matrix["strict_matched_control"].sum()) != 6:
        fail("strict matched-control total is not six")
    return matrix


def plot_matrix(matrix: pd.DataFrame, png: Path, pdf: Path, svg: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(17.5, 10.5))
    ax.set_xlim(0, 19.2)
    ax.set_ylim(-1.9, 15.5)
    ax.axis("off")

    ax.text(
        0,
        15.15,
        "Frozen Tier-A candidates: convergent evidence without post hoc re-ranking",
        fontsize=16,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0,
        14.72,
        "All 14 candidates are retained. G-box denotes a candidate promoter input; literature evidence is family/pathway context.",
        fontsize=9.5,
        color="#444444",
        va="center",
    )

    headers = [
        (0.30, "Rank"),
        (0.92, "Frozen candidate / retrospective family"),
        (7.10, "Leaf GO"),
        (10.10, "Min G-box"),
        (12.25, "Endpoint"),
        (13.35, "Strict\ncontrol"),
        (14.45, "Full-Pfam support"),
        (17.05, "Prior literature"),
    ]
    for x, label in headers:
        ax.text(x, 14.15, label, fontweight="bold", va="center", ha="left")

    max_go = max(1, int(matrix["leaf_go_term_count"].max()))
    for idx, row in matrix.iterrows():
        y = 13.45 - idx
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.45), 19.0, 0.90, color="#F7F7F7", zorder=0))
        ax.add_patch(
            Rectangle(
                (0.02, y - 0.45),
                0.14,
                0.90,
                color=MODULE_COLOR[row["mechanism_module"]],
                zorder=2,
            )
        )
        ax.text(0.34, y, str(row["catalog_rank"]), va="center", ha="center", fontweight="bold")
        family = row["retrospective_family_label"]
        ax.text(0.92, y + 0.13, row["orthogroup"], va="center", ha="left", fontweight="bold")
        ax.text(0.92, y - 0.16, family, va="center", ha="left", fontsize=8.3, color="#333333")

        go_count = int(row["leaf_go_term_count"])
        ax.add_patch(Rectangle((7.10, y - 0.18), 2.25, 0.36, color="#E5E5E5", zorder=1))
        ax.add_patch(
            Rectangle((7.10, y - 0.18), 2.25 * go_count / max_go, 0.36, color="#4C78A8", zorder=2)
        )
        ax.text(9.46, y, str(go_count), va="center", ha="right", fontweight="bold")

        gbox = float(row["min_genus_gbox_promoter_fraction"])
        ax.add_patch(Rectangle((10.10, y - 0.24), 1.55, 0.48, color="#EEEEEE", zorder=1))
        ax.add_patch(
            Rectangle(
                (10.10, y - 0.24),
                1.55 * gbox,
                0.48,
                color=plt.cm.YlGnBu(0.25 + 0.65 * gbox),
                zorder=2,
            )
        )
        ax.text(11.83, y, f"{gbox:.2f}", va="center", ha="right")

        ax.text(12.30, y, "DOWN", va="center", ha="left", color="#3B528B", fontweight="bold", fontsize=8)
        has_control = bool(row["strict_matched_control"])
        ax.text(
            13.58,
            y,
            "yes" if has_control else "no",
            va="center",
            ha="center",
            color="#1B7837" if has_control else "#777777",
            fontweight="bold",
        )

        domain_status = row["pfam_support_status"]
        domain_label = {
            "cross_genus_anchor_supported": "3-way anchor",
            "partial": "partial",
            "unresolved": "unresolved",
        }[domain_status]
        pfam_n = len([x for x in row["supported_pfam_accessions"].split(";") if x])
        if domain_status == "cross_genus_anchor_supported":
            domain_label += f" ({pfam_n})"
        ax.text(
            14.48,
            y,
            domain_label,
            va="center",
            ha="left",
            fontsize=8.1,
            color="white" if domain_status == "cross_genus_anchor_supported" else "#333333",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": DOMAIN_COLOR[domain_status],
                "edgecolor": "none",
            },
        )

        lit_level = int(row["literature_evidence_level"])
        ax.text(
            17.08,
            y,
            LITERATURE_SHORT[row["literature_evidence_grade"]],
            va="center",
            ha="left",
            fontsize=7.7,
            color="white" if lit_level >= 3 else "#222222",
            linespacing=0.9,
            bbox={
                "boxstyle": "round,pad=0.23",
                "facecolor": LITERATURE_COLOR[lit_level],
                "edgecolor": "none",
            },
        )

    legend_y = -0.80
    ax.text(0, legend_y + 0.55, "Mechanistic hypothesis module:", fontweight="bold", va="center")
    x = 3.70
    for label, color in MODULE_COLOR.items():
        ax.add_patch(Rectangle((x, legend_y + 0.39), 0.20, 0.30, color=color))
        ax.text(x + 0.28, legend_y + 0.55, label, va="center", fontsize=8.2)
        x += {
            "transcriptional relay": 2.65,
            "ABA metabolism/signaling": 3.15,
            "receptor/transport/metabolism": 3.75,
            "unresolved stress protein": 3.15,
        }[label]
    ax.text(
        0,
        -1.45,
        "Post-selection descriptive layer; no model outputs or held-out Malus data accessed. "
        "DOWN = lower expression toward the release-associated endpoint, not a causal direction.",
        fontsize=8.6,
        color="#444444",
    )

    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_report(matrix: pd.DataFrame) -> str:
    status_counts = Counter(matrix["pfam_support_status"])
    grade_counts = Counter(matrix["literature_evidence_grade"])
    module_rows: dict[str, pd.DataFrame] = {
        module: matrix.loc[matrix["mechanism_module"] == module].copy()
        for module in MODULE_COLOR
    }

    candidate_lines = []
    for row in matrix.to_dict("records"):
        pfam = row["supported_pfam_names"] or "无通过的三方一致 Pfam"
        control = row["matched_control_orthogroup"] or "无合格严格匹配"
        candidate_lines.append(
            f"- Rank {row['catalog_rank']} `{row['orthogroup']}` — "
            f"{row['retrospective_family_label']}；GO 叶节点 {row['leaf_go_term_count']}；"
            f"两属较低 G-box 比例 {row['min_genus_gbox_promoter_fraction']:.2f}；"
            f"Pfam={row['pfam_support_status']}（{pfam}）；严格对照={control}；"
            f"文献层级={row['literature_evidence_grade']}。"
        )

    module_lines = []
    for module, sub in module_rows.items():
        if sub.empty:
            continue
        ids = "、".join(f"`{og}`" for og in sub["orthogroup"])
        module_lines.append(f"- **{module}**：{ids}")

    return f"""# Publication v3：Tier-A 候选机制证据与实验推进方案

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论摘要

修正后的共同 26 物种 orthogroup 命名空间中，14 个冻结 Tier-A 候选全部保留，
没有因为结构域或文献结果进行事后删除或重排。完整 Pfam 扫描显示：

- {status_counts.get('cross_genus_anchor_supported', 0)}/14 具有 Arabidopsis–Prunus–Pyrus 三方一致结构域支持；
- {status_counts.get('partial', 0)}/14 为部分结构域支持；
- {status_counts.get('unresolved', 0)}/14 暂无通过完整 Pfam 阈值的结构域；
- 6/14 有预先定义条件下的严格匹配阴性对照，另外 8 个不放宽匹配标准；
- 14/14 在 Prunus 和 Pyrus 中都表现为向“休眠释放相关终点”下降。

这些结果支持一个可检验、但尚非因果结论的工作模型：休眠转换不是单个标志基因的
孤立变化，而可能涉及 ABA 代谢/信号、转录因子接力以及受体–转运–代谢界面的
协同衰减或重配置。这里的“下降”仅指源数据对比中的终点方向，不能外推为整个
通路被关闭。

## 三层证据的边界

1. GO、方向、G-box 和严格对照来自模型无关的冻结候选目录。
2. Pfam 是候选确定后的完整库无偏注释，只用于解释，不用于选择。
3. 文献证据是家族或通路上下文。即使同家族扰动改变芽萌发时间，也不证明本研究
   orthogroup 就是文献中的同一基因或已具备因果作用。

G-box 位于候选基因启动子，提示可能存在上游 G-box 结合因子的调控输入；它不表示
AP2/ERF、WRKY、VQ 等候选蛋白本身结合 G-box。

## 可检验的机制模块

{chr(10).join(module_lines)}

ABA 轴的核心可检验关系是 NCED/CCD–ABA 含量、PP2C–SnRK2 门控和
PP2C–MAPKKK 应激串扰；转录层可检验 AP2/ERF、WRKY–VQ 与 bHLH/BBX
在时间序列中的先后关系；受体和转运层则检验 CRK/DUF26、LRR-RLK、ABC
与代谢状态是否解释跨属一致的终点下降。该模块化组织只用于实验设计，不改变
冻结排名。

## 候选逐项证据

{chr(10).join(candidate_lines)}

## 实验推进顺序

### 第一阶段：低成本正交验证

- 在同一品种、同一枝条层级建立不少于 6 个时间点的冷量–萌芽–表达联合序列；
- 对 14 个候选全部做 RT-qPCR，优先使用跨外显子引物并核验扩增单一性；
- 同步测 ABA、相位酸、GA 和可溶性糖，避免把发育阶段与激素变化混为一谈；
- 对有严格匹配对照的 6 个候选执行一一配对检验；另外 8 个只报告候选结果，
  不使用放宽后对照补齐。

### 第二阶段：顺式调控检验

- 为每个候选构建野生型启动子和精确 `CACGTG` 破坏型启动子报告载体；
- 在相同背景下比较冷处理、ABA/GA 处理与时间的交互；
- 用 DAP-seq/EMSA/ChIP-qPCR 寻找真正的上游 G-box 结合因子，避免把 G-box
  误归因于候选编码蛋白。

### 第三阶段：模块因果与互作

- 对 AP2/ERF、WRKY、PP2C、MAPKKK、NCED/CCD 和 VQ 模块代表执行
  VIGS/CRISPRi 与过表达双向扰动，并测量萌芽时间和激素表型；
- 用 Y2H、BiFC 或 Co-IP 检验 VQ–WRKY 和 PP2C–MAPKKK/SnRK2 互作；
- 用双扰动或药理补偿做遗传顺序检验，而不是只依赖相关性；
- `OG0004506` 先做转录本/翻译真实性、亚细胞定位和结构预测；
  `OG0001507` 先排查注释完整性与属间结构域缺失是否由基因模型造成。

## 文献证据计数

{json.dumps(dict(sorted(grade_counts.items())), ensure_ascii=False)}

## 论文表述建议

主文应使用“跨属模型无关候选”“三方结构域一致”“家族/通路支持的机制假说”和
“待扰动验证”等措辞。不要使用“发现了 EBB1 正交基因”“证明 G-box 由 WRKY
直接调控”或“证明 ABA 通路被关闭”等超出证据范围的表述。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    _, _, accessed_paths = validate_frozen_inputs(root)
    matrix = reconstruct_matrix(root)

    out_dir = root / "results/biological_cases/publication_v3_tier_a_mechanism_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    matrix_path = root / "metadata/publication_v3_tier_a_evidence_matrix.tsv"
    report_path = root / "reports/PUBLICATION_V3_TIER_A_MECHANISM_EVIDENCE_20260717_CN.md"
    png_path = root / "figures/publication_v3_tier_a_evidence_matrix.png"
    pdf_path = root / "figures/publication_v3_tier_a_evidence_matrix.pdf"
    svg_path = root / "figures/publication_v3_tier_a_evidence_matrix.svg"
    provenance_path = out_dir / "provenance.json"

    matrix.to_csv(matrix_path, sep="\t", index=False, lineterminator="\n")
    report_path.write_text(build_report(matrix), encoding="utf-8")
    plot_matrix(matrix, png_path, pdf_path, svg_path)

    output_paths = [matrix_path, report_path, png_path, pdf_path, svg_path]
    provenance = {
        "status": "pass",
        "scope": "retrospective_fixed_tier_a_mechanism_evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_authority": False,
        "candidate_reranking_performed": False,
        "model_outputs_accessed": False,
        "malus_accessed": False,
        "candidate_count": len(matrix),
        "candidate_order": matrix["orthogroup"].tolist(),
        "summary": {
            "strict_matched_controls": int(matrix["strict_matched_control"].sum()),
            "pfam_support_counts": dict(Counter(matrix["pfam_support_status"])),
            "literature_grade_counts": dict(Counter(matrix["literature_evidence_grade"])),
            "all_conserved_endpoint_down": bool(
                (matrix["endpoint_direction_prunus"] == "down").all()
                and (matrix["endpoint_direction_pyrus"] == "down").all()
            ),
        },
        "gates": {
            "candidate_freeze_v2_current": True,
            "annotation_freeze_v2_current": True,
            "full_pfam_independent_audit_pass": True,
            "all_14_candidates_retained": len(matrix) == 14,
            "frozen_rank_order_preserved": matrix["orthogroup"].tolist() == EXPECTED_OG_ORDER,
            "strict_controls_not_relaxed": int(matrix["strict_matched_control"].sum()) == 6,
            "no_model_access": True,
            "no_malus_access": True,
        },
        "accessed_paths": [path.relative_to(root).as_posix() for path in accessed_paths],
        "inputs": {path.relative_to(root).as_posix(): sha256(path) for path in accessed_paths},
        "outputs": {path.relative_to(root).as_posix(): sha256(path) for path in output_paths},
        "violations": [],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
