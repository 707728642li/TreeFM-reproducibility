#!/usr/bin/env python3
"""Build publication-v4 GO robustness source data and figures.

This script is additive and descriptive.  It reuses the frozen 52-term result,
the fixed 18 nonredundant leaf terms, and every leave-one-chromosome refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODULE_ORDER = [
    "Environmental / defense",
    "Hormone signaling",
    "Cell-wall remodeling",
    "Specialized metabolism",
]
MODULE_COLOR = {
    "Environmental / defense": "#3B82A0",
    "Hormone signaling": "#D9822B",
    "Cell-wall remodeling": "#6A8E3A",
    "Specialized metabolism": "#8B5E9F",
}
TERM_MODULE = {
    "GO:0006970": "Environmental / defense",
    "GO:0042742": "Environmental / defense",
    "GO:1901701": "Environmental / defense",
    "GO:0009409": "Environmental / defense",
    "GO:0009414": "Environmental / defense",
    "GO:0009611": "Environmental / defense",
    "GO:0009411": "Environmental / defense",
    "GO:0071456": "Environmental / defense",
    "GO:0009737": "Hormone signaling",
    "GO:0032870": "Hormone signaling",
    "GO:0071396": "Hormone signaling",
    "GO:0009753": "Hormone signaling",
    "GO:0009751": "Hormone signaling",
    "GO:0071669": "Cell-wall remodeling",
    "GO:0042545": "Cell-wall remodeling",
    "GO:0006721": "Specialized metabolism",
    "GO:0008299": "Specialized metabolism",
    "GO:0009699": "Specialized metabolism",
}
GENUS_COLOR = {"prunus": "#B84A62", "pyrus": "#2E75B6"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_obo(path: Path) -> tuple[nx.DiGraph, dict[str, str]]:
    """Parse GO term names and child-to-parent is_a/part_of edges."""
    graph = nx.DiGraph()
    names: dict[str, str] = {}
    current: dict[str, object] | None = None

    def commit(term: dict[str, object] | None) -> None:
        if not term or "id" not in term or term.get("obsolete"):
            return
        term_id = str(term["id"])
        graph.add_node(term_id)
        names[term_id] = str(term.get("name", term_id))
        for parent in term.get("parents", []):
            graph.add_edge(term_id, str(parent))

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit(current)
                current = {"parents": []}
            elif line.startswith("["):
                commit(current)
                current = None
            elif current is not None:
                if line.startswith("id: GO:"):
                    current["id"] = line.split("id: ", 1)[1].strip()
                elif line.startswith("name: "):
                    current["name"] = line.split("name: ", 1)[1].strip()
                elif line.startswith("is_a: GO:"):
                    current["parents"].append(line.split()[1])
                elif line.startswith("relationship: part_of GO:"):
                    current["parents"].append(line.split()[2])
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
    commit(current)
    return graph, names


def ontology_edges(leaf: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    """Create a fixed, effect-independent semantic graph among leaf terms.

    Within each frozen broad module, retain a maximum spanning tree based on
    Jaccard overlap of GO ancestor sets, plus every edge with Jaccard >= 0.45.
    """
    ancestor: dict[str, set[str]] = {}
    for term in leaf["term_id"]:
        ancestor[term] = {term} | nx.descendants(graph, term)
    rows: list[dict[str, object]] = []
    for module in MODULE_ORDER:
        terms = leaf.loc[leaf["module"] == module, "term_id"].tolist()
        complete = nx.Graph()
        complete.add_nodes_from(terms)
        for i, left in enumerate(terms):
            for right in terms[i + 1 :]:
                union = ancestor[left] | ancestor[right]
                score = len(ancestor[left] & ancestor[right]) / len(union)
                complete.add_edge(left, right, weight=float(score))
        selected: set[tuple[str, str]] = set()
        if len(terms) > 1:
            for left, right in nx.maximum_spanning_tree(complete, weight="weight").edges():
                selected.add(tuple(sorted((left, right))))
        for left, right, data in complete.edges(data=True):
            if data["weight"] >= 0.45:
                selected.add(tuple(sorted((left, right))))
        for left, right in sorted(selected):
            rows.append(
                {
                    "source": left,
                    "target": right,
                    "module": module,
                    "ancestor_jaccard": complete[left][right]["weight"],
                    "edge_basis": "GO ancestor-set similarity; module maximum-spanning tree or Jaccard >= 0.45",
                }
            )
    return pd.DataFrame(rows)


def short_term(text: str, width: int = 27) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def plot_main(
    leaf: pd.DataFrame,
    loo_leaf: pd.DataFrame,
    permutation: pd.DataFrame,
    edges: pd.DataFrame,
    out_base: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(19.2, 15.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.28], height_ratios=[0.92, 1.08])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A: permutation null.
    counts = permutation["replicated_fdr_hits"].astype(int)
    maximum = int(counts.max())
    bins = np.arange(-0.5, maximum + 1.5, 1)
    ax_a.hist(counts, bins=bins, color="#8FA9B8", edgecolor="white", linewidth=0.6)
    ax_a.set_yscale("log")
    ax_a.set_xlim(-0.7, maximum + 0.7)
    ax_a.set_xlabel("Replicated FDR terms under permutation")
    ax_a.set_ylabel("Permutation count (log scale)")
    ax_a.set_title("A  Cross-genus enrichment exceeds the frozen null", loc="left", fontweight="bold")
    ax_a.text(
        0.98,
        0.96,
        "Observed = 52\nNull maximum = 12\nEmpirical P = 1 / 10,001",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        color="#A42032",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF5F5", "edgecolor": "#D8A3AA"},
    )
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.grid(axis="y", alpha=0.18, linewidth=0.6)

    # B: all fixed leaf terms, paired log2 odds-ratio intervals.
    forest = leaf.sort_values(["module", "q_crossgenus", "term_name"], key=lambda s: s.map({m: i for i, m in enumerate(MODULE_ORDER)}) if s.name == "module" else s).reset_index(drop=True)
    y = np.arange(len(forest))[::-1]
    offsets = {"prunus": 0.16, "pyrus": -0.16}
    for genus in ["prunus", "pyrus"]:
        center = np.log2(forest[f"{genus}_odds_ratio_haldane"].astype(float).to_numpy())
        low = forest[f"{genus}_log2_or_ci_low"].astype(float).to_numpy()
        high = forest[f"{genus}_log2_or_ci_high"].astype(float).to_numpy()
        ax_b.errorbar(
            center,
            y + offsets[genus],
            xerr=np.vstack([center - low, high - center]),
            fmt="o",
            markersize=4.2,
            elinewidth=1.0,
            capsize=2,
            color=GENUS_COLOR[genus],
            label=genus.capitalize(),
            zorder=3,
        )
    ax_b.axvline(0, color="#4B5563", linewidth=0.8, linestyle="--")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(forest["term_name"], fontsize=8.3)
    for tick, module in zip(ax_b.get_yticklabels(), forest["module"]):
        tick.set_color(MODULE_COLOR[module])
    ax_b.set_xlabel("log2 odds ratio (95% CI)")
    ax_b.set_title("B  All 18 fixed nonredundant leaf terms", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="lower right")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="x", alpha=0.18, linewidth=0.6)

    # C: every LOO refit for the 18 leaves.
    chrom_order = [f"Pp{i:02d}" for i in range(1, 9)] + [f"Chr{i}" for i in range(1, 18)]
    term_order = forest["term_id"].tolist()
    matrix = (
        loo_leaf.assign(log2_or=lambda d: np.log2(d["odds_ratio_haldane"].astype(float)))
        .pivot(index="term_id", columns="left_out_chromosome", values="log2_or")
        .reindex(index=term_order, columns=chrom_order)
    )
    finite = matrix.to_numpy()[np.isfinite(matrix.to_numpy())]
    vmax = float(np.quantile(finite, 0.98))
    image = ax_c.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", norm=Normalize(vmin=0, vmax=max(vmax, 1.0)))
    ax_c.axvline(7.5, color="white", linewidth=2.2)
    ax_c.set_xticks(np.arange(len(chrom_order)))
    ax_c.set_xticklabels(chrom_order, rotation=90, fontsize=7.1)
    ax_c.set_yticks(np.arange(len(term_order)))
    name_map = leaf.set_index("term_id")["term_name"].to_dict()
    ax_c.set_yticklabels([name_map[x] for x in term_order], fontsize=7.3)
    for tick, term in zip(ax_c.get_yticklabels(), term_order):
        tick.set_color(MODULE_COLOR[TERM_MODULE[term]])
    ax_c.text(3.5, -1.05, "Prunus", ha="center", va="bottom", color=GENUS_COLOR["prunus"], fontweight="bold")
    ax_c.text(16.0, -1.05, "Pyrus", ha="center", va="bottom", color=GENUS_COLOR["pyrus"], fontweight="bold")
    ax_c.set_title("C  Leave-one-chromosome stability", loc="left", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax_c, fraction=0.035, pad=0.02)
    cbar.set_label("log2 odds ratio")

    # D: ontology-derived, effect-independent semantic map.
    centers = {
        "Environmental / defense": (-1.15, 0.72),
        "Hormone signaling": (1.05, 0.72),
        "Cell-wall remodeling": (-0.85, -0.82),
        "Specialized metabolism": (0.95, -0.82),
    }
    positions: dict[str, tuple[float, float]] = {}
    for module in MODULE_ORDER:
        terms = leaf.loc[leaf["module"] == module].sort_values("term_id")["term_id"].tolist()
        radius = 0.44 if len(terms) <= 3 else 0.58
        angles = np.linspace(0, 2 * math.pi, len(terms), endpoint=False) + math.pi / 2
        cx, cy = centers[module]
        for term, angle in zip(terms, angles):
            positions[term] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    for row in edges.itertuples(index=False):
        x1, y1 = positions[row.source]
        x2, y2 = positions[row.target]
        ax_d.plot(
            [x1, x2],
            [y1, y2],
            color=MODULE_COLOR[row.module],
            alpha=0.18 + 0.72 * min(float(row.ancestor_jaccard), 1.0),
            linewidth=0.7 + 3.0 * float(row.ancestor_jaccard),
            zorder=1,
        )
    leaf_index = leaf.set_index("term_id")
    for term, (x, y_pos) in positions.items():
        row = leaf_index.loc[term]
        size = 65 + 8.0 * min(float(row["prunus_term_positive"]), float(row["pyrus_term_positive"]))
        ax_d.scatter(
            [x],
            [y_pos],
            s=size,
            color=MODULE_COLOR[row["module"]],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax_d.text(x, y_pos - 0.12, short_term(str(row["term_name"]), 20), ha="center", va="top", fontsize=6.6, color="#263238", zorder=4)
    for module, (x, y_pos) in centers.items():
        ax_d.text(x, y_pos + 0.68, module, ha="center", va="bottom", color=MODULE_COLOR[module], fontweight="bold", fontsize=8.4)
    ax_d.set_xlim(-2.0, 1.9)
    ax_d.set_ylim(-1.65, 1.55)
    ax_d.axis("off")
    ax_d.set_title("D  GO-ancestry map (independent of effect size)", loc="left", fontweight="bold")
    ax_d.legend(
        handles=[Patch(facecolor=MODULE_COLOR[m], label=m) for m in MODULE_ORDER],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        fontsize=7.4,
    )

    fig.suptitle(
        "A cross-genus stress–hormone program is reproducible across ontology and chromosomes",
        fontsize=15.2,
        fontweight="bold",
    )
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(out_base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_full_stability(terms: pd.DataFrame, loo: pd.DataFrame, out_base: Path) -> None:
    chrom_order = [f"Pp{i:02d}" for i in range(1, 9)] + [f"Chr{i}" for i in range(1, 18)]
    order = terms.sort_values(["q_crossgenus", "term_name"])["term_id"].tolist()
    matrix = (
        loo.assign(log2_or=lambda d: np.log2(d["odds_ratio_haldane"].astype(float)))
        .pivot(index="term_id", columns="left_out_chromosome", values="log2_or")
        .reindex(index=order, columns=chrom_order)
    )
    finite = matrix.to_numpy()[np.isfinite(matrix.to_numpy())]
    vmax = float(np.quantile(finite, 0.985))
    fig, ax = plt.subplots(figsize=(14.8, 16.0), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", norm=Normalize(vmin=0, vmax=max(vmax, 1.0)))
    ax.axvline(7.5, color="white", linewidth=2.5)
    ax.set_xticks(np.arange(len(chrom_order)))
    ax.set_xticklabels(chrom_order, rotation=90, fontsize=7.5)
    names = terms.set_index("term_id")["term_name"].to_dict()
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"{term}  {names[term]}" for term in order], fontsize=6.6)
    ax.set_title("All 52 frozen replicated GO terms across every leave-one-chromosome refit", fontweight="bold")
    ax.text(3.5, -1.1, "Prunus", ha="center", va="bottom", color=GENUS_COLOR["prunus"], fontweight="bold")
    ax.text(16.0, -1.1, "Pyrus", ha="center", va="bottom", color=GENUS_COLOR["pyrus"], fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("log2 odds ratio (display capped at 98.5th percentile)")
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(out_base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    source_dir = root / "results/biological_cases/publication_v3_crossgenus_go"
    metric_dir = root / "results/metrics/publication_v4_go_stability"
    figure_dir = root / "results/figures"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_v3": source_dir / "summary.json",
        "terms": source_dir / "robust_replicated_terms.tsv",
        "leaf": root / "results/tables/publication_v3_crossgenus_go_leaf_terms.tsv",
        "loo": source_dir / "curated_no_iea_leave_one_chromosome.tsv",
        "permutation": source_dir / "curated_no_iea_permutation_null.tsv.gz",
        "obo": root / "data/raw/publication_v3_go/go-basic.obo",
        "contract": root / "docs/publication_v4_posthoc_analysis_contract_20260803.md",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required GO inputs: {missing}")

    summary_v3 = json.loads(paths["summary_v3"].read_text(encoding="utf-8"))
    terms_all_layers = pd.read_csv(paths["terms"], sep="\t")
    terms = terms_all_layers.loc[terms_all_layers["evidence_layer"] == "curated_no_iea"].copy()
    leaf = pd.read_csv(paths["leaf"], sep="\t")
    loo = pd.read_csv(paths["loo"], sep="\t")
    permutation = pd.read_csv(paths["permutation"], sep="\t")
    if len(terms) != 52 or len(leaf) != 18 or len(permutation) != 10_000:
        raise ValueError("Frozen GO dimensions changed")
    if set(leaf["term_id"]) != set(TERM_MODULE):
        raise ValueError("Fixed 18-term module map does not match the leaf table")
    if summary_v3.get("primary_robust_replicated_terms") != 52:
        raise ValueError("Frozen primary robust term count is not 52")

    leaf = leaf.copy()
    leaf["module"] = leaf["term_id"].map(TERM_MODULE)
    leaf["min_genus_positive_orthogroups"] = leaf[["prunus_term_positive", "pyrus_term_positive"]].min(axis=1)
    graph, obo_names = parse_obo(paths["obo"])
    absent = sorted(set(leaf["term_id"]) - set(graph))
    if absent:
        raise ValueError(f"Leaf GO terms missing from OBO: {absent}")
    edges = ontology_edges(leaf, graph)
    loo = loo.merge(terms[["term_id", "term_name", "q_crossgenus"]], on="term_id", how="left", validate="many_to_one")
    loo["log2_odds_ratio"] = np.log2(loo["odds_ratio_haldane"].astype(float))
    loo_leaf = loo[loo["term_id"].isin(leaf["term_id"])].copy()
    loo_leaf["module"] = loo_leaf["term_id"].map(TERM_MODULE)

    leaf.to_csv(metric_dir / "go_leaf_forest.tsv", sep="\t", index=False)
    loo.to_csv(metric_dir / "go_loo_stability_all52.tsv", sep="\t", index=False)
    loo_leaf.to_csv(metric_dir / "go_loo_stability_leaf18.tsv", sep="\t", index=False)
    edges.to_csv(metric_dir / "go_ontology_edges.tsv", sep="\t", index=False)
    permutation.to_csv(metric_dir / "go_permutation_null.tsv.gz", sep="\t", index=False, compression="gzip")

    main_base = figure_dir / "publication_v4_go_robustness"
    supplement_base = figure_dir / "publication_v4_go_loo_all52"
    plot_main(leaf, loo_leaf, permutation, edges, main_base)
    plot_full_stability(terms, loo, supplement_base)

    outputs = [main_base.with_suffix(f".{x}") for x in ["png", "pdf", "svg"]] + [
        supplement_base.with_suffix(f".{x}") for x in ["png", "pdf", "svg"]
    ]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "posthoc_descriptive_seed23_no_decision_authority",
        "scientific_decision_authority": False,
        "model_selection_allowed": False,
        "candidate_selection_allowed": False,
        "malus_accessed": False,
        "tested_universe_terms": int(summary_v3["layers"]["curated_no_iea"]["terms_eligible_both"]),
        "robust_replicated_terms": len(terms),
        "fixed_leaf_terms": len(leaf),
        "leave_one_chromosome_refits": len(loo),
        "prunus_chromosomes": int(loo.loc[loo["genus"] == "prunus", "left_out_chromosome"].nunique()),
        "pyrus_chromosomes": int(loo.loc[loo["genus"] == "pyrus", "left_out_chromosome"].nunique()),
        "permutations": len(permutation),
        "null_maximum_hits": int(permutation["replicated_fdr_hits"].max()),
        "observed_hits": 52,
        "empirical_p": 1 / 10001,
        "ontology_edge_method": "within each frozen module: maximum spanning tree of GO ancestor-set Jaccard plus all pairs with Jaccard >= 0.45; no effect sizes used",
        "ontology_edges": len(edges),
        "modules": {m: int((leaf["module"] == m).sum()) for m in MODULE_ORDER},
        "inputs": {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths.values()},
        "outputs": {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs},
    }
    (metric_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "robust_replicated_terms", "fixed_leaf_terms", "leave_one_chromosome_refits", "ontology_edges"]}, indent=2))


if __name__ == "__main__":
    main()
