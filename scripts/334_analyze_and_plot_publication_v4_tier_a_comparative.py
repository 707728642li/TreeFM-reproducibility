#!/usr/bin/env python3
"""Build Tier-A comparative-genomics source data and publication-v4 figure."""

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
import numpy as np
import pandas as pd
from Bio import Phylo
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


MODULE_ORDER = [
    "transcriptional relay",
    "ABA metabolism/signaling",
    "receptor/transport/metabolism",
    "unresolved stress protein",
]
MODULE_COLOR = {
    "transcriptional relay": "#3B82A0",
    "ABA metabolism/signaling": "#D9822B",
    "receptor/transport/metabolism": "#6A8E3A",
    "unresolved stress protein": "#8B5E9F",
}
SHORT_LABEL = {
    1: "AP2/ERF",
    2: "WRKY-I",
    3: "PP2C",
    4: "WRKY-II",
    5: "SUP",
    6: "MAPKKK",
    7: "NCED/CCD",
    8: "VQ",
    9: "B-box",
    10: "DUF26-CRK",
    11: "bHLH",
    12: "GAPDH",
    13: "ABC-2",
    14: "LRR-RLK",
}
EVIDENCE_LABELS = {
    "leaf_go": "Leaf GO",
    "direction_conserved": "2-genus\ndirection",
    "gbox_both_genera": "2-genus\nG-box",
    "pfam_three_way": "3-way\nPfam",
    "h3k4me3": "Prunus\nH3K4me3",
    "strict_matched_control": "Matched\ncontrol",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def tree_layout(tree) -> tuple[dict[object, float], dict[object, float]]:
    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)
    terminals = tree.get_terminals()
    y_pos: dict[object, float] = {tip: float(i) for i, tip in enumerate(terminals)}

    def assign(clade) -> float:
        if clade in y_pos:
            return y_pos[clade]
        values = [assign(child) for child in clade.clades]
        y_pos[clade] = float(np.mean(values))
        return y_pos[clade]

    assign(tree.root)
    return depths, y_pos


def draw_tree(ax, tree, annotations: pd.DataFrame) -> list[str]:
    depths, y_pos = tree_layout(tree)
    ann = annotations.set_index("slug")
    x_max = max(depths.values())
    for clade in tree.find_clades(order="preorder"):
        x_here = depths[clade]
        if clade.clades:
            ys = [y_pos[ch] for ch in clade.clades]
            ax.plot([x_here, x_here], [min(ys), max(ys)], color="#4B5563", linewidth=0.75)
            for child in clade.clades:
                ax.plot([x_here, depths[child]], [y_pos[child], y_pos[child]], color="#4B5563", linewidth=0.75)
        else:
            slug = clade.name
            scientific = ann.loc[slug, "scientific_name"]
            color = "#A42032" if slug in {"prunus_persica", "pyrus_pyrifolia"} else "#253238"
            weight = "bold" if slug in {"prunus_persica", "pyrus_pyrifolia"} else "normal"
            ax.text(x_max * 1.035, y_pos[clade], scientific, va="center", ha="left", fontsize=7.8, color=color, fontweight=weight, fontstyle="italic")
    ax.set_xlim(0, x_max * 1.86)
    ax.set_ylim(-0.7, len(tree.get_terminals()) - 0.3)
    ax.axis("off")
    return [tip.name for tip in tree.get_terminals()]


def plot_figure(
    tree,
    annotations: pd.DataFrame,
    copy_long: pd.DataFrame,
    candidates: pd.DataFrame,
    evidence: pd.DataFrame,
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
    fig = plt.figure(figsize=(19.0, 15.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 0.82])
    top = gs[0].subgridspec(1, 2, width_ratios=[1.12, 1.88], wspace=0.02)
    bottom = gs[1].subgridspec(1, 2, width_ratios=[1.26, 0.74], wspace=0.16)
    ax_tree = fig.add_subplot(top[0, 0])
    ax_heat = fig.add_subplot(top[0, 1])
    ax_evidence = fig.add_subplot(bottom[0, 0])
    ax_breadth = fig.add_subplot(bottom[0, 1])

    species_order = draw_tree(ax_tree, tree, annotations)
    ax_tree.set_title("A  Phylogeny", loc="left", fontweight="bold")
    candidate_order = candidates.sort_values("catalog_rank")["orthogroup"].tolist()
    matrix = (
        copy_long.pivot(index="species", columns="orthogroup", values="copy_number")
        .reindex(index=species_order, columns=candidate_order)
        .astype(int)
    )
    display = np.log2(matrix.to_numpy() + 1)
    im = ax_heat.imshow(display, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=max(5.0, float(np.quantile(display, 0.98))))
    ax_heat.set_yticks([])
    labels = [SHORT_LABEL[int(x)] for x in candidates.sort_values("catalog_rank")["catalog_rank"]]
    ax_heat.set_xticks(np.arange(len(labels)))
    ax_heat.set_xticklabels(labels, rotation=48, ha="right", fontsize=8.6)
    ax_heat.set_title("A  Tier-A copy-number landscape (all 26 species × all 14 families)", loc="left", fontweight="bold")
    # Family modules as a non-statistical annotation strip.
    candidate_module = candidates.set_index("orthogroup")["mechanism_module"].to_dict()
    for x, orthogroup in enumerate(candidate_order):
        ax_heat.add_patch(Rectangle((x - 0.5, len(species_order) - 0.42), 1.0, 0.35, facecolor=MODULE_COLOR[candidate_module[orthogroup]], edgecolor="none", clip_on=False))
    for slug in ["prunus_persica", "pyrus_pyrifolia", "fragaria_vesca"]:
        idx = species_order.index(slug)
        ax_heat.add_patch(Rectangle((-0.5, idx - 0.5), len(candidate_order), 1.0, fill=False, edgecolor="#E7C55A" if slug == "fragaria_vesca" else "#D1495B", linewidth=1.2))
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.015)
    ticks = np.arange(0, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(2**x - 1)) for x in ticks])
    cbar.set_label("OrthoFinder gene representatives (log2 scale)")
    ax_heat.legend(
        handles=[Patch(facecolor=MODULE_COLOR[m], label=m) for m in MODULE_ORDER],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.23),
        ncol=2,
        fontsize=7.8,
    )

    # B: complete evidence matrix, not a selected subset.
    evidence_cols = list(EVIDENCE_LABELS)
    ordered = evidence.sort_values("catalog_rank").reset_index(drop=True)
    y = np.arange(len(ordered))[::-1]
    for x, col in enumerate(evidence_cols):
        values = as_bool(ordered[col])
        for yi, present, module in zip(y, values, ordered["mechanism_module"]):
            ax_evidence.scatter(
                x,
                yi,
                s=82,
                color=MODULE_COLOR[module] if present else "#E5E7EB",
                edgecolor="#374151" if present else "#C7CDD4",
                linewidth=0.7,
                zorder=3,
            )
    # Literature evidence is ordinal context only.
    lit_x = len(evidence_cols)
    lit_sizes = 42 + ordered["literature_evidence_level"].astype(float).to_numpy() * 38
    ax_evidence.scatter(
        np.full(len(y), lit_x),
        y,
        s=lit_sizes,
        c=ordered["literature_evidence_level"].astype(float),
        cmap="Greys",
        vmin=0,
        vmax=4,
        edgecolor="#374151",
        linewidth=0.7,
        zorder=3,
    )
    ax_evidence.set_xticks(np.arange(len(evidence_cols) + 1))
    ax_evidence.set_xticklabels([EVIDENCE_LABELS[x] for x in evidence_cols] + ["Literature\nlevel (0–4)"], fontsize=8.1)
    ylabels = [f"{int(r.catalog_rank):02d}  {SHORT_LABEL[int(r.catalog_rank)]}  ({r.orthogroup})" for r in ordered.itertuples(index=False)]
    ax_evidence.set_yticks(y)
    ax_evidence.set_yticklabels(ylabels, fontsize=8.1)
    for tick, module in zip(ax_evidence.get_yticklabels(), ordered["mechanism_module"]):
        tick.set_color(MODULE_COLOR[module])
    ax_evidence.set_xlim(-0.7, len(evidence_cols) + 0.7)
    ax_evidence.set_ylim(-0.8, len(ordered) - 0.2)
    ax_evidence.set_title("B  Every Tier-A family is shown across every audited evidence layer", loc="left", fontweight="bold")
    ax_evidence.grid(color="#EEF0F2", linewidth=0.8)
    ax_evidence.tick_params(axis="both", length=0)
    for spine in ax_evidence.spines.values():
        spine.set_visible(False)

    # C: phylogenetic breadth and total family size.
    breadth = candidates.sort_values("catalog_rank").reset_index(drop=True)
    for yi, row in zip(y, breadth.itertuples(index=False)):
        color = MODULE_COLOR[row.mechanism_module]
        ax_breadth.plot([0, row.species_present], [yi, yi], color=color, linewidth=2.2, alpha=0.72)
        ax_breadth.scatter(row.species_present, yi, s=65 + 1.4 * min(row.total_copy_number, 120), color=color, edgecolor="white", linewidth=0.8, zorder=3)
    ax_breadth.axvline(26, color="#4B5563", linestyle="--", linewidth=0.8)
    ax_breadth.set_yticks(y)
    ax_breadth.set_yticklabels([SHORT_LABEL[int(x)] for x in breadth["catalog_rank"]], fontsize=8.2)
    for tick, module in zip(ax_breadth.get_yticklabels(), breadth["mechanism_module"]):
        tick.set_color(MODULE_COLOR[module])
    ax_breadth.set_xlim(0, 27.5)
    ax_breadth.set_xlabel("Species with ≥1 family member (of 26)")
    ax_breadth.set_title("C  Phylogenetic breadth", loc="left", fontweight="bold")
    ax_breadth.spines[["top", "right", "left"]].set_visible(False)
    ax_breadth.grid(axis="x", alpha=0.18)
    size_values = [10, 50, 100]
    handles = [ax_breadth.scatter([], [], s=65 + 1.4 * x, color="#6B7280", edgecolor="white", label=f"{x} total copies") for x in size_values]
    ax_breadth.legend(handles=handles, frameon=False, loc="lower right", fontsize=7.7)

    fig.suptitle(
        "Frozen Tier-A candidates are conserved plant families supported by layered, non-causal evidence",
        fontsize=15.0,
        fontweight="bold",
    )
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(out_base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    metric_dir = root / "results/metrics/publication_v4_tier_a_comparative"
    figure_dir = root / "results/figures"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "membership_extract": metric_dir / "tier_a_orthogroups.txt",
        "tier_a": root / "results/biological_cases/publication_v3_crossgenus_candidates/tier_a_candidates.tsv",
        "evidence": root / "metadata/publication_v3_tier_a_evidence_matrix.tsv",
        "chromatin": root / "results/biological_cases/publication_v3_tier_a_chromatin_overlap/tier_a_orthogroup_chromatin_overlap.tsv",
        "species_tree": root / "results/metrics/publication_v4_corpus_phylogeny/species_tree_named_rooted.nwk",
        "species_annotations": root / "results/metrics/publication_v4_corpus_phylogeny/technical_species_annotations.tsv",
        "contract": root / "docs/publication_v4_posthoc_analysis_contract_20260803.md",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Tier-A inputs: {missing}")

    tier = pd.read_csv(paths["tier_a"], sep="\t")
    evidence_v3 = pd.read_csv(paths["evidence"], sep="\t")
    chromatin = pd.read_csv(paths["chromatin"], sep="\t")
    annotations_all = pd.read_csv(paths["species_annotations"], sep="\t")
    annotations = annotations_all.drop_duplicates("slug")[['slug', 'scientific_name', 'order', 'family', 'life_form', 'life_form_group']].copy()
    tree = Phylo.read(paths["species_tree"], "newick")
    species = [tip.name for tip in tree.get_terminals()]
    if len(species) != 26 or set(species) != set(annotations["slug"]):
        raise ValueError("The named tree and technical species annotations do not match exactly")
    if len(tier) != 14 or set(tier["tier"]) != {"A"}:
        raise ValueError("Frozen Tier-A catalog is not exactly 14 rows")

    raw_lines: dict[str, list[str]] = {}
    for line in paths["membership_extract"].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        orthogroup, members = line.split(":", 1)
        raw_lines[orthogroup] = members.strip().split()
    expected_orthogroups = tier.sort_values("catalog_rank")["orthogroup"].tolist()
    if set(raw_lines) != set(expected_orthogroups) or len(raw_lines) != 14:
        raise ValueError("The server-side membership extract does not match the frozen Tier-A catalog")
    unknown_species: set[str] = set()
    long_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for orthogroup in expected_orthogroups:
        counts = Counter()
        for member in raw_lines[orthogroup]:
            slug = member.split("|", 1)[0]
            counts[slug] += 1
            if slug not in species:
                unknown_species.add(slug)
            member_rows.append({"orthogroup": orthogroup, "species": slug, "member_id": member})
        for slug in species:
            long_rows.append({"orthogroup": orthogroup, "species": slug, "copy_number": int(counts[slug])})
    if unknown_species:
        raise ValueError(f"Unexpected species in membership extract: {sorted(unknown_species)}")
    copy_long = pd.DataFrame(long_rows)
    members = pd.DataFrame(member_rows)

    candidates = tier.merge(
        evidence_v3[["orthogroup", "retrospective_family_label", "mechanism_module", "pfam_support_status", "literature_evidence_level", "strict_matched_control"]],
        on="orthogroup",
        how="left",
        validate="one_to_one",
    )
    copy_summary = copy_long.groupby("orthogroup").agg(
        species_present=("copy_number", lambda x: int((x > 0).sum())),
        total_copy_number=("copy_number", "sum"),
        median_copy_number=("copy_number", "median"),
        max_copy_number=("copy_number", "max"),
    ).reset_index()
    candidates = candidates.merge(copy_summary, on="orthogroup", how="left", validate="one_to_one")
    candidates["display_label"] = candidates["catalog_rank"].map(SHORT_LABEL)

    evidence = candidates[["catalog_rank", "orthogroup", "display_label", "mechanism_module", "literature_evidence_level", "strict_matched_control"]].copy()
    evidence["leaf_go"] = as_bool(tier["leaf_go"])
    evidence["direction_conserved"] = as_bool(tier["direction_conserved"])
    evidence["gbox_both_genera"] = as_bool(tier["gbox_both_genera"])
    evidence["pfam_three_way"] = candidates["pfam_support_status"].eq("cross_genus_anchor_supported")
    h3_map = chromatin.set_index("orthogroup")["direction_concordant_h3k4me3"].astype(int).to_dict()
    evidence["h3k4me3"] = evidence["orthogroup"].map(h3_map).fillna(0).astype(int) > 0

    copy_long = copy_long.merge(annotations, left_on="species", right_on="slug", how="left", validate="many_to_one").drop(columns="slug")
    copy_long = copy_long.merge(candidates[["orthogroup", "catalog_rank", "display_label", "mechanism_module"]], on="orthogroup", how="left", validate="many_to_one")
    copy_long.to_csv(metric_dir / "tier_a_copy_number.tsv", sep="\t", index=False)
    members.to_csv(metric_dir / "tier_a_members.tsv.gz", sep="\t", index=False, compression="gzip")
    candidates.to_csv(metric_dir / "tier_a_candidate_summary.tsv", sep="\t", index=False)
    evidence.to_csv(metric_dir / "tier_a_evidence_layers.tsv", sep="\t", index=False)
    annotations.assign(tree_order=annotations["slug"].map({x: i for i, x in enumerate(species)})).sort_values("tree_order").to_csv(metric_dir / "tier_a_species_order.tsv", sep="\t", index=False)

    out_base = figure_dir / "publication_v4_tier_a_comparative"
    plot_figure(tree, annotations, copy_long, candidates, evidence, out_base)
    outputs = [out_base.with_suffix(f".{x}") for x in ["png", "pdf", "svg"]]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "posthoc_descriptive_frozen_tier_a",
        "scientific_decision_authority": False,
        "candidate_selection_allowed": False,
        "causal_claim": False,
        "malus_accessed": False,
        "orthogroups": len(expected_orthogroups),
        "species": len(species),
        "copy_number_cells": len(copy_long),
        "member_gene_representatives": len(members),
        "species_present_min": int(candidates["species_present"].min()),
        "species_present_max": int(candidates["species_present"].max()),
        "families_present_all26": int((candidates["species_present"] == 26).sum()),
        "three_way_pfam_supported": int(evidence["pfam_three_way"].sum()),
        "h3k4me3_intersecting": int(evidence["h3k4me3"].sum()),
        "modules": candidates["mechanism_module"].value_counts().to_dict(),
        "inputs": {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths.values()},
        "outputs": {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs},
    }
    (metric_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "orthogroups", "species", "copy_number_cells", "families_present_all26", "three_way_pfam_supported", "h3k4me3_intersecting"]}, indent=2))


if __name__ == "__main__":
    main()
