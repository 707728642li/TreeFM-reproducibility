#!/usr/bin/env python3
"""Build the publication-v4 corpus-composition and 26-species phylogeny panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ARM_ORDER = ["tree", "herb", "random_plant", "phylogc_match"]
ARM_LABEL = {
    "tree": "Tree",
    "herb": "Herb",
    "random_plant": "RandomPlant",
    "phylogc_match": "PhyloGCMatch",
}
ARM_COLOR = {
    "tree": "#2E8B57",
    "herb": "#8B5CF6",
    "random_plant": "#D98E04",
    "phylogc_match": "#2E75B6",
}
ROLE_COLOR = {
    "training_reference": "#4C78A8",
    "control_train": "#9C6ADE",
    "development_reference": "#72B7B2",
    "primary_test": "#D1495B",
    "primary_test_candidate": "#D1495B",
    "secondary_qc": "#A0A0A0",
}
LIFE_COLOR = {
    "tree": "#3B7A57",
    "shrub": "#79A95B",
    "herb": "#B58AD8",
    "herb_shrub": "#D2A6C8",
    "other": "#A0A0A0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_jitter(text: str, width: float = 0.22) -> float:
    raw = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return ((raw / 0xFFFFFFFF) - 0.5) * 2.0 * width


def collapse_life_form(value: object) -> str:
    text = str(value).strip().lower()
    if text in LIFE_COLOR:
        return text
    if "herb" in text and "shrub" in text:
        return "herb_shrub"
    if "tree" in text:
        return "tree"
    if "shrub" in text:
        return "shrub"
    if "herb" in text:
        return "herb"
    return "other"


def load_species_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split(":", 1)
        mapping[left.strip()] = Path(right.strip()).stem
    return mapping


def tree_layout(tree) -> tuple[dict[object, float], dict[object, float]]:
    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)
    terminals = tree.get_terminals()
    y_terminal = {tip: float(i) for i, tip in enumerate(terminals)}
    y_pos: dict[object, float] = dict(y_terminal)

    def assign(clade) -> float:
        if clade in y_pos:
            return y_pos[clade]
        values = [assign(child) for child in clade.clades]
        y_pos[clade] = float(np.mean(values))
        return y_pos[clade]

    assign(tree.root)
    return depths, y_pos


def draw_tree_panel(ax, tree, annotations: pd.DataFrame, membership: pd.DataFrame) -> None:
    depths, y_pos = tree_layout(tree)
    tip_by_slug = {tip.name: tip for tip in tree.get_terminals()}
    ann = annotations.set_index("slug")
    mem = membership.set_index("slug")
    x_max = max(depths.values())
    label_x = x_max + x_max * 0.025
    role_x = x_max + x_max * 0.46
    tile_start = x_max + x_max * 0.58
    tile_w = x_max * 0.070

    for clade in tree.find_clades(order="preorder"):
        x_here = depths[clade]
        if clade.clades:
            child_y = [y_pos[child] for child in clade.clades]
            ax.plot([x_here, x_here], [min(child_y), max(child_y)], color="#8B98A5", lw=0.75)
            for child in clade.clades:
                ax.plot(
                    [x_here, depths[child]],
                    [y_pos[child], y_pos[child]],
                    color="#4C5967",
                    lw=0.85,
                )
            if clade.confidence is not None and float(clade.confidence) >= 0.80:
                ax.text(
                    x_here,
                    y_pos[clade] - 0.25,
                    f"{float(clade.confidence):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color="#6B7280",
                )

    for slug, tip in tip_by_slug.items():
        row = ann.loc[slug]
        y = y_pos[tip]
        ax.text(
            label_x,
            y,
            row["scientific_name"],
            ha="left",
            va="center",
            fontsize=7.0,
            fontstyle="italic",
            color="#17212B",
        )
        role = row["analysis_tier"]
        ax.scatter(
            [role_x],
            [y],
            s=18,
            color=ROLE_COLOR.get(role, "#A0A0A0"),
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )
        for j, arm in enumerate(ARM_ORDER):
            face = ARM_COLOR[arm] if int(mem.loc[slug, arm]) else "white"
            rect = Rectangle(
                (tile_start + j * tile_w, y - 0.32),
                tile_w * 0.72,
                0.64,
                facecolor=face,
                edgecolor=ARM_COLOR[arm],
                linewidth=0.55,
            )
            ax.add_patch(rect)

    ax.text(role_x, -1.35, "role", ha="center", va="center", fontsize=6.4, color="#52606D")
    for j, arm in enumerate(ARM_ORDER):
        ax.text(
            tile_start + j * tile_w + tile_w * 0.36,
            -1.35,
            ARM_LABEL[arm].replace("RandomPlant", "Random").replace("PhyloGCMatch", "PhyloGC"),
            ha="center",
            va="center",
            rotation=50,
            fontsize=5.7,
            color=ARM_COLOR[arm],
        )
    ax.set_xlim(-x_max * 0.02, tile_start + len(ARM_ORDER) * tile_w + x_max * 0.03)
    ax.set_ylim(-2.1, len(tree.get_terminals()) - 0.25)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("A  26-species phylogeny and corpus membership", loc="left", fontsize=11.5, weight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()

    input_paths = {
        "tree": root / "results/metrics/publication_v4_species_phylogeny/species_tree_ids.nwk",
        "species_ids": root / "metadata/publication_v4_species_ids.txt",
        "species_panel": root / "config/species_panel_seed.tsv",
        "pyrus_source": root / "config/publication_v3_pyrus_genome_sources.tsv",
        "embedding_manifest": root / "results/embeddings/plantcad_dapt_publication_v3/base/seed_0/manifest.tsv",
        "corpus_shards": root / "metadata/publication_v3_dapt_corpus_shards.tsv",
        "corpus_audit": root / "metadata/publication_v3_dapt_corpus_audit.tsv",
        "feature_match": root / "metadata/publication_v3_phylogc_selected_feature_match.tsv",
        "candidate_features": root / "metadata/publication_v3_phylogc_genome_features.tsv",
        "selected_phylogc": root / "config/publication_v3_phylogc_match_selected.tsv",
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    species_map = load_species_map(input_paths["species_ids"])
    if set(species_map) != {str(i) for i in range(26)}:
        raise ValueError("Species-ID mapping must contain exactly numeric IDs 0..25")

    tree = Phylo.read(input_paths["tree"], "newick")
    numeric_tips = {tip.name for tip in tree.get_terminals()}
    if numeric_tips != set(species_map):
        raise ValueError(f"Tree tips do not match SpeciesIDs: {sorted(numeric_tips)}")
    for tip in tree.get_terminals():
        tip.name = species_map[tip.name]
    outgroup = next(tip for tip in tree.get_terminals() if tip.name == "magnolia_biondii")
    tree.root_with_outgroup(outgroup)
    tree.ladderize(reverse=True)

    panel = pd.read_csv(input_paths["species_panel"], sep="\t")
    manifest = pd.read_csv(input_paths["embedding_manifest"], sep="\t")
    annotations = manifest.merge(
        panel[["slug", "scientific_name", "order", "family", "life_form"]],
        on="slug",
        how="left",
        validate="one_to_one",
    )
    missing_annotation = annotations.loc[
        annotations[["scientific_name", "order", "family", "life_form"]].isna().any(axis=1), "slug"
    ].tolist()
    if missing_annotation == ["pyrus_pyrifolia"]:
        pyrus_source = pd.read_csv(input_paths["pyrus_source"], sep="\t")
        pyrus_names = pyrus_source.loc[pyrus_source["slug"] == "pyrus_pyrifolia", "scientific_name"].dropna().unique()
        if pyrus_names.tolist() != ["Pyrus pyrifolia"]:
            raise ValueError("Pyrus source identity is not uniquely documented")
        mask = annotations["slug"] == "pyrus_pyrifolia"
        annotations.loc[mask, ["scientific_name", "order", "family", "life_form"]] = [
            "Pyrus pyrifolia",
            "Rosales",
            "Rosaceae",
            "tree",
        ]
    if annotations[["scientific_name", "order", "family", "life_form"]].isna().any().any():
        raise ValueError("Technical species annotations are incomplete")
    annotations["life_form_group"] = annotations["life_form"].map(collapse_life_form)

    shards = pd.read_csv(input_paths["corpus_shards"], sep="\t")
    corpus_audit = pd.read_csv(input_paths["corpus_audit"], sep="\t").set_index("corpus").loc[ARM_ORDER].reset_index()
    if set(shards["corpus"]) != set(ARM_ORDER):
        raise ValueError("Corpus shard table must contain all four DAPT arms and no extra arm")
    if not (corpus_audit["windows"] == 1_000_000).all() or not (corpus_audit["bases"] == 512_000_000).all():
        raise ValueError("Equal-budget contract failed")
    if shards.duplicated(["corpus", "slug"]).any():
        raise ValueError("Duplicate corpus/species rows")
    shards["life_form_group"] = shards["life_form"].map(collapse_life_form)

    member_wide = (
        shards.assign(present=1)
        .pivot_table(index="slug", columns="corpus", values="present", fill_value=0, aggfunc="max")
        .reindex(columns=ARM_ORDER, fill_value=0)
    )
    membership = pd.DataFrame({"slug": annotations["slug"]}).join(member_wide, on="slug").fillna(0)
    for arm in ARM_ORDER:
        membership[arm] = membership[arm].astype(int)

    feature_match = pd.read_csv(input_paths["feature_match"], sep="\t")
    if len(feature_match) != 9 or feature_match["feature"].duplicated().any():
        raise ValueError("Frozen PhyloGC feature match must contain exactly nine unique features")
    candidate_features = pd.read_csv(input_paths["candidate_features"], sep="\t")
    selected = pd.read_csv(input_paths["selected_phylogc"], sep="\t")
    if len(selected) != 8 or selected["slug"].nunique() != 8:
        raise ValueError("PhyloGCMatch selected set must contain eight species")

    metric_dir = root / "results/metrics/publication_v4_corpus_phylogeny"
    figure_dir = root / "results/figures"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(metric_dir / "technical_species_annotations.tsv", sep="\t", index=False)
    membership.to_csv(metric_dir / "technical_species_corpus_membership.tsv", sep="\t", index=False)
    shards.to_csv(metric_dir / "corpus_species_source.tsv", sep="\t", index=False)
    corpus_audit.to_csv(metric_dir / "corpus_equal_budget_summary.tsv", sep="\t", index=False)
    feature_match.to_csv(metric_dir / "phylogc_feature_match.tsv", sep="\t", index=False)
    candidate_features.to_csv(metric_dir / "phylogc_candidate_features.tsv", sep="\t", index=False)

    tree_named = metric_dir / "species_tree_named_rooted.nwk"
    Phylo.write(tree, tree_named, "newick")
    support_rows = []
    for idx, clade in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        support_rows.append(
            {
                "node_id": f"N{idx:02d}",
                "support": clade.confidence,
                "descendant_tips": ";".join(sorted(t.name for t in clade.get_terminals())),
            }
        )
    pd.DataFrame(support_rows).to_csv(metric_dir / "species_tree_support.tsv", sep="\t", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(15.5, 10.2), constrained_layout=False)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.55, 1.0], height_ratios=[0.9, 0.9, 1.15], hspace=0.48, wspace=0.28)
    ax_tree = fig.add_subplot(grid[:, 0])
    ax_comp = fig.add_subplot(grid[0, 1])
    ax_gc = fig.add_subplot(grid[1, 1])
    ax_match = fig.add_subplot(grid[2, 1])

    draw_tree_panel(ax_tree, tree, annotations, membership)

    # Panel B: species composition; window totals are identical by design.
    life_order = ["tree", "shrub", "herb", "herb_shrub", "other"]
    life_counts = (
        shards.groupby(["corpus", "life_form_group"])["slug"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=ARM_ORDER, columns=life_order, fill_value=0)
    )
    x = np.arange(len(ARM_ORDER))
    bottom = np.zeros(len(ARM_ORDER), dtype=float)
    for life in life_order:
        values = life_counts[life].to_numpy(dtype=float)
        if values.sum() == 0:
            continue
        ax_comp.bar(x, values, bottom=bottom, color=LIFE_COLOR[life], width=0.72, edgecolor="white", linewidth=0.5, label=life.replace("_", "/"))
        bottom += values
    for idx, arm in enumerate(ARM_ORDER):
        total = int(corpus_audit.set_index("corpus").loc[arm, "species"])
        ax_comp.text(idx, bottom[idx] + 0.45, f"{total} species\n1.0M windows", ha="center", va="bottom", fontsize=7.2, color=ARM_COLOR[arm], weight="bold")
    ax_comp.set_xticks(x, [ARM_LABEL[a] for a in ARM_ORDER], rotation=16, ha="right")
    ax_comp.set_ylabel("Species in DAPT corpus")
    ax_comp.set_title("B  Equal-budget corpus composition", loc="left", fontsize=11.5, weight="bold")
    ax_comp.spines[["top", "right"]].set_visible(False)
    ax_comp.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax_comp.set_axisbelow(True)
    ax_comp.set_ylim(0, float(bottom.max()) + 5.5)
    ax_comp.legend(frameon=False, ncol=3, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.96))

    # Panel C: per-species GC distributions with weighted corpus means.
    for idx, arm in enumerate(ARM_ORDER):
        part = shards.loc[shards["corpus"] == arm].copy()
        xs = [idx + stable_jitter(f"{arm}|{slug}") for slug in part["slug"]]
        ax_gc.scatter(xs, part["mean_gc_fraction"], s=23, color=ARM_COLOR[arm], alpha=0.72, edgecolor="white", linewidth=0.35)
        mean_gc = float(corpus_audit.set_index("corpus").loc[arm, "mean_gc_fraction"])
        ax_gc.plot([idx - 0.28, idx + 0.28], [mean_gc, mean_gc], color="#17212B", lw=2.0)
        ax_gc.text(idx, mean_gc + 0.004, f"{mean_gc:.3f}", ha="center", va="bottom", fontsize=6.8, color="#17212B")
    ax_gc.set_xticks(x, [ARM_LABEL[a] for a in ARM_ORDER], rotation=16, ha="right")
    ax_gc.set_ylabel("Mean GC fraction per species shard")
    ax_gc.set_title("C  Composition differs despite identical exposure", loc="left", fontsize=11.5, weight="bold")
    ax_gc.spines[["top", "right"]].set_visible(False)
    ax_gc.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax_gc.set_axisbelow(True)

    # Panel D: all nine frozen standardized matching differences.
    label_map = {
        "taxonomy_distance_hevea_brasiliensis": "Taxonomy distance: Hevea",
        "taxonomy_distance_prunus_persica": "Taxonomy distance: Prunus",
        "taxonomy_distance_pyrus_pyrifolia": "Taxonomy distance: Pyrus",
        "taxonomy_distance_malus_domestica": "Taxonomy distance: Malus*",
        "gc_fraction": "GC fraction",
        "repetitive_21mer_fraction": "Repetitive 21-mer fraction",
        "source_softmask_fraction": "Soft-masked fraction",
        "gene_fraction": "Gene fraction",
        "cds_fraction": "CDS fraction",
        "intergenic_fraction": "Intergenic fraction",
    }
    match = feature_match.copy()
    match["label"] = match["feature"].map(label_map).fillna(match["feature"])
    match = match.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(match))
    ax_match.axvspan(-0.5, 0.5, color="#E8F3EE", zorder=0)
    ax_match.axvline(0, color="#52606D", lw=0.9)
    ax_match.scatter(match["standardized_difference"], y, s=42, color=ARM_COLOR["phylogc_match"], edgecolor="white", linewidth=0.6, zorder=3)
    for yi, value in zip(y, match["standardized_difference"]):
        ax_match.plot([0, value], [yi, yi], color="#9FBAD1", lw=1.0, zorder=1)
    ax_match.set_yticks(y, match["label"])
    limit = max(0.65, float(np.abs(match["standardized_difference"]).max()) + 0.15)
    ax_match.set_xlim(-limit, limit)
    ax_match.set_xlabel("Selected PhyloGCMatch − Tree mean (standardized)")
    ax_match.set_title("D  Frozen phylogeny/composition matching", loc="left", fontsize=11.5, weight="bold")
    ax_match.spines[["top", "right"]].set_visible(False)
    ax_match.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax_match.set_axisbelow(True)
    ax_match.text(0.99, -0.20, "* pre-outcome taxonomy covariate only; Malus outcomes remain sealed", transform=ax_match.transAxes, ha="right", va="top", fontsize=6.3, color="#6B7280")

    role_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=6, label=label)
        for label, color in [
            ("training reference", ROLE_COLOR["training_reference"]),
            ("control train", ROLE_COLOR["control_train"]),
            ("development holdout", ROLE_COLOR["development_reference"]),
            ("primary heldout", ROLE_COLOR["primary_test"]),
            ("secondary QC", ROLE_COLOR["secondary_qc"]),
        ]
    ]
    ax_tree.legend(handles=role_handles, frameon=False, fontsize=6.2, ncol=2, loc="lower left", bbox_to_anchor=(0.0, -0.015))

    fig.suptitle("Equal-budget plant DAPT corpora occupy distinct phylogenetic and compositional domains", x=0.04, ha="left", fontsize=15, weight="bold", color="#17212B")
    fig.text(0.04, 0.012, "All four DAPT arms contain exactly 1,000,000 512-bp windows. Tree-specific attribution is tested against Herb, RandomPlant and PhyloGCMatch, not against Base alone.", ha="left", va="bottom", fontsize=7.5, color="#52606D")
    fig.subplots_adjust(left=0.04, right=0.985, top=0.935, bottom=0.065)

    stem = figure_dir / "publication_v4_corpus_phylogeny"
    output_hashes: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        output_hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256(path)
    plt.close(fig)

    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "publication_v4_posthoc_corpus_phylogeny",
        "analysis_tier": "posthoc_seed23_descriptive",
        "decision_authority": False,
        "tree_species": len(tree.get_terminals()),
        "tree_alignment_model": "FastTree 2 LG+CAT20 with SH-like local support",
        "display_root": "magnolia_biondii",
        "corpus_species": {row.corpus: int(row.species) for row in corpus_audit.itertuples()},
        "corpus_windows": {row.corpus: int(row.windows) for row in corpus_audit.itertuples()},
        "corpus_bases": {row.corpus: int(row.bases) for row in corpus_audit.itertuples()},
        "corpus_mean_gc": {row.corpus: float(row.mean_gc_fraction) for row in corpus_audit.itertuples()},
        "phylogc_matching_features": len(feature_match),
        "maximum_absolute_standardized_difference": float(feature_match["standardized_difference"].abs().max()),
        "maximum_absolute_standardized_difference_feature": str(feature_match.loc[feature_match["standardized_difference"].abs().idxmax(), "feature"]),
        "preoutcome_malus_taxonomy_covariate_reproduced": True,
        "malus_outcomes_accessed": False,
        "input_sha256": {key: sha256(path) for key, path in input_paths.items()},
        "output_sha256": output_hashes,
    }
    summary_path = metric_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
