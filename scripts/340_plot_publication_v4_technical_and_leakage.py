#!/usr/bin/env python3
"""Build publication-v4 technical effect and sequence-leakage QC figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


ARM_ORDER = ["tree", "herb", "random_plant", "phylogc_match"]
ARM_LABEL = {"tree": "Tree", "herb": "Herb", "random_plant": "RandomPlant", "phylogc_match": "PhyloGCMatch"}
ARM_COLOR = {"tree": "#2E8B57", "herb": "#8B5CF6", "random_plant": "#D98E04", "phylogc_match": "#2E75B6"}
TASK_ORDER = ["tis", "tts", "splice_donor", "splice_acceptor"]
TASK_LABEL = {"tis": "TIS", "tts": "TTS", "splice_donor": "Donor", "splice_acceptor": "Acceptor"}
TASK_COLOR = {"tis": "#CC6677", "tts": "#EE7733", "splice_donor": "#4477AA", "splice_acceptor": "#228833"}
SPECIES_ORDER = ["hevea_brasiliensis", "prunus_persica", "pyrus_pyrifolia"]
SPECIES_LABEL = {"hevea_brasiliensis": "Hevea", "prunus_persica": "Prunus", "pyrus_pyrifolia": "Pyrus"}
READOUT_ORDER = ["linear", "xgboost"]
READOUT_LABEL = {"linear": "linear", "xgboost": "XGBoost"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_order_frame() -> pd.DataFrame:
    rows = []
    order = 0
    for readout in READOUT_ORDER:
        for task in TASK_ORDER:
            for slug in SPECIES_ORDER:
                rows.append({"readout": readout, "task": task, "slug": slug, "cell_order": order})
                order += 1
    return pd.DataFrame(rows)


def plot_technical(scope96: pd.DataFrame, tree24: pd.DataFrame, out_base: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.9, "axes.titlesize": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(19.2, 15.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.08], height_ratios=[1.25, 0.75])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    order = cell_order_frame()
    scope = scope96.merge(order, on=["readout", "task", "slug"], validate="many_to_one").sort_values(["cell_order", "arm"])
    matrix = scope.pivot(index="cell_order", columns="arm", values="arm_minus_base").reindex(index=range(24), columns=ARM_ORDER) * 1000
    qmatrix = scope.pivot(index="cell_order", columns="arm", values="two_sided_q_bh_96").reindex(index=range(24), columns=ARM_ORDER)
    vmax = max(6.0, float(np.quantile(np.abs(matrix.to_numpy()), 0.98)))
    im = ax_a.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax_a.set_xticks(np.arange(4))
    ax_a.set_xticklabels([ARM_LABEL[x] for x in ARM_ORDER], rotation=35, ha="right")
    ordered_cells = order.sort_values("cell_order")
    labels = [f"{READOUT_LABEL[r.readout]} · {TASK_LABEL[r.task]} · {SPECIES_LABEL[r.slug]}" for r in ordered_cells.itertuples(index=False)]
    ax_a.set_yticks(np.arange(24))
    ax_a.set_yticklabels(labels, fontsize=7.2)
    for i in range(24):
        for j in range(4):
            value = matrix.iloc[i, j]
            star = "★" if qmatrix.iloc[i, j] < 0.05 else ""
            ax_a.text(j, i, f"{value:+.1f}{star}", ha="center", va="center", fontsize=6.8, color="white" if abs(value) > vmax * 0.52 else "#263238", fontweight="bold" if star else "normal")
    for boundary in [11.5]:
        ax_a.axhline(boundary, color="#111827", linewidth=1.5)
    ax_a.set_title("A  All 96 arm-versus-Base NovelFamily effects", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.036, pad=0.02)
    cbar.set_label("ΔAUPRC × 1,000")
    ax_a.text(0.99, -0.14, "★ BH q < 0.05 across all 96 post-hoc tests", transform=ax_a.transAxes, ha="right", va="top", fontsize=7.2, color="#4B5563")

    # B: paired Tree contrasts for every technical cell.
    tree = tree24.merge(order, on=["readout", "task", "slug"], validate="one_to_one").sort_values("cell_order")
    y = np.arange(24)[::-1]
    specs = [
        ("delta_vs_base", "delta_vs_base_ci_low", "delta_vs_base_ci_high", 0.16, "o", "#6B7280", "Tree − Base"),
        ("woody_control_gain", "woody_control_gain_ci_low", "woody_control_gain_ci_high", -0.16, "s", "#A42032", "Tree − strongest matched DAPT"),
    ]
    for value_col, low_col, high_col, offset, marker, color, label in specs:
        center = tree[value_col].to_numpy(float) * 1000
        low = tree[low_col].to_numpy(float) * 1000
        high = tree[high_col].to_numpy(float) * 1000
        ax_b.errorbar(center, y + offset, xerr=np.vstack([center - low, high - center]), fmt=marker, markersize=3.7, elinewidth=0.8, capsize=1.8, color=color, label=label, zorder=3)
    for yi, row in zip(y, tree.itertuples(index=False)):
        ax_b.plot([row.delta_vs_base * 1000, row.woody_control_gain * 1000], [yi + 0.16, yi - 0.16], color="#D1D5DB", linewidth=0.8, zorder=1)
    ax_b.axvline(0, color="#374151", linestyle="--", linewidth=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=7.2)
    ax_b.set_xlabel("ΔAUPRC × 1,000 (95% paired bootstrap CI)")
    ax_b.set_title("B  Tree does not outperform the strongest matched DAPT arm", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="lower left", fontsize=7.7)
    ax_b.grid(axis="x", alpha=0.18)
    ax_b.spines[["top", "right"]].set_visible(False)

    # C: arm-level directional count summary from all cells.
    summary_rows = []
    for arm in ARM_ORDER:
        sub = scope.loc[scope["arm"] == arm]
        summary_rows.append({
            "arm": arm,
            "positive": int((sub["arm_minus_base"] > 0).sum()),
            "ci_positive": int((sub["ci_low"] > 0).sum()),
            "ci_negative": int((sub["ci_high"] < 0).sum()),
            "bh_significant": int((sub["two_sided_q_bh_96"] < 0.05).sum()),
            "mean": float(sub["arm_minus_base"].mean() * 1000),
        })
    arm_summary = pd.DataFrame(summary_rows)
    x = np.arange(len(ARM_ORDER))
    ax_c.bar(x - 0.24, arm_summary["positive"], width=0.22, color="#A7C7B7", label="point positive")
    ax_c.bar(x, arm_summary["ci_positive"], width=0.22, color="#2E8B57", label="95% CI > 0")
    ax_c.bar(x + 0.24, arm_summary["ci_negative"], width=0.22, color="#B84A62", label="95% CI < 0")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], rotation=25, ha="right")
    ax_c.set_ylabel("Cells (of 24)")
    ax_c.set_ylim(0, 24.8)
    ax_c.set_title("C  Directional counts by DAPT arm", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, ncol=3, fontsize=7.4, loc="upper left")
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.grid(axis="y", alpha=0.18)
    for xi, row in zip(x, arm_summary.itertuples(index=False)):
        ax_c.text(xi, 23.2, f"mean {row.mean:+.2f}", ha="center", va="center", fontsize=7.1, color=ARM_COLOR[row.arm], fontweight="bold")

    # D: full distributions, preserving all 24 effects per arm.
    values = [scope.loc[scope["arm"] == arm, "arm_minus_base"].to_numpy(float) * 1000 for arm in ARM_ORDER]
    parts = ax_d.violinplot(values, positions=np.arange(1, 5), showmeans=False, showmedians=True, widths=0.75)
    for body, arm in zip(parts["bodies"], ARM_ORDER):
        body.set_facecolor(ARM_COLOR[arm]); body.set_edgecolor(ARM_COLOR[arm]); body.set_alpha(0.24)
    parts["cmedians"].set_color("#111827"); parts["cmedians"].set_linewidth(1.2)
    for xpos, (arm, vals) in enumerate(zip(ARM_ORDER, values), start=1):
        jitter = np.linspace(-0.16, 0.16, len(vals))
        ax_d.scatter(np.full(len(vals), xpos) + jitter, vals, s=15, color=ARM_COLOR[arm], alpha=0.75, edgecolor="none")
        ax_d.scatter(xpos, np.mean(vals), marker="D", s=45, color=ARM_COLOR[arm], edgecolor="white", linewidth=0.8, zorder=4)
    ax_d.axhline(0, color="#374151", linestyle="--", linewidth=0.8)
    ax_d.set_xticks(np.arange(1, 5))
    ax_d.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], rotation=25, ha="right")
    ax_d.set_ylabel("arm − Base ΔAUPRC × 1,000")
    ax_d.set_title("D  Technical effects are heterogeneous across task and readout", loc="left", fontweight="bold")
    ax_d.grid(axis="y", alpha=0.18)
    ax_d.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Technical gains are arm- and task-contingent rather than Tree-specific", fontsize=15.2, fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(out_base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)
    return arm_summary


def plot_leakage(leakage: pd.DataFrame, out_base: Path) -> pd.DataFrame:
    data = leakage.copy()
    data["exact_pair_pct"] = data["exact_pairs"] / data["pairs"] * 100
    data["near95_pair_pct"] = data["near_0_95_pairs"] / data["pairs"] * 100
    data["near90_pair_pct"] = data["near_0_90_pairs"] / data["pairs"] * 100
    data["opposite90_row_pct"] = data["opposite_label_identity_ge_0_90_rows"] / data["rows"] * 100
    data["below80_censored_row_pct"] = data["identity_search_censored_rows"] / data["rows"] * 100
    species_order = sorted(data["slug"].unique())
    data["row_order"] = data.apply(lambda r: species_order.index(r.slug) * 4 + TASK_ORDER.index(r.task), axis=1)
    data = data.sort_values("row_order").reset_index(drop=True)
    labels = [f"{x.slug.replace('_', ' ').title()} · {TASK_LABEL[x.task]}" for x in data.itertuples(index=False)]
    metrics = ["exact_pair_pct", "near95_pair_pct", "near90_pair_pct", "opposite90_row_pct", "below80_censored_row_pct"]
    metric_labels = ["Exact train hit\n(pair %)", "≥95% train hit\n(pair %)", "≥90% train hit\n(pair %)", "Opp-label ≥90%\n(row %)", "No ≥80% train hit\n(row %)"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(18.5, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.78, 0.78])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    matrix = data[metrics].to_numpy(float)
    shown = np.log10(matrix + 0.01)
    im = ax_a.imshow(shown, aspect="auto", cmap="magma", vmin=-2, vmax=2)
    ax_a.set_xticks(np.arange(len(metrics)))
    ax_a.set_xticklabels(metric_labels, rotation=28, ha="right")
    ax_a.set_yticks(np.arange(len(labels)))
    ax_a.set_yticklabels(labels, fontsize=7.1)
    for i in range(len(data)):
        for j in range(len(metrics)):
            value = matrix[i, j]
            ax_a.text(j, i, f"{value:.2g}", ha="center", va="center", fontsize=6.3, color="white" if shown[i, j] < -0.8 or shown[i, j] > 1.1 else "#111827")
    for boundary in np.arange(3.5, len(data), 4):
        ax_a.axhline(boundary, color="white", linewidth=0.8)
    ax_a.set_title("A  Leakage diagnostics for every evaluation species × task", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.038, pad=0.02)
    cbar.set_label("log10(percent + 0.01)")

    for task in TASK_ORDER:
        sub = data.loc[data["task"] == task]
        ax_b.scatter(sub["near90_pair_pct"], sub["below80_censored_row_pct"], s=55, color=TASK_COLOR[task], label=TASK_LABEL[task], edgecolor="white", linewidth=0.6)
    ax_b.set_xlabel("Pairs with ≥90% task-matched train identity (%)")
    ax_b.set_ylabel("Rows with no reported ≥80% train hit (%)")
    ax_b.set_title("B  Sparse near matches and below-80% censoring", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=7.4)
    ax_b.grid(alpha=0.18)
    ax_b.spines[["top", "right"]].set_visible(False)

    aggregate = data.groupby("slug", sort=False).agg(pairs=("pairs", "sum"), near90=("near_0_90_pairs", "sum"), near95=("near_0_95_pairs", "sum"), exact=("exact_pairs", "sum")).reset_index()
    aggregate["exact_pct"] = aggregate["exact"] / aggregate["pairs"] * 100
    aggregate["near95_nonexact_pct"] = (aggregate["near95"] - aggregate["exact"]) / aggregate["pairs"] * 100
    aggregate["near90_95_pct"] = (aggregate["near90"] - aggregate["near95"]) / aggregate["pairs"] * 100
    aggregate["below90_pct"] = 100 - aggregate["near90"] / aggregate["pairs"] * 100
    y = np.arange(len(aggregate))
    left = np.zeros(len(aggregate))
    for col, label, color in [
        ("exact_pct", "exact", "#A42032"),
        ("near95_nonexact_pct", "≥95%, non-exact", "#D98E04"),
        ("near90_95_pct", "90–95%", "#E9C46A"),
        ("below90_pct", "<90% / no ≥90% hit", "#D7E5E8"),
    ]:
        ax_c.barh(y, aggregate[col], left=left, color=color, label=label, height=0.7)
        left += aggregate[col].to_numpy(float)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([x.replace("_", " ").title() for x in aggregate["slug"]], fontsize=7.5)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Pair fraction (%)")
    ax_c.set_title("C  Aggregate identity categories", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.grid(axis="x", alpha=0.18)

    fig.suptitle("Sequence leakage audit: no exact evaluation-to-training matches", fontsize=15.0, fontweight="bold")
    fig.text(0.5, 0.003, "Near-identity search: MMseqs2 nucleotide, both strands, ≥80% query and target coverage, maximum 5 hits; 'no ≥80% hit' is left-censoring, not a measured zero identity.", ha="center", va="bottom", fontsize=7.4, color="#5B6470")
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(out_base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    metric = root / "results/metrics/publication_v4_technical_qc"
    metric.mkdir(parents=True, exist_ok=True)
    figure_dir = root / "results/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "arm_vs_base": root / "results/metrics/publication_v3_technical_arm_posthoc/arm_vs_base_scope_effects.tsv",
        "tree_contrasts": root / "results/metrics/plantcad_dapt_publication_v3_seed23_comparison/bootstrap_scope_effects.tsv",
        "leakage": root / "metadata/publication_v3_sequence_leakage_summary.tsv",
        "leakage_audit": root / "results/metrics/publication_v3_sequence_leakage_audit.json",
        "contract": root / "docs/publication_v4_posthoc_analysis_contract_20260803.md",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing technical/QC inputs: {missing}")
    scope96 = pd.read_csv(paths["arm_vs_base"], sep="\t")
    tree24 = pd.read_csv(paths["tree_contrasts"], sep="\t")
    leakage = pd.read_csv(paths["leakage"], sep="\t")
    leakage_audit = json.loads(paths["leakage_audit"].read_text(encoding="utf-8"))
    if len(scope96) != 96 or len(tree24) != 24 or len(leakage) != 28 or leakage_audit.get("status") != "pass":
        raise ValueError("Frozen technical or leakage dimensions changed")
    if scope96["decision_authority"].astype(bool).any():
        raise ValueError("Post-hoc arm-versus-Base contrasts unexpectedly have decision authority")
    technical_base = figure_dir / "publication_v4_technical_effects"
    leakage_base = figure_dir / "publication_v4_sequence_leakage_qc"
    arm_summary = plot_technical(scope96, tree24, technical_base)
    leakage_derived = plot_leakage(leakage, leakage_base)
    scope96.to_csv(metric / "technical_arm_vs_base_all96.tsv", sep="\t", index=False)
    tree24.to_csv(metric / "technical_tree_contrasts_all24.tsv", sep="\t", index=False)
    arm_summary.to_csv(metric / "technical_arm_summary.tsv", sep="\t", index=False)
    leakage_derived.to_csv(metric / "sequence_leakage_all28.tsv", sep="\t", index=False)
    outputs = [technical_base.with_suffix(f".{x}") for x in ["png", "pdf", "svg"]] + [leakage_base.with_suffix(f".{x}") for x in ["png", "pdf", "svg"]]
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "posthoc_seed23_descriptive_plus_frozen_leakage_qc",
        "scientific_decision_authority": False,
        "malus_accessed": False,
        "technical_cells": 24,
        "arm_vs_base_contrasts": len(scope96),
        "tree_vs_base_positive_cells": int((tree24["delta_vs_base"] > 0).sum()),
        "tree_vs_strongest_control_positive_cells": int((tree24["woody_control_gain"] > 0).sum()),
        "tree_vs_strongest_control_ci_positive": int((tree24["woody_control_gain_ci_low"] > 0).sum()),
        "tree_vs_strongest_control_ci_negative": int((tree24["woody_control_gain_ci_high"] < 0).sum()),
        "leakage_strata": len(leakage),
        "exact_training_match_rows": int(leakage["exact_rows"].sum()),
        "exact_training_match_pairs": int(leakage["exact_pairs"].sum()),
        "near90_pair_fraction": float(leakage["near_0_90_pairs"].sum() / leakage["pairs"].sum()),
        "near95_pair_fraction": float(leakage["near_0_95_pairs"].sum() / leakage["pairs"].sum()),
        "inputs": {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths.values()},
        "outputs": {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs},
    }
    (metric / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "technical_cells", "arm_vs_base_contrasts", "tree_vs_base_positive_cells", "tree_vs_strongest_control_positive_cells", "tree_vs_strongest_control_ci_positive", "tree_vs_strongest_control_ci_negative", "exact_training_match_pairs", "near90_pair_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
