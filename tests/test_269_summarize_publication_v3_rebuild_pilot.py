#!/usr/bin/env python3
"""End-to-end smoke test for the prospective seed-23 pilot decision."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ARMS = ("base", "tree", "herb", "random_plant", "phylogc_match")
SLUGS = ("hevea_brasiliensis", "prunus_persica", "pyrus_pyrifolia")
TASKS = ("tis", "tts", "splice_donor", "splice_acceptor")
READOUTS = ("linear", "xgboost")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    temp_parent = project / "results/tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="summary_269_", dir=temp_parent))
    try:
        technical_root = root / "results/metrics/plantcad_dapt_publication_v3_probes"
        functional_root = root / "results/metrics/plantcad_dapt_publication_v3_functional_probes"
        control_scores = {
            "base": 0.50,
            "herb": 0.51,
            "random_plant": 0.52,
            "phylogc_match": 0.53,
        }
        task_gains = {
            "tis": 0.03,
            "tts": 0.02,
            "splice_donor": 0.01,
            "splice_acceptor": -0.005,
        }
        for arm in ARMS:
            technical_rows = []
            for slug in SLUGS:
                for task in TASKS:
                    for readout in READOUTS:
                        auprc = (
                            control_scores["phylogc_match"] + task_gains[task]
                            if arm == "tree"
                            else control_scores[arm]
                        )
                        technical_rows.append(
                            {
                                "arm": arm,
                                "seed": 23,
                                "scope": "species",
                                "slug": slug,
                                "task": task,
                                "readout": readout,
                                "family_transfer_class": "logo_novel_family",
                                "identity_population": "all",
                                "auprc": auprc,
                            }
                        )
            path = technical_root / arm / "seed_23"
            path.mkdir(parents=True)
            pd.DataFrame(technical_rows).to_csv(path / "metrics.tsv", sep="\t", index=False)

            functional_rows = []
            for training, heldout in (("pyrus", "prunus"), ("prunus", "pyrus")):
                for readout in READOUTS:
                    if arm == "tree":
                        auprc = 0.57 if heldout == "prunus" else 0.54
                    else:
                        auprc = control_scores[arm]
                    functional_rows.append(
                        {
                            "arm": arm,
                            "seed": 23,
                            "training_genus": training,
                            "heldout_genus": heldout,
                            "readout": readout,
                            "population": "all",
                            "auprc": auprc,
                        }
                    )
            path = functional_root / arm / "seed_23"
            path.mkdir(parents=True)
            pd.DataFrame(functional_rows).to_csv(path / "metrics.tsv", sep="\t", index=False)

        subprocess.run(
            [
                sys.executable,
                str(project / "scripts/269_summarize_publication_v3_rebuild_pilot.py"),
                "--project-root",
                str(root),
            ],
            check=True,
        )
        output = root / "results/metrics/publication_v3_rebuild_pilot_summary"
        decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
        observed = decision["observed"]
        if decision["decision"] != "continue_full_multiseed":
            raise AssertionError(decision)
        if observed["positive_technical_tasks"] != 3:
            raise AssertionError(observed)
        if observed["technical_material_reversals"] != 0:
            raise AssertionError(observed)
        if observed["functional_primary_cells_ge_0_02"] != 2:
            raise AssertionError(observed)
        technical = pd.read_csv(output / "technical_primary_cell_effects.tsv", sep="\t")
        functional = pd.read_csv(output / "functional_primary_cell_effects.tsv", sep="\t")
        if len(technical) != 24 or len(functional) != 4:
            raise AssertionError((len(technical), len(functional)))
        print(json.dumps({"status": "pass", "decision": decision["decision"]}))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
