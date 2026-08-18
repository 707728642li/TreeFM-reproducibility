# Reviewer quick start

This repository is a code-and-methods companion. It deliberately contains no manuscript, publication figure, or result narrative.

## Five-minute audit

```bash
python tools/validate_repository.py
python -m pytest tests/test_repository_policy.py
```

The validator checks Python syntax, forbidden publication artifacts, file size, machine-specific paths, and common secret patterns.

## Method audit order

1. `docs/ANALYSIS_TIMELINE.md` — separates prospective plans from later amendments and sensitivities.
2. `docs/protocols/publication_v3_statistical_analysis_plan.md` — early estimands and planned multi-seed logic.
3. `docs/protocols/publication_v3_statistical_analysis_plan_v1.md` — frozen ROPE and analysis definitions.
4. `docs/protocols/publication_v3_rebuild_pilot_amendment_20260801.md` — rebuilt seed-23 pilot and continuation decision.
5. `docs/protocols/publication_v4_posthoc_analysis_contract_20260803.md` and the publication-v5 contracts — explicitly post-hoc extensions.

## Code audit order

- Corpus construction: scripts 60, 63–64, 100, 103–105, and 107–108.
- DAPT and technical evaluation: scripts 109 and 116–118.
- Functional transfer: scripts 127–133.
- Bootstrap and decision summaries: scripts 269, 280, 291–302, 310, 315, 322, and 381.
- Biological interpretation: scripts 155–256 selected in `scripts/`.
- Comparative and sensitivity analyses: scripts 327–367 selected in `scripts/`.

## Running analyses

The full workflows require public genomes, public sequencing accessions, upstream model snapshots, and large intermediate embeddings that are not stored in Git. Configure:

```bash
export TREEFM_ROOT=/path/to/TreeFM-workdir
export PUBLIC_GENOME_ROOT=/path/to/public-genome-cache
```

Use the exact conda records in `environment/` for the relevant module. Scripts should be run from the repository root unless their `--help` output states otherwise.

## Interpretation guardrails

- The rebuilt pilot continuation rule originates from the dated 1 August amendment, not the 16 July plan.
- Failure to exceed `+0.02` is not, by itself, proof that an effect lies inside `[-0.02, +0.02]`.
- Species-disjoint analyses are post-hoc sensitivities and do not retroactively determine the prospective stopping decision.
- Equal-budget controls test whether an apparent gain is unique to the Tree corpus; they do not erase the separate Tree-versus-Base point contrast.
