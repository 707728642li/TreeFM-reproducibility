# Environments

The repository includes exact conda exports captured from the original analysis environments and a portable core specification.

## Portable core environment

```bash
conda env create -f environment/core-environment.yml
conda activate treefm-repro
```

This environment supports repository validation and the Python/R analysis layer. External bioinformatics programs used by specific workflows should be installed in separate project-prefix environments to avoid dependency conflicts.

## Exact historical records

- `treefm-genome.*`: genome and annotation processing.
- `treefm_chip_conda_explicit.txt`: chromatin workflows.
- `treefm_match_conda_explicit.txt`: genome matching and technical features.
- `publication-v5-motif.*`: FIMO/motif workflows.
- `publication-v5-reconcile.*`: tree reconciliation and domain-sensitivity workflows.

Files ending in `.conda-explicit.txt` are platform-specific exact exports. Files ending in `.conda-history.txt` are shorter human-readable histories. Prefer the exact export for archival reconstruction and the portable YAML for code inspection or lightweight tests.
