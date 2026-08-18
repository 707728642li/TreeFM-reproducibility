# TreeFM reproducibility code

Private, reviewer-oriented repository for the computational methods used in the TreeFM study. The repository contains analysis code, frozen method contracts, configuration snapshots, environment records, validation tools, and tests.

## Repository boundary

Included:

- genome and corpus preparation code;
- domain-adaptive pretraining and embedding extraction code;
- functional and technical benchmark code;
- bootstrap and sensitivity-analysis code;
- cross-genus, chromatin, GO, candidate-gene, phylogenetic, microsynteny, and motif-analysis code;
- dated prospective and post-hoc method contracts;
- exact or historical conda environment records;
- repository-policy and syntax checks.

Intentionally excluded:

- manuscript, supplement, cover letter, reviewer reports, and submission files;
- figures or other publication artwork;
- raw sequencing data, public genomes, intermediate arrays, embeddings, checkpoints, and model weights;
- paper result tables and narrative result summaries;
- author identities, affiliations, funding records, credentials, tokens, and machine-specific absolute paths.

## Start here

1. Read [`docs/REVIEWER_QUICKSTART.md`](docs/REVIEWER_QUICKSTART.md).
2. Review the method modules in [`docs/METHODS_OVERVIEW.md`](docs/METHODS_OVERVIEW.md).
3. Check the analysis chronology in [`docs/ANALYSIS_TIMELINE.md`](docs/ANALYSIS_TIMELINE.md).
4. Recreate the required environment using [`environment/README.md`](environment/README.md).
5. Run the repository checks:

```bash
python tools/validate_repository.py
python -m pytest tests/test_repository_policy.py
```

## Layout

```text
configs/                    frozen analysis settings without private machine paths
docs/protocols/             dated analysis contracts and amendments
environment/                exact and human-readable conda records
scripts/                    frozen numbered analysis scripts; see the method index
tests/                      repository and selected analysis tests
tools/                      validation and manifest utilities
```

## Reproducibility model

Large public inputs are referenced by accession in the method documentation but are not redistributed. Each workflow expects a project root supplied by `TREEFM_ROOT` or by a command-line argument where supported. Public genome mirrors may be supplied through `PUBLIC_GENOME_ROOT`. Generated data and results must remain outside version control.

The dated contracts are preserved to distinguish prospective decisions from later sensitivity analyses. In particular, the 16 July plan, 17 July frozen analysis plan, and 1 August rebuilt-pilot amendment have different roles; see the chronology before interpreting stopping rules. The original flat script layout is retained because many frozen scripts resolve the project root relative to their own location.

## Status and licensing

This repository is private during peer review. No public reuse license is granted yet; see [`LICENSE.md`](LICENSE.md). A public license and citation metadata should be added only after all authors approve the release.
