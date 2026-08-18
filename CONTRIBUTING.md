# Contributing

## Scope

Contributions must be limited to computational methods, analysis code, configuration, environments, tests, and technical documentation.

Do not commit manuscripts, supplements, cover letters, peer-review reports, publication figures, result narratives, raw or controlled data, model weights, embeddings, author metadata, credentials, or local absolute paths.

## Change process

1. Create a focused branch.
2. State whether the change affects a prospective analysis, a post-hoc sensitivity analysis, or engineering only.
3. Preserve frozen contracts. Add a dated amendment rather than silently rewriting a historical protocol.
4. Add or update a test for material code changes.
5. Run `python tools/validate_repository.py` and `python -m pytest tests/test_repository_policy.py`.
6. Record any change to thresholds, estimands, data exclusions, seeds, or comparison arms explicitly in the pull request.

## Scientific safeguards

- Never present post-hoc analyses as prospective.
- Never infer equivalence from failure to exceed a positive threshold.
- Keep equal-budget controls distinct from an untrained or base-model comparison.
- Preserve the distinction between point estimates, confidence intervals, and bootstrap replicate proportions.
- Keep model attribution, motif compatibility, chromatin overlap, and direct regulatory binding as separate evidential claims.
