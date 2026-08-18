# Publication-v5 copy-level cross-genus association addendum

**Frozen: 3 August 2026 UTC, before any final 14-family summary, pooled statistic or candidate-neighbor result was inspected**

## Rationale

The original gene-tree plan reported the nearest other-genus leaf for each frozen Tier-A candidate. A raw nearest-candidate fraction is not sufficient evidence of copy-level conservation because the probability depends on family size and the number of candidates in each genus. This addendum replaces that purely descriptive proportion with a fixed label-shuffling null while retaining the descriptive mapping.

## Locked inputs and statistic

- Primary full-length IQ-TREE topology for each of the 14 frozen Tier-A orthogroups; no family can be excluded.
- Frozen Tier-A candidate labels and the complete *Prunus persica* and *Pyrus pyrifolia* family leaves.
- Directed nearest-neighbor mapping from every *Prunus* leaf to the nearest *Pyrus* patristic leaf and vice versa, with lexical resolution of exact ties.
- Observed statistic: fraction of Tier-A candidate queries whose nearest other-genus leaf is also a Tier-A candidate.

## Null model and multiplicity

- Independently shuffle candidate labels among available leaves within each family and genus, preserving the exact candidate count in every family-by-genus cell.
- Recompute both the candidate-query set and directed successes in 10,000 replicates with seed 20260803.
- Use the one-sided empirical tail probability `(1 + null >= observed) / 10,001`.
- Apply Benjamini-Hochberg correction across the 14 family tests.
- Report one prespecified pooled statistic by summing successes and candidate queries over families within each replicate. The pooled test is not selected in response to family-level results.

## Interpretation boundary

The test asks whether frozen candidate copies show more cross-genus phylogenetic association than expected from their family- and genus-specific counts. A positive result does not prove one-to-one orthology, conserved biochemical function, transcriptional regulation or a causal dormancy role. A negative result does not invalidate family-level Tier-A evidence. The analysis is post-hoc descriptive, has no continuation authority, does not alter candidate membership or the seed-23 stopping decision, and does not access sealed Malus downstream outcomes.
