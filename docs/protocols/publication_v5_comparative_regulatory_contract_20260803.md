# Publication v5 comparative phylogenomics and regulatory analysis contract

Frozen: 2026-08-03, before publication-v5 result generation  
Tier: additive post-hoc descriptive analysis  
Decision authority: none

## Immutable boundaries

- The seed-23 stopping decision remains unchanged.
- Seeds 41 and 59 remain unauthorized.
- Malus downstream labels, outcomes, predictions and embeddings remain sealed.
- The 14 Tier-A orthogroups, their order and all v4 evidence annotations are frozen.
- New results may strengthen, qualify or falsify candidate evolutionary/regulatory interpretations, but cannot rescue the failed Tree-specific functional continuation condition.

## Resource gate

The authoritative gate is `results/metrics/publication_v5_resource_audit.json`.

- Required: 14/14 orthogroups, 26 species, all 2,836 Tier-A member proteins, unique GFF locations for all 2,836 members, and complete protein/GFF/promoter mapping for all 79 *Prunus*/*Pyrus* Tier-A positive genes.
- Observed before analysis: all requirements passed (9/9 audit checks).
- Fixed input fingerprints are recorded in the resource-audit summary.

## A. Gene-family phylogenomics

### Scope

- Analyze all 14 Tier-A orthogroups; no family may be removed because its topology is inconvenient.
- Extract all primary protein sequences belonging to each frozen orthogroup from the same 26-species OrthoFinder namespace.
- Sequence identifiers are sanitized only for software compatibility; a reversible identifier map is mandatory.

### Primary workflow

1. MAFFT `--auto` protein alignment.
2. ClipKIT `smart-gap` trimming.
3. IQ-TREE 3.1.2 ModelFinder (`-m MFP`) with 1,000 ultrafast bootstrap and 1,000 SH-aLRT replicates, fixed random seed 20260803 and maximum eight threads per family. The executable version was recorded from the existing project environment before tree inference.
4. If Treerecs installation and smoke-test gates pass, contract branches with ultrafast bootstrap below 80 and root/reconcile against the frozen rooted 26-species tree under duplication/loss parsimony. If the gate fails, omit reconciliation-derived duplication/loss claims rather than substituting an unregistered method.

### Sensitivity

- For the three display families, infer an untrimmed-alignment IQ-TREE under the primary selected model and report normalized Robinson–Foulds distance to the primary tree.
- Composition-QC amendment, frozen after software startup diagnostics but before any final tree: if more than 20% of sequences in a display family fail IQ-TREE's amino-acid composition chi-square test, scan all family proteins with its already-frozen consensus Pfam model (`PF00847` for `OG0000025`, `PF00481` for `OG0000413`, `PF03055` for `OG0000277`), extract the best supported domain per protein, and infer a domain-only MAFFT/ClipKIT/IQ-TREE sensitivity tree. The full-length tree remains primary; biological interpretation requires the relevant candidate relationship not to contradict the domain tree.

### Frozen copy-level cross-genus association test

- This addition was frozen before any final primary tree or candidate-neighbor result was inspected.
- Within each primary full-length family tree, map every *Prunus* leaf to its nearest *Pyrus* patristic neighbor and every *Pyrus* leaf to its nearest *Prunus* neighbor, resolving exact ties lexically.
- The observed statistic is the directed fraction of frozen Tier-A candidate leaves whose nearest other-genus neighbor is also a frozen Tier-A candidate.
- Generate 10,000 null replicates with seed 20260803 by independently shuffling candidate labels among the available leaves of each genus while preserving the exact family-by-genus candidate counts. Recompute the candidate-query set and directed success fraction in every replicate.
- Report one-sided empirical *P* values, null medians and null 95th percentiles for all 14 families, with Benjamini-Hochberg correction across families. Also report one prespecified pooled statistic formed by summing directed successes and queries across all families in each replicate.
- This tests copy-level cross-genus phylogenetic association beyond family membership. It does not establish one-to-one orthology, conserved biochemical function or causal dormancy roles, and it cannot alter Tier-A membership or any model decision.
- Report alignment length, retained-site fraction, missing-character fraction, selected model, leaf count, species count and support distribution for every family.

### Display-family rule

Select exactly three families by descending frozen literature-evidence level, breaking ties by ascending frozen catalog rank. This yields, before tree inspection:

1. `OG0000025` — AP2/ERF-DREB, literature level 4.
2. `OG0000413` — clade-A PP2C, literature level 3.
3. `OG0000277` — NCED/CCD, literature level 3.

All 14 results remain in source data and a complete supplementary matrix.

## B. Orthogroup-anchored microsynteny

### Gene-order representation

- Parse protein-coding gene order from the 26 frozen GFF3 annotations, preferring `gene` records and using unique `mRNA/transcript` records only where a source lacks gene features.
- Map genes into the same frozen OrthoFinder namespace.
- For every Tier-A member, define a local window of the ten annotated genes upstream and ten downstream on the same sequence. Truncated scaffold-edge windows are retained and flagged.

### Scores

- Primary score: Jaccard similarity of non-target neighboring orthogroup sets.
- Order-aware sensitivity: normalized longest-common-subsequence score after orienting windows to the target-gene strand.
- For families with multiple copies, the family/species-pair value is the maximum bipartite one-to-one matching mean after padding the smaller copy set with zero-similarity dummy genes; division by the larger copy count penalizes unmatched duplicates. It is not an unrestricted best pair.
- The principal comparison is *Prunus*–*Pyrus*. *Fragaria* and the remaining species provide descriptive depth profiles.

### Matched background

- Exclude all 14 Tier-A families and the frozen 111 shared positive-only candidate families from control eligibility.
- Require both *Prunus* and *Pyrus* membership.
- Match each Tier-A family on species breadth (±1), exact target-genus copy counts when feasible, and log2 total family size within 0.25; relax in that order only if fewer than 50 controls exist, recording every relaxation.
- Retain up to 100 closest controls by standardized distance with SHA-256 lexical tie-breaking. If fewer than 50 controls survive the recorded caliper relaxations, use the 100 globally nearest eligible families that contain both target genera; this final fallback is labeled `global_nearest` and cannot be described as exact matching.
- Empirical one-sided *P*=(1 + controls with score ≥ observed)/(1 + number of controls); BH correction across 14 families.

### Interpretation

Call the result “orthogroup-anchored microsynteny” or “local gene-order conservation.” Do not call a region a conserved regulatory block or infer function from neighborhood alone.

## C. Promoter TF-family enrichment and position

### Motif universe

- JASPAR 2026 CORE plants, nonredundant MEME-format release, with the downloaded file hash recorded.
- Scan both strands with FIMO using motif *P*≤1e-5.
- Collapse individual profiles to the JASPAR metadata TF-family level before primary testing.

### Foreground and background

- Foreground: all 79 frozen Tier-A positive promoters across *Prunus* and *Pyrus*.
- Background pool: positive labeled genes of the same genus excluding all Tier-A genes.
- Every resampled background set preserves genus and promoter length and matches chromosome where feasible; GC difference is minimized within a 0.02 caliper. Unmatched genes and relaxations are reported.
- Generate 10,000 deterministic matched-background replicates with seed 20260803.

### Primary statistic

- For each TF family, compute the difference in promoter-level motif presence between Tier-A and matched background separately by genus.
- The cross-genus statistic is the equally weighted mean of the two genus differences.
- A family is called convergently enriched only if both genus differences are positive and the one-sided empirical cross-genus *P* passes BH FDR <0.05 across all tested TF families.
- Positional density relative to the TSS is descriptive and plotted only for convergently enriched families plus the frozen mechanistic families AP2/ERF, WRKY, bZIP and bHLH when represented in JASPAR metadata.

### Interpretation

Motif hits indicate sequence compatibility, not binding. Shared family enrichment does not establish conservation of an individual binding site. No causal or validated-regulatory language is permitted without experimental binding/accessibility data.

## Output and audit requirements

- Every analysis emits compact TSV/JSON source data, SHA-256 fingerprints, software versions and a standalone audit JSON.
- Every final figure is emitted as PNG, PDF and SVG and independently rendered for manual QA.
- New figures enter the manuscript only if their audit passes; null results are retained and described.
- The integrated v5 audit must verify the v4 boundary statements, all v5 component audits, manuscript consistency and bundle fingerprints.
