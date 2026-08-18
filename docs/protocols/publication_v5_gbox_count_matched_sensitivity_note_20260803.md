# Publication-v5 exact-G-box-count-matched sensitivity note

**Frozen before inspecting the sensitivity result: 3 August 2026**

## Status and purpose

This is an explicitly post-hoc circularity diagnostic. It has no authority to alter Tier-A membership, the seed-23 stopping decision, authorization of seeds 41/59, or the sealed Malus branch.

The primary motif analysis matches Tier-A promoters to same-genus positive-gene promoters by length, GC and chromosome where possible. The first circularity sensitivity removes FIMO hits overlapping exact `CACGTG` sites. A residual concern remains because foreground and background promoters may differ in whether, or how many times, they contain exact `CACGTG`. This sensitivity addresses that concern by additionally matching the exact-`CACGTG` count.

## Frozen implementation

1. Use the same 79 foreground promoters, 2,125 background promoters, 927 JASPAR 2026 plant profiles collapsed to the same 62 TF-family categories, and the same FIMO calls as the primary analysis.
2. Remove every FIMO hit whose interval overlaps an exact `CACGTG` interval, exactly as in the existing overlap-masked sensitivity.
3. For each foreground promoter, restrict eligible controls to the same genus, identical promoter length and identical exact-`CACGTG` count.
4. Within that set, prefer same-chromosome controls with absolute GC difference at most 0.02 when at least 10 exist; otherwise use same-genus controls at the same GC tolerance when at least 10 exist; otherwise use the nearest 100 GC-matched controls within the exact-count stratum. Require at least 10 controls per foreground promoter.
5. Generate 10,000 deterministic matched-background replicates with seed 20260803, avoiding within-replicate control reuse whenever the available pools permit.
6. Use the same promoter-presence statistic, one-sided empirical test, 62-family Benjamini–Hochberg correction and convergence rule: BH `q < 0.05` plus positive foreground-minus-control effects in both genera.

## Interpretation boundary

Families significant here have evidence of promoter-sequence enrichment beyond both direct exact-G-box-overlapping motif calls and differences in exact-G-box count. This remains motif compatibility, not TF occupancy, regulatory direction or causality. Loss of significance is interpreted as sensitivity to selection-linked promoter composition, not evidence that a TF family is biologically irrelevant.
