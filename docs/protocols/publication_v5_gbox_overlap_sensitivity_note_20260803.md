# Publication-v5 exact-G-box overlap sensitivity note

Date: 2026-08-03  
Tier: explicitly post-hoc circularity diagnostic; no decision authority

The primary JASPAR analysis returned convergent enrichment only for bZIP/bHLH/BES-BZR TF-family categories. Because frozen Tier-A membership included exact `CACGTG` presence in both genera, this result is mechanistically coherent but not independent of candidate construction.

To distinguish direct recovery of the selection motif from broader promoter architecture, remove every FIMO hit whose genomic interval overlaps an exact `CACGTG` occurrence in the scanned promoter. Recompute promoter-level TF-family presence with the same 62-family JASPAR-derived universe, the same GC/chromosome-matched background pools, 10,000 deterministic replicates and seed 20260803. No family, threshold or replicate may be changed after inspecting the masked result.

Interpretation is constrained as follows:

- Loss of enrichment indicates that the primary result is mainly an expected internal validation of the exact-G-box Tier-A rule.
- Retained enrichment indicates compatible motif architecture beyond the exact selection sites, but still does not establish binding or causal regulation.
- This sensitivity cannot change the seed-23 stopping decision or authorize any sealed downstream analysis.
