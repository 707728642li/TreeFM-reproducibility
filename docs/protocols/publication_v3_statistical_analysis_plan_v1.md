# TreeFM publication-v3 statistical analysis plan v1.0

Frozen UTC: 2026-07-17T04:45:28.846936+00:00

The primary technical endpoint is NovelFamily AUPRC in Hevea, Prunus and Pyrus, evaluated with both a frozen linear probe and a frozen XGBoost probe across DAPT seeds 23, 41 and 59.

The primary functional endpoint is leave-one-genus-out robust dormancy AUPRC across Prunus and Pyrus; Malus remains a sealed one-time external validation.

- Tree effect: Tree minus RandomPlant.
- Woody-control gain: Tree minus the best of Herb, RandomPlant and PhyloGCMatch.
- Technical AUPRC equivalence ROPE: [-0.01, +0.01].
- Functional AUPRC equivalence ROPE: [-0.02, +0.02].
- Primary pair-block bootstrap: 2,000 replicates.
- Secondary families use FDR q=0.05.
- Minimum primary NovelFamily pairs per genus-task: 636.

Pilot-free MDE uses prevalence 0.5 and paired-arm correlation 0.5; per-scope values are frozen in `metadata/publication_v3_technical_mde.tsv`.

Malus remains sealed under `config/publication_v3_malus_blind_lock.json`.
