# Publication v5 frontier scope: phylogenomic and cis-regulatory strengthening

Date frozen: 2026-08-03  
Purpose: identify additions that materially strengthen the TreeFM manuscript without reopening the stopped downstream experiment.

## Frontier signals

1. **Orthology inference is moving toward explicit gene-tree/species-tree reconciliation.** The 2026 OrthoFinder update emphasizes resolved gene trees, duplication/loss/coalescence-aware orthology, and scalable gene-family histories rather than treating an orthogroup as a flat presence/absence bin. The OrthoFinder result guide likewise distinguishes rooted, reconciled gene trees and mapped duplication events.  
   Sources: https://www.nature.com/articles/s41592-026-03126-6 ; https://orthofinder.github.io/OrthoFinder/tutorials/guide-to-results/

2. **Plant comparative genomics increasingly couples sequence phylogeny with synteny.** Recent Oryza work combined orthology and gene coordinates to track polyploidization, rearrangement and translocation, and used GENESPACE/synteny-derived homologs in phylogenomics. Synteny-network methods formally treat conserved local gene order as an evolutionary character rather than as a decorative genome browser view.  
   Sources: https://www.nature.com/articles/s41588-025-02183-5 ; https://academic.oup.com/bioinformatics/article/39/1/btac806/6947985 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8190143/

3. **Deep-time plant regulatory comparison is now synteny anchored.** Conservatory, reported in *Science* in 2026, uses homolog groups as anchors and local synteny-guided alignment windows to detect conserved noncoding sequences across rapid plant genome turnover. This argues against interpreting raw motif occurrence alone as functional conservation.  
   Sources: https://doi.org/10.1126/science.adt8983 ; https://conservatorycns.com/dist/pages/conservatory/about.php

4. **Current motif resources support family-level, curated analysis.** JASPAR 2026 contains 927 nonredundant plant CORE profiles and 55 plant familial profiles, with updated quality review. The appropriate comparison is therefore curated PWM scanning with explicit background matching and family-level multiplicity control, not a hand-picked collection of consensus words.  
   Sources: https://academic.oup.com/nar/article/54/D1/D184/8343514 ; https://jaspar2026.elixir.no/downloads/

5. **Tree reconciliation uncertainty must remain visible.** Treerecs can root unrooted gene trees, contract weak branches and reconcile them against a rooted species tree under duplication/loss parsimony. This is appropriate for targeted descriptive family histories, but duplication counts remain model- and support-threshold-dependent.  
   Source: https://academic.oup.com/bioinformatics/article/36/18/4822/5872524

## Consequences for TreeFM

The v4 manuscript already establishes corpus balance, robust cross-genus GO biology, candidate conservation, functional non-superiority to matched DAPT controls, representation proximity and leakage/technical boundaries. The v5 additions should answer three residual reviewer questions:

1. Are Tier-A families merely broad orthogroups, or do sequence-resolved family histories support interpretable Rosaceae subclades and duplication patterns?
2. Do target-genus candidate copies retain local gene-order context beyond what is expected for copy-number- and breadth-matched orthogroups?
3. Do Tier-A promoters show cross-genus enrichment and positional organization of curated plant TF-binding families beyond a positive-gene, GC-matched background?

## Methods selected

- Infer all 14 fixed Tier-A protein-family trees with MAFFT, ClipKIT and IQ-TREE 3 ModelFinder plus 1,000 ultrafast bootstrap and 1,000 SH-aLRT replicates.
- Reconcile supported trees to the frozen rooted 26-species tree using Treerecs if the reproducible installation gate passes; otherwise report rooted gene trees without inventing duplication/loss counts.
- Build orthogroup-anchored local gene-order windows directly from the 26 GFF3 files and the same frozen OrthoFinder namespace; compare all Tier-A families with deterministic occupancy/copy-number-matched controls.
- Scan the complete JASPAR 2026 CORE plant PWM collection, collapse results to TF families, and test Tier-A positive promoters against positive, same-genus, GC/length/chromosome-matched backgrounds.
- Reserve a synteny-aware conserved-noncoding-sequence claim for a future analysis with alignable local genomic windows. Motif presence alone will be called regulatory compatibility, not conserved binding or function.

## Methods rejected

- Hand-picked gene trees chosen after inspecting topology.
- Uncorrected duplication counts from copy-number tables alone.
- Whole-genome ribbon plots without a candidate-centered statistical question.
- Motif logos or raw motif counts without matched background and multiplicity correction.
- Claims of conserved cis-regulation based only on two promoters sharing one short motif.
- Any reopening of seeds 41/59 or any Malus downstream access.
