# Computational methods overview

## 1. Input audit and corpus construction

Genome assemblies and annotations are audited before use. Candidate genomes are characterized using genome-scale composition features and BUSCO summaries. Domain-adaptive pretraining corpora are constructed under fixed sequence-budget rules so that the Tree, Herb, Random-plant, and phylogenetically matched plant arms can be compared against the same Base model.

Relevant code: scripts 60, 63–64, 100, 103–105, and 107–108.

## 2. Domain-adaptive pretraining

PlantCaduceus-derived checkpoints are continued on frozen DAPT corpora. Training code records the seed, optimizer state, corpus identity, checkpoint manifest, and runtime metadata. Embeddings are extracted in evaluation mode with reverse-complement parameter sharing and fixed pooling.

Relevant code: scripts 109 and 116–118.

## 3. Technical and functional transfer

Technical probes evaluate sequence-level transfer under fixed train/test policies. Functional transfer is evaluated using cross-genus dormancy-associated labels and promoter sequences. The primary functional design uses reciprocal Prunus/Pyrus evaluation, linear and tree-based probes, and model-arm contrasts defined before metric inspection in the rebuilt pilot.

Relevant code: scripts 127–133.

## 4. Estimation and uncertainty

Primary summaries keep model arm, training genus, test genus, probe family, and readout explicit. Paired bootstrap routines estimate point contrasts and uncertainty. Secondary endpoints and prediction-agreement analyses are kept separate from the primary continuation decision.

Relevant code: scripts 269, 280, 291–302, 310, 315, 322, and 381.

## 5. Biological interpretation

Cross-genus promoter k-mers, GO enrichment, candidate-gene catalogs, H3K4me3 overlap, chromosome-level robustness, model attribution, and Tier-A annotations are evaluated in separate modules. These analyses distinguish association, motif compatibility, chromatin overlap, and model attribution from direct transcription-factor binding.

Relevant code: selected scripts 155–256.

## 6. Comparative and sensitivity extensions

Post-hoc extensions evaluate corpus phylogeny, representation structure, GO stability, comparative candidate context, gene trees, microsynteny, motif enrichment, domain sensitivity, G-box overlap sensitivity, and count-matched sensitivity. They provide robustness and biological context but cannot alter the earlier prospective stopping decision.

Relevant code: selected scripts 327–367.

## 7. Compute separation

CPU-heavy genome processing and bioinformatics are intended for a high-memory Linux server. GPU pretraining and embedding extraction are intended for a local CUDA workstation. Paths and device selection must be supplied at runtime; no private hostnames or absolute paths are committed.
