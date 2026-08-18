# Publication-v4 post-hoc analysis and figure contract

Frozen on: 2026-08-03  
Status: additive, explicitly post-hoc and descriptive  
Authority: no continuation, model-selection, candidate-selection or primary-claim authority

## 1. Immutable boundaries

1. The publication-v3 decision remains `stop_and_reassess` because the necessary functional continuation condition failed at seed 23.
2. Seeds 41/59 are not authorized. No new DAPT training is permitted under this contract.
3. Malus held-out outcomes, labels, embeddings and downstream predictions remain sealed and are not accessed directly or indirectly. The already frozen, pre-outcome Malus taxonomy-distance covariate used when constructing PhyloGCMatch may be reproduced from its fingerprinted summary table, but no new Malus data are read.
4. Base, Tree, Herb, RandomPlant and PhyloGCMatch checkpoints, embeddings, predictions, labels, primary populations and bootstrap outputs are unchanged.
5. All new analyses are reported as post-hoc description. They may explain effect structure but cannot rescue or replace the failed functional endpoint.

## 2. Literature-motivated questions

The additive analyses answer four questions that are not resolved by the frozen score tables:

1. Were the four DAPT corpora compositionally and phylogenetically distinct in the intended way?
2. Did Tree DAPT create a representation shift distinguishable from other equal-budget plant DAPT arms?
3. How are the frozen cross-genus biological terms and Tier-A families organized across ontology, species and evidence modules?
4. Can the Base-only versus matched-control conclusion reversal be communicated without selecting favorable tasks or cells?

## 3. Corpus and phylogeny analysis

### 3.1 Corpus membership

- Include every species and every written 512-bp window in the authoritative four-arm tables `metadata/publication_v3_dapt_corpus_shards.tsv`, `metadata/publication_v3_dapt_corpus_audit.tsv` and `data/processed/publication_v3_dapt_hf/manifest.tsv`. The older three-arm `metadata/dapt_corpora_summary.tsv` is provenance only and is not the publication-v4 source table.
- Summaries include species count, written windows, bases, per-species mean GC, N fraction, rejection counts and life form.
- The fixed four DAPT arms are Tree, Herb, RandomPlant and PhyloGCMatch. Base has no added corpus and is shown only as the unchanged model reference.
- No species or feature is removed for visual convenience.

### 3.2 Phylogeny/GC matching

- Use the complete candidate table in `metadata/publication_v3_phylogc_genome_features.tsv`, the selected set in `config/publication_v3_phylogc_match_selected.tsv`, and the frozen feature table in `metadata/publication_v3_phylogc_selected_feature_match.tsv`.
- Display all nine frozen matching features and all eligible candidate species.
- Standardized mean differences use the already frozen standardization scales. No re-optimization or reselection is allowed.

### 3.3 Technical-panel species tree

- Infer one descriptive maximum-likelihood-like tree with FastTree 2 under the LG amino-acid model from the complete 26-species OrthoFinder concatenated alignment:
  `data/processed/orthofinder_benchmark_publication_v3/OrthoFinder/Results_PublicationV3/WorkingDirectory/Alignments_ids/SpeciesTreeAlignment.fa`.
- Map numeric species IDs with the unchanged OrthoFinder `SpeciesIDs.txt` from `Results_PublicationV3Seed/WorkingDirectory`.
- FastTree local support values are displayed or supplied in source data; they are not interpreted as divergence-time estimates.
- Root for display on the magnoliid `Magnolia biondii`, the earliest-diverging taxon in this 26-species panel. The root choice is presentational and does not enter any statistical analysis.

## 4. Representation analysis

### 4.1 Eligible embeddings

- Include all five arms and all 26 technical-panel species from the fingerprinted manifests under `results/embeddings/plantcad_dapt_publication_v3/`.
- Before analysis, require exact agreement across arms for species order, row count, 384 dimensions and `row_hash`.
- Use the original benchmark row index; never match rows by approximate sequence or gene name.

### 4.2 Deterministic common sample

- Within every available `species × task × label` stratum, rank rows by SHA-256 of `publication-v4|slug|task|label|pair_id|benchmark_row_index`.
- Retain the first 128 rows per stratum, or every row when a stratum contains fewer than 128.
- The same row indices are then used for all five arms.
- This produces at most 26,624 rows and balances the four boundary tasks and positive/negative labels without consulting model output.
- Record the complete selected-row table, stratum counts and SHA-256 fingerprint.

### 4.3 Quantitative representation endpoints

Primary descriptive representation endpoints are:

1. pairwise **linear centered-kernel alignment (CKA)** among the five arms on the complete deterministic sample;
2. the same pairwise CKA within every species and within every task;
3. per-row direct cosine distance and relative L2 displacement of each DAPT arm from Base, summarized by species and task;
4. orthogonal-Procrustes residual distance between each DAPT arm and Base on the common sample;
5. the identity of the DAPT arm closest to Tree by CKA and by median paired cosine distance, reported globally and across all species/tasks.

Uncertainty for median paired displacement is obtained from 2,000 deterministic species-stratified bootstrap replicates. These intervals quantify variation among sampled sequence rows and species strata within seed 23; they do not quantify adaptation-seed uncertainty.

### 4.4 Projection

- Fit PCA in float32 to the concatenated arm embeddings after a common global feature centering; show an equal deterministic subset of at most 2,000 rows per arm.
- A UMAP panel is permitted only as a supplementary illustration, using 50 PCA components, cosine metric, `n_neighbors=50`, `min_dist=0.15`, and fixed seed 20260803.
- UMAP robustness seeds 1, 23 and 101 are generated in the supplement. No claim is based on apparent two-dimensional cluster separation.

### 4.5 Prediction agreement

- Reuse every row of all four primary functional prediction files for all five arms.
- Report pairwise Spearman correlation and mean absolute probability difference for every primary cell.
- RandomPlant or another arm may be described as closest to Tree only if the same identity is supported by both measures; otherwise the measures are reported separately.

## 5. GO organization and stability

- The tested universe remains the frozen 808-term common family; the primary count remains 52.
- The main forest plot displays all 18 fixed nonredundant leaf terms.
- The stability matrix includes all 52 primary terms and every eligible leave-one-chromosome-out refit in both genera. Missing ineligible combinations are marked, never imputed.
- An ontology network is descriptive. Nodes are the 18 fixed terms. Edges are derived only from GO ontology ancestry/relationships and not from observed effect sizes.
- Node size may encode the smaller genus-specific gene count; color encodes a fixed broad biological module. Effects and FDR are not used to choose nodes because all 18 are shown.

## 6. Tier-A comparative genomics and hypothesis network

- Retain all 14 Tier-A orthogroups in frozen catalog order.
- Copy-number analysis uses the complete 26-species OrthoFinder orthogroup membership table. Every species and every Tier-A family is shown.
- Pfam/domain panels use every currently audited Arabidopsis, Prunus and Pyrus protein annotation. A 26-species Pfam expansion may be added only if all Tier-A proteins from all species are scanned with one unchanged Pfam release and threshold.
- The mechanism network uses the four fixed descriptive modules: transcriptional relay; ABA metabolism/signaling; receptor/transport/metabolism; unresolved stress protein.
- Node borders encode evidence sources (GO, two-genus direction, two-genus G-box, three-way Pfam, independent H3K4me3 intersection). Literature affects labels and dashed hypothesis edges only; it never changes candidate rank or inclusion.
- All network edges are labeled as literature-supported context or study-generated hypothesis. No causal language is permitted.

## 7. Functional and technical visual redesign

- Functional conclusion-reversal panels include all four primary cells and all 16 prespecified populations.
- A paired slope/dumbbell panel displays `Tree − Base` beside `Tree − strongest matched DAPT control` for the same cell; cell order is fixed as Prunus-linear, Prunus-XGBoost, Pyrus-linear, Pyrus-XGBoost.
- Technical panels include all 24 primary NovelFamily cells and all 96 arm-versus-Base descriptive contrasts.
- Existing 2,000-replicate bootstrap intervals and BH results are reused. No new significance threshold or favorable subset is introduced.
- Calibration and top-k panels show every prespecified secondary cell in supplementary material; four significant calibration disadvantages may be annotated but not shown alone.

## 8. Output, provenance and QA contract

Each new main or supplementary figure must have:

1. a source-data TSV/Parquet file;
2. a metadata JSON recording inputs, parameters, hashes, analysis tier and `malus_accessed=false`;
3. PNG, PDF and SVG outputs generated from one plotting script;
4. a test or independent audit checking row counts, arm/species coverage, numerical alignment and forbidden-scope access;
5. manual cross-format visual inspection for clipping, legibility, color consistency and annotation accuracy.

The final paper and submission bundle must retain the stopped-pilot language, contain no pending markers, include all new source data and pass a new integrated audit. New artifacts must not overwrite publication-v3 frozen source tables; they are written under publication-v4-specific result, figure and report paths.
