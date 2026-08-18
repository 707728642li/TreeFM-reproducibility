# TreeFM publication-v3 statistical analysis plan

Version: 0.2 gate-triggered Pyrus amendment, 2026-07-16

## Analysis populations

Primary technical inference uses Hevea, Prunus and Pyrus after Pyrus passes Genome-QC and Task-QC. Malus remains a one-time external blind evaluation. Vitis and low-completeness species are sensitivity-only.

Primary functional inference is leave-one-genus-out across Prunus, Pyrus and Malus. Prunus avium and Prunus persica are not counted as independent genera. Vitis is reported as a failed-gate secondary biological endpoint and cannot enter the confirmatory functional meta-effect.

Family populations are all, LOGO-Homolog and LOGO-NovelFamily. Task-specific supervised exposure is primary; any-task exposure and maximum sequence identity are sensitivities.

## Primary estimands

For metric `M`, held-out genus `g`, task `t` and seed `s`:

```text
Delta_tree[g,t,s] = M(Tree) - M(RandomPlant)
Woody_control_gain[g,t,s]
  = M(Tree) - max(M(Herb), M(RandomPlant), M(PhyloGCMatch))
```

The primary technical endpoint is NovelFamily AUPRC. The primary functional endpoint is leave-one-genus-out robust-target AUPRC. Top-k enrichment and calibration are key secondary endpoints.

## Transfer determinant model

The confirmatory explanatory model is:

```text
Delta_AUPRC ~ corpus_target_kmer_similarity
            + phylogenetic_distance
            + homolog_exposure_fraction
            + GC_repeat_distance
            + task_class
            + context_length
            + (1 | heldout_genus)
            + (1 | DAPT_seed)
```

Predictors are computed without model outcomes. Collinearity, scaling and missing-data rules are frozen in version 1.0 before confirmation training.

## Uncertainty hierarchy

- gene/pair block bootstrap measures within-genus sampling uncertainty;
- DAPT seed measures training stochasticity;
- held-out genus is the biological replication unit;
- a hierarchical mixed-effects or Bayesian meta-analysis combines levels without treating windows as independent genera.

Every aggregate is accompanied by per-genus effects, worst-genus effect, direction proportion across seeds and between-seed variance.

## MDE and equivalence

Numerical MDE/ROPE values are not selected from model effects. They are set in version 1.0 using benchmark-v3 sample counts, pilot-free simulations, baseline prevalence and an application-relevance rule. Equivalence is tested with TOST or a Bayesian ROPE; `p > 0.05` alone is never called equivalence.

## Multiplicity

Two primary families are tested hierarchically:

1. NovelFamily structural/regulatory transfer;
2. cross-genus robust dormancy/cold-acclimation transfer.

All other task, layer, motif, readout and subgroup analyses are secondary or exploratory and use family-wise FDR. Mechanistic analyses occur only after the locked primary analysis and cannot redefine the winning endpoint.

The Pyrus H3K4me3 analysis is mechanistic secondary evidence under
`docs/publication_v3_pyrus_h3k4me3_mechanistic_contract.md`. Its primary
statistic is the frozen-positive minus robust-negative difference in signed
day-50-minus-day-0 promoter H3K4me3 change. A 10,000-permutation test is
stratified by baseline-expression and promoter-GC deciles. Because each
time-point ChIP/Input pair lacks biological replication, peak calls and
time-course trajectories are descriptive and cannot be used as independent
replication or differential-peak significance.

## Blind evaluation

Malus source files may be copied and integrity-checked, but outcome construction, label counts, model scores and plots remain sealed until:

1. v3 benchmark and models are frozen;
2. all non-Malus hyperparameters are selected;
3. the statistical script produces fixed empty Malus table shells;
4. the decision log records the unlock timestamp and hashes.

The Malus evaluation is run once. Failures caused by file corruption or identifier mismatch may be repaired with a documented deviation, but no model or threshold changes are allowed.
