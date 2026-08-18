# Publication-v3 local rebuild pilot amendment (2026-08-01)

## Reason and scope

The previous local TreeFM workspace, including all DAPT checkpoints, was accidentally deleted. The user explicitly chose not to attempt data recovery and authorized a clean rerun. Server-retained frozen data, scripts, release gates, and non-model biological results were copied into a newly created local project workspace. No prior DAPT outcome or downstream model-comparison result is available in the rebuilt workspace.

This amendment does not change the frozen training or evaluation contracts. It defines only an efficiency-first pilot order after the deletion.

## Frozen pilot

- Backbone: `kuleshov-group/PlantCaduceus_l20`, revision `d1b3c7d42abacce3125f0119adcec161fd8d59bd`.
- DAPT arms: `tree`, `herb`, `random_plant`, and `phylogc_match`.
- Seed: 23 for all four arms.
- Training implementation and all hyperparameters: unchanged `scripts/109_train_plantcad_publication_v3_dapt.py` under the existing publication-v3 release gate.
- Compute: local WSL2 only, two local RTX 3090 GPUs, exactly two DDP ranks.
- Evaluation: frozen non-Malus technical benchmarks and Prunus/Pyrus functional datasets only. Malus remains sealed and is not permitted for pilot selection.

## Prospective continuation rule

Proceed to the remaining preregistered seeds only when the seed-23 Tree arm shows a coherent advantage over the strongest matched control:

1. at least one primary cross-species functional readout improves by at least 0.02 absolute AUPRC; and
2. technical transfer is positive in at least two of four frozen tasks, without a systematic reversal across tasks.

If these conditions are not met, do not spend the full multi-seed budget or unseal Malus. Report the pilot as exploratory evidence and reassess the model or biological hypothesis. The original multi-seed estimands remain confirmatory only if the full frozen design is subsequently completed.

## Integrity notes

- The pilot order was fixed before the rebuilt models produced any downstream result.
- Existing frozen training/evaluation implementations, labels, splits, and release gates are not edited by this amendment.
- On 2026-08-01, before any rebuilt downstream result existed, the embedding-only runtime scheduler was changed from per-arm barriers to two independent queues: all technical embeddings run sequentially on local GPU0 and all functional embeddings run sequentially on local GPU1. This changes wall time only; arm order, model checkpoints, seeds, inputs, embedding transformation, and output contracts are unchanged.
- After all five seed-23 functional embedding manifests were complete but before any seed-23 downstream metric existed, GPU1 was assigned the technical `phylogc_match` and `random_plant` tail arms while GPU0 retained `base`, `tree`, and `herb`. The GPU1 tail worker uses inference batch size 256 with an OOM fallback of 192; GPU0 uses the frozen default of 128. Batch size affects inference scheduling only: all checkpoints remain in evaluation mode, no parameter is fitted during extraction, the same center-token RCPS pooling is applied, and every float16 output is row/model/hash audited.
- Also before any downstream metric existed, the technical extractor gained atomic per-species locks, reverse panel traversal, and canonical manifest ordering. When GPU1 becomes idle after its two tail arms, it may traverse the active GPU0 `tree` or `herb` panel in reverse. The two workers can therefore divide species without computing or overwriting the same species; the numerical embedding implementation and final manifest contract are unchanged.
- Completed-arm technical and functional embeddings are synchronized to an arm-specific snapshot workspace under the `bioserver` project rather than waiting for all five arms. The two frozen probe programs then run concurrently against that immutable snapshot. On success, probe models and compact metrics are promoted under a per-track file lock to the main result tree. This isolation prevents the later all-arm embedding sync from overwriting arrays that an incremental probe is reading. A completion marker stores the SHA256 of the embedding manifest, whose rows already contain the hashes of every embedding matrix. The final all-arm launcher reuses an arm only when the current manifest SHA256 matches this marker and both metrics and run-spec artifacts exist.
- Because all five functional embedding manifests completed well before the larger technical panel, the five frozen Prunus/Pyrus probe jobs are permitted to run immediately and concurrently (12 server threads per arm, 60 maximum) in those same immutable arm snapshots. This exposes the preregistered functional gate earlier but does not change its estimand, labels, cross-genus split, readouts, controls, or the requirement that the technical gate also pass before multi-seed continuation.
- The incremental finalizer applies the unchanged prospective rule and writes the same audited Chinese pilot report as soon as all five fingerprint-matched arm results are local. This is a redundant result-delivery path, not an alternative analysis.
- A Windows PowerShell compatibility defect was corrected before embedding evaluation: redirected `wsl.exe` processes returned a null `ExitCode`, causing successful QC to be misclassified as failed. The replacement background-job wrapper captures `$LASTEXITCODE` explicitly. The QC output itself was successful and no checkpoint or metric was recomputed because of this controller-only correction.
- The same independent-queue schedule is used for a gate-authorized full three-seed evaluation. Existing pilot embeddings are passed back through the extractors' frozen row-hash, model-weight-hash, and tensor-shape reuse checks; only missing or invalid matrices are recomputed.
- Rebuild provenance, environment versions, model/data hashes, run statuses, and logs must be retained under the project root.
