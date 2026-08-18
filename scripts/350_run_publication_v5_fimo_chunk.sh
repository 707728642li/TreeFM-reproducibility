#!/usr/bin/env bash
set -euo pipefail

ROOT="$1"
chunk="$2"
OUT="$ROOT/results/metrics/publication_v5_motif"
PREPARED="$OUT/prepared"
HITS="$OUT/fimo_chunks"
LOGS="$OUT/logs"
motif="$PREPARED/motif_chunks/jaspar_plants_${chunk}.meme"
output="$HITS/fimo_${chunk}.tsv.gz"
log="$LOGS/fimo_${chunk}.stderr.log"

test -s "$motif"
fimo \
    --text \
    --thresh 1e-5 \
    --max-stored-scores 10000000 \
    "$motif" \
    "$PREPARED/positive_promoters.fasta" \
    2>"$log" \
    | gzip -c >"$output.partial"
gzip -t "$output.partial"
mv "$output.partial" "$output"
