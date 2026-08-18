#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${TREEFM_ROOT:-$(pwd)}}"
OUT="$ROOT/results/metrics/publication_v5_motif"
PREPARED="$OUT/prepared"
HITS="$OUT/fimo_chunks"
LOGS="$OUT/logs"
PARALLEL="$ROOT/envs/treefm-genome/bin/parallel"
mkdir -p "$HITS" "$LOGS"

fimo --version >"$LOGS/fimo.version.txt" 2>&1
test -x "$PARALLEL"
test -x "$ROOT/scripts/350_run_publication_v5_fimo_chunk.sh"

seq -w 1 32 | "$PARALLEL" --jobs 32 --delay 0.1 --halt soon,fail=1 \
    "$ROOT/scripts/350_run_publication_v5_fimo_chunk.sh" "$ROOT" {}

test "$(find "$HITS" -name 'fimo_*.tsv.gz' | wc -l)" -eq 32
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/fimo_all.complete"
