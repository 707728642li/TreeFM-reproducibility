#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${TREEFM_ROOT:-$(pwd)}}"
OUT="$ROOT/results/metrics/publication_v5_gene_trees"
PREPARED="$OUT/prepared"
RUNS="$OUT/runs"
LOGS="$OUT/logs"
mkdir -p "$RUNS" "$LOGS"

mafft --version >"$LOGS/mafft.version.txt" 2>&1
clipkit --version >"$LOGS/clipkit.version.txt" 2>&1
iqtree --version >"$LOGS/iqtree.version.txt" 2>&1

run_family() {
    local og="$1"
    local work="$RUNS/$og"
    mkdir -p "$work"
    cp "$PREPARED/$og.raw.faa" "$work/$og.raw.faa"
    cp "$PREPARED/$og.smap.tsv" "$work/$og.smap.tsv"

    mafft --auto --thread 8 "$work/$og.raw.faa" \
        >"$work/$og.aligned.faa" \
        2>"$work/mafft.log"

    clipkit "$work/$og.aligned.faa" \
        -m smart-gap \
        -o "$work/$og.trimmed.faa" \
        >"$work/clipkit.log" 2>&1

    if [ "$og" = "OG0000025" ]; then
        # Frozen feasibility amendment: the unrestricted search proposed 1,232
        # models and completed only 17 in ~2 wall-clock hours.
        iqtree \
            -s "$work/$og.trimmed.faa" \
            -m MFP \
            --mset LG,JTT,WAG,Q.PLANT \
            --mrate E,G4,R4,R6,R8 \
            -B 1000 \
            --alrt 1000 \
            -T 32 \
            --seed 20260803 \
            --prefix "$work/primary" \
            --redo \
            >"$work/iqtree.stdout.log" 2>"$work/iqtree.stderr.log"
    else
        iqtree \
            -s "$work/$og.trimmed.faa" \
            -m MFP \
            -B 1000 \
            --alrt 1000 \
            -T 8 \
            --seed 20260803 \
            --prefix "$work/primary" \
            --redo \
            >"$work/iqtree.stdout.log" 2>"$work/iqtree.stderr.log"
    fi

    test -s "$work/primary.treefile"
    test -s "$work/primary.contree"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$work/primary.complete"
}
export -f run_family
export ROOT OUT PREPARED RUNS LOGS

cut -f1 "$OUT/family_input_summary.tsv" | tail -n +2 \
    | parallel --jobs 14 --delay 0.2 --halt soon,fail=1 run_family {}

date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/primary_all.complete"
