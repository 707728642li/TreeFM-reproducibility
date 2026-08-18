#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${TREEFM_ROOT:-$(pwd)}}"
OUT="$ROOT/results/metrics/publication_v5_gene_trees"
DOMAIN="$OUT/sensitivity_domain"

run_domain_tree() {
    local og="$1"
    local work="$DOMAIN/$og"
    mafft --auto --thread 8 "$work/$og.domain.faa" \
        >"$work/$og.domain.aligned.faa" 2>"$work/mafft.log"
    clipkit "$work/$og.domain.aligned.faa" -m smart-gap \
        -o "$work/$og.domain.trimmed.faa" >"$work/clipkit.log" 2>&1
    iqtree \
        -s "$work/$og.domain.trimmed.faa" \
        -m MFP \
        -B 1000 \
        --alrt 1000 \
        -T 8 \
        --seed 20260803 \
        --prefix "$work/domain" \
        --redo \
        >"$work/iqtree.stdout.log" 2>"$work/iqtree.stderr.log"
    test -s "$work/domain.contree"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$work/domain.complete"
}
export -f run_domain_tree
export ROOT OUT DOMAIN

printf '%s\n' OG0000025 OG0000413 OG0000277 \
    | parallel --jobs 3 --delay 0.2 --halt soon,fail=1 run_domain_tree {}

date -u +%Y-%m-%dT%H:%M:%SZ >"$DOMAIN/domain_all.complete"
