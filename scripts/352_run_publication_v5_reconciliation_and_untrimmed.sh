#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${TREEFM_ROOT:-$(pwd)}}"
OUT="$ROOT/results/metrics/publication_v5_gene_trees"
RUNS="$OUT/runs"
REC="$OUT/reconciliation"
SENS="$OUT/sensitivity_untrimmed"
TREERECS="$ROOT/envs/publication-v5-reconcile/bin/treerecs"
mkdir -p "$REC" "$SENS" "$OUT/logs"

test -s "$OUT/primary_all.complete"
test -x "$TREERECS"

reconcile_family() {
    local og="$1"
    local work="$RUNS/$og"
    local dest="$REC/$og"
    mkdir -p "$dest"
    if [ -s "$dest/primary.contree_recs.nhx" ] \
        && [ -s "$dest/primary.contree_recs.relationships_summary.txt" ] \
        && [ -s "$dest/reconciliation.complete" ]; then
        return 0
    fi
    "$TREERECS" \
        -g "$work/primary.contree" \
        -s "$OUT/prepared/species_tree.nwk" \
        -S "$work/$og.smap.tsv" \
        -r -t 80 \
        -o "$dest" \
        -O newick:nhx:recphyloxml:svg \
        --fevent -f \
        >"$dest/stdout.log" 2>"$dest/stderr.log"
    test -s "$dest/primary.contree_recs.nhx"
    test -s "$dest/primary.contree_recs.relationships_summary.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$dest/reconciliation.complete"
}
export -f reconcile_family
export ROOT OUT RUNS REC TREERECS

cut -f1 "$OUT/family_input_summary.tsv" | tail -n +2 \
    | parallel --jobs 14 --delay 0.1 --halt soon,fail=1 reconcile_family {}

run_untrimmed() {
    local og="$1"
    local work="$RUNS/$og"
    local dest="$SENS/$og"
    local model
    local threads=8
    local precompute_pid
    local precompute_command
    mkdir -p "$dest"
    if [ -s "$dest/untrimmed.diagnostic_nonconverged.json" ]; then
        rm -f "$dest/untrimmed.running"
        return 0
    fi
    if [ -s "$dest/untrimmed.contree" ] && [ -s "$dest/untrimmed.complete" ]; then
        rm -f "$dest/untrimmed.running"
        return 0
    fi
    if [ -s "$dest/untrimmed.running" ]; then
        precompute_pid=$(awk 'NR==1{print $1}' "$dest/untrimmed.running")
        precompute_command=$(ps -p "$precompute_pid" -o args= 2>/dev/null || true)
        if [[ "$precompute_command" == *"372_precompute_publication_v5_ap2_untrimmed.sh"* ]]; then
            for _ in $(seq 1 2880); do
                if [ -s "$dest/untrimmed.diagnostic_nonconverged.json" ]; then
                    rm -f "$dest/untrimmed.running"
                    return 0
                fi
                if [ -s "$dest/untrimmed.contree" ] && [ -s "$dest/untrimmed.complete" ]; then
                    rm -f "$dest/untrimmed.running"
                    return 0
                fi
                sleep 30
            done
            echo "Timed out waiting for active AP2/ERF untrimmed precompute" >&2
            return 1
        fi
        rm -f "$dest/untrimmed.running"
    fi
    model=$(awk -F': ' '/Best-fit model according to BIC:/{print $2; exit}' "$work/primary.iqtree")
    test -n "$model"
    if [ "$og" = "OG0000025" ]; then
        threads=32
    fi
    printf '%s\n' "$model" >"$dest/selected_primary_model.txt"
    iqtree \
        -s "$work/$og.aligned.faa" \
        -m "$model" \
        -B 1000 \
        --alrt 1000 \
        -T "$threads" \
        --seed 20260803 \
        --prefix "$dest/untrimmed" \
        --redo \
        >"$dest/iqtree.stdout.log" 2>"$dest/iqtree.stderr.log"
    test -s "$dest/untrimmed.contree"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$dest/untrimmed.complete"
}
export -f run_untrimmed
export SENS

printf '%s\n' OG0000025 OG0000413 OG0000277 \
    | parallel --jobs 3 --delay 0.2 --halt soon,fail=1 run_untrimmed {}

date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/reconciliation_and_untrimmed.complete"
