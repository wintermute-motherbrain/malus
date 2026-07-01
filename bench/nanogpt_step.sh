#!/usr/bin/env bash
# M30 (ADR-0038): the dispatch-overhead regression benchmark.
#
# Runs examples/nanogpt.ml (toy config: C=32, T=16, B=4; max_steps hardcoded
# to 300) with `malus --bench`, which reports the warm per-step median: the
# runtime skips 3 warmup steps and times each remaining step with a GPU flush
# inside the timed region — the same methodology as bench/nanogpt_pytorch.py
# (3 warmup steps, torch.mps.synchronize() inside the timed step, median).
# Pair the two median lines to compute the Nx ratio; record results in
# docs/milestones/m29-benchmark-results.md.
#
# At this toy scale both runtimes are dispatch-bound, not compute-bound, so
# the warm median isolates dispatch-architecture cost. Run this manually
# before/after each V5 substrate milestone (M31/M32) to confirm the number
# moves and nothing regresses it. Deliberately NOT a CI assert — wall-clock
# gates flake. The M35 capstone benchmark (Karpathy config) is the hard gate;
# this is the fast tracking check.
#
# Whole-process wall-clock is also printed as a sanity number; it includes
# one-time cost (startup, MSL compile, data load/tokenize) plus post-training
# generation, so it is NOT the headline figure.
#
# Usage: bench/nanogpt_step.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MAX_STEPS=300

echo "Building malus-cli (release)..."
cargo build --release -p malus-cli

BIN=target/release/malus
if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found after build" >&2
    exit 1
fi

if [ ! -f data/tiny_shakespeare.txt ]; then
    echo "error: data/tiny_shakespeare.txt not found (examples/nanogpt.ml reads it via read_file)" >&2
    exit 1
fi

echo "Running examples/nanogpt.ml ($MAX_STEPS steps, --bench)..."
START=$(date +%s.%N)
"$BIN" --bench examples/nanogpt.ml > /tmp/malus_nanogpt_bench.log 2>&1 || {
    echo "malus run failed — see /tmp/malus_nanogpt_bench.log" >&2
    tail -40 /tmp/malus_nanogpt_bench.log >&2
    exit 1
}
END=$(date +%s.%N)

# -a: the log's generated-sample section can contain bytes grep mistakes for binary.
MEDIAN_LINE=$(grep -a '^malus bench:' /tmp/malus_nanogpt_bench.log || true)
if [ -z "$MEDIAN_LINE" ]; then
    echo "error: no 'malus bench:' line in output — bench_step_begin/end missing from the training loop?" >&2
    exit 1
fi

TOTAL=$(echo "$END - $START" | bc)

echo "$MEDIAN_LINE"
echo "(whole-process wall-clock incl. startup/MSL-compile/data-load/generation: ${TOTAL}s)"
echo "(full log: /tmp/malus_nanogpt_bench.log)"
