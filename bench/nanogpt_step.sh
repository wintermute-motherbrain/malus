#!/usr/bin/env bash
# M29 (ADR-0026, D7): malus-side half of the V4 performance baseline.
#
# Builds malus-cli in release mode and runs examples/nanogpt.ml end to end
# (its `max_steps` is hardcoded to 300 in the source), measuring total
# wall-clock time and reporting the per-step average. This is informational
# (V4 plan: "no hard pass/fail threshold at this milestone") — pair with
# bench/nanogpt_pytorch.py's output and record both, plus machine/version
# info, in docs/milestones/m29-benchmark-results.md.
#
# malus has no built-in per-step timer yet — this is the coarse,
# always-available measurement (whole-process wall-clock / 300). A
# --bench flag reporting a true per-step median is a natural follow-up if
# per-step variance (data loading, first-call JIT/kernel-compile warmup)
# turns out to matter more precisely than this average captures.
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

echo "Running examples/nanogpt.ml ($MAX_STEPS steps)..."
START=$(date +%s.%N)
"$BIN" examples/nanogpt.ml > /tmp/malus_nanogpt_bench.log 2>&1 || {
    echo "malus run failed — see /tmp/malus_nanogpt_bench.log" >&2
    tail -40 /tmp/malus_nanogpt_bench.log >&2
    exit 1
}
END=$(date +%s.%N)

TOTAL=$(echo "$END - $START" | bc)
PER_STEP=$(echo "$TOTAL / $MAX_STEPS" | bc -l)

echo "malus nanoGPT: full run ($MAX_STEPS steps) = ${TOTAL}s, avg/step = ${PER_STEP}s"
echo "(full log: /tmp/malus_nanogpt_bench.log)"
