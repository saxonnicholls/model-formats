#!/usr/bin/env bash
#
#  run_benchmarks.sh
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Builds the benchmarks in Release and rewrites BENCHMARKS.md (never edited by hand) with the platform, the
#  compiler, the date and each benchmark's numbers.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-bench"
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL_FORMATS_BUILD_BENCHMARKS=ON -DMODEL_FORMATS_BUILD_TESTS=OFF \
      -DMODEL_FORMATS_BUILD_TOOLS=OFF >/dev/null
cmake --build "$BUILD" >/dev/null
{
    echo "# Benchmarks"
    echo
    echo "Written by scripts/run_benchmarks.sh; never edited by hand. Release, $(uname -sm), $(c++ --version | head -1), $(date -u +%Y-%m-%d)."
    for b in dtype safetensors; do
        echo
        echo "## $b"
        echo
        echo '```'
        "$BUILD/benchmarks/bench_$b" --benchmark_min_time=0.5s 2>/dev/null | grep -E "^(BM_|Benchmark|---)"
        echo '```'
    done
} > "$ROOT/BENCHMARKS.md"
echo "run_benchmarks: BENCHMARKS.md written"
