#!/usr/bin/env bash
#
#  bootstrap.sh
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Checks out the submodules this build needs, and no more:
#
#    third_party/core                         and its googletest, benchmark (the tests and benchmarks)
#    third_party/ts-moveables                 (its own copy of core is not needed: ours wins)
#    third_party/core-adapters                and its nlohmann_json, and its core (its CMakeLists adds it, and core's
#                                             diamond rules skip it)
#
#  Development pin (CLAUDE.md "One core version"): while third_party/core is pinned to a core commit that its
#  dependencies have not been released against, the copy of core inside core-adapters is checked out at the same
#  commit, so the build reaches one core version, as it will once the release has moved through the family. The
#  script says when it does this. It is a no-op once core-adapters pins the same core as we do.
#
#      scripts/bootstrap.sh
#

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git submodule update --init third_party/core third_party/ts-moveables third_party/core-adapters
git -C third_party/core submodule update --init third_party/googletest third_party/benchmark
git -C third_party/core-adapters submodule update --init third_party/nlohmann_json
git -C third_party/core-adapters submodule update --init --no-recommend-shallow third_party/core || true

ours="$(git -C third_party/core rev-parse HEAD)"
theirs="$(git -C third_party/core-adapters/third_party/core rev-parse HEAD 2>/dev/null || echo none)"
if [ "$ours" != "$theirs" ]; then
    echo "bootstrap: development pin - core-adapters' core ($theirs) checked out at ours ($ours)"
    git -C third_party/core-adapters/third_party/core fetch --quiet "$ROOT/third_party/core" "$ours" 2>/dev/null ||
        git -C third_party/core-adapters/third_party/core fetch --quiet --unshallow "$ROOT/third_party/core" 2>/dev/null || true
    git -C third_party/core-adapters/third_party/core checkout --quiet "$ours"
fi
echo "bootstrap: done"
