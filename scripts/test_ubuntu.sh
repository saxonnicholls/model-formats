#!/usr/bin/env bash
#
#  test_ubuntu.sh
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  The suite on Ubuntu 24.04 in Docker, from any machine with Docker: the local half of the gate (CLAUDE.md "Build
#  and gates"), as core's script does it. docker/ubuntu: GCC 13, Clang 18, Doxygen 1.18, OpenSSL 3.
#
#      scripts/test_ubuntu.sh            every configuration below
#      scripts/test_ubuntu.sh --quick    GCC Release only
#      scripts/test_ubuntu.sh -j 4       at most 4 compiles and tests at once (the default: 4, Docker is shared)
#
#  Each configuration is a fresh build with warnings as errors, then every CTest:
#
#      gcc     Release   + documentation (Doxygen) + benchmark smoke runs
#      gcc     Debug
#      clang   Release
#      clang   Debug
#      clang   ASan+UBSan
#      gcc     ASan+UBSan
#
#  The checkout is mounted read-only and copied in from `git ls-files` plus the submodules' files, so nothing
#  untracked is tested and nothing is written back. Exits non-zero if any configuration fails, after running all.
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUICK=0
JOBS=4
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) QUICK=1 ;;
        -j) shift; JOBS="${1:-4}" ;;
        -j*) JOBS="${1#-j}" ;;
        -h|--help) sed -n '2,28p' "$0" | sed 's/^#\s\{0,1\}//'; exit 0 ;;
        *) echo "test_ubuntu: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

command -v docker >/dev/null || { echo "test_ubuntu: docker is not installed" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "test_ubuntu: docker is not running" >&2; exit 2; }

hash="$(shasum -a 256 "$ROOT/docker/ubuntu/Dockerfile" 2>/dev/null || sha256sum "$ROOT/docker/ubuntu/Dockerfile")"
IMAGE="model-formats-ubuntu:${hash:0:12}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "test_ubuntu: building $IMAGE" >&2
    DOCKER_BUILDKIT=1 docker build -q -t "$IMAGE" "$ROOT/docker/ubuntu" >/dev/null
fi

LIST="$(mktemp)"
OUT="$(mktemp)"
trap 'rm -f "$LIST" "$OUT"' EXIT
( cd "$ROOT" && git ls-files && git submodule foreach --quiet --recursive 'git ls-files | sed "s|^|$displaypath/|"' ) > "$LIST"

cat <<'IN_CONTAINER' | docker run --rm -i -v "$ROOT:/src:ro" -v "$LIST:/filelist:ro" -e QUICK="$QUICK" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" -e CTEST_PARALLEL_LEVEL="$JOBS" --cpus "$JOBS" "$IMAGE" nice bash -s | tee "$OUT"
set -uo pipefail
mkdir -p /work/mf && cd /src && tar -cf - -T /filelist 2>/dev/null | tar -C /work/mf -xf -
cd /work/mf
run() {   # run <label> <cmake args...>
    local label="$1"; shift
    local dir="/work/build-${label// /-}"
    local log="$dir.log"
    printf '  %-24s ' "$label"
    if cmake -S . -B "$dir" -G Ninja -DMODEL_FORMATS_WERROR=ON "$@" >"$log" 2>&1 &&
       cmake --build "$dir" >>"$log" 2>&1 &&
       ctest --test-dir "$dir" --output-on-failure >>"$log" 2>&1; then
        echo "ok   $(grep -E '^[0-9]+% tests passed' "$log" | tail -1)"
    else
        echo "FAILED"
        grep -m 8 -E "(error|warning): " "$log" | sed "s/^/      /"
        tail -40 "$log" | sed 's/^/      /'
    fi
}
echo "model-formats on $(. /etc/os-release && echo "$PRETTY_NAME"), $(uname -m)"
echo "  $(g++ --version | head -1)"
echo "  $(clang++ --version | head -1)"
echo "  doxygen $(doxygen --version), $(openssl version)"
docs=-DMODEL_FORMATS_BUILD_DOCS=ON
case "$(doxygen --version)" in 1.9.*|1.8.*) docs=-DMODEL_FORMATS_BUILD_DOCS=OFF; echo "  (docs skipped: an older Doxygen)";; esac
run "gcc Release"        -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DMODEL_FORMATS_BUILD_BENCHMARKS=ON "$docs"
if [ "$QUICK" != 1 ]; then
    run "gcc Debug"          -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug
    run "clang Release"      -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
    run "clang Debug"        -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
    run "clang ASan+UBSan"   -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMODEL_FORMATS_SANITIZE=address,undefined
    run "gcc ASan+UBSan"     -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMODEL_FORMATS_SANITIZE=address,undefined
fi
IN_CONTAINER

passed="$(grep -cE '^  [^ ].* ok   ' "$OUT" || true)"
failed="$(grep -cE '^  [^ ].* FAILED$' "$OUT" || true)"
echo
if [ "$failed" = 0 ] && [ "$passed" -gt 0 ]; then
    echo "test_ubuntu: all $passed configurations passed"
    exit 0
fi
echo "test_ubuntu: $failed of $((passed + failed)) configurations FAILED"
exit 1
