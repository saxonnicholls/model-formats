#!/usr/bin/env bash
#
#  run_demos.sh
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Build and run the demos and show what they print. Every demo is also a CTest
#  (label demo), but ctest keeps the output; this is for reading it.
#
#      scripts/run_demos.sh --all                 every demo in demos/CMakeLists.txt
#      scripts/run_demos.sh gguf/gguf npy         one or more, by name
#      scripts/run_demos.sh --list                the names, one per line
#
#  A name is as registered in demos/CMakeLists.txt (gguf/gguf, hf/config), its
#  last part alone when that is unambiguous (npy, config), with or without a
#  _demo suffix, or the path of its source (demos/hf/config_demo.cpp). The build
#  directory is BUILD (default build, as the Makefile), configured as CONFIG
#  (default Release) with the Makefile's generator if it does not exist yet; only
#  the demos asked for are built. Exits non-zero if any demo fails, after running
#  them all. The demos need no network.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-build}"
CONFIG="${CONFIG:-Release}"
CXX="${CXX:-c++}"

# The registered names, one per line: model_formats_add_demo(<name> <target>)
registered() {
    sed -n 's/^[[:space:]]*model_formats_add_demo([[:space:]]*\([^[:space:])]*\).*/\1/p' "$ROOT/demos/CMakeLists.txt"
}
# The executable's name for a registered name, as the CMake function makes it (MAKE_C_IDENTIFIER)
target_of() { printf 'demo_%s' "$(printf '%s' "$1" | sed 's/[^A-Za-z0-9_]/_/g')"; }

ALL=0; LIST=0; WANTED=()
for arg in "$@"; do
    case "$arg" in
        --all) ALL=1 ;;
        --list) LIST=1 ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^#[[:space:]]\{0,1\}//'; exit 0 ;;
        -*) echo "run_demos: unknown option $arg" >&2; exit 2 ;;
        *) WANTED+=("$arg") ;;
    esac
done

if [ "$LIST" = 1 ]; then registered; exit 0; fi
if [ "$ALL" = 0 ] && [ "${#WANTED[@]}" = 0 ]; then
    echo "run_demos: name a demo, or --all; --list shows the names" >&2
    exit 2
fi

# Resolve what was asked for to registered names
NAMES=()
if [ "$ALL" = 1 ]; then
    while read -r name; do NAMES+=("$name"); done < <(registered)
else
    for want in "${WANTED[@]}"; do
        w="${want#demos/}"; w="${w%.cpp}"; w="${w%_demo}"
        if registered | grep -qx "$w"; then
            NAMES+=("$w"); continue
        fi
        matches="$(registered | grep -E "(^|/)${w}$" || true)"
        if [ -n "$matches" ] && [ "$(printf '%s\n' "$matches" | wc -l | tr -d ' ')" = 1 ]; then
            NAMES+=("$matches"); continue
        fi
        if [ -n "$matches" ]; then
            echo "run_demos: $want is ambiguous:" >&2; printf '    %s\n' $matches >&2
        else
            echo "run_demos: no demo named $want; the names are:" >&2; registered | sed 's/^/    /' >&2
        fi
        exit 2
    done
fi

cd "$ROOT"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
if [ ! -d third_party/core/include ]; then
    echo "run_demos: submodules missing; running scripts/bootstrap.sh"
    scripts/bootstrap.sh >"$LOG" 2>&1 || { cat "$LOG"; exit 1; }
fi
# Configure the first time with the settings above, and again with the cached ones every time after,
# so a demo registered since the tree was configured has a target to build
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    echo "run_demos: configuring $BUILD ($CONFIG)"
    gen=(); command -v ninja >/dev/null && gen=(-G Ninja)
    cmake -S . -B "$BUILD" ${gen[@]+"${gen[@]}"} -DCMAKE_BUILD_TYPE="$CONFIG" -DCMAKE_CXX_COMPILER="$CXX" >"$LOG" 2>&1 || { cat "$LOG"; exit 1; }
else
    cmake -B "$BUILD" >"$LOG" 2>&1 || { cat "$LOG"; exit 1; }
fi

# Build only what is asked for; the compiler's output is shown only if it fails
TARGETS=()
for name in "${NAMES[@]}"; do TARGETS+=("$(target_of "$name")"); done
echo "run_demos: building ${#TARGETS[@]} demo(s) in $BUILD"
cmake --build "$BUILD" --parallel --target "${TARGETS[@]}" >"$LOG" 2>&1 || { cat "$LOG"; exit 1; }

# Run each, whole output shown, and keep score
FAILED=()
for name in "${NAMES[@]}"; do
    t="$(target_of "$name")"
    exe="$BUILD/demos/$t"
    [ -x "$exe" ] || exe="$BUILD/demos/$CONFIG/$t"   # a multi-config generator
    echo
    echo "==== $name ($exe)"
    if "$exe"; then
        echo "---- $name: ok"
    else
        echo "---- $name: FAILED (exit $?)"
        FAILED+=("$name")
    fi
done

echo
if [ "${#FAILED[@]}" = 0 ]; then
    echo "run_demos: ${#NAMES[@]} demo(s) ok"
    exit 0
fi
echo "run_demos: ${#FAILED[@]} of ${#NAMES[@]} demo(s) FAILED: ${FAILED[*]}" >&2
exit 1
