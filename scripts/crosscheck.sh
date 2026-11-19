#!/usr/bin/env bash
#
#  crosscheck.sh
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Runs scripts/crosscheck.py in a venv of its own (numpy, safetensors, gguf-py's dependencies), never the system's.
#
#      scripts/crosscheck.sh [--venv DIR] [--gguf-py DIR] [--build DIR]
#
#  --gguf-py is a directory holding llama.cpp's gguf-py `gguf` package (MIT); it is copied into the work directory
#  and imported from the copy, so the original is only read. Defaults: MF_CROSSCHECK_VENV, MF_GGUF_PY, build/.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MF_CROSSCHECK_VENV:-$ROOT/build-crosscheck/venv}"
GGUF_PY="${MF_GGUF_PY:-}"
BUILD="$ROOT/build"
while [ $# -gt 0 ]; do
    case "$1" in
        --venv) VENV="$2"; shift ;;
        --gguf-py) GGUF_PY="$2"; shift ;;
        --build) BUILD="$2"; shift ;;
        *) echo "crosscheck: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$GGUF_PY" ] || { echo "crosscheck: --gguf-py <llama.cpp/gguf-py> is needed" >&2; exit 2; }
WORK="$ROOT/build-crosscheck/work"
rm -rf "$WORK" && mkdir -p "$WORK/gguf-py"
cp -R "$GGUF_PY/gguf" "$WORK/gguf-py/"
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" -q install numpy safetensors tqdm pyyaml requests
fi
cmake --build "$BUILD" --target mf_crosscheck >/dev/null
"$VENV/bin/python" -I "$ROOT/scripts/crosscheck.py" --binary "$BUILD/tests/mf_crosscheck" --gguf-py "$WORK/gguf-py" --work "$WORK"
