# model-formats

Header-only C++20 library that reads and writes the machine-learning model file formats and downloads models from
the Hugging Face Hub, for the snicholls libraries. Namespace `snicholls::formats`, include root
`<snicholls/formats/...>`, CMake targets `snicholls::formats`, `snicholls::formats_io`, `snicholls::formats_hub`.

## Authorship and commits

Commits are authored by the repository owner and carry **no AI attribution**: no `Co-Authored-By` trailers, no
"Generated with" lines, in commit messages, pull request descriptions, file headers, README, CHANGELOG or this file.
Every source file starts with the banner

    //
    //  <name>.hpp
    //  model-formats
    //
    //  Copyright 2010-2026 Saxon Herschel Nicholls
    //

(`#` instead of `//` in Python, shell and CMake). Commits are small, one area each, with plain messages. The
repository is local; it has no remote yet.

## Why a library of its own (placement)

The family places code by what it needs. Core is the standard library only, with no I/O and no codecs; core-adapters
joins core to one third-party library each; ts-moveables owns the operating system. Model files need files,
networking and JSON together, so they are a library of their own, above all three:

- **core** (`third_party/core`): the vocabulary: `result`, `ml::transformer_config`, `ml::minifloat` (F16, BF16, the
  eight-bit floats), `ml/quantize.hpp` (the Q8_0, Q4_0 and MXFP4 blocks), `ml/bpe.hpp`, `reflect::tie`, `endian`,
  `checked`.
- **ts-moveables** (`third_party/ts-moveables`): `mapped_region` (reading mapped files), `write_file_atomically`,
  `read_file`, `load_config` (the environment, never `getenv` here), the event loop, `http::client` and
  `openssl_client_context`.
- **core-adapters** (`third_party/core-adapters`): `snicholls::core_json`, nlohmann/json with implicit conversions off.

Decided 2026-11-19 (approved): model-formats is its own library, and the Hugging Face download uses ts-moveables'
`http::client` over TLS and its file and mmap modules for the cache, and no other HTTP library.

## What belongs here

Parsers and writers of model files, the mapping of their metadata onto core's types, model-level conversion, the
Hub client and its cache, and the tools. Not here:

- anything the standard library alone can do that is not tied to a file format belongs in core: the ggml quantisation
  layouts, the byte-level BPE alphabet, `transformer_config`'s equality, SHA-2 and CRC-32 are core's, and the copies
  here (`gguf/ggml_types.hpp`'s non-core rows, `detail/byte_level.hpp`, `model_config.hpp`'s `same_block`,
  `hub/digest.hpp`'s OpenSSL SHA-256, `detail/crc32.hpp`) are marked to move there when core has them (proposals in
  the family's list);
- operating-system code that is not specific to model files belongs in ts-moveables: `io::write_streamed` (a
  streaming atomic write) and a read-only mapping are proposed there;
- tensor arithmetic and tokenisation belong in core (`ml/`); this library hands core the data.

## Layout

- `include/snicholls/formats/`: `version`, `error`, `dtype`, `tensor`, `model_config`, `vocabulary`; areas
  `safetensors/` (`safetensors`, `index`), `gguf/` (`ggml_types`, `gguf`, `keys`, `metadata`), `hf/` (`config`,
  `tokenizer`, `names`), `numpy/` (`npy`), `llmc/` (`llmc`), `pytorch/` (`pickle`, `checkpoint`), `archive/` (`zip`),
  `convert/` (`convert`, `export`), `io/` (`files`, `model_dir`: ts-moveables), `hub/` (`digest`, `cache`, `pattern`,
  `client`: ts-moveables and OpenSSL). `detail/` holds internals (`snicholls::formats::detail`), excluded from the
  documentation and the mirror.
- `tests/`, `demos/` mirror the headers; `tests/support/` (test-only helpers: a fake Hub on localhost, pickles built
  opcode by opcode, temporary directories); `tests/crosscheck/` (the C++ half of the cross-check); `benchmarks/`;
  `tools/` (`mf-inspect`, `mf-convert`, `mf-download`, their smoke run); `scripts/`; `docs/`; `docker/ubuntu/`;
  `third_party/` (submodules only).
- Header guards `snicholls_formats_<path>_hpp`; macros `SNICHOLLS_FORMATS_*`; CMake options `MODEL_FORMATS_*`.

## Design rules (MUST)

- **Expected failures are values**: every reader, writer and the Hub client return `formats::result<T>` (core's
  `result<T, formats::error>`): a code (`errc`) and a message. A malformed file is expected; nothing throws for one.
  No nlohmann exception escapes (`detail/json.hpp` parses without exceptions and serialises in try/catch).
- **Zero-copy views**: readers take `std::span<const std::byte>` (a mapped file) and return views into it; names
  owned only where the format escapes them (JSON). A view type borrows; say so in its documentation.
- **Validate before use**: every length, count and offset from a file is checked against the bytes left before it is
  used or allocated; extents multiply with `checked_mul`; tensors may not overlap or leave their data; duplicate keys
  and names are errors; names from files are quoted with `quote_name` (64 characters at most) in messages.
- **Never execute a pickle**: `pytorch/pickle.hpp` is the only unpickler, its opcode and global lists are closed, and
  a new entry needs a test that the dangerous neighbours stay refused.
- **The Hub**: the token is a `secret`, sent to the endpoint's origin only, never logged or put in an error; nothing
  from the server names a path outside the cache (`valid_repo_id`, `valid_revision`, `safe_relative_name`); no blob
  is renamed into the cache unverified.
- **Deterministic writers**: the same tensors give the same bytes (safetensors sorted by element size then name, GGUF
  in insertion order, zip with fixed timestamps, `.npy` byte for byte as `np.save`).
- **Little-endian hosts** only (a `static_assert`); big-endian files are refused.
- **Never an `if constexpr` chain over a type** in public code (core's rule); type dispatch by overloads.

## One core version (MUST)

Core's diamond rules (core CLAUDE.md "Versioning and the diamond"): `third_party/core` pinned by tag (`v0.16.0`),
added first so its copies inside ts-moveables and core-adapters are skipped; `snicholls_core_require(0.16.0)`; the
top-level `CMakeLists.txt` ends with `snicholls_core_assert_single_version()`. core-adapters (`v0.1.1`) still pins core
`v0.9.0`: `scripts/bootstrap.sh` checks its copy out at our commit until core-adapters is bumped in the family's
dependency order; then the step does nothing. ts-moveables (`v1.9.0`) adds core only when no target exists, so its
copy is never reached. A core bump is `third_party: bump core vA.B.C -> vX.Y.Z`, one commit, gate green.

## Tests (MUST)

Every public entity has tests; GoogleTest (core's pinned copy). `<sub>/<name>.hpp` has `tests/<sub>/<name>_test.cpp`,
registered with `model_formats_add_test(<sub>/<name> <target>)`; `scripts/check_mirror.py` (CTest `check.mirror`)
enforces it both ways. Names `TEST(<Unit><Aspect>, <BehaviourInPlainWords>)`. Every error path a reader has is a test
case; every writer round-trips through its reader. CTest labels: `unit`, `demo`, `boundary`, `docs`, `benchmark`,
`tool` (the tools' smoke run), `network` (off by default: `-DMODEL_FORMATS_NETWORK_TESTS=ON`, a few MB from
huggingface.co; never more than 200 MB).

**Cross-checks** (`make crosscheck`, `scripts/crosscheck.sh --gguf-py <llama.cpp>/gguf-py`): gguf-py, the
`safetensors` package and numpy read our files, and our readers read theirs (including a torch.save-shaped zip
pickled by Python), in a venv of the script's own. Run it after any change to a reader or writer.

## Demos (MUST)

`<sub>/<name>.hpp` has `demos/<sub>/<name>_demo.cpp`, registered with `model_formats_add_demo`; each prints what it
does and exits non-zero when a result is not what it documents. Regions `//! [tag]` are the documentation's
`\snippet`s; demo file names are unique across directories.

## Benchmarks

Anything on a hot path has a Google Benchmark against the hand-written equivalent on the same data, refusing to run
if they disagree (`benchmarks/`); a smoke run of each is a CTest (label `benchmark`). `scripts/run_benchmarks.sh`
rewrites `BENCHMARKS.md` (never edited by hand); only Release numbers mean anything.

## Documentation (MUST) - Doxygen

Core's standard: every public entity documented, in the order brief, overview, relationships, implementations,
guarantees, errors, examples (`\snippet`), performance, tests and demos, dependencies, known issues, status (the
aliases in `docs/Doxyfile.in`). Warnings are errors (`make docs`; CTest `docs.build` in the Ubuntu gate's Release
build).

## Build and gates

    scripts/bootstrap.sh   # the submodules
    make test-werror       # macOS: every CTest, warnings as errors
    make docs              # Doxygen, warnings as errors
    make check             # version and mirror
    make ubuntu            # Ubuntu 24.04 in Docker: GCC 13 and Clang 18, Release, Debug, ASan+UBSan under both
    make crosscheck        # the Python cross-check
    make network           # the network tests

A change is done when `make test-werror`, `make docs`, `make check` and `make ubuntu` pass. Docker is shared with
other work: never stop a container or process this repository did not start; if Docker is busy, wait and retry.
OpenSSL: on macOS CMake takes Homebrew's newest (`openssl@4`, else `@3`), so its headers agree with
`/usr/local/include`; the hub target builds against the 3.0 API (`OPENSSL_API_COMPAT`).

## Dependencies

| Dependency | Path | Pinned to | Used by |
|---|---|---|---|
| snicholls-core | `third_party/core` | `v0.16.0` | everything |
| ts-moveables | `third_party/ts-moveables` | `v1.9.0` | `formats_io`, `formats_hub` |
| core-adapters (nlohmann/json `v3.12.0`) | `third_party/core-adapters` | `v0.1.1` | `formats` |
| OpenSSL | the system | 3.x or 4.x | `formats_hub` (TLS, SHA-256 until core has it) |
| GoogleTest, Google Benchmark | core's `third_party/` | `v1.18.0`, `v1.9.5` | tests, benchmarks |

Rules (core's): pin a release tag, https, `shallow = true`, never edit inside a submodule, `add_subdirectory(...
EXCLUDE_FROM_ALL)` with the dependency's tests off, one bump per commit (`third_party: bump <name> <old> -> <new>`).
Defects found in a dependency are recorded in `erratum/<library>.md` with the workaround.

## Reference material

llama.cpp's gguf-py and docs (MIT) are the reference for GGUF, its keys, ggml's block sizes and the converter's
naming and RoPE permutation; transformers' configurations for `config.json`; `huggingface_hub` for the cache layout
and the Hub's API. Read them; never copy code from a licence that does not allow it; record what was taken where.

## Left to do

ONNX (read and write); SentencePiece, WordPiece and Unigram encoding when core's tokenizer track adds them (the
`tokenizer_kind` seam is here); the remaining ggml quantisations (K-quants, i-quants, ...) when core's quantisation
track adds their layouts; DeepSeek's latent attention in `hf_to_gguf`; writing PyTorch files (not planned: write
safetensors).
