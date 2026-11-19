# model-formats

[![CI](https://github.com/saxonnicholls/model-formats/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saxonnicholls/model-formats/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-doxygen-blue)](https://saxonnicholls.github.io/model-formats/)

Read and write the machine-learning model file formats, and download models from the Hugging Face Hub, for the
snicholls libraries (the ML training library first). C++20, header-only, namespace `snicholls::formats`, include root
`<snicholls/formats/...>`.

| Format | Read | Write | Notes |
|---|---|---|---|
| safetensors | yes, zero-copy, validated | yes, aligned, deterministic | every dtype: F64 F32 F16 BF16 F8_E4M3 F8_E5M2 I8-I64 U8-U64 BOOL; `__metadata__`; sharded `model.safetensors.index.json` |
| GGUF v2 / v3 | yes, zero-copy, validated | yes | every metadata type and nested arrays; every ggml type's block layout; the standard keys; metadata to and from core's `transformer_config` |
| Hugging Face directory | yes | via safetensors + config | `config.json` to `transformer_config` (GPT-2, Llama, Mistral, Qwen 2/3, OLMoE, DeepSeek-V2/V3), `generation_config.json`, `tokenizer.json` + `tokenizer_config.json` (byte-level BPE; SentencePiece, WordPiece, Unigram read, not encoded), chat template as text, tensor names both ways, llama.cpp's RoPE permutation |
| NumPy `.npy` / `.npz` | yes | yes, byte for byte `np.save` | stored `.npz`; `savez_compressed` refused |
| llm.c | yes | yes | GPT-2 checkpoint (float32 v3, bfloat16 v5) and token shards |
| PyTorch `.bin` / `.pt` | yes, **safely** | no (write safetensors) | the zip format through a restricted unpickler: only state-dict opcodes and globals, nothing executed |
| zip | yes (stored, ZIP64) | yes (stored, aligned) | the container of npz and PyTorch |
| ONNX | later | later | |

Conversion: any of these to safetensors, GGUF, `.npz` or `.npy` tensor by tensor, with casts and dequantisation (Q8_0,
Q4_0 and MXFP4 through core); model-level export of our llm.c checkpoints to Hugging Face safetensors and to llama.cpp
GGUF, and Hugging Face checkpoints to GGUF and back.

Hub: resolve a repository and revision to its commit and files; download with resume, parallel files, SHA-256 / git
blob verification, into a cache laid out as `huggingface_hub`'s (`blobs/`, `snapshots/<sha>/`, `refs/`); `HF_TOKEN`,
`HF_HUB_OFFLINE`, `HF_ENDPOINT`, `HF_HOME` / `HF_HUB_CACHE`.

Tools: `mf-inspect <file|dir>`, `mf-convert <in> <out>`, `mf-download <repo>[@rev] [--include pattern]`.

```cpp
#include <snicholls/formats/io/files.hpp>

auto m = snicholls::formats::io::open_model("model.safetensors");   // mapped; any format
for (const auto& t : m->model.tensors)
    std::printf("%s %s\n", std::string(t.name).c_str(), to_string(t.shape).c_str());
```

## Where it sits in the family

- **core** is the standard library only, with no I/O and no codecs: it owns the vocabulary this library speaks
  (`result`, `ml::transformer_config`, `ml::minifloat`, the ggml blocks of `ml/quantize.hpp`, `ml/bpe.hpp`).
- **core-adapters** joins core to one third-party library each: here, nlohmann/json (`snicholls::core_json`).
- **ts-moveables** owns the operating system: memory-mapped files (`mmap/`), atomic writes (`fs/`), the event loop,
  and the HTTP/1.1 + HTTP/2 client with TLS (`http/client.hpp`, `tls/openssl.hpp`).

Reading and writing model files needs files, networking and JSON together, so it is a library of its own above all
three, pinning core, ts-moveables and core-adapters. The Hugging Face download uses ts-moveables' `http::client` over
TLS and its file and mmap modules for the cache, and no other HTTP library (decided 2026-11-19).

| Target | Adds | For |
|---|---|---|
| `snicholls::formats` | core, core-adapters' `core_json` | the formats over bytes, conversion, export |
| `snicholls::formats_io` | ts-moveables | files: mapping, atomic writing, model directories |
| `snicholls::formats_hub` | OpenSSL 3 | the Hub client and cache |

## Pins

| Dependency | Path | Pinned to |
|---|---|---|
| snicholls-core | `third_party/core` | `v0.16.0` |
| ts-moveables | `third_party/ts-moveables` | `v1.10.0` |
| core-adapters | `third_party/core-adapters` | `v0.1.1` (nlohmann/json `v3.12.0`) |
| GoogleTest, Google Benchmark | core's `third_party/` | `v1.18.0`, `v1.9.5` |

`scripts/bootstrap.sh` checks them out. core-adapters still pins core `v0.9.0`; until its pin moves, the bootstrap
checks its copy of core out at ours, so the build reaches one core version (see CLAUDE.md).

## Build

    scripts/bootstrap.sh
    make test-werror     # every CTest, warnings as errors
    make docs            # Doxygen, warnings as errors
    make check           # version and mirror
    make ubuntu          # Ubuntu 24.04 in Docker: GCC 13 and Clang 18, Release, Debug, sanitizers
    make crosscheck      # gguf-py, safetensors and numpy against our files and theirs
    make network         # the tests that download from huggingface.co (a few MB)

MIT licence.
