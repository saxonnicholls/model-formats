# Changelog

## 0.2.0 - 2026-11-27

On snicholls-core v0.28.0, ts-moveables v1.12.1, core-adapters v0.1.5 and tooling v0.3.2, so every copy of core in
the build is v0.28.0.

### Added

- The umbrella `formats.hpp` and the single include `snicholls_formats.hpp`, generated from it by snicholls-tooling's
  `single_include.py` (`third_party/tooling`, build-time only, checked out by `scripts/bootstrap.sh`),
  with snicholls-core, ts-moveables and core-adapters expanded, so it needs nothing but the standard library. The
  headers that parse JSON (nlohmann/json) or link OpenSSL are opted out, each with its reason in `tooling.conf`. It is
  never committed: every build generates it and compiles it alone, the CTest `check.single_include`, `make check` and
  CI fail while a header is missing from `formats.hpp`, every release tag attaches it to the GitHub release, and
  `make single-include` writes it locally. tooling's interface check (`check.interfaces`) runs too.

## 0.1.0 - 2026-11-23

The first version: safetensors (read, write, sharded indexes), GGUF v2/v3 (read, write, every value type and ggml
type, the standard keys, metadata to and from core's `transformer_config`), Hugging Face `config.json`,
`generation_config.json`, `tokenizer.json` / `tokenizer_config.json` and tensor names, NumPy `.npy` / `.npz`, llm.c
checkpoints and token shards, PyTorch zip checkpoints through a restricted unpickler, stored zip with ZIP64,
conversion between formats and model-level export, memory-mapped files and Hugging Face model directories, the Hub
client and its huggingface_hub-compatible cache, and the tools mf-inspect, mf-convert and mf-download.
Pins core v0.16.0, ts-moveables v1.10.0, core-adapters v0.1.1.
