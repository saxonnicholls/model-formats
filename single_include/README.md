# The single include

`snicholls_formats.hpp` is the part of model-formats that needs no third party, in one file: `formats.hpp` (GGUF,
`.npy`, llm.c, PyTorch checkpoints, zip and the shared types) with every model-formats, snicholls-core, ts-moveables
and core-adapters header it uses expanded in place, once each, needing nothing but the standard library. The headers
that parse JSON or link OpenSSL (safetensors, the Hugging Face config and tokenizer, conversion, `io/`, `hub/`) are not
in it; `tooling.conf` lists them with the reason for each.

- **Download it** from the release for the version you pin, once the repository publishes releases: each release's
  file is generated from that release's tagged commit and compiled on its own before it is attached.
- **Build it** for any commit with `make single-include`, which writes it here (git-ignored), or with
  `python3 third_party/tooling/scripts/single_include.py --out <dir>`.

It is never committed, so it is never stale: every build generates it and compiles it alone
(`model_formats_single_include`), and the CTest `check.single_include` fails if a header is in neither `formats.hpp`
nor the opt-in list.
