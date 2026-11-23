# Changelog

## 0.1.0 - 2026-11-23

The first version: safetensors (read, write, sharded indexes), GGUF v2/v3 (read, write, every value type and ggml
type, the standard keys, metadata to and from core's `transformer_config`), Hugging Face `config.json`,
`generation_config.json`, `tokenizer.json` / `tokenizer_config.json` and tensor names, NumPy `.npy` / `.npz`, llm.c
checkpoints and token shards, PyTorch zip checkpoints through a restricted unpickler, stored zip with ZIP64,
conversion between formats and model-level export, memory-mapped files and Hugging Face model directories, the Hub
client and its huggingface_hub-compatible cache, and the tools mf-inspect, mf-convert and mf-download.
Pins core v0.16.0, ts-moveables v1.10.0, core-adapters v0.1.1.
