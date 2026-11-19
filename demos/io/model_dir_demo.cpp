//
//  model_dir_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A Hugging Face model directory written here and opened: config, tokenizer, chat template and weights at once.

#include <snicholls/formats/io/model_dir.hpp>

#include <cstdio>
#include <fstream>

using namespace snicholls::formats;

int main()
{
    const auto dir = std::filesystem::temp_directory_path() / "model-formats-model-dir-demo";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "config.json") << R"({"model_type":"gpt2","n_embd":4,"n_head":2,"n_layer":1,"n_positions":8,"vocab_size":2})";
    std::ofstream(dir / "tokenizer.json") << R"({"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}})";
    const std::vector<float> wte(8);
    (void)io::save_tensors(dir / "model.safetensors", std::vector<tensor_view>{{"wte.weight", dtype::f32, {2, 4}, std::as_bytes(std::span(wte))}});
    //! [model_dir]
    const auto m = io::open_model_dir(dir);   // config.json, tokenizer.json, model.safetensors (or shards), mapped
    //! [model_dir]
    if (m.is_ok())
        std::printf("%s: %zu layers, %zu tokens, %zu tensors in %zu file(s)\n", hf_model_type(m->config->arch).data(), m->config->layers,
                    m->tokenizer->vocab.tokens.size(), m->tensors.size(), m->files.size());
    const bool good = m.is_ok() && m->config && m->tensors.size() == 1;
    std::filesystem::remove_all(dir);
    return good ? 0 : 1;
}
