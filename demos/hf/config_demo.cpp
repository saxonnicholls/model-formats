//
//  config_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A Hugging Face config.json read into core's transformer_config.

#include <snicholls/formats/hf/config.hpp>

#include <cstdio>

using namespace snicholls::formats;

int main()
{
    //! [config]
    const auto c = hf::parse_config(R"({"model_type":"llama","hidden_size":4096,"num_attention_heads":32,
        "num_key_value_heads":8,"num_hidden_layers":32,"intermediate_size":14336,"rms_norm_eps":1e-05,
        "rope_theta":500000.0,"vocab_size":128256,"max_position_embeddings":8192,"hidden_act":"silu",
        "rope_scaling":{"rope_type":"llama3","factor":8.0}})");
    const bool same_block_as_core = same_block(c->block, snicholls::ml::llama3_8b);
    for (const auto& note : c->notes)
        std::printf("not mapped: %s\n", note.c_str());
    //! [config]
    std::printf("%zu layers, GQA group %zu, RoPE base %g; core's llama3_8b: %s\n", c->layers, c->block.group(), c->block.rope_base,
                same_block_as_core ? "the same block" : "different");
    const auto g = hf::parse_generation_config(R"({"eos_token_id":[128001,128009],"temperature":0.6,"top_p":0.9})");
    std::printf("generation: %zu end tokens, temperature %g\n", g->eos.size(), *g->temperature);
    return c.is_ok() && same_block_as_core && c->notes.size() == 1 && g.is_ok() && g->eos.size() == 2 ? 0 : 1;
}
