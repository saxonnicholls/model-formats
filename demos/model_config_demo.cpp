//
//  model_config_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A model_config: core's block for one architecture and the sizes around it.

#include <snicholls/formats/model_config.hpp>

#include <cstdio>

using namespace snicholls::formats;

int main()
{
    //! [model_config]
    model_config c;
    c.arch = architecture::llama;
    c.block = snicholls::ml::llama3_8b;   // core's preset
    c.layers = 32;
    c.vocab = 128256;
    c.context = 8192;
    //! [model_config]
    std::printf("%s (GGUF %s): %zu layers of d_model %zu, %zu heads / %zu KV heads, vocab %zu\n", hf_model_type(c.arch).data(),
                gguf_architecture(c.arch).data(), c.layers, c.block.d_model, c.block.n_heads, c.block.n_kv_heads, c.vocab);
    return snicholls::ml::valid(c.block) && architecture_from_hf("qwen3") == architecture::qwen3 ? 0 : 1;
}
