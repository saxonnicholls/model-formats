//
//  model_config_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/model_config.hpp>

#include <gtest/gtest.h>

using namespace snicholls::formats;

TEST(ModelConfig, ArchitectureNamesRoundTrip)
{
    for (const auto a : {architecture::gpt2, architecture::llama, architecture::mistral, architecture::qwen2, architecture::qwen3,
                         architecture::olmoe, architecture::deepseek_v2, architecture::deepseek_v3})
        EXPECT_EQ(architecture_from_hf(hf_model_type(a)), a);
    static_assert(gguf_architecture(architecture::mistral) == "llama" && gguf_architecture(architecture::deepseek_v3) == "deepseek2");
    static_assert(!architecture_from_hf("bert").has_value());
}

TEST(ModelConfig, HoldsCoresBlock)
{
    model_config c;
    c.block = snicholls::ml::llama3_8b;
    c.layers = 32;
    EXPECT_TRUE(snicholls::ml::valid(c.block));
    EXPECT_EQ(c.block.group(), 4u);
    EXPECT_FALSE(c.bos.has_value());
}
