//
//  keys_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/gguf/keys.hpp>

#include <gtest/gtest.h>

using namespace snicholls::formats::gguf;

TEST(GgufKeys, ArchKeysJoinTheArchitecture)
{
    EXPECT_EQ(arch_key("llama", llm::context_length), "llama.context_length");
    EXPECT_EQ(arch_key("qwen3", attention::head_count_kv), "qwen3.attention.head_count_kv");
    EXPECT_EQ(arch_key("deepseek2", rope::scaling_yarn_log_multiplier), "deepseek2.rope.scaling.yarn_log_multiplier");
}

TEST(GgufKeys, NamesAreLlamaCpps)
{
    static_assert(general::architecture == "general.architecture" && tokenizer::merges == "tokenizer.ggml.merges");
    static_assert(tokenizer::separator_token_id == "tokenizer.ggml.seperator_token_id");
    static_assert(static_cast<int>(token_type::byte) == 6 && static_cast<unsigned>(file_type::mostly_bf16) == 32);
}
