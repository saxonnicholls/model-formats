//
//  names_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/hf/names.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using namespace snicholls::formats;
using namespace snicholls::formats::hf;

TEST(HfNames, Gpt2NamesRoundTrip)
{
    const auto t = parse_hf_name(architecture::gpt2, "h.11.attn.c_attn.weight");
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(t->what, role::qkv);
    EXPECT_EQ(t->layer, 11u);
    EXPECT_TRUE(hf_transposed(architecture::gpt2, *t));
    EXPECT_EQ(hf_name(architecture::gpt2, *t), "h.11.attn.c_attn.weight");
    EXPECT_EQ(gguf_name(*t), "blk.11.attn_qkv.weight");
    EXPECT_EQ(parse_hf_name(architecture::gpt2, "transformer.wte.weight")->what, role::token_embedding);
    EXPECT_EQ(parse_hf_name(architecture::gpt2, "ln_f.bias")->bias, true);
    EXPECT_EQ(parse_hf_name(architecture::gpt2, "h.0.attn.bias")->what, role::mask_buffer);
    EXPECT_EQ(parse_hf_name(architecture::gpt2, "h.0.attn.masked_bias")->what, role::mask_buffer);
    EXPECT_FALSE(hf_transposed(architecture::gpt2, *parse_hf_name(architecture::gpt2, "h.0.mlp.c_fc.bias")));
    EXPECT_FALSE(parse_hf_name(architecture::gpt2, "h.x.ln_1.weight").has_value());
    EXPECT_FALSE(parse_hf_name(architecture::gpt2, "h.0.ln_3.weight").has_value());
    for (const char* n : {"wte.weight", "wpe.weight", "ln_f.weight", "lm_head.weight", "h.3.ln_1.bias", "h.3.attn.c_proj.weight",
                          "h.3.ln_2.weight", "h.3.mlp.c_fc.weight", "h.3.mlp.c_proj.bias", "h.3.attn.bias"})
        EXPECT_EQ(hf_name(architecture::gpt2, *parse_hf_name(architecture::gpt2, n)), n);
}

TEST(HfNames, LlamaFamilyNamesRoundTrip)
{
    for (const char* n : {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "model.layers.0.input_layernorm.weight",
                          "model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.bias", "model.layers.1.self_attn.v_proj.weight",
                          "model.layers.1.self_attn.o_proj.weight", "model.layers.2.self_attn.q_norm.weight", "model.layers.2.post_attention_layernorm.weight",
                          "model.layers.2.mlp.gate_proj.weight", "model.layers.2.mlp.up_proj.weight", "model.layers.2.mlp.down_proj.weight",
                          "model.layers.3.mlp.gate.weight", "model.layers.3.mlp.experts.17.up_proj.weight",
                          "model.layers.3.mlp.shared_experts.down_proj.weight", "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
                          "model.layers.3.self_attn.q_a_layernorm.weight", "model.layers.3.mlp.gate.e_score_correction_bias"}) {
        const auto t = parse_hf_name(architecture::deepseek_v3, n);
        ASSERT_TRUE(t.has_value()) << n;
        EXPECT_EQ(hf_name(architecture::deepseek_v3, *t), n);
    }
    const auto e = parse_hf_name(architecture::olmoe, "model.layers.3.mlp.experts.17.up_proj.weight");
    EXPECT_EQ(e->expert, 17u);
    EXPECT_EQ(gguf_name(*e), "blk.3.ffn_up_exps.weight");   // stacked in GGUF
    EXPECT_FALSE(parse_hf_name(architecture::llama, "model.layers.1.self_attn.rotary_emb.inv_freq").has_value());
    EXPECT_FALSE(parse_hf_name(architecture::llama, "model.layers.1.mlp.experts.x.up_proj.weight").has_value());
    EXPECT_TRUE(hf_name(architecture::llama, tensor_name{role::qkv, 0, {}, false}).empty());
}

TEST(HfNames, GgufNamesRoundTrip)
{
    for (const char* n : {"token_embd.weight", "position_embd.weight", "output_norm.bias", "output.weight", "blk.0.attn_norm.weight",
                          "blk.0.attn_q.weight", "blk.0.attn_k.bias", "blk.5.attn_output.weight", "blk.5.ffn_gate_inp.weight",
                          "blk.5.ffn_down_exps.weight", "blk.5.attn_kv_b.weight", "blk.5.attn_q_norm.weight", "blk.5.exp_probs_b.bias"}) {
        const auto t = parse_gguf_name(n);
        ASSERT_TRUE(t.has_value()) << n;
        EXPECT_EQ(gguf_name(*t), n);
    }
    EXPECT_FALSE(parse_gguf_name("blk.0.unknown.weight").has_value());
    EXPECT_FALSE(parse_gguf_name("rope_freqs").has_value());
    EXPECT_EQ(hf_name(architecture::llama, *parse_gguf_name("blk.4.ffn_gate.weight")), "model.layers.4.mlp.gate_proj.weight");
    EXPECT_TRUE(gguf_name(tensor_name{role::mask_buffer, 0, {}, true}).empty());
}

TEST(HfNames, PermuteRopeIsLlamaCppsReshapeAndSwap)
{
    // numpy: w.reshape(heads, 2, d/2, cols).swapaxes(1, 2).reshape(w.shape) for heads = 2, d = 4, one column:
    // rows 0..7 -> 0 2 1 3 4 6 5 7
    std::vector<std::int32_t> in(8), out(8), back(8);
    std::iota(in.begin(), in.end(), 0);
    permute_rope(std::as_bytes(std::span(in)), std::as_writable_bytes(std::span(out)), 2, sizeof(std::int32_t));
    EXPECT_EQ(out, (std::vector<std::int32_t>{0, 2, 1, 3, 4, 6, 5, 7}));
    unpermute_rope(std::as_bytes(std::span(out)), std::as_writable_bytes(std::span(back)), 2, sizeof(std::int32_t));
    EXPECT_EQ(back, in);
    // rows of three columns move whole; head_dim 6: halves (0 1 2)(3 4 5) -> 0 3 1 4 2 5
    std::vector<std::int32_t> wide(18), moved(18);
    std::iota(wide.begin(), wide.end(), 0);
    permute_rope(std::as_bytes(std::span(wide)), std::as_writable_bytes(std::span(moved)), 1, 3 * sizeof(std::int32_t));
    EXPECT_EQ(moved, (std::vector<std::int32_t>{0, 1, 2, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17}));
    EXPECT_TRUE(gguf_permutes(architecture::llama, tensor_name{role::k, 0, {}, false}));
    EXPECT_FALSE(gguf_permutes(architecture::qwen2, tensor_name{role::k, 0, {}, false}));
    EXPECT_FALSE(gguf_permutes(architecture::llama, tensor_name{role::v, 0, {}, false}));
}
