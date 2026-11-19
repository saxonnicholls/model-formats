//
//  config_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/hf/config.hpp>

#include <gtest/gtest.h>

using namespace snicholls::formats;
namespace ml = snicholls::ml;

namespace
{
constexpr const char* gpt2 = R"({"model_type":"gpt2","n_embd":768,"n_head":12,"n_layer":12,"n_positions":1024,"n_ctx":1024,
  "vocab_size":50257,"layer_norm_epsilon":1e-05,"activation_function":"gelu_new","n_inner":null,"bos_token_id":50256,
  "eos_token_id":50256})";
constexpr const char* llama3 = R"({"model_type":"llama","hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,
  "num_hidden_layers":32,"intermediate_size":14336,"rms_norm_eps":1e-05,"rope_theta":500000.0,"vocab_size":128256,
  "max_position_embeddings":8192,"hidden_act":"silu","tie_word_embeddings":false,"bos_token_id":128000,"eos_token_id":128001,
  "rope_scaling":null,"attention_bias":false})";
constexpr const char* qwen3 = R"({"model_type":"qwen3","hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,
  "head_dim":128,"intermediate_size":3072,"num_hidden_layers":28,"rms_norm_eps":1e-06,"rope_theta":1000000,"vocab_size":151936,
  "max_position_embeddings":40960,"tie_word_embeddings":true,"attention_bias":false,"use_sliding_window":false,
  "sliding_window":null,"max_window_layers":28,"hidden_act":"silu"})";
constexpr const char* olmoe = R"({"model_type":"olmoe","hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":16,
  "intermediate_size":1024,"num_hidden_layers":16,"num_experts":64,"num_experts_per_tok":8,"norm_topk_prob":false,
  "rope_theta":10000.0,"rms_norm_eps":1e-05,"router_aux_loss_coef":0.01,"vocab_size":50304,"max_position_embeddings":4096,
  "clip_qkv":null,"hidden_act":"silu"})";
constexpr const char* deepseek3 = R"({"model_type":"deepseek_v3","hidden_size":7168,"num_attention_heads":128,
  "num_key_value_heads":128,"num_hidden_layers":61,"intermediate_size":18432,"moe_intermediate_size":2048,
  "n_routed_experts":256,"n_shared_experts":1,"num_experts_per_tok":8,"n_group":8,"topk_group":4,"topk_method":"noaux_tc",
  "scoring_func":"sigmoid","norm_topk_prob":true,"routed_scaling_factor":2.5,"first_k_dense_replace":3,"q_lora_rank":1536,
  "kv_lora_rank":512,"qk_rope_head_dim":64,"qk_nope_head_dim":128,"v_head_dim":128,"rms_norm_eps":1e-06,"rope_theta":10000,
  "vocab_size":129280,"max_position_embeddings":163840,"num_nextn_predict_layers":1,"aux_loss_alpha":0.001,"hidden_act":"silu",
  "rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":40,"mscale":1.0,"mscale_all_dim":1.0,
                  "original_max_position_embeddings":4096,"type":"yarn"}})";
} // namespace

TEST(HfConfig, Gpt2IsCoresGpt2Small)
{
    const auto c = hf::parse_config(gpt2);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    EXPECT_EQ(c->arch, architecture::gpt2);
    EXPECT_TRUE(same_block(c->block, ml::gpt2_small));
    EXPECT_EQ(c->layers, 12u);
    EXPECT_EQ(c->vocab, 50257u);
    EXPECT_EQ(c->context, 1024u);
    EXPECT_TRUE(c->tied_embeddings);
    EXPECT_EQ(c->eos, 50256u);
}

TEST(HfConfig, Llama3IsCoresLlama3_8b)
{
    const auto c = hf::parse_config(llama3);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    EXPECT_TRUE(same_block(c->block, ml::llama3_8b));
    EXPECT_EQ(c->bos, 128000u);
    EXPECT_FALSE(c->tied_embeddings);
    EXPECT_TRUE(c->notes.empty());
}

TEST(HfConfig, Qwen3NormalisesQueriesAndKeysBeforeRope)
{
    const auto c = hf::parse_config(qwen3);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    EXPECT_EQ(c->block.qk_norm, ml::qk_norm_kind::before_rope);
    EXPECT_EQ(c->block.head_dim, 128u);   // not hidden / heads = 64
    EXPECT_EQ(c->block.window, 0u);
    EXPECT_EQ(c->block.rope, ml::rope_kind::half);
    EXPECT_EQ(c->block.rope_base, 1e6);
}

TEST(HfConfig, OlmoeIsCoresPreset)
{
    const auto c = hf::parse_config(olmoe);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    EXPECT_TRUE(same_block(c->block, ml::olmoe_1b_7b));
    EXPECT_FALSE(c->notes.empty());   // the whole-projection Q/K norm
}

TEST(HfConfig, DeepseekV3MapsLatentAttentionExpertsAndYarn)
{
    const auto c = hf::parse_config(deepseek3);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    auto expected = ml::deepseek_v3;
    expected.bias_update_rate = 0;           // not in config.json (noted)
    expected.balance_loss_weight = 0.001;    // aux_loss_alpha as configured
    expected.mtp_loss_weight = 0;            // a training choice, not in config.json
    EXPECT_TRUE(same_block(c->block, expected));
    EXPECT_EQ(c->leading_dense_layers, 3u);
    EXPECT_EQ(c->layers, 61u);
    EXPECT_EQ(c->arch, architecture::deepseek_v3);
}

TEST(HfConfig, MistralWindowAndQwen2Biases)
{
    const auto m = hf::parse_config(R"({"model_type":"mistral","hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":8,
        "num_hidden_layers":32,"intermediate_size":14336,"vocab_size":32000,"sliding_window":4096,"rope_theta":10000.0})");
    ASSERT_TRUE(m.is_ok()) << m.unwrap_err_unchecked().what();
    EXPECT_EQ(m->block.window, 4096u);
    const auto q = hf::parse_config(R"({"model_type":"qwen2","hidden_size":896,"num_attention_heads":14,"num_key_value_heads":2,
        "num_hidden_layers":24,"intermediate_size":4864,"vocab_size":151936,"use_sliding_window":false,"sliding_window":32768})");
    ASSERT_TRUE(q.is_ok()) << q.unwrap_err_unchecked().what();
    EXPECT_TRUE(q->block.bias);
    EXPECT_EQ(q->block.window, 0u);
    EXPECT_EQ(q->notes.size(), 1u);
}

TEST(HfConfig, RopeScalingMapsOrIsNoted)
{
    const std::string base = R"({"model_type":"llama","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":1,
        "intermediate_size":128,"vocab_size":10,"max_position_embeddings":4096,"rope_scaling":)";
    const auto linear = hf::parse_config(base + R"({"type":"linear","factor":4.0}})");
    ASSERT_TRUE(linear.is_ok());
    EXPECT_EQ(linear->block.rope_scaling, ml::rope_scaling_kind::linear);
    EXPECT_EQ(linear->block.rope_factor, 4.0);
    const auto llama3 = hf::parse_config(base + R"({"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0}})");
    ASSERT_TRUE(llama3.is_ok());
    EXPECT_EQ(llama3->block.rope_scaling, ml::rope_scaling_kind::none);
    ASSERT_EQ(llama3->notes.size(), 1u);
    const auto yarn = hf::parse_config(base + R"({"type":"yarn","factor":4.0,"original_max_position_embeddings":1024}})");
    ASSERT_TRUE(yarn.is_ok());
    EXPECT_EQ(yarn->block.rope_original_context, 1024u);
}

TEST(HfConfig, RefusesWhatDoesNotMap)
{
    EXPECT_EQ(hf::parse_config(R"({"model_type":"bert"})").unwrap_err_unchecked().code, errc::unsupported);
    EXPECT_EQ(hf::parse_config(R"({"hidden_size":1})").unwrap_err_unchecked().code, errc::bad_header);
    EXPECT_EQ(hf::parse_config(R"({"model_type":"llama","hidden_size":64})").unwrap_err_unchecked().code, errc::bad_header);
    EXPECT_EQ(hf::parse_config(R"({"model_type":"llama","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":1,
        "intermediate_size":128,"vocab_size":10,"hidden_act":"gelu"})").unwrap_err_unchecked().code, errc::unsupported);
    EXPECT_EQ(hf::parse_config(R"({"model_type":"llama","hidden_size":64,"num_attention_heads":3,"num_hidden_layers":1,
        "intermediate_size":128,"vocab_size":10,"num_key_value_heads":2})").unwrap_err_unchecked().code, errc::bad_header);
    EXPECT_EQ(hf::parse_config(R"({"model_type":"gpt2","n_embd":8,"n_head":2,"n_layer":1,"vocab_size":5,
        "activation_function":"relu"})").unwrap_err_unchecked().code, errc::unsupported);
    EXPECT_EQ(hf::parse_config("{").unwrap_err_unchecked().code, errc::bad_json);
}

TEST(HfConfig, GenerationConfig)
{
    const auto g = hf::parse_generation_config(R"({"bos_token_id":1,"eos_token_id":[2,3],"do_sample":true,"temperature":0.6,
        "top_p":0.9,"top_k":50,"max_new_tokens":256})");
    ASSERT_TRUE(g.is_ok());
    EXPECT_EQ(g->bos, 1u);
    EXPECT_EQ(g->eos, (std::vector<std::uint32_t>{2, 3}));
    EXPECT_EQ(g->temperature, 0.6);
    EXPECT_EQ(g->top_k, 50u);
    EXPECT_EQ(g->do_sample, true);
    EXPECT_FALSE(g->pad.has_value());
    EXPECT_EQ(hf::parse_generation_config(R"({"eos_token_id":7})")->eos, (std::vector<std::uint32_t>{7}));
    EXPECT_TRUE(hf::parse_generation_config("[]").is_err());
}
