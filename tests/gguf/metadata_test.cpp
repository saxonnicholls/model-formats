//
//  metadata_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/gguf/metadata.hpp>

#include <gtest/gtest.h>

using namespace snicholls::formats;
using namespace snicholls::formats::gguf;
namespace ml = snicholls::ml;

namespace
{
model_config round_trip(const model_config& c, bool with_output = true)
{
    writer w;
    EXPECT_TRUE(write_model_config(w, c, "test").is_ok());
    static const std::vector<float> one(1);
    if (with_output)
        w.add_tensor("output.weight", ggml_type::f32, {1}, std::as_bytes(std::span(one)));
    static std::vector<std::byte> bytes;
    bytes = *w.write();
    const auto f = parse(bytes);
    EXPECT_TRUE(f.is_ok());
    const auto back = read_model_config(*f);
    EXPECT_TRUE(back.is_ok()) << back.unwrap_err_unchecked().what();
    return *back;
}
} // namespace

TEST(GgufMetadata, LlamaRoundTripsWithInterleavedRope)
{
    model_config c;
    c.arch = architecture::llama;
    c.block = ml::llama3_8b;
    c.layers = 32;
    c.vocab = 128256;
    c.context = 8192;
    c.bos = 128000;
    const auto b = round_trip(c);
    auto expected = ml::llama3_8b;
    expected.rope = ml::rope_kind::interleaved;   // GGUF's Q/K are permuted for adjacent pairs
    EXPECT_TRUE(same_block(b.block, expected));
    EXPECT_EQ(b.layers, 32u);
    EXPECT_EQ(b.vocab, 128256u);
    EXPECT_EQ(b.context, 8192u);
    EXPECT_EQ(b.bos, 128000u);
    EXPECT_FALSE(b.tied_embeddings);
    EXPECT_TRUE(round_trip(c, false).tied_embeddings);
}

TEST(GgufMetadata, Gpt2RoundTrips)
{
    model_config c;
    c.arch = architecture::gpt2;
    c.block = ml::gpt2_small;
    c.layers = 12;
    c.vocab = 50257;
    c.context = 1024;
    EXPECT_TRUE(same_block(round_trip(c).block, ml::gpt2_small));
}

TEST(GgufMetadata, Qwen3AndOlmoeRoundTrip)
{
    model_config q;
    q.arch = architecture::qwen3;
    q.block = ml::llama3_8b;
    q.block.qk_norm = ml::qk_norm_kind::before_rope;
    q.layers = 2;
    EXPECT_TRUE(same_block(round_trip(q).block, q.block));
    model_config o;
    o.arch = architecture::olmoe;
    o.block = ml::olmoe_1b_7b;
    o.layers = 16;
    auto expected = ml::olmoe_1b_7b;
    expected.balance_loss_weight = 0;   // training-only: not in GGUF
    expected.z_loss_weight = 0;
    const auto back = round_trip(o);
    EXPECT_TRUE(same_block(back.block, expected));
    EXPECT_FALSE(back.notes.empty());
}

TEST(GgufMetadata, DeepseekV3RoundTrips)
{
    model_config c;
    c.arch = architecture::deepseek_v3;
    c.block = ml::deepseek_v3;
    c.layers = 61;
    c.leading_dense_layers = 3;
    c.context = 163840;
    const auto back = round_trip(c);
    auto expected = ml::deepseek_v3;
    expected.bias_update_rate = 0;
    expected.balance_loss_weight = 0;
    expected.mtp_loss_weight = 0;
    EXPECT_TRUE(same_block(back.block, expected));
    EXPECT_EQ(back.arch, architecture::deepseek_v3);
    EXPECT_EQ(back.layers, 61u);
    EXPECT_EQ(back.leading_dense_layers, 3u);
}

TEST(GgufMetadata, RefusesWhatDoesNotMap)
{
    writer w;
    w.set(general::architecture, value_string("rwkv6"));
    const auto bytes = *w.write();
    EXPECT_EQ(read_model_config(*parse(bytes)).unwrap_err_unchecked().code, errc::unsupported);
    writer m;
    m.set(general::architecture, value_string("llama"));
    const auto missing = *m.write();
    EXPECT_EQ(read_model_config(*parse(missing)).unwrap_err_unchecked().code, errc::bad_header);
    model_config vit;
    vit.block = ml::vit_base;
    writer v;
    EXPECT_EQ(write_model_config(v, vit, "vit").unwrap_err_unchecked().code, errc::unsupported);
}

TEST(GgufMetadata, VocabularyRoundTrips)
{
    vocabulary v;
    v.pre = "gpt-2";
    v.tokens = {"a", "b", "ab", "<|endoftext|>"};
    v.types = {token_type::normal, token_type::normal, token_type::normal, token_type::control};
    v.merges = {"a b"};
    v.bos = 3;
    v.eos = 3;
    writer w;
    write_vocabulary(w, v, "{{ messages }}");
    const auto bytes = *w.write();
    const auto f = parse(bytes);
    const auto back = read_vocabulary(*f);
    ASSERT_TRUE(back.is_ok()) << back.unwrap_err_unchecked().what();
    EXPECT_EQ(back->kind, tokenizer_kind::byte_level_bpe);
    EXPECT_EQ(back->pre, "gpt-2");
    EXPECT_EQ(back->tokens, v.tokens);
    EXPECT_EQ(back->types, v.types);
    EXPECT_EQ(back->merges, v.merges);
    EXPECT_EQ(back->eos, 3u);
    EXPECT_EQ(f->find(tokenizer::chat_template)->as_string(), "{{ messages }}");
    vocabulary sp;
    sp.kind = tokenizer_kind::sentencepiece;
    sp.tokens = {"<unk>", "\xe2\x96\x81the"};
    sp.scores = {0.0f, -1.0f};
    writer s;
    write_vocabulary(s, sp);
    const auto sb = *s.write();
    const auto sback = read_vocabulary(*parse(sb));
    EXPECT_EQ(sback->kind, tokenizer_kind::sentencepiece);
    EXPECT_EQ(sback->scores, sp.scores);
    writer none;
    const auto nb = *none.write();
    EXPECT_EQ(read_vocabulary(*parse(nb)).unwrap_err_unchecked().code, errc::not_found);
}
