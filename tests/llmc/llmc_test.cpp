//
//  llmc_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/llmc/llmc.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using namespace snicholls::formats;
namespace lc = snicholls::formats::llmc;

namespace
{
constexpr lc::gpt2_shape tiny{.max_seq = 8, .vocab = 10, .layers = 2, .heads = 2, .channels = 4, .padded_vocab = 16};
}

TEST(Llmc, ParameterCountIsGpt2s)
{
    // GPT-2 small, padded: 124,475,904 parameters (llm.c prints this)
    static_assert(lc::parameter_count(lc::gpt2_shape{}) == 124475904u);
    static_assert(lc::tensor_shapes(tiny)[4] == tensor_shape{2, 12, 4});
}

TEST(Llmc, CheckpointRoundTrips)
{
    std::vector<float> p(lc::parameter_count(tiny));
    std::iota(p.begin(), p.end(), 0.0f);
    const auto w = lc::write_checkpoint(tiny, dtype::f32, std::as_bytes(std::span(p)));
    ASSERT_TRUE(w.is_ok());
    const auto c = lc::parse_checkpoint(*w);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    EXPECT_EQ(c->shape.padded_vocab, 16u);
    EXPECT_EQ(c->type, dtype::f32);
    EXPECT_EQ(c->tensors[0].name, "wte");
    EXPECT_EQ(c->tensors[15].shape, (tensor_shape{4}));
    EXPECT_EQ(reinterpret_cast<const float*>(c->tensors[1].data.data())[0], 64.0f);   // wpe follows wte's 16 * 4
    std::size_t total = 0;
    for (const auto& t : c->tensors)
        total += t.data.size();
    EXPECT_EQ(total, p.size() * 4);
    std::vector<std::uint16_t> bf(p.size());
    const auto b = lc::write_checkpoint(tiny, dtype::bf16, std::as_bytes(std::span(bf)));
    EXPECT_EQ(lc::parse_checkpoint(*b)->type, dtype::bf16);
}

TEST(Llmc, CheckpointRefusesBadHeaders)
{
    std::vector<float> p(lc::parameter_count(tiny));
    auto w = *lc::write_checkpoint(tiny, dtype::f32, std::as_bytes(std::span(p)));
    const auto code = [](const std::vector<std::byte>& b) { return lc::parse_checkpoint(b).unwrap_err_unchecked().code; };
    EXPECT_EQ(code(std::vector<std::byte>(100)), errc::truncated);
    auto magic = w;
    magic[0] = std::byte{0};
    EXPECT_EQ(code(magic), errc::bad_magic);
    auto version = w;
    version[4] = std::byte{4};
    EXPECT_EQ(code(version), errc::unsupported_version);
    auto heads = w;
    heads[20] = std::byte{3};   // 3 heads do not divide 4 channels
    EXPECT_EQ(code(heads), errc::bad_header);
    w.pop_back();
    EXPECT_EQ(code(w), errc::bad_shape);
    EXPECT_EQ(lc::write_checkpoint(tiny, dtype::f16, {}).unwrap_err_unchecked().code, errc::bad_dtype);
    EXPECT_EQ(lc::write_checkpoint(tiny, dtype::f32, {}).unwrap_err_unchecked().code, errc::bad_shape);
}

TEST(Llmc, TokenShardsRoundTrip)
{
    const std::vector<std::uint16_t> t{50256, 1, 2, 65535};
    const auto w = lc::write_tokens(t);
    const auto r = lc::parse_tokens(*w);
    ASSERT_TRUE(r.is_ok());
    ASSERT_EQ(r->size(), 8u);
    EXPECT_EQ(snicholls::load_le<std::uint16_t>(r->data() + 6), 65535);
    auto bad = *w;
    bad[8] = std::byte{9};
    EXPECT_EQ(lc::parse_tokens(bad).unwrap_err_unchecked().code, errc::bad_shape);
    bad[4] = std::byte{2};
    EXPECT_EQ(lc::parse_tokens(bad).unwrap_err_unchecked().code, errc::unsupported_version);
    EXPECT_TRUE(lc::parse_tokens(*lc::write_tokens({})).is_ok());
}
