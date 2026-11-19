//
//  convert_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/convert/convert.hpp>

#include <gtest/gtest.h>

#include <numeric>

#include "support/torch_pickle.hpp"

using namespace snicholls::formats;
using namespace snicholls::formats::convert;

namespace
{
struct sample
{
    std::vector<float> w = std::vector<float>(6);
    std::vector<std::int32_t> ids{1, 2, 3};
    std::vector<std::uint16_t> h{0x3c00, 0x4000};   // f16 1, 2
    sample() { std::iota(w.begin(), w.end(), 0.5f); }
    std::vector<tensor_view> views() const
    {
        return {{"w", dtype::f32, {2, 3}, std::as_bytes(std::span(w))},
                {"ids", dtype::i32, {3}, std::as_bytes(std::span(ids))},
                {"h", dtype::f16, {2}, std::as_bytes(std::span(h))}};
    }
};

void expect_same(const loaded& l, const std::vector<tensor_view>& want)
{
    ASSERT_EQ(l.tensors.size(), want.size());
    for (const auto& w : want) {
        const tensor_view* t = l.find(w.name);
        ASSERT_NE(t, nullptr) << w.name;
        EXPECT_EQ(t->type, w.type);
        EXPECT_EQ(t->shape, w.shape);
        EXPECT_TRUE(std::equal(t->data.begin(), t->data.end(), w.data.begin(), w.data.end())) << w.name;
    }
}
} // namespace

TEST(Convert, EveryFormatRoundTripsThroughEveryOther)
{
    const sample s;
    const auto want = s.views();
    for (const auto a : {file_format::safetensors, file_format::gguf, file_format::npz}) {
        const auto first = write_any(a, want);
        ASSERT_TRUE(first.is_ok()) << to_string(a);
        EXPECT_EQ(detect(*first), a);
        const auto la = read_any(*first);
        ASSERT_TRUE(la.is_ok()) << la.unwrap_err_unchecked().what();
        expect_same(*la, want);
        for (const auto b : {file_format::safetensors, file_format::gguf, file_format::npz}) {
            const auto second = write_any(b, la->tensors, la->metadata);
            ASSERT_TRUE(second.is_ok()) << to_string(a) << " -> " << to_string(b);
            const auto lb = read_any(*second);
            ASSERT_TRUE(lb.is_ok());
            expect_same(*lb, want);
        }
    }
}

TEST(Convert, NpyAndMetadata)
{
    const sample s;
    const auto all = s.views();
    const auto one = write_any(file_format::npy, std::span(all).first(1));
    ASSERT_TRUE(one.is_ok());
    EXPECT_EQ(detect(*one), file_format::npy);
    EXPECT_EQ(read_any(*one)->tensors[0].shape, (tensor_shape{2, 3}));
    EXPECT_EQ(write_any(file_format::npy, s.views()).unwrap_err_unchecked().code, errc::bad_argument);
    const std::vector<std::pair<std::string, std::string>> meta{{"format", "pt"}};
    const auto st = write_any(file_format::safetensors, s.views(), meta);
    EXPECT_EQ(read_any(*st)->metadata, meta);
    const auto g = write_any(file_format::gguf, s.views(), meta);
    EXPECT_EQ(read_any(*g)->metadata, meta);
    EXPECT_EQ(write_any(file_format::pytorch, s.views()).unwrap_err_unchecked().code, errc::unsupported);
}

TEST(Convert, DetectsByContentAndExtension)
{
    static_assert(format_from_extension("m.safetensors") == file_format::safetensors && format_from_extension("a.pth") == file_format::pytorch);
    static_assert(!format_from_extension("model.bin").has_value());
    EXPECT_FALSE(detect({}).has_value());
    const std::vector<float> p(llmc::parameter_count({.max_seq = 2, .vocab = 2, .layers = 1, .heads = 1, .channels = 2, .padded_vocab = 2}));
    const auto ck = llmc::write_checkpoint({.max_seq = 2, .vocab = 2, .layers = 1, .heads = 1, .channels = 2, .padded_vocab = 2}, dtype::f32,
                                           std::as_bytes(std::span(p)));
    EXPECT_EQ(detect(*ck), file_format::llmc_checkpoint);
    const auto l = read_any(*ck);
    ASSERT_TRUE(l.is_ok());
    EXPECT_EQ(l->tensors.size(), 16u);
    const std::vector<std::uint16_t> toks{1, 2};
    const auto shard = llmc::write_tokens(toks);
    EXPECT_EQ(detect(*shard), file_format::llmc_tokens);
    EXPECT_EQ(read_any(*shard)->tensors[0].shape, (tensor_shape{2}));
    const std::vector<std::byte> junk(64, std::byte{7});
    EXPECT_EQ(read_any(junk).unwrap_err_unchecked().code, errc::unsupported);
}

TEST(Convert, PytorchTensorsBecomeSafetensors)
{
    test_support::pickle_writer w;
    w.op(0x80).op(2).op('}').op('(').str("a").tensor("FloatStorage", "0", 4, 0, {2, 2}, {1, 2});   // transposed
    w.str("b").tensor("FloatStorage", "0", 4, 0, {4}, {1}).op('u').op('.');
    std::vector<float> f{1, 2, 3, 4};
    const auto b = std::as_bytes(std::span(f));
    const auto file = test_support::torch_zip(w.out, {{"0", {b.begin(), b.end()}}});
    EXPECT_EQ(detect(file), file_format::pytorch);
    const auto l = read_any(file);
    ASSERT_TRUE(l.is_ok()) << l.unwrap_err_unchecked().what();
    const auto* a = reinterpret_cast<const float*>(l->find("a")->data.data());
    EXPECT_EQ(std::vector<float>(a, a + 4), (std::vector<float>{1, 3, 2, 4}));
    EXPECT_EQ(l->notes.size(), 1u);
    EXPECT_TRUE(write_any(file_format::safetensors, l->tensors).is_ok());
}

TEST(Convert, CastAndDequantize)
{
    const std::vector<float> f{1.0f, -2.0f};
    const auto h = cast(tensor_view{"x", dtype::f32, {2}, std::as_bytes(std::span(f))}, dtype::bf16);
    ASSERT_TRUE(h.is_ok());
    EXPECT_EQ(h->data.size(), 4u);
    EXPECT_EQ(h->type, dtype::bf16);
    // a Q8_0 row quantised by core, written to GGUF, read back dequantised
    std::vector<float> row(64);
    std::iota(row.begin(), row.end(), -32.0f);
    std::vector<snicholls::ml::block_q8_0> q(2);
    snicholls::ml::quantize(row, std::span(q));
    gguf::writer w;
    w.add_tensor("q", gguf::ggml_type::q8_0, {64}, std::as_bytes(std::span(q)));
    const auto bytes = *w.write();
    const auto l = read_any(bytes);
    ASSERT_TRUE(l.is_ok()) << l.unwrap_err_unchecked().what();
    const auto* d = reinterpret_cast<const float*>(l->tensors[0].data.data());
    for (std::size_t i = 0; i < 64; ++i)
        EXPECT_NEAR(d[i], row[i], 0.13f);   // within half a step of 32/127
    EXPECT_EQ(l->notes.size(), 1u);
    EXPECT_EQ(read_any(bytes, {.dequantize = false}).unwrap_err_unchecked().code, errc::unsupported);
    std::vector<std::byte> k(144);
    gguf::writer kq;
    kq.add_tensor("k", gguf::ggml_type::q4_k, {256}, k);
    const auto kb = *kq.write();
    EXPECT_EQ(read_any(kb).unwrap_err_unchecked().code, errc::unsupported);
}
