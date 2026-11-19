//
//  checkpoint_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/pytorch/checkpoint.hpp>

#include <gtest/gtest.h>

#include <numeric>

#include "support/torch_pickle.hpp"

using namespace snicholls::formats;
using test_support::pickle_writer;
using test_support::torch_zip;

namespace
{
std::vector<std::byte> floats(std::size_t n)
{
    std::vector<float> f(n);
    std::iota(f.begin(), f.end(), 0.0f);
    const auto b = std::as_bytes(std::span(f));
    return {b.begin(), b.end()};
}
} // namespace

TEST(PytorchCheckpoint, ReadsTensorsAsViews)
{
    pickle_writer w;
    w.op(0x80).op(2).op('}').op('(');
    w.str("layer.weight").tensor("FloatStorage", "0", 6, 0, {2, 3}, {3, 1});
    w.str("layer.bias").tensor("FloatStorage", "1", 2, 0, {2}, {1});
    w.str("tied").tensor("FloatStorage", "0", 6, 3, {3}, {1});   // the second row of storage 0
    w.str("epoch").integer(3);
    w.op('u').op('.');
    const auto file = torch_zip(w.out, {{"0", floats(6)}, {"1", floats(2)}});
    const auto c = pytorch::parse(file);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    EXPECT_EQ(c->prefix, "archive");
    ASSERT_EQ(c->tensors.size(), 3u);
    EXPECT_EQ(c->skipped, (std::vector<std::string>{"epoch"}));
    const auto* t = c->find("layer.weight");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape, (tensor_shape{2, 3}));
    EXPECT_TRUE(t->contiguous);
    EXPECT_EQ(reinterpret_cast<const float*>(t->view().data.data())[5], 5.0f);
    const auto* tied = c->find("tied");
    EXPECT_EQ(reinterpret_cast<const float*>(tied->data.data())[0], 3.0f);
    EXPECT_EQ(tied->storage.data(), t->storage.data());   // one storage, two tensors
    EXPECT_EQ(c->find("nope"), nullptr);
}

TEST(PytorchCheckpoint, NonContiguousTensorsMaterialize)
{
    pickle_writer w;   // a [3, 2] transposed view of a [2, 3] storage: stride (1, 3)
    w.op(0x80).op(2).op('}').str("t").tensor("FloatStorage", "0", 6, 0, {3, 2}, {1, 3}).op('s').op('.');
    const auto file = torch_zip(w.out, {{"0", floats(6)}});
    const auto c = pytorch::parse(file);
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    const auto& t = c->tensors[0];
    EXPECT_FALSE(t.contiguous);
    EXPECT_TRUE(t.data.empty());
    const tensor m = t.materialize();
    const auto* f = reinterpret_cast<const float*>(m.data.data());
    EXPECT_EQ(std::vector<float>(f, f + 6), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}

TEST(PytorchCheckpoint, NestedDictsFlattenAndOtherPrefixes)
{
    pickle_writer w;
    w.op(0x80).op(2).op('}').str("model").op('}').str("w").tensor("HalfStorage", "k", 4, 0, {4}, {1}).op('s').op('s').op('.');
    std::vector<std::byte> half(8);
    const auto c = pytorch::parse(torch_zip(w.out, {{"k", half}}, "pytorch_model"));
    ASSERT_TRUE(c.is_ok()) << c.unwrap_err_unchecked().what();
    EXPECT_EQ(c->tensors[0].name, "model.w");
    EXPECT_EQ(c->tensors[0].type, dtype::f16);
    EXPECT_EQ(c->prefix, "pytorch_model");
}

TEST(PytorchCheckpoint, RefusesBadArchives)
{
    const auto code = [](const std::vector<std::byte>& b) { return pytorch::parse(b).unwrap_err_unchecked().code; };
    EXPECT_EQ(code(std::vector<std::byte>{std::byte{0x80}, std::byte{0x02}}), errc::unsupported);   // legacy
    EXPECT_EQ(code(std::vector<std::byte>(30)), errc::bad_magic);
    pickle_writer w;
    w.op(0x80).op(2).op('}').str("w").tensor("FloatStorage", "0", 6, 0, {2, 3}, {3, 1}).op('s').op('.');
    EXPECT_EQ(code(torch_zip(w.out, {})), errc::not_found);                         // storage missing
    EXPECT_EQ(code(torch_zip(w.out, {{"0", floats(5)}})), errc::out_of_bounds);     // storage shorter than numel
    pickle_writer far;
    far.op(0x80).op(2).op('}').str("w").tensor("FloatStorage", "0", 6, 4, {2, 3}, {3, 1}).op('s').op('.');
    EXPECT_EQ(code(torch_zip(far.out, {{"0", floats(6)}})), errc::out_of_bounds);   // offset + extent past numel
    pickle_writer odd;
    odd.op(0x80).op(2).op('}').str("w").tensor("ComplexFloatStorage", "0", 1, 0, {1}, {1}).op('s').op('.');
    EXPECT_EQ(code(torch_zip(odd.out, {{"0", floats(2)}})), errc::unsafe_pickle);   // not an allowed global at all
    pickle_writer list;
    list.op(0x80).op(2).op(']').op('.');
    EXPECT_EQ(code(torch_zip(list.out, {})), errc::bad_header);
    pickle_writer attack;
    attack.op(0x80).op(2).global("os", "system").str("x").op(0x85).op('R').op('.');
    EXPECT_EQ(code(torch_zip(attack.out, {})), errc::unsafe_pickle);
    static_assert(pytorch::storage_dtype("BFloat16Storage") == dtype::bf16 && !pytorch::storage_dtype("QInt8Storage").has_value());
}
