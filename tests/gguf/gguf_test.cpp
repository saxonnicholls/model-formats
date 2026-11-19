//
//  gguf_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/gguf/gguf.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace snicholls::formats;
using namespace snicholls::formats::gguf;

namespace
{
std::vector<std::byte> sample()
{
    static const std::vector<float> w(2 * 64, 0.25f);
    static const std::vector<std::byte> q(2 * 34, std::byte{1});   // two Q8_0 blocks
    writer g;
    g.set("general.architecture", value_string("llama"));
    g.set("u8", value_u8(200));
    g.set("i8", value_i8(-100));
    g.set("u16", value_u16(60000));
    g.set("i16", value_i16(-30000));
    g.set("u32", value_u32(4000000000u));
    g.set("i32", value_i32(-2000000000));
    g.set("f32", value_f32(0.5f));
    g.set("b", value_bool(true));
    g.set("u64", value_u64(std::numeric_limits<std::uint64_t>::max()));
    g.set("i64", value_i64(std::numeric_limits<std::int64_t>::min()));
    g.set("f64", value_f64(1e-300));
    const std::vector<std::string> toks{"a", "", "\xc4\xa0hello"};
    g.set("strings", value_strings(toks));
    g.set("ints", value_array(value_type::i32, {value_i32(1), value_i32(-2), value_i32(3)}));
    g.set("nested", value_array(value_type::array, {value_array(value_type::u8, {value_u8(1)}), value_strings(toks)}));
    g.set("empty", value_array(value_type::f32, {}));
    g.add_tensor("w", ggml_type::f32, {2, 64}, std::as_bytes(std::span(w)));
    g.add_tensor("q", ggml_type::q8_0, {64}, q);
    return *g.write();
}

std::vector<std::byte> header(std::uint32_t version, std::uint64_t tensors, std::uint64_t kv)
{
    std::vector<std::byte> b(24);
    snicholls::store_le(b.data(), magic);
    snicholls::store_le(b.data() + 4, version);
    snicholls::store_le(b.data() + 8, tensors);
    snicholls::store_le(b.data() + 16, kv);
    return b;
}
errc code_of(std::span<const std::byte> b) { return parse(b).unwrap_err_unchecked().code; }
} // namespace

TEST(GgufParse, EveryValueTypeRoundTrips)
{
    const auto b = sample();
    const auto f = parse(b);
    ASSERT_TRUE(f.is_ok()) << f.unwrap_err_unchecked().what();
    EXPECT_EQ(f->version, 3u);
    EXPECT_EQ(f->architecture(), "llama");
    EXPECT_EQ(f->find("u8")->as_u64(), 200u);
    EXPECT_EQ(f->find("i8")->as_i64(), -100);
    EXPECT_EQ(f->find("u16")->as_u64(), 60000u);
    EXPECT_EQ(f->find("i16")->as_i64(), -30000);
    EXPECT_EQ(f->find("u32")->as_u64(), 4000000000u);
    EXPECT_EQ(f->find("i32")->as_i64(), -2000000000);
    EXPECT_EQ(f->find("f32")->as_f64(), 0.5);
    EXPECT_EQ(f->find("b")->as_bool(), true);
    EXPECT_EQ(f->find("u64")->as_u64(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_FALSE(f->find("u64")->as_i64().has_value());
    EXPECT_EQ(f->find("i64")->as_i64(), std::numeric_limits<std::int64_t>::min());
    EXPECT_FALSE(f->find("i64")->as_u64().has_value());
    EXPECT_EQ(f->find("f64")->as_f64(), 1e-300);
    EXPECT_EQ(*f->find("strings")->strings(), (std::vector<std::string_view>{"a", "", "\xc4\xa0hello"}));
    EXPECT_EQ(*f->find("ints")->integers(), (std::vector<std::int64_t>{1, -2, 3}));
    EXPECT_EQ(*f->find("ints")->numbers(), (std::vector<double>{1, -2, 3}));
    EXPECT_FALSE(f->find("ints")->strings().has_value());
    EXPECT_FALSE(f->find("strings")->integers().has_value());
    EXPECT_EQ(f->find("ints")->element(1)->as_i64(), -2);
    EXPECT_FALSE(f->find("ints")->element(3).has_value());
    EXPECT_EQ(f->find("nested")->count, 2u);
    EXPECT_EQ(f->find("empty")->count, 0u);
    EXPECT_EQ(f->find("missing"), nullptr);
    // the nested array survives a copy into owned values
    const metadata_value n = to_owned(*f->find("nested"));
    ASSERT_EQ(n.elements.size(), 2u);
    EXPECT_EQ(n.elements[1].elements[2].s, "\xc4\xa0hello");
    EXPECT_EQ(n.elements[0].elements[0].u, 1u);
}

TEST(GgufParse, TensorsAreAlignedZeroCopyViews)
{
    const auto b = sample();
    const auto f = parse(b);
    ASSERT_TRUE(f.is_ok());
    ASSERT_EQ(f->tensors.size(), 2u);
    const auto* w = f->find_tensor("w");
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->shape, (tensor_shape{2, 64}));
    EXPECT_EQ(w->ne(), (tensor_shape{64, 2}));
    EXPECT_EQ(f->data_offset % 32, 0u);
    EXPECT_GE(w->data.data(), b.data());
    EXPECT_EQ(reinterpret_cast<const float*>(w->data.data())[127], 0.25f);
    const auto* q = f->find_tensor("q");
    EXPECT_EQ(q->offset % 32, 0u);
    EXPECT_EQ(q->data.size(), 68u);
    EXPECT_FALSE(q->dense().has_value());
    EXPECT_EQ(w->dense()->type, dtype::f32);
    EXPECT_EQ(f->find_tensor("nope"), nullptr);
}

TEST(GgufParse, WriteIsDeterministicAndRewritesIdentically)
{
    const auto a = sample(), b = sample();
    EXPECT_EQ(a, b);
    const auto f = parse(a);
    writer w;
    for (const auto& e : f->metadata)
        w.set(e.key, to_owned(e.val));
    for (const auto& t : f->tensors)
        w.add_tensor(t.name, t.type, t.shape, t.data);
    EXPECT_EQ(*w.write(), a);
}

TEST(GgufParse, CustomAlignment)
{
    const std::vector<std::uint8_t> one{1};
    writer w;
    w.set("general.alignment", value_u32(64));
    w.add_tensor("a", ggml_type::i8, {1}, std::as_bytes(std::span(one)));
    w.add_tensor("b", ggml_type::i8, {1}, std::as_bytes(std::span(one)));
    const auto bytes = *w.write();
    const auto f = parse(bytes);
    ASSERT_TRUE(f.is_ok()) << f.unwrap_err_unchecked().what();
    EXPECT_EQ(f->alignment, 64u);
    EXPECT_EQ(f->find_tensor("b")->offset, 64u);
    w.set("general.alignment", value_u32(48));
    EXPECT_EQ(w.write().unwrap_err_unchecked().code, errc::bad_argument);
}

TEST(GgufParse, RefusesBadHeaders)
{
    EXPECT_EQ(code_of({}), errc::truncated);
    auto h = header(3, 0, 0);
    h[0] = std::byte{'X'};
    EXPECT_EQ(code_of(h), errc::bad_magic);
    EXPECT_EQ(code_of(header(1, 0, 0)), errc::unsupported_version);
    EXPECT_EQ(code_of(header(0x03000000, 0, 0)), errc::unsupported_version);   // big-endian
    EXPECT_EQ(code_of(header(3, 0, std::numeric_limits<std::uint64_t>::max())), errc::too_large);
    EXPECT_EQ(code_of(header(3, 0, 1000)), errc::truncated);
    EXPECT_TRUE(parse(header(3, 0, 0)).is_ok());
    EXPECT_TRUE(parse(header(2, 0, 0)).is_ok());
}

TEST(GgufParse, RefusesBadValues)
{
    auto b = header(3, 0, 1);
    const auto key = [&](std::string_view k) {
        std::vector<std::byte> e(8 + k.size());
        snicholls::store_le(e.data(), static_cast<std::uint64_t>(k.size()));
        std::memcpy(e.data() + 8, k.data(), k.size());
        return e;
    };
    const auto append = [](std::vector<std::byte>& to, std::span<const std::byte> s) { to.insert(to.end(), s.begin(), s.end()); };
    const auto u32 = [](std::uint32_t v) { std::vector<std::byte> e(4); snicholls::store_le(e.data(), v); return e; };
    const auto u64 = [](std::uint64_t v) { std::vector<std::byte> e(8); snicholls::store_le(e.data(), v); return e; };
    append(b, key("k"));
    auto bad_type = b;
    append(bad_type, u32(13));
    append(bad_type, u32(0));
    EXPECT_EQ(code_of(bad_type), errc::bad_header);
    auto long_string = b;
    append(long_string, u32(8));
    append(long_string, u64(1u << 30));
    EXPECT_EQ(code_of(long_string), errc::truncated);
    auto huge_array = b;
    append(huge_array, u32(9));
    append(huge_array, u32(4));
    append(huge_array, u64(std::numeric_limits<std::uint64_t>::max()));
    EXPECT_EQ(code_of(huge_array), errc::truncated);
    auto huge_strings = b;
    append(huge_strings, u32(9));
    append(huge_strings, u32(8));
    append(huge_strings, u64(std::uint64_t{1} << 40));
    EXPECT_EQ(code_of(huge_strings), errc::truncated);
    auto deep = b;   // arrays of arrays, deeper than 8
    append(deep, u32(9));
    for (int i = 0; i < 10; ++i) {
        append(deep, u32(9));
        append(deep, u64(1));
    }
    EXPECT_EQ(code_of(deep), errc::bad_header);
    auto bad_alignment = header(3, 0, 1);
    append(bad_alignment, key("general.alignment"));
    append(bad_alignment, u32(4));
    append(bad_alignment, u32(3));
    EXPECT_EQ(code_of(bad_alignment), errc::bad_header);
    auto dup = header(3, 0, 2);
    for (int i = 0; i < 2; ++i) {
        append(dup, key("k"));
        append(dup, u32(4));
        append(dup, u32(1));
    }
    EXPECT_EQ(code_of(dup), errc::duplicate_name);
}

TEST(GgufParse, RefusesBadTensors)
{
    const auto tensor_file = [](std::uint32_t dims, std::uint64_t ne0, std::uint32_t type, std::uint64_t offset, std::size_t data,
                                int count = 1, bool same_name = false) {
        auto b = header(3, static_cast<std::uint64_t>(count), 0);
        for (int i = 0; i < count; ++i) {
            const std::string name = same_name ? "t" : "t" + std::to_string(i);
            std::vector<std::byte> e(8 + name.size() + 4 + 8 * dims + 4 + 8);
            std::size_t at = 0;
            snicholls::store_le(e.data(), static_cast<std::uint64_t>(name.size()));
            std::memcpy(e.data() + 8, name.data(), name.size());
            at = 8 + name.size();
            snicholls::store_le(e.data() + at, dims);
            at += 4;
            for (std::uint32_t d = 0; d < dims; ++d, at += 8)
                snicholls::store_le(e.data() + at, d == 0 ? ne0 : std::uint64_t{1});
            snicholls::store_le(e.data() + at, type);
            snicholls::store_le(e.data() + at + 4, offset * static_cast<std::uint64_t>(i + 1));
            b.insert(b.end(), e.begin(), e.end());
        }
        b.resize((b.size() + 31) / 32 * 32 + data);
        return b;
    };
    EXPECT_TRUE(parse(tensor_file(1, 8, 0, 0, 32)).is_ok());
    EXPECT_EQ(code_of(tensor_file(5, 8, 0, 0, 32)), errc::bad_shape);
    EXPECT_EQ(code_of(tensor_file(1, 8, 4, 0, 32)), errc::bad_dtype);   // Q4_2, removed
    EXPECT_EQ(code_of(tensor_file(1, 33, 8, 0, 64)), errc::bad_shape);  // not whole Q8_0 blocks
    EXPECT_EQ(code_of(tensor_file(1, 8, 0, 4, 64)), errc::misaligned);
    EXPECT_EQ(code_of(tensor_file(1, 8, 0, 0, 16)), errc::out_of_bounds);
    EXPECT_EQ(code_of(tensor_file(1, 16, 0, 0, 128, 2)), errc::overlap);   // both at 0: offset * (i + 1) = 0
    EXPECT_EQ(code_of(tensor_file(1, 8, 0, 32, 96, 2, true)), errc::duplicate_name);
    EXPECT_EQ(code_of(tensor_file(2, std::numeric_limits<std::uint64_t>::max(), 0, 0, 32)), errc::bad_shape);
}

TEST(GgufWrite, RefusesBadInput)
{
    const std::vector<float> f(4);
    writer w;
    w.add_tensor("a", ggml_type::f32, {3}, std::as_bytes(std::span(f)));
    EXPECT_EQ(w.write().unwrap_err_unchecked().code, errc::bad_shape);
    writer d;
    d.add_tensor("a", ggml_type::f32, {4}, std::as_bytes(std::span(f)));
    d.add_tensor("a", ggml_type::f32, {4}, std::as_bytes(std::span(f)));
    EXPECT_EQ(d.write().unwrap_err_unchecked().code, errc::duplicate_name);
    writer five;
    five.add_tensor("a", ggml_type::f32, {1, 1, 1, 1, 4}, std::as_bytes(std::span(f)));
    EXPECT_EQ(five.write().unwrap_err_unchecked().code, errc::bad_shape);
    writer mixed;
    mixed.set("m", value_array(value_type::u8, {value_u8(1), value_string("x")}));
    EXPECT_EQ(mixed.write().unwrap_err_unchecked().code, errc::bad_argument);
    writer dense;
    const std::vector<std::uint32_t> u(1);
    EXPECT_EQ(dense.add_tensor(tensor_view{"u", dtype::u32, {1}, std::as_bytes(std::span(u))}).unwrap_err_unchecked().code, errc::bad_dtype);
    EXPECT_TRUE(dense.add_tensor(tensor_view{"f", dtype::f32, {4}, std::as_bytes(std::span(f))}).is_ok());
    EXPECT_EQ(dense.tensor_count(), 1u);
}

TEST(GgufWrite, SetReplacesInPlace)
{
    writer w;
    w.set("a", value_u32(1));
    w.set("b", value_u32(2));
    w.set("a", value_string("x"));
    const auto bytes = *w.write();
    const auto f = parse(bytes);
    ASSERT_EQ(f->metadata.size(), 2u);
    EXPECT_EQ(f->metadata[0].key, "a");
    EXPECT_EQ(f->metadata[0].val.as_string(), "x");
    EXPECT_EQ(w.get("b")->u, 2u);
    EXPECT_EQ(w.get("c"), nullptr);
    EXPECT_EQ(to_string(value_type::f64), "FLOAT64");
}
