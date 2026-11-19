//
//  dtype_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/dtype.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace snicholls::formats;

namespace
{
template <typename T>
std::vector<std::byte> bytes_of(std::initializer_list<T> v)
{
    std::vector<std::byte> b(v.size() * sizeof(T));
    std::memcpy(b.data(), std::data(v), b.size());
    return b;
}
template <typename T>
std::vector<T> values_of(const std::vector<std::byte>& b)
{
    std::vector<T> v(b.size() / sizeof(T));
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}
} // namespace

TEST(FormatsDtype, SizesAndNamesRoundTrip)
{
    static_assert(size_of(dtype::bf16) == 2 && size_of(dtype::f8_e4m3) == 1 && size_of(dtype::u64) == 8);
    static_assert(dtype_from_safetensors("F8_E5M2") == dtype::f8_e5m2);
    static_assert(!dtype_from_safetensors("F4").has_value());
    for (const dtype t : all_dtypes) {
        EXPECT_EQ(dtype_from_safetensors(safetensors_name(t)), t);
        EXPECT_GT(size_of(t), 0u);
    }
    static_assert(is_float(dtype::f8_e5m2) && !is_float(dtype::i64) && is_signed_integer(dtype::i8) && !is_signed_integer(dtype::u8));
}

TEST(FormatsDtype, Bf16ToF32IsExact)
{
    // 1.0, -2.5, the bf16 encodings 0x3f80 and 0xc020
    const auto in = bytes_of<std::uint16_t>({0x3f80, 0xc020});
    std::vector<std::byte> out(8);
    ASSERT_TRUE(convert_elements(dtype::bf16, in, dtype::f32, out).is_ok());
    EXPECT_EQ(values_of<float>(out), (std::vector<float>{1.0f, -2.5f}));
}

TEST(FormatsDtype, F32ToF16RoundsToNearestEven)
{
    const auto in = bytes_of<float>({1.0f, 65504.0f, 1.0f + 1.0f / 2048});   // the last is a tie between two halves
    std::vector<std::byte> out(6);
    ASSERT_TRUE(convert_elements(dtype::f32, in, dtype::f16, out).is_ok());
    EXPECT_EQ(values_of<std::uint16_t>(out), (std::vector<std::uint16_t>{0x3c00, 0x7bff, 0x3c00}));
}

TEST(FormatsDtype, EightBitFloatsDecode)
{
    // E4M3: 0x38 = 1.0, 0x7e = 448 (the largest); E5M2: 0x3c = 1.0
    const auto e4 = bytes_of<std::uint8_t>({0x38, 0x7e});
    std::vector<std::byte> out(8);
    ASSERT_TRUE(convert_elements(dtype::f8_e4m3, e4, dtype::f32, out).is_ok());
    EXPECT_EQ(values_of<float>(out), (std::vector<float>{1.0f, 448.0f}));
    const auto e5 = bytes_of<std::uint8_t>({0x3c});
    std::vector<std::byte> one(4);
    ASSERT_TRUE(convert_elements(dtype::f8_e5m2, e5, dtype::f32, one).is_ok());
    EXPECT_EQ(values_of<float>(one), (std::vector<float>{1.0f}));
}

TEST(FormatsDtype, IntegersConvertWhenTheyFit)
{
    const auto in = bytes_of<std::int64_t>({-128, 127, 0});
    std::vector<std::byte> out(3);
    ASSERT_TRUE(convert_elements(dtype::i64, in, dtype::i8, out).is_ok());
    EXPECT_EQ(values_of<std::int8_t>(out), (std::vector<std::int8_t>{-128, 127, 0}));
    const auto too_big = bytes_of<std::int64_t>({128});
    std::vector<std::byte> one(1, std::byte{0x55});
    const auto r = convert_elements(dtype::i64, too_big, dtype::i8, one);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err_unchecked().code, errc::out_of_bounds);
    EXPECT_EQ(one[0], std::byte{0x55});   // untouched on failure
    const auto neg = bytes_of<std::int32_t>({-1});
    EXPECT_TRUE(convert_elements(dtype::i32, neg, dtype::u32, std::span<std::byte>(out).first(1)).is_err());
}

TEST(FormatsDtype, ExtremeIntegersAndBool)
{
    const auto in = bytes_of<std::int64_t>({std::numeric_limits<std::int64_t>::min()});
    std::vector<std::byte> out(8);
    ASSERT_TRUE(convert_elements(dtype::i64, in, dtype::i64, out).is_ok());
    EXPECT_EQ(values_of<std::int64_t>(out)[0], std::numeric_limits<std::int64_t>::min());
    const auto u = bytes_of<std::uint64_t>({std::numeric_limits<std::uint64_t>::max()});
    EXPECT_TRUE(convert_elements(dtype::u64, u, dtype::i64, out).is_err());
    const auto two = bytes_of<std::uint8_t>({2});
    std::vector<std::byte> b(1);
    EXPECT_TRUE(convert_elements(dtype::u8, two, dtype::boolean, b).is_err());
    const auto yes = bytes_of<std::uint8_t>({1});
    std::vector<std::byte> f(4);
    ASSERT_TRUE(convert_elements(dtype::boolean, yes, dtype::f32, f).is_ok());
    EXPECT_EQ(values_of<float>(f)[0], 1.0f);
}

TEST(FormatsDtype, FloatToIntegerIsRefused)
{
    const auto in = bytes_of<float>({1.0f});
    std::vector<std::byte> out(4);
    const auto r = convert_elements(dtype::f32, in, dtype::i32, out);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err_unchecked().code, errc::unsupported);
}

TEST(FormatsDtype, MismatchedCountsAreBadShape)
{
    std::vector<std::byte> in(8), out(2);
    const auto r = convert_elements(dtype::f32, in, dtype::f16, out);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err_unchecked().code, errc::bad_shape);
    std::vector<std::byte> odd(3);
    EXPECT_TRUE(convert_elements(dtype::f16, odd, dtype::f32, std::span<std::byte>(out)).is_err());
}

TEST(FormatsDtype, EmptyConvertsToEmpty)
{
    EXPECT_TRUE(convert_elements(dtype::f32, {}, dtype::bf16, {}).is_ok());
}

TEST(FormatsDtype, FastPathsAgreeWithTheGeneralPath)
{
    // every bf16 and every 8-bit float to f32, and f32 to bf16, against core's to_double / to_minifloat
    std::vector<std::uint16_t> all(65536);
    for (std::size_t i = 0; i < all.size(); ++i)
        all[i] = static_cast<std::uint16_t>(i);
    std::vector<float> f(65536);
    ASSERT_TRUE(convert_elements(dtype::bf16, std::as_bytes(std::span(all)), dtype::f32, std::as_writable_bytes(std::span(f))).is_ok());
    for (std::size_t i = 0; i < all.size(); ++i) {
        const double want = snicholls::ml::to_double(snicholls::ml::bfloat16::from_bits(all[i]));
        if (want != want)
            EXPECT_NE(f[i], f[i]);
        else
            EXPECT_EQ(f[i], static_cast<float>(want)) << i;
    }
    for (const dtype t : {dtype::f8_e4m3, dtype::f8_e5m2}) {
        std::vector<std::uint8_t> b(256);
        for (std::size_t i = 0; i < 256; ++i)
            b[i] = static_cast<std::uint8_t>(i);
        std::vector<float> g(256);
        ASSERT_TRUE(convert_elements(t, std::as_bytes(std::span(b)), dtype::f32, std::as_writable_bytes(std::span(g))).is_ok());
        for (std::size_t i = 0; i < 256; ++i) {
            const double want = t == dtype::f8_e4m3 ? snicholls::ml::to_double(snicholls::ml::float8_e4m3::from_bits(static_cast<std::uint32_t>(i)))
                                                    : snicholls::ml::to_double(snicholls::ml::float8_e5m2::from_bits(static_cast<std::uint32_t>(i)));
            if (want == want)
                EXPECT_EQ(g[i], static_cast<float>(want));
        }
    }
    std::vector<std::uint16_t> back(65536);
    ASSERT_TRUE(convert_elements(dtype::f32, std::as_bytes(std::span(f)), dtype::bf16, std::as_writable_bytes(std::span(back))).is_ok());
    for (std::size_t i = 0; i < all.size(); ++i)
        if (f[i] == f[i])
            EXPECT_EQ(back[i], snicholls::ml::to_minifloat<snicholls::ml::bfloat16>(static_cast<double>(f[i])).bits) << i;
}
