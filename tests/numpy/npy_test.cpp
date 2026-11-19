//
//  npy_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/numpy/npy.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

using namespace snicholls::formats;
namespace np = snicholls::formats::numpy;

namespace
{
std::vector<std::byte> npy_of(std::string dict, std::size_t data, int major = 1)
{
    const std::size_t prefix = major == 1 ? 10 : 12;
    while ((prefix + dict.size() + 1) % 64)
        dict += ' ';
    dict += '\n';
    std::vector<std::byte> b(prefix);
    std::memcpy(b.data(), "\x93NUMPY", 6);
    b[6] = std::byte(major);
    if (major == 1)
        snicholls::store_le(b.data() + 8, static_cast<std::uint16_t>(dict.size()));
    else
        snicholls::store_le(b.data() + 8, static_cast<std::uint32_t>(dict.size()));
    for (const char c : dict)
        b.push_back(static_cast<std::byte>(c));
    b.resize(b.size() + data);
    return b;
}
errc code_of(const std::vector<std::byte>& b) { return np::parse_npy(b).unwrap_err_unchecked().code; }
} // namespace

TEST(NumpyNpy, ReadsWhatNumpyWrites)
{
    // np.save of np.zeros((2, 3), np.float32): this header, byte for byte
    const auto b = npy_of("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", 24);
    const auto v = np::parse_npy(b, "x");
    ASSERT_TRUE(v.is_ok()) << v.unwrap_err_unchecked().what();
    EXPECT_EQ(v->type, dtype::f32);
    EXPECT_EQ(v->shape, (tensor_shape{2, 3}));
    EXPECT_EQ(v->data.data(), b.data() + b.size() - 24);
    EXPECT_EQ(v->name, "x");
    EXPECT_TRUE(np::parse_npy(npy_of("{'descr': '|b1', 'fortran_order': False, 'shape': (), }", 1)).is_ok());
    EXPECT_EQ(np::parse_npy(npy_of("{'descr': '<i8', 'fortran_order': False, 'shape': (5,), }", 40, 2))->shape, (tensor_shape{5}));
    EXPECT_EQ(np::parse_npy(npy_of("{\"descr\": \"=u2\", \"fortran_order\": False, \"shape\": (1,)}", 2))->type, dtype::u16);
}

TEST(NumpyNpy, WriteIsNumpysLayout)
{
    const std::vector<float> f{1, 2, 3};
    const auto w = np::write_npy(tensor_view{"v", dtype::f32, {3}, std::as_bytes(std::span(f))});
    ASSERT_TRUE(w.is_ok());
    // np.save(np.array([1, 2, 3], np.float32)): a 128-byte header, the dict, numpy's 20 spare spaces, padding
    ASSERT_EQ(w->size(), 128u + 12);
    const std::string head(reinterpret_cast<const char*>(w->data()), 128);
    EXPECT_EQ(head.substr(0, 10), std::string("\x93NUMPY\x01\x00\x76\x00", 10));
    EXPECT_EQ(head.substr(10, 57), "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }");
    EXPECT_EQ(head.back(), '\n');
    const auto back = np::parse_npy(*w);
    ASSERT_TRUE(back.is_ok());
    EXPECT_EQ(back->shape, (tensor_shape{3}));
    const std::vector<std::int32_t> m(6);
    const auto two = np::write_npy(tensor_view{"m", dtype::i32, {2, 3}, std::as_bytes(std::span(m))});
    EXPECT_NE(std::string(reinterpret_cast<const char*>(two->data()), 128).find("'shape': (2, 3)"), std::string::npos);
    const std::vector<std::uint16_t> bf(1);
    EXPECT_EQ(np::write_npy(tensor_view{"b", dtype::bf16, {1}, std::as_bytes(std::span(bf))}).unwrap_err_unchecked().code, errc::bad_dtype);
    EXPECT_EQ(np::write_npy(tensor_view{"s", dtype::f32, {4}, std::as_bytes(std::span(f))}).unwrap_err_unchecked().code, errc::bad_shape);
    static_assert(np::dtype_of_descr("<i1") == dtype::i8 && !np::dtype_of_descr("<f16").has_value() && np::descr_of(dtype::u8) == "|u1");
}

TEST(NumpyNpy, RefusesWhatItCannotRead)
{
    EXPECT_EQ(code_of({}), errc::truncated);
    EXPECT_EQ(code_of(std::vector<std::byte>(20)), errc::bad_magic);
    EXPECT_EQ(code_of(npy_of("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }", 24)), errc::unsupported);
    EXPECT_EQ(code_of(npy_of("{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }", 4)), errc::unsupported);
    EXPECT_EQ(code_of(npy_of("{'descr': [('a', '<f4')], 'fortran_order': False, 'shape': (1,), }", 4)), errc::unsupported);
    EXPECT_EQ(code_of(npy_of("{'descr': '<c8', 'fortran_order': False, 'shape': (1,), }", 8)), errc::bad_dtype);
    EXPECT_EQ(code_of(npy_of("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }", 4)), errc::bad_shape);
    EXPECT_EQ(code_of(npy_of("{'descr': '<f4', 'shape': (1,), }", 4)), errc::bad_header);
    EXPECT_EQ(code_of(npy_of("{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'x': 1}", 4)), errc::bad_header);
    EXPECT_EQ(code_of(npy_of("__import__('os')", 4)), errc::bad_header);
    auto v9 = npy_of("{}", 0);
    v9[6] = std::byte{9};
    EXPECT_EQ(code_of(v9), errc::unsupported_version);
}

TEST(NumpyNpz, RoundTripsLikeSavez)
{
    const std::vector<double> a{1.5, 2.5};
    const std::vector<std::uint8_t> b{1, 2, 3};
    const std::vector<tensor_view> ts{{"a", dtype::f64, {2}, std::as_bytes(std::span(a))}, {"b", dtype::u8, {3}, std::as_bytes(std::span(b))}};
    const auto z = np::write_npz(ts);
    ASSERT_TRUE(z.is_ok());
    const auto r = np::parse_npz(*z);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    ASSERT_EQ(r->size(), 2u);
    EXPECT_EQ((*r)[0].name, "a");
    EXPECT_EQ((*r)[1].name, "b");
    EXPECT_EQ(reinterpret_cast<const double*>((*r)[0].data.data())[1], 2.5);
    const std::string junk = "x";
    const auto other = zip::write(std::vector<zip::member>{{"notes.txt", std::as_bytes(std::span(junk.data(), 1))}});
    EXPECT_EQ(np::parse_npz(*other).unwrap_err_unchecked().code, errc::bad_header);
}
