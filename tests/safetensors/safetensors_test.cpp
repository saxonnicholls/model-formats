//
//  safetensors_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/safetensors/safetensors.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

using namespace snicholls::formats;
namespace st = snicholls::formats::safetensors;

namespace
{
// A file from a header written by hand and `data` bytes of zeros
std::vector<std::byte> file_of(const std::string& header, std::size_t data, std::uint64_t claimed = ~0ull)
{
    std::vector<std::byte> b(8 + header.size() + data);
    const std::uint64_t n = claimed == ~0ull ? header.size() : claimed;
    std::memcpy(b.data(), &n, 8);
    std::memcpy(b.data() + 8, header.data(), header.size());
    return b;
}
errc code_of(const std::vector<std::byte>& b, st::parse_options opt = {})
{
    const auto r = st::parse(b, opt);
    return r.is_ok() ? errc{} : r.unwrap_err_unchecked().code;
}
} // namespace

TEST(SafetensorsParse, ReadsTensorsAndMetadata)
{
    const auto b = file_of(R"({"__metadata__":{"format":"pt"},"b":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},)"
                           R"("a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
                           12);
    const auto r = st::parse(b);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    ASSERT_EQ(r->tensors.size(), 2u);
    EXPECT_EQ(r->tensors[0].name, "a");   // data order
    EXPECT_EQ(r->tensors[1].type, dtype::f16);
    EXPECT_EQ(r->tensors[1].shape, (tensor_shape{2}));
    EXPECT_EQ(r->tensors[1].data.data(), b.data() + 8 + r->header_size + 8);   // zero-copy
    ASSERT_NE(r->meta("format"), nullptr);
    EXPECT_EQ(*r->meta("format"), "pt");
    EXPECT_EQ(r->meta("nope"), nullptr);
    EXPECT_NE(r->find("b"), nullptr);
    EXPECT_EQ(r->find("c"), nullptr);
    EXPECT_EQ(r->views().size(), 2u);
}

TEST(SafetensorsParse, EmptyFileAndScalars)
{
    EXPECT_TRUE(st::parse(file_of("{}", 0)).is_ok());
    const auto r = st::parse(file_of(R"({"s":{"dtype":"I64","shape":[],"data_offsets":[0,8]},"z":{"dtype":"F32","shape":[0,4],"data_offsets":[8,8]}})", 8));
    ASSERT_TRUE(r.is_ok());
    EXPECT_EQ(r->find("s")->shape.rank, 0u);
    EXPECT_EQ(r->find("z")->data.size(), 0u);
}

TEST(SafetensorsParse, RejectsMalformedHeaders)
{
    EXPECT_EQ(code_of({}), errc::truncated);
    EXPECT_EQ(code_of(file_of("{}", 0, 1000)), errc::truncated);
    EXPECT_EQ(code_of(file_of("{}", 0, 200'000'000)), errc::too_large);
    EXPECT_EQ(code_of(file_of("[]", 0)), errc::bad_header);
    EXPECT_EQ(code_of(file_of("{\"a\":", 0)), errc::bad_json);
    EXPECT_EQ(code_of(file_of(R"({"a":1})", 0)), errc::bad_header);
    EXPECT_EQ(code_of(file_of(R"({"a":{"shape":[1],"data_offsets":[0,4]}})", 4)), errc::bad_header);
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F4","shape":[1],"data_offsets":[0,4]}})", 4)), errc::bad_dtype);
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})", 4)), errc::bad_shape);
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,4]}})", 4)), errc::bad_shape);
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", 4)), errc::bad_header);
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}})", 4)), errc::bad_header);
    EXPECT_EQ(code_of(file_of(R"({"__metadata__":{"k":1}})", 0)), errc::bad_header);
    EXPECT_EQ(code_of(file_of(R"({"__metadata__":[]})", 0)), errc::bad_header);
}

TEST(SafetensorsParse, DuplicateKeysAreRefused)
{
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", 1)),
              errc::duplicate_name);
}

TEST(SafetensorsParse, ValidatesSizesBoundsOverlapsAndHoles)
{
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", 4)), errc::bad_shape);
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", 4)), errc::out_of_bounds);
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8)),
              errc::overlap);
    const auto hole = file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8);
    EXPECT_EQ(code_of(hole), errc::overlap);
    EXPECT_TRUE(st::parse(hole, st::parse_options{100000000, true}).is_ok());
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", 6)), errc::overlap);   // trailing bytes
    EXPECT_EQ(code_of(file_of(R"({"a":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,4]}})", 4)), errc::bad_shape);
}

TEST(SafetensorsWrite, RoundTripsAndAligns)
{
    const std::vector<std::uint16_t> h{1, 2, 3};
    const std::vector<double> d{1.5, -2.5};
    const std::vector<std::uint8_t> m{7};
    const std::vector<tensor_view> ts{{"half", dtype::f16, {3}, std::as_bytes(std::span(h))},
                                      {"mask", dtype::boolean, {1}, std::as_bytes(std::span(m))},
                                      {"dbl", dtype::f64, {2, 1}, std::as_bytes(std::span(d))}};
    const std::vector<st::metadata_entry> meta{{"z", "last"}, {"format", "pt"}};
    const auto w = st::write(ts, meta);
    ASSERT_TRUE(w.is_ok()) << w.unwrap_err_unchecked().what();
    EXPECT_EQ((8 + snicholls::load_le<std::uint64_t>(w->data())) % 8, 0u);   // data starts 8-aligned
    const auto r = st::parse(*w);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    ASSERT_EQ(r->tensors.size(), 3u);
    EXPECT_EQ(r->tensors[0].name, "dbl");   // largest elements first
    EXPECT_EQ(r->tensors[1].name, "half");
    EXPECT_EQ(r->tensors[2].name, "mask");
    for (const auto& t : r->tensors)
        EXPECT_EQ(t.begin % size_of(t.type), 0u);
    EXPECT_TRUE(std::equal(r->find("dbl")->data.begin(), r->find("dbl")->data.end(), ts[2].data.begin()));
    EXPECT_EQ(r->metadata[0].first, "format");
    EXPECT_EQ(*r->meta("z"), "last");
}

TEST(SafetensorsWrite, IsDeterministic)
{
    const std::vector<float> a{1, 2}, b{3};
    const std::vector<tensor_view> one{{"a", dtype::f32, {2}, std::as_bytes(std::span(a))}, {"b", dtype::f32, {1}, std::as_bytes(std::span(b))}};
    const std::vector<tensor_view> two{one[1], one[0]};
    EXPECT_EQ(*st::write(one), *st::write(two));
}

TEST(SafetensorsWrite, RefusesBadInput)
{
    const std::vector<float> a{1, 2};
    const auto code = [](std::vector<tensor_view> ts) { return st::write(ts).unwrap_err_unchecked().code; };
    EXPECT_EQ(code({{"a", dtype::f32, {3}, std::as_bytes(std::span(a))}}), errc::bad_shape);
    EXPECT_EQ(code({{"a", dtype::f32, {2}, std::as_bytes(std::span(a))}, {"a", dtype::f32, {2}, std::as_bytes(std::span(a))}}), errc::duplicate_name);
    EXPECT_EQ(code({{"__metadata__", dtype::f32, {2}, std::as_bytes(std::span(a))}}), errc::bad_argument);
    EXPECT_EQ(code({{"\xff", dtype::f32, {2}, std::as_bytes(std::span(a))}}), errc::bad_argument);
}

TEST(SafetensorsWrite, SinkReceivesTheSameBytes)
{
    const std::vector<std::int32_t> a{1, 2, 3};
    const std::vector<tensor_view> ts{{"a", dtype::i32, {3}, std::as_bytes(std::span(a))}};
    std::vector<std::byte> streamed;
    ASSERT_TRUE(st::write_to(ts, {}, [&](std::span<const std::byte> s) { streamed.insert(streamed.end(), s.begin(), s.end()); }).is_ok());
    EXPECT_EQ(streamed, *st::write(ts));
}
