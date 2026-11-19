//
//  index_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/safetensors/index.hpp>

#include <gtest/gtest.h>

using namespace snicholls::formats;
namespace st = snicholls::formats::safetensors;

TEST(SafetensorsIndex, ParsesWeightMapAndMetadata)
{
    const auto r = st::parse_index(R"({"metadata":{"total_size":123,"format":"pt"},"weight_map":{"b":"m-2.safetensors","a":"m-1.safetensors","c":"m-1.safetensors"}})");
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    EXPECT_EQ(r->total_size, 123u);
    ASSERT_EQ(r->weight_map.size(), 3u);
    EXPECT_EQ(r->weight_map[0].first, "a");
    EXPECT_EQ(*r->shard_of("b"), "m-2.safetensors");
    EXPECT_EQ(r->shard_of("z"), nullptr);
    EXPECT_EQ(r->files(), (std::vector<std::string>{"m-1.safetensors", "m-2.safetensors"}));
    ASSERT_EQ(r->metadata.size(), 1u);
    EXPECT_EQ(r->metadata[0].second, "\"pt\"");
}

TEST(SafetensorsIndex, RefusesUnsafeShardNames)
{
    for (const char* bad : {"/etc/passwd", "../x.safetensors", "a/../../b", "a\\\\b", "", "a//b", "./a"}) {
        const std::string text = std::string(R"({"weight_map":{"t":")") + bad + "\"}}";
        const auto r = st::parse_index(text);
        ASSERT_TRUE(r.is_err()) << bad;
        EXPECT_EQ(r.unwrap_err_unchecked().code, errc::bad_argument) << bad;
    }
    static_assert(st::safe_relative_name("sub/model.safetensors") && !st::safe_relative_name(".."));
}

TEST(SafetensorsIndex, RefusesWrongShapes)
{
    EXPECT_EQ(st::parse_index("{}").unwrap_err_unchecked().code, errc::bad_header);
    EXPECT_EQ(st::parse_index(R"({"weight_map":{"t":1}})").unwrap_err_unchecked().code, errc::bad_header);
    EXPECT_EQ(st::parse_index(R"({"weight_map":{},"metadata":{"total_size":-1}})").unwrap_err_unchecked().code, errc::bad_header);
    EXPECT_EQ(st::parse_index(R"({"weight_map":{"t":"a","t":"b"}})").unwrap_err_unchecked().code, errc::duplicate_name);
    EXPECT_EQ(st::parse_index("nope").unwrap_err_unchecked().code, errc::bad_json);
}

TEST(SafetensorsIndex, WriteRoundTrips)
{
    st::index ix;
    ix.total_size = 42;
    ix.metadata = {{"format", "\"pt\""}};
    ix.weight_map = {{"a", "x.safetensors"}, {"b", "y.safetensors"}};
    const auto text = st::write_index(ix);
    ASSERT_TRUE(text.is_ok());
    const auto back = st::parse_index(*text);
    ASSERT_TRUE(back.is_ok());
    EXPECT_EQ(back->weight_map, ix.weight_map);
    EXPECT_EQ(back->total_size, 42u);
    EXPECT_EQ(back->metadata, ix.metadata);
    ix.weight_map.push_back({"c", "../evil"});
    EXPECT_TRUE(st::write_index(ix).is_err());
}

TEST(SafetensorsIndex, ShardNamesFollowTheConvention)
{
    EXPECT_EQ(st::shard_names("model", 1), (std::vector<std::string>{"model.safetensors"}));
    EXPECT_EQ(st::shard_names("model", 2), (std::vector<std::string>{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}));
}
