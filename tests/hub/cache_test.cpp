//
//  cache_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/hub/cache.hpp>

#include <gtest/gtest.h>

#include "support/temp_dir.hpp"

using namespace snicholls::formats;
using namespace snicholls::formats::hub;

namespace
{
constexpr const char* commit = "0123456789abcdef0123456789abcdef01234567";
}

TEST(HubCache, LayoutIsHuggingFaceHubs)
{
    const auto c = cache::open("/c", "openai-community/gpt2");
    ASSERT_TRUE(c.is_ok());
    EXPECT_EQ(c->dir(), std::filesystem::path("/c/models--openai-community--gpt2"));
    EXPECT_EQ(c->blob("abc"), std::filesystem::path("/c/models--openai-community--gpt2/blobs/abc"));
    EXPECT_EQ(c->incomplete("abc").filename(), "abc.incomplete");
    EXPECT_EQ(c->snapshot(commit), std::filesystem::path("/c/models--openai-community--gpt2/snapshots") / commit);
    EXPECT_EQ(cache::open("/c", "gpt2")->dir().filename(), "models--gpt2");
}

TEST(HubCache, RefusesUnsafeNames)
{
    for (const char* bad : {"", "../x", "a/b/c", "a b", "/abs", "org/..", "-rf"})
        EXPECT_FALSE(valid_repo_id(bad)) << bad;
    EXPECT_TRUE(valid_repo_id("hf-internal-testing/tiny-random-gpt2"));
    EXPECT_TRUE(valid_revision("refs/pr/1") && valid_revision("main") && !valid_revision("../main") && !valid_revision("a b"));
    static_assert(is_commit("0123456789abcdef0123456789abcdef01234567") && !is_commit("main"));
    EXPECT_EQ(cache::open("/c", "../../etc").unwrap_err_unchecked().code, errc::bad_argument);
}

TEST(HubCache, RefsLinksAndOfflineResolution)
{
    const test_support::temp_dir dir("cache");
    const auto c = *cache::open(dir.path, "org/m");
    EXPECT_EQ(c.resolve_offline("main").unwrap_err_unchecked().code, errc::offline);
    ASSERT_TRUE(c.write_ref("main", commit).is_ok());
    EXPECT_EQ(*c.resolve_offline("main"), commit);
    std::filesystem::create_directories(c.blob("").parent_path());
    const std::string text = "x";
    ASSERT_TRUE(io::write_file(c.blob("e1"), std::as_bytes(std::span(text.data(), 1))).is_ok());
    ASSERT_TRUE(c.link(commit, "sub/dir/f.txt", "e1").is_ok());
    EXPECT_TRUE(c.has(commit, "sub/dir/f.txt"));
    EXPECT_EQ(std::filesystem::read_symlink(c.snapshot(commit) / "sub/dir/f.txt"), std::filesystem::path("../../../../blobs/e1"));
    EXPECT_EQ(*c.resolve_offline(commit), commit);
    EXPECT_FALSE(c.has(commit, "missing"));
    EXPECT_EQ(c.link(commit, "../evil", "e1").unwrap_err_unchecked().code, errc::bad_argument);
    EXPECT_EQ(c.write_ref("../x", commit).unwrap_err_unchecked().code, errc::bad_argument);
}

TEST(HubCache, DefaultDirectoryFollowsTheEnvironment)
{
    using snicholls::load_config;
    const auto with = [](std::map<std::string, std::string> env) { return default_cache_dir(load_config({.env_files = {}, .environment = env})); };
    EXPECT_EQ(with({{"HF_HUB_CACHE", "/a"}, {"HF_HOME", "/b"}}), std::filesystem::path("/a"));
    EXPECT_EQ(with({{"HF_HOME", "/b"}}), std::filesystem::path("/b/hub"));
    EXPECT_EQ(with({{"XDG_CACHE_HOME", "/x"}}), std::filesystem::path("/x/huggingface/hub"));
    EXPECT_EQ(with({{"HOME", "/home/u"}}), std::filesystem::path("/home/u/.cache/huggingface/hub"));
}
