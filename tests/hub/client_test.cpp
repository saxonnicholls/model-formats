//
//  client_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/hub/client.hpp>
#include <snicholls/formats/io/model_dir.hpp>
#include <snicholls/formats/vocabulary.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "support/fake_hub.hpp"
#include "support/temp_dir.hpp"

using namespace snicholls::formats;
using namespace snicholls::formats::hub;

namespace
{
std::string big(std::size_t n)
{
    std::string s(n, '\0');
    for (std::size_t i = 0; i < n; ++i)
        s[i] = static_cast<char>((i * 131 + 7) & 0xff);
    return s;
}

struct fixture
{
    test_support::temp_dir dir{"hub"};
    test_support::fake_hub hub;
    fixture()
    {
        hub.files["config.json"] = {"{\"model_type\":\"gpt2\"}", false, false};
        hub.files["model.safetensors"] = {big(200 * 1024 + 17), true, false};
        hub.files["sub/dir/notes.txt"] = {"hello\n", false, false};
        hub.files["empty.txt"] = {"", false, false};
        hub.start();
    }
    hub_options options(bool with_token = true)
    {
        hub_options o;
        o.endpoint = hub.hub.origin();
        o.cache_dir = dir.path;
        o.chunk_bytes = 64 * 1024;
        if (with_token)
            o.token = snicholls::secret<std::string>(hub.token);
        return o;
    }
};
} // namespace

TEST(HubClient, ResolvesAndDownloadsIntoTheCache)
{
    fixture f;
    client c(f.options());
    const auto info = c.resolve("org/tiny", "main");
    ASSERT_TRUE(info.is_ok()) << info.unwrap_err_unchecked().what();
    EXPECT_EQ(info->commit, test_support::fake_hub::commit);
    ASSERT_EQ(info->files.size(), 4u);
    const auto r = c.download(*info);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    EXPECT_EQ(r->downloaded.size(), 4u);
    EXPECT_EQ(r->bytes, 200u * 1024 + 17 + 21 + 6);
    const auto blob = cache::open(f.dir.path, "org/tiny")->blob(*info->files[1].sha256);   // sorted: config, empty, model, sub
    (void)blob;
    for (const auto& [path, file] : f.hub.files) {
        const auto p = r->snapshot / path;
        ASSERT_TRUE(std::filesystem::exists(p)) << path;
        EXPECT_TRUE(std::filesystem::is_symlink(p));
        EXPECT_EQ(std::filesystem::file_size(p), file.content.size());
    }
    EXPECT_EQ(std::filesystem::read_symlink(r->snapshot / "model.safetensors").parent_path(), std::filesystem::path("../../blobs"));
    EXPECT_EQ(f.hub.auth_leaks.load(), 0);   // the token stayed with the Hub's origin
    EXPECT_GE(f.hub.cdn_requests.load(), 4);  // 200 KiB in 64 KiB ranges
    EXPECT_EQ(*cache::open(f.dir.path, "org/tiny")->resolve_offline("main"), info->commit);
    // a second download fetches nothing
    const int before = f.hub.requests.load();
    const auto again = c.download(*info);
    ASSERT_TRUE(again.is_ok());
    EXPECT_EQ(again->cached.size(), 4u);
    EXPECT_EQ(f.hub.requests.load(), before);
}

TEST(HubClient, PatternsChooseFiles)
{
    fixture f;
    client c(f.options());
    const std::vector<std::string> include{"*.json"};
    const auto r = c.snapshot_download("org/tiny", "main", include);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    EXPECT_EQ(r->downloaded, (std::vector<std::string>{"config.json"}));
    EXPECT_FALSE(std::filesystem::exists(r->snapshot / "model.safetensors"));
}

TEST(HubClient, ResumesFromWholeRanges)
{
    fixture f;
    client c(f.options());
    const auto info = *c.resolve("org/tiny", "main");
    const auto& model = f.hub.files["model.safetensors"].content;
    const auto ch = *cache::open(f.dir.path, "org/tiny");
    const auto sha = sha256_hex(std::as_bytes(std::span(model.data(), model.size())));
    std::filesystem::create_directories(ch.blob("").parent_path());
    // an interrupted download: two whole ranges and part of a third
    const std::string part = model.substr(0, 2 * 64 * 1024 + 1000);
    ASSERT_TRUE(io::write_file(ch.incomplete(sha), std::as_bytes(std::span(part.data(), part.size()))).is_ok());
    const std::vector<std::string> only{"model.safetensors"};
    const auto r = c.download(info, only);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    EXPECT_EQ(r->resumed, 2u * 64 * 1024);
    EXPECT_EQ(r->bytes, model.size() - 2 * 64 * 1024);
    EXPECT_TRUE(std::filesystem::exists(ch.blob(sha)));
    EXPECT_FALSE(std::filesystem::exists(ch.incomplete(sha)));
}

TEST(HubClient, ACorruptedFileIsNeverCached)
{
    fixture f;
    f.hub.files["model.safetensors"].corrupt = true;
    client c(f.options());
    const std::vector<std::string> only{"model.safetensors"};
    const auto r = c.snapshot_download("org/tiny", "main", only);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err_unchecked().code, errc::checksum_mismatch);
    const auto& model = f.hub.files["model.safetensors"].content;
    const auto ch = *cache::open(f.dir.path, "org/tiny");
    const auto sha = sha256_hex(std::as_bytes(std::span(model.data(), model.size())));
    EXPECT_FALSE(std::filesystem::exists(ch.blob(sha)));
    EXPECT_FALSE(std::filesystem::exists(ch.incomplete(sha)));
}

TEST(HubClient, HttpErrorsAreReported)
{
    fixture f;
    client c(f.options());
    EXPECT_EQ(c.resolve("org/missing", "main").unwrap_err_unchecked().code, errc::not_found);
    client anonymous(f.options(false));
    const auto denied = anonymous.resolve("org/private", "main");
    ASSERT_TRUE(denied.is_err());
    EXPECT_EQ(denied.unwrap_err_unchecked().code, errc::http);
    EXPECT_NE(denied.unwrap_err_unchecked().message.find("401"), std::string::npos);
    EXPECT_EQ(denied.unwrap_err_unchecked().message.find(f.hub.token), std::string::npos);
    EXPECT_TRUE(c.resolve("org/private", "main").is_ok());   // with the token
    EXPECT_EQ(c.resolve("../etc", "main").unwrap_err_unchecked().code, errc::bad_argument);
    hub_options unreachable = f.options();
    unreachable.endpoint = "http://127.0.0.1:1";
    unreachable.timeout = std::chrono::milliseconds(2000);
    EXPECT_EQ(client(unreachable).resolve("org/tiny", "main").unwrap_err_unchecked().code, errc::http);
}

TEST(HubClient, TheCapIsCheckedFirst)
{
    fixture f;
    auto o = f.options();
    o.max_download_bytes = 1000;
    client c(o);
    const auto r = c.snapshot_download("org/tiny", "main");
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err_unchecked().code, errc::too_large);
    EXPECT_EQ(f.hub.bytes_served.load(), 0u);
}

TEST(HubClient, OfflineUsesTheCacheOnly)
{
    fixture f;
    {
        client online(f.options());
        const std::vector<std::string> json{"*.json"};
        ASSERT_TRUE(online.snapshot_download("org/tiny", "main", json).is_ok());
    }
    auto o = f.options();
    o.offline = true;
    client offline(o);
    const int before = f.hub.requests.load();
    const auto info = offline.resolve("org/tiny", "main");
    ASSERT_TRUE(info.is_ok()) << info.unwrap_err_unchecked().what();
    ASSERT_EQ(info->files.size(), 1u);
    EXPECT_EQ(info->files[0].path, "config.json");
    EXPECT_TRUE(offline.download(*info).is_ok());
    repo_info more = *info;
    more.files.push_back({"model.safetensors", 10, std::string(40, 'a'), std::string(64, 'b')});
    EXPECT_EQ(offline.download(more).unwrap_err_unchecked().code, errc::offline);
    EXPECT_EQ(offline.resolve("org/tiny", "v2").unwrap_err_unchecked().code, errc::offline);
    EXPECT_EQ(f.hub.requests.load(), before);   // nothing went to the network
}

TEST(HubClient, OptionsFromTheEnvironment)
{
    using snicholls::load_config;
    const auto o = hub_options::from_environment(load_config(
        {.env_files = {}, .environment = std::map<std::string, std::string>{{"HF_TOKEN", "hf_x"}, {"HF_HUB_OFFLINE", "1"}, {"HF_ENDPOINT", "https://mirror.example/"}, {"HF_HOME", "/h"}}}));
    EXPECT_EQ(o.token->reveal(), "hf_x");
    EXPECT_TRUE(o.offline);
    EXPECT_EQ(o.endpoint, "https://mirror.example");
    EXPECT_EQ(o.cache_dir, std::filesystem::path("/h/hub"));
    const auto none = hub_options::from_environment(load_config({.env_files = {}, .environment = std::map<std::string, std::string>{}}));
    EXPECT_FALSE(none.token.has_value());
    EXPECT_FALSE(none.offline);
}

// The real Hub. Off unless MODEL_FORMATS_NETWORK=1 (CTest network.hub/client, label network, sets it when configured
// with -DMODEL_FORMATS_NETWORK_TESTS=ON). A tiny public repository: a few MB.
TEST(HubClientNetwork, DownloadsATinyPublicModel)
{
    const char* on = std::getenv("MODEL_FORMATS_NETWORK");
    if (!on || std::string(on) != "1")
        GTEST_SKIP() << "set MODEL_FORMATS_NETWORK=1 to reach huggingface.co";
    const test_support::temp_dir dir("hub-network");
    auto o = hub_options::from_environment();
    o.cache_dir = dir.path;
    o.offline = false;
    o.max_download_bytes = 200u << 20;
    client c(o);
    const auto info = c.resolve("hf-internal-testing/tiny-random-gpt2", "main");
    ASSERT_TRUE(info.is_ok()) << info.unwrap_err_unchecked().what();
    EXPECT_TRUE(is_commit(info->commit));
    const std::vector<std::string> include{"*.json", "*.safetensors", "*.txt"};
    const auto r = c.download(*info, include);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    EXPECT_FALSE(r->downloaded.empty());
    const auto m = io::open_model_dir(r->snapshot);
    ASSERT_TRUE(m.is_ok()) << m.unwrap_err_unchecked().what();
    ASSERT_TRUE(m->config.has_value()) << m->config_error->what();
    EXPECT_EQ(m->config->arch, architecture::gpt2);
    EXPECT_FALSE(m->tensors.empty());
    ASSERT_TRUE(m->tokenizer.has_value());
    EXPECT_EQ(m->tokenizer->vocab.kind, tokenizer_kind::byte_level_bpe);
    EXPECT_TRUE(bpe_ranks::make(m->tokenizer->vocab).is_ok());
    std::printf("[network] %s@%s: %zu files, %llu bytes, %zu tensors\n", info->repo.c_str(), info->commit.c_str(), r->downloaded.size(),
                static_cast<unsigned long long>(r->bytes), m->tensors.size());
    const auto again = c.download(*info, include);
    ASSERT_TRUE(again.is_ok());
    EXPECT_TRUE(again->downloaded.empty());
    EXPECT_EQ(again->bytes, 0u);
}
