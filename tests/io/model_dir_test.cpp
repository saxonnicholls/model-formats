//
//  model_dir_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/io/model_dir.hpp>

#include <gtest/gtest.h>

#include <fstream>

#include "support/temp_dir.hpp"

using namespace snicholls::formats;

namespace
{
void write_text(const std::filesystem::path& p, const std::string& s) { std::ofstream(p) << s; }
constexpr const char* gpt2_config = R"({"model_type":"gpt2","n_embd":4,"n_head":2,"n_layer":1,"n_positions":8,"vocab_size":3})";
constexpr const char* tokenizer = R"({"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}})";
} // namespace

TEST(IoModelDir, OpensASingleFileModel)
{
    const test_support::temp_dir dir("model-dir");
    write_text(dir.path / "config.json", gpt2_config);
    write_text(dir.path / "tokenizer.json", tokenizer);
    write_text(dir.path / "tokenizer_config.json", R"({"chat_template":"{{ x }}"})");
    write_text(dir.path / "generation_config.json", R"({"eos_token_id":2})");
    const std::vector<float> w(12);
    const std::vector<tensor_view> ts{{"wte.weight", dtype::f32, {3, 4}, std::as_bytes(std::span(w))}};
    ASSERT_TRUE(io::save_tensors(dir.path / "model.safetensors", ts).is_ok());
    const auto m = io::open_model_dir(dir.path);
    ASSERT_TRUE(m.is_ok()) << m.unwrap_err_unchecked().what();
    ASSERT_TRUE(m->config.has_value());
    EXPECT_EQ(m->config->arch, architecture::gpt2);
    EXPECT_EQ(m->tokenizer->vocab.merges.size(), 1u);
    EXPECT_EQ(*m->tokenizer->default_chat_template(), "{{ x }}");
    EXPECT_EQ(m->generation->eos, (std::vector<std::uint32_t>{2}));
    ASSERT_NE(m->find("wte.weight"), nullptr);
    EXPECT_EQ(m->find("nope"), nullptr);
}

TEST(IoModelDir, OpensAndChecksShards)
{
    const test_support::temp_dir dir("shards");
    write_text(dir.path / "config.json", R"({"model_type":"bert"})");
    write_text(dir.path / "chat_template.jinja", "J");
    write_text(dir.path / "tokenizer.json", tokenizer);
    const std::vector<float> a(2), b(3);
    ASSERT_TRUE(io::save_tensors(dir.path / "m-1.safetensors", std::vector<tensor_view>{{"a", dtype::f32, {2}, std::as_bytes(std::span(a))}}).is_ok());
    ASSERT_TRUE(io::save_tensors(dir.path / "m-2.safetensors", std::vector<tensor_view>{{"b", dtype::f32, {3}, std::as_bytes(std::span(b))}}).is_ok());
    write_text(dir.path / "model.safetensors.index.json", R"({"weight_map":{"a":"m-1.safetensors","b":"m-2.safetensors"}})");
    const auto m = io::open_model_dir(dir.path);
    ASSERT_TRUE(m.is_ok()) << m.unwrap_err_unchecked().what();
    EXPECT_FALSE(m->config.has_value());
    EXPECT_EQ(m->config_error->code, errc::unsupported);
    EXPECT_EQ(m->tensors.size(), 2u);
    EXPECT_EQ(m->files.size(), 2u);
    EXPECT_EQ(*m->tokenizer->default_chat_template(), "J");
    write_text(dir.path / "model.safetensors.index.json", R"({"weight_map":{"a":"m-2.safetensors","b":"m-2.safetensors"}})");
    EXPECT_EQ(io::open_model_dir(dir.path).unwrap_err_unchecked().code, errc::bad_header);
    write_text(dir.path / "model.safetensors.index.json", R"({"weight_map":{"a":"m-1.safetensors","b":"m-2.safetensors","c":"m-2.safetensors"}})");
    EXPECT_EQ(io::open_model_dir(dir.path).unwrap_err_unchecked().code, errc::bad_header);
    write_text(dir.path / "model.safetensors.index.json", R"({"weight_map":{"a":"../m-1.safetensors"}})");
    EXPECT_EQ(io::open_model_dir(dir.path).unwrap_err_unchecked().code, errc::bad_argument);
}

TEST(IoModelDir, RefusesIncompleteDirectories)
{
    const test_support::temp_dir dir("empty-dir");
    EXPECT_EQ(io::open_model_dir(dir.path).unwrap_err_unchecked().code, errc::not_found);
    write_text(dir.path / "config.json", gpt2_config);
    EXPECT_EQ(io::open_model_dir(dir.path).unwrap_err_unchecked().code, errc::not_found);
}
