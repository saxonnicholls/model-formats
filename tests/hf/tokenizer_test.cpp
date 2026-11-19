//
//  tokenizer_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/hf/tokenizer.hpp>

#include <gtest/gtest.h>

using namespace snicholls::formats;

namespace
{
// A GPT-2-shaped tokenizer.json: four tokens, two merges, one special token
constexpr const char* byte_level = R"({"version":"1.0","added_tokens":[{"id":5,"content":"<|endoftext|>","special":true}],
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false},"decoder":{"type":"ByteLevel"},
  "model":{"type":"BPE","vocab":{"h":0,"e":1,"he":2,"Ġ":3,"Ġhe":4},"merges":["h e",["Ġ","he"]]}})";
} // namespace

TEST(HfTokenizer, ReadsAByteLevelBpe)
{
    const auto t = hf::parse_tokenizer(byte_level);
    ASSERT_TRUE(t.is_ok()) << t.unwrap_err_unchecked().what();
    const auto& v = t->vocab;
    EXPECT_EQ(v.kind, tokenizer_kind::byte_level_bpe);
    ASSERT_EQ(v.tokens.size(), 6u);
    EXPECT_EQ(v.tokens[4], "\xc4\xa0he");
    EXPECT_EQ(v.merges, (std::vector<std::string>{"h e", "\xc4\xa0 he"}));
    EXPECT_EQ(v.type_of(5), gguf::token_type::control);
    EXPECT_EQ(v.type_of(0), gguf::token_type::normal);
    EXPECT_EQ((*v.token_bytes())[4], " he");
    EXPECT_EQ(t->id_of("<|endoftext|>"), 5u);
}

TEST(HfTokenizer, ConfigAddsSpecialTokensAndTemplates)
{
    auto t = *hf::parse_tokenizer(byte_level);
    ASSERT_TRUE(hf::parse_tokenizer_config(t, R"({"add_bos_token":false,"bos_token":"<|endoftext|>",
        "eos_token":{"content":"<|endoftext|>","special":true},"chat_template":"{% for m in messages %}{{ m.content }}{% endfor %}"})")
                    .is_ok());
    EXPECT_EQ(t.vocab.bos, 5u);
    EXPECT_EQ(t.vocab.eos, 5u);
    EXPECT_EQ(t.add_bos, false);
    ASSERT_NE(t.default_chat_template(), nullptr);
    EXPECT_EQ(t.default_chat_template()->substr(0, 6), "{% for");
    auto u = *hf::parse_tokenizer(byte_level);
    ASSERT_TRUE(hf::parse_tokenizer_config(u, R"({"chat_template":[{"name":"tool_use","template":"T"},{"name":"default","template":"D"}]})").is_ok());
    EXPECT_EQ(*u.default_chat_template(), "D");
    EXPECT_EQ(u.chat_templates.size(), 2u);
    EXPECT_TRUE(hf::parse_tokenizer_config(u, R"({"chat_template":3})").is_err());
    hf::tokenizer none;
    EXPECT_EQ(none.default_chat_template(), nullptr);
}

TEST(HfTokenizer, FillsGapsAndRefusesClashes)
{
    const auto gap = hf::parse_tokenizer(R"({"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"a":0,"b":3},"merges":[]}})");
    ASSERT_TRUE(gap.is_ok());
    EXPECT_EQ(gap->vocab.tokens[1], "[PAD1]");
    EXPECT_EQ(gap->vocab.type_of(2), gguf::token_type::unused);
    const auto clash = hf::parse_tokenizer(R"({"pre_tokenizer":{"type":"ByteLevel"},"added_tokens":[{"id":0,"content":"<x>"}],
        "model":{"type":"BPE","vocab":{"a":0},"merges":[]}})");
    EXPECT_EQ(clash.unwrap_err_unchecked().code, errc::bad_header);
}

TEST(HfTokenizer, SentencePieceWordPieceAndUnigramAreReadNotEncoded)
{
    const auto sp = hf::parse_tokenizer(R"({"decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"}]},
        "model":{"type":"BPE","byte_fallback":true,"vocab":{"<unk>":0,"<0x0A>":1,"▁the":2},"merges":[],"unk_token":"<unk>"}})");
    ASSERT_TRUE(sp.is_ok()) << sp.unwrap_err_unchecked().what();
    EXPECT_EQ(sp->vocab.kind, tokenizer_kind::sentencepiece);
    EXPECT_EQ(sp->vocab.type_of(1), gguf::token_type::byte);
    EXPECT_EQ(sp->vocab.unk, 0u);
    EXPECT_EQ(sp->vocab.token_bytes().unwrap_err_unchecked().code, errc::unsupported);
    const auto wp = hf::parse_tokenizer(R"({"model":{"type":"WordPiece","vocab":{"[UNK]":0,"##s":1}}})");
    ASSERT_TRUE(wp.is_ok());
    EXPECT_EQ(wp->vocab.kind, tokenizer_kind::wordpiece);
    const auto ug = hf::parse_tokenizer(R"({"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["a",-1.5]]}})");
    ASSERT_TRUE(ug.is_ok());
    EXPECT_EQ(ug->vocab.scores, (std::vector<float>{0.0f, -1.5f}));
    EXPECT_EQ(hf::parse_tokenizer(R"({"model":{"type":"Unigram","vocab":[["a"]]}})").unwrap_err_unchecked().code, errc::bad_header);
}

TEST(HfTokenizer, RefusesMalformedFiles)
{
    EXPECT_EQ(hf::parse_tokenizer("{}").unwrap_err_unchecked().code, errc::bad_header);
    EXPECT_EQ(hf::parse_tokenizer(R"({"model":{"type":"WordLevel","vocab":{}}})").unwrap_err_unchecked().code, errc::unsupported);
    EXPECT_EQ(hf::parse_tokenizer(R"({"model":{"type":"BPE","vocab":{},"merges":[]}})").unwrap_err_unchecked().code, errc::unsupported);
    EXPECT_EQ(hf::parse_tokenizer(R"({"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"a":-1}}})").unwrap_err_unchecked().code,
              errc::bad_header);
    EXPECT_EQ(hf::parse_tokenizer(R"({"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{},"merges":[1]}})").unwrap_err_unchecked().code,
              errc::bad_header);
}
