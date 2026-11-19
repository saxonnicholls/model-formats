//
//  vocabulary_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/vocabulary.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace snicholls::formats;

namespace
{
// The 256 bytes as ids 0..255, then "he", "ll", "hell"
vocabulary tiny()
{
    vocabulary v;
    for (unsigned b = 0; b < 256; ++b)
        v.tokens.push_back(detail::byte_level_encode(std::string(1, static_cast<char>(b))));
    v.tokens.push_back("he");
    v.tokens.push_back("ll");
    v.tokens.push_back("hell");
    v.tokens.push_back("<|endoftext|>");
    v.types.assign(v.tokens.size(), gguf::token_type::normal);
    v.types.back() = gguf::token_type::control;
    v.merges = {"h e", "l l", "he ll"};
    return v;
}
} // namespace

TEST(Vocabulary, ByteLevelAlphabetIsGpt2s)
{
    EXPECT_EQ(detail::byte_level_encode(" "), "\xc4\xa0");   // U+0120, 'Ġ'
    EXPECT_EQ(detail::byte_level_encode("\n"), "\xc4\x8a");  // U+010A, 'Ċ'
    EXPECT_EQ(detail::byte_level_encode("a"), "a");
    for (unsigned b = 0; b < 256; ++b) {
        const std::string one(1, static_cast<char>(b));
        EXPECT_EQ(detail::byte_level_decode(detail::byte_level_encode(one)), one);
    }
    EXPECT_FALSE(detail::byte_level_decode("\xe2\x82\xac").has_value());   // U+20AC is not in the alphabet
    EXPECT_FALSE(detail::byte_level_decode("\xc4").has_value());           // malformed UTF-8
}

TEST(Vocabulary, TokenBytesDecodeTheAlphabet)
{
    const auto v = tiny();
    const auto b = v.token_bytes();
    ASSERT_TRUE(b.is_ok()) << b.unwrap_err_unchecked().what();
    EXPECT_EQ((*b)[32], " ");
    EXPECT_EQ((*b)[258], "hell");
    EXPECT_EQ((*b)[259], "<|endoftext|>");   // a control token is its UTF-8 text
    vocabulary bad = v;
    bad.tokens.push_back("\xe2\x82\xac");
    bad.types.push_back(gguf::token_type::normal);
    EXPECT_EQ(bad.token_bytes().unwrap_err_unchecked().code, errc::bad_header);
    vocabulary sp;
    sp.kind = tokenizer_kind::sentencepiece;
    EXPECT_EQ(sp.token_bytes().unwrap_err_unchecked().code, errc::unsupported);
    EXPECT_EQ(gguf_tokenizer_model(tokenizer_kind::unigram), "t5");
}

TEST(Vocabulary, CoresBpeEncodesOverTheRanks)
{
    const auto r = bpe_ranks::make(tiny());
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    EXPECT_TRUE(r->ids_follow_merges());
    EXPECT_EQ((*r)("hell"), 258u);
    EXPECT_EQ((*r)("<|endoftext|>"), snicholls::ml::bpe_no_rank);   // special tokens are not merge results
    std::array<std::uint32_t, 8> out{};
    std::array<snicholls::ml::bpe_part, 8> scratch{};
    const auto n = snicholls::ml::bpe_encode_piece("hello", *r, out, scratch);
    ASSERT_TRUE(n.is_ok());
    EXPECT_EQ(*n, 2u);
    EXPECT_EQ(out[0], 258u);
    EXPECT_EQ(out[1], static_cast<std::uint32_t>('o'));
}

TEST(Vocabulary, RanksNoticeIdsOutOfMergeOrder)
{
    auto v = tiny();
    std::swap(v.merges[0], v.merges[1]);   // "l l" (id 257) before "h e" (id 256)
    const auto r = bpe_ranks::make(v);
    ASSERT_TRUE(r.is_ok());
    EXPECT_FALSE(r->ids_follow_merges());
    v.merges.push_back("x zz");
    EXPECT_EQ(bpe_ranks::make(v).unwrap_err_unchecked().code, errc::bad_header);
    v.merges.back() = "nospace";
    EXPECT_EQ(bpe_ranks::make(v).unwrap_err_unchecked().code, errc::bad_header);
}

TEST(Vocabulary, TypeOfDefaultsToNormal)
{
    vocabulary v;
    v.tokens = {"a"};
    EXPECT_EQ(v.type_of(0), gguf::token_type::normal);
    EXPECT_EQ(v.type_of(99), gguf::token_type::normal);
}
