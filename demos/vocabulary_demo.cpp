//
//  vocabulary_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A byte-level vocabulary, its tokens' bytes, and core's BPE encoding over it.

#include <snicholls/formats/vocabulary.hpp>

#include <array>
#include <cstdio>

using namespace snicholls::formats;

int main()
{
    vocabulary v;
    for (unsigned b = 0; b < 256; ++b)
        v.tokens.push_back(detail::byte_level_encode(std::string(1, static_cast<char>(b))));
    for (const char* t : {"\xc4\xa0w", "or", "ld", "\xc4\xa0wor", "\xc4\xa0world"})   // "Ġw" is " w"
        v.tokens.push_back(t);
    v.merges = {"\xc4\xa0 w", "o r", "l d", "\xc4\xa0w or", "\xc4\xa0wor ld"};
    //! [vocabulary]
    const auto bytes = v.token_bytes();         // each token's raw bytes, by id
    const auto ranks = bpe_ranks::make(v);      // core's rank(bytes)
    std::array<std::uint32_t, 16> ids{};
    std::array<snicholls::ml::bpe_part, 16> scratch{};
    const auto n = snicholls::ml::bpe_encode_piece(" world", *ranks, ids, scratch);
    //! [vocabulary]
    std::printf("token 260 is \"%s\"; \" world\" encodes to %zu token(s): %u; ids follow merges: %s\n", (*bytes)[260].c_str(), *n, ids[0],
                ranks->ids_follow_merges() ? "yes" : "no");
    return bytes.is_ok() && ranks.is_ok() && n.is_ok() && *n == 1 && ids[0] == 260 && ranks->ids_follow_merges() ? 0 : 1;
}
