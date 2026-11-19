//
//  vocabulary.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `vocabulary`: a tokenizer's tokens, merges and special tokens, and the rank lookup core's BPE takes.
/// \ingroup formats
///
/// \overview
/// A tokenizer as files carry it: the tokens by id (as text: byte-level BPE writes each byte as one printable code
/// point, so `"Ġhello"` is the bytes `" hello"`), the merges in rank order, each token's type, and the special tokens.
/// `hf::parse_tokenizer` and `gguf::read_vocabulary` fill it; `gguf::write_vocabulary` writes it.
///
/// For byte-level BPE (GPT-2, Llama 3, Qwen, DeepSeek, OLMoE), `token_bytes()` gives each token's raw bytes, which is
/// what core's `ml::bpe_encode_piece` and the ml library's GPT-2 tokenizer take; `bpe_ranks` is the `rank(bytes)`
/// function core's BPE calls. Core's merge loop uses the token id as the merge rank (tiktoken's rule), which agrees
/// with Hugging Face's merge order exactly when the ids of merged tokens increase with their merge's rank:
/// `bpe_ranks::ids_follow_merges()` checks it (true for GPT-2).
///
/// SentencePiece (Llama 1 and 2, Mistral's first tokenizer), WordPiece and Unigram vocabularies are read as tokens
/// with their kind, so they can be inspected and converted, but `token_bytes` and `bpe_ranks` refuse them: core's
/// tokenizer track will add their encoders, and this is the seam they plug into.
///
/// \relations
/// \see snicholls::ml::bpe_encode_piece (core), formats::hf::parse_tokenizer, formats::gguf::read_vocabulary
///
/// \impls
/// `vocabulary` (data) and `bpe_ranks` (an index over it).
///
/// \contracts
/// `bpe_ranks` copies what it needs; the vocabulary may go after it is built. Lookups are O(1) on average.
///
/// \errors
/// `token_bytes` and `bpe_ranks::make`: `unsupported` for a kind that is not byte-level BPE, `bad_header` for a token
/// that is not in the byte-level alphabet or a merge that names no tokens.
///
/// \examples
/// \snippet vocabulary_demo.cpp vocabulary
///
/// \perf
/// Not benchmarked: built once per tokenizer.
///
/// \testing
/// tests/vocabulary_test.cpp (core's BPE over a vocabulary built here), demos/vocabulary_demo.cpp.
///
/// \deps
/// Core's `ml/bpe.hpp`.
///
/// \gotchas
/// Added and special tokens (`<|endoftext|>`) are written as plain UTF-8 in byte-level files, not in the alphabet:
/// `token_bytes` takes such a token as its UTF-8 bytes when its type is control or user-defined.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_vocabulary_hpp
#define snicholls_formats_vocabulary_hpp

#include <snicholls/formats/detail/byte_level.hpp>
#include <snicholls/formats/error.hpp>
#include <snicholls/formats/gguf/keys.hpp>

#include <snicholls/core/ml/bpe.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace snicholls::formats
{
/// \brief What kind of tokenizer.
enum class tokenizer_kind : std::uint8_t
{
    byte_level_bpe,   ///< BPE over bytes, written in GPT-2's alphabet (GGUF "gpt2").
    sentencepiece,    ///< SentencePiece BPE with byte fallback (GGUF "llama"): read only, no encoder yet.
    wordpiece,        ///< WordPiece (GGUF "bert"): read only, no encoder yet.
    unigram,          ///< Unigram (GGUF "t5"): read only, no encoder yet.
};

/// \brief GGUF's `tokenizer.ggml.model` name of a kind. \param k The kind. \return "gpt2", "llama", "bert" or "t5".
[[nodiscard]] constexpr std::string_view gguf_tokenizer_model(tokenizer_kind k) noexcept
{
    switch (k) {
    case tokenizer_kind::byte_level_bpe: return "gpt2";
    case tokenizer_kind::sentencepiece: return "llama";
    case tokenizer_kind::wordpiece: return "bert";
    case tokenizer_kind::unigram: return "t5";
    }
    return "?";
}

/// \brief A tokenizer's data.
struct vocabulary
{
    tokenizer_kind kind = tokenizer_kind::byte_level_bpe;   ///< The kind.
    std::string pre;                                        ///< The pre-tokenizer's name ("gpt-2", "llama-bpe", ...), if known.
    std::vector<std::string> tokens;                        ///< Each token as written, by id.
    std::vector<gguf::token_type> types;                    ///< Each token's type (empty: all normal).
    std::vector<float> scores;                              ///< Each token's score (SentencePiece, Unigram; else empty).
    std::vector<std::string> merges;                        ///< BPE merges "left right", in rank order.
    std::optional<std::uint32_t> bos;                       ///< Beginning of sequence.
    std::optional<std::uint32_t> eos;                       ///< End of sequence.
    std::optional<std::uint32_t> unk;                       ///< Unknown.
    std::optional<std::uint32_t> pad;                       ///< Padding.

    /// \brief A token's type. \param id The token. \return Its type (normal if not recorded).
    [[nodiscard]] gguf::token_type type_of(std::uint32_t id) const noexcept
    {
        return id < types.size() ? types[id] : gguf::token_type::normal;
    }

    /// \brief Each token's raw bytes, by id (byte-level BPE only). \return The bytes, or `unsupported` /
    /// `bad_header`.
    [[nodiscard]] result<std::vector<std::string>> token_bytes() const
    {
        if (kind != tokenizer_kind::byte_level_bpe)
            return fail(errc::unsupported, "vocabulary: token bytes need a byte-level BPE vocabulary; "
                                           + std::string(gguf_tokenizer_model(kind)) + " waits for core's tokenizer track");
        std::vector<std::string> out;
        out.reserve(tokens.size());
        for (std::uint32_t id = 0; id < tokens.size(); ++id) {
            if (auto b = detail::byte_level_decode(tokens[id])) {
                out.push_back(std::move(*b));
                continue;
            }
            const auto t = type_of(id);
            if (t != gguf::token_type::control && t != gguf::token_type::user_defined && t != gguf::token_type::unused)
                return fail(errc::bad_header, "vocabulary: token " + std::to_string(id) + " " + quote_name(tokens[id]) + " is not byte-level");
            out.push_back(tokens[id]);   // an added token is its UTF-8 text
        }
        return ok(std::move(out));
    }
};

/// \brief Core's `rank(bytes)` over a byte-level vocabulary: a token's id, or `ml::bpe_no_rank`.
class bpe_ranks
{
public:
    /// \brief Indexes a vocabulary. \param v A byte-level BPE vocabulary. \return The index, or `token_bytes`' errors,
    /// or `bad_header` for a merge that names unknown tokens.
    [[nodiscard]] static result<bpe_ranks> make(const vocabulary& v)
    {
        SNICHOLLS_TRY_ASSIGN(std::vector<std::string> bytes, v.token_bytes());
        bpe_ranks r;
        r.ids_.reserve(bytes.size());
        for (std::uint32_t id = 0; id < bytes.size(); ++id)
            if (v.type_of(id) == gguf::token_type::normal || v.type_of(id) == gguf::token_type::byte)
                r.ids_.emplace(bytes[id], id);   // the first id of a duplicated text wins
        // Merged tokens' ids must increase with the merge's rank for id-as-rank to be Hugging Face's merge order
        std::int64_t last = -1;
        for (const auto& m : v.merges) {
            const auto space = m.find(' ');
            if (space == std::string::npos)
                return fail(errc::bad_header, "vocabulary: merge " + quote_name(m) + " is not \"left right\"");
            const auto left = detail::byte_level_decode(std::string_view(m).substr(0, space));
            const auto right = detail::byte_level_decode(std::string_view(m).substr(space + 1));
            if (!left || !right)
                return fail(errc::bad_header, "vocabulary: merge " + quote_name(m) + " is not byte-level");
            const auto it = r.ids_.find(*left + *right);
            if (it == r.ids_.end() || r.ids_.find(*left) == r.ids_.end() || r.ids_.find(*right) == r.ids_.end())
                return fail(errc::bad_header, "vocabulary: merge " + quote_name(m) + " names a token not in the vocabulary");
            if (static_cast<std::int64_t>(it->second) <= last)
                r.ordered_ = false;
            last = it->second;
        }
        return ok(std::move(r));
    }

    /// \brief The id of a byte string. \param bytes The bytes. \return The id, or `ml::bpe_no_rank`.
    [[nodiscard]] std::uint32_t operator()(std::string_view bytes) const
    {
        const auto it = ids_.find(bytes);
        return it == ids_.end() ? ml::bpe_no_rank : it->second;
    }
    /// \brief Whether merged tokens' ids follow the merge order, so core's id-as-rank merge is the file's.
    /// \return True if they do.
    [[nodiscard]] bool ids_follow_merges() const noexcept { return ordered_; }
    /// \brief The tokens indexed. \return Their count.
    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }

private:
    struct hash
    {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
    };
    std::unordered_map<std::string, std::uint32_t, hash, std::equal_to<>> ids_;
    bool ordered_ = true;
};
} // namespace snicholls::formats

#endif // snicholls_formats_vocabulary_hpp
