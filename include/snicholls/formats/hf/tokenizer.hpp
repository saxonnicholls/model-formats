//
//  tokenizer.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Hugging Face's `tokenizer.json` and `tokenizer_config.json` to a `vocabulary`, and the chat template.
/// \ingroup hf
///
/// \overview
/// `parse_tokenizer` reads the `tokenizers` library's file: the model's type (BPE, WordPiece, Unigram), its vocabulary
/// and merges (either form: `"a b"` strings or `["a", "b"]` pairs), the added tokens (special ones become control
/// tokens) and which alphabet BPE uses (a ByteLevel pre-tokenizer or decoder means byte-level; `byte_fallback` means
/// SentencePiece). `parse_tokenizer_config` adds `tokenizer_config.json`: the special tokens by name, whether BOS and EOS
/// are added, and the chat template, exposed as text (a Jinja template; nothing here renders it).
///
/// \relations
/// \see formats::vocabulary, formats::bpe_ranks, formats::gguf::write_vocabulary
///
/// \impls
/// Two parsers into one `tokenizer` struct.
///
/// \contracts
/// The vocabulary's `tokens` cover every id from 0 to the largest; an id no token names is `"[PAD<id>]"`, unused, as
/// llama.cpp's converter writes it.
///
/// \errors
/// `bad_json`, `bad_header` (a wrong shape, a token id twice), `unsupported` (a model type not listed).
///
/// \examples
/// \snippet tokenizer_demo.cpp tokenizer
///
/// \perf
/// Not benchmarked: read once per model.
///
/// \testing
/// tests/hf/tokenizer_test.cpp, demos/hf/tokenizer_demo.cpp.
///
/// \deps
/// nlohmann/json through core-adapters.
///
/// \gotchas
/// The pre-tokenizer's split pattern is not read: core's tokenizer owns the patterns it supports.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_hf_tokenizer_hpp
#define snicholls_formats_hf_tokenizer_hpp

#include <snicholls/formats/detail/json.hpp>
#include <snicholls/formats/vocabulary.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::formats::hf
{
/// \brief A named chat template.
struct chat_template
{
    std::string name;   ///< "default", "tool_use", ...
    std::string text;   ///< The Jinja source, not rendered.
};

/// \brief A Hugging Face tokenizer's contents.
struct tokenizer
{
    formats::vocabulary vocab;                   ///< Tokens, merges, types, special ids.
    std::vector<chat_template> chat_templates;   ///< From tokenizer_config.json (the first is the default).
    std::optional<bool> add_bos;                 ///< `add_bos_token`.
    std::optional<bool> add_eos;                 ///< `add_eos_token`.

    /// \brief The default chat template. \return Its text, or null if none.
    [[nodiscard]] const std::string* default_chat_template() const noexcept
    {
        for (const auto& t : chat_templates)
            if (t.name == "default")
                return &t.text;
        return chat_templates.empty() ? nullptr : &chat_templates.front().text;
    }
    /// \brief A token's id by its text. \param text The token as written. \return The id, or none.
    [[nodiscard]] std::optional<std::uint32_t> id_of(std::string_view text) const noexcept
    {
        for (std::uint32_t i = 0; i < vocab.tokens.size(); ++i)
            if (vocab.tokens[i] == text)
                return i;
        return std::nullopt;
    }
};

namespace detail
{
    using formats::detail::json;
    // Whether a component (pre_tokenizer, decoder) is, or contains, one of type `type`
    inline bool has_type(const json* j, std::string_view type)
    {
        if (!j || !j->is_object())
            return false;
        if (const std::string* t = formats::detail::string_at(*j, "type"); t && *t == type)
            return true;
        for (const char* list : {"pretokenizers", "decoders", "processors"})
            if (const json* l = formats::detail::member(*j, list); l && l->is_array())
                for (const auto& e : *l)
                    if (has_type(&e, type))
                        return true;
        return false;
    }
    inline result<void> set_token(vocabulary& v, std::uint64_t id, std::string text, gguf::token_type type)
    {
        if (id > 0x7fffffffu)
            return fail(errc::bad_header, "tokenizer.json: token id " + std::to_string(id) + " is too large");
        if (v.tokens.size() <= id) {
            const std::size_t from = v.tokens.size();
            v.tokens.resize(static_cast<std::size_t>(id) + 1);
            v.types.resize(static_cast<std::size_t>(id) + 1, gguf::token_type::unused);
            for (std::size_t i = from; i < v.tokens.size(); ++i)
                v.tokens[i] = "[PAD" + std::to_string(i) + "]";
        } else if (v.types[id] != gguf::token_type::unused) {
            if (v.tokens[id] == text)
                return ok();   // an added token that repeats the vocabulary's entry
            return fail(errc::bad_header, "tokenizer.json: id " + std::to_string(id) + " is given twice");
        }
        v.tokens[id] = std::move(text);
        v.types[id] = type;
        return ok();
    }
    // A special token named in tokenizer_config.json: a string, or {"content": ...}
    inline const std::string* token_text(const json& cfg, std::string_view key)
    {
        const json* m = formats::detail::member(cfg, key);
        if (!m)
            return nullptr;
        if (const std::string* s = formats::detail::as_string(*m))
            return s;
        return formats::detail::string_at(*m, "content");
    }
} // namespace detail

/// \brief Reads a `tokenizer.json`. \param text The JSON. \return The tokenizer, or the problem.
[[nodiscard]] inline result<tokenizer> parse_tokenizer(std::string_view text)
{
    using formats::detail::member;
    using formats::detail::string_at;
    SNICHOLLS_TRY_ASSIGN(const detail::json j, formats::detail::parse_json(text, "tokenizer.json"));
    const detail::json* model = member(j, "model");
    if (!model || !model->is_object())
        return fail(errc::bad_header, "tokenizer.json: no model");
    const std::string* type = string_at(*model, "type");
    const std::string t = type ? *type : "BPE";   // tokenizers omits the type of a BPE model in old files
    tokenizer tok;
    auto& v = tok.vocab;
    if (t == "BPE") {
        const bool fallback = formats::detail::bool_at(*model, "byte_fallback").value_or(false);
        const bool byte_level = detail::has_type(member(j, "pre_tokenizer"), "ByteLevel") || detail::has_type(member(j, "decoder"), "ByteLevel");
        if (!byte_level && !fallback)
            return fail(errc::unsupported, "tokenizer.json: a BPE model that is neither byte-level nor SentencePiece-style");
        v.kind = byte_level ? tokenizer_kind::byte_level_bpe : tokenizer_kind::sentencepiece;
    } else if (t == "WordPiece") {
        v.kind = tokenizer_kind::wordpiece;
    } else if (t == "Unigram") {
        v.kind = tokenizer_kind::unigram;
    } else {
        return fail(errc::unsupported, "tokenizer.json: model type \"" + t + "\" is not read");
    }
    const detail::json* vocab = member(*model, "vocab");
    if (v.kind == tokenizer_kind::unigram) {
        if (!vocab || !vocab->is_array())
            return fail(errc::bad_header, "tokenizer.json: a Unigram vocab is a list of [token, score]");
        std::uint64_t id = 0;
        for (const auto& e : *vocab) {
            const std::string* s = e.is_array() && e.size() == 2 ? formats::detail::as_string(e[0]) : nullptr;
            const auto score = s ? formats::detail::as_double(e[1]) : std::nullopt;
            if (!s || !score)
                return fail(errc::bad_header, "tokenizer.json: a Unigram entry is not [token, score]");
            SNICHOLLS_TRY_VOID(detail::set_token(v, id++, *s, gguf::token_type::normal));
            v.scores.push_back(static_cast<float>(*score));
        }
        if (const auto unk = formats::detail::u64_at(*model, "unk_id"); unk && *unk <= 0xffffffffu)
            v.unk = static_cast<std::uint32_t>(*unk);
    } else {
        if (!vocab || !vocab->is_object())
            return fail(errc::bad_header, "tokenizer.json: no vocab object");
        for (const auto& [k, idv] : vocab->items()) {
            const auto id = formats::detail::as_u64(idv);
            if (!id)
                return fail(errc::bad_header, "tokenizer.json: the id of " + quote_name(k) + " is not a non-negative integer");
            gguf::token_type type = gguf::token_type::normal;
            if (v.kind == tokenizer_kind::sentencepiece && k.size() == 6 && k.starts_with("<0x") && k.back() == '>')
                type = gguf::token_type::byte;
            SNICHOLLS_TRY_VOID(detail::set_token(v, *id, k, type));
        }
    }
    if (const detail::json* merges = member(*model, "merges"); merges && merges->is_array()) {
        v.merges.reserve(merges->size());
        for (const auto& m : *merges) {
            if (const std::string* s = formats::detail::as_string(m)) {
                v.merges.push_back(*s);
            } else if (m.is_array() && m.size() == 2 && m[0].is_string() && m[1].is_string()) {
                v.merges.push_back(*formats::detail::as_string(m[0]) + " " + *formats::detail::as_string(m[1]));
            } else {
                return fail(errc::bad_header, "tokenizer.json: a merge is neither \"a b\" nor [\"a\", \"b\"]");
            }
        }
    }
    if (const std::string* unk = string_at(*model, "unk_token"))
        v.unk = tok.id_of(*unk);
    if (const detail::json* added = member(j, "added_tokens"); added && added->is_array()) {
        for (const auto& a : *added) {
            const auto id = formats::detail::u64_at(a, "id");
            const std::string* content = string_at(a, "content");
            if (!id || !content)
                return fail(errc::bad_header, "tokenizer.json: an added token without id or content");
            const bool special = formats::detail::bool_at(a, "special").value_or(false);
            const auto type = special ? gguf::token_type::control : gguf::token_type::user_defined;
            if (*id < v.tokens.size() && v.tokens[*id] == *content) {
                v.types[*id] = type;
                continue;
            }
            SNICHOLLS_TRY_VOID(detail::set_token(v, *id, *content, type));
        }
    }
    return ok(std::move(tok));
}

/// \brief Adds `tokenizer_config.json`: special tokens, BOS / EOS flags and the chat templates.
/// \param tok The tokenizer from `parse_tokenizer`. \param text The JSON. \return Nothing, or `bad_json` /
/// `bad_header`.
[[nodiscard]] inline result<void> parse_tokenizer_config(tokenizer& tok, std::string_view text)
{
    SNICHOLLS_TRY_ASSIGN(const detail::json j, formats::detail::parse_json(text, "tokenizer_config.json"));
    if (!j.is_object())
        return fail(errc::bad_header, "tokenizer_config.json: not an object");
    tok.add_bos = formats::detail::bool_at(j, "add_bos_token");
    tok.add_eos = formats::detail::bool_at(j, "add_eos_token");
    const auto special = [&](std::string_view key, std::optional<std::uint32_t>& out) {
        if (const std::string* s = detail::token_text(j, key))
            if (const auto id = tok.id_of(*s))
                out = id;
    };
    special("bos_token", tok.vocab.bos);
    special("eos_token", tok.vocab.eos);
    special("unk_token", tok.vocab.unk);
    special("pad_token", tok.vocab.pad);
    if (const detail::json* ct = formats::detail::member(j, "chat_template")) {
        if (const std::string* s = formats::detail::as_string(*ct)) {
            tok.chat_templates.push_back({"default", *s});
        } else if (ct->is_array()) {
            for (const auto& e : *ct) {
                const std::string* name = formats::detail::string_at(e, "name");
                const std::string* tpl = formats::detail::string_at(e, "template");
                if (!name || !tpl)
                    return fail(errc::bad_header, "tokenizer_config.json: a chat template without name or template");
                tok.chat_templates.push_back({*name, *tpl});
            }
        } else if (!ct->is_null()) {
            return fail(errc::bad_header, "tokenizer_config.json: chat_template is neither a string nor a list");
        }
    }
    return ok();
}
} // namespace snicholls::formats::hf

#endif // snicholls_formats_hf_tokenizer_hpp
