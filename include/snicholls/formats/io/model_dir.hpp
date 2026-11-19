//
//  model_dir.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief A Hugging Face model directory (or Hub snapshot): configuration, tokenizer, chat template and every
/// weight shard, opened at once.
/// \ingroup io
///
/// \overview
/// `open_model_dir` reads what a transformers checkpoint directory holds: `config.json` (to a `model_config`),
/// `generation_config.json`, `tokenizer.json` with `tokenizer_config.json` (to a `vocabulary` and the chat templates;
/// `chat_template.jinja` too, where newer repositories keep it), and the weights: `model.safetensors`, or
/// `model.safetensors.index.json` and its shards, or (only when there is no safetensors) `pytorch_model.bin` and its
/// index, read through the restricted unpickler. Every weight file is mapped; the tensors are views of the mappings.
///
/// A sharded checkpoint is checked against its index: every tensor the index names is in the shard it names, no
/// tensor is in two shards, and the shard names are safe (relative, no `..`).
///
/// \relations
/// \see formats::hf::parse_config, formats::hf::parse_tokenizer, formats::safetensors::index, formats::io::mapped_file,
/// formats::hub::client (downloads such a directory)
///
/// \impls
/// One function into one struct.
///
/// \contracts
/// The `model_dir` owns its mappings; its tensor views are valid while it lives, and moving it keeps them valid.
///
/// \errors
/// `not_found` (no config.json, no weights), `io`, the parsers' errors, `duplicate_name` (a tensor in two shards),
/// `bad_header` (a tensor missing from the shard its index names).
///
/// \examples
/// \snippet model_dir_demo.cpp model_dir
///
/// \perf
/// Not benchmarked: dominated by mapping the files.
///
/// \testing
/// tests/io/model_dir_test.cpp (a directory written by the test, sharded and not), demos/io/model_dir_demo.cpp.
///
/// \deps
/// ts-moveables, through `io/files.hpp`.
///
/// \gotchas
/// A configuration this library does not map (`model_type` "bert") still opens: `config` is then none and
/// `config_error` says why.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_io_model_dir_hpp
#define snicholls_formats_io_model_dir_hpp

#include <snicholls/formats/hf/config.hpp>
#include <snicholls/formats/hf/tokenizer.hpp>
#include <snicholls/formats/io/files.hpp>
#include <snicholls/formats/safetensors/index.hpp>

#include <deque>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace snicholls::formats::io
{
/// \brief An opened model directory.
struct model_dir
{
    std::filesystem::path root;                           ///< The directory.
    std::optional<model_config> config;                   ///< config.json, mapped onto core (none if not mappable).
    std::optional<error> config_error;                    ///< Why `config` is none, when config.json exists.
    std::optional<hf::generation_config> generation;      ///< generation_config.json.
    std::optional<hf::tokenizer> tokenizer;               ///< tokenizer.json (+ tokenizer_config.json, chat_template.jinja).
    std::deque<mapped_file> files;                        ///< The weight files, mapped.
    std::deque<convert::loaded> parts;                    ///< Each file's tensors.
    std::vector<tensor_view> tensors;                     ///< Every tensor of every shard.

    /// \brief A tensor by name. \param name Its name. \return The view, or null.
    [[nodiscard]] const tensor_view* find(std::string_view name) const noexcept
    {
        for (const auto& t : tensors)
            if (t.name == name)
                return &t;
        return nullptr;
    }
};

namespace detail
{
    inline std::optional<std::string> read_text(const std::filesystem::path& p)
    {
        auto r = read_file(p);
        return r.is_ok() ? std::optional<std::string>(std::move(*r)) : std::nullopt;
    }
} // namespace detail

/// \brief Opens a Hugging Face model directory. \param root The directory. \return The model, or the problem.
[[nodiscard]] inline result<model_dir> open_model_dir(const std::filesystem::path& root)
{
    model_dir m;
    m.root = root;
    const auto cfg = detail::read_text(root / "config.json");
    if (!cfg)
        return fail(errc::not_found, root.string() + ": no config.json");
    auto c = hf::parse_config(*cfg);
    if (c.is_ok())
        m.config = std::move(*c);
    else
        m.config_error = c.unwrap_err_unchecked();
    if (const auto g = detail::read_text(root / "generation_config.json")) {
        SNICHOLLS_TRY_ASSIGN(m.generation, hf::parse_generation_config(*g));
    }
    if (const auto t = detail::read_text(root / "tokenizer.json")) {
        SNICHOLLS_TRY_ASSIGN(hf::tokenizer tok, hf::parse_tokenizer(*t));
        if (const auto tc = detail::read_text(root / "tokenizer_config.json"))
            SNICHOLLS_TRY_VOID(hf::parse_tokenizer_config(tok, *tc));
        if (const auto j = detail::read_text(root / "chat_template.jinja"))
            tok.chat_templates.insert(tok.chat_templates.begin(), hf::chat_template{"default", *j});
        m.tokenizer = std::move(tok);
    }
    // The weights: safetensors first, PyTorch only when there is none
    std::optional<safetensors::index> ix;
    std::vector<std::string> shards;
    for (const char* index_name : {"model.safetensors.index.json", "pytorch_model.bin.index.json"}) {
        if (const auto text = detail::read_text(root / index_name)) {
            SNICHOLLS_TRY_ASSIGN(ix, safetensors::parse_index(*text));
            shards = ix->files();
            break;
        }
        const std::string single = std::string(index_name).substr(0, std::string(index_name).size() - 11);   // without ".index.json"
        if (std::filesystem::exists(root / single)) {
            shards = {single};
            break;
        }
    }
    if (shards.empty())
        return fail(errc::not_found, root.string() + ": no model.safetensors, pytorch_model.bin or index");
    std::set<std::string_view> seen;
    for (const auto& s : shards) {
        SNICHOLLS_TRY_ASSIGN(mapped_file f, mapped_file::open(root / s));
        m.files.push_back(std::move(f));
        auto l = convert::read_any(m.files.back().bytes());
        if (l.is_err())
            return err(error{l.unwrap_err_unchecked().code, (root / s).string() + ": " + l.unwrap_err_unchecked().message});
        m.parts.push_back(std::move(*l));
        for (const auto& t : m.parts.back().tensors) {
            if (!seen.insert(t.name).second)
                return fail(errc::duplicate_name, root.string() + ": " + quote_name(t.name) + " is in two shards");
            if (ix) {
                const std::string* where = ix->shard_of(t.name);
                if (!where || *where != s)
                    return fail(errc::bad_header, root.string() + ": " + quote_name(t.name) + " is in " + s + ", the index says "
                                                      + (where ? *where : std::string("nowhere")));
            }
            m.tensors.push_back(t);
        }
    }
    if (ix)
        for (const auto& [t, s] : ix->weight_map)
            if (!seen.contains(t))
                return fail(errc::bad_header, root.string() + ": the index names " + quote_name(t) + " in " + s + ", which lacks it");
    return ok(std::move(m));
}
} // namespace snicholls::formats::io

#endif // snicholls_formats_io_model_dir_hpp
