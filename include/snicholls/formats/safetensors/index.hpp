//
//  index.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The sharded-checkpoint index, `model.safetensors.index.json`: which shard holds which tensor.
/// \ingroup safetensors
///
/// \overview
/// A checkpoint too large for one file is split into shards (`model-00001-of-00004.safetensors` ...) and an index:
/// `{"metadata": {"total_size": N, ...}, "weight_map": {"<tensor>": "<shard file>"}}`. `parse_index` reads it,
/// refusing a shard name that could leave the directory (an absolute path, `..`, a separator other than `/` within
/// the name); `write_index` writes it with sorted keys; `shard_names` makes the conventional names.
///
/// \relations
/// \see formats::safetensors::parse, formats::io::model_dir (opens every shard)
///
/// \impls
/// One struct, three functions.
///
/// \contracts
/// `weight_map` is sorted by tensor name; `files()` lists each shard once, sorted.
///
/// \errors
/// `bad_json`, `duplicate_name`, `bad_header` (a wrong shape), `bad_argument` (an unsafe shard name).
///
/// \examples
/// \snippet index_demo.cpp index
///
/// \perf
/// Not benchmarked: read once per checkpoint.
///
/// \testing
/// tests/safetensors/index_test.cpp, demos/safetensors/index_demo.cpp.
///
/// \deps
/// nlohmann/json through core-adapters.
///
/// \gotchas
/// Metadata values other than `total_size` are kept as JSON text.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_safetensors_index_hpp
#define snicholls_formats_safetensors_index_hpp

#include <snicholls/formats/detail/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::formats::safetensors
{
/// \brief A sharded checkpoint's index.
struct index
{
    std::optional<std::uint64_t> total_size;                       ///< `metadata.total_size`, the bytes of all tensors.
    std::vector<std::pair<std::string, std::string>> metadata;     ///< Other metadata, each value as JSON text.
    std::vector<std::pair<std::string, std::string>> weight_map;   ///< Tensor name to shard file, sorted by name.

    /// \brief The shard of a tensor. \param tensor The tensor's name. \return The shard file, or null.
    [[nodiscard]] const std::string* shard_of(std::string_view tensor) const noexcept
    {
        const auto it = std::lower_bound(weight_map.begin(), weight_map.end(), tensor,
                                         [](const auto& e, std::string_view n) { return e.first < n; });
        return it != weight_map.end() && it->first == tensor ? &it->second : nullptr;
    }
    /// \brief Every shard, once. \return The shard files, sorted.
    [[nodiscard]] std::vector<std::string> files() const
    {
        std::vector<std::string> f;
        for (const auto& [t, s] : weight_map)
            f.push_back(s);
        std::sort(f.begin(), f.end());
        f.erase(std::unique(f.begin(), f.end()), f.end());
        return f;
    }
};

/// \brief Whether a file name is safe to join to a directory: relative, no `..`, no backslash, no empty part.
/// \param name The name. \return True if safe.
[[nodiscard]] constexpr bool safe_relative_name(std::string_view name) noexcept
{
    if (name.empty() || name.front() == '/' || name.find('\\') != std::string_view::npos || name.find('\0') != std::string_view::npos)
        return false;
    std::size_t at = 0;
    while (at <= name.size()) {
        const std::size_t slash = std::min(name.find('/', at), name.size());
        const std::string_view part = name.substr(at, slash - at);
        if (part.empty() || part == "." || part == "..")
            return false;
        at = slash + 1;
    }
    return true;
}

/// \brief Parses an index. \param text The JSON. \return The index, or the problem.
[[nodiscard]] inline result<index> parse_index(std::string_view text)
{
    SNICHOLLS_TRY_ASSIGN(const detail::json j, detail::parse_json(text, "safetensors index"));
    const detail::json* map = detail::member(j, "weight_map");
    if (!map || !map->is_object())
        return fail(errc::bad_header, "safetensors index: no weight_map object");
    index ix;
    for (const auto& [k, v] : map->items()) {
        const std::string* s = detail::as_string(v);
        if (!s)
            return fail(errc::bad_header, "safetensors index: shard of " + quote_name(k) + " is not a string");
        if (!safe_relative_name(*s))
            return fail(errc::bad_argument, "safetensors index: unsafe shard name " + quote_name(*s));
        ix.weight_map.emplace_back(k, *s);
    }
    std::sort(ix.weight_map.begin(), ix.weight_map.end());
    if (const detail::json* meta = detail::member(j, "metadata")) {
        if (!meta->is_object())
            return fail(errc::bad_header, "safetensors index: metadata is not an object");
        for (const auto& [k, v] : meta->items()) {
            if (k == "total_size") {
                ix.total_size = detail::as_u64(v);
                if (!ix.total_size)
                    return fail(errc::bad_header, "safetensors index: total_size is not a non-negative integer");
                continue;
            }
            SNICHOLLS_TRY_ASSIGN(std::string t, detail::dump_json(v, "safetensors index"));
            ix.metadata.emplace_back(k, std::move(t));
        }
    }
    return ok(std::move(ix));
}

/// \brief Writes an index. \param ix The index (metadata values must be JSON text). \return The JSON, indented by 2.
[[nodiscard]] inline result<std::string> write_index(const index& ix)
{
    detail::json j = detail::json::object();
    detail::json meta = detail::json::object();
    if (ix.total_size)
        meta["total_size"] = *ix.total_size;
    for (const auto& [k, v] : ix.metadata) {
        SNICHOLLS_TRY_ASSIGN(detail::json parsed, detail::parse_json(v, "safetensors index metadata"));
        meta[k] = std::move(parsed);
    }
    j["metadata"] = std::move(meta);
    detail::json map = detail::json::object();
    for (const auto& [t, s] : ix.weight_map) {
        if (!safe_relative_name(s))
            return fail(errc::bad_argument, "safetensors index: unsafe shard name " + quote_name(s));
        if (map.contains(t))
            return fail(errc::duplicate_name, "safetensors index: two entries for " + quote_name(t));
        map[t] = s;
    }
    j["weight_map"] = std::move(map);
    SNICHOLLS_TRY_ASSIGN(std::string out, detail::dump_json(j, "safetensors index", 2));
    return ok(out + "\n");
}

/// \brief The conventional shard names. \param stem "model". \param count The shards (at least 1).
/// \return `<stem>-00001-of-0000N.safetensors` ... (or `<stem>.safetensors` for one).
[[nodiscard]] inline std::vector<std::string> shard_names(std::string_view stem, std::size_t count)
{
    std::vector<std::string> names;
    if (count <= 1) {
        names.push_back(std::string(stem) + ".safetensors");
        return names;
    }
    for (std::size_t i = 1; i <= count; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "-%05zu-of-%05zu.safetensors", i, count);
        names.push_back(std::string(stem) + buf);
    }
    return names;
}
} // namespace snicholls::formats::safetensors

#endif // snicholls_formats_safetensors_index_hpp
