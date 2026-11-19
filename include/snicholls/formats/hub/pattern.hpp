//
//  pattern.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief File patterns for choosing what to download (`--include "*.safetensors"`), as `huggingface_hub` matches them.
/// \ingroup hub
///
/// \overview
/// `huggingface_hub`'s `allow_patterns` / `ignore_patterns` are Python `fnmatch` patterns over the file's path in the
/// repository: `*` matches any run of characters, `/` included; `?` one character; `[abc]`, `[a-z]` and `[!abc]` a
/// set; and a pattern ending in `/` matches everything under that directory. `matches` implements exactly that, and
/// `select` applies include and exclude lists (no include list means everything).
///
/// \relations
/// \see formats::hub::client::download
///
/// \impls
/// A backtracking matcher, linear in practice for these patterns.
///
/// \contracts
/// `constexpr`, `noexcept`, no allocation.
///
/// \errors
/// None: an unclosed `[` matches a literal `[`, as fnmatch does.
///
/// \examples
/// \snippet pattern_demo.cpp pattern
///
/// \perf
/// Not benchmarked: a few patterns over a few files.
///
/// \testing
/// tests/hub/pattern_test.cpp (fnmatch's cases), demos/hub/pattern_demo.cpp.
///
/// \deps
/// The standard library.
///
/// \gotchas
/// Matching is case-sensitive (fnmatch's is on POSIX).
///
/// \status
/// Experimental.

#ifndef snicholls_formats_hub_pattern_hpp
#define snicholls_formats_hub_pattern_hpp

#include <span>
#include <string>
#include <string_view>

namespace snicholls::formats::hub
{
namespace detail
{
    // The set at p[i] ('[' already seen); its end in `end`, whether c is in it in `hit`; false if unclosed
    constexpr bool match_set(std::string_view p, std::size_t i, char c, std::size_t& end, bool& hit) noexcept
    {
        std::size_t j = i;
        bool negate = false;
        if (j < p.size() && (p[j] == '!' || p[j] == '^')) {
            negate = true;
            ++j;
        }
        bool in = false;
        bool first = true;
        while (j < p.size() && (p[j] != ']' || first)) {
            first = false;
            if (j + 2 < p.size() && p[j + 1] == '-' && p[j + 2] != ']') {
                in = in || (c >= p[j] && c <= p[j + 2]);
                j += 3;
            } else {
                in = in || c == p[j];
                ++j;
            }
        }
        if (j >= p.size())
            return false;
        end = j + 1;
        hit = in != negate;
        return true;
    }
} // namespace detail

/// \brief Whether a path matches a pattern. \param pattern The fnmatch pattern. \param path The path.
/// \return True if it matches.
[[nodiscard]] constexpr bool matches(std::string_view pattern, std::string_view path) noexcept
{
    if (!pattern.empty() && pattern.back() == '/')   // a directory: everything below it
        return path.starts_with(pattern);
    std::size_t p = 0, s = 0, star = std::string_view::npos, resume = 0;
    while (s < path.size()) {
        if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            resume = s;
            continue;
        }
        if (p < pattern.size()) {
            if (pattern[p] == '?') {
                ++p;
                ++s;
                continue;
            }
            if (pattern[p] == '[') {
                std::size_t end = 0;
                bool hit = false;
                if (detail::match_set(pattern, p + 1, path[s], end, hit)) {
                    if (hit) {
                        p = end;
                        ++s;
                        continue;
                    }
                } else if (path[s] == '[') {   // unclosed: a literal
                    ++p;
                    ++s;
                    continue;
                }
            } else if (pattern[p] == path[s]) {
                ++p;
                ++s;
                continue;
            }
        }
        if (star == std::string_view::npos)
            return false;
        p = star + 1;
        s = ++resume;
    }
    while (p < pattern.size() && pattern[p] == '*')
        ++p;
    return p == pattern.size();
}

/// \brief Whether a path is chosen. \param path The path. \param include Patterns to keep (empty: everything).
/// \param exclude Patterns to drop. \return True if some include matches and no exclude does.
[[nodiscard]] constexpr bool select(std::string_view path, std::span<const std::string> include, std::span<const std::string> exclude = {}) noexcept
{
    bool in = include.empty();
    for (const auto& p : include)
        in = in || matches(p, path);
    for (const auto& p : exclude)
        in = in && !matches(p, path);
    return in;
}
} // namespace snicholls::formats::hub

#endif // snicholls_formats_hub_pattern_hpp
