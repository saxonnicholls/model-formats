//
//  json.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  JSON for the readers, through nlohmann/json (core-adapters' snicholls::core_json): parsing without exceptions,
//  with every duplicate key an error (a header with two entries of one name is how a validator and a loader come to
//  disagree), typed accessors that return results, and serialisation that cannot throw. No nlohmann exception
//  escapes a function here.
//

#ifndef snicholls_formats_detail_json_hpp
#define snicholls_formats_detail_json_hpp

#include <snicholls/formats/error.hpp>

#include <snicholls/core_adapters/nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::detail
{
using json = nlohmann::json;

// Parses `text`; `what` names the document in a message. Duplicate keys at any depth are errors.
inline result<json> parse_json(std::string_view text, std::string_view what)
{
    std::vector<std::set<std::string>> keys;
    std::string duplicate;
    const json::parser_callback_t cb = [&](int, json::parse_event_t event, json& parsed) {
        switch (event) {
        case json::parse_event_t::object_start: keys.emplace_back(); break;
        case json::parse_event_t::object_end:
            if (!keys.empty())
                keys.pop_back();
            break;
        case json::parse_event_t::key:
            if (const auto* k = parsed.get_ptr<const json::string_t*>(); k && !keys.empty() && !keys.back().insert(*k).second && duplicate.empty())
                duplicate = *k;
            break;
        default: break;
        }
        return true;
    };
    json j;
    try {
        j = json::parse(text.begin(), text.end(), cb, false);
    } catch (...) {   // allow_exceptions = false covers parse errors; this covers anything else (bad_alloc aside)
        return fail(errc::bad_json, std::string(what) + ": does not parse");
    }
    if (j.is_discarded())
        return fail(errc::bad_json, std::string(what) + ": does not parse");
    if (!duplicate.empty())
        return fail(errc::duplicate_name, std::string(what) + ": duplicate key " + quote_name(duplicate));
    return ok(std::move(j));
}

// Serialises without throwing: invalid UTF-8 in a string is an error, not an exception
inline result<std::string> dump_json(const json& j, std::string_view what, int indent = -1)
{
    try {
        return ok(j.dump(indent, ' ', false, json::error_handler_t::strict));
    } catch (...) {
        return fail(errc::bad_argument, std::string(what) + ": a string is not valid UTF-8");
    }
}

inline const json* member(const json& j, std::string_view key)
{
    if (!j.is_object())
        return nullptr;
    const auto it = j.find(key);
    return it == j.end() ? nullptr : &*it;
}

inline std::optional<std::uint64_t> as_u64(const json& j)
{
    if (const auto* u = j.get_ptr<const json::number_unsigned_t*>())
        return *u;
    if (const auto* i = j.get_ptr<const json::number_integer_t*>(); i && *i >= 0)
        return static_cast<std::uint64_t>(*i);
    return std::nullopt;
}

inline std::optional<std::int64_t> as_i64(const json& j)
{
    if (const auto* i = j.get_ptr<const json::number_integer_t*>())
        return *i;
    if (const auto* u = j.get_ptr<const json::number_unsigned_t*>(); u && *u <= static_cast<std::uint64_t>(INT64_MAX))
        return static_cast<std::int64_t>(*u);
    return std::nullopt;
}

inline std::optional<double> as_double(const json& j)
{
    if (const auto* f = j.get_ptr<const json::number_float_t*>())
        return *f;
    if (const auto* i = j.get_ptr<const json::number_integer_t*>())
        return static_cast<double>(*i);
    if (const auto* u = j.get_ptr<const json::number_unsigned_t*>())
        return static_cast<double>(*u);
    return std::nullopt;
}

inline const std::string* as_string(const json& j) { return j.get_ptr<const json::string_t*>(); }

inline std::optional<bool> as_bool(const json& j)
{
    if (const auto* b = j.get_ptr<const json::boolean_t*>())
        return *b;
    return std::nullopt;
}

// A member of a given kind, if present and of that kind
inline std::optional<std::uint64_t> u64_at(const json& j, std::string_view key)
{
    const json* m = member(j, key);
    return m ? as_u64(*m) : std::nullopt;
}
inline std::optional<double> double_at(const json& j, std::string_view key)
{
    const json* m = member(j, key);
    return m ? as_double(*m) : std::nullopt;
}
inline std::optional<bool> bool_at(const json& j, std::string_view key)
{
    const json* m = member(j, key);
    return m ? as_bool(*m) : std::nullopt;
}
inline const std::string* string_at(const json& j, std::string_view key)
{
    const json* m = member(j, key);
    return m ? as_string(*m) : nullptr;
}
} // namespace snicholls::formats::detail

#endif // snicholls_formats_detail_json_hpp
