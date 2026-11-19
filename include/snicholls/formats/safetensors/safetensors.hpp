//
//  safetensors.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief safetensors: read (validated, zero-copy) and write (aligned, deterministic).
/// \ingroup safetensors
///
/// \overview
/// A safetensors file is an 8-byte little-endian header length N, N bytes of JSON, then the data. The JSON maps each
/// tensor name to `{"dtype", "shape", "data_offsets": [begin, end]}` (offsets into the data, which follows the
/// header) and may hold `"__metadata__"`, a map of strings to strings. `parse` reads the header and returns views into
/// the bytes it was given: nothing is copied, so parsing a mapped 10 GB file costs the header alone.
///
/// Validation is the reference implementation's and stricter: the header starts with `{` and is at most 100 MB;
/// every key appears once; every dtype is known; every extent product and byte count is computed without overflow and
/// equals `end - begin`; every range lies inside the data; ranges do not overlap; and, unless `allow_holes`, they
/// cover the data exactly, as the reference requires.
///
/// `write` sorts the tensors by element size (largest first) and then by name, so every tensor starts at a multiple of
/// its element size, pads the header with spaces to a multiple of 8, and emits the JSON with sorted keys: the same
/// tensors always give the same bytes.
///
/// \relations
/// \see formats::tensor_view, formats::safetensors::index (sharded checkpoints), formats::io::open_model
///
/// \impls
/// `parse` and `write` over bytes; `write_to` over any byte sink, so a caller can stream to a file.
///
/// \contracts
/// `parse` never reads outside its span and never throws for malformed input; the returned `file` borrows the bytes
/// for its tensors' `data`. `write` produces a file `parse` accepts.
///
/// \errors
/// `truncated`, `too_large` (header over the limit), `bad_header`, `bad_json`, `duplicate_name`, `bad_dtype`,
/// `bad_shape`, `out_of_bounds`, `overlap` (including holes). Writing: `duplicate_name`, `bad_shape` (data size does
/// not match), `bad_argument` (a name that is not valid UTF-8, or `__metadata__`).
///
/// \examples
/// \snippet safetensors_demo.cpp safetensors
///
/// \perf
/// benchmarks/safetensors_benchmark.cpp: parsing a header of many tensors.
///
/// \testing
/// tests/safetensors/safetensors_test.cpp (and the Python cross-check, scripts/crosscheck.sh),
/// demos/safetensors/safetensors_demo.cpp.
///
/// \deps
/// nlohmann/json through core-adapters.
///
/// \gotchas
/// Tensor names are JSON strings, so they are owned by the `file`, not views into the bytes.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_safetensors_safetensors_hpp
#define snicholls_formats_safetensors_safetensors_hpp

#include <snicholls/formats/detail/json.hpp>
#include <snicholls/formats/tensor.hpp>

#include <snicholls/core/endian.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::formats::safetensors
{
/// \brief One tensor of a parsed file.
struct entry
{
    std::string name;                  ///< The tensor's name.
    dtype type = dtype::f32;           ///< The element type.
    tensor_shape shape;                ///< The shape.
    std::uint64_t begin = 0;           ///< The first byte, from the start of the data.
    std::uint64_t end = 0;             ///< One past the last byte.
    std::span<const std::byte> data;   ///< The bytes, inside the span given to `parse`.

    /// \brief The tensor as a view. \return The view (borrowing this entry's name).
    [[nodiscard]] tensor_view view() const noexcept { return {name, type, shape, data}; }
};

/// \brief A string-to-string metadata entry.
using metadata_entry = std::pair<std::string, std::string>;

/// \brief A parsed safetensors file.
struct file
{
    std::vector<entry> tensors;             ///< In the order of their data.
    std::vector<metadata_entry> metadata;   ///< `__metadata__`, sorted by key.
    std::uint64_t header_size = 0;          ///< N, the JSON's length.
    std::span<const std::byte> data;        ///< The data section.

    /// \brief A tensor by name. \param name The name. \return The entry, or null.
    [[nodiscard]] const entry* find(std::string_view name) const noexcept
    {
        for (const auto& t : tensors)
            if (t.name == name)
                return &t;
        return nullptr;
    }
    /// \brief A metadata value. \param key The key. \return The value, or null.
    [[nodiscard]] const std::string* meta(std::string_view key) const noexcept
    {
        for (const auto& [k, v] : metadata)
            if (k == key)
                return &v;
        return nullptr;
    }
    /// \brief Every tensor as a view. \return The views, in data order.
    [[nodiscard]] std::vector<tensor_view> views() const
    {
        std::vector<tensor_view> v;
        v.reserve(tensors.size());
        for (const auto& t : tensors)
            v.push_back(t.view());
        return v;
    }
};

/// \brief What `parse` accepts.
struct parse_options
{
    std::uint64_t max_header = 100'000'000;   ///< The largest JSON header accepted (the reference's limit).
    bool allow_holes = false;                 ///< Accept data not covered by any tensor (the reference refuses it).
};

/// \brief Parses and validates a safetensors file.
/// \param bytes The whole file. \param opt The limits.
/// \return The file (borrowing `bytes`), or the first problem found.
/// Complexity: O(H + T log T) for a header of H bytes and T tensors.
[[nodiscard]] inline result<file> parse(std::span<const std::byte> bytes, const parse_options& opt = {})
{
    if (bytes.size() < 8)
        return fail(errc::truncated, "safetensors: shorter than the 8-byte header length");
    const std::uint64_t n = load_le<std::uint64_t>(bytes.data());
    if (n > opt.max_header)
        return fail(errc::too_large, "safetensors: header of " + std::to_string(n) + " bytes is over the limit");
    if (n > bytes.size() - 8)
        return fail(errc::truncated, "safetensors: header runs past the end of the file");
    const std::string_view text(reinterpret_cast<const char*>(bytes.data() + 8), static_cast<std::size_t>(n));
    if (text.empty() || text.front() != '{')
        return fail(errc::bad_header, "safetensors: header does not start with '{'");
    SNICHOLLS_TRY_ASSIGN(const detail::json j, detail::parse_json(text, "safetensors header"));
    if (!j.is_object())
        return fail(errc::bad_header, "safetensors: header is not an object");

    file f;
    f.header_size = n;
    f.data = bytes.subspan(static_cast<std::size_t>(8 + n));
    for (const auto& [key, value] : j.items()) {
        if (key == "__metadata__") {
            if (!value.is_object())
                return fail(errc::bad_header, "safetensors: __metadata__ is not an object");
            for (const auto& [mk, mv] : value.items()) {
                const std::string* s = detail::as_string(mv);
                if (!s)
                    return fail(errc::bad_header, "safetensors: metadata value of " + quote_name(mk) + " is not a string");
                f.metadata.emplace_back(mk, *s);
            }
            continue;
        }
        if (!value.is_object())
            return fail(errc::bad_header, "safetensors: entry " + quote_name(key) + " is not an object");
        entry e;
        e.name = key;
        const std::string* dt = detail::string_at(value, "dtype");
        if (!dt)
            return fail(errc::bad_header, "safetensors: " + quote_name(key) + " has no dtype");
        const auto t = dtype_from_safetensors(*dt);
        if (!t)
            return fail(errc::bad_dtype, "safetensors: " + quote_name(key) + " has dtype " + quote_name(*dt));
        e.type = *t;
        const detail::json* shape = detail::member(value, "shape");
        if (!shape || !shape->is_array())
            return fail(errc::bad_header, "safetensors: " + quote_name(key) + " has no shape array");
        for (const auto& d : *shape) {
            const auto v = detail::as_u64(d);
            if (!v)
                return fail(errc::bad_shape, "safetensors: " + quote_name(key) + " has an extent that is not a non-negative integer");
            if (!e.shape.push_back(*v))
                return fail(errc::bad_shape, "safetensors: " + quote_name(key) + " has more than " + std::to_string(max_rank) + " dimensions");
        }
        const detail::json* offsets = detail::member(value, "data_offsets");
        if (!offsets || !offsets->is_array() || offsets->size() != 2)
            return fail(errc::bad_header, "safetensors: " + quote_name(key) + " has no [begin, end] data_offsets");
        const auto b = detail::as_u64((*offsets)[0]), en = detail::as_u64((*offsets)[1]);
        if (!b || !en || *b > *en)
            return fail(errc::bad_header, "safetensors: " + quote_name(key) + " has invalid data_offsets");
        e.begin = *b;
        e.end = *en;
        const auto size = checked_byte_size(e.shape, e.type);
        if (!size || *size != e.end - e.begin)
            return fail(errc::bad_shape, "safetensors: " + quote_name(key) + " has " + std::to_string(e.end - e.begin)
                                             + " bytes for shape " + to_string(e.shape));
        if (e.end > f.data.size())
            return fail(errc::out_of_bounds, "safetensors: " + quote_name(key) + " ends past the data");
        e.data = f.data.subspan(static_cast<std::size_t>(e.begin), static_cast<std::size_t>(e.end - e.begin));
        f.tensors.push_back(std::move(e));
    }
    std::sort(f.tensors.begin(), f.tensors.end(), [](const entry& a, const entry& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    std::uint64_t at = 0;
    for (const auto& e : f.tensors) {
        if (e.begin < at)
            return fail(errc::overlap, "safetensors: " + quote_name(e.name) + " overlaps the tensor before it");
        if (e.begin > at && !opt.allow_holes)
            return fail(errc::overlap, "safetensors: a hole in the data before " + quote_name(e.name));
        at = e.end;
    }
    if (at != f.data.size() && !opt.allow_holes)
        return fail(errc::overlap, "safetensors: " + std::to_string(f.data.size() - at) + " bytes after the last tensor");
    std::sort(f.metadata.begin(), f.metadata.end());
    return ok(std::move(f));
}

/// \brief The layout `write` gives: the header bytes (length prefix, JSON, padding) and the tensors in data order.
struct layout
{
    std::vector<std::byte> header;    ///< The first bytes of the file, up to the data.
    std::vector<tensor_view> order;   ///< The tensors in the order their bytes follow.
    std::uint64_t data_size = 0;      ///< The data section's length.
};

/// \brief Plans a file: checks the tensors, orders them, and builds the header.
/// \param tensors The tensors. \param metadata `__metadata__` (any order; written sorted).
/// \return The layout, or `duplicate_name`, `bad_shape` or `bad_argument`.
[[nodiscard]] inline result<layout> plan(std::span<const tensor_view> tensors, std::span<const metadata_entry> metadata = {})
{
    layout l;
    l.order.assign(tensors.begin(), tensors.end());
    std::sort(l.order.begin(), l.order.end(), [](const tensor_view& a, const tensor_view& b) {
        const std::size_t sa = size_of(a.type), sb = size_of(b.type);
        return sa != sb ? sa > sb : a.name < b.name;
    });
    detail::json j = detail::json::object();
    for (std::size_t i = 0; i < l.order.size(); ++i) {
        const auto& t = l.order[i];
        if (t.name == "__metadata__")
            return fail(errc::bad_argument, "safetensors: a tensor may not be called __metadata__");
        const auto size = checked_byte_size(t.shape, t.type);
        if (!size || *size != t.data.size())
            return fail(errc::bad_shape, "safetensors: " + quote_name(t.name) + " has " + std::to_string(t.data.size())
                                             + " bytes for shape " + to_string(t.shape));
        detail::json shape = detail::json::array();
        for (const auto d : t.shape.view())
            shape.push_back(d);
        const std::string name(t.name);
        if (j.contains(name))
            return fail(errc::duplicate_name, "safetensors: two tensors named " + quote_name(t.name));
        j[name] = {{"dtype", std::string(safetensors_name(t.type))}, {"shape", std::move(shape)},
                   {"data_offsets", {l.data_size, l.data_size + *size}}};
        l.data_size += *size;
    }
    if (!metadata.empty()) {
        detail::json m = detail::json::object();
        for (const auto& [k, v] : metadata)
            m[k] = v;
        j["__metadata__"] = std::move(m);
    }
    SNICHOLLS_TRY_ASSIGN(std::string text, detail::dump_json(j, "safetensors header"));
    while ((8 + text.size()) % 8 != 0)
        text.push_back(' ');
    l.header.resize(8 + text.size());
    store_le(l.header.data(), static_cast<std::uint64_t>(text.size()));
    std::memcpy(l.header.data() + 8, text.data(), text.size());
    return ok(std::move(l));
}

/// \brief Writes a file to a sink, a piece at a time.
/// \tparam Sink Callable with `std::span<const std::byte>`.
/// \param tensors The tensors. \param metadata `__metadata__`. \param sink Receives the file's bytes in order.
/// \return Nothing, or `plan`'s errors (then nothing was written).
template <typename Sink>
[[nodiscard]] result<void> write_to(std::span<const tensor_view> tensors, std::span<const metadata_entry> metadata, Sink&& sink)
{
    SNICHOLLS_TRY_ASSIGN(const layout l, plan(tensors, metadata));
    sink(std::span<const std::byte>(l.header));
    for (const auto& t : l.order)
        sink(t.data);
    return ok();
}

/// \brief Writes a file into memory. \param tensors The tensors. \param metadata `__metadata__`.
/// \return The file's bytes, or `plan`'s errors.
[[nodiscard]] inline result<std::vector<std::byte>> write(std::span<const tensor_view> tensors, std::span<const metadata_entry> metadata = {})
{
    std::vector<std::byte> out;
    SNICHOLLS_TRY_VOID(write_to(tensors, metadata, [&out](std::span<const std::byte> s) { out.insert(out.end(), s.begin(), s.end()); }));
    return ok(std::move(out));
}
} // namespace snicholls::formats::safetensors

#endif // snicholls_formats_safetensors_safetensors_hpp
