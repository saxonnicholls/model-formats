//
//  gguf.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief GGUF (versions 2 and 3): read (validated, zero-copy) and write, every metadata type, arrays, tensors.
/// \ingroup gguf
///
/// \overview
/// A GGUF file is a header (magic `GGUF`, version, tensor and metadata counts), the metadata (key, type, value), the
/// tensor infos (name, dimensions innermost first, ggml type, offset), padding to the alignment (`general.alignment`,
/// default 32), then the tensor data, each tensor at an aligned offset. llama.cpp's docs/gguf.md is the reference.
///
/// `parse` returns a `file` whose keys, strings and tensor data are views into the bytes given: nothing is copied, so
/// opening a mapped model costs the metadata walk alone. A `value` is decoded on access; an array of strings is walked
/// once by `strings()`. Validation: the magic and version; every length, count and offset checked against the bytes
/// left before it is used (a 2^64 count is refused, not allocated); at most 4 dimensions; every row a whole number of
/// blocks; every tensor aligned, inside the data, and not overlapping another; keys and tensor names unique; nested
/// arrays at most 8 deep.
///
/// `writer` builds a file: `set` keeps insertion order (replacing a key in place), `add_tensor` takes the shape
/// outermost first (as every other format here) and writes ggml's order, and the output is deterministic.
///
/// \relations
/// \see formats::gguf::ggml_type, formats::gguf::keys, formats::convert::model_config (GGUF metadata to core's
/// `transformer_config`)
///
/// \impls
/// `value` (a view) for reading, `metadata_value` (owned) for writing; `to_owned` converts.
///
/// \contracts
/// `parse` never reads outside its span and never throws on malformed input. Big-endian GGUF is refused.
///
/// \errors
/// `truncated`, `bad_magic`, `unsupported_version`, `bad_header` (an unknown value type, a bad alignment, too deep),
/// `too_large`, `bad_dtype` (an unknown ggml type), `bad_shape`, `misaligned`, `out_of_bounds`, `overlap`,
/// `duplicate_name`. Writing: `duplicate_name`, `bad_shape`, `bad_argument`.
///
/// \examples
/// \snippet gguf_demo.cpp gguf
///
/// \perf
/// Not benchmarked yet: the walk is linear in the metadata.
///
/// \testing
/// tests/gguf/gguf_test.cpp (and the gguf-py cross-check, scripts/crosscheck.sh), demos/gguf/gguf_demo.cpp.
///
/// \deps
/// The standard library and core.
///
/// \gotchas
/// Keys and strings are bytes, not checked as UTF-8. Version 1 (32-bit counts) is not read.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_gguf_gguf_hpp
#define snicholls_formats_gguf_gguf_hpp

#include <snicholls/formats/gguf/ggml_types.hpp>
#include <snicholls/formats/tensor.hpp>

#include <snicholls/core/endian.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::formats::gguf
{
/// \brief GGUF's magic, the bytes "GGUF" read as a little-endian u32.
inline constexpr std::uint32_t magic = 0x46554747;
/// \brief The version written.
inline constexpr std::uint32_t current_version = 3;
/// \brief The alignment when `general.alignment` is absent.
inline constexpr std::uint32_t default_alignment = 32;
/// \brief ggml's limit on dimensions.
inline constexpr std::size_t max_dims = 4;

/// \brief A metadata value's type, by its GGUF code.
enum class value_type : std::uint32_t
{
    u8 = 0,        ///< uint8.
    i8 = 1,        ///< int8.
    u16 = 2,       ///< uint16.
    i16 = 3,       ///< int16.
    u32 = 4,       ///< uint32.
    i32 = 5,       ///< int32.
    f32 = 6,       ///< float32.
    boolean = 7,   ///< bool, one byte.
    string = 8,    ///< u64 length, then bytes.
    array = 9,     ///< element type, u64 count, then the elements.
    u64 = 10,      ///< uint64.
    i64 = 11,      ///< int64.
    f64 = 12,      ///< float64.
};

/// \brief GGUF's name of a value type. \param t The type. \return "UINT8", ..., "FLOAT64".
[[nodiscard]] constexpr std::string_view to_string(value_type t) noexcept
{
    constexpr std::string_view names[] = {"UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32",
                                          "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64"};
    const auto i = static_cast<std::uint32_t>(t);
    return i < 13 ? names[i] : "?";
}

/// \brief The bytes of a fixed-size value type. \param t The type. \return 1 to 8, or 0 for a string or array.
[[nodiscard]] constexpr std::size_t fixed_size(value_type t) noexcept
{
    switch (t) {
    case value_type::u8: case value_type::i8: case value_type::boolean: return 1;
    case value_type::u16: case value_type::i16: return 2;
    case value_type::u32: case value_type::i32: case value_type::f32: return 4;
    case value_type::u64: case value_type::i64: case value_type::f64: return 8;
    default: return 0;
    }
}

/// \brief A metadata value, borrowed from the file's bytes.
struct value
{
    value_type type = value_type::u8;              ///< The type.
    std::span<const std::byte> bytes;              ///< A scalar's bytes; a string's bytes; an array's elements.
    value_type element_type = value_type::u8;      ///< An array's element type.
    std::uint64_t count = 0;                       ///< An array's element count.

    /// \brief An unsigned integer (any width; a non-negative signed one too). \return The value, or none.
    [[nodiscard]] std::optional<std::uint64_t> as_u64() const noexcept
    {
        switch (type) {
        case value_type::u8: return std::to_integer<std::uint8_t>(bytes[0]);
        case value_type::u16: return load_le<std::uint16_t>(bytes.data());
        case value_type::u32: return load_le<std::uint32_t>(bytes.data());
        case value_type::u64: return load_le<std::uint64_t>(bytes.data());
        case value_type::i8: case value_type::i16: case value_type::i32: case value_type::i64: {
            const auto v = as_i64();
            return v && *v >= 0 ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(*v)) : std::nullopt;
        }
        default: return std::nullopt;
        }
    }
    /// \brief A signed integer (any width; an unsigned one that fits too). \return The value, or none.
    [[nodiscard]] std::optional<std::int64_t> as_i64() const noexcept
    {
        switch (type) {
        case value_type::i8: return static_cast<std::int8_t>(std::to_integer<std::uint8_t>(bytes[0]));
        case value_type::i16: return static_cast<std::int16_t>(load_le<std::uint16_t>(bytes.data()));
        case value_type::i32: return static_cast<std::int32_t>(load_le<std::uint32_t>(bytes.data()));
        case value_type::i64: return static_cast<std::int64_t>(load_le<std::uint64_t>(bytes.data()));
        case value_type::u8: case value_type::u16: case value_type::u32: case value_type::u64: {
            const auto v = as_u64();
            return v && *v <= static_cast<std::uint64_t>(INT64_MAX) ? std::optional<std::int64_t>(static_cast<std::int64_t>(*v)) : std::nullopt;
        }
        default: return std::nullopt;
        }
    }
    /// \brief A number as a double (floats, and integers). \return The value, or none.
    [[nodiscard]] std::optional<double> as_f64() const noexcept
    {
        if (type == value_type::f32)
            return static_cast<double>(std::bit_cast<float>(load_le<std::uint32_t>(bytes.data())));
        if (type == value_type::f64)
            return std::bit_cast<double>(load_le<std::uint64_t>(bytes.data()));
        if (const auto i = as_i64())
            return static_cast<double>(*i);
        if (const auto u = as_u64())
            return static_cast<double>(*u);
        return std::nullopt;
    }
    /// \brief A bool. \return The value, or none.
    [[nodiscard]] std::optional<bool> as_bool() const noexcept
    {
        if (type != value_type::boolean)
            return std::nullopt;
        return bytes[0] != std::byte{0};
    }
    /// \brief A string. \return Its bytes, or none.
    [[nodiscard]] std::optional<std::string_view> as_string() const noexcept
    {
        if (type != value_type::string)
            return std::nullopt;
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    /// \brief Element `i` of an array of a fixed-size type. \param i Below `count`. \return The element, or none (not
    /// such an array, or out of range).
    [[nodiscard]] std::optional<value> element(std::uint64_t i) const noexcept
    {
        const std::size_t s = fixed_size(element_type);
        if (type != value_type::array || s == 0 || i >= count)
            return std::nullopt;
        return value{element_type, bytes.subspan(static_cast<std::size_t>(i) * s, s), value_type::u8, 0};
    }
    /// \brief The elements of an array of strings. \return The strings (views), or none.
    [[nodiscard]] std::optional<std::vector<std::string_view>> strings() const
    {
        if (type != value_type::array || element_type != value_type::string)
            return std::nullopt;
        std::vector<std::string_view> out;
        out.reserve(static_cast<std::size_t>(count));
        std::size_t at = 0;
        for (std::uint64_t i = 0; i < count; ++i) {   // validated by parse: every length fits
            const auto n = static_cast<std::size_t>(load_le<std::uint64_t>(bytes.data() + at));
            out.emplace_back(reinterpret_cast<const char*>(bytes.data() + at + 8), n);
            at += 8 + n;
        }
        return out;
    }
    /// \brief The elements of a numeric array as doubles. \return The values, or none (not a numeric array).
    [[nodiscard]] std::optional<std::vector<double>> numbers() const
    {
        if (type != value_type::array || fixed_size(element_type) == 0 || element_type == value_type::boolean)
            return std::nullopt;
        std::vector<double> out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i)
            out.push_back(*element(i)->as_f64());
        return out;
    }
    /// \brief The elements of an integer array. \return The values, or none (not an integer array, or one that
    /// does not fit `int64_t`).
    [[nodiscard]] std::optional<std::vector<std::int64_t>> integers() const
    {
        if (type != value_type::array || fixed_size(element_type) == 0 || element_type == value_type::f32
            || element_type == value_type::f64 || element_type == value_type::boolean)
            return std::nullopt;
        std::vector<std::int64_t> out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto v = element(i)->as_i64();
            if (!v)
                return std::nullopt;
            out.push_back(*v);
        }
        return out;
    }
};

/// \brief One key and its value.
struct metadata_entry
{
    std::string_view key;   ///< The key.
    value val;              ///< The value.
};

/// \brief One tensor of a parsed file.
struct tensor_info
{
    std::string_view name;             ///< The tensor's name.
    ggml_type type = ggml_type::f32;   ///< Its ggml type.
    tensor_shape shape;                ///< The shape, outermost first (GGUF's dimensions reversed).
    std::uint64_t offset = 0;          ///< From the start of the data section.
    std::span<const std::byte> data;   ///< Its bytes.

    /// \brief The dimensions as ggml writes them, innermost first. \return `shape` reversed.
    [[nodiscard]] tensor_shape ne() const noexcept { return shape.reversed(); }
    /// \brief The tensor as a dense view. \return The view, or none for a quantised type.
    [[nodiscard]] std::optional<tensor_view> dense() const noexcept
    {
        const auto t = to_dtype(type);
        if (!t)
            return std::nullopt;
        return tensor_view{name, *t, shape, data};
    }
};

/// \brief A parsed GGUF file.
struct file
{
    std::uint32_t version = current_version;   ///< 2 or 3.
    std::vector<metadata_entry> metadata;       ///< In file order.
    std::vector<tensor_info> tensors;           ///< In file order.
    std::uint32_t alignment = default_alignment;   ///< The data alignment.
    std::uint64_t data_offset = 0;              ///< Where the data section starts.
    std::span<const std::byte> data;            ///< The data section.

    /// \brief A value by key. \param key The key. \return The value, or null.
    [[nodiscard]] const value* find(std::string_view key) const noexcept
    {
        for (const auto& e : metadata)
            if (e.key == key)
                return &e.val;
        return nullptr;
    }
    /// \brief A tensor by name. \param name The name. \return The tensor, or null.
    [[nodiscard]] const tensor_info* find_tensor(std::string_view name) const noexcept
    {
        for (const auto& t : tensors)
            if (t.name == name)
                return &t;
        return nullptr;
    }
    /// \brief `general.architecture`. \return It, or empty.
    [[nodiscard]] std::string_view architecture() const noexcept
    {
        const value* v = find("general.architecture");
        return v ? v->as_string().value_or(std::string_view{}) : std::string_view{};
    }
};

/// \brief What `parse` accepts.
struct parse_options
{
    std::uint64_t max_tensors = 1u << 20;      ///< The most tensor infos.
    std::uint64_t max_metadata = 1u << 20;     ///< The most metadata entries.
};

namespace detail
{
    class reader
    {
    public:
        explicit reader(std::span<const std::byte> b) noexcept : b_(b) {}
        std::size_t at() const noexcept { return at_; }
        std::size_t left() const noexcept { return b_.size() - at_; }
        bool take(std::size_t n, std::span<const std::byte>& out) noexcept
        {
            if (n > left())
                return false;
            out = b_.subspan(at_, n);
            at_ += n;
            return true;
        }
        template <typename T>
        bool get(T& v) noexcept
        {
            std::span<const std::byte> s;
            if (!take(sizeof(T), s))
                return false;
            v = load_le<T>(s.data());
            return true;
        }
        bool string(std::string_view& s) noexcept
        {
            std::uint64_t n = 0;
            std::span<const std::byte> bytes;
            if (!get(n) || n > left() || !take(static_cast<std::size_t>(n), bytes))
                return false;
            s = std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            return true;
        }

    private:
        std::span<const std::byte> b_;
        std::size_t at_ = 0;
    };

    // Reads a value of type `t` (after its type code); arrays recursively, at most `depth` deep
    inline result<value> read_value(reader& r, value_type t, int depth)
    {
        value v;
        v.type = t;
        if (const std::size_t s = fixed_size(t)) {
            if (!r.take(s, v.bytes))
                return fail(errc::truncated, "gguf: a value runs past the end");
            return ok(v);
        }
        if (t == value_type::string) {
            std::string_view s;
            if (!r.string(s))
                return fail(errc::truncated, "gguf: a string runs past the end");
            v.bytes = std::as_bytes(std::span(s.data(), s.size()));
            return ok(v);
        }
        if (t != value_type::array)
            return fail(errc::bad_header, "gguf: unknown value type " + std::to_string(static_cast<std::uint32_t>(t)));
        if (depth >= 8)
            return fail(errc::bad_header, "gguf: arrays nested more than 8 deep");
        std::uint32_t et = 0;
        if (!r.get(et) || !r.get(v.count))
            return fail(errc::truncated, "gguf: an array header runs past the end");
        if (et > 12)
            return fail(errc::bad_header, "gguf: unknown array element type " + std::to_string(et));
        v.element_type = static_cast<value_type>(et);
        const std::size_t start = r.at();
        if (const std::size_t s = fixed_size(v.element_type)) {
            if (v.count > r.left() / s)
                return fail(errc::truncated, "gguf: an array of " + std::to_string(v.count) + " elements runs past the end");
            if (!r.take(static_cast<std::size_t>(v.count) * s, v.bytes))
                return fail(errc::truncated, "gguf: an array runs past the end");
            return ok(v);
        }
        // Strings and arrays: each element at least 8 bytes, so the count is bounded by what is left
        if (v.count > r.left() / 8)
            return fail(errc::truncated, "gguf: an array of " + std::to_string(v.count) + " elements runs past the end");
        std::span<const std::byte> all;
        reader copy = r;
        for (std::uint64_t i = 0; i < v.count; ++i) {
            SNICHOLLS_TRY_VOID(read_value(r, v.element_type, depth + 1));
        }
        if (!copy.take(r.at() - start, all))
            return fail(errc::truncated, "gguf: an array runs past the end");
        v.bytes = all;
        return ok(v);
    }
} // namespace detail

/// \brief Parses and validates a GGUF file.
/// \param bytes The whole file. \param opt The limits.
/// \return The file (borrowing `bytes`), or the first problem found.
/// Complexity: linear in the metadata and tensor infos, plus O(T log T) for the overlap check.
[[nodiscard]] inline result<file> parse(std::span<const std::byte> bytes, const parse_options& opt = {})
{
    detail::reader r(bytes);
    std::uint32_t m = 0, version = 0;
    std::uint64_t n_tensors = 0, n_kv = 0;
    if (!r.get(m))
        return fail(errc::truncated, "gguf: shorter than the magic");
    if (m != magic)
        return fail(errc::bad_magic, "gguf: the file does not start with GGUF");
    if (!r.get(version))
        return fail(errc::truncated, "gguf: no version");
    if (version != 2 && version != 3)
        return fail(errc::unsupported_version, "gguf: version " + std::to_string(version)
                                                   + (version > 0xffff ? " (a big-endian file?)" : ""));
    if (!r.get(n_tensors) || !r.get(n_kv))
        return fail(errc::truncated, "gguf: no counts");
    if (n_tensors > opt.max_tensors || n_kv > opt.max_metadata)
        return fail(errc::too_large, "gguf: " + std::to_string(n_tensors) + " tensors and " + std::to_string(n_kv) + " keys");
    if (n_kv > r.left() / 12 || n_tensors > r.left() / 24)
        return fail(errc::truncated, "gguf: the counts are larger than the file");
    file f;
    f.version = version;
    f.metadata.reserve(static_cast<std::size_t>(n_kv));
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        metadata_entry e;
        std::uint32_t t = 0;
        if (!r.string(e.key) || !r.get(t))
            return fail(errc::truncated, "gguf: metadata entry " + std::to_string(i) + " runs past the end");
        if (t > 12)
            return fail(errc::bad_header, "gguf: " + quote_name(e.key) + " has unknown type " + std::to_string(t));
        SNICHOLLS_TRY_ASSIGN(e.val, detail::read_value(r, static_cast<value_type>(t), 0));
        if (f.find(e.key))
            return fail(errc::duplicate_name, "gguf: key " + quote_name(e.key) + " appears twice");
        f.metadata.push_back(e);
    }
    if (const value* a = f.find("general.alignment")) {
        const auto v = a->type == value_type::u32 ? a->as_u64() : std::nullopt;
        if (!v || *v == 0 || !std::has_single_bit(*v) || *v > (1u << 20))
            return fail(errc::bad_header, "gguf: general.alignment must be a u32 power of two");
        f.alignment = static_cast<std::uint32_t>(*v);
    }
    f.tensors.reserve(static_cast<std::size_t>(n_tensors));
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        tensor_info ti;
        std::uint32_t n_dims = 0, type = 0;
        if (!r.string(ti.name) || !r.get(n_dims))
            return fail(errc::truncated, "gguf: tensor info " + std::to_string(i) + " runs past the end");
        if (n_dims > max_dims)
            return fail(errc::bad_shape, "gguf: " + quote_name(ti.name) + " has " + std::to_string(n_dims) + " dimensions");
        tensor_shape ne;
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            std::uint64_t e = 0;
            if (!r.get(e))
                return fail(errc::truncated, "gguf: tensor info " + quote_name(ti.name) + " runs past the end");
            ne.push_back(e);
        }
        if (!r.get(type) || !r.get(ti.offset))
            return fail(errc::truncated, "gguf: tensor info " + quote_name(ti.name) + " runs past the end");
        if (!traits(type))
            return fail(errc::bad_dtype, "gguf: " + quote_name(ti.name) + " has unknown ggml type " + std::to_string(type));
        ti.type = static_cast<ggml_type>(type);
        ti.shape = ne.reversed();
        f.tensors.push_back(ti);
    }
    {
        std::vector<std::string_view> names;
        for (const auto& t : f.tensors)
            names.push_back(t.name);
        std::sort(names.begin(), names.end());
        if (const auto d = std::adjacent_find(names.begin(), names.end()); d != names.end())
            return fail(errc::duplicate_name, "gguf: tensor " + quote_name(*d) + " appears twice");
    }
    const std::uint64_t data_start = (r.at() + f.alignment - 1) / f.alignment * f.alignment;
    if (data_start > bytes.size() && !f.tensors.empty())
        return fail(errc::truncated, "gguf: no data section");
    f.data_offset = std::min<std::uint64_t>(data_start, bytes.size());
    f.data = bytes.subspan(static_cast<std::size_t>(f.data_offset));
    for (auto& ti : f.tensors) {
        const tensor_shape ne = ti.ne();
        std::optional<std::uint64_t> size = ne.rank == 0 ? row_bytes(ti.type, 1) : row_bytes(ti.type, ne[0]);
        for (std::size_t d = 1; d < ne.rank && size; ++d)
            size = checked_mul(*size, ne[d]);
        if (!size)
            return fail(errc::bad_shape, "gguf: " + quote_name(ti.name) + " " + to_string(ne) + " is not whole blocks of "
                                             + std::string(traits(ti.type).name));
        if (ti.offset % f.alignment != 0)
            return fail(errc::misaligned, "gguf: " + quote_name(ti.name) + " is not at a multiple of " + std::to_string(f.alignment));
        if (ti.offset > f.data.size() || *size > f.data.size() - ti.offset)
            return fail(errc::out_of_bounds, "gguf: " + quote_name(ti.name) + " runs past the end of the data");
        ti.data = f.data.subspan(static_cast<std::size_t>(ti.offset), static_cast<std::size_t>(*size));
    }
    std::vector<const tensor_info*> order;
    for (const auto& t : f.tensors)
        order.push_back(&t);
    std::sort(order.begin(), order.end(), [](const tensor_info* a, const tensor_info* b) { return a->offset < b->offset; });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (order[i]->offset < order[i - 1]->offset + order[i - 1]->data.size())
            return fail(errc::overlap, "gguf: " + quote_name(order[i]->name) + " overlaps " + quote_name(order[i - 1]->name));
    return ok(std::move(f));
}

// ---------------------------------------------------------------------------------------------------------- writing

/// \brief A metadata value that owns its contents, for writing.
struct metadata_value
{
    value_type type = value_type::u32;            ///< The type.
    std::uint64_t u = 0;                          ///< An unsigned value (u8 to u64; bool as 0 or 1).
    std::int64_t i = 0;                           ///< A signed value (i8 to i64).
    double f = 0;                                 ///< A float value (f32, f64).
    std::string s;                                ///< A string.
    value_type element_type = value_type::u8;     ///< An array's element type.
    std::vector<metadata_value> elements;         ///< An array's elements, each of `element_type`.
};

/// \brief A u8 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_u8(std::uint8_t v) { return {value_type::u8, v, 0, 0, {}, {}, {}}; }
/// \brief An i8 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_i8(std::int8_t v) { return {value_type::i8, 0, v, 0, {}, {}, {}}; }
/// \brief A u16 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_u16(std::uint16_t v) { return {value_type::u16, v, 0, 0, {}, {}, {}}; }
/// \brief An i16 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_i16(std::int16_t v) { return {value_type::i16, 0, v, 0, {}, {}, {}}; }
/// \brief A u32 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_u32(std::uint32_t v) { return {value_type::u32, v, 0, 0, {}, {}, {}}; }
/// \brief An i32 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_i32(std::int32_t v) { return {value_type::i32, 0, v, 0, {}, {}, {}}; }
/// \brief A u64 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_u64(std::uint64_t v) { return {value_type::u64, v, 0, 0, {}, {}, {}}; }
/// \brief An i64 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_i64(std::int64_t v) { return {value_type::i64, 0, v, 0, {}, {}, {}}; }
/// \brief An f32 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_f32(float v) { return {value_type::f32, 0, 0, v, {}, {}, {}}; }
/// \brief An f64 value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_f64(double v) { return {value_type::f64, 0, 0, v, {}, {}, {}}; }
/// \brief A bool value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_bool(bool v) { return {value_type::boolean, v ? 1u : 0u, 0, 0, {}, {}, {}}; }
/// \brief A string value. \param v The value. \return The metadata value.
[[nodiscard]] inline metadata_value value_string(std::string_view v) { return {value_type::string, 0, 0, 0, std::string(v), {}, {}}; }
/// \brief An array. \param element The element type. \param elements The elements, each of that type.
/// \return The metadata value.
[[nodiscard]] inline metadata_value value_array(value_type element, std::vector<metadata_value> elements)
{
    return {value_type::array, 0, 0, 0, {}, element, std::move(elements)};
}
/// \brief An array of strings. \param v The strings. \return The metadata value.
[[nodiscard]] inline metadata_value value_strings(std::span<const std::string> v)
{
    std::vector<metadata_value> e;
    e.reserve(v.size());
    for (const auto& s : v)
        e.push_back(value_string(s));
    return value_array(value_type::string, std::move(e));
}

/// \brief An owned copy of a parsed value. \param v The value (validated by `parse`). \return The copy.
[[nodiscard]] inline metadata_value to_owned(const value& v)
{
    metadata_value o;
    o.type = v.type;
    switch (v.type) {
    case value_type::u8: case value_type::u16: case value_type::u32: case value_type::u64: o.u = *v.as_u64(); break;
    case value_type::boolean: o.u = *v.as_bool() ? 1 : 0; break;
    case value_type::i8: case value_type::i16: case value_type::i32: case value_type::i64: o.i = *v.as_i64(); break;
    case value_type::f32: case value_type::f64: o.f = *v.as_f64(); break;
    case value_type::string: o.s = std::string(*v.as_string()); break;
    case value_type::array: {
        o.element_type = v.element_type;
        detail::reader r(v.bytes);
        for (std::uint64_t k = 0; k < v.count; ++k)
            o.elements.push_back(to_owned(*detail::read_value(r, v.element_type, 0)));
        break;
    }
    }
    return o;
}

namespace detail
{
    template <typename T>
    void put(std::vector<std::byte>& out, T v)
    {
        const auto b = to_le_bytes(v);
        for (const auto c : b)
            out.push_back(static_cast<std::byte>(c));
    }
    inline void put_string(std::vector<std::byte>& out, std::string_view s)
    {
        put(out, static_cast<std::uint64_t>(s.size()));
        for (const char c : s)
            out.push_back(static_cast<std::byte>(c));
    }
    inline result<void> put_value(std::vector<std::byte>& out, const metadata_value& v, int depth)
    {
        switch (v.type) {
        case value_type::u8: case value_type::boolean: put(out, static_cast<std::uint8_t>(v.u)); break;
        case value_type::i8: put(out, static_cast<std::int8_t>(v.i)); break;
        case value_type::u16: put(out, static_cast<std::uint16_t>(v.u)); break;
        case value_type::i16: put(out, static_cast<std::int16_t>(v.i)); break;
        case value_type::u32: put(out, static_cast<std::uint32_t>(v.u)); break;
        case value_type::i32: put(out, static_cast<std::int32_t>(v.i)); break;
        case value_type::u64: put(out, v.u); break;
        case value_type::i64: put(out, v.i); break;
        case value_type::f32: put(out, static_cast<float>(v.f)); break;
        case value_type::f64: put(out, v.f); break;
        case value_type::string: put_string(out, v.s); break;
        case value_type::array:
            if (depth >= 8)
                return fail(errc::bad_argument, "gguf: arrays nested more than 8 deep");
            put(out, static_cast<std::uint32_t>(v.element_type));
            put(out, static_cast<std::uint64_t>(v.elements.size()));
            for (const auto& e : v.elements) {
                if (e.type != v.element_type)
                    return fail(errc::bad_argument, "gguf: an array element of type " + std::string(to_string(e.type))
                                                        + " in an array of " + std::string(to_string(v.element_type)));
                SNICHOLLS_TRY_VOID(put_value(out, e, depth + 1));
            }
            break;
        default: return fail(errc::bad_argument, "gguf: unknown value type");
        }
        return ok();
    }
} // namespace detail

/// \brief Builds a GGUF file.
class writer
{
public:
    /// \brief Sets a key, replacing its value in place if it was set. \param key The key. \param v The value.
    void set(std::string_view key, metadata_value v)
    {
        for (auto& [k, old] : metadata_)
            if (k == key) {
                old = std::move(v);
                return;
            }
        metadata_.emplace_back(std::string(key), std::move(v));
    }
    /// \brief A key's value. \param key The key. \return The value, or null.
    [[nodiscard]] const metadata_value* get(std::string_view key) const noexcept
    {
        for (const auto& [k, v] : metadata_)
            if (k == key)
                return &v;
        return nullptr;
    }
    /// \brief Adds a tensor. \param name Its name. \param type Its ggml type.
    /// \param shape Its shape, outermost first. \param data Its bytes (borrowed until `write`).
    void add_tensor(std::string_view name, ggml_type type, tensor_shape shape, std::span<const std::byte> data)
    {
        tensors_.push_back({std::string(name), type, shape, data});
    }
    /// \brief Adds a dense tensor. \param t The tensor (borrowed until `write`). \return Nothing, or `bad_dtype` for a
    /// dtype GGUF has no type for.
    [[nodiscard]] result<void> add_tensor(const tensor_view& t)
    {
        const auto g = from_dtype(t.type);
        if (!g)
            return fail(errc::bad_dtype, "gguf: no ggml type for " + std::string(safetensors_name(t.type)) + " (" + quote_name(t.name) + ")");
        add_tensor(t.name, *g, t.shape, t.data);
        return ok();
    }
    /// \brief The tensors added. \return Their count.
    [[nodiscard]] std::size_t tensor_count() const noexcept { return tensors_.size(); }

    /// \brief Writes the file to a sink. \tparam Sink Callable with `std::span<const std::byte>`.
    /// \param sink Receives the bytes in order. \return Nothing, or `duplicate_name`, `bad_shape`, `bad_argument`.
    template <typename Sink>
    [[nodiscard]] result<void> write_to(Sink&& sink) const
    {
        std::uint32_t alignment = default_alignment;
        if (const metadata_value* a = get("general.alignment")) {
            if (a->type != value_type::u32 || a->u == 0 || !std::has_single_bit(a->u))
                return fail(errc::bad_argument, "gguf: general.alignment must be a u32 power of two");
            alignment = static_cast<std::uint32_t>(a->u);
        }
        std::vector<std::byte> head;
        detail::put(head, magic);
        detail::put(head, current_version);
        detail::put(head, static_cast<std::uint64_t>(tensors_.size()));
        detail::put(head, static_cast<std::uint64_t>(metadata_.size()));
        for (const auto& [k, v] : metadata_) {
            detail::put_string(head, k);
            detail::put(head, static_cast<std::uint32_t>(v.type));
            SNICHOLLS_TRY_VOID(detail::put_value(head, v, 0));
        }
        {
            std::vector<std::string_view> names;
            for (const auto& t : tensors_)
                names.push_back(t.name);
            std::sort(names.begin(), names.end());
            if (const auto d = std::adjacent_find(names.begin(), names.end()); d != names.end())
                return fail(errc::duplicate_name, "gguf: two tensors named " + quote_name(*d));
        }
        std::uint64_t offset = 0;
        for (const auto& t : tensors_) {
            if (t.shape.rank > max_dims)
                return fail(errc::bad_shape, "gguf: " + quote_name(t.name) + " has more than 4 dimensions");
            const tensor_shape ne = t.shape.reversed();
            std::optional<std::uint64_t> size = row_bytes(t.type, ne.rank ? ne[0] : 1);
            for (std::size_t d = 1; d < ne.rank && size; ++d)
                size = checked_mul(*size, ne[d]);
            if (!size || *size != t.data.size())
                return fail(errc::bad_shape, "gguf: " + quote_name(t.name) + " has " + std::to_string(t.data.size()) + " bytes for "
                                                 + std::string(traits(t.type).name) + " " + to_string(t.shape));
            detail::put_string(head, t.name);
            detail::put(head, static_cast<std::uint32_t>(ne.rank));
            for (std::size_t d = 0; d < ne.rank; ++d)
                detail::put(head, ne[d]);
            detail::put(head, static_cast<std::uint32_t>(t.type));
            detail::put(head, offset);
            offset += (*size + alignment - 1) / alignment * alignment;
        }
        while (head.size() % alignment != 0)
            head.push_back(std::byte{0});
        sink(std::span<const std::byte>(head));
        static constexpr std::byte zeros[64]{};
        for (const auto& t : tensors_) {
            sink(t.data);
            for (std::size_t pad = (alignment - t.data.size() % alignment) % alignment; pad > 0;) {
                const std::size_t n = std::min<std::size_t>(pad, sizeof zeros);
                sink(std::span<const std::byte>(zeros, n));
                pad -= n;
            }
        }
        return ok();
    }
    /// \brief Writes the file into memory. \return The bytes, or `write_to`'s errors.
    [[nodiscard]] result<std::vector<std::byte>> write() const
    {
        std::vector<std::byte> out;
        SNICHOLLS_TRY_VOID(write_to([&out](std::span<const std::byte> s) { out.insert(out.end(), s.begin(), s.end()); }));
        return ok(std::move(out));
    }

private:
    struct pending
    {
        std::string name;
        ggml_type type;
        tensor_shape shape;
        std::span<const std::byte> data;
    };
    std::vector<std::pair<std::string, metadata_value>> metadata_;
    std::vector<pending> tensors_;
};
} // namespace snicholls::formats::gguf

#endif // snicholls_formats_gguf_gguf_hpp
