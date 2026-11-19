//
//  npy.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief NumPy's `.npy` (versions 1 to 3) and `.npz` (a zip of `.npy` members): read and write.
/// \ingroup numpy
///
/// \overview
/// An `.npy` file is the magic `\x93NUMPY`, a version, a header length, then a Python dict literal,
/// `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }`, padded with spaces and a newline so the data starts
/// at a multiple of 64. `parse_npy` reads the dict with a small parser that accepts exactly that shape (it never
/// evaluates Python) and returns a `tensor_view` into the bytes. `write_npy` writes what `np.save` writes, byte for byte (version 1.0, or
/// 2.0 when the header is over 64 KiB, with numpy's spare space after the dict). `parse_npz` / `write_npz` do the same for `np.savez` archives: one stored `<name>.npy` per array.
///
/// Element types: `<f8 <f4 <f2 <i8 <i4 <i2 |i1 <u8 <u4 <u2 |u1 |b1` (and `=` for native, which is little-endian
/// here). NumPy has no bfloat16 or eight-bit floats: writing those is `bad_dtype`.
///
/// \relations
/// \see formats::zip, formats::tensor_view
///
/// \impls
/// One header parser; npz through `zip`.
///
/// \contracts
/// The view borrows the bytes. A Fortran-ordered, big-endian or structured array is refused, never misread.
///
/// \errors
/// `truncated`, `bad_magic`, `unsupported_version`, `bad_header` (the dict is malformed), `bad_dtype`, `bad_shape`,
/// `unsupported` (Fortran order, big-endian); npz adds zip's errors (`savez_compressed` is `unsupported`).
///
/// \examples
/// \snippet npy_demo.cpp npy
///
/// \perf
/// Not benchmarked: a header per array.
///
/// \testing
/// tests/numpy/npy_test.cpp (and numpy itself in scripts/crosscheck.sh), demos/numpy/npy_demo.cpp.
///
/// \deps
/// The standard library and core.
///
/// \gotchas
/// A 0-dimensional array's shape is `()`; a 1-dimensional one's `(n,)`, with the comma.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_numpy_npy_hpp
#define snicholls_formats_numpy_npy_hpp

#include <snicholls/formats/archive/zip.hpp>
#include <snicholls/formats/tensor.hpp>

#include <snicholls/core/endian.hpp>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::numpy
{
/// \brief NumPy's `descr` of a dtype. \param t The dtype. \return "<f4" and so on, or none (bf16, the 8-bit floats).
[[nodiscard]] constexpr std::optional<std::string_view> descr_of(dtype t) noexcept
{
    switch (t) {
    case dtype::f64: return "<f8";
    case dtype::f32: return "<f4";
    case dtype::f16: return "<f2";
    case dtype::i64: return "<i8";
    case dtype::i32: return "<i4";
    case dtype::i16: return "<i2";
    case dtype::i8: return "|i1";
    case dtype::u64: return "<u8";
    case dtype::u32: return "<u4";
    case dtype::u16: return "<u2";
    case dtype::u8: return "|u1";
    case dtype::boolean: return "|b1";
    default: return std::nullopt;
    }
}

/// \brief The dtype of a `descr`. \param d "<f4" and so on ("=" for native and "<" on one-byte types accepted).
/// \return The dtype, or none.
[[nodiscard]] constexpr std::optional<dtype> dtype_of_descr(std::string_view d) noexcept
{
    if (d.size() != 3 || (d[0] != '<' && d[0] != '=' && d[0] != '|'))
        return std::nullopt;
    const char canonical[3] = {d[2] == '1' ? '|' : '<', d[1], d[2]};
    for (const dtype t : all_dtypes)
        if (const auto s = descr_of(t); s && *s == std::string_view(canonical, 3))
            return t;
    return std::nullopt;
}

namespace detail
{
    inline constexpr std::string_view magic = "\x93NUMPY";
    inline void skip_space(std::string_view& s)
    {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\n'))
            s.remove_prefix(1);
    }
    inline bool eat(std::string_view& s, char c)
    {
        skip_space(s);
        if (s.empty() || s.front() != c)
            return false;
        s.remove_prefix(1);
        return true;
    }
    inline std::optional<std::string_view> quoted_string(std::string_view& s)
    {
        skip_space(s);
        if (s.empty() || (s.front() != '\'' && s.front() != '"'))
            return std::nullopt;
        const char q = s.front();
        const auto end = s.find(q, 1);
        if (end == std::string_view::npos)
            return std::nullopt;
        const auto v = s.substr(1, end - 1);
        s.remove_prefix(end + 1);
        return v;
    }
    struct header
    {
        std::string_view descr;
        bool fortran = false;
        tensor_shape shape;
    };
    inline result<header> parse_dict(std::string_view s)
    {
        header h;
        bool have_descr = false, have_order = false, have_shape = false;
        if (!eat(s, '{'))
            return fail(errc::bad_header, "npy: the header is not a dict");
        while (true) {
            if (eat(s, '}'))
                break;
            const auto key = quoted_string(s);
            if (!key || !eat(s, ':'))
                return fail(errc::bad_header, "npy: a malformed key in the header");
            skip_space(s);
            if (*key == "descr") {
                const auto v = quoted_string(s);
                if (!v)
                    return fail(errc::unsupported, "npy: a structured dtype is not read");
                h.descr = *v;
                have_descr = true;
            } else if (*key == "fortran_order") {
                if (s.starts_with("True"))
                    h.fortran = true;
                else if (!s.starts_with("False"))
                    return fail(errc::bad_header, "npy: fortran_order is neither True nor False");
                s.remove_prefix(h.fortran ? 4 : 5);
                have_order = true;
            } else if (*key == "shape") {
                if (!eat(s, '('))
                    return fail(errc::bad_header, "npy: shape is not a tuple");
                while (!eat(s, ')')) {
                    skip_space(s);
                    std::uint64_t v = 0;
                    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
                    if (ec != std::errc{})
                        return fail(errc::bad_shape, "npy: an extent is not a non-negative integer");
                    if (!h.shape.push_back(v))
                        return fail(errc::bad_shape, "npy: more than " + std::to_string(max_rank) + " dimensions");
                    s.remove_prefix(static_cast<std::size_t>(p - s.data()));
                    if (!eat(s, ',')) {
                        if (!eat(s, ')'))
                            return fail(errc::bad_header, "npy: a malformed shape");
                        break;
                    }
                }
                have_shape = true;
            } else {
                return fail(errc::bad_header, "npy: unknown header key " + quote_name(*key));
            }
            if (!eat(s, ',')) {
                if (!eat(s, '}'))
                    return fail(errc::bad_header, "npy: a malformed header dict");
                break;
            }
        }
        if (!have_descr || !have_order || !have_shape)
            return fail(errc::bad_header, "npy: the header needs descr, fortran_order and shape");
        return ok(h);
    }
} // namespace detail

/// \brief Parses an `.npy` file. \param bytes The file. \param name The name to give the view.
/// \return The array as a view into `bytes`, or the problem.
[[nodiscard]] inline result<tensor_view> parse_npy(std::span<const std::byte> bytes, std::string_view name = {})
{
    if (bytes.size() < 10)
        return fail(errc::truncated, "npy: shorter than the header");
    if (std::string_view(reinterpret_cast<const char*>(bytes.data()), 6) != detail::magic)
        return fail(errc::bad_magic, "npy: no \\x93NUMPY magic");
    const auto major = std::to_integer<int>(bytes[6]);
    std::size_t len = 0, start = 0;
    if (major == 1) {
        len = load_le<std::uint16_t>(bytes.data() + 8);
        start = 10;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < 12)
            return fail(errc::truncated, "npy: shorter than the header");
        len = load_le<std::uint32_t>(bytes.data() + 8);
        start = 12;
    } else {
        return fail(errc::unsupported_version, "npy: version " + std::to_string(major));
    }
    if (len > bytes.size() - start)
        return fail(errc::truncated, "npy: the header runs past the end");
    SNICHOLLS_TRY_ASSIGN(const detail::header h, detail::parse_dict(std::string_view(reinterpret_cast<const char*>(bytes.data() + start), len)));
    if (h.fortran)
        return fail(errc::unsupported, "npy: a Fortran-ordered array is not read");
    if (!h.descr.empty() && h.descr[0] == '>')
        return fail(errc::unsupported, "npy: a big-endian array is not read");
    const auto t = dtype_of_descr(h.descr);
    if (!t)
        return fail(errc::bad_dtype, "npy: dtype " + quote_name(h.descr) + " is not read");
    const auto size = checked_byte_size(h.shape, *t);
    const std::size_t data = start + len;
    if (!size || *size != bytes.size() - data)
        return fail(errc::bad_shape, "npy: " + std::to_string(bytes.size() - data) + " bytes of data for shape " + to_string(h.shape));
    return ok(tensor_view{name, *t, h.shape, bytes.subspan(data)});
}

/// \brief The header `write_npy` writes for a tensor: magic, version, length, dict, padding.
/// \param t The tensor. \return The bytes, or `bad_dtype`.
[[nodiscard]] inline result<std::vector<std::byte>> npy_header(const tensor_view& t)
{
    const auto d = descr_of(t.type);
    if (!d)
        return fail(errc::bad_dtype, "npy: NumPy has no " + std::string(safetensors_name(t.type)));
    std::string dict = "{'descr': '" + std::string(*d) + "', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < t.shape.rank; ++i)
        dict += std::to_string(t.shape[i]) + (t.shape.rank == 1 || i + 1 < t.shape.rank ? ", " : "");
    if (t.shape.rank == 1)
        dict.pop_back();   // "(3,)"
    dict += "), }";
    // numpy's spare space, so the first extent can grow in place (GROWTH_AXIS_MAX_DIGITS = 21): byte for byte np.save
    if (t.shape.rank > 0)
        dict.append(21 - std::to_string(t.shape[0]).size(), ' ');
    const bool v2 = dict.size() + 1 + 10 > 0xffff;
    const std::size_t prefix = v2 ? 12 : 10;
    while ((prefix + dict.size() + 1) % 64 != 0)
        dict.push_back(' ');
    dict.push_back('\n');
    std::vector<std::byte> h(prefix);
    for (std::size_t i = 0; i < 6; ++i)
        h[i] = static_cast<std::byte>(detail::magic[i]);
    h[6] = std::byte{static_cast<unsigned char>(v2 ? 2 : 1)};
    h[7] = std::byte{0};
    if (v2)
        store_le(h.data() + 8, static_cast<std::uint32_t>(dict.size()));
    else
        store_le(h.data() + 8, static_cast<std::uint16_t>(dict.size()));
    for (const char c : dict)
        h.push_back(static_cast<std::byte>(c));
    return ok(std::move(h));
}

/// \brief Writes an `.npy` file. \param t The tensor. \return The bytes, or `bad_dtype` / `bad_shape`.
[[nodiscard]] inline result<std::vector<std::byte>> write_npy(const tensor_view& t)
{
    const auto size = checked_byte_size(t.shape, t.type);
    if (!size || *size != t.data.size())
        return fail(errc::bad_shape, "npy: " + std::to_string(t.data.size()) + " bytes for shape " + to_string(t.shape));
    SNICHOLLS_TRY_ASSIGN(std::vector<std::byte> out, npy_header(t));
    out.insert(out.end(), t.data.begin(), t.data.end());
    return ok(std::move(out));
}

/// \brief Parses an `.npz` archive. \param bytes The archive. \return One view per array, named without ".npy" (the
/// name a view into the archive's directory), in the archive's order, or the problem.
[[nodiscard]] inline result<std::vector<tensor_view>> parse_npz(std::span<const std::byte> bytes)
{
    SNICHOLLS_TRY_ASSIGN(const zip::archive a, zip::parse(bytes));
    std::vector<tensor_view> out;
    for (const auto& e : a.entries) {
        if (!e.name.ends_with(".npy"))
            return fail(errc::bad_header, "npz: member " + quote_name(e.name) + " is not a .npy");
        SNICHOLLS_TRY_ASSIGN(tensor_view v, parse_npy(e.data, e.name.substr(0, e.name.size() - 4)));
        out.push_back(v);
    }
    return ok(std::move(out));
}

/// \brief Writes an `.npz` archive (as `np.savez`): each tensor a stored `<name>.npy`.
/// \param tensors The arrays. \return The archive's bytes, or the first error.
[[nodiscard]] inline result<std::vector<std::byte>> write_npz(std::span<const tensor_view> tensors)
{
    std::vector<std::vector<std::byte>> files;
    std::vector<std::string> names;
    for (const auto& t : tensors) {
        SNICHOLLS_TRY_ASSIGN(std::vector<std::byte> f, write_npy(t));
        files.push_back(std::move(f));
        names.push_back(std::string(t.name) + ".npy");
    }
    std::vector<zip::member> members;
    for (std::size_t i = 0; i < files.size(); ++i)
        members.push_back({names[i], files[i]});
    return zip::write(members);
}
} // namespace snicholls::formats::numpy

#endif // snicholls_formats_numpy_npy_hpp
