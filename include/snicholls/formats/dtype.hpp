//
//  dtype.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `dtype`, the element types every format shares, and conversion between them.
/// \ingroup formats
///
/// \overview
/// One enumeration for the element types of dense tensors: the IEEE floats (F64, F32, F16), bfloat16, the two OCP
/// eight-bit floats (E4M3 "fn": no infinities, one NaN; E5M2: IEEE-like), the signed and unsigned integers of 8 to 64
/// bits, and bool (one byte, 0 or 1). safetensors names them `F64` ... `BOOL`; NumPy, GGUF and PyTorch map onto it in
/// their own headers. Block-quantised GGUF types are not dtypes: they are `gguf::ggml_type`.
///
/// `convert_elements` turns a run of elements of one dtype into another: any number to a float type (through `double`; the
/// narrow floats round to nearest even with core's `ml::minifloat`), and integers to integers when every value fits.
/// A float to an integer is refused: it is never what a checkpoint conversion means.
///
/// \relations
/// \see snicholls::ml::minifloat (core's `ml/minifloat.hpp`), formats::tensor_view.
///
/// \impls
/// `constexpr` tables over the enumeration; `convert_elements` is a loop per element.
///
/// \contracts
/// Every function but `convert_elements` is `constexpr` and `noexcept`. `convert_elements` writes nothing on failure. Elements are
/// little-endian, as every supported format stores them.
///
/// \errors
/// `convert_elements`: `bad_shape` when the spans do not hold the same count, `unsupported` for a float to an integer,
/// `out_of_bounds` for an integer that does not fit the target (or a bool other than 0 or 1).
///
/// \examples
/// \snippet dtype_demo.cpp dtype
///
/// \perf
/// benchmarks/dtype_benchmark.cpp: `convert_elements` bf16 to f32 against the hand-written loop.
///
/// \testing
/// tests/dtype_test.cpp, demos/dtype_demo.cpp.
///
/// \deps
/// Core's `ml/minifloat.hpp`, `endian.hpp`.
///
/// \gotchas
/// A host must be little-endian (x86-64, ARM64): a static_assert says so.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_dtype_hpp
#define snicholls_formats_dtype_hpp

#include <snicholls/formats/error.hpp>

#include <snicholls/core/endian.hpp>
#include <snicholls/core/ml/minifloat.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace snicholls::formats
{
static_assert(std::endian::native == std::endian::little, "snicholls::formats: a little-endian host is required");

/// \brief The element type of a dense tensor.
enum class dtype : std::uint8_t
{
    f64,       ///< IEEE binary64.
    f32,       ///< IEEE binary32.
    f16,       ///< IEEE binary16.
    bf16,      ///< bfloat16: binary32's exponent, 7 mantissa bits.
    f8_e4m3,   ///< OCP E4M3 ("fn"): 4 exponent bits, 3 mantissa, no infinities, NaN all ones.
    f8_e5m2,   ///< OCP E5M2: 5 exponent bits, 2 mantissa, IEEE-like.
    i64,       ///< Signed 64-bit.
    i32,       ///< Signed 32-bit.
    i16,       ///< Signed 16-bit.
    i8,        ///< Signed 8-bit.
    u64,       ///< Unsigned 64-bit.
    u32,       ///< Unsigned 32-bit.
    u16,       ///< Unsigned 16-bit.
    u8,        ///< Unsigned 8-bit.
    boolean,   ///< One byte, 0 or 1.
};

/// \brief Every dtype, in the enumeration's order.
inline constexpr dtype all_dtypes[] = {dtype::f64, dtype::f32, dtype::f16, dtype::bf16, dtype::f8_e4m3,
                                       dtype::f8_e5m2, dtype::i64, dtype::i32, dtype::i16, dtype::i8,
                                       dtype::u64, dtype::u32, dtype::u16, dtype::u8, dtype::boolean};

/// \brief The bytes of one element. \param t The dtype. \return 1, 2, 4 or 8.
[[nodiscard]] constexpr std::size_t size_of(dtype t) noexcept
{
    switch (t) {
    case dtype::f64: case dtype::i64: case dtype::u64: return 8;
    case dtype::f32: case dtype::i32: case dtype::u32: return 4;
    case dtype::f16: case dtype::bf16: case dtype::i16: case dtype::u16: return 2;
    case dtype::f8_e4m3: case dtype::f8_e5m2: case dtype::i8: case dtype::u8: case dtype::boolean: return 1;
    }
    return 0;
}

/// \brief Whether a dtype is a floating-point type. \param t The dtype. \return True for the six floats.
[[nodiscard]] constexpr bool is_float(dtype t) noexcept { return t <= dtype::f8_e5m2; }

/// \brief Whether a dtype is a signed integer. \param t The dtype. \return True for i8 to i64.
[[nodiscard]] constexpr bool is_signed_integer(dtype t) noexcept { return t >= dtype::i64 && t <= dtype::i8; }

/// \brief safetensors' name of a dtype. \param t The dtype. \return "F64", "F32", ..., "BOOL".
[[nodiscard]] constexpr std::string_view safetensors_name(dtype t) noexcept
{
    switch (t) {
    case dtype::f64: return "F64";
    case dtype::f32: return "F32";
    case dtype::f16: return "F16";
    case dtype::bf16: return "BF16";
    case dtype::f8_e4m3: return "F8_E4M3";
    case dtype::f8_e5m2: return "F8_E5M2";
    case dtype::i64: return "I64";
    case dtype::i32: return "I32";
    case dtype::i16: return "I16";
    case dtype::i8: return "I8";
    case dtype::u64: return "U64";
    case dtype::u32: return "U32";
    case dtype::u16: return "U16";
    case dtype::u8: return "U8";
    case dtype::boolean: return "BOOL";
    }
    return "?";
}

/// \brief The dtype of a safetensors name. \param name "F32" and so on. \return The dtype, or none.
[[nodiscard]] constexpr std::optional<dtype> dtype_from_safetensors(std::string_view name) noexcept
{
    for (const dtype t : all_dtypes)
        if (safetensors_name(t) == name)
            return t;
    return std::nullopt;
}

namespace detail::dtype_
{
    // One element as a double (floats) or as an integer (the rest)
    inline double load_float(dtype t, const std::byte* p) noexcept
    {
        switch (t) {
        case dtype::f64: return std::bit_cast<double>(load_le<std::uint64_t>(p));
        case dtype::f32: return static_cast<double>(std::bit_cast<float>(load_le<std::uint32_t>(p)));
        case dtype::f16: return ml::to_double(ml::float16::from_bits(load_le<std::uint16_t>(p)));
        case dtype::bf16: return ml::to_double(ml::bfloat16::from_bits(load_le<std::uint16_t>(p)));
        case dtype::f8_e4m3: return ml::to_double(ml::float8_e4m3::from_bits(std::to_integer<std::uint32_t>(*p)));
        case dtype::f8_e5m2: return ml::to_double(ml::float8_e5m2::from_bits(std::to_integer<std::uint32_t>(*p)));
        default: return 0;
        }
    }
    struct integer
    {
        bool negative;        // value is -magnitude
        std::uint64_t magnitude;
    };
    inline integer load_integer(dtype t, const std::byte* p) noexcept
    {
        const auto s = [](std::int64_t v) { return v < 0 ? integer{true, std::uint64_t(0) - std::uint64_t(v)} : integer{false, std::uint64_t(v)}; };
        switch (t) {
        case dtype::i64: return s(static_cast<std::int64_t>(load_le<std::uint64_t>(p)));
        case dtype::i32: return s(static_cast<std::int32_t>(load_le<std::uint32_t>(p)));
        case dtype::i16: return s(static_cast<std::int16_t>(load_le<std::uint16_t>(p)));
        case dtype::i8: return s(static_cast<std::int8_t>(std::to_integer<std::uint8_t>(*p)));
        case dtype::u64: return {false, load_le<std::uint64_t>(p)};
        case dtype::u32: return {false, load_le<std::uint32_t>(p)};
        case dtype::u16: return {false, load_le<std::uint16_t>(p)};
        default: return {false, std::to_integer<std::uint8_t>(*p)};
        }
    }
    inline double integer_as_double(integer v) noexcept
    {
        return v.negative ? -static_cast<double>(v.magnitude) : static_cast<double>(v.magnitude);
    }
    inline void store_float(dtype t, std::byte* p, double v) noexcept
    {
        switch (t) {
        case dtype::f64: store_le(p, std::bit_cast<std::uint64_t>(v)); break;
        case dtype::f32: store_le(p, std::bit_cast<std::uint32_t>(static_cast<float>(v))); break;
        case dtype::f16: store_le(p, ml::to_minifloat<ml::float16>(v).bits); break;
        case dtype::bf16: store_le(p, ml::to_minifloat<ml::bfloat16>(v).bits); break;
        case dtype::f8_e4m3: *p = std::byte{ml::to_minifloat<ml::float8_e4m3>(v).bits}; break;
        case dtype::f8_e5m2: *p = std::byte{ml::to_minifloat<ml::float8_e5m2>(v).bits}; break;
        default: break;
        }
    }
    // The range of an integer dtype, as (max magnitude when negative, max when positive)
    inline bool fits(dtype t, integer v) noexcept
    {
        const std::size_t bits = 8 * size_of(t);
        if (t == dtype::boolean)
            return !v.negative && v.magnitude <= 1;
        if (is_signed_integer(t)) {
            const std::uint64_t limit = std::uint64_t{1} << (bits - 1);
            return v.negative ? v.magnitude <= limit : v.magnitude < limit;
        }
        return !v.negative && (bits == 64 || v.magnitude < (std::uint64_t{1} << bits));
    }
    inline void store_integer(dtype t, std::byte* p, integer v) noexcept
    {
        const std::uint64_t bits = v.negative ? std::uint64_t(0) - v.magnitude : v.magnitude;
        switch (size_of(t)) {
        case 8: store_le(p, bits); break;
        case 4: store_le(p, static_cast<std::uint32_t>(bits)); break;
        case 2: store_le(p, static_cast<std::uint16_t>(bits)); break;
        default: *p = static_cast<std::byte>(bits & 0xff); break;
        }
    }
} // namespace detail::dtype_

/// \brief Converts `in`, elements of `from`, into `out`, elements of `to`.
/// \param from The dtype of `in`. \param in The source bytes.
/// \param to The dtype of `out`. \param out The destination bytes: as many elements as `in`.
/// \return Nothing, or `bad_shape`, `unsupported` (a float to an integer) or `out_of_bounds` (an integer that does
/// not fit); `out` is untouched on failure.
/// Complexity: linear; two passes when the target is an integer (check, then write).
[[nodiscard]] inline result<void> convert_elements(dtype from, std::span<const std::byte> in, dtype to, std::span<std::byte> out)
{
    const std::size_t a = size_of(from), b = size_of(to);
    if (in.size() % a != 0 || out.size() % b != 0 || in.size() / a != out.size() / b)
        return fail(errc::bad_shape, "convert_elements: the spans hold different numbers of elements");
    const std::size_t n = in.size() / a;
    if (is_float(to)) {
        for (std::size_t i = 0; i < n; ++i) {
            const double v = is_float(from) ? detail::dtype_::load_float(from, in.data() + i * a)
                                            : detail::dtype_::integer_as_double(detail::dtype_::load_integer(from, in.data() + i * a));
            detail::dtype_::store_float(to, out.data() + i * b, v);
        }
        return ok();
    }
    if (is_float(from))
        return fail(errc::unsupported, "convert_elements: a float to an integer is refused");
    for (std::size_t i = 0; i < n; ++i)
        if (!detail::dtype_::fits(to, detail::dtype_::load_integer(from, in.data() + i * a)))
            return fail(errc::out_of_bounds, "convert_elements: element " + std::to_string(i) + " does not fit " + std::string(safetensors_name(to)));
    for (std::size_t i = 0; i < n; ++i)
        detail::dtype_::store_integer(to, out.data() + i * b, detail::dtype_::load_integer(from, in.data() + i * a));
    return ok();
}
} // namespace snicholls::formats

#endif // snicholls_formats_dtype_hpp
