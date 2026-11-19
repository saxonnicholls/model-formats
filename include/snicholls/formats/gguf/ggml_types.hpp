//
//  ggml_types.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Every ggml tensor type: its GGUF code, name, block size and bytes per block.
/// \ingroup gguf
///
/// \overview
/// A GGUF tensor's type is a `ggml_type`: a dense element type (F32, F16, BF16, F64, the integers, F8_E4M3) or a
/// block-quantised format, where `block_size` consecutive elements of a row are stored together in `type_size` bytes.
/// `traits` gives both for every type GGUF defines; `row_bytes` and `tensor_bytes` compute the size of a row and a
/// tensor, refusing a row that is not a whole number of blocks. `to_dtype` / `from_dtype` map the dense types onto
/// `formats::dtype`.
///
/// Where core has a block's layout (`ml/quantize.hpp`: Q8_0, Q4_0, MXFP4) the size here is checked against core's
/// struct by `static_assert`, and `traits(t).in_core` says so; core's `quantize` / `dequantize` then convert. The other
/// sizes are taken from ggml-common.h (llama.cpp, MIT) and belong in core: they move there when core's quantisation
/// track adds the formats, and this table then checks them against core as it does the first three.
///
/// \relations
/// \see formats::gguf::file, snicholls::ml::block_q8_0 (core)
///
/// \impls
/// One `constexpr` table.
///
/// \contracts
/// Everything is `constexpr` and `noexcept`. Unknown codes give none.
///
/// \errors
/// None: unknown types and partial blocks are `std::nullopt`.
///
/// \examples
/// \snippet ggml_types_demo.cpp ggml_types
///
/// \perf
/// Not benchmarked: a table lookup.
///
/// \testing
/// tests/gguf/ggml_types_test.cpp (every size against ggml-common.h's formulas, the core three against core),
/// demos/gguf/ggml_types_demo.cpp.
///
/// \deps
/// Core's `ml/quantize.hpp`.
///
/// \gotchas
/// Codes 4, 5 (Q4_2, Q4_3), 31 to 33 and 36 to 38 were removed from ggml and are not types.
///
/// \status
/// Experimental. The non-core rows move to core when it gains the formats.

#ifndef snicholls_formats_gguf_ggml_types_hpp
#define snicholls_formats_gguf_ggml_types_hpp

#include <snicholls/formats/dtype.hpp>

#include <snicholls/core/checked.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls::formats::gguf
{
/// \brief A ggml tensor type, by its GGUF code.
enum class ggml_type : std::uint32_t
{
    f32 = 0,        ///< binary32.
    f16 = 1,        ///< binary16.
    q4_0 = 2,       ///< 32 elements: a half scale, 4-bit codes.
    q4_1 = 3,       ///< 32 elements: half scale and minimum, 4-bit codes.
    q5_0 = 6,       ///< 32 elements: a half scale, 5-bit codes.
    q5_1 = 7,       ///< 32 elements: half scale and minimum, 5-bit codes.
    q8_0 = 8,       ///< 32 elements: a half scale, 8-bit codes.
    q8_1 = 9,       ///< 32 elements: half scale and sum, 8-bit codes.
    q2_k = 10,      ///< K-quant, 256 elements, 2 bits.
    q3_k = 11,      ///< K-quant, 256 elements, 3 bits.
    q4_k = 12,      ///< K-quant, 256 elements, 4 bits.
    q5_k = 13,      ///< K-quant, 256 elements, 5 bits.
    q6_k = 14,      ///< K-quant, 256 elements, 6 bits.
    q8_k = 15,      ///< K-quant, 256 elements, 8 bits (intermediate).
    iq2_xxs = 16,   ///< i-quant, 2.06 bits.
    iq2_xs = 17,    ///< i-quant, 2.31 bits.
    iq3_xxs = 18,   ///< i-quant, 3.06 bits.
    iq1_s = 19,     ///< i-quant, 1.56 bits.
    iq4_nl = 20,    ///< i-quant, 4.5 bits, non-linear, 32 elements.
    iq3_s = 21,     ///< i-quant, 3.44 bits.
    iq2_s = 22,     ///< i-quant, 2.5 bits.
    iq4_xs = 23,    ///< i-quant, 4.25 bits.
    i8 = 24,        ///< Signed 8-bit integers.
    i16 = 25,       ///< Signed 16-bit integers.
    i32 = 26,       ///< Signed 32-bit integers.
    i64 = 27,       ///< Signed 64-bit integers.
    f64 = 28,       ///< binary64.
    iq1_m = 29,     ///< i-quant, 1.75 bits.
    bf16 = 30,      ///< bfloat16.
    tq1_0 = 34,     ///< Ternary, 1.69 bits.
    tq2_0 = 35,     ///< Ternary, 2.06 bits.
    mxfp4 = 39,     ///< OCP MX FP4: 32 E2M1 codes, an E8M0 scale.
    nvfp4 = 40,     ///< NVIDIA FP4: 64 elements, four E4M3 sub-block scales.
    q1_0 = 41,      ///< 128 elements, 1 bit, a half scale.
    q2_0 = 42,      ///< 64 elements, 2 bits, a half scale.
    f8_e4m3 = 43,   ///< OCP E4M3 elements.
};

/// \brief A type's layout.
struct ggml_type_traits
{
    ggml_type type = ggml_type::f32;   ///< The type.
    std::string_view name;             ///< ggml's name: "f32", "q4_k", ...
    std::uint32_t block_size = 1;      ///< Elements per block (1 for a dense type).
    std::uint32_t type_size = 4;       ///< Bytes per block.
    bool in_core = false;              ///< Whether core's `ml/quantize.hpp` (or a dtype) has the layout.
};

/// \brief ggml's K-quant super-block.
inline constexpr std::uint32_t qk_k = 256;

/// \brief Every type, in code order. Sizes from ggml-common.h's `static_assert`s.
inline constexpr std::array<ggml_type_traits, 36> ggml_types{{
    {ggml_type::f32, "f32", 1, 4, true},
    {ggml_type::f16, "f16", 1, 2, true},
    {ggml_type::q4_0, "q4_0", 32, 2 + 16, true},
    {ggml_type::q4_1, "q4_1", 32, 2 + 2 + 16, false},
    {ggml_type::q5_0, "q5_0", 32, 2 + 4 + 16, false},
    {ggml_type::q5_1, "q5_1", 32, 2 + 2 + 4 + 16, false},
    {ggml_type::q8_0, "q8_0", 32, 2 + 32, true},
    {ggml_type::q8_1, "q8_1", 32, 2 + 2 + 32, false},
    {ggml_type::q2_k, "q2_k", qk_k, 2 + 2 + qk_k / 16 + qk_k / 4, false},
    {ggml_type::q3_k, "q3_k", qk_k, 2 + qk_k / 4 + qk_k / 8 + 12, false},
    {ggml_type::q4_k, "q4_k", qk_k, 2 + 2 + qk_k / 2 + 12, false},
    {ggml_type::q5_k, "q5_k", qk_k, 2 + 2 + qk_k / 2 + qk_k / 8 + 12, false},
    {ggml_type::q6_k, "q6_k", qk_k, 2 + qk_k / 2 + qk_k / 4 + qk_k / 16, false},
    {ggml_type::q8_k, "q8_k", qk_k, 4 + qk_k + qk_k / 8, false},
    {ggml_type::iq2_xxs, "iq2_xxs", qk_k, 2 + qk_k / 4, false},
    {ggml_type::iq2_xs, "iq2_xs", qk_k, 2 + qk_k / 4 + qk_k / 32, false},
    {ggml_type::iq3_xxs, "iq3_xxs", qk_k, 2 + qk_k / 4 + qk_k / 8, false},
    {ggml_type::iq1_s, "iq1_s", qk_k, 2 + qk_k / 8 + qk_k / 16, false},
    {ggml_type::iq4_nl, "iq4_nl", 32, 2 + 16, false},
    {ggml_type::iq3_s, "iq3_s", qk_k, 2 + qk_k / 4 + qk_k / 8 + qk_k / 32 + 4, false},
    {ggml_type::iq2_s, "iq2_s", qk_k, 2 + qk_k / 4 + qk_k / 16, false},
    {ggml_type::iq4_xs, "iq4_xs", qk_k, 2 + 2 + qk_k / 2 + qk_k / 64, false},
    {ggml_type::i8, "i8", 1, 1, true},
    {ggml_type::i16, "i16", 1, 2, true},
    {ggml_type::i32, "i32", 1, 4, true},
    {ggml_type::i64, "i64", 1, 8, true},
    {ggml_type::f64, "f64", 1, 8, true},
    {ggml_type::iq1_m, "iq1_m", qk_k, qk_k / 8 + qk_k / 16 + qk_k / 32, false},
    {ggml_type::bf16, "bf16", 1, 2, true},
    {ggml_type::tq1_0, "tq1_0", qk_k, 2 + 4 * 13, false},
    {ggml_type::tq2_0, "tq2_0", qk_k, 2 + 64, false},
    {ggml_type::mxfp4, "mxfp4", 32, 1 + 16, true},
    {ggml_type::nvfp4, "nvfp4", 64, 4 + 32, false},
    {ggml_type::q1_0, "q1_0", 128, 2 + 16, false},
    {ggml_type::q2_0, "q2_0", 64, 2 + 16, false},
    {ggml_type::f8_e4m3, "f8_e4m3", 1, 1, true},
}};

/// \brief The layout of a type code. \param code A GGUF type code. \return Its traits, or none for an unknown code.
[[nodiscard]] constexpr std::optional<ggml_type_traits> traits(std::uint32_t code) noexcept
{
    for (const auto& t : ggml_types)
        if (static_cast<std::uint32_t>(t.type) == code)
            return t;
    return std::nullopt;
}

/// \brief The layout of a type. \param t The type. \return Its traits.
[[nodiscard]] constexpr ggml_type_traits traits(ggml_type t) noexcept { return *traits(static_cast<std::uint32_t>(t)); }

/// \brief A type by ggml's name. \param name "q8_0" and so on (lower case). \return The type, or none.
[[nodiscard]] constexpr std::optional<ggml_type> ggml_type_from_name(std::string_view name) noexcept
{
    for (const auto& t : ggml_types)
        if (t.name == name)
            return t.type;
    return std::nullopt;
}

/// \brief The bytes of a row. \param t The type. \param elements The row's elements (ne[0]).
/// \return The bytes, or none when the row is not a whole number of blocks or overflows.
[[nodiscard]] constexpr std::optional<std::uint64_t> row_bytes(ggml_type t, std::uint64_t elements) noexcept
{
    const auto tr = traits(t);
    if (elements % tr.block_size != 0)
        return std::nullopt;
    return checked_mul(elements / tr.block_size, std::uint64_t{tr.type_size});
}

/// \brief The dtype of a dense ggml type. \param t The type. \return The dtype, or none for a quantised type.
[[nodiscard]] constexpr std::optional<dtype> to_dtype(ggml_type t) noexcept
{
    switch (t) {
    case ggml_type::f32: return dtype::f32;
    case ggml_type::f16: return dtype::f16;
    case ggml_type::bf16: return dtype::bf16;
    case ggml_type::f64: return dtype::f64;
    case ggml_type::i8: return dtype::i8;
    case ggml_type::i16: return dtype::i16;
    case ggml_type::i32: return dtype::i32;
    case ggml_type::i64: return dtype::i64;
    case ggml_type::f8_e4m3: return dtype::f8_e4m3;
    default: return std::nullopt;
    }
}

/// \brief The ggml type of a dtype. \param t The dtype. \return The type, or none (E5M2, unsigned, bool).
[[nodiscard]] constexpr std::optional<ggml_type> from_dtype(dtype t) noexcept
{
    switch (t) {
    case dtype::f32: return ggml_type::f32;
    case dtype::f16: return ggml_type::f16;
    case dtype::bf16: return ggml_type::bf16;
    case dtype::f64: return ggml_type::f64;
    case dtype::i8: return ggml_type::i8;
    case dtype::i16: return ggml_type::i16;
    case dtype::i32: return ggml_type::i32;
    case dtype::i64: return ggml_type::i64;
    case dtype::f8_e4m3: return ggml_type::f8_e4m3;
    default: return std::nullopt;
    }
}

static_assert(traits(ggml_type::q8_0).type_size == sizeof(ml::block_q8_0) && traits(ggml_type::q8_0).block_size == ml::quant_block_size,
              "snicholls::formats: Q8_0 must have core's layout");
static_assert(traits(ggml_type::q4_0).type_size == sizeof(ml::block_q4_0), "snicholls::formats: Q4_0 must have core's layout");
static_assert(traits(ggml_type::mxfp4).type_size == sizeof(ml::block_mxfp4), "snicholls::formats: MXFP4 must have core's layout");
} // namespace snicholls::formats::gguf

#endif // snicholls_formats_gguf_ggml_types_hpp
