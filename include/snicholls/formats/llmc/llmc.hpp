//
//  llmc.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief llm.c's files: the GPT-2 checkpoint (float or bfloat16) and the token shard, read and write.
/// \ingroup llmc
///
/// \overview
/// The ml library trains in llm.c's formats so the two read each other's files.
///
/// - A **checkpoint** is 256 int32s of header (magic 20240326; version 3 for float32 parameters, 5 for bfloat16;
///   then max_seq_len, vocab_size, num_layers, num_heads, channels, padded_vocab_size), then every parameter in llm.c's
///   order: wte `[Vp, C]`, wpe `[T, C]`, ln1w, ln1b `[L, C]`, qkvw `[L, 3C, C]`, qkvb `[L, 3C]`, attprojw `[L, C, C]`,
///   attprojb, ln2w, ln2b `[L, C]`, fcw `[L, 4C, C]`, fcb `[L, 4C]`, fcprojw `[L, C, 4C]`, fcprojb `[L, C]`, lnfw,
///   lnfb `[C]`. Matrices are `[out, in]`, as PyTorch's `Linear`.
/// - A **token shard** is 256 int32s (magic 20240520, version 1, the token count), then the tokens as uint16.
///
/// `parse_checkpoint` checks the header and the size and returns the sixteen tensors as views; `write_checkpoint`
/// writes one from a shape and the flat parameters. `parse_tokens` / `write_tokens` do the same for shards.
///
/// \relations
/// \see formats::convert::export_llmc_gpt2 (to safetensors and GGUF), the ml library's `apps/llmc_files.hpp`
///
/// \impls
/// Header structs and four functions.
///
/// \contracts
/// Views borrow the bytes. The header's sizes are checked against each other and the file before any view is made.
///
/// \errors
/// `truncated`, `bad_magic`, `unsupported_version`, `bad_header` (non-positive sizes, heads not dividing channels,
/// padded vocabulary below the vocabulary), `bad_shape` (the file's length is not what the header says).
///
/// \examples
/// \snippet llmc_demo.cpp llmc
///
/// \perf
/// Not benchmarked: a header check.
///
/// \testing
/// tests/llmc/llmc_test.cpp, demos/llmc/llmc_demo.cpp.
///
/// \deps
/// The standard library and core.
///
/// \gotchas
/// The rows of wte beyond `vocab_size` (up to `padded_vocab_size`) are padding, zero in llm.c's files.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_llmc_llmc_hpp
#define snicholls_formats_llmc_llmc_hpp

#include <snicholls/formats/tensor.hpp>

#include <snicholls/core/endian.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace snicholls::formats::llmc
{
/// \brief The checkpoint's magic.
inline constexpr std::int32_t checkpoint_magic = 20240326;
/// \brief The token shard's magic.
inline constexpr std::int32_t tokens_magic = 20240520;
/// \brief The header's length in int32s.
inline constexpr std::size_t header_ints = 256;

/// \brief A GPT-2's sizes, as the checkpoint header gives them.
struct gpt2_shape
{
    std::size_t max_seq = 1024;          ///< The position table's rows.
    std::size_t vocab = 50257;           ///< The tokens.
    std::size_t layers = 12;             ///< The blocks.
    std::size_t heads = 12;              ///< The heads.
    std::size_t channels = 768;          ///< d_model.
    std::size_t padded_vocab = 50304;    ///< The token table's rows.
};

/// \brief The sixteen tensors' names, in llm.c's order.
inline constexpr std::array<std::string_view, 16> tensor_names{"wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
                                                                "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb"};

/// \brief Each tensor's shape, in llm.c's order. \param s The sizes. \return The sixteen shapes.
[[nodiscard]] constexpr std::array<tensor_shape, 16> tensor_shapes(const gpt2_shape& s) noexcept
{
    const auto l = s.layers, c = s.channels;
    return {tensor_shape{s.padded_vocab, c}, tensor_shape{s.max_seq, c}, tensor_shape{l, c}, tensor_shape{l, c},
            tensor_shape{l, 3 * c, c}, tensor_shape{l, 3 * c}, tensor_shape{l, c, c}, tensor_shape{l, c},
            tensor_shape{l, c}, tensor_shape{l, c}, tensor_shape{l, 4 * c, c}, tensor_shape{l, 4 * c},
            tensor_shape{l, c, 4 * c}, tensor_shape{l, c}, tensor_shape{c}, tensor_shape{c}};
}

/// \brief The parameters of a shape. \param s The sizes. \return Their number.
[[nodiscard]] constexpr std::uint64_t parameter_count(const gpt2_shape& s) noexcept
{
    std::uint64_t n = 0;
    for (const auto& t : tensor_shapes(s))
        n += *t.elements();
    return n;
}

/// \brief A parsed checkpoint.
struct checkpoint
{
    gpt2_shape shape;                        ///< The sizes.
    dtype type = dtype::f32;                 ///< f32 (version 3) or bf16 (version 5).
    std::span<const std::byte> parameters;   ///< Every parameter, in llm.c's order.
    std::array<tensor_view, 16> tensors;     ///< The sixteen tensors, named as `tensor_names`.
};

/// \brief Parses a checkpoint. \param bytes The file. \return The checkpoint (borrowing `bytes`), or the problem.
[[nodiscard]] inline result<checkpoint> parse_checkpoint(std::span<const std::byte> bytes)
{
    if (bytes.size() < header_ints * 4)
        return fail(errc::truncated, "llm.c checkpoint: shorter than the header");
    const auto h = [&](std::size_t i) { return static_cast<std::int32_t>(load_le<std::uint32_t>(bytes.data() + 4 * i)); };
    if (h(0) != checkpoint_magic)
        return fail(errc::bad_magic, "llm.c checkpoint: magic " + std::to_string(h(0)) + ", not 20240326");
    checkpoint c;
    if (h(1) == 3)
        c.type = dtype::f32;
    else if (h(1) == 5)
        c.type = dtype::bf16;
    else
        return fail(errc::unsupported_version, "llm.c checkpoint: version " + std::to_string(h(1)));
    for (int i = 2; i <= 7; ++i)
        if (h(static_cast<std::size_t>(i)) <= 0)
            return fail(errc::bad_header, "llm.c checkpoint: header field " + std::to_string(i) + " is not positive");
    c.shape = {std::size_t(h(2)), std::size_t(h(3)), std::size_t(h(4)), std::size_t(h(5)), std::size_t(h(6)), std::size_t(h(7))};
    if (c.shape.channels % c.shape.heads != 0 || c.shape.padded_vocab < c.shape.vocab)
        return fail(errc::bad_header, "llm.c checkpoint: heads must divide channels and the padded vocabulary cover the vocabulary");
    const std::uint64_t size = parameter_count(c.shape) * size_of(c.type);
    if (size != bytes.size() - header_ints * 4)
        return fail(errc::bad_shape, "llm.c checkpoint: " + std::to_string(bytes.size() - header_ints * 4) + " bytes of parameters, "
                                         + std::to_string(size) + " expected");
    c.parameters = bytes.subspan(header_ints * 4);
    const auto shapes = tensor_shapes(c.shape);
    std::size_t at = 0;
    for (std::size_t i = 0; i < 16; ++i) {
        const auto n = static_cast<std::size_t>(*shapes[i].elements()) * size_of(c.type);
        c.tensors[i] = tensor_view{tensor_names[i], c.type, shapes[i], c.parameters.subspan(at, n)};
        at += n;
    }
    return ok(c);
}

/// \brief Writes a checkpoint. \param s The sizes. \param type f32 or bf16. \param parameters Every parameter, in
/// llm.c's order. \return The file's bytes, or `bad_dtype` / `bad_shape` / `bad_header`.
[[nodiscard]] inline result<std::vector<std::byte>> write_checkpoint(const gpt2_shape& s, dtype type, std::span<const std::byte> parameters)
{
    if (type != dtype::f32 && type != dtype::bf16)
        return fail(errc::bad_dtype, "llm.c checkpoint: only f32 and bf16 parameters");
    if (s.heads == 0 || s.channels % s.heads != 0 || s.padded_vocab < s.vocab || s.layers == 0 || s.max_seq == 0)
        return fail(errc::bad_header, "llm.c checkpoint: the sizes are not a GPT-2");
    for (const std::size_t v : {s.max_seq, s.vocab, s.layers, s.heads, s.channels, s.padded_vocab})
        if (v > 0x7fffffff)
            return fail(errc::bad_header, "llm.c checkpoint: a size does not fit the int32 header");
    if (parameters.size() != parameter_count(s) * size_of(type))
        return fail(errc::bad_shape, "llm.c checkpoint: " + std::to_string(parameters.size()) + " bytes of parameters for this shape");
    std::vector<std::byte> out(header_ints * 4);
    const std::array<std::int32_t, 8> head{checkpoint_magic, type == dtype::f32 ? 3 : 5, std::int32_t(s.max_seq), std::int32_t(s.vocab),
                                           std::int32_t(s.layers), std::int32_t(s.heads), std::int32_t(s.channels), std::int32_t(s.padded_vocab)};
    for (std::size_t i = 0; i < head.size(); ++i)
        store_le(out.data() + 4 * i, static_cast<std::uint32_t>(head[i]));
    out.insert(out.end(), parameters.begin(), parameters.end());
    return ok(std::move(out));
}

/// \brief Parses a token shard. \param bytes The file. \return The tokens, a view into `bytes` (little-endian
/// uint16), or the problem.
[[nodiscard]] inline result<std::span<const std::byte>> parse_tokens(std::span<const std::byte> bytes)
{
    if (bytes.size() < header_ints * 4)
        return fail(errc::truncated, "llm.c tokens: shorter than the header");
    const auto h = [&](std::size_t i) { return load_le<std::uint32_t>(bytes.data() + 4 * i); };
    if (static_cast<std::int32_t>(h(0)) != tokens_magic)
        return fail(errc::bad_magic, "llm.c tokens: magic " + std::to_string(h(0)) + ", not 20240520");
    if (h(1) != 1)
        return fail(errc::unsupported_version, "llm.c tokens: version " + std::to_string(h(1)));
    const std::uint64_t n = h(2);
    if (n * 2 != bytes.size() - header_ints * 4)
        return fail(errc::bad_shape, "llm.c tokens: the header says " + std::to_string(n) + " tokens");
    return ok(bytes.subspan(header_ints * 4));
}

/// \brief Writes a token shard. \param tokens The tokens. \return The file's bytes, or `too_large` (over 2^32 - 1).
[[nodiscard]] inline result<std::vector<std::byte>> write_tokens(std::span<const std::uint16_t> tokens)
{
    if (tokens.size() > 0xffffffffu)
        return fail(errc::too_large, "llm.c tokens: more than 2^32 - 1 tokens in a shard");
    std::vector<std::byte> out(header_ints * 4 + tokens.size() * 2);
    store_le(out.data(), static_cast<std::uint32_t>(tokens_magic));
    store_le(out.data() + 4, std::uint32_t{1});
    store_le(out.data() + 8, static_cast<std::uint32_t>(tokens.size()));
    for (std::size_t i = 0; i < tokens.size(); ++i)
        store_le(out.data() + header_ints * 4 + 2 * i, tokens[i]);
    return ok(std::move(out));
}
} // namespace snicholls::formats::llmc

#endif // snicholls_formats_llmc_llmc_hpp
