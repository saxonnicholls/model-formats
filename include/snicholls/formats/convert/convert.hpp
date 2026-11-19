//
//  convert.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Any format to any: detect a file's format, read its tensors, write them in another, cast and dequantise.
/// \ingroup convert
///
/// \overview
/// `detect` names a file's format from its first bytes (safetensors' length-prefixed `{`, `GGUF`, `\x93NUMPY`, a zip
/// holding `.npy` members or a `data.pkl`, llm.c's magics); `read_any` reads any of them into one `loaded` set of
/// tensors and string metadata, as views where the bytes allow it and as owned copies where they do not (GGUF's
/// quantised tensors dequantised to F32 through core, PyTorch's non-contiguous tensors made row-major). `write_any`
/// writes a set of tensors as safetensors, GGUF, `.npz` or `.npy`. `cast` changes a tensor's dtype; `dequantize`
/// turns a GGUF tensor of a type core has (Q8_0, Q4_0, MXFP4) into F32.
///
/// This is the tensor-level conversion: names and layouts are kept as they are. Model-level conversion (Hugging Face
/// names to llama.cpp's, the RoPE permutation, GPT-2's transposes, stacked experts) is `convert::export.hpp`'s.
///
/// \relations
/// \see formats::convert::hf_to_gguf, formats::io::open_model (from a path)
///
/// \impls
/// One reader per format behind `read_any`; one writer per format behind `write_any`.
///
/// \contracts
/// A `loaded` borrows the bytes it was read from for its views and owns its copies; moving it keeps every view valid.
///
/// \errors
/// The format readers' errors; `unsupported` for an unknown format, a GGUF quantisation core does not have, or a
/// target that cannot hold a dtype (NumPy and bfloat16, GGUF and unsigned integers: `cast` first).
///
/// \examples
/// \snippet convert_demo.cpp convert
///
/// \perf
/// Views cost nothing; casts and dequantisation are one pass per element.
///
/// \testing
/// tests/convert/convert_test.cpp (round trips between every pair of formats), demos/convert/convert_demo.cpp.
///
/// \deps
/// Every format header here; core's `ml/quantize.hpp`.
///
/// \gotchas
/// GGUF metadata becomes `key = value` text in `loaded::metadata` (arrays summarised by their length), so a GGUF to
/// GGUF round trip through `write_any` keeps strings and numbers as strings: use `gguf::writer` to keep types.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_convert_convert_hpp
#define snicholls_formats_convert_convert_hpp

#include <snicholls/formats/archive/zip.hpp>
#include <snicholls/formats/gguf/gguf.hpp>
#include <snicholls/formats/llmc/llmc.hpp>
#include <snicholls/formats/numpy/npy.hpp>
#include <snicholls/formats/pytorch/checkpoint.hpp>
#include <snicholls/formats/safetensors/safetensors.hpp>

#include <snicholls/core/ml/quantize.hpp>

#include <cstddef>
#include <cstring>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::formats::convert
{
/// \brief A file format.
enum class file_format : std::uint8_t
{
    safetensors,       ///< `.safetensors`.
    gguf,              ///< `.gguf`.
    npy,               ///< `.npy`.
    npz,               ///< `.npz`.
    pytorch,           ///< `torch.save`'s zip: `.bin`, `.pt`, `.pth`.
    llmc_checkpoint,   ///< llm.c's GPT-2 checkpoint, `.bin`.
    llmc_tokens,       ///< llm.c's token shard, `.bin`.
};

/// \brief A format's name. \param f The format. \return "safetensors", "gguf", ...
[[nodiscard]] constexpr std::string_view to_string(file_format f) noexcept
{
    switch (f) {
    case file_format::safetensors: return "safetensors";
    case file_format::gguf: return "gguf";
    case file_format::npy: return "npy";
    case file_format::npz: return "npz";
    case file_format::pytorch: return "pytorch";
    case file_format::llmc_checkpoint: return "llm.c checkpoint";
    case file_format::llmc_tokens: return "llm.c tokens";
    }
    return "?";
}

/// \brief The format a path's extension names (for writing). \param path The path.
/// \return The format, or none (`.bin` is ambiguous: detect it from its bytes).
[[nodiscard]] constexpr std::optional<file_format> format_from_extension(std::string_view path) noexcept
{
    if (path.ends_with(".safetensors"))
        return file_format::safetensors;
    if (path.ends_with(".gguf"))
        return file_format::gguf;
    if (path.ends_with(".npy"))
        return file_format::npy;
    if (path.ends_with(".npz"))
        return file_format::npz;
    if (path.ends_with(".pt") || path.ends_with(".pth"))
        return file_format::pytorch;
    return std::nullopt;
}

/// \brief The format of a file from its bytes. \param bytes The file (the first few hundred bytes suffice, but a zip
/// needs all of it to be told apart). \return The format, or none.
[[nodiscard]] inline std::optional<file_format> detect(std::span<const std::byte> bytes) noexcept
{
    if (bytes.size() >= 4 && load_le<std::uint32_t>(bytes.data()) == gguf::magic)
        return file_format::gguf;
    if (bytes.size() >= 6 && std::memcmp(bytes.data(), "\x93NUMPY", 6) == 0)
        return file_format::npy;
    if (bytes.size() >= 1024) {
        const auto m = static_cast<std::int32_t>(load_le<std::uint32_t>(bytes.data()));
        if (m == llmc::checkpoint_magic)
            return file_format::llmc_checkpoint;
        if (m == llmc::tokens_magic)
            return file_format::llmc_tokens;
    }
    if (bytes.size() >= 4 && load_le<std::uint32_t>(bytes.data()) == 0x04034b50) {
        const auto a = zip::parse(bytes);
        if (a.is_ok())
            for (const auto& e : a->entries)
                if (e.name.ends_with("/data.pkl"))
                    return file_format::pytorch;
        return file_format::npz;
    }
    if (bytes.size() >= 9 && std::to_integer<char>(bytes[8]) == '{' && load_le<std::uint64_t>(bytes.data()) <= bytes.size() - 8)
        return file_format::safetensors;
    return std::nullopt;
}

/// \brief Tensors read from one file.
struct loaded
{
    file_format format = file_format::safetensors;               ///< What the file was.
    std::vector<tensor_view> tensors;                            ///< Every tensor, in the file's order.
    std::deque<tensor> owned;                                    ///< Copies some views point into (stable addresses).
    std::vector<std::pair<std::string, std::string>> metadata;   ///< String metadata, as text.
    std::vector<std::string> notes;                              ///< What was changed or left out, one line each.

    /// \brief A tensor by name. \param name The name. \return The view, or null.
    [[nodiscard]] const tensor_view* find(std::string_view name) const noexcept
    {
        for (const auto& t : tensors)
            if (t.name == name)
                return &t;
        return nullptr;
    }
};

/// \brief A copy of a tensor in another dtype. \param t The tensor. \param to The dtype.
/// \return The copy, or `convert_elements`'s errors.
[[nodiscard]] inline result<tensor> cast(const tensor_view& t, dtype to)
{
    tensor out{std::string(t.name), to, t.shape, {}};
    out.data.resize(t.data.size() / size_of(t.type) * size_of(to));
    SNICHOLLS_TRY_VOID(convert_elements(t.type, t.data, to, out.data));
    return ok(std::move(out));
}

/// \brief A GGUF tensor as F32. \param t The tensor. \return The F32 copy, or `unsupported` for a type core has no
/// layout for.
[[nodiscard]] inline result<tensor> dequantize(const gguf::tensor_info& t)
{
    if (const auto v = t.dense())
        return cast(*v, dtype::f32);
    tensor out{std::string(t.name), dtype::f32, t.shape, {}};
    const auto n = t.shape.elements();
    if (!n)
        return fail(errc::bad_shape, "dequantize: " + quote_name(t.name) + " overflows");
    out.data.resize(static_cast<std::size_t>(*n) * 4);
    const auto f = std::span(reinterpret_cast<float*>(out.data.data()), static_cast<std::size_t>(*n));
    const auto blocks = [&]<typename B>(std::type_identity<B>) {
        const std::span<const B> b(reinterpret_cast<const B*>(t.data.data()), t.data.size() / sizeof(B));
        ml::dequantize(b, f);
    };
    switch (t.type) {
    case gguf::ggml_type::q8_0: blocks(std::type_identity<ml::block_q8_0>{}); break;
    case gguf::ggml_type::q4_0: blocks(std::type_identity<ml::block_q4_0>{}); break;
    case gguf::ggml_type::mxfp4: blocks(std::type_identity<ml::block_mxfp4>{}); break;
    default:
        return fail(errc::unsupported, "dequantize: " + std::string(gguf::traits(t.type).name) + " (" + quote_name(t.name)
                                           + ") waits for core's quantisation track");
    }
    return ok(std::move(out));
}

namespace detail
{
    inline std::string value_text(const gguf::value& v)
    {
        switch (v.type) {
        case gguf::value_type::string: return std::string(*v.as_string());
        case gguf::value_type::boolean: return *v.as_bool() ? "true" : "false";
        case gguf::value_type::f32: case gguf::value_type::f64: return std::to_string(*v.as_f64());
        case gguf::value_type::array: return "[" + std::to_string(v.count) + " x " + std::string(gguf::to_string(v.element_type)) + "]";
        case gguf::value_type::i8: case gguf::value_type::i16: case gguf::value_type::i32: case gguf::value_type::i64:
            return std::to_string(*v.as_i64());
        default: return std::to_string(*v.as_u64());
        }
    }
} // namespace detail

/// \brief What `read_any` does with what it cannot view.
struct read_options
{
    bool dequantize = true;   ///< Dequantise GGUF's quantised tensors to F32 (else they are refused).
};

/// \brief Reads every tensor of a file in any format. \param bytes The file. \param opt What to do with quantised
/// tensors. \return The tensors (borrowing `bytes`), or the problem.
[[nodiscard]] inline result<loaded> read_any(std::span<const std::byte> bytes, const read_options& opt = {})
{
    const auto f = detect(bytes);
    if (!f)
        return fail(errc::unsupported, "convert: not a format this library reads");
    loaded l;
    l.format = *f;
    switch (*f) {
    case file_format::safetensors: {
        SNICHOLLS_TRY_ASSIGN(safetensors::file s, safetensors::parse(bytes));
        for (auto& e : s.tensors) {   // names are owned by the parsed file: keep a copy beside the view
            l.owned.push_back(tensor{std::move(e.name), e.type, e.shape, {}});
            l.tensors.push_back(tensor_view{l.owned.back().name, e.type, e.shape, e.data});
        }
        l.metadata = std::move(s.metadata);
        break;
    }
    case file_format::gguf: {
        SNICHOLLS_TRY_ASSIGN(const gguf::file g, gguf::parse(bytes));
        for (const auto& e : g.metadata)
            l.metadata.emplace_back(std::string(e.key), detail::value_text(e.val));
        for (const auto& t : g.tensors) {
            if (const auto v = t.dense()) {
                l.tensors.push_back(*v);
                continue;
            }
            if (!opt.dequantize)
                return fail(errc::unsupported, "convert: " + quote_name(t.name) + " is quantised (" + std::string(gguf::traits(t.type).name) + ")");
            SNICHOLLS_TRY_ASSIGN(tensor d, dequantize(t));
            l.notes.push_back(std::string(t.name) + ": dequantised from " + std::string(gguf::traits(t.type).name) + " to f32");
            l.owned.push_back(std::move(d));
            l.tensors.push_back(l.owned.back().view());
        }
        break;
    }
    case file_format::npy: {
        SNICHOLLS_TRY_ASSIGN(const tensor_view v, numpy::parse_npy(bytes, "array"));
        l.tensors.push_back(v);
        break;
    }
    case file_format::npz: {
        SNICHOLLS_TRY_ASSIGN(l.tensors, numpy::parse_npz(bytes));
        break;
    }
    case file_format::pytorch: {
        SNICHOLLS_TRY_ASSIGN(pytorch::checkpoint c, pytorch::parse(bytes));
        for (auto& t : c.tensors) {
            if (t.contiguous) {
                l.owned.push_back(tensor{t.name, t.type, t.shape, {}});
                l.tensors.push_back(tensor_view{l.owned.back().name, t.type, t.shape, t.data});
            } else {
                l.owned.push_back(t.materialize());
                l.tensors.push_back(l.owned.back().view());
                l.notes.push_back(t.name + ": not contiguous; copied to row-major order");
            }
        }
        for (const auto& s : c.skipped)
            l.notes.push_back(s + ": not a tensor; left out");
        break;
    }
    case file_format::llmc_checkpoint: {
        SNICHOLLS_TRY_ASSIGN(const llmc::checkpoint c, llmc::parse_checkpoint(bytes));
        l.tensors.assign(c.tensors.begin(), c.tensors.end());
        const auto& s = c.shape;
        for (const auto& [k, v] : {std::pair<const char*, std::size_t>{"max_seq", s.max_seq}, {"vocab", s.vocab}, {"layers", s.layers},
                                   {"heads", s.heads}, {"channels", s.channels}, {"padded_vocab", s.padded_vocab}})
            l.metadata.emplace_back(std::string("llmc.") + k, std::to_string(v));
        break;
    }
    case file_format::llmc_tokens: {
        SNICHOLLS_TRY_ASSIGN(const std::span<const std::byte> t, llmc::parse_tokens(bytes));
        l.tensors.push_back(tensor_view{"tokens", dtype::u16, tensor_shape{t.size() / 2}, t});
        break;
    }
    }
    return ok(std::move(l));
}

/// \brief Writes tensors in a format. \param to The format (safetensors, gguf, npz, or npy for one tensor).
/// \param tensors The tensors. \param metadata String metadata (safetensors' `__metadata__`; GGUF string keys).
/// \return The file's bytes, or the problem.
[[nodiscard]] inline result<std::vector<std::byte>> write_any(file_format to, std::span<const tensor_view> tensors,
                                                              std::span<const std::pair<std::string, std::string>> metadata = {})
{
    switch (to) {
    case file_format::safetensors: return safetensors::write(tensors, metadata);
    case file_format::gguf: {
        gguf::writer w;
        for (const auto& [k, v] : metadata)
            w.set(k, gguf::value_string(v));
        for (const auto& t : tensors)
            SNICHOLLS_TRY_VOID(w.add_tensor(t));
        return w.write();
    }
    case file_format::npz: return numpy::write_npz(tensors);
    case file_format::npy:
        if (tensors.size() != 1)
            return fail(errc::bad_argument, "convert: an .npy holds exactly one array, not " + std::to_string(tensors.size()));
        return numpy::write_npy(tensors[0]);
    default: return fail(errc::unsupported, "convert: " + std::string(to_string(to)) + " is not written by write_any");
    }
}
} // namespace snicholls::formats::convert

#endif // snicholls_formats_convert_convert_hpp
