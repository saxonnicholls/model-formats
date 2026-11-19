//
//  checkpoint.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief PyTorch's `torch.save` files (`.bin`, `.pt`, `.pth`), read safely: the zip, the restricted unpickler, the
/// tensors as views.
/// \ingroup pytorch
///
/// \overview
/// Since PyTorch 1.6 `torch.save` writes a zip: `<archive>/data.pkl` (the pickled object), `<archive>/data/<key>` (one
/// member per storage), `<archive>/byteorder`, `<archive>/version`. `parse` opens the zip, runs `unpickle` on
/// `data.pkl` (never Python's unpickler: nothing in the file is executed), and turns every tensor of the state dict
/// into a `tensor_entry` whose bytes are a view into its storage's member. Nested dicts (`{"model": {...}}`) are
/// flattened with `.`; values that are not tensors (an epoch, a learning rate) are listed in `skipped`.
///
/// A tensor that is contiguous (row-major strides) has `data`, a view; one that is not (a transposed view saved as
/// is) has `contiguous == false`, and `materialize` copies it into row-major order.
///
/// The legacy format (before 1.6: a pickle of pickles, with tensors after it) and anything not a zip are
/// `unsupported`; use `torch.save` with its default zip format, or convert the checkpoint to safetensors.
///
/// \relations
/// \see formats::pytorch::unpickle, formats::zip, formats::convert
///
/// \impls
/// One parse; one copy for non-contiguous tensors.
///
/// \contracts
/// Read-only: nothing writes PyTorch files here (write safetensors). Every tensor's extent is checked against its
/// storage, and every storage against its member, before a view is made.
///
/// \errors
/// The unpickler's (`unsafe_pickle` above all), zip's, and: `bad_magic` (not a zip), `unsupported` (the legacy
/// format, big-endian, an unknown storage type), `not_found` (no data.pkl, or a storage member missing),
/// `out_of_bounds` (a tensor outside its storage), `bad_header` (a state dict of the wrong shape).
///
/// \examples
/// \snippet checkpoint_demo.cpp checkpoint
///
/// \perf
/// Not benchmarked: the pickle holds no tensor data, so parsing is fast; views copy nothing.
///
/// \testing
/// tests/pytorch/checkpoint_test.cpp (an archive assembled by hand as torch.save writes it), demos/pytorch/checkpoint_demo.cpp.
///
/// \deps
/// The standard library and core.
///
/// \gotchas
/// Two tensors may share one storage (tied weights): both are listed, viewing the same bytes.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_pytorch_checkpoint_hpp
#define snicholls_formats_pytorch_checkpoint_hpp

#include <snicholls/formats/archive/zip.hpp>
#include <snicholls/formats/pytorch/pickle.hpp>
#include <snicholls/formats/tensor.hpp>

#include <snicholls/core/checked.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::pytorch
{
/// \brief The dtype of a storage type. \param name "FloatStorage" and so on. \return The dtype, or none.
[[nodiscard]] constexpr std::optional<dtype> storage_dtype(std::string_view name) noexcept
{
    struct row
    {
        std::string_view name;
        dtype type;
    };
    constexpr row rows[] = {{"DoubleStorage", dtype::f64},       {"FloatStorage", dtype::f32},  {"HalfStorage", dtype::f16},
                            {"BFloat16Storage", dtype::bf16},    {"LongStorage", dtype::i64},   {"IntStorage", dtype::i32},
                            {"ShortStorage", dtype::i16},        {"CharStorage", dtype::i8},    {"ByteStorage", dtype::u8},
                            {"BoolStorage", dtype::boolean},     {"Float8_e4m3fnStorage", dtype::f8_e4m3},
                            {"Float8_e5m2Storage", dtype::f8_e5m2}};
    for (const auto& r : rows)
        if (r.name == name)
            return r.type;
    return std::nullopt;
}

/// \brief One tensor of a checkpoint.
struct tensor_entry
{
    std::string name;                     ///< Its key in the (flattened) state dict.
    dtype type = dtype::f32;              ///< Its element type.
    tensor_shape shape;                   ///< Its size.
    tensor_shape stride;                  ///< Its stride, in elements.
    std::uint64_t offset = 0;             ///< Its first element in the storage.
    std::string storage_key;              ///< Its storage's key (`<archive>/data/<key>`).
    std::span<const std::byte> storage;   ///< The whole storage's bytes.
    bool contiguous = true;               ///< Whether the strides are row-major.
    std::span<const std::byte> data;      ///< Its bytes, when contiguous (else empty).

    /// \brief The tensor as a view (contiguous tensors only). \return The view.
    [[nodiscard]] tensor_view view() const noexcept { return {name, type, shape, data}; }
    /// \brief A row-major copy, for any strides. \return The tensor.
    [[nodiscard]] tensor materialize() const
    {
        tensor t{name, type, shape, {}};
        const std::size_t es = size_of(type);
        const auto n = static_cast<std::size_t>(*shape.elements());
        t.data.resize(n * es);
        std::array<std::uint64_t, max_rank> index{};
        for (std::size_t i = 0; i < n; ++i) {
            std::uint64_t at = offset;
            for (std::size_t d = 0; d < shape.rank; ++d)
                at += index[d] * stride[d];
            std::memcpy(t.data.data() + i * es, storage.data() + at * es, es);
            for (std::size_t d = shape.rank; d > 0; --d) {   // odometer, innermost first
                if (++index[d - 1] < shape[d - 1])
                    break;
                index[d - 1] = 0;
            }
        }
        return t;
    }
};

/// \brief A parsed checkpoint.
struct checkpoint
{
    std::string prefix;                 ///< The archive's top directory ("archive", "pytorch_model", ...).
    std::vector<tensor_entry> tensors;  ///< In the state dict's order.
    std::vector<std::string> skipped;   ///< Keys whose values are not tensors.

    /// \brief A tensor by name. \param name The key. \return The tensor, or null.
    [[nodiscard]] const tensor_entry* find(std::string_view name) const noexcept
    {
        for (const auto& t : tensors)
            if (t.name == name)
                return &t;
        return nullptr;
    }
};

namespace detail
{
    inline result<tensor_shape> shape_of(const pickled& p, std::size_t tuple, std::string_view what)
    {
        tensor_shape s;
        for (const std::size_t i : p.at(tuple).items) {
            const object& e = p.at(i);
            if (e.kind != kind::integer || e.integer < 0 || !s.push_back(static_cast<std::uint64_t>(e.integer)))
                return fail(errc::bad_shape, "pytorch: a tensor's " + std::string(what) + " is not a tuple of at most 8 non-negative ints");
        }
        return ok(s);
    }
    inline result<void> flatten(const pickled& p, std::size_t dict, const std::string& prefix, const zip::archive& a,
                                const std::string& root, checkpoint& out, int depth)
    {
        if (depth > 16)
            return fail(errc::bad_header, "pytorch: dicts nested more than 16 deep");
        const auto& items = p.at(dict).items;
        for (std::size_t k = 0; k + 1 < items.size(); k += 2) {
            const object& key = p.at(items[k]);
            if (key.kind != kind::string && key.kind != kind::integer)
                return fail(errc::bad_header, "pytorch: a state dict key is neither a string nor an int");
            const std::string name = prefix + (key.kind == kind::string ? key.text : std::to_string(key.integer));
            const object& v = p.at(items[k + 1]);
            if (v.kind == kind::dict) {
                SNICHOLLS_TRY_VOID(flatten(p, items[k + 1], name + ".", a, root, out, depth + 1));
                continue;
            }
            if (v.kind != kind::tensor) {
                out.skipped.push_back(name);
                continue;
            }
            const object& st = p.at(v.items[0]);
            tensor_entry t;
            t.name = name;
            const auto type = storage_dtype(st.text);
            if (!type)
                return fail(errc::unsupported, "pytorch: " + quote_name(name) + " has storage type " + quote_name(st.text));
            t.type = *type;
            t.storage_key = st.key;
            if (p.at(v.items[1]).integer < 0)
                return fail(errc::bad_header, "pytorch: " + quote_name(name) + " has a negative storage offset");
            t.offset = static_cast<std::uint64_t>(p.at(v.items[1]).integer);
            SNICHOLLS_TRY_ASSIGN(t.shape, shape_of(p, v.items[2], "size"));
            SNICHOLLS_TRY_ASSIGN(t.stride, shape_of(p, v.items[3], "stride"));
            if (t.stride.rank != t.shape.rank)
                return fail(errc::bad_shape, "pytorch: " + quote_name(name) + " has a stride of another rank");
            const zip::entry* member = a.find(root + "/data/" + t.storage_key);
            if (!member)
                return fail(errc::not_found, "pytorch: storage " + quote_name(t.storage_key) + " of " + quote_name(name) + " is missing");
            const auto storage_bytes = checked_mul(static_cast<std::uint64_t>(st.integer), std::uint64_t{size_of(t.type)});
            if (!storage_bytes || *storage_bytes > member->data.size())
                return fail(errc::out_of_bounds, "pytorch: storage " + quote_name(t.storage_key) + " is shorter than its element count");
            t.storage = member->data.first(static_cast<std::size_t>(*storage_bytes));
            // the extent: offset + sum((size - 1) * stride) + 1 elements, unless empty
            const auto n = t.shape.elements();
            if (!n)
                return fail(errc::bad_shape, "pytorch: " + quote_name(name) + "'s size overflows");
            std::uint64_t expected = 1;
            for (std::size_t d = t.shape.rank; d > 0; --d) {
                if (t.shape[d - 1] != 1 && t.stride[d - 1] != expected)
                    t.contiguous = false;
                expected *= t.shape[d - 1];
            }
            if (*n > 0) {
                std::optional<std::uint64_t> last = t.offset;
                for (std::size_t d = 0; d < t.shape.rank && last; ++d) {
                    const auto step = checked_mul(t.shape[d] - 1, t.stride[d]);
                    last = step ? checked_add(*last, *step) : std::nullopt;
                }
                if (!last || *last >= static_cast<std::uint64_t>(st.integer))
                    return fail(errc::out_of_bounds, "pytorch: " + quote_name(name) + " reaches past its storage");
                if (t.contiguous)
                    t.data = t.storage.subspan(static_cast<std::size_t>(t.offset * size_of(t.type)), static_cast<std::size_t>(*n * size_of(t.type)));
            }
            out.tensors.push_back(std::move(t));
        }
        return ok();
    }
} // namespace detail

/// \brief Parses a `torch.save` zip. \param bytes The file. \param limits The unpickler's bounds.
/// \return The checkpoint (borrowing `bytes`), or the first problem found.
[[nodiscard]] inline result<checkpoint> parse(std::span<const std::byte> bytes, const unpickle_limits& limits = {})
{
    if (bytes.size() >= 2 && bytes[0] == std::byte{0x80} && std::to_integer<int>(bytes[1]) <= 5)
        return fail(errc::unsupported, "pytorch: the legacy (pre-1.6, non-zip) format is not read; re-save with torch.save's default or convert to safetensors");
    if (bytes.size() < 4 || load_le<std::uint32_t>(bytes.data()) != 0x04034b50)
        return fail(errc::bad_magic, "pytorch: not a zip archive");
    SNICHOLLS_TRY_ASSIGN(const zip::archive a, zip::parse(bytes));
    checkpoint c;
    const zip::entry* pkl = nullptr;
    for (const auto& e : a.entries) {
        const auto slash = e.name.find('/');
        if (slash != std::string_view::npos && e.name.substr(slash) == "/data.pkl") {
            if (pkl)
                return fail(errc::bad_header, "pytorch: more than one data.pkl");
            pkl = &e;
            c.prefix = std::string(e.name.substr(0, slash));
        }
    }
    if (!pkl)
        return fail(errc::not_found, "pytorch: no <archive>/data.pkl");
    if (const zip::entry* order = a.find(c.prefix + "/byteorder")) {
        const std::string_view o(reinterpret_cast<const char*>(order->data.data()), order->data.size());
        if (o != "little")
            return fail(errc::unsupported, "pytorch: byte order " + quote_name(o));
    }
    SNICHOLLS_TRY_ASSIGN(const pickled p, unpickle(pkl->data, limits));
    if (p.at(p.root).kind == kind::tensor) {   // torch.save(tensor): one tensor, named ""
        object wrapper;
        wrapper.kind = kind::dict;
        pickled q = p;
        object key;
        key.kind = kind::string;
        q.objects.push_back(key);
        wrapper.items = {q.objects.size() - 1, p.root};
        q.objects.push_back(wrapper);
        SNICHOLLS_TRY_VOID(detail::flatten(q, q.objects.size() - 1, "", a, c.prefix, c, 0));
        return ok(std::move(c));
    }
    if (p.at(p.root).kind != kind::dict)
        return fail(errc::bad_header, "pytorch: the saved object is not a dict of tensors");
    SNICHOLLS_TRY_VOID(detail::flatten(p, p.root, "", a, c.prefix, c, 0));
    return ok(std::move(c));
}
} // namespace snicholls::formats::pytorch

#endif // snicholls_formats_pytorch_checkpoint_hpp
