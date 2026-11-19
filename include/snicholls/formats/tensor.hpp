//
//  tensor.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `tensor_shape`, `tensor_view` (a name, a dtype, a shape and bytes, borrowed) and `tensor` (owned).
/// \ingroup formats
///
/// \overview
/// The common currency of the readers and writers. A reader returns `tensor_view`s whose `data` points into the bytes
/// it was given (a mapped file, usually), so opening a 10 GB checkpoint copies nothing; a writer takes a span of
/// `tensor_view`s. `tensor` owns its name and bytes, for the results of a conversion.
///
/// Shapes are written outermost first (row-major, as safetensors, NumPy and PyTorch write them); GGUF's reversed order
/// is converted at the GGUF boundary.
///
/// \relations
/// \see formats::dtype, formats::safetensors::file, formats::gguf::file
///
/// \impls
/// `tensor_shape` is a fixed array of `max_rank` extents and a rank: trivially copyable, no allocation.
///
/// \contracts
/// A `tensor_view` borrows: the bytes behind `name` and `data` must outlive it. `elements()` and `byte_size()` return
/// none on overflow rather than wrapping.
///
/// \errors
/// `tensor_shape::push_back` returns false past `max_rank`; `checked_byte_size` returns none on overflow.
///
/// \examples
/// \snippet tensor_demo.cpp tensor
///
/// \perf
/// Not benchmarked: plain data.
///
/// \testing
/// tests/tensor_test.cpp, demos/tensor_demo.cpp.
///
/// \deps
/// Core's `checked.hpp`.
///
/// \gotchas
/// A rank-0 tensor (a scalar) has one element.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_tensor_hpp
#define snicholls_formats_tensor_hpp

#include <snicholls/formats/dtype.hpp>

#include <snicholls/core/checked.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats
{
/// \brief The most dimensions a tensor may have here (GGUF allows 4; no checkpoint in use has more than 6).
inline constexpr std::size_t max_rank = 8;

/// \brief A shape, outermost dimension first.
struct tensor_shape
{
    std::array<std::uint64_t, max_rank> dims{};   ///< The extents; only the first `rank` are used.
    std::size_t rank = 0;                         ///< The number of dimensions.

    constexpr tensor_shape() noexcept = default;
    /// \brief A shape from its extents. \param d The extents, outermost first (at most `max_rank`; extra are dropped).
    constexpr tensor_shape(std::initializer_list<std::uint64_t> d) noexcept
    {
        for (const auto e : d)
            if (!push_back(e))
                break;
    }

    /// \brief Appends an extent. \param e The extent. \return False if the shape already has `max_rank`.
    constexpr bool push_back(std::uint64_t e) noexcept
    {
        if (rank == max_rank)
            return false;
        dims[rank++] = e;
        return true;
    }
    /// \brief The extents in use. \return A view of the first `rank`.
    [[nodiscard]] constexpr std::span<const std::uint64_t> view() const noexcept { return {dims.data(), rank}; }
    /// \brief One extent. \param i The dimension, below `rank`. \return Its extent.
    [[nodiscard]] constexpr std::uint64_t operator[](std::size_t i) const noexcept { return dims[i]; }
    /// \brief The number of elements. \return The product of the extents (1 for a scalar), or none on overflow.
    [[nodiscard]] constexpr std::optional<std::uint64_t> elements() const noexcept
    {
        std::uint64_t n = 1;
        for (std::size_t i = 0; i < rank; ++i) {
            const auto m = checked_mul(n, dims[i]);
            if (!m)
                return std::nullopt;
            n = *m;
        }
        return n;
    }
    /// \brief The shape with its dimensions in the opposite order (GGUF's). \return The reversed shape.
    [[nodiscard]] constexpr tensor_shape reversed() const noexcept
    {
        tensor_shape r;
        for (std::size_t i = rank; i > 0; --i)
            r.push_back(dims[i - 1]);
        return r;
    }
    /// \brief Equal when the ranks and the extents in use are. \param a One. \param b The other. \return Whether equal.
    friend constexpr bool operator==(const tensor_shape& a, const tensor_shape& b) noexcept
    {
        if (a.rank != b.rank)
            return false;
        for (std::size_t i = 0; i < a.rank; ++i)
            if (a.dims[i] != b.dims[i])
                return false;
        return true;
    }
};

/// \brief The bytes of `elements` elements of `t`. \param shape The shape. \param t The dtype. \return The size, or
/// none on overflow.
[[nodiscard]] constexpr std::optional<std::uint64_t> checked_byte_size(const tensor_shape& shape, dtype t) noexcept
{
    const auto n = shape.elements();
    if (!n)
        return std::nullopt;
    return checked_mul(*n, std::uint64_t{size_of(t)});
}

/// \brief "[2, 3]". \param s The shape. \return The text.
[[nodiscard]] inline std::string to_string(const tensor_shape& s)
{
    std::string out = "[";
    for (std::size_t i = 0; i < s.rank; ++i) {
        if (i)
            out += ", ";
        out += std::to_string(s.dims[i]);
    }
    return out + "]";
}

/// \brief A dense tensor, borrowed: its name and bytes belong to someone else.
struct tensor_view
{
    std::string_view name;              ///< The tensor's name.
    dtype type = dtype::f32;            ///< The element type.
    tensor_shape shape;                 ///< The shape, outermost first.
    std::span<const std::byte> data;    ///< The elements, little-endian, row-major.
};

/// \brief A dense tensor that owns its name and bytes.
struct tensor
{
    std::string name;               ///< The tensor's name.
    dtype type = dtype::f32;        ///< The element type.
    tensor_shape shape;             ///< The shape, outermost first.
    std::vector<std::byte> data;    ///< The elements, little-endian, row-major.

    /// \brief A view of this tensor, valid while it lives and is not changed. \return The view.
    [[nodiscard]] tensor_view view() const noexcept { return {name, type, shape, data}; }

    /// \brief A copy of a view. \param v The view. \return The owned tensor.
    [[nodiscard]] static tensor from(const tensor_view& v)
    {
        return {std::string(v.name), v.type, v.shape, std::vector<std::byte>(v.data.begin(), v.data.end())};
    }
};

/// \brief Views of owned tensors, for a writer. \param ts The tensors. \return One view each, in order.
[[nodiscard]] inline std::vector<tensor_view> views(std::span<const tensor> ts)
{
    std::vector<tensor_view> v;
    v.reserve(ts.size());
    for (const auto& t : ts)
        v.push_back(t.view());
    return v;
}

} // namespace snicholls::formats

#endif // snicholls_formats_tensor_hpp
