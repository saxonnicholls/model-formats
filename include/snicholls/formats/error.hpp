//
//  error.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The library's one error type, `formats::error`, and `formats::result<T>`.
/// \ingroup formats
///
/// \overview
/// Every reader, writer and the Hub client report an expected failure as a value: a `result<T>`, which is core's
/// `snicholls::result<T, error>`. An `error` is a code (`errc`, what kind of failure, for a program to branch on) and a
/// message (where and why, for a person). Exceptions are not used for malformed input: a model file from the internet
/// is untrusted, and a malformed one is an expected outcome.
///
/// \relations
/// \see snicholls::result (core's `result.hpp`)
///
/// \impls
/// One enumeration, one struct.
///
/// \contracts
/// `to_string` is `noexcept` and never returns null. An `error` owns its message.
///
/// \errors
/// None of its own.
///
/// \examples
/// \snippet error_demo.cpp error
///
/// \perf
/// Not benchmarked: only on failure paths.
///
/// \testing
/// tests/error_test.cpp, demos/error_demo.cpp.
///
/// \deps
/// Core's `result.hpp`; `<string>`.
///
/// \gotchas
/// A message may quote a name from the file (a tensor name, a key), cut to 64 characters; it never quotes a secret
/// (the Hub token is never part of an error).
///
/// \status
/// Experimental.

#ifndef snicholls_formats_error_hpp
#define snicholls_formats_error_hpp

#include <snicholls/core/result.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace snicholls::formats
{
/// \brief What kind of failure.
enum class errc : std::uint8_t
{
    truncated = 1,         ///< The bytes end before the format says they do.
    bad_magic,             ///< Not this format: the magic number or signature is wrong.
    unsupported_version,   ///< This format, at a version not supported.
    bad_header,            ///< The header is malformed (a field out of range, a wrong type).
    bad_json,              ///< JSON that does not parse, or has the wrong shape.
    bad_dtype,             ///< An element type not known, or not allowed here.
    bad_shape,             ///< A shape that does not match the data, or too many dimensions.
    out_of_bounds,         ///< An offset or size outside the data.
    overlap,               ///< Two tensors' data overlap, or leave a hole where the format forbids one.
    misaligned,            ///< Data not at the alignment the format requires.
    duplicate_name,        ///< Two tensors, keys or files of one name.
    not_found,             ///< A name, file or key asked for is not there.
    unsupported,           ///< Valid, but a feature this library does not implement (deflate, a quant type).
    unsafe_pickle,         ///< A pickle opcode or global outside the restricted set.
    checksum_mismatch,     ///< A CRC or digest does not match.
    io,                    ///< The operating system failed a file operation.
    http,                  ///< The network or the server failed a request.
    offline,               ///< Offline mode, and the file is not in the cache.
    too_large,             ///< Beyond a limit this library sets (a header size, a count).
    bad_argument,          ///< The caller passed something invalid (a bad repository name, an empty path).
};

/// \brief The name of a code.
/// \param e The code.
/// \return A static string, never null.
[[nodiscard]] constexpr const char* to_string(errc e) noexcept
{
    switch (e) {
    case errc::truncated: return "truncated";
    case errc::bad_magic: return "bad magic";
    case errc::unsupported_version: return "unsupported version";
    case errc::bad_header: return "bad header";
    case errc::bad_json: return "bad JSON";
    case errc::bad_dtype: return "bad dtype";
    case errc::bad_shape: return "bad shape";
    case errc::out_of_bounds: return "out of bounds";
    case errc::overlap: return "overlap";
    case errc::misaligned: return "misaligned";
    case errc::duplicate_name: return "duplicate name";
    case errc::not_found: return "not found";
    case errc::unsupported: return "unsupported";
    case errc::unsafe_pickle: return "unsafe pickle";
    case errc::checksum_mismatch: return "checksum mismatch";
    case errc::io: return "I/O error";
    case errc::http: return "HTTP error";
    case errc::offline: return "offline";
    case errc::too_large: return "too large";
    case errc::bad_argument: return "bad argument";
    }
    return "unknown";
}

/// \brief A failure: its kind and what happened.
struct error
{
    errc code = errc::bad_argument;   ///< The kind.
    std::string message;              ///< Where and why.

    /// \brief "<kind>: <message>". \return The text.
    [[nodiscard]] std::string what() const { return message.empty() ? std::string(to_string(code)) : std::string(to_string(code)) + ": " + message; }
};

/// \brief A value or an `error`: core's `result`.
/// \tparam T The value type (or `void`).
template <typename T>
using result = snicholls::result<T, error>;

/// \brief An error to return from a function that returns any `result<T>`.
/// \param code The kind. \param message Where and why.
/// \return The error, convertible to every `result<T>`.
[[nodiscard]] inline auto fail(errc code, std::string message) { return snicholls::err(error{code, std::move(message)}); }

/// \brief A name from a file, cut to 64 characters, for a message.
/// \param s The name.
/// \return At most 64 characters of it, with "..." if cut.
[[nodiscard]] inline std::string quote_name(std::string_view s)
{
    constexpr std::size_t max = 64;
    std::string out = "\"";
    out.append(s.substr(0, max));
    if (s.size() > max)
        out += "...";
    out += '"';
    return out;
}
} // namespace snicholls::formats

#endif // snicholls_formats_error_hpp
