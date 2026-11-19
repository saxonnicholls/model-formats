//
//  byte_level.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  GPT-2's byte-level alphabet (`bytes_to_unicode` in its encoder.py, and Hugging Face's ByteLevel pre-tokenizer):
//  each of the 256 bytes is one printable code point, so a vocabulary of byte strings can be written as text. The
//  bytes '!'..'~', 0xA1..0xAC and 0xAE..0xFF are their own code points; the other 68 are 256, 257, ... in byte order.
//  Byte-level BPE is core's (ml/bpe.hpp), so this alphabet belongs beside it; it moves there when core takes it
//  (proposal in family/proposals/model-formats.md).
//

#ifndef snicholls_formats_detail_byte_level_hpp
#define snicholls_formats_detail_byte_level_hpp

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace snicholls::formats::detail
{
constexpr bool byte_is_printable(unsigned b) noexcept
{
    return (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
}

// The code point of each byte
inline constexpr std::array<std::uint16_t, 256> byte_to_code_point = [] {
    std::array<std::uint16_t, 256> t{};
    unsigned next = 256;
    for (unsigned b = 0; b < 256; ++b)
        t[b] = static_cast<std::uint16_t>(byte_is_printable(b) ? b : next++);
    return t;
}();

// The bytes of code points 256 to 323, in order
inline constexpr std::array<std::uint8_t, 68> high_code_point_byte = [] {
    std::array<std::uint8_t, 68> t{};
    for (unsigned b = 0; b < 256; ++b)
        if (byte_to_code_point[b] >= 256)
            t[byte_to_code_point[b] - 256] = static_cast<std::uint8_t>(b);
    return t;
}();

// The byte of a code point of the alphabet, or none
constexpr std::optional<std::uint8_t> code_point_to_byte(std::uint32_t cp) noexcept
{
    if (cp < 256)
        return byte_is_printable(cp) ? std::optional<std::uint8_t>(static_cast<std::uint8_t>(cp)) : std::nullopt;
    if (cp < 256 + high_code_point_byte.size())
        return high_code_point_byte[cp - 256];
    return std::nullopt;
}

inline void append_utf8(std::string& out, std::uint32_t cp)
{
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

// The next code point of UTF-8 text at `at`, advancing it; none if malformed
constexpr std::optional<std::uint32_t> next_code_point(std::string_view s, std::size_t& at) noexcept
{
    const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(s[i]); };
    const unsigned c = byte(at);
    std::size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || at + n > s.size())
        return std::nullopt;
    std::uint32_t cp = n == 1 ? c : c & (0x7f >> n);
    for (std::size_t i = 1; i < n; ++i) {
        if ((byte(at + i) & 0xc0) != 0x80)
            return std::nullopt;
        cp = (cp << 6) | (byte(at + i) & 0x3f);
    }
    at += n;
    return cp;
}

// Byte-level text to bytes ("Ġhello" to " hello"); none if a code point is outside the alphabet
inline std::optional<std::string> byte_level_decode(std::string_view text)
{
    std::string out;
    out.reserve(text.size());
    for (std::size_t at = 0; at < text.size();) {
        const auto cp = next_code_point(text, at);
        if (!cp)
            return std::nullopt;
        const auto b = code_point_to_byte(*cp);
        if (!b)
            return std::nullopt;
        out.push_back(static_cast<char>(*b));
    }
    return out;
}

// Bytes to byte-level text
inline std::string byte_level_encode(std::string_view bytes)
{
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const char c : bytes)
        append_utf8(out, byte_to_code_point[static_cast<unsigned char>(c)]);
    return out;
}
} // namespace snicholls::formats::detail

#endif // snicholls_formats_detail_byte_level_hpp
