//
//  crc32.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320), as zip stores it. A byte-at-a-time table, built at
//  compile time. Core owns every constexpr hash and checksum (its approved hash/crc.hpp, not yet written): this moves
//  there when it lands, and the zip reader calls core's.
//

#ifndef snicholls_formats_detail_crc32_hpp
#define snicholls_formats_detail_crc32_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls::formats::detail
{
inline constexpr std::array<std::uint32_t, 256> crc32_table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        t[i] = c;
    }
    return t;
}();

// Continues a CRC: crc32(b, crc32(a)) == crc32(a + b); the CRC of nothing is 0
constexpr std::uint32_t crc32(std::span<const std::byte> data, std::uint32_t crc = 0) noexcept
{
    crc = ~crc;
    for (const std::byte b : data)
        crc = crc32_table[(crc ^ std::to_integer<std::uint32_t>(b)) & 0xff] ^ (crc >> 8);
    return ~crc;
}
} // namespace snicholls::formats::detail

#endif // snicholls_formats_detail_crc32_hpp
