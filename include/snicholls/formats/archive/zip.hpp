//
//  zip.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Zip archives of stored (uncompressed) members, with ZIP64: the container of `.npz` and PyTorch's `.pt`.
/// \ingroup archive
///
/// \overview
/// NumPy's `savez` and PyTorch's `torch.save` both write zip archives whose members are stored, not compressed, and
/// PyTorch aligns each member's data to 64 bytes so it can be mapped. `parse` reads the central directory (ZIP64
/// included, for archives and members over 4 GB) and returns each member's bytes as a view into the archive: nothing is
/// copied or inflated. `write` writes such an archive, deterministically (fixed timestamps, the order given), with
/// each member's data aligned (64 by default) by padding the local header's extra field, as PyTorch does.
///
/// Validation: the end-of-central-directory record is found within the last 64 KiB plus 22 bytes; every offset and
/// size is checked against the archive before use; a member's local header must agree with the central directory's
/// name length; data ranges may not overlap (the overlapping-member zip bomb); names are unique. A compressed member
/// (deflate, `savez_compressed`) is `unsupported`, as are encryption and multi-disk archives. `verify` checks a
/// member's CRC-32.
///
/// \relations
/// \see formats::numpy::parse_npz, formats::pytorch::parse
///
/// \impls
/// Read and write; a CRC-32 table (`detail/crc32.hpp`, to move to core's checksums).
///
/// \contracts
/// `parse` never reads outside its span and never allocates in proportion to a size field it has not checked.
///
/// \errors
/// `truncated`, `bad_magic` (no end record), `bad_header`, `out_of_bounds`, `overlap`, `duplicate_name`,
/// `unsupported`; `verify`: `checksum_mismatch`. Writing: `duplicate_name`, `bad_argument` (a name over 65535 bytes).
///
/// \examples
/// \snippet zip_demo.cpp zip
///
/// \perf
/// Not benchmarked: the directory walk is linear in the members; `verify` runs at CRC-32's byte-at-a-time speed.
///
/// \testing
/// tests/archive/zip_test.cpp (and Python's zipfile in scripts/crosscheck.sh), demos/archive/zip_demo.cpp.
///
/// \deps
/// The standard library and core.
///
/// \gotchas
/// Data descriptors (flag bit 3) are accepted: the central directory's sizes are authoritative.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_archive_zip_hpp
#define snicholls_formats_archive_zip_hpp

#include <snicholls/formats/detail/crc32.hpp>
#include <snicholls/formats/error.hpp>

#include <snicholls/core/endian.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::zip
{
/// \brief One member of a parsed archive.
struct entry
{
    std::string_view name;             ///< The member's name, as stored in the central directory (in the archive's bytes).
    std::uint32_t crc = 0;             ///< Its CRC-32, from the central directory.
    std::uint64_t offset = 0;          ///< Where its data starts in the archive.
    std::span<const std::byte> data;   ///< Its bytes (stored: the bytes as written).
};

/// \brief A parsed archive.
struct archive
{
    std::vector<entry> entries;   ///< In central-directory order.
    bool zip64 = false;           ///< Whether the archive used ZIP64 records.

    /// \brief A member by name. \param name The name. \return The member, or null.
    [[nodiscard]] const entry* find(std::string_view name) const noexcept
    {
        for (const auto& e : entries)
            if (e.name == name)
                return &e;
        return nullptr;
    }
};

namespace detail
{
    inline constexpr std::uint32_t local_sig = 0x04034b50, central_sig = 0x02014b50, end_sig = 0x06054b50;
    inline constexpr std::uint32_t end64_sig = 0x06064b50, locator64_sig = 0x07064b50;
    inline bool fits(std::span<const std::byte> b, std::uint64_t at, std::uint64_t n) noexcept { return at <= b.size() && n <= b.size() - at; }
    template <typename T>
    T at(std::span<const std::byte> b, std::uint64_t i) noexcept
    {
        return load_le<T>(b.data() + i);
    }
} // namespace detail

/// \brief Parses an archive's directory. \param bytes The whole archive.
/// \return The archive (borrowing `bytes`), or the first problem found.
[[nodiscard]] inline result<archive> parse(std::span<const std::byte> bytes)
{
    using detail::at;
    using detail::fits;
    if (bytes.size() < 22)
        return fail(errc::truncated, "zip: shorter than an end-of-central-directory record");
    std::uint64_t end = bytes.size() - 22;
    const std::uint64_t lowest = bytes.size() > 22 + 65535 ? bytes.size() - 22 - 65535 : 0;
    for (;; --end) {
        if (at<std::uint32_t>(bytes, end) == detail::end_sig && end + 22 + at<std::uint16_t>(bytes, end + 20) == bytes.size())
            break;
        if (end == lowest)
            return fail(errc::bad_magic, "zip: no end-of-central-directory record");
    }
    archive a;
    if (at<std::uint16_t>(bytes, end + 4) != 0 || at<std::uint16_t>(bytes, end + 6) != 0)
        return fail(errc::unsupported, "zip: multi-disk archives are not read");
    std::uint64_t count = at<std::uint16_t>(bytes, end + 10);
    std::uint64_t cd_size = at<std::uint32_t>(bytes, end + 12);
    std::uint64_t cd_offset = at<std::uint32_t>(bytes, end + 16);
    if (count == 0xffff || cd_size == 0xffffffff || cd_offset == 0xffffffff) {
        if (end < 20 || at<std::uint32_t>(bytes, end - 20) != detail::locator64_sig)
            return fail(errc::bad_header, "zip: ZIP64 values without a ZIP64 locator");
        const std::uint64_t rec = at<std::uint64_t>(bytes, end - 20 + 8);
        if (!fits(bytes, rec, 56) || at<std::uint32_t>(bytes, rec) != detail::end64_sig)
            return fail(errc::bad_header, "zip: the ZIP64 end record is not where the locator says");
        count = at<std::uint64_t>(bytes, rec + 32);
        cd_size = at<std::uint64_t>(bytes, rec + 40);
        cd_offset = at<std::uint64_t>(bytes, rec + 48);
        a.zip64 = true;
    }
    if (!fits(bytes, cd_offset, cd_size))
        return fail(errc::out_of_bounds, "zip: the central directory runs past the end");
    if (count > cd_size / 46)
        return fail(errc::bad_header, "zip: more members than the central directory can hold");
    a.entries.reserve(static_cast<std::size_t>(count));
    std::uint64_t p = cd_offset;
    const std::uint64_t cd_end = cd_offset + cd_size;
    for (std::uint64_t i = 0; i < count; ++i) {
        if (p + 46 > cd_end || at<std::uint32_t>(bytes, p) != detail::central_sig)
            return fail(errc::bad_header, "zip: central directory entry " + std::to_string(i) + " is malformed");
        const std::uint16_t flags = at<std::uint16_t>(bytes, p + 8), method = at<std::uint16_t>(bytes, p + 10);
        entry e;
        e.crc = at<std::uint32_t>(bytes, p + 16);
        std::uint64_t csize = at<std::uint32_t>(bytes, p + 20), usize = at<std::uint32_t>(bytes, p + 24);
        const std::uint16_t n = at<std::uint16_t>(bytes, p + 28), x = at<std::uint16_t>(bytes, p + 30), c = at<std::uint16_t>(bytes, p + 32);
        std::uint64_t local = at<std::uint32_t>(bytes, p + 42);
        if (p + 46 + n + x + c > cd_end)
            return fail(errc::bad_header, "zip: central directory entry " + std::to_string(i) + " runs past the directory");
        e.name = std::string_view(reinterpret_cast<const char*>(bytes.data() + p + 46), n);
        // ZIP64 extended information: the fields that were 0xFFFFFFFF, in order
        for (std::uint64_t q = p + 46 + n; q + 4 <= p + 46 + n + x;) {
            const std::uint16_t id = at<std::uint16_t>(bytes, q), len = at<std::uint16_t>(bytes, q + 2);
            if (q + 4 + len > p + 46 + n + x)
                return fail(errc::bad_header, "zip: an extra field of " + quote_name(e.name) + " runs past its entry");
            if (id == 0x0001) {
                std::uint64_t r = q + 4;
                const auto take = [&](std::uint64_t& v) {
                    if (v != 0xffffffff)
                        return true;
                    if (r + 8 > q + 4 + len)
                        return false;
                    v = at<std::uint64_t>(bytes, r);
                    r += 8;
                    return true;
                };
                if (!take(usize) || !take(csize) || !take(local))
                    return fail(errc::bad_header, "zip: the ZIP64 field of " + quote_name(e.name) + " is too short");
            }
            q += 4 + len;
        }
        if (flags & 1)
            return fail(errc::unsupported, "zip: " + quote_name(e.name) + " is encrypted");
        if (method != 0)
            return fail(errc::unsupported, "zip: " + quote_name(e.name) + " is compressed (method " + std::to_string(method)
                                               + "); only stored members are read");
        if (csize != usize)
            return fail(errc::bad_header, "zip: stored member " + quote_name(e.name) + " has different sizes");
        if (!fits(bytes, local, 30) || at<std::uint32_t>(bytes, local) != detail::local_sig)
            return fail(errc::bad_header, "zip: the local header of " + quote_name(e.name) + " is missing");
        if (at<std::uint16_t>(bytes, local + 26) != n)
            return fail(errc::bad_header, "zip: the local header of " + quote_name(e.name) + " disagrees with the directory");
        e.offset = local + 30 + n + at<std::uint16_t>(bytes, local + 28);
        if (!fits(bytes, e.offset, usize))
            return fail(errc::out_of_bounds, "zip: " + quote_name(e.name) + " runs past the end");
        e.data = bytes.subspan(static_cast<std::size_t>(e.offset), static_cast<std::size_t>(usize));
        a.entries.push_back(e);
        p += 46 + n + x + c;
    }
    std::vector<const entry*> order;
    for (const auto& e : a.entries)
        order.push_back(&e);
    std::sort(order.begin(), order.end(), [](const entry* l, const entry* r) { return l->offset < r->offset; });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (order[i]->offset < order[i - 1]->offset + order[i - 1]->data.size())
            return fail(errc::overlap, "zip: " + quote_name(order[i]->name) + " overlaps " + quote_name(order[i - 1]->name));
    std::sort(order.begin(), order.end(), [](const entry* l, const entry* r) { return l->name < r->name; });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (order[i]->name == order[i - 1]->name)
            return fail(errc::duplicate_name, "zip: two members named " + quote_name(order[i]->name));
    return ok(std::move(a));
}

/// \brief Checks a member's CRC-32. \param e The member. \return Nothing, or `checksum_mismatch`.
[[nodiscard]] inline result<void> verify(const entry& e)
{
    if (formats::detail::crc32(e.data) != e.crc)
        return fail(errc::checksum_mismatch, "zip: the CRC-32 of " + quote_name(e.name) + " does not match");
    return ok();
}

/// \brief A member to write.
struct member
{
    std::string_view name;             ///< Its name.
    std::span<const std::byte> data;   ///< Its bytes (borrowed until `write` returns).
};

/// \brief How `write` lays out the archive.
struct write_options
{
    std::size_t alignment = 64;   ///< Each member's data starts at a multiple of this (1 for none).
    bool force_zip64 = false;     ///< Write ZIP64 records even when nothing needs them.
};

/// \brief Writes an archive of stored members. \param members The members, in order. \param opt The layout.
/// \return The archive's bytes, or `duplicate_name` / `bad_argument`.
[[nodiscard]] inline result<std::vector<std::byte>> write(std::span<const member> members, const write_options& opt = {})
{
    std::vector<std::string_view> names;
    for (const auto& m : members) {
        if (m.name.size() > 0xffff || m.name.empty())
            return fail(errc::bad_argument, "zip: a member name must be 1 to 65535 bytes");
        names.push_back(m.name);
    }
    std::sort(names.begin(), names.end());
    if (const auto d = std::adjacent_find(names.begin(), names.end()); d != names.end())
        return fail(errc::duplicate_name, "zip: two members named " + quote_name(*d));
    std::vector<std::byte> out, cd;
    const auto put16 = [](std::vector<std::byte>& v, std::uint16_t x) { for (const auto c : to_le_bytes(x)) v.push_back(std::byte{c}); };
    const auto put32 = [](std::vector<std::byte>& v, std::uint32_t x) { for (const auto c : to_le_bytes(x)) v.push_back(std::byte{c}); };
    const auto put64 = [](std::vector<std::byte>& v, std::uint64_t x) { for (const auto c : to_le_bytes(x)) v.push_back(std::byte{c}); };
    const auto put = [](std::vector<std::byte>& v, std::string_view s) { for (const char c : s) v.push_back(static_cast<std::byte>(c)); };
    constexpr std::uint32_t max32 = 0xffffffff;
    const std::size_t align = opt.alignment ? opt.alignment : 1;
    bool any64 = opt.force_zip64 || members.size() >= 0xffff;
    for (const auto& m : members) {
        const std::uint64_t local = out.size();
        const std::uint64_t size = m.data.size();
        const bool big = opt.force_zip64 || size >= max32 || local >= max32;
        any64 = any64 || big;
        const std::uint32_t crc = formats::detail::crc32(m.data);
        // local header: the ZIP64 field (sizes) when big, then padding so the data is aligned
        const std::size_t fixed = 30 + m.name.size() + (big ? 20 : 0);
        std::size_t pad = (align - (local + fixed + 4) % align) % align;
        put32(out, detail::local_sig);
        put16(out, big ? 45 : 20);
        put16(out, 0);                       // flags
        put16(out, 0);                       // stored
        put16(out, 0);                       // time 00:00
        put16(out, 0x21);                    // date 1980-01-01
        put32(out, crc);
        put32(out, big ? max32 : static_cast<std::uint32_t>(size));
        put32(out, big ? max32 : static_cast<std::uint32_t>(size));
        put16(out, static_cast<std::uint16_t>(m.name.size()));
        put16(out, static_cast<std::uint16_t>((big ? 20 : 0) + 4 + pad));
        put(out, m.name);
        if (big) {
            put16(out, 0x0001);
            put16(out, 16);
            put64(out, size);
            put64(out, size);
        }
        put16(out, 0x4246);                  // "FB", PyTorch's padding field
        put16(out, static_cast<std::uint16_t>(pad));
        out.insert(out.end(), pad, std::byte{0});
        out.insert(out.end(), m.data.begin(), m.data.end());
        // central directory entry
        const bool big_cd = big || local >= max32;
        put32(cd, detail::central_sig);
        put16(cd, big_cd ? 45 : 20);
        put16(cd, big_cd ? 45 : 20);
        put16(cd, 0);
        put16(cd, 0);
        put16(cd, 0);
        put16(cd, 0x21);
        put32(cd, crc);
        put32(cd, big_cd ? max32 : static_cast<std::uint32_t>(size));
        put32(cd, big_cd ? max32 : static_cast<std::uint32_t>(size));
        put16(cd, static_cast<std::uint16_t>(m.name.size()));
        put16(cd, big_cd ? 28 : 0);
        put16(cd, 0);                        // comment
        put16(cd, 0);                        // disk
        put16(cd, 0);                        // internal attributes
        put32(cd, 0);                        // external attributes
        put32(cd, big_cd ? max32 : static_cast<std::uint32_t>(local));
        put(cd, m.name);
        if (big_cd) {
            put16(cd, 0x0001);
            put16(cd, 24);
            put64(cd, size);
            put64(cd, size);
            put64(cd, local);
        }
    }
    const std::uint64_t cd_offset = out.size(), cd_size = cd.size();
    out.insert(out.end(), cd.begin(), cd.end());
    any64 = any64 || cd_offset >= max32;
    if (any64) {
        const std::uint64_t rec = out.size();
        put32(out, detail::end64_sig);
        put64(out, 44);
        put16(out, 45);
        put16(out, 45);
        put32(out, 0);
        put32(out, 0);
        put64(out, members.size());
        put64(out, members.size());
        put64(out, cd_size);
        put64(out, cd_offset);
        put32(out, detail::locator64_sig);
        put32(out, 0);
        put64(out, rec);
        put32(out, 1);
    }
    put32(out, detail::end_sig);
    put16(out, 0);
    put16(out, 0);
    put16(out, any64 ? 0xffff : static_cast<std::uint16_t>(members.size()));
    put16(out, any64 ? 0xffff : static_cast<std::uint16_t>(members.size()));
    put32(out, any64 ? max32 : static_cast<std::uint32_t>(cd_size));
    put32(out, any64 ? max32 : static_cast<std::uint32_t>(cd_offset));
    put16(out, 0);
    return ok(std::move(out));
}
} // namespace snicholls::formats::zip

#endif // snicholls_formats_archive_zip_hpp
