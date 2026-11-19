//
//  zip_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/archive/zip.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace snicholls::formats;

namespace
{
std::span<const std::byte> bytes(const std::string& s) { return std::as_bytes(std::span(s.data(), s.size())); }
} // namespace

TEST(Zip, RoundTripsStoredMembersAligned)
{
    const std::string a = "hello", b(1000, 'x'), empty;
    const std::vector<zip::member> m{{"a.txt", bytes(a)}, {"dir/b.bin", bytes(b)}, {"empty", bytes(empty)}};
    const auto w = zip::write(m);
    ASSERT_TRUE(w.is_ok());
    const auto r = zip::parse(*w);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    ASSERT_EQ(r->entries.size(), 3u);
    EXPECT_FALSE(r->zip64);
    EXPECT_EQ(r->entries[1].name, "dir/b.bin");
    for (const auto& e : r->entries) {
        EXPECT_EQ(e.offset % 64, 0u);
        EXPECT_TRUE(zip::verify(e).is_ok());
    }
    const auto* f = r->find("a.txt");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(f->data.data()), f->data.size()), "hello");
    EXPECT_EQ(r->find("nope"), nullptr);
    EXPECT_EQ(*zip::write(m), *w);   // deterministic
}

TEST(Zip, Zip64RecordsAreReadAndWritten)
{
    const std::string a = "zip64";
    const std::vector<zip::member> m{{"a", bytes(a)}};
    const auto w = zip::write(m, zip::write_options{1, true});
    const auto r = zip::parse(*w);
    ASSERT_TRUE(r.is_ok()) << r.unwrap_err_unchecked().what();
    EXPECT_TRUE(r->zip64);
    EXPECT_EQ(r->entries[0].data.size(), 5u);
}

TEST(Zip, CrcMismatchIsDetected)
{
    const std::string a = "payload";
    const std::vector<zip::member> m{{"a", bytes(a)}};
    auto w = *zip::write(m);
    const auto r = zip::parse(w);
    const auto at = r->entries[0].offset;
    w[at] = std::byte{'P'};
    const auto again = zip::parse(w);
    EXPECT_EQ(zip::verify(again->entries[0]).unwrap_err_unchecked().code, errc::checksum_mismatch);
    static_assert(snicholls::formats::detail::crc32({}) == 0);
}

TEST(Zip, CrcIsTheStandardOne)
{
    const std::string check = "123456789";
    EXPECT_EQ(snicholls::formats::detail::crc32(bytes(check)), 0xCBF43926u);   // the CRC-32 check value
}

TEST(Zip, RefusesMalformedArchives)
{
    const auto code = [](const std::vector<std::byte>& b) { return zip::parse(b).unwrap_err_unchecked().code; };
    EXPECT_EQ(code({}), errc::truncated);
    EXPECT_EQ(code(std::vector<std::byte>(100)), errc::bad_magic);
    const std::string a = "abc";
    const std::vector<zip::member> m{{"a", bytes(a)}};
    auto w = *zip::write(m, zip::write_options{1, false});
    auto compressed = w;
    const auto cd = zip::parse(w)->entries[0].offset + 3;   // the central directory follows the data
    compressed[cd + 10] = std::byte{8};                    // method: deflate
    EXPECT_EQ(code(compressed), errc::unsupported);
    auto encrypted = w;
    encrypted[cd + 8] = std::byte{1};
    EXPECT_EQ(code(encrypted), errc::unsupported);
    auto cut = std::vector<std::byte>(w.begin() + 10, w.end());   // offsets now point wrong
    EXPECT_NE(zip::parse(cut).is_ok(), true);
    EXPECT_EQ(zip::write(std::vector<zip::member>{{"x", bytes(a)}, {"x", bytes(a)}}).unwrap_err_unchecked().code, errc::duplicate_name);
    EXPECT_EQ(zip::write(std::vector<zip::member>{{"", bytes(a)}}).unwrap_err_unchecked().code, errc::bad_argument);
}

TEST(Zip, DuplicateAndOverlappingMembersAreRefused)
{
    const std::string a = "abc";
    const std::vector<zip::member> m{{"a", bytes(a)}, {"b", bytes(a)}};
    auto w = *zip::write(m, zip::write_options{1, false});
    const auto r = zip::parse(w);
    // rename b to a in both headers: same length, so the archive stays well-formed
    for (std::size_t i = 0; i + 1 < w.size(); ++i)
        if (w[i] == std::byte{'b'} && (i + 1 == w.size() || w[i + 1] != std::byte{'c'}))
            w[i] = std::byte{'a'};
    EXPECT_EQ(zip::parse(w).unwrap_err_unchecked().code, errc::duplicate_name);
    EXPECT_TRUE(r.is_ok());
}
