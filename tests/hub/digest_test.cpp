//
//  digest_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/hub/digest.hpp>

#include <gtest/gtest.h>

#include "support/temp_dir.hpp"

using namespace snicholls::formats;
using namespace snicholls::formats::hub;

namespace
{
std::span<const std::byte> bytes(std::string_view s) { return std::as_bytes(std::span(s.data(), s.size())); }
} // namespace

TEST(HubDigest, Sha256MatchesFips180)
{
    EXPECT_EQ(sha256_hex(bytes("")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex(bytes("abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    digest d(algorithm::sha256);
    d.update("ab");
    d.update(bytes("c"));
    EXPECT_EQ(d.hex(), sha256_hex(bytes("abc")));
    EXPECT_EQ(d.hex(), sha256_hex(bytes("abc")));   // finished: hex() again is the same
}

TEST(HubDigest, GitBlobIdsAreGits)
{
    EXPECT_EQ(git_blob_sha1_hex(bytes("")), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");             // git hash-object /dev/null
    EXPECT_EQ(git_blob_sha1_hex(bytes("hello\n")), "ce013625030ba8dba906f756967f9e9ca394464a");    // echo hello | git hash-object --stdin
}

TEST(HubDigest, FilesHashThroughTheirMapping)
{
    const test_support::temp_dir dir("digest");
    const std::string text = "abc";
    ASSERT_TRUE(io::write_file(dir.path / "f", bytes(text)).is_ok());
    EXPECT_EQ(*file_digest_hex(dir.path / "f"), sha256_hex(bytes("abc")));
    EXPECT_EQ(*file_digest_hex(dir.path / "f", true), git_blob_sha1_hex(bytes("abc")));
    EXPECT_EQ(file_digest_hex(dir.path / "missing").unwrap_err_unchecked().code, errc::io);
}
