//
//  digest_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The digests the Hub names files by: SHA-256 for LFS files, git's blob id for the rest.

#include <snicholls/formats/hub/digest.hpp>

#include <cstdio>
#include <string_view>

using namespace snicholls::formats::hub;

int main()
{
    const std::string_view config = "{\"model_type\": \"gpt2\"}\n";
    //! [digest]
    const std::string lfs = sha256_hex(std::as_bytes(std::span(config.data(), config.size())));          // an LFS file's etag
    const std::string blob = git_blob_sha1_hex(std::as_bytes(std::span(config.data(), config.size())));  // a small file's
    //! [digest]
    std::printf("sha256 %s\ngit blob %s\n", lfs.c_str(), blob.c_str());
    return lfs.size() == 64 && blob.size() == 40 && sha256_hex({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ? 0 : 1;
}
