//
//  cache_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The cache huggingface_hub and this library share.

#include <snicholls/formats/hub/cache.hpp>

#include <cstdio>

using namespace snicholls::formats::hub;

int main()
{
    //! [cache]
    const auto c = cache::open(default_cache_dir(), "hf-internal-testing/tiny-random-gpt2");
    const auto blob = c->blob("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    const auto snap = c->snapshot("0123456789abcdef0123456789abcdef01234567");
    //! [cache]
    std::printf("repository %s\nblob       %s\nsnapshot   %s\n", c->dir().string().c_str(), blob.string().c_str(), snap.string().c_str());
    return c.is_ok() && c->dir().filename() == "models--hf-internal-testing--tiny-random-gpt2" && !cache::open("/", "../x").is_ok() ? 0 : 1;
}
