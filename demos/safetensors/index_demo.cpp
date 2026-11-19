//
//  index_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A sharded checkpoint's index: where each tensor lives, and the shard names.

#include <snicholls/formats/safetensors/index.hpp>

#include <cstdio>

namespace st = snicholls::formats::safetensors;

int main()
{
    //! [index]
    const auto ix = st::parse_index(R"({"metadata":{"total_size":96},
        "weight_map":{"lm_head.weight":"model-00002-of-00002.safetensors",
                      "model.embed_tokens.weight":"model-00001-of-00002.safetensors"}})");
    for (const auto& f : ix->files())
        std::printf("shard %s\n", f.c_str());
    std::printf("lm_head.weight is in %s\n", ix->shard_of("lm_head.weight")->c_str());
    //! [index]
    const auto names = st::shard_names("model", 2);
    const auto text = st::write_index(*ix);
    const auto unsafe = st::parse_index(R"({"weight_map":{"t":"../../etc/passwd"}})");
    std::printf("unsafe shard name: %s\n", unsafe.unwrap_err_unchecked().what().c_str());
    return ix.is_ok() && ix->total_size == 96u && ix->files() == names && text.is_ok() && unsafe.is_err()
                   && st::parse_index(*text)->weight_map == ix->weight_map
               ? 0
               : 1;
}
