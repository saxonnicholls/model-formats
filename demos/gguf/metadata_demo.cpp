//
//  metadata_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A model_config and vocabulary written as GGUF metadata, and read back.

#include <snicholls/formats/gguf/metadata.hpp>

#include <cstdio>

using namespace snicholls::formats;
using namespace snicholls::formats::gguf;

int main()
{
    //! [metadata]
    model_config c;
    c.arch = architecture::gpt2;
    c.block = snicholls::ml::gpt2_small;
    c.layers = 12;
    c.vocab = 50257;
    c.context = 1024;
    writer w;
    const auto written = write_model_config(w, c, "gpt2-small");   // gpt2.embedding_length, gpt2.block_count, ...
    vocabulary v;
    v.tokens = {"a", "b", "ab"};
    v.merges = {"a b"};
    write_vocabulary(w, v);
    const auto bytes = w.write();
    const auto f = parse(*bytes);
    const auto back = read_model_config(*f);
    //! [metadata]
    for (const auto& e : f->metadata)
        std::printf("%s\n", std::string(e.key).c_str());
    const auto vb = read_vocabulary(*f);
    return written.is_ok() && back.is_ok() && same_block(back->block, snicholls::ml::gpt2_small) && back->layers == 12 && vb.is_ok()
                   && vb->merges.size() == 1
               ? 0
               : 1;
}
