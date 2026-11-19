//
//  keys_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  GGUF's standard keys, spelt once.

#include <snicholls/formats/gguf/keys.hpp>

#include <cstdio>

using namespace snicholls::formats::gguf;

int main()
{
    //! [keys]
    const std::string heads = arch_key("llama", attention::head_count);   // "llama.attention.head_count"
    //! [keys]
    std::printf("%s\n%s\n%s\n", std::string(general::architecture).c_str(), heads.c_str(), std::string(tokenizer::tokens).c_str());
    return heads == "llama.attention.head_count" ? 0 : 1;
}
