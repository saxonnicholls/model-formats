//
//  ggml_types_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  ggml's tensor types: block sizes and bytes, and which layouts core has.

#include <snicholls/formats/gguf/ggml_types.hpp>

#include <cstdio>

using namespace snicholls::formats::gguf;

int main()
{
    for (const auto& t : ggml_types)
        std::printf("%-8s code %2u: %3u elements in %3u bytes%s\n", t.name.data(), static_cast<unsigned>(t.type), t.block_size, t.type_size,
                    t.in_core ? "  (core)" : "");
    //! [ggml_types]
    const auto q8 = traits(ggml_type::q8_0);                 // 32 elements in 34 bytes
    const auto row = row_bytes(ggml_type::q4_k, 4096);       // a 4096-wide Q4_K row: 16 blocks of 144 bytes
    const auto partial = row_bytes(ggml_type::q8_0, 100);    // not whole blocks: none
    //! [ggml_types]
    std::printf("q8_0 %u/%u, q4_k row of 4096: %llu bytes, q8_0 row of 100: %s\n", q8.block_size, q8.type_size,
                static_cast<unsigned long long>(*row), partial ? "?" : "refused");
    return q8.type_size == 34 && row == 2304u && !partial && to_dtype(ggml_type::bf16) == snicholls::formats::dtype::bf16 ? 0 : 1;
}
