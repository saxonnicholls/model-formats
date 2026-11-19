//
//  names_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  One weight under three names, and llama.cpp's Q/K permutation.

#include <snicholls/formats/hf/names.hpp>

#include <array>
#include <cstdio>

using namespace snicholls::formats;
using namespace snicholls::formats::hf;

int main()
{
    //! [names]
    const auto q = parse_hf_name(architecture::llama, "model.layers.3.self_attn.q_proj.weight");
    const std::string in_gguf = gguf_name(*q);                                       // "blk.3.attn_q.weight"
    const bool permute = gguf_permutes(architecture::llama, *q);                     // true: RoPE pairs
    const auto fused = parse_hf_name(architecture::gpt2, "h.0.attn.c_attn.weight");  // Conv1D, [in, out]
    //! [names]
    std::printf("%s -> %s (permuted: %s); GPT-2 %s transposed: %s\n", hf_name(architecture::llama, *q).c_str(), in_gguf.c_str(),
                permute ? "yes" : "no", hf_name(architecture::gpt2, *fused).c_str(), hf_transposed(architecture::gpt2, *fused) ? "yes" : "no");
    const std::array<float, 4> head{0, 1, 2, 3};   // one head of 4 rows: halves (0 1)(2 3)
    std::array<float, 4> pairs{};
    permute_rope(std::as_bytes(std::span(head)), std::as_writable_bytes(std::span(pairs)), 1, sizeof(float));
    std::printf("rows 0 1 2 3 permuted: %g %g %g %g\n", pairs[0], pairs[1], pairs[2], pairs[3]);
    std::array<float, 4> back{};
    unpermute_rope(std::as_bytes(std::span(pairs)), std::as_writable_bytes(std::span(back)), 1, sizeof(float));
    const auto e = parse_gguf_name("blk.1.ffn_down_exps.weight");
    return in_gguf == "blk.3.attn_q.weight" && permute && hf_transposed(architecture::gpt2, *fused) && pairs == std::array<float, 4>{0, 2, 1, 3}
                   && back == head && e && e->what == role::expert_down
               ? 0
               : 1;
}
