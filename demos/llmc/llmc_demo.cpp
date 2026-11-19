//
//  llmc_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  llm.c's GPT-2 checkpoint and token shard, written and read back.

#include <snicholls/formats/llmc/llmc.hpp>

#include <cstdio>
#include <vector>

using namespace snicholls::formats;
namespace lc = snicholls::formats::llmc;

int main()
{
    //! [llmc]
    const lc::gpt2_shape s{.max_seq = 16, .vocab = 50, .layers = 1, .heads = 2, .channels = 8, .padded_vocab = 64};
    const std::vector<float> params(lc::parameter_count(s), 0.01f);
    const auto file = lc::write_checkpoint(s, dtype::f32, std::as_bytes(std::span(params)));
    const auto ckpt = lc::parse_checkpoint(*file);   // sixteen views, llm.c's order
    //! [llmc]
    for (const auto& t : ckpt->tensors)
        std::printf("%-9s %s\n", std::string(t.name).c_str(), to_string(t.shape).c_str());
    const std::vector<std::uint16_t> tokens{50256, 15496, 11, 995};
    const auto shard = lc::write_tokens(tokens);
    const auto back = lc::parse_tokens(*shard);
    std::printf("%llu parameters; shard of %zu tokens\n", static_cast<unsigned long long>(lc::parameter_count(s)), back->size() / 2);
    return ckpt.is_ok() && ckpt->tensors[4].shape == tensor_shape{1, 24, 8} && back.is_ok() && back->size() == 8 ? 0 : 1;
}
