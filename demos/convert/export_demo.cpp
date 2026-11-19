//
//  export_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Our llm.c checkpoint exported two ways: Hugging Face's GPT-2 safetensors (and config.json) and llama.cpp's GGUF.

#include <snicholls/formats/convert/export.hpp>

#include <cstdio>
#include <vector>

using namespace snicholls::formats;
using namespace snicholls::formats::convert;

int main()
{
    const llmc::gpt2_shape s{.max_seq = 16, .vocab = 3, .layers = 1, .heads = 2, .channels = 8, .padded_vocab = 8};
    const std::vector<float> params(llmc::parameter_count(s), 0.02f);
    const auto file = llmc::write_checkpoint(s, dtype::f32, std::as_bytes(std::span(params)));
    vocabulary v;
    v.tokens = {"a", "b", "ab"};
    v.merges = {"a b"};
    //! [export]
    const auto ckpt = llmc::parse_checkpoint(*file);
    const auto hf = llmc_to_hf(*ckpt);                                   // wte.weight, h.0.attn.c_attn.weight, ...
    const auto st = safetensors::write(hf->tensors, {{{"format", "pt"}}});
    const std::string config = hf_gpt2_config(s);                        // config.json beside it
    const auto gg = llmc_to_gguf(*ckpt, &v, "tiny-gpt2");                // llama.cpp's gpt2, tensors viewed in place
    const auto gguf_bytes = gg->writer.write();
    //! [export]
    std::printf("safetensors: %zu tensors, %zu bytes; GGUF: %zu tensors, %zu bytes; config.json %zu bytes\n", hf->tensors.size(),
                st->size(), gg->writer.tensor_count(), gguf_bytes->size(), config.size());
    for (const auto& n : gg->notes)
        std::printf("note: %s\n", n.c_str());
    const auto parsed = gguf::parse(*gguf_bytes);
    return hf.is_ok() && st.is_ok() && gguf_bytes.is_ok() && parsed.is_ok() && parsed->architecture() == "gpt2" && gg->writer.tensor_count() == 16
               ? 0
               : 1;
}
