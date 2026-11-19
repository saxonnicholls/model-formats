//
//  gguf_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Build a small GGUF file (metadata of several types, a dense and a quantised tensor), then read it back.

#include <snicholls/formats/gguf/gguf.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace snicholls::formats;
using namespace snicholls::formats::gguf;

int main()
{
    //! [gguf]
    const std::vector<float> norm(64, 1.0f);
    std::vector<snicholls::ml::block_q8_0> q(2);   // a 64-wide Q8_0 row, quantised by core
    const std::vector<float> row(64, 0.5f);
    snicholls::ml::quantize(row, std::span(q));
    writer w;
    w.set("general.architecture", value_string("llama"));
    w.set("llama.block_count", value_u32(1));
    const std::vector<std::string> tokens{"<s>", "a", "b"};
    w.set("tokenizer.ggml.tokens", value_strings(tokens));
    w.add_tensor("output_norm.weight", ggml_type::f32, {64}, std::as_bytes(std::span(norm)));
    w.add_tensor("output.weight", ggml_type::q8_0, {1, 64}, std::as_bytes(std::span(q)));
    const auto bytes = w.write();
    const auto f = parse(*bytes);   // views into `bytes`
    for (const auto& t : f->tensors)
        std::printf("%-20s %-5s %s, %zu bytes at %llu\n", std::string(t.name).c_str(), traits(t.type).name.data(), to_string(t.shape).c_str(),
                    t.data.size(), static_cast<unsigned long long>(t.offset));
    //! [gguf]
    int failures = !(bytes.is_ok() && f.is_ok() && f->architecture() == "llama" && f->tensors.size() == 2);
    for (const auto& e : f->metadata)
        std::printf("%s: %s\n", std::string(e.key).c_str(), to_string(e.val.type).data());
    failures += f->find("llama.block_count")->as_u64() != 1u;
    failures += f->find("tokenizer.ggml.tokens")->strings()->at(2) != "b";
    failures += f->find_tensor("output.weight")->data.size() != 68;
    std::vector<std::byte> cut(bytes->begin(), bytes->begin() + 40);
    const auto bad = parse(cut);
    std::printf("first 40 bytes: %s\n", bad.is_err() ? bad.unwrap_err_unchecked().what().c_str() : "accepted");
    return failures + !bad.is_err();
}
