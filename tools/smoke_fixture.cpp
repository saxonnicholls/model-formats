//
//  smoke_fixture.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The tools' smoke run starts from these files: a tiny llm.c GPT-2 checkpoint and a byte-level tokenizer.json of
//  the same vocabulary. Usage: smoke_fixture <directory>.

#include <snicholls/formats/detail/byte_level.hpp>
#include <snicholls/formats/io/files.hpp>
#include <snicholls/formats/llmc/llmc.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace snicholls::formats;

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;
    const std::filesystem::path dir = argv[1];
    // 256 byte tokens and <|endoftext|>: GPT-2's alphabet, a vocabulary of 257
    const llmc::gpt2_shape s{.max_seq = 16, .vocab = 257, .layers = 2, .heads = 2, .channels = 8, .padded_vocab = 320};
    std::vector<float> p(llmc::parameter_count(s));
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = static_cast<float>((i * 7919) % 1000) / 1000.0f - 0.5f;
    const auto ck = llmc::write_checkpoint(s, dtype::f32, std::as_bytes(std::span(p)));
    if (ck.is_err() || io::write_file(dir / "tiny_gpt2.bin", *ck).is_err())
        return 1;
    std::string vocab;
    for (unsigned b = 0; b < 256; ++b) {
        std::string t = snicholls::formats::detail::byte_level_encode(std::string(1, static_cast<char>(b)));
        std::string escaped;
        for (const char c : t)
            escaped += c == '"' ? std::string("\\\"") : c == '\\' ? std::string("\\\\") : std::string(1, c);
        vocab += (b ? "," : "") + std::string("\"") + escaped + "\":" + std::to_string(b);
    }
    const std::string tok = "{\"added_tokens\":[{\"id\":256,\"content\":\"<|endoftext|>\",\"special\":true}],\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
                            "\"model\":{\"type\":\"BPE\",\"vocab\":{" + vocab + "},\"merges\":[]}}";
    if (io::write_file(dir / "tokenizer.json", std::as_bytes(std::span(tok.data(), tok.size()))).is_err())
        return 1;
    std::printf("%s\n", dir.string().c_str());
    return 0;
}
