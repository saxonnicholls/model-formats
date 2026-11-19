//
//  mf_inspect.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  mf-inspect: what a model file (or a Hugging Face model directory) holds.
//
//      mf-inspect model.safetensors | model.gguf | pytorch_model.bin | weights.npz | gpt2_124M.bin | <model directory>
//      mf-inspect --no-tensors model.gguf          metadata only
//
//  Prints the format, the metadata and, for each tensor, its name, dtype (or ggml type) and shape. Exit status 0, or
//  1 with the error when the file is not one this library reads.

#include <snicholls/formats/gguf/gguf.hpp>
#include <snicholls/formats/io/model_dir.hpp>
#include <snicholls/formats/llmc/llmc.hpp>
#include <snicholls/formats/pytorch/checkpoint.hpp>
#include <snicholls/formats/safetensors/safetensors.hpp>

#include <cstdio>
#include <cstring>
#include <string>

using namespace snicholls::formats;

namespace
{
bool show_tensors = true;

void tensor_line(std::string_view name, std::string_view type, const tensor_shape& shape, std::size_t bytes)
{
    if (show_tensors)
        std::printf("  %-48s %-8s %-20s %zu bytes\n", std::string(name).c_str(), std::string(type).c_str(), to_string(shape).c_str(), bytes);
}

std::string summary(const gguf::value& v)
{
    switch (v.type) {
    case gguf::value_type::string: {
        std::string s(*v.as_string());
        return "\"" + (s.size() > 80 ? s.substr(0, 80) + "..." : s) + "\"";
    }
    case gguf::value_type::array: return "[" + std::to_string(v.count) + " x " + std::string(gguf::to_string(v.element_type)) + "]";
    case gguf::value_type::boolean: return *v.as_bool() ? "true" : "false";
    case gguf::value_type::f32: case gguf::value_type::f64: return std::to_string(*v.as_f64());
    case gguf::value_type::i8: case gguf::value_type::i16: case gguf::value_type::i32: case gguf::value_type::i64: return std::to_string(*v.as_i64());
    default: return std::to_string(*v.as_u64());
    }
}

int fail_with(const error& e)
{
    std::fprintf(stderr, "mf-inspect: %s\n", e.what().c_str());
    return 1;
}

int directory(const std::filesystem::path& p)
{
    const auto m = io::open_model_dir(p);
    if (m.is_err())
        return fail_with(m.unwrap_err_unchecked());
    std::printf("%s: Hugging Face model directory, %zu weight file(s), %zu tensors\n", p.string().c_str(), m->files.size(), m->tensors.size());
    if (m->config) {
        const auto& c = *m->config;
        std::printf("  architecture %s: %zu layers, d_model %zu, %zu heads (%zu KV) of %zu, d_ff %zu, vocab %zu, context %zu\n",
                    hf_model_type(c.arch).data(), c.layers, c.block.d_model, c.block.n_heads, c.block.n_kv_heads, c.block.head_dim, c.block.d_ff,
                    c.vocab, c.context);
        if (c.block.moe())
            std::printf("  experts %zu (top %zu, shared %zu) of d_ff %zu\n", c.block.experts, c.block.top_k, c.block.shared_experts, c.block.expert_d_ff);
        for (const auto& n : c.notes)
            std::printf("  not mapped onto core: %s\n", n.c_str());
    } else if (m->config_error) {
        std::printf("  config.json: %s\n", m->config_error->what().c_str());
    }
    if (m->tokenizer) {
        const auto& v = m->tokenizer->vocab;
        std::printf("  tokenizer: %s, %zu tokens, %zu merges%s\n", gguf_tokenizer_model(v.kind).data(), v.tokens.size(), v.merges.size(),
                    m->tokenizer->default_chat_template() ? ", chat template" : "");
    }
    for (const auto& t : m->tensors)
        tensor_line(t.name, safetensors_name(t.type), t.shape, t.data.size());
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    int at = 1;
    if (at < argc && std::strcmp(argv[at], "--no-tensors") == 0) {
        show_tensors = false;
        ++at;
    }
    if (at + 1 != argc) {
        std::fprintf(stderr, "usage: mf-inspect [--no-tensors] <file or model directory>\n");
        return 2;
    }
    const std::filesystem::path path = argv[at];
    if (std::filesystem::is_directory(path))
        return directory(path);
    const auto f = io::mapped_file::open(path);
    if (f.is_err())
        return fail_with(f.unwrap_err_unchecked());
    const auto bytes = f->bytes();
    const auto format = convert::detect(bytes);
    if (!format)
        return fail_with(error{errc::unsupported, path.string() + ": not a format this library reads"});
    std::printf("%s: %s, %zu bytes\n", path.string().c_str(), convert::to_string(*format).data(), bytes.size());
    switch (*format) {
    case convert::file_format::gguf: {
        const auto g = gguf::parse(bytes);
        if (g.is_err())
            return fail_with(g.unwrap_err_unchecked());
        std::printf("  GGUF v%u, alignment %u, %zu keys, %zu tensors\n", g->version, g->alignment, g->metadata.size(), g->tensors.size());
        for (const auto& e : g->metadata)
            std::printf("  %-48s %-8s %s\n", std::string(e.key).c_str(), gguf::to_string(e.val.type).data(), summary(e.val).c_str());
        for (const auto& t : g->tensors)
            tensor_line(t.name, gguf::traits(t.type).name, t.shape, t.data.size());
        return 0;
    }
    case convert::file_format::safetensors: {
        const auto s = safetensors::parse(bytes);
        if (s.is_err())
            return fail_with(s.unwrap_err_unchecked());
        std::printf("  header %llu bytes, %zu tensors\n", static_cast<unsigned long long>(s->header_size), s->tensors.size());
        for (const auto& [k, v] : s->metadata)
            std::printf("  %s = %s\n", k.c_str(), v.c_str());
        for (const auto& t : s->tensors)
            tensor_line(t.name, safetensors_name(t.type), t.shape, t.data.size());
        return 0;
    }
    case convert::file_format::pytorch: {
        const auto c = pytorch::parse(bytes);
        if (c.is_err())
            return fail_with(c.unwrap_err_unchecked());
        std::printf("  torch.save archive \"%s\", %zu tensors (read with the restricted unpickler)\n", c->prefix.c_str(), c->tensors.size());
        for (const auto& s : c->skipped)
            std::printf("  %s: not a tensor\n", s.c_str());
        for (const auto& t : c->tensors)
            tensor_line(t.name, std::string(safetensors_name(t.type)) + (t.contiguous ? "" : "*"), t.shape,
                        static_cast<std::size_t>(*t.shape.elements()) * size_of(t.type));
        return 0;
    }
    case convert::file_format::llmc_checkpoint: {
        const auto c = llmc::parse_checkpoint(bytes);
        if (c.is_err())
            return fail_with(c.unwrap_err_unchecked());
        const auto& s = c->shape;
        std::printf("  llm.c GPT-2 (%s): %zu layers, %zu heads, %zu channels, vocab %zu (padded %zu), context %zu, %llu parameters\n",
                    safetensors_name(c->type).data(), s.layers, s.heads, s.channels, s.vocab, s.padded_vocab, s.max_seq,
                    static_cast<unsigned long long>(llmc::parameter_count(s)));
        for (const auto& t : c->tensors)
            tensor_line(t.name, safetensors_name(t.type), t.shape, t.data.size());
        return 0;
    }
    default: {
        const auto l = convert::read_any(bytes);
        if (l.is_err())
            return fail_with(l.unwrap_err_unchecked());
        for (const auto& t : l->tensors)
            tensor_line(t.name, safetensors_name(t.type), t.shape, t.data.size());
        return 0;
    }
    }
}
