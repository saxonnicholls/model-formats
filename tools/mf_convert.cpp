//
//  mf_convert.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  mf-convert: one model format to another.
//
//      mf-convert <in> <out> [--dtype f32|f16|bf16] [--name NAME] [--tokenizer tokenizer.json]
//
//    <model directory>  -> out.gguf             a Hugging Face model as llama.cpp's GGUF (names, Q/K permutation,
//                                               transposes, stacked experts, config and tokenizer as metadata)
//    llm.c checkpoint   -> out.gguf             GPT-2 for llama.cpp (--tokenizer for the vocabulary it needs)
//    llm.c checkpoint   -> <directory>/         Hugging Face GPT-2: model.safetensors and config.json
//    any file           -> out.safetensors | out.gguf | out.npz | out.npy      tensor by tensor, names kept
//
//  --dtype casts every floating-point tensor first. Exit status 0, or 1 with the error.

#include <snicholls/formats/convert/export.hpp>
#include <snicholls/formats/hf/tokenizer.hpp>
#include <snicholls/formats/io/model_dir.hpp>

#include <cstdio>
#include <cstring>
#include <deque>
#include <string>

using namespace snicholls::formats;

namespace
{
int fail_with(const error& e)
{
    std::fprintf(stderr, "mf-convert: %s\n", e.what().c_str());
    return 1;
}

std::optional<dtype> parse_dtype(std::string_view s)
{
    if (s == "f32")
        return dtype::f32;
    if (s == "f16")
        return dtype::f16;
    if (s == "bf16")
        return dtype::bf16;
    return std::nullopt;
}

// Casts the float tensors; the copies live in `owned`
result<std::vector<tensor_view>> cast_all(std::span<const tensor_view> in, std::optional<dtype> to, std::deque<tensor>& owned)
{
    std::vector<tensor_view> out;
    for (const auto& t : in) {
        if (!to || !is_float(t.type) || t.type == *to) {
            out.push_back(t);
            continue;
        }
        SNICHOLLS_TRY_ASSIGN(tensor c, convert::cast(t, *to));
        owned.push_back(std::move(c));
        out.push_back(owned.back().view());
    }
    return snicholls::ok(std::move(out));
}

result<void> write_gguf(const std::filesystem::path& out, const gguf::writer& w)
{
    return io::write_streamed(out, [&](auto&& sink) { return w.write_to(sink); });
}
} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> pos;
    std::optional<dtype> to;
    std::string name = "model";
    std::filesystem::path tokenizer_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dtype" && i + 1 < argc) {
            to = parse_dtype(argv[++i]);
            if (!to) {
                std::fprintf(stderr, "mf-convert: --dtype takes f32, f16 or bf16\n");
                return 2;
            }
        } else if (a == "--name" && i + 1 < argc) {
            name = argv[++i];
        } else if (a == "--tokenizer" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 2) {
        std::fprintf(stderr, "usage: mf-convert <in> <out> [--dtype f32|f16|bf16] [--name NAME] [--tokenizer tokenizer.json]\n");
        return 2;
    }
    const std::filesystem::path in = pos[0], out = pos[1];
    const bool out_gguf = out.extension() == ".gguf";
    std::deque<tensor> owned;

    if (std::filesystem::is_directory(in)) {   // a Hugging Face model
        const auto m = io::open_model_dir(in);
        if (m.is_err())
            return fail_with(m.unwrap_err_unchecked());
        if (!out_gguf) {
            auto views = cast_all(m->tensors, to, owned);
            if (views.is_err())
                return fail_with(views.unwrap_err_unchecked());
            const auto w = io::save_tensors(out, *views);
            return w.is_ok() ? 0 : fail_with(w.unwrap_err_unchecked());
        }
        if (!m->config)
            return fail_with(m->config_error.value_or(error{errc::not_found, "no configuration"}));
        auto views = cast_all(m->tensors, to, owned);
        if (views.is_err())
            return fail_with(views.unwrap_err_unchecked());
        const vocabulary* v = m->tokenizer ? &m->tokenizer->vocab : nullptr;
        const std::string* tpl = m->tokenizer ? m->tokenizer->default_chat_template() : nullptr;
        const auto g = convert::hf_to_gguf(*m->config, *views, v, tpl ? std::string_view(*tpl) : std::string_view{}, name);
        if (g.is_err())
            return fail_with(g.unwrap_err_unchecked());
        for (const auto& n : g->notes)
            std::printf("note: %s\n", n.c_str());
        const auto w = write_gguf(out, g->writer);
        if (w.is_err())
            return fail_with(w.unwrap_err_unchecked());
        std::printf("%s -> %s: %zu tensors\n", in.string().c_str(), out.string().c_str(), g->writer.tensor_count());
        return 0;
    }

    const auto f = io::mapped_file::open(in);
    if (f.is_err())
        return fail_with(f.unwrap_err_unchecked());
    const auto format = convert::detect(f->bytes());
    if (format == convert::file_format::llmc_checkpoint && (out_gguf || out.extension().empty())) {
        const auto c = llmc::parse_checkpoint(f->bytes());
        if (c.is_err())
            return fail_with(c.unwrap_err_unchecked());
        if (out_gguf) {
            std::optional<hf::tokenizer> tok;
            if (!tokenizer_path.empty()) {
                const auto text = snicholls::read_file(tokenizer_path);
                if (text.is_err())
                    return fail_with(io::io_error(tokenizer_path, "cannot read", text.unwrap_err_unchecked()));
                auto t = hf::parse_tokenizer(*text);
                if (t.is_err())
                    return fail_with(t.unwrap_err_unchecked());
                tok = std::move(*t);
                if (const auto eot = tok->id_of("<|endoftext|>"))
                    tok->vocab.bos = tok->vocab.eos = *eot;
            }
            const auto g = convert::llmc_to_gguf(*c, tok ? &tok->vocab : nullptr, name);
            if (g.is_err())
                return fail_with(g.unwrap_err_unchecked());
            const auto w = write_gguf(out, g->writer);
            if (w.is_err())
                return fail_with(w.unwrap_err_unchecked());
            std::printf("%s -> %s: llama.cpp gpt2, %zu tensors%s\n", in.string().c_str(), out.string().c_str(), g->writer.tensor_count(),
                        tok ? "" : " (no tokenizer: pass --tokenizer for llama.cpp)");
            return 0;
        }
        const auto hf = convert::llmc_to_hf(*c);   // a directory: model.safetensors + config.json
        if (hf.is_err())
            return fail_with(hf.unwrap_err_unchecked());
        auto views = cast_all(hf->tensors, to, owned);
        if (views.is_err())
            return fail_with(views.unwrap_err_unchecked());
        const std::vector<std::pair<std::string, std::string>> meta{{"format", "pt"}};
        if (auto w = io::save_tensors(out / "model.safetensors", *views, meta); w.is_err())
            return fail_with(w.unwrap_err_unchecked());
        const std::string config = convert::hf_gpt2_config(c->shape);
        if (auto w = io::write_file(out / "config.json", std::as_bytes(std::span(config.data(), config.size()))); w.is_err())
            return fail_with(w.unwrap_err_unchecked());
        std::printf("%s -> %s: Hugging Face GPT-2, %zu tensors\n", in.string().c_str(), out.string().c_str(), views->size());
        return 0;
    }
    const auto l = convert::read_any(f->bytes());
    if (l.is_err())
        return fail_with(l.unwrap_err_unchecked());
    for (const auto& n : l->notes)
        std::printf("note: %s\n", n.c_str());
    auto views = cast_all(l->tensors, to, owned);
    if (views.is_err())
        return fail_with(views.unwrap_err_unchecked());
    const auto w = io::save_tensors(out, *views, l->format == convert::file_format::safetensors ? l->metadata : decltype(l->metadata){});
    if (w.is_err())
        return fail_with(w.unwrap_err_unchecked());
    std::printf("%s (%s) -> %s: %zu tensors\n", in.string().c_str(), convert::to_string(l->format).data(), out.string().c_str(), views->size());
    return 0;
}
