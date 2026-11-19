//
//  export.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Model-level conversion: our llm.c checkpoints to Hugging Face safetensors and to GGUF, and Hugging Face
/// checkpoints to GGUF and back, with the names, layouts and metadata each side expects.
/// \ingroup convert
///
/// \overview
/// - `llmc_to_hf` turns an llm.c GPT-2 checkpoint into Hugging Face's GPT-2 tensors (`wte.weight`, `h.N.attn.c_attn.*`,
///   ...): Q/K/V/O and the MLP transposed to `Conv1D`'s `[in, out]`, the token table cut to the real vocabulary, the
///   head left tied; `hf_gpt2_config` writes the matching `config.json`.
/// - `llmc_to_gguf` writes the same model as llama.cpp's `gpt2` architecture, every tensor a view into the checkpoint
///   (llm.c's `[out, in]` layout is ggml's), with the metadata llama.cpp needs to load it, including the tokenizer.
/// - `hf_to_gguf` converts a Hugging Face checkpoint of GPT-2, Llama, Mistral, Qwen 2, Qwen 3 or OLMoE: names through
///   `hf::gguf_name`, GPT-2's transposes, llama.cpp's Q/K permutation for Llama and Mistral, routed experts stacked
///   into one tensor per layer, the tied head dropped; `gguf_to_hf` is the inverse (names, unpermuted, unstacked).
///
/// Tensors that need no change are views; the rest are copies held in the result's `owned` (a deque, so the writer's
/// views of them stay valid).
///
/// \relations
/// \see formats::hf::names, formats::gguf::write_model_config, formats::gguf::write_vocabulary, formats::llmc
///
/// \impls
/// One function per direction.
///
/// \contracts
/// `gguf_to_hf(parse(hf_to_gguf(t)))` gives back `t`'s tensors bit for bit (a test proves it for every architecture
/// supported). The GGUF written by `llmc_to_gguf` is read by gguf-py (scripts/crosscheck.sh).
///
/// \errors
/// `unsupported` (DeepSeek's latent attention, which llama.cpp splits differently; a tensor no rule names),
/// `bad_shape` (a tensor whose shape does not match the configuration), `not_found` (a tensor the architecture needs).
///
/// \examples
/// \snippet export_demo.cpp export
///
/// \perf
/// Not benchmarked: one copy of each transposed, permuted or stacked tensor; views for the rest.
///
/// \testing
/// tests/convert/export_test.cpp, demos/convert/export_demo.cpp.
///
/// \deps
/// The format headers here.
///
/// \gotchas
/// llama.cpp refuses an unknown `tokenizer.ggml.pre`: an empty `vocabulary::pre` is written as "gpt-2" for GPT-2 and
/// "default" otherwise.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_convert_export_hpp
#define snicholls_formats_convert_export_hpp

#include <snicholls/formats/convert/convert.hpp>
#include <snicholls/formats/gguf/metadata.hpp>
#include <snicholls/formats/hf/names.hpp>
#include <snicholls/formats/llmc/llmc.hpp>

#include <cstddef>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::convert
{
/// \brief Converted tensors: the views to write, and the copies some of them point into.
struct converted
{
    std::vector<tensor_view> tensors;   ///< In the order to write them.
    std::deque<tensor> owned;           ///< Copies (stable addresses).
    std::vector<std::string> notes;     ///< What was changed or dropped.
};

/// \brief A GGUF writer and the copies its tensors point into.
struct gguf_export
{
    gguf::writer writer;                ///< The file, ready to `write()`.
    std::deque<tensor> owned;           ///< Copies the writer's tensors point into.
    std::vector<std::string> notes;     ///< What was changed or dropped.
};

namespace detail
{
    // A 2-D [rows, cols] tensor transposed to [cols, rows]
    inline tensor transposed(const tensor_view& t, std::string name)
    {
        const std::size_t rows = static_cast<std::size_t>(t.shape[0]), cols = static_cast<std::size_t>(t.shape[1]), es = size_of(t.type);
        tensor out{std::move(name), t.type, tensor_shape{cols, rows}, std::vector<std::byte>(t.data.size())};
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < cols; ++c)
                std::memcpy(out.data.data() + (c * rows + r) * es, t.data.data() + (r * cols + c) * es, es);
        return out;
    }
    inline tensor_view slice(const tensor_view& t, std::string_view name, std::size_t index)   // t[index] of a stacked tensor
    {
        tensor_shape s;
        for (std::size_t d = 1; d < t.shape.rank; ++d)
            s.push_back(t.shape[d]);
        const auto n = static_cast<std::size_t>(*s.elements()) * size_of(t.type);
        return {name, t.type, s, t.data.subspan(index * n, n)};
    }
} // namespace detail

/// \brief An llm.c GPT-2 checkpoint as Hugging Face's GPT-2 tensors. \param c The checkpoint.
/// \return The tensors (Conv1D weights transposed, the token table cut to the vocabulary).
[[nodiscard]] inline result<converted> llmc_to_hf(const llmc::checkpoint& c)
{
    converted out;
    const auto& s = c.shape;
    const std::size_t es = size_of(c.type), C = s.channels;
    const auto& t = c.tensors;
    const auto add = [&](std::string name, tensor_view v) {
        out.owned.push_back(tensor{std::move(name), v.type, v.shape, {}});
        out.tensors.push_back(tensor_view{out.owned.back().name, v.type, v.shape, v.data});
    };
    add("wte.weight", tensor_view{"", c.type, {s.vocab, C}, t[0].data.first(s.vocab * C * es)});
    add("wpe.weight", t[1]);
    for (std::size_t l = 0; l < s.layers; ++l) {
        const std::string h = "h." + std::to_string(l) + ".";
        add(h + "ln_1.weight", detail::slice(t[2], "", l));
        add(h + "ln_1.bias", detail::slice(t[3], "", l));
        out.owned.push_back(detail::transposed(detail::slice(t[4], "", l), h + "attn.c_attn.weight"));
        out.tensors.push_back(out.owned.back().view());
        add(h + "attn.c_attn.bias", detail::slice(t[5], "", l));
        out.owned.push_back(detail::transposed(detail::slice(t[6], "", l), h + "attn.c_proj.weight"));
        out.tensors.push_back(out.owned.back().view());
        add(h + "attn.c_proj.bias", detail::slice(t[7], "", l));
        add(h + "ln_2.weight", detail::slice(t[8], "", l));
        add(h + "ln_2.bias", detail::slice(t[9], "", l));
        out.owned.push_back(detail::transposed(detail::slice(t[10], "", l), h + "mlp.c_fc.weight"));
        out.tensors.push_back(out.owned.back().view());
        add(h + "mlp.c_fc.bias", detail::slice(t[11], "", l));
        out.owned.push_back(detail::transposed(detail::slice(t[12], "", l), h + "mlp.c_proj.weight"));
        out.tensors.push_back(out.owned.back().view());
        add(h + "mlp.c_proj.bias", detail::slice(t[13], "", l));
    }
    add("ln_f.weight", t[14]);
    add("ln_f.bias", t[15]);
    if (s.padded_vocab > s.vocab)
        out.notes.push_back("wte: the " + std::to_string(s.padded_vocab - s.vocab) + " padding rows are left out");
    return ok(std::move(out));
}

/// \brief The `config.json` of an llm.c GPT-2. \param s The sizes. \param bos_eos The end-of-text token (50256 in
/// GPT-2's vocabulary). \return The JSON.
[[nodiscard]] inline std::string hf_gpt2_config(const llmc::gpt2_shape& s, std::uint32_t bos_eos = 50256)
{
    return "{\n  \"architectures\": [\"GPT2LMHeadModel\"],\n  \"model_type\": \"gpt2\",\n  \"activation_function\": \"gelu_new\",\n"
           "  \"n_embd\": " + std::to_string(s.channels) + ",\n  \"n_head\": " + std::to_string(s.heads) + ",\n  \"n_layer\": "
           + std::to_string(s.layers) + ",\n  \"n_positions\": " + std::to_string(s.max_seq) + ",\n  \"n_ctx\": " + std::to_string(s.max_seq)
           + ",\n  \"vocab_size\": " + std::to_string(s.vocab) + ",\n  \"layer_norm_epsilon\": 1e-05,\n  \"bos_token_id\": "
           + std::to_string(bos_eos) + ",\n  \"eos_token_id\": " + std::to_string(bos_eos) + ",\n  \"tie_word_embeddings\": true\n}\n";
}

/// \brief An llm.c GPT-2 checkpoint as llama.cpp's `gpt2` GGUF. \param c The checkpoint (borrowed: the writer's
/// tensors are views of it). \param vocab The tokenizer (byte-level BPE; null writes none, and llama.cpp then cannot
/// tokenise). \param name `general.name`. \return The writer, or the problem.
[[nodiscard]] inline result<gguf_export> llmc_to_gguf(const llmc::checkpoint& c, const vocabulary* vocab, std::string_view name)
{
    gguf_export out;
    const auto& s = c.shape;
    const auto g = gguf::from_dtype(c.type);
    model_config m;
    m.arch = architecture::gpt2;
    m.block = ml::gpt2_small;
    m.block.d_model = s.channels;
    m.block.n_heads = m.block.n_kv_heads = s.heads;
    m.block.head_dim = s.channels / s.heads;
    m.block.d_ff = 4 * s.channels;
    m.layers = s.layers;
    m.context = s.max_seq;
    if (vocab) {
        if (vocab->tokens.size() != s.vocab)
            return fail(errc::bad_shape, "llmc_to_gguf: the vocabulary has " + std::to_string(vocab->tokens.size()) + " tokens, the model "
                                             + std::to_string(s.vocab));
        m.bos = vocab->bos;
        m.eos = vocab->eos;
    }
    SNICHOLLS_TRY_VOID(gguf::write_model_config(out.writer, m, name));
    out.writer.set(gguf::general::file_type, gguf::value_u32(static_cast<std::uint32_t>(c.type == dtype::bf16 ? gguf::file_type::mostly_bf16
                                                                                                                  : gguf::file_type::all_f32)));
    if (vocab) {
        vocabulary v = *vocab;
        if (v.pre.empty())
            v.pre = "gpt-2";
        gguf::write_vocabulary(out.writer, v);
    }
    const std::size_t es = size_of(c.type), C = s.channels;
    const auto& t = c.tensors;
    const auto add = [&](const std::string& n, tensor_view v) { out.writer.add_tensor(n, *g, v.shape, v.data); };   // views of the checkpoint
    add("token_embd.weight", tensor_view{"", c.type, {s.vocab, C}, t[0].data.first(s.vocab * C * es)});
    add("position_embd.weight", t[1]);
    for (std::size_t l = 0; l < s.layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        add(b + "attn_norm.weight", detail::slice(t[2], "", l));
        add(b + "attn_norm.bias", detail::slice(t[3], "", l));
        add(b + "attn_qkv.weight", detail::slice(t[4], "", l));
        add(b + "attn_qkv.bias", detail::slice(t[5], "", l));
        add(b + "attn_output.weight", detail::slice(t[6], "", l));
        add(b + "attn_output.bias", detail::slice(t[7], "", l));
        add(b + "ffn_norm.weight", detail::slice(t[8], "", l));
        add(b + "ffn_norm.bias", detail::slice(t[9], "", l));
        add(b + "ffn_up.weight", detail::slice(t[10], "", l));
        add(b + "ffn_up.bias", detail::slice(t[11], "", l));
        add(b + "ffn_down.weight", detail::slice(t[12], "", l));
        add(b + "ffn_down.bias", detail::slice(t[13], "", l));
    }
    add("output_norm.weight", t[14]);
    add("output_norm.bias", t[15]);
    out.notes.push_back("output.weight is not written: llama.cpp ties it to token_embd");
    return ok(std::move(out));
}

/// \brief A Hugging Face checkpoint as GGUF. \param c The configuration (`hf::parse_config`). \param tensors The
/// checkpoint's tensors, Hugging Face names. \param vocab The tokenizer, or null. \param chat_template The chat
/// template, or empty. \param name `general.name`. \return The writer, or the problem.
[[nodiscard]] inline result<gguf_export> hf_to_gguf(const model_config& c, std::span<const tensor_view> tensors, const vocabulary* vocab,
                                                    std::string_view chat_template, std::string_view name)
{
    if (c.arch == architecture::deepseek_v2 || c.arch == architecture::deepseek_v3)
        return fail(errc::unsupported, "hf_to_gguf: DeepSeek's latent attention is split by llama.cpp's converter in a way not done here");
    gguf_export out;
    SNICHOLLS_TRY_VOID(gguf::write_model_config(out.writer, c, name));
    if (vocab) {
        vocabulary v = *vocab;
        if (v.pre.empty())
            v.pre = c.arch == architecture::gpt2 ? "gpt-2" : "default";
        gguf::write_vocabulary(out.writer, v, chat_template);
    }
    // routed experts, gathered per layer and role, stacked after the loop
    std::map<std::pair<std::size_t, int>, std::vector<const tensor_view*>> experts;
    std::optional<dtype> common;
    for (const auto& t : tensors) {
        const auto n = hf::parse_hf_name(c.arch, t.name);
        if (!n)
            return fail(errc::unsupported, "hf_to_gguf: no rule names " + quote_name(t.name));
        if (n->what == hf::role::mask_buffer) {
            out.notes.push_back(std::string(t.name) + ": a mask buffer, dropped");
            continue;
        }
        if (n->what == hf::role::lm_head && c.tied_embeddings) {
            out.notes.push_back(std::string(t.name) + ": tied to the token embedding, dropped");
            continue;
        }
        const auto g = gguf::from_dtype(t.type);
        if (!g)
            return fail(errc::bad_dtype, "hf_to_gguf: " + quote_name(t.name) + " is " + std::string(safetensors_name(t.type)));
        common = common ? common : t.type;
        if (n->expert) {
            auto& list = experts[{*n->layer, static_cast<int>(n->what)}];
            if (list.size() <= *n->expert)
                list.resize(*n->expert + 1, nullptr);
            list[*n->expert] = &t;
            continue;
        }
        const std::string gname = hf::gguf_name(*n);
        if (hf::hf_transposed(c.arch, *n)) {
            if (t.shape.rank != 2)
                return fail(errc::bad_shape, "hf_to_gguf: " + quote_name(t.name) + " is not a matrix");
            out.owned.push_back(detail::transposed(t, gname));
            out.writer.add_tensor(out.owned.back().name, *g, out.owned.back().shape, out.owned.back().data);
        } else if (hf::gguf_permutes(c.arch, *n)) {
            const std::size_t heads = n->what == hf::role::q ? c.block.n_heads : c.block.n_kv_heads;
            const std::size_t rows = static_cast<std::size_t>(t.shape[0]);
            if (rows != heads * c.block.head_dim)
                return fail(errc::bad_shape, "hf_to_gguf: " + quote_name(t.name) + " has " + std::to_string(rows) + " rows for "
                                                 + std::to_string(heads) + " heads of " + std::to_string(c.block.head_dim));
            out.owned.push_back(tensor{gname, t.type, t.shape, std::vector<std::byte>(t.data.size())});
            hf::permute_rope(t.data, out.owned.back().data, heads, t.data.size() / rows);
            out.writer.add_tensor(out.owned.back().name, *g, t.shape, out.owned.back().data);
        } else {
            out.owned.push_back(tensor{gname, t.type, t.shape, {}});
            out.writer.add_tensor(out.owned.back().name, *g, t.shape, t.data);
        }
    }
    for (const auto& [key, list] : experts) {
        if (list.size() != c.block.experts)
            return fail(errc::not_found, "hf_to_gguf: layer " + std::to_string(key.first) + " has " + std::to_string(list.size()) + " of "
                                             + std::to_string(c.block.experts) + " experts");
        tensor stacked;
        for (std::size_t e = 0; e < list.size(); ++e) {
            if (!list[e])
                return fail(errc::not_found, "hf_to_gguf: layer " + std::to_string(key.first) + " has no expert " + std::to_string(e));
            if (list[e]->shape != list[0]->shape || list[e]->type != list[0]->type)
                return fail(errc::bad_shape, "hf_to_gguf: the experts of layer " + std::to_string(key.first) + " differ");
            stacked.data.insert(stacked.data.end(), list[e]->data.begin(), list[e]->data.end());
        }
        hf::tensor_name n{static_cast<hf::role>(key.second), key.first, {}, false};
        stacked.name = hf::gguf_name(n);
        stacked.type = list[0]->type;
        stacked.shape = tensor_shape{list.size()};
        for (const auto d : list[0]->shape.view())
            stacked.shape.push_back(d);
        out.owned.push_back(std::move(stacked));
        out.writer.add_tensor(out.owned.back().name, *gguf::from_dtype(out.owned.back().type), out.owned.back().shape, out.owned.back().data);
    }
    if (common)
        out.writer.set(gguf::general::file_type,
                       gguf::value_u32(static_cast<std::uint32_t>(*common == dtype::bf16  ? gguf::file_type::mostly_bf16
                                                                  : *common == dtype::f16 ? gguf::file_type::mostly_f16
                                                                                          : gguf::file_type::all_f32)));
    return ok(std::move(out));
}

/// \brief A GGUF model's tensors with Hugging Face names and layouts (the inverse of `hf_to_gguf`).
/// \param f The file. \param c Its configuration (`gguf::read_model_config`, or the original Hugging Face one).
/// \return The tensors, or the problem.
[[nodiscard]] inline result<converted> gguf_to_hf(const gguf::file& f, const model_config& c)
{
    converted out;
    for (const auto& t : f.tensors) {
        const auto n = hf::parse_gguf_name(t.name);
        if (!n)
            return fail(errc::unsupported, "gguf_to_hf: no rule names " + quote_name(t.name));
        const auto v = t.dense();
        if (!v)
            return fail(errc::unsupported, "gguf_to_hf: " + quote_name(t.name) + " is quantised; dequantize it first");
        if (n->what == hf::role::expert_gate || n->what == hf::role::expert_up || n->what == hf::role::expert_down) {
            for (std::size_t e = 0; e < t.shape[0]; ++e) {
                hf::tensor_name one = *n;
                one.expert = e;
                out.owned.push_back(tensor::from(detail::slice(*v, "", e)));
                out.owned.back().name = hf::hf_name(c.arch, one);
                out.tensors.push_back(out.owned.back().view());
            }
            continue;
        }
        const std::string hname = hf::hf_name(c.arch, *n);
        if (hname.empty())
            return fail(errc::unsupported, "gguf_to_hf: " + quote_name(t.name) + " has no Hugging Face name in " + std::string(hf_model_type(c.arch)));
        if (hf::hf_transposed(c.arch, *n)) {
            out.owned.push_back(detail::transposed(*v, hname));
        } else if (hf::gguf_permutes(c.arch, *n)) {
            const std::size_t heads = n->what == hf::role::q ? c.block.n_heads : c.block.n_kv_heads;
            out.owned.push_back(tensor{hname, v->type, v->shape, std::vector<std::byte>(v->data.size())});
            hf::unpermute_rope(v->data, out.owned.back().data, heads, v->data.size() / static_cast<std::size_t>(v->shape[0]));
        } else {
            out.owned.push_back(tensor{hname, v->type, v->shape, {}});
            out.tensors.push_back(tensor_view{out.owned.back().name, v->type, v->shape, v->data});
            continue;
        }
        out.tensors.push_back(out.owned.back().view());
    }
    return ok(std::move(out));
}
} // namespace snicholls::formats::convert

#endif // snicholls_formats_convert_export_hpp
