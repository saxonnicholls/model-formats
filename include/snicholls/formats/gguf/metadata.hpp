//
//  metadata.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief GGUF metadata to a `model_config` and a `vocabulary`, and back, for the architectures core describes.
/// \ingroup gguf
///
/// \overview
/// `read_model_config` maps llama.cpp's keys for `gpt2`, `llama` (Llama and Mistral), `qwen2`, `qwen3`, `olmoe` and
/// `deepseek2` (DeepSeek-V2 and V3) onto core's `transformer_config`; `write_model_config` writes the keys llama.cpp
/// reads for the same architectures. The mapping:
///
/// | GGUF | core |
/// |---|---|
/// | `embedding_length`, `feed_forward_length`, `block_count`, `context_length` | `d_model`, `d_ff`, `layers`, `context` |
/// | `attention.head_count`, `head_count_kv`, `key_length` | `n_heads`, `n_kv_heads`, `head_dim` |
/// | `attention.layer_norm_epsilon` / `layer_norm_rms_epsilon` | `norm`, `norm_eps` |
/// | `rope.freq_base`, `rope.scaling.*` | `rope_base`, `rope_scaling` (linear, YaRN) |
/// | `attention.sliding_window` | `window` |
/// | `expert_count`, `expert_used_count`, `expert_shared_count`, `expert_feed_forward_length` | the MoE fields |
/// | `expert_gating_func` (1 softmax, 2 sigmoid), `expert_weights_norm`, `expert_weights_scale` | `scoring`, `route_scale` |
/// | `expert_group_count`, `expert_group_used_count`, `leading_dense_block_count` | groups, `leading_dense_layers` |
/// | `attention.q_lora_rank`, `kv_lora_rank`, `key_length_mla`, `value_length_mla`, `rope.dimension_count` | MLA's ranks and widths |
/// | `nextn_predict_layers` | `mtp_depth` (and `block_count` less it) |
///
/// Floats are stored as f32 and read back as the shortest decimal that round-trips, so `1e-5` is `1e-5` again.
/// What GGUF does not carry: biases (read from the tensors: `blk.0.attn_q.bias` present), the embedding tie (no
/// `output.weight`), the training-only weights (balance and z losses, the bias update rate, MTP's lambda): left at 0
/// with a note. RoPE: llama and deepseek2 are adjacent-pair (`interleaved`, the Q/K weights permuted at conversion);
/// qwen2, qwen3 and olmoe are `half`.
///
/// `read_vocabulary` and `write_vocabulary` do the same for `tokenizer.ggml.*`: the model (gpt2, llama, bert, t5),
/// the tokens, their types and scores, the merges, the special ids, and the chat template.
///
/// \relations
/// \see formats::model_config, formats::vocabulary, formats::gguf::keys, formats::hf::parse_config
///
/// \impls
/// Four functions.
///
/// \contracts
/// A configuration returned satisfies `ml::valid`. `read_model_config(parse(write(...)))` gives back the mapped fields.
///
/// \errors
/// `unsupported` (another architecture), `bad_header` (a required key missing, or a configuration core rejects).
///
/// \examples
/// \snippet metadata_demo.cpp metadata
///
/// \perf
/// Not benchmarked: read once per model.
///
/// \testing
/// tests/gguf/metadata_test.cpp, demos/gguf/metadata_demo.cpp.
///
/// \deps
/// Core's `ml/transformer_config.hpp`.
///
/// \gotchas
/// OLMoE's whole-projection Q/K norm is not core's per-head `qk_norm`: noted, left none.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_gguf_metadata_hpp
#define snicholls_formats_gguf_metadata_hpp

#include <snicholls/formats/gguf/gguf.hpp>
#include <snicholls/formats/gguf/keys.hpp>
#include <snicholls/formats/model_config.hpp>
#include <snicholls/formats/vocabulary.hpp>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

namespace snicholls::formats::gguf
{
namespace detail
{
    inline std::optional<std::size_t> size_key(const file& f, std::string_view arch, std::string_view suffix)
    {
        const value* v = f.find(arch_key(arch, suffix));
        const auto u = v ? v->as_u64() : std::nullopt;
        return u ? std::optional<std::size_t>(static_cast<std::size_t>(*u)) : std::nullopt;
    }
    // A float32 key as the decimal it was written from: 1e-5 stored as f32 reads back as 1e-5, not 9.99999974e-6
    // (the shortest text that round-trips the float, read as a double)
    inline std::optional<double> double_key(const file& f, std::string_view arch, std::string_view suffix)
    {
        const value* v = f.find(arch_key(arch, suffix));
        if (!v)
            return std::nullopt;
        if (v->type != value_type::f32)
            return v->as_f64();
        char buf[32];
        const auto r = std::to_chars(buf, buf + sizeof buf - 1, static_cast<float>(*v->as_f64()));
        *r.ptr = '\0';
        return std::strtod(buf, nullptr);
    }
    inline std::optional<std::uint32_t> token_key(const file& f, std::string_view key)
    {
        const value* v = f.find(key);
        const auto u = v ? v->as_u64() : std::nullopt;
        return u && *u <= 0xffffffffu ? std::optional<std::uint32_t>(static_cast<std::uint32_t>(*u)) : std::nullopt;
    }
} // namespace detail

/// \brief Reads the architecture from a GGUF file's metadata. \param f The file.
/// \return The configuration, or `unsupported` / `bad_header`.
[[nodiscard]] inline result<model_config> read_model_config(const file& f)
{
    const std::string_view a = f.architecture();
    model_config c;
    if (a == "gpt2")
        c.arch = architecture::gpt2;
    else if (a == "llama")
        c.arch = architecture::llama;
    else if (a == "qwen2")
        c.arch = architecture::qwen2;
    else if (a == "qwen3")
        c.arch = architecture::qwen3;
    else if (a == "olmoe")
        c.arch = architecture::olmoe;
    else if (a == "deepseek2")
        c.arch = architecture::deepseek_v2;
    else
        return fail(errc::unsupported, "gguf: architecture \"" + std::string(a) + "\" is not mapped onto core");
    const auto need = [&](std::string_view suffix) -> result<std::size_t> {
        if (const auto v = detail::size_key(f, a, suffix))
            return ok(*v);
        return fail(errc::bad_header, "gguf: no " + arch_key(a, suffix));
    };
    auto& b = c.block;
    SNICHOLLS_TRY_ASSIGN(b.d_model, need(llm::embedding_length));
    SNICHOLLS_TRY_ASSIGN(c.layers, need(llm::block_count));
    SNICHOLLS_TRY_ASSIGN(b.n_heads, need(attention::head_count));
    SNICHOLLS_TRY_ASSIGN(b.d_ff, need(llm::feed_forward_length));
    c.context = detail::size_key(f, a, llm::context_length).value_or(0);
    b.n_kv_heads = detail::size_key(f, a, attention::head_count_kv).value_or(b.n_heads);
    b.head_dim = detail::size_key(f, a, attention::key_length).value_or(b.n_heads ? b.d_model / b.n_heads : 0);
    c.vocab = detail::size_key(f, a, llm::vocab_size).value_or(0);
    if (c.vocab == 0)
        if (const value* t = f.find(tokenizer::tokens); t && t->type == value_type::array)
            c.vocab = static_cast<std::size_t>(t->count);
    c.bos = detail::token_key(f, tokenizer::bos_token_id);
    c.eos = detail::token_key(f, tokenizer::eos_token_id);
    c.tied_embeddings = f.find_tensor("output.weight") == nullptr;
    b.bias = f.find_tensor("blk.0.attn_q.bias") != nullptr || f.find_tensor("blk.0.attn_qkv.bias") != nullptr;
    b.rope_base = detail::double_key(f, a, rope::freq_base).value_or(10000.0);
    if (const auto e = detail::double_key(f, a, attention::layer_norm_rms_epsilon)) {
        b.norm = ml::norm_kind::rms;
        b.norm_eps = *e;
    } else if (const auto l = detail::double_key(f, a, attention::layer_norm_epsilon)) {
        b.norm = ml::norm_kind::layer;
        b.norm_eps = *l;
    }
    if (const value* s = f.find(arch_key(a, rope::scaling_type))) {
        const std::string_view kind = s->as_string().value_or("");
        const double factor = detail::double_key(f, a, rope::scaling_factor).value_or(1.0);
        if (kind == "linear") {
            b.rope_scaling = ml::rope_scaling_kind::linear;
            b.rope_factor = factor;
        } else if (kind == "yarn") {
            b.rope_scaling = ml::rope_scaling_kind::yarn_index_ramp;
            b.rope_factor = factor;
            b.rope_original_context = detail::size_key(f, a, rope::scaling_original_context_length).value_or(c.context);
            b.yarn_beta_fast = detail::double_key(f, a, rope::scaling_yarn_beta_fast).value_or(32.0);
            b.yarn_beta_slow = detail::double_key(f, a, rope::scaling_yarn_beta_slow).value_or(1.0);
            // llama.cpp stores DeepSeek's 0.1 * mscale_all_dim as the log multiplier
            if (const auto m = detail::double_key(f, a, rope::scaling_yarn_log_multiplier))
                b.yarn_mscale = *m / 0.1;
        } else if (kind != "none" && !kind.empty()) {
            c.notes.push_back("rope.scaling.type \"" + std::string(kind) + "\" has no counterpart in core");
        }
    }
    switch (c.arch) {
    case architecture::gpt2:
        b.mlp = ml::mlp_kind::gelu_tanh;
        b.rope = ml::rope_kind::none;
        b.bias = true;
        break;
    case architecture::llama:
    case architecture::mistral:
        b.mlp = ml::mlp_kind::swiglu;
        b.rope = ml::rope_kind::interleaved;
        b.window = detail::size_key(f, a, attention::sliding_window).value_or(0);
        break;
    case architecture::qwen2:
    case architecture::qwen3:
        b.mlp = ml::mlp_kind::swiglu;
        b.rope = ml::rope_kind::half;
        if (c.arch == architecture::qwen3)
            b.qk_norm = ml::qk_norm_kind::before_rope;
        break;
    case architecture::olmoe:
    case architecture::deepseek_v2:
    case architecture::deepseek_v3:
        b.mlp = ml::mlp_kind::swiglu;
        b.rope = c.arch == architecture::olmoe ? ml::rope_kind::half : ml::rope_kind::interleaved;
        break;
    }
    if (const auto experts = detail::size_key(f, a, llm::expert_count); experts && *experts > 0) {
        b.experts = *experts;
        b.top_k = detail::size_key(f, a, llm::expert_used_count).value_or(0);
        b.shared_experts = detail::size_key(f, a, llm::expert_shared_count).value_or(0);
        b.expert_d_ff = detail::size_key(f, a, llm::expert_feed_forward_length).value_or(b.d_ff);
        b.route_scale = detail::double_key(f, a, llm::expert_weights_scale).value_or(1.0);
        b.expert_groups = detail::size_key(f, a, llm::expert_group_count).value_or(0);
        b.group_top_k = detail::size_key(f, a, llm::expert_group_used_count).value_or(0);
        c.leading_dense_layers = detail::size_key(f, a, llm::leading_dense_block_count).value_or(0);
        const auto gating = detail::size_key(f, a, llm::expert_gating_func).value_or(1);
        const value* norm = f.find(arch_key(a, llm::expert_weights_norm));
        const bool renormalise = norm && norm->as_bool().value_or(false);
        b.scoring = gating == 2 ? ml::moe_scoring::sigmoid : renormalise ? ml::moe_scoring::softmax_top_k : ml::moe_scoring::softmax;
        if (gating == 2)
            c.arch = architecture::deepseek_v3;
        c.notes.push_back("gguf: the router's training weights (balance and z losses, the bias update rate) are not stored; left 0");
    }
    if (c.arch == architecture::deepseek_v2 || c.arch == architecture::deepseek_v3) {
        b.q_lora_rank = detail::size_key(f, a, attention::q_lora_rank).value_or(0);
        b.kv_lora_rank = detail::size_key(f, a, attention::kv_lora_rank).value_or(0);
        b.qk_rope_dim = detail::size_key(f, a, rope::dimension_count).value_or(0);
        b.head_dim = detail::size_key(f, a, attention::key_length_mla).value_or(b.head_dim);
        b.v_head_dim = detail::size_key(f, a, attention::value_length_mla)
                           .value_or(detail::size_key(f, a, attention::value_length).value_or(0));
        b.n_kv_heads = b.n_heads;   // latent attention; llama.cpp writes 1 for its absorbed form
        b.mtp_depth = detail::size_key(f, a, llm::nextn_predict_layers).value_or(0);
        if (c.layers > b.mtp_depth)
            c.layers -= b.mtp_depth;   // llama.cpp counts the MTP layers as blocks
    }
    if (c.arch == architecture::olmoe)
        c.notes.push_back("olmoe: q_norm and k_norm normalise the whole projection; core's qk_norm is per head, so it is left none");
    if (!ml::valid(b))
        return fail(errc::bad_header, "gguf: the " + std::string(a) + " configuration is not one core accepts (ml::valid)");
    return ok(std::move(c));
}

/// \brief Writes the architecture's keys. \param w The writer. \param c The configuration. \param name `general.name`.
/// \return Nothing, or `unsupported` for a configuration GGUF cannot express (a qk_norm other than Qwen 3's, a mask
/// other than causal).
[[nodiscard]] inline result<void> write_model_config(writer& w, const model_config& c, std::string_view name)
{
    const auto& b = c.block;
    if (b.mask != ml::attention_mask::causal || (b.qk_norm != ml::qk_norm_kind::none && c.arch != architecture::qwen3))
        return fail(errc::unsupported, "gguf: llama.cpp's architectures have no such attention");
    const std::string_view a = gguf_architecture(c.arch);
    const auto u32 = [&](std::string_view suffix, std::size_t v) { w.set(arch_key(a, suffix), value_u32(static_cast<std::uint32_t>(v))); };
    const auto f32 = [&](std::string_view suffix, double v) { w.set(arch_key(a, suffix), value_f32(static_cast<float>(v))); };
    w.set(general::architecture, value_string(a));
    w.set(general::name, value_string(name));
    w.set(general::quantization_version, value_u32(2));
    u32(llm::context_length, c.context);
    u32(llm::embedding_length, b.d_model);
    u32(llm::block_count, c.layers + b.mtp_depth);
    u32(llm::feed_forward_length, b.d_ff);
    u32(attention::head_count, b.n_heads);
    if (c.vocab)
        u32(llm::vocab_size, c.vocab);
    if (b.norm == ml::norm_kind::layer)
        f32(attention::layer_norm_epsilon, b.norm_eps);
    else
        f32(attention::layer_norm_rms_epsilon, b.norm_eps);
    if (b.rope != ml::rope_kind::none) {
        u32(attention::head_count_kv, b.latent() ? 1 : b.n_kv_heads);
        u32(attention::key_length, b.latent() ? b.kv_lora_rank + b.qk_rope_dim : b.head_dim);
        u32(attention::value_length, b.latent() ? b.kv_lora_rank : b.head_dim);
        f32(rope::freq_base, b.rope_base);
        u32(rope::dimension_count, b.latent() ? b.qk_rope_dim : b.head_dim);
    }
    if (b.window)
        u32(attention::sliding_window, b.window);
    if (b.rope_scaling == ml::rope_scaling_kind::linear) {
        w.set(arch_key(a, rope::scaling_type), value_string("linear"));
        f32(rope::scaling_factor, b.rope_factor);
    } else if (b.rope_scaling == ml::rope_scaling_kind::yarn || b.rope_scaling == ml::rope_scaling_kind::yarn_index_ramp) {
        w.set(arch_key(a, rope::scaling_type), value_string("yarn"));
        f32(rope::scaling_factor, b.rope_factor);
        u32(rope::scaling_original_context_length, b.rope_original_context);
        f32(rope::scaling_yarn_beta_fast, b.yarn_beta_fast);
        f32(rope::scaling_yarn_beta_slow, b.yarn_beta_slow);
        f32(rope::scaling_yarn_log_multiplier, 0.1 * b.yarn_mscale);
    }
    if (b.moe()) {
        u32(llm::expert_count, b.experts);
        u32(llm::expert_used_count, b.top_k);
        u32(llm::expert_shared_count, b.shared_experts);
        u32(llm::expert_feed_forward_length, b.expert_d_ff);
        f32(llm::expert_weights_scale, b.route_scale);
        w.set(arch_key(a, llm::expert_weights_norm), value_bool(b.scoring != ml::moe_scoring::softmax));
        u32(llm::expert_gating_func, b.scoring == ml::moe_scoring::sigmoid ? 2 : 1);
        if (b.expert_groups) {
            u32(llm::expert_group_count, b.expert_groups);
            u32(llm::expert_group_used_count, b.group_top_k);
        }
        u32(llm::leading_dense_block_count, c.leading_dense_layers);
    }
    if (b.latent()) {
        if (b.q_lora_rank)
            u32(attention::q_lora_rank, b.q_lora_rank);
        u32(attention::kv_lora_rank, b.kv_lora_rank);
        u32(attention::key_length_mla, b.head_dim);
        u32(attention::value_length_mla, b.v_head_dim);
        if (b.mtp_depth)
            u32(llm::nextn_predict_layers, b.mtp_depth);
    }
    if (c.bos)
        w.set(tokenizer::bos_token_id, value_u32(*c.bos));
    if (c.eos)
        w.set(tokenizer::eos_token_id, value_u32(*c.eos));
    return ok();
}

/// \brief Reads `tokenizer.ggml.*`. \param f The file. \return The vocabulary, or `not_found` (no tokenizer),
/// `unsupported` (another model) or `bad_header`.
[[nodiscard]] inline result<vocabulary> read_vocabulary(const file& f)
{
    const value* model = f.find(tokenizer::model);
    const value* toks = f.find(tokenizer::tokens);
    if (!model || !toks)
        return fail(errc::not_found, "gguf: no tokenizer.ggml.model and tokens");
    vocabulary v;
    const std::string_view m = model->as_string().value_or("");
    if (m == "gpt2")
        v.kind = tokenizer_kind::byte_level_bpe;
    else if (m == "llama")
        v.kind = tokenizer_kind::sentencepiece;
    else if (m == "bert")
        v.kind = tokenizer_kind::wordpiece;
    else if (m == "t5")
        v.kind = tokenizer_kind::unigram;
    else
        return fail(errc::unsupported, "gguf: tokenizer model \"" + std::string(m) + "\" is not read");
    if (const value* pre = f.find(tokenizer::pre))
        v.pre = std::string(pre->as_string().value_or(""));
    const auto strings = toks->strings();
    if (!strings)
        return fail(errc::bad_header, "gguf: tokenizer.ggml.tokens is not an array of strings");
    v.tokens.assign(strings->begin(), strings->end());
    if (const value* types = f.find(tokenizer::token_type)) {
        const auto t = types->integers();
        if (!t || t->size() != v.tokens.size())
            return fail(errc::bad_header, "gguf: tokenizer.ggml.token_type does not match the tokens");
        for (const auto x : *t)
            v.types.push_back(static_cast<token_type>(x));
    }
    if (const value* scores = f.find(tokenizer::scores)) {
        const auto s = scores->numbers();
        if (!s || s->size() != v.tokens.size())
            return fail(errc::bad_header, "gguf: tokenizer.ggml.scores does not match the tokens");
        for (const auto x : *s)
            v.scores.push_back(static_cast<float>(x));
    }
    if (const value* merges = f.find(tokenizer::merges)) {
        const auto s = merges->strings();
        if (!s)
            return fail(errc::bad_header, "gguf: tokenizer.ggml.merges is not an array of strings");
        v.merges.assign(s->begin(), s->end());
    }
    v.bos = detail::token_key(f, tokenizer::bos_token_id);
    v.eos = detail::token_key(f, tokenizer::eos_token_id);
    v.unk = detail::token_key(f, tokenizer::unknown_token_id);
    v.pad = detail::token_key(f, tokenizer::padding_token_id);
    return ok(std::move(v));
}

/// \brief Writes `tokenizer.ggml.*`. \param w The writer. \param v The vocabulary. \param chat_template The chat
/// template's text, or empty for none.
inline void write_vocabulary(writer& w, const vocabulary& v, std::string_view chat_template = {})
{
    w.set(tokenizer::model, value_string(gguf_tokenizer_model(v.kind)));
    w.set(tokenizer::pre, value_string(v.pre.empty() ? std::string_view("default") : std::string_view(v.pre)));
    w.set(tokenizer::tokens, value_strings(v.tokens));
    std::vector<metadata_value> types;
    types.reserve(v.tokens.size());
    for (std::uint32_t i = 0; i < v.tokens.size(); ++i)
        types.push_back(value_i32(static_cast<std::int32_t>(v.type_of(i))));
    w.set(tokenizer::token_type, value_array(value_type::i32, std::move(types)));
    if (!v.scores.empty()) {
        std::vector<metadata_value> scores;
        for (const float s : v.scores)
            scores.push_back(value_f32(s));
        w.set(tokenizer::scores, value_array(value_type::f32, std::move(scores)));
    }
    if (!v.merges.empty() || v.kind == tokenizer_kind::byte_level_bpe)   // llama.cpp requires the key for gpt2, even empty
        w.set(tokenizer::merges, value_strings(v.merges));
    if (v.bos)
        w.set(tokenizer::bos_token_id, value_u32(*v.bos));
    if (v.eos)
        w.set(tokenizer::eos_token_id, value_u32(*v.eos));
    if (v.unk)
        w.set(tokenizer::unknown_token_id, value_u32(*v.unk));
    if (v.pad)
        w.set(tokenizer::padding_token_id, value_u32(*v.pad));
    if (!chat_template.empty())
        w.set(tokenizer::chat_template, value_string(chat_template));
}
} // namespace snicholls::formats::gguf

#endif // snicholls_formats_gguf_metadata_hpp
