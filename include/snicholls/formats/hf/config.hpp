//
//  config.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Hugging Face's `config.json` to a `model_config`, and `generation_config.json`.
/// \ingroup hf
///
/// \overview
/// `parse_config` reads a transformers `config.json` for GPT-2, Llama, Mistral, Qwen 2, Qwen 3, OLMoE, DeepSeek-V2
/// and DeepSeek-V3 into core's `transformer_config` and the counts around it, using transformers' defaults where a
/// field is absent. What does not map is kept in `notes`, never dropped silently:
///
/// | Field | Where it goes |
/// |---|---|
/// | `rope_scaling` linear / YaRN | `rope_scaling`, `rope_factor`, YaRN's context, betas and mscale (YaRN as Hugging Face computes it: `yarn_index_ramp`) |
/// | `rope_scaling` llama3, dynamic, longrope | a note: core has no such rescaling |
/// | Qwen 2's Q/K/V biases | `bias = true`, with a note that the O and MLP biases are zero |
/// | OLMoE's `q_norm` / `k_norm` | a note: OLMoE normalises the whole projection, core's `qk_norm` is per head |
/// | `norm_topk_prob` with softmax routing | `softmax_top_k` when true, `softmax` when false |
/// | DeepSeek's `first_k_dense_replace` | `leading_dense_layers`; the dense blocks' width is `d_ff` |
/// | DeepSeek-V3's bias update rate | a note: the configuration does not carry it |
/// | `hidden_act` other than silu (gated) / gelu_new / gelu | an error: core has no such MLP |
///
/// `parse_generation_config` reads the sampling defaults a model ships with.
///
/// \relations
/// \see formats::model_config, formats::hf::checkpoint_names, formats::gguf::write_model_config
///
/// \impls
/// One parser per architecture family, sharing the attention and RoPE fields.
///
/// \contracts
/// A configuration returned satisfies `ml::valid(c.block)`.
///
/// \errors
/// `bad_json`; `bad_header` for a missing required field or a configuration core rejects; `unsupported` for a
/// `model_type` or activation not mapped.
///
/// \examples
/// \snippet config_demo.cpp config
///
/// \perf
/// Not benchmarked: read once per model.
///
/// \testing
/// tests/hf/config_test.cpp (the configurations of GPT-2, Llama 3 8B, Qwen 3, OLMoE and DeepSeek-V3 against core's
/// presets), demos/hf/config_demo.cpp.
///
/// \deps
/// nlohmann/json through core-adapters.
///
/// \gotchas
/// Hugging Face Llama applies RoPE to the halves of a head (`rotate_half`): `rope_kind::half`.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_hf_config_hpp
#define snicholls_formats_hf_config_hpp

#include <snicholls/formats/detail/json.hpp>
#include <snicholls/formats/model_config.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::hf
{
namespace detail
{
    using formats::detail::json;
    inline std::optional<std::size_t> size_at(const json& j, std::string_view key)
    {
        const auto v = formats::detail::u64_at(j, key);
        return v ? std::optional<std::size_t>(static_cast<std::size_t>(*v)) : std::nullopt;
    }
    inline result<std::size_t> required(const json& j, std::string_view key)
    {
        if (const auto v = size_at(j, key))
            return ok(*v);
        return fail(errc::bad_header, "config.json: no " + std::string(key));
    }
    inline std::optional<std::uint32_t> token_at(const json& j, std::string_view key)
    {
        const formats::detail::json* m = formats::detail::member(j, key);
        if (!m)
            return std::nullopt;
        if (m->is_array() && !m->empty())   // generation configs list several end tokens; the first is the one
            m = &(*m)[0];
        const auto v = formats::detail::as_u64(*m);
        return v && *v <= 0xffffffffu ? std::optional<std::uint32_t>(static_cast<std::uint32_t>(*v)) : std::nullopt;
    }

    // RoPE: base and scaling, shared by every RoPE architecture
    inline void rope(const json& j, model_config& c)
    {
        c.block.rope_base = formats::detail::double_at(j, "rope_theta").value_or(10000.0);
        const json* s = formats::detail::member(j, "rope_scaling");
        if (!s || s->is_null())
            return;
        const std::string* kind = formats::detail::string_at(*s, "rope_type");
        if (!kind)
            kind = formats::detail::string_at(*s, "type");
        const std::string k = kind ? *kind : "";
        const double factor = formats::detail::double_at(*s, "factor").value_or(1.0);
        if (k == "linear") {
            c.block.rope_scaling = ml::rope_scaling_kind::linear;
            c.block.rope_factor = factor;
        } else if (k == "yarn") {
            c.block.rope_scaling = ml::rope_scaling_kind::yarn_index_ramp;
            c.block.rope_factor = factor;
            c.block.rope_original_context = size_at(*s, "original_max_position_embeddings").value_or(c.context);
            c.block.yarn_beta_fast = formats::detail::double_at(*s, "beta_fast").value_or(32.0);
            c.block.yarn_beta_slow = formats::detail::double_at(*s, "beta_slow").value_or(1.0);
            const double mscale = formats::detail::double_at(*s, "mscale").value_or(1.0);
            const double mscale_all = formats::detail::double_at(*s, "mscale_all_dim").value_or(0.0);
            c.block.yarn_mscale = mscale_all > 0 ? mscale_all : mscale;
            if (mscale_all > 0 && mscale != mscale_all)
                c.notes.push_back("rope_scaling: YaRN's mscale " + std::to_string(mscale) + " differs from mscale_all_dim "
                                  + std::to_string(mscale_all) + "; core takes one coefficient (mscale_all_dim)");
        } else if (!k.empty() && k != "default") {
            c.notes.push_back("rope_scaling: \"" + k + "\" (factor " + std::to_string(factor) + ") has no counterpart in core; the frequencies are as trained");
        }
    }

    inline result<ml::mlp_kind> gated_act(const json& j, std::string_view fallback)
    {
        const std::string* act = formats::detail::string_at(j, "hidden_act");
        const std::string a = act ? *act : std::string(fallback);
        if (a == "silu" || a == "swish")
            return ok(ml::mlp_kind::swiglu);
        return fail(errc::unsupported, "config.json: gated activation \"" + a + "\" is not core's SwiGLU");
    }

    // The Llama family: attention, MLP, norms, RoPE
    inline result<void> llama_like(const json& j, model_config& c)
    {
        SNICHOLLS_TRY_ASSIGN(c.block.d_model, required(j, "hidden_size"));
        SNICHOLLS_TRY_ASSIGN(c.block.n_heads, required(j, "num_attention_heads"));
        SNICHOLLS_TRY_ASSIGN(c.layers, required(j, "num_hidden_layers"));
        SNICHOLLS_TRY_ASSIGN(c.vocab, required(j, "vocab_size"));
        SNICHOLLS_TRY_ASSIGN(c.block.d_ff, required(j, "intermediate_size"));
        c.block.n_kv_heads = size_at(j, "num_key_value_heads").value_or(c.block.n_heads);
        c.block.head_dim = size_at(j, "head_dim").value_or(c.block.n_heads ? c.block.d_model / c.block.n_heads : 0);
        c.context = size_at(j, "max_position_embeddings").value_or(2048);
        c.block.norm = ml::norm_kind::rms;
        c.block.norm_eps = formats::detail::double_at(j, "rms_norm_eps").value_or(1e-6);
        SNICHOLLS_TRY_ASSIGN(c.block.mlp, gated_act(j, "silu"));
        c.block.rope = ml::rope_kind::half;
        c.block.bias = formats::detail::bool_at(j, "attention_bias").value_or(false);
        if (c.block.bias)
            c.notes.push_back("attention_bias: biases on Q, K, V and O; core's bias puts them on the MLP too, where they are zero");
        c.tied_embeddings = formats::detail::bool_at(j, "tie_word_embeddings").value_or(false);
        rope(j, c);
        return ok();
    }
} // namespace detail

/// \brief Reads a `config.json`. \param text The JSON. \return The configuration, or the problem.
[[nodiscard]] inline result<model_config> parse_config(std::string_view text)
{
    using formats::detail::bool_at;
    using formats::detail::double_at;
    using formats::detail::string_at;
    SNICHOLLS_TRY_ASSIGN(const detail::json j, formats::detail::parse_json(text, "config.json"));
    const std::string* type = string_at(j, "model_type");
    if (!type)
        return fail(errc::bad_header, "config.json: no model_type");
    const auto arch = architecture_from_hf(*type);
    if (!arch)
        return fail(errc::unsupported, "config.json: model_type \"" + *type + "\" is not mapped onto core");
    model_config c;
    c.arch = *arch;
    c.bos = detail::token_at(j, "bos_token_id");
    c.eos = detail::token_at(j, "eos_token_id");
    switch (c.arch) {
    case architecture::gpt2: {
        SNICHOLLS_TRY_ASSIGN(c.block.d_model, detail::required(j, "n_embd"));
        SNICHOLLS_TRY_ASSIGN(c.block.n_heads, detail::required(j, "n_head"));
        SNICHOLLS_TRY_ASSIGN(c.layers, detail::required(j, "n_layer"));
        SNICHOLLS_TRY_ASSIGN(c.vocab, detail::required(j, "vocab_size"));
        c.block.n_kv_heads = c.block.n_heads;
        c.block.head_dim = c.block.n_heads ? c.block.d_model / c.block.n_heads : 0;
        c.block.d_ff = detail::size_at(j, "n_inner").value_or(4 * c.block.d_model);
        c.context = detail::size_at(j, "n_positions").value_or(detail::size_at(j, "n_ctx").value_or(1024));
        c.block.norm = ml::norm_kind::layer;
        c.block.norm_eps = double_at(j, "layer_norm_epsilon").value_or(1e-5);
        const std::string* act = string_at(j, "activation_function");
        const std::string a = act ? *act : "gelu_new";
        if (a == "gelu_new" || a == "gelu_pytorch_tanh" || a == "gelu_fast")
            c.block.mlp = ml::mlp_kind::gelu_tanh;
        else if (a == "gelu")
            c.block.mlp = ml::mlp_kind::gelu;
        else
            return fail(errc::unsupported, "config.json: GPT-2 activation \"" + a + "\" is not one of core's");
        c.block.rope = ml::rope_kind::none;
        c.block.bias = true;
        c.tied_embeddings = bool_at(j, "tie_word_embeddings").value_or(true);
        if (bool_at(j, "scale_attn_by_inverse_layer_idx").value_or(false))
            c.notes.push_back("scale_attn_by_inverse_layer_idx: per-layer score scaling has no counterpart in core");
        break;
    }
    case architecture::llama:
    case architecture::mistral:
    case architecture::qwen2:
    case architecture::qwen3: {
        SNICHOLLS_TRY_VOID(detail::llama_like(j, c));
        if (c.arch == architecture::qwen2) {
            c.block.bias = true;   // Q, K and V carry biases whatever attention_bias says
            c.notes.assign(1, "qwen2: biases on Q, K and V only; core's bias puts them on O and the MLP too, where they are zero");
        }
        if (c.arch == architecture::qwen3)
            c.block.qk_norm = ml::qk_norm_kind::before_rope;
        const bool sliding = c.arch == architecture::mistral || bool_at(j, "use_sliding_window").value_or(false);
        if (sliding)
            if (const auto w = detail::size_at(j, "sliding_window"))
                c.block.window = *w;
        if (c.arch == architecture::qwen2 || c.arch == architecture::qwen3)
            if (const auto m = detail::size_at(j, "max_window_layers"); m && c.block.window && *m < c.layers)
                c.notes.push_back("max_window_layers: the window applies to some layers only; core's is per block");
        break;
    }
    case architecture::olmoe: {
        SNICHOLLS_TRY_VOID(detail::llama_like(j, c));
        SNICHOLLS_TRY_ASSIGN(c.block.experts, detail::required(j, "num_experts"));
        SNICHOLLS_TRY_ASSIGN(c.block.top_k, detail::required(j, "num_experts_per_tok"));
        c.block.expert_d_ff = c.block.d_ff;
        c.block.scoring = bool_at(j, "norm_topk_prob").value_or(false) ? ml::moe_scoring::softmax_top_k : ml::moe_scoring::softmax;
        c.block.balance_loss_weight = double_at(j, "router_aux_loss_coef").value_or(0.01);
        c.block.z_loss_weight = double_at(j, "router_z_loss_coef").value_or(0.001);
        c.notes.push_back("olmoe: q_norm and k_norm normalise the whole projection; core's qk_norm is per head, so it is left none");
        if (const auto clip = double_at(j, "clip_qkv"); clip && *clip > 0)
            c.notes.push_back("clip_qkv: clipping Q, K and V has no counterpart in core");
        break;
    }
    case architecture::deepseek_v2:
    case architecture::deepseek_v3: {
        SNICHOLLS_TRY_ASSIGN(c.block.d_model, detail::required(j, "hidden_size"));
        SNICHOLLS_TRY_ASSIGN(c.block.n_heads, detail::required(j, "num_attention_heads"));
        SNICHOLLS_TRY_ASSIGN(c.layers, detail::required(j, "num_hidden_layers"));
        SNICHOLLS_TRY_ASSIGN(c.vocab, detail::required(j, "vocab_size"));
        SNICHOLLS_TRY_ASSIGN(c.block.d_ff, detail::required(j, "intermediate_size"));
        SNICHOLLS_TRY_ASSIGN(c.block.kv_lora_rank, detail::required(j, "kv_lora_rank"));
        SNICHOLLS_TRY_ASSIGN(c.block.qk_rope_dim, detail::required(j, "qk_rope_head_dim"));
        SNICHOLLS_TRY_ASSIGN(const std::size_t nope, detail::required(j, "qk_nope_head_dim"));
        SNICHOLLS_TRY_ASSIGN(c.block.v_head_dim, detail::required(j, "v_head_dim"));
        c.block.n_kv_heads = c.block.n_heads;   // latent attention: one key and value per query head
        c.block.head_dim = nope + c.block.qk_rope_dim;
        c.block.q_lora_rank = detail::size_at(j, "q_lora_rank").value_or(0);   // null in V2-Lite: the query is projected directly
        c.context = detail::size_at(j, "max_position_embeddings").value_or(4096);
        c.block.norm = ml::norm_kind::rms;
        c.block.norm_eps = double_at(j, "rms_norm_eps").value_or(1e-6);
        SNICHOLLS_TRY_ASSIGN(c.block.mlp, detail::gated_act(j, "silu"));
        c.block.rope = ml::rope_kind::interleaved;
        c.tied_embeddings = bool_at(j, "tie_word_embeddings").value_or(false);
        detail::rope(j, c);
        if (const auto experts = detail::size_at(j, "n_routed_experts"); experts && *experts > 0) {
            c.block.experts = *experts;
            SNICHOLLS_TRY_ASSIGN(c.block.top_k, detail::required(j, "num_experts_per_tok"));
            SNICHOLLS_TRY_ASSIGN(c.block.expert_d_ff, detail::required(j, "moe_intermediate_size"));
            c.block.shared_experts = detail::size_at(j, "n_shared_experts").value_or(0);
            c.block.route_scale = double_at(j, "routed_scaling_factor").value_or(1.0);
            c.block.balance_loss_weight = double_at(j, "aux_loss_alpha").value_or(0.0);
            c.leading_dense_layers = detail::size_at(j, "first_k_dense_replace").value_or(0);
            const std::string* scoring = string_at(j, "scoring_func");
            const std::string s = scoring ? *scoring : "softmax";
            const bool renormalise = bool_at(j, "norm_topk_prob").value_or(false);
            if (s == "sigmoid") {
                c.block.scoring = ml::moe_scoring::sigmoid;
                if (!renormalise)
                    c.notes.push_back("norm_topk_prob false with sigmoid scoring: core's sigmoid weights are always renormalised");
            } else if (s == "softmax") {
                c.block.scoring = renormalise ? ml::moe_scoring::softmax_top_k : ml::moe_scoring::softmax;
            } else {
                return fail(errc::unsupported, "config.json: scoring_func \"" + s + "\" is not one of core's");
            }
            const std::string* method = string_at(j, "topk_method");
            if (method && (*method == "group_limited_greedy" || *method == "noaux_tc")) {
                c.block.expert_groups = detail::size_at(j, "n_group").value_or(0);
                c.block.group_top_k = detail::size_at(j, "topk_group").value_or(0);
            }
            if (method && *method == "noaux_tc")
                c.notes.push_back("noaux_tc: the selection bias's update rate is not in config.json (DeepSeek-V3 used 0.001); left 0");
        }
        c.block.mtp_depth = detail::size_at(j, "num_nextn_predict_layers").value_or(0);
        break;
    }
    }
    if (!ml::valid(c.block))
        return fail(errc::bad_header, "config.json: the " + *type + " configuration is not one core accepts (ml::valid)");
    return ok(std::move(c));
}

/// \brief The sampling defaults of `generation_config.json`.
struct generation_config
{
    std::optional<std::uint32_t> bos;           ///< `bos_token_id`.
    std::vector<std::uint32_t> eos;             ///< `eos_token_id`: one or several.
    std::optional<std::uint32_t> pad;           ///< `pad_token_id`.
    std::optional<bool> do_sample;              ///< Whether to sample (else greedy).
    std::optional<double> temperature;          ///< The temperature.
    std::optional<double> top_p;                ///< Nucleus sampling's p.
    std::optional<std::size_t> top_k;           ///< Top-k sampling's k.
    std::optional<double> repetition_penalty;   ///< The repetition penalty.
    std::optional<std::size_t> max_new_tokens;  ///< The longest continuation.
    std::optional<std::size_t> max_length;      ///< The longest sequence, prompt included.
};

/// \brief Reads a `generation_config.json`. \param text The JSON. \return The defaults, or `bad_json`.
[[nodiscard]] inline result<generation_config> parse_generation_config(std::string_view text)
{
    SNICHOLLS_TRY_ASSIGN(const detail::json j, formats::detail::parse_json(text, "generation_config.json"));
    if (!j.is_object())
        return fail(errc::bad_json, "generation_config.json: not an object");
    generation_config g;
    g.bos = detail::token_at(j, "bos_token_id");
    g.pad = detail::token_at(j, "pad_token_id");
    if (const detail::json* e = formats::detail::member(j, "eos_token_id")) {
        if (e->is_array()) {
            for (const auto& t : *e)
                if (const auto v = formats::detail::as_u64(t); v && *v <= 0xffffffffu)
                    g.eos.push_back(static_cast<std::uint32_t>(*v));
        } else if (const auto v = formats::detail::as_u64(*e); v && *v <= 0xffffffffu) {
            g.eos.push_back(static_cast<std::uint32_t>(*v));
        }
    }
    g.do_sample = formats::detail::bool_at(j, "do_sample");
    g.temperature = formats::detail::double_at(j, "temperature");
    g.top_p = formats::detail::double_at(j, "top_p");
    g.top_k = detail::size_at(j, "top_k");
    g.repetition_penalty = formats::detail::double_at(j, "repetition_penalty");
    g.max_new_tokens = detail::size_at(j, "max_new_tokens");
    g.max_length = detail::size_at(j, "max_length");
    return ok(std::move(g));
}
} // namespace snicholls::formats::hf

#endif // snicholls_formats_hf_config_hpp
