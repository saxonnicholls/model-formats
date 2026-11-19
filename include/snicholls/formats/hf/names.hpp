//
//  names.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Checkpoint tensor names: Hugging Face's and llama.cpp's to one `tensor_name` (a role, a layer, an expert)
/// and back, and the RoPE permutation llama.cpp's converter applies to Q and K.
/// \ingroup hf
///
/// \overview
/// The same weight is `model.layers.3.self_attn.q_proj.weight` in a Hugging Face Llama, `h.3.attn.c_attn.weight` (Q,
/// K and V fused) in GPT-2, and `blk.3.attn_q.weight` in GGUF. `parse_hf_name` and `parse_gguf_name` turn a name into
/// a `tensor_name`, a `role` (core's parts of a block: q, k, v, o, gate, up, down, the norms, the embeddings, the head,
/// the router, the experts, latent attention's projections) with its layer and expert; `hf_name` and `gguf_name` turn
/// it back. A name no rule matches is none, never guessed.
///
/// Two layout facts go with the names:
///
/// - GPT-2's Hugging Face checkpoints store the projections as `Conv1D`, `[in, out]`; every other format (llm.c,
///   GGUF, PyTorch `Linear`, core) stores `[out, in]`. `hf_transposed` says which: transpose those when converting.
/// - llama.cpp stores Llama's Q and K for adjacent-pair RoPE: its converter permutes the rows of each head from
///   halves to pairs (`LlamaModel.permute`: reshape to `[heads, 2, head_dim / 2, ...]`, swap axes 1 and 2).
///   `permute_rope` applies it and `unpermute_rope` undoes it, over rows of any type (quantised rows included). The
///   Qwen and OLMoE architectures are not permuted (llama.cpp runs them with NeoX RoPE).
///
/// GGUF stacks a layer's experts into one tensor (`blk.N.ffn_up_exps.weight`, `[experts, out, in]`); Hugging Face
/// keeps one per expert. `gguf_name` of an expert's weight is the stacked tensor's name, and conversion stacks them.
///
/// \relations
/// \see formats::model_config, formats::convert::export_llmc_gpt2
///
/// \impls
/// A table of patterns per architecture family.
///
/// \contracts
/// `parse_hf_name(a, hf_name(a, n)) == n` for every name `hf_name` gives; the same for GGUF.
/// `unpermute_rope(permute_rope(x)) == x`.
///
/// \errors
/// None: an unknown name is `std::nullopt`; `permute_rope` asserts its sizes.
///
/// \examples
/// \snippet names_demo.cpp names
///
/// \perf
/// Not benchmarked: a few string comparisons per tensor; the permutation is one copy.
///
/// \testing
/// tests/hf/names_test.cpp (the permutation against numpy's reshape and swapaxes, by hand), demos/hf/names_demo.cpp.
///
/// \deps
/// The standard library.
///
/// \gotchas
/// GPT-2's `attn.bias` and `attn.masked_bias` are causal-mask buffers, not weights: `role::mask_buffer`, which
/// conversion drops.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_hf_names_hpp
#define snicholls_formats_hf_names_hpp

#include <snicholls/formats/model_config.hpp>

#include <array>
#include <cassert>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace snicholls::formats::hf
{
/// \brief What a tensor is, in core's terms.
enum class role : std::uint8_t
{
    token_embedding,    ///< The token table, `[vocab, d_model]`.
    position_embedding, ///< GPT-2's learned positions, `[context, d_model]`.
    attn_norm,          ///< The norm before attention (weight or bias).
    q,                  ///< The query projection.
    k,                  ///< The key projection.
    v,                  ///< The value projection.
    qkv,                ///< Q, K and V fused, Q first (GPT-2).
    o,                  ///< The output projection.
    q_norm,             ///< Qwen 3's per-head query norm.
    k_norm,             ///< Qwen 3's per-head key norm.
    mlp_norm,           ///< The norm before the MLP.
    gate,               ///< The gated MLP's gate.
    up,                 ///< The MLP's up projection (GPT-2's `c_fc`).
    down,               ///< The MLP's down projection.
    router,             ///< The mixture of experts' router.
    router_bias,        ///< DeepSeek-V3's selection bias (`e_score_correction_bias`).
    expert_gate,        ///< A routed expert's gate.
    expert_up,          ///< A routed expert's up.
    expert_down,        ///< A routed expert's down.
    shared_gate,        ///< The shared experts' gate.
    shared_up,          ///< The shared experts' up.
    shared_down,        ///< The shared experts' down.
    q_a,                ///< MLA: the query's down projection.
    q_a_norm,           ///< MLA: its norm.
    q_b,                ///< MLA: the query's up projection.
    kv_a,               ///< MLA: the key-value latent and the decoupled RoPE key.
    kv_a_norm,          ///< MLA: the latent's norm.
    kv_b,               ///< MLA: the key and value up projection.
    final_norm,         ///< The norm before the head.
    lm_head,            ///< The output head, `[vocab, d_model]`.
    mask_buffer,        ///< GPT-2's causal-mask buffers: not weights.
};

/// \brief A tensor's place in a model.
struct tensor_name
{
    role what = role::q;                   ///< The role.
    std::optional<std::size_t> layer;      ///< The block, for per-block tensors.
    std::optional<std::size_t> expert;     ///< The expert, for a routed expert's weights (Hugging Face).
    bool bias = false;                     ///< The bias (or a norm's shift) rather than the weight.
    /// \brief Equal when every field is. \param a One. \param b The other. \return Whether equal.
    friend bool operator==(const tensor_name& a, const tensor_name& b) = default;
};

namespace detail
{
    struct pattern
    {
        role what;
        std::string_view name;   // per-block names after "<prefix><layer>."; others whole
    };
    // GPT-2 (prefix "h."), with or without "transformer."
    inline constexpr std::array<pattern, 6> gpt2_block{{{role::attn_norm, "ln_1"},
                                                        {role::qkv, "attn.c_attn"},
                                                        {role::o, "attn.c_proj"},
                                                        {role::mlp_norm, "ln_2"},
                                                        {role::up, "mlp.c_fc"},
                                                        {role::down, "mlp.c_proj"}}};
    // Llama family and DeepSeek (prefix "model.layers.")
    inline constexpr std::array<pattern, 24> llama_block{{{role::attn_norm, "input_layernorm"},
                                                          {role::q, "self_attn.q_proj"},
                                                          {role::k, "self_attn.k_proj"},
                                                          {role::v, "self_attn.v_proj"},
                                                          {role::o, "self_attn.o_proj"},
                                                          {role::q_norm, "self_attn.q_norm"},
                                                          {role::k_norm, "self_attn.k_norm"},
                                                          {role::mlp_norm, "post_attention_layernorm"},
                                                          {role::gate, "mlp.gate_proj"},
                                                          {role::up, "mlp.up_proj"},
                                                          {role::down, "mlp.down_proj"},
                                                          {role::router, "mlp.gate"},
                                                          {role::shared_gate, "mlp.shared_experts.gate_proj"},
                                                          {role::shared_up, "mlp.shared_experts.up_proj"},
                                                          {role::shared_down, "mlp.shared_experts.down_proj"},
                                                          {role::q_a, "self_attn.q_a_proj"},
                                                          {role::q_a_norm, "self_attn.q_a_layernorm"},
                                                          {role::q_b, "self_attn.q_b_proj"},
                                                          {role::kv_a, "self_attn.kv_a_proj_with_mqa"},
                                                          {role::kv_a_norm, "self_attn.kv_a_layernorm"},
                                                          {role::kv_b, "self_attn.kv_b_proj"},
                                                          {role::expert_gate, "gate_proj"},   // after "mlp.experts.<e>."
                                                          {role::expert_up, "up_proj"},
                                                          {role::expert_down, "down_proj"}}};
    // GGUF (prefix "blk.")
    inline constexpr std::array<pattern, 26> gguf_block{{{role::attn_norm, "attn_norm"},
                                                         {role::q, "attn_q"},
                                                         {role::k, "attn_k"},
                                                         {role::v, "attn_v"},
                                                         {role::qkv, "attn_qkv"},
                                                         {role::o, "attn_output"},
                                                         {role::q_norm, "attn_q_norm"},
                                                         {role::k_norm, "attn_k_norm"},
                                                         {role::mlp_norm, "ffn_norm"},
                                                         {role::gate, "ffn_gate"},
                                                         {role::up, "ffn_up"},
                                                         {role::down, "ffn_down"},
                                                         {role::router, "ffn_gate_inp"},
                                                         {role::router_bias, "exp_probs_b"},
                                                         {role::expert_gate, "ffn_gate_exps"},
                                                         {role::expert_up, "ffn_up_exps"},
                                                         {role::expert_down, "ffn_down_exps"},
                                                         {role::shared_gate, "ffn_gate_shexp"},
                                                         {role::shared_up, "ffn_up_shexp"},
                                                         {role::shared_down, "ffn_down_shexp"},
                                                         {role::q_a, "attn_q_a"},
                                                         {role::q_a_norm, "attn_q_a_norm"},
                                                         {role::q_b, "attn_q_b"},
                                                         {role::kv_a, "attn_kv_a_mqa"},
                                                         {role::kv_a_norm, "attn_kv_a_norm"},
                                                         {role::kv_b, "attn_kv_b"}}};

    // "<digits>." at the front of s: the number, and s advanced past it
    inline std::optional<std::size_t> take_index(std::string_view& s)
    {
        std::size_t n = 0;
        const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), n);
        if (ec != std::errc{} || p == s.data() || p == s.data() + s.size() || *p != '.')
            return std::nullopt;
        s.remove_prefix(static_cast<std::size_t>(p - s.data()) + 1);
        return n;
    }
    // "<base>.weight" or "<base>.bias": the base and which
    inline std::optional<std::pair<std::string_view, bool>> split_suffix(std::string_view s)
    {
        if (s.ends_with(".weight"))
            return std::pair{s.substr(0, s.size() - 7), false};
        if (s.ends_with(".bias"))
            return std::pair{s.substr(0, s.size() - 5), true};
        return std::nullopt;
    }
    template <std::size_t N>
    std::optional<role> find_role(const std::array<pattern, N>& table, std::string_view base)
    {
        for (const auto& p : table)
            if (!p.name.empty() && p.name == base)
                return p.what;
        return std::nullopt;
    }
    template <std::size_t N>
    std::string_view role_name(const std::array<pattern, N>& table, role r)
    {
        for (const auto& p : table)
            if (p.what == r && !p.name.empty())
                return p.name;
        return {};
    }
} // namespace detail

/// \brief A Hugging Face tensor name to its place. \param a The architecture. \param name The name.
/// \return The place, or none if no rule matches.
[[nodiscard]] inline std::optional<tensor_name> parse_hf_name(architecture a, std::string_view name)
{
    tensor_name t;
    if (a == architecture::gpt2) {
        if (name.starts_with("transformer."))
            name.remove_prefix(12);
        if (name == "attn.bias" || name.ends_with(".attn.bias") || name.ends_with(".attn.masked_bias")) {
            std::string_view rest = name;
            if (rest.starts_with("h.")) {
                rest.remove_prefix(2);
                t.layer = detail::take_index(rest);
            }
            t.what = role::mask_buffer;
            t.bias = !name.ends_with("masked_bias");
            return t;
        }
        const auto s = detail::split_suffix(name);
        if (!s)
            return std::nullopt;
        t.bias = s->second;
        std::string_view base = s->first;
        if (base == "wte" || base == "lm_head" || base == "wpe" || base == "ln_f") {
            t.what = base == "wte" ? role::token_embedding : base == "lm_head" ? role::lm_head : base == "wpe" ? role::position_embedding : role::final_norm;
            return t;
        }
        if (!base.starts_with("h."))
            return std::nullopt;
        base.remove_prefix(2);
        t.layer = detail::take_index(base);
        const auto r = t.layer ? detail::find_role(detail::gpt2_block, base) : std::nullopt;
        if (!r)
            return std::nullopt;
        t.what = *r;
        return t;
    }
    const auto s = detail::split_suffix(name);
    if (!s) {
        if (name.ends_with(".mlp.gate.e_score_correction_bias") && name.starts_with("model.layers.")) {
            std::string_view rest = name.substr(13);
            t.layer = detail::take_index(rest);
            t.what = role::router_bias;
            t.bias = true;
            return t.layer ? std::optional<tensor_name>(t) : std::nullopt;
        }
        return std::nullopt;
    }
    t.bias = s->second;
    std::string_view base = s->first;
    if (base == "model.embed_tokens" || base == "lm_head" || base == "model.norm") {
        t.what = base == "model.embed_tokens" ? role::token_embedding : base == "lm_head" ? role::lm_head : role::final_norm;
        return t;
    }
    if (!base.starts_with("model.layers."))
        return std::nullopt;
    base.remove_prefix(13);
    t.layer = detail::take_index(base);
    if (!t.layer)
        return std::nullopt;
    if (base.starts_with("mlp.experts.")) {
        base.remove_prefix(12);
        t.expert = detail::take_index(base);
        if (!t.expert)
            return std::nullopt;
        if (base == "gate_proj" || base == "up_proj" || base == "down_proj") {
            t.what = base == "gate_proj" ? role::expert_gate : base == "up_proj" ? role::expert_up : role::expert_down;
            return t;
        }
        return std::nullopt;
    }
    const auto r = detail::find_role(detail::llama_block, base);
    if (!r || *r == role::expert_gate || *r == role::expert_up || *r == role::expert_down)
        return std::nullopt;
    t.what = *r;
    return t;
}

/// \brief A place to its Hugging Face name. \param a The architecture. \param t The place.
/// \return The name, or empty if the architecture has no such tensor.
[[nodiscard]] inline std::string hf_name(architecture a, const tensor_name& t)
{
    const std::string suffix = t.bias ? ".bias" : ".weight";
    if (a == architecture::gpt2) {
        switch (t.what) {
        case role::token_embedding: return "wte" + suffix;
        case role::position_embedding: return "wpe" + suffix;
        case role::final_norm: return "ln_f" + suffix;
        case role::lm_head: return "lm_head" + suffix;
        case role::mask_buffer: return t.layer ? "h." + std::to_string(*t.layer) + (t.bias ? ".attn.bias" : ".attn.masked_bias") : std::string();
        default: break;
        }
        const std::string_view n = detail::role_name(detail::gpt2_block, t.what);
        if (n.empty() || !t.layer)
            return {};
        return "h." + std::to_string(*t.layer) + "." + std::string(n) + suffix;
    }
    switch (t.what) {
    case role::token_embedding: return "model.embed_tokens" + suffix;
    case role::final_norm: return "model.norm" + suffix;
    case role::lm_head: return "lm_head" + suffix;
    case role::position_embedding: case role::qkv: case role::mask_buffer: return {};
    default: break;
    }
    if (!t.layer)
        return {};
    const std::string layer = "model.layers." + std::to_string(*t.layer) + ".";
    if (t.what == role::router_bias)
        return layer + "mlp.gate.e_score_correction_bias";
    if (t.what == role::expert_gate || t.what == role::expert_up || t.what == role::expert_down) {
        if (!t.expert)
            return {};
        return layer + "mlp.experts." + std::to_string(*t.expert) + "." + std::string(detail::role_name(detail::llama_block, t.what)) + suffix;
    }
    const std::string_view n = detail::role_name(detail::llama_block, t.what);
    return n.empty() ? std::string() : layer + std::string(n) + suffix;
}

/// \brief A GGUF (llama.cpp) tensor name to its place. \param name The name. \return The place, or none.
[[nodiscard]] inline std::optional<tensor_name> parse_gguf_name(std::string_view name)
{
    const auto s = detail::split_suffix(name);
    if (!s)
        return std::nullopt;
    tensor_name t;
    t.bias = s->second;
    std::string_view base = s->first;
    if (base == "token_embd" || base == "position_embd" || base == "output_norm" || base == "output") {
        t.what = base == "token_embd" ? role::token_embedding : base == "position_embd" ? role::position_embedding
                                                              : base == "output_norm"   ? role::final_norm
                                                                                        : role::lm_head;
        return t;
    }
    if (!base.starts_with("blk."))
        return std::nullopt;
    base.remove_prefix(4);
    t.layer = detail::take_index(base);
    const auto r = t.layer ? detail::find_role(detail::gguf_block, base) : std::nullopt;
    if (!r)
        return std::nullopt;
    t.what = *r;
    return t;
}

/// \brief A place to its GGUF name (an expert's weight to its layer's stacked tensor). \param t The place.
/// \return The name, or empty if llama.cpp has no such tensor.
[[nodiscard]] inline std::string gguf_name(const tensor_name& t)
{
    const std::string suffix = t.bias ? ".bias" : ".weight";
    switch (t.what) {
    case role::token_embedding: return "token_embd" + suffix;
    case role::position_embedding: return "position_embd" + suffix;
    case role::final_norm: return "output_norm" + suffix;
    case role::lm_head: return "output" + suffix;
    case role::mask_buffer: return {};
    default: break;
    }
    const std::string_view n = detail::role_name(detail::gguf_block, t.what);
    if (n.empty() || !t.layer)
        return {};
    return "blk." + std::to_string(*t.layer) + "." + std::string(n) + suffix;
}

/// \brief Whether Hugging Face stores this weight transposed, `[in, out]` (GPT-2's `Conv1D`).
/// \param a The architecture. \param t The place. \return True for GPT-2's qkv, o, up and down weights.
[[nodiscard]] constexpr bool hf_transposed(architecture a, const tensor_name& t) noexcept
{
    return a == architecture::gpt2 && !t.bias && (t.what == role::qkv || t.what == role::o || t.what == role::up || t.what == role::down);
}

/// \brief Whether llama.cpp's converter permutes this tensor for adjacent-pair RoPE.
/// \param a The architecture. \param t The place. \return True for Llama's and Mistral's Q and K (weights and biases).
[[nodiscard]] constexpr bool gguf_permutes(architecture a, const tensor_name& t) noexcept
{
    return (a == architecture::llama || a == architecture::mistral) && (t.what == role::q || t.what == role::k);
}

/// \brief llama.cpp's Q/K permutation: in each head, rows from halves `[0..d/2) [d/2..d)` to pairs.
/// \param in The tensor's bytes, `heads * head_dim` rows. \param out As many bytes. \param heads The heads (KV heads for K).
/// \param row_bytes The bytes of a row (one element for a bias).
/// Complexity: one copy of every row.
inline void permute_rope(std::span<const std::byte> in, std::span<std::byte> out, std::size_t heads, std::size_t row_bytes) noexcept
{
    assert(in.size() == out.size() && heads > 0 && row_bytes > 0 && in.size() % (heads * row_bytes * 2) == 0);
    const std::size_t rows = in.size() / row_bytes, head_rows = rows / heads, half = head_rows / 2;
    for (std::size_t h = 0; h < heads; ++h)
        for (std::size_t i = 0; i < half; ++i)
            for (std::size_t p = 0; p < 2; ++p)
                std::memcpy(out.data() + (h * head_rows + 2 * i + p) * row_bytes, in.data() + (h * head_rows + p * half + i) * row_bytes, row_bytes);
}

/// \brief The inverse of `permute_rope`: pairs back to halves. \param in The permuted bytes. \param out As many bytes.
/// \param heads The heads. \param row_bytes The bytes of a row.
inline void unpermute_rope(std::span<const std::byte> in, std::span<std::byte> out, std::size_t heads, std::size_t row_bytes) noexcept
{
    assert(in.size() == out.size() && heads > 0 && row_bytes > 0 && in.size() % (heads * row_bytes * 2) == 0);
    const std::size_t rows = in.size() / row_bytes, head_rows = rows / heads, half = head_rows / 2;
    for (std::size_t h = 0; h < heads; ++h)
        for (std::size_t i = 0; i < half; ++i)
            for (std::size_t p = 0; p < 2; ++p)
                std::memcpy(out.data() + (h * head_rows + p * half + i) * row_bytes, in.data() + (h * head_rows + 2 * i + p) * row_bytes, row_bytes);
}
} // namespace snicholls::formats::hf

#endif // snicholls_formats_hf_names_hpp
