//
//  model_config.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `model_config`: a checkpoint's architecture as core's `ml::transformer_config` plus the sizes around it.
/// \ingroup formats
///
/// \overview
/// Hugging Face's `config.json` and GGUF's metadata describe a model in two vocabularies; core describes a block in
/// one, `ml::transformer_config`. A `model_config` is the meeting point: the block, the counts a block does not hold
/// (layers, vocabulary, context, the dense layers before the mixture-of-experts ones), the embedding tie and the
/// special tokens, and `notes`, one line for every field of the source that does not map onto core (a RoPE scaling
/// core lacks, a bias core places differently). `hf::parse_config` and `gguf::read_model_config` fill it;
/// `gguf::write_model_config` writes it back as GGUF keys.
///
/// The values are run-time values: core's `transformer_config` is a literal type, so a configuration known at
/// compile time can still be checked against one read from a file (`same_block(c.block, ml::llama3_8b)`).
///
/// \relations
/// \see snicholls::ml::transformer_config (core), formats::hf::parse_config, formats::gguf::read_model_config
///
/// \impls
/// One struct and an enumeration of the architectures mapped: GPT-2, Llama, Mistral, Qwen 2, Qwen 3, OLMoE,
/// DeepSeek-V2 and DeepSeek-V3.
///
/// \contracts
/// A `model_config` returned by a reader satisfies `ml::valid(block)`.
///
/// \errors
/// None of its own.
///
/// \examples
/// \snippet model_config_demo.cpp model_config
///
/// \perf
/// Not benchmarked: plain data.
///
/// \testing
/// tests/model_config_test.cpp, demos/model_config_demo.cpp.
///
/// \deps
/// Core's `ml/transformer_config.hpp`.
///
/// \gotchas
/// GGUF stores Llama's Q and K permuted for adjacent-pair RoPE, so the same model reads as `rope_kind::half` from
/// Hugging Face and `rope_kind::interleaved` from GGUF; `hf::permute_rope` converts the weights.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_model_config_hpp
#define snicholls_formats_model_config_hpp

#include <snicholls/core/ml/transformer_config.hpp>
#include <snicholls/core/reflect/struct_tuple.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats
{
/// \brief The architectures mapped onto core.
enum class architecture : std::uint8_t
{
    gpt2,         ///< GPT-2: layer norm, GELU (tanh), learned positions, fused QKV.
    llama,        ///< Llama 1 to 3: RMS norm, SwiGLU, RoPE, grouped-query attention.
    mistral,      ///< Mistral: Llama with a sliding window.
    qwen2,        ///< Qwen 2 / 2.5: Llama with biases on Q, K and V.
    qwen3,        ///< Qwen 3: Llama with a per-head RMS norm of Q and K before RoPE.
    olmoe,        ///< OLMoE: a softmax-routed mixture of experts.
    deepseek_v2,  ///< DeepSeek-V2: latent attention, softmax-routed experts.
    deepseek_v3,  ///< DeepSeek-V3: latent attention, sigmoid-routed grouped experts, multi-token prediction.
};

/// \brief Hugging Face's `model_type` of an architecture. \param a The architecture. \return "gpt2", "llama", ...
[[nodiscard]] constexpr std::string_view hf_model_type(architecture a) noexcept
{
    switch (a) {
    case architecture::gpt2: return "gpt2";
    case architecture::llama: return "llama";
    case architecture::mistral: return "mistral";
    case architecture::qwen2: return "qwen2";
    case architecture::qwen3: return "qwen3";
    case architecture::olmoe: return "olmoe";
    case architecture::deepseek_v2: return "deepseek_v2";
    case architecture::deepseek_v3: return "deepseek_v3";
    }
    return "?";
}

/// \brief llama.cpp's GGUF architecture name. \param a The architecture. \return "gpt2", "llama", ... (Mistral is
/// "llama", both DeepSeeks "deepseek2", as llama.cpp writes them).
[[nodiscard]] constexpr std::string_view gguf_architecture(architecture a) noexcept
{
    switch (a) {
    case architecture::gpt2: return "gpt2";
    case architecture::llama: case architecture::mistral: return "llama";
    case architecture::qwen2: return "qwen2";
    case architecture::qwen3: return "qwen3";
    case architecture::olmoe: return "olmoe";
    case architecture::deepseek_v2: case architecture::deepseek_v3: return "deepseek2";
    }
    return "?";
}

/// \brief The architecture of a Hugging Face `model_type`. \param model_type "llama" and so on. \return It, or none.
[[nodiscard]] constexpr std::optional<architecture> architecture_from_hf(std::string_view model_type) noexcept
{
    for (const auto a : {architecture::gpt2, architecture::llama, architecture::mistral, architecture::qwen2, architecture::qwen3,
                         architecture::olmoe, architecture::deepseek_v2, architecture::deepseek_v3})
        if (hf_model_type(a) == model_type)
            return a;
    return std::nullopt;
}

/// \brief Whether two blocks are the same, field by field (core's `tie` over the aggregate). Core's
/// `transformer_config` has no `operator==` yet; this moves there as a defaulted one (proposal in
/// family/proposals/model-formats.md). \param a One. \param b The other. \return Whether every field is equal.
[[nodiscard]] constexpr bool same_block(const ml::transformer_config& a, const ml::transformer_config& b) noexcept
{
    return snicholls::tie(a) == snicholls::tie(b);
}

/// \brief A checkpoint's architecture: core's block and what surrounds it.
struct model_config
{
    architecture arch = architecture::llama;     ///< The architecture.
    ml::transformer_config block;                ///< One block, as core describes it.
    std::size_t layers = 0;                      ///< The blocks.
    std::size_t vocab = 0;                       ///< The tokens of the embedding.
    std::size_t context = 0;                     ///< The trained context (GPT-2: the position table's rows).
    std::size_t leading_dense_layers = 0;        ///< Dense blocks before the mixture-of-experts ones (DeepSeek's `first_k_dense_replace`).
    bool tied_embeddings = false;                ///< Whether the output head is the token embedding.
    std::optional<std::uint32_t> bos;            ///< The beginning-of-sequence token.
    std::optional<std::uint32_t> eos;            ///< The end-of-sequence token.
    std::vector<std::string> notes;              ///< What in the source did not map onto core, one line each.
};
} // namespace snicholls::formats

#endif // snicholls_formats_model_config_hpp
