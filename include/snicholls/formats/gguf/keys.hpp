//
//  keys.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief GGUF's standard metadata keys (`general.*`, `<arch>.*`, `tokenizer.ggml.*`) and their enumerations.
/// \ingroup gguf
///
/// \overview
/// The key names llama.cpp reads, from gguf-py's `constants.py` (llama.cpp, MIT), so a key is spelt once.
/// `general::*` and `tokenizer::*` are whole keys; the per-architecture keys are suffixes, joined to the architecture by
/// `arch_key("llama", llm::context_length)` = `"llama.context_length"`. `token_type` is `tokenizer.ggml.token_type`'s
/// values and `file_type` the `general.file_type` codes used here.
///
/// \relations
/// \see formats::gguf::writer, formats::convert::model_config
///
/// \impls
/// `constexpr std::string_view` constants; one function.
///
/// \contracts
/// Compile-time data.
///
/// \errors
/// None.
///
/// \examples
/// \snippet keys_demo.cpp keys
///
/// \perf
/// Not benchmarked: constants.
///
/// \testing
/// tests/gguf/keys_test.cpp, demos/gguf/keys_demo.cpp.
///
/// \deps
/// `<string>`.
///
/// \gotchas
/// `tokenizer.ggml.seperator_token_id` is spelt as llama.cpp spells it.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_gguf_keys_hpp
#define snicholls_formats_gguf_keys_hpp

#include <cstdint>
#include <string>
#include <string_view>

namespace snicholls::formats::gguf
{
/// \brief `"<arch>.<suffix>"`. \param arch The architecture. \param suffix A key of `llm`, `attention` or `rope`.
/// \return The key.
[[nodiscard]] inline std::string arch_key(std::string_view arch, std::string_view suffix)
{
    std::string k(arch);
    k += '.';
    k += suffix;
    return k;
}

/// \brief Keys every file may carry.
namespace general
{
    inline constexpr std::string_view architecture = "general.architecture";                   ///< The architecture, e.g. "llama".
    inline constexpr std::string_view quantization_version = "general.quantization_version";   ///< The quantisation version (2).
    inline constexpr std::string_view alignment = "general.alignment";                         ///< The data alignment, u32.
    inline constexpr std::string_view file_type = "general.file_type";                         ///< The `file_type`, u32.
    inline constexpr std::string_view name = "general.name";                                   ///< The model's name.
    inline constexpr std::string_view author = "general.author";                               ///< Its author.
    inline constexpr std::string_view version = "general.version";                             ///< Its version.
    inline constexpr std::string_view organization = "general.organization";                   ///< Its organisation.
    inline constexpr std::string_view basename = "general.basename";                           ///< The base name.
    inline constexpr std::string_view finetune = "general.finetune";                           ///< The fine-tune.
    inline constexpr std::string_view description = "general.description";                     ///< A description.
    inline constexpr std::string_view size_label = "general.size_label";                       ///< "7B" and so on.
    inline constexpr std::string_view license = "general.license";                             ///< The licence.
    inline constexpr std::string_view url = "general.url";                                     ///< A web page.
    inline constexpr std::string_view source_url = "general.source.url";                       ///< Where it came from.
    inline constexpr std::string_view source_repo_url = "general.source.repo_url";             ///< The source repository.
    inline constexpr std::string_view tags = "general.tags";                                   ///< Tags, an array of strings.
    inline constexpr std::string_view languages = "general.languages";                         ///< Languages, an array of strings.
} // namespace general

/// \brief Per-architecture model keys, suffixes for `arch_key`.
namespace llm
{
    inline constexpr std::string_view vocab_size = "vocab_size";                                 ///< Tokens.
    inline constexpr std::string_view context_length = "context_length";                         ///< The trained context.
    inline constexpr std::string_view embedding_length = "embedding_length";                     ///< d_model.
    inline constexpr std::string_view block_count = "block_count";                               ///< Layers.
    inline constexpr std::string_view leading_dense_block_count = "leading_dense_block_count";   ///< Dense layers before the MoE ones.
    inline constexpr std::string_view feed_forward_length = "feed_forward_length";               ///< d_ff.
    inline constexpr std::string_view expert_feed_forward_length = "expert_feed_forward_length";   ///< An expert's d_ff.
    inline constexpr std::string_view expert_shared_feed_forward_length = "expert_shared_feed_forward_length";   ///< The shared experts' d_ff.
    inline constexpr std::string_view expert_count = "expert_count";                             ///< Routed experts.
    inline constexpr std::string_view expert_used_count = "expert_used_count";                   ///< Experts per token.
    inline constexpr std::string_view expert_shared_count = "expert_shared_count";               ///< Shared experts.
    inline constexpr std::string_view expert_group_count = "expert_group_count";                 ///< Expert groups.
    inline constexpr std::string_view expert_group_used_count = "expert_group_used_count";       ///< Groups per token.
    inline constexpr std::string_view expert_weights_scale = "expert_weights_scale";             ///< The routed weights' factor.
    inline constexpr std::string_view expert_weights_norm = "expert_weights_norm";               ///< Whether the routed weights are renormalised.
    inline constexpr std::string_view expert_gating_func = "expert_gating_func";                 ///< 1 softmax, 2 sigmoid.
    inline constexpr std::string_view nextn_predict_layers = "nextn_predict_layers";             ///< Multi-token prediction layers.
    inline constexpr std::string_view attn_logit_softcapping = "attn_logit_softcapping";         ///< The score soft cap.
} // namespace llm

/// \brief Per-architecture attention keys, suffixes for `arch_key`.
namespace attention
{
    inline constexpr std::string_view head_count = "attention.head_count";                         ///< Query heads.
    inline constexpr std::string_view head_count_kv = "attention.head_count_kv";                   ///< Key-value heads.
    inline constexpr std::string_view key_length = "attention.key_length";                         ///< A key head's channels.
    inline constexpr std::string_view value_length = "attention.value_length";                     ///< A value head's channels.
    inline constexpr std::string_view layer_norm_epsilon = "attention.layer_norm_epsilon";         ///< Layer norm's epsilon.
    inline constexpr std::string_view layer_norm_rms_epsilon = "attention.layer_norm_rms_epsilon"; ///< RMS norm's epsilon.
    inline constexpr std::string_view causal = "attention.causal";                                 ///< Whether causal.
    inline constexpr std::string_view q_lora_rank = "attention.q_lora_rank";                       ///< MLA's query rank.
    inline constexpr std::string_view kv_lora_rank = "attention.kv_lora_rank";                     ///< MLA's key-value rank.
    inline constexpr std::string_view key_length_mla = "attention.key_length_mla";                 ///< MLA's key head width.
    inline constexpr std::string_view value_length_mla = "attention.value_length_mla";             ///< MLA's value head width.
    inline constexpr std::string_view sliding_window = "attention.sliding_window";                 ///< The window.
} // namespace attention

/// \brief Per-architecture RoPE keys, suffixes for `arch_key`.
namespace rope
{
    inline constexpr std::string_view dimension_count = "rope.dimension_count";                         ///< Rotated channels per head.
    inline constexpr std::string_view freq_base = "rope.freq_base";                                     ///< The base.
    inline constexpr std::string_view scaling_type = "rope.scaling.type";                               ///< "none", "linear", "yarn".
    inline constexpr std::string_view scaling_factor = "rope.scaling.factor";                           ///< The factor.
    inline constexpr std::string_view scaling_original_context_length = "rope.scaling.original_context_length";   ///< YaRN's L.
    inline constexpr std::string_view scaling_yarn_log_multiplier = "rope.scaling.yarn_log_multiplier"; ///< DeepSeek's mscale coefficient.
    inline constexpr std::string_view scaling_yarn_beta_fast = "rope.scaling.yarn_beta_fast";           ///< YaRN's beta.
    inline constexpr std::string_view scaling_yarn_beta_slow = "rope.scaling.yarn_beta_slow";           ///< YaRN's alpha.
} // namespace rope

/// \brief Tokenizer keys.
namespace tokenizer
{
    inline constexpr std::string_view model = "tokenizer.ggml.model";                     ///< "gpt2", "llama", ...
    inline constexpr std::string_view pre = "tokenizer.ggml.pre";                         ///< The pre-tokenizer, "gpt-2", "llama-bpe", ...
    inline constexpr std::string_view tokens = "tokenizer.ggml.tokens";                   ///< The tokens, an array of strings.
    inline constexpr std::string_view token_type = "tokenizer.ggml.token_type";           ///< Each token's `token_type`, i32.
    inline constexpr std::string_view scores = "tokenizer.ggml.scores";                   ///< Scores, f32 (SentencePiece).
    inline constexpr std::string_view merges = "tokenizer.ggml.merges";                   ///< BPE merges, "a b" strings.
    inline constexpr std::string_view bos_token_id = "tokenizer.ggml.bos_token_id";       ///< Beginning of sequence.
    inline constexpr std::string_view eos_token_id = "tokenizer.ggml.eos_token_id";       ///< End of sequence.
    inline constexpr std::string_view eot_token_id = "tokenizer.ggml.eot_token_id";       ///< End of turn.
    inline constexpr std::string_view unknown_token_id = "tokenizer.ggml.unknown_token_id";   ///< Unknown.
    inline constexpr std::string_view separator_token_id = "tokenizer.ggml.seperator_token_id";   ///< Separator (llama.cpp's spelling).
    inline constexpr std::string_view padding_token_id = "tokenizer.ggml.padding_token_id";   ///< Padding.
    inline constexpr std::string_view add_bos_token = "tokenizer.ggml.add_bos_token";     ///< Whether to prepend BOS.
    inline constexpr std::string_view add_eos_token = "tokenizer.ggml.add_eos_token";     ///< Whether to append EOS.
    inline constexpr std::string_view chat_template = "tokenizer.chat_template";          ///< The Jinja chat template.
    inline constexpr std::string_view huggingface_json = "tokenizer.huggingface.json";    ///< A whole tokenizer.json.
} // namespace tokenizer

/// \brief `tokenizer.ggml.token_type` values.
enum class token_type : std::int32_t
{
    normal = 1,         ///< A text token.
    unknown = 2,        ///< The unknown token.
    control = 3,        ///< A special token (BOS, EOS, ...).
    user_defined = 4,   ///< An added token.
    unused = 5,         ///< Reserved.
    byte = 6,           ///< A byte fallback token.
};

/// \brief `general.file_type` values (llama.cpp's `llama_ftype`) for the files written here.
enum class file_type : std::uint32_t
{
    all_f32 = 0,       ///< Every tensor F32.
    mostly_f16 = 1,    ///< Matrices F16.
    mostly_q4_0 = 2,   ///< Matrices Q4_0.
    mostly_q8_0 = 7,   ///< Matrices Q8_0.
    mostly_bf16 = 32,  ///< Matrices BF16.
};
} // namespace snicholls::formats::gguf

#endif // snicholls_formats_gguf_keys_hpp
