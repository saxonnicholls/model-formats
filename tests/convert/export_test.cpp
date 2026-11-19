//
//  export_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/convert/export.hpp>
#include <snicholls/formats/hf/config.hpp>

#include <gtest/gtest.h>

#include <numeric>

using namespace snicholls::formats;
using namespace snicholls::formats::convert;
namespace ml = snicholls::ml;

namespace
{
constexpr llmc::gpt2_shape tiny{.max_seq = 8, .vocab = 10, .layers = 2, .heads = 2, .channels = 4, .padded_vocab = 16};

std::vector<std::byte> tiny_checkpoint()
{
    std::vector<float> p(llmc::parameter_count(tiny));
    std::iota(p.begin(), p.end(), 0.0f);
    return *llmc::write_checkpoint(tiny, dtype::f32, std::as_bytes(std::span(p)));
}

vocabulary tiny_vocab()
{
    vocabulary v;
    for (const char* t : {"a", "b", "c", "d", "e", "f", "g", "h", "ab", "<|endoftext|>"})
        v.tokens.push_back(t);
    v.types.assign(10, gguf::token_type::normal);
    v.types[9] = gguf::token_type::control;
    v.merges = {"a b"};
    v.eos = v.bos = 9;
    return v;
}

// Random-looking tensors for a Hugging Face model, named as the architecture names them
struct hf_model
{
    std::deque<tensor> owned;
    std::vector<tensor_view> views;
    void add(std::string name, tensor_shape shape)
    {
        tensor t{std::move(name), dtype::f32, shape, std::vector<std::byte>(static_cast<std::size_t>(*shape.elements()) * 4)};
        for (std::size_t i = 0; i < t.data.size(); ++i)
            t.data[i] = static_cast<std::byte>((i * 2654435761u + owned.size() * 97) >> 7);
        owned.push_back(std::move(t));
        views.push_back(owned.back().view());
    }
};

void expect_same(std::span<const tensor_view> got, std::span<const tensor_view> want)
{
    ASSERT_EQ(got.size(), want.size());
    for (const auto& w : want) {
        const tensor_view* g = nullptr;
        for (const auto& x : got)
            if (x.name == w.name)
                g = &x;
        ASSERT_NE(g, nullptr) << w.name;
        EXPECT_EQ(g->shape, w.shape) << w.name;
        EXPECT_TRUE(std::equal(g->data.begin(), g->data.end(), w.data.begin(), w.data.end())) << w.name;
    }
}
} // namespace

TEST(ExportLlmc, ToHuggingFaceGpt2)
{
    const auto bytes = tiny_checkpoint();
    const auto c = llmc::parse_checkpoint(bytes);
    const auto hf = llmc_to_hf(*c);
    ASSERT_TRUE(hf.is_ok()) << hf.unwrap_err_unchecked().what();
    EXPECT_EQ(hf->tensors.size(), 2u + 12u * 2u + 2u);
    const tensor_view* qkv = nullptr;
    for (const auto& t : hf->tensors)
        if (t.name == "h.1.attn.c_attn.weight")
            qkv = &t;
    ASSERT_NE(qkv, nullptr);
    EXPECT_EQ(qkv->shape, (tensor_shape{4, 12}));   // Conv1D: [in, out]
    // element [in i, out o] is llm.c's qkvw[layer 1][o][i]
    const auto* llm = reinterpret_cast<const float*>(c->tensors[4].data.data()) + 12 * 4;
    const auto* conv = reinterpret_cast<const float*>(qkv->data.data());
    EXPECT_EQ(conv[2 * 12 + 7], llm[7 * 4 + 2]);
    EXPECT_EQ(hf->tensors[0].shape, (tensor_shape{10, 4}));   // padding rows cut
    const auto cfg = hf::parse_config(hf_gpt2_config(tiny));
    ASSERT_TRUE(cfg.is_ok()) << cfg.unwrap_err_unchecked().what();
    EXPECT_EQ(cfg->layers, 2u);
    EXPECT_EQ(cfg->vocab, 10u);
    // and through GGUF and back to the same Hugging Face tensors
    const auto g = hf_to_gguf(*cfg, hf->tensors, nullptr, {}, "tiny");
    ASSERT_TRUE(g.is_ok()) << g.unwrap_err_unchecked().what();
    const auto gb = *g->writer.write();
    const auto gf = gguf::parse(gb);
    ASSERT_TRUE(gf.is_ok());
    const auto back = gguf_to_hf(*gf, *cfg);
    ASSERT_TRUE(back.is_ok()) << back.unwrap_err_unchecked().what();
    expect_same(back->tensors, hf->tensors);
}

TEST(ExportLlmc, ToGgufMatchesTheHuggingFacePath)
{
    const auto bytes = tiny_checkpoint();
    const auto c = llmc::parse_checkpoint(bytes);
    const auto v = tiny_vocab();
    const auto direct = llmc_to_gguf(*c, &v, "tiny");
    ASSERT_TRUE(direct.is_ok()) << direct.unwrap_err_unchecked().what();
    const auto db = *direct->writer.write();
    const auto df = gguf::parse(db);
    ASSERT_TRUE(df.is_ok());
    EXPECT_EQ(df->architecture(), "gpt2");
    EXPECT_EQ(df->find_tensor("token_embd.weight")->shape, (tensor_shape{10, 4}));
    EXPECT_EQ(df->find_tensor("position_embd.weight")->shape, (tensor_shape{8, 4}));
    EXPECT_EQ(df->find_tensor("blk.1.ffn_up.weight")->ne(), (tensor_shape{4, 16}));   // llama.cpp's {n_embd, n_ff}
    EXPECT_EQ(df->find("tokenizer.ggml.pre")->as_string(), "gpt-2");
    const auto m = gguf::read_model_config(*df);
    ASSERT_TRUE(m.is_ok()) << m.unwrap_err_unchecked().what();
    EXPECT_EQ(m->layers, 2u);
    EXPECT_TRUE(m->tied_embeddings);
    // the HF path writes the same tensors
    const auto hf = llmc_to_hf(*c);
    const auto cfg = hf::parse_config(hf_gpt2_config(tiny));
    const auto viahf = hf_to_gguf(*cfg, hf->tensors, &v, {}, "tiny");
    const auto vb = *viahf->writer.write();
    const auto vf = gguf::parse(vb);
    for (const auto& t : df->tensors) {
        const auto* u = vf->find_tensor(t.name);
        ASSERT_NE(u, nullptr) << t.name;
        EXPECT_TRUE(std::equal(t.data.begin(), t.data.end(), u->data.begin(), u->data.end())) << t.name;
    }
    vocabulary wrong = v;
    wrong.tokens.pop_back();
    EXPECT_EQ(llmc_to_gguf(*c, &wrong, "x").unwrap_err_unchecked().code, errc::bad_shape);
}

TEST(ExportHf, LlamaPermutesQueriesAndKeysAndComesBack)
{
    model_config c;
    c.arch = architecture::llama;
    c.block = ml::llama3_8b;
    c.block.d_model = 8;
    c.block.n_heads = 2;
    c.block.n_kv_heads = 1;
    c.block.head_dim = 4;
    c.block.d_ff = 16;
    c.layers = 1;
    c.vocab = 5;
    c.context = 16;
    hf_model m;
    m.add("model.embed_tokens.weight", {5, 8});
    m.add("model.layers.0.input_layernorm.weight", {8});
    m.add("model.layers.0.self_attn.q_proj.weight", {8, 8});
    m.add("model.layers.0.self_attn.k_proj.weight", {4, 8});
    m.add("model.layers.0.self_attn.v_proj.weight", {4, 8});
    m.add("model.layers.0.self_attn.o_proj.weight", {8, 8});
    m.add("model.layers.0.post_attention_layernorm.weight", {8});
    m.add("model.layers.0.mlp.gate_proj.weight", {16, 8});
    m.add("model.layers.0.mlp.up_proj.weight", {16, 8});
    m.add("model.layers.0.mlp.down_proj.weight", {8, 16});
    m.add("model.norm.weight", {8});
    m.add("lm_head.weight", {5, 8});
    const auto g = hf_to_gguf(c, m.views, nullptr, {}, "tiny-llama");
    ASSERT_TRUE(g.is_ok()) << g.unwrap_err_unchecked().what();
    const auto bytes = *g->writer.write();
    const auto f = gguf::parse(bytes);
    ASSERT_TRUE(f.is_ok());
    const auto* q = f->find_tensor("blk.0.attn_q.weight");
    ASSERT_NE(q, nullptr);
    EXPECT_FALSE(std::equal(q->data.begin(), q->data.end(), m.views[2].data.begin()));   // permuted
    // row 1 of the permuted head is row 2 of the original (halves to pairs: 0 2 1 3)
    EXPECT_TRUE(std::equal(q->data.begin() + 32, q->data.begin() + 64, m.views[2].data.begin() + 64));
    const auto cfg = gguf::read_model_config(*f);
    ASSERT_TRUE(cfg.is_ok());
    EXPECT_EQ(cfg->block.rope, ml::rope_kind::interleaved);
    const auto back = gguf_to_hf(*f, c);
    ASSERT_TRUE(back.is_ok()) << back.unwrap_err_unchecked().what();
    expect_same(back->tensors, m.views);
}

TEST(ExportHf, OlmoeStacksExperts)
{
    model_config c;
    c.arch = architecture::olmoe;
    c.block = ml::tiny_moe;
    c.block.shared_experts = 0;
    c.block.experts = 2;
    c.block.top_k = 1;
    c.layers = 1;
    hf_model m;
    m.add("model.layers.0.mlp.gate.weight", {2, 8});
    for (int e = 0; e < 2; ++e) {
        m.add("model.layers.0.mlp.experts." + std::to_string(e) + ".gate_proj.weight", {16, 8});
        m.add("model.layers.0.mlp.experts." + std::to_string(e) + ".up_proj.weight", {16, 8});
        m.add("model.layers.0.mlp.experts." + std::to_string(e) + ".down_proj.weight", {8, 16});
    }
    const auto g = hf_to_gguf(c, m.views, nullptr, {}, "tiny-moe");
    ASSERT_TRUE(g.is_ok()) << g.unwrap_err_unchecked().what();
    const auto bytes = *g->writer.write();
    const auto f = gguf::parse(bytes);
    const auto* up = f->find_tensor("blk.0.ffn_up_exps.weight");
    ASSERT_NE(up, nullptr);
    EXPECT_EQ(up->ne(), (tensor_shape{8, 16, 2}));   // llama.cpp's {n_embd, n_ff_exp, n_expert}
    const auto back = gguf_to_hf(*f, c);
    ASSERT_TRUE(back.is_ok()) << back.unwrap_err_unchecked().what();
    expect_same(back->tensors, m.views);
    hf_model missing;
    missing.add("model.layers.0.mlp.experts.1.up_proj.weight", {16, 8});
    EXPECT_EQ(hf_to_gguf(c, missing.views, nullptr, {}, "x").unwrap_err_unchecked().code, errc::not_found);
}

TEST(ExportHf, RefusesWhatItCannotConvert)
{
    model_config d;
    d.arch = architecture::deepseek_v3;
    d.block = ml::deepseek_v3;
    EXPECT_EQ(hf_to_gguf(d, {}, nullptr, {}, "x").unwrap_err_unchecked().code, errc::unsupported);
    model_config l;
    l.arch = architecture::llama;
    l.block = ml::llama3_8b;
    hf_model odd;
    odd.add("model.layers.0.self_attn.rotary_emb.inv_freq", {4});
    EXPECT_EQ(hf_to_gguf(l, odd.views, nullptr, {}, "x").unwrap_err_unchecked().code, errc::unsupported);
    hf_model wrong;
    wrong.add("model.layers.0.self_attn.q_proj.weight", {6, 8});
    EXPECT_EQ(hf_to_gguf(l, wrong.views, nullptr, {}, "x").unwrap_err_unchecked().code, errc::bad_shape);
}
