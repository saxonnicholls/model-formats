//
//  tokenizer_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A tokenizer.json and tokenizer_config.json read into a vocabulary, and the chat template as text.

#include <snicholls/formats/hf/tokenizer.hpp>

#include <cstdio>

using namespace snicholls::formats;

int main()
{
    //! [tokenizer]
    auto tok = hf::parse_tokenizer(R"({"added_tokens":[{"id":3,"content":"<|im_end|>","special":true}],
        "pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":["h i"]}})");
    const auto cfg = hf::parse_tokenizer_config(*tok, R"({"eos_token":"<|im_end|>",
        "chat_template":"{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"})");
    //! [tokenizer]
    std::printf("%zu tokens, %zu merge(s), eos %u, template of %zu characters (exposed, not rendered)\n", tok->vocab.tokens.size(),
                tok->vocab.merges.size(), *tok->vocab.eos, tok->default_chat_template()->size());
    return tok.is_ok() && cfg.is_ok() && tok->vocab.eos == 3u && tok->vocab.type_of(3) == gguf::token_type::control ? 0 : 1;
}
