//
//  pattern_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/hub/pattern.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace snicholls::formats::hub;

TEST(HubPattern, FnmatchSemantics)
{
    static_assert(matches("*.safetensors", "model.safetensors"));
    static_assert(matches("*.safetensors", "sub/model.safetensors"));   // '*' crosses '/', as fnmatch's does
    static_assert(!matches("*.safetensors", "model.bin"));
    static_assert(matches("model-0000?-of-0000?.safetensors", "model-00001-of-00002.safetensors"));
    static_assert(matches("[ct]*.json", "config.json") && matches("[ct]*.json", "tokenizer.json") && !matches("[ct]*.json", "vocab.json"));
    static_assert(matches("[!c]*", "vocab.json") && !matches("[!c]*", "config.json"));
    static_assert(matches("[a-c]x", "bx") && !matches("[a-c]x", "dx"));
    static_assert(matches("onnx/", "onnx/model.onnx") && !matches("onnx/", "model.onnx"));
    static_assert(matches("*", "") && matches("", "") && !matches("", "a"));
    static_assert(matches("a[b", "a[b"));   // an unclosed set is literal
    static_assert(matches("*a*b*c", "xxaxxbxxc") && !matches("*a*b*c", "xxaxxbxx"));
}

TEST(HubPattern, SelectIncludesThenExcludes)
{
    const std::vector<std::string> include{"*.json", "*.safetensors"}, exclude{"*training*"};
    EXPECT_TRUE(select("config.json", include, exclude));
    EXPECT_FALSE(select("training_args.json", include, exclude));
    EXPECT_FALSE(select("pytorch_model.bin", include, exclude));
    EXPECT_TRUE(select("anything", {}, {}));
}
