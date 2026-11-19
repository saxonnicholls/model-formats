//
//  pattern_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Choosing files to download with huggingface_hub's patterns.

#include <snicholls/formats/hub/pattern.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace snicholls::formats::hub;

int main()
{
    const std::vector<std::string> files{"config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5", "onnx/model.onnx"};
    //! [pattern]
    const std::vector<std::string> include{"*.json", "*.safetensors"};
    const std::vector<std::string> exclude{"onnx/"};
    //! [pattern]
    int chosen = 0;
    for (const auto& f : files)
        if (select(f, include, exclude)) {
            std::printf("download %s\n", f.c_str());
            ++chosen;
        }
    return chosen == 2 ? 0 : 1;
}
