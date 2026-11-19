//
//  tensor_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A shape, a borrowed view and an owned tensor.

#include <snicholls/formats/tensor.hpp>

#include <cstdio>
#include <vector>

using namespace snicholls::formats;

int main()
{
    //! [tensor]
    const std::vector<float> weights(6, 0.5f);
    const tensor_view v{"proj.weight", dtype::f32, {2, 3}, std::as_bytes(std::span(weights))};   // borrows `weights`
    const tensor owned = tensor::from(v);                                                       // copies
    //! [tensor]
    std::printf("%s %s %s, %llu elements, %zu bytes\n", std::string(v.name).c_str(), safetensors_name(v.type).data(),
                to_string(v.shape).c_str(), static_cast<unsigned long long>(*v.shape.elements()), v.data.size());
    std::printf("reversed (GGUF order): %s\n", to_string(v.shape.reversed()).c_str());
    const auto vs = views(std::span(&owned, 1));
    return v.shape.elements() == 6u && checked_byte_size(v.shape, v.type) == 24u && owned.data.size() == 24 && vs.size() == 1
                   && v.shape.reversed() == tensor_shape{3, 2}
               ? 0
               : 1;
}
