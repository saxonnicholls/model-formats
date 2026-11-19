//
//  convert_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Tensor-level conversion: safetensors to GGUF to NumPy, detecting each format from its bytes.

#include <snicholls/formats/convert/convert.hpp>

#include <cstdio>
#include <vector>

using namespace snicholls::formats;
using namespace snicholls::formats::convert;

int main()
{
    const std::vector<float> w(12, 0.5f);
    const std::vector<tensor_view> tensors{{"proj.weight", dtype::f32, {3, 4}, std::as_bytes(std::span(w))}};
    //! [convert]
    const auto st = write_any(file_format::safetensors, tensors);
    const auto from_st = read_any(*st);                                 // detects safetensors
    const auto gg = write_any(file_format::gguf, from_st->tensors);     // the same tensors as GGUF
    const auto from_gg = read_any(*gg);
    const auto npz = write_any(file_format::npz, from_gg->tensors);     // and as np.savez
    const auto bf16 = cast(from_gg->tensors[0], dtype::bf16);           // half the bytes
    //! [convert]
    int failures = 0;
    for (const auto* b : {&*st, &*gg, &*npz}) {
        const auto f = detect(*b);
        std::printf("%-12s %6zu bytes\n", f ? to_string(*f).data() : "?", b->size());
        failures += !f;
    }
    std::printf("bf16 copy: %zu bytes\n", bf16->data.size());
    return failures + !(from_gg.is_ok() && from_gg->tensors[0].shape == tensor_shape{3, 4} && bf16->data.size() == 24);
}
