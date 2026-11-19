//
//  npy_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  NumPy's .npy and .npz: write two arrays as np.savez does, read them back as views.

#include <snicholls/formats/numpy/npy.hpp>

#include <cstdio>
#include <vector>

using namespace snicholls::formats;
namespace np = snicholls::formats::numpy;

int main()
{
    //! [npy]
    const std::vector<float> embedding(4 * 8, 0.125f);
    const std::vector<std::int64_t> ids{5, 7, 11};
    const std::vector<tensor_view> arrays{{"embedding", dtype::f32, {4, 8}, std::as_bytes(std::span(embedding))},
                                          {"ids", dtype::i64, {3}, std::as_bytes(std::span(ids))}};
    const auto npz = np::write_npz(arrays);    // np.load reads it
    const auto back = np::parse_npz(*npz);     // views into `npz`
    //! [npy]
    for (const auto& a : *back)
        std::printf("%-10s %s %s\n", std::string(a.name).c_str(), np::descr_of(a.type)->data(), to_string(a.shape).c_str());
    const auto one = np::write_npy(arrays[1]);
    const auto ids_back = np::parse_npy(*one);
    return npz.is_ok() && back.is_ok() && back->size() == 2 && (*back)[0].shape == tensor_shape{4, 8} && ids_back.is_ok()
                   && reinterpret_cast<const std::int64_t*>(ids_back->data.data())[2] == 11
               ? 0
               : 1;
}
