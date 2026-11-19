//
//  checkpoint_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A torch.save archive (assembled here byte for byte as PyTorch writes one) read without executing its pickle.

#include <snicholls/formats/pytorch/checkpoint.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace snicholls::formats;

int main()
{
    // data.pkl for {"w": tensor of 2 floats in storage "0"}, protocol 2, as torch.save emits it
    using namespace std::string_view_literals;
    const std::string pkl = std::string("\x80\x02}X\x01\x00\x00\x00w"sv) + "ctorch._utils\n_rebuild_tensor_v2\n(("
                            + std::string("X\x07\x00\x00\x00storage"sv) + "ctorch\nFloatStorage\n" + std::string("X\x01\x00\x00\x00" "0"sv)
                            + std::string("X\x03\x00\x00\x00" "cpu"sv) + std::string("K\x02tQK\x00K\x02\x85K\x01\x85\x89"sv)
                            + "ccollections\nOrderedDict\n)RtRs.";
    const std::vector<float> storage{0.5f, -0.5f};
    const std::string order = "little";
    const std::vector<zip::member> members{{"archive/data.pkl", std::as_bytes(std::span(pkl.data(), pkl.size()))},
                                           {"archive/byteorder", std::as_bytes(std::span(order.data(), order.size()))},
                                           {"archive/data/0", std::as_bytes(std::span(storage))}};
    const auto file = zip::write(members);
    //! [checkpoint]
    const auto ckpt = pytorch::parse(*file);   // the restricted unpickler: nothing in the file runs
    for (const auto& t : ckpt->tensors)
        std::printf("%s %s %s contiguous=%d\n", t.name.c_str(), safetensors_name(t.type).data(), to_string(t.shape).c_str(), t.contiguous);
    //! [checkpoint]
    if (ckpt.is_err())
        std::printf("%s\n", ckpt.unwrap_err_unchecked().what().c_str());
    return ckpt.is_ok() && ckpt->tensors.size() == 1 && reinterpret_cast<const float*>(ckpt->tensors[0].data.data())[1] == -0.5f ? 0 : 1;
}
