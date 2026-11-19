//
//  safetensors_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Write two tensors and metadata to safetensors bytes, parse them back without copying, and show validation.

#include <snicholls/formats/safetensors/safetensors.hpp>

#include <cstdio>
#include <vector>

using namespace snicholls::formats;
namespace st = snicholls::formats::safetensors;

int main()
{
    int failures = 0;
    //! [safetensors]
    const std::vector<float> w{1, 2, 3, 4, 5, 6};
    const std::vector<float> b{0.5f, -0.5f};
    const std::vector<tensor_view> tensors{{"proj.weight", dtype::f32, {2, 3}, std::as_bytes(std::span(w))},
                                           {"proj.bias", dtype::f32, {2}, std::as_bytes(std::span(b))}};
    const std::vector<st::metadata_entry> meta{{"format", "pt"}};
    const auto bytes = st::write(tensors, meta);   // sorted, aligned, deterministic
    const auto file = st::parse(*bytes);           // views into `bytes`, nothing copied
    for (const auto& t : file->tensors)
        std::printf("%-12s %s %s at [%llu, %llu)\n", t.name.c_str(), safetensors_name(t.type).data(), to_string(t.shape).c_str(),
                    static_cast<unsigned long long>(t.begin), static_cast<unsigned long long>(t.end));
    //! [safetensors]
    failures += !(bytes.is_ok() && file.is_ok() && file->tensors.size() == 2 && *file->meta("format") == "pt");
    const auto* p = file->find("proj.weight");
    failures += !(p && p->data.size() == 24 && reinterpret_cast<const float*>(p->data.data())[5] == 6.0f);

    std::vector<std::byte> cut(bytes->begin(), bytes->end() - 4);   // the last tensor loses a float
    const auto bad = st::parse(cut);
    std::printf("truncated copy: %s\n", bad.is_err() ? bad.unwrap_err_unchecked().what().c_str() : "accepted");
    failures += !bad.is_err();

    const auto l = st::plan(tensors, meta);
    std::printf("header %zu bytes, data %llu bytes, first tensor %s\n", l->header.size(),
                static_cast<unsigned long long>(l->data_size), std::string(l->order[0].name).c_str());
    failures += !(l.is_ok() && l->header.size() % 8 == 0 && l->data_size == 32);
    std::size_t streamed = 0;
    failures += st::write_to(tensors, meta, [&](std::span<const std::byte> s) { streamed += s.size(); }).is_err();
    failures += streamed != bytes->size();
    return failures;
}
