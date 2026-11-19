//
//  safetensors_benchmark.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  safetensors::parse of a header of 1,000 tensors, against a hand-written walk of the same JSON with nlohmann that
//  only reads each entry's dtype, shape and offsets (no validation). The fairness check requires both to find the same
//  tensors at the same offsets.

#include <snicholls/formats/safetensors/safetensors.hpp>

#include <benchmark/benchmark.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace snicholls::formats;
namespace st = snicholls::formats::safetensors;

namespace
{
const std::vector<std::byte>& file()
{
    static const std::vector<std::byte> bytes = [] {
        static std::vector<float> data(1000 * 16, 1.0f);
        static std::vector<std::string> names;
        for (int i = 0; i < 1000; ++i)
            names.push_back("model.layers." + std::to_string(i / 10) + ".mlp.weight_" + std::to_string(i));
        std::vector<tensor_view> ts;
        for (int i = 0; i < 1000; ++i)
            ts.push_back({names[i], dtype::f32, {4, 4}, std::as_bytes(std::span(data).subspan(16 * i, 16))});
        return *st::write(ts);
    }();
    return bytes;
}

std::map<std::string, std::uint64_t> hand_written(const std::vector<std::byte>& b)
{
    const std::uint64_t n = snicholls::load_le<std::uint64_t>(b.data());
    const auto j = nlohmann::json::parse(reinterpret_cast<const char*>(b.data() + 8), reinterpret_cast<const char*>(b.data() + 8 + n));
    std::map<std::string, std::uint64_t> out;
    for (const auto& [k, v] : j.items())
        if (k != "__metadata__")
            out[k] = v["data_offsets"][0].get<std::uint64_t>();
    return out;
}

std::map<std::string, std::uint64_t> library(const std::vector<std::byte>& b)
{
    std::map<std::string, std::uint64_t> out;
    const auto f = st::parse(b);
    for (const auto& t : f->tensors)
        out[t.name] = t.begin;
    return out;
}

const bool fair = [] {
    if (hand_written(file()) != library(file())) {
        std::fprintf(stderr, "safetensors_benchmark: the parsers disagree; refusing to benchmark\n");
        std::exit(1);
    }
    return true;
}();

void BM_SafetensorsHeaderHandWritten(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(hand_written(file()));
}
void BM_SafetensorsHeaderParse(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(st::parse(file()));
}
} // namespace

BENCHMARK(BM_SafetensorsHeaderHandWritten);
BENCHMARK(BM_SafetensorsHeaderParse);
BENCHMARK_MAIN();
