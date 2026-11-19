//
//  dtype_benchmark.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  convert_elements bf16 -> f32 against the hand-written loop (shift each bf16 into the high half of a float), on the
//  same data. The fairness check refuses to run if the two disagree on any element.

#include <snicholls/formats/dtype.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace snicholls::formats;

namespace
{
std::vector<std::uint16_t> input(std::size_t n)
{
    std::vector<std::uint16_t> v(n);
    std::uint32_t x = 12345;
    for (auto& e : v) {
        x = x * 1664525u + 1013904223u;
        e = static_cast<std::uint16_t>(0x3000 + (x >> 20));   // finite values around 1
    }
    return v;
}

void hand_written(const std::vector<std::uint16_t>& in, std::vector<float>& out)
{
    for (std::size_t i = 0; i < in.size(); ++i)
        out[i] = std::bit_cast<float>(static_cast<std::uint32_t>(in[i]) << 16);
}

void library(const std::vector<std::uint16_t>& in, std::vector<float>& out)
{
    (void)convert_elements(dtype::bf16, std::as_bytes(std::span(in)), dtype::f32, std::as_writable_bytes(std::span(out)));
}

const bool fair = [] {
    const auto in = input(1 << 16);
    std::vector<float> a(in.size()), b(in.size());
    hand_written(in, a);
    library(in, b);
    if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
        std::fprintf(stderr, "dtype_benchmark: the library and the hand-written loop disagree; refusing to benchmark\n");
        std::exit(1);
    }
    return true;
}();

void BM_Bf16ToF32HandWritten(benchmark::State& s)
{
    const auto in = input(static_cast<std::size_t>(s.range(0)));
    std::vector<float> out(in.size());
    for (auto _ : s) {
        hand_written(in, out);
        benchmark::DoNotOptimize(out.data());
    }
    s.SetBytesProcessed(static_cast<std::int64_t>(s.iterations() * in.size() * 2));
}
void BM_Bf16ToF32ConvertElements(benchmark::State& s)
{
    const auto in = input(static_cast<std::size_t>(s.range(0)));
    std::vector<float> out(in.size());
    for (auto _ : s) {
        library(in, out);
        benchmark::DoNotOptimize(out.data());
    }
    s.SetBytesProcessed(static_cast<std::int64_t>(s.iterations() * in.size() * 2));
}
} // namespace

BENCHMARK(BM_Bf16ToF32HandWritten)->Arg(1 << 20);
BENCHMARK(BM_Bf16ToF32ConvertElements)->Arg(1 << 20);
BENCHMARK_MAIN();
