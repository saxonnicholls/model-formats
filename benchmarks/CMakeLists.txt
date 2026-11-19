# model-formats - the benchmarks (Google Benchmark, core's pinned third_party/benchmark v1.9.5). Each compares the
# library with the hand-written equivalent on the same data and refuses to run if they disagree. A short smoke run of
# each is a CTest (label benchmark). Only Release numbers mean anything: scripts/run_benchmarks.sh writes BENCHMARKS.md.

foreach(_bench dtype safetensors)
    add_executable(bench_${_bench} ${_bench}_benchmark.cpp)
    target_link_libraries(bench_${_bench} PRIVATE snicholls::formats benchmark::benchmark)
    model_formats_compile_options(bench_${_bench})
    if(MODEL_FORMATS_BUILD_TESTS)
        add_test(NAME benchmark.${_bench} COMMAND bench_${_bench} --benchmark_min_time=0.01s)
        set_tests_properties(benchmark.${_bench} PROPERTIES LABELS benchmark)
    endif()
endforeach()
