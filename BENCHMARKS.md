# Benchmarks

Written by scripts/run_benchmarks.sh; never edited by hand. Release, Darwin x86_64, Apple clang version 17.0.0 (clang-1700.3.9.908), 2026-11-19.

## dtype

```
----------------------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------
BM_Bf16ToF32HandWritten/1048576         238584 ns       223671 ns         3183 bytes_per_second=8.73214Gi/s
BM_Bf16ToF32ConvertElements/1048576     229077 ns       226053 ns         3173 bytes_per_second=8.64012Gi/s
```

## safetensors

```
--------------------------------------------------------------------------
Benchmark                                Time             CPU   Iterations
--------------------------------------------------------------------------
BM_SafetensorsHeaderHandWritten    4648446 ns      4310250 ns          124
BM_SafetensorsHeaderParse         12052262 ns     10699043 ns           47
```
