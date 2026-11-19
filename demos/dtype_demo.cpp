//
//  dtype_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The shared element types, their safetensors names, and conversion: bfloat16 weights to float, then to half.

#include <snicholls/formats/dtype.hpp>

#include <array>
#include <cstdio>
#include <cstring>

using namespace snicholls::formats;

int main()
{
    int failures = 0;
    for (const dtype t : all_dtypes)
        std::printf("%-8s %zu byte(s)%s\n", safetensors_name(t).data(), size_of(t), is_float(t) ? ", float" : "");
    //! [dtype]
    const std::array<std::uint16_t, 3> bf16{0x3f80, 0x4000, 0xbf00};   // 1, 2, -0.5 as bfloat16
    std::array<float, 3> f32{};
    const auto r = convert_elements(dtype::bf16, std::as_bytes(std::span(bf16)), dtype::f32, std::as_writable_bytes(std::span(f32)));
    //! [dtype]
    std::printf("bf16 -> f32: %g %g %g\n", f32[0], f32[1], f32[2]);
    failures += !(r.is_ok() && f32[0] == 1.0f && f32[1] == 2.0f && f32[2] == -0.5f);
    std::array<std::uint16_t, 3> f16{};
    failures += convert_elements(dtype::f32, std::as_bytes(std::span(f32)), dtype::f16, std::as_writable_bytes(std::span(f16))).is_err();
    std::printf("f32 -> f16 bits: %04x %04x %04x\n", f16[0], f16[1], f16[2]);
    failures += !(f16[0] == 0x3c00 && f16[1] == 0x4000 && f16[2] == 0xb800);
    std::array<std::int8_t, 1> narrow{};
    const std::array<std::int32_t, 1> wide{300};
    const auto refused = convert_elements(dtype::i32, std::as_bytes(std::span(wide)), dtype::i8, std::as_writable_bytes(std::span(narrow)));
    std::printf("300 as i8: %s\n", refused.is_err() ? refused.unwrap_err_unchecked().what().c_str() : "converted");
    failures += !(refused.is_err() && refused.unwrap_err_unchecked().code == errc::out_of_bounds);
    const std::array<float, 1> f{1.5f};
    std::array<std::int32_t, 1> i32{};
    const auto to_int = convert_elements(dtype::f32, std::as_bytes(std::span(f)), dtype::i32, std::as_writable_bytes(std::span(i32)));
    std::printf("1.5 as i32: %s\n", to_int.is_err() ? to_int.unwrap_err_unchecked().what().c_str() : "converted");
    failures += !(to_int.is_err() && to_int.unwrap_err_unchecked().code == errc::unsupported);
    return failures;
}
