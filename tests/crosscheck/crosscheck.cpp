//
//  crosscheck.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The C++ half of scripts/crosscheck.py: independent readers check our files, and we check theirs.
//
//      mf_crosscheck write <dir>     our files: every GGUF value type and four tensor types, an llm.c GPT-2 as GGUF,
//                                    safetensors of every dtype, .npy and .npz
//      mf_crosscheck read <file>...  any files: one line per tensor, "file<TAB>name<TAB>dtype<TAB>shape<TAB>sha256",
//                                    and for GGUF one per key, "file<TAB>key<TAB>type<TAB>value"
//
//  scripts/crosscheck.py computes the same lines with gguf-py, safetensors and numpy, and compares.

#include <snicholls/formats/convert/export.hpp>
#include <snicholls/formats/hub/digest.hpp>
#include <snicholls/formats/io/files.hpp>

#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace snicholls::formats;

namespace
{
std::string shape_text(const tensor_shape& s)
{
    std::string t;
    for (std::size_t i = 0; i < s.rank; ++i)
        t += (i ? "," : "") + std::to_string(s[i]);
    return t;
}

std::string value_text(const gguf::value& v)
{
    switch (v.type) {
    case gguf::value_type::string: return std::string(*v.as_string());
    case gguf::value_type::boolean: return *v.as_bool() ? "true" : "false";
    case gguf::value_type::f32: case gguf::value_type::f64: {
        char b[64];
        std::snprintf(b, sizeof b, "%.17g", *v.as_f64());
        return b;
    }
    case gguf::value_type::array: {
        std::string t = "[" + std::string(gguf::to_string(v.element_type)) + " x " + std::to_string(v.count) + "]";
        if (const auto s = v.strings())
            for (const auto x : *s)
                t += "|" + std::string(x);
        else if (const auto n = v.integers())
            for (const auto x : *n)
                t += "|" + std::to_string(x);
        return t;
    }
    case gguf::value_type::i8: case gguf::value_type::i16: case gguf::value_type::i32: case gguf::value_type::i64: return std::to_string(*v.as_i64());
    default: return std::to_string(*v.as_u64());
    }
}

int read(int argc, char** argv)
{
    for (int i = 2; i < argc; ++i) {
        const std::filesystem::path p = argv[i];
        const auto f = io::mapped_file::open(p);
        if (f.is_err()) {
            std::fprintf(stderr, "%s\n", f.unwrap_err_unchecked().what().c_str());
            return 1;
        }
        const std::string name = p.filename().string();
        if (convert::detect(f->bytes()) == convert::file_format::gguf) {
            const auto g = gguf::parse(f->bytes());
            if (g.is_err()) {
                std::fprintf(stderr, "%s: %s\n", name.c_str(), g.unwrap_err_unchecked().what().c_str());
                return 1;
            }
            for (const auto& e : g->metadata)
                std::printf("%s\t%s\t%s\t%s\n", name.c_str(), std::string(e.key).c_str(), gguf::to_string(e.val.type).data(), value_text(e.val).c_str());
            for (const auto& t : g->tensors)
                std::printf("%s\t%s\t%s\t%s\t%s\n", name.c_str(), std::string(t.name).c_str(), gguf::traits(t.type).name.data(),
                            shape_text(t.shape).c_str(), hub::sha256_hex(t.data).c_str());
            continue;
        }
        const auto l = convert::read_any(f->bytes(), {.dequantize = false});
        if (l.is_err()) {
            std::fprintf(stderr, "%s: %s\n", name.c_str(), l.unwrap_err_unchecked().what().c_str());
            return 1;
        }
        for (const auto& t : l->tensors)
            std::printf("%s\t%s\t%s\t%s\t%s\n", name.c_str(), std::string(t.name).c_str(), safetensors_name(t.type).data(), shape_text(t.shape).c_str(),
                        hub::sha256_hex(t.data).c_str());
    }
    return 0;
}

template <typename T>
std::vector<std::byte> ramp(std::size_t n, double scale)
{
    std::vector<std::byte> b(n * sizeof(T));
    for (std::size_t i = 0; i < n; ++i) {
        const T v = static_cast<T>(static_cast<double>(i % 97) * scale);
        std::memcpy(b.data() + i * sizeof(T), &v, sizeof(T));
    }
    return b;
}

int write(const std::filesystem::path& dir)
{
    std::filesystem::create_directories(dir);
    // GGUF: every value type, nested arrays, four tensor types
    gguf::writer w;
    w.set("general.architecture", gguf::value_string("crosscheck"));
    w.set("t.u8", gguf::value_u8(250));
    w.set("t.i8", gguf::value_i8(-120));
    w.set("t.u16", gguf::value_u16(65000));
    w.set("t.i16", gguf::value_i16(-32000));
    w.set("t.u32", gguf::value_u32(4000000000u));
    w.set("t.i32", gguf::value_i32(-2000000000));
    w.set("t.f32", gguf::value_f32(0.25f));
    w.set("t.bool", gguf::value_bool(true));
    w.set("t.string", gguf::value_string("h\xc3\xa9llo"));
    w.set("t.u64", gguf::value_u64(std::numeric_limits<std::uint64_t>::max()));
    w.set("t.i64", gguf::value_i64(std::numeric_limits<std::int64_t>::min()));
    w.set("t.f64", gguf::value_f64(1e-300));
    const std::vector<std::string> strings{"a", "", "\xc4\xa0z"};
    w.set("t.strings", gguf::value_strings(strings));
    w.set("t.ints", gguf::value_array(gguf::value_type::i32, {gguf::value_i32(1), gguf::value_i32(-2), gguf::value_i32(3)}));
    static const auto f32 = ramp<float>(4 * 64, 0.5), f16 = ramp<std::uint16_t>(2 * 32, 1), bf16 = ramp<std::uint16_t>(3 * 32, 1);
    std::vector<float> row(64);
    std::iota(row.begin(), row.end(), -32.0f);
    static std::vector<snicholls::ml::block_q8_0> q(2);
    snicholls::ml::quantize(row, std::span(q));
    w.add_tensor("f32", gguf::ggml_type::f32, {4, 64}, f32);
    w.add_tensor("f16", gguf::ggml_type::f16, {2, 32}, f16);
    w.add_tensor("bf16", gguf::ggml_type::bf16, {3, 32}, bf16);
    w.add_tensor("q8_0", gguf::ggml_type::q8_0, {64}, std::as_bytes(std::span(q)));
    if (io::write_file(dir / "ours.gguf", *w.write()).is_err())
        return 1;
    // llm.c GPT-2 -> llama.cpp's gpt2
    const llmc::gpt2_shape s{.max_seq = 16, .vocab = 3, .layers = 2, .heads = 2, .channels = 8, .padded_vocab = 8};
    std::vector<float> p(llmc::parameter_count(s));
    std::iota(p.begin(), p.end(), 0.0f);
    const auto ck = *llmc::write_checkpoint(s, dtype::f32, std::as_bytes(std::span(p)));
    const auto c = llmc::parse_checkpoint(ck);
    vocabulary v;
    v.tokens = {"a", "b", "ab"};
    v.merges = {"a b"};
    const auto g = convert::llmc_to_gguf(*c, &v, "tiny-gpt2");
    if (g.is_err() || io::write_streamed(dir / "gpt2.gguf", [&](auto&& sink) { return g->writer.write_to(sink); }).is_err())
        return 1;
    // safetensors of every dtype
    std::vector<std::vector<std::byte>> data;
    std::vector<std::string> names;
    for (const dtype t : all_dtypes) {
        names.push_back(std::string("t_") + std::string(safetensors_name(t)));
        std::vector<std::byte> b(6 * size_of(t));
        for (std::size_t i = 0; i < b.size(); ++i)
            b[i] = static_cast<std::byte>(t == dtype::boolean ? i % 2 : (i * 37 + 11) % 251);
        if (t == dtype::f8_e4m3 || t == dtype::f8_e5m2 || t == dtype::f16 || t == dtype::bf16 || t == dtype::f32 || t == dtype::f64)
            for (std::size_t i = 0; i < b.size(); ++i)
                b[i] = static_cast<std::byte>(i % size_of(t) == size_of(t) - 1 ? 0x3c : 0x01);   // finite numbers
        data.push_back(std::move(b));
    }
    std::vector<tensor_view> views;
    for (std::size_t i = 0; i < data.size(); ++i)
        views.push_back({names[i], all_dtypes[i], {2, 3}, data[i]});
    const std::vector<std::pair<std::string, std::string>> meta{{"format", "pt"}, {"source", "model-formats"}};
    if (io::save_tensors(dir / "ours.safetensors", views, meta).is_err())
        return 1;
    // NumPy
    std::vector<tensor_view> np;
    for (const auto& t : views)
        if (numpy::descr_of(t.type))
            np.push_back(t);
    if (io::save_tensors(dir / "ours.npz", np).is_err() || io::save_tensors(dir / "ours.npy", std::span(np).first(1)).is_err())
        return 1;
    std::printf("%s\n", dir.string().c_str());
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc >= 3 && std::strcmp(argv[1], "write") == 0)
        return write(argv[2]);
    if (argc >= 3 && std::strcmp(argv[1], "read") == 0)
        return read(argc, argv);
    std::fprintf(stderr, "usage: mf_crosscheck write <dir> | read <file>...\n");
    return 2;
}
