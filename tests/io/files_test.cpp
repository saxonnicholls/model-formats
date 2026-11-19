//
//  files_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/io/files.hpp>

#include <gtest/gtest.h>

#include <sys/stat.h>

#include "support/temp_dir.hpp"

using namespace snicholls::formats;

TEST(IoFiles, SaveAndOpenEveryWrittenFormat)
{
    const test_support::temp_dir dir("files");
    const std::vector<float> w{1, 2, 3, 4, 5, 6};
    const std::vector<tensor_view> ts{{"w", dtype::f32, {2, 3}, std::as_bytes(std::span(w))}};
    for (const char* name : {"m.safetensors", "m.gguf", "m.npz", "m.npy"}) {
        const auto path = dir.path / name;
        ASSERT_TRUE(io::save_tensors(path, ts).is_ok()) << name;
        const auto m = io::open_model(path);
        ASSERT_TRUE(m.is_ok()) << name << ": " << m.unwrap_err_unchecked().what();
        EXPECT_TRUE(m->file.mapped());
        ASSERT_EQ(m->model.tensors.size(), 1u);
        EXPECT_EQ(m->model.tensors[0].shape, (tensor_shape{2, 3}));
        EXPECT_EQ(reinterpret_cast<const float*>(m->model.tensors[0].data.data())[5], 6.0f);
        EXPECT_GE(m->model.tensors[0].data.data(), m->file.bytes().data());   // a view of the mapping
    }
    EXPECT_EQ(io::save_tensors(dir.path / "m.txt", ts).unwrap_err_unchecked().code, errc::bad_argument);
}

TEST(IoFiles, StreamedWritesAreExactAndAtomic)
{
    const test_support::temp_dir dir("streamed");
    const auto path = dir.path / "sub" / "out.bin";
    const std::string a(5000, 'a'), b = "tail";
    ASSERT_TRUE(io::write_streamed(path, [&](auto&& sink) {
                    sink(std::as_bytes(std::span(a.data(), a.size())));
                    sink(std::as_bytes(std::span(b.data(), b.size())));
                    return result<void>(snicholls::ok());
                }).is_ok());
    EXPECT_EQ(std::filesystem::file_size(path), 5004u);
    // a producer that fails leaves the old file and no temporary
    const auto r = io::write_streamed(path, [&](auto&& sink) {
        sink(std::as_bytes(std::span(b.data(), b.size())));
        return result<void>(fail(errc::bad_argument, "stop"));
    });
    EXPECT_TRUE(r.is_err());
    EXPECT_EQ(std::filesystem::file_size(path), 5004u);
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(path.parent_path()), std::filesystem::directory_iterator{}), 1);
}

TEST(IoFiles, MissingReadOnlyAndEmptyFiles)
{
    const test_support::temp_dir dir("ro");
    EXPECT_EQ(io::mapped_file::open(dir.path / "missing").unwrap_err_unchecked().code, errc::io);
    const std::vector<float> w{1};
    const std::vector<tensor_view> ts{{"w", dtype::f32, {1}, std::as_bytes(std::span(w))}};
    const auto path = dir.path / "ro.safetensors";
    ASSERT_TRUE(io::save_tensors(path, ts).is_ok());
    ::chmod(path.c_str(), 0444);
    const auto m = io::open_model(path);   // read, not mapped
    ASSERT_TRUE(m.is_ok()) << m.unwrap_err_unchecked().what();
    EXPECT_EQ(m->model.tensors.size(), 1u);
    ASSERT_TRUE(io::write_file(dir.path / "empty", {}).is_ok());
    const auto e = io::mapped_file::open(dir.path / "empty");
    ASSERT_TRUE(e.is_ok());
    EXPECT_TRUE(e->bytes().empty());
    EXPECT_EQ(io::open_model(dir.path / "empty").unwrap_err_unchecked().code, errc::unsupported);
}
