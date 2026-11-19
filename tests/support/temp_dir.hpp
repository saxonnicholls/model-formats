//
//  temp_dir.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Test support: a fresh directory under the system's temporary directory, removed when the test ends.

#ifndef model_formats_tests_support_temp_dir_hpp
#define model_formats_tests_support_temp_dir_hpp

#include <filesystem>
#include <string>
#include <system_error>

#include <unistd.h>

namespace test_support
{
struct temp_dir
{
    std::filesystem::path path;
    explicit temp_dir(const std::string& name)
        : path(std::filesystem::temp_directory_path() / ("model-formats-" + name + "-" + std::to_string(::getpid())))
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path);
    }
    ~temp_dir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    temp_dir(const temp_dir&) = delete;
    temp_dir& operator=(const temp_dir&) = delete;
};
} // namespace test_support

#endif // model_formats_tests_support_temp_dir_hpp
