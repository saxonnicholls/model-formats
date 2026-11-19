//
//  files_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Save tensors as safetensors (streamed, atomic), then map the file and read it.

#include <snicholls/formats/io/files.hpp>

#include <cstdio>
#include <filesystem>
#include <vector>

using namespace snicholls::formats;

int main()
{
    const auto path = std::filesystem::temp_directory_path() / "model-formats-files-demo.safetensors";
    const std::vector<float> w(1024, 0.25f);
    const std::vector<tensor_view> tensors{{"embed.weight", dtype::f32, {32, 32}, std::as_bytes(std::span(w))}};
    //! [files]
    const auto saved = io::save_tensors(path, tensors);   // written beside, renamed over: never half a file
    const auto model = io::open_model(path);              // mapped: the tensors are views of the page cache
    //! [files]
    if (model.is_ok())
        std::printf("%s: %s, %zu tensor(s), mapped: %s\n", path.string().c_str(), convert::to_string(model->model.format).data(),
                    model->model.tensors.size(), model->file.mapped() ? "yes" : "no");
    const bool good = saved.is_ok() && model.is_ok() && model->model.tensors[0].shape == tensor_shape{32, 32};
    std::filesystem::remove(path);
    return good ? 0 : 1;
}
