# model-formats - the unit tests. tests/ mirrors include/snicholls/formats/: <sub>/<name>.hpp has
# tests/<sub>/<name>_test.cpp, listed here (scripts/check_mirror.py checks both ways).
#
# model_formats_add_test(<sub/name> <target>) builds tests/<sub/name>_test.cpp against <target> and registers it as
# the CTest unit.<sub/name>, label unit.

function(model_formats_add_test name target)
    string(MAKE_C_IDENTIFIER "${name}" _id)
    add_executable(test_${_id} ${name}_test.cpp ${ARGN})
    target_link_libraries(test_${_id} PRIVATE ${target} GTest::gtest_main GTest::gmock)
    target_include_directories(test_${_id} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(test_${_id} PRIVATE MODEL_FORMATS_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
    model_formats_compile_options(test_${_id})
    add_test(NAME unit.${name} COMMAND test_${_id})
    set_tests_properties(unit.${name} PROPERTIES LABELS unit)
endfunction()

model_formats_add_test(version snicholls::formats)
model_formats_add_test(error snicholls::formats)
model_formats_add_test(model_config snicholls::formats)
model_formats_add_test(dtype snicholls::formats)
model_formats_add_test(tensor snicholls::formats)
model_formats_add_test(vocabulary snicholls::formats)
model_formats_add_test(archive/zip snicholls::formats)
model_formats_add_test(gguf/ggml_types snicholls::formats)
model_formats_add_test(gguf/gguf snicholls::formats)
model_formats_add_test(gguf/keys snicholls::formats)
model_formats_add_test(gguf/metadata snicholls::formats)
model_formats_add_test(hf/config snicholls::formats)
model_formats_add_test(hf/names snicholls::formats)
model_formats_add_test(hf/tokenizer snicholls::formats)
model_formats_add_test(llmc/llmc snicholls::formats)
model_formats_add_test(numpy/npy snicholls::formats)
model_formats_add_test(pytorch/checkpoint snicholls::formats)
model_formats_add_test(pytorch/pickle snicholls::formats)
model_formats_add_test(safetensors/index snicholls::formats)
model_formats_add_test(safetensors/safetensors snicholls::formats)
