# model-formats - read and write the machine-learning model file formats, and
# download models from the Hugging Face Hub, for the snicholls libraries.
#
# Three targets, so a consumer links only what it uses:
#
#   snicholls::formats       the formats over bytes: safetensors, GGUF, NumPy,
#                            llm.c, the PyTorch zip and its restricted
#                            unpickler, Hugging Face config / tokenizer / names,
#                            conversion and export. Core and nlohmann/json
#                            (core-adapters' snicholls::core_json); no I/O.
#   snicholls::formats_io    files: memory-mapped reading and atomic writing
#                            of any format, Hugging Face model directories.
#                            Adds ts-moveables (mmap/, fs/).
#   snicholls::formats_hub   the Hugging Face Hub client and its cache. Adds
#                            ts-moveables' http::client over OpenSSL.
#
# Driven as the top-level project (tests, demos, benchmarks, tools, docs) or as
# a subdirectory of a consumer (the targets and the tools only).

cmake_minimum_required(VERSION 3.20)

project(snicholls_formats
    VERSION 0.1.0
    DESCRIPTION "model-formats - machine-learning model file formats and the Hugging Face Hub for the snicholls libraries"
    LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MODEL_FORMATS_TOP_LEVEL ON)
else()
    set(MODEL_FORMATS_TOP_LEVEL OFF)
endif()

option(MODEL_FORMATS_BUILD_TESTS "Build the model-formats tests and demos" ${MODEL_FORMATS_TOP_LEVEL})
option(MODEL_FORMATS_BUILD_TOOLS "Build mf-inspect, mf-convert and mf-download" ${MODEL_FORMATS_TOP_LEVEL})
option(MODEL_FORMATS_WITH_HUB "Define snicholls::formats_hub (needs OpenSSL)" ON)
option(MODEL_FORMATS_BUILD_BENCHMARKS "Build the benchmarks (Google Benchmark)" OFF)
option(MODEL_FORMATS_BUILD_DOCS "Build the Doxygen documentation; warnings are errors" OFF)
option(MODEL_FORMATS_WERROR "Treat warnings as errors in everything this project compiles" OFF)
option(MODEL_FORMATS_NETWORK_TESTS "Register the tests that reach huggingface.co (label network)" OFF)
set(MODEL_FORMATS_SANITIZE "" CACHE STRING "Sanitizers for the tests, demos and tools, e.g. address,undefined")

if(MODEL_FORMATS_TOP_LEVEL AND NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# ---------------------------------------------------------------------------
# snicholls-core (third_party/core). The diamond rules (core CLAUDE.md
# "Versioning and the diamond"): the first copy added wins, so core is added
# before ts-moveables and core-adapters, whose own copies are then skipped.

function(model_formats_require_submodule path)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${path}/CMakeLists.txt")
        message(FATAL_ERROR "Submodule '${path}' is not checked out.\nRun: scripts/bootstrap.sh")
    endif()
endfunction()

model_formats_require_submodule(third_party/core)
add_subdirectory(third_party/core ${CMAKE_CURRENT_BINARY_DIR}/third_party/core EXCLUDE_FROM_ALL)
snicholls_core_require(0.16.0)

# ts-moveables (third_party/ts-moveables): mapped files, atomic writes, the event loop and http::client
model_formats_require_submodule(third_party/ts-moveables)
set(TS_MOVEABLES_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ts-moveables ${CMAKE_CURRENT_BINARY_DIR}/third_party/ts-moveables EXCLUDE_FROM_ALL)

# core-adapters (third_party/core-adapters): snicholls::core_json, nlohmann/json v3.12.0 with implicit conversions off
model_formats_require_submodule(third_party/core-adapters)
set(CORE_ADAPTERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CORE_ADAPTERS_WITH_NANOBIND OFF CACHE BOOL "" FORCE)
set(CORE_ADAPTERS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(CORE_ADAPTERS_WITH_NLOHMANN ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/core-adapters ${CMAKE_CURRENT_BINARY_DIR}/third_party/core-adapters EXCLUDE_FROM_ALL)

function(model_formats_compile_options target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(MODEL_FORMATS_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(MODEL_FORMATS_SANITIZE)
        target_compile_options(${target} PRIVATE -fsanitize=${MODEL_FORMATS_SANITIZE} -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=${MODEL_FORMATS_SANITIZE})
    endif()
endfunction()

# ---------------------------------------------------------------------------
# The targets

add_library(snicholls_formats INTERFACE)
add_library(snicholls::formats ALIAS snicholls_formats)
target_include_directories(snicholls_formats INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(snicholls_formats INTERFACE snicholls::core snicholls::core_json)
target_compile_features(snicholls_formats INTERFACE cxx_std_20)

add_library(snicholls_formats_io INTERFACE)
add_library(snicholls::formats_io ALIAS snicholls_formats_io)
target_link_libraries(snicholls_formats_io INTERFACE snicholls::formats snicholls::ts_moveables)
target_compile_features(snicholls_formats_io INTERFACE cxx_std_20)

if(MODEL_FORMATS_WITH_HUB)
    # Homebrew: the newest OpenSSL installed, so its headers agree with whatever sits in /usr/local/include (which the
    # compiler searches before any -isystem directory) and with the library linked
    if(APPLE AND NOT OPENSSL_ROOT_DIR)
        foreach(_mf_formula openssl@4 openssl@3)
            execute_process(COMMAND brew --prefix ${_mf_formula} OUTPUT_VARIABLE _mf_openssl
                            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _mf_brew)
            if(_mf_brew EQUAL 0 AND EXISTS "${_mf_openssl}/include/openssl/ssl.h")
                set(OPENSSL_ROOT_DIR "${_mf_openssl}")
                break()
            endif()
        endforeach()
    endif()
    find_package(OpenSSL 3 REQUIRED)
    add_library(snicholls_formats_hub INTERFACE)
    add_library(snicholls::formats_hub ALIAS snicholls_formats_hub)
    target_link_libraries(snicholls_formats_hub INTERFACE snicholls::formats_io OpenSSL::SSL OpenSSL::Crypto)
    # The 3.0 API: ts-moveables' tls/openssl.hpp calls SSL_set1_host, deprecated in OpenSSL 4 (erratum/ts-moveables.md)
    target_compile_definitions(snicholls_formats_hub INTERFACE OPENSSL_API_COMPAT=0x30000000L)
    target_compile_features(snicholls_formats_hub INTERFACE cxx_std_20)
endif()

# ---------------------------------------------------------------------------
# Tests, demos, tools, benchmarks, docs

if(MODEL_FORMATS_BUILD_TESTS)
    enable_testing()
    if(NOT TARGET GTest::gtest_main)
        # GoogleTest: core's pinned copy (third_party/core/third_party/googletest, v1.18.0)
        set(_gtest_root ${CMAKE_CURRENT_SOURCE_DIR}/third_party/core/third_party/googletest)
        if(NOT EXISTS ${_gtest_root}/CMakeLists.txt)
            message(FATAL_ERROR "${_gtest_root} is empty - run: scripts/bootstrap.sh")
        endif()
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        add_subdirectory(${_gtest_root} ${CMAKE_CURRENT_BINARY_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
        # Plain -I, never SYSTEM: a Homebrew googletest in /usr/local/include would shadow it otherwise
        set_property(TARGET gtest gtest_main gmock gmock_main PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
    endif()

    # Every header compiles on its own, twice (a broken include guard is caught too)
    file(GLOB_RECURSE _mf_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include CONFIGURE_DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/include/snicholls/*.hpp)
    set(_mf_check_sources)
    foreach(_header IN LISTS _mf_headers)
        if(_header MATCHES "/hub/" AND NOT MODEL_FORMATS_WITH_HUB)
            continue()
        endif()
        string(MAKE_C_IDENTIFIER "${_header}" _id)
        set(HEADER "${_header}")
        configure_file(cmake/header_check.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp @ONLY)
        list(APPEND _mf_check_sources ${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp)
    endforeach()
    add_library(model_formats_header_check OBJECT ${_mf_check_sources})
    if(MODEL_FORMATS_WITH_HUB)
        target_link_libraries(model_formats_header_check PRIVATE snicholls::formats_hub)
    else()
        target_link_libraries(model_formats_header_check PRIVATE snicholls::formats_io)
    endif()
    model_formats_compile_options(model_formats_header_check)

    # tooling (third_party/tooling, a build-time submodule pinned to a tag): snicholls_formats.hpp, the single include,
    # is generated in the build tree and compiled on its own, and the CTests check.single_include (formats.hpp is
    # complete) and check.interfaces (every interface has its concept twin), from tooling.conf
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tooling/cmake/SnichollsTooling.cmake)
        message(FATAL_ERROR "Submodule 'third_party/tooling' is not checked out.\nRun: scripts/bootstrap.sh")
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tooling/cmake/SnichollsTooling.cmake)
    snicholls_single_include_check(model_formats_single_include)
    model_formats_compile_options(model_formats_single_include)
    snicholls_tooling_add_checks(SINGLE_INCLUDE INTERFACES)

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME check.version COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_version.py)
    add_test(NAME check.mirror COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_mirror.py)
    set_tests_properties(check.version check.mirror PROPERTIES LABELS boundary)

    add_subdirectory(tests)
    add_subdirectory(demos)
endif()

if(MODEL_FORMATS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(MODEL_FORMATS_BUILD_BENCHMARKS)
    if(NOT TARGET benchmark::benchmark)
        set(_bench_root ${CMAKE_CURRENT_SOURCE_DIR}/third_party/core/third_party/benchmark)
        if(NOT EXISTS ${_bench_root}/CMakeLists.txt)
            message(FATAL_ERROR "${_bench_root} is empty - run: scripts/bootstrap.sh")
        endif()
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
        add_subdirectory(${_bench_root} ${CMAKE_CURRENT_BINARY_DIR}/third_party/benchmark EXCLUDE_FROM_ALL)
    endif()
    add_subdirectory(benchmarks)
endif()

if(MODEL_FORMATS_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
    configure_file(docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Doxygen: every public entity documented, warnings are errors"
        VERBATIM)
    if(MODEL_FORMATS_BUILD_TESTS)
        add_test(NAME docs.build COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(docs.build PROPERTIES LABELS docs)
    endif()
endif()

# snicholls-core diamond rule 3: exactly one version of core in this build. Only the top-level project asserts it; as
# a subdirectory, the consumer's last line does.
if(MODEL_FORMATS_TOP_LEVEL)
    snicholls_core_assert_single_version()
endif()
