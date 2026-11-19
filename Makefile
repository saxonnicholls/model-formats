# model-formats - the short way to drive CMake and the scripts, with the same gates as the family's CI.
#
#   make bootstrap     the submodules this build needs (scripts/bootstrap.sh)
#   make test          configure, build, run every CTest (unit, demo, boundary, tool)
#   make test-werror   the same with warnings as errors (the gate)
#   make asan          Address + UB sanitizers
#   make docs          Doxygen, warnings are errors; output build-docs/docs/html
#   make check         the static checks: version and mirror
#   make ubuntu        the gate on Ubuntu 24.04 in Docker, GCC 13 and Clang 18 (scripts/test_ubuntu.sh)
#   make crosscheck    our files read by gguf-py, safetensors and numpy in a scratch venv (scripts/crosscheck.sh)
#   make network       the tests that reach huggingface.co (a few MB)
#   make benchmarks    Release benchmarks; rewrites BENCHMARKS.md
#   make clean
#
#   make test CXX=clang++      any target with another compiler

CXX      ?= c++
BUILD    ?= build
CONFIG   ?= Release
CMAKE_FLAGS ?=

.PHONY: all bootstrap test test-werror asan docs check ubuntu crosscheck network benchmarks clean

all: test

bootstrap:
	scripts/bootstrap.sh

check:
	python3 scripts/check_version.py
	python3 scripts/check_mirror.py

$(BUILD)/CMakeCache.txt:
	cmake -S . -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=$(CONFIG) -DCMAKE_CXX_COMPILER=$(CXX) $(CMAKE_FLAGS)

test: check $(BUILD)/CMakeCache.txt
	cmake --build $(BUILD)
	ctest --test-dir $(BUILD) --output-on-failure -j 8

test-werror:
	$(MAKE) test BUILD=build-werror CMAKE_FLAGS="-DMODEL_FORMATS_WERROR=ON -DMODEL_FORMATS_BUILD_BENCHMARKS=ON"

asan:
	$(MAKE) test BUILD=build-asan CONFIG=RelWithDebInfo \
	    CMAKE_FLAGS="-DMODEL_FORMATS_WERROR=ON -DMODEL_FORMATS_SANITIZE=address,undefined"

docs:
	cmake -S . -B build-docs -DMODEL_FORMATS_BUILD_TESTS=OFF -DMODEL_FORMATS_BUILD_TOOLS=OFF -DMODEL_FORMATS_BUILD_DOCS=ON
	cmake --build build-docs --target docs

ubuntu:
	scripts/test_ubuntu.sh

crosscheck:
	scripts/crosscheck.sh

network:
	$(MAKE) test BUILD=build-network CMAKE_FLAGS="-DMODEL_FORMATS_NETWORK_TESTS=ON"
	ctest --test-dir build-network -L network --output-on-failure

benchmarks:
	scripts/run_benchmarks.sh

clean:
	rm -rf build build-*
