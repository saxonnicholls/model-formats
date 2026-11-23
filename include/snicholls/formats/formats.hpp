//
//  formats.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The umbrella: every model-formats header that needs nothing but the standard library and core.
/// \ingroup formats
///
/// \overview
/// One include for the readers and writers that need no third party (GGUF, `.npy`, llm.c, PyTorch checkpoints, zip,
/// and the shared vocabulary types), sorted, and the source of the single include `snicholls_formats.hpp` (tooling's
/// `single_include.py`), which is this with every model-formats and snicholls-core header it uses expanded in one
/// file, generated and never committed. The headers that parse JSON (nlohmann/json through core-adapters) or link
/// OpenSSL are left out (`single_include_opt_in` in `tooling.conf`, each with its reason) and are included one by one,
/// through `snicholls::formats_io` and `snicholls::formats_hub`.
///
/// \relations
/// Every header it includes; `tooling.conf`.
///
/// \impls
/// None: an umbrella.
///
/// \contracts
/// Those of the headers it includes.
///
/// \errors
/// None.
///
/// \examples
/// None: an umbrella.
///
/// \perf
/// Not benchmarked: compile time only.
///
/// \testing
/// None of its own: an umbrella, exempt from the mirror (`scripts/check_mirror.py`); every build generates the single
/// include from it and compiles that alone (`model_formats_single_include`).
///
/// \deps
/// The standard library and snicholls-core.
///
/// \gotchas
/// None.
///
/// \status
/// Stable.

#ifndef snicholls_formats_formats_hpp
#define snicholls_formats_formats_hpp

#include <snicholls/formats/archive/zip.hpp>
#include <snicholls/formats/dtype.hpp>
#include <snicholls/formats/error.hpp>
#include <snicholls/formats/gguf/ggml_types.hpp>
#include <snicholls/formats/gguf/gguf.hpp>
#include <snicholls/formats/gguf/keys.hpp>
#include <snicholls/formats/gguf/metadata.hpp>
#include <snicholls/formats/hf/names.hpp>
#include <snicholls/formats/hub/pattern.hpp>
#include <snicholls/formats/llmc/llmc.hpp>
#include <snicholls/formats/model_config.hpp>
#include <snicholls/formats/numpy/npy.hpp>
#include <snicholls/formats/pytorch/checkpoint.hpp>
#include <snicholls/formats/pytorch/pickle.hpp>
#include <snicholls/formats/tensor.hpp>
#include <snicholls/formats/version.hpp>
#include <snicholls/formats/vocabulary.hpp>

#endif /* snicholls_formats_formats_hpp */
