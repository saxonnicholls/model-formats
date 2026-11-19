//
//  version.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The version of model-formats, as macros and as values.
/// \ingroup formats
///
/// \overview
/// The library's version, written in three places (here, `project(... VERSION)` in CMakeLists.txt, and the release
/// tag), which `scripts/check_version.py` keeps equal. `SNICHOLLS_FORMATS_REQUIRE_VERSION` is the compile-time check a
/// consumer's header makes.
///
/// \relations
/// The counterpart of core's `version.hpp`; `snicholls::formats::version` holds the values.
///
/// \impls
/// Macros for the preprocessor, `constexpr` values for C++.
///
/// \contracts
/// Compile time only; no state.
///
/// \errors
/// `SNICHOLLS_FORMATS_REQUIRE_VERSION` fails to compile when the library is older than asked.
///
/// \examples
/// \snippet version_demo.cpp version
///
/// \perf
/// Not benchmarked: compile time only.
///
/// \testing
/// tests/version_test.cpp, demos/version_demo.cpp.
///
/// \deps
/// None.
///
/// \gotchas
/// None.
///
/// \status
/// Stable.

#ifndef snicholls_formats_version_hpp
#define snicholls_formats_version_hpp

/// \brief The major version.
#define SNICHOLLS_FORMATS_VERSION_MAJOR 0
/// \brief The minor version.
#define SNICHOLLS_FORMATS_VERSION_MINOR 1
/// \brief The patch version.
#define SNICHOLLS_FORMATS_VERSION_PATCH 0

/// \brief A version as one comparable number. \param major Major. \param minor Minor. \param patch Patch.
#define SNICHOLLS_FORMATS_VERSION_CHECK(major, minor, patch) ((major) * 10000 + (minor) * 100 + (patch))

/// \brief This library's version as one comparable number.
#define SNICHOLLS_FORMATS_VERSION                                                                                      \
    SNICHOLLS_FORMATS_VERSION_CHECK(SNICHOLLS_FORMATS_VERSION_MAJOR, SNICHOLLS_FORMATS_VERSION_MINOR,                  \
                                    SNICHOLLS_FORMATS_VERSION_PATCH)

/// \brief Fails to compile unless this library is at least the version given.
/// \param major Major. \param minor Minor. \param patch Patch.
#define SNICHOLLS_FORMATS_REQUIRE_VERSION(major, minor, patch)                                                         \
    static_assert(SNICHOLLS_FORMATS_VERSION >= SNICHOLLS_FORMATS_VERSION_CHECK(major, minor, patch),                   \
                  "snicholls::formats: model-formats is older than required")

namespace snicholls::formats
{
/// \brief The library's version as values.
struct version
{
    static constexpr int major = SNICHOLLS_FORMATS_VERSION_MAJOR;   ///< Major.
    static constexpr int minor = SNICHOLLS_FORMATS_VERSION_MINOR;   ///< Minor.
    static constexpr int patch = SNICHOLLS_FORMATS_VERSION_PATCH;   ///< Patch.
    static constexpr const char* string = "0.1.0";                  ///< "major.minor.patch".
};
} // namespace snicholls::formats

#endif // snicholls_formats_version_hpp
