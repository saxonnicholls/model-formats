//
//  version_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/version.hpp>

#include <gtest/gtest.h>

#include <string>

SNICHOLLS_FORMATS_REQUIRE_VERSION(0, 1, 0);

TEST(FormatsVersion, StringIsMajorDotMinorDotPatch)
{
    using v = snicholls::formats::version;
    EXPECT_EQ(std::string(v::string), std::to_string(v::major) + "." + std::to_string(v::minor) + "." + std::to_string(v::patch));
}

TEST(FormatsVersion, CheckOrdersVersions)
{
    static_assert(SNICHOLLS_FORMATS_VERSION_CHECK(0, 1, 0) < SNICHOLLS_FORMATS_VERSION_CHECK(0, 2, 0));
    static_assert(SNICHOLLS_FORMATS_VERSION == SNICHOLLS_FORMATS_VERSION_CHECK(0, 2, 0));
}
