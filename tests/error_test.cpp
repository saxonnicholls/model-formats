//
//  error_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/error.hpp>

#include <gtest/gtest.h>

#include <string>

using namespace snicholls::formats;

namespace
{
result<int> half(int x)
{
    if (x % 2)
        return fail(errc::bad_argument, "odd: " + std::to_string(x));
    return snicholls::ok(x / 2);
}
} // namespace

TEST(FormatsError, EveryCodeHasAName)
{
    for (int c = 1; c <= static_cast<int>(errc::bad_argument); ++c)
        EXPECT_STRNE(to_string(static_cast<errc>(c)), "unknown");
    static_assert(noexcept(to_string(errc::io)));
}

TEST(FormatsError, FailConvertsToAnyResult)
{
    const auto r = half(3);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err_unchecked().code, errc::bad_argument);
    EXPECT_EQ(r.unwrap_err_unchecked().what(), "bad argument: odd: 3");
    EXPECT_EQ(*half(4), 2);
}

TEST(FormatsError, WhatWithoutMessageIsTheCodeName) { EXPECT_EQ((error{errc::io, ""}).what(), "I/O error"); }

TEST(FormatsError, QuotedCutsLongNames)
{
    EXPECT_EQ(quote_name("abc"), "\"abc\"");
    const std::string longer(100, 'x');
    EXPECT_EQ(quote_name(longer), "\"" + std::string(64, 'x') + "...\"");
}
