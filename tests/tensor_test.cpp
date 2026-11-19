//
//  tensor_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/tensor.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

using namespace snicholls::formats;

TEST(FormatsTensorShape, ElementsAndBytes)
{
    constexpr tensor_shape s{2, 3, 4};
    static_assert(s.rank == 3 && s.elements() == 24 && checked_byte_size(s, dtype::bf16) == 48);
    static_assert(tensor_shape{}.elements() == 1);   // a scalar
    static_assert(tensor_shape{0, 5}.elements() == 0);
    EXPECT_EQ(to_string(s), "[2, 3, 4]");
    EXPECT_EQ(to_string(tensor_shape{}), "[]");
}

TEST(FormatsTensorShape, OverflowIsNone)
{
    const tensor_shape s{std::numeric_limits<std::uint64_t>::max(), 2};
    EXPECT_FALSE(s.elements().has_value());
    const tensor_shape t{std::numeric_limits<std::uint64_t>::max() / 2};
    EXPECT_FALSE(checked_byte_size(t, dtype::f32).has_value());
}

TEST(FormatsTensorShape, RankIsBounded)
{
    tensor_shape s;
    for (std::size_t i = 0; i < max_rank; ++i)
        EXPECT_TRUE(s.push_back(1));
    EXPECT_FALSE(s.push_back(1));
    EXPECT_EQ(s.rank, max_rank);
}

TEST(FormatsTensorShape, ReversedAndEquality)
{
    constexpr tensor_shape s{2, 3, 4};
    static_assert(s.reversed() == tensor_shape{4, 3, 2});
    static_assert(s.reversed().reversed() == s);
    static_assert(!(tensor_shape{2, 3} == tensor_shape{2, 3, 1}));
    EXPECT_EQ(s[1], 3u);
    EXPECT_EQ(s.view().size(), 3u);
}

TEST(FormatsTensor, OwnedViewsAndCopies)
{
    tensor t{"w", dtype::u8, {2}, {std::byte{1}, std::byte{2}}};
    const tensor_view v = t.view();
    EXPECT_EQ(v.name, "w");
    EXPECT_EQ(v.data.data(), t.data.data());   // a view, not a copy
    const tensor c = tensor::from(v);
    EXPECT_NE(c.data.data(), t.data.data());
    EXPECT_EQ(c.data, t.data);
    const std::vector<tensor> ts{t, c};
    const auto vs = views(ts);
    ASSERT_EQ(vs.size(), 2u);
    EXPECT_EQ(vs[1].data.data(), ts[1].data.data());
}
