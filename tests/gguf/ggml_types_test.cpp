//
//  ggml_types_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/gguf/ggml_types.hpp>

#include <gtest/gtest.h>

#include <set>

using namespace snicholls::formats;
using namespace snicholls::formats::gguf;

TEST(GgmlTypes, SizesMatchGgmlCommon)
{
    // ggml-common.h's static_asserts, with sizeof(ggml_half) = 2 and QK_K = 256
    static_assert(traits(ggml_type::q4_1).type_size == 2 * 2 + 32 / 2);
    static_assert(traits(ggml_type::q5_0).type_size == 2 + 4 + 32 / 2);
    static_assert(traits(ggml_type::q2_k).type_size == 2 * 2 + 256 / 16 + 256 / 4);
    static_assert(traits(ggml_type::q3_k).type_size == 2 + 256 / 4 + 256 / 8 + 12);
    static_assert(traits(ggml_type::q4_k).type_size == 2 * 2 + 12 + 256 / 2);
    static_assert(traits(ggml_type::q5_k).type_size == 2 * 2 + 12 + 256 / 2 + 256 / 8);
    static_assert(traits(ggml_type::q6_k).type_size == 2 + 256 / 16 + 3 * 256 / 4);
    static_assert(traits(ggml_type::q8_k).type_size == 4 + 256 + 256 / 16 * 2);
    static_assert(traits(ggml_type::iq2_xxs).type_size == 2 + 256 / 8 * 2);
    static_assert(traits(ggml_type::iq3_s).type_size == 2 + 13 * (256 / 32) + 256 / 64);
    static_assert(traits(ggml_type::iq1_m).type_size == 256 / 8 + 256 / 16 + 256 / 32);
    static_assert(traits(ggml_type::iq4_xs).type_size == 2 + 2 + 256 / 64 + 256 / 2);
    static_assert(traits(ggml_type::tq1_0).type_size == 2 + 256 / 64 + (256 - 4 * 256 / 64) / 5);
    static_assert(traits(ggml_type::nvfp4).type_size == 64 / 16 + 64 / 2);
    static_assert(traits(ggml_type::q1_0).type_size == 2 + 128 / 8 && traits(ggml_type::q2_0).type_size == 2 + 64 / 4);
}

TEST(GgmlTypes, CoreLayoutsAreCoresStructs)
{
    static_assert(traits(ggml_type::q8_0).type_size == sizeof(snicholls::ml::block_q8_0));
    static_assert(traits(ggml_type::q4_0).in_core && traits(ggml_type::mxfp4).in_core && !traits(ggml_type::q4_k).in_core);
}

TEST(GgmlTypes, CodesAndNamesAreUnique)
{
    std::set<std::uint32_t> codes;
    std::set<std::string_view> names;
    for (const auto& t : ggml_types) {
        EXPECT_TRUE(codes.insert(static_cast<std::uint32_t>(t.type)).second);
        EXPECT_TRUE(names.insert(t.name).second);
        EXPECT_EQ(ggml_type_from_name(t.name), t.type);
        EXPECT_EQ(traits(static_cast<std::uint32_t>(t.type))->name, t.name);
    }
    for (const std::uint32_t removed : {4u, 5u, 31u, 32u, 33u, 36u, 37u, 38u, 44u})
        EXPECT_FALSE(traits(removed).has_value());
    EXPECT_FALSE(ggml_type_from_name("q9_9").has_value());
}

TEST(GgmlTypes, RowBytesNeedWholeBlocks)
{
    static_assert(row_bytes(ggml_type::q8_0, 64) == 68u);
    static_assert(!row_bytes(ggml_type::q8_0, 33).has_value());
    static_assert(row_bytes(ggml_type::f32, 3) == 12u);
    static_assert(row_bytes(ggml_type::q4_k, 512) == 288u);
    static_assert(row_bytes(ggml_type::q8_0, 0) == 0u);
}

TEST(GgmlTypes, DenseTypesMapToDtypes)
{
    for (const auto& t : ggml_types) {
        const auto d = to_dtype(t.type);
        if (d) {
            EXPECT_EQ(t.block_size, 1u);
            EXPECT_EQ(size_of(*d), t.type_size);
            EXPECT_EQ(from_dtype(*d), t.type);
        }
    }
    static_assert(!from_dtype(dtype::u32).has_value() && !from_dtype(dtype::f8_e5m2).has_value() && !to_dtype(ggml_type::q8_0).has_value());
}
