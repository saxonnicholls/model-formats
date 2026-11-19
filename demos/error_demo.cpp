//
//  error_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  An expected failure is a value: a result<T> carrying an error with a code and a message.

#include <snicholls/formats/error.hpp>

#include <cstdio>
#include <string>

using namespace snicholls::formats;

//! [error]
result<int> parse_rank(int r)
{
    if (r < 0 || r > 8)
        return fail(errc::bad_shape, "rank " + std::to_string(r) + " is not 0 to 8");
    return snicholls::ok(r);
}
//! [error]

int main()
{
    const auto good = parse_rank(3), bad = parse_rank(9);
    std::printf("rank 3: %s; rank 9: %s\n", good.is_ok() ? "ok" : "error", bad.is_err() ? bad.unwrap_err_unchecked().what().c_str() : "ok");
    std::printf("quoted: %s\n", quote_name("model.embed_tokens.weight").c_str());
    return good.is_ok() && *good == 3 && bad.is_err() && bad.unwrap_err_unchecked().code == errc::bad_shape ? 0 : 1;
}
