//
//  pickle_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The restricted unpickler: a state dict is read, os.system is refused.

#include <snicholls/formats/pytorch/pickle.hpp>

#include <cstdio>
#include <string_view>
#include <vector>

using namespace snicholls::formats;
using namespace snicholls::formats::pytorch;

namespace
{
std::vector<std::byte> bytes(std::string_view s)
{
    const auto b = std::as_bytes(std::span(s.data(), s.size()));
    return {b.begin(), b.end()};
}
} // namespace

int main()
{
    //! [pickle]
    // pickle.dumps({"lr": 3}, protocol=2), then the classic attack: os.system("echo pwned")
    using namespace std::string_view_literals;
    const auto safe = unpickle(bytes("\x80\x02}q\x00X\x02\x00\x00\x00lrq\x01K\x03s."sv));
    const auto attack = unpickle(bytes("\x80\x02" "cos\nsystem\n" "X\x0a\x00\x00\x00" "echo pwned" "\x85R."));
    //! [pickle]
    std::printf("state dict: %s, %zu objects\n", safe.is_ok() ? "read" : safe.unwrap_err_unchecked().what().c_str(),
                safe.is_ok() ? safe->objects.size() : 0);
    std::printf("os.system: %s\n", attack.is_err() ? attack.unwrap_err_unchecked().what().c_str() : "EXECUTED?");
    return safe.is_ok() && safe->at(safe->root).kind == kind::dict && attack.is_err() && attack.unwrap_err_unchecked().code == errc::unsafe_pickle
               ? 0
               : 1;
}
