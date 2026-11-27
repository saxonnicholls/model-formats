//
//  version_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The library's version, as values and as the compile-time check a consumer makes.

#include <snicholls/formats/version.hpp>

#include <cstdio>
#include <cstring>

int main()
{
    //! [version]
    SNICHOLLS_FORMATS_REQUIRE_VERSION(0, 1, 0);   // fails to compile on an older model-formats
    std::printf("model-formats %s\n", snicholls::formats::version::string);
    //! [version]
    return std::strcmp(snicholls::formats::version::string, "0.2.0") == 0 ? 0 : 1;
}
