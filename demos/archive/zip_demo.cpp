//
//  zip_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  A stored zip: written aligned, read back as views, checked with CRC-32.

#include <snicholls/formats/archive/zip.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace snicholls::formats;

int main()
{
    //! [zip]
    const std::string weights(256, 'w'), version = "3";
    const std::vector<zip::member> members{{"archive/data/0", std::as_bytes(std::span(weights.data(), weights.size()))},
                                           {"archive/version", std::as_bytes(std::span(version.data(), version.size()))}};
    const auto bytes = zip::write(members);   // stored, each member 64-byte aligned, as torch.save writes
    const auto a = zip::parse(*bytes);        // views into `bytes`
    //! [zip]
    int failures = !(bytes.is_ok() && a.is_ok() && a->entries.size() == 2);
    for (const auto& e : a->entries) {
        std::printf("%-16s %4zu bytes at %4llu, crc %08x %s\n", std::string(e.name).c_str(), e.data.size(),
                    static_cast<unsigned long long>(e.offset), e.crc, zip::verify(e).is_ok() ? "ok" : "BAD");
        failures += e.offset % 64 != 0 || zip::verify(e).is_err();
    }
    return failures;
}
