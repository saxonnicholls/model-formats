//
//  mf_download.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  mf-download: a Hugging Face repository into the shared cache.
//
//      mf-download <repo>[@revision] [--include PATTERN]... [--exclude PATTERN]... [--cache DIR] [--offline]
//                  [--max-bytes N] [--parallel N]
//
//  Prints the snapshot directory (for mf-inspect or mf-convert). HF_TOKEN, HF_ENDPOINT, HF_HUB_OFFLINE and HF_HOME /
//  HF_HUB_CACHE are read as huggingface_hub reads them; the token is never printed. Exit status 0, or 1 with the error.

#include <snicholls/formats/hub/client.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace snicholls::formats;

int main(int argc, char** argv)
{
    hub::hub_options opt = hub::hub_options::from_environment();
    std::vector<std::string> include, exclude;
    std::string target;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--include" && more)
            include.push_back(argv[++i]);
        else if (a == "--exclude" && more)
            exclude.push_back(argv[++i]);
        else if (a == "--cache" && more)
            opt.cache_dir = argv[++i];
        else if (a == "--offline")
            opt.offline = true;
        else if (a == "--max-bytes" && more)
            opt.max_download_bytes = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--parallel" && more)
            opt.parallel = std::strtoul(argv[++i], nullptr, 10);
        else if (target.empty() && !a.starts_with("--"))
            target = a;
        else {
            std::fprintf(stderr, "mf-download: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (target.empty()) {
        std::fprintf(stderr, "usage: mf-download <repo>[@revision] [--include PATTERN]... [--exclude PATTERN]... [--cache DIR] [--offline] "
                             "[--max-bytes N] [--parallel N]\n");
        return 2;
    }
    const auto at = target.find('@');
    const std::string repo = target.substr(0, at), revision = at == std::string::npos ? "main" : target.substr(at + 1);
    hub::client client(opt);
    const auto info = client.resolve(repo, revision);
    if (info.is_err()) {
        std::fprintf(stderr, "mf-download: %s\n", info.unwrap_err_unchecked().what().c_str());
        return 1;
    }
    const auto r = client.download(*info, include, exclude);
    if (r.is_err()) {
        std::fprintf(stderr, "mf-download: %s\n", r.unwrap_err_unchecked().what().c_str());
        return 1;
    }
    std::fprintf(stderr, "%s@%s (%s): %zu downloaded (%llu bytes, %llu resumed), %zu cached\n", repo.c_str(), revision.c_str(),
                 info->commit.c_str(), r->downloaded.size(), static_cast<unsigned long long>(r->bytes),
                 static_cast<unsigned long long>(r->resumed), r->cached.size());
    std::printf("%s\n", r->snapshot.string().c_str());
    return 0;
}
