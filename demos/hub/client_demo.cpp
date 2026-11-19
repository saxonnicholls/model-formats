//
//  client_demo.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  The Hub client against a small Hub run here on localhost (so the demo needs no network): resolve, download into
//  a cache, open the snapshot.

#include <snicholls/formats/hub/client.hpp>
#include <snicholls/formats/io/model_dir.hpp>

#include <http/server.hpp>

#include <cstdio>
#include <thread>

using namespace snicholls::formats;

int main()
{
    // The "Hub": the API listing and one file, config.json
    const std::string config = R"({"model_type":"gpt2","n_embd":4,"n_head":2,"n_layer":1,"n_positions":8,"vocab_size":3})";
    const auto bytes = std::as_bytes(std::span(config.data(), config.size()));
    const std::string commit = "0123456789abcdef0123456789abcdef01234567";
    snicholls::http::server srv;
    srv.get("/*", [&](const auto& req, auto res) {
        if (req.path.rfind("/api/models/", 0) == 0)
            return res.send(200, "application/json", "{\"sha\":\"" + commit + "\",\"siblings\":[{\"rfilename\":\"config.json\",\"size\":"
                                                         + std::to_string(config.size()) + ",\"blobId\":\"" + hub::git_blob_sha1_hex(bytes) + "\"}]}");
        res.send(200, "application/json", config);
    });
    const auto port = srv.listen("127.0.0.1", 0);
    std::thread th([&] { srv.run(); });
    while (!srv.running())
        std::this_thread::yield();
    const auto cache = std::filesystem::temp_directory_path() / "model-formats-client-demo";
    std::filesystem::remove_all(cache);

    //! [client]
    hub::hub_options opt = hub::hub_options::from_environment();   // HF_TOKEN, HF_HUB_OFFLINE, the cache
    opt.endpoint = "http://127.0.0.1:" + std::to_string(port);     // the real one is https://huggingface.co
    opt.cache_dir = cache;
    opt.offline = false;
    hub::client hub(opt);
    const auto info = hub.resolve("org/tiny-model", "main");       // the commit and every file, sized and hashed
    const auto got = hub.download(*info);                          // verified, then linked into snapshots/<commit>
    //! [client]
    if (got.is_ok())
        std::printf("commit %s: %zu downloaded, %zu cached, into %s\n", info->commit.c_str(), got->downloaded.size(), got->cached.size(),
                    got->snapshot.string().c_str());
    else
        std::printf("%s\n", got.unwrap_err_unchecked().what().c_str());
    const bool good = info.is_ok() && got.is_ok() && got->downloaded.size() == 1
                      && hf::parse_config(*snicholls::read_file(got->snapshot / "config.json")).is_ok();
    srv.stop();
    th.join();
    std::filesystem::remove_all(cache);
    return good ? 0 : 1;
}
