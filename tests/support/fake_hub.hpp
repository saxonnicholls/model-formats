//
//  fake_hub.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Test support: a Hugging Face Hub on localhost, over ts-moveables' own http::server. Two servers: the "Hub" (the
//  API and resolve URLs) and a "CDN" on another port that the Hub redirects large files to, so a test can check the
//  token never leaves the Hub's origin. Ranges, 404, a private repository and a corrupted file are served.

#ifndef model_formats_tests_support_fake_hub_hpp
#define model_formats_tests_support_fake_hub_hpp

#include <snicholls/formats/hub/digest.hpp>

#include <http/server.hpp>

#include <atomic>
#include <map>
#include <string>
#include <thread>

namespace test_support
{
struct live_server
{
    snicholls::http::server srv;
    std::uint16_t port = 0;
    std::thread th;
    void start()
    {
        port = srv.listen("127.0.0.1", 0);
        th = std::thread([this] { srv.run(); });
        while (!srv.running())
            std::this_thread::yield();
    }
    ~live_server()
    {
        srv.stop();
        if (th.joinable())
            th.join();
    }
    std::string origin() const { return "http://127.0.0.1:" + std::to_string(port); }
};

struct fake_file
{
    std::string content;
    bool lfs = false;
    bool corrupt = false;   // serve one byte changed
};

struct fake_hub
{
    static constexpr const char* commit = "abcdef0123456789abcdef0123456789abcdef01";
    std::map<std::string, fake_file> files;
    std::atomic<int> requests{0}, cdn_requests{0}, auth_leaks{0};
    std::atomic<std::uint64_t> bytes_served{0};
    std::string token = "hf_secret";
    live_server hub, cdn;

    static void serve_range(const snicholls::http::request& req, snicholls::http::responder res, const std::string& body, std::atomic<std::uint64_t>& served)
    {
        const std::string range = req.header_or("range");
        std::size_t a = 0, b = body.empty() ? 0 : body.size() - 1;
        if (range.rfind("bytes=", 0) == 0) {
            const auto dash = range.find('-');
            a = std::stoull(range.substr(6, dash - 6));
            if (dash + 1 < range.size())
                b = std::min<std::size_t>(std::stoull(range.substr(dash + 1)), body.size() - 1);
            if (a >= body.size()) {
                res.send_status(416);
                return;
            }
            snicholls::http::response r(206);
            r.set("Content-Range", "bytes " + std::to_string(a) + "-" + std::to_string(b) + "/" + std::to_string(body.size()));
            r.content(body.substr(a, b - a + 1), "application/octet-stream");
            served += b - a + 1;
            res.send(std::move(r));
            return;
        }
        served += body.size();
        res.send(200, "application/octet-stream", body);
    }

    std::string listing(const std::string& repo) const
    {
        std::string siblings;
        for (const auto& [path, f] : files) {
            const auto bytes = std::as_bytes(std::span(f.content.data(), f.content.size()));
            if (!siblings.empty())
                siblings += ",";
            siblings += "{\"rfilename\":\"" + path + "\",\"size\":" + std::to_string(f.content.size()) + ",\"blobId\":\""
                        + snicholls::formats::hub::git_blob_sha1_hex(bytes) + "\"";
            if (f.lfs)
                siblings += ",\"lfs\":{\"size\":" + std::to_string(f.content.size()) + ",\"sha256\":\"" + snicholls::formats::hub::sha256_hex(bytes)
                            + "\",\"pointerSize\":130}";
            siblings += "}";
        }
        return "{\"id\":\"" + repo + "\",\"sha\":\"" + commit + "\",\"siblings\":[" + siblings + "]}";
    }

    void start()
    {
        cdn.srv.get("/*", [this](const auto& req, auto res) {
            ++cdn_requests;
            if (req.header("authorization"))
                ++auth_leaks;
            const std::string name = req.path.substr(5);   // "/cdn/<path>"
            const auto it = files.find(name);
            if (it == files.end())
                return res.send_status(404);
            std::string body = it->second.content;
            if (it->second.corrupt && !body.empty())
                body[0] ^= 1;
            serve_range(req, std::move(res), body, bytes_served);
        });
        cdn.start();
        hub.srv.get("/*", [this](const auto& req, auto res) {
            ++requests;
            const std::string& p = req.path;
            const bool private_repo = p.find("org/private") != std::string::npos;
            if (private_repo && req.header_or("authorization") != "Bearer " + token)
                return res.send_status(401);
            if (p.rfind("/api/models/", 0) == 0) {
                const auto rev = p.find("/revision/");
                const std::string repo = p.substr(12, rev - 12);
                if (repo != "org/tiny" && repo != "org/private")
                    return res.send_status(404);
                return res.send(200, "application/json", listing(repo));
            }
            // /<org>/<name>/resolve/<commit>/<path>
            const auto at = p.find("/resolve/");
            if (at == std::string::npos)
                return res.send_status(404);
            const std::string path = p.substr(at + 9 + 41);
            const auto it = files.find(path);
            if (it == files.end())
                return res.send_status(404);
            if (it->second.lfs) {
                snicholls::http::response r(302);
                r.set("Location", cdn.origin() + "/cdn/" + path);
                return res.send(std::move(r));
            }
            serve_range(req, std::move(res), it->second.content, bytes_served);
        });
        hub.start();
    }
};
} // namespace test_support

#endif // model_formats_tests_support_fake_hub_hpp
