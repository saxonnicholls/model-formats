//
//  client.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The Hugging Face Hub client: resolve a repository and revision, download its files into the shared cache,
/// resumably, in parallel, verified, over ts-moveables' HTTP client and TLS.
/// \ingroup hub
///
/// \overview
/// `resolve` asks the public API (`GET <endpoint>/api/models/<repo>/revision/<rev>?blobs=true`) for the commit a
/// revision names and every file in it with its size, git blob id and, for large files, its SHA-256. `download`
/// fetches the files chosen by fnmatch patterns from `<endpoint>/<repo>/resolve/<commit>/<path>`, following the
/// redirects to the CDN, in ranges of `chunk_bytes` with up to `parallel` files at once on one event loop, into
/// `blobs/<etag>.incomplete`. An interrupted download resumes from the last whole range. A finished file is checked
/// (SHA-256 for LFS files, the git blob id for the rest) before it is renamed into `blobs/` and linked from
/// `snapshots/<commit>/`, as `huggingface_hub` lays it out, so the two share one cache.
///
/// - **Auth**: `HF_TOKEN` (or `hub_options::token`), sent as `Authorization: Bearer` to the endpoint's origin only
///   (scheme, host and port), never to a redirect target elsewhere, never logged, never in an error (core's `secret`).
/// - **Offline**: `HF_HUB_OFFLINE=1` (or `offline`) answers from the cache alone: a revision resolved before, files
///   present; anything else is `offline`.
/// - **Errors are respected**: 401 / 403 (gated or private: a token is needed), 404 (no such repository, revision or
///   file), 416 at a resumed end (complete), 429 and 5xx (`http` with the status), at most 10 redirects.
/// - **A cap**: `max_download_bytes` refuses a download whose missing files sum to more, before any byte is fetched.
///
/// \relations
/// \see snicholls::http::client, snicholls::http::openssl_client_context (ts-moveables), formats::hub::cache,
/// formats::hub::digest, formats::io::open_model_dir (opens the snapshot)
///
/// \impls
/// One event loop and one `http::client` (HTTP/1.1 over OpenSSL, certificates and host names verified); each file a
/// chain of range requests in callbacks.
///
/// \contracts
/// A failed download leaves the cache consistent: no blob is ever renamed into place unverified; `.incomplete` files
/// stay for the next attempt. The client is not thread-safe: one call at a time.
///
/// \errors
/// `bad_argument` (an unsafe repository id, revision or path from the server), `http` (status, network, TLS),
/// `bad_json` (an API answer of the wrong shape), `checksum_mismatch`, `too_large` (over the cap), `offline`, `io`.
///
/// \examples
/// \snippet client_demo.cpp client
///
/// \perf
/// Not benchmarked; tiny-random-gpt2 (about 2.5 MB) downloads in about a second on a home connection.
///
/// \testing
/// tests/hub/client_test.cpp (against ts-moveables' own http::server on localhost: listings, redirects, ranges,
/// resume, a corrupted file, 404, 401, the cap, offline); tests/hub/client_network_test.cpp (label `network`, off by
/// default: a small public repository from huggingface.co); demos/hub/client_demo.cpp.
///
/// \deps
/// ts-moveables (`http/client.hpp`, `tls/openssl.hpp`, `event/loop.hpp`, `mmap/`, `fs/`, `process/config.hpp`),
/// OpenSSL 3.
///
/// \gotchas
/// Verification runs on the loop's thread, so a very large file pauses the other downloads while it is hashed. The
/// ts-moveables client holds each response body in memory: `chunk_bytes` bounds it.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_hub_client_hpp
#define snicholls_formats_hub_client_hpp

#include <snicholls/formats/detail/json.hpp>
#include <snicholls/formats/hub/cache.hpp>
#include <snicholls/formats/hub/digest.hpp>
#include <snicholls/formats/hub/pattern.hpp>
#include <snicholls/formats/version.hpp>

#include <snicholls/core/secret.hpp>

#include <event/loop.hpp>
#include <http/client.hpp>
#include <net/url.hpp>
#include <tls/openssl.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::hub
{
/// \brief How the client reaches the Hub.
struct hub_options
{
    std::string endpoint = "https://huggingface.co";            ///< The Hub (`HF_ENDPOINT`).
    std::filesystem::path cache_dir;                            ///< The cache; empty for `default_cache_dir()`.
    std::optional<secret<std::string>> token;                   ///< `HF_TOKEN`: sent to the endpoint's origin only.
    bool offline = false;                                       ///< `HF_HUB_OFFLINE`: the cache alone.
    std::size_t parallel = 4;                                   ///< Files downloaded at once.
    std::size_t chunk_bytes = std::size_t{16} << 20;            ///< The range of one request (a multiple of 64 KiB).
    std::uint64_t max_download_bytes = 0;                       ///< Refuse more than this in one call; 0 for no cap.
    std::chrono::milliseconds timeout{120'000};                 ///< Per request.
    std::string user_agent = std::string("model-formats/") + version::string;   ///< The User-Agent.

    /// \brief Options from the environment: `HF_ENDPOINT`, `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`), `HF_HUB_OFFLINE`
    /// and the cache variables. \param env The configuration (the process environment by default, no .env files).
    /// \return The options.
    [[nodiscard]] static hub_options from_environment(const config& env = load_config({.env_files = {}}))
    {
        hub_options o;
        if (const std::string* e = env.raw("HF_ENDPOINT"); e && !e->empty())
            o.endpoint = *e;
        while (!o.endpoint.empty() && o.endpoint.back() == '/')
            o.endpoint.pop_back();
        const std::string* t = env.raw("HF_TOKEN");
        if (!t || t->empty())
            t = env.raw("HUGGING_FACE_HUB_TOKEN");
        if (t && !t->empty())
            o.token = secret<std::string>(*t);
        if (const std::string* off = env.raw("HF_HUB_OFFLINE"))
            o.offline = *off == "1" || *off == "true" || *off == "TRUE" || *off == "yes" || *off == "ON" || *off == "on";
        o.cache_dir = default_cache_dir(env);
        return o;
    }
};

/// \brief A file of a repository at a commit.
struct repo_file
{
    std::string path;                       ///< Its path in the repository.
    std::uint64_t size = 0;                 ///< Its size.
    std::string blob_id;                    ///< git's blob id (of the LFS pointer, for an LFS file).
    std::optional<std::string> sha256;      ///< An LFS (or Xet) file's SHA-256.

    /// \brief The blob's name in the cache: the SHA-256 if LFS, else the blob id. \return It.
    [[nodiscard]] const std::string& etag() const noexcept { return sha256 ? *sha256 : blob_id; }
};

/// \brief A repository at a revision.
struct repo_info
{
    std::string repo;                 ///< `org/name`.
    std::string revision;             ///< What was asked for ("main").
    std::string commit;               ///< The commit it resolved to.
    std::vector<repo_file> files;     ///< Every file.
};

/// \brief What `download` did.
struct download_report
{
    std::filesystem::path snapshot;          ///< `snapshots/<commit>`: open it with `io::open_model_dir`.
    std::vector<std::string> downloaded;     ///< Files fetched.
    std::vector<std::string> cached;         ///< Files already there.
    std::uint64_t bytes = 0;                 ///< Bytes fetched.
    std::uint64_t resumed = 0;               ///< Bytes not fetched again because a partial download was resumed.
};

namespace detail
{
    // Percent-encoding, as huggingface_hub's quote: unreserved characters and `safe` kept
    inline std::string quote(std::string_view s, std::string_view safe)
    {
        static constexpr char hex[] = "0123456789ABCDEF";
        std::string out;
        for (const char c : s) {
            const auto u = static_cast<unsigned char>(c);
            if ((u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') || c == '-' || c == '.' || c == '_' || c == '~'
                || safe.find(c) != std::string_view::npos) {
                out.push_back(c);
            } else {
                out.push_back('%');
                out.push_back(hex[u >> 4]);
                out.push_back(hex[u & 15]);
            }
        }
        return out;
    }
    inline std::string origin_of(const std::string& url)
    {
        net::url u;
        if (!net::parse_url(url, u))
            return {};
        return u.scheme + "://" + u.authority();
    }
    inline error status_error(int status, const std::string& what)
    {
        std::string why = "HTTP " + std::to_string(status);
        if (status == 401 || status == 403)
            why += ": the repository is private or gated (set HF_TOKEN), or the token lacks access";
        else if (status == 404)
            why += ": not found";
        else if (status == 429)
            why += ": rate limited; try again later";
        return error{status == 404 ? errc::not_found : errc::http, what + ": " + why};
    }
} // namespace detail

/// \brief The Hub client.
class client
{
public:
    /// \brief A client. \param opt How to reach the Hub (`hub_options::from_environment()` for the usual).
    explicit client(hub_options opt)
        : opt_(std::move(opt)),
          tls_(std::make_unique<http::openssl_client_context>(http::openssl_client_config{.insecure_skip_verify = false, .ca_file = {},
                                                                                           .ca_path = {}, .alpn = {"http/1.1"}, .send_sni = true}))
    {
        if (opt_.cache_dir.empty())
            opt_.cache_dir = default_cache_dir();
        if (opt_.chunk_bytes < (std::size_t{1} << 16))
            opt_.chunk_bytes = std::size_t{1} << 16;
        opt_.chunk_bytes -= opt_.chunk_bytes % (std::size_t{1} << 16);
        if (opt_.parallel == 0)
            opt_.parallel = 1;
    }

    /// \brief The options in use. \return They.
    [[nodiscard]] const hub_options& options() const noexcept { return opt_; }

    /// \brief The commit and files of a repository at a revision (online, or from the cache when offline).
    /// \param repo `org/name`. \param revision A branch, tag or commit. \return The listing, or the problem.
    [[nodiscard]] result<repo_info> resolve(std::string_view repo, std::string_view revision = "main")
    {
        if (!valid_repo_id(repo) || !valid_revision(revision))
            return fail(errc::bad_argument, "hub: " + quote_name(repo) + "@" + quote_name(revision) + " is not a repository and revision");
        SNICHOLLS_TRY_ASSIGN(const cache c, cache::open(opt_.cache_dir, repo));
        if (opt_.offline)
            return resolve_offline(c, repo, revision);
        const std::string url = opt_.endpoint + "/api/models/" + std::string(repo) + "/revision/" + detail::quote(revision, "") + "?blobs=true";
        SNICHOLLS_TRY_ASSIGN(const http::response r, get(url, {}));
        if (r.status != 200)
            return err(detail::status_error(r.status, "hub: " + std::string(repo) + "@" + std::string(revision)));
        SNICHOLLS_TRY_ASSIGN(const formats::detail::json j, formats::detail::parse_json(r.body, "hub API answer"));
        repo_info info;
        info.repo = repo;
        info.revision = revision;
        const std::string* sha = formats::detail::string_at(j, "sha");
        const formats::detail::json* siblings = formats::detail::member(j, "siblings");
        if (!sha || !is_commit(*sha) || !siblings || !siblings->is_array())
            return fail(errc::bad_json, "hub: the API answer has no commit sha and siblings");
        info.commit = *sha;
        for (const auto& s : *siblings) {
            repo_file f;
            const std::string* name = formats::detail::string_at(s, "rfilename");
            const std::string* blob = formats::detail::string_at(s, "blobId");
            if (!name || !safetensors::safe_relative_name(*name))
                return fail(errc::bad_argument, "hub: the API lists an unsafe file name");
            f.path = *name;
            f.size = formats::detail::u64_at(s, "size").value_or(0);
            f.blob_id = blob ? *blob : std::string();
            if (const formats::detail::json* lfs = formats::detail::member(s, "lfs"); lfs && lfs->is_object()) {
                if (const std::string* h = formats::detail::string_at(*lfs, "sha256"))
                    f.sha256 = *h;
                f.size = formats::detail::u64_at(*lfs, "size").value_or(f.size);
            }
            if (!safetensors::safe_relative_name(f.etag()) || f.etag().size() < 40)
                return fail(errc::bad_json, "hub: " + quote_name(f.path) + " has no usable blob id");
            info.files.push_back(std::move(f));
        }
        if (revision != info.commit)
            SNICHOLLS_TRY_VOID(c.write_ref(revision, info.commit));
        return ok(std::move(info));
    }

    /// \brief Downloads the files chosen into the cache. \param info From `resolve`.
    /// \param include fnmatch patterns to keep (empty: all). \param exclude Patterns to drop.
    /// \return What was fetched and what was there, or the first problem.
    [[nodiscard]] result<download_report> download(const repo_info& info, std::span<const std::string> include = {},
                                                   std::span<const std::string> exclude = {})
    {
        SNICHOLLS_TRY_ASSIGN(const cache c, cache::open(opt_.cache_dir, info.repo));
        download_report report;
        report.snapshot = c.snapshot(info.commit);
        std::vector<const repo_file*> todo;
        std::uint64_t needed = 0;
        for (const auto& f : info.files) {
            if (!select(f.path, include, exclude))
                continue;
            if (c.has(info.commit, f.path)) {
                report.cached.push_back(f.path);
                continue;
            }
            if (std::filesystem::exists(c.blob(f.etag()))) {   // the same blob under another commit
                SNICHOLLS_TRY_VOID(c.link(info.commit, f.path, f.etag()));
                report.cached.push_back(f.path);
                continue;
            }
            if (opt_.offline)
                return fail(errc::offline, "hub: " + f.path + " is not in the cache");
            todo.push_back(&f);
            needed += f.size;
        }
        if (opt_.max_download_bytes && needed > opt_.max_download_bytes)
            return fail(errc::too_large, "hub: " + std::to_string(needed) + " bytes to download, over the cap of " + std::to_string(opt_.max_download_bytes));
        std::filesystem::create_directories(c.snapshot(info.commit));
        if (todo.empty())
            return ok(std::move(report));
        SNICHOLLS_TRY_VOID(fetch_all(c, info, todo, report));
        return ok(std::move(report));
    }

    /// \brief `resolve` then `download`. \param repo `org/name`. \param revision The revision.
    /// \param include Patterns to keep. \param exclude Patterns to drop. \return The report.
    [[nodiscard]] result<download_report> snapshot_download(std::string_view repo, std::string_view revision = "main",
                                                            std::span<const std::string> include = {}, std::span<const std::string> exclude = {})
    {
        SNICHOLLS_TRY_ASSIGN(const repo_info info, resolve(repo, revision));
        return download(info, include, exclude);
    }

private:
    hub_options opt_;
    std::unique_ptr<http::openssl_client_context> tls_;

    result<repo_info> resolve_offline(const cache& c, std::string_view repo, std::string_view revision) const
    {
        SNICHOLLS_TRY_ASSIGN(const std::string commit, c.resolve_offline(revision));
        repo_info info{std::string(repo), std::string(revision), commit, {}};
        std::error_code ec;
        const auto root = c.snapshot(commit);
        for (auto it = std::filesystem::recursive_directory_iterator(root, ec); !ec && it != std::filesystem::recursive_directory_iterator();
             it.increment(ec)) {
            if (!it->is_symlink() && !it->is_regular_file())
                continue;
            repo_file f;
            f.path = it->path().lexically_relative(root).generic_string();
            const auto target = std::filesystem::read_symlink(it->path(), ec);
            f.blob_id = ec ? std::string() : target.filename().string();
            ec.clear();
            f.size = std::filesystem::file_size(it->path(), ec);
            ec.clear();
            info.files.push_back(std::move(f));
        }
        return ok(std::move(info));
    }

    http::client_config http_config() const
    {
        http::client_config cfg;
        auto* tls = tls_.get();
        cfg.transport_factory = [tls](const std::string& host) -> std::unique_ptr<http::transport_delegate> { return tls->connect(host); };
        cfg.request_timeout = opt_.timeout;
        cfg.max_connections_per_host = opt_.parallel;
        cfg.limits.max_body = opt_.chunk_bytes + (std::size_t{1} << 20);
        cfg.headers = {{"User-Agent", opt_.user_agent}};
        return cfg;
    }

    std::vector<std::pair<std::string, std::string>> headers_for(const std::string& url) const
    {
        std::vector<std::pair<std::string, std::string>> h;
        if (opt_.token && detail::origin_of(url) == detail::origin_of(opt_.endpoint))
            h.emplace_back("Authorization", "Bearer " + opt_.token->reveal());
        return h;
    }

    // One request on its own loop, following redirects
    result<http::response> get(std::string url, std::vector<std::pair<std::string, std::string>> extra)
    {
        for (int hop = 0; hop <= 10; ++hop) {
            auto headers = headers_for(url);
            headers.insert(headers.end(), extra.begin(), extra.end());
            auto r = http::fetch({"GET", url, std::move(headers), {}, opt_.timeout}, http_config());
            if (r.is_err())
                return fail(errc::http, "hub: " + detail::origin_of(url) + ": " + r.unwrap_err_unchecked().what());
            const int s = r->status;
            if (s == 301 || s == 302 || s == 303 || s == 307 || s == 308) {
                const std::string* loc = r->header("location");
                if (!loc)
                    return fail(errc::http, "hub: a redirect without a Location");
                url = loc->starts_with("http://") || loc->starts_with("https://") ? *loc : detail::origin_of(url) + *loc;
                continue;
            }
            return ok(std::move(*r));
        }
        return fail(errc::http, "hub: more than 10 redirects");
    }

    struct job
    {
        const repo_file* file = nullptr;
        std::string url;                 // the resolve URL, then the redirect's target
        std::uint64_t offset = 0;
        int hops = 0;
        mapped_region out;
    };

    result<void> start(const cache& c, const repo_info& info, job& j, download_report& report)
    {
        std::error_code ec;
        std::filesystem::create_directories(c.blob("").parent_path(), ec);
        const auto part = c.incomplete(j.file->etag());
        const auto existing = std::filesystem::exists(part, ec) ? std::filesystem::file_size(part, ec) : 0;
        j.offset = std::min<std::uint64_t>(existing / opt_.chunk_bytes * opt_.chunk_bytes, j.file->size);   // whole ranges only
        report.resumed += j.offset;
        auto r = mapped_region::open(part.string(), {.reserve = std::max<std::size_t>(static_cast<std::size_t>(j.file->size), 1), .create = true,
                                                     .truncate = false, .preallocate = false, .mode = 0644});
        if (r.is_err())
            return err(io::io_error(part, "cannot open", r.unwrap_err_unchecked()));
        j.out = std::move(*r);
        j.url = opt_.endpoint + "/" + info.repo + "/resolve/" + info.commit + "/" + detail::quote(j.file->path, "/");
        return ok();
    }

    result<void> finish(const cache& c, const repo_info& info, job& j)
    {
        const int fd = j.out.fd();
        if (auto s = j.out.flush(); s.is_err())
            return err(io::io_error(c.incomplete(j.file->etag()), "cannot sync", s.unwrap_err_unchecked()));
        if (::ftruncate(fd, static_cast<off_t>(j.file->size)) != 0 || snicholls::full_sync(fd).is_err())
            return err(io::io_error(c.incomplete(j.file->etag()), "cannot finish", std::error_code(errno, std::generic_category())));
        j.out = mapped_region{};
        const auto part = c.incomplete(j.file->etag());
        SNICHOLLS_TRY_ASSIGN(const std::string got, file_digest_hex(part, !j.file->sha256));
        if (got != j.file->etag()) {
            std::error_code ec;
            std::filesystem::remove(part, ec);
            return fail(errc::checksum_mismatch, "hub: " + j.file->path + " has digest " + got + ", the Hub says " + j.file->etag());
        }
        std::error_code ec;
        std::filesystem::rename(part, c.blob(j.file->etag()), ec);
        if (ec)
            return err(io::io_error(c.blob(j.file->etag()), "cannot rename onto", ec));
        return c.link(info.commit, j.file->path, j.file->etag());
    }

    result<void> fetch_all(const cache& c, const repo_info& info, const std::vector<const repo_file*>& files, download_report& report)
    {
        event_loop loop;
        std::optional<error> failure;
        std::deque<job> jobs;
        std::size_t next = 0, active = 0;
        {
            http::client net(loop, http_config());
            std::function<void()> start_more;
            std::function<void(job&)> request;
            const auto done = [&](job& j, std::optional<error> e) {
                if (e && !failure)
                    failure = std::move(e);
                if (!e)
                    report.downloaded.push_back(j.file->path);
                j.out = mapped_region{};
                --active;
                start_more();
                if (active == 0)
                    loop.stop();
            };
            request = [&](job& j) {
                if (j.offset >= j.file->size) {
                    auto r = finish(c, info, j);
                    done(j, r.is_err() ? std::optional<error>(r.unwrap_err_unchecked()) : std::nullopt);
                    return;
                }
                const std::uint64_t last = std::min<std::uint64_t>(j.offset + opt_.chunk_bytes, j.file->size) - 1;
                auto headers = headers_for(j.url);
                headers.emplace_back("Range", "bytes=" + std::to_string(j.offset) + "-" + std::to_string(last));
                net.request({"GET", j.url, std::move(headers), {}, opt_.timeout}, [&, jp = &j](http::client_result r) {
                    job& j = *jp;
                    if (r.is_err())
                        return done(j, error{errc::http, "hub: " + j.file->path + ": " + r.unwrap_err_unchecked().what()});
                    const int s = r->status;
                    if (s == 301 || s == 302 || s == 303 || s == 307 || s == 308) {
                        const std::string* loc = r->header("location");
                        if (!loc || ++j.hops > 10)
                            return done(j, error{errc::http, "hub: " + j.file->path + ": a bad or endless redirect"});
                        j.url = loc->starts_with("http://") || loc->starts_with("https://") ? *loc : detail::origin_of(j.url) + *loc;
                        return request(j);
                    }
                    if (s == 416 && j.offset == j.file->size)
                        return request(j);
                    if (s != 200 && s != 206)
                        return done(j, detail::status_error(s, "hub: " + j.file->path));
                    if (s == 200 && j.offset > 0)   // the server ignored the range: start again from the beginning
                        j.offset = 0;
                    const auto& body = r->body;
                    if (body.empty() || j.offset + body.size() > j.file->size)
                        return done(j, error{errc::http, "hub: " + j.file->path + ": a range of the wrong length"});
                    if (auto g = j.out.grow_or_move(static_cast<std::size_t>(j.offset + body.size())); g.is_err())
                        return done(j, io::io_error(c.incomplete(j.file->etag()), "cannot grow", g.unwrap_err_unchecked()));
                    std::memcpy(j.out.data() + j.offset, body.data(), body.size());
                    (void)j.out.flush_async(static_cast<std::size_t>(j.offset), body.size());
                    j.offset += body.size();
                    report.bytes += body.size();
                    request(j);
                });
            };
            start_more = [&] {
                while (!failure && active < opt_.parallel && next < files.size()) {
                    jobs.emplace_back();
                    job& j = jobs.back();
                    j.file = files[next++];
                    ++active;
                    if (auto s = start(c, info, j, report); s.is_err()) {
                        done(j, s.unwrap_err_unchecked());
                        continue;
                    }
                    loop.post([&, jp = &j] { request(*jp); });
                }
            };
            start_more();
            if (active > 0)
                loop.run();
        }
        if (failure)
            return err(std::move(*failure));
        return ok();
    }
};
} // namespace snicholls::formats::hub

#endif // snicholls_formats_hub_client_hpp
