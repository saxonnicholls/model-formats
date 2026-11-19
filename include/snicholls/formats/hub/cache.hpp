//
//  cache.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The Hugging Face cache, laid out as `huggingface_hub` lays it out, so either can use the other's downloads.
/// \ingroup hub
///
/// \overview
/// ```
/// <cache>/models--<org>--<name>/
///     blobs/<etag>                      the file, named by its SHA-256 (LFS) or git blob id
///     blobs/<etag>.incomplete           a download in progress (resumed from where it stopped)
///     snapshots/<commit>/<path>         a relative symlink to ../../blobs/<etag> (deeper for deeper paths)
///     refs/<revision>                   the commit a branch or tag resolved to
/// ```
/// `default_cache_dir` follows `huggingface_hub`: `HF_HUB_CACHE`, else `HF_HOME/hub`, else
/// `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`, read through ts-moveables' configuration (never
/// `getenv` here). `cache` computes the paths, records refs, links snapshots, and resolves a revision offline.
///
/// \relations
/// \see formats::hub::client
///
/// \impls
/// Paths and small files; nothing is held open.
///
/// \contracts
/// Repository ids are checked (`org/name` or `name`, characters `[A-Za-z0-9._-]`, no `..`) and file paths with
/// `safe_relative_name`, so nothing from the server can name a path outside the cache.
///
/// \errors
/// `bad_argument` (an unsafe repository id, revision or path), `io`, `offline` (a revision never resolved).
///
/// \examples
/// \snippet cache_demo.cpp cache
///
/// \perf
/// Not benchmarked: a few system calls.
///
/// \testing
/// tests/hub/cache_test.cpp, demos/hub/cache_demo.cpp.
///
/// \deps
/// ts-moveables (`process/config.hpp`, `fs/file.hpp`).
///
/// \gotchas
/// Symlinks need a filesystem that has them; on one that does not (some network shares) linking fails with `io`.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_hub_cache_hpp
#define snicholls_formats_hub_cache_hpp

#include <snicholls/formats/io/files.hpp>
#include <snicholls/formats/safetensors/index.hpp>

#include <process/config.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace snicholls::formats::hub
{
/// \brief Whether a repository id is safe: `name` or `org/name`, of `[A-Za-z0-9._-]`, no part `.` or `..`.
/// \param repo The id. \return True if safe.
[[nodiscard]] constexpr bool valid_repo_id(std::string_view repo) noexcept
{
    if (repo.empty() || repo.size() > 200)
        return false;
    int slashes = 0;
    for (const char c : repo) {
        if (c == '/')
            ++slashes;
        else if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return false;
    }
    return slashes <= 1 && safetensors::safe_relative_name(repo) && !repo.starts_with("-");
}

/// \brief Whether a revision is safe to name a ref file: a branch, tag or commit, with `/` inside allowed
/// (`refs/pr/1`), nothing that leaves the directory. \param rev The revision. \return True if safe.
[[nodiscard]] constexpr bool valid_revision(std::string_view rev) noexcept
{
    if (rev.empty() || rev.size() > 256)
        return false;
    for (const char c : rev)
        if (static_cast<unsigned char>(c) < 0x21 || c == '\\' || c == '%' || c == '?' || c == '#')
            return false;
    return safetensors::safe_relative_name(rev);
}

/// \brief Whether a revision is a full commit id (40 hex digits). \param rev The revision. \return True if so.
[[nodiscard]] constexpr bool is_commit(std::string_view rev) noexcept
{
    if (rev.size() != 40)
        return false;
    for (const char c : rev)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}

/// \brief The cache directory `huggingface_hub` would use. \param env The configuration (by default the process
/// environment, no .env files). \return The directory.
[[nodiscard]] inline std::filesystem::path default_cache_dir(const config& env = load_config({.env_files = {}}))
{
    if (const std::string* c = env.raw("HF_HUB_CACHE"); c && !c->empty())
        return *c;
    if (const std::string* h = env.raw("HF_HOME"); h && !h->empty())
        return std::filesystem::path(*h) / "hub";
    if (const std::string* x = env.raw("XDG_CACHE_HOME"); x && !x->empty())
        return std::filesystem::path(*x) / "huggingface" / "hub";
    const std::string* home = env.raw("HOME");
    return std::filesystem::path(home ? *home : ".") / ".cache" / "huggingface" / "hub";
}

/// \brief One repository's corner of the cache.
class cache
{
public:
    /// \brief The cache of a repository. \param root The cache directory. \param repo `org/name`.
    /// \return The cache, or `bad_argument` for an unsafe id.
    [[nodiscard]] static result<cache> open(const std::filesystem::path& root, std::string_view repo)
    {
        if (!valid_repo_id(repo))
            return fail(errc::bad_argument, "hub: " + quote_name(repo) + " is not a repository id");
        cache c;
        std::string folder = "models--";
        for (const char ch : repo)
            folder += ch == '/' ? std::string("--") : std::string(1, ch);
        c.dir_ = root / folder;
        return ok(std::move(c));
    }

    /// \brief The repository's directory. \return `<cache>/models--org--name`.
    [[nodiscard]] const std::filesystem::path& dir() const noexcept { return dir_; }
    /// \brief A blob. \param etag Its digest. \return `blobs/<etag>`.
    [[nodiscard]] std::filesystem::path blob(std::string_view etag) const { return dir_ / "blobs" / std::string(etag); }
    /// \brief A download in progress. \param etag Its digest. \return `blobs/<etag>.incomplete`.
    [[nodiscard]] std::filesystem::path incomplete(std::string_view etag) const { return dir_ / "blobs" / (std::string(etag) + ".incomplete"); }
    /// \brief A snapshot's directory. \param commit The commit. \return `snapshots/<commit>`.
    [[nodiscard]] std::filesystem::path snapshot(std::string_view commit) const { return dir_ / "snapshots" / std::string(commit); }

    /// \brief Records the commit a revision resolved to. \param revision The revision. \param commit The commit.
    /// \return Nothing, or `bad_argument` / `io`.
    [[nodiscard]] result<void> write_ref(std::string_view revision, std::string_view commit) const
    {
        if (!valid_revision(revision) || !is_commit(commit))
            return fail(errc::bad_argument, "hub: an unsafe revision or commit");
        if (revision == commit)
            return ok();
        return io::write_file(dir_ / "refs" / std::string(revision), std::as_bytes(std::span(commit.data(), commit.size())), false);
    }
    /// \brief The commit of a revision, from the cache alone. \param revision A branch, tag or commit.
    /// \return The commit, or `offline` if it was never resolved.
    [[nodiscard]] result<std::string> resolve_offline(std::string_view revision) const
    {
        if (!valid_revision(revision))
            return fail(errc::bad_argument, "hub: " + quote_name(revision) + " is not a revision");
        if (is_commit(revision)) {
            if (std::filesystem::is_directory(snapshot(revision)))
                return ok(std::string(revision));
            return fail(errc::offline, "hub: commit " + std::string(revision) + " is not in the cache");
        }
        auto r = read_file(dir_ / "refs" / std::string(revision));
        if (r.is_err() || !is_commit(*r))
            return fail(errc::offline, "hub: revision " + quote_name(revision) + " was never resolved into this cache");
        return ok(std::move(*r));
    }
    /// \brief Links a snapshot's file to its blob, relatively, as huggingface_hub does.
    /// \param commit The commit. \param path The file's path in the repository. \param etag Its blob.
    /// \return Nothing, or `bad_argument` / `io`.
    [[nodiscard]] result<void> link(std::string_view commit, std::string_view path, std::string_view etag) const
    {
        if (!is_commit(commit) || !safetensors::safe_relative_name(path) || !safetensors::safe_relative_name(etag))
            return fail(errc::bad_argument, "hub: an unsafe commit, path or blob name");
        const auto target = snapshot(commit) / std::string(path);
        std::error_code ec;
        std::filesystem::create_directories(target.parent_path(), ec);
        std::string up = "../../";
        for (const char c : path)
            if (c == '/')
                up += "../";
        std::filesystem::remove(target, ec);
        std::filesystem::create_symlink(std::filesystem::path(up + "blobs/" + std::string(etag)), target, ec);
        if (ec)
            return err(io::io_error(target, "cannot link", ec));
        return ok();
    }
    /// \brief Whether a snapshot has a file whose blob is present. \param commit The commit. \param path The file.
    /// \return True if so.
    [[nodiscard]] bool has(std::string_view commit, std::string_view path) const
    {
        std::error_code ec;
        return std::filesystem::exists(snapshot(commit) / std::string(path), ec);   // follows the link to the blob
    }

private:
    std::filesystem::path dir_;
};
} // namespace snicholls::formats::hub

#endif // snicholls_formats_hub_cache_hpp
