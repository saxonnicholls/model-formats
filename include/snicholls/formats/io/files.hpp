//
//  files.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Model files on disk: memory-mapped reading of any format, and writing that a crash cannot leave half done.
/// \ingroup io
///
/// \overview
/// `mapped_file::open` maps a file with ts-moveables' `mapped_region`, so the readers' views are views of the page
/// cache: opening a 15 GB checkpoint maps it and parses the header, and a tensor's pages are read when it is first
/// touched. `open_model` maps a file and reads it with `convert::read_any`. `write_file` replaces a file atomically
/// with ts-moveables' `write_file_atomically` (durable: synced before the rename); `write_streamed` does the same for
/// output too large to build in memory, writing the pieces into a mapped temporary beside the target and renaming it
/// over the target when complete. `save_tensors` writes tensors in the format the path's extension names.
///
/// \relations
/// \see snicholls::mapped_region, snicholls::write_file_atomically (ts-moveables), formats::convert::read_any
///
/// \impls
/// A mapping, or the whole file read into memory when the file cannot be mapped (an empty file, a pipe, a file opened
/// read-only: ts-moveables' region maps for writing too).
///
/// \contracts
/// A `mapped_file` owns its mapping; moving it keeps every view valid (the address does not change). Writing never
/// leaves a partial file at the target path: the temporary is removed on failure.
///
/// \errors
/// `io` with the path and the operating system's message; the format readers' and writers' errors.
///
/// \examples
/// \snippet files_demo.cpp files
///
/// \perf
/// Not benchmarked: the cost is the operating system's.
///
/// \testing
/// tests/io/files_test.cpp, demos/io/files_demo.cpp.
///
/// \deps
/// ts-moveables (`mmap/mapped_region.hpp`, `fs/file.hpp`); POSIX.
///
/// \gotchas
/// ts-moveables maps files read-write: a file the process may not write is read into memory instead (proposal:
/// a read-only mapping in ts-moveables). `write_streamed`'s temporary-and-rename belongs in ts-moveables as a streaming
/// `write_file_atomically`; it moves there when that exists (proposal).
///
/// \status
/// Experimental.

#ifndef snicholls_formats_io_files_hpp
#define snicholls_formats_io_files_hpp

#include <snicholls/formats/convert/convert.hpp>

#include <fs/file.hpp>
#include <mmap/mapped_region.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

namespace snicholls::formats::io
{
/// \brief An I/O error for a path. \param path The path. \param what What was done. \param ec The error.
/// \return The error.
[[nodiscard]] inline error io_error(const std::filesystem::path& path, std::string_view what, std::error_code ec)
{
    return error{errc::io, std::string(what) + " " + path.string() + ": " + ec.message()};
}

/// \brief A file's bytes, mapped (or read, when it cannot be mapped).
class mapped_file
{
public:
    /// \brief Maps a file. \param path The file. \return The mapping, or `io`.
    [[nodiscard]] static result<mapped_file> open(const std::filesystem::path& path)
    {
        mapped_file f;
        f.path_ = path;
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec)
            return err(io_error(path, "cannot open", ec));
        if (size > 0 && ::access(path.c_str(), W_OK) == 0) {
            auto r = mapped_region::open(path.string(), {.reserve = static_cast<std::size_t>(size), .create = false, .truncate = false,
                                                         .preallocate = false, .mode = 0644});
            if (r.is_ok()) {
                f.region_ = std::move(*r);
                (void)f.region_.advise(map_advice::willneed, 0, std::min<std::size_t>(f.region_.size(), std::size_t{1} << 20));
                return ok(std::move(f));
            }
        }
        auto bytes = read_file_bytes(path);
        if (bytes.is_err())
            return err(io_error(path, "cannot read", bytes.unwrap_err_unchecked()));
        f.buffer_ = std::move(*bytes);
        return ok(std::move(f));
    }

    /// \brief The bytes. \return A view, valid while this lives.
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    {
        if (region_.mapped())
            return {region_.data(), region_.size()};
        return std::as_bytes(std::span(buffer_));
    }
    /// \brief Whether the bytes are mapped (not read). \return True if mapped.
    [[nodiscard]] bool mapped() const noexcept { return region_.mapped(); }
    /// \brief The path. \return It.
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    mapped_region region_;
    std::vector<std::uint8_t> buffer_;
};

/// \brief Replaces a file with bytes, atomically and durably. \param path The file. \param bytes The contents.
/// \param durable Sync before and after the rename. \return Nothing, or `io`.
[[nodiscard]] inline result<void> write_file(const std::filesystem::path& path, std::span<const std::byte> bytes, bool durable = true)
{
    const auto r = write_file_atomically(path, std::span(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
                                         {.durable = durable, .mode = {}, .create_directories = true});
    if (r.is_err())
        return err(io_error(path, "cannot write", r.unwrap_err_unchecked()));
    return ok();
}

/// \brief Writes a file piece by piece and renames it over `path` when complete.
/// \tparam Produce Callable with a sink (callable with `std::span<const std::byte>`), returning `result<void>`.
/// \param path The file. \param produce Writes the contents through the sink. \return Nothing, or `io` or
/// `produce`'s error (the target is then untouched).
template <typename Produce>
[[nodiscard]] result<void> write_streamed(const std::filesystem::path& path, Produce&& produce)
{
    std::error_code ec;
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path(), ec);
    const std::filesystem::path tmp = path.string() + ".partial-" + std::to_string(::getpid());
    auto region = mapped_region::open(tmp.string(), {.create = true, .truncate = true});
    if (region.is_err())
        return err(io_error(tmp, "cannot create", region.unwrap_err_unchecked()));
    std::size_t at = 0;
    std::error_code failed;
    const auto sink = [&](std::span<const std::byte> s) {
        if (failed || s.empty())
            return;
        if (auto g = region->grow_or_move(at + s.size()); g.is_err()) {
            failed = g.unwrap_err_unchecked();
            return;
        }
        std::memcpy(region->data() + at, s.data(), s.size());
        at += s.size();
    };
    const auto cleanup = [&] {
        *region = mapped_region{};
        std::filesystem::remove(tmp, ec);
    };
    if (auto r = produce(sink); r.is_err()) {
        cleanup();
        return r;
    }
    if (!failed)
        if (auto s = region->full_sync(); s.is_err())
            failed = s.unwrap_err_unchecked();
    const int fd = region->fd();
    if (!failed && ::ftruncate(fd, static_cast<off_t>(at)) != 0)   // the region rounds the file up to a page
        failed = std::error_code(errno, std::generic_category());
    if (!failed)
        if (auto s = snicholls::full_sync(fd); s.is_err())
            failed = s.unwrap_err_unchecked();
    if (failed) {
        cleanup();
        return err(io_error(tmp, "cannot write", failed));
    }
    *region = mapped_region{};
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return err(io_error(path, "cannot rename onto", ec));
    }
    return ok();
}

/// \brief A file mapped and read.
struct opened_model
{
    mapped_file file;          ///< The mapping the tensors view.
    convert::loaded model;     ///< The tensors and metadata.
};

/// \brief Maps a file and reads its tensors, whatever its format. \param path The file. \param opt The reader's
/// options. \return The model, or `io` / the reader's errors.
[[nodiscard]] inline result<opened_model> open_model(const std::filesystem::path& path, const convert::read_options& opt = {})
{
    SNICHOLLS_TRY_ASSIGN(mapped_file f, mapped_file::open(path));
    auto m = convert::read_any(f.bytes(), opt);
    if (m.is_err())
        return err(error{m.unwrap_err_unchecked().code, path.string() + ": " + m.unwrap_err_unchecked().message});
    return ok(opened_model{std::move(f), std::move(*m)});
}

/// \brief Writes tensors in the format the path's extension names (`.safetensors`, `.gguf`, `.npz`, `.npy`).
/// \param path The file. \param tensors The tensors. \param metadata String metadata.
/// \return Nothing, or `bad_argument` (no such extension), the writer's errors, or `io`.
[[nodiscard]] inline result<void> save_tensors(const std::filesystem::path& path, std::span<const tensor_view> tensors,
                                               std::span<const std::pair<std::string, std::string>> metadata = {})
{
    const auto f = convert::format_from_extension(path.string());
    if (!f)
        return fail(errc::bad_argument, "save_tensors: " + path.string() + " has no extension naming a format");
    if (*f == convert::file_format::safetensors)
        return write_streamed(path, [&](auto&& sink) { return safetensors::write_to(tensors, metadata, sink); });
    SNICHOLLS_TRY_ASSIGN(const std::vector<std::byte> bytes, convert::write_any(*f, tensors, metadata));
    return write_file(path, bytes);
}
} // namespace snicholls::formats::io

#endif // snicholls_formats_io_files_hpp
