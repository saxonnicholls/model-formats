//
//  digest.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief SHA-256 and SHA-1, for verifying what the Hub sends: an LFS file's SHA-256, a small file's git blob SHA-1.
/// \ingroup hub
///
/// \overview
/// The Hub names every file by a digest: a large (LFS or Xet) file by the SHA-256 of its contents, a small one by its
/// git blob id, SHA-1 of `"blob <size>\0"` and the contents. Both are checked before a download is moved into the
/// cache. `digest` hashes incrementally; `sha256_hex` and `git_blob_sha1_hex` hash a buffer; `file_digest_hex` hashes
/// a file through its mapping.
///
/// The family's SHA-2 belongs to core (its approved, not yet written, `hash/sha256.hpp`), and base-encode-decode has
/// none, so until core's lands this uses OpenSSL's, which the Hub client links for TLS anyway: a vetted
/// implementation, not a new one. The calls switch to core's when it exists (proposal in
/// family/proposals/model-formats.md).
///
/// \relations
/// \see formats::hub::client
///
/// \impls
/// OpenSSL's EVP interface.
///
/// \contracts
/// `digest` is not copyable; `hex()` finishes it (further `update`s are ignored).
///
/// \errors
/// `file_digest_hex`: `io`.
///
/// \examples
/// \snippet digest_demo.cpp digest
///
/// \perf
/// OpenSSL's speed (SHA-NI / ARMv8 SHA where present).
///
/// \testing
/// tests/hub/digest_test.cpp (FIPS 180 vectors, git's empty blob), demos/hub/digest_demo.cpp.
///
/// \deps
/// OpenSSL 3 (libcrypto).
///
/// \gotchas
/// None.
///
/// \status
/// Experimental; to be replaced by core's SHA-2.

#ifndef snicholls_formats_hub_digest_hpp
#define snicholls_formats_hub_digest_hpp

#include <snicholls/formats/io/files.hpp>

#include <openssl/evp.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace snicholls::formats::hub
{
/// \brief Which digest.
enum class algorithm : std::uint8_t
{
    sha256,   ///< SHA-256.
    sha1,     ///< SHA-1 (git's blob ids only).
};

/// \brief An incremental digest.
class digest
{
public:
    /// \brief Starts a digest. \param a The algorithm.
    explicit digest(algorithm a) : ctx_(EVP_MD_CTX_new())
    {
        EVP_DigestInit_ex(ctx_.get(), a == algorithm::sha256 ? EVP_sha256() : EVP_sha1(), nullptr);
    }
    /// \brief Adds bytes. \param b The bytes.
    void update(std::span<const std::byte> b)
    {
        if (!done_)
            EVP_DigestUpdate(ctx_.get(), b.data(), b.size());
    }
    /// \brief Adds text. \param s The text.
    void update(std::string_view s) { update(std::as_bytes(std::span(s.data(), s.size()))); }
    /// \brief Finishes. \return The digest in lower-case hex.
    [[nodiscard]] std::string hex()
    {
        if (!done_) {
            unsigned int n = 0;
            EVP_DigestFinal_ex(ctx_.get(), out_, &n);
            size_ = n;
            done_ = true;
        }
        static constexpr char digits[] = "0123456789abcdef";
        std::string h;
        for (unsigned i = 0; i < size_; ++i) {
            h.push_back(digits[out_[i] >> 4]);
            h.push_back(digits[out_[i] & 15]);
        }
        return h;
    }

private:
    struct free_ctx
    {
        void operator()(EVP_MD_CTX* c) const noexcept { EVP_MD_CTX_free(c); }
    };
    std::unique_ptr<EVP_MD_CTX, free_ctx> ctx_;
    unsigned char out_[EVP_MAX_MD_SIZE]{};
    unsigned size_ = 0;
    bool done_ = false;
};

/// \brief SHA-256 of bytes. \param b The bytes. \return Lower-case hex.
[[nodiscard]] inline std::string sha256_hex(std::span<const std::byte> b)
{
    digest d(algorithm::sha256);
    d.update(b);
    return d.hex();
}

/// \brief git's blob id of bytes: SHA-1 of "blob <size>\0" and the bytes. \param b The bytes. \return Lower-case hex.
[[nodiscard]] inline std::string git_blob_sha1_hex(std::span<const std::byte> b)
{
    digest d(algorithm::sha1);
    const std::string head = "blob " + std::to_string(b.size());
    d.update(std::as_bytes(std::span(head.data(), head.size() + 1)));   // with its NUL
    d.update(b);
    return d.hex();
}

/// \brief A file's SHA-256, or its git blob id. \param path The file. \param git_blob True for the blob id.
/// \return Lower-case hex, or `io`.
[[nodiscard]] inline result<std::string> file_digest_hex(const std::filesystem::path& path, bool git_blob = false)
{
    SNICHOLLS_TRY_ASSIGN(const io::mapped_file f, io::mapped_file::open(path));
    return ok(git_blob ? git_blob_sha1_hex(f.bytes()) : sha256_hex(f.bytes()));
}
} // namespace snicholls::formats::hub

#endif // snicholls_formats_hub_digest_hpp
