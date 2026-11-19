# ts-moveables errata (v1.9.0)

## tls/openssl.hpp: SSL_set1_host is deprecated in OpenSSL 4

- Evidence: `tls/openssl.hpp:493` calls `SSL_set1_host`; with OpenSSL 4.1 headers (Homebrew's `openssl@4`, which also
  sits in `/usr/local/include`) `-Werror` fails: "'SSL_set1_host' is deprecated: Since OpenSSL 4.0".
- Impact: any consumer building with OpenSSL 4 and warnings as errors.
- Proposed fix: `SSL_set1_dnsname` / `SSL_add1_host` under `#if OPENSSL_VERSION_MAJOR >= 4`.
- Workaround here: `snicholls::formats_hub` defines `OPENSSL_API_COMPAT=0x30000000L` (CMakeLists.txt).

## mmap/mapped_region.hpp: files are mapped read-write only

- Evidence: `mapped_region::open` opens `O_RDWR` and maps `PROT_READ | PROT_WRITE`, `MAP_SHARED`.
- Impact: a file the process may read but not write (a read-only cache, a file of another user) cannot be mapped.
- Proposed fix: a read-only open (`O_RDONLY`, `PROT_READ`).
- Workaround here: `io::mapped_file` reads such a file into memory with `read_file_bytes`.

## fs/file.hpp: no streaming atomic write

- Evidence: `write_file_atomically` takes the whole contents; a 10 GB checkpoint would have to be built in memory.
- Proposed fix: a writer object (temporary beside the target, appends, `commit()` syncs and renames).
- Workaround here: `io::write_streamed` (a mapped temporary, renamed when complete), to be replaced by it.

## http/client.hpp: response bodies are held whole

- Evidence: "Not here yet: request and response bodies streamed in chunks" (the header's notes).
- Impact: downloads must be split into ranges so no body is larger than memory allows.
- Workaround here: the Hub client fetches in `chunk_bytes` ranges (16 MiB by default), which also gives resume.
