# syntax=docker/dockerfile:1.6
#
# model-formats - the Ubuntu build and test image: core's image (GCC 13, Clang 18 with compiler-rt, CMake, Ninja,
# the pinned Doxygen) plus OpenSSL's headers for ts-moveables' TLS and Python 3 for the static checks.

FROM ubuntu:24.04

LABEL org.opencontainers.image.title="model-formats Ubuntu builder" \
      org.opencontainers.image.description="Ubuntu 24.04: GCC 13, Clang 18 with compiler-rt, CMake, Ninja, Doxygen, OpenSSL 3"

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential g++-13 \
      clang-18 libclang-rt-18-dev \
      cmake ninja-build \
      libssl-dev \
      python3-minimal git ca-certificates curl && \
    ln -sf /usr/bin/clang-18 /usr/local/bin/clang && \
    ln -sf /usr/bin/clang++-18 /usr/local/bin/clang++ && \
    rm -rf /var/lib/apt/lists/*

# Doxygen 1.18.0, the release binary pinned by SHA-256, as core's image (x86-64; the apt package on ARM64)
ARG DOXYGEN_VERSION=1.18.0
ARG DOXYGEN_TAG=Release_1_18_0
ARG DOXYGEN_SHA256=14fa81bdc34171edb5f1f02b1d60e74802f0439b77fa44e592565d517d72df90
ARG TARGETARCH
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
      curl -fsSL -o /tmp/doxygen.tgz \
        "https://github.com/doxygen/doxygen/releases/download/${DOXYGEN_TAG}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz" && \
      echo "${DOXYGEN_SHA256}  /tmp/doxygen.tgz" | sha256sum -c - && \
      tar -xzf /tmp/doxygen.tgz -C /tmp && \
      install -m 0755 "/tmp/doxygen-${DOXYGEN_VERSION}/bin/doxygen" /usr/local/bin/doxygen && \
      rm -rf /tmp/doxygen.tgz "/tmp/doxygen-${DOXYGEN_VERSION}"; \
    else \
      apt-get update && apt-get install -y --no-install-recommends doxygen && rm -rf /var/lib/apt/lists/*; \
    fi && \
    doxygen --version

WORKDIR /work
CMD ["bash"]
