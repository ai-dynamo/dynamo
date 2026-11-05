#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the jemalloc shared library that ships inside the ai-dynamo-runtime wheel
# (dynamo/libjemalloc_dynamo.so.2). Dynamo's Python entrypoints preload it unless
# DYN_DISABLE_JEMALLOC is set; see lib/bindings/python/src/dynamo/_jemalloc.py.
#
# Usage: build.sh VERSION SHA256 OUT_DIR [ARCHIVE_DIR]
#   OUT_DIR receives libjemalloc_dynamo.so.2 and COPYING.
#   ARCHIVE_DIR, if given, receives the source tarball (compliance source archival).
set -euo pipefail

version=$1
sha256=$2
out_dir=$3
archive_dir=${4:-}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tarball="jemalloc-${version}.tar.bz2"
cd "$work"
for attempt in 1 2 3 4 5; do
    curl --retry 3 --retry-delay 5 --retry-connrefused --connect-timeout 30 -fsSL -o "$tarball" \
        "https://github.com/jemalloc/jemalloc/releases/download/${version}/${tarball}" && break
    echo "jemalloc download attempt ${attempt}/5 failed; retrying in 10s" >&2
    sleep 10
done
echo "${sha256}  ${tarball}" | sha256sum -c -
tar xjf "$tarball"
cd "jemalloc-${version}"

configure_args=(
    --with-install-suffix=_dynamo
    --disable-static
    # No C++ operator new/delete, so no libstdc++ dependency: preloading the library must not
    # change which libstdc++ the process loads first. C++ allocations still reach jemalloc
    # through malloc.
    --disable-cxx
    # Built-in options for a long-running server; MALLOC_CONF overrides them per option.
    # background_thread purges freed pages on a schedule instead of only when the owning thread
    # allocates again; narenas caps arenas instead of four per CPU.
    --with-malloc-conf=background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16
)
# jemalloc fixes its page size at build time and fails to start on a kernel with larger pages,
# so one aarch64 build must use 64 KiB to run on 4, 16 and 64 KiB page kernels.
if [ "$(uname -m)" = "aarch64" ]; then
    configure_args+=(--with-lg-page=16)
fi
./configure "${configure_args[@]}"
make -j"$(nproc)" build_lib_shared

# jemalloc builds with -g3; keep the symbol table (perf profiles) but drop the debug info.
strip --strip-debug lib/libjemalloc_dynamo.so.2

mkdir -p "$out_dir"
cp lib/libjemalloc_dynamo.so.2 COPYING "$out_dir/"
if [ -n "$archive_dir" ]; then
    mkdir -p "$archive_dir"
    cp "$work/$tarball" "$archive_dir/"
fi
