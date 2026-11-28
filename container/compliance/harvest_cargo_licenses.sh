#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copy crate license files from Cargo registry/src caches to DIR/<name>-<version>/
# for the Rust NOTICES generator. Run inside the RUN that mounts the registry cache.

set -u

usage() {
    echo "usage: $0 --policy best-effort|strict --out DIR REGISTRY_SRC..." >&2
    exit 2
}

policy=
out=
while [ $# -gt 0 ]; do
    case "$1" in
        --policy) [ $# -ge 2 ] || usage; policy=$2; shift 2 ;;
        --out) [ $# -ge 2 ] || usage; out=$2; shift 2 ;;
        --) shift; break ;;
        -*) usage ;;
        *) break ;;
    esac
done
[ -n "$out" ] || usage

# fd 3 receives find and cp diagnostics: hidden for best-effort, shown for strict.
case "$policy" in
    best-effort) select=-e; exec 3>/dev/null ;;
    strict) select=-f; exec 3>&2 ;;
    *) usage ;;
esac

if ! mkdir -p "$out" && [ "$policy" = strict ]; then
    exit 1
fi

status=0
for root in "$@"; do
    [ -d "$root" ] || continue
    # Without pipefail the pipeline status is the loop's, so a strict failure
    # inside the loop's subshell still reaches `status`.
    find "$root" -mindepth 2 -maxdepth 2 -type d 2>&3 | while IFS= read -r crate; do
        dest="$out/$(basename "$crate")"
        for lf in "$crate"/LICENSE* "$crate"/LICENCE* "$crate"/COPYING* "$crate"/NOTICE* "$crate"/UNLICENSE*; do
            [ "$select" "$lf" ] || continue
            if ! { mkdir -p "$dest" && cp "$lf" "$dest/" 2>&3; }; then
                [ "$policy" = best-effort ] || exit 1
            fi
        done
    done || { status=1; break; }
done

echo "rust license harvest: $(find "$out" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) crates with license files in $out"
exit "$status"
