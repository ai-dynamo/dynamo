#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copy each crate's license files out of Cargo registry source caches so the
# Rust NOTICES generator can inline upstream license text.
#
# Usage: harvest_cargo_licenses.sh --policy best-effort|strict --out DIR REGISTRY_SRC...
#
# Each REGISTRY_SRC is a Cargo `registry/src` directory; a missing one is
# skipped. Crates are the directories exactly two levels below it
# (`registry/src/<index>/<name>-<version>`). Only the crate's top-level
# LICENSE*, LICENCE*, COPYING*, NOTICE* and UNLICENSE* entries are copied, to
# DIR/<name>-<version>/, the key compliance/generators/rust.py looks up.
#
# Policies:
#   best-effort  Select any existing path. find and cp errors are ignored and
#                the exit status is always 0; the generator falls back to
#                canonical SPDX text for anything missing.
#   strict       Select regular files only. find errors are printed, and the
#                first mkdir or cp failure makes the script exit non-zero.
#
# Run it inside the RUN that mounts the registry cache, or it sees an empty or
# partial registry.

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
