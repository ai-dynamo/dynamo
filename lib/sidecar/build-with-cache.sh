#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run under the Dockerfile's locked target-cache mount. Cargo uses input mtimes,
# which can predate artifacts built from another checkout in the shared cache.
# Retain one timestamp for the current source contents so sibling builders reuse
# fresh artifacts and an interrupted build still invalidates its stale ones.
cache_dir="${CARGO_TARGET_DIR:-$PWD/target}"
mkdir -p "$cache_dir"
cache_dir="$(realpath "$cache_dir")"
stamp="$cache_dir/.workspace-inputs"
candidate="$(mktemp "$cache_dir/.workspace-inputs.XXXXXX")"
trap 'rm -f "$candidate"' EXIT

find "$PWD" -path "$cache_dir" -prune -o -type f -print0 \
    | sort -z \
    | xargs -0 -r sha256sum \
    | sha256sum > "$candidate"
if ! cmp -s "$candidate" "$stamp"; then
    \mv -f "$candidate" "$stamp"
fi

# GNU touch preserves the stamp's nanoseconds; truncating to seconds can leave
# changed inputs older than a Cargo fingerprint from the same second.
find "$PWD" -path "$cache_dir" -prune -o -type f \
    -exec touch -r "$stamp" {} +

rm -f "$candidate"
trap - EXIT
exec cargo "$@"
