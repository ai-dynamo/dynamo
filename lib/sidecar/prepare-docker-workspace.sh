#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

# Cargo loads every workspace manifest before selecting a package. Create empty
# targets for workspace members whose implementation is intentionally absent
# from the backend-specific Docker build context.
find . -mindepth 2 -name Cargo.toml -print | while IFS= read -r manifest; do
    crate_dir="$(dirname "$manifest")"
    mkdir -p "$crate_dir/src"
    touch "$crate_dir/src/lib.rs"

    awk '
        function value(line) {
            sub(/^[^=]*=[[:space:]]*"/, "", line)
            sub(/".*$/, "", line)
            return line
        }
        function emit() {
            if (kind == "" || name == "") {
                return
            }
            if (path != "") {
                print path
            } else if (kind == "bin") {
                print "src/bin/" name ".rs"
            } else if (kind == "bench") {
                print "benches/" name ".rs"
            } else if (kind == "example") {
                print "examples/" name ".rs"
            } else if (kind == "test") {
                print "tests/" name ".rs"
            }
        }
        /^\[\[(bin|bench|example|test)\]\]$/ {
            emit()
            kind = $0
            gsub(/^\[\[|\]\]$/, "", kind)
            name = ""
            path = ""
            next
        }
        /^\[/ {
            emit()
            kind = ""
            name = ""
            path = ""
            next
        }
        kind != "" && /^[[:space:]]*name[[:space:]]*=/ {
            name = value($0)
            next
        }
        kind != "" && /^[[:space:]]*path[[:space:]]*=/ {
            path = value($0)
            next
        }
        END {
            emit()
        }
    ' "$manifest" | while IFS= read -r target; do
        mkdir -p "$(dirname "$crate_dir/$target")"
        touch "$crate_dir/$target"
    done

    build_script="$(sed -n 's/^[[:space:]]*build[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$manifest" | head -n 1)"
    if [ -n "$build_script" ]; then
        mkdir -p "$(dirname "$crate_dir/$build_script")"
        touch "$crate_dir/$build_script"
    fi
done
