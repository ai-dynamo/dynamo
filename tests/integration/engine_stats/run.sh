#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: $0 [--stargate-root PATH] [--artifact-dir PATH] [--allow-dirty]" \
        "" \
        "Builds the NVCF stats consumer and runs the local engine-stats E2E harness."
}

external_lock_packages() {
    awk '
        function emit() {
            if (name != "" && source != "") {
                print name "\t" version "\t" source
            }
        }
        /^\[\[package\]\]$/ {
            emit()
            name = version = source = ""
            next
        }
        /^name = / {
            name = $0
            sub(/^name = /, "", name)
            next
        }
        /^version = / {
            version = $0
            sub(/^version = /, "", version)
            next
        }
        /^source = / {
            source = $0
            sub(/^source = /, "", source)
            next
        }
        END {
            emit()
        }
    ' "$1" | sort -u
}

verify_generated_lock() {
    local source_lock="$1"
    local generated_lock="$2"
    local label="$3"
    local unexpected
    unexpected="$(
        comm -13 \
            <(external_lock_packages "$source_lock") \
            <(external_lock_packages "$generated_lock")
    )"
    if [[ -n "$unexpected" ]]; then
        printf '%s\n%s\n' \
            "$label resolved dependencies outside its source lock:" \
            "$unexpected" >&2
        return 1
    fi
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dynamo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
stargate_root="$dynamo_root/../nvcf/src/libraries/rust/stargate"
artifact_dir=""
allow_dirty=false

while (($#)); do
    case "$1" in
        --stargate-root)
            stargate_root="${2:?missing value for --stargate-root}"
            shift 2
            ;;
        --artifact-dir)
            artifact_dir="${2:?missing value for --artifact-dir}"
            shift 2
            ;;
        --allow-dirty)
            allow_dirty=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

stargate_root="$(cd -- "$stargate_root" && pwd)"
nvcf_root="$(git -C "$stargate_root" rev-parse --show-toplevel)"

if [[ -z "$artifact_dir" ]]; then
    run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    artifact_dir="$dynamo_root/target/engine-stats-e2e/runs/$run_id"
fi
mkdir -p "$artifact_dir"
artifact_dir="$(cd -- "$artifact_dir" && pwd)"

dynamo_status="$(git -C "$dynamo_root" status --porcelain)"
nvcf_status="$(git -C "$nvcf_root" status --porcelain)"
if ! $allow_dirty && [[ -n "$dynamo_status$nvcf_status" ]]; then
    printf '%s\n' \
        "source worktrees must be clean; pass --allow-dirty for an exploratory run" \
        "Dynamo status:" \
        "$dynamo_status" \
        "NVCF status:" \
        "$nvcf_status" >&2
    exit 2
fi

dynamo_sha="$(git -C "$dynamo_root" rev-parse HEAD)"
stargate_sha="$(git -C "$nvcf_root" rev-parse HEAD)"

git -C "$dynamo_root" diff --binary > "$artifact_dir/dynamo.diff"
git -C "$nvcf_root" diff --binary > "$artifact_dir/nvcf.diff"
rustc --version > "$artifact_dir/rustc-version.txt"
cargo --version > "$artifact_dir/cargo-version.txt"
printf '%s\n' "$dynamo_sha" > "$artifact_dir/dynamo-sha.txt"
printf '%s\n' "$stargate_sha" > "$artifact_dir/stargate-sha.txt"

build_root="$dynamo_root/target/engine-stats-e2e/build"
consumer_target="$build_root/stats-consumer-target"
probe_root="$build_root/stargate-probe"
probe_target="$build_root/stargate-probe-target"
mkdir -p "$probe_root"

probe_manifest="$probe_root/Cargo.toml"
probe_source="$script_dir/src/stargate_probe.rs"
sed \
    -e "s|@STARGATE_ROOT@|$stargate_root|g" \
    -e "s|@PROBE_SOURCE@|$probe_source|g" \
    "$script_dir/StargateProbe.Cargo.toml.in" > "$probe_manifest"

printf 'Building NVCF stats consumer from %s\n' "$stargate_root"
cargo build \
    --locked \
    --manifest-path "$stargate_root/Cargo.toml" \
    --target-dir "$consumer_target" \
    -p pylon \
    -p mock-dynamo

printf 'Building Stargate state probe against NVCF\n'
cp "$stargate_root/Cargo.lock" "$probe_root/Cargo.lock"
cargo metadata \
    --manifest-path "$probe_manifest" \
    --format-version 1 \
    > /dev/null
verify_generated_lock \
    "$stargate_root/Cargo.lock" \
    "$probe_root/Cargo.lock" \
    "Stargate state probe"
cp "$probe_root/Cargo.lock" "$artifact_dir/stargate-probe-Cargo.lock"
cargo build \
    --locked \
    --manifest-path "$probe_manifest" \
    --target-dir "$probe_target"

consumer_bin="$consumer_target/debug/pylon"
mock_dynamo_bin="$consumer_target/debug/mock-dynamo"
stargate_probe_bin="$probe_target/debug/stargate-state-probe"
printf 'Running E2E harness; artifacts: %s\n' "$artifact_dir"
set +e
cargo run \
    --locked \
    --manifest-path "$dynamo_root/Cargo.toml" \
    -p engine-stats-e2e \
    -- \
    --stats-consumer-bin "$consumer_bin" \
    --mock-dynamo-bin "$mock_dynamo_bin" \
    --stargate-probe-bin "$stargate_probe_bin" \
    --artifact-dir "$artifact_dir" \
    2>&1 | tee "$artifact_dir/harness.log"
status=${PIPESTATUS[0]}
set -e

if ((status != 0)); then
    printf 'E2E failed; artifacts retained at %s\n' "$artifact_dir" >&2
    exit "$status"
fi

printf 'E2E passed; artifacts: %s\n' "$artifact_dir"
