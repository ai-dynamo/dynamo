#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The public surfaces of the tier-1 router crates are committed snapshots (SPLIT_V2
# section 10): dynamo-kv-router, the one crate plugins and standalone consumers name, and
# dynamo-router-protocol, the vocabulary every crate names. This script regenerates each
# with cargo-public-api and fails when a committed file differs; --update rewrites them.
# Needs cargo-public-api (cargo install cargo-public-api@0.52.0 --locked) and the nightly
# toolchain it drives rustdoc with.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# crate directory | snapshot | features ('-' for none)
SNAPSHOTS=(
  "lib/kv-router | lib/kv-router/public-api.txt | http,ingest-zmq,replica-sync-zmq,peer-recovery,metrics"
  "lib/router/protocol | lib/router/protocol/public-api.txt | -"
)
update=0
case "${1:-}" in
  "") ;;
  --update) update=1 ;;
  *) echo "usage: $0 [--update]" >&2; exit 2 ;;
esac
if ! command -v cargo-public-api >/dev/null 2>&1; then
  echo "ERROR cargo-public-api is not installed (cargo install cargo-public-api@0.52.0 --locked)" >&2
  exit 2
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
trim() { sed 's/^ *//; s/ *$//' <<<"$1"; }
failures=0
for row in "${SNAPSHOTS[@]}"; do
  IFS='|' read -r dir snapshot features <<<"$row"
  dir="$(trim "$dir")"; snapshot="$(trim "$snapshot")"; features="$(trim "$features")"
  [ -f "$dir/Cargo.toml" ] || { echo "ERROR $dir/Cargo.toml does not exist" >&2; exit 2; }
  args=(--manifest-path "$dir/Cargo.toml" --simplified --omit blanket-impls,auto-trait-impls,auto-derived-impls)
  [ "$features" = "-" ] || args+=(--features "$features")
  out="$tmp/$(basename "$dir").txt"
  cargo public-api "${args[@]}" >"$out"
  [ -s "$out" ] || { echo "ERROR cargo public-api printed nothing for $dir" >&2; exit 2; }
  if [ "$update" -eq 1 ]; then
    cp "$out" "$snapshot"
    echo "router public api: wrote $snapshot ($(wc -l <"$snapshot") items)"
    continue
  fi
  [ -f "$snapshot" ] || { echo "ERROR $snapshot is missing (run $0 --update)" >&2; exit 2; }
  if diff -u "$snapshot" "$out" >"$out.diff"; then
    echo "router public api: $snapshot matches ($(wc -l <"$snapshot") items)"
  else
    echo "FAIL router public api: $snapshot differs from the crate (run $0 --update and review the diff):" >&2
    cat "$out.diff" >&2
    failures=$((failures + 1))
  fi
done
[ "$failures" -eq 0 ]
