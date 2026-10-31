#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The public surface of dynamo-kv-router, the one crate plugins and standalone consumers
# name, is a committed snapshot (SPLIT_V2 section 10). This script regenerates it with
# cargo-public-api and fails when the committed file differs; --update rewrites the file.
# Needs cargo-public-api (cargo install cargo-public-api --locked) and the nightly toolchain
# it asks for.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SNAPSHOT="lib/kv-router/public-api.txt"
FEATURES="http,ingest-zmq,replica-sync-zmq,peer-recovery,metrics"
update=0
case "${1:-}" in
  "") ;;
  --update) update=1 ;;
  *) echo "usage: $0 [--update]" >&2; exit 2 ;;
esac
if ! command -v cargo-public-api >/dev/null 2>&1; then
  echo "ERROR cargo-public-api is not installed (cargo install cargo-public-api --locked)" >&2
  exit 2
fi
tmp="$(mktemp)"
trap 'mrm -f "$tmp"' EXIT
cargo public-api --manifest-path lib/kv-router/Cargo.toml --features "$FEATURES" --simplified --omit blanket-impls,auto-trait-impls,auto-derived-impls >"$tmp"
[ -s "$tmp" ] || { echo "ERROR cargo public-api printed nothing" >&2; exit 2; }
if [ "$update" -eq 1 ]; then
  cp "$tmp" "$SNAPSHOT"
  echo "router public api: wrote $SNAPSHOT ($(wc -l <"$SNAPSHOT") items)"
  exit 0
fi
if diff -u "$SNAPSHOT" "$tmp" >"$tmp.diff"; then
  echo "router public api: $SNAPSHOT matches ($(wc -l <"$SNAPSHOT") items)"
else
  echo "FAIL router public api: $SNAPSHOT differs from the crate (run $0 --update and review the diff):" >&2
  cat "$tmp.diff" >&2
  mrm -f "$tmp.diff"
  exit 1
fi
mrm -f "$tmp.diff"
